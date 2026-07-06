#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Unit tests for the cost-budget profile-guided chunk core.

These cover the pure-Python pieces (latency-model fit, chunk selection, table
round-trip, config validation) with no NPU or model dependency.
"""

import tempfile

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import ProfilingChunkConfig
from vllm_ascend.core.profiling_chunk_predictor import (
    ChunkLatencyTable,
    LatencyModel,
    ProfilingChunkManager,
    SmoothChunkSelector,
)

BIG = 10**9


def build_table(e, b, a_cross, a_intra, histories, chunks, max_model_len=BIG):
    """Synthesize a lookup table from a known T(h, c)."""
    latencies = []
    for h in histories:
        row = []
        for c in chunks:
            if h + c > max_model_len:
                row.append(None)
            else:
                row.append(e + b * c + a_cross * c * h + a_intra * c * c)
        latencies.append(row)
    return ChunkLatencyTable(
        history_sizes=list(histories),
        chunk_sizes=list(chunks),
        latencies_ms=latencies,
    )


class TestLatencyModel(TestBase):
    CHUNKS = [1024, 2048, 4096, 8192]
    HISTORIES = [0, 8192, 32768, 131072]

    def test_fit_recovers_coefficients(self):
        table = build_table(2.0, 0.01, 1e-6, 5e-7, self.HISTORIES, self.CHUNKS)
        model = LatencyModel.fit(table)
        self.assertAlmostEqual(model.e, 2.0, delta=0.5)
        self.assertAlmostEqual(model.b, 0.01, delta=2e-3)
        self.assertAlmostEqual(model.a_cross, 1e-6, delta=2e-7)
        self.assertLess(model.max_rel_residual, 0.01)

    def test_negative_shape_coeffs_clamped(self):
        # A flat table can fit tiny negative slopes from noise; they must clamp
        # to 0 so T stays monotone in history and chunk size.
        table = build_table(5.0, 0.0, 0.0, 0.0, self.HISTORIES, self.CHUNKS)
        model = LatencyModel.fit(table)
        self.assertGreaterEqual(model.b, 0.0)
        self.assertGreaterEqual(model.a_cross, 0.0)
        self.assertGreaterEqual(model.a_intra, 0.0)

    def test_predict_monotone_in_history_and_chunk(self):
        model = LatencyModel(e=1.0, b=0.02, a_cross=2e-6, a_intra=1e-7)
        self.assertLess(model.predict(0, 1024), model.predict(100000, 1024))
        self.assertLess(model.predict(1000, 1024), model.predict(1000, 4096))


class TestSmoothChunkSelector(TestBase):
    def _selector(self):
        model = LatencyModel(e=1.0, b=0.02, a_cross=2e-6, a_intra=0.0)
        return SmoothChunkSelector(model, [1024, 2048, 4096, 8192])

    def test_chunk_non_increasing_with_history(self):
        sel = self._selector()
        budget = sel.model.predict(0, 8192)  # T0 = cost of c_max at h=0
        prev = BIG
        for history in [0, 10_000, 50_000, 200_000, 1_000_000]:
            chunk = sel.select_chunk(history, BIG, BIG, budget)
            self.assertLessEqual(chunk, prev)
            prev = chunk

    def test_rides_floor_at_high_history(self):
        sel = self._selector()
        budget = sel.model.predict(0, 8192)
        # Floor(h) far exceeds the budget -> only c_min fits.
        self.assertEqual(sel.select_chunk(BIG, BIG, BIG, budget), sel.c_min)

    def test_c_min_always_feasible_even_over_budget(self):
        sel = self._selector()
        # A spent budget (0) still yields a valid chunk (floor guarantees progress).
        self.assertEqual(sel.select_chunk(500_000, BIG, BIG, 0.0), sel.c_min)

    def test_tail_returns_exact_remainder(self):
        sel = self._selector()
        self.assertEqual(sel.select_chunk(0, 500, BIG, BIG), 500)

    def test_respects_token_budget(self):
        sel = self._selector()
        # Cost budget is generous; the token cap of 3000 bounds it to 2048.
        self.assertEqual(sel.select_chunk(0, BIG, 3000, BIG), 2048)

    def test_larger_budget_gives_at_least_as_large_chunks(self):
        sel = self._selector()
        history = 50_000
        small = sel.select_chunk(history, BIG, BIG, sel.model.predict(history, 1024))
        large = sel.select_chunk(history, BIG, BIG, sel.model.predict(history, 8192))
        self.assertGreaterEqual(large, small)

    def test_zero_remaining_returns_none(self):
        sel = self._selector()
        self.assertIsNone(sel.select_chunk(0, 0, BIG, BIG))


class TestChunkLatencyTable(TestBase):
    def test_json_round_trip(self):
        table = build_table(1.0, 0.01, 1e-6, 0.0, [0, 4096], [1024, 2048])
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            path = handle.name
        table.to_json_file(path)
        loaded = ChunkLatencyTable.from_json_file(path)
        self.assertEqual(loaded.history_sizes, table.history_sizes)
        self.assertEqual(loaded.chunk_sizes, table.chunk_sizes)
        self.assertEqual(loaded.latencies_ms, table.latencies_ms)

    def test_latency_or_none_for_missing_chunk(self):
        table = build_table(1.0, 0.01, 1e-6, 0.0, [0], [1024, 2048])
        self.assertIsNone(table.latency_or_none(0, 9999))
        self.assertIsNotNone(table.latency_or_none(0, 1024))


class TestProfilingChunkConfig(TestBase):
    def test_defaults(self):
        cfg = ProfilingChunkConfig({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.min_chunk, 1024)
        self.assertIsNone(cfg.target_latency_ms)

    def test_target_latency_ms_parsed(self):
        cfg = ProfilingChunkConfig({"target_latency_ms": 12.5})
        self.assertEqual(cfg.target_latency_ms, 12.5)

    def test_non_positive_target_latency_rejected(self):
        with self.assertRaises(ValueError):
            ProfilingChunkConfig({"target_latency_ms": 0})

    def test_legacy_knobs_ignored_not_rejected(self):
        # search_depth / beam_width are accepted (back-compat) but inert.
        cfg = ProfilingChunkConfig({"search_depth": 1, "beam_width": 1})
        self.assertIsInstance(cfg, ProfilingChunkConfig)

    def test_backfill_reserve_ms_parsed(self):
        cfg = ProfilingChunkConfig({"backfill_reserve_ms": 5.0})
        self.assertEqual(cfg.backfill_reserve_ms, 5.0)

    def test_negative_backfill_reserve_rejected(self):
        with self.assertRaises(ValueError):
            ProfilingChunkConfig({"backfill_reserve_ms": -1})

    def test_protect_inflight_prefill_default_true(self):
        self.assertTrue(ProfilingChunkConfig({}).protect_inflight_prefill)


def _ready_manager(target=None, backfill=None):
    mgr = ProfilingChunkManager(
        max_model_len=BIG,
        page_size=128,
        min_chunk=1024,
        max_chunk=8192,
        allowed_chunk_sizes=[1024, 2048, 4096, 8192],
        profile_num_history_points=4,
        profile_repeats=1,
        target_latency_ms=target,
        backfill_reserve_ms=backfill,
        profile_file=None,
        metadata={},
    )
    mgr._set_table(
        build_table(1.0, 0.02, 2e-6, 0.0, [0, 8192, 32768, 131072], [1024, 2048, 4096, 8192])
    )
    return mgr


class TestBackfillReserve(TestBase):
    def test_auto_backfill_is_fraction_of_target(self):
        mgr = _ready_manager(backfill=None)
        self.assertAlmostEqual(
            mgr.backfill_reserve_ms,
            ProfilingChunkManager.AUTO_BACKFILL_FRACTION * mgr.target_latency_ms,
            places=6,
        )

    def test_explicit_backfill_reserve(self):
        self.assertEqual(_ready_manager(backfill=7.5).backfill_reserve_ms, 7.5)

    def test_backfill_none_before_ready(self):
        mgr = ProfilingChunkManager(
            max_model_len=BIG, page_size=128, min_chunk=1024, max_chunk=8192,
            allowed_chunk_sizes=[1024, 2048, 4096, 8192], profile_num_history_points=4,
            profile_repeats=1, target_latency_ms=None, backfill_reserve_ms=None,
            profile_file=None, metadata={},
        )
        self.assertIsNone(mgr.backfill_reserve_ms)
        self.assertFalse(mgr.is_ready)
