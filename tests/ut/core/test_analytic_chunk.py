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
"""Unit tests for the analytic (profiling-free) PP chunk scheduler core."""

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import AnalyticChunkConfig
from vllm_ascend.core.analytic_chunk_model import (
    AnalyticChunkManager,
    AnalyticLatencyModel,
    SmoothChunkSelector,
    estimate_h_cross,
    h_cross_from_dims,
)

BIG = 10**9


class _FakeHF:
    num_attention_heads = 64
    num_key_value_heads = 8
    intermediate_size = 29568
    n_routed_experts = None
    kv_lora_rank = None


class _FakeModelConfig:
    hf_text_config = _FakeHF()
    hf_config = _FakeHF()
    is_deepseek_mla = False

    def get_hidden_size(self):
        return 8192

    def get_head_size(self):
        return 128

    def get_num_attention_heads(self, pc):  # per-rank, like vllm
        return 64 // pc.tensor_parallel_size

    def get_num_kv_heads(self, pc):
        return max(1, 8 // pc.tensor_parallel_size)


class _FakeParallelConfig:
    def __init__(self, tp):
        self.tensor_parallel_size = tp

DENSE = dict(hidden_size=4096, n_q_heads=32, n_kv_heads=32, head_dim=128,
             intermediate_size=11008, is_mla=False, is_moe=False)
MOE = dict(hidden_size=2048, n_q_heads=32, n_kv_heads=4, head_dim=128,
           intermediate_size=768, is_mla=False, is_moe=True, n_routed_experts=128,
           num_experts_per_tok=8, moe_intermediate_size=768, n_shared_experts=0)
MLA = dict(hidden_size=7168, n_q_heads=128, n_kv_heads=128, head_dim=128,
           intermediate_size=18432, is_mla=True, is_moe=True, kv_lora_rank=512,
           q_lora_rank=1536, qk_rope_head_dim=64, qk_nope_head_dim=128, v_head_dim=128,
           n_routed_experts=256, num_experts_per_tok=8, moe_intermediate_size=2048,
           n_shared_experts=1)


class TestHCross(TestBase):
    def test_per_family_in_sensible_range(self):
        dense = h_cross_from_dims(DENSE, 0.8)
        moe = h_cross_from_dims(MOE, 0.8)
        mla = h_cross_from_dims(MLA, 0.8)
        # dense ~ 6*hidden*k_mfu; MoE lower (sparse FFN); all a few thousand..tens of k
        self.assertTrue(15_000 < dense < 25_000, dense)
        self.assertTrue(3_000 < moe < 9_000, moe)
        self.assertTrue(5_000 < mla < 20_000, mla)
        self.assertLess(moe, dense)  # sparse MoE FFN lowers h_cross vs dense

    def test_k_mfu_scales_linearly(self):
        self.assertAlmostEqual(
            h_cross_from_dims(DENSE, 0.4), h_cross_from_dims(DENSE, 0.8) / 2, delta=2
        )

    def test_estimate_h_cross_is_tp_invariant(self):
        # TP shards attention and FFN equally -> h_cross must not depend on TP.
        model_config = _FakeModelConfig()
        values = {
            estimate_h_cross(model_config, _FakeParallelConfig(tp), 0.8)
            for tp in (1, 2, 4, 8)
        }
        self.assertEqual(len(values), 1, f"h_cross varies with TP: {values}")


class TestAnalyticLatencyModel(TestBase):
    def test_formula_and_monotonicity(self):
        model = AnalyticLatencyModel(10_000)
        self.assertAlmostEqual(model.predict(0, 1000), 1000 * (1 + 500 / 10_000))
        self.assertLess(model.predict(0, 1024), model.predict(50_000, 1024))
        self.assertLess(model.predict(1000, 1024), model.predict(1000, 4096))


class TestSmoothChunkSelector(TestBase):
    def _sel(self):
        return SmoothChunkSelector(AnalyticLatencyModel(12_000), [1024, 2048, 4096, 8192])

    def test_non_increasing_with_history(self):
        sel = self._sel()
        budget = sel.model.predict(0, 8192)
        prev = BIG
        for history in [0, 4000, 16_000, 64_000, 256_000]:
            chunk = sel.select_chunk(history, BIG, BIG, budget)
            self.assertLessEqual(chunk, prev)
            prev = chunk

    def test_floor_and_tail_and_cap(self):
        sel = self._sel()
        self.assertEqual(sel.select_chunk(BIG, BIG, BIG, sel.model.predict(0, 8192)), 1024)
        self.assertEqual(sel.select_chunk(0, 500, BIG, BIG), 500)
        self.assertEqual(sel.select_chunk(0, BIG, 3000, BIG), 2048)
        self.assertIsNone(sel.select_chunk(0, 0, BIG, BIG))


class TestAnalyticChunkManager(TestBase):
    def _mgr(self, **kw):
        base = dict(h_cross=12_000, page_size=128, min_chunk=1024, max_chunk=8192,
                    allowed_chunk_sizes=None, target_chunk=None, backfill_reserve_frac=0.25)
        base.update(kw)
        return AnalyticChunkManager(**base)

    def test_ready_and_budgets(self):
        mgr = self._mgr()
        self.assertTrue(mgr.is_ready)
        self.assertGreaterEqual(mgr.allowed_chunk_sizes[0], 1024)
        self.assertEqual(mgr.target_latency, mgr.model.predict(0, mgr.allowed_chunk_sizes[-1]))
        self.assertAlmostEqual(mgr.backfill_reserve, 0.25 * mgr.target_latency)

    def test_target_chunk_overrides_budget(self):
        mgr = self._mgr(target_chunk=2048)
        self.assertEqual(mgr.target_latency, mgr.model.predict(0, 2048))

    def test_select_and_predict(self):
        mgr = self._mgr()
        chunk = mgr.select_next_chunk(
            history=0, remaining_prompt_tokens=BIG, token_budget=BIG, cost_budget=mgr.target_latency
        )
        self.assertEqual(chunk, mgr.allowed_chunk_sizes[-1])
        self.assertGreater(mgr.predict_cost(0, 1024), 0)


class TestAnalyticChunkConfig(TestBase):
    def test_defaults(self):
        cfg = AnalyticChunkConfig({})
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.min_chunk, 1024)
        self.assertEqual(cfg.k_mfu, 0.8)
        self.assertEqual(cfg.backfill_reserve_frac, 0.25)
        self.assertTrue(cfg.protect_inflight_prefill)
        self.assertIsNone(cfg.h_cross)

    def test_validation(self):
        with self.assertRaises(ValueError):
            AnalyticChunkConfig({"backfill_reserve_frac": 1.5})
        with self.assertRaises(ValueError):
            AnalyticChunkConfig({"h_cross": 0})
        with self.assertRaises(ValueError):
            AnalyticChunkConfig({"min_chunk": 4096, "max_chunk": 1024})
        with self.assertRaises(ValueError):
            AnalyticChunkConfig({"k_mfu": 0})

    def test_explicit_values(self):
        cfg = AnalyticChunkConfig({"enabled": True, "h_cross": 9000, "k_mfu": 0.5,
                                   "target_chunk": 4096, "allowed_chunk_sizes": [1024, 2048]})
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.h_cross, 9000)
        self.assertEqual(cfg.k_mfu, 0.5)
        self.assertEqual(cfg.target_chunk, 4096)
        self.assertEqual(cfg.allowed_chunk_sizes, [1024, 2048])
