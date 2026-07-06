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
"""Lookup-table dynamic chunked prefill scheduling.

The lookup table is the source of truth.  Startup either loads a JSON table or
profiles full forward passes for a fixed grid of history and chunk sizes.  At
runtime a small receding-horizon beam search chooses the next chunk from the
allowed chunk set.
"""

import json
import os
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from vllm.logger import logger

PROFILE_SCHEMA_VERSION = 1


def _align_down(value: int, alignment: int) -> int:
    return value // alignment * alignment


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)


def generate_allowed_chunk_sizes(
    min_chunk: int,
    max_chunk: int,
    alignment: int,
) -> list[int]:
    """Generate hardware-friendly chunks of the form 2^n and 3*2^(n-1)."""
    candidates: set[int] = set()
    power = 1
    while power <= max_chunk:
        if power >= min_chunk:
            candidates.add(power)
        three_half = 3 * power // 2
        if power % 2 == 0 and min_chunk <= three_half <= max_chunk:
            candidates.add(three_half)
        power *= 2

    aligned = {
        _align_up(chunk_size, alignment)
        for chunk_size in candidates
        if min_chunk <= chunk_size <= max_chunk
    }
    return sorted(
        chunk_size
        for chunk_size in aligned
        if min_chunk <= chunk_size <= max_chunk
        and chunk_size % alignment == 0
    )


def generate_history_sizes(
    max_model_len: int,
    num_points: int,
    alignment: int,
) -> list[int]:
    """Generate an evenly spaced, aligned history grid."""
    history_sizes: set[int] = {0, max_model_len}
    for index in range(num_points):
        raw_history = round(index * max_model_len / (num_points - 1))
        if index == num_points - 1:
            history = max_model_len
        elif raw_history == 0:
            history = 0
        else:
            history = _align_down(raw_history, alignment)
        history_sizes.add(max(0, min(history, max_model_len)))
    return sorted(history_sizes)


@dataclass
class ChunkLatencyTable:
    history_sizes: list[int]
    chunk_sizes: list[int]
    latencies_ms: list[list[float | None]]
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_json_file(cls, path: str) -> "ChunkLatencyTable":
        with open(path, encoding="utf-8") as profile_file:
            payload = json.load(profile_file)

        history_sizes = [int(item) for item in payload["history_sizes"]]
        chunk_sizes = [int(item) for item in payload["chunk_sizes"]]
        latencies_ms = [
            [None if value is None else float(value) for value in row]
            for row in payload["latencies_ms"]
        ]
        table = cls(
            history_sizes=history_sizes,
            chunk_sizes=chunk_sizes,
            latencies_ms=latencies_ms,
            metadata=dict(payload.get("metadata", {})),
        )
        table.validate()
        return table

    def to_json_file(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "metadata": self.metadata or {},
            "history_sizes": self.history_sizes,
            "chunk_sizes": self.chunk_sizes,
            "latencies_ms": self.latencies_ms,
        }
        with open(path, "w", encoding="utf-8") as profile_file:
            json.dump(payload, profile_file, indent=2, sort_keys=True)

    def validate(self) -> None:
        if not self.history_sizes:
            raise ValueError("profiling table history_sizes is empty")
        if not self.chunk_sizes:
            raise ValueError("profiling table chunk_sizes is empty")
        if sorted(set(self.history_sizes)) != self.history_sizes:
            raise ValueError("profiling table history_sizes must be unique and sorted")
        if sorted(set(self.chunk_sizes)) != self.chunk_sizes:
            raise ValueError("profiling table chunk_sizes must be unique and sorted")
        if len(self.latencies_ms) != len(self.history_sizes):
            raise ValueError("latency row count does not match history count")
        for row in self.latencies_ms:
            if len(row) != len(self.chunk_sizes):
                raise ValueError("latency column count does not match chunk count")

    def has_usable_entry(self) -> bool:
        return any(value is not None for row in self.latencies_ms for value in row)

    def _history_index(self, history: int) -> int:
        index = bisect_left(self.history_sizes, history)
        if index >= len(self.history_sizes):
            return len(self.history_sizes) - 1
        return index

    def latency_or_none(self, history: int, chunk: int) -> float | None:
        try:
            chunk_index = self.chunk_sizes.index(chunk)
        except ValueError:
            return None
        return self.latencies_ms[self._history_index(history)][chunk_index]

    def lookup(self, history: int, chunk: int) -> float:
        latency = self.latency_or_none(history, chunk)
        if latency is None:
            raise ValueError(f"missing latency for history={history}, chunk={chunk}")
        return latency


class FullForwardChunkProfiler:
    """Root-side orchestration for distributed full-forward profiling."""

    def __init__(
        self,
        model_executor: Any,
        metadata: dict[str, Any],
    ) -> None:
        self.model_executor = model_executor
        self.metadata = metadata

    def profile(
        self,
        history_sizes: list[int],
        chunk_sizes: list[int],
        repeats: int,
        max_model_len: int,
    ) -> ChunkLatencyTable:
        latencies_ms: list[list[float | None]] = []
        total_points = len(history_sizes) * len(chunk_sizes)
        completed = 0

        logger.info(
            "[ProfilingChunk] Running full-forward profiling: %d histories, "
            "%d chunks, %d repeats",
            len(history_sizes),
            len(chunk_sizes),
            repeats,
        )

        for history in history_sizes:
            row: list[float | None] = []
            for chunk in chunk_sizes:
                completed += 1
                if history + chunk > max_model_len:
                    row.append(None)
                    continue
                try:
                    result = self.model_executor.collective_rpc(
                        "profile_prefill_latency",
                        args=(history, chunk, repeats),
                    )
                    latency_ms = self._extract_latency(result)
                    row.append(latency_ms)
                    logger.info(
                        "[ProfilingChunk] Profiled %d/%d: history=%d, chunk=%d, "
                        "latency=%s ms",
                        completed,
                        total_points,
                        history,
                        chunk,
                        "%.3f" % latency_ms if latency_ms is not None else "null",
                    )
                except Exception as e:
                    logger.warning(
                        "[ProfilingChunk] Profiling failed for history=%d, "
                        "chunk=%d: %s",
                        history,
                        chunk,
                        e,
                    )
                    row.append(None)
            latencies_ms.append(row)

        table = ChunkLatencyTable(
            history_sizes=history_sizes,
            chunk_sizes=chunk_sizes,
            latencies_ms=latencies_ms,
            metadata=self.metadata,
        )
        table.validate()
        if not table.has_usable_entry():
            raise RuntimeError("profiling produced no usable latency entries")
        return table

    @classmethod
    def _extract_latency(cls, result: Any) -> float | None:
        runs = cls._extract_runs(result)
        if runs:
            return _median(runs)
        values = cls._extract_scalar_latencies(result)
        if not values:
            return None
        return max(values)

    @classmethod
    def _extract_runs(cls, result: Any) -> list[float]:
        if isinstance(result, dict) and isinstance(result.get("runs_ms"), list):
            return [float(value) for value in result["runs_ms"]]
        if isinstance(result, list):
            per_rank_runs = []
            for item in result:
                runs = cls._extract_runs(item)
                if runs:
                    per_rank_runs.append(runs)
            if not per_rank_runs:
                return []
            min_len = min(len(runs) for runs in per_rank_runs)
            return [
                max(runs[index] for runs in per_rank_runs)
                for index in range(min_len)
            ]
        return []

    @classmethod
    def _extract_scalar_latencies(cls, result: Any) -> list[float]:
        if isinstance(result, (int, float)):
            return [float(result)]
        if isinstance(result, dict):
            for key in ("latency_ms", "median_ms"):
                if key in result and result[key] is not None:
                    return [float(result[key])]
            return []
        if isinstance(result, list):
            values: list[float] = []
            for item in result:
                values.extend(cls._extract_scalar_latencies(item))
            return values
        return []


@dataclass
class LatencyModel:
    """Least-squares fit of per-forward prefill latency from the lookup table.

    ``T(h, c) = e + b*c + a_cross*(c*h) + a_intra*(c*c)``

    The shape coefficients are clamped non-negative so ``T`` is monotonically
    non-decreasing in both history and chunk size -- which is what lets
    :class:`SmoothChunkSelector` scan chunk sizes in ascending order.
    """

    e: float
    b: float
    a_cross: float
    a_intra: float
    max_rel_residual: float = 0.0

    @classmethod
    def fit(cls, table: "ChunkLatencyTable") -> "LatencyModel":
        features: list[list[float]] = []
        targets: list[float] = []
        for row_index, history in enumerate(table.history_sizes):
            for col_index, chunk in enumerate(table.chunk_sizes):
                latency = table.latencies_ms[row_index][col_index]
                if latency is None:
                    continue
                features.append(
                    [1.0, float(chunk), float(chunk) * history, float(chunk) * chunk]
                )
                targets.append(float(latency))
        if len(features) < 4:
            raise ValueError("not enough usable latency entries to fit the latency model")
        coeffs, *_ = np.linalg.lstsq(
            np.asarray(features, dtype=np.float64),
            np.asarray(targets, dtype=np.float64),
            rcond=None,
        )
        model = cls(
            e=float(coeffs[0]),
            b=max(float(coeffs[1]), 0.0),
            a_cross=max(float(coeffs[2]), 0.0),
            a_intra=max(float(coeffs[3]), 0.0),
        )
        model.max_rel_residual = model._max_rel_residual(features, targets)
        return model

    def _max_rel_residual(self, features: list[list[float]], targets: list[float]) -> float:
        worst = 0.0
        for feat, actual in zip(features, targets):
            predicted = (
                self.e + self.b * feat[1] + self.a_cross * feat[2] + self.a_intra * feat[3]
            )
            if actual > 0:
                worst = max(worst, abs(predicted - actual) / actual)
        return worst

    def predict(self, history: int, chunk: int) -> float:
        return (
            self.e
            + self.b * chunk
            + self.a_cross * chunk * history
            + self.a_intra * chunk * chunk
        )


class SmoothChunkSelector:
    """Floor-tracking chunk selection under a shared per-step cost budget.

    Returns the largest allowed chunk whose predicted latency stays under the
    per-request target ``max(cost_budget, floor(history))``, where
    ``floor(history) = T(history, c_min)``.  Because the target is always at
    least the floor, ``c_min`` is always feasible so progress is guaranteed;
    the final short chunk returns the exact remainder.
    """

    def __init__(self, model: LatencyModel, allowed_chunk_sizes: list[int]) -> None:
        self.model = model
        self.allowed = sorted(allowed_chunk_sizes)
        self.c_min = self.allowed[0]

    def select_chunk(
        self,
        history: int,
        remaining_prompt_tokens: int,
        token_budget: int,
        cost_budget: float,
    ) -> int | None:
        cap = min(remaining_prompt_tokens, token_budget)
        if cap <= 0:
            return None
        if cap <= self.c_min:
            # Tail shorter than one min-chunk: schedule the exact remainder.
            return cap
        target = max(cost_budget, self.model.predict(history, self.c_min))
        best = self.c_min
        for chunk in self.allowed:
            if chunk > cap:
                break
            if self.model.predict(history, chunk) <= target:
                best = chunk
        return best


class ProfilingChunkManager:
    """Owns the lookup table and chunk optimizer used by the scheduler."""

    def __init__(
        self,
        *,
        max_model_len: int,
        page_size: int,
        min_chunk: int,
        max_chunk: int,
        allowed_chunk_sizes: list[int] | None,
        profile_num_history_points: int,
        profile_repeats: int,
        target_latency_ms: float | None,
        backfill_reserve_ms: float | None,
        profile_file: str | None,
        metadata: dict[str, Any],
    ) -> None:
        self.max_model_len = max_model_len
        self.page_size = page_size
        self.alignment = max(page_size, 64)
        self.min_chunk = min_chunk
        self.max_chunk = max_chunk
        self.allowed_chunk_sizes = self._resolve_allowed_chunk_sizes(allowed_chunk_sizes)
        self.history_sizes = generate_history_sizes(
            max_model_len=max_model_len,
            num_points=profile_num_history_points,
            alignment=self.alignment,
        )
        self.profile_repeats = profile_repeats
        self.target_latency_ms_config = target_latency_ms
        self.backfill_reserve_ms_config = backfill_reserve_ms
        self.profile_file = profile_file
        self.metadata = metadata
        self.table: ChunkLatencyTable | None = None
        self.model: LatencyModel | None = None
        self.selector: SmoothChunkSelector | None = None
        self._target_latency_ms: float | None = None
        self._backfill_reserve_ms: float | None = None

    # Default backfill reserve when unset: a quarter of the step budget kept
    # available every step for admitting small piggyback requests.
    AUTO_BACKFILL_FRACTION = 0.25

    @property
    def is_ready(self) -> bool:
        return self.selector is not None

    @property
    def target_latency_ms(self) -> float | None:
        """Per-step cost budget ``T*`` (ms).  ``None`` until the model is fit."""
        return self._target_latency_ms

    @property
    def backfill_reserve_ms(self) -> float | None:
        """Cost budget (ms) reserved every step for waiting requests."""
        return self._backfill_reserve_ms

    def initialize(self, model_executor: Any) -> None:
        if self.profile_file is not None and Path(self.profile_file).exists():
            try:
                self._set_table(ChunkLatencyTable.from_json_file(self.profile_file))
                logger.info("[ProfilingChunk] Loaded latency table from %s", self.profile_file)
                return
            except Exception as e:
                logger.warning(
                    "[ProfilingChunk] Failed to load profiling file %s: %s. "
                    "Reprofiling.",
                    self.profile_file,
                    e,
                )

        if model_executor is None:
            logger.warning("[ProfilingChunk] No model_executor provided, skipping profiling")
            return

        profiler = FullForwardChunkProfiler(model_executor=model_executor, metadata=self.metadata)
        table = profiler.profile(
            history_sizes=self.history_sizes,
            chunk_sizes=self.allowed_chunk_sizes,
            repeats=self.profile_repeats,
            max_model_len=self.max_model_len,
        )
        self._set_table(table)
        if self.profile_file is not None:
            try:
                table.to_json_file(self.profile_file)
                logger.info("[ProfilingChunk] Saved latency table to %s", self.profile_file)
            except Exception as e:
                logger.warning(
                    "[ProfilingChunk] Failed to write profiling file %s: %s",
                    self.profile_file,
                    e,
                )

    def select_next_chunk(
        self,
        *,
        history: int,
        remaining_prompt_tokens: int,
        token_budget: int,
        cost_budget: float,
    ) -> int | None:
        if self.selector is None:
            return None
        return self.selector.select_chunk(
            history=history,
            remaining_prompt_tokens=remaining_prompt_tokens,
            token_budget=token_budget,
            cost_budget=cost_budget,
        )

    def predict_cost(self, history: int, chunk: int) -> float:
        """Predicted per-forward latency (ms) for ``chunk`` at ``history``."""
        if self.model is None:
            return 0.0
        return self.model.predict(history, chunk)

    def _set_table(self, table: ChunkLatencyTable) -> None:
        table.validate()
        if not table.has_usable_entry():
            raise ValueError("profiling table has no usable latency entries")
        self.table = table
        self.allowed_chunk_sizes = table.chunk_sizes
        self.model = LatencyModel.fit(table)
        self.selector = SmoothChunkSelector(self.model, table.chunk_sizes)
        if self.target_latency_ms_config is not None:
            self._target_latency_ms = float(self.target_latency_ms_config)
        else:
            # Default target = cost of the largest chunk at zero history.
            self._target_latency_ms = self.model.predict(0, table.chunk_sizes[-1])
        if self.backfill_reserve_ms_config is not None:
            self._backfill_reserve_ms = float(self.backfill_reserve_ms_config)
        else:
            self._backfill_reserve_ms = self.AUTO_BACKFILL_FRACTION * self._target_latency_ms
        logger.info(
            "[ProfilingChunk] Fitted latency model: e=%.3g, b=%.3g, a_cross=%.3g, "
            "a_intra=%.3g (max rel residual=%.1f%%); target_latency=%.3f ms",
            self.model.e,
            self.model.b,
            self.model.a_cross,
            self.model.a_intra,
            self.model.max_rel_residual * 100.0,
            self._target_latency_ms,
        )

    def _resolve_allowed_chunk_sizes(self, configured: list[int] | None) -> list[int]:
        if configured is not None:
            chunk_sizes = sorted(set(configured))
        else:
            chunk_sizes = generate_allowed_chunk_sizes(
                min_chunk=self.min_chunk,
                max_chunk=self.max_chunk,
                alignment=self.alignment,
            )

        if not chunk_sizes:
            raise ValueError("profiling_chunk_config produced no allowed chunk sizes")
        for chunk_size in chunk_sizes:
            if chunk_size <= 0:
                raise ValueError("allowed chunk sizes must be positive")
            if chunk_size < self.min_chunk or chunk_size > self.max_chunk:
                raise ValueError(
                    "allowed chunk size %d is outside [%d, %d]"
                    % (chunk_size, self.min_chunk, self.max_chunk)
                )
            if chunk_size % self.alignment != 0:
                raise ValueError(
                    "allowed chunk size %d must be aligned to %d"
                    % (chunk_size, self.alignment)
                )
        return chunk_sizes
