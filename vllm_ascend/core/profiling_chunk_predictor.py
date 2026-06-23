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


@dataclass(frozen=True)
class SearchState:
    history: int
    scheduled_tokens: int
    sum_latency_ms: float
    max_latency_ms: float
    chunks: tuple[int, ...]


class ChunkBeamSearch:
    """Bounded receding-horizon beam search over discrete chunk sizes."""

    def __init__(
        self,
        table: ChunkLatencyTable,
        allowed_chunk_sizes: list[int],
        search_depth: int,
        beam_width: int,
    ) -> None:
        self.table = table
        self.allowed_chunk_sizes = allowed_chunk_sizes
        self.search_depth = search_depth
        self.beam_width = beam_width

    @property
    def min_chunk(self) -> int:
        return self.allowed_chunk_sizes[0]

    def select_next_chunk(
        self,
        history: int,
        remaining_prompt_tokens: int,
        token_budget: int,
        pp_size: int,
    ) -> int | None:
        if remaining_prompt_tokens <= 0 or token_budget <= 0:
            return None

        initial = SearchState(
            history=history,
            scheduled_tokens=0,
            sum_latency_ms=0.0,
            max_latency_ms=0.0,
            chunks=(),
        )
        beam = [initial]
        best_states: list[SearchState] = []

        for _ in range(self.search_depth):
            candidates: list[SearchState] = []
            for state in beam:
                remaining = remaining_prompt_tokens - state.scheduled_tokens
                if remaining <= 0:
                    best_states.append(state)
                    continue
                candidates.extend(
                    self._expand_state(
                        state=state,
                        remaining=remaining,
                        token_budget=token_budget,
                    )
                )

            if not candidates:
                break

            candidates = self._remove_dominated(candidates)
            candidates.sort(key=lambda state: self._sort_key(state, pp_size))
            beam = candidates[: self.beam_width]
            best_states.extend(
                state
                for state in beam
                if state.scheduled_tokens >= remaining_prompt_tokens
            )
            if all(state.scheduled_tokens >= remaining_prompt_tokens for state in beam):
                break

        selectable = [state for state in (best_states or beam) if state.chunks]
        if not selectable:
            return None
        selectable.sort(key=lambda state: self._sort_key(state, pp_size))
        return selectable[0].chunks[0]

    def fallback_chunk(
        self,
        history: int,
        remaining_prompt_tokens: int,
        token_budget: int,
    ) -> int | None:
        if remaining_prompt_tokens <= 0 or token_budget <= 0:
            return None
        if remaining_prompt_tokens < self.min_chunk and remaining_prompt_tokens <= token_budget:
            return remaining_prompt_tokens
        for chunk in reversed(self.allowed_chunk_sizes):
            if chunk <= remaining_prompt_tokens and chunk <= token_budget:
                if self.table.latency_or_none(history, chunk) is not None:
                    return chunk
        return None

    def _expand_state(
        self,
        state: SearchState,
        remaining: int,
        token_budget: int,
    ) -> list[SearchState]:
        expanded: list[SearchState] = []
        if remaining < self.min_chunk and remaining <= token_budget:
            latency = self.table.latency_or_none(state.history, self.min_chunk)
            if latency is not None:
                expanded.append(self._append_chunk(state, remaining, latency))
            return expanded

        for chunk in self.allowed_chunk_sizes:
            if chunk > remaining or chunk > token_budget:
                continue
            latency = self.table.latency_or_none(state.history, chunk)
            if latency is None:
                continue
            expanded.append(self._append_chunk(state, chunk, latency))
        return expanded

    @staticmethod
    def _append_chunk(
        state: SearchState,
        chunk: int,
        latency: float,
    ) -> SearchState:
        return SearchState(
            history=state.history + chunk,
            scheduled_tokens=state.scheduled_tokens + chunk,
            sum_latency_ms=state.sum_latency_ms + latency,
            max_latency_ms=max(state.max_latency_ms, latency),
            chunks=state.chunks + (chunk,),
        )

    @staticmethod
    def _remove_dominated(states: list[SearchState]) -> list[SearchState]:
        grouped: dict[int, list[SearchState]] = {}
        for state in states:
            grouped.setdefault(state.history, []).append(state)

        kept: list[SearchState] = []
        for group in grouped.values():
            for candidate in group:
                dominated = False
                for other in group:
                    if candidate is other:
                        continue
                    no_worse = (
                        other.sum_latency_ms <= candidate.sum_latency_ms
                        and other.max_latency_ms <= candidate.max_latency_ms
                    )
                    strictly_better = (
                        other.sum_latency_ms < candidate.sum_latency_ms
                        or other.max_latency_ms < candidate.max_latency_ms
                    )
                    if no_worse and strictly_better:
                        dominated = True
                        break
                if not dominated:
                    kept.append(candidate)
        return kept

    @staticmethod
    def _pipeline_cost(state: SearchState, pp_size: int) -> float:
        return state.sum_latency_ms + max(pp_size - 1, 0) * state.max_latency_ms

    def _sort_key(self, state: SearchState, pp_size: int) -> tuple[float, float, int, int, int]:
        cost = self._pipeline_cost(state, pp_size)
        score = cost / state.scheduled_tokens
        first_chunk = state.chunks[0]
        return (
            score,
            cost,
            -state.scheduled_tokens,
            len(state.chunks),
            -first_chunk,
        )


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
        search_depth: int,
        beam_width: int,
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
        self.search_depth = search_depth
        self.beam_width = beam_width
        self.profile_file = profile_file
        self.metadata = metadata
        self.table: ChunkLatencyTable | None = None
        self.optimizer: ChunkBeamSearch | None = None

    @property
    def is_ready(self) -> bool:
        return self.optimizer is not None

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
        pp_size: int,
    ) -> int | None:
        if self.optimizer is None:
            return None
        chunk = self.optimizer.select_next_chunk(
            history=history,
            remaining_prompt_tokens=remaining_prompt_tokens,
            token_budget=token_budget,
            pp_size=pp_size,
        )
        if chunk is not None:
            return chunk
        fallback = self.optimizer.fallback_chunk(
            history=history,
            remaining_prompt_tokens=remaining_prompt_tokens,
            token_budget=token_budget,
        )
        if fallback is None:
            logger.warning(
                "[ProfilingChunk] Optimizer could not find a valid chunk for "
                "history=%d, remaining=%d, token_budget=%d",
                history,
                remaining_prompt_tokens,
                token_budget,
            )
        else:
            logger.warning(
                "[ProfilingChunk] Falling back to chunk=%d for history=%d",
                fallback,
                history,
            )
        return fallback

    def _set_table(self, table: ChunkLatencyTable) -> None:
        table.validate()
        if not table.has_usable_entry():
            raise ValueError("profiling table has no usable latency entries")
        self.table = table
        self.allowed_chunk_sizes = table.chunk_sizes
        self.optimizer = ChunkBeamSearch(
            table=table,
            allowed_chunk_sizes=table.chunk_sizes,
            search_depth=self.search_depth,
            beam_width=self.beam_width,
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
