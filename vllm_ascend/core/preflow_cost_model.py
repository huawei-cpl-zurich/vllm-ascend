# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Work models and startup-profile fitting for PREFLOW."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_MIN_COST = float(np.finfo(np.float64).eps)

PREFLOW_PROFILE_METADATA_ATTR = "_vllm_ascend_preflow_profile_metadata"
PREFLOW_PROFILE_ELAPSED_MS_ATTR = "_vllm_ascend_preflow_profile_elapsed_ms"
PREFLOW_CALIBRATION_REQUEST_ATTR = "_vllm_ascend_preflow_calibration_request"
PREFLOW_CALIBRATION_REQUEST_PREFIX = "__vllm_ascend_preflow_calibration__"


@dataclass(frozen=True)
class PreflowProfileSample:
    history: int
    chunk_size: int
    elapsed_ms: float
    is_final: bool
    discard: bool


class TriangularPrefillCostModel:
    """The original triangular PREFLOW work proxy."""

    name = "triangular"

    @staticmethod
    def _work(num_tokens: int) -> float:
        tokens = float(max(0, int(num_tokens)))
        return tokens * (tokens + 1.0) / 2.0

    def chunk_cost(
        self,
        history: int,
        chunk_size: int,
        *,
        is_final: bool = False,
    ) -> float:
        del is_final
        return self._work(history + chunk_size) - self._work(history)

    def interval_cost(
        self,
        start_history: int,
        end_history: int,
        *,
        include_final: bool = True,
    ) -> float:
        del include_final
        return max(0.0, self._work(end_history) - self._work(start_history))


@dataclass(frozen=True)
class ProfiledPrefillCostModel:
    """Deterministic fixed-chunk PREFLOW runtime model, measured in ms."""

    chunk_size: int
    history_scale: float
    theta_0: float
    theta_1: float
    theta_2: float
    short_chunk_sizes: tuple[int, ...]
    short_chunk_costs: tuple[float, ...]
    final_overhead: float

    name = "profiled"

    def _full_chunk_cost(self, history: int | float) -> float:
        z = max(0.0, float(history)) / self.history_scale
        return max(
            _MIN_COST,
            self.theta_0 + self.theta_1 * z + self.theta_2 * z * z,
        )

    def _short_cost(self, chunk_size: int) -> float:
        if chunk_size <= 0:
            return 0.0
        sizes = self.short_chunk_sizes
        costs = self.short_chunk_costs
        if chunk_size <= sizes[0]:
            # A real non-empty invocation pays the fixed launch floor.
            return costs[0]
        for index in range(1, len(sizes)):
            if chunk_size <= sizes[index]:
                lo_size = sizes[index - 1]
                hi_size = sizes[index]
                fraction = (chunk_size - lo_size) / (hi_size - lo_size)
                return costs[index - 1] + fraction * (costs[index] - costs[index - 1])
        return costs[-1]

    def chunk_cost(
        self,
        history: int,
        chunk_size: int,
        *,
        is_final: bool = False,
    ) -> float:
        history = max(0, int(history))
        chunk_size = max(0, int(chunk_size))
        if chunk_size == 0:
            return 0.0
        if chunk_size == self.chunk_size:
            cost = self._full_chunk_cost(history)
        elif chunk_size < self.chunk_size:
            fraction = chunk_size / self.chunk_size
            cost = self._short_cost(chunk_size) + fraction * (self._full_chunk_cost(history) - self._full_chunk_cost(0))
        else:
            # This should not occur with the calibrated static chunk size, but
            # retaining additivity makes the model safe for unusual token-budget
            # configurations rather than extrapolating S(c) beyond C.
            cost = self.interval_cost(
                history,
                history + chunk_size,
                include_final=False,
            )
        if is_final:
            cost += self.final_overhead
        return max(_MIN_COST, cost)

    def interval_cost(
        self,
        start_history: int,
        end_history: int,
        *,
        include_final: bool = True,
    ) -> float:
        start = max(0, int(start_history))
        end = max(start, int(end_history))
        distance = end - start
        if distance == 0:
            return 0.0

        num_full_chunks, tail = divmod(distance, self.chunk_size)
        total = 0.0
        if num_full_chunks:
            count = float(num_full_chunks)
            step = float(self.chunk_size)
            base = float(start)
            sum_history = count * base + step * count * (count - 1.0) / 2.0
            sum_history_sq = (
                count * base * base
                + base * step * count * (count - 1.0)
                + step * step * count * (count - 1.0) * (2.0 * count - 1.0) / 6.0
            )
            total += (
                count * self.theta_0
                + self.theta_1 * sum_history / self.history_scale
                + self.theta_2 * sum_history_sq / (self.history_scale * self.history_scale)
            )
        if tail:
            tail_history = start + num_full_chunks * self.chunk_size
            total += self.chunk_cost(tail_history, tail, is_final=False)
        if include_final:
            total += self.final_overhead
        return max(0.0, total)


def _project_monotone_quadratic_slopes(
    linear: float,
    quadratic: float,
    max_z: float,
) -> tuple[float, float]:
    """Project slopes onto dT/dz >= 0 for every z in [0, max_z]."""
    if linear >= 0.0 and linear + 2.0 * quadratic * max_z >= 0.0:
        return linear, quadratic

    candidates: list[tuple[float, float]] = [(0.0, 0.0)]

    # Boundary linear == 0, whose remaining constraint is quadratic >= 0.
    candidates.append((0.0, max(0.0, quadratic)))

    # Boundary linear + 2 * quadratic * max_z == 0.
    normal_linear = 1.0
    normal_quadratic = 2.0 * max_z
    denominator = normal_linear**2 + normal_quadratic**2
    projection_scale = (linear * normal_linear + quadratic * normal_quadratic) / denominator
    boundary_linear = linear - projection_scale * normal_linear
    boundary_quadratic = quadratic - projection_scale * normal_quadratic
    if boundary_linear >= 0.0:
        candidates.append((boundary_linear, boundary_quadratic))

    return min(
        candidates,
        key=lambda value: (value[0] - linear) ** 2 + (value[1] - quadratic) ** 2,
    )


def fit_profiled_prefill_cost_model(
    samples: list[PreflowProfileSample],
    *,
    chunk_size: int,
    calibration_history: int,
    maximum_history: int,
) -> ProfiledPrefillCostModel:
    """Fit the constrained deterministic PREFLOW model from startup data."""
    usable = [sample for sample in samples if not sample.discard]
    full_nonfinal = [sample for sample in usable if sample.chunk_size == chunk_size and not sample.is_final]
    if len(full_nonfinal) < 3:
        raise RuntimeError(
            f"PREFLOW profiling needs at least three measured non-final full chunks; collected {len(full_nonfinal)}."
        )

    history_scale = float(max(1, calibration_history))
    full_by_history: dict[int, list[float]] = {}
    for sample in full_nonfinal:
        full_by_history.setdefault(sample.history, []).append(sample.elapsed_ms)
    ordered_histories = sorted(full_by_history)
    histories = np.asarray(
        [history / history_scale for history in ordered_histories],
        dtype=np.float64,
    )
    latencies = np.asarray(
        [np.median(np.asarray(full_by_history[history], dtype=np.float64)) for history in ordered_histories],
        dtype=np.float64,
    )
    design = np.column_stack((np.ones_like(histories), histories, histories * histories))
    coefficients, _, _, _ = np.linalg.lstsq(design, latencies, rcond=None)
    theta_0, theta_1, theta_2 = (float(value) for value in coefficients)
    if not all(math.isfinite(value) for value in (theta_0, theta_1, theta_2)):
        raise RuntimeError("PREFLOW profiling produced non-finite full-chunk coefficients.")

    max_z = max(1.0, maximum_history / history_scale)
    theta_1, theta_2 = _project_monotone_quadratic_slopes(theta_1, theta_2, max_z)
    theta_0 = float(np.mean(latencies - theta_1 * histories - theta_2 * histories * histories))
    theta_0 = max(theta_0, _MIN_COST)

    def full_cost(history: int) -> float:
        z = max(0.0, float(history)) / history_scale
        return max(
            _MIN_COST,
            theta_0 + theta_1 * z + theta_2 * z * z,
        )

    final_full = [sample for sample in usable if sample.chunk_size == chunk_size and sample.is_final]
    if not final_full:
        raise RuntimeError("PREFLOW profiling did not collect a final full-chunk sample.")
    final_residuals = [sample.elapsed_ms - full_cost(sample.history) for sample in final_full]
    final_overhead = max(0.0, float(np.median(np.asarray(final_residuals, dtype=np.float64))))

    short_observations: dict[int, list[float]] = {}
    for sample in usable:
        if not sample.is_final or sample.history != 0:
            continue
        base_cost = max(
            _MIN_COST,
            sample.elapsed_ms - final_overhead,
        )
        short_observations.setdefault(sample.chunk_size, []).append(base_cost)
    short_observations.setdefault(chunk_size, []).append(full_cost(0))
    if len(short_observations) < 2:
        raise RuntimeError("PREFLOW profiling did not collect enough short-query sizes.")

    sizes = sorted(short_observations)
    raw_costs = [float(np.median(np.asarray(short_observations[size], dtype=np.float64))) for size in sizes]
    # Project the sparse S(c) observations onto positive, non-decreasing costs
    # and anchor S(C) to the fitted full-chunk curve.
    anchor = full_cost(0)
    projected_costs: list[float] = []
    running_cost = _MIN_COST
    for size, raw_cost in zip(sizes, raw_costs):
        bounded_cost = min(raw_cost, anchor) if size < chunk_size else anchor
        running_cost = max(running_cost, bounded_cost)
        projected_costs.append(running_cost)
    if sizes[-1] != chunk_size:
        sizes.append(chunk_size)
        projected_costs.append(anchor)
    else:
        projected_costs[-1] = anchor
    for index in range(len(projected_costs) - 2, -1, -1):
        projected_costs[index] = min(projected_costs[index], projected_costs[index + 1])

    if not all(math.isfinite(value) and value > 0.0 for value in projected_costs):
        raise RuntimeError("PREFLOW profiling produced non-finite or non-positive short-query costs.")

    return ProfiledPrefillCostModel(
        chunk_size=chunk_size,
        history_scale=history_scale,
        theta_0=theta_0,
        theta_1=theta_1,
        theta_2=theta_2,
        short_chunk_sizes=tuple(sizes),
        short_chunk_costs=tuple(projected_costs),
        final_overhead=final_overhead,
    )


__all__ = [
    "PREFLOW_CALIBRATION_REQUEST_ATTR",
    "PREFLOW_CALIBRATION_REQUEST_PREFIX",
    "PREFLOW_PROFILE_ELAPSED_MS_ATTR",
    "PREFLOW_PROFILE_METADATA_ATTR",
    "PreflowProfileSample",
    "ProfiledPrefillCostModel",
    "TriangularPrefillCostModel",
    "fit_profiled_prefill_cost_model",
]
