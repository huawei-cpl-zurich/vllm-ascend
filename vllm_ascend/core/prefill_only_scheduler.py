# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefillOnly's scheduling policy adapted to a disaggregated prefill node.

This module deliberately reuses PREFLOW's scheduler integration and changes
only request selection. The policy is Algorithm 1 from PrefillOnly
(arXiv:2505.07203): continuously recompute the cache-miss-token JCT proxy,
subtract linear waiting-time aging, and schedule the minimum-score request.
"""

import time
from collections.abc import Iterable
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.core.preflow_scheduler import (
    AsyncPREFLOWScheduler,
    PREFLOWScheduler,
)


class _PrefillOnlyPolicyMixin:
    """Replace PREFLOW's hard shield with PrefillOnly Algorithm 1."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = get_ascend_config().scheduler_config.prefill_only_config
        self.prefill_only_aging_rate = config.aging_rate

        # Section 6.1 of the paper schedules one prefill request at a time.
        # Keep vLLM's resident-request capacity independent from compute width.
        self.preflow_micro_prefill_isl_threshold = 0
        self.preflow_max_num_batched_seqs = 1

        # Queue time starts when the request enters this scheduler. A monotonic
        # clock avoids wall-clock corrections changing the fairness score.
        self._prefill_only_enqueue_time: dict[str, float] = {}

    def _prefill_only_cached_tokens(self, request: Request) -> int:
        """Return the current cache/progress estimate used by Algorithm 1."""
        computed_history = self._preflow_prompt_history(request)
        if computed_history > 0:
            return computed_history

        # Re-query the live local prefix-cache index at every ranking boundary,
        # rather than using PREFLOW's frozen arrival-time cache estimate.
        return self._preflow_estimate_initial_history(request)

    def _prefill_only_cache_miss_tokens(self, request: Request) -> int:
        return max(0, request.num_prompt_tokens - self._prefill_only_cached_tokens(request))

    def _prefill_only_queue_time(self, request: Request, now: float | None = None) -> float:
        if now is None:
            now = time.monotonic()
        enqueue_time = self._prefill_only_enqueue_time[request.request_id]
        return max(0.0, now - enqueue_time)

    def _prefill_only_score(self, request: Request, now: float | None = None) -> float:
        """Return ``n_input - n_cached - lambda * T_queue`` from the paper."""
        return self._prefill_only_cache_miss_tokens(request) - (
            self.prefill_only_aging_rate * self._prefill_only_queue_time(request, now)
        )

    def _preflow_candidate_key(self, request: Request) -> tuple[float, int, str]:
        request_id = request.request_id
        enqueue_time = self._prefill_only_enqueue_time[request_id]

        # For a common scheduling time `now`, Algorithm 1's score is
        #   miss_tokens + lambda * enqueue_time - lambda * now.
        # The last term is identical for every candidate, so omitting it gives
        # the exact same ordering without sampling the clock once per request.
        ordering_score = self._prefill_only_cache_miss_tokens(request) + (self.prefill_only_aging_rate * enqueue_time)
        return (
            ordering_score,
            self._preflow_arrival_order[request_id],
            request_id,
        )

    def _preflow_select_candidate_action(
        self,
        actions: Iterable[tuple[Request, float]],
    ) -> Request | None:
        """Choose the minimum continuously calibrated PrefillOnly score."""
        return min(
            (request for request, _ in actions),
            key=self._preflow_candidate_key,
            default=None,
        )

    def _preflow_chunk_is_safe(
        self,
        request: Request,
        chunk_work: float,
        candidate_remaining_work: float | None = None,
    ) -> bool:
        # PrefillOnly has no hard deadline or feasibility shield.
        return True

    def _preflow_log_admission_not_evaluable(self, request: Request) -> None:
        # This diagnostic is specific to PREFLOW's hard feasibility guarantee.
        return None

    def _preflow_remember_batch_work(
        self,
        scheduler_output: SchedulerOutput,
        scheduled_chunks: list[tuple[str, int, int]],
    ) -> None:
        # PrefillOnly uses wall-clock queue age and cache-miss tokens, not
        # PREFLOW's triangular logical-service clock.
        return None

    def _preflow_apply_completed_batch_work(self, scheduler_output: SchedulerOutput) -> None:
        return None

    def _preflow_register_request(self, request: Request) -> None:
        request_id = request.request_id
        self._preflow_initial_history[request_id] = self._preflow_estimate_initial_history(request)
        self._preflow_arrival_order[request_id] = self._preflow_next_arrival_order
        self._preflow_next_arrival_order += 1
        self._prefill_only_enqueue_time[request_id] = time.monotonic()

    def _preflow_forget_request(self, request_id: str) -> None:
        super()._preflow_forget_request(request_id)
        self._prefill_only_enqueue_time.pop(request_id, None)

    def shutdown(self) -> None:
        self._prefill_only_enqueue_time.clear()
        super().shutdown()


class PrefillOnlyScheduler(_PrefillOnlyPolicyMixin, PREFLOWScheduler):
    """Synchronous PD-prefill scheduler using PrefillOnly Algorithm 1."""


class AsyncPrefillOnlyScheduler(_PrefillOnlyPolicyMixin, AsyncPREFLOWScheduler):
    """Asynchronous PD-prefill scheduler using PrefillOnly Algorithm 1."""


__all__ = ["AsyncPrefillOnlyScheduler", "PrefillOnlyScheduler"]
