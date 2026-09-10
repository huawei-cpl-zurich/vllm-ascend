# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill scheduling baselines implemented on PREFLOW's engine path.

These schedulers deliberately inherit PREFLOW's admission, KV-cache,
chunk-construction, queue, request-lifecycle, and output-integration behavior.
Only the policy used to select the next single-request prefill chunk changes.

The policies are intended both as experimental baselines and as precise
ablations of hard PREFLOW:

* FCFS selects the oldest unfinished prefill at every chunk boundary.
* SJF selects the smallest isolated prefill and locks it until its final
  prefill chunk has been dispatched.
* SRPT selects the smallest triangular remaining work at every boundary.
* EDF selects the earliest frozen FCFS-relative deadline at every boundary.
"""

from collections.abc import Iterable
from typing import Any

from vllm.v1.request import Request

from vllm_ascend.core.preflow_scheduler import AsyncPREFLOWScheduler, PREFLOWScheduler


class _UnshieldedPrefillPolicyMixin:
    """Disable PREFLOW's feasibility shield without changing execution."""

    def _preflow_chunk_is_safe(
        self,
        request: Request,
        chunk_work: float,
        candidate_remaining_work: float | None = None,
    ) -> bool:
        return True

    def _preflow_log_admission_not_evaluable(self, request: Request) -> None:
        # Only hard PREFLOW makes an FCFS-relative deadline guarantee.
        return None

    def _preflow_admission_action_is_safe(
        self,
        request: Request,
        chunk_work: float,
        *,
        admission: bool,
        candidate_remaining_work: float | None = None,
    ) -> bool:
        # Experimental baselines preserve vLLM admission without PREFLOW's
        # resident-drain protection.
        return True


class _FCFSPolicyMixin(_UnshieldedPrefillPolicyMixin):
    """Select requests in immutable scheduler-arrival order."""

    def _preflow_candidate_key(self, request: Request) -> tuple[int, str]:
        request_id = request.request_id
        return self._preflow_arrival_order[request_id], request_id

    def _preflow_select_candidate_action(
        self,
        actions: Iterable[tuple[Request, float]],
    ) -> Request | None:
        return min(
            (request for request, _ in actions),
            key=self._preflow_candidate_key,
            default=None,
        )


class _SJFPolicyMixin(_UnshieldedPrefillPolicyMixin):
    """Non-preemptive shortest-job-first over triangular isolated work.

    Physical execution remains chunked. Once the first chunk of a request is
    actually scheduled, the logical lock prevents another prefill from being
    selected until that request's final prompt chunk has been dispatched. In
    async mode, a later request may then be queued behind that final chunk;
    FIFO batch execution still preserves non-preemptive SJF service order.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sjf_locked_request_id: str | None = None

    def _sjf_active_locked_request(self) -> Request | None:
        """Return the lock holder while it has undispatched prefill work."""
        request_id = self._sjf_locked_request_id
        if request_id is None:
            return None
        request = self.requests.get(request_id)
        if request is None or request.is_finished() or not self._preflow_has_unfinished_prefill(request):
            self._sjf_locked_request_id = None
            return None
        return request

    def _sjf_locked_request(
        self,
        actions: Iterable[tuple[Request, float]],
    ) -> Request | None:
        request = self._sjf_active_locked_request()
        if request is None:
            return None
        return next(
            (candidate for candidate, _ in actions if candidate.request_id == request.request_id),
            None,
        )

    def _preflow_chunk_is_safe(
        self,
        request: Request,
        chunk_work: float,
        candidate_remaining_work: float | None = None,
    ) -> bool:
        """Prevent a genuine mid-prefill switch away from the SJF lock."""
        locked = self._sjf_active_locked_request()
        return locked is None or locked.request_id == request.request_id

    def _preflow_candidate_key(self, request: Request) -> tuple[float, int, str]:
        request_id = request.request_id
        return (
            self._preflow_total_required_work(request),
            self._preflow_arrival_order[request_id],
            request_id,
        )

    def _preflow_select_candidate_action(
        self,
        actions: Iterable[tuple[Request, float]],
    ) -> Request | None:
        candidates = list(actions)
        locked = self._sjf_locked_request(candidates)
        if self._sjf_locked_request_id is not None:
            # The lock may temporarily be absent from this legal-action set.
            # Do not turn that admission/dependency stall into preemption.
            return locked
        return min(
            (request for request, _ in candidates),
            key=self._preflow_candidate_key,
            default=None,
        )

    def _preflow_record_scheduled_prefill(
        self,
        request_id: str,
        remaining_prefill_tokens: int,
        num_new_tokens: int,
        batch_state: Any,
    ) -> None:
        super()._preflow_record_scheduled_prefill(
            request_id,
            remaining_prefill_tokens,
            num_new_tokens,
            batch_state,
        )
        if min(remaining_prefill_tokens, num_new_tokens) > 0:
            locked = self._sjf_active_locked_request()
            if locked is not None and locked.request_id != request_id:
                raise AssertionError("SJF scheduled a different prefill while its non-preemptive lock was active")
            self._sjf_locked_request_id = request_id

    def _preflow_forget_request(self, request_id: str) -> None:
        if self._sjf_locked_request_id == request_id:
            self._sjf_locked_request_id = None
        super()._preflow_forget_request(request_id)

    def shutdown(self) -> None:
        self._sjf_locked_request_id = None
        super().shutdown()


class _SRPTPolicyMixin(_UnshieldedPrefillPolicyMixin):
    """Preemptive shortest-remaining-processing-time in triangular work."""

    def _preflow_candidate_key(self, request: Request) -> tuple[float, int, str]:
        request_id = request.request_id
        return (
            self._preflow_logical_remaining_work(request),
            self._preflow_arrival_order[request_id],
            request_id,
        )

    def _preflow_select_candidate_action(
        self,
        actions: Iterable[tuple[Request, float]],
    ) -> Request | None:
        return min(
            (request for request, _ in actions),
            key=self._preflow_candidate_key,
            default=None,
        )


class _EDFPolicyMixin(_UnshieldedPrefillPolicyMixin):
    """Pure EDF using PREFLOW's immutable FCFS-relative deadlines."""

    def _preflow_candidate_key(self, request: Request) -> tuple[float, int, str]:
        return self._preflow_edf_key(request)

    def _preflow_select_candidate_action(
        self,
        actions: Iterable[tuple[Request, float]],
    ) -> Request | None:
        return min(
            (request for request, _ in actions),
            key=self._preflow_edf_key,
            default=None,
        )


class FCFSPrefillScheduler(_FCFSPolicyMixin, PREFLOWScheduler):
    """Single-request chunked FCFS baseline."""


class SJFPrefillScheduler(_SJFPolicyMixin, PREFLOWScheduler):
    """Single-request non-preemptive triangular-SJF baseline."""


class SRPTPrefillScheduler(_SRPTPolicyMixin, PREFLOWScheduler):
    """Single-request preemptive triangular-SRPT baseline."""


class EDFPrefillScheduler(_EDFPolicyMixin, PREFLOWScheduler):
    """Single-request preemptive FCFS-relative EDF baseline."""


class AsyncFCFSPrefillScheduler(_FCFSPolicyMixin, AsyncPREFLOWScheduler):
    """Asynchronous single-request chunked FCFS baseline."""


class AsyncSJFPrefillScheduler(_SJFPolicyMixin, AsyncPREFLOWScheduler):
    """Asynchronous single-request non-preemptive triangular-SJF baseline."""


class AsyncSRPTPrefillScheduler(_SRPTPolicyMixin, AsyncPREFLOWScheduler):
    """Asynchronous single-request preemptive triangular-SRPT baseline."""


class AsyncEDFPrefillScheduler(_EDFPolicyMixin, AsyncPREFLOWScheduler):
    """Asynchronous single-request preemptive FCFS-relative EDF baseline."""


__all__ = [
    "AsyncEDFPrefillScheduler",
    "AsyncFCFSPrefillScheduler",
    "AsyncSJFPrefillScheduler",
    "AsyncSRPTPrefillScheduler",
    "EDFPrefillScheduler",
    "FCFSPrefillScheduler",
    "SJFPrefillScheduler",
    "SRPTPrefillScheduler",
]
