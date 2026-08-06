#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Longest-prefix-match scheduler for prefix-cache-heavy prefill workloads.

This mirrors SGLang's LPM policy at vLLM's waiting-queue boundary: before the
scheduler admits a waiting request, sort the FCFS waiting queue by each
request's current local prefix-cache hit length.  Once a request from a prefix
group warms the KV cache, its siblings naturally move to the front.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class _QueuedRequest:
    request: Request
    fcfs_order: int


class LPMRequestQueue(RequestQueue):
    """FCFS-compatible queue with lazy LPM ordering.

    The queue keeps ordinary FCFS order as a stable tie breaker.  Setting
    ``max_lpm_queue_size`` to 0 disables the SGLang-style large-queue fallback.
    """

    def __init__(self, scheduler: Scheduler, max_lpm_queue_size: int = 128) -> None:
        self._scheduler = scheduler
        self._max_lpm_queue_size = max_lpm_queue_size
        self._items: list[_QueuedRequest] = []
        self._next_order = 0
        self._next_prepend_order = -1
        self._dirty = False

    def add_request(self, request: Request) -> None:
        self._items.append(_QueuedRequest(request, self._next_order))
        self._next_order += 1
        self._dirty = True

    def pop_request(self) -> Request:
        if not self._items:
            raise IndexError("pop from empty LPMRequestQueue")
        self._ensure_ordered()
        item = self._items.pop(0)
        self._dirty = bool(self._items)
        return item.request

    def peek_request(self) -> Request:
        if not self._items:
            raise IndexError("peek from empty LPMRequestQueue")
        self._ensure_ordered()
        return self._items[0].request

    def prepend_request(self, request: Request) -> None:
        self._items.insert(0, _QueuedRequest(request, self._next_prepend_order))
        self._next_prepend_order -= 1
        self._dirty = True

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        for idx, item in enumerate(self._items):
            if item.request is request:
                del self._items[idx]
                self._dirty = True
                return
        raise ValueError("request not in LPMRequestQueue")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._items = [
            item for item in self._items if item.request not in requests_to_remove
        ]
        self._dirty = True

    def __bool__(self) -> bool:
        return bool(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Request]:
        self._ensure_ordered()
        for item in self._items:
            yield item.request

    def _ensure_ordered(self) -> None:
        if not self._dirty:
            return
        self._dirty = False
        if self._max_lpm_queue_size and len(self._items) > self._max_lpm_queue_size:
            self._items.sort(key=lambda item: item.fcfs_order)
            return
        self._items.sort(
            key=lambda item: (
                -self._matched_prefix_tokens(item.request),
                item.fcfs_order,
            )
        )

    def _matched_prefix_tokens(self, request: Request) -> int:
        if request.num_computed_tokens > 0:
            return request.num_computed_tokens
        if not self._scheduler.cache_config.enable_prefix_caching:
            return 0
        if request.skip_reading_prefix_cache:
            return 0

        max_cache_hit_length = request.num_tokens - 1
        if max_cache_hit_length <= 0:
            return 0

        coordinator = self._scheduler.kv_cache_manager.coordinator
        if (
            self._scheduler.connector is not None
            and self._scheduler.has_mamba_layers
            and isinstance(coordinator, HybridKVCacheCoordinator)
        ):
            _, per_group_hits = coordinator.find_longest_cache_hit_per_group(
                request.block_hashes,
                max_cache_hit_length,
            )
            return max(per_group_hits)

        _, matched_tokens = coordinator.find_longest_cache_hit(
            request.block_hashes,
            max_cache_hit_length,
        )
        return matched_tokens


def install_lpm_waiting_queue(
    scheduler: Scheduler,
    *,
    max_lpm_queue_size: int = 128,
) -> LPMRequestQueue:
    """Install the LPM waiting queue on a newly constructed FCFS scheduler."""
    if scheduler.policy != SchedulingPolicy.FCFS:
        raise ValueError(f"LPM scheduling requires FCFS scheduling policy, got {scheduler.policy!s}")

    waiting = scheduler.waiting
    if isinstance(waiting, LPMRequestQueue):
        return waiting
    if waiting:
        raise RuntimeError("LPM waiting queue must be installed before request admission.")

    queue = LPMRequestQueue(scheduler, max_lpm_queue_size=max_lpm_queue_size)
    scheduler.waiting = queue
    scheduler.skipped_waiting = create_request_queue(SchedulingPolicy.FCFS)
    logger.info("LPM waiting queue installed: max_lpm_queue_size=%d", max_lpm_queue_size)
    return queue


class LPMScheduler(Scheduler):
    """Scheduler that admits waiting requests by longest local prefix-cache hit."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        from vllm_ascend.ascend_config import init_ascend_config

        lpm_config = init_ascend_config(self.vllm_config).scheduler_config.lpm_scheduler_config
        install_lpm_waiting_queue(
            self,
            max_lpm_queue_size=lpm_config.max_lpm_queue_size,
        )


class AsyncLPMScheduler(AsyncScheduler, LPMScheduler):
    """LPM scheduler composed with vLLM's async scheduler."""

    pass
