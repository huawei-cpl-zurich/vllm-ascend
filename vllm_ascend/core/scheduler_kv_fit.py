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
"""KV-fit scheduler: upstream vLLM scheduling with predictive KV-cache admission.

``KVFitScheduler`` inherits the upstream ``Scheduler.schedule()`` verbatim in
step 1 and adds a predictive KV-cache admission gate (``_can_admit``) in step 2.
``AsyncKVFitScheduler`` combines the admission logic with
``AsyncScheduler``'s PP decode cadence.
"""

import time

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import (
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext


class KVFitScheduler(Scheduler):
    """Scheduler with predictive KV-cache-aware admission.

    In step 1 this class inherits :meth:`schedule` verbatim from the upstream
    :class:`Scheduler` and behaves identically.  In step 2 the method is copied
    and a ``>>> KV-FIT`` admission gate is injected into the WAITING loop so
    that a request is only admitted when its predicted peak KV-cache footprint,
    together with all currently running requests, fits within the available GPU
    block budget.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        hash_block_size: int | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config,
            kv_cache_config,
            structured_output_manager,
            block_size,
            hash_block_size=hash_block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config

        init_ascend_config(vllm_config)
        cfg = get_ascend_config().kv_fit_config
        self._kv_fit_enabled: bool = cfg.enabled
        self._kv_safety_margin: float = cfg.kv_safety_margin
        self._kv_fit_log: bool = cfg.log_admission

        if self._kv_fit_enabled:
            logger.info(
                "[KVFit] Scheduler initialized: num_gpu_blocks=%d, "
                "safety_margin=%.2f, block_size=%d",
                self.cache_config.num_gpu_blocks,
                self._kv_safety_margin,
                self.cache_config.block_size,
            )

    # ------------------------------------------------------------------
    # schedule() — currently inherited verbatim from upstream Scheduler.
    # Will be copied here with ``>>> KV-FIT`` injections in step 2.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # KV admission helpers (step 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _cdiv(a: int, b: int) -> int:
        """Integer ceiling division."""
        return (a + b - 1) // b

    def _peak_kv_blocks(self, request: Request) -> int:
        """Predicted peak KV-cache blocks for *request* through its full
        lifetime (prefill + decode).

        For PD-disaggregated prefill nodes (``max_tokens == 0``) this equals
        ``ceil(prompt_len / block_size)``.  For combined P+D nodes it uses
        ``max_tokens`` as a conservative decode-length bound.
        """
        total_tokens = request.num_prompt_tokens + request.max_tokens
        return self._cdiv(total_tokens, self.cache_config.block_size)

    def _running_peak_kv_blocks(self) -> int:
        """Sum of peak KV-cache blocks for all currently running requests."""
        return sum(self._peak_kv_blocks(r) for r in self.running)

    def _can_admit(self, request: Request) -> bool:
        """Return ``True`` if *request* can be admitted without risking a
        KV-cache overflow before any running request finishes.

        This is the step-2 admission gate.  In step 1 it always returns
        ``True`` so behaviour is identical to the upstream scheduler.
        """
        # Step 1: pass-through — admission is always allowed.
        # Step 2: replace with predictive KV-cache check.
        _ = request
        return True


class AsyncKVFitScheduler(AsyncScheduler, KVFitScheduler):
    """Async-scheduling variant.

    MRO ``AsyncKVFitScheduler -> AsyncScheduler -> KVFitScheduler ->
    Scheduler`` gives it the upstream ``schedule()`` (with future KV-FIT
    admission) together with :class:`AsyncScheduler`'s PP decode cadence
    (``next_decode_eligible_step``) and async output handling.
    """
