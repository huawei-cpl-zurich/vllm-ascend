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
"""Scheduler instrumentation for KV-cache preemption experiments.

``KVMetricsScheduler`` intentionally delegates scheduling to upstream vLLM's
``Scheduler`` without changing admission, ordering, or preemption behavior.
It only records preemptions and KV-cache page usage after each schedule step.
"""

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.structured_output import StructuredOutputManager

from vllm_ascend.ascend_config import get_ascend_config, init_ascend_config


class KVSchedulerMetricsMixin:
    """Shared preemption/KV-page counters for scheduler experiments."""

    def _init_kv_scheduler_metrics(self, vllm_config: VllmConfig) -> None:
        init_ascend_config(vllm_config)
        cfg = get_ascend_config().kv_scheduler_metrics_config
        self._kv_metrics_enabled: bool = cfg.enabled
        self._kv_metrics_log_interval: int = cfg.log_interval
        self._kv_metrics_label: str = cfg.label
        self._kv_metrics_total_preemptions: int = 0
        self._kv_metrics_peak_used_pages: int = 0
        self._kv_metrics_steps: int = 0
        if self._kv_metrics_enabled:
            logger.info(
                "[KVSchedulerMetrics][%s] initialized: "
                "max_num_scheduled_tokens=%d max_num_batched_tokens=%d "
                "max_num_seqs=%d long_prefill_token_threshold=%d "
                "scheduler_reserve_full_isl=%s",
                self._kv_metrics_label,
                self.max_num_scheduled_tokens,
                self.scheduler_config.max_num_batched_tokens,
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.long_prefill_token_threshold,
                self.scheduler_reserve_full_isl,
            )

    def _kv_page_counts(self) -> tuple[int, int, int, float]:
        block_pool = self.kv_cache_manager.block_pool
        # Exclude vLLM's permanently allocated null block so page counts match
        # real usable KV capacity.
        total_pages = max(block_pool.num_gpu_blocks - 1, 0)
        free_pages = block_pool.get_num_free_blocks()
        used_pages = max(total_pages - free_pages, 0)
        usage = block_pool.get_usage()
        return used_pages, free_pages, total_pages, usage

    def _record_kv_scheduler_metrics(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        if not getattr(self, "_kv_metrics_enabled", False):
            return

        preemptions = len(scheduler_output.preempted_req_ids or ())
        self._kv_metrics_total_preemptions += preemptions
        self._kv_metrics_steps += 1

        used_pages, free_pages, total_pages, usage = self._kv_page_counts()
        self._kv_metrics_peak_used_pages = max(
            self._kv_metrics_peak_used_pages,
            used_pages,
        )

        if self._kv_metrics_steps % self._kv_metrics_log_interval != 0:
            return

        logger.info(
            "[KVSchedulerMetrics][%s] step=%d preemptions=%d "
            "total_preemptions=%d used_kv_pages=%d free_kv_pages=%d "
            "total_kv_pages=%d kv_usage=%.6f peak_used_kv_pages=%d "
            "running=%d waiting=%d skipped_waiting=%d scheduled_tokens=%d",
            self._kv_metrics_label,
            self.current_step,
            preemptions,
            self._kv_metrics_total_preemptions,
            used_pages,
            free_pages,
            total_pages,
            usage,
            self._kv_metrics_peak_used_pages,
            len(self.running),
            len(self.waiting),
            len(self.skipped_waiting),
            scheduler_output.total_num_scheduled_tokens,
        )


class KVMetricsScheduler(KVSchedulerMetricsMixin, Scheduler):
    """Upstream vLLM scheduler with KV metrics only.

    This class must not change scheduling behavior.  Keep ``schedule`` as a
    delegate to ``Scheduler.schedule`` followed by instrumentation.
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
        self._init_kv_scheduler_metrics(vllm_config)

    def schedule(self) -> SchedulerOutput:
        scheduler_output = super().schedule()
        self._record_kv_scheduler_metrics(scheduler_output)
        return scheduler_output


class AsyncKVMetricsScheduler(AsyncScheduler, KVMetricsScheduler):
    """Async upstream scheduler with KV metrics only."""
