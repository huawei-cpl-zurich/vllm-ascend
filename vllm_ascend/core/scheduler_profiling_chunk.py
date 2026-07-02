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
"""Scheduler subclass with lookup-table dynamic chunk sizing.

Compatible with vLLM v0.15.x scheduler.  When the upstream ``schedule()``
method is refactored, this override should be updated accordingly.
"""

import time

from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import (
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

from vllm_ascend.core.profiling_chunk_predictor import ProfilingChunkManager


class ProfilingChunkScheduler(Scheduler):
    """Scheduler with lookup-table dynamic chunk sizing.

    During initialization, the scheduler loads or profiles a latency lookup
    table.  During scheduling it uses a bounded beam search over the allowed
    chunk sizes to propose the next prefill chunk.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        # `hash_block_size` was added in vLLM #40946; keep it optional so the
        # subclass works on both pinned vllm and main.
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
        profiling_cfg = get_ascend_config().profiling_chunk_config
        max_chunk = getattr(profiling_cfg, "max_chunk", None)
        if max_chunk is None:
            max_chunk = self.max_num_scheduled_tokens

        self.profiling_chunk_manager = ProfilingChunkManager(
            max_model_len=self.max_model_len,
            page_size=self.cache_config.block_size,
            min_chunk=profiling_cfg.min_chunk,
            max_chunk=max_chunk,
            allowed_chunk_sizes=getattr(profiling_cfg, "allowed_chunk_sizes", None),
            profile_num_history_points=getattr(profiling_cfg, "profile_num_history_points", 33),
            profile_repeats=getattr(profiling_cfg, "profile_repeats", 3),
            search_depth=getattr(profiling_cfg, "search_depth", 6),
            beam_width=getattr(profiling_cfg, "beam_width", 16),
            profile_file=getattr(profiling_cfg, "profile_file", None),
            metadata=self._build_profile_metadata(),
        )
        self._profiling_initialized = False

        logger.info(
            "[ProfilingChunk] Scheduler initialized. page_size=%d, "
            "min_chunk=%d, max_chunk=%d, chunks=%s",
            self.cache_config.block_size,
            profiling_cfg.min_chunk,
            max_chunk,
            self.profiling_chunk_manager.allowed_chunk_sizes,
        )

    # ------------------------------------------------------------------
    # Profiling initialization
    # ------------------------------------------------------------------

    def run_profiling_chunk_init(self, model_executor) -> None:
        """Load or profile the full-forward latency lookup table.

        Called by EngineCore after model_executor is ready.
        """
        if self._profiling_initialized:
            return
        self._profiling_initialized = True

        self.profiling_chunk_manager.initialize(model_executor)
        if not self.profiling_chunk_manager.is_ready:
            logger.warning("[ProfilingChunk] Lookup-table optimizer is not ready")
            return

        logger.info("[ProfilingChunk] Lookup-table initialization completed successfully")

    def _build_profile_metadata(self) -> dict:
        parallel_config = self.vllm_config.parallel_config
        model_config = self.vllm_config.model_config
        scheduler_config = self.vllm_config.scheduler_config
        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        return {
            "model": getattr(model_config, "model", None),
            "pipeline_parallel_size": getattr(parallel_config, "pipeline_parallel_size", None),
            "tensor_parallel_size": getattr(parallel_config, "tensor_parallel_size", None),
            "prefill_context_parallel_size": getattr(
                parallel_config, "prefill_context_parallel_size", None
            ),
            "execution_mode": str(getattr(compilation_config, "cudagraph_mode", "unknown")),
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": getattr(scheduler_config, "max_num_batched_tokens", None),
            "kv_cache_block_size": self.cache_config.block_size,
        }

    def _apply_dynamic_chunk_size(
        self,
        *,
        history: int,
        remaining_prompt_tokens: int,
        num_new_tokens: int,
    ) -> int:
        if (
            self.profiling_chunk_manager is None
            or not self.profiling_chunk_manager.is_ready
            or remaining_prompt_tokens <= 1
            or num_new_tokens <= 1
        ):
            return num_new_tokens

        preferred_chunk = self.profiling_chunk_manager.select_next_chunk(
            history=history,
            remaining_prompt_tokens=remaining_prompt_tokens,
            token_budget=num_new_tokens,
            pp_size=self.vllm_config.parallel_config.pipeline_parallel_size,
        )
        if preferred_chunk is None or preferred_chunk <= 0:
            return num_new_tokens
        return min(preferred_chunk, remaining_prompt_tokens, num_new_tokens)

    # ------------------------------------------------------------------
    # schedule() override
    # ------------------------------------------------------------------
    # The method below is based on the upstream Scheduler.schedule()
    # with profiling-based chunk sizing applied to both RUNNING requests
    # (chunked prefill continuation) and WAITING requests (new prefill).
    # Modified sections are marked with ">>> PROFILING CHUNK" comments.
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:  # noqa: C901
        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            )
            remaining_prompt_tokens = num_new_tokens
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens,
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            num_new_tokens = self._apply_dynamic_chunk_size(
                history=request.num_computed_tokens,
                remaining_prompt_tokens=remaining_prompt_tokens,
                num_new_tokens=num_new_tokens,
            )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(request, num_new_tokens)

            if num_new_tokens == 0:
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        break

                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(preempted_req_id, None)
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i) for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        break

            if new_blocks is None:
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens + request.num_computed_tokens - request.num_tokens - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request_id] = spec_token_ids

                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # Try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(request.status) and not self._try_promote_blocked_waiting_request(
                    request
                ):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens = self.kv_cache_manager.get_computed_blocks(
                        request
                    )

                    if self.connector is not None:
                        ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
                            request, num_new_local_computed_tokens
                        )

                        if ext_tokens is None:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = request.num_tokens - num_new_local_computed_tokens
                        connector_prefix_cache_hits = num_external_computed_tokens

                    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens

                    if request.prefill_stats is not None:
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=num_new_local_computed_tokens,
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                    assert num_computed_tokens <= request.num_tokens
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    remaining_prompt_tokens = num_new_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    num_new_tokens = self._apply_dynamic_chunk_size(
                        history=num_computed_tokens,
                        remaining_prompt_tokens=remaining_prompt_tokens,
                        num_new_tokens=num_new_tokens,
                    )
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                effective_lookahead_tokens = 0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if self.is_encoder_decoder and request.has_encoder_inputs and encoder_inputs_to_schedule:
                    num_encoder_tokens = sum(request.get_num_encoder_embeds(i) for i in encoder_inputs_to_schedule)

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                    full_sequence_must_fit=self.scheduler_reserve_full_isl,
                )

                if new_blocks is None:
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if self.connector_prefix_cache_stats is not None and connector_prefix_cache_queries != 0:
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()
                if load_kv_async:
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    request.num_computed_tokens = num_computed_tokens
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED, scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(request_id)
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # Re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(scheduled_running_reqs) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(req, req_to_new_blocks[req.request_id].get_block_ids())
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None) if self.needs_kv_cache_zeroing else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        if self.ec_connector is not None:
            ec_meta = self.ec_connector.build_connector_meta(scheduler_output)
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
