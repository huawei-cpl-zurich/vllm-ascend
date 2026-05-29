#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv

from vllm_ascend.ascend_config import AscendConfig, QuestDecodeConfig
from vllm_ascend.attention.utils import enable_cp
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

QUEST_PAGE_SIZE = 128
QUEST_HEAD_SIZE = 128
QUEST_MAX_METADATA_BLOCKS_PER_REQ = 6
QUEST_INDEX_ALIGNMENT = 8


def get_quest_decode_config(vllm_config: VllmConfig) -> QuestDecodeConfig:
    """Resolve QUEST config without requiring the global AscendConfig singleton."""
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, Mapping):
        return QuestDecodeConfig()
    return QuestDecodeConfig(additional_config.get("quest_decode_config"))


@dataclass
class QuestPreparedMetadata:
    metadata_block_tables: torch.Tensor | None = None
    refresh_start_seq_lens: torch.Tensor | None = None
    refresh_seq_lens: torch.Tensor | None = None
    ready: bool = False
    refresh_required: bool = False


@dataclass(frozen=True)
class QuestLayerTensors:
    maxblocks: torch.Tensor
    minblocks: torch.Tensor


class QuestBatchMetadataState:
    """Batch/request-row QUEST metadata state owned by NPUInputBatch."""

    def __init__(
        self,
        max_num_reqs: int,
        max_num_metadata_blocks_per_req: int,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        total_metadata_blocks = max_num_reqs * max_num_metadata_blocks_per_req
        self.metadata_block_tables = torch.arange(
            total_metadata_blocks,
            dtype=torch.int32,
            device=device,
        ).view(max_num_reqs, max_num_metadata_blocks_per_req)
        self.owner_req_ids: list[str | None] = [None] * max_num_reqs
        self.valid_tokens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.valid_tokens = self.valid_tokens_cpu_tensor.numpy()
        self.refresh_start_seq_lens = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
        self.refresh_start_seq_lens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.refresh_start_seq_lens_cpu = self.refresh_start_seq_lens_cpu_tensor.numpy()
        self.refresh_seq_lens = torch.zeros((max_num_reqs,), dtype=torch.int32, device=device)
        self.refresh_seq_lens_cpu_tensor = torch.zeros(
            (max_num_reqs,),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self.refresh_seq_lens_cpu = self.refresh_seq_lens_cpu_tensor.numpy()

    def prepare(
        self,
        num_reqs: int,
        req_ids: Sequence[str | None],
        seq_lens_cpu: torch.Tensor | np.ndarray,
    ) -> QuestPreparedMetadata:
        if num_reqs <= 0:
            return QuestPreparedMetadata(
                metadata_block_tables=self.metadata_block_tables[:0],
                ready=True,
            )

        self.refresh_start_seq_lens_cpu[:num_reqs].fill(0)
        self.refresh_seq_lens_cpu[:num_reqs].fill(0)
        refresh_required = False
        for row_idx, req_id in enumerate(req_ids[:num_reqs]):
            seq_len = int(seq_lens_cpu[row_idx])
            valid_tokens = int(self.valid_tokens[row_idx])
            new_owner_or_shrunk = self.owner_req_ids[row_idx] != req_id or valid_tokens > seq_len
            crossed_page_boundary = seq_len // QUEST_PAGE_SIZE > valid_tokens // QUEST_PAGE_SIZE
            if new_owner_or_shrunk or crossed_page_boundary:
                if not new_owner_or_shrunk:
                    self.refresh_start_seq_lens_cpu[row_idx] = (valid_tokens // QUEST_PAGE_SIZE) * QUEST_PAGE_SIZE
                self.refresh_seq_lens_cpu[row_idx] = seq_len
                if self.refresh_seq_lens_cpu[row_idx] > self.refresh_start_seq_lens_cpu[row_idx]:
                    refresh_required = True

        refresh_start_seq_lens = None
        refresh_seq_lens = None
        if refresh_required:
            self.refresh_start_seq_lens[:num_reqs].copy_(
                self.refresh_start_seq_lens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )
            self.refresh_seq_lens[:num_reqs].copy_(
                self.refresh_seq_lens_cpu_tensor[:num_reqs],
                non_blocking=True,
            )
            refresh_start_seq_lens = self.refresh_start_seq_lens[:num_reqs]
            refresh_seq_lens = self.refresh_seq_lens[:num_reqs]

        return QuestPreparedMetadata(
            metadata_block_tables=self.metadata_block_tables[:num_reqs],
            refresh_start_seq_lens=refresh_start_seq_lens,
            refresh_seq_lens=refresh_seq_lens,
            ready=True,
            refresh_required=refresh_required,
        )

    def commit(self, num_reqs: int, req_ids: Sequence[str | None]) -> None:
        if num_reqs <= 0:
            return

        for row_idx, req_id in enumerate(req_ids[:num_reqs]):
            refreshed_seq_len = int(self.refresh_seq_lens_cpu[row_idx])
            if refreshed_seq_len <= 0:
                continue
            self.owner_req_ids[row_idx] = req_id
            self.valid_tokens[row_idx] = refreshed_seq_len


def _clear_layer_tensors(attn_layers: Mapping[str, AttentionLayerBase]) -> None:
    for attn_layer in attn_layers.values():
        impl = getattr(attn_layer, "impl", None)
        if impl is not None and hasattr(impl, "quest_layer_tensors"):
            impl.quest_layer_tensors = None


def _clear_metadata(input_batch: Any, attn_layers: Mapping[str, AttentionLayerBase]) -> None:
    clear_quest_metadata = getattr(input_batch, "clear_quest_metadata", None)
    if clear_quest_metadata is not None:
        clear_quest_metadata()
    _clear_layer_tensors(attn_layers)


def _disable(
    reason: str,
    input_batch: Any,
    attn_layers: Mapping[str, AttentionLayerBase],
) -> None:
    logger.warning_once(f"QUEST decode is disabled: {reason}")
    _clear_metadata(input_batch, attn_layers)


def _get_max_num_metadata_blocks_per_req(model_config: Any, max_encoder_len: int | None) -> int:
    quest_max_model_len = max(model_config.max_model_len, max_encoder_len or 0)
    return cdiv(cdiv(quest_max_model_len, QUEST_PAGE_SIZE), QUEST_PAGE_SIZE)


def initialize_metadata(
    *,
    vllm_config: VllmConfig,
    ascend_config: AscendConfig,
    model_config: Any,
    max_encoder_len: int | None,
    max_num_reqs: int,
    device: torch.device,
    use_sparse: bool,
    input_batch: Any,
    kv_caches: dict[str, Any],
    shared_kv_cache_layers: dict[str, str],
) -> None:
    """Validate and allocate all QUEST metadata for a loaded model."""
    attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    _clear_metadata(input_batch, attn_layers)

    if not ascend_config.quest_decode_config.enable:
        return

    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    if cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs():
        _disable(
            "full graph execution is enabled, but QUEST decode currently requires "
            "runtime switching between dense and sparse attention paths.",
            input_batch,
            attn_layers,
        )
        return

    if get_ascend_device_type() not in {AscendDeviceType.A2, AscendDeviceType.A3}:
        _disable(
            "current hardware is unsupported, QUEST decode currently supports only "
            "Ascend A2/A3 (ascend910b/ascend910_93).",
            input_batch,
            attn_layers,
        )
        return

    if vllm_config.kv_transfer_config is not None:
        _disable(
            "kv_transfer_config is set, but QUEST decode requires a local KV cache.",
            input_batch,
            attn_layers,
        )
        return
    if enable_cp():
        _disable(
            "context parallel is enabled, but QUEST decode requires unsharded request metadata.",
            input_batch,
            attn_layers,
        )
        return
    if ascend_config.xlite_graph_config.enabled:
        _disable(
            "xLite graph execution is enabled, but QUEST decode only supports the standard v1 decode path.",
            input_batch,
            attn_layers,
        )
        return
    if model_config.use_mla:
        _disable(
            "MLA attention is enabled, but QUEST decode only supports standard v1 attention.",
            input_batch,
            attn_layers,
        )
        return
    if use_sparse:
        _disable(
            "sparse attention is enabled, but QUEST decode only supports standard v1 attention.",
            input_batch,
            attn_layers,
        )
        return

    max_num_metadata_blocks_per_req = _get_max_num_metadata_blocks_per_req(model_config, max_encoder_len)
    if max_num_metadata_blocks_per_req > QUEST_MAX_METADATA_BLOCKS_PER_REQ:
        _disable(
            "the configured max_model_len requires more metadata blocks per request "
            f"({max_num_metadata_blocks_per_req}) than the kernel limit "
            f"({QUEST_MAX_METADATA_BLOCKS_PER_REQ}).",
            input_batch,
            attn_layers,
        )
        return

    base_layer_k_caches: dict[str, torch.Tensor] = {}
    for layer_name, attn_layer in attn_layers.items():
        if layer_name in shared_kv_cache_layers:
            continue
        if layer_name not in kv_caches:
            _disable(
                f"attention layer {layer_name} does not have a local KV cache.",
                input_batch,
                attn_layers,
            )
            return

        impl = getattr(attn_layer, "impl", None)
        if not getattr(impl, "quest_layer_supported", False):
            _disable(f"attention layer {layer_name} is not QUEST-compatible.", input_batch, attn_layers)
            return

        kv_cache = kv_caches[layer_name]
        if not isinstance(kv_cache, tuple) or len(kv_cache) < 2:
            _disable(
                f"attention layer {layer_name} does not expose a standard KV cache tuple.",
                input_batch,
                attn_layers,
            )
            return

        k_cache = kv_cache[0]
        if not isinstance(k_cache, torch.Tensor) or k_cache.ndim != 4:
            _disable(f"attention layer {layer_name} has an unsupported key-cache layout.", input_batch, attn_layers)
            return
        if k_cache.shape[1] != QUEST_PAGE_SIZE or k_cache.shape[-1] != QUEST_HEAD_SIZE:
            _disable(
                f"attention layer {layer_name} has block_size={k_cache.shape[1]} and "
                f"head_size={k_cache.shape[-1]}, but QUEST requires block_size={QUEST_PAGE_SIZE} "
                f"and head_size={QUEST_HEAD_SIZE}.",
                input_batch,
                attn_layers,
            )
            return

        base_layer_k_caches[layer_name] = k_cache

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        if layer_name not in attn_layers:
            continue
        if target_layer_name not in base_layer_k_caches:
            _disable(
                f"attention layer {layer_name} shares KV cache with {target_layer_name}, "
                "but the target layer is not QUEST-compatible.",
                input_batch,
                attn_layers,
            )
            return

    if not base_layer_k_caches:
        _disable(
            "no compatible v1 attention layers were found with block_size 128 and head_size 128.",
            input_batch,
            attn_layers,
        )
        return

    input_batch.init_quest_metadata(max_num_metadata_blocks_per_req)
    num_metadata_blocks = max_num_reqs * max_num_metadata_blocks_per_req
    layer_tensors: dict[str, QuestLayerTensors] = {}
    for layer_name, k_cache in base_layer_k_caches.items():
        metadata_shape = (num_metadata_blocks, QUEST_PAGE_SIZE, k_cache.shape[2], QUEST_HEAD_SIZE)
        maxblocks = torch.zeros(metadata_shape, dtype=k_cache.dtype, device=device)
        layer_tensors[layer_name] = QuestLayerTensors(
            maxblocks=maxblocks,
            minblocks=torch.zeros_like(maxblocks),
        )
        attn_layers[layer_name].impl.quest_layer_tensors = layer_tensors[layer_name]

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        if layer_name not in attn_layers:
            continue
        attn_layers[layer_name].impl.quest_layer_tensors = layer_tensors[target_layer_name]


def attach_layer_tensors(attn_metadata: Any, impl: Any) -> Any:
    layer_tensors = getattr(impl, "quest_layer_tensors", None)
    if layer_tensors is None:
        return attn_metadata

    attn_metadata_layer = copy(attn_metadata)
    attn_metadata_layer.quest_maxblocks = layer_tensors.maxblocks
    attn_metadata_layer.quest_minblocks = layer_tensors.minblocks
    return attn_metadata_layer


def commit_batch_metadata(input_batch: Any, num_reqs: int) -> None:
    quest_batch_metadata = getattr(input_batch, "quest_metadata", None)
    if quest_batch_metadata is None:
        return
    quest_batch_metadata.commit(num_reqs, input_batch.req_ids)
