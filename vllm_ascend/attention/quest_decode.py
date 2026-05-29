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
class QuestBatchPreparedMetadata:
    metadata_block_tables: torch.Tensor | None = None
    ready: bool = False
    batch_size: int = 0


@dataclass
class QuestLayerPreparedMetadata:
    layer_metadata: Any = None
    metadata_block_tables: torch.Tensor | None = None
    refresh_start_seq_lens: torch.Tensor | None = None
    refresh_seq_lens: torch.Tensor | None = None
    maxblocks: torch.Tensor | None = None
    minblocks: torch.Tensor | None = None
    ready: bool = False
    refresh_required: bool = False
    batch_size: int = 0


class QuestLayerMetadata:
    """Per-layer QUEST metadata tensors and freshness bookkeeping."""

    def __init__(
        self,
        *,
        layer_name: str,
        max_num_reqs: int,
        maxblocks: torch.Tensor,
        minblocks: torch.Tensor,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        self.layer_name = layer_name
        self.maxblocks = maxblocks
        self.minblocks = minblocks
        self.valid_owner_generations = np.full(max_num_reqs, -1, dtype=np.int64)
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

    def invalidate_rows(self, row_indices: Sequence[int]) -> None:
        for row_idx in row_indices:
            self.valid_owner_generations[row_idx] = -1
            self.valid_tokens[row_idx] = 0

    def prepare(
        self,
        manager: "QuestDecodeMetadataManager",
    ) -> QuestLayerPreparedMetadata:
        num_reqs = manager.active_num_reqs
        metadata_block_tables = manager.active_metadata_block_tables
        if num_reqs <= 0:
            return QuestLayerPreparedMetadata(
                layer_metadata=self,
                metadata_block_tables=metadata_block_tables,
                maxblocks=self.maxblocks,
                minblocks=self.minblocks,
                ready=True,
                batch_size=0,
            )

        self.refresh_start_seq_lens_cpu[:num_reqs].fill(0)
        self.refresh_seq_lens_cpu[:num_reqs].fill(0)
        refresh_required = False
        for row_idx in range(num_reqs):
            seq_len = manager.active_seq_lens_cpu[row_idx]
            valid_tokens = int(self.valid_tokens[row_idx])
            stale_owner = self.valid_owner_generations[row_idx] != manager.owner_generations[row_idx]
            new_owner_or_shrunk = stale_owner or valid_tokens > seq_len
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

        return QuestLayerPreparedMetadata(
            layer_metadata=self,
            metadata_block_tables=metadata_block_tables,
            refresh_start_seq_lens=refresh_start_seq_lens,
            refresh_seq_lens=refresh_seq_lens,
            maxblocks=self.maxblocks,
            minblocks=self.minblocks,
            ready=True,
            refresh_required=refresh_required,
            batch_size=num_reqs,
        )

    def commit(self, manager: "QuestDecodeMetadataManager", num_reqs: int) -> None:
        if num_reqs <= 0:
            return

        for row_idx in range(num_reqs):
            refreshed_seq_len = int(self.refresh_seq_lens_cpu[row_idx])
            if refreshed_seq_len <= 0:
                continue
            self.valid_tokens[row_idx] = refreshed_seq_len
            self.valid_owner_generations[row_idx] = manager.owner_generations[row_idx]


class QuestDecodeMetadataManager:
    """Central QUEST metadata owner for batch rows and per-layer tensors."""

    def __init__(
        self,
        *,
        max_num_reqs: int,
        device: torch.device,
        pin_memory: bool,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.pin_memory = pin_memory
        self.metadata_block_tables: torch.Tensor | None = None
        self.max_num_metadata_blocks_per_req = 0
        self.owner_req_ids: list[str | None] = [None] * max_num_reqs
        self.owner_generations = np.zeros(max_num_reqs, dtype=np.int64)
        self.layers: dict[str, QuestLayerMetadata] = {}
        self.ready = False
        self.active_num_reqs = 0
        self.active_seq_lens_cpu: tuple[int, ...] = ()
        self.active_metadata_block_tables: torch.Tensor | None = None
        self._owner_signature: tuple[int, tuple[str | None, ...]] | None = None
        self._active_ready = False

    def clear(self) -> None:
        self.metadata_block_tables = None
        self.max_num_metadata_blocks_per_req = 0
        self.owner_req_ids = [None] * self.max_num_reqs
        self.owner_generations.fill(0)
        self.layers.clear()
        self.ready = False
        self.active_num_reqs = 0
        self.active_seq_lens_cpu = ()
        self.active_metadata_block_tables = None
        self._owner_signature = None
        self._active_ready = False

    def _disable(self, reason: str) -> None:
        logger.warning_once(f"QUEST decode is disabled: {reason}")
        self.clear()

    def _invalidate_rows(self, row_indices: Sequence[int]) -> None:
        if not row_indices:
            return
        seen_layers: set[int] = set()
        for layer_metadata in self.layers.values():
            layer_id = id(layer_metadata)
            if layer_id in seen_layers:
                continue
            seen_layers.add(layer_id)
            layer_metadata.invalidate_rows(row_indices)

    def initialize(
        self,
        *,
        vllm_config: VllmConfig,
        ascend_config: AscendConfig,
        model_config: Any,
        max_encoder_len: int | None,
        max_num_reqs: int,
        device: torch.device,
        use_sparse: bool,
        kv_caches: dict[str, Any],
        shared_kv_cache_layers: dict[str, str],
    ) -> None:
        """Validate and allocate all QUEST metadata for a loaded model."""
        self.clear()
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.owner_req_ids = [None] * max_num_reqs
        self.owner_generations = np.zeros(max_num_reqs, dtype=np.int64)

        attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)

        if not ascend_config.quest_decode_config.enable:
            return

        cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
        if cudagraph_mode is not None and cudagraph_mode.has_full_cudagraphs():
            self._disable(
                "full graph execution is enabled, but QUEST decode currently requires "
                "runtime switching between dense and sparse attention paths."
            )
            return

        if get_ascend_device_type() not in {AscendDeviceType.A2, AscendDeviceType.A3}:
            self._disable(
                "current hardware is unsupported, QUEST decode currently supports only "
                "Ascend A2/A3 (ascend910b/ascend910_93)."
            )
            return

        if vllm_config.kv_transfer_config is not None:
            self._disable("kv_transfer_config is set, but QUEST decode requires a local KV cache.")
            return
        if enable_cp():
            self._disable("context parallel is enabled, but QUEST decode requires unsharded request metadata.")
            return
        if ascend_config.xlite_graph_config.enabled:
            self._disable(
                "xLite graph execution is enabled, but QUEST decode only supports the standard v1 decode path."
            )
            return
        if model_config.use_mla:
            self._disable("MLA attention is enabled, but QUEST decode only supports standard v1 attention.")
            return
        if use_sparse:
            self._disable("sparse attention is enabled, but QUEST decode only supports standard v1 attention.")
            return

        max_num_metadata_blocks_per_req = _get_max_num_metadata_blocks_per_req(model_config, max_encoder_len)
        if max_num_metadata_blocks_per_req > QUEST_MAX_METADATA_BLOCKS_PER_REQ:
            self._disable(
                "the configured max_model_len requires more metadata blocks per request "
                f"({max_num_metadata_blocks_per_req}) than the kernel limit "
                f"({QUEST_MAX_METADATA_BLOCKS_PER_REQ})."
            )
            return

        base_layer_k_caches: dict[str, torch.Tensor] = {}
        for layer_name, attn_layer in attn_layers.items():
            if layer_name in shared_kv_cache_layers:
                continue
            if layer_name not in kv_caches:
                self._disable(f"attention layer {layer_name} does not have a local KV cache.")
                return

            impl = getattr(attn_layer, "impl", None)
            if not getattr(impl, "quest_layer_supported", False):
                self._disable(f"attention layer {layer_name} is not QUEST-compatible.")
                return

            kv_cache = kv_caches[layer_name]
            if not isinstance(kv_cache, tuple) or len(kv_cache) < 2:
                self._disable(f"attention layer {layer_name} does not expose a standard KV cache tuple.")
                return

            k_cache = kv_cache[0]
            if not isinstance(k_cache, torch.Tensor) or k_cache.ndim != 4:
                self._disable(f"attention layer {layer_name} has an unsupported key-cache layout.")
                return
            if k_cache.shape[1] != QUEST_PAGE_SIZE or k_cache.shape[-1] != QUEST_HEAD_SIZE:
                self._disable(
                    f"attention layer {layer_name} has block_size={k_cache.shape[1]} and "
                    f"head_size={k_cache.shape[-1]}, but QUEST requires block_size={QUEST_PAGE_SIZE} "
                    f"and head_size={QUEST_HEAD_SIZE}."
                )
                return

            base_layer_k_caches[layer_name] = k_cache

        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            if layer_name not in attn_layers:
                continue
            if target_layer_name not in base_layer_k_caches:
                self._disable(
                    f"attention layer {layer_name} shares KV cache with {target_layer_name}, "
                    "but the target layer is not QUEST-compatible."
                )
                return

        if not base_layer_k_caches:
            self._disable("no compatible v1 attention layers were found with block_size 128 and head_size 128.")
            return

        self.max_num_metadata_blocks_per_req = max_num_metadata_blocks_per_req
        num_metadata_blocks = max_num_reqs * max_num_metadata_blocks_per_req
        self.metadata_block_tables = torch.arange(
            num_metadata_blocks,
            dtype=torch.int32,
            device=device,
        ).view(max_num_reqs, max_num_metadata_blocks_per_req)

        base_layer_metadata: dict[str, QuestLayerMetadata] = {}
        for layer_name, k_cache in base_layer_k_caches.items():
            metadata_shape = (num_metadata_blocks, QUEST_PAGE_SIZE, k_cache.shape[2], QUEST_HEAD_SIZE)
            maxblocks = torch.zeros(metadata_shape, dtype=k_cache.dtype, device=device)
            layer_metadata = QuestLayerMetadata(
                layer_name=layer_name,
                max_num_reqs=max_num_reqs,
                maxblocks=maxblocks,
                minblocks=torch.zeros_like(maxblocks),
                device=device,
                pin_memory=self.pin_memory,
            )
            base_layer_metadata[layer_name] = layer_metadata
            self.layers[layer_name] = layer_metadata

        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            if layer_name not in attn_layers:
                continue
            self.layers[layer_name] = base_layer_metadata[target_layer_name]

        self.ready = True

    def prepare_batch(
        self,
        *,
        num_reqs: int,
        req_ids: Sequence[str | None] | None,
        seq_lens_cpu: torch.Tensor | np.ndarray,
        attn_state: Any,
        max_query_len: int | None,
    ) -> QuestBatchPreparedMetadata:
        self.active_num_reqs = 0
        self.active_seq_lens_cpu = ()
        self.active_metadata_block_tables = None
        self._active_ready = False

        if (
            not self.ready
            or self.metadata_block_tables is None
            or req_ids is None
            or getattr(attn_state, "name", None) != "DecodeOnly"
            or max_query_len != 1
        ):
            self._owner_signature = None
            return QuestBatchPreparedMetadata()

        num_reqs = min(num_reqs, len(req_ids), self.max_num_reqs)
        req_ids_tuple = tuple(req_ids[:num_reqs])
        seq_lens_tuple = tuple(int(seq_lens_cpu[row_idx]) for row_idx in range(num_reqs))
        owner_signature = (num_reqs, req_ids_tuple)

        if owner_signature != self._owner_signature:
            changed_rows: list[int] = []
            for row_idx, req_id in enumerate(req_ids_tuple):
                if self.owner_req_ids[row_idx] != req_id:
                    self.owner_req_ids[row_idx] = req_id
                    self.owner_generations[row_idx] += 1
                    changed_rows.append(row_idx)
            for row_idx in range(num_reqs, self.max_num_reqs):
                if self.owner_req_ids[row_idx] is not None:
                    self.owner_req_ids[row_idx] = None
                    self.owner_generations[row_idx] += 1
                    changed_rows.append(row_idx)
            self._invalidate_rows(changed_rows)
            self._owner_signature = owner_signature

        self.active_num_reqs = num_reqs
        self.active_seq_lens_cpu = seq_lens_tuple
        self.active_metadata_block_tables = self.metadata_block_tables[:num_reqs]
        self._active_ready = True
        return QuestBatchPreparedMetadata(
            metadata_block_tables=self.active_metadata_block_tables,
            ready=True,
            batch_size=num_reqs,
        )

    def get_layer(self, layer_name: str) -> QuestLayerPreparedMetadata:
        if not self._active_ready:
            return QuestLayerPreparedMetadata()
        layer_metadata = self.layers.get(layer_name)
        if layer_metadata is None:
            return QuestLayerPreparedMetadata()
        return QuestLayerPreparedMetadata(
            layer_metadata=layer_metadata,
            metadata_block_tables=self.active_metadata_block_tables,
            maxblocks=layer_metadata.maxblocks,
            minblocks=layer_metadata.minblocks,
            ready=True,
            batch_size=self.active_num_reqs,
        )


def _get_max_num_metadata_blocks_per_req(model_config: Any, max_encoder_len: int | None) -> int:
    quest_max_model_len = max(model_config.max_model_len, max_encoder_len or 0)
    return cdiv(cdiv(quest_max_model_len, QUEST_PAGE_SIZE), QUEST_PAGE_SIZE)


def attach_layer_metadata(attn_metadata: Any, layer_name: str) -> Any:
    manager = getattr(attn_metadata, "quest_manager", None)
    if manager is None:
        return attn_metadata

    prepared_metadata = manager.get_layer(layer_name)
    if not prepared_metadata.ready:
        return attn_metadata

    attn_metadata_layer = copy(attn_metadata)
    attn_metadata_layer.quest_layer_metadata = prepared_metadata.layer_metadata
    attn_metadata_layer.quest_metadata_block_tables = prepared_metadata.metadata_block_tables
    attn_metadata_layer.quest_refresh_start_seq_lens = prepared_metadata.refresh_start_seq_lens
    attn_metadata_layer.quest_refresh_seq_lens = prepared_metadata.refresh_seq_lens
    attn_metadata_layer.quest_refresh_required = prepared_metadata.refresh_required
    attn_metadata_layer.quest_maxblocks = prepared_metadata.maxblocks
    attn_metadata_layer.quest_minblocks = prepared_metadata.minblocks
    attn_metadata_layer.quest_ready = prepared_metadata.ready
    return attn_metadata_layer


def prepare_attached_layer_metadata(attn_metadata: Any) -> Any:
    manager = getattr(attn_metadata, "quest_manager", None)
    layer_metadata = getattr(attn_metadata, "quest_layer_metadata", None)
    if manager is None or layer_metadata is None:
        return attn_metadata

    prepared_metadata = layer_metadata.prepare(manager)
    if not prepared_metadata.ready:
        return attn_metadata

    attn_metadata.quest_metadata_block_tables = prepared_metadata.metadata_block_tables
    attn_metadata.quest_refresh_start_seq_lens = prepared_metadata.refresh_start_seq_lens
    attn_metadata.quest_refresh_seq_lens = prepared_metadata.refresh_seq_lens
    attn_metadata.quest_refresh_required = prepared_metadata.refresh_required
    attn_metadata.quest_maxblocks = prepared_metadata.maxblocks
    attn_metadata.quest_minblocks = prepared_metadata.minblocks
    attn_metadata.quest_ready = prepared_metadata.ready
    return attn_metadata
