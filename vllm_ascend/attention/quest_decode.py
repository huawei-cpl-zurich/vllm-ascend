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
    refresh_seq_lens: torch.Tensor | None = None
    ready: bool = False


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

        self.refresh_seq_lens_cpu[:num_reqs].fill(0)
        for row_idx, req_id in enumerate(req_ids[:num_reqs]):
            seq_len = int(seq_lens_cpu[row_idx])
            if (
                self.owner_req_ids[row_idx] != req_id
                or self.valid_tokens[row_idx] > seq_len
                or seq_len // QUEST_PAGE_SIZE > self.valid_tokens[row_idx] // QUEST_PAGE_SIZE
            ):
                self.refresh_seq_lens_cpu[row_idx] = seq_len

        self.refresh_seq_lens[:num_reqs].copy_(
            self.refresh_seq_lens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )
        return QuestPreparedMetadata(
            metadata_block_tables=self.metadata_block_tables[:num_reqs],
            refresh_seq_lens=self.refresh_seq_lens[:num_reqs],
            ready=True,
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


class QuestDecodeMetadataManager:
    """Model-level QUEST metadata policy and per-layer state."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        ascend_config: AscendConfig,
        model_config: Any,
        max_encoder_len: int | None,
        max_num_reqs: int,
        device: torch.device,
        use_sparse: bool,
    ) -> None:
        self.vllm_config = vllm_config
        self.ascend_config = ascend_config
        self.model_config = model_config
        self.max_encoder_len = max_encoder_len
        self.max_num_reqs = max_num_reqs
        self.device = device
        self.use_sparse = use_sparse

        self.enabled = ascend_config.quest_decode_config.enable
        self.topk_pages = ascend_config.quest_decode_config.topk_pages
        self.model_supported = self.enabled
        self.max_num_metadata_blocks_per_req = 0
        self.layer_tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._configure()

    def _disable(self, reason: str) -> None:
        if self.enabled:
            logger.warning_once(f"QUEST decode is disabled: {reason}")
        self.model_supported = False
        self.max_num_metadata_blocks_per_req = 0
        self.layer_tensors.clear()

    def _configure(self) -> None:
        if not self.enabled:
            self.model_supported = False
            return

        if get_ascend_device_type() not in {AscendDeviceType.A2, AscendDeviceType.A3}:
            self._disable(
                "current hardware is unsupported, QUEST decode currently supports only "
                "Ascend A2/A3 (ascend910b/ascend910_93)."
            )
            return

        if self.vllm_config.kv_transfer_config is not None:
            self._disable("kv_transfer_config is set, but QUEST decode requires a local KV cache.")
            return
        if enable_cp():
            self._disable("context parallel is enabled, but QUEST decode requires unsharded request metadata.")
            return
        if self.ascend_config.xlite_graph_config.enabled:
            self._disable(
                "xLite graph execution is enabled, but QUEST decode only supports the standard v1 decode path."
            )
            return
        if self.model_config.use_mla:
            self._disable("MLA attention is enabled, but QUEST decode only supports standard v1 attention.")
            return
        if self.use_sparse:
            self._disable("sparse attention is enabled, but QUEST decode only supports standard v1 attention.")
            return

        quest_max_model_len = max(self.model_config.max_model_len, self.max_encoder_len or 0)
        self.max_num_metadata_blocks_per_req = cdiv(cdiv(quest_max_model_len, QUEST_PAGE_SIZE), QUEST_PAGE_SIZE)
        if self.max_num_metadata_blocks_per_req > QUEST_MAX_METADATA_BLOCKS_PER_REQ:
            self._disable(
                "the configured max_model_len requires more metadata blocks per request "
                f"({self.max_num_metadata_blocks_per_req}) than the kernel limit "
                f"({QUEST_MAX_METADATA_BLOCKS_PER_REQ})."
            )

    def initialize_layer_tensors(
        self,
        kv_caches: dict[str, Any],
        shared_kv_cache_layers: dict[str, str],
    ) -> None:
        self.layer_tensors.clear()
        if not self.model_supported or self.max_num_metadata_blocks_per_req == 0:
            return

        attn_layers = get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase)
        num_metadata_blocks = self.max_num_reqs * self.max_num_metadata_blocks_per_req
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

            metadata_shape = (num_metadata_blocks, QUEST_PAGE_SIZE, k_cache.shape[2], QUEST_HEAD_SIZE)
            maxblocks = torch.zeros(metadata_shape, dtype=k_cache.dtype, device=self.device)
            minblocks = torch.zeros_like(maxblocks)
            self.layer_tensors[layer_name] = (maxblocks, minblocks)

        for layer_name, target_layer_name in shared_kv_cache_layers.items():
            if layer_name not in attn_layers:
                continue
            if target_layer_name in self.layer_tensors:
                self.layer_tensors[layer_name] = self.layer_tensors[target_layer_name]
            else:
                self._disable(
                    f"attention layer {layer_name} shares KV cache with {target_layer_name}, "
                    "but the target layer is not QUEST-compatible."
                )
                return

        if not self.layer_tensors:
            self._disable("no compatible v1 attention layers were found with block_size 128 and head_size 128.")

    def prepare_common_metadata(
        self,
        *,
        input_batch: Any,
        num_reqs: int,
        seq_lens_cpu: torch.Tensor,
        attn_state: Any,
        max_query_len: int,
    ) -> QuestPreparedMetadata:
        quest_batch_metadata = getattr(input_batch, "quest_metadata", None)
        if (
            not self.model_supported
            or getattr(attn_state, "name", None) != "DecodeOnly"
            or max_query_len != 1
            or quest_batch_metadata is None
        ):
            return QuestPreparedMetadata()
        return quest_batch_metadata.prepare(num_reqs, input_batch.req_ids, seq_lens_cpu)

    def with_layer_tensors(self, attn_metadata: Any, layer_name: str) -> Any:
        layer_tensors = self.layer_tensors.get(layer_name)
        if layer_tensors is None:
            return attn_metadata

        attn_metadata_layer = copy(attn_metadata)
        attn_metadata_layer.quest_maxblocks = layer_tensors[0]
        attn_metadata_layer.quest_minblocks = layer_tensors[1]
        return attn_metadata_layer

    def commit_batch_metadata(self, input_batch: Any, num_reqs: int) -> None:
        quest_batch_metadata = getattr(input_batch, "quest_metadata", None)
        if not self.model_supported or quest_batch_metadata is None:
            return
        quest_batch_metadata.commit(num_reqs, input_batch.req_ids)
