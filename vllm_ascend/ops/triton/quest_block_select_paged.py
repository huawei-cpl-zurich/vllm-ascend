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

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_aicore_num, init_device_properties_triton

QUEST_BLOCK_SELECT_BLOCK_SIZE = 128
QUEST_BLOCK_SELECT_HEAD_DIM = 128
QUEST_BLOCK_SELECT_MAX_MMBPR = 6
QUEST_BLOCK_SELECT_MAX_SELECTED_BLOCKS = 64
QUEST_BLOCK_SELECT_SCORE_PAGE_TILE = 16


def _cdiv(x: int, y: int) -> int:
    triton_cdiv = getattr(triton, "cdiv", None)
    if triton_cdiv is not None:
        return triton_cdiv(x, y)
    return (x + y - 1) // y


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


@triton.jit
def _quest_score_pages_kernel(
    query,
    maxblocks,
    minblocks,
    metadata_block_tables,
    seq_lens,
    page_scores,
    num_score_tiles,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    max_metadata_blocks_per_request: tl.constexpr,
    max_pages: tl.constexpr,
    num_page_tiles: tl.constexpr,
    USE_FIXED_ANCHORS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_TILE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    for score_tile_idx in range(pid, num_score_tiles, num_programs):
        page_tile_idx = score_tile_idx % num_page_tiles
        batch_head_idx = score_tile_idx // num_page_tiles
        batch_idx = batch_head_idx // num_heads
        query_head_idx = batch_head_idx - batch_idx * num_heads

        page_offsets = page_tile_idx * PAGE_TILE + tl.arange(0, PAGE_TILE)
        page_mask = page_offsets < max_pages
        score_offsets = batch_head_idx * max_pages + page_offsets

        seq_len = tl.load(seq_lens + batch_idx).to(tl.int32)
        valid_page_count = tl.where(seq_len > 0, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE, 0)
        valid_page_count = tl.minimum(valid_page_count, max_pages)
        valid_page_mask = page_mask & (page_offsets < valid_page_count)

        query_heads_per_kv_head = num_heads // num_kv_heads
        kv_head_idx = query_head_idx // query_heads_per_kv_head
        meta_blocks = page_offsets // BLOCK_SIZE
        page_offsets_in_block = page_offsets - meta_blocks * BLOCK_SIZE

        meta_block_ids = tl.load(
            metadata_block_tables + batch_idx * max_metadata_blocks_per_request + meta_blocks,
            mask=page_mask,
            other=0,
        ).to(tl.int32)

        dim_offsets = tl.arange(0, HEAD_DIM)
        query_offset = batch_head_idx * HEAD_DIM + dim_offsets
        metadata_stride = BLOCK_SIZE * num_kv_heads * HEAD_DIM
        metadata_offset = (
            meta_block_ids[:, None] * metadata_stride
            + page_offsets_in_block[:, None] * num_kv_heads * HEAD_DIM
            + kv_head_idx * HEAD_DIM
            + dim_offsets[None, :]
        )

        query_values = tl.load(query + query_offset).to(tl.float32)
        max_values = tl.load(maxblocks + metadata_offset, mask=valid_page_mask[:, None], other=0.0).to(tl.float32)
        min_values = tl.load(minblocks + metadata_offset, mask=valid_page_mask[:, None], other=0.0).to(tl.float32)

        max_products = query_values[None, :] * max_values
        min_products = query_values[None, :] * min_values
        scores = tl.sum(tl.maximum(max_products, min_products), axis=1)

        if USE_FIXED_ANCHORS:
            is_anchor = (page_offsets == 0) | (page_offsets == valid_page_count - 1)
            scores = tl.where(is_anchor, float("inf"), scores)

        scores = tl.where(valid_page_mask, scores, -float("inf"))
        tl.store(page_scores + score_offsets, scores, mask=page_mask)


@triton.jit
def _quest_select_topk_kernel(
    page_scores,
    seq_lens,
    selected_indices,
    num_batch_heads,
    num_heads: tl.constexpr,
    k: tl.constexpr,
    k_pad: tl.constexpr,
    max_pages: tl.constexpr,
    pages_pad: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    rank_offsets = tl.arange(0, k_pad)
    rank_mask = rank_offsets < k

    page_offsets = tl.arange(0, pages_pad)
    page_mask = page_offsets < max_pages

    for batch_head_idx in range(pid, num_batch_heads, num_programs):
        batch_idx = batch_head_idx // num_heads

        seq_len = tl.load(seq_lens + batch_idx).to(tl.int32)
        valid_page_count = tl.where(seq_len > 0, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE, 0)
        valid_page_count = tl.minimum(valid_page_count, max_pages)

        out_base = batch_head_idx * k

        if valid_page_count <= 0:
            tl.store(selected_indices + out_base + rank_offsets, 0, mask=rank_mask)
        else:
            if k >= valid_page_count:
                sequential = tl.where(rank_offsets < valid_page_count, rank_offsets, 0)
                tl.store(selected_indices + out_base + rank_offsets, sequential.to(tl.int32), mask=rank_mask)
            else:
                scores = tl.load(
                    page_scores + batch_head_idx * max_pages + page_offsets,
                    mask=page_mask,
                    other=-float("inf"),
                )
                scores = tl.where(page_offsets < valid_page_count, scores, -float("inf"))

                for out_idx in range(0, k):
                    best_page = tl.argmax(scores, axis=0).to(tl.int32)
                    tl.store(selected_indices + out_base + out_idx, best_page)
                    scores = tl.where(page_offsets == best_page, -float("inf"), scores)


def _validate_quest_block_select_paged_inputs(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    tokens_since_metadata_update: int,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 3:
        raise ValueError(f"query must be 3D, got shape {tuple(query.shape)}")
    if maxblocks.ndim != 4 or minblocks.ndim != 4:
        raise ValueError("maxblocks and minblocks must be 4D")
    if metadata_block_tables.ndim != 2:
        raise ValueError(f"metadata_block_tables must be 2D, got shape {tuple(metadata_block_tables.shape)}")
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be 1D, got shape {tuple(seq_lens.shape)}")
    if output.ndim != 3:
        raise ValueError(f"selected_indices must be 3D, got shape {tuple(output.shape)}")

    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"quest_block_select_paged supports float16 and bfloat16 query tensors, got {query.dtype}")
    if maxblocks.dtype != query.dtype or minblocks.dtype != query.dtype:
        raise ValueError("query, maxblocks, and minblocks must share the same dtype")
    if metadata_block_tables.dtype != torch.int32 or seq_lens.dtype != torch.int32:
        raise ValueError("metadata_block_tables and seq_lens must be int32")
    if output.dtype != torch.int32:
        raise ValueError(f"selected_indices must be int32, got {output.dtype}")

    batch_size = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    block_size = maxblocks.shape[1]
    num_kv_heads = maxblocks.shape[2]
    output_k = output.shape[2]
    max_metadata_blocks_per_request = metadata_block_tables.shape[1]

    if head_dim != QUEST_BLOCK_SELECT_HEAD_DIM:
        raise ValueError(f"quest_block_select_paged requires head_dim == 128, got {head_dim}")
    if block_size != QUEST_BLOCK_SELECT_BLOCK_SIZE:
        raise ValueError(f"quest_block_select_paged requires block_size == 128, got {block_size}")
    if maxblocks.shape[3] != head_dim:
        raise ValueError("maxblocks last dimension must match query head_dim")
    if maxblocks.shape != minblocks.shape:
        raise ValueError("maxblocks and minblocks must have matching shapes")
    if metadata_block_tables.shape[0] != batch_size or seq_lens.shape[0] != batch_size:
        raise ValueError("metadata_block_tables and seq_lens batch dimensions must match query")
    if output.shape[0] != batch_size or output.shape[1] != num_heads:
        raise ValueError("selected_indices must have shape [batch_size, num_heads, k]")
    if output_k <= 0:
        raise ValueError("k must be positive")
    if output_k > QUEST_BLOCK_SELECT_MAX_SELECTED_BLOCKS:
        raise ValueError(
            "quest_block_select_paged supports at most "
            f"{QUEST_BLOCK_SELECT_MAX_SELECTED_BLOCKS} selected blocks, got {output_k}"
        )
    if num_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_heads and num_kv_heads must be positive")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if max_metadata_blocks_per_request > QUEST_BLOCK_SELECT_MAX_MMBPR:
        raise ValueError(
            "metadata_block_tables.shape[1] cannot exceed "
            f"{QUEST_BLOCK_SELECT_MAX_MMBPR}, got {max_metadata_blocks_per_request}"
        )
    if max_metadata_blocks_per_request <= 0:
        raise ValueError("metadata_block_tables.shape[1] must be positive")
    if tokens_since_metadata_update != -1 and not (0 <= tokens_since_metadata_update <= block_size):
        raise ValueError("tokens_since_metadata_update must be -1 or in [0, block_size]")
    if tokens_since_metadata_update != -1 and output_k < 2:
        raise ValueError("quest_block_select_paged requires k >= 2 when fixed anchors are enabled")

    tensors = (query, maxblocks, minblocks, metadata_block_tables, seq_lens, output)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("quest_block_select_paged tensors must all be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("quest_block_select_paged expects contiguous tensors")

    return batch_size, num_heads, num_kv_heads, output_k, max_metadata_blocks_per_request


def quest_block_select_paged_out_triton(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    output: torch.Tensor,
    tokens_since_metadata_update: int = -1,
) -> torch.Tensor:
    (
        batch_size,
        num_heads,
        num_kv_heads,
        k,
        max_metadata_blocks_per_request,
    ) = _validate_quest_block_select_paged_inputs(
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        output,
        tokens_since_metadata_update,
    )

    max_pages = max_metadata_blocks_per_request * QUEST_BLOCK_SELECT_BLOCK_SIZE
    pages_pad = _next_power_of_2(max_pages)
    k_pad = _next_power_of_2(k)
    num_batch_heads = batch_size * num_heads
    if num_batch_heads == 0:
        return output
    score_page_tile = min(QUEST_BLOCK_SELECT_SCORE_PAGE_TILE, _next_power_of_2(max_pages))
    num_page_tiles = _cdiv(max_pages, score_page_tile)
    num_score_tiles = num_batch_heads * num_page_tiles
    init_device_properties_triton()
    num_cores = get_aicore_num()
    num_score_programs = min(num_score_tiles, num_cores)
    num_select_programs = min(num_batch_heads, num_cores)

    page_scores = torch.empty(
        (num_batch_heads, max_pages),
        dtype=torch.float32,
        device=query.device,
    )

    _quest_score_pages_kernel[(num_score_programs,)](
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        page_scores,
        num_score_tiles,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        max_metadata_blocks_per_request=max_metadata_blocks_per_request,
        max_pages=max_pages,
        num_page_tiles=num_page_tiles,
        USE_FIXED_ANCHORS=tokens_since_metadata_update >= 0,
        BLOCK_SIZE=QUEST_BLOCK_SELECT_BLOCK_SIZE,
        HEAD_DIM=QUEST_BLOCK_SELECT_HEAD_DIM,
        PAGE_TILE=score_page_tile,
    )
    _quest_select_topk_kernel[(num_select_programs,)](
        page_scores,
        seq_lens,
        output,
        num_batch_heads,
        num_heads=num_heads,
        k=k,
        k_pad=k_pad,
        max_pages=max_pages,
        pages_pad=pages_pad,
        BLOCK_SIZE=QUEST_BLOCK_SELECT_BLOCK_SIZE,
    )
    return output


def quest_block_select_paged_triton(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    tokens_since_metadata_update: int = -1,
) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be positive")
    if k > QUEST_BLOCK_SELECT_MAX_SELECTED_BLOCKS:
        raise ValueError(
            "quest_block_select_paged supports at most "
            f"{QUEST_BLOCK_SELECT_MAX_SELECTED_BLOCKS} selected blocks, got {k}"
        )
    output = torch.empty(
        (query.shape[0], query.shape[1], k),
        dtype=torch.int32,
        device=query.device,
    )
    return quest_block_select_paged_out_triton(
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        output,
        tokens_since_metadata_update,
    )
