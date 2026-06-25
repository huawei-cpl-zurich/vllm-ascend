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

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

QUEST_METADATA_BLOCK_SIZE = 128
QUEST_METADATA_HEAD_DIM = 128
QUEST_METADATA_DIM_BLOCK = 32


def _cdiv(x: int, y: int) -> int:
    triton_cdiv = getattr(triton, "cdiv", None)
    if triton_cdiv is not None:
        return triton_cdiv(x, y)
    return (x + y - 1) // y


@triton.jit(do_not_specialize=["total_tasks"])
def _quest_prefill_metadata_kernel(
    k_cache,
    block_tables,
    refresh_start_seq_lens,
    refresh_end_seq_lens,
    metadata_block_tables,
    maxblocks,
    minblocks,
    total_tasks,
    num_kv_heads: tl.constexpr,
    max_kv_blocks_per_request: tl.constexpr,
    max_metadata_blocks_per_request: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DIM_BLOCK: tl.constexpr,
    NUM_DIM_BLOCKS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    token_offsets = tl.arange(0, PAGE_SIZE)[:, None]
    dim_offsets = tl.arange(0, DIM_BLOCK)

    page_stride = PAGE_SIZE * num_kv_heads * HEAD_DIM
    head_stride = HEAD_DIM
    metadata_page_stride = num_kv_heads * HEAD_DIM

    for task_idx in range(pid, total_tasks, num_programs):
        dim_block = task_idx % NUM_DIM_BLOCKS
        tmp = task_idx // NUM_DIM_BLOCKS
        meta_block = tmp % max_metadata_blocks_per_request
        tmp = tmp // max_metadata_blocks_per_request
        head_idx = tmp % num_kv_heads
        request_idx = tmp // num_kv_heads

        start_len = tl.load(refresh_start_seq_lens + request_idx).to(tl.int32)
        end_len = tl.load(refresh_end_seq_lens + request_idx).to(tl.int32)
        if end_len <= start_len:
            continue

        start_page = start_len // PAGE_SIZE
        end_page = end_len // PAGE_SIZE
        meta_block_start_page = meta_block * PAGE_SIZE

        first_page = start_page - meta_block_start_page
        first_page = tl.maximum(first_page, 0)
        last_page = end_page - meta_block_start_page
        last_page = tl.minimum(last_page, PAGE_SIZE)
        if last_page <= first_page:
            continue

        dim_start = dim_block * DIM_BLOCK
        dims = dim_start + dim_offsets
        dim_mask = dims < HEAD_DIM

        metadata_block_id = tl.load(
            metadata_block_tables + request_idx * max_metadata_blocks_per_request + meta_block,
        ).to(tl.int32)

        for page_offset in range(first_page, last_page):
            logical_page = meta_block_start_page + page_offset
            kv_block_id = tl.load(
                block_tables + request_idx * max_kv_blocks_per_request + logical_page,
            ).to(tl.int32)

            cache_offsets = (
                kv_block_id * page_stride
                + token_offsets * metadata_page_stride
                + head_idx * head_stride
                + dims[None, :]
            )
            values = tl.load(k_cache + cache_offsets, mask=dim_mask[None, :], other=0.0).to(tl.float32)
            max_values = tl.max(values, axis=0)
            min_values = tl.min(values, axis=0)

            output_offsets = (
                metadata_block_id * page_stride
                + page_offset * metadata_page_stride
                + head_idx * head_stride
                + dims
            )
            tl.store(maxblocks + output_offsets, max_values, mask=dim_mask)
            tl.store(minblocks + output_offsets, min_values, mask=dim_mask)

        if first_page == 0:
            for page_offset in range(last_page, PAGE_SIZE):
                output_offsets = (
                    metadata_block_id * page_stride
                    + page_offset * metadata_page_stride
                    + head_idx * head_stride
                    + dims
                )
                zeros = tl.zeros((DIM_BLOCK,), dtype=tl.float32)
                tl.store(maxblocks + output_offsets, zeros, mask=dim_mask)
                tl.store(minblocks + output_offsets, zeros, mask=dim_mask)


def _validate_quest_prefill_metadata_inputs(
    k_cache: torch.Tensor,
    block_tables: torch.Tensor,
    refresh_start_seq_lens: torch.Tensor,
    refresh_end_seq_lens: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
) -> tuple[int, int, int, int]:
    if k_cache.ndim != 4:
        raise ValueError(f"k_cache must be 4D, got shape {tuple(k_cache.shape)}")
    if block_tables.ndim != 2:
        raise ValueError(f"block_tables must be 2D, got shape {tuple(block_tables.shape)}")
    if refresh_start_seq_lens.ndim != 1 or refresh_end_seq_lens.ndim != 1:
        raise ValueError("refresh_start_seq_lens and refresh_end_seq_lens must be 1D")
    if metadata_block_tables.ndim != 2:
        raise ValueError(f"metadata_block_tables must be 2D, got shape {tuple(metadata_block_tables.shape)}")
    if maxblocks.ndim != 4 or minblocks.ndim != 4:
        raise ValueError("maxblocks and minblocks must be 4D")
    if k_cache.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"quest_prefill_metadata supports float16 and bfloat16 k_cache, got {k_cache.dtype}")
    if maxblocks.dtype != k_cache.dtype or minblocks.dtype != k_cache.dtype:
        raise ValueError("maxblocks and minblocks must match k_cache dtype")
    if block_tables.dtype != torch.int32 or metadata_block_tables.dtype != torch.int32:
        raise ValueError("block_tables and metadata_block_tables must be int32")
    if refresh_start_seq_lens.dtype != torch.int32 or refresh_end_seq_lens.dtype != torch.int32:
        raise ValueError("refresh_start_seq_lens and refresh_end_seq_lens must be int32")

    batch_size = refresh_end_seq_lens.shape[0]
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    max_kv_blocks_per_request = block_tables.shape[1]
    max_metadata_blocks_per_request = metadata_block_tables.shape[1]

    if block_size != QUEST_METADATA_BLOCK_SIZE:
        raise ValueError(f"quest_prefill_metadata requires block_size == 128, got {block_size}")
    if head_dim != QUEST_METADATA_HEAD_DIM:
        raise ValueError(f"quest_prefill_metadata requires head_dim == 128, got {head_dim}")
    if refresh_start_seq_lens.shape[0] != batch_size:
        raise ValueError("refresh_start_seq_lens and refresh_end_seq_lens batch size mismatch")
    if block_tables.shape[0] != batch_size or metadata_block_tables.shape[0] != batch_size:
        raise ValueError("block table batch dimensions must match refresh seq lens")
    if maxblocks.shape[1:] != (block_size, num_kv_heads, head_dim):
        raise ValueError("maxblocks must have shape [num_blocks, 128, num_kv_heads, 128]")
    if minblocks.shape[1:] != (block_size, num_kv_heads, head_dim):
        raise ValueError("minblocks must have shape [num_blocks, 128, num_kv_heads, 128]")

    tensors = (
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
    )
    if any(tensor.device != k_cache.device for tensor in tensors):
        raise ValueError("quest_prefill_metadata tensors must all be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("quest_prefill_metadata expects contiguous tensors")

    return batch_size, num_kv_heads, max_kv_blocks_per_request, max_metadata_blocks_per_request


def quest_prefill_metadata_triton(
    k_cache: torch.Tensor,
    block_tables: torch.Tensor,
    refresh_start_seq_lens: torch.Tensor,
    refresh_end_seq_lens: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
) -> None:
    (
        batch_size,
        num_kv_heads,
        max_kv_blocks_per_request,
        max_metadata_blocks_per_request,
    ) = _validate_quest_prefill_metadata_inputs(
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
    )

    if batch_size == 0 or num_kv_heads == 0 or max_metadata_blocks_per_request == 0:
        return

    num_dim_blocks = _cdiv(QUEST_METADATA_HEAD_DIM, QUEST_METADATA_DIM_BLOCK)
    total_tasks = batch_size * num_kv_heads * max_metadata_blocks_per_request * num_dim_blocks
    num_programs = min(get_vectorcore_num(), total_tasks)

    _quest_prefill_metadata_kernel[(num_programs,)](
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
        total_tasks,
        num_kv_heads=num_kv_heads,
        max_kv_blocks_per_request=max_kv_blocks_per_request,
        max_metadata_blocks_per_request=max_metadata_blocks_per_request,
        PAGE_SIZE=QUEST_METADATA_BLOCK_SIZE,
        HEAD_DIM=QUEST_METADATA_HEAD_DIM,
        DIM_BLOCK=QUEST_METADATA_DIM_BLOCK,
        NUM_DIM_BLOCKS=num_dim_blocks,
    )
