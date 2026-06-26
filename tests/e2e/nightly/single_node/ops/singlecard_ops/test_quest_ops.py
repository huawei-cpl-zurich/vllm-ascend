import gc

import pytest
import torch

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

BLOCK_SIZE = 128
HEAD_DIM = 128


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _cleanup_npu():
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


def cpu_quest_prefill_metadata(
    k_cache: torch.Tensor,
    block_tables: torch.Tensor,
    refresh_start_seq_lens: torch.Tensor,
    refresh_end_seq_lens: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_maxblocks = maxblocks.clone()
    expected_minblocks = minblocks.clone()
    batch_size = refresh_end_seq_lens.size(0)
    num_kv_heads = k_cache.size(2)

    for batch_idx in range(batch_size):
        start_len = int(refresh_start_seq_lens[batch_idx].item())
        end_len = int(refresh_end_seq_lens[batch_idx].item())
        if end_len <= start_len:
            continue

        start_page = start_len // BLOCK_SIZE
        end_page = end_len // BLOCK_SIZE
        start_meta_block = start_page // BLOCK_SIZE
        end_meta_block = _ceil_div(end_page, BLOCK_SIZE)

        for meta_block in range(start_meta_block, end_meta_block):
            meta_block_start_page = meta_block * BLOCK_SIZE
            first_page = max(start_page - meta_block_start_page, 0)
            last_page = min(end_page - meta_block_start_page, BLOCK_SIZE)
            pages_to_refresh = last_page - first_page
            if pages_to_refresh <= 0:
                continue

            metadata_block_id = int(metadata_block_tables[batch_idx, meta_block].item())
            for page_offset in range(first_page, last_page):
                logical_page = meta_block_start_page + page_offset
                kv_block_id = int(block_tables[batch_idx, logical_page].item())
                for kv_head_idx in range(num_kv_heads):
                    k_block = k_cache[kv_block_id, :, kv_head_idx, :].to(torch.float32)
                    expected_maxblocks[metadata_block_id, page_offset, kv_head_idx, :] = (
                        k_block.max(dim=0).values.to(maxblocks.dtype)
                    )
                    expected_minblocks[metadata_block_id, page_offset, kv_head_idx, :] = (
                        k_block.min(dim=0).values.to(minblocks.dtype)
                    )

            if first_page == 0 and last_page < BLOCK_SIZE:
                expected_maxblocks[metadata_block_id, pages_to_refresh:, :, :] = 0
                expected_minblocks[metadata_block_id, pages_to_refresh:, :, :] = 0

    return expected_maxblocks, expected_minblocks


def cpu_quest_block_select_paged(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    tokens_since_metadata_update: int,
) -> torch.Tensor:
    batch_size, num_heads, _ = query.shape
    num_kv_heads = maxblocks.size(2)
    query_heads_per_kv_head = num_heads // num_kv_heads
    selected_indices = torch.zeros((batch_size, num_heads, k), dtype=torch.int32)
    use_fixed_anchors = tokens_since_metadata_update >= 0

    for batch_idx in range(batch_size):
        seq_len = int(seq_lens[batch_idx].item())
        valid_page_count = _ceil_div(seq_len, BLOCK_SIZE) if seq_len > 0 else 0
        if valid_page_count <= 0:
            continue

        if k >= valid_page_count:
            selected_indices[batch_idx, :, :valid_page_count] = torch.arange(
                valid_page_count, dtype=torch.int32
            )
            continue

        for query_head_idx in range(num_heads):
            kv_head_idx = query_head_idx // query_heads_per_kv_head
            scores: list[float] = []
            query_vec = query[batch_idx, query_head_idx, :].to(torch.float32)
            for page_idx in range(valid_page_count):
                meta_block = page_idx // BLOCK_SIZE
                page_offset = page_idx % BLOCK_SIZE
                metadata_block_id = int(metadata_block_tables[batch_idx, meta_block].item())
                max_vec = maxblocks[metadata_block_id, page_offset, kv_head_idx, :].to(torch.float32)
                min_vec = minblocks[metadata_block_id, page_offset, kv_head_idx, :].to(torch.float32)
                scores.append(float(torch.maximum(query_vec * max_vec, query_vec * min_vec).sum().item()))

            if use_fixed_anchors:
                scores[0] = float("inf")
                scores[valid_page_count - 1] = float("inf")

            ranked_pages = sorted(range(valid_page_count), key=lambda page_idx: (-scores[page_idx], page_idx))
            selected_indices[batch_idx, query_head_idx, :] = torch.tensor(ranked_pages[:k], dtype=torch.int32)

    return selected_indices


def assert_block_select_indices_equal(
    actual_indices: torch.Tensor,
    expected_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    tokens_since_metadata_update: int,
) -> None:
    if tokens_since_metadata_update < 0:
        torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
        return

    batch_size, num_heads, _ = actual_indices.shape
    for batch_idx in range(batch_size):
        seq_len = int(seq_lens[batch_idx].item())
        valid_page_count = _ceil_div(seq_len, BLOCK_SIZE) if seq_len > 0 else 0
        if valid_page_count <= 0 or k >= valid_page_count:
            torch.testing.assert_close(
                actual_indices[batch_idx],
                expected_indices[batch_idx],
                rtol=0,
                atol=0,
            )
            continue

        anchors = {0, valid_page_count - 1}
        for head_idx in range(num_heads):
            actual_row = actual_indices[batch_idx, head_idx].tolist()
            expected_row = expected_indices[batch_idx, head_idx].tolist()
            actual_selected = actual_row[:k]
            expected_selected = expected_row[:k]

            for anchor in anchors:
                if anchor not in actual_selected:
                    pytest.fail(
                        f"Missing fixed anchor {anchor} in selected pages: "
                        f"batch={batch_idx}, head={head_idx}, actual={actual_selected}"
                    )
                if anchor not in expected_selected:
                    pytest.fail(
                        f"CPU reference did not select fixed anchor {anchor}: "
                        f"batch={batch_idx}, head={head_idx}, expected={expected_selected}"
                    )

            actual_without_anchors = [page for page in actual_selected if page not in anchors]
            expected_without_anchors = [page for page in expected_selected if page not in anchors]
            if actual_without_anchors != expected_without_anchors:
                pytest.fail(
                    "Non-anchor selected pages differ after removing fixed anchors: "
                    f"batch={batch_idx}, head={head_idx}, "
                    f"actual={actual_without_anchors}, expected={expected_without_anchors}"
                )


def _make_prefill_metadata_case(dtype: torch.dtype):
    batch_size = 2
    num_kv_heads = 2
    max_pages = 136
    num_kv_blocks = 384
    num_metadata_blocks = 4
    generator = torch.Generator(device="cpu").manual_seed(123)

    k_cache = torch.randn(
        (num_kv_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)
    block_tables = torch.empty((batch_size, max_pages), dtype=torch.int32)
    block_tables[0, :] = torch.arange(10, 10 + max_pages, dtype=torch.int32)
    block_tables[1, :] = torch.arange(200, 200 + max_pages, dtype=torch.int32)
    refresh_start_seq_lens = torch.tensor([0, 3 * BLOCK_SIZE + 17], dtype=torch.int32)
    refresh_end_seq_lens = torch.tensor([130 * BLOCK_SIZE + 63, 132 * BLOCK_SIZE + 11], dtype=torch.int32)
    metadata_block_tables = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    maxblocks = torch.full(
        (num_metadata_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM),
        -7,
        dtype=dtype,
    )
    minblocks = torch.full(
        (num_metadata_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM),
        7,
        dtype=dtype,
    )
    return (
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
    )


def _make_block_select_case(dtype: torch.dtype):
    batch_size = 3
    num_heads = 4
    num_kv_heads = 2
    num_metadata_blocks = 6
    generator = torch.Generator(device="cpu").manual_seed(456)

    query = torch.randn((batch_size, num_heads, HEAD_DIM), generator=generator, dtype=torch.float32).to(dtype)
    maxblocks = torch.randn(
        (num_metadata_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)
    minblocks = torch.randn(
        (num_metadata_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype)
    metadata_block_tables = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    seq_lens = torch.tensor([0, 5 * BLOCK_SIZE - 3, 130 * BLOCK_SIZE + 7], dtype=torch.int32)
    return query, maxblocks, minblocks, metadata_block_tables, seq_lens


def ascendc_prefill_metadata_exec(
    k_cache: torch.Tensor,
    block_tables: torch.Tensor,
    refresh_start_seq_lens: torch.Tensor,
    refresh_end_seq_lens: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    maxblocks_npu = maxblocks.npu()
    minblocks_npu = minblocks.npu()
    torch.ops._C_ascend.npu_quest_prefill_metadata(
        k_cache.npu(),
        block_tables.npu(),
        refresh_start_seq_lens.npu(),
        refresh_end_seq_lens.npu(),
        metadata_block_tables.npu(),
        maxblocks_npu,
        minblocks_npu,
    )
    return maxblocks_npu.cpu(), minblocks_npu.cpu()


def ascendc_block_select_exec(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    tokens_since_metadata_update: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_npu = query.npu()
    maxblocks_npu = maxblocks.npu()
    minblocks_npu = minblocks.npu()
    metadata_block_tables_npu = metadata_block_tables.npu()
    seq_lens_npu = seq_lens.npu()

    selected_indices = torch.ops._C_ascend.npu_quest_block_select_paged(
        query_npu,
        maxblocks_npu,
        minblocks_npu,
        metadata_block_tables_npu,
        seq_lens_npu,
        k,
        tokens_since_metadata_update,
    )
    selected_indices_out = torch.empty((query.size(0), query.size(1), k), dtype=torch.int32, device=query_npu.device)
    torch.ops._C_ascend.npu_quest_block_select_paged_out(
        query_npu,
        maxblocks_npu,
        minblocks_npu,
        metadata_block_tables_npu,
        seq_lens_npu,
        selected_indices_out,
        tokens_since_metadata_update,
    )
    return selected_indices.cpu(), selected_indices_out.cpu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_npu_quest_prefill_metadata(dtype):
    (
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
    ) = _make_prefill_metadata_case(dtype)
    expected_maxblocks, expected_minblocks = cpu_quest_prefill_metadata(
        k_cache,
        block_tables,
        refresh_start_seq_lens,
        refresh_end_seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
    )

    try:
        actual_maxblocks, actual_minblocks = ascendc_prefill_metadata_exec(
            k_cache,
            block_tables,
            refresh_start_seq_lens,
            refresh_end_seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
        )

        torch.testing.assert_close(actual_maxblocks, expected_maxblocks, rtol=0, atol=0)
        torch.testing.assert_close(actual_minblocks, expected_minblocks, rtol=0, atol=0)
    finally:
        _cleanup_npu()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("k", [8, 9, 13, 16])
@pytest.mark.parametrize("tokens_since_metadata_update", [-1, 0])
def test_npu_quest_block_select_paged(dtype, k, tokens_since_metadata_update):
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = _make_block_select_case(dtype)
    expected_indices = cpu_quest_block_select_paged(
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        k,
        tokens_since_metadata_update,
    )

    try:
        actual_indices, actual_indices_out = ascendc_block_select_exec(
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            k,
            tokens_since_metadata_update,
        )

        assert_block_select_indices_equal(
            actual_indices,
            expected_indices,
            seq_lens,
            k,
            tokens_since_metadata_update,
        )
        assert_block_select_indices_equal(
            actual_indices_out,
            expected_indices,
            seq_lens,
            k,
            tokens_since_metadata_update,
        )
    finally:
        _cleanup_npu()
