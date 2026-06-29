import argparse
import gc

import torch

from vllm_ascend.utils import enable_custom_op


BLOCK_SIZE = 128
HEAD_DIM = 128


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def make_case(dtype: torch.dtype, pages: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_meta_blocks = max(1, ceil_div(pages, BLOCK_SIZE))
    query = torch.randn((1, 1, HEAD_DIM), generator=generator, dtype=dtype)
    maxblocks = torch.randn(
        (num_meta_blocks, BLOCK_SIZE, 1, HEAD_DIM),
        generator=generator,
        dtype=dtype,
    )
    minblocks = torch.randn(
        (num_meta_blocks, BLOCK_SIZE, 1, HEAD_DIM),
        generator=generator,
        dtype=dtype,
    )
    metadata_block_tables = torch.arange(num_meta_blocks, dtype=torch.int32).reshape(1, num_meta_blocks)
    seq_len = pages * BLOCK_SIZE - 3 if pages > 0 else 0
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)
    return query, maxblocks, minblocks, metadata_block_tables, seq_lens


def cpu_reference(
    query: torch.Tensor,
    maxblocks: torch.Tensor,
    minblocks: torch.Tensor,
    metadata_block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    k: int,
    tokens_since_metadata_update: int,
) -> torch.Tensor:
    selected = torch.zeros((query.size(0), query.size(1), k), dtype=torch.int32)
    use_anchors = tokens_since_metadata_update >= 0

    for batch_idx in range(query.size(0)):
        seq_len = int(seq_lens[batch_idx].item())
        valid_pages = ceil_div(seq_len, BLOCK_SIZE) if seq_len > 0 else 0
        if valid_pages <= 0:
            continue
        if k >= valid_pages:
            selected[batch_idx, :, :valid_pages] = torch.arange(valid_pages, dtype=torch.int32)
            continue

        query_vec = query[batch_idx, 0].to(torch.float32)
        scores = []
        for page_idx in range(valid_pages):
            meta_block = page_idx // BLOCK_SIZE
            page_offset = page_idx % BLOCK_SIZE
            meta_block_id = int(metadata_block_tables[batch_idx, meta_block].item())
            max_vec = maxblocks[meta_block_id, page_offset, 0].to(torch.float32)
            min_vec = minblocks[meta_block_id, page_offset, 0].to(torch.float32)
            score = torch.maximum(query_vec * max_vec, query_vec * min_vec).sum()
            scores.append(float(score.item()))

        if use_anchors:
            scores[0] = float("inf")
            scores[valid_pages - 1] = float("inf")

        ranked = sorted(range(valid_pages), key=lambda page_idx: (-scores[page_idx], page_idx))
        selected[batch_idx, 0] = torch.tensor(ranked[:k], dtype=torch.int32)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--pages", type=int, default=14)
    parser.add_argument("--tokens-since-metadata-update", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()

    enable_custom_op()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    query, maxblocks, minblocks, metadata_block_tables, seq_lens = make_case(dtype, args.pages, args.seed)
    expected = cpu_reference(
        query,
        maxblocks,
        minblocks,
        metadata_block_tables,
        seq_lens,
        args.k,
        args.tokens_since_metadata_update,
    )

    query_npu = query.npu()
    maxblocks_npu = maxblocks.npu()
    minblocks_npu = minblocks.npu()
    metadata_block_tables_npu = metadata_block_tables.npu()
    seq_lens_npu = seq_lens.npu()

    actual = torch.ops._C_ascend.npu_quest_block_select_paged(
        query_npu,
        maxblocks_npu,
        minblocks_npu,
        metadata_block_tables_npu,
        seq_lens_npu,
        args.k,
        args.tokens_since_metadata_update,
    )
    actual_out = torch.empty((1, 1, args.k), dtype=torch.int32, device=query_npu.device)
    torch.ops._C_ascend.npu_quest_block_select_paged_out(
        query_npu,
        maxblocks_npu,
        minblocks_npu,
        metadata_block_tables_npu,
        seq_lens_npu,
        actual_out,
        args.tokens_since_metadata_update,
    )

    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    actual_out_cpu = actual_out.cpu()
    print("seq_lens:", seq_lens.tolist())
    print("expected:", expected.tolist())
    print("actual:", actual_cpu.tolist())
    print("actual_out:", actual_out_cpu.tolist())
    print("alloc_matches:", torch.equal(actual_cpu, expected))
    print("out_matches:", torch.equal(actual_out_cpu, expected))

    del query_npu, maxblocks_npu, minblocks_npu, metadata_block_tables_npu, seq_lens_npu
    del actual, actual_out
    gc.collect()
    torch.npu.empty_cache()


if __name__ == "__main__":
    main()
