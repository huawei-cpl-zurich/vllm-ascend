/**
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

#include "kernel_operator.h"
#include "quest_block_select_paged_tilingkey.h"

#define BYTES_UB_BLOCK 32
#define BYTES_DATA_BLOCK 32
#define NUM_HALF_ELEMS_PER_VECTOR 128
#define NUM_FLOAT_ELEMS_PER_VECTOR 64
#define DIV_ROUNDUP(x, y) (((x) + (y)-1) / (y))
#define DIV_ROUNDUP_MUL(bytes, bytes_per_block) (DIV_ROUNDUP(bytes, bytes_per_block) * (bytes_per_block))
#define NUM_UB_BYTES(bytes) (DIV_ROUNDUP_MUL(bytes, BYTES_UB_BLOCK))
#define NUM_DATA_BLOCKS(bytes) (DIV_ROUNDUP(bytes, BYTES_DATA_BLOCK))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MINHALF -65504.0f
#define MAXHALF 65504.0f
#define MINFLOAT -3.4028235e38f
#define MAXFLOAT 3.4028235e+38f

constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_V200 = 8;
constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_HALF_V220 = 4;
constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 = 2;

using namespace AscendC;

// QuestBlockSelectPagedTilingData is generated from the op_host tiling
// definition. The kernel must not redeclare it locally.

__aicore__ inline void quest_emit_unique_index(
    LocalTensor<uint32_t> &selected_indices_lt,
    int32_t selection_limit,
    int32_t &write_idx,
    uint32_t candidate)
{
    if (write_idx >= selection_limit) {
        return;
    }
    for (int32_t idx = 0; idx < write_idx; ++idx) {
        if (selected_indices_lt.GetValue(idx) == candidate) {
            return;
        }
    }
    selected_indices_lt.SetValue(write_idx, candidate);
    ++write_idx;
}

__aicore__ inline void quest_apply_anchor_selection(
    LocalTensor<uint32_t> &selected_indices_lt,
    LocalTensor<uint32_t> &scratch_indices_lt,
    int32_t seq_len,
    int32_t block_size,
    int32_t k)
{
    if (seq_len <= 0 || k <= 0) {
        return;
    }

    int32_t valid_page_count = DIV_ROUNDUP(seq_len, block_size);
    int32_t selection_limit = MIN(k, valid_page_count);
    if (selection_limit <= 0) {
        return;
    }

    for (int32_t idx = 0; idx < selection_limit; ++idx) {
        scratch_indices_lt.SetValue(idx, selected_indices_lt.GetValue(idx));
    }

    if (selection_limit == 1) {
        selected_indices_lt.SetValue(0, static_cast<uint32_t>(valid_page_count - 1));
        for (int32_t idx = 1; idx < k; ++idx) {
            selected_indices_lt.SetValue(idx, 0U);
        }
        return;
    }

    int32_t write_idx = 0;
    quest_emit_unique_index(selected_indices_lt, selection_limit, write_idx, 0U);
    if (valid_page_count >= 2) {
        quest_emit_unique_index(
            selected_indices_lt,
            selection_limit,
            write_idx,
            static_cast<uint32_t>(valid_page_count - 1));
    }

    for (int32_t idx = 0; idx < selection_limit && write_idx < selection_limit; ++idx) {
        uint32_t candidate = scratch_indices_lt.GetValue(idx);
        if (candidate >= static_cast<uint32_t>(valid_page_count)) {
            continue;
        }
        quest_emit_unique_index(selected_indices_lt, selection_limit, write_idx, candidate);
    }

    for (int32_t idx = write_idx; idx < k; ++idx) {
        selected_indices_lt.SetValue(idx, 0U);
    }
}

__aicore__ inline void quest_block_select_paged_bfloat16_impl(
    GM_ADDR query,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    GM_ADDR metadata_block_tables,
    GM_ADDR seq_lens,
    GM_ADDR selected_indices,
    int32_t batch_size,
    int32_t num_kv_heads,
    int32_t num_heads,
    int32_t block_size,
    int32_t head_dim,
    int32_t max_metadata_blocks_per_request,
    int32_t tokens_since_metadata_update,
    int32_t k)
{
    AscendC::SetAtomicNone();

    int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int32_t num_batch_heads = batch_size * num_heads;
    int32_t query_heads_per_kv_head = num_heads / num_kv_heads;

    AscendC::GlobalTensor<bfloat16_t> query_gm;
    AscendC::GlobalTensor<bfloat16_t> maxblocks_gm;
    AscendC::GlobalTensor<bfloat16_t> minblocks_gm;
    AscendC::GlobalTensor<int32_t> metadata_block_tables_gm;
    AscendC::GlobalTensor<int32_t> seq_lens_gm;
    AscendC::GlobalTensor<uint32_t> selected_indices_gm;

    query_gm.SetGlobalBuffer((__gm__ bfloat16_t *)query);
    maxblocks_gm.SetGlobalBuffer((__gm__ bfloat16_t *)maxblocks);
    minblocks_gm.SetGlobalBuffer((__gm__ bfloat16_t *)minblocks);
    metadata_block_tables_gm.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
    seq_lens_gm.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
    selected_indices_gm.SetGlobalBuffer((__gm__ uint32_t *)selected_indices);

    using VecBuf_t = AscendC::TBuf<AscendC::QuePosition::VECCALC>;
    VecBuf_t input_bf16_buf;
    VecBuf_t query_float_buf, maxblock_float_buf, minblock_float_buf;
    VecBuf_t block_scores_buf, accumulated_scores_buf;
    VecBuf_t selected_indices_buf, selected_values_buf;
    VecBuf_t tmp_concat_buf, concat_buf, index_local_buf, sort_tmp_buf;

    uint32_t input_bf16_buf_size = NUM_UB_BYTES(block_size * head_dim * sizeof(bfloat16_t));
    uint32_t query_float_size = NUM_UB_BYTES(head_dim * sizeof(float));
    uint32_t block_float_buf_size = NUM_UB_BYTES(block_size * head_dim * sizeof(float));
    uint32_t reduced_buf_size = NUM_UB_BYTES(block_size * sizeof(float));
    uint32_t accumulated_scores_size =
        NUM_UB_BYTES(max_metadata_blocks_per_request * block_size * sizeof(float));
    uint32_t selected_indices_buf_size = NUM_UB_BYTES(k * sizeof(uint32_t));
    uint32_t selected_values_buf_size = NUM_UB_BYTES(k * sizeof(float));
    uint32_t tmp_concat_buf_size = NUM_UB_BYTES(
        max_metadata_blocks_per_request * block_size * REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 *
        sizeof(float));
    uint32_t concat_buf_size = NUM_UB_BYTES(
        (max_metadata_blocks_per_request * block_size +
         max_metadata_blocks_per_request * block_size * REGION_PROPOSAL_DATA_SIZE_FLOAT_V220) *
        sizeof(float));
    uint32_t index_local_buf_size =
        NUM_UB_BYTES(max_metadata_blocks_per_request * block_size * sizeof(uint32_t));
    uint32_t sort_tmp_buf_size = NUM_UB_BYTES(
        max_metadata_blocks_per_request * block_size * REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 *
        sizeof(float));

    AscendC::TPipe pipe;
    pipe.InitBuffer(input_bf16_buf, input_bf16_buf_size);
    pipe.InitBuffer(query_float_buf, query_float_size);
    pipe.InitBuffer(maxblock_float_buf, block_float_buf_size);
    pipe.InitBuffer(minblock_float_buf, block_float_buf_size);
    pipe.InitBuffer(block_scores_buf, reduced_buf_size);
    pipe.InitBuffer(accumulated_scores_buf, accumulated_scores_size);
    pipe.InitBuffer(selected_indices_buf, selected_indices_buf_size);
    pipe.InitBuffer(selected_values_buf, selected_values_buf_size);
    pipe.InitBuffer(tmp_concat_buf, tmp_concat_buf_size);
    pipe.InitBuffer(concat_buf, concat_buf_size);
    pipe.InitBuffer(index_local_buf, index_local_buf_size);
    pipe.InitBuffer(sort_tmp_buf, sort_tmp_buf_size);

    AscendC::LocalTensor<bfloat16_t> input_bf16_lt = input_bf16_buf.Get<bfloat16_t>(block_size * head_dim);
    AscendC::LocalTensor<float> query_float_lt = query_float_buf.Get<float>();
    AscendC::LocalTensor<float> maxblock_float_lt = maxblock_float_buf.Get<float>();
    AscendC::LocalTensor<float> minblock_float_lt = minblock_float_buf.Get<float>();
    AscendC::LocalTensor<float> block_scores_lt = block_scores_buf.Get<float>();
    AscendC::LocalTensor<float> accumulated_scores_lt = accumulated_scores_buf.Get<float>();
    AscendC::LocalTensor<uint32_t> selected_indices_lt = selected_indices_buf.Get<uint32_t>();
    AscendC::LocalTensor<float> selected_values_lt = selected_values_buf.Get<float>();
    AscendC::LocalTensor<float> tmp_concat_lt = tmp_concat_buf.Get<float>();
    AscendC::LocalTensor<float> concat_lt = concat_buf.Get<float>();
    AscendC::LocalTensor<uint32_t> index_local_lt = index_local_buf.Get<uint32_t>();
    AscendC::LocalTensor<float> sort_tmp_lt = sort_tmp_buf.Get<float>();

    for (int32_t batch_head_idx = AscendC::GetBlockIdx(); batch_head_idx < num_batch_heads;
         batch_head_idx += num_blocks) {
        int32_t batch_idx = batch_head_idx / num_heads;
        int32_t query_head_idx = batch_head_idx % num_heads;
        int32_t kv_head_idx = query_head_idx / query_heads_per_kv_head;

        int32_t query_offset = batch_idx * num_heads * head_dim + query_head_idx * head_dim;
        int32_t output_offset = batch_head_idx * k;

        int32_t seq_len = seq_lens_gm.GetValue(batch_idx);
        int32_t num_tokens_per_meta_block = block_size * block_size;
        int32_t num_meta_blocks_in_request = DIV_ROUNDUP(seq_len, num_tokens_per_meta_block);

        uint16_t query_copy_block_len =
            NUM_DATA_BLOCKS(head_dim * sizeof(bfloat16_t));
        auto query_copy_params = AscendC::DataCopyParams(1, query_copy_block_len, 0, 0);
        AscendC::DataCopy(input_bf16_lt, query_gm[query_offset], query_copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

        AscendC::Cast<float, bfloat16_t>(
            query_float_lt,
            input_bf16_lt,
            AscendC::RoundMode::CAST_NONE,
            head_dim);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_V>();
        uint64_t mask = NUM_FLOAT_ELEMS_PER_VECTOR;
        uint64_t masks_per_head_dim = head_dim / mask;

        AscendC::Duplicate(
            accumulated_scores_lt,
            (float)MINFLOAT,
            max_metadata_blocks_per_request * num_meta_blocks_in_request);

        for (int32_t meta_block = 0; meta_block < num_meta_blocks_in_request; meta_block++) {
            int32_t meta_block_id =
                metadata_block_tables_gm.GetValue(batch_idx * max_metadata_blocks_per_request + meta_block);
            int32_t meta_block_offset =
                meta_block_id * block_size * num_kv_heads * head_dim + kv_head_idx * head_dim;

            AscendC::DataCopyParams gm_ub_cp;
            gm_ub_cp.blockCount = block_size;
            gm_ub_cp.blockLen = DIV_ROUNDUP(head_dim * sizeof(bfloat16_t), BYTES_DATA_BLOCK);
            gm_ub_cp.srcStride =
                DIV_ROUNDUP((num_kv_heads - 1) * head_dim * sizeof(bfloat16_t), BYTES_DATA_BLOCK);
            gm_ub_cp.dstStride = 0;

            AscendC::DataCopy(input_bf16_lt, maxblocks_gm[meta_block_offset], gm_ub_cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Cast<float, bfloat16_t>(
                maxblock_float_lt,
                input_bf16_lt,
                AscendC::RoundMode::CAST_NONE,
                block_size * head_dim);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::BinaryRepeatParams mul_repeat_params = AscendC::BinaryRepeatParams(
                1,
                1,
                1,
                8 * masks_per_head_dim,
                0,
                8 * masks_per_head_dim);
            AscendC::Mul(
                maxblock_float_lt,
                query_float_lt,
                maxblock_float_lt,
                NUM_FLOAT_ELEMS_PER_VECTOR,
                block_size,
                mul_repeat_params);
            AscendC::Mul(
                maxblock_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR],
                query_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR],
                maxblock_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR],
                NUM_FLOAT_ELEMS_PER_VECTOR,
                block_size,
                mul_repeat_params);

            AscendC::DataCopy(input_bf16_lt, minblocks_gm[meta_block_offset], gm_ub_cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::Cast<float, bfloat16_t>(
                minblock_float_lt,
                input_bf16_lt,
                AscendC::RoundMode::CAST_NONE,
                block_size * head_dim);
            AscendC::Mul(
                minblock_float_lt,
                query_float_lt,
                minblock_float_lt,
                NUM_FLOAT_ELEMS_PER_VECTOR,
                block_size,
                mul_repeat_params);
            AscendC::Mul(
                minblock_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR],
                query_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR],
                minblock_float_lt[NUM_FLOAT_ELEMS_PER_VECTOR],
                NUM_FLOAT_ELEMS_PER_VECTOR,
                block_size,
                mul_repeat_params);

            AscendC::Max(maxblock_float_lt, maxblock_float_lt, minblock_float_lt, block_size * head_dim);

            AscendC::RepeatReduceSum(
                minblock_float_lt,
                maxblock_float_lt,
                block_size,
                mask,
                0,
                1,
                1,
                8);
            AscendC::RepeatReduceSum(
                minblock_float_lt[block_size],
                maxblock_float_lt[block_size * head_dim / masks_per_head_dim],
                block_size,
                mask,
                0,
                1,
                1,
                8);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::PairReduceSum(
                block_scores_lt,
                minblock_float_lt,
                masks_per_head_dim * block_size / mask,
                mask,
                1,
                1,
                8);
            AscendC::PipeBarrier<PIPE_V>();

            uint64_t seq_len_curr_meta_block =
                MIN(seq_len - (meta_block * num_tokens_per_meta_block), block_size);
            for (int32_t sub_meta_block_id = 0; sub_meta_block_id < static_cast<int32_t>(masks_per_head_dim);
                 sub_meta_block_id++) {
                int32_t block_scores_offset = sub_meta_block_id * NUM_FLOAT_ELEMS_PER_VECTOR;
                int32_t accumulated_offset = meta_block * block_size + block_scores_offset;
                AscendC::Copy(
                    accumulated_scores_lt[accumulated_offset],
                    block_scores_lt[block_scores_offset],
                    seq_len_curr_meta_block - sub_meta_block_id * NUM_FLOAT_ELEMS_PER_VECTOR,
                    1,
                    {1, 1, 8, 8});
                if (meta_block < num_meta_blocks_in_request - 1) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                }
            }
        }

        uint32_t total_elements = num_meta_blocks_in_request * block_size;
        uint32_t concat_repeat_times = DIV_ROUNDUP(total_elements, 32);
        uint32_t sort_repeat_times = DIV_ROUNDUP(total_elements, 32);
        uint32_t extract_repeat_times = DIV_ROUNDUP(total_elements, 32);

        for (uint32_t idx = 0; idx < total_elements; idx++) {
            index_local_lt.SetValue(idx, idx);
        }

        AscendC::Concat(concat_lt, accumulated_scores_lt, tmp_concat_lt, concat_repeat_times);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sort<float, true>(
            maxblock_float_lt,
            concat_lt,
            index_local_lt,
            sort_tmp_lt,
            sort_repeat_times);
        AscendC::Extract(selected_values_lt, selected_indices_lt, maxblock_float_lt, extract_repeat_times);

        if (tokens_since_metadata_update >= 0) {
            quest_apply_anchor_selection(selected_indices_lt, index_local_lt, seq_len, block_size, k);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        uint16_t indices_copy_block_len = NUM_DATA_BLOCKS(k * sizeof(int32_t));
        auto indices_copy_params = AscendC::DataCopyParams(1, indices_copy_block_len, 0, 0);
        AscendC::DataCopy(selected_indices_gm[output_offset], selected_indices_lt, indices_copy_params);
    }
}

__aicore__ inline void quest_block_select_paged_half_impl(
    GM_ADDR query,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    GM_ADDR metadata_block_tables,
    GM_ADDR seq_lens,
    GM_ADDR selected_indices,
    int32_t batch_size,
    int32_t num_kv_heads,
    int32_t num_heads,
    int32_t block_size,
    int32_t head_dim,
    int32_t max_metadata_blocks_per_request,
    int32_t tokens_since_metadata_update,
    int32_t k)
{
    AscendC::SetAtomicNone();

    int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    int32_t num_batch_heads = batch_size * num_heads;
    int32_t query_heads_per_kv_head = num_heads / num_kv_heads;

    AscendC::GlobalTensor<half> query_gm;
    AscendC::GlobalTensor<half> maxblocks_gm;
    AscendC::GlobalTensor<half> minblocks_gm;
    AscendC::GlobalTensor<int32_t> metadata_block_tables_gm;
    AscendC::GlobalTensor<int32_t> seq_lens_gm;
    AscendC::GlobalTensor<uint32_t> selected_indices_gm;

    query_gm.SetGlobalBuffer((__gm__ half *)query);
    maxblocks_gm.SetGlobalBuffer((__gm__ half *)maxblocks);
    minblocks_gm.SetGlobalBuffer((__gm__ half *)minblocks);
    metadata_block_tables_gm.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
    seq_lens_gm.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
    selected_indices_gm.SetGlobalBuffer((__gm__ uint32_t *)selected_indices);

    using VecBuf_t = AscendC::TBuf<AscendC::QuePosition::VECCALC>;
    VecBuf_t query_buf, maxblock_buf, minblock_buf;
    VecBuf_t block_scores_buf, accumulated_scores_buf;
    VecBuf_t selected_indices_buf, selected_values_buf;
    VecBuf_t tmp_concat_buf, concat_buf, index_local_buf, sort_tmp_buf;

    uint32_t query_buf_size = NUM_UB_BYTES(head_dim * sizeof(half));
    uint32_t block_buf_size = NUM_UB_BYTES(block_size * head_dim * sizeof(half));
    uint32_t reduced_buf_size = NUM_UB_BYTES(block_size * sizeof(half));
    uint32_t accumulated_scores_size =
        NUM_UB_BYTES(max_metadata_blocks_per_request * block_size * sizeof(half));
    uint32_t selected_indices_buf_size = NUM_UB_BYTES(k * sizeof(uint32_t));
    uint32_t selected_values_buf_size = NUM_UB_BYTES(k * sizeof(half));
    uint32_t tmp_concat_buf_size = NUM_UB_BYTES(
        max_metadata_blocks_per_request * block_size * REGION_PROPOSAL_DATA_SIZE_V200 *
        sizeof(half));
    uint32_t concat_buf_size = NUM_UB_BYTES(
        (max_metadata_blocks_per_request * block_size +
         max_metadata_blocks_per_request * block_size * REGION_PROPOSAL_DATA_SIZE_V200) *
        sizeof(half));
    uint32_t index_local_buf_size =
        NUM_UB_BYTES(max_metadata_blocks_per_request * block_size * sizeof(uint32_t));
    uint32_t sort_tmp_buf_size = NUM_UB_BYTES(
        max_metadata_blocks_per_request * block_size * REGION_PROPOSAL_DATA_SIZE_HALF_V220 *
        sizeof(half));

    AscendC::TPipe pipe;
    pipe.InitBuffer(query_buf, query_buf_size);
    pipe.InitBuffer(maxblock_buf, block_buf_size);
    pipe.InitBuffer(minblock_buf, block_buf_size);
    pipe.InitBuffer(block_scores_buf, reduced_buf_size);
    pipe.InitBuffer(accumulated_scores_buf, accumulated_scores_size);
    pipe.InitBuffer(selected_indices_buf, selected_indices_buf_size);
    pipe.InitBuffer(selected_values_buf, selected_values_buf_size);
    pipe.InitBuffer(tmp_concat_buf, tmp_concat_buf_size);
    pipe.InitBuffer(concat_buf, concat_buf_size);
    pipe.InitBuffer(index_local_buf, index_local_buf_size);
    pipe.InitBuffer(sort_tmp_buf, sort_tmp_buf_size);

    AscendC::LocalTensor<half> query_lt = query_buf.Get<half>();
    AscendC::LocalTensor<half> maxblock_lt = maxblock_buf.Get<half>();
    AscendC::LocalTensor<half> minblock_lt = minblock_buf.Get<half>();
    AscendC::LocalTensor<half> block_scores_lt = block_scores_buf.Get<half>();
    AscendC::LocalTensor<half> accumulated_scores_lt = accumulated_scores_buf.Get<half>();
    AscendC::LocalTensor<uint32_t> selected_indices_lt = selected_indices_buf.Get<uint32_t>();
    AscendC::LocalTensor<half> selected_values_lt = selected_values_buf.Get<half>();
    AscendC::LocalTensor<half> tmp_concat_lt = tmp_concat_buf.Get<half>();
    AscendC::LocalTensor<half> concat_lt = concat_buf.Get<half>();
    AscendC::LocalTensor<uint32_t> index_local_lt = index_local_buf.Get<uint32_t>();
    AscendC::LocalTensor<half> sort_tmp_lt = sort_tmp_buf.Get<half>();

    for (int32_t batch_head_idx = AscendC::GetBlockIdx(); batch_head_idx < num_batch_heads;
         batch_head_idx += num_blocks) {
        int32_t batch_idx = batch_head_idx / num_heads;
        int32_t query_head_idx = batch_head_idx % num_heads;
        int32_t kv_head_idx = query_head_idx / query_heads_per_kv_head;

        int32_t query_offset = batch_idx * num_heads * head_dim + query_head_idx * head_dim;
        int32_t output_offset = batch_head_idx * k;

        int32_t seq_len = seq_lens_gm.GetValue(batch_idx);
        int32_t num_tokens_per_meta_block = block_size * block_size;
        int32_t num_meta_blocks_in_request = DIV_ROUNDUP(seq_len, num_tokens_per_meta_block);

        uint16_t query_copy_block_len =
            NUM_DATA_BLOCKS(head_dim * sizeof(half));
        auto query_copy_params = AscendC::DataCopyParams(1, query_copy_block_len, 0, 0);
        AscendC::DataCopy(query_lt, query_gm[query_offset], query_copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        uint64_t mask = head_dim;

        AscendC::Duplicate(
            accumulated_scores_lt,
            (half)(MINHALF),
            max_metadata_blocks_per_request * num_meta_blocks_in_request);
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t meta_block = 0; meta_block < num_meta_blocks_in_request; meta_block++) {
            int32_t meta_block_id =
                metadata_block_tables_gm.GetValue(batch_idx * max_metadata_blocks_per_request + meta_block);
            int32_t meta_block_offset =
                meta_block_id * block_size * num_kv_heads * head_dim + kv_head_idx * head_dim;

            AscendC::DataCopyParams gm_ub_cp;
            gm_ub_cp.blockCount = block_size;
            gm_ub_cp.blockLen = DIV_ROUNDUP(head_dim * sizeof(half), BYTES_DATA_BLOCK);
            gm_ub_cp.srcStride =
                DIV_ROUNDUP((num_kv_heads - 1) * head_dim * sizeof(half), BYTES_DATA_BLOCK);
            gm_ub_cp.dstStride = 0;

            AscendC::DataCopy(maxblock_lt, maxblocks_gm[meta_block_offset], gm_ub_cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::DataCopy(minblock_lt, minblocks_gm[meta_block_offset], gm_ub_cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);

            uint8_t repeat_times = block_size;
            AscendC::BinaryRepeatParams mul_repeat_params = {1, 1, 1, 8, 0, 8};
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Mul(maxblock_lt, query_lt, maxblock_lt, mask, repeat_times, mul_repeat_params);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::Mul(minblock_lt, query_lt, minblock_lt, mask, repeat_times, mul_repeat_params);
            AscendC::Max(
                maxblock_lt,
                maxblock_lt,
                minblock_lt,
                mask,
                repeat_times,
                {1, 1, 1, 8, 8, 8});
            AscendC::RepeatReduceSum<half, true>(
                block_scores_lt,
                maxblock_lt,
                repeat_times,
                mask,
                0,
                1,
                1,
                8);
            AscendC::PipeBarrier<PIPE_V>();

            int32_t accumulated_offset = meta_block * block_size;
            uint64_t seq_len_curr_meta_block =
                MIN(seq_len - (meta_block * num_tokens_per_meta_block), block_size);
            AscendC::Copy(
                accumulated_scores_lt[accumulated_offset],
                block_scores_lt,
                seq_len_curr_meta_block,
                1,
                {1, 1, 8, 8});
            if (meta_block < num_meta_blocks_in_request - 1) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            }
        }

        uint32_t total_elements = num_meta_blocks_in_request * block_size;
        uint32_t sort_repeat_times = DIV_ROUNDUP(total_elements, 32);
        uint32_t concat_repeat_times = DIV_ROUNDUP(total_elements, 32);
        uint32_t extract_repeat_times = DIV_ROUNDUP(total_elements, 32);

        for (uint32_t idx = 0; idx < total_elements; idx++) {
            index_local_lt.SetValue(idx, idx);
        }

        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Concat(concat_lt, accumulated_scores_lt, tmp_concat_lt, concat_repeat_times);
        AscendC::Sort<half, true>(maxblock_lt, concat_lt, index_local_lt, sort_tmp_lt, sort_repeat_times);
        AscendC::Extract(selected_values_lt, selected_indices_lt, maxblock_lt, extract_repeat_times);

        if (tokens_since_metadata_update >= 0) {
            quest_apply_anchor_selection(selected_indices_lt, index_local_lt, seq_len, block_size, k);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
        uint16_t indices_copy_block_len = NUM_DATA_BLOCKS(k * sizeof(int32_t));
        auto indices_copy_params = AscendC::DataCopyParams(1, indices_copy_block_len, 0, 0);
        AscendC::DataCopy(selected_indices_gm[output_offset], selected_indices_lt, indices_copy_params);
    }
}

extern "C" __global__ __aicore__ void quest_block_select_paged(
    GM_ADDR query,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    GM_ADDR metadata_block_tables,
    GM_ADDR seq_lens,
    GM_ADDR selected_indices,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    QUEST_BLOCK_SELECT_PAGED_COPY_TILING_DATA(QuestBlockSelectPagedTilingData, tiling);

    if (TILING_KEY_IS(QUEST_BLOCK_SELECT_PAGED_TILING_FP16)) {
        quest_block_select_paged_half_impl(
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            selected_indices,
            static_cast<int32_t>(tiling_data->batchSize),
            static_cast<int32_t>(tiling_data->numKvHeads),
            static_cast<int32_t>(tiling_data->numHeads),
            static_cast<int32_t>(tiling_data->blockSize),
            static_cast<int32_t>(tiling_data->headDim),
            static_cast<int32_t>(tiling_data->maxMetadataBlocksPerRequest),
            tiling_data->tokensSinceMetadataUpdate,
            static_cast<int32_t>(tiling_data->k));
        return;
    }

    if (TILING_KEY_IS(QUEST_BLOCK_SELECT_PAGED_TILING_BF16)) {
        quest_block_select_paged_bfloat16_impl(
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            selected_indices,
            static_cast<int32_t>(tiling_data->batchSize),
            static_cast<int32_t>(tiling_data->numKvHeads),
            static_cast<int32_t>(tiling_data->numHeads),
            static_cast<int32_t>(tiling_data->blockSize),
            static_cast<int32_t>(tiling_data->headDim),
            static_cast<int32_t>(tiling_data->maxMetadataBlocksPerRequest),
            tiling_data->tokensSinceMetadataUpdate,
            static_cast<int32_t>(tiling_data->k));
        return;
    }

    ASSERT(false && "Unsupported quest_block_select_paged tiling key.");
}
