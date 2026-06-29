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
#define BYTES_VECTOR_REPEAT 256
#define NUM_FLOAT_ELEMS_PER_VECTOR 64
#define NUM_SORT_PAIRS_PER_REPEAT 32
#define NUM_SORT_PAIR_ELEMS 2
#define QUEST_GATHER_INDEX_PATTERN 2
#define QUEST_MAX_SELECTED_BLOCKS 64
#define DIV_ROUNDUP(x, y) (((x) + (y)-1) / (y))
#define DIV_ROUNDUP_MUL(bytes, bytes_per_block) (DIV_ROUNDUP(bytes, bytes_per_block) * (bytes_per_block))
#define NUM_UB_BYTES(bytes) (DIV_ROUNDUP_MUL(bytes, BYTES_UB_BLOCK))
#define NUM_DATA_BLOCKS(bytes) (DIV_ROUNDUP(bytes, BYTES_DATA_BLOCK))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define QUEST_MIN_SCORE -3.4028235e38f
#define QUEST_MAX_SCORE 3.4028235e38f

constexpr uint32_t REGION_PROPOSAL_DATA_SIZE_FLOAT_V220 = 2;

using namespace AscendC;

// QuestBlockSelectPagedTilingData is generated from the op_host tiling
// definition. The kernel must not redeclare it locally.
using QuestPageIndexT = int32_t;
using QuestSortIndexT = uint32_t;

__aicore__ inline void quest_zero_indices(
    LocalTensor<QuestPageIndexT> indices_lt,
    int32_t count)
{
    if (count <= 0) {
        return;
    }
    Duplicate(indices_lt, static_cast<QuestPageIndexT>(0), count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void quest_apply_sequential_selection(
    LocalTensor<QuestPageIndexT> &selected_indices_lt,
    int32_t valid_page_count,
    int32_t k)
{
    int32_t num_selected_pages = valid_page_count > 0 ? MIN(k, valid_page_count) : 0;
    quest_zero_indices(selected_indices_lt, k);
    if (num_selected_pages > 0) {
        ArithProgression(
            selected_indices_lt,
            static_cast<QuestPageIndexT>(0),
            static_cast<QuestPageIndexT>(1),
            static_cast<int32_t>(num_selected_pages));
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename StorageT>
class KernelQuestBlockSelectPaged {
    using VecBufT = AscendC::TBuf<AscendC::QuePosition::VECCALC>;
    using ComputeT = float;
    static constexpr uint32_t REGION_SIZE = REGION_PROPOSAL_DATA_SIZE_FLOAT_V220;

    struct LocalTensors {
        AscendC::LocalTensor<ComputeT> query;
        AscendC::LocalTensor<ComputeT> maxblock;
        AscendC::LocalTensor<ComputeT> minblock;
        AscendC::LocalTensor<ComputeT> block_scores;
        AscendC::LocalTensor<ComputeT> accumulated_scores;
        AscendC::LocalTensor<QuestPageIndexT> selected_indices;
        // Sort requires uint32_t index lanes; page IDs stay int32_t everywhere else.
        AscendC::LocalTensor<QuestSortIndexT> index_local;
        AscendC::LocalTensor<ComputeT> sort_tmp;
    };

public:
    __aicore__ inline KernelQuestBlockSelectPaged() {}

    __aicore__ inline void Init(
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
        int32_t k,
        int32_t output_stride)
    {
        AscendC::SetAtomicNone();

        batch_size_ = batch_size;
        num_kv_heads_ = num_kv_heads;
        num_heads_ = num_heads;
        block_size_ = block_size;
        head_dim_ = head_dim;
        max_metadata_blocks_per_request_ = max_metadata_blocks_per_request;
        tokens_since_metadata_update_ = tokens_since_metadata_update;
        k_ = k;
        output_stride_ = output_stride;
        ASSERT(k_ <= static_cast<int32_t>(QUEST_MAX_SELECTED_BLOCKS) &&
               "quest_block_select_paged requires k <= 64.");
        ASSERT(output_stride_ >= k_ &&
               "quest_block_select_paged requires output_stride >= k.");
        head_dim_storage_blocks_ =
            NUM_DATA_BLOCKS(head_dim_ * static_cast<int32_t>(sizeof(StorageT)));
        inter_kv_head_stride_blocks_ = NUM_DATA_BLOCKS(
            (num_kv_heads_ - 1) * head_dim_ * static_cast<int32_t>(sizeof(StorageT)));
        metadata_block_stride_elems_ = block_size_ * num_kv_heads_ * head_dim_;

        query_gm_.SetGlobalBuffer((__gm__ StorageT *)query);
        maxblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)maxblocks);
        minblocks_gm_.SetGlobalBuffer((__gm__ StorageT *)minblocks);
        metadata_block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
        seq_lens_gm_.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
        selected_indices_gm_.SetGlobalBuffer((__gm__ QuestPageIndexT *)selected_indices);

        uint32_t input_storage_buf_size =
            NUM_UB_BYTES(block_size_ * head_dim_ * static_cast<int32_t>(sizeof(StorageT)));
        pipe_.InitBuffer(input_storage_buf_, input_storage_buf_size);

        uint32_t query_buf_size =
            NUM_UB_BYTES(head_dim_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t block_buf_size =
            NUM_UB_BYTES(block_size_ * head_dim_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t reduced_buf_size =
            NUM_UB_BYTES(block_size_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t accumulated_scores_size = NUM_UB_BYTES(
            max_metadata_blocks_per_request_ * block_size_ * static_cast<int32_t>(sizeof(ComputeT)));
        uint32_t selected_indices_buf_size = NUM_UB_BYTES(
            DIV_ROUNDUP(k_, NUM_SORT_PAIRS_PER_REPEAT) * NUM_SORT_PAIRS_PER_REPEAT *
            static_cast<int32_t>(sizeof(QuestPageIndexT)));
        uint32_t index_local_buf_size =
            NUM_UB_BYTES(max_metadata_blocks_per_request_ * block_size_ *
                         static_cast<int32_t>(sizeof(QuestSortIndexT)));
        uint32_t sort_tmp_buf_size = NUM_UB_BYTES(
            max_metadata_blocks_per_request_ * block_size_ * REGION_SIZE *
            static_cast<int32_t>(sizeof(ComputeT)));

        pipe_.InitBuffer(query_buf_, query_buf_size);
        pipe_.InitBuffer(maxblock_buf_, block_buf_size);
        pipe_.InitBuffer(minblock_buf_, block_buf_size);
        pipe_.InitBuffer(block_scores_buf_, reduced_buf_size);
        pipe_.InitBuffer(accumulated_scores_buf_, accumulated_scores_size);
        pipe_.InitBuffer(selected_indices_buf_, selected_indices_buf_size);
        pipe_.InitBuffer(index_local_buf_, index_local_buf_size);
        pipe_.InitBuffer(sort_tmp_buf_, sort_tmp_buf_size);
    }

    __aicore__ inline void Process()
    {
        int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        int32_t num_batch_heads = batch_size_ * num_heads_;
        int32_t query_heads_per_kv_head = num_heads_ / num_kv_heads_;

        LocalTensors tensors = GetLocalTensors();

        for (int32_t batch_head_idx = AscendC::GetBlockIdx(); batch_head_idx < num_batch_heads;
             batch_head_idx += num_blocks) {
            int32_t batch_idx = batch_head_idx / num_heads_;
            int32_t query_head_idx = batch_head_idx % num_heads_;
            int32_t kv_head_idx = query_head_idx / query_heads_per_kv_head;

            int32_t query_offset = batch_idx * num_heads_ * head_dim_ + query_head_idx * head_dim_;
            int32_t output_offset = batch_head_idx * output_stride_;

            int32_t seq_len = seq_lens_gm_.GetValue(batch_idx);
            int32_t valid_page_count = seq_len > 0 ? DIV_ROUNDUP(seq_len, block_size_) : 0;
            bool use_fixed_anchors = tokens_since_metadata_update_ >= 0;
            if (unlikely(valid_page_count <= 0 || k_ >= valid_page_count)) {
                quest_apply_sequential_selection(
                    tensors.selected_indices,
                    valid_page_count,
                    k_);
            } else {
                int32_t num_meta_blocks_in_request = DIV_ROUNDUP(valid_page_count, block_size_);
                int32_t sort_element_count = DIV_ROUNDUP(valid_page_count, 32) * 32;

                LoadQuery(tensors, query_offset);
                DuplicateAccumulatedScores(tensors, sort_element_count);

                for (int32_t meta_block = 0; meta_block < num_meta_blocks_in_request; meta_block++) {
                    int32_t meta_block_id = metadata_block_tables_gm_.GetValue(
                        batch_idx * max_metadata_blocks_per_request_ + meta_block);
                    int32_t meta_block_offset =
                        meta_block_id * metadata_block_stride_elems_ + kv_head_idx * head_dim_;

                    ScoreMetadataBlock(tensors, meta_block_offset);
                    CopyScoresToAccumulated(
                        tensors,
                        valid_page_count,
                        meta_block);
                }

                if (likely(use_fixed_anchors)) {
                    PinAnchorScores(tensors, valid_page_count);
                }
                SortAndGatherTopK(tensors, sort_element_count);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID3);
            AscendC::DataCopyExtParams indices_copy_params{
                1,
                static_cast<uint32_t>(k_ * static_cast<int32_t>(sizeof(QuestPageIndexT))),
                0,
                0,
                0};
            AscendC::DataCopyPad(
                selected_indices_gm_[output_offset],
                tensors.selected_indices,
                indices_copy_params);
            // The selected_indices UB buffer is reused on the next loop iteration.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        }
    }

private:
    __aicore__ inline LocalTensors GetLocalTensors()
    {
        return {
            query_buf_.Get<ComputeT>(),
            maxblock_buf_.Get<ComputeT>(),
            minblock_buf_.Get<ComputeT>(),
            block_scores_buf_.Get<ComputeT>(),
            accumulated_scores_buf_.Get<ComputeT>(),
            selected_indices_buf_.Get<QuestPageIndexT>(),
            index_local_buf_.Get<QuestSortIndexT>(),
            sort_tmp_buf_.Get<ComputeT>()};
    }

    __aicore__ inline void LoadQuery(
        LocalTensors &tensors,
        int32_t query_offset)
    {
        uint16_t query_copy_block_len =
            NUM_DATA_BLOCKS(head_dim_ * static_cast<int32_t>(sizeof(StorageT)));
        auto query_copy_params = AscendC::DataCopyParams(1, query_copy_block_len, 0, 0);

        AscendC::LocalTensor<StorageT> input_storage_lt =
            input_storage_buf_.Get<StorageT>(block_size_ * head_dim_);
        AscendC::DataCopy(input_storage_lt, query_gm_[query_offset], query_copy_params);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Cast<ComputeT, StorageT>(
            tensors.query,
            input_storage_lt,
            AscendC::RoundMode::CAST_NONE,
            head_dim_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DuplicateAccumulatedScores(
        LocalTensors &tensors,
        int32_t sort_element_count)
    {
        AscendC::Duplicate(
            tensors.accumulated_scores,
            static_cast<ComputeT>(QUEST_MIN_SCORE),
            sort_element_count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ScoreMetadataBlock(
        LocalTensors &tensors,
        int32_t meta_block_offset)
    {
        AscendC::LocalTensor<StorageT> input_storage_lt =
            input_storage_buf_.Get<StorageT>(block_size_ * head_dim_);
        AscendC::DataCopyParams gm_ub_cp;
        gm_ub_cp.blockCount = block_size_;
        gm_ub_cp.blockLen = head_dim_storage_blocks_;
        gm_ub_cp.srcStride = inter_kv_head_stride_blocks_;
        gm_ub_cp.dstStride = 0;

        uint64_t mask = NUM_FLOAT_ELEMS_PER_VECTOR;
        uint64_t masks_per_head_dim = head_dim_ / mask;
        AscendC::BinaryRepeatParams mul_repeat_params = AscendC::BinaryRepeatParams(
            1,
            1,
            1,
            8 * masks_per_head_dim,
            0,
            8 * masks_per_head_dim);

        AscendC::DataCopy(input_storage_lt, maxblocks_gm_[meta_block_offset], gm_ub_cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Cast<ComputeT, StorageT>(
            tensors.maxblock,
            input_storage_lt,
            AscendC::RoundMode::CAST_NONE,
            block_size_ * head_dim_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::Mul(
            tensors.maxblock,
            tensors.query,
            tensors.maxblock,
            NUM_FLOAT_ELEMS_PER_VECTOR,
            block_size_,
            mul_repeat_params);
        AscendC::Mul(
            tensors.maxblock[NUM_FLOAT_ELEMS_PER_VECTOR],
            tensors.query[NUM_FLOAT_ELEMS_PER_VECTOR],
            tensors.maxblock[NUM_FLOAT_ELEMS_PER_VECTOR],
            NUM_FLOAT_ELEMS_PER_VECTOR,
            block_size_,
            mul_repeat_params);

        AscendC::DataCopy(input_storage_lt, minblocks_gm_[meta_block_offset], gm_ub_cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
        AscendC::Cast<ComputeT, StorageT>(
            tensors.minblock,
            input_storage_lt,
            AscendC::RoundMode::CAST_NONE,
            block_size_ * head_dim_);
        AscendC::Mul(
            tensors.minblock,
            tensors.query,
            tensors.minblock,
            NUM_FLOAT_ELEMS_PER_VECTOR,
            block_size_,
            mul_repeat_params);
        AscendC::Mul(
            tensors.minblock[NUM_FLOAT_ELEMS_PER_VECTOR],
            tensors.query[NUM_FLOAT_ELEMS_PER_VECTOR],
            tensors.minblock[NUM_FLOAT_ELEMS_PER_VECTOR],
            NUM_FLOAT_ELEMS_PER_VECTOR,
            block_size_,
            mul_repeat_params);

        // Max consumes the vector Mul outputs from both metadata bounds.
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(tensors.maxblock, tensors.maxblock, tensors.minblock, block_size_ * head_dim_);

        AscendC::RepeatReduceSum(
            tensors.minblock,
            tensors.maxblock,
            block_size_,
            mask,
            0,
            1,
            1,
            8);
        AscendC::RepeatReduceSum(
            tensors.minblock[block_size_],
            tensors.maxblock[block_size_ * head_dim_ / masks_per_head_dim],
            block_size_,
            mask,
            0,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::PairReduceSum(
            tensors.block_scores,
            tensors.minblock,
            masks_per_head_dim * block_size_ / mask,
            mask,
            1,
            1,
            8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyScoresToAccumulated(
        LocalTensors &tensors,
        int32_t valid_page_count,
        int32_t meta_block)
    {
        int32_t start_page = meta_block * block_size_;
        int32_t pages_in_meta_block = MIN(valid_page_count - start_page, block_size_);
        if (pages_in_meta_block <= 0) {
            return;
        }

        uint64_t mask = NUM_FLOAT_ELEMS_PER_VECTOR;
        uint64_t masks_per_head_dim = head_dim_ / mask;
        for (int32_t sub_meta_block_id = 0; sub_meta_block_id < static_cast<int32_t>(masks_per_head_dim);
             sub_meta_block_id++) {
            int32_t block_scores_offset = sub_meta_block_id * NUM_FLOAT_ELEMS_PER_VECTOR;
            int32_t pages_remaining = pages_in_meta_block - block_scores_offset;
            if (pages_remaining <= 0) {
                break;
            }

            int32_t accumulated_offset = meta_block * block_size_ + block_scores_offset;
            AscendC::Copy(
                tensors.accumulated_scores[accumulated_offset],
                tensors.block_scores[block_scores_offset],
                static_cast<uint64_t>(MIN(pages_remaining, NUM_FLOAT_ELEMS_PER_VECTOR)),
                1,
                {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PinAnchorScores(LocalTensors &tensors, int32_t valid_page_count)
    {
        if (valid_page_count <= 0) {
            return;
        }

        tensors.accumulated_scores.SetValue(0, static_cast<ComputeT>(QUEST_MAX_SCORE));
        tensors.accumulated_scores.SetValue(valid_page_count - 1, static_cast<ComputeT>(QUEST_MAX_SCORE));
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SortAndGatherTopK(
        LocalTensors &tensors,
        int32_t sort_element_count)
    {
        uint32_t repeat_times = sort_element_count / 32;

        ArithProgression(
            tensors.index_local.template ReinterpretCast<QuestPageIndexT>(),
            static_cast<QuestPageIndexT>(0),
            static_cast<QuestPageIndexT>(1),
            static_cast<int32_t>(sort_element_count));

        AscendC::Sort<ComputeT, true>(
            tensors.maxblock,
            tensors.accumulated_scores,
            tensors.index_local,
            tensors.sort_tmp,
            repeat_times);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::GatherMaskParams gather_mask_params;
        gather_mask_params.repeatTimes = static_cast<uint8_t>(
            DIV_ROUNDUP(k_ * static_cast<int32_t>(sizeof(ComputeT)) * NUM_SORT_PAIR_ELEMS,
                        BYTES_VECTOR_REPEAT));
        gather_mask_params.src0BlockStride = 1;
        gather_mask_params.src0RepeatStride = BYTES_VECTOR_REPEAT / BYTES_DATA_BLOCK;
        gather_mask_params.src1RepeatStride = 0;

        uint64_t rsvd_cnt = 0;
        uint8_t src1_pattern = QUEST_GATHER_INDEX_PATTERN;
        AscendC::GatherMask(
            tensors.selected_indices,
            tensors.maxblock.template ReinterpretCast<QuestPageIndexT>(),
            src1_pattern,
            false,
            static_cast<uint32_t>(0),
            gather_mask_params,
            rsvd_cnt);
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::TPipe pipe_;
    VecBufT input_storage_buf_;
    VecBufT query_buf_;
    VecBufT maxblock_buf_;
    VecBufT minblock_buf_;
    VecBufT block_scores_buf_;
    VecBufT accumulated_scores_buf_;
    VecBufT selected_indices_buf_;
    VecBufT index_local_buf_;
    VecBufT sort_tmp_buf_;

    AscendC::GlobalTensor<StorageT> query_gm_;
    AscendC::GlobalTensor<StorageT> maxblocks_gm_;
    AscendC::GlobalTensor<StorageT> minblocks_gm_;
    AscendC::GlobalTensor<int32_t> metadata_block_tables_gm_;
    AscendC::GlobalTensor<int32_t> seq_lens_gm_;
    AscendC::GlobalTensor<QuestPageIndexT> selected_indices_gm_;

    int32_t batch_size_;
    int32_t num_kv_heads_;
    int32_t num_heads_;
    int32_t block_size_;
    int32_t head_dim_;
    int32_t max_metadata_blocks_per_request_;
    int32_t tokens_since_metadata_update_;
    int32_t k_;
    int32_t output_stride_;
    uint16_t head_dim_storage_blocks_;
    uint16_t inter_kv_head_stride_blocks_;
    int32_t metadata_block_stride_elems_;
};

template <typename StorageT>
__aicore__ inline void RunQuestBlockSelectPaged(
    GM_ADDR query,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    GM_ADDR metadata_block_tables,
    GM_ADDR seq_lens,
    GM_ADDR selected_indices,
    const QuestBlockSelectPagedTilingData *__restrict tiling_data)
{
    KernelQuestBlockSelectPaged<StorageT> op;
    op.Init(
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
        static_cast<int32_t>(tiling_data->k),
        static_cast<int32_t>(tiling_data->outputStride));
    op.Process();
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

    if (!TILING_KEY_IS(QUEST_BLOCK_SELECT_PAGED_TILING)) {
        ASSERT(false && "Unsupported quest_block_select_paged tiling key.");
        return;
    }

    if (tiling_data->dataType == QUEST_BLOCK_SELECT_PAGED_DTYPE_FP16) {
        RunQuestBlockSelectPaged<half>(
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            selected_indices,
            tiling_data);
        return;
    }

    if (tiling_data->dataType == QUEST_BLOCK_SELECT_PAGED_DTYPE_BF16) {
        RunQuestBlockSelectPaged<bfloat16_t>(
            query,
            maxblocks,
            minblocks,
            metadata_block_tables,
            seq_lens,
            selected_indices,
            tiling_data);
        return;
    }

    ASSERT(false && "Unsupported quest_block_select_paged dtype.");
}
