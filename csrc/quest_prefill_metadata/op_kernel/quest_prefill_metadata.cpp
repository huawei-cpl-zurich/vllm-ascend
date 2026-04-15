/**
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */

/*******************************************************************************
 *  quest_prefill_metadata_kernel - vector-core, 1 core = (batch, head)
 *  Loads each KV-block ONCE, keeps copy, reduces min & max logarithmically
 *******************************************************************************/
#include "kernel_operator.h"
#include "quest_prefill_metadata_tilingkey.h"

#define DOUBLEBUFFER 2
#define SINGLEBUFFER 1

constexpr int32_t BYTES_UB_BLOCK = 32;
constexpr int32_t BYTES_DATA_BLOCK = 32;

inline __aicore__ int32_t ceilDiv(int32_t x, int32_t d) { return (x + d - 1) / d; }
inline __aicore__ int32_t ceilDivMul(int32_t x, int32_t d) { return d * ((x + d - 1) / d); }

using namespace AscendC;

// QuestPrefillMetadataTilingData is generated from the op_host tiling
// definition. The kernel must not redeclare it locally.

template <typename A, typename B>
struct quest_is_same {
    static constexpr bool value = false;
};

template <typename A>
struct quest_is_same<A, A> {
    static constexpr bool value = true;
};

template <typename T>
class KernelQuestMetadata {
public:
    __aicore__ inline KernelQuestMetadata() {}

    __aicore__ void Init(
        GM_ADDR k_cache,
        GM_ADDR block_tables,
        GM_ADDR seq_lens,
        GM_ADDR metadata_block_tables,
        GM_ADDR maxblocks,
        GM_ADDR minblocks,
        int32_t batch_size,
        int32_t num_kv_heads,
        int32_t block_size,
        int32_t head_dim,
        int32_t max_kv_blocks_per_request,
        int32_t max_metadata_blocks_per_request)
    {
        batch_size_ = batch_size;
        num_kv_heads_ = num_kv_heads;
        block_size_ = block_size;
        head_dim_ = head_dim;
        max_kv_blocks_per_request_ = max_kv_blocks_per_request;
        max_metadata_blocks_per_request_ = max_metadata_blocks_per_request;

        k_cache_gm_.SetGlobalBuffer((__gm__ T *)k_cache);
        block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)block_tables);
        seq_lens_gm_.SetGlobalBuffer((__gm__ int32_t *)seq_lens);
        metadata_block_tables_gm_.SetGlobalBuffer((__gm__ int32_t *)metadata_block_tables);
        maxblocks_gm_.SetGlobalBuffer((__gm__ T *)maxblocks);
        minblocks_gm_.SetGlobalBuffer((__gm__ T *)minblocks);

        int32_t tile_bytes =
            ceilDivMul(block_size_ * head_dim_ * static_cast<int32_t>(sizeof(T)), BYTES_UB_BLOCK);
        pipe_.InitBuffer(k_block_in_q_, DOUBLEBUFFER, tile_bytes);
        if constexpr (quest_is_same<T, bfloat16_t>::value) {
            int32_t work_tile_bytes =
                ceilDivMul(3 * head_dim_ * static_cast<int32_t>(sizeof(float)), BYTES_UB_BLOCK);
            pipe_.InitBuffer(work_calc_buf_, work_tile_bytes);
        } else {
            pipe_.InitBuffer(work_calc_q_, SINGLEBUFFER, tile_bytes);
        }
        pipe_.InitBuffer(max_out_q_, SINGLEBUFFER, tile_bytes);
        pipe_.InitBuffer(min_out_q_, SINGLEBUFFER, tile_bytes);
    }

    __aicore__ void Process()
    {
        int32_t num_blocks = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        int32_t num_batch_heads = batch_size_ * num_kv_heads_;

        for (int32_t batch_head_idx = GetBlockIdx(); batch_head_idx < num_batch_heads;
             batch_head_idx += num_blocks) {
            int32_t request_idx = batch_head_idx / num_kv_heads_;
            int32_t head_idx = batch_head_idx % num_kv_heads_;

            int32_t seq_len = seq_lens_gm_.GetValue(request_idx);
            int32_t num_kv_blocks_in_request = ceilDiv(seq_len, block_size_);
            int32_t num_meta_blocks_in_request = ceilDiv(num_kv_blocks_in_request, block_size_);

            for (int32_t meta_block = 0; meta_block < num_meta_blocks_in_request; meta_block++) {
                LocalTensor<T> max_lt = max_out_q_.AllocTensor<T>();
                LocalTensor<T> min_lt = min_out_q_.AllocTensor<T>();

                int32_t completed_kv_blocks = meta_block * block_size_;
                int32_t kv_blocks_this_iter = num_kv_blocks_in_request - completed_kv_blocks;
                if (kv_blocks_this_iter > block_size_) {
                    kv_blocks_this_iter = block_size_;
                }
                for (int32_t kv_block_offset = 0; kv_block_offset < kv_blocks_this_iter;
                     ++kv_block_offset) {
                    int32_t tokens_to_reduce;
                    if ((kv_block_offset == kv_blocks_this_iter - 1) &&
                        (meta_block == num_meta_blocks_in_request - 1)) {
                        int32_t reduced_tokens_so_far =
                            (meta_block * block_size_ + kv_block_offset) * block_size_;
                        tokens_to_reduce = seq_len - reduced_tokens_so_far;
                    } else {
                        tokens_to_reduce = block_size_;
                    }

                    int32_t kv_block_id = block_tables_gm_.GetValue(
                        request_idx * max_kv_blocks_per_request_ + completed_kv_blocks + kv_block_offset);
                    int32_t kv_block_offset_gm =
                        (kv_block_id * block_size_ * num_kv_heads_ * head_dim_) + head_idx * head_dim_;

                    LocalTensor<T> k_block_lt = k_block_in_q_.AllocTensor<T>();
                    DataCopyParams gm_ub_cp;
                    gm_ub_cp.blockCount = tokens_to_reduce;
                    gm_ub_cp.blockLen =
                        ceilDiv(head_dim_ * static_cast<int32_t>(sizeof(T)), BYTES_DATA_BLOCK);
                    gm_ub_cp.srcStride = ceilDiv(
                        (num_kv_heads_ - 1) * head_dim_ * static_cast<int32_t>(sizeof(T)),
                        BYTES_DATA_BLOCK);
                    gm_ub_cp.dstStride = 0;
                    DataCopy(k_block_lt, k_cache_gm_[kv_block_offset_gm], gm_ub_cp);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

                    if constexpr (quest_is_same<T, bfloat16_t>::value) {
                        LocalTensor<float> work_fp32_lt = work_calc_buf_.Get<float>();
                        LocalTensor<float> max_acc_lt = work_fp32_lt;
                        LocalTensor<float> min_acc_lt = work_fp32_lt[head_dim_];
                        LocalTensor<float> row_fp32_lt = work_fp32_lt[2 * head_dim_];

                        Cast(max_acc_lt, k_block_lt, RoundMode::CAST_NONE, head_dim_);
                        Cast(min_acc_lt, k_block_lt, RoundMode::CAST_NONE, head_dim_);
                        AscendC::PipeBarrier<PIPE_V>();

                        for (int32_t token_idx = 1; token_idx < tokens_to_reduce; ++token_idx) {
                            Cast(
                                row_fp32_lt,
                                k_block_lt[token_idx * head_dim_],
                                RoundMode::CAST_NONE,
                                head_dim_);
                            AscendC::PipeBarrier<PIPE_V>();
                            Max(max_acc_lt, max_acc_lt, row_fp32_lt, head_dim_);
                            Min(min_acc_lt, min_acc_lt, row_fp32_lt, head_dim_);
                            AscendC::PipeBarrier<PIPE_V>();
                        }

                        Cast(
                            max_lt[kv_block_offset * head_dim_],
                            max_acc_lt,
                            RoundMode::CAST_RINT,
                            head_dim_);
                        Cast(
                            min_lt[kv_block_offset * head_dim_],
                            min_acc_lt,
                            RoundMode::CAST_RINT,
                            head_dim_);
                        AscendC::PipeBarrier<PIPE_V>();
                    } else {
                        uint64_t mask = head_dim_;
                        CopyRepeatParams ub_ub_cp = {1, 1, 8, 8};
                        LocalTensor<T> work_lt = work_calc_q_.AllocTensor<T>();
                        Copy(work_lt, k_block_lt, mask, tokens_to_reduce, ub_ub_cp);
                        ReduceTokenDim<T, true>(work_lt, tokens_to_reduce * head_dim_);
                        Copy(max_lt[kv_block_offset * head_dim_], work_lt, mask, 1, ub_ub_cp);

                        Copy(work_lt, k_block_lt, mask, tokens_to_reduce, ub_ub_cp);
                        ReduceTokenDim<T, false>(work_lt, tokens_to_reduce * head_dim_);
                        Copy(min_lt[kv_block_offset * head_dim_], work_lt, mask, 1, ub_ub_cp);
                        work_calc_q_.FreeTensor(work_lt);
                    }
                    k_block_in_q_.FreeTensor(k_block_lt);
                }

                int32_t unused_metadata_rows = block_size_ - kv_blocks_this_iter;
                if (unused_metadata_rows > 0) {
                    Duplicate<T>(
                        max_lt[kv_blocks_this_iter * head_dim_],
                        static_cast<T>(0),
                        unused_metadata_rows * head_dim_);
                    Duplicate<T>(
                        min_lt[kv_blocks_this_iter * head_dim_],
                        static_cast<T>(0),
                        unused_metadata_rows * head_dim_);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

                int32_t meta_block_id = metadata_block_tables_gm_.GetValue(
                    request_idx * max_metadata_blocks_per_request_ + meta_block);
                int32_t meta_offset =
                    (meta_block_id * block_size_ * num_kv_heads_ * head_dim_) + head_idx * head_dim_;

                DataCopyParams ub_gm_cp;
                ub_gm_cp.blockCount = block_size_;
                ub_gm_cp.blockLen =
                    ceilDiv(head_dim_ * static_cast<int32_t>(sizeof(T)), BYTES_UB_BLOCK);
                ub_gm_cp.srcStride = 0;
                ub_gm_cp.dstStride = ceilDiv(
                    (num_kv_heads_ - 1) * head_dim_ * static_cast<int32_t>(sizeof(T)),
                    BYTES_UB_BLOCK);
                DataCopy(maxblocks_gm_[meta_offset], max_lt, ub_gm_cp);
                DataCopy(minblocks_gm_[meta_offset], min_lt, ub_gm_cp);

                max_out_q_.FreeTensor(max_lt);
                min_out_q_.FreeTensor(min_lt);
            }
        }
    }

private:
    template <typename ElementT, bool isMax>
    __aicore__ void ReduceTokenDim(LocalTensor<ElementT> vec_lt, int32_t initial_length)
    {
        if (initial_length != block_size_ * head_dim_) {
            AscendC::PipeBarrier<PIPE_V>();
        }

        int32_t len = initial_length;
        while (len > head_dim_) {
            int32_t num_vec = len / head_dim_;
            int32_t pair_vec = num_vec >> 1;
            int32_t has_tail = num_vec & 1;
            int32_t reduce_len = pair_vec * head_dim_;

            if (reduce_len > 0) {
                if (isMax) {
                    Max(vec_lt[0], vec_lt[0], vec_lt[reduce_len], reduce_len);
                } else {
                    Min(vec_lt[0], vec_lt[0], vec_lt[reduce_len], reduce_len);
                }
            }

            if (has_tail) {
                Copy(
                    vec_lt[reduce_len],
                    vec_lt[(num_vec - 1) * head_dim_],
                    head_dim_,
                    1,
                    {1, 1, 8, 8});
                reduce_len += head_dim_;
            }

            len = reduce_len;
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, DOUBLEBUFFER> k_block_in_q_;
    TBuf<TPosition::VECCALC> work_calc_buf_;
    TQue<TPosition::VECCALC, SINGLEBUFFER> work_calc_q_;
    TQue<TPosition::VECOUT, SINGLEBUFFER> max_out_q_;
    TQue<TPosition::VECOUT, SINGLEBUFFER> min_out_q_;

    GlobalTensor<T> k_cache_gm_;
    GlobalTensor<T> maxblocks_gm_;
    GlobalTensor<T> minblocks_gm_;
    GlobalTensor<int32_t> block_tables_gm_;
    GlobalTensor<int32_t> seq_lens_gm_;
    GlobalTensor<int32_t> metadata_block_tables_gm_;

    int32_t batch_size_;
    int32_t num_kv_heads_;
    int32_t block_size_;
    int32_t head_dim_;
    int32_t max_kv_blocks_per_request_;
    int32_t max_metadata_blocks_per_request_;
};

template <typename T>
__aicore__ inline void RunQuestPrefillMetadata(
    GM_ADDR k_cache,
    GM_ADDR block_tables,
    GM_ADDR seq_lens,
    GM_ADDR metadata_block_tables,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    const QuestPrefillMetadataTilingData *tiling_data)
{
    KernelQuestMetadata<T> op;
    op.Init(
        k_cache,
        block_tables,
        seq_lens,
        metadata_block_tables,
        maxblocks,
        minblocks,
        static_cast<int32_t>(tiling_data->batchSize),
        static_cast<int32_t>(tiling_data->numKvHeads),
        static_cast<int32_t>(tiling_data->blockSize),
        static_cast<int32_t>(tiling_data->headDim),
        static_cast<int32_t>(tiling_data->maxKvBlocksPerRequest),
        static_cast<int32_t>(tiling_data->maxMetadataBlocksPerRequest));
    op.Process();
}

extern "C" __global__ __aicore__ void quest_prefill_metadata(
    GM_ADDR k_cache,
    GM_ADDR block_tables,
    GM_ADDR seq_lens,
    GM_ADDR metadata_block_tables,
    GM_ADDR maxblocks,
    GM_ADDR minblocks,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;
    QUEST_PREFILL_METADATA_COPY_TILING_DATA(QuestPrefillMetadataTilingData, tiling);

    #define QUEST_SKIP_BF16_PATH
    // Reduction operators for MIN and MAX currently do not support BF16
    // As a result, the BF16 execution path is 5x-10x slower than the fp16 counterpart
    // However, given that these kernels only use MIN and MAX operators, we can
    // simply reinterpret the bit representation of BF16 as fp16 and call the kernel
    // for the wrong type, while still getting the correct results.
    // This is a HACK and should be fixed once a better solution is found.

    #if defined(QUEST_SKIP_BF16_PATH)
    if (TILING_KEY_IS(QUEST_PREFILL_METADATA_TILING_FP16) ||
        TILING_KEY_IS(QUEST_PREFILL_METADATA_TILING_BF16)) {
        RunQuestPrefillMetadata<half>(
            k_cache,
            block_tables,
            seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
            tiling_data);
        return;
    }
    #else
    if (TILING_KEY_IS(QUEST_PREFILL_METADATA_TILING_FP16)) {
        RunQuestPrefillMetadata<half>(
            k_cache,
            block_tables,
            seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
            tiling_data);
        return;
    }


    if (TILING_KEY_IS(QUEST_PREFILL_METADATA_TILING_BF16)) {
        RunQuestPrefillMetadata<bfloat16_t>(
            k_cache,
            block_tables,
            seq_lens,
            metadata_block_tables,
            maxblocks,
            minblocks,
            tiling_data);
        return;
    }
    #endif

    ASSERT(false && "Unsupported quest_prefill_metadata tiling key.");
}
