/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_INFER_TILING_H
#define FLASH_ATTN_INFER_TILING_H
#include "exe_graph/runtime/tiling_context.h"
#include "register/tilingdata_base.h"
#include "select_infer_attention_score_tiling.h"

namespace optiling{
    BEGIN_TILING_DATA_DEF(FAInferTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, numHeads)
    TILING_DATA_FIELD_DEF(uint32_t, embeddingSize)
    TILING_DATA_FIELD_DEF(uint32_t, embeddingSizeV)
    TILING_DATA_FIELD_DEF(uint32_t, numBlocks)
    TILING_DATA_FIELD_DEF(uint32_t, blockSize)
    TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen)
    TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen)
    TILING_DATA_FIELD_DEF(uint32_t, kvHeads)
    TILING_DATA_FIELD_DEF(uint32_t, batch)
    TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch)
    TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum)
    TILING_DATA_FIELD_DEF(uint32_t, maskType)
    TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize)
    TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize)
    TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize)
    TILING_DATA_FIELD_DEF(uint64_t, UpdateSize)
    TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize)
    TILING_DATA_FIELD_DEF(float, scaleValue)
    TILING_DATA_FIELD_DEF(uint64_t, padding1)
    TILING_DATA_FIELD_DEF(uint64_t, padding2)
    TILING_DATA_FIELD_DEF(uint32_t, padding3)
    END_TILING_DATA_DEF
    
    const uint32_t SIZE_OF_16BIT = 2;
    const uint32_t SIZE_OF_32BIT = 4;
    const uint32_t N_SPLIT_HELPER = 2;
    const uint32_t MAX_KV_STACK_LEN = 512;
    const uint32_t Q_TILE_CEIL = 128;
    const uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
    const uint32_t BASE_KV_SIZE = 128;
    const uint32_t PRELANCH_NUM = 3;

    enum class MaskType : uint32_t {
        NO_MASK = 0,
        MASK_SPEC = 1
    };

    enum class DataType : uint32_t {
        FP16 = 0,
        BF16 = 1
    };

    struct FAInferContext {
        int32_t numTokens = 0;
        int32_t numHeads = 0;
        int32_t embeddingSize = 0;
        int32_t embeddingSizeV = 0;
        int32_t numBlocks = 0;
        int32_t blockSize = 0;
        int32_t kvHeads = 0;
        int32_t batch = 0;
        int32_t innerPrecise = 0;
        int64_t maxQSeqlen = 0;
        int64_t maxKvSeqlen = 0;
        uint32_t maxNumBlocksPerBatch = 0;
        const int64_t *qSeqlenList{nullptr};
        const int64_t *kvSeqlenList{nullptr};
        float scaleValue = 0.0;
        size_t* workspaces{nullptr};
        MaskType maskType = MaskType::MASK_SPEC;
        DataType dataType = DataType::FP16;
        bool pagedCacheFlag = false;
        bool lseFlag = false;
        bool learnableSinkFlag = false;
        bool isTilingSink = false;
        string layout;
    };

    class FAInferTiling {
    public:
        FAInferTiling() = default;
        explicit FAInferTiling(const FAInferContext &faInfo);
        ge::graphStatus DoTiling(FAInferTilingData &tilingdata);
        void SetCoreNum(uint32_t blockNum) {
            this->blockNum_ = blockNum;
        }
        uint32_t GetCoreNum() {
            return this->blockNum_; 
        }
        uint64_t GetTilingKey();
    private:
        void FillSplitCoreTilingData(FAInferTilingData &tilingdata);
        void FillWorkSpaceTilingData(FAInferTilingData &faTilingData);
        uint32_t GetQSBlockTile(int64_t kvSeqlen);
        uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize);
        void FillBasicTilingData(FAInferTilingData &faTilingData);
    private:
        FAInferContext faInfo_;
        uint32_t blockNum_;
    };

    FAInferTiling::FAInferTiling(const FAInferContext &faInfo): faInfo_(faInfo) {}

    uint32_t FAInferTiling::GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
    {
        uint32_t qRowNumCeil = Q_TILE_CEIL;
        uint32_t qNBlockTile = (qSeqlen != 0) ?
            (qRowNumCeil / qSeqlen) / N_SPLIT_HELPER * N_SPLIT_HELPER : Q_TILE_CEIL;
        qNBlockTile = std::min(qNBlockTile, groupSize);
        qNBlockTile = std::max(qNBlockTile, static_cast<uint32_t>(1));
        return qNBlockTile;
    }

    uint32_t FAInferTiling::GetQSBlockTile(int64_t kvSeqlen)
    {
        uint32_t qSBlockTile = Q_TILE_CEIL;
        return qSBlockTile;
    }

    void FAInferTiling::FillBasicTilingData(FAInferTilingData &faTilingData)
    {
        faTilingData.set_batch(static_cast<uint32_t>(faInfo_.batch));
        faTilingData.set_numHeads(static_cast<uint32_t>(faInfo_.numHeads));
        faTilingData.set_kvHeads(static_cast<uint32_t>(faInfo_.kvHeads));

        faTilingData.set_embeddingSize(static_cast<uint32_t>(faInfo_.embeddingSize));
        faTilingData.set_embeddingSizeV(static_cast<uint32_t>(faInfo_.embeddingSizeV));
        faTilingData.set_numBlocks(static_cast<uint32_t>(faInfo_.numBlocks));
        if (faInfo_.pagedCacheFlag) {
            faTilingData.set_blockSize(static_cast<uint32_t>(faInfo_.blockSize));
        } else {
            faTilingData.set_blockSize(BASE_KV_SIZE);
        }
        faTilingData.set_maxQSeqlen(faInfo_.maxQSeqlen);
        faTilingData.set_maxKvSeqlen(faInfo_.maxKvSeqlen);
        faTilingData.set_maxNumBlocksPerBatch(faInfo_.maxNumBlocksPerBatch);
        faTilingData.set_maskType(static_cast<uint32_t>(faInfo_.maskType));
        faTilingData.set_scaleValue(faInfo_.scaleValue);
    }

    uint64_t FAInferTiling::GetTilingKey() 
    {
        constexpr uint64_t SPLIT_FUSE_BASE_KEY = 5000000000000000000;
        constexpr uint64_t PAGED_CACHE_KEY = 10000000;
        constexpr uint64_t COMP_CAUSAL_MASK_KEY = 3;
        constexpr uint64_t LAYOUTQ_TND_KEY = 200000;
        constexpr uint64_t DTYPE_FP16_KEY = 100;
        constexpr uint64_t DTYPE_BF16_KEY = 200;
        constexpr uint64_t LSE_OUT_ONLY_KEY = 1000;
        constexpr uint64_t INNER_LOW_PREC_KEY = 10000;
        constexpr uint64_t LEARNABLE_SINK_KEY = 100000000;
        uint64_t tilingKey = SPLIT_FUSE_BASE_KEY;
        if (faInfo_.pagedCacheFlag) {
            tilingKey += static_cast<uint64_t>(PAGED_CACHE_KEY);
        }
        if (faInfo_.maskType == MaskType::MASK_SPEC) {
            tilingKey += static_cast<uint64_t>(COMP_CAUSAL_MASK_KEY);
        }
        if (faInfo_.layout == "TND") {
            tilingKey += static_cast<uint64_t>(LAYOUTQ_TND_KEY);
        }
        if (faInfo_.dataType == DataType::FP16) {
            tilingKey += static_cast<uint64_t>(DTYPE_FP16_KEY);
        } else if (faInfo_.dataType == DataType::BF16) {
            tilingKey += static_cast<uint64_t>(DTYPE_BF16_KEY);
        }
        if (faInfo_.lseFlag) {
            tilingKey += static_cast<uint64_t>(LSE_OUT_ONLY_KEY);
        }
        if (faInfo_.learnableSinkFlag) {
            tilingKey += static_cast<uint64_t>(LEARNABLE_SINK_KEY);
        }
        if (faInfo_.innerPrecise == 1) {
            tilingKey += static_cast<uint64_t>(INNER_LOW_PREC_KEY);
        }
        return tilingKey;
    }

    void FAInferTiling::FillWorkSpaceTilingData(FAInferTilingData &faTilingData)
    {
        uint64_t mm1OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t smOnlineOutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_16BIT * PRELANCH_NUM;
        uint64_t mm2OutSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t UpdateSize = static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *
            SIZE_OF_32BIT * PRELANCH_NUM;
        uint64_t workSpaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;
        faTilingData.set_mm1OutSize(mm1OutSize);
        faTilingData.set_smOnlineOutSize(smOnlineOutSize);
        faTilingData.set_mm2OutSize(mm2OutSize);
        faTilingData.set_UpdateSize(UpdateSize);
        faTilingData.set_workSpaceSize(workSpaceSize);
    }

    void FAInferTiling::FillSplitCoreTilingData(FAInferTilingData &faTilingData)
    {
        uint32_t totalTaskNum = 0;
        uint32_t groupSize = faInfo_.numHeads / faInfo_.kvHeads;
        for (int32_t batchIdx = 0; batchIdx < faInfo_.batch; batchIdx++) {
            uint32_t qSeqlen = *(faInfo_.qSeqlenList + batchIdx);
            uint32_t kvSeqlen = *(faInfo_.kvSeqlenList + batchIdx);
            if (batchIdx > 0) {
                uint64_t prevQSeqlenSum = *(faInfo_.qSeqlenList + batchIdx - 1);
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if (!faInfo_.pagedCacheFlag) {
                    uint64_t prevKvSeqlenSum = *(faInfo_.kvSeqlenList + batchIdx - 1);
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = (groupSize + curQNBlockTile - 1) / curQNBlockTile;
            uint32_t curQNBlockNum = qNBlockNumPerGroup * faInfo_.kvHeads;
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = (qSeqlen + curQSBlockTile - 1) / curQSBlockTile;
            uint32_t curTaskNum = curQNBlockNum * curQSBlockNum;
            if (batchIdx == 0) {
                faTilingData.set_firstBatchTaskNum(curTaskNum);
            }
            totalTaskNum += curTaskNum;
        }
        faTilingData.set_totalTaskNum(totalTaskNum);
    }

    ge::graphStatus FAInferTiling::DoTiling(FAInferTilingData &tilingdata)
    {
        FillBasicTilingData(tilingdata);
        if (!faInfo_.isTilingSink) {
            FillSplitCoreTilingData(tilingdata);
        }
        FillWorkSpaceTilingData(tilingdata);
        return ge::GRAPH_SUCCESS;
    }
}
#endif