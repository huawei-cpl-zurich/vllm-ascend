/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_regular.h
 * \brief
 */
#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

#include "kernel_common.hpp"

using namespace NpuArch;
using namespace KernelCommon;

namespace SplitFuse {
    template <
        class BlockMmadQK,
        class BlockMmadPV,
        class EpilogueOnlineSoftmax,
        class EpilogueRescaleO,
        class EpilogueInitOut,
        bool PAGED_CACHE_FLAG,
        FaiKernel::MaskType MASK_TYPE = FaiKernel::MaskType::NO_MASK,
        FaiKernel::inputLayout INPUT_LAYOUT = FaiKernel::inputLayout::BSND>
    class FAInferKernel {
    public:
        using ArchTag = typename BlockMmadQK::ArchTag;
        using L1TileShape = typename BlockMmadQK::L1TileShape;
        using ElementQ = typename BlockMmadQK::ElementA;
        using LayoutQ = typename BlockMmadQK::LayoutA;
        using ElementK = typename BlockMmadQK::ElementB;
        using LayoutK = typename BlockMmadQK::LayoutB;
        using ElementS = typename BlockMmadQK::ElementC;
        using LayoutS = typename BlockMmadQK::LayoutC;

        using ElementP = typename BlockMmadPV::ElementA;
        using LayoutP = typename BlockMmadPV::LayoutA;
        using ElementV = typename BlockMmadPV::ElementB;
        using LayoutV = typename BlockMmadPV::LayoutB;

        using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
        using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

        using ElementO = typename EpilogueRescaleO::ElementOutput;
        using LayoutO = typename EpilogueRescaleO::LayoutOutput;

        using ElementOTmp = typename EpilogueRescaleO::ElementInput;
        using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;

        using ElementLse = typename EpilogueRescaleO::ElementLse;
        using LayoutLse = typename EpilogueRescaleO::LayoutLse;

        using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
        using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

        static constexpr Epilogue::LseMode LSE_MODE = EpilogueRescaleO::LSE_MODE;
        static constexpr Epilogue::SinkMode SINK_MODE = EpilogueOnlineSoftmax::SINK_MODE;

        // Methods
        __aicore__ inline
        FAInferKernel() {}

        __aicore__ inline
        void operator()(FAIKernelParams const &params)
        {
            __gm__ FAInferTilingData *fATilingData = reinterpret_cast<__gm__ FAInferTilingData *>(params.tiling);
            uint64_t mm1OutSize = fATilingData->mm1OutSize;
            uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;
            uint64_t mm2OutSize = fATilingData->mm2OutSize;
            uint32_t batch = fATilingData->batch;
            uint32_t qHeads = fATilingData->numHeads;
            uint32_t kvHeads = fATilingData->kvHeads;
            uint32_t embed = fATilingData->embeddingSize;
            uint32_t embedV = fATilingData->embeddingSizeV;
            uint32_t pagedBlockSize = fATilingData->blockSize;
            uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
            uint32_t firstBatchTaskNum = fATilingData->firstBatchTaskNum;
            uint32_t totalTaskNum = fATilingData->totalTaskNum;
            uint32_t blockSize = fATilingData->blockSize;
            uint32_t maskType = fATilingData->maskType;
            float scaleValue = fATilingData->scaleValue;

            AscendC::GlobalTensor<ElementQ> gQ;
            gQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
            AscendC::ListTensorDesc keyListTensorDescInit((__gm__ void*)params.k);
            AscendC::ListTensorDesc valueListTensorDescInit((__gm__ void*)params.v);
            __gm__ uint8_t* currentKey = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            __gm__ uint8_t* currentValue = (__gm__ uint8_t*)valueListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
            AscendC::GlobalTensor<ElementK> gK;
            gK.SetGlobalBuffer((__gm__ ElementK *)currentKey);
            AscendC::GlobalTensor<ElementK> gV;
            gV.SetGlobalBuffer((__gm__ ElementK *)currentValue);
            AscendC::GlobalTensor<ElementMask> gMask;
            gMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
            AscendC::GlobalTensor<int32_t> gBlockTable;
            gBlockTable.SetGlobalBuffer((__gm__ int32_t *)(params.blockTables));
            AscendC::GlobalTensor<int64_t> gActualQseqlen;
            gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualQseqlen);
            AscendC::GlobalTensor<int64_t> gActualKvseqlen;
            gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualKvseqlen);
            AscendC::GlobalTensor<ElementO> gO;
            gO.SetGlobalBuffer((__gm__ ElementO *)params.o);
            AscendC::GlobalTensor<ElementLse> gLse;
            gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
            AscendC::GlobalTensor<ElementS> gS;
            gS.SetGlobalBuffer((__gm__ ElementS *)(params.workSpace));
            AscendC::GlobalTensor<ElementP> gP;
            gP.SetGlobalBuffer((__gm__ ElementP *)(params.workSpace + mm1OutSize));
            AscendC::GlobalTensor<ElementOTmp> gOTmp;
            gOTmp.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace + mm1OutSize + smOnlineOutSize));
            AscendC::GlobalTensor<ElementOTmp> gOUpdate;
            gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp *)(params.workSpace +
                mm1OutSize + smOnlineOutSize + mm2OutSize));
            AscendC::GlobalTensor<bfloat16_t> gSink;
            gSink.SetGlobalBuffer((__gm__ bfloat16_t *)(params.sink));

            uint32_t coreIdx = AscendC::GetBlockIdx();
            uint32_t coreNum = AscendC::GetBlockNum();
#ifdef __DAV_C220_CUBE__
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            
            uint32_t kDynNum = NpuArch::Detail::Alignment::RoundUp(embed, NUM_128);
            kDynNum = kDynNum < NUM_256 ? NUM_256 : kDynNum;
            uint32_t maxQKPL1Size = L1_MAX_SIZE - embedV * MAX_KV_STACK_LEN * sizeof(ElementV);
            uint32_t maxQL1Size = Q_TILE_CEIL * kDynNum * sizeof(ElementQ);
            uint32_t maxNDynNum =
                ((maxQKPL1Size - maxQL1Size) / kDynNum / sizeof(ElementV) / DOUBLE_BUFFER) / NUM_32 * NUM_32;

            uint32_t nDynNum = maxNDynNum < L1_MAX_N_NUM ? maxNDynNum : L1_MAX_N_NUM;
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ?
                NpuArch::Detail::Alignment::RoundDown((nDynNum - 1), NUM_32) : nDynNum;

            uint32_t L1_QK_SIZE = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);
            BlockMmadQK blockMmadQK(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M;
            BlockMmadPV blockMmadPV(resource, nDynNum, kPVDynNum, MAX_KV_STACK_LEN, L1_QK_SIZE);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);

            EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue);
            EpilogueRescaleO epilogueRescaleO(resource);
            EpilogueInitOut epilogueInitOut(resource);

            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
#endif
            uint64_t strideQ = static_cast<uint64_t>(qHeads * embed);
            uint64_t strideO = static_cast<uint64_t>(qHeads * embedV);
            uint64_t strideK = static_cast<uint64_t>(kvHeads * embed);
            uint64_t strideV = static_cast<uint64_t>(kvHeads * embedV);
            uint32_t embedRound = NpuArch::Detail::Alignment::RoundUp(embed, FaiKernel::BLOCK_SIZE);
            uint32_t embedRoundV = NpuArch::Detail::Alignment::RoundUp(embedV, FaiKernel::BLOCK_SIZE);
            uint32_t groupSize = qHeads / kvHeads;

            uint64_t qBOffset = 0;
            uint64_t kBOffset = 0;
            uint64_t vBOffset = 0;
            uint64_t oBOffset = 0;
            uint64_t lseBOffset = 0;
            uint64_t blockBOffset = 0;

            uint32_t preTotalTaskNum = 0;
            uint32_t curBatch = 0;
            uint32_t totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch - 1));
            uint32_t qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
            uint32_t kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
            if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                uint32_t prevQSeqlenSum = (curBatch == 0) ?
                    0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch - 1));
                qSeqlen = qSeqlen - prevQSeqlenSum;
                if constexpr (!PAGED_CACHE_FLAG) {
                    uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                        0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                    kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                }
            }
            uint32_t curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
            uint32_t qNBlockNumPerGroup = NpuArch::Detail::Alignment::CeilDiv(groupSize, curQNBlockTile);
            uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
            uint32_t curQSBlockTile = GetQSBlockTile(kvSeqlen);
            uint32_t curQSBlockNum = NpuArch::Detail::Alignment::CeilDiv(qSeqlen, curQSBlockTile);
            uint32_t curTotalTaskNum = firstBatchTaskNum;

            //  prepare for addding sink
            bool isLastStackTile = false;
            // Go through each task.
            for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += uint32_t(coreNum)) {
                isLastStackTile = false;
                // Get the offset of each core on the GM.
                while (taskIdx >= curTotalTaskNum) {
                    ++curBatch;
                    preTotalTaskNum = curTotalTaskNum;
                    qBOffset += qSeqlen * strideQ;
                    if constexpr (!PAGED_CACHE_FLAG) {
                        kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
                        vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
                    } else {
                        blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch);
                    }
                    oBOffset += static_cast<uint64_t>(qSeqlen * strideO);
                    lseBOffset += static_cast<uint64_t>(qSeqlen * qHeads);

                    qSeqlen = static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch));
                    kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                    if constexpr(INPUT_LAYOUT == FaiKernel::inputLayout::TND) {
                        uint32_t prevQSeqlenSum = (curBatch == 0) ?
                            0 : static_cast<uint32_t>(gActualQseqlen.GetValue(curBatch - 1));
                        qSeqlen = qSeqlen - prevQSeqlenSum;
                        if constexpr (!PAGED_CACHE_FLAG) {
                            uint32_t prevKvSeqlenSum = (curBatch == 0) ?
                                0 : static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch - 1));
                            kvSeqlen = kvSeqlen - prevKvSeqlenSum;
                        }
                    }
                    curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);
                    qNBlockNumPerGroup = NpuArch::Detail::Alignment::CeilDiv(groupSize, curQNBlockTile);
                    curQNBlockNum = qNBlockNumPerGroup * kvHeads;
                    curQSBlockTile = GetQSBlockTile(kvSeqlen);
                    curQSBlockNum = NpuArch::Detail::Alignment::CeilDiv(qSeqlen, curQSBlockTile);
                    curTotalTaskNum += curQNBlockNum * curQSBlockNum;
                }
                uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
                uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;
                uint32_t qNBlockIdx = taskIdxCurBatch - qSBlockIdx * curQNBlockNum;
                uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;

                uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;
                uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
                uint32_t lseTokenOffset = qSBlockIdx * curQSBlockTile * qHeads;
                uint64_t gmOffsetSink =  qNStartIdx;

                uint64_t gmOffsetQ = qBOffset +
                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ +
                    static_cast<uint64_t>(qNStartIdx * embed);
                uint64_t gmOffsetK = kBOffset + static_cast<uint64_t>(kvNIdx * embed);
                uint64_t gmOffsetV = vBOffset + static_cast<uint64_t>(kvNIdx * embedV);
                uint64_t gmOffsetO = oBOffset +
                    static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO +
                    static_cast<uint64_t>(qNStartIdx * embedV);
                uint64_t gmOffsetLse = lseBOffset +
                    static_cast<uint64_t>(lseTokenOffset + qNStartIdx);

                uint32_t qSBlockSize = (qSBlockIdx == (curQSBlockNum - 1U)) ?
                    (qSeqlen - qSBlockIdx * curQSBlockTile) : curQSBlockTile;
                uint32_t qNBlockSize = (qNBlockIdxCurGroup == (qNBlockNumPerGroup - 1U)) ?
                    (groupSize - qNBlockIdxCurGroup * curQNBlockTile) : curQNBlockTile;
                uint32_t rowNum = qSBlockSize * qNBlockSize;
                uint32_t rowNumRound = NpuArch::Detail::Alignment::RoundUp(rowNum, FaiKernel::BLOCK_SIZE);

                int64_t noSkipKvS = static_cast<int64_t>(kvSeqlen);
                if (maskType != 0U) {
                    int64_t diffS = kvSeqlen - qSeqlen;
                    diffS = (diffS < 0) ? 0 : diffS;
                    noSkipKvS = (qSBlockIdx + 1U) * curQSBlockTile + diffS;
                    noSkipKvS = AscendC::Std::min(static_cast<int64_t>(kvSeqlen), noSkipKvS);
                }
                uint32_t kvSLoopNumTotal = NpuArch::Detail::Alignment::CeilDiv(noSkipKvS, MAX_KV_STACK_LEN);

                uint32_t blockStackNum = (MAX_KV_STACK_LEN - 1 + pagedBlockSize) / pagedBlockSize;
                uint32_t stackSeqTile = MAX_KV_STACK_LEN;
                uint32_t stackSeqTilePad = MAX_KV_STACK_LEN;
                uint32_t preKVNum = PRE_LAUNCH;
                int32_t stackSeqCount = 0;
#ifdef __DAV_C220_VEC__
                if (kvSLoopNumTotal <= 0) {
                    LayoutO layoutO(qSeqlen, embed * qHeads);
                    LayoutLse layoutLse(totalQTokens, qHeads);
                    epilogueInitOut(gO[gmOffsetO], gLse[gmOffsetLse], layoutO, layoutLse, qSBlockSize, qNBlockSize);
                }
#endif
#ifdef __DAV_C220_CUBE__
                LayoutQ layoutQTemp(rowNum, embed);
                LayoutK layoutKTemp(strideK, stackSeqTile);
                LayoutV layoutVTemp(stackSeqTile, strideV);
                blockMmadQK.resetBlockStart();
                blockMmadPV.resetBlockStart();
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTemp, rowNum, qNBlockSize, qHeads);
#endif
                for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + preKVNum; kvSIdx ++) {
                    if (kvSIdx < kvSLoopNumTotal) {
                        if (kvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                        } else {
                            stackSeqTile = MAX_KV_STACK_LEN;
                        }
                        isLastStackTile = (kvSIdx + 1) >= kvSLoopNumTotal;
                        uint32_t curStackTileMod = stackSeqCount % (PRE_LAUNCH + 1U);
                        uint64_t gmOffsetS =
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                        LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
#ifdef __DAV_C220_CUBE__
                        if constexpr (PAGED_CACHE_FLAG) {
                            blockMmadQK(
                                gQ[gmOffsetQ],
                                gK[gmOffsetK],
                                gS[gmOffsetS],
                                gBlockTable[blockBOffset],
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK);
                        } else {
                            blockMmadQK(
                                gQ[gmOffsetQ],
                                gK[gmOffsetK],
                                gS[gmOffsetS],
                                gBlockTable,
                                layoutQTemp,
                                layoutKTemp,
                                layOutS,
                                actualBlockShapeQK,
                                kvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                strideK);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_C220_VEC__
                        LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
                        LayoutMask layOutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
                        uint64_t gmOffsetP = gmOffsetS;
                        // causal mask的左上起点
                        uint32_t triUp = noSkipKvS - qSBlockSize;
                        // causal mask的右下止点
                        uint32_t triDown = noSkipKvS;
                        uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;
                        uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                        // 在causal mask场景下，由mask的左上起点判断当前基块是否需要加mask
                        // 如果实际加mask长度只有1，那么相当于不加mask（主对角线需要被计算）
                        bool doTriUMask = triUp < kvSEndIdx - 1;
                        if constexpr (MASK_TYPE == FaiKernel::MaskType::MASK_CAUSAL) {
                            if (doTriUMask) {
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gSink[gmOffsetSink],
                                    gMask,
                                    layOutP,
                                    layOutS,
                                    layOutMask,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    qkReady,
                                    triUp,
                                    triDown,
                                    kvSStartIdx,
                                    kvSEndIdx,
                                    isLastStackTile);
                            } else {
                                uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                                Arch::CrossCoreWaitFlag(qkReady);
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP],
                                    gS[gmOffsetS],
                                    gSink[gmOffsetSink],
                                    layOutP,
                                    layOutS,
                                    actualBlockShapeQK,
                                    (stackSeqCount == 0),
                                    (stackSeqCount == noMaskStackSeqNum - 1),
                                    qSBlockSize,
                                    qNBlockSize,
                                    curStackTileMod,
                                    isLastStackTile);
                            }
                        } else {
                            Arch::CrossCoreWaitFlag(qkReady);
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP],
                                gS[gmOffsetS],
                                gSink[gmOffsetSink],
                                layOutP,
                                layOutS,
                                actualBlockShapeQK,
                                (stackSeqCount == 0),
                                0,
                                qSBlockSize,
                                qNBlockSize,
                                curStackTileMod,
                                isLastStackTile);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
                    }
                    if (kvSIdx >= preKVNum) {
                        uint32_t nowkvSIdx = kvSIdx - preKVNum;
                        if (nowkvSIdx + 1 > kvSLoopNumTotal - 1U) {
                            stackSeqTile = noSkipKvS - nowkvSIdx * MAX_KV_STACK_LEN;
                        } 
                        else {
                            stackSeqTile = MAX_KV_STACK_LEN;
                        }
                        uint32_t curStackTileMod = (stackSeqCount - PRE_LAUNCH) % (PRE_LAUNCH + 1U);
                        uint64_t gmOffsetOTmp =
                            static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1U) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB);
                        GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                        LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);
#ifdef __DAV_C220_CUBE__
                        LayoutP layoutPTemp(rowNum, stackSeqTile, stackSeqTilePad);
                        uint64_t gmOffsetP = coreIdx * WORKSPACE_BLOCK_SIZE_DB * (PRE_LAUNCH + 1) +
                            curStackTileMod * WORKSPACE_BLOCK_SIZE_DB;;
                        if constexpr (PAGED_CACHE_FLAG) {
                            blockMmadPV(
                                gP[gmOffsetP],
                                gV[gmOffsetV],
                                gOTmp[gmOffsetOTmp],
                                gBlockTable[blockBOffset],
                                layoutPTemp,
                                layoutVTemp,
                                layoutOTmp,
                                actualBlockShapePV,
                                nowkvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                noSkipKvS,
                                strideV,
                                blockStackNum,
                                softmaxReady);
                        } else {
                            blockMmadPV(
                                gP[gmOffsetP],
                                gV[gmOffsetV],
                                gOTmp[gmOffsetOTmp],
                                gBlockTable,
                                layoutPTemp,
                                layoutVTemp,
                                layoutOTmp,
                                actualBlockShapePV,
                                nowkvSIdx,
                                kvSLoopNumTotal,
                                pagedBlockSize,
                                noSkipKvS,
                                strideV,
                                blockStackNum,
                                softmaxReady);
                        }
                        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_C220_VEC__
                        LayoutO layoutO(qSeqlen, embed * qHeads);
                        LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                        LayoutLse layoutLse(totalQTokens, qHeads);
                        uint64_t gmOffsetUpdate = (uint64_t)(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                        Arch::CrossCoreWaitFlag(pvReady);
                        // rescale O
                        epilogueRescaleO(
                            gO[gmOffsetO],
                            gOTmp[gmOffsetOTmp],
                            gOUpdate[gmOffsetUpdate],
                            gLse[gmOffsetLse],
                            layoutO,
                            layoutOTmp,
                            layoutUpdate,
                            layoutLse,
                            actualBlockShapePV,
                            qSBlockSize,
                            qNBlockSize,
                            (stackSeqCount - PRE_LAUNCH == 0),
                            nowkvSIdx + 1 >= kvSLoopNumTotal,
                            curStackTileMod);
#endif
                    }
                    stackSeqCount++;
                }
            }
#ifdef __DAV_C220_CUBE__
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
#ifdef __DAV_C220_VEC__
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
#endif
            AscendC::PipeBarrier<PIPE_ALL>();
        }

    private:
        Arch::Resource<ArchTag> resource;
        Arch::CrossCoreFlag qkReady{QK_READY_ID};
        Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
        Arch::CrossCoreFlag pvReady{PV_READY_ID};
    };
}
#endif
