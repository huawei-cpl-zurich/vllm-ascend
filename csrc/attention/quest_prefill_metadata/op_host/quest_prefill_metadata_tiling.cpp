/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quest_prefill_metadata_tiling.h"
#include <algorithm>
#include "../op_kernel/quest_prefill_metadata_tilingkey.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t K_CACHE_INDEX = 0;
constexpr uint32_t BLOCK_TABLES_INDEX = 1;
constexpr uint32_t REFRESH_START_SEQ_LENS_INDEX = 2;
constexpr uint32_t REFRESH_END_SEQ_LENS_INDEX = 3;
constexpr uint32_t METADATA_BLOCK_TABLES_INDEX = 4;
constexpr uint32_t MAXBLOCKS_OUTPUT_INDEX = 0;
constexpr uint32_t MINBLOCKS_OUTPUT_INDEX = 1;
constexpr uint32_t K_CACHE_DIM_NUM = 4;
constexpr uint32_t TABLE_DIM_NUM = 2;
constexpr uint32_t SEQ_LEN_DIM_NUM = 1;
constexpr int64_t QUEST_BLOCK_SIZE = 128;
constexpr int64_t QUEST_HEAD_DIM = 128;
constexpr int64_t QUEST_MAX_METADATA_BLOCKS_PER_REQUEST = 6;
constexpr int64_t QUEST_TILING_UINT32_MAX = 4294967295LL;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
} // namespace

static ge::graphStatus QuestPrefillMetadataTilingFunc(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("QuestPrefillMetadata", "Tiling context is null."),
               return ge::GRAPH_FAILED);

    auto platform_info = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
    auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
    const uint32_t aiv_num = ascendc_platform.GetCoreNumAiv();
    OPS_ERR_IF(aiv_num == 0,
               OPS_LOG_E(context->GetNodeName(), "GetCoreNumAiv returned 0."),
               return ge::GRAPH_FAILED);

    const gert::StorageShape *k_cache_shape = context->GetInputShape(K_CACHE_INDEX);
    const gert::StorageShape *block_tables_shape = context->GetInputShape(BLOCK_TABLES_INDEX);
    const gert::StorageShape *refresh_start_seq_lens_shape =
        context->GetInputShape(REFRESH_START_SEQ_LENS_INDEX);
    const gert::StorageShape *refresh_end_seq_lens_shape = context->GetInputShape(REFRESH_END_SEQ_LENS_INDEX);
    const gert::StorageShape *metadata_block_tables_shape = context->GetInputShape(METADATA_BLOCK_TABLES_INDEX);
    const gert::StorageShape *maxblocks_shape = context->GetOutputShape(MAXBLOCKS_OUTPUT_INDEX);
    const gert::StorageShape *minblocks_shape = context->GetOutputShape(MINBLOCKS_OUTPUT_INDEX);
    OPS_ERR_IF(k_cache_shape == nullptr || block_tables_shape == nullptr ||
                   refresh_start_seq_lens_shape == nullptr || refresh_end_seq_lens_shape == nullptr ||
                   metadata_block_tables_shape == nullptr || maxblocks_shape == nullptr ||
                   minblocks_shape == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Required tensor shape is null."),
               return ge::GRAPH_FAILED);

    const auto &k_cache_storage = k_cache_shape->GetStorageShape();
    const auto &block_tables_storage = block_tables_shape->GetStorageShape();
    const auto &refresh_start_seq_lens_storage = refresh_start_seq_lens_shape->GetStorageShape();
    const auto &refresh_end_seq_lens_storage = refresh_end_seq_lens_shape->GetStorageShape();
    const auto &metadata_block_tables_storage =
        metadata_block_tables_shape->GetStorageShape();
    const auto &maxblocks_storage = maxblocks_shape->GetStorageShape();
    const auto &minblocks_storage = minblocks_shape->GetStorageShape();

    OPS_ERR_IF(k_cache_storage.GetDimNum() != K_CACHE_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "k_cache must be 4D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(block_tables_storage.GetDimNum() != TABLE_DIM_NUM ||
                   metadata_block_tables_storage.GetDimNum() != TABLE_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "block tables must be 2D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(refresh_start_seq_lens_storage.GetDimNum() != SEQ_LEN_DIM_NUM ||
                   refresh_end_seq_lens_storage.GetDimNum() != SEQ_LEN_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "refresh start/end seq lens must be 1D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(maxblocks_storage.GetDimNum() != K_CACHE_DIM_NUM ||
                   minblocks_storage.GetDimNum() != K_CACHE_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "metadata outputs must be 4D."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(refresh_start_seq_lens_storage.GetDim(DIM_0) != refresh_end_seq_lens_storage.GetDim(DIM_0),
               OPS_LOG_E(context->GetNodeName(), "refresh start/end seq lens batch sizes must match."),
               return ge::GRAPH_FAILED);

    const int64_t batch_size = refresh_end_seq_lens_storage.GetDim(DIM_0);
    const int64_t num_kv_blocks = k_cache_storage.GetDim(DIM_0);
    const int64_t num_kv_heads = k_cache_storage.GetDim(DIM_2);
    const int64_t block_size = k_cache_storage.GetDim(DIM_1);
    const int64_t head_dim = k_cache_storage.GetDim(DIM_3);
    const int64_t max_kv_blocks_per_request = block_tables_storage.GetDim(DIM_1);
    const int64_t max_metadata_blocks_per_request = metadata_block_tables_storage.GetDim(DIM_1);
    const int64_t num_metadata_blocks = maxblocks_storage.GetDim(DIM_0);

    OPS_ERR_IF(block_size != QUEST_BLOCK_SIZE || head_dim != QUEST_HEAD_DIM,
               OPS_LOG_E(context->GetNodeName(), "QUEST metadata requires block_size == 128 and head_dim == 128."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(batch_size < 0 || num_metadata_blocks < 0,
               OPS_LOG_E(context->GetNodeName(), "Tensor dimensions must be non-negative."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(num_kv_blocks <= 0 || num_kv_heads <= 0 || max_kv_blocks_per_request <= 0,
               OPS_LOG_E(context->GetNodeName(), "k_cache and block_tables dimensions must be positive."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(block_tables_storage.GetDim(DIM_0) != batch_size ||
                   metadata_block_tables_storage.GetDim(DIM_0) != batch_size,
               OPS_LOG_E(context->GetNodeName(), "Batch dimensions must match refresh_end_seq_lens."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(max_metadata_blocks_per_request <= 0 ||
                   max_metadata_blocks_per_request > QUEST_MAX_METADATA_BLOCKS_PER_REQUEST,
               OPS_LOG_E(context->GetNodeName(), "metadata_block_tables.shape[1] exceeds QUEST kernel limit."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(maxblocks_storage.GetDim(DIM_0) != minblocks_storage.GetDim(DIM_0) ||
                   maxblocks_storage.GetDim(DIM_1) != minblocks_storage.GetDim(DIM_1) ||
                   maxblocks_storage.GetDim(DIM_2) != minblocks_storage.GetDim(DIM_2) ||
                   maxblocks_storage.GetDim(DIM_3) != minblocks_storage.GetDim(DIM_3),
               OPS_LOG_E(context->GetNodeName(), "maxblocks and minblocks outputs must have matching shapes."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(maxblocks_storage.GetDim(DIM_1) != block_size ||
                   maxblocks_storage.GetDim(DIM_2) != num_kv_heads ||
                   maxblocks_storage.GetDim(DIM_3) != head_dim,
               OPS_LOG_E(context->GetNodeName(), "metadata output shape must match k_cache page layout."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(batch_size > 0 &&
                   num_metadata_blocks / batch_size < max_metadata_blocks_per_request,
               OPS_LOG_E(context->GetNodeName(), "metadata outputs are too small for metadata_block_tables."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(batch_size > QUEST_TILING_UINT32_MAX || num_kv_heads > QUEST_TILING_UINT32_MAX ||
                   max_kv_blocks_per_request > QUEST_TILING_UINT32_MAX,
               OPS_LOG_E(context->GetNodeName(), "Tensor dimensions exceed QUEST tiling range."),
               return ge::GRAPH_FAILED);

    QuestPrefillMetadataTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(batch_size));
    tiling.set_numKvHeads(static_cast<uint32_t>(num_kv_heads));
    tiling.set_blockSize(static_cast<uint32_t>(block_size));
    tiling.set_headDim(static_cast<uint32_t>(head_dim));
    tiling.set_maxKvBlocksPerRequest(static_cast<uint32_t>(max_kv_blocks_per_request));
    tiling.set_maxMetadataBlocksPerRequest(
        static_cast<uint32_t>(max_metadata_blocks_per_request));

    const int64_t batch_heads = batch_size * num_kv_heads;
    const uint32_t block_dim =
        batch_heads == 0 ? 1 : static_cast<uint32_t>(std::min(batch_heads, static_cast<int64_t>(aiv_num)));
    context->SetBlockDim(block_dim);

    const auto k_cache_desc = context->GetInputDesc(K_CACHE_INDEX);
    OPS_LOG_E_IF_NULL(context, k_cache_desc, return ge::GRAPH_FAILED);
    const auto maxblocks_desc = context->GetOutputDesc(MAXBLOCKS_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, maxblocks_desc, return ge::GRAPH_FAILED);
    const auto minblocks_desc = context->GetOutputDesc(MINBLOCKS_OUTPUT_INDEX);
    OPS_LOG_E_IF_NULL(context, minblocks_desc, return ge::GRAPH_FAILED);
    const auto k_cache_data_type = k_cache_desc->GetDataType();
    OPS_ERR_IF(k_cache_data_type != ge::DT_FLOAT16 && k_cache_data_type != ge::DT_BF16,
               OPS_LOG_E(context->GetNodeName(), "k_cache dtype must be float16 or bfloat16."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(maxblocks_desc->GetDataType() != k_cache_data_type ||
                   minblocks_desc->GetDataType() != k_cache_data_type,
               OPS_LOG_E(context->GetNodeName(), "metadata output dtypes must match k_cache dtype."),
               return ge::GRAPH_FAILED);
    tiling.set_dataType(k_cache_data_type == ge::DT_BF16 ? QUEST_PREFILL_METADATA_DTYPE_BF16
                                                         : QUEST_PREFILL_METADATA_DTYPE_FP16);
    context->SetTilingKey(QUEST_PREFILL_METADATA_TILING);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct QuestPrefillMetadataCompileInfo {};

static ge::graphStatus QuestPrefillMetadataTilingParse(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuestPrefillMetadata)
    .Tiling(QuestPrefillMetadataTilingFunc)
    .TilingParse<QuestPrefillMetadataCompileInfo>(QuestPrefillMetadataTilingParse);
} // namespace optiling
