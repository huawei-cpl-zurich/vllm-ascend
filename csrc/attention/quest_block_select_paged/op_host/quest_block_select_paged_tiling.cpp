/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "quest_block_select_paged_tiling.h"
#include <algorithm>
#include "../op_kernel/quest_block_select_paged_tilingkey.h"
#include "error/ops_error.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t MAXBLOCKS_INDEX = 1;
constexpr uint32_t MINBLOCKS_INDEX = 2;
constexpr uint32_t METADATA_BLOCK_TABLES_INDEX = 3;
constexpr uint32_t SEQ_LENS_INDEX = 4;
constexpr uint32_t ATTR_K_INDEX = 0;
constexpr uint32_t ATTR_TOKENS_SINCE_METADATA_UPDATE_INDEX = 1;
constexpr uint32_t QUERY_DIM_NUM = 3;
constexpr uint32_t BLOCKS_DIM_NUM = 4;
constexpr uint32_t TABLE_DIM_NUM = 2;
constexpr uint32_t SEQ_LEN_DIM_NUM = 1;
constexpr uint32_t OUTPUT_DIM_NUM = 3;
constexpr int64_t QUEST_BLOCK_SIZE = 128;
constexpr int64_t QUEST_HEAD_DIM = 128;
constexpr int64_t QUEST_MAX_SELECTED_BLOCKS = 64;
constexpr int64_t QUEST_MAX_METADATA_BLOCKS_PER_REQUEST = 6;
constexpr int64_t QUEST_TILING_UINT32_MAX = 4294967295LL;
constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
} // namespace

static ge::graphStatus QuestBlockSelectPagedTilingFunc(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr,
               OPS_LOG_E("QuestBlockSelectPaged", "Tiling context is null."),
               return ge::GRAPH_FAILED);

    auto platform_info = context->GetPlatformInfo();
    OPS_LOG_E_IF_NULL(context, platform_info, return ge::GRAPH_FAILED);
    auto ascendc_platform = platform_ascendc::PlatformAscendC(platform_info);
    const uint32_t aiv_num = ascendc_platform.GetCoreNumAiv();
    OPS_ERR_IF(aiv_num == 0,
               OPS_LOG_E(context->GetNodeName(), "GetCoreNumAiv returned 0."),
               return ge::GRAPH_FAILED);

    const gert::StorageShape *query_shape = context->GetInputShape(QUERY_INDEX);
    const gert::StorageShape *maxblocks_shape = context->GetInputShape(MAXBLOCKS_INDEX);
    const gert::StorageShape *minblocks_shape = context->GetInputShape(MINBLOCKS_INDEX);
    const gert::StorageShape *metadata_block_tables_shape =
        context->GetInputShape(METADATA_BLOCK_TABLES_INDEX);
    const gert::StorageShape *seq_lens_shape = context->GetInputShape(SEQ_LENS_INDEX);
    const gert::StorageShape *selected_indices_shape = context->GetOutputShape(0);
    OPS_ERR_IF(query_shape == nullptr || maxblocks_shape == nullptr || minblocks_shape == nullptr ||
                   metadata_block_tables_shape == nullptr || seq_lens_shape == nullptr ||
                   selected_indices_shape == nullptr,
               OPS_LOG_E(context->GetNodeName(), "Required tensor shape is null."),
               return ge::GRAPH_FAILED);

    const auto &query_storage = query_shape->GetStorageShape();
    const auto &maxblocks_storage = maxblocks_shape->GetStorageShape();
    const auto &minblocks_storage = minblocks_shape->GetStorageShape();
    const auto &metadata_block_tables_storage =
        metadata_block_tables_shape->GetStorageShape();
    const auto &seq_lens_storage = seq_lens_shape->GetStorageShape();
    const auto &selected_indices_storage = selected_indices_shape->GetStorageShape();

    OPS_ERR_IF(query_storage.GetDimNum() != QUERY_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "query must be 3D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(maxblocks_storage.GetDimNum() != BLOCKS_DIM_NUM ||
                   minblocks_storage.GetDimNum() != BLOCKS_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "metadata tensors must be 4D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(metadata_block_tables_storage.GetDimNum() != TABLE_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "metadata_block_tables must be 2D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(seq_lens_storage.GetDimNum() != SEQ_LEN_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "seq_lens must be 1D."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(selected_indices_storage.GetDimNum() != OUTPUT_DIM_NUM,
               OPS_LOG_E(context->GetNodeName(), "selected_indices must be 3D."),
               return ge::GRAPH_FAILED);

    const int64_t batch_size = query_storage.GetDim(DIM_0);
    const int64_t num_heads = query_storage.GetDim(DIM_1);
    const int64_t head_dim = query_storage.GetDim(DIM_2);
    const int64_t num_metadata_blocks = maxblocks_storage.GetDim(DIM_0);
    const int64_t block_size = maxblocks_storage.GetDim(DIM_1);
    const int64_t num_kv_heads = maxblocks_storage.GetDim(DIM_2);
    const int64_t metadata_head_dim = maxblocks_storage.GetDim(DIM_3);
    const int64_t max_metadata_blocks_per_request = metadata_block_tables_storage.GetDim(DIM_1);
    const int64_t output_stride = selected_indices_storage.GetDim(DIM_2);

    OPS_ERR_IF(batch_size < 0 || num_metadata_blocks < 0,
               OPS_LOG_E(context->GetNodeName(), "Tensor dimensions must be non-negative."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(head_dim != QUEST_HEAD_DIM || metadata_head_dim != QUEST_HEAD_DIM,
               OPS_LOG_E(context->GetNodeName(), "QUEST block selection requires head_dim == 128."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(block_size != QUEST_BLOCK_SIZE,
               OPS_LOG_E(context->GetNodeName(), "QUEST block selection requires metadata block_size == 128."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0,
               OPS_LOG_E(context->GetNodeName(), "num_heads must be positive and divisible by num_kv_heads."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(maxblocks_storage.GetDim(DIM_0) != minblocks_storage.GetDim(DIM_0) ||
                   maxblocks_storage.GetDim(DIM_1) != minblocks_storage.GetDim(DIM_1) ||
                   maxblocks_storage.GetDim(DIM_2) != minblocks_storage.GetDim(DIM_2) ||
                   maxblocks_storage.GetDim(DIM_3) != minblocks_storage.GetDim(DIM_3),
               OPS_LOG_E(context->GetNodeName(), "maxblocks and minblocks must have matching shapes."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(metadata_block_tables_storage.GetDim(DIM_0) != batch_size ||
                   seq_lens_storage.GetDim(DIM_0) != batch_size ||
                   selected_indices_storage.GetDim(DIM_0) != batch_size,
               OPS_LOG_E(context->GetNodeName(), "Batch dimensions must match query batch size."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(selected_indices_storage.GetDim(DIM_1) != num_heads,
               OPS_LOG_E(context->GetNodeName(), "selected_indices.shape[1] must equal query num_heads."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(max_metadata_blocks_per_request <= 0 ||
                   max_metadata_blocks_per_request > QUEST_MAX_METADATA_BLOCKS_PER_REQUEST,
               OPS_LOG_E(context->GetNodeName(), "metadata_block_tables.shape[1] exceeds QUEST kernel limit."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(batch_size > 0 &&
                   num_metadata_blocks / batch_size < max_metadata_blocks_per_request,
               OPS_LOG_E(context->GetNodeName(), "metadata tensors are too small for metadata_block_tables."),
               return ge::GRAPH_FAILED);
    const auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const int64_t *selected_k = attrs->GetInt(ATTR_K_INDEX);
    OPS_LOG_E_IF_NULL(context, selected_k, return ge::GRAPH_FAILED);
    OPS_ERR_IF(*selected_k <= 0 || *selected_k > QUEST_MAX_SELECTED_BLOCKS,
               OPS_LOG_E(context->GetNodeName(), "k must be in (0, 64]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(output_stride < *selected_k,
               OPS_LOG_E(context->GetNodeName(), "selected_indices.shape[2] must be >= k."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(batch_size > QUEST_TILING_UINT32_MAX || num_heads > QUEST_TILING_UINT32_MAX ||
                   num_kv_heads > QUEST_TILING_UINT32_MAX || output_stride > QUEST_TILING_UINT32_MAX,
               OPS_LOG_E(context->GetNodeName(), "Tensor dimensions exceed QUEST tiling range."),
               return ge::GRAPH_FAILED);

    const int64_t *tokens_since_metadata_update =
        attrs->GetInt(ATTR_TOKENS_SINCE_METADATA_UPDATE_INDEX);
    OPS_LOG_E_IF_NULL(context, tokens_since_metadata_update, return ge::GRAPH_FAILED);
    OPS_ERR_IF(*tokens_since_metadata_update != -1 &&
                   (*tokens_since_metadata_update < 0 || *tokens_since_metadata_update > block_size),
               OPS_LOG_E(context->GetNodeName(),
                         "tokens_since_metadata_update must be -1 or in [0, block_size]."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(*tokens_since_metadata_update != -1 && *selected_k < 2,
               OPS_LOG_E(context->GetNodeName(),
                         "QUEST block selection requires k >= 2 when fixed anchors are enabled."),
               return ge::GRAPH_FAILED);

    QuestBlockSelectPagedTilingData tiling;
    tiling.set_batchSize(static_cast<uint32_t>(batch_size));
    tiling.set_numKvHeads(static_cast<uint32_t>(num_kv_heads));
    tiling.set_numHeads(static_cast<uint32_t>(num_heads));
    tiling.set_blockSize(static_cast<uint32_t>(block_size));
    tiling.set_headDim(static_cast<uint32_t>(head_dim));
    tiling.set_maxMetadataBlocksPerRequest(
        static_cast<uint32_t>(max_metadata_blocks_per_request));
    tiling.set_k(static_cast<uint32_t>(*selected_k));
    tiling.set_outputStride(static_cast<uint32_t>(output_stride));
    tiling.set_tokensSinceMetadataUpdate(
        static_cast<int32_t>(*tokens_since_metadata_update));

    const int64_t batch_heads = batch_size * num_heads;
    const uint32_t block_dim =
        batch_heads == 0 ? 1 : static_cast<uint32_t>(std::min(batch_heads, static_cast<int64_t>(aiv_num)));
    context->SetBlockDim(block_dim);

    const auto query_desc = context->GetInputDesc(QUERY_INDEX);
    OPS_LOG_E_IF_NULL(context, query_desc, return ge::GRAPH_FAILED);
    const auto maxblocks_desc = context->GetInputDesc(MAXBLOCKS_INDEX);
    OPS_LOG_E_IF_NULL(context, maxblocks_desc, return ge::GRAPH_FAILED);
    const auto minblocks_desc = context->GetInputDesc(MINBLOCKS_INDEX);
    OPS_LOG_E_IF_NULL(context, minblocks_desc, return ge::GRAPH_FAILED);
    const auto query_data_type = query_desc->GetDataType();
    OPS_ERR_IF(query_data_type != ge::DT_FLOAT16 && query_data_type != ge::DT_BF16,
               OPS_LOG_E(context->GetNodeName(), "query dtype must be float16 or bfloat16."),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(maxblocks_desc->GetDataType() != query_data_type ||
                   minblocks_desc->GetDataType() != query_data_type,
               OPS_LOG_E(context->GetNodeName(), "query, maxblocks, and minblocks dtypes must match."),
               return ge::GRAPH_FAILED);
    tiling.set_dataType(query_data_type == ge::DT_BF16 ? QUEST_BLOCK_SELECT_PAGED_DTYPE_BF16
                                                       : QUEST_BLOCK_SELECT_PAGED_DTYPE_FP16);
    context->SetTilingKey(QUEST_BLOCK_SELECT_PAGED_TILING);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct QuestBlockSelectPagedCompileInfo {};

static ge::graphStatus QuestBlockSelectPagedTilingParse(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(QuestBlockSelectPaged)
    .Tiling(QuestBlockSelectPagedTilingFunc)
    .TilingParse<QuestBlockSelectPagedCompileInfo>(QuestBlockSelectPagedTilingParse);
} // namespace optiling
