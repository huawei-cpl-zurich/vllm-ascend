/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Decode-only/arch32-only tiling entry for SelectInferAttentionScore.
 */

#include "select_infer_attention_score_tiling.h"
#include "arch32/select_infer_attention_score_tiling_v3.h"
#include "log/log.h"
#include "log/error_code.h"

namespace optiling {

ge::graphStatus TilingSelectInferAttentionScore(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR("SelectInferAttentionScore", "Tiling context is null."),
        return ge::GRAPH_FAILED);

    // Decode-only port: route all tiling through arch32 V3 and keep prefill/arch35 disabled.
    return TilingSelectInferAttentionScoreV3(context);
}

FIA_EXTERN_C ge::graphStatus DoOpTilingSelectInferAttentionScore(gert::TilingContext *context)
{
    return TilingSelectInferAttentionScore(context);
}

extern "C" {
__attribute__((visibility("default"))) ge::graphStatus DeviceDoOpTilingSelectInferAttentionScore(
    gert::TilingContext *context)
{
    return DoOpTilingSelectInferAttentionScore(context);
}
}

} // namespace optiling
