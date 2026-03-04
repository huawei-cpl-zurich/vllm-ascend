/**
 * Compatibility shim for legacy ops-transformer-style logging includes.
 */
#pragma once

#include "log/ops_log.h"

#ifndef OP_LOGD
#define OP_LOGD(OPS_DESC, ...) OPS_LOG_D(OPS_DESC, ##__VA_ARGS__)
#endif
#ifndef OP_LOGI
#define OP_LOGI(OPS_DESC, ...) OPS_LOG_I(OPS_DESC, ##__VA_ARGS__)
#endif
#ifndef OP_LOGW
#define OP_LOGW(OPS_DESC, ...) OPS_LOG_W(OPS_DESC, ##__VA_ARGS__)
#endif
#ifndef OP_LOGE
#define OP_LOGE(OPS_DESC, ...) OPS_LOG_E(OPS_DESC, ##__VA_ARGS__)
#endif

#ifndef OP_CHECK_IF
#define OP_CHECK_IF(COND, LOG_FUNC, EXPR) OP_CHECK((COND), LOG_FUNC, EXPR)
#endif

#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    OP_CHECK((ptr) == nullptr, OP_LOGE((context)->GetNodeName(), "%s is nullptr!", #ptr), return ge::GRAPH_FAILED)
#endif
