# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0.

# Copy common/op_kernel into generated src tree for kernel-side shared headers.
set(select_infer_attention_score_depends common PARENT_SCOPE)

add_ops_compile_options(
        OP_NAME SelectInferAttentionScore
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnnInner PRIVATE
        select_infer_attention_score_def.cpp
)

# Expose only V4 public ACLNN API for this decode-only port.
target_sources(opapi PRIVATE
        op_api/aclnn_select_infer_attention_score_v4.cpp
        op_api/aclnn_select_infer_attention_score_inner.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(aclnn_ops_train PRIVATE
            op_api/aclnn_select_infer_attention_score_v4.cpp
            op_api/aclnn_select_infer_attention_score_inner.cpp
    )

    target_sources(aclnn_ops_infer PRIVATE
            op_api/aclnn_select_infer_attention_score_v4.cpp
            op_api/aclnn_select_infer_attention_score_inner.cpp
    )
endif ()

target_sources(optiling PRIVATE
        select_infer_attention_score_tiling.cpp
        select_infer_attention_score_tiling_register.cpp
        arch32/select_infer_attention_score_tiling_v3.cpp
        arch32/select_infer_attention_score_tiling_info_parser.cpp
        arch32/select_infer_attention_score_tiling_check.cpp
        arch32/select_infer_attention_score_tiling_check_consistency.cpp
        arch32/select_infer_attention_score_tiling_check_feature.cpp
        arch32/select_infer_attention_score_tiling_check_existence.cpp
        arch32/select_infer_attention_score_tiling_check_single_para.cpp
        ../../common/op_host/fia_tiling_shape.cpp
        ../../common/op_host/fia_tiling_info.cpp
        ../../common/op_host/split_core.cpp
        ../../common/op_host/arch32/fia_tiling_nonquant_mla.cpp
        ../../common/op_host/arch32/fia_tiling_nonquant.cpp
        ../../common/op_host/arch32/fia_tiling_empty_tensor.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        select_infer_attention_score_infershape.cpp
)

file(GLOB _SELECT_ACLNN_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_select_infer_attention_score*.h")

install(FILES ${_SELECT_ACLNN_HEADER}
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)
