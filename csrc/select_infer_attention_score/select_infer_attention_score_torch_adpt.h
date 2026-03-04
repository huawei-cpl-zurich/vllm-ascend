/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#ifndef SELECT_INFER_ATTENTION_SCORE_TORCH_ADPT_H
#define SELECT_INFER_ATTENTION_SCORE_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> select_infer_attention_score(
    const at::Tensor &query,
    const at::TensorList &key,
    const at::TensorList &value,
    const c10::optional<at::Tensor> &atten_mask,
    const c10::optional<at::Tensor> &block_table,
    c10::string_view input_layout,
    int64_t block_size,
    at::IntArrayRef actual_seq_lengths,
    at::IntArrayRef actual_seq_lengths_kv,
    int64_t num_key_value_heads,
    int64_t num_heads,
    double scale,
    int64_t sparse_mode)
{
    std::string input_layout_str = std::string(input_layout);
    char *input_layout_ptr = const_cast<char *>(input_layout_str.c_str());

    at::Tensor attention_out = at::empty_like(query);
    at::Tensor softmax_lse = at::empty({0}, query.options().dtype(at::kFloat));

    constexpr int64_t pre_tokens = 2147483647;
    constexpr int64_t next_tokens = 2147483647;
    constexpr int64_t inner_precise = 1;
    constexpr int64_t antiquant_mode = 0;
    constexpr bool softmax_lse_flag = false;
    constexpr int64_t key_antiquant_mode = 0;
    constexpr int64_t value_antiquant_mode = 0;
    constexpr int64_t query_quant_mode = 0;

    c10::optional<at::Tensor> none_tensor;
    c10::optional<at::IntArrayRef> none_int_array;

    EXEC_NPU_CMD(
        aclnnSelectInferAttentionScoreV4,
        query,
        key,
        value,
        none_tensor,
        atten_mask,
        actual_seq_lengths,
        actual_seq_lengths_kv,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        block_table,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_int_array,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        none_tensor,
        num_heads,
        scale,
        pre_tokens,
        next_tokens,
        input_layout_ptr,
        num_key_value_heads,
        sparse_mode,
        inner_precise,
        block_size,
        antiquant_mode,
        softmax_lse_flag,
        key_antiquant_mode,
        value_antiquant_mode,
        query_quant_mode,
        attention_out,
        softmax_lse);

    return {attention_out, softmax_lse};
}

} // namespace vllm_ascend
#endif
