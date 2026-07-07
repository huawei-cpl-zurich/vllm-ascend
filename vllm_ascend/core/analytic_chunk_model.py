#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Analytic (profiling-free) latency model for pipeline-parallel chunked prefill.

The per-forward prefill latency of a chunk of ``c`` new tokens at history ``h``
is modeled as

    T(h, c) = c * (1 + (h + c/2) / h_cross)

where ``h_cross`` is the history length at which per-token attention cost equals
per-token MLP cost.  The chunk-sizing decision is invariant to the overall scale
and to any fixed per-step overhead (both cancel against a self-derived cost
budget), so **only ``h_cross`` matters** and it can be estimated from the model
architecture with approximate FLOP counts -- no profiling or hardware
calibration required.
"""

from typing import Any

from vllm.logger import logger


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def generate_allowed_chunk_sizes(min_chunk: int, max_chunk: int, alignment: int) -> list[int]:
    """Hardware-friendly chunk grid of the form 2^n and 3*2^(n-1), aligned."""
    candidates: set[int] = set()
    power = 1
    while power <= max_chunk:
        candidates.add(power)
        if power % 2 == 0:
            candidates.add(3 * power // 2)
        power *= 2
    aligned = {
        _align_up(c, alignment)
        for c in candidates
        if min_chunk <= c <= max_chunk
    }
    return sorted(c for c in aligned if min_chunk <= c <= max_chunk and c % alignment == 0)


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------
class AnalyticLatencyModel:
    """``T(h, c) = c * (1 + (h + c/2) / h_cross)`` (scale-free: b=1, e=0)."""

    def __init__(self, h_cross: float) -> None:
        self.h_cross = max(float(h_cross), 1.0)

    def predict(self, history: int, chunk: int) -> float:
        return chunk * (1.0 + (history + chunk / 2.0) / self.h_cross)


class SmoothChunkSelector:
    """Floor-tracking chunk selection under a shared per-step cost budget.

    Returns the largest allowed chunk whose predicted latency stays under
    ``max(cost_budget, floor(history))`` where ``floor(history) = T(history,
    c_min)``.  ``c_min`` is therefore always feasible (progress guaranteed); the
    final short chunk returns the exact remainder.
    """

    def __init__(self, model: AnalyticLatencyModel, allowed_chunk_sizes: list[int]) -> None:
        self.model = model
        self.allowed = sorted(allowed_chunk_sizes)
        self.c_min = self.allowed[0]

    def select_chunk(
        self,
        history: int,
        remaining_prompt_tokens: int,
        token_budget: int,
        cost_budget: float,
    ) -> int | None:
        cap = min(remaining_prompt_tokens, token_budget)
        if cap <= 0:
            return None
        if cap <= self.c_min:
            return cap
        target = max(cost_budget, self.model.predict(history, self.c_min))
        best = self.c_min
        for chunk in self.allowed:
            if chunk > cap:
                break
            if self.model.predict(history, chunk) <= target:
                best = chunk
        return best


# ---------------------------------------------------------------------------
# h_cross estimation from model architecture
# ---------------------------------------------------------------------------
# h_cross = (L / A) * k_mfu, per layer, per token:
#   L = linear FLOPs (MLP + projections; MoE -> active experts only)
#   A = attention FLOPs per history key (score + value aggregation)
#   k_mfu = attention-vs-linear MFU ratio (single hardware fudge, default 0.8)
#
# FLOPs use the "2 * params" convention for matmuls and "2 * dim" per dot
# product.  Absolute values are irrelevant -- only the ratio L/A is used.

_DEFAULT_H_CROSS = 12000


def _attention_flops_per_key(dims: dict) -> float:
    """Attention FLOPs per token per history key per layer."""
    n_q = dims["n_q_heads"]
    if dims.get("is_mla"):
        # Prefill MLA decompresses to per-head qk/v dims (absorption is a decode
        # optimization); score over qk_head_dim, value over v_head_dim.
        qk = dims["qk_nope_head_dim"] + dims["qk_rope_head_dim"]
        return 2.0 * n_q * qk + 2.0 * n_q * dims["v_head_dim"]
    # Standard MHA/GQA: QK^T + AV over all query heads (GQA compute is the same).
    return 4.0 * n_q * dims["head_dim"]


def _linear_flops(dims: dict) -> float:
    """Linear (projection + FFN) FLOPs per token per layer."""
    d = dims["hidden_size"]

    if dims.get("is_mla"):
        n_q = dims["n_q_heads"]
        q_lora = dims.get("q_lora_rank") or 0
        kv_lora = dims["kv_lora_rank"]
        qk_rope = dims["qk_rope_head_dim"]
        qk_head = dims["qk_nope_head_dim"] + qk_rope
        v_head = dims["v_head_dim"]
        if q_lora:
            q_params = d * q_lora + q_lora * n_q * qk_head
        else:
            q_params = d * n_q * qk_head
        proj_params = (
            q_params
            + d * (kv_lora + qk_rope)          # kv_a
            + kv_lora * n_q * (dims["qk_nope_head_dim"] + v_head)  # kv_b
            + n_q * v_head * d                 # o_proj
        )
    else:
        n_q, n_kv, dh = dims["n_q_heads"], dims["n_kv_heads"], dims["head_dim"]
        proj_params = d * dh * (n_q + 2 * n_kv) + n_q * dh * d  # QKV + O

    if dims.get("is_moe"):
        moe_ff = dims["moe_intermediate_size"]
        active = dims["num_experts_per_tok"] + dims.get("n_shared_experts", 0)
        ffn_params = 3 * d * moe_ff * active + d * dims["n_routed_experts"]  # SwiGLU + router
    else:
        ffn_params = 3 * d * dims["intermediate_size"]  # SwiGLU gate+up+down

    return 2.0 * (proj_params + ffn_params)


def h_cross_from_dims(dims: dict, k_mfu: float = 0.8) -> int:
    attn = _attention_flops_per_key(dims)
    if attn <= 0:
        return _DEFAULT_H_CROSS
    return max(1, int(_linear_flops(dims) / attn * k_mfu))


def _extract_dims(model_config, parallel_config) -> dict | None:
    """Pull the fields needed for h_cross from a vllm ModelConfig; None on failure."""
    try:
        hf = getattr(model_config, "hf_text_config", None) or model_config.hf_config
        d = model_config.get_hidden_size()
        n_q = model_config.get_num_attention_heads(parallel_config)
        n_kv = model_config.get_num_kv_heads(parallel_config)
        dh = model_config.get_head_size()
        is_mla = bool(getattr(model_config, "is_deepseek_mla", False)) or bool(
            getattr(hf, "kv_lora_rank", None)
        )
        n_routed = getattr(hf, "n_routed_experts", None)
        is_moe = bool(n_routed)
        dims: dict[str, Any] = {
            "hidden_size": d,
            "n_q_heads": n_q,
            "n_kv_heads": n_kv,
            "head_dim": dh,
            "intermediate_size": getattr(hf, "intermediate_size", 4 * d),
            "is_mla": is_mla,
            "is_moe": is_moe,
        }
        if is_moe:
            dims.update(
                n_routed_experts=n_routed,
                num_experts_per_tok=getattr(hf, "num_experts_per_tok", 1),
                moe_intermediate_size=getattr(hf, "moe_intermediate_size", dims["intermediate_size"]),
                n_shared_experts=getattr(hf, "n_shared_experts", 0) or 0,
            )
        if is_mla:
            dims.update(
                kv_lora_rank=getattr(hf, "kv_lora_rank", 512),
                q_lora_rank=getattr(hf, "q_lora_rank", None),
                qk_rope_head_dim=getattr(hf, "qk_rope_head_dim", 64),
                qk_nope_head_dim=getattr(hf, "qk_nope_head_dim", 128),
                v_head_dim=getattr(hf, "v_head_dim", dh),
            )
        return dims
    except Exception as exc:  # noqa: BLE001 - estimation must never crash startup
        logger.warning("[AnalyticChunk] Could not read model dims for h_cross: %s", exc)
        return None


def estimate_h_cross(model_config, parallel_config, k_mfu: float = 0.8) -> int:
    """Estimate h_cross (tokens) for a model, dispatching on architecture.

    Falls back to ``6 * hidden_size`` and then a constant if the config cannot
    be read.  Family selection is implicit in ``_extract_dims`` (MLA vs standard
    attention, MoE vs dense FFN); add a new branch there for a new family.
    """
    dims = _extract_dims(model_config, parallel_config)
    if dims is None:
        try:
            return int(6 * model_config.get_hidden_size())
        except Exception:  # noqa: BLE001
            return _DEFAULT_H_CROSS
    h_cross = h_cross_from_dims(dims, k_mfu)
    family = "mla" if dims.get("is_mla") else ("moe" if dims.get("is_moe") else "dense")
    logger.info(
        "[AnalyticChunk] Estimated h_cross=%d (family=%s, hidden=%d, n_q=%d, k_mfu=%.2f)",
        h_cross, family, dims["hidden_size"], dims["n_q_heads"], k_mfu,
    )
    return h_cross


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class AnalyticChunkManager:
    """Owns the analytic model + selector and derives the per-step budgets.

    Built eagerly (no profiling); ready from construction.  All budgets are in
    the model's relative units and self-consistent.
    """

    def __init__(
        self,
        *,
        h_cross: int,
        page_size: int,
        min_chunk: int,
        max_chunk: int,
        allowed_chunk_sizes: list[int] | None,
        target_chunk: int | None,
        backfill_reserve_frac: float,
    ) -> None:
        alignment = max(page_size, 64)
        if allowed_chunk_sizes is not None:
            chunks = sorted(set(int(c) for c in allowed_chunk_sizes))
        else:
            chunks = generate_allowed_chunk_sizes(min_chunk, max_chunk, alignment)
        if not chunks:
            raise ValueError("analytic_chunk_config produced no allowed chunk sizes")
        self.allowed_chunk_sizes = chunks
        self.model = AnalyticLatencyModel(h_cross)
        self.selector = SmoothChunkSelector(self.model, chunks)
        budget_chunk = target_chunk if target_chunk is not None else chunks[-1]
        self._target_latency = self.model.predict(0, budget_chunk)
        self._backfill_reserve = backfill_reserve_frac * self._target_latency
        logger.info(
            "[AnalyticChunk] Manager ready: h_cross=%d, chunks=%s, target=%.3g, backfill=%.3g",
            int(self.model.h_cross), chunks, self._target_latency, self._backfill_reserve,
        )

    @property
    def is_ready(self) -> bool:
        return True

    @property
    def target_latency(self) -> float:
        return self._target_latency

    @property
    def backfill_reserve(self) -> float:
        return self._backfill_reserve

    def select_next_chunk(
        self, *, history: int, remaining_prompt_tokens: int, token_budget: int, cost_budget: float
    ) -> int | None:
        return self.selector.select_chunk(
            history=history,
            remaining_prompt_tokens=remaining_prompt_tokens,
            token_budget=token_budget,
            cost_budget=cost_budget,
        )

    def predict_cost(self, history: int, chunk: int) -> float:
        return self.model.predict(history, chunk)
