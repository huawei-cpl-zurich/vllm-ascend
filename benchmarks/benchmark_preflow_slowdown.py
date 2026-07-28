#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Benchmark PREFLOW relative slowdown against the vLLM v0.25.1 scheduler.

This is an editable research benchmark for real vLLM-Ascend runs. It uses
vLLM's supported ``load_format="dummy"`` mechanism, so results validate
scheduler behavior under dummy weights rather than final real-model latency.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# =============================================================================
# Editable Experiment Configuration
# =============================================================================

RUN_PROFILE: Literal["prototype", "full"] = "prototype"

MODEL = "/data/weights/Qwen3-30B-A3B-Instruct-2507/"
TOKENIZER = MODEL
TRUST_REMOTE_CODE = True
DTYPE = "bfloat16"
LOAD_FORMAT = "dummy"
TENSOR_PARALLEL_SIZE = 2
PIPELINE_PARALLEL_SIZE = 1
MAX_MODEL_LEN = 131_072
GPU_MEMORY_UTILIZATION = 0.7
MAX_NUM_BATCHED_TOKENS = 32_768
MAX_NUM_SEQS = 32
MAX_NUM_PARTIAL_PREFILLS = 1
MAX_LONG_PARTIAL_PREFILLS = 1
ENABLE_CHUNKED_PREFILL = True
LONG_PREFILL_TOKEN_THRESHOLD = 8_192
SCHEDULER_RESERVE_FULL_ISL = True
SCHEDULING_POLICY = "fcfs"
ENABLE_PREFIX_CACHING = False
ENFORCE_EAGER = False

PROMPT_LENGTHS = [
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    100_000,
]

PROMPT_LENGTH_PROBABILITIES = [
    0.30,
    0.25,
    0.20,
    0.14,
    0.08,
    0.03,
]

NUM_SOLO_WARMUP_RUNS = 1
NUM_SOLO_MEASURED_RUNS = 3
NUM_MIXED_WARMUP_REQUESTS = 16
NUM_MIXED_MEASURED_REQUESTS = 128
SEEDS = [20260728, 20260729, 20260730]
ARRIVAL_MODE: Literal["all_at_once", "poisson"] = "all_at_once"  # "poisson"
TARGET_REQUEST_RATE = 0.35
MAX_TOKENS = 1

PROTOTYPE_PROMPT_LENGTHS = [
    4_096,
    8_192,
    16_384,
    32_768,
]
PROTOTYPE_PROMPT_LENGTH_PROBABILITIES = [
    0.35,
    0.30,
    0.22,
    0.13,
]
PROTOTYPE_NUM_SOLO_WARMUP_RUNS = 0
PROTOTYPE_NUM_SOLO_MEASURED_RUNS = 1
PROTOTYPE_NUM_MIXED_WARMUP_REQUESTS = 0
PROTOTYPE_NUM_MIXED_MEASURED_REQUESTS = 32
PROTOTYPE_SEEDS = [20260728]

PREFLOW_WORK_EXPONENT = 1.5
PREFLOW_ADMISSION_BYPASS_BUDGET = 0.3
PREFLOW_AGE_PRIORITY_DOUBLE = 2.0

TOKEN_ID_LOW = 100
TOKEN_ID_HIGH_EXCLUSIVE = 32_000
SPECIAL_TOKEN_IDS = {0, 1, 2, 3}
PROMPT_VARIANTS_PER_LENGTH = 4
REUSE_PROMPT_TOKEN_LISTS = True

LENGTH_BUCKETS = [
    ("short", 0, 8_192),
    ("medium", 8_193, 32_768),
    ("long", 32_769, 65_536),
    ("very_long", 65_537, 10**12),
]
SLOWDOWN_THRESHOLDS = [1.05, 1.25, 1.5, 2.0, 4.0]
NEAR_ONE_SLOWDOWN_THRESHOLD = 1.05
NEAR_ONE_WARNING_FRACTION = 0.90
PAIRED_TIE_EPSILON = 0.01
TOP_PAIRED_EXTREMES = 10

OUTPUT_ROOT = Path("benchmark_results")


# =============================================================================
# Data Model
# =============================================================================


@dataclass(frozen=True)
class BenchmarkConfig:
    run_profile: str
    model: str
    tokenizer: str
    trust_remote_code: bool
    dtype: str
    load_format: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    max_num_partial_prefills: int
    max_long_partial_prefills: int
    enable_chunked_prefill: bool
    long_prefill_token_threshold: int
    scheduler_reserve_full_isl: bool
    scheduling_policy: str
    enable_prefix_caching: bool
    enforce_eager: bool
    prompt_lengths: list[int]
    prompt_length_probabilities: list[float]
    num_solo_warmup_runs: int
    num_solo_measured_runs: int
    num_mixed_warmup_requests: int
    num_mixed_measured_requests: int
    seeds: list[int]
    arrival_mode: str
    target_request_rate: float
    max_tokens: int
    preflow_work_exponent: float
    preflow_admission_bypass_budget: float
    preflow_age_priority_double: float
    token_id_low: int
    token_id_high_exclusive: int
    special_token_ids: list[int]
    prompt_variants_per_length: int
    reuse_prompt_token_lists: bool
    length_buckets: list[tuple[str, int, int]]
    slowdown_thresholds: list[float]
    near_one_slowdown_threshold: float
    near_one_warning_fraction: float
    paired_tie_epsilon: float
    top_paired_extremes: int


@dataclass(frozen=True)
class TraceRequest:
    seed: int
    phase: Literal["warmup", "measured"]
    request_index: int
    request_id: str
    prompt_length: int
    scheduled_arrival_offset_s: float
    token_seed: int
    token_variant: int


@dataclass
class ConcurrencyTracker:
    current: int = 0
    peak: int = 0
    area: float = 0.0
    last_time_s: float = 0.0

    def _advance(self, now_s: float) -> None:
        now_s = max(now_s, self.last_time_s)
        self.area += self.current * (now_s - self.last_time_s)
        self.last_time_s = now_s

    def on_submit(self, now_s: float) -> None:
        self._advance(now_s)
        self.current += 1
        self.peak = max(self.peak, self.current)

    def on_finish(self, now_s: float) -> None:
        self._advance(now_s)
        self.current = max(0, self.current - 1)

    def summary(self, wall_time_s: float) -> dict[str, float | int]:
        self._advance(wall_time_s)
        avg = self.area / wall_time_s if wall_time_s > 0 else 0.0
        return {
            "average_unfinished_requests": avg,
            "peak_unfinished_requests": self.peak,
        }


@dataclass(frozen=True)
class RuntimeImports:
    AsyncEngineArgs: Any
    AsyncLLM: Any
    SamplingParams: Any
    RequestOutputKind: Any


def make_config() -> BenchmarkConfig:
    cfg = BenchmarkConfig(
        run_profile=RUN_PROFILE,
        model=MODEL,
        tokenizer=TOKENIZER,
        trust_remote_code=TRUST_REMOTE_CODE,
        dtype=DTYPE,
        load_format=LOAD_FORMAT,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_partial_prefills=MAX_NUM_PARTIAL_PREFILLS,
        max_long_partial_prefills=MAX_LONG_PARTIAL_PREFILLS,
        enable_chunked_prefill=ENABLE_CHUNKED_PREFILL,
        long_prefill_token_threshold=LONG_PREFILL_TOKEN_THRESHOLD,
        scheduler_reserve_full_isl=SCHEDULER_RESERVE_FULL_ISL,
        scheduling_policy=SCHEDULING_POLICY,
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
        enforce_eager=ENFORCE_EAGER,
        prompt_lengths=list(PROMPT_LENGTHS),
        prompt_length_probabilities=list(PROMPT_LENGTH_PROBABILITIES),
        num_solo_warmup_runs=NUM_SOLO_WARMUP_RUNS,
        num_solo_measured_runs=NUM_SOLO_MEASURED_RUNS,
        num_mixed_warmup_requests=NUM_MIXED_WARMUP_REQUESTS,
        num_mixed_measured_requests=NUM_MIXED_MEASURED_REQUESTS,
        seeds=list(SEEDS),
        arrival_mode=ARRIVAL_MODE,
        target_request_rate=TARGET_REQUEST_RATE,
        max_tokens=MAX_TOKENS,
        preflow_work_exponent=PREFLOW_WORK_EXPONENT,
        preflow_admission_bypass_budget=PREFLOW_ADMISSION_BYPASS_BUDGET,
        preflow_age_priority_double=PREFLOW_AGE_PRIORITY_DOUBLE,
        token_id_low=TOKEN_ID_LOW,
        token_id_high_exclusive=TOKEN_ID_HIGH_EXCLUSIVE,
        special_token_ids=sorted(SPECIAL_TOKEN_IDS),
        prompt_variants_per_length=PROMPT_VARIANTS_PER_LENGTH,
        reuse_prompt_token_lists=REUSE_PROMPT_TOKEN_LISTS,
        length_buckets=list(LENGTH_BUCKETS),
        slowdown_thresholds=list(SLOWDOWN_THRESHOLDS),
        near_one_slowdown_threshold=NEAR_ONE_SLOWDOWN_THRESHOLD,
        near_one_warning_fraction=NEAR_ONE_WARNING_FRACTION,
        paired_tie_epsilon=PAIRED_TIE_EPSILON,
        top_paired_extremes=TOP_PAIRED_EXTREMES,
    )
    if RUN_PROFILE == "full":
        return cfg
    if RUN_PROFILE == "prototype":
        return replace(
            cfg,
            prompt_lengths=list(PROTOTYPE_PROMPT_LENGTHS),
            prompt_length_probabilities=list(PROTOTYPE_PROMPT_LENGTH_PROBABILITIES),
            num_solo_warmup_runs=PROTOTYPE_NUM_SOLO_WARMUP_RUNS,
            num_solo_measured_runs=PROTOTYPE_NUM_SOLO_MEASURED_RUNS,
            num_mixed_warmup_requests=PROTOTYPE_NUM_MIXED_WARMUP_REQUESTS,
            num_mixed_measured_requests=PROTOTYPE_NUM_MIXED_MEASURED_REQUESTS,
            seeds=list(PROTOTYPE_SEEDS),
        )
    raise ValueError(f"Unsupported RUN_PROFILE={RUN_PROFILE!r}.")


# =============================================================================
# Validation and Runtime Setup
# =============================================================================


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_local_import_paths() -> None:
    root = repo_root()
    # Prefer the checked-out vLLM v0.25.1 source in ./vllm over any ambient
    # installation, and keep this repository importable for vllm_ascend.
    for path in (root, root / "vllm"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def import_runtime() -> RuntimeImports:
    ensure_local_import_paths()
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm import SamplingParams

    return RuntimeImports(
        AsyncEngineArgs=AsyncEngineArgs,
        AsyncLLM=AsyncLLM,
        SamplingParams=SamplingParams,
        RequestOutputKind=RequestOutputKind,
    )


def validate_config(cfg: BenchmarkConfig, schedulers: list[str]) -> None:
    if cfg.run_profile not in {"prototype", "full"}:
        raise ValueError(f"Unsupported RUN_PROFILE={cfg.run_profile!r}.")
    if cfg.load_format != "dummy":
        raise ValueError("This benchmark must use load_format='dummy'.")
    if len(cfg.prompt_lengths) != len(cfg.prompt_length_probabilities):
        raise ValueError("PROMPT_LENGTHS and PROMPT_LENGTH_PROBABILITIES differ.")
    if not cfg.prompt_lengths:
        raise ValueError("At least one prompt length is required.")
    if any(length <= 0 for length in cfg.prompt_lengths):
        raise ValueError("All prompt lengths must be positive.")
    if any(prob < 0 for prob in cfg.prompt_length_probabilities):
        raise ValueError("Prompt length probabilities must be nonnegative.")
    prob_sum = sum(cfg.prompt_length_probabilities)
    if not math.isclose(prob_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Prompt length probabilities sum to {prob_sum}, not 1.")
    for length in cfg.prompt_lengths:
        if length > cfg.max_model_len:
            raise ValueError(f"Prompt length {length} exceeds MAX_MODEL_LEN.")
        if length + cfg.max_tokens > cfg.max_model_len:
            raise ValueError(
                f"Prompt length {length} plus MAX_TOKENS={cfg.max_tokens} exceeds MAX_MODEL_LEN={cfg.max_model_len}."
            )
    if cfg.arrival_mode not in {"all_at_once", "poisson"}:
        raise ValueError(f"Unsupported ARRIVAL_MODE={cfg.arrival_mode!r}.")
    if cfg.arrival_mode == "poisson" and cfg.target_request_rate <= 0:
        raise ValueError("TARGET_REQUEST_RATE must be positive for poisson mode.")
    if cfg.max_tokens != 1:
        raise ValueError("This TTFT benchmark expects MAX_TOKENS = 1.")
    if cfg.token_id_low >= cfg.token_id_high_exclusive:
        raise ValueError("TOKEN_ID_LOW must be less than TOKEN_ID_HIGH_EXCLUSIVE.")
    if cfg.prompt_variants_per_length <= 0:
        raise ValueError("PROMPT_VARIANTS_PER_LENGTH must be positive.")
    if cfg.max_num_partial_prefills <= 0:
        raise ValueError("MAX_NUM_PARTIAL_PREFILLS must be positive.")
    if cfg.max_long_partial_prefills <= 0:
        raise ValueError("MAX_LONG_PARTIAL_PREFILLS must be positive.")
    if cfg.max_num_partial_prefills > cfg.max_num_seqs:
        raise ValueError("MAX_NUM_PARTIAL_PREFILLS cannot exceed MAX_NUM_SEQS.")
    if cfg.max_long_partial_prefills > cfg.max_num_partial_prefills:
        raise ValueError("MAX_LONG_PARTIAL_PREFILLS cannot exceed MAX_NUM_PARTIAL_PREFILLS.")
    token_range = set(range(cfg.token_id_low, cfg.token_id_high_exclusive))
    if token_range.issubset(set(cfg.special_token_ids)):
        raise ValueError("No usable non-special token IDs remain.")
    if "preflow" in schedulers:
        if cfg.scheduling_policy != "fcfs":
            raise ValueError("PREFLOW requires SCHEDULING_POLICY = 'fcfs'.")
        if not cfg.scheduler_reserve_full_isl:
            raise ValueError("PREFLOW requires SCHEDULER_RESERVE_FULL_ISL = True.")
        if not cfg.enable_chunked_prefill:
            raise ValueError("PREFLOW comparison expects ENABLE_CHUNKED_PREFILL = True.")
        if cfg.preflow_work_exponent <= 0 or not math.isfinite(cfg.preflow_work_exponent):
            raise ValueError("PREFLOW_WORK_EXPONENT must be finite and positive.")
        if cfg.preflow_admission_bypass_budget < 0 or not math.isfinite(cfg.preflow_admission_bypass_budget):
            raise ValueError("PREFLOW_ADMISSION_BYPASS_BUDGET must be finite and nonnegative.")
        if cfg.preflow_age_priority_double <= 0 or not math.isfinite(cfg.preflow_age_priority_double):
            raise ValueError("PREFLOW_AGE_PRIORITY_DOUBLE must be finite and positive.")


def verify_preflow_importable() -> None:
    ensure_local_import_paths()
    try:
        import vllm_ascend.core.preflow_scheduler  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "PREFLOW scheduler cannot be imported. Run from the vllm-ascend "
            "repository with the vLLM v0.25.1 checkout available, and make "
            "sure vllm_ascend is importable."
        ) from exc


# =============================================================================
# Trace and Prompt Generation
# =============================================================================


def stable_u64(*parts: object) -> int:
    data = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def sample_prompt_length(rng: random.Random, cfg: BenchmarkConfig) -> int:
    draw = rng.random()
    cumulative = 0.0
    for length, prob in zip(cfg.prompt_lengths, cfg.prompt_length_probabilities):
        cumulative += prob
        if draw <= cumulative:
            return length
    return cfg.prompt_lengths[-1]


def make_arrival_offsets(count: int, rng: random.Random, cfg: BenchmarkConfig) -> list[float]:
    if count <= 0:
        return []
    if cfg.arrival_mode == "all_at_once":
        return [0.0] * count
    offsets = [0.0]
    current = 0.0
    for _ in range(1, count):
        current += rng.expovariate(cfg.target_request_rate)
        offsets.append(current)
    return offsets


def make_trace_phase(
    seed: int,
    phase: Literal["warmup", "measured"],
    count: int,
    cfg: BenchmarkConfig,
) -> list[TraceRequest]:
    rng = random.Random(stable_u64("trace", seed, phase))
    offsets = make_arrival_offsets(count, rng, cfg)
    trace = []
    for idx in range(count):
        prompt_length = sample_prompt_length(rng, cfg)
        token_variant = stable_u64("variant", seed, idx, prompt_length) % cfg.prompt_variants_per_length
        token_seed = stable_u64("tokens", seed, token_variant, prompt_length)
        trace.append(
            TraceRequest(
                seed=seed,
                phase=phase,
                request_index=idx,
                request_id=f"seed{seed}-{phase}-{idx:06d}",
                prompt_length=prompt_length,
                scheduled_arrival_offset_s=offsets[idx],
                token_seed=token_seed,
                token_variant=token_variant,
            )
        )
    return trace


def make_workload_traces(cfg: BenchmarkConfig) -> dict[int, list[TraceRequest]]:
    traces = {}
    for seed in cfg.seeds:
        traces[seed] = [
            *make_trace_phase(seed, "warmup", cfg.num_mixed_warmup_requests, cfg),
            *make_trace_phase(seed, "measured", cfg.num_mixed_measured_requests, cfg),
        ]
    return traces


def generate_token_ids(length: int, token_seed: int, cfg: BenchmarkConfig) -> list[int]:
    rng = random.Random(token_seed)
    special = set(cfg.special_token_ids)
    tokens: list[int] = []
    while len(tokens) < length:
        token_id = rng.randrange(cfg.token_id_low, cfg.token_id_high_exclusive)
        if token_id not in special:
            tokens.append(token_id)
    assert len(tokens) == length
    return tokens


class TokenBank:
    """Small prompt-token cache keyed by length and deterministic token seed."""

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self._cache: dict[tuple[int, int], list[int]] = {}

    def get(self, length: int, token_seed: int) -> list[int]:
        key = (length, token_seed)
        if key not in self._cache:
            self._cache[key] = generate_token_ids(length, token_seed, self.cfg)
        tokens = self._cache[key]
        if self.cfg.reuse_prompt_token_lists:
            return tokens
        return list(tokens)

    def materialize(self, trace: list[TraceRequest]) -> None:
        for request in trace:
            self.get(request.prompt_length, request.token_seed)


# =============================================================================
# Engine Construction and Request Execution
# =============================================================================


def scheduler_additional_config(scheduler: str, cfg: BenchmarkConfig) -> dict[str, Any]:
    if scheduler == "baseline":
        return {}
    if scheduler != "preflow":
        raise ValueError(f"Unknown scheduler {scheduler!r}.")
    return {
        "scheduler_config": {
            "preflow_config": {
                "enabled": True,
                "work_exponent": cfg.preflow_work_exponent,
                "admission_bypass_budget": cfg.preflow_admission_bypass_budget,
                "age_priority_double": cfg.preflow_age_priority_double,
            }
        }
    }


def make_engine_args(runtime: RuntimeImports, scheduler: str, cfg: BenchmarkConfig) -> Any:
    if scheduler == "preflow":
        verify_preflow_importable()
    return runtime.AsyncEngineArgs(
        model=cfg.model,
        tokenizer=cfg.tokenizer,
        trust_remote_code=cfg.trust_remote_code,
        dtype=cfg.dtype,
        load_format=cfg.load_format,
        tensor_parallel_size=cfg.tensor_parallel_size,
        pipeline_parallel_size=cfg.pipeline_parallel_size,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
        max_num_partial_prefills=cfg.max_num_partial_prefills,
        max_long_partial_prefills=cfg.max_long_partial_prefills,
        enable_chunked_prefill=cfg.enable_chunked_prefill,
        long_prefill_token_threshold=cfg.long_prefill_token_threshold,
        scheduler_reserve_full_isl=cfg.scheduler_reserve_full_isl,
        scheduling_policy=cfg.scheduling_policy,
        enable_prefix_caching=cfg.enable_prefix_caching,
        enforce_eager=cfg.enforce_eager,
        additional_config=scheduler_additional_config(scheduler, cfg),
        disable_log_stats=False,
    )


def make_engine(runtime: RuntimeImports, scheduler: str, cfg: BenchmarkConfig) -> Any:
    engine_args = make_engine_args(runtime, scheduler, cfg)
    engine = runtime.AsyncLLM.from_engine_args(engine_args)
    verify_engine_configuration(engine, scheduler, cfg)
    return engine


def config_value_to_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def resolved_scheduler_class_metadata(scheduler_config: Any) -> dict[str, str]:
    scheduler_cls = scheduler_config.get_scheduler_cls()
    return {
        "module": scheduler_cls.__module__,
        "name": scheduler_cls.__name__,
        "qualname": f"{scheduler_cls.__module__}.{scheduler_cls.__qualname__}",
    }


def verify_engine_configuration(engine: Any, scheduler: str, cfg: BenchmarkConfig) -> None:
    load_format = config_value_to_text(engine.vllm_config.load_config.load_format)
    if load_format != cfg.load_format:
        raise RuntimeError(f"Expected vLLM load_format={cfg.load_format!r}, got {load_format!r}.")
    scheduler_config = engine.vllm_config.scheduler_config
    scheduler_cls_text = config_value_to_text(scheduler_config.scheduler_cls)
    scheduler_metadata = resolved_scheduler_class_metadata(scheduler_config)
    resolved_qualname = scheduler_metadata["qualname"]
    if scheduler == "preflow" and scheduler_metadata["name"] not in {
        "PREFLOWScheduler",
        "AsyncPREFLOWScheduler",
    }:
        raise RuntimeError(
            "PREFLOW was requested but vLLM resolved scheduler_cls="
            f"{scheduler_cls_text!r} to {resolved_qualname!r}. Refusing to "
            "silently fall back."
        )
    if scheduler == "baseline" and "PREFLOW" in resolved_qualname.upper():
        raise RuntimeError(f"Baseline run unexpectedly resolved PREFLOW: {resolved_qualname!r}.")


def effective_engine_configuration(engine: Any) -> dict[str, Any]:
    """Capture effective settings after vLLM-Ascend platform mutation."""

    vllm_config = engine.vllm_config
    model_config = vllm_config.model_config
    load_config = vllm_config.load_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    return {
        "model": getattr(model_config, "model", None),
        "tokenizer": getattr(model_config, "tokenizer", None),
        "dtype": config_value_to_text(getattr(model_config, "dtype", None)),
        "max_model_len": getattr(model_config, "max_model_len", None),
        "load_format": config_value_to_text(getattr(load_config, "load_format", None)),
        "tensor_parallel_size": getattr(
            parallel_config,
            "tensor_parallel_size",
            None,
        ),
        "pipeline_parallel_size": getattr(
            parallel_config,
            "pipeline_parallel_size",
            None,
        ),
        "gpu_memory_utilization": getattr(
            cache_config,
            "gpu_memory_utilization",
            None,
        ),
        "enable_prefix_caching": getattr(
            cache_config,
            "enable_prefix_caching",
            None,
        ),
        "max_num_batched_tokens": getattr(
            scheduler_config,
            "max_num_batched_tokens",
            None,
        ),
        "max_num_seqs": getattr(scheduler_config, "max_num_seqs", None),
        "max_num_partial_prefills": getattr(
            scheduler_config,
            "max_num_partial_prefills",
            None,
        ),
        "max_long_partial_prefills": getattr(
            scheduler_config,
            "max_long_partial_prefills",
            None,
        ),
        "enable_chunked_prefill": getattr(
            scheduler_config,
            "enable_chunked_prefill",
            None,
        ),
        "long_prefill_token_threshold": getattr(
            scheduler_config,
            "long_prefill_token_threshold",
            None,
        ),
        "scheduler_reserve_full_isl": getattr(
            scheduler_config,
            "scheduler_reserve_full_isl",
            None,
        ),
        "scheduling_policy": config_value_to_text(getattr(scheduler_config, "policy", None)),
        "scheduler_cls": config_value_to_text(getattr(scheduler_config, "scheduler_cls", None)),
        "resolved_scheduler_class": resolved_scheduler_class_metadata(scheduler_config),
        "additional_config": vllm_config.additional_config,
    }


def make_sampling_params(runtime: RuntimeImports, cfg: BenchmarkConfig, seed: int) -> Any:
    return runtime.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=0.0,
        seed=seed,
        ignore_eos=True,
        detokenize=False,
        output_kind=runtime.RequestOutputKind.DELTA,
    )


async def sleep_until(phase_start_s: float, offset_s: float) -> None:
    while True:
        remaining = phase_start_s + offset_s - time.perf_counter()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 0.050))


async def run_one_request(
    *,
    engine: Any,
    runtime: RuntimeImports,
    scheduler: str,
    seed: int,
    request_id: str,
    prompt_length: int,
    token_ids: list[int],
    token_seed: int | None,
    token_variant: int | None,
    scheduled_arrival_offset_s: float,
    phase_start_s: float,
    solo_ttft_s: float | None,
    tracker: ConcurrencyTracker | None,
    cfg: BenchmarkConfig,
    phase: str,
) -> dict[str, Any]:
    assert len(token_ids) == prompt_length
    await sleep_until(phase_start_s, scheduled_arrival_offset_s)

    sampling_seed = stable_u64("sampling", seed, request_id) % (2**31 - 1)
    sampling_params = make_sampling_params(runtime, cfg, sampling_seed)
    actual_submission_time_s = time.perf_counter() - phase_start_s
    if tracker is not None:
        tracker.on_submit(actual_submission_time_s)

    first_token_time_s: float | None = None
    completion_time_s: float | None = None
    token_output_events = 0
    success = False
    error = ""

    try:
        async for output in engine.generate(
            request_id=request_id,
            prompt={"type": "token", "prompt_token_ids": token_ids},
            sampling_params=sampling_params,
        ):
            now_s = time.perf_counter() - phase_start_s
            if output.prompt_token_ids is None:
                raise RuntimeError("Exact prompt-token output was not returned.")
            if len(output.prompt_token_ids) != prompt_length:
                raise RuntimeError(f"Expected prompt length {prompt_length}, got {len(output.prompt_token_ids)}.")

            generated_now = sum(len(completion.token_ids) for completion in output.outputs)
            if generated_now > 0:
                token_output_events += 1
                if first_token_time_s is None:
                    first_token_time_s = now_s

            if output.finished:
                completion_time_s = now_s
                break

        if first_token_time_s is None:
            raise RuntimeError("Request completed without a first generated token.")
        if token_output_events != 1:
            raise RuntimeError(f"Expected exactly one first-token event, got {token_output_events}.")
        if completion_time_s is None:
            completion_time_s = time.perf_counter() - phase_start_s
        success = True
    except Exception as exc:  # noqa: BLE001 - record request-level failures.
        completion_time_s = time.perf_counter() - phase_start_s
        error = f"{exc.__class__.__name__}: {exc}"
    finally:
        if tracker is not None and completion_time_s is not None:
            tracker.on_finish(completion_time_s)

    ttft_s = first_token_time_s - actual_submission_time_s if first_token_time_s is not None else None
    relative_slowdown = ttft_s / solo_ttft_s if success and ttft_s is not None and solo_ttft_s is not None else None
    return {
        "scheduler": scheduler,
        "seed": seed,
        "phase": phase,
        "request_id": request_id,
        "prompt_length": prompt_length,
        "token_seed": token_seed,
        "token_variant": token_variant,
        "scheduled_arrival_offset_s": scheduled_arrival_offset_s,
        "actual_submission_time_s": actual_submission_time_s,
        "first_token_time_s": first_token_time_s,
        "completion_time_s": completion_time_s,
        "ttft_s": ttft_s,
        "solo_ttft_s": solo_ttft_s,
        "relative_slowdown": relative_slowdown,
        "success": success,
        "error": error,
    }


# =============================================================================
# Metrics
# =============================================================================


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "stddev": None,
            "min": None,
            "max": None,
            "p95": None,
            "p99": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def bucket_for_length(length: int, cfg: BenchmarkConfig) -> str:
    for name, lo, hi in cfg.length_buckets:
        if lo <= length <= hi:
            return name
    return "unbucketed"


def summarize_records(
    records: list[dict[str, Any]],
    cfg: BenchmarkConfig,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    measured = [row for row in records if row["phase"] == "measured"]
    successes = [row for row in measured if row["success"]]
    failures = [row for row in measured if not row["success"]]
    slowdowns = [row["relative_slowdown"] for row in successes if row["relative_slowdown"] is not None]
    ttfts = [row["ttft_s"] for row in successes if row["ttft_s"] is not None]

    by_length = {}
    for length in cfg.prompt_lengths:
        rows = [row for row in successes if row["prompt_length"] == length]
        values = [row["relative_slowdown"] for row in rows if row["relative_slowdown"] is not None]
        by_length[str(length)] = {
            "request_count": len(rows),
            "mean_relative_slowdown": statistics.fmean(values) if values else None,
            "p95_relative_slowdown": percentile(values, 95),
            "max_relative_slowdown": max(values) if values else None,
        }

    by_bucket = {}
    for name, _, _ in cfg.length_buckets:
        rows = [row for row in successes if bucket_for_length(row["prompt_length"], cfg) == name]
        values = [row["relative_slowdown"] for row in rows if row["relative_slowdown"] is not None]
        by_bucket[name] = {
            "request_count": len(rows),
            "mean_relative_slowdown": statistics.fmean(values) if values else None,
            "p95_relative_slowdown": percentile(values, 95),
            "max_relative_slowdown": max(values) if values else None,
        }

    submissions = [row["actual_submission_time_s"] for row in measured]
    completions = [row["completion_time_s"] for row in measured if row["completion_time_s"] is not None]
    first_submission = min(submissions) if submissions else 0.0
    last_submission = max(submissions) if submissions else 0.0
    last_completion = max(completions) if completions else last_submission
    experiment_wall_time = max(0.0, last_completion - first_submission)
    submission_span = max(last_submission - first_submission, 1e-12)
    achieved_submission_rate = len(measured) / submission_span if len(measured) > 1 else None
    completion_throughput = len(successes) / experiment_wall_time if experiment_wall_time > 0 else None
    near_one_count = sum(1 for value in slowdowns if value <= cfg.near_one_slowdown_threshold)

    summary = {
        "request_count": len(measured),
        "successful_request_count": len(successes),
        "failed_request_count": len(failures),
        "failed_request_fraction": len(failures) / len(measured) if measured else 0.0,
        "relative_slowdown": describe(slowdowns),
        "mean_relative_slowdown": statistics.fmean(slowdowns) if slowdowns else None,
        "median_relative_slowdown": statistics.median(slowdowns) if slowdowns else None,
        "p95_relative_slowdown": percentile(slowdowns, 95),
        "p99_relative_slowdown": percentile(slowdowns, 99),
        "max_relative_slowdown": max(slowdowns) if slowdowns else None,
        "arithmetic_mean_ttft_s": statistics.fmean(ttfts) if ttfts else None,
        "offered_request_rate": (cfg.target_request_rate if cfg.arrival_mode == "poisson" else "all_at_once"),
        "achieved_submission_rate": achieved_submission_rate,
        "completion_throughput": completion_throughput,
        "experiment_wall_time_s": experiment_wall_time,
        "by_prompt_length": by_length,
        "by_length_bucket": by_bucket,
        "slowdown_threshold_counts": {
            str(threshold): sum(1 for value in slowdowns if value > threshold) for threshold in cfg.slowdown_thresholds
        },
        "slowdown_threshold_fractions": {
            str(threshold): (sum(1 for value in slowdowns if value > threshold) / len(slowdowns) if slowdowns else None)
            for threshold in cfg.slowdown_thresholds
        },
        "near_one_slowdown_fraction": (near_one_count / len(slowdowns) if slowdowns else None),
    }
    if extra:
        summary.update(extra)
    if (
        summary["near_one_slowdown_fraction"] is not None
        and summary["near_one_slowdown_fraction"] >= cfg.near_one_warning_fraction
    ):
        summary["warning"] = (
            "Workload may be too lightly loaded to differentiate schedulers: "
            f"{summary['near_one_slowdown_fraction']:.1%} of successful requests "
            f"have slowdown <= {cfg.near_one_slowdown_threshold}."
        )
    return summary


def summarize_solo(rows: list[dict[str, Any]]) -> tuple[dict[int, float], dict[str, Any]]:
    baselines = {}
    summary = {}
    measured_rows = [row for row in rows if row["phase"] == "measured" and row["success"]]
    for length in sorted({row["prompt_length"] for row in measured_rows}):
        values = [row["ttft_s"] for row in measured_rows if row["prompt_length"] == length]
        stats = describe(values)
        baselines[length] = stats["median"]  # type: ignore[assignment]
        summary[str(length)] = stats
    return baselines, summary


def paired_comparison(
    baseline_records: list[dict[str, Any]],
    preflow_records: list[dict[str, Any]],
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    baseline_by_id = {
        row["request_id"]: row for row in baseline_records if row["phase"] == "measured" and row["success"]
    }
    preflow_by_id = {row["request_id"]: row for row in preflow_records if row["phase"] == "measured" and row["success"]}
    common_ids = sorted(set(baseline_by_id) & set(preflow_by_id))
    paired_rows = []
    for request_id in common_ids:
        base = baseline_by_id[request_id]
        pre = preflow_by_id[request_id]
        base_slowdown = base["relative_slowdown"]
        pre_slowdown = pre["relative_slowdown"]
        if base_slowdown is None or pre_slowdown is None or base_slowdown <= 0:
            continue
        improvement = (base_slowdown - pre_slowdown) / base_slowdown
        paired_rows.append(
            {
                "request_id": request_id,
                "prompt_length": base["prompt_length"],
                "baseline_slowdown": base_slowdown,
                "preflow_slowdown": pre_slowdown,
                "paired_improvement": improvement,
            }
        )

    improvements = [row["paired_improvement"] for row in paired_rows]
    wins = sum(1 for value in improvements if value > cfg.paired_tie_epsilon)
    losses = sum(1 for value in improvements if value < -cfg.paired_tie_epsilon)
    ties = len(improvements) - wins - losses

    by_length = {}
    for length in cfg.prompt_lengths:
        values = [row["paired_improvement"] for row in paired_rows if row["prompt_length"] == length]
        by_length[str(length)] = describe(values)

    baseline_slowdowns = [row["baseline_slowdown"] for row in paired_rows]
    preflow_slowdowns = [row["preflow_slowdown"] for row in paired_rows]
    baseline_p95 = percentile(baseline_slowdowns, 95)
    preflow_p95 = percentile(preflow_slowdowns, 95)
    p95_slowdown_improvement = (
        (baseline_p95 - preflow_p95) / baseline_p95 if baseline_p95 and preflow_p95 is not None else None
    )

    regressions = sorted(paired_rows, key=lambda row: row["paired_improvement"])
    gains = sorted(
        paired_rows,
        key=lambda row: row["paired_improvement"],
        reverse=True,
    )
    return {
        "matched_request_count": len(paired_rows),
        "mean_paired_improvement": (statistics.fmean(improvements) if improvements else None),
        "median_paired_improvement": (statistics.median(improvements) if improvements else None),
        "p95_slowdown_improvement": p95_slowdown_improvement,
        "preflow_wins": wins,
        "preflow_ties": ties,
        "preflow_losses": losses,
        "per_length_paired_improvement": by_length,
        "largest_preflow_regressions": regressions[: cfg.top_paired_extremes],
        "largest_preflow_gains": gains[: cfg.top_paired_extremes],
    }


# =============================================================================
# Benchmark Phases
# =============================================================================


async def run_solo_calibration(
    scheduler: str,
    cfg: BenchmarkConfig,
    runtime: RuntimeImports,
) -> tuple[list[dict[str, Any]], dict[int, float], dict[str, Any]]:
    engine = make_engine(runtime, scheduler, cfg)
    engine_effective_config = effective_engine_configuration(engine)
    bank = TokenBank(cfg)
    rows: list[dict[str, Any]] = []
    try:
        for length in cfg.prompt_lengths:
            for idx in range(cfg.num_solo_warmup_runs):
                token_seed = stable_u64("solo", "warmup", idx, length)
                token_ids = bank.get(length, token_seed)
                row = await run_one_request(
                    engine=engine,
                    runtime=runtime,
                    scheduler=scheduler,
                    seed=-1,
                    request_id=f"solo-{scheduler}-{length}-warmup-{idx}",
                    prompt_length=length,
                    token_ids=token_ids,
                    token_seed=token_seed,
                    token_variant=None,
                    scheduled_arrival_offset_s=0.0,
                    phase_start_s=time.perf_counter(),
                    solo_ttft_s=None,
                    tracker=None,
                    cfg=cfg,
                    phase="warmup",
                )
                rows.append(row)
                if not row["success"]:
                    raise RuntimeError(f"Solo warmup failed: {row['error']}")

            for idx in range(cfg.num_solo_measured_runs):
                token_seed = stable_u64("solo", "measured", idx, length)
                token_ids = bank.get(length, token_seed)
                row = await run_one_request(
                    engine=engine,
                    runtime=runtime,
                    scheduler=scheduler,
                    seed=-1,
                    request_id=f"solo-{scheduler}-{length}-measured-{idx}",
                    prompt_length=length,
                    token_ids=token_ids,
                    token_seed=token_seed,
                    token_variant=None,
                    scheduled_arrival_offset_s=0.0,
                    phase_start_s=time.perf_counter(),
                    solo_ttft_s=None,
                    tracker=None,
                    cfg=cfg,
                    phase="measured",
                )
                rows.append(row)
                if not row["success"]:
                    raise RuntimeError(f"Solo measurement failed: {row['error']}")
    finally:
        engine.shutdown()

    baselines, summary = summarize_solo(rows)
    missing = set(cfg.prompt_lengths) - set(baselines)
    if missing:
        raise RuntimeError(f"Missing solo baselines for lengths: {sorted(missing)}")
    return (
        rows,
        baselines,
        {
            "latency_by_prompt_length": summary,
            "effective_engine_config": engine_effective_config,
            "resolved_scheduler_class": engine_effective_config["resolved_scheduler_class"]["qualname"],
        },
    )


async def run_trace_phase(
    *,
    engine: Any,
    runtime: RuntimeImports,
    scheduler: str,
    seed: int,
    trace: list[TraceRequest],
    baselines: dict[int, float],
    bank: TokenBank,
    cfg: BenchmarkConfig,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    bank.materialize(trace)
    tracker = ConcurrencyTracker()
    phase_start_s = time.perf_counter()
    tasks = []
    for request in trace:
        solo_ttft = baselines.get(request.prompt_length)
        if request.phase == "measured" and solo_ttft is None:
            raise RuntimeError(f"No solo TTFT for length {request.prompt_length}.")
        token_ids = bank.get(request.prompt_length, request.token_seed)
        tasks.append(
            asyncio.create_task(
                run_one_request(
                    engine=engine,
                    runtime=runtime,
                    scheduler=scheduler,
                    seed=seed,
                    request_id=request.request_id,
                    prompt_length=request.prompt_length,
                    token_ids=token_ids,
                    token_seed=request.token_seed,
                    token_variant=request.token_variant,
                    scheduled_arrival_offset_s=request.scheduled_arrival_offset_s,
                    phase_start_s=phase_start_s,
                    solo_ttft_s=solo_ttft if request.phase == "measured" else None,
                    tracker=tracker,
                    cfg=cfg,
                    phase=request.phase,
                )
            )
        )
    records = await asyncio.gather(*tasks)
    wall_time_s = time.perf_counter() - phase_start_s
    return records, tracker.summary(wall_time_s)


async def run_mixed_workload(
    scheduler: str,
    seed: int,
    trace: list[TraceRequest],
    baselines: dict[int, float],
    cfg: BenchmarkConfig,
    runtime: RuntimeImports,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    engine = make_engine(runtime, scheduler, cfg)
    engine_effective_config = effective_engine_configuration(engine)
    bank = TokenBank(cfg)
    try:
        warmup_trace = [request for request in trace if request.phase == "warmup"]
        measured_trace = [request for request in trace if request.phase == "measured"]

        if warmup_trace:
            warmup_records, _ = await run_trace_phase(
                engine=engine,
                runtime=runtime,
                scheduler=scheduler,
                seed=seed,
                trace=warmup_trace,
                baselines=baselines,
                bank=bank,
                cfg=cfg,
            )
            warmup_errors = [row for row in warmup_records if not row["success"]]
            if warmup_errors:
                raise RuntimeError(f"Mixed warmup failed: {warmup_errors[0]['error']}")
        else:
            warmup_records = []

        measured_records, concurrency_summary = await run_trace_phase(
            engine=engine,
            runtime=runtime,
            scheduler=scheduler,
            seed=seed,
            trace=measured_trace,
            baselines=baselines,
            bank=bank,
            cfg=cfg,
        )
    finally:
        engine.shutdown()

    records = [*warmup_records, *measured_records]
    summary = summarize_records(records, cfg, extra=concurrency_summary)
    summary["scheduler"] = scheduler
    summary["seed"] = seed
    summary["effective_engine_config"] = engine_effective_config
    summary["resolved_scheduler_class"] = engine_effective_config["resolved_scheduler_class"]["qualname"]
    return records, summary


# =============================================================================
# Output
# =============================================================================


REQUEST_CSV_FIELDS = [
    "scheduler",
    "seed",
    "phase",
    "request_id",
    "prompt_length",
    "token_seed",
    "token_variant",
    "scheduled_arrival_offset_s",
    "actual_submission_time_s",
    "first_token_time_s",
    "completion_time_s",
    "ttft_s",
    "solo_ttft_s",
    "relative_slowdown",
    "success",
    "error",
]


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=json_default) + "\n")


def clean_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_csv_value(row.get(field)) for field in fields})


def create_output_dir(base: Path | None) -> Path:
    if base is not None:
        out_dir = base
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = OUTPUT_ROOT / f"preflow_slowdown_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def print_run_summary(summary: dict[str, Any]) -> None:
    print(
        "{scheduler:8s} seed={seed} n={n:4d} "
        "meanS={mean_s!s:>9} p95S={p95_s!s:>9} "
        "ttft={ttft!s:>9} thr={thr!s:>9} scheduler_cls={scheduler_cls}".format(
            scheduler=summary["scheduler"],
            seed=summary["seed"],
            n=summary["request_count"],
            mean_s=format_float(summary["mean_relative_slowdown"]),
            p95_s=format_float(summary["p95_relative_slowdown"]),
            ttft=format_float(summary["arithmetic_mean_ttft_s"]),
            thr=format_float(summary["completion_throughput"]),
            scheduler_cls=summary["resolved_scheduler_class"],
        )
    )
    if warning := summary.get("warning"):
        print(f"  WARNING: {warning}")


def format_float(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    return f"{float(value):.4f}"


def make_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PREFLOW Slowdown Benchmark Summary",
        "",
        "Dummy weights were enabled with `load_format='dummy'`; these results "
        "validate scheduler behavior, not final real-model latency.",
        "",
        "## Runs",
        "",
        "| scheduler | resolved scheduler class | seed | requests | "
        "mean slowdown | p95 slowdown | mean TTFT | throughput | failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, run in sorted(summary["mixed_runs"].items()):
        lines.append(
            "| {scheduler} | `{scheduler_cls}` | {seed} | {n} | {mean_s} | {p95_s} | {ttft} | {thr} | {fail} |".format(
                scheduler=run["scheduler"],
                scheduler_cls=run["resolved_scheduler_class"],
                seed=run["seed"],
                n=run["request_count"],
                mean_s=format_float(run["mean_relative_slowdown"]),
                p95_s=format_float(run["p95_relative_slowdown"]),
                ttft=format_float(run["arithmetic_mean_ttft_s"]),
                thr=format_float(run["completion_throughput"]),
                fail=run["failed_request_count"],
            )
        )
    if summary.get("paired"):
        lines.extend(["", "## Paired Comparisons", ""])
        for seed, paired in sorted(summary["paired"].items()):
            lines.append(
                "- seed {seed}: mean improvement={mean}, median improvement={median}, "
                "p95 slowdown improvement={p95}, wins/ties/losses={wins}/{ties}/{losses}".format(
                    seed=seed,
                    mean=format_float(paired["mean_paired_improvement"]),
                    median=format_float(paired["median_paired_improvement"]),
                    p95=format_float(paired["p95_slowdown_improvement"]),
                    wins=paired["preflow_wins"],
                    ties=paired["preflow_ties"],
                    losses=paired["preflow_losses"],
                )
            )
    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================


async def run_benchmark(args: argparse.Namespace) -> None:
    cfg = make_config()
    schedulers = ["baseline", "preflow"] if args.scheduler == "both" else [args.scheduler]
    validate_config(cfg, schedulers)

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    out_dir = create_output_dir(args.output_dir)
    traces = make_workload_traces(cfg)
    trace_rows = [asdict(request) for seed in cfg.seeds for request in traces[seed]]
    write_jsonl(out_dir / "workload_trace.jsonl", trace_rows)

    runtime = import_runtime()
    config_payload = {
        "config": asdict(cfg),
        "selected_schedulers": schedulers,
        "vllm_checkout": str(repo_root() / "vllm"),
        "dummy_weight_note": (
            "load_format='dummy' initializes random weights for profiling; latencies are for scheduler comparison only."
        ),
    }
    write_json(out_dir / "config.json", config_payload)

    print("Using vLLM load_format='dummy'. Results are scheduler-behavior data,")
    print(f"not final real-model latency. Profile={cfg.run_profile!r}. Output: {out_dir}")

    solo_rows: list[dict[str, Any]] = []
    solo_summaries: dict[str, Any] = {}
    solo_baselines_by_scheduler: dict[str, dict[int, float]] = {}
    mixed_records: dict[tuple[str, int], list[dict[str, Any]]] = {}
    mixed_summaries: dict[str, Any] = {}

    for scheduler in schedulers:
        rows, baselines, solo_summary = await run_solo_calibration(
            scheduler,
            cfg,
            runtime,
        )
        solo_rows.extend(rows)
        solo_baselines_by_scheduler[scheduler] = baselines
        solo_summaries[scheduler] = solo_summary
        write_csv(out_dir / "solo_latencies.csv", solo_rows, REQUEST_CSV_FIELDS)

        for seed in cfg.seeds:
            records, run_summary = await run_mixed_workload(
                scheduler,
                seed,
                traces[seed],
                baselines,
                cfg,
                runtime,
            )
            mixed_records[(scheduler, seed)] = records
            mixed_summaries[f"{scheduler}_{seed}"] = run_summary
            write_csv(
                out_dir / f"requests_{scheduler}_{seed}.csv",
                records,
                REQUEST_CSV_FIELDS,
            )
            print_run_summary(run_summary)

    paired = {}
    if "baseline" in schedulers and "preflow" in schedulers:
        for seed in cfg.seeds:
            paired[str(seed)] = paired_comparison(
                mixed_records[("baseline", seed)],
                mixed_records[("preflow", seed)],
                cfg,
            )

    summary = {
        "config": asdict(cfg),
        "solo": solo_summaries,
        "solo_baselines": {
            scheduler: {str(length): ttft for length, ttft in baselines.items()}
            for scheduler, baselines in solo_baselines_by_scheduler.items()
        },
        "mixed_runs": mixed_summaries,
        "paired": paired,
    }
    config_payload["effective_engine_configs"] = {
        "solo": {scheduler: solo_summaries[scheduler]["effective_engine_config"] for scheduler in solo_summaries},
        "mixed": {key: run_summary["effective_engine_config"] for key, run_summary in mixed_summaries.items()},
    }
    write_json(out_dir / "config.json", config_payload)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(make_summary_markdown(summary), encoding="utf-8")

    if paired:
        print("\nPaired PREFLOW improvements:")
        for seed, data in paired.items():
            print(
                "seed={seed} mean={mean} median={median} p95_slowdown={p95} "
                "wins/ties/losses={wins}/{ties}/{losses}".format(
                    seed=seed,
                    mean=format_float(data["mean_paired_improvement"]),
                    median=format_float(data["median_paired_improvement"]),
                    p95=format_float(data["p95_slowdown_improvement"]),
                    wins=data["preflow_wins"],
                    ties=data["preflow_ties"],
                    losses=data["preflow_losses"],
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PREFLOW and vLLM v0.25.1 baseline relative slowdown.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["baseline", "preflow", "both"],
        default="both",
        help="Scheduler run set. Workload settings remain editable in this file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional exact output directory. Defaults to benchmark_results/...",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_benchmark(parse_args()))


if __name__ == "__main__":
    main()
