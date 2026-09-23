#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
"""Run a synthetic slowdown workload through a live vLLM PD router.

The router, prefill workers, and decode workers must already be running. This
client never changes their policies or scheduler settings. The policy and
``run.local_scheduler`` fields are result labels that must describe the deployment.

Examples:

  # Inspect a 32-request prototype without sending traffic.
  python benchmark_preflow_slowdown.py run.mode=dry_run workload.request_count=32

  # Run one MMPP baseline against the live router.
  python benchmark_preflow_slowdown.py run.mode=benchmark \
      router.base_url=http://127.0.0.1:8090 \
      run.prefill_policy=power_of_two run.decode_policy=power_of_two \
      run.local_scheduler=fcfs \
      workload.request_count=1000 arrival.mode=mmpp \
      arrival.target_request_rate=0.5 request.decode_tokens=1 \
      'observability.prefill_metrics_urls=[http://prefill-0:8000/metrics]' \
      'observability.decode_metrics_urls=[http://decode-0:8000/metrics]'

  # Run the same workload after starting P workers with --enable-preflow.
  python benchmark_preflow_slowdown.py run.mode=benchmark \
      run.local_scheduler=preflow

  # Sweep offered load. Hydra runs these jobs sequentially by default.
  python benchmark_preflow_slowdown.py -m run.mode=benchmark \
      arrival.target_request_rate=0.25,0.5,0.75,1.0

  # Calibrate cold solo TTFT/E2E latency. Restart the deployment before the
  # measured run, then pass calibration.baseline_path=<...>/solo_baselines.json.
  python benchmark_preflow_slowdown.py run.mode=calibrate

Queue gauges are sampled from each URL in ``observability.prefill_metrics_urls``
and ``observability.decode_metrics_urls``. Configure those as the workers'
Prometheus endpoints, not their OpenAI API endpoints.

By default, client results are written below ``benchmark_output/``. To bundle
the scheduler traces with a client result, set
``output.queue_stats_source_dir=<setup-output>/queue_stats``. The client copies
the trace snapshot to ``seed-*/queue_stats/`` for automatic analysis.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import random
import re
import statistics
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import hydra
import urllib3
import yaml
from hydra.core.config_store import ConfigStore
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Hydra configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    mode: str = "dry_run"  # dry_run | benchmark | calibrate
    name: str = "pd_synthetic"
    prefill_policy: str = "round_robin"
    decode_policy: str = "round_robin"
    local_scheduler: str = "fcfs"
    seeds: list[int] = field(default_factory=lambda: [20260728])
    require_worker_metrics: bool = True
    allow_state_carryover_across_seeds: bool = False


@dataclass
class WorkloadConfig:
    request_count: int = 1_000
    prompt_lengths: list[int] = field(
        default_factory=lambda: [4_096, 8_192, 16_384, 32_768, 65_536, 100_000]
    )
    prompt_length_probabilities: list[float] = field(
        default_factory=lambda: [0.30, 0.25, 0.20, 0.14, 0.08, 0.03]
    )
    prompt_length_order: str = "space_filling"  # random_shuffle | space_filling
    space_filling_window_size: int = 100
    prompt_variants_per_length: int = 4
    workload_seed: int = 20260728
    max_model_len: int = 131_072


@dataclass
class ArrivalConfig:
    mode: str = "mmpp"  # all_at_once | poisson | bursty | mmpp
    target_request_rate: float = 0.5
    burst_size: int = 16
    burst_rate_multiplier: float = 8.0
    mmpp_background_rate_fraction: float = 0.20
    mmpp_burst_rate_multiplier: float = 6.0
    mmpp_mean_burst_requests: float = 12.0
    time_scale: float = 1.0


@dataclass
class RouterConfig:
    base_url: str = "http://127.0.0.1:8090"
    completions_path: str = "/v1/completions"
    metrics_url: str = "http://127.0.0.1:29000/metrics"
    api_key: str = ""
    request_id_header: str = "X-Request-ID"
    session_id_header: str = "X-Session-ID"
    health_path: str = "/health"
    models_path: str = "/v1/models"
    server_info_path: str = "/get_server_info"
    strict_preflight: bool = True
    require_dummy_verification: bool = False


@dataclass
class RequestConfig:
    model: str = "Qwen3-30B-A3B"
    decode_tokens: int = 1
    temperature: float = 0.0
    ignore_eos: bool = True
    stream: bool = True
    token_id_low: int = 1_000
    token_id_high_exclusive: int = 30_000
    special_token_ids: list[int] = field(default_factory=list)
    token_seed: int = 12_345
    prompt_build_lead_s: float = 1.0
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionConfig:
    max_connections: int = 256
    connect_timeout_s: float = 30.0
    request_timeout_s: float = 1_800.0
    progress_interval_requests: int = 250
    progress_interval_s: float = 20.0
    max_failure_fraction: float = 0.01


@dataclass
class ObservabilityConfig:
    enabled: bool = True
    poll_interval_s: float = 0.1
    scrape_timeout_s: float = 5.0
    prefill_metrics_urls: list[str] = field(default_factory=list)
    decode_metrics_urls: list[str] = field(default_factory=list)
    include_regex: str = r"^(vllm_router_|vllm:|process_)"
    write_raw_samples: bool = True


@dataclass
class CalibrationConfig:
    baseline_path: str = ""
    lengths: list[int] = field(
        default_factory=lambda: [4_096, 8_192, 16_384, 32_768, 65_536, 100_000]
    )
    warmup_repetitions: int = 1
    measured_repetitions: int = 3


@dataclass
class OutputConfig:
    directory: str = ""
    queue_stats_source_dir: str = ""
    write_csv: bool = True
    write_parquet: bool = True
    overwrite: bool = False


@dataclass
class SyntheticBenchmarkConfig:
    run: RunConfig = field(default_factory=RunConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    request: RequestConfig = field(default_factory=RequestConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


ConfigStore.instance().store(name="preflow_router_synthetic", node=SyntheticBenchmarkConfig)


# ---------------------------------------------------------------------------
# Deterministic workload generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticRequest:
    seed: int
    request_index: int
    request_id: str
    prompt_length: int
    token_variant: int
    scheduled_arrival_offset_s: float


def stable_u64(*parts: object) -> int:
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def normalize_arrival_span(offsets: list[float], target_rate: float) -> list[float]:
    if len(offsets) <= 1:
        return [0.0] * len(offsets)
    first = offsets[0]
    normalized = [offset - first for offset in offsets]
    span = normalized[-1]
    if span <= 0 or target_rate <= 0:
        raise ValueError("arrival offsets and target rate must be positive")
    scale = len(offsets) / target_rate / span
    return [offset * scale for offset in normalized]


def make_mmpp_offsets(count: int, seed: int, cfg: ArrivalConfig) -> list[float]:
    """Exact MMPP construction used by the original PREFLOW benchmark."""
    if count <= 1:
        return [0.0] * count
    low_rate = cfg.target_request_rate * cfg.mmpp_background_rate_fraction
    high_rate = cfg.target_request_rate * cfg.mmpp_burst_rate_multiplier
    burst_probability = (cfg.target_request_rate - low_rate) / (high_rate - low_rate)
    mean_burst_duration = cfg.mmpp_mean_burst_requests / high_rate
    mean_background_duration = mean_burst_duration * (1 - burst_probability) / burst_probability
    rng = random.Random(stable_u64("arrival", "mmpp", seed, "measured"))
    in_burst = rng.random() < burst_probability
    current = 0.0
    offsets: list[float] = []
    while len(offsets) < count:
        rate = high_rate if in_burst else low_rate
        mean_duration = mean_burst_duration if in_burst else mean_background_duration
        end = current + rng.expovariate(1.0 / mean_duration)
        if rate > 0:
            arrival = current
            while len(offsets) < count:
                arrival += rng.expovariate(rate)
                if arrival > end:
                    break
                offsets.append(arrival)
        current = end
        in_burst = not in_burst
    return normalize_arrival_span(sorted(offsets), cfg.target_request_rate)


def make_bursty_offsets(count: int, cfg: ArrivalConfig) -> list[float]:
    if count <= 1:
        return [0.0] * count
    burst_rate = cfg.target_request_rate * cfg.burst_rate_multiplier
    intra_burst_interval = 1.0 / burst_rate
    burst_period = cfg.burst_size / cfg.target_request_rate
    if (cfg.burst_size - 1) * intra_burst_interval >= burst_period:
        raise ValueError("bursty parameters leave no inter-burst gap")
    return [
        (index // cfg.burst_size) * burst_period + (index % cfg.burst_size) * intra_burst_interval
        for index in range(count)
    ]


def make_arrival_offsets(count: int, seed: int, cfg: ArrivalConfig) -> list[float]:
    if cfg.mode == "all_at_once":
        offsets = [0.0] * count
    elif cfg.mode == "bursty":
        offsets = make_bursty_offsets(count, cfg)
    elif cfg.mode == "mmpp":
        offsets = make_mmpp_offsets(count, seed, cfg)
    elif cfg.mode == "poisson":
        rng = random.Random(stable_u64("arrival", seed, "measured"))
        offsets = [0.0]
        for _ in range(1, count):
            offsets.append(offsets[-1] + rng.expovariate(cfg.target_request_rate))
    else:
        raise ValueError(f"unsupported arrival.mode={cfg.mode!r}")
    return [offset * cfg.time_scale for offset in offsets]


def length_counts(count: int, workload: WorkloadConfig) -> list[int]:
    quotas = [count * probability for probability in workload.prompt_length_probabilities]
    counts = [math.floor(quota) for quota in quotas]
    remaining = count - sum(counts)
    order = sorted(
        range(len(quotas)),
        key=lambda index: (quotas[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    if sum(counts) != count:
        raise AssertionError("failed to allocate exact prompt-length counts")
    return counts


def shuffled(values: list[int], *seed_parts: object) -> list[int]:
    result = list(values)
    random.Random(stable_u64(*seed_parts)).shuffle(result)
    return result


def random_length_sequence(count: int, cfg: WorkloadConfig) -> list[int]:
    values: list[int] = []
    for length, occurrences in zip(cfg.prompt_lengths, length_counts(count, cfg), strict=True):
        values.extend([length] * occurrences)
    return shuffled(
        values,
        "request_order",
        cfg.workload_seed,
        count,
        cfg.prompt_lengths,
        cfg.prompt_length_probabilities,
    )


def space_filling_length_sequence(count: int, cfg: WorkloadConfig) -> list[int]:
    remaining_counts = length_counts(count, cfg)
    result: list[int] = []
    window_size = min(cfg.space_filling_window_size, count)
    window_index = 0
    while len(result) < count:
        remaining_total = count - len(result)
        current_size = min(window_size, remaining_total)
        quotas = [current_size * value / remaining_total for value in remaining_counts]
        window_counts = [math.floor(quota) for quota in quotas]
        slots = current_size - sum(window_counts)
        tie_order = shuffled(
            list(range(len(remaining_counts))),
            "space_filling_ties",
            cfg.workload_seed,
            count,
            window_index,
        )
        for index in sorted(
            tie_order, key=lambda item: quotas[item] - window_counts[item], reverse=True
        ):
            if slots == 0:
                break
            if window_counts[index] < remaining_counts[index]:
                window_counts[index] += 1
                slots -= 1
        if slots:
            raise ValueError("could not fill a prompt-length window")
        window: list[int] = []
        for length, occurrences in zip(cfg.prompt_lengths, window_counts, strict=True):
            window.extend([length] * occurrences)
        result.extend(
            shuffled(
                window,
                "space_filling_window",
                cfg.workload_seed,
                count,
                window_index,
            )
        )
        remaining_counts = [
            remaining - used
            for remaining, used in zip(remaining_counts, window_counts, strict=True)
        ]
        window_index += 1
    return result


def make_length_sequence(count: int, cfg: WorkloadConfig) -> list[int]:
    if cfg.prompt_length_order == "random_shuffle":
        return random_length_sequence(count, cfg)
    if cfg.prompt_length_order == "space_filling":
        return space_filling_length_sequence(count, cfg)
    raise ValueError(f"unsupported workload.prompt_length_order={cfg.prompt_length_order!r}")


def make_workload(seed: int, cfg: SyntheticBenchmarkConfig) -> list[SyntheticRequest]:
    lengths = make_length_sequence(cfg.workload.request_count, cfg.workload)
    offsets = make_arrival_offsets(cfg.workload.request_count, seed, cfg.arrival)
    return [
        SyntheticRequest(
            seed=seed,
            request_index=index,
            request_id=f"seed{seed}-measured-{index:06d}",
            prompt_length=length,
            token_variant=stable_u64("variant", cfg.workload.workload_seed, index, length)
            % cfg.workload.prompt_variants_per_length,
            scheduled_arrival_offset_s=offsets[index],
        )
        for index, length in enumerate(lengths)
    ]


# ---------------------------------------------------------------------------
# Live execution and output
# ---------------------------------------------------------------------------


def validate_config(cfg: SyntheticBenchmarkConfig) -> None:
    if cfg.run.mode not in {"dry_run", "benchmark", "calibrate"}:
        raise ValueError("run.mode must be dry_run, benchmark, or calibrate")
    if not cfg.run.seeds:
        raise ValueError("run.seeds must not be empty")
    if len(cfg.run.seeds) > 1 and not cfg.run.allow_state_carryover_across_seeds:
        raise ValueError(
            "multiple seeds would share live worker/router cache state; run one seed per "
            "deployment reset, or explicitly set run.allow_state_carryover_across_seeds=true"
        )
    if cfg.workload.request_count <= 0:
        raise ValueError("workload.request_count must be positive")
    if len(cfg.workload.prompt_lengths) != len(cfg.workload.prompt_length_probabilities):
        raise ValueError("prompt lengths and probabilities must have equal length")
    if not cfg.workload.prompt_lengths or any(value <= 0 for value in cfg.workload.prompt_lengths):
        raise ValueError("prompt lengths must be positive")
    if not math.isclose(sum(cfg.workload.prompt_length_probabilities), 1.0, abs_tol=1e-9):
        raise ValueError("prompt-length probabilities must sum to 1")
    if cfg.workload.prompt_variants_per_length <= 0:
        raise ValueError("prompt_variants_per_length must be positive")
    if cfg.workload.space_filling_window_size <= 0:
        raise ValueError("space_filling_window_size must be positive")
    context_lengths = (
        cfg.calibration.lengths if cfg.run.mode == "calibrate" else cfg.workload.prompt_lengths
    )
    if max(context_lengths) + cfg.request.decode_tokens > cfg.workload.max_model_len:
        raise ValueError("a prompt plus fixed decode length exceeds max_model_len")
    if cfg.arrival.mode != "all_at_once" and cfg.arrival.target_request_rate <= 0:
        raise ValueError("target_request_rate must be positive")
    if cfg.arrival.time_scale <= 0:
        raise ValueError("arrival.time_scale must be positive")
    if cfg.arrival.mode == "mmpp" and not (
        0 <= cfg.arrival.mmpp_background_rate_fraction < 1
        and cfg.arrival.mmpp_burst_rate_multiplier > 1
        and cfg.arrival.mmpp_mean_burst_requests > 0
    ):
        raise ValueError("invalid MMPP parameters")
    if cfg.run.mode == "benchmark" and cfg.run.require_worker_metrics:
        if not cfg.observability.enabled:
            raise ValueError("worker queue measurement requires observability.enabled=true")
        if (
            not (cfg.output.write_csv or cfg.output.write_parquet)
            or not cfg.observability.write_raw_samples
        ):
            raise ValueError(
                "dedicated node_queue_samples output requires at least one of "
                "output.write_csv/output.write_parquet and observability.write_raw_samples=true"
            )
        if not cfg.observability.prefill_metrics_urls:
            raise ValueError(
                "prefill queue measurement requires observability.prefill_metrics_urls; "
                "set run.require_worker_metrics=false to run without it"
            )
        if not cfg.observability.decode_metrics_urls:
            raise ValueError(
                "decode queue measurement requires observability.decode_metrics_urls; "
                "set run.require_worker_metrics=false to run without it"
            )
    if cfg.request.decode_tokens <= 0:
        raise ValueError("request.decode_tokens must be positive")
    if cfg.execution.max_connections <= 0:
        raise ValueError("execution.max_connections must be positive")
    if cfg.observability.poll_interval_s <= 0:
        raise ValueError("observability.poll_interval_s must be positive")
    if cfg.request.token_id_low >= cfg.request.token_id_high_exclusive:
        raise ValueError("request token range is empty")


class TokenBank:
    """Bounded set of deterministic full-prompt variants used by the workload."""

    def __init__(self, cfg: RequestConfig):
        self.cfg = cfg
        self.special = set(cfg.special_token_ids)
        self.cache: dict[tuple[int, int], tuple[int, ...]] = {}
        self.lock = threading.Lock()

    def prompt(self, request: SyntheticRequest) -> list[int]:
        key = (request.prompt_length, request.token_variant)
        with self.lock:
            cached = self.cache.get(key)
        if cached is None:
            rng = random.Random(
                stable_u64(
                    "synthetic-token-prompt",
                    self.cfg.token_seed,
                    request.prompt_length,
                    request.token_variant,
                )
            )
            values = []
            while len(values) < request.prompt_length:
                token = rng.randrange(self.cfg.token_id_low, self.cfg.token_id_high_exclusive)
                if token not in self.special:
                    values.append(token)
            generated = tuple(values)
            with self.lock:
                cached = self.cache.setdefault(key, generated)
        return list(cached)


def join_url(base: str, path: str) -> str:
    return urljoin(base.rstrip("/") + "/", path.lstrip("/"))


def http_timeout(cfg: ExecutionConfig) -> urllib3.Timeout:
    return urllib3.Timeout(connect=cfg.connect_timeout_s, read=cfg.request_timeout_s)


def read_sse_event(line: bytes) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == b"[DONE]":
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def send_completion_sync(
    pool: urllib3.PoolManager,
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout: urllib3.Timeout,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_token: float | None = None
    completed: float | None = None
    status = 0
    error = ""
    response_headers: dict[str, str] = {}
    usage: dict[str, Any] = {}
    token_events = 0
    response = None
    try:
        response = pool.request(
            "POST",
            url,
            body=body,
            headers=headers,
            timeout=timeout,
            retries=False,
            preload_content=False,
        )
        status = int(response.status)
        response_headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower().startswith(("x-", "server-timing", "content-type"))
        }
        if status < 200 or status >= 300:
            error_body = response.read(16_384).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {status}: {error_body}")
        if "text/event-stream" in response.headers.get("content-type", ""):
            buffer = b""
            for chunk in response.stream(amt=65_536, decode_content=True):
                buffer += chunk
                while b"\n" in buffer:
                    raw_line, buffer = buffer.split(b"\n", 1)
                    event = read_sse_event(raw_line)
                    if event is None:
                        continue
                    if isinstance(event.get("usage"), dict):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if choices and any(choice.get("finish_reason") is None for choice in choices):
                        token_events += 1
                        if first_token is None:
                            first_token = time.perf_counter()
        else:
            payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
            first_token = time.perf_counter()
            token_events = int(usage.get("completion_tokens", 1))
        completed = time.perf_counter()
        if first_token is None:
            raise RuntimeError("response completed without an observable token event")
    except Exception as exc:  # noqa: BLE001 - failures belong in request-level output.
        completed = time.perf_counter()
        error = f"{exc.__class__.__name__}: {exc}"
    finally:
        if response is not None:
            response.release_conn()
    return {
        "request_start_perf": started,
        "first_token_perf": first_token,
        "completion_perf": completed,
        "http_status": status,
        "success": not error,
        "error": error,
        "response_headers_json": json.dumps(response_headers, sort_keys=True),
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
        "usage_total_tokens": usage.get("total_tokens"),
        "stream_token_events": token_events,
    }


def request_payload(
    cfg: SyntheticBenchmarkConfig,
    request: SyntheticRequest,
    prompt: list[int],
) -> tuple[bytes, dict[str, str]]:
    body: dict[str, Any] = {
        "model": cfg.request.model,
        "prompt": prompt,
        "max_tokens": cfg.request.decode_tokens,
        "min_tokens": cfg.request.decode_tokens,
        "temperature": cfg.request.temperature,
        "ignore_eos": cfg.request.ignore_eos,
        "stream": cfg.request.stream,
        "stream_options": {"include_usage": True},
        "seed": stable_u64("decode", request.seed, request.request_id) % (2**31 - 1),
    }
    body.update(cfg.request.extra_body)
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if cfg.request.stream else "application/json",
        cfg.router.request_id_header: request.request_id,
        cfg.router.session_id_header: (
            f"length-{request.prompt_length}-variant-{request.token_variant}"
        ),
    }
    if cfg.router.api_key:
        headers["Authorization"] = f"Bearer {cfg.router.api_key}"
    return json.dumps(body, separators=(",", ":")).encode(), headers


def get_sync(pool: urllib3.PoolManager, url: str, timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = pool.request(
            "GET",
            url,
            timeout=urllib3.Timeout(connect=timeout_s, read=timeout_s),
            retries=False,
        )
        text = response.data.decode("utf-8", errors="replace")
        try:
            body: Any = json.loads(text)
        except json.JSONDecodeError:
            body = text
        return {
            "url": url,
            "status": response.status,
            "latency_s": time.perf_counter() - started,
            "body": body,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - report actionable preflight errors.
        return {
            "url": url,
            "status": None,
            "latency_s": time.perf_counter() - started,
            "body": None,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


async def preflight(cfg: SyntheticBenchmarkConfig) -> dict[str, Any]:
    pool = urllib3.PoolManager(num_pools=4, maxsize=4)
    paths = (cfg.router.health_path, cfg.router.models_path, cfg.router.server_info_path)
    rows = await asyncio.gather(
        *(
            asyncio.to_thread(
                get_sync,
                pool,
                join_url(cfg.router.base_url, path),
                cfg.execution.connect_timeout_s,
            )
            for path in paths
        )
    )
    result = {path: row for path, row in zip(paths, rows, strict=True)}
    if cfg.router.strict_preflight and result[cfg.router.health_path]["status"] != 200:
        raise RuntimeError(f"router preflight failed: {result[cfg.router.health_path]}")
    serialized = json.dumps(result.get(cfg.router.server_info_path, {}), sort_keys=True).lower()
    dummy_verified = '"load_format": "dummy"' in serialized or '"load_format":"dummy"' in serialized
    result["dummy_weights"] = {
        "expected": True,
        "verified_from_server_info": dummy_verified,
        "note": "The client cannot configure weights; every worker must use load_format=dummy.",
    }
    if cfg.router.require_dummy_verification and not dummy_verified:
        raise RuntimeError("could not verify load_format=dummy through /get_server_info")
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    low, high = math.floor(rank), math.ceil(rank)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
    numbers = [
        float(value) for value in values if value is not None and math.isfinite(float(value))
    ]
    if not numbers:
        return {key: None for key in ("count", "mean", "median", "p95", "p99", "max")}
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p95": percentile(numbers, 0.95),
        "p99": percentile(numbers, 0.99),
        "max": max(numbers),
    }


def load_baselines(path: Path | None) -> dict[str, dict[int, float]]:
    if path is None:
        return {"ttft_s": {}, "e2e_latency_s": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        metric: {int(length): float(value) for length, value in payload.get(metric, {}).items()}
        for metric in ("ttft_s", "e2e_latency_s")
    }


def interpolate_baseline(length: int, baselines: dict[int, float]) -> float | None:
    if not baselines:
        return None
    if length in baselines:
        return baselines[length]
    points = sorted(baselines)
    if length <= points[0]:
        return baselines[points[0]]
    if length >= points[-1]:
        return baselines[points[-1]]
    upper_index = next(index for index, point in enumerate(points) if point > length)
    lower, upper = points[upper_index - 1], points[upper_index]
    weight = (math.log(length) - math.log(lower)) / (math.log(upper) - math.log(lower))
    return baselines[lower] * (1 - weight) + baselines[upper] * weight


METRIC_LINE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)(?:\s+\d+)?$"
)


def parse_prometheus(text: str, include: re.Pattern[str]) -> list[tuple[str, str, float]]:
    rows = []
    for line in text.splitlines():
        match = METRIC_LINE.match(line.strip())
        if not match or not include.search(match.group(1)):
            continue
        try:
            value = float(match.group(3))
        except ValueError:
            continue
        if math.isfinite(value):
            rows.append((match.group(1), match.group(2) or "", value))
    return rows


@dataclass(frozen=True)
class MetricsEndpoint:
    role: str
    name: str
    url: str


@dataclass
class MetricStats:
    samples: int = 0
    first_time_s: float = 0.0
    first: float = 0.0
    last: float = 0.0
    total: float = 0.0
    maximum: float = -math.inf
    active_samples: int = 0

    def add(self, time_s: float, value: float) -> None:
        if self.samples == 0:
            self.first_time_s = time_s
            self.first = value
        self.samples += 1
        self.last = value
        self.total += value
        self.maximum = max(self.maximum, value)
        self.active_samples += int(value > 0)

    def series_summary(self, is_counter: bool) -> dict[str, float | int]:
        baseline = self.first if self.first_time_s <= 1e-9 or not is_counter else 0.0
        return {
            "samples": self.samples,
            "first": self.first,
            "last": self.last,
            "delta": self.last - baseline,
            "mean": self.total / self.samples,
            "max": self.maximum,
        }

    def gauge_summary(self) -> dict[str, float | int]:
        return {
            "samples": self.samples,
            "mean": self.total / self.samples,
            "max": self.maximum,
            "active_fraction": self.active_samples / self.samples,
        }


class MetricsCollector:
    RAW_COLUMNS = ("time_s", "role", "endpoint", "url", "metric", "labels", "value")

    def __init__(self, cfg: SyntheticBenchmarkConfig, output_dir: Path):
        self._outer = cfg
        self.cfg = cfg.observability
        self.include = re.compile(self.cfg.include_regex)
        self.pool = urllib3.PoolManager(num_pools=32, maxsize=32, block=True)
        self.endpoints: list[MetricsEndpoint] = []
        if cfg.router.metrics_url:
            self.endpoints.append(MetricsEndpoint("router", "router", cfg.router.metrics_url))
        self.endpoints.extend(
            MetricsEndpoint("prefill", f"prefill-{index}", url)
            for index, url in enumerate(self.cfg.prefill_metrics_urls)
        )
        self.endpoints.extend(
            MetricsEndpoint("decode", f"decode-{index}", url)
            for index, url in enumerate(self.cfg.decode_metrics_urls)
        )
        self.errors: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.series: dict[tuple[str, str, str, str], MetricStats] = defaultdict(MetricStats)
        self.node_gauges: dict[tuple[str, str, str], MetricStats] = defaultdict(MetricStats)
        self.node_counters: dict[tuple[str, str, str], MetricStats] = defaultdict(MetricStats)
        self.router_series: dict[tuple[str, str], MetricStats] = defaultdict(MetricStats)
        self.raw_csv_handle: Any = None
        self.raw_csv_writer: csv.DictWriter | None = None
        self.raw_parquet_writer: Any = None
        self.queue_csv_handle: Any = None
        self.queue_csv_writer: csv.DictWriter | None = None
        self.queue_parquet_writer: Any = None
        self.raw_parquet_schema: Any = None
        self.raw_parquet_status = "disabled"
        self.queue_sample_rows = 0
        if self.cfg.enabled and self.cfg.write_raw_samples and cfg.output.write_csv:
            self.raw_csv_handle = (output_dir / "metrics_samples.csv").open(
                "w", encoding="utf-8", newline=""
            )
            self.raw_csv_writer = csv.DictWriter(
                self.raw_csv_handle, fieldnames=self.RAW_COLUMNS, extrasaction="ignore"
            )
            self.raw_csv_writer.writeheader()
            self.queue_csv_handle = (output_dir / "node_queue_samples.csv").open(
                "w", encoding="utf-8", newline=""
            )
            self.queue_csv_writer = csv.DictWriter(
                self.queue_csv_handle, fieldnames=self.RAW_COLUMNS, extrasaction="ignore"
            )
            self.queue_csv_writer.writeheader()
        if self.cfg.enabled and self.cfg.write_raw_samples and cfg.output.write_parquet:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                self.raw_parquet_schema = pa.schema(
                    [
                        ("time_s", pa.float64()),
                        ("role", pa.string()),
                        ("endpoint", pa.string()),
                        ("url", pa.string()),
                        ("metric", pa.string()),
                        ("labels", pa.string()),
                        ("value", pa.float64()),
                    ]
                )
                self.raw_parquet_writer = pq.ParquetWriter(
                    output_dir / "metrics_samples.parquet", self.raw_parquet_schema
                )
                self.queue_parquet_writer = pq.ParquetWriter(
                    output_dir / "node_queue_samples.parquet", self.raw_parquet_schema
                )
                self.raw_parquet_status = "streaming"
            except ImportError:
                self.raw_parquet_status = "pyarrow unavailable"

    def scrape(self, endpoint: MetricsEndpoint) -> tuple[list[tuple[str, str, float]], str]:
        try:
            response = self.pool.request(
                "GET",
                endpoint.url,
                timeout=urllib3.Timeout(
                    connect=self.cfg.scrape_timeout_s,
                    read=self.cfg.scrape_timeout_s,
                ),
                retries=False,
            )
            if response.status != 200:
                return [], f"HTTP {response.status}"
            return parse_prometheus(
                response.data.decode("utf-8", errors="replace"), self.include
            ), ""
        except Exception as exc:  # noqa: BLE001 - metrics loss is reported, not fatal.
            return [], f"{exc.__class__.__name__}: {exc}"

    async def collect(self, time_s: float) -> None:
        results = await asyncio.gather(
            *(asyncio.to_thread(self.scrape, endpoint) for endpoint in self.endpoints)
        )
        batch: list[dict[str, Any]] = []
        router_metrics = {
            "vllm_router_pd_prefill_requests_total",
            "vllm_router_pd_decode_requests_total",
            "vllm_router_policy_decisions_total",
            "vllm_router_processed_requests_total",
            "vllm_router_cache_hits_total",
            "vllm_router_cache_misses_total",
        }
        for endpoint, (metrics, error) in zip(self.endpoints, results, strict=True):
            if error:
                key = (endpoint.role, endpoint.name, error)
                record = self.errors.setdefault(
                    key,
                    {
                        "role": endpoint.role,
                        "name": endpoint.name,
                        "error": error,
                        "count": 0,
                        "first_time_s": time_s,
                        "last_time_s": time_s,
                    },
                )
                record["count"] += 1
                record["last_time_s"] = time_s
            rows = [
                {
                    "time_s": time_s,
                    "role": endpoint.role,
                    "endpoint": endpoint.name,
                    "url": endpoint.url,
                    "metric": metric,
                    "labels": labels,
                    "value": value,
                }
                for metric, labels, value in metrics
            ]
            batch.extend(rows)
            self.record_node_snapshot(endpoint, time_s, rows)
            for row in rows:
                self.series[(endpoint.role, endpoint.name, row["metric"], row["labels"])].add(
                    time_s, float(row["value"])
                )
                if endpoint.role == "router" and row["metric"] in router_metrics:
                    self.router_series[(row["metric"], row["labels"])].add(
                        time_s, float(row["value"])
                    )
        self.write_raw(batch)

    def record_node_snapshot(
        self,
        endpoint: MetricsEndpoint,
        time_s: float,
        rows: list[dict[str, Any]],
    ) -> None:
        if endpoint.role not in {"prefill", "decode"}:
            return
        gauges = {
            "running_requests": ("num_requests_running",),
            "waiting_requests": ("num_requests_waiting",),
            "kv_cache_usage": ("gpu_cache_usage_perc", "kv_cache_usage_perc"),
        }
        counters = {
            "prompt_tokens": ("prompt_tokens_total",),
            "generation_tokens": ("generation_tokens_total",),
        }
        for name, suffixes in gauges.items():
            values = [
                float(row["value"])
                for row in rows
                if any(row["metric"].endswith(suffix) for suffix in suffixes)
            ]
            if values:
                self.node_gauges[(endpoint.role, endpoint.name, name)].add(time_s, sum(values))
        for name, suffixes in counters.items():
            values = [
                float(row["value"])
                for row in rows
                if any(row["metric"].endswith(suffix) for suffix in suffixes)
            ]
            if values:
                self.node_counters[(endpoint.role, endpoint.name, name)].add(time_s, sum(values))

    def write_raw(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        activity_suffixes = (
            "num_requests_running",
            "num_requests_waiting",
            "gpu_cache_usage_perc",
            "kv_cache_usage_perc",
        )
        queue_rows = [
            row
            for row in rows
            if row["role"] in {"prefill", "decode"}
            and (
                "queue" in row["metric"].lower()
                or any(row["metric"].endswith(suffix) for suffix in activity_suffixes)
            )
        ]
        self.queue_sample_rows += len(queue_rows)
        if self.raw_csv_writer is not None:
            self.raw_csv_writer.writerows(rows)
            self.raw_csv_handle.flush()
        if self.queue_csv_writer is not None and queue_rows:
            self.queue_csv_writer.writerows(queue_rows)
            self.queue_csv_handle.flush()
        if self.raw_parquet_writer is not None:
            import pyarrow as pa

            self.raw_parquet_writer.write_table(
                pa.Table.from_pylist(rows, schema=self.raw_parquet_schema)
            )
            if queue_rows:
                self.queue_parquet_writer.write_table(
                    pa.Table.from_pylist(queue_rows, schema=self.raw_parquet_schema)
                )

    async def run(self, stop: asyncio.Event, phase_start: float) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.cfg.poll_interval_s)
            except TimeoutError:
                await self.collect(time.perf_counter() - phase_start)

    def close(self) -> None:
        if self.raw_csv_handle is not None:
            self.raw_csv_handle.close()
            self.raw_csv_handle = None
        if self.queue_csv_handle is not None:
            self.queue_csv_handle.close()
            self.queue_csv_handle = None
        if self.raw_parquet_writer is not None:
            self.raw_parquet_writer.close()
            self.raw_parquet_writer = None
            self.queue_parquet_writer.close()
            self.queue_parquet_writer = None
            self.raw_parquet_status = "written"

    def summary(self) -> dict[str, Any]:
        series_summary = {
            f"{role}/{endpoint}/{metric}{labels}": stats.series_summary(metric.endswith("_total"))
            for (role, endpoint, metric, labels), stats in sorted(self.series.items())
        }
        nodes = {(role, endpoint) for role, endpoint, _ in self.node_gauges | self.node_counters}
        node_activity: dict[str, Any] = {}
        for role, endpoint in sorted(nodes):
            node_activity[f"{role}/{endpoint}"] = {
                name: (
                    stats.gauge_summary()
                    if (stats := self.node_gauges.get((role, endpoint, name))) is not None
                    else None
                )
                for name in ("running_requests", "waiting_requests", "kv_cache_usage")
            }
            node_activity[f"{role}/{endpoint}"].update(
                {
                    f"{name}_delta": (
                        stats.last - (stats.first if stats.first_time_s <= 1e-9 else 0.0)
                        if (stats := self.node_counters.get((role, endpoint, name))) is not None
                        else None
                    )
                    for name in ("prompt_tokens", "generation_tokens")
                }
            )
        assignments = {
            f"{metric}{labels}": stats.last - (stats.first if stats.first_time_s <= 1e-9 else 0.0)
            for (metric, labels), stats in sorted(self.router_series.items())
        }
        cache_hits = sum(
            value
            for key, value in assignments.items()
            if key.startswith("vllm_router_cache_hits_total")
        )
        cache_misses = sum(
            value
            for key, value in assignments.items()
            if key.startswith("vllm_router_cache_misses_total")
        )
        per_policy: dict[str, float] = defaultdict(float)
        for key, value in assignments.items():
            if not key.startswith("vllm_router_policy_decisions_total") or value <= 0:
                continue
            match = re.search(r'(?:^|[, {])policy="([^"]+)"', key)
            if match:
                per_policy[match.group(1)] += value
        expected = sorted({self._outer.run.prefill_policy, self._outer.run.decode_policy})
        observed = sorted(per_policy)
        return {
            "endpoint_count": len(self.endpoints),
            "sample_rows": sum(stats.samples for stats in self.series.values()),
            "scrape_errors": list(self.errors.values()),
            "raw_parquet_status": self.raw_parquet_status,
            "node_queue_sample_rows": self.queue_sample_rows,
            "node_activity": node_activity,
            "router_assignments": assignments,
            "router_cache_proxy": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": (
                    cache_hits / (cache_hits + cache_misses) if cache_hits + cache_misses else None
                ),
                "unit": "router approximate character-tree decisions, not physical KV tokens",
            },
            "router_policies": {
                "observed_policy_labels": observed,
                "decision_count_by_policy": dict(sorted(per_policy.items())),
                "expected_policy_labels": expected,
                "expected_labels_observed": set(expected).issubset(observed) if observed else None,
            },
            "series": series_summary,
        }


class Progress:
    def __init__(self, total: int, cfg: ExecutionConfig):
        self.total = total
        self.cfg = cfg
        self.completed = self.failed = self.inflight = 0
        self.started = time.perf_counter()
        self.last_report = self.started

    def submitted(self) -> None:
        self.inflight += 1

    def finished(self, success: bool) -> None:
        self.inflight = max(0, self.inflight - 1)
        self.completed += 1
        self.failed += int(not success)
        now = time.perf_counter()
        due_count = self.cfg.progress_interval_requests > 0 and (
            self.completed % self.cfg.progress_interval_requests == 0
        )
        due_time = now - self.last_report >= self.cfg.progress_interval_s
        if not (due_count or due_time or self.completed == self.total):
            return
        elapsed = now - self.started
        rate = self.completed / elapsed if elapsed else 0.0
        eta = (self.total - self.completed) / rate if rate else 0.0
        print(
            f"progress completed={self.completed}/{self.total} "
            f"({self.completed / self.total:.1%}) failed={self.failed} "
            f"inflight={self.inflight} elapsed={elapsed:.1f}s "
            f"completion_rate={rate:.2f}/s eta={eta:.1f}s",
            flush=True,
        )
        self.last_report = now


async def sleep_until(target: float) -> None:
    while True:
        remaining = target - time.perf_counter()
        if remaining <= 0:
            return
        await asyncio.sleep(min(remaining, 0.05))


async def run_one_request(
    *,
    cfg: SyntheticBenchmarkConfig,
    request: SyntheticRequest,
    phase_start: float,
    bank: TokenBank,
    pool: urllib3.PoolManager,
    executor: ThreadPoolExecutor,
    semaphore: asyncio.Semaphore,
    progress: Progress,
    baselines: dict[str, dict[int, float]],
) -> dict[str, Any]:
    build_at = phase_start + request.scheduled_arrival_offset_s - cfg.request.prompt_build_lead_s
    await sleep_until(build_at)
    async with semaphore:
        prompt = await asyncio.to_thread(bank.prompt, request)
        body, headers = await asyncio.to_thread(request_payload, cfg, request, prompt)
        del prompt
        await sleep_until(phase_start + request.scheduled_arrival_offset_s)
        progress.submitted()
        result = await asyncio.get_running_loop().run_in_executor(
            executor,
            send_completion_sync,
            pool,
            join_url(cfg.router.base_url, cfg.router.completions_path),
            body,
            headers,
            http_timeout(cfg.execution),
        )
    actual_submission = result["request_start_perf"] - phase_start
    first_token = (
        result["first_token_perf"] - phase_start if result["first_token_perf"] is not None else None
    )
    completion = result["completion_perf"] - phase_start
    ttft = first_token - actual_submission if first_token is not None else None
    latency = completion - actual_submission
    output_tokens = result["usage_completion_tokens"] or result["stream_token_events"] or 0
    itl = (
        (completion - first_token) / (output_tokens - 1)
        if first_token is not None and output_tokens > 1
        else None
    )
    solo_ttft = interpolate_baseline(request.prompt_length, baselines["ttft_s"])
    solo_e2e = interpolate_baseline(request.prompt_length, baselines["e2e_latency_s"])
    row = {
        "request_id": request.request_id,
        "source_index": request.request_index,
        "input_tokens": request.prompt_length,
        "requested_decode_tokens": cfg.request.decode_tokens,
        "scheduled_arrival_time_s": request.scheduled_arrival_offset_s,
        "actual_submission_time_s": actual_submission,
        "client_dispatch_delay_s": actual_submission - request.scheduled_arrival_offset_s,
        "first_token_time_s": first_token,
        "completion_time_s": completion,
        "ttft_s": ttft,
        "e2e_latency_s": latency,
        "inter_token_latency_s": itl,
        "cold_solo_ttft_s": solo_ttft,
        "cold_solo_e2e_latency_s": solo_e2e,
        "cold_ttft_slowdown": ttft / solo_ttft if ttft is not None and solo_ttft else None,
        "cold_e2e_slowdown": latency / solo_e2e if solo_e2e else None,
        **{key: value for key, value in result.items() if not key.endswith("_perf")},
    }
    progress.finished(bool(row["success"]))
    return row


def request_concurrency(records: list[dict[str, Any]]) -> dict[str, float | int]:
    events = []
    for row in records:
        events.append((float(row["actual_submission_time_s"]), 1))
        events.append((float(row["completion_time_s"]), -1))
    events.sort(key=lambda item: (item[0], item[1]))
    if not events:
        return {"average_unfinished_requests": 0.0, "peak_unfinished_requests": 0}
    current = peak = 0
    area = 0.0
    last = events[0][0]
    for timestamp, delta in events:
        area += current * (timestamp - last)
        current += delta
        peak = max(peak, current)
        last = timestamp
    span = max(events[-1][0] - events[0][0], 0.0)
    return {
        "average_unfinished_requests": area / span if span else 0.0,
        "peak_unfinished_requests": peak,
    }


def summarize_requests(
    records: list[dict[str, Any]], cfg: SyntheticBenchmarkConfig
) -> dict[str, Any]:
    successes = [row for row in records if row["success"]]
    failures = [row for row in records if not row["success"]]
    first_submission = min((float(row["actual_submission_time_s"]) for row in records), default=0.0)
    last_completion = max(
        (float(row["completion_time_s"]) for row in records), default=first_submission
    )
    wall = max(0.0, last_completion - first_submission)
    output_tokens = sum(
        int(row["usage_completion_tokens"] or row["stream_token_events"] or 0) for row in successes
    )
    verified = [
        int(row["usage_completion_tokens"])
        for row in successes
        if row["usage_completion_tokens"] is not None
    ]
    mismatches = sum(value != cfg.request.decode_tokens for value in verified)
    return {
        "requests": {
            "submitted": len(records),
            "completed": len(successes),
            "failed": len(failures),
            "failure_fraction": len(failures) / len(records) if records else 0.0,
        },
        "ttft_s": distribution(row.get("ttft_s") for row in successes),
        "e2e_latency_s": distribution(row.get("e2e_latency_s") for row in successes),
        "inter_token_latency_s": distribution(
            row.get("inter_token_latency_s") for row in successes
        ),
        "cold_ttft_slowdown": distribution(row.get("cold_ttft_slowdown") for row in successes),
        "cold_e2e_slowdown": distribution(row.get("cold_e2e_slowdown") for row in successes),
        "client_dispatch_delay_s": distribution(
            row.get("client_dispatch_delay_s") for row in records
        ),
        "throughput": {
            "wall_time_s": wall,
            "completed_requests_s": len(successes) / wall if wall else None,
            "output_tokens_s": output_tokens / wall if wall else None,
            "requested_decode_tokens": cfg.request.decode_tokens,
            "observed_output_tokens": output_tokens,
            "decode_length_verified_requests": len(verified),
            "decode_length_unverified_requests": len(successes) - len(verified),
            "decode_length_mismatch_count": mismatches,
            "arrival_mode": cfg.arrival.mode,
            "target_request_rate": cfg.arrival.target_request_rate,
        },
        "client_concurrency": request_concurrency(records),
        "errors": {
            error: sum(row["error"] == error for row in failures)
            for error in sorted({row["error"] for row in failures})
        },
    }


async def benchmark_requests(
    cfg: SyntheticBenchmarkConfig,
    requests: list[SyntheticRequest],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preflight_result = await preflight(cfg)
    baseline_path = Path(cfg.calibration.baseline_path) if cfg.calibration.baseline_path else None
    if baseline_path is not None and not baseline_path.is_absolute():
        baseline_path = Path(get_original_cwd()) / baseline_path
    baselines = load_baselines(baseline_path)
    bank = TokenBank(cfg.request)
    pool = urllib3.PoolManager(
        num_pools=cfg.execution.max_connections,
        maxsize=cfg.execution.max_connections,
        block=True,
    )
    executor = ThreadPoolExecutor(
        max_workers=cfg.execution.max_connections,
        thread_name_prefix="router-synthetic-http",
    )
    semaphore = asyncio.Semaphore(cfg.execution.max_connections)
    progress = Progress(len(requests), cfg.execution)
    collector = MetricsCollector(cfg, output_dir)
    phase_start = time.perf_counter() + cfg.request.prompt_build_lead_s
    stop_metrics = asyncio.Event()
    metrics_task = None
    if cfg.observability.enabled:
        await collector.collect(time.perf_counter() - phase_start)
        metrics_task = asyncio.create_task(collector.run(stop_metrics, phase_start))
    try:
        records = await asyncio.gather(
            *(
                run_one_request(
                    cfg=cfg,
                    request=request,
                    phase_start=phase_start,
                    bank=bank,
                    pool=pool,
                    executor=executor,
                    semaphore=semaphore,
                    progress=progress,
                    baselines=baselines,
                )
                for request in requests
            )
        )
    finally:
        stop_metrics.set()
        try:
            if metrics_task is not None:
                await metrics_task
                await collector.collect(time.perf_counter() - phase_start)
        finally:
            collector.close()
            executor.shutdown(wait=True)
    summary = summarize_requests(records, cfg)
    summary["observability"] = (
        collector.summary() if cfg.observability.enabled else {"enabled": False}
    )
    summary["preflight"] = preflight_result
    summary["limitations"] = {
        "physical_cache_hit_tokens": "not exposed per request by the unmodified router/vLLM API",
        "cache_aware_units": (
            "the PD router scores serialized JSON characters; shared token-ID prefixes remain "
            "shared prefixes, but match length is not measured in KV tokens"
        ),
        "dummy_weights": "deployment property, never configured by this client",
    }
    failure_fraction = summary["requests"]["failure_fraction"]
    mismatches = summary["throughput"]["decode_length_mismatch_count"]
    if failure_fraction > cfg.execution.max_failure_fraction:
        summary["invalid"] = True
        summary["invalid_reason"] = (
            f"failure fraction {failure_fraction:.3%} exceeds configured maximum "
            f"{cfg.execution.max_failure_fraction:.3%}"
        )
    elif mismatches:
        summary["invalid"] = True
        summary["invalid_reason"] = (
            f"{mismatches} successful responses did not contain exactly "
            f"request.decode_tokens={cfg.request.decode_tokens}"
        )
    else:
        summary["invalid"] = False
    return records, summary


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no rows"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return "pyarrow unavailable"
    pq.write_table(pa.Table.from_pylist(rows), path)
    return "written"


async def calibrate_router(
    cfg: SyntheticBenchmarkConfig, output_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    preflight_result = await preflight(cfg)
    bank = TokenBank(cfg.request)
    pool = urllib3.PoolManager(num_pools=1, maxsize=1, block=True)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="router-calibration")
    rows: list[dict[str, Any]] = []
    try:
        for length in cfg.calibration.lengths:
            total = cfg.calibration.warmup_repetitions + cfg.calibration.measured_repetitions
            for repetition in range(total):
                phase = "warmup" if repetition < cfg.calibration.warmup_repetitions else "measured"
                request = SyntheticRequest(
                    seed=-1,
                    request_index=repetition,
                    request_id=f"calibration-{length}-{phase}-{repetition}",
                    prompt_length=length,
                    token_variant=stable_u64("calibration", length, repetition),
                    scheduled_arrival_offset_s=0.0,
                )
                prompt = await asyncio.to_thread(bank.prompt, request)
                body, headers = request_payload(cfg, request, prompt)
                result = await asyncio.get_running_loop().run_in_executor(
                    executor,
                    send_completion_sync,
                    pool,
                    join_url(cfg.router.base_url, cfg.router.completions_path),
                    body,
                    headers,
                    http_timeout(cfg.execution),
                )
                ttft = (
                    result["first_token_perf"] - result["request_start_perf"]
                    if result["first_token_perf"] is not None
                    else None
                )
                e2e = result["completion_perf"] - result["request_start_perf"]
                row = {
                    "length": length,
                    "phase": phase,
                    "repetition": repetition,
                    "ttft_s": ttft,
                    "e2e_latency_s": e2e,
                    **{key: value for key, value in result.items() if not key.endswith("_perf")},
                }
                rows.append(row)
                if not result["success"]:
                    raise RuntimeError(f"calibration request failed: {result['error']}")
    finally:
        executor.shutdown(wait=True)
    measured = [row for row in rows if row["phase"] == "measured"]
    baselines = {
        "ttft_s": {
            str(length): statistics.median(
                row["ttft_s"] for row in measured if row["length"] == length
            )
            for length in cfg.calibration.lengths
        },
        "e2e_latency_s": {
            str(length): statistics.median(
                row["e2e_latency_s"] for row in measured if row["length"] == length
            )
            for length in cfg.calibration.lengths
        },
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "prefill_policy": cfg.run.prefill_policy,
            "decode_policy": cfg.run.decode_policy,
            "decode_tokens": cfg.request.decode_tokens,
            "preflight": preflight_result,
            "warning": "Restart the deployment before a cache-sensitive benchmark run.",
        },
    }
    (output_dir / "solo_baselines.json").write_text(
        json.dumps(baselines, indent=2) + "\n", encoding="utf-8"
    )
    return rows, baselines


def output_root(cfg: SyntheticBenchmarkConfig, original_cwd: Path) -> Path:
    if cfg.output.directory:
        path = Path(cfg.output.directory)
        return path if path.is_absolute() else original_cwd / path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        original_cwd
        / "benchmark_output"
        / cfg.run.name
        / (
            f"prefill-{cfg.run.prefill_policy}_decode-{cfg.run.decode_policy}"
            f"_scheduler-{cfg.run.local_scheduler}"
        )
        / cfg.arrival.mode
        / f"rate-{cfg.arrival.target_request_rate:g}-{timestamp}"
    )


def prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is nonempty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def workload_manifest(
    seed: int, requests: list[SyntheticRequest], cfg: SyntheticBenchmarkConfig
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for request in requests:
        digest.update(
            f"{request.request_id}\0{request.prompt_length}\0{request.token_variant}\0"
            f"{request.scheduled_arrival_offset_s:.17g}\n".encode()
        )
    gaps = [
        right.scheduled_arrival_offset_s - left.scheduled_arrival_offset_s
        for left, right in zip(requests, requests[1:])
    ]
    counts: dict[int, int] = {}
    for request in requests:
        counts[request.prompt_length] = counts.get(request.prompt_length, 0) + 1
    return {
        "seed": seed,
        "request_count": len(requests),
        "sha256": digest.hexdigest(),
        "arrival_mode": cfg.arrival.mode,
        "arrival_span_s": (
            requests[-1].scheduled_arrival_offset_s - requests[0].scheduled_arrival_offset_s
            if len(requests) > 1
            else 0.0
        ),
        "zero_gap_fraction": sum(gap == 0 for gap in gaps) / len(gaps) if gaps else 0.0,
        "prompt_length_counts": {str(key): value for key, value in sorted(counts.items())},
        "prompt_variants_per_length": cfg.workload.prompt_variants_per_length,
        "fixed_decode_tokens": cfg.request.decode_tokens,
    }


def annotate_records(
    rows: list[dict[str, Any]],
    requests: list[SyntheticRequest],
    cfg: SyntheticBenchmarkConfig,
) -> list[dict[str, Any]]:
    metadata = {request.request_id: request for request in requests}
    for row in rows:
        request = metadata[row["request_id"]]
        row["seed"] = request.seed
        row["token_variant"] = request.token_variant
        row["prompt_length"] = request.prompt_length
        row["prefill_policy"] = cfg.run.prefill_policy
        row["decode_policy"] = cfg.run.decode_policy
        row["local_scheduler"] = cfg.run.local_scheduler
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def capture_queue_stats(
    source_dir: Path | None,
    output_dir: Path,
    start_time_ns: int | None = None,
    end_time_ns: int | None = None,
) -> dict[str, Any]:
    """Best-effort copy of worker queue traces into the client result."""
    if source_dir is None:
        return {"captured": False, "reason": "output.queue_stats_source_dir is unset"}
    try:
        source_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {
            "captured": False,
            "source_dir": str(source_dir),
            "reason": f"cannot create or access source directory: {exc}",
        }

    destination_dir = output_dir / "queue_stats"
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {
            "captured": False,
            "source_dir": str(source_dir),
            "reason": f"cannot create destination directory: {exc}",
        }

    copied = []
    errors = []
    for source_path in sorted(source_dir.glob("vllm_ascend_queue_stats_*.csv")):
        if not source_path.is_file():
            continue
        destination_path = destination_dir / source_path.name
        try:
            with source_path.open(newline="", encoding="utf-8") as source_file:
                reader = csv.DictReader(source_file)
                rows = [
                    row
                    for row in reader
                    if start_time_ns is None
                    or end_time_ns is None
                    or start_time_ns <= int(row["timestamp_ns"]) <= end_time_ns
                ]
                fieldnames = reader.fieldnames
            if not rows or fieldnames is None:
                continue
            with destination_path.open("w", newline="", encoding="utf-8") as destination_file:
                writer = csv.DictWriter(destination_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            copied.append(destination_path.name)
        except (KeyError, OSError, ValueError) as exc:
            errors.append(f"{source_path.name}: {exc}")
    result = {
        "captured": bool(copied),
        "source_dir": str(source_dir),
        "files": copied,
        "errors": errors,
        "window_start_ns": start_time_ns,
        "window_end_ns": end_time_ns,
    }
    if not copied:
        result["reason"] = "; ".join(errors) or "no queue trace CSV files found"
    return result


async def run_seed(
    cfg: SyntheticBenchmarkConfig,
    seed: int,
    root: Path,
    queue_stats_source_dir: Path | None,
) -> dict[str, Any]:
    requests = make_workload(seed, cfg)
    manifest = workload_manifest(seed, requests, cfg)
    run_dir = root / f"seed-{seed}"
    prepare_directory(run_dir, cfg.output.overwrite)
    (run_dir / "workload_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    with (run_dir / "workload_trace.jsonl").open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request.__dict__) + "\n")
    queue_stats_start_ns = time.time_ns()
    rows, summary = await benchmark_requests(cfg, requests, run_dir)
    queue_stats_end_ns = time.time_ns()
    annotate_records(rows, requests, cfg)
    summary["run"] = {
        "prefill_policy_label": cfg.run.prefill_policy,
        "decode_policy_label": cfg.run.decode_policy,
        "local_scheduler_label": cfg.run.local_scheduler,
        "seed": seed,
        "router_url": cfg.router.base_url,
        "workload": manifest,
    }
    summary["observability"]["queue_measurement"] = {
        "poll_interval_s": cfg.observability.poll_interval_s,
        "queue_lengths": "sampled gauges; mean, maximum, and active fraction are in node_activity",
        "queue_latency": (
            "any worker request_queue_time histogram series are retained in metrics_samples "
            "and node_queue_samples"
        ),
        "short_decode_note": (
            "decode activity shorter than the polling interval may be absent from queue gauges; "
            "router PD decode counters remain exact"
        ),
    }
    if cfg.run.require_worker_metrics:
        node_activity = summary["observability"].get("node_activity", {})
        observed_roles = {
            key.split("/", 1)[0]
            for key, activity in node_activity.items()
            if activity.get("running_requests") is not None
            and activity.get("waiting_requests") is not None
        }
        missing_roles = {"prefill", "decode"} - observed_roles
        if missing_roles:
            summary["invalid"] = True
            summary["invalid_reason"] = (
                "no worker queue metrics were observed for roles: "
                + ", ".join(sorted(missing_roles))
            )
    # Persist client results before the optional trace snapshot. Diagnostic
    # queue statistics must never discard a completed benchmark.
    if cfg.output.write_csv:
        write_rows(run_dir / "requests.csv", rows)
    if cfg.output.write_parquet:
        write_parquet(run_dir / "requests.parquet", rows)
    summary["queue_stats"] = capture_queue_stats(
        queue_stats_source_dir,
        run_dir,
        queue_stats_start_ns,
        queue_stats_end_ns,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if queue_stats_source_dir is not None and not summary["queue_stats"]["captured"]:
        print(
            "Warning: scheduler queue traces were not captured: "
            f"{summary['queue_stats']['reason']}"
        )
    return summary


def dry_run(cfg: SyntheticBenchmarkConfig, root: Path) -> None:
    print(OmegaConf.to_yaml(OmegaConf.structured(cfg), resolve=True))
    print("Workloads:")
    for seed in cfg.run.seeds:
        requests = make_workload(seed, cfg)
        print(json.dumps(workload_manifest(seed, requests, cfg), indent=2))
        print("First five requests:")
        print(json.dumps([request.__dict__ for request in requests[:5]], indent=2))
    print(f"Output directory: {root}")
    print("No traffic was sent.")


async def calibrate(cfg: SyntheticBenchmarkConfig, root: Path) -> None:
    prepare_directory(root, cfg.output.overwrite)
    rows, baselines = await calibrate_router(cfg, root)
    if cfg.output.write_csv:
        write_rows(root / "calibration_requests.csv", rows)
    if cfg.output.write_parquet:
        write_parquet(root / "calibration_requests.parquet", rows)
    print(json.dumps(baselines, indent=2))
    print(f"Calibration output: {root}")
    print("Restart all workers before a cache-sensitive benchmark run.")


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[float]:
        result = []
        for summary in summaries:
            value: Any = summary
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                result.append(float(value))
        return result

    fields = {
        "mean_ttft_s": ("ttft_s", "mean"),
        "p95_ttft_s": ("ttft_s", "p95"),
        "p99_ttft_s": ("ttft_s", "p99"),
        "mean_e2e_latency_s": ("e2e_latency_s", "mean"),
        "p95_e2e_latency_s": ("e2e_latency_s", "p95"),
        "mean_cold_ttft_slowdown": ("cold_ttft_slowdown", "mean"),
        "p95_cold_ttft_slowdown": ("cold_ttft_slowdown", "p95"),
        "completion_throughput": ("throughput", "completed_requests_s"),
    }
    return {
        name: {
            "mean_across_seeds": statistics.fmean(numbers) if numbers else None,
            "per_seed": numbers,
        }
        for name, path in fields.items()
        if (numbers := values(path))
    }


@hydra.main(version_base=None, config_name="preflow_router_synthetic")
def main(config: DictConfig) -> None:
    values = OmegaConf.to_object(config)
    if not isinstance(values, SyntheticBenchmarkConfig):
        raise TypeError("Hydra did not construct SyntheticBenchmarkConfig")
    cfg = values
    validate_config(cfg)
    root = output_root(cfg, Path(get_original_cwd()))
    if cfg.run.mode == "dry_run":
        dry_run(cfg, root)
        return
    if cfg.run.mode == "calibrate":
        asyncio.run(calibrate(cfg, root))
        return
    prepare_directory(root, cfg.output.overwrite)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(OmegaConf.to_container(config, resolve=True), sort_keys=False),
        encoding="utf-8",
    )
    queue_stats_source_dir = None
    if cfg.output.queue_stats_source_dir:
        queue_stats_source_dir = Path(cfg.output.queue_stats_source_dir)
        if not queue_stats_source_dir.is_absolute():
            queue_stats_source_dir = Path(get_original_cwd()) / queue_stats_source_dir
    summaries = [
        asyncio.run(run_seed(cfg, seed, root, queue_stats_source_dir))
        for seed in cfg.run.seeds
    ]
    aggregate = {
        "run": {
            "prefill_policy_label": cfg.run.prefill_policy,
            "decode_policy_label": cfg.run.decode_policy,
            "local_scheduler_label": cfg.run.local_scheduler,
            "router_url": cfg.router.base_url,
            "seeds": cfg.run.seeds,
        },
        "aggregate": aggregate_summaries(summaries),
        "seed_summaries": summaries,
    }
    (root / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run": aggregate["run"], "aggregate": aggregate["aggregate"]}, indent=2))
    print(f"Benchmark output: {root}")
    invalid = [summary.get("invalid_reason") for summary in summaries if summary.get("invalid")]
    if invalid:
        raise SystemExit("; ".join(str(reason) for reason in invalid))


if __name__ == "__main__":
    main()
