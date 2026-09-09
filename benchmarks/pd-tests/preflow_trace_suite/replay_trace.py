#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay one timestamped prefill trace against one standalone vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import urllib3


@dataclass(frozen=True)
class TraceRequest:
    trace_name: str
    source_index: int
    original_timestamp: float
    prompt_tokens: int
    predicted_service_s: float
    scheduled_offset_s: float

    @property
    def request_id(self) -> str:
        return f"{self.trace_name}-{self.source_index:08d}"


@dataclass(frozen=True)
class CalibratedChunkCost:
    quadratic_c: float
    history_c: float
    linear_c: float
    constant: float
    overhead_floor_s: float
    max_chunk_tokens: int
    max_history_tokens: int
    max_model_len: int

    def chunk_duration_s(self, history_tokens: int, new_tokens: int) -> float:
        if new_tokens <= 0 or new_tokens > self.max_chunk_tokens:
            raise ValueError(f"invalid calibrated chunk size {new_tokens}")
        chunk = float(new_tokens)
        history = float(history_tokens)
        duration = (
            self.quadratic_c * chunk * (chunk + 1.0) / 2.0
            + self.history_c * chunk * history
            + self.linear_c * chunk
            + self.constant
        )
        duration = max(duration, self.overhead_floor_s)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("calibrated chunk model produced an invalid duration")
        return duration

    def request_duration_s(self, prompt_tokens: int, chunk_size: int) -> float:
        history = 0
        duration = 0.0
        while history < prompt_tokens:
            chunk = min(chunk_size, prompt_tokens - history)
            duration += self.chunk_duration_s(history, chunk)
            history += chunk
        return duration


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numbers:
        return {key: None for key in ("count", "mean", "median", "p90", "p95", "p99", "max")}
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "p90": percentile(numbers, 0.90),
        "p95": percentile(numbers, 0.95),
        "p99": percentile(numbers, 0.99),
        "max": max(numbers),
    }


def load_calibrated_chunk_cost(path: Path) -> tuple[CalibratedChunkCost, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "preflow_parametric_chunk_cost_model":
        raise ValueError(f"{path}: unexpected calibration schema")
    coefficients = payload["coefficients"]
    domain = payload["domain"]
    cost = CalibratedChunkCost(
        quadratic_c=float(coefficients["quadratic_c"]),
        history_c=float(coefficients["history_c"]),
        linear_c=float(coefficients["linear_c"]),
        constant=float(coefficients["constant"]),
        overhead_floor_s=float(payload.get("overhead_floor_s", 0.0)),
        max_chunk_tokens=int(domain["max_chunk_tokens"]),
        max_history_tokens=int(domain["max_history_tokens"]),
        max_model_len=int(domain["max_model_len"]),
    )
    return cost, {
        "path": str(path),
        "sha256": sha256(path),
        "schema": payload["schema"],
        "version": payload.get("version"),
        "timing_provenance": payload.get("timing_provenance"),
        "paper_eligible": payload.get("paper_eligible"),
        "partition_additive": payload.get("partition_additive"),
        "coefficients": coefficients,
        "domain": domain,
        "overhead_floor_s": payload.get("overhead_floor_s"),
        "extrapolation_enabled": True,
    }


def load_workload(
    trace_path: Path,
    trace_name: str,
    target_load: float,
    service_budget_s: float,
    minimum_requests: int,
    maximum_requests: int,
    max_model_len: int,
    chunk_size: int,
    cost_model: CalibratedChunkCost,
) -> tuple[list[TraceRequest], dict[str, Any]]:
    selected: list[tuple[int, float, int, float]] = []
    total_service = 0.0
    last_timestamp: float | None = None
    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
                source_index = int(row["source_index"])
                timestamp = float(row["timestamp"])
                prompt_tokens = int(row["input_length"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{trace_path}:{line_number}: invalid trace row") from error
            if prompt_tokens <= 0 or prompt_tokens + 1 > max_model_len:
                raise ValueError(
                    f"{trace_path}:{line_number}: prompt length {prompt_tokens} "
                    f"does not fit max_model_len={max_model_len} with one output token"
                )
            if last_timestamp is not None and timestamp < last_timestamp:
                raise ValueError(f"{trace_path}:{line_number}: timestamps are not nondecreasing")
            service = cost_model.request_duration_s(prompt_tokens, chunk_size)
            selected.append((source_index, timestamp, prompt_tokens, service))
            total_service += service
            last_timestamp = timestamp
            if len(selected) >= minimum_requests and total_service >= service_budget_s:
                break
            if len(selected) >= maximum_requests:
                break
    if len(selected) < 2:
        raise ValueError(f"{trace_path}: workload window has fewer than two requests")
    first_timestamp = selected[0][1]
    raw_span = selected[-1][1] - first_timestamp
    if raw_span <= 0:
        raise ValueError(f"{trace_path}: selected workload has no positive arrival span")
    target_span = total_service / target_load
    time_scale = target_span / raw_span
    requests = [
        TraceRequest(
            trace_name=trace_name,
            source_index=source_index,
            original_timestamp=timestamp,
            prompt_tokens=prompt_tokens,
            predicted_service_s=service,
            scheduled_offset_s=(timestamp - first_timestamp) * time_scale,
        )
        for source_index, timestamp, prompt_tokens, service in selected
    ]
    return requests, {
        "selection": (
            "contiguous prefix until both minimum_requests and service_budget_s are met, capped by maximum_requests"
        ),
        "service_budget_reached": total_service >= service_budget_s,
        "maximum_requests": maximum_requests,
        "selected_requests": len(requests),
        "source_index_first": requests[0].source_index,
        "source_index_last": requests[-1].source_index,
        "original_timestamp_first": first_timestamp,
        "original_timestamp_last": selected[-1][1],
        "original_span_s": raw_span,
        "arrival_time_scale": time_scale,
        "scheduled_span_s": target_span,
        "predicted_service_s": total_service,
        "target_load": target_load,
        "realized_offered_load_by_construction": total_service / target_span,
        "requests_exceeding_calibrated_history_domain": sum(
            request.prompt_tokens > cost_model.max_history_tokens for request in requests
        ),
        "prompt_tokens": distribution(float(request.prompt_tokens) for request in requests),
    }


def deterministic_prompt(request: TraceRequest) -> list[int]:
    # Prefix caching is disabled. The affine pattern gives each request a
    # deterministic valid-token prompt without requiring a tokenizer.
    offset = (request.source_index * 104_729) % 29_000
    return [1_000 + ((offset + position * 15_485_863) % 29_000) for position in range(request.prompt_tokens)]


def read_sse_event(line: bytes) -> dict[str, Any] | None:
    stripped = line.strip()
    if not stripped.startswith(b"data:"):
        return None
    payload = stripped[5:].strip()
    if not payload or payload == b"[DONE]":
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def send_request(
    pool: urllib3.PoolManager,
    url: str,
    model: str,
    request: TraceRequest,
    benchmark_start: float,
    timeout: urllib3.Timeout,
) -> dict[str, Any]:
    worker_start = time.perf_counter()
    response = None
    first_token: float | None = None
    completed: float | None = None
    usage: dict[str, Any] = {}
    status = 0
    error = ""
    try:
        body = json.dumps(
            {
                "model": model,
                "prompt": deterministic_prompt(request),
                "max_tokens": 1,
                "min_tokens": 1,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
                "seed": request.source_index % (2**31 - 1),
            },
            separators=(",", ":"),
        ).encode()
        request_start = time.perf_counter()
        response = pool.request(
            "POST",
            url,
            body=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "X-Request-ID": request.request_id,
            },
            timeout=timeout,
            retries=False,
            preload_content=False,
        )
        status = int(response.status)
        if not 200 <= status < 300:
            message = response.read(16_384).decode(errors="replace")
            raise RuntimeError(f"HTTP {status}: {message}")
        buffer = b""
        for chunk in response.stream(amt=65_536, decode_content=True):
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                event = read_sse_event(line)
                if event is None:
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if choices and first_token is None:
                    first_token = time.perf_counter()
        completed = time.perf_counter()
        if first_token is None:
            raise RuntimeError("stream ended without an observable completion event")
    except Exception as exception:  # noqa: BLE001 - persisted per request.
        request_start = locals().get("request_start", time.perf_counter())
        completed = time.perf_counter()
        error = f"{exception.__class__.__name__}: {exception}"
    finally:
        if response is not None:
            response.release_conn()
    scheduled_absolute = benchmark_start + request.scheduled_offset_s
    return {
        **asdict(request),
        "request_id": request.request_id,
        "worker_start_offset_s": worker_start - benchmark_start,
        "request_start_offset_s": request_start - benchmark_start,
        "first_token_offset_s": None if first_token is None else first_token - benchmark_start,
        "completion_offset_s": None if completed is None else completed - benchmark_start,
        "client_dispatch_lag_s": request_start - scheduled_absolute,
        "prompt_serialization_s": request_start - worker_start,
        "ttft_s": None if first_token is None else first_token - request_start,
        "e2e_s": None if completed is None else completed - request_start,
        "http_status": status,
        "success": not error,
        "error": error,
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
    }


METRIC_PATTERN = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|Inf|-Inf)(?:\s+\d+)?$"
)


def scrape_metrics(pool: urllib3.PoolManager, url: str) -> list[tuple[str, str, float]]:
    response = pool.request(
        "GET",
        url,
        timeout=urllib3.Timeout(connect=2.0, read=5.0),
        retries=False,
    )
    if response.status != 200:
        return []
    rows = []
    for line in response.data.decode(errors="replace").splitlines():
        match = METRIC_PATTERN.match(line.strip())
        if match is None:
            continue
        name = match.group(1)
        if not any(
            marker in name
            for marker in (
                "num_requests_running",
                "num_requests_waiting",
                "gpu_cache_usage",
                "kv_cache_usage",
                "prompt_tokens_total",
            )
        ):
            continue
        value = float(match.group(3))
        if math.isfinite(value):
            rows.append((name, match.group(2) or "", value))
    return rows


async def collect_metrics(
    pool: urllib3.PoolManager,
    url: str,
    benchmark_start: float,
    stop: asyncio.Event,
    output_path: Path,
) -> dict[str, Any]:
    samples = 0
    errors = 0
    maxima: dict[str, float] = {}
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "metric", "labels", "value"))
        while not stop.is_set():
            try:
                rows = await asyncio.to_thread(scrape_metrics, pool, url)
                sample_time = time.perf_counter() - benchmark_start
                for name, labels, value in rows:
                    writer.writerow((sample_time, name, labels, value))
                    maxima[name] = max(maxima.get(name, -math.inf), value)
                handle.flush()
                samples += 1
            except Exception:  # noqa: BLE001 - metrics loss must not stop traffic.
                errors += 1
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=0.1)
    return {"samples": samples, "errors": errors, "maxima": maxima}


def warmup(pool: urllib3.PoolManager, url: str, model: str) -> None:
    request = TraceRequest("warmup", 0, 0.0, 2_048, 0.0, 0.0)
    result = send_request(
        pool,
        url,
        model,
        request,
        time.perf_counter(),
        urllib3.Timeout(connect=30.0, read=900.0),
    )
    if not result["success"]:
        raise RuntimeError(f"warmup failed: {result['error']}")


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    cost_model, calibration = load_calibrated_chunk_cost(args.calibration)
    if args.chunk_size > cost_model.max_chunk_tokens:
        raise ValueError(f"chunk_size={args.chunk_size} exceeds calibrated maximum {cost_model.max_chunk_tokens}")
    requests, workload = load_workload(
        trace_path=args.trace,
        trace_name=args.trace_name,
        target_load=args.target_load,
        service_budget_s=args.service_budget_s,
        minimum_requests=args.minimum_requests,
        maximum_requests=args.maximum_requests,
        max_model_len=args.max_model_len,
        chunk_size=args.chunk_size,
        cost_model=cost_model,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        args.output_dir / "workload.json",
        {
            "trace_path": str(args.trace),
            "trace_sha256": sha256(args.trace),
            "calibration": calibration,
            **workload,
        },
    )

    request_pool = urllib3.PoolManager(
        num_pools=args.max_connections,
        maxsize=args.max_connections,
        block=True,
    )
    metrics_pool = urllib3.PoolManager(num_pools=1, maxsize=1, block=True)
    completion_url = urljoin(args.base_url.rstrip("/") + "/", "v1/completions")
    metrics_url = urljoin(args.base_url.rstrip("/") + "/", "metrics")
    await asyncio.to_thread(warmup, request_pool, completion_url, args.model)

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=args.max_connections)
    semaphore = asyncio.Semaphore(args.max_connections)
    stop_metrics = asyncio.Event()
    benchmark_start = time.perf_counter() + 1.0
    metrics_task = asyncio.create_task(
        collect_metrics(
            metrics_pool,
            metrics_url,
            benchmark_start,
            stop_metrics,
            args.output_dir / "metrics.csv",
        )
    )
    timeout = urllib3.Timeout(connect=30.0, read=args.request_timeout_s)

    async def issue(request: TraceRequest) -> dict[str, Any]:
        delay = benchmark_start + request.scheduled_offset_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        async with semaphore:
            return await loop.run_in_executor(
                executor,
                send_request,
                request_pool,
                completion_url,
                args.model,
                request,
                benchmark_start,
                timeout,
            )

    try:
        results = await asyncio.gather(*(issue(request) for request in requests))
    finally:
        stop_metrics.set()
        metrics_summary = await metrics_task
        executor.shutdown(wait=True, cancel_futures=False)

    results.sort(key=lambda row: int(row["source_index"]))
    result_path = args.output_dir / "requests.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    successes = [row for row in results if row["success"]]
    completion_tokens = [
        int(row["usage_completion_tokens"]) for row in successes if row["usage_completion_tokens"] is not None
    ]
    summary = {
        "schema_version": 1,
        "trace": args.trace_name,
        "target_load": args.target_load,
        "output_tokens_requested": 1,
        "requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(results) - len(successes),
        "all_reported_completion_counts_are_one": bool(completion_tokens)
        and all(count == 1 for count in completion_tokens),
        "wall_time_s": time.perf_counter() - benchmark_start,
        "workload": workload,
        "calibration": calibration,
        "ttft_s": distribution(row["ttft_s"] for row in successes),
        "e2e_s": distribution(row["e2e_s"] for row in successes),
        "client_dispatch_lag_s": distribution(row["client_dispatch_lag_s"] for row in results),
        "prompt_serialization_s": distribution(row["prompt_serialization_s"] for row in results),
        "metrics": metrics_summary,
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    if len(successes) != len(results):
        raise RuntimeError(f"{len(results) - len(successes)} requests failed")
    if completion_tokens and not summary["all_reported_completion_counts_are_one"]:
        raise RuntimeError("server reported a completion-token count other than one")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-name", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-load", type=float, default=0.95)
    parser.add_argument("--service-budget-s", type=float, default=600.0)
    parser.add_argument("--minimum-requests", type=int, default=1_000)
    parser.add_argument("--maximum-requests", type=int, default=20_000)
    parser.add_argument("--max-model-len", type=int, default=262_144)
    parser.add_argument("--chunk-size", type=int, default=2_048)
    parser.add_argument("--max-connections", type=int, default=128)
    parser.add_argument("--request-timeout-s", type=float, default=1_800.0)
    args = parser.parse_args()
    if not 0 < args.target_load < 1:
        parser.error("--target-load must be strictly between zero and one")
    if args.service_budget_s <= 0 or args.minimum_requests <= 0:
        parser.error("service budget and minimum requests must be positive")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.maximum_requests < args.minimum_requests:
        parser.error("--maximum-requests must be at least --minimum-requests")
    return args


def main() -> int:
    args = parse_args()
    summary = asyncio.run(run_benchmark(args))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
