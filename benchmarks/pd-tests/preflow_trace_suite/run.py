#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the configured prefill-scheduler trace suite on three TP4 servers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import urllib3

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
TRACE_DIR = HERE / "traces"
REPLAY_SCRIPT = HERE / "replay_trace.py"
ANALYZE_SCRIPT = HERE / "analyze_results.py"
CALIBRATION = HERE / "calibration" / "parametric_chunk_cost_model.json"
SUITE_CONFIG = HERE / "suite_config.json"

MODEL = "/data/weights/Qwen3-Coder-30B-A3B-Instruct/"
SERVED_MODEL_NAME = "qwen"
TARGET_LOAD = 0.95
TENSOR_PARALLEL_SIZE = 4
MAX_MODEL_LEN = 262_144
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 2_048
MAX_NUM_BATCHED_SEQS = 1
SERVICE_BUDGET_S = 600.0
MINIMUM_REQUESTS = 1_000
MAXIMUM_REQUESTS = 20_000
GPU_MEMORY_UTILIZATION = 0.80
SERVER_SEED = 1_024
OUTPUT_TOKENS = 1
ASYNC_SCHEDULING = True
STARTUP_TIMEOUT_S = 1_200.0
SHUTDOWN_TIMEOUT_S = 60.0
BASE_PORT = 18_000
STARTUP_LOG_TAIL_BYTES = 128 * 1024
FATAL_SERVER_MARKERS = (
    "EngineCore failed to start.",
    "EngineCore encountered a fatal error.",
    "EngineDeadError: EngineCore encountered an issue.",
    "Engine core initialization failed",
)


class SuiteError(RuntimeError):
    pass


@dataclass(frozen=True)
class PolicySpec:
    name: str
    scheduler_cls: str
    traces: tuple[str, ...]
    scheduler_config: dict[str, dict[str, Any]]


def load_suite_definition() -> tuple[tuple[str, ...], tuple[str, ...], tuple[PolicySpec, ...]]:
    try:
        payload = json.loads(SUITE_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SuiteError(f"cannot read suite configuration {SUITE_CONFIG}: {error}") from error
    if not isinstance(payload, dict):
        raise SuiteError(f"{SUITE_CONFIG} must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise SuiteError("suite_config.json schema_version must be 1")

    name_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    raw_traces = payload.get("traces")
    if not isinstance(raw_traces, list) or not raw_traces:
        raise SuiteError("suite_config.json 'traces' must be a non-empty list")
    if not all(isinstance(trace, str) and name_pattern.fullmatch(trace) for trace in raw_traces):
        raise SuiteError("trace names may contain only letters, digits, '.', '_', and '-'")
    if len(set(raw_traces)) != len(raw_traces):
        raise SuiteError("suite_config.json contains duplicate trace names")
    traces = tuple(raw_traces)

    raw_npu_groups = payload.get("npu_groups")
    if not isinstance(raw_npu_groups, list) or not raw_npu_groups:
        raise SuiteError("suite_config.json 'npu_groups' must be a non-empty list")
    if not all(isinstance(group, str) and group for group in raw_npu_groups):
        raise SuiteError("every NPU group must be a non-empty string")
    npu_groups = tuple(raw_npu_groups)
    visible_devices: set[str] = set()
    for group in npu_groups:
        devices = group.split(",")
        if len(devices) != TENSOR_PARALLEL_SIZE or not all(device.isdigit() for device in devices):
            raise SuiteError(f"NPU group {group!r} must contain exactly {TENSOR_PARALLEL_SIZE} numeric device IDs")
        if len(set(devices)) != len(devices) or visible_devices.intersection(devices):
            raise SuiteError(f"NPU group {group!r} contains or reuses a device ID")
        visible_devices.update(devices)

    raw_policies = payload.get("policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise SuiteError("suite_config.json 'policies' must be a non-empty list")
    policies = []
    policy_names: set[str] = set()
    for index, raw_policy in enumerate(raw_policies):
        if not isinstance(raw_policy, dict):
            raise SuiteError(f"policy {index} must be a JSON object")
        name = raw_policy.get("name")
        scheduler_cls = raw_policy.get("scheduler_cls")
        if not isinstance(name, str) or not name_pattern.fullmatch(name):
            raise SuiteError(f"policy {index} has an invalid name")
        if name in policy_names:
            raise SuiteError(f"duplicate policy name: {name}")
        if not isinstance(scheduler_cls, str) or not scheduler_cls:
            raise SuiteError(f"policy {name} has an invalid scheduler_cls")

        raw_policy_traces = raw_policy.get("traces", "all")
        if raw_policy_traces == "all":
            policy_traces = traces
        elif isinstance(raw_policy_traces, list) and raw_policy_traces:
            if not all(isinstance(trace, str) and trace in traces for trace in raw_policy_traces):
                raise SuiteError(f"policy {name} references an unknown trace")
            if len(set(raw_policy_traces)) != len(raw_policy_traces):
                raise SuiteError(f"policy {name} contains duplicate traces")
            policy_traces = tuple(raw_policy_traces)
        else:
            raise SuiteError(f"policy {name} 'traces' must be 'all' or a non-empty list")

        raw_scheduler_config = raw_policy.get("scheduler_config", {})
        if not isinstance(raw_scheduler_config, dict) or not all(
            isinstance(section, str) and isinstance(values, dict) for section, values in raw_scheduler_config.items()
        ):
            raise SuiteError(f"policy {name} scheduler_config must map section names to objects")
        policies.append(
            PolicySpec(
                name=name,
                scheduler_cls=scheduler_cls,
                traces=policy_traces,
                scheduler_config=raw_scheduler_config,
            )
        )
        policy_names.add(name)
    return traces, npu_groups, tuple(policies)


ALL_TRACES, NPU_GROUPS, POLICIES = load_suite_definition()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SuiteError(f"{path} does not contain a JSON object")
    return value


class ProcessRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen[Any]] = set()

    def add(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes.add(process)

    def discard(self, process: subprocess.Popen[Any]) -> None:
        with self._lock:
            self._processes.discard(process)

    @staticmethod
    def terminate(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10.0)

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._processes)
        for process in processes:
            self.terminate(process)


PROCESS_REGISTRY = ProcessRegistry()
STOP_EVENT = threading.Event()
INDEX_LOCK = threading.Lock()


def condition_dir(output_root: Path, policy: PolicySpec, trace: str) -> Path:
    return output_root / "policies" / policy.name / "runs" / trace


def policy_runtime_config(policy: PolicySpec) -> dict[str, Any]:
    """Return policy identity without its independently extensible trace coverage."""
    return {
        "name": policy.name,
        "scheduler_cls": policy.scheduler_cls,
        "scheduler_config": policy.scheduler_config,
    }


def stored_policy_runtime_config(value: Any) -> dict[str, Any] | None:
    """Normalize current and pre-suite-config status records for resumability."""
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    scheduler_cls = value.get("scheduler_cls")
    if not isinstance(name, str) or not isinstance(scheduler_cls, str):
        return None
    scheduler_config = value.get("scheduler_config")
    if isinstance(scheduler_config, dict):
        return {
            "name": name,
            "scheduler_cls": scheduler_cls,
            "scheduler_config": scheduler_config,
        }

    # Older status files stored the coverage tuple and both fixed knobs. Infer
    # only the setting consumed by that scheduler, so those results remain
    # resumable after moving the matrix into suite_config.json.
    legacy_config: dict[str, dict[str, Any]] = {}
    if scheduler_cls.endswith(("PREFLOWScheduler", "EDFPrefillScheduler")):
        legacy_config["preflow_config"] = {"max_fcfs_inflation": value.get("max_fcfs_inflation", 0.5)}
    elif scheduler_cls.endswith("PrefillOnlyScheduler"):
        legacy_config["prefill_only_config"] = {"aging_rate": value.get("prefill_only_aging_rate", 500.0)}
    return {
        "name": name,
        "scheduler_cls": scheduler_cls,
        "scheduler_config": legacy_config,
    }


def policy_manifest(policy: PolicySpec) -> dict[str, Any]:
    return {**policy_runtime_config(policy), "traces": list(policy.traces)}


def configured_value(policy: PolicySpec, section: str, key: str, default: Any) -> Any:
    return policy.scheduler_config.get(section, {}).get(key, default)


def condition_complete(path: Path, policy: PolicySpec, trace: str) -> bool:
    status_path = path / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = load_json(status_path)
        result_path = Path(str(status["result_path"]))
        summary = load_json(result_path / "summary.json")
    except (KeyError, OSError, ValueError, SuiteError, json.JSONDecodeError):
        return False
    try:
        current_trace_digest = sha256(TRACE_DIR / f"{trace}.jsonl")
        current_calibration_digest = sha256(CALIBRATION)
    except OSError:
        return False
    stored_trace_digest = status.get("trace_sha256")
    stored_calibration_digest = status.get("calibration_sha256")
    return (
        status.get("status") == "completed"
        and stored_policy_runtime_config(status.get("policy")) == policy_runtime_config(policy)
        and status.get("trace") == trace
        and (stored_trace_digest is None or stored_trace_digest == current_trace_digest)
        and (stored_calibration_digest is None or stored_calibration_digest == current_calibration_digest)
        and status.get("output_tokens") == OUTPUT_TOKENS
        and status.get("max_num_seqs") == MAX_NUM_SEQS
        and status.get("max_num_batched_tokens") == MAX_NUM_BATCHED_TOKENS
        and status.get("max_num_batched_seqs") == MAX_NUM_BATCHED_SEQS
        and status.get("tensor_parallel_size") == TENSOR_PARALLEL_SIZE
        and status.get("async_scheduling") is ASYNC_SCHEDULING
        and status.get("service_budget_s") == SERVICE_BUDGET_S
        and status.get("minimum_requests") == MINIMUM_REQUESTS
        and status.get("maximum_requests") == MAXIMUM_REQUESTS
        and summary.get("requests") == summary.get("successful_requests")
        and summary.get("target_load") == TARGET_LOAD
    )


def missing_traces(output_root: Path, policy: PolicySpec) -> list[str]:
    return [
        trace
        for trace in policy.traces
        if not condition_complete(condition_dir(output_root, policy, trace), policy, trace)
    ]


def allocate_attempt(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    existing = [
        int(child.name.split("-", 1)[1])
        for child in path.iterdir()
        if child.is_dir() and child.name.startswith("attempt-") and child.name[8:].isdigit()
    ]
    attempt = path / f"attempt-{max(existing, default=0) + 1:03d}"
    attempt.mkdir()
    return attempt


def server_additional_config(policy: PolicySpec, queue_stats_dir: Path) -> str:
    scheduler_config: dict[str, dict[str, Any]] = {
        "preflow_config": {
            "max_fcfs_inflation": 0.5,
            "micro_prefill_isl_threshold": 0,
            "max_num_batched_seqs": MAX_NUM_BATCHED_SEQS,
        },
        "prefill_only_config": {"aging_rate": 500.0},
        "queue_stats_config": {
            "enabled": True,
            "output_dir": str(queue_stats_dir),
        },
    }
    for section, values in policy.scheduler_config.items():
        scheduler_config.setdefault(section, {}).update(values)
    # These are suite invariants, not per-policy tuning dimensions.
    scheduler_config["preflow_config"]["micro_prefill_isl_threshold"] = 0
    scheduler_config["preflow_config"]["max_num_batched_seqs"] = MAX_NUM_BATCHED_SEQS
    scheduler_config["queue_stats_config"] = {
        "enabled": True,
        "output_dir": str(queue_stats_dir),
    }
    return json.dumps(
        {"scheduler_config": scheduler_config},
        separators=(",", ":"),
    )


def child_environment(npu_ids: str) -> dict[str, str]:
    environment = os.environ.copy()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        environment.pop(name, None)
    environment["NO_PROXY"] = "127.0.0.1,localhost"
    environment["no_proxy"] = "127.0.0.1,localhost"
    environment["ASCEND_RT_VISIBLE_DEVICES"] = npu_ids
    environment["VLLM_USE_V1"] = "1"
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + current_pythonpath if current_pythonpath else "")
    return environment


def server_command(
    policy: PolicySpec,
    port: int,
    queue_stats_dir: Path,
) -> list[str]:
    vllm_bin = os.environ.get("VLLM_BIN", "vllm")
    return [
        vllm_bin,
        "serve",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--seed",
        str(SERVER_SEED),
        "--load-format",
        "dummy",
        "--async-scheduling",
        "--enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--scheduler-cls",
        policy.scheduler_cls,
        "--additional-config",
        server_additional_config(policy, queue_stats_dir),
    ]


def read_log_tail(log_path: Path) -> str:
    with log_path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - STARTUP_LOG_TAIL_BYTES))
        return handle.read().decode(encoding="utf-8", errors="replace")


def wait_for_health(process: subprocess.Popen[Any], port: int, log_path: Path) -> None:
    pool = urllib3.PoolManager(num_pools=1, maxsize=1)
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if STOP_EVENT.is_set():
            raise SuiteError("termination requested")
        tail = read_log_tail(log_path)
        if any(marker in tail for marker in FATAL_SERVER_MARKERS):
            raise SuiteError(f"server reported a fatal startup failure\n{tail}")
        return_code = process.poll()
        if return_code is not None:
            tail = "\n".join(tail.splitlines()[-80:])
            raise SuiteError(f"server exited with code {return_code} before health check\n{tail}")
        try:
            response = pool.request(
                "GET",
                url,
                timeout=urllib3.Timeout(connect=2.0, read=5.0),
                retries=False,
            )
            if response.status == 200:
                return
        except Exception:  # noqa: BLE001 - retry until timeout/process exit.
            pass
        STOP_EVENT.wait(2.0)
    raise SuiteError(f"server did not become healthy within {STARTUP_TIMEOUT_S:.0f}s; see {log_path}")


class ManagedServer:
    def __init__(self, policy: PolicySpec, slot: int, output_root: Path):
        self.policy = policy
        self.slot = slot
        self.npu_ids = NPU_GROUPS[slot]
        self.port = BASE_PORT + slot
        self.output_root = output_root
        self.process: subprocess.Popen[Any] | None = None
        self.log_handle: Any = None
        self.attempt_dir: Path | None = None

    def start(self) -> None:
        policy_dir = self.output_root / "policies" / self.policy.name
        self.attempt_dir = allocate_attempt(policy_dir / "server_attempts")
        queue_stats_dir = self.attempt_dir / "queue_stats"
        queue_stats_dir.mkdir()
        command = server_command(self.policy, self.port, queue_stats_dir)
        atomic_write_json(
            self.attempt_dir / "server.json",
            {
                "policy": policy_manifest(self.policy),
                "slot": self.slot,
                "npu_ids": self.npu_ids,
                "port": self.port,
                "command": command,
                "command_shell": shlex.join(command),
                "started_at": utc_now(),
            },
        )
        log_path = self.attempt_dir / "server.log"
        self.log_handle = log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=child_environment(self.npu_ids),
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        PROCESS_REGISTRY.add(self.process)
        try:
            wait_for_health(self.process, self.port, log_path)
        except BaseException:
            self.stop()
            raise
        atomic_write_json(
            self.attempt_dir / "status.json",
            {"status": "healthy", "healthy_at": utc_now(), "pid": self.process.pid},
        )

    def stop(self) -> None:
        if self.process is not None:
            PROCESS_REGISTRY.terminate(self.process)
            PROCESS_REGISTRY.discard(self.process)
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None

    def failure(self) -> str | None:
        if self.process is None or self.attempt_dir is None:
            return "server is not running"
        log_path = self.attempt_dir / "server.log"
        tail = read_log_tail(log_path)
        marker = next((item for item in FATAL_SERVER_MARKERS if item in tail), None)
        if marker is not None:
            return f"server log contains {marker!r}\n{tail}"
        return_code = self.process.poll()
        if return_code is not None:
            return f"server exited with code {return_code}\n{tail}"
        return None


def client_command(server: ManagedServer, trace: str, result_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(REPLAY_SCRIPT),
        "--base-url",
        f"http://127.0.0.1:{server.port}",
        "--model",
        SERVED_MODEL_NAME,
        "--trace",
        str(TRACE_DIR / f"{trace}.jsonl"),
        "--trace-name",
        trace,
        "--calibration",
        str(CALIBRATION),
        "--output-dir",
        str(result_dir),
        "--target-load",
        str(TARGET_LOAD),
        "--service-budget-s",
        str(SERVICE_BUDGET_S),
        "--minimum-requests",
        str(MINIMUM_REQUESTS),
        "--maximum-requests",
        str(MAXIMUM_REQUESTS),
        "--max-model-len",
        str(MAX_MODEL_LEN),
    ]


def run_client(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
    server: ManagedServer,
) -> int:
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=HERE,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        PROCESS_REGISTRY.add(process)
        try:
            while process.poll() is None:
                if STOP_EVENT.wait(1.0):
                    PROCESS_REGISTRY.terminate(process)
                    raise SuiteError("termination requested")
                server_failure = server.failure()
                if server_failure is not None:
                    PROCESS_REGISTRY.terminate(process)
                    raise SuiteError(server_failure)
            return int(process.returncode)
        finally:
            PROCESS_REGISTRY.discard(process)


def execute_trace(output_root: Path, server: ManagedServer, trace: str) -> None:
    destination = condition_dir(output_root, server.policy, trace)
    if condition_complete(destination, server.policy, trace):
        print(f"[slot {server.slot}] skip {server.policy.name}/{trace}: complete", flush=True)
        return
    attempt = allocate_attempt(destination)
    result_dir = attempt / "result"
    result_dir.mkdir()
    command = client_command(server, trace, result_dir)
    metadata = {
        "policy": policy_runtime_config(server.policy),
        "trace": trace,
        "trace_sha256": sha256(TRACE_DIR / f"{trace}.jsonl"),
        "calibration_sha256": sha256(CALIBRATION),
        "target_load": TARGET_LOAD,
        "output_tokens": OUTPUT_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_batched_seqs": MAX_NUM_BATCHED_SEQS,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "async_scheduling": ASYNC_SCHEDULING,
        "service_budget_s": SERVICE_BUDGET_S,
        "minimum_requests": MINIMUM_REQUESTS,
        "maximum_requests": MAXIMUM_REQUESTS,
        "npu_ids": server.npu_ids,
        "attempt": str(attempt),
        "result_path": str(result_dir),
        "command": command,
        "started_at": utc_now(),
    }
    atomic_write_json(destination / "condition.json", metadata)
    atomic_write_json(destination / "status.json", {**metadata, "status": "running"})
    print(f"[slot {server.slot}] run {server.policy.name}/{trace}", flush=True)
    started = time.monotonic()
    try:
        return_code = run_client(
            command,
            attempt / "client.log",
            child_environment(server.npu_ids),
            server,
        )
        if return_code != 0:
            raise SuiteError(f"client exited with code {return_code}; see {attempt / 'client.log'}")
        summary = load_json(result_dir / "summary.json")
        if summary.get("requests") != summary.get("successful_requests"):
            raise SuiteError("client summary contains failed requests")
        atomic_write_json(
            destination / "status.json",
            {
                **metadata,
                "status": "completed",
                "finished_at": utc_now(),
                "duration_s": time.monotonic() - started,
            },
        )
        print(f"[slot {server.slot}] done {server.policy.name}/{trace}", flush=True)
    except BaseException as exception:
        atomic_write_json(
            destination / "status.json",
            {
                **metadata,
                "status": "failed",
                "finished_at": utc_now(),
                "duration_s": time.monotonic() - started,
                "error": f"{exception.__class__.__name__}: {exception}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        refresh_index(output_root)


def execute_policy(output_root: Path, policy: PolicySpec, slot: int) -> None:
    traces = missing_traces(output_root, policy)
    if not traces:
        print(f"[slot {slot}] skip deployment {policy.name}: complete", flush=True)
        return
    server = ManagedServer(policy, slot, output_root)
    try:
        print(f"[slot {slot}] starting {policy.name} on NPUs {server.npu_ids}", flush=True)
        server.start()
        for trace in traces:
            if STOP_EVENT.is_set():
                raise SuiteError("termination requested")
            execute_trace(output_root, server, trace)
            if server.process is None or server.process.poll() is not None:
                raise SuiteError(f"server for {policy.name} exited unexpectedly")
    finally:
        server.stop()


def refresh_index(output_root: Path) -> None:
    with INDEX_LOCK:
        rows = []
        for policy in POLICIES:
            for trace in policy.traces:
                path = condition_dir(output_root, policy, trace)
                status_path = path / "status.json"
                status = load_json(status_path) if status_path.is_file() else {}
                summary: dict[str, Any] = {}
                result_value = status.get("result_path")
                if isinstance(result_value, str) and result_value:
                    summary_path = Path(result_value) / "summary.json"
                    if summary_path.is_file():
                        summary = load_json(summary_path)
                ttft = summary.get("ttft_s", {})
                client_lag = summary.get("client_dispatch_lag_s", {})
                rows.append(
                    {
                        "policy": policy.name,
                        "scheduler_cls": policy.scheduler_cls,
                        "trace": trace,
                        "max_fcfs_inflation": configured_value(policy, "preflow_config", "max_fcfs_inflation", ""),
                        "prefill_only_aging_rate": configured_value(policy, "prefill_only_config", "aging_rate", ""),
                        "target_load": TARGET_LOAD,
                        "async_scheduling": ASYNC_SCHEDULING,
                        "status": status.get("status", "pending"),
                        "requests": summary.get("requests", ""),
                        "mean_ttft_s": ttft.get("mean", ""),
                        "p95_ttft_s": ttft.get("p95", ""),
                        "p99_ttft_s": ttft.get("p99", ""),
                        "max_ttft_s": ttft.get("max", ""),
                        "p99_client_dispatch_lag_s": client_lag.get("p99", ""),
                        "result_path": status.get("result_path", ""),
                        "started_at": status.get("started_at", ""),
                        "finished_at": status.get("finished_at", ""),
                        "duration_s": status.get("duration_s", ""),
                        "error": status.get("error", ""),
                    }
                )
        destination = output_root / "run_index.csv"
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)


def validate_bundle() -> None:
    required = [
        SUITE_CONFIG,
        REPLAY_SCRIPT,
        ANALYZE_SCRIPT,
        CALIBRATION,
        TRACE_DIR / "manifest.json",
        *(TRACE_DIR / f"{trace}.jsonl" for trace in ALL_TRACES),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SuiteError("bundle is incomplete; missing: " + ", ".join(missing))
    manifest = load_json(TRACE_DIR / "manifest.json")
    for trace in ALL_TRACES:
        entry = manifest.get("traces", {}).get(trace, {})
        path = TRACE_DIR / f"{trace}.jsonl"
        if entry.get("derived_sha256") != sha256(path):
            raise SuiteError(f"trace digest mismatch: {path}")


def validate_environment() -> None:
    vllm_bin = os.environ.get("VLLM_BIN", "vllm")
    if shutil.which(vllm_bin) is None:
        raise SuiteError(f"vLLM executable not found: {vllm_bin!r}")
    if not Path(MODEL).is_dir():
        raise SuiteError(f"model directory does not exist: {MODEL}")


def suite_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "model": MODEL,
        "served_model_name": SERVED_MODEL_NAME,
        "target_load": TARGET_LOAD,
        "target_load_definition": (
            "sum of calibrated fixed-chunk execution time divided by the stretched arrival span"
        ),
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_batched_seqs": MAX_NUM_BATCHED_SEQS,
        "output_tokens": OUTPUT_TOKENS,
        "load_format": "dummy",
        "prefix_caching": False,
        "async_scheduling": ASYNC_SCHEDULING,
        "service_budget_s": SERVICE_BUDGET_S,
        "minimum_requests": MINIMUM_REQUESTS,
        "maximum_requests": MAXIMUM_REQUESTS,
        "npu_groups": NPU_GROUPS,
        "suite_config": str(SUITE_CONFIG.relative_to(HERE)),
        "suite_config_sha256": sha256(SUITE_CONFIG),
        "policies": [policy_manifest(policy) for policy in POLICIES],
        "total_conditions": sum(len(policy.traces) for policy in POLICIES),
        "components": {
            path.name: sha256(path)
            for path in (
                Path(__file__),
                SUITE_CONFIG,
                REPLAY_SCRIPT,
                ANALYZE_SCRIPT,
                CALIBRATION,
                TRACE_DIR / "manifest.json",
            )
        },
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }


def print_plan(output_root: Path) -> None:
    print(f"Model: {MODEL} (dummy weights)")
    print(f"Output: {output_root}")
    print(f"Hardware: {len(NPU_GROUPS)} concurrent TP4 servers on {', '.join(NPU_GROUPS)}")
    print(
        f"Common: target_load={TARGET_LOAD}, K={MAX_NUM_SEQS}, "
        f"chunk={MAX_NUM_BATCHED_TOKENS}, batch_width={MAX_NUM_BATCHED_SEQS}, "
        f"output_tokens={OUTPUT_TOKENS}, async={ASYNC_SCHEDULING}"
    )
    print(f"Total conditions: {sum(len(policy.traces) for policy in POLICIES)}")
    for policy in POLICIES:
        missing = missing_traces(output_root, policy) if output_root.exists() else list(policy.traces)
        print(f"  {policy.name}: {len(policy.traces)} conditions, {len(missing)} pending")


def worker(output_root: Path, slot: int, queue: Queue[PolicySpec], errors: list[str]) -> None:
    while not STOP_EVENT.is_set():
        try:
            policy = queue.get_nowait()
        except Empty:
            return
        try:
            execute_policy(output_root, policy, slot)
        except BaseException as exception:
            error = f"{policy.name}: {exception.__class__.__name__}: {exception}"
            errors.append(error)
            print(f"[slot {slot}] FAILED {error}", file=sys.stderr, flush=True)
        finally:
            queue.task_done()


def capture_git_state(output_root: Path) -> None:
    provenance = output_root / "provenance"
    provenance.mkdir(exist_ok=True)
    for name, command in {
        "git_head.txt": ["git", "rev-parse", "HEAD"],
        "git_status.txt": ["git", "status", "--short"],
    }.items():
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (provenance / name).write_text(result.stdout, encoding="utf-8")


def record_suite_configuration(output_root: Path) -> None:
    provenance = output_root / "provenance"
    history = provenance / "suite_configs"
    history.mkdir(parents=True, exist_ok=True)
    config_digest = sha256(SUITE_CONFIG)
    archived_config = history / f"{config_digest}.json"
    if not archived_config.exists():
        shutil.copyfile(SUITE_CONFIG, archived_config)
    atomic_write_json(output_root / "suite_manifest.json", suite_manifest())
    if not (provenance / "git_head.txt").exists():
        capture_git_state(output_root)


def analyze_results(output_root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(ANALYZE_SCRIPT), "--output-root", str(output_root)],
        cwd=HERE,
        check=False,
    )
    if result.returncode != 0:
        raise SuiteError(f"result analysis exited with code {result.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HERE / "benchmark_output")
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if args.plan:
        print_plan(output_root)
        return 0
    validate_bundle()
    validate_environment()
    output_root.mkdir(parents=True, exist_ok=True)
    record_suite_configuration(output_root)
    refresh_index(output_root)
    print_plan(output_root)

    queue: Queue[PolicySpec] = Queue()
    for policy in POLICIES:
        if missing_traces(output_root, policy):
            queue.put(policy)
    if queue.empty():
        print("All conditions are already complete.")
        analyze_results(output_root)
        return 0

    atomic_write_json(
        output_root / "suite_status.json",
        {
            "status": "running",
            "started_at": utc_now(),
            "pending_conditions": sum(len(missing_traces(output_root, policy)) for policy in POLICIES),
        },
    )

    errors: list[str] = []
    executor = ThreadPoolExecutor(max_workers=len(NPU_GROUPS))
    futures = [executor.submit(worker, output_root, slot, queue, errors) for slot in range(len(NPU_GROUPS))]
    try:
        for future in as_completed(futures):
            future.result()
    except KeyboardInterrupt:
        STOP_EVENT.set()
        PROCESS_REGISTRY.terminate_all()
        for future in futures:
            future.cancel()
        print("Interrupted; completed conditions remain resumable.", file=sys.stderr)
        return 130
    finally:
        STOP_EVENT.set()
        PROCESS_REGISTRY.terminate_all()
        executor.shutdown(wait=True, cancel_futures=True)
        refresh_index(output_root)

    if errors:
        atomic_write_json(
            output_root / "suite_status.json",
            {"status": "failed", "finished_at": utc_now(), "errors": errors},
        )
        print("Suite finished with failures; rerun run.py to retry incomplete conditions.", file=sys.stderr)
        return 1
    atomic_write_json(
        output_root / "suite_status.json",
        {"status": "completed", "finished_at": utc_now()},
    )
    analyze_results(output_root)
    print("Suite completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
