#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the final synthetic PD scheduler comparison on NPUs 0,1,2,3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PROVENANCE_CWD = HERE
SETUP_SCRIPT = HERE / "setup_pd.sh"
BENCHMARK_SCRIPT = HERE / "benchmark_preflow_slowdown.py"
SUITE_CONFIG = HERE / "suite_config.json"
BUNDLED_COMPONENTS = (
    Path(__file__).resolve(),
    HERE / "README.md",
    HERE / "requirements.txt",
    SETUP_SCRIPT,
    BENCHMARK_SCRIPT,
    SUITE_CONFIG,
)

DEFAULT_MODEL = "/data/weights/Qwen3-30B-A3B-Instruct-2507/"
DEFAULT_SERVED_MODEL_NAME = "qwen"
DEFAULT_OUTPUT_PARENT = HERE / "benchmark_output"
DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_PARENT / "final-synthetic-policy-suite"

ARRIVAL_RATES = (0.5,)
DECODE_LENGTHS = (128,)
REQUEST_COUNT = 1_000
WORKLOAD_SEED = 20260728
MAX_RETRIES = 2
VISIBLE_DEVICES_ENV = "ASCEND_RT_VISIBLE_DEVICES"
REQUIRED_NPU_COUNT = 4

ROUTER_URL = "http://127.0.0.1:8090"
ROUTER_METRICS_URL = "http://127.0.0.1:29000/metrics"
PREFILL_METRICS_URL = "http://127.0.0.1:13700/metrics"
DECODE_METRICS_URL = "http://127.0.0.1:13701/metrics"
MANAGED_PORTS = (8090, 13700, 13701, 30000, 30100)

# Allow for model loading and distributed worker initialization before /health.
STARTUP_TIMEOUT_S = 3_600.0
SHUTDOWN_GRACE_S = 180.0
FORCE_KILL_WAIT_S = 30.0
PORT_RELEASE_TIMEOUT_S = 30.0

LOGGER = logging.getLogger("preflow_synthetic_policy_suite")


class SweepError(RuntimeError):
    """A failure that should stop the sweep without discarding diagnostics."""


class TerminationRequested(BaseException):
    """Raised when SIGINT or SIGTERM requests an orderly sweep shutdown."""


@dataclass(frozen=True)
class SweepConfig:
    model: str
    npu_ids: str
    served_model_name: str = DEFAULT_SERVED_MODEL_NAME
    max_model_len: int = 200_000
    gpu_memory_utilization: float = 0.7
    npus_per_worker: int = 2
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1
    prefill_workers: int = 1
    decode_workers: int = 1
    max_num_batched_tokens: int = 4_096
    prefill_max_num_seqs: int = 8
    decode_max_num_seqs: int = 16
    server_seed: int = 1_024
    router_port: int = 8_090
    worker_port_base: int = 13_700
    prefill_kv_port_base: int = 30_000
    decode_kv_port_base: int = 30_100
    request_count: int = REQUEST_COUNT
    workload_seed: int = WORKLOAD_SEED
    arrival_rates: tuple[float, ...] = ARRIVAL_RATES
    decode_lengths: tuple[int, ...] = DECODE_LENGTHS
    prefix_caching: bool = False


@dataclass(frozen=True)
class DeploymentSpec:
    ordinal: int
    name: str
    scheduler_cls: str
    scheduler_config: dict[str, dict[str, Any]]
    scheduler_label: str


@dataclass(frozen=True)
class RunSpec:
    deployment: DeploymentSpec
    arrival_rate: float
    decode_length: int

    @property
    def name(self) -> str:
        return f"rate-{number_slug(self.arrival_rate)}_decode-{self.decode_length}"


class UTCFormatter(logging.Formatter):
    converter = time.gmtime


class EventLog:
    def __init__(self, path: Path):
        self._handle = path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: str, **fields: Any) -> None:
        row = {"timestamp": utc_now(), "event": event, **fields}
        self._handle.write(json.dumps(row, sort_keys=True) + "\n")

    def close(self) -> None:
        self._handle.close()


class OutputPump(threading.Thread):
    """Persist launcher output while also forwarding it to the sweep log."""

    def __init__(self, process: subprocess.Popen[str], path: Path, label: str):
        super().__init__(name=f"output-{label}", daemon=True)
        self.process = process
        self.path = path
        self.label = label
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            assert self.process.stdout is not None
            with self.path.open("x", encoding="utf-8", buffering=1) as handle:
                for line in self.process.stdout:
                    handle.write(line)
                    LOGGER.info("[%s] %s", self.label, line.rstrip())
        except BaseException as exc:
            self.error = exc
            LOGGER.exception("Output capture failed for %s", self.label)


def scheduler_additional_config(spec: DeploymentSpec, queue_stats_dir: Path) -> str:
    """Build the common single-prefill policy configuration."""
    scheduler_config: dict[str, dict[str, Any]] = {
        "preflow_config": {
            "max_fcfs_inflation": 0.5,
            "work_model": "triangular",
            "micro_prefill_isl_threshold": 0,
            "max_num_batched_seqs": 1,
        },
        "prefill_only_config": {"aging_rate": 500.0},
        "queue_stats_config": {
            "enabled": True,
            "output_dir": str(queue_stats_dir),
        },
    }
    for section, values in spec.scheduler_config.items():
        scheduler_config.setdefault(section, {}).update(values)
    # These are experiment invariants rather than policy parameters. The work
    # model remains policy-specific so hard PREFLOW can use startup profiling.
    scheduler_config["preflow_config"]["micro_prefill_isl_threshold"] = 0
    scheduler_config["preflow_config"]["max_num_batched_seqs"] = 1
    return json.dumps({"scheduler_config": scheduler_config}, separators=(",", ":"))


class ManagedDeployment:
    def __init__(
        self,
        config: SweepConfig,
        spec: DeploymentSpec,
        attempt_dir: Path,
        events: EventLog,
    ):
        self.config = config
        self.spec = spec
        self.attempt_dir = attempt_dir
        self.events = events
        self.instance_dir = attempt_dir / "instance"
        self.process: subprocess.Popen[str] | None = None
        self.output_pump: OutputPump | None = None

    @property
    def process_manifest(self) -> Path:
        return self.instance_dir / "processes.tsv"

    def command(self) -> list[str]:
        command = [
            "bash",
            str(SETUP_SCRIPT),
            "--model",
            self.config.model,
            "--served-model-name",
            self.config.served_model_name,
            "--max-model-len",
            str(self.config.max_model_len),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
            "--npus-per-worker",
            str(self.config.npus_per_worker),
            "--tp-size",
            str(self.config.tensor_parallel_size),
            "--pp-size",
            str(self.config.pipeline_parallel_size),
            "--npu-ids",
            self.config.npu_ids,
            "--prefill-workers",
            str(self.config.prefill_workers),
            "--decode-workers",
            str(self.config.decode_workers),
            "--max-num-batched-tokens",
            str(self.config.max_num_batched_tokens),
            "--prefill-max-num-seqs",
            str(self.config.prefill_max_num_seqs),
            "--decode-max-num-seqs",
            str(self.config.decode_max_num_seqs),
            "--prefill-scheduler-cls",
            self.spec.scheduler_cls,
            "--scheduler-config-json",
            scheduler_additional_config(self.spec, self.instance_dir / "queue_stats"),
            "--seed",
            str(self.config.server_seed),
            "--router-port",
            str(self.config.router_port),
            "--worker-port-base",
            str(self.config.worker_port_base),
            "--prefill-kv-port-base",
            str(self.config.prefill_kv_port_base),
            "--decode-kv-port-base",
            str(self.config.decode_kv_port_base),
            "--startup-timeout",
            str(int(STARTUP_TIMEOUT_S)),
            "--output-dir",
            str(self.instance_dir),
        ]
        return command

    def start(self) -> None:
        assert self.process is None
        occupied = [port for port in MANAGED_PORTS if port_is_open(port)]
        if occupied:
            raise SweepError(
                "cannot start deployment because managed ports are already in use: "
                + ", ".join(str(port) for port in occupied)
            )

        command = self.command()
        write_command(self.attempt_dir / "launch_command", command)
        self.events.write(
            "deployment_starting",
            deployment=self.spec.name,
            attempt=str(self.attempt_dir),
            command=command,
        )
        LOGGER.info("Starting deployment %s", self.spec.name)
        self.process = subprocess.Popen(
            command,
            cwd=HERE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        self.output_pump = OutputPump(
            self.process,
            self.attempt_dir / "launcher.log",
            f"launcher:{self.spec.name}",
        )
        self.output_pump.start()
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop(force=True)
            raise

    def _wait_until_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        health_url = f"{ROUTER_URL}/health"
        while time.monotonic() < deadline:
            if self.output_pump is not None and self.output_pump.error is not None:
                raise SweepError(f"launcher output capture failed: {self.output_pump.error}")
            return_code = self.process.poll()
            if return_code is not None:
                raise SweepError(
                    f"deployment {self.spec.name} exited during startup with code "
                    f"{return_code}; see {self.attempt_dir / 'launcher.log'}"
                )
            if http_is_healthy(health_url):
                self.events.write(
                    "deployment_ready",
                    deployment=self.spec.name,
                    launcher_pid=self.process.pid,
                )
                LOGGER.info("Deployment %s is ready", self.spec.name)
                return
            time.sleep(2.0)
        raise SweepError(f"deployment {self.spec.name} did not become ready within {STARTUP_TIMEOUT_S:.0f}s")

    def stop(self, *, force: bool = False) -> None:
        process = self.process
        if process is None:
            return
        self.events.write(
            "deployment_stopping",
            deployment=self.spec.name,
            launcher_pid=process.pid,
            grace_seconds=SHUTDOWN_GRACE_S,
        )
        LOGGER.info(
            "Stopping deployment %s (%.0fs graceful-shutdown allowance)",
            self.spec.name,
            SHUTDOWN_GRACE_S,
        )
        forced = force
        recorded_processes = read_process_manifest(self.process_manifest)
        if force:
            LOGGER.error(
                "Failure detected in %s; immediately killing every recorded process group",
                self.spec.name,
            )
            self._force_kill()
        else:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            orphan_term_sent = False
            deadline = time.monotonic() + SHUTDOWN_GRACE_S
            while time.monotonic() < deadline:
                launcher_stopped = process.poll() is not None
                active_processes = [(role, pid) for role, pid in recorded_processes if process_group_matches(pid)]
                if launcher_stopped and not active_processes:
                    break
                if launcher_stopped and active_processes and not orphan_term_sent:
                    LOGGER.warning(
                        "Launcher exited while deployment processes remained; "
                        "sending SIGTERM to the recorded process groups"
                    )
                    for role, pid in active_processes:
                        signal_process_group(pid, signal.SIGTERM, role)
                    orphan_term_sent = True
                time.sleep(1.0)
            else:
                forced = True
                LOGGER.error(
                    "Deployment %s did not stop in %.0fs; force-killing its recorded process groups",
                    self.spec.name,
                    SHUTDOWN_GRACE_S,
                )
                self._force_kill()
        if self.output_pump is not None:
            self.output_pump.join(timeout=FORCE_KILL_WAIT_S)
        output_error = self.output_pump.error if self.output_pump is not None else None
        self._wait_for_ports_to_close()
        return_code = process.poll()
        self.events.write(
            "deployment_stopped",
            deployment=self.spec.name,
            launcher_return_code=return_code,
            forced=forced,
        )
        LOGGER.info(
            "Deployment %s stopped (launcher return code %s, forced=%s)",
            self.spec.name,
            return_code,
            forced,
        )
        self.process = None
        if output_error is not None:
            raise SweepError(f"launcher output capture failed: {output_error}")

    def _force_kill(self) -> None:
        assert self.process is not None
        for role, pid in read_process_manifest(self.process_manifest):
            force_kill_process_group(pid, role)
        force_kill_process_group(self.process.pid, "launcher")
        try:
            self.process.wait(timeout=FORCE_KILL_WAIT_S)
        except subprocess.TimeoutExpired as exc:
            raise SweepError(f"launcher PID {self.process.pid} survived SIGKILL") from exc

    def _wait_for_ports_to_close(self) -> None:
        deadline = time.monotonic() + PORT_RELEASE_TIMEOUT_S
        while time.monotonic() < deadline:
            occupied = [port for port in MANAGED_PORTS if port_is_open(port)]
            if not occupied:
                return
            time.sleep(1.0)
        occupied = [port for port in MANAGED_PORTS if port_is_open(port)]
        if occupied:
            raise SweepError(
                "deployment processes stopped but managed ports remain in use: "
                + ", ".join(str(port) for port in occupied)
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def number_text(value: float) -> str:
    return f"{value:g}"


def number_slug(value: float) -> str:
    return number_text(value).replace("-", "m").replace(".", "p")


def required_visible_npu_ids() -> str:
    raw_value = os.environ.get(VISIBLE_DEVICES_ENV)
    if raw_value is None or not raw_value.strip():
        raise SweepError(f"{VISIBLE_DEVICES_ENV} must list exactly {REQUIRED_NPU_COUNT} comma-separated NPU IDs")
    device_ids = [value.strip() for value in raw_value.split(",")]
    if len(device_ids) != REQUIRED_NPU_COUNT or any(
        not value.isascii() or not value.isdecimal() for value in device_ids
    ):
        raise SweepError(
            f"{VISIBLE_DEVICES_ENV} must contain exactly {REQUIRED_NPU_COUNT} "
            f"distinct non-negative integer NPU IDs; got {raw_value!r}"
        )
    normalized_ids = [str(int(value)) for value in device_ids]
    if len(set(normalized_ids)) != REQUIRED_NPU_COUNT:
        raise SweepError(
            f"{VISIBLE_DEVICES_ENV} must contain exactly {REQUIRED_NPU_COUNT} "
            f"distinct non-negative integer NPU IDs; got {raw_value!r}"
        )
    return ",".join(normalized_ids)


def deployment_specs(_config: SweepConfig) -> list[DeploymentSpec]:
    try:
        payload = json.loads(SUITE_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SweepError(f"cannot read {SUITE_CONFIG}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise SweepError(f"{SUITE_CONFIG} must use schema_version 1")
    raw_policies = payload.get("policies")
    if not isinstance(raw_policies, list) or not raw_policies:
        raise SweepError(f"{SUITE_CONFIG} must define a non-empty policies list")

    specs: list[DeploymentSpec] = []
    names: set[str] = set()
    for ordinal, raw in enumerate(raw_policies):
        if not isinstance(raw, dict):
            raise SweepError(f"policy {ordinal} in {SUITE_CONFIG} is not an object")
        name = raw.get("name")
        scheduler_cls = raw.get("scheduler_cls")
        scheduler_config = raw.get("scheduler_config", {})
        if not isinstance(name, str) or not name or name in names:
            raise SweepError(f"policy {ordinal} has an invalid or duplicate name")
        if not isinstance(scheduler_cls, str) or not scheduler_cls:
            raise SweepError(f"policy {name} has an invalid scheduler_cls")
        if not isinstance(scheduler_config, dict) or not all(
            isinstance(section, str) and isinstance(values, dict) for section, values in scheduler_config.items()
        ):
            raise SweepError(f"policy {name} scheduler_config must map section names to objects")
        specs.append(
            DeploymentSpec(
                ordinal=ordinal,
                name=name,
                scheduler_cls=scheduler_cls,
                scheduler_config=scheduler_config,
                scheduler_label=name,
            )
        )
        names.add(name)
    return specs


def run_specs(config: SweepConfig, deployment: DeploymentSpec) -> list[RunSpec]:
    return [
        RunSpec(deployment, rate, decode_length)
        for rate in config.arrival_rates
        for decode_length in config.decode_lengths
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed 11-policy synthetic PD scheduler comparison.")
    parser.add_argument(
        "--model",
        default=None,
        help=f"model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="optional output path; an automatic unique path is used by default",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume an existing sweep root, skipping completed conditions",
    )
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help=(
            "with --resume, rerun only conditions whose saved status is failed; "
            "completed and not-yet-started conditions are left untouched"
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the deployment/run plan without launching anything",
    )
    args = parser.parse_args()
    if args.resume is not None and args.output_root is not None:
        parser.error("--resume and --output-root are mutually exclusive")
    if args.retry_failed_only and args.resume is None:
        parser.error("--retry-failed-only requires --resume")
    return args


def configure_logging(root: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = UTCFormatter("%(asctime)sZ %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(root / "sweep.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)
    LOGGER.propagate = False


def create_unique_root(requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        requested = DEFAULT_OUTPUT_PARENT / f"sweep-{stamp}-{uuid.uuid4().hex[:8]}"
    requested = requested.expanduser().resolve()
    candidate = requested
    suffix = 1
    while candidate.exists():
        candidate = requested.with_name(f"{requested.name}-{suffix:03d}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SweepError(f"expected a JSON object in {path}")
    return value


def write_command(prefix: Path, command: list[str]) -> None:
    atomic_write_json(prefix.with_suffix(".json"), {"argv": command})
    prefix.with_suffix(".txt").write_text(
        shlex.join(command) + "\n",
        encoding="utf-8",
    )


def allocate_attempt(parent: Path) -> Path:
    attempts = parent / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in attempts.glob("attempt-*"):
        try:
            indices.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    attempt = attempts / f"attempt-{max(indices, default=0) + 1:03d}"
    attempt.mkdir(exist_ok=False)
    return attempt


def http_is_healthy(url: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5.0) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def read_process_manifest(path: Path) -> list[tuple[str, int]]:
    if not path.is_file():
        LOGGER.error("Process manifest is missing: %s", path)
        return []
    rows: list[tuple[str, int]] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows.append((row["role"], int(row["pid"])))
    except (KeyError, OSError, ValueError):
        LOGGER.exception("Unable to read process manifest %s", path)
    return rows


def process_group_matches(pid: int) -> bool:
    """Return whether the recorded process group still has any members.

    A vLLM group leader can die before its worker children. ``killpg`` remains
    valid in that state even though querying the dead leader PID does not.
    """
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        LOGGER.error("Cannot inspect process group for PID %d", pid)
        return False


def force_kill_process_group(pid: int, role: str) -> None:
    signal_process_group(pid, signal.SIGKILL, role)


def signal_process_group(pid: int, signum: signal.Signals, role: str) -> None:
    if not process_group_matches(pid):
        return
    try:
        LOGGER.warning(
            "Sending %s to %s process group %d",
            signal.Signals(signum).name,
            role,
            pid,
        )
        os.killpg(pid, signum)
    except ProcessLookupError:
        return


def terminate_subprocess(process: subprocess.Popen[str], label: str) -> None:
    if process.poll() is not None:
        return
    LOGGER.warning("Terminating active subprocess %s", label)
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        LOGGER.error("Subprocess %s did not stop; sending SIGKILL", label)
        if process_group_matches(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30.0)
    except ProcessLookupError:
        return


def benchmark_command(
    config: SweepConfig,
    run: RunSpec,
    attempt_dir: Path,
    queue_stats_dir: Path,
) -> list[str]:
    client_dir = attempt_dir / "client"
    hydra_dir = attempt_dir / "hydra"
    return [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "run.mode=benchmark",
        f"run.name={run.deployment.name}",
        "run.prefill_policy=power_of_two",
        "run.decode_policy=power_of_two",
        f"run.local_scheduler={run.deployment.scheduler_label}",
        f"run.seeds=[{config.workload_seed}]",
        f"router.base_url={ROUTER_URL}",
        f"router.metrics_url={ROUTER_METRICS_URL}",
        f"request.model={config.served_model_name}",
        f"request.decode_tokens={run.decode_length}",
        f"workload.request_count={config.request_count}",
        f"workload.max_model_len={config.max_model_len}",
        "arrival.mode=mmpp",
        f"arrival.target_request_rate={number_text(run.arrival_rate)}",
        f"output.directory={client_dir}",
        f"output.queue_stats_source_dir={queue_stats_dir}",
        f"observability.prefill_metrics_urls=[{PREFILL_METRICS_URL}]",
        f"observability.decode_metrics_urls=[{DECODE_METRICS_URL}]",
        f"hydra.run.dir={hydra_dir}",
        "hydra.output_subdir=.hydra",
        "hydra.job.chdir=false",
    ]


def run_benchmark_process(
    command: list[str],
    log_path: Path,
    label: str,
    launcher_process: subprocess.Popen[str],
    deployment_processes: list[tuple[str, int]],
) -> int:
    process = subprocess.Popen(
        command,
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    output_pump = OutputPump(process, log_path, label)
    output_pump.start()
    try:
        while process.poll() is None:
            if output_pump.error is not None:
                raise SweepError(f"benchmark output capture failed: {output_pump.error}")
            if launcher_process.poll() is not None:
                raise SweepError(f"deployment launcher exited while benchmark {label} was running")
            dead_roles = [role for role, pid in deployment_processes if not process_group_matches(pid)]
            if dead_roles:
                raise SweepError(
                    f"deployment process groups died while benchmark {label} was running: " + ", ".join(dead_roles)
                )
            time.sleep(1.0)
        output_pump.join(timeout=30.0)
        if output_pump.error is not None:
            raise SweepError(f"benchmark output capture failed: {output_pump.error}")
        return process.returncode
    except BaseException:
        terminate_subprocess(process, label)
        output_pump.join(timeout=30.0)
        raise


def run_is_complete(condition_dir: Path) -> bool:
    status_path = condition_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = load_json(status_path)
    except (OSError, ValueError, SweepError):
        return False
    result_value = status.get("result_path")
    if not isinstance(result_value, str) or not result_value:
        return False
    result_path = Path(result_value)
    return status.get("status") == "completed" and (result_path / "summary.json").is_file()


def run_has_failed(condition_dir: Path) -> bool:
    status_path = condition_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = load_json(status_path)
    except (OSError, ValueError, SweepError):
        return False
    return status.get("status") == "failed"


def execute_run(
    config: SweepConfig,
    run: RunSpec,
    deployment_dir: Path,
    deployment_attempt: Path,
    launcher_process: subprocess.Popen[str],
    events: EventLog,
) -> None:
    condition_dir = deployment_dir / "runs" / run.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        condition_dir / "run.json",
        {
            "deployment": asdict(run.deployment),
            "arrival_rate": run.arrival_rate,
            "decode_length": run.decode_length,
            "request_count": config.request_count,
            "arrival_mode": "mmpp",
            "workload_seed": config.workload_seed,
        },
    )
    if run_is_complete(condition_dir):
        LOGGER.info("Skipping completed run %s/%s", run.deployment.name, run.name)
        events.write(
            "run_skipped_completed",
            deployment=run.deployment.name,
            run=run.name,
        )
        return

    attempt_dir = allocate_attempt(condition_dir)
    command = benchmark_command(
        config,
        run,
        attempt_dir,
        deployment_attempt / "instance" / "queue_stats",
    )
    write_command(attempt_dir / "benchmark_command", command)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    running_status = {
        "status": "running",
        "deployment": run.deployment.name,
        "scheduler_label": run.deployment.scheduler_label,
        "scheduler_cls": run.deployment.scheduler_cls,
        "scheduler_config": run.deployment.scheduler_config,
        "arrival_rate": run.arrival_rate,
        "decode_length": run.decode_length,
        "request_count": config.request_count,
        "attempt": str(attempt_dir),
        "started_at": started_at,
    }
    atomic_write_json(condition_dir / "status.json", running_status)
    events.write("run_started", **running_status)
    LOGGER.info(
        "Running %s: arrival_rate=%s decode_length=%d",
        run.deployment.name,
        number_text(run.arrival_rate),
        run.decode_length,
    )

    return_code: int | None = None
    try:
        return_code = run_benchmark_process(
            command,
            attempt_dir / "benchmark.log",
            f"benchmark:{run.deployment.name}:{run.name}",
            launcher_process,
            read_process_manifest(deployment_attempt / "instance" / "processes.tsv"),
        )
        if return_code != 0:
            raise SweepError(
                f"benchmark {run.deployment.name}/{run.name} exited with code "
                f"{return_code}; see {attempt_dir / 'benchmark.log'}"
            )
        result_path = attempt_dir / "client"
        if not (result_path / "summary.json").is_file():
            raise SweepError(
                f"benchmark {run.deployment.name}/{run.name} succeeded but did not write {result_path / 'summary.json'}"
            )
        status = {
            **running_status,
            "status": "completed",
            "finished_at": utc_now(),
            "duration_s": time.monotonic() - started_monotonic,
            "return_code": return_code,
            "result_path": str(result_path),
            "benchmark_log": str(attempt_dir / "benchmark.log"),
        }
        atomic_write_json(condition_dir / "status.json", status)
        events.write("run_completed", **status)
        LOGGER.info("Completed %s/%s", run.deployment.name, run.name)
    except BaseException as exc:
        status = {
            **running_status,
            "status": "failed",
            "finished_at": utc_now(),
            "duration_s": time.monotonic() - started_monotonic,
            "return_code": return_code,
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(condition_dir / "status.json", status)
        events.write("run_failed", **status)
        raise


def refresh_run_index(root: Path, config: SweepConfig) -> None:
    rows: list[dict[str, Any]] = []
    for deployment in deployment_specs(config):
        deployment_dir = root / "deployments" / f"{deployment.ordinal:02d}-{deployment.name}"
        for run in run_specs(config, deployment):
            condition_dir = deployment_dir / "runs" / run.name
            status_path = condition_dir / "status.json"
            status = load_json(status_path) if status_path.is_file() else {}
            rows.append(
                {
                    "deployment": deployment.name,
                    "scheduler_label": deployment.scheduler_label,
                    "scheduler_cls": deployment.scheduler_cls,
                    "scheduler_config": json.dumps(deployment.scheduler_config, sort_keys=True),
                    "arrival_rate": run.arrival_rate,
                    "decode_length": run.decode_length,
                    "request_count": config.request_count,
                    "status": status.get("status", "pending"),
                    "attempt": status.get("attempt", ""),
                    "result_path": status.get("result_path", ""),
                    "benchmark_log": status.get("benchmark_log", ""),
                    "started_at": status.get("started_at", ""),
                    "finished_at": status.get("finished_at", ""),
                    "duration_s": status.get("duration_s", ""),
                    "return_code": status.get("return_code", ""),
                    "error": status.get("error", ""),
                }
            )
    path = root / "run_index.csv"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def deployment_is_complete(root: Path, config: SweepConfig, spec: DeploymentSpec) -> bool:
    deployment_dir = root / "deployments" / f"{spec.ordinal:02d}-{spec.name}"
    return all(run_is_complete(deployment_dir / "runs" / run.name) for run in run_specs(config, spec))


def capture_command(path: Path, command: list[str]) -> None:
    try:
        result = subprocess.run(
            command,
            cwd=PROVENANCE_CWD,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30.0,
            check=False,
        )
        output = f"command: {shlex.join(command)}\nreturn_code: {result.returncode}\n\n{result.stdout}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = f"command: {shlex.join(command)}\nerror: {exc.__class__.__name__}: {exc}\n"
    path.write_text(output, encoding="utf-8")


def capture_provenance(root: Path) -> None:
    provenance = root / "provenance"
    provenance.mkdir(exist_ok=True)
    component_dir = provenance / "components"
    component_dir.mkdir(exist_ok=True)
    component_hashes = {}
    for source in BUNDLED_COMPONENTS:
        destination = component_dir / source.name
        shutil.copy2(source, destination)
        component_hashes[source.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    atomic_write_json(provenance / "component_hashes.json", component_hashes)
    atomic_write_json(
        provenance / "runtime.json",
        {
            "captured_at": utc_now(),
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "environment": {
                key: os.environ[key]
                for key in (
                    VISIBLE_DEVICES_ENV,
                    "VLLM_BIN",
                    "VLLM_ROUTER_BIN",
                    "PATH",
                    "PYTHONPATH",
                    "VIRTUAL_ENV",
                    "CONDA_PREFIX",
                )
                if key in os.environ
            },
        },
    )
    capture_command(provenance / "git_head.txt", ["git", "rev-parse", "HEAD"])
    capture_command(
        provenance / "git_status.txt",
        ["git", "status", "--short", "--untracked-files=no"],
    )
    capture_command(provenance / "git_diff.patch", ["git", "diff", "--binary"])


def validate_environment(root: Path, config: SweepConfig) -> None:
    executables = {
        "bash": "bash",
        "vllm": os.environ.get("VLLM_BIN", "vllm"),
        "vllm_router": os.environ.get("VLLM_ROUTER_BIN", "vllm-router"),
        "python": sys.executable,
    }
    resolved = {name: shutil.which(command) for name, command in executables.items()}
    missing = [name for name, path in resolved.items() if path is None]
    if missing:
        raise SweepError(
            "required executables were not found: " + ", ".join(f"{name}={executables[name]!r}" for name in missing)
        )
    model_path = Path(config.model).expanduser()
    if model_path.is_absolute() and not model_path.exists():
        raise SweepError(f"model path does not exist: {model_path}")
    atomic_write_json(
        root / "provenance" / "resolved_environment.json",
        {
            "validated_at": utc_now(),
            "executables": resolved,
            "model": config.model,
            "model_path_exists": model_path.exists(),
        },
    )


def validate_bundle() -> None:
    missing = [path.name for path in BUNDLED_COMPONENTS if not path.is_file()]
    if missing:
        raise SweepError("incomplete synthetic policy suite; missing files: " + ", ".join(sorted(missing)))


def new_manifest(config: SweepConfig, root: Path) -> dict[str, Any]:
    deployments = deployment_specs(config)
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "output_root": str(root),
        "config": asdict(config),
        "execution_plan": {
            "model_loads": len(deployments),
            "measured_runs": sum(len(run_specs(config, spec)) for spec in deployments),
            "deployments": [
                {
                    **asdict(spec),
                    "runs": [
                        {
                            "name": run.name,
                            "arrival_rate": run.arrival_rate,
                            "decode_length": run.decode_length,
                            "request_count": config.request_count,
                        }
                        for run in run_specs(config, spec)
                    ],
                }
                for spec in deployments
            ],
        },
        "components": {
            "launcher": str(SETUP_SCRIPT),
            "benchmark": str(BENCHMARK_SCRIPT),
            "harness": str(Path(__file__).resolve()),
            "requirements": str(HERE / "requirements.txt"),
        },
    }


def config_from_manifest(manifest: dict[str, Any]) -> SweepConfig:
    raw = manifest.get("config")
    if not isinstance(raw, dict):
        raise SweepError("resume manifest has no config object")
    values = dict(raw)
    for key in ("arrival_rates", "decode_lengths"):
        values[key] = tuple(values[key])
    return SweepConfig(**values)


def selected_runs(
    config: SweepConfig,
    root: Path,
    deployment: DeploymentSpec,
    retry_failed_only: bool,
) -> list[RunSpec]:
    runs = run_specs(config, deployment)
    if not retry_failed_only:
        return runs
    deployment_dir = root / "deployments" / f"{deployment.ordinal:02d}-{deployment.name}"
    return [run for run in runs if run_has_failed(deployment_dir / "runs" / run.name)]


def print_plan(
    config: SweepConfig,
    root: Path | None = None,
    retry_failed_only: bool = False,
) -> None:
    deployments = deployment_specs(config)
    if retry_failed_only:
        assert root is not None
        planned = [(deployment, selected_runs(config, root, deployment, True)) for deployment in deployments]
        planned = [(deployment, runs) for deployment, runs in planned if runs]
    else:
        planned = [(deployment, run_specs(config, deployment)) for deployment in deployments]
    print(f"Model: {config.model}")
    print(f"NPUs: {config.npu_ids} (one TP=2 prefill and one TP=2 decode worker)")
    print(f"Selection: {'failed conditions only' if retry_failed_only else 'full sweep'}")
    print(f"Model loads: {len(planned)}")
    print(f"Measured runs: {sum(len(runs) for _, runs in planned)}")
    for deployment, runs in planned:
        print(
            f"  {deployment.ordinal + 1}. {deployment.name}: "
            f"scheduler={deployment.scheduler_cls}, "
            f"config={json.dumps(deployment.scheduler_config, sort_keys=True)}"
        )
        for run in runs:
            print(
                f"       arrival_rate={number_text(run.arrival_rate)}, "
                f"decode_tokens={run.decode_length}, requests={config.request_count}"
            )


def install_signal_handlers() -> None:
    def handle_signal(signum: int, _frame: Any) -> None:
        raise TerminationRequested(f"received signal {signal.Signals(signum).name}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def execute_sweep(
    config: SweepConfig,
    root: Path,
    events: EventLog,
    retry_failed_only: bool = False,
) -> None:
    refresh_run_index(root, config)
    for deployment_spec in deployment_specs(config):
        runs = selected_runs(config, root, deployment_spec, retry_failed_only)
        if retry_failed_only and not runs:
            LOGGER.info("Skipping deployment %s: no failed runs", deployment_spec.name)
            events.write(
                "deployment_skipped_no_failed_runs",
                deployment=deployment_spec.name,
            )
            continue
        if not retry_failed_only and deployment_is_complete(root, config, deployment_spec):
            LOGGER.info("Skipping completed deployment %s", deployment_spec.name)
            events.write(
                "deployment_skipped_completed",
                deployment=deployment_spec.name,
            )
            continue

        deployment_dir = root / "deployments" / f"{deployment_spec.ordinal:02d}-{deployment_spec.name}"
        deployment_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(deployment_dir / "deployment.json", asdict(deployment_spec))
        last_error: BaseException | None = None
        for retry_index in range(MAX_RETRIES + 1):
            attempt_number = retry_index + 1
            attempt_dir = allocate_attempt(deployment_dir)
            manager = ManagedDeployment(config, deployment_spec, attempt_dir, events)
            deployment_started = time.monotonic()
            deployment_error: BaseException | None = None
            shutdown_error: BaseException | None = None
            LOGGER.info(
                "Policy %s attempt %d/%d",
                deployment_spec.name,
                attempt_number,
                MAX_RETRIES + 1,
            )
            try:
                manager.start()
                assert manager.process is not None
                for run in runs:
                    execute_run(
                        config,
                        run,
                        deployment_dir,
                        attempt_dir,
                        manager.process,
                        events,
                    )
                    refresh_run_index(root, config)
                    if not http_is_healthy(f"{ROUTER_URL}/health"):
                        raise SweepError(f"router became unhealthy after {deployment_spec.name}/{run.name}")
            except BaseException as exc:
                deployment_error = exc
            finally:
                try:
                    # A failed condition must not reuse any vLLM, router, or
                    # connector process from the failed attempt.
                    manager.stop(force=deployment_error is not None)
                except BaseException as stop_exc:
                    shutdown_error = stop_exc
                    LOGGER.exception(
                        "Deployment cleanup failed for %s attempt %d",
                        deployment_spec.name,
                        attempt_number,
                    )
                    events.write(
                        "deployment_shutdown_failed",
                        deployment=deployment_spec.name,
                        attempt=attempt_number,
                        error=f"{stop_exc.__class__.__name__}: {stop_exc}",
                    )

                final_error = deployment_error or shutdown_error
                atomic_write_json(
                    attempt_dir / "attempt_status.json",
                    {
                        "status": "failed" if final_error is not None else "completed",
                        "deployment": deployment_spec.name,
                        "attempt": attempt_number,
                        "max_attempts": MAX_RETRIES + 1,
                        "finished_at": utc_now(),
                        "duration_s": time.monotonic() - deployment_started,
                        "error": (None if final_error is None else f"{final_error.__class__.__name__}: {final_error}"),
                    },
                )
                refresh_run_index(root, config)

            if isinstance(deployment_error, TerminationRequested):
                raise deployment_error

            # A shutdown-only failure cannot invalidate a fully written client
            # result. Preserve it and proceed after the scoped cleanup.
            if deployment_is_complete(root, config, deployment_spec):
                if shutdown_error is not None:
                    LOGGER.warning(
                        "Policy %s produced a complete result despite a shutdown error",
                        deployment_spec.name,
                    )
                break

            last_error = (
                deployment_error
                or shutdown_error
                or SweepError(f"policy {deployment_spec.name} ended without a complete result")
            )
            if retry_index < MAX_RETRIES:
                LOGGER.error(
                    "Policy %s attempt %d failed: %s; retrying with a fresh deployment",
                    deployment_spec.name,
                    attempt_number,
                    last_error,
                )
                events.write(
                    "deployment_retrying",
                    deployment=deployment_spec.name,
                    failed_attempt=attempt_number,
                    next_attempt=attempt_number + 1,
                    error=f"{last_error.__class__.__name__}: {last_error}",
                )
                continue

            LOGGER.error(
                "Policy %s failed after %d attempts; continuing to the next policy",
                deployment_spec.name,
                MAX_RETRIES + 1,
            )
            events.write(
                "deployment_retries_exhausted",
                deployment=deployment_spec.name,
                attempts=MAX_RETRIES + 1,
                error=f"{last_error.__class__.__name__}: {last_error}",
            )
            for run in runs:
                condition_dir = deployment_dir / "runs" / run.name
                if run_is_complete(condition_dir):
                    continue
                condition_dir.mkdir(parents=True, exist_ok=True)
                status_path = condition_dir / "status.json"
                status = load_json(status_path) if status_path.is_file() else {}
                status.update(
                    {
                        "status": "failed",
                        "deployment": deployment_spec.name,
                        "scheduler_label": deployment_spec.scheduler_label,
                        "attempts_exhausted": MAX_RETRIES + 1,
                        "finished_at": utc_now(),
                        "error": f"{last_error.__class__.__name__}: {last_error}",
                    }
                )
                atomic_write_json(status_path, status)
            refresh_run_index(root, config)
            break


def main() -> int:
    args = parse_args()
    try:
        visible_npu_ids = required_visible_npu_ids()
    except SweepError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    resume_root = args.resume
    if resume_root is None and args.output_root is None and (DEFAULT_OUTPUT_ROOT / "sweep_manifest.json").is_file():
        resume_root = DEFAULT_OUTPUT_ROOT

    is_new_sweep = resume_root is None
    if resume_root is not None:
        root = resume_root.expanduser().resolve()
        manifest_path = root / "sweep_manifest.json"
        if not manifest_path.is_file():
            raise SweepError(f"resume manifest does not exist: {manifest_path}")
        manifest = load_json(manifest_path)
        config = replace(config_from_manifest(manifest), npu_ids=visible_npu_ids)
        if args.model is not None and args.model != config.model:
            raise SweepError(f"resume model {config.model!r} does not match --model {args.model!r}")
    else:
        config = SweepConfig(
            model=args.model or DEFAULT_MODEL,
            npu_ids=visible_npu_ids,
        )
        if args.plan_only:
            print_plan(config)
            return 0
        if args.output_root is None:
            root = DEFAULT_OUTPUT_ROOT.resolve()
            root.mkdir(parents=True, exist_ok=False)
        else:
            root = create_unique_root(args.output_root)
        manifest = new_manifest(config, root)
        atomic_write_json(root / "sweep_manifest.json", manifest)

    if args.plan_only:
        print_plan(config, root, args.retry_failed_only)
        return 0

    configure_logging(root)
    events = EventLog(root / "events.jsonl")
    install_signal_handlers()
    started = time.monotonic()
    atomic_write_json(
        root / "sweep_status.json",
        {"status": "running", "started_at": utc_now(), "output_root": str(root)},
    )
    LOGGER.info("Sweep output root: %s", root)
    deployments = deployment_specs(config)
    LOGGER.info(
        "Execution selection=%s uses %d deployments and %d measured runs",
        "failed-only" if args.retry_failed_only else "all-incomplete",
        sum(bool(selected_runs(config, root, spec, args.retry_failed_only)) for spec in deployments),
        sum(len(selected_runs(config, root, spec, args.retry_failed_only)) for spec in deployments),
    )
    events.write(
        "sweep_started",
        output_root=str(root),
        config=asdict(config),
        retry_failed_only=args.retry_failed_only,
    )
    try:
        validate_bundle()
        if is_new_sweep:
            capture_provenance(root)
        validate_environment(root, config)
        execute_sweep(config, root, events, args.retry_failed_only)
    except TerminationRequested as exc:
        status = {
            "status": "interrupted",
            "finished_at": utc_now(),
            "duration_s": time.monotonic() - started,
            "error": str(exc),
            "output_root": str(root),
        }
        atomic_write_json(root / "sweep_status.json", status)
        events.write("sweep_interrupted", **status)
        LOGGER.warning("Sweep interrupted: %s", exc)
        return 130
    except BaseException as exc:
        status = {
            "status": "failed",
            "finished_at": utc_now(),
            "duration_s": time.monotonic() - started,
            "error": f"{exc.__class__.__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "output_root": str(root),
        }
        atomic_write_json(root / "sweep_status.json", status)
        events.write("sweep_failed", **status)
        LOGGER.exception("Sweep failed; all completed results remain resumable in %s", root)
        return 1
    finally:
        refresh_run_index(root, config)
        events.close()

    all_complete = all(deployment_is_complete(root, config, spec) for spec in deployments)
    status = {
        "status": "completed" if all_complete else "partial",
        "finished_at": utc_now(),
        "duration_s": time.monotonic() - started,
        "output_root": str(root),
        "retry_failed_only": args.retry_failed_only,
    }
    atomic_write_json(root / "sweep_status.json", status)
    if all_complete:
        LOGGER.info("Sweep completed successfully: %s", root)
    else:
        LOGGER.error("Sweep finished with incomplete policies; rerun the same command to resume: %s", root)
    return 0 if all_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
