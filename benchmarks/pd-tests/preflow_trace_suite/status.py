#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print a read-only progress and ETA snapshot for the trace suite."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Importing the suite's existing readers gives this script exactly the same
# workload window and cost model without creating __pycache__ beside a run.
sys.dont_write_bytecode = True
import replay_trace as replay  # noqa: E402
import run as suite  # noqa: E402


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def parse_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def show_duration(value: float | None) -> str:
    if value is None:
        return "?"
    seconds = max(0, round(value))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{seconds:02d}s"


@dataclass
class Task:
    policy: str
    trace: str
    state: str
    status: dict[str, Any]
    result: Path | None = None
    span: float = 0.0
    elapsed: float | None = None
    estimate: float = 0.0
    remaining: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.policy}/{self.trace}"


def measured_duration(task: Task) -> float | None:
    value = task.status.get("duration_s")
    if isinstance(value, (int, float)) and value >= 0:
        return float(value)
    started = parse_time(task.status.get("started_at"))
    finished = parse_time(task.status.get("finished_at"))
    return None if started is None or finished is None else max(0.0, (finished - started).total_seconds())


def resolve_result_path(condition_dir: Path, status: dict[str, Any]) -> Path | None:
    recorded = status.get("result_path")
    if isinstance(recorded, str) and (candidate := Path(recorded)).is_dir():
        return candidate
    attempt = status.get("attempt")
    if isinstance(attempt, str) and (candidate := condition_dir / Path(attempt).name / "result").is_dir():
        return candidate
    return None


def condition_state(
    status: dict[str, Any] | None,
    result: Path | None,
) -> tuple[str, dict[str, Any]]:
    if status is None:
        return "pending", {}
    state = str(status.get("status", "pending"))
    if state == "completed":
        summary = read_json(result / "summary.json") if result is not None else None
        if summary is None or summary.get("requests") != summary.get("successful_requests"):
            state = "invalid"
        elif (
            status.get("client_protocol_version") != suite.CLIENT_PROTOCOL_VERSION
            or status.get("max_p99_dispatch_lag_s") != suite.MAX_P99_DISPATCH_LAG_S
            or status.get("target_load") != suite.TARGET_LOAD
            or status.get("fcfs_service_calibration_sha256")
            != suite.sha256(suite.FCFS_SERVICE_CALIBRATION)
            or status.get("fcfs_service_time_scale")
            != suite.FCFS_SERVICE_SCALES.get(str(status.get("trace")))
            or summary.get("client_protocol_version") != suite.CLIENT_PROTOCOL_VERSION
            or summary.get("target_load") != suite.TARGET_LOAD
            or summary.get("transport", {}).get("connection_limit") != 0
        ):
            state = "stale"
        elif summary.get("arrival_fidelity", {}).get("passed") is not True:
            state = "invalid"
    return state, status


def trace_spans(tasks: list[Task]) -> dict[str, float]:
    del tasks  # Spans are cheap to reconstruct and must use the current calibration.
    spans: dict[str, float] = {}
    cost_model, _ = replay.load_calibrated_chunk_cost(suite.CALIBRATION)
    for trace in suite.ALL_TRACES:
        _, workload = replay.load_workload(
            suite.TRACE_DIR / f"{trace}.jsonl",
            trace,
            suite.TARGET_LOAD,
            suite.SERVICE_BUDGET_S,
            suite.MINIMUM_REQUESTS,
            suite.MAXIMUM_REQUESTS,
            suite.MAX_MODEL_LEN,
            suite.MAX_NUM_BATCHED_TOKENS,
            cost_model,
            suite.FCFS_SERVICE_SCALES[trace],
        )
        spans[trace] = float(workload["scheduled_span_s"])
    return spans


def startup_estimate(output_root: Path) -> float:
    samples = []
    for path in output_root.glob("policies/*/server_attempts/attempt-*/status.json"):
        status = read_json(path)
        server = read_json(path.with_name("server.json"))
        healthy = parse_time(None if status is None else status.get("healthy_at"))
        started = parse_time(None if server is None else server.get("started_at"))
        if healthy is not None and started is not None and healthy >= started:
            samples.append((healthy - started).total_seconds())
    return statistics.median(samples) if samples else 180.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=suite.HERE / "benchmark_output")
    parser.add_argument("--unfinished", action="store_true", help="hide completed conditions")
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    now = datetime.now(timezone.utc)

    tasks: list[Task] = []
    by_policy: dict[str, list[Task]] = {}
    for policy in suite.POLICIES:
        for trace in policy.traces:
            condition_dir = suite.condition_dir(output_root, policy, trace)
            status = read_json(condition_dir / "status.json")
            result = resolve_result_path(condition_dir, status or {})
            state, status = condition_state(status, result)
            task = Task(policy.name, trace, state, status, result)
            tasks.append(task)
            by_policy.setdefault(policy.name, []).append(task)

    spans = trace_spans(tasks)
    ratios: dict[str, list[float]] = {}
    for task in tasks:
        task.span = spans[task.trace]
        measured = measured_duration(task)
        if task.state == "completed" and measured is not None:
            ratios.setdefault(task.trace, []).append(measured / task.span)
    global_ratios = [value for values in ratios.values() for value in values]

    for task in tasks:
        measured = measured_duration(task)
        if task.state == "completed" and measured is not None:
            task.estimate = measured
            continue
        if ratios.get(task.trace):
            task.estimate = task.span * max(1.0, statistics.median(ratios[task.trace]))
        elif global_ratios:
            task.estimate = task.span * max(1.0, statistics.median(global_ratios))
        else:
            task.estimate = task.span * 1.05 + 30.0
        if task.state == "running":
            started = parse_time(task.status.get("started_at"))
            task.elapsed = None if started is None else max(0.0, (now - started).total_seconds())
            elapsed = task.elapsed or 0.0
            if elapsed >= task.estimate:
                task.estimate = elapsed + max(60.0, task.estimate * 0.1)
            task.remaining = task.estimate - elapsed
        else:
            task.elapsed = measured if task.state == "failed" else None
            task.remaining = task.estimate

    # run.py assigns whole policies to slots and runs their traces serially.
    # Replaying that shape is more realistic than dividing 88 tasks blindly.
    slots = [0.0] * max(1, len(suite.NPU_GROUPS))
    startup = startup_estimate(output_root)
    active = {name for name, items in by_policy.items() if any(item.state == "running" for item in items)}
    for index, name in enumerate(active):
        slots[index % len(slots)] += sum(item.remaining for item in by_policy[name] if item.state != "completed")
    for policy in suite.POLICIES:
        remaining = [item for item in by_policy[policy.name] if item.state != "completed"]
        if not remaining or policy.name in active:
            continue
        slot = min(range(len(slots)), key=slots.__getitem__)
        slots[slot] += startup + sum(item.remaining for item in remaining)
    total_eta = max(slots, default=0.0)

    counts = {state: sum(task.state == state for task in tasks) for state in {task.state for task in tasks}}
    completed = counts.get("completed", 0)
    finish = now + timedelta(seconds=total_eta)
    print(f"Snapshot: {now:%Y-%m-%d %H:%M:%SZ}")
    print(
        f"Progress: {completed}/{len(tasks)} ({100 * completed / max(1, len(tasks)):.1f}%) complete; "
        f"{counts.get('running', 0)} running, {counts.get('failed', 0)} failed, "
        f"{counts.get('stale', 0)} stale"
    )
    print(f"Estimated remaining: {show_duration(total_eta)}; finish: {finish:%Y-%m-%d %H:%MZ}")
    print(f"Estimate includes {show_duration(startup)} per future policy deployment on {len(slots)} TP4 slots.\n")
    print(f"{'STATE':<9} {'TASK':<62} {'ELAPSED':>9} {'EST.RUN':>9} {'REMAIN':>9}")
    for task in tasks:
        if args.unfinished and task.state == "completed":
            continue
        print(
            f"{task.state.upper():<9} {task.name:<62} {show_duration(task.elapsed):>9} "
            f"{show_duration(task.estimate):>9} {show_duration(task.remaining):>9}"
        )
    if counts.get("failed", 0):
        print("\nFailed conditions are counted as full retries; rerun run.py to execute them.")
    if counts.get("stale", 0):
        print("\nStale conditions do not match the current harness configuration and will be rerun by run.py.")
    print(
        "ETA learns from same-trace completed runs, then falls back to the calibrated arrival span. Read-only snapshot."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
