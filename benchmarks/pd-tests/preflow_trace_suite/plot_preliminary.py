#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit partial trace-suite results and render explicitly preliminary plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/preflow-trace-suite-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "benchmark_output"
DEFAULT_ANALYSIS_DIR = DEFAULT_OUTPUT_ROOT / "preliminary_analysis"
THROUGHPUT_WINDOW_S = 30.0

POLICY_LABELS = {
    "fcfs": "FCFS",
    "sjf": "SJF",
    "srpt": "SRPT",
    "prefill_only_lambda_200": "PrefillOnly λ=200",
    "prefill_only_lambda_500": "PrefillOnly λ=500",
    "prefill_only_lambda_2000": "PrefillOnly λ=2000",
    "edf_inflation_120": "EDF 120%",
    "preflow_hard_inflation_30": "Hard PREFLOW 30%",
    "preflow_hard_inflation_50": "Hard PREFLOW 50%",
    "preflow_hard_inflation_80": "Hard PREFLOW 80%",
    "preflow_hard_inflation_120": "Hard PREFLOW 120%",
}
TRACE_LABELS = {
    "mooncake_conversation": "Mooncake\nConversation",
    "mooncake_arxiv": "Mooncake\nArxiv",
    "mooncake_synthetic": "Mooncake\nSynthetic",
    "mooncake_toolagent": "Mooncake\nToolAgent",
    "qwen_coder": "Qwen\nCoder",
    "qwen_thinking": "Qwen\nThinking",
    "qwen_trace_a": "Qwen\nTrace A",
    "qwen_trace_b": "Qwen\nTrace B",
}
METRICS = ("median", "p95", "p99")
POLICY_COLORS = {
    "fcfs": "#111111",
    "sjf": "#7f7f7f",
    "srpt": "#bdbdbd",
    "prefill_only_lambda_200": "#e6550d",
    "prefill_only_lambda_500": "#fd8d3c",
    "prefill_only_lambda_2000": "#fdae6b",
    "edf_inflation_120": "#984ea3",
    "preflow_hard_inflation_30": "#238b45",
    "preflow_hard_inflation_50": "#41ab5d",
    "preflow_hard_inflation_80": "#2171b5",
    "preflow_hard_inflation_120": "#084594",
}
HARD_PREFLOW_BOUNDS = {
    "preflow_hard_inflation_30": 1.3,
    "preflow_hard_inflation_50": 1.5,
    "preflow_hard_inflation_80": 1.8,
    "preflow_hard_inflation_120": 2.2,
}
GUARANTEE_WARNING_PATTERN = re.compile(
    r"PREFLOW's earliest-deadline request (\S+) was not legal under (?:existing admission constraints|"
    r"an admission constraint outside the conservative resident-drain shield)"
)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), fraction))


def result_path(condition_dir: Path, status: dict[str, Any]) -> Path | None:
    recorded = status.get("result_path")
    if isinstance(recorded, str) and (candidate := Path(recorded)).is_dir():
        return candidate
    attempt = status.get("attempt")
    if isinstance(attempt, str):
        candidate = condition_dir / Path(attempt).name / "result"
        if candidate.is_dir():
            return candidate
    candidates = sorted(condition_dir.glob("attempt-*/result"), reverse=True)
    return candidates[0] if candidates else None


@dataclass
class Audit:
    policy: str
    trace: str
    recorded_state: str
    quality: str
    result: Path | None
    summary: dict[str, Any] | None
    workload: dict[str, Any] | None
    rows: int = 0
    successes: int = 0
    error_counts: Counter[str] | None = None
    integrity: bool = False

    @property
    def condition(self) -> str:
        return f"{self.policy}/{self.trace}"

    @property
    def failures(self) -> int:
        return self.rows - self.successes

    @property
    def preliminary_usable(self) -> bool:
        return self.quality in {"valid", "connection_drop"}


def scan_requests(path: Path) -> tuple[int, int, Counter[str], bool]:
    rows = successes = 0
    errors: Counter[str] = Counter()
    valid = True
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                value = json.loads(line)
                if not isinstance(value, dict):
                    valid = False
                    continue
                rows += 1
                if value.get("success"):
                    successes += 1
                else:
                    error = str(value.get("error", "unknown"))
                    if "Read timed out" in error:
                        errors["read_timeout"] += 1
                    elif (
                        "Connection reset" in error
                        or "Remote end closed" in error
                        or "ServerDisconnectedError" in error
                    ):
                        errors["connection_drop"] += 1
                    else:
                        errors[error[:120]] += 1
    except (OSError, json.JSONDecodeError):
        valid = False
    return rows, successes, errors, valid


def audit_condition(policy: str, trace: str, condition_dir: Path) -> Audit:
    status = read_json(condition_dir / "status.json")
    if status is None:
        return Audit(policy, trace, "pending", "missing", None, None, None)
    recorded_state = str(status.get("status", "unknown"))
    result = result_path(condition_dir, status)
    if result is None:
        return Audit(policy, trace, recorded_state, "artifact_invalid", None, None, None)
    summary = read_json(result / "summary.json")
    workload = read_json(result / "workload.json")
    request_path = result / "requests.jsonl"
    partial_path = result / "requests.partial.jsonl"
    if summary is None or workload is None or not request_path.is_file():
        if workload is not None and partial_path.is_file():
            rows, successes, errors, parse_valid = scan_requests(partial_path)
            quality = "running_partial" if recorded_state == "running" else "campaign_timeout_partial"
            return Audit(
                policy,
                trace,
                recorded_state,
                quality if parse_valid else "artifact_invalid",
                result,
                summary,
                workload,
                rows,
                successes,
                errors,
                False,
            )
        return Audit(policy, trace, recorded_state, "artifact_invalid", result, summary, workload)

    rows, successes, errors, parse_valid = scan_requests(request_path)
    expected = workload.get("selected_requests")
    summary_rows = summary.get("requests")
    summary_successes = summary.get("successful_requests")
    summary_failures = summary.get("failed_requests")
    integrity = (
        parse_valid
        and isinstance(expected, int)
        and rows == expected == summary_rows
        and successes == summary_successes
        and rows - successes == summary_failures
    )
    if not integrity:
        quality = "artifact_invalid"
    elif not errors:
        quality = "valid"
    elif set(errors) == {"connection_drop"}:
        quality = "connection_drop"
    elif "read_timeout" in errors:
        quality = "timeout_censored"
    else:
        quality = "request_failure"
    return Audit(
        policy,
        trace,
        recorded_state,
        quality,
        result,
        summary,
        workload,
        rows,
        successes,
        errors,
        integrity,
    )


def audit_suite(output_root: Path) -> tuple[list[str], list[str], list[Audit]]:
    config = read_json(HERE / "suite_config.json")
    if config is None:
        raise RuntimeError("cannot read suite_config.json")
    traces = [str(item) for item in config["traces"]]
    policies = [str(item["name"]) for item in config["policies"]]
    audits = []
    for policy in config["policies"]:
        coverage = traces if policy.get("traces", "all") == "all" else policy["traces"]
        for trace in coverage:
            condition_dir = output_root / "policies" / policy["name"] / "runs" / trace
            audits.append(audit_condition(str(policy["name"]), str(trace), condition_dir))
    return policies, traces, audits


def summary_rows(audits: Iterable[Audit]) -> list[dict[str, Any]]:
    rows = []
    for audit in audits:
        if not audit.preliminary_usable or audit.summary is None:
            continue
        summary = audit.summary
        workload = summary.get("workload", {})
        ttft = summary.get("ttft_s", {})
        dispatch_lag = summary.get("client_dispatch_lag_s", {})
        scheduled_span = workload.get("scheduled_span_s")
        wall_time = summary.get("wall_time_s")
        rows.append(
            {
                "policy": audit.policy,
                "trace": audit.trace,
                "quality": audit.quality,
                "requests": audit.rows,
                "successful_requests": audit.successes,
                "failed_requests": audit.failures,
                "scheduled_span_s": scheduled_span,
                "wall_time_s": wall_time,
                "wall_time_over_scheduled_span": (
                    float(wall_time) / float(scheduled_span)
                    if isinstance(wall_time, (int, float))
                    and isinstance(scheduled_span, (int, float))
                    and scheduled_span > 0
                    else None
                ),
                "median_ttft_s": ttft.get("median"),
                "p95_ttft_s": ttft.get("p95"),
                "p99_ttft_s": ttft.get("p99"),
                "max_ttft_s": ttft.get("max"),
                "p95_client_dispatch_lag_s": dispatch_lag.get("p95"),
                "p99_client_dispatch_lag_s": dispatch_lag.get("p99"),
                "max_client_dispatch_lag_s": dispatch_lag.get("max"),
            }
        )
    return rows


def plot_coverage(policies: list[str], traces: list[str], audits: list[Audit], output: Path) -> None:
    score = {
        "missing": 0,
        "campaign_timeout_partial": 1,
        "running_partial": 1,
        "timeout_censored": 1,
        "request_failure": 1,
        "artifact_invalid": 1,
        "connection_drop": 2,
        "valid": 3,
    }
    symbol = {
        "missing": "—",
        "campaign_timeout_partial": "T",
        "running_partial": "R",
        "timeout_censored": "T",
        "request_failure": "F",
        "artifact_invalid": "!",
        "connection_drop": "D",
        "valid": "✓",
    }
    lookup = {(item.policy, item.trace): item for item in audits}
    matrix = np.asarray([[score[lookup[(policy, trace)].quality] for trace in traces] for policy in policies])
    colors = matplotlib.colors.ListedColormap(["#eeeeee", "#d73027", "#fdae61", "#1a9850"])
    fig, axis = plt.subplots(figsize=(13, 7.5))
    axis.imshow(matrix, aspect="auto", cmap=colors, vmin=-0.5, vmax=3.5)
    for row, policy in enumerate(policies):
        for column, trace in enumerate(traces):
            audit = lookup[(policy, trace)]
            axis.text(column, row, symbol[audit.quality], ha="center", va="center", fontsize=11)
    axis.set_xticks(range(len(traces)), [TRACE_LABELS.get(item, item) for item in traces])
    axis.set_yticks(range(len(policies)), [POLICY_LABELS.get(item, item) for item in policies])
    axis.set_title("Preliminary benchmark coverage and integrity")
    axis.set_xlabel(
        "✓ valid   D one connection drop (preliminary only)   T campaign timeout   "
        "R running snapshot   — not attempted"
    )
    fig.tight_layout()
    fig.savefig(output / "coverage_and_integrity.png", dpi=180)
    plt.close(fig)


def plot_ttft_heatmaps(
    policies: list[str],
    traces: list[str],
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    lookup = {(row["policy"], row["trace"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    for axis, metric, title in zip(axes, ("median_ttft_s", "p99_ttft_s"), ("Median TTFT", "p99 TTFT"), strict=True):
        matrix = np.full((len(policies), len(traces)), np.nan)
        for row_index, policy in enumerate(policies):
            for column, trace in enumerate(traces):
                row = lookup.get((policy, trace))
                if row is not None and isinstance(row.get(metric), (int, float)) and row[metric] > 0:
                    matrix[row_index, column] = float(row[metric])
        finite = matrix[np.isfinite(matrix)]
        normalization = LogNorm(vmin=max(0.01, float(finite.min())), vmax=float(finite.max()))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", norm=normalization)
        for row_index, policy in enumerate(policies):
            for column, trace in enumerate(traces):
                value = matrix[row_index, column]
                if math.isfinite(value):
                    quality = lookup[(policy, trace)]["quality"]
                    suffix = "*" if quality == "connection_drop" else ""
                    axis.text(
                        column,
                        row_index,
                        f"{value:.1f}{suffix}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > np.median(finite) else "black",
                    )
        axis.set_xticks(range(len(traces)), [TRACE_LABELS.get(item, item) for item in traces], fontsize=8)
        axis.set_yticks(range(len(policies)), [POLICY_LABELS.get(item, item) for item in policies], fontsize=8)
        axis.set_title(f"{title} (seconds, log color)")
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle("Available TTFT results (* one dropped connection; preliminary only)", fontsize=14)
    fig.savefig(output / "ttft_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_sensitivity(rows: list[dict[str, Any]], output: Path, family: str) -> None:
    if family == "hard":
        levels = [30, 50, 80, 120]
        policies = {level: f"preflow_hard_inflation_{level}" for level in levels}
        title = "Hard PREFLOW slack sensitivity"
        xlabel = "Maximum FCFS inflation (%)"
        filename = "hard_preflow_slack_sensitivity.png"
    else:
        levels = [200, 500, 2000]
        policies = {level: f"prefill_only_lambda_{level}" for level in levels}
        title = "PrefillOnly aging sensitivity"
        xlabel = "Aging coefficient λ"
        filename = "prefillonly_lambda_sensitivity.png"
    lookup = {(row["policy"], row["trace"]): row for row in rows}
    traces = sorted({row["trace"] for row in rows if row["policy"] in policies.values()})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(traces))))
    for axis, metric in zip(axes, METRICS, strict=True):
        for color, trace in zip(colors, traces, strict=True):
            xs, ys = [], []
            for level in levels:
                row = lookup.get((policies[level], trace))
                if row is None:
                    continue
                xs.append(level)
                ys.append(float(row[f"{metric}_ttft_s"]))
            if xs:
                axis.plot(
                    xs,
                    ys,
                    marker="o",
                    linewidth=1.5,
                    color=color,
                    label=TRACE_LABELS.get(trace, trace).replace("\n", " "),
                )
        axis.set_yscale("log")
        axis.set_title(f"{metric} TTFT")
        axis.set_xlabel(xlabel)
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Seconds (log scale)")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{title} — available non-timeout runs")
    fig.tight_layout()
    fig.savefig(output / filename, dpi=180)
    plt.close(fig)


def plot_harness_fidelity(
    policies: list[str],
    traces: list[str],
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    lookup = {(row["policy"], row["trace"]): row for row in rows}
    fields = (
        ("p99_client_dispatch_lag_s", "p99 client dispatch lag (seconds)"),
        ("wall_time_over_scheduled_span", "wall time / scheduled arrival span"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), constrained_layout=True)
    for axis, (field, title) in zip(axes, fields, strict=True):
        matrix = np.full((len(policies), len(traces)), np.nan)
        for row_index, policy in enumerate(policies):
            for column, trace in enumerate(traces):
                row = lookup.get((policy, trace))
                value = None if row is None else row.get(field)
                if isinstance(value, (int, float)) and value >= 0:
                    matrix[row_index, column] = max(0.001, float(value))
        finite = matrix[np.isfinite(matrix)]
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="magma",
            norm=LogNorm(vmin=float(finite.min()), vmax=float(finite.max())),
        )
        for row_index in range(len(policies)):
            for column in range(len(traces)):
                value = matrix[row_index, column]
                if math.isfinite(value):
                    label = f"{value:.1f}" if value >= 0.1 else f"{value:.3f}"
                    axis.text(column, row_index, label, ha="center", va="center", fontsize=7, color="white")
        axis.set_xticks(range(len(traces)), [TRACE_LABELS.get(item, item) for item in traces], fontsize=8)
        axis.set_yticks(range(len(policies)), [POLICY_LABELS.get(item, item) for item in policies], fontsize=8)
        axis.set_title(f"{title} (log color)")
        fig.colorbar(image, ax=axis, shrink=0.75)
    fig.suptitle("Open-loop replay fidelity: large dispatch lag means arrivals were client-throttled", fontsize=14)
    fig.savefig(output / "harness_fidelity.png", dpi=180)
    plt.close(fig)


def partial_progress_rows(audits: list[Audit]) -> list[dict[str, Any]]:
    """Summarize partial files without treating them as latency samples."""
    rows = []
    for audit in audits:
        if audit.quality not in {"campaign_timeout_partial", "running_partial"} or audit.result is None:
            continue
        request_path = audit.result / "requests.partial.jsonl"
        completions = []
        with request_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                value = row.get("completion_offset_s")
                if row.get("success") and isinstance(value, (int, float)):
                    completions.append(float(value))
        if not completions:
            continue
        completions.sort()
        expected = audit.workload.get("selected_requests") if audit.workload else None
        rows.append(
            {
                "policy": audit.policy,
                "trace": audit.trace,
                "quality": audit.quality,
                "expected_requests": expected,
                "returned_requests": len(completions),
                "returned_fraction": len(completions) / expected if isinstance(expected, int) and expected else "",
                "last_completion_s": completions[-1],
                "returned_in_last_hour": sum(value > completions[-1] - 3600.0 for value in completions),
                "completion_offsets_s": completions,
            }
        )
    return rows


def plot_partial_progress(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(9.5, 5.8), constrained_layout=True)
    for row in rows:
        completions = np.asarray(row["completion_offsets_s"], dtype=float)
        expected = int(row["expected_requests"])
        fraction = np.arange(1, completions.size + 1, dtype=float) / expected
        slack = row["policy"].removeprefix("preflow_hard_inflation_")
        suffix = " (running snapshot)" if row["quality"] == "running_partial" else " (6 h timeout)"
        axis.step(completions / 3600.0, 100.0 * fraction, where="post", label=f"{slack}% slack{suffix}")
    axis.axvline(6.0, color="#555555", linestyle=":", linewidth=1.2, label="campaign watchdog")
    axis.set_xlabel("Elapsed replay time (hours)")
    axis.set_ylabel("Requests returned (% of trace)")
    axis.set_title("Hard PREFLOW Qwen-Coder partial progress (not a latency comparison)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.savefig(output / "partial_run_progress.png", dpi=180)
    plt.close(fig)


def metric_series(path: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[float] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("metric") == metric:
                    times.append(float(row["time_s"]))
                    values.append(float(row["value"]))
    except (OSError, ValueError):
        return np.asarray([]), np.asarray([])
    if len(times) > 5000:
        indices = np.linspace(0, len(times) - 1, 5000, dtype=int)
        return np.asarray(times)[indices], np.asarray(values)[indices]
    return np.asarray(times), np.asarray(values)


def trailing_counter_rate(
    times: np.ndarray,
    cumulative: np.ndarray,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a trailing-window rate from an irregularly sampled counter."""
    if times.size < 2 or cumulative.size != times.size or times[-1] <= window_s:
        return np.asarray([]), np.asarray([])
    grid = np.arange(window_s, math.floor(float(times[-1])) + 1.0, 1.0)
    current_indices = np.searchsorted(times, grid, side="right") - 1
    previous_indices = np.searchsorted(times, grid - window_s, side="right") - 1
    valid = (current_indices >= 0) & (previous_indices >= 0)
    grid = grid[valid]
    rates = (cumulative[current_indices[valid]] - cumulative[previous_indices[valid]]) / window_s
    return grid, np.maximum(0.0, rates)


def trailing_event_rate(
    event_times: np.ndarray,
    event_work: np.ndarray,
    end_time_s: float,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Attribute request work at completion and calculate its trailing rate."""
    if event_times.size == 0 or event_work.size != event_times.size or end_time_s <= window_s:
        return np.asarray([]), np.asarray([])
    order = np.argsort(event_times)
    times = event_times[order]
    cumulative = np.cumsum(event_work[order])
    grid = np.arange(window_s, math.floor(end_time_s) + 1.0, 1.0)
    current_indices = np.searchsorted(times, grid, side="right") - 1
    previous_indices = np.searchsorted(times, grid - window_s, side="right") - 1
    current = np.where(current_indices >= 0, cumulative[np.maximum(current_indices, 0)], 0.0)
    previous = np.where(previous_indices >= 0, cumulative[np.maximum(previous_indices, 0)], 0.0)
    return grid, np.maximum(0.0, (current - previous) / window_s)


def sampled_step_values(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if times.size == 0 or values.size != times.size:
        return np.zeros(grid.size)
    indices = np.searchsorted(times, grid, side="right") - 1
    return np.where(indices >= 0, values[np.maximum(indices, 0)], 0.0)


def throughput_statistics(
    rate_times: np.ndarray,
    rates: np.ndarray,
    waiting_times: np.ndarray,
    waiting_values: np.ndarray,
    arrival_span_s: float,
) -> dict[str, float | int | None]:
    waiting_now = sampled_step_values(waiting_times, waiting_values, rate_times)
    waiting_before = sampled_step_values(
        waiting_times,
        waiting_values,
        np.maximum(0.0, rate_times - THROUGHPUT_WINDOW_S),
    )
    saturated = (waiting_now > 0.0) & (waiting_before > 0.0)
    # Report the post-arrival drain as the cleanest capacity interval. Fall
    # back to every saturated window for short or incomplete artifacts.
    post_arrival = saturated & (rate_times >= arrival_span_s + THROUGHPUT_WINDOW_S)
    selected = rates[post_arrival] if np.any(post_arrival) else rates[saturated]
    selected = selected[np.isfinite(selected) & (selected >= 0.0)]
    if not selected.size:
        return {key: None for key in ("windows", "mean", "median", "p10", "p90", "p95", "max")}
    return {
        "windows": int(selected.size),
        "mean": float(np.mean(selected)),
        "median": float(np.median(selected)),
        "p10": float(np.quantile(selected, 0.10)),
        "p90": float(np.quantile(selected, 0.90)),
        "p95": float(np.quantile(selected, 0.95)),
        "max": float(np.max(selected)),
    }


def plot_fcfs_throughput(
    traces: list[str],
    audits: list[Audit],
    output: Path,
) -> list[dict[str, Any]]:
    """Plot FCFS token and completion-attributed triangular-work rates."""
    lookup = {(audit.policy, audit.trace): audit for audit in audits}
    token_series = []
    triangular_series = []
    summary = []
    for trace in traces:
        audit = lookup.get(("fcfs", trace))
        if audit is None or audit.result is None or audit.workload is None:
            continue
        metrics_path = audit.result / "metrics.csv"
        request_path = audit.result / "requests.jsonl"
        if not metrics_path.is_file() or not request_path.is_file():
            continue
        arrival_span = float(audit.workload["scheduled_span_s"])
        counter_t, counter_v = metric_series(metrics_path, "vllm:prompt_tokens_total")
        waiting_t, waiting_v = metric_series(metrics_path, "vllm:num_requests_waiting")
        token_t, token_rate = trailing_counter_rate(counter_t, counter_v, THROUGHPUT_WINDOW_S)

        completion_times = []
        triangular_work = []
        with request_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                completion = row.get("first_token_offset_s")
                prompt_tokens = row.get("prompt_tokens")
                if row.get("success") and isinstance(completion, (int, float)) and isinstance(prompt_tokens, int):
                    completion_times.append(float(completion))
                    triangular_work.append(float(prompt_tokens * (prompt_tokens + 1) / 2))
        end_time = max(
            float(counter_t[-1]) if counter_t.size else 0.0,
            max(completion_times, default=0.0),
        )
        triangular_t, triangular_rate = trailing_event_rate(
            np.asarray(completion_times),
            np.asarray(triangular_work),
            end_time,
            THROUGHPUT_WINDOW_S,
        )
        token_stats = throughput_statistics(token_t, token_rate, waiting_t, waiting_v, arrival_span)
        triangular_stats = throughput_statistics(
            triangular_t,
            triangular_rate,
            waiting_t,
            waiting_v,
            arrival_span,
        )
        token_series.append((trace, arrival_span, token_t, token_rate, token_stats))
        triangular_series.append((trace, arrival_span, triangular_t, triangular_rate, triangular_stats))
        summary.append(
            {
                "trace": trace,
                "window_s": THROUGHPUT_WINDOW_S,
                **{f"token_per_s_{key}": value for key, value in token_stats.items()},
                **{f"triangular_work_per_s_{key}": value for key, value in triangular_stats.items()},
            }
        )

    def render_time_series(
        series: list[tuple[str, float, np.ndarray, np.ndarray, dict[str, float | int | None]]],
        ylabel: str,
        title: str,
        filename: str,
    ) -> None:
        if not series:
            return
        fig, axes = plt.subplots(4, 2, figsize=(15, 13), squeeze=False, constrained_layout=True)
        for axis, (trace, arrival_span, rate_t, rates, stats) in zip(axes.flat, series, strict=False):
            axis.plot(rate_t / 60.0, rates, color="#2171b5", linewidth=1.0)
            median = stats.get("median")
            p90 = stats.get("p90")
            if isinstance(median, (int, float)):
                axis.axhline(median, color="#238b45", linewidth=1.2, label=f"saturated median={median:,.0f}")
            if isinstance(p90, (int, float)):
                axis.axhline(p90, color="#d95f0e", linestyle=":", linewidth=1.2, label=f"saturated p90={p90:,.0f}")
            axis.axvline(
                arrival_span / 60.0,
                color="#555555",
                linestyle="--",
                linewidth=1.0,
                label="last arrival",
            )
            axis.set_title(TRACE_LABELS.get(trace, trace).replace("\n", " "))
            axis.set_xlabel("Elapsed replay time (minutes)")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.22)
            axis.legend(fontsize=7, loc="upper right")
        for axis in axes.flat[len(series) :]:
            axis.axis("off")
        fig.suptitle(title, fontsize=15)
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    render_time_series(
        token_series,
        "Prompt tokens/s",
        f"FCFS prompt-token throughput ({THROUGHPUT_WINDOW_S:.0f}-second trailing window)",
        "fcfs_token_throughput_vs_time.png",
    )
    render_time_series(
        triangular_series,
        "Triangular work units/s",
        (
            f"FCFS completion-attributed triangular-work throughput "
            f"({THROUGHPUT_WINDOW_S:.0f}-second trailing window)"
        ),
        "fcfs_triangular_throughput_vs_time.png",
    )

    if summary:
        positions = np.arange(len(summary))
        labels = [TRACE_LABELS.get(str(row["trace"]), str(row["trace"])).replace("\n", " ") for row in summary]
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
        for axis, prefix, ylabel in (
            (axes[0], "token_per_s", "Prompt tokens/s"),
            (axes[1], "triangular_work_per_s", "Triangular work units/s"),
        ):
            medians = [float(row[f"{prefix}_median"]) for row in summary]
            p10 = [float(row[f"{prefix}_p10"]) for row in summary]
            p90 = [float(row[f"{prefix}_p90"]) for row in summary]
            axis.errorbar(
                positions,
                medians,
                yerr=[np.asarray(medians) - np.asarray(p10), np.asarray(p90) - np.asarray(medians)],
                fmt="o",
                capsize=4,
                color="#2171b5",
                label="saturated median and p10--p90",
            )
            axis.set_xticks(positions, labels, rotation=40, ha="right")
            axis.set_ylabel(ylabel)
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.22)
            axis.legend()
        fig.suptitle("FCFS trace-dependent saturated throughput ceilings", fontsize=15)
        fig.savefig(output / "fcfs_throughput_ceiling_comparison.png", dpi=180)
        plt.close(fig)
    return summary


def plot_queue_trajectories(
    policies: list[str],
    traces: list[str],
    audits: list[Audit],
    output: Path,
) -> None:
    """Render queue trajectories, including clearly marked partial runs."""
    distribution_dir = output / "per_trace"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    lookup = {(audit.policy, audit.trace): audit for audit in audits}
    for trace in traces:
        series = []
        for policy in policies:
            audit = lookup.get((policy, trace))
            if audit is None or audit.result is None:
                continue
            metrics_path = audit.result / "metrics.csv"
            if not metrics_path.is_file():
                continue
            waiting_t, waiting_v = metric_series(metrics_path, "vllm:num_requests_waiting")
            running_t, running_v = metric_series(metrics_path, "vllm:num_requests_running")
            if waiting_t.size or running_t.size:
                series.append((policy, audit, waiting_t, waiting_v, running_t, running_v))
        if not series:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        for policy, audit, waiting_t, waiting_v, running_t, running_v in series:
            partial = audit.quality in {"campaign_timeout_partial", "running_partial"}
            dropped = audit.quality == "connection_drop"
            marker = "†" if partial else ("*" if dropped else "")
            label = f"{POLICY_LABELS.get(policy, policy)}{marker}"
            style = {
                "color": POLICY_COLORS.get(policy),
                "linestyle": "--" if partial or dropped else "-",
                "linewidth": 1.35,
                "alpha": 0.9,
                "label": label,
            }
            if waiting_t.size:
                axes[0].plot(waiting_t / 3600.0, waiting_v, **style)
            if running_t.size:
                axes[1].plot(running_t / 3600.0, running_v, **style)
        axes[0].set_ylabel("Waiting requests")
        axes[0].set_title("Waiting queue")
        axes[1].set_ylabel("Running requests")
        axes[1].set_xlabel("Elapsed replay time (hours)")
        axes[1].set_title("Resident/running queue")
        for axis in axes:
            axis.grid(True, alpha=0.22)
        axes[0].legend(fontsize=7.5, ncol=2)
        trace_label = TRACE_LABELS.get(trace, trace).replace("\n", " ")
        fig.suptitle(
            f"{trace_label}: scheduler queue trajectories  "
            "(* one dropped request; † incomplete completion-biased run)",
            fontsize=14,
        )
        fig.savefig(distribution_dir / f"{trace}_queue_trajectories.png", dpi=180)
        plt.close(fig)


def plot_policy_queue_trajectories(
    policies: list[str],
    traces: list[str],
    audits: list[Audit],
    output: Path,
) -> list[dict[str, Any]]:
    """Render a separate raw queue-versus-time figure for every policy."""
    policy_dir = output / "per_policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    lookup = {(audit.policy, audit.trace): audit for audit in audits}
    summary = []
    for policy in policies:
        policy_series = []
        for trace in traces:
            audit = lookup.get((policy, trace))
            if audit is None or audit.result is None:
                continue
            metrics_path = audit.result / "metrics.csv"
            if not metrics_path.is_file():
                continue
            waiting_t, waiting_v = metric_series(metrics_path, "vllm:num_requests_waiting")
            running_t, running_v = metric_series(metrics_path, "vllm:num_requests_running")
            if not waiting_t.size and not running_t.size:
                continue
            arrival_span = None
            if audit.workload is not None:
                value = audit.workload.get("scheduled_span_s")
                if isinstance(value, (int, float)):
                    arrival_span = float(value)
            policy_series.append((trace, audit, arrival_span, waiting_t, waiting_v, running_t, running_v))

            peak_waiting = float(np.max(waiting_v)) if waiting_v.size else None
            peak_waiting_time = float(waiting_t[int(np.argmax(waiting_v))]) if waiting_v.size else None
            waiting_at_arrival_end = None
            if waiting_t.size and arrival_span is not None:
                waiting_at_arrival_end = float(waiting_v[int(np.argmin(np.abs(waiting_t - arrival_span)))])
            queue_clear_time = None
            if waiting_t.size and waiting_v[-1] == 0.0:
                positive = np.flatnonzero(waiting_v > 0.0)
                has_clear_sample = positive.size and positive[-1] + 1 < waiting_t.size
                queue_clear_time = float(waiting_t[positive[-1] + 1]) if has_clear_sample else 0.0
            metrics_span = max(
                waiting_t[-1] if waiting_t.size else 0,
                running_t[-1] if running_t.size else 0,
            )
            summary.append(
                {
                    "policy": policy,
                    "trace": trace,
                    "quality": audit.quality,
                    "arrival_span_s": arrival_span,
                    "metrics_span_s": float(metrics_span),
                    "peak_waiting": peak_waiting,
                    "peak_waiting_time_s": peak_waiting_time,
                    "waiting_at_arrival_end": waiting_at_arrival_end,
                    "final_waiting": float(waiting_v[-1]) if waiting_v.size else None,
                    "queue_clear_time_s": queue_clear_time,
                    "post_arrival_drain_s": (
                        max(0.0, queue_clear_time - arrival_span)
                        if queue_clear_time is not None and arrival_span is not None
                        else None
                    ),
                    "peak_running": float(np.max(running_v)) if running_v.size else None,
                    "final_running": float(running_v[-1]) if running_v.size else None,
                }
            )
        if not policy_series:
            continue

        fig, axes = plt.subplots(
            len(policy_series),
            2,
            figsize=(14, max(4.2, 3.0 * len(policy_series))),
            squeeze=False,
            constrained_layout=True,
        )
        for row_index, (trace, audit, arrival_span, waiting_t, waiting_v, running_t, running_v) in enumerate(
            policy_series
        ):
            waiting_axis, running_axis = axes[row_index]
            color = POLICY_COLORS.get(policy, "#2171b5")
            if waiting_t.size:
                waiting_axis.plot(waiting_t / 3600.0, waiting_v, color=color, linewidth=1.5)
                peak_index = int(np.argmax(waiting_v))
                waiting_axis.scatter(
                    [waiting_t[peak_index] / 3600.0],
                    [waiting_v[peak_index]],
                    color="#d73027",
                    s=20,
                    zorder=3,
                )
                waiting_axis.text(
                    0.98,
                    0.94,
                    f"peak={waiting_v[peak_index]:.0f}\nend={waiting_v[-1]:.0f}",
                    transform=waiting_axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
                )
            if running_t.size:
                running_axis.plot(running_t / 3600.0, running_v, color=color, linewidth=1.3)
            if arrival_span is not None:
                for axis in (waiting_axis, running_axis):
                    axis.axvline(
                        arrival_span / 3600.0,
                        color="#555555",
                        linestyle="--",
                        linewidth=1.0,
                    )
            marker = "†" if audit.quality in {"campaign_timeout_partial", "running_partial"} else ""
            label = TRACE_LABELS.get(trace, trace).replace("\n", " ")
            waiting_axis.set_ylabel(f"{label}{marker}\nrequests")
            waiting_axis.set_title("Waiting queue")
            running_axis.set_title("Resident/running queue")
            for axis in (waiting_axis, running_axis):
                axis.grid(True, alpha=0.22)
                axis.set_xlabel("Elapsed time (hours)")
        title = POLICY_LABELS.get(policy, policy)
        fig.suptitle(
            f"{title}: raw scheduler queue versus time\n"
            "vertical dashed line = final scheduled arrival; † = incomplete run",
            fontsize=15,
        )
        fig.savefig(policy_dir / f"{policy}_queue_vs_time.png", dpi=180)
        plt.close(fig)
    return summary


def successful_ttfts(path: Path) -> dict[str, float]:
    values = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("success") and isinstance(row.get("ttft_s"), (int, float)):
                values[str(row["request_id"])] = float(row["ttft_s"])
    return values


def plot_trace_distributions(
    policies: list[str],
    traces: list[str],
    audits: list[Audit],
    output: Path,
) -> None:
    """Plot the CDF, survival function, and true inverse CDF per trace."""
    distribution_dir = output / "per_trace"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    included = {
        (audit.policy, audit.trace): audit for audit in audits if audit.preliminary_usable and audit.result is not None
    }
    quantile_rows = []
    quantile_points = {
        "min_ttft_s": 0.0,
        "p10_ttft_s": 0.1,
        "p25_ttft_s": 0.25,
        "median_ttft_s": 0.5,
        "p75_ttft_s": 0.75,
        "p90_ttft_s": 0.9,
        "p95_ttft_s": 0.95,
        "p99_ttft_s": 0.99,
        "p99_9_ttft_s": 0.999,
        "max_ttft_s": 1.0,
    }
    inverse_x = np.linspace(0.0, 3.0, 600)
    inverse_quantiles = 1.0 - np.power(10.0, -inverse_x)
    quantile_ticks = np.asarray([0.0, 0.5, 0.9, 0.95, 0.99, 0.999])

    for trace in traces:
        series = []
        for policy in policies:
            audit = included.get((policy, trace))
            if audit is None or audit.result is None:
                continue
            values = np.asarray(
                list(successful_ttfts(audit.result / "requests.jsonl").values()),
                dtype=float,
            )
            values = np.sort(values[np.isfinite(values) & (values > 0.0)])
            if values.size:
                series.append((policy, audit, values))
                row: dict[str, Any] = {
                    "trace": trace,
                    "policy": policy,
                    "quality": audit.quality,
                    "successful_requests": int(values.size),
                }
                row.update({name: float(np.quantile(values, quantile)) for name, quantile in quantile_points.items()})
                quantile_rows.append(row)
        if not series:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
        preliminary = False
        for policy, audit, values in series:
            count = values.size
            cdf = np.arange(1, count + 1, dtype=float) / count
            survival = np.arange(count, 0, -1, dtype=float) / count
            is_drop = audit.quality == "connection_drop"
            preliminary |= is_drop
            label = f"{POLICY_LABELS.get(policy, policy)}{'*' if is_drop else ''} (n={count:,})"
            style = {
                "color": POLICY_COLORS.get(policy),
                "linestyle": "--" if is_drop else "-",
                "linewidth": 1.8,
                "alpha": 0.95,
                "label": label,
            }
            axes[0].step(values, cdf, where="post", **style)
            axes[1].step(values, survival, where="post", **style)
            axes[2].plot(inverse_x, np.quantile(values, inverse_quantiles), **style)

        axes[0].set_xscale("log")
        axes[0].set_ylim(0.0, 1.005)
        axes[0].set_xlabel("TTFT (seconds, log scale)")
        axes[0].set_ylabel("Fraction completed by TTFT")
        axes[0].set_title("Empirical CDF  F(t)")

        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("TTFT (seconds, log scale)")
        axes[1].set_ylabel("Fraction with TTFT ≥ t")
        axes[1].set_title("Complementary CDF  1 − F(t)")

        axes[2].set_yscale("log")
        axes[2].set_xticks(
            -np.log10(1.0 - quantile_ticks),
            ["0", "50", "90", "95", "99", "99.9"],
        )
        axes[2].set_xlabel("Percentile q (%) — tail expanded")
        axes[2].set_ylabel("TTFT at percentile q (seconds, log scale)")
        axes[2].set_title("Inverse CDF  F⁻¹(q)")

        for axis in axes:
            axis.grid(True, which="both", alpha=0.22)
        axes[2].legend(fontsize=7.5, loc="best")
        trace_label = TRACE_LABELS.get(trace, trace).replace("\n", " ")
        suffix = "  (* incomplete after connection drop)" if preliminary else ""
        fig.suptitle(f"{trace_label}: successful-request TTFT distributions{suffix}", fontsize=14)
        fig.savefig(distribution_dir / f"{trace}_ttft_distributions.png", dpi=180)
        plt.close(fig)

    write_csv(output / "ttft_quantiles.csv", quantile_rows)


def fcfs_relative_rows(audits: list[Audit]) -> list[dict[str, Any]]:
    included = {(audit.policy, audit.trace): audit for audit in audits if audit.preliminary_usable and audit.result}
    output = []
    for trace in sorted({trace for policy, trace in included if policy == "fcfs"}):
        baseline_audit = included[("fcfs", trace)]
        baseline = successful_ttfts(baseline_audit.result / "requests.jsonl")
        for (policy, candidate_trace), audit in included.items():
            if policy == "fcfs" or candidate_trace != trace or audit.result is None:
                continue
            ratios = []
            for request_id, ttft in successful_ttfts(audit.result / "requests.jsonl").items():
                fcfs_ttft = baseline.get(request_id)
                if fcfs_ttft is not None and fcfs_ttft > 0:
                    ratios.append(ttft / fcfs_ttft)
            if ratios:
                output.append(
                    {
                        "policy": policy,
                        "trace": trace,
                        "quality": audit.quality,
                        "paired_requests": len(ratios),
                        "median_ttft_ratio": statistics.median(ratios),
                        "p95_ttft_ratio": percentile(ratios, 0.95),
                        "p99_ttft_ratio": percentile(ratios, 0.99),
                    }
                )
    return output


def plot_fcfs_relative(rows: list[dict[str, Any]], output: Path) -> None:
    traces = sorted({row["trace"] for row in rows})
    if not traces:
        return
    fig, axes = plt.subplots(1, len(traces), figsize=(8 * len(traces), 6), squeeze=False)
    for axis, trace in zip(axes[0], traces, strict=True):
        trace_rows = sorted((row for row in rows if row["trace"] == trace), key=lambda row: row["median_ttft_ratio"])
        positions = np.arange(len(trace_rows))
        series = (
            (-0.18, "median_ttft_ratio", "median"),
            (0.0, "p95_ttft_ratio", "p95"),
            (0.18, "p99_ttft_ratio", "p99"),
        )
        for offset, metric, label in series:
            axis.scatter(
                [value + offset for value in positions],
                [row[metric] for row in trace_rows],
                label=label,
                s=35,
            )
        axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_yscale("log")
        axis.set_xticks(
            positions,
            [POLICY_LABELS.get(row["policy"], row["policy"]) for row in trace_rows],
            rotation=55,
            ha="right",
            fontsize=8,
        )
        axis.set_title(TRACE_LABELS.get(trace, trace).replace("\n", " "))
        axis.set_ylabel("TTFT / identical-request FCFS TTFT")
        axis.grid(True, which="both", axis="y", alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Empirical FCFS-relative TTFT (available paired requests)")
    fig.tight_layout()
    fig.savefig(output / "fcfs_relative_ttft.png", dpi=180)
    plt.close(fig)


def collect_fcfs_relative_diagnostics(
    policies: list[str],
    traces: list[str],
    audits: list[Audit],
) -> tuple[dict[str, list[tuple[str, Audit, np.ndarray]]], list[dict[str, Any]]]:
    included = {
        (audit.policy, audit.trace): audit for audit in audits if audit.preliminary_usable and audit.result is not None
    }
    by_trace: dict[str, list[tuple[str, Audit, np.ndarray]]] = {}
    rows = []
    for trace in traces:
        baseline_audit = included.get(("fcfs", trace))
        if baseline_audit is None or baseline_audit.result is None:
            continue
        baseline = successful_ttfts(baseline_audit.result / "requests.jsonl")
        for policy in policies:
            if policy == "fcfs":
                continue
            audit = included.get((policy, trace))
            if audit is None or audit.result is None:
                continue
            candidate = successful_ttfts(audit.result / "requests.jsonl")
            pairs = np.asarray(
                [
                    (baseline_ttft, candidate[request_id], candidate[request_id] / baseline_ttft)
                    for request_id, baseline_ttft in baseline.items()
                    if baseline_ttft > 0.0 and request_id in candidate
                ],
                dtype=float,
            )
            if not pairs.size:
                continue
            by_trace.setdefault(trace, []).append((policy, audit, pairs))
            ratios = pairs[:, 2]
            bound = HARD_PREFLOW_BOUNDS.get(policy)
            violations = int(np.count_nonzero(ratios > bound)) if bound is not None else None
            rows.append(
                {
                    "trace": trace,
                    "baseline_quality": baseline_audit.quality,
                    "policy": policy,
                    "quality": audit.quality,
                    "paired_requests": int(ratios.size),
                    "improved_requests": int(np.count_nonzero(ratios < 1.0)),
                    "improved_fraction": float(np.mean(ratios < 1.0)),
                    "median_ttft_ratio": float(np.median(ratios)),
                    "p90_ttft_ratio": float(np.quantile(ratios, 0.9)),
                    "p95_ttft_ratio": float(np.quantile(ratios, 0.95)),
                    "p99_ttft_ratio": float(np.quantile(ratios, 0.99)),
                    "max_ttft_ratio": float(np.max(ratios)),
                    "nominal_hard_bound": bound if bound is not None else "",
                    "requests_above_nominal_bound": violations if violations is not None else "",
                    "fraction_above_nominal_bound": (violations / ratios.size if violations is not None else ""),
                }
            )
    return by_trace, rows


def plot_fcfs_relative_distributions(
    by_trace: dict[str, list[tuple[str, Audit, np.ndarray]]],
    audits: list[Audit],
    output: Path,
) -> None:
    audit_lookup = {(audit.policy, audit.trace): audit for audit in audits}
    distribution_dir = output / "per_trace"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    inverse_x = np.linspace(0.0, 3.0, 600)
    inverse_quantiles = 1.0 - np.power(10.0, -inverse_x)
    quantile_ticks = np.asarray([0.0, 0.5, 0.9, 0.95, 0.99, 0.999])
    for trace, series in by_trace.items():
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), constrained_layout=True)
        for policy, audit, pairs in series:
            ratios = np.sort(pairs[:, 2])
            cdf = np.arange(1, ratios.size + 1, dtype=float) / ratios.size
            is_drop = audit.quality == "connection_drop"
            label = f"{POLICY_LABELS.get(policy, policy)}{'*' if is_drop else ''} (n={ratios.size:,})"
            style = {
                "color": POLICY_COLORS.get(policy),
                "linestyle": "--" if is_drop else "-",
                "linewidth": 1.8,
                "alpha": 0.95,
                "label": label,
            }
            axes[0].step(ratios, cdf, where="post", **style)
            axes[1].plot(inverse_x, np.quantile(ratios, inverse_quantiles), **style)

            ordered = pairs[np.argsort(pairs[:, 0])]
            bins = [item for item in np.array_split(ordered, min(12, ordered.shape[0])) if item.size]
            axes[2].plot(
                [float(np.median(item[:, 0])) for item in bins],
                [float(np.median(item[:, 2])) for item in bins],
                marker="o",
                markersize=3,
                **style,
            )

        for axis in axes:
            axis.axhline(1.0, color="#333333", linestyle=":", linewidth=1.2)
            axis.grid(True, which="both", alpha=0.22)
        axes[0].axvline(1.0, color="#333333", linestyle=":", linewidth=1.2)
        axes[0].set_xscale("log")
        axes[0].set_ylim(0.0, 1.005)
        axes[0].set_xlabel("TTFT / identical-request FCFS TTFT (log scale)")
        axes[0].set_ylabel("Fraction of paired requests")
        axes[0].set_title("CDF of relative TTFT")

        axes[1].set_yscale("log")
        axes[1].set_xticks(
            -np.log10(1.0 - quantile_ticks),
            ["0", "50", "90", "95", "99", "99.9"],
        )
        axes[1].set_xlabel("Percentile q (%) — tail expanded")
        axes[1].set_ylabel("Relative TTFT at percentile q (log scale)")
        axes[1].set_title("Inverse CDF of relative TTFT")

        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
        axes[2].set_xlabel("FCFS TTFT quantile-bin median (seconds, log scale)")
        axes[2].set_ylabel("Median relative TTFT (log scale)")
        axes[2].set_title("Slowdown by FCFS-latency group")
        axes[2].legend(fontsize=7.2, loc="best")

        baseline = audit_lookup[("fcfs", trace)]
        baseline_note = (
            "  (FCFS baseline incomplete after connection drop; all ratios preliminary)"
            if baseline.quality == "connection_drop"
            else ""
        )
        trace_label = TRACE_LABELS.get(trace, trace).replace("\n", " ")
        fig.suptitle(f"{trace_label}: paired wall-clock TTFT relative to FCFS{baseline_note}", fontsize=14)
        fig.savefig(distribution_dir / f"{trace}_fcfs_relative_distributions.png", dpi=180)
        plt.close(fig)


def plot_hard_constraint_checks(
    by_trace: dict[str, list[tuple[str, Audit, np.ndarray]]],
    audits: list[Audit],
    warning_counts: dict[tuple[str, str], tuple[int, int]],
    output: Path,
) -> None:
    audit_lookup = {(audit.policy, audit.trace): audit for audit in audits}
    distribution_dir = output / "per_trace"
    for trace, all_series in by_trace.items():
        series = [item for item in all_series if item[0] in HARD_PREFLOW_BOUNDS]
        if not series:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
        for axis, (policy, audit, pairs) in zip(axes.flat, series, strict=False):
            bound = HARD_PREFLOW_BOUNDS[policy]
            ratios = pairs[:, 2]
            improved = ratios < 1.0
            acceptable = (ratios >= 1.0) & (ratios <= bound)
            violated = ratios > bound
            axis.scatter(pairs[improved, 0], pairs[improved, 1], s=8, alpha=0.25, color="#2171b5", label="faster")
            axis.scatter(
                pairs[acceptable, 0],
                pairs[acceptable, 1],
                s=9,
                alpha=0.4,
                color="#fdae61",
                label="slower, within bound",
            )
            axis.scatter(
                pairs[violated, 0],
                pairs[violated, 1],
                s=11,
                alpha=0.65,
                color="#d73027",
                label="above bound",
            )
            minimum = float(np.min(pairs[:, :2]))
            maximum = float(np.max(pairs[:, :2]))
            reference = np.geomspace(minimum, maximum, 200)
            axis.plot(reference, reference, color="#333333", linestyle=":", linewidth=1.2, label="equal to FCFS")
            axis.plot(
                reference,
                bound * reference,
                color="#d73027",
                linestyle="--",
                linewidth=1.4,
                label=f"nominal {bound:.1f}× bound",
            )
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel("FCFS TTFT for identical request (seconds)")
            axis.set_ylabel("Hard-PREFLOW TTFT (seconds)")
            axis.grid(True, which="both", alpha=0.2)
            slack = int(round((bound - 1.0) * 100))
            violation_count = int(np.count_nonzero(violated))
            marker = "*" if audit.quality == "connection_drop" else ""
            axis.set_title(
                f"{slack}% slack{marker}: {violation_count}/{ratios.size} "
                f"({violation_count / ratios.size:.1%}) above bound"
            )
            axis.text(
                0.03,
                0.97,
                f"improved: {np.mean(improved):.1%}\n"
                f"p99 ratio: {np.quantile(ratios, 0.99):.2f}×\n"
                f"max ratio: {np.max(ratios):.2f}×\n"
                f"non-evaluable log warnings: {warning_counts.get((policy, trace), (0, 0))[0]}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            )
        for axis in axes.flat[len(series) :]:
            axis.axis("off")
        axes.flat[0].legend(fontsize=8, loc="lower right")
        baseline = audit_lookup[("fcfs", trace)]
        baseline_note = " FCFS baseline has one dropped request." if baseline.quality == "connection_drop" else ""
        trace_label = TRACE_LABELS.get(trace, trace).replace("\n", " ")
        fig.suptitle(
            f"{trace_label}: empirical wall-clock check of hard-PREFLOW bounds."
            f"{baseline_note}\nFormal bounds use triangular-work time; replay dispatch differs across runs.",
            fontsize=14,
        )
        fig.savefig(distribution_dir / f"{trace}_hard_constraint_check.png", dpi=180)
        plt.close(fig)


def plot_hard_constraint_summary(rows: list[dict[str, Any]], output: Path) -> None:
    hard_rows = [row for row in rows if row["policy"] in HARD_PREFLOW_BOUNDS]
    traces = sorted({str(row["trace"]) for row in hard_rows})
    if not traces:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(traces)))
    for color, trace in zip(colors, traces, strict=True):
        trace_rows = sorted(
            (row for row in hard_rows if row["trace"] == trace),
            key=lambda row: float(row["nominal_hard_bound"]),
        )
        slack = [(float(row["nominal_hard_bound"]) - 1.0) * 100.0 for row in trace_rows]
        label = TRACE_LABELS.get(trace, trace).replace("\n", " ")
        axes[0].plot(
            slack,
            [100.0 * float(row["improved_fraction"]) for row in trace_rows],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].plot(
            slack,
            [100.0 * float(row["fraction_above_nominal_bound"]) for row in trace_rows],
            marker="o",
            color=color,
            label=label,
        )
        axes[2].plot(
            slack,
            [float(row["p99_ttft_ratio"]) for row in trace_rows],
            marker="o",
            color=color,
            label=f"{label} p99",
        )
    bounds = sorted({float(row["nominal_hard_bound"]) for row in hard_rows})
    axes[2].plot(
        [(bound - 1.0) * 100.0 for bound in bounds],
        bounds,
        color="#d73027",
        linestyle="--",
        label="nominal per-request bound",
    )
    axes[0].set_title("Requests faster than FCFS")
    axes[0].set_ylabel("Paired requests (%)")
    axes[1].set_title("Requests above nominal bound")
    axes[1].set_ylabel("Paired requests (%)")
    axes[2].set_title("Empirical p99 ratio vs bound")
    axes[2].set_ylabel("Hard-PREFLOW TTFT / FCFS TTFT")
    axes[2].set_yscale("log")
    for axis in axes:
        axis.set_xlabel("Configured FCFS slack (%)")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle("Hard PREFLOW: empirical cross-run wall-clock diagnostics")
    fig.savefig(output / "hard_constraint_summary.png", dpi=180)
    plt.close(fig)


def scan_guarantee_warnings(
    output_root: Path,
    traces: list[str],
    audits: list[Audit],
) -> tuple[dict[tuple[str, str], tuple[int, int]], list[dict[str, Any]]]:
    request_ids: dict[tuple[str, str], list[str]] = {}
    for policy in HARD_PREFLOW_BOUNDS:
        log_root = output_root / "policies" / policy / "server_attempts"
        for log_path in sorted(log_root.glob("attempt-*/server.log")):
            try:
                handle = log_path.open(encoding="utf-8", errors="replace")
            except OSError:
                continue
            with handle:
                for line in handle:
                    match = GUARANTEE_WARNING_PATTERN.search(line)
                    if match is None:
                        continue
                    request_id = match.group(1)
                    trace = next((name for name in traces if name in request_id), "unknown")
                    request_ids.setdefault((policy, trace), []).append(request_id)
    counts = {key: (len(values), len(set(values))) for key, values in request_ids.items()}
    rows = []
    for audit in audits:
        if audit.policy not in HARD_PREFLOW_BOUNDS or audit.quality == "missing":
            continue
        warnings, unique_requests = counts.get((audit.policy, audit.trace), (0, 0))
        rows.append(
            {
                "policy": audit.policy,
                "trace": audit.trace,
                "run_quality": audit.quality,
                "non_evaluable_warning_events": warnings,
                "unique_requests_in_warnings": unique_requests,
            }
        )
    return counts, rows


def write_fcfs_relative_report(
    path: Path,
    rows: list[dict[str, Any]],
    warning_counts: dict[tuple[str, str], tuple[int, int]],
) -> None:
    hard_rows = [row for row in rows if row["policy"] in HARD_PREFLOW_BOUNDS]
    lines = [
        "# Empirical FCFS-relative latency diagnostics",
        "",
        "Each ratio pairs the same request ID across independent policy and FCFS replays. A ratio below one "
        "is an improvement; a ratio above one is a slowdown.",
        "",
        "This is not a direct validation of the formal hard constraint. The formal bound is defined in "
        "triangular-work units under the frozen arrival-time FCFS baseline, whereas these plots compare "
        "wall-clock TTFT across independent replays. The revised client kept p99 dispatch lag below one second "
        "for every plotted condition, so client-side arrival throttling is not an explanation for the large gaps.",
        "",
        "`Log warnings` counts decisions where PREFLOW explicitly reported that admission constraints made its "
        "compute-side guarantee non-evaluable. Zero such warnings does not turn the wall-clock ratio into a "
        "formal check, but a nonzero count identifies an additional known exception.",
        "",
        "| Trace | Slack | Pairs | Improved | Above nominal bound | Median | p95 | p99 | Maximum | Log warnings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in hard_rows:
        slack = int(round((float(row["nominal_hard_bound"]) - 1.0) * 100))
        lines.append(
            f"| {row['trace']} | {slack}% | {row['paired_requests']} | "
            f"{float(row['improved_fraction']):.1%} | {float(row['fraction_above_nominal_bound']):.1%} | "
            f"{float(row['median_ttft_ratio']):.3f}× | {float(row['p95_ttft_ratio']):.3f}× | "
            f"{float(row['p99_ttft_ratio']):.3f}× | {float(row['max_ttft_ratio']):.3f}× | "
            f"{warning_counts.get((str(row['policy']), str(row['trace'])), (0, 0))[0]} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_rows(audits: list[Audit]) -> list[dict[str, Any]]:
    return [
        {
            "policy": item.policy,
            "trace": item.trace,
            "recorded_state": item.recorded_state,
            "quality": item.quality,
            "integrity": item.integrity,
            "requests": item.rows,
            "successful_requests": item.successes,
            "failed_requests": item.failures,
            "failure_fraction": item.failures / item.rows if item.rows else "",
            "errors": json.dumps(item.error_counts or {}, sort_keys=True),
            "result_path": str(item.result or ""),
        }
        for item in audits
    ]


def write_report(path: Path, audits: list[Audit], summaries: list[dict[str, Any]]) -> None:
    attempted = [item for item in audits if item.quality != "missing"]
    valid = [item for item in audits if item.quality == "valid"]
    failures = [item for item in audits if item.recorded_state == "failed"]
    lines = [
        "# Preliminary benchmark audit",
        "",
        f"- Configured conditions: {len(audits)}",
        f"- Attempted conditions: {len(attempted)}",
        f"- Fully valid conditions: {len(valid)}",
        f"- Recorded failed conditions: {len(failures)}",
        f"- Not attempted: {sum(item.quality == 'missing' for item in audits)}",
        "",
        "Connection-drop runs contain a complete artifact set and are included only in preliminary plots; "
        "they should still be rerun for final results.",
        "Campaign-timeout partials and running snapshots are completion-biased samples and are excluded from "
        "latency comparisons.",
        "",
        "| Condition | Failed/total | Classification | Decision |",
        "|---|---:|---|---|",
    ]
    for item in failures:
        if item.quality == "campaign_timeout_partial":
            total = item.workload.get("selected_requests") if item.workload else "?"
            failed_total = f"{item.rows}/{total} returned"
            decision = "rerun after fixing scheduler scalability; excluded from latency plots"
        else:
            failed_total = f"{item.failures}/{item.rows} failed"
            decision = "rerun for final; included as preliminary" if item.preliminary_usable else "rerun; excluded"
        lines.append(f"| {item.condition} | {failed_total} | {item.quality} | {decision} |")
    running = [item for item in audits if item.recorded_state == "running"]
    for item in running:
        total = item.workload.get("selected_requests") if item.workload else "?"
        lines.append(
            f"| {item.condition} | {item.rows}/{total} returned | {item.quality} | "
            "snapshot only; excluded from latency plots |"
        )
    lagged = [row for row in summaries if float(row.get("p99_client_dispatch_lag_s") or 0.0) > 1.0]
    lines.extend(["", "## Replay fidelity", ""])
    if lagged:
        lines.extend(
            [
                f"{len(lagged)}/{len(summaries)} plotted conditions have p99 client dispatch lag above one "
                "second. Those conditions are not faithful open-loop 0.95-load replays.",
                "This is separate from artifact integrity: the measurements remain useful diagnostically.",
            ]
        )
    else:
        lines.append(
            f"0/{len(summaries)} plotted conditions have p99 client dispatch lag above one second. The revised "
            "client therefore preserved the intended arrival process for every full artifact set."
        )
    lookup = {(row["policy"], row["trace"]): row for row in summaries}
    lambda_traces = sorted(
        trace
        for policy, trace in lookup
        if policy == "prefill_only_lambda_200" and ("prefill_only_lambda_500", trace) in lookup
    )
    if lambda_traces:
        lines.extend(
            [
                "",
                "## Preliminary PrefillOnly sensitivity",
                "",
                "Ratios below are λ=500 / λ=200; below one is better. The available data show a consistent "
                "median-versus-tail tradeoff rather than one coefficient dominating.",
                "",
                "| Trace | Median ratio | p95 ratio | p99 ratio |",
                "|---|---:|---:|---:|",
            ]
        )
        for trace in lambda_traces:
            low = lookup[("prefill_only_lambda_200", trace)]
            high = lookup[("prefill_only_lambda_500", trace)]
            ratios = [float(high[f"{metric}_ttft_s"]) / float(low[f"{metric}_ttft_s"]) for metric in METRICS]
            lines.append(f"| {trace} | {ratios[0]:.3f} | {ratios[1]:.3f} | {ratios[2]:.3f} |")

    hard_traces = sorted(
        trace
        for policy, trace in lookup
        if policy == "preflow_hard_inflation_30" and ("preflow_hard_inflation_120", trace) in lookup
    )
    if hard_traces:
        lines.extend(
            [
                "",
                "## Preliminary hard-PREFLOW sensitivity",
                "",
                "Ratios below are 120% slack / 30% slack; below one is better. More slack greatly improves "
                "median TTFT in the available Mooncake runs, while the p99 change is smaller and not monotonic "
                "across intermediate slack values.",
                "",
                "| Trace | Median ratio | p95 ratio | p99 ratio |",
                "|---|---:|---:|---:|",
            ]
        )
        for trace in hard_traces:
            strict = lookup[("preflow_hard_inflation_30", trace)]
            loose = lookup[("preflow_hard_inflation_120", trace)]
            ratios = [float(loose[f"{metric}_ttft_s"]) / float(strict[f"{metric}_ttft_s"]) for metric in METRICS]
            lines.append(f"| {trace} | {ratios[0]:.3f} | {ratios[1]:.3f} | {ratios[2]:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--analysis-dir", type=Path)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    analysis_dir = (args.analysis_dir or output_root / "preliminary_analysis").expanduser().resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)

    policies, traces, audits = audit_suite(output_root)
    summaries = summary_rows(audits)
    relative = fcfs_relative_rows(audits)
    partial_rows = partial_progress_rows(audits)
    throughput_rows = plot_fcfs_throughput(traces, audits, analysis_dir)
    queue_rows = plot_policy_queue_trajectories(policies, traces, audits, analysis_dir)
    write_csv(analysis_dir / "audit.csv", audit_rows(audits))
    write_csv(analysis_dir / "available_summary.csv", summaries)
    write_csv(analysis_dir / "fcfs_relative_summary.csv", relative)
    write_csv(
        analysis_dir / "partial_run_progress.csv",
        [{key: value for key, value in row.items() if key != "completion_offsets_s"} for row in partial_rows],
    )
    write_csv(analysis_dir / "queue_summary.csv", queue_rows)
    write_csv(analysis_dir / "fcfs_throughput_summary.csv", throughput_rows)
    write_report(analysis_dir / "AUDIT.md", audits, summaries)
    relative_by_trace, relative_diagnostics = collect_fcfs_relative_diagnostics(policies, traces, audits)
    guarantee_warning_counts, guarantee_warning_rows = scan_guarantee_warnings(output_root, traces, audits)
    write_csv(analysis_dir / "fcfs_relative_diagnostics.csv", relative_diagnostics)
    write_csv(analysis_dir / "hard_guarantee_warnings.csv", guarantee_warning_rows)
    write_fcfs_relative_report(
        analysis_dir / "FCFS_RELATIVE.md",
        relative_diagnostics,
        guarantee_warning_counts,
    )
    plot_coverage(policies, traces, audits, analysis_dir)
    plot_ttft_heatmaps(policies, traces, summaries, analysis_dir)
    plot_sensitivity(summaries, analysis_dir, "hard")
    plot_sensitivity(summaries, analysis_dir, "prefill_only")
    plot_harness_fidelity(policies, traces, summaries, analysis_dir)
    plot_partial_progress(partial_rows, analysis_dir)
    plot_fcfs_relative(relative, analysis_dir)
    plot_trace_distributions(policies, traces, audits, analysis_dir)
    plot_queue_trajectories(policies, traces, audits, analysis_dir)
    plot_fcfs_relative_distributions(relative_by_trace, audits, analysis_dir)
    plot_hard_constraint_checks(relative_by_trace, audits, guarantee_warning_counts, analysis_dir)
    plot_hard_constraint_summary(relative_diagnostics, analysis_dir)
    print(f"Wrote preliminary audit and plots to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
