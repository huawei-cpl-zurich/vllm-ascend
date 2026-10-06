#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a preprint-oriented analysis of the synthetic policy suite."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/preflow-synthetic-analysis-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "benchmark_output" / "final-synthetic-policy-suite"

POLICY_ORDER = [
    "fcfs",
    "sjf",
    "srpt",
    "prefill_only_lambda_200",
    "prefill_only_lambda_500",
    "prefill_only_lambda_2000",
    "edf_inflation_120",
    "preflow_hard_inflation_30",
    "preflow_hard_inflation_50",
    "preflow_hard_inflation_80",
    "preflow_hard_inflation_120",
]
POLICY_LABELS = {
    "fcfs": "FCFS",
    "sjf": "SJF",
    "srpt": "SRPT",
    "prefill_only_lambda_200": "PrefillOnly λ=200",
    "prefill_only_lambda_500": "PrefillOnly λ=500",
    "prefill_only_lambda_2000": "PrefillOnly λ=2000",
    "edf_inflation_120": "EDF 120%",
    "preflow_hard_inflation_30": "PREFLOW 30%",
    "preflow_hard_inflation_50": "PREFLOW 50%",
    "preflow_hard_inflation_80": "PREFLOW 80%",
    "preflow_hard_inflation_120": "PREFLOW 120%",
}
POLICY_SHORT_LABELS = {
    "fcfs": "FCFS",
    "sjf": "SJF",
    "srpt": "SRPT",
    "prefill_only_lambda_200": "PO-200",
    "prefill_only_lambda_500": "PO-500",
    "prefill_only_lambda_2000": "PO-2000",
    "edf_inflation_120": "EDF-120",
    "preflow_hard_inflation_30": "PF-30",
    "preflow_hard_inflation_50": "PF-50",
    "preflow_hard_inflation_80": "PF-80",
    "preflow_hard_inflation_120": "PF-120",
}
POLICY_COLORS = {
    "fcfs": "#111111",
    "sjf": "#7570b3",
    "srpt": "#d01c8b",
    "prefill_only_lambda_200": "#e6550d",
    "prefill_only_lambda_500": "#fd8d3c",
    "prefill_only_lambda_2000": "#fdae6b",
    "edf_inflation_120": "#984ea3",
    "preflow_hard_inflation_30": "#238b45",
    "preflow_hard_inflation_50": "#41ab5d",
    "preflow_hard_inflation_80": "#2171b5",
    "preflow_hard_inflation_120": "#084594",
}
POLICY_MARKERS = {
    "fcfs": "o",
    "sjf": "s",
    "srpt": "X",
    "prefill_only_lambda_200": "D",
    "prefill_only_lambda_500": "D",
    "prefill_only_lambda_2000": "D",
    "edf_inflation_120": "P",
    "preflow_hard_inflation_30": "o",
    "preflow_hard_inflation_50": "o",
    "preflow_hard_inflation_80": "o",
    "preflow_hard_inflation_120": "o",
}
POLICY_GROUPS = [
    ("Scheduling baselines", ["fcfs", "sjf", "srpt", "edf_inflation_120"]),
    (
        "PrefillOnly aging sensitivity",
        ["fcfs", "prefill_only_lambda_200", "prefill_only_lambda_500", "prefill_only_lambda_2000"],
    ),
    (
        "PREFLOW slack sensitivity",
        [
            "fcfs",
            "preflow_hard_inflation_30",
            "preflow_hard_inflation_50",
            "preflow_hard_inflation_80",
            "preflow_hard_inflation_120",
        ],
    ),
]
HARD_BOUNDS = {
    "preflow_hard_inflation_30": 1.3,
    "preflow_hard_inflation_50": 1.5,
    "preflow_hard_inflation_80": 1.8,
    "preflow_hard_inflation_120": 2.2,
}
HARD_SLACK = {policy: int(policy.rsplit("_", 1)[1]) for policy in HARD_BOUNDS}
PREFILL_ONLY_LAMBDA = {
    "prefill_only_lambda_200": 200,
    "prefill_only_lambda_500": 500,
    "prefill_only_lambda_2000": 2000,
}


@dataclass
class RunData:
    policy: str
    deployment_dir: Path
    result_dir: Path
    requests: pd.DataFrame
    summary: dict[str, Any]
    workload: dict[str, Any]
    queues: dict[str, pd.DataFrame]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--exclude-first-request",
        action="store_true",
        help="exclude the lowest source_index from each policy's request-level analysis",
    )
    return parser.parse_args()


def policy_from_deployment(path: Path) -> str:
    return path.name.split("-", 1)[1]


def latest_result_dir(deployment: Path) -> Path:
    candidates = sorted(deployment.glob("runs/*/attempts/*/client/seed-*"))
    valid = [path for path in candidates if (path / "requests.csv").is_file() and (path / "summary.json").is_file()]
    if not valid:
        raise RuntimeError(f"no complete result found under {deployment}")
    return valid[-1]


def load_queue_files(result: Path) -> dict[str, pd.DataFrame]:
    queues: dict[str, pd.DataFrame] = {}
    selected: dict[str, Path] = {}
    for role in ("prefill", "decode"):
        candidates = sorted((result / "queue_stats").glob(f"*_{role}_*.csv"))
        if not candidates:
            continue
        # Some non-profiled runs contain two equivalent prefill streams from
        # two instrumentation hooks. One stream is sufficient and avoids
        # double-counting scheduler iterations.
        selected[role] = candidates[0]
    if not selected:
        return queues
    origin_ns = min(int(pd.read_csv(path, nrows=1).iloc[0]["timestamp_ns"]) for path in selected.values())
    for role, path in selected.items():
        frame = pd.read_csv(path)
        frame = frame.sort_values("timestamp_ns").drop_duplicates("timestamp_ns")
        frame["time_s"] = (frame["timestamp_ns"].astype(np.int64) - origin_ns) / 1e9
        queues[role] = frame.reset_index(drop=True)
    return queues


def discover_runs(root: Path) -> dict[str, RunData]:
    deployments = root / "deployments"
    if not deployments.is_dir():
        raise RuntimeError(f"missing deployments directory: {deployments}")
    runs: dict[str, RunData] = {}
    for deployment in sorted(deployments.iterdir()):
        if not deployment.is_dir():
            continue
        policy = policy_from_deployment(deployment)
        result = latest_result_dir(deployment)
        requests = pd.read_csv(result / "requests.csv")
        summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
        workload = json.loads((result / "workload_manifest.json").read_text(encoding="utf-8"))
        runs[policy] = RunData(
            policy=policy,
            deployment_dir=deployment,
            result_dir=result,
            requests=requests.sort_values("source_index").reset_index(drop=True),
            summary=summary,
            workload=workload,
            queues=load_queue_files(result),
        )
    missing = [policy for policy in POLICY_ORDER if policy not in runs]
    if missing:
        raise RuntimeError(f"missing completed policies: {', '.join(missing)}")
    return {policy: runs[policy] for policy in POLICY_ORDER}


def exclude_first_request(runs: dict[str, RunData]) -> dict[str, int]:
    """Remove each policy's first trace request from request-level analysis."""
    excluded: dict[str, int] = {}
    for policy, run in runs.items():
        if len(run.requests) < 2:
            raise RuntimeError(f"cannot exclude the first request from {policy}: fewer than two requests")
        source_index = int(run.requests["source_index"].min())
        excluded[policy] = source_index
        run.requests = run.requests[run.requests["source_index"] != source_index].reset_index(drop=True)
    return excluded


def validate_runs(runs: dict[str, RunData]) -> dict[str, Any]:
    fcfs = runs["fcfs"]
    reference = fcfs.requests.set_index("source_index")
    workload_hashes = set()
    audit: dict[str, Any] = {"policies": {}, "matched_workload": True}
    for policy, run in runs.items():
        requests = run.requests
        successes = requests["success"].astype(str).str.lower().eq("true")
        indexed = requests.set_index("source_index")
        workload_hashes.add(run.workload.get("sha256"))
        matched = (
            indexed.index.equals(reference.index)
            and indexed["prompt_length"].equals(reference["prompt_length"])
            and np.allclose(
                indexed["scheduled_arrival_time_s"],
                reference["scheduled_arrival_time_s"],
                rtol=0,
                atol=1e-9,
            )
        )
        audit["policies"][policy] = {
            "requests": int(len(requests)),
            "successes": int(successes.sum()),
            "failures": int((~successes).sum()),
            "workload_sha256": run.workload.get("sha256"),
            "matched_fcfs_requests": bool(matched),
            "queue_roles": sorted(run.queues),
        }
        audit["matched_workload"] = audit["matched_workload"] and matched
        if not successes.all():
            raise RuntimeError(f"{policy} contains failed requests")
    audit["one_workload_hash"] = len(workload_hashes) == 1
    if not audit["matched_workload"] or not audit["one_workload_hash"]:
        raise RuntimeError("runs do not contain the same request-level workload")
    return audit


def q(values: pd.Series | np.ndarray, fraction: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), fraction))


def process_cpu_delta(summary: dict[str, Any], role: str) -> float | None:
    series = summary.get("observability", {}).get("series", {})
    matches = [
        value.get("delta")
        for key, value in series.items()
        if key.startswith(f"{role}/") and "/process_cpu_seconds_total" in key
    ]
    numeric = [float(value) for value in matches if isinstance(value, (int, float))]
    return numeric[0] if numeric else None


def policy_summary(runs: dict[str, RunData]) -> pd.DataFrame:
    baseline = runs["fcfs"].requests.set_index("source_index")
    baseline_mean = float(baseline["ttft_s"].mean())
    baseline_e2e = baseline["e2e_latency_s"].astype(float)
    baseline_e2e_mean = float(baseline_e2e.mean())
    rows = []
    for policy, run in runs.items():
        requests = run.requests.set_index("source_index")
        ttft = requests["ttft_s"].astype(float)
        e2e = requests["e2e_latency_s"].astype(float)
        ratio = ttft / baseline["ttft_s"].astype(float)
        e2e_ratio = e2e / baseline_e2e
        delta = ttft - baseline["ttft_s"].astype(float)
        rows.append(
            {
                "policy": policy,
                "label": POLICY_LABELS[policy],
                "requests": len(requests),
                "mean_ttft_s": float(ttft.mean()),
                "median_ttft_s": float(ttft.median()),
                "p95_ttft_s": q(ttft, 0.95),
                "p99_ttft_s": q(ttft, 0.99),
                "max_ttft_s": float(ttft.max()),
                "mean_e2e_s": float(e2e.mean()),
                "p99_e2e_s": q(e2e, 0.99),
                "mean_e2e_improvement_pct": 100.0 * (1.0 - float(e2e.mean()) / baseline_e2e_mean),
                "mean_e2e_fcfs_ratio": float(e2e_ratio.mean()),
                "median_e2e_fcfs_ratio": float(e2e_ratio.median()),
                "p95_e2e_fcfs_ratio": q(e2e_ratio, 0.95),
                "p99_e2e_fcfs_ratio": q(e2e_ratio, 0.99),
                "max_e2e_fcfs_ratio": float(e2e_ratio.max()),
                "mean_decode_phase_s": float((e2e - ttft).mean()),
                "p99_decode_phase_s": q(e2e - ttft, 0.99),
                "mean_itl_ms": 1000.0 * float(requests["inter_token_latency_s"].mean()),
                "mean_ttft_improvement_pct": 100.0 * (1.0 - float(ttft.mean()) / baseline_mean),
                "mean_fcfs_ratio": float(ratio.mean()),
                "median_fcfs_ratio": float(ratio.median()),
                "p95_fcfs_ratio": q(ratio, 0.95),
                "p99_fcfs_ratio": q(ratio, 0.99),
                "max_fcfs_ratio": float(ratio.max()),
                "max_additive_slowdown_s": float(delta.max()),
                "fraction_improved": float((ratio < 1.0).mean()),
                "wall_time_s": float(run.summary["throughput"]["wall_time_s"]),
                "completed_requests_s": float(run.summary["throughput"]["completed_requests_s"]),
                "output_tokens_s": float(run.summary["throughput"]["output_tokens_s"]),
                "average_unfinished_requests": float(run.summary["client_concurrency"]["average_unfinished_requests"]),
                "peak_unfinished_requests": int(run.summary["client_concurrency"]["peak_unfinished_requests"]),
                "prefill_process_cpu_s": process_cpu_delta(run.summary, "prefill"),
                "decode_process_cpu_s": process_cpu_delta(run.summary, "decode"),
            }
        )
    return pd.DataFrame(rows).set_index("policy").loc[POLICY_ORDER].reset_index()


def hard_constraint_summary(runs: dict[str, RunData]) -> pd.DataFrame:
    baseline = runs["fcfs"].requests.set_index("source_index")["ttft_s"].astype(float)
    rows = []
    for policy, bound in HARD_BOUNDS.items():
        ttft = runs[policy].requests.set_index("source_index")["ttft_s"].astype(float)
        ratio = ttft / baseline
        excess = ratio - bound
        rows.append(
            {
                "policy": policy,
                "label": POLICY_LABELS[policy],
                "slack_pct": HARD_SLACK[policy],
                "configured_ratio_bound": bound,
                "requests": len(ratio),
                "violations": int((excess > 0).sum()),
                "violation_fraction": float((excess > 0).mean()),
                "p95_ratio": q(ratio, 0.95),
                "p99_ratio": q(ratio, 0.99),
                "max_ratio": float(ratio.max()),
                "max_excess_ratio": float(excess.max()),
                "max_excess_pct_of_fcfs": 100.0 * float(excess.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("slack_pct")


def prompt_summary(runs: dict[str, RunData]) -> pd.DataFrame:
    baseline = runs["fcfs"].requests.set_index("source_index")
    rows = []
    for policy, run in runs.items():
        indexed = run.requests.set_index("source_index")
        for length, reference_group in baseline.groupby("prompt_length"):
            indices = reference_group.index
            ttft = indexed.loc[indices, "ttft_s"].astype(float)
            baseline_ttft = baseline.loc[indices, "ttft_s"].astype(float)
            ratio = ttft / baseline_ttft
            rows.append(
                {
                    "policy": policy,
                    "label": POLICY_LABELS[policy],
                    "prompt_tokens": int(length),
                    "requests": len(indices),
                    "mean_ttft_s": float(ttft.mean()),
                    "median_ttft_s": float(ttft.median()),
                    "p99_ttft_s": q(ttft, 0.99),
                    "ratio_of_mean_ttft": float(ttft.mean() / baseline_ttft.mean()),
                    "median_request_ratio": float(ratio.median()),
                    "p99_request_ratio": q(ratio, 0.99),
                    "max_request_ratio": float(ratio.max()),
                    "fraction_improved": float((ratio < 1.0).mean()),
                }
            )
    return pd.DataFrame(rows)


def sample_queue(frame: pd.DataFrame, grid: np.ndarray, field: str) -> np.ndarray:
    times = frame["time_s"].to_numpy(dtype=float)
    values = frame[field].to_numpy(dtype=float)
    indices = np.searchsorted(times, grid, side="right") - 1
    sampled = np.zeros(grid.size, dtype=float)
    valid = indices >= 0
    sampled[valid] = values[indices[valid]]
    return sampled


def queue_summary(runs: dict[str, RunData]) -> pd.DataFrame:
    rows = []
    for policy, run in runs.items():
        end_s = float(run.summary["throughput"]["wall_time_s"])
        grid = np.arange(0.0, np.ceil(end_s) + 1.0, 1.0)
        for role, frame in run.queues.items():
            sampled = {field: sample_queue(frame, grid, field) for field in ("waiting", "running", "total")}
            rows.append(
                {
                    "policy": policy,
                    "label": POLICY_LABELS[policy],
                    "role": role,
                    **{
                        f"{field}_{metric}": value
                        for field, values in sampled.items()
                        for metric, value in (
                            ("mean", float(values.mean())),
                            ("p95", q(values, 0.95)),
                            ("p99", q(values, 0.99)),
                            ("max", float(values.max())),
                        )
                    },
                }
            )
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def empirical_cdf(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, x.size + 1, dtype=float) / x.size


def plot_latency_quantiles(summary: pd.DataFrame, output: Path) -> None:
    positions = np.arange(len(summary))
    fig, axis = plt.subplots(figsize=(12.5, 5.8))
    for index, row in summary.iterrows():
        policy = str(row["policy"])
        color = POLICY_COLORS[policy]
        axis.vlines(index, row["median_ttft_s"], row["p99_ttft_s"], color=color, linewidth=3, alpha=0.75)
        axis.scatter(index, row["median_ttft_s"], color=color, marker="o", s=36, zorder=3)
        axis.scatter(index, row["mean_ttft_s"], color=color, marker="D", s=28, zorder=3)
        axis.scatter(index, row["p95_ttft_s"], color=color, marker="^", s=36, zorder=3)
        axis.scatter(index, row["p99_ttft_s"], color=color, marker="v", s=36, zorder=3)
    axis.set_yscale("log")
    axis.set_xticks(positions, [POLICY_SHORT_LABELS[p] for p in summary["policy"]], rotation=35, ha="right")
    axis.set_ylabel("TTFT (seconds, log scale)")
    request_count = int(summary["requests"].min())
    axis.set_title(f"TTFT distribution summary ({request_count:,} matched requests)")
    handles = [
        plt.Line2D([], [], color="#333333", marker=marker, linestyle="", label=label)
        for marker, label in (("o", "median"), ("D", "mean"), ("^", "p95"), ("v", "p99"))
    ]
    axis.legend(handles=handles, ncol=4, loc="upper left")
    save_figure(fig, output, "01_ttft_quantile_overview")


def plot_distribution_families(runs: dict[str, RunData], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True)
    for axis, (group_title, policies) in zip(axes, POLICY_GROUPS, strict=True):
        for policy in policies:
            x, y = empirical_cdf(runs[policy].requests["ttft_s"])
            axis.plot(x, y, color=POLICY_COLORS[policy], linewidth=2, label=POLICY_LABELS[policy])
        axis.set_xscale("log")
        axis.set_title(group_title)
        axis.set_xlabel("TTFT (seconds, log scale)")
        axis.legend(loc="lower right")
    axes[0].set_ylabel("Fraction of requests completed")
    fig.suptitle("TTFT empirical CDF")
    fig.tight_layout()
    save_figure(fig, output, "02_ttft_cdf_by_policy_family")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True)
    for axis, (group_title, policies) in zip(axes, POLICY_GROUPS, strict=True):
        for policy in policies:
            x = np.sort(runs[policy].requests["ttft_s"].to_numpy(dtype=float))
            survival = (x.size - np.arange(x.size, dtype=float)) / x.size
            axis.step(
                x,
                survival,
                where="post",
                color=POLICY_COLORS[policy],
                linewidth=1.8,
                label=POLICY_LABELS[policy],
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(8e-4, 1.05)
        axis.set_title(group_title)
        axis.set_xlabel("TTFT (seconds, log scale)")
        axis.legend(loc="upper right")
    axes[0].set_ylabel("Fraction slower than x (log scale)")
    fig.suptitle("TTFT tail distribution (inverse CDF)")
    fig.tight_layout()
    save_figure(fig, output, "03_ttft_inverse_cdf_by_policy_family")


def _plot_relative_distributions(
    runs: dict[str, RunData],
    output: Path,
    *,
    metric_field: str,
    metric_label: str,
    title: str,
    filename: str,
) -> None:
    baseline = runs["fcfs"].requests.set_index("source_index")[metric_field].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), sharey=True)
    for axis, (group_title, policies) in zip(axes, POLICY_GROUPS, strict=True):
        for policy in policies:
            if policy == "fcfs":
                continue
            latency = runs[policy].requests.set_index("source_index")[metric_field].astype(float)
            x, y = empirical_cdf(latency / baseline)
            axis.plot(x, y, color=POLICY_COLORS[policy], linewidth=2, label=POLICY_LABELS[policy])
        axis.axvline(1.0, color="#111111", linestyle="--", linewidth=1.2, label="equal to FCFS")
        axis.set_xscale("log")
        axis.set_xlim(3e-3, 10.0)
        axis.set_title(group_title)
        axis.set_xlabel(f"{metric_label} ratio (policy ÷ FCFS; <1 faster, >1 slowdown)")
        axis.legend(loc="lower right")
    axes[0].set_ylabel("Fraction of requests")
    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, output, filename)


def plot_relative_distributions(runs: dict[str, RunData], output: Path) -> None:
    _plot_relative_distributions(
        runs,
        output,
        metric_field="ttft_s",
        metric_label="TTFT",
        title="Matched-request TTFT improvement and slowdown relative to FCFS",
        filename="04_fcfs_relative_ttft_cdf",
    )


def plot_e2e_relative_distributions(runs: dict[str, RunData], output: Path) -> None:
    _plot_relative_distributions(
        runs,
        output,
        metric_field="e2e_latency_s",
        metric_label="E2E latency",
        title="Matched-request E2E improvement and slowdown relative to FCFS",
        filename="04b_fcfs_relative_e2e_cdf",
    )


def plot_tradeoff(
    summary: pd.DataFrame,
    output: Path,
    *,
    benefit_field: str = "mean_ttft_improvement_pct",
    harm_field: str = "max_fcfs_ratio",
    filename: str = "05_benefit_fairness_tradeoff",
    points_filename: str = "tradeoff_measured_points.csv",
    title: str = "Benefit-harm tradeoff: measured operating points",
    x_label: str = "Mean TTFT improvement over FCFS (%) →",
    y_label: str = "Worst per-request TTFT / matched FCFS TTFT (lower is safer)",
    is_e2e_metric: bool = False,
) -> None:
    # This figure is about the per-request protection claim. EDF is
    # intentionally omitted: its low-benefit point expands the x-axis without
    # helping distinguish the measured benefit-harm tradeoff.
    candidates = summary[~summary["policy"].isin(["fcfs", "edf_inflation_120"])]
    indexed = candidates.set_index("policy")
    srpt = indexed.loc["srpt"]
    hard_policies = [
        "preflow_hard_inflation_30",
        "preflow_hard_inflation_50",
        "preflow_hard_inflation_80",
        "preflow_hard_inflation_120",
    ]
    prefill_only_policies = [
        "prefill_only_lambda_200",
        "prefill_only_lambda_500",
        "prefill_only_lambda_2000",
    ]
    hard_slack = np.array([30.0, 50.0, 80.0, 120.0])
    prefill_only_lambda = np.array([200.0, 500.0, 2000.0])
    field = harm_field
    srpt_benefit = float(srpt["mean_ttft_improvement_pct"])
    if benefit_field != "mean_ttft_improvement_pct":
        srpt_benefit = float(srpt[benefit_field])
    hard = indexed.loc[hard_policies]
    hard_benefit = hard[benefit_field].to_numpy(dtype=float)
    hard_harm = hard[field].to_numpy(dtype=float)
    prefill_only = indexed.loc[prefill_only_policies]
    prefill_only_benefit = prefill_only[benefit_field].to_numpy(dtype=float)
    prefill_only_harm = prefill_only[field].to_numpy(dtype=float)

    fig, axis = plt.subplots(figsize=(11.5, 6.7))
    axis.plot(
        hard_benefit,
        hard_harm,
        color="#006d2c",
        linewidth=2.6,
        alpha=0.9,
        label="PREFLOW measured settings",
        zorder=1,
    )
    axis.plot(
        prefill_only_benefit,
        prefill_only_harm,
        color="#e6550d",
        linewidth=2.6,
        alpha=0.9,
        label="PrefillOnly measured settings",
        zorder=1,
    )

    point_labels = {
        "sjf": "SJF",
        "srpt": "SRPT",
        "prefill_only_lambda_200": "λ=200",
        "prefill_only_lambda_500": "λ=500",
        "prefill_only_lambda_2000": "λ=2000",
        "preflow_hard_inflation_30": "30%",
        "preflow_hard_inflation_50": "50%",
        "preflow_hard_inflation_80": "80%",
        "preflow_hard_inflation_120": "120%",
    }
    annotation_positions = {
        "sjf": ((0, -25), "center", "top"),
        "srpt": ((-24, -20), "right", "top"),
        "prefill_only_lambda_200": ((-8, 18), "center", "bottom"),
        "prefill_only_lambda_500": ((0, 18), "center", "bottom"),
        "prefill_only_lambda_2000": ((10, 13), "left", "bottom"),
        "preflow_hard_inflation_30": ((-12, 16), "right", "bottom"),
        "preflow_hard_inflation_50": ((-18, 17), "right", "bottom"),
        "preflow_hard_inflation_80": ((18, -20), "left", "top"),
        "preflow_hard_inflation_120": ((18, 12), "left", "bottom"),
    }
    for _, row in candidates.iterrows():
        policy = str(row["policy"])
        axis.scatter(
            row[benefit_field],
            row[field],
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
            s=95,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        offset, horizontal_alignment, vertical_alignment = annotation_positions[policy]
        axis.annotate(
            point_labels[policy],
            (row[benefit_field], row[field]),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            ha=horizontal_alignment,
            va=vertical_alignment,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": POLICY_COLORS[policy],
                "linewidth": 0.8,
                "alpha": 0.75,
                "shrinkA": 2,
                "shrinkB": 5,
            },
            zorder=4,
        )

    axis.axvline(
        srpt_benefit,
        color=POLICY_COLORS["srpt"],
        linestyle=":",
        linewidth=1.8,
        label="Measured SRPT E2E reference" if is_e2e_metric else "Measured SRPT TTFT reference",
    )
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axis.set_xlim(left=20.0, right=srpt_benefit + 4.0)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend(loc="upper left", fontsize=8.5)
    fig.suptitle(title)
    figure_note = (
        "Lines connect measured settings only. SRPT is the highest measured mean-TTFT improvement "
        "and the classical known-size preemptive reference."
        if not is_e2e_metric
        else "Lines connect measured settings only. SRPT is a measured reference, not a theoretical "
        "E2E upper bound in the two-stage PD system."
    )
    fig.text(
        0.5,
        0.01,
        figure_note,
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output, filename)
    measured_points = []
    for family, policies, parameters in (
        ("preflow", hard_policies, hard_slack),
        ("prefill_only", prefill_only_policies, prefill_only_lambda),
    ):
        for policy, parameter in zip(policies, parameters, strict=True):
            row = indexed.loc[policy]
            measured_points.append(
                {
                    "panel_metric": field,
                    "family": family,
                    "policy": policy,
                    "parameter": parameter,
                    "benefit": row[benefit_field],
                    "harm": row[field],
                }
            )
    pd.DataFrame(measured_points).to_csv(output / points_filename, index=False)


def plot_e2e_tradeoff(summary: pd.DataFrame, output: Path) -> None:
    plot_tradeoff(
        summary,
        output,
        benefit_field="mean_e2e_improvement_pct",
        harm_field="max_e2e_fcfs_ratio",
        filename="05c_e2e_benefit_fairness_tradeoff",
        points_filename="e2e_tradeoff_measured_points.csv",
        title="E2E benefit-harm tradeoff: measured operating points",
        x_label="Mean E2E latency improvement over FCFS (%) →",
        y_label="Worst per-request E2E latency / matched FCFS E2E latency (lower is safer)",
        is_e2e_metric=True,
    )


def slowdown_tail(ratios: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    harmful = np.sort(np.asarray(ratios, dtype=float)[np.asarray(ratios, dtype=float) > 1.0])
    survival = np.arange(harmful.size, 0, -1, dtype=float) / len(ratios)
    return harmful, survival


def plot_slowdown_tails(runs: dict[str, RunData], output: Path) -> None:
    baseline = runs["fcfs"].requests.set_index("source_index")["ttft_s"].astype(float)
    soft_policies = [
        "sjf",
        "srpt",
        "prefill_only_lambda_200",
        "prefill_only_lambda_500",
        "prefill_only_lambda_2000",
    ]
    hard_policies = list(HARD_BOUNDS)
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.0), sharex=True, sharey=True)

    for index, policy in enumerate(soft_policies):
        policy_ttft = runs[policy].requests.set_index("source_index")["ttft_s"].astype(float)
        x, y = slowdown_tail((policy_ttft / baseline).to_numpy())
        linestyle = "--" if policy in {"sjf", "srpt"} else "-"
        axes[0].step(
            x,
            y,
            where="post",
            color=POLICY_COLORS[policy],
            linestyle=linestyle,
            linewidth=2.1,
            label=POLICY_LABELS[policy],
        )
        axes[0].scatter(x[-1], y[-1], color=POLICY_COLORS[policy], s=26, zorder=3)
        axes[0].annotate(
            f"{x[-1]:.2f}×",
            (x[-1], y[-1]),
            xytext=(-4, 5 + 4 * (index % 2)),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
            color=POLICY_COLORS[policy],
        )

    hard_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for policy in hard_policies:
        policy_ttft = runs[policy].requests.set_index("source_index")["ttft_s"].astype(float)
        x, y = slowdown_tail((policy_ttft / baseline).to_numpy())
        hard_curves[policy] = (x, y)
        axes[1].step(
            x,
            y,
            where="post",
            color=POLICY_COLORS[policy],
            linewidth=2.3,
            label=POLICY_LABELS[policy],
        )
        axes[1].scatter(x[-1], y[-1], color=POLICY_COLORS[policy], s=30, zorder=3)
        axes[1].axvline(HARD_BOUNDS[policy], color=POLICY_COLORS[policy], linestyle=":", linewidth=1.2, alpha=0.55)

    common_xlim = (0.95, 8.0)
    maximum_hard_ratio = max(curve[0][-1] for curve in hard_curves.values())
    for axis in axes:
        axis.set_xlim(*common_xlim)
        axis.set_yscale("log")
        axis.set_ylim(8e-4, 0.25)
        axis.axvline(1.0, color="#222222", linestyle="--", linewidth=1)
        axis.set_xlabel("Per-request TTFT / matched FCFS TTFT")
    axes[0].set_ylabel("Fraction of all requests slower than x")
    axes[0].set_title("Soft policies: measured tails extend to 7.52×")
    axes[1].set_title("PREFLOW: every measured tail ends by 2.37× (same x-axis)")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="lower right")

    axes[1].axvspan(maximum_hard_ratio, common_xlim[1], color="#e5f5e0", alpha=0.7, zorder=-10)
    axes[1].text(
        5.2,
        0.005,
        f"No PREFLOW request observed\nbeyond {maximum_hard_ratio:.2f}×",
        ha="center",
        va="center",
        fontsize=9,
        color="#236b3a",
    )

    zoom = axes[1].inset_axes([0.42, 0.48, 0.55, 0.46])
    for index, policy in enumerate(hard_policies):
        x, y = hard_curves[policy]
        color = POLICY_COLORS[policy]
        zoom.step(x, y, where="post", color=color, linewidth=1.8)
        zoom.scatter(x[-1], y[-1], color=color, s=24, zorder=3)
        zoom.axvline(HARD_BOUNDS[policy], color=color, linestyle=":", linewidth=1.0, alpha=0.65)
        zoom.annotate(
            f"{x[-1]:.2f}×",
            (x[-1], y[-1]),
            xytext=(0, 5 + 7 * (index % 2)),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=color,
        )
    zoom.set_xlim(0.95, 2.5)
    zoom.set_yscale("log")
    zoom.set_ylim(8e-4, 0.25)
    zoom.axvline(1.0, color="#222222", linestyle="--", linewidth=0.8)
    zoom.set_title("Endpoint zoom (dotted lines are configured bounds)", fontsize=8)
    zoom.set_xticks([1.0, 1.5, 2.0, 2.5])
    zoom.tick_params(axis="both", labelsize=7)
    zoom.grid(True, alpha=0.2)

    fig.suptitle("A shared x-axis separates tapering slowdown tails from PREFLOW's bounded endpoints")
    fig.text(
        0.5,
        0.01,
        "Both main panels use the same 1×-8× scale. Endpoints are observed maxima in this finite trace; "
        "the policy guarantee is established by the feasibility analysis, not by one run.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, output, "05b_fcfs_slowdown_tail_behavior")


def plot_prompt_length(summary: pd.DataFrame, output: Path) -> None:
    policies = ["fcfs", "preflow_hard_inflation_50"]
    labels = ["FCFS", r"PREFLOW (50% slack, $\rho=1.5$)"]
    colors = ["#4c78a8", "#e45756"]
    lengths = sorted(summary["prompt_tokens"].unique())
    positions = np.arange(len(lengths), dtype=float)
    width = 0.36

    fig, axis = plt.subplots(figsize=(10.2, 5.8))
    for index, (policy, label, color) in enumerate(zip(policies, labels, colors, strict=True)):
        rows = summary[summary["policy"] == policy].set_index("prompt_tokens").loc[lengths]
        offset = (index - 0.5) * width
        axis.bar(
            positions + offset,
            rows["mean_ttft_s"],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
    axis.set_xticks(positions, ["4K", "8K", "16K", "32K", "65K", "100K"])
    axis.set_xlabel("Prompt length")
    axis.set_ylabel("Mean wall-clock TTFT (seconds)")
    axis.set_title("Queueing flattens FCFS TTFT across prompt sizes")
    axis.grid(axis="y", alpha=0.25, zorder=0)
    axis.legend(loc="upper left")
    fig.tight_layout()
    save_figure(fig, output, "06_mean_ttft_by_prompt_length")

    matrix = np.asarray(
        [
            [
                float(
                    summary[(summary["policy"] == policy) & (summary["prompt_tokens"] == length)][
                        "ratio_of_mean_ttft"
                    ].iloc[0]
                )
                for length in lengths
            ]
            for policy in POLICY_ORDER
        ]
    )
    fig, axis = plt.subplots(figsize=(11.2, 7.0))
    norm = TwoSlopeNorm(vmin=max(0.0, float(matrix.min())), vcenter=1.0, vmax=max(1.01, float(matrix.max())))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn_r", norm=norm)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}×", ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(lengths)), [f"{value // 1000}k" if value < 100000 else "100k" for value in lengths])
    axis.set_yticks(range(len(POLICY_ORDER)), [POLICY_LABELS[p] for p in POLICY_ORDER])
    axis.set_xlabel("Prompt length")
    axis.set_title("Mean TTFT relative to FCFS within each prompt length (lower is better)")
    fig.colorbar(image, ax=axis, label="Mean TTFT / FCFS mean TTFT")
    fig.tight_layout()
    save_figure(fig, output, "07_prompt_length_fcfs_ratio_heatmap")


def plot_hard_constraints(runs: dict[str, RunData], hard: pd.DataFrame, output: Path) -> None:
    baseline = runs["fcfs"].requests.set_index("source_index")["ttft_s"].astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=True)
    for axis, policy in zip(axes.flat, HARD_BOUNDS, strict=True):
        ttft = runs[policy].requests.set_index("source_index")["ttft_s"].astype(float)
        ratio = np.sort((ttft / baseline).to_numpy())
        percentile = 100.0 * (np.arange(ratio.size) + 1) / ratio.size
        bound = HARD_BOUNDS[policy]
        row = hard[hard["policy"] == policy].iloc[0]
        axis.plot(percentile, ratio, color=POLICY_COLORS[policy], linewidth=2)
        axis.axhline(bound, color="#c51b7d", linestyle="--", linewidth=1.6, label=f"configured bound {bound:.1f}×")
        axis.axhline(1.0, color="#555555", linestyle=":", linewidth=1.1)
        axis.fill_between(percentile, bound, ratio, where=ratio > bound, color="#d7301f", alpha=0.22)
        axis.set_title(
            f"{POLICY_LABELS[policy]}: {int(row['violations'])}/{int(row['requests'])} above bound; "
            f"max {row['max_ratio']:.3f}×"
        )
        axis.set_ylabel("TTFT / matched FCFS TTFT")
        axis.legend(loc="upper left")
    for axis in axes[-1]:
        axis.set_xlabel("Request percentile")
    fig.suptitle("Operational wall-clock check of PREFLOW's FCFS-relative constraint")
    fig.tight_layout()
    save_figure(fig, output, "08_hard_constraint_validation")


def plot_hard_sensitivity(summary: pd.DataFrame, hard: pd.DataFrame, output: Path) -> None:
    merged = hard.merge(summary, on=["policy", "label"], how="left").sort_values("slack_pct")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    axes[0].plot(merged["slack_pct"], merged["mean_ttft_s"], marker="o", linewidth=2, label="mean")
    axes[0].plot(merged["slack_pct"], merged["median_ttft_s"], marker="s", linewidth=2, label="median")
    axes[0].plot(merged["slack_pct"], merged["p95_ttft_s"], marker="^", linewidth=2, label="p95")
    axes[0].plot(merged["slack_pct"], merged["p99_ttft_s"], marker="v", linewidth=2, label="p99")
    axes[0].set_xlabel("Configured FCFS slack (%)")
    axes[0].set_ylabel("TTFT (seconds)")
    axes[0].set_title("Latency benefit as slack increases")
    axes[0].legend()
    axes[1].plot(
        merged["slack_pct"],
        merged["configured_ratio_bound"],
        color="#c51b7d",
        linestyle="--",
        marker="o",
        linewidth=1.8,
        label="configured bound",
    )
    axes[1].plot(merged["slack_pct"], merged["p99_ratio"], marker="s", linewidth=2, label="empirical p99")
    axes[1].plot(merged["slack_pct"], merged["max_ratio"], marker="^", linewidth=2, label="empirical max")
    axes[1].set_xlabel("Configured FCFS slack (%)")
    axes[1].set_ylabel("Per-request TTFT / FCFS TTFT")
    axes[1].set_title("Configured versus observed slowdown")
    axes[1].legend()
    fig.suptitle("PREFLOW slack sensitivity")
    fig.tight_layout()
    save_figure(fig, output, "09_hard_preflow_slack_sensitivity")


def plot_prefill_only_sensitivity(summary: pd.DataFrame, output: Path) -> None:
    rows = summary[summary["policy"].isin(PREFILL_ONLY_LAMBDA)].copy()
    rows["lambda"] = rows["policy"].map(PREFILL_ONLY_LAMBDA)
    rows = rows.sort_values("lambda")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))
    for field, marker, label in (
        ("mean_ttft_s", "o", "mean"),
        ("median_ttft_s", "s", "median"),
        ("p95_ttft_s", "^", "p95"),
        ("p99_ttft_s", "v", "p99"),
    ):
        axes[0].plot(rows["lambda"], rows[field], marker=marker, linewidth=2, label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Aging coefficient λ (log scale)")
    axes[0].set_ylabel("TTFT (seconds)")
    axes[0].set_title("Latency sensitivity")
    axes[0].legend()
    axes[1].plot(rows["lambda"], rows["p99_fcfs_ratio"], marker="s", linewidth=2, label="p99 slowdown")
    axes[1].plot(rows["lambda"], rows["max_fcfs_ratio"], marker="^", linewidth=2, label="worst slowdown")
    axes[1].plot(rows["lambda"], rows["fraction_improved"], marker="o", linewidth=2, label="fraction improved")
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Aging coefficient λ (log scale)")
    axes[1].set_ylabel("Ratio or fraction")
    axes[1].set_title("Fairness changes sharply with tuning")
    axes[1].legend()
    fig.suptitle("PrefillOnly requires workload-specific aging tuning")
    fig.tight_layout()
    save_figure(fig, output, "10_prefillonly_lambda_sensitivity")


def plot_queue_summary(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    positions = np.arange(len(POLICY_ORDER))
    for axis, role, title in (
        (axes[0], "prefill", "Prefill queue"),
        (axes[1], "decode", "Decode queue"),
    ):
        role_rows = summary[summary["role"] == role].set_index("policy").loc[POLICY_ORDER]
        axis.bar(
            positions - 0.18,
            role_rows["total_mean"],
            width=0.36,
            color=[POLICY_COLORS[p] for p in POLICY_ORDER],
            alpha=0.75,
            label="time-weighted mean",
        )
        axis.bar(
            positions + 0.18,
            role_rows["total_p95"],
            width=0.36,
            color=[POLICY_COLORS[p] for p in POLICY_ORDER],
            hatch="//",
            alpha=0.45,
            label="p95",
        )
        for index, policy in enumerate(POLICY_ORDER):
            axis.text(
                index + 0.18,
                role_rows.loc[policy, "total_p95"],
                f"max {role_rows.loc[policy, 'total_max']:.0f}",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize=6.5,
            )
        axis.set_ylabel("Requests")
        axis.set_title(title)
        axis.legend(ncol=2, loc="upper right")
    axes[-1].set_xticks(positions, [POLICY_SHORT_LABELS[p] for p in POLICY_ORDER], rotation=35, ha="right")
    fig.suptitle("Time-weighted queue occupancy")
    fig.tight_layout()
    save_figure(fig, output, "11_prefill_decode_queue_summary")


def plot_queue_small_multiples(runs: dict[str, RunData], output: Path, role: str) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15.5, 11.0), sharex=True, sharey=True)
    for axis, policy in zip(axes.flat, POLICY_ORDER, strict=False):
        frame = runs[policy].queues[role]
        end_s = float(runs[policy].summary["throughput"]["wall_time_s"])
        grid = np.arange(0.0, np.ceil(end_s) + 1.0, 5.0)
        waiting = sample_queue(frame, grid, "waiting")
        running = sample_queue(frame, grid, "running")
        axis.fill_between(grid / 60.0, waiting, color=POLICY_COLORS[policy], alpha=0.55, label="waiting")
        axis.plot(grid / 60.0, running, color="#111111", linewidth=0.7, alpha=0.8, label="running")
        axis.set_title(POLICY_LABELS[policy])
    for axis in axes.flat[len(POLICY_ORDER) :]:
        axis.axis("off")
    for axis in axes[-1]:
        axis.set_xlabel("Elapsed time (minutes)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Requests")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"{role.capitalize()} queue trajectories (5-second sampling)")
    fig.tight_layout()
    save_figure(fig, output, f"12_{role}_queue_trajectories")


def plot_temporal_ttft(runs: dict[str, RunData], output: Path) -> None:
    policies = [
        "fcfs",
        "srpt",
        "prefill_only_lambda_500",
        "preflow_hard_inflation_30",
        "preflow_hard_inflation_50",
        "preflow_hard_inflation_120",
    ]
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    for policy in policies:
        requests = runs[policy].requests.sort_values("scheduled_arrival_time_s")
        rolling = requests["ttft_s"].rolling(51, center=True, min_periods=20)
        x = requests["scheduled_arrival_time_s"] / 60.0
        axes[0].plot(x, rolling.median(), color=POLICY_COLORS[policy], linewidth=1.8, label=POLICY_LABELS[policy])
        axes[1].plot(
            x,
            rolling.quantile(0.95),
            color=POLICY_COLORS[policy],
            linewidth=1.8,
            label=POLICY_LABELS[policy],
        )
    axes[0].set_ylabel("Rolling median TTFT (s)")
    axes[0].set_title("Median response through MMPP load phases")
    axes[1].set_ylabel("Rolling p95 TTFT (s)")
    axes[1].set_xlabel("Scheduled arrival time (minutes)")
    axes[1].set_title("Tail response through MMPP load phases")
    axes[0].legend(ncol=2)
    fig.suptitle("Temporal stability (51-request centered windows)")
    fig.tight_layout()
    save_figure(fig, output, "13_temporal_ttft_under_bursts")


def plot_reordering(runs: dict[str, RunData], output: Path) -> None:
    policies = [
        "fcfs",
        "srpt",
        "prefill_only_lambda_200",
        "prefill_only_lambda_2000",
        "preflow_hard_inflation_30",
        "preflow_hard_inflation_120",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.0), sharex=True, sharey=True)
    scatter = None
    for axis, policy in zip(axes.flat, policies, strict=True):
        requests = runs[policy].requests.sort_values("source_index").reset_index(drop=True).copy()
        requests["arrival_rank"] = np.arange(len(requests))
        requests["completion_rank"] = requests["first_token_time_s"].rank(method="first").astype(int) - 1
        scatter = axis.scatter(
            requests["arrival_rank"],
            requests["completion_rank"],
            c=np.log2(requests["prompt_length"]),
            cmap="viridis",
            s=7,
            alpha=0.65,
            rasterized=True,
        )
        maximum_rank = len(requests) - 1
        axis.plot([0, maximum_rank], [0, maximum_rank], color="#cc0000", linestyle="--", linewidth=1)
        axis.set_title(POLICY_LABELS[policy])
    for axis in axes[-1]:
        axis.set_xlabel("Arrival rank")
    for axis in axes[:, 0]:
        axis.set_ylabel("First-token completion rank")
    assert scatter is not None
    colorbar = fig.colorbar(scatter, ax=axes, shrink=0.8, pad=0.02)
    ticks = np.log2([4096, 8192, 16384, 32768, 65536, 100000])
    colorbar.set_ticks(ticks, labels=["4k", "8k", "16k", "32k", "65k", "100k"])
    colorbar.set_label("Prompt length")
    fig.suptitle("How each policy reorders requests (diagonal is FCFS order)")
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.07, top=0.92, wspace=0.14, hspace=0.22)
    save_figure(fig, output, "14_request_reordering_map")


def plot_decode_and_efficiency(summary: pd.DataFrame, output: Path) -> None:
    positions = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    axes[0].bar(
        positions - 0.18,
        summary["mean_decode_phase_s"],
        0.36,
        color=[POLICY_COLORS[p] for p in summary["policy"]],
        alpha=0.7,
        label="mean",
    )
    axes[0].bar(
        positions + 0.18,
        summary["p99_decode_phase_s"],
        0.36,
        color=[POLICY_COLORS[p] for p in summary["policy"]],
        alpha=0.42,
        hatch="//",
        label="p99",
    )
    axes[0].set_ylabel("E2E latency − TTFT (seconds)")
    axes[0].set_title("Downstream decode-phase latency")
    axes[0].legend()
    axes[1].bar(
        positions,
        summary["completed_requests_s"],
        color=[POLICY_COLORS[p] for p in summary["policy"]],
        alpha=0.75,
    )
    axes[1].set_ylabel("Completed requests/s over run wall time")
    axes[1].set_title("End-to-end throughput remains comparable")
    for axis in axes:
        axis.set_xticks(positions, [POLICY_SHORT_LABELS[p] for p in summary["policy"]], rotation=35, ha="right")
    fig.suptitle("PD-side effects and whole-run efficiency")
    fig.tight_layout()
    save_figure(fig, output, "15_decode_latency_and_throughput")


def write_report(
    output: Path,
    runs: dict[str, RunData],
    summary: pd.DataFrame,
    hard: pd.DataFrame,
    audit: dict[str, Any],
) -> None:
    indexed = summary.set_index("policy")
    hard30 = indexed.loc["preflow_hard_inflation_30"]
    srpt = indexed.loc["srpt"]
    po200 = indexed.loc["prefill_only_lambda_200"]
    po2000 = indexed.loc["prefill_only_lambda_2000"]
    hard120 = indexed.loc["preflow_hard_inflation_120"]
    request_count = len(runs["fcfs"].requests)
    request_filter = audit.get("request_filter")
    if request_filter:
        excluded_indices = sorted(set(request_filter["excluded_source_index_by_policy"].values()))
        filter_description = (
            f"- Request-level metrics exclude the first request from every run "
            f"(source index values: {excluded_indices}); {request_count:,} matched requests remain.\n"
            "- Queue trajectories, run throughput, and process telemetry still describe the original full runs."
        )
    else:
        filter_description = "- No request-level filtering was applied."
    hard_lines = "\n".join(
        f"- {int(row.slack_pct)}% slack: mean TTFT {indexed.loc[row.policy, 'mean_ttft_s']:.1f}s; "
        f"p99 ratio {row.p99_ratio:.3f}×; max ratio {row.max_ratio:.3f}×; "
        f"{int(row.violations)}/{int(row.requests)} observed ratios above {row.configured_ratio_bound:.1f}×."
        for row in hard.itertuples()
    )
    catalog = """1. `01_ttft_quantile_overview`: compact absolute-latency overview.
2. `02_ttft_cdf_by_policy_family`: full TTFT CDFs without placing 11 curves in one panel.
3. `03_ttft_inverse_cdf_by_policy_family`: tail-focused inverse CDFs.
4. `04_fcfs_relative_ttft_cdf` and `04b_fcfs_relative_e2e_cdf`:
   matched-request TTFT and E2E improvements and regressions.
5. `05_benefit_fairness_tradeoff` and `05c_e2e_benefit_fairness_tradeoff`:
   TTFT and E2E benefit versus worst-request harm.
6. `05b_fcfs_slowdown_tail_behavior`: tapering soft-policy tails versus hard protection endpoints.
7. `06_mean_ttft_by_prompt_length`: FCFS and PREFLOW-50 mean wall-clock TTFT by request size.
8. `07_prompt_length_fcfs_ratio_heatmap`: who benefits and who is penalized.
9. `08_hard_constraint_validation`: direct empirical check against each configured bound.
10. `09_hard_preflow_slack_sensitivity`: benefit/constraint trade-off across slack values.
11. `10_prefillonly_lambda_sensitivity`: tuning sensitivity and lack of a hard cap.
12. `11_prefill_decode_queue_summary`: time-weighted mean, p95, and peak occupancy.
13. `12_prefill_queue_trajectories` and `12_decode_queue_trajectories`: per-policy queue timelines.
14. `13_temporal_ttft_under_bursts`: rolling median and p95 through MMPP phases.
15. `14_request_reordering_map`: how policies reorder short and long requests.
16. `15_decode_latency_and_throughput`: downstream decode cost and run throughput.
"""
    report = f"""# Synthetic policy suite analysis

## Data integrity

- All {len(runs)} policies contain {request_count:,} successful requests in this analysis.
- All policies use the same included request IDs, prompt lengths, scheduled arrivals, and workload hash.
- Workload hash: `{runs["fcfs"].workload.get("sha256")}`.
- Arrival process: MMPP at 0.5 requests/s over {runs["fcfs"].workload.get("arrival_span_s")} seconds.
{filter_description}
- This is one seed. The figures therefore contain no confidence intervals and
  should not be presented as statistical replication.

## Main observations

- FCFS mean TTFT is {indexed.loc["fcfs", "mean_ttft_s"]:.1f}s.
- PREFLOW 30% reduces mean TTFT by {hard30.mean_ttft_improvement_pct:.1f}%
  to {hard30.mean_ttft_s:.1f}s, while its worst matched-request wall-clock
  ratio is {hard30.max_fcfs_ratio:.3f}×.
- SRPT reduces mean TTFT by {srpt.mean_ttft_improvement_pct:.1f}% but reaches
  a {srpt.max_fcfs_ratio:.2f}× worst matched-request slowdown. This is the
  clearest unconstrained-efficiency comparison.
- PrefillOnly is highly tuning-sensitive: λ=200 has mean TTFT
  {po200.mean_ttft_s:.1f}s and worst slowdown {po200.max_fcfs_ratio:.2f}×;
  λ=2000 has mean TTFT {po2000.mean_ttft_s:.1f}s and worst slowdown
  {po2000.max_fcfs_ratio:.2f}×.
- The slowdown survival curves make the qualitative distinction clearer than
  aggregate percentiles: SJF, SRPT, and every measured PrefillOnly setting
  taper through isolated 3.63–7.52× regressions. PREFLOW instead has a
  compact endpoint between 1.37× and 2.37× as its slack is increased.
- The same ordering survives through decode. PREFLOW spans
  {hard30.mean_e2e_improvement_pct:.1f}–{hard120.mean_e2e_improvement_pct:.1f}%
  mean E2E improvement with 1.43–2.30× worst-request E2E slowdown. In
  comparison, PrefillOnly λ=200 reaches {po200.mean_e2e_improvement_pct:.1f}%
  mean E2E improvement but 5.38× worst-request slowdown, while measured SRPT
  reaches {srpt.mean_e2e_improvement_pct:.1f}% and 6.95×.

PREFLOW operational checks:

{hard_lines}

The measured ratios slightly exceed every configured bound. These ratios
compare separate real FCFS and policy executions and include runtime noise,
admission effects, asynchronous execution, decode transfer, and model error;
they are not the scheduler's internal modeled completion ratio. Nevertheless,
the preprint should not claim a strict end-to-end wall-clock bound from this
experiment without explaining these violations.

## Recommended preprint figures

1. **Main trade-off:** `05_benefit_fairness_tradeoff`. SRPT minimizes mean
   flow time in the classical known-size preemptive single-server model and has
   the highest measured mean-TTFT improvement in this experiment. It is the
   aggressive efficiency reference, not a formal bound for the real PD system.
   Lines connect measured parameter settings only; the figure contains no fitted
   or extrapolated policy paths.
   The figure deliberately uses worst-request slowdown rather than p99 because
   this is the metric governed by the protection claim. EDF is omitted because
   its low-benefit point compresses the informative region without clarifying
   the comparison.
2. **Core tail distinction:** `05b_fcfs_slowdown_tail_behavior`. This should
   be the main protection figure: SJF, SRPT, and aging have long tapering harm
   tails, while PREFLOW ends sharply near its configured bound.
3. **Guarantee/slack mechanism:** combine `08_hard_constraint_validation` and
   `09_hard_preflow_slack_sensitivity`. Report configured and observed ratios.
4. **Why tuning is insufficient:** `10_prefillonly_lambda_sensitivity`,
   optionally paired with the PrefillOnly panel from
   `04_fcfs_relative_ttft_cdf`.
5. **Workload heterogeneity:** `07_prompt_length_fcfs_ratio_heatmap` or `06_mean_ttft_by_prompt_length`.
6. **Systems appendix:** `11_prefill_decode_queue_summary`, the queue
   trajectories, `15_decode_latency_and_throughput`, and the E2E counterparts
   `04b_fcfs_relative_e2e_cdf` and `05c_e2e_benefit_fairness_tradeoff`.

Avoid using only mean TTFT bars: they conceal the long-request regressions that motivate PREFLOW.

## Figure catalog

{catalog}
Each figure is emitted as a preview PNG and a vector PDF. Machine-readable tables are stored beside the figures.
"""
    (output / "analysis_report.md").write_text(report, encoding="utf-8")
    (output / "integrity_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    default_output_name = "analysis_excluding_first_request" if args.exclude_first_request else "analysis"
    output = (args.output or root / default_output_name).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    runs = discover_runs(root)
    excluded_first = exclude_first_request(runs) if args.exclude_first_request else {}
    audit = validate_runs(runs)
    if excluded_first:
        audit["request_filter"] = {
            "kind": "exclude_first_request",
            "excluded_source_index_by_policy": excluded_first,
        }
    summary = policy_summary(runs)
    hard = hard_constraint_summary(runs)
    prompts = prompt_summary(runs)
    queues = queue_summary(runs)

    summary.to_csv(output / "policy_summary.csv", index=False)
    hard.to_csv(output / "hard_constraint_summary.csv", index=False)
    prompts.to_csv(output / "prompt_length_summary.csv", index=False)
    queues.to_csv(output / "queue_summary.csv", index=False)

    plot_latency_quantiles(summary, output)
    plot_distribution_families(runs, output)
    plot_relative_distributions(runs, output)
    plot_e2e_relative_distributions(runs, output)
    plot_tradeoff(summary, output)
    plot_e2e_tradeoff(summary, output)
    plot_slowdown_tails(runs, output)
    plot_prompt_length(prompts, output)
    plot_hard_constraints(runs, hard, output)
    plot_hard_sensitivity(summary, hard, output)
    plot_prefill_only_sensitivity(summary, output)
    plot_queue_summary(queues, output)
    plot_queue_small_multiples(runs, output, "prefill")
    plot_queue_small_multiples(runs, output, "decode")
    plot_temporal_ttft(runs, output)
    plot_reordering(runs, output)
    plot_decode_and_efficiency(summary, output)
    write_report(output, runs, summary, hard, audit)

    print(f"Rendered analysis to {output}")
    print(summary[["label", "mean_ttft_s", "p99_ttft_s", "max_fcfs_ratio"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
