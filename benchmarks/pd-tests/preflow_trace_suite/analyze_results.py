#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate completed trace-suite results and pair them with FCFS."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values if math.isfinite(float(value))]
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def flatten_summary(policy: str, trace: str, summary: dict[str, Any]) -> dict[str, Any]:
    ttft = summary.get("ttft_s", {})
    lag = summary.get("client_dispatch_lag_s", {})
    workload = summary.get("workload", {})
    return {
        "policy": policy,
        "trace": trace,
        "target_load": summary.get("target_load"),
        "requests": summary.get("requests"),
        "successful_requests": summary.get("successful_requests"),
        "predicted_service_s": workload.get("predicted_service_s"),
        "scheduled_span_s": workload.get("scheduled_span_s"),
        "service_budget_reached": workload.get("service_budget_reached"),
        "wall_time_s": summary.get("wall_time_s"),
        "mean_ttft_s": ttft.get("mean"),
        "median_ttft_s": ttft.get("median"),
        "p90_ttft_s": ttft.get("p90"),
        "p95_ttft_s": ttft.get("p95"),
        "p99_ttft_s": ttft.get("p99"),
        "max_ttft_s": ttft.get("max"),
        "p95_client_dispatch_lag_s": lag.get("p95"),
        "p99_client_dispatch_lag_s": lag.get("p99"),
        "max_client_dispatch_lag_s": lag.get("max"),
    }


def discover(output_root: Path) -> dict[tuple[str, str], Path]:
    results = {}
    policies_dir = output_root / "policies"
    if not policies_dir.is_dir():
        return results
    active_conditions: set[tuple[str, str]] | None = None
    suite_manifest_path = output_root / "suite_manifest.json"
    if suite_manifest_path.is_file():
        suite_manifest = load_json(suite_manifest_path)
        active_conditions = {
            (str(policy["name"]), str(trace))
            for policy in suite_manifest.get("policies", [])
            for trace in policy.get("traces", [])
        }
    for status_path in policies_dir.glob("*/runs/*/status.json"):
        status = load_json(status_path)
        if status.get("status") != "completed":
            continue
        policy = status_path.parents[2].name
        trace = status_path.parent.name
        if active_conditions is not None and (policy, trace) not in active_conditions:
            continue
        result_path = Path(str(status["result_path"]))
        if (result_path / "summary.json").is_file() and (result_path / "requests.jsonl").is_file():
            results[(policy, trace)] = result_path
    return results


def analyze(output_root: Path) -> None:
    results = discover(output_root)
    aggregate_rows = []
    request_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (policy, trace), result_path in sorted(results.items()):
        summary = load_json(result_path / "summary.json")
        aggregate_rows.append(flatten_summary(policy, trace, summary))
        request_rows[(policy, trace)] = load_jsonl(result_path / "requests.jsonl")
    write_csv(output_root / "aggregate_results.csv", aggregate_rows)

    fcfs_by_trace: dict[str, dict[str, float]] = {}
    for (policy, trace), rows in request_rows.items():
        if policy != "fcfs":
            continue
        fcfs_by_trace[trace] = {
            str(row["request_id"]): float(row["ttft_s"])
            for row in rows
            if row.get("success") and row.get("ttft_s") is not None
        }

    paired_rows = []
    paired_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (policy, trace), rows in sorted(request_rows.items()):
        baseline = fcfs_by_trace.get(trace)
        if policy == "fcfs" or baseline is None:
            continue
        for row in rows:
            request_id = str(row["request_id"])
            ttft_value = row.get("ttft_s")
            fcfs_ttft = baseline.get(request_id)
            if not row.get("success") or ttft_value is None or fcfs_ttft is None or fcfs_ttft <= 0:
                continue
            ttft = float(ttft_value)
            inflation = ttft / fcfs_ttft - 1.0
            paired_groups[(policy, trace)].append(inflation)
            paired_rows.append(
                {
                    "policy": policy,
                    "trace": trace,
                    "request_id": request_id,
                    "source_index": row["source_index"],
                    "prompt_tokens": row["prompt_tokens"],
                    "ttft_s": ttft,
                    "fcfs_ttft_s": fcfs_ttft,
                    "empirical_fcfs_inflation": inflation,
                }
            )
    write_csv(output_root / "fcfs_relative_requests.csv", paired_rows)

    comparison_rows = []
    for (policy, trace), values in sorted(paired_groups.items()):
        metrics = distribution(values)
        comparison_rows.append(
            {
                "policy": policy,
                "trace": trace,
                "paired_requests": metrics["count"],
                "mean_empirical_fcfs_inflation": metrics["mean"],
                "median_empirical_fcfs_inflation": metrics["median"],
                "p90_empirical_fcfs_inflation": metrics["p90"],
                "p95_empirical_fcfs_inflation": metrics["p95"],
                "p99_empirical_fcfs_inflation": metrics["p99"],
                "max_empirical_fcfs_inflation": metrics["max"],
            }
        )
    write_csv(output_root / "fcfs_relative_summary.csv", comparison_rows)
    print(
        f"Aggregated {len(results)} completed conditions; wrote {len(paired_rows)} paired FCFS-relative request rows."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.output_root.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
