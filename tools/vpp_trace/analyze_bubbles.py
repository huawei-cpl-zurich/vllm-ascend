#!/usr/bin/env python3
"""Summarize pipeline bubble time from VPP JSONL traces."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def iter_records(input_dir: Path):
    for path in sorted(input_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSON in {path}:{line_no}: {exc}"
                    ) from exc


def rank_key(record: dict[str, Any]) -> str | None:
    if "pp_rank" not in record:
        return None
    return (
        f"rank={record.get('rank', '?')}:"
        f"dp={record.get('dp_rank', '?')}:"
        f"pp={record.get('pp_rank', '?')}:"
        f"tp={record.get('tp_rank', '?')}:"
        f"pid={record.get('pid', '?')}"
    )


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_total(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def summarize(input_dir: Path) -> dict[str, Any]:
    rank_spans: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scheduler_batches = 0

    for record in iter_records(input_dir):
        if record.get("name") == "scheduler_batch":
            scheduler_batches += 1
        if record.get("type") != "span":
            continue
        key = rank_key(record)
        if key is not None:
            rank_spans[key].append(record)

    ranks: dict[str, Any] = {}
    for key, spans in sorted(rank_spans.items()):
        intervals = [
            (
                int(span.get("ts_ns", 0)),
                int(span.get("ts_ns", 0)) + int(span.get("dur_ns", 0)),
            )
            for span in spans
            if int(span.get("dur_ns", 0)) > 0
        ]
        merged = merge_intervals(intervals)
        if not merged:
            continue

        active_start = merged[0][0]
        active_end = merged[-1][1]
        active_ns = active_end - active_start
        busy_ns = interval_total(merged)
        idle_ns = max(active_ns - busy_ns, 0)

        by_name: dict[str, int] = defaultdict(int)
        by_cat: dict[str, int] = defaultdict(int)
        for span in spans:
            dur = int(span.get("dur_ns", 0))
            by_name[str(span.get("name", "unknown"))] += dur
            by_cat[str(span.get("cat", "unknown"))] += dur

        useful_forward_ns = by_name.get("model_forward_host", 0)
        if useful_forward_ns == 0:
            useful_forward_ns = by_name.get("forward_region", 0)

        ranks[key] = {
            "active_window_ns": active_ns,
            "busy_traced_ns": busy_ns,
            "idle_gap_ns": idle_ns,
            "bubble_pct": (idle_ns / active_ns * 100.0) if active_ns else 0.0,
            "useful_forward_host_ns": useful_forward_ns,
            "device_forward_ns": by_name.get("model_forward_device", 0),
            "pp_recv_wait_ns": by_name.get("pp_recv_wait", 0),
            "pp_prev_send_wait_ns": by_name.get("pp_prev_send_wait", 0),
            "future_wait_ns": by_name.get("future_wait", 0),
            "by_name_ns": dict(sorted(by_name.items())),
            "by_cat_ns": dict(sorted(by_cat.items())),
        }

    return {
        "input_dir": str(input_dir),
        "scheduler_batches": scheduler_batches,
        "num_rank_lanes": len(ranks),
        "ranks": ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summarize(args.input_dir), f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
