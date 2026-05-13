#!/usr/bin/env python3
"""Convert VPP JSONL traces to Chrome/Perfetto trace JSON."""

from __future__ import annotations

import argparse
import json
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


def lane_name(record: dict[str, Any]) -> str:
    if "pp_rank" in record:
        return (
            f"rank={record.get('rank', '?')} "
            f"dp={record.get('dp_rank', '?')} "
            f"pp={record.get('pp_rank', '?')} "
            f"tp={record.get('tp_rank', '?')}"
        )
    if record.get("cat") in {"engine", "scheduler"}:
        return f"engine pid={record.get('pid', '?')}"
    return f"pid={record.get('pid', '?')}"


def event_args(record: dict[str, Any]) -> dict[str, Any]:
    args = dict(record.get("args") or {})
    for key in (
        "run_id",
        "rank",
        "local_rank",
        "dp_rank",
        "pp_rank",
        "tp_rank",
        "device_id",
        "trace_batch_id",
        "engine_step",
    ):
        if key in record:
            args[key] = record[key]
    return args


def convert(input_dir: Path) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    named_threads: set[tuple[int, int, str]] = set()

    for record in iter_records(input_dir):
        pid = int(record.get("pid", 0))
        tid = 0
        name = lane_name(record)
        if (pid, tid, name) not in named_threads:
            named_threads.add((pid, tid, name))
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": name},
                }
            )
            trace_events.append(
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": name},
                }
            )

        ts_us = int(record.get("ts_ns", 0)) / 1000.0
        common = {
            "name": record.get("name", "unknown"),
            "cat": record.get("cat", "vpp"),
            "pid": pid,
            "tid": tid,
            "ts": ts_us,
            "args": event_args(record),
        }
        if record.get("type") == "span":
            trace_events.append(
                {
                    **common,
                    "ph": "X",
                    "dur": int(record.get("dur_ns", 0)) / 1000.0,
                }
            )
        else:
            trace_events.append({**common, "ph": "i", "s": "p"})

    trace_events.sort(key=lambda event: event.get("ts", 0))
    return {"traceEvents": trace_events}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(convert(args.input_dir), f)


if __name__ == "__main__":
    main()
