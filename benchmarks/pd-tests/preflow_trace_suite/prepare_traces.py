#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create compact, lossless-for-this-benchmark trace files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = Path("/home/m84434548/preflow_traces")

SOURCES = {
    "mooncake_conversation": "mooncake/conversation_trace.jsonl",
    "mooncake_arxiv": "mooncake/mooncake_arxiv_trace.jsonl",
    "mooncake_synthetic": "mooncake/synthetic_trace.jsonl",
    "mooncake_toolagent": "mooncake/toolagent_trace.jsonl",
    "qwen_coder": "qwen-bailian/qwen_coder_blksz_16.jsonl",
    "qwen_thinking": "qwen-bailian/qwen_thinking_blksz_16.jsonl",
    "qwen_trace_a": "qwen-bailian/qwen_traceA_blksz_16.jsonl",
    "qwen_trace_b": "qwen-bailian/qwen_traceB_blksz_16.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def compact_trace(source: Path, destination: Path) -> dict[str, Any]:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    count = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    minimum_input: int | None = None
    maximum_input: int | None = None
    with source.open(encoding="utf-8") as input_handle, temporary.open("w", encoding="utf-8") as output_handle:
        for line_number, line in enumerate(input_handle, 1):
            try:
                source_row = json.loads(line)
                timestamp = float(source_row["timestamp"])
                input_length = int(source_row["input_length"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{source}:{line_number}: invalid trace row") from error
            if input_length <= 0:
                raise ValueError(f"{source}:{line_number}: input_length must be positive")
            if last_timestamp is not None and timestamp < last_timestamp:
                raise ValueError(f"{source}:{line_number}: timestamps are not nondecreasing")
            output_handle.write(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "input_length": input_length,
                        "source_index": count,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            first_timestamp = timestamp if first_timestamp is None else first_timestamp
            last_timestamp = timestamp
            minimum_input = input_length if minimum_input is None else min(minimum_input, input_length)
            maximum_input = input_length if maximum_input is None else max(maximum_input, input_length)
            count += 1
    temporary.replace(destination)
    return {
        "rows": count,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "minimum_input_length": minimum_input,
        "maximum_input_length": maximum_input,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=("prepare or update one named trace; PATH may be absolute or relative to --source-root (repeatable)"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_trace_specs(specifications: list[str]) -> dict[str, str]:
    if not specifications:
        return dict(SOURCES)
    selected: dict[str, str] = {}
    name_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    for specification in specifications:
        name, separator, path = specification.partition("=")
        if not separator or not name_pattern.fullmatch(name) or not path:
            raise ValueError(f"invalid --trace {specification!r}; expected NAME=PATH")
        if name in selected:
            raise ValueError(f"duplicate --trace name: {name}")
        selected[name] = path
    return selected


def new_manifest(source_root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "description": (
            "Derived traces retain timestamp, input_length, and original row index. "
            "Output length and prefix fields are intentionally omitted because this "
            "suite disables prefix caching and always requests one output token."
        ),
        "source_root_at_generation": str(source_root),
        "traces": {},
    }


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    trace_dir = HERE / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = trace_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("traces"), dict):
            raise ValueError(f"invalid existing manifest: {manifest_path}")
    else:
        manifest = new_manifest(source_root)

    for name, configured_path in parse_trace_specs(args.trace).items():
        requested_path = Path(configured_path).expanduser()
        source = requested_path if requested_path.is_absolute() else source_root / requested_path
        source = source.resolve()
        destination = trace_dir / f"{name}.jsonl"
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists() and not args.force:
            raise FileExistsError(f"{destination} already exists; pass --force to regenerate")
        statistics = compact_trace(source, destination)
        try:
            source_description = str(source.relative_to(source_root))
        except ValueError:
            source_description = str(source)
        manifest["traces"][name] = {
            "source_relative_path": source_description,
            "source_sha256": sha256(source),
            "derived_relative_path": str(destination.relative_to(HERE)),
            "derived_sha256": sha256(destination),
            **statistics,
        }
        print(f"prepared {name}: {statistics['rows']} rows")
    atomic_write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
