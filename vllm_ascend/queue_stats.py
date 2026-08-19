# SPDX-License-Identifier: Apache-2.0
"""Shared scheduler queue-size tracing for PD benchmark diagnostics."""

import atexit
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from vllm.logger import logger

_CSV_HEADER = "timestamp_ns,iteration,role,waiting,running,total\n"


def get_pd_role(vllm_config: Any) -> str:
    """Return a stable role label for a PD worker."""
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    kv_role = getattr(kv_transfer_config, "kv_role", None)
    return {
        "kv_producer": "prefill",
        "kv_consumer": "decode",
        "kv_both": "mixed",
    }.get(kv_role, "unknown")


class QueueStatsTracer:
    """Write non-idle scheduler queue sizes to an engine-core-specific CSV."""

    def __init__(self, output_dir: str, vllm_config: Any) -> None:
        trace_dir = Path(output_dir).expanduser()
        trace_dir.mkdir(parents=True, exist_ok=True)

        self.role = get_pd_role(vllm_config)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        dp_rank = getattr(parallel_config, "data_parallel_rank", 0)
        self.path = trace_dir / (
            f"vllm_ascend_queue_stats_{self.role}_dp{dp_rank}_pid{os.getpid()}_{uuid.uuid4().hex}.csv"
        )
        self._file = self.path.open("x", encoding="utf-8", buffering=1)
        self._file.write(_CSV_HEADER)
        self._lock = threading.Lock()
        atexit.register(self.close)

    def record(self, scheduler: Any) -> None:
        """Append one post-schedule sample when the worker has active work."""
        waiting = len(scheduler.waiting) + len(scheduler.skipped_waiting)
        running = len(scheduler.running)
        if waiting + running == 0:
            return

        iteration = getattr(scheduler, "current_step", 0)
        with self._lock:
            if self._file.closed:
                return
            try:
                self._file.write(f"{time.time_ns()},{iteration},{self.role},{waiting},{running},{waiting + running}\n")
            except OSError:
                logger.exception(
                    "Queue-size tracing disabled because the trace file cannot be written: %s",
                    self.path,
                )
                self._file.close()

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()


def create_queue_stats_tracer(config: Any, vllm_config: Any) -> QueueStatsTracer | None:
    """Create an enabled queue tracer, keeping tracing failures non-fatal."""
    if not config.enabled:
        return None
    try:
        tracer = QueueStatsTracer(config.output_dir, vllm_config)
    except OSError:
        logger.exception("Unable to create scheduler queue-size trace in %s", config.output_dir)
        return None
    logger.info("Writing scheduler queue-size trace to %s", tracer.path)
    return tracer
