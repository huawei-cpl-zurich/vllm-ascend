# SPDX-License-Identifier: Apache-2.0
"""Per-iteration scheduler queue-size tracing for PD disaggregation.

The prefill node uses ``PREFLOWScheduler`` while the decode node normally uses
vLLM's ``Scheduler``. Wrapping the scheduler instance after ``EngineCore``
constructs it instruments both implementations without changing vendored
vLLM sources or duplicating either scheduler's ``schedule()`` method.
"""

import atexit
import functools
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from vllm.logger import logger
from vllm.v1.engine.core import EngineCore, EngineCoreProc

from vllm_ascend.ascend_config import QueueStatsConfig, init_ascend_config

_PATCH_INSTALLED_ATTR = "_vllm_ascend_queue_stats_installed"
_TRACER_ATTR = "_vllm_ascend_queue_stats_tracer"


def _get_pd_role(vllm_config: Any) -> str:
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    kv_role = getattr(kv_transfer_config, "kv_role", None)
    return {
        "kv_producer": "prefill",
        "kv_consumer": "decode",
        "kv_both": "mixed",
    }.get(kv_role, "unknown")


class QueueStatsTracer:
    """Write scheduler queue sizes to an engine-core-specific CSV file."""

    def __init__(self, config: QueueStatsConfig, vllm_config: Any) -> None:
        output_dir = Path(config.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        role = _get_pd_role(vllm_config)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        dp_rank = getattr(parallel_config, "data_parallel_rank", 0)
        file_name = f"vllm_ascend_queue_stats_{role}_dp{dp_rank}_pid{os.getpid()}_{uuid.uuid4().hex}.csv"
        self.path = output_dir / file_name
        self._file = self.path.open("x", encoding="utf-8", buffering=1)
        self._file.write("timestamp_ns,iteration,role,waiting,running,total\n")
        self._role = role
        self._lock = threading.Lock()
        atexit.register(self.close)

    def record(self, scheduler: Any) -> None:
        """Append the state after one successful scheduler iteration."""
        with self._lock:
            if self._file.closed:
                return
            try:
                waiting = len(scheduler.waiting) + len(scheduler.skipped_waiting)
                running = len(scheduler.running)
                iteration = getattr(scheduler, "current_step", 0)
                self._file.write(f"{time.time_ns()},{iteration},{self._role},{waiting},{running},{waiting + running}\n")
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


def install_queue_stats_tracer(
    scheduler: Any,
    config: QueueStatsConfig,
    vllm_config: Any,
) -> QueueStatsTracer | None:
    """Wrap ``scheduler.schedule`` once and return its tracer when enabled."""
    if not config.enabled or getattr(scheduler, _PATCH_INSTALLED_ATTR, False):
        return getattr(scheduler, _TRACER_ATTR, None)

    tracer = QueueStatsTracer(config, vllm_config)
    original_schedule = scheduler.schedule

    @functools.wraps(original_schedule)
    def schedule_with_queue_stats(*args, **kwargs):
        scheduler_output = original_schedule(*args, **kwargs)
        tracer.record(scheduler)
        return scheduler_output

    scheduler.schedule = schedule_with_queue_stats
    setattr(scheduler, _PATCH_INSTALLED_ATTR, True)
    setattr(scheduler, _TRACER_ATTR, tracer)
    logger.info("Writing scheduler queue-size trace to %s", tracer.path)
    return tracer


_queue_stats_patches_applied = False


def _apply_queue_stats_patches() -> None:
    """Install the EngineCore hook once per process."""
    global _queue_stats_patches_applied
    if _queue_stats_patches_applied:
        return
    _queue_stats_patches_applied = True

    original_init = EngineCore.__init__

    @functools.wraps(original_init)
    def patched_engine_core_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        config = init_ascend_config(self.vllm_config).scheduler_config.queue_stats_config
        install_queue_stats_tracer(self.scheduler, config, self.vllm_config)

    EngineCore.__init__ = patched_engine_core_init


_apply_queue_stats_patches()

# Spawned engine-core processes import this entry point directly. Re-apply the
# EngineCore patch there because module-level monkey-patches are process-local.
_original_run_engine_core = EngineCoreProc.run_engine_core


def _patched_run_engine_core(*args, **kwargs):
    _apply_queue_stats_patches()
    return _original_run_engine_core(*args, **kwargs)


EngineCoreProc.run_engine_core = _patched_run_engine_core
