# SPDX-License-Identifier: Apache-2.0
"""Per-iteration scheduler queue-size tracing for PD disaggregation.

The prefill node uses ``PREFLOWScheduler`` while the decode node normally uses
vLLM's ``Scheduler``. Wrapping the scheduler instance after ``EngineCore``
constructs it instruments both implementations without changing vendored
vLLM sources or duplicating either scheduler's ``schedule()`` method.
"""

import functools
from typing import Any

from vllm.v1.engine.core import EngineCore, EngineCoreProc

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.queue_stats import QueueStatsTracer, create_queue_stats_tracer

_PATCH_INSTALLED_ATTR = "_vllm_ascend_queue_stats_installed"
_TRACER_ATTR = "_vllm_ascend_queue_stats_tracer"


def install_queue_stats_tracer(
    scheduler: Any,
    config: Any,
    vllm_config: Any,
) -> QueueStatsTracer | None:
    """Wrap ``scheduler.schedule`` once and return its tracer when enabled."""
    if type(scheduler).__module__ == "vllm_ascend.core.preflow_scheduler":
        return None
    if not config.enabled or getattr(scheduler, _PATCH_INSTALLED_ATTR, False):
        return getattr(scheduler, _TRACER_ATTR, None)

    tracer = create_queue_stats_tracer(config, vllm_config)
    if tracer is None:
        return None
    original_schedule = scheduler.schedule

    @functools.wraps(original_schedule)
    def schedule_with_queue_stats(*args, **kwargs):
        scheduler_output = original_schedule(*args, **kwargs)
        tracer.record(scheduler)
        return scheduler_output

    scheduler.schedule = schedule_with_queue_stats
    setattr(scheduler, _PATCH_INSTALLED_ATTR, True)
    setattr(scheduler, _TRACER_ATTR, tracer)
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
