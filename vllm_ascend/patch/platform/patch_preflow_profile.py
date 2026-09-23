# SPDX-License-Identifier: Apache-2.0
"""Run optional PREFLOW cost calibration during EngineCore startup."""

import functools

from vllm.v1.engine.core import EngineCore, EngineCoreProc

_preflow_profile_patch_applied = False


def _apply_preflow_profile_patch() -> None:
    global _preflow_profile_patch_applied
    if _preflow_profile_patch_applied:
        return
    _preflow_profile_patch_applied = True

    original_init = EngineCore.__init__

    @functools.wraps(original_init)
    def patched_engine_core_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        run_profile = getattr(
            self.scheduler,
            "run_preflow_startup_profile",
            None,
        )
        if run_profile is not None:
            run_profile(self)

    EngineCore.__init__ = patched_engine_core_init


_apply_preflow_profile_patch()

_original_run_engine_core = EngineCoreProc.run_engine_core


def _patched_run_engine_core(*args, **kwargs):
    _apply_preflow_profile_patch()
    return _original_run_engine_core(*args, **kwargs)


EngineCoreProc.run_engine_core = _patched_run_engine_core
