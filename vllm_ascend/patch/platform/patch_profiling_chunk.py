#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Patches for profiling-based dynamic chunk sizing.

This module patches ``EngineCore`` to:
1. Run startup profiling (after model_executor is ready), loading cached
   profiling data first when available.

In multiprocessing ``spawn`` mode the child process starts a fresh Python
interpreter, so class-level monkey-patches applied in the parent are lost.
To handle this we additionally wrap ``EngineCoreProc.run_engine_core``
(the subprocess entry-point): when pickle resolves the wrapper it triggers
an import of this module, which re-applies the ``EngineCore.__init__``
patches inside the child process before any ``EngineCore`` is instantiated.
"""

from vllm.logger import logger
from vllm.v1.engine.core import EngineCore, EngineCoreProc

_profiling_patches_applied = False


# ---------------------------------------------------------------------------
# Core: apply EngineCore.__init__ patches (idempotent)
# ---------------------------------------------------------------------------


def _apply_profiling_patches():
    """Patch ``EngineCore.__init__`` to trigger startup profiling.

    Safe to call multiple times; the guard ``_profiling_patches_applied``
    ensures the patch is applied at most once per process.
    """
    global _profiling_patches_applied
    if _profiling_patches_applied:
        return
    _profiling_patches_applied = True

    original_init = EngineCore.__init__

    def _patched_engine_core_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if hasattr(self.scheduler, "run_profiling_chunk_init"):
            logger.info("[ProfilingChunk] Running profiling initialization...")
            self.scheduler.run_profiling_chunk_init(self.model_executor)

    EngineCore.__init__ = _patched_engine_core_init


# ---------------------------------------------------------------------------
# 1. Apply patches at module level for the InprocClient (in-process) path.
# ---------------------------------------------------------------------------
_apply_profiling_patches()

# ---------------------------------------------------------------------------
# 2. Wrap EngineCoreProc.run_engine_core so that spawned subprocesses
#    re-apply the patches.  When the child unpickles this wrapper it
#    imports this module, which triggers _apply_profiling_patches() above,
#    ensuring EngineCore.__init__ is patched before any instance is created.
# ---------------------------------------------------------------------------
_original_run_engine_core = EngineCoreProc.run_engine_core


def _patched_run_engine_core(*args, **kwargs):
    _apply_profiling_patches()
    return _original_run_engine_core(*args, **kwargs)


EngineCoreProc.run_engine_core = _patched_run_engine_core
