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
"""Patches for lookup-table dynamic chunk sizing.

This module patches ``EngineCore`` to run latency-table initialization after
``model_executor`` is ready.  In multiprocessing ``spawn`` mode the child
process starts a fresh interpreter, so ``EngineCoreProc.run_engine_core`` is
also wrapped to re-apply the patch before any ``EngineCore`` instance is
created.
"""

from vllm.logger import logger
from vllm.v1.engine.core import EngineCore, EngineCoreProc

_profiling_patches_applied = False


def _apply_profiling_patches():
    """Patch ``EngineCore.__init__`` to trigger lookup-table initialization."""
    global _profiling_patches_applied
    if _profiling_patches_applied:
        return
    _profiling_patches_applied = True

    original_init = EngineCore.__init__

    def _patched_engine_core_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        if hasattr(self.scheduler, "run_profiling_chunk_init"):
            logger.info("[ProfilingChunk] Running lookup-table initialization...")
            self.scheduler.run_profiling_chunk_init(self.model_executor)

    EngineCore.__init__ = _patched_engine_core_init


_apply_profiling_patches()

_original_run_engine_core = EngineCoreProc.run_engine_core


def _patched_run_engine_core(*args, **kwargs):
    _apply_profiling_patches()
    return _original_run_engine_core(*args, **kwargs)


EngineCoreProc.run_engine_core = _patched_run_engine_core
