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
"""
Profiling-based Dynamic Chunk Size Predictor.

This module implements a dynamic chunk sizing strategy based on cached startup
profiling data and fitting a chunked-prefill latency surface.

The approach:
1. Load: Reuse cached raw samples for the current model/runtime when available
2. Profile: On cache miss, run repeated forward passes over (chunk, history) points
3. Fit: Use f(C,H) = a·C(C+H) + b·C + d·H + c to fit median latency
4. Predict: Given current num_computed_tokens H, solve for chunk size x
   that achieves target latency T using the correct incremental cost model:

       T_incr(x, H) = a·x·(H+x) + b·x + d·H + c

   This is NOT f(H+x) − f(H) — the latter double-counts the attention
   cross-term between cached and new tokens (see ChunkSizePredictor).
"""

import hashlib
import json
import math
import os
import platform
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from vllm.logger import logger

PROFILE_CACHE_SCHEMA_VERSION = 1
PROFILE_FIT_VERSION = "chunk_surface_v1"
VALID_CACHE_MODES = {"auto", "refresh", "readonly", "off"}


def _distribution_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    return str(value)


def _get_nested_attr(obj: Any, *names: str) -> Any:
    current = obj
    for name in names:
        if current is None:
            return None
        current = getattr(current, name, None)
    return _json_safe(current)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return float((sorted_values[mid - 1] + sorted_values[mid]) / 2.0)


class ChunkSizePredictor:
    """Predictor for dynamic chunk size based on profiled latency models.

    Startup cache data fits the history-aware surface:

        f(C,H) = a·C(C+H) + b·C + d·H + c

    The incremental latency of processing x new tokens when H tokens are
    already in KVCache is:

        T_incr(x, H) = a·x·(H+x) + b·x + d·H + c

    Only the x new query tokens attend to the H+x KV positions — the
    H cached tokens do not recompute attention.  This is fundamentally
    different from the naïve f(H+x) − f(H) subtraction, which would give
    2aHx + ax² + bx (overcounting the cross-term by a factor of 2).

    Given a target latency T and current history length H, predicts next
    chunk size x such that:

        a·x² + (aH + b)·x + dH + c − T = 0
    """

    def __init__(self, smooth_factor: float = 0.8, min_chunk: int = 4096):
        self.quadratic_coeff_a: float = 0.0
        self.linear_coeff_b: float = 0.0
        self.constant_coeff_c: float = 0.0

        self.quadratic_chunk_a: float = 0.0
        self.linear_chunk_b: float = 0.0
        self.history_chunk_d: float = 0.0
        self.constant_chunk_c: float = 0.0

        self.target_latency: float | None = None
        self.is_ready: bool = False
        self.with_history_ready: bool = False
        self.smooth_factor = smooth_factor
        self.min_chunk = min_chunk
        self.history_fitted = False

    def clamp_quadratic_and_linear_if_negative(self, fitted_a: float, fitted_b: float) -> tuple[float, float]:
        """In theory, for the Transfomur structure of LLM, the fitted quadratic and linear
        terms should not be negative. Can perform zero clamping for inaccurate fitting
        """
        if fitted_a < 0:
            logger.warning("Fitted a=%.2e is not positive. Setting a=1e-9.", fitted_a)
            fitted_a = 1e-9
        if fitted_b < 0:
            logger.warning("Fitted b=%.2e is not positive. Setting b=0.0.", fitted_b)
            fitted_b = 1e-9

        return fitted_a, fitted_b

    def fit(self, seq_lens: list[int], latencies: list[float]) -> bool:
        """Fit quadratic coefficients f(l) = al^2 + bl + c from data points.

        Returns:
            True if fitting succeeded, False otherwise
        """
        L = np.array(seq_lens, dtype=np.float64)
        T = np.array(latencies, dtype=np.float64)
        MIN_FIT_POINTS_NO_CHUNK = 8

        if len(L) < MIN_FIT_POINTS_NO_CHUNK:
            logger.warning(
                "Not enough data points for quadratic fitting (%d < 8)",
                len(L),
            )
            return False

        X = np.column_stack([L * L, L, np.ones_like(L)])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, T, rcond=None)
            fitted_a = float(coeffs[0])
            fitted_b = float(coeffs[1])
            fitted_c = float(coeffs[2])
        except Exception as e:
            # Keep a robust fallback for environments where least-squares may
            # fail due backend/LAPACK differences.
            try:
                poly = np.polyfit(L, T, 2)
                fitted_a = float(poly[0])
                fitted_b = float(poly[1])
                fitted_c = float(poly[2])
                logger.warning(
                    "Least-squares fitting failed (%s), fallback to polyfit succeeded.",
                    e,
                )
            except Exception as fallback_error:
                logger.warning("Failed to fit quadratic model: %s", fallback_error)
                return False

        fitted_a, fitted_b = self.clamp_quadratic_and_linear_if_negative(fitted_a, fitted_b)

        self.quadratic_coeff_a = fitted_a
        self.linear_coeff_b = fitted_b
        self.constant_coeff_c = fitted_c

        logger.info(
            "[ProfilingChunk] Fitted: a=%.2e, b=%.2e, c=%.2e",
            fitted_a,
            fitted_b,
            fitted_c,
        )
        return True

    def fit_chunk(self, chunked_data: list) -> bool:
        """Fit chunked prefill latency.

        Preferred startup-profile model:

            f(C,H) = a*C(C+H) + b*C + d*H + c

        where C is the scheduled chunk size and H is cached history length.

        Returns:
            True if fitting succeeded, False otherwise
        """
        num_points = len(chunked_data)
        MIN_FIT_POINTS_CHUNK = 5
        if num_points < MIN_FIT_POINTS_CHUNK:
            logger.warning(
                "Not enough data points for chunked data fitting (%d < 5)",
                num_points,
            )
            return False

        first = chunked_data[0]
        legacy_feature_matrix = not isinstance(first, dict)
        if legacy_feature_matrix:
            chunked_data_array = np.array(chunked_data, dtype=np.float64)
            execute_time = chunked_data_array[:, -1]
            input_x = chunked_data_array[:, :-1]
        else:
            rows = []
            execute_time_list = []
            for sample in chunked_data:
                chunk_size = int(sample["chunk_size"])
                history_len = int(sample["history_len"])
                latency_ms = float(sample.get("median_ms", sample.get("latency_ms")))
                rows.append([
                    chunk_size * (chunk_size + history_len),
                    chunk_size,
                    history_len,
                    1.0,
                ])
                execute_time_list.append(latency_ms)
            input_x = np.array(rows, dtype=np.float64)
            execute_time = np.array(execute_time_list, dtype=np.float64)

        try:
            params, _, _, _ = np.linalg.lstsq(input_x, execute_time, rcond=None)
            fitted_a = float(params[0])
            fitted_b = float(params[1])
            if legacy_feature_matrix:
                fitted_d = 0.0
                fitted_c = float(params[2])
            else:
                fitted_d = float(params[2])
                fitted_c = float(params[3])
        except np.linalg.LinAlgError as e:
            logger.warning("Failed to fit chunked model: %s", e)
            return False

        fitted_a, fitted_b = self.clamp_quadratic_and_linear_if_negative(fitted_a, fitted_b)
        if fitted_d < 0:
            logger.warning("Fitted d=%.2e is not positive. Setting d=0.0.", fitted_d)
            fitted_d = 0.0

        self.quadratic_chunk_a = fitted_a
        self.linear_chunk_b = fitted_b
        self.history_chunk_d = fitted_d
        self.constant_chunk_c = fitted_c
        self.history_fitted = True

        logger.info(
            "[ProfilingChunk With History] Fitted: a=%.2e, b=%.2e, d=%.2e, c=%.2e",
            fitted_a,
            fitted_b,
            fitted_d,
            fitted_c,
        )
        return True

    def set_target_latency(self, base_chunk_size: int, elapsed_time: float = 0.0) -> None:
        """Set target latency based on base chunk size.

        Uses the corrected incremental model: T_target = a·base·(base+0) + b·base + c
        which simplifies to f(base_chunk_size) — the full cost of a forward pass
        processing base_chunk_size tokens with no KVCache.
        """
        if elapsed_time > 0:
            self.target_latency = elapsed_time
        else:
            self.target_latency = self.quadratic_coeff_a * base_chunk_size * base_chunk_size \
                                + self.linear_coeff_b * base_chunk_size \
                                + self.constant_coeff_c
        if self.target_latency <= 0:
            self.target_latency = 1.0

        logger.info(
            "[ProfilingChunk] Target latency: %.2f ms (base_chunk=%d)",
            self.target_latency,
            base_chunk_size,
        )

    def get_time(
        self,
        query_len: int,
        num_computed_tokens: int,
    ) -> float:
        """Get predicted incremental latency for processing x new tokens with H cached.

        Correct model: T_incr(x, H) = a·x·(H+x) + b·x + c

        Only the x new tokens compute attention (attending to H+x KV positions),
        unlike full prefill where all H+x tokens attend to all H+x tokens.
        Using f(H+x) - f(H) = 2aHx + ax² + bx overcounts the cross-term by aHx.
        """
        return (
            self.quadratic_coeff_a * query_len * (query_len + num_computed_tokens)
            + self.linear_coeff_b * query_len
            + self.constant_coeff_c
        )

    def get_time_with_history(
        self,
        query_len: int,
        num_computed_tokens: int,
    ) -> float:
        """Get time T based on current seq_lens.

        f(C,H) = a*C(C+H) + b*C + d*H + c
        """
        return (
            self.quadratic_chunk_a * query_len * (query_len + num_computed_tokens)
            + self.linear_chunk_b * query_len
            + self.history_chunk_d * num_computed_tokens
            + self.constant_chunk_c
        )

    def predict(
        self,
        num_computed_tokens: int,
        base_chunk_size: int,
        page_size: int,
    ) -> int | None:
        """Predict next chunk size x such that a·x·(H+x) + b·x = T.

        The incremental latency of processing x new tokens when H tokens are
        already cached is:  T_incr(x, H) = a·x·(H+x) + b·x + c

        The constant overhead c affects every forward pass equally regardless
        of chunk size, so it cancels when solving for the optimal x:
            a·x² + (a·H + b)·x - (a·base² + b·base) = 0

        The key difference from the naive f(H+x)-f(H) subtraction is that
        full prefill attention cost is a·(H+x)² (all H+x tokens attend to all
        H+x tokens), while incremental attention cost is only a·x·(H+x)
        (only the x new tokens attend to the H+x KV positions — the H cached
        tokens do not recompute attention).  Using f(H+x)-f(H) overcounts the
        cross-term by a factor of 2, which systematically shrinks predictions
        as H grows.
        """
        if not self.is_ready or self.target_latency is None:
            return None

        if self.quadratic_coeff_a <= 0:
            return None

        A = self.quadratic_coeff_a
        B = self.quadratic_coeff_a * num_computed_tokens + self.linear_coeff_b
        C = -self.target_latency

        discriminant = B * B - 4 * A * C
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        x = (-B + sqrt_disc) / (2 * A)

        if x <= 0:
            return None

        smoothed = base_chunk_size + self.smooth_factor * (x - base_chunk_size)
        chunk_size = max(int(smoothed), self.min_chunk)

        align = max(page_size, 64)
        chunk_size = ((chunk_size + align - 1) // align) * align
        if chunk_size < align:
            chunk_size = align

        logger.debug("[ProfilingChunk] Predicted chunk_size=%d", chunk_size)
        return chunk_size if chunk_size >= align else None

    def predict_with_history(
        self,
        num_computed_tokens: int,
        base_chunk_size: int,
        page_size: int,
    ) -> int | None:
        """Predict next chunk size x using the history-aware model
        f(C,H) = a*C(C+H) + b*C + d*H + c."""
        if not self.is_ready or self.target_latency is None:
            return None

        if not self.with_history_ready:
            return None

        if self.quadratic_chunk_a <= 0:
            return None

        def aligned_min_chunk() -> int:
            align = max(page_size, 64)
            chunk_size = max(self.min_chunk, align)
            return ((chunk_size + align - 1) // align) * align

        # a*C^2 + (a*H + b)*C + d*H + c - T = 0
        A = self.quadratic_chunk_a
        B = self.quadratic_chunk_a * num_computed_tokens + self.linear_chunk_b
        C = self.history_chunk_d * num_computed_tokens + self.constant_chunk_c - self.target_latency

        discriminant = B * B - 4 * A * C
        if discriminant < 0:
            return aligned_min_chunk()

        sqrt_disc = math.sqrt(discriminant)
        x = (-B + sqrt_disc) / (2 * A)

        if x <= 0:
            return aligned_min_chunk()

        logger.debug("[ProfilingChunk] History-aware raw prediction: %.1f", x)
        smoothed = base_chunk_size + self.smooth_factor * (x - base_chunk_size)
        chunk_size = max(int(smoothed), self.min_chunk)

        align = max(page_size, 64)
        chunk_size = ((chunk_size + align - 1) // align) * align
        if chunk_size < align:
            chunk_size = align

        return chunk_size if chunk_size >= align else None


class ProfilingChunkManager:
    """Manager for profiling-based dynamic chunk sizing.

    Handles the profiling process and maintains the ChunkSizePredictor.
    """

    def __init__(
        self,
        base_chunk_size: int,
        page_size: int,
        smooth_factor: float = 0.8,
        min_chunk: int = 4096,
        cache_dir: str | None = None,
        cache_mode: str = "auto",
        profile_repeats: int = 5,
        profile_max_seq_len: int | None = None,
        profile_chunk_sizes: list[int] | None = None,
        profile_history_sizes: list[int] | None = None,
    ):
        self.base_chunk_size = base_chunk_size
        self.page_size = page_size
        self.min_chunk = min_chunk
        self.profile_samples: list[dict[str, Any]] = []
        self.profile_repeats = profile_repeats
        self.profile_max_seq_len = profile_max_seq_len
        self.profile_chunk_sizes = profile_chunk_sizes
        self.profile_history_sizes = profile_history_sizes
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "vllm-ascend",
            "profiling_chunk",
        )
        self.cache_mode = cache_mode
        if self.cache_mode not in VALID_CACHE_MODES:
            raise ValueError(
                f"profiling_chunk_config.cache_mode must be one of "
                f"{sorted(VALID_CACHE_MODES)}, got {self.cache_mode}"
            )

        self.predictor = ChunkSizePredictor(smooth_factor=smooth_factor, min_chunk=min_chunk)
        self._profiling_done = False

    @property
    def is_ready(self) -> bool:
        return self._profiling_done and self.predictor.is_ready

    @property
    def history_ready(self) -> bool:
        return self.is_ready and self.predictor.with_history_ready

    def get_profile_grid(self, max_model_len: int) -> list[tuple[int, int]]:
        """Return startup profiling points as (chunk_size, history_len)."""
        max_profile_seq_len = self._get_max_profile_seq_len(max_model_len)
        chunk_sizes = self._get_profile_chunk_sizes()
        history_sizes = self._get_profile_history_sizes(max_profile_seq_len)

        points = set()
        for history_len in history_sizes:
            for chunk_size in chunk_sizes:
                if history_len + chunk_size <= max_profile_seq_len:
                    points.add((chunk_size, history_len))
        if self.base_chunk_size <= max_profile_seq_len:
            points.add((self.base_chunk_size, 0))
        return sorted(points, key=lambda item: (item[1], item[0]))

    def _align(self, value: int, *, up: bool = True) -> int:
        align = max(self.page_size, 64)
        value = max(value, align)
        if up:
            return ((value + align - 1) // align) * align
        return (value // align) * align

    def _get_max_profile_seq_len(self, max_model_len: int) -> int:
        if self.profile_max_seq_len is not None:
            return max(1, min(int(self.profile_max_seq_len), max_model_len))
        return max(1, min(max_model_len, self.base_chunk_size * 8))

    def _get_profile_chunk_sizes(self) -> list[int]:
        if self.profile_chunk_sizes is not None:
            candidates = self.profile_chunk_sizes
        else:
            fractions = (0.125, 0.167, 0.25, 0.333, 0.5, 0.667, 0.75, 1.0)
            candidates = [self.min_chunk]
            candidates.extend(int(self.base_chunk_size * fraction) for fraction in fractions)
        chunk_sizes = {
            self._align(int(chunk_size), up=True)
            for chunk_size in candidates
            if int(chunk_size) > 0
        }
        chunk_sizes = {
            chunk_size
            for chunk_size in chunk_sizes
            if chunk_size <= self.base_chunk_size
        }
        if not chunk_sizes:
            chunk_sizes.add(self._align(self.base_chunk_size, up=False))
        return sorted(chunk_sizes)

    def _get_profile_history_sizes(self, max_profile_seq_len: int) -> list[int]:
        if self.profile_history_sizes is not None:
            candidates = self.profile_history_sizes
        else:
            candidates = [
                0,
                self.base_chunk_size // 2,
                self.base_chunk_size,
                self.base_chunk_size * 2,
                self.base_chunk_size * 4,
                self.base_chunk_size * 8,
                max_profile_seq_len - self.min_chunk,
            ]
        history_sizes = {
            self._align(int(history_len), up=False) if int(history_len) > 0 else 0
            for history_len in candidates
            if int(history_len) >= 0
        }
        return sorted(history_len for history_len in history_sizes if history_len < max_profile_seq_len)

    def build_cache_fingerprint(self, vllm_config: Any, max_model_len: int) -> dict[str, Any]:
        model_config = getattr(vllm_config, "model_config", None)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
        parallel_config = getattr(vllm_config, "parallel_config", None)
        grid = self.get_profile_grid(max_model_len)
        payload = {
            "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "fit_version": PROFILE_FIT_VERSION,
            "model": _get_nested_attr(model_config, "model"),
            "tokenizer": _get_nested_attr(model_config, "tokenizer"),
            "revision": _get_nested_attr(model_config, "revision"),
            "dtype": _get_nested_attr(model_config, "dtype"),
            "quantization": _get_nested_attr(model_config, "quantization"),
            "runner_type": _get_nested_attr(model_config, "runner_type"),
            "max_model_len": _get_nested_attr(model_config, "max_model_len"),
            "scheduler_max_model_len": _get_nested_attr(scheduler_config, "max_model_len"),
            "max_num_batched_tokens": _get_nested_attr(scheduler_config, "max_num_batched_tokens"),
            "max_num_seqs": _get_nested_attr(scheduler_config, "max_num_seqs"),
            "enable_chunked_prefill": _get_nested_attr(scheduler_config, "enable_chunked_prefill"),
            "block_size": _get_nested_attr(cache_config, "block_size"),
            "cache_dtype": _get_nested_attr(cache_config, "cache_dtype"),
            "pipeline_parallel_size": _get_nested_attr(parallel_config, "pipeline_parallel_size"),
            "tensor_parallel_size": _get_nested_attr(parallel_config, "tensor_parallel_size"),
            "decode_context_parallel_size": _get_nested_attr(parallel_config, "decode_context_parallel_size"),
            "prefill_context_parallel_size": _get_nested_attr(parallel_config, "prefill_context_parallel_size"),
            "base_chunk_size": self.base_chunk_size,
            "page_size": self.page_size,
            "min_chunk": self.predictor.min_chunk,
            "smooth_factor": self.predictor.smooth_factor,
            "profile_repeats": self.profile_repeats,
            "profile_max_seq_len": self._get_max_profile_seq_len(max_model_len),
            "profile_grid": grid,
            "platform_machine": platform.machine(),
            "python": platform.python_version(),
            "vllm_version": _distribution_version("vllm"),
            "vllm_ascend_version": _distribution_version("vllm-ascend"),
            "torch_version": _distribution_version("torch"),
            "torch_npu_version": _distribution_version("torch-npu"),
        }
        return _json_safe(payload)

    def get_cache_path(self, vllm_config: Any, max_model_len: int) -> Path:
        fingerprint = self.build_cache_fingerprint(vllm_config, max_model_len)
        digest = hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return Path(self.cache_dir).expanduser() / f"{digest}.json"

    def load_cached_profile(self, vllm_config: Any, max_model_len: int) -> bool:
        if self.cache_mode in {"off", "refresh"}:
            return False

        cache_path = self.get_cache_path(vllm_config, max_model_len)
        if not cache_path.exists():
            logger.info("[ProfilingChunk] No profiling cache found at %s", cache_path)
            return False

        try:
            with cache_path.open("r", encoding="utf-8") as cache_file:
                cache_data = json.load(cache_file)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("[ProfilingChunk] Failed to read profiling cache %s: %s", cache_path, e)
            return False

        expected_fingerprint = self.build_cache_fingerprint(vllm_config, max_model_len)
        if cache_data.get("fingerprint") != expected_fingerprint:
            logger.info("[ProfilingChunk] Ignoring stale profiling cache %s", cache_path)
            return False

        if cache_data.get("schema_version") != PROFILE_CACHE_SCHEMA_VERSION:
            logger.info("[ProfilingChunk] Ignoring unsupported profiling cache schema at %s", cache_path)
            return False

        if not self.fit_profile_samples(cache_data.get("samples", [])):
            logger.warning("[ProfilingChunk] Cached profiling data could not be fitted: %s", cache_path)
            return False

        logger.info("[ProfilingChunk] Loaded profiling cache from %s", cache_path)
        return True

    def save_profile_cache(self, vllm_config: Any, max_model_len: int) -> None:
        if self.cache_mode not in {"auto", "refresh"}:
            return
        if not self.profile_samples:
            return

        cache_path = self.get_cache_path(vllm_config, max_model_len)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "fit_version": PROFILE_FIT_VERSION,
            "fingerprint": self.build_cache_fingerprint(vllm_config, max_model_len),
            "base_chunk_size": self.base_chunk_size,
            "page_size": self.page_size,
            "target_latency_ms": self.predictor.target_latency,
            "coefficients": {
                "quadratic_chunk_a": self.predictor.quadratic_chunk_a,
                "linear_chunk_b": self.predictor.linear_chunk_b,
                "history_chunk_d": self.predictor.history_chunk_d,
                "constant_chunk_c": self.predictor.constant_chunk_c,
            },
            "samples": self.profile_samples,
        }
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=cache_path.parent,
                delete=False,
            ) as tmp_file:
                json.dump(cache_data, tmp_file, indent=2, sort_keys=True)
                tmp_name = tmp_file.name
            os.replace(tmp_name, cache_path)
        except OSError as e:
            logger.warning("[ProfilingChunk] Failed to save profiling cache %s: %s", cache_path, e)
            return
        logger.info("[ProfilingChunk] Saved profiling cache to %s", cache_path)

    def fit_profile_samples(self, samples: list[dict[str, Any]]) -> bool:
        """Fit predictor from cached/startup raw profiling samples."""
        if not samples:
            logger.warning("[ProfilingChunk] No profiling samples available for fitting")
            return False

        normalized_samples = []
        for sample in samples:
            runs = [float(value) for value in sample.get("runs_ms", [])]
            if not runs and "latency_ms" in sample:
                runs = [float(sample["latency_ms"])]
            if not runs and "median_ms" in sample:
                runs = [float(sample["median_ms"])]
            if not runs:
                logger.debug("[ProfilingChunk] Dropping profiling sample without timing data: %s", sample)
                continue
            normalized_samples.append({
                "chunk_size": int(sample["chunk_size"]),
                "history_len": int(sample["history_len"]),
                "seq_len": int(sample.get("seq_len", int(sample["chunk_size"]) + int(sample["history_len"]))),
                "runs_ms": runs,
                "mean_ms": float(sample.get("mean_ms", _mean(runs))),
                "median_ms": float(sample.get("median_ms", _median(runs))),
            })

        if not self.predictor.fit_chunk(normalized_samples):
            return False

        self.profile_samples = normalized_samples
        self.predictor.quadratic_coeff_a = self.predictor.quadratic_chunk_a
        self.predictor.linear_coeff_b = self.predictor.linear_chunk_b
        self.predictor.constant_coeff_c = self.predictor.constant_chunk_c

        target_sample = self._find_target_sample(normalized_samples)
        if target_sample is not None:
            self.predictor.target_latency = target_sample["median_ms"]
        else:
            self.predictor.target_latency = self.predictor.get_time_with_history(self.base_chunk_size, 0)
        if self.predictor.target_latency <= 0:
            self.predictor.target_latency = 1.0

        self.predictor.is_ready = True
        self.predictor.with_history_ready = True
        self._profiling_done = True
        logger.info(
            "[ProfilingChunk] Target latency: %.2f ms (base_chunk=%d)",
            self.predictor.target_latency,
            self.base_chunk_size,
        )
        return True

    def _find_target_sample(self, samples: list[dict[str, Any]]) -> dict[str, Any] | None:
        no_history_samples = [sample for sample in samples if sample["history_len"] == 0]
        if not no_history_samples:
            return None
        return min(
            no_history_samples,
            key=lambda sample: abs(sample["chunk_size"] - self.base_chunk_size),
        )

    def predict_chunk_size(self, num_computed_tokens: int, target_time: float) -> int | None:
        """Predict optimal chunk size for given history length."""
        if not self.is_ready:
            return None

        # NOTE(gjc): We found that the FIA operator has abnormal performance
        # when processing multiple request groups in a batch, so the target_latency
        # feature is temporarily fixed. It will be enabled again after the
        # issues with the FIA operator are resolved. Therefore, in multi-request
        # concurrent scenarios, there is still room for performance improvement in CPP.
        # self.predictor.target_latency = target_time

        if not self.history_ready:
            predict_func = self.predictor.predict
        else:
            predict_func = self.predictor.predict_with_history
        return predict_func(
            num_computed_tokens=num_computed_tokens, base_chunk_size=self.base_chunk_size, page_size=self.page_size
        )

    def predict_time(self, num_new_tokens: int, num_computed_tokens: int) -> float:
        """Get the consumed time of scheduled reqs for time_budget."""
        if not self.is_ready:
            return 0.0

        if not self.history_ready:
            predict_func = self.predictor.get_time
        else:
            predict_func = self.predictor.get_time_with_history
        return predict_func(query_len=num_new_tokens, num_computed_tokens=num_computed_tokens)
