# SPDX-License-Identifier: Apache-2.0
"""FCFS schedulers that consume PREFLOW partial-prefill spill handoffs.

The Mooncake connector models remote prefill as a complete remote prompt. A
PREFLOW spill deliberately exports only a prefix of the original prompt. This
shim temporarily exposes that prefix as the request's complete prompt while
Mooncake allocates and receives its KV, then restores the original prompt so
the standard scheduler computes the missing suffix. Scheduling order and all
other admission behavior remain upstream FCFS.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.preflow_spill import (
    PREFLOW_SPILL_AVAILABLE_HISTORY_KEY,
    PREFLOW_SPILL_NEW_TOKENS_KEY,
    PREFLOW_SPILL_ORIGINAL_PROMPT_TOKENS_KEY,
    PREFLOW_SPILL_REMOTE_TOKENS_KEY,
)


@dataclass
class _OriginalPromptState:
    prompt_token_suffix: list[int]
    all_token_suffix: list[int]
    num_prompt_tokens: int


class _PREFLOWSpillConsumerMixin:
    """Temporarily present a spilled remote prefix as a complete prompt."""

    def __init__(self, *args, **kwargs) -> None:
        self._preflow_original_prompts: dict[str, _OriginalPromptState] = {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def _spill_remote_tokens(request: Request) -> int | None:
        params = request.kv_transfer_params
        if params is None:
            return None
        remote_tokens = params.get(PREFLOW_SPILL_REMOTE_TOKENS_KEY)
        if remote_tokens is None:
            return None
        if type(remote_tokens) is not int:
            raise ValueError(f"{PREFLOW_SPILL_REMOTE_TOKENS_KEY} must be an int, got {type(remote_tokens).__name__}")
        return remote_tokens

    def _stage_spilled_prompt(self, request: Request) -> None:
        remote_tokens = self._spill_remote_tokens(request)
        if remote_tokens is None:
            return
        assert request.kv_transfer_params is not None
        if request.prompt_token_ids is None or request.prompt_embeds is not None:
            raise ValueError("PREFLOW spill currently requires a token-ID prompt without prompt embeddings")
        if request.prompt_is_token_ids is not None or request.mm_features:
            raise ValueError("PREFLOW spill currently does not support mixed or multimodal prompts")

        original_prompt_tokens = request.num_prompt_tokens
        spill_integers = {
            key: request.kv_transfer_params.get(key)
            for key in (
                PREFLOW_SPILL_ORIGINAL_PROMPT_TOKENS_KEY,
                PREFLOW_SPILL_AVAILABLE_HISTORY_KEY,
                PREFLOW_SPILL_NEW_TOKENS_KEY,
            )
        }
        invalid_types = [key for key, value in spill_integers.items() if type(value) is not int]
        if invalid_types:
            raise ValueError("PREFLOW spill integer metadata is missing or invalid: " + ", ".join(invalid_types))
        declared_prompt_tokens = spill_integers[PREFLOW_SPILL_ORIGINAL_PROMPT_TOKENS_KEY]
        available_history = spill_integers[PREFLOW_SPILL_AVAILABLE_HISTORY_KEY]
        new_prompt_tokens = spill_integers[PREFLOW_SPILL_NEW_TOKENS_KEY]
        assert isinstance(declared_prompt_tokens, int)
        assert isinstance(available_history, int)
        assert isinstance(new_prompt_tokens, int)
        if declared_prompt_tokens != original_prompt_tokens:
            raise ValueError(
                "PREFLOW spill original prompt length does not match the "
                f"decode request: {declared_prompt_tokens} != {original_prompt_tokens}"
            )
        if not 0 <= available_history < original_prompt_tokens:
            raise ValueError(
                "PREFLOW spill available prompt history must be in "
                f"[0, {original_prompt_tokens}), got {available_history}"
            )
        if new_prompt_tokens != original_prompt_tokens - available_history:
            raise ValueError(
                "PREFLOW spill new prompt token count is inconsistent with the original prompt and available history"
            )
        if remote_tokens != available_history + 1:
            raise ValueError(
                "PREFLOW spill remote prompt length must contain the available "
                "history plus one source-computed position"
            )
        if not 0 < remote_tokens <= original_prompt_tokens:
            raise ValueError(
                f"{PREFLOW_SPILL_REMOTE_TOKENS_KEY} must be in [1, {original_prompt_tokens}], got {remote_tokens}"
            )
        if len(request._all_token_ids) != original_prompt_tokens:
            raise ValueError("PREFLOW spill must be staged before a request has generated output tokens")

        self._preflow_original_prompts[request.request_id] = _OriginalPromptState(
            prompt_token_suffix=list(request.prompt_token_ids[remote_tokens:]),
            all_token_suffix=list(request._all_token_ids[remote_tokens:]),
            num_prompt_tokens=original_prompt_tokens,
        )
        del request.prompt_token_ids[remote_tokens:]
        del request._all_token_ids[remote_tokens:]
        request.num_prompt_tokens = remote_tokens

    def _restore_spilled_prompt(self, request: Request) -> bool:
        state = self._preflow_original_prompts.pop(request.request_id, None)
        if state is None:
            return False
        assert request.prompt_token_ids is not None
        request.prompt_token_ids.extend(state.prompt_token_suffix)
        request._all_token_ids.extend(state.all_token_suffix)
        request.num_prompt_tokens = state.num_prompt_tokens
        return True

    def add_request(self, request: Request) -> None:
        self._stage_spilled_prompt(request)
        try:
            super().add_request(request)
        except Exception:
            self._restore_spilled_prompt(request)
            raise

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        # Restore first. The base method's full-hit correction must compare the
        # received prefix against the original prompt length, not the temporary
        # remote-prefix length.
        self._restore_spilled_prompt(request)
        super()._update_waiting_for_remote_kv(request)

    def finish_requests(
        self,
        request_ids: str | Iterable[str] | None,
        finished_status: RequestStatus,
    ) -> list[Request]:
        if request_ids is None:
            ids_to_restore = tuple(self._preflow_original_prompts)
        elif isinstance(request_ids, str):
            ids_to_restore = (request_ids,)
        else:
            ids_to_restore = tuple(request_ids)
            request_ids = ids_to_restore

        for request_id in ids_to_restore:
            request = self.requests.get(request_id)
            if request is not None:
                self._restore_spilled_prompt(request)
        return super().finish_requests(request_ids, finished_status)


class PREFLOWSpillDecodeScheduler(_PREFLOWSpillConsumerMixin, Scheduler):
    """Standard FCFS scheduler with PREFLOW spill prompt restoration."""


class AsyncPREFLOWSpillDecodeScheduler(_PREFLOWSpillConsumerMixin, AsyncScheduler):
    """Async FCFS scheduler with PREFLOW spill prompt restoration."""


__all__ = ["AsyncPREFLOWSpillDecodeScheduler", "PREFLOWSpillDecodeScheduler"]
