"""Custom logits processor for Alpamayo stage-0 stopping semantics."""

from __future__ import annotations

from typing import Any

import torch
from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

STOP_AFTER_TOKEN_ARG = "alpamayo_stop_after_token_id"
FORCED_STOP_TOKEN_ARG = "alpamayo_forced_stop_token_id"


class _ForceTokenAfterMarker:
    """Force one final token immediately after a marker token is generated."""

    def __init__(self, marker_token_id: int, forced_token_id: int) -> None:
        self.marker_token_id = marker_token_id
        self.forced_token_id = forced_token_id

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if not output_ids or output_ids[-1] != self.marker_token_id:
            return logits

        keep = logits[self.forced_token_id].item()
        logits[:] = float("-inf")
        logits[self.forced_token_id] = keep
        return logits


class AlpamayoStopAfterFutureStartLogitsProcessor(AdapterLogitsProcessor):
    """Mimic Alpamayo's StopAfterEOS semantics for stage-0 generation.

    Once ``<|traj_future_start|>`` is generated, the next decode step is forced
    to emit a designated stop token (typically ``pad_token_id``). vLLM then
    stops on that forced token, which means the KV cache has already advanced
    through ``<|traj_future_start|>`` just like the original HuggingFace
    implementation that stops one token later.
    """

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra_args = params.extra_args or {}
        stop_after_token_id = extra_args.get(STOP_AFTER_TOKEN_ARG)
        forced_stop_token_id = extra_args.get(FORCED_STOP_TOKEN_ARG)

        if stop_after_token_id is None and forced_stop_token_id is None:
            return

        if not isinstance(stop_after_token_id, int):
            raise ValueError(
                f"`{STOP_AFTER_TOKEN_ARG}` must be an integer, got {stop_after_token_id!r}."
            )
        if not isinstance(forced_stop_token_id, int):
            raise ValueError(
                f"`{FORCED_STOP_TOKEN_ARG}` must be an integer, got {forced_stop_token_id!r}."
            )

        stop_token_ids = params.stop_token_ids or []
        if forced_stop_token_id not in stop_token_ids:
            raise ValueError(
                f"`{FORCED_STOP_TOKEN_ARG}`={forced_stop_token_id} must be present in "
                f"`stop_token_ids`, got {stop_token_ids}."
            )

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        extra_args: dict[str, Any] = params.extra_args or {}
        stop_after_token_id = extra_args.get(STOP_AFTER_TOKEN_ARG)
        forced_stop_token_id = extra_args.get(FORCED_STOP_TOKEN_ARG)
        if stop_after_token_id is None or forced_stop_token_id is None:
            return None
        return _ForceTokenAfterMarker(
            marker_token_id=int(stop_after_token_id),
            forced_token_id=int(forced_stop_token_id),
        )
