"""Optional request-state dumping helpers for stage debugging.

This module is intentionally independent from model logic. The main runner
only calls ``maybe_dump_request_state`` at a few points, while the dump
format, filtering, and serialization live here.

Enable with environment variables:
- ``VLLM_OMNI_REQUEST_DUMP_DIR``: output directory
- ``VLLM_OMNI_REQUEST_DUMP_REQ_IDS``: comma-separated request-id allowlist
- ``VLLM_OMNI_REQUEST_DUMP_PHASES``: comma-separated phase allowlist
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_PHASE_INITIALIZED = "request_state_initialized"
_PHASE_BATCHED = "request_state_batched"


def _parse_csv_env(name: str) -> set[str] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _to_tensor_if_numeric_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, list) and value and all(isinstance(item, (int, np.integer)) for item in value):
        return torch.tensor(value, dtype=torch.long)
    return value


def _to_serializable(value: Any) -> Any:
    value = _to_tensor_if_numeric_list(value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _to_serializable(vars(value))
        except Exception:
            pass
    return repr(value)


def _extract_mm_features(mm_features: Any) -> list[dict[str, Any]]:
    extracted: list[dict[str, Any]] = []
    for idx, mm_feature in enumerate(mm_features or []):
        item_data: dict[str, Any] = {"index": idx}
        try:
            mm_item = getattr(mm_feature, "data", None)
            if mm_item is None:
                item_data["data"] = _to_serializable(mm_feature)
            elif hasattr(mm_item, "get_data"):
                item_data["data"] = _to_serializable(mm_item.get_data())
            else:
                item_data["data"] = _to_serializable(mm_item)
        except Exception as exc:
            item_data["error"] = repr(exc)
        extracted.append(item_data)
    return extracted


def _extract_sampling_params(sampling_params: Any) -> dict[str, Any] | None:
    if sampling_params is None:
        return None
    fields = (
        "seed",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "n",
        "stop_token_ids",
    )
    payload = {name: _to_serializable(getattr(sampling_params, name, None)) for name in fields}
    payload["extra_args"] = _to_serializable(getattr(sampling_params, "extra_args", None))
    return payload


@dataclass(frozen=True)
class RequestStateDumpConfig:
    dump_dir: Path
    request_ids: set[str] | None = None
    phases: set[str] | None = None

    @classmethod
    def from_env(cls) -> "RequestStateDumpConfig | None":
        dump_dir = os.environ.get("VLLM_OMNI_REQUEST_DUMP_DIR", "").strip()
        if not dump_dir:
            return None
        return cls(
            dump_dir=Path(dump_dir),
            request_ids=_parse_csv_env("VLLM_OMNI_REQUEST_DUMP_REQ_IDS"),
            phases=_parse_csv_env("VLLM_OMNI_REQUEST_DUMP_PHASES"),
        )


class RequestStateDumpTool:
    def __init__(self, config: RequestStateDumpConfig):
        self.config = config
        self._lock = threading.Lock()
        self._dumped: set[tuple[str, str]] = set()

    @classmethod
    def from_env(cls) -> "RequestStateDumpTool | None":
        config = RequestStateDumpConfig.from_env()
        if config is None:
            return None
        return cls(config)

    def _matches(self, req_id: str, phase: str) -> bool:
        if self.config.request_ids is not None and req_id not in self.config.request_ids:
            return False
        if self.config.phases is not None and phase not in self.config.phases:
            return False
        return True

    def maybe_dump_request_state(
        self,
        *,
        phase: str,
        req_state: Any,
        additional_information: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path | None:
        req_id = str(getattr(req_state, "req_id", "unknown"))
        if not self._matches(req_id, phase):
            return None

        dump_key = (req_id, phase)
        with self._lock:
            if dump_key in self._dumped:
                return None
            self._dumped.add(dump_key)

        try:
            req_dir = self.config.dump_dir / req_id
            req_dir.mkdir(parents=True, exist_ok=True)
            output_path = req_dir / f"{phase}.pt"

            payload = {
                "phase": phase,
                "req_id": req_id,
                "timestamp_s": time.time(),
                "prompt_token_ids": _to_serializable(getattr(req_state, "prompt_token_ids", None)),
                "output_token_ids": _to_serializable(getattr(req_state, "output_token_ids", None)),
                "block_ids": _to_serializable(getattr(req_state, "block_ids", None)),
                "num_computed_tokens": int(getattr(req_state, "num_computed_tokens", 0)),
                "mrope_positions": _to_serializable(getattr(req_state, "mrope_positions", None)),
                "mrope_position_delta": _to_serializable(getattr(req_state, "mrope_position_delta", None)),
                "sampling_params": _extract_sampling_params(getattr(req_state, "sampling_params", None)),
                "mm_features": _extract_mm_features(getattr(req_state, "mm_features", None)),
                "additional_information": _to_serializable(additional_information),
                "extra": _to_serializable(extra or {}),
            }
            torch.save(payload, output_path)
            logger.info(
                "Dumped request state for req=%s phase=%s to %s",
                req_id,
                phase,
                output_path,
            )
            return output_path
        except Exception:
            with self._lock:
                self._dumped.discard(dump_key)
            logger.exception("Failed to dump request state for req=%s phase=%s", req_id, phase)
            return None


def request_dump_phase_initialized() -> str:
    return _PHASE_INITIALIZED


def request_dump_phase_batched() -> str:
    return _PHASE_BATCHED
