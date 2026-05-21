"""Optional Alpamayo stage-1 rollout dumping helpers.

This module stays separate from the rollout implementation. The pipeline only
hands over a small payload at a few checkpoints when explicitly enabled via
environment variables.

Enable with:
- ``VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR``
- ``VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS`` (optional CSV allowlist)
- ``VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES`` (optional CSV allowlist)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def _parse_csv_env(name: str) -> set[str] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
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


@dataclass(frozen=True)
class AlpamayoStage1DumpConfig:
    dump_dir: Path
    request_ids: set[str] | None = None
    phases: set[str] | None = None

    @classmethod
    def from_env(cls) -> "AlpamayoStage1DumpConfig | None":
        dump_dir = os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR", "").strip()
        if not dump_dir:
            return None
        return cls(
            dump_dir=Path(dump_dir),
            request_ids=_parse_csv_env("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"),
            phases=_parse_csv_env("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"),
        )


class AlpamayoStage1DumpTool:
    def __init__(self, config: AlpamayoStage1DumpConfig):
        self.config = config
        self._lock = threading.Lock()
        self._dumped: set[tuple[str, str]] = set()

    @classmethod
    def from_env(cls) -> "AlpamayoStage1DumpTool | None":
        config = AlpamayoStage1DumpConfig.from_env()
        if config is None:
            return None
        return cls(config)

    def _matches(self, req_id: str, phase: str) -> bool:
        if self.config.request_ids is not None and req_id not in self.config.request_ids:
            return False
        if self.config.phases is not None and phase not in self.config.phases:
            return False
        return True

    def maybe_dump(
        self,
        *,
        req_id: str,
        phase: str,
        payload: dict[str, Any],
    ) -> Path | None:
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
            serializable = {
                "req_id": req_id,
                "phase": phase,
                **{str(k): _to_serializable(v) for k, v in payload.items()},
            }
            torch.save(serializable, output_path)
            logger.info(
                "Dumped Alpamayo stage-1 rollout payload for req=%s phase=%s to %s",
                req_id,
                phase,
                output_path,
            )
            return output_path
        except Exception:
            with self._lock:
                self._dumped.discard(dump_key)
            logger.exception(
                "Failed to dump Alpamayo stage-1 rollout payload for req=%s phase=%s",
                req_id,
                phase,
            )
            return None


_ENV_NAMES = (
    "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR",
    "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS",
    "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES",
)
_TOOL: AlpamayoStage1DumpTool | None = None
_TOOL_ENV_SNAPSHOT: tuple[str | None, ...] | None = None


def _get_tool() -> AlpamayoStage1DumpTool | None:
    global _TOOL
    global _TOOL_ENV_SNAPSHOT

    env_snapshot = tuple(os.environ.get(name) for name in _ENV_NAMES)
    if env_snapshot != _TOOL_ENV_SNAPSHOT:
        _TOOL_ENV_SNAPSHOT = env_snapshot
        _TOOL = AlpamayoStage1DumpTool.from_env()
    return _TOOL


def maybe_dump_alpamayo_stage1_rollout(
    *,
    req_id: str,
    phase: str,
    payload: dict[str, Any],
) -> Path | None:
    tool = _get_tool()
    if tool is None:
        return None
    return tool.maybe_dump(req_id=req_id, phase=phase, payload=payload)
