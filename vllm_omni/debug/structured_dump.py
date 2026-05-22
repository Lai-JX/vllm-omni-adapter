from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

DUMP_VERSION = 1
_RESERVED_TOP_LEVEL_KEYS = {
    "dump_version",
    "req_id",
    "phase",
    "source",
    "call_index",
    "sample_index",
    "step_index",
    "metadata",
}


def parse_csv_env(name: str) -> set[str] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def to_serializable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "__tensor__": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "data": tensor.tolist(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return to_serializable(vars(value))
        except Exception:
            pass
    return repr(value)


@dataclass(frozen=True)
class DumpIdentity:
    call_index: int = 0
    sample_index: int = 0
    step_index: int | None = None


@dataclass(frozen=True)
class StructuredDumpConfig:
    dump_dir: Path
    source: str
    request_ids: set[str] | None = None
    phases: set[str] | None = None


@dataclass(frozen=True)
class DumpDecision:
    should_dump: bool
    reason: str


def load_dump_config_from_env(
    *,
    dump_dir_env: str,
    request_ids_env: str,
    phases_env: str,
    source: str,
) -> StructuredDumpConfig | None:
    dump_dir = os.environ.get(dump_dir_env, "").strip()
    if not dump_dir:
        return None
    return StructuredDumpConfig(
        dump_dir=Path(dump_dir),
        source=source,
        request_ids=parse_csv_env(request_ids_env),
        phases=parse_csv_env(phases_env),
    )


def build_dump_filename(
    phase: str,
    *,
    call_index: int = 0,
    sample_index: int = 0,
    step_index: int | None = None,
) -> str:
    parts = [phase, f"call{int(call_index):03d}"]
    if step_index is not None:
        parts.append(f"step{int(step_index):03d}")
    parts.append(f"sample{int(sample_index):03d}")
    return ".".join(parts) + ".json"


def build_dump_path(
    base_dir: Path,
    *,
    req_id: str,
    phase: str,
    identity: DumpIdentity | None = None,
) -> Path:
    normalized_identity = identity or DumpIdentity()
    return (
        Path(base_dir)
        / req_id
        / build_dump_filename(
            phase,
            call_index=normalized_identity.call_index,
            sample_index=normalized_identity.sample_index,
            step_index=normalized_identity.step_index,
        )
    )


class StructuredDumpTool:
    def __init__(
        self,
        *,
        name: str,
        config: StructuredDumpConfig | None,
        dump_dir_env: str,
        request_ids_env: str,
        phases_env: str,
    ):
        self.name = name
        self.config = config
        self.dump_dir_env = dump_dir_env
        self.request_ids_env = request_ids_env
        self.phases_env = phases_env
        self._lock = threading.Lock()
        self._dumped: set[tuple[str, str, int, int, int | None]] = set()

    def _matches_request_id(self, req_id: str) -> bool:
        if self.config is None or self.config.request_ids is None:
            return True
        if req_id in self.config.request_ids:
            return True
        return any(req_id.startswith(f"{allowed}-") for allowed in self.config.request_ids)

    def _build_decision(self, *, req_id: str, phase: str, identity: DumpIdentity) -> DumpDecision:
        if self.config is None:
            return DumpDecision(
                should_dump=False,
                reason=f"{self.dump_dir_env} is not set",
            )
        if not self._matches_request_id(req_id):
            return DumpDecision(
                should_dump=False,
                reason=(
                    f"req_id={req_id} does not match allowlist from {self.request_ids_env}"
                ),
            )
        if self.config.phases is not None and phase not in self.config.phases:
            return DumpDecision(
                should_dump=False,
                reason=f"phase={phase} does not match allowlist from {self.phases_env}",
            )

        dump_key = (
            req_id,
            phase,
            int(identity.call_index),
            int(identity.sample_index),
            None if identity.step_index is None else int(identity.step_index),
        )
        with self._lock:
            if dump_key in self._dumped:
                return DumpDecision(
                    should_dump=False,
                    reason=(
                        "deduped by "
                        f"(req_id, phase, call_index, sample_index, step_index)={dump_key}"
                    ),
                )
            self._dumped.add(dump_key)
        return DumpDecision(should_dump=True, reason="matched")

    def maybe_dump(
        self,
        *,
        req_id: str,
        phase: str,
        payload: dict[str, Any],
        identity: DumpIdentity | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        normalized_identity = identity or DumpIdentity()
        decision = self._build_decision(
            req_id=req_id,
            phase=phase,
            identity=normalized_identity,
        )
        if not decision.should_dump:
            logger.info(
                "Skip %s for req=%s phase=%s call=%s sample=%s step=%s: %s",
                self.name,
                req_id,
                phase,
                normalized_identity.call_index,
                normalized_identity.sample_index,
                normalized_identity.step_index,
                decision.reason,
            )
            return None

        dump_key = (
            req_id,
            phase,
            int(normalized_identity.call_index),
            int(normalized_identity.sample_index),
            None if normalized_identity.step_index is None else int(normalized_identity.step_index),
        )
        try:
            assert self.config is not None
            output_path = build_dump_path(
                self.config.dump_dir,
                req_id=req_id,
                phase=phase,
                identity=normalized_identity,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            serializable_payload = {
                "dump_version": DUMP_VERSION,
                "req_id": req_id,
                "phase": phase,
                "source": self.config.source,
                "call_index": int(normalized_identity.call_index),
                "sample_index": int(normalized_identity.sample_index),
                "step_index": None
                if normalized_identity.step_index is None
                else int(normalized_identity.step_index),
                "metadata": to_serializable(metadata or {}),
            }
            for key, value in payload.items():
                output_key = str(key)
                if output_key in _RESERVED_TOP_LEVEL_KEYS:
                    output_key = f"payload_{output_key}"
                serializable_payload[output_key] = to_serializable(value)
            output_path.write_text(
                json.dumps(serializable_payload, indent=2, ensure_ascii=False)
            )
            logger.info(
                "Dumped %s for req=%s phase=%s call=%s sample=%s step=%s to %s",
                self.name,
                req_id,
                phase,
                normalized_identity.call_index,
                normalized_identity.sample_index,
                normalized_identity.step_index,
                output_path,
            )
            return output_path
        except Exception:
            with self._lock:
                self._dumped.discard(dump_key)
            logger.exception(
                "Failed to dump %s for req=%s phase=%s call=%s sample=%s step=%s",
                self.name,
                req_id,
                phase,
                normalized_identity.call_index,
                normalized_identity.sample_index,
                normalized_identity.step_index,
            )
            return None
