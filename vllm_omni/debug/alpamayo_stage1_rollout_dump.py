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
from pathlib import Path
from typing import Any

from vllm_omni.debug.structured_dump import (
    DumpIdentity,
    StructuredDumpTool,
    load_dump_config_from_env,
)


class AlpamayoStage1DumpTool(StructuredDumpTool):
    @classmethod
    def from_env(cls) -> "AlpamayoStage1DumpTool":
        config = load_dump_config_from_env(
            dump_dir_env="VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR",
            request_ids_env="VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS",
            phases_env="VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES",
            source="alpamayo_stage1",
        )
        return cls(
            name="Alpamayo stage-1 rollout payload",
            config=config,
            dump_dir_env="VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR",
            request_ids_env="VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS",
            phases_env="VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES",
        )

    def maybe_dump(
        self,
        *,
        req_id: str,
        phase: str,
        payload: dict[str, Any],
        identity: DumpIdentity | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        return super().maybe_dump(
            req_id=req_id,
            phase=phase,
            payload=payload,
            identity=identity,
            metadata=metadata,
        )


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
    call_index: int = 0,
    sample_index: int = 0,
    step_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    tool = _get_tool()
    if tool is None:
        return None
    return tool.maybe_dump(
        req_id=req_id,
        phase=phase,
        payload=payload,
        identity=DumpIdentity(
            call_index=call_index,
            sample_index=sample_index,
            step_index=step_index,
        ),
        metadata=metadata,
    )
