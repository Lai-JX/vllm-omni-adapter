"""
Backward-compatible diffusion-only entrypoint.

Historically some local tests imported ``AsyncOmniDiffusion`` from this module
and expected ``generate`` to return a single final ``OmniRequestOutput``
instead of an async generator. The diffusion-only orchestration now lives in
``AsyncOmni`` and is configured automatically when no stage config is passed.
"""

from __future__ import annotations

from typing import Any

from vllm_omni.entrypoints.async_omni import AsyncOmni


class AsyncOmniDiffusion(AsyncOmni):
    """Compatibility shim for legacy diffusion-only callers."""

    def __init__(self, *args: Any, batch_size: int | None = None, **kwargs: Any) -> None:
        # Legacy callers passed ``batch_size`` to the diffusion entrypoint.
        # The current engine expects ``max_num_seqs`` for single-stage diffusion.
        if batch_size is not None and kwargs.get("max_num_seqs") is None:
            kwargs["max_num_seqs"] = batch_size
        super().__init__(*args, **kwargs)

    async def generate(self, *args: Any, **kwargs: Any):
        """Return the terminal diffusion output for legacy call sites."""
        final_output = None
        async for output in super().generate(*args, **kwargs):
            final_output = output
        if final_output is None:
            raise RuntimeError("no diffusion output generated")
        return final_output
