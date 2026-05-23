from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import torch
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from transformers import AutoConfig
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessingInfo,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.transformers_utils.processor import cached_get_processor_without_dynamic_kwargs
from vllm.utils.mistral import is_mistral_tokenizer


@lru_cache(maxsize=8)
def _load_alpamayo_processor_overrides(model_path: str) -> dict[str, object]:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return {}

    with config_path.open("r") as f:
        config = json.load(f)

    overrides: dict[str, object] = {}
    for key in ("min_pixels", "max_pixels"):
        value = config.get(key)
        if value is not None:
            overrides[key] = value
    return overrides


class Alpamayo1_5Qwen3VLProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        tokenizer = self.ctx.tokenizer
        if is_mistral_tokenizer(tokenizer):
            tokenizer = tokenizer.transformers_tokenizer

        merged_kwargs = self.ctx.get_merged_mm_kwargs(kwargs)
        merged_kwargs.pop("tokenizer", None)

        hf_config = self.get_hf_config()
        processor_overrides = _load_alpamayo_processor_overrides(str(self.ctx.model_config.model))
        for key in ("min_pixels", "max_pixels"):
            value = getattr(hf_config, key, None)
            if value is None:
                value = processor_overrides.get(key)
            if value is not None:
                merged_kwargs.setdefault(key, value)

        processor_path = (
            getattr(hf_config, "vlm_name_or_path", None)
            or self.ctx.model_config.tokenizer
            or self.ctx.model_config.model
        )
        revision = self.ctx.model_config.tokenizer_revision or self.ctx.model_config.revision
        return cached_get_processor_without_dynamic_kwargs(
            processor_path,
            revision=revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
            processor_cls=Qwen3VLProcessor,
            tokenizer=tokenizer,
            **merged_kwargs,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Alpamayo1_5Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Alpamayo1_5Qwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3-VL wrapper that loads the Alpamayo stage-0 VLM weights.

    Alpamayo checkpoints store stage-0 backbone weights under the ``vlm.``
    prefix and keep stage-1 trajectory modules in the same checkpoint. This
    wrapper rebuilds a Qwen3-VL HF config from Alpamayo's ``vlm_name_or_path``
    while preserving Alpamayo's expanded vocabulary size, then filters the
    checkpoint down to the stage-0 VLM weights during loading.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vlm.": "",
            "action_in_proj.": None,
            "action_out_proj.": None,
            "expert.": None,
        }
    ) | Qwen3VLForConditionalGeneration.hf_to_vllm_mapper

    def __init__(self, *, vllm_config, prefix: str = "model"):
        hf_config = vllm_config.model_config.hf_config

        if getattr(hf_config, "text_config", None) is None:
            model_path = Path(str(vllm_config.model_config.model))
            with (model_path / "config.json").open("r") as f:
                alpamayo_config = json.load(f)

            base_hf_config = AutoConfig.from_pretrained(
                alpamayo_config["vlm_name_or_path"],
                trust_remote_code=True,
                local_files_only=True,
            )
            base_hf_config.text_config.vocab_size = int(alpamayo_config["vocab_size"])
            base_hf_config.vocab_size = int(alpamayo_config["vocab_size"])

            vllm_config = vllm_config.with_hf_config(
                base_hf_config,
                architectures=["Alpamayo1_5Qwen3VLForConditionalGeneration"],
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def embed_multimodal(self, **kwargs: object):
        should_sync = torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing()
        if should_sync:
            torch.cuda.synchronize()
        wall_start = time.time()
        perf_start = time.perf_counter()
        try:
            return super().embed_multimodal(**kwargs)
        finally:
            if should_sync:
                torch.cuda.synchronize()
            wall_end = time.time()
            self._last_embed_start_time = wall_start
            self._last_embed_end_time = wall_end
            self._last_embed_multimodal_ms = (time.perf_counter() - perf_start) * 1000.0

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs: object,
    ):
        should_sync = torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing()
        if should_sync:
            torch.cuda.synchronize()
        wall_start = time.time()
        perf_start = time.perf_counter()
        try:
            return super().forward(
                input_ids,
                positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        finally:
            if should_sync:
                torch.cuda.synchronize()
            wall_end = time.time()
            self._last_forward_start_time = wall_start
            self._last_forward_end_time = wall_end
            self._last_forward_ms = (time.perf_counter() - perf_start) * 1000.0

    def pop_last_profile_metrics(self) -> dict[str, float]:
        metrics = {
            "embed_multimodal_ms": float(getattr(self, "_last_embed_multimodal_ms", 0.0) or 0.0),
            "forward_ms": float(getattr(self, "_last_forward_ms", 0.0) or 0.0),
            "embed_start": float(getattr(self, "_last_embed_start_time", 0.0) or 0.0),
            "embed_end": float(getattr(self, "_last_embed_end_time", 0.0) or 0.0),
            "forward_start": float(getattr(self, "_last_forward_start_time", 0.0) or 0.0),
            "forward_end": float(getattr(self, "_last_forward_end_time", 0.0) or 0.0),
        }
        self._last_embed_multimodal_ms = 0.0
        self._last_forward_ms = 0.0
        self._last_embed_start_time = 0.0
        self._last_embed_end_time = 0.0
        self._last_forward_start_time = 0.0
        self._last_forward_end_time = 0.0
        return metrics
