# SPDX-License-Identifier: Apache-2.0
"""HF config registration for Alpamayo1.5 checkpoints.

This adapter makes Alpamayo configs behave like Qwen3-VL configs for vLLM's
multimodal initialization while still preserving Alpamayo-specific fields.
"""

from __future__ import annotations

from transformers import AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig


class Alpamayo1_5Config(Qwen3VLConfig):
    model_type = "alpamayo1_5"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        vlm_name_or_path = kwargs.get("vlm_name_or_path")

        if text_config is None and vision_config is None and vlm_name_or_path:
            base_config = AutoConfig.from_pretrained(
                vlm_name_or_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            text_config = base_config.text_config.to_dict()
            vision_config = base_config.vision_config.to_dict()
            kwargs.setdefault("image_token_id", base_config.image_token_id)
            kwargs.setdefault("video_token_id", base_config.video_token_id)
            kwargs.setdefault("vision_start_token_id", base_config.vision_start_token_id)
            kwargs.setdefault("vision_end_token_id", base_config.vision_end_token_id)
            kwargs.setdefault("tie_word_embeddings", base_config.tie_word_embeddings)

        super().__init__(text_config=text_config, vision_config=vision_config, **kwargs)

        vocab_size = kwargs.get("vocab_size")
        if vocab_size is not None:
            self.text_config.vocab_size = int(vocab_size)
            self.vocab_size = int(vocab_size)
