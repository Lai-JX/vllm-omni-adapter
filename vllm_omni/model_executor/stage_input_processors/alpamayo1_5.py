"""Stage input processors for Alpamayo1.5."""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTextPrompt, OmniTokensPrompt

logger = logging.getLogger(__name__)

_DEFAULT_ALPAMAYO_MODEL_PATH = "/share/models/Alpamayo-1.5-10B"
_FUSION_ASSETS: dict[str, dict[str, Any]] = {}
_BASE_TOKENIZERS: dict[str, Any] = {}


def _ensure_alpamayo_import_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    alpamayo_src = repo_root / "alpamayo1.5" / "src"
    alpamayo_src_str = str(alpamayo_src)
    if alpamayo_src.is_dir() and alpamayo_src_str not in sys.path:
        sys.path.insert(0, alpamayo_src_str)


def _validate_stage_inputs(stage_list: list[Any], engine_input_source: list[int]) -> list[Any]:
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    return stage_list[source_stage_id].engine_outputs


def _normalize_prompt(prompt: Any) -> dict[str, Any]:
    if prompt is None:
        return {}
    if isinstance(prompt, dict):
        return dict(prompt)
    if hasattr(prompt, "_asdict"):
        return dict(prompt._asdict())
    if hasattr(prompt, "__dict__"):
        return dict(vars(prompt))
    return {}


def _detach_payload(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {k: _detach_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_detach_payload(v) for v in value]
    return value


def _strip_multimodal_payload_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep chat-template placeholders while removing actual image payloads."""

    sanitized_messages: list[dict[str, Any]] = []
    for message in messages:
        sanitized_message = dict(message)
        content = []
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                content.append({"type": "image"})
            elif isinstance(item, dict) and item.get("type") == "video":
                content.append({"type": "video"})
            else:
                content.append(item)
        sanitized_message["content"] = content
        sanitized_messages.append(sanitized_message)
    return sanitized_messages


def _to_cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.contiguous()


def _resolve_runtime_token_id(tokenizer: Any, token: str) -> int | None:
    if tokenizer is None:
        return None

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert_tokens_to_ids is None:
        return None

    try:
        token_id = convert_tokens_to_ids(token)
    except Exception:
        return None

    if token_id is None or isinstance(token_id, list):
        return None

    try:
        token_id = int(token_id)
    except (TypeError, ValueError):
        return None

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if unk_token_id is not None and token_id == int(unk_token_id):
        return None
    if token_id < 0:
        return None
    return token_id


def _as_batched_input_ids(input_ids: Any) -> torch.Tensor:
    tensor = _to_cpu_tensor(input_ids, dtype=torch.long)
    if tensor is None:
        raise ValueError("tokenized_data.input_ids is required for stage-0 traj fusion")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 1D or 2D input_ids, got shape {tuple(tensor.shape)}")
    return tensor


def _resolve_alpamayo_model_path(
    prompt_dict: dict[str, Any],
    additional_information: dict[str, Any],
    sampling_params: Any,
) -> str:
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    return str(
        additional_information.get("alpamayo_model_path")
        or prompt_dict.get("alpamayo_model_path")
        or extra_args.get("alpamayo_model_path")
        or os.environ.get("ALPAMAYO_MODEL_PATH")
        or _DEFAULT_ALPAMAYO_MODEL_PATH
    )


def _load_alpamayo_config_dict(alpamayo_model_path: str) -> dict[str, Any]:
    config_path = Path(alpamayo_model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Alpamayo config not found: {config_path}")
    with config_path.open("r") as f:
        return json.load(f)


def build_alpamayo_mm_processor_kwargs(model_path: str) -> dict[str, int]:
    """Return Alpamayo image sizing kwargs for vLLM multimodal preprocessing."""

    config = _load_alpamayo_config_dict(str(model_path or _DEFAULT_ALPAMAYO_MODEL_PATH))
    mm_processor_kwargs: dict[str, int] = {}
    for key in ("min_pixels", "max_pixels"):
        value = config.get(key)
        if value is not None:
            mm_processor_kwargs[key] = int(value)
    return mm_processor_kwargs


def _get_alpamayo_base_tokenizer(model_path: str) -> Any:
    resolved_model_path = str(model_path or _DEFAULT_ALPAMAYO_MODEL_PATH)
    cached = _BASE_TOKENIZERS.get(resolved_model_path)
    if cached is not None:
        return cached

    config = _load_alpamayo_config_dict(resolved_model_path)
    tokenizer = AutoProcessor.from_pretrained(
        config["vlm_name_or_path"],
        trust_remote_code=True,
        local_files_only=True,
    ).tokenizer
    _BASE_TOKENIZERS[resolved_model_path] = tokenizer
    return tokenizer


def _find_subsequence(sequence: list[int], pattern: list[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    limit = len(sequence) - len(pattern) + 1
    for idx in range(limit):
        if sequence[idx : idx + len(pattern)] == pattern:
            return idx
    return -1


def _retokenize_stage0_traj_segment_if_needed(
    input_ids: torch.Tensor,
    additional_information: dict[str, Any],
    alpamayo_model_path: str,
    tokenizer: Any | None = None,
) -> torch.Tensor:
    prompt_text = additional_information.get("stage0_prompt_text") or additional_information.get("prompt")
    if not isinstance(prompt_text, str):
        return input_ids

    marker = "<|traj_history_start|>"
    marker_pos = prompt_text.find(marker)
    if marker_pos < 0:
        return input_ids

    traj_text = prompt_text[marker_pos:]
    base_tokenizer = _get_alpamayo_base_tokenizer(alpamayo_model_path)
    target_tokenizer = tokenizer or build_alpamayo_stage0_tokenizer(alpamayo_model_path)

    base_ids = list(base_tokenizer.encode(traj_text, add_special_tokens=False))
    target_ids = list(target_tokenizer.encode(traj_text, add_special_tokens=False))
    if not base_ids or not target_ids or base_ids == target_ids:
        return input_ids

    prompt_ids = input_ids[0].tolist()
    start_idx = _find_subsequence(prompt_ids, base_ids)
    if start_idx < 0:
        return input_ids

    rewritten_ids = prompt_ids[:start_idx] + target_ids + prompt_ids[start_idx + len(base_ids) :]
    logger.info(
        "Retokenized Alpamayo stage-0 trajectory suffix from %d to %d tokens at offset %d",
        len(base_ids),
        len(target_ids),
        start_idx,
    )
    return torch.tensor([rewritten_ids], dtype=torch.long)


def _load_fusion_assets(alpamayo_model_path: str) -> dict[str, Any]:
    cached = _FUSION_ASSETS.get(alpamayo_model_path)
    if cached is not None:
        return cached

    _ensure_alpamayo_import_path()

    import hydra.utils as hyu

    from alpamayo1_5.models.base_model import tokenize_history_trajectory

    config = _load_alpamayo_config_dict(alpamayo_model_path)

    traj_tokenizer_cfg = config.get("traj_tokenizer_cfg")
    hist_traj_tokenizer_cfg = config.get("hist_traj_tokenizer_cfg")

    traj_tokenizer = None
    if traj_tokenizer_cfg is not None:
        try:
            traj_tokenizer = hyu.instantiate(traj_tokenizer_cfg, load_weights=False)
        except TypeError:
            traj_tokenizer = hyu.instantiate(traj_tokenizer_cfg)

    if hist_traj_tokenizer_cfg is not None:
        hist_traj_tokenizer = hyu.instantiate(hist_traj_tokenizer_cfg)
        hist_token_start_idx = int(config["traj_token_start_idx"])
        if traj_tokenizer is not None:
            hist_token_start_idx += int(getattr(traj_tokenizer, "vocab_size"))
    elif traj_tokenizer is not None:
        hist_traj_tokenizer = traj_tokenizer
        hist_token_start_idx = int(config["traj_token_start_idx"])
    else:
        raise ValueError(
            f"{alpamayo_model_path} does not define a usable trajectory tokenizer configuration"
        )

    assets = {
        "config": config,
        "hist_traj_tokenizer": hist_traj_tokenizer,
        "hist_token_start_idx": hist_token_start_idx,
        "tokenize_history_trajectory": tokenize_history_trajectory,
    }
    _FUSION_ASSETS[alpamayo_model_path] = assets
    return assets


def _fuse_stage0_history_tokens(
    input_ids: torch.Tensor,
    additional_information: dict[str, Any],
    alpamayo_model_path: str,
    tokenizer: Any | None = None,
) -> torch.Tensor:
    assets = _load_fusion_assets(alpamayo_model_path)

    ego_history_xyz = _to_cpu_tensor(additional_information.get("ego_history_xyz"), dtype=torch.float32)
    ego_history_rot = _to_cpu_tensor(additional_information.get("ego_history_rot"), dtype=torch.float32)
    if ego_history_xyz is None or ego_history_rot is None:
        return input_ids

    traj_data = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    hist_idx = assets["tokenize_history_trajectory"](
        assets["hist_traj_tokenizer"],
        traj_data,
        assets["hist_token_start_idx"],
    ).to(dtype=torch.long)

    configured_history_token_id = int(assets["config"]["traj_token_ids"]["history"])
    runtime_history_token_id = _resolve_runtime_token_id(tokenizer, "<|traj_history|>")
    candidate_history_token_ids = [configured_history_token_id]
    if runtime_history_token_id is not None and runtime_history_token_id not in candidate_history_token_ids:
        candidate_history_token_ids.append(runtime_history_token_id)

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in candidate_history_token_ids:
        mask |= input_ids == int(token_id)
    expected_tokens = hist_idx.shape[-1]
    if not mask.any():
        prompt_text = additional_information.get("stage0_prompt_text") or additional_information.get("prompt")
        prompt_occurrences = (
            int(str(prompt_text).count("<|traj_history|>")) if prompt_text is not None else None
        )
        logger.warning(
            "Stage-0 traj fusion skipped: no <|traj_history|> placeholder found in prompt ids "
            "(configured_id=%s runtime_id=%s prompt_occurrences=%s prompt_len=%s)",
            configured_history_token_id,
            runtime_history_token_id,
            prompt_occurrences,
            int(input_ids.shape[-1]),
        )
        return input_ids
    if not torch.all(mask.sum(dim=1) == expected_tokens):
        raise ValueError(
            "Stage-0 traj fusion placeholder count mismatch: "
            f"expected {expected_tokens} tokens per sample, got {mask.sum(dim=1).tolist()}"
        )
    if hist_idx.shape != input_ids[mask].view(input_ids.shape[0], -1).shape:
        raise ValueError(
            "Stage-0 traj fusion token shape mismatch: "
            f"hist_idx={tuple(hist_idx.shape)} mask_view={tuple(input_ids[mask].view(input_ids.shape[0], -1).shape)}"
        )

    fused = input_ids.clone()
    fused[mask] = hist_idx.reshape(-1)
    return fused.contiguous()


def build_alpamayo_fused_tokenized_data(
    model_path: str,
    messages: list[dict[str, Any]],
    ego_history_xyz: Any | None = None,
    ego_history_rot: Any | None = None,
) -> dict[str, Any]:
    """Build Alpamayo tokenized_data exactly like the original inference path.

    This keeps the VLM tokenization outside vLLM and returns the final
    ``tokenized_data`` payload whose ``input_ids`` already contain the fused
    history trajectory tokens.
    """

    processor = build_alpamayo_stage0_processor(model_path)

    tokenized_data = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    tokenized_data = {key: _detach_payload(value) for key, value in tokenized_data.items()}
    input_ids = _as_batched_input_ids(tokenized_data["input_ids"])

    if ego_history_xyz is not None and ego_history_rot is not None:
        input_ids = _fuse_stage0_history_tokens(
            input_ids,
            {
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            },
            model_path,
            tokenizer=processor.tokenizer,
        )

    tokenized_data["input_ids"] = input_ids
    return tokenized_data


def build_alpamayo_stage0_processor(
    model_path: str,
    tokenizer: Any | None = None,
) -> Any:
    """Build the shared Alpamayo processor used by stage-0 prompt preparation."""

    resolved_model_path = str(model_path or _DEFAULT_ALPAMAYO_MODEL_PATH)
    config = _load_alpamayo_config_dict(resolved_model_path)
    processor_kwargs = build_alpamayo_mm_processor_kwargs(resolved_model_path)

    processor = AutoProcessor.from_pretrained(
        config["vlm_name_or_path"],
        trust_remote_code=True,
        local_files_only=True,
        **processor_kwargs,
    )
    processor.tokenizer = tokenizer or build_alpamayo_stage0_tokenizer(resolved_model_path)
    return processor


def build_alpamayo_stage0_prompt_text(
    model_path: str,
    messages: list[dict[str, Any]],
    tokenizer: Any | None = None,
) -> str:
    """Render Alpamayo chat messages into raw text for vLLM multimodal expansion."""

    processor = build_alpamayo_stage0_processor(model_path, tokenizer=tokenizer)
    prompt_messages = _strip_multimodal_payload_from_messages(messages)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    if not isinstance(prompt_text, str):
        raise TypeError(f"Expected string prompt text, got {type(prompt_text)!r}")
    return prompt_text


def build_alpamayo_stage0_tokenizer(
    model_path: str,
    tokenizer: Any | None = None,
) -> Any:
    """Build the Alpamayo-expanded tokenizer used by original inference."""

    del tokenizer
    _ensure_alpamayo_import_path()
    from alpamayo1_5.models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN
    from vllm.tokenizers.hf import get_cached_tokenizer

    resolved_model_path = str(model_path or _DEFAULT_ALPAMAYO_MODEL_PATH)
    if not (Path(resolved_model_path) / "config.json").is_file():
        resolved_model_path = os.environ.get("ALPAMAYO_MODEL_PATH") or _DEFAULT_ALPAMAYO_MODEL_PATH

    config = _load_alpamayo_config_dict(resolved_model_path)
    processor_kwargs = build_alpamayo_mm_processor_kwargs(resolved_model_path)

    processor = AutoProcessor.from_pretrained(
        config["vlm_name_or_path"],
        trust_remote_code=True,
        local_files_only=True,
        **processor_kwargs,
    )
    tokenizer = processor.tokenizer

    traj_vocab_size = int(config.get("traj_vocab_size") or 0)
    if traj_vocab_size > 0:
        tokenizer.add_tokens([f"<i{v}>" for v in range(traj_vocab_size)])
        tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")

    if config.get("add_special_tokens", True):
        tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
    else:
        tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)

    tokenizer.traj_token_ids = {
        key: tokenizer.convert_tokens_to_ids(value) for key, value in TRAJ_TOKEN.items()
    }
    expected_traj_token_ids = config.get("traj_token_ids") or {}
    if tokenizer.traj_token_start_idx != int(config["traj_token_start_idx"]):
        raise ValueError(
            "Alpamayo tokenizer rewrite produced an unexpected trajectory token start index: "
            f"expected {config['traj_token_start_idx']}, got {tokenizer.traj_token_start_idx}"
        )
    for key, expected_id in expected_traj_token_ids.items():
        actual_id = tokenizer.traj_token_ids.get(key)
        if actual_id != int(expected_id):
            raise ValueError(
                "Alpamayo tokenizer rewrite produced an unexpected token id for "
                f"{key}: expected {expected_id}, got {actual_id}"
            )
    return get_cached_tokenizer(tokenizer)


def rewrite_stage0_prompt_for_vllm_multimodal(
    prompt: dict[str, Any] | OmniTextPrompt | OmniTokensPrompt | str,
    sampling_params: Any,
) -> dict[str, Any] | OmniTokensPrompt | str:
    """Inject Alpamayo mm_processor kwargs before vLLM multimodal expansion."""

    prompt_dict = _normalize_prompt(prompt)
    if not prompt_dict or "multi_modal_data" not in prompt_dict:
        return prompt

    additional_information = _detach_payload(prompt_dict.get("additional_information") or {})
    alpamayo_model_path = _resolve_alpamayo_model_path(prompt_dict, additional_information, sampling_params)

    mm_processor_kwargs = dict(prompt_dict.get("mm_processor_kwargs") or {})
    for key, value in build_alpamayo_mm_processor_kwargs(alpamayo_model_path).items():
        mm_processor_kwargs.setdefault(key, value)

    if "prompt" in prompt_dict:
        additional_information["stage0_prompt_text"] = prompt_dict["prompt"]

    if not mm_processor_kwargs:
        return prompt

    if isinstance(prompt, dict):
        prompt.setdefault("additional_information", {}).update(additional_information)
        prompt["mm_processor_kwargs"] = mm_processor_kwargs
        return prompt

    prompt_dict["mm_processor_kwargs"] = mm_processor_kwargs
    prompt_dict["additional_information"] = additional_information
    return prompt_dict


def postprocess_stage0_request_for_traj_fusion(
    request: Any,
    prompt: Any,
    sampling_params: Any,
    tokenizer: Any | None = None,
    model_path: str | None = None,
) -> Any:
    """Swap in Alpamayo-finalized stage-0 token ids after vLLM MM preprocessing."""
    prompt_dict = _normalize_prompt(prompt)
    if not prompt_dict:
        return request

    additional_information = _detach_payload(prompt_dict.get("additional_information") or {})
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    if not prompt_token_ids:
        return request

    alpamayo_model_path = str(model_path or _DEFAULT_ALPAMAYO_MODEL_PATH)
    alpamayo_model_path = _resolve_alpamayo_model_path(
        prompt_dict,
        additional_information,
        sampling_params,
    )
    if additional_information.get("ego_history_xyz") is None or additional_information.get("ego_history_rot") is None:
        return request
    prompt_input_ids = _as_batched_input_ids(prompt_token_ids)
    prompt_input_ids = _retokenize_stage0_traj_segment_if_needed(
        prompt_input_ids,
        additional_information,
        alpamayo_model_path,
        tokenizer=tokenizer,
    )
    fused_input_ids = _fuse_stage0_history_tokens(
        prompt_input_ids,
        additional_information,
        alpamayo_model_path,
        tokenizer=tokenizer,
    )

    request.prompt_token_ids = fused_input_ids[0].tolist()

    tokenized_data = dict(additional_information.get("tokenized_data") or {})
    tokenized_data["input_ids"] = fused_input_ids
    tokenized_data.setdefault("attention_mask", torch.ones_like(fused_input_ids, dtype=torch.long))
    additional_information["tokenized_data"] = tokenized_data
    additional_information["stage0_fused_traj_tokens"] = True
    additional_information["stage0_used_external_tokenized_data"] = False
    additional_information["stage0_multimodal_expansion"] = "vllm"
    additional_information["stage0_fused_prompt_length"] = int(fused_input_ids.shape[-1])
    additional_information["alpamayo_model_path"] = alpamayo_model_path
    if isinstance(prompt_dict.get("additional_information"), dict):
        prompt_dict["additional_information"].update(additional_information)
    elif isinstance(prompt, dict):
        prompt["additional_information"] = additional_information

    return request


def rewrite_stage0_prompt_for_traj_fusion(
    prompt: dict[str, Any] | OmniTextPrompt | OmniTokensPrompt | str,
    sampling_params: Any,
) -> dict[str, Any] | OmniTokensPrompt | str:
    """Rewrite the stage-0 prompt so Qwen3VL consumes fused Alpamayo token ids.

    Original Alpamayo inference tokenizes with the Alpamayo tokenizer first and
    then replaces `<|traj_history|>` placeholders with discrete trajectory
    tokens before VLM generation. We mirror that behavior here by converting the
    incoming request into an `OmniTokensPrompt` carrying the fused token ids and
    the original multimodal payload.
    """

    prompt_dict = _normalize_prompt(prompt)
    if not prompt_dict:
        return prompt

    additional_information = _detach_payload(prompt_dict.get("additional_information") or {})
    tokenized_data = copy.deepcopy(additional_information.get("tokenized_data") or {})
    input_ids = tokenized_data.get("input_ids")
    if input_ids is None:
        return prompt

    if additional_information.get("ego_history_xyz") is None or additional_information.get("ego_history_rot") is None:
        return prompt

    fused_input_ids = _as_batched_input_ids(input_ids)
    alpamayo_model_path = _resolve_alpamayo_model_path(prompt_dict, additional_information, sampling_params)
    fused_input_ids = _fuse_stage0_history_tokens(
        fused_input_ids,
        additional_information,
        alpamayo_model_path,
    )

    tokenized_data["input_ids"] = fused_input_ids
    additional_information["tokenized_data"] = tokenized_data
    additional_information["stage0_fused_traj_tokens"] = True
    additional_information["stage0_fused_prompt_length"] = int(fused_input_ids.shape[-1])
    additional_information["alpamayo_model_path"] = alpamayo_model_path

    rewritten_prompt = OmniTokensPrompt(
        prompt_token_ids=fused_input_ids[0].tolist(),
        additional_information=additional_information,
    )

    for key in (
        "prompt",
        "modalities",
        "multi_modal_data",
        "mm_processor_kwargs",
        "multi_modal_uuids",
        "cache_salt",
    ):
        if key in prompt_dict:
            rewritten_prompt[key] = prompt_dict[key]

    return rewritten_prompt


def vlm2trajectory(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTextPrompt | TextPrompt | list[Any] | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTextPrompt]:
    """Package VLM outputs plus original trajectory context for stage 1."""

    del requires_multimodal_data
    stage_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    prompts = prompt if isinstance(prompt, list) else [prompt] * len(stage_outputs)

    trajectory_inputs: list[OmniTextPrompt] = []
    for i, stage_output in enumerate(stage_outputs):
        outputs = list(stage_output.outputs or [])
        if not outputs:
            raise RuntimeError("Stage 0 produced no completion outputs for Alpamayo trajectory rollout")
        original_prompt = _normalize_prompt(prompts[i] if i < len(prompts) else None)
        original_additional_information = _detach_payload(
            original_prompt.get("additional_information") or {}
        )

        prompt_token_ids = list(stage_output.prompt_token_ids or [])
        output_token_ids_list = [list(output.token_ids or []) for output in outputs]
        if len(output_token_ids_list) == 1:
            output_token_ids_payload: list[int] | list[list[int]] = output_token_ids_list[0]
            full_sequence: list[int] | list[list[int]] = prompt_token_ids + output_token_ids_list[0]
        else:
            output_token_ids_payload = output_token_ids_list
            full_sequence = [prompt_token_ids + output_token_ids for output_token_ids in output_token_ids_list]

        transformed_info = dict(original_additional_information)
        tokenized_data = dict(transformed_info.get("tokenized_data") or {})
        tokenized_data["input_ids"] = full_sequence
        transformed_info["tokenized_data"] = tokenized_data
        transformed_info["stage0_prompt_token_ids"] = prompt_token_ids
        transformed_info["stage0_output_token_ids"] = output_token_ids_payload
        transformed_info["stage0_sequences"] = full_sequence
        transformed_info["stage0_prompt_length"] = len(prompt_token_ids)
        if len(output_token_ids_list) == 1:
            transformed_info["stage0_output_length"] = len(output_token_ids_list[0])
        else:
            transformed_info["stage0_output_lengths"] = [len(output_token_ids) for output_token_ids in output_token_ids_list]
        transformed_info["stage0_num_return_sequences"] = len(output_token_ids_list)
        transformed_info["num_return_sequences"] = max(
            int(transformed_info.get("num_return_sequences", 1) or 1),
            len(output_token_ids_list),
        )

        multimodal_output = getattr(outputs[0], "multimodal_output", None) or {}
        latent = multimodal_output.get("latent")
        if isinstance(latent, torch.Tensor):
            transformed_info["stage0_latent"] = latent.detach().cpu().to(torch.float32).contiguous()
            transformed_info["stage0_latent_shape"] = list(latent.shape)
        for key in ("rope_deltas", "position_ids", "attention_mask"):
            value = multimodal_output.get(key)
            if isinstance(value, torch.Tensor):
                transformed_info[f"stage0_{key}"] = value.detach().cpu().contiguous()

        trajectory_inputs.append(
            OmniTextPrompt(
                prompt="",
                additional_information=transformed_info,
            )
        )

    return trajectory_inputs
