import asyncio
import copy
import gc
import hashlib
import json
import os
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from transformers.cache_utils import DynamicCache

import numpy as np
import torch
import yaml
from transformers import AutoConfig

import common as ct
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import StopAfterEOS, replace_padding_after_eos, to_special_token
from vllm_omni.debug.compare_request_state_dump import compare_dump_files
from vllm_omni.debug.structured_dump import DUMP_VERSION, DumpIdentity, build_dump_filename, build_dump_path
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.model_executor.layers.rotary_embedding.mrope import OmniMRotaryEmbedding
from vllm_omni.model_executor.stage_input_processors.alpamayo1_5 import (
    build_alpamayo_fused_tokenized_data,
    build_alpamayo_stage0_prompt_text,
    build_alpamayo_stage0_tokenizer,
)
from vllm_omni.model_executor.stage_input_processors import alpamayo1_5 as alp_stage_processors

REQUEST_ID = "alpamayo-compare-original"
FIXED_X0_SEED = 20260520
FIXED_X0_SHAPE = (1, 64, 2)
COMPARE_ARTIFACTS_ROOT = Path.cwd() / "alpamayo_compare_artifacts"


def load_shared_data(
    *,
    clip_id: str = ct.CLIP_ID,
    t0_us: int = ct.T0_US,
):
    return ct.load_shared_data(
        use_helper_messages=True,
        clip_id=clip_id,
        t0_us=t0_us,
    )


def extract_cot_body(text: str) -> str:
    start_token = "<|cot_start|>"
    end_token = "<|cot_end|>"
    if start_token in text:
        text = text.split(start_token, 1)[1]
    if end_token in text:
        text = text.split(end_token, 1)[0]
    return text.strip()


def compare_tensors(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, object]:
    equal = torch.equal(lhs, rhs)
    same_shape = tuple(lhs.shape) == tuple(rhs.shape)
    max_abs_diff = None
    if lhs.shape == rhs.shape and lhs.dtype.is_floating_point and rhs.dtype.is_floating_point:
        max_abs_diff = float((lhs - rhs).abs().max().item()) if lhs.numel() else 0.0
    elif lhs.shape == rhs.shape:
        max_abs_diff = 0.0 if equal else float((lhs != rhs).sum().item())
    status = "match" if equal and same_shape else "diff"
    print(
        f"COMPARE phase={name} status={status} same_shape={same_shape} equal={equal} max_abs_diff={max_abs_diff}"
    )
    return {
        "comparison_target": name,
        "status": status,
        "same_shape": same_shape,
        "equal": equal,
        "max_abs_diff": max_abs_diff,
        "lhs_shape": list(lhs.shape),
        "rhs_shape": list(rhs.shape),
        "lhs_dtype": str(lhs.dtype),
        "rhs_dtype": str(rhs.dtype),
    }


def build_fixed_initial_noise_x0(
    *,
    seed: int = FIXED_X0_SEED,
    shape: tuple[int, int, int] = FIXED_X0_SHAPE,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).contiguous()


def build_stage_params(tokenizer):
    return ct.build_stage_params(tokenizer)


def _reference_additional_information(data):
    return ct.build_additional_information(data)


def load_reference_hf_config():
    config_path = Path(ct.MODEL_PATH) / "config.json"
    with config_path.open("r") as f:
        alpamayo_cfg = json.load(f)
    return AutoConfig.from_pretrained(
        alpamayo_cfg["vlm_name_or_path"],
        trust_remote_code=True,
        local_files_only=True,
    )


def _to_cpu_payload(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {k: _to_cpu_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu_payload(v) for v in value]
    return value


def _to_json_payload(value):
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
        return {str(k): _to_json_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_payload(v) for v in value]
    if isinstance(value, tuple):
        return [_to_json_payload(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _hash_debug_value(value: object) -> str:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        payload = b"|".join(
            [
                str(tensor.dtype).encode(),
                json.dumps(list(tensor.shape)).encode(),
                tensor.numpy().tobytes(),
            ]
        )
        return hashlib.sha256(payload).hexdigest()[:16]
    normalized = _to_json_payload(value)
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _from_json_payload(value):
    if isinstance(value, dict):
        if value.get("__tensor__"):
            dtype_name = str(value["dtype"]).split(".")[-1]
            return torch.tensor(value["data"], dtype=getattr(torch, dtype_name))
        if value.get("__ndarray__"):
            return np.asarray(value["data"], dtype=np.dtype(value["dtype"]))
        return {k: _from_json_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_json_payload(v) for v in value]
    return value


def log_stage1_transition_hashes(label: str, stage0_debug: dict[str, object]) -> None:
    fields = [
        "stage0_prompt_token_ids",
        "stage0_output_token_ids",
        "stage0_sequences",
        "stage0_rope_deltas",
        "stage0_attention_mask",
        "initial_noise_x0",
    ]
    summary = {field: _hash_debug_value(stage0_debug.get(field)) for field in fields}
    print(f"stage1_transition_hashes[{label}] {json.dumps(summary, sort_keys=True)}")


def _write_json_dump(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_json_payload(payload), indent=2, ensure_ascii=False))
    return path


def make_compare_run_dir() -> Path:
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir = COMPARE_ARTIFACTS_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _build_reference_dump_path(
    base_dir: Path,
    *,
    phase: str,
    identity: DumpIdentity | None = None,
) -> Path:
    normalized_identity = identity or DumpIdentity()
    return (
        Path(base_dir)
        / REQUEST_ID
        / f"reference_{build_dump_filename(
            phase,
            call_index=normalized_identity.call_index,
            sample_index=normalized_identity.sample_index,
            step_index=normalized_identity.step_index,
        )}"
    )


def reference_request_state_dump_path(base_dir: Path, phase: str) -> Path:
    return _build_reference_dump_path(
        base_dir,
        phase=phase,
        identity=DumpIdentity(call_index=0, sample_index=0),
    )


def reference_stage1_transition_dump_path(base_dir: Path) -> Path:
    return _build_reference_dump_path(
        base_dir,
        phase="stage1_transition",
        identity=DumpIdentity(call_index=0, sample_index=0),
    )


def reference_stage1_rollout_context_dump_path(base_dir: Path) -> Path:
    return _build_reference_dump_path(
        base_dir,
        phase="stage1_rollout_context",
        identity=DumpIdentity(call_index=0, sample_index=0),
    )


def reference_stage1_rollout_step_dump_path(base_dir: Path) -> Path:
    return _build_reference_dump_path(
        base_dir,
        phase="stage1_rollout_step",
        identity=DumpIdentity(call_index=0, sample_index=0, step_index=0),
    )


def reference_action_in_proj_internal_dump_path(base_dir: Path) -> Path:
    return _build_reference_dump_path(
        base_dir,
        phase="action_in_proj_internal",
        identity=DumpIdentity(call_index=0, sample_index=0, step_index=0),
    )


def actual_action_in_proj_internal_dump_path(base_dir: Path) -> Path:
    return build_dump_path(
        base_dir,
        req_id=REQUEST_ID,
        phase="action_in_proj_internal",
        identity=DumpIdentity(call_index=0, sample_index=0, step_index=0),
    )


def actual_stage1_rollout_step_dump_path(base_dir: Path) -> Path:
    return build_dump_path(
        base_dir,
        req_id=REQUEST_ID,
        phase="stage1_rollout_step",
        identity=DumpIdentity(call_index=0, sample_index=0, step_index=0),
    )


def reference_expert_internal_dump_path(base_dir: Path) -> Path:
    return _build_reference_dump_path(
        base_dir,
        phase="expert_internal",
        identity=DumpIdentity(call_index=0, sample_index=0, step_index=0),
    )


def actual_expert_internal_dump_path(base_dir: Path) -> Path:
    return build_dump_path(
        base_dir,
        req_id=REQUEST_ID,
        phase="expert_internal",
        identity=DumpIdentity(call_index=0, sample_index=0, step_index=0),
    )


def write_action_in_proj_internal_dump(
    base_dir: Path,
    *,
    source: str,
    captured_debug: dict[str, object],
) -> Path:
    output_path = (
        reference_action_in_proj_internal_dump_path(base_dir)
        if source == "reference"
        else actual_action_in_proj_internal_dump_path(base_dir)
    )
    return _write_json_dump(
        output_path,
        {
            "dump_version": DUMP_VERSION,
            "phase": "action_in_proj_internal",
            "req_id": REQUEST_ID,
            "source": source,
            "call_index": 0,
            "sample_index": 0,
            "step_index": 0,
            "metadata": {},
            **captured_debug,
        },
    )


def write_expert_internal_dump(
    base_dir: Path,
    *,
    source: str,
    captured_debug: dict[str, object],
) -> Path:
    output_path = (
        reference_expert_internal_dump_path(base_dir)
        if source == "reference"
        else actual_expert_internal_dump_path(base_dir)
    )
    return _write_json_dump(
        output_path,
        {
            "dump_version": DUMP_VERSION,
            "phase": "expert_internal",
            "req_id": REQUEST_ID,
            "source": source,
            "call_index": 0,
            "sample_index": 0,
            "step_index": 0,
            "metadata": {},
            **captured_debug,
        },
    )


def _effective_fourier_freqs(encoder, ref_input: torch.Tensor) -> torch.Tensor | None:
    if hasattr(encoder, "_build_freqs"):
        return encoder._build_freqs(ref_input)
    freqs = getattr(encoder, "freqs", None)
    if isinstance(freqs, torch.Tensor):
        return freqs.to(device=ref_input.device, dtype=ref_input.dtype)
    return None


def _run_action_in_proj_with_debug(module, x: torch.Tensor, timesteps: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch_size, num_steps, _ = x.shape
    debug: dict[str, torch.Tensor] = {
        "x_in": x.detach().cpu().contiguous(),
        "t_in": timesteps.detach().cpu().contiguous(),
    }
    action_branches = []
    for i, encoder in enumerate(module.sinus):
        action_input = x[:, :, i]
        debug[f"action_input_{i}"] = action_input.detach().cpu().contiguous()
        freqs = _effective_fourier_freqs(encoder, action_input)
        if freqs is not None:
            debug[f"freqs_action_{i}"] = freqs.detach().cpu().contiguous()
        branch_out = encoder(action_input)
        debug[f"fourier_action_{i}"] = branch_out.detach().cpu().contiguous()
        action_branches.append(branch_out)
    action_feats = torch.cat(action_branches, dim=-1)
    debug["action_feats"] = action_feats.detach().cpu().contiguous()

    timestep_scalar = timesteps[..., -1]
    debug["timestep_scalar"] = timestep_scalar.detach().cpu().contiguous()
    timestep_freqs = _effective_fourier_freqs(module.timestep_fourier_encoder, timestep_scalar)
    if timestep_freqs is not None:
        debug["freqs_t"] = timestep_freqs.detach().cpu().contiguous()
    timestep_single = module.timestep_fourier_encoder(timestep_scalar)
    debug["fourier_t"] = timestep_single.detach().cpu().contiguous()
    timestep_feats = timestep_single.repeat(1, num_steps, 1)
    debug["timestep_feats"] = timestep_feats.detach().cpu().contiguous()

    fused = torch.cat((action_feats, timestep_feats), dim=-1)
    debug["concat_before_mlp"] = fused.detach().cpu().contiguous()
    mlp_in = fused.flatten(0, 1)
    debug["mlp_in"] = mlp_in.detach().cpu().contiguous()
    current = mlp_in
    for idx, layer in enumerate(module.encoder.trunk):
        current = layer(current)
        debug[f"mlp_layer_{idx:02d}"] = current.detach().cpu().contiguous()
    debug["encoder_out"] = current.detach().cpu().contiguous()
    reshaped = current.reshape(batch_size, num_steps, -1)
    debug["encoder_out_reshaped"] = reshaped.detach().cpu().contiguous()
    norm_out = module.norm(reshaped)
    debug["norm_out"] = norm_out.detach().cpu().contiguous()
    debug["future_token_embeds"] = norm_out.detach().cpu().contiguous()
    return norm_out, debug


@contextmanager
def capture_action_in_proj_internal(module, captured_debug: dict[str, torch.Tensor]):
    original_forward = module.forward

    def wrapped_forward(x: torch.Tensor, timesteps: torch.Tensor):
        output = original_forward(x, timesteps)
        if not captured_debug:
            _, debug = _run_action_in_proj_with_debug(module, x, timesteps)
            captured_debug.update(debug)
        return output

    module.forward = wrapped_forward
    try:
        yield
    finally:
        module.forward = original_forward


@contextmanager
def capture_runtime_action_in_proj_internal(captured_debug: dict[str, torch.Tensor]):
    from vllm_omni.diffusion.models.alpamayo1_5 import runtime as alp_runtime

    original_forward = alp_runtime.PerWaypointActionInProjV2.forward

    def wrapped_forward(module, x: torch.Tensor, timesteps: torch.Tensor):
        output = original_forward(module, x, timesteps)
        if not captured_debug:
            _, debug = _run_action_in_proj_with_debug(module, x, timesteps)
            captured_debug.update(debug)
        return output

    alp_runtime.PerWaypointActionInProjV2.forward = wrapped_forward
    try:
        yield
    finally:
        alp_runtime.PerWaypointActionInProjV2.forward = original_forward


@contextmanager
def capture_expert_internal(module, captured_debug: dict[str, object]):
    original_forward = module.forward

    def wrapped_forward(*args, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        position_ids = kwargs.get("position_ids")
        attention_mask = kwargs.get("attention_mask")
        prompt_cache = kwargs.get("past_key_values")
        n_diffusion_tokens = int(inputs_embeds.shape[1]) if isinstance(inputs_embeds, torch.Tensor) else None
        prompt_cache_seq_len = None if prompt_cache is None else int(prompt_cache.get_seq_length())
        prompt_cache_summary_before_expert = None if prompt_cache is None else _summarize_prompt_cache(prompt_cache)

        output = original_forward(*args, **kwargs)
        prompt_cache_summary_after_expert = None if prompt_cache is None else _summarize_prompt_cache(prompt_cache)
        if not captured_debug:
            last_hidden_state = getattr(output, "last_hidden_state", None)
            captured_debug.update(
                {
                    "future_token_embeds": None if inputs_embeds is None else inputs_embeds.detach().cpu().contiguous(),
                    "position_ids": None if position_ids is None else position_ids.detach().cpu().contiguous(),
                    "attention_mask": None if attention_mask is None else attention_mask.detach().cpu().contiguous(),
                    "prompt_cache_seq_len": prompt_cache_seq_len,
                    "prompt_cache_summary": prompt_cache_summary_before_expert,
                    "prompt_cache_summary_before_expert": prompt_cache_summary_before_expert,
                    "prompt_cache_summary_after_expert": prompt_cache_summary_after_expert,
                    "last_hidden_state": None if last_hidden_state is None else last_hidden_state.detach().cpu().contiguous(),
                    "last_hidden": None
                    if last_hidden_state is None or n_diffusion_tokens is None
                    else last_hidden_state[:, -n_diffusion_tokens:].detach().cpu().contiguous(),
                }
            )
        return output

    module.forward = wrapped_forward
    try:
        yield
    finally:
        module.forward = original_forward


def _summarize_cache_tensor(tensor: torch.Tensor | None) -> dict[str, object] | None:
    if tensor is None:
        return None
    detached = tensor.detach().to(dtype=torch.float32)
    flat = detached.reshape(-1)
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(detached.mean().item()) if detached.numel() else 0.0,
        "std": float(detached.std(unbiased=False).item()) if detached.numel() else 0.0,
        "abs_max": float(detached.abs().max().item()) if detached.numel() else 0.0,
        "first": float(flat[0].item()) if flat.numel() else 0.0,
        "last": float(flat[-1].item()) if flat.numel() else 0.0,
    }


def _summarize_prompt_cache(prompt_cache) -> dict[str, object] | None:
    layers = getattr(prompt_cache, "layers", None)
    if isinstance(layers, list):
        return {
            "num_layers": len(layers),
            "seq_len": int(prompt_cache.get_seq_length()),
            "layers": [
                {
                    "layer_index": layer_idx,
                    "key": _summarize_cache_tensor(getattr(layer, "keys", None)),
                    "value": _summarize_cache_tensor(getattr(layer, "values", None)),
                }
                for layer_idx, layer in enumerate(layers)
            ],
        }

    key_cache = getattr(prompt_cache, "key_cache", None)
    value_cache = getattr(prompt_cache, "value_cache", None)
    if not isinstance(key_cache, list) or not isinstance(value_cache, list):
        return None
    return {
        "num_layers": len(key_cache),
        "seq_len": int(key_cache[0].shape[-2]) if key_cache and isinstance(key_cache[0], torch.Tensor) and key_cache[0].ndim >= 2 else 0,
        "layers": [
            {
                "layer_index": layer_idx,
                "key": _summarize_cache_tensor(k),
                "value": _summarize_cache_tensor(v),
            }
            for layer_idx, (k, v) in enumerate(zip(key_cache, value_cache, strict=False))
        ],
    }


def _serialize_prompt_cache(prompt_cache):
    key_cache = getattr(prompt_cache, "key_cache", None)
    value_cache = getattr(prompt_cache, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list):
        return SimpleNamespace(
            key_cache=[None if t is None else t.detach().cpu().contiguous() for t in key_cache],
            value_cache=[None if t is None else t.detach().cpu().contiguous() for t in value_cache],
        )

    layers = getattr(prompt_cache, "layers", None)
    if isinstance(layers, list):
        return SimpleNamespace(
            key_cache=[None if getattr(layer, "keys", None) is None else layer.keys.detach().cpu().contiguous() for layer in layers],
            value_cache=[None if getattr(layer, "values", None) is None else layer.values.detach().cpu().contiguous() for layer in layers],
        )
    return None


def load_dumped_omni_prompt_cache(kv_dump_path: Path) -> SimpleNamespace:
    payload = torch.load(kv_dump_path, map_location="cpu")
    key_cache = payload.get("key_cache")
    value_cache = payload.get("value_cache")
    if not isinstance(key_cache, list) or not isinstance(value_cache, list):
        raise RuntimeError(f"invalid dumped omni KV payload: {kv_dump_path}")
    return SimpleNamespace(key_cache=key_cache, value_cache=value_cache)


def build_dynamic_cache_from_serialized_prompt_cache(prompt_cache, *, device: str | torch.device):
    key_cache = getattr(prompt_cache, "key_cache", None)
    value_cache = getattr(prompt_cache, "value_cache", None)
    if not isinstance(key_cache, list) or not isinstance(value_cache, list) or len(key_cache) != len(value_cache):
        raise RuntimeError("invalid serialized prompt cache")
    legacy_cache = []
    for layer_idx, (k, v) in enumerate(zip(key_cache, value_cache, strict=False)):
        if k is None or v is None:
            raise RuntimeError(f"missing prompt cache layer {layer_idx}")
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise RuntimeError(f"prompt cache layer {layer_idx} is not tensor-backed")
        if k.ndim == 3:
            k = k.permute(1, 0, 2).unsqueeze(0)
            v = v.permute(1, 0, 2).unsqueeze(0)
        elif k.ndim != 4 or v.ndim != 4:
            raise RuntimeError(
                f"unsupported prompt cache shape at layer {layer_idx}: key={tuple(k.shape)} value={tuple(v.shape)}"
            )
        legacy_cache.append((k.to(device=device).contiguous(), v.to(device=device).contiguous()))
    legacy_cache_tuple = tuple(legacy_cache)
    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(legacy_cache_tuple)
    return DynamicCache(legacy_cache_tuple)


def dumped_omni_kv_cache_path(base_dir: Path, *, call_index: int = 0, sample_index: int = 0) -> Path:
    return base_dir / REQUEST_ID / f"stage0_kv_cache.call{call_index:03d}.sample{sample_index:03d}.pt"


def dumped_reference_kv_cache_path(base_dir: Path, *, call_index: int = 0, sample_index: int = 0) -> Path:
    return base_dir / REQUEST_ID / f"reference_stage0_kv_cache.call{call_index:03d}.sample{sample_index:03d}.pt"


def _tensor_has_non_finite(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.dtype.is_floating_point and not torch.isfinite(value).all()


def _first_bad_step_index(step_records: list[dict[str, object]]) -> int | None:
    for idx, record in enumerate(step_records):
        for key in ("x", "t", "pred", "x_next"):
            if _tensor_has_non_finite(record.get(key)):
                return idx
    return None


def _write_reference_stage1_step_trace_dump(base_dir: Path, step_records: list[dict[str, object]]) -> Path:
    output_path = _build_reference_dump_path(
        base_dir,
        phase="stage1_step_trace",
        identity=DumpIdentity(call_index=0, sample_index=0),
    )
    return _write_json_dump(
        output_path,
        {
            "dump_version": DUMP_VERSION,
            "phase": "stage1_step_trace",
            "req_id": REQUEST_ID,
            "source": "reference",
            "call_index": 0,
            "sample_index": 0,
            "step_index": None,
            "metadata": {},
            "first_bad_step": _first_bad_step_index(step_records),
            "steps": [
                {k: _to_cpu_payload(v) for k, v in record.items()}
                for record in step_records
            ],
        },
    )


def _as_long_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.long).contiguous()
    return torch.as_tensor(value, dtype=torch.long).contiguous()


def _as_output_payload(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.long).contiguous()
    if isinstance(value, list) and value and isinstance(value[0], list):
        return [_as_output_payload(v) for v in value]
    if value is None:
        return None
    return torch.as_tensor(value, dtype=torch.long).contiguous()


def _recommended_gpu_memory_utilization(
    default_cap: float,
    *,
    reserve_ratio: float,
    safety_margin_gib: float = 1.5,
) -> float:
    if not torch.cuda.is_available():
        return default_cap
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    safety_margin_bytes = int(safety_margin_gib * (1024**3))
    adjusted_free_bytes = max(0, int(free_bytes) - safety_margin_bytes)
    free_ratio = float(adjusted_free_bytes) / float(total_bytes)
    return min(default_cap, max(0.05, free_ratio * reserve_ratio))


def actual_request_state_dump_path(base_dir: Path, phase: str) -> Path:
    return build_dump_path(
        base_dir,
        req_id=REQUEST_ID,
        phase=phase,
        identity=DumpIdentity(call_index=0, sample_index=0),
    )


def actual_stage1_transition_dump_path(base_dir: Path) -> Path:
    return build_dump_path(
        base_dir,
        req_id=REQUEST_ID,
        phase="stage1_transition",
        identity=DumpIdentity(call_index=0, sample_index=0),
    )


def actual_stage1_rollout_context_dump_path(base_dir: Path) -> Path:
    return build_dump_path(
        base_dir,
        req_id=REQUEST_ID,
        phase="stage1_rollout_context",
        identity=DumpIdentity(call_index=0, sample_index=0),
    )


def actual_stage1_rollout_step_dump_path(base_dir: Path) -> Path:
    return build_dump_path(
        base_dir,
        req_id=REQUEST_ID,
        phase="stage1_rollout_step",
        identity=DumpIdentity(call_index=0, sample_index=0, step_index=0),
    )


@contextmanager
def capture_stage1_transition_dump(base_dir: Path):
    captured: dict[str, Path | None] = {"path": None}
    original_vlm2trajectory = alp_stage_processors.vlm2trajectory

    def wrapped_vlm2trajectory(*args, **kwargs):
        trajectory_inputs = original_vlm2trajectory(*args, **kwargs)
        try:
            if trajectory_inputs:
                captured["path"] = actual_stage1_transition_dump_path(base_dir)
        except Exception as exc:
            print("failed to capture actual stage1 transition dump:", repr(exc))
        return trajectory_inputs

    alp_stage_processors.vlm2trajectory = wrapped_vlm2trajectory
    try:
        yield captured
    finally:
        alp_stage_processors.vlm2trajectory = original_vlm2trajectory


def write_reference_request_dumps(base_dir: Path) -> dict[str, Path]:
    data, _, messages = load_shared_data()
    tokenizer = build_alpamayo_stage0_tokenizer(ct.MODEL_PATH)
    tokenized = build_alpamayo_fused_tokenized_data(
        ct.MODEL_PATH,
        messages,
        ego_history_xyz=data["ego_history_xyz"],
        ego_history_rot=data["ego_history_rot"],
    )
    prompt_token_ids = tokenized["input_ids"][0].detach().cpu().contiguous()
    attention_mask = tokenized.get("attention_mask")
    image_grid_thw = tokenized.get("image_grid_thw")
    image_grid_thw_list = image_grid_thw.tolist() if image_grid_thw is not None else []
    hf_config = load_reference_hf_config()
    reference_mrope_positions, reference_mrope_delta = OmniMRotaryEmbedding.get_input_positions_tensor(
        input_tokens=prompt_token_ids.tolist(),
        hf_config=hf_config,
        image_grid_thw=image_grid_thw_list,
        video_grid_thw=[],
        second_per_grid_ts=[],
    )
    stage0_params, _ = build_stage_params(tokenizer)
    additional_information = _reference_additional_information(data)

    initialized_path = reference_request_state_dump_path(base_dir, "request_state_initialized")
    _write_json_dump(
        initialized_path,
        {
            "dump_version": DUMP_VERSION,
            "phase": "request_state_initialized",
            "req_id": REQUEST_ID,
            "source": "reference",
            "call_index": 0,
            "sample_index": 0,
            "step_index": None,
            "metadata": {},
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": {
                "seed": stage0_params.seed,
                "temperature": stage0_params.temperature,
                "top_p": stage0_params.top_p,
                "top_k": stage0_params.top_k,
                "max_tokens": stage0_params.max_tokens,
                "n": stage0_params.n,
                "stop_token_ids": stage0_params.stop_token_ids,
                "extra_args": stage0_params.extra_args,
            },
            "mrope_positions": reference_mrope_positions.detach().cpu().contiguous(),
            "mrope_position_delta": int(reference_mrope_delta),
            "additional_information": additional_information,
            "extra": {
                "attention_mask": attention_mask.detach().cpu().contiguous() if attention_mask is not None else None,
                "image_grid_thw": image_grid_thw.detach().cpu().contiguous() if image_grid_thw is not None else None,
            },
        },
    )

    batched_path = reference_request_state_dump_path(base_dir, "request_state_batched")
    _write_json_dump(
        batched_path,
        {
            "dump_version": DUMP_VERSION,
            "phase": "request_state_batched",
            "req_id": REQUEST_ID,
            "source": "reference",
            "call_index": 0,
            "sample_index": 0,
            "step_index": None,
            "metadata": {},
            "prompt_token_ids": prompt_token_ids,
            "mrope_positions": reference_mrope_positions.detach().cpu().contiguous(),
            "mrope_position_delta": int(reference_mrope_delta),
            "additional_information": additional_information,
            "extra": {
                "input_batch_prompt_token_ids": prompt_token_ids.clone(),
                "input_batch_num_tokens_no_spec": int(prompt_token_ids.shape[0]),
                "attention_mask": attention_mask.detach().cpu().contiguous() if attention_mask is not None else None,
                "image_grid_thw": image_grid_thw.detach().cpu().contiguous() if image_grid_thw is not None else None,
            },
        },
    )
    return {
        "initialized": initialized_path,
        "batched": batched_path,
    }


def write_reference_stage1_transition_dump(base_dir: Path, stage0_debug: dict[str, object]) -> Path:
    output_path = reference_stage1_transition_dump_path(base_dir)
    return _write_json_dump(
        output_path,
        {
            "dump_version": DUMP_VERSION,
            "phase": "stage1_transition",
            "req_id": REQUEST_ID,
            "source": "reference",
            "call_index": 0,
            "sample_index": 0,
            "step_index": None,
            "metadata": {},
            "stage0_prompt_token_ids": _as_long_tensor(stage0_debug.get("stage0_prompt_token_ids")),
            "stage0_output_token_ids": _as_output_payload(stage0_debug.get("stage0_output_token_ids")),
            "stage0_sequences": _as_output_payload(stage0_debug.get("stage0_sequences")),
            "stage0_rope_deltas": _as_long_tensor(stage0_debug.get("stage0_rope_deltas")),
            "stage0_prefill_seq_len": stage0_debug.get("stage0_prefill_seq_len"),
            "initial_noise_x0": _to_cpu_payload(stage0_debug.get("initial_noise_x0")),
            "stage0_prompt_length": stage0_debug.get("stage0_prompt_length"),
            "stage0_output_length": stage0_debug.get("stage0_output_length"),
            "stage0_num_return_sequences": stage0_debug.get("stage0_num_return_sequences"),
            "stage0_attention_mask": stage0_debug.get("stage0_attention_mask"),
        },
    )


def write_actual_stage1_transition_dump(base_dir: Path, stage0_debug: dict[str, object]) -> Path:
    output_path = actual_stage1_transition_dump_path(base_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_json_dump(
        output_path,
        {
            "dump_version": 1,
            "phase": "stage1_transition",
            "req_id": REQUEST_ID,
            "source": "reference",
            "call_index": 0,
            "sample_index": 0,
            "step_index": None,
            "metadata": {},
            "stage0_prompt_token_ids": _as_long_tensor(stage0_debug.get("stage0_prompt_token_ids")),
            "stage0_output_token_ids": _as_output_payload(stage0_debug.get("stage0_output_token_ids")),
            "stage0_sequences": _as_output_payload(stage0_debug.get("stage0_sequences")),
            "stage0_rope_deltas": _as_long_tensor(stage0_debug.get("stage0_rope_deltas")),
            "stage0_prefill_seq_len": stage0_debug.get("stage0_prefill_seq_len"),
            "initial_noise_x0": _to_cpu_payload(stage0_debug.get("initial_noise_x0")),
            "stage0_prompt_length": stage0_debug.get("stage0_prompt_length"),
            "stage0_output_length": stage0_debug.get("stage0_output_length"),
            "stage0_num_return_sequences": stage0_debug.get("stage0_num_return_sequences"),
        },
    )


def load_stage1_rollout_context_dump(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    return _from_json_payload(payload)


def write_reference_stage1_rollout_context_dump(
    base_dir: Path,
    *,
    prompt_cache_seq_len: int,
    prompt_cache_summary: dict[str, object] | None,
    sequences: torch.Tensor,
    rope_deltas: torch.Tensor | None,
    prefix_mask: torch.Tensor | None,
    initial_noise_x0: torch.Tensor | None,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    offset: torch.Tensor,
    hist_xyz: torch.Tensor,
    hist_rot: torch.Tensor,
    hist_xyz_rep: torch.Tensor,
    hist_rot_rep: torch.Tensor,
) -> Path:
    output_path = reference_stage1_rollout_context_dump_path(base_dir)
    return _write_json_dump(
        output_path,
        {
            "dump_version": DUMP_VERSION,
            "phase": "stage1_rollout_context",
            "req_id": REQUEST_ID,
            "source": "reference",
            "call_index": 0,
            "sample_index": 0,
            "step_index": None,
            "metadata": {},
            "sequence_tensor": sequences.detach().cpu().contiguous(),
            "rope_deltas": None if rope_deltas is None else rope_deltas.detach().cpu().contiguous(),
            "prefix_mask": None if prefix_mask is None else prefix_mask.detach().cpu().contiguous(),
            "initial_noise_x0": None if initial_noise_x0 is None else initial_noise_x0.detach().cpu().contiguous(),
            "prefill_seq_len": int(prompt_cache_seq_len),
            "prompt_cache_summary": prompt_cache_summary,
            "offset": offset.detach().cpu().contiguous(),
            "position_ids": position_ids.detach().cpu().contiguous(),
            "attention_mask": attention_mask.detach().cpu().contiguous(),
            "hist_xyz": hist_xyz.detach().cpu().contiguous(),
            "hist_rot": hist_rot.detach().cpu().contiguous(),
            "hist_xyz_rep": hist_xyz_rep.detach().cpu().contiguous(),
            "hist_rot_rep": hist_rot_rep.detach().cpu().contiguous(),
        },
    )


def write_reference_stage1_rollout_step_dump(base_dir: Path, captured_step: dict[str, torch.Tensor]) -> Path:
    output_path = reference_stage1_rollout_step_dump_path(base_dir)
    return _write_json_dump(
        output_path,
        {
            "dump_version": DUMP_VERSION,
            "phase": "stage1_rollout_step",
            "req_id": REQUEST_ID,
            "source": "reference",
            "call_index": 0,
            "sample_index": 0,
            "step_index": 0,
            "metadata": {},
            "x": captured_step["x"].detach().cpu().contiguous(),
            "t": captured_step["t"].detach().cpu().contiguous(),
            "future_token_embeds": captured_step["future_token_embeds"].detach().cpu().contiguous(),
            "last_hidden": captured_step["last_hidden"].detach().cpu().contiguous(),
            "pred": captured_step["pred"].detach().cpu().contiguous(),
        },
    )


def run_original(
    dump_dir: Path | None = None,
    forced_initial_noise_x0: torch.Tensor | None = None,
    *,
    clip_id: str = ct.CLIP_ID,
    t0_us: int = ct.T0_US,
    num_inference_steps: int = 10,
):
    data, _, messages = load_shared_data(clip_id=clip_id, t0_us=t0_us)

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    model = Alpamayo1_5.from_pretrained(ct.MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")
    fused_prompt_token_ids = model_inputs["tokenized_data"]["input_ids"].detach().cpu().contiguous()

    captured_vlm_outputs: dict[str, object] = {}
    captured_step0: dict[str, torch.Tensor] = {}
    captured_action_in_proj_internal: dict[str, object] = {}
    captured_expert_internal: dict[str, object] = {}
    active_step_capture: dict[str, torch.Tensor] | None = None
    captured_step_trace: list[dict[str, object]] = []
    original_generate = model.vlm.generate
    original_diffusion_sample = model.diffusion.sample
    original_action_in_proj_forward = model.action_in_proj.forward
    original_action_out_proj_forward = model.action_out_proj.forward
    expert_param = next(model.expert.parameters(), None)
    expert_dtype = expert_param.dtype if expert_param is not None else torch.float32

    def wrapped_generate(*args, **kwargs):
        outputs = original_generate(*args, **kwargs)
        captured_vlm_outputs["outputs"] = outputs
        captured_vlm_outputs["prompt_cache_snapshot"] = _serialize_prompt_cache(outputs.past_key_values)
        return outputs

    def wrapped_action_in_proj_forward(*args, **kwargs):
        nonlocal active_step_capture
        output = original_action_in_proj_forward(*args, **kwargs)
        if active_step_capture is not None and "future_token_embeds" not in active_step_capture:
            active_step_capture["future_token_embeds"] = output.detach().cpu().contiguous()
        return output

    def wrapped_action_out_proj_forward(*args, **kwargs):
        nonlocal active_step_capture
        if active_step_capture is not None and args and "last_hidden" not in active_step_capture:
            active_step_capture["last_hidden"] = args[0].detach().cpu().contiguous()
        output = original_action_out_proj_forward(*args, **kwargs)
        if active_step_capture is not None and "pred" not in active_step_capture:
            active_step_capture["pred"] = output.detach().cpu().contiguous()
        return output

    def wrapped_diffusion_sample(*args, **kwargs):
        step_fn = kwargs.get("step_fn")
        if step_fn is None and len(args) >= 2:
            step_fn = args[1]
        if step_fn is None:
            return original_diffusion_sample(*args, **kwargs)

        if forced_initial_noise_x0 is not None:
            batch_size = kwargs.get("batch_size")
            if batch_size is None and args:
                batch_size = args[0]
            if batch_size is None:
                raise RuntimeError("Unable to resolve diffusion batch_size for forced initial_noise_x0")

            unguided_step_fn = kwargs.get("unguided_step_fn")
            if unguided_step_fn is None and len(args) >= 3:
                unguided_step_fn = args[2]
            device = kwargs.get("device", torch.device("cpu"))
            return_all_steps = bool(kwargs.get("return_all_steps", False))
            inference_step = int(kwargs.get("inference_step") or model.diffusion.num_inference_steps)
            use_classifier_free_guidance = kwargs.get("use_classifier_free_guidance")
            if use_classifier_free_guidance is None:
                use_classifier_free_guidance = model.diffusion.use_classifier_free_guidance
            inference_guidance_weight = kwargs.get("inference_guidance_weight")
            if inference_guidance_weight is None:
                inference_guidance_weight = model.diffusion.inference_guidance_weight

            x = forced_initial_noise_x0.detach().to(device=device, dtype=torch.float32).contiguous()
            if x.ndim == len(model.diffusion.x_dims):
                x = x.unsqueeze(0)
            expected_shape = (int(batch_size), *tuple(model.diffusion.x_dims))
            if tuple(x.shape) != expected_shape:
                raise RuntimeError(
                    f"forced_initial_noise_x0 shape mismatch: expected={expected_shape}, got={tuple(x.shape)}"
                )

            time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
            n_dim = len(model.diffusion.x_dims)
            if return_all_steps:
                all_steps = [x]

            def wrapped_step_fn(*, x, t):
                nonlocal active_step_capture
                active_step_capture = {
                    "x": x.detach().cpu().contiguous(),
                    "t": t.detach().cpu().contiguous(),
                }
                autocast_ctx = nullcontext()
                if x.device.type == "cuda" and expert_dtype in (torch.float16, torch.bfloat16):
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=expert_dtype)
                with autocast_ctx:
                    pred = step_fn(x=x, t=t)
                active_step_capture.setdefault("pred", pred.detach().cpu().contiguous())
                if not captured_step0 and {"x", "t", "future_token_embeds", "last_hidden", "pred"} <= set(active_step_capture):
                    captured_step0.update(active_step_capture)
                active_step_capture = None
                return pred

            for i in range(inference_step):
                dt = time_steps[i + 1] - time_steps[i]
                dt = dt.view(1, *[1] * n_dim).expand(int(batch_size), *[1] * n_dim)
                t_start = time_steps[i].view(1, *[1] * n_dim).expand(int(batch_size), *[1] * n_dim)
                if use_classifier_free_guidance:
                    if unguided_step_fn is None:
                        raise RuntimeError("unguided_step_fn is required for classifier-free guidance")
                    v = model.diffusion._guided_v(
                        step_fn=wrapped_step_fn,
                        x=x,
                        t=t_start,
                        unguided_step_fn=unguided_step_fn,
                        inference_guidance_weight=float(inference_guidance_weight),
                    )
                else:
                    v = wrapped_step_fn(x=x, t=t_start)
                x_next = x + dt * v
                captured_step_trace.append(
                    {
                        "step_index": i,
                        "x": x.detach().cpu().contiguous(),
                        "t": t_start.detach().cpu().contiguous(),
                        "pred": v.detach().cpu().contiguous(),
                        "x_next": x_next.detach().cpu().contiguous(),
                    }
                )
                x = x_next
                if return_all_steps:
                    all_steps.append(x)

            if return_all_steps:
                return torch.stack(all_steps, dim=1), time_steps
            return x

        def wrapped_step_fn(*, x, t):
            nonlocal active_step_capture
            active_step_capture = {
                "x": x.detach().cpu().contiguous(),
                "t": t.detach().cpu().contiguous(),
            }
            autocast_ctx = nullcontext()
            if x.device.type == "cuda" and expert_dtype in (torch.float16, torch.bfloat16):
                autocast_ctx = torch.autocast(device_type="cuda", dtype=expert_dtype)
            with autocast_ctx:
                pred = step_fn(x=x, t=t)
            active_step_capture.setdefault("pred", pred.detach().cpu().contiguous())
            if not captured_step0 and {"x", "t", "future_token_embeds", "last_hidden", "pred"} <= set(active_step_capture):
                captured_step0.update(active_step_capture)
            active_step_capture = None
            return pred

        if "step_fn" in kwargs:
            kwargs = dict(kwargs)
            kwargs["step_fn"] = wrapped_step_fn
            return original_diffusion_sample(*args, **kwargs)

        args = list(args)
        args[1] = wrapped_step_fn
        return original_diffusion_sample(*args, **kwargs)

    model.vlm.generate = wrapped_generate
    model.diffusion.sample = wrapped_diffusion_sample
    model.action_in_proj.forward = wrapped_action_in_proj_forward
    model.action_out_proj.forward = wrapped_action_out_proj_forward

    try:
        with capture_action_in_proj_internal(model.action_in_proj, captured_action_in_proj_internal):
            with capture_expert_internal(model.expert, captured_expert_internal):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=0.98,
                        top_k=40,
                        temperature=0.6,
                        num_traj_samples=1,
                        max_generation_length=256,
                        diffusion_kwargs={"inference_step": int(num_inference_steps)},
                        return_extra=True,
                    )
    finally:
        model.vlm.generate = original_generate
        model.diffusion.sample = original_diffusion_sample
        model.action_in_proj.forward = original_action_in_proj_forward
        model.action_out_proj.forward = original_action_out_proj_forward

    vlm_outputs = captured_vlm_outputs.get("outputs")
    if vlm_outputs is None:
        raise RuntimeError("Failed to capture original stage-0 VLM outputs")
    eos_token_id = model.tokenizer.convert_tokens_to_ids(to_special_token("traj_future_start"))
    sequences = replace_padding_after_eos(
        token_ids=vlm_outputs.sequences.detach().clone(),
        eos_token_id=eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    )
    prompt_len = int(fused_prompt_token_ids.shape[-1])
    stage0_sequences = sequences[0].detach().cpu().contiguous()
    stage0_prompt_token_ids = stage0_sequences[:prompt_len]
    stage0_output_token_ids = stage0_sequences[prompt_len:]
    stage0_debug = {
        "stage0_prompt_token_ids": stage0_prompt_token_ids,
        "stage0_output_token_ids": stage0_output_token_ids,
        "stage0_sequences": stage0_sequences,
        "stage0_rope_deltas": getattr(vlm_outputs, "rope_deltas", None),
        "stage0_prefill_seq_len": int(vlm_outputs.past_key_values.get_seq_length()),
        "initial_noise_x0": captured_step0.get("x"),
        "stage0_prompt_length": prompt_len,
        "stage0_output_length": int(stage0_output_token_ids.shape[-1]),
        "stage0_num_return_sequences": int(sequences.shape[0]),
        "stage0_attention_mask": model_inputs["tokenized_data"]["attention_mask"].detach().cpu().contiguous(),
    }
    stage1_prompt_info = {
        **_reference_additional_information(data),
        "tokenized_data": {
            "input_ids": stage0_sequences.clone(),
            "attention_mask": model_inputs["tokenized_data"]["attention_mask"].detach().cpu().contiguous(),
        },
        "stage0_prompt_token_ids": stage0_prompt_token_ids.clone(),
        "stage0_output_token_ids": stage0_output_token_ids.clone(),
        "stage0_sequences": stage0_sequences.clone(),
        "stage0_rope_deltas": _to_cpu_payload(getattr(vlm_outputs, "rope_deltas", None)),
        "stage0_attention_mask": model_inputs["tokenized_data"]["attention_mask"].detach().cpu().contiguous(),
        "stage0_prefill_seq_len": int(vlm_outputs.past_key_values.get_seq_length()),
        "stage0_prompt_length": prompt_len,
        "stage0_output_length": int(stage0_output_token_ids.shape[-1]),
        "stage0_num_return_sequences": int(sequences.shape[0]),
        "num_return_sequences": int(sequences.shape[0]),
        "initial_noise_x0": _to_cpu_payload(captured_step0.get("x")),
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": 7.5,
    }
    prompt_cache_snapshot = captured_vlm_outputs.get("prompt_cache_snapshot")
    if dump_dir is not None:
        prompt_cache = prompt_cache_snapshot or _serialize_prompt_cache(vlm_outputs.past_key_values)
        reference_kv_dump_path = dumped_reference_kv_cache_path(dump_dir)
        reference_kv_dump_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "req_id": REQUEST_ID,
                "call_index": 0,
                "sample_index": 0,
                "seq_len": int(vlm_outputs.past_key_values.get_seq_length()),
                "num_layers": 0 if prompt_cache is None else len(getattr(prompt_cache, "key_cache", []) or []),
                "key_cache": None if prompt_cache is None else getattr(prompt_cache, "key_cache", None),
                "value_cache": None if prompt_cache is None else getattr(prompt_cache, "value_cache", None),
            },
            reference_kv_dump_path,
        )
        prompt_cache_seq_len = int(vlm_outputs.past_key_values.get_seq_length())
        prefix_mask = model_inputs["tokenized_data"].get("attention_mask")
        hist_xyz = model_inputs["ego_history_xyz"][:, -1]
        hist_rot = model_inputs["ego_history_rot"][:, -1]
        offset = model._find_eos_offset(
            sequences=sequences,
            eos_token_id=eos_token_id,
            device=sequences.device,
        )
        position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
            offset=offset,
            rope_deltas=vlm_outputs.rope_deltas,
            kv_cache_seq_len=prompt_cache_seq_len,
            n_diffusion_tokens=model.action_space.get_action_space_dims()[0],
            b_star=sequences.shape[0],
            device=sequences.device,
            prefix_mask=prefix_mask,
        )
        write_reference_stage1_rollout_context_dump(
            dump_dir,
            prompt_cache_seq_len=prompt_cache_seq_len,
            prompt_cache_summary=_summarize_prompt_cache(prompt_cache),
            sequences=sequences,
            rope_deltas=vlm_outputs.rope_deltas,
            prefix_mask=prefix_mask,
            initial_noise_x0=captured_step0.get("x"),
            position_ids=position_ids,
            attention_mask=attention_mask,
            offset=offset,
            hist_xyz=hist_xyz,
            hist_rot=hist_rot,
            hist_xyz_rep=hist_xyz,
            hist_rot_rep=hist_rot,
        )
        if {"x", "t", "future_token_embeds", "last_hidden", "pred"} <= set(captured_step0):
            write_reference_stage1_rollout_step_dump(dump_dir, captured_step0)
        if captured_action_in_proj_internal:
            write_action_in_proj_internal_dump(
                dump_dir,
                source="reference",
                captured_debug=captured_action_in_proj_internal,
            )
        if captured_expert_internal:
            write_expert_internal_dump(
                dump_dir,
                source="reference",
                captured_debug=captured_expert_internal,
            )
        if captured_step_trace:
            _write_reference_stage1_step_trace_dump(dump_dir, captured_step_trace)

    result = {
        "pred_xyz": pred_xyz.detach().cpu()[0],
        "pred_rot": pred_rot.detach().cpu()[0],
        "cot_text": str(extra["cot"][0, 0, 0]),
        "stage0_debug": _to_cpu_payload(stage0_debug),
        "stage1_prompt_info": _to_cpu_payload(stage1_prompt_info),
        "prompt_cache": prompt_cache_snapshot or _serialize_prompt_cache(vlm_outputs.past_key_values),
        "model_inputs": _to_cpu_payload(model_inputs),
    }

    del sequences
    del stage0_sequences
    del stage0_prompt_token_ids
    del stage0_output_token_ids
    del stage0_debug
    del fused_prompt_token_ids
    del vlm_outputs
    del captured_vlm_outputs
    del captured_step0
    del extra
    del pred_xyz
    del pred_rot
    del model_inputs
    del inputs
    del processor
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    return result


def run_reference_stage1_with_omni_kv(
    original: dict[str, object],
    *,
    kv_dump_path: Path,
    dump_dir: Path | None = None,
) -> dict[str, object]:
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    model = Alpamayo1_5.from_pretrained(ct.MODEL_PATH, dtype=torch.bfloat16).to("cuda")
    prompt_cache_serialized = load_dumped_omni_prompt_cache(kv_dump_path)
    prompt_cache = build_dynamic_cache_from_serialized_prompt_cache(
        prompt_cache_serialized,
        device="cuda",
    )
    stage0_debug = original["stage0_debug"]
    model_inputs = helper.to_device(copy.deepcopy(original["model_inputs"]), "cuda")
    sequences = torch.as_tensor(stage0_debug["stage0_sequences"], device="cuda", dtype=torch.long).unsqueeze(0)
    rope_deltas = torch.as_tensor(stage0_debug["stage0_rope_deltas"], device="cuda", dtype=torch.long)
    prefix_mask = model_inputs["tokenized_data"].get("attention_mask")
    initial_noise_x0 = torch.as_tensor(stage0_debug["initial_noise_x0"], device="cuda", dtype=torch.float32)
    external_rollout_context = None
    if dump_dir is not None:
        rollout_context_path = actual_stage1_rollout_context_dump_path(dump_dir)
        if rollout_context_path.is_file():
            external_rollout_context = load_stage1_rollout_context_dump(rollout_context_path)

    captured_step0: dict[str, torch.Tensor] = {}
    captured_action_in_proj_internal: dict[str, object] = {}
    captured_expert_internal: dict[str, object] = {}
    active_step_capture: dict[str, torch.Tensor] | None = None
    original_diffusion_sample = model.diffusion.sample
    original_action_in_proj_forward = model.action_in_proj.forward
    original_action_out_proj_forward = model.action_out_proj.forward
    expert_param = next(model.expert.parameters(), None)
    expert_dtype = expert_param.dtype if expert_param is not None else torch.float32

    def wrapped_action_in_proj_forward(*args, **kwargs):
        nonlocal active_step_capture
        output = original_action_in_proj_forward(*args, **kwargs)
        if active_step_capture is not None and "future_token_embeds" not in active_step_capture:
            active_step_capture["future_token_embeds"] = output.detach().cpu().contiguous()
        return output

    def wrapped_action_out_proj_forward(*args, **kwargs):
        nonlocal active_step_capture
        if active_step_capture is not None and args and "last_hidden" not in active_step_capture:
            active_step_capture["last_hidden"] = args[0].detach().cpu().contiguous()
        output = original_action_out_proj_forward(*args, **kwargs)
        if active_step_capture is not None and "pred" not in active_step_capture:
            active_step_capture["pred"] = output.detach().cpu().contiguous()
        return output

    def wrapped_diffusion_sample(*args, **kwargs):
        step_fn = kwargs.get("step_fn")
        if step_fn is None and len(args) >= 2:
            step_fn = args[1]
        if step_fn is None:
            return original_diffusion_sample(*args, **kwargs)
        batch_size = kwargs.get("batch_size")
        if batch_size is None and args:
            batch_size = args[0]
        if batch_size is None:
            raise RuntimeError("Unable to resolve diffusion batch_size for forced initial_noise_x0")
        device = kwargs.get("device", torch.device("cpu"))
        return_all_steps = bool(kwargs.get("return_all_steps", False))
        inference_step = int(kwargs.get("inference_step") or model.diffusion.num_inference_steps)
        use_classifier_free_guidance = kwargs.get("use_classifier_free_guidance")
        if use_classifier_free_guidance is None:
            use_classifier_free_guidance = model.diffusion.use_classifier_free_guidance
        inference_guidance_weight = kwargs.get("inference_guidance_weight")
        if inference_guidance_weight is None:
            inference_guidance_weight = model.diffusion.inference_guidance_weight

        x = initial_noise_x0.detach().to(device=device, dtype=torch.float32).contiguous()
        if x.ndim == len(model.diffusion.x_dims):
            x = x.unsqueeze(0)
        expected_shape = (int(batch_size), *tuple(model.diffusion.x_dims))
        if tuple(x.shape) != expected_shape:
            raise RuntimeError(
                f"forced_initial_noise_x0 shape mismatch: expected={expected_shape}, got={tuple(x.shape)}"
            )
        time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)
        n_dim = len(model.diffusion.x_dims)
        if return_all_steps:
            all_steps = [x]

        def wrapped_step_fn(*, x, t):
            nonlocal active_step_capture
            active_step_capture = {
                "x": x.detach().cpu().contiguous(),
                "t": t.detach().cpu().contiguous(),
            }
            autocast_ctx = nullcontext()
            if x.device.type == "cuda" and expert_dtype in (torch.float16, torch.bfloat16):
                autocast_ctx = torch.autocast(device_type="cuda", dtype=expert_dtype)
            with autocast_ctx:
                pred = step_fn(x=x, t=t)
            active_step_capture.setdefault("pred", pred.detach().cpu().contiguous())
            if not captured_step0 and {"x", "t", "future_token_embeds", "last_hidden", "pred"} <= set(active_step_capture):
                captured_step0.update(active_step_capture)
            active_step_capture = None
            return pred

        for i in range(inference_step):
            dt = time_steps[i + 1] - time_steps[i]
            dt = dt.view(1, *[1] * n_dim).expand(int(batch_size), *[1] * n_dim)
            t_start = time_steps[i].view(1, *[1] * n_dim).expand(int(batch_size), *[1] * n_dim)
            if use_classifier_free_guidance:
                raise RuntimeError("reference stage1-only omni-KV replay does not support CFG yet")
            v = wrapped_step_fn(x=x, t=t_start)
            x = x + dt * v
            if return_all_steps:
                all_steps.append(x)
        if return_all_steps:
            return torch.stack(all_steps, dim=1), time_steps
        return x

    model.diffusion.sample = wrapped_diffusion_sample
    model.action_in_proj.forward = wrapped_action_in_proj_forward
    model.action_out_proj.forward = wrapped_action_out_proj_forward
    try:
        with capture_action_in_proj_internal(model.action_in_proj, captured_action_in_proj_internal):
            with capture_expert_internal(model.expert, captured_expert_internal):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred_xyz, pred_rot = model.sample_trajectories_from_stage1_context(
                        data=model_inputs,
                        sequences=sequences,
                        rope_deltas=rope_deltas,
                        prompt_cache=prompt_cache,
                        prefix_mask=prefix_mask,
                        offset=None
                        if external_rollout_context is None
                        else torch.as_tensor(external_rollout_context["offset"], device="cuda", dtype=torch.long),
                        position_ids=None
                        if external_rollout_context is None
                        else torch.as_tensor(external_rollout_context["position_ids"], device="cuda", dtype=torch.long),
                        attention_mask=None
                        if external_rollout_context is None
                        else torch.as_tensor(external_rollout_context["attention_mask"], device="cuda"),
                        diffusion_kwargs={"inference_step": int(original["stage1_prompt_info"]["num_inference_steps"])},
                        num_traj_samples=1,
                        num_traj_sets=1,
                    )
    finally:
        model.diffusion.sample = original_diffusion_sample
        model.action_in_proj.forward = original_action_in_proj_forward
        model.action_out_proj.forward = original_action_out_proj_forward

    if dump_dir is not None:
        prompt_cache_seq_len = int(prompt_cache.get_seq_length())
        hist_xyz = model_inputs["ego_history_xyz"][:, -1]
        hist_rot = model_inputs["ego_history_rot"][:, -1]
        if external_rollout_context is None:
            offset = model._find_eos_offset(
                sequences=sequences,
                eos_token_id=model.tokenizer.eos_token_id,
                device=sequences.device,
            )
            position_ids, attention_mask = model._build_expert_pos_ids_and_attn_mask(
                offset=offset,
                rope_deltas=rope_deltas,
                kv_cache_seq_len=prompt_cache_seq_len,
                n_diffusion_tokens=model.action_space.get_action_space_dims()[0],
                b_star=sequences.shape[0],
                device=sequences.device,
                prefix_mask=prefix_mask,
            )
            hist_xyz_rep = hist_xyz
            hist_rot_rep = hist_rot
        else:
            offset = torch.as_tensor(external_rollout_context["offset"], device=sequences.device, dtype=torch.long)
            position_ids = torch.as_tensor(external_rollout_context["position_ids"], device=sequences.device, dtype=torch.long)
            attention_mask = torch.as_tensor(external_rollout_context["attention_mask"], device=sequences.device)
            hist_xyz_rep = torch.as_tensor(external_rollout_context.get("hist_xyz_rep", hist_xyz), device=hist_xyz.device, dtype=hist_xyz.dtype)
            hist_rot_rep = torch.as_tensor(external_rollout_context.get("hist_rot_rep", hist_rot), device=hist_rot.device, dtype=hist_rot.dtype)
        write_reference_stage1_rollout_context_dump(
            dump_dir,
            prompt_cache_seq_len=prompt_cache_seq_len,
            prompt_cache_summary=_summarize_prompt_cache(prompt_cache),
            sequences=sequences,
            rope_deltas=rope_deltas,
            prefix_mask=prefix_mask,
            initial_noise_x0=initial_noise_x0,
            position_ids=position_ids,
            attention_mask=attention_mask,
            offset=offset,
            hist_xyz=hist_xyz,
            hist_rot=hist_rot,
            hist_xyz_rep=hist_xyz_rep,
            hist_rot_rep=hist_rot_rep,
        )
        if {"x", "t", "future_token_embeds", "last_hidden", "pred"} <= set(captured_step0):
            write_reference_stage1_rollout_step_dump(dump_dir, captured_step0)
        if captured_action_in_proj_internal:
            write_action_in_proj_internal_dump(
                dump_dir,
                source="reference",
                captured_debug=captured_action_in_proj_internal,
            )
        if captured_expert_internal:
            write_expert_internal_dump(
                dump_dir,
                source="reference",
                captured_debug=captured_expert_internal,
            )

    result = {
        "pred_xyz": pred_xyz.detach().cpu()[0],
        "pred_rot": pred_rot.detach().cpu()[0],
        "prompt_cache": prompt_cache_serialized,
        "captured_step0": captured_step0,
        "captured_action_in_proj_internal": captured_action_in_proj_internal,
        "captured_expert_internal": captured_expert_internal,
        "mode": "reference_stage1_with_omni_kv",
    }

    del pred_xyz
    del pred_rot
    del prompt_cache
    del model_inputs
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    return result


def get_omni_prompt_token_count(
    *,
    clip_id: str = ct.CLIP_ID,
    t0_us: int = ct.T0_US,
) -> int:
    data, _, messages = load_shared_data(clip_id=clip_id, t0_us=t0_us)
    fused_tokenized = build_alpamayo_fused_tokenized_data(
        ct.MODEL_PATH,
        messages,
        ego_history_xyz=data["ego_history_xyz"],
        ego_history_rot=data["ego_history_rot"],
    )
    return int(fused_tokenized["input_ids"].shape[-1])



def create_omni_for_compare(
    *,
    prompt_token_count: int,
    requested_max_tokens: int = 256,
) -> tuple[AsyncOmni, str]:
    yaml_path = ct.build_two_stage_yaml(ct.YAML_PATH)
    yaml_config = yaml.safe_load(Path(yaml_path).read_text())
    stage0_engine_args = yaml_config["stage_args"][0].setdefault("engine_args", {})
    stage1_engine_args = yaml_config["stage_args"][1].setdefault("engine_args", {})
    resolved_prompt_token_count = int(prompt_token_count)
    resolved_requested_max_tokens = int(requested_max_tokens)
    debug_max_model_len = max(
        3328,
        ((resolved_prompt_token_count + resolved_requested_max_tokens + 127) // 128) * 128,
    )
    debug_max_batched_tokens = max(
        resolved_prompt_token_count,
        min(debug_max_model_len, 3584),
    )

    stage0_engine_args["gpu_memory_utilization"] = _recommended_gpu_memory_utilization(
        min(float(stage0_engine_args.get("gpu_memory_utilization", 0.85)), 0.58),
        reserve_ratio=1,
    )
    stage0_engine_args["skip_mm_profiling"] = True
    stage0_engine_args["max_model_len"] = min(
        int(stage0_engine_args.get("max_model_len", debug_max_model_len)),
        debug_max_model_len,
    )
    stage0_engine_args["max_num_batched_tokens"] = min(
        int(stage0_engine_args.get("max_num_batched_tokens", debug_max_batched_tokens)),
        debug_max_batched_tokens,
    )
    stage1_engine_args["gpu_memory_utilization"] = _recommended_gpu_memory_utilization(
        min(float(stage1_engine_args.get("gpu_memory_utilization", 0.15)), 0.10),
        reserve_ratio=0.2,
    )
    Path(yaml_path).write_text(yaml.safe_dump(yaml_config, sort_keys=False))
    print(
        "create_omni_for_compare debug config:",
        {
            "prompt_token_count": resolved_prompt_token_count,
            "stage0_max_tokens": resolved_requested_max_tokens,
            "stage0_gpu_memory_utilization": stage0_engine_args["gpu_memory_utilization"],
            "stage0_max_model_len": stage0_engine_args["max_model_len"],
            "stage0_max_num_batched_tokens": stage0_engine_args["max_num_batched_tokens"],
            "stage1_gpu_memory_utilization": stage1_engine_args["gpu_memory_utilization"],
        },
    )
    return AsyncOmni(model=ct.MODEL_PATH, stage_configs_path=yaml_path), yaml_path


async def run_omni(
    dump_dir: Path | None = None,
    initial_noise_x0: torch.Tensor | None = None,
    override_kv_path: Path | None = None,
    *,
    clip_id: str = ct.CLIP_ID,
    t0_us: int = ct.T0_US,
    num_inference_steps: int = 10,
    omni: AsyncOmni | None = None,
):
    data, frames, messages = load_shared_data(clip_id=clip_id, t0_us=t0_us)
    tokenizer = build_alpamayo_stage0_tokenizer(ct.MODEL_PATH)
    stage0_params, stage1_params = build_stage_params(tokenizer)
    stage1_params.num_inference_steps = int(num_inference_steps)
    prompt_text = build_alpamayo_stage0_prompt_text(
        ct.MODEL_PATH,
        messages,
        tokenizer=tokenizer,
    )
    owned_yaml_path: str | None = None
    owns_omni = omni is None
    if omni is None:
        prompt_token_count = get_omni_prompt_token_count(clip_id=clip_id, t0_us=t0_us)
        omni, owned_yaml_path = create_omni_for_compare(
            prompt_token_count=prompt_token_count,
            requested_max_tokens=int(getattr(stage0_params, "max_tokens", 256) or 256),
        )

    prompt = {
        "prompt": prompt_text,
        "multi_modal_data": {
            "image": [ct.tensor_to_pil(frame) for frame in frames],
        },
        "additional_information": {
            **_reference_additional_information(data),
            **(
                {
                    "initial_noise_x0": initial_noise_x0.detach().cpu().to(torch.float32).contiguous()
                }
                if initial_noise_x0 is not None
                else {}
            ),
        },
    }

    captured_action_in_proj_internal: dict[str, torch.Tensor] = {}
    previous_env = {
        "VLLM_OMNI_REQUEST_DUMP_DIR": os.environ.get("VLLM_OMNI_REQUEST_DUMP_DIR"),
        "VLLM_OMNI_REQUEST_DUMP_REQ_IDS": os.environ.get("VLLM_OMNI_REQUEST_DUMP_REQ_IDS"),
        "VLLM_OMNI_REQUEST_DUMP_PHASES": os.environ.get("VLLM_OMNI_REQUEST_DUMP_PHASES"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"),
        "VLLM_OMNI_ALPAMAYO_KV_DEBUG_DIR": os.environ.get("VLLM_OMNI_ALPAMAYO_KV_DEBUG_DIR"),
        "VLLM_OMNI_ALPAMAYO_OVERRIDE_KV_PATH": os.environ.get("VLLM_OMNI_ALPAMAYO_OVERRIDE_KV_PATH"),
    }
    if dump_dir is not None:
        os.environ["VLLM_OMNI_REQUEST_DUMP_DIR"] = str(dump_dir)
        os.environ["VLLM_OMNI_REQUEST_DUMP_REQ_IDS"] = REQUEST_ID
        os.environ["VLLM_OMNI_REQUEST_DUMP_PHASES"] = "request_state_initialized,request_state_batched"
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"] = str(dump_dir)
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"] = REQUEST_ID
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"] = "stage1_transition,stage1_rollout_context,stage1_rollout_step,action_in_proj_internal,expert_internal,stage0_kv_sender_probe"
        os.environ["VLLM_OMNI_ALPAMAYO_KV_DEBUG_DIR"] = str(dump_dir)
    if override_kv_path is not None:
        os.environ["VLLM_OMNI_ALPAMAYO_OVERRIDE_KV_PATH"] = str(override_kv_path)

    try:
        with capture_runtime_action_in_proj_internal(captured_action_in_proj_internal):
            final_output = None
            async for out in omni.generate(
                prompt=prompt,
                request_id=REQUEST_ID,
                sampling_params_list=[stage0_params, stage1_params],
            ):
                final_output = out
        if final_output is None:
            raise RuntimeError("no final output")

        if dump_dir is not None and captured_action_in_proj_internal:
            write_action_in_proj_internal_dump(
                dump_dir,
                source="omni",
                captured_debug=captured_action_in_proj_internal,
            )

        custom = final_output.custom_output
        if dump_dir is not None:
            transition_path = actual_stage1_transition_dump_path(dump_dir)
            if transition_path.is_file():
                transition_payload = json.loads(transition_path.read_text())
                log_stage1_transition_hashes(
                    "omni_with_override_kv" if override_kv_path is not None else "omni",
                    transition_payload,
                )
        cot_token_ids = custom["cot_token_ids"].detach().cpu()
        result = {
            "pred_xyz": custom["pred_xyz"].detach().cpu(),
            "pred_rot": custom["pred_rot"].detach().cpu(),
            "cot_token_ids": cot_token_ids,
            "cot_text_raw": tokenizer.decode(cot_token_ids[0, 0].tolist(), skip_special_tokens=False),
            "stage0_context_used": custom.get("stage0_context_used"),
        }
        result["cot_text"] = extract_cot_body(result["cot_text_raw"])
        return result
    finally:
        if owns_omni and omni is not None:
            omni.shutdown()
        if owned_yaml_path is not None:
            Path(owned_yaml_path).unlink(missing_ok=True)
        for env_name, env_value in previous_env.items():
            if env_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = env_value


async def run_omni_stage1_only(
    original: dict[str, object],
    dump_dir: Path | None = None,
):
    yaml_path = ct.build_two_stage_yaml(ct.YAML_PATH)
    yaml_config = yaml.safe_load(Path(yaml_path).read_text())
    stage1_cfg = dict(yaml_config["stage_args"][1].get("engine_args", {}))
    stage1_cfg["gpu_memory_utilization"] = min(float(stage1_cfg.get("gpu_memory_utilization", 0.15)), 0.15)

    tokenizer = build_alpamayo_stage0_tokenizer(ct.MODEL_PATH)
    _, stage1_params = build_stage_params(tokenizer)
    stage1_info = _to_cpu_payload(original["stage1_prompt_info"])
    if not isinstance(stage1_info, dict):
        raise RuntimeError("original stage1_prompt_info must be a dict")

    stage1_params.need_kv_receive = False
    stage1_params.past_key_values = original["prompt_cache"]
    stage1_params.seed = 42
    stage1_params.num_outputs_per_prompt = int(stage1_info.get("num_return_sequences") or 1)
    stage1_params.num_inference_steps = int(stage1_info.get("num_inference_steps") or 10)
    stage1_params.guidance_scale = float(stage1_info.get("guidance_scale") or 7.5)

    prompt = {
        "prompt": "dummy run",
        "additional_information": stage1_info,
    }

    captured_action_in_proj_internal: dict[str, torch.Tensor] = {}
    previous_env = {
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"),
        "VLLM_OMNI_ALPAMAYO_OVERRIDE_KV_PATH": os.environ.get("VLLM_OMNI_ALPAMAYO_OVERRIDE_KV_PATH"),
    }
    if dump_dir is not None:
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"] = str(dump_dir)
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"] = REQUEST_ID
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"] = "stage1_transition,stage1_rollout_context,stage1_rollout_step,action_in_proj_internal,expert_internal,stage0_kv_sender_probe"
        os.environ["VLLM_OMNI_ALPAMAYO_OVERRIDE_KV_PATH"] = str(dumped_reference_kv_cache_path(dump_dir))
        write_actual_stage1_transition_dump(dump_dir, original["stage0_debug"])

    diffusion = AsyncOmniDiffusion(model=ct.MODEL_PATH, batch_size=1, **stage1_cfg)
    try:
        with capture_runtime_action_in_proj_internal(captured_action_in_proj_internal):
            result = await diffusion.generate(
                prompt=prompt,
                sampling_params=stage1_params,
                request_id=REQUEST_ID,
            )
        if dump_dir is not None and captured_action_in_proj_internal:
            write_action_in_proj_internal_dump(
                dump_dir,
                source="omni",
                captured_debug=captured_action_in_proj_internal,
            )
        custom = result.custom_output
        cot_token_ids = custom["cot_token_ids"].detach().cpu()
        output = {
            "pred_xyz": custom["pred_xyz"].detach().cpu(),
            "pred_rot": custom["pred_rot"].detach().cpu(),
            "cot_token_ids": cot_token_ids,
            "cot_text_raw": tokenizer.decode(cot_token_ids[0, 0].tolist(), skip_special_tokens=False),
            "stage0_context_used": custom.get("stage0_context_used"),
            "mode": "stage1_only",
        }
        output["cot_text"] = extract_cot_body(output["cot_text_raw"])
        return output
    finally:
        diffusion.close()
        Path(yaml_path).unlink(missing_ok=True)
        for env_name, env_value in previous_env.items():
            if env_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = env_value


def write_compare_summary_dump(
    base_dir: Path,
    *,
    comparison_target: str,
    lhs_path: Path | None,
    rhs_path: Path | None,
    keys: list[str],
    status: str,
    diff_lines: list[str],
) -> Path:
    sanitized_target = comparison_target.lower().replace(" ", "_")
    output_path = _build_reference_dump_path(
        base_dir,
        phase=f"compare_summary_{sanitized_target}",
        identity=DumpIdentity(call_index=0, sample_index=0),
    )
    return _write_json_dump(
        output_path,
        {
            "dump_version": DUMP_VERSION,
            "phase": "compare_summary",
            "req_id": REQUEST_ID,
            "source": "reference",
            "call_index": 0,
            "sample_index": 0,
            "step_index": None,
            "metadata": {},
            "comparison_target": comparison_target,
            "lhs_path": None if lhs_path is None else str(lhs_path),
            "rhs_path": None if rhs_path is None else str(rhs_path),
            "keys": keys,
            "status": status,
            "diff_count": len(diff_lines),
            "diff_lines": diff_lines,
        },
    )


def print_dump_compare_report(
    *,
    base_dir: Path,
    comparison_target: str,
    lhs_path: Path | None,
    rhs_path: Path | None,
    keys: list[str],
) -> dict[str, object]:
    diff_lines: list[str] = []
    status = "missing"
    if lhs_path is not None and rhs_path is not None and lhs_path.is_file() and rhs_path.is_file():
        diff_lines = compare_dump_files(lhs_path, rhs_path, keys=keys)
        status = "match" if not diff_lines else "diff"
    print(f"COMPARE phase={comparison_target} status={status} diff_count={len(diff_lines)}")
    if lhs_path is not None:
        print("lhs:", lhs_path)
    if rhs_path is not None:
        print("rhs:", rhs_path)
    if diff_lines:
        for diff in diff_lines:
            print("-", diff)
    write_compare_summary_dump(
        base_dir,
        comparison_target=comparison_target,
        lhs_path=lhs_path,
        rhs_path=rhs_path,
        keys=keys,
        status=status,
        diff_lines=diff_lines,
    )
    return {
        "comparison_target": comparison_target,
        "status": status,
        "diff_count": len(diff_lines),
        "diff_lines": diff_lines,
    }


async def main():
    dump_dir = make_compare_run_dir()
    reference_dumps = write_reference_request_dumps(dump_dir)
    original = run_original(dump_dir=dump_dir)
    reference_stage1_dump = write_reference_stage1_transition_dump(
        dump_dir,
        original["stage0_debug"],
    )
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    # sleeping for a short time to ensure gpu memory is freed after original run before starting omni, to reduce risk of OOM during omni run
    await asyncio.sleep(5)
    actual_stage1_capture: dict[str, Path | None]
    try:
        with capture_stage1_transition_dump(dump_dir) as actual_stage1_capture:
            omni = await run_omni(dump_dir=dump_dir)
    except RuntimeError as exc:
        print(
            "\nfull two-stage AsyncOmni run failed in the current GPU environment; "
            "falling back to stage1-only replay"
        )
        print("fallback trigger:", exc)
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        await asyncio.sleep(2)
        omni = await run_omni_stage1_only(original, dump_dir=dump_dir)
        actual_stage1_capture = {"path": actual_stage1_transition_dump_path(dump_dir)}

    print("original pred_xyz.shape:", tuple(original["pred_xyz"].shape))
    print("original pred_rot.shape:", tuple(original["pred_rot"].shape))
    print("original cot:", original["cot_text"])

    print("omni pred_xyz.shape:", tuple(omni["pred_xyz"].shape))
    print("omni pred_rot.shape:", tuple(omni["pred_rot"].shape))
    print("omni stage0_context_used:", omni["stage0_context_used"])
    print("omni cot raw:", omni["cot_text_raw"])
    print("omni cot extracted:", omni["cot_text"])

    compare_tensors("pred_xyz", original["pred_xyz"], omni["pred_xyz"])
    compare_tensors("pred_rot", original["pred_rot"], omni["pred_rot"])
    print("COMPARE cot_text_equal:", original["cot_text"] == omni["cot_text"])

    actual_initialized = actual_request_state_dump_path(dump_dir, "request_state_initialized")
    actual_batched = actual_request_state_dump_path(dump_dir, "request_state_batched")
    if omni.get("mode") != "stage1_only":
        if actual_initialized.is_file():
            print_dump_compare_report(
                comparison_target="REQUEST STATE INITIALIZED DUMP COMPARE",
                lhs_path=reference_dumps["initialized"],
                rhs_path=actual_initialized,
                keys=[
                    "prompt_token_ids",
                    "mrope_positions",
                    "mrope_position_delta",
                    "sampling_params",
                    "additional_information",
                ],
            )
        else:
            print("\nREQUEST STATE INITIALIZED DUMP COMPARE")
            print("missing actual dump:", actual_initialized)

        if actual_batched.is_file():
            print_dump_compare_report(
                comparison_target="REQUEST STATE BATCHED DUMP COMPARE",
                lhs_path=reference_dumps["batched"],
                rhs_path=actual_batched,
                keys=[
                    "prompt_token_ids",
                    "mrope_positions",
                    "mrope_position_delta",
                    "additional_information",
                    "extra.input_batch_prompt_token_ids",
                    "extra.input_batch_num_tokens_no_spec",
                ],
            )
        else:
            print("\nREQUEST STATE BATCHED DUMP COMPARE")
            print("missing actual dump:", actual_batched)
    else:
        print("\nREQUEST STATE INITIALIZED DUMP COMPARE")
        print("skipped in stage1-only fallback mode")
        print("\nREQUEST STATE BATCHED DUMP COMPARE")
        print("skipped in stage1-only fallback mode")

    actual_stage1_dump = actual_stage1_capture.get("path")
    if actual_stage1_dump is not None and Path(actual_stage1_dump).is_file():
        print_dump_compare_report(
            comparison_target="STAGE1 TRANSITION DUMP COMPARE",
            lhs_path=reference_stage1_dump,
            rhs_path=Path(actual_stage1_dump),
            keys=[
                "stage0_prompt_token_ids",
                "stage0_output_token_ids",
                "stage0_sequences",
                "stage0_rope_deltas",
                "stage0_prefill_seq_len",
                "initial_noise_x0",
                "stage0_prompt_length",
                "stage0_output_length",
                "stage0_num_return_sequences",
            ],
        )
    else:
        print("\nSTAGE1 TRANSITION DUMP COMPARE")
        print("missing actual dump:", actual_stage1_dump)

    actual_rollout_context = actual_stage1_rollout_context_dump_path(dump_dir)
    reference_rollout_context = reference_stage1_rollout_context_dump_path(dump_dir)
    if actual_rollout_context.is_file() and reference_rollout_context.is_file():
        print_dump_compare_report(
            comparison_target="STAGE1 ROLLOUT CONTEXT DUMP COMPARE",
            lhs_path=reference_rollout_context,
            rhs_path=actual_rollout_context,
            keys=[
                "sequence_tensor",
                "rope_deltas",
                "prefix_mask",
                "initial_noise_x0",
                "prefill_seq_len",
                "offset",
                "position_ids",
                "attention_mask",
                "hist_xyz",
                "hist_rot",
                "hist_xyz_rep",
                "hist_rot_rep",
            ],
        )
    else:
        print("\nSTAGE1 ROLLOUT CONTEXT DUMP COMPARE")
        print("missing reference dump:", reference_rollout_context)
        print("missing actual dump:", actual_rollout_context)

    actual_rollout_step0 = actual_stage1_rollout_step_dump_path(dump_dir)
    reference_rollout_step0 = reference_stage1_rollout_step_dump_path(dump_dir)
    if actual_rollout_step0.is_file() and reference_rollout_step0.is_file():
        print_dump_compare_report(
            comparison_target="STAGE1 ROLLOUT STEP0 DUMP COMPARE",
            lhs_path=reference_rollout_step0,
            rhs_path=actual_rollout_step0,
            keys=[
                "x",
                "t",
                "future_token_embeds",
                "last_hidden",
                "pred",
            ],
        )
    else:
        print("\nSTAGE1 ROLLOUT STEP0 DUMP COMPARE")
        print("missing reference dump:", reference_rollout_step0)
        print("missing actual dump:", actual_rollout_step0)

    reference_action_in_proj_internal = reference_action_in_proj_internal_dump_path(dump_dir)
    actual_action_in_proj_internal = actual_action_in_proj_internal_dump_path(dump_dir)
    if actual_action_in_proj_internal.is_file() and reference_action_in_proj_internal.is_file():
        print_dump_compare_report(
            comparison_target="ACTION IN PROJ INTERNAL DUMP COMPARE",
            lhs_path=reference_action_in_proj_internal,
            rhs_path=actual_action_in_proj_internal,
            keys=[
                "x_in",
                "t_in",
                "action_input_0",
                "action_input_1",
                "freqs_action_0",
                "freqs_action_1",
                "freqs_t",
                "fourier_action_0",
                "fourier_action_1",
                "fourier_t",
                "action_feats",
                "timestep_feats",
                "concat_before_mlp",
                "mlp_in",
                "mlp_layer_00",
                "mlp_layer_01",
                "mlp_layer_02",
                "mlp_layer_03",
                "mlp_layer_04",
                "mlp_layer_05",
                "mlp_layer_06",
                "encoder_out",
                "encoder_out_reshaped",
                "norm_out",
                "future_token_embeds",
            ],
        )
    else:
        print("\nACTION IN PROJ INTERNAL DUMP COMPARE")
        print("missing reference dump:", reference_action_in_proj_internal)
        print("missing actual dump:", actual_action_in_proj_internal)

    print("\ndump_dir:", dump_dir)


if __name__ == "__main__":
    asyncio.run(main())
