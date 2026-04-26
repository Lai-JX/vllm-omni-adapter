import asyncio
import gc
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from transformers import AutoConfig
from vllm import SamplingParams

sys.path.insert(0, "/workspace/project/RL-learning/vllm-omni")
sys.path.insert(0, "/workspace/project/RL-learning/alpamayo1.5/src")

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.models.alpamayo1_5 import ExpertLogitsProcessor
from alpamayo1_5.models.token_utils import StopAfterEOS, replace_padding_after_eos, to_special_token
from tests.diffusion.models.alpamoya.custom_test import alpamoya_2stage_test as t
from vllm_omni.debug.compare_request_state_dump import compare_dump_files
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.layers.rotary_embedding.mrope import OmniMRotaryEmbedding
from vllm_omni.model_executor.stage_input_processors.alpamayo1_5 import (
    build_alpamayo_fused_tokenized_data,
    build_alpamayo_stage0_prompt_text,
    build_alpamayo_stage0_tokenizer,
)
from vllm_omni.model_executor.stage_input_processors import alpamayo1_5 as alp_stage_processors

REQUEST_ID = "alpamayo-compare-original"
FIXED_X0_SEED = 20260425
FIXED_X0_SHAPE = (1, 64, 2)


def load_shared_data():
    clip_msg = t.get_clip_msg(t.CLIP_ID)
    data = t.load_physical_aiavdataset(
        clip_id=t.CLIP_ID,
        t0_us=t.T0_US,
        ncore_manifest_path=clip_msg["ncore_manifest_path"],
        ncore_root=clip_msg["ncore_root"],
        extract_cache_dir=clip_msg["extract_cache_dir"],
    )
    image_frames = data["image_frames"]
    frames = image_frames.flatten(0, 1)
    messages = helper.create_message(
        frames=frames,
        camera_indices=data["camera_indices"],
    )
    return data, frames, messages


def extract_cot_body(text: str) -> str:
    start_token = "<|cot_start|>"
    end_token = "<|cot_end|>"
    if start_token in text:
        text = text.split(start_token, 1)[1]
    if end_token in text:
        text = text.split(end_token, 1)[0]
    return text.strip()


def compare_tensors(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    equal = torch.equal(lhs, rhs)
    same_shape = tuple(lhs.shape) == tuple(rhs.shape)
    max_abs_diff = None
    if lhs.shape == rhs.shape and lhs.dtype.is_floating_point and rhs.dtype.is_floating_point:
        max_abs_diff = float((lhs - rhs).abs().max().item()) if lhs.numel() else 0.0
    elif lhs.shape == rhs.shape:
        max_abs_diff = 0.0 if equal else float((lhs != rhs).sum().item())
    print(
        f"COMPARE {name}: "
        f"same_shape={same_shape} "
        f"equal={equal} "
        f"max_abs_diff={max_abs_diff}"
    )


def build_fixed_initial_noise_x0(
    *,
    seed: int = FIXED_X0_SEED,
    shape: tuple[int, int, int] = FIXED_X0_SHAPE,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, dtype=torch.float32).contiguous()


def build_stage_params(tokenizer):
    future_start_token_id = int(tokenizer.traj_token_ids["future_start"])
    pad_token_id = int(tokenizer.pad_token_id)
    stage0_params = SamplingParams(
        temperature=0.6,
        top_p=0.98,
        top_k=40,
        max_tokens=256,
        stop_token_ids=[pad_token_id],
        detokenize=False,
        seed=42,
        n=1,
        extra_args={
            "alpamayo_stop_after_token_id": future_start_token_id,
            "alpamayo_forced_stop_token_id": pad_token_id,
        },
    )
    stage1_params = OmniDiffusionSamplingParams(
        seed=42,
        num_outputs_per_prompt=1,
        num_inference_steps=10,
        guidance_scale=7.5,
    )
    return stage0_params, stage1_params


def _reference_additional_information(data):
    return {
        "ego_history_xyz": data["ego_history_xyz"].cpu(),
        "ego_history_rot": data["ego_history_rot"].cpu(),
        "alpamayo_model_path": t.MODEL_PATH,
    }


def load_reference_hf_config():
    config_path = Path(t.MODEL_PATH) / "config.json"
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


def _serialize_prompt_cache(prompt_cache):
    key_cache = getattr(prompt_cache, "key_cache", None)
    value_cache = getattr(prompt_cache, "value_cache", None)
    if not isinstance(key_cache, list) or not isinstance(value_cache, list):
        return None
    return SimpleNamespace(
        key_cache=[None if t is None else t.detach().cpu().contiguous() for t in key_cache],
        value_cache=[None if t is None else t.detach().cpu().contiguous() for t in value_cache],
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


@contextmanager
def capture_stage1_transition_dump(base_dir: Path):
    captured: dict[str, Path | None] = {"path": None}
    original_vlm2trajectory = alp_stage_processors.vlm2trajectory

    def wrapped_vlm2trajectory(*args, **kwargs):
        trajectory_inputs = original_vlm2trajectory(*args, **kwargs)
        try:
            if trajectory_inputs:
                info = dict((trajectory_inputs[0] or {}).get("additional_information") or {})
                dump_path = base_dir / REQUEST_ID / "actual_stage1_transition.pt"
                dump_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "phase": "actual_stage1_transition",
                        "req_id": REQUEST_ID,
                        "stage0_prompt_token_ids": _as_long_tensor(info.get("stage0_prompt_token_ids")),
                        "stage0_output_token_ids": _as_output_payload(info.get("stage0_output_token_ids")),
                        "stage0_sequences": _as_output_payload(info.get("stage0_sequences")),
                        "stage0_rope_deltas": _as_long_tensor(info.get("stage0_rope_deltas")),
                        "stage0_prefill_seq_len": info.get("stage0_prefill_seq_len"),
                        "initial_noise_x0": _to_cpu_payload(info.get("initial_noise_x0")),
                        "stage0_prompt_length": info.get("stage0_prompt_length"),
                        "stage0_output_length": info.get("stage0_output_length"),
                        "stage0_output_lengths": _to_cpu_payload(info.get("stage0_output_lengths")),
                        "stage0_num_return_sequences": info.get("stage0_num_return_sequences"),
                    },
                    dump_path,
                )
                captured["path"] = dump_path
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
    tokenizer = build_alpamayo_stage0_tokenizer(t.MODEL_PATH)
    tokenized = build_alpamayo_fused_tokenized_data(
        t.MODEL_PATH,
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

    ref_dir = base_dir / REQUEST_ID
    ref_dir.mkdir(parents=True, exist_ok=True)

    initialized_path = ref_dir / "reference_request_state_initialized.pt"
    torch.save(
        {
            "phase": "reference_request_state_initialized",
            "req_id": REQUEST_ID,
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
        initialized_path,
    )

    batched_path = ref_dir / "reference_request_state_batched.pt"
    torch.save(
        {
            "phase": "reference_request_state_batched",
            "req_id": REQUEST_ID,
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
        batched_path,
    )
    return {
        "initialized": initialized_path,
        "batched": batched_path,
    }


def write_reference_stage1_transition_dump(base_dir: Path, stage0_debug: dict[str, object]) -> Path:
    ref_dir = base_dir / REQUEST_ID
    ref_dir.mkdir(parents=True, exist_ok=True)
    output_path = ref_dir / "reference_stage1_transition.pt"
    torch.save(
        {
            "phase": "reference_stage1_transition",
            "req_id": REQUEST_ID,
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
        output_path,
    )
    return output_path


def write_actual_stage1_transition_dump(base_dir: Path, stage0_debug: dict[str, object]) -> Path:
    ref_dir = base_dir / REQUEST_ID
    ref_dir.mkdir(parents=True, exist_ok=True)
    output_path = ref_dir / "actual_stage1_transition.pt"
    torch.save(
        {
            "phase": "actual_stage1_transition",
            "req_id": REQUEST_ID,
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
        output_path,
    )
    return output_path


def write_reference_stage1_rollout_context_dump(
    base_dir: Path,
    *,
    prompt_cache_seq_len: int,
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
    ref_dir = base_dir / REQUEST_ID
    ref_dir.mkdir(parents=True, exist_ok=True)
    output_path = ref_dir / "reference_stage1_rollout_context.pt"
    torch.save(
        {
            "phase": "reference_stage1_rollout_context",
            "req_id": REQUEST_ID,
            "sequence_tensor": sequences.detach().cpu().contiguous(),
            "rope_deltas": None if rope_deltas is None else rope_deltas.detach().cpu().contiguous(),
            "prefix_mask": None if prefix_mask is None else prefix_mask.detach().cpu().contiguous(),
            "initial_noise_x0": None if initial_noise_x0 is None else initial_noise_x0.detach().cpu().contiguous(),
            "prefill_seq_len": int(prompt_cache_seq_len),
            "offset": offset.detach().cpu().contiguous(),
            "position_ids": position_ids.detach().cpu().contiguous(),
            "attention_mask": attention_mask.detach().cpu().contiguous(),
            "hist_xyz": hist_xyz.detach().cpu().contiguous(),
            "hist_rot": hist_rot.detach().cpu().contiguous(),
            "hist_xyz_rep": hist_xyz_rep.detach().cpu().contiguous(),
            "hist_rot_rep": hist_rot_rep.detach().cpu().contiguous(),
        },
        output_path,
    )
    return output_path


def write_reference_stage1_rollout_step0_dump(base_dir: Path, captured_step0: dict[str, torch.Tensor]) -> Path:
    ref_dir = base_dir / REQUEST_ID
    ref_dir.mkdir(parents=True, exist_ok=True)
    output_path = ref_dir / "reference_stage1_rollout_step0.pt"
    torch.save(
        {
            "phase": "reference_stage1_rollout_step0",
            "req_id": REQUEST_ID,
            "x": captured_step0["x"].detach().cpu().contiguous(),
            "t": captured_step0["t"].detach().cpu().contiguous(),
            "future_token_embeds": captured_step0["future_token_embeds"].detach().cpu().contiguous(),
            "last_hidden": captured_step0["last_hidden"].detach().cpu().contiguous(),
            "pred": captured_step0["pred"].detach().cpu().contiguous(),
        },
        output_path,
    )
    return output_path


def run_original(
    dump_dir: Path | None = None,
    forced_initial_noise_x0: torch.Tensor | None = None,
):
    data, _, messages = load_shared_data()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)

    model = Alpamayo1_5.from_pretrained(t.MODEL_PATH, dtype=torch.bfloat16).to("cuda")
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
    original_generate = model.vlm.generate
    original_diffusion_sample = model.diffusion.sample
    original_action_in_proj_forward = model.action_in_proj.forward
    original_action_out_proj_forward = model.action_out_proj.forward

    def wrapped_generate(*args, **kwargs):
        outputs = original_generate(*args, **kwargs)
        captured_vlm_outputs["outputs"] = outputs
        return outputs

    def wrapped_action_in_proj_forward(*args, **kwargs):
        output = original_action_in_proj_forward(*args, **kwargs)
        if "future_token_embeds" not in captured_step0:
            captured_step0["future_token_embeds"] = output.detach().cpu().contiguous()
        return output

    def wrapped_action_out_proj_forward(*args, **kwargs):
        if args and "last_hidden" not in captured_step0:
            captured_step0["last_hidden"] = args[0].detach().cpu().contiguous()
        output = original_action_out_proj_forward(*args, **kwargs)
        if "pred" not in captured_step0:
            captured_step0["pred"] = output.detach().cpu().contiguous()
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
                if "x" not in captured_step0:
                    captured_step0["x"] = x.detach().cpu().contiguous()
                if "t" not in captured_step0:
                    captured_step0["t"] = t.detach().cpu().contiguous()
                pred = step_fn(x=x, t=t)
                if "pred" not in captured_step0:
                    captured_step0["pred"] = pred.detach().cpu().contiguous()
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
                x = x + dt * v
                if return_all_steps:
                    all_steps.append(x)

            if return_all_steps:
                return torch.stack(all_steps, dim=1), time_steps
            return x

        def wrapped_step_fn(*, x, t):
            if "x" not in captured_step0:
                captured_step0["x"] = x.detach().cpu().contiguous()
            if "t" not in captured_step0:
                captured_step0["t"] = t.detach().cpu().contiguous()
            pred = step_fn(x=x, t=t)
            if "pred" not in captured_step0:
                captured_step0["pred"] = pred.detach().cpu().contiguous()
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                top_k=40,
                temperature=0.6,
                num_traj_samples=1,
                max_generation_length=256,
                diffusion_kwargs={"inference_step": 10},
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
        "num_inference_steps": 10,
        "guidance_scale": 7.5,
    }
    if dump_dir is not None:
        prompt_cache = vlm_outputs.past_key_values
        prompt_cache_seq_len = int(prompt_cache.get_seq_length())
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
            write_reference_stage1_rollout_step0_dump(dump_dir, captured_step0)

    result = {
        "pred_xyz": pred_xyz.detach().cpu()[0],
        "pred_rot": pred_rot.detach().cpu()[0],
        "cot_text": str(extra["cot"][0, 0, 0]),
        "stage0_debug": _to_cpu_payload(stage0_debug),
        "stage1_prompt_info": _to_cpu_payload(stage1_prompt_info),
        "prompt_cache": _serialize_prompt_cache(vlm_outputs.past_key_values),
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


async def run_omni(
    dump_dir: Path | None = None,
    initial_noise_x0: torch.Tensor | None = None,
):
    data, frames, messages = load_shared_data()
    yaml_path = t.build_two_stage_yaml(t.YAML_PATH)
    yaml_config = yaml.safe_load(Path(yaml_path).read_text())
    tokenizer = build_alpamayo_stage0_tokenizer(t.MODEL_PATH)
    stage0_params, stage1_params = build_stage_params(tokenizer)
    prompt_text = build_alpamayo_stage0_prompt_text(
        t.MODEL_PATH,
        messages,
        tokenizer=tokenizer,
    )
    fused_tokenized = build_alpamayo_fused_tokenized_data(
        t.MODEL_PATH,
        messages,
        ego_history_xyz=data["ego_history_xyz"],
        ego_history_rot=data["ego_history_rot"],
    )
    prompt_token_count = int(fused_tokenized["input_ids"].shape[-1])

    stage0_engine_args = yaml_config["stage_args"][0].setdefault("engine_args", {})
    stage1_engine_args = yaml_config["stage_args"][1].setdefault("engine_args", {})
    requested_max_tokens = int(getattr(stage0_params, "max_tokens", 256) or 256)
    debug_max_model_len = max(
        3328,
        ((prompt_token_count + requested_max_tokens + 127) // 128) * 128,
    )
    debug_max_batched_tokens = max(
        prompt_token_count,
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
        "run_omni debug config:",
        {
            "prompt_token_count": prompt_token_count,
            "stage0_max_tokens": requested_max_tokens,
            "stage0_gpu_memory_utilization": stage0_engine_args["gpu_memory_utilization"],
            "stage0_max_model_len": stage0_engine_args["max_model_len"],
            "stage0_max_num_batched_tokens": stage0_engine_args["max_num_batched_tokens"],
            "stage1_gpu_memory_utilization": stage1_engine_args["gpu_memory_utilization"],
        },
    )

    prompt = {
        "prompt": prompt_text,
        "multi_modal_data": {
            "image": [t.tensor_to_pil(frame) for frame in frames],
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

    previous_env = {
        "VLLM_OMNI_REQUEST_DUMP_DIR": os.environ.get("VLLM_OMNI_REQUEST_DUMP_DIR"),
        "VLLM_OMNI_REQUEST_DUMP_REQ_IDS": os.environ.get("VLLM_OMNI_REQUEST_DUMP_REQ_IDS"),
        "VLLM_OMNI_REQUEST_DUMP_PHASES": os.environ.get("VLLM_OMNI_REQUEST_DUMP_PHASES"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"),
    }
    if dump_dir is not None:
        os.environ["VLLM_OMNI_REQUEST_DUMP_DIR"] = str(dump_dir)
        os.environ["VLLM_OMNI_REQUEST_DUMP_REQ_IDS"] = REQUEST_ID
        os.environ["VLLM_OMNI_REQUEST_DUMP_PHASES"] = "request_state_initialized,request_state_batched"
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"] = str(dump_dir)
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"] = REQUEST_ID
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"] = "stage1_rollout_context,stage1_rollout_step0"

    omni = AsyncOmni(model=t.MODEL_PATH, stage_configs_path=yaml_path)
    try:
        final_output = None
        async for out in omni.generate(
            prompt=prompt,
            request_id=REQUEST_ID,
            sampling_params_list=[stage0_params, stage1_params],
        ):
            final_output = out
        if final_output is None:
            raise RuntimeError("no final output")

        custom = final_output.custom_output
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
        omni.shutdown()
        Path(yaml_path).unlink(missing_ok=True)
        for env_name, env_value in previous_env.items():
            if env_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = env_value


async def run_omni_stage1_only(
    original: dict[str, object],
    dump_dir: Path | None = None,
):
    yaml_path = t.build_two_stage_yaml(t.YAML_PATH)
    yaml_config = yaml.safe_load(Path(yaml_path).read_text())
    stage1_cfg = dict(yaml_config["stage_args"][1].get("engine_args", {}))
    stage1_cfg["gpu_memory_utilization"] = min(float(stage1_cfg.get("gpu_memory_utilization", 0.15)), 0.15)

    tokenizer = build_alpamayo_stage0_tokenizer(t.MODEL_PATH)
    _, stage1_params = build_stage_params(tokenizer)
    stage1_params.need_kv_receive = False
    stage1_params.past_key_values = original["prompt_cache"]

    prompt = {
        "prompt": "",
        "additional_information": _to_cpu_payload(original["stage1_prompt_info"]),
    }

    previous_env = {
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"),
        "VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES": os.environ.get("VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"),
    }
    if dump_dir is not None:
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_DIR"] = str(dump_dir)
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_REQ_IDS"] = REQUEST_ID
        os.environ["VLLM_OMNI_ALPAMAYO_STAGE1_DUMP_PHASES"] = "stage1_rollout_context,stage1_rollout_step0"
        write_actual_stage1_transition_dump(dump_dir, original["stage0_debug"])

    diffusion = AsyncOmniDiffusion(model=t.MODEL_PATH, batch_size=1, **stage1_cfg)
    try:
        result = await diffusion.generate(
            prompt=prompt,
            sampling_params=stage1_params,
            request_id=REQUEST_ID,
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


def print_dump_compare_report(title: str, lhs_path: Path, rhs_path: Path, keys: list[str]) -> None:
    print(f"\n{title}")
    print("lhs:", lhs_path)
    print("rhs:", rhs_path)
    print("keys:", keys)
    diffs = compare_dump_files(lhs_path, rhs_path, keys=keys)
    if not diffs:
        print("No differences found.")
        return
    print(f"Differences found: {len(diffs)}")
    for diff in diffs:
        print("-", diff)


async def main():
    dump_dir = Path(tempfile.mkdtemp(prefix="alpamayo_compare_dump_"))
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
        actual_stage1_capture = {"path": dump_dir / REQUEST_ID / "actual_stage1_transition.pt"}

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

    actual_initialized = dump_dir / REQUEST_ID / "request_state_initialized.pt"
    actual_batched = dump_dir / REQUEST_ID / "request_state_batched.pt"
    if omni.get("mode") != "stage1_only":
        if actual_initialized.is_file():
            print_dump_compare_report(
                title="REQUEST STATE INITIALIZED DUMP COMPARE",
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
                title="REQUEST STATE BATCHED DUMP COMPARE",
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
            title="STAGE1 TRANSITION DUMP COMPARE",
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

    actual_rollout_context = dump_dir / REQUEST_ID / "stage1_rollout_context.pt"
    reference_rollout_context = dump_dir / REQUEST_ID / "reference_stage1_rollout_context.pt"
    if actual_rollout_context.is_file() and reference_rollout_context.is_file():
        print_dump_compare_report(
            title="STAGE1 ROLLOUT CONTEXT DUMP COMPARE",
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

    actual_rollout_step0 = dump_dir / REQUEST_ID / "stage1_rollout_step0.pt"
    reference_rollout_step0 = dump_dir / REQUEST_ID / "reference_stage1_rollout_step0.pt"
    if actual_rollout_step0.is_file() and reference_rollout_step0.is_file():
        print_dump_compare_report(
            title="STAGE1 ROLLOUT STEP0 DUMP COMPARE",
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

    print("\ndump_dir:", dump_dir)


if __name__ == "__main__":
    asyncio.run(main())
