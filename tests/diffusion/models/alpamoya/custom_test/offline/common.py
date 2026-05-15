import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from vllm import SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[6]
ALPAMAYO_SRC = REPO_ROOT.parent / "alpamayo1.5" / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if ALPAMAYO_SRC.is_dir() and str(ALPAMAYO_SRC) not in sys.path:
    sys.path.insert(0, str(ALPAMAYO_SRC))

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset_local import load_physical_aiavdataset
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.alpamayo1_5 import (
    build_alpamayo_stage0_prompt_text,
    build_alpamayo_stage0_tokenizer,
)

MODEL_PATH = "/share/models/Alpamayo-1.5-10B"
YAML_PATH = str(REPO_ROOT / "vllm_omni" / "model_executor" / "stage_configs" / "alpamayo1_5.yaml")
DATASET_PATH = "/share/datasets/ncore_10clips"
CLIP_ID = "100ae358-f548-49b8-af4d-c0afdbcfe9ed"
T0_US = 5_100_000


def get_clip_msg(clip_id: str = CLIP_ID) -> dict[str, Any]:
    json_path = Path(DATASET_PATH) / "train.jsonl"
    with json_path.open("r") as f:
        for line in f:
            item = json.loads(line)
            if item["clip_id"] == clip_id:
                return item
    raise ValueError(f"clip_id not found: {clip_id}")


def load_clip_data(
    clip_id: str = CLIP_ID,
    t0_us: int = T0_US,
) -> dict[str, Any]:
    clip_msg = get_clip_msg(clip_id)
    return load_physical_aiavdataset(
        clip_id=clip_id,
        t0_us=t0_us,
        ncore_manifest_path=clip_msg["ncore_manifest_path"],
        ncore_root=clip_msg["ncore_root"],
        extract_cache_dir=clip_msg["extract_cache_dir"],
    )


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    image = frame.detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"unexpected frame shape: {tuple(image.shape)}")
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    if image.dtype != torch.uint8:
        if image.max() <= 1.0:
            image = (image * 255).clamp(0, 255)
        else:
            image = image.clamp(0, 255)
        image = image.to(torch.uint8)
    return Image.fromarray(image.numpy())


def flatten_frames(data: dict[str, Any]) -> torch.Tensor:
    return data["image_frames"].flatten(0, 1)


def build_prompt_messages(
    camera_indices: torch.Tensor,
    num_frames_per_camera: int,
) -> list[dict[str, Any]]:
    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )
    user_text = (
        f"{hist_traj_placeholder}"
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )

    content: list[dict[str, Any]] = []
    for cam_id_tensor in camera_indices:
        cam_id = int(cam_id_tensor.item())
        cam_name = helper.CAMERA_DISPLAY_NAMES.get(cam_id, f"Camera {cam_id}")
        content.append({"type": "text", "text": f"{cam_name}: "})
        for frame_idx in range(num_frames_per_camera):
            content.append({"type": "text", "text": f"frame {frame_idx} "})
            content.append({"type": "image"})

    content.append({"type": "text", "text": user_text})
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": content,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]


def load_shared_data(
    *,
    use_helper_messages: bool = False,
    clip_id: str = CLIP_ID,
    t0_us: int = T0_US,
) -> tuple[dict[str, Any], torch.Tensor, list[dict[str, Any]]]:
    data = load_clip_data(clip_id=clip_id, t0_us=t0_us)
    frames = flatten_frames(data)
    if use_helper_messages:
        messages = helper.create_message(
            frames=frames,
            camera_indices=data["camera_indices"],
        )
    else:
        messages = build_prompt_messages(
            camera_indices=data["camera_indices"],
            num_frames_per_camera=int(data["image_frames"].shape[1]),
        )
    return data, frames, messages


def build_additional_information(
    data: dict[str, Any],
    *,
    initial_noise_x0: torch.Tensor | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "ego_history_xyz": data["ego_history_xyz"].cpu(),
        "ego_history_rot": data["ego_history_rot"].cpu(),
        "alpamayo_model_path": MODEL_PATH,
    }
    if initial_noise_x0 is not None:
        info["initial_noise_x0"] = initial_noise_x0.detach().cpu().to(torch.float32).contiguous()
    return info


def build_prompt_from_messages(
    messages: list[dict[str, Any]],
    *,
    data: dict[str, Any],
    frames: torch.Tensor,
    tokenizer: Any | None = None,
    initial_noise_x0: torch.Tensor | None = None,
) -> tuple[Any, str]:
    resolved_tokenizer = tokenizer or build_alpamayo_stage0_tokenizer(MODEL_PATH)
    prompt_text = build_alpamayo_stage0_prompt_text(
        MODEL_PATH,
        messages,
        tokenizer=resolved_tokenizer,
    )
    prompt = {
        "prompt": prompt_text,
        "multi_modal_data": {
            "image": [tensor_to_pil(frame) for frame in frames],
        },
        "additional_information": build_additional_information(
            data,
            initial_noise_x0=initial_noise_x0,
        ),
    }
    return prompt, prompt_text


def build_stage_params(tokenizer: Any) -> tuple[SamplingParams, OmniDiffusionSamplingParams]:
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


def build_stage0_only_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.6,
        top_p=0.98,
        top_k=40,
        max_tokens=256,
        stop_token_ids=[155681],
        detokenize=False,
        seed=42,
        n=1,
    )


def _write_temp_yaml(config: dict[str, Any], prefix: str) -> str:
    _, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".yaml")
    Path(temp_path).write_text(yaml.safe_dump(config, sort_keys=False))
    return temp_path


def build_stage0_only_yaml(src_yaml_path: str = YAML_PATH) -> str:
    with open(src_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    stage0 = dict(config["stage_args"][0])
    stage0["final_output"] = True
    stage0["final_output_type"] = "latent"
    stage0.pop("output_connectors", None)

    config["stage_args"] = [stage0]
    runtime_cfg = dict(config.get("runtime") or {})
    runtime_cfg.pop("connectors", None)
    runtime_cfg.pop("edges", None)
    config["runtime"] = runtime_cfg
    return _write_temp_yaml(config, prefix="alpamayo_stage0_")


def build_two_stage_yaml(
    src_yaml_path: str = YAML_PATH,
    *,
    batch_size: int = 1,
) -> str:
    with open(src_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    resolved_batch_size = max(1, int(batch_size))
    stage0 = dict(config["stage_args"][0])
    stage1 = dict(config["stage_args"][1])

    stage0_engine_args = dict(stage0.get("engine_args") or {})
    stage0_engine_args["gpu_memory_utilization"] = 0.65
    stage0_engine_args["max_num_seqs"] = resolved_batch_size
    stage0_engine_args["max_num_batched_tokens"] = 4096 * resolved_batch_size
    stage0_engine_args["max_model_len"] = 4096
    stage0["engine_args"] = stage0_engine_args

    stage1_engine_args = dict(stage1.get("engine_args") or {})
    stage1_engine_args["gpu_memory_utilization"] = 0.15
    stage1_engine_args["max_num_seqs"] = resolved_batch_size
    stage1["engine_args"] = stage1_engine_args

    config["stage_args"] = [stage0, stage1]
    runtime_cfg = dict(config.get("runtime") or {})
    runtime_defaults = dict(runtime_cfg.get("defaults") or {})
    runtime_defaults["max_inflight"] = 1
    runtime_cfg["defaults"] = runtime_defaults
    config["runtime"] = runtime_cfg
    return _write_temp_yaml(config, prefix="alpamayo_2stage_")
