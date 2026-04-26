import asyncio
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from vllm import SamplingParams

sys.path.insert(0, "/workspace/project/RL-learning/vllm-omni")
sys.path.insert(0, "/workspace/project/RL-learning/alpamayo1.5/src")

from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset_local import load_physical_aiavdataset
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.alpamayo1_5 import (
    build_alpamayo_stage0_prompt_text,
    build_alpamayo_stage0_tokenizer,
)

MODEL_PATH = "/share/models/Alpamayo-1.5-10B"
YAML_PATH = "/workspace/project/RL-learning/vllm-omni/vllm_omni/deploy/alpamayo1_5.yaml"
DATASET_PATH = "/share/datasets/ncore_10clips"
CLIP_ID = "100ae358-f548-49b8-af4d-c0afdbcfe9ed"
T0_US = 5_100_000


def get_clip_msg(clip_id: str) -> dict:
    json_path = f"{DATASET_PATH}/train.jsonl"
    with open(json_path, "r") as f:
        for line in f:
            item = json.loads(line)
            if item["clip_id"] == clip_id:
                return item
    raise ValueError(f"clip_id not found: {clip_id}")


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    x = frame.detach().cpu()
    if x.ndim != 3:
        raise ValueError(f"unexpected frame shape: {tuple(x.shape)}")
    if x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    if x.dtype != torch.uint8:
        if x.max() <= 1.0:
            x = (x * 255).clamp(0, 255)
        else:
            x = x.clamp(0, 255)
        x = x.to(torch.uint8)
    return Image.fromarray(x.numpy())


def build_prompt_messages(
    camera_indices: torch.Tensor,
    num_frames_per_camera: int,
) -> list[dict]:
    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )
    user_text = (
        f"{hist_traj_placeholder}"
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )

    content: list[dict] = []
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


def build_two_stage_yaml(src_yaml_path: str) -> str:
    with open(src_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    stage0 = dict(config["stage_args"][0])
    stage1 = dict(config["stage_args"][1])

    stage0_engine_args = dict(stage0.get("engine_args") or {})
    stage0_engine_args["gpu_memory_utilization"] = 0.65
    stage0_engine_args["max_num_seqs"] = 1
    stage0_engine_args["max_num_batched_tokens"] = 4096
    stage0_engine_args["max_model_len"] = 4096
    stage0["engine_args"] = stage0_engine_args

    stage1_engine_args = dict(stage1.get("engine_args") or {})
    stage1_engine_args["gpu_memory_utilization"] = 0.15
    stage1_engine_args["max_num_seqs"] = 1
    stage1["engine_args"] = stage1_engine_args

    config["stage_args"] = [stage0, stage1]
    runtime_cfg = dict(config.get("runtime") or {})
    runtime_defaults = dict(runtime_cfg.get("defaults") or {})
    runtime_defaults["max_inflight"] = 1
    runtime_cfg["defaults"] = runtime_defaults
    config["runtime"] = runtime_cfg

    fd, temp_path = tempfile.mkstemp(prefix="alpamayo_2stage_", suffix=".yaml")
    Path(temp_path).write_text(yaml.safe_dump(config, sort_keys=False))
    return temp_path


async def main():
    two_stage_yaml_path = build_two_stage_yaml(YAML_PATH)
    clip_msg = get_clip_msg(CLIP_ID)
    data = load_physical_aiavdataset(
        clip_id=CLIP_ID,
        t0_us=T0_US,
        ncore_manifest_path=clip_msg["ncore_manifest_path"],
        ncore_root=clip_msg["ncore_root"],
        extract_cache_dir=clip_msg["extract_cache_dir"],
    )

    image_frames = data["image_frames"]
    frames = image_frames.flatten(0, 1)
    prompt_messages = build_prompt_messages(
        camera_indices=data["camera_indices"],
        num_frames_per_camera=int(image_frames.shape[1]),
    )
    tokenizer = build_alpamayo_stage0_tokenizer(MODEL_PATH)
    future_start_token_id = int(tokenizer.traj_token_ids["future_start"])
    pad_token_id = int(tokenizer.pad_token_id)
    prompt_text = build_alpamayo_stage0_prompt_text(
        MODEL_PATH,
        prompt_messages,
        tokenizer=tokenizer,
    )

    prompt = {
        "prompt": prompt_text,
        "multi_modal_data": {
            "image": [tensor_to_pil(frame) for frame in frames],
        },
        "additional_information": {
            "ego_history_xyz": data["ego_history_xyz"].cpu(),
            "ego_history_rot": data["ego_history_rot"].cpu(),
            "alpamayo_model_path": MODEL_PATH,
        },
    }

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

    omni = AsyncOmni(
        model=MODEL_PATH,
        stage_configs_path=two_stage_yaml_path,
    )

    try:
        final_output = None
        async for out in omni.generate(
            prompt=prompt,
            request_id="alpamayo-2stage-test",
            sampling_params_list=[stage0_params, stage1_params],
        ):
            final_output = out

        if final_output is None:
            raise RuntimeError("no final output")

        custom = final_output.custom_output
        pred_xyz = custom["pred_xyz"]
        pred_rot = custom["pred_rot"]
        cot_ids = custom["cot_token_ids"]

        print("stage0_context_used:", custom.get("stage0_context_used"))
        print("pred_xyz.shape:", tuple(pred_xyz.shape))
        print("pred_rot.shape:", tuple(pred_rot.shape))
        print("cot_token_ids.shape:", tuple(cot_ids.shape))

        cot_text = tokenizer.decode(cot_ids[0, 0].tolist(), skip_special_tokens=False)
        print("\nCOT text:\n", cot_text)

        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()

        pred_xyz_np = pred_xyz.detach().cpu().numpy()
        if pred_xyz_np.ndim == 5:
            pred_xy = pred_xyz_np[0, 0, :, :, :2].transpose(0, 2, 1)
        elif pred_xyz_np.ndim == 4:
            pred_xy = pred_xyz_np[0, :, :, :2].transpose(0, 2, 1)
        else:
            raise ValueError(f"unexpected pred_xyz shape: {pred_xyz_np.shape}")

        ade = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        print("\nminADE:", float(ade.min()), "meters")
    finally:
        omni.shutdown()
        Path(two_stage_yaml_path).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
