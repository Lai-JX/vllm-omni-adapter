import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
from vllm import SamplingParams

sys.path.insert(0, "/workspace/project/RL-learning/vllm-omni")
sys.path.insert(0, "/workspace/project/RL-learning/alpamayo1.5/src")

from tests.diffusion.models.alpamoya.custom_test import alpamoya_2stage_test as t
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.alpamayo1_5 import (
    build_alpamayo_stage0_prompt_text,
    build_alpamayo_stage0_tokenizer,
)


def build_request():
    two_stage_yaml_path = t.build_two_stage_yaml(t.YAML_PATH)
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
    prompt_messages = t.build_prompt_messages(
        camera_indices=data["camera_indices"],
        num_frames_per_camera=int(image_frames.shape[1]),
    )
    tokenizer = build_alpamayo_stage0_tokenizer(t.MODEL_PATH)
    future_start_token_id = int(tokenizer.traj_token_ids["future_start"])
    pad_token_id = int(tokenizer.pad_token_id)
    prompt_text = build_alpamayo_stage0_prompt_text(
        t.MODEL_PATH,
        prompt_messages,
        tokenizer=tokenizer,
    )

    prompt = {
        "prompt": prompt_text,
        "multi_modal_data": {
            "image": [t.tensor_to_pil(frame) for frame in frames],
        },
        "additional_information": {
            "ego_history_xyz": data["ego_history_xyz"].cpu(),
            "ego_history_rot": data["ego_history_rot"].cpu(),
            "alpamayo_model_path": t.MODEL_PATH,
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
    return two_stage_yaml_path, tokenizer, prompt, stage0_params, stage1_params


async def run_once(tag: str):
    yaml_path, tokenizer, prompt, stage0_params, stage1_params = build_request()
    omni = AsyncOmni(model=t.MODEL_PATH, stage_configs_path=yaml_path)
    try:
        final_output = None
        async for out in omni.generate(
            prompt=prompt,
            request_id=f"alpamayo-repro-{tag}",
            sampling_params_list=[stage0_params, stage1_params],
        ):
            final_output = out
        if final_output is None:
            raise RuntimeError("no final output")

        custom = final_output.custom_output
        result = {
            "pred_xyz": custom["pred_xyz"].detach().cpu(),
            "pred_rot": custom["pred_rot"].detach().cpu(),
            "cot_token_ids": custom["cot_token_ids"].detach().cpu(),
            "stage0_context_used": custom.get("stage0_context_used"),
            "cot_text": tokenizer.decode(
                custom["cot_token_ids"][0, 0].tolist(),
                skip_special_tokens=False,
            ),
        }
        print(f"{tag}: stage0_context_used={result['stage0_context_used']}")
        print(f"{tag}: pred_xyz.shape={tuple(result['pred_xyz'].shape)}")
        print(f"{tag}: pred_rot.shape={tuple(result['pred_rot'].shape)}")
        print(f"{tag}: cot_token_ids.shape={tuple(result['cot_token_ids'].shape)}")
        print(f"{tag}: cot_text={result['cot_text']}")
        return result
    finally:
        omni.shutdown()
        Path(yaml_path).unlink(missing_ok=True)


def compare_tensors(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    equal = torch.equal(lhs, rhs)
    if lhs.dtype.is_floating_point:
        max_abs_diff = float((lhs - rhs).abs().max().item()) if lhs.numel() else 0.0
    else:
        max_abs_diff = 0.0 if equal else float((lhs != rhs).sum().item())
    print(f"COMPARE {name}: equal={equal} max_abs_diff={max_abs_diff}")


async def main():
    torch.manual_seed(42)
    np.random.seed(42)

    out1 = await run_once("run1")
    out2 = await run_once("run2")

    compare_tensors("pred_xyz", out1["pred_xyz"], out2["pred_xyz"])
    compare_tensors("pred_rot", out1["pred_rot"], out2["pred_rot"])
    compare_tensors("cot_token_ids", out1["cot_token_ids"], out2["cot_token_ids"])


if __name__ == "__main__":
    asyncio.run(main())
