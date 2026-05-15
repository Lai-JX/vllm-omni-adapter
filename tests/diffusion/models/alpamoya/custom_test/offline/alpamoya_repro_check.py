import asyncio
from pathlib import Path

import numpy as np
import torch
from vllm_omni.entrypoints.async_omni import AsyncOmni
from tests.diffusion.models.alpamoya.custom_test.offline.common import (
    MODEL_PATH,
    build_alpamayo_stage0_tokenizer,
    build_prompt_from_messages,
    build_stage_params,
    build_two_stage_yaml,
    load_shared_data,
)


def build_request():
    two_stage_yaml_path = build_two_stage_yaml()
    data, frames, prompt_messages = load_shared_data()
    tokenizer = build_alpamayo_stage0_tokenizer(MODEL_PATH)
    prompt, _ = build_prompt_from_messages(
        prompt_messages,
        data=data,
        frames=frames,
        tokenizer=tokenizer,
    )
    stage0_params, stage1_params = build_stage_params(tokenizer)
    return two_stage_yaml_path, tokenizer, prompt, stage0_params, stage1_params


async def run_once(tag: str):
    yaml_path, tokenizer, prompt, stage0_params, stage1_params = build_request()
    omni = AsyncOmni(model=MODEL_PATH, stage_configs_path=yaml_path)
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
