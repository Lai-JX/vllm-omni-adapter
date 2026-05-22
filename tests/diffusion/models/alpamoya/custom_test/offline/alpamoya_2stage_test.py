import asyncio
from pathlib import Path

import numpy as np

from vllm_omni.entrypoints.async_omni import AsyncOmni
from common import (
    MODEL_PATH,
    build_alpamayo_stage0_tokenizer,
    build_prompt_from_messages,
    build_stage_params,
    build_two_stage_yaml,
    load_shared_data,
)


async def main():
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
