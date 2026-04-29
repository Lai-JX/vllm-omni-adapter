import asyncio
import gc
import tempfile
from pathlib import Path

import numpy as np
import torch

from tests.diffusion.models.alpamoya.custom_test.offline.alpamoya_compare_original import (
    REQUEST_ID,
    build_fixed_initial_noise_x0,
    compare_tensors,
    load_shared_data,
    print_dump_compare_report,
    run_original,
    run_omni,
    write_reference_stage1_transition_dump,
)


def _extract_pred_xy(pred_xyz: torch.Tensor) -> np.ndarray:
    pred_xyz_np = pred_xyz.detach().cpu().numpy()
    if pred_xyz_np.ndim == 5:
        return pred_xyz_np[0, 0, :, :, :2].transpose(0, 2, 1)
    if pred_xyz_np.ndim == 4:
        return pred_xyz_np[0, :, :, :2].transpose(0, 2, 1)
    if pred_xyz_np.ndim == 3:
        return pred_xyz_np[:, :, :2].transpose(0, 2, 1)
    raise ValueError(f"unexpected pred_xyz shape: {pred_xyz_np.shape}")


def compute_min_ade(pred_xyz: torch.Tensor, gt_future_xyz: torch.Tensor) -> float:
    gt_xy = gt_future_xyz.detach().cpu()[0, 0, :, :2].T.numpy()
    pred_xy = _extract_pred_xy(pred_xyz)
    ade = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    return float(ade.min())


async def main() -> None:
    dump_dir = Path(tempfile.mkdtemp(prefix="alpamayo_fixed_x0_compare_"))
    initial_noise_x0 = build_fixed_initial_noise_x0()
    shared_data, _, _ = load_shared_data()
    print("initial_noise_x0.shape:", tuple(initial_noise_x0.shape))
    print("initial_noise_x0.mean:", float(initial_noise_x0.mean().item()))
    print("initial_noise_x0.std:", float(initial_noise_x0.std().item()))

    original = run_original(dump_dir=dump_dir, forced_initial_noise_x0=initial_noise_x0)
    reference_stage1_dump = write_reference_stage1_transition_dump(
        dump_dir,
        original["stage0_debug"],
    )

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    await asyncio.sleep(2)

    omni = await run_omni(dump_dir=dump_dir, initial_noise_x0=initial_noise_x0)

    print("original pred_xyz.shape:", tuple(original["pred_xyz"].shape))
    print("omni pred_xyz.shape:", tuple(omni["pred_xyz"].shape))
    print("original pred_rot.shape:", tuple(original["pred_rot"].shape))
    print("omni pred_rot.shape:", tuple(omni["pred_rot"].shape))
    print("original cot:", original["cot_text"])
    print("omni cot:", omni["cot_text"])
    print("omni stage0_context_used:", omni["stage0_context_used"])

    compare_tensors("pred_xyz", original["pred_xyz"], omni["pred_xyz"])
    compare_tensors("pred_rot", original["pred_rot"], omni["pred_rot"])
    print("COMPARE cot_text_equal:", original["cot_text"] == omni["cot_text"])
    original_ade = compute_min_ade(original["pred_xyz"], shared_data["ego_future_xyz"])
    omni_ade = compute_min_ade(omni["pred_xyz"], shared_data["ego_future_xyz"])
    print(f"original minADE: {original_ade:.6f} meters")
    print(f"omni minADE: {omni_ade:.6f} meters")
    print(f"minADE delta: {abs(original_ade - omni_ade):.6f} meters")

    actual_stage1_dump = dump_dir / REQUEST_ID / "actual_stage1_transition.pt"
    if actual_stage1_dump.is_file():
        print_dump_compare_report(
            title="FIXED X0 STAGE1 TRANSITION DUMP COMPARE",
            lhs_path=reference_stage1_dump,
            rhs_path=actual_stage1_dump,
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

    reference_rollout_context = dump_dir / REQUEST_ID / "reference_stage1_rollout_context.pt"
    actual_rollout_context = dump_dir / REQUEST_ID / "stage1_rollout_context.pt"
    if reference_rollout_context.is_file() and actual_rollout_context.is_file():
        print_dump_compare_report(
            title="FIXED X0 STAGE1 ROLLOUT CONTEXT DUMP COMPARE",
            lhs_path=reference_rollout_context,
            rhs_path=actual_rollout_context,
            keys=[
                "initial_noise_x0",
                "prefill_seq_len",
                "offset",
                "position_ids",
                "attention_mask",
            ],
        )

    reference_rollout_step0 = dump_dir / REQUEST_ID / "reference_stage1_rollout_step0.pt"
    actual_rollout_step0 = dump_dir / REQUEST_ID / "stage1_rollout_step0.pt"
    if reference_rollout_step0.is_file() and actual_rollout_step0.is_file():
        print_dump_compare_report(
            title="FIXED X0 STAGE1 ROLLOUT STEP0 DUMP COMPARE",
            lhs_path=reference_rollout_step0,
            rhs_path=actual_rollout_step0,
            keys=["x", "t", "future_token_embeds", "last_hidden", "pred"],
        )

    print("dump_dir:", dump_dir)


if __name__ == "__main__":
    asyncio.run(main())
