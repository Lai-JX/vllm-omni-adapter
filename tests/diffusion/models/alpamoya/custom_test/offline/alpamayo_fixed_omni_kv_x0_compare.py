import asyncio
import gc

import numpy as np
import torch

from alpamoya_compare_original import (
    actual_action_in_proj_internal_dump_path,
    actual_expert_internal_dump_path,
    actual_stage1_rollout_context_dump_path,
    actual_stage1_rollout_step_dump_path,
    actual_stage1_transition_dump_path,
    build_fixed_initial_noise_x0,
    compare_tensors,
    dumped_omni_kv_cache_path,
    load_shared_data,
    make_compare_run_dir,
    print_dump_compare_report,
    reference_action_in_proj_internal_dump_path,
    reference_expert_internal_dump_path,
    reference_stage1_rollout_context_dump_path,
    reference_stage1_rollout_step_dump_path,
    run_original,
    run_omni,
    run_reference_stage1_with_omni_kv,
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


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_run_summary(name: str, result: dict[str, object], *, ade: float) -> None:
    pred_xyz = result["pred_xyz"]
    pred_rot = result["pred_rot"]
    cot_text = result.get("cot_text")
    context_used = result.get("stage0_context_used")
    mode = result.get("mode")
    print(
        f"[{name}] "
        f"pred_xyz={tuple(pred_xyz.shape)} "
        f"pred_rot={tuple(pred_rot.shape)} "
        f"minADE={ade:.6f}m "
        f"cot_equal_ref={result.get('cot_equal_ref')} "
        f"stage0_context_used={context_used} "
        f"mode={mode}"
    )
    if cot_text is not None:
        print(f"[{name}] cot={cot_text}")


async def main() -> None:
    dump_dir = make_compare_run_dir()
    initial_noise_x0 = build_fixed_initial_noise_x0()
    shared_data, _, _ = load_shared_data()

    _print_section("fixed_x0 setup")
    print(
        "initial_noise_x0 "
        f"shape={tuple(initial_noise_x0.shape)} "
        f"mean={float(initial_noise_x0.mean().item()):.6f} "
        f"std={float(initial_noise_x0.std().item()):.6f}"
    )

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
    omni_kv_path = dumped_omni_kv_cache_path(dump_dir)
    if not omni_kv_path.is_file():
        raise RuntimeError(f"Omni KV dump not found: {omni_kv_path}")

    reference_with_omni_kv = run_reference_stage1_with_omni_kv(
        original,
        kv_dump_path=omni_kv_path,
        dump_dir=dump_dir,
    )

    original_ade = compute_min_ade(original["pred_xyz"], shared_data["ego_future_xyz"])
    omni_ade = compute_min_ade(omni["pred_xyz"], shared_data["ego_future_xyz"])
    reference_with_omni_kv_ade = compute_min_ade(
        reference_with_omni_kv["pred_xyz"],
        shared_data["ego_future_xyz"],
    )

    original["cot_equal_ref"] = True
    omni["cot_equal_ref"] = original["cot_text"] == omni.get("cot_text")
    reference_with_omni_kv["cot_equal_ref"] = None

    _print_section("run summary")
    _print_run_summary("original", original, ade=original_ade)
    _print_run_summary("omni", omni, ade=omni_ade)
    _print_run_summary("reference_with_omni_kv", reference_with_omni_kv, ade=reference_with_omni_kv_ade)

    _print_section("metric deltas")
    print(f"omni vs original minADE delta={abs(original_ade - omni_ade):.6f}m")
    print(
        "reference_with_omni_kv vs original "
        f"minADE delta={abs(original_ade - reference_with_omni_kv_ade):.6f}m"
    )
    print(
        "reference_with_omni_kv vs omni "
        f"minADE delta={abs(omni_ade - reference_with_omni_kv_ade):.6f}m"
    )

    _print_section("tensor compare")
    compare_tensors("pred_xyz_omni", original["pred_xyz"], omni["pred_xyz"])
    compare_tensors("pred_rot_omni", original["pred_rot"], omni["pred_rot"])
    compare_tensors(
        "pred_xyz_reference_with_omni_kv",
        original["pred_xyz"],
        reference_with_omni_kv["pred_xyz"],
    )
    compare_tensors(
        "pred_rot_reference_with_omni_kv",
        original["pred_rot"],
        reference_with_omni_kv["pred_rot"],
    )
    compare_tensors(
        "pred_xyz_reference_with_omni_kv_vs_omni",
        omni["pred_xyz"],
        reference_with_omni_kv["pred_xyz"],
    )
    compare_tensors(
        "pred_rot_reference_with_omni_kv_vs_omni",
        omni["pred_rot"],
        reference_with_omni_kv["pred_rot"],
    )

    _print_section("structured dump compare")
    actual_stage1_dump = actual_stage1_transition_dump_path(dump_dir)
    if actual_stage1_dump.is_file():
        print_dump_compare_report(
            base_dir=dump_dir,
            comparison_target="fixed_omni_kv_x0_stage1_transition",
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

    reference_rollout_context = reference_stage1_rollout_context_dump_path(dump_dir)
    actual_rollout_context = actual_stage1_rollout_context_dump_path(dump_dir)
    if reference_rollout_context.is_file() and actual_rollout_context.is_file():
        print_dump_compare_report(
            base_dir=dump_dir,
            comparison_target="fixed_omni_kv_x0_stage1_rollout_context",
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

    reference_rollout_step0 = reference_stage1_rollout_step_dump_path(dump_dir)
    actual_rollout_step0 = actual_stage1_rollout_step_dump_path(dump_dir)
    if reference_rollout_step0.is_file() and actual_rollout_step0.is_file():
        print_dump_compare_report(
            base_dir=dump_dir,
            comparison_target="fixed_omni_kv_x0_stage1_rollout_step",
            lhs_path=reference_rollout_step0,
            rhs_path=actual_rollout_step0,
            keys=["x", "t", "future_token_embeds", "last_hidden", "pred"],
        )

    reference_action_in_proj_internal = reference_action_in_proj_internal_dump_path(dump_dir)
    actual_action_in_proj_internal = actual_action_in_proj_internal_dump_path(dump_dir)
    if reference_action_in_proj_internal.is_file() and actual_action_in_proj_internal.is_file():
        print_dump_compare_report(
            base_dir=dump_dir,
            comparison_target="fixed_omni_kv_x0_action_in_proj_internal",
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

    reference_expert_internal = reference_expert_internal_dump_path(dump_dir)
    actual_expert_internal = actual_expert_internal_dump_path(dump_dir)
    if reference_expert_internal.is_file() and actual_expert_internal.is_file():
        print_dump_compare_report(
            base_dir=dump_dir,
            comparison_target="fixed_omni_kv_x0_expert_internal",
            lhs_path=reference_expert_internal,
            rhs_path=actual_expert_internal,
            keys=[
                "future_token_embeds",
                "position_ids",
                "attention_mask",
                "prompt_cache_seq_len",
                "prompt_cache_summary",
                "prompt_cache_summary_before_expert",
                "prompt_cache_summary_after_expert",
                "last_hidden_state",
                "last_hidden",
            ],
        )

    _print_section("reference replay internals")
    captured_step0 = reference_with_omni_kv.get("captured_step0") or {}
    captured_action_in_proj_internal = reference_with_omni_kv.get("captured_action_in_proj_internal") or {}
    captured_expert_internal = reference_with_omni_kv.get("captured_expert_internal") or {}
    print(f"reference_with_omni_kv captured_step0_keys={sorted(captured_step0.keys())}")
    print(
        "reference_with_omni_kv captured_action_in_proj_internal_keys="
        f"{sorted(captured_action_in_proj_internal.keys())}"
    )
    print(
        "reference_with_omni_kv captured_expert_internal_keys="
        f"{sorted(captured_expert_internal.keys())}"
    )
    _print_section("artifacts")
    print(f"omni_kv_path={omni_kv_path}")
    print(f"dump_dir={dump_dir}")


if __name__ == "__main__":
    asyncio.run(main())
