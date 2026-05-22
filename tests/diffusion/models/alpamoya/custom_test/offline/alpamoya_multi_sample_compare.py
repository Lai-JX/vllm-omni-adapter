import argparse
import asyncio
import gc
import json
from pathlib import Path

import numpy as np
import torch

import common as ct
from alpamoya_compare_original import (
    build_fixed_initial_noise_x0,
    create_omni_for_compare,
    get_omni_prompt_token_count,
    make_compare_run_dir,
    run_original,
    run_omni,
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


def _load_cases(limit: int | None, *, t0_us: int) -> list[dict[str, object]]:
    json_path = Path(ct.DATASET_PATH) / "train.jsonl"
    cases: list[dict[str, object]] = []
    with json_path.open("r") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            clip_id = item.get("clip_id")
            if clip_id is None:
                continue
            cases.append(
                {
                    "case_index": idx,
                    "clip_id": str(clip_id),
                    "t0_us": int(t0_us),
                }
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


def _run_original_case(
    case: dict[str, object],
    *,
    fixed_x0: bool,
    steps: int,
) -> dict[str, object]:
    clip_id = str(case["clip_id"])
    t0_us = int(case["t0_us"])
    case_index = int(case["case_index"])
    dump_dir = make_compare_run_dir()
    initial_noise_x0 = build_fixed_initial_noise_x0() if fixed_x0 else None
    shared_data, _, _ = ct.load_shared_data(
        use_helper_messages=True,
        clip_id=clip_id,
        t0_us=t0_us,
    )
    original = run_original(
        dump_dir=dump_dir,
        forced_initial_noise_x0=initial_noise_x0,
        clip_id=clip_id,
        t0_us=t0_us,
        num_inference_steps=steps,
    )
    original_ade = compute_min_ade(original["pred_xyz"], shared_data["ego_future_xyz"])
    return {
        "case_index": case_index,
        "clip_id": clip_id,
        "t0_us": t0_us,
        "dump_dir": str(dump_dir),
        "initial_noise_x0": initial_noise_x0,
        "shared_gt_future_xyz": shared_data["ego_future_xyz"].detach().cpu(),
        "original_minADE": original_ade,
        "original_cot_text": str(original.get("cot_text") or ""),
    }


async def _run_omni_case(
    prepared_case: dict[str, object],
    *,
    steps: int,
    omni,
) -> dict[str, object]:
    clip_id = str(prepared_case["clip_id"])
    t0_us = int(prepared_case["t0_us"])
    dump_dir = Path(str(prepared_case["dump_dir"]))
    initial_noise_x0 = prepared_case["initial_noise_x0"]
    omni_result = await run_omni(
        dump_dir=dump_dir,
        initial_noise_x0=initial_noise_x0,
        clip_id=clip_id,
        t0_us=t0_us,
        num_inference_steps=steps,
        omni=omni,
    )
    original_ade = float(prepared_case["original_minADE"])
    original_cot_text = str(prepared_case.get("original_cot_text") or "")
    omni_cot_text = str(omni_result.get("cot_text") or "")
    omni_ade = compute_min_ade(omni_result["pred_xyz"], prepared_case["shared_gt_future_xyz"])
    delta = omni_ade - original_ade
    prepared_case["omni_minADE"] = omni_ade
    prepared_case["delta"] = delta
    prepared_case["abs_delta"] = abs(delta)
    prepared_case["omni_worse"] = omni_ade > original_ade
    prepared_case["omni_cot_text"] = omni_cot_text
    prepared_case["cot_equal"] = original_cot_text == omni_cot_text
    prepared_case.pop("initial_noise_x0", None)
    prepared_case.pop("shared_gt_future_xyz", None)
    return prepared_case


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare original vs omni across multiple Alpamayo samples.")
    parser.add_argument("--limit", type=int, default=5, help="Number of dataset samples to evaluate")
    parser.add_argument(
        "--fixed-x0",
        action="store_true",
        help="Reuse the same fixed initial noise x0 for every case",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=ct.T0_US,
        help="Shared t0_us used for every selected clip",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Stage-1 diffusion num_inference_steps used for both original and omni",
    )
    args = parser.parse_args()

    cases = _load_cases(args.limit, t0_us=int(args.t0_us))
    if not cases:
        raise RuntimeError("No dataset cases found")

    _print_section("multi-sample setup")
    print(
        f"num_cases={len(cases)} fixed_x0={bool(args.fixed_x0)} "
        f"steps={int(args.steps)}"
    )

    prepared_cases: list[dict[str, object]] = []
    _print_section("original phase")
    for case in cases:
        print(f"clip_id={case['clip_id']} t0_us={case['t0_us']} steps={int(args.steps)}")
        prepared_case = _run_original_case(case, fixed_x0=bool(args.fixed_x0), steps=int(args.steps))
        prepared_cases.append(prepared_case)
        print(f"original_minADE={float(prepared_case['original_minADE']):.6f}m")
        print(f"dump_dir={prepared_case['dump_dir']}")

    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    await asyncio.sleep(2)

    prompt_token_count = get_omni_prompt_token_count(
        clip_id=str(cases[0]["clip_id"]),
        t0_us=int(args.t0_us),
    )
    omni, omni_yaml_path = create_omni_for_compare(prompt_token_count=prompt_token_count)

    results: list[dict[str, object]] = []
    _print_section("omni phase")
    try:
        for prepared_case in prepared_cases:
            print(
                f"clip_id={prepared_case['clip_id']} "
                f"t0_us={prepared_case['t0_us']} steps={int(args.steps)}"
            )
            result = await _run_omni_case(prepared_case, steps=int(args.steps), omni=omni)
            results.append(result)
            print(
                f"original_minADE={float(result['original_minADE']):.6f}m "
                f"omni_minADE={float(result['omni_minADE']):.6f}m "
                f"delta={float(result['delta']):.6f}m omni_worse={bool(result['omni_worse'])} "
                f"cot_equal={bool(result['cot_equal'])}"
            )
            print(f"dump_dir={result['dump_dir']}")
    finally:
        omni.shutdown()
        Path(omni_yaml_path).unlink(missing_ok=True)

    worse_count = sum(1 for result in results if bool(result["omni_worse"]))
    equal_count = sum(1 for result in results if float(result["delta"]) == 0.0)
    better_count = len(results) - worse_count - equal_count
    cot_equal_count = sum(1 for result in results if bool(result["cot_equal"]))
    mean_delta = float(np.mean([float(result["delta"]) for result in results]))
    mean_abs_delta = float(np.mean([float(result["abs_delta"]) for result in results]))
    max_worse = max(results, key=lambda result: float(result["delta"]))
    max_better = min(results, key=lambda result: float(result["delta"]))

    _print_section("summary")
    print(
        f"cases={len(results)} omni_worse_count={worse_count} "
        f"omni_better_count={better_count} equal_count={equal_count} "
        f"cot_equal_count={cot_equal_count}"
    )
    print(f"mean_delta={mean_delta:.6f}m mean_abs_delta={mean_abs_delta:.6f}m")
    print(
        "max_worse_case="
        f"index={max_worse['case_index']} clip_id={max_worse['clip_id']} t0_us={max_worse['t0_us']} delta={float(max_worse['delta']):.6f}m"
    )
    print(
        "max_better_case="
        f"index={max_better['case_index']} clip_id={max_better['clip_id']} t0_us={max_better['t0_us']} delta={float(max_better['delta']):.6f}m"
    )

    print("\nRESULTS_JSON=")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
