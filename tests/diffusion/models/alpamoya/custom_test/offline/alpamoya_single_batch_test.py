"""Run one Omni call with multiple Alpamayo prompts."""

import argparse
from pathlib import Path

import numpy as np

from vllm_omni.entrypoints.omni import Omni
from tests.diffusion.models.alpamoya.custom_test.offline.common import (
    CLIP_ID,
    MODEL_PATH,
    T0_US,
    build_alpamayo_stage0_tokenizer,
    build_prompt_from_messages,
    build_stage_params,
    build_two_stage_yaml,
    load_shared_data,
)
from tests.diffusion.models.alpamoya.custom_test.offline.alpamoya_compare_original import (
    build_fixed_initial_noise_x0,
)
CLIP_ID_LIST = [
    "100ae358-f548-49b8-af4d-c0afdbcfe9ed",
    "d00c117c-e1bb-4e8d-b49c-ff7482dc2aa5"
]


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one Omni call with multiple Alpamayo prompts.",
    )
    parser.add_argument(
        "--clip-id",
        nargs="+",
        default=CLIP_ID_LIST,
        help="One or more dataset clip_id values to batch together.",
    )
    parser.add_argument(
        "--t0-us",
        nargs="*",
        type=int,
        default=None,
        help="One timestamp for all clips, or one timestamp per clip.",
    )
    parser.add_argument(
        "--use-helper-messages",
        action="store_true",
        help="Use alpamayo helper.create_message() instead of the local prompt builder.",
    )
    return parser


def _resolve_t0_values(clip_ids: list[str], t0_values: list[int] | None) -> list[int]:
    if not t0_values:
        return [T0_US] * len(clip_ids)
    if len(t0_values) == 1:
        return t0_values * len(clip_ids)
    if len(t0_values) != len(clip_ids):
        raise ValueError(
            f"expected 1 or {len(clip_ids)} --t0-us values, got {len(t0_values)}"
        )
    return t0_values


def _extract_pred_xy(pred_xyz_np: np.ndarray) -> np.ndarray:
    if pred_xyz_np.ndim == 4:
        return pred_xyz_np[0, :, :, :2].transpose(0, 2, 1)
    if pred_xyz_np.ndim == 3:
        return pred_xyz_np[:, :, :2].transpose(0, 2, 1)
    raise ValueError(f"unexpected pred_xyz shape: {pred_xyz_np.shape}")


def _extract_first_token_ids(cot_ids_np: np.ndarray) -> list[int]:
    token_ids = cot_ids_np
    while token_ids.ndim > 1:
        token_ids = token_ids[0]
    return token_ids.tolist()


def _request_index(request_id: str) -> int:
    request_index, _, _ = request_id.partition("_")
    try:
        return int(request_index)
    except ValueError as exc:
        raise ValueError(f"unexpected request_id format: {request_id}") from exc


def main() -> None:
    args = _build_argparser().parse_args()
    tokenizer = build_alpamayo_stage0_tokenizer(MODEL_PATH)
    clip_ids = list(args.clip_id)
    t0_values = _resolve_t0_values(clip_ids, args.t0_us)
    batch_items: list[dict[str, object]] = []
    print(f"clip_ids: {clip_ids}")

    for clip_id, t0_us in zip(clip_ids, t0_values, strict=True):
        data, frames, prompt_messages = load_shared_data(
            clip_id=clip_id,
            t0_us=t0_us,
            use_helper_messages=args.use_helper_messages,
        )
        prompt, prompt_text = build_prompt_from_messages(
            prompt_messages,
            data=data,
            frames=frames,
            tokenizer=tokenizer,
            initial_noise_x0=build_fixed_initial_noise_x0()
        )
        batch_items.append(
            {
                "clip_id": clip_id,
                "t0_us": t0_us,
                "data": data,
                "frames": frames,
                "prompt": prompt,
                "prompt_text": prompt_text,
            }
        )

    batch_size = len(batch_items)
    yaml_path = build_two_stage_yaml(batch_size=batch_size)
    stage0_params, stage1_params = build_stage_params(tokenizer)

    omni = Omni(
        model=MODEL_PATH,
        stage_configs_path=yaml_path,
    )

    try:
        final_outputs = omni.generate(
            prompts=[item["prompt"] for item in batch_items],
            sampling_params_list=[stage0_params, stage1_params],
        )
        if not final_outputs:
            raise RuntimeError("no final outputs")
        if len(final_outputs) != batch_size:
            raise RuntimeError(
                f"expected {batch_size} final outputs, got {len(final_outputs)}"
            )
        outputs_by_index = {
            _request_index(final_output.request_id): final_output for final_output in final_outputs
        }
        if len(outputs_by_index) != batch_size:
            raise RuntimeError(
                f"expected {batch_size} unique request ids, got {len(outputs_by_index)}"
            )

        print("batch_size:", batch_size)
        print("clip_ids:", clip_ids)
        print("t0_us_values:", t0_values)

        for batch_index, item in enumerate(batch_items):
            final_output = outputs_by_index.get(batch_index)
            if final_output is None:
                raise RuntimeError(f"missing final output for request index {batch_index}")
            custom = final_output.custom_output
            pred_xyz = custom["pred_xyz"]
            pred_rot = custom["pred_rot"]
            cot_ids = custom["cot_token_ids"]
            gt_xy = item["data"]["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
            pred_xy = _extract_pred_xy(pred_xyz.detach().cpu().numpy())
            ade = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
            cot_text = tokenizer.decode(
                _extract_first_token_ids(cot_ids.detach().cpu().numpy()),
                skip_special_tokens=False,
            )

            print(f"\n[{batch_index}] clip_id: {item['clip_id']}")
            print(f"[{batch_index}] t0_us: {item['t0_us']}")
            print(f"[{batch_index}] request_id: {final_output.request_id}")
            print(f"[{batch_index}] stage0_context_used: {custom.get('stage0_context_used')}")
            print(f"[{batch_index}] num_frames: {int(item['frames'].shape[0])}")
            print(f"[{batch_index}] prompt_chars: {len(item['prompt_text'])}")
            print(f"[{batch_index}] pred_xyz.shape: {tuple(pred_xyz.shape)}")
            print(f"[{batch_index}] pred_rot.shape: {tuple(pred_rot.shape)}")
            print(f"[{batch_index}] cot_token_ids.shape: {tuple(cot_ids.shape)}")
            print(f"[{batch_index}] minADE: {float(ade.min())} meters")
            print(f"[{batch_index}] COT text:\n{cot_text}")
    finally:
        omni.close()
        Path(yaml_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
