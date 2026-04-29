"""Client-side Alpamayo request adapter for `vllm serve --omni`."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time
from typing import Any
import uuid
from urllib.error import HTTPError
from urllib import request as urllib_request

import numpy as np
import torch

OMNI_ROOT = Path(__file__).resolve().parents[5]
WORKSPACE_ROOT = OMNI_ROOT.parent

for candidate in (str(OMNI_ROOT), str(WORKSPACE_ROOT / "vllm")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from tests.diffusion.models.alpamoya.custom_test.offline.common import (
    CLIP_ID,
    MODEL_PATH,
    T0_US,
    build_additional_information,
    build_alpamayo_stage0_tokenizer,
    load_shared_data,
    tensor_to_pil,
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="alpamayo1.5")
    parser.add_argument("--clip-id", default=CLIP_ID)
    parser.add_argument("--t0-us", type=int, default=T0_US)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--use-helper-messages", action="store_true")
    parser.add_argument("--include-raw-response", action="store_true")
    return parser


def _image_to_data_url(frame: Any) -> str:
    image = tensor_to_pil(frame)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _adapt_messages(messages: list[dict[str, Any]], frames: Any) -> list[dict[str, Any]]:
    frame_urls = [_image_to_data_url(frame) for frame in frames]
    frame_index = 0
    adapted_messages: list[dict[str, Any]] = []

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            adapted_messages.append(message)
            continue

        adapted_content: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                adapted_content.append(item)
                continue
            if item.get("type") == "image":
                if frame_index >= len(frame_urls):
                    raise ValueError("not enough frames to fill image placeholders")
                adapted_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": frame_urls[frame_index]},
                    }
                )
                frame_index += 1
            else:
                adapted_content.append(item)
        adapted_messages.append(
            {
                **message,
                "content": adapted_content,
            }
        )

    if frame_index != len(frame_urls):
        raise ValueError(
            f"unused frames after message adaptation: used={frame_index}, total={len(frame_urls)}"
        )
    return adapted_messages


def _extract_first_token_ids(cot_ids: Any) -> list[int]:
    token_ids = np.asarray(cot_ids)
    while token_ids.ndim > 1:
        token_ids = token_ids[0]
    return token_ids.astype(np.int64).tolist()


def _extract_pred_xy(pred_xyz: Any) -> np.ndarray:
    pred_xyz_np = np.asarray(pred_xyz)
    if pred_xyz_np.ndim == 5:
        return pred_xyz_np[0, 0, :, :, :2].transpose(0, 2, 1)
    if pred_xyz_np.ndim == 4:
        return pred_xyz_np[0, :, :, :2].transpose(0, 2, 1)
    if pred_xyz_np.ndim == 3:
        return pred_xyz_np[:, :, :2].transpose(0, 2, 1)
    raise ValueError(f"unexpected pred_xyz shape: {pred_xyz_np.shape}")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (bytes, bytearray)):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(_to_jsonable(payload)).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = body
        try:
            parsed = json.loads(body)
            detail = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP {exc.code} when POST {url}\nResponse body:\n{detail}"
        ) from exc
def print_dict_keys_recursive(obj: Any, prefix: str = "") -> None:
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        print(path)
        print_dict_keys_recursive(value, path)
        if isinstance(value, list):
            for item in value:
                print_dict_keys_recursive(item, path)

def main() -> None:
    args = _build_argparser().parse_args()
    request_id = args.request_id or f"alpamayo-openai-client-{uuid.uuid4()}"
    tokenizer = build_alpamayo_stage0_tokenizer(MODEL_PATH)
    data, frames, prompt_messages = load_shared_data(
        clip_id=args.clip_id,
        t0_us=args.t0_us,
        use_helper_messages=args.use_helper_messages,
    )
    request_messages = _adapt_messages(prompt_messages, frames)
    additional_information = build_additional_information(data)

    payload = {
        # "model": args.model,
        "request_id": request_id,
        "messages": request_messages,
        "add_generation_prompt": False,
        "continue_final_message": True,
        "additional_information": additional_information,
        "temperature": 0.6,
        "top_p": 0.98,
        "top_k": 40,
        "max_tokens": 256,
        "stop_token_ids": None, # 如果不设，默认值是[]，系统判断不为None就不会用yaml里配置的stop_token_ids了，所以这里设置为None
        "seed": 42,
        "return_custom_output": True,
    }
    print(print_dict_keys_recursive(payload))
    st = time.time()
    response = _post_json(
        f"http://{args.host}:{args.port}/v1/chat/completions",
        payload,
    )
    et = time.time()
    print(f"Request latency: {(et - st) * 1000:.2f} ms")

    metrics = response.get("metrics") or {}
    custom_output = metrics.get("custom_output") or {}
    pred_xyz = custom_output.get("pred_xyz")
    pred_rot = custom_output.get("pred_rot")
    cot_token_ids = custom_output.get("cot_token_ids")
    stage0_context_used = custom_output.get("stage0_context_used")
    # print(custom_output)
    # print(metrics)
    print(response.keys())

    summary: dict[str, Any] = {
        "clip_id": args.clip_id,
        "t0_us": args.t0_us,
        "request_id": response.get("id"),
        "client_request_id": request_id,
        "stage0_context_used": stage0_context_used,
        "response_has_custom_output": bool(custom_output),
    }

    if pred_xyz is not None:
        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = _extract_pred_xy(pred_xyz)
        ade = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        summary["pred_xyz_shape"] = list(np.asarray(pred_xyz).shape)
        summary["minADE_m"] = float(ade.min())

    if pred_rot is not None:
        summary["pred_rot_shape"] = list(np.asarray(pred_rot).shape)

    if cot_token_ids is not None:
        first_token_ids = _extract_first_token_ids(cot_token_ids)
        summary["cot_token_ids_shape"] = list(np.asarray(cot_token_ids).shape)
        summary["cot_text"] = tokenizer.decode(first_token_ids, skip_special_tokens=False)

    if args.include_raw_response:
        summary["raw_response"] = response

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
