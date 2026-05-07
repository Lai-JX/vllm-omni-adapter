"""
GRPO Rollout Timing Script for Alpamayo Model.

Simulates the GRPO rollout process:
- 2 GPUs, each with a complete model replica
- Each GPU handles local batch = 2 samples
- Global batch = 4 samples per round
- Rollout N = 12 per sample (may be split across multiple passes if OOM)
- 4 experiment groups using val.parquet test data

Statistics collected:
  - Per-batch total time (4 batches)
  - Per-rollout generation time (up to 24 per batch)
  - Total rollout time across both GPUs

Usage:
    python run_rollout_timing.py [--rollout-n N] [--local-batch B] [--num-groups G]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import torch

# === Path setup ===
OMNI_ROOT = Path(__file__).resolve().parents[1]
ALPAMAYO_SRC = OMNI_ROOT.parent / "alpamayo1.5" / "src"
VERL_SRC = OMNI_ROOT.parent / "verl-liming" / "my_example" / "alpamayo" / "src"
WORKSPACE_ROOT = OMNI_ROOT.parents[2]

for p in (str(OMNI_ROOT), str(ALPAMAYO_SRC), str(VERL_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# === Dataset & model config ===
DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    str("/share/datasets/Alpamayo_pai_av_big"),
)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str("/share/models/Alpamayo-1.5-10B"),
)
VAL_PARQUET = str(OMNI_ROOT.parent / "verl-liming" / "my_example" / "data" / "val.parquet")
CHUNK_IDS = [3116]

# === Service config ===
GPU0_PORT = 8000
GPU1_PORT = 8001
HOST = "127.0.0.1"

# === GRPO config ===
DEFAULT_LOCAL_BATCH = 2   # per GPU per round
DEFAULT_ROLLOUT_N = 12    # rollouts per sample
DEFAULT_NUM_GROUPS = 4    # experiment groups
MAX_N_PER_REQUEST = 1     # single rollout per HTTP request (conservative; increase if no OOM)


def _load_local_avdi():
    from alpamayo_r1.data.pai_utils import PhysicalAIAVDatasetLocalInterface
    return PhysicalAIAVDatasetLocalInterface(
        local_dir=DATASET_PATH,
        chunk_ids=CHUNK_IDS,
    )


def _load_clip_data(clip_id: str, t0_us: int, avdi) -> dict[str, Any]:
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
    return load_physical_aiavdataset(
        clip_id=clip_id,
        t0_us=t0_us,
        avdi=avdi,
        maybe_stream=False,
    )


def _tensor_to_data_url(frame: torch.Tensor) -> str:
    from PIL import Image as PILImage
    img = frame.detach().cpu()
    if img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    if img.dtype != torch.uint8:
        if img.max() <= 1.0:
            img = (img * 255).clamp(0, 255)
        else:
            img = img.clamp(0, 255)
        img = img.to(torch.uint8)
    pil = PILImage.fromarray(img.numpy())
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_prompt_messages(
    camera_indices: torch.Tensor,
    num_frames_per_camera: int,
) -> list[dict[str, Any]]:
    from alpamayo1_5 import helper
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
            "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}],
        },
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
    ]


def _adapt_messages_with_frames(
    messages: list[dict[str, Any]],
    frames: torch.Tensor,
) -> list[dict[str, Any]]:
    frame_urls = [_tensor_to_data_url(frame) for frame in frames]
    frame_index = 0
    adapted = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            adapted.append(message)
            continue
        adapted_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                adapted_content.append({"type": "image_url", "image_url": {"url": frame_urls[frame_index]}})
                frame_index += 1
            else:
                adapted_content.append(item)
        adapted.append({**message, "content": adapted_content})
    return adapted


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
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(_to_jsonable(payload)).encode("utf-8")
    req = urllib_request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} when POST {url}\n{body}") from exc


def _single_rollout(
    host: str,
    port: int,
    clip_id: str,
    t0_us: int,
    request_messages: list[dict[str, Any]],
    additional_information: dict[str, Any],
    rollout_idx: int,
) -> dict[str, Any]:
    """Send one rollout request and return timing + result info."""
    request_id = f"rollout-{clip_id[:8]}-{rollout_idx}-{uuid.uuid4().hex[:6]}"
    payload = {
        "request_id": request_id,
        "messages": request_messages,
        "add_generation_prompt": False,
        "continue_final_message": True,
        "additional_information": additional_information,
        "temperature": 0.6,
        "top_p": 0.98,
        "top_k": 40,
        "max_tokens": 256,
        "stop_token_ids": None,
        "seed": rollout_idx,  # vary seed per rollout for diversity
        "return_custom_output": True,
    }
    url = f"http://{host}:{port}/v1/chat/completions"
    t_start = time.time()
    try:
        response = _post_json(url, payload)
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000
        return {
            "success": True,
            "rollout_idx": rollout_idx,
            "clip_id": clip_id,
            "port": port,
            "latency_ms": latency_ms,
            "request_id": request_id,
        }
    except Exception as e:
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000
        return {
            "success": False,
            "rollout_idx": rollout_idx,
            "clip_id": clip_id,
            "port": port,
            "latency_ms": latency_ms,
            "error": str(e)[:200],
            "request_id": request_id,
        }


def prepare_sample_request(
    clip_id: str,
    t0_us: int,
    avdi,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load clip data and build request messages + additional_information."""
    data = _load_clip_data(clip_id, t0_us, avdi)
    frames = data["image_frames"].flatten(0, 1)
    prompt_messages = _build_prompt_messages(
        camera_indices=data["camera_indices"],
        num_frames_per_camera=int(data["image_frames"].shape[1]),
    )
    request_messages = _adapt_messages_with_frames(prompt_messages, frames)
    additional_information = {
        "ego_history_xyz": data["ego_history_xyz"].cpu(),
        "ego_history_rot": data["ego_history_rot"].cpu(),
        "alpamayo_model_path": MODEL_PATH,
    }
    return request_messages, additional_information


def run_batch(
    batch_idx: int,
    samples: list[dict],  # list of {clip_id, t0_us}
    avdi,
    rollout_n: int,
    host: str = HOST,
    gpu0_port: int = GPU0_PORT,
    gpu1_port: int = GPU1_PORT,
) -> dict[str, Any]:
    """
    Run one global batch of rollouts across 2 GPUs.
    
    samples has 4 entries: GPU0 gets samples[0:2], GPU1 gets samples[2:4]
    Each sample gets rollout_n rollouts (sent sequentially or in parallel).
    """
    print(f"\n{'='*60}")
    print(f"Batch {batch_idx+1}: preparing {len(samples)} samples...")
    
    # Prepare all sample data first
    prepared = []
    for s in samples:
        print(f"  Loading clip {s['clip_id']}...")
        req_msgs, add_info = prepare_sample_request(s["clip_id"], s["t0_us"], avdi)
        prepared.append({
            "clip_id": s["clip_id"],
            "t0_us": s["t0_us"],
            "request_messages": req_msgs,
            "additional_information": add_info,
        })
    
    # GPU0 handles prepared[0:2], GPU1 handles prepared[2:4]
    gpu_assignments = [
        (prepared[0], gpu0_port),
        (prepared[1], gpu0_port),
        (prepared[2], gpu1_port),
        (prepared[3], gpu1_port),
    ]
    
    batch_rollout_results: list[dict] = []
    batch_t_start = time.time()
    
    # Send rollouts: for each sample, send rollout_n requests
    # We use ThreadPoolExecutor to parallelize across both GPUs concurrently
    all_tasks = []
    for sample_data, port in gpu_assignments:
        for ri in range(rollout_n):
            all_tasks.append((sample_data, port, ri))
    
    print(f"  Sending {len(all_tasks)} total rollout requests ({rollout_n} per sample × 4 samples)...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for sample_data, port, ri in all_tasks:
            future = executor.submit(
                _single_rollout,
                host, port,
                sample_data["clip_id"],
                sample_data["t0_us"],
                sample_data["request_messages"],
                sample_data["additional_information"],
                ri,
            )
            futures[future] = (sample_data["clip_id"], port, ri)
        
        for future in as_completed(futures):
            result = future.result()
            batch_rollout_results.append(result)
            status = "✓" if result["success"] else "✗"
            print(f"    {status} clip={result['clip_id'][:8]} rollout={result['rollout_idx']} "
                  f"port={result['port']} latency={result['latency_ms']:.0f}ms")
    
    batch_t_end = time.time()
    batch_duration_ms = (batch_t_end - batch_t_start) * 1000
    
    # Stats
    successful = [r for r in batch_rollout_results if r["success"]]
    failed = [r for r in batch_rollout_results if not r["success"]]
    latencies = [r["latency_ms"] for r in successful]
    
    batch_stats = {
        "batch_idx": batch_idx,
        "batch_duration_ms": batch_duration_ms,
        "total_rollouts": len(batch_rollout_results),
        "successful_rollouts": len(successful),
        "failed_rollouts": len(failed),
        "rollout_latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else 0,
            "min": float(np.min(latencies)) if latencies else 0,
            "max": float(np.max(latencies)) if latencies else 0,
            "median": float(np.median(latencies)) if latencies else 0,
        },
        "per_rollout_details": batch_rollout_results,
    }
    
    print(f"\n  Batch {batch_idx+1} summary:")
    print(f"    Duration: {batch_duration_ms:.0f}ms")
    print(f"    Rollouts: {len(successful)}/{len(batch_rollout_results)} succeeded")
    if latencies:
        print(f"    Latency: mean={np.mean(latencies):.0f}ms, min={np.min(latencies):.0f}ms, max={np.max(latencies):.0f}ms")
    
    return batch_stats


def check_service_health(host: str, port: int, timeout: int = 300) -> bool:
    """Wait for service to be ready."""
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    print(f"Waiting for service at {url}...")
    while time.time() < deadline:
        try:
            with urllib_request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"  Service at port {port} is ready!")
                    return True
        except Exception:
            pass
        time.sleep(5)
    print(f"  Timeout waiting for service at port {port}")
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-n", type=int, default=DEFAULT_ROLLOUT_N, help="Rollouts per sample")
    parser.add_argument("--local-batch", type=int, default=DEFAULT_LOCAL_BATCH, help="Samples per GPU")
    parser.add_argument("--num-groups", type=int, default=DEFAULT_NUM_GROUPS, help="Number of experiment groups")
    parser.add_argument("--gpu0-port", type=int, default=GPU0_PORT)
    parser.add_argument("--gpu1-port", type=int, default=GPU1_PORT)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--skip-health-check", action="store_true")
    parser.add_argument("--output", default=str(OMNI_ROOT / "profiler" / "timing_results.json"))
    args = parser.parse_args()
    
    global_batch = args.local_batch * 2  # 2 GPUs
    print(f"GRPO Rollout Timing Configuration:")
    print(f"  GPUs: 2 (ports {args.gpu0_port}, {args.gpu1_port})")
    print(f"  Local batch per GPU: {args.local_batch}")
    print(f"  Global batch: {global_batch}")
    print(f"  Rollout N per sample: {args.rollout_n}")
    print(f"  Experiment groups: {args.num_groups}")
    print(f"  Total rollouts per batch: {global_batch * args.rollout_n}")
    
    # Health check
    if not args.skip_health_check:
        ok0 = check_service_health(args.host, args.gpu0_port)
        ok1 = check_service_health(args.host, args.gpu1_port)
        if not (ok0 and ok1):
            print("ERROR: One or both services not ready. Exiting.")
            sys.exit(1)
    
    # Load val.parquet for test samples
    print(f"\nLoading test samples from {VAL_PARQUET}...")
    val_df = pd.read_parquet(VAL_PARQUET)
    print(f"  {len(val_df)} samples available: {list(val_df['clip_id'])}")
    
    # Build 4 batches of global_batch samples each (cycling through val set)
    batches = []
    all_samples = [{"clip_id": row["clip_id"], "t0_us": int(row["t0_us"])} for _, row in val_df.iterrows()]
    n = len(all_samples)
    for gi in range(args.num_groups):
        batch = [all_samples[(gi * global_batch + i) % n] for i in range(global_batch)]
        batches.append(batch)
    
    print(f"\nBatch assignments:")
    for i, batch in enumerate(batches):
        print(f"  Group {i+1}: {[s['clip_id'][:8] for s in batch]}")
    
    # Initialize dataset interface
    print(f"\nInitializing local dataset interface from {DATASET_PATH}...")
    avdi = _load_local_avdi()
    print("  Done.")
    
    # Run experiments
    all_batch_stats = []
    total_t_start = time.time()
    
    for gi, batch in enumerate(batches):
        print(f"\n{'#'*60}")
        print(f"EXPERIMENT GROUP {gi+1}/{args.num_groups}")
        batch_stats = run_batch(
            batch_idx=gi,
            samples=batch,
            avdi=avdi,
            rollout_n=args.rollout_n,
            host=args.host,
            gpu0_port=args.gpu0_port,
            gpu1_port=args.gpu1_port,
        )
        all_batch_stats.append(batch_stats)
    
    total_t_end = time.time()
    total_duration_ms = (total_t_end - total_t_start) * 1000
    
    # Overall summary
    all_rollout_latencies = []
    for bs in all_batch_stats:
        for r in bs["per_rollout_details"]:
            if r["success"]:
                all_rollout_latencies.append(r["latency_ms"])
    
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiment duration: {total_duration_ms:.0f}ms ({total_duration_ms/1000:.1f}s)")
    print(f"\nPer-batch durations:")
    for bs in all_batch_stats:
        n_ok = bs["successful_rollouts"]
        n_total = bs["total_rollouts"]
        print(f"  Group {bs['batch_idx']+1}: {bs['batch_duration_ms']:.0f}ms  ({n_ok}/{n_total} rollouts OK)")
    
    if all_rollout_latencies:
        print(f"\nOverall rollout latency (all {len(all_rollout_latencies)} successful):")
        print(f"  mean:   {np.mean(all_rollout_latencies):.0f}ms")
        print(f"  median: {np.median(all_rollout_latencies):.0f}ms")
        print(f"  min:    {np.min(all_rollout_latencies):.0f}ms")
        print(f"  max:    {np.max(all_rollout_latencies):.0f}ms")
    
    # Save results
    results = {
        "config": {
            "local_batch": args.local_batch,
            "global_batch": global_batch,
            "rollout_n": args.rollout_n,
            "num_groups": args.num_groups,
            "gpu0_port": args.gpu0_port,
            "gpu1_port": args.gpu1_port,
        },
        "total_duration_ms": total_duration_ms,
        "all_batch_stats": all_batch_stats,
        "overall_latency_summary": {
            "n_successful": len(all_rollout_latencies),
            "mean_ms": float(np.mean(all_rollout_latencies)) if all_rollout_latencies else 0,
            "median_ms": float(np.median(all_rollout_latencies)) if all_rollout_latencies else 0,
            "min_ms": float(np.min(all_rollout_latencies)) if all_rollout_latencies else 0,
            "max_ms": float(np.max(all_rollout_latencies)) if all_rollout_latencies else 0,
        } if all_rollout_latencies else {},
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
