"""Offline batch sweep for Alpamayo using in-process AsyncOmni requests.

Borrowed from:
- profiler/batch_sweep_cont_new.py: sample preload, grouping, throughput report
- tests/.../offline/alpamoya_compare_original.py: direct AsyncOmni startup/invoke

This version avoids the HTTP client/server path and calls AsyncOmni.generate()
directly to reduce networking and JSON encode/decode overhead.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any
import uuid

import numpy as np
import torch
import yaml
from vllm import SamplingParams


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "vllm-omni":
            return parent
    raise RuntimeError("failed to locate repo root 'vllm-omni'")


OMNI_ROOT = _find_repo_root()
ALPAMAYO_SRC = OMNI_ROOT.parent / "alpamayo1.5" / "src"
VERL_SRC = OMNI_ROOT.parent / "verl-liming" / "my_example" / "alpamayo" / "src"
REQUEST_UID = os.environ.get("REQUEST_UID", uuid.uuid4().hex[:8])

for path in (str(OMNI_ROOT), str(ALPAMAYO_SRC), str(VERL_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from profiler.run_rollout_timing import (  # noqa: E402
    _build_prompt_messages,
    _load_clip_data,
    _load_local_avdi,
    _to_jsonable,
)
import tests.diffusion.models.alpamoya.custom_test.offline.common as ct  # noqa: E402
from vllm_omni.entrypoints.async_omni import AsyncOmni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402
from vllm_omni.model_executor.stage_input_processors.alpamayo1_5 import (  # noqa: E402
    build_alpamayo_stage0_tokenizer,
)

DEFAULT_MODEL_PATH = "/share/models/Alpamayo-1.5-10B"
MODEL = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
T0_US = int(os.environ.get("T0_US", "5100000"))
N_UNIQUE = int(os.environ.get("N_UNIQUE", "64"))
N_TOTAL = int(os.environ.get("N_TOTAL", "64"))
BS_LIST = [int(v) for v in os.environ.get("BS_LIST", "4,8").split(",") if v.strip()]
MAX_REQ_PER_GROUP = int(os.environ.get("MAX_REQ_PER_GROUP", "24"))
CHUNK_SAMPLES = int(os.environ.get("CHUNK_SAMPLES", "16"))
SVC_YAML = OMNI_ROOT / "profiler" / "alpamayo1_5_gpu0.yaml"
GPU_RECOVERY_POLL_S = float(os.environ.get("GPU_RECOVERY_POLL_S", "2.0"))
GPU_RECOVERY_TIMEOUT_S = float(os.environ.get("GPU_RECOVERY_TIMEOUT_S", "180.0"))
GPU_RECOVERY_MARGIN_GB = float(os.environ.get("GPU_RECOVERY_MARGIN_GB", "1.0"))

LOG_DIR = OMNI_ROOT / "profiler" / "logs" / str(N_TOTAL) / f"async_omni_inproc_{int(time.time())}"
ASYNC_OMNI_PROFILER_DIR = LOG_DIR / "async_omni_inproc_profiler"
ASYNC_OMNI_LOG_DIR = LOG_DIR / "svc_logs"
OUT = LOG_DIR / "metrics" / "batch_results.md"
# OUT = OMNI_ROOT / "profiler/inproc/" / "batch_results_async_omni_inproc.md"
# ASYNC_OMNI_PROFILER_DIR = OMNI_ROOT / "profiler" / "async_omni_inproc_profiler"
# ASYNC_OMNI_LOG_DIR = OMNI_ROOT / "profiler" / "logs"

KV_TIMING_RE = re.compile(
    r"KV transfer timing: req=(?P<req>\S+) "
    r"extract_only_ms=(?P<extract_only_ms>\d+(?:\.\d+)?) "
    r"transfer_only_ms=(?P<transfer_only_ms>\d+(?:\.\d+)?) "
    r"extract_plus_transfer_ms=(?P<extract_plus_transfer_ms>\d+(?:\.\d+)?)"
)
RID_SUFFIX_RE = re.compile(r"(async-inproc-bs\d+-\S+)$")


def _output_prefix(default_path: Path) -> Path:
    return default_path.with_suffix("")


@contextmanager
def _omni_env():
    previous_env = {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "FLASHINFER_DISABLE_VERSION_CHECK": os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK"),
        "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        "PROFILER_DIR": os.environ.get("PROFILER_DIR"),
    }
    ASYNC_OMNI_PROFILER_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PROFILER_DIR": str(ASYNC_OMNI_PROFILER_DIR),
    })
    try:
        yield
    finally:
        for env_name, env_value in previous_env.items():
            if env_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = env_value


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone().contiguous()
    if isinstance(value, list):
        cloned: list[Any] = []
        for item in value:
            if hasattr(item, "copy"):
                try:
                    cloned.append(item.copy())
                    continue
                except Exception:
                    pass
            cloned.append(_clone_value(item))
        return cloned
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    return deepcopy(value)


def _materialize_prompt(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": sample["prompt_text"],
        "multi_modal_data": {
            "image": [image.copy() for image in sample["images"]],
        },
        "additional_information": _clone_value(sample["additional_information"]),
    }


def _build_result_from_output(output, cid: str, rid: str, core_request_ms: float) -> dict[str, Any]:
    metrics = dict(getattr(output, "metrics", {}) or {})
    custom = dict(metrics.get("custom_output", {}) or {})
    custom.update(dict(getattr(output, "custom_output", {}) or {}))

    inf = float(metrics.get("inference_only_ms", 0.0) or 0.0)
    llm_ms = float(metrics.get("stage0_llm_ms", 0.0) or 0.0)
    kv_tran_s0 = float(custom.get("kv_tran_s0_ms", 0.0) or 0.0)
    kv_tran_s0_total_ms = kv_tran_s0
    kv_tran_s1_receive = float(custom.get("kv_tran_s1_receive_ms", 0.0) or 0.0)
    kv_tran_s1_prep = float(custom.get("kv_tran_s1_prep_ms", 0.0) or 0.0)
    df = float(custom.get("df_ms", 0.0) or 0.0)
    s0 = llm_ms + kv_tran_s0_total_ms
    s1 = kv_tran_s1_receive + df
    cot_token_ids = custom.get("cot_token_ids")
    print(f"  DEBUG rid={rid} cot_token_ids={cot_token_ids} kv_tran_s0={kv_tran_s0:.2f}ms kv_tran_s1_receive={kv_tran_s1_receive:.2f}ms kv_tran_s1_prep={kv_tran_s1_prep:.2f}ms df={df:.2f}ms")
    if isinstance(cot_token_ids, torch.Tensor):
        output_tokens = int(cot_token_ids.numel())
    elif isinstance(cot_token_ids, list):
        output_tokens = len(cot_token_ids)
    else:
        output_tokens = 0

    return {
        "ok": True,
        "c": cid,
        "rid": rid,
        "core_request_ms": core_request_ms,
        "inf": inf,
        "stage0_llm_ms": llm_ms,
        "s0": s0,
        "s1": s1,
        "qw": max(0.0, core_request_ms - s0 - s1),
        "kv": kv_tran_s1_receive,
        "stage0_extract_only_ms": kv_tran_s0,
        "stage0_transfer_only_ms": 0.0,
        "stage0_extract_plus_transfer_ms": kv_tran_s0_total_ms,
        "kv_tran_s0_total_ms": kv_tran_s0_total_ms,
        "kv_tran_s0": kv_tran_s0,
        "kv_tran_s1_receive": kv_tran_s1_receive,
        "kv_tran_s1_prep": kv_tran_s1_prep,
        "df": df,
        "it": int(metrics.get("input_tokens", 0) or 0),
        "ot": output_tokens,
    }


def _build_group_agg(results: list[dict[str, Any]], wall_ms: float) -> dict[str, Any]:
    ok_results = [r for r in results if r.get("ok")]
    agg: dict[str, Any] = {"wall_ms": wall_ms, "ok": len(ok_results)}
    metric_keys = [
        "core_request_ms",
        "inf",
        "qw",
        "s0",
        "s1",
        "kv",
        "kv_tran_s0",
        "kv_tran_s1_receive",
        "kv_tran_s1_prep",
        "df",
        "it",
        "ot",
    ]
    if ok_results:
        for key in metric_keys:
            vals = [float(r[key]) for r in ok_results]
            agg[f"avg_{key}"] = float(np.mean(vals))
            agg[f"min_{key}"] = float(np.min(vals))
            agg[f"max_{key}"] = float(np.max(vals))
        agg["bs_it"] = sum(int(r["it"]) for r in ok_results)
        agg["bs_ot"] = sum(int(r["ot"]) for r in ok_results)
        agg["bs_at"] = agg["bs_it"] + agg["bs_ot"]
    else:
        for key in metric_keys:
            agg[f"avg_{key}"] = 0.0
            agg[f"min_{key}"] = 0.0
            agg[f"max_{key}"] = 0.0
        agg["bs_it"] = 0
        agg["bs_ot"] = 0
        agg["bs_at"] = 0
    return agg


def _load_kv_metrics_from_log(bs: int) -> dict[str, dict[str, float]]:
    log_path = ASYNC_OMNI_LOG_DIR / f"svc_bs{bs}.log"
    if not log_path.exists():
        return {}

    kv_metrics_by_rid: dict[str, dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = KV_TIMING_RE.search(line)
        if not match:
            continue
        logged_req = match.group("req")
        rid_match = RID_SUFFIX_RE.search(logged_req)
        req_key = rid_match.group(1) if rid_match else logged_req
        kv_metrics_by_rid[req_key] = {
            "extract_only_ms": float(match.group("extract_only_ms")),
            "transfer_only_ms": float(match.group("transfer_only_ms")),
            "extract_plus_transfer_ms": float(match.group("extract_plus_transfer_ms")),
        }
    return kv_metrics_by_rid


def _backfill_kv_metrics_from_log(bs: int, bs_data: list[dict[str, Any]]) -> None:
    kv_metrics_by_rid = _load_kv_metrics_from_log(bs)
    for grp in bs_data:
        for req in grp.get("reqs", []):
            if not req.get("ok"):
                continue
            log_metrics = kv_metrics_by_rid.get(req["rid"])
            if log_metrics:
                req["stage0_extract_only_ms"] = log_metrics["extract_only_ms"]
                req["stage0_transfer_only_ms"] = log_metrics["transfer_only_ms"]
                req["stage0_extract_plus_transfer_ms"] = log_metrics["extract_plus_transfer_ms"]
                req["kv_tran_s0_total_ms"] = log_metrics["extract_plus_transfer_ms"]
            req["s0"] = req["stage0_llm_ms"]
            req["qw"] = max(0.0, req["core_request_ms"] - req["s0"] - req["s1"])
        grp["agg"] = _build_group_agg(grp.get("reqs", []), grp["agg"]["wall_ms"])


def _start_omni(bs, log_stat=True) -> AsyncOmni:
    kwargs: dict[str, Any] = {
        "stage_configs_path": str(SVC_YAML),
    }
    if log_stat:
        output_prefix = _output_prefix(OUT)
        log_stat_filepath = output_prefix.with_name(f"{output_prefix.name}_engine_metrics_bs_{bs}")
        kwargs["log_stat_filepath"] = str(log_stat_filepath)
        kwargs["log_stats"] = True
    return AsyncOmni(model=MODEL, **kwargs)


@contextmanager
def _redirect_process_output(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(log_file.fileno(), stdout_fd)
        os.dup2(log_file.fileno(), stderr_fd)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _load_sampling_params_from_yaml(tokenizer) -> tuple[SamplingParams, OmniDiffusionSamplingParams]:
    config = yaml.safe_load(SVC_YAML.read_text(encoding="utf-8"))
    stage_args = config.get("stage_args") or []
    if len(stage_args) < 2:
        raise RuntimeError(f"unexpected stage config layout in {SVC_YAML}")

    stage0_defaults = dict((stage_args[0] or {}).get("default_sampling_params") or {})
    stage1_defaults = dict((stage_args[1] or {}).get("default_sampling_params") or {})

    stage0_stop_token_ids = stage0_defaults.get("stop_token_ids")
    if stage0_stop_token_ids is None:
        stage0_stop_token_ids = [int(tokenizer.pad_token_id)]

    stage0_params = SamplingParams(
        temperature=float(stage0_defaults.get("temperature", 0.6)),
        top_p=float(stage0_defaults.get("top_p", 0.98)),
        top_k=int(stage0_defaults.get("top_k", 40)),
        max_tokens=int(stage0_defaults.get("max_tokens", 256)),
        stop_token_ids=[int(token_id) for token_id in stage0_stop_token_ids],
        detokenize=bool(stage0_defaults.get("detokenize", False)),
        seed=stage0_defaults.get("seed"),
        n=int(stage0_defaults.get("n", 1)),
        extra_args=dict(stage0_defaults.get("extra_args") or {}),
    )

    stage1_params = OmniDiffusionSamplingParams(
        seed=stage1_defaults.get("seed"),
        num_outputs_per_prompt=int(stage1_defaults.get("num_outputs_per_prompt", 1)),
        num_inference_steps=stage1_defaults.get("num_inference_steps"),
        guidance_scale=float(stage1_defaults.get("guidance_scale", 0.0)),
    )

    return stage0_params, stage1_params


async def one_async(
    omni: AsyncOmni,
    sampling_params_list: tuple[SamplingParams, OmniDiffusionSamplingParams],
    sample: dict[str, Any],
    idx: int,
    bs: int,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    rid = f"async-inproc-bs{bs}-{REQUEST_UID}-{sample['cid'][:8]}-{idx}"
    start = time.perf_counter()

    async with sem:
        try:
            stage0_params, stage1_params = sampling_params_list
            final_output = None
            async for out in omni.generate(
                prompt=_materialize_prompt(sample),
                request_id=rid,
                sampling_params_list=[stage0_params, stage1_params],
            ):
                final_output = out
            if final_output is None:
                raise RuntimeError("no final output")
            core_request_ms = (time.perf_counter() - start) * 1000.0
            return _build_result_from_output(final_output, sample["cid"], rid, core_request_ms)
        except Exception as exc:
            err_str = repr(exc)[:240]
            print(f"  ERR rid={rid} {err_str}")
            return {
                "ok": False,
                "c": sample["cid"],
                "rid": rid,
                "core_request_ms": (time.perf_counter() - start) * 1000.0,
                "err": err_str,
            }


async def run_group_async(
    omni: AsyncOmni,
    sampling_params_list: tuple[SamplingParams, OmniDiffusionSamplingParams],
    grp: list[dict[str, Any]],
    gi: int,
    bs: int,
) -> tuple[float, list[dict[str, Any]], dict[str, Any]]:
    workers = min(len(grp), 8 if bs <= 8 else 4)
    sem = asyncio.Semaphore(workers)
    start = time.perf_counter()
    tasks = [
        asyncio.create_task(one_async(omni, sampling_params_list, sample, gi * bs + ii, bs, sem))
        for ii, sample in enumerate(grp)
    ]
    results = await asyncio.gather(*tasks)
    wall_ms = (time.perf_counter() - start) * 1000.0
    return wall_ms, results, _build_group_agg(results, wall_ms)


def gen(all_batches: dict[int, list[dict[str, Any]]], clip_stats: dict[str, float]) -> None:
    lines = [
        "# AsyncOmni Batch Sweep",
        "",
        f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | {N_TOTAL} samples | in-process AsyncOmni",
        f"- disk_io mean={clip_stats['disk']:.0f}ms | prompt_prep mean={clip_stats['prep']:.0f}ms",
        "",
        "## 表1: 逐请求详细指标",
        "",
        "| bs | gid | rid | core_request_ms | inf_ms | qw_ms | s0_ms | s1_ms | kv_ms | df_ms | itok | otok |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for bs in BS_LIST:
        for grp in all_batches.get(bs, []):
            for req in grp.get("reqs", []):
                if not req.get("ok"):
                    continue
                lines.append(
                    f"| {bs} | {grp['gid']} | {req['rid']} | {req['core_request_ms']:.0f}"
                    f" | {req['inf']:.0f} | {req['qw']:.0f} | {req['s0']:.0f} | {req['s1']:.0f}"
                    f" | {req['kv']:.0f} | {req['df']:.0f} | {req['it']} | {req['ot']} |"
                )

    lines += [
        "",
        "## 表2: 逐 BS 请求级指标统计",
        "",
        "| bs | metric | min | max | mean |",
        "|---:|---:|---:|---:|---:|",
    ]

    metric_names = [
        ("core_request_ms", "core_request_ms"),
        ("inf_ms", "inf"),
        ("qw_ms", "qw"),
        ("s0_ms", "s0"),
        ("s1_ms", "s1"),
        ("kv_tran_s0_ms", "kv_tran_s0"),
        ("kv_tran_s1_receive_ms", "kv_tran_s1_receive"),
        ("kv_tran_s1_prep_ms", "kv_tran_s1_prep"),
        ("df_ms", "df"),
        ("itok", "it"),
        ("otok", "ot"),
    ]

    for bs in BS_LIST:
        all_vals = {key: [] for _, key in metric_names}
        for grp in all_batches.get(bs, []):
            for req in grp.get("reqs", []):
                if not req.get("ok"):
                    continue
                for _, key in metric_names:
                    all_vals[key].append(req.get(key, 0))
        for label, key in metric_names:
            vals = all_vals[key]
            if not vals:
                continue
            lines.append(
                f"| {bs} | {label} | {np.min(vals):.0f} | {np.max(vals):.0f} | {np.mean(vals):.0f} |"
            )

    lines += [
        "",
        "## 表3: 每 BS 吞吐量",
        "",
        "| bs | n_grp | ok | E2E_s | batch_it/s | batch_ot/s | batch_at/s | avg_sample_ms | samples/s | 相比BS=1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    base_samples_per_s = None
    for bs in BS_LIST:
        grps = [g for g in all_batches.get(bs, []) if g["agg"]["ok"] > 0]
        if not grps:
            continue
        ok_total = sum(g["agg"]["ok"] for g in grps)
        e2e_s = sum(g["agg"]["wall_ms"] for g in grps) / 1000.0
        total_it = sum(g["agg"]["bs_it"] for g in grps)
        total_ot = sum(g["agg"]["bs_ot"] for g in grps)
        total_at = sum(g["agg"]["bs_at"] for g in grps)
        total_req = sum(len(g.get("reqs", [])) for g in grps)
        avg_sample_ms = (e2e_s * 1000.0 / total_req) if total_req > 0 else 0.0
        samples_per_s = (total_req / e2e_s) if e2e_s > 0 else 0.0
        if base_samples_per_s is None:
            base_samples_per_s = samples_per_s
        speedup = samples_per_s / base_samples_per_s if base_samples_per_s else 0.0
        lines.append(
            f"| {bs} | {len(grps)} | {ok_total} | {e2e_s:.1f}"
            f" | {total_it / e2e_s:.0f} | {total_ot / e2e_s:.0f} | {total_at / e2e_s:.0f}"
            f" | {avg_sample_ms:.0f} | {samples_per_s:.3f} | {speedup:.2f}x |"
        )

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {OUT}")


def _cleanup_runtime() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _get_gpu_mem_info() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.mem_get_info()
    except Exception:
        return None


async def _wait_for_gpu_recovery(
    *,
    expected_free_bytes: int | None,
    bs: int,
) -> None:
    if expected_free_bytes is None:
        await asyncio.sleep(8)
        return

    margin_bytes = int(GPU_RECOVERY_MARGIN_GB * (1024**3))
    min_free_bytes = max(0, expected_free_bytes - margin_bytes)
    deadline = time.monotonic() + GPU_RECOVERY_TIMEOUT_S
    last_reported_gib: float | None = None

    while True:
        _cleanup_runtime()
        mem_info = _get_gpu_mem_info()
        if mem_info is not None:
            free_bytes, total_bytes = mem_info
            if free_bytes >= min_free_bytes:
                print(
                    f"BATCH={bs} GPU recovered: free={free_bytes / (1024**3):.2f}GiB "
                    f"target>={min_free_bytes / (1024**3):.2f}GiB "
                    f"total={total_bytes / (1024**3):.2f}GiB"
                )
                return

            free_gib = free_bytes / (1024**3)
            if last_reported_gib is None or abs(free_gib - last_reported_gib) >= 0.25:
                print(
                    f"BATCH={bs} waiting for GPU recovery: free={free_gib:.2f}GiB "
                    f"target>={min_free_bytes / (1024**3):.2f}GiB"
                )
                last_reported_gib = free_gib

        if time.monotonic() >= deadline:
            if mem_info is None:
                print(f"BATCH={bs} GPU recovery check unavailable, continue after timeout.")
            else:
                free_bytes, _ = mem_info
                print(
                    f"BATCH={bs} GPU recovery timeout: free={free_bytes / (1024**3):.2f}GiB "
                    f"target>={min_free_bytes / (1024**3):.2f}GiB, continue anyway."
                )
            return

        await asyncio.sleep(GPU_RECOVERY_POLL_S)


async def _prepare_samples(tokenizer) -> tuple[list[dict[str, Any]], dict[str, float]]:
    print(f"Batch sweep: {N_TOTAL} samples ({N_UNIQUE} unique clips cycled), BS={BS_LIST}")
    print("Loading dataset + preparing prompts...")

    avdi = _load_local_avdi()
    clip_index = avdi.clip_index
    unique_cids = list(clip_index[clip_index.chunk == 3116].index)[:N_UNIQUE]

    samples_unique: list[dict[str, Any]] = []
    disk_ms_list: list[float] = []
    prep_ms_list: list[float] = []

    for i, cid in enumerate(unique_cids):
        t_disk = time.perf_counter()
        data = _load_clip_data(cid, T0_US, avdi)
        disk_ms = (time.perf_counter() - t_disk) * 1000.0

        t_prep = time.perf_counter()
        frames = data["image_frames"].flatten(0, 1)
        messages = _build_prompt_messages(
            camera_indices=data["camera_indices"],
            num_frames_per_camera=int(data["image_frames"].shape[1]),
        )
        prompt, prompt_text = ct.build_prompt_from_messages(
            messages,
            data=data,
            frames=frames,
            tokenizer=tokenizer,
        )
        prep_ms = (time.perf_counter() - t_prep) * 1000.0

        samples_unique.append(
            {
                "cid": cid,
                "prompt_text": prompt_text,
                "images": list(prompt["multi_modal_data"]["image"]),
                "additional_information": dict(prompt["additional_information"]),
            }
        )
        disk_ms_list.append(disk_ms)
        prep_ms_list.append(prep_ms)
        print(f"  {i + 1}/{N_UNIQUE} clip={cid[:8]} disk={disk_ms:.0f}ms prep={prep_ms:.0f}ms")

    clip_stats = {
        "disk": float(np.mean(disk_ms_list)) if disk_ms_list else 0.0,
        "prep": float(np.mean(prep_ms_list)) if prep_ms_list else 0.0,
    }
    print(
        f"Preload done. disk_mean={clip_stats['disk']:.0f}ms "
        f"prep_mean={clip_stats['prep']:.0f}ms\n"
    )

    samples = (samples_unique * ((N_TOTAL // max(1, N_UNIQUE)) + 1))[:N_TOTAL]
    return samples, clip_stats


async def main_async() -> None:
    tokenizer = build_alpamayo_stage0_tokenizer(MODEL)
    sampling_params_list = _load_sampling_params_from_yaml(tokenizer)
    samples, clip_stats = await _prepare_samples(tokenizer)

    all_batches: dict[int, list[dict[str, Any]]] = {}
    for bs in BS_LIST:
        if bs > MAX_REQ_PER_GROUP:
            raise ValueError(f"BS={bs} exceeds request-side cap MAX_REQ_PER_GROUP={MAX_REQ_PER_GROUP}")
        startup_mem_info = _get_gpu_mem_info()
        startup_free_bytes = startup_mem_info[0] if startup_mem_info is not None else None
        ASYNC_OMNI_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = ASYNC_OMNI_LOG_DIR / f"svc_bs{bs}.log"
        log_path.write_text("", encoding="utf-8")
        print(f"BATCH={bs} logs -> {log_path}")

        with _omni_env():
            with _redirect_process_output(log_path):
                omni = _start_omni(bs)
        try:
            n_groups = (N_TOTAL + bs - 1) // bs
            grp_per_chunk = max(1, CHUNK_SAMPLES // bs)
            print(f"BATCH={bs}  ({n_groups} groups x {bs}, chunk={grp_per_chunk} groups)")

            bs_data: list[dict[str, Any]] = []
            done_reqs = 0

            for chunk_start in range(0, n_groups, grp_per_chunk):
                chunk_end = min(chunk_start + grp_per_chunk, n_groups)
                for gi in range(chunk_start, chunk_end):
                    start = gi * bs
                    end = min(start + bs, N_TOTAL)
                    grp = samples[start:end]
                    if len(grp) > MAX_REQ_PER_GROUP:
                        raise ValueError(f"Group size {len(grp)} exceeds cap {MAX_REQ_PER_GROUP}")
                    with _redirect_process_output(log_path):
                        wall_ms, results, agg = await run_group_async(
                            omni,
                            sampling_params_list,
                            grp,
                            gi,
                            bs,
                        )
                    bs_data.append({"gid": gi, "reqs": results, "agg": agg})
                    total_ok = sum(item["agg"]["ok"] for item in bs_data)
                    done_reqs += len(grp)
                    print(
                        f"  g{gi + 1:4d}/{n_groups} ok={agg['ok']}/{len(grp)} "
                        f"wall={wall_ms:.0f}ms [cum ok={total_ok}/{done_reqs}]"
                    )
                if chunk_end < n_groups:
                    print("  chunk done, sleep 2s...")
                    await asyncio.sleep(2)

            all_batches[bs] = bs_data
            _backfill_kv_metrics_from_log(bs, bs_data)
            total_ok = sum(item["agg"]["ok"] for item in bs_data)
            total_wall = sum(item["agg"]["wall_ms"] for item in bs_data)
            print(f"  DONE ok={total_ok}/{N_TOTAL} E2E={total_wall / 1000.0:.1f}s")

            output_prefix = _output_prefix(OUT)
            bs_output_path = output_prefix.with_name(f"{output_prefix.name}_bs_{bs}.json")
            bs_output_path.write_text(
                json.dumps(_to_jsonable({"bs_data": bs_data, "cs": clip_stats}), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            gen(all_batches, clip_stats)
        finally:
            with _redirect_process_output(log_path):
                omni.shutdown()
            await _wait_for_gpu_recovery(
                expected_free_bytes=startup_free_bytes,
                bs=bs,
            )

    print(f"\nALL DONE => {OUT}")


if __name__ == "__main__":
    asyncio.run(main_async())
