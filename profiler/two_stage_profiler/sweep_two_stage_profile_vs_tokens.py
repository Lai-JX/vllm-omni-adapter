"""Collect full two-stage timing against merged natural-sample token count.

This script reuses the tokenized data-building path from
profiler/batch_sweep_cont_inproc_tokenized_warmup.py and the merged-sample
construction from profiler/sweep_stage0_profile_vs_tokens.py, but runs the full
stage-0 -> stage-1 pipeline.

For each merged request it writes one datapoint containing:
- stage-0 profile fields (embed_multimodal_ms / forward_ms)
- subsequent stage-0 decode time
- KV transfer time
- stage-1 diffusion time
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import yaml
from vllm import SamplingParams

OMNI = Path(__file__).resolve().parents[2]
if str(OMNI) not in sys.path:
    sys.path.insert(0, str(OMNI))

import profiler.batch_sweep_cont_inproc_tokenized_warmup as warmup_mod  # noqa: E402
from profiler.batch_sweep_cont_inproc_tokenized_warmup import (  # noqa: E402
    MODEL,
    T0_US,
    _build_model_input,
    _build_model_input_components,
    _cleanup_runtime,
    _clone_value,
    _omni_env,
    _redirect_process_output,
)
from vllm_omni.entrypoints.async_omni import AsyncOmni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402

BS = int(os.environ.get("BS", "1"))
CLIP_START = int(os.environ.get("CLIP_START", "0"))
CLIP_CHUNK = os.environ.get("CLIP_CHUNK", "3116").strip()
REPEAT = int(os.environ.get("REPEAT", "1"))
SKIP_WARMUP = os.environ.get("SKIP_WARMUP", "0").strip().lower() in {"1", "true", "yes", "on"}
ENABLE_PREFIX_CACHING = os.environ.get("ENABLE_PREFIX_CACHING", "0").strip().lower() in {"1", "true", "yes", "on"}
SAMPLE_COUNT_LIST = [
    int(v)
    for v in os.environ.get("SAMPLE_COUNT_LIST", "1,2,4,8,16,32").split(",")
    if v.strip()
]

STAGE0_PROFILE_PATTERN = re.compile(r"\[Stage0Profile\]\s+(\{.*\})")
KV_SEND_REQ_RE = re.compile(r"KV Send req (?P<req>\S+) time_ms=(?P<time_ms>\d+(?:\.\d+)?) start=(?P<start>\d+(?:\.\d+)?) now=(?P<now>\d+(?:\.\d+)?)")
KV_RECEIVE_REQ_RE = re.compile(r"KV Receive req (?P<req>\S+) time_ms=(?P<time_ms>\d+(?:\.\d+)?) start=(?P<start>\d+(?:\.\d+)?) now=(?P<now>\d+(?:\.\d+)?)")
STAGE1_DIFFUSION_REQ_RE = re.compile(r"Stage 1 diffusion req (?P<req>\S+) time_ms=(?P<time_ms>\d+(?:\.\d+)?) start=(?P<start>\d+(?:\.\d+)?) now=(?P<now>\d+(?:\.\d+)?)")
STAGE_GEN_REQ_RE = re.compile(r"Stage (?P<stage_id>\d+) req (?P<req>\S+) gen_time_ms=(?P<time_ms>\d+(?:\.\d+)?) start=(?P<start>\d+(?:\.\d+)?) now=(?P<now>\d+(?:\.\d+)?)")
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_STEM = f"two_stage_profile_vs_tokens_{int(time.time())}"
RUN_DIR = SCRIPT_DIR / "log" / RUN_STEM
METRICS_DIR = RUN_DIR / "metrics"
SVC_LOG_DIR = RUN_DIR / "svc_logs"
TRACE_DIR = RUN_DIR / "torch_traces"
RUNTIME_STAGE_CONFIG_PATH = RUN_DIR / "alpamayo1_5_gpu0_two_stage_profile_vs_tokens.yaml"
RESULT_JSON = METRICS_DIR / "two_stage_profile_vs_tokens_metrics.json"
RESULT_CSV = METRICS_DIR / "two_stage_profile_vs_tokens_metrics.csv"
ENGINE_METRICS_PREFIX = METRICS_DIR / "two_stage_profile_vs_tokens_engine_metrics"
SERVICE_LOG_PATH = SVC_LOG_DIR / "two_stage_profile_vs_tokens_service.log"
SOURCE_STAGE_CONFIG_PATH = OMNI / "profiler" / "alpamayo1_5_gpu0.yaml"


def _load_stage0_max_model_len(config_path: str | Path) -> int:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage_args = config.get("stage_args") or []
    if not stage_args:
        raise RuntimeError(f"unexpected stage config layout in {config_path}")
    engine_args = dict((stage_args[0] or {}).get("engine_args") or {})
    return int(engine_args.get("max_model_len", 8192))


@contextmanager
def _script_profiler_env():
    previous_profiler_dir = os.environ.get("PROFILER_DIR")
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["PROFILER_DIR"] = str(TRACE_DIR)
    try:
        yield
    finally:
        if previous_profiler_dir is None:
            os.environ.pop("PROFILER_DIR", None)
        else:
            os.environ["PROFILER_DIR"] = previous_profiler_dir


def _load_sampling_params_from_yaml(config_path: str | Path, tokenizer) -> tuple[SamplingParams, OmniDiffusionSamplingParams]:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage_args = config.get("stage_args") or []
    if len(stage_args) < 2:
        raise RuntimeError(f"unexpected stage config layout in {config_path}")

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


def _build_runtime_stage_config_local(
    *,
    apply_tokenized_stage0_overrides: bool = True,
    enable_prefix_caching: bool = True,
) -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    yaml_config = yaml.safe_load(SOURCE_STAGE_CONFIG_PATH.read_text(encoding="utf-8"))
    stage_args = yaml_config.get("stage_args") or []
    replacements = 0

    for stage_config in stage_args:
        if stage_config.get("stage_id") == 0:
            stage_config.pop("request_postprocess_func", None)
            stage_config.pop("prompt_rewrite_func", None)
            engine_args = stage_config.setdefault("engine_args", {})
            engine_args["enable_prefix_caching"] = enable_prefix_caching
            engine_args["max_model_len"] = 64000
            if apply_tokenized_stage0_overrides:
                warmup_mod._apply_tokenized_stage0_overrides(stage_config)

        engine_args = stage_config.get("engine_args") or {}
        profiler_config = engine_args.get("profiler_config") or {}
        if "torch_profiler_dir" not in profiler_config:
            continue
        profiler_config["torch_profiler_dir"] = str(TRACE_DIR)
        replacements += 1

    if replacements == 0:
        raise RuntimeError(
            f"No torch_profiler_dir entries found in stage_args engine_args profiler_config of "
            f"{SOURCE_STAGE_CONFIG_PATH}"
        )

    RUNTIME_STAGE_CONFIG_PATH.write_text(
        yaml.safe_dump(yaml_config, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"  Wrote runtime stage config {RUNTIME_STAGE_CONFIG_PATH} "
        f"(torch_profiler_dir -> {TRACE_DIR}, "
        f"tokenized_stage0_overrides={'on' if apply_tokenized_stage0_overrides else 'off'})"
    )
    return RUNTIME_STAGE_CONFIG_PATH


def _start_two_stage_omni(stage_config_path: str | Path, bs_val: int, log_stat: bool = True) -> AsyncOmni:
    kwargs: dict[str, Any] = {
        "stage_configs_path": str(stage_config_path),
    }
    if log_stat:
        kwargs["log_stat_filepath"] = str(ENGINE_METRICS_PREFIX.with_name(f"{ENGINE_METRICS_PREFIX.name}_bs_{bs_val}"))
        kwargs["log_stats"] = True
    return AsyncOmni(model=MODEL, **kwargs)


def _build_sample_dict(model_input: dict[str, Any], cid: str) -> dict[str, Any]:
    return {
        "cid": str(cid),
        "prompt": _clone_value(model_input.get("prompt")),
        "prompt_token_ids": list(model_input.get("prompt_token_ids") or []),
        "postprocess_prompt_token_ids": list(
            model_input.get("postprocess_prompt_token_ids") or model_input.get("prompt_token_ids") or []
        ),
        "multi_modal_data": _clone_value(model_input.get("multi_modal_data") or {}),
        "mm_processor_kwargs": _clone_value(model_input.get("mm_processor_kwargs")),
        "hf_processor_mm_kwargs": _clone_value(model_input.get("hf_processor_mm_kwargs")),
        "additional_information": _clone_value(model_input.get("additional_information") or {}),
    }


def _merge_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty")

    merged_prompt_parts: list[str] = []
    merged_prompt_token_ids: list[int] = []
    merged_postprocess_prompt_token_ids: list[int] = []
    merged_images: list[Any] = []
    merged_additional_information = _clone_value(samples[0]["additional_information"])
    merged_multi_modal_data = _clone_value(samples[0].get("multi_modal_data") or {})

    tokenized_data = dict(merged_additional_information.get("tokenized_data") or {})
    pixel_values_parts: list[torch.Tensor] = []
    image_grid_parts: list[torch.Tensor] = []
    ego_history_xyz_parts: list[torch.Tensor] = []
    ego_history_rot_parts: list[torch.Tensor] = []
    ego_future_xyz_parts: list[torch.Tensor] = []
    ego_future_rot_parts: list[torch.Tensor] = []

    for sample in samples:
        raw_prompt = sample.get("prompt")
        if isinstance(raw_prompt, str) and raw_prompt:
            merged_prompt_parts.append(raw_prompt)
        merged_prompt_token_ids.extend(list(sample.get("prompt_token_ids") or []))
        merged_postprocess_prompt_token_ids.extend(list(sample.get("postprocess_prompt_token_ids") or []))
        merged_images.extend(list((sample.get("multi_modal_data") or {}).get("image") or []))

        sample_info = sample.get("additional_information") or {}
        sample_tokenized = dict(sample_info.get("tokenized_data") or {})

        pixel_values = sample_tokenized.get("pixel_values")
        if pixel_values is not None:
            pixel_values_parts.append(torch.as_tensor(pixel_values).contiguous())

        image_grid_thw = sample_tokenized.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_parts.append(torch.as_tensor(image_grid_thw, dtype=torch.long).contiguous())

        for key, parts in [
            ("ego_history_xyz", ego_history_xyz_parts),
            ("ego_history_rot", ego_history_rot_parts),
            ("ego_future_xyz", ego_future_xyz_parts),
            ("ego_future_rot", ego_future_rot_parts),
        ]:
            value = sample_info.get(key)
            if value is not None:
                parts.append(torch.as_tensor(value).contiguous())

    tokenized_data["input_ids"] = torch.tensor(merged_postprocess_prompt_token_ids, dtype=torch.long).unsqueeze(0)
    tokenized_data["attention_mask"] = torch.ones((1, len(merged_postprocess_prompt_token_ids)), dtype=torch.long)
    if pixel_values_parts:
        tokenized_data["pixel_values"] = torch.cat(pixel_values_parts, dim=0)
    if image_grid_parts:
        tokenized_data["image_grid_thw"] = torch.cat(image_grid_parts, dim=0)
    merged_additional_information["tokenized_data"] = tokenized_data
    if ego_history_xyz_parts:
        merged_additional_information["ego_history_xyz"] = torch.cat(ego_history_xyz_parts, dim=0)
    if ego_history_rot_parts:
        merged_additional_information["ego_history_rot"] = torch.cat(ego_history_rot_parts, dim=0)
    if ego_future_xyz_parts:
        merged_additional_information["ego_future_xyz"] = torch.cat(ego_future_xyz_parts, dim=0)
    if ego_future_rot_parts:
        merged_additional_information["ego_future_rot"] = torch.cat(ego_future_rot_parts, dim=0)

    if merged_images:
        merged_multi_modal_data["image"] = merged_images

    merged_prompt: dict[str, Any] = {
        "prompt_token_ids": merged_prompt_token_ids,
        "postprocess_prompt_token_ids": merged_postprocess_prompt_token_ids,
        "additional_information": merged_additional_information,
    }
    if merged_prompt_parts:
        merged_prompt["prompt"] = "\n".join(merged_prompt_parts)
    if merged_multi_modal_data:
        merged_prompt["multi_modal_data"] = merged_multi_modal_data
    if samples[0].get("mm_processor_kwargs") is not None:
        merged_prompt["mm_processor_kwargs"] = _clone_value(samples[0].get("mm_processor_kwargs"))
    if samples[0].get("hf_processor_mm_kwargs") is not None:
        merged_prompt["hf_processor_mm_kwargs"] = _clone_value(samples[0].get("hf_processor_mm_kwargs"))
    return merged_prompt


async def _run_request(
    omni: AsyncOmni,
    sampling_params_list: list[Any] | tuple[Any, ...],
    samples: list[dict[str, Any]],
    request_id: str,
):
    prompt = _merge_samples(samples)
    final_output = None
    async for out in omni.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=list(sampling_params_list),
    ):
        final_output = out
    if final_output is None:
        raise RuntimeError(f"request {request_id} produced no output")
    return final_output


def _extract_stage0_profiles(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log_path.exists():
        return rows
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = STAGE0_PROFILE_PATTERN.search(line)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _matches_request_id(logged_request_id: Any, request_id: str) -> bool:
    if not isinstance(logged_request_id, str):
        return False
    return logged_request_id == request_id or request_id in logged_request_id


def _flatten_profile_list(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        raw_value = row.get(key, [])
        if isinstance(raw_value, list):
            values.extend(float(item) for item in raw_value)
    return values


def _flatten_profile_int_list(rows: list[dict[str, Any]], key: str) -> list[int]:
    values: list[int] = []
    for row in rows:
        raw_value = row.get(key, [])
        if isinstance(raw_value, list):
            values.extend(int(item) for item in raw_value)
    return values


def _float_or_sum(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().sum().item())
    if isinstance(value, (list, tuple)):
        return float(sum(_float_or_sum(item) for item in value))
    return float(value or 0.0)


def _count_output_tokens(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, list):
        total = 0
        for item in value:
            if isinstance(item, torch.Tensor):
                total += int(item.numel())
            elif isinstance(item, list):
                total += _count_output_tokens(item)
            else:
                total += 1
        return total
    return 0


def _build_result_from_output(output, request_id: str) -> dict[str, Any]:
    metrics = dict(getattr(output, "metrics", {}) or {})
    custom_output = dict(metrics.get("custom_output", {}) or {})
    custom_output.update(dict(getattr(output, "custom_output", {}) or {}))

    s0_llm_ms = float(metrics.get("stage0_llm_ms", 0.0) or 0.0)
    s1_diffusion_ms = _float_or_sum(
        custom_output.get("df_ms", custom_output.get("s1_diffusion_ms", custom_output.get("stage1_diffusion_ms", 0.0)))
    )
    stage1_forward_total_ms = _float_or_sum(custom_output.get("stage1_forward_total_ms", 0.0))
    stage1_diffusion_inner_ms = _float_or_sum(custom_output.get("stage1_diffusion_inner_ms", 0.0))
    kv_s0_extract_ms = _float_or_sum(custom_output.get("kv_tran_s0_ms", custom_output.get("kv_s0_extract_ms", 0.0)))
    kv_s1_receive_ms = _float_or_sum(
        custom_output.get("kv_tran_s1_receive_ms", custom_output.get("kv_s1_receive_ms", 0.0))
    )
    kv_s1_tran_ms = _float_or_sum(custom_output.get("kv_s1_tran_ms", 0.0))
    kv_s1_prep_ms = _float_or_sum(custom_output.get("kv_tran_s1_prep_ms", custom_output.get("kv_s1_prep_ms", 0.0)))
    kv_s0_extract_plus_transfer_ms = kv_s0_extract_ms
    kv_s0_start_time = float(custom_output.get("kv_s0_start_time", 0.0) or 0.0)
    kv_s1_end_time = float(custom_output.get("kv_s1_end_time", 0.0) or 0.0)
    kv_s1_receive_start_time = (
        kv_s1_end_time - (kv_s1_tran_ms / 1000.0) if kv_s1_end_time and kv_s1_tran_ms > 0.0 else 0.0
    )
    stage1_diffusion_end_time = float(custom_output.get("stage1_diffusion_end_time", 0.0) or 0.0)
    stage1_diffusion_start_time = (
        stage1_diffusion_end_time - (stage1_forward_total_ms / 1000.0)
        if stage1_diffusion_end_time and stage1_forward_total_ms > 0.0
        else 0.0
    )
    stage1_submit_start = (
        stage1_diffusion_end_time - (float(metrics.get("stage1_diffusion_ms", 0.0) or 0.0) / 1000.0)
        if stage1_diffusion_end_time and float(metrics.get("stage1_diffusion_ms", 0.0) or 0.0) > 0.0
        else 0.0
    )
    kv_tran_total = (kv_s1_end_time - kv_s0_start_time) * 1000.0 if kv_s0_start_time and kv_s1_end_time else 0.0

    cot_token_ids = custom_output.get("cot_token_ids")
    output_tokens = _count_output_tokens(cot_token_ids)

    return {
        "request_id": request_id,
        "lat": 0.0,
        "inf": float(metrics.get("inference_only_ms", 0.0) or 0.0),
        "net": 0.0,
        "stage0_llm_ms": s0_llm_ms,
        "stage0_submit_start": 0.0,
        "stage0_submit_end": 0.0,
        "subsequent_decode_ms": 0.0,
        "kv_transfer_ms": kv_tran_total,
        "diffusion_ms": s1_diffusion_ms,
        "stage1_forward_total_ms": stage1_forward_total_ms,
        "stage1_diffusion_inner_ms": stage1_diffusion_inner_ms,
        "kv_tran_total": kv_tran_total,
        "kv_s0_start_time": kv_s0_start_time,
        "kv_s0_end_time": 0.0,
        "kv_s0_extract_ms": kv_s0_extract_ms,
        "kv_s0_transfer_only_ms": 0.0,
        "kv_s0_extract_plus_transfer_ms": kv_s0_extract_plus_transfer_ms,
        "kv_s1_receive_ms": kv_s1_receive_ms,
        "kv_s1_receive_start_time": kv_s1_receive_start_time,
        "kv_s1_tran_ms": kv_s1_tran_ms,
        "kv_s1_end_time": kv_s1_end_time,
        "kv_s1_prep_ms": kv_s1_prep_ms,
        "stage1_submit_start": stage1_submit_start,
        "stage1_submit_end": stage1_diffusion_end_time,
        "stage1_diffusion_start_time": stage1_diffusion_start_time,
        "stage1_diffusion_end_time": stage1_diffusion_end_time,
        "s1_diffusion_ms": s1_diffusion_ms,
        "input_tokens": int(metrics.get("input_tokens", 0) or 0),
        "output_tokens": output_tokens,
    }


def _load_kv_metrics_from_log(log_path: Path) -> dict[str, dict[str, float]]:
    kv_metrics_by_rid: dict[str, dict[str, float]] = {}
    if not log_path.exists():
        return kv_metrics_by_rid

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = warmup_mod.KV_TIMING_RE.search(line)
            if match:
                request_id = match.group("req")
                entry = kv_metrics_by_rid.setdefault(request_id, {})
                entry.update(
                    {
                        "extract_only_ms": float(match.group("extract_only_ms")),
                        "transfer_only_ms": float(match.group("transfer_only_ms")),
                        "extract_plus_transfer_ms": float(match.group("extract_plus_transfer_ms")),
                    }
                )
                continue

            send_match = KV_SEND_REQ_RE.search(line)
            if send_match:
                request_id = send_match.group("req")
                entry = kv_metrics_by_rid.setdefault(request_id, {})
                entry.update(
                    {
                        "kv_send_start": float(send_match.group("start")),
                        "kv_send_end": float(send_match.group("now")),
                        "kv_send_time_ms": float(send_match.group("time_ms")),
                    }
                )
                continue

            receive_match = KV_RECEIVE_REQ_RE.search(line)
            if receive_match:
                request_id = receive_match.group("req")
                entry = kv_metrics_by_rid.setdefault(request_id, {})
                entry.update(
                    {
                        "kv_receive_start": float(receive_match.group("start")),
                        "kv_receive_end": float(receive_match.group("now")),
                        "kv_receive_time_ms": float(receive_match.group("time_ms")),
                    }
                )
                continue

            stage1_diff_match = STAGE1_DIFFUSION_REQ_RE.search(line)
            if stage1_diff_match:
                request_id = stage1_diff_match.group("req")
                entry = kv_metrics_by_rid.setdefault(request_id, {})
                entry.update(
                    {
                        "stage1_diffusion_start": float(stage1_diff_match.group("start")),
                        "stage1_diffusion_end": float(stage1_diff_match.group("now")),
                        "stage1_diffusion_time_ms": float(stage1_diff_match.group("time_ms")),
                    }
                )
                continue

            stage_gen_match = STAGE_GEN_REQ_RE.search(line)
            if stage_gen_match:
                request_id = stage_gen_match.group("req")
                stage_id = int(stage_gen_match.group("stage_id"))
                entry = kv_metrics_by_rid.setdefault(request_id, {})
                entry.update(
                    {
                        f"stage{stage_id}_submit_start": float(stage_gen_match.group("start")),
                        f"stage{stage_id}_submit_end": float(stage_gen_match.group("now")),
                        f"stage{stage_id}_gen_time_ms": float(stage_gen_match.group("time_ms")),
                    }
                )
    return kv_metrics_by_rid


def _write_results(rows: list[dict[str, Any]]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "request_id",
        "sample_count",
        "clip_ids",
        "repeat_index",
        "total_prompt_tokens",
        "total_postprocess_prompt_tokens",
        "mean_prompt_tokens_per_sample",
        "logged_profile_rows",
        "logged_prompt_tokens_sum",
        "batch_size",
        "prompt_tokens",
        "input_tokens",
        "output_tokens",
        "scheduled_tokens",
        "batch_total_scheduled_tokens",
        "lat",
        "inf",
        "net",
        "stage0_llm_ms",
        "stage0_submit_start",
        "stage0_submit_end",
        "subsequent_decode_ms",
        "kv_transfer_ms",
        "diffusion_ms",
        "embed_multimodal_ms",
        "forward_ms",
        "stage1_forward_total_ms",
        "stage1_diffusion_inner_ms",
        "kv_tran_total",
        "kv_s0_extract_ms",
        "kv_s0_transfer_only_ms",
        "kv_s0_extract_plus_transfer_ms",
        "kv_s1_receive_ms",
        "kv_s1_receive_start_time",
        "kv_s1_tran_ms",
        "kv_s1_end_time",
        "kv_s1_prep_ms",
        "stage1_submit_start",
        "stage1_submit_end",
        "stage1_diffusion_start_time",
        "stage1_diffusion_end_time",
        "s1_diffusion_ms",
        "encoder_cache_hit",
        "encoder_cache_miss",
        "encoder_cache_skipped",
        "encoder_not_needed_this_step",
        "embed_multimodal_ms_list",
        "forward_ms_list",
        "encoder_cache_hit_list",
        "encoder_cache_miss_list",
        "encoder_cache_skipped_list",
        "encoder_not_needed_this_step_list",
        "embed_start",
        "embed_end",
        "forward_start",
        "forward_end",
        "kv_s0_start_time",
        "kv_s0_end_time",
        "start",
        "now",
    ]
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


async def main_async() -> None:
    if not SAMPLE_COUNT_LIST:
        raise RuntimeError("SAMPLE_COUNT_LIST must not be empty")

    sample_counts = sorted({count for count in SAMPLE_COUNT_LIST if count > 0})
    if not sample_counts:
        raise RuntimeError(f"No positive values in SAMPLE_COUNT_LIST={SAMPLE_COUNT_LIST}")

    print(
        f"Two-stage token sweep: sample_counts={sample_counts}, repeat={REPEAT}, bs={BS}, "
        f"enable_prefix_caching={ENABLE_PREFIX_CACHING}"
    )
    runtime_stage_config_path = _build_runtime_stage_config_local(
        apply_tokenized_stage0_overrides=False,
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
    )
    tokenizer = warmup_mod.build_alpamayo_stage0_tokenizer(MODEL)
    sampling_params_list = _load_sampling_params_from_yaml(runtime_stage_config_path, tokenizer)
    max_model_len = _load_stage0_max_model_len(runtime_stage_config_path)

    print("Preparing tokenized samples...")
    components = _build_model_input_components()
    ci = components["avdi"].clip_index
    if CLIP_CHUNK:
        selected = list(ci[ci.chunk == int(CLIP_CHUNK)].index)
    else:
        selected = list(ci.index)
    if not selected:
        raise RuntimeError("No clip ids available for the requested CLIP_CHUNK")

    warmup_sample_count = 0 if SKIP_WARMUP else 1
    total_required_samples = sum(sample_counts) + warmup_sample_count
    if CLIP_START < 0 or CLIP_START >= len(selected):
        raise IndexError(f"CLIP_START={CLIP_START} out of range for {len(selected)} available clips")
    if CLIP_START + total_required_samples > len(selected):
        raise IndexError(
            f"Need {total_required_samples} samples from CLIP_START={CLIP_START}, but only "
            f"{len(selected) - CLIP_START} remain"
        )

    selected_clip_ids = [str(cid) for cid in selected[CLIP_START : CLIP_START + total_required_samples]]
    prepared_samples: list[dict[str, Any]] = []
    for idx, cid in enumerate(selected_clip_ids):
        model_input = _build_model_input(
            clip_id=cid,
            t0_us=T0_US,
            preprocess_fn=components["preprocess_fn"],
            processor=components["processor"],
            model_config=components["model_config"],
            traj_fuser=components["traj_fuser"],
            avdi=components["avdi"],
        )
        sample = _build_sample_dict(model_input, cid)
        prepared_samples.append(sample)
        print(f"Prepared sample[{idx}] clip={cid[:8]} prompt_tokens={len(sample['prompt_token_ids'])}")

    warmup_sample = prepared_samples[0] if warmup_sample_count else None
    samples = prepared_samples[warmup_sample_count:]

    SVC_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SERVICE_LOG_PATH
    log_path.write_text("", encoding="utf-8")
    print(f"Service logs -> {log_path}")

    run_specs: list[dict[str, Any]] = []
    sample_offset = 0
    for sample_count in sample_counts:
        merged_samples = samples[sample_offset : sample_offset + sample_count]
        if len(merged_samples) < sample_count:
            print(
                f"Skip sample_count={sample_count}: need {sample_count} samples from offset={sample_offset}, "
                f"but only {len(merged_samples)} remain"
            )
            continue
        total_prompt_tokens = sum(len(sample["prompt_token_ids"]) for sample in merged_samples)
        total_postprocess_prompt_tokens = sum(len(sample["postprocess_prompt_token_ids"]) for sample in merged_samples)
        if total_postprocess_prompt_tokens > max_model_len:
            print(
                f"Skip sample_count={sample_count}: postprocess_prompt_tokens={total_postprocess_prompt_tokens} "
                f"> max_model_len={max_model_len}"
            )
            sample_offset += sample_count
            continue
        run_specs.append(
            {
                "sample_count": sample_count,
                "merged_samples": merged_samples,
                "clip_ids": ",".join(str(sample["cid"]) for sample in merged_samples),
                "total_prompt_tokens": total_prompt_tokens,
                "total_postprocess_prompt_tokens": total_postprocess_prompt_tokens,
                "sample_offset": sample_offset,
            }
        )
        sample_offset += sample_count

    results: list[dict[str, Any]] = []
    try:
        for repeat_index in range(REPEAT):
            omni = None
            with _script_profiler_env():
                with _omni_env():
                    with _redirect_process_output(log_path):
                        omni = _start_two_stage_omni(runtime_stage_config_path, BS)
            try:
                if not SKIP_WARMUP:
                    if warmup_sample is None:
                        raise RuntimeError("warmup sample is missing")
                    print(f"Warmup once with sample_count=1 repeat={repeat_index}")
                    with _redirect_process_output(log_path):
                        await _run_request(omni, sampling_params_list, [warmup_sample], f"two-stage-sweep-warmup-r{repeat_index}")

                for spec in run_specs:
                    sample_count = int(spec["sample_count"])
                    merged_samples = list(spec["merged_samples"])
                    request_id = f"two-stage-sweep-n{sample_count}-r{repeat_index}"
                    total_prompt_tokens = int(spec["total_prompt_tokens"])
                    total_postprocess_prompt_tokens = int(spec["total_postprocess_prompt_tokens"])
                    print(
                        f"Run request sample_count={sample_count} total_prompt_tokens={total_prompt_tokens} "
                        f"postprocess_prompt_tokens={total_postprocess_prompt_tokens} repeat={repeat_index}"
                    )
                    with _redirect_process_output(log_path):
                        start = asyncio.get_running_loop().time()
                        final_output = await _run_request(omni, sampling_params_list, merged_samples, request_id)
                        lat_ms = (asyncio.get_running_loop().time() - start) * 1000.0
                    row = _build_result_from_output(final_output, request_id)
                    row.update(
                        {
                            "sample_count": sample_count,
                            "clip_ids": str(spec["clip_ids"]),
                            "repeat_index": repeat_index,
                            "total_prompt_tokens": total_prompt_tokens,
                            "total_postprocess_prompt_tokens": total_postprocess_prompt_tokens,
                            "mean_prompt_tokens_per_sample": total_prompt_tokens / sample_count,
                            "lat": lat_ms,
                            "net": lat_ms - row["inf"],
                        }
                    )
                    results.append(row)
            finally:
                if omni is not None:
                    with _redirect_process_output(log_path):
                        try:
                            await omni.engine.shutdown_background_loop()
                        except Exception:
                            pass
                        omni.shutdown()

        profile_rows = _extract_stage0_profiles(log_path)
        kv_metrics_by_rid = _load_kv_metrics_from_log(log_path)

        for row in results:
            request_id = str(row["request_id"])
            matched_rows = [profile for profile in profile_rows if _matches_request_id(profile.get("request_id"), request_id)]
            if matched_rows:
                row.update(
                    {
                        "batch_size": max(int(profile.get("batch_size", 0) or 0) for profile in matched_rows),
                        "batch_total_scheduled_tokens": sum(
                            int(profile.get("batch_total_scheduled_tokens", 0) or 0) for profile in matched_rows
                        ),
                        "scheduled_tokens": sum(int(profile.get("scheduled_tokens", 0) or 0) for profile in matched_rows),
                        "prompt_tokens": max(int(profile.get("prompt_tokens", 0) or 0) for profile in matched_rows),
                        "embed_multimodal_ms": sum(
                            float(profile.get("embed_multimodal_ms", 0.0) or 0.0) for profile in matched_rows
                        ),
                        "forward_ms": sum(float(profile.get("forward_ms", 0.0) or 0.0) for profile in matched_rows),
                        "embed_start": min(float(profile.get("embed_start", 0.0) or 0.0) for profile in matched_rows),
                        "embed_end": max(float(profile.get("embed_end", 0.0) or 0.0) for profile in matched_rows),
                        "forward_start": min(float(profile.get("forward_start", 0.0) or 0.0) for profile in matched_rows),
                        "forward_end": max(float(profile.get("forward_end", 0.0) or 0.0) for profile in matched_rows),
                        "encoder_cache_hit": sum(int(profile.get("encoder_cache_hit", 0) or 0) for profile in matched_rows),
                        "encoder_cache_miss": sum(int(profile.get("encoder_cache_miss", 0) or 0) for profile in matched_rows),
                        "encoder_cache_skipped": sum(
                            int(profile.get("encoder_cache_skipped", 0) or 0) for profile in matched_rows
                        ),
                        "encoder_not_needed_this_step": sum(
                            int(profile.get("encoder_not_needed_this_step", 0) or 0) for profile in matched_rows
                        ),
                        "embed_multimodal_ms_list": _flatten_profile_list(matched_rows, "embed_multimodal_ms_list"),
                        "forward_ms_list": _flatten_profile_list(matched_rows, "forward_ms_list"),
                        "encoder_cache_hit_list": _flatten_profile_int_list(matched_rows, "encoder_cache_hit_list"),
                        "encoder_cache_miss_list": _flatten_profile_int_list(matched_rows, "encoder_cache_miss_list"),
                        "encoder_cache_skipped_list": _flatten_profile_int_list(
                            matched_rows, "encoder_cache_skipped_list"
                        ),
                        "encoder_not_needed_this_step_list": _flatten_profile_int_list(
                            matched_rows, "encoder_not_needed_this_step_list"
                        ),
                        "start": min(float(profile.get("start", 0.0) or 0.0) for profile in matched_rows),
                        "now": max(float(profile.get("now", 0.0) or 0.0) for profile in matched_rows),
                    }
                )
            row["logged_profile_rows"] = len(matched_rows)
            row["logged_prompt_tokens_sum"] = sum(int(profile.get("prompt_tokens", 0) or 0) for profile in matched_rows)

            log_kv = kv_metrics_by_rid.get(request_id)
            if log_kv:
                if "extract_only_ms" in log_kv:
                    row["kv_s0_extract_ms"] = log_kv["extract_only_ms"]
                if "transfer_only_ms" in log_kv:
                    row["kv_s0_transfer_only_ms"] = log_kv["transfer_only_ms"]
                if "extract_plus_transfer_ms" in log_kv:
                    row["kv_s0_extract_plus_transfer_ms"] = log_kv["extract_plus_transfer_ms"]
                if "kv_send_start" in log_kv:
                    row["kv_s0_start_time"] = log_kv["kv_send_start"]
                if "kv_send_end" in log_kv:
                    row["kv_s0_end_time"] = log_kv["kv_send_end"]
                if "kv_receive_start" in log_kv:
                    row["kv_s1_receive_start_time"] = log_kv["kv_receive_start"]
                if "kv_receive_end" in log_kv:
                    row["kv_s1_end_time"] = log_kv["kv_receive_end"]
                if "stage0_submit_start" in log_kv:
                    row["stage0_submit_start"] = log_kv["stage0_submit_start"]
                if "stage0_submit_end" in log_kv:
                    row["stage0_submit_end"] = log_kv["stage0_submit_end"]
                if "stage1_submit_start" in log_kv:
                    row["stage1_submit_start"] = log_kv["stage1_submit_start"]
                if "stage1_submit_end" in log_kv:
                    row["stage1_submit_end"] = log_kv["stage1_submit_end"]
                if "stage1_diffusion_start" in log_kv:
                    row["stage1_diffusion_start_time"] = log_kv["stage1_diffusion_start"]
                if "stage1_diffusion_end" in log_kv:
                    row["stage1_diffusion_end_time"] = log_kv["stage1_diffusion_end"]

            kv_s0_start_time = float(row.get("kv_s0_start_time", 0.0) or 0.0)
            kv_s1_end_time = float(row.get("kv_s1_end_time", 0.0) or 0.0)
            row["kv_tran_total"] = (
                (kv_s1_end_time - kv_s0_start_time) * 1000.0 if kv_s0_start_time and kv_s1_end_time else 0.0
            )
            row["kv_transfer_ms"] = row["kv_tran_total"]
            row["diffusion_ms"] = float(row.get("s1_diffusion_ms", 0.0) or 0.0)

            kv_s0_start_time = float(row.get("kv_s0_start_time", 0.0) or 0.0)
            forward_end = float(row.get("forward_end", 0.0) or 0.0)
            row["subsequent_decode_ms"] = max((kv_s0_start_time - forward_end) * 1000.0, 0.0)

        _write_results(results)
        print(f"Results JSON -> {RESULT_JSON}")
        print(f"Results CSV  -> {RESULT_CSV}")
    finally:
        _cleanup_runtime()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
