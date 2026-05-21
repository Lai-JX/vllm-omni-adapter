"""Collect stage-0 timing against merged natural-sample token count using tokenized Alpamayo inputs.

This script reuses the tokenized data-building path from
profiler/batch_sweep_cont_inproc_tokenized_warmup.py, but focuses only on
collecting the stage-0 profile log fields:
- embed_multimodal_ms
- forward_ms

Each datapoint merges multiple naturally tokenized samples into a single
request. The script varies the number of merged samples, runs one request per
sample-count setting, then extracts the emitted [Stage0Profile] JSON lines into
a CSV/JSON summary.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from vllm import SamplingParams

OMNI = Path(__file__).resolve().parents[1]
if str(OMNI) not in sys.path:
    sys.path.insert(0, str(OMNI))

import profiler.batch_sweep_cont_inproc_tokenized_warmup as warmup_mod  # noqa: E402
from profiler.batch_sweep_cont_inproc_tokenized_warmup import (  # noqa: E402
    ASYNC_OMNI_LOG_DIR,
    LOG_DIR,
    MODEL,
    OUT,
    RUNTIME_SVC_YAML,
    T0_US,
    _build_model_input,
    _build_model_input_components,
    _build_runtime_stage_config,
    _cleanup_runtime,
    _clone_value,
    _materialize_prompt,
    _omni_env,
    _output_prefix,
    _redirect_process_output,
)
from vllm_omni.entrypoints.async_omni import AsyncOmni  # noqa: E402

BS = int(os.environ.get("BS", "1"))
CLIP_START = int(os.environ.get("CLIP_START", "0"))
CLIP_CHUNK = os.environ.get("CLIP_CHUNK", "3116").strip()
REPEAT = int(os.environ.get("REPEAT", "1"))
SKIP_WARMUP = os.environ.get("SKIP_WARMUP", "0").strip().lower() in {"1", "true", "yes", "on"}
SAMPLE_COUNT_LIST = [
    int(v)
    for v in os.environ.get("SAMPLE_COUNT_LIST", "1,2,4,8,16,32").split(",")
    if v.strip()
]

STAGE0_PROFILE_PATTERN = re.compile(r"\[Stage0Profile\]\s+(\{.*\})")
RESULT_JSON = LOG_DIR / "metrics" / "stage0_profile_vs_tokens.json"
RESULT_CSV = LOG_DIR / "metrics" / "stage0_profile_vs_tokens.csv"


def _load_stage0_max_model_len(config_path: str | Path = RUNTIME_SVC_YAML) -> int:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage_args = config.get("stage_args") or []
    if not stage_args:
        raise RuntimeError(f"unexpected stage config layout in {config_path}")
    engine_args = dict((stage_args[0] or {}).get("engine_args") or {})
    return int(engine_args.get("max_model_len", 8192))


def _build_stage0_only_sampling_params() -> SamplingParams:
    return SamplingParams(
        temperature=0.6,
        top_p=0.98,
        top_k=40,
        max_tokens=256,
        stop_token_ids=[155681],
        detokenize=False,
        seed=42,
        n=1,
    )


def _write_temp_yaml(config: dict[str, Any], prefix: str) -> str:
    fd, temp_path = tempfile.mkstemp(prefix=prefix, suffix=".yaml")
    os.close(fd)
    Path(temp_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return temp_path


def _build_stage0_only_yaml(src_yaml_path: Path = RUNTIME_SVC_YAML) -> str:
    config = yaml.safe_load(src_yaml_path.read_text(encoding="utf-8"))
    stage_args = config.get("stage_args") or []
    if not stage_args:
        raise RuntimeError(f"unexpected stage config layout in {src_yaml_path}")

    stage0 = dict(stage_args[0] or {})
    stage0["final_output"] = True
    stage0["final_output_type"] = "latent"
    stage0.pop("output_connectors", None)

    engine_args = dict(stage0.get("engine_args") or {})
    max_model_len = int(os.environ.get("STAGE0_MAX_MODEL_LEN", "64000"))
    # stage0_gpu_mem_util = float(os.environ.get("STAGE0_GPU_MEMORY_UTILIZATION", "0.15"))
    # engine_args["gpu_memory_utilization"] = 0.95
    engine_args["max_model_len"] = max_model_len
    engine_args["max_num_seqs"] = 1
    engine_args["max_num_batched_tokens"] = max_model_len
    stage0["engine_args"] = engine_args

    config["stage_args"] = [stage0]
    runtime_cfg = dict(config.get("runtime") or {})
    runtime_cfg.pop("connectors", None)
    runtime_cfg.pop("edges", None)
    config["runtime"] = runtime_cfg
    return _write_temp_yaml(config, prefix="alpamayo_stage0_profile_")


def _start_stage0_only_omni(stage0_yaml_path: str, bs_val: int, log_stat: bool = True) -> AsyncOmni:
    kwargs: dict[str, Any] = {
        "stage_configs_path": stage0_yaml_path,
    }
    if log_stat:
        output_prefix = _output_prefix(OUT)
        log_stat_filepath = output_prefix.with_name(f"{output_prefix.name}_engine_metrics_bs_{bs_val}")
        kwargs["log_stat_filepath"] = str(log_stat_filepath)
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


async def _run_request(omni, sampling_params: SamplingParams, samples: list[dict[str, Any]], request_id: str) -> None:
    prompt = _merge_samples(samples)
    final_output = None
    async for out in omni.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=[sampling_params],
    ):
        final_output = out
    if final_output is None:
        raise RuntimeError(f"request {request_id} produced no output")


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


def _write_results(rows: list[dict[str, Any]]) -> None:
    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fieldnames = [
        "request_id",
        "sample_count",
        "clip_ids",
        "repeat_index",
        "total_prompt_tokens",
        "total_postprocess_prompt_tokens",
        "mean_prompt_tokens_per_sample",
        "logged_prompt_tokens_sum",
        "logged_profile_rows",
        "scheduled_tokens",
        "batch_total_scheduled_tokens",
        "output_tokens",
        "embed_multimodal_ms",
        "forward_ms",
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
        "batch_size",
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

    print(f"Stage0 token sweep: sample_counts={sample_counts}, repeat={REPEAT}, bs={BS}")
    runtime_stage_config_path = _build_runtime_stage_config(apply_tokenized_stage0_overrides=True, enable_prefix_caching=False)
    stage0_yaml_path = _build_stage0_only_yaml(runtime_stage_config_path)
    sampling_params = _build_stage0_only_sampling_params()
    max_model_len = _load_stage0_max_model_len(stage0_yaml_path)

    print("Preparing tokenized samples...")
    components = _build_model_input_components()
    ci = components["avdi"].clip_index
    if CLIP_CHUNK:
        selected = list(ci[ci.chunk == int(CLIP_CHUNK)].index)
    else:
        selected = list(ci.index)
    if not selected:
        raise RuntimeError("No clip ids available for the requested CLIP_CHUNK")

    total_required_samples = sum(sample_counts)
    if CLIP_START < 0 or CLIP_START >= len(selected):
        raise IndexError(f"CLIP_START={CLIP_START} out of range for {len(selected)} available clips")
    if CLIP_START + total_required_samples > len(selected):
        raise IndexError(
            f"Need {total_required_samples} samples from CLIP_START={CLIP_START}, but only {len(selected) - CLIP_START} remain"
        )

    selected_clip_ids = [str(cid) for cid in selected[CLIP_START : CLIP_START + total_required_samples]]
    samples: list[dict[str, Any]] = []
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
        samples.append(sample)
        print(f"Prepared sample[{idx}] clip={cid[:8]} prompt_tokens={len(sample['prompt_token_ids'])}")

    ASYNC_OMNI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ASYNC_OMNI_LOG_DIR / "svc_stage0_profile_sweep.log"
    log_path.write_text("", encoding="utf-8")
    print(f"Service logs -> {log_path}")

    expected_runs: list[dict[str, Any]] = []
    sample_offset = 0
    run_specs: list[dict[str, Any]] = []
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
        clip_ids = ",".join(str(sample["cid"]) for sample in merged_samples)
        run_specs.append(
            {
                "sample_count": sample_count,
                "merged_samples": merged_samples,
                "clip_ids": clip_ids,
                "total_prompt_tokens": total_prompt_tokens,
                "total_postprocess_prompt_tokens": total_postprocess_prompt_tokens,
                "sample_offset": sample_offset,
            }
        )
        sample_offset += sample_count

    try:
        for repeat_index in range(REPEAT):
            omni = None
            with _omni_env():
                with _redirect_process_output(log_path):
                    omni = _start_stage0_only_omni(stage0_yaml_path, BS)
            try:
                if not SKIP_WARMUP:
                    print(f"Warmup once with sample_count=1 repeat={repeat_index}")
                    with _redirect_process_output(log_path):
                        await _run_request(omni, sampling_params, [samples[0]], f"stage0-sweep-warmup-r{repeat_index}")

                for spec in run_specs:
                    sample_count = int(spec["sample_count"])
                    merged_samples = spec["merged_samples"]
                    clip_ids = str(spec["clip_ids"])
                    total_prompt_tokens = int(spec["total_prompt_tokens"])
                    total_postprocess_prompt_tokens = int(spec["total_postprocess_prompt_tokens"])
                    current_sample_offset = int(spec["sample_offset"])
                    request_id = f"stage0-sweep-n{sample_count}-r{repeat_index}"
                    expected_runs.append(
                        {
                            "request_id": request_id,
                            "sample_count": sample_count,
                            "clip_ids": clip_ids,
                            "repeat_index": repeat_index,
                            "total_prompt_tokens": total_prompt_tokens,
                            "total_postprocess_prompt_tokens": total_postprocess_prompt_tokens,
                            "mean_prompt_tokens_per_sample": total_prompt_tokens / sample_count,
                        }
                    )
                    print(
                        f"Run request sample_count={sample_count} total_prompt_tokens={total_prompt_tokens} "
                        f"postprocess_prompt_tokens={total_postprocess_prompt_tokens} repeat={repeat_index} "
                        f"sample_offset={current_sample_offset}"
                    )
                    with _redirect_process_output(log_path):
                        await _run_request(omni, sampling_params, merged_samples, request_id)
            finally:
                if omni is not None:
                    with _redirect_process_output(log_path):
                        try:
                            await omni.engine.shutdown_background_loop()
                        except Exception:
                            pass
                        omni.shutdown()

        profile_rows = _extract_stage0_profiles(log_path)
        results: list[dict[str, Any]] = []
        for run in expected_runs:
            request_id = str(run["request_id"])
            matched_rows = [row for row in profile_rows if _matches_request_id(row.get("request_id"), request_id)]
            result: dict[str, Any] = {}
            if matched_rows:
                result = {
                    "request_id": request_id,
                    "batch_size": max(int(row.get("batch_size", 0) or 0) for row in matched_rows),
                    "batch_total_scheduled_tokens": sum(
                        int(row.get("batch_total_scheduled_tokens", 0) or 0) for row in matched_rows
                    ),
                    "scheduled_tokens": sum(int(row.get("scheduled_tokens", 0) or 0) for row in matched_rows),
                    "output_tokens": sum(int(row.get("output_tokens", 0) or 0) for row in matched_rows),
                    "prompt_tokens": max(int(row.get("prompt_tokens", 0) or 0) for row in matched_rows),
                    "embed_multimodal_ms": sum(float(row.get("embed_multimodal_ms", 0.0) or 0.0) for row in matched_rows),
                    "forward_ms": sum(float(row.get("forward_ms", 0.0) or 0.0) for row in matched_rows),
                    "encoder_cache_hit": sum(int(row.get("encoder_cache_hit", 0) or 0) for row in matched_rows),
                    "encoder_cache_miss": sum(int(row.get("encoder_cache_miss", 0) or 0) for row in matched_rows),
                    "encoder_cache_skipped": sum(int(row.get("encoder_cache_skipped", 0) or 0) for row in matched_rows),
                    "encoder_not_needed_this_step": sum(
                        int(row.get("encoder_not_needed_this_step", 0) or 0) for row in matched_rows
                    ),
                    "embed_multimodal_ms_list": _flatten_profile_list(matched_rows, "embed_multimodal_ms_list"),
                    "forward_ms_list": _flatten_profile_list(matched_rows, "forward_ms_list"),
                    "encoder_cache_hit_list": _flatten_profile_int_list(matched_rows, "encoder_cache_hit_list"),
                    "encoder_cache_miss_list": _flatten_profile_int_list(matched_rows, "encoder_cache_miss_list"),
                    "encoder_cache_skipped_list": _flatten_profile_int_list(matched_rows, "encoder_cache_skipped_list"),
                    "encoder_not_needed_this_step_list": _flatten_profile_int_list(
                        matched_rows, "encoder_not_needed_this_step_list"
                    ),
                    "start": min(float(row.get("start", 0.0) or 0.0) for row in matched_rows),
                    "now": max(float(row.get("now", 0.0) or 0.0) for row in matched_rows),
                }
            result.update(run)
            result["logged_profile_rows"] = len(matched_rows)
            result["logged_prompt_tokens_sum"] = sum(int(row.get("prompt_tokens", 0) or 0) for row in matched_rows)
            results.append(result)

        _write_results(results)
        print(f"Results JSON -> {RESULT_JSON}")
        print(f"Results CSV  -> {RESULT_CSV}")
    finally:
        Path(stage0_yaml_path).unlink(missing_ok=True)
        _cleanup_runtime()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
