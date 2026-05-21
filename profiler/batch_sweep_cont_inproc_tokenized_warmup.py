"""Batch sweep with in-process AsyncOmni requests using tokenized Alpamayo inputs."""

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

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from transformers import AutoConfig
from vllm import SamplingParams

OMNI = Path(__file__).resolve().parents[1]
VERL_ROOT = OMNI.parent / "verl"
MY_EXAMPLE2_ROOT = VERL_ROOT / "my_example2"

for p in [
    str(OMNI),
    str(OMNI.parent / "alpamayo1.5" / "src"),
    str(OMNI.parent / "verl-liming" / "my_example" / "alpamayo" / "src"),
    str(VERL_ROOT),
    str(MY_EXAMPLE2_ROOT),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from my_example2.src.inputs.alpamayo_batch import (  # noqa: E402
    build_alpamayo_sample,
    build_omni_additional_information,
    build_rollout_preprocess_fn,
    build_rollout_processor,
    build_traj_fuser,
    sample_to_vllm_prompt,
    validate_local_pai_dir,
)
from my_example2.src.models.register_vla_models import register_vla_models  # noqa: E402
from profiler.run_rollout_timing import _to_jsonable  # noqa: E402
from profiler.timestamps import render_request_timeline as request_timeline  # noqa: E402
from vllm_omni.entrypoints.async_omni import AsyncOmni  # noqa: E402
from vllm_omni.inputs.data import OmniDiffusionSamplingParams  # noqa: E402
from vllm_omni.model_executor.stage_input_processors.alpamayo1_5 import (  # noqa: E402
    build_alpamayo_stage0_tokenizer,
)

DEFAULT_MODEL_PATH = "/share/models/Alpamayo-1.5-10B"
DEFAULT_PAI_LOCAL_DIR = "/share/datasets/Alpamayo_pai_av_big/"
DEFAULT_TRAIN_FILE = str((MY_EXAMPLE2_ROOT / "datasets" / "train.parquet").resolve())

MODEL = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
PAI_LOCAL_DIR = os.environ.get("PAI_LOCAL_DIR", DEFAULT_PAI_LOCAL_DIR)
TRAIN_FILE = os.environ.get("TRAIN_FILE", DEFAULT_TRAIN_FILE)
PAI_CHUNK_IDS_ENV = os.environ.get("PAI_CHUNK_IDS", "").strip()
T0_US = int(os.environ.get("T0_US", "5100000"))
N_UNIQUE = int(os.environ.get("N_UNIQUE", "64"))
N_TOTAL = int(os.environ.get("N_TOTAL", "64"))
BS_LIST = [int(v) for v in os.environ.get("BS_LIST", "8").split(",") if v.strip()]
MAX_REQ_PER_GROUP = int(os.environ.get("MAX_REQ_PER_GROUP", "24"))
CHUNK_SAMPLES = int(os.environ.get("CHUNK_SAMPLES", "16"))
SVC_YAML = OMNI / "profiler" / "alpamayo1_5_gpu0.yaml"
PROFILE_GID_ENV = os.environ.get("PROFILE_GID", "").strip()
PROFILE_STAGES_ENV = os.environ.get("PROFILE_STAGES", "0,1").strip()
GPU_RECOVERY_POLL_S = float(os.environ.get("GPU_RECOVERY_POLL_S", "2.0"))
GPU_RECOVERY_TIMEOUT_S = float(os.environ.get("GPU_RECOVERY_TIMEOUT_S", "180.0"))
GPU_RECOVERY_MARGIN_GB = float(os.environ.get("GPU_RECOVERY_MARGIN_GB", "1.0"))
GENERATE_TRAJ_TOKENS_ENV = os.environ.get("GENERATE_TRAJ_TOKENS", "0").strip()
SKIP_WARMUP_SAMPLE_ENV = os.environ.get("SKIP_WARMUP_SAMPLE", "0").strip()

LOG_DIR = OMNI / "profiler" / "logs" / str(N_TOTAL) / f"async_omni_inproc_tokenized_warmup_trace-gid{PROFILE_GID_ENV}_{int(time.time())}"
ASYNC_OMNI_LOG_DIR = LOG_DIR / "svc_logs"
OUT = LOG_DIR / "metrics" / "batch_results.md"
PROFILE_DIR = LOG_DIR / "torch_traces"
RUNTIME_SVC_YAML = LOG_DIR / SVC_YAML.name
TOKENIZED_STAGE0_TRAJ_START_TOKEN_ID = 155681
TOKENIZED_STAGE0_TRAJ_END_TOKEN_ID = 155683

KV_TIMING_RE = re.compile(
    r"KV transfer timing: req=(?P<req>\S+) "
    r"extract_only_ms=(?P<extract_only_ms>\d+(?:\.\d+)?) "
    r"transfer_only_ms=(?P<transfer_only_ms>\d+(?:\.\d+)?) "
    r"extract_plus_transfer_ms=(?P<extract_plus_transfer_ms>\d+(?:\.\d+)?)"
)
RID_SUFFIX_RE = re.compile(r"(async-inproc-bs\d+-\S+)$")
WARMUP_REQUEST_ID_MARKER = "-warmup-"


def _output_prefix(default_path: Path) -> Path:
    return default_path.with_suffix("")


def _parse_profile_gid():
    if not PROFILE_GID_ENV:
        return None
    gids = set()
    for gid in PROFILE_GID_ENV.split(","):
        gid = gid.strip()
        if not gid:
            continue
        if not gid.isdigit() or int(gid) < 0:
            raise ValueError(f"Invalid PROFILE_GID value: {gid!r}")
        gids.add(int(gid))
    return gids


def _parse_profile_stages():
    if not PROFILE_STAGES_ENV:
        return None
    stages = []
    for raw_stage in PROFILE_STAGES_ENV.split(","):
        stage = raw_stage.strip()
        if not stage:
            continue
        stages.append(int(stage))
    return stages or None


def _parse_chunk_ids():
    if not PAI_CHUNK_IDS_ENV:
        return None
    return [chunk_id.strip() for chunk_id in PAI_CHUNK_IDS_ENV.split(",") if chunk_id.strip()]


def _parse_bool_flag(raw_value: str, *, name: str, default: bool) -> bool:
    if not raw_value:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid {name} value: {raw_value!r}")


def _build_dataset_config() -> Any:
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": MODEL,
                }
            },
            "data": {
                "train_files": TRAIN_FILE,
                "pai_local_dir": PAI_LOCAL_DIR,
                "pai_chunk_ids": _parse_chunk_ids(),
            },
        }
    )


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def _cfg_select(config: Any, path: str, default: Any = None) -> Any:
    cur = config
    for part in path.split("."):
        cur = _cfg_get(cur, part, None)
        if cur is None:
            return default
    return cur


def _build_local_pai_interface(local_dir: str, chunk_ids: Any | None = None) -> Any:
    from alpamayo_r1.data.pai_utils import PhysicalAIAVDatasetLocalInterface

    return PhysicalAIAVDatasetLocalInterface(local_dir=local_dir, chunk_ids=chunk_ids)


def _build_runtime_stage_config(
    source_stage_config_path: Path = SVC_YAML,
    runtime_stage_config_path: Path = RUNTIME_SVC_YAML,
    profile_dir: Path = PROFILE_DIR,
    apply_tokenized_stage0_overrides: bool = True,
    enable_prefix_caching: bool = True,
) -> Path:
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    runtime_stage_config_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_config = yaml.safe_load(source_stage_config_path.read_text(encoding="utf-8"))
    stage_args = yaml_config.get("stage_args") or []
    replacements = 0

    for stage_config in stage_args:
        if stage_config.get("stage_id") == 0:
            stage_config.pop("request_postprocess_func", None)
            stage_config.pop("prompt_rewrite_func", None)
            engine_args = stage_config.setdefault("engine_args", {})
            engine_args["enable_prefix_caching"] = enable_prefix_caching
            if apply_tokenized_stage0_overrides:
                _apply_tokenized_stage0_overrides(stage_config)

        engine_args = stage_config.get("engine_args") or {}
        profiler_config = engine_args.get("profiler_config") or {}
        if "torch_profiler_dir" not in profiler_config:
            continue
        profiler_config["torch_profiler_dir"] = str(profile_dir)
        replacements += 1

    if replacements == 0:
        raise RuntimeError(
            f"No torch_profiler_dir entries found in stage_args engine_args profiler_config of "
            f"{source_stage_config_path}"
        )

    runtime_stage_config_path.write_text(
        yaml.safe_dump(yaml_config, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"  Wrote runtime stage config {runtime_stage_config_path} "
        f"(torch_profiler_dir -> {profile_dir}, "
        f"tokenized_stage0_overrides={'on' if apply_tokenized_stage0_overrides else 'off'})"
    )
    return runtime_stage_config_path


def _apply_tokenized_stage0_overrides(stage_config: dict[str, Any]) -> None:
    engine_args = stage_config.setdefault("engine_args", {})
    engine_args.pop("logits_processors", None)

    omni_kv_config = engine_args.setdefault("omni_kv_config", {})
    kv_transfer_criteria = omni_kv_config.setdefault("kv_transfer_criteria", {})
    kv_transfer_criteria["type"] = "next_step_after_special_token"
    kv_transfer_criteria["token_id"] = TOKENIZED_STAGE0_TRAJ_START_TOKEN_ID

    default_sampling_params = stage_config.setdefault("default_sampling_params", {})
    default_sampling_params["stop_token_ids"] = [TOKENIZED_STAGE0_TRAJ_END_TOKEN_ID]


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
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PROFILER_DIR": str(PROFILE_DIR),
    })
    try:
        yield
    finally:
        for env_name, env_value in previous_env.items():
            if env_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = env_value


def _start_omni(bs_val, log_stat=True) -> AsyncOmni:
    kwargs: dict[str, Any] = {
        "stage_configs_path": str(RUNTIME_SVC_YAML),
    }
    if log_stat:
        output_prefix = _output_prefix(OUT)
        log_stat_filepath = output_prefix.with_name(f"{output_prefix.name}_engine_metrics_bs_{bs_val}")
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


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone().contiguous()
    if isinstance(value, list):
        cloned = []
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


def _materialize_prompt(sample):
    prompt: dict[str, Any] = {
        "prompt_token_ids": list(sample["prompt_token_ids"]),
        "additional_information": _clone_value(sample["additional_information"]),
    }
    if sample.get("postprocess_prompt_token_ids") is not None:
        prompt["postprocess_prompt_token_ids"] = list(sample["postprocess_prompt_token_ids"])
    if sample.get("multi_modal_data") is not None:
        prompt["multi_modal_data"] = _clone_value(sample["multi_modal_data"])
    if sample.get("mm_processor_kwargs") is not None:
        prompt["mm_processor_kwargs"] = _clone_value(sample["mm_processor_kwargs"])
    if sample.get("hf_processor_mm_kwargs") is not None:
        prompt["hf_processor_mm_kwargs"] = _clone_value(sample["hf_processor_mm_kwargs"])
    return prompt


def _load_sampling_params_from_yaml(tokenizer) -> tuple[SamplingParams, OmniDiffusionSamplingParams]:
    config = yaml.safe_load(RUNTIME_SVC_YAML.read_text(encoding="utf-8"))
    stage_args = config.get("stage_args") or []
    if len(stage_args) < 2:
        raise RuntimeError(f"unexpected stage config layout in {RUNTIME_SVC_YAML}")

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


async def _run_request(omni, prompt, request_id, sampling_params_list):
    final_output = None
    async for out in omni.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=sampling_params_list,
    ):
        final_output = out
    if final_output is None:
        raise RuntimeError("no final output")
    return final_output


def _build_result_from_output(output, cid, rid, lat):
    m = dict(getattr(output, "metrics", {}) or {})
    co = dict(m.get("custom_output", {}) or {})
    co.update(dict(getattr(output, "custom_output", {}) or {}))

    inf = float(m.get("inference_only_ms", 0) or 0)
    llm_ms = float(m.get("stage0_llm_ms", 0) or 0)
    kv_s0_extract_ms = float(co.get("kv_tran_s0_ms", co.get("kv_s0_extract_ms", 0)) or 0)
    kv_s1_receive_ms = float(co.get("kv_tran_s1_receive_ms", co.get("kv_s1_receive_ms", 0)) or 0)
    kv_s1_tran_ms = float(co.get("kv_s1_tran_ms", 0) or 0)
    kv_s1_prep_ms = float(co.get("kv_tran_s1_prep_ms", co.get("kv_s1_prep_ms", 0)) or 0)
    s1_diffusion_ms = float(co.get("df_ms", co.get("s1_diffusion_ms", 0)) or 0)
    kv_s0_extract_plus_transfer_ms = kv_s0_extract_ms
    kv_tran_total = kv_s0_extract_plus_transfer_ms + kv_s1_receive_ms + kv_s1_tran_ms + kv_s1_prep_ms
    cot_token_ids = co.get("cot_token_ids")
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
        "lat": lat,
        "inf": inf,
        "net": lat - inf,
        "s0_llm_ms": llm_ms,
        "client_encode_ms": 0.0,
        "client_http_ms": 0.0,
        "client_decode_ms": 0.0,
        "client_codec_ms": 0.0,
        "openai_handler_total_ms": 0.0,
        "openai_pre_full_generator_ms": 0.0,
        "openai_result_wait_ms": 0.0,
        "openai_postprocess_ms": 0.0,
        "openai_response_build_ms": 0.0,
        "openai_response_logging_ms": 0.0,
        "openai_check_model_ms": 0.0,
        "openai_prepare_runtime_ms": 0.0,
        "openai_preprocess_chat_ms": 0.0,
        "openai_image_prompt_rewrite_ms": 0.0,
        "openai_schedule_generator_ms": 0.0,
        "openai_preprocess_merge_kwargs_ms": 0.0,
        "openai_preprocess_build_params_ms": 0.0,
        "openai_preprocess_audio_injection_ms": 0.0,
        "openai_preprocess_render_chat_ms": 0.0,
        "openai_preprocess_get_tokenizer_ms": 0.0,
        "openai_preprocess_tool_adjust_ms": 0.0,
        "openai_preprocess_image_cleanup_ms": 0.0,
        "openai_preprocess_finalize_prompt_ms": 0.0,
        "api_server_pre_route_ms": 0.0,
        "api_server_request_body_ms": 0.0,
        "api_server_request_json_ms": 0.0,
        "api_server_pre_route_other_ms": 0.0,
        "api_server_request_body_bytes": 0.0,
        "api_server_endpoint_setup_ms": 0.0,
        "api_server_handler_call_ms": 0.0,
        "api_server_post_handler_ms": 0.0,
        "api_server_response_dump_ms": 0.0,
        "api_server_response_render_ms": 0.0,
        "api_server_route_total_ms": 0.0,
        "kv_tran_total": kv_tran_total,
        "kv_s0_extract_ms": kv_s0_extract_ms,
        "kv_s0_transfer_only_ms": 0.0,
        "kv_s0_extract_plus_transfer_ms": kv_s0_extract_plus_transfer_ms,
        "kv_s1_receive_ms": kv_s1_receive_ms,
        "kv_s1_tran_ms": kv_s1_tran_ms,
        "kv_s1_prep_ms": kv_s1_prep_ms,
        "s1_diffusion_ms": s1_diffusion_ms,
        "it": int(m.get("input_tokens", 0) or 0),
        "ot": output_tokens,
    }


def _load_kv_metrics_from_log(bs):
    log_path = ASYNC_OMNI_LOG_DIR / f"svc_bs{bs}.log"
    if not log_path.exists():
        return {}

    kv_metrics_by_rid = {}
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


def _build_group_agg(results, wall_ms):
    ok_results = [r for r in results if r.get("ok")]
    n_ok = len(ok_results)
    agg = {"wall_ms": wall_ms, "ok": n_ok}
    metric_keys = [
        "inf", "net",
        "s0_llm_ms",
        "kv_tran_total",
        "kv_s0_extract_ms", "kv_s0_transfer_only_ms", "kv_s0_extract_plus_transfer_ms",
        "kv_s1_receive_ms", "kv_s1_tran_ms", "kv_s1_prep_ms",
        "s1_diffusion_ms",
        "it", "ot",
        "client_encode_ms", "client_http_ms", "client_decode_ms", "client_codec_ms",
        "openai_handler_total_ms", "openai_pre_full_generator_ms", "openai_result_wait_ms",
        "openai_postprocess_ms", "openai_response_build_ms", "openai_response_logging_ms",
        "openai_check_model_ms", "openai_prepare_runtime_ms", "openai_preprocess_chat_ms",
        "openai_image_prompt_rewrite_ms", "openai_schedule_generator_ms",
        "openai_preprocess_merge_kwargs_ms", "openai_preprocess_build_params_ms",
        "openai_preprocess_audio_injection_ms", "openai_preprocess_render_chat_ms",
        "openai_preprocess_get_tokenizer_ms", "openai_preprocess_tool_adjust_ms",
        "openai_preprocess_image_cleanup_ms", "openai_preprocess_finalize_prompt_ms",
        "api_server_pre_route_ms", "api_server_handler_call_ms",
        "api_server_request_body_ms", "api_server_request_json_ms",
        "api_server_pre_route_other_ms", "api_server_request_body_bytes",
        "api_server_endpoint_setup_ms", "api_server_post_handler_ms",
        "api_server_response_dump_ms", "api_server_response_render_ms",
        "api_server_route_total_ms",
    ]
    if n_ok:
        for key in metric_keys:
            vals = [r[key] for r in ok_results]
            agg[f"avg_{key}"] = np.mean(vals)
            agg[f"min_{key}"] = np.min(vals)
            agg[f"max_{key}"] = np.max(vals)
        agg["bs_it"] = sum(r["it"] for r in ok_results)
        agg["bs_ot"] = sum(r["ot"] for r in ok_results)
        agg["bs_at"] = agg["bs_it"] + agg["bs_ot"]
    else:
        for key in metric_keys:
            agg[f"avg_{key}"] = agg[f"min_{key}"] = agg[f"max_{key}"] = 0.0
        agg["bs_it"] = agg["bs_ot"] = agg["bs_at"] = 0
    return agg


def _backfill_kv_metrics_from_log(bs, bs_data):
    kv_metrics_by_rid = _load_kv_metrics_from_log(bs)
    for grp in bs_data:
        for req in grp.get("reqs", []):
            if not req.get("ok"):
                continue
            log_metrics = kv_metrics_by_rid.get(req["rid"])
            if log_metrics:
                req["kv_s0_extract_ms"] = log_metrics["extract_only_ms"]
                req["kv_s0_transfer_only_ms"] = log_metrics["transfer_only_ms"]
                req["kv_s0_extract_plus_transfer_ms"] = log_metrics["extract_plus_transfer_ms"]
            req["kv_tran_total"] = (
                req["kv_s0_extract_plus_transfer_ms"]
                + req["kv_s1_receive_ms"]
                + req["kv_s1_tran_ms"]
                + req["kv_s1_prep_ms"]
            )
            req["net"] = req["lat"] - req["inf"]
        grp["agg"] = _build_group_agg(grp.get("reqs", []), grp["agg"]["wall_ms"])


def _is_warmup_request_id(request_id: str) -> bool:
    return WARMUP_REQUEST_ID_MARKER in request_id


def _render_timeline_html_for_log(log_path: Path, bs: int) -> Path | None:
    if not log_path.exists():
        print(f"  timeline skip: missing log {log_path}")
        return None

    try:
        order, rows, profiles = request_timeline.parse_log(log_path)
        filtered_order = [req_id for req_id in order if not _is_warmup_request_id(req_id)]
        filtered_rows = {req_id: rows[req_id] for req_id in filtered_order}
        filtered_profiles = {req_id: profiles.get(req_id, {}) for req_id in filtered_order}
        batch_size = request_timeline.infer_batch_size(log_path)
        if batch_size is None:
            batch_size = request_timeline.infer_batch_size_from_requests(filtered_order)
        if batch_size is None:
            batch_size = bs
        payload = request_timeline.build_requests_payload(filtered_order, filtered_rows, batch_size, filtered_profiles)
        if not payload["requests"]:
            print(f"  timeline skip: no request metrics found in {log_path.name}")
            return None
        output = log_path.with_suffix(log_path.suffix + ".timeline.html")
        output.write_text(request_timeline.build_html(payload, log_path), encoding="utf-8")
        print(f"  timeline -> {output}")
        return output
    except Exception as exc:
        print(f"  timeline render failed for {log_path.name}: {exc!r}")
        return None


async def one_async(omni, sampling_params_list, sample, idx, bs, sem):
    rid = f"async-inproc-bs{bs}-{sample['cid'][:8]}-{idx}"
    st = time.time()
    async with sem:
        try:
            prompt = _materialize_prompt(sample)
            final_output = await _run_request(omni, prompt, rid, list(sampling_params_list))
            lat = (time.time() - st) * 1e3
            return _build_result_from_output(final_output, sample["cid"], rid, lat)
        except Exception as e:
            err_str = repr(e)[:200]
            print(f"  ERR rid={rid} {err_str}")
            return {"ok": False, "c": sample["cid"], "rid": rid, "lat": (time.time() - st) * 1e3, "err": err_str}


async def run_group_async(omni, sampling_params_list, grp, gi, bs):
    workers = min(len(grp), 8 if bs <= 8 else 4)
    sem = asyncio.Semaphore(workers)
    t0 = time.time()
    tasks = [
        asyncio.create_task(one_async(omni, sampling_params_list, sample, gi * bs + ii, bs, sem))
        for ii, sample in enumerate(grp)
    ]
    results = await asyncio.gather(*tasks)
    wall_ms = (time.time() - t0) * 1e3
    return wall_ms, results, _build_group_agg(results, wall_ms)


async def run_warmup_once(omni, sampling_params_list, sample, bs):
    warmup_rid = f"async-inproc-bs{bs}{WARMUP_REQUEST_ID_MARKER}{sample['cid'][:8]}-init"
    prompt = _materialize_prompt(sample)
    await _run_request(omni, prompt, warmup_rid, list(sampling_params_list))


def gen(all_batches, clip_stats):
    lines = [
        "# Batch Sweep",
        "",
        f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | {N_TOTAL} samples | GPU0 | tokenized input",
        f"- build_input mean={clip_stats['build']:.0f}ms",
        "",
        "## 表1: 逐请求详细指标",
        "",
        "| bs | gid | rid | lat_ms | net_ms | inf_ms | s0_llm_ms | kv_tran_total | s1_diffusion_ms | kv_s0_extract_plus_transfer_ms | kv_s1_receive_ms | itok | otok |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for bs in BS_LIST:
        for grp in all_batches.get(bs, []):
            for req in grp.get("reqs", []):
                if not req.get("ok"):
                    continue
                lines.append(
                    f"| {bs} | {grp['gid']} | {req['rid']} | {req['lat']:.0f} | {req['net']:.0f}"
                    f" | {req['inf']:.0f} | {req['s0_llm_ms']:.0f} | {req['kv_tran_total']:.0f}"
                    f" | {req['s1_diffusion_ms']:.0f} | {req['kv_s0_extract_plus_transfer_ms']:.0f}"
                    f" | {req['kv_s1_receive_ms']:.0f} | {req['it']} | {req['ot']} |"
                )

    lines += [
        "",
        "## 表2: 逐 BS 请求级指标统计",
        "",
        "| bs | metric | min | max | mean |",
        "|---:|---:|---:|---:|---:|",
    ]

    metric_names = [
        ("lat_ms", "lat"),
        ("net_ms", "net"),
        ("inf_ms", "inf"),
        ("s0_llm_ms", "s0_llm_ms"),
        ("kv_tran_total_ms", "kv_tran_total"),
        ("kv_s0_extract_ms", "kv_s0_extract_ms"),
        ("kv_s0_transfer_only_ms", "kv_s0_transfer_only_ms"),
        ("kv_s0_extract_plus_transfer_ms", "kv_s0_extract_plus_transfer_ms"),
        ("kv_s1_receive_ms", "kv_s1_receive_ms"),
        ("kv_s1_tran_ms", "kv_s1_tran_ms"),
        ("kv_s1_prep_ms", "kv_s1_prep_ms"),
        ("s1_diffusion_ms", "s1_diffusion_ms"),
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
            lines.append(f"| {bs} | {label} | {np.min(vals):.0f} | {np.max(vals):.0f} | {np.mean(vals):.0f} |")

    lines += [
        "",
        "## 表3: 每 BS 吞吐量",
        "",
        "| bs | n_grp | ok | E2E_s | batch_it/s | batch_ot/s | batch_at/s | avg_sample_ms | samples/s | 相比BS=1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    base_ss = None
    for bs in BS_LIST:
        grps = [g for g in all_batches.get(bs, []) if g["agg"]["ok"] > 0]
        if not grps:
            continue
        ok_total = sum(g["agg"]["ok"] for g in grps)
        e2e = sum(g["agg"]["wall_ms"] for g in grps) / 1e3
        total_it = sum(g["agg"]["bs_it"] for g in grps)
        total_ot = sum(g["agg"]["bs_ot"] for g in grps)
        total_at = sum(g["agg"]["bs_at"] for g in grps)
        total_req = sum(len(g.get("reqs", [])) for g in grps)
        avg_sample_ms = e2e * 1e3 / total_req if total_req > 0 else 0
        samples_per_s = total_req / e2e if e2e > 0 else 0
        if base_ss is None:
            base_ss = samples_per_s
        speedup = samples_per_s / base_ss if base_ss > 0 else 0
        lines.append(
            f"| {bs} | {len(grps)} | {ok_total} | {e2e:.1f}"
            f" | {total_it / e2e:.0f} | {total_ot / e2e:.0f} | {total_at / e2e:.0f}"
            f" | {avg_sample_ms:.0f} | {samples_per_s:.3f} | {speedup:.2f}x |"
        )

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> {OUT}")


def _cleanup_runtime():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _get_gpu_mem_info():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.mem_get_info()
    except Exception:
        return None


async def _wait_for_gpu_recovery(*, expected_free_bytes, bs):
    if expected_free_bytes is None:
        await asyncio.sleep(8)
        return

    margin_bytes = int(GPU_RECOVERY_MARGIN_GB * (1024 ** 3))
    min_free_bytes = max(0, expected_free_bytes - margin_bytes)
    deadline = time.monotonic() + GPU_RECOVERY_TIMEOUT_S
    last_reported_gib = None

    while True:
        _cleanup_runtime()
        mem_info = _get_gpu_mem_info()
        if mem_info is not None:
            free_bytes, total_bytes = mem_info
            if free_bytes >= min_free_bytes:
                print(
                    f"BATCH={bs} GPU recovered: free={free_bytes / (1024 ** 3):.2f}GiB "
                    f"target>={min_free_bytes / (1024 ** 3):.2f}GiB "
                    f"total={total_bytes / (1024 ** 3):.2f}GiB"
                )
                return

            free_gib = free_bytes / (1024 ** 3)
            if last_reported_gib is None or abs(free_gib - last_reported_gib) >= 0.25:
                print(
                    f"BATCH={bs} waiting for GPU recovery: free={free_gib:.2f}GiB "
                    f"target>={min_free_bytes / (1024 ** 3):.2f}GiB"
                )
                last_reported_gib = free_gib

        if time.monotonic() >= deadline:
            if mem_info is None:
                print(f"BATCH={bs} GPU recovery check unavailable, continue after timeout.")
            else:
                free_bytes, _ = mem_info
                print(
                    f"BATCH={bs} GPU recovery timeout: free={free_bytes / (1024 ** 3):.2f}GiB "
                    f"target>={min_free_bytes / (1024 ** 3):.2f}GiB, continue anyway."
                )
            return

        await asyncio.sleep(GPU_RECOVERY_POLL_S)


def _build_model_input_components() -> dict[str, Any]:
    register_vla_models()
    # Importing the original Alpamayo model module registers
    # model_type="alpamayo1_5" with Hugging Face AutoConfig.
    import alpamayo1_5.models.alpamayo1_5  # noqa: F401

    config = _build_dataset_config()
    model_path = (
        _cfg_select(config, "actor_rollout_ref.model.path")
        or _cfg_select(config, "model.path")
        or _cfg_get(config, "model_path")
        or _cfg_get(config, "path")
    )
    if model_path is None:
        raise ValueError("model path is required to build tokenized Alpamayo inputs.")

    validate_local_pai_dir(PAI_LOCAL_DIR)
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = build_rollout_processor(model_config)
    preprocess_fn = build_rollout_preprocess_fn(model_config)
    traj_fuser = build_traj_fuser(model_config)
    avdi = _build_local_pai_interface(PAI_LOCAL_DIR, _parse_chunk_ids())
    return {
        "config": config,
        "model_config": model_config,
        "processor": processor,
        "preprocess_fn": preprocess_fn,
        "traj_fuser": traj_fuser,
        "avdi": avdi,
    }


def _build_model_input(
    *,
    clip_id: str,
    t0_us: int,
    preprocess_fn: Any,
    processor: Any,
    model_config: Any,
    traj_fuser: Any,
    avdi: Any,
) -> dict[str, Any]:
    sample = build_alpamayo_sample(clip_id, t0_us, preprocess_fn, avdi)
    prompt = sample_to_vllm_prompt(sample, processor.tokenizer, model_config, traj_fuser)
    if "additional_information" not in prompt:
        sample_with_cfg = dict(sample)
        sample_with_cfg["model_config"] = model_config
        postprocess_prompt_ids = list(prompt.get("postprocess_prompt_token_ids") or prompt.get("prompt_token_ids") or [])
        prompt["additional_information"] = build_omni_additional_information(
            sample_with_cfg,
            postprocess_prompt_ids=postprocess_prompt_ids,
        )
    return prompt


async def main_async():
    print(f"Batch sweep: {N_TOTAL} samples ({N_UNIQUE} unique clips cycled), BS={BS_LIST}")
    skip_warmup_sample = _parse_bool_flag(
        SKIP_WARMUP_SAMPLE_ENV,
        name="SKIP_WARMUP_SAMPLE",
        default=False,
    )
    print(
        "Warmup once after engine start: "
        f"sampling_params=formal_request_params, "
        f"skip_warmup_sample={skip_warmup_sample}"
    )
    apply_tokenized_stage0_overrides = _parse_bool_flag(
        GENERATE_TRAJ_TOKENS_ENV,
        name="GENERATE_TRAJ_TOKENS",
        default=True,
    )
    _build_runtime_stage_config(
        apply_tokenized_stage0_overrides=apply_tokenized_stage0_overrides,
    )
    tokenizer = build_alpamayo_stage0_tokenizer(MODEL)
    sampling_params_list = _load_sampling_params_from_yaml(tokenizer)

    print("Loading dataset + preparing tokenized inputs...")
    components = _build_model_input_components()
    avdi = components["avdi"]
    ci = avdi.clip_index
    print(len(list(ci[ci.chunk == 3116].index)))
    unique_cids = list(ci[ci.chunk == 3116].index)[:N_UNIQUE]

    samples_unique = []
    build_ms_list = []
    for i, cid in enumerate(unique_cids):
        t_build = time.perf_counter()
        model_input = _build_model_input(
            clip_id=str(cid),
            t0_us=T0_US,
            preprocess_fn=components["preprocess_fn"],
            processor=components["processor"],
            model_config=components["model_config"],
            traj_fuser=components["traj_fuser"],
            avdi=components["avdi"],
        )
        build_ms = (time.perf_counter() - t_build) * 1000.0
        samples_unique.append(
            {
                "cid": str(cid),
                "prompt_token_ids": list(model_input.get("prompt_token_ids") or []),
                "postprocess_prompt_token_ids": list(
                    model_input.get("postprocess_prompt_token_ids") or model_input.get("prompt_token_ids") or []
                ),
                "multi_modal_data": _clone_value(model_input.get("multi_modal_data") or {}),
                "mm_processor_kwargs": _clone_value(model_input.get("mm_processor_kwargs")),
                "hf_processor_mm_kwargs": _clone_value(model_input.get("hf_processor_mm_kwargs")),
                "additional_information": _clone_value(model_input.get("additional_information") or {}),
            }
        )
        build_ms_list.append(build_ms)
        print(
            f"  {i + 1}/{N_UNIQUE} clip={str(cid)[:8]} "
            f"build={build_ms:.0f}ms prompt_tokens={len(samples_unique[-1]['prompt_token_ids'])}"
        )

    cs = {"build": float(np.mean(build_ms_list)) if build_ms_list else 0.0}
    print(f"Preload done. build_mean={cs['build']:.0f}ms\n")

    warmup_sample = samples_unique[0] if samples_unique else None
    samples = (samples_unique * ((N_TOTAL // max(1, N_UNIQUE)) + 1))[:N_TOTAL]
    if skip_warmup_sample and warmup_sample is not None and samples:
        samples = samples[1:]
        print(
            f"Skipping warmup sample from formal requests: clip={warmup_sample['cid'][:8]}, "
            f"remaining_samples={len(samples)}"
        )

    all_batches = {}
    profile_gids = _parse_profile_gid()
    profile_stages = _parse_profile_stages()

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
            n_groups = (len(samples) + bs - 1) // bs if samples else 0
            grp_per_chunk = max(1, CHUNK_SAMPLES // bs)
            print(f"BATCH={bs}  ({n_groups} groups x {bs}, chunk={grp_per_chunk} groups)")
            bs_data = []
            done_reqs = 0

            if warmup_sample is not None:
                print(f"BATCH={bs} warmup once with clip={warmup_sample['cid'][:8]}")
                with _redirect_process_output(log_path):
                    await run_warmup_once(omni, sampling_params_list, warmup_sample, bs)

            for chunk_start in range(0, n_groups, grp_per_chunk):
                chunk_end = min(chunk_start + grp_per_chunk, n_groups)
                for gi in range(chunk_start, chunk_end):
                    start = gi * bs
                    end = min(start + bs, N_TOTAL)
                    grp = samples[start:end]
                    if len(grp) > MAX_REQ_PER_GROUP:
                        raise ValueError(f"Group size {len(grp)} exceeds cap {MAX_REQ_PER_GROUP}")
                    should_profile_group = profile_gids and gi in profile_gids
                    if should_profile_group:
                        stages_text = PROFILE_STAGES_ENV
                        print(f"  profiling gid={gi} (g{gi+1}/{n_groups}) for bs={bs} (stages={stages_text})")
                        with _redirect_process_output(log_path):
                            await omni.start_profile(profile_prefix=f"bs{bs}_gid{gi}", stages=profile_stages)
                    try:
                        with _redirect_process_output(log_path):
                            wm, results, agg = await run_group_async(omni, sampling_params_list, grp, gi, bs)
                    finally:
                        if should_profile_group:
                            with _redirect_process_output(log_path):
                                await omni.stop_profile(stages=profile_stages)
                    bs_data.append({"gid": gi, "reqs": results, "agg": agg})
                    total_ok = sum(d["agg"]["ok"] for d in bs_data)
                    done_reqs += len(grp)
                    print(
                        f"  g{gi+1:4d}/{n_groups} ok={agg['ok']}/{len(grp)} "
                        f"wall={wm:.0f}ms [cum ok={total_ok}/{done_reqs}]"
                    )
                if chunk_end < n_groups:
                    print("  chunk done, sleep 8s...")
                    await asyncio.sleep(8)

            _backfill_kv_metrics_from_log(bs, bs_data)
            all_batches[bs] = bs_data
            total_ok = sum(d["agg"]["ok"] for d in bs_data)
            total_wall = sum(d["agg"]["wall_ms"] for d in bs_data)
            print(f"  DONE ok={total_ok}/{N_TOTAL} E2E={total_wall/1e3:.1f}s")

            output_prefix = _output_prefix(OUT)
            bs_output_path = output_prefix.with_name(f"{output_prefix.name}_bs_{bs}.json")
            bs_output_payload = _to_jsonable({"all_batches": all_batches, "cs": cs})
            bs_output_path.write_text(
                json.dumps(
                    bs_output_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            gen(all_batches, cs)
            _render_timeline_html_for_log(ASYNC_OMNI_LOG_DIR / f"svc_bs{bs}.log", bs)
        finally:
            with _redirect_process_output(log_path):
                omni.shutdown()
            await _wait_for_gpu_recovery(expected_free_bytes=startup_free_bytes, bs=bs)

    print(f"\nALL DONE => {OUT}")


if __name__ == "__main__":
    asyncio.run(main_async())
