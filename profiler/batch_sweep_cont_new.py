"""Batch sweep: 256 samples (22 unique clips cycled), per-batch-group rows."""
import asyncio, json, re, sys, time, numpy as np, subprocess, os
import urllib.request as urllib_request
from urllib.error import HTTPError
from pathlib import Path

try:
    import aiohttp
    print(f"Using aiohttp {aiohttp.__version__} for async HTTP")
except ImportError:
    aiohttp = None

try:
    import httpx
    if aiohttp is None:
        print(f"Using httpx {httpx.__version__} for async HTTP")
except ImportError:
    httpx = None

OMNI = Path(__file__).resolve().parents[1]
for p in [str(OMNI), str(OMNI.parent/"alpamayo1.5"/"src"), str(OMNI.parent/"verl-liming"/"my_example"/"alpamayo"/"src")]:
    sys.path.insert(0, p)

from profiler.run_rollout_timing import (
    _load_local_avdi, _load_clip_data, _build_prompt_messages,
    _adapt_messages_with_frames, _to_jsonable,
)

DEFAULT_MODEL_PATH = "/share/models/Alpamayo-1.5-10B"
MODEL    = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
HOST, PORT = "127.0.0.1", 8300
N_UNIQUE = 64 # 22
N_TOTAL  = 64 # 256
# BS_LIST  = [1, 2, 4, 8, 12, 16, 24]
BS_LIST  = [1, 2, 4, 8]
MAX_REQ_PER_GROUP = 24  # Request-side cap for each vLLM processing group
CHUNK_SAMPLES = 16  # max samples per chunk
SVC_YAML = OMNI / "profiler" / "alpamayo1_5_gpu0.yaml"
PROFILE_GID_ENV = os.environ.get("PROFILE_GID", "").strip()
PROFILE_STAGES_ENV = os.environ.get("PROFILE_STAGES", "0,1").strip()

LOG_DIR = OMNI / "profiler" / "logs" / str(N_TOTAL) / f"async_omni_trace-gid{PROFILE_GID_ENV}_{int(time.time())}"
ASYNC_OMNI_LOG_DIR = LOG_DIR / "svc_logs"
OUT = LOG_DIR / "metrics" / "batch_results.md"
PROFILE_DIR = LOG_DIR / "torch_traces"
RUNTIME_SVC_YAML = LOG_DIR / SVC_YAML.name

KV_TIMING_RE = re.compile(
    r"KV transfer timing: req=(?P<req>\S+) "
    r"extract_only_ms=(?P<extract_only_ms>\d+(?:\.\d+)?) "
    r"transfer_only_ms=(?P<transfer_only_ms>\d+(?:\.\d+)?) "
    r"extract_plus_transfer_ms=(?P<extract_plus_transfer_ms>\d+(?:\.\d+)?)"
)
RID_SUFFIX_RE = re.compile(r"(bs\d+-\S+)$")


def _output_prefix(default_path: Path) -> Path:
    return default_path.with_suffix("")


def _parse_profile_gid():
    """Parse PROFILE_GID env into a 0-based group id to profile."""
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


def _build_runtime_stage_config(
    source_stage_config_path: Path = SVC_YAML,
    runtime_stage_config_path: Path = RUNTIME_SVC_YAML,
    profile_dir: Path = PROFILE_DIR,
) -> Path:
    """Create a temporary stage config with torch_profiler_dir rewritten."""
    profile_dir = Path(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    runtime_stage_config_path.parent.mkdir(parents=True, exist_ok=True)

    config_text = source_stage_config_path.read_text(encoding="utf-8")
    updated_text, replacements = re.subn(
        r"^(\s*torch_profiler_dir:\s*).*$",
        rf"\1{profile_dir}",
        config_text,
        flags=re.MULTILINE,
    )
    if replacements == 0:
        raise RuntimeError(f"No torch_profiler_dir entries found in {source_stage_config_path}")
    runtime_stage_config_path.write_text(updated_text, encoding="utf-8")
    print(
        f"  Wrote runtime stage config {runtime_stage_config_path} "
        f"(torch_profiler_dir -> {profile_dir})"
    )
    return runtime_stage_config_path

def _restart_service(bs_val, log_stat=True):
    """Kill old vllm-omni service and start fresh for a given batch size."""
    import subprocess, os
    print(f"  Restarting service for BATCH={bs_val}...")
    subprocess.run("pkill -f 'vllm-omni serve' 2>/dev/null", shell=True)
    time.sleep(5)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    output_prefix = _output_prefix(OUT)
    log_stat_filepath = output_prefix.with_name(f"{output_prefix.name}_engine_metrics_bs_{bs_val}")
    log_stat_arg = f" --log-stat-filepath {str(log_stat_filepath)}" if log_stat else ""
    print(log_stat_arg)
    cmd = (f"vllm-omni serve {MODEL} --omni --host 127.0.0.1 --port {PORT} "
           f"--served-model-name alpamayo1.5 --stage-configs-path {RUNTIME_SVC_YAML}"
           f"{log_stat_arg}")
    log = ASYNC_OMNI_LOG_DIR / f"svc_bs{bs_val}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        subprocess.Popen(cmd, shell=True, env=env, stdout=f, stderr=f,
                         start_new_session=True)
    # Poll until healthy
    for _ in range(120):
        try:
            r = urllib_request.urlopen(f"http://{HOST}:{PORT}/health", timeout=2)
            if r.status == 200:
                print(f"  Service READY (log={log})")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("Service failed to start")

def _post(url, payload, timeout=60):
    """Short-timeout HTTP POST (skip rather than hang 300s)."""
    t_encode_start = time.perf_counter()
    data = json.dumps(_to_jsonable(payload)).encode("utf-8")
    encode_ms = (time.perf_counter() - t_encode_start) * 1000.0
    req = urllib_request.Request(url, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        t_http_start = time.perf_counter()
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        http_ms = (time.perf_counter() - t_http_start) * 1000.0
        t_decode_start = time.perf_counter()
        parsed = json.loads(raw.decode("utf-8"))
        decode_ms = (time.perf_counter() - t_decode_start) * 1000.0
        return parsed, {
            "client_encode_ms": encode_ms,
            "client_http_ms": http_ms,
            "client_decode_ms": decode_ms,
        }
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} when POST {url}\n{body[:300]}") from exc


def _parse_profile_stages():
    """Parse PROFILE_STAGES env into request payload value."""
    if not PROFILE_STAGES_ENV:
        return None
    stages = []
    for raw_stage in PROFILE_STAGES_ENV.split(","):
        stage = raw_stage.strip()
        if not stage:
            continue
        stages.append(int(stage))
    return stages or None


def _profile_request_body():
    stages = _parse_profile_stages()
    return {} if stages is None else {"stages": stages}


def _set_profiler_enabled(is_start: bool, timeout=60):
    """Call the service profiler API directly from the batch sweep script."""
    action = "start_profile" if is_start else "stop_profile"
    resp, _ = _post(
        f"http://{HOST}:{PORT}/{action}",
        _profile_request_body(),
        timeout=timeout,
    )
    return resp


async def _post_async(url, payload, timeout=60):
    """Async HTTP POST with graceful fallback."""
    t_encode_start = time.perf_counter()
    body = json.dumps(_to_jsonable(payload)).encode("utf-8")
    encode_ms = (time.perf_counter() - t_encode_start) * 1000.0
    headers = {"Content-Type": "application/json"}

    if httpx is not None:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                t_http_start = time.perf_counter()
                resp = await client.post(url, content=body, headers=headers)
                http_ms = (time.perf_counter() - t_http_start) * 1000.0
                resp.raise_for_status()
                t_decode_start = time.perf_counter()
                parsed = json.loads(resp.content.decode("utf-8"))
                decode_ms = (time.perf_counter() - t_decode_start) * 1000.0
                return parsed, {
                    "client_encode_ms": encode_ms,
                    "client_http_ms": http_ms,
                    "client_decode_ms": decode_ms,
                }
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            raise RuntimeError(f"HTTP {exc.response.status_code} when POST {url}\n{body[:300]}") from exc

    if aiohttp is not None:
        timeout_cfg = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
            t_http_start = time.perf_counter()
            async with session.post(url, data=body, headers=headers) as resp:
                raw = await resp.read()
            http_ms = (time.perf_counter() - t_http_start) * 1000.0
            if resp.status >= 400:
                raise RuntimeError(f"HTTP {resp.status} when POST {url}\n{raw.decode('utf-8', errors='replace')[:300]}")
            t_decode_start = time.perf_counter()
            parsed = json.loads(raw.decode("utf-8"))
            decode_ms = (time.perf_counter() - t_decode_start) * 1000.0
            return parsed, {
                "client_encode_ms": encode_ms,
                "client_http_ms": http_ms,
                "client_decode_ms": decode_ms,
            }

    return await asyncio.to_thread(_post, url, payload, timeout)


def _load_kv_metrics_from_log(bs):
    """Load all stage-0 KV timing metrics for one batch size from service log."""
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
    """Build aggregate stats for one group from per-request results."""
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
        for k in metric_keys:
            vals = [r[k] for r in ok_results]
            agg[f"avg_{k}"] = np.mean(vals)
            agg[f"min_{k}"] = np.min(vals)
            agg[f"max_{k}"] = np.max(vals)
        agg["bs_it"] = sum(r["it"] for r in ok_results)
        agg["bs_ot"] = sum(r["ot"] for r in ok_results)
        agg["bs_at"] = agg["bs_it"] + agg["bs_ot"]
    else:
        for k in metric_keys:
            agg[f"avg_{k}"] = agg[f"min_{k}"] = agg[f"max_{k}"] = 0.0
        agg["bs_it"] = agg["bs_ot"] = agg["bs_at"] = 0
    return agg


def _backfill_kv_metrics_from_log(bs, bs_data):
    """Backfill per-request KV timing metrics by reading the service log once."""
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
            # print(req["inf"], req["s0"], req["s1"], req["inf"] - req["s0"] - req["s1"])
        grp["agg"] = _build_group_agg(grp.get("reqs", []), grp["agg"]["wall_ms"])


def _build_result_from_response(resp, cid, rid, lat, client_timings=None):
    """Convert response payload into profiler metrics dict."""
    client_timings = client_timings or {}
    m = resp.get("metrics", {}) or {}
    co = m.get("custom_output", {}) or {}
    inf = float(m.get("inference_only_ms", 0) or 0)
    llm_ms = float(m.get("stage0_llm_ms", 0) or 0)
    kv_s0_start_time = float(co.get("kv_s0_start_time", 0) or 0)
    kv_s0_extract_ms = float(co.get("kv_s0_extract_ms", 0) or 0) # 只包含stage-0 extract kv的时间
    kv_s1_receive_ms = float(co.get("kv_s1_receive_ms", 0) or 0)
    kv_s1_tran_ms = float(co.get("kv_s1_tran_ms", 0) or 0)
    kv_s1_end_time = float(co.get("kv_s1_end_time", 0) or 0)
    kv_s1_prep_ms = float(co.get("kv_s1_prep_ms", 0) or 0)
    s1_diffusion_ms = float(co.get("s1_diffusion_ms", 0) or 0)
    client_encode_ms = float(client_timings.get("client_encode_ms", 0.0) or 0.0)
    client_http_ms = float(client_timings.get("client_http_ms", 0.0) or 0.0)
    client_decode_ms = float(client_timings.get("client_decode_ms", 0.0) or 0.0)
    openai_handler_total_ms = float(m.get("openai_handler_total_ms", 0.0) or 0.0)
    openai_pre_full_generator_ms = float(m.get("openai_pre_full_generator_ms", 0.0) or 0.0)
    openai_result_wait_ms = float(m.get("openai_result_wait_ms", 0.0) or 0.0)
    openai_postprocess_ms = float(m.get("openai_postprocess_ms", 0.0) or 0.0)
    openai_response_build_ms = float(m.get("openai_response_build_ms", 0.0) or 0.0)
    openai_response_logging_ms = float(m.get("openai_response_logging_ms", 0.0) or 0.0)
    openai_check_model_ms = float(m.get("openai_check_model_ms", 0.0) or 0.0)
    openai_prepare_runtime_ms = float(m.get("openai_prepare_runtime_ms", 0.0) or 0.0)
    openai_preprocess_chat_ms = float(m.get("openai_preprocess_chat_ms", 0.0) or 0.0)
    openai_image_prompt_rewrite_ms = float(m.get("openai_image_prompt_rewrite_ms", 0.0) or 0.0)
    openai_schedule_generator_ms = float(m.get("openai_schedule_generator_ms", 0.0) or 0.0)
    openai_preprocess_merge_kwargs_ms = float(m.get("openai_preprocess_merge_kwargs_ms", 0.0) or 0.0)
    openai_preprocess_build_params_ms = float(m.get("openai_preprocess_build_params_ms", 0.0) or 0.0)
    openai_preprocess_audio_injection_ms = float(m.get("openai_preprocess_audio_injection_ms", 0.0) or 0.0)
    openai_preprocess_render_chat_ms = float(m.get("openai_preprocess_render_chat_ms", 0.0) or 0.0)
    openai_preprocess_get_tokenizer_ms = float(m.get("openai_preprocess_get_tokenizer_ms", 0.0) or 0.0)
    openai_preprocess_tool_adjust_ms = float(m.get("openai_preprocess_tool_adjust_ms", 0.0) or 0.0)
    openai_preprocess_image_cleanup_ms = float(m.get("openai_preprocess_image_cleanup_ms", 0.0) or 0.0)
    openai_preprocess_finalize_prompt_ms = float(m.get("openai_preprocess_finalize_prompt_ms", 0.0) or 0.0)
    api_server_pre_route_ms = float(m.get("api_server_pre_route_ms", 0.0) or 0.0)
    api_server_request_body_ms = float(m.get("api_server_request_body_ms", 0.0) or 0.0)
    api_server_request_json_ms = float(m.get("api_server_request_json_ms", 0.0) or 0.0)
    api_server_pre_route_other_ms = float(m.get("api_server_pre_route_other_ms", 0.0) or 0.0)
    api_server_request_body_bytes = float(m.get("api_server_request_body_bytes", 0.0) or 0.0)
    api_server_endpoint_setup_ms = float(m.get("api_server_endpoint_setup_ms", 0.0) or 0.0)
    api_server_handler_call_ms = float(m.get("api_server_handler_call_ms", 0.0) or 0.0)
    api_server_post_handler_ms = float(m.get("api_server_post_handler_ms", 0.0) or 0.0)
    api_server_response_dump_ms = float(m.get("api_server_response_dump_ms", 0.0) or 0.0)
    api_server_response_render_ms = float(m.get("api_server_response_render_ms", 0.0) or 0.0)
    api_server_route_total_ms = float(m.get("api_server_route_total_ms", 0.0) or 0.0)
    return {"ok": True, "c": cid, "rid": rid, "lat": lat, "inf": inf,
            "s0_llm_ms": llm_ms, "net": lat-inf,
        "client_encode_ms": client_encode_ms,
        "client_http_ms": client_http_ms,
        "client_decode_ms": client_decode_ms,
        "client_codec_ms": client_encode_ms + client_decode_ms,
        "openai_handler_total_ms": openai_handler_total_ms,
        "openai_pre_full_generator_ms": openai_pre_full_generator_ms,
        "openai_result_wait_ms": openai_result_wait_ms,
        "openai_postprocess_ms": openai_postprocess_ms,
        "openai_response_build_ms": openai_response_build_ms,
        "openai_response_logging_ms": openai_response_logging_ms,
        "openai_check_model_ms": openai_check_model_ms,
        "openai_prepare_runtime_ms": openai_prepare_runtime_ms,
        "openai_preprocess_chat_ms": openai_preprocess_chat_ms,
        "openai_image_prompt_rewrite_ms": openai_image_prompt_rewrite_ms,
        "openai_schedule_generator_ms": openai_schedule_generator_ms,
        "openai_preprocess_merge_kwargs_ms": openai_preprocess_merge_kwargs_ms,
        "openai_preprocess_build_params_ms": openai_preprocess_build_params_ms,
        "openai_preprocess_audio_injection_ms": openai_preprocess_audio_injection_ms,
        "openai_preprocess_render_chat_ms": openai_preprocess_render_chat_ms,
        "openai_preprocess_get_tokenizer_ms": openai_preprocess_get_tokenizer_ms,
        "openai_preprocess_tool_adjust_ms": openai_preprocess_tool_adjust_ms,
        "openai_preprocess_image_cleanup_ms": openai_preprocess_image_cleanup_ms,
        "openai_preprocess_finalize_prompt_ms": openai_preprocess_finalize_prompt_ms,
        "api_server_pre_route_ms": api_server_pre_route_ms,
        "api_server_request_body_ms": api_server_request_body_ms,
        "api_server_request_json_ms": api_server_request_json_ms,
        "api_server_pre_route_other_ms": api_server_pre_route_other_ms,
        "api_server_request_body_bytes": api_server_request_body_bytes,
        "api_server_endpoint_setup_ms": api_server_endpoint_setup_ms,
        "api_server_handler_call_ms": api_server_handler_call_ms,
        "api_server_post_handler_ms": api_server_post_handler_ms,
        "api_server_response_dump_ms": api_server_response_dump_ms,
        "api_server_response_render_ms": api_server_response_render_ms,
        "api_server_route_total_ms": api_server_route_total_ms,
        "kv_tran_total": (kv_s1_end_time-kv_s0_start_time) * 1000 if kv_s0_start_time and kv_s1_end_time else 0.0,
        "kv_s0_extract_ms": kv_s0_extract_ms,
        "kv_s0_transfer_only_ms": 0.0,          # 后续填充
        "kv_s0_extract_plus_transfer_ms": 0.0,  # 后续填充
        "kv_s1_receive_ms": kv_s1_receive_ms,
        "kv_s1_tran_ms": kv_s1_tran_ms,
        "kv_s1_prep_ms": kv_s1_prep_ms,
        "s1_diffusion_ms": s1_diffusion_ms,
        "it": int(m.get("input_tokens", 0) or 0),
        "ot": len(co.get("cot_token_ids", []))}



def one(cid, msg, ai, idx, bs):
    """Send one request; return per-request metrics dict. Skip on timeout."""
    rid = f"bs{bs}-{cid[:8]}-{idx}"
    st = time.time()
    try:
        resp, client_timings = _post(f"http://{HOST}:{PORT}/v1/chat/completions", {
            "request_id": rid, "messages": msg, "add_generation_prompt": False,
            "continue_final_message": True, "additional_information": ai,
            "temperature": 0.6, "top_p": 0.98, "top_k": 40, "max_tokens": 256,
            "stop_token_ids": None, "seed": 0, "return_custom_output": True})
        lat = (time.time()-st)*1e3
        return _build_result_from_response(resp, cid, rid, lat, client_timings)
    except Exception as e:
        err_str = repr(e)[:200]
        print(f"  ERR rid={rid} {err_str}")
        return {"ok": False, "c": cid, "rid": rid,
                "lat": (time.time()-st)*1e3, "err": err_str}


async def one_async(cid, msg, ai, idx, bs, sem):
    """Send one request asynchronously; return per-request metrics dict."""
    rid = f"bs{bs}-{cid[:8]}-{idx}"
    st = time.time()
    async with sem:
        try:
            resp, client_timings = await _post_async(f"http://{HOST}:{PORT}/v1/chat/completions", {
                "request_id": rid, "messages": msg, "add_generation_prompt": False,
                "continue_final_message": True, "additional_information": ai,
                "temperature": 0.6, "top_p": 0.98, "top_k": 40, "max_tokens": 256,
                "stop_token_ids": None, "seed": 0, "return_custom_output": True})
            lat = (time.time()-st)*1e3
            return _build_result_from_response(resp, cid, rid, lat, client_timings)
        except Exception as e:
            err_str = repr(e)[:200]
            print(f"  ERR rid={rid} {err_str}")
            return {"ok": False, "c": cid, "rid": rid,
                    "lat": (time.time()-st)*1e3, "err": err_str}


async def run_group_async(grp, gi, bs):
    """Run one batch group; return (wall_ms, per_req_list, group_agg)."""
    workers = min(len(grp), 8 if bs <= 8 else 4)
    sem = asyncio.Semaphore(workers)
    t0 = time.time()
    tasks = [
        asyncio.create_task(one_async(cid, msg, ai, gi * bs + ii, bs, sem))
        for ii, (cid, msg, ai) in enumerate(grp)
    ]
    results = await asyncio.gather(*tasks)
    wall_ms = (time.time()-t0)*1e3
    return wall_ms, results, _build_group_agg(results, wall_ms)

def gen(all_batches, clip_stats):
    """3 tables: per-request detail, per-group stats, per-BS throughput."""
    L = ["# Batch Sweep", "",
         f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | {N_TOTAL} samples | GPU0",
         f"- disk_io mean={clip_stats['disk']:.0f}ms | prep mean={clip_stats['prep']:.0f}ms",
         "",
         "## 表1: 逐请求详细指标",
         "",
         "| bs | gid | rid | lat_ms | net_ms | inf_ms | s0_llm_ms | kv_tran_total | s1_diffusion_ms | kv_s0_extract_plus_transfer_ms | kv_s1_receive_ms | itok | otok |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    for bs in BS_LIST:
        for grp in all_batches.get(bs, []):
            for req in grp.get("reqs", []):
                if not req.get("ok"):
                    continue
                L.append(f"| {bs} | {grp['gid']} | {req['rid']} | {req['lat']:.0f} | {req['net']:.0f}"
                         f" | {req['inf']:.0f} | {req['s0_llm_ms']:.0f} | {req['kv_tran_total']:.0f}"
                         f" | {req['s1_diffusion_ms']:.0f} | {req['kv_s0_extract_plus_transfer_ms']:.0f}"
                         f" | {req['kv_s1_receive_ms']:.0f} | {req['it']} | {req['ot']} |")

    L += ["",
          "## 表2: 逐 BS 请求级指标统计",
          "",
          "说明：",
          "- metric: 指标名称。",
          "- min: 该 bs 下所有请求该指标最小值。",
          "- max: 该 bs 下所有请求该指标最大值。",
          "- mean: 该 bs 下所有请求该指标平均值。",
          "",
          "| bs | metric | min | max | mean |",
          "|---:|---:|---:|---:|---:|"]

    metric_names = [
        ("lat_ms", "lat"), ("net_ms", "net"), ("inf_ms", "inf"),
        ("client_encode_ms", "client_encode_ms"), ("client_http_ms", "client_http_ms"),
        ("client_decode_ms", "client_decode_ms"), ("client_codec_ms", "client_codec_ms"),
        ("openai_handler_total_ms", "openai_handler_total_ms"),
        ("openai_pre_full_generator_ms", "openai_pre_full_generator_ms"),
        ("openai_check_model_ms", "openai_check_model_ms"),
        ("openai_prepare_runtime_ms", "openai_prepare_runtime_ms"),
        ("openai_preprocess_chat_ms", "openai_preprocess_chat_ms"),
        ("openai_preprocess_merge_kwargs_ms", "openai_preprocess_merge_kwargs_ms"),
        ("openai_preprocess_build_params_ms", "openai_preprocess_build_params_ms"),
        ("openai_preprocess_audio_injection_ms", "openai_preprocess_audio_injection_ms"),
        ("openai_preprocess_render_chat_ms", "openai_preprocess_render_chat_ms"),
        ("openai_preprocess_get_tokenizer_ms", "openai_preprocess_get_tokenizer_ms"),
        ("openai_preprocess_tool_adjust_ms", "openai_preprocess_tool_adjust_ms"),
        ("openai_preprocess_image_cleanup_ms", "openai_preprocess_image_cleanup_ms"),
        ("openai_preprocess_finalize_prompt_ms", "openai_preprocess_finalize_prompt_ms"),
        ("api_server_pre_route_ms", "api_server_pre_route_ms"),
        ("api_server_request_body_ms", "api_server_request_body_ms"),
        ("api_server_request_json_ms", "api_server_request_json_ms"),
        ("api_server_pre_route_other_ms", "api_server_pre_route_other_ms"),
        ("api_server_request_body_bytes", "api_server_request_body_bytes"),
        ("api_server_endpoint_setup_ms", "api_server_endpoint_setup_ms"),
        ("api_server_handler_call_ms", "api_server_handler_call_ms"),
        ("api_server_post_handler_ms", "api_server_post_handler_ms"),
        ("api_server_response_dump_ms", "api_server_response_dump_ms"),
        ("api_server_response_render_ms", "api_server_response_render_ms"),
        ("api_server_route_total_ms", "api_server_route_total_ms"),
        ("openai_image_prompt_rewrite_ms", "openai_image_prompt_rewrite_ms"),
        ("openai_schedule_generator_ms", "openai_schedule_generator_ms"),
        ("openai_result_wait_ms", "openai_result_wait_ms"),
        ("openai_postprocess_ms", "openai_postprocess_ms"),
        ("openai_response_build_ms", "openai_response_build_ms"),
        ("openai_response_logging_ms", "openai_response_logging_ms"),
        ("s0_llm_ms", "s0_llm_ms"),
        ("kv_tran_total_ms", "kv_tran_total"),
        ("kv_s0_extract_ms", "kv_s0_extract_ms"),
        ("kv_s0_transfer_only_ms", "kv_s0_transfer_only_ms"),
        ("kv_s0_extract_plus_transfer_ms", "kv_s0_extract_plus_transfer_ms"),
        ("kv_s1_receive_ms", "kv_s1_receive_ms"),
        ("kv_s1_tran_ms", "kv_s1_tran_ms"),
        ("kv_s1_prep_ms", "kv_s1_prep_ms"),
        ("s1_diffusion_ms", "s1_diffusion_ms"),
        ("itok", "it"), ("otok", "ot"),
    ]
    for bs in BS_LIST:
        # collect all request values across all groups
        all_vals = {key: [] for _, key in metric_names}
        for grp in all_batches.get(bs, []):
            for req in grp.get("reqs", []):
                if not req.get("ok"):
                    continue
                for label, key in metric_names:
                    all_vals[key].append(req.get(key, 0))
        for label, key in metric_names:
            vals = all_vals[key]
            if not vals:
                continue
            L.append(f"| {bs} | {label} | {np.min(vals):.0f} | {np.max(vals):.0f} | {np.mean(vals):.0f} |")

    L += ["",
          "## 表3: 每 BS 吞吐量",
          "",
          "说明：",
          "- bs: 批大小。",
          "- n_grp: 该 bs 下分组数。",
          "- ok: 成功请求总数。",
          "- E2E_s: 全部分组累计端到端耗时（秒）。",
          "- batch_it/s: 输入 token 吞吐（tokens/s）。",
          "- batch_ot/s: 输出 token 吞吐（tokens/s）。",
          "- batch_at/s: 总 token 吞吐（输入+输出，tokens/s）。",
          "- avg_sample_ms: 平均每样本耗时（毫秒）。",
          "- samples/s: 样本吞吐（samples/s）。",
          "- 相比BS=1: 相对 bs=1 的吞吐加速比。",
          "",
          "| bs | n_grp | ok | E2E_s"
          " | batch_it/s | batch_ot/s | batch_at/s"
          " | avg_sample_ms | samples/s | 相比BS=1 |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

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
        L.append(f"| {bs} | {len(grps)} | {ok_total} | {e2e:.1f}"
                 f" | {total_it/e2e:.0f} | {total_ot/e2e:.0f} | {total_at/e2e:.0f}"
                 f" | {avg_sample_ms:.0f} | {samples_per_s:.3f} | {speedup:.2f}x |")

    OUT.write_text("\n".join(L))
    print(f"  -> {OUT}")

async def main_async():
    print(f"Batch sweep: {N_TOTAL} samples ({N_UNIQUE} unique clips cycled), BS={BS_LIST}")
    _build_runtime_stage_config()
    print("Loading dataset + preparing clips...")
    avdi = _load_local_avdi()
    ci = avdi.clip_index
    print(len(list(ci[ci.chunk==3116].index)))
    unique_cids = list(ci[ci.chunk==3116].index)[:N_UNIQUE]

    clips_unique, disk_ms_list, prep_ms_list = [], [], []
    for i, cid in enumerate(unique_cids):
        td = time.time()
        data = _load_clip_data(cid, 5100000, avdi)
        dms = (time.time()-td)*1e3
        tp = time.time()
        fr = data["image_frames"].flatten(0, 1)
        pm = _build_prompt_messages(
            camera_indices=data["camera_indices"],
            num_frames_per_camera=int(data["image_frames"].shape[1]))
        rm = _adapt_messages_with_frames(pm, fr)
        ai = {"ego_history_xyz": data["ego_history_xyz"].cpu(),
              "ego_history_rot": data["ego_history_rot"].cpu(),
              "alpamayo_model_path": MODEL}
        pms = (time.time()-tp)*1e3
        clips_unique.append((cid, rm, ai))
        disk_ms_list.append(dms)
        prep_ms_list.append(pms)
        print(f"  {i+1}/{N_UNIQUE} clip={cid[:8]} disk={dms:.0f}ms prep={pms:.0f}ms")

    cs = {"disk": np.mean(disk_ms_list), "prep": np.mean(prep_ms_list)}
    print(f"Preload done. disk_mean={cs['disk']:.0f}ms prep_mean={cs['prep']:.0f}ms\n")

    # Cycle unique clips to N_TOTAL samples
    samples = (clips_unique * ((N_TOTAL // N_UNIQUE) + 1))[:N_TOTAL]

    all_batches = {}
    for bs in BS_LIST:
        if bs > MAX_REQ_PER_GROUP:
            raise ValueError(f"BS={bs} exceeds request-side cap MAX_REQ_PER_GROUP={MAX_REQ_PER_GROUP}")
        _restart_service(bs)
        n_groups = (N_TOTAL + bs - 1) // bs
        grp_per_chunk = max(1, CHUNK_SAMPLES // bs)
        print(f"BATCH={bs}  ({n_groups} groups x {bs}, chunk={grp_per_chunk} groups)")
        bs_data = []  # list of {gid, reqs: [...], agg: {...}}
        done_reqs = 0
        profile_gids = _parse_profile_gid()

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
                    _set_profiler_enabled(True)
                try:
                    wm, results, agg = await run_group_async(grp, gi, bs)
                finally:
                    if should_profile_group:
                        _set_profiler_enabled(False, 3600)
                bs_data.append({"gid": gi, "reqs": results, "agg": agg})
                total_ok = sum(d["agg"]["ok"] for d in bs_data)
                done_reqs += len(grp)
                print(f"  g{gi+1:4d}/{n_groups} ok={agg['ok']}/{len(grp)} "
                      f"wall={wm:.0f}ms [cum ok={total_ok}/{done_reqs}]")
            if chunk_end < n_groups:
                print(f"  chunk done, sleep 8s...")
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

    print(f"\nALL DONE => {OUT}")

if __name__ == "__main__":
    asyncio.run(main_async())
