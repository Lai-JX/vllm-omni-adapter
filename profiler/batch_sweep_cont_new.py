"""Batch sweep: 256 samples (22 unique clips cycled), per-batch-group rows."""
import json, sys, time, numpy as np, subprocess, os
import urllib.request as urllib_request
from urllib.error import HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OMNI = Path(__file__).resolve().parents[1]
for p in [str(OMNI), str(OMNI.parent/"alpamayo1.5"/"src"), str(OMNI.parent/"verl-liming"/"my_example"/"alpamayo"/"src")]:
    sys.path.insert(0, p)

from profiler.run_rollout_timing import (
    _load_local_avdi, _load_clip_data, _build_prompt_messages,
    _adapt_messages_with_frames, _to_jsonable,
)

WORKSPACE_ROOT = OMNI.parents[1]
DEFAULT_MODEL_PATH = WORKSPACE_ROOT / "model" / "Alpamayo-1.5-10B"
MODEL    = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
HOST, PORT = "127.0.0.1", 8300
N_UNIQUE = 22
N_TOTAL  = 256
BS_LIST  = [1, 2, 4, 8, 12, 16, 24]
MAX_REQ_PER_GROUP = 24  # Request-side cap for each vLLM processing group
CHUNK_SAMPLES = 16  # max samples per chunk
OUT      = OMNI / "profiler" / "batch_sweep_results_new.md"
SVC_YAML = OMNI / "profiler" / "alpamayo1_5_gpu0.yaml"

def _restart_service(bs_val):
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
    cmd = (f"vllm-omni serve {MODEL} --omni --host 127.0.0.1 --port {PORT} "
           f"--served-model-name alpamayo1.5 --stage-configs-path {SVC_YAML}")
    log = OMNI / "profiler" / "logs" / f"svc_bs{bs_val}.log"
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
    data = json.dumps(_to_jsonable(payload)).encode("utf-8")
    req = urllib_request.Request(url, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} when POST {url}\n{body[:300]}") from exc


def one(cid, msg, ai, idx, bs):
    """Send one request; return per-request metrics dict. Skip on timeout."""
    rid = f"bs{bs}-{cid[:8]}-{idx}"
    st = time.time()
    try:
        resp = _post(f"http://{HOST}:{PORT}/v1/chat/completions", {
            "request_id": rid, "messages": msg, "add_generation_prompt": False,
            "continue_final_message": True, "additional_information": ai,
            "temperature": 0.6, "top_p": 0.98, "top_k": 40, "max_tokens": 256,
            "stop_token_ids": None, "seed": 0, "return_custom_output": True})
        lat = (time.time()-st)*1e3
        m = resp.get("metrics", {}) or {}
        co = m.get("custom_output", {}) or {}
        inf = float(m.get("inference_only_ms", 0) or 0)
        llm_ms = float(m.get("stage0_llm_ms", 0) or 0)
        kv_tran_s0 = float(co.get("kv_tran_s0_ms", 0) or 0)
        kv_tran_s1_receive = float(co.get("kv_tran_s1_receive_ms", 0) or 0)
        df = float(co.get("df_ms", 0) or 0)
        s0  = llm_ms + kv_tran_s0
        s1  = kv_tran_s1_receive + df
        return {"ok": True, "c": cid, "rid": rid, "lat": lat, "inf": inf,
                "s0": s0, "s1": s1, "qw": max(0., inf-s0-s1), "net": lat-inf,
            "kv": kv_tran_s1_receive,
            "kv_tran_s0": kv_tran_s0,
            "kv_tran_s1_receive": kv_tran_s1_receive,
                "kv_tran_s1_prep": float(co.get("kv_tran_s1_prep_ms", 0) or 0),
            "df": df,
                "it": int(m.get("input_tokens", 0) or 0),
                "ot": len(co.get("cot_token_ids", []))}
    except Exception as e:
        err_str = repr(e)[:200]
        print(f"  ERR rid={rid} {err_str}")
        return {"ok": False, "c": cid, "rid": rid,
                "lat": (time.time()-st)*1e3, "err": err_str}


def run_group(grp, gi, bs):
    """Run one batch group; return (wall_ms, per_req_list, group_agg)."""
    workers = min(len(grp), 8 if bs <= 8 else 4)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as p:
        fs = [p.submit(one, cid, msg, ai, gi*bs+ii, bs)
              for ii, (cid, msg, ai) in enumerate(grp)]
        results = [f.result() for f in as_completed(fs)]
    wall_ms = (time.time()-t0)*1e3
    ok_results = [r for r in results if r.get("ok")]
    n_ok = len(ok_results)
    agg = {"wall_ms": wall_ms, "ok": n_ok}
    if n_ok:
        for k in ["inf","s0","s1","kv_tran_s0","kv_tran_s1_receive","kv_tran_s1_prep","df","qw","net","it","ot"]:
            vals = [r[k] for r in ok_results]
            agg[f"avg_{k}"] = np.mean(vals)
            agg[f"min_{k}"] = np.min(vals)
            agg[f"max_{k}"] = np.max(vals)
        agg["bs_it"] = sum(r["it"] for r in ok_results)
        agg["bs_ot"] = sum(r["ot"] for r in ok_results)
        agg["bs_at"] = agg["bs_it"] + agg["bs_ot"]
    else:
        for k in ["inf","s0","s1","kv_tran_s0","kv_tran_s1_receive","kv_tran_s1_prep","df","qw","net","it","ot"]:
            agg[f"avg_{k}"] = agg[f"min_{k}"] = agg[f"max_{k}"] = 0.0
        agg["bs_it"] = agg["bs_ot"] = agg["bs_at"] = 0
    return wall_ms, results, agg

def gen(all_batches, clip_stats):
    """3 tables: per-request detail, per-group stats, per-BS throughput."""
    L = ["# Batch Sweep", "",
         f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | {N_TOTAL} samples | GPU0",
         f"- disk_io mean={clip_stats['disk']:.0f}ms | prep mean={clip_stats['prep']:.0f}ms",
         "",
         "## 表1: 逐请求详细指标",
         "",
         "| bs | gid | rid | lat_ms | net_ms | inf_ms | qw_ms | s0_ms | s1_ms | kv_ms | df_ms | itok | otok |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    for bs in BS_LIST:
        for grp in all_batches.get(bs, []):
            for req in grp.get("reqs", []):
                if not req.get("ok"):
                    continue
                L.append(f"| {bs} | {grp['gid']} | {req['rid']} | {req['lat']:.0f} | {req['net']:.0f}"
                         f" | {req['inf']:.0f} | {req['qw']:.0f} | {req['s0']:.0f} | {req['s1']:.0f}"
                         f" | {req['kv']:.0f} | {req['df']:.0f} | {req['it']} | {req['ot']} |")

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
        ("qw_ms", "qw"), ("s0_ms", "s0"), ("s1_ms", "s1"),
        ("kv_tran_s0_ms", "kv_tran_s0"), ("kv_tran_s1_receive_ms", "kv_tran_s1_receive"),
        ("kv_tran_s1_prep_ms", "kv_tran_s1_prep"),
        ("df_ms", "df"), ("itok", "it"), ("otok", "ot"),
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

def main():
    print(f"Batch sweep: {N_TOTAL} samples ({N_UNIQUE} unique clips cycled), BS={BS_LIST}")
    print("Loading dataset + preparing clips...")
    avdi = _load_local_avdi()
    ci = avdi.clip_index
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

        for chunk_start in range(0, n_groups, grp_per_chunk):
            chunk_end = min(chunk_start + grp_per_chunk, n_groups)
            for gi in range(chunk_start, chunk_end):
                start = gi * bs
                end = min(start + bs, N_TOTAL)
                grp = samples[start:end]
                if len(grp) > MAX_REQ_PER_GROUP:
                    raise ValueError(f"Group size {len(grp)} exceeds cap {MAX_REQ_PER_GROUP}")
                wm, results, agg = run_group(grp, gi, bs)
                bs_data.append({"gid": gi, "reqs": results, "agg": agg})
                total_ok = sum(d["agg"]["ok"] for d in bs_data)
                done_reqs += len(grp)
                print(f"  g{gi+1:4d}/{n_groups} ok={agg['ok']}/{len(grp)} "
                      f"wall={wm:.0f}ms [cum ok={total_ok}/{done_reqs}]")
            if chunk_end < n_groups:
                print(f"  chunk done, sleep 8s...")
                time.sleep(8)

        all_batches[bs] = bs_data
        total_ok = sum(d["agg"]["ok"] for d in bs_data)
        total_wall = sum(d["agg"]["wall_ms"] for d in bs_data)
        print(f"  DONE ok={total_ok}/{N_TOTAL} E2E={total_wall/1e3:.1f}s")
        gen(all_batches, cs)

    print(f"\nALL DONE => {OUT}")

if __name__ == "__main__":
    main()
