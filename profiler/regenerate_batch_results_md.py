"""Regenerate batch_results.md from metrics JSON files.

Example:
    python profiler/regenerate_batch_results_md.py \
        --metrics-dir /workspace/project/RL-learning/vllm-omni/profiler/logs/8/async_omni_trace-gid3_1778753529/metrics \
        --add-field kv_tran_s0_total_ms
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

MD_SAMPLE_RE = re.compile(r"- .*?\| (?P<samples>\d+) samples \| (?P<gpu>.+)$")
MD_CLIP_RE = re.compile(r"- disk_io mean=(?P<disk>[\d.]+)ms \| prep mean=(?P<prep>[\d.]+)ms")
BS_JSON_RE = re.compile(r"batch_results_bs_(?P<bs>\d+)\.json$")

BASE_DETAIL_COLUMNS = [
    ("lat_ms", "lat"),
    ("net_ms", "net"),
    ("inf_ms", "inf"),
    ("qw_ms", "qw"),
    ("s0_ms", "s0"),
    ("s1_ms", "s1"),
    ("kv_ms", "kv"),
    ("df_ms", "df"),
    ("itok", "it"),
    ("otok", "ot"),
]

BASE_METRIC_NAMES = [
    ("lat_ms", "lat"),
    ("net_ms", "net"),
    ("inf_ms", "inf"),
    ("client_encode_ms", "client_encode_ms"),
    ("client_http_ms", "client_http_ms"),
    ("client_decode_ms", "client_decode_ms"),
    ("client_codec_ms", "client_codec_ms"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate batch_results.md from JSON metrics.")
    parser.add_argument("--metrics-dir", required=True, type=Path, help="Path to metrics directory.")
    parser.add_argument(
        "--add-field",
        action="append",
        default=[],
        help="Per-request field to add to markdown tables. Repeatable.",
    )
    parser.add_argument(
        "--del-field",
        action="append",
        default=[],
        help="Field to remove from markdown tables. Repeatable.",
    )
    return parser.parse_args()


def parse_existing_md(md_path: Path) -> tuple[dict[str, float], int | None, str]:
    clip_stats = {"disk": 0.0, "prep": 0.0}
    sample_count = None
    gpu_label = "GPU0"

    if not md_path.exists():
        return clip_stats, sample_count, gpu_label

    text = md_path.read_text(encoding="utf-8", errors="replace")
    if sample_match := MD_SAMPLE_RE.search(text):
        sample_count = int(sample_match.group("samples"))
        gpu_label = sample_match.group("gpu").strip()
    if clip_match := MD_CLIP_RE.search(text):
        clip_stats = {
            "disk": float(clip_match.group("disk")),
            "prep": float(clip_match.group("prep")),
        }
    return clip_stats, sample_count, gpu_label


def load_all_batches(metrics_dir: Path) -> tuple[dict[int, list[dict[str, Any]]], dict[str, float]]:
    all_batches: dict[int, list[dict[str, Any]]] = {}
    fallback_clip_stats = {"disk": 0.0, "prep": 0.0}

    for json_path in sorted(metrics_dir.glob("batch_results_bs_*.json")):
        match = BS_JSON_RE.match(json_path.name)
        if not match:
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        payload_batches = payload.get("all_batches", {})
        for bs_key, groups in payload_batches.items():
            all_batches[int(bs_key)] = groups
        cs = payload.get("cs")
        if isinstance(cs, dict):
            fallback_clip_stats["disk"] = float(cs.get("disk", fallback_clip_stats["disk"]))
            fallback_clip_stats["prep"] = float(cs.get("prep", fallback_clip_stats["prep"]))

    if not all_batches:
        raise RuntimeError(f"No batch_results_bs_*.json files found in {metrics_dir}")

    return dict(sorted(all_batches.items())), fallback_clip_stats


def merge_metric_names(extra_fields: list[str]) -> list[tuple[str, str]]:
    merged = list(BASE_METRIC_NAMES)
    seen = {key for _, key in merged}
    for field in extra_fields:
        if field not in seen:
            merged.append((field, field))
            seen.add(field)
    return merged


def merge_detail_columns(extra_fields: list[str]) -> list[tuple[str, str]]:
    detail = list(BASE_DETAIL_COLUMNS[:-2])
    seen = {key for _, key in detail}
    for field in extra_fields:
        if field not in seen:
            detail.append((field, field))
            seen.add(field)
    detail.extend(BASE_DETAIL_COLUMNS[-2:])
    return detail


def filter_fields(fields: list[tuple[str, str]], removed_fields: list[str]) -> list[tuple[str, str]]:
    if not removed_fields:
        return fields
    removed = set(removed_fields)
    return [(label, key) for label, key in fields if key not in removed]


def infer_sample_count(all_batches: dict[int, list[dict[str, Any]]]) -> int:
    sample_count = 0
    for groups in all_batches.values():
        sample_count = max(sample_count, sum(len(g.get("reqs", [])) for g in groups))
    return sample_count


def render_markdown(
    *,
    all_batches: dict[int, list[dict[str, Any]]],
    clip_stats: dict[str, float],
    sample_count: int,
    gpu_label: str,
    detail_columns: list[tuple[str, str]],
    metric_names: list[tuple[str, str]],
) -> str:
    bs_list = sorted(all_batches)
    detail_header = " | ".join(["bs", "gid", "rid"] + [label for label, _ in detail_columns])
    detail_sep = " | ".join(["---:"] * (3 + len(detail_columns)))

    lines = [
        "# Batch Sweep",
        "",
        f"- {time.strftime('%Y-%m-%d %H:%M:%S')} | {sample_count} samples | {gpu_label}",
        f"- disk_io mean={clip_stats['disk']:.0f}ms | prep mean={clip_stats['prep']:.0f}ms",
        "",
        "## 表1: 逐请求详细指标",
        "",
        f"| {detail_header} |",
        f"| {detail_sep} |",
    ]

    for bs in bs_list:
        for grp in all_batches.get(bs, []):
            for req in grp.get("reqs", []):
                if not req.get("ok"):
                    continue
                values = []
                for _, key in detail_columns:
                    value = req.get(key, 0)
                    if isinstance(value, float):
                        values.append(f"{value:.0f}")
                    else:
                        values.append(str(value))
                lines.append(f"| {bs} | {grp['gid']} | {req['rid']} | " + " | ".join(values) + " |")

    lines += [
        "",
        "## 表2: 逐 BS 请求级指标统计",
        "",
        "说明：",
        "- metric: 指标名称。",
        "- min: 该 bs 下所有请求该指标最小值。",
        "- max: 该 bs 下所有请求该指标最大值。",
        "- mean: 该 bs 下所有请求该指标平均值。",
        "",
        "| bs | metric | min | max | mean |",
        "|---:|---:|---:|---:|---:|",
    ]

    for bs in bs_list:
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
        "| bs | n_grp | ok | E2E_s | batch_it/s | batch_ot/s | batch_at/s | avg_sample_ms | samples/s | 相比BS=1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    base_ss = None
    for bs in bs_list:
        groups = [g for g in all_batches.get(bs, []) if g.get("agg", {}).get("ok", 0) > 0]
        if not groups:
            continue
        ok_total = sum(g["agg"]["ok"] for g in groups)
        e2e = sum(g["agg"]["wall_ms"] for g in groups) / 1e3
        total_it = sum(g["agg"]["bs_it"] for g in groups)
        total_ot = sum(g["agg"]["bs_ot"] for g in groups)
        total_at = sum(g["agg"]["bs_at"] for g in groups)
        total_req = sum(len(g.get("reqs", [])) for g in groups)
        avg_sample_ms = e2e * 1e3 / total_req if total_req > 0 else 0
        samples_per_s = total_req / e2e if e2e > 0 else 0
        if base_ss is None:
            base_ss = samples_per_s
        speedup = samples_per_s / base_ss if base_ss else 0
        lines.append(
            f"| {bs} | {len(groups)} | {ok_total} | {e2e:.1f} | "
            f"{total_it / e2e:.0f} | {total_ot / e2e:.0f} | {total_at / e2e:.0f} | "
            f"{avg_sample_ms:.0f} | {samples_per_s:.3f} | {speedup:.2f}x |"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    metrics_dir = args.metrics_dir.resolve()
    md_path = metrics_dir / "batch_results.md"

    clip_stats, sample_count, gpu_label = parse_existing_md(md_path)
    all_batches, fallback_clip_stats = load_all_batches(metrics_dir)
    if clip_stats == {"disk": 0.0, "prep": 0.0}:
        clip_stats = fallback_clip_stats
    if sample_count is None:
        sample_count = infer_sample_count(all_batches)

    detail_columns = filter_fields(merge_detail_columns(args.add_field), args.del_field)
    metric_names = filter_fields(merge_metric_names(args.add_field), args.del_field)
    markdown = render_markdown(
        all_batches=all_batches,
        clip_stats=clip_stats,
        sample_count=sample_count,
        gpu_label=gpu_label,
        detail_columns=detail_columns,
        metric_names=metric_names,
    )
    md_path.write_text(markdown, encoding="utf-8")
    print(f"Updated {md_path}")


if __name__ == "__main__":
    main()
