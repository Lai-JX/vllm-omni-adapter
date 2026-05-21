#!/usr/bin/env python3
"""Extract per-request timestamps from vllm-omni metrics logs.

Example:
    python profiler/extract_request_timestamps.py \
        profiler/logs/64/async_omni_trace-gid_1778778376/svc_logs/svc_bs8.log
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


PATTERNS = [
    (
        "stage0",
        re.compile(
            r"Stage 0 req (\S+) gen_time_ms=([\d.]+) start=([\d.]+) now=([\d.]+)"
        ),
        "gen_time_ms",
    ),
    (
        "kv_send",
        re.compile(r"KV Send req (\S+) time_ms=([\d.]+) start=([\d.]+) now=([\d.]+)"),
        "time_ms",
    ),
    (
        "kv_recv",
        re.compile(
            r"KV Receive req (\S+) time_ms=([\d.]+) start=([\d.]+) now=([\d.]+)"
        ),
        "time_ms",
    ),
    (
        "diffusion",
        re.compile(
            r"Stage 1 diffusion req (\S+) time_ms=([\d.]+) start=([\d.]+) now=([\d.]+)"
        ),
        "time_ms",
    ),
    (
        "stage1",
        re.compile(
            r"Stage 1 req (\S+) gen_time_ms=([\d.]+) start=([\d.]+) now=([\d.]+)"
        ),
        "gen_time_ms",
    ),
]

PHASES = ["stage0", "kv_send", "kv_recv", "diffusion", "stage1"]
STAGE0_PROFILE_PATTERN = re.compile(r"\[Stage0Profile\]\s+(\{.*\})")
STAGE0_PROFILE_FIELDS = [
    "prompt_tokens",
    "output_tokens",
    "scheduled_tokens",
    "batch_size",
    "batch_total_scheduled_tokens",
    "embed_multimodal_ms",
    "forward_ms",
]
BS_PATTERNS = [
    re.compile(r"(?:^|[_-])bs[_-]?(\d+)(?:\D|$)", re.IGNORECASE),
    re.compile(r"\bbs(\d+)\b", re.IGNORECASE),
]


def parse_log(path: Path) -> tuple[list[str], dict[str, dict[str, dict[str, str]]], dict[str, dict[str, str]]]:
    rows: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    order: list[str] = []
    profiles: dict[str, dict[str, str]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            profile_match = STAGE0_PROFILE_PATTERN.search(line)
            if profile_match:
                try:
                    payload = json.loads(profile_match.group(1))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    req_id = payload.get("request_id")
                    if isinstance(req_id, str) and req_id:
                        if req_id not in rows:
                            order.append(req_id)
                        profile_data = {
                            key: str(payload.get(key, ""))
                            for key in STAGE0_PROFILE_FIELDS
                        }
                        start_ts = payload.get("start")
                        now_ts = payload.get("now")
                        if start_ts is not None and now_ts is not None:
                            rows[req_id]["stage0_profile"] = {
                                "time_ms": str(payload.get("embed_multimodal_ms", "") or ""),
                                "start": str(start_ts),
                                "now": str(now_ts),
                            }
                        profiles[req_id] = profile_data
                continue
            for phase, pattern, duration_key in PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue
                req_id, duration_ms, start_ts, now_ts = match.groups()
                if req_id not in rows:
                    order.append(req_id)
                rows[req_id][phase] = {
                    duration_key: duration_ms,
                    "start": start_ts,
                    "now": now_ts,
                }
                break

    return order, rows, profiles


def infer_batch_size(path: Path) -> int | None:
    candidates = [path.name]
    candidates.extend(parent.name for parent in path.parents)
    for candidate in candidates:
        for pattern in BS_PATTERNS:
            match = pattern.search(candidate)
            if match:
                return int(match.group(1))
    return None


def build_table(
    order: list[str],
    rows: dict[str, dict[str, dict[str, str]]],
    batch_size: int | None,
    profiles: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    table: list[dict[str, str]] = []
    for index, req_id in enumerate(order):
        row: dict[str, str] = {"request_id": req_id}
        if batch_size:
            row["bs"] = str(batch_size)
            row["group_id"] = str(index // batch_size + 1)
            row["group_pos"] = str(index % batch_size + 1)
        for phase in PHASES:
            phase_data = rows[req_id].get(phase, {})
            row[f"{phase}_start"] = phase_data.get("start", "")
            row[f"{phase}_now"] = phase_data.get("now", "")
            row[f"{phase}_ms"] = phase_data.get("gen_time_ms", "") or phase_data.get(
                "time_ms", ""
            )
        for key in STAGE0_PROFILE_FIELDS:
            row[key] = profiles.get(req_id, {}).get(key, "")
        table.append(row)
    return table


def write_csv(table: list[dict[str, str]], output_path: Path, delimiter: str) -> None:
    if not table:
        fieldnames = ["request_id"]
    else:
        fieldnames = list(table[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(table)


def print_pretty(table: list[dict[str, str]]) -> None:
    if not table:
        print("No matching request metrics found.", file=sys.stderr)
        return

    current_group_id = None
    for row in table:
        group_id = row.get("group_id")
        if group_id and group_id != current_group_id:
            if current_group_id is not None:
                print()
            print(f"group {group_id} (bs={row['bs']})")
            current_group_id = group_id
        print(row["request_id"])
        for phase in PHASES:
            start = row[f"{phase}_start"]
            now = row[f"{phase}_now"]
            duration = row[f"{phase}_ms"]
            if not start and not now and not duration:
                continue
            print(f"  {phase}: start={start} now={now} dur_ms={duration}")
        print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract per-request timestamps from vllm-omni metrics logs."
    )
    parser.add_argument("logfile", type=Path, help="Path to the input log file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional output file path. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=["pretty", "tsv", "csv", "json"],
        default="pretty",
        help="Output format. Defaults to pretty.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        help="Override batch size. If omitted, the script tries to extract it from the file path.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.logfile.exists():
        parser.error(f"log file does not exist: {args.logfile}")

    batch_size = args.bs if args.bs is not None else infer_batch_size(args.logfile)
    if batch_size is not None and batch_size <= 0:
        parser.error(f"batch size must be positive, got: {batch_size}")

    order, rows, profiles = parse_log(args.logfile)
    table = build_table(order, rows, batch_size, profiles)

    if args.format == "pretty":
        if args.output:
            with args.output.open("w", encoding="utf-8") as handle:
                original_stdout = sys.stdout
                try:
                    sys.stdout = handle
                    print_pretty(table)
                finally:
                    sys.stdout = original_stdout
        else:
            print_pretty(table)
        return 0

    if args.format == "json":
        text = json.dumps(table, ensure_ascii=False, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    delimiter = "\t" if args.format == "tsv" else ","
    if args.output:
        write_csv(table, args.output, delimiter)
    else:
        if not table:
            return 0
        writer = csv.DictWriter(
            sys.stdout, fieldnames=list(table[0].keys()), delimiter=delimiter
        )
        writer.writeheader()
        writer.writerows(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
