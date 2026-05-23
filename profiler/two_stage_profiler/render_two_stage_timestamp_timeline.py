#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

PHASE_SPECS = [
    {
        "id": "stage0_submit",
        "label": "Stage 0 total",
        "color": "#2456d3",
        "start_field": "stage0_submit_start",
        "end_field": "stage0_submit_end",
    },
    {
        "id": "embed",
        "label": "Embed",
        "color": "#5b8ff9",
        "start_field": "embed_start",
        "end_field": "embed_end",
    },
    {
        "id": "forward",
        "label": "Forward",
        "color": "#91b7ff",
        "start_field": "forward_start",
        "end_field": "forward_end",
    },
    {
        "id": "subsequent_decode",
        "label": "Subsequent decode",
        "color": "#33a02c",
        "start_field": "forward_end",
        "end_field": "kv_s0_start_time",
    },
    {
        "id": "kv_send",
        "label": "KV send",
        "color": "#e2742f",
        "start_field": "kv_s0_start_time",
        "end_field": "kv_s0_end_time",
    },
    {
        "id": "stage1_submit",
        "label": "Stage 1 total",
        "color": "#3f3a96",
        "start_field": "stage1_submit_start",
        "end_field": "stage1_submit_end",
    },
    {
        "id": "stage1_queue",
        "label": "Stage 1 queue",
        "color": "#7a68c7",
        "start_field": "stage1_submit_start",
        "end_field": "kv_s1_receive_start_time",
    },
    {
        "id": "kv_recv",
        "label": "KV receive",
        "color": "#1f8f6b",
        "start_field": "kv_s1_receive_start_time",
        "end_field": "kv_s1_end_time",
    },
    {
        "id": "stage1_gap",
        "label": "Receive → diffusion",
        "color": "#2aa198",
        "start_field": "kv_s1_end_time",
        "end_field": "stage1_diffusion_start_time",
    },
    {
        "id": "diffusion",
        "label": "Diffusion",
        "color": "#bf3277",
        "start_field": "stage1_diffusion_start_time",
        "end_field": "stage1_diffusion_end_time",
    },
]

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f5efe3;
      --panel: rgba(255, 250, 242, 0.94);
      --ink: #1e1a17;
      --muted: #6b6258;
      --grid: rgba(78, 63, 47, 0.12);
      --grid-strong: rgba(78, 63, 47, 0.22);
      --accent: #af3a2f;
      --shadow: 0 18px 40px rgba(68, 42, 18, 0.12);
      --label-width: 320px;
      --lane-height: 14px;
      --lane-gap: 8px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.9), transparent 28%),
        linear-gradient(180deg, #efe6d7 0%, #f7f2e9 48%, #efe5d8 100%);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }

    .shell {
      max-width: 1840px;
      margin: 0 auto;
      padding: 24px 20px 32px;
    }

    .hero {
      background: linear-gradient(135deg, rgba(255,250,242,0.94), rgba(249,239,224,0.92));
      border: 1px solid rgba(90, 68, 45, 0.12);
      border-radius: 26px;
      box-shadow: var(--shadow);
      padding: 24px;
      position: sticky;
      top: 0;
      z-index: 12;
      backdrop-filter: blur(10px);
    }

    .eyebrow {
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 8px;
      font-weight: 700;
    }

    h1 {
      margin: 0;
      font-size: 34px;
      line-height: 1.05;
      font-family: "IBM Plex Serif", Georgia, serif;
    }

    .sub {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }

    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }

    .chip {
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(90, 68, 45, 0.12);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }

    .controls {
      margin-top: 18px;
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 16px;
    }

    .panel {
      background: rgba(255,255,255,0.56);
      border: 1px solid rgba(90, 68, 45, 0.12);
      border-radius: 18px;
      padding: 14px;
    }

    .panel-title {
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
      margin-bottom: 10px;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.78);
      border: 1px solid rgba(90, 68, 45, 0.12);
      font-size: 13px;
      cursor: pointer;
    }

    .legend-item.off {
      opacity: 0.35;
    }

    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      flex: none;
    }

    .control label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }

    .control input[type="range"] {
      width: 100%;
    }

    .control input[type="text"] {
      width: 100%;
      padding: 10px 12px;
      border-radius: 12px;
      border: 1px solid rgba(90, 68, 45, 0.14);
      font: inherit;
      background: rgba(255,255,255,0.9);
    }

    .timeline-card {
      margin-top: 18px;
      background: var(--panel);
      border: 1px solid rgba(90, 68, 45, 0.12);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .timeline-scroll {
      overflow: auto;
      max-height: calc(100vh - 260px);
    }

    .request-row {
      display: grid;
      grid-template-columns: var(--label-width) 1fr;
      min-width: 1200px;
      border-top: 1px solid rgba(90, 68, 45, 0.08);
      background: rgba(255,255,255,0.5);
    }

    .request-row:first-child {
      border-top: none;
    }

    .req-label {
      padding: 16px 18px;
      border-right: 1px solid rgba(90, 68, 45, 0.08);
      background: rgba(255,255,255,0.65);
      position: sticky;
      left: 0;
      z-index: 3;
    }

    .req-id {
      font-weight: 700;
      font-size: 15px;
      word-break: break-all;
    }

    .req-meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }

    .req-lanes {
      position: relative;
      padding: 16px 0 18px;
      background-image: linear-gradient(to right, var(--grid) 1px, transparent 1px);
      background-size: var(--grid-step, 120px) 100%;
    }

    .tick-band {
      position: sticky;
      top: 0;
      z-index: 4;
      display: grid;
      grid-template-columns: var(--label-width) 1fr;
      min-width: 1200px;
      background: rgba(255,250,242,0.97);
      border-bottom: 1px solid rgba(90, 68, 45, 0.1);
      backdrop-filter: blur(8px);
    }

    .tick-band .left {
      border-right: 1px solid rgba(90, 68, 45, 0.08);
      padding: 14px 18px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
    }

    .tick-band .right {
      position: relative;
      height: 44px;
      background-image: linear-gradient(to right, var(--grid-strong) 1px, transparent 1px);
      background-size: var(--grid-step, 120px) 100%;
    }

    .tick {
      position: absolute;
      top: 0;
      transform: translateX(-50%);
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }

    .lane {
      position: relative;
      height: var(--lane-height);
      margin: 0 0 var(--lane-gap) 0;
    }

    .lane-name {
      position: absolute;
      left: 10px;
      top: -1px;
      font-size: 10px;
      color: var(--muted);
      letter-spacing: 0.04em;
      text-transform: uppercase;
      pointer-events: none;
    }

    .bar {
      position: absolute;
      top: 0;
      height: var(--lane-height);
      border-radius: 999px;
      min-width: 2px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.12);
      cursor: pointer;
    }

    .bar:hover {
      outline: 2px solid rgba(30, 26, 23, 0.35);
      outline-offset: 1px;
    }

    .tooltip {
      position: fixed;
      z-index: 999;
      pointer-events: none;
      max-width: 320px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(30, 26, 23, 0.94);
      color: #fff9f0;
      box-shadow: 0 18px 38px rgba(0,0,0,0.24);
      font-size: 12px;
      line-height: 1.45;
      opacity: 0;
      transform: translateY(6px);
      transition: opacity 0.12s ease, transform 0.12s ease;
      white-space: pre-line;
    }

    .tooltip.show {
      opacity: 1;
      transform: translateY(0);
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">Two-stage timestamps</div>
      <h1>Two-stage request timeline</h1>
      <div class="sub">Source: __SOURCE__</div>
      <div class="chip-row" id="summaryChips"></div>
      <div class="controls">
        <div class="panel">
          <div class="panel-title">Phases</div>
          <div class="legend" id="legend"></div>
        </div>
        <div class="panel">
          <div class="control">
            <label for="zoomRange">Pixels per second</label>
            <input id="zoomRange" type="range" min="20" max="240" step="5" value="90" />
          </div>
          <div class="control" style="margin-top: 14px;">
            <label for="searchBox">Request filter</label>
            <input id="searchBox" type="text" placeholder="filter request id" />
          </div>
        </div>
      </div>
    </section>

    <section class="timeline-card">
      <div class="timeline-scroll" id="timelineScroll">
        <div id="timelineMount"></div>
      </div>
    </section>
  </div>

  <div class="tooltip" id="tooltip"></div>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const payload = JSON.parse(document.getElementById('payload').textContent);
    const tooltip = document.getElementById('tooltip');
    const legend = document.getElementById('legend');
    const mount = document.getElementById('timelineMount');
    const zoomRange = document.getElementById('zoomRange');
    const searchBox = document.getElementById('searchBox');
    const summaryChips = document.getElementById('summaryChips');
    const phaseOrder = payload.phase_order;
    const phaseLabels = payload.phase_labels;
    const phaseColors = payload.phase_colors;
    const allRequests = payload.requests;
    const globalStart = payload.global_start;
    const globalEnd = payload.global_end;
    const spanSec = Math.max(globalEnd - globalStart, 0.001);
    const activePhases = new Set(phaseOrder);

    function formatMs(ms) {
      return `${ms.toFixed(3)} ms`;
    }

    function formatAbs(ts) {
      return ts.toFixed(6);
    }

    function getVisibleRequests() {
      const query = searchBox.value.trim().toLowerCase();
      return allRequests.filter(req => !query || req.request_id.toLowerCase().includes(query));
    }

    function buildLegend() {
      legend.innerHTML = phaseOrder.map(phase => `
        <button type="button" class="legend-item ${activePhases.has(phase) ? '' : 'off'}" data-phase="${phase}">
          <span class="swatch" style="background:${phaseColors[phase]}"></span>
          <span>${phaseLabels[phase]}</span>
        </button>
      `).join('');
      legend.querySelectorAll('.legend-item').forEach(button => {
        button.addEventListener('click', () => {
          const phase = button.dataset.phase;
          if (activePhases.has(phase)) {
            activePhases.delete(phase);
          } else {
            activePhases.add(phase);
          }
          renderAll();
        });
      });
    }

    function chooseTickStep(pxPerSec) {
      const targetPx = 140;
      const choices = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10];
      for (const choice of choices) {
        if (choice * pxPerSec >= targetPx) return choice;
      }
      return choices[choices.length - 1];
    }

    function showTooltip(event, text) {
      tooltip.textContent = text;
      tooltip.classList.add('show');
      const pad = 16;
      const x = Math.min(event.clientX + 14, window.innerWidth - tooltip.offsetWidth - pad);
      const y = Math.max(event.clientY - tooltip.offsetHeight - 14, pad);
      tooltip.style.left = `${x}px`;
      tooltip.style.top = `${y}px`;
    }

    function hideTooltip() {
      tooltip.classList.remove('show');
    }

    function renderSummary(visibleRequests) {
      const chips = [
        `Requests: ${visibleRequests.length}/${allRequests.length}`,
        `Visible span: ${spanSec.toFixed(3)} s`,
        `Phases shown: ${activePhases.size}/${phaseOrder.length}`,
      ];
      summaryChips.innerHTML = chips.map(text => `<div class="chip">${text}</div>`).join('');
    }

    function renderTimeline() {
      const visibleRequests = getVisibleRequests();
      renderSummary(visibleRequests);
      const pxPerSec = Number(zoomRange.value);
      const tickStepSec = chooseTickStep(pxPerSec);
      const gridStepPx = tickStepSec * pxPerSec;
      const timelineWidth = Math.max(1100, spanSec * pxPerSec + 120);
      const visiblePhaseOrder = phaseOrder.filter(phase => activePhases.has(phase));

      const tickMarks = [];
      for (let value = globalStart; value <= globalEnd + 1e-9; value += tickStepSec) {
        const left = (value - globalStart) * pxPerSec;
        tickMarks.push(`<div class="tick" style="left:${left}px">${(value - globalStart).toFixed(3)}s</div>`);
      }

      const header = `
        <div class="tick-band" style="--grid-step:${gridStepPx}px">
          <div class="left">Request</div>
          <div class="right" style="width:${timelineWidth}px">${tickMarks.join('')}</div>
        </div>
      `;

      const rows = visibleRequests.map(req => {
        const meta = [
          `n=${req.sample_count}`,
          `repeat=${req.repeat_index}`,
          `batch_tokens=${req.batch_total_scheduled_tokens}`,
          `out=${req.output_tokens}`,
          `out/sample=${req.output_tokens_per_sample.toFixed(1)}`,
        ].join(' · ');

        const lanes = visiblePhaseOrder.map(phase => {
          const laneItems = req.phases.filter(item => item.phase === phase);
          const bars = laneItems.map(item => {
            const left = (item.start - globalStart) * pxPerSec;
            const width = Math.max((item.end - item.start) * pxPerSec, 2);
            const tip = [
              `${req.request_id}`,
              `${item.label}`,
              `start=${formatAbs(item.start)}`,
              `end=${formatAbs(item.end)}`,
              `dur=${formatMs(item.duration_ms)}`,
            ].join('\\n');
            return `<div class="bar" data-tip="${htmlEscape(tip)}" style="left:${left}px;width:${width}px;background:${phaseColors[phase]}"></div>`;
          }).join('');
          return `
            <div class="lane">
              <div class="lane-name">${phaseLabels[phase]}</div>
              ${bars}
            </div>
          `;
        }).join('');

        return `
          <div class="request-row">
            <div class="req-label">
              <div class="req-id">${req.request_id}</div>
              <div class="req-meta">${meta}</div>
            </div>
            <div class="req-lanes" style="width:${timelineWidth}px;--grid-step:${gridStepPx}px">${lanes}</div>
          </div>
        `;
      }).join('');

      mount.innerHTML = header + rows;
      mount.querySelectorAll('.bar').forEach(node => {
        node.addEventListener('mouseenter', event => showTooltip(event, htmlUnescape(node.dataset.tip)));
        node.addEventListener('mousemove', event => showTooltip(event, htmlUnescape(node.dataset.tip)));
        node.addEventListener('mouseleave', hideTooltip);
      });
    }

    function htmlEscape(value) {
      return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    function htmlUnescape(value) {
      return value
        .replaceAll('&quot;', '"')
        .replaceAll('&gt;', '>')
        .replaceAll('&lt;', '<')
        .replaceAll('&amp;', '&');
    }

    function renderAll() {
      buildLegend();
      renderTimeline();
    }

    zoomRange.addEventListener('input', renderTimeline);
    searchBox.addEventListener('input', renderTimeline);
    renderAll();
  </script>
</body>
</html>
"""


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_rows(json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(data).__name__}")
    return [row for row in data if isinstance(row, dict)]


def _build_phase_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for spec in PHASE_SPECS:
        start = _safe_float(row.get(spec["start_field"]))
        end = _safe_float(row.get(spec["end_field"]))
        if start <= 0.0 or end <= 0.0 or end < start:
            continue
        items.append(
            {
                "phase": spec["id"],
                "label": spec["label"],
                "start": start,
                "end": end,
                "duration_ms": (end - start) * 1000.0,
            }
        )
    return items


def build_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    requests = []
    global_start: float | None = None
    global_end: float | None = None

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("sample_count", 0) or 0),
            int(row.get("repeat_index", 0) or 0),
            str(row.get("request_id", "")),
        ),
    )

    for index, row in enumerate(sorted_rows):
        phases = _build_phase_items(row)
        if not phases:
            continue
        req_start = min(item["start"] for item in phases)
        req_end = max(item["end"] for item in phases)
        global_start = req_start if global_start is None else min(global_start, req_start)
        global_end = req_end if global_end is None else max(global_end, req_end)
        sample_count = int(row.get("sample_count", 0) or 0)
        output_tokens = int(row.get("output_tokens", 0) or 0)
        requests.append(
            {
                "request_id": str(row.get("request_id", f"row-{index}")),
                "index": index + 1,
                "request_start": req_start,
                "request_end": req_end,
                "sample_count": sample_count,
                "repeat_index": int(row.get("repeat_index", 0) or 0),
                "batch_total_scheduled_tokens": int(row.get("batch_total_scheduled_tokens", 0) or 0),
                "output_tokens": output_tokens,
                "output_tokens_per_sample": (output_tokens / sample_count) if sample_count > 0 else 0.0,
                "phases": phases,
            }
        )

    if not requests or global_start is None or global_end is None:
        raise ValueError("No usable timestamp phases found in metrics JSON")

    return {
        "requests": requests,
        "global_start": global_start,
        "global_end": global_end,
        "phase_order": [spec["id"] for spec in PHASE_SPECS],
        "phase_labels": {spec["id"]: spec["label"] for spec in PHASE_SPECS},
        "phase_colors": {spec["id"]: spec["color"] for spec in PHASE_SPECS},
    }


def build_html(payload: dict[str, Any], source_json: Path) -> str:
    return (
        HTML_TEMPLATE.replace("__TITLE__", html.escape(f"Two-stage Timeline - {source_json.name}"))
        .replace("__SOURCE__", html.escape(str(source_json)))
        .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an interactive timeline from two-stage metrics JSON.")
    parser.add_argument("json_path", type=Path, help="Path to two_stage_profile_vs_tokens_metrics.json")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <json>.timeline.html.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.json_path.exists():
        parser.error(f"metrics json does not exist: {args.json_path}")

    rows = _load_rows(args.json_path)
    payload = build_payload(rows)
    output_path = args.output or args.json_path.with_suffix(args.json_path.suffix + ".timeline.html")
    output_path.write_text(build_html(payload, args.json_path), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
