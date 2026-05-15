#!/usr/bin/env python3
"""Render an interactive request timeline HTML from vllm-omni metrics logs.

Example:
    python profiler/render_request_timeline.py \
        profiler/logs/64/async_omni_trace-gid_1778778376/svc_logs/svc_bs8.log
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

try:
    from profiler.timestamps.extract_request_timestamps import (
        PHASES,
        infer_batch_size,
        parse_log,
    )
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from timestamps.extract_request_timestamps import (
        PHASES,
        infer_batch_size,
        parse_log,
    )

PHASE_LABELS = {
    "stage0": "Stage 0",
    "kv_send": "KV Send",
    "kv_recv": "KV Receive",
    "diffusion": "Diffusion",
    "stage1": "Stage 1 Total",
}

REQ_BS_PATTERN = re.compile(r"\bbs(\d+)\b", re.IGNORECASE)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #f5efe3;
      --panel: rgba(255, 250, 242, 0.92);
      --panel-strong: #fffaf2;
      --ink: #1e1a17;
      --muted: #6b6258;
      --grid: rgba(78, 63, 47, 0.12);
      --grid-strong: rgba(78, 63, 47, 0.22);
      --accent: #af3a2f;
      --stage0: #2456d3;
      --kv-send: #e2742f;
      --kv-recv: #1f8f6b;
      --diffusion: #bf3277;
      --stage1: #3f3a96;
      --shadow: 0 18px 40px rgba(68, 42, 18, 0.12);
      --label-width: 270px;
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
      padding: 24px 24px 18px;
      position: sticky;
      top: 0;
      z-index: 12;
      backdrop-filter: blur(10px);
    }

    .hero-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }

    .hero-copy {
      min-width: 0;
      flex: 1;
    }

    .collapse-btn {
      border: 1px solid rgba(90, 68, 45, 0.14);
      background: rgba(255,255,255,0.88);
      color: var(--ink);
      border-radius: 999px;
      padding: 9px 14px;
      font: inherit;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
      flex-shrink: 0;
    }

    .collapse-btn:hover {
      border-color: rgba(175, 58, 47, 0.36);
      color: var(--accent);
    }

    .hero-body {
      margin-top: 18px;
    }

    .hero.collapsed .hero-body {
      display: none;
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
      font-family: "IBM Plex Serif", Georgia, serif;
      font-size: 34px;
      line-height: 1.05;
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
      color: var(--ink);
    }

    .zoom-dock {
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 30;
      margin: 0;
      touch-action: none;
    }

    .zoom-panel {
      width: min(320px, calc(100vw - 34px));
      background: rgba(255, 250, 242, 0.96);
      border: 1px solid rgba(90, 68, 45, 0.12);
      border-radius: 18px;
      box-shadow: 0 14px 32px rgba(68, 42, 18, 0.16);
      padding: 12px 14px;
      backdrop-filter: blur(12px);
      pointer-events: auto;
      user-select: none;
    }

    .zoom-handle {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      cursor: grab;
      margin: -2px -2px 8px;
      padding: 2px;
    }

    .zoom-handle:active {
      cursor: grabbing;
    }

    .zoom-grip {
      display: inline-flex;
      align-items: center;
      gap: 3px;
    }

    .zoom-grip span {
      width: 5px;
      height: 5px;
      border-radius: 999px;
      background: rgba(90, 68, 45, 0.35);
      display: block;
    }

    .zoom-panel.dragging {
      box-shadow: 0 18px 38px rgba(68, 42, 18, 0.22);
    }

    .zoom-panel .panel-title {
      font-size: 11px;
    }

    .zoom-panel .panel-note {
      margin-top: 6px;
      font-size: 11px;
      line-height: 1.4;
    }

    .controls {
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 14px;
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid rgba(90, 68, 45, 0.12);
    }

    .panel {
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(90, 68, 45, 0.12);
      border-radius: 18px;
      padding: 14px;
      min-height: 100px;
    }

    .span-3 { grid-column: span 3; }
    .span-4 { grid-column: span 4; }
    .span-5 { grid-column: span 5; }
    .span-6 { grid-column: span 6; }
    .span-12 { grid-column: span 12; }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }

    .panel-title {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 700;
    }

    .panel-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .mini-btn {
      border: 1px solid rgba(90, 68, 45, 0.14);
      background: rgba(255,255,255,0.88);
      color: var(--ink);
      border-radius: 999px;
      padding: 5px 10px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }

    .mini-btn:hover {
      border-color: rgba(175, 58, 47, 0.36);
      color: var(--accent);
    }

    .filter-cloud {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: flex-start;
    }

    .filter-cloud.scroll {
      max-height: 170px;
      overflow: auto;
      padding-right: 4px;
    }

    .filter-pill {
      border: 1px solid rgba(90, 68, 45, 0.14);
      background: rgba(255,255,255,0.88);
      color: var(--ink);
      border-radius: 999px;
      padding: 7px 11px;
      font: inherit;
      font-size: 12px;
      line-height: 1.1;
      cursor: pointer;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }

    .filter-pill:hover {
      transform: translateY(-1px);
      border-color: rgba(175, 58, 47, 0.36);
    }

    .filter-pill.active {
      background: rgba(175, 58, 47, 0.12);
      border-color: rgba(175, 58, 47, 0.28);
      color: var(--accent);
      font-weight: 700;
    }

    .filter-pill.phase.active {
      color: #fff;
      border-color: transparent;
      box-shadow: 0 8px 18px rgba(0,0,0,0.12);
    }

    .filter-pill.muted {
      opacity: 0.5;
    }

    .filter-pill small {
      opacity: 0.74;
      font-size: 11px;
    }

    .panel-note {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }

    .panel-note-list {
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }

    .panel-note-list div + div {
      margin-top: 6px;
    }

    .control label {
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 700;
      margin-bottom: 6px;
    }

    .control input[type=text],
    .control input[type=range] {
      width: 100%;
    }

    .control input[type=text] {
      appearance: none;
      border: 1px solid rgba(90, 68, 45, 0.16);
      background: rgba(255,255,255,0.92);
      border-radius: 12px;
      padding: 10px 12px;
      color: var(--ink);
      font: inherit;
    }

    .control input[type=range] {
      accent-color: var(--accent);
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      margin-top: 16px;
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      transition: opacity 120ms ease;
    }

    .legend-item.dimmed {
      opacity: 0.4;
    }

    .swatch {
      width: 14px;
      height: 14px;
      border-radius: 4px;
      box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
    }

    .timeline-card {
      margin-top: 22px;
      border-radius: 24px;
      overflow: hidden;
      background: var(--panel);
      border: 1px solid rgba(90, 68, 45, 0.12);
      box-shadow: var(--shadow);
    }

    .timeline-scroll {
      overflow: auto;
      max-height: calc(100vh - 340px);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.4), rgba(255,255,255,0.15)),
        repeating-linear-gradient(
          180deg,
          transparent 0,
          transparent 44px,
          rgba(90,68,45,0.03) 44px,
          rgba(90,68,45,0.03) 88px
        );
    }

    .shell.hero-collapsed .timeline-scroll {
      max-height: calc(100vh - 210px);
    }

    .timeline-layout {
      display: grid;
      grid-template-columns: var(--label-width) 1fr;
      min-width: calc(var(--label-width) + 1000px);
    }

    .axis-labels,
    .axis-track {
      position: sticky;
      top: 0;
      z-index: 4;
      background: rgba(255, 249, 240, 0.95);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid rgba(90, 68, 45, 0.12);
    }

    .axis-labels {
      padding: 16px 16px 14px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }

    .axis-track {
      height: 62px;
    }

    .rows {
      position: relative;
    }

    .label-col {
      position: sticky;
      left: 0;
      z-index: 3;
      background: rgba(255, 249, 240, 0.92);
      border-right: 1px solid rgba(90, 68, 45, 0.12);
    }

    .request-label {
      padding: 14px 16px;
      border-bottom: 1px solid rgba(90, 68, 45, 0.08);
      min-height: 112px;
    }

    .request-title {
      font-size: 13px;
      font-weight: 700;
      line-height: 1.35;
      word-break: break-all;
    }

    .request-meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }

    .track-col {
      position: relative;
      background-image:
        linear-gradient(180deg, rgba(90,68,45,0.05), rgba(90,68,45,0.05)),
        linear-gradient(180deg, transparent, transparent);
      background-size: 100% 1px, 100% 100%;
      background-repeat: no-repeat;
      background-position: 0 100%, 0 0;
    }

    .request-track {
      position: relative;
      border-bottom: 1px solid rgba(90, 68, 45, 0.08);
      min-height: 112px;
    }

    .lane-tag {
      position: absolute;
      left: 10px;
      font-size: 10px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: rgba(17, 17, 17, 0.62);
      pointer-events: none;
      font-weight: 700;
    }

    .bar {
      position: absolute;
      height: var(--lane-height);
      border-radius: 999px;
      box-shadow: 0 5px 10px rgba(0,0,0,0.12);
      border: 1px solid rgba(255,255,255,0.5);
      cursor: pointer;
      opacity: 0.92;
      transition: transform 120ms ease, opacity 120ms ease;
    }

    .bar:hover {
      opacity: 1;
      transform: scaleY(1.18);
      z-index: 5;
    }

    .group-band {
      position: absolute;
      left: 0;
      right: 0;
      background: rgba(175, 58, 47, 0.04);
      border-top: 1px dashed rgba(175, 58, 47, 0.12);
      border-bottom: 1px dashed rgba(175, 58, 47, 0.12);
      pointer-events: none;
    }

    .axis-tick {
      position: absolute;
      top: 0;
      bottom: 0;
      width: 1px;
      background: var(--grid);
    }

    .axis-tick.major {
      background: var(--grid-strong);
    }

    .axis-text {
      position: absolute;
      top: 10px;
      transform: translateX(-50%);
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
    }

    .tooltip {
      position: fixed;
      z-index: 50;
      pointer-events: none;
      min-width: 220px;
      max-width: 320px;
      background: rgba(21, 18, 15, 0.92);
      color: #fff8ef;
      border-radius: 16px;
      padding: 12px 14px;
      box-shadow: 0 16px 30px rgba(0,0,0,0.22);
      opacity: 0;
      transform: translateY(8px);
      transition: opacity 120ms ease, transform 120ms ease;
    }

    .tooltip.visible {
      opacity: 1;
      transform: translateY(0);
    }

    .tooltip-title {
      font-weight: 700;
      margin-bottom: 6px;
      line-height: 1.35;
      word-break: break-word;
    }

    .tooltip-line {
      font-size: 12px;
      color: rgba(255,248,239,0.85);
      line-height: 1.5;
    }

    .empty {
      padding: 40px 32px;
      color: var(--muted);
      text-align: center;
      font-size: 14px;
    }

    @media (max-width: 1200px) {
      .span-3, .span-4, .span-5, .span-6 { grid-column: span 6; }
    }

    @media (max-width: 900px) {
      .shell { padding: 14px; }
      .hero { border-radius: 22px; padding: 18px 16px 16px; }
      h1 { font-size: 28px; }
      .controls { grid-template-columns: 1fr; }
      .span-3, .span-4, .span-5, .span-6, .span-12 { grid-column: span 1; }
      .timeline-layout { grid-template-columns: 210px 1fr; }
      .zoom-dock {
        right: 10px;
        bottom: 10px;
        left: auto;
        top: auto;
      }
      .zoom-panel { width: min(280px, calc(100vw - 20px)); }
      :root { --label-width: 210px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero" id="heroPanel">
      <div class="hero-top">
        <div class="hero-copy">
          <div class="eyebrow">Profiler Timeline</div>
          <h1>Request Stage Timeline</h1>
          <div class="sub">Source log: __SOURCE__</div>
        </div>
        <button class="collapse-btn" id="collapseHeroBtn" type="button" aria-expanded="true">Collapse Filters</button>
      </div>
      <div class="hero-body" id="heroBody">
        <div class="chip-row" id="summaryChips"></div>

        <div class="controls">
        <div class="panel span-4">
          <div class="panel-head">
            <div class="panel-title">Groups</div>
            <div class="panel-actions">
              <button class="mini-btn" id="groupsAllBtn" type="button">All</button>
              <button class="mini-btn" id="groupsNoneBtn" type="button">None</button>
            </div>
          </div>
          <div class="filter-cloud" id="groupFilters"></div>
          <div class="panel-note">Choose which groups to show. Changing groups resets request selection to all requests in those groups.</div>
        </div>

        <div class="panel span-4">
          <div class="panel-head">
            <div class="panel-title">Stages</div>
            <div class="panel-actions">
              <button class="mini-btn" id="phasesAllBtn" type="button">All</button>
              <button class="mini-btn" id="phasesNoneBtn" type="button">None</button>
            </div>
          </div>
          <div class="filter-cloud" id="phaseFilters"></div>
          <div class="panel-note">Toggle stage bars on the timeline. Hidden stages also disappear from each request lane.</div>
          <div class="panel-note-list">
            <div><strong>Stage 0</strong>: VLM submits the request until it finishes, before KV send.</div>
            <div><strong>KV Send</strong>: time spent sending KV in Stage 0.</div>
            <div><strong>KV Receive</strong>: time spent receiving KV in Stage 1.</div>
            <div><strong>Diffusion</strong>: actual diffusion execution time.</div>
            <div><strong>Stage 1 Total</strong>: from diffusion request submission to finish, including waiting time.</div>
          </div>
        </div>

        <div class="panel span-4">
          <div class="panel-head">
            <div class="panel-title">Requests</div>
            <div class="panel-actions">
              <button class="mini-btn" id="requestsAllBtn" type="button">All</button>
              <button class="mini-btn" id="requestsNoneBtn" type="button">None</button>
            </div>
          </div>
          <div class="control">
            <label for="requestPickerSearch">Filter Request Options</label>
            <input id="requestPickerSearch" type="text" placeholder="search request ids for selection" />
          </div>
          <div class="filter-cloud scroll" id="requestFilters" style="margin-top:10px"></div>
          <div class="panel-note">Default is all requests inside the selected groups.</div>
        </div>

        <div class="panel span-6">
          <div class="panel-title">Display Filter</div>
          <div class="control" style="margin-top:10px">
            <label for="searchBox">Quick Search On Visible Requests</label>
            <input id="searchBox" type="text" placeholder="filter the rendered timeline by request id" />
          </div>
          <div class="panel-note">This only filters what is currently drawn. It does not change the request selection above.</div>
        </div>

        </div>

        <div class="legend" id="legend"></div>
      </div>
    </section>

    <section class="zoom-dock" id="zoomDock">
      <div class="zoom-panel" id="zoomPanel">
        <div class="zoom-handle" id="zoomHandle">
          <div class="panel-title">Zoom</div>
          <div class="zoom-grip" aria-hidden="true">
            <span></span><span></span><span></span>
          </div>
        </div>
        <div class="control" style="margin-top:8px">
          <label for="zoomRange">Pixels Per Second</label>
          <input id="zoomRange" type="range" min="6" max="80" step="1" value="18" />
        </div>
        <div class="panel-note">Drag this panel to move it. It stays available for fast timeline scaling.</div>
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
    const palette = {
      stage0: 'var(--stage0)',
      kv_send: 'var(--kv-send)',
      kv_recv: 'var(--kv-recv)',
      diffusion: 'var(--diffusion)',
      stage1: 'var(--stage1)',
    };

    const payload = JSON.parse(document.getElementById('payload').textContent);
    const tooltip = document.getElementById('tooltip');
    const legend = document.getElementById('legend');
    const mount = document.getElementById('timelineMount');
    const shell = document.querySelector('.shell');
    const heroPanel = document.getElementById('heroPanel');
    const collapseHeroBtn = document.getElementById('collapseHeroBtn');
    const zoomDock = document.getElementById('zoomDock');
    const zoomPanel = document.getElementById('zoomPanel');
    const zoomHandle = document.getElementById('zoomHandle');
    const summaryChips = document.getElementById('summaryChips');
    const searchBox = document.getElementById('searchBox');
    const zoomRange = document.getElementById('zoomRange');
    const requestPickerSearch = document.getElementById('requestPickerSearch');
    const groupFilters = document.getElementById('groupFilters');
    const phaseFilters = document.getElementById('phaseFilters');
    const requestFilters = document.getElementById('requestFilters');

    const laneOrder = payload.phase_order;
    const phaseLabels = payload.phase_labels;
    const allRequests = payload.requests;
    const globalStart = payload.global_start;
    const globalEnd = payload.global_end;
    const spanSec = Math.max((globalEnd - globalStart), 0.001);
    const groups = [...new Set(allRequests.map(r => r.group_id).filter(v => v !== null))].sort((a, b) => a - b);

    const state = {
      selectedGroups: new Set(groups.map(String)),
      selectedPhases: new Set(laneOrder),
      selectedRequests: new Set(allRequests.map(r => r.request_id)),
    };

    function formatSec(sec) {
      return `${sec.toFixed(3)}s`;
    }

    function formatAbs(ts) {
      return ts.toFixed(3);
    }

    function getRequestsForSelectedGroups() {
      if (!groups.length) return allRequests.slice();
      if (!state.selectedGroups.size) return [];
      return allRequests.filter(req => req.group_id === null || state.selectedGroups.has(String(req.group_id)));
    }

    function resetRequestSelectionToVisibleGroups() {
      const reqs = getRequestsForSelectedGroups();
      state.selectedRequests = new Set(reqs.map(req => req.request_id));
    }

    function getVisiblePhaseOrder() {
      return laneOrder.filter(phase => state.selectedPhases.has(phase));
    }

    function buildSummary(visibleRequests, activePhases) {
      const chips = [
        `Total requests: ${allRequests.length}`,
        `Rendered requests: ${visibleRequests.length}`,
        `Visible span: ${spanSec.toFixed(3)}s`,
        `Stages shown: ${activePhases.length}`,
      ];
      if (payload.batch_size) chips.push(`Batch size: ${payload.batch_size}`);
      if (groups.length) chips.push(`Groups selected: ${state.selectedGroups.size}/${groups.length}`);
      summaryChips.innerHTML = chips.map(text => `<div class="chip">${text}</div>`).join('');
    }

    function buildLegend() {
      legend.innerHTML = laneOrder.map(phase => {
        const active = state.selectedPhases.has(phase);
        return `
          <div class="legend-item ${active ? '' : 'dimmed'}">
            <span class="swatch" style="background:${palette[phase]}"></span>
            <span>${phaseLabels[phase]}</span>
          </div>
        `;
      }).join('');
    }

    function renderPills(container, items, selectedSet, options = {}) {
      const {
        labelFor = item => String(item),
        valueFor = item => String(item),
        className = '',
        colorFor = null,
        extraFor = null,
        mutedSet = null,
      } = options;

      container.innerHTML = items.map(item => {
        const value = valueFor(item);
        const active = selectedSet.has(value);
        const muted = mutedSet ? mutedSet.has(value) : false;
        const style = active && colorFor ? ` style="background:${colorFor(item)}"` : '';
        const extra = extraFor ? extraFor(item) : '';
        return `
          <button
            type="button"
            class="filter-pill ${className} ${active ? 'active' : ''} ${muted ? 'muted' : ''}"
            data-value="${value}"${style}
          >
            ${labelFor(item)}${extra}
          </button>
        `;
      }).join('');
    }

    function renderGroupFilters() {
      if (!groups.length) {
        groupFilters.innerHTML = '<div class="panel-note">No group metadata available.</div>';
        return;
      }
      renderPills(groupFilters, groups, state.selectedGroups, {
        labelFor: groupId => `Group ${groupId}`,
      });
      groupFilters.querySelectorAll('.filter-pill').forEach(button => {
        button.addEventListener('click', () => {
          const value = button.dataset.value;
          if (state.selectedGroups.has(value)) {
            state.selectedGroups.delete(value);
          } else {
            state.selectedGroups.add(value);
          }
          resetRequestSelectionToVisibleGroups();
          renderAll();
        });
      });
    }

    function renderPhaseFilters() {
      renderPills(phaseFilters, laneOrder, state.selectedPhases, {
        className: 'phase',
        labelFor: phase => phaseLabels[phase],
        colorFor: phase => palette[phase],
      });
      phaseFilters.querySelectorAll('.filter-pill').forEach(button => {
        button.addEventListener('click', () => {
          const value = button.dataset.value;
          if (state.selectedPhases.has(value)) {
            state.selectedPhases.delete(value);
          } else {
            state.selectedPhases.add(value);
          }
          renderAll();
        });
      });
    }

    function renderRequestFilters() {
      const query = requestPickerSearch.value.trim().toLowerCase();
      const availableRequests = getRequestsForSelectedGroups();
      const filteredRequests = availableRequests.filter(req => {
        if (!query) return true;
        return req.request_id.toLowerCase().includes(query);
      });

      if (!availableRequests.length) {
        requestFilters.innerHTML = '<div class="panel-note">No requests in the selected groups.</div>';
        return;
      }

      renderPills(requestFilters, filteredRequests, state.selectedRequests, {
        labelFor: req => req.request_id,
        valueFor: req => req.request_id,
        extraFor: req => req.group_id ? ` <small>g${req.group_id}</small>` : '',
      });

      if (!filteredRequests.length) {
        requestFilters.innerHTML = '<div class="panel-note">No request options match the search above.</div>';
        return;
      }

      requestFilters.querySelectorAll('.filter-pill').forEach(button => {
        button.addEventListener('click', () => {
          const value = button.dataset.value;
          if (state.selectedRequests.has(value)) {
            state.selectedRequests.delete(value);
          } else {
            state.selectedRequests.add(value);
          }
          renderAll({ skipRequestFilterRender: true });
        });
      });
    }

    function filteredRequests() {
      const query = searchBox.value.trim().toLowerCase();
      const base = getRequestsForSelectedGroups().filter(req => state.selectedRequests.has(req.request_id));
      return base.filter(req => {
        if (query && !req.request_id.toLowerCase().includes(query)) return false;
        return true;
      });
    }

    function chooseTickStep(pxPerSec) {
      const targetPx = 120;
      const roughSec = targetPx / pxPerSec;
      const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60];
      for (const step of steps) {
        if (step >= roughSec) return step;
      }
      return 120;
    }

    function showTooltip(event, req, phase) {
      const startOffset = phase.start - globalStart;
      const endOffset = phase.end - globalStart;
      tooltip.innerHTML = `
        <div class="tooltip-title">${req.request_id} · ${phase.label}</div>
        <div class="tooltip-line">group=${req.group_id ?? '-'} pos=${req.group_pos ?? '-'} duration=${phase.duration_ms.toFixed(2)} ms</div>
        <div class="tooltip-line">start=${formatAbs(phase.start)} (${formatSec(startOffset)})</div>
        <div class="tooltip-line">end=${formatAbs(phase.end)} (${formatSec(endOffset)})</div>
      `;
      tooltip.classList.add('visible');
      moveTooltip(event);
    }

    function moveTooltip(event) {
      const pad = 18;
      const x = Math.min(window.innerWidth - tooltip.offsetWidth - pad, event.clientX + 16);
      const y = Math.min(window.innerHeight - tooltip.offsetHeight - pad, event.clientY + 16);
      tooltip.style.left = `${Math.max(pad, x)}px`;
      tooltip.style.top = `${Math.max(pad, y)}px`;
    }

    function hideTooltip() {
      tooltip.classList.remove('visible');
    }

    function clamp(value, min, max) {
      return Math.min(Math.max(value, min), max);
    }

    function attachZoomDrag() {
      let dragging = false;
      let offsetX = 0;
      let offsetY = 0;

      function dockWidth() {
        return zoomDock.offsetWidth || 320;
      }

      function dockHeight() {
        return zoomDock.offsetHeight || 120;
      }

      function setDockPosition(left, top) {
        const maxLeft = Math.max(8, window.innerWidth - dockWidth() - 8);
        const maxTop = Math.max(8, window.innerHeight - dockHeight() - 8);
        const safeLeft = clamp(left, 8, maxLeft);
        const safeTop = clamp(top, 8, maxTop);
        zoomDock.style.left = `${safeLeft}px`;
        zoomDock.style.top = `${safeTop}px`;
        zoomDock.style.right = 'auto';
        zoomDock.style.bottom = 'auto';
      }

      function onPointerMove(event) {
        if (!dragging) return;
        setDockPosition(event.clientX - offsetX, event.clientY - offsetY);
      }

      function onPointerUp() {
        if (!dragging) return;
        dragging = false;
        zoomPanel.classList.remove('dragging');
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
      }

      zoomHandle.addEventListener('pointerdown', event => {
        if (event.button !== 0) return;
        const rect = zoomDock.getBoundingClientRect();
        dragging = true;
        offsetX = event.clientX - rect.left;
        offsetY = event.clientY - rect.top;
        zoomPanel.classList.add('dragging');
        zoomHandle.setPointerCapture?.(event.pointerId);
        window.addEventListener('pointermove', onPointerMove);
        window.addEventListener('pointerup', onPointerUp);
        event.preventDefault();
      });

      window.addEventListener('resize', () => {
        const rect = zoomDock.getBoundingClientRect();
        setDockPosition(rect.left, rect.top);
      });
    }

    function renderTimeline() {
      const pxPerSec = Number(zoomRange.value);
      const visibleRequests = filteredRequests();
      const visiblePhases = getVisiblePhaseOrder();

      buildSummary(visibleRequests, visiblePhases);
      buildLegend();

      if (!visiblePhases.length) {
        mount.innerHTML = '<div class="empty">No stages selected. Pick at least one stage to render the timeline.</div>';
        return;
      }

      if (!visibleRequests.length) {
        mount.innerHTML = '<div class="empty">No requests match the current group, request, and search filters.</div>';
        return;
      }

      const laneCount = visiblePhases.length;
      const rowPaddingTop = 14;
      const rowHeight = rowPaddingTop + laneCount * 14 + (laneCount - 1) * 8 + 18;
      const timelineWidth = Math.max(1100, Math.ceil(spanSec * pxPerSec) + 80);
      const tickStepSec = chooseTickStep(pxPerSec);
      const majorEvery = tickStepSec >= 1 ? 5 : 4;

      let axisTicks = '';
      let tickIndex = 0;
      for (let t = 0; t <= spanSec + 1e-9; t += tickStepSec) {
        const left = t * pxPerSec;
        const major = tickIndex % majorEvery === 0;
        axisTicks += `
          <div class="axis-tick ${major ? 'major' : ''}" style="left:${left}px"></div>
          <div class="axis-text" style="left:${left}px">${formatSec(t)}</div>
        `;
        tickIndex += 1;
      }

      const labelHtml = visibleRequests.map(req => `
        <div class="request-label" style="height:${rowHeight}px">
          <div class="request-title">${req.request_id}</div>
          <div class="request-meta">
            index=${req.index}
            ${req.group_id ? `<br>group=${req.group_id} pos=${req.group_pos}` : ''}
            <br>window=${formatSec(req.request_start - globalStart)} -> ${formatSec(req.request_end - globalStart)}
          </div>
        </div>
      `).join('');

      let tracksHtml = '';
      let currentGroup = null;
      visibleRequests.forEach((req, rowIndex) => {
        const groupHeader = req.group_id !== null && req.group_id !== currentGroup;
        currentGroup = req.group_id;
        let bars = '';
        let tags = '';

        visiblePhases.forEach((phaseName, laneIndex) => {
          const phase = req.phases.find(item => item.phase === phaseName);
          const top = rowPaddingTop + laneIndex * (14 + 8);
          tags += `<div class="lane-tag" style="top:${top - 1}px">${phaseLabels[phaseName]}</div>`;
          if (!phase) return;
          const left = (phase.start - globalStart) * pxPerSec;
          const width = Math.max((phase.end - phase.start) * pxPerSec, 2);
          bars += `<div class="bar" data-request='${JSON.stringify(req.request_id)}' data-phase='${phaseName}' style="left:${left}px;top:${top}px;width:${width}px;background:${palette[phaseName]}"></div>`;
        });

        const groupBand = groupHeader && req.group_id !== null
          ? `<div class="group-band" style="top:0;height:${rowHeight}px"></div>`
          : '';

        tracksHtml += `
          <div class="request-track" style="height:${rowHeight}px;width:${timelineWidth}px">
            ${groupBand}
            ${tags}
            ${bars}
          </div>
        `;
      });

      mount.innerHTML = `
        <div class="timeline-layout">
          <div class="axis-labels">Requests</div>
          <div class="axis-track" style="width:${timelineWidth}px">${axisTicks}</div>
          <div class="label-col rows">${labelHtml}</div>
          <div class="track-col rows">${tracksHtml}</div>
        </div>
      `;

      mount.querySelectorAll('.bar').forEach(bar => {
        const reqId = JSON.parse(bar.dataset.request);
        const phaseName = bar.dataset.phase;
        const req = visibleRequests.find(item => item.request_id === reqId);
        const phase = req.phases.find(item => item.phase === phaseName);
        bar.addEventListener('mouseenter', event => showTooltip(event, req, phase));
        bar.addEventListener('mousemove', moveTooltip);
        bar.addEventListener('mouseleave', hideTooltip);
      });
    }

    function renderAll(options = {}) {
      renderGroupFilters();
      renderPhaseFilters();
      if (!options.skipRequestFilterRender) {
        renderRequestFilters();
      }
      renderTimeline();
    }

    document.getElementById('groupsAllBtn').addEventListener('click', () => {
      state.selectedGroups = new Set(groups.map(String));
      resetRequestSelectionToVisibleGroups();
      renderAll();
    });

    document.getElementById('groupsNoneBtn').addEventListener('click', () => {
      state.selectedGroups = new Set();
      state.selectedRequests = new Set();
      renderAll();
    });

    document.getElementById('phasesAllBtn').addEventListener('click', () => {
      state.selectedPhases = new Set(laneOrder);
      renderAll();
    });

    document.getElementById('phasesNoneBtn').addEventListener('click', () => {
      state.selectedPhases = new Set();
      renderAll();
    });

    document.getElementById('requestsAllBtn').addEventListener('click', () => {
      state.selectedRequests = new Set(getRequestsForSelectedGroups().map(req => req.request_id));
      renderAll();
    });

    document.getElementById('requestsNoneBtn').addEventListener('click', () => {
      state.selectedRequests = new Set();
      renderAll();
    });

    collapseHeroBtn.addEventListener('click', () => {
      const collapsed = heroPanel.classList.toggle('collapsed');
      shell.classList.toggle('hero-collapsed', collapsed);
      collapseHeroBtn.textContent = collapsed ? 'Expand Filters' : 'Collapse Filters';
      collapseHeroBtn.setAttribute('aria-expanded', String(!collapsed));
    });

    requestPickerSearch.addEventListener('input', () => renderAll());
    searchBox.addEventListener('input', () => renderTimeline());
    zoomRange.addEventListener('input', () => renderTimeline());

    attachZoomDrag();
    renderAll();
  </script>
</body>
</html>
"""


def infer_batch_size_from_requests(order: list[str]) -> int | None:
    found: set[int] = set()
    for req_id in order:
        match = REQ_BS_PATTERN.search(req_id)
        if match:
            found.add(int(match.group(1)))
    if len(found) == 1:
        return next(iter(found))
    return None



def build_requests_payload(
    order: list[str],
    rows: dict[str, dict[str, dict[str, str]]],
    batch_size: int | None,
) -> dict:
    requests = []
    global_start = None
    global_end = None

    for index, req_id in enumerate(order):
        phase_items = []
        req_start = None
        req_end = None

        for phase in PHASES:
            phase_data = rows[req_id].get(phase)
            if not phase_data:
                continue
            start = float(phase_data["start"])
            end = float(phase_data["now"])
            duration_ms = float(
                phase_data.get("gen_time_ms", "") or phase_data.get("time_ms", "0")
            )
            phase_items.append(
                {
                    "phase": phase,
                    "label": PHASE_LABELS[phase],
                    "start": start,
                    "end": end,
                    "duration_ms": duration_ms,
                }
            )
            req_start = start if req_start is None else min(req_start, start)
            req_end = end if req_end is None else max(req_end, end)

        if not phase_items:
            continue

        global_start = req_start if global_start is None else min(global_start, req_start)
        global_end = req_end if global_end is None else max(global_end, req_end)

        requests.append(
            {
                "request_id": req_id,
                "index": index + 1,
                "group_id": (index // batch_size + 1) if batch_size else None,
                "group_pos": (index % batch_size + 1) if batch_size else None,
                "request_start": req_start,
                "request_end": req_end,
                "phases": phase_items,
            }
        )

    return {
        "requests": requests,
        "batch_size": batch_size,
        "global_start": global_start,
        "global_end": global_end,
        "phase_order": PHASES,
        "phase_labels": PHASE_LABELS,
    }



def build_html(payload: dict, source_log: Path) -> str:
    return (
        HTML_TEMPLATE.replace("__TITLE__", html.escape(f"Request Timeline - {source_log.name}"))
        .replace("__SOURCE__", html.escape(str(source_log)))
        .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
    )



def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render an interactive HTML timeline for request stage timings."
    )
    parser.add_argument("logfile", type=Path, help="Path to the input log file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output HTML path. Defaults to <logfile>.timeline.html.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        help="Override batch size. If omitted, the script tries the file path first, then request ids.",
    )
    return parser



def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.logfile.exists():
        parser.error(f"log file does not exist: {args.logfile}")

    order, rows = parse_log(args.logfile)
    batch_size = args.bs
    if batch_size is None:
        batch_size = infer_batch_size(args.logfile)
    if batch_size is None:
        batch_size = infer_batch_size_from_requests(order)
    if batch_size is not None and batch_size <= 0:
        parser.error(f"batch size must be positive, got: {batch_size}")

    payload = build_requests_payload(order, rows, batch_size)
    if not payload["requests"]:
        parser.error("no matching request metrics found in the log")

    output = args.output or args.logfile.with_suffix(args.logfile.suffix + ".timeline.html")
    output.write_text(build_html(payload, args.logfile), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
