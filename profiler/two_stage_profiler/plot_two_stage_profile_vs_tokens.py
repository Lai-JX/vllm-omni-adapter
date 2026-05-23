from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METRIC_SPECS = [
    {
        "y_key": "embed_multimodal_ms",
        "x_key": "image_count",
        "x_label": "Image count",
        "y_label": "embed_multimodal_ms",
        "title": "Image count vs embed_multimodal_ms",
        "color": "#1f77b4",
        "annotate_output_tokens": False,
    },
    {
        "y_key": "forward_ms",
        "x_key": "batch_total_scheduled_tokens",
        "x_label": "batch_total_scheduled_tokens",
        "y_label": "forward_ms",
        "title": "batch_total_scheduled_tokens vs forward_ms",
        "color": "#ff7f0e",
        "annotate_output_tokens": False,
    },
    {
        "y_key": "subsequent_decode_ms",
        "x_key": "batch_total_scheduled_tokens",
        "x_label": "batch_total_scheduled_tokens",
        "y_label": "subsequent_decode_ms / (output_tokens / sample_count)",
        "title": "batch_total_scheduled_tokens vs subsequent single decode ms",
        "color": "#2ca02c",
        "annotate_output_tokens": True,
        "normalize_by_output_tokens_per_sample": True,
    },
    {
        "y_key": "kv_transfer_ms",
        "x_key": "batch_total_scheduled_tokens",
        "x_label": "batch_total_scheduled_tokens",
        "y_label": "kv_transfer_ms",
        "title": "batch_total_scheduled_tokens vs kv_transfer_ms",
        "color": "#d62728",
        "annotate_output_tokens": False,
    },
    {
        "y_key": "s1_diffusion_ms",
        "x_key": "batch_total_scheduled_tokens",
        "x_label": "batch_total_scheduled_tokens",
        "y_label": "s1_diffusion_ms",
        "title": "batch_total_scheduled_tokens vs s1_diffusion_ms",
        "color": "#9467bd",
        "annotate_output_tokens": False,
    },
]


def _load_rows(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(data).__name__}")
    return [row for row in data if isinstance(row, dict)]



def _default_output_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}_plots.png")



def _mean(values: list[float]) -> float:
    return sum(values) / len(values)



def _std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)



def _format_sample_counts(sample_counts: list[int]) -> str:
    unique_counts = sorted(set(sample_counts))
    if len(unique_counts) == 1:
        return f"n={unique_counts[0]}"
    return "n=" + "/".join(str(value) for value in unique_counts)



def _format_output_tokens(output_tokens: list[int], sample_counts: list[int], *, per_sample: bool = False) -> str:
    if per_sample:
        normalized_tokens = []
        for output_token, sample_count in zip(output_tokens, sample_counts):
            if sample_count > 0:
                normalized_tokens.append(output_token / sample_count)
            else:
                normalized_tokens.append(0.0)
        unique_tokens = sorted(set(normalized_tokens))
        if len(unique_tokens) == 1:
            return f"out={unique_tokens[0]:.1f}"
        return "out=" + "/".join(f"{value:.1f}" for value in unique_tokens)

    unique_tokens = sorted(set(output_tokens))
    if len(unique_tokens) == 1:
        return f"out={unique_tokens[0]}"
    return "out=" + "/".join(str(value) for value in unique_tokens)



def _aggregate_points(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    normalize_by_output_tokens_per_sample: bool = False,
) -> tuple[list[float], list[float], list[float], list[list[int]], list[list[int]], list[tuple[float, float]]]:
    grouped: dict[float, list[tuple[float, int, int]]] = defaultdict(list)
    for row in rows:
        x_value = float(row.get(x_key, 0) or 0)
        y_value = float(row.get(y_key, 0.0) or 0.0)
        sample_count = int(row.get("sample_count", 0) or 0)
        output_tokens = int(row.get("output_tokens", 0) or 0)
        if normalize_by_output_tokens_per_sample:
            decode_count = output_tokens / sample_count if sample_count > 0 else 0.0
            y_value = y_value / decode_count if decode_count > 0 else 0.0
        grouped[x_value].append((y_value, sample_count, output_tokens))

    xs = sorted(grouped)
    means: list[float] = []
    stds: list[float] = []
    sample_count_groups: list[list[int]] = []
    output_token_groups: list[list[int]] = []
    raw_points: list[tuple[float, float]] = []
    for x_value in xs:
        triples = grouped[x_value]
        y_values = [y for y, _, _ in triples]
        mean_value = _mean(y_values)
        means.append(mean_value)
        stds.append(_std(y_values, mean_value))
        sample_count_groups.append([sample_count for _, sample_count, _ in triples])
        output_token_groups.append([output_tokens for _, _, output_tokens in triples])
        raw_points.extend((x_value, y) for y in y_values)
    return xs, means, stds, sample_count_groups, output_token_groups, raw_points



def _plot_metric(
    ax,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    color: str,
    annotate_output_tokens: bool,
    normalize_by_output_tokens_per_sample: bool = False,
) -> None:
    xs, means, stds, sample_count_groups, output_token_groups, raw_points = _aggregate_points(
        rows,
        x_key=x_key,
        y_key=y_key,
        normalize_by_output_tokens_per_sample=normalize_by_output_tokens_per_sample,
    )
    if raw_points:
        raw_xs = [point[0] for point in raw_points]
        raw_ys = [point[1] for point in raw_points]
        ax.scatter(raw_xs, raw_ys, color=color, alpha=0.35, s=35, label="raw points")

    ax.errorbar(
        xs,
        means,
        yerr=stds,
        fmt="-o",
        color=color,
        linewidth=2,
        capsize=4,
        markersize=6,
        label="mean ± std",
    )
    last_index = len(xs) - 1
    for index, (x_value, mean_value, sample_counts, output_tokens) in enumerate(
        zip(xs, means, sample_count_groups, output_token_groups)
    ):
        lines = [_format_sample_counts(sample_counts), f"mean={mean_value:.1f}"]
        if annotate_output_tokens:
            lines.append(
                _format_output_tokens(
                    output_tokens,
                    sample_counts,
                    per_sample=normalize_by_output_tokens_per_sample,
                )
            )

        xytext = (0, 8)
        ha = "center"
        va = "bottom"
        if index == last_index:
            xytext = (-10, -4)
            ha = "right"
            va = "top"

        ax.annotate(
            "\n".join(lines),
            (x_value, mean_value),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            va=va,
            fontsize=9,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    ax.margins(x=0.08, y=0.18)



def plot_two_stage_profile(rows: list[dict[str, Any]], output_path: Path, title: str | None = None) -> None:
    rows = sorted(rows, key=lambda row: (int(row.get("sample_count", 0) or 0), int(row.get("repeat_index", 0) or 0)))
    if not rows:
        raise ValueError("No rows to plot")

    for row in rows:
        row["image_count"] = int(row.get("encoder_cache_hit", 0) or 0) + int(row.get("encoder_cache_miss", 0) or 0)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title or "Two-stage Profile Metrics", fontsize=16, fontweight="bold")
    flat_axes = axes.flatten()

    for ax, spec in zip(flat_axes, METRIC_SPECS):
        _plot_metric(ax, rows, **spec)

    flat_axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot two-stage profile metrics from two_stage_profile_vs_tokens_metrics.json"
    )
    parser.add_argument("json_path", type=Path, help="Path to two_stage_profile_vs_tokens_metrics.json")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: beside input JSON)")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title")
    args = parser.parse_args()

    json_path = args.json_path.resolve()
    output_path = args.output.resolve() if args.output is not None else _default_output_path(json_path)

    rows = _load_rows(json_path)
    plot_two_stage_profile(rows, output_path, title=args.title)


if __name__ == "__main__":
    main()
