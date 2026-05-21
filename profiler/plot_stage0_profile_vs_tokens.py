from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def _load_rows(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, got {type(data).__name__}")
    return [row for row in data if isinstance(row, dict)]


def _default_output_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}_plots.png")


def _default_ratio_output_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}_ratio.png")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float], mean_value: float) -> float:
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _aggregate_points(rows: list[dict], x_key: str, y_key: str) -> tuple[list[float], list[float], list[float], list[list[int]], list[tuple[float, float]]]:
    grouped: dict[float, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        x_value = float(row.get(x_key, 0) or 0)
        y_value = float(row.get(y_key, 0.0) or 0.0)
        sample_count = int(row.get("sample_count", 0) or 0)
        grouped[x_value].append((y_value, sample_count))

    xs = sorted(grouped)
    means: list[float] = []
    stds: list[float] = []
    sample_count_groups: list[list[int]] = []
    raw_points: list[tuple[float, float]] = []
    for x_value in xs:
        pairs = grouped[x_value]
        y_values = [y for y, _ in pairs]
        mean_value = _mean(y_values)
        means.append(mean_value)
        stds.append(_std(y_values, mean_value))
        sample_count_groups.append([sample_count for _, sample_count in pairs])
        raw_points.extend((x_value, y) for y in y_values)
    return xs, means, stds, sample_count_groups, raw_points


def _format_sample_counts(sample_counts: list[int]) -> str:
    unique_counts = sorted(set(sample_counts))
    if len(unique_counts) == 1:
        return f"n={unique_counts[0]}"
    return "n=" + "/".join(str(value) for value in unique_counts)


def _plot_metric(
    ax,
    rows: list[dict],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    color: str,
) -> None:
    xs, means, stds, sample_count_groups, raw_points = _aggregate_points(rows, x_key, y_key)
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
    for x_value, mean_value, sample_counts in zip(xs, means, sample_count_groups):
        ax.annotate(
            f"{_format_sample_counts(sample_counts)}\nmean={mean_value:.1f}",
            (x_value, mean_value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()


def plot_stage0_profile(rows: list[dict], output_path: Path, title: str | None = None) -> None:
    rows = sorted(rows, key=lambda row: (int(row.get("sample_count", 0) or 0), int(row.get("repeat_index", 0) or 0)))
    if not rows:
        raise ValueError("No rows to plot")

    for row in rows:
        row["image_count"] = int(row.get("encoder_cache_hit", 0) or 0) + int(row.get("encoder_cache_miss", 0) or 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title or "Stage-0 Profile Metrics", fontsize=15, fontweight="bold")

    _plot_metric(
        axes[0],
        rows,
        x_key="image_count",
        y_key="embed_multimodal_ms",
        x_label="Image count",
        y_label="embed_multimodal_ms",
        title="Image count vs embed_multimodal_ms",
        color="#1f77b4",
    )
    _plot_metric(
        axes[1],
        rows,
        x_key="batch_total_scheduled_tokens",
        y_key="forward_ms",
        x_label="batch_total_scheduled_tokens",
        y_label="forward_ms",
        title="batch_total_scheduled_tokens vs forward_ms",
        color="#ff7f0e",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)


def plot_stage0_ratio(rows: list[dict], output_path: Path, title: str | None = None) -> None:
    rows = sorted(rows, key=lambda row: (int(row.get("sample_count", 0) or 0), int(row.get("repeat_index", 0) or 0)))
    if not rows:
        raise ValueError("No rows to plot")

    ratio_rows: list[dict] = []
    for row in rows:
        forward_ms = float(row.get("forward_ms", 0.0) or 0.0)
        if forward_ms <= 0:
            continue
        ratio_row = dict(row)
        ratio_row["time_ratio"] = float(row.get("embed_multimodal_ms", 0.0) or 0.0) / forward_ms
        ratio_rows.append(ratio_row)

    if not ratio_rows:
        raise ValueError("No valid rows to plot ratio")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(title or "Stage-0 Time Ratio", fontsize=15, fontweight="bold")
    _plot_metric(
        ax,
        ratio_rows,
        x_key="sample_count",
        y_key="time_ratio",
        x_label="sample_count (n)",
        y_label="embed_multimodal_ms / forward_ms",
        title="sample_count vs embed_multimodal_ms / forward_ms",
        color="#2ca02c",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Ratio plot saved to {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stage-0 profile metrics from stage0_profile_vs_tokens.json")
    parser.add_argument("json_path", type=Path, help="Path to stage0_profile_vs_tokens.json")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: beside input JSON)")
    parser.add_argument("--ratio-output", type=Path, default=None, help="Ratio image path (default: beside input JSON)")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title")
    args = parser.parse_args()

    json_path = args.json_path.resolve()
    output_path = args.output.resolve() if args.output is not None else _default_output_path(json_path)
    ratio_output_path = (
        args.ratio_output.resolve() if args.ratio_output is not None else _default_ratio_output_path(json_path)
    )

    rows = _load_rows(json_path)
    plot_stage0_profile(rows, output_path, title=args.title)
    plot_stage0_ratio(rows, ratio_output_path, title=(f"{args.title} Ratio" if args.title else None))


if __name__ == "__main__":
    main()
