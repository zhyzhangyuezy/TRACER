from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from paper_plot_style import AXIS, METHOD_COLORS, add_row_bands, save_figure, style_axis

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = ROOT / "outputs" / "results" / "r017_public_mechanism_analysis.json"


def _draw_gate_panel(axis: plt.Axes, payload: dict) -> None:
    categories = ["TP", "FN", "TN"]
    splits = [("test", "#ffffff"), ("test_event_disjoint", "#d9d9d9")]
    positions: list[float] = []
    values: list[list[float]] = []
    colors: list[str] = []
    base_positions = np.arange(len(categories), dtype=float)
    offsets = [-0.18, 0.18]

    for offset, (split_key, color) in zip(offsets, splits):
        for idx, category in enumerate(categories):
            raw = payload["gate_values_raw"][split_key].get(category, [])
            if not raw:
                continue
            positions.append(base_positions[idx] + offset)
            values.append(raw)
            colors.append(color)

    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.7},
        capprops={"color": "black", "linewidth": 0.7},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)

    axis.set_xticks(base_positions, categories)
    axis.set_ylabel("Base gate $g_t$")
    axis.set_ylim(0.55, 1.02)
    style_axis(axis, grid_axis="y")


def _draw_margin_panel(axis: plt.Axes, payload: dict) -> None:
    summary = payload["analog_change_summary"]
    split_rows = [("test", "Chronological"), ("test_event_disjoint", "Held-out family")]
    variant_specs = [
        ("same_top1_score_margin_mean", "Same Top-1", "o", "#ffffff", -0.10),
        ("changed_top1_score_margin_mean", "Changed Top-1", "D", "#444444", 0.10),
    ]
    y_base = np.arange(len(split_rows), dtype=float)
    values_for_limits: list[float] = []

    for y, (split_key, _) in enumerate(split_rows):
        for metric_key, _, marker, facecolor, offset in variant_specs:
            entry = summary[split_key][metric_key]
            value = float(entry["mean"])
            std = float(entry["std"] or 0.0)
            values_for_limits.extend([value - std, value + std])
            ypos = y + offset
            axis.hlines(ypos, value - std, value + std, color="#4f4f4f", linewidth=1.0, zorder=2)
            axis.vlines([value - std, value + std], ypos - 0.05, ypos + 0.05, color="#4f4f4f", linewidth=0.8, zorder=2)
            axis.scatter(
                [value],
                [ypos],
                s=58,
                marker=marker,
                facecolors=facecolor,
                edgecolors="#444444",
                linewidths=1.15,
                zorder=3,
            )

    xmin = min(values_for_limits) - 0.004
    xmax = max(values_for_limits) + 0.004
    axis.axvline(0.0, color="#b5b5b5", linewidth=0.9, linestyle="--", zorder=0)
    axis.set_xlim(xmin, xmax)
    axis.set_yticks(y_base, [label for _, label in split_rows])
    axis.set_xlabel("Score margin vs Prefix-Only")
    style_axis(axis, grid_axis="x")


def main() -> None:
    payload = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(7.25, 3.55))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.20, 1.0], hspace=0.34, wspace=0.42)
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.axis("off")
    axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    _draw_gate_panel(axes[0], payload)
    _draw_margin_panel(axes[1], payload)

    axes[0].text(0.0, 1.04, "A  Retrieval base gate", transform=axes[0].transAxes, fontsize=11.0, fontweight="bold", ha="left", va="bottom")
    axes[1].text(0.0, 1.04, "B  Score-margin shift", transform=axes[1].transAxes, fontsize=11.0, fontweight="bold", ha="left", va="bottom")

    legend_handles = [
        Patch(facecolor="#ffffff", edgecolor="black", linewidth=0.7, label="Chronological"),
        Patch(facecolor="#d9d9d9", edgecolor="black", linewidth=0.7, label="Held-out family"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffffff", markeredgecolor="#444444", markersize=6.2, label="Same Top-1"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#444444", markeredgecolor="#444444", markersize=6.2, label="Changed Top-1"),
    ]
    legend_ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="center",
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.45,
    )

    for axis in axes:
        axis.spines["left"].set_color(AXIS)
        axis.spines["bottom"].set_color(AXIS)

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.18, top=0.95)
    save_figure(fig, "fig_public_mechanism_analysis")


if __name__ == "__main__":
    main()
