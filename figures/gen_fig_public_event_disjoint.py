from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from atlasv2_public_results import EVENT_DISJOINT_EXTENDED, collect_rows
from paper_plot_style import METHOD_COLORS, metric_axis_label, method_label, save_figure, style_axis, valid_metric_rows


METRIC_LIMITS = {
    "AUPRC": (0.18, 0.83),
    "LeadTime@P80": (0.0, 16.6),
    "Brier": (0.044, 0.071),
    "Analog-Fidelity@5": (64.0, 80.5),
}

HIGHLIGHT_METHODS = {
    "TRACER",
    "Prefix-Only-Retrieval + Fusion",
    "Small-Transformer-Forecaster",
    "DLinear-Forecaster",
}


def _sorted_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: float(row["metrics"]["AUPRC"] or 0.0), reverse=True)


def _bar_color(method: str) -> str:
    if method in HIGHLIGHT_METHODS:
        return METHOD_COLORS[method]
    return "#d4d4d4"


def _edge_color(method: str) -> str:
    if method in HIGHLIGHT_METHODS:
        return "#2a2a2a"
    return "#8a8a8a"


def _draw_panel(axis: plt.Axes, rows: list[dict[str, object]], metric_key: str, panel_label: str, show_ylabels: bool, show_xlabel: bool) -> None:
    rows = valid_metric_rows(rows, metric_key)
    xmin, xmax = METRIC_LIMITS[metric_key]
    y_positions = list(range(len(rows)))
    widths = [float(row["metrics"][metric_key]) - xmin for row in rows]
    errors = [float(row["std"][metric_key] or 0.0) for row in rows]
    colors = [_bar_color(str(row["display_name"])) for row in rows]
    edgecolors = [_edge_color(str(row["display_name"])) for row in rows]

    bars = axis.barh(
        y_positions,
        widths,
        left=xmin,
        xerr=errors,
        height=0.64,
        color=colors,
        edgecolor=edgecolors,
        linewidth=0.72,
        error_kw={"elinewidth": 0.95, "capsize": 2.5, "ecolor": "#222222"},
        zorder=3,
    )
    for bar, row in zip(bars, rows):
        if row["display_name"] == "TRACER":
            bar.set_linewidth(1.05)
        if row["display_name"] not in HIGHLIGHT_METHODS:
            bar.set_alpha(0.85)

    axis.set_xlim(xmin, xmax)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([method_label(str(row["display_name"])) for row in rows] if show_ylabels else [])
    axis.invert_yaxis()
    axis.tick_params(axis="y", length=0, labelleft=show_ylabels)
    axis.set_xlabel(metric_axis_label(metric_key) if show_xlabel else "")
    style_axis(axis, grid_axis="x")
    axis.grid(axis="y", visible=False)
    axis.text(
        0.0,
        1.04,
        f"{panel_label}  {metric_axis_label(metric_key)}",
        transform=axis.transAxes,
        fontsize=10.9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> None:
    rows = _sorted_rows(collect_rows(EVENT_DISJOINT_EXTENDED, split="test_event_disjoint"))

    fig = plt.figure(figsize=(7.2, 8.7))
    grid = fig.add_gridspec(3, 2, height_ratios=[0.18, 1.0, 1.0], hspace=0.30, wspace=0.24)
    legend_ax = fig.add_subplot(grid[0, :])
    legend_ax.axis("off")
    axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
    ]
    metrics = ["AUPRC", "LeadTime@P80", "Brier", "Analog-Fidelity@5"]
    labels = ["A", "B", "C", "D"]

    for index, (axis, metric_key, panel_label) in enumerate(zip(axes, metrics, labels)):
        _draw_panel(
            axis,
            rows,
            metric_key,
            panel_label=panel_label,
            show_ylabels=index in (0, 2),
            show_xlabel=index in (2, 3),
        )

    axes[0].set_ylabel("Method")
    axes[2].set_ylabel("Method")
    legend_handles = [
        Patch(facecolor=METHOD_COLORS["TRACER"], edgecolor="#2a2a2a", label="TRACER"),
        Patch(facecolor=METHOD_COLORS["Prefix-Only-Retrieval + Fusion"], edgecolor="#2a2a2a", label="Prefix-Only"),
        Patch(facecolor=METHOD_COLORS["Small-Transformer-Forecaster"], edgecolor="#2a2a2a", label="Transformer"),
        Patch(facecolor=METHOD_COLORS["DLinear-Forecaster"], edgecolor="#2a2a2a", label="DLinear"),
        Patch(facecolor="#d4d4d4", edgecolor="#8a8a8a", label="Other baselines"),
    ]
    legend_ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="center",
        ncol=5,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.27, right=0.99, bottom=0.08, top=0.98)
    save_figure(fig, "fig_public_event_disjoint")


if __name__ == "__main__":
    main()
