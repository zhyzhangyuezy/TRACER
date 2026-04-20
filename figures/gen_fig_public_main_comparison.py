from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from atlasv2_public_results import CHRONO_MAIN, collect_rows
from paper_plot_style import METHOD_COLORS, metric_axis_label, method_label, save_figure, style_axis


METRIC_LIMITS = {
    "AUPRC": (0.0, 0.74),
    "LeadTime@P80": (0.0, 8.2),
    "Brier": (0.031, 0.0485),
}

HIGHLIGHT_METHODS = {
    "TRACER",
    "Prefix-Only-Retrieval + Fusion",
    "TCN-Forecaster",
    "DLinear-Forecaster",
}


def _bar_color(method: str) -> str:
    if method in HIGHLIGHT_METHODS:
        return METHOD_COLORS[method]
    return "#d4d4d4"


def _edge_color(method: str) -> str:
    if method in HIGHLIGHT_METHODS:
        return "#2a2a2a"
    return "#8a8a8a"


def _sorted_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda row: float(row["metrics"]["AUPRC"] or 0.0), reverse=True)


def _draw_panel(
    axis: plt.Axes,
    rows: list[dict[str, object]],
    metric_key: str,
    panel_label: str,
    show_ylabels: bool,
    show_xlabel: bool,
) -> None:
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
        height=0.66,
        color=colors,
        edgecolor=edgecolors,
        linewidth=0.75,
        error_kw={"elinewidth": 0.95, "capsize": 2.6, "ecolor": "#222222"},
        zorder=3,
    )
    for bar, row in zip(bars, rows):
        if row["display_name"] == "TRACER":
            bar.set_linewidth(1.1)
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
        fontsize=11.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def main() -> None:
    rows = _sorted_rows(collect_rows(CHRONO_MAIN))

    fig = plt.figure(figsize=(7.2, 8.1))
    grid = fig.add_gridspec(4, 1, height_ratios=[0.20, 1.0, 1.0, 1.0], hspace=0.40)
    legend_ax = fig.add_subplot(grid[0, 0])
    legend_ax.axis("off")
    axes = [fig.add_subplot(grid[i, 0]) for i in range(1, 4)]

    _draw_panel(axes[0], rows, "AUPRC", panel_label="A", show_ylabels=True, show_xlabel=False)
    _draw_panel(axes[1], rows, "LeadTime@P80", panel_label="B", show_ylabels=True, show_xlabel=False)
    _draw_panel(axes[2], rows, "Brier", panel_label="C", show_ylabels=True, show_xlabel=True)
    axes[0].set_ylabel("Method")
    axes[1].set_ylabel("Method")
    axes[2].set_ylabel("Method")

    legend_handles = [
        Patch(facecolor=METHOD_COLORS["TRACER"], edgecolor="#2a2a2a", label="TRACER"),
        Patch(facecolor=METHOD_COLORS["Prefix-Only-Retrieval + Fusion"], edgecolor="#2a2a2a", label="Prefix-Only"),
        Patch(facecolor=METHOD_COLORS["TCN-Forecaster"], edgecolor="#2a2a2a", label="TCN"),
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
    fig.subplots_adjust(left=0.28, right=0.99, bottom=0.08, top=0.98)
    save_figure(fig, "fig_public_main_comparison")


if __name__ == "__main__":
    main()
