from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from atlasv2_public_results import CHRONO_MAIN, aggregate_run, collect_rows
from paper_plot_style import METHOD_COLORS, METHOD_MARKERS, method_label, save_figure, style_axis


FRONTIER_METHODS = {"LR-TailRisk", "TCN-Forecaster"}
LABEL_OFFSETS = {
    "LR-TailRisk": (10, 10),
    "TCN-Forecaster": (10, 8),
    "TRACER": (8, -12),
    "Prefix-Only-Retrieval + Fusion": (8, 8),
}


def _is_frontier(method: str) -> bool:
    return method in FRONTIER_METHODS


def _draw_forecasting_panel(axis: plt.Axes) -> None:
    chrono_rows = collect_rows(CHRONO_MAIN)
    random_row = aggregate_run("r011_random_retrieval_atlasv2_public", "Random-Retrieval + Fusion")
    rows = chrono_rows + [random_row]
    sorted_frontier = sorted(
        [row for row in rows if _is_frontier(str(row["display_name"]))],
        key=lambda row: float(row["metrics"]["AUPRC"] or 0.0),
    )

    for row in rows:
        method = str(row["display_name"])
        x = float(row["metrics"]["AUPRC"] or 0.0)
        y = float(row["metrics"]["LeadTime@P80"] or 0.0)
        color = METHOD_COLORS[method]
        if _is_frontier(method) or method == "TRACER":
            axis.scatter(
                x,
                y,
                s=92 if method == "TRACER" else 76,
                c=color,
                marker=METHOD_MARKERS[method],
                edgecolors="#222222",
                linewidths=0.7,
                zorder=3,
            )
        else:
            axis.scatter(
                x,
                y,
                s=70,
                c="#cdcdcd",
                marker=METHOD_MARKERS[method],
                edgecolors="#555555",
                linewidths=0.6,
                zorder=2,
            )

        if method in LABEL_OFFSETS:
            dx, dy = LABEL_OFFSETS[method]
            axis.annotate(
                method_label(method),
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9.4,
                ha="left",
                va="bottom",
                color=color if method in METHOD_COLORS else "#666666",
                fontweight="bold" if method == "TRACER" else None,
            )

    axis.plot(
        [float(row["metrics"]["AUPRC"] or 0.0) for row in sorted_frontier],
        [float(row["metrics"]["LeadTime@P80"] or 0.0) for row in sorted_frontier],
        color="#666666",
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )

    axis.set_xlabel("AUPRC")
    axis.set_ylabel("LeadTime@P80 (min)")
    axis.set_xlim(0.0, 0.82)
    axis.set_ylim(-0.4, 8.1)
    style_axis(axis, grid_axis="both")

    handles = [
        Line2D([0], [0], marker="o", color="#666666", markerfacecolor="#666666", linestyle="--", linewidth=1.0, markersize=6.5, label="Pareto frontier"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#cdcdcd", markeredgecolor="#555555", markersize=6.5, label="Dominated control"),
    ]
    axis.legend(handles=handles, frameon=False, loc="lower right")


def _draw_retrieval_panel(axis: plt.Axes) -> None:
    prefix_chrono = aggregate_run("r008_prefix_retrieval_atlasv2_public", "Prefix-Only-Retrieval + Fusion")
    prefix_event = aggregate_run("r010_prefix_retrieval_atlasv2_public_event_disjoint", "Prefix-Only-Retrieval + Fusion")
    campaign_chrono = aggregate_run("r215_campaign_mem_decomp_modular_patch_atlasv2_public", "TRACER")
    campaign_event = aggregate_run("r215_campaign_mem_decomp_modular_patch_atlasv2_public", "TRACER", split="test_event_disjoint")
    pure_knn = aggregate_run("r007_pure_knn_atlasv2_public", "Pure-kNN-Retrieval")

    paired_methods = [
        ("Prefix-Only", prefix_chrono, prefix_event, METHOD_COLORS["Prefix-Only-Retrieval + Fusion"]),
        ("TRACER core", campaign_chrono, campaign_event, METHOD_COLORS["TRACER Core Mode"]),
    ]

    for label, chrono_row, event_row, color in paired_methods:
        x0 = float(chrono_row["metrics"]["Analog-Fidelity@5"] or 0.0)
        y0 = float(chrono_row["metrics"]["AUPRC"] or 0.0)
        x1 = float(event_row["metrics"]["Analog-Fidelity@5"] or 0.0)
        y1 = float(event_row["metrics"]["AUPRC"] or 0.0)
        axis.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4, "alpha": 0.85},
            zorder=1,
        )
        axis.scatter(x0, y0, s=78, facecolors="white", edgecolors=color, linewidths=1.2, marker="o", zorder=3)
        axis.scatter(x1, y1, s=80, facecolors=color, edgecolors="#222222", linewidths=0.7, marker="s", zorder=3)
        axis.annotate(
            label,
            xy=(x1, y1),
            xytext=(8, 8 if label == "Prefix-Only" else -12),
            textcoords="offset points",
            color=color,
            fontsize=9.1,
            ha="left",
            va="center",
            fontweight="bold" if label == "TRACER core" else None,
        )

    xk = float(pure_knn["metrics"]["Analog-Fidelity@5"] or 0.0)
    yk = float(pure_knn["metrics"]["AUPRC"] or 0.0)
    axis.scatter(xk, yk, s=84, c="#d0d0d0", marker="P", edgecolors="#444444", linewidths=0.7, zorder=2)
    axis.annotate("Pure-kNN", xy=(xk, yk), xytext=(8, -8), textcoords="offset points", color="#8a8a8a", fontsize=8.8, ha="left", va="center")

    axis.set_xlabel("AF@5")
    axis.set_ylabel("AUPRC")
    axis.set_xlim(69.0, 74.8)
    axis.set_ylim(0.0, 0.82)
    style_axis(axis, grid_axis="both")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", markeredgewidth=1.1, markersize=6.5, label="Chronological"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#555555", markeredgecolor="#555555", markersize=6.5, label="Held-out family"),
    ]
    axis.legend(handles=handles, frameon=False, loc="lower right")


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.25, 6.0))
    _draw_forecasting_panel(axes[0])
    _draw_retrieval_panel(axes[1])
    axes[0].text(0.0, 1.04, "A  Chronological forecast frontier", transform=axes[0].transAxes, fontsize=11.0, fontweight="bold", ha="left", va="bottom")
    axes[1].text(0.0, 1.04, "B  Retrieval shift across splits", transform=axes[1].transAxes, fontsize=11.0, fontweight="bold", ha="left", va="bottom")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.10, top=0.97, hspace=0.50)
    save_figure(fig, "fig_public_tradeoff_frontier")


if __name__ == "__main__":
    main()
