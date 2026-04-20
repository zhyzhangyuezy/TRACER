from __future__ import annotations

import matplotlib.pyplot as plt

from atlasv2_public_results import CHRONO_MAIN, EVENT_DISJOINT, collect_rows
from paper_plot_style import METHOD_COLORS, METHOD_MARKERS, method_label, save_figure, style_axis


CHRONO_METHODS = [
    "TRACER",
    "TCN-Forecaster",
    "Small-Transformer-Forecaster",
    "Prefix-Only-Retrieval + Fusion",
]

EVENT_METHODS = [
    "TRACER",
    "Small-Transformer-Forecaster",
    "LSTM-Forecaster",
    "DLinear-Forecaster",
    "Prefix-Only-Retrieval + Fusion",
]

LABEL_OFFSETS = {
    ("Chronological", "TRACER"): (8, 6),
    ("Chronological", "TCN-Forecaster"): (8, -14),
    ("Chronological", "Small-Transformer-Forecaster"): (8, 6),
    ("Chronological", "Prefix-Only-Retrieval + Fusion"): (8, 6),
    ("Event-disjoint", "TRACER"): (8, 6),
    ("Event-disjoint", "Small-Transformer-Forecaster"): (8, -14),
    ("Event-disjoint", "LSTM-Forecaster"): (-72, 6),
    ("Event-disjoint", "DLinear-Forecaster"): (-68, -14),
    ("Event-disjoint", "Prefix-Only-Retrieval + Fusion"): (8, 6),
}


def _draw_panel(axis: plt.Axes, split_label: str, rows_by_method: dict[str, dict[str, object]], methods: list[str]) -> None:
    for method in methods:
        row = rows_by_method[method]
        recall = float(row["metrics"]["Recall@P80"] or 0.0)
        lead_time = float(row["metrics"]["LeadTime@P80"] or 0.0)
        precision = float(row["metrics"]["Precision@P80"] or 0.0)
        color = METHOD_COLORS[method]
        marker = METHOD_MARKERS[method]
        axis.scatter(
            recall,
            lead_time,
            s=94 if method == "TRACER" else 76,
            c=color,
            marker=marker,
            edgecolors="#222222",
            linewidths=0.7,
            zorder=3,
        )
        dx, dy = LABEL_OFFSETS.get((split_label, method), (8, 6))
        axis.annotate(
            f"{method_label(method)}\nP={precision:.2f}",
            xy=(recall, lead_time),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.9,
            ha="left",
            va="center",
            color=color,
            fontweight="bold" if method == "TRACER" else None,
        )

    axis.set_xlabel("Recall@P80")
    axis.set_ylabel("LeadTime@P80 (min)")
    style_axis(axis, grid_axis="both")


def main() -> None:
    chrono_rows = {str(row["display_name"]): row for row in collect_rows(CHRONO_MAIN)}
    event_rows = {str(row["display_name"]): row for row in collect_rows(EVENT_DISJOINT, split="test_event_disjoint")}

    fig, axes = plt.subplots(2, 1, figsize=(7.25, 6.05))
    _draw_panel(axes[0], "Chronological", chrono_rows, CHRONO_METHODS)
    _draw_panel(axes[1], "Event-disjoint", event_rows, EVENT_METHODS)

    axes[0].set_xlim(-0.02, 0.34)
    axes[0].set_ylim(-0.4, 7.4)
    axes[1].set_xlim(0.50, 0.64)
    axes[1].set_ylim(10.2, 12.1)

    axes[0].text(0.0, 1.04, "A  Chronological P80 operating point", transform=axes[0].transAxes, fontsize=11.0, fontweight="bold", ha="left", va="bottom")
    axes[1].text(0.0, 1.04, "B  Held-out-family P80 operating point", transform=axes[1].transAxes, fontsize=11.0, fontweight="bold", ha="left", va="bottom")

    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.09, top=0.97, hspace=0.50)
    save_figure(fig, "fig_public_operating_point_map")


if __name__ == "__main__":
    main()
