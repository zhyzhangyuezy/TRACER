from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_plot_style import save_figure


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, title: str, body: str, facecolor: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.4,
        edgecolor="#334155",
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + 0.02, xy[1] + height - 0.06, title, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
    ax.text(xy[0] + 0.02, xy[1] + height - 0.12, body, transform=ax.transAxes, fontsize=9, va="top")


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#475569", style: str = "-|>") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=1.5,
            color=color,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    ax.axis("off")

    _box(
        ax,
        (0.02, 0.58),
        0.18,
        0.24,
        "Alert Prefix",
        "4-bin alert prefix\nwith host and alert features\nfor the current query window",
        "#edf6f9",
    )
    _box(
        ax,
        (0.24, 0.58),
        0.18,
        0.24,
        "Decomposition",
        "Moving-average split into\ntrend and residual views\nof the same alert prefix",
        "#e0ecf4",
    )
    _box(
        ax,
        (0.47, 0.72),
        0.17,
        0.20,
        "Trend Path",
        "Stable-regime forecast\nfrom the decomposed\ntrend signal",
        "#fef3c7",
    )
    _box(
        ax,
        (0.47, 0.44),
        0.17,
        0.20,
        "Shock Path",
        "Burst-sensitive forecast\nfrom the residual signal\nusing local patches",
        "#fae8ff",
    )
    _box(
        ax,
        (0.47, 0.16),
        0.17,
        0.20,
        "Retrieval Branch",
        "Top-$k$ nearest windows\nfrom train-only memory bank\nusing cosine similarity",
        "#dcfce7",
    )
    _box(
        ax,
        (0.71, 0.54),
        0.14,
        0.28,
        "Analog Calibration",
        "Retrieval-backed base score\nplus bounded correction\nfrom trend and shock paths",
        "#fee2e2",
    )
    _box(
        ax,
        (0.88, 0.58),
        0.10,
        0.24,
        "Output",
        "Final escalation\nscore plus returned\nhistorical analogs",
        "#ede9fe",
    )
    _box(
        ax,
        (0.54, 0.08),
        0.24,
        0.20,
        "Training Objective",
        "Main BCE + auxiliary BCE + contrastive loss;\nhard negatives emphasize prefixes that look similar now\nbut diverge later in future risk",
        "#f8fafc",
    )
    _box(
        ax,
        (0.24, 0.08),
        0.22,
        0.20,
        "Future Signature Supervision",
        "30-min main label,\n10-min auxiliary label,\nand future-trajectory signature",
        "#f8fafc",
    )

    ax.text(0.55, 0.39, "Train-only memory bank of historical windows", transform=ax.transAxes, fontsize=9, color="#166534", ha="center", va="center")

    _arrow(ax, (0.20, 0.70), (0.24, 0.70))
    _arrow(ax, (0.42, 0.76), (0.47, 0.82))
    _arrow(ax, (0.42, 0.70), (0.47, 0.54))
    _arrow(ax, (0.42, 0.64), (0.47, 0.26))
    _arrow(ax, (0.64, 0.82), (0.71, 0.73))
    _arrow(ax, (0.64, 0.54), (0.71, 0.66))
    _arrow(ax, (0.64, 0.26), (0.71, 0.59))
    _arrow(ax, (0.85, 0.68), (0.88, 0.70))
    _arrow(ax, (0.35, 0.28), (0.35, 0.58), color="#64748b")
    _arrow(ax, (0.66, 0.28), (0.58, 0.16), color="#64748b")

    fig.suptitle("TRACER: Decomposition-Guided Retrieval Calibration for Alert Escalation Forecasting", y=0.98, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "fig_campaign_mem_overview")


if __name__ == "__main__":
    main()
