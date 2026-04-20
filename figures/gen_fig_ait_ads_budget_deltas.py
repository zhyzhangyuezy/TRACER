from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import METHOD_COLORS, method_label, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "outputs" / "results" / "ait_ads_chronology_budget_audit.json"
TRACER_COLOR = METHOD_COLORS["TRACER"]
BASELINE_ORDER = [
    "Prefix-Only-Retrieval + Fusion",
    "DLinear-Forecaster",
    "Small-Transformer-Forecaster",
]
METRIC_SPECS = [
    ("PosRecall", r"$\Delta$PosRecall@2.0\%", (-1.5, 5.2), "A  Recall"),
    ("PosPrec", r"$\Delta$PosPrec@2.0\%", (-2.2, 6.0), "B  Precision"),
    ("MeanLead", r"$\Delta$MeanLead@2.0\% (min)", (-1.5, 2.2), "C  Lead Time"),
]


def _load_rows() -> dict[str, dict[str, dict[str, float]]]:
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    pairwise_map: dict[str, dict[str, dict[str, float]]] = {}
    for row in payload["pairwise"]:
        baseline = str(row["baseline"])
        pairwise_map[baseline] = {}
        for metric_key, _, _, _ in METRIC_SPECS:
            pairwise_map[baseline][metric_key] = {
                "mean": float(row["mean_delta"][metric_key]),
                "ci_low": float(row["bootstrap"][metric_key]["ci95_low"]),
                "ci_high": float(row["bootstrap"][metric_key]["ci95_high"]),
            }
    return pairwise_map


def main() -> None:
    rows = _load_rows()
    y_positions = np.arange(len(BASELINE_ORDER), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.05), sharey=True)

    for axis, (metric_key, xlabel, xlim, panel_title) in zip(axes, METRIC_SPECS):
        for y, baseline in zip(y_positions, BASELINE_ORDER):
            metric = rows[baseline][metric_key]
            axis.hlines(y, metric["ci_low"], metric["ci_high"], color="#4f4f4f", linewidth=1.2, zorder=2)
            axis.vlines([metric["ci_low"], metric["ci_high"]], y - 0.08, y + 0.08, color="#4f4f4f", linewidth=0.85, zorder=2)
            axis.scatter(
                [metric["mean"]],
                [y],
                s=74,
                marker="D",
                facecolors=TRACER_COLOR,
                edgecolors="#222222",
                linewidths=0.75,
                zorder=3,
            )
        axis.axvline(0.0, color="#bdbdbd", linewidth=1.0, linestyle="--", zorder=1)
        axis.set_xlim(*xlim)
        axis.set_xlabel(xlabel)
        axis.text(0.0, 1.06, panel_title, transform=axis.transAxes, fontsize=11.0, fontweight="bold", ha="left", va="bottom")
        style_axis(axis, grid_axis="x")
        axis.grid(axis="y", visible=False)

    axes[0].set_yticks(y_positions, [method_label(name) for name in BASELINE_ORDER])
    axes[0].invert_yaxis()

    fig.subplots_adjust(left=0.28, right=0.99, bottom=0.24, top=0.84, wspace=0.30)
    save_figure(fig, "fig_ait_ads_budget_deltas")


if __name__ == "__main__":
    main()
