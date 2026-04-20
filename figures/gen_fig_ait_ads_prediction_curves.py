from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve

from paper_plot_style import METHOD_COLORS, FIG_DIR, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "results" / "ait_ads_prediction_visuals_seed7.json"

DISPLAY_NAMES = {
    "r068_dlinear_forecaster_ait_ads_public": "DLinear-Forecaster",
    "r070_transformer_forecaster_ait_ads_public": "Small-Transformer-Forecaster",
    "r071_prefix_retrieval_ait_ads_public": "Prefix-Only-Retrieval + Fusion",
    "r117_campaign_mem_dual_selector_proxy_strict_ait_ads_public": "TRACER",
}


def _load_runs() -> list[dict]:
    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["runs"]


def _plot_split(ax: plt.Axes, runs: list[dict], split_name: str, title: str) -> None:
    for run in runs:
        split = run.get(split_name)
        if not split:
            continue
        predictions = split["predictions"]
        y_true = np.asarray(predictions["y_true"], dtype=int)
        y_score = np.asarray(predictions["y_score"], dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score) if np.unique(y_true).size > 1 else 0.0
        display_name = DISPLAY_NAMES.get(run["experiment_name"], run["experiment_name"])
        ax.plot(
            recall,
            precision,
            linewidth=2.0 if display_name == "TRACER" else 1.6,
            color=METHOD_COLORS.get(display_name, "#444444"),
            label=f"{display_name} ({ap:.3f})",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    style_axis(ax, grid_axis="both")
    ax.set_title(title, loc="left", pad=5)


def main() -> None:
    runs = _load_runs()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    _plot_split(axes[0], runs, "test", "Chronological Test")
    _plot_split(axes[1], runs, "test_event_disjoint", "Scenario-Held-Out Test")
    axes[0].text(-0.12, 1.03, "A", transform=axes[0].transAxes, fontsize=11.4, fontweight="bold")
    axes[1].text(-0.12, 1.03, "B", transform=axes[1].transAxes, fontsize=11.4, fontweight="bold")
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    unique_labels: list[str] = []
    unique_handles: list[Line2D] = []
    for handle, label in zip(legend_handles, legend_labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    fig.legend(
        unique_handles,
        unique_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.14, top=0.80, wspace=0.20)
    save_figure(fig, "fig_ait_ads_prediction_curves")


if __name__ == "__main__":
    main()
