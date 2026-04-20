from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import METHOD_COLORS, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
ATLAS_EVENT_PATTERN = (
    ROOT
    / "outputs"
    / "results"
    / "audits"
    / "public_event_significance_seed_exports"
    / "r240_tracer_adaptive_event_atlasv2_public_test_event_disjoint_seed*.json"
)
AIT_CHRONO_PATTERN = (
    ROOT
    / "outputs"
    / "results"
    / "audits"
    / "ait_ads_chronology_significance_seed_exports"
    / "r241_tracer_adaptive_ait_ads_public_test_seed*.json"
)
N_BINS = 10

RISK_COLOR = METHOD_COLORS["TRACER"]
EVIDENCE_COLOR = METHOD_COLORS["Prefix-Only-Retrieval + Fusion"]


def _score_bin_summary(pattern: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(pattern.parent.glob(pattern.name))
    score_means: list[list[float]] = []
    positive_rates: list[list[float]] = []
    evidence_rates: list[list[float]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        predictions = payload["predictions"]
        y_score = np.asarray(predictions["y_score"], dtype=float)
        y_true = np.asarray(predictions["y_true"], dtype=float)
        retrieved_positive_fraction = np.asarray(
            [sum(labels) / len(labels) if labels else 0.0 for labels in predictions["retrieved_label_main"]],
            dtype=float,
        )
        order = np.argsort(y_score)
        bins = np.array_split(order, N_BINS)
        score_means.append([float(y_score[idx].mean()) for idx in bins])
        positive_rates.append([float(y_true[idx].mean()) for idx in bins])
        evidence_rates.append([float(retrieved_positive_fraction[idx].mean()) for idx in bins])

    score_stack = np.vstack(score_means)
    positive_stack = np.vstack(positive_rates)
    evidence_stack = np.vstack(evidence_rates)
    x = np.arange(1, N_BINS + 1, dtype=int)
    return (
        x,
        score_stack.mean(axis=0),
        positive_stack.mean(axis=0),
        positive_stack.std(axis=0),
        evidence_stack.mean(axis=0),
        evidence_stack.std(axis=0),
    )


def _plot_panel(ax: plt.Axes, pattern: Path, title: str) -> None:
    x, score_mean, positive_mean, positive_std, evidence_mean, evidence_std = _score_bin_summary(pattern)
    ax.plot(
        x,
        positive_mean,
        color=RISK_COLOR,
        marker="o",
        markersize=4.0,
        linewidth=2.2,
        label="Empirical positive rate",
    )
    ax.fill_between(
        x,
        np.clip(positive_mean - positive_std, 0.0, 1.0),
        np.clip(positive_mean + positive_std, 0.0, 1.0),
        color=RISK_COLOR,
        alpha=0.14,
        linewidth=0.0,
    )
    ax.plot(
        x,
        evidence_mean,
        color=EVIDENCE_COLOR,
        marker="s",
        markersize=4.0,
        linewidth=2.0,
        label="Mean retrieved positive fraction@5",
    )
    ax.fill_between(
        x,
        np.clip(evidence_mean - evidence_std, 0.0, 1.0),
        np.clip(evidence_mean + evidence_std, 0.0, 1.0),
        color=EVIDENCE_COLOR,
        alpha=0.14,
        linewidth=0.0,
    )

    ax.set_xlim(1, N_BINS)
    ax.set_ylim(0.0, 0.68)
    ax.set_xticks(x)
    ax.set_xlabel("Warning-score decile (low to high)")
    ax.set_ylabel("Rate")
    ax.set_title(title, loc="left", pad=5)
    style_axis(ax, grid_axis="y")

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(x)
    top.set_xticklabels([f"{value:.2f}" for value in score_mean], fontsize=8.8)
    top.set_xlabel("Mean score in decile")
    top.tick_params(axis="x", pad=2)
    top.spines["bottom"].set_visible(False)
    top.spines["top"].set_color("#9c9c9c")


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.55), sharey=True)
    _plot_panel(axes[0], ATLAS_EVENT_PATTERN, "ATLASv2 held-out-family route")
    _plot_panel(axes[1], AIT_CHRONO_PATTERN, "AIT-ADS chronology route")
    axes[0].text(-0.12, 1.03, "A", transform=axes[0].transAxes, fontsize=11.4, fontweight="bold")
    axes[1].text(-0.12, 1.03, "B", transform=axes[1].transAxes, fontsize=11.4, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.19, top=0.74, wspace=0.16)
    save_figure(fig, "fig_warning_score_evidence_alignment")


if __name__ == "__main__":
    main()
