from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paper_plot_style import METHOD_COLORS, METHOD_LABELS, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
SEEDS = [7, 13, 21]

PANELS = [
    {
        "title": "Chronology route convergence",
        "series": [
            ("r020_dlinear_forecaster_atlasv2_public", "DLinear-Forecaster"),
            ("r006_transformer_forecaster_atlasv2_public", "Small-Transformer-Forecaster"),
            ("r008_prefix_retrieval_atlasv2_public", "Prefix-Only-Retrieval + Fusion"),
            ("r239_tracer_adaptive_chronology_atlasv2_public", "TRACER policy (TCN route)"),
        ],
    },
    {
        "title": "Retrieval-active family convergence",
        "series": [
            ("r009_campaign_mem_atlasv2_public", "Shared-Encoder TRACER"),
            ("r218_campaign_mem_decomp_modular_patch_noaux_atlasv2_public", "TRACER w/o auxiliary horizon"),
            ("r215_campaign_mem_decomp_modular_patch_atlasv2_public", "TRACER Core Mode"),
            ("r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public", "Held-Out-Family Support Line"),
        ],
    },
]

COLOR_OVERRIDES = {
    "TRACER policy (TCN route)": METHOD_COLORS["TRACER"],
}

LABEL_OVERRIDES = {
    "TRACER policy (TCN route)": "TRACER policy (TCN route)",
}


def _load_history(base_experiment_name: str, seed: int) -> list[dict]:
    path = RESULT_DIR / f"{base_experiment_name}_seed{seed}.json"
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("history", [])


def _best_so_far_auprc(base_experiment_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seeded_curves: list[np.ndarray] = []
    max_epochs = 0
    for seed in SEEDS:
        history = _load_history(base_experiment_name, seed)
        values = np.asarray([float(row["dev_metrics"]["AUPRC"]) for row in history], dtype=float)
        values = np.maximum.accumulate(values)
        seeded_curves.append(values)
        max_epochs = max(max_epochs, values.shape[0])

    padded_curves: list[np.ndarray] = []
    for curve in seeded_curves:
        if curve.shape[0] < max_epochs:
            pad = np.full(max_epochs - curve.shape[0], curve[-1], dtype=float)
            curve = np.concatenate([curve, pad])
        padded_curves.append(curve)

    stacked = np.vstack(padded_curves)
    epochs = np.arange(1, max_epochs + 1, dtype=int)
    return epochs, stacked.mean(axis=0), stacked.std(axis=0)


def _display_label(label: str) -> str:
    return LABEL_OVERRIDES.get(label, METHOD_LABELS.get(label, label))


def _display_color(label: str) -> str:
    return COLOR_OVERRIDES.get(label, METHOD_COLORS.get(label, "#444444"))


def _plot_panel(ax: plt.Axes, panel: dict) -> list[Line2D]:
    legend_handles: list[Line2D] = []
    max_epochs = 0
    for base_experiment_name, display_name in panel["series"]:
        epochs, mean_curve, std_curve = _best_so_far_auprc(base_experiment_name)
        max_epochs = max(max_epochs, int(epochs[-1]))
        color = _display_color(display_name)
        linewidth = 2.4 if "TRACER" in display_name or "Support Line" in display_name else 1.9
        alpha = 0.18 if "TRACER" in display_name or "Support Line" in display_name else 0.12
        ax.plot(epochs, mean_curve, color=color, linewidth=linewidth)
        ax.fill_between(
            epochs,
            np.clip(mean_curve - std_curve, 0.0, 1.0),
            np.clip(mean_curve + std_curve, 0.0, 1.0),
            color=color,
            alpha=alpha,
            linewidth=0.0,
        )
        legend_handles.append(Line2D([0], [0], color=color, lw=linewidth, label=_display_label(display_name)))

    ax.set_xlim(1, max_epochs)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Best-so-far dev AUPRC")
    ax.set_xticks(np.arange(1, max_epochs + 1, max(1, max_epochs // 6)))
    ax.set_title(panel["title"], loc="left", pad=5)
    style_axis(ax, grid_axis="y")
    return legend_handles


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.45), sharey=True)
    all_handles: list[Line2D] = []
    for panel_index, (ax, panel) in enumerate(zip(axes, PANELS)):
        panel_handles = _plot_panel(ax, panel)
        if panel_index == 0:
            all_handles.extend(panel_handles)
        else:
            all_handles.extend(panel_handles)
        ax.text(-0.12, 1.03, chr(ord("A") + panel_index), transform=ax.transAxes, fontsize=11.4, fontweight="bold")

    deduped: dict[str, Line2D] = {}
    for handle in all_handles:
        deduped.setdefault(handle.get_label(), handle)
    fig.legend(
        list(deduped.values()),
        list(deduped.keys()),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.16, top=0.78, wspace=0.16)
    save_figure(fig, "fig_public_training_dynamics")


if __name__ == "__main__":
    main()
