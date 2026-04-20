from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import METHOD_COLORS, method_label, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "outputs" / "results" / "public_event_significance_audit.json"
TRACER_COLOR = METHOD_COLORS["TRACER"]
BASELINE_ORDER = [
    "Small-Transformer-Forecaster",
    "LSTM-Forecaster",
    "DLinear-Forecaster",
    "Prefix-Only-Retrieval + Fusion",
]


def _load_rows() -> list[dict[str, object]]:
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for baseline in BASELINE_ORDER:
        pairwise = payload["pairwise"][baseline]
        bootstrap = pairwise["bootstrap"]
        rows.append(
            {
                "baseline": baseline,
                "mean_delta": float(pairwise["mean_delta_auprc"]),
                "ci_low": float(bootstrap["ci95_low"]),
                "ci_high": float(bootstrap["ci95_high"]),
                "seed_record": f"{pairwise['seed_wins']}/{pairwise['seed_ties']}/{pairwise['seed_losses']}",
            }
        )
    return rows


def main() -> None:
    rows = _load_rows()
    y_positions = np.arange(len(rows), dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 3.65))

    for y, row in zip(y_positions, rows):
        ax.hlines(y, row["ci_low"], row["ci_high"], color="#4f4f4f", linewidth=1.35, zorder=2)
        ax.vlines([row["ci_low"], row["ci_high"]], y - 0.08, y + 0.08, color="#4f4f4f", linewidth=0.95, zorder=2)
        ax.scatter(
            [row["mean_delta"]],
            [y],
            s=86,
            marker="D",
            facecolors=TRACER_COLOR,
            edgecolors="#222222",
            linewidths=0.8,
            zorder=3,
        )
        ax.text(
            row["ci_high"] + 0.004,
            y,
            row["seed_record"],
            ha="left",
            va="center",
            fontsize=8.9,
            color="#4a4a4a",
        )

    ax.axvline(0.0, color="#bdbdbd", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_yticks(y_positions, [method_label(str(row["baseline"])) for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel(r"Mean $\Delta$AUPRC of TRACER vs baseline")
    ax.set_xlim(-0.01, 0.115)
    style_axis(ax, grid_axis="x")
    ax.grid(axis="y", visible=False)
    ax.text(0.0, 1.04, "Held-out-family paired audit", transform=ax.transAxes, fontsize=11.0, fontweight="bold", ha="left", va="bottom")
    ax.text(0.985, 1.04, "Seed W/T/L", transform=ax.transAxes, fontsize=9.0, ha="right", va="bottom", color="#555555")
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.20, top=0.90)
    save_figure(fig, "fig_public_event_audit_forest")


if __name__ == "__main__":
    main()
