from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import METHOD_COLORS, METHOD_LABELS, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "outputs" / "results" / "public_label_grounded_evidence_audit.json"
TRACER_COLOR = METHOD_COLORS["TRACER"]
BASELINE_ORDER = [
    "Prefix-Only-Retrieval + Fusion",
    "Shared-Encoder TRACER",
]
METRIC_SPECS = [
    ("PosHit@5", r"$\Delta$PosHit@5"),
    ("AlertedPosPrec@5", r"$\Delta$AlertedPosPrec@5"),
    ("CleanNeg@5", r"$\Delta$CleanNeg@5"),
]


def _load_rows() -> list[dict[str, object]]:
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    pairwise_map = {row["baseline"]: row for row in payload["pairwise"]}
    for baseline in BASELINE_ORDER:
        pairwise = pairwise_map[baseline]
        bootstrap = pairwise["bootstrap"]
        mean_delta = pairwise["mean_delta"]
        rows.append(
            {
                "baseline": baseline,
                "metrics": [
                    {
                        "label": metric_label,
                        "mean": float(mean_delta[metric_key]),
                        "ci_low": float(bootstrap[metric_key]["ci95_low"]),
                        "ci_high": float(bootstrap[metric_key]["ci95_high"]),
                    }
                    for metric_key, metric_label in METRIC_SPECS
                ],
            }
        )
    return rows


def _panel_title(baseline: str) -> str:
    short = METHOD_LABELS.get(baseline, baseline)
    return f"vs {short}"


def main() -> None:
    rows = _load_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    metric_positions = np.arange(len(METRIC_SPECS), dtype=float)

    xmin, xmax = -38.0, 24.0

    for axis, row in zip(axes, rows):
        for y, metric in zip(metric_positions, row["metrics"]):
            axis.hlines(y, metric["ci_low"], metric["ci_high"], color="#4f4f4f", linewidth=1.25, zorder=2)
            axis.vlines([metric["ci_low"], metric["ci_high"]], y - 0.08, y + 0.08, color="#4f4f4f", linewidth=0.9, zorder=2)
            axis.scatter(
                [metric["mean"]],
                [y],
                s=82,
                marker="D",
                facecolors=TRACER_COLOR,
                edgecolors="#222222",
                linewidths=0.75,
                zorder=3,
            )
            label = f"{metric['mean']:+.1f}"
            dx = 0.95 if metric["mean"] >= 0 else -0.95
            axis.text(
                metric["mean"] + dx,
                y,
                label,
                ha="left" if metric["mean"] >= 0 else "right",
                va="center",
                fontsize=8.8,
                color="#4a4a4a",
            )

        axis.axvline(0.0, color="#bdbdbd", linewidth=1.0, linestyle="--", zorder=1)
        axis.set_xlim(xmin, xmax)
        style_axis(axis, grid_axis="x")
        axis.grid(axis="y", visible=False)
        axis.set_xlabel("Delta vs baseline")

    axes[0].set_yticks(metric_positions, [label for _, label in METRIC_SPECS])
    axes[0].invert_yaxis()
    axes[0].text(0.0, 1.08, "A  vs Prefix-Only", transform=axes[0].transAxes, fontsize=10.8, fontweight="bold", ha="left", va="bottom", color=METHOD_COLORS[BASELINE_ORDER[0]])
    axes[1].text(0.0, 1.08, "B  vs Shared-Encoder", transform=axes[1].transAxes, fontsize=10.8, fontweight="bold", ha="left", va="bottom", color=METHOD_COLORS[BASELINE_ORDER[1]])

    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.84, wspace=0.18)
    save_figure(fig, "fig_public_label_grounded_deltas")


if __name__ == "__main__":
    main()
