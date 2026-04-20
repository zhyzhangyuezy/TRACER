from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from atlasv2_public_results import CHRONO_MAIN, EVENT_DISJOINT_EXTENDED, collect_rows
from paper_plot_style import SCIENTIFIC_SEQUENTIAL_CMAP, method_label, save_figure


SEED_LABELS = ["Seed 7", "Seed 13", "Seed 21", "Mean"]


def _build_matrix(rows: list[dict[str, object]]) -> tuple[np.ndarray, list[str]]:
    sorted_rows = sorted(rows, key=lambda row: float(row["metrics"]["AUPRC"] or 0.0), reverse=True)
    matrix = []
    labels = []
    for row in sorted_rows:
        values = list(np.asarray(row["values"]["AUPRC"], dtype=float))
        mean_value = float(row["metrics"]["AUPRC"])
        matrix.append(values + [mean_value])
        labels.append(method_label(str(row["display_name"])))
    return np.asarray(matrix, dtype=float), labels


def _draw_panel(axis: plt.Axes, matrix: np.ndarray, labels: list[str], panel_label: str, title: str, show_ylabels: bool):
    image = axis.imshow(matrix, cmap=SCIENTIFIC_SEQUENTIAL_CMAP, aspect="auto", vmin=0.0, vmax=max(0.82, float(matrix.max())))
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            axis.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.3f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color="#16202a",
            )
    axis.set_xticks(np.arange(len(SEED_LABELS)), SEED_LABELS)
    axis.set_yticks(np.arange(len(labels)), labels if show_ylabels else [])
    axis.tick_params(axis="y", length=0, labelleft=show_ylabels)
    axis.axvline(2.5, color="#ffffff", linewidth=1.2)
    axis.text(
        0.0,
        1.04,
        f"{panel_label}  {title}",
        transform=axis.transAxes,
        fontsize=11.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    for spine in axis.spines.values():
        spine.set_color("#a2a2a2")
        spine.set_linewidth(0.8)
    return image


def main() -> None:
    chrono_matrix, chrono_labels = _build_matrix(collect_rows(CHRONO_MAIN))
    event_matrix, event_labels = _build_matrix(collect_rows(EVENT_DISJOINT_EXTENDED, split="test_event_disjoint"))

    fig = plt.figure(figsize=(7.25, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.05], hspace=0.42, wspace=0.08)
    ax_chrono = fig.add_subplot(grid[0, 0])
    ax_event = fig.add_subplot(grid[1, 0])
    cax = fig.add_subplot(grid[:, 1])

    image = _draw_panel(ax_chrono, chrono_matrix, chrono_labels, panel_label="A", title="Chronological AUPRC by seed", show_ylabels=True)
    _draw_panel(ax_event, event_matrix, event_labels, panel_label="B", title="Held-out-family AUPRC by seed", show_ylabels=True)
    ax_chrono.set_ylabel("Method")
    ax_event.set_ylabel("Method")

    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("AUPRC")

    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.08, top=0.98)
    save_figure(fig, "fig_public_seed_stability")


if __name__ == "__main__":
    main()
