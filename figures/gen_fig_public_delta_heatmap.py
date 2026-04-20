from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm

from atlasv2_public_results import ABLATIONS_RETRIEVAL, EVENT_DISJOINT_RETRIEVAL, collect_rows
from paper_plot_style import method_label, save_figure


HIGHER_IS_BETTER = {"AUPRC", "LeadTime@P80", "Analog-Fidelity@5"}
LOWER_IS_BETTER = {"Brier", "TTE-Err@1"}

CHRONO_COLUMNS = [
    ("AUPRC", "AUPRC"),
    ("LeadTime@P80", "Lead"),
    ("Brier", "Brier"),
    ("Analog-Fidelity@5", "AF@5"),
    ("TTE-Err@1", "TTE"),
]
EVENT_COLUMNS = [
    ("AUPRC", "AUPRC"),
    ("LeadTime@P80", "Lead"),
    ("Brier", "Brier"),
    ("Analog-Fidelity@5", "AF@5"),
    ("TTE-Err@1", "TTE"),
]
ROW_ORDER = [
    "Prefix-Only-Retrieval + Fusion",
    "Shared-Encoder TRACER",
    "Event-Focused TRACER Variant",
    "Chronology Support Line",
    "Held-Out-Family Support Line",
    "TRACER w/o auxiliary horizon",
    "TRACER",
    "Random-Retrieval + Fusion",
    "Pure-kNN-Retrieval",
]


def _signed_delta(value: float | None, baseline: float | None, metric_key: str) -> float | None:
    if value is None or baseline is None:
        return None
    if metric_key in HIGHER_IS_BETTER:
        return float(value - baseline)
    if metric_key in LOWER_IS_BETTER:
        return float(baseline - value)
    return float(value - baseline)


def _build_panel(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> tuple[np.ndarray, list[list[str]]]:
    baseline_row = next(row for row in rows if row["display_name"] == "Prefix-Only-Retrieval + Fusion")
    delta_matrix = np.full((len(rows), len(columns)), np.nan, dtype=float)
    annotation_matrix: list[list[str]] = []

    for row_index, row in enumerate(rows):
        annotation_row: list[str] = []
        for col_index, (metric_key, _) in enumerate(columns):
            delta = _signed_delta(row["metrics"].get(metric_key), baseline_row["metrics"].get(metric_key), metric_key)
            if delta is None:
                annotation_row.append("")
                continue
            delta_matrix[row_index, col_index] = delta
            if metric_key in {"AUPRC", "Brier"}:
                annotation_row.append(f"{delta:+.3f}")
            else:
                annotation_row.append(f"{delta:+.2f}")
        annotation_matrix.append(annotation_row)

    normalized = np.full_like(delta_matrix, np.nan)
    for col_index in range(delta_matrix.shape[1]):
        column = delta_matrix[:, col_index]
        valid = column[~np.isnan(column)]
        if valid.size == 0:
            continue
        scale = np.max(np.abs(valid))
        if scale < 1e-8:
            normalized[:, col_index] = 0.0
        else:
            normalized[:, col_index] = column / scale
    return normalized, annotation_matrix


def _ordered_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    row_map = {row["display_name"]: row for row in rows}
    return [row_map[name] for name in ROW_ORDER if name in row_map]


def _draw_panel(
    axis: plt.Axes,
    rows: list[dict[str, object]],
    columns: list[tuple[str, str]],
    header: str,
    show_ylabels: bool,
    panel_label: str,
) -> None:
    values, annotations = _build_panel(rows, columns)
    masked = np.ma.masked_invalid(values)
    cmap = LinearSegmentedColormap.from_list("paper_delta", ["#8c2d04", "#f7f7f7", "#3d4f6a"]).copy()
    cmap.set_bad("#eeeeee")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    axis.imshow(masked, aspect="auto", cmap=cmap, norm=norm)

    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels([f"{label}\n$\\uparrow$" for _, label in columns])
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([method_label(row["display_name"]) for row in rows] if show_ylabels else [])
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", length=0, labelleft=show_ylabels)
    axis.set_title(header, pad=9)
    axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.axhline(5.5, color="#c7c7c7", linewidth=0.9, linestyle="--")
    axis.text(-0.12, 1.03, panel_label, transform=axis.transAxes, fontsize=11.4, fontweight="bold")

    for row_index in range(len(rows)):
        for col_index in range(len(columns)):
            text = annotations[row_index][col_index]
            if not text:
                continue
            color = "white" if abs(masked[row_index, col_index]) >= 0.55 else "#222222"
            axis.text(col_index, row_index, text, ha="center", va="center", fontsize=9.4, color=color)
    return ScalarMappable(norm=norm, cmap=cmap)


def main() -> None:
    chrono_rows = _ordered_rows(collect_rows(ABLATIONS_RETRIEVAL))
    event_rows = _ordered_rows(collect_rows(EVENT_DISJOINT_RETRIEVAL, split="test_event_disjoint"))

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.7), sharey=True)
    sm = _draw_panel(axes[0], chrono_rows, CHRONO_COLUMNS, "Chronological", True, "A")
    _draw_panel(axes[1], event_rows, EVENT_COLUMNS, "Held-out family", False, "B")

    cax = fig.add_axes([0.905, 0.16, 0.022, 0.68])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Normalized delta")
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.set_ticklabels(["Worse", "Parity", "Better"])

    fig.text(
        0.5,
        0.035,
        "All cells report signed deltas against Prefix-Only, so upward direction is consistently better across metrics.",
        ha="center",
        fontsize=9.0,
        color="#5b6778",
    )
    fig.subplots_adjust(left=0.27, right=0.88, bottom=0.14, top=0.90, wspace=0.14)
    save_figure(fig, "fig_public_delta_heatmap")


if __name__ == "__main__":
    main()
