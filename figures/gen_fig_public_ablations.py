from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from atlasv2_public_results import ABLATIONS_RETRIEVAL, EVENT_DISJOINT_RETRIEVAL, collect_rows
from paper_plot_style import SCIENTIFIC_SEQUENTIAL_CMAP, save_figure


METHOD_ORDER = [
    "Prefix-Only-Retrieval + Fusion",
    "Pure-kNN-Retrieval",
    "Shared-Encoder TRACER",
    "TRACER Core Mode",
    "TRACER (adaptive policy)",
    "Event-Focused TRACER Variant",
    "Chronology Support Line",
    "TRACER w/o auxiliary horizon",
    "Held-Out-Family Support Line",
]

DISPLAY_LABELS = {
    "Prefix-Only-Retrieval + Fusion": "Prefix-Only",
    "Pure-kNN-Retrieval": "Pure-kNN",
    "Shared-Encoder TRACER": "Shared-Encoder",
    "TRACER Core Mode": "TRACER core",
    "TRACER (adaptive policy)": "TRACER policy",
    "Event-Focused TRACER Variant": "event support",
    "Chronology Support Line": "chrono support",
    "TRACER w/o auxiliary horizon": "TRACER w/o aux",
    "Held-Out-Family Support Line": "family support",
}


def _row_map(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["display_name"]): row for row in rows}


def _normalize_column(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan)
    low = float(finite.min())
    high = float(finite.max())
    if abs(high - low) <= 1e-12:
        out = np.full_like(values, 0.5)
        out[~np.isfinite(values)] = np.nan
        return out
    out = (values - low) / (high - low)
    out[~np.isfinite(values)] = np.nan
    return out


def _build_matrix(chrono_map: dict[str, dict[str, object]], event_map: dict[str, dict[str, object]], metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.array(
        [
            [
                float(chrono_map[method]["metrics"][metric_key]) if chrono_map[method]["metrics"][metric_key] is not None else np.nan,
                float(event_map[method]["metrics"][metric_key]) if event_map[method]["metrics"][metric_key] is not None else np.nan,
            ]
            for method in METHOD_ORDER
        ],
        dtype=float,
    )
    norm = np.zeros_like(raw)
    for col_idx in range(raw.shape[1]):
        norm[:, col_idx] = _normalize_column(raw[:, col_idx])
    return raw, norm


def _annotate_cells(axis: plt.Axes, raw: np.ndarray) -> None:
    for row_idx in range(raw.shape[0]):
        for col_idx in range(raw.shape[1]):
            value = raw[row_idx, col_idx]
            if not np.isfinite(value):
                label = "--"
            else:
                label = f"{value:.3f}" if value < 10 else f"{value:.1f}"
            axis.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=9.2,
                color="#202020",
            )


def _draw_panel(axis: plt.Axes, raw: np.ndarray, norm: np.ndarray, panel_label: str, panel_title: str, show_ylabels: bool):
    image = axis.imshow(np.ma.masked_invalid(norm), cmap=SCIENTIFIC_SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    _annotate_cells(axis, raw)
    axis.set_xticks([0, 1], ["Chronological", "Held-out family"])
    axis.set_yticks(np.arange(len(METHOD_ORDER)), [DISPLAY_LABELS[name] for name in METHOD_ORDER] if show_ylabels else [])
    axis.tick_params(axis="y", length=0, labelleft=show_ylabels)
    axis.text(
        0.0,
        1.04,
        f"{panel_label}  {panel_title}",
        transform=axis.transAxes,
        fontsize=11.0,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    axis.axhline(4.5, color="#ffffff", linewidth=1.2, linestyle="--")
    tracer_row = METHOD_ORDER.index("TRACER (adaptive policy)")
    axis.add_patch(Rectangle((-0.5, tracer_row - 0.5), 2.0, 1.0, fill=False, edgecolor="#7f2704", linewidth=1.6))
    for spine in axis.spines.values():
        spine.set_color("#a2a2a2")
        spine.set_linewidth(0.8)
    axis.grid(False)
    return image


def main() -> None:
    chrono_map = _row_map(collect_rows(ABLATIONS_RETRIEVAL))
    event_map = _row_map(collect_rows(EVENT_DISJOINT_RETRIEVAL, split="test_event_disjoint"))

    auprc_raw, auprc_norm = _build_matrix(chrono_map, event_map, "AUPRC")
    af_raw, af_norm = _build_matrix(chrono_map, event_map, "Analog-Fidelity@5")

    fig = plt.figure(figsize=(7.2, 6.0))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.05], hspace=0.42, wspace=0.08)
    ax_auprc = fig.add_subplot(grid[0, 0])
    ax_af = fig.add_subplot(grid[1, 0])
    cax = fig.add_subplot(grid[:, 1])

    image = _draw_panel(ax_auprc, auprc_raw, auprc_norm, panel_label="A", panel_title="AUPRC", show_ylabels=True)
    _draw_panel(ax_af, af_raw, af_norm, panel_label="B", panel_title="AF@5", show_ylabels=True)
    ax_auprc.set_ylabel("Method")
    ax_af.set_ylabel("Method")

    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("Within-panel normalized score", rotation=90)

    fig.subplots_adjust(left=0.29, right=0.97, bottom=0.08, top=0.98)
    save_figure(fig, "fig_public_ablations")


if __name__ == "__main__":
    main()
