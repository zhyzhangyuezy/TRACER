from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parent
DPI = 300
GRID = "#d9d9d9"
AXIS = "#9c9c9c"
TEXT = "#222222"
MUTED = "#555555"

matplotlib.rcParams.update(
    {
        "font.size": 12.6,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.labelsize": 12.6,
        "axes.titlesize": 12.2,
        "axes.titlepad": 4.0,
        "xtick.labelsize": 11.2,
        "ytick.labelsize": 11.2,
        "legend.fontsize": 10.8,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "grid.alpha": 0.16,
        "grid.linewidth": 0.55,
        "grid.color": GRID,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "mathtext.fontset": "stix",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": TEXT,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)

VALUE_LABEL_BBOX = None

METHOD_COLORS = {
    "LR-TailRisk": "#5f5f5f",
    "TCN-Forecaster": "#3d4f6a",
    "DLinear-Forecaster": "#6e6e6e",
    "LSTM-Forecaster": "#556b8d",
    "TimesNet-Forecaster": "#355c7d",
    "TiDE-Forecaster": "#3c6e71",
    "TSMixer-Forecaster": "#52796f",
    "PatchTST-Forecaster": "#4e5d73",
    "Small-Transformer-Forecaster": "#7b7b7b",
    "iTransformer-Forecaster": "#73859b",
    "No-Memory Forecaster": "#8a8a8a",
    "Pure-kNN-Retrieval": "#9b9b9b",
    "Prefix-Only-Retrieval + Fusion": "#6a8aa6",
    "Random-Retrieval + Fusion": "#a7a7a7",
    "TRACER": "#8c2d04",
    "TRACER Core Mode": "#b86a3a",
    "TRACER (adaptive policy)": "#7f2704",
    "Shared-Encoder TRACER": "#b86a3a",
    "Event-Focused TRACER Variant": "#a85415",
    "Chronology Support Line": "#c67d28",
    "Held-Out-Family Support Line": "#7f2704",
    "TRACER w/o auxiliary horizon": "#d08c41",
    "DLinear-Calibrated TRACER": "#b35806",
    "Conservative TRACER": "#8c2d04",
    "Selector-Conservative TRACER": "#c58a5c",
    "ATLASv2-Tuned TRACER Variant": "#c67d28",
    "ATLASv2-Tuned TRACER Core Transfer": "#c67d28",
    "Campaign-MEM": "#8c2d04",
    "Shared-Encoder Campaign-MEM": "#b86a3a",
    "Campaign-MEM w/o forecast calibration": "#c58a5c",
    "Campaign-MEM w/o utility": "#8c2d04",
    "Campaign-MEM w/o hard negatives": "#b35806",
    "Campaign-MEM w/o auxiliary horizon": "#c67d28",
    "Campaign-MEM w/ utility": "#c58a5c",
}

METHOD_MARKERS = {
    "LR-TailRisk": "s",
    "TCN-Forecaster": "o",
    "DLinear-Forecaster": "P",
    "LSTM-Forecaster": "v",
    "TimesNet-Forecaster": "h",
    "TiDE-Forecaster": ">",
    "TSMixer-Forecaster": "<",
    "PatchTST-Forecaster": "X",
    "Small-Transformer-Forecaster": "^",
    "iTransformer-Forecaster": "*",
    "No-Memory Forecaster": "D",
    "Pure-kNN-Retrieval": "P",
    "Prefix-Only-Retrieval + Fusion": "o",
    "Random-Retrieval + Fusion": "X",
    "TRACER": "D",
    "TRACER Core Mode": "D",
    "TRACER (adaptive policy)": "D",
    "Shared-Encoder TRACER": "o",
    "Event-Focused TRACER Variant": "s",
    "Chronology Support Line": "<",
    "Held-Out-Family Support Line": "^",
    "TRACER w/o auxiliary horizon": "v",
    "DLinear-Calibrated TRACER": "P",
    "Conservative TRACER": "D",
    "Selector-Conservative TRACER": "X",
    "ATLASv2-Tuned TRACER Variant": "D",
    "ATLASv2-Tuned TRACER Core Transfer": "D",
    "Campaign-MEM": "D",
    "Shared-Encoder Campaign-MEM": "o",
    "Campaign-MEM w/o forecast calibration": "s",
    "Campaign-MEM w/o utility": "D",
    "Campaign-MEM w/o hard negatives": "v",
    "Campaign-MEM w/o auxiliary horizon": "<",
    "Campaign-MEM w/ utility": ">",
}


def metric_axis_label(metric_key: str) -> str:
    mapping = {
        "AUPRC": "AUPRC",
        "LeadTime@P80": "LeadTime@P80 (min)",
        "Brier": "Brier (lower better)",
        "Analog-Fidelity@5": "AF@5",
        "TTE-Err@1": "TTE-Err@1 (min, lower better)",
    }
    return mapping.get(metric_key, metric_key)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png")


def add_bar_labels(ax: plt.Axes, values: list[float], errors: list[float] | None = None, decimals: int = 3) -> None:
    if not values:
        return
    max_value = max(values)
    pad = max(0.015 * max_value, 0.01)
    for index, value in enumerate(values):
        error = 0.0 if errors is None else errors[index]
        label = f"{value:.{decimals}f}"
        if errors is not None and error > 0:
            label = f"{value:.{decimals}f}+/-{error:.{decimals}f}"
        ax.text(value + error + pad, index, label, va="center", ha="left", fontsize=8)


def add_row_bands(ax: plt.Axes, n_rows: int, color: str = "#f2f4f8") -> None:
    for row in range(n_rows):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color=color, zorder=0)


def valid_metric_rows(rows: list[dict[str, object]], metric_key: str) -> list[dict[str, object]]:
    return [row for row in rows if row["metrics"].get(metric_key) is not None]


def metric_label_pad(rows: list[dict[str, object]], metric_key: str, floor: float = 0.01) -> float:
    values: list[float] = []
    for row in rows:
        metric_value = row["metrics"].get(metric_key)
        metric_std = row["std"].get(metric_key)
        if metric_value is None:
            continue
        values.append(float(metric_value) + float(metric_std or 0.0))
    if not values:
        return floor
    max_value = max(values)
    return max(0.018 * max(1.0, max_value), floor)


METHOD_LABELS = {
    "LR-TailRisk": "LR-TailRisk",
    "TCN-Forecaster": "TCN",
    "DLinear-Forecaster": "DLinear",
    "LSTM-Forecaster": "LSTM",
    "TimesNet-Forecaster": "TimesNet",
    "TiDE-Forecaster": "TiDE",
    "TSMixer-Forecaster": "TSMixer",
    "PatchTST-Forecaster": "PatchTST",
    "Small-Transformer-Forecaster": "Transformer",
    "iTransformer-Forecaster": "iTransformer",
    "No-Memory Forecaster": "No-Memory",
    "Pure-kNN-Retrieval": "Pure-kNN",
    "Prefix-Only-Retrieval + Fusion": "Prefix-Only",
    "Random-Retrieval + Fusion": "Random",
    "TRACER": "TRACER",
    "TRACER Core Mode": "TRACER core",
    "TRACER (adaptive policy)": "TRACER policy",
    "Shared-Encoder TRACER": "Shared-Encoder",
    "Event-Focused TRACER Variant": "event support",
    "Chronology Support Line": "chrono support",
    "Held-Out-Family Support Line": "family support",
    "TRACER w/o auxiliary horizon": "TRACER w/o aux",
    "DLinear-Calibrated TRACER": "DLinear-Calibrated TRACER",
    "Conservative TRACER": "Conservative TRACER",
    "Selector-Conservative TRACER": "Selector-Conservative TRACER",
    "ATLASv2-Tuned TRACER Variant": "ATLASv2-Tuned TRACER",
    "ATLASv2-Tuned TRACER Core Transfer": "ATLASv2-Tuned TRACER core transfer",
    "Campaign-MEM": "TRACER core",
    "Shared-Encoder Campaign-MEM": "Shared-Encoder TRACER",
    "Campaign-MEM w/o forecast calibration": "w/o calib",
    "Campaign-MEM w/o utility": "Campaign-MEM",
    "Campaign-MEM w/o hard negatives": "w/o hard neg",
    "Campaign-MEM w/o auxiliary horizon": "w/o aux",
    "Campaign-MEM w/ utility": "w/ utility",
}


def method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def style_axis(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", colors=TEXT)


def highlight_method_band(ax: plt.Axes, y: float, color: str = "#ffffff") -> None:
    return None


def add_panel_header(ax: plt.Axes, title: str) -> None:
    return None


SCIENTIFIC_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "scientific_sequential",
    ["#fbfbfb", "#dbe3ea", "#9eb4c6", "#5f7f97", "#2d4f67"],
)
