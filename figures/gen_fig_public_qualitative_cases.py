from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paper_plot_style import save_figure, style_axis

ROOT = Path(__file__).resolve().parents[1]
DIAGNOSIS_PATH = ROOT / "outputs" / "results" / "r014_atlasv2_public_case_diagnosis.json"
MAX_SCORE = 0.48


def _wrap(text: str, width: int) -> str:
    return fill(str(text), width=width)


def _short_id(identifier: str) -> str:
    text = str(identifier)
    if text.startswith("atlasv2/"):
        text = text.split("/", 1)[1]
    return text


def _case_meta(case: dict) -> list[tuple[str, str]]:
    label_text = "positive" if int(case["label_main"]) == 1 else "negative"
    return [
        ("Incident", _short_id(case["incident_id"])),
        ("Family", case["family_id"].split("/")[-1]),
        ("Time", case["timestamp_iso"].replace("+00:00", " UTC")),
        ("Label", label_text),
    ]


def _draw_left_card(ax: plt.Axes, title: str, case: dict, color: str) -> None:
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="#d8dde6",
            linewidth=0.9,
            zorder=0,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            0.014,
            1.0,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            linewidth=0.0,
            zorder=1,
        )
    )

    ax.text(0.035, 0.91, title, fontsize=10.0, fontweight="bold", color=color, ha="left", va="top", transform=ax.transAxes)
    label_text = "positive" if int(case["label_main"]) == 1 else "negative"
    ax.text(
        0.04,
        0.76,
        f"{_short_id(case['incident_id'])} | {case['family_id'].split('/')[-1]} | {label_text}",
        fontsize=8.95,
        color="#111827",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.04,
        0.64,
        case["timestamp_iso"].replace("+00:00", " UTC"),
        fontsize=8.85,
        color="#374151",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    ax.plot([0.04, 0.96], [0.50, 0.50], transform=ax.transAxes, color="#d8dde6", linewidth=0.9, solid_capstyle="butt")
    ax.text(0.04, 0.44, "Score comparison", fontsize=8.6, color="#6b7280", ha="left", va="top", transform=ax.transAxes)
    ax.text(0.96, 0.44, f"delta {case['score_margin']:+.3f}", fontsize=9.2, color=color, fontweight="bold", ha="right", va="top", transform=ax.transAxes)

    score_ax = ax.inset_axes([0.19, 0.08, 0.75, 0.24])
    prefix_score = float(case["prefix_score"])
    campaign_score = float(case["campaign_score"])
    y_positions = [1.0, 0.0]
    labels = ["Prefix", "TRACER"]
    score_ax.barh([y_positions[0]], [prefix_score], height=0.26, color="white", edgecolor="#2d2d2d", linewidth=1.0, zorder=2)
    score_ax.barh([y_positions[1]], [campaign_score], height=0.26, color=color, edgecolor="#2d2d2d", linewidth=0.8, zorder=2)

    def annotate(value: float, y_loc: float, above: bool) -> None:
        x = min(value + 0.008, MAX_SCORE - 0.01)
        y = y_loc + 0.24 if above else y_loc - 0.24
        va = "bottom" if above else "top"
        score_ax.text(
            x,
            y,
            f"{value:.3f}",
            ha="left",
            va=va,
            fontsize=8.6,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
            zorder=4,
        )

    annotate(prefix_score, y_positions[0], above=True)
    annotate(campaign_score, y_positions[1], above=False)
    score_ax.set_xlim(0.0, MAX_SCORE)
    score_ax.set_ylim(-0.55, 1.55)
    score_ax.set_yticks(y_positions)
    score_ax.set_yticklabels(labels)
    style_axis(score_ax, grid_axis="x")
    score_ax.spines["left"].set_visible(False)
    score_ax.tick_params(axis="y", length=0, labelsize=8.3)
    score_ax.tick_params(axis="x", labelsize=8.1)
    score_ax.set_xlabel("Risk score", fontsize=8.5)


def _draw_right_card(ax: plt.Axes, case: dict, note: str, color: str) -> None:
    ax.axis("off")
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            1.0,
            1.0,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="#d8dde6",
            linewidth=0.9,
            zorder=0,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.0),
            0.012,
            1.0,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor=color,
            linewidth=0.0,
            zorder=1,
        )
    )

    changed = case["prefix_top1_incident_id"] != case["campaign_top1_incident_id"]
    ax.text(
        0.03,
        0.90,
        f"Top-1 analog: {'changed' if changed else 'unchanged'}",
        fontsize=9.5,
        fontweight="bold",
        color="#1f2937",
        ha="left",
        va="top",
        transform=ax.transAxes,
    )

    rows = [
        ("Prefix", _wrap(_short_id(case["prefix_top1_incident_id"]), width=26)),
        ("Campaign", _wrap(_short_id(case["campaign_top1_incident_id"]), width=26)),
        ("Future dist.", f"{case['prefix_top1_future_distance']:.2f} -> {case['campaign_top1_future_distance']:.2f}"),
        ("TTE error", f"{case['prefix_top1_tte_error']:.0f} -> {case['campaign_top1_tte_error']:.0f} min"),
    ]

    y = 0.76
    for label, value in rows:
        ax.text(0.03, y, label, fontsize=8.6, color="#6b7280", ha="left", va="top", transform=ax.transAxes)
        ax.text(0.22, y, value, fontsize=8.85, color="#111827", ha="left", va="top", transform=ax.transAxes, linespacing=1.20)
        y -= 0.135 if "\n" in value else 0.105

    separator_y = max(y - 0.015, 0.40)
    ax.plot([0.03, 0.97], [separator_y, separator_y], transform=ax.transAxes, color="#d8dde6", linewidth=0.9, solid_capstyle="butt")
    interpret_y = separator_y - 0.075
    ax.text(
        0.03,
        interpret_y,
        "Interpretation",
        fontsize=8.6,
        color="#6b7280",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.05},
    )
    ax.text(
        0.03,
        interpret_y - 0.075,
        _wrap(note, width=58),
        fontsize=8.7,
        color="#374151",
        ha="left",
        va="top",
        transform=ax.transAxes,
        linespacing=1.22,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.08},
    )


def main() -> None:
    payload = json.loads(DIAGNOSIS_PATH.read_text(encoding="utf-8"))
    panel_specs = [
        (
            "A. Small positive uplift",
            payload["cases"]["campaign_advantage"][0],
            "Both methods retrieve the same benign analog. The uplift comes from score fusion rather than a new retrieved future.",
            "#2a9d8f",
        ),
        (
            "B. Missed true positive",
            payload["cases"]["campaign_failure"][0],
            "The top-1 analog is unchanged, but TRACER suppresses the score too much. This is a scoring failure rather than a retrieval failure.",
            "#e76f51",
        ),
        (
            "C. False-positive uplift",
            payload["cases"]["campaign_false_positive"][0],
            "TRACER swaps in a more attack-like analog on a negative window and raises the score. This exposes retrieval noise under weak evidence.",
            "#f4a261",
        ),
    ]

    fig = plt.figure(figsize=(7.25, 8.95))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.10, 1.35], hspace=0.38, wspace=0.14)

    for row_index, (title, case, note, color) in enumerate(panel_specs):
        left_ax = fig.add_subplot(grid[row_index, 0])
        right_ax = fig.add_subplot(grid[row_index, 1])
        _draw_left_card(left_ax, title, case, color)
        _draw_right_card(right_ax, case, note, color)

    legend_handles = [
        Line2D([0], [0], color="#2d2d2d", linewidth=1.0, marker="s", markerfacecolor="white", markeredgecolor="#2d2d2d", markersize=7.5, label="Prefix"),
        Line2D([0], [0], color="#7f7f7f", linewidth=1.0, marker="s", markerfacecolor="#7f7f7f", markeredgecolor="#2d2d2d", markersize=7.5, label="TRACER"),
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        columnspacing=1.4,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.05, right=0.985, top=0.92, bottom=0.06)
    save_figure(fig, "fig_public_qualitative_cases")


if __name__ == "__main__":
    main()
