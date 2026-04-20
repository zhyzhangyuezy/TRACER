from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from paper_plot_style import METHOD_COLORS, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "results" / "atlasv2_warning_visuals_seed7.json"

MODEL_ORDER = [
    ("r020_dlinear_forecaster_atlasv2_public", "DLinear-Forecaster"),
    ("r008_prefix_retrieval_atlasv2_public", "Prefix-Only-Retrieval + Fusion"),
    ("r082_campaign_mem_selector_staged_atlasv2_public", "TRACER support variant"),
]


def _load_runs() -> dict[str, dict]:
    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {run["experiment_name"]: run for run in payload["runs"]}


def _split_frame(run: dict, split_name: str) -> pd.DataFrame:
    split = run[split_name]["predictions"]
    frame = pd.DataFrame(
        {
            "incident_id": split["incident_id"],
            "family_id": split["family_id"],
            "timestamp": pd.to_datetime(split["timestamp"], unit="s", utc=True),
            "y_true": split["y_true"],
            "y_score": split["y_score"],
            "time_to_escalation": split["time_to_escalation"],
        }
    )
    return frame.sort_values(["incident_id", "timestamp"])


def _estimate_onset(frame: pd.DataFrame) -> pd.Timestamp | None:
    positive = frame[frame["y_true"] == 1].copy()
    if positive.empty:
        return None
    onset_candidates = positive["timestamp"] + pd.to_timedelta(positive["time_to_escalation"], unit="m")
    return onset_candidates.min()


def _choose_case(
    campaign: pd.DataFrame,
    prefix: pd.DataFrame,
    dlinear: pd.DataFrame,
    split_name: str,
) -> str:
    merged = campaign.merge(
        prefix[["incident_id", "timestamp", "y_score"]].rename(columns={"y_score": "prefix_score"}),
        on=["incident_id", "timestamp"],
        how="inner",
    ).merge(
        dlinear[["incident_id", "timestamp", "y_score"]].rename(columns={"y_score": "dlinear_score"}),
        on=["incident_id", "timestamp"],
        how="inner",
    )
    merged = merged[merged["y_true"] == 1].copy()
    if merged.empty:
        return str(campaign["incident_id"].iloc[0])
    merged["margin"] = merged["y_score"] - merged[["prefix_score", "dlinear_score"]].max(axis=1)
    incident_margin = merged.groupby("incident_id")["margin"].mean().sort_values(ascending=False)
    return str(incident_margin.index[0])


def _add_positive_band(ax: plt.Axes, frame: pd.DataFrame) -> None:
    positive = frame[frame["y_true"] == 1]
    if positive.empty:
        return
    start = positive["timestamp"].min()
    end = positive["timestamp"].max()
    ax.axvspan(start, end, color="#f3e6dc", alpha=0.55, linewidth=0.0)


def _plot_case(ax: plt.Axes, runs: dict[str, dict], split_name: str, incident_id: str, title: str) -> None:
    reference_frame: pd.DataFrame | None = None
    for experiment_name, display_name in MODEL_ORDER:
        frame = _split_frame(runs[experiment_name], split_name)
        frame = frame[frame["incident_id"] == incident_id].sort_values("timestamp")
        if reference_frame is None:
            reference_frame = frame
        ax.plot(
            frame["timestamp"],
            frame["y_score"],
            label=display_name,
            color=METHOD_COLORS.get(display_name, METHOD_COLORS["TRACER"]),
            linewidth=2.2 if "TRACER" in display_name else 1.7,
        )
    assert reference_frame is not None
    _add_positive_band(ax, reference_frame)
    onset = _estimate_onset(reference_frame)
    if onset is not None:
        ax.axvline(onset, color="#111111", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.annotate(
            "Escalation onset",
            xy=(onset, 0.98),
            xytext=(6, -6),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8.8,
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    ax.set_ylabel("Risk score")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylim(0.0, 1.02)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    style_axis(ax, grid_axis="y")
    ax.set_title(title, loc="left", pad=5)


def main() -> None:
    runs = _load_runs()
    campaign_test = _split_frame(runs["r082_campaign_mem_selector_staged_atlasv2_public"], "test")
    prefix_test = _split_frame(runs["r008_prefix_retrieval_atlasv2_public"], "test")
    dlinear_test = _split_frame(runs["r020_dlinear_forecaster_atlasv2_public"], "test")
    campaign_event = _split_frame(runs["r082_campaign_mem_selector_staged_atlasv2_public"], "test_event_disjoint")
    prefix_event = _split_frame(runs["r008_prefix_retrieval_atlasv2_public"], "test_event_disjoint")
    dlinear_event = _split_frame(runs["r020_dlinear_forecaster_atlasv2_public"], "test_event_disjoint")

    chrono_case = _choose_case(campaign_test, prefix_test, dlinear_test, "test")
    heldout_case = _choose_case(campaign_event, prefix_event, dlinear_event, "test_event_disjoint")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.9))
    _plot_case(
        axes[0],
        runs,
        "test",
        chrono_case,
        f"Chronological early-warning case: {chrono_case.replace('atlasv2/', '')}",
    )
    _plot_case(
        axes[1],
        runs,
        "test_event_disjoint",
        heldout_case,
        f"Held-out-family early-warning case: {heldout_case.replace('atlasv2/', '')}",
    )
    axes[0].text(-0.08, 1.03, "A", transform=axes[0].transAxes, fontsize=11.4, fontweight="bold")
    axes[1].text(-0.08, 1.03, "B", transform=axes[1].transAxes, fontsize=11.4, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    handles_final = list(dedup.values()) + [Patch(facecolor="#f3e6dc", edgecolor="none", alpha=0.55, label="Positive window")]
    labels_final = list(dedup.keys()) + ["Positive window"]
    fig.legend(handles_final, labels_final, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.11, top=0.84, hspace=0.60)
    save_figure(fig, "fig_public_warning_timelines")


if __name__ == "__main__":
    main()
