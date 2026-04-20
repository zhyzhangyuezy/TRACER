from __future__ import annotations

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from paper_plot_style import METHOD_COLORS, save_figure, style_axis


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs" / "results" / "ait_ads_prediction_visuals_seed7.json"

MODEL_ORDER = [
    ("r068_dlinear_forecaster_ait_ads_public", "DLinear-Forecaster"),
    ("r071_prefix_retrieval_ait_ads_public", "Prefix-Only-Retrieval + Fusion"),
    ("r117_campaign_mem_dual_selector_proxy_strict_ait_ads_public", "TRACER"),
]
def _load_runs() -> dict[str, dict]:
    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {run["experiment_name"]: run for run in payload["runs"]}


def _split_frame(run: dict, split_name: str) -> pd.DataFrame:
    split = run[split_name]["predictions"]
    return pd.DataFrame(
        {
            "incident_id": split["incident_id"],
            "family_id": split["family_id"],
            "timestamp": pd.to_datetime(split["timestamp"], unit="s", utc=True),
            "y_true": split["y_true"],
            "y_score": split["y_score"],
            "time_to_escalation": split["time_to_escalation"],
        }
    )


def _choose_case(campaign: pd.DataFrame, prefix: pd.DataFrame, maximize: bool) -> str:
    merged = campaign.merge(
        prefix[["incident_id", "timestamp", "y_score"]].rename(columns={"y_score": "prefix_score"}),
        on=["incident_id", "timestamp"],
        how="inner",
    )
    merged = merged[merged["y_true"] == 1].copy()
    merged["score_delta"] = merged["y_score"] - merged["prefix_score"]
    incident_gain = merged.groupby("incident_id")["score_delta"].mean().sort_values(ascending=not maximize)
    if incident_gain.empty:
        return str(campaign["incident_id"].iloc[0])
    return str(incident_gain.index[0])


def _focus_window(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    positives = frame[frame["y_true"] == 1].sort_values("timestamp")
    if positives.empty:
        onset = frame["timestamp"].iloc[len(frame) // 2]
        return onset - pd.Timedelta(hours=3), onset + pd.Timedelta(hours=8), onset
    onset = positives["timestamp"].iloc[0]
    start = max(frame["timestamp"].min(), onset - pd.Timedelta(hours=3))
    end = min(frame["timestamp"].max(), onset + pd.Timedelta(hours=8))
    return start, end, onset


def _positive_spans(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    positives = frame[frame["y_true"] == 1].sort_values("timestamp")
    if positives.empty:
        return []
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = positives["timestamp"].iloc[0]
    prev = start
    step = positives["timestamp"].diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(minutes=5)
    for timestamp in positives["timestamp"].iloc[1:]:
        if timestamp - prev > step * 1.5:
            spans.append((start, prev + step))
            start = timestamp
        prev = timestamp
    spans.append((start, prev + step))
    return spans


def _plot_case(plot_ax: plt.Axes, runs: dict[str, dict], split_name: str, incident_id: str, title: str, panel_label: str, show_xlabel: bool) -> None:
    campaign_frame = _split_frame(runs["r117_campaign_mem_dual_selector_proxy_strict_ait_ads_public"], split_name)
    campaign_frame = campaign_frame[campaign_frame["incident_id"] == incident_id].sort_values("timestamp")
    start, end, onset = _focus_window(campaign_frame)

    for span_start, span_end in _positive_spans(campaign_frame):
        clipped_start = max(span_start, start)
        clipped_end = min(span_end, end)
        if clipped_end > clipped_start:
            plot_ax.axvspan(clipped_start, clipped_end, color="#f9d7d3", alpha=0.45, linewidth=0.0, zorder=0)

    for experiment_name, display_name in MODEL_ORDER:
        frame = _split_frame(runs[experiment_name], split_name)
        frame = frame[(frame["incident_id"] == incident_id) & (frame["timestamp"] >= start) & (frame["timestamp"] <= end)].sort_values("timestamp")
        plot_ax.plot(
            frame["timestamp"],
            frame["y_score"],
            label=display_name,
            color=METHOD_COLORS.get(display_name, "#444444"),
            linewidth=2.2 if display_name == "TRACER" else 1.6,
            alpha=0.98,
        )

    plot_ax.axvline(onset, color="#2c2c2c", linestyle="--", linewidth=1.1)
    plot_ax.set_ylabel("Risk score")
    plot_ax.set_ylim(0.0, 1.02)
    plot_ax.set_xlim(start, end)
    plot_ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=7))
    plot_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    if show_xlabel:
        plot_ax.set_xlabel("Time (UTC)")
    else:
        plot_ax.set_xlabel("")
        plot_ax.tick_params(axis="x", labelbottom=False)
    style_axis(plot_ax, grid_axis="y")
    plot_ax.text(-0.055, 1.04, panel_label, transform=plot_ax.transAxes, fontsize=11.2, fontweight="bold", ha="left", va="bottom")
    plot_ax.text(0.02, 1.04, title, transform=plot_ax.transAxes, fontsize=10.8, ha="left", va="bottom")


def main() -> None:
    runs = _load_runs()

    campaign_test = _split_frame(runs["r117_campaign_mem_dual_selector_proxy_strict_ait_ads_public"], "test")
    prefix_test = _split_frame(runs["r071_prefix_retrieval_ait_ads_public"], "test")
    campaign_event = _split_frame(runs["r117_campaign_mem_dual_selector_proxy_strict_ait_ads_public"], "test_event_disjoint")
    prefix_event = _split_frame(runs["r071_prefix_retrieval_ait_ads_public"], "test_event_disjoint")

    chrono_case = _choose_case(campaign_test, prefix_test, maximize=True)
    heldout_case = _choose_case(campaign_event, prefix_event, maximize=False)

    fig = plt.figure(figsize=(7.25, 5.65))
    grid = fig.add_gridspec(2, 1, hspace=0.28)
    ax_1 = fig.add_subplot(grid[0, 0])
    ax_2 = fig.add_subplot(grid[1, 0])

    _plot_case(ax_1, runs, "test", chrono_case, f"Chronological case: {chrono_case.replace('aitads/', '')}", "A", False)
    _plot_case(ax_2, runs, "test_event_disjoint", heldout_case, f"Scenario-held-out case: {heldout_case.replace('aitads/', '')}", "B", True)

    model_handles = [
        Line2D([0], [0], color=METHOD_COLORS["DLinear-Forecaster"], lw=1.8, label="DLinear"),
        Line2D([0], [0], color=METHOD_COLORS["Prefix-Only-Retrieval + Fusion"], lw=1.8, label="Prefix-Only"),
        Line2D([0], [0], color=METHOD_COLORS["TRACER"], lw=2.4, label="TRACER"),
        Line2D([0], [0], color="#2c2c2c", lw=1.1, linestyle="--", label="Escalation onset"),
        Line2D([0], [0], color="#f9d7d3", lw=8.0, alpha=0.8, label="Positive window"),
    ]
    fig.legend(
        handles=model_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        columnspacing=0.9,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.11, top=0.85, hspace=0.28)
    save_figure(fig, "fig_ait_ads_attack_timelines")


if __name__ == "__main__":
    main()
