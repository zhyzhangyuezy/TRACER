from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def apply_stage_intervals(config: dict[str, Any]) -> dict[str, Any]:
    events_path = Path(config["events_path"])
    intervals_path = Path(config["intervals_path"])
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(events_path)
    intervals = pd.read_csv(intervals_path)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, format="mixed")
    intervals["start_time"] = pd.to_datetime(intervals["start_time"], utc=True, format="mixed")
    intervals["end_time"] = pd.to_datetime(intervals["end_time"], utc=True, format="mixed")

    if "family_id" not in events.columns:
        events["family_id"] = events["incident_id"]
    if "stage" not in events.columns:
        events["stage"] = ""
    else:
        events["stage"] = events["stage"].fillna("").astype(object)
    if "is_high_risk" not in events.columns:
        events["is_high_risk"] = False

    intervals = intervals.sort_values(["incident_id", "start_time"]).reset_index(drop=True)
    labeled_count = 0
    for interval in intervals.to_dict("records"):
        mask = (
            (events["incident_id"].astype(str) == str(interval["incident_id"]))
            & (events["timestamp"] >= interval["start_time"])
            & (events["timestamp"] <= interval["end_time"])
        )
        if "family_id" in interval and pd.notna(interval["family_id"]):
            events.loc[mask, "family_id"] = str(interval["family_id"])
        if "stage" in interval and pd.notna(interval["stage"]):
            events.loc[mask, "stage"] = str(interval["stage"])
        if "is_high_risk" in interval and pd.notna(interval["is_high_risk"]):
            high_risk = interval["is_high_risk"]
            if isinstance(high_risk, str):
                high_risk = high_risk.strip().lower() in {"1", "true", "yes", "y"}
            else:
                high_risk = bool(high_risk)
            events.loc[mask, "is_high_risk"] = high_risk
        labeled_count += int(mask.sum())

    events = events.sort_values(["incident_id", "timestamp"]).reset_index(drop=True)
    events.to_csv(output_path, index=False)
    return {
        "output_path": str(output_path),
        "events": int(len(events)),
        "intervals": int(len(intervals)),
        "labeled_event_rows": int(labeled_count),
    }
