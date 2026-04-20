from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .atlasv2 import ALERT_TYPES, collapse_alert_type


HIGH_RISK_STAGES = {"priv_esc", "cred_access", "lateral_move", "collection_exfil", "impact"}


def _load_events(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if input_path.suffix.lower() == ".csv":
        frame = pd.read_csv(input_path)
    elif input_path.suffix.lower() == ".jsonl":
        frame = pd.read_json(input_path, lines=True)
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"timestamp", "incident_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    frame["incident_id"] = frame["incident_id"].astype(str)
    frame["family_id"] = frame.get("family_id", frame["incident_id"]).astype(str)
    frame["severity"] = pd.to_numeric(frame.get("severity", 0.0), errors="coerce").fillna(0.0)
    if "alert_type" not in frame.columns:
        if "report_text" in frame.columns:
            frame["alert_type"] = frame["report_text"].map(collapse_alert_type)
        else:
            frame["alert_type"] = "other"
    frame["alert_type"] = frame["alert_type"].fillna("other").astype(str).str.lower().map(
        lambda value: value if value in ALERT_TYPES else collapse_alert_type(value)
    )
    if "is_high_risk" in frame.columns:
        frame["is_high_risk"] = frame["is_high_risk"].astype(bool)
    elif "stage" in frame.columns:
        frame["stage"] = frame["stage"].fillna("").astype(str).str.lower()
        frame["is_high_risk"] = frame["stage"].isin(HIGH_RISK_STAGES)
    else:
        frame["is_high_risk"] = False
    frame["host_hash"] = frame.get("host_hash", "").fillna("").astype(str)
    return frame


def _build_windows(
    incident_events: pd.DataFrame,
    lookback_bins: int,
    main_horizon_bins: int,
    aux_horizon_bins: int,
    bin_minutes: int,
) -> list[dict[str, Any]]:
    incident_events = incident_events.sort_values("timestamp").copy()
    binned = incident_events.set_index("timestamp")
    category_counts = (
        pd.get_dummies(binned["alert_type"])
        .resample(f"{bin_minutes}min")
        .sum()
        .reindex(columns=ALERT_TYPES, fill_value=0)
    )
    host_count = (
        binned["host_hash"]
        .replace("", np.nan)
        .groupby(pd.Grouper(freq=f"{bin_minutes}min"))
        .nunique()
        .fillna(0)
        .rename("host_count")
    )
    stats = pd.DataFrame(
        {
            "event_count": binned["severity"].resample(f"{bin_minutes}min").count(),
            "severity_mean": binned["severity"].resample(f"{bin_minutes}min").mean(),
            "severity_max": binned["severity"].resample(f"{bin_minutes}min").max(),
            "high_risk_count": binned["is_high_risk"].astype(int).resample(f"{bin_minutes}min").sum(),
        }
    ).fillna(0.0)
    feature_frame = pd.concat([category_counts, stats, host_count], axis=1).fillna(0.0)
    if len(feature_frame) < lookback_bins + 2:
        return []

    high_risk_bins = feature_frame.index[feature_frame["high_risk_count"] > 0]
    onset_positions = []
    previous_position = None
    for timestamp in high_risk_bins:
        position = int(feature_frame.index.get_loc(timestamp))
        if previous_position is None or position - previous_position >= 3:
            onset_positions.append(position)
        previous_position = position

    feature_columns = ALERT_TYPES + ["event_count", "severity_mean", "severity_max", "high_risk_count", "host_count"]
    feature_values = feature_frame[feature_columns].to_numpy(dtype=np.float32)
    windows = []
    incident_id = incident_events["incident_id"].iloc[0]
    family_id = incident_events["family_id"].iloc[0]
    for current in range(lookback_bins - 1, len(feature_frame) - 1):
        future_onsets = [position for position in onset_positions if position > current]
        next_onset = future_onsets[0] if future_onsets else None
        delta_bins = (next_onset - current) if next_onset is not None else None
        label_main = 1.0 if delta_bins is not None and delta_bins <= main_horizon_bins else 0.0
        label_aux = 1.0 if delta_bins is not None and delta_bins <= aux_horizon_bins else 0.0
        time_to_escalation = float(delta_bins * bin_minutes) if delta_bins is not None else float((main_horizon_bins + 3) * bin_minutes)

        prefix = feature_values[current - lookback_bins + 1 : current + 1].copy()
        future_slice = feature_values[current + 1 : current + 1 + main_horizon_bins]
        if prefix.shape[0] != lookback_bins or future_slice.shape[0] == 0:
            continue
        future_category = future_slice[:, : len(ALERT_TYPES)].sum(axis=0)
        future_event_count = future_slice[:, len(ALERT_TYPES)].sum()
        future_signature = np.asarray(
            [
                min(time_to_escalation, main_horizon_bins * bin_minutes) / float(main_horizon_bins * bin_minutes),
                future_slice[:, len(ALERT_TYPES) + 1].mean(),
                future_slice[:, len(ALERT_TYPES) + 2].max(),
                future_slice[:, len(ALERT_TYPES) + 3].sum() / float(max(main_horizon_bins, 1)),
                future_slice[:, len(ALERT_TYPES) + 4].max(),
                future_category[ALERT_TYPES.index("execution")] / float(max(future_event_count, 1.0)),
                future_category[ALERT_TYPES.index("lateral_move")] / float(max(future_event_count, 1.0)),
                future_category[ALERT_TYPES.index("cred_access")] / float(max(future_event_count, 1.0)),
            ],
            dtype=np.float32,
        )
        windows.append(
            {
                "prefix": prefix,
                "label_main": label_main,
                "label_aux": label_aux,
                "future_signature": future_signature,
                "time_to_escalation": time_to_escalation,
                "incident_id": incident_id,
                "family_id": family_id,
                "timestamp": int(feature_frame.index[current].timestamp()),
            }
        )
    return windows


def _to_payload(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "prefix": np.stack([row["prefix"] for row in rows]).astype(np.float32),
        "label_main": np.asarray([row["label_main"] for row in rows], dtype=np.float32),
        "label_aux": np.asarray([row["label_aux"] for row in rows], dtype=np.float32),
        "future_signature": np.stack([row["future_signature"] for row in rows]).astype(np.float32),
        "time_to_escalation": np.asarray([row["time_to_escalation"] for row in rows], dtype=np.float32),
        "incident_id": np.asarray([row["incident_id"] for row in rows]),
        "family_id": np.asarray([row["family_id"] for row in rows]),
        "timestamp": np.asarray([row["timestamp"] for row in rows], dtype=np.int64),
    }


def prepare_canonical_alert_dataset(config: dict[str, Any]) -> dict[str, Any]:
    input_path = Path(config["input_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_minutes = int(config.get("bin_minutes", 5))
    lookback_bins = int(config.get("lookback_bins", 12))
    main_horizon_bins = int(config.get("main_horizon_bins", 6))
    aux_horizon_bins = int(config.get("aux_horizon_bins", 2))
    split_ratios = config.get("split_ratios", {"train": 0.6, "dev": 0.2, "test": 0.2})

    events = _load_events(input_path)
    windows = []
    incident_stats = []
    for incident_id, incident_events in events.groupby("incident_id"):
        rows = _build_windows(
            incident_events=incident_events,
            lookback_bins=lookback_bins,
            main_horizon_bins=main_horizon_bins,
            aux_horizon_bins=aux_horizon_bins,
            bin_minutes=bin_minutes,
        )
        if not rows:
            continue
        windows.extend(rows)
        incident_stats.append(
            {
                "incident_id": incident_id,
                "family_id": incident_events["family_id"].iloc[0],
                "events": int(len(incident_events)),
                "windows": int(len(rows)),
                "high_risk_events": int(incident_events["is_high_risk"].sum()),
            }
        )
    if not windows:
        raise ValueError("No windows generated from canonical alert input.")

    incidents = {}
    for row in windows:
        incidents.setdefault(row["incident_id"], []).append(row)
    ordered = sorted(incidents.items(), key=lambda item: min(entry["timestamp"] for entry in item[1]))
    count = len(ordered)
    train_cut = max(1, int(round(count * split_ratios.get("train", 0.6))))
    dev_cut = max(train_cut + 1, int(round(count * (split_ratios.get("train", 0.6) + split_ratios.get("dev", 0.2)))))
    split_map = {
        "train": ordered[:train_cut],
        "dev": ordered[train_cut:dev_cut],
        "test": ordered[dev_cut:],
    }
    for split_name, groups in split_map.items():
        rows = [entry for _, incident_rows in groups for entry in incident_rows]
        if rows:
            np.savez(output_dir / f"{split_name}.npz", **_to_payload(rows))

    metadata = {
        "dataset_name": config.get("dataset_name", input_path.stem),
        "description": "Canonical alert-event table projected into Campaign-MEM rolling windows.",
        "input_path": str(input_path),
        "bin_minutes": bin_minutes,
        "lookback_bins": lookback_bins,
        "main_horizon_minutes": main_horizon_bins * bin_minutes,
        "aux_horizon_minutes": aux_horizon_bins * bin_minutes,
        "feature_channels": ALERT_TYPES + ["event_count", "severity_mean", "severity_max", "high_risk_count", "host_count"],
        "analog_fidelity_distance_threshold": float(config.get("analog_fidelity_distance_threshold", 0.45)),
        "incidents": incident_stats,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata
