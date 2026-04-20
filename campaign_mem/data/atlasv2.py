from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ALERT_TYPES = [
    "recon",
    "auth_abuse",
    "execution",
    "persistence",
    "priv_esc",
    "cred_access",
    "lateral_move",
    "c2",
    "collection_exfil",
    "impact",
    "defense_evasion",
    "other",
]

ALERT_PATTERNS = {
    "recon": [r"recon", r"scan", r"discovery", r"enumerat", r"packet capture"],
    "auth_abuse": [r"auth", r"credential", r"brute", r"login", r"password"],
    "execution": [r"execution", r"command", r"script", r"powershell", r"cmd\.exe", r"interpreter"],
    "persistence": [r"persist", r"service", r"scheduled", r"registry", r"startup"],
    "priv_esc": [r"privilege", r"uac", r"elevation", r"sudo", r"token manipulation"],
    "cred_access": [r"lsass", r"credential access", r"token", r"dump", r"key"],
    "lateral_move": [r"lateral", r"remote", r"rdp", r"smb", r"winrm", r"dcom", r"psexec"],
    "c2": [r"command and control", r"beacon", r"c2", r"callback"],
    "collection_exfil": [r"collection", r"exfil", r"archive", r"compress"],
    "impact": [r"impact", r"encrypt", r"delete", r"destroy", r"wiper"],
    "defense_evasion": [r"defense evasion", r"masquerad", r"disable", r"obfus", r"hide"],
}


def collapse_alert_type(text: str) -> str:
    normalized = str(text or "").lower()
    for alert_type, patterns in ALERT_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return alert_type
    return "other"


def _normalize_workbook_rows(workbook_path: str | Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(workbook_path)
    frames = []
    for sheet_name in workbook.sheet_names:
        frame = workbook.parse(sheet_name).copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        if "" in frame.columns:
            frame = frame.rename(columns={"": "report_text"})
        elif " " in frame.columns:
            frame = frame.rename(columns={" ": "report_text"})
        if "parent_path" not in frame.columns and "report" in frame.columns and "report_text" in frame.columns:
            frame = frame.rename(columns={"report": "parent_path"})
        if "report_text" not in frame.columns and "report" in frame.columns:
            frame = frame.rename(columns={"report": "report_text"})
        frame["sheet_name"] = sheet_name
        frames.append(frame)
    events = pd.concat(frames, ignore_index=True)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True, format="mixed")
    events["attack_window"] = events["attack_window"].astype(str).str.lower()
    events["host"] = events["host"].astype(str).str.lower()
    events["incident_id"] = "atlasv2/" + events["host"] + "-" + events["attack_window"]
    events["family_id"] = "atlasv2/" + events["attack_window"]
    events["report_text"] = events["report_text"].fillna("").astype(str)
    events["severity"] = pd.to_numeric(events["severity"], errors="coerce").fillna(0.0)
    events["label"] = events["label"].fillna("benign").astype(str).str.lower()
    events["alert_type"] = events["report_text"].map(collapse_alert_type)
    return events


def _load_reapr_summary(reapr_csv_path: str | Path | None) -> dict[str, Any]:
    if not reapr_csv_path:
        return {}
    path = Path(reapr_csv_path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "attack" not in frame.columns or "label" not in frame.columns:
        return {}
    summary = (
        frame.groupby(["attack", "label"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
        .astype(int)
    )
    return {
        attack: {label: int(count) for label, count in row.items()}
        for attack, row in summary.iterrows()
    }


def _build_incident_windows(
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
    severity_stats = pd.DataFrame(
        {
            "event_count": binned["severity"].resample(f"{bin_minutes}min").count(),
            "severity_mean": binned["severity"].resample(f"{bin_minutes}min").mean(),
            "severity_max": binned["severity"].resample(f"{bin_minutes}min").max(),
            "malicious_count": (binned["label"] == "malicious").astype(int).resample(f"{bin_minutes}min").sum(),
            "artifact_count": (binned["label"] == "artifact").astype(int).resample(f"{bin_minutes}min").sum(),
        }
    ).fillna(0.0)
    feature_frame = pd.concat([category_counts, severity_stats], axis=1).fillna(0.0)
    if len(feature_frame) < lookback_bins + 2:
        return []

    malicious_bins = feature_frame.index[feature_frame["malicious_count"] > 0]
    onset_positions = []
    previous_position = None
    for timestamp in malicious_bins:
        position = int(feature_frame.index.get_loc(timestamp))
        if previous_position is None or position - previous_position >= 3:
            onset_positions.append(position)
        previous_position = position

    windows = []
    incident_id = incident_events["incident_id"].iloc[0]
    family_id = incident_events["family_id"].iloc[0]
    host = incident_events["host"].iloc[0]
    host_flag = np.array([1.0, 0.0], dtype=np.float32) if host == "h1" else np.array([0.0, 1.0], dtype=np.float32)

    feature_columns = ALERT_TYPES + ["event_count", "severity_mean", "severity_max", "malicious_count", "artifact_count"]
    feature_values = feature_frame[feature_columns].to_numpy(dtype=np.float32)

    for current in range(lookback_bins - 1, len(feature_frame) - 1):
        future_onsets = [position for position in onset_positions if position > current]
        next_onset = future_onsets[0] if future_onsets else None
        delta_bins = (next_onset - current) if next_onset is not None else None
        label_main = 1.0 if delta_bins is not None and delta_bins <= main_horizon_bins else 0.0
        label_aux = 1.0 if delta_bins is not None and delta_bins <= aux_horizon_bins else 0.0
        time_to_escalation = float(delta_bins * bin_minutes) if delta_bins is not None else float(main_horizon_bins * bin_minutes + 15)

        prefix = feature_values[current - lookback_bins + 1 : current + 1].copy()
        if prefix.shape[0] != lookback_bins:
            continue
        prefix = np.concatenate([prefix, np.repeat(host_flag[None, :], lookback_bins, axis=0)], axis=1)

        future_slice = feature_values[current + 1 : current + 1 + main_horizon_bins]
        if future_slice.shape[0] == 0:
            continue
        future_category = future_slice[:, : len(ALERT_TYPES)].sum(axis=0)
        future_event_count = future_slice[:, len(ALERT_TYPES)].sum()
        future_severity_mean = future_slice[:, len(ALERT_TYPES) + 1].mean()
        future_malicious = future_slice[:, len(ALERT_TYPES) + 3].sum()
        future_artifact = future_slice[:, len(ALERT_TYPES) + 4].sum()
        time_to_peak = int(np.argmax(future_slice[:, len(ALERT_TYPES) + 1])) + 1
        future_signature = np.asarray(
            [
                min(time_to_escalation, main_horizon_bins * bin_minutes) / float(main_horizon_bins * bin_minutes),
                time_to_peak / float(max(main_horizon_bins, 1)),
                future_severity_mean,
                future_malicious / float(max(main_horizon_bins, 1)),
                future_artifact / float(max(main_horizon_bins, 1)),
                future_event_count / float(max(main_horizon_bins, 1)),
                future_category[ALERT_TYPES.index("execution")] / float(max(future_event_count, 1.0)),
                future_category[ALERT_TYPES.index("lateral_move")] / float(max(future_event_count, 1.0)),
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


def _split_rows_chronologically(rows: list[dict[str, Any]], ratios: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: row["timestamp"])
    count = len(ordered)
    train_cut = max(1, int(round(count * ratios.get("train", 0.6))))
    dev_cut = max(train_cut + 1, int(round(count * (ratios.get("train", 0.6) + ratios.get("dev", 0.2)))))
    return {
        "train": ordered[:train_cut],
        "dev": ordered[train_cut:dev_cut],
        "test": ordered[dev_cut:],
    }


def _split_incidents(windows: list[dict[str, Any]], ratios: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    incidents: dict[str, list[dict[str, Any]]] = {}
    for row in windows:
        incidents.setdefault(row["incident_id"], []).append(row)

    benign_incidents = []
    attack_incidents = []
    for incident_id, rows in incidents.items():
        if rows[0]["family_id"].endswith("/benign"):
            benign_incidents.append((incident_id, rows))
        else:
            attack_incidents.append((incident_id, rows))
    attack_incidents = sorted(attack_incidents, key=lambda item: min(entry["timestamp"] for entry in item[1]))

    attack_count = len(attack_incidents)
    train_cut = max(1, int(round(attack_count * ratios.get("train", 0.6))))
    dev_cut = max(train_cut + 1, int(round(attack_count * (ratios.get("train", 0.6) + ratios.get("dev", 0.2)))))
    split_map: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for split_name, groups in {
        "train": attack_incidents[:train_cut],
        "dev": attack_incidents[train_cut:dev_cut],
        "test": attack_incidents[dev_cut:],
    }.items():
        split_map[split_name].extend(entry for _, rows in groups for entry in rows)

    for _, rows in benign_incidents:
        benign_splits = _split_rows_chronologically(rows, ratios)
        for split_name, split_rows in benign_splits.items():
            split_map[split_name].extend(split_rows)

    return {split_name: sorted(rows, key=lambda row: row["timestamp"]) for split_name, rows in split_map.items() if rows}


def _to_npz_payload(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
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


def _normalize_family_name(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return normalized
    return normalized if normalized.startswith("atlasv2/") else f"atlasv2/{normalized}"


def _split_ordered_groups(
    groups: list[tuple[str, list[dict[str, Any]]]],
    ratios: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    count = len(groups)
    if count == 0:
        return {"train": [], "dev": [], "test": []}
    if count == 1:
        return {"train": [row for _, rows in groups for row in rows], "dev": [], "test": []}
    if count == 2:
        return {"train": groups[0][1], "dev": groups[1][1], "test": []}

    train_ratio = float(ratios.get("train", 0.6))
    dev_ratio = float(ratios.get("dev", 0.2))
    train_count = max(1, int(round(count * train_ratio)))
    dev_count = max(1, int(round(count * dev_ratio)))
    if train_count + dev_count >= count:
        overflow = train_count + dev_count - (count - 1)
        if overflow > 0:
            train_count = max(1, train_count - overflow)
        if train_count + dev_count >= count:
            dev_count = max(1, count - train_count - 1)
        if train_count + dev_count >= count:
            train_count = max(1, count - dev_count - 1)

    train_groups = groups[:train_count]
    dev_groups = groups[train_count : train_count + dev_count]
    test_groups = groups[train_count + dev_count :]
    return {
        "train": [row for _, rows in train_groups for row in rows],
        "dev": [row for _, rows in dev_groups for row in rows],
        "test": [row for _, rows in test_groups for row in rows],
    }


def _segment_benign_events(benign_events: pd.DataFrame, segment_minutes: int) -> list[pd.DataFrame]:
    if segment_minutes <= 0:
        return []

    segments: list[pd.DataFrame] = []
    segment_delta = pd.Timedelta(minutes=segment_minutes)
    for host, host_events in benign_events.groupby("host"):
        host_events = host_events.sort_values("timestamp").copy()
        if host_events.empty:
            continue
        start = host_events["timestamp"].min().floor(f"{segment_minutes}min")
        end = host_events["timestamp"].max().ceil(f"{segment_minutes}min")
        boundaries = pd.date_range(start=start, end=end + segment_delta, freq=f"{segment_minutes}min", tz="UTC")
        for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            segment = host_events[(host_events["timestamp"] >= left) & (host_events["timestamp"] < right)].copy()
            if segment.empty:
                continue
            segment["incident_id"] = f"atlasv2/{host}-benign-seg{index:03d}"
            segment["family_id"] = f"atlasv2/benign-{host}-seg{index:03d}"
            segments.append(segment)
    return segments


def _prepare_atlasv2_public_benchmark(
    events: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    reapr_summary: dict[str, Any],
) -> dict[str, Any]:
    bin_minutes = int(config.get("bin_minutes", 5))
    lookback_bins = int(config.get("lookback_bins", 12))
    main_horizon_bins = int(config.get("main_horizon_bins", 6))
    aux_horizon_bins = int(config.get("aux_horizon_bins", 2))
    split_ratios = config.get("split_ratios", {"train": 0.6, "dev": 0.2, "test": 0.2})
    context_minutes = int(config.get("context_minutes", 60))
    benign_segment_minutes = int(config.get("benign_segment_minutes", 180))
    event_disjoint_benign_ratio = float(config.get("event_disjoint_benign_ratio", 0.15))
    held_out_attack_families = {
        _normalize_family_name(value) for value in config.get("event_disjoint_attack_families", [])
    }

    benign_pool = events[events["attack_window"] == "benign"].copy()
    benign_context_pool = benign_pool[benign_pool["label"] != "malicious"].copy()
    attack_pool = events[events["attack_window"] != "benign"].copy()
    context_delta = pd.Timedelta(minutes=context_minutes)

    attack_rows: list[dict[str, Any]] = []
    benign_rows: list[dict[str, Any]] = []
    incident_table: list[dict[str, Any]] = []

    for incident_id, attack_events in attack_pool.groupby("incident_id"):
        attack_events = attack_events.sort_values("timestamp").copy()
        family_id = attack_events["family_id"].iloc[0]
        host = attack_events["host"].iloc[0]
        context_events = attack_events.iloc[0:0].copy()
        if context_minutes > 0:
            attack_start = attack_events["timestamp"].min()
            context_events = benign_context_pool[
                (benign_context_pool["host"] == host)
                & (benign_context_pool["timestamp"] >= attack_start - context_delta)
                & (benign_context_pool["timestamp"] < attack_start)
            ].copy()
            if not context_events.empty:
                context_events["incident_id"] = incident_id
                context_events["family_id"] = family_id
                attack_events = pd.concat([context_events, attack_events], ignore_index=True)

        windows = _build_incident_windows(
            incident_events=attack_events,
            lookback_bins=lookback_bins,
            main_horizon_bins=main_horizon_bins,
            aux_horizon_bins=aux_horizon_bins,
            bin_minutes=bin_minutes,
        )
        if windows:
            attack_rows.extend(windows)
        incident_table.append(
            {
                "incident_id": incident_id,
                "family_id": family_id,
                "events": int(len(attack_events)),
                "context_events": int(len(context_events)),
                "windows": int(len(windows)),
                "positive_windows": int(sum(int(row["label_main"]) for row in windows)),
                "malicious_events": int((attack_events["label"] == "malicious").sum()),
                "attack_window": attack_events["attack_window"].iloc[-1],
                "host": host,
                "benchmark_role": "attack_window",
            }
        )

    for segment_events in _segment_benign_events(benign_context_pool, benign_segment_minutes):
        windows = _build_incident_windows(
            incident_events=segment_events,
            lookback_bins=lookback_bins,
            main_horizon_bins=main_horizon_bins,
            aux_horizon_bins=aux_horizon_bins,
            bin_minutes=bin_minutes,
        )
        if not windows:
            continue
        benign_rows.extend(windows)
        incident_table.append(
            {
                "incident_id": segment_events["incident_id"].iloc[0],
                "family_id": segment_events["family_id"].iloc[0],
                "events": int(len(segment_events)),
                "context_events": 0,
                "windows": int(len(windows)),
                "positive_windows": int(sum(int(row["label_main"]) for row in windows)),
                "malicious_events": int((segment_events["label"] == "malicious").sum()),
                "attack_window": "benign_segment",
                "host": segment_events["host"].iloc[0],
                "benchmark_role": "benign_segment",
            }
        )

    if not attack_rows or not benign_rows:
        raise ValueError("ATLASv2 public benchmark preparation requires both attack and benign windows.")

    attack_groups: dict[str, list[dict[str, Any]]] = {}
    for row in attack_rows:
        attack_groups.setdefault(row["family_id"], []).append(row)
    ordered_attack_groups = sorted(
        attack_groups.items(),
        key=lambda item: min(entry["timestamp"] for entry in item[1]),
    )

    benign_groups: dict[str, list[dict[str, Any]]] = {}
    for row in benign_rows:
        benign_groups.setdefault(row["family_id"], []).append(row)
    ordered_benign_groups = sorted(
        benign_groups.items(),
        key=lambda item: min(entry["timestamp"] for entry in item[1]),
    )

    if not held_out_attack_families and len(ordered_attack_groups) >= 3:
        auto_count = max(1, int(round(len(ordered_attack_groups) * 0.2)))
        auto_count = min(auto_count, max(len(ordered_attack_groups) - 2, 1))
        held_out_attack_families = {family_id for family_id, _ in ordered_attack_groups[-auto_count:]}

    chrono_attack_groups = [
        item for item in ordered_attack_groups if item[0] not in held_out_attack_families
    ]
    event_attack_groups = [
        item for item in ordered_attack_groups if item[0] in held_out_attack_families
    ]

    benign_event_count = 0
    if len(ordered_benign_groups) >= 4 and event_disjoint_benign_ratio > 0:
        benign_event_count = max(1, int(round(len(ordered_benign_groups) * event_disjoint_benign_ratio)))
        benign_event_count = min(benign_event_count, max(len(ordered_benign_groups) - 3, 1))
    chrono_benign_groups = ordered_benign_groups[:-benign_event_count] if benign_event_count else ordered_benign_groups
    event_benign_groups = ordered_benign_groups[-benign_event_count:] if benign_event_count else []

    attack_splits = _split_ordered_groups(chrono_attack_groups, split_ratios)
    benign_splits = _split_ordered_groups(chrono_benign_groups, split_ratios)

    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "dev", "test"):
        rows = attack_splits.get(split_name, []) + benign_splits.get(split_name, [])
        if rows:
            splits[split_name] = sorted(rows, key=lambda row: row["timestamp"])

    event_rows = [row for _, rows in event_attack_groups for row in rows] + [row for _, rows in event_benign_groups for row in rows]
    if event_rows:
        splits["test_event_disjoint"] = sorted(event_rows, key=lambda row: row["timestamp"])

    if not {"train", "dev", "test"} <= set(splits):
        raise ValueError("ATLASv2 public benchmark did not produce train/dev/test splits.")

    for split_name, rows in splits.items():
        np.savez(output_dir / f"{split_name}.npz", **_to_npz_payload(rows))

    metadata = {
        "dataset_name": config.get("dataset_name", "atlasv2_public_campaign_mem"),
        "description": "ATLASv2 workbook public benchmark with benign context augmentation and benign-segment splits.",
        "source_workbook": str(config["workbook_path"]),
        "reapr_labels_csv": str(config.get("reapr_labels_csv", "")),
        "bin_minutes": bin_minutes,
        "lookback_bins": lookback_bins,
        "main_horizon_minutes": main_horizon_bins * bin_minutes,
        "aux_horizon_minutes": aux_horizon_bins * bin_minutes,
        "context_minutes": context_minutes,
        "benign_segment_minutes": benign_segment_minutes,
        "event_disjoint_benign_ratio": event_disjoint_benign_ratio,
        "event_disjoint_attack_families": sorted(held_out_attack_families),
        "feature_channels": ALERT_TYPES + ["event_count", "severity_mean", "severity_max", "malicious_count", "artifact_count", "host_h1", "host_h2"],
        "analog_fidelity_distance_threshold": float(config.get("analog_fidelity_distance_threshold", 0.35)),
        "incidents": incident_table,
        "reapr_process_summary": reapr_summary,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def prepare_atlasv2_workbook(config: dict[str, Any]) -> dict[str, Any]:
    workbook_path = Path(config["workbook_path"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_minutes = int(config.get("bin_minutes", 5))
    lookback_bins = int(config.get("lookback_bins", 12))
    main_horizon_bins = int(config.get("main_horizon_bins", 6))
    aux_horizon_bins = int(config.get("aux_horizon_bins", 2))
    split_ratios = config.get("split_ratios", {"train": 0.6, "dev": 0.2, "test": 0.2})
    events = _normalize_workbook_rows(workbook_path)
    reapr_summary = _load_reapr_summary(config.get("reapr_labels_csv"))

    if bool(config.get("public_benchmark_mode", False)):
        return _prepare_atlasv2_public_benchmark(
            events=events,
            config=config,
            output_dir=output_dir,
            reapr_summary=reapr_summary,
        )

    all_windows = []
    incident_table = []
    for incident_id, incident_events in events.groupby("incident_id"):
        windows = _build_incident_windows(
            incident_events=incident_events,
            lookback_bins=lookback_bins,
            main_horizon_bins=main_horizon_bins,
            aux_horizon_bins=aux_horizon_bins,
            bin_minutes=bin_minutes,
        )
        if not windows:
            continue
        all_windows.extend(windows)
        incident_table.append(
            {
                "incident_id": incident_id,
                "family_id": incident_events["family_id"].iloc[0],
                "events": int(len(incident_events)),
                "windows": int(len(windows)),
                "malicious_events": int((incident_events["label"] == "malicious").sum()),
                "attack_window": incident_events["attack_window"].iloc[0],
                "host": incident_events["host"].iloc[0],
            }
        )

    if not all_windows:
        raise ValueError("No windows were generated from the ATLASv2 workbook.")

    splits = _split_incidents(all_windows, split_ratios)
    for split_name, rows in splits.items():
        np.savez(output_dir / f"{split_name}.npz", **_to_npz_payload(rows))

    metadata = {
        "dataset_name": "atlasv2_workbook_campaign_mem",
        "description": "ATLASv2 alert-label workbook projected into Campaign-MEM rolling windows.",
        "source_workbook": str(workbook_path),
        "reapr_labels_csv": str(config.get("reapr_labels_csv", "")),
        "bin_minutes": bin_minutes,
        "lookback_bins": lookback_bins,
        "main_horizon_minutes": main_horizon_bins * bin_minutes,
        "aux_horizon_minutes": aux_horizon_bins * bin_minutes,
        "feature_channels": ALERT_TYPES + ["event_count", "severity_mean", "severity_max", "malicious_count", "artifact_count", "host_h1", "host_h2"],
        "analog_fidelity_distance_threshold": float(config.get("analog_fidelity_distance_threshold", 0.35)),
        "incidents": incident_table,
        "reapr_process_summary": reapr_summary,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata
