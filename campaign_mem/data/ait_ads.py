from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .atlasv2 import _split_ordered_groups, collapse_alert_type
from .canonical_alerts import _build_windows, _to_payload


HIGH_RISK_STAGES = {
    "webshell",
    "reverse_shell",
    "privilege_escalation",
    "service_stop",
    "dnsteal",
}

STAGE_TO_ALERT_TYPE = {
    "network_scans": "recon",
    "service_scans": "recon",
    "dirb": "recon",
    "wpscan": "recon",
    "cracking": "auth_abuse",
    "webshell": "execution",
    "reverse_shell": "execution",
    "privilege_escalation": "priv_esc",
    "dnsteal": "collection_exfil",
    "service_stop": "impact",
}


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            timestamp = _parse_timestamp(item)
            if timestamp is not None:
                return timestamp
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            parsed = pd.to_datetime(float(value), unit="s", utc=True, errors="coerce")
        else:
            parsed = pd.to_datetime(value, utc=True, format="mixed", errors="coerce")
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed


def _load_stage_intervals(labels_csv_path: str | Path) -> dict[str, list[tuple[float, float, str]]]:
    frame = pd.read_csv(labels_csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"scenario", "attack", "start", "end"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"AIT-ADS labels CSV missing columns: {sorted(missing)}")
    frame["scenario"] = frame["scenario"].astype(str).str.strip().str.lower()
    frame["attack"] = frame["attack"].astype(str).str.strip().str.lower()
    frame["start"] = pd.to_numeric(frame["start"], errors="coerce")
    frame["end"] = pd.to_numeric(frame["end"], errors="coerce")
    frame = frame.dropna(subset=["start", "end"]).sort_values(["scenario", "start", "end"])
    intervals: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        intervals[str(row.scenario)].append((float(row.start), float(row.end), str(row.attack)))
    return dict(intervals)


def _lookup_stage(intervals: list[tuple[float, float, str]], timestamp: pd.Timestamp) -> str:
    stamp = float(timestamp.timestamp())
    for start, end, attack in intervals:
        if start <= stamp < end:
            return attack
    if intervals and stamp >= intervals[-1][1]:
        return intervals[-1][2]
    return ""


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                record = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _join_text(parts: list[Any]) -> str:
    cleaned = [str(part).strip() for part in parts if str(part or "").strip()]
    return " | ".join(cleaned)


def _extract_stage_aware_alert_type(stage: str, report_text: str) -> str:
    alert_type = collapse_alert_type(report_text)
    if alert_type == "other" and stage:
        return STAGE_TO_ALERT_TYPE.get(stage, "other")
    return alert_type


def _normalize_wazuh_severity(record: dict[str, Any], stage: str) -> float:
    rule = record.get("rule", {}) if isinstance(record.get("rule"), dict) else {}
    data = record.get("data", {}) if isinstance(record.get("data"), dict) else {}
    suricata_alert = data.get("alert", {}) if isinstance(data.get("alert"), dict) else {}
    level = pd.to_numeric(rule.get("level"), errors="coerce")
    base = float(level) / 15.0 if pd.notna(level) else 0.2
    suricata_severity = pd.to_numeric(suricata_alert.get("severity"), errors="coerce")
    if pd.notna(suricata_severity):
        mapped = max(0.0, 1.0 - (float(suricata_severity) - 1.0) / 4.0)
        base = max(base, mapped)
    if stage in HIGH_RISK_STAGES:
        base = max(base, 0.75)
    return float(min(max(base, 0.0), 1.0))


def _normalize_aminer_severity(record: dict[str, Any], stage: str) -> float:
    component = record.get("AnalysisComponent", {}) if isinstance(record.get("AnalysisComponent"), dict) else {}
    component_type = str(component.get("AnalysisComponentType", "")).strip().lower()
    component_name = str(component.get("AnalysisComponentName", "")).strip().lower()
    base = 0.35
    if "valuecombo" in component_type:
        base = 0.45
    elif "newmatchpathdetector" in component_type:
        base = 0.4
    elif "anomaly" in component_name or "rare" in component_name:
        base = 0.5
    if stage in HIGH_RISK_STAGES:
        base = max(base, 0.7)
    return float(min(max(base, 0.0), 1.0))


def _canonicalize_wazuh_record(record: dict[str, Any], scenario: str, intervals: list[tuple[float, float, str]]) -> dict[str, Any] | None:
    timestamp = _parse_timestamp(record.get("@timestamp"))
    if timestamp is None:
        return None
    stage = _lookup_stage(intervals, timestamp)
    rule = record.get("rule", {}) if isinstance(record.get("rule"), dict) else {}
    data = record.get("data", {}) if isinstance(record.get("data"), dict) else {}
    suricata_alert = data.get("alert", {}) if isinstance(data.get("alert"), dict) else {}
    report_text = _join_text(
        [
            "wazuh",
            rule.get("description"),
            " ".join(rule.get("groups", [])) if isinstance(rule.get("groups"), list) else rule.get("groups"),
            suricata_alert.get("signature"),
            suricata_alert.get("category"),
            record.get("full_log"),
        ]
    )
    host_hash = _join_text(
        [
            record.get("agent", {}).get("id") if isinstance(record.get("agent"), dict) else "",
            record.get("agent", {}).get("ip") if isinstance(record.get("agent"), dict) else "",
            record.get("predecoder", {}).get("hostname") if isinstance(record.get("predecoder"), dict) else "",
            record.get("location"),
        ]
    )
    return {
        "timestamp": timestamp,
        "incident_id": f"aitads/{scenario}-wazuh",
        "family_id": f"aitads/{scenario}",
        "detector": "wazuh",
        "stage": stage,
        "is_high_risk": stage in HIGH_RISK_STAGES,
        "report_text": report_text,
        "severity": _normalize_wazuh_severity(record, stage),
        "host_hash": host_hash,
        "alert_type": _extract_stage_aware_alert_type(stage, report_text),
    }


def _canonicalize_aminer_record(record: dict[str, Any], scenario: str, intervals: list[tuple[float, float, str]]) -> dict[str, Any] | None:
    log_data = record.get("LogData", {}) if isinstance(record.get("LogData"), dict) else {}
    timestamp = _parse_timestamp(log_data.get("DetectionTimestamp")) or _parse_timestamp(log_data.get("Timestamps"))
    if timestamp is None:
        return None
    stage = _lookup_stage(intervals, timestamp)
    component = record.get("AnalysisComponent", {}) if isinstance(record.get("AnalysisComponent"), dict) else {}
    raw_lines = log_data.get("RawLogData", [])
    first_line = raw_lines[0] if isinstance(raw_lines, list) and raw_lines else ""
    report_text = _join_text(
        [
            "aminer",
            component.get("AnalysisComponentName"),
            component.get("Message"),
            first_line,
        ]
    )
    resources = log_data.get("LogResources", [])
    host_hash = _join_text(
        [
            record.get("AMiner", {}).get("ID") if isinstance(record.get("AMiner"), dict) else "",
            resources[0] if isinstance(resources, list) and resources else "",
        ]
    )
    return {
        "timestamp": timestamp,
        "incident_id": f"aitads/{scenario}-aminer",
        "family_id": f"aitads/{scenario}",
        "detector": "aminer",
        "stage": stage,
        "is_high_risk": stage in HIGH_RISK_STAGES,
        "report_text": report_text,
        "severity": _normalize_aminer_severity(record, stage),
        "host_hash": host_hash,
        "alert_type": _extract_stage_aware_alert_type(stage, report_text),
    }


def _canonicalize_source_file(
    path: Path,
    stage_intervals: dict[str, list[tuple[float, float, str]]],
) -> list[dict[str, Any]]:
    stem = path.stem.lower()
    if "_" not in stem:
        return []
    scenario, detector = stem.rsplit("_", 1)
    intervals = stage_intervals.get(scenario, [])
    rows: list[dict[str, Any]] = []
    parser = _canonicalize_wazuh_record if detector == "wazuh" else _canonicalize_aminer_record
    for record in _iter_jsonl(path):
        row = parser(record, scenario, intervals)
        if row is not None:
            rows.append(row)
    return rows


def _normalize_ait_window(window: dict[str, Any]) -> dict[str, Any]:
    prefix = np.asarray(window["prefix"], dtype=np.float32).copy()
    future_signature = np.asarray(window["future_signature"], dtype=np.float32).copy()
    count_indices = list(range(12)) + [12, 15, 16]
    prefix[:, count_indices] = np.log1p(np.maximum(prefix[:, count_indices], 0.0))
    future_signature[[3, 4]] = np.log1p(np.maximum(future_signature[[3, 4]], 0.0))
    normalized = dict(window)
    normalized["prefix"] = prefix
    normalized["future_signature"] = future_signature
    return normalized


def prepare_ait_ads_public(config: dict[str, Any]) -> dict[str, Any]:
    raw_dir = Path(config["raw_dir"])
    labels_csv_path = Path(config["labels_csv"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_minutes = int(config.get("bin_minutes", 5))
    lookback_bins = int(config.get("lookback_bins", 4))
    main_horizon_bins = int(config.get("main_horizon_bins", 6))
    aux_horizon_bins = int(config.get("aux_horizon_bins", 2))
    split_ratios = config.get("split_ratios", {"train": 0.6, "dev": 0.2, "test": 0.2})
    held_out_families = {
        f"aitads/{str(value).strip().lower().removeprefix('aitads/')}"
        for value in config.get("event_disjoint_attack_families", [])
    }
    canonical_csv_path = output_dir / "ait_ads_canonical_events.csv"

    stage_intervals = _load_stage_intervals(labels_csv_path)
    source_files = sorted(raw_dir.glob("*.json"))
    if not source_files:
        raise ValueError(f"No AIT-ADS source files found in {raw_dir}")

    windows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    windows_by_incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incident_family: dict[str, str] = {}
    incident_table: list[dict[str, Any]] = []
    stage_totals = Counter()
    detector_totals = Counter()
    raw_event_totals = Counter()

    fieldnames = [
        "timestamp",
        "incident_id",
        "family_id",
        "detector",
        "stage",
        "is_high_risk",
        "report_text",
        "severity",
        "host_hash",
        "alert_type",
    ]
    with canonical_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for source_file in source_files:
            rows = _canonicalize_source_file(source_file, stage_intervals)
            if not rows:
                continue
            detector = rows[0]["detector"]
            detector_totals[detector] += len(rows)
            for row in rows:
                raw_event_totals["events"] += 1
                stage_totals[row["stage"] or "unlabeled"] += 1
                serializable = dict(row)
                serializable["timestamp"] = row["timestamp"].isoformat()
                writer.writerow(serializable)

            events = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            incident_id = str(events["incident_id"].iloc[0])
            family_id = str(events["family_id"].iloc[0])
            windows = _build_windows(
                incident_events=events,
                lookback_bins=lookback_bins,
                main_horizon_bins=main_horizon_bins,
                aux_horizon_bins=aux_horizon_bins,
                bin_minutes=bin_minutes,
            )
            if not windows:
                continue
            windows = [_normalize_ait_window(window) for window in windows]
            windows_by_family[family_id].extend(windows)
            windows_by_incident[incident_id].extend(windows)
            incident_family[incident_id] = family_id
            incident_table.append(
                {
                    "incident_id": incident_id,
                    "family_id": family_id,
                    "detector": detector,
                    "events": int(len(events)),
                    "windows": int(len(windows)),
                    "positive_windows": int(sum(int(row["label_main"]) for row in windows)),
                    "high_risk_events": int(events["is_high_risk"].sum()),
                    "stage_counts": {
                        stage or "unlabeled": int(count)
                        for stage, count in events["stage"].value_counts().sort_index().items()
                    },
                    "time_min": str(events["timestamp"].min()),
                    "time_max": str(events["timestamp"].max()),
                }
            )

    ordered_groups = sorted(
        windows_by_family.items(),
        key=lambda item: min(entry["timestamp"] for entry in item[1]),
    )
    if not ordered_groups:
        raise ValueError("AIT-ADS benchmark preparation produced no rolling windows.")

    if not held_out_families and len(ordered_groups) >= 4:
        auto_count = max(1, int(round(len(ordered_groups) * 0.25)))
        auto_count = min(auto_count, max(len(ordered_groups) - 2, 1))
        held_out_families = {family_id for family_id, _ in ordered_groups[-auto_count:]}

    ordered_incidents = sorted(
        windows_by_incident.items(),
        key=lambda item: min(entry["timestamp"] for entry in item[1]),
    )
    chrono_groups = [
        item for item in ordered_incidents if incident_family[str(item[0])] not in held_out_families
    ]
    held_out_groups = [item for item in ordered_groups if item[0] in held_out_families]
    chrono_splits = _split_ordered_groups(chrono_groups, split_ratios)

    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "dev", "test"):
        rows = chrono_splits.get(split_name, [])
        if rows:
            splits[split_name] = sorted(rows, key=lambda row: row["timestamp"])

    held_out_rows = [row for _, rows in held_out_groups for row in rows]
    if held_out_rows:
        splits["test_event_disjoint"] = sorted(held_out_rows, key=lambda row: row["timestamp"])

    if not {"train", "dev", "test"} <= set(splits):
        raise ValueError("AIT-ADS benchmark did not produce train/dev/test splits.")

    for split_name, rows in splits.items():
        np.savez(output_dir / f"{split_name}.npz", **_to_payload(rows))

    metadata = {
        "dataset_name": config.get("dataset_name", "ait_ads_public_campaign_mem"),
        "description": "AIT Alert Data Set projected into Campaign-MEM rolling windows with scenario-held-out evaluation.",
        "count_feature_transform": "log1p on alert-count channels, event_count, high_risk_count, host_count, and future-signature count channels.",
        "raw_dir": str(raw_dir),
        "labels_csv": str(labels_csv_path),
        "canonical_csv": str(canonical_csv_path),
        "bin_minutes": bin_minutes,
        "lookback_bins": lookback_bins,
        "main_horizon_minutes": main_horizon_bins * bin_minutes,
        "aux_horizon_minutes": aux_horizon_bins * bin_minutes,
        "feature_channels": [
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
            "event_count",
            "severity_mean",
            "severity_max",
            "high_risk_count",
            "host_count",
        ],
        "analog_fidelity_distance_threshold": float(config.get("analog_fidelity_distance_threshold", 0.35)),
        "event_disjoint_attack_families": sorted(held_out_families),
        "high_risk_stages": sorted(HIGH_RISK_STAGES),
        "stage_totals": {stage: int(count) for stage, count in stage_totals.items()},
        "detector_totals": {detector: int(count) for detector, count in detector_totals.items()},
        "raw_event_totals": {name: int(count) for name, count in raw_event_totals.items()},
        "family_group_sizes": {
            family_id: {"windows": int(len(rows))}
            for family_id, rows in ordered_groups
        },
        "incidents": incident_table,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata
