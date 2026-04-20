from __future__ import annotations

import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .atlasv2 import (
    ALERT_TYPES,
    _build_incident_windows,
    _normalize_family_name,
    _split_ordered_groups,
    _to_npz_payload,
    collapse_alert_type,
)


RAW_COLUMNS = ["Keywords", "Date and Time", "Source", "Event ID", "Task Category", "Message"]
PID_RE = re.compile(r"Process ID:\s*\t*\s*(0x[0-9a-fA-F]+|\d+)", re.I)
PROC_RE = re.compile(r"Process Name:\s*\t*\s*(.+)")
FIRST_LINE_RE = re.compile(r"^\s*([^\r\n]+)")
RISK_RANK = {"benign": 0, "contaminated": 1, "attack": 2}


def _parse_process_identity(message: str) -> tuple[int, str]:
    text = str(message or "")
    pid_match = PID_RE.search(text)
    proc_match = PROC_RE.search(text)
    pid = -1
    if pid_match:
        raw = pid_match.group(1)
        pid = int(raw, 16) if raw.lower().startswith("0x") else int(raw)
    process_name = ""
    if proc_match:
        process_name = proc_match.group(1).splitlines()[0].strip().lower()
    return pid, process_name


def _first_line(message: str) -> str:
    match = FIRST_LINE_RE.search(str(message or ""))
    return match.group(1).strip() if match else ""


def _canonical_alert_label(raw_label: str) -> str:
    normalized = str(raw_label or "benign").strip().lower()
    if normalized == "attack":
        return "malicious"
    if normalized == "contaminated":
        return "artifact"
    return "benign"


def _severity_from_components(raw_label: str, alert_type: str, event_id: int | float | str | None) -> float:
    severity = {"attack": 0.9, "contaminated": 0.45}.get(str(raw_label or "").strip().lower(), 0.08)
    if alert_type in {"execution", "lateral_move", "cred_access", "impact", "defense_evasion"}:
        severity += 0.1
    elif alert_type != "other":
        severity += 0.05
    try:
        event_id_int = int(float(event_id)) if event_id is not None else -1
    except (TypeError, ValueError):
        event_id_int = -1
    if event_id_int in {4624, 4625, 4688, 4697, 7045}:
        severity += 0.05
    return float(min(severity, 1.0))


def _normalize_raw_window(window: dict[str, Any]) -> dict[str, Any]:
    prefix = np.asarray(window["prefix"], dtype=np.float32).copy()
    future_signature = np.asarray(window["future_signature"], dtype=np.float32).copy()
    count_indices = list(range(len(ALERT_TYPES) + 1)) + [len(ALERT_TYPES) + 3, len(ALERT_TYPES) + 4]
    prefix[:, count_indices] = np.log1p(prefix[:, count_indices])
    future_signature[[3, 4, 5]] = np.log1p(np.maximum(future_signature[[3, 4, 5]], 0.0))
    normalized = dict(window)
    normalized["prefix"] = prefix
    normalized["future_signature"] = future_signature
    return normalized


def _load_process_labels(labels_csv_path: str | Path) -> tuple[dict[str, dict[str, dict[Any, str]]], dict[str, dict[str, int]]]:
    frame = pd.read_csv(labels_csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in ("attack", "process_name", "label"):
        frame[column] = frame[column].astype(str).str.strip().str.lower()
    frame["process_id"] = pd.to_numeric(frame["process_id"], errors="coerce").fillna(-1).astype(int)

    attack_maps: dict[str, dict[str, dict[Any, str]]] = {}
    attack_summary: dict[str, dict[str, int]] = {}
    for attack_name, group in frame.groupby("attack"):
        pid_map: dict[int, str] = {}
        proc_map: dict[str, str] = {}
        summary = Counter(group["label"].tolist())
        for row in group.itertuples(index=False):
            current_pid = pid_map.get(int(row.process_id))
            if row.process_id >= 0 and RISK_RANK[row.label] >= RISK_RANK.get(current_pid or "benign", 0):
                pid_map[int(row.process_id)] = row.label
            process_name = str(row.process_name)
            current_proc = proc_map.get(process_name)
            if process_name and RISK_RANK[row.label] >= RISK_RANK.get(current_proc or "benign", 0):
                proc_map[process_name] = row.label
        attack_maps[attack_name] = {"pid": pid_map, "proc": proc_map}
        attack_summary[attack_name] = {label: int(count) for label, count in summary.items()}
    return attack_maps, attack_summary


def _match_process_label(
    attack_key: str,
    pid: int,
    process_name: str,
    label_maps: dict[str, dict[str, dict[Any, str]]],
) -> str:
    maps = label_maps.get(attack_key)
    if not maps:
        return "benign"
    if pid >= 0 and pid in maps["pid"]:
        return str(maps["pid"][pid])
    if process_name and process_name in maps["proc"]:
        return str(maps["proc"][process_name])
    return "benign"


def _load_security_events_from_zip(
    zip_path: Path,
    labels_by_attack: dict[str, dict[str, dict[Any, str]]],
) -> list[pd.DataFrame]:
    scenario = zip_path.stem.lower()
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as archive:
        entries = [entry for entry in archive.namelist() if entry.endswith("/logs/security_events.txt")]
        for entry_name in entries:
            parts = Path(entry_name).parts
            if len(parts) >= 4:
                host = parts[1].lower()
            elif len(parts) == 3:
                host = "h1"
            else:
                continue
            attack_key = f"atlasv2/{host}-{scenario}"
            with archive.open(entry_name) as handle:
                frame = pd.read_csv(
                    handle,
                    sep="\t",
                    engine="python",
                    quotechar='"',
                    header=0,
                    names=RAW_COLUMNS,
                )
            frame = frame.dropna(subset=["Date and Time"]).copy()
            frame["timestamp"] = pd.to_datetime(frame["Date and Time"], utc=True, errors="coerce", format="mixed")
            frame["event_id"] = pd.to_numeric(frame["Event ID"], errors="coerce").fillna(-1).astype(int)
            frame = frame.dropna(subset=["timestamp"]).copy()
            parsed = frame["Message"].astype(str).map(_parse_process_identity)
            frame["pid"] = [item[0] for item in parsed]
            frame["path"] = [item[1] for item in parsed]
            frame["host"] = host
            frame["scenario"] = scenario
            frame["incident_id"] = f"atlasraw/{host}-{scenario}"
            frame["family_id"] = f"atlasraw/{scenario}"
            frame["raw_label"] = [
                _match_process_label(attack_key=attack_key, pid=int(pid), process_name=path, label_maps=labels_by_attack)
                for pid, path in zip(frame["pid"], frame["path"])
            ]
            frame["label"] = frame["raw_label"].map(_canonical_alert_label)
            frame["message_first_line"] = frame["Message"].astype(str).map(_first_line)
            frame["report_text"] = (
                frame["Task Category"].fillna("").astype(str)
                + " "
                + frame["message_first_line"].fillna("").astype(str)
                + " "
                + frame["path"].fillna("").astype(str)
            ).str.strip()
            frame["alert_type"] = frame["report_text"].map(collapse_alert_type)
            frame["severity"] = [
                _severity_from_components(raw_label, alert_type, event_id)
                for raw_label, alert_type, event_id in zip(frame["raw_label"], frame["alert_type"], frame["event_id"])
            ]
            frames.append(
                frame[
                    [
                        "timestamp",
                        "incident_id",
                        "family_id",
                        "host",
                        "pid",
                        "path",
                        "event_id",
                        "Task Category",
                        "report_text",
                        "severity",
                        "label",
                        "raw_label",
                        "alert_type",
                    ]
                ].rename(columns={"Task Category": "task_category"})
            )
    return frames


def prepare_atlas_raw_public(config: dict[str, Any]) -> dict[str, Any]:
    raw_logs_dir = Path(config["raw_logs_dir"])
    labels_csv_path = Path(config["labels_csv"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bin_minutes = int(config.get("bin_minutes", 5))
    lookback_bins = int(config.get("lookback_bins", 4))
    main_horizon_bins = int(config.get("main_horizon_bins", 6))
    aux_horizon_bins = int(config.get("aux_horizon_bins", 2))
    split_ratios = config.get("split_ratios", {"train": 0.6, "dev": 0.2, "test": 0.2})
    held_out_families = {
        _normalize_family_name(value).replace("atlasv2/", "atlasraw/")
        for value in config.get("event_disjoint_attack_families", [])
    }

    label_maps, label_summary = _load_process_labels(labels_csv_path)
    all_frames: list[pd.DataFrame] = []
    for zip_path in sorted(raw_logs_dir.glob("*.zip")):
        all_frames.extend(_load_security_events_from_zip(zip_path, label_maps))
    if not all_frames:
        raise ValueError(f"No raw security events were parsed from {raw_logs_dir}")

    windows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    windows_by_incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
    incident_family: dict[str, str] = {}
    incident_table: list[dict[str, Any]] = []
    label_totals = Counter()
    scenario_count_by_family = Counter()

    for incident_events in sorted(all_frames, key=lambda frame: str(frame["incident_id"].iloc[0])):
        incident_events = incident_events.sort_values("timestamp").copy()
        windows = _build_incident_windows(
            incident_events=incident_events,
            lookback_bins=lookback_bins,
            main_horizon_bins=main_horizon_bins,
            aux_horizon_bins=aux_horizon_bins,
            bin_minutes=bin_minutes,
        )
        if not windows:
            continue
        windows = [_normalize_raw_window(window) for window in windows]
        family_id = str(incident_events["family_id"].iloc[0])
        incident_id = str(incident_events["incident_id"].iloc[0])
        windows_by_family[family_id].extend(windows)
        windows_by_incident[incident_id].extend(windows)
        incident_family[incident_id] = family_id
        scenario_count_by_family[family_id] += 1
        label_totals.update(incident_events["raw_label"].tolist())
        incident_table.append(
            {
                "incident_id": incident_id,
                "family_id": family_id,
                "host": str(incident_events["host"].iloc[0]),
                "events": int(len(incident_events)),
                "windows": int(len(windows)),
                "positive_windows": int(sum(int(row["label_main"]) for row in windows)),
                "raw_attack_events": int((incident_events["raw_label"] == "attack").sum()),
                "raw_contaminated_events": int((incident_events["raw_label"] == "contaminated").sum()),
                "raw_benign_events": int((incident_events["raw_label"] == "benign").sum()),
                "time_min": str(incident_events["timestamp"].min()),
                "time_max": str(incident_events["timestamp"].max()),
            }
        )

    ordered_groups = sorted(
        windows_by_family.items(),
        key=lambda item: min(entry["timestamp"] for entry in item[1]),
    )
    if not ordered_groups:
        raise ValueError("ATLAS raw benchmark preparation produced no rolling windows.")

    if not held_out_families and len(ordered_groups) >= 4:
        auto_count = max(1, int(round(len(ordered_groups) * 0.2)))
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
        raise ValueError("ATLAS raw benchmark did not produce train/dev/test splits.")

    for split_name, rows in splits.items():
        np.savez(output_dir / f"{split_name}.npz", **_to_npz_payload(rows))

    metadata = {
        "dataset_name": config.get("dataset_name", "atlas_raw_public_campaign_mem"),
        "description": "ATLAS raw security-event benchmark projected into Campaign-MEM rolling windows using public REAPr process labels.",
        "count_feature_transform": "log1p on alert-count, event-count, malicious-count, artifact-count, and corresponding future-signature rate channels.",
        "raw_logs_dir": str(raw_logs_dir),
        "labels_csv": str(labels_csv_path),
        "bin_minutes": bin_minutes,
        "lookback_bins": lookback_bins,
        "main_horizon_minutes": main_horizon_bins * bin_minutes,
        "aux_horizon_minutes": aux_horizon_bins * bin_minutes,
        "feature_channels": ALERT_TYPES + ["event_count", "severity_mean", "severity_max", "malicious_count", "artifact_count", "host_h1", "host_h2"],
        "analog_fidelity_distance_threshold": float(config.get("analog_fidelity_distance_threshold", 0.35)),
        "event_disjoint_attack_families": sorted(held_out_families),
        "family_group_sizes": {
            family_id: {
                "windows": int(len(rows)),
                "incidents": int(scenario_count_by_family[family_id]),
            }
            for family_id, rows in ordered_groups
        },
        "raw_label_totals": {label: int(count) for label, count in label_totals.items()},
        "process_label_summary": label_summary,
        "incidents": incident_table,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata
