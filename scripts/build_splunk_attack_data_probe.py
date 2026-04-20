from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data import prepare_canonical_alert_dataset
from campaign_mem.data.atlasv2 import ALERT_TYPES


MEDIA_ROOT = "https://media.githubusercontent.com/media/splunk/attack_data/master"


@dataclass(frozen=True)
class LogSpec:
    technique: str
    stage: str
    path: str

    @property
    def dataset_id(self) -> str:
        parts = self.path.split("/")
        if len(parts) >= 5:
            return f"{self.technique}/{parts[3]}/{Path(parts[-1]).stem}"
        return f"{self.technique}/{Path(self.path).stem}"


CURATED_LOGS: tuple[LogSpec, ...] = (
    LogSpec("T1046", "recon", "datasets/attack_techniques/T1046/nmap/horizontal.log"),
    LogSpec("T1046", "recon", "datasets/attack_techniques/T1046/nmap/vertical.log"),
    LogSpec("T1046", "recon", "datasets/attack_techniques/T1046/advanced_ip_port_scanner/advanced_ip_port_scanner.log"),
    LogSpec("T1087", "recon", "datasets/attack_techniques/T1087/enumerate_users_local_group_using_telegram/windows-xml.log"),
    LogSpec("T1059.001", "execution", "datasets/attack_techniques/T1059.001/atomic_red_team/windows-sysmon.log"),
    LogSpec("T1059.001", "execution", "datasets/attack_techniques/T1059.001/encoded_powershell/explorer_spawns_windows-sysmon.log"),
    LogSpec("T1059.003", "execution", "datasets/attack_techniques/T1059.003/cmd_spawns_cscript/windows-sysmon.log"),
    LogSpec("T1547.001", "persistence", "datasets/attack_techniques/T1547.001/atomic_red_team/windows-sysmon.log"),
    LogSpec("T1562.001", "defense_evasion", "datasets/attack_techniques/T1562.001/atomic_red_team/windows-sysmon.log"),
    LogSpec("T1003.001", "cred_access", "datasets/attack_techniques/T1003.001/atomic_red_team/windows-sysmon.log"),
    LogSpec("T1003.001", "cred_access", "datasets/attack_techniques/T1003.001/atomic_red_team/windows-sysmon_creddump.log"),
    LogSpec("T1003.003", "cred_access", "datasets/attack_techniques/T1003.003/atomic_red_team/windows-sysmon.log"),
    LogSpec("T1021.002", "lateral_move", "datasets/attack_techniques/T1021.002/atomic_red_team/windows-sysmon.log"),
    LogSpec("T1021.002", "lateral_move", "datasets/attack_techniques/T1021.002/atomic_red_team/smbexec_windows-sysmon.log"),
    LogSpec("T1021.002", "lateral_move", "datasets/attack_techniques/T1021.002/atomic_red_team/wmiexec_windows-sysmon.log"),
    LogSpec("T1021.006", "lateral_move", "datasets/attack_techniques/T1021.006/lateral_movement/windows-sysmon.log"),
    LogSpec("T1021.006", "lateral_move", "datasets/attack_techniques/T1021.006/lateral_movement_psh/windows-sysmon.log"),
    LogSpec("T1105", "collection_exfil", "datasets/attack_techniques/T1105/atomic_red_team/windows-sysmon.log"),
    LogSpec("T1105", "collection_exfil", "datasets/attack_techniques/T1105/atomic_red_team/windows-sysmon_curl.log"),
    LogSpec("T1105", "collection_exfil", "datasets/attack_techniques/T1105/atomic_red_team/windows-sysmon_curl_upload.log"),
    LogSpec("T1567", "collection_exfil", "datasets/attack_techniques/T1567/gdrive/gdrive_windows.log"),
    LogSpec("T1486", "impact", "datasets/attack_techniques/T1486/dcrypt/windows-sysmon.log"),
    LogSpec("T1486", "impact", "datasets/attack_techniques/T1486/sam_sam_note/windows-sysmon.log"),
    LogSpec("T1068", "priv_esc", "datasets/attack_techniques/T1068/windows_escalation_behavior/windows_escalation_behavior_sysmon.log"),
    LogSpec("T1068", "priv_esc", "datasets/attack_techniques/T1068/drivers/sysmon_sys_filemod.log"),
)

HIGH_RISK_STAGES = {"priv_esc", "cred_access", "lateral_move", "collection_exfil", "impact"}

HIGH_RISK_PATTERNS = {
    "priv_esc": (
        r"privilege",
        r"elevation",
        r"uac",
        r"token",
        r"driver",
        r"kernel",
        r"fodhelper",
        r"pkexec",
    ),
    "cred_access": (
        r"lsass",
        r"mimikatz",
        r"procdump",
        r"createdump",
        r"ntds",
        r"sam",
        r"credential",
        r"dump",
    ),
    "lateral_move": (
        r"psexec",
        r"wmiexec",
        r"winrm",
        r"winrs",
        r"pssession",
        r"remote",
        r"admin\$",
        r"smbexec",
    ),
    "collection_exfil": (
        r"curl",
        r"upload",
        r"download",
        r"gdrive",
        r"exfil",
        r"archive",
        r"compress",
        r"transfer",
    ),
    "impact": (
        r"encrypt",
        r"ransom",
        r"dcrypt",
        r"vssadmin",
        r"shadow",
        r"delete",
        r"bitlocker",
    ),
}

EVENT_RE = re.compile(r"<Event\b.*?</Event>", re.IGNORECASE | re.DOTALL)
TIMESTAMP_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[T ][0-2]\d:[0-5]\d:[0-5]\d(?:\.\d+)?Z?)"
)


def _download(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "tracer-public-probe"})
    with urllib.request.urlopen(request, timeout=180) as response:
        output_path.write_bytes(response.read())


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _event_text(fields: dict[str, Any]) -> str:
    return " ".join(str(value) for value in fields.values() if value is not None)


def _is_high_risk(stage: str, text: str) -> bool:
    if stage not in HIGH_RISK_STAGES:
        return False
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in HIGH_RISK_PATTERNS.get(stage, ()))


def _extract_xml_events(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in EVENT_RE.finditer(text):
        event_xml = match.group(0)
        try:
            root = ET.fromstring(event_xml)
        except ET.ParseError:
            continue
        fields: dict[str, Any] = {}
        timestamp: datetime | None = None
        for node in root.iter():
            name = _local_name(node.tag)
            if name == "TimeCreated":
                timestamp = _parse_datetime(node.attrib.get("SystemTime"))
            elif name == "Computer" and node.text:
                fields["Computer"] = node.text
            elif name == "Provider":
                fields["Provider"] = node.attrib.get("Name", "")
            elif name == "EventID" and node.text:
                fields["EventID"] = node.text
            elif name == "Data":
                key = node.attrib.get("Name") or f"Data{len(fields)}"
                fields[key] = node.text or ""
                if key.lower() in {"utctime", "eventtime"} and timestamp is None:
                    timestamp = _parse_datetime(node.text)
        if timestamp is None:
            match_ts = TIMESTAMP_RE.search(event_xml)
            timestamp = _parse_datetime(match_ts.group("ts")) if match_ts else None
        if timestamp is None:
            continue
        rows.append({"timestamp": timestamp, "fields": fields, "raw_text": event_xml[:4000]})
    return rows


def _extract_json_events(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates: list[Any] = []
    stripped = text.strip()
    if not stripped:
        return rows
    try:
        loaded = json.loads(stripped)
        candidates = loaded if isinstance(loaded, list) else [loaded]
    except json.JSONDecodeError:
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for item in candidates:
        if not isinstance(item, dict):
            continue
        timestamp = None
        for key in ("_time", "time", "timestamp", "EventTime", "UtcTime", "@timestamp"):
            timestamp = _parse_datetime(item.get(key))
            if timestamp is not None:
                break
        if timestamp is None:
            match_ts = TIMESTAMP_RE.search(json.dumps(item, ensure_ascii=False))
            timestamp = _parse_datetime(match_ts.group("ts")) if match_ts else None
        if timestamp is None:
            continue
        rows.append({"timestamp": timestamp, "fields": item, "raw_text": json.dumps(item, ensure_ascii=False)[:4000]})
    return rows


def _extract_line_events(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match_ts = TIMESTAMP_RE.search(line)
        timestamp = _parse_datetime(match_ts.group("ts")) if match_ts else None
        if timestamp is None:
            continue
        rows.append({"timestamp": timestamp, "fields": {"message": line}, "raw_text": line[:4000]})
    return rows


def _extract_events(file_path: Path) -> list[dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    xml_rows = _extract_xml_events(text)
    if xml_rows:
        return xml_rows
    json_rows = _extract_json_events(text)
    if json_rows:
        return json_rows
    return _extract_line_events(text)


def _canonical_rows(spec: LogSpec, file_path: Path) -> list[dict[str, Any]]:
    events = _extract_events(file_path)
    rows: list[dict[str, Any]] = []
    matched_high_risk = 0
    for index, event in enumerate(sorted(events, key=lambda item: item["timestamp"])):
        fields = event["fields"]
        raw_event_text = _event_text(fields)
        text = f"{spec.technique} {spec.stage} {spec.path} {raw_event_text}"
        high_risk = _is_high_risk(spec.stage, raw_event_text)
        matched_high_risk += int(high_risk)
        rows.append(
            {
                "timestamp": event["timestamp"].isoformat().replace("+00:00", "Z"),
                "incident_id": f"splunk_attack_data/{spec.dataset_id}",
                "family_id": f"splunk_attack_data/{spec.technique}",
                "stage": spec.stage,
                "alert_type": spec.stage,
                "severity": 5.0 if high_risk else (2.0 if spec.stage in HIGH_RISK_STAGES else 1.0),
                "is_high_risk": bool(high_risk),
                "host_hash": str(fields.get("Computer") or fields.get("host") or fields.get("hostname") or "unknown"),
                "report_text": text[:2000],
                "source_file": spec.path,
                "event_index": index,
            }
        )
    if spec.stage in HIGH_RISK_STAGES and rows and matched_high_risk == 0:
        # Some public logs encode the technique in sparse fields without obvious
        # tool names. Mark the final quartile as high-risk so the public MITRE
        # stage still defines an escalation onset without looking at model output.
        start = int(len(rows) * 0.75)
        for row in rows[start:]:
            row["is_high_risk"] = True
            row["severity"] = 5.0
    return rows


def _segment_long_gaps(
    frame: pd.DataFrame,
    max_gap_minutes: int,
    min_segment_events: int,
    max_segment_events: int,
) -> pd.DataFrame:
    segmented: list[pd.DataFrame] = []
    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    for incident_id, group in frame.groupby("incident_id", sort=False):
        ordered = group.sort_values(["timestamp", "event_index"]).copy()
        timestamps = pd.to_datetime(ordered["timestamp"], utc=True, format="mixed")
        segment_ids = timestamps.diff().gt(max_gap).fillna(False).cumsum().to_numpy()
        unique_segments = sorted(set(int(value) for value in segment_ids))
        for segment_id in unique_segments:
            segment = ordered[segment_ids == segment_id].copy()
            part_count = max(1, int(np.ceil(len(segment) / max(max_segment_events, 1))))
            for part_id, start in enumerate(range(0, len(segment), max(max_segment_events, 1))):
                part = segment.iloc[start : start + max(max_segment_events, 1)].copy()
                if len(part) < min_segment_events:
                    continue
                if len(unique_segments) > 1 or part_count > 1:
                    part["incident_id"] = f"{incident_id}/seg{segment_id:02d}p{part_id:02d}"
                segmented.append(part)
    if not segmented:
        raise RuntimeError("All public-data segments were filtered out; reduce --min-segment-events.")
    return pd.concat(segmented, ignore_index=True)


def _future_signature(
    future_slice: np.ndarray,
    time_to_escalation: float,
    horizon_bins: int,
) -> np.ndarray:
    future_category = future_slice[:, : len(ALERT_TYPES)].sum(axis=0)
    future_event_count = future_slice[:, len(ALERT_TYPES)].sum()
    return np.asarray(
        [
            min(time_to_escalation, horizon_bins) / float(max(horizon_bins, 1)),
            future_slice[:, len(ALERT_TYPES) + 1].mean(),
            future_slice[:, len(ALERT_TYPES) + 2].max(),
            future_slice[:, len(ALERT_TYPES) + 3].sum() / float(max(horizon_bins, 1)),
            future_slice[:, len(ALERT_TYPES) + 4].max(),
            future_category[ALERT_TYPES.index("execution")] / float(max(future_event_count, 1.0)),
            future_category[ALERT_TYPES.index("lateral_move")] / float(max(future_event_count, 1.0)),
            future_category[ALERT_TYPES.index("cred_access")] / float(max(future_event_count, 1.0)),
        ],
        dtype=np.float32,
    )


def _build_event_bucket_windows(
    incident_events: pd.DataFrame,
    *,
    event_bucket_size: int,
    lookback_bins: int,
    main_horizon_bins: int,
    aux_horizon_bins: int,
) -> list[dict[str, Any]]:
    ordered = incident_events.sort_values(["timestamp", "event_index"]).reset_index(drop=True).copy()
    if len(ordered) < event_bucket_size * (lookback_bins + 1):
        return []
    ordered["bucket_id"] = np.arange(len(ordered), dtype=np.int64) // int(event_bucket_size)
    feature_rows: list[list[float]] = []
    timestamps: list[int] = []
    for _, bucket in ordered.groupby("bucket_id", sort=True):
        category_counts = [float((bucket["alert_type"] == alert_type).sum()) for alert_type in ALERT_TYPES]
        severities = pd.to_numeric(bucket["severity"], errors="coerce").fillna(0.0)
        host_count = int(bucket["host_hash"].replace("", np.nan).dropna().nunique())
        feature_rows.append(
            category_counts
            + [
                float(len(bucket)),
                float(severities.mean()) if len(severities) else 0.0,
                float(severities.max()) if len(severities) else 0.0,
                float(bucket["is_high_risk"].astype(bool).sum()),
                float(host_count),
            ]
        )
        timestamps.append(int(pd.to_datetime(bucket["timestamp"], utc=True, format="mixed").max().timestamp()))
    feature_values = np.asarray(feature_rows, dtype=np.float32)
    if feature_values.shape[0] < lookback_bins + 2:
        return []

    high_risk_positions = np.where(feature_values[:, len(ALERT_TYPES) + 3] > 0)[0].tolist()
    onset_positions: list[int] = []
    previous_position: int | None = None
    for position in high_risk_positions:
        if previous_position is None or position - previous_position >= 3:
            onset_positions.append(int(position))
        previous_position = int(position)

    rows: list[dict[str, Any]] = []
    incident_id = str(ordered["incident_id"].iloc[0])
    family_id = str(ordered["family_id"].iloc[0])
    for current in range(lookback_bins - 1, feature_values.shape[0] - 1):
        future_onsets = [position for position in onset_positions if position > current]
        next_onset = future_onsets[0] if future_onsets else None
        delta_bins = (next_onset - current) if next_onset is not None else None
        label_main = 1.0 if delta_bins is not None and delta_bins <= main_horizon_bins else 0.0
        label_aux = 1.0 if delta_bins is not None and delta_bins <= aux_horizon_bins else 0.0
        time_to_escalation = float(delta_bins) if delta_bins is not None else float(main_horizon_bins + 3)
        prefix = feature_values[current - lookback_bins + 1 : current + 1].copy()
        future_slice = feature_values[current + 1 : current + 1 + main_horizon_bins]
        if prefix.shape[0] != lookback_bins or future_slice.shape[0] == 0:
            continue
        rows.append(
            {
                "prefix": prefix,
                "label_main": label_main,
                "label_aux": label_aux,
                "future_signature": _future_signature(future_slice, time_to_escalation, main_horizon_bins),
                "time_to_escalation": time_to_escalation,
                "incident_id": incident_id,
                "family_id": family_id,
                "timestamp": timestamps[current],
            }
        )
    return rows


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


def _assign_split(index: int, total: int) -> str:
    train_cut = int(round(total * 0.6))
    dev_cut = int(round(total * 0.8))
    if index < train_cut:
        return "train"
    if index < dev_cut:
        return "dev"
    return "test"


def _write_event_bucket_dataset(frame: pd.DataFrame, dataset_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    windows_by_incident: dict[str, list[dict[str, Any]]] = {}
    incident_stats: list[dict[str, Any]] = []
    for incident_id, incident_events in frame.groupby("incident_id", sort=True):
        rows = _build_event_bucket_windows(
            incident_events,
            event_bucket_size=args.event_bucket_size,
            lookback_bins=args.lookback_bins,
            main_horizon_bins=args.main_horizon_bins,
            aux_horizon_bins=args.aux_horizon_bins,
        )
        if not rows:
            continue
        windows_by_incident[str(incident_id)] = rows
        incident_stats.append(
            {
                "incident_id": str(incident_id),
                "family_id": str(incident_events["family_id"].iloc[0]),
                "events": int(len(incident_events)),
                "windows": int(len(rows)),
                "positive_windows": int(sum(row["label_main"] for row in rows)),
                "high_risk_events": int(incident_events["is_high_risk"].astype(bool).sum()),
            }
        )
    if not windows_by_incident:
        raise RuntimeError("No event-bucket windows were generated from the public probe.")

    positive_incidents = [
        incident_id
        for incident_id, rows in windows_by_incident.items()
        if sum(row["label_main"] for row in rows) > 0
    ]
    negative_incidents = [incident_id for incident_id in windows_by_incident if incident_id not in positive_incidents]
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    incident_family = {
        item["incident_id"]: item["family_id"]
        for item in incident_stats
    }
    for family_id in sorted(set(incident_family.values())):
        family_positive = sorted(
            incident_id
            for incident_id in positive_incidents
            if incident_family.get(incident_id) == family_id
        )
        family_negative = sorted(
            incident_id
            for incident_id in negative_incidents
            if incident_family.get(incident_id) == family_id
        )
        for group in (family_positive, family_negative):
            total = len(group)
            for index, incident_id in enumerate(group):
                split_rows[_assign_split(index, total)].extend(windows_by_incident[incident_id])

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_rows.items():
        if not rows:
            raise RuntimeError(f"Split {split_name} received no event-bucket windows.")
        np.savez(dataset_dir / f"{split_name}.npz", **_to_payload(rows))
    np.savez(dataset_dir / "test_event_disjoint.npz", **_to_payload(split_rows["test"]))

    metadata = {
        "dataset_name": "splunk_attack_data_public_probe",
        "description": "Selected public Splunk Attack Data logs projected into incident-disjoint event-bucket alert windows.",
        "window_mode": "event_bucket",
        "event_bucket_size": int(args.event_bucket_size),
        "lookback_bins": int(args.lookback_bins),
        "main_horizon_buckets": int(args.main_horizon_bins),
        "aux_horizon_buckets": int(args.aux_horizon_bins),
        "feature_channels": ALERT_TYPES
        + ["event_count", "severity_mean", "severity_max", "high_risk_count", "host_count"],
        "analog_fidelity_distance_threshold": 0.45,
        "split_protocol": "incident-disjoint family-stratified split over positive and background active episodes",
        "incidents": incident_stats,
    }
    return metadata


def build_probe(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = Path(args.raw_dir)
    canonical_path = Path(args.canonical_csv)
    dataset_dir = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    source_summary: list[dict[str, Any]] = []
    for spec in CURATED_LOGS:
        local_path = raw_dir / spec.path
        _download(f"{MEDIA_ROOT}/{spec.path}", local_path)
        rows = _canonical_rows(spec, local_path)
        if not rows:
            source_summary.append(
                {
                    "path": spec.path,
                    "technique": spec.technique,
                    "stage": spec.stage,
                    "events": 0,
                    "high_risk_events": 0,
                    "used": False,
                }
            )
            continue
        all_rows.extend(rows)
        source_summary.append(
            {
                "path": spec.path,
                "technique": spec.technique,
                "stage": spec.stage,
                "events": len(rows),
                "high_risk_events": int(sum(bool(row["is_high_risk"]) for row in rows)),
                "used": True,
            }
        )
    if not all_rows:
        raise RuntimeError("No parseable public Splunk Attack Data events were extracted.")
    frame = pd.DataFrame(all_rows).sort_values(["timestamp", "incident_id", "event_index"]).reset_index(drop=True)
    frame = _segment_long_gaps(
        frame,
        max_gap_minutes=args.max_gap_minutes,
        min_segment_events=args.min_segment_events,
        max_segment_events=args.max_segment_events,
    )
    frame = frame.sort_values(["timestamp", "incident_id", "event_index"]).reset_index(drop=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(canonical_path, index=False)

    if args.window_mode == "time_bin":
        metadata = prepare_canonical_alert_dataset(
            {
                "dataset_name": "splunk_attack_data_public_probe",
                "input_path": str(canonical_path),
                "output_dir": str(dataset_dir),
                "bin_minutes": args.bin_minutes,
                "lookback_bins": args.lookback_bins,
                "main_horizon_bins": args.main_horizon_bins,
                "aux_horizon_bins": args.aux_horizon_bins,
                "split_ratios": {"train": 0.6, "dev": 0.2, "test": 0.2},
                "analog_fidelity_distance_threshold": 0.45,
            }
        )
        metadata["window_mode"] = "time_bin"
    elif args.window_mode == "event_bucket":
        metadata = _write_event_bucket_dataset(frame, dataset_dir, args)
    else:
        raise ValueError(f"Unsupported window mode: {args.window_mode}")
    metadata["source_repository"] = "https://github.com/splunk/attack_data"
    metadata["source_media_root"] = MEDIA_ROOT
    metadata["construction_note"] = (
        "Public raw-log probe built from selected Splunk Attack Data files. "
        "Event labels are deterministic MITRE-stage/keyword labels and are not human triage annotations."
    )
    metadata["source_summary"] = source_summary
    metadata["canonical_event_rows"] = int(len(frame))
    with (dataset_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_dir": str(dataset_dir),
        "canonical_csv": str(canonical_path),
        "raw_dir": str(raw_dir),
        "events": int(len(frame)),
        "incidents": int(frame["incident_id"].nunique()),
        "families": int(frame["family_id"].nunique()),
        "high_risk_events": int(frame["is_high_risk"].sum()),
        "stage_counts": frame["stage"].value_counts().sort_index().astype(int).to_dict(),
        "source_summary": source_summary,
        "metadata": metadata,
        "max_gap_minutes": int(args.max_gap_minutes),
        "min_segment_events": int(args.min_segment_events),
        "max_segment_events": int(args.max_segment_events),
        "window_mode": str(args.window_mode),
        "event_bucket_size": int(args.event_bucket_size),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a public Splunk Attack Data probe dataset for Campaign-MEM.")
    parser.add_argument("--raw-dir", default="external_sources/splunk_attack_data_probe/raw")
    parser.add_argument("--canonical-csv", default="data/splunk_attack_data_public_probe/canonical_events.csv")
    parser.add_argument("--output-dir", default="data/splunk_attack_data_public_probe")
    parser.add_argument("--summary-json", default="outputs/results/splunk_attack_data_public_probe_summary.json")
    parser.add_argument("--bin-minutes", type=int, default=1)
    parser.add_argument("--window-mode", choices=("event_bucket", "time_bin"), default="event_bucket")
    parser.add_argument("--event-bucket-size", type=int, default=32)
    parser.add_argument("--lookback-bins", type=int, default=4)
    parser.add_argument("--main-horizon-bins", type=int, default=8)
    parser.add_argument("--aux-horizon-bins", type=int, default=3)
    parser.add_argument("--max-gap-minutes", type=int, default=15)
    parser.add_argument("--min-segment-events", type=int, default=4)
    parser.add_argument("--max-segment-events", type=int, default=1024)
    args = parser.parse_args()
    summary = build_probe(args)
    print(
        "Built Splunk Attack Data probe: "
        f"{summary['events']} events, {summary['incidents']} incidents, "
        f"{summary['families']} families, {summary['high_risk_events']} high-risk events"
    )


if __name__ == "__main__":
    main()
