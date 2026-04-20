from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .atlasv2 import collapse_alert_type


def _iter_suricata_alerts(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("event_type") != "alert":
                continue
            alert = event.get("alert", {})
            signature = str(alert.get("signature") or alert.get("category") or "other")
            severity = alert.get("severity", 0)
            rows.append(
                {
                    "timestamp": event.get("timestamp"),
                    "alert_type": collapse_alert_type(signature),
                    "severity": float(severity) if severity is not None else 0.0,
                    "report_text": signature,
                    "src_ip": event.get("src_ip", ""),
                    "dest_ip": event.get("dest_ip", ""),
                    "src_port": event.get("src_port", ""),
                    "dest_port": event.get("dest_port", ""),
                    "proto": event.get("proto", ""),
                    "signature_id": alert.get("signature_id", ""),
                }
            )
    return rows


def normalize_suricata_eve(config: dict[str, Any]) -> dict[str, Any]:
    input_glob = config.get("input_glob")
    if not input_glob:
        raise ValueError("normalize_suricata_eve requires `input_glob`.")
    base_dir = Path(config.get("base_dir", "."))
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path_mode = config.get("path_mode", "relative_parent_stem")
    files = sorted(base_dir.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched input_glob={input_glob!r} under {base_dir}")

    rows = []
    for file_path in files:
        rel = file_path.relative_to(base_dir)
        if path_mode == "stem":
            incident_id = rel.stem
        elif path_mode == "relative":
            incident_id = str(rel.with_suffix("")).replace("\\", "/")
        else:
            parent = rel.parent.name if rel.parent.name else "root"
            incident_id = f"{parent}/{rel.stem}"
        family_id = incident_id.split("/")[0]
        host_hash = rel.parent.name if rel.parent.name else "unknown"
        for row in _iter_suricata_alerts(file_path):
            row["incident_id"] = incident_id
            row["family_id"] = family_id
            row["host_hash"] = host_hash
            row["source_file"] = str(rel).replace("\\", "/")
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No Suricata alert events were extracted.")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    frame = frame.sort_values(["incident_id", "timestamp"]).reset_index(drop=True)
    frame["stage"] = ""
    frame["is_high_risk"] = False
    frame.to_csv(output_path, index=False)
    return {
        "output_path": str(output_path),
        "files": len(files),
        "rows": int(len(frame)),
        "incidents": int(frame["incident_id"].nunique()),
    }
