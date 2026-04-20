from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import SplitBundle, load_metadata, load_split
from ..utils import save_json


def _positive_rate(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _tte_stats(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    positives = values[labels.astype(bool)]
    if positives.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(positives)),
        "median": float(np.median(positives)),
        "min": float(np.min(positives)),
        "max": float(np.max(positives)),
    }


def _split_summary(split: SplitBundle) -> dict[str, Any]:
    return {
        "samples": split.size,
        "seq_len": split.seq_len,
        "feature_dim": split.feature_dim,
        "positive_rate_main": _positive_rate(split.label_main),
        "positive_rate_aux": _positive_rate(split.label_aux),
        "incident_count": int(len(np.unique(split.incident_id))),
        "family_count": int(len(np.unique(split.family_id))),
        "future_signature_dim": int(split.future_signature.shape[-1]),
        "time_to_escalation_positive": _tte_stats(split.time_to_escalation, split.label_main),
    }


def audit_dataset(
    dataset_dir: str | Path,
    split_names: list[str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    metadata = load_metadata(dataset_dir)
    split_names = split_names or ["train", "dev", "test", "test_event_disjoint"]
    available_splits = [name for name in split_names if (dataset_dir / f"{name}.npz").exists()]
    if not available_splits:
        raise FileNotFoundError(f"No splits found in {dataset_dir}")

    splits = {name: load_split(dataset_dir, name) for name in available_splits}
    overlap_report: dict[str, dict[str, int]] = {}
    for left_name, right_name in combinations(available_splits, 2):
        left = splits[left_name]
        right = splits[right_name]
        overlap_report[f"{left_name}__{right_name}"] = {
            "incident_overlap": int(len(set(left.incident_id) & set(right.incident_id))),
            "family_overlap": int(len(set(left.family_id) & set(right.family_id))),
        }

    report = {
        "dataset_dir": str(dataset_dir),
        "metadata": metadata,
        "splits": {name: _split_summary(split) for name, split in splits.items()},
        "overlap": overlap_report,
        "checks": {
            "incident_leakage_free": all(item["incident_overlap"] == 0 for item in overlap_report.values()),
            "event_disjoint_family_free": all(
                item["family_overlap"] == 0
                for key, item in overlap_report.items()
                if "test_event_disjoint" in key
            ),
        },
    }
    if output_path is not None:
        save_json(output_path, report)
    return report
