from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np
from torch.utils.data import Dataset


REQUIRED_FIELDS = (
    "prefix",
    "label_main",
    "label_aux",
    "future_signature",
    "time_to_escalation",
    "incident_id",
    "family_id",
    "timestamp",
)


@dataclass
class SplitBundle:
    name: str
    prefix: np.ndarray
    label_main: np.ndarray
    label_aux: np.ndarray
    future_signature: np.ndarray
    time_to_escalation: np.ndarray
    incident_id: np.ndarray
    family_id: np.ndarray
    timestamp: np.ndarray
    metadata: dict[str, Any]

    @property
    def size(self) -> int:
        return int(self.prefix.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.prefix.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.prefix.shape[2])

    def summary_features(self) -> np.ndarray:
        prefix = self.prefix.astype(np.float32)
        mean = prefix.mean(axis=1)
        std = prefix.std(axis=1)
        maximum = prefix.max(axis=1)
        last = prefix[:, -1, :]
        slope = (prefix[:, -1, :] - prefix[:, 0, :]) / max(prefix.shape[1] - 1, 1)
        return np.concatenate([mean, std, maximum, last, slope], axis=-1).astype(np.float32)


class WindowDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, split: SplitBundle) -> None:
        self.split = split

    def __len__(self) -> int:
        return self.split.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "prefix": torch.from_numpy(self.split.prefix[index]).float(),
            "label_main": torch.tensor(self.split.label_main[index], dtype=torch.float32),
            "label_aux": torch.tensor(self.split.label_aux[index], dtype=torch.float32),
            "future_signature": torch.from_numpy(self.split.future_signature[index]).float(),
            "time_to_escalation": torch.tensor(self.split.time_to_escalation[index], dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }


def load_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    meta_path = Path(dataset_dir) / "metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split(dataset_dir: str | Path, split_name: str) -> SplitBundle:
    dataset_path = Path(dataset_dir)
    split_path = dataset_path / f"{split_name}.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    data = np.load(split_path, allow_pickle=True)
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Split {split_name} missing required fields: {missing}")
    metadata = load_metadata(dataset_dir)
    return SplitBundle(
        name=split_name,
        prefix=data["prefix"].astype(np.float32),
        label_main=data["label_main"].astype(np.float32),
        label_aux=data["label_aux"].astype(np.float32),
        future_signature=data["future_signature"].astype(np.float32),
        time_to_escalation=data["time_to_escalation"].astype(np.float32),
        incident_id=data["incident_id"].astype(str),
        family_id=data["family_id"].astype(str),
        timestamp=data["timestamp"].astype(np.int64),
        metadata=metadata,
    )
