from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def summarize_json(path: str) -> None:
    target = ROOT / path
    print(f"--- {path} exists={target.exists()}")
    if not target.exists():
        return
    data = json.loads(target.read_text(encoding="utf-8"))
    print("keys", list(data.keys())[:50])
    for key, value in list(data.items())[:20]:
        size = len(value) if hasattr(value, "__len__") else ""
        sample = None
        if isinstance(value, list) and value:
            sample = value[:2]
        elif isinstance(value, dict):
            sample = list(value.items())[:3]
        print(f"  {key}: {type(value).__name__} {size} {sample if sample is not None else ''}")


def summarize_npz(path: str) -> None:
    target = ROOT / path
    print(f"--- {path} exists={target.exists()}")
    if not target.exists():
        return
    with np.load(target, allow_pickle=True) as data:
        print("files", list(data.files))
        for key in data.files:
            arr = data[key]
            sample = ""
            if arr.ndim == 1 and arr.shape[0] > 0:
                sample = arr[:3].tolist()
            print(f"  {key}: shape={arr.shape} dtype={arr.dtype} sample={sample}")


def main() -> None:
    for path in [
        "data/atlasv2_public/metadata.json",
        "data/ait_ads_public/metadata.json",
        "data/atlas_raw_public/metadata.json",
        "data/atlasv2_workbook/metadata.json",
        "data/synthetic_cam_lds_controlled/metadata.json",
    ]:
        summarize_json(path)
    for path in [
        "data/atlasv2_public/train.npz",
        "data/atlasv2_public/test_event_disjoint.npz",
        "data/ait_ads_public/train.npz",
        "data/ait_ads_public/test.npz",
        "data/atlas_raw_public/train.npz",
        "data/examples/attackbed_suricata_train.npz",
        "data/examples/attackbed_suricata_test.npz",
        "data/examples/cam_lds_attackbed_train.npz",
        "data/examples/cam_lds_attackbed_test.npz",
    ]:
        summarize_npz(path)


if __name__ == "__main__":
    main()
