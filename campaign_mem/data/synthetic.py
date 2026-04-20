from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _family_style(rng: np.random.Generator, feature_dim: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=(feature_dim,)).astype(np.float32)


def _risk_shape_bank(rng: np.random.Generator, signature_dim: int) -> np.ndarray:
    bank = []
    for cluster_id in range(6):
        base = rng.normal(0.0, 0.25, size=(signature_dim,))
        base[0] = cluster_id / 5.0
        base[1] = 1.0 - base[0]
        base[2:] += cluster_id * 0.15
        bank.append(base.astype(np.float32))
    return np.stack(bank, axis=0)


def _make_split(
    rng: np.random.Generator,
    split_name: str,
    sample_count: int,
    family_ids: list[str],
    seq_len: int,
    feature_dim: int,
    signature_bank: np.ndarray,
    start_timestamp: int,
) -> dict[str, np.ndarray]:
    style_map = {family_id: _family_style(rng, feature_dim) for family_id in family_ids}
    incident_ids: list[str] = []
    families: list[str] = []
    prefixes = np.zeros((sample_count, seq_len, feature_dim), dtype=np.float32)
    label_main = np.zeros((sample_count,), dtype=np.float32)
    label_aux = np.zeros((sample_count,), dtype=np.float32)
    future_signature = np.zeros((sample_count, signature_bank.shape[1]), dtype=np.float32)
    time_to_escalation = np.zeros((sample_count,), dtype=np.float32)
    timestamps = np.zeros((sample_count,), dtype=np.int64)

    for index in range(sample_count):
        family_id = family_ids[index % len(family_ids)]
        cluster_id = int(rng.integers(0, len(signature_bank)))
        incident_id = f"{split_name}-incident-{index:05d}"
        style = style_map[family_id]
        prototype = signature_bank[cluster_id]

        severity = float(0.2 + 0.14 * cluster_id + rng.normal(0.0, 0.03))
        tte = float(np.clip(33.0 - cluster_id * 4.2 + rng.normal(0.0, 2.0), 2.0, 30.0))
        y_main = 1.0 if tte <= 30.0 and cluster_id >= 2 else 0.0
        y_aux = 1.0 if tte <= 10.0 and cluster_id >= 4 else 0.0

        trend = np.linspace(0.0, severity, seq_len, dtype=np.float32)[:, None]
        precursor = np.zeros((seq_len, feature_dim), dtype=np.float32)
        precursor[:, 0:4] = trend
        precursor[:, 4:8] = trend[::-1]
        precursor[:, 8:12] = np.sin(np.linspace(0.0, np.pi, seq_len, dtype=np.float32))[:, None] * severity

        alias = signature_bank[(cluster_id + 2) % len(signature_bank)]
        alias = np.resize(alias, (feature_dim,)).astype(np.float32)
        prefix = precursor + style[None, :] * 0.6 + alias[None, :] * 0.25 + rng.normal(0.0, 0.12, size=(seq_len, feature_dim))

        future_signature[index] = prototype + rng.normal(0.0, 0.06, size=prototype.shape)
        prefixes[index] = prefix.astype(np.float32)
        label_main[index] = y_main
        label_aux[index] = y_aux
        time_to_escalation[index] = tte if y_main else 30.0 + float(rng.uniform(0.0, 20.0))
        incident_ids.append(incident_id)
        families.append(family_id)
        timestamps[index] = start_timestamp + index * 300

    return {
        "prefix": prefixes,
        "label_main": label_main,
        "label_aux": label_aux,
        "future_signature": future_signature,
        "time_to_escalation": time_to_escalation,
        "incident_id": np.asarray(incident_ids),
        "family_id": np.asarray(families),
        "timestamp": timestamps,
    }


def generate_synthetic_dataset(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 7))
    rng = np.random.default_rng(seed)

    seq_len = int(config.get("seq_len", 12))
    feature_dim = int(config.get("feature_dim", 16))
    signature_dim = int(config.get("signature_dim", 8))
    counts = config.get(
        "split_counts",
        {"train": 640, "dev": 192, "test": 192, "test_event_disjoint": 192},
    )
    family_counts = config.get(
        "family_counts",
        {"train": 18, "dev": 6, "test": 6, "test_event_disjoint": 8},
    )
    signature_bank = _risk_shape_bank(rng, signature_dim)

    train_families = [f"train-family-{idx:02d}" for idx in range(int(family_counts["train"]))]
    dev_families = [f"dev-family-{idx:02d}" for idx in range(int(family_counts["dev"]))]
    test_families = [f"test-family-{idx:02d}" for idx in range(int(family_counts["test"]))]
    disjoint_families = [
        f"event-family-{idx:02d}" for idx in range(int(family_counts["test_event_disjoint"]))
    ]
    split_specs = {
        "train": (counts["train"], train_families, 0),
        "dev": (counts["dev"], dev_families, 10_000_000),
        "test": (counts["test"], test_families, 20_000_000),
        "test_event_disjoint": (counts["test_event_disjoint"], disjoint_families, 30_000_000),
    }

    for split_name, (sample_count, families, start_ts) in split_specs.items():
        arrays = _make_split(
            rng=rng,
            split_name=split_name,
            sample_count=sample_count,
            family_ids=families,
            seq_len=seq_len,
            feature_dim=feature_dim,
            signature_bank=signature_bank,
            start_timestamp=start_ts,
        )
        np.savez(output_dir / f"{split_name}.npz", **arrays)

    metadata = {
        "dataset_name": str(config.get("dataset_name", "synthetic_cam_lds_smoke")),
        "description": str(
            config.get(
                "description",
                "Synthetic Campaign-MEM benchmark for sanity and smoke testing.",
            )
        ),
        "seed": seed,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "signature_dim": signature_dim,
        "analog_fidelity_distance_threshold": float(config.get("analog_fidelity_distance_threshold", 0.45)),
        "main_horizon_minutes": int(config.get("main_horizon_minutes", 30)),
        "aux_horizon_minutes": int(config.get("aux_horizon_minutes", 10)),
        "split_counts": counts,
        "family_counts": family_counts,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata
