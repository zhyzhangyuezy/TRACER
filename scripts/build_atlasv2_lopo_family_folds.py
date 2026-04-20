from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "atlasv2_public"
OUTPUT_DIR = ROOT / "data" / "atlasv2_lopo_family"
CONFIG_DIR = ROOT / "configs" / "experiments" / "atlasv2_lopo_family"
RESULT_DIR = ROOT / "outputs" / "results"

SOURCE_SPLITS = ["train", "dev", "test", "test_event_disjoint"]

CONFIG_TEMPLATES = {
    "adaptive": ROOT / "configs" / "experiments" / "r240_tracer_adaptive_event_atlasv2_public.yaml",
    "dlinear": ROOT / "configs" / "experiments" / "r020_dlinear_forecaster_atlasv2_public.yaml",
    "transformer": ROOT / "configs" / "experiments" / "r006_transformer_forecaster_atlasv2_public.yaml",
    "prefix": ROOT / "configs" / "experiments" / "r008_prefix_retrieval_atlasv2_public.yaml",
    "core_bounded": ROOT / "configs" / "experiments" / "r215_campaign_mem_decomp_modular_patch_atlasv2_public.yaml",
    "core_linear": ROOT / "configs" / "experiments" / "r271_tracer_clean_linear_correction_atlasv2_public.yaml",
    "core_no_route": ROOT / "configs" / "experiments" / "r274_tracer_clean_no_route_gates_atlasv2_public.yaml",
}


FIELDS = (
    "prefix",
    "label_main",
    "label_aux",
    "future_signature",
    "time_to_escalation",
    "incident_id",
    "family_id",
    "timestamp",
)


def _load_source() -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {field: [] for field in FIELDS}
    source_name: list[str] = []
    for split in SOURCE_SPLITS:
        payload = np.load(SOURCE_DIR / f"{split}.npz", allow_pickle=True)
        n = int(payload["label_main"].shape[0])
        for field in FIELDS:
            arrays[field].append(payload[field])
        source_name.extend([split] * n)
    merged = {field: np.concatenate(parts, axis=0) for field, parts in arrays.items()}
    merged["source_split"] = np.asarray(source_name)
    return merged


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_split(path: Path, data: dict[str, np.ndarray], mask: np.ndarray) -> None:
    np.savez_compressed(
        path,
        prefix=data["prefix"][mask].astype(np.float32),
        label_main=data["label_main"][mask].astype(np.float32),
        label_aux=data["label_aux"][mask].astype(np.float32),
        future_signature=data["future_signature"][mask].astype(np.float32),
        time_to_escalation=data["time_to_escalation"][mask].astype(np.float32),
        incident_id=data["incident_id"][mask].astype(str),
        family_id=data["family_id"][mask].astype(str),
        timestamp=data["timestamp"][mask].astype(np.int64),
    )


def _split_summary(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    labels = data["label_main"][mask].astype(bool)
    families = data["family_id"][mask].astype(str)
    incidents = data["incident_id"][mask].astype(str)
    return {
        "size": int(mask.sum()),
        "positives": int(labels.sum()),
        "families": int(np.unique(families).size),
        "positive_families": sorted(str(fam) for fam in np.unique(families[labels])),
        "incidents": int(np.unique(incidents).size),
        "positive_incidents": sorted(str(inc) for inc in np.unique(incidents[labels])),
    }


def _load_metadata() -> dict[str, Any]:
    path = SOURCE_DIR / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_configs(folds: list[dict[str, Any]]) -> list[str]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_paths: list[str] = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        dataset_dir = str((OUTPUT_DIR / fold_id).relative_to(ROOT)).replace("\\", "/")
        for method, template_path in CONFIG_TEMPLATES.items():
            config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
            config = deepcopy(config)
            config["experiment_name"] = f"r300_lopo_{fold_id}_{method}"
            config.setdefault("data", {})
            config["data"]["dataset_dir"] = dataset_dir
            config["data"]["test_split"] = "test"
            config["data"]["event_disjoint_split"] = "test_event_disjoint"
            if method == "adaptive":
                config.setdefault("auto_component_policy", {})
                config["auto_component_policy"]["name"] = "tracer_adaptive"
                config["auto_component_policy"]["objective"] = "event_disjoint"
            config.setdefault("output", {})
            config["output"]["dir"] = "outputs/results"
            out_path = CONFIG_DIR / f"r300_lopo_{fold_id}_{method}.yaml"
            out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            config_paths.append(str(out_path.relative_to(ROOT)).replace("\\", "/"))
    return config_paths


def build_folds() -> dict[str, Any]:
    data = _load_source()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = _load_metadata()

    labels = data["label_main"].astype(bool)
    families = data["family_id"].astype(str)
    all_families = sorted(str(fam) for fam in np.unique(families))
    positive_families = sorted(str(fam) for fam in np.unique(families[labels]))
    negative_families = [fam for fam in all_families if fam not in positive_families]
    if len(positive_families) < 3:
        raise ValueError("Need at least three positive families for LOPO construction")

    folds: list[dict[str, Any]] = []
    n_folds = len(positive_families)
    for index, test_family in enumerate(positive_families):
        fold_id = f"fold{index:02d}_{test_family.replace('/', '-')}"
        dev_family = positive_families[(index + 1) % n_folds]
        test_neg = [fam for j, fam in enumerate(negative_families) if j % n_folds == index]
        dev_neg = [fam for j, fam in enumerate(negative_families) if j % n_folds == (index + 1) % n_folds]
        test_families = {test_family, *test_neg}
        dev_families = {dev_family, *dev_neg}
        train_families = set(all_families) - test_families - dev_families

        test_mask = np.isin(families, sorted(test_families))
        dev_mask = np.isin(families, sorted(dev_families))
        train_mask = np.isin(families, sorted(train_families))
        if np.any(test_mask & dev_mask) or np.any(test_mask & train_mask) or np.any(dev_mask & train_mask):
            raise RuntimeError(f"Family overlap in {fold_id}")
        if data["label_main"][test_mask].sum() <= 0 or data["label_main"][dev_mask].sum() <= 0:
            raise RuntimeError(f"Fold {fold_id} lacks positive test/dev labels")
        if data["label_main"][train_mask].sum() <= 0:
            raise RuntimeError(f"Fold {fold_id} lacks positive train labels")

        fold_dir = OUTPUT_DIR / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        _write_split(fold_dir / "train.npz", data, train_mask)
        _write_split(fold_dir / "dev.npz", data, dev_mask)
        _write_split(fold_dir / "test.npz", data, test_mask)
        _write_split(fold_dir / "test_event_disjoint.npz", data, test_mask)
        fold_meta = deepcopy(metadata)
        fold_meta["dataset_name"] = f"atlasv2_lopo_family_{fold_id}"
        fold_meta["source_dataset"] = "atlasv2_public"
        fold_meta["source_splits"] = SOURCE_SPLITS
        fold_meta["fold_id"] = fold_id
        fold_meta["lopo_test_family"] = test_family
        fold_meta["lopo_dev_family"] = dev_family
        fold_meta["lopo_test_negative_families"] = test_neg
        fold_meta["lopo_dev_negative_families"] = dev_neg
        fold_meta["lopo_protocol"] = (
            "Processed-window leave-one-positive-family-out audit. The test family and dev family are disjoint "
            "from training families; negative/background families are partitioned by family."
        )
        (fold_dir / "metadata.json").write_text(
            json.dumps(_jsonable(fold_meta), indent=2), encoding="utf-8"
        )
        folds.append(
            {
                "fold_id": fold_id,
                "test_family": test_family,
                "dev_family": dev_family,
                "train": _split_summary(data, train_mask),
                "dev": _split_summary(data, dev_mask),
                "test": _split_summary(data, test_mask),
            }
        )

    config_paths = _write_configs(folds)
    summary = {
        "source_dataset": str(SOURCE_DIR.relative_to(ROOT)).replace("\\", "/"),
        "output_dataset": str(OUTPUT_DIR.relative_to(ROOT)).replace("\\", "/"),
        "positive_families": positive_families,
        "negative_family_count": len(negative_families),
        "folds": folds,
        "configs": config_paths,
        "seeds": [7, 13, 21],
        "note": "This audit rebuilds folds from processed public windows; it is an additional robustness audit, not a replacement for raw-pipeline benchmark construction.",
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "atlasv2_lopo_family_folds.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    summary = build_folds()
    print(f"Wrote {len(summary['folds'])} LOPO folds under {summary['output_dataset']}")
    print(f"Wrote {len(summary['configs'])} configs under {CONFIG_DIR.relative_to(ROOT)}")
    print("Summary: outputs/results/atlasv2_lopo_family_folds.json")


if __name__ == "__main__":
    main()
