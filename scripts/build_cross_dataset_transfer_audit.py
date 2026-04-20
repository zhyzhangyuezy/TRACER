from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.training import run_experiment


DATASETS = {
    "atlasv2": {
        "path": ROOT / "data" / "atlasv2_public",
        "label": "ATLASv2",
        "test_split": "test",
    },
    "aitads": {
        "path": ROOT / "data" / "ait_ads_public",
        "label": "AIT-ADS",
        "test_split": "test",
    },
}

TRANSFER_PAIRS = [
    ("atlasv2", "aitads"),
    ("aitads", "atlasv2"),
]

COMMON_CHANNELS = [
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
]

REQUIRED_FIELDS = [
    "prefix",
    "label_main",
    "label_aux",
    "future_signature",
    "time_to_escalation",
    "incident_id",
    "family_id",
    "timestamp",
]

METHODS = {
    "DLinear": {
        "model": {
            "type": "dlinear",
            "encoder": "dlinear",
            "hidden_dim": 128,
            "embedding_dim": 128,
            "use_auxiliary": True,
        },
        "training": {
            "epochs": 8,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "auxiliary_weight": 0.25,
        },
    },
    "Prefix-Only": {
        "model": {
            "type": "prefix_retrieval",
            "encoder": "transformer",
            "hidden_dim": 96,
            "embedding_dim": 96,
            "top_k": 5,
            "use_auxiliary": True,
            "use_contrastive": False,
            "use_hard_negatives": False,
            "use_utility": False,
        },
        "training": {
            "epochs": 8,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "auxiliary_weight": 0.25,
        },
    },
    "TRACER adaptive": {
        "model": {
            "type": "campaign_mem_decomp_modular",
            "retrieval_encoder": "transformer",
            "stable_encoder": "dlinear",
            "shock_encoder": "patchtst",
            "hidden_dim": 128,
            "embedding_dim": 128,
            "top_k": 5,
            "similarity_temperature": 0.15,
            "delta_scale": 0.12,
            "trend_kernel": 5,
            "use_auxiliary": True,
            "use_contrastive": True,
            "use_hard_negatives": True,
            "use_utility": False,
            "use_abstention": True,
            "use_uncertainty_gate": False,
            "use_shift_gate": True,
            "use_aggressive_gate": True,
            "aggressive_route_on_delta": True,
            "selector_agreement_floor": 0.20,
            "shift_floor": 0.35,
            "aggressive_gate_floor": 0.15,
        },
        "auto_component_policy": {
            "name": "tracer_adaptive",
            "objective": "balanced",
        },
        "training": {
            "epochs": 12,
            "batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "auxiliary_weight": 0.25,
            "contrastive_weight": 0.2,
            "calibration_penalty_weight": 0.05,
        },
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_split(dataset_dir: Path, split_name: str) -> dict[str, np.ndarray]:
    with np.load(dataset_dir / f"{split_name}.npz", allow_pickle=True) as handle:
        return {field: np.asarray(handle[field]) for field in REQUIRED_FIELDS}


def _channel_indices(dataset_dir: Path) -> list[int]:
    metadata = _load_json(dataset_dir / "metadata.json")
    channels = [str(item) for item in metadata["feature_channels"]]
    missing = [channel for channel in COMMON_CHANNELS if channel not in channels]
    if missing:
        raise ValueError(f"{dataset_dir} missing common channels: {missing}")
    return [channels.index(channel) for channel in COMMON_CHANNELS]


def _project(split: dict[str, np.ndarray], channel_indices: list[int]) -> dict[str, np.ndarray]:
    out = {field: np.asarray(value).copy() for field, value in split.items()}
    out["prefix"] = np.asarray(split["prefix"], dtype=np.float32)[:, :, channel_indices]
    return out


def _select(split: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {field: np.asarray(value)[indices] for field, value in split.items()}


def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    non_empty = [part for part in parts if int(part["label_main"].shape[0]) > 0]
    if not non_empty:
        raise ValueError("Cannot concatenate empty split list.")
    return {field: np.concatenate([part[field] for part in non_empty], axis=0) for field in REQUIRED_FIELDS}


def _empty_like(split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {field: np.asarray(value)[:0] for field, value in split.items()}


def _repeat_split(split: dict[str, np.ndarray], repeat: int) -> dict[str, np.ndarray]:
    repeat = max(int(repeat), 1)
    if repeat == 1 or int(split["label_main"].shape[0]) == 0:
        return split
    return {field: np.repeat(np.asarray(value), repeat, axis=0) for field, value in split.items()}


def _sample_support(split: dict[str, np.ndarray], positive_count: int, seed: int, neg_ratio: int = 3) -> dict[str, np.ndarray]:
    if positive_count <= 0:
        return _empty_like(split)
    rng = np.random.default_rng(seed)
    labels = np.asarray(split["label_main"]).astype(int)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    pos_n = min(int(positive_count), int(positive.shape[0]))
    neg_n = min(int(neg_ratio * pos_n), int(negative.shape[0]))
    if pos_n == 0:
        return _empty_like(split)
    pos_idx = rng.choice(positive, size=pos_n, replace=False)
    neg_idx = rng.choice(negative, size=neg_n, replace=False) if neg_n > 0 else np.asarray([], dtype=int)
    indices = np.concatenate([pos_idx, neg_idx]).astype(int)
    rng.shuffle(indices)
    return _select(split, indices)


def _standardize(train: dict[str, np.ndarray], dev: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    train_prefix = np.asarray(train["prefix"], dtype=np.float32)
    mean_values = train_prefix.mean(axis=(0, 1), keepdims=True)
    std_values = train_prefix.std(axis=(0, 1), keepdims=True)
    std_values = np.where(std_values < 1e-6, 1.0, std_values)
    out: dict[str, dict[str, np.ndarray]] = {}
    for name, split in [("train", train), ("dev", dev), ("test", test)]:
        current = {field: np.asarray(value).copy() for field, value in split.items()}
        current["prefix"] = ((np.asarray(split["prefix"], dtype=np.float32) - mean_values) / std_values).astype(np.float32)
        out[name] = current
    out["_stats"] = {
        "mean": mean_values.reshape(-1).astype(float),
        "std": std_values.reshape(-1).astype(float),
    }
    return out


def _save_split(path: Path, split: dict[str, np.ndarray]) -> None:
    arrays = {
        "prefix": np.asarray(split["prefix"], dtype=np.float32),
        "label_main": np.asarray(split["label_main"], dtype=np.float32),
        "label_aux": np.asarray(split["label_aux"], dtype=np.float32),
        "future_signature": np.asarray(split["future_signature"], dtype=np.float32),
        "time_to_escalation": np.asarray(split["time_to_escalation"], dtype=np.float32),
        "incident_id": np.asarray(split["incident_id"]).astype(str),
        "family_id": np.asarray(split["family_id"]).astype(str),
        "timestamp": np.asarray(split["timestamp"], dtype=np.int64),
    }
    np.savez_compressed(path, **arrays)


def _split_summary(split: dict[str, np.ndarray]) -> dict[str, Any]:
    labels = np.asarray(split["label_main"]).astype(int)
    return {
        "samples": int(labels.shape[0]),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "families": int(len(set(np.asarray(split["family_id"]).astype(str).tolist()))),
        "incidents": int(len(set(np.asarray(split["incident_id"]).astype(str).tolist()))),
    }


def build_transfer_dataset(source: str, target: str, shot: int, seed: int, support_repeat: int = 1) -> Path:
    source_dir = DATASETS[source]["path"]
    target_dir = DATASETS[target]["path"]
    source_idx = _channel_indices(source_dir)
    target_idx = _channel_indices(target_dir)

    source_train = _project(_load_split(source_dir, "train"), source_idx)
    source_dev = _project(_load_split(source_dir, "dev"), source_idx)
    target_train = _project(_load_split(target_dir, "train"), target_idx)
    target_dev = _project(_load_split(target_dir, "dev"), target_idx)
    target_test = _project(_load_split(target_dir, str(DATASETS[target]["test_split"])), target_idx)

    train_support_raw = _sample_support(target_train, shot, seed=seed * 1009 + 17)
    dev_support_raw = _sample_support(target_dev, max(1, shot // 2) if shot > 0 else 0, seed=seed * 1009 + 53)
    train_support = _repeat_split(train_support_raw, support_repeat if shot > 0 else 1)
    dev_support = _repeat_split(dev_support_raw, support_repeat if shot > 0 else 1)

    train = _concat([source_train, train_support])
    dev = _concat([source_dev, dev_support])
    standardized = _standardize(train, dev, target_test)

    repeat_suffix = "" if int(support_repeat) == 1 else f"_rep{int(support_repeat)}"
    dataset_dir = ROOT / "data" / "cross_dataset_transfer" / f"{source}_to_{target}_shot{shot}_seed{seed}{repeat_suffix}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _save_split(dataset_dir / "train.npz", standardized["train"])
    _save_split(dataset_dir / "dev.npz", standardized["dev"])
    _save_split(dataset_dir / "test.npz", standardized["test"])

    metadata = {
        "dataset_name": f"cross_transfer_{source}_to_{target}_shot{shot}_seed{seed}",
        "description": "Common-channel cross-dataset transfer/few-shot audit dataset.",
        "source_dataset": source,
        "target_dataset": target,
        "target_test_split": DATASETS[target]["test_split"],
        "target_positive_support": int(shot),
        "target_support_repeat": int(support_repeat),
        "support_sampling_seed": int(seed),
        "feature_channels": COMMON_CHANNELS,
        "projection": "shared 15 current-prefix channels; dataset-specific host and count channels removed",
        "normalization": "z-score using the constructed train split only",
        "analog_fidelity_distance_threshold": 0.35,
        "source_train": _split_summary(source_train),
        "source_dev": _split_summary(source_dev),
        "target_train_support_raw": _split_summary(train_support_raw),
        "target_dev_support_raw": _split_summary(dev_support_raw),
        "target_train_support_after_repeat": _split_summary(train_support),
        "target_dev_support_after_repeat": _split_summary(dev_support),
        "train": _split_summary(standardized["train"]),
        "dev": _split_summary(standardized["dev"]),
        "test": _split_summary(standardized["test"]),
        "zscore_mean": standardized["_stats"]["mean"].tolist(),
        "zscore_std": standardized["_stats"]["std"].tolist(),
    }
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dataset_dir


def _base_config(method: str, dataset_dir: Path, source: str, target: str, shot: int, seed: int, support_repeat: int = 1) -> dict[str, Any]:
    method_spec = deepcopy(METHODS[method])
    safe_method = method.lower().replace(" ", "_").replace("-", "only").replace("+", "plus")
    repeat_suffix = "" if int(support_repeat) == 1 else f"_rep{int(support_repeat)}"
    config: dict[str, Any] = {
        "experiment_name": f"r330_transfer_{safe_method}_{source}_to_{target}_shot{shot}_seed{seed}{repeat_suffix}",
        "seed": int(seed),
        "device": "cuda",
        "data": {
            "dataset_dir": str(dataset_dir.relative_to(ROOT)).replace("\\", "/"),
            "test_split": "test",
            "event_disjoint_split": "skip_event_copy",
        },
        "model": method_spec["model"],
        "training": method_spec["training"],
        "metrics": {
            "target_precision": 0.8,
            "analog_fidelity_distance_threshold": 0.35,
        },
        "output": {
            "dir": "outputs/results/cross_dataset_transfer",
        },
    }
    if "auto_component_policy" in method_spec:
        config["auto_component_policy"] = method_spec["auto_component_policy"]
    return config


def _write_config(config: dict[str, Any]) -> Path:
    config_dir = ROOT / "configs" / "experiments" / "cross_dataset_transfer"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{config['experiment_name']}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run_one(config: dict[str, Any], force: bool) -> dict[str, Any]:
    result_path = ROOT / str(config["output"]["dir"]) / f"{config['experiment_name']}.json"
    if result_path.exists() and not force:
        return _load_json(result_path)
    return run_experiment(config)


def _summarize(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(mean(finite)), float(pstdev(finite))


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}"


def _fmt_pm(avg: float, std: float) -> str:
    if not np.isfinite(avg):
        return "--"
    return f"{avg:.3f} $\\pm$ {std:.3f}"


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in runs:
        key = (row["source"], row["target"], int(row["shot"]), row["method"])
        groups.setdefault(key, []).append(row)

    rows = []
    for (source, target, shot, method), items in sorted(groups.items()):
        auprc_avg, auprc_std = _summarize([item["test"]["AUPRC"] for item in items])
        auroc_avg, auroc_std = _summarize([item["test"].get("AUROC", float("nan")) for item in items])
        brier_avg, brier_std = _summarize([item["test"].get("Brier", float("nan")) for item in items])
        af_avg, af_std = _summarize([item["test"].get("Analog-Fidelity@5", float("nan")) for item in items])
        regimes = sorted(set(str(item.get("regime", "")) for item in items if item.get("regime")))
        rows.append(
            {
                "source": source,
                "target": target,
                "setting": f"{DATASETS[source]['label']} -> {DATASETS[target]['label']}",
                "shot": int(shot),
                "method": method,
                "seeds": [int(item["seed"]) for item in items],
                "AUPRC": auprc_avg,
                "AUPRC_std": auprc_std,
                "AUROC": auroc_avg,
                "AUROC_std": auroc_std,
                "Brier": brier_avg,
                "Brier_std": brier_std,
                "AF@5": af_avg,
                "AF@5_std": af_std,
                "regimes": regimes,
            }
        )
    return {
        "audit": "Cross-dataset transfer and few-shot adaptation audit",
        "common_channels": COMMON_CHANNELS,
        "rows": rows,
        "runs": runs,
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Cross-dataset transfer and few-shot adaptation audit",
        "",
        "All datasets are projected to the shared 15 current-prefix feature channels and z-scored using the constructed train split only. Few-shot support is sampled from the target-domain train/dev splits; target test windows are never sampled as support.",
        "",
        "| Setting | Shot positives | Method | AUPRC | AUROC | Brier | AF@5 | Route regime |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {setting} | {shot} | {method} | {auprc} | {auroc} | {brier} | {af} | {regime} |".format(
                setting=row["setting"],
                shot=row["shot"],
                method=row["method"],
                auprc=f"{row['AUPRC']:.3f} +/- {row['AUPRC_std']:.3f}",
                auroc=f"{row['AUROC']:.3f} +/- {row['AUROC_std']:.3f}",
                brier=f"{row['Brier']:.3f} +/- {row['Brier_std']:.3f}",
                af="--" if not np.isfinite(row["AF@5"]) else f"{row['AF@5']:.1f} +/- {row['AF@5_std']:.1f}",
                regime=", ".join(row["regimes"]) if row["regimes"] else "--",
            )
        )
    output = ROOT / "outputs" / "results" / "cross_dataset_transfer_audit.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(summary: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Cross-dataset transfer and few-shot adaptation audit. Each source--target pair is projected to the shared 15 current-prefix channels; target support is sampled only from target train/dev splits, and target test windows are never used for support. Values are mean $\pm$ standard deviation over seeds.}",
        r"\label{tab:cross-dataset-transfer}",
        r"\maxtablewidth{",
        r"\begin{tabular}{llrlrrrr}",
        r"\toprule",
        r"Setting & Method & Shot & Route & AUPRC & AUROC & Brier & AF@5 \\",
        r"\midrule",
    ]
    previous = None
    for row in summary["rows"]:
        setting = row["setting"] if row["setting"] != previous else ""
        route = ", ".join(row["regimes"]) if row["regimes"] else "--"
        lines.append(
            "{setting} & {method} & {shot} & {route} & {auprc} & {auroc} & {brier} & {af} \\\\".format(
                setting=setting.replace("_", r"\_"),
                method=row["method"].replace("_", r"\_"),
                shot=row["shot"],
                route=route.replace("_", r"\_"),
                auprc=_fmt_pm(row["AUPRC"], row["AUPRC_std"]),
                auroc=_fmt_pm(row["AUROC"], row["AUROC_std"]),
                brier=_fmt_pm(row["Brier"], row["Brier_std"]),
                af="--" if not np.isfinite(row["AF@5"]) else f"{row['AF@5']:.1f} $\\pm$ {row['AF@5_std']:.1f}",
            )
        )
        previous = row["setting"]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    (ROOT / "figures" / "tab_cross_dataset_transfer.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run cross-dataset transfer/few-shot audit.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21])
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 5, 20])
    parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()))
    parser.add_argument("--support-repeat", type=int, default=1)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runs: list[dict[str, Any]] = []
    for source, target in TRANSFER_PAIRS:
        for shot in args.shots:
            for seed in args.seeds:
                dataset_dir = build_transfer_dataset(source, target, shot, seed, support_repeat=int(args.support_repeat))
                for method in args.methods:
                    config = _base_config(method, dataset_dir, source, target, shot, seed, support_repeat=int(args.support_repeat))
                    config_path = _write_config(config)
                    if args.build_only:
                        continue
                    result = _run_one(config, force=bool(args.force))
                    row = {
                        "source": source,
                        "target": target,
                        "shot": int(shot),
                        "seed": int(seed),
                        "method": method if int(args.support_repeat) == 1 else f"{method} + replay{int(args.support_repeat)}",
                        "base_method": method,
                        "support_repeat": int(args.support_repeat),
                        "config": str(config_path.relative_to(ROOT)).replace("\\", "/"),
                        "dataset_dir": str(dataset_dir.relative_to(ROOT)).replace("\\", "/"),
                        "test": result["test"],
                    }
                    policy = result.get("auto_component_policy", {})
                    if policy:
                        row["regime"] = policy.get("regime")
                        row["resolved_model_type"] = policy.get("resolved_model_type")
                    runs.append(row)
                    print(
                        f"{source}->{target} shot={shot} seed={seed} method={method}: "
                        f"AUPRC={float(result['test']['AUPRC']):.4f}"
                    )

    if args.build_only:
        print("Built datasets/configs only.")
        return

    summary = aggregate(runs)
    output_json = ROOT / "outputs" / "results" / "cross_dataset_transfer_audit.json"
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)
    write_latex(summary)
    print("Wrote outputs/results/cross_dataset_transfer_audit.json")
    print("Wrote figures/tab_cross_dataset_transfer.tex")


if __name__ == "__main__":
    main()
