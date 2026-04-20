from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data.dataset import SplitBundle, load_split
from campaign_mem.metrics import auprc, build_metric_report
from campaign_mem.models.forecasting import ParametricForecaster, RetrievalForecaster, build_model
from campaign_mem.training.engine import (
    _device_from_config,
    _encode_split,
    _evaluate_parametric,
    _evaluate_retrieval,
    _make_loader,
    _resolve_auto_component_policy,
    _train_epoch,
)
from campaign_mem.utils import load_yaml, save_json, set_seed


def _train_model(config: dict[str, Any]) -> tuple[torch.nn.Module, SplitBundle, SplitBundle, SplitBundle, SplitBundle | None, torch.device, int]:
    set_seed(int(config.get("seed", 7)))
    data_cfg = config["data"]
    model_cfg = deepcopy(config["model"])
    train_cfg = deepcopy(config.get("training", {}))
    metric_cfg = config.get("metrics", {})

    train_split = load_split(data_cfg["dataset_dir"], data_cfg.get("train_split", "train"))
    dev_split = load_split(data_cfg["dataset_dir"], data_cfg.get("dev_split", "dev"))
    test_split = load_split(data_cfg["dataset_dir"], data_cfg.get("test_split", "test"))
    event_split = None
    event_name = data_cfg.get("event_disjoint_split", "test_event_disjoint")
    if (Path(data_cfg["dataset_dir"]) / f"{event_name}.npz").exists():
        event_split = load_split(data_cfg["dataset_dir"], event_name)

    if "auto_component_policy" in config:
        model_cfg, train_cfg, _ = _resolve_auto_component_policy(
            model_cfg,
            train_cfg,
            train_split,
            dev_split,
            config["auto_component_policy"],
        )

    device = _device_from_config(config)
    batch_size = int(train_cfg.get("batch_size", 64))
    model = build_model(model_cfg, input_dim=train_split.feature_dim, seq_len=train_split.seq_len).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    train_loader = _make_loader(train_split, batch_size=batch_size, shuffle=True)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_score = float("-inf")
    for _ in range(int(train_cfg.get("epochs", 6))):
        _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
        )
        if isinstance(model, RetrievalForecaster):
            dev_metrics = _evaluate_retrieval(model, train_split, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
        else:
            dev_metrics = _evaluate_parametric(model, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
        score = dev_metrics["AUPRC"] + 0.01 * dev_metrics.get("Analog-Fidelity@5", 0.0)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, train_split, dev_split, test_split, event_split, device, batch_size


def _collect_parametric_predictions(
    model: ParametricForecaster,
    eval_split: SplitBundle,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    loader = _make_loader(eval_split, batch_size=batch_size, shuffle=False)
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(prefix=batch["prefix"].to(device))
            scores.append(torch.sigmoid(outputs["final_main_logit"]).detach().cpu().numpy())
    return {"score": np.concatenate(scores, axis=0)}


def _collect_retrieval_predictions(
    model: RetrievalForecaster,
    train_split: SplitBundle,
    eval_split: SplitBundle,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    train_embedding = _encode_split(model, train_split, device=device, batch_size=batch_size).to(device)
    memory_main = torch.from_numpy(train_split.label_main).float().to(device)
    memory_aux = torch.from_numpy(train_split.label_aux).float().to(device)
    loader = _make_loader(eval_split, batch_size=batch_size, shuffle=False)

    scores: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model.forward_with_external_memory(
                prefix=batch["prefix"].to(device),
                memory_embedding=train_embedding,
                memory_main_label=memory_main,
                memory_aux_label=memory_aux,
            )
            scores.append(torch.sigmoid(outputs["final_main_logit"]).detach().cpu().numpy())
            gates.append(outputs["gate"].detach().cpu().numpy())
            indices.append(outputs["retrieved_indices"].detach().cpu().numpy())
    retrieved_indices = np.concatenate(indices, axis=0)
    top1 = retrieved_indices[:, 0]
    return {
        "score": np.concatenate(scores, axis=0),
        "gate": np.concatenate(gates, axis=0),
        "retrieved_indices": retrieved_indices,
        "top1_incident_id": train_split.incident_id[top1].astype(str),
        "top1_family_id": train_split.family_id[top1].astype(str),
        "top1_tte": train_split.time_to_escalation[top1].astype(np.float32),
        "top1_future_distance": np.mean(np.abs(eval_split.future_signature - train_split.future_signature[top1]), axis=1).astype(np.float32),
    }


def _threshold_from_dev(
    model: torch.nn.Module,
    train_split: SplitBundle,
    dev_split: SplitBundle,
    device: torch.device,
    batch_size: int,
    metric_cfg: dict[str, Any],
) -> float:
    if isinstance(model, RetrievalForecaster):
        dev_metrics = _evaluate_retrieval(model, train_split, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
    else:
        dev_metrics = _evaluate_parametric(model, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
    threshold = float(dev_metrics.get("LeadTimeDetail", {}).get("threshold", 0.0))
    return threshold if threshold > 0 else 0.5


def _append_family_metric(container: dict[tuple[str, str, str], list[float]], split_name: str, family_id: str, method_name: str, value: float) -> None:
    container[(split_name, family_id, method_name)].append(float(value))


def _family_rows(
    split_name: str,
    split: SplitBundle,
    method_predictions: dict[str, np.ndarray],
    metrics_store: dict[tuple[str, str, str], list[float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = np.unique(split.family_id.astype(str))
    for family_id in families:
        mask = split.family_id.astype(str) == family_id
        y_true = split.label_main[mask].astype(int)
        positive_n = int(y_true.sum())
        negative_n = int(mask.sum()) - positive_n
        if positive_n == 0 or negative_n == 0:
            continue
        row = {
            "split": split_name,
            "family_id": family_id,
            "n": int(mask.sum()),
            "positive_n": positive_n,
        }
        for method_name, prediction in method_predictions.items():
            metric_value = auprc(y_true, prediction[mask])
            row[f"{method_name}_auprc"] = metric_value
            _append_family_metric(metrics_store, split_name, family_id, method_name, metric_value)
        rows.append(row)
    return rows


def _outcome_bucket(y_true: np.ndarray, y_pred: np.ndarray) -> list[str]:
    buckets: list[str] = []
    for label, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if label == 1 and pred == 1:
            buckets.append("TP")
        elif label == 0 and pred == 1:
            buckets.append("FP")
        elif label == 1 and pred == 0:
            buckets.append("FN")
        else:
            buckets.append("TN")
    return buckets


def _summarize_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
    }


def _format_summary(summary: dict[str, float] | None) -> str:
    if summary is None:
        return "n=0"
    return f"n={summary['n']}, mean={summary['mean']:.3f}, median={summary['median']:.3f}, q1={summary['q1']:.3f}, q3={summary['q3']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gate and family-sliced mechanism analysis on public ATLASv2.")
    parser.add_argument("--prefix-config", required=True)
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--transformer-config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21])
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    prefix_template = load_yaml(args.prefix_config)
    campaign_template = load_yaml(args.campaign_config)
    transformer_template = load_yaml(args.transformer_config)

    gate_values: dict[str, dict[str, list[float]]] = {
        "test": defaultdict(list),
        "test_event_disjoint": defaultdict(list),
    }
    analog_change: dict[str, dict[str, list[float]]] = {
        "test": defaultdict(list),
        "test_event_disjoint": defaultdict(list),
    }
    family_metric_store: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    family_rows_all: list[dict[str, Any]] = []

    for seed in args.seeds:
        prefix_cfg = dict(prefix_template)
        prefix_cfg["seed"] = seed
        campaign_cfg = dict(campaign_template)
        campaign_cfg["seed"] = seed
        transformer_cfg = dict(transformer_template)
        transformer_cfg["seed"] = seed

        prefix_model, prefix_train, prefix_dev, prefix_test, prefix_event, prefix_device, prefix_batch = _train_model(prefix_cfg)
        campaign_model, campaign_train, campaign_dev, campaign_test, campaign_event, campaign_device, campaign_batch = _train_model(campaign_cfg)
        transformer_model, _, transformer_dev, transformer_test, transformer_event, transformer_device, transformer_batch = _train_model(transformer_cfg)

        prefix_threshold = _threshold_from_dev(prefix_model, prefix_train, prefix_dev, prefix_device, prefix_batch, prefix_cfg.get("metrics", {}))
        campaign_threshold = _threshold_from_dev(campaign_model, campaign_train, campaign_dev, campaign_device, campaign_batch, campaign_cfg.get("metrics", {}))
        _ = _threshold_from_dev(transformer_model, campaign_train, transformer_dev, transformer_device, transformer_batch, transformer_cfg.get("metrics", {}))

        split_specs = [
            ("test", campaign_test, prefix_test, transformer_test),
            ("test_event_disjoint", campaign_event, prefix_event, transformer_event),
        ]
        for split_name, campaign_split, prefix_split, transformer_split in split_specs:
            if campaign_split is None or prefix_split is None or transformer_split is None:
                continue
            campaign_pred = _collect_retrieval_predictions(campaign_model, campaign_train, campaign_split, campaign_device, campaign_batch)
            prefix_pred = _collect_retrieval_predictions(prefix_model, prefix_train, prefix_split, prefix_device, prefix_batch)
            transformer_pred = _collect_parametric_predictions(transformer_model, transformer_split, transformer_device, transformer_batch)

            campaign_y_pred = (campaign_pred["score"] >= campaign_threshold).astype(int)
            buckets = _outcome_bucket(campaign_split.label_main, campaign_y_pred)
            for bucket, gate in zip(buckets, campaign_pred["gate"]):
                gate_values[split_name][bucket].append(float(gate))
            positive_mask = campaign_split.label_main.astype(int) == 1
            negative_mask = ~positive_mask
            gate_values[split_name]["positive"].extend([float(v) for v in campaign_pred["gate"][positive_mask]])
            gate_values[split_name]["negative"].extend([float(v) for v in campaign_pred["gate"][negative_mask]])

            same_top1 = campaign_pred["top1_incident_id"] == prefix_pred["top1_incident_id"]
            analog_change[split_name]["same_top1_rate"].append(float(same_top1.mean()))
            analog_change[split_name]["changed_top1_rate"].append(float((~same_top1).mean()))
            analog_change[split_name]["changed_top1_gate_mean"].append(float(campaign_pred["gate"][~same_top1].mean()) if (~same_top1).any() else 0.0)
            analog_change[split_name]["same_top1_gate_mean"].append(float(campaign_pred["gate"][same_top1].mean()) if same_top1.any() else 0.0)
            score_margin = campaign_pred["score"] - prefix_pred["score"]
            analog_change[split_name]["changed_top1_score_margin_mean"].append(float(score_margin[~same_top1].mean()) if (~same_top1).any() else 0.0)
            analog_change[split_name]["same_top1_score_margin_mean"].append(float(score_margin[same_top1].mean()) if same_top1.any() else 0.0)

            family_rows_all.extend(
                _family_rows(
                    split_name=split_name,
                    split=campaign_split,
                    method_predictions={
                        "prefix": prefix_pred["score"],
                        "campaign": campaign_pred["score"],
                        "transformer": transformer_pred["score"],
                    },
                    metrics_store=family_metric_store,
                )
            )

    gate_summary = {
        split_name: {bucket: _summarize_values(values) for bucket, values in split_values.items()}
        for split_name, split_values in gate_values.items()
    }
    analog_summary = {
        split_name: {key: _summarize_values(values) for key, values in split_values.items()}
        for split_name, split_values in analog_change.items()
    }

    aggregated_family_rows: list[dict[str, Any]] = []
    family_keys = sorted({(split_name, family_id) for split_name, family_id, _ in family_metric_store.keys()})
    for split_name, family_id in family_keys:
        row_candidates = [row for row in family_rows_all if row["split"] == split_name and row["family_id"] == family_id]
        if not row_candidates:
            continue
        row = {
            "split": split_name,
            "family_id": family_id,
            "n": row_candidates[0]["n"],
            "positive_n": row_candidates[0]["positive_n"],
        }
        for method_name in ["prefix", "campaign", "transformer"]:
            values = family_metric_store[(split_name, family_id, method_name)]
            row[f"{method_name}_auprc_mean"] = float(mean(values))
            row[f"{method_name}_auprc_std"] = float(pstdev(values) if len(values) > 1 else 0.0)
        row["campaign_minus_prefix"] = row["campaign_auprc_mean"] - row["prefix_auprc_mean"]
        row["campaign_minus_transformer"] = row["campaign_auprc_mean"] - row["transformer_auprc_mean"]
        aggregated_family_rows.append(row)

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seeds": args.seeds,
        "gate_values_raw": {
            split_name: {bucket: [float(value) for value in values] for bucket, values in split_values.items()}
            for split_name, split_values in gate_values.items()
        },
        "gate_summary": gate_summary,
        "analog_change_summary": analog_summary,
        "family_slices": aggregated_family_rows,
    }
    save_json(output_prefix.with_suffix(".json"), payload)

    markdown = [
        "# Public Mechanism Analysis",
        "",
        "## Gate Summary",
        "",
    ]
    for split_name in ["test", "test_event_disjoint"]:
        markdown.append(f"### {split_name}")
        for bucket in ["TP", "FP", "FN", "TN", "positive", "negative"]:
            markdown.append(f"- `{bucket}`: {_format_summary(gate_summary[split_name].get(bucket))}")
        markdown.append("")
        markdown.append("### Analog Change Summary")
        for key in [
            "same_top1_rate",
            "changed_top1_rate",
            "same_top1_gate_mean",
            "changed_top1_gate_mean",
            "same_top1_score_margin_mean",
            "changed_top1_score_margin_mean",
        ]:
            markdown.append(f"- `{key}`: {_format_summary(analog_summary[split_name].get(key))}")
        markdown.append("")

    markdown.extend(
        [
            "## Family Slices",
            "",
            "| Split | Family | n | Pos | Prefix AUPRC | Campaign AUPRC | Transformer AUPRC | Campaign-Prefix | Campaign-Transformer |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregated_family_rows:
        markdown.append(
            f"| {row['split']} | {row['family_id']} | {row['n']} | {row['positive_n']} | "
            f"{row['prefix_auprc_mean']:.3f} +- {row['prefix_auprc_std']:.3f} | "
            f"{row['campaign_auprc_mean']:.3f} +- {row['campaign_auprc_std']:.3f} | "
            f"{row['transformer_auprc_mean']:.3f} +- {row['transformer_auprc_std']:.3f} | "
            f"{row['campaign_minus_prefix']:+.3f} | {row['campaign_minus_transformer']:+.3f} |"
        )
    output_prefix.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Saved mechanism analysis to {output_prefix.with_suffix('.json')} and {output_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
