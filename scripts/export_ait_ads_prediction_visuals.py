from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data.dataset import SplitBundle, load_split
from campaign_mem.models.forecasting import ParametricForecaster, RetrievalForecaster, build_model
from campaign_mem.training.engine import (
    _device_from_config,
    _encode_split,
    _evaluate_parametric,
    _evaluate_retrieval,
    _make_loader,
    _move_batch,
    _resolve_auto_component_policy,
    _train_epoch,
)
from campaign_mem.utils import load_yaml, save_json, set_seed, to_builtin


def _predict_parametric(
    model: ParametricForecaster,
    split: SplitBundle,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(prefix=batch["prefix"].to(device))
            scores.append(torch.sigmoid(outputs["final_main_logit"]).detach().cpu())
    y_score = torch.cat(scores).numpy().astype(float)
    return {
        "y_true": split.label_main.astype(int).tolist(),
        "y_score": y_score.tolist(),
        "incident_id": split.incident_id.astype(str).tolist(),
        "family_id": split.family_id.astype(str).tolist(),
        "timestamp": split.timestamp.astype(np.int64).tolist(),
        "time_to_escalation": split.time_to_escalation.astype(float).tolist(),
    }


def _predict_retrieval(
    model: RetrievalForecaster,
    train_split: SplitBundle,
    eval_split: SplitBundle,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    train_embedding = _encode_split(model, train_split, device=device, batch_size=batch_size).to(device)
    loader = _make_loader(eval_split, batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        memory_main = torch.from_numpy(train_split.label_main).float().to(device)
        memory_aux = torch.from_numpy(train_split.label_aux).float().to(device)
        for batch in loader:
            outputs = model.forward_with_external_memory(
                prefix=batch["prefix"].to(device),
                memory_embedding=train_embedding,
                memory_main_label=memory_main,
                memory_aux_label=memory_aux,
            )
            scores.append(torch.sigmoid(outputs["final_main_logit"]).detach().cpu())
    y_score = torch.cat(scores).numpy().astype(float)
    return {
        "y_true": eval_split.label_main.astype(int).tolist(),
        "y_score": y_score.tolist(),
        "incident_id": eval_split.incident_id.astype(str).tolist(),
        "family_id": eval_split.family_id.astype(str).tolist(),
        "timestamp": eval_split.timestamp.astype(np.int64).tolist(),
        "time_to_escalation": eval_split.time_to_escalation.astype(float).tolist(),
    }


def _train_and_export(config: dict[str, Any]) -> dict[str, Any]:
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

    requested_model_cfg = deepcopy(model_cfg)
    requested_train_cfg = deepcopy(train_cfg)
    policy_info = None
    if "auto_component_policy" in config:
        model_cfg, train_cfg, policy_info = _resolve_auto_component_policy(
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
    history = []

    for epoch in range(1, int(train_cfg.get("epochs", 6)) + 1):
        train_loss = _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            teacher_models=None,
        )
        if isinstance(model, RetrievalForecaster):
            dev_metrics = _evaluate_retrieval(model, train_split, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
        else:
            dev_metrics = _evaluate_parametric(model, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
        score = dev_metrics["AUPRC"] + 0.01 * dev_metrics.get("Analog-Fidelity@5", 0.0)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_metrics": dev_metrics})
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    if isinstance(model, RetrievalForecaster):
        test_predictions = _predict_retrieval(model, train_split, test_split, device=device, batch_size=batch_size)
        event_predictions = (
            _predict_retrieval(model, train_split, event_split, device=device, batch_size=batch_size)
            if event_split is not None
            else None
        )
        test_metrics = _evaluate_retrieval(model, train_split, test_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
        event_metrics = (
            _evaluate_retrieval(model, train_split, event_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
            if event_split is not None
            else None
        )
    else:
        test_predictions = _predict_parametric(model, test_split, device=device, batch_size=batch_size)
        event_predictions = (
            _predict_parametric(model, event_split, device=device, batch_size=batch_size)
            if event_split is not None
            else None
        )
        test_metrics = _evaluate_parametric(model, test_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
        event_metrics = (
            _evaluate_parametric(model, event_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
            if event_split is not None
            else None
        )

    payload = {
        "experiment_name": config["experiment_name"],
        "dataset_dir": data_cfg["dataset_dir"],
        "model": model_cfg,
        "best_dev_score": best_score,
        "history": history,
        "test": {"metrics": test_metrics, "predictions": test_predictions},
        "test_event_disjoint": {"metrics": event_metrics, "predictions": event_predictions} if event_predictions is not None else None,
    }
    if policy_info is not None:
        payload["auto_component_policy"] = policy_info
        payload["requested_model"] = requested_model_cfg
        payload["requested_training"] = requested_train_cfg
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Export per-window predictions for appendix visualizations.")
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment YAML configs to rerun and export.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    payload = {"runs": []}
    for config_path in args.configs:
        payload["runs"].append(_train_and_export(load_yaml(config_path)))
    save_json(args.output, to_builtin(payload))
    print(f"Saved appendix prediction export: {args.output}")


if __name__ == "__main__":
    main()
