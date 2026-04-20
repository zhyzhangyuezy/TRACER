from __future__ import annotations

import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data.dataset import SplitBundle, load_split
from campaign_mem.metrics import build_metric_report
from campaign_mem.models.forecasting import ParametricForecaster, RetrievalForecaster, build_model
from campaign_mem.training.engine import (
    _average_metric_reports,
    _average_state_dicts,
    _build_optimizer,
    _build_proxy_event_split_bank,
    _clone_state_dict,
    _device_from_config,
    _encode_split,
    _evaluate_parametric,
    _evaluate_retrieval,
    _init_ema_state,
    _load_state_dict,
    _make_loader,
    _model_selection_score,
    _resolve_positive_weight,
    _resolve_auto_component_policy,
    _train_epoch,
    _update_optimizer_schedule,
)
from campaign_mem.utils import set_seed, to_builtin


@dataclass
class TrainedArtifacts:
    experiment_name: str
    model: torch.nn.Module
    model_cfg: dict[str, Any]
    train_cfg: dict[str, Any]
    metric_cfg: dict[str, Any]
    train_split: SplitBundle
    dev_split: SplitBundle
    test_split: SplitBundle
    event_split: SplitBundle | None
    device: torch.device
    batch_size: int
    policy_info: dict[str, Any] | None
    requested_model: dict[str, Any] | None
    requested_training: dict[str, Any] | None


def train_best_model(config: dict[str, Any]) -> TrainedArtifacts:
    set_seed(int(config.get("seed", 7)))
    data_cfg = config["data"]
    model_cfg = deepcopy(config["model"])
    train_cfg = deepcopy(config.get("training", {}))
    metric_cfg = deepcopy(config.get("metrics", {}))

    train_split = load_split(data_cfg["dataset_dir"], data_cfg.get("train_split", "train"))
    dev_split = load_split(data_cfg["dataset_dir"], data_cfg.get("dev_split", "dev"))
    test_split = load_split(data_cfg["dataset_dir"], data_cfg.get("test_split", "test"))
    event_split = None
    event_name = data_cfg.get("event_disjoint_split", "test_event_disjoint")
    if (Path(data_cfg["dataset_dir"]) / f"{event_name}.npz").exists():
        event_split = load_split(data_cfg["dataset_dir"], event_name)

    requested_model = deepcopy(model_cfg)
    requested_training = deepcopy(train_cfg)
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
    if model_cfg["type"] == "pure_knn":
        raise ValueError("train_best_model does not support pure_knn exports.")
    model = build_model(model_cfg, input_dim=train_split.feature_dim, seq_len=train_split.seq_len).to(device)
    optimizer = _build_optimizer(model, train_cfg)
    train_loader = _make_loader(train_split, batch_size=batch_size, shuffle=True)
    loss_cfg = {
        "main_pos_weight": _resolve_positive_weight(
            train_cfg.get("main_pos_weight"),
            train_split.label_main,
            default=1.0,
        ),
        "aux_pos_weight": _resolve_positive_weight(
            train_cfg.get("aux_pos_weight"),
            train_split.label_aux,
            default=1.0,
        ),
        "focal_gamma": float(train_cfg.get("focal_gamma", 0.0)),
    }
    proxy_event_splits = _build_proxy_event_split_bank(
        train_split,
        seed=int(config.get("seed", 7)),
        family_ratio=float(train_cfg.get("proxy_event_family_ratio", 0.0)),
        min_families=int(train_cfg.get("proxy_event_min_families", 1)),
        num_samples=int(train_cfg.get("proxy_event_num_samples", 1)),
        seed_stride=int(train_cfg.get("proxy_event_seed_stride", 17)),
    )
    ema_state = _init_ema_state(model) if float(train_cfg.get("ema_decay", 0.0)) > 0 else None
    teacher_models: list[torch.nn.Module] = []
    teacher_cfgs = train_cfg.get("teacher_models", [])
    if teacher_cfgs:
        teacher_epochs = int(train_cfg.get("teacher_epochs", train_cfg.get("epochs", 6)))
        for teacher_cfg in teacher_cfgs:
            teacher_model_cfg = deepcopy(teacher_cfg)
            teacher_model = build_model(teacher_model_cfg, input_dim=train_split.feature_dim, seq_len=train_split.seq_len).to(device)
            teacher_train_cfg = {
                **train_cfg,
                "lr": float(train_cfg.get("teacher_lr", train_cfg.get("lr", 1e-3))),
                "weight_decay": float(train_cfg.get("teacher_weight_decay", train_cfg.get("weight_decay", 1e-4))),
                "ema_decay": 0.0,
            }
            teacher_optimizer = _build_optimizer(teacher_model, teacher_train_cfg)
            teacher_best_state = _clone_state_dict(teacher_model)
            teacher_best_score = float("-inf")
            for teacher_epoch in range(1, teacher_epochs + 1):
                _update_optimizer_schedule(teacher_optimizer, teacher_train_cfg, epoch=teacher_epoch, total_epochs=teacher_epochs)
                _train_epoch(
                    model=teacher_model,
                    loader=train_loader,
                    optimizer=teacher_optimizer,
                    device=device,
                    train_cfg={
                        **teacher_train_cfg,
                        "teacher_weight": 0.0,
                        "calibration_penalty_weight": 0.0,
                    },
                    model_cfg=teacher_model_cfg,
                    teacher_models=None,
                    loss_cfg=loss_cfg,
                    ema_state=None,
                )
                teacher_dev_metrics = _evaluate_parametric(
                    teacher_model,
                    dev_split,
                    device=device,
                    batch_size=batch_size,
                    metric_cfg=metric_cfg,
                )
                teacher_score = float(teacher_dev_metrics["AUPRC"])
                if teacher_score > teacher_best_score:
                    teacher_best_score = teacher_score
                    teacher_best_state = _clone_state_dict(teacher_model)
            _load_state_dict(teacher_model, teacher_best_state, device)
            teacher_model.eval()
            teacher_models.append(teacher_model)

    best_state = _clone_state_dict(model)
    best_score = float("-inf")
    checkpoint_average_top_k = max(int(train_cfg.get("checkpoint_average_top_k", 1)), 1)
    model_selection_start_epoch = max(int(train_cfg.get("model_selection_start_epoch", 1)), 1)
    top_states: list[tuple[float, dict[str, torch.Tensor]]] = []
    total_epochs = int(train_cfg.get("epochs", 6))
    for epoch in range(1, total_epochs + 1):
        _update_optimizer_schedule(optimizer, train_cfg, epoch=epoch, total_epochs=total_epochs)
        _train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            teacher_models=teacher_models or None,
            loss_cfg=loss_cfg,
            ema_state=ema_state,
        )
        if ema_state is not None:
            raw_state = _clone_state_dict(model)
            _load_state_dict(model, {key: value.detach().cpu() for key, value in ema_state.items()}, device)
        if isinstance(model, RetrievalForecaster):
            dev_metrics = _evaluate_retrieval(model, train_split, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
        else:
            dev_metrics = _evaluate_parametric(model, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
        if ema_state is not None:
            _load_state_dict(model, raw_state, device)
        proxy_event_score = None
        if proxy_event_splits is not None:
            proxy_reports: list[dict[str, Any]] = []
            proxy_scores: list[float] = []
            for proxy_train_split, proxy_eval_split in proxy_event_splits:
                if isinstance(model, RetrievalForecaster):
                    current_proxy_metrics = _evaluate_retrieval(
                        model,
                        proxy_train_split,
                        proxy_eval_split,
                        device=device,
                        batch_size=batch_size,
                        metric_cfg=metric_cfg,
                    )
                else:
                    current_proxy_metrics = _evaluate_parametric(
                        model,
                        proxy_eval_split,
                        device=device,
                        batch_size=batch_size,
                        metric_cfg=metric_cfg,
                    )
                proxy_reports.append(current_proxy_metrics)
                proxy_scores.append(_model_selection_score(current_proxy_metrics, train_cfg, "proxy_event"))
            _ = _average_metric_reports(proxy_reports)
            proxy_event_score = float(sum(proxy_scores) / max(len(proxy_scores), 1))
        score = float(train_cfg.get("model_selection_dev_weight", 1.0)) * _model_selection_score(
            dev_metrics,
            train_cfg,
            "dev",
        )
        if proxy_event_score is not None:
            score += float(train_cfg.get("model_selection_proxy_event_weight", 0.0)) * proxy_event_score
        if epoch >= model_selection_start_epoch:
            current_state = (
                {key: value.detach().cpu().clone() for key, value in ema_state.items()}
                if ema_state is not None
                else _clone_state_dict(model)
            )
            if score > best_score:
                best_score = score
                best_state = current_state
            top_states.append((score, current_state))
            top_states.sort(key=lambda item: item[0], reverse=True)
            if len(top_states) > checkpoint_average_top_k:
                top_states = top_states[:checkpoint_average_top_k]

    if checkpoint_average_top_k > 1 and top_states:
        best_state = _average_state_dicts([state for _, state in top_states])
    _load_state_dict(model, best_state, device)
    return TrainedArtifacts(
        experiment_name=str(config["experiment_name"]),
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        metric_cfg=metric_cfg,
        train_split=train_split,
        dev_split=dev_split,
        test_split=test_split,
        event_split=event_split,
        device=device,
        batch_size=batch_size,
        policy_info=policy_info,
        requested_model=requested_model,
        requested_training=requested_training,
    )


def dev_threshold(artifacts: TrainedArtifacts) -> float:
    if isinstance(artifacts.model, RetrievalForecaster):
        metrics = _evaluate_retrieval(
            artifacts.model,
            artifacts.train_split,
            artifacts.dev_split,
            device=artifacts.device,
            batch_size=artifacts.batch_size,
            metric_cfg=artifacts.metric_cfg,
        )
    else:
        metrics = _evaluate_parametric(
            artifacts.model,
            artifacts.dev_split,
            device=artifacts.device,
            batch_size=artifacts.batch_size,
            metric_cfg=artifacts.metric_cfg,
        )
    threshold = float(metrics.get("LeadTimeDetail", {}).get("threshold", 0.0))
    return threshold if threshold > 0.0 else 0.5


def _predict_parametric(
    model: ParametricForecaster,
    split: SplitBundle,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(prefix=batch["prefix"].to(device))
            scores.append(torch.sigmoid(outputs["final_main_logit"]).detach().cpu().numpy())
    y_score = np.concatenate(scores, axis=0).astype(np.float64)
    return {
        "y_true": split.label_main.astype(int),
        "y_score": y_score,
        "incident_id": split.incident_id.astype(str),
        "family_id": split.family_id.astype(str),
        "timestamp": split.timestamp.astype(np.int64),
        "time_to_escalation": split.time_to_escalation.astype(np.float64),
    }


def _predict_retrieval(
    model: RetrievalForecaster,
    train_split: SplitBundle,
    eval_split: SplitBundle,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
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
            if "gate" in outputs:
                gates.append(outputs["gate"].detach().cpu().numpy())
            indices.append(outputs["retrieved_indices"].detach().cpu().numpy())

    retrieved_indices = np.concatenate(indices, axis=0).astype(np.int64)
    top1 = retrieved_indices[:, 0]
    payload: dict[str, Any] = {
        "y_true": eval_split.label_main.astype(int),
        "y_score": np.concatenate(scores, axis=0).astype(np.float64),
        "incident_id": eval_split.incident_id.astype(str),
        "family_id": eval_split.family_id.astype(str),
        "timestamp": eval_split.timestamp.astype(np.int64),
        "time_to_escalation": eval_split.time_to_escalation.astype(np.float64),
        "retrieved_indices": retrieved_indices,
        "retrieved_label_main": train_split.label_main[retrieved_indices].astype(int),
        "retrieved_incident_id": train_split.incident_id[retrieved_indices].astype(str),
        "retrieved_family_id": train_split.family_id[retrieved_indices].astype(str),
        "top1_incident_id": train_split.incident_id[top1].astype(str),
        "top1_family_id": train_split.family_id[top1].astype(str),
        "top1_label_main": train_split.label_main[top1].astype(int),
    }
    if gates:
        payload["gate"] = np.concatenate(gates, axis=0).astype(np.float64)
    return payload


def predict_split(artifacts: TrainedArtifacts, split_name: str) -> dict[str, Any]:
    if split_name == "test":
        eval_split = artifacts.test_split
    elif split_name == "test_event_disjoint":
        if artifacts.event_split is not None:
            eval_split = artifacts.event_split
        elif artifacts.test_split.name == "test_event_disjoint":
            eval_split = artifacts.test_split
        else:
            raise ValueError(f"{artifacts.experiment_name} has no event-disjoint split.")
    elif split_name == "dev":
        eval_split = artifacts.dev_split
    else:
        raise ValueError(f"Unsupported split name: {split_name}")

    if isinstance(artifacts.model, RetrievalForecaster):
        predictions = _predict_retrieval(
            artifacts.model,
            artifacts.train_split,
            eval_split,
            device=artifacts.device,
            batch_size=artifacts.batch_size,
        )
        metrics = build_metric_report(
            y_true=predictions["y_true"],
            y_score=predictions["y_score"],
            time_to_escalation=predictions["time_to_escalation"],
            target_precision=float(artifacts.metric_cfg.get("target_precision", 0.8)),
            query_future=eval_split.future_signature,
            memory_future=artifacts.train_split.future_signature,
            retrieved_indices=predictions["retrieved_indices"],
            analog_threshold=float(
                artifacts.metric_cfg.get(
                    "analog_fidelity_distance_threshold",
                    eval_split.metadata.get("analog_fidelity_distance_threshold", 0.45),
                )
            ),
            memory_tte=artifacts.train_split.time_to_escalation,
        )
    else:
        predictions = _predict_parametric(
            artifacts.model,
            eval_split,
            device=artifacts.device,
            batch_size=artifacts.batch_size,
        )
        metrics = build_metric_report(
            y_true=predictions["y_true"],
            y_score=predictions["y_score"],
            time_to_escalation=predictions["time_to_escalation"],
            target_precision=float(artifacts.metric_cfg.get("target_precision", 0.8)),
        )

    return {
        "experiment_name": artifacts.experiment_name,
        "split": split_name,
        "model": deepcopy(artifacts.model_cfg),
        "policy_info": deepcopy(artifacts.policy_info),
        "metrics": to_builtin(metrics),
        "predictions": to_builtin(predictions),
        "threshold": dev_threshold(artifacts),
    }
