from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ..data.dataset import SplitBundle, WindowDataset, load_split
from ..metrics import build_metric_report
from ..models.forecasting import (
    ParametricForecaster,
    RetrievalForecaster,
    build_model,
    pairwise_future_contrastive_loss,
    retrieval_utility_loss,
)
from ..utils import save_json, set_seed, to_builtin


def _make_loader(split: SplitBundle, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(WindowDataset(split), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _slice_split(split: SplitBundle, mask: np.ndarray, name: str) -> SplitBundle:
    return SplitBundle(
        name=name,
        prefix=split.prefix[mask],
        label_main=split.label_main[mask],
        label_aux=split.label_aux[mask],
        future_signature=split.future_signature[mask],
        time_to_escalation=split.time_to_escalation[mask],
        incident_id=split.incident_id[mask],
        family_id=split.family_id[mask],
        timestamp=split.timestamp[mask],
        metadata=deepcopy(split.metadata),
    )


def _device_from_config(config: dict[str, Any]) -> torch.device:
    requested = config.get("device", "cuda")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device=device) for key, value in state_dict.items()})


def _average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("Cannot average an empty list of state dicts.")
    if len(state_dicts) == 1:
        return {key: value.detach().cpu().clone() for key, value in state_dicts[0].items()}
    averaged: dict[str, torch.Tensor] = {}
    keys = state_dicts[0].keys()
    for key in keys:
        tensors = [state[key].detach().cpu() for state in state_dicts]
        first = tensors[0]
        if torch.is_floating_point(first):
            averaged[key] = torch.stack(tensors, dim=0).mean(dim=0)
        else:
            averaged[key] = first.clone()
    return averaged


def _init_ema_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _update_ema_state(model: nn.Module, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    with torch.no_grad():
        for key, value in model.state_dict().items():
            ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def _resolve_positive_weight(
    spec: Any,
    labels: np.ndarray,
    *,
    default: float = 1.0,
    max_weight: float = 12.0,
) -> float:
    if spec is None:
        return default
    if isinstance(spec, (float, int)):
        return float(max(spec, 1.0))
    mode = str(spec).strip().lower()
    positives = float(labels.sum())
    negatives = float(labels.shape[0] - positives)
    if positives <= 0 or negatives <= 0:
        return default
    ratio = negatives / positives
    if mode == "auto":
        return float(min(max(ratio, 1.0), max_weight))
    if mode in {"auto_sqrt", "sqrt"}:
        return float(min(max(np.sqrt(ratio), 1.0), max_weight))
    return default


def _binary_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float = 1.0,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    reduction = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype),
    )
    if focal_gamma > 0:
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        reduction = reduction * torch.pow((1.0 - p_t).clamp_min(1e-6), focal_gamma)
    return reduction.mean()


def _optimizer_param_groups(
    model: nn.Module,
    train_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    base_lr = float(train_cfg.get("lr", 1e-3))
    base_weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add_group(name: str, params: list[nn.Parameter], lr_scale: float, wd_scale: float = 1.0) -> None:
        filtered = [parameter for parameter in params if parameter.requires_grad and id(parameter) not in seen]
        if not filtered:
            return
        seen.update(id(parameter) for parameter in filtered)
        groups.append(
            {
                "name": name,
                "params": filtered,
                "lr": base_lr * lr_scale,
                "weight_decay": base_weight_decay * wd_scale,
                "base_lr": base_lr * lr_scale,
                "base_weight_decay": base_weight_decay * wd_scale,
            }
        )

    def module_params(name: str) -> list[nn.Parameter]:
        module = getattr(model, name, None)
        if module is None:
            return []
        return list(module.parameters())

    add_group("retrieval_encoder", module_params("encoder"), float(train_cfg.get("retrieval_lr_scale", 1.0)))
    add_group("retrieval_heads", module_params("main_head") + module_params("aux_head"), float(train_cfg.get("retrieval_head_lr_scale", 1.0)))
    add_group("forecast_encoder", module_params("forecast_encoder"), float(train_cfg.get("forecast_lr_scale", 1.0)))
    add_group(
        "forecast_heads",
        module_params("forecast_main_head") + module_params("forecast_aux_head"),
        float(train_cfg.get("forecast_head_lr_scale", train_cfg.get("forecast_lr_scale", 1.0))),
    )
    add_group("gate_head", module_params("gate_head"), float(train_cfg.get("gate_lr_scale", 1.0)))
    add_group(
        "calibration_heads",
        module_params("main_correction_head")
        + module_params("aux_correction_head")
        + module_params("main_calibration_head")
        + module_params("aux_calibration_head"),
        float(train_cfg.get("calibration_lr_scale", 1.0)),
    )
    add_group(
        "selector_heads",
        module_params("main_selector_head")
        + module_params("aux_selector_head")
        + module_params("retrieval_reliability_head"),
        float(train_cfg.get("selector_lr_scale", train_cfg.get("calibration_lr_scale", 1.0))),
    )
    remaining = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in seen]
    add_group("remaining", remaining, float(train_cfg.get("remaining_lr_scale", 1.0)))
    return groups


def _build_optimizer(model: nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    param_groups = _optimizer_param_groups(model, train_cfg)
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer construction.")
    optimizer_type = str(train_cfg.get("optimizer", "adamw")).strip().lower()
    if optimizer_type != "adamw":
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    return torch.optim.AdamW(
        param_groups,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        betas=tuple(train_cfg.get("betas", (0.9, 0.999))),
    )


def _schedule_multiplier(progress: float, train_cfg: dict[str, Any]) -> float:
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    warmdown_ratio = float(train_cfg.get("warmdown_ratio", 0.0))
    final_lr_frac = float(train_cfg.get("final_lr_frac", 1.0))
    if warmup_ratio > 0 and progress < warmup_ratio:
        return max(progress / max(warmup_ratio, 1e-8), 1e-3)
    if warmdown_ratio > 0 and progress > 1.0 - warmdown_ratio:
        cooldown = max((1.0 - progress) / max(warmdown_ratio, 1e-8), 0.0)
        return cooldown + (1.0 - cooldown) * final_lr_frac
    return 1.0


def _weight_decay_multiplier(progress: float, train_cfg: dict[str, Any]) -> float:
    final_frac = float(train_cfg.get("final_weight_decay_frac", 1.0))
    return (1.0 - progress) + progress * final_frac


def _stage_lr_multiplier(group_name: str, epoch: int, train_cfg: dict[str, Any]) -> float:
    stage_specs = (
        (
            {"forecast_encoder", "forecast_heads"},
            int(train_cfg.get("freeze_forecast_until_epoch", 0)),
            int(train_cfg.get("forecast_ramp_epochs", 0)),
        ),
        (
            {"calibration_heads", "selector_heads"},
            int(train_cfg.get("freeze_calibration_until_epoch", 0)),
            int(train_cfg.get("calibration_ramp_epochs", 0)),
        ),
        (
            {"gate_head"},
            int(train_cfg.get("freeze_gate_until_epoch", 0)),
            int(train_cfg.get("gate_ramp_epochs", 0)),
        ),
        (
            {"retrieval_encoder", "retrieval_heads"},
            int(train_cfg.get("freeze_retrieval_until_epoch", 0)),
            int(train_cfg.get("retrieval_ramp_epochs", 0)),
        ),
    )
    for names, freeze_until_epoch, ramp_epochs in stage_specs:
        if group_name not in names:
            continue
        if epoch <= freeze_until_epoch:
            return 0.0
        if ramp_epochs > 0 and epoch <= freeze_until_epoch + ramp_epochs:
            progress = (epoch - freeze_until_epoch) / max(ramp_epochs, 1)
            return float(min(max(progress, 0.0), 1.0))
        return 1.0
    return 1.0


def _update_optimizer_schedule(
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    *,
    epoch: int,
    total_epochs: int,
) -> None:
    if total_epochs <= 1:
        progress = 1.0
    else:
        progress = float(max(epoch - 1, 0) / max(total_epochs - 1, 1))
    lr_mult = _schedule_multiplier(progress, train_cfg)
    wd_mult = _weight_decay_multiplier(progress, train_cfg)
    for group in optimizer.param_groups:
        stage_mult = _stage_lr_multiplier(str(group.get("name", "")), epoch, train_cfg)
        group["lr"] = float(group.get("base_lr", group["lr"])) * lr_mult * stage_mult
        group["weight_decay"] = float(group.get("base_weight_decay", group["weight_decay"])) * wd_mult


def _build_proxy_event_splits(
    split: SplitBundle,
    *,
    seed: int,
    family_ratio: float,
    min_families: int = 1,
) -> tuple[SplitBundle, SplitBundle] | None:
    if family_ratio <= 0:
        return None
    positive_mask = split.label_main.astype(bool)
    positive_families = np.unique(split.family_id[positive_mask].astype(str))
    if positive_families.size < 2:
        return None
    holdout_count = max(min_families, int(round(float(positive_families.size) * family_ratio)))
    holdout_count = min(holdout_count, int(positive_families.size) - 1)
    if holdout_count <= 0:
        return None
    rng = np.random.default_rng(seed)
    holdout_families = set(rng.permutation(positive_families)[:holdout_count].tolist())
    eval_mask = np.isin(split.family_id.astype(str), list(holdout_families))
    train_mask = ~eval_mask
    if eval_mask.sum() == 0 or train_mask.sum() == 0:
        return None
    if split.label_main[eval_mask].sum() <= 0 or split.label_main[train_mask].sum() <= 0:
        return None
    proxy_train = _slice_split(split, train_mask, f"{split.name}_proxy_train")
    proxy_eval = _slice_split(split, eval_mask, f"{split.name}_proxy_event")
    return proxy_train, proxy_eval


def _build_proxy_event_split_bank(
    split: SplitBundle,
    *,
    seed: int,
    family_ratio: float,
    min_families: int = 1,
    num_samples: int = 1,
    seed_stride: int = 17,
) -> list[tuple[SplitBundle, SplitBundle]] | None:
    num_samples = max(int(num_samples), 1)
    if num_samples == 1:
        single = _build_proxy_event_splits(
            split,
            seed=seed,
            family_ratio=family_ratio,
            min_families=min_families,
        )
        return [single] if single is not None else None

    split_bank: list[tuple[SplitBundle, SplitBundle]] = []
    seen_family_keys: set[tuple[str, ...]] = set()
    max_attempts = max(num_samples * 4, num_samples)
    for index in range(max_attempts):
        current = _build_proxy_event_splits(
            split,
            seed=seed + index * seed_stride,
            family_ratio=family_ratio,
            min_families=min_families,
        )
        if current is None:
            continue
        _, proxy_eval = current
        family_key = tuple(sorted(np.unique(proxy_eval.family_id.astype(str)).tolist()))
        if family_key in seen_family_keys:
            continue
        seen_family_keys.add(family_key)
        split_bank.append(current)
        if len(split_bank) >= num_samples:
            break
    return split_bank or None


def _model_selection_score(metrics: dict[str, Any], train_cfg: dict[str, Any], prefix: str) -> float:
    mode = str(train_cfg.get("model_selection_mode", "legacy")).strip().lower()
    auprc = float(metrics.get("AUPRC", 0.0))
    auroc = float(metrics.get("AUROC", 0.5))
    best_f1 = float(metrics.get("BestF1", 0.0))
    recall_at_precision = float(metrics.get("Recall@P80", 0.0))
    af = float(metrics.get("Analog-Fidelity@5", 0.0))
    lead = float(metrics.get("LeadTime@P80", 0.0))
    brier = float(metrics.get("Brier", 0.0))
    ece = float(metrics.get("ECE@10", 0.0))
    logloss = float(metrics.get("LogLoss", 0.0))
    if mode == "legacy":
        return auprc + 0.01 * af
    if mode == "balanced_v2":
        af_bonus_weight = float(
            train_cfg.get(f"{prefix}_selection_af_bonus_weight", train_cfg.get("selection_af_bonus_weight", 0.02))
        )
        auroc_weight = float(
            train_cfg.get(f"{prefix}_selection_auroc_weight", train_cfg.get("selection_auroc_weight", 0.10))
        )
        best_f1_weight = float(
            train_cfg.get(f"{prefix}_selection_best_f1_weight", train_cfg.get("selection_best_f1_weight", 0.08))
        )
        recall_weight = float(
            train_cfg.get(f"{prefix}_selection_recall_weight", train_cfg.get("selection_recall_weight", 0.06))
        )
        lead_weight = float(
            train_cfg.get(f"{prefix}_selection_leadtime_weight", train_cfg.get("selection_leadtime_weight", 0.002))
        )
        brier_weight = float(
            train_cfg.get(f"{prefix}_selection_brier_weight", train_cfg.get("selection_brier_weight", 0.08))
        )
        ece_weight = float(
            train_cfg.get(f"{prefix}_selection_ece_weight", train_cfg.get("selection_ece_weight", 0.08))
        )
        logloss_weight = float(
            train_cfg.get(f"{prefix}_selection_logloss_weight", train_cfg.get("selection_logloss_weight", 0.03))
        )
        return (
            auprc * (1.0 + af_bonus_weight * af / 100.0)
            + auroc_weight * auroc
            + best_f1_weight * best_f1
            + recall_weight * recall_at_precision
            + lead_weight * lead
            - brier_weight * brier
            - ece_weight * ece
            - logloss_weight * logloss
        )
    af_bonus_weight = float(
        train_cfg.get(f"{prefix}_selection_af_bonus_weight", train_cfg.get("selection_af_bonus_weight", 0.05))
    )
    lead_weight = float(
        train_cfg.get(f"{prefix}_selection_leadtime_weight", train_cfg.get("selection_leadtime_weight", 0.0))
    )
    brier_weight = float(
        train_cfg.get(f"{prefix}_selection_brier_weight", train_cfg.get("selection_brier_weight", 0.0))
    )
    return auprc * (1.0 + af_bonus_weight * af / 100.0) + lead_weight * lead - brier_weight * brier


def _average_metric_reports(metric_reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not metric_reports:
        return {}
    averaged: dict[str, Any] = {}
    keys = metric_reports[0].keys()
    for key in keys:
        values = [report.get(key) for report in metric_reports if key in report]
        if not values:
            continue
        first = values[0]
        if isinstance(first, dict):
            nested: dict[str, Any] = {}
            for nested_key in first.keys():
                nested_values = [value.get(nested_key) for value in values if isinstance(value, dict) and nested_key in value]
                if not nested_values:
                    continue
                nested_first = nested_values[0]
                if isinstance(nested_first, (int, float)):
                    nested[nested_key] = float(sum(float(item) for item in nested_values) / len(nested_values))
                else:
                    nested[nested_key] = nested_first
            averaged[key] = nested
        elif isinstance(first, (int, float)):
            averaged[key] = float(sum(float(value) for value in values) / len(values))
        else:
            averaged[key] = first
    return averaged


def _split_statistics(split: SplitBundle) -> dict[str, Any]:
    labels = split.label_main.astype(bool)
    positive_families = np.unique(split.family_id[labels].astype(str))
    positive_incidents = np.unique(split.incident_id[labels].astype(str))
    prefix = split.prefix.astype(np.float32, copy=False)
    diff1 = np.diff(prefix, axis=1) if prefix.shape[1] > 1 else np.zeros_like(prefix)
    diff2 = np.diff(prefix, n=2, axis=1) if prefix.shape[1] > 2 else np.zeros_like(prefix[:, :1])
    flattened_std = np.std(prefix, axis=(1, 2)) if prefix.size > 0 else np.zeros((split.size,), dtype=np.float32)
    peak_ratio = (
        np.max(np.abs(prefix), axis=(1, 2)) / np.clip(flattened_std, 1e-6, None)
        if prefix.size > 0
        else np.zeros((split.size,), dtype=np.float32)
    )
    signature_std = (
        np.std(split.future_signature.astype(np.float32, copy=False), axis=1)
        if split.future_signature.size > 0
        else np.zeros((split.size,), dtype=np.float32)
    )
    return {
        "size": int(split.size),
        "positive_count": int(split.label_main.sum()),
        "positive_rate": float(split.label_main.mean()) if split.size > 0 else 0.0,
        "family_count": int(np.unique(split.family_id.astype(str)).size),
        "positive_family_count": int(positive_families.size),
        "positive_families": positive_families.tolist(),
        "incident_count": int(np.unique(split.incident_id.astype(str)).size),
        "positive_incident_count": int(positive_incidents.size),
        "feature_dim": int(split.feature_dim),
        "seq_len": int(split.seq_len),
        "prefix_scale": float(np.mean(np.std(prefix, axis=1))) if prefix.size > 0 else 0.0,
        "diff1_scale": float(np.mean(np.std(diff1, axis=1))) if diff1.size > 0 else 0.0,
        "diff2_abs_mean": float(np.mean(np.abs(diff2))) if diff2.size > 0 else 0.0,
        "peak_ratio": float(np.mean(peak_ratio)) if peak_ratio.size > 0 else 0.0,
        "signature_std": float(np.mean(signature_std)) if signature_std.size > 0 else 0.0,
    }


def _merge_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_overrides(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_policy_override(
    base: dict[str, Any],
    preset: dict[str, Any],
    *,
    replace: bool,
) -> dict[str, Any]:
    if replace:
        return deepcopy(preset)
    return _merge_overrides(base, preset)


def _auto_component_presets(objective: str) -> dict[str, dict[str, dict[str, Any]]]:
    robust_objective = objective in {"balanced", "robust", "event", "event_disjoint", "ood"}
    if robust_objective:
        dense_low_diversity_model = {
            "type": "campaign_mem_v3",
            "forecast_encoder": "dlinear",
            "retrieval_encoder": "transformer",
            "hidden_dim": 128,
            "embedding_dim": 128,
            "top_k": 5,
            "similarity_temperature": 0.15,
            "delta_scale": 0.22,
            "use_auxiliary": True,
            "use_contrastive": True,
            "use_hard_negatives": True,
            "use_utility": False,
        }
        dense_low_diversity_training = {
            "epochs": 18,
            "batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "auxiliary_weight": 0.25,
            "contrastive_weight": 0.2,
            "calibration_penalty_weight": 0.07,
            "grad_clip": 0.85,
            "retrieval_lr_scale": 0.95,
            "retrieval_head_lr_scale": 1.0,
            "forecast_lr_scale": 0.85,
            "forecast_head_lr_scale": 0.9,
            "gate_lr_scale": 0.95,
            "calibration_lr_scale": 0.95,
            "warmup_ratio": 0.05,
            "warmdown_ratio": 0.35,
            "final_lr_frac": 0.2,
            "final_weight_decay_frac": 0.2,
            "ema_decay": 0.99,
            "freeze_forecast_until_epoch": 4,
            "forecast_ramp_epochs": 2,
            "freeze_calibration_until_epoch": 7,
            "calibration_ramp_epochs": 3,
        }
    else:
        dense_low_diversity_model = {
            "type": "campaign_mem_abstain",
            "forecast_encoder": "dlinear",
            "retrieval_encoder": "transformer",
            "hidden_dim": 128,
            "embedding_dim": 128,
            "top_k": 5,
            "similarity_temperature": 0.15,
            "delta_scale": 0.18,
            "use_auxiliary": True,
            "use_contrastive": True,
            "use_hard_negatives": True,
            "use_utility": False,
        }
        dense_low_diversity_training = {
            "epochs": 18,
            "batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "auxiliary_weight": 0.25,
            "contrastive_weight": 0.2,
            "calibration_penalty_weight": 0.06,
            "selector_weight": 0.15,
            "selector_margin": 0.02,
            "abstention_weight": 0.10,
            "abstention_margin": 0.015,
            "grad_clip": 0.85,
            "retrieval_lr_scale": 0.95,
            "retrieval_head_lr_scale": 1.0,
            "forecast_lr_scale": 0.85,
            "forecast_head_lr_scale": 0.9,
            "gate_lr_scale": 0.95,
            "calibration_lr_scale": 0.95,
            "selector_lr_scale": 0.95,
            "warmup_ratio": 0.05,
            "warmdown_ratio": 0.35,
            "final_lr_frac": 0.2,
            "final_weight_decay_frac": 0.2,
            "ema_decay": 0.99,
            "freeze_forecast_until_epoch": 4,
            "forecast_ramp_epochs": 2,
            "freeze_calibration_until_epoch": 7,
            "calibration_ramp_epochs": 3,
            "model_selection_mode": "balanced",
            "selection_af_bonus_weight": 0.0,
            "checkpoint_average_top_k": 5,
        }

    scarce_diverse_model = {
        "type": "campaign_mem_dual_selector",
        "retrieval_encoder": "transformer",
        "hidden_dim": 128,
        "embedding_dim": 128,
        "top_k": 5,
        "similarity_temperature": 0.15,
        "delta_scale": 0.12,
        "use_auxiliary": not (objective in {"chronology", "chrono", "test"}),
        "use_contrastive": True,
        "use_hard_negatives": True,
        "use_utility": False,
    }
    scarce_diverse_training = {
        "epochs": 20,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
        "auxiliary_weight": 0.25 if scarce_diverse_model["use_auxiliary"] else 0.0,
        "contrastive_weight": 0.2,
        "calibration_penalty_weight": 0.06,
        "selector_weight": 0.10,
        "selector_margin": 0.02,
        "grad_clip": 0.85,
        "retrieval_lr_scale": 0.95,
        "retrieval_head_lr_scale": 1.0,
        "forecast_lr_scale": 0.9,
        "forecast_head_lr_scale": 0.95,
        "gate_lr_scale": 0.95,
        "calibration_lr_scale": 0.95,
        "selector_lr_scale": 0.95,
        "remaining_lr_scale": 0.95,
        "warmup_ratio": 0.05,
        "warmdown_ratio": 0.35,
        "final_lr_frac": 0.2,
        "final_weight_decay_frac": 0.2,
        "ema_decay": 0.99,
        "freeze_forecast_until_epoch": 2,
        "forecast_ramp_epochs": 2,
        "freeze_calibration_until_epoch": 5,
        "calibration_ramp_epochs": 3,
        "model_selection_mode": "balanced",
        "selection_af_bonus_weight": 0.03,
        "proxy_event_selection_af_bonus_weight": 0.04,
        "model_selection_dev_weight": 1.0,
        "model_selection_proxy_event_weight": 1.0,
        "proxy_event_family_ratio": 0.25,
        "checkpoint_average_top_k": 5,
    }
    if not scarce_diverse_model["use_auxiliary"]:
        scarce_diverse_training["selection_af_bonus_weight"] = 0.03

    extreme_sparse_model = {
        "type": "campaign_mem_v3",
        "forecast_encoder": "dlinear",
        "retrieval_encoder": "transformer",
        "hidden_dim": 128,
        "embedding_dim": 128,
        "top_k": 5,
        "similarity_temperature": 0.15,
        "delta_scale": 0.04,
        "use_auxiliary": True,
        "use_contrastive": True,
        "use_hard_negatives": True,
        "use_utility": False,
    }
    extreme_sparse_training = {
        "epochs": 18,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
        "auxiliary_weight": 0.25,
        "contrastive_weight": 0.2,
        "calibration_penalty_weight": 0.14,
        "grad_clip": 0.85,
        "retrieval_lr_scale": 1.0,
        "retrieval_head_lr_scale": 1.0,
        "forecast_lr_scale": 0.55,
        "forecast_head_lr_scale": 0.6,
        "gate_lr_scale": 0.95,
        "calibration_lr_scale": 0.55,
        "warmup_ratio": 0.05,
        "warmdown_ratio": 0.35,
        "final_lr_frac": 0.2,
        "final_weight_decay_frac": 0.2,
        "ema_decay": 0.99,
        "freeze_forecast_until_epoch": 6,
        "forecast_ramp_epochs": 3,
        "freeze_calibration_until_epoch": 10,
        "calibration_ramp_epochs": 4,
        "model_selection_mode": "balanced",
        "selection_af_bonus_weight": 0.0,
    }
    return {
        "extreme_sparse": {"model": extreme_sparse_model, "training": extreme_sparse_training},
        "dense_low_diversity": {"model": dense_low_diversity_model, "training": dense_low_diversity_training},
        "scarce_diverse": {"model": scarce_diverse_model, "training": scarce_diverse_training},
    }


def _tracer_auto_regime(train_stats: dict[str, Any], dev_stats: dict[str, Any]) -> str:
    if int(dev_stats.get("positive_count", 0)) <= 0 and float(train_stats.get("positive_rate", 0.0)) <= 0.02:
        return "cold_start_sparse"
    if int(dev_stats.get("positive_count", 0)) <= 0 or float(train_stats.get("positive_rate", 0.0)) <= 0.005:
        return "extreme_sparse"
    if (
        float(train_stats.get("positive_rate", 0.0)) >= 0.35
        and float(train_stats.get("peak_ratio", 0.0)) <= 3.2
        and float(train_stats.get("diff2_abs_mean", 0.0)) <= 0.26
    ):
        return "simple_dense"
    if (
        float(train_stats.get("positive_rate", 0.0)) >= 0.05
        and int(train_stats.get("family_count", 0)) <= 6
    ):
        return "dense_low_diversity"
    return "sparse_diverse"


def _tracer_auto_v2_regime(
    train_stats: dict[str, Any],
    dev_stats: dict[str, Any],
    objective: str,
) -> str:
    chronology_objective = objective in {"chronology", "chrono", "test"}
    event_objective = objective in {"event", "event_disjoint", "ood"}
    if int(dev_stats.get("positive_count", 0)) <= 0 and float(train_stats.get("positive_rate", 0.0)) <= 0.02:
        return "cold_start_sparse"
    if float(train_stats.get("positive_rate", 0.0)) <= 0.005:
        return "extreme_sparse"
    if (
        float(train_stats.get("positive_rate", 0.0)) >= 0.35
        and float(train_stats.get("peak_ratio", 0.0)) <= 3.2
        and float(train_stats.get("diff2_abs_mean", 0.0)) <= 0.26
    ):
        return "simple_dense"
    if event_objective and float(train_stats.get("positive_rate", 0.0)) >= 0.05 and int(train_stats.get("family_count", 0)) <= 6:
        return "dense_low_diversity_event"
    if (
        chronology_objective
        and int(train_stats.get("positive_family_count", 0)) <= 2
        and float(train_stats.get("diff2_abs_mean", 0.0)) >= 0.28
        and float(train_stats.get("peak_ratio", 0.0)) >= 4.0
    ):
        return "sparse_diverse_chrono_spiky"
    if (
        float(train_stats.get("positive_rate", 0.0)) >= 0.05
        and int(train_stats.get("family_count", 0)) <= 6
    ):
        return "dense_low_diversity"
    return "sparse_diverse"


def _tracer_auto_component_presets(objective: str) -> dict[str, dict[str, dict[str, Any]]]:
    chronology_objective = objective in {"chronology", "chrono", "test"}
    event_objective = objective in {"event", "event_disjoint", "ood"}

    presets = deepcopy(_auto_component_presets(objective))
    for preset in presets.values():
        training = preset["training"]
        training["model_selection_mode"] = "balanced_v2"
        training.setdefault("selection_af_bonus_weight", 0.02)
        training.setdefault("selection_auroc_weight", 0.10)
        training.setdefault("selection_best_f1_weight", 0.08)
        training.setdefault("selection_recall_weight", 0.06)
        training.setdefault("selection_ece_weight", 0.08)
        training.setdefault("selection_logloss_weight", 0.03)
        training.setdefault("selection_brier_weight", 0.08)
        training.setdefault("checkpoint_average_top_k", 5)
    presets["extreme_sparse"]["training"]["model_selection_mode"] = "balanced"
    presets["extreme_sparse"]["training"]["selection_af_bonus_weight"] = 0.0
    presets["extreme_sparse"]["training"]["checkpoint_average_top_k"] = 1

    cold_start_sparse_model = {
        "type": "lstm",
        "encoder": "lstm",
        "hidden_dim": 128,
        "embedding_dim": 128,
        "use_auxiliary": True,
    }
    cold_start_sparse_training = {
        "epochs": 12,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "auxiliary_weight": 0.25,
        "model_selection_mode": "legacy",
        "checkpoint_average_top_k": 1,
    }

    dense_low_diversity_event_model = {
        "type": "dlinear",
        "encoder": "dlinear",
        "hidden_dim": 128,
        "embedding_dim": 128,
        "use_auxiliary": True,
    }
    dense_low_diversity_event_training = {
        "epochs": 12,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "auxiliary_weight": 0.25,
        "model_selection_mode": "legacy",
        "checkpoint_average_top_k": 1,
    }

    sparse_diverse_chrono_spiky_model = {
        "type": "tcn",
        "encoder": "tcn",
        "hidden_dim": 96,
        "embedding_dim": 96,
        "use_auxiliary": True,
    }
    sparse_diverse_chrono_spiky_training = {
        "epochs": 12,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "auxiliary_weight": 0.25,
        "model_selection_mode": "legacy",
        "checkpoint_average_top_k": 1,
    }

    simple_dense_model = {
        "type": "tail_risk_linear",
        "hidden_dim": 64,
        "embedding_dim": 64,
        "use_auxiliary": True,
    }
    simple_dense_training = {
        "epochs": 6,
        "batch_size": 64,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
        "auxiliary_weight": 0.25,
        "grad_clip": 1.0,
        "model_selection_mode": "legacy",
        "checkpoint_average_top_k": 1,
    }

    if event_objective:
        sparse_diverse_model = {
            "type": "campaign_mem_decomp_modular",
            "retrieval_encoder": "transformer",
            "stable_encoder": "dlinear",
            "shock_encoder": "patchtst",
            "hidden_dim": 128,
            "embedding_dim": 128,
            "top_k": 5,
            "similarity_temperature": 0.15,
            "delta_scale": 0.18,
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
        }
        sparse_diverse_training = {
            "epochs": 20,
            "batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "auxiliary_weight": 0.25,
            "contrastive_weight": 0.2,
            "calibration_penalty_weight": 0.05,
            "selector_weight": 0.10,
            "selector_margin": 0.02,
            "abstention_weight": 0.06,
            "abstention_margin": 0.015,
            "shift_weight": 0.06,
            "shift_target_quantile": 0.75,
            "aggressive_weight": 0.04,
            "main_pos_weight": "auto_sqrt",
            "aux_pos_weight": "auto_sqrt",
            "grad_clip": 0.85,
            "retrieval_lr_scale": 0.95,
            "retrieval_head_lr_scale": 1.0,
            "forecast_lr_scale": 0.9,
            "forecast_head_lr_scale": 0.95,
            "gate_lr_scale": 0.95,
            "calibration_lr_scale": 0.95,
            "selector_lr_scale": 0.95,
            "remaining_lr_scale": 0.95,
            "warmup_ratio": 0.05,
            "warmdown_ratio": 0.35,
            "final_lr_frac": 0.2,
            "final_weight_decay_frac": 0.2,
            "ema_decay": 0.99,
            "freeze_forecast_until_epoch": 2,
            "forecast_ramp_epochs": 2,
            "freeze_calibration_until_epoch": 5,
            "calibration_ramp_epochs": 3,
            "model_selection_mode": "balanced",
            "selection_af_bonus_weight": 0.02,
            "selection_auroc_weight": 0.08,
            "selection_best_f1_weight": 0.10,
            "selection_recall_weight": 0.08,
            "selection_ece_weight": 0.08,
            "selection_logloss_weight": 0.03,
            "selection_brier_weight": 0.08,
            "proxy_event_selection_af_bonus_weight": 0.03,
            "model_selection_dev_weight": 0.85,
            "model_selection_proxy_event_weight": 1.5,
            "model_selection_start_epoch": 8,
            "proxy_event_family_ratio": 0.25,
            "checkpoint_average_top_k": 3,
        }
    elif chronology_objective:
        sparse_diverse_model = {
            "type": "campaign_mem_dual_selector",
            "retrieval_encoder": "transformer",
            "hidden_dim": 128,
            "embedding_dim": 128,
            "top_k": 5,
            "similarity_temperature": 0.15,
            "delta_scale": 0.12,
            "use_auxiliary": False,
            "use_contrastive": True,
            "use_hard_negatives": True,
            "use_utility": False,
        }
        sparse_diverse_training = {
            "epochs": 20,
            "batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "auxiliary_weight": 0.0,
            "contrastive_weight": 0.2,
            "calibration_penalty_weight": 0.06,
            "selector_weight": 0.10,
            "selector_margin": 0.02,
            "grad_clip": 0.85,
            "retrieval_lr_scale": 0.95,
            "retrieval_head_lr_scale": 1.0,
            "forecast_lr_scale": 0.9,
            "forecast_head_lr_scale": 0.95,
            "gate_lr_scale": 0.95,
            "calibration_lr_scale": 0.95,
            "selector_lr_scale": 0.95,
            "remaining_lr_scale": 0.95,
            "warmup_ratio": 0.05,
            "warmdown_ratio": 0.35,
            "final_lr_frac": 0.2,
            "final_weight_decay_frac": 0.2,
            "ema_decay": 0.99,
            "freeze_forecast_until_epoch": 2,
            "forecast_ramp_epochs": 2,
            "freeze_calibration_until_epoch": 5,
            "calibration_ramp_epochs": 3,
            "model_selection_mode": "balanced_v2",
            "selection_af_bonus_weight": 0.02,
            "selection_auroc_weight": 0.08,
            "selection_best_f1_weight": 0.08,
            "selection_recall_weight": 0.10,
            "selection_ece_weight": 0.08,
            "selection_logloss_weight": 0.03,
            "selection_brier_weight": 0.08,
            "proxy_event_selection_af_bonus_weight": 0.03,
            "model_selection_dev_weight": 0.85,
            "model_selection_proxy_event_weight": 1.5,
            "model_selection_start_epoch": 6,
            "proxy_event_family_ratio": 0.25,
            "checkpoint_average_top_k": 5,
        }
    else:
        sparse_diverse_model = {
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
        }
        sparse_diverse_training = {
            "epochs": 20,
            "batch_size": 32,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "auxiliary_weight": 0.25,
            "contrastive_weight": 0.2,
            "calibration_penalty_weight": 0.06,
            "selector_weight": 0.10,
            "selector_margin": 0.02,
            "abstention_weight": 0.06,
            "abstention_margin": 0.015,
            "shift_weight": 0.06,
            "shift_target_quantile": 0.75,
            "aggressive_weight": 0.04,
            "grad_clip": 0.85,
            "retrieval_lr_scale": 0.95,
            "retrieval_head_lr_scale": 1.0,
            "forecast_lr_scale": 0.9,
            "forecast_head_lr_scale": 0.95,
            "gate_lr_scale": 0.95,
            "calibration_lr_scale": 0.95,
            "selector_lr_scale": 0.95,
            "remaining_lr_scale": 0.95,
            "warmup_ratio": 0.05,
            "warmdown_ratio": 0.35,
            "final_lr_frac": 0.2,
            "final_weight_decay_frac": 0.2,
            "ema_decay": 0.99,
            "freeze_forecast_until_epoch": 2,
            "forecast_ramp_epochs": 2,
            "freeze_calibration_until_epoch": 5,
            "calibration_ramp_epochs": 3,
            "model_selection_mode": "balanced_v2",
            "selection_af_bonus_weight": 0.02,
            "selection_auroc_weight": 0.08,
            "selection_best_f1_weight": 0.10,
            "selection_recall_weight": 0.08,
            "selection_ece_weight": 0.08,
            "selection_logloss_weight": 0.03,
            "selection_brier_weight": 0.08,
            "proxy_event_selection_af_bonus_weight": 0.03,
            "model_selection_dev_weight": 0.85,
            "model_selection_proxy_event_weight": 1.5,
            "model_selection_start_epoch": 8,
            "proxy_event_family_ratio": 0.25,
            "checkpoint_average_top_k": 3,
        }

    presets["cold_start_sparse"] = {
        "model": cold_start_sparse_model,
        "training": cold_start_sparse_training,
    }
    presets["dense_low_diversity_event"] = {
        "model": dense_low_diversity_event_model,
        "training": dense_low_diversity_event_training,
    }
    presets["simple_dense"] = {"model": simple_dense_model, "training": simple_dense_training}
    presets["sparse_diverse_chrono_spiky"] = {
        "model": sparse_diverse_chrono_spiky_model,
        "training": sparse_diverse_chrono_spiky_training,
    }
    presets["sparse_diverse"] = {"model": sparse_diverse_model, "training": sparse_diverse_training}
    return presets


def _tracer_adaptive_presets(objective: str) -> dict[str, dict[str, dict[str, Any] | bool]]:
    balanced_family = deepcopy(_auto_component_presets("balanced"))
    tracer_family = deepcopy(_tracer_auto_component_presets(objective))

    presets: dict[str, dict[str, dict[str, Any] | bool]] = {
        "extreme_sparse": {
            "model": balanced_family["extreme_sparse"]["model"],
            "training": balanced_family["extreme_sparse"]["training"],
            "replace_model": True,
            "replace_training": True,
        },
        "dense_low_diversity": {
            "model": balanced_family["dense_low_diversity"]["model"],
            "training": balanced_family["dense_low_diversity"]["training"],
            "replace_model": True,
            "replace_training": True,
        },
        "cold_start_sparse": {
            "model": tracer_family["cold_start_sparse"]["model"],
            "training": tracer_family["cold_start_sparse"]["training"],
            "replace_model": True,
            "replace_training": True,
        },
        "dense_low_diversity_event": {
            "model": tracer_family["dense_low_diversity_event"]["model"],
            "training": tracer_family["dense_low_diversity_event"]["training"],
            "replace_model": True,
            "replace_training": True,
        },
        "simple_dense": {
            "model": tracer_family["simple_dense"]["model"],
            "training": tracer_family["simple_dense"]["training"],
            "replace_model": True,
            "replace_training": True,
        },
        "sparse_diverse_chrono_spiky": {
            "model": tracer_family["sparse_diverse_chrono_spiky"]["model"],
            "training": tracer_family["sparse_diverse_chrono_spiky"]["training"],
            "replace_model": True,
            "replace_training": True,
        },
        "sparse_diverse": {
            "model": tracer_family["sparse_diverse"]["model"],
            "training": tracer_family["sparse_diverse"]["training"],
            "replace_model": True,
            "replace_training": True,
        },
    }
    return presets


def _resolve_auto_component_policy(
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    train_split: SplitBundle,
    dev_split: SplitBundle,
    policy_cfg: dict[str, Any] | str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if isinstance(policy_cfg, str):
        policy_name = policy_cfg
        objective = "balanced"
    else:
        policy_name = str(policy_cfg.get("name", "public_modular_v1"))
        objective = str(policy_cfg.get("objective", "balanced")).strip().lower()
    if policy_name not in {"public_modular_v1", "tracer_auto", "tracer_auto_v2", "tracer_adaptive"}:
        raise ValueError(f"Unsupported auto component policy: {policy_name}")

    train_stats = _split_statistics(train_split)
    dev_stats = _split_statistics(dev_split)

    if policy_name == "public_modular_v1":
        if train_stats["positive_rate"] <= 0.005:
            regime = "extreme_sparse"
        elif train_stats["positive_family_count"] <= 4 and train_stats["positive_rate"] >= 0.05:
            regime = "dense_low_diversity"
        else:
            regime = "scarce_diverse"
        presets = _auto_component_presets(objective)
    elif policy_name == "tracer_auto":
        regime = _tracer_auto_regime(train_stats, dev_stats)
        presets = _tracer_auto_component_presets(objective)
    elif policy_name == "tracer_adaptive":
        regime = _tracer_auto_v2_regime(train_stats, dev_stats, objective)
        presets = _tracer_adaptive_presets(objective)
    else:
        regime = _tracer_auto_v2_regime(train_stats, dev_stats, objective)
        presets = _tracer_auto_component_presets(objective)
    preset = presets[regime]
    replace_model = bool(preset.get("replace_model", False))
    replace_training = bool(preset.get("replace_training", False))
    resolved_model_cfg = _apply_policy_override(model_cfg, preset["model"], replace=replace_model)
    resolved_train_cfg = _apply_policy_override(train_cfg, preset["training"], replace=replace_training)

    policy_info = {
        "name": policy_name,
        "objective": objective,
        "regime": regime,
        "train_stats": train_stats,
        "dev_stats": dev_stats,
        "resolved_model_type": str(resolved_model_cfg.get("type")),
        "resolved_use_auxiliary": bool(resolved_model_cfg.get("use_auxiliary", True)),
        "resolved_use_hard_negatives": bool(resolved_model_cfg.get("use_hard_negatives", False)),
    }
    return resolved_model_cfg, resolved_train_cfg, policy_info


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    teacher_models: list[nn.Module] | None = None,
    loss_cfg: dict[str, float] | None = None,
    ema_state: dict[str, torch.Tensor] | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    optimized_examples = 0
    loss_cfg = loss_cfg or {}
    main_pos_weight = float(loss_cfg.get("main_pos_weight", 1.0))
    aux_pos_weight = float(loss_cfg.get("aux_pos_weight", 1.0))
    focal_gamma = float(loss_cfg.get("focal_gamma", 0.0))
    ema_decay = float(train_cfg.get("ema_decay", 0.0))
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, RetrievalForecaster):
            if int(batch["prefix"].shape[0]) < 2:
                continue
            outputs = model.forward_with_batch_memory(
                prefix=batch["prefix"],
                label_main=batch["label_main"],
                label_aux=batch["label_aux"],
            )
        else:
            outputs = model(prefix=batch["prefix"])

        loss_main = _binary_classification_loss(
            outputs["final_main_logit"],
            batch["label_main"],
            pos_weight=main_pos_weight,
            focal_gamma=focal_gamma,
        )
        aux_weight = float(train_cfg.get("auxiliary_weight", 0.25)) if model_cfg.get("use_auxiliary", True) else 0.0
        loss_aux = (
            _binary_classification_loss(
                outputs["final_aux_logit"],
                batch["label_aux"],
                pos_weight=aux_pos_weight,
                focal_gamma=focal_gamma,
            )
            if aux_weight > 0
            else outputs["final_main_logit"].new_tensor(0.0)
        )

        contrastive_weight = float(train_cfg.get("contrastive_weight", 0.2)) if model_cfg.get("use_contrastive", False) else 0.0
        contrastive_loss = (
            pairwise_future_contrastive_loss(
                embedding=outputs["embedding"],
                future_signature=batch["future_signature"],
                prefix=batch["prefix"],
                use_hard_negatives=bool(model_cfg.get("use_hard_negatives", False)),
            )
            if contrastive_weight > 0
            else outputs["final_main_logit"].new_tensor(0.0)
        )

        utility_weight = float(train_cfg.get("utility_weight", 0.05)) if model_cfg.get("use_utility", False) else 0.0
        utility_loss = (
            retrieval_utility_loss(
                retrieved_indices=outputs["retrieved_indices"],
                future_signature=batch["future_signature"],
            )
            if utility_weight > 0 and "retrieved_indices" in outputs
            else outputs["final_main_logit"].new_tensor(0.0)
        )

        calibration_penalty_weight = float(train_cfg.get("calibration_penalty_weight", 0.0))
        calibration_penalty = (
            outputs["calibration_penalty"]
            if calibration_penalty_weight > 0 and "calibration_penalty" in outputs
            else outputs["final_main_logit"].new_tensor(0.0)
        )

        selector_weight = float(train_cfg.get("selector_weight", 0.0))
        selector_margin = float(train_cfg.get("selector_margin", 0.0))
        selector_loss = outputs["final_main_logit"].new_tensor(0.0)
        selector_main_target = outputs["final_main_logit"].new_zeros(outputs["final_main_logit"].shape)
        selector_aux_target = outputs["final_aux_logit"].new_zeros(outputs["final_aux_logit"].shape)
        if selector_weight > 0 and "selector_main_logit" in outputs and "base_main_logit" in outputs:
            with torch.no_grad():
                base_main_loss = F.binary_cross_entropy_with_logits(
                    outputs["base_main_logit"].detach(),
                    batch["label_main"],
                    reduction="none",
                )
                forecast_main_loss = F.binary_cross_entropy_with_logits(
                    outputs["forecast_main_logit"].detach(),
                    batch["label_main"],
                    reduction="none",
                )
                selector_main_target = (forecast_main_loss + selector_margin < base_main_loss).float()

                base_aux_loss = F.binary_cross_entropy_with_logits(
                    outputs["base_aux_logit"].detach(),
                    batch["label_aux"],
                    reduction="none",
                )
                forecast_aux_loss = F.binary_cross_entropy_with_logits(
                    outputs["forecast_aux_logit"].detach(),
                    batch["label_aux"],
                    reduction="none",
                )
                selector_aux_target = (forecast_aux_loss + selector_margin < base_aux_loss).float()
            selector_loss = (
                F.binary_cross_entropy_with_logits(outputs["selector_main_logit"], selector_main_target)
                + F.binary_cross_entropy_with_logits(outputs["selector_aux_logit"], selector_aux_target)
            )

        abstention_weight = float(train_cfg.get("abstention_weight", 0.0))
        abstention_margin = float(train_cfg.get("abstention_margin", selector_margin))
        abstention_loss = outputs["final_main_logit"].new_tensor(0.0)
        if abstention_weight > 0 and "abstain_main_logit" in outputs and "base_main_logit" in outputs:
            with torch.no_grad():
                base_main_loss = F.binary_cross_entropy_with_logits(
                    outputs["base_main_logit"].detach(),
                    batch["label_main"],
                    reduction="none",
                )
                forecast_main_loss = F.binary_cross_entropy_with_logits(
                    outputs["forecast_main_logit"].detach(),
                    batch["label_main"],
                    reduction="none",
                )
                abstain_main_target = (forecast_main_loss + abstention_margin >= base_main_loss).float()

                base_aux_loss = F.binary_cross_entropy_with_logits(
                    outputs["base_aux_logit"].detach(),
                    batch["label_aux"],
                    reduction="none",
                )
                forecast_aux_loss = F.binary_cross_entropy_with_logits(
                    outputs["forecast_aux_logit"].detach(),
                    batch["label_aux"],
                    reduction="none",
                )
                abstain_aux_target = (forecast_aux_loss + abstention_margin >= base_aux_loss).float()
            abstention_loss = (
                F.binary_cross_entropy_with_logits(outputs["abstain_main_logit"], abstain_main_target)
                + F.binary_cross_entropy_with_logits(outputs["abstain_aux_logit"], abstain_aux_target)
            )

        shift_weight = float(train_cfg.get("shift_weight", 0.0))
        shift_quantile = float(train_cfg.get("shift_target_quantile", 0.75))
        shift_loss = outputs["final_main_logit"].new_tensor(0.0)
        shift_main_target = outputs["final_main_logit"].new_zeros(outputs["final_main_logit"].shape)
        shift_aux_target = outputs["final_aux_logit"].new_zeros(outputs["final_aux_logit"].shape)
        if shift_weight > 0 and "shift_main_logit" in outputs and "prefix_scale" in outputs:
            with torch.no_grad():
                main_disagreement = torch.abs(
                    torch.sigmoid(outputs["forecast_main_logit"].detach())
                    - torch.sigmoid(outputs["base_main_logit"].detach())
                )
                aux_disagreement = torch.abs(
                    torch.sigmoid(outputs["forecast_aux_logit"].detach())
                    - torch.sigmoid(outputs["base_aux_logit"].detach())
                )
                prefix_scale = outputs["prefix_scale"].detach()
                prefix_peak = outputs["prefix_peak"].detach()
                score_std = outputs["retrieval_score_std"].detach()
                main_disp = outputs["retrieval_main_dispersion"].detach()
                aux_disp = outputs["retrieval_aux_dispersion"].detach()

                shift_main_target = (
                    (prefix_scale >= torch.quantile(prefix_scale, shift_quantile))
                    | (prefix_peak >= torch.quantile(prefix_peak, shift_quantile))
                    | (score_std >= torch.quantile(score_std, shift_quantile))
                    | (main_disp >= torch.quantile(main_disp, shift_quantile))
                    | (main_disagreement >= torch.quantile(main_disagreement, shift_quantile))
                ).float()
                shift_aux_target = (
                    (prefix_scale >= torch.quantile(prefix_scale, shift_quantile))
                    | (prefix_peak >= torch.quantile(prefix_peak, shift_quantile))
                    | (score_std >= torch.quantile(score_std, shift_quantile))
                    | (aux_disp >= torch.quantile(aux_disp, shift_quantile))
                    | (aux_disagreement >= torch.quantile(aux_disagreement, shift_quantile))
                ).float()
            shift_loss = (
                F.binary_cross_entropy_with_logits(outputs["shift_main_logit"], shift_main_target)
                + F.binary_cross_entropy_with_logits(outputs["shift_aux_logit"], shift_aux_target)
            )

        aggressive_weight = float(train_cfg.get("aggressive_weight", 0.0))
        aggressive_loss = outputs["final_main_logit"].new_tensor(0.0)
        if aggressive_weight > 0 and "aggressive_main_logit" in outputs:
            with torch.no_grad():
                aggressive_main_target = torch.maximum(selector_main_target, shift_main_target)
                aggressive_aux_target = torch.maximum(selector_aux_target, shift_aux_target)
            aggressive_loss = (
                F.binary_cross_entropy_with_logits(outputs["aggressive_main_logit"], aggressive_main_target)
                + F.binary_cross_entropy_with_logits(outputs["aggressive_aux_logit"], aggressive_aux_target)
            )

        teacher_weight = float(train_cfg.get("teacher_weight", 0.0))
        teacher_loss = outputs["final_main_logit"].new_tensor(0.0)
        if teacher_weight > 0 and teacher_models:
            teacher_main_probs = []
            teacher_aux_probs = []
            with torch.no_grad():
                for teacher_model in teacher_models:
                    teacher_outputs = teacher_model(prefix=batch["prefix"])
                    teacher_main_probs.append(torch.sigmoid(teacher_outputs["final_main_logit"]))
                    teacher_aux_probs.append(torch.sigmoid(teacher_outputs["final_aux_logit"]))
            teacher_main_prob = torch.stack(teacher_main_probs, dim=0).mean(dim=0).clamp(1e-4, 1 - 1e-4)
            teacher_aux_prob = torch.stack(teacher_aux_probs, dim=0).mean(dim=0).clamp(1e-4, 1 - 1e-4)
            student_main_logit = outputs["forecast_main_logit"] if "forecast_main_logit" in outputs else outputs["final_main_logit"]
            student_aux_logit = outputs["forecast_aux_logit"] if "forecast_aux_logit" in outputs else outputs["final_aux_logit"]
            teacher_loss = F.binary_cross_entropy_with_logits(student_main_logit, teacher_main_prob) + F.binary_cross_entropy_with_logits(student_aux_logit, teacher_aux_prob)

        loss = (
            loss_main
            + aux_weight * loss_aux
            + contrastive_weight * contrastive_loss
            + utility_weight * utility_loss
            + calibration_penalty_weight * calibration_penalty
            + selector_weight * selector_loss
            + abstention_weight * abstention_loss
            + shift_weight * shift_loss
            + aggressive_weight * aggressive_loss
            + teacher_weight * teacher_loss
        )
        if not torch.isfinite(loss):
            raise ValueError("Encountered non-finite training loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg.get("grad_clip", 1.0)))
        optimizer.step()
        if ema_state is not None and ema_decay > 0:
            _update_ema_state(model, ema_state, ema_decay)
        batch_examples = int(batch["prefix"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_examples
        optimized_examples += batch_examples
    return total_loss / max(optimized_examples, 1)


def _encode_split(model: ParametricForecaster, split: SplitBundle, device: torch.device, batch_size: int) -> torch.Tensor:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            prefix = batch["prefix"].to(device)
            if hasattr(model, "encode_memory"):
                embedding = getattr(model, "encode_memory")(prefix)
            else:
                outputs = model(prefix=prefix)
                embedding = outputs["embedding"]
            embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)


def _evaluate_parametric(
    model: ParametricForecaster,
    split: SplitBundle,
    device: torch.device,
    batch_size: int,
    metric_cfg: dict[str, Any],
) -> dict[str, Any]:
    loader = _make_loader(split, batch_size=batch_size, shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(prefix=batch["prefix"].to(device))
            scores.append(torch.sigmoid(outputs["final_main_logit"]).detach().cpu())
    y_score = torch.cat(scores).numpy()
    return build_metric_report(
        y_true=split.label_main.astype(np.int64),
        y_score=y_score,
        time_to_escalation=split.time_to_escalation,
        target_precision=float(metric_cfg.get("target_precision", 0.8)),
    )


def _evaluate_retrieval(
    model: RetrievalForecaster,
    train_split: SplitBundle,
    eval_split: SplitBundle,
    device: torch.device,
    batch_size: int,
    metric_cfg: dict[str, Any],
) -> dict[str, Any]:
    train_embedding = _encode_split(model, train_split, device=device, batch_size=batch_size).to(device)
    loader = _make_loader(eval_split, batch_size=batch_size, shuffle=False)
    scores = []
    retrieved_indices = []
    memory_main = torch.from_numpy(train_split.label_main).float().to(device)
    memory_aux = torch.from_numpy(train_split.label_aux).float().to(device)
    with torch.no_grad():
        for batch in loader:
            outputs = model.forward_with_external_memory(
                prefix=batch["prefix"].to(device),
                memory_embedding=train_embedding,
                memory_main_label=memory_main,
                memory_aux_label=memory_aux,
            )
            scores.append(torch.sigmoid(outputs["final_main_logit"]).detach().cpu())
            retrieved_indices.append(outputs["retrieved_indices"].detach().cpu())
    y_score = torch.cat(scores).numpy()
    indices = torch.cat(retrieved_indices).numpy()
    analog_threshold = float(
        metric_cfg.get(
            "analog_fidelity_distance_threshold",
            eval_split.metadata.get("analog_fidelity_distance_threshold", 0.45),
        )
    )
    return build_metric_report(
        y_true=eval_split.label_main.astype(np.int64),
        y_score=y_score,
        time_to_escalation=eval_split.time_to_escalation,
        target_precision=float(metric_cfg.get("target_precision", 0.8)),
        query_future=eval_split.future_signature,
        memory_future=train_split.future_signature,
        retrieved_indices=indices,
        analog_threshold=analog_threshold,
        memory_tte=train_split.time_to_escalation,
    )


def _evaluate_knn(
    train_split: SplitBundle,
    eval_split: SplitBundle,
    top_k: int,
    metric_cfg: dict[str, Any],
) -> dict[str, Any]:
    train_features = train_split.summary_features()
    eval_features = eval_split.summary_features()
    train_norm = train_features / np.linalg.norm(train_features, axis=1, keepdims=True).clip(min=1e-6)
    eval_norm = eval_features / np.linalg.norm(eval_features, axis=1, keepdims=True).clip(min=1e-6)
    similarity = eval_norm @ train_norm.T
    indices = np.argsort(-similarity, axis=1)[:, :top_k]
    scores = train_split.label_main[indices].mean(axis=1)
    return build_metric_report(
        y_true=eval_split.label_main.astype(np.int64),
        y_score=scores,
        time_to_escalation=eval_split.time_to_escalation,
        target_precision=float(metric_cfg.get("target_precision", 0.8)),
        query_future=eval_split.future_signature,
        memory_future=train_split.future_signature,
        retrieved_indices=indices,
        analog_threshold=float(
            metric_cfg.get(
                "analog_fidelity_distance_threshold",
                eval_split.metadata.get("analog_fidelity_distance_threshold", 0.45),
            )
        ),
        memory_tte=train_split.time_to_escalation,
    )


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 7)))
    data_cfg = config["data"]
    model_cfg = deepcopy(config["model"])
    train_cfg = deepcopy(config.get("training", {}))
    metric_cfg = config.get("metrics", {})
    output_cfg = config.get("output", {})

    train_split = load_split(data_cfg["dataset_dir"], data_cfg.get("train_split", "train"))
    dev_split = load_split(data_cfg["dataset_dir"], data_cfg.get("dev_split", "dev"))
    test_split = load_split(data_cfg["dataset_dir"], data_cfg.get("test_split", "test"))
    event_split = None
    event_split_name = data_cfg.get("event_disjoint_split", "test_event_disjoint")
    if (Path(data_cfg["dataset_dir"]) / f"{event_split_name}.npz").exists():
        event_split = load_split(data_cfg["dataset_dir"], event_split_name)

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

    result: dict[str, Any] = {
        "experiment_name": config["experiment_name"],
        "dataset_dir": data_cfg["dataset_dir"],
        "model": deepcopy(model_cfg),
    }
    if policy_info is not None:
        result["auto_component_policy"] = policy_info
        result["requested_model"] = requested_model_cfg
        result["requested_training"] = requested_train_cfg

    if model_cfg["type"] == "pure_knn":
        result["dev"] = _evaluate_knn(train_split, dev_split, top_k=int(model_cfg.get("top_k", 5)), metric_cfg=metric_cfg)
        result["test"] = _evaluate_knn(train_split, test_split, top_k=int(model_cfg.get("top_k", 5)), metric_cfg=metric_cfg)
        if event_split is not None:
            result["test_event_disjoint"] = _evaluate_knn(train_split, event_split, top_k=int(model_cfg.get("top_k", 5)), metric_cfg=metric_cfg)
    else:
        device = _device_from_config(config)
        batch_size = int(train_cfg.get("batch_size", 64))
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
        teacher_models: list[nn.Module] = []
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
                    teacher_score = teacher_dev_metrics["AUPRC"]
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
        history = []
        total_epochs = int(train_cfg.get("epochs", 6))
        for epoch in range(1, total_epochs + 1):
            _update_optimizer_schedule(optimizer, train_cfg, epoch=epoch, total_epochs=total_epochs)
            train_loss = _train_epoch(
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
            proxy_event_metrics = None
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
                proxy_event_metrics = _average_metric_reports(proxy_reports)
                proxy_event_score = float(sum(proxy_scores) / max(len(proxy_scores), 1))
            score = float(train_cfg.get("model_selection_dev_weight", 1.0)) * _model_selection_score(
                dev_metrics,
                train_cfg,
                "dev",
            )
            if proxy_event_score is not None:
                score += float(train_cfg.get("model_selection_proxy_event_weight", 0.0)) * proxy_event_score
            history_entry = {"epoch": epoch, "train_loss": train_loss, "dev_metrics": dev_metrics}
            if proxy_event_metrics is not None:
                history_entry["proxy_event_metrics"] = proxy_event_metrics
            history.append(history_entry)
            if epoch >= model_selection_start_epoch:
                if score > best_score:
                    best_score = score
                    best_state = (
                        {key: value.detach().cpu().clone() for key, value in ema_state.items()}
                        if ema_state is not None
                        else _clone_state_dict(model)
                    )
                current_state = (
                    {key: value.detach().cpu().clone() for key, value in ema_state.items()}
                    if ema_state is not None
                    else _clone_state_dict(model)
                )
                top_states.append((score, current_state))
                top_states.sort(key=lambda item: item[0], reverse=True)
                if len(top_states) > checkpoint_average_top_k:
                    top_states = top_states[:checkpoint_average_top_k]
        if checkpoint_average_top_k > 1 and top_states:
            best_state = _average_state_dicts([state for _, state in top_states])
        _load_state_dict(model, best_state, device)
        result["history"] = history
        result["best_dev_score"] = best_score
        if checkpoint_average_top_k > 1:
            result["checkpoint_average_top_k"] = checkpoint_average_top_k
        if model_selection_start_epoch > 1:
            result["model_selection_start_epoch"] = model_selection_start_epoch
        if proxy_event_splits is not None and len(proxy_event_splits) > 1:
            result["proxy_event_num_samples"] = len(proxy_event_splits)
        if isinstance(model, RetrievalForecaster):
            result["dev"] = _evaluate_retrieval(model, train_split, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
            result["test"] = _evaluate_retrieval(model, train_split, test_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
            if event_split is not None:
                result["test_event_disjoint"] = _evaluate_retrieval(model, train_split, event_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
        else:
            result["dev"] = _evaluate_parametric(model, dev_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
            result["test"] = _evaluate_parametric(model, test_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)
            if event_split is not None:
                result["test_event_disjoint"] = _evaluate_parametric(model, event_split, device=device, batch_size=batch_size, metric_cfg=metric_cfg)

    output_dir = Path(output_cfg.get("dir", "outputs/results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / f"{config['experiment_name']}.json", to_builtin(result))
    return to_builtin(result)
