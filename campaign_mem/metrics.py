from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
)


def _clip_scores(y_score: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_score, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_score))


def negative_log_likelihood(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(log_loss(y_true, _clip_scores(y_score), labels=[0, 1]))


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_bins: int = 10,
) -> float:
    if y_true.size == 0:
        return 0.0
    clipped = _clip_scores(y_score)
    bin_edges = np.linspace(0.0, 1.0, max(int(num_bins), 1) + 1)
    total = float(clipped.shape[0])
    ece = 0.0
    for index in range(bin_edges.shape[0] - 1):
        left = float(bin_edges[index])
        right = float(bin_edges[index + 1])
        if index == bin_edges.shape[0] - 2:
            mask = (clipped >= left) & (clipped <= right)
        else:
            mask = (clipped >= left) & (clipped < right)
        if not np.any(mask):
            continue
        confidence = float(clipped[mask].mean())
        accuracy = float(y_true[mask].mean())
        ece += (mask.sum() / total) * abs(accuracy - confidence)
    return float(ece)


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = (2.0 * precision * recall) / np.clip(precision + recall, 1e-8, None)
    return float(np.max(f1))


def lead_time_at_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    time_to_escalation: np.ndarray,
    target_precision: float = 0.8,
) -> dict[str, float]:
    best_threshold = None
    best_recall = -1.0
    best_precision = 0.0
    best_lead_time = 0.0
    thresholds = np.unique(np.round(y_score, 6))[::-1]
    for threshold in thresholds:
        predictions = (y_score >= threshold).astype(np.int64)
        if predictions.sum() == 0:
            continue
        precision = precision_score(y_true, predictions, zero_division=0.0)
        if precision < target_precision:
            continue
        true_positive_mask = (predictions == 1) & (y_true == 1)
        recall = float(true_positive_mask.sum() / max(int(y_true.sum()), 1))
        lead_time = float(time_to_escalation[true_positive_mask].mean()) if true_positive_mask.any() else 0.0
        if recall > best_recall:
            best_threshold = float(threshold)
            best_recall = recall
            best_precision = float(precision)
            best_lead_time = lead_time
    return {
        "threshold": float(best_threshold or 0.0),
        "precision": best_precision,
        "recall": best_recall if best_recall >= 0.0 else 0.0,
        "lead_time": best_lead_time,
    }


def analog_fidelity_at_k(
    query_future: np.ndarray,
    memory_future: np.ndarray,
    retrieved_indices: np.ndarray | None,
    threshold: float,
) -> float:
    if retrieved_indices is None or retrieved_indices.size == 0:
        return 0.0
    query_expand = query_future[:, None, :]
    retrieved_future = memory_future[retrieved_indices]
    distances = np.mean(np.abs(query_expand - retrieved_future), axis=-1)
    return float((distances <= threshold).mean() * 100.0)


def top1_tte_error(
    y_true: np.ndarray,
    query_tte: np.ndarray,
    memory_tte: np.ndarray,
    retrieved_indices: np.ndarray | None,
) -> float:
    if retrieved_indices is None or retrieved_indices.size == 0:
        return 0.0
    positive_mask = y_true.astype(bool)
    if not positive_mask.any():
        return 0.0
    top1 = retrieved_indices[:, 0]
    return float(np.mean(np.abs(query_tte[positive_mask] - memory_tte[top1][positive_mask])))


def build_metric_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    time_to_escalation: np.ndarray,
    target_precision: float,
    query_future: np.ndarray | None = None,
    memory_future: np.ndarray | None = None,
    retrieved_indices: np.ndarray | None = None,
    analog_threshold: float | None = None,
    memory_tte: np.ndarray | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "AUPRC": auprc(y_true, y_score),
        "AUROC": auroc(y_true, y_score),
        "Brier": brier(y_true, y_score),
        "LogLoss": negative_log_likelihood(y_true, y_score),
        "ECE@10": expected_calibration_error(y_true, y_score, num_bins=10),
        "BestF1": best_f1(y_true, y_score),
    }
    lead_time = lead_time_at_precision(
        y_true=y_true,
        y_score=y_score,
        time_to_escalation=time_to_escalation,
        target_precision=target_precision,
    )
    report["LeadTime@P80"] = lead_time["lead_time"]
    report["Precision@P80"] = lead_time["precision"]
    report["Recall@P80"] = lead_time["recall"]
    report["LeadTimeDetail"] = lead_time
    if query_future is not None and memory_future is not None and analog_threshold is not None:
        report["Analog-Fidelity@5"] = analog_fidelity_at_k(
            query_future=query_future,
            memory_future=memory_future,
            retrieved_indices=retrieved_indices,
            threshold=analog_threshold,
        )
    if memory_tte is not None:
        report["TTE-Err@1"] = top1_tte_error(
            y_true=y_true,
            query_tte=time_to_escalation,
            memory_tte=memory_tte,
            retrieved_indices=retrieved_indices,
        )
    return report
