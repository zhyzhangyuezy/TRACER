from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

ROOT = SCRIPT_DIR.parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_common import predict_split, train_best_model
from campaign_mem.data.dataset import load_split
from campaign_mem.metrics import build_metric_report
from campaign_mem.utils import load_yaml, save_json, set_seed
import numpy as np


SEEDS = [7, 13, 21]
TOP_K = 5
SPLIT_NAME = "test_event_disjoint"
BOOTSTRAP_SAMPLES = 5000
METHOD_SPECS = [
    ("configs/experiments/r015_pure_knn_atlasv2_public_event_disjoint.yaml", "Pure-kNN-Retrieval"),
    ("configs/experiments/r010_prefix_retrieval_atlasv2_public_event_disjoint.yaml", "Prefix-Only-Retrieval + Fusion"),
    ("configs/experiments/r010_campaign_mem_atlasv2_public_event_disjoint.yaml", "Shared-Encoder TRACER"),
    ("configs/experiments/r240_tracer_adaptive_event_atlasv2_public.yaml", "TRACER"),
]
PAIRWISE_BASELINES = ["Prefix-Only-Retrieval + Fusion", "Shared-Encoder TRACER"]
TRACER_NAME = "TRACER"


def _cosine_topk_indices(train_summary: np.ndarray, eval_summary: np.ndarray, top_k: int) -> np.ndarray:
    train_norm = train_summary / np.linalg.norm(train_summary, axis=1, keepdims=True).clip(min=1e-6)
    eval_norm = eval_summary / np.linalg.norm(eval_summary, axis=1, keepdims=True).clip(min=1e-6)
    similarity = eval_norm @ train_norm.T
    return np.argsort(-similarity, axis=1)[:, :top_k]


def _pure_knn_export(config_path: Path, seed: int) -> dict[str, Any]:
    config = load_yaml(config_path)
    set_seed(seed)
    data_cfg = config["data"]
    metric_cfg = config.get("metrics", {})
    train_split = load_split(data_cfg["dataset_dir"], data_cfg.get("train_split", "train"))
    dev_split = load_split(data_cfg["dataset_dir"], data_cfg.get("dev_split", "dev"))
    eval_split = load_split(data_cfg["dataset_dir"], data_cfg.get("test_split", SPLIT_NAME))

    train_summary = train_split.summary_features()
    dev_summary = dev_split.summary_features()
    eval_summary = eval_split.summary_features()
    dev_indices = _cosine_topk_indices(train_summary, dev_summary, TOP_K)
    eval_indices = _cosine_topk_indices(train_summary, eval_summary, TOP_K)

    dev_scores = train_split.label_main[dev_indices].mean(axis=1)
    eval_scores = train_split.label_main[eval_indices].mean(axis=1)
    dev_metrics = build_metric_report(
        y_true=dev_split.label_main.astype(np.int64),
        y_score=dev_scores,
        time_to_escalation=dev_split.time_to_escalation,
        target_precision=float(metric_cfg.get("target_precision", 0.8)),
        query_future=dev_split.future_signature,
        memory_future=train_split.future_signature,
        retrieved_indices=dev_indices,
        analog_threshold=float(metric_cfg.get("analog_fidelity_distance_threshold", 0.35)),
        memory_tte=train_split.time_to_escalation,
    )
    threshold = float(dev_metrics.get("LeadTimeDetail", {}).get("threshold", 0.0))
    threshold = threshold if threshold > 0.0 else 0.5

    return {
        "metrics": build_metric_report(
            y_true=eval_split.label_main.astype(np.int64),
            y_score=eval_scores,
            time_to_escalation=eval_split.time_to_escalation,
            target_precision=float(metric_cfg.get("target_precision", 0.8)),
            query_future=eval_split.future_signature,
            memory_future=train_split.future_signature,
            retrieved_indices=eval_indices,
            analog_threshold=float(metric_cfg.get("analog_fidelity_distance_threshold", 0.35)),
            memory_tte=train_split.time_to_escalation,
        ),
        "threshold": threshold,
        "predictions": {
            "y_true": eval_split.label_main.astype(int).tolist(),
            "y_score": eval_scores.astype(float).tolist(),
            "retrieved_label_main": train_split.label_main[eval_indices].astype(int).tolist(),
            "incident_id": list(eval_split.incident_id),
        },
    }


def _load_or_rerun(config_path: Path, display_name: str, seed: int, cache_dir: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    base_name = str(config["experiment_name"])
    cache_path = cache_dir / f"{base_name}_seed{seed}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["display_name"] = display_name
        return payload

    if base_name.startswith("r015_pure_knn"):
        export = _pure_knn_export(config_path, seed)
    else:
        config["seed"] = seed
        config["experiment_name"] = f"{base_name}_labelaudit_seed{seed}"
        artifacts = train_best_model(config)
        export = predict_split(artifacts, SPLIT_NAME)

    payload = {
        "display_name": display_name,
        "base_experiment_name": base_name,
        "seed": seed,
        "metrics": export["metrics"],
        "threshold": export["threshold"],
        "predictions": export["predictions"],
    }
    save_json(cache_path, payload)
    return payload


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size > 0 else 0.0


def _label_grounded_metrics_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
    retrieved: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    alerted_mask = y_score >= float(threshold)

    pos_rows = retrieved[positive_mask]
    neg_rows = retrieved[negative_mask]
    alerted_rows = retrieved[alerted_mask]
    return {
        "PosHit@1": 100.0 * _safe_mean(pos_rows[:, 0]) if pos_rows.size > 0 else 0.0,
        "PosHit@5": 100.0 * _safe_mean((pos_rows.sum(axis=1) > 0).astype(float)) if pos_rows.size > 0 else 0.0,
        "PosPrec@5": 100.0 * _safe_mean(pos_rows.mean(axis=1)) if pos_rows.size > 0 else 0.0,
        "AlertedPosPrec@5": 100.0 * _safe_mean(alerted_rows.mean(axis=1)) if alerted_rows.size > 0 else 0.0,
        "NegContam@5": 100.0 * _safe_mean(neg_rows.mean(axis=1)) if neg_rows.size > 0 else 0.0,
        "AlertRate": 100.0 * float(alerted_mask.mean()),
    }


def _label_grounded_metrics(predictions: dict[str, Any], threshold: float) -> dict[str, float]:
    return _label_grounded_metrics_arrays(
        np.asarray(predictions["y_true"], dtype=int),
        np.asarray(predictions["y_score"], dtype=float),
        np.asarray(predictions["retrieved_label_main"], dtype=int),
        threshold,
    )


def _sample_indices_by_incident(incident_id: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique_incidents = np.unique(incident_id.astype(str))
    sampled = rng.choice(unique_incidents, size=unique_incidents.size, replace=True)
    chunks = [np.flatnonzero(incident_id.astype(str) == value) for value in sampled]
    return np.concatenate(chunks, axis=0)


def _pairwise_deltas_from_indices(
    tracer_payloads: list[dict[str, Any]],
    baseline_payloads: list[dict[str, Any]],
    indices: np.ndarray | None,
) -> dict[str, float]:
    poshit_deltas: list[float] = []
    alerted_prec_deltas: list[float] = []
    cleanneg_deltas: list[float] = []
    for tracer_payload, baseline_payload in zip(tracer_payloads, baseline_payloads):
        tracer_pred = tracer_payload["predictions"]
        baseline_pred = baseline_payload["predictions"]

        tracer_y = np.asarray(tracer_pred["y_true"], dtype=int)
        baseline_y = np.asarray(baseline_pred["y_true"], dtype=int)
        if indices is not None:
            tracer_y = tracer_y[indices]
            baseline_y = baseline_y[indices]

        tracer_metrics = _label_grounded_metrics_arrays(
            tracer_y,
            np.asarray(tracer_pred["y_score"], dtype=float)[indices] if indices is not None else np.asarray(tracer_pred["y_score"], dtype=float),
            np.asarray(tracer_pred["retrieved_label_main"], dtype=int)[indices] if indices is not None else np.asarray(tracer_pred["retrieved_label_main"], dtype=int),
            float(tracer_payload["threshold"]),
        )
        baseline_metrics = _label_grounded_metrics_arrays(
            baseline_y,
            np.asarray(baseline_pred["y_score"], dtype=float)[indices] if indices is not None else np.asarray(baseline_pred["y_score"], dtype=float),
            np.asarray(baseline_pred["retrieved_label_main"], dtype=int)[indices] if indices is not None else np.asarray(baseline_pred["retrieved_label_main"], dtype=int),
            float(baseline_payload["threshold"]),
        )
        poshit_deltas.append(tracer_metrics["PosHit@5"] - baseline_metrics["PosHit@5"])
        alerted_prec_deltas.append(tracer_metrics["AlertedPosPrec@5"] - baseline_metrics["AlertedPosPrec@5"])
        cleanneg_deltas.append(baseline_metrics["NegContam@5"] - tracer_metrics["NegContam@5"])
    return {
        "PosHit@5": float(mean(poshit_deltas)),
        "AlertedPosPrec@5": float(mean(alerted_prec_deltas)),
        "CleanNeg@5": float(mean(cleanneg_deltas)),
    }


def _bootstrap_pairwise_deltas(
    tracer_payloads: list[dict[str, Any]],
    baseline_payloads: list[dict[str, Any]],
    *,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    incident_id = np.asarray(tracer_payloads[0]["predictions"]["incident_id"], dtype=str)
    rng = np.random.default_rng(seed)
    metrics = {
        "PosHit@5": [],
        "AlertedPosPrec@5": [],
        "CleanNeg@5": [],
    }
    while len(metrics["PosHit@5"]) < num_samples:
        indices = _sample_indices_by_incident(incident_id, rng)
        deltas = _pairwise_deltas_from_indices(tracer_payloads, baseline_payloads, indices)
        for key, value in deltas.items():
            metrics[key].append(value)
    summary: dict[str, Any] = {}
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=float)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "positive_mass": float((arr > 0.0).mean()),
        }
    return summary


def _format_metric(mean_value: float, std_value: float, decimals: int = 1) -> str:
    return f"${mean_value:.{decimals}f} \\pm {std_value:.{decimals}f}$"


def _format_delta(value: float, decimals: int = 1) -> str:
    return f"${value:+.{decimals}f}$"


def _format_ci(low: float, high: float, decimals: int = 1) -> str:
    return f"$[{low:.{decimals}f}, {high:.{decimals}f}]$"


def build_table_tex(summary: dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Label-grounded evidence audit on the ATLASv2 held-out-family split. These metrics are computed only from retrieved train labels, not from the future-signature space used by AF@5 or TTE-Err@1. Higher is better for PosHit@1, PosHit@5, PosPrec@5, and AlertedPosPrec@5; lower is better for NegContam@5. The alert rate column reports the fraction of held-out windows whose score exceeds the dev-derived operating threshold.}",
        r"\label{tab:atlasv2-label-grounded-evidence}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & PosHit@1 & PosHit@5 & PosPrec@5 & AlertedPosPrec@5 & NegContam@5 & Alert rate \\",
        r"\midrule",
    ]
    for row in summary["methods"]:
        lines.append(
            row["display_name"]
            + " & "
            + _format_metric(row["mean"]["PosHit@1"], row["std"]["PosHit@1"])
            + " & "
            + _format_metric(row["mean"]["PosHit@5"], row["std"]["PosHit@5"])
            + " & "
            + _format_metric(row["mean"]["PosPrec@5"], row["std"]["PosPrec@5"])
            + " & "
            + _format_metric(row["mean"]["AlertedPosPrec@5"], row["std"]["AlertedPosPrec@5"])
            + " & "
            + _format_metric(row["mean"]["NegContam@5"], row["std"]["NegContam@5"])
            + " & "
            + _format_metric(row["mean"]["AlertRate"], row["std"]["AlertRate"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_pairwise_table_tex(summary: dict[str, Any]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Paired label-grounded evidence deltas on the ATLASv2 held-out-family split. All deltas favor the final TRACER route when positive. $\Delta$CleanNeg@5 is defined as baseline NegContam@5 minus TRACER NegContam@5, so positive values mean that TRACER retrieves cleaner negatives. Confidence intervals come from paired incident-block bootstrap over the released held-out incidents.}",
        r"\label{tab:atlasv2-label-grounded-pairwise}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Baseline & $\Delta$PosHit@5 & 95\% CI & $\Delta$AlertedPosPrec@5 & 95\% CI & $\Delta$CleanNeg@5 & 95\% CI \\",
        r"\midrule",
    ]
    for row in summary["pairwise"]:
        lines.append(
            row["baseline"]
            + " & "
            + _format_delta(row["mean_delta"]["PosHit@5"])
            + " & "
            + _format_ci(row["bootstrap"]["PosHit@5"]["ci95_low"], row["bootstrap"]["PosHit@5"]["ci95_high"])
            + " & "
            + _format_delta(row["mean_delta"]["AlertedPosPrec@5"])
            + " & "
            + _format_ci(row["bootstrap"]["AlertedPosPrec@5"]["ci95_low"], row["bootstrap"]["AlertedPosPrec@5"]["ci95_high"])
            + " & "
            + _format_delta(row["mean_delta"]["CleanNeg@5"])
            + " & "
            + _format_ci(row["bootstrap"]["CleanNeg@5"]["ci95_low"], row["bootstrap"]["CleanNeg@5"]["ci95_high"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    cache_dir = ROOT / "outputs" / "results" / "audits" / "label_grounded_evidence_seed_exports"
    method_rows: list[dict[str, Any]] = []
    payloads_by_method: dict[str, list[dict[str, Any]]] = {}

    for config_path_str, display_name in METHOD_SPECS:
        config_path = ROOT / config_path_str
        per_seed_metrics: list[dict[str, float]] = []
        per_seed_payloads: list[dict[str, Any]] = []
        for seed in SEEDS:
            payload = _load_or_rerun(config_path, display_name, seed, cache_dir)
            per_seed_payloads.append(payload)
            per_seed_metrics.append(_label_grounded_metrics(payload["predictions"], float(payload["threshold"])))
        payloads_by_method[display_name] = per_seed_payloads
        metric_keys = list(per_seed_metrics[0].keys())
        method_rows.append(
            {
                "display_name": display_name,
                "mean": {key: float(mean(row[key] for row in per_seed_metrics)) for key in metric_keys},
                "std": {key: float(pstdev(row[key] for row in per_seed_metrics)) for key in metric_keys},
                "per_seed": {str(SEEDS[idx]): per_seed_metrics[idx] for idx in range(len(SEEDS))},
            }
        )

    tracer_payloads = payloads_by_method[TRACER_NAME]
    pairwise_rows: list[dict[str, Any]] = []
    for baseline_name in PAIRWISE_BASELINES:
        baseline_payloads = payloads_by_method[baseline_name]
        mean_delta = _pairwise_deltas_from_indices(tracer_payloads, baseline_payloads, indices=None)
        bootstrap = _bootstrap_pairwise_deltas(
            tracer_payloads,
            baseline_payloads,
            num_samples=BOOTSTRAP_SAMPLES,
            seed=20260405 + len(pairwise_rows),
        )
        pairwise_rows.append(
            {
                "baseline": baseline_name,
                "mean_delta": mean_delta,
                "bootstrap": bootstrap,
            }
        )

    summary = {
        "split": SPLIT_NAME,
        "seeds": SEEDS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "methods": method_rows,
        "pairwise": pairwise_rows,
    }
    save_json(ROOT / "outputs" / "results" / "public_label_grounded_evidence_audit.json", summary)
    (ROOT / "figures" / "tab_public_label_grounded_evidence.tex").write_text(build_table_tex(summary), encoding="utf-8")
    (ROOT / "figures" / "tab_public_label_grounded_pairwise.tex").write_text(build_pairwise_table_tex(summary), encoding="utf-8")
    print(ROOT / "outputs" / "results" / "public_label_grounded_evidence_audit.json")
    print(ROOT / "figures" / "tab_public_label_grounded_evidence.tex")
    print(ROOT / "figures" / "tab_public_label_grounded_pairwise.tex")


if __name__ == "__main__":
    main()
