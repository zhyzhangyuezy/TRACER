from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"
CACHE_DIR = RESULT_DIR / "audits" / "label_grounded_evidence_seed_exports"

SEEDS = [7, 13, 21]
BUDGETS = [0.05, 0.10, 0.20]
FUSION_WEIGHT = 0.25
BOOTSTRAPS = 2000


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rank01(values: np.ndarray) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort").astype(np.float64)
    return order / float(values.shape[0] - 1)


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(values.mean())


def _payload(seed: int) -> dict[str, Any]:
    path = CACHE_DIR / f"r240_tracer_adaptive_event_atlasv2_public_seed{seed}.json"
    return _load_json(path)


def _arrays(seed: int) -> dict[str, Any]:
    pred = _payload(seed)["predictions"]
    y_true = np.asarray(pred["y_true"], dtype=int)
    y_score = np.asarray(pred["y_score"], dtype=np.float64)
    retrieved = np.asarray(pred["retrieved_label_main"], dtype=np.float64)
    if retrieved.ndim == 1:
        retrieved = retrieved[:, None]
    analog_fraction = retrieved.mean(axis=1)
    score_rank = _rank01(y_score)
    analog_rank = analog_fraction
    fused = (1.0 - FUSION_WEIGHT) * score_rank + FUSION_WEIGHT * analog_rank
    return {
        "y_true": y_true,
        "time_to_escalation": np.asarray(pred["time_to_escalation"], dtype=np.float64),
        "incident_id": np.asarray(pred["incident_id"], dtype=object),
        "scores": {
            "score_only": score_rank,
            "analog_only": analog_rank + 1e-9 * score_rank,
            "score_plus_analog": fused,
        },
    }


def _review_metrics(y_true: np.ndarray, ranking_score: np.ndarray, time_to_escalation: np.ndarray, budget: float) -> dict[str, Any]:
    k = max(1, int(math.ceil(float(y_true.shape[0]) * budget)))
    order = np.argsort(-ranking_score, kind="mergesort")[:k]
    reviewed_y = y_true[order]
    positive_total = int(y_true.sum())
    reviewed_positives = int(reviewed_y.sum())
    lead_values = time_to_escalation[order][reviewed_y.astype(bool)]
    return {
        "budget_fraction": budget,
        "budget_windows": k,
        "positive_recall": reviewed_positives / max(positive_total, 1),
        "positive_precision": reviewed_positives / max(k, 1),
        "reviewed_positives": reviewed_positives,
        "total_positives": positive_total,
        "mean_lead_time": _mean_or_none(lead_values),
    }


def _bootstrap_delta(seed: int, budget: float, lhs: str, rhs: str) -> dict[str, float]:
    arrays = _arrays(seed)
    y_true = arrays["y_true"]
    time_to_escalation = arrays["time_to_escalation"]
    incidents = arrays["incident_id"]
    unique_incidents = np.unique(incidents)
    rng = np.random.default_rng(20260417 + seed + int(1000 * budget))
    deltas = []
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(unique_incidents, size=unique_incidents.shape[0], replace=True)
        indices = np.concatenate([np.flatnonzero(incidents == incident) for incident in sampled])
        if indices.size == 0 or y_true[indices].sum() == 0:
            continue
        lhs_metric = _review_metrics(y_true[indices], arrays["scores"][lhs][indices], time_to_escalation[indices], budget)
        rhs_metric = _review_metrics(y_true[indices], arrays["scores"][rhs][indices], time_to_escalation[indices], budget)
        deltas.append(float(lhs_metric["positive_recall"]) - float(rhs_metric["positive_recall"]))
    if not deltas:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def build_audit() -> dict[str, Any]:
    methods = {
        "score_only": "Score only",
        "analog_only": "Retrieved-label analogs only",
        "score_plus_analog": "Score + retrieved-label analogs",
    }
    rows = []
    per_method_budget: dict[str, dict[float, list[dict[str, Any]]]] = {
        method: {budget: [] for budget in BUDGETS} for method in methods
    }
    for seed in SEEDS:
        arrays = _arrays(seed)
        for method in methods:
            for budget in BUDGETS:
                metric = _review_metrics(
                    arrays["y_true"],
                    arrays["scores"][method],
                    arrays["time_to_escalation"],
                    budget,
                )
                metric["seed"] = seed
                metric["method"] = method
                per_method_budget[method][budget].append(metric)
                rows.append(metric)

    summary_rows = []
    for method, label in methods.items():
        entry: dict[str, Any] = {"method": method, "label": label}
        for budget in BUDGETS:
            metrics = per_method_budget[method][budget]
            recalls = [float(item["positive_recall"]) for item in metrics]
            precisions = [float(item["positive_precision"]) for item in metrics]
            leads = [float(item["mean_lead_time"]) for item in metrics if item["mean_lead_time"] is not None]
            entry[f"recall_at_{int(100 * budget)}"] = float(mean(recalls))
            entry[f"recall_std_at_{int(100 * budget)}"] = float(pstdev(recalls))
            entry[f"precision_at_{int(100 * budget)}"] = float(mean(precisions))
            entry[f"precision_std_at_{int(100 * budget)}"] = float(pstdev(precisions))
            entry[f"lead_at_{int(100 * budget)}"] = float(mean(leads)) if leads else None
        summary_rows.append(entry)

    bootstrap = {
        f"{int(100 * budget)}pct": [
            _bootstrap_delta(seed, budget, "score_plus_analog", "score_only") for seed in SEEDS
        ]
        for budget in BUDGETS
    }
    bootstrap_summary = {}
    for budget in BUDGETS:
        key = f"{int(100 * budget)}pct"
        means = [item["mean"] for item in bootstrap[key]]
        lows = [item["ci_low"] for item in bootstrap[key]]
        highs = [item["ci_high"] for item in bootstrap[key]]
        bootstrap_summary[key] = {
            "mean_delta_recall": float(mean(means)),
            "mean_ci_low": float(mean(lows)),
            "mean_ci_high": float(mean(highs)),
        }

    return {
        "audit": "ATLASv2 retrospective review-budget evidence augmentation audit",
        "seeds": SEEDS,
        "budgets": BUDGETS,
        "fusion_weight": FUSION_WEIGHT,
        "rows": rows,
        "summary": summary_rows,
        "bootstrap_score_plus_analog_minus_score_only": bootstrap_summary,
        "note": "This audit uses the train-memory retrieved labels exposed in the label-grounded evidence cache as a retrospective, non-human review-budget proxy. The fusion weight is fixed before evaluation and no threshold is tuned on the held-out windows.",
    }


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# ATLASv2 review-budget evidence augmentation audit",
        "",
        audit["note"],
        "",
        "| Method | Recall@5% | Prec@5% | Recall@10% | Prec@10% | Recall@20% | Prec@20% | Lead@10% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["summary"]:
        lead = row["lead_at_10"]
        lead_text = "n/a" if lead is None else f"{lead:.1f}"
        lines.append(
            "| {label} | {r5:.3f} | {p5:.3f} | {r10:.3f} | {p10:.3f} | {r20:.3f} | {p20:.3f} | {lead} |".format(
                label=row["label"],
                r5=float(row["recall_at_5"]),
                p5=float(row["precision_at_5"]),
                r10=float(row["recall_at_10"]),
                p10=float(row["precision_at_10"]),
                r20=float(row["recall_at_20"]),
                p20=float(row["precision_at_20"]),
                lead=lead_text,
            )
        )
    lines += ["", "## Paired incident-block bootstrap: score+analog minus score-only recall", "", "| Budget | Mean delta | Mean 95% CI |", "|---|---:|---:|"]
    for key, row in audit["bootstrap_score_plus_analog_minus_score_only"].items():
        lines.append(f"| {key} | {row['mean_delta_recall']:+.3f} | [{row['mean_ci_low']:+.3f}, {row['mean_ci_high']:+.3f}] |")
    (RESULT_DIR / "atlasv2_review_augmentation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float) -> str:
    return f"${value:.3f}$"


def write_latex(audit: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Retrospective review-budget evidence augmentation on ATLASv2 held-out-family windows. The audit compares score-only ranking with retrieved-label analog evidence from the train-only memory bank. The fused rank uses a fixed 0.25 analog weight without held-out tuning; the numbers are means over three seeds.}",
        r"\label{tab:atlasv2-review-augmentation}",
        r"\maxtablewidth{",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Ranking signal & Rec@5\% & Prec@5\% & Rec@10\% & Prec@10\% & Rec@20\% & Prec@20\% & Lead@10\% \\",
        r"\midrule",
    ]
    for row in audit["summary"]:
        lead = row["lead_at_10"]
        lead_text = "--" if lead is None else f"${float(lead):.1f}$"
        lines.append(
            "{label} & {r5} & {p5} & {r10} & {p10} & {r20} & {p20} & {lead} \\\\".format(
                label=str(row["label"]).replace("_", r"\_"),
                r5=_fmt(float(row["recall_at_5"])),
                p5=_fmt(float(row["precision_at_5"])),
                r10=_fmt(float(row["recall_at_10"])),
                p10=_fmt(float(row["precision_at_10"])),
                r20=_fmt(float(row["recall_at_20"])),
                p20=_fmt(float(row["precision_at_20"])),
                lead=lead_text,
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    (FIG_DIR / "tab_atlasv2_review_augmentation_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_audit()
    (RESULT_DIR / "atlasv2_review_augmentation_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit)
    write_latex(audit)
    print("Wrote outputs/results/atlasv2_review_augmentation_audit.json")
    print("Wrote figures/tab_atlasv2_review_augmentation_audit.tex")


if __name__ == "__main__":
    main()
