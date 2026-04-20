from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SAMPLES = 5000


def _parse_spec(text: str) -> tuple[str, str]:
    if "::" not in text:
        raise ValueError(f"Expected EXPERIMENT::LABEL format, got: {text}")
    experiment, label = text.split("::", 1)
    experiment = experiment.strip()
    label = label.strip()
    if not experiment or not label:
        raise ValueError(f"Malformed spec: {text}")
    return experiment, label


def _load_payloads(cache_dir: Path, experiment: str, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob(f"{experiment}_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["display_name"] = label
        rows.append(payload)
    if not rows:
        raise FileNotFoundError(f"No cached exports found for {experiment} under {cache_dir}")
    rows.sort(key=lambda row: int(row["seed"]))
    return rows


def _budget_metrics_from_indices(
    y_true: np.ndarray,
    y_score: np.ndarray,
    time_to_escalation: np.ndarray,
    *,
    budget_fraction: float,
) -> dict[str, float]:
    budget = max(1, int(round(len(y_true) * budget_fraction)))
    order = np.argsort(-y_score)
    top = order[:budget]
    positives_total = max(int(y_true.sum()), 1)
    positives_top = y_true[top] == 1
    lead_values = time_to_escalation[top][positives_top]
    return {
        "budget_count": float(budget),
        "PosRecall": 100.0 * float(positives_top.sum()) / positives_total,
        "PosPrec": 100.0 * float(positives_top.mean()),
        "MeanLead": float(lead_values.mean()) if lead_values.size > 0 else 0.0,
    }


def _sample_indices_by_incident(incident_id: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique_incidents = np.unique(incident_id.astype(str))
    sampled = rng.choice(unique_incidents, size=unique_incidents.size, replace=True)
    chunks = [np.flatnonzero(incident_id.astype(str) == value) for value in sampled]
    return np.concatenate(chunks, axis=0)


def _pairwise_delta(
    tracer_payloads: list[dict[str, Any]],
    baseline_payloads: list[dict[str, Any]],
    *,
    budget_fraction: float,
    indices: np.ndarray | None = None,
) -> dict[str, float]:
    recall_deltas: list[float] = []
    precision_deltas: list[float] = []
    lead_deltas: list[float] = []
    for tracer_payload, baseline_payload in zip(tracer_payloads, baseline_payloads):
        tracer_pred = tracer_payload["predictions"]
        baseline_pred = baseline_payload["predictions"]
        tracer_y = np.asarray(tracer_pred["y_true"], dtype=int)
        baseline_y = np.asarray(baseline_pred["y_true"], dtype=int)
        tracer_score = np.asarray(tracer_pred["y_score"], dtype=float)
        baseline_score = np.asarray(baseline_pred["y_score"], dtype=float)
        tracer_tte = np.asarray(tracer_pred["time_to_escalation"], dtype=float)
        baseline_tte = np.asarray(baseline_pred["time_to_escalation"], dtype=float)
        if indices is not None:
            tracer_y = tracer_y[indices]
            baseline_y = baseline_y[indices]
            tracer_score = tracer_score[indices]
            baseline_score = baseline_score[indices]
            tracer_tte = tracer_tte[indices]
            baseline_tte = baseline_tte[indices]
        tracer_metrics = _budget_metrics_from_indices(tracer_y, tracer_score, tracer_tte, budget_fraction=budget_fraction)
        baseline_metrics = _budget_metrics_from_indices(
            baseline_y,
            baseline_score,
            baseline_tte,
            budget_fraction=budget_fraction,
        )
        recall_deltas.append(tracer_metrics["PosRecall"] - baseline_metrics["PosRecall"])
        precision_deltas.append(tracer_metrics["PosPrec"] - baseline_metrics["PosPrec"])
        lead_deltas.append(tracer_metrics["MeanLead"] - baseline_metrics["MeanLead"])
    return {
        "PosRecall": float(mean(recall_deltas)),
        "PosPrec": float(mean(precision_deltas)),
        "MeanLead": float(mean(lead_deltas)),
    }


def _bootstrap_pairwise(
    tracer_payloads: list[dict[str, Any]],
    baseline_payloads: list[dict[str, Any]],
    *,
    budget_fraction: float,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    incident_id = np.asarray(tracer_payloads[0]["predictions"]["incident_id"], dtype=str)
    rng = np.random.default_rng(seed)
    values = {"PosRecall": [], "PosPrec": [], "MeanLead": []}
    while len(values["PosRecall"]) < num_samples:
        indices = _sample_indices_by_incident(incident_id, rng)
        delta = _pairwise_delta(
            tracer_payloads,
            baseline_payloads,
            budget_fraction=budget_fraction,
            indices=indices,
        )
        for key, value in delta.items():
            values[key].append(value)
    summary: dict[str, Any] = {}
    for key, rows in values.items():
        arr = np.asarray(rows, dtype=float)
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


def build_summary_table(summary: dict[str, Any]) -> str:
    budget_pct = 100.0 * float(summary["budget_fraction"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Budgeted analyst-review audit on the {summary['benchmark_caption']} benchmark. We rank windows by model score and let the analyst review only the top {budget_pct:.1f}\% of windows. Higher is better for PosRecall and PosPrec; MeanLead reports the average time-to-escalation among reviewed positive windows.}}",
        rf"\label{{tab:{summary['table_label']}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"Method & PosRecall@{budget_pct:.1f}\% & PosPrec@{budget_pct:.1f}\% & MeanLead@{budget_pct:.1f}\% \\",
        r"\midrule",
    ]
    for row in summary["methods"]:
        lines.append(
            row["display_name"]
            + " & "
            + _format_metric(row["mean"]["PosRecall"], row["std"]["PosRecall"])
            + " & "
            + _format_metric(row["mean"]["PosPrec"], row["std"]["PosPrec"])
            + " & "
            + _format_metric(row["mean"]["MeanLead"], row["std"]["MeanLead"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_pairwise_table(summary: dict[str, Any]) -> str:
    budget_pct = 100.0 * float(summary["budget_fraction"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Paired budgeted-review deltas on the {summary['benchmark_caption']} benchmark at a top-{budget_pct:.1f}\% review budget. All deltas favor TRACER when positive. Confidence intervals come from paired incident-block bootstrap over the evaluated split.}}",
        rf"\label{{tab:{summary['pairwise_table_label']}}}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        rf"Baseline & $\Delta$PosRecall@{budget_pct:.1f}\% & 95\% CI & $\Delta$PosPrec@{budget_pct:.1f}\% & 95\% CI & $\Delta$MeanLead@{budget_pct:.1f}\% & 95\% CI \\",
        r"\midrule",
    ]
    for row in summary["pairwise"]:
        lines.append(
            row["baseline"]
            + " & "
            + _format_delta(row["mean_delta"]["PosRecall"])
            + " & "
            + _format_ci(row["bootstrap"]["PosRecall"]["ci95_low"], row["bootstrap"]["PosRecall"]["ci95_high"])
            + " & "
            + _format_delta(row["mean_delta"]["PosPrec"])
            + " & "
            + _format_ci(row["bootstrap"]["PosPrec"]["ci95_low"], row["bootstrap"]["PosPrec"]["ci95_high"])
            + " & "
            + _format_delta(row["mean_delta"]["MeanLead"])
            + " & "
            + _format_ci(row["bootstrap"]["MeanLead"]["ci95_low"], row["bootstrap"]["MeanLead"]["ci95_high"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a budgeted analyst-review audit from cached prediction exports.")
    parser.add_argument("--cache-subdir", required=True, help="Subdirectory under outputs/results/audits that stores cached seed exports.")
    parser.add_argument("--benchmark-caption", required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.02)
    parser.add_argument("--tag", required=True, help="Output stem for JSON/TEX artifacts.")
    parser.add_argument("--table-label", required=True, help="LaTeX label stem for the summary table.")
    parser.add_argument("--pairwise-table-label", required=True, help="LaTeX label stem for the pairwise table.")
    parser.add_argument("--tracer-spec", required=True, help="EXPERIMENT::LABEL for TRACER.")
    parser.add_argument("--baseline-spec", action="append", default=[], help="EXPERIMENT::LABEL for each baseline.")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    cache_dir = ROOT / "outputs" / "results" / "audits" / args.cache_subdir
    tracer_experiment, tracer_label = _parse_spec(args.tracer_spec)
    tracer_payloads = _load_payloads(cache_dir, tracer_experiment, tracer_label)

    methods: list[dict[str, Any]] = []
    method_payloads: dict[str, list[dict[str, Any]]] = {tracer_label: tracer_payloads}
    for experiment, label in [_parse_spec(spec) for spec in args.baseline_spec]:
        method_payloads[label] = _load_payloads(cache_dir, experiment, label)

    for label, payloads in method_payloads.items():
        metrics_per_seed = []
        for payload in payloads:
            pred = payload["predictions"]
            metrics_per_seed.append(
                _budget_metrics_from_indices(
                    np.asarray(pred["y_true"], dtype=int),
                    np.asarray(pred["y_score"], dtype=float),
                    np.asarray(pred["time_to_escalation"], dtype=float),
                    budget_fraction=float(args.budget_fraction),
                )
            )
        methods.append(
            {
                "display_name": label,
                "mean": {key: float(mean([row[key] for row in metrics_per_seed])) for key in metrics_per_seed[0].keys()},
                "std": {key: float(pstdev([row[key] for row in metrics_per_seed])) for key in metrics_per_seed[0].keys()},
            }
        )

    pairwise_rows: list[dict[str, Any]] = []
    for experiment, label in [_parse_spec(spec) for spec in args.baseline_spec]:
        baseline_payloads = method_payloads[label]
        mean_delta = _pairwise_delta(
            tracer_payloads,
            baseline_payloads,
            budget_fraction=float(args.budget_fraction),
            indices=None,
        )
        bootstrap = _bootstrap_pairwise(
            tracer_payloads,
            baseline_payloads,
            budget_fraction=float(args.budget_fraction),
            num_samples=int(args.bootstrap_samples),
            seed=20260405 + len(pairwise_rows),
        )
        pairwise_rows.append({"baseline": label, "mean_delta": mean_delta, "bootstrap": bootstrap})

    summary = {
        "benchmark_caption": args.benchmark_caption,
        "budget_fraction": float(args.budget_fraction),
        "table_label": args.table_label,
        "pairwise_table_label": args.pairwise_table_label,
        "methods": methods,
        "pairwise": pairwise_rows,
    }
    json_path = ROOT / "outputs" / "results" / f"{args.tag}.json"
    tex_path = ROOT / "figures" / f"tab_{args.tag}.tex"
    pairwise_tex_path = ROOT / "figures" / f"tab_{args.tag}_pairwise.tex"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    tex_path.write_text(build_summary_table(summary), encoding="utf-8")
    pairwise_tex_path.write_text(build_pairwise_table(summary), encoding="utf-8")
    print(json_path)
    print(tex_path)
    print(pairwise_tex_path)


if __name__ == "__main__":
    main()
