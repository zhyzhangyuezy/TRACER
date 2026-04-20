from __future__ import annotations

import argparse
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
from campaign_mem.metrics import auprc
from campaign_mem.utils import load_yaml, save_json
import numpy as np


DEFAULT_TRACER_SPEC = ("configs/experiments/r240_tracer_adaptive_event_atlasv2_public.yaml", "TRACER")
BASELINE_SPECS = [
    ("configs/experiments/r010_transformer_forecaster_atlasv2_public_event_disjoint.yaml", "Small-Transformer-Forecaster"),
    ("configs/experiments/r018_lstm_forecaster_atlasv2_public_event_disjoint.yaml", "LSTM-Forecaster"),
    ("configs/experiments/r020_dlinear_forecaster_atlasv2_public_event_disjoint.yaml", "DLinear-Forecaster"),
    ("configs/experiments/r010_prefix_retrieval_atlasv2_public_event_disjoint.yaml", "Prefix-Only-Retrieval + Fusion"),
]
DEFAULT_SEEDS = [7, 13, 21, 34, 55, 89, 144, 233, 377, 610]
TRACER_NAME = "TRACER"
BOOTSTRAP_SAMPLES = 5000


def _pairwise_outcome(a: float, b: float, eps: float = 1e-12) -> str:
    if a > b + eps:
        return "W"
    if b > a + eps:
        return "L"
    return "T"


def _sample_indices_by_incident(incident_id: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique_incidents = np.unique(incident_id.astype(str))
    sampled = rng.choice(unique_incidents, size=unique_incidents.size, replace=True)
    chunks = [np.flatnonzero(incident_id.astype(str) == value) for value in sampled]
    return np.concatenate(chunks, axis=0)


def _bootstrap_delta(
    tracer_scores: list[np.ndarray],
    baseline_scores: list[np.ndarray],
    y_true: np.ndarray,
    incident_id: np.ndarray,
    *,
    num_samples: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    while len(deltas) < num_samples:
        indices = _sample_indices_by_incident(incident_id, rng)
        if np.unique(y_true[indices]).size < 2:
            continue
        sampled = [
            auprc(y_true[indices], tracer_scores[idx][indices]) - auprc(y_true[indices], baseline_scores[idx][indices])
            for idx in range(len(tracer_scores))
        ]
        deltas.append(float(np.mean(sampled)))
    arr = np.asarray(deltas, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "positive_mass": float((arr > 0.0).mean()),
        "nonnegative_mass": float((arr >= 0.0).mean()),
    }


def _leave_one_incident_out(
    tracer_scores: list[np.ndarray],
    baseline_scores: list[np.ndarray],
    y_true: np.ndarray,
    incident_id: np.ndarray,
) -> dict[str, Any]:
    unique_incidents = np.unique(incident_id.astype(str))
    rows: list[dict[str, Any]] = []
    for held_out in unique_incidents:
        mask = incident_id.astype(str) != held_out
        if np.unique(y_true[mask]).size < 2:
            continue
        delta = float(
            np.mean(
                [
                    auprc(y_true[mask], tracer_scores[idx][mask]) - auprc(y_true[mask], baseline_scores[idx][mask])
                    for idx in range(len(tracer_scores))
                ]
            )
        )
        rows.append({"held_out_incident": str(held_out), "mean_delta_auprc": delta, "windows_kept": int(mask.sum())})
    if not rows:
        return {"rows": [], "positive_count": 0, "total": 0, "min_delta_auprc": None, "max_delta_auprc": None}
    deltas = np.asarray([row["mean_delta_auprc"] for row in rows], dtype=float)
    worst_row = min(rows, key=lambda row: row["mean_delta_auprc"])
    return {
        "rows": rows,
        "positive_count": int((deltas > 0.0).sum()),
        "nonnegative_count": int((deltas >= 0.0).sum()),
        "total": len(rows),
        "min_delta_auprc": float(deltas.min()),
        "max_delta_auprc": float(deltas.max()),
        "worst_case_incident": str(worst_row["held_out_incident"]),
    }


def _load_or_rerun(config_path: Path, display_name: str, seed: int, cache_dir: Path, force: bool) -> dict[str, Any]:
    config = load_yaml(config_path)
    base_name = str(config["experiment_name"])
    cache_path = cache_dir / f"{base_name}_seed{seed}.json"
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["display_name"] = display_name
        return payload

    config["seed"] = seed
    config["experiment_name"] = f"{base_name}_audit_seed{seed}"
    artifacts = train_best_model(config)
    export = predict_split(artifacts, "test_event_disjoint")
    payload = {
        "display_name": display_name,
        "base_experiment_name": base_name,
        "seed": seed,
        "metrics": export["metrics"],
        "threshold": export["threshold"],
        "policy_info": export.get("policy_info"),
        "predictions": export["predictions"],
    }
    save_json(cache_path, payload)
    return payload


def _format_pm(mean_value: float, std_value: float, decimals: int = 3) -> str:
    return f"${mean_value:.{decimals}f} \\pm {std_value:.{decimals}f}$"


def _format_ci(low: float, high: float, decimals: int = 3) -> str:
    return f"$[{low:.{decimals}f}, {high:.{decimals}f}]$"


def build_table_tex(summary: dict[str, Any]) -> str:
    tracer_label = str(summary.get("tracer_label", TRACER_NAME))
    tracer_summary = summary["models"][tracer_label]
    num_seeds = len(summary.get("seeds", []))
    loio_total = 0
    if summary.get("pairwise"):
        first_pairwise = next(iter(summary["pairwise"].values()))
        loio_total = int(first_pairwise.get("leave_one_incident_out", {}).get("total", 0))
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Expanded held-out-family audit on the primary ATLASv2 split. We rerun the strongest event-disjoint contenders over {num_seeds} seeds and then evaluate the frozen TRACER event route's AUPRC margin with paired incident-block bootstrap on the 83-window, 6-positive held-out-family test set. The leave-one-incident-out column reports how often the mean delta remains positive after removing one held-out incident at a time.}}",
        r"\label{tab:atlasv2-event-significance}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        rf"Baseline & TRACER AUPRC & Baseline AUPRC & Seed W/T/L & Mean $\Delta$AUPRC & 95\% bootstrap CI & $\Pr(\Delta > 0)$ & Leave-one-incident-out $+$ / {max(loio_total, 1)} \\",
        r"\midrule",
    ]
    for baseline_name, row in summary["pairwise"].items():
        baseline_summary = summary["models"][baseline_name]
        lines.append(
            baseline_name
            + " & "
            + _format_pm(float(tracer_summary["mean_auprc"]), float(tracer_summary["std_auprc"]))
            + " & "
            + _format_pm(float(baseline_summary["mean_auprc"]), float(baseline_summary["std_auprc"]))
            + " & "
            + f"{row['seed_wins']} / {row['seed_ties']} / {row['seed_losses']}"
            + " & "
            + f"${row['mean_delta_auprc']:.3f}$"
            + " & "
            + _format_ci(float(row["bootstrap"]["ci95_low"]), float(row["bootstrap"]["ci95_high"]))
            + " & "
            + f"${100.0 * float(row['bootstrap']['positive_mass']):.1f}\\%$"
            + " & "
            + f"{row['leave_one_incident_out']['positive_count']} / {row['leave_one_incident_out']['total']}"
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paired-bootstrap significance audit for ATLASv2 held-out-family results.")
    parser.add_argument("--force", action="store_true", help="Ignore cached per-seed exports and rerun training.")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--tracer-config", default=DEFAULT_TRACER_SPEC[0], help="Config path for the candidate TRACER event route.")
    parser.add_argument("--tracer-label", default=DEFAULT_TRACER_SPEC[1], help="Display label for the candidate TRACER row.")
    parser.add_argument("--tag", default="public_event_significance_audit", help="Output stem for JSON/TEX artifacts.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Seed list for the expanded audit.")
    args = parser.parse_args()

    specs = [(args.tracer_config, args.tracer_label), *BASELINE_SPECS]
    cache_dir = ROOT / "outputs" / "results" / "audits" / "public_event_significance_seed_exports"
    payloads: dict[str, list[dict[str, Any]]] = {display_name: [] for _, display_name in specs}
    seeds = list(args.seeds)

    for config_path_str, display_name in specs:
        config_path = ROOT / config_path_str
        for seed in seeds:
            payloads[display_name].append(_load_or_rerun(config_path, display_name, seed, cache_dir, args.force))

    model_summary: dict[str, Any] = {}
    for display_name, rows in payloads.items():
        auprc_values = [float(row["metrics"]["AUPRC"]) for row in rows]
        model_summary[display_name] = {
            "seeds": [int(row["seed"]) for row in rows],
            "base_experiment_name": str(rows[0]["base_experiment_name"]),
            "mean_auprc": float(mean(auprc_values)),
            "std_auprc": float(pstdev(auprc_values)),
            "seed_metrics": {
                str(row["seed"]): {"AUPRC": float(row["metrics"]["AUPRC"]), "LeadTime@P80": float(row["metrics"].get("LeadTime@P80", 0.0))}
                for row in rows
            },
        }

    tracer_rows = payloads[args.tracer_label]
    y_true = np.asarray(tracer_rows[0]["predictions"]["y_true"], dtype=int)
    incident_id = np.asarray(tracer_rows[0]["predictions"]["incident_id"], dtype=str)
    tracer_scores = [np.asarray(row["predictions"]["y_score"], dtype=float) for row in tracer_rows]

    pairwise: dict[str, Any] = {}
    for baseline_name, baseline_rows in payloads.items():
        if baseline_name == args.tracer_label:
            continue
        baseline_scores = [np.asarray(row["predictions"]["y_score"], dtype=float) for row in baseline_rows]
        tracer_auprc = [float(row["metrics"]["AUPRC"]) for row in tracer_rows]
        baseline_auprc = [float(row["metrics"]["AUPRC"]) for row in baseline_rows]
        outcomes = [_pairwise_outcome(tracer_auprc[idx], baseline_auprc[idx]) for idx in range(len(seeds))]
        bootstrap = _bootstrap_delta(
            tracer_scores,
            baseline_scores,
            y_true,
            incident_id,
            num_samples=int(args.bootstrap_samples),
            seed=20260404 + len(pairwise),
        )
        loio = _leave_one_incident_out(tracer_scores, baseline_scores, y_true, incident_id)
        deltas = [tracer_auprc[idx] - baseline_auprc[idx] for idx in range(len(seeds))]
        pairwise[baseline_name] = {
            "seed_wins": int(sum(outcome == "W" for outcome in outcomes)),
            "seed_ties": int(sum(outcome == "T" for outcome in outcomes)),
            "seed_losses": int(sum(outcome == "L" for outcome in outcomes)),
            "mean_delta_auprc": float(mean(deltas)),
            "std_delta_auprc": float(pstdev(deltas)),
            "seedwise_delta_auprc": {str(seed): float(deltas[idx]) for idx, seed in enumerate(seeds)},
            "bootstrap": bootstrap,
            "leave_one_incident_out": loio,
        }

    summary = {
        "split": "test_event_disjoint",
        "tracer_label": args.tracer_label,
        "tracer_config": args.tracer_config,
        "seeds": seeds,
        "bootstrap_samples": int(args.bootstrap_samples),
        "models": model_summary,
        "pairwise": pairwise,
    }
    json_path = ROOT / "outputs" / "results" / f"{args.tag}.json"
    tex_path = ROOT / "figures" / f"tab_{args.tag}.tex"
    save_json(json_path, summary)
    tex_path.write_text(build_table_tex(summary), encoding="utf-8")
    print(json_path)
    print(tex_path)


if __name__ == "__main__":
    main()
