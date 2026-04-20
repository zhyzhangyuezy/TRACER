from __future__ import annotations

import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.training import run_experiment


DATASET_DIR = "data/splunk_attack_data_public_probe"
SEEDS = (7, 13, 21)


METHODS: dict[str, dict[str, Any]] = {
    "TRACER adaptive": {
        "model": {
            "type": "campaign_mem_decomp_modular",
            "retrieval_encoder": "transformer",
            "stable_encoder": "dlinear",
            "shock_encoder": "patchtst",
            "hidden_dim": 64,
            "embedding_dim": 64,
            "top_k": 5,
            "use_auxiliary": True,
            "use_contrastive": True,
            "use_hard_negatives": True,
        },
        "training": {
            "epochs": 10,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
        },
        "auto_component_policy": {"name": "tracer_adaptive", "objective": "event"},
    },
    "TRACER core": {
        "model": {
            "type": "campaign_mem_decomp_modular",
            "retrieval_encoder": "transformer",
            "stable_encoder": "dlinear",
            "shock_encoder": "patchtst",
            "hidden_dim": 64,
            "embedding_dim": 64,
            "top_k": 5,
            "similarity_temperature": 0.15,
            "delta_scale": 0.18,
            "trend_kernel": 3,
            "use_auxiliary": True,
            "use_contrastive": True,
            "use_hard_negatives": True,
            "use_utility": False,
            "use_abstention": True,
            "use_uncertainty_gate": False,
            "use_shift_gate": True,
            "use_aggressive_gate": True,
            "aggressive_route_on_delta": True,
        },
        "training": {
            "epochs": 10,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "auxiliary_weight": 0.25,
            "contrastive_weight": 0.15,
            "calibration_penalty_weight": 0.03,
            "selector_weight": 0.06,
            "abstention_weight": 0.04,
            "shift_weight": 0.04,
            "aggressive_weight": 0.03,
            "main_pos_weight": "auto_sqrt",
            "aux_pos_weight": "auto_sqrt",
            "grad_clip": 0.9,
            "warmup_ratio": 0.05,
            "warmdown_ratio": 0.25,
            "final_lr_frac": 0.25,
            "ema_decay": 0.98,
            "model_selection_mode": "balanced",
            "model_selection_start_epoch": 4,
            "checkpoint_average_top_k": 2,
        },
    },
    "DLinear": {
        "model": {
            "type": "campaign_mem",
            "encoder": "dlinear",
            "hidden_dim": 64,
            "embedding_dim": 64,
            "top_k": 5,
            "use_auxiliary": True,
            "use_contrastive": True,
            "use_hard_negatives": True,
            "use_utility": False,
        },
        "training": {
            "epochs": 10,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "auxiliary_weight": 0.25,
            "contrastive_weight": 0.15,
            "main_pos_weight": "auto_sqrt",
            "aux_pos_weight": "auto_sqrt",
            "model_selection_mode": "balanced",
            "model_selection_start_epoch": 4,
            "checkpoint_average_top_k": 2,
        },
    },
    "Prefix-only": {
        "model": {
            "type": "prefix_retrieval",
            "encoder": "transformer",
            "hidden_dim": 64,
            "embedding_dim": 64,
            "top_k": 5,
            "use_auxiliary": True,
            "use_contrastive": False,
            "use_hard_negatives": False,
            "use_utility": False,
        },
        "training": {
            "epochs": 10,
            "batch_size": 64,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "auxiliary_weight": 0.25,
            "main_pos_weight": "auto_sqrt",
            "aux_pos_weight": "auto_sqrt",
            "model_selection_mode": "balanced",
            "model_selection_start_epoch": 4,
            "checkpoint_average_top_k": 2,
        },
    },
    "Pure kNN": {
        "model": {
            "type": "pure_knn",
            "top_k": 5,
        },
        "training": {
            "batch_size": 64,
        },
    },
}


def _base_config(method_name: str, seed: int, spec: dict[str, Any]) -> dict[str, Any]:
    slug = method_name.lower().replace(" ", "_").replace("-", "_")
    config = {
        "experiment_name": f"r400_splunk_attack_data_probe_{slug}_seed{seed}",
        "seed": seed,
        "device": "cuda",
        "data": {"dataset_dir": DATASET_DIR},
        "model": deepcopy(spec["model"]),
        "training": deepcopy(spec.get("training", {})),
        "metrics": {"target_precision": 0.8, "analog_fidelity_distance_threshold": 0.45},
        "output": {"dir": "outputs/results/splunk_attack_data_probe_runs"},
    }
    if "auto_component_policy" in spec:
        config["auto_component_policy"] = deepcopy(spec["auto_component_policy"])
    return config


def _metric(result: dict[str, Any], metric: str, split: str = "test") -> float:
    return float(result.get(split, {}).get(metric, 0.0))


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)) if values else 0.0,
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _fmt(stats: dict[str, float], digits: int = 3) -> str:
    return f"{stats['mean']:.{digits}f} $\\pm$ {stats['std']:.{digits}f}"


def run_audit() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for method_name, spec in METHODS.items():
        seeds = (SEEDS[0],) if spec["model"]["type"] == "pure_knn" else SEEDS
        for seed in seeds:
            config = _base_config(method_name, seed, spec)
            print(f"Running {method_name} seed={seed}")
            result = run_experiment(config)
            results.append({"method": method_name, "seed": seed, "result": result})

    summary_rows: list[dict[str, Any]] = []
    for method_name in METHODS:
        method_results = [item["result"] for item in results if item["method"] == method_name]
        row = {
            "method": method_name,
            "seeds": [item["seed"] for item in results if item["method"] == method_name],
            "test_auprc": _mean_std([_metric(result, "AUPRC") for result in method_results]),
            "test_auroc": _mean_std([_metric(result, "AUROC") for result in method_results]),
            "test_brier": _mean_std([_metric(result, "Brier") for result in method_results]),
            "test_af5": _mean_std([_metric(result, "Analog-Fidelity@5") for result in method_results]),
            "event_auprc": _mean_std([_metric(result, "AUPRC", "test_event_disjoint") for result in method_results]),
        }
        summary_rows.append(row)
    best = max(summary_rows, key=lambda row: row["test_auprc"]["mean"])
    summary = {
        "audit": "splunk_attack_data_public_probe",
        "dataset_dir": DATASET_DIR,
        "seeds": list(SEEDS),
        "note": (
            "Independent public raw-log probe from selected Splunk Attack Data files. "
            "The probe uses deterministic MITRE-stage/keyword labels and incident-disjoint event-bucket windows; "
            "it is a stress test rather than a replacement for analyst-labeled triage data."
        ),
        "rows": summary_rows,
        "best_by_test_auprc": best["method"],
    }
    return {"summary": summary, "runs": results}


def write_outputs(payload: dict[str, Any]) -> None:
    output_json = Path("outputs/results/splunk_attack_data_probe_audit.json")
    output_md = Path("outputs/results/splunk_attack_data_probe_audit.md")
    output_tex = Path("figures/tab_splunk_attack_data_probe.tex")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = payload["summary"]["rows"]
    md_lines = [
        "# Splunk Attack Data public probe audit",
        "",
        payload["summary"]["note"],
        "",
        "| Method | Seeds | Test AUPRC | Event AUPRC | AUROC | Brier | AF@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {method} | {seeds} | {auprc} | {event_auprc} | {auroc} | {brier} | {af5} |".format(
                method=row["method"],
                seeds=",".join(str(seed) for seed in row["seeds"]),
                auprc=f"{row['test_auprc']['mean']:.3f} +/- {row['test_auprc']['std']:.3f}",
                event_auprc=f"{row['event_auprc']['mean']:.3f} +/- {row['event_auprc']['std']:.3f}",
                auroc=f"{row['test_auroc']['mean']:.3f} +/- {row['test_auroc']['std']:.3f}",
                brier=f"{row['test_brier']['mean']:.3f} +/- {row['test_brier']['std']:.3f}",
                af5=f"{row['test_af5']['mean']:.1f} +/- {row['test_af5']['std']:.1f}",
            )
        )
    output_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Splunk Attack Data public raw-log probe. Selected public logs are projected into incident-disjoint event-bucket windows using deterministic MITRE-stage/keyword labels. Values are mean$\pm$std over three seeds except Pure kNN, which is deterministic. This is an external stress test rather than analyst-labeled triage evidence.}",
        r"\label{tab:splunk-attack-data-probe}",
        r"\maxtablewidth{\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Test AUPRC & Event AUPRC & AUROC & Brier & AF@5 \\",
        r"\midrule",
    ]
    best_method = payload["summary"]["best_by_test_auprc"]
    for row in rows:
        label = row["method"].replace("_", r"\_")
        auprc = _fmt(row["test_auprc"])
        if row["method"] == best_method:
            auprc = r"\textbf{" + auprc + "}"
        tex_lines.append(
            rf"{label} & {auprc} & {_fmt(row['event_auprc'])} & {_fmt(row['test_auroc'])} & {_fmt(row['test_brier'])} & {_fmt(row['test_af5'], 1)} \\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table}"])
    output_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main() -> None:
    payload = run_audit()
    write_outputs(payload)
    best = payload["summary"]["best_by_test_auprc"]
    print(f"Saved Splunk Attack Data probe audit. Best by test AUPRC: {best}")


if __name__ == "__main__":
    main()
