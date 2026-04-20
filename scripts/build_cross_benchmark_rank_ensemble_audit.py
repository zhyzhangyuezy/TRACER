from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"

SEEDS_20 = [7, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025]

SPECS = [
    {
        "setting": "ATLASv2 held-out-family",
        "cache": "public_event_significance_seed_exports",
        "adaptive": "r240_tracer_adaptive_event_atlasv2_public_test_event_disjoint_seed{seed}.json",
        "fixed": {
            "DLinear": "r020_dlinear_forecaster_atlasv2_public_event_disjoint_test_event_disjoint_seed{seed}.json",
            "Small-Transformer": "r010_transformer_forecaster_atlasv2_public_event_disjoint_test_event_disjoint_seed{seed}.json",
            "Prefix-Only": "r010_prefix_retrieval_atlasv2_public_event_disjoint_test_event_disjoint_seed{seed}.json",
        },
    },
    {
        "setting": "AIT-ADS chronology",
        "cache": "ait_ads_chronology_significance_seed_exports",
        "adaptive": "r241_tracer_adaptive_ait_ads_public_test_seed{seed}.json",
        "fixed": {
            "DLinear": "r068_dlinear_forecaster_ait_ads_public_test_seed{seed}.json",
            "Small-Transformer": "r070_transformer_forecaster_ait_ads_public_test_seed{seed}.json",
            "Prefix-Only": "r071_prefix_retrieval_ait_ads_public_test_seed{seed}.json",
        },
    },
    {
        "setting": "AIT-ADS scenario-held-out",
        "cache": "ait_ads_event_significance_seed_exports",
        "adaptive": "r242_tracer_adaptive_event_ait_ads_public_test_event_disjoint_seed{seed}.json",
        "fixed": {
            "DLinear": "r068_dlinear_forecaster_ait_ads_public_test_event_disjoint_seed{seed}.json",
            "TCN": "r069_tcn_forecaster_ait_ads_public_test_event_disjoint_seed{seed}.json",
            "Small-Transformer": "r070_transformer_forecaster_ait_ads_public_test_event_disjoint_seed{seed}.json",
            "Prefix-Only": "r071_prefix_retrieval_ait_ads_public_test_event_disjoint_seed{seed}.json",
        },
    },
    {
        "setting": "CAM-LDS chronology",
        "cache": "synthetic_cam_lds_chronology_significance",
        "adaptive": "r266_tracer_adaptive_chronology_synthetic_cam_lds_controlled_test_seed{seed}.json",
        "fixed": {
            "TCN": "r259_tcn_forecaster_synthetic_cam_lds_controlled_test_seed{seed}.json",
            "Small-Transformer": "r260_transformer_forecaster_synthetic_cam_lds_controlled_test_seed{seed}.json",
            "Prefix-Only": "r262_prefix_retrieval_synthetic_cam_lds_controlled_test_seed{seed}.json",
            "Campaign-Mem": "r263_campaign_mem_synthetic_cam_lds_controlled_test_seed{seed}.json",
        },
    },
    {
        "setting": "CAM-LDS event-held-out",
        "cache": "synthetic_cam_lds_event_significance",
        "adaptive": "r267_tracer_adaptive_event_synthetic_cam_lds_controlled_test_event_disjoint_seed{seed}.json",
        "fixed": {
            "TCN": "r259_tcn_forecaster_synthetic_cam_lds_controlled_test_event_disjoint_seed{seed}.json",
            "Small-Transformer": "r260_transformer_forecaster_synthetic_cam_lds_controlled_test_event_disjoint_seed{seed}.json",
            "Prefix-Only": "r262_prefix_retrieval_synthetic_cam_lds_controlled_test_event_disjoint_seed{seed}.json",
            "Campaign-Mem": "r263_campaign_mem_synthetic_cam_lds_controlled_test_event_disjoint_seed{seed}.json",
        },
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _predictions(payload: dict[str, Any]) -> dict[str, Any]:
    if "predictions" in payload:
        return payload["predictions"]
    raise KeyError("prediction export is missing 'predictions'")


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _rank_average(scores: list[np.ndarray]) -> np.ndarray:
    ranks = []
    for score in scores:
        order = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort").astype(np.float64)
        ranks.append(order / max(score.shape[0] - 1, 1))
    return np.vstack(ranks).mean(axis=0)


def _audit_spec(spec: dict[str, Any]) -> dict[str, Any]:
    cache = RESULT_DIR / "audits" / str(spec["cache"])
    rows = []
    adaptive_values = []
    rank_values = []
    consensus_values = []
    uniform_values = []
    oracle_values = []
    best_fixed_counts = {label: 0 for label in spec["fixed"]}
    for seed in SEEDS_20:
        y_true: np.ndarray | None = None
        fixed_scores = []
        fixed_auprc = {}
        for label, template in spec["fixed"].items():
            payload = _load_json(cache / str(template).format(seed=seed))
            pred = _predictions(payload)
            current_y = np.asarray(pred["y_true"], dtype=int)
            score = np.asarray(pred["y_score"], dtype=np.float64)
            if y_true is None:
                y_true = current_y
            elif not np.array_equal(y_true, current_y):
                raise ValueError(f"Mismatched y_true in {spec['setting']} seed {seed} for {label}")
            fixed_scores.append(score)
            fixed_auprc[label] = _auprc(current_y, score)

        adaptive_payload = _load_json(cache / str(spec["adaptive"]).format(seed=seed))
        adaptive_pred = _predictions(adaptive_payload)
        adaptive_y = np.asarray(adaptive_pred["y_true"], dtype=int)
        adaptive_score = np.asarray(adaptive_pred["y_score"], dtype=np.float64)
        if y_true is None or not np.array_equal(y_true, adaptive_y):
            raise ValueError(f"Mismatched y_true in {spec['setting']} adaptive seed {seed}")

        adaptive_auprc = _auprc(adaptive_y, adaptive_score)
        rank_auprc = _auprc(y_true, _rank_average(fixed_scores))
        consensus_auprc = _auprc(y_true, _rank_average([*fixed_scores, adaptive_score]))
        uniform_auprc = _auprc(y_true, np.vstack(fixed_scores).mean(axis=0))
        oracle_label = max(fixed_auprc, key=fixed_auprc.get)
        oracle_auprc = fixed_auprc[oracle_label]
        best_fixed_counts[oracle_label] += 1
        adaptive_values.append(adaptive_auprc)
        rank_values.append(rank_auprc)
        consensus_values.append(consensus_auprc)
        uniform_values.append(uniform_auprc)
        oracle_values.append(oracle_auprc)
        rows.append(
            {
                "seed": seed,
                "adaptive": adaptive_auprc,
                "rank_ensemble": rank_auprc,
                "rank_plus_adaptive": consensus_auprc,
                "uniform_ensemble": uniform_auprc,
                "fixed_oracle": oracle_auprc,
                "fixed_oracle_label": oracle_label,
                "fixed": fixed_auprc,
            }
        )

    return {
        "setting": spec["setting"],
        "fixed_families": list(spec["fixed"].keys()),
        "rows": rows,
        "summary": {
            "adaptive_mean": float(mean(adaptive_values)),
            "adaptive_std": float(pstdev(adaptive_values)),
            "rank_ensemble_mean": float(mean(rank_values)),
            "rank_ensemble_std": float(pstdev(rank_values)),
            "rank_plus_adaptive_mean": float(mean(consensus_values)),
            "rank_plus_adaptive_std": float(pstdev(consensus_values)),
            "uniform_ensemble_mean": float(mean(uniform_values)),
            "uniform_ensemble_std": float(pstdev(uniform_values)),
            "fixed_oracle_mean": float(mean(oracle_values)),
            "fixed_oracle_std": float(pstdev(oracle_values)),
            "rank_minus_adaptive": float(mean(rank_values) - mean(adaptive_values)),
            "rank_plus_adaptive_minus_fixed_rank": float(mean(consensus_values) - mean(rank_values)),
            "rank_plus_adaptive_minus_adaptive": float(mean(consensus_values) - mean(adaptive_values)),
            "uniform_minus_adaptive": float(mean(uniform_values) - mean(adaptive_values)),
            "oracle_minus_adaptive": float(mean(oracle_values) - mean(adaptive_values)),
            "rank_beats_adaptive_seeds": int(sum(rank > adaptive for rank, adaptive in zip(rank_values, adaptive_values))),
            "rank_plus_adaptive_beats_fixed_rank_seeds": int(sum(consensus > rank for consensus, rank in zip(consensus_values, rank_values))),
            "uniform_beats_adaptive_seeds": int(sum(uniform > adaptive for uniform, adaptive in zip(uniform_values, adaptive_values))),
            "oracle_beats_adaptive_seeds": int(sum(oracle > adaptive for oracle, adaptive in zip(oracle_values, adaptive_values))),
            "best_fixed_counts": best_fixed_counts,
        },
    }


def build_audit() -> dict[str, Any]:
    rows = [_audit_spec(spec) for spec in SPECS]
    return {
        "audit": "Cross-benchmark fixed-family rank-ensemble stress audit",
        "seeds": SEEDS_20,
        "settings": rows,
        "note": "The rank ensemble is a calibration-free average of fixed-family prediction ranks. It is evaluated only from prediction exports and is not used to tune the released deterministic adaptive policy.",
    }


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Cross-benchmark rank-ensemble stress audit",
        "",
        audit["note"],
        "",
        "| Setting | Adaptive | Rank fixed | Delta | Rank+adaptive | Gain vs rank | Uniform | Single-family oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in audit["settings"]:
        s = item["summary"]
        lines.append(
            "| {setting} | {adaptive:.3f} +/- {adaptive_std:.3f} | {rank:.3f} +/- {rank_std:.3f} | {delta:+.3f} | {consensus:.3f} +/- {consensus_std:.3f} | {consensus_delta:+.3f} | {uniform:.3f} | {oracle:.3f} |".format(
                setting=item["setting"],
                adaptive=s["adaptive_mean"],
                adaptive_std=s["adaptive_std"],
                rank=s["rank_ensemble_mean"],
                rank_std=s["rank_ensemble_std"],
                delta=s["rank_minus_adaptive"],
                consensus=s["rank_plus_adaptive_mean"],
                consensus_std=s["rank_plus_adaptive_std"],
                consensus_delta=s["rank_plus_adaptive_minus_fixed_rank"],
                uniform=s["uniform_ensemble_mean"],
                oracle=s["fixed_oracle_mean"],
            )
        )
    (RESULT_DIR / "cross_benchmark_rank_ensemble_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(audit: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Cross-benchmark stress test for transparent rank-consensus deployment. The fixed rank ensemble averages only fixed-family prediction ranks; the rank+adaptive variant adds the released TRACER adaptive score as one additional ranked signal. Both are evaluated from the same 20-seed prediction exports without held-out tuning.}",
        r"\label{tab:cross-benchmark-rank-ensemble}",
        r"\maxtablewidth{",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Setting & Adaptive & Rank fixed & $\Delta$ & Rank+adaptive & Gain & Single-family oracle \\",
        r"\midrule",
    ]
    for item in audit["settings"]:
        s = item["summary"]
        lines.append(
            "{setting} & ${adaptive:.3f}\\pm{adaptive_std:.3f}$ & ${rank:.3f}\\pm{rank_std:.3f}$ & ${delta:+.3f}$ & ${consensus:.3f}\\pm{consensus_std:.3f}$ & ${gain:+.3f}$ & ${oracle:.3f}$ \\\\".format(
                setting=str(item["setting"]).replace("_", r"\_"),
                adaptive=s["adaptive_mean"],
                adaptive_std=s["adaptive_std"],
                rank=s["rank_ensemble_mean"],
                rank_std=s["rank_ensemble_std"],
                delta=s["rank_minus_adaptive"],
                consensus=s["rank_plus_adaptive_mean"],
                consensus_std=s["rank_plus_adaptive_std"],
                gain=s["rank_plus_adaptive_minus_fixed_rank"],
                oracle=s["fixed_oracle_mean"],
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    (FIG_DIR / "tab_cross_benchmark_rank_ensemble_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_audit()
    (RESULT_DIR / "cross_benchmark_rank_ensemble_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit)
    write_latex(audit)
    print("Wrote outputs/results/cross_benchmark_rank_ensemble_audit.json")
    print("Wrote figures/tab_cross_benchmark_rank_ensemble_audit.tex")


if __name__ == "__main__":
    main()
