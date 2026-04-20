from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"
SEEDS = [7, 13, 21]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _adaptive_payload(fold_id: str) -> dict[str, Any]:
    return _load_json(RESULT_DIR / f"r300_lopo_{fold_id}_adaptive_seed{SEEDS[0]}.json")


def _route_from_stats(stats: dict[str, Any], objective: str) -> str:
    positive_rate = float(stats.get("positive_rate", 0.0))
    family_count = int(stats.get("family_count", 0))
    positive_family_count = int(stats.get("positive_family_count", 0))
    diff2_abs_mean = float(stats.get("diff2_abs_mean", 0.0))
    peak_ratio = float(stats.get("peak_ratio", 0.0))
    if positive_rate <= 0.02 and int(stats.get("positive_count", 0)) <= 10:
        return "cold_start_sparse"
    if positive_rate <= 0.005:
        return "extreme_sparse"
    if objective in {"chronology", "chrono", "test"} and positive_family_count <= 2 and diff2_abs_mean >= 0.28 and peak_ratio >= 4.0:
        return "sparse_diverse_chrono_spiky"
    if positive_rate >= 0.05 and family_count <= 6:
        return "dense_low_diversity_event" if objective == "event_disjoint" else "dense_low_diversity"
    if positive_rate >= 0.35:
        return "simple_dense"
    return "sparse_diverse"


def build_audit() -> dict[str, Any]:
    folds = _load_json(RESULT_DIR / "atlasv2_lopo_family_folds.json")["folds"]
    rows = []
    cold_rate_margins = []
    cold_count_margins = []
    extreme_margins = []
    dense_rate_margins = []
    dense_family_margins = []
    route_flip_counts = {0.8: 0, 0.9: 0, 1.1: 0, 1.2: 0}
    for fold in folds:
        fold_id = str(fold["fold_id"])
        payload = _adaptive_payload(fold_id)
        policy = payload["auto_component_policy"]
        stats = policy["train_stats"]
        objective = str(policy["objective"])
        positive_rate = float(stats["positive_rate"])
        positive_count = int(stats["positive_count"])
        family_count = int(stats["family_count"])
        cold_rate_margin = positive_rate - 0.02
        cold_count_margin = positive_count - 10
        extreme_margin = positive_rate - 0.005
        dense_rate_margin = 0.05 - positive_rate
        dense_family_margin = family_count - 6
        cold_rate_margins.append(cold_rate_margin)
        cold_count_margins.append(cold_count_margin)
        extreme_margins.append(extreme_margin)
        dense_rate_margins.append(dense_rate_margin)
        dense_family_margins.append(dense_family_margin)
        base_route = str(policy["regime"])
        for scale in route_flip_counts:
            perturbed_stats = dict(stats)
            # Perturb scalar route thresholds by scaling equivalent feature values in the conservative direction.
            # For the LOPO event folds only cold/extreme/dense thresholds can bind.
            cold_threshold = 0.02 * scale
            extreme_threshold = 0.005 * scale
            dense_threshold = 0.05 * scale
            route = "sparse_diverse"
            if positive_rate <= cold_threshold and positive_count <= 10:
                route = "cold_start_sparse"
            elif positive_rate <= extreme_threshold:
                route = "extreme_sparse"
            elif positive_rate >= dense_threshold and family_count <= 6:
                route = "dense_low_diversity_event"
            if route != base_route:
                route_flip_counts[scale] += 1
        rows.append(
            {
                "fold_id": fold_id,
                "test_family": fold["test_family"],
                "regime": base_route,
                "positive_rate": positive_rate,
                "positive_count": positive_count,
                "family_count": family_count,
                "cold_rate_margin": cold_rate_margin,
                "cold_count_margin": cold_count_margin,
                "extreme_margin": extreme_margin,
                "dense_rate_margin": dense_rate_margin,
                "dense_family_margin": dense_family_margin,
            }
        )

    summary = {
        "fold_count": len(rows),
        "regimes": sorted(set(row["regime"] for row in rows)),
        "min_cold_rate_margin": float(min(cold_rate_margins)),
        "min_cold_count_margin": int(min(cold_count_margins)),
        "min_extreme_margin": float(min(extreme_margins)),
        "min_dense_rate_margin": float(min(dense_rate_margins)),
        "min_dense_family_margin": float(min(dense_family_margins)),
        "route_flips_under_threshold_scale": {str(key): int(value) for key, value in route_flip_counts.items()},
    }
    return {
        "audit": "LOPO route-threshold sensitivity audit",
        "rows": rows,
        "summary": summary,
        "note": "All processed-window LOPO event folds resolve to sparse_diverse. The audit reports margins to the cold-start, extreme-sparse, and dense-low-diversity gates and counts route flips under simple scalar threshold perturbations.",
    }


def write_markdown(audit: dict[str, Any]) -> None:
    summary = audit["summary"]
    lines = [
        "# LOPO route-threshold sensitivity audit",
        "",
        audit["note"],
        "",
        f"Regimes: {', '.join(summary['regimes'])}.",
        f"Minimum margin to cold-start rate gate: {summary['min_cold_rate_margin']:.4f}.",
        f"Minimum margin to cold-start count gate: {summary['min_cold_count_margin']}.",
        f"Minimum margin to extreme-sparse rate gate: {summary['min_extreme_margin']:.4f}.",
        f"Minimum margin to dense rate gate: {summary['min_dense_rate_margin']:.4f}.",
        f"Minimum family-count margin away from dense gate: {summary['min_dense_family_margin']:.1f}.",
        "",
        "| Threshold scale | Route flips |",
        "|---:|---:|",
    ]
    for scale, flips in summary["route_flips_under_threshold_scale"].items():
        lines.append(f"| {scale}x | {flips}/{summary['fold_count']} |")
    (RESULT_DIR / "lopo_route_sensitivity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(audit: dict[str, Any]) -> None:
    summary = audit["summary"]
    regime_text = ", ".join(summary["regimes"]).replace("_", r"\_")
    flip_text = ", ".join(
        f"{float(scale):.1f}$\\times$: {flips}/{summary['fold_count']}"
        for scale, flips in summary["route_flips_under_threshold_scale"].items()
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Route-threshold sensitivity on the processed-window ATLASv2 LOPO event folds. All folds resolve to the same sparse-diverse event route; margins report the closest distance to the cold-start, extreme-sparse, and dense-low-diversity gates under the frozen rule ordering.}",
        r"\label{tab:lopo-route-sensitivity}",
        r"\maxtablewidth{",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Check & Value & Interpretation \\",
        r"\midrule",
        f"Resolved regimes & {regime_text} & one route across 7 folds \\\\",
        f"Min cold-start rate margin & ${summary['min_cold_rate_margin']:.4f}$ & positive means rate is above cold gate \\\\",
        f"Min cold-start count margin & ${summary['min_cold_count_margin']}$ & positive means count is above cold gate \\\\",
        f"Min extreme-sparse margin & ${summary['min_extreme_margin']:.4f}$ & positive means no extreme-sparse flip \\\\",
        f"Min dense-rate margin & ${summary['min_dense_rate_margin']:.4f}$ & positive means rate below dense gate \\\\",
        f"Min dense-family margin & ${summary['min_dense_family_margin']:.1f}$ & positive means families exceed dense cap \\\\",
        f"Route flips under threshold scaling & {flip_text} & scalar stress test \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    (FIG_DIR / "tab_lopo_route_sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_audit()
    (RESULT_DIR / "lopo_route_sensitivity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit)
    write_latex(audit)
    print("Wrote outputs/results/lopo_route_sensitivity_audit.json")
    print("Wrote figures/tab_lopo_route_sensitivity.tex")


if __name__ == "__main__":
    main()
