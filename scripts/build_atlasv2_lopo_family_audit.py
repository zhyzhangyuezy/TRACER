from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"

SEEDS = [7, 13, 21]
METRIC = "AUPRC"
SPLIT = "test_event_disjoint"

POLICY_METHODS = [
    ("adaptive", "Adaptive policy"),
    ("dlinear", "DLinear"),
    ("transformer", "Small-Transformer"),
    ("prefix", "Prefix-Only retrieval"),
]

CORE_METHODS = [
    ("core_bounded", "Core bounded"),
    ("core_linear", "Linear correction"),
    ("core_no_route", "No route gates"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _folds() -> list[dict[str, Any]]:
    payload = _load_json(RESULT_DIR / "atlasv2_lopo_family_folds.json")
    return list(payload["folds"])


def _result_path(fold_id: str, method: str, seed: int) -> Path:
    return RESULT_DIR / f"r300_lopo_{fold_id}_{method}_seed{seed}.json"


def _metric_values(fold_id: str, method: str, metric: str = METRIC) -> list[float]:
    values: list[float] = []
    missing: list[str] = []
    for seed in SEEDS:
        path = _result_path(fold_id, method, seed)
        if not path.exists():
            missing.append(path.name)
            continue
        payload = _load_json(path)
        if SPLIT not in payload or metric not in payload[SPLIT]:
            missing.append(f"{path.name}:{SPLIT}.{metric}")
            continue
        values.append(float(payload[SPLIT][metric]))
    if missing:
        raise FileNotFoundError("Missing LOPO audit inputs: " + ", ".join(missing))
    return values


def _hierarchical_ci(values_by_fold: list[list[float]], samples: int = 5000, seed: int = 20260417) -> list[float]:
    rng = np.random.default_rng(seed)
    n_folds = len(values_by_fold)
    draws = np.empty(samples, dtype=np.float64)
    arrays = [np.asarray(values, dtype=np.float64) for values in values_by_fold]
    for i in range(samples):
        chosen_folds = rng.integers(0, n_folds, size=n_folds)
        fold_means = []
        for fold_idx in chosen_folds:
            arr = arrays[int(fold_idx)]
            chosen_seeds = rng.integers(0, len(arr), size=len(arr))
            fold_means.append(float(arr[chosen_seeds].mean()))
        draws[i] = float(np.mean(fold_means))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return [float(lo), float(hi)]


def _summarize_method(folds: list[dict[str, Any]], method: str) -> dict[str, Any]:
    values_by_fold: list[list[float]] = []
    fold_rows = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        values = _metric_values(fold_id, method)
        values_by_fold.append(values)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "test_family": fold["test_family"],
                "mean": float(mean(values)),
                "std": float(pstdev(values)),
                "values": values,
                "test_positives": int(fold["test"]["positives"]),
                "test_windows": int(fold["test"]["size"]),
            }
        )
    fold_means = [row["mean"] for row in fold_rows]
    ci = _hierarchical_ci(values_by_fold)
    return {
        "method": method,
        "macro_mean": float(mean(fold_means)),
        "fold_std": float(pstdev(fold_means)),
        "worst_fold": float(min(fold_means)),
        "best_fold": float(max(fold_means)),
        "hierarchical_ci": ci,
        "folds": fold_rows,
    }


def _paired_delta_summary(folds: list[dict[str, Any]], method: str, baseline: str = "core_bounded") -> dict[str, Any]:
    values_by_fold: list[list[float]] = []
    fold_rows = []
    for fold in folds:
        fold_id = str(fold["fold_id"])
        current = np.asarray(_metric_values(fold_id, method), dtype=np.float64)
        base = np.asarray(_metric_values(fold_id, baseline), dtype=np.float64)
        delta = current - base
        values_by_fold.append(delta.tolist())
        fold_rows.append(
            {
                "fold_id": fold_id,
                "test_family": fold["test_family"],
                "mean_delta": float(delta.mean()),
                "values": delta.tolist(),
            }
        )
    fold_deltas = [row["mean_delta"] for row in fold_rows]
    ci = _hierarchical_ci(values_by_fold)
    return {
        "method": method,
        "baseline": baseline,
        "macro_delta": float(mean(fold_deltas)),
        "worst_delta": float(min(fold_deltas)),
        "best_delta": float(max(fold_deltas)),
        "hierarchical_ci": ci,
        "folds": fold_rows,
    }


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(values.items(), key=lambda item: item[1], reverse=True)
    ranks: dict[str, float] = {}
    position = 1
    index = 0
    while index < len(sorted_items):
        tied = [sorted_items[index]]
        index += 1
        while index < len(sorted_items) and abs(sorted_items[index][1] - tied[0][1]) <= 1e-12:
            tied.append(sorted_items[index])
            index += 1
        avg_rank = (position + position + len(tied) - 1) / 2.0
        for method, _ in tied:
            ranks[method] = avg_rank
        position += len(tied)
    return ranks


def build_summary() -> dict[str, Any]:
    folds = _folds()
    policy = {method: _summarize_method(folds, method) for method, _ in POLICY_METHODS}
    core = {method: _summarize_method(folds, method) for method, _ in CORE_METHODS}
    core_delta = {
        method: _paired_delta_summary(folds, method)
        for method, _ in CORE_METHODS
        if method != "core_bounded"
    }

    best_counts = {method: 0 for method, _ in POLICY_METHODS}
    top_two_counts = {method: 0 for method, _ in POLICY_METHODS}
    rank_lists = {method: [] for method, _ in POLICY_METHODS}
    for fold in folds:
        fold_id = str(fold["fold_id"])
        values = {method: float(mean(_metric_values(fold_id, method))) for method, _ in POLICY_METHODS}
        ranks = _average_ranks(values)
        best = max(values.values())
        for method, _ in POLICY_METHODS:
            if abs(values[method] - best) <= 1e-12:
                best_counts[method] += 1
            if ranks[method] <= 2.0:
                top_two_counts[method] += 1
            rank_lists[method].append(float(ranks[method]))

    policy_table = []
    for method, label in POLICY_METHODS:
        row = dict(policy[method])
        row["label"] = label
        row["best_or_tie_best"] = int(best_counts[method])
        row["top_two"] = int(top_two_counts[method])
        row["mean_rank"] = float(mean(rank_lists[method]))
        policy_table.append(row)
    policy_table.sort(key=lambda row: (row["mean_rank"], -row["best_or_tie_best"], -row["macro_mean"]))

    core_table = []
    for method, label in CORE_METHODS:
        row = dict(core[method])
        row["label"] = label
        if method == "core_bounded":
            row["macro_delta_vs_bounded"] = 0.0
            row["delta_ci"] = [0.0, 0.0]
        else:
            row["macro_delta_vs_bounded"] = core_delta[method]["macro_delta"]
            row["delta_ci"] = core_delta[method]["hierarchical_ci"]
        core_table.append(row)

    return {
        "split": SPLIT,
        "metric": METRIC,
        "seeds": SEEDS,
        "folds": folds,
        "policy_methods": policy_table,
        "core_methods": core_table,
        "core_delta": core_delta,
        "note": "Processed-window leave-one-positive-family-out ATLASv2 audit. Folds are family-disjoint and include one positive test family plus disjoint background families; this supplements but does not replace raw-pipeline multi-fold benchmark construction.",
    }


def _fmt_ci(row: dict[str, Any]) -> str:
    lo, hi = row["hierarchical_ci"]
    return f"[{lo:.3f}, {hi:.3f}]"


def _fmt_delta_ci(row: dict[str, Any]) -> str:
    lo, hi = row["delta_ci"]
    return f"[{lo:+.3f}, {hi:+.3f}]"


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# ATLASv2 processed-window LOPO family audit",
        "",
        summary["note"],
        "",
        "## Policy vs fixed families",
        "",
        "| Method | Macro AUPRC | 95% CI | Worst fold | Best/tie | Top-two | Mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["policy_methods"]:
        lines.append(
            f"| {row['label']} | {row['macro_mean']:.3f} | {_fmt_ci(row)} | {row['worst_fold']:.3f} | {row['best_or_tie_best']}/7 | {row['top_two']}/7 | {row['mean_rank']:.2f} |"
        )
    lines += [
        "",
        "## Core mechanism stress",
        "",
        "| Method | Macro AUPRC | 95% CI | Worst fold | Delta vs bounded | Delta CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["core_methods"]:
        lines.append(
            f"| {row['label']} | {row['macro_mean']:.3f} | {_fmt_ci(row)} | {row['worst_fold']:.3f} | {row['macro_delta_vs_bounded']:+.3f} | {_fmt_delta_ci(row)} |"
        )
    (RESULT_DIR / "atlasv2_lopo_family_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_policy_latex(summary: dict[str, Any]) -> None:
    best_macro = max(row["macro_mean"] for row in summary["policy_methods"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Processed-window leave-one-positive-family-out ATLASv2 audit. Each of the seven folds holds out one positive family for test and another positive family for dev, with disjoint background families. We report macro AUPRC across folds, hierarchical bootstrap confidence intervals over folds and seeds, worst-fold AUPRC, and fixed-family coverage statistics. This supplements the original held-out-family split rather than replacing raw-pipeline benchmark construction.}",
        r"\label{tab:atlasv2-lopo-family-audit}",
        r"\maxtablewidth{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Macro AUPRC & 95\% CI & Worst fold & Best/tie & Top-two & Mean rank \\",
        r"\midrule",
    ]
    for row in summary["policy_methods"]:
        macro = f"{row['macro_mean']:.3f}"
        if abs(row["macro_mean"] - best_macro) <= 1e-12:
            macro = r"\mathbf{" + macro + "}"
        lines.append(
            "{label} & ${macro}$ & ${ci}$ & ${worst:.3f}$ & ${best}/7$ & ${top}/7$ & ${rank:.2f}$ \\\\".format(
                label=_latex_escape(row["label"]),
                macro=macro,
                ci=_fmt_ci(row),
                worst=row["worst_fold"],
                best=row["best_or_tie_best"],
                top=row["top_two"],
                rank=row["mean_rank"],
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    (FIG_DIR / "tab_atlasv2_lopo_family_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_core_latex(summary: dict[str, Any]) -> None:
    best_macro = max(row["macro_mean"] for row in summary["core_methods"])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Cross-fold stress test for the two weakest mechanism claims in the original ATLASv2 split: bounded calibration and route-statistic gates. Deltas are paired against the bounded TRACER core across the same processed-window LOPO folds and seeds.}",
        r"\label{tab:atlasv2-lopo-core-ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Core variant & Macro AUPRC & 95\% CI & Worst fold & $\Delta$ vs bounded \\",
        r"\midrule",
    ]
    for row in summary["core_methods"]:
        macro = f"{row['macro_mean']:.3f}"
        if abs(row["macro_mean"] - best_macro) <= 1e-12:
            macro = r"\mathbf{" + macro + "}"
        lines.append(
            "{label} & ${macro}$ & ${ci}$ & ${worst:.3f}$ & ${delta:+.3f}$ {delta_ci} \\\\".format(
                label=_latex_escape(row["label"]),
                macro=macro,
                ci=_fmt_ci(row),
                worst=row["worst_fold"],
                delta=row["macro_delta_vs_bounded"],
                delta_ci=_fmt_delta_ci(row),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (FIG_DIR / "tab_atlasv2_lopo_core_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = build_summary()
    (RESULT_DIR / "atlasv2_lopo_family_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown(summary)
    write_policy_latex(summary)
    write_core_latex(summary)
    print("Wrote outputs/results/atlasv2_lopo_family_audit.json")
    print("Wrote figures/tab_atlasv2_lopo_family_audit.tex")
    print("Wrote figures/tab_atlasv2_lopo_core_ablation.tex")


if __name__ == "__main__":
    main()
