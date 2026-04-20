from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"

SEEDS_20 = [7, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025]
SEEDS_3 = [7, 13, 21]

PREDICTION_CACHE = RESULT_DIR / "audits" / "public_event_significance_seed_exports"
FIXED_EVENT_PREDICTIONS = {
    "DLinear": "r020_dlinear_forecaster_atlasv2_public_event_disjoint_test_event_disjoint_seed{seed}.json",
    "Small-Transformer": "r010_transformer_forecaster_atlasv2_public_event_disjoint_test_event_disjoint_seed{seed}.json",
    "Prefix-Only": "r010_prefix_retrieval_atlasv2_public_event_disjoint_test_event_disjoint_seed{seed}.json",
}
TRACER_EVENT_PREDICTION = "r240_tracer_adaptive_event_atlasv2_public_test_event_disjoint_seed{seed}.json"

LOPO_METHODS = {
    "adaptive": "Adaptive policy",
    "dlinear": "DLinear",
    "transformer": "Small-Transformer",
    "prefix": "Prefix-Only",
}
LOPO_FIXED = ["dlinear", "transformer", "prefix"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _rank_average(scores: list[np.ndarray]) -> np.ndarray:
    ranks = []
    for score in scores:
        # Stable rank transform into [0, 1]. This avoids scale calibration assumptions.
        order = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort").astype(np.float64)
        ranks.append(order / max(score.shape[0] - 1, 1))
    return np.vstack(ranks).mean(axis=0)


def build_score_ensemble_audit() -> dict[str, Any]:
    rows = []
    per_method = {name: [] for name in [*FIXED_EVENT_PREDICTIONS.keys(), "TRACER", "Uniform fixed ensemble", "Rank fixed ensemble", "Single-family oracle"]}
    for seed in SEEDS_20:
        fixed_scores = []
        fixed_auprc = {}
        y_true = None
        for label, template in FIXED_EVENT_PREDICTIONS.items():
            payload = _load_json(PREDICTION_CACHE / template.format(seed=seed))
            pred = payload["predictions"]
            current_y = np.asarray(pred["y_true"], dtype=int)
            score = np.asarray(pred["y_score"], dtype=np.float64)
            if y_true is None:
                y_true = current_y
            elif not np.array_equal(y_true, current_y):
                raise ValueError(f"Mismatched y_true for seed {seed}, {label}")
            value = _auprc(current_y, score)
            fixed_auprc[label] = value
            fixed_scores.append(score)
            per_method[label].append(value)

        tracer_payload = _load_json(PREDICTION_CACHE / TRACER_EVENT_PREDICTION.format(seed=seed))
        tracer_pred = tracer_payload["predictions"]
        tracer_y = np.asarray(tracer_pred["y_true"], dtype=int)
        tracer_score = np.asarray(tracer_pred["y_score"], dtype=np.float64)
        if not np.array_equal(y_true, tracer_y):
            raise ValueError(f"Mismatched y_true for TRACER seed {seed}")
        tracer_auprc = _auprc(tracer_y, tracer_score)
        uniform_auprc = _auprc(y_true, np.vstack(fixed_scores).mean(axis=0))
        rank_auprc = _auprc(y_true, _rank_average(fixed_scores))
        oracle_auprc = max(fixed_auprc.values())

        per_method["TRACER"].append(tracer_auprc)
        per_method["Uniform fixed ensemble"].append(uniform_auprc)
        per_method["Rank fixed ensemble"].append(rank_auprc)
        per_method["Single-family oracle"].append(oracle_auprc)
        rows.append(
            {
                "seed": seed,
                "tracer": tracer_auprc,
                "uniform_fixed": uniform_auprc,
                "rank_fixed": rank_auprc,
                "oracle_fixed": oracle_auprc,
                "fixed": fixed_auprc,
            }
        )

    methods = []
    for label, values in per_method.items():
        methods.append(
            {
                "method": label,
                "mean_auprc": float(mean(values)),
                "std_auprc": float(pstdev(values)),
                "delta_vs_tracer": float(mean(values) - mean(per_method["TRACER"])),
            }
        )
    return {
        "audit": "ATLASv2 held-out-family score-level fixed-family ensemble audit",
        "seeds": SEEDS_20,
        "methods": methods,
        "rows": rows,
        "note": "Uniform and rank ensembles average only fixed-family DLinear, Small-Transformer, and Prefix-Only predictions. The single-family oracle is an ex-post upper bound over choosing one fixed family per seed and is not a deployable stacker.",
    }


def _lopo_path(fold_id: str, method: str, seed: int) -> Path:
    return RESULT_DIR / f"r300_lopo_{fold_id}_{method}_seed{seed}.json"


def _fold_method_mean(fold_id: str, method: str) -> float:
    values = []
    for seed in SEEDS_3:
        payload = _load_json(_lopo_path(fold_id, method, seed))
        values.append(float(payload["test_event_disjoint"]["AUPRC"]))
    return float(mean(values))


def _fold_features(fold: dict[str, Any]) -> list[float]:
    fold_id = str(fold["fold_id"])
    payload = _load_json(_lopo_path(fold_id, "adaptive", SEEDS_3[0]))
    train_stats = payload["auto_component_policy"]["train_stats"]
    dev_stats = payload["auto_component_policy"]["dev_stats"]
    return [
        float(train_stats["positive_rate"]),
        float(train_stats["positive_count"]),
        float(train_stats["family_count"]),
        float(train_stats["positive_family_count"]),
        float(train_stats["diff2_abs_mean"]),
        float(train_stats["peak_ratio"]),
        float(train_stats["signature_std"]),
        float(dev_stats["positive_rate"]),
        float(dev_stats["positive_count"]),
        float(dev_stats["family_count"]),
        float(dev_stats["positive_family_count"]),
        float(dev_stats["diff2_abs_mean"]),
        float(dev_stats["peak_ratio"]),
        float(fold["test"]["positives"]),
        float(fold["test"]["size"]),
    ]


def build_lopo_learned_selector_audit() -> dict[str, Any]:
    folds_payload = _load_json(RESULT_DIR / "atlasv2_lopo_family_folds.json")
    folds = list(folds_payload["folds"])
    features = np.asarray([_fold_features(fold) for fold in folds], dtype=np.float64)
    method_scores = {
        method: np.asarray([_fold_method_mean(str(fold["fold_id"]), method) for fold in folds], dtype=np.float64)
        for method in LOPO_METHODS
    }

    rows = []
    selected_ridge = []
    selected_nearest = []
    selected_global = []
    for held_idx, fold in enumerate(folds):
        train_idx = np.asarray([idx for idx in range(len(folds)) if idx != held_idx], dtype=int)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx])
        x_held = scaler.transform(features[[held_idx]])

        ridge_predictions = {}
        nearest_predictions = {}
        for method in LOPO_FIXED:
            ridge = Ridge(alpha=1.0)
            ridge.fit(x_train, method_scores[method][train_idx])
            ridge_predictions[method] = float(ridge.predict(x_held)[0])

            distances = np.linalg.norm(x_train - x_held[0], axis=1)
            nearest_fold = int(train_idx[int(np.argmin(distances))])
            nearest_predictions[method] = float(method_scores[method][nearest_fold])

        global_means = {method: float(method_scores[method][train_idx].mean()) for method in LOPO_FIXED}
        ridge_choice = max(ridge_predictions, key=ridge_predictions.get)
        nearest_choice = max(nearest_predictions, key=nearest_predictions.get)
        global_choice = max(global_means, key=global_means.get)
        oracle_choice = max(LOPO_FIXED, key=lambda method: float(method_scores[method][held_idx]))
        adaptive_value = float(method_scores["adaptive"][held_idx])

        selected_ridge.append(float(method_scores[ridge_choice][held_idx]))
        selected_nearest.append(float(method_scores[nearest_choice][held_idx]))
        selected_global.append(float(method_scores[global_choice][held_idx]))
        rows.append(
            {
                "fold_id": fold["fold_id"],
                "test_family": fold["test_family"],
                "adaptive": adaptive_value,
                "fixed_oracle": float(method_scores[oracle_choice][held_idx]),
                "fixed_oracle_choice": LOPO_METHODS[oracle_choice],
                "ridge_selector": float(method_scores[ridge_choice][held_idx]),
                "ridge_choice": LOPO_METHODS[ridge_choice],
                "nearest_selector": float(method_scores[nearest_choice][held_idx]),
                "nearest_choice": LOPO_METHODS[nearest_choice],
                "global_selector": float(method_scores[global_choice][held_idx]),
                "global_choice": LOPO_METHODS[global_choice],
            }
        )

    adaptive_values = method_scores["adaptive"]
    oracle_values = np.asarray([row["fixed_oracle"] for row in rows], dtype=np.float64)
    ridge_values = np.asarray(selected_ridge, dtype=np.float64)
    nearest_values = np.asarray(selected_nearest, dtype=np.float64)
    global_values = np.asarray(selected_global, dtype=np.float64)
    return {
        "audit": "LOPO learned split-level selector audit",
        "rows": rows,
        "summary": {
            "adaptive_macro": float(adaptive_values.mean()),
            "fixed_oracle_macro": float(oracle_values.mean()),
            "ridge_selector_macro": float(ridge_values.mean()),
            "nearest_selector_macro": float(nearest_values.mean()),
            "global_selector_macro": float(global_values.mean()),
            "adaptive_minus_ridge": float(adaptive_values.mean() - ridge_values.mean()),
            "adaptive_minus_nearest": float(adaptive_values.mean() - nearest_values.mean()),
            "adaptive_minus_global": float(adaptive_values.mean() - global_values.mean()),
            "adaptive_minus_oracle": float(adaptive_values.mean() - oracle_values.mean()),
        },
        "note": "Selectors are trained leave-one-fold-out using only the other processed-window LOPO folds and train/dev regime statistics. The single-family oracle is an ex-post upper bound over DLinear, Small-Transformer, and Prefix-Only.",
    }


def _fmt(value: float, bold: bool = False) -> str:
    body = f"{value:.3f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def _fmt_delta(value: float, bold: bool = False) -> str:
    body = f"{value:+.3f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def write_markdown(score_audit: dict[str, Any], selector_audit: dict[str, Any]) -> None:
    lines = [
        "# Ensemble and learned-selector audits",
        "",
        "## ATLASv2 held-out-family score ensembles",
        "",
        score_audit["note"],
        "",
        "| Method | AUPRC | Delta vs TRACER |",
        "|---|---:|---:|",
    ]
    for row in score_audit["methods"]:
        lines.append(f"| {row['method']} | {row['mean_auprc']:.3f} +/- {row['std_auprc']:.3f} | {row['delta_vs_tracer']:+.3f} |")
    lines += [
        "",
        "## LOPO learned split-level selectors",
        "",
        selector_audit["note"],
        "",
        "| Selector | Macro AUPRC | Delta vs adaptive |",
        "|---|---:|---:|",
    ]
    summary = selector_audit["summary"]
    selector_rows = [
        ("Adaptive policy", summary["adaptive_macro"], 0.0),
        ("LOFO ridge selector", summary["ridge_selector_macro"], -summary["adaptive_minus_ridge"]),
        ("Nearest-fold selector", summary["nearest_selector_macro"], -summary["adaptive_minus_nearest"]),
        ("Global-mean selector", summary["global_selector_macro"], -summary["adaptive_minus_global"]),
        ("Single-family oracle", summary["fixed_oracle_macro"], -summary["adaptive_minus_oracle"]),
    ]
    for label, value, delta in selector_rows:
        lines.append(f"| {label} | {value:.3f} | {delta:+.3f} |")
    (RESULT_DIR / "ensemble_and_selector_audits.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(score_audit: dict[str, Any], selector_audit: dict[str, Any]) -> None:
    methods = score_audit["methods"]
    best_score = max(float(row["mean_auprc"]) for row in methods)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Fixed-family ensemble and learned-selector stress tests. Panel A evaluates simple score-level ensembles on the ATLASv2 held-out-family prediction exports; the single-family oracle row is an ex-post upper bound over choosing one fixed family and is not deployable. Panel B evaluates leave-one-fold-out learned split-level selectors on the processed-window ATLASv2 LOPO folds using only train/dev regime statistics from the other folds.}",
        r"\label{tab:ensemble-selector-audit}",
        r"\maxtablewidth{",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Panel & Method & AUPRC & $\Delta$ vs adaptive \\",
        r"\midrule",
    ]
    for row in methods:
        value = float(row["mean_auprc"])
        delta = float(row["delta_vs_tracer"])
        lines.append(
            "A & {method} & {value} $\\pm$ {std:.3f} & {delta} \\\\".format(
                method=str(row["method"]).replace("_", r"\_"),
                value=_fmt(value, bold=abs(value - best_score) <= 1e-12),
                std=float(row["std_auprc"]),
                delta=_fmt_delta(delta, bold=delta >= -1e-12),
            )
        )
    summary = selector_audit["summary"]
    selector_rows = [
        ("Adaptive policy", summary["adaptive_macro"], 0.0),
        ("LOFO ridge selector", summary["ridge_selector_macro"], -summary["adaptive_minus_ridge"]),
        ("Nearest-fold selector", summary["nearest_selector_macro"], -summary["adaptive_minus_nearest"]),
        ("Global-mean selector", summary["global_selector_macro"], -summary["adaptive_minus_global"]),
        ("Single-family oracle", summary["fixed_oracle_macro"], -summary["adaptive_minus_oracle"]),
    ]
    best_selector = max(value for _, value, _ in selector_rows)
    lines.append(r"\midrule")
    for label, value, delta in selector_rows:
        lines.append(
            "B & {label} & {value} & {delta} \\\\".format(
                label=label.replace("_", r"\_"),
                value=_fmt(float(value), bold=abs(float(value) - best_selector) <= 1e-12),
                delta=_fmt_delta(float(delta), bold=delta >= -1e-12),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    (FIG_DIR / "tab_ensemble_selector_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    score_audit = build_score_ensemble_audit()
    selector_audit = build_lopo_learned_selector_audit()
    payload = {"score_ensemble": score_audit, "lopo_selector": selector_audit}
    (RESULT_DIR / "ensemble_and_selector_audits.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(score_audit, selector_audit)
    write_latex(score_audit, selector_audit)
    print("Wrote outputs/results/ensemble_and_selector_audits.json")
    print("Wrote figures/tab_ensemble_selector_audit.tex")


if __name__ == "__main__":
    main()
