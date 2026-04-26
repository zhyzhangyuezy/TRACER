from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_DIR = ROOT / "outputs" / "blinded_expert_evidence_rating_study"
RESULTS_PATH = ROOT / "outputs" / "results" / "blinded_expert_evidence_rating_results.json"
TABLE_RATINGS = ROOT / "figures" / "tab_blinded_expert_evidence_rating.tex"
TABLE_PREF = ROOT / "figures" / "tab_blinded_expert_preference.tex"

DIMENSIONS = [
    ("relevance", "Relevance"),
    ("supportiveness", "Supportiveness"),
    ("actionability", "Actionability"),
    ("explanation_quality", "Explanation quality"),
    ("misleading_safety", "Misleading safety"),
]

PREFERENCE_CATEGORIES = ["TRACER route", "Prefix-Only", "Tie", "Neither"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float | None:
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bootstrap_ci(values: list[float], seed: int = 20260426, n_boot: int = 5000) -> tuple[float, float]:
    if not values:
        return (math.nan, math.nan)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.shape[0], replace=True)
        boots.append(float(sample.mean()))
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def _sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / math.sqrt(len(values)))


def _fleiss_kappa(items: list[list[str]], categories: list[str]) -> float | None:
    usable = [item for item in items if len(item) >= 2]
    if not usable:
        return None
    n_values = {len(item) for item in usable}
    if len(n_values) != 1:
        return None
    n = n_values.pop()
    if n <= 1:
        return None
    category_to_idx = {category: idx for idx, category in enumerate(categories)}
    p_i = []
    category_counts = np.zeros(len(categories), dtype=float)
    for item in usable:
        counts = np.zeros(len(categories), dtype=float)
        for label in item:
            if label in category_to_idx:
                counts[category_to_idx[label]] += 1.0
        category_counts += counts
        p_i.append(float((np.sum(counts * counts) - n) / (n * (n - 1))))
    p_bar = float(np.mean(p_i))
    p_j = category_counts / (len(usable) * n)
    p_e = float(np.sum(p_j * p_j))
    if abs(1.0 - p_e) < 1e-12:
        return None
    return float((p_bar - p_e) / (1.0 - p_e))


def _rating_files(study_dir: Path, ratings_dir: Path | None) -> list[Path]:
    if ratings_dir and ratings_dir.exists():
        return sorted(ratings_dir.glob("*.csv"))
    completed = sorted(study_dir.glob("rater_*_completed.csv"))
    if completed:
        return completed
    return sorted(study_dir.glob("rater_*_sheet.csv"))


def _is_completed_row(row: dict[str, str]) -> bool:
    pref = str(row.get("preferred_set", "")).strip()
    any_rating = any(
        _to_float(row.get(f"set_{set_name}_{dim}_1_5", ""))
        for set_name in ["a", "b"]
        for dim, _ in DIMENSIONS
    )
    return bool(pref or any_rating)


def analyze(study_dir: Path, ratings_dir: Path | None = None) -> dict[str, Any]:
    key_path = study_dir / "pairwise_key_private.csv"
    if not key_path.exists():
        raise FileNotFoundError(f"missing private key: {key_path}")
    key_by_pair = {row["pair_id"]: row for row in _read_csv(key_path)}

    files = _rating_files(study_dir, ratings_dir)
    all_rows: list[dict[str, str]] = []
    for path in files:
        for row in _read_csv(path):
            if _is_completed_row(row):
                if not row.get("rater_id"):
                    row["rater_id"] = path.stem
                all_rows.append(row)

    result: dict[str, Any] = {
        "study_dir": str(study_dir),
        "rating_files": [str(path) for path in files],
        "completed_rows": len(all_rows),
    }
    if not all_rows:
        result["status"] = "no_completed_ratings"
        return result

    method_scores: dict[str, dict[str, list[float]]] = {
        "TRACER route": {dim: [] for dim, _ in DIMENSIONS},
        "Prefix-Only": {dim: [] for dim, _ in DIMENSIONS},
    }
    paired_diffs: dict[str, list[float]] = {dim: [] for dim, _ in DIMENSIONS}
    preferences = Counter()
    preference_by_pair: dict[str, list[str]] = defaultdict(list)

    for row in all_rows:
        pair_id = row["pair_id"]
        key = key_by_pair[pair_id]
        set_to_method = {
            "a": key["set_a_method"],
            "b": key["set_b_method"],
        }

        per_row_method_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for set_name, method in set_to_method.items():
            for dim, _ in DIMENSIONS:
                value = _to_float(row.get(f"set_{set_name}_{dim}_1_5", ""))
                if value is None:
                    continue
                method_scores[method][dim].append(value)
                per_row_method_scores[method][dim] = value

        for dim, _ in DIMENSIONS:
            if dim in per_row_method_scores["TRACER route"] and dim in per_row_method_scores["Prefix-Only"]:
                paired_diffs[dim].append(
                    per_row_method_scores["TRACER route"][dim] - per_row_method_scores["Prefix-Only"][dim]
                )

        pref = str(row.get("preferred_set", "")).strip()
        mapped_pref = ""
        if pref.upper() == "A":
            mapped_pref = set_to_method["a"]
        elif pref.upper() == "B":
            mapped_pref = set_to_method["b"]
        elif pref.lower() == "tie":
            mapped_pref = "Tie"
        elif pref.lower() == "neither":
            mapped_pref = "Neither"
        if mapped_pref:
            preferences[mapped_pref] += 1
            preference_by_pair[pair_id].append(mapped_pref)

    rating_summary: dict[str, Any] = {}
    for dim, label in DIMENSIONS:
        dim_result: dict[str, Any] = {"label": label, "methods": {}}
        for method in ["TRACER route", "Prefix-Only"]:
            values = method_scores[method][dim]
            dim_result["methods"][method] = {
                "n": len(values),
                "mean": float(mean(values)) if values else math.nan,
                "sem": _sem(values),
            }
        diffs = paired_diffs[dim]
        ci_low, ci_high = _bootstrap_ci(diffs)
        dim_result["paired_delta_tracer_minus_prefix"] = {
            "n": len(diffs),
            "mean": float(mean(diffs)) if diffs else math.nan,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
        }
        rating_summary[dim] = dim_result

    total_pref = sum(preferences.values())
    pref_summary = {
        category: {
            "count": int(preferences[category]),
            "percent": float(100.0 * preferences[category] / total_pref) if total_pref else math.nan,
        }
        for category in PREFERENCE_CATEGORIES
    }
    kappa = _fleiss_kappa(list(preference_by_pair.values()), PREFERENCE_CATEGORIES)

    result.update(
        {
            "status": "completed",
            "rating_summary": rating_summary,
            "preference_summary": pref_summary,
            "preference_fleiss_kappa": kappa,
        }
    )
    return result


def _fmt(value: float) -> str:
    if value is None or math.isnan(float(value)):
        return "--"
    return f"{float(value):.2f}"


def write_tables(result: dict[str, Any]) -> None:
    if result.get("status") != "completed":
        return
    TABLE_RATINGS.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dim, label in DIMENSIONS:
        entry = result["rating_summary"][dim]
        tracer = entry["methods"]["TRACER route"]
        prefix = entry["methods"]["Prefix-Only"]
        delta = entry["paired_delta_tracer_minus_prefix"]
        rows.append(
            " & ".join(
                [
                    label,
                    f"{_fmt(tracer['mean'])} $\\pm$ {_fmt(tracer['sem'])}",
                    f"{_fmt(prefix['mean'])} $\\pm$ {_fmt(prefix['sem'])}",
                    f"{_fmt(delta['mean'])} [{_fmt(delta['ci95_low'])}, {_fmt(delta['ci95_high'])}]",
                ]
            )
            + r" \\"
        )
    TABLE_RATINGS.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{Blinded expert evidence-rating results. Scores use a 1--5 ordinal scale; higher is better.}",
                r"\label{tab:blinded-expert-evidence-rating}",
                r"\begin{tabular}{lccc}",
                r"\toprule",
                r"Metric & TRACER & Prefix-Only & Paired $\Delta$ \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    pref = result["preference_summary"]
    pref_rows = [
        f"{category} & {pref[category]['count']} & {_fmt(pref[category]['percent'])}\\% \\\\"
        for category in PREFERENCE_CATEGORIES
    ]
    kappa = result.get("preference_fleiss_kappa")
    kappa_text = "--" if kappa is None else _fmt(float(kappa))
    TABLE_PREF.write_text(
        "\n".join(
            [
                r"\begin{table}[t]",
                r"\centering",
                r"\caption{Pairwise expert preference in the blinded evidence-rating study.}",
                r"\label{tab:blinded-expert-preference}",
                r"\begin{tabular}{lcc}",
                r"\toprule",
                r"Preferred evidence set & Count & Share \\",
                r"\midrule",
                *pref_rows,
                r"\midrule",
                f"Fleiss-style $\\kappa$ & \\multicolumn{{2}}{{c}}{{{kappa_text}}} \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--ratings-dir", type=Path, default=None)
    args = parser.parse_args()

    result = analyze(args.study_dir, args.ratings_dir)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_tables(result)
    print(json.dumps({"status": result["status"], "completed_rows": result["completed_rows"]}, indent=2))
    if result["status"] == "no_completed_ratings":
        print("No completed ratings were found. Fill rater sheets or place completed CSV files in the study directory.")


if __name__ == "__main__":
    main()
