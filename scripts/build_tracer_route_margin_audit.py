from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "outputs" / "results"

ROUTE_ROWS = [
    {
        "dataset": "ATLASv2",
        "objective": "chronology",
        "result_candidates": [
            RESULT_DIR / "r239_tracer_adaptive_chronology_atlasv2_public_seed7.json",
            RESULT_DIR / "r239_tracer_adaptive_chronology_atlasv2_public.json",
        ],
    },
    {
        "dataset": "ATLASv2",
        "objective": "event-disjoint",
        "result_candidates": [
            RESULT_DIR / "r240_tracer_adaptive_event_atlasv2_public_seed7.json",
            RESULT_DIR / "r240_tracer_adaptive_event_atlasv2_public.json",
        ],
    },
    {
        "dataset": "AIT-ADS",
        "objective": "balanced",
        "result_candidates": [
            RESULT_DIR / "r241_tracer_adaptive_ait_ads_public_seed7.json",
            RESULT_DIR / "r241_tracer_adaptive_ait_ads_public.json",
        ],
    },
    {
        "dataset": "AIT-ADS",
        "objective": "event-disjoint",
        "result_candidates": [
            RESULT_DIR / "r242_tracer_adaptive_event_ait_ads_public_seed7.json",
            RESULT_DIR / "r242_tracer_adaptive_event_ait_ads_public.json",
        ],
    },
    {
        "dataset": "ATLAS-Raw",
        "objective": "balanced",
        "result_candidates": [
            RESULT_DIR / "r243_tracer_adaptive_atlas_raw_public_seed7.json",
            RESULT_DIR / "r243_tracer_adaptive_atlas_raw_public.json",
        ],
    },
    {
        "dataset": "Synthetic-CAM-LDS",
        "objective": "balanced",
        "result_candidates": [
            RESULT_DIR / "r244_tracer_adaptive_synthetic_cam_lds_seed7.json",
            RESULT_DIR / "r244_tracer_adaptive_synthetic_cam_lds.json",
        ],
    },
    {
        "dataset": "Workbook probe",
        "objective": "balanced",
        "result_candidates": [
            RESULT_DIR / "r245_tracer_adaptive_atlasv2_workbook_seed7.json",
            RESULT_DIR / "r245_tracer_adaptive_atlasv2_workbook.json",
        ],
    },
]

PUBLISHED_THRESHOLDS = {
    "cold_pos_rate_max": 0.02,
    "extreme_pos_rate_max": 0.005,
    "dense_pos_rate_min": 0.05,
    "dense_family_max": 6,
    "chrono_pos_family_max": 2,
    "chrono_diff2_min": 0.28,
    "chrono_peak_min": 4.0,
    "simple_pos_rate_min": 0.35,
    "simple_peak_max": 3.2,
    "simple_diff2_max": 0.26,
}


def _load_first(path_candidates: list[Path]) -> dict[str, Any]:
    for path in path_candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Missing result artifact for candidates: {path_candidates}")


def _collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in ROUTE_ROWS:
        payload = _load_first(spec["result_candidates"])
        policy = payload["auto_component_policy"]
        train_stats = policy["train_stats"]
        rows.append(
            {
                "dataset": spec["dataset"],
                "objective": spec["objective"],
                "regime": policy["regime"],
                "resolved_model_type": policy["resolved_model_type"],
                "positive_rate": float(train_stats["positive_rate"]),
                "dev_positive_count": int(policy["dev_stats"]["positive_count"]),
                "family_count": int(train_stats["family_count"]),
                "positive_family_count": int(train_stats["positive_family_count"]),
                "diff2_abs_mean": float(train_stats["diff2_abs_mean"]),
                "peak_ratio": float(train_stats["peak_ratio"]),
            }
        )
    return rows


def _threshold_specs() -> list[dict[str, Any]]:
    return [
        {
            "key": "cold_pos_rate_max",
            "threshold": r"$r_{+}^{\mathrm{train}} \le 0.02$ (cold-start sparse)",
            "published": PUBLISHED_THRESHOLDS["cold_pos_rate_max"],
            "direction": "max",
            "positive_filter": lambda row: row["regime"] == "cold_start_sparse",
            "negative_filter": lambda row: False,
        },
        {
            "key": "extreme_pos_rate_max",
            "threshold": r"$r_{+}^{\mathrm{train}} \le 0.005$ (extreme sparse)",
            "published": PUBLISHED_THRESHOLDS["extreme_pos_rate_max"],
            "direction": "max",
            "positive_filter": lambda row: row["regime"] == "extreme_sparse",
            "negative_filter": lambda row: row["regime"] not in {"cold_start_sparse", "extreme_sparse"},
        },
        {
            "key": "dense_pos_rate_min",
            "threshold": r"$r_{+}^{\mathrm{train}} \ge 0.05$ (dense activation)",
            "published": PUBLISHED_THRESHOLDS["dense_pos_rate_min"],
            "direction": "min",
            "positive_filter": lambda row: row["regime"] in {"dense_low_diversity", "dense_low_diversity_event"},
            "negative_filter": lambda row: row["regime"] not in {"dense_low_diversity", "dense_low_diversity_event", "simple_dense"},
        },
        {
            "key": "chrono_pos_family_max",
            "threshold": r"$F_{+}^{\mathrm{train}} \le 2$ (chronology-spiky)",
            "published": PUBLISHED_THRESHOLDS["chrono_pos_family_max"],
            "direction": "max",
            "positive_filter": lambda row: row["regime"] == "sparse_diverse_chrono_spiky",
            "negative_filter": lambda row: row["objective"] == "chronology" and row["regime"] != "sparse_diverse_chrono_spiky",
            "value_getter": lambda row: float(row["positive_family_count"]),
        },
        {
            "key": "chrono_diff2_min",
            "threshold": r"$\bar{\delta}_{2}^{\mathrm{train}} \ge 0.28$ (chronology-spiky)",
            "published": PUBLISHED_THRESHOLDS["chrono_diff2_min"],
            "direction": "min",
            "positive_filter": lambda row: row["regime"] == "sparse_diverse_chrono_spiky",
            "negative_filter": lambda row: row["objective"] == "chronology" and row["regime"] != "sparse_diverse_chrono_spiky",
            "value_getter": lambda row: float(row["diff2_abs_mean"]),
        },
        {
            "key": "chrono_peak_min",
            "threshold": r"$\rho_{\mathrm{peak}}^{\mathrm{train}} \ge 4.0$ (chronology-spiky)",
            "published": PUBLISHED_THRESHOLDS["chrono_peak_min"],
            "direction": "min",
            "positive_filter": lambda row: row["regime"] == "sparse_diverse_chrono_spiky",
            "negative_filter": lambda row: row["objective"] == "chronology" and row["regime"] != "sparse_diverse_chrono_spiky",
            "value_getter": lambda row: float(row["peak_ratio"]),
        },
        {
            "key": "simple_pos_rate_min",
            "threshold": r"$r_{+}^{\mathrm{train}} \ge 0.35$ (simple dense)",
            "published": PUBLISHED_THRESHOLDS["simple_pos_rate_min"],
            "direction": "min",
            "positive_filter": lambda row: row["regime"] == "simple_dense",
            "negative_filter": lambda row: row["regime"] not in {"cold_start_sparse", "extreme_sparse", "simple_dense"},
        },
    ]


def _value_getter(spec: dict[str, Any]) -> Callable[[dict[str, Any]], float]:
    if "value_getter" in spec:
        return spec["value_getter"]
    return lambda row: float(row["positive_rate"])


def _compute_interval(positive_values: list[float], negative_values: list[float], *, direction: str) -> dict[str, Any]:
    if not positive_values:
        return {
            "interval": "-",
            "identification": "unidentified",
            "lower": None,
            "upper": None,
            "lower_inclusive": False,
            "upper_inclusive": False,
        }
    pos_min = min(positive_values)
    pos_max = max(positive_values)
    if not negative_values:
        if direction == "max":
            return {
                "interval": f"[{pos_max:.4f}, +inf)",
                "identification": "one-sided",
                "lower": pos_max,
                "upper": None,
                "lower_inclusive": True,
                "upper_inclusive": False,
            }
        return {
            "interval": f"(-inf, {pos_min:.4f}]",
            "identification": "one-sided",
            "lower": None,
            "upper": pos_min,
            "lower_inclusive": False,
            "upper_inclusive": True,
        }

    if direction == "max":
        upper = min(negative_values)
        identification = "two-sided" if pos_max < upper else "conflict"
        return {
            "interval": f"[{pos_max:.4f}, {upper:.4f})",
            "identification": identification,
            "lower": pos_max,
            "upper": upper,
            "lower_inclusive": True,
            "upper_inclusive": False,
        }

    lower = max(negative_values)
    identification = "two-sided" if lower < pos_min else "conflict"
    return {
        "interval": f"({lower:.4f}, {pos_min:.4f}]",
        "identification": identification,
        "lower": lower,
        "upper": pos_min,
        "lower_inclusive": False,
        "upper_inclusive": True,
    }


def _evaluate_threshold_spec(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    getter = _value_getter(spec)
    positive_values = [getter(row) for row in rows if spec["positive_filter"](row)]
    negative_values = [getter(row) for row in rows if spec["negative_filter"](row)]
    return _compute_interval(positive_values, negative_values, direction=str(spec["direction"]))


def _is_supported(interval_info: dict[str, Any], published: float) -> bool:
    if interval_info["identification"] in {"unidentified", "conflict"}:
        return False
    lower = interval_info["lower"]
    upper = interval_info["upper"]
    if lower is not None:
        if interval_info["lower_inclusive"]:
            if published < lower:
                return False
        elif published <= lower:
            return False
    if upper is not None:
        if interval_info["upper_inclusive"]:
            if published > upper:
                return False
        elif published >= upper:
            return False
    return True


def _support_slack(interval_info: dict[str, Any], published: float) -> float | None:
    if not _is_supported(interval_info, published):
        return None
    distances: list[float] = []
    if interval_info["lower"] is not None:
        distances.append(published - float(interval_info["lower"]))
    if interval_info["upper"] is not None:
        distances.append(float(interval_info["upper"]) - published)
    if not distances:
        return None
    return min(distances)


def _threshold_interval_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    interval_rows: list[dict[str, str]] = []
    for spec in _threshold_specs():
        info = _evaluate_threshold_spec(spec, rows)
        interval_rows.append(
            {
                "threshold": str(spec["threshold"]),
                "published": f"{float(spec['published']):.3f}" if isinstance(spec["published"], float) and spec["published"] < 1 else f"{float(spec['published']):.0f}" if float(spec["published"]).is_integer() else f"{float(spec['published']):.2f}",
                "interval": str(info["interval"]),
                "identification": str(info["identification"]),
            }
        )
    return interval_rows


def _leave_one_dataset_out_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    lobo_rows: list[dict[str, str]] = []
    for spec in _threshold_specs():
        identifiable = 0
        supported = 0
        unidentified_datasets: list[str] = []
        contradictory_datasets: list[str] = []
        tightest_interval = "-"
        tightest_slack: float | None = None

        for held_out in datasets:
            subset = [row for row in rows if row["dataset"] != held_out]
            info = _evaluate_threshold_spec(spec, subset)
            if info["identification"] == "unidentified":
                unidentified_datasets.append(held_out)
                continue
            if info["identification"] == "conflict":
                identifiable += 1
                contradictory_datasets.append(held_out)
                continue

            identifiable += 1
            if _is_supported(info, float(spec["published"])):
                supported += 1
                slack = _support_slack(info, float(spec["published"]))
                if tightest_slack is None or (slack is not None and slack < tightest_slack):
                    tightest_slack = slack
                    tightest_interval = f"{held_out}: {info['interval']}"
            else:
                contradictory_datasets.append(held_out)

        if contradictory_datasets:
            support_text = f"{supported} / {identifiable}"
        else:
            support_text = f"{supported} / {identifiable} identifiable"

        if not unidentified_datasets:
            unidentified_text = "-"
        else:
            unidentified_text = ", ".join(unidentified_datasets)
        if contradictory_datasets:
            if unidentified_text == "-":
                unidentified_text = "contradicted: " + ", ".join(contradictory_datasets)
            else:
                unidentified_text += "; contradicted: " + ", ".join(contradictory_datasets)

        lobo_rows.append(
            {
                "threshold": str(spec["threshold"]),
                "published": f"{float(spec['published']):.3f}" if isinstance(spec["published"], float) and spec["published"] < 1 else f"{float(spec['published']):.0f}" if float(spec["published"]).is_integer() else f"{float(spec['published']):.2f}",
                "support": support_text,
                "tightest_interval": tightest_interval,
                "unidentified": unidentified_text,
            }
        )
    return lobo_rows


def _route_margin_row(row: dict[str, Any]) -> dict[str, str]:
    regime = row["regime"]
    if regime == "cold_start_sparse":
        nearest = PUBLISHED_THRESHOLDS["cold_pos_rate_max"] - row["positive_rate"]
        reason = r"$0.02 - r_{+}^{\mathrm{train}}$"
    elif regime == "extreme_sparse":
        nearest = PUBLISHED_THRESHOLDS["extreme_pos_rate_max"] - row["positive_rate"]
        reason = r"$0.005 - r_{+}^{\mathrm{train}}$"
    elif regime in {"dense_low_diversity", "dense_low_diversity_event"}:
        margins = [
            (row["positive_rate"] - PUBLISHED_THRESHOLDS["dense_pos_rate_min"], r"$r_{+}^{\mathrm{train}} - 0.05$"),
            (PUBLISHED_THRESHOLDS["dense_family_max"] - row["family_count"], r"$6 - F^{\mathrm{train}}$"),
        ]
        nearest, reason = min(margins, key=lambda item: item[0])
    elif regime == "sparse_diverse_chrono_spiky":
        margins = [
            (PUBLISHED_THRESHOLDS["chrono_pos_family_max"] - row["positive_family_count"], r"$2 - F_{+}^{\mathrm{train}}$"),
            (row["diff2_abs_mean"] - PUBLISHED_THRESHOLDS["chrono_diff2_min"], r"$\bar{\delta}_{2}^{\mathrm{train}} - 0.28$"),
            (row["peak_ratio"] - PUBLISHED_THRESHOLDS["chrono_peak_min"], r"$\rho_{\mathrm{peak}}^{\mathrm{train}} - 4.0$"),
        ]
        nearest, reason = min(margins, key=lambda item: item[0])
    elif regime == "simple_dense":
        margins = [
            (row["positive_rate"] - PUBLISHED_THRESHOLDS["simple_pos_rate_min"], r"$r_{+}^{\mathrm{train}} - 0.35$"),
            (PUBLISHED_THRESHOLDS["simple_peak_max"] - row["peak_ratio"], r"$3.2 - \rho_{\mathrm{peak}}^{\mathrm{train}}$"),
            (PUBLISHED_THRESHOLDS["simple_diff2_max"] - row["diff2_abs_mean"], r"$0.26 - \bar{\delta}_{2}^{\mathrm{train}}$"),
        ]
        nearest, reason = min(margins, key=lambda item: item[0])
    else:
        if row["objective"] == "event-disjoint":
            candidates = [
                (row["positive_rate"] - PUBLISHED_THRESHOLDS["extreme_pos_rate_max"], r"$r_{+}^{\mathrm{train}} - 0.005$"),
                (PUBLISHED_THRESHOLDS["dense_pos_rate_min"] - row["positive_rate"], r"$0.05 - r_{+}^{\mathrm{train}}$"),
                (row["family_count"] - PUBLISHED_THRESHOLDS["dense_family_max"], r"$F^{\mathrm{train}} - 6$"),
            ]
        else:
            candidates = [
                (row["positive_rate"] - PUBLISHED_THRESHOLDS["extreme_pos_rate_max"], r"$r_{+}^{\mathrm{train}} - 0.005$"),
                (PUBLISHED_THRESHOLDS["dense_pos_rate_min"] - row["positive_rate"], r"$0.05 - r_{+}^{\mathrm{train}}$"),
            ]
        nearest, reason = min(candidates, key=lambda item: abs(item[0]))
    return {
        "dataset": row["dataset"],
        "objective": row["objective"],
        "regime": regime.replace("_", r"\_"),
        "resolved_model_type": row["resolved_model_type"].replace("_", r"\_"),
        "nearest_margin": f"{nearest:.3f}",
        "reason": reason,
    }


def build_threshold_table_tex(interval_rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Threshold-interval audit for the frozen \texttt{tracer\_adaptive} rule table. The interval column reports the feasible range that preserves the observed route assignments on the released benchmark suite under the current rule ordering. Two-sided intervals indicate that the suite contains both supporting and opposing examples for that scalar cutoff; one-sided intervals indicate underidentified thresholds that remain conservative heuristics rather than fully pinned-down separators.}",
        r"\label{tab:tracer-threshold-intervals}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Threshold & Published value & Feasible interval & Identification \\",
        r"\midrule",
    ]
    for row in interval_rows:
        lines.append(
            row["threshold"]
            + " & "
            + row["published"]
            + " & "
            + row["interval"]
            + " & "
            + row["identification"]
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_lobo_table_tex(lobo_rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Leave-one-benchmark-out threshold audit for the frozen \texttt{tracer\_adaptive} rule table. Each row removes one benchmark family at a time, recomputes the admissible interval from the remaining released benchmarks, and checks whether the published threshold still lies inside that interval whenever the threshold remains identifiable. This audit distinguishes actual contradictions from simple lack of coverage when a held-out benchmark is the only positive anchor for a regime.}",
        r"\label{tab:tracer-threshold-lobo}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{p{4.4cm}cccc}",
        r"\toprule",
        r"Threshold & Published value & Supported folds & Tightest held-out interval & Unidentified / contradicted hold-out \\",
        r"\midrule",
    ]
    for row in lobo_rows:
        lines.append(
            row["threshold"]
            + " & "
            + row["published"]
            + " & "
            + row["support"]
            + " & "
            + row["tightest_interval"]
            + " & "
            + row["unidentified"]
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_margin_table_tex(margin_rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Boundary-margin audit for the released \texttt{tracer\_adaptive} routes. The nearest-margin column reports the smallest signed slack to the threshold expression that most tightly supports the current route. Positive margins indicate that the observed benchmark sits on the intended side of the active boundary rather than exactly on it.}",
        r"\label{tab:tracer-route-margins}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llllcc}",
        r"\toprule",
        r"Dataset & Objective & Regime & Resolved model & Nearest margin & Tightest expression \\",
        r"\midrule",
    ]
    for row in margin_rows:
        lines.append(
            row["dataset"]
            + " & "
            + row["objective"]
            + " & "
            + row["regime"]
            + " & "
            + row["resolved_model_type"]
            + " & "
            + row["nearest_margin"]
            + " & "
            + row["reason"]
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = _collect_rows()
    interval_rows = _threshold_interval_rows(rows)
    lobo_rows = _leave_one_dataset_out_rows(rows)
    margin_rows = [_route_margin_row(row) for row in rows]

    payload = {
        "rows": rows,
        "threshold_intervals": interval_rows,
        "threshold_leave_one_dataset_out": lobo_rows,
        "route_margins": margin_rows,
    }
    (ROOT / "outputs" / "results" / "tracer_route_margin_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (FIG_DIR / "tab_tracer_threshold_intervals.tex").write_text(build_threshold_table_tex(interval_rows), encoding="utf-8")
    (FIG_DIR / "tab_tracer_threshold_lobo.tex").write_text(build_lobo_table_tex(lobo_rows), encoding="utf-8")
    (FIG_DIR / "tab_tracer_route_margins.tex").write_text(build_margin_table_tex(margin_rows), encoding="utf-8")
    print(ROOT / "outputs" / "results" / "tracer_route_margin_audit.json")
    print(FIG_DIR / "tab_tracer_threshold_intervals.tex")
    print(FIG_DIR / "tab_tracer_threshold_lobo.tex")
    print(FIG_DIR / "tab_tracer_route_margins.tex")


if __name__ == "__main__":
    main()
