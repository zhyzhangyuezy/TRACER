from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"

SEEDS = [7, 13, 21]
METHODS = [
    "TRACER",
    "DLinear-Forecaster",
    "Small-Transformer-Forecaster",
    "Prefix-Only-Retrieval + Fusion",
]

SEEDED_SETTINGS = [
    {
        "key": "atlasv2_chrono",
        "header": "ATLASv2-C\\,(3s)",
        "caption_name": "ATLASv2 chronology",
        "kind": "seeded",
        "split": "test",
        "experiments": {
            "TRACER": "r239_tracer_adaptive_chronology_atlasv2_public",
            "DLinear-Forecaster": "r020_dlinear_forecaster_atlasv2_public",
            "Small-Transformer-Forecaster": "r006_transformer_forecaster_atlasv2_public",
            "Prefix-Only-Retrieval + Fusion": "r008_prefix_retrieval_atlasv2_public",
        },
    },
    {
        "key": "atlas_raw_chrono",
        "header": "Raw-C\\,(3s)",
        "caption_name": "ATLAS-Raw chronology",
        "kind": "seeded",
        "split": "test",
        "experiments": {
            "TRACER": "r243_tracer_adaptive_atlas_raw_public",
            "DLinear-Forecaster": "r027_dlinear_forecaster_atlas_raw_public",
            "Small-Transformer-Forecaster": "r032_transformer_forecaster_atlas_raw_public",
            "Prefix-Only-Retrieval + Fusion": "r035_prefix_retrieval_atlas_raw_public",
        },
    },
    {
        "key": "atlas_raw_event",
        "header": "Raw-E\\,(3s)",
        "caption_name": "ATLAS-Raw event-disjoint",
        "kind": "seeded",
        "split": "test_event_disjoint",
        "experiments": {
            "TRACER": "r243_tracer_adaptive_atlas_raw_public",
            "DLinear-Forecaster": "r027_dlinear_forecaster_atlas_raw_public",
            "Small-Transformer-Forecaster": "r032_transformer_forecaster_atlas_raw_public",
            "Prefix-Only-Retrieval + Fusion": "r035_prefix_retrieval_atlas_raw_public",
        },
    },
]

AUDIT_SETTINGS = [
    {
        "key": "atlasv2_event_20seed",
        "header": "ATLASv2-HF\\,(20s)",
        "caption_name": "ATLASv2 held-out-family",
        "kind": "audit",
        "path": RESULT_DIR / "public_event_significance_audit.json",
    },
    {
        "key": "ait_ads_chrono_20seed",
        "header": "AIT-C\\,(20s)",
        "caption_name": "AIT-ADS chronology",
        "kind": "audit",
        "path": RESULT_DIR / "ait_ads_chronology_significance_audit.json",
    },
    {
        "key": "ait_ads_event_20seed",
        "header": "AIT-HO\\,(20s)",
        "caption_name": "AIT-ADS held-out-scenario",
        "kind": "audit",
        "path": RESULT_DIR / "ait_ads_event_significance_audit.json",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_seeded_metric(base_experiment: str, split: str) -> dict[str, float]:
    values: list[float] = []
    for seed in SEEDS:
        payload = _load_json(RESULT_DIR / f"{base_experiment}_seed{seed}.json")
        row = payload.get(split)
        if row is None:
            if split == "test" and "test" in payload:
                row = payload["test"]
            elif split == "test_event_disjoint" and "test_event_disjoint" in payload:
                row = payload["test_event_disjoint"]
            else:
                raise KeyError(f"Missing split {split} in {base_experiment}_seed{seed}.json")
        values.append(float(row["AUPRC"]))
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "n": len(values),
        "evidence": "seeded_3",
    }


def _load_audit_metric(path: Path, label: str) -> dict[str, float]:
    payload = _load_json(path)
    row = payload["models"][label]
    return {
        "mean": float(row["mean_auprc"]),
        "std": float(row["std_auprc"]),
        "n": len(row["seeds"]),
        "evidence": f"seeded_{len(row['seeds'])}",
    }


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    sorted_items = sorted(values.items(), key=lambda item: item[1], reverse=True)
    ranks: dict[str, float] = {}
    position = 1
    idx = 0
    while idx < len(sorted_items):
        tied = [sorted_items[idx]]
        idx += 1
        while idx < len(sorted_items) and abs(sorted_items[idx][1] - tied[0][1]) <= 1e-12:
            tied.append(sorted_items[idx])
            idx += 1
        avg_rank = (position + position + len(tied) - 1) / 2.0
        for method, _ in tied:
            ranks[method] = avg_rank
        position += len(tied)
    return ranks


def _format_score(value: float, *, bold: bool = False, underline: bool = False) -> str:
    body = f"{value:.3f}"
    if bold:
        return f"$\\mathbf{{{body}}}$"
    if underline:
        return f"$\\underline{{{body}}}$"
    return f"${body}$"


def _format_count(value: int, max_value: int) -> str:
    if value == max_value:
        return f"$\\mathbf{{{value}}}$"
    return f"${value}$"


def _format_rank(value: float, best_value: float) -> str:
    body = f"{value:.2f}"
    if abs(value - best_value) <= 1e-12:
        return f"$\\mathbf{{{body}}}$"
    return f"${body}$"


def build_audit() -> dict[str, Any]:
    settings = SEEDED_SETTINGS + AUDIT_SETTINGS
    per_setting: list[dict[str, Any]] = []
    method_summary = {
        method: {
            "best_or_tie_best": 0,
            "top_two": 0,
            "ranks": [],
        }
        for method in METHODS
    }

    for spec in settings:
        method_rows: dict[str, dict[str, float]] = {}
        if spec["kind"] == "seeded":
            for method, experiment in spec["experiments"].items():
                method_rows[method] = _load_seeded_metric(experiment, str(spec["split"]))
        else:
            for method in METHODS:
                method_rows[method] = _load_audit_metric(Path(spec["path"]), method)

        means = {method: row["mean"] for method, row in method_rows.items()}
        ranks = _average_ranks(means)
        best_value = max(means.values())
        sorted_unique = sorted(set(means.values()), reverse=True)
        second_value = sorted_unique[1] if len(sorted_unique) > 1 else None

        for method in METHODS:
            if abs(means[method] - best_value) <= 1e-12:
                method_summary[method]["best_or_tie_best"] += 1
            if ranks[method] <= 2.0:
                method_summary[method]["top_two"] += 1
            method_summary[method]["ranks"].append(ranks[method])

        per_setting.append(
            {
                "key": spec["key"],
                "header": spec["header"],
                "caption_name": spec["caption_name"],
                "means": means,
                "stds": {method: row["std"] for method, row in method_rows.items()},
                "ranks": ranks,
                "best_value": best_value,
                "second_value": second_value,
                "evidence": next(iter(method_rows.values()))["evidence"],
            }
        )

    methods_payload = []
    for method in METHODS:
        ranks = [float(value) for value in method_summary[method]["ranks"]]
        methods_payload.append(
            {
                "method": method,
                "best_or_tie_best": int(method_summary[method]["best_or_tie_best"]),
                "top_two": int(method_summary[method]["top_two"]),
                "mean_rank": float(mean(ranks)),
            }
        )

    methods_payload.sort(key=lambda row: (row["mean_rank"], -row["best_or_tie_best"], -row["top_two"]))
    return {
        "settings": per_setting,
        "methods": methods_payload,
        "note": "Workbook probe excluded from this audit because it is a weaker chronological-only cold-start probe with a different candidate family set and no held-out-family split.",
    }


def build_table_tex(summary: dict[str, Any]) -> str:
    method_rows = {row["method"]: row for row in summary["methods"]}
    best_count = max(int(row["best_or_tie_best"]) for row in summary["methods"])
    best_rank = min(float(row["mean_rank"]) for row in summary["methods"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Cross-benchmark fixed-family audit on the six stronger public benchmark-split settings using the fixed families that remain comparable across the full suite and the released 20-seed audits. The ATLASv2 and AIT-ADS held-out columns use the 20-seed audits, while ATLASv2 chronology and the two ATLAS-Raw columns use the direct three-seed reruns. Best available AUPRC in each benchmark column is bolded and the second-best value is underlined. The summary columns show how often each method is best or tie-best, how often it stays top-two, and its average rank across the six settings. Workbook is excluded because it is a weaker chronological-only cold-start probe with a different candidate family set.}",
        r"\label{tab:policy-fixed-family-audit}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        "Method & "
        + " & ".join(str(setting["header"]) for setting in summary["settings"])
        + r" & Best/tie-best & Top-two & Mean rank \\",
        r"\midrule",
    ]
    for method in METHODS:
        row = method_rows[method]
        cell_parts = [method]
        for setting in summary["settings"]:
            value = float(setting["means"][method])
            bold = abs(value - float(setting["best_value"])) <= 1e-12
            underline = (
                setting["second_value"] is not None
                and abs(value - float(setting["second_value"])) <= 1e-12
                and not bold
            )
            cell_parts.append(_format_score(value, bold=bold, underline=underline))
        cell_parts.append(_format_count(int(row["best_or_tie_best"]), best_count))
        cell_parts.append(_format_count(int(row["top_two"]), max(int(item["top_two"]) for item in summary["methods"])))
        cell_parts.append(_format_rank(float(row["mean_rank"]), best_rank))
        lines.append(" & ".join(cell_parts) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Policy vs Fixed-Family Audit",
        "",
        summary["note"],
        "",
        "| Method | Best/tie-best | Top-two | Mean rank |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["methods"]:
        lines.append(
            f"| {row['method']} | {row['best_or_tie_best']} | {row['top_two']} | {row['mean_rank']:.2f} |"
        )
    lines.extend(["", "| Setting | TRACER | DLinear | Small-Transformer | Prefix |", "| --- | ---: | ---: | ---: | ---: |"])
    name_map = {
        "TRACER": "TRACER",
        "DLinear-Forecaster": "DLinear",
        "Small-Transformer-Forecaster": "Small-Transformer",
        "Prefix-Only-Retrieval + Fusion": "Prefix",
    }
    for setting in summary["settings"]:
        lines.append(
            "| "
            + setting["caption_name"]
            + " | "
            + " | ".join(f"{setting['means'][method]:.3f}" for method in name_map.keys())
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    summary = build_audit()
    json_path = RESULT_DIR / "policy_vs_fixed_family_audit.json"
    md_path = RESULT_DIR / "policy_vs_fixed_family_audit.md"
    tex_path = FIG_DIR / "tab_policy_vs_fixed_family_audit.tex"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(summary), encoding="utf-8")
    tex_path.write_text(build_table_tex(summary), encoding="utf-8")
    print(json_path)
    print(md_path)
    print(tex_path)


if __name__ == "__main__":
    main()
