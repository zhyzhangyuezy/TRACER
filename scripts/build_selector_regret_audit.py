from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"
FIXED_METHODS = [
    "DLinear-Forecaster",
    "Small-Transformer-Forecaster",
    "Prefix-Only-Retrieval + Fusion",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, *, bold: bool = False) -> str:
    body = f"{value:.3f}"
    if bold:
        return f"$\\mathbf{{{body}}}$"
    return f"${body}$"


def _fmt_gap(value: float, *, bold: bool = False) -> str:
    body = f"{value:+.3f}"
    if bold:
        return f"$\\mathbf{{{body}}}$"
    return f"${body}$"


def build_summary() -> dict[str, Any]:
    base = _load_json(RESULT_DIR / "policy_vs_fixed_family_audit.json")
    rows = []
    for setting in base["settings"]:
        fixed_values = {method: float(setting["means"][method]) for method in FIXED_METHODS}
        oracle_method, oracle_value = max(fixed_values.items(), key=lambda item: item[1])
        tracer_value = float(setting["means"]["TRACER"])
        gap = tracer_value - oracle_value
        rows.append(
            {
                "key": setting["key"],
                "benchmark": setting["caption_name"],
                "evidence": setting["evidence"],
                "adaptive_policy_auprc": tracer_value,
                "oracle_fixed_family_auprc": oracle_value,
                "oracle_fixed_family": oracle_method,
                "gap_to_oracle": gap,
                "adaptive_rank": float(setting["ranks"]["TRACER"]),
            }
        )
    gaps = [float(row["gap_to_oracle"]) for row in rows]
    return {
        "rows": rows,
        "summary": {
            "mean_gap_to_oracle": float(mean(gaps)),
            "min_gap_to_oracle": float(min(gaps)),
            "nonnegative_gap_count": int(sum(1 for gap in gaps if gap >= -1e-12)),
            "settings": len(rows),
        },
        "note": "The oracle single-family baseline is an ex-post upper bound over choosing one fixed DLinear, small-transformer, or prefix-retrieval family in the same released audit.",
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Selector regret audit",
        "",
        summary["note"],
        "",
        "| Benchmark | Evidence | Adaptive AUPRC | Oracle single-family | Oracle AUPRC | Gap |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {benchmark} | {evidence} | {adaptive:.3f} | {oracle_method} | {oracle:.3f} | {gap:+.3f} |".format(
                benchmark=row["benchmark"],
                evidence=row["evidence"],
                adaptive=row["adaptive_policy_auprc"],
                oracle_method=row["oracle_fixed_family"],
                oracle=row["oracle_fixed_family_auprc"],
                gap=row["gap_to_oracle"],
            )
        )
    s = summary["summary"]
    lines += [
        "",
        f"Mean gap to oracle single-family: {s['mean_gap_to_oracle']:+.3f}.",
        f"Nonnegative gaps: {s['nonnegative_gap_count']} / {s['settings']}.",
        f"Worst gap: {s['min_gap_to_oracle']:+.3f}.",
    ]
    (RESULT_DIR / "selector_regret_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(summary: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Selector-regret audit against an ex-post oracle single-family baseline. For each benchmark split, the oracle chooses one best fixed family among DLinear, small-transformer, and prefix-retrieval using the same released audit values. Positive gap means the deterministic adaptive policy exceeds this single-family oracle on that split; negative gap is regret.}",
        r"\label{tab:selector-regret-audit}",
        r"\maxtablewidth{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Benchmark & Evidence & Adaptive & Oracle single family & Oracle & Gap \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        gap = float(row["gap_to_oracle"])
        oracle_method = str(row["oracle_fixed_family"]).replace("_", r"\_")
        lines.append(
            "{benchmark} & {evidence} & {adaptive} & {oracle_method} & {oracle} & {gap} \\\\".format(
                benchmark=str(row["benchmark"]).replace("_", r"\_"),
                evidence=str(row["evidence"]).replace("_", r"\_"),
                adaptive=_fmt(float(row["adaptive_policy_auprc"]), bold=gap >= -1e-12),
                oracle_method=oracle_method,
                oracle=_fmt(float(row["oracle_fixed_family_auprc"]), bold=gap < -1e-12),
                gap=_fmt_gap(gap, bold=gap >= -1e-12),
            )
        )
    s = summary["summary"]
    lines += [
        r"\midrule",
        r"\multicolumn{5}{r}{Mean gap / nonnegative splits / worst gap} & "
        + f"${s['mean_gap_to_oracle']:+.3f}$ / ${s['nonnegative_gap_count']}/{s['settings']}$ / ${s['min_gap_to_oracle']:+.3f}$ \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    (FIG_DIR / "tab_selector_regret_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = build_summary()
    (RESULT_DIR / "selector_regret_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown(summary)
    write_latex(summary)
    print("Wrote outputs/results/selector_regret_audit.json")
    print("Wrote figures/tab_selector_regret_audit.tex")


if __name__ == "__main__":
    main()
