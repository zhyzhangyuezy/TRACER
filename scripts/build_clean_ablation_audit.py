from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"

SEEDS = [7, 13, 21]

ABLATIONS = [
    {
        "label": "Full TRACER core",
        "change": "all mechanisms on",
        "experiment": "r215_campaign_mem_decomp_modular_patch_atlasv2_public",
    },
    {
        "label": "No auxiliary horizon",
        "change": "auxiliary loss off",
        "experiment": "r218_campaign_mem_decomp_modular_patch_noaux_atlasv2_public",
    },
    {
        "label": "No hard negatives",
        "change": "contrastive hard weighting off",
        "experiment": "r268_tracer_clean_no_hardneg_atlasv2_public",
    },
    {
        "label": "No contrastive loss",
        "change": "contrastive objective off",
        "experiment": "r269_tracer_clean_no_contrastive_atlasv2_public",
    },
    {
        "label": "No correction",
        "change": "final score is base fusion",
        "experiment": "r270_tracer_clean_no_correction_atlasv2_public",
    },
    {
        "label": "Linear correction",
        "change": "unbounded linear logit gap",
        "experiment": "r271_tracer_clean_linear_correction_atlasv2_public",
    },
    {
        "label": "Forecast-only base gate",
        "change": "base gate fixed to 1",
        "experiment": "r272_tracer_clean_forecast_gate_atlasv2_public",
    },
    {
        "label": "Retrieval-only base gate",
        "change": "base gate fixed to 0",
        "experiment": "r273_tracer_clean_retrieval_gate_atlasv2_public",
    },
    {
        "label": "No route gates",
        "change": "shift/aggressive gates off",
        "experiment": "r274_tracer_clean_no_route_gates_atlasv2_public",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_values(experiment: str, split: str, metric: str) -> list[float]:
    values: list[float] = []
    missing: list[str] = []
    for seed in SEEDS:
        path = RESULT_DIR / f"{experiment}_seed{seed}.json"
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)).replace("\\", "/"))
            continue
        payload = _load_json(path)
        if split not in payload or metric not in payload[split]:
            missing.append(f"{path.name}:{split}.{metric}")
            continue
        values.append(float(payload[split][metric]))
    if missing:
        raise FileNotFoundError("Missing clean ablation inputs: " + ", ".join(missing))
    return values


def _summarize(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "values": [float(value) for value in values],
    }


def _fmt_mean_std(row: dict[str, Any], key: str) -> str:
    stats = row[key]
    return f"${stats['mean']:.3f} \\pm {stats['std']:.3f}$"


def _fmt_delta(value: float, *, bold: bool = False) -> str:
    body = f"{value:+.3f}"
    if bold:
        return f"$\\mathbf{{{body}}}$"
    return f"${body}$"


def build_summary() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in ABLATIONS:
        chron = _summarize(_metric_values(str(spec["experiment"]), "test", "AUPRC"))
        event = _summarize(_metric_values(str(spec["experiment"]), "test_event_disjoint", "AUPRC"))
        event_af = _summarize(_metric_values(str(spec["experiment"]), "test_event_disjoint", "Analog-Fidelity@5"))
        event_tte = _summarize(_metric_values(str(spec["experiment"]), "test_event_disjoint", "TTE-Err@1"))
        rows.append(
            {
                "label": spec["label"],
                "change": spec["change"],
                "experiment": spec["experiment"],
                "seeds": SEEDS,
                "chronology_auprc": chron,
                "heldout_family_auprc": event,
                "heldout_family_af5": event_af,
                "heldout_family_tte1": event_tte,
            }
        )

    full_event = rows[0]["heldout_family_auprc"]["mean"]
    full_chron = rows[0]["chronology_auprc"]["mean"]
    for row in rows:
        row["delta_heldout_family_auprc"] = float(row["heldout_family_auprc"]["mean"] - full_event)
        row["delta_chronology_auprc"] = float(row["chronology_auprc"]["mean"] - full_chron)
    return {
        "seeds": SEEDS,
        "rows": rows,
        "note": "All rows use real three-seed reruns or existing seed-matched result files with the same ATLASv2 public split and 20-epoch budget.",
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Clean mechanism ablation audit",
        "",
        summary["note"],
        "",
        "| Mechanism | Changed setting | Chron. AUPRC | HF AUPRC | Delta HF | HF AF@5 | HF TTE@1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {label} | {change} | {chron:.3f} +/- {chron_std:.3f} | {event:.3f} +/- {event_std:.3f} | {delta:+.3f} | {af:.1f} +/- {af_std:.1f} | {tte:.1f} +/- {tte_std:.1f} |".format(
                label=row["label"],
                change=row["change"],
                chron=row["chronology_auprc"]["mean"],
                chron_std=row["chronology_auprc"]["std"],
                event=row["heldout_family_auprc"]["mean"],
                event_std=row["heldout_family_auprc"]["std"],
                delta=row["delta_heldout_family_auprc"],
                af=row["heldout_family_af5"]["mean"],
                af_std=row["heldout_family_af5"]["std"],
                tte=row["heldout_family_tte1"]["mean"],
                tte_std=row["heldout_family_tte1"]["std"],
            )
        )
    (RESULT_DIR / "clean_ablation_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(summary: dict[str, Any]) -> None:
    best_event = max(row["heldout_family_auprc"]["mean"] for row in summary["rows"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Clean mechanism ablation on ATLASv2 public using real three-seed reruns with the same split and 20-epoch training budget. The table changes one mechanism at a time from the TRACER core except where the row explicitly tests an alternative correction transform or fixed base gate. HF denotes the held-out-family split.}",
        r"\label{tab:clean-mechanism-ablation}",
        r"\maxtablewidth{",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Mechanism & Changed setting & Chron. AUPRC & HF AUPRC & $\Delta$ HF & HF AF@5 & HF TTE@1 \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        event_cell = _fmt_mean_std(row, "heldout_family_auprc")
        if abs(row["heldout_family_auprc"]["mean"] - best_event) <= 1e-12:
            event_cell = f"$\\mathbf{{{row['heldout_family_auprc']['mean']:.3f}}} \\pm {row['heldout_family_auprc']['std']:.3f}$"
        lines.append(
            "{label} & {change} & {chron} & {event} & {delta} & {af:.1f} $\\pm$ {af_std:.1f} & {tte:.1f} $\\pm$ {tte_std:.1f} \\\\".format(
                label=str(row["label"]).replace("_", r"\_"),
                change=str(row["change"]).replace("_", r"\_"),
                chron=_fmt_mean_std(row, "chronology_auprc"),
                event=event_cell,
                delta=_fmt_delta(float(row["delta_heldout_family_auprc"])),
                af=row["heldout_family_af5"]["mean"],
                af_std=row["heldout_family_af5"]["std"],
                tte=row["heldout_family_tte1"]["mean"],
                tte_std=row["heldout_family_tte1"]["std"],
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    (FIG_DIR / "tab_clean_mechanism_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = build_summary()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "clean_ablation_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown(summary)
    write_latex(summary)
    print("Wrote outputs/results/clean_ablation_audit.json")
    print("Wrote figures/tab_clean_mechanism_ablation.tex")


if __name__ == "__main__":
    main()
