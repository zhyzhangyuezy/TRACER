from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"


SETTINGS = [
    {
        "setting": "ATLASv2 chronology",
        "dataset_key": "atlasv2_public",
        "tabular_split": "chronology",
        "policy_key": "atlasv2_chrono",
    },
    {
        "setting": "ATLASv2 held-out-family",
        "dataset_key": "atlasv2_public",
        "tabular_split": "event_disjoint",
        "policy_key": "atlasv2_event_20seed",
    },
    {
        "setting": "AIT-ADS chronology",
        "dataset_key": "ait_ads_public",
        "tabular_split": "chronology",
        "policy_key": "ait_ads_chrono_20seed",
    },
    {
        "setting": "AIT-ADS held-out-scenario",
        "dataset_key": "ait_ads_public",
        "tabular_split": "event_disjoint",
        "policy_key": "ait_ads_event_20seed",
    },
    {
        "setting": "ATLAS-Raw chronology",
        "dataset_key": "atlas_raw_public",
        "tabular_split": "chronology",
        "policy_key": "atlas_raw_chrono",
    },
    {
        "setting": "ATLAS-Raw held-out-family",
        "dataset_key": "atlas_raw_public",
        "tabular_split": "event_disjoint",
        "policy_key": "atlas_raw_event",
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_score(value: float) -> str:
    return f"{value:.3f}"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _method_cell(method: str, dev_value: float, test_value: float) -> str:
    return f"{_latex_escape(method)} ({_fmt_score(dev_value)} $\\rightarrow$ {_fmt_score(test_value)})"


def _policy_lookup(policy_payload: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in policy_payload["settings"]:
        values[str(row["key"])] = float(row["means"]["TRACER"])
    return values


def _tabular_rows(tabular_payload: dict[str, Any], dataset_key: str) -> list[dict[str, Any]]:
    rows = []
    for recipe in tabular_payload["recipes"]:
        dataset = recipe["datasets"][dataset_key]
        summary = dataset["summary"]
        rows.append(
            {
                "key": recipe["key"],
                "label": recipe["label"],
                "dev": float(summary["dev"]["AUPRC"]["mean"]),
                "chronology": float(summary["chronology"]["AUPRC"]["mean"]),
                "event_disjoint": float(summary["event_disjoint"]["AUPRC"]["mean"]),
            }
        )
    return rows


def _build_tabular_gate() -> dict[str, Any]:
    tabular_payload = _load_json(RESULT_DIR / "tabular_baseline_audit.json")
    policy_values = _policy_lookup(_load_json(RESULT_DIR / "policy_vs_fixed_family_audit.json"))

    rows: list[dict[str, Any]] = []
    for spec in SETTINGS:
        candidates = _tabular_rows(tabular_payload, str(spec["dataset_key"]))
        dev_best = max(candidates, key=lambda row: row["dev"])
        oracle_best = max(candidates, key=lambda row: row[str(spec["tabular_split"])])
        policy = float(policy_values[str(spec["policy_key"])])
        dev_best_test = float(dev_best[str(spec["tabular_split"])])
        oracle_test = float(oracle_best[str(spec["tabular_split"])])
        rows.append(
            {
                "setting": spec["setting"],
                "released_policy_auprc": policy,
                "dev_best_tabular": dev_best["label"],
                "dev_best_dev_auprc": float(dev_best["dev"]),
                "dev_best_test_auprc": dev_best_test,
                "ex_post_best_tabular": oracle_best["label"],
                "ex_post_best_test_auprc": oracle_test,
                "dev_gated_delta": dev_best_test - policy,
                "oracle_delta": oracle_test - policy,
                "would_replace_under_dev_gate": bool(dev_best_test > policy),
            }
        )
    return {
        "audit": "Policy-v2 tabular admission audit",
        "selection_rule": "Choose the tabular recipe with the largest development AUPRC within each dataset, then compare its held-out target split to the released policy. Ex-post best rows are reported only as test-only oracle diagnostics.",
        "rows": rows,
    }


def _write_tabular_gate_tex(payload: dict[str, Any]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Policy-v2 tabular-admission audit. The dev-best tabular column applies a train/dev-only maintenance rule to the stronger classical controls, while the ex-post column reports the best tabular row on the held-out target split only as an oracle diagnostic. This table tests whether the FlatPrefix-Logistic chronology result should be silently folded into the released route table.}",
        "\\label{tab:policy-v2-tabular-gate}",
        "\\maxtablewidth{",
        "\\begin{tabular}{@{}p{3.1cm}p{1.8cm}p{4.1cm}p{3.0cm}p{3.0cm}@{}}",
        "\\toprule",
        "Setting & Released policy & Dev-best tabular (dev $\\rightarrow$ target) & Ex-post best tabular & Maintenance read \\\\",
        "\\midrule",
    ]
    for row in payload["rows"]:
        maintenance = "replace" if row["would_replace_under_dev_gate"] else "keep policy"
        if row["oracle_delta"] > 0 and not row["would_replace_under_dev_gate"]:
            maintenance = "keep policy; oracle gain is not dev-supported"
        cells = [
            _latex_escape(str(row["setting"])),
            f"${_fmt_score(float(row['released_policy_auprc']))}$",
            _method_cell(
                str(row["dev_best_tabular"]),
                float(row["dev_best_dev_auprc"]),
                float(row["dev_best_test_auprc"]),
            ),
            f"{_latex_escape(str(row['ex_post_best_tabular']))} ({_fmt_score(float(row['ex_post_best_test_auprc']))})",
            _latex_escape(maintenance),
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]
    (FIG_DIR / "tab_policy_v2_tabular_gate_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tabular_gate_md(payload: dict[str, Any]) -> None:
    lines = [
        "# Policy-v2 tabular admission audit",
        "",
        payload["selection_rule"],
        "",
        "| Setting | Released policy | Dev-best tabular | Dev AUPRC | Target AUPRC | Ex-post best | Ex-post target | Read |",
        "|---|---:|---|---:|---:|---|---:|---|",
    ]
    for row in payload["rows"]:
        read = "replace" if row["would_replace_under_dev_gate"] else "keep policy"
        if row["oracle_delta"] > 0 and not row["would_replace_under_dev_gate"]:
            read = "keep policy; oracle gain not dev-supported"
        lines.append(
            "| {setting} | {policy:.3f} | {dev_best} | {dev:.3f} | {target:.3f} | {oracle} | {oracle_target:.3f} | {read} |".format(
                setting=row["setting"],
                policy=float(row["released_policy_auprc"]),
                dev_best=row["dev_best_tabular"],
                dev=float(row["dev_best_dev_auprc"]),
                target=float(row["dev_best_test_auprc"]),
                oracle=row["ex_post_best_tabular"],
                oracle_target=float(row["ex_post_best_test_auprc"]),
                read=read,
            )
        )
    (RESULT_DIR / "policy_v2_tabular_gate_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_route_signature_table() -> None:
    payload = _load_json(RESULT_DIR / "tracer_route_margin_audit.json")
    rows = payload["rows"]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Route-signature provenance for the released \\texttt{tracer\\_adaptive} rule table. All quantities are computed before held-out evaluation from train/dev data and the declared deployment profile. The table complements the rule predicates by showing the actual regime signatures that activate each route.}",
        "\\label{tab:route-signature-provenance}",
        "\\maxtablewidth{",
        "\\begin{tabular}{@{}llrrrrrll@{}}",
        "\\toprule",
        "Dataset & Profile & $r_{+}^{\\mathrm{train}}$ & $n_{+}^{\\mathrm{dev}}$ & $F^{\\mathrm{train}}$ & $F_{+}^{\\mathrm{train}}$ & $\\bar{\\delta}_{2}^{\\mathrm{train}}$ / $\\rho_{\\mathrm{peak}}^{\\mathrm{train}}$ & Regime & Mode \\\\",
        "\\midrule",
    ]
    for row in rows:
        mode = str(row["resolved_model_type"]).replace("campaign_mem_decomp_modular", "decomp TRACER").replace("campaign_mem_v3", "retrieval TRACER")
        lines.append(
            "{dataset} & {profile} & {rate:.4f} & {devpos:d} & {families:d} & {posfam:d} & {diff2:.3f} / {peak:.2f} & {regime} & {mode} \\\\".format(
                dataset=_latex_escape(str(row["dataset"])),
                profile=_latex_escape(str(row["objective"])),
                rate=float(row["positive_rate"]),
                devpos=int(row["dev_positive_count"]),
                families=int(row["family_count"]),
                posfam=int(row["positive_family_count"]),
                diff2=float(row["diff2_abs_mean"]),
                peak=float(row["peak_ratio"]),
                regime=_latex_escape(str(row["regime"])),
                mode=_latex_escape(mode),
            )
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]
    (FIG_DIR / "tab_route_signature_provenance.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = _build_tabular_gate()
    (RESULT_DIR / "policy_v2_tabular_gate_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_tabular_gate_tex(payload)
    _write_tabular_gate_md(payload)
    _build_route_signature_table()
    print(f"Wrote {RESULT_DIR / 'policy_v2_tabular_gate_audit.json'}")
    print(f"Wrote {FIG_DIR / 'tab_policy_v2_tabular_gate_audit.tex'}")
    print(f"Wrote {FIG_DIR / 'tab_route_signature_provenance.tex'}")


if __name__ == "__main__":
    main()
