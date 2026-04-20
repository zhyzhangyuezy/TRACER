from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"

SEEDS_3 = [7, 13, 21]
RETRIEVAL_ROUTES = {
    "campaign_mem",
    "campaign_mem_v2",
    "campaign_mem_v3",
    "campaign_mem_v4",
    "campaign_mem_v5",
    "campaign_mem_decomp_modular",
    "campaign_mem_final",
    "prefix_retrieval",
}

SETTINGS = [
    {
        "label": "ATLASv2 chronology",
        "experiment": "r239_tracer_adaptive_chronology_atlasv2_public",
        "split": "test",
        "auprc_source": ("policy", "atlasv2_chrono"),
    },
    {
        "label": "ATLASv2 held-out-family",
        "experiment": "r240_tracer_adaptive_event_atlasv2_public",
        "split": "test_event_disjoint",
        "auprc_source": ("policy", "atlasv2_event_20seed"),
    },
    {
        "label": "AIT-ADS chronology",
        "experiment": "r241_tracer_adaptive_ait_ads_public",
        "split": "test",
        "auprc_source": ("policy", "ait_ads_chrono_20seed"),
    },
    {
        "label": "AIT-ADS held-out-scenario",
        "experiment": "r242_tracer_adaptive_event_ait_ads_public",
        "split": "test_event_disjoint",
        "auprc_source": ("policy", "ait_ads_event_20seed"),
    },
    {
        "label": "ATLAS-Raw chronology",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "split": "test",
        "auprc_source": ("policy", "atlas_raw_chrono"),
    },
    {
        "label": "ATLAS-Raw event-disjoint",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "split": "test_event_disjoint",
        "auprc_source": ("policy", "atlas_raw_event"),
    },
    {
        "label": "Synthetic CAM-LDS",
        "experiment": "r244_tracer_adaptive_synthetic_cam_lds",
        "split": "test",
        "auprc_source": ("seeded", "r244_tracer_adaptive_synthetic_cam_lds"),
    },
    {
        "label": "Workbook probe",
        "experiment": "r245_tracer_adaptive_atlasv2_workbook",
        "split": "test",
        "auprc_source": ("seeded", "r245_tracer_adaptive_atlasv2_workbook"),
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_payload(experiment: str, seed: int = 7) -> dict[str, Any]:
    return _load_json(RESULT_DIR / f"{experiment}_seed{seed}.json")


def _seeded_metric(experiment: str, split: str = "test", metric: str = "AUPRC") -> dict[str, Any]:
    values = []
    for seed in SEEDS_3:
        payload = _seed_payload(experiment, seed)
        values.append(float(payload[split][metric]))
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)),
        "n": len(values),
    }


def _policy_metric(key: str) -> dict[str, Any]:
    policy = _load_json(RESULT_DIR / "policy_vs_fixed_family_audit.json")
    for row in policy["settings"]:
        if row["key"] == key:
            return {
                "mean": float(row["means"]["TRACER"]),
                "std": float(row["stds"]["TRACER"]),
                "n": int(str(row["evidence"]).split("_")[-1].replace("seeded", "") or 0)
                if str(row["evidence"]).startswith("seeded_")
                else None,
            }
    raise KeyError(key)


def _auprc(source: tuple[str, str], split: str) -> dict[str, Any]:
    kind, key = source
    if kind == "policy":
        return _policy_metric(key)
    if kind == "seeded":
        return _seeded_metric(key, split)
    raise ValueError(f"Unknown AUPRC source: {source}")


def _is_retrieval_route(route: str) -> bool:
    return route in RETRIEVAL_ROUTES or route.startswith("campaign_mem")


def _bank_mib(stats: dict[str, Any], model: dict[str, Any], retrieval_active: bool) -> float:
    if not retrieval_active:
        return 0.0
    rows = int(stats["size"])
    embedding_dim = int(model.get("embedding_dim", 128))
    return rows * embedding_dim * 4 / (1024 * 1024)


def build_summary() -> dict[str, Any]:
    rows = []
    for spec in SETTINGS:
        payload = _seed_payload(str(spec["experiment"]), 7)
        policy = payload.get("auto_component_policy", {})
        train_stats = policy.get("train_stats", {})
        dev_stats = policy.get("dev_stats", {})
        route = str(policy.get("resolved_model_type", payload.get("model", {}).get("type", "unknown")))
        retrieval_active = _is_retrieval_route(route)
        metric = _auprc(spec["auprc_source"], str(spec["split"]))
        rows.append(
            {
                "label": spec["label"],
                "experiment": spec["experiment"],
                "split": spec["split"],
                "regime": policy.get("regime", "n/a"),
                "route": route,
                "retrieval_active": retrieval_active,
                "train_windows": int(train_stats.get("size", 0)),
                "train_positives": int(train_stats.get("positive_count", 0)),
                "train_positive_rate": float(train_stats.get("positive_rate", 0.0)),
                "dev_positives": int(dev_stats.get("positive_count", 0)),
                "family_count": int(train_stats.get("family_count", 0)),
                "positive_family_count": int(train_stats.get("positive_family_count", 0)),
                "auprc": metric,
                "bank_rows": int(train_stats.get("size", 0)) if retrieval_active else 0,
                "bank_mib": _bank_mib(train_stats, payload.get("model", {}), retrieval_active),
            }
        )
    return {
        "rows": rows,
        "note": "The table is computed from released seed result JSONs and train/dev policy statistics; bank footprint is the dense float32 embedding-bank estimate for retrieval-active routes.",
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Cold-start and cost audit",
        "",
        summary["note"],
        "",
        "| Setting | Train +/N | Dev + | Families +/N | Regime | Route | Retrieval | AUPRC | Bank MiB |",
        "|---|---:|---:|---:|---|---|---|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {label} | {pos}/{n} | {dev_pos} | {pf}/{fam} | {regime} | {route} | {retrieval} | {auprc:.3f} +/- {std:.3f} | {mib:.2f} |".format(
                label=row["label"],
                pos=row["train_positives"],
                n=row["train_windows"],
                dev_pos=row["dev_positives"],
                pf=row["positive_family_count"],
                fam=row["family_count"],
                regime=row["regime"],
                route=row["route"],
                retrieval="yes" if row["retrieval_active"] else "no",
                auprc=row["auprc"]["mean"],
                std=row["auprc"]["std"],
                mib=row["bank_mib"],
            )
        )
    (RESULT_DIR / "cold_start_cost_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_latex(summary: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Cold-start and deployment-cost audit for the deterministic adaptive policy. Counts and regimes are read from the released seed result JSONs; AUPRC values come from the corresponding seeded or 20-seed audits. The bank footprint is a dense float32 embedding-bank estimate for retrieval-active routes and is zero for fallback routes.}",
        r"\label{tab:cold-start-cost-audit}",
        r"\maxtablewidth{",
        r"\begin{tabular}{lrrrllcc}",
        r"\toprule",
        r"Setting & Train +/N & Dev + & Fam. +/N & Regime & Route & AUPRC & Bank MiB \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        route = _latex_escape(str(row["route"]))
        regime = _latex_escape(str(row["regime"]))
        if not row["retrieval_active"]:
            route = route + r" (fallback)"
        lines.append(
            "{label} & {pos}/{n} & {dev_pos} & {pf}/{fam} & {regime} & {route} & ${auprc:.3f} \\pm {std:.3f}$ & ${mib:.2f}$ \\\\".format(
                label=_latex_escape(str(row["label"])),
                pos=row["train_positives"],
                n=row["train_windows"],
                dev_pos=row["dev_positives"],
                pf=row["positive_family_count"],
                fam=row["family_count"],
                regime=regime,
                route=route,
                auprc=row["auprc"]["mean"],
                std=row["auprc"]["std"],
                mib=row["bank_mib"],
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    (FIG_DIR / "tab_cold_start_cost_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = build_summary()
    (RESULT_DIR / "cold_start_cost_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown(summary)
    write_latex(summary)
    print("Wrote outputs/results/cold_start_cost_audit.json")
    print("Wrote figures/tab_cold_start_cost_audit.tex")


if __name__ == "__main__":
    main()
