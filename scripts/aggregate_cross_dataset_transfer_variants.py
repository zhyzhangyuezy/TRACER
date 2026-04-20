from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
RUN_DIR = RESULT_DIR / "cross_dataset_transfer"
FIG_DIR = ROOT / "figures"

METHODS = {
    "dlinear": "DLinear",
    "prefixonlyonly": "Prefix-Only",
    "tracer_adaptive": "TRACER",
}

DATASET_LABELS = {
    "atlasv2": "ATLASv2",
    "aitads": "AIT-ADS",
}

FILENAME_RE = re.compile(
    r"r330_transfer_(dlinear|prefixonlyonly|tracer_adaptive)_(atlasv2|aitads)_to_(atlasv2|aitads)_shot(\d+)_seed(\d+)(?:_rep(\d+))?\.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return finite[0], 0.0
    return float(mean(finite)), float(pstdev(finite))


def _fmt_pm(avg: float, std: float) -> str:
    if not np.isfinite(avg):
        return "--"
    return f"{avg:.3f} $\\pm$ {std:.3f}"


def _fmt_md(avg: float, std: float) -> str:
    if not np.isfinite(avg):
        return "--"
    return f"{avg:.3f} +/- {std:.3f}"


def collect() -> dict[str, Any]:
    runs = []
    for path in sorted(RUN_DIR.glob("r330_transfer_*.json")):
        match = FILENAME_RE.fullmatch(path.name)
        if not match:
            continue
        method_key, source, target, shot_raw, seed_raw, repeat_raw = match.groups()
        payload = _load_json(path)
        repeat = int(repeat_raw) if repeat_raw else 1
        variant = "standard" if repeat == 1 else f"target-replay{repeat}"
        runs.append(
            {
                "source": source,
                "target": target,
                "setting": f"{DATASET_LABELS[source]} -> {DATASET_LABELS[target]}",
                "shot": int(shot_raw),
                "seed": int(seed_raw),
                "repeat": repeat,
                "variant": variant,
                "method": METHODS[method_key],
                "AUPRC": float(payload["test"]["AUPRC"]),
                "AUROC": float(payload["test"].get("AUROC", float("nan"))),
                "Brier": float(payload["test"].get("Brier", float("nan"))),
                "AF@5": float(payload["test"].get("Analog-Fidelity@5", float("nan"))),
                "route": str(payload.get("auto_component_policy", {}).get("regime", "")),
            }
        )

    grouped: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        key = (run["source"], run["target"], run["shot"], run["variant"], run["method"])
        grouped.setdefault(key, []).append(run)

    method_rows = []
    for (source, target, shot, variant, method), items in sorted(grouped.items()):
        auprc, auprc_std = _summarize([item["AUPRC"] for item in items])
        auroc, auroc_std = _summarize([item["AUROC"] for item in items])
        brier, brier_std = _summarize([item["Brier"] for item in items])
        af, af_std = _summarize([item["AF@5"] for item in items])
        routes = sorted({item["route"] for item in items if item["route"]})
        method_rows.append(
            {
                "source": source,
                "target": target,
                "setting": f"{DATASET_LABELS[source]} -> {DATASET_LABELS[target]}",
                "shot": shot,
                "variant": variant,
                "method": method,
                "seeds": sorted(item["seed"] for item in items),
                "AUPRC": auprc,
                "AUPRC_std": auprc_std,
                "AUROC": auroc,
                "AUROC_std": auroc_std,
                "Brier": brier,
                "Brier_std": brier_std,
                "AF@5": af,
                "AF@5_std": af_std,
                "routes": routes,
            }
        )

    compact: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in method_rows:
        key = (row["source"], row["target"], row["shot"], row["variant"])
        base = compact.setdefault(
            key,
            {
                "source": row["source"],
                "target": row["target"],
                "setting": row["setting"],
                "shot": row["shot"],
                "variant": row["variant"],
                "methods": {},
            },
        )
        base["methods"][row["method"]] = row

    compact_rows = []
    for row in sorted(compact.values(), key=lambda item: (item["source"], item["target"], item["shot"], item["variant"])):
        methods = row["methods"]
        best_method = max(methods.items(), key=lambda item: item[1]["AUPRC"])[0]
        tracer = methods.get("TRACER")
        compact_rows.append(
            {
                **row,
                "best_method": best_method,
                "best_AUPRC": methods[best_method]["AUPRC"],
                "best_AUPRC_std": methods[best_method]["AUPRC_std"],
                "tracer_route": ", ".join(tracer["routes"]) if tracer else "--",
            }
        )

    return {
        "audit": "Cross-dataset transfer variants audit",
        "runs": runs,
        "method_rows": method_rows,
        "compact_rows": compact_rows,
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Cross-dataset transfer variants audit",
        "",
        "Standard uses source train/dev plus optional target support once. Target-replay10 repeats the sampled target train/dev support windows ten times during training/model selection; target test windows are never used as support.",
        "",
        "| Setting | Variant | Shot | DLinear | Prefix | TRACER | Best | TRACER route |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary["compact_rows"]:
        methods = row["methods"]
        dlinear = methods.get("DLinear", {})
        prefix = methods.get("Prefix-Only", {})
        tracer = methods.get("TRACER", {})
        lines.append(
            "| {setting} | {variant} | {shot} | {dlinear} | {prefix} | {tracer} | {best} | {route} |".format(
                setting=row["setting"],
                variant=row["variant"],
                shot=row["shot"],
                dlinear=_fmt_md(dlinear.get("AUPRC", float("nan")), dlinear.get("AUPRC_std", float("nan"))),
                prefix=_fmt_md(prefix.get("AUPRC", float("nan")), prefix.get("AUPRC_std", float("nan"))),
                tracer=_fmt_md(tracer.get("AUPRC", float("nan")), tracer.get("AUPRC_std", float("nan"))),
                best=f"{row['best_method']} ({row['best_AUPRC']:.3f})",
                route=row["tracer_route"] or "--",
            )
        )
    (RESULT_DIR / "cross_dataset_transfer_variants_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(summary: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Cross-dataset transfer and few-shot adaptation audit. Standard uses source train/dev plus optional target support once. Target-replay10 repeats sampled target-domain support windows ten times during training and model selection, while never using target test windows as support. Values are AUPRC mean $\pm$ standard deviation over three seeds.}",
        r"\label{tab:cross-dataset-transfer}",
        r"\maxtablewidth{",
        r"\begin{tabular}{llrllll}",
        r"\toprule",
        r"Setting & Support & Shot & DLinear & Prefix & TRACER & Best / route \\",
        r"\midrule",
    ]
    previous = None
    for row in summary["compact_rows"]:
        setting = row["setting"] if row["setting"] != previous else ""
        methods = row["methods"]
        dlinear = methods.get("DLinear", {})
        prefix = methods.get("Prefix-Only", {})
        tracer = methods.get("TRACER", {})
        support = "standard" if row["variant"] == "standard" else "replay10"
        best = f"{row['best_method']}; {row['tracer_route']}".replace("_", r"\_")
        lines.append(
            "{setting} & {support} & {shot} & {dlinear} & {prefix} & {tracer} & {best} \\\\".format(
                setting=setting,
                support=support,
                shot=row["shot"],
                dlinear=_fmt_pm(dlinear.get("AUPRC", float("nan")), dlinear.get("AUPRC_std", float("nan"))),
                prefix=_fmt_pm(prefix.get("AUPRC", float("nan")), prefix.get("AUPRC_std", float("nan"))),
                tracer=_fmt_pm(tracer.get("AUPRC", float("nan")), tracer.get("AUPRC_std", float("nan"))),
                best=best,
            )
        )
        previous = row["setting"]
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    (FIG_DIR / "tab_cross_dataset_transfer.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = collect()
    (RESULT_DIR / "cross_dataset_transfer_variants_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)
    write_latex(summary)
    print("Wrote outputs/results/cross_dataset_transfer_variants_audit.json")
    print("Wrote figures/tab_cross_dataset_transfer.tex")


if __name__ == "__main__":
    main()
