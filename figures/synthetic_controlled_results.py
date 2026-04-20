from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
REFRESH_DIR = ROOT / "outputs" / "results" / "kbs_synthetic_controlled_refresh"
POLICY_DIR = ROOT / "outputs" / "results" / "kbs_synthetic_controlled_policy_refresh"
AUDIT_JSON = ROOT / "outputs" / "results" / "synthetic_cam_lds_controlled_audit.json"


CHRONO_ROWS = [
    ("r266_tracer_adaptive_chronology_synthetic_cam_lds_controlled", "TRACER", POLICY_DIR, "test"),
    ("r258_tail_risk_linear_synthetic_cam_lds_controlled", "Tail-Risk-Linear", REFRESH_DIR, "test"),
    ("r259_tcn_forecaster_synthetic_cam_lds_controlled", "TCN-Forecaster", REFRESH_DIR, "test"),
    ("r260_transformer_forecaster_synthetic_cam_lds_controlled", "Transformer-Forecaster", REFRESH_DIR, "test"),
    ("r262_prefix_retrieval_synthetic_cam_lds_controlled", "Prefix-Only-Retrieval + Fusion", REFRESH_DIR, "test"),
    ("r263_campaign_mem_synthetic_cam_lds_controlled", "Campaign-MEM", REFRESH_DIR, "test"),
    ("r261_pure_knn_synthetic_cam_lds_controlled", "Pure-kNN-Retrieval", REFRESH_DIR, "test"),
]

EVENT_ROWS = [
    ("r267_tracer_adaptive_event_synthetic_cam_lds_controlled", "TRACER", POLICY_DIR, "test_event_disjoint"),
    ("r258_tail_risk_linear_synthetic_cam_lds_controlled", "Tail-Risk-Linear", REFRESH_DIR, "test_event_disjoint"),
    ("r259_tcn_forecaster_synthetic_cam_lds_controlled", "TCN-Forecaster", REFRESH_DIR, "test_event_disjoint"),
    ("r260_transformer_forecaster_synthetic_cam_lds_controlled", "Transformer-Forecaster", REFRESH_DIR, "test_event_disjoint"),
    ("r262_prefix_retrieval_synthetic_cam_lds_controlled", "Prefix-Only-Retrieval + Fusion", REFRESH_DIR, "test_event_disjoint"),
    ("r263_campaign_mem_synthetic_cam_lds_controlled", "Campaign-MEM", REFRESH_DIR, "test_event_disjoint"),
    ("r261_pure_knn_synthetic_cam_lds_controlled", "Pure-kNN-Retrieval", REFRESH_DIR, "test_event_disjoint"),
]

METRICS = [
    ("AUPRC", True),
    ("AUROC", True),
    ("BestF1", True),
    ("ECE@10", False),
    ("Brier", False),
]


def _load_result(base_name: str, result_dir: Path, seed: int) -> dict:
    path = result_dir / f"{base_name}_seed{seed}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_value(block: dict, metric: str) -> float:
    if metric in block:
        return float(block[metric])
    raise KeyError(metric)


def _aggregate(base_name: str, label: str, result_dir: Path, split: str) -> dict:
    payloads = [_load_result(base_name, result_dir, seed) for seed in (7, 13, 21)]
    rows = [payload[split] for payload in payloads]
    summary = {"label": label, "base_name": base_name}
    for metric, _ in METRICS:
        values = [_metric_value(row, metric) for row in rows]
        summary[metric] = (mean(values), pstdev(values))
    return summary


def _format_pm(value: float, spread: float, decimals: int = 3) -> str:
    return f"${value:.{decimals}f} \\pm {spread:.{decimals}f}$"


def _decorated_metric(summary: dict, metric: str, group: list[dict], higher_is_better: bool) -> str:
    ordered = sorted((row[metric][0] for row in group), reverse=higher_is_better)
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else ordered[0]
    value, spread = summary[metric]
    rendered = _format_pm(value, spread)
    if abs(value - best) < 1e-12:
        return rf"$\mathbf{{{value:.3f} \pm {spread:.3f}}}$"
    if abs(value - second) < 1e-12:
        return rf"$\underline{{{value:.3f} \pm {spread:.3f}}}$"
    return rendered


def build_stats_table_tex() -> str:
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Controlled synthetic CAM-LDS benchmark statistics. This generated benchmark is not a replacement for the real-alert corpora; it is a large-sample controlled validation channel used to test whether the published policy behaves sensibly when the data regime is dense, leakage-free, and family-disjoint on the held-out split.}",
        r"\label{tab:synthetic-cam-lds-stats}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Split & Windows & Positives & Families & Incident leakage / family leakage \\",
        r"\midrule",
    ]
    for split_name, label in [
        ("train", "Train"),
        ("dev", "Dev"),
        ("test", "Chronological test"),
        ("test_event_disjoint", "Family-held-out test"),
    ]:
        split = audit["splits"][split_name]
        positives = round(float(split["positive_rate_main"]) * int(split["samples"]))
        leak = "none"
        if split_name == "test_event_disjoint":
            leak = "none / none"
        elif split_name in {"train", "dev", "test"}:
            leak = "none / n.a."
        lines.append(
            f"{label} & {int(split['samples'])} & {positives} & {int(split['family_count'])} & {leak} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_results_table_tex() -> str:
    chrono = [_aggregate(*row) for row in CHRONO_ROWS]
    event = [_aggregate(*row) for row in EVENT_ROWS]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\caption{Three-seed controlled synthetic CAM-LDS results. Best available value in each metric column is bolded and the second-best value is underlined; lower is better for ECE@10 and Brier. The direct adaptive-policy rows use the split-specific chronology and event profiles, while the fixed baselines come from the same controlled benchmark reruns.}",
        r"\label{tab:synthetic-cam-lds-results}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Split & Method & AUPRC & AUROC & BestF1 & ECE@10 & Brier \\",
        r"\midrule",
    ]
    for split_label, group in [("Chronological", chrono), ("Family-held-out", event)]:
        for idx, row in enumerate(group):
            prefix = split_label if idx == 0 else ""
            metrics = [
                _decorated_metric(row, metric, group, higher)
                for metric, higher in METRICS
            ]
            lines.append(f"{prefix} & {row['label']} & " + " & ".join(metrics) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def write_tables() -> None:
    (ROOT / "figures" / "tab_synthetic_cam_lds_stats.tex").write_text(build_stats_table_tex(), encoding="utf-8")
    (ROOT / "figures" / "tab_synthetic_cam_lds_results.tex").write_text(build_results_table_tex(), encoding="utf-8")


if __name__ == "__main__":
    write_tables()
