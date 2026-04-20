from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"
AUDIT_DIR = RESULT_DIR / "audits"

SEEDS = [
    7,
    13,
    21,
    34,
    55,
    89,
    144,
    233,
    377,
    610,
    987,
    1597,
    2584,
    4181,
    6765,
    10946,
    17711,
    28657,
    46368,
    75025,
]
BUDGETS = [0.05, 0.10, 0.20]
PRIMARY_BUDGET = 0.05
STABLE_RUN = 2

SETTINGS = [
    {
        "setting": "ATLASv2 held-out-family",
        "export_dir": "public_event_significance_seed_exports",
        "methods": {
            "Adaptive": "r240_tracer_adaptive_event_atlasv2_public_test_event_disjoint_seed{seed}.json",
            "DLinear": "r020_dlinear_forecaster_atlasv2_public_event_disjoint_test_event_disjoint_seed{seed}.json",
            "LSTM": "r018_lstm_forecaster_atlasv2_public_event_disjoint_test_event_disjoint_seed{seed}.json",
            "Prefix-Only": "r010_prefix_retrieval_atlasv2_public_event_disjoint_test_event_disjoint_seed{seed}.json",
        },
    },
    {
        "setting": "AIT-ADS chronology",
        "export_dir": "ait_ads_chronology_significance_seed_exports",
        "methods": {
            "Adaptive": "r241_tracer_adaptive_ait_ads_public_test_seed{seed}.json",
            "DLinear": "r068_dlinear_forecaster_ait_ads_public_test_seed{seed}.json",
            "Transformer": "r070_transformer_forecaster_ait_ads_public_test_seed{seed}.json",
            "Prefix-Only": "r071_prefix_retrieval_ait_ads_public_test_seed{seed}.json",
        },
    },
    {
        "setting": "AIT-ADS scenario-held-out",
        "export_dir": "ait_ads_event_significance_seed_exports",
        "methods": {
            "Adaptive": "r242_tracer_adaptive_event_ait_ads_public_test_event_disjoint_seed{seed}.json",
            "DLinear": "r068_dlinear_forecaster_ait_ads_public_test_event_disjoint_seed{seed}.json",
            "TCN": "r069_tcn_forecaster_ait_ads_public_test_event_disjoint_seed{seed}.json",
            "Prefix-Only": "r071_prefix_retrieval_ait_ads_public_test_event_disjoint_seed{seed}.json",
        },
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _top_budget_mask(scores: np.ndarray, budget: float) -> np.ndarray:
    k = max(1, int(math.ceil(float(scores.shape[0]) * float(budget))))
    order = np.argsort(-scores, kind="mergesort")[:k]
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[order] = True
    return mask


def _episode_indices(selected: np.ndarray, incident_id: np.ndarray, timestamp: np.ndarray) -> list[np.ndarray]:
    episodes: list[np.ndarray] = []
    for incident in np.unique(incident_id):
        idx = np.flatnonzero(incident_id == incident)
        order = idx[np.argsort(timestamp[idx], kind="mergesort")]
        current: list[int] = []
        previous_position: int | None = None
        for position, original_index in enumerate(order):
            if selected[original_index]:
                if previous_position is None or position == previous_position + 1:
                    current.append(int(original_index))
                else:
                    if current:
                        episodes.append(np.asarray(current, dtype=int))
                    current = [int(original_index)]
                previous_position = position
            else:
                if current:
                    episodes.append(np.asarray(current, dtype=int))
                current = []
                previous_position = None
        if current:
            episodes.append(np.asarray(current, dtype=int))
    return episodes


def _metrics_for_budget(predictions: dict[str, Any], budget: float) -> dict[str, Any]:
    y_true = np.asarray(predictions["y_true"], dtype=int)
    y_score = np.asarray(predictions["y_score"], dtype=float)
    incident_id = np.asarray(predictions["incident_id"], dtype=object)
    timestamp = np.asarray(predictions["timestamp"], dtype=np.int64)
    time_to_escalation = np.asarray(predictions["time_to_escalation"], dtype=float)

    selected = _top_budget_mask(y_score, budget)
    selected_count = int(selected.sum())
    positives = y_true.astype(bool)
    positive_total = int(positives.sum())
    positive_incidents = {str(item) for item in np.unique(incident_id[positives])}
    selected_positive_incidents = {
        str(item) for item in np.unique(incident_id[selected & positives])
    }

    episodes = _episode_indices(selected, incident_id, timestamp)
    false_episodes = 0
    positive_episodes = 0
    stable_positive_incidents: set[str] = set()
    stable_leads: list[float] = []
    episode_lengths: list[int] = []
    for episode in episodes:
        episode_lengths.append(int(episode.shape[0]))
        episode_positive = positives[episode]
        if bool(episode_positive.any()):
            positive_episodes += 1
        else:
            false_episodes += 1
        if episode.shape[0] >= STABLE_RUN and bool(episode_positive.any()):
            positive_indices = episode[episode_positive]
            incident = str(incident_id[positive_indices[0]])
            stable_positive_incidents.add(incident)
            stable_leads.append(float(np.max(time_to_escalation[positive_indices])))

    selected_positive = int((selected & positives).sum())
    n = int(y_true.shape[0])
    return {
        "budget": budget,
        "budget_windows": selected_count,
        "n_windows": n,
        "positive_windows": positive_total,
        "positive_incidents": len(positive_incidents),
        "window_precision": selected_positive / max(selected_count, 1),
        "window_recall": selected_positive / max(positive_total, 1),
        "incident_recall": len(selected_positive_incidents) / max(len(positive_incidents), 1),
        "stable_incident_recall": len(stable_positive_incidents) / max(len(positive_incidents), 1),
        "mean_stable_lead": float(mean(stable_leads)) if stable_leads else None,
        "episodes": len(episodes),
        "positive_episodes": positive_episodes,
        "false_episodes": false_episodes,
        "episodes_per_100_windows": 100.0 * len(episodes) / max(n, 1),
        "false_episodes_per_100_windows": 100.0 * false_episodes / max(n, 1),
        "mean_episode_length": float(mean(episode_lengths)) if episode_lengths else 0.0,
    }


def _summarize(values: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "AUPRC",
        "window_precision",
        "window_recall",
        "incident_recall",
        "stable_incident_recall",
        "episodes_per_100_windows",
        "false_episodes_per_100_windows",
        "mean_episode_length",
    ]
    summary: dict[str, Any] = {}
    for key in keys:
        series = [float(item[key]) for item in values if item.get(key) is not None]
        summary[f"{key}_mean"] = float(mean(series)) if series else None
        summary[f"{key}_std"] = float(pstdev(series)) if len(series) > 1 else 0.0
    leads = [float(item["mean_stable_lead"]) for item in values if item.get("mean_stable_lead") is not None]
    summary["mean_stable_lead_mean"] = float(mean(leads)) if leads else None
    summary["mean_stable_lead_std"] = float(pstdev(leads)) if len(leads) > 1 else 0.0
    summary["n_runs"] = len(values)
    summary["budget_windows_mean"] = float(mean([float(item["budget_windows"]) for item in values]))
    summary["positive_incidents_mean"] = float(mean([float(item["positive_incidents"]) for item in values]))
    return summary


def build_audit() -> dict[str, Any]:
    rows = []
    summary = []
    for setting in SETTINGS:
        export_dir = AUDIT_DIR / str(setting["export_dir"])
        for method, template in setting["methods"].items():
            for seed in SEEDS:
                path = export_dir / template.format(seed=seed)
                if not path.exists():
                    continue
                payload = _load_json(path)
                predictions = payload["predictions"]
                for budget in BUDGETS:
                    row = _metrics_for_budget(predictions, budget)
                    row.update(
                        {
                            "setting": setting["setting"],
                            "method": method,
                            "seed": seed,
                            "AUPRC": float(payload["metrics"]["AUPRC"]),
                        }
                    )
                    rows.append(row)

    for setting in sorted({str(row["setting"]) for row in rows}):
        for method in sorted({str(row["method"]) for row in rows if row["setting"] == setting}):
            for budget in BUDGETS:
                values = [
                    row
                    for row in rows
                    if row["setting"] == setting
                    and row["method"] == method
                    and abs(float(row["budget"]) - budget) < 1e-12
                ]
                if not values:
                    continue
                entry = {
                    "setting": setting,
                    "method": method,
                    "budget": budget,
                }
                entry.update(_summarize(values))
                summary.append(entry)

    return {
        "audit": "fixed-budget alert stability audit",
        "seeds": SEEDS,
        "budgets": BUDGETS,
        "primary_budget": PRIMARY_BUDGET,
        "stable_run": STABLE_RUN,
        "note": (
            "Scores are converted into a fixed review budget without tuning a threshold on the test split. "
            "Episodes are contiguous selected windows within the same incident after timestamp sorting. "
            "Stable incident recall requires at least two consecutive selected windows containing a positive "
            "forecast-horizon window."
        ),
        "rows": rows,
        "summary": summary,
    }


def _fmt_mean_std(row: dict[str, Any], key: str, digits: int = 3) -> str:
    value = row.get(f"{key}_mean")
    std = row.get(f"{key}_std")
    if value is None:
        return "--"
    return f"${float(value):.{digits}f}\\pm{float(std):.{digits}f}$"


def _fmt_value(value: Any, digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"${float(value):.{digits}f}$"


def _primary_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in audit["summary"]
        if abs(float(row["budget"]) - PRIMARY_BUDGET) < 1e-12
    ]


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Fixed-budget alert stability audit",
        "",
        audit["note"],
        "",
        f"Primary table uses a {int(PRIMARY_BUDGET * 100)}% review budget and a {STABLE_RUN}-window confirmation rule.",
        "",
        "| Setting | Method | AUPRC | IncRec | StableIncRec | StableLead | Episodes/100 | FalseEpisodes/100 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _primary_rows(audit):
        lines.append(
            "| {setting} | {method} | {auprc} | {inc} | {stable} | {lead} | {episodes} | {false_eps} |".format(
                setting=row["setting"],
                method=row["method"],
                auprc=_fmt_mean_std(row, "AUPRC"),
                inc=_fmt_mean_std(row, "incident_recall"),
                stable=_fmt_mean_std(row, "stable_incident_recall"),
                lead=_fmt_value(row["mean_stable_lead_mean"]),
                episodes=_fmt_mean_std(row, "episodes_per_100_windows"),
                false_eps=_fmt_mean_std(row, "false_episodes_per_100_windows"),
            )
        )
    (RESULT_DIR / "alert_stability_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(audit: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Fixed-budget alert stability audit. Scores are converted into a 5\% review budget without test-threshold tuning. Episodes are contiguous selected windows within an incident; stable incident recall requires at least two consecutive selected windows containing a positive forecast-horizon window. Values are mean$\pm$std over 20 seeds.}",
        r"\label{tab:alert-stability-audit}",
        r"\maxtablewidth{",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Setting & Method & AUPRC & IncRec & StableIncRec & StableLead & Episodes/100 & FalseEpisodes/100 \\",
        r"\midrule",
    ]
    previous_setting: str | None = None
    for row in _primary_rows(audit):
        setting = str(row["setting"])
        setting_text = setting if setting != previous_setting else ""
        previous_setting = setting
        lines.append(
            "{setting} & {method} & {auprc} & {inc} & {stable} & {lead} & {episodes} & {false_eps} \\\\".format(
                setting=setting_text.replace("_", r"\_"),
                method=str(row["method"]).replace("_", r"\_"),
                auprc=_fmt_mean_std(row, "AUPRC"),
                inc=_fmt_mean_std(row, "incident_recall"),
                stable=_fmt_mean_std(row, "stable_incident_recall"),
                lead=_fmt_value(row["mean_stable_lead_mean"]),
                episodes=_fmt_mean_std(row, "episodes_per_100_windows"),
                false_eps=_fmt_mean_std(row, "false_episodes_per_100_windows"),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    (FIG_DIR / "tab_alert_stability_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_audit()
    (RESULT_DIR / "alert_stability_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit)
    write_latex(audit)
    print("Wrote outputs/results/alert_stability_audit.json")
    print("Wrote outputs/results/alert_stability_audit.md")
    print("Wrote figures/tab_alert_stability_audit.tex")


if __name__ == "__main__":
    main()
