from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"

SEEDS_3 = [7, 13, 21]
SEEDS_20 = [7, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025]

KNOWLEDGE_CHANNELS = {
    "recon",
    "auth_abuse",
    "execution",
    "persistence",
    "priv_esc",
    "cred_access",
    "lateral_move",
    "c2",
    "collection_exfil",
    "impact",
    "defense_evasion",
}
HIGH_RISK_CHANNELS = {
    "execution",
    "persistence",
    "priv_esc",
    "cred_access",
    "lateral_move",
    "c2",
    "collection_exfil",
    "impact",
    "defense_evasion",
}

SETTINGS = [
    {
        "setting": "ATLASv2 held-out-family",
        "dataset_dir": "data/atlasv2_public",
        "test_split": "test_event_disjoint",
        "metadata": "data/atlasv2_public/metadata.json",
        "seeds": SEEDS_3,
        "methods": {
            "TRACER route": "outputs/results/audits/label_grounded_evidence_seed_exports/r240_tracer_adaptive_event_atlasv2_public_seed{seed}.json",
            "Prefix-Only": "outputs/results/audits/label_grounded_evidence_seed_exports/r010_prefix_retrieval_atlasv2_public_event_disjoint_seed{seed}.json",
            "Pure-kNN": "outputs/results/audits/label_grounded_evidence_seed_exports/r015_pure_knn_atlasv2_public_event_disjoint_seed{seed}.json",
            "Shared-Encoder": "outputs/results/audits/label_grounded_evidence_seed_exports/r010_campaign_mem_atlasv2_public_event_disjoint_seed{seed}.json",
        },
    },
    {
        "setting": "AIT-ADS chronology",
        "dataset_dir": "data/ait_ads_public",
        "test_split": "test",
        "metadata": "data/ait_ads_public/metadata.json",
        "seeds": SEEDS_20,
        "methods": {
            "TRACER route": "outputs/results/audits/ait_ads_chronology_significance_seed_exports/r241_tracer_adaptive_ait_ads_public_test_seed{seed}.json",
            "Prefix-Only": "outputs/results/audits/ait_ads_chronology_significance_seed_exports/r071_prefix_retrieval_ait_ads_public_test_seed{seed}.json",
        },
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_matrix(prefix: np.ndarray, channels: list[str], allowed: set[str]) -> np.ndarray:
    indices = [idx for idx, name in enumerate(channels) if name in allowed]
    if not indices:
        return np.zeros((prefix.shape[0], 0), dtype=bool)
    values = prefix[:, :, indices].sum(axis=1)
    return values > 1e-8


def _jaccard(query: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    if neighbors.size == 0:
        return np.zeros((query.shape[0], 0), dtype=np.float64)
    query_expanded = query[:, None, :]
    intersection = np.logical_and(query_expanded, neighbors).sum(axis=2)
    union = np.logical_or(query_expanded, neighbors).sum(axis=2)
    result = np.zeros_like(intersection, dtype=np.float64)
    np.divide(intersection, union, out=result, where=union > 0)
    return result


def _hit(query: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    if neighbors.size == 0:
        return np.zeros((query.shape[0], 0), dtype=bool)
    return np.logical_and(query[:, None, :], neighbors).any(axis=2)


def _stage_profiles(metadata: dict[str, Any]) -> tuple[dict[str, set[str]], set[str]]:
    high_risk = set(str(item) for item in metadata.get("high_risk_stages", []))
    profiles: dict[str, set[str]] = {}
    for incident in metadata.get("incidents", []):
        incident_id = str(incident.get("incident_id", ""))
        counts = incident.get("stage_counts", {})
        active = {str(stage) for stage, count in counts.items() if stage != "unlabeled" and float(count) > 0}
        if active:
            profiles[incident_id] = active
    return profiles, high_risk


def _stage_overlap(query_ids: np.ndarray, neighbor_ids: np.ndarray, profiles: dict[str, set[str]], high_risk: set[str]) -> tuple[np.ndarray, np.ndarray]:
    stage = np.zeros(neighbor_ids.shape, dtype=bool)
    high = np.zeros(neighbor_ids.shape, dtype=bool)
    for row, query_id in enumerate(query_ids.tolist()):
        query_profile = profiles.get(str(query_id), set())
        query_high = query_profile & high_risk
        for col, neighbor_id in enumerate(neighbor_ids[row].tolist()):
            neighbor_profile = profiles.get(str(neighbor_id), set())
            stage[row, col] = bool(query_profile & neighbor_profile)
            high[row, col] = bool(query_high & (neighbor_profile & high_risk))
    return stage, high


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _summary_features(prefix: np.ndarray) -> np.ndarray:
    prefix = prefix.astype(np.float32)
    mean_values = prefix.mean(axis=1)
    std_values = prefix.std(axis=1)
    max_values = prefix.max(axis=1)
    last_values = prefix[:, -1, :]
    slope_values = (prefix[:, -1, :] - prefix[:, 0, :]) / max(prefix.shape[1] - 1, 1)
    return np.concatenate([mean_values, std_values, max_values, last_values, slope_values], axis=-1).astype(np.float32)


def _cosine_topk_indices(train_prefix: np.ndarray, test_prefix: np.ndarray, top_k: int) -> np.ndarray:
    train_summary = _summary_features(train_prefix)
    test_summary = _summary_features(test_prefix)
    train_norm = train_summary / np.linalg.norm(train_summary, axis=1, keepdims=True).clip(min=1e-6)
    test_norm = test_summary / np.linalg.norm(test_summary, axis=1, keepdims=True).clip(min=1e-6)
    similarity = test_norm @ train_norm.T
    return np.argsort(-similarity, axis=1)[:, :top_k]


def _seed_metrics(spec: dict[str, Any], method: str, template: str, seed: int) -> dict[str, float]:
    dataset_dir = ROOT / str(spec["dataset_dir"])
    metadata = _load_json(ROOT / str(spec["metadata"]))
    channels = [str(item) for item in metadata["feature_channels"]]
    with np.load(dataset_dir / "train.npz", allow_pickle=True) as train_npz:
        train_prefix = np.asarray(train_npz["prefix"], dtype=np.float32)
        train_incident = np.asarray(train_npz["incident_id"], dtype=object)
    with np.load(dataset_dir / f"{spec['test_split']}.npz", allow_pickle=True) as test_npz:
        test_prefix = np.asarray(test_npz["prefix"], dtype=np.float32)
        test_incident = np.asarray(test_npz["incident_id"], dtype=object)

    payload = _load_json(ROOT / template.format(seed=seed))
    pred = payload["predictions"]
    if "retrieved_indices" in pred:
        retrieved_indices = np.asarray(pred["retrieved_indices"], dtype=int)
    else:
        top_k = 5
        retrieved_labels = pred.get("retrieved_label_main")
        if retrieved_labels is not None:
            retrieved_labels_arr = np.asarray(retrieved_labels)
            if retrieved_labels_arr.ndim >= 2:
                top_k = int(retrieved_labels_arr.shape[1])
        retrieved_indices = _cosine_topk_indices(train_prefix, test_prefix, top_k)
    if retrieved_indices.ndim == 1:
        retrieved_indices = retrieved_indices[:, None]
    y_true = np.asarray(pred["y_true"], dtype=int)
    y_score = np.asarray(pred["y_score"], dtype=np.float64)
    query_incident = np.asarray(pred.get("incident_id", test_incident), dtype=object)
    if "retrieved_incident_id" in pred:
        retrieved_incident = np.asarray(pred["retrieved_incident_id"], dtype=object)
    else:
        retrieved_incident = train_incident[retrieved_indices]
    if retrieved_incident.ndim == 1:
        retrieved_incident = retrieved_incident[:, None]

    query_knowledge = _active_matrix(test_prefix, channels, KNOWLEDGE_CHANNELS)
    train_knowledge = _active_matrix(train_prefix, channels, KNOWLEDGE_CHANNELS)
    query_high = _active_matrix(test_prefix, channels, HIGH_RISK_CHANNELS)
    train_high = _active_matrix(train_prefix, channels, HIGH_RISK_CHANNELS)
    neighbor_knowledge = train_knowledge[retrieved_indices]
    neighbor_high = train_high[retrieved_indices]

    jaccard = _jaccard(query_knowledge, neighbor_knowledge)
    hit = _hit(query_knowledge, neighbor_knowledge)
    high_jaccard = _jaccard(query_high, neighbor_high)
    high_hit = _hit(query_high, neighbor_high)

    top_count = max(1, int(np.ceil(0.10 * y_score.shape[0])))
    top_mask = np.zeros(y_score.shape[0], dtype=bool)
    top_mask[np.argsort(-y_score, kind="mergesort")[:top_count]] = True
    pos_mask = y_true.astype(bool)

    stage_profiles, high_risk_stages = _stage_profiles(metadata)
    if stage_profiles:
        stage_hit, stage_high_hit = _stage_overlap(query_incident, retrieved_incident, stage_profiles, high_risk_stages)
        stage_slot = 100.0 * _safe_mean(stage_hit.astype(np.float64))
        stage_any = 100.0 * _safe_mean(stage_hit.any(axis=1).astype(np.float64))
        high_stage_any = 100.0 * _safe_mean(stage_high_hit.any(axis=1).astype(np.float64))
    else:
        stage_slot = stage_any = high_stage_any = float("nan")

    return {
        "knowledge_hit_slot_at_5": 100.0 * _safe_mean(hit.astype(np.float64)),
        "knowledge_any_hit_at_5": 100.0 * _safe_mean(hit.any(axis=1).astype(np.float64)),
        "knowledge_jaccard_at_5": 100.0 * _safe_mean(jaccard),
        "high_risk_any_hit_at_5": 100.0 * _safe_mean(high_hit.any(axis=1).astype(np.float64)),
        "high_risk_jaccard_at_5": 100.0 * _safe_mean(high_jaccard),
        "positive_knowledge_jaccard_at_5": 100.0 * _safe_mean(jaccard[pos_mask]) if pos_mask.any() else 0.0,
        "top_decile_knowledge_jaccard_at_5": 100.0 * _safe_mean(jaccard[top_mask]),
        "incident_stage_slot_hit_at_5": stage_slot,
        "incident_stage_any_hit_at_5": stage_any,
        "incident_high_stage_any_hit_at_5": high_stage_any,
    }


def _summarize(values: list[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    if len(finite) == 1:
        return float(finite[0]), 0.0
    return float(mean(finite)), float(pstdev(finite))


def build_audit() -> dict[str, Any]:
    setting_rows = []
    for spec in SETTINGS:
        method_rows = []
        for method, template in spec["methods"].items():
            per_seed = [_seed_metrics(spec, method, template, seed) for seed in spec["seeds"]]
            metric_names = list(per_seed[0].keys())
            summary = {}
            for metric in metric_names:
                avg, std = _summarize([row[metric] for row in per_seed])
                summary[metric] = avg
                summary[f"{metric}_std"] = std
            method_rows.append(
                {
                    "method": method,
                    "seeds": spec["seeds"],
                    "summary": summary,
                    "per_seed": per_seed,
                }
            )
        setting_rows.append(
            {
                "setting": spec["setting"],
                "methods": method_rows,
            }
        )
    return {
        "audit": "Knowledge-grounded retrieved-evidence audit",
        "settings": setting_rows,
        "note": "The audit uses current-prefix ATT&CK-like feature channels and, for AIT-ADS, incident-level stage profiles. It does not use future-signature distances, so it complements AF@5 and TTE-Err@1 as a knowledge-consistency check for case-based evidence.",
    }


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.1f}"


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Knowledge-grounded retrieved-evidence audit",
        "",
        audit["note"],
        "",
    ]
    for setting in audit["settings"]:
        lines += [
            f"## {setting['setting']}",
            "",
            "| Method | K-Hit@5 | K-Jacc@5 | HighRisk-Hit@5 | Pos K-Jacc@5 | Top10 K-Jacc@5 | Stage-Hit@5 | HighStage-Hit@5 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in setting["methods"]:
            s = row["summary"]
            lines.append(
                "| {method} | {khit} | {kj} | {hhit} | {pos} | {top} | {stage} | {hstage} |".format(
                    method=row["method"],
                    khit=_fmt(s["knowledge_any_hit_at_5"]),
                    kj=_fmt(s["knowledge_jaccard_at_5"]),
                    hhit=_fmt(s["high_risk_any_hit_at_5"]),
                    pos=_fmt(s["positive_knowledge_jaccard_at_5"]),
                    top=_fmt(s["top_decile_knowledge_jaccard_at_5"]),
                    stage=_fmt(s["incident_stage_any_hit_at_5"]),
                    hstage=_fmt(s["incident_high_stage_any_hit_at_5"]),
                )
            )
        lines.append("")
    (RESULT_DIR / "knowledge_grounded_evidence_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_latex(audit: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Knowledge-grounded case-evidence audit. The audit measures whether retrieved train-only analogs share current-prefix ATT\&CK-like knowledge channels with the query, and for AIT-ADS also whether incident-level attack-stage profiles overlap. Unlike AF@5 and TTE-Err@1, these metrics do not use future-signature distances. Values are percentages.}",
        r"\label{tab:knowledge-grounded-evidence}",
        r"\maxtablewidth{",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Setting & Method & K-Hit@5 & K-Jacc@5 & HR-Hit@5 & Pos K-Jacc@5 & Top10 K-Jacc@5 & Stage-Hit@5 & HRStage-Hit@5 \\",
        r"\midrule",
    ]
    for setting in audit["settings"]:
        first = True
        for row in setting["methods"]:
            s = row["summary"]
            lines.append(
                "{setting} & {method} & {khit} & {kj} & {hhit} & {pos} & {top} & {stage} & {hstage} \\\\".format(
                    setting=str(setting["setting"]).replace("_", r"\_") if first else "",
                    method=str(row["method"]).replace("_", r"\_"),
                    khit=_fmt(s["knowledge_any_hit_at_5"]),
                    kj=_fmt(s["knowledge_jaccard_at_5"]),
                    hhit=_fmt(s["high_risk_any_hit_at_5"]),
                    pos=_fmt(s["positive_knowledge_jaccard_at_5"]),
                    top=_fmt(s["top_decile_knowledge_jaccard_at_5"]),
                    stage=_fmt(s["incident_stage_any_hit_at_5"]),
                    hstage=_fmt(s["incident_high_stage_any_hit_at_5"]),
                )
            )
            first = False
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    (FIG_DIR / "tab_knowledge_grounded_evidence.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_audit()
    (RESULT_DIR / "knowledge_grounded_evidence_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit)
    write_latex(audit)
    print("Wrote outputs/results/knowledge_grounded_evidence_audit.json")
    print("Wrote figures/tab_knowledge_grounded_evidence.tex")


if __name__ == "__main__":
    main()
