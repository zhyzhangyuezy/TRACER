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
SEEDS_20 = [
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

JACCARD_RELEVANCE_THRESHOLD = 0.25

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
    out = np.zeros_like(intersection, dtype=np.float64)
    np.divide(intersection, union, out=out, where=union > 0)
    return out


def _hit(query: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    if neighbors.size == 0:
        return np.zeros((query.shape[0], 0), dtype=bool)
    return np.logical_and(query[:, None, :], neighbors).any(axis=2)


def _summary_features(prefix: np.ndarray) -> np.ndarray:
    prefix = prefix.astype(np.float32)
    return np.concatenate(
        [
            prefix.mean(axis=1),
            prefix.std(axis=1),
            prefix.max(axis=1),
            prefix[:, -1, :],
            (prefix[:, -1, :] - prefix[:, 0, :]) / max(prefix.shape[1] - 1, 1),
        ],
        axis=-1,
    ).astype(np.float32)


def _cosine_topk_indices(train_prefix: np.ndarray, test_prefix: np.ndarray, top_k: int) -> np.ndarray:
    train_summary = _summary_features(train_prefix)
    test_summary = _summary_features(test_prefix)
    train_norm = train_summary / np.linalg.norm(train_summary, axis=1, keepdims=True).clip(min=1e-6)
    test_norm = test_summary / np.linalg.norm(test_summary, axis=1, keepdims=True).clip(min=1e-6)
    similarity = test_norm @ train_norm.T
    return np.argsort(-similarity, axis=1)[:, :top_k]


def _stage_profiles(metadata: dict[str, Any]) -> tuple[dict[str, set[str]], set[str]]:
    high_risk = {str(item) for item in metadata.get("high_risk_stages", [])}
    profiles: dict[str, set[str]] = {}
    for incident in metadata.get("incidents", []):
        incident_id = str(incident.get("incident_id", ""))
        counts = incident.get("stage_counts", {})
        active = {str(stage) for stage, count in counts.items() if stage != "unlabeled" and float(count) > 0}
        if active:
            profiles[incident_id] = active
    return profiles, high_risk


def _stage_overlap(
    query_ids: np.ndarray,
    neighbor_ids: np.ndarray,
    profiles: dict[str, set[str]],
    high_risk: set[str],
) -> tuple[np.ndarray, np.ndarray]:
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


def _safe_mean(mask: np.ndarray, values: np.ndarray) -> float:
    if mask.size == 0 or not bool(mask.any()):
        return 0.0
    return float(values[mask].mean())


def _percent(value: float) -> float:
    return 100.0 * float(value)


def _cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool).ravel()
    b = np.asarray(b, dtype=bool).ravel()
    if a.size == 0:
        return 0.0
    po = float((a == b).mean())
    p_yes_a = float(a.mean())
    p_yes_b = float(b.mean())
    pe = p_yes_a * p_yes_b + (1.0 - p_yes_a) * (1.0 - p_yes_b)
    if abs(1.0 - pe) < 1e-12:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def _seed_metrics(spec: dict[str, Any], method: str, template: str, seed: int) -> dict[str, Any]:
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
            retrieved_arr = np.asarray(retrieved_labels)
            if retrieved_arr.ndim >= 2:
                top_k = int(retrieved_arr.shape[1])
        retrieved_indices = _cosine_topk_indices(train_prefix, test_prefix, top_k)
    if retrieved_indices.ndim == 1:
        retrieved_indices = retrieved_indices[:, None]

    y_true = np.asarray(pred["y_true"], dtype=int)
    y_score = np.asarray(pred["y_score"], dtype=np.float64)
    retrieved_labels = np.asarray(pred.get("retrieved_label_main", np.zeros_like(retrieved_indices)), dtype=np.float64)
    if retrieved_labels.ndim == 1:
        retrieved_labels = retrieved_labels[:, None]
    retrieved_positive = retrieved_labels > 0.5

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

    knowledge_jaccard = _jaccard(query_knowledge, neighbor_knowledge)
    high_hit = _hit(query_high, neighbor_high)
    tactic_relevant = knowledge_jaccard >= JACCARD_RELEVANCE_THRESHOLD
    risk_relevant = high_hit

    stage_profiles, high_risk_stages = _stage_profiles(metadata)
    if stage_profiles:
        stage_hit, stage_high_hit = _stage_overlap(query_incident, retrieved_incident, stage_profiles, high_risk_stages)
        risk_relevant = np.logical_or(risk_relevant, stage_high_hit)
        stage_any_rate = stage_hit.any(axis=1)
    else:
        stage_any_rate = np.zeros(y_true.shape[0], dtype=bool)

    relevant = np.logical_or(tactic_relevant, risk_relevant)
    escalation_support = np.logical_and(relevant, retrieved_positive)
    unanchored_positive = np.logical_and(~relevant, retrieved_positive)

    top_count = max(1, int(np.ceil(0.10 * y_score.shape[0])))
    top_mask = np.zeros(y_score.shape[0], dtype=bool)
    top_mask[np.argsort(-y_score, kind="mergesort")[:top_count]] = True
    pos_mask = y_true.astype(bool)

    any_relevant = relevant.any(axis=1)
    any_support = escalation_support.any(axis=1)
    any_unanchored = unanchored_positive.any(axis=1)
    clean_support = np.logical_and(any_support, ~any_unanchored)
    tactic_any = tactic_relevant.any(axis=1)
    risk_any = risk_relevant.any(axis=1)

    positive_slots = retrieved_positive
    relevant_positive_precision = 0.0
    if bool(positive_slots.any()):
        relevant_positive_precision = float(escalation_support[positive_slots].mean())

    return {
        "setting": spec["setting"],
        "method": method,
        "seed": seed,
        "reviewable_case_rate": _percent(any_relevant.mean()),
        "supportive_case_rate": _percent(any_support.mean()),
        "clean_support_rate": _percent(clean_support.mean()),
        "unanchored_positive_rate": _percent(any_unanchored.mean()),
        "top_decile_reviewable_case_rate": _percent(_safe_mean(top_mask, any_relevant.astype(np.float64))),
        "top_decile_supportive_case_rate": _percent(_safe_mean(top_mask, any_support.astype(np.float64))),
        "top_decile_clean_support_rate": _percent(_safe_mean(top_mask, clean_support.astype(np.float64))),
        "top_decile_unanchored_positive_rate": _percent(_safe_mean(top_mask, any_unanchored.astype(np.float64))),
        "positive_supportive_case_rate": _percent(_safe_mean(pos_mask, any_support.astype(np.float64))),
        "positive_clean_support_rate": _percent(_safe_mean(pos_mask, clean_support.astype(np.float64))),
        "relevant_positive_precision": _percent(relevant_positive_precision),
        "tactic_risk_kappa": _cohen_kappa(tactic_any, risk_any),
        "stage_any_rate": _percent(stage_any_rate.mean()),
    }


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "reviewable_case_rate",
        "supportive_case_rate",
        "clean_support_rate",
        "unanchored_positive_rate",
        "top_decile_reviewable_case_rate",
        "top_decile_supportive_case_rate",
        "top_decile_clean_support_rate",
        "top_decile_unanchored_positive_rate",
        "positive_supportive_case_rate",
        "positive_clean_support_rate",
        "relevant_positive_precision",
        "tactic_risk_kappa",
        "stage_any_rate",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["setting"]), str(row["method"])), []).append(row)
    summary = []
    for (setting, method), items in grouped.items():
        out: dict[str, Any] = {"setting": setting, "method": method, "n_seeds": len(items)}
        for key in keys:
            values = [float(item[key]) for item in items]
            out[key] = float(mean(values))
            out[f"{key}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        summary.append(out)
    order = {spec["setting"]: idx for idx, spec in enumerate(SETTINGS)}
    method_order = {
        "TRACER route": 0,
        "Prefix-Only": 1,
        "Pure-kNN": 2,
        "Shared-Encoder": 3,
    }
    summary.sort(key=lambda row: (order.get(str(row["setting"]), 99), method_order.get(str(row["method"]), 99)))
    return summary


def build_audit() -> dict[str, Any]:
    rows = []
    missing = []
    for spec in SETTINGS:
        for method, template in spec["methods"].items():
            for seed in spec["seeds"]:
                path = ROOT / template.format(seed=seed)
                if not path.exists():
                    missing.append(str(path.relative_to(ROOT)))
                    continue
                rows.append(_seed_metrics(spec, method, template, seed))
    return {
        "audit": "Rubric-based case evidence utility audit",
        "note": (
            "This is a deterministic analyst-utility proxy, not a human user study. "
            "A retrieved case is reviewable when it shares current-prefix tactic evidence "
            f"(Jaccard >= {JACCARD_RELEVANCE_THRESHOLD}) or a high-risk channel/stage with the query. "
            "Escalation support additionally requires a positive retrieved train label. "
            "Unanchored positive evidence means a positive retrieved label without current-prefix semantic support."
        ),
        "jaccard_relevance_threshold": JACCARD_RELEVANCE_THRESHOLD,
        "rows": rows,
        "summary": _summarize(rows),
        "missing": missing,
    }


def _fmt(row: dict[str, Any], key: str) -> str:
    return f"{float(row[key]):.1f}"


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Rubric-based case evidence utility audit",
        "",
        audit["note"],
        "",
        "| Setting | Method | Top10 reviewable | Top10 support | Top10 clean support | Top10 unanchored+ | Pos support | RelPosPrec | kappa tactic/risk |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["summary"]:
        lines.append(
            "| {setting} | {method} | {tr} | {ts} | {tc} | {tu} | {ps} | {rp} | {kap:.2f} |".format(
                setting=row["setting"],
                method=row["method"],
                tr=_fmt(row, "top_decile_reviewable_case_rate"),
                ts=_fmt(row, "top_decile_supportive_case_rate"),
                tc=_fmt(row, "top_decile_clean_support_rate"),
                tu=_fmt(row, "top_decile_unanchored_positive_rate"),
                ps=_fmt(row, "positive_supportive_case_rate"),
                rp=_fmt(row, "relevant_positive_precision"),
                kap=float(row["tactic_risk_kappa"]),
            )
        )
    lines += [
        "",
        "## Additional all-window metrics",
        "",
        "| Setting | Method | Reviewable | Support | Clean support | Unanchored+ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in audit["summary"]:
        lines.append(
            "| {setting} | {method} | {r} | {s} | {c} | {u} |".format(
                setting=row["setting"],
                method=row["method"],
                r=_fmt(row, "reviewable_case_rate"),
                s=_fmt(row, "supportive_case_rate"),
                c=_fmt(row, "clean_support_rate"),
                u=_fmt(row, "unanchored_positive_rate"),
            )
        )
    if audit["missing"]:
        lines += ["", "## Missing files", ""]
        lines += [f"- `{item}`" for item in audit["missing"]]
    (RESULT_DIR / "case_evidence_utility_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(audit: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Rubric-based case-evidence utility audit. This deterministic proxy is not a human user study. A retrieved case is reviewable if it shares current-prefix tactic evidence or a high-risk channel/stage with the query; it is supportive if the reviewable case is also a positive train-memory analog. Unanchored positive evidence is a positive retrieved label without current-prefix semantic support. Values are percentages except $\kappa$.}",
        r"\label{tab:case-evidence-utility}",
        r"\maxtablewidth{",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Setting & Method & Top10 Rev. & Top10 Sup. & Top10 Clean & Top10 Unanch. & Pos. Sup. & RelPosPrec & $\kappa$ \\",
        r"\midrule",
    ]
    for row in audit["summary"]:
        lines.append(
            "{setting} & {method} & {tr} & {ts} & {tc} & {tu} & {ps} & {rp} & {kap:.2f} \\\\".format(
                setting=str(row["setting"]).replace("_", r"\_").replace("->", r"$\rightarrow$"),
                method=str(row["method"]).replace("_", r"\_"),
                tr=f"${float(row['top_decile_reviewable_case_rate']):.1f}$",
                ts=f"${float(row['top_decile_supportive_case_rate']):.1f}$",
                tc=f"${float(row['top_decile_clean_support_rate']):.1f}$",
                tu=f"${float(row['top_decile_unanchored_positive_rate']):.1f}$",
                ps=f"${float(row['positive_supportive_case_rate']):.1f}$",
                rp=f"${float(row['relevant_positive_precision']):.1f}$",
                kap=float(row["tactic_risk_kappa"]),
            )
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table*}",
    ]
    (FIG_DIR / "tab_case_evidence_utility.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    (RESULT_DIR / "case_evidence_utility_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit)
    write_latex(audit)
    print(f"wrote {RESULT_DIR / 'case_evidence_utility_audit.json'}")
    print(f"wrote {FIG_DIR / 'tab_case_evidence_utility.tex'}")


if __name__ == "__main__":
    main()
