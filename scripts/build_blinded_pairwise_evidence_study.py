from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "blinded_expert_evidence_rating_study"

KNOWLEDGE_CHANNELS = [
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
]

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
        "dataset": "ATLASv2 held-out-family",
        "dataset_dir": "data/atlasv2_public",
        "test_split": "test_event_disjoint",
        "metadata": "data/atlasv2_public/metadata.json",
        "seed": 7,
        "sample_queries": 40,
        "methods": {
            "TRACER route": "outputs/results/audits/label_grounded_evidence_seed_exports/r240_tracer_adaptive_event_atlasv2_public_seed7.json",
            "Prefix-Only": "outputs/results/audits/label_grounded_evidence_seed_exports/r010_prefix_retrieval_atlasv2_public_event_disjoint_seed7.json",
        },
    },
    {
        "dataset": "AIT-ADS chronology",
        "dataset_dir": "data/ait_ads_public",
        "test_split": "test",
        "metadata": "data/ait_ads_public/metadata.json",
        "seed": 7,
        "sample_queries": 40,
        "methods": {
            "TRACER route": "outputs/results/audits/ait_ads_chronology_significance_seed_exports/r241_tracer_adaptive_ait_ads_public_test_seed7.json",
            "Prefix-Only": "outputs/results/audits/ait_ads_chronology_significance_seed_exports/r071_prefix_retrieval_ait_ads_public_test_seed7.json",
        },
    },
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _anon(value: Any, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return digest[:12]


def _load_split(dataset_dir: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(dataset_dir / f"{split}.npz", allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _active_channels(prefix: np.ndarray, channels: list[str], allowed: list[str] | set[str]) -> list[str]:
    allowed_set = set(allowed)
    active: list[str] = []
    for idx, name in enumerate(channels):
        if name in allowed_set and float(prefix[:, idx].sum()) > 1e-8:
            active.append(name)
    return active


def _stage_profiles(metadata: dict[str, Any]) -> dict[str, set[str]]:
    profiles: dict[str, set[str]] = {}
    for incident in metadata.get("incidents", []):
        incident_id = str(incident.get("incident_id", ""))
        counts = incident.get("stage_counts", {})
        active = {str(stage) for stage, count in counts.items() if stage != "unlabeled" and float(count) > 0}
        if active:
            profiles[incident_id] = active
    return profiles


def _score_band(score: float, all_scores: np.ndarray) -> str:
    rank = float((all_scores <= score).mean())
    if rank >= 0.90:
        return "top_decile"
    if rank >= 0.75:
        return "high_quartile"
    if rank <= 0.25:
        return "low_quartile"
    return "middle"


def _select_queries(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int,
    seed: int,
    eligible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    strata: dict[int, str] = {}
    seen: set[int] = set()

    def add_many(candidates: list[int], limit: int, stratum: str) -> None:
        added = 0
        for idx in candidates:
            if idx in seen:
                continue
            selected.append(idx)
            strata[idx] = stratum
            seen.add(idx)
            added += 1
            if added >= limit:
                break

    if eligible_mask is None:
        eligible_mask = np.ones_like(y_true, dtype=bool)
    else:
        eligible_mask = np.asarray(eligible_mask, dtype=bool)

    def eligible_first(candidates: list[int]) -> list[int]:
        return [idx for idx in candidates if eligible_mask[idx]]

    positives = np.flatnonzero(y_true.astype(bool)).tolist()
    positives = sorted(positives, key=lambda idx: float(y_score[idx]), reverse=True)
    add_many(eligible_first(positives), max(1, n // 4), "positive_or_high_score_tp")

    high_negatives = np.flatnonzero(~y_true.astype(bool)).tolist()
    high_negatives = sorted(high_negatives, key=lambda idx: float(y_score[idx]), reverse=True)
    add_many(eligible_first(high_negatives), max(1, n // 4), "high_score_negative")

    median = float(np.median(y_score))
    middle = list(range(y_score.shape[0]))
    middle = sorted(middle, key=lambda idx: abs(float(y_score[idx]) - median))
    add_many(eligible_first(middle), max(1, n // 4), "middle_score_ambiguous")

    low_negatives = sorted(high_negatives, key=lambda idx: float(y_score[idx]))
    add_many(eligible_first(low_negatives), max(1, n // 4), "low_score_negative")

    if len(selected) < n:
        remaining = [idx for idx in range(y_score.shape[0]) if idx not in seen and eligible_mask[idx]]
        rng.shuffle(remaining)
        add_many(remaining, n - len(selected), "stratified_fill")

    if len(selected) < n:
        remaining = [idx for idx in range(y_score.shape[0]) if idx not in seen]
        rng.shuffle(remaining)
        add_many(remaining, n - len(selected), "inactive_stratified_fill")

    selected_array = np.asarray(selected[:n], dtype=int)
    return selected_array, {idx: strata[idx] for idx in selected_array.tolist()}


def _evidence_for_method(
    *,
    method: str,
    payload: dict[str, Any],
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    metadata: dict[str, Any],
    query_idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pred = payload["predictions"]
    channels = [str(item) for item in metadata["feature_channels"]]
    stage_profiles = _stage_profiles(metadata)

    y_score = np.asarray(pred["y_score"], dtype=np.float64)
    retrieved_indices = np.asarray(pred["retrieved_indices"], dtype=int)
    if retrieved_indices.ndim == 1:
        retrieved_indices = retrieved_indices[:, None]
    retrieved_labels = np.asarray(pred.get("retrieved_label_main", np.zeros_like(retrieved_indices)), dtype=float)
    if retrieved_labels.ndim == 1:
        retrieved_labels = retrieved_labels[:, None]

    retrieved_incidents = np.asarray(
        pred.get("retrieved_incident_id", np.asarray(train["incident_id"], dtype=object)[retrieved_indices]),
        dtype=object,
    )
    retrieved_families = np.asarray(
        pred.get("retrieved_family_id", np.asarray(train["family_id"], dtype=object)[retrieved_indices]),
        dtype=object,
    )
    if retrieved_incidents.ndim == 1:
        retrieved_incidents = retrieved_incidents[:, None]
    if retrieved_families.ndim == 1:
        retrieved_families = retrieved_families[:, None]

    blind: dict[str, Any] = {}
    key: dict[str, Any] = {
        "method": method,
        "method_score": float(y_score[query_idx]),
        "method_score_band": _score_band(float(y_score[query_idx]), y_score),
    }

    for rank in range(min(5, retrieved_indices.shape[1])):
        neighbor_idx = int(retrieved_indices[query_idx, rank])
        neighbor_prefix = np.asarray(train["prefix"][neighbor_idx], dtype=np.float32)
        neighbor_channels = _active_channels(neighbor_prefix, channels, KNOWLEDGE_CHANNELS)
        neighbor_high = [name for name in neighbor_channels if name in HIGH_RISK_CHANNELS]
        neighbor_incident = str(retrieved_incidents[query_idx, rank])
        neighbor_profile = sorted(stage_profiles.get(neighbor_incident, set()))
        neighbor_label = int(retrieved_labels[query_idx, rank] > 0.5)
        neighbor_tte = float(np.asarray(train["time_to_escalation"], dtype=float)[neighbor_idx])

        prefix = f"analog{rank + 1}"
        blind[f"{prefix}_ref"] = f"a_{_anon(neighbor_incident + ':' + str(rank), 'pairwise-analog')}"
        blind[f"{prefix}_active_channels"] = ";".join(neighbor_channels) if neighbor_channels else "none"
        blind[f"{prefix}_high_risk_channels"] = ";".join(neighbor_high) if neighbor_high else "none"
        blind[f"{prefix}_stage_profile"] = ";".join(neighbor_profile) if neighbor_profile else "not_provided"
        blind[f"{prefix}_historical_outcome"] = "escalated_within_30m" if neighbor_label else "not_escalated_within_30m"
        blind[f"{prefix}_time_to_escalation_min"] = f"{neighbor_tte:.1f}" if neighbor_label else "not_applicable"

        key[f"{prefix}_train_index"] = neighbor_idx
        key[f"{prefix}_incident_id"] = neighbor_incident
        key[f"{prefix}_family_id"] = str(retrieved_families[query_idx, rank])
        key[f"{prefix}_label"] = neighbor_label
        key[f"{prefix}_time_to_escalation"] = neighbor_tte

    return blind, key


def _add_rating_columns(row: dict[str, Any], set_name: str) -> None:
    row[f"{set_name}_relevance_1_5"] = ""
    row[f"{set_name}_supportiveness_1_5"] = ""
    row[f"{set_name}_actionability_1_5"] = ""
    row[f"{set_name}_explanation_quality_1_5"] = ""
    row[f"{set_name}_misleading_safety_1_5"] = ""


def _prefix_set_fields(row: dict[str, Any], set_name: str, evidence: dict[str, Any]) -> None:
    for key, value in evidence.items():
        row[f"{set_name}_{key}"] = value


def _build_pairs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(20260426)
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"settings": [], "total_pairs": 0}

    for spec in SETTINGS:
        dataset_dir = ROOT / str(spec["dataset_dir"])
        metadata = _load_json(ROOT / str(spec["metadata"]))
        channels = [str(item) for item in metadata["feature_channels"]]
        train = _load_split(dataset_dir, "train")
        test = _load_split(dataset_dir, str(spec["test_split"]))
        stage_profiles = _stage_profiles(metadata)

        method_payloads = {method: _load_json(ROOT / str(path)) for method, path in spec["methods"].items()}
        tracer_payload = method_payloads["TRACER route"]
        tracer_pred = tracer_payload["predictions"]
        y_true = np.asarray(tracer_pred["y_true"], dtype=int)
        y_score = np.asarray(tracer_pred["y_score"], dtype=np.float64)
        active_channel_indices = [channels.index(name) for name in KNOWLEDGE_CHANNELS if name in channels]
        eligible_mask = np.asarray(test["prefix"][:, :, active_channel_indices].sum(axis=(1, 2)) > 1e-8, dtype=bool)
        selected_queries, strata = _select_queries(
            y_true,
            y_score,
            int(spec["sample_queries"]),
            int(spec["seed"]),
            eligible_mask=eligible_mask,
        )

        setting_summary = {
            "dataset": spec["dataset"],
            "seed": spec["seed"],
            "sampled_pairs": int(selected_queries.shape[0]),
            "sampled_positive_queries": int(y_true[selected_queries].sum()),
            "methods": list(spec["methods"].keys()),
            "strata": {name: int(sum(1 for idx in selected_queries.tolist() if strata[idx] == name)) for name in sorted(set(strata.values()))},
        }
        summary["settings"].append(setting_summary)

        incident_ids = np.asarray(tracer_pred.get("incident_id", test.get("incident_id", np.arange(y_true.shape[0]))), dtype=object)
        family_ids = np.asarray(tracer_pred.get("family_id", test.get("family_id", np.asarray(["unknown"] * y_true.shape[0]))), dtype=object)
        timestamps = np.asarray(tracer_pred.get("timestamp", test.get("timestamp", np.arange(y_true.shape[0]))), dtype=object)
        tte = np.asarray(tracer_pred["time_to_escalation"], dtype=float)

        for query_idx in selected_queries.tolist():
            query_incident = str(incident_ids[query_idx])
            pair_id = f"pair_{_anon(f'{spec['dataset']}|{query_idx}|{spec['seed']}', 'pairwise-study')}"
            query_prefix = np.asarray(test["prefix"][query_idx], dtype=np.float32)
            query_channels = _active_channels(query_prefix, channels, KNOWLEDGE_CHANNELS)
            query_high = [name for name in query_channels if name in HIGH_RISK_CHANNELS]
            query_profile = sorted(stage_profiles.get(query_incident, set()))

            evidence_by_method: dict[str, dict[str, Any]] = {}
            key_by_method: dict[str, dict[str, Any]] = {}
            for method, payload in method_payloads.items():
                evidence_by_method[method], key_by_method[method] = _evidence_for_method(
                    method=method,
                    payload=payload,
                    train=train,
                    test=test,
                    metadata=metadata,
                    query_idx=query_idx,
                )

            methods = list(spec["methods"].keys())
            if rng.random() < 0.5:
                set_a_method, set_b_method = methods[0], methods[1]
            else:
                set_a_method, set_b_method = methods[1], methods[0]

            blind_row: dict[str, Any] = {
                "rater_id": "",
                "pair_id": pair_id,
                "dataset": spec["dataset"],
                "query_ref": f"q_{_anon(query_incident, 'pairwise-query')}",
                "sampling_stratum": strata[query_idx],
                "score_band_blinded": _score_band(float(y_score[query_idx]), y_score),
                "query_active_channels": ";".join(query_channels) if query_channels else "none",
                "query_high_risk_channels": ";".join(query_high) if query_high else "none",
                "query_stage_profile": ";".join(query_profile) if query_profile else "not_provided",
            }
            _prefix_set_fields(blind_row, "set_a", evidence_by_method[set_a_method])
            _add_rating_columns(blind_row, "set_a")
            _prefix_set_fields(blind_row, "set_b", evidence_by_method[set_b_method])
            _add_rating_columns(blind_row, "set_b")
            blind_row["preferred_set"] = ""
            blind_row["preference_confidence_1_5"] = ""
            blind_row["free_text_rationale"] = ""
            blinded_rows.append(blind_row)

            key_row: dict[str, Any] = {
                "pair_id": pair_id,
                "dataset": spec["dataset"],
                "query_index": query_idx,
                "query_incident_id": query_incident,
                "query_family_id": str(family_ids[query_idx]),
                "query_timestamp": str(timestamps[query_idx]),
                "query_y_true": int(y_true[query_idx]),
                "query_time_to_escalation": float(tte[query_idx]),
                "set_a_method": set_a_method,
                "set_b_method": set_b_method,
                "seed": int(spec["seed"]),
            }
            for set_name, method in [("set_a", set_a_method), ("set_b", set_b_method)]:
                for key, value in key_by_method[method].items():
                    key_row[f"{set_name}_{key}"] = value
            key_rows.append(key_row)

    order = rng.permutation(len(blinded_rows))
    blinded_rows = [blinded_rows[int(idx)] for idx in order]
    summary["total_pairs"] = len(blinded_rows)
    return blinded_rows, key_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_rater_sheets(master_rows: list[dict[str, Any]], n_raters: int = 3) -> None:
    for rater_idx in range(1, n_raters + 1):
        rng = np.random.default_rng(20260426 + rater_idx)
        order = rng.permutation(len(master_rows))
        rows = []
        for idx in order:
            row = dict(master_rows[int(idx)])
            row["rater_id"] = f"rater_{rater_idx:02d}"
            rows.append(row)
        _write_csv(OUT_DIR / f"rater_{rater_idx:02d}_sheet.csv", rows)


def _write_protocol(summary: dict[str, Any]) -> None:
    protocol = f"""# Blinded Expert Evidence-Rating Study

This package supports a blinded expert evidence-rating study for TRACER. It is designed to measure human-rated evidence quality, not live SOC performance.

## Study Design

The study uses {summary["total_pairs"]} paired query cases sampled from ATLASv2 held-out-family and AIT-ADS chronology. Each case shows one current alert-prefix summary and two anonymized evidence sets, `set_a` and `set_b`. One set is produced by TRACER route retrieval and the other by Prefix-Only retrieval. The order is randomized per case. Raters do not see model identities, exact warning scores, true query futures, incident identifiers, or family identifiers.

Each evidence set contains top-5 train-memory analogs. Each analog is represented by current-prefix channels, high-risk channels, stage profile, historical escalation outcome, and historical time-to-escalation when applicable. Query future labels are hidden.

## Files

- `pairwise_items_blinded.csv`: master blinded A/B packet.
- `rater_01_sheet.csv`, `rater_02_sheet.csv`, `rater_03_sheet.csv`: randomized response sheets for three raters.
- `pairwise_key_private.csv`: private key that maps A/B sets to methods and contains true query outcomes. Do not share it with raters before all ratings are complete.
- `study_summary.json`: sample counts and strata.
- `RATER_INSTRUCTIONS.md`: concise instructions to send to raters.

## Rater Task

For each row, compare `set_a` and `set_b` using the same rubric. Fill all 1--5 fields, then choose one pairwise preference:

- `A`: set A is more useful for triage.
- `B`: set B is more useful for triage.
- `Tie`: both are similarly useful.
- `Neither`: neither set is useful.

Use the free-text field for short reasons, especially when an evidence set appears misleading.

## Rating Rubric

Use 1--5 ordinal scores:

- Relevance: 1 = unrelated, 5 = highly similar to the query context.
- Supportiveness: 1 = contradicts or does not support escalation review, 5 = strongly supports escalation review.
- Actionability: 1 = no next-step investigation value, 5 = clearly suggests useful next steps.
- Explanation quality: 1 = unsuitable for a triage note, 5 = could be directly cited as supporting evidence.
- Misleading safety: 1 = high misleading risk, 5 = low misleading risk.

## Analysis Plan

After collecting completed rater sheets, join responses with `pairwise_key_private.csv`. Report method-level means, paired TRACER-minus-Prefix differences, bootstrap confidence intervals, and pairwise preference percentages. Agreement should be reported with ordinal agreement for Likert fields and Fleiss-style agreement for preference labels when three raters are available.

## Claim Boundary

A positive result supports the claim that TRACER analogs have higher human-rated evidence quality under the audited settings. It does not prove improved SOC efficiency, deployment-level triage gain, or universal user benefit.
"""
    (OUT_DIR / "STUDY_PROTOCOL.md").write_text(protocol, encoding="utf-8")

    instructions = """# Rater Instructions

Thank you for helping with this evidence-quality study. You will review alert-prefix cases and compare two anonymized evidence sets.

Please do not try to infer the model identity behind each set. Model identities and true future labels are intentionally hidden.

For each row:

1. Read the query channels and stage profile.
2. Read all top-5 analogs in set A and set B.
3. Score each set from 1 to 5 for relevance, supportiveness, actionability, explanation quality, and misleading safety.
4. Choose `A`, `B`, `Tie`, or `Neither` in `preferred_set`.
5. Add a short rationale when one set is clearly better, both are poor, or either set appears misleading.

Do not use the private key file. It is only for post-study analysis.
"""
    (OUT_DIR / "RATER_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blinded_rows, key_rows, summary = _build_pairs()
    _write_csv(OUT_DIR / "pairwise_items_blinded.csv", blinded_rows)
    _write_csv(OUT_DIR / "pairwise_key_private.csv", key_rows)
    _write_rater_sheets(blinded_rows, n_raters=3)
    (OUT_DIR / "study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_protocol(summary)
    print(f"wrote {len(blinded_rows)} blinded pairwise cases to {OUT_DIR}")


if __name__ == "__main__":
    main()
