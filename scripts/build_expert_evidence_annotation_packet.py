from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "expert_evidence_annotation_packet"

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
        "sample_queries": 30,
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
        "sample_queries": 50,
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


def _active_channels(prefix: np.ndarray, channels: list[str], allowed: list[str] | set[str]) -> list[str]:
    allowed_set = set(allowed)
    active = []
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


def _select_queries(y_true: np.ndarray, y_score: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    seen: set[int] = set()

    def add_many(candidates: list[int], limit: int) -> None:
        nonlocal selected, seen
        added = 0
        for idx in candidates:
            if idx in seen:
                continue
            selected.append(idx)
            seen.add(idx)
            added += 1
            if added >= limit:
                break

    positives = np.flatnonzero(y_true.astype(bool)).tolist()
    rng.shuffle(positives)
    add_many(positives, max(1, n // 3))

    top_count = max(1, int(np.ceil(0.10 * y_score.shape[0])))
    top_indices = np.argsort(-y_score, kind="mergesort")[:top_count].tolist()
    add_many(top_indices, max(1, n // 3))

    mid_order = np.argsort(np.abs(y_score - np.median(y_score)), kind="mergesort").tolist()
    add_many(mid_order, max(1, n // 6))

    remaining = [idx for idx in range(y_score.shape[0]) if idx not in seen]
    rng.shuffle(remaining)
    add_many(remaining, max(0, n - len(selected)))
    return np.asarray(selected[:n], dtype=int)


def _score_band(score: float, y_score: np.ndarray) -> str:
    rank = float((y_score <= score).mean())
    if rank >= 0.90:
        return "top_decile"
    if rank >= 0.75:
        return "high_quartile"
    if rank <= 0.25:
        return "low_quartile"
    return "middle"


def _load_split(dataset_dir: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(dataset_dir / f"{split}.npz", allow_pickle=True) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files}


def _build_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"settings": [], "total_items": 0}

    for spec in SETTINGS:
        dataset_dir = ROOT / str(spec["dataset_dir"])
        metadata = _load_json(ROOT / str(spec["metadata"]))
        channels = [str(item) for item in metadata["feature_channels"]]
        train = _load_split(dataset_dir, "train")
        test = _load_split(dataset_dir, str(spec["test_split"]))
        stage_profiles = _stage_profiles(metadata)

        first_payload = _load_json(ROOT / next(iter(spec["methods"].values())))
        first_pred = first_payload["predictions"]
        y_true = np.asarray(first_pred["y_true"], dtype=int)
        y_score = np.asarray(first_pred["y_score"], dtype=np.float64)
        selected_queries = _select_queries(y_true, y_score, int(spec["sample_queries"]), int(spec["seed"]))

        setting_summary = {
            "dataset": spec["dataset"],
            "seed": spec["seed"],
            "sampled_queries": int(selected_queries.shape[0]),
            "sampled_positive_queries": int(y_true[selected_queries].sum()),
            "methods": list(spec["methods"].keys()),
        }
        summary["settings"].append(setting_summary)

        for method, template in spec["methods"].items():
            payload = _load_json(ROOT / str(template))
            pred = payload["predictions"]
            method_score = np.asarray(pred["y_score"], dtype=np.float64)
            retrieved_indices = np.asarray(pred["retrieved_indices"], dtype=int)
            if retrieved_indices.ndim == 1:
                retrieved_indices = retrieved_indices[:, None]
            retrieved_labels = np.asarray(pred.get("retrieved_label_main", np.zeros_like(retrieved_indices)), dtype=float)
            if retrieved_labels.ndim == 1:
                retrieved_labels = retrieved_labels[:, None]

            incident_ids = np.asarray(pred.get("incident_id", test.get("incident_id", np.arange(y_true.shape[0]))), dtype=object)
            family_ids = np.asarray(pred.get("family_id", test.get("family_id", np.asarray(["unknown"] * y_true.shape[0]))), dtype=object)
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

            for query_idx in selected_queries.tolist():
                item_hash = _anon(f"{spec['dataset']}|{method}|{query_idx}|{spec['seed']}", "expert-packet")
                item_id = f"item_{item_hash}"
                query_prefix = np.asarray(test["prefix"][query_idx], dtype=np.float32)
                query_channels = _active_channels(query_prefix, channels, KNOWLEDGE_CHANNELS)
                query_high = [name for name in query_channels if name in HIGH_RISK_CHANNELS]
                query_incident = str(incident_ids[query_idx])
                query_profile = sorted(stage_profiles.get(query_incident, set()))

                blind: dict[str, Any] = {
                    "item_id": item_id,
                    "dataset": spec["dataset"],
                    "query_ref": f"q_{_anon(query_incident, 'query')}",
                    "score_band": _score_band(float(method_score[query_idx]), method_score),
                    "query_active_channels": ";".join(query_channels) if query_channels else "none",
                    "query_high_risk_channels": ";".join(query_high) if query_high else "none",
                    "query_stage_profile": ";".join(query_profile) if query_profile else "not_provided",
                    "rating_semantic_similarity_0_2": "",
                    "rating_shared_high_risk_precursor_0_1": "",
                    "rating_supports_escalation_0_2": "",
                    "rating_misleading_0_1": "",
                    "rating_confidence_1_5": "",
                    "free_text_rationale": "",
                }
                key: dict[str, Any] = {
                    "item_id": item_id,
                    "dataset": spec["dataset"],
                    "method": method,
                    "query_index": query_idx,
                    "query_incident_id": query_incident,
                    "query_family_id": str(family_ids[query_idx]),
                    "y_true": int(y_true[query_idx]),
                    "y_score": float(method_score[query_idx]),
                    "time_to_escalation": float(np.asarray(pred["time_to_escalation"], dtype=float)[query_idx]),
                    "seed": int(spec["seed"]),
                }

                for rank in range(min(5, retrieved_indices.shape[1])):
                    neighbor_idx = int(retrieved_indices[query_idx, rank])
                    neighbor_prefix = np.asarray(train["prefix"][neighbor_idx], dtype=np.float32)
                    neighbor_channels = _active_channels(neighbor_prefix, channels, KNOWLEDGE_CHANNELS)
                    neighbor_high = [name for name in neighbor_channels if name in HIGH_RISK_CHANNELS]
                    neighbor_incident = str(retrieved_incidents[query_idx, rank])
                    neighbor_profile = sorted(stage_profiles.get(neighbor_incident, set()))
                    blind[f"analog{rank + 1}_ref"] = f"a_{_anon(neighbor_incident + ':' + str(rank), 'analog')}"
                    blind[f"analog{rank + 1}_active_channels"] = ";".join(neighbor_channels) if neighbor_channels else "none"
                    blind[f"analog{rank + 1}_high_risk_channels"] = ";".join(neighbor_high) if neighbor_high else "none"
                    blind[f"analog{rank + 1}_stage_profile"] = ";".join(neighbor_profile) if neighbor_profile else "not_provided"
                    key[f"analog{rank + 1}_train_index"] = neighbor_idx
                    key[f"analog{rank + 1}_incident_id"] = neighbor_incident
                    key[f"analog{rank + 1}_family_id"] = str(retrieved_families[query_idx, rank])
                    key[f"analog{rank + 1}_label"] = int(retrieved_labels[query_idx, rank] > 0.5)

                blinded_rows.append(blind)
                key_rows.append(key)

    order = np.random.default_rng(20260418).permutation(len(blinded_rows))
    blinded_rows = [blinded_rows[int(idx)] for idx in order]
    summary["total_items"] = len(blinded_rows)
    return blinded_rows, key_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_protocol(summary: dict[str, Any]) -> None:
    text = f"""# Expert Evidence-Rating Packet

This packet is prepared for a future human analyst study. It is not a completed user study and should not be reported as human-evaluation evidence until independent raters fill the blinded CSV.

## Files

- `annotation_items_blinded.csv`: blinded items for raters. It hides the model family, true future label, exact score, incident IDs, and retrieved train labels.
- `annotation_key_private.csv`: private key for post-study analysis. Do not share this file with raters.
- `annotation_packet_summary.json`: sampling summary and item counts.

## Sampling

The packet contains {summary["total_items"]} item-method rows from ATLASv2 held-out-family and AIT-ADS chronology. Queries are sampled deterministically from positives, top-decile scores, median-score windows, and random background windows. Each sampled query is paired across TRACER route and Prefix-Only retrieval so a paired analysis can compare evidence quality without exposing the method name to raters.

## Rating Rubric

For each item, inspect the query channels and the top-5 retrieved analog channel summaries.

- `semantic_similarity_0_2`: 0 = unrelated, 1 = partially related, 2 = clearly related.
- `shared_high_risk_precursor_0_1`: 1 if at least one analog shares a plausible high-risk precursor with the query.
- `supports_escalation_0_2`: 0 = does not support escalation, 1 = weak support, 2 = strong support.
- `misleading_0_1`: 1 if the analog set appears likely to mislead escalation judgment.
- `confidence_1_5`: rater confidence in the above judgment.
- `free_text_rationale`: short reason, especially when marking misleading evidence.

## Recommended Study Protocol

Use 2 or 3 raters with security operations or intrusion-analysis experience. Randomize item order per rater. Compute Cohen/Fleiss kappa for binary fields, weighted kappa or ICC for ordinal fields, and paired bootstrap differences between TRACER and Prefix-Only after joining ratings with `annotation_key_private.csv`.

## Claim Boundary

This packet only prepares a human evidence-rating study. Until ratings are collected, it should be cited as a released annotation protocol/artifact, not as evidence that TRACER improves analyst decisions.
"""
    (OUT_DIR / "ANNOTATION_PROTOCOL.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blinded_rows, key_rows, summary = _build_rows()
    _write_csv(OUT_DIR / "annotation_items_blinded.csv", blinded_rows)
    _write_csv(OUT_DIR / "annotation_key_private.csv", key_rows)
    (OUT_DIR / "annotation_packet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_protocol(summary)
    print(f"wrote {len(blinded_rows)} blinded annotation items to {OUT_DIR}")


if __name__ == "__main__":
    main()
