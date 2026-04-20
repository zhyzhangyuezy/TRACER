from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data.dataset import load_split
from campaign_mem.models.forecasting import RetrievalForecaster, build_model
from campaign_mem.training.engine import (
    _device_from_config,
    _encode_split,
    _evaluate_retrieval,
    _make_loader,
    _train_epoch,
)
from campaign_mem.utils import load_yaml, save_json, set_seed


def _train_retrieval_model(config: dict[str, Any]) -> tuple[RetrievalForecaster, Any, Any, torch.device, int]:
    set_seed(int(config.get("seed", 7)))
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config.get("training", {})
    metric_cfg = config.get("metrics", {})

    train_split = load_split(data_cfg["dataset_dir"], data_cfg.get("train_split", "train"))
    dev_split = load_split(data_cfg["dataset_dir"], data_cfg.get("dev_split", "dev"))
    device = _device_from_config(config)
    batch_size = int(train_cfg.get("batch_size", 64))
    model = build_model(model_cfg, input_dim=train_split.feature_dim, seq_len=train_split.seq_len).to(device)
    if not isinstance(model, RetrievalForecaster):
        raise ValueError("Case diagnosis currently supports retrieval models only.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    loader = _make_loader(train_split, batch_size=batch_size, shuffle=True)
    best_state = model.state_dict()
    best_score = float("-inf")
    for _ in range(int(train_cfg.get("epochs", 6))):
        _train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
        )
        dev_metrics = _evaluate_retrieval(
            model=model,
            train_split=train_split,
            eval_split=dev_split,
            device=device,
            batch_size=batch_size,
            metric_cfg=metric_cfg,
        )
        score = dev_metrics["AUPRC"] + 0.01 * dev_metrics.get("Analog-Fidelity@5", 0.0)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, train_split, load_split(data_cfg["dataset_dir"], data_cfg.get("test_split", "test")), device, batch_size


def _collect_predictions(
    model: RetrievalForecaster,
    train_split: Any,
    eval_split: Any,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    train_embedding = _encode_split(model, train_split, device=device, batch_size=batch_size).to(device)
    memory_main = torch.from_numpy(train_split.label_main).float().to(device)
    memory_aux = torch.from_numpy(train_split.label_aux).float().to(device)
    loader = _make_loader(eval_split, batch_size=batch_size, shuffle=False)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model.forward_with_external_memory(
                prefix=batch["prefix"].to(device),
                memory_embedding=train_embedding,
                memory_main_label=memory_main,
                memory_aux_label=memory_aux,
            )
            scores = torch.sigmoid(outputs["final_main_logit"]).detach().cpu().numpy()
            indices = outputs["retrieved_indices"].detach().cpu().numpy()
            query_indices = batch["index"].detach().cpu().numpy()
            for local_idx, query_index in enumerate(query_indices):
                top1 = int(indices[local_idx, 0])
                future_distance = float(
                    np.linalg.norm(eval_split.future_signature[query_index] - train_split.future_signature[top1])
                )
                tte_error = float(
                    abs(eval_split.time_to_escalation[query_index] - train_split.time_to_escalation[top1])
                )
                rows.append(
                    {
                        "query_index": int(query_index),
                        "score": float(scores[local_idx]),
                        "retrieved_indices": [int(value) for value in indices[local_idx].tolist()],
                        "top1_incident_id": str(train_split.incident_id[top1]),
                        "top1_family_id": str(train_split.family_id[top1]),
                        "top1_timestamp": int(train_split.timestamp[top1]),
                        "top1_label_main": float(train_split.label_main[top1]),
                        "top1_tte": float(train_split.time_to_escalation[top1]),
                        "top1_future_distance": future_distance,
                        "top1_tte_error": tte_error,
                    }
                )
    return rows


def _timestamp_iso(value: int) -> str:
    return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()


def _select_cases(merged: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    positives = [row for row in merged if row["label_main"] == 1]
    negatives = [row for row in merged if row["label_main"] == 0]
    positive_margin = [row for row in positives if row["score_margin"] > 0]
    negative_margin = [row for row in positives if row["score_margin"] < 0]
    positives_sorted = sorted(positive_margin, key=lambda row: row["score_margin"], reverse=True)
    failures_sorted = sorted(negative_margin, key=lambda row: row["score_margin"])
    false_positive_sorted = sorted(
        negatives,
        key=lambda row: (row["campaign_score"] - row["prefix_score"], row["campaign_score"]),
        reverse=True,
    )
    return {
        "campaign_advantage": positives_sorted[:3],
        "campaign_failure": failures_sorted[:3],
        "campaign_false_positive": false_positive_sorted[:3],
    }


def _format_case_rows(rows: list[dict[str, Any]]) -> str:
    header = (
        "| Query | Label | Campaign | Prefix | Margin | Campaign Top1 | Prefix Top1 | "
        "Campaign Dist | Prefix Dist | Campaign TTE Err | Prefix TTE Err |\n"
        "|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|\n"
    )
    lines = []
    for row in rows:
        query = f"{row['incident_id']} @ {row['timestamp_iso']}"
        campaign_top1 = f"{row['campaign_top1_incident_id']} @ {row['campaign_top1_timestamp_iso']}"
        prefix_top1 = f"{row['prefix_top1_incident_id']} @ {row['prefix_top1_timestamp_iso']}"
        lines.append(
            f"| {query} | {row['label_main']} | {row['campaign_score']:.4f} | {row['prefix_score']:.4f} | "
            f"{row['score_margin']:.4f} | {campaign_top1} | {prefix_top1} | "
            f"{row['campaign_top1_future_distance']:.3f} | {row['prefix_top1_future_distance']:.3f} | "
            f"{row['campaign_top1_tte_error']:.1f} | {row['prefix_top1_tte_error']:.1f} |"
        )
    return header + "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export retrieval case diagnosis for prefix-only vs Campaign-MEM.")
    parser.add_argument("--prefix-config", required=True)
    parser.add_argument("--campaign-config", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    prefix_model, prefix_train, test_split, prefix_device, prefix_batch = _train_retrieval_model(load_yaml(args.prefix_config))
    prefix_rows = _collect_predictions(prefix_model, prefix_train, test_split, prefix_device, prefix_batch)

    campaign_model, campaign_train, campaign_test, campaign_device, campaign_batch = _train_retrieval_model(load_yaml(args.campaign_config))
    campaign_rows = _collect_predictions(campaign_model, campaign_train, campaign_test, campaign_device, campaign_batch)

    if test_split.size != campaign_test.size:
        raise ValueError("Prefix and Campaign configs must use the same evaluation split.")

    prefix_by_index = {row["query_index"]: row for row in prefix_rows}
    campaign_by_index = {row["query_index"]: row for row in campaign_rows}

    merged = []
    for query_index in sorted(prefix_by_index):
        prefix_row = prefix_by_index[query_index]
        campaign_row = campaign_by_index[query_index]
        merged.append(
            {
                "query_index": query_index,
                "incident_id": str(test_split.incident_id[query_index]),
                "family_id": str(test_split.family_id[query_index]),
                "timestamp": int(test_split.timestamp[query_index]),
                "timestamp_iso": _timestamp_iso(int(test_split.timestamp[query_index])),
                "label_main": int(test_split.label_main[query_index]),
                "prefix_score": prefix_row["score"],
                "campaign_score": campaign_row["score"],
                "score_margin": campaign_row["score"] - prefix_row["score"],
                "prefix_top1_incident_id": prefix_row["top1_incident_id"],
                "prefix_top1_timestamp_iso": _timestamp_iso(prefix_row["top1_timestamp"]),
                "prefix_top1_future_distance": prefix_row["top1_future_distance"],
                "prefix_top1_tte_error": prefix_row["top1_tte_error"],
                "campaign_top1_incident_id": campaign_row["top1_incident_id"],
                "campaign_top1_timestamp_iso": _timestamp_iso(campaign_row["top1_timestamp"]),
                "campaign_top1_future_distance": campaign_row["top1_future_distance"],
                "campaign_top1_tte_error": campaign_row["top1_tte_error"],
            }
        )

    selected = _select_cases(merged)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_prefix.with_suffix(".json"), {"cases": selected, "all_cases": merged})

    markdown = [
        "# Retrieval Case Diagnosis",
        "",
        "## Campaign Advantage Cases",
        "",
        _format_case_rows(selected["campaign_advantage"]),
        "## Campaign Failure Cases",
        "",
        _format_case_rows(selected["campaign_failure"]),
        "## Campaign False Positive Cases",
        "",
        _format_case_rows(selected["campaign_false_positive"]),
    ]
    output_prefix.with_suffix(".md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Saved case diagnosis to {output_prefix.with_suffix('.md')} and {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
