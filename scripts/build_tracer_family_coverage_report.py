from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FAMILY_PLAN: list[dict[str, str]] = [
    {
        "dataset": "data/atlasv2_public",
        "split": "test",
        "experiment": "r005_tcn_forecaster_atlasv2_public",
        "mode": "sparse_diverse_chrono_spiky -> TCN",
    },
    {
        "dataset": "data/atlasv2_public",
        "split": "test_event_disjoint",
        "experiment": "r215_campaign_mem_decomp_modular_patch_atlasv2_public",
        "mode": "sparse_diverse_event -> decomposition-guided TRACER core",
    },
    {
        "dataset": "data/ait_ads_public",
        "split": "test",
        "experiment": "r154_campaign_mem_auto_balanced_ait_ads_public",
        "mode": "dense_low_diversity -> balanced TRACER",
    },
    {
        "dataset": "data/ait_ads_public",
        "split": "test_event_disjoint",
        "experiment": "r068_dlinear_forecaster_ait_ads_public",
        "mode": "dense_low_diversity_event -> DLinear",
    },
    {
        "dataset": "data/atlas_raw_public",
        "split": "test",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "mode": "extreme_sparse -> conservative TRACER",
    },
    {
        "dataset": "data/atlas_raw_public",
        "split": "test_event_disjoint",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "mode": "extreme_sparse -> conservative TRACER",
    },
    {
        "dataset": "data/synthetic_cam_lds",
        "split": "test",
        "experiment": "r229_tracer_auto_synthetic_cam_lds",
        "mode": "simple_dense -> linear",
    },
    {
        "dataset": "data/synthetic_cam_lds",
        "split": "test_event_disjoint",
        "experiment": "r229_tracer_auto_synthetic_cam_lds",
        "mode": "simple_dense -> linear",
    },
    {
        "dataset": "data/atlasv2_workbook",
        "split": "test",
        "experiment": "r019_lstm_forecaster_atlasv2_workbook",
        "mode": "cold_start_sparse -> LSTM",
    },
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_row(rows: list[dict[str, Any]], experiment: str) -> tuple[int, dict[str, Any]] | None:
    for index, row in enumerate(rows, start=1):
        if str(row.get("base_experiment_name")) == experiment:
            return index, row
    return None


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = _load_json(root / "outputs" / "results" / "leaderboard_summary.json")
    lines = [
        "# TRACER Family Coverage Report",
        "",
        "- Archived ranks below are computed against the full seeded_3 leaderboard for each dataset/split, so they include historical sweeps beyond the paper's direct policy reruns.",
        "- This report is diagnostic about family coverage; it does not replace the paper's claim-bearing ATLASv2 held-out-family significance audit.",
        "",
        "| Dataset | Split | TRACER mode | Bound experiment | Archived seeded_3 rank | Mean AUPRC | Evidence |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    wins = 0
    top2 = 0
    total = 0
    for item in FAMILY_PLAN:
        dataset = item["dataset"]
        split = item["split"]
        experiment = item["experiment"]
        rows = list(summary["datasets"][dataset][split])
        seeded_rows = [row for row in rows if str(row.get("evidence")) == "seeded_3"]
        candidate_rows = seeded_rows or rows
        candidate_rows.sort(key=lambda row: float(row.get("mean_auprc", 0.0)), reverse=True)
        found = _find_row(candidate_rows, experiment)
        if found is None:
            rank = "-"
            mean_auprc = "-"
            evidence = "-"
        else:
            rank_value, row = found
            rank = str(rank_value)
            mean_auprc = f"{float(row['mean_auprc']):.4f}"
            evidence = str(row["evidence"])
            total += 1
            if rank_value == 1:
                wins += 1
            if rank_value <= 2:
                top2 += 1
        lines.append(
            "| "
            + dataset
            + " | "
            + split
            + " | "
            + item["mode"]
            + " | "
            + experiment
            + " | "
            + rank
            + " | "
            + mean_auprc
            + " | "
            + evidence
            + " |"
        )
    lines.extend(
        [
            "",
            f"- `wins`: {wins}/{total} = {wins / max(total, 1):.3f}",
            f"- `top2`: {top2}/{total} = {top2 / max(total, 1):.3f}",
        ]
    )
    output_path = root / "outputs" / "results" / "tracer_family_coverage_report.md"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
