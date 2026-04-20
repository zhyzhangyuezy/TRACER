from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROUTE_BINDING_PLAN: list[dict[str, str]] = [
    {
        "dataset": "data/atlasv2_public",
        "split": "test",
        "experiment": "r005_tcn_forecaster_atlasv2_public",
        "binding": "sparse_diverse_chrono_spiky -> TCN",
    },
    {
        "dataset": "data/atlasv2_public",
        "split": "test_event_disjoint",
        "experiment": "r215_campaign_mem_decomp_modular_patch_atlasv2_public",
        "binding": "sparse_diverse -> decomposition-guided TRACER core",
    },
    {
        "dataset": "data/ait_ads_public",
        "split": "test",
        "experiment": "r154_campaign_mem_auto_balanced_ait_ads_public",
        "binding": "dense_low_diversity -> balanced TRACER",
    },
    {
        "dataset": "data/ait_ads_public",
        "split": "test_event_disjoint",
        "experiment": "r242_tracer_adaptive_event_ait_ads_public",
        "binding": "dense_low_diversity_event -> DLinear",
    },
    {
        "dataset": "data/atlas_raw_public",
        "split": "test",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "binding": "extreme_sparse -> conservative TRACER",
    },
    {
        "dataset": "data/atlas_raw_public",
        "split": "test_event_disjoint",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "binding": "extreme_sparse -> conservative TRACER",
    },
    {
        "dataset": "data/synthetic_cam_lds",
        "split": "test",
        "experiment": "r229_tracer_auto_synthetic_cam_lds",
        "binding": "simple_dense -> linear",
    },
    {
        "dataset": "data/synthetic_cam_lds",
        "split": "test_event_disjoint",
        "experiment": "r229_tracer_auto_synthetic_cam_lds",
        "binding": "simple_dense -> linear",
    },
    {
        "dataset": "data/atlasv2_workbook",
        "split": "test",
        "experiment": "r019_lstm_forecaster_atlasv2_workbook",
        "binding": "cold_start_sparse -> LSTM",
    },
]


DIRECT_POLICY_PLAN: list[dict[str, str]] = [
    {
        "dataset": "data/atlasv2_public",
        "split": "test",
        "experiment": "r239_tracer_adaptive_chronology_atlasv2_public",
        "request": "chronology",
    },
    {
        "dataset": "data/atlasv2_public",
        "split": "test_event_disjoint",
        "experiment": "r240_tracer_adaptive_event_atlasv2_public",
        "request": "event-disjoint",
    },
    {
        "dataset": "data/ait_ads_public",
        "split": "test",
        "experiment": "r241_tracer_adaptive_ait_ads_public",
        "request": "balanced",
    },
    {
        "dataset": "data/ait_ads_public",
        "split": "test_event_disjoint",
        "experiment": "r242_tracer_adaptive_event_ait_ads_public",
        "request": "event-disjoint",
    },
    {
        "dataset": "data/atlas_raw_public",
        "split": "test",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "request": "balanced",
    },
    {
        "dataset": "data/atlas_raw_public",
        "split": "test_event_disjoint",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "request": "balanced",
    },
    {
        "dataset": "data/synthetic_cam_lds",
        "split": "test",
        "experiment": "r244_tracer_adaptive_synthetic_cam_lds",
        "request": "balanced",
    },
    {
        "dataset": "data/synthetic_cam_lds",
        "split": "test_event_disjoint",
        "experiment": "r244_tracer_adaptive_synthetic_cam_lds",
        "request": "balanced",
    },
    {
        "dataset": "data/atlasv2_workbook",
        "split": "test",
        "experiment": "r245_tracer_adaptive_atlasv2_workbook",
        "request": "balanced",
    },
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _seeded_rows(summary: dict[str, Any], dataset: str, split: str) -> list[dict[str, Any]]:
    rows = list(summary["datasets"][dataset][split])
    seeded_rows = [row for row in rows if str(row.get("evidence")) == "seeded_3"]
    seeded_rows.sort(key=lambda row: float(row.get("mean_auprc", 0.0)), reverse=True)
    return seeded_rows or rows


def _find_row(rows: list[dict[str, Any]], experiment: str) -> tuple[int, dict[str, Any]] | None:
    for index, row in enumerate(rows, start=1):
        if str(row.get("base_experiment_name")) == experiment:
            return index, row
    return None


def _summarize_plan(
    summary: dict[str, Any],
    plan: list[dict[str, str]],
    *,
    include_gap: bool,
    label_key: str,
) -> tuple[list[str], dict[str, int]]:
    lines: list[str] = []
    counts = {"strict_wins": 0, "tie_best": 0, "top2": 0, "total": 0}
    for item in plan:
        rows = _seeded_rows(summary, item["dataset"], item["split"])
        found = _find_row(rows, item["experiment"])
        if found is None:
            if include_gap:
                lines.append(
                    "| "
                    + item["dataset"]
                    + " | "
                    + item["split"]
                    + " | "
                    + item[label_key]
                    + " | "
                    + item["experiment"]
                    + " | - | - | - | - |"
                )
            else:
                lines.append(
                    "| "
                    + item["dataset"]
                    + " | "
                    + item["split"]
                    + " | "
                    + item[label_key]
                    + " | "
                    + item["experiment"]
                    + " | - | - | - |"
                )
            continue

        rank, row = found
        best = float(rows[0]["mean_auprc"])
        mean_auprc = float(row["mean_auprc"])
        tie_best = abs(mean_auprc - best) < 1e-12
        counts["total"] += 1
        counts["strict_wins"] += int(rank == 1)
        counts["tie_best"] += int(tie_best)
        counts["top2"] += int(rank <= 2)

        if include_gap:
            gap = mean_auprc - best
            lines.append(
                "| "
                + item["dataset"]
                + " | "
                + item["split"]
                + " | "
                + item[label_key]
                + " | "
                + item["experiment"]
                + " | "
                + str(rank)
                + " | "
                + f"{mean_auprc:.4f}"
                + " | "
                + f"{gap:.4f}"
                + " | "
                + ("yes" if tie_best else "no")
                + " |"
            )
        else:
            lines.append(
                "| "
                + item["dataset"]
                + " | "
                + item["split"]
                + " | "
                + item[label_key]
                + " | "
                + item["experiment"]
                + " | "
                + str(rank)
                + " | "
                + f"{mean_auprc:.4f}"
                + " | "
                + str(row.get("evidence", "-"))
                + " |"
            )
    return lines, counts


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = _load_json(root / "outputs" / "results" / "leaderboard_summary.json")

    binding_lines, binding_counts = _summarize_plan(
        summary,
        ROUTE_BINDING_PLAN,
        include_gap=False,
        label_key="binding",
    )
    policy_lines, policy_counts = _summarize_plan(
        summary,
        DIRECT_POLICY_PLAN,
        include_gap=True,
        label_key="request",
    )

    lines = [
        "# TRACER Adaptive Policy Coverage Report",
        "",
        "- Archived ranks below are computed against the full seeded_3 leaderboard for each dataset/split, so they include historical family sweeps and exploratory variants in addition to the paper's direct reruns.",
        "- On ATLASv2 held-out-family, paper-level judgment is governed by the 20-seed incident-block audit rather than by archived rank alone.",
        "",
        "## Published Route Bindings",
        "",
        "| Dataset | Split | Published route | Bound experiment | Archived seeded_3 rank | Mean AUPRC | Evidence |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
        *binding_lines,
        "",
        f"- `strict wins`: {binding_counts['strict_wins']}/{binding_counts['total']} = {binding_counts['strict_wins'] / max(binding_counts['total'], 1):.3f}",
        f"- `top2`: {binding_counts['top2']}/{binding_counts['total']} = {binding_counts['top2'] / max(binding_counts['total'], 1):.3f}",
        "",
        "## Direct tracer_adaptive Policy Reruns",
        "",
        "| Dataset | Split | Requested objective | Experiment | Archived seeded_3 rank | Mean AUPRC | Gap to best | Tie-best |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        *policy_lines,
        "",
        f"- `strict wins`: {policy_counts['strict_wins']}/{policy_counts['total']} = {policy_counts['strict_wins'] / max(policy_counts['total'], 1):.3f}",
        f"- `tie-best`: {policy_counts['tie_best']}/{policy_counts['total']} = {policy_counts['tie_best'] / max(policy_counts['total'], 1):.3f}",
        f"- `top2`: {policy_counts['top2']}/{policy_counts['total']} = {policy_counts['top2'] / max(policy_counts['total'], 1):.3f}",
        "",
        "- The first table records the exact route-table bindings implied by the published predicates; it is a diagnostic of how the frozen policy maps benchmark traits to family members.",
        "- The direct `tracer_adaptive` reruns are the claim-bearing evidence used in the paper; the main residual gaps now come from archived family-sweep headroom on ATLASv2 held-out-family, workbook stability, and a small synthetic reproduction gap.",
    ]

    output_path = root / "outputs" / "results" / "tracer_adaptive_coverage_report.md"
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
