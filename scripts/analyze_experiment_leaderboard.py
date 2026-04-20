from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SEEDS = (7, 13, 21)
EXCLUDE_TOKENS = ("manifest", "audit", "diagnosis", "analysis")
CLAIM_BEARING_ROWS = (
    {
        "dataset": "data/atlasv2_public",
        "split": "test",
        "experiment": "r239_tracer_adaptive_chronology_atlasv2_public",
        "role": "Primary chronology route in the paper; direct frozen-policy rerun.",
    },
    {
        "dataset": "data/atlasv2_public",
        "split": "test_event_disjoint",
        "experiment": "r240_tracer_adaptive_event_atlasv2_public",
        "role": "Primary held-out-family route; interpreted together with the 20-seed incident-block audit.",
    },
    {
        "dataset": "data/ait_ads_public",
        "split": "test",
        "experiment": "r241_tracer_adaptive_ait_ads_public",
        "role": "Supplementary chronology benchmark for the same frozen policy.",
    },
    {
        "dataset": "data/ait_ads_public",
        "split": "test_event_disjoint",
        "experiment": "r242_tracer_adaptive_event_ait_ads_public",
        "role": "Supplementary held-out benchmark for the same frozen policy.",
    },
    {
        "dataset": "data/atlas_raw_public",
        "split": "test",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "role": "Supplementary raw-observation benchmark under the same policy.",
    },
    {
        "dataset": "data/atlas_raw_public",
        "split": "test_event_disjoint",
        "experiment": "r243_tracer_adaptive_atlas_raw_public",
        "role": "Supplementary raw held-out benchmark under the same policy.",
    },
    {
        "dataset": "data/atlasv2_workbook",
        "split": "test",
        "experiment": "r245_tracer_adaptive_atlasv2_workbook",
        "role": "Workbook stress probe; diagnostic only, not claim-bearing.",
    },
)


def _load_result_payloads(result_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(result_dir.glob("r*.json")):
        lowered = path.name.lower()
        if any(token in lowered for token in EXCLUDE_TOKENS):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        if "dataset_dir" not in payload or "test" not in payload or "model" not in payload:
            continue
        payloads[path.stem] = payload
    return payloads


def _group_runs(payloads: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for stem, payload in payloads.items():
        if any(stem.endswith(f"_seed{seed}") for seed in SEEDS):
            continue
        seed_items = [payloads[f"{stem}_seed{seed}"] for seed in SEEDS if f"{stem}_seed{seed}" in payloads]
        if seed_items:
            evidence = "seeded_3" if len(seed_items) == len(SEEDS) else f"seeded_partial_{len(seed_items)}"
            grouped[stem] = {"items": seed_items, "evidence": evidence}
        else:
            grouped[stem] = {"items": [payload], "evidence": "single_run"}
    for stem, payload in payloads.items():
        for seed in SEEDS:
            suffix = f"_seed{seed}"
            if not stem.endswith(suffix):
                continue
            base = stem[: -len(suffix)]
            if base not in grouped:
                seed_items = [payloads[f"{base}_seed{item_seed}"] for item_seed in SEEDS if f"{base}_seed{item_seed}" in payloads]
                evidence = "seeded_3" if len(seed_items) == len(SEEDS) else f"seeded_partial_{len(seed_items)}"
                grouped[base] = {"items": seed_items, "evidence": evidence}
            break
    return grouped


def _metric_values(items: list[dict[str, Any]], split_key: str, metric_key: str = "AUPRC") -> list[float]:
    values: list[float] = []
    for item in items:
        split = item.get(split_key)
        if isinstance(split, dict) and metric_key in split and isinstance(split[metric_key], (int, float)):
            values.append(float(split[metric_key]))
    return values


def _aggregate_numeric_metrics(items: list[dict[str, Any]], split_key: str) -> tuple[dict[str, float], dict[str, float]]:
    collected: dict[str, list[float]] = {}
    for item in items:
        split = item.get(split_key)
        if not isinstance(split, dict):
            continue
        for key, value in split.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            collected.setdefault(str(key), []).append(float(value))
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key, values in collected.items():
        means[key] = mean(values)
        stds[key] = pstdev(values) if len(values) > 1 else 0.0
    return means, stds


def _config_path(root: Path, base_name: str) -> str | None:
    path = root / "configs" / "experiments" / f"{base_name}.yaml"
    return str(path.relative_to(root)).replace("\\", "/") if path.exists() else None


def build_leaderboard(root: Path) -> dict[str, Any]:
    result_dir = root / "outputs" / "results"
    payloads = _load_result_payloads(result_dir)
    grouped = _group_runs(payloads)
    report: dict[str, Any] = {"datasets": {}}
    for base_name, grouped_entry in sorted(grouped.items()):
        items = list(grouped_entry["items"])
        if not items:
            continue
        dataset_dir = str(items[0]["dataset_dir"]).replace("\\", "/")
        evidence = str(grouped_entry["evidence"])
        dataset_entry = report["datasets"].setdefault(dataset_dir, {"test": [], "test_event_disjoint": []})
        if "event_disjoint" not in base_name:
            chrono_values = _metric_values(items, "test")
            if chrono_values:
                metric_means, metric_stds = _aggregate_numeric_metrics(items, "test")
                dataset_entry["test"].append(
                    {
                        "base_experiment_name": base_name,
                        "mean_auprc": mean(chrono_values),
                        "std_auprc": pstdev(chrono_values) if len(chrono_values) > 1 else 0.0,
                        "metrics_mean": metric_means,
                        "metrics_std": metric_stds,
                        "n": len(chrono_values),
                        "evidence": evidence,
                        "config_path": _config_path(root, base_name),
                    }
                )
        has_event_block = any(isinstance(item.get("test_event_disjoint"), dict) for item in items)
        event_split = "test_event_disjoint" if has_event_block else ("test" if "event_disjoint" in base_name else None)
        if event_split is not None:
            event_values = _metric_values(items, event_split)
            if event_values:
                metric_means, metric_stds = _aggregate_numeric_metrics(items, event_split)
                dataset_entry["test_event_disjoint"].append(
                    {
                        "base_experiment_name": base_name,
                        "mean_auprc": mean(event_values),
                        "std_auprc": pstdev(event_values) if len(event_values) > 1 else 0.0,
                        "metrics_mean": metric_means,
                        "metrics_std": metric_stds,
                        "n": len(event_values),
                        "evidence": evidence,
                        "config_path": _config_path(root, base_name),
                    }
                )
    for dataset_entry in report["datasets"].values():
        for split_name in ("test", "test_event_disjoint"):
            dataset_entry[split_name].sort(key=lambda row: row["mean_auprc"], reverse=True)
    return report


def _format_metric_value(row: dict[str, Any], key: str) -> str:
    metrics_mean = row.get("metrics_mean", {})
    if not isinstance(metrics_mean, dict) or key not in metrics_mean:
        return "-"
    return f"{float(metrics_mean[key]):.4f}"


def _format_rows(rows: list[dict[str, Any]], top_k: int, *, evidence: str | None) -> list[str]:
    filtered = [row for row in rows if evidence is None or row["evidence"] == evidence][:top_k]
    if not filtered:
        return [
            "| - | - | - | - | - | - | - | - |",
            "| - | - | - | - | - | - | - | - |",
        ]
    lines = [
        "| Rank | Experiment | Mean AUPRC | AUROC | BestF1 | Recall@P80 | ECE@10 | Evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for index, row in enumerate(filtered, start=1):
        lines.append(
            "| "
            + str(index)
            + " | "
            + str(row["base_experiment_name"])
            + " | "
            + f"{row['mean_auprc']:.4f}"
            + " | "
            + _format_metric_value(row, "AUROC")
            + " | "
            + _format_metric_value(row, "BestF1")
            + " | "
            + _format_metric_value(row, "Recall@P80")
            + " | "
            + _format_metric_value(row, "ECE@10")
            + " | "
            + str(row["evidence"])
            + " |"
        )
    return lines


def _claim_bearing_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Paper Claim-Bearing Rows",
        "",
        "| Dataset | Split | Experiment | Archived seeded_3 rank | Mean AUPRC | Evidence | Paper role |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for item in CLAIM_BEARING_ROWS:
        dataset_entry = report["datasets"].get(item["dataset"], {})
        rows = list(dataset_entry.get(item["split"], []))
        seeded_rows = [row for row in rows if row.get("evidence") == "seeded_3"]
        candidate_rows = seeded_rows or rows
        found = None
        for index, row in enumerate(candidate_rows, start=1):
            if row.get("base_experiment_name") == item["experiment"]:
                found = (index, row)
                break
        if found is None:
            lines.append(
                "| "
                + item["dataset"]
                + " | "
                + item["split"]
                + " | "
                + item["experiment"]
                + " | - | - | - | "
                + item["role"]
                + " |"
            )
            continue
        rank, row = found
        lines.append(
            "| "
            + item["dataset"]
            + " | "
            + item["split"]
            + " | "
            + item["experiment"]
            + " | "
            + str(rank)
            + " | "
            + f"{float(row['mean_auprc']):.4f}"
            + " | "
            + str(row["evidence"])
            + " | "
            + item["role"]
            + " |"
        )
    return lines


def write_markdown(report: dict[str, Any], output_path: Path, top_k: int) -> None:
    lines = [
        "# Experiment Leaderboard",
        "",
        "- This report ranks all archived experiments found in `outputs/results`, including historical sweeps and exploratory family variants.",
        "- These leaderboard ranks are therefore broader than the paper's claim gate; for ATLASv2 held-out-family, the manuscript relies on the frozen-policy rerun plus the 20-seed incident-block audit rather than leaderboard rank alone.",
        "",
    ]
    lines.extend(_claim_bearing_lines(report))
    lines.extend([""])
    for dataset_name, dataset_entry in sorted(report["datasets"].items()):
        lines.extend([f"## {dataset_name}", ""])
        for split_name in ("test", "test_event_disjoint"):
            lines.extend([f"### {split_name}", "", "Seeded three-run leaderboard:", ""])
            lines.extend(_format_rows(dataset_entry[split_name], top_k, evidence="seeded_3"))
            lines.extend(["", "Seeded partial-progress leaderboard:", ""])
            lines.extend(_format_rows(dataset_entry[split_name], top_k, evidence="seeded_partial_1"))
            partial_two = _format_rows(dataset_entry[split_name], top_k, evidence="seeded_partial_2")
            if partial_two[0] != "| - | - | - | - | - | - | - | - |":
                lines.extend(["", "Seeded two-run leaderboard:", ""])
                lines.extend(partial_two)
            lines.extend(["", "Exploratory single-run leaderboard:", ""])
            lines.extend(_format_rows(dataset_entry[split_name], top_k, evidence="single_run"))
            lines.extend([""])
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seeded-vs-exploratory experiment leaderboards.")
    parser.add_argument(
        "--output-json",
        default="outputs/results/leaderboard_summary.json",
        help="Path to the JSON leaderboard summary.",
    )
    parser.add_argument(
        "--output-markdown",
        default="outputs/results/leaderboard_summary.md",
        help="Path to the Markdown leaderboard summary.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Rows to keep per section in the Markdown report.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    report = build_leaderboard(root)
    output_json = root / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_markdown = root / args.output_markdown
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report, output_markdown, max(args.top_k, 1))

    print(f"JSON: {output_json}")
    print(f"Markdown: {output_markdown}")


if __name__ == "__main__":
    main()
