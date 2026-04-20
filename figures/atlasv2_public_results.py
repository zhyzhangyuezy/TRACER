from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PUBLIC_DATASET_DIR = DATA_DIR / "atlasv2_public"
AIT_ADS_DATASET_DIR = DATA_DIR / "ait_ads_public"
RAW_PUBLIC_DATASET_DIR = DATA_DIR / "atlas_raw_public"
WORKBOOK_DATASET_DIR = DATA_DIR / "atlasv2_workbook"
REFRESH_PUBLIC_SUITE_RESULT_DIR = RESULT_DIR / "kbs_metric_refresh_public_suite"
SEEDS = [7, 13, 21]

CHRONO_MAIN = [
    ("r004_tail_risk_linear_atlasv2_public", "LR-TailRisk"),
    ("r020_dlinear_forecaster_atlasv2_public", "DLinear-Forecaster"),
    ("r023_timesnet_forecaster_atlasv2_public", "TimesNet-Forecaster"),
    ("r024_tide_forecaster_atlasv2_public", "TiDE-Forecaster"),
    ("r025_tsmixer_forecaster_atlasv2_public", "TSMixer-Forecaster"),
    ("r005_tcn_forecaster_atlasv2_public", "TCN-Forecaster"),
    ("r018_lstm_forecaster_atlasv2_public", "LSTM-Forecaster"),
    ("r021_patchtst_forecaster_atlasv2_public", "PatchTST-Forecaster"),
    ("r006_transformer_forecaster_atlasv2_public", "Small-Transformer-Forecaster"),
    ("r022_itransformer_forecaster_atlasv2_public", "iTransformer-Forecaster"),
    ("r007_pure_knn_atlasv2_public", "Pure-kNN-Retrieval"),
    ("r008_prefix_retrieval_atlasv2_public", "Prefix-Only-Retrieval + Fusion"),
    ("r239_tracer_adaptive_chronology_atlasv2_public", "TRACER"),
]

EVENT_DISJOINT = [
    ("r020_dlinear_forecaster_atlasv2_public_event_disjoint", "DLinear-Forecaster"),
    ("r023_timesnet_forecaster_atlasv2_public_event_disjoint", "TimesNet-Forecaster"),
    ("r024_tide_forecaster_atlasv2_public_event_disjoint", "TiDE-Forecaster"),
    ("r025_tsmixer_forecaster_atlasv2_public_event_disjoint", "TSMixer-Forecaster"),
    ("r010_tcn_forecaster_atlasv2_public_event_disjoint", "TCN-Forecaster"),
    ("r018_lstm_forecaster_atlasv2_public_event_disjoint", "LSTM-Forecaster"),
    ("r021_patchtst_forecaster_atlasv2_public_event_disjoint", "PatchTST-Forecaster"),
    ("r010_transformer_forecaster_atlasv2_public_event_disjoint", "Small-Transformer-Forecaster"),
    ("r022_itransformer_forecaster_atlasv2_public_event_disjoint", "iTransformer-Forecaster"),
    ("r010_prefix_retrieval_atlasv2_public_event_disjoint", "Prefix-Only-Retrieval + Fusion"),
    ("r240_tracer_adaptive_event_atlasv2_public", "TRACER"),
]

EVENT_DISJOINT_EXTENDED = [
    ("r015_tail_risk_linear_atlasv2_public_event_disjoint", "LR-TailRisk"),
    ("r020_dlinear_forecaster_atlasv2_public_event_disjoint", "DLinear-Forecaster"),
    ("r023_timesnet_forecaster_atlasv2_public_event_disjoint", "TimesNet-Forecaster"),
    ("r024_tide_forecaster_atlasv2_public_event_disjoint", "TiDE-Forecaster"),
    ("r025_tsmixer_forecaster_atlasv2_public_event_disjoint", "TSMixer-Forecaster"),
    ("r010_tcn_forecaster_atlasv2_public_event_disjoint", "TCN-Forecaster"),
    ("r018_lstm_forecaster_atlasv2_public_event_disjoint", "LSTM-Forecaster"),
    ("r021_patchtst_forecaster_atlasv2_public_event_disjoint", "PatchTST-Forecaster"),
    ("r010_transformer_forecaster_atlasv2_public_event_disjoint", "Small-Transformer-Forecaster"),
    ("r022_itransformer_forecaster_atlasv2_public_event_disjoint", "iTransformer-Forecaster"),
    ("r015_pure_knn_atlasv2_public_event_disjoint", "Pure-kNN-Retrieval"),
    ("r015_random_retrieval_atlasv2_public_event_disjoint", "Random-Retrieval + Fusion"),
    ("r010_prefix_retrieval_atlasv2_public_event_disjoint", "Prefix-Only-Retrieval + Fusion"),
    ("r240_tracer_adaptive_event_atlasv2_public", "TRACER"),
]

ABLATIONS_ALL = [
    ("r011_no_memory_forecaster_atlasv2_public", "No-Memory Forecaster"),
    ("r007_pure_knn_atlasv2_public", "Pure-kNN-Retrieval"),
    ("r008_prefix_retrieval_atlasv2_public", "Prefix-Only-Retrieval + Fusion"),
    ("r009_campaign_mem_atlasv2_public", "Shared-Encoder TRACER"),
    ("r215_campaign_mem_decomp_modular_patch_atlasv2_public", "TRACER Core Mode"),
    ("r239_tracer_adaptive_chronology_atlasv2_public", "TRACER (adaptive policy)"),
    ("r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public", "Event-Focused TRACER Variant"),
    ("r120_campaign_mem_dual_selector_no_auxiliary_atlasv2_public", "Chronology Support Line"),
    ("r218_campaign_mem_decomp_modular_patch_noaux_atlasv2_public", "TRACER w/o auxiliary horizon"),
    ("r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public", "Held-Out-Family Support Line"),
]

ABLATIONS_RETRIEVAL = [
    ("r007_pure_knn_atlasv2_public", "Pure-kNN-Retrieval"),
    ("r008_prefix_retrieval_atlasv2_public", "Prefix-Only-Retrieval + Fusion"),
    ("r009_campaign_mem_atlasv2_public", "Shared-Encoder TRACER"),
    ("r215_campaign_mem_decomp_modular_patch_atlasv2_public", "TRACER Core Mode"),
    ("r239_tracer_adaptive_chronology_atlasv2_public", "TRACER (adaptive policy)"),
    ("r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public", "Event-Focused TRACER Variant"),
    ("r120_campaign_mem_dual_selector_no_auxiliary_atlasv2_public", "Chronology Support Line"),
    ("r218_campaign_mem_decomp_modular_patch_noaux_atlasv2_public", "TRACER w/o auxiliary horizon"),
    ("r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public", "Held-Out-Family Support Line"),
]

EVENT_DISJOINT_RETRIEVAL = [
    ("r015_pure_knn_atlasv2_public_event_disjoint", "Pure-kNN-Retrieval"),
    ("r010_prefix_retrieval_atlasv2_public_event_disjoint", "Prefix-Only-Retrieval + Fusion"),
    ("r010_campaign_mem_atlasv2_public_event_disjoint", "Shared-Encoder TRACER"),
    ("r215_campaign_mem_decomp_modular_patch_atlasv2_public", "TRACER Core Mode"),
    ("r240_tracer_adaptive_event_atlasv2_public", "TRACER (adaptive policy)"),
    ("r115_campaign_mem_dual_selector_proxy_strict_atlasv2_public", "Event-Focused TRACER Variant"),
    ("r120_campaign_mem_dual_selector_no_auxiliary_atlasv2_public", "Chronology Support Line"),
    ("r218_campaign_mem_decomp_modular_patch_noaux_atlasv2_public", "TRACER w/o auxiliary horizon"),
    ("r201_campaign_mem_modular_delta_router_mid_soft_proxy_top3_later_atlasv2_public", "Held-Out-Family Support Line"),
]

ROBUSTNESS = [
    ("r019_tail_risk_linear_atlasv2_workbook", "LR-TailRisk"),
    ("r019_tcn_forecaster_atlasv2_workbook", "TCN-Forecaster"),
    ("r019_lstm_forecaster_atlasv2_workbook", "LSTM-Forecaster"),
    ("r013_transformer_atlasv2_workbook", "Small-Transformer-Forecaster"),
    ("r013_prefix_retrieval_atlasv2_workbook", "Prefix-Only-Retrieval + Fusion"),
    ("r245_tracer_adaptive_atlasv2_workbook", "TRACER"),
    ("r013_campaign_mem_atlasv2_workbook", "Shared-Encoder TRACER"),
]

AIT_ADS_MAIN = [
    ("r068_dlinear_forecaster_ait_ads_public", "DLinear-Forecaster"),
    ("r069_tcn_forecaster_ait_ads_public", "TCN-Forecaster"),
    ("r070_transformer_forecaster_ait_ads_public", "Small-Transformer-Forecaster"),
    ("r071_prefix_retrieval_ait_ads_public", "Prefix-Only-Retrieval + Fusion"),
    ("r241_tracer_adaptive_ait_ads_public", "TRACER"),
    ("r081_campaign_mem_staged_ait_ads_public", "Staged TRACER"),
    ("r098_campaign_mem_abstain_ait_ads_public", "Abstention-Aware TRACER"),
]

AIT_ADS_EVENT_MAIN = [
    ("r068_dlinear_forecaster_ait_ads_public", "DLinear-Forecaster"),
    ("r069_tcn_forecaster_ait_ads_public", "TCN-Forecaster"),
    ("r070_transformer_forecaster_ait_ads_public", "Small-Transformer-Forecaster"),
    ("r071_prefix_retrieval_ait_ads_public", "Prefix-Only-Retrieval + Fusion"),
    ("r242_tracer_adaptive_event_ait_ads_public", "TRACER"),
    ("r081_campaign_mem_staged_ait_ads_public", "Staged TRACER"),
    ("r098_campaign_mem_abstain_ait_ads_public", "Abstention-Aware TRACER"),
]

RAW_PUBLIC_MAIN = [
    ("r026_tail_risk_linear_atlas_raw_public", "LR-TailRisk"),
    ("r027_dlinear_forecaster_atlas_raw_public", "DLinear-Forecaster"),
    ("r031_tcn_forecaster_atlas_raw_public", "TCN-Forecaster"),
    ("r032_transformer_forecaster_atlas_raw_public", "Small-Transformer-Forecaster"),
    ("r035_prefix_retrieval_atlas_raw_public", "Prefix-Only-Retrieval + Fusion"),
    ("r036_campaign_mem_atlas_raw_public", "Shared-Encoder TRACER"),
    ("r047_campaign_mem_v3_dlinear_atlas_raw_public", "DLinear-Calibrated TRACER"),
    ("r243_tracer_adaptive_atlas_raw_public", "TRACER"),
    ("r089_campaign_mem_selector_conservative_atlas_raw_public", "Selector-Conservative TRACER"),
    ("r121_campaign_mem_dual_selector_proxy_strict_atlas_raw_public", "ATLASv2-Tuned TRACER Core Transfer"),
]

PRIMARY_METRICS = ["AUPRC", "LeadTime@P80", "Brier", "Analog-Fidelity@5", "TTE-Err@1"]
SUPPLEMENTARY_METRICS = ["AUROC", "LogLoss", "ECE@10", "BestF1", "Precision@P80", "Recall@P80"]
AGGREGATE_METRICS = PRIMARY_METRICS + SUPPLEMENTARY_METRICS
LOWER_IS_BETTER = {"Brier", "TTE-Err@1", "LogLoss", "ECE@10"}
DEFAULT_METRIC_DECIMALS = {
    "AUPRC": 3,
    "AUROC": 3,
    "LeadTime@P80": 2,
    "Precision@P80": 3,
    "Recall@P80": 3,
    "Brier": 3,
    "LogLoss": 3,
    "ECE@10": 3,
    "BestF1": 3,
    "Analog-Fidelity@5": 2,
    "TTE-Err@1": 2,
}


@dataclass
class SplitBundle:
    name: str
    label_main: np.ndarray
    incident_id: np.ndarray
    family_id: np.ndarray

    @property
    def size(self) -> int:
        return int(self.label_main.shape[0])


def load_metadata(dataset_dir: str | Path) -> dict:
    meta_path = Path(dataset_dir) / "metadata.json"
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split(dataset_dir: str | Path, split_name: str) -> SplitBundle:
    split_path = Path(dataset_dir) / f"{split_name}.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    data = np.load(split_path, allow_pickle=True)
    return SplitBundle(
        name=split_name,
        label_main=data["label_main"].astype(np.float32),
        incident_id=data["incident_id"].astype(str),
        family_id=data["family_id"].astype(str),
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _seed_paths(base_experiment_name: str, result_dir: Path | None = None) -> list[Path]:
    active_dir = RESULT_DIR if result_dir is None else Path(result_dir)
    return [active_dir / f"{base_experiment_name}_seed{seed}.json" for seed in SEEDS]


def _metric_from_row(row: dict[str, object], metric_key: str) -> float | None:
    if metric_key in row:
        value = row.get(metric_key)
        return float(value) if value is not None else None
    if metric_key == "Precision@P80":
        detail = row.get("LeadTimeDetail")
        if isinstance(detail, dict):
            value = detail.get("precision")
            return float(value) if value is not None else None
    if metric_key == "Recall@P80":
        detail = row.get("LeadTimeDetail")
        if isinstance(detail, dict):
            value = detail.get("recall")
            return float(value) if value is not None else None
    return None


def aggregate_run(
    base_experiment_name: str,
    display_name: str,
    split: str = "test",
    result_dir: Path | None = None,
) -> dict[str, object]:
    active_dir = RESULT_DIR if result_dir is None else Path(result_dir)
    seeded_paths = _seed_paths(base_experiment_name, result_dir=active_dir)
    payloads: list[dict] = []
    if all(path.exists() for path in seeded_paths):
        payloads = [_load_json(path) for path in seeded_paths]
        source = "seed_sweep"
    else:
        payloads = [_load_json(active_dir / f"{base_experiment_name}.json")]
        source = "single_run"

    resolved_split = split
    if split == "test" and "event_disjoint" in base_experiment_name:
        resolved_split = "test_event_disjoint"
    rows = [payload.get(resolved_split, payload["test"]) for payload in payloads]
    metrics: dict[str, float | None] = {}
    stds: dict[str, float | None] = {}
    raw_values: dict[str, list[float]] = {}
    for metric_key in AGGREGATE_METRICS:
        values = []
        for row in rows:
            value = _metric_from_row(row, metric_key)
            if value is not None:
                values.append(value)
        raw_values[metric_key] = values
        if values:
            metrics[metric_key] = mean(values)
            stds[metric_key] = pstdev(values) if len(values) > 1 else 0.0
        else:
            metrics[metric_key] = None
            stds[metric_key] = None

    return {
        "base_experiment_name": base_experiment_name,
        "display_name": display_name,
        "split": resolved_split,
        "n": len(rows),
        "source": source,
        "metrics": metrics,
        "std": stds,
        "values": raw_values,
    }


def collect_rows(
    specs: list[tuple[str, str]],
    split: str = "test",
    result_dir: Path | None = None,
) -> list[dict[str, object]]:
    return [aggregate_run(base_name, display_name, split=split, result_dir=result_dir) for base_name, display_name in specs]


def _format_metric(value: float | None, std: float | None, decimals: int = 3, bold: bool = False) -> str:
    if value is None:
        return "--"
    if std is None:
        body = f"{value:.{decimals}f}"
    else:
        body = f"{value:.{decimals}f} \\pm {std:.{decimals}f}"
    if bold:
        return f"$\\mathbf{{{body}}}$"
    return f"${body}$"


def _format_metric_emph(
    value: float | None,
    std: float | None,
    decimals: int = 3,
    *,
    bold: bool = False,
    underline: bool = False,
) -> str:
    if value is None:
        return "--"
    if std is None:
        body = f"{value:.{decimals}f}"
    else:
        body = f"{value:.{decimals}f} \\pm {std:.{decimals}f}"
    if bold:
        return f"$\\mathbf{{{body}}}$"
    if underline:
        return f"$\\underline{{{body}}}$"
    return f"${body}$"


def _format_scalar(value: float | None, decimals: int = 3) -> str:
    if value is None:
        return "--"
    return f"${value:.{decimals}f}$"


def _metric_display_value(
    row: dict[str, object],
    metric_key: str,
    decimals_by_metric: dict[str, int] | None = None,
) -> float | None:
    value = row["metrics"].get(metric_key)
    if value is None:
        return None
    decimals = (decimals_by_metric or DEFAULT_METRIC_DECIMALS).get(metric_key, 3)
    return round(float(value), decimals)


def _best_methods(
    rows: list[dict[str, object]],
    metric_keys: list[str],
    decimals_by_metric: dict[str, int] | None = None,
) -> dict[str, set[str]]:
    best_by_metric: dict[str, set[str]] = {}
    for metric_key in metric_keys:
        available_rows = [row for row in rows if _metric_display_value(row, metric_key, decimals_by_metric) is not None]
        if not available_rows:
            best_by_metric[metric_key] = set()
            continue
        values = [_metric_display_value(row, metric_key, decimals_by_metric) for row in available_rows]
        best_value = min(values) if metric_key in LOWER_IS_BETTER else max(values)
        best_by_metric[metric_key] = {
            str(row["display_name"])
            for row in available_rows
            if abs(_metric_display_value(row, metric_key, decimals_by_metric) - best_value) <= 1e-12
        }
    return best_by_metric


def _best_and_second_methods(
    rows: list[dict[str, object]],
    metric_keys: list[str],
    decimals_by_metric: dict[str, int] | None = None,
) -> dict[str, dict[str, set[str]]]:
    ranking: dict[str, dict[str, set[str]]] = {}
    for metric_key in metric_keys:
        available_rows = [row for row in rows if _metric_display_value(row, metric_key, decimals_by_metric) is not None]
        if not available_rows:
            ranking[metric_key] = {"best": set(), "second": set()}
            continue
        values = sorted(
            {_metric_display_value(row, metric_key, decimals_by_metric) for row in available_rows},
            reverse=(metric_key not in LOWER_IS_BETTER),
        )
        best_value = values[0]
        second_value = values[1] if len(values) > 1 else None
        ranking[metric_key] = {
            "best": {
                str(row["display_name"])
                for row in available_rows
                if abs(_metric_display_value(row, metric_key, decimals_by_metric) - best_value) <= 1e-12
            },
            "second": {
                str(row["display_name"])
                for row in available_rows
                if second_value is not None and abs(_metric_display_value(row, metric_key, decimals_by_metric) - second_value) <= 1e-12
            },
        }
    return ranking


def _split_summary(dataset_dir: Path, split_name: str) -> dict[str, object]:
    split = load_split(dataset_dir, split_name)
    return {
        "split": split_name,
        "windows": split.size,
        "positives": int(split.label_main.sum()),
        "positive_rate": 100.0 * float(split.label_main.mean()),
        "incidents": int(len(set(split.incident_id.tolist()))),
        "families": int(len(set(split.family_id.tolist()))),
    }


def _format_rate(value: float) -> str:
    return f"{value:.2f}\\%"


def write_summary_files() -> tuple[Path, Path]:
    payload = {
        "seeds": SEEDS,
        "public_benchmark_stats": [
            _split_summary(PUBLIC_DATASET_DIR, split_name)
            for split_name in ("train", "dev", "test", "test_event_disjoint")
        ],
        "raw_public_benchmark_stats": [
            _split_summary(RAW_PUBLIC_DATASET_DIR, split_name)
            for split_name in ("train", "dev", "test", "test_event_disjoint")
        ],
        "workbook_probe_stats": [
            _split_summary(WORKBOOK_DATASET_DIR, split_name)
            for split_name in ("train", "dev", "test")
        ],
        "chrono_main": collect_rows(CHRONO_MAIN),
        "raw_public_main": collect_rows(RAW_PUBLIC_MAIN),
        "raw_public_event_disjoint": collect_rows(RAW_PUBLIC_MAIN, split="test_event_disjoint"),
        "event_disjoint": collect_rows(EVENT_DISJOINT, split="test_event_disjoint"),
        "event_disjoint_extended": collect_rows(EVENT_DISJOINT_EXTENDED, split="test_event_disjoint"),
        "ablations_all": collect_rows(ABLATIONS_ALL),
        "ablations_retrieval": collect_rows(ABLATIONS_RETRIEVAL),
        "event_disjoint_retrieval": collect_rows(EVENT_DISJOINT_RETRIEVAL, split="test_event_disjoint"),
        "robustness": collect_rows(ROBUSTNESS),
    }
    json_path = FIG_DIR / "atlasv2_public_seed_summary.json"
    csv_path = FIG_DIR / "atlasv2_public_seed_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "run_id", "method", "n", "metric", "mean", "std"])
        for section_name, rows in payload.items():
            if section_name in {"seeds", "public_benchmark_stats", "raw_public_benchmark_stats", "workbook_probe_stats"}:
                continue
            for row in rows:
                for metric_key in AGGREGATE_METRICS:
                    writer.writerow(
                        [
                            section_name,
                            row["base_experiment_name"],
                            row["display_name"],
                            row["n"],
                            metric_key,
                            row["metrics"][metric_key],
                            row["std"][metric_key],
                        ]
                    )
    return json_path, csv_path


def build_main_table_tex() -> str:
    chrono_rows = collect_rows(CHRONO_MAIN)
    event_rows = collect_rows(EVENT_DISJOINT, split="test_event_disjoint")
    metric_keys = ["AUPRC", "LeadTime@P80", "Analog-Fidelity@5", "TTE-Err@1", "Brier"]
    chrono_rank = _best_and_second_methods(chrono_rows, metric_keys, DEFAULT_METRIC_DECIMALS)
    event_rank = _best_and_second_methods(event_rows, metric_keys, DEFAULT_METRIC_DECIMALS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Public ATLASv2 benchmark results on chronological and family-held-out evaluation. Mean and standard deviation are computed over seeds 7, 13, and 21. Best available value in each metric column is bolded and the second-best value is underlined; lower is better for Brier and TTE-Err@1.}",
        r"\label{tab:atlasv2-public-main}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Split & Method & AUPRC & LeadTime@P80 & AF@5 & TTE-Err@1 & Brier \\",
        r"\midrule",
    ]
    for row in chrono_rows:
        lines.append(
            "Chronological"
            + " & "
            + row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in chrono_rank["AUPRC"]["best"], underline=row["display_name"] in chrono_rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in chrono_rank["LeadTime@P80"]["best"], underline=row["display_name"] in chrono_rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in chrono_rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in chrono_rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["TTE-Err@1"], row["std"]["TTE-Err@1"], decimals=2, bold=row["display_name"] in chrono_rank["TTE-Err@1"]["best"], underline=row["display_name"] in chrono_rank["TTE-Err@1"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in chrono_rank["Brier"]["best"], underline=row["display_name"] in chrono_rank["Brier"]["second"])
            + r" \\"
        )
    lines.append(r"\midrule")
    for row in event_rows:
        lines.append(
            "Event-disjoint"
            + " & "
            + row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in event_rank["AUPRC"]["best"], underline=row["display_name"] in event_rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in event_rank["LeadTime@P80"]["best"], underline=row["display_name"] in event_rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in event_rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in event_rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["TTE-Err@1"], row["std"]["TTE-Err@1"], decimals=2, bold=row["display_name"] in event_rank["TTE-Err@1"]["best"], underline=row["display_name"] in event_rank["TTE-Err@1"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in event_rank["Brier"]["best"], underline=row["display_name"] in event_rank["Brier"]["second"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_primary_operating_point_table_tex() -> str:
    chrono_rows = {str(row["display_name"]): row for row in collect_rows(CHRONO_MAIN)}
    event_rows = {str(row["display_name"]): row for row in collect_rows(EVENT_DISJOINT, split="test_event_disjoint")}
    chrono_methods = [
        "TRACER",
        "TCN-Forecaster",
        "Small-Transformer-Forecaster",
        "Prefix-Only-Retrieval + Fusion",
    ]
    event_methods = [
        "TRACER",
        "Small-Transformer-Forecaster",
        "LSTM-Forecaster",
        "DLinear-Forecaster",
        "Prefix-Only-Retrieval + Fusion",
    ]
    metric_keys = ["Precision@P80", "Recall@P80"]
    chrono_rank = _best_and_second_methods([chrono_rows[name] for name in chrono_methods], metric_keys, DEFAULT_METRIC_DECIMALS)
    event_rank = _best_and_second_methods([event_rows[name] for name in event_methods], metric_keys, DEFAULT_METRIC_DECIMALS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\caption{Target-precision operating-point metrics for headline methods on the primary ATLASv2 benchmark. The main tables already report AUPRC, LeadTime@P80, AF@5, TTE-Err@1, and Brier; this supplementary table adds the precision and recall attained at the same $P80$ operating point, making it clear whether a longer lead time is achieved at comparable alert quality or by covering fewer positives. Best available value in each metric column is bolded and the second-best value is underlined.}",
        r"\label{tab:atlasv2-operating-point-metrics}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Split & Method & Precision@P80 & Recall@P80 \\",
        r"\midrule",
    ]
    for method in chrono_methods:
        row = chrono_rows[method]
        lines.append(
            "Chronological"
            + " & "
            + method
            + " & "
            + _format_metric_emph(row["metrics"]["Precision@P80"], row["std"]["Precision@P80"], bold=method in chrono_rank["Precision@P80"]["best"], underline=method in chrono_rank["Precision@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Recall@P80"], row["std"]["Recall@P80"], bold=method in chrono_rank["Recall@P80"]["best"], underline=method in chrono_rank["Recall@P80"]["second"])
            + r" \\"
        )
    lines.append(r"\midrule")
    for method in event_methods:
        row = event_rows[method]
        lines.append(
            "Event-disjoint"
            + " & "
            + method
            + " & "
            + _format_metric_emph(row["metrics"]["Precision@P80"], row["std"]["Precision@P80"], bold=method in event_rank["Precision@P80"]["best"], underline=method in event_rank["Precision@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Recall@P80"], row["std"]["Recall@P80"], bold=method in event_rank["Recall@P80"]["best"], underline=method in event_rank["Recall@P80"]["second"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_primary_discrimination_calibration_table_tex() -> str:
    chrono_rows = {
        "TRACER": aggregate_run("r239_tracer_adaptive_chronology_atlasv2_public", "TRACER", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "TCN-Forecaster": aggregate_run("r005_tcn_forecaster_atlasv2_public", "TCN-Forecaster", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Small-Transformer-Forecaster": aggregate_run("r006_transformer_forecaster_atlasv2_public", "Small-Transformer-Forecaster", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Prefix-Only-Retrieval + Fusion": aggregate_run("r008_prefix_retrieval_atlasv2_public", "Prefix-Only-Retrieval + Fusion", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
    }
    event_rows = {
        "TRACER": aggregate_run("r240_tracer_adaptive_event_atlasv2_public", "TRACER", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "DLinear-Forecaster": aggregate_run("r020_dlinear_forecaster_atlasv2_public_event_disjoint", "DLinear-Forecaster", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "LSTM-Forecaster": aggregate_run("r018_lstm_forecaster_atlasv2_public_event_disjoint", "LSTM-Forecaster", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Small-Transformer-Forecaster": aggregate_run("r010_transformer_forecaster_atlasv2_public_event_disjoint", "Small-Transformer-Forecaster", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Prefix-Only-Retrieval + Fusion": aggregate_run("r010_prefix_retrieval_atlasv2_public_event_disjoint", "Prefix-Only-Retrieval + Fusion", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
    }
    return _build_discrimination_calibration_table_tex(
        split_rows=[
            ("Chronological", chrono_rows),
            ("Event-disjoint", event_rows),
        ],
        caption=(
            "Additional discrimination and calibration diagnostics for headline methods on the primary ATLASv2 benchmark. "
            "These rows come from a full metric-refresh rerun under the current evaluator because the original public "
            "baseline archives predate the expanded metric set. They are supplementary diagnostics and do not replace the "
            "archived headline tables. Best available value in each metric column is bolded and the second-best value is "
            "underlined; lower is better for ECE@10 and LogLoss."
        ),
        label="tab:atlasv2-discrimination-calibration",
    )


def _build_discrimination_calibration_table_tex(
    *,
    split_rows: list[tuple[str, dict[str, dict[str, object]]]],
    caption: str,
    label: str,
) -> str:
    metric_keys = ["AUROC", "BestF1", "ECE@10", "LogLoss"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Split & Method & AUROC & BestF1 & ECE@10 & LogLoss \\",
        r"\midrule",
    ]
    for split_index, (split_name, rows_by_method) in enumerate(split_rows):
        methods = list(rows_by_method.keys())
        rank = _best_and_second_methods([rows_by_method[name] for name in methods], metric_keys, DEFAULT_METRIC_DECIMALS)
        if split_index > 0:
            lines.append(r"\midrule")
        for method in methods:
            row = rows_by_method[method]
            lines.append(
                split_name
                + " & "
                + method
                + " & "
                + _format_metric_emph(row["metrics"]["AUROC"], row["std"]["AUROC"], bold=method in rank["AUROC"]["best"], underline=method in rank["AUROC"]["second"])
                + " & "
                + _format_metric_emph(row["metrics"]["BestF1"], row["std"]["BestF1"], bold=method in rank["BestF1"]["best"], underline=method in rank["BestF1"]["second"])
                + " & "
                + _format_metric_emph(row["metrics"]["ECE@10"], row["std"]["ECE@10"], bold=method in rank["ECE@10"]["best"], underline=method in rank["ECE@10"]["second"])
                + " & "
                + _format_metric_emph(row["metrics"]["LogLoss"], row["std"]["LogLoss"], bold=method in rank["LogLoss"]["best"], underline=method in rank["LogLoss"]["second"])
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_ait_ads_discrimination_calibration_table_tex() -> str:
    chrono_rows = {
        "TRACER": aggregate_run("r241_tracer_adaptive_ait_ads_public", "TRACER", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "DLinear-Forecaster": aggregate_run("r068_dlinear_forecaster_ait_ads_public", "DLinear-Forecaster", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Small-Transformer-Forecaster": aggregate_run("r070_transformer_forecaster_ait_ads_public", "Small-Transformer-Forecaster", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Prefix-Only-Retrieval + Fusion": aggregate_run("r071_prefix_retrieval_ait_ads_public", "Prefix-Only-Retrieval + Fusion", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Staged TRACER": aggregate_run("r081_campaign_mem_staged_ait_ads_public", "Staged TRACER", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Abstention-Aware TRACER": aggregate_run("r098_campaign_mem_abstain_ait_ads_public", "Abstention-Aware TRACER", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
    }
    event_rows = {
        "TRACER": aggregate_run("r242_tracer_adaptive_event_ait_ads_public", "TRACER", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "DLinear-Forecaster": aggregate_run("r068_dlinear_forecaster_ait_ads_public", "DLinear-Forecaster", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Small-Transformer-Forecaster": aggregate_run("r070_transformer_forecaster_ait_ads_public", "Small-Transformer-Forecaster", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Prefix-Only-Retrieval + Fusion": aggregate_run("r071_prefix_retrieval_ait_ads_public", "Prefix-Only-Retrieval + Fusion", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Staged TRACER": aggregate_run("r081_campaign_mem_staged_ait_ads_public", "Staged TRACER", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Abstention-Aware TRACER": aggregate_run("r098_campaign_mem_abstain_ait_ads_public", "Abstention-Aware TRACER", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
    }
    return _build_discrimination_calibration_table_tex(
        split_rows=[
            ("Chronological", chrono_rows),
            ("Event-disjoint", event_rows),
        ],
        caption=(
            "Additional discrimination and calibration diagnostics for the supplementary AIT-ADS benchmark, regenerated "
            "from the full metric-refresh public-suite rerun under the current evaluator. Best available value in each "
            "metric column is bolded and the second-best value is underlined; lower is better for ECE@10 and LogLoss."
        ),
        label="tab:aitads-discrimination-calibration",
    )


def build_atlas_raw_discrimination_calibration_table_tex() -> str:
    chrono_rows = {
        "TRACER": aggregate_run("r243_tracer_adaptive_atlas_raw_public", "TRACER", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Small-Transformer-Forecaster": aggregate_run("r032_transformer_forecaster_atlas_raw_public", "Small-Transformer-Forecaster", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Prefix-Only-Retrieval + Fusion": aggregate_run("r035_prefix_retrieval_atlas_raw_public", "Prefix-Only-Retrieval + Fusion", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "DLinear-Calibrated TRACER": aggregate_run("r047_campaign_mem_v3_dlinear_atlas_raw_public", "DLinear-Calibrated TRACER", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Selector-Conservative TRACER": aggregate_run("r089_campaign_mem_selector_conservative_atlas_raw_public", "Selector-Conservative TRACER", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "ATLASv2-Tuned TRACER Core Transfer": aggregate_run("r121_campaign_mem_dual_selector_proxy_strict_atlas_raw_public", "ATLASv2-Tuned TRACER Core Transfer", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
    }
    event_rows = {
        "TRACER": aggregate_run("r243_tracer_adaptive_atlas_raw_public", "TRACER", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Small-Transformer-Forecaster": aggregate_run("r032_transformer_forecaster_atlas_raw_public", "Small-Transformer-Forecaster", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Prefix-Only-Retrieval + Fusion": aggregate_run("r035_prefix_retrieval_atlas_raw_public", "Prefix-Only-Retrieval + Fusion", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "DLinear-Calibrated TRACER": aggregate_run("r047_campaign_mem_v3_dlinear_atlas_raw_public", "DLinear-Calibrated TRACER", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "Selector-Conservative TRACER": aggregate_run("r089_campaign_mem_selector_conservative_atlas_raw_public", "Selector-Conservative TRACER", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
        "ATLASv2-Tuned TRACER Core Transfer": aggregate_run("r121_campaign_mem_dual_selector_proxy_strict_atlas_raw_public", "ATLASv2-Tuned TRACER Core Transfer", split="test_event_disjoint", result_dir=REFRESH_PUBLIC_SUITE_RESULT_DIR),
    }
    return _build_discrimination_calibration_table_tex(
        split_rows=[
            ("Chronological", chrono_rows),
            ("Event-disjoint", event_rows),
        ],
        caption=(
            "Additional discrimination and calibration diagnostics for the supplementary ATLAS-Raw benchmark, regenerated "
            "from the full metric-refresh public-suite rerun under the current evaluator. Best available value in each "
            "metric column is bolded and the second-best value is underlined; lower is better for ECE@10 and LogLoss."
        ),
        label="tab:atlasraw-discrimination-calibration",
    )


def build_uncertainty_table_tex() -> str:
    chrono_rows = {str(row["display_name"]): row for row in collect_rows(CHRONO_MAIN)}
    event_rows = {str(row["display_name"]): row for row in collect_rows(EVENT_DISJOINT, split="test_event_disjoint")}
    chrono_best = _best_methods(list(chrono_rows.values()), ["AUPRC"], DEFAULT_METRIC_DECIMALS)
    event_best = _best_methods(list(event_rows.values()), ["AUPRC"], DEFAULT_METRIC_DECIMALS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Seed-level AUPRC values for headline methods on the primary ATLASv2 benchmark. The table complements \Cref{fig:appendix-seed-stability} by exposing the exact three-seed values behind the main ATLASv2 rows. Mean and standard deviation are reported over seeds 7, 13, and 21.}",
        r"\label{tab:atlasv2-public-uncertainty}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Split & Method & Seed 7 & Seed 13 & Seed 21 & Mean $\pm$ Std \\",
        r"\midrule",
    ]
    chrono_methods = [
        "TCN-Forecaster",
        "LSTM-Forecaster",
        "Small-Transformer-Forecaster",
        "Prefix-Only-Retrieval + Fusion",
        "TRACER",
    ]
    for method in chrono_methods:
        row = chrono_rows[method]
        values = row["values"]["AUPRC"]
        lines.append(
            "Chronological"
            + " & "
            + method
            + " & "
            + _format_scalar(values[0])
            + " & "
            + _format_scalar(values[1])
            + " & "
            + _format_scalar(values[2])
            + " & "
            + _format_metric(
                row["metrics"]["AUPRC"],
                row["std"]["AUPRC"],
                bold=(method in chrono_best["AUPRC"]),
            )
            + r" \\"
        )
    lines.append(r"\midrule")
    event_methods = [
        "Small-Transformer-Forecaster",
        "Prefix-Only-Retrieval + Fusion",
        "TRACER",
    ]
    for method in event_methods:
        row = event_rows[method]
        values = row["values"]["AUPRC"]
        lines.append(
            "Event-disjoint"
            + " & "
            + method
            + " & "
            + _format_scalar(values[0])
            + " & "
            + _format_scalar(values[1])
            + " & "
            + _format_scalar(values[2])
            + " & "
            + _format_metric(
                row["metrics"]["AUPRC"],
                row["std"]["AUPRC"],
                bold=(method in event_best["AUPRC"]),
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def _pairwise_outcome(a: float, b: float, decimals: int = 3) -> tuple[str, str]:
    eps = 1e-12
    if a > b + eps:
        label = "W"
    elif b > a + eps:
        label = "L"
    else:
        label = "T"
    return label, f"{label} ({a:.{decimals}f}/{b:.{decimals}f})"


def build_pairwise_consistency_table_tex() -> str:
    chrono_rows = {str(row["display_name"]): row for row in collect_rows(CHRONO_MAIN)}
    event_rows = {str(row["display_name"]): row for row in collect_rows(EVENT_DISJOINT, split="test_event_disjoint")}
    comparisons = [
        ("Chronological", chrono_rows["TRACER"], chrono_rows["Prefix-Only-Retrieval + Fusion"], "TRACER vs Prefix-Only"),
        ("Chronological", chrono_rows["TRACER"], chrono_rows["Small-Transformer-Forecaster"], "TRACER vs Small-Transformer"),
        ("Event-disjoint", event_rows["TRACER"], event_rows["Prefix-Only-Retrieval + Fusion"], "TRACER vs Prefix-Only"),
        ("Event-disjoint", event_rows["TRACER"], event_rows["Small-Transformer-Forecaster"], "TRACER vs Small-Transformer"),
    ]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Seed-wise head-to-head AUPRC outcomes for TRACER against key baselines on the primary ATLASv2 benchmark. Each seed entry is reported from TRACER's perspective as win (W), tie (T), or loss (L), followed by \texttt{TRACER / baseline} AUPRC.}",
        r"\label{tab:atlasv2-public-pairwise}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Split & Comparison & Seed 7 & Seed 13 & Seed 21 & W / T / L \\",
        r"\midrule",
    ]
    for split_name, campaign_row, baseline_row, label in comparisons:
        cvals = campaign_row["values"]["AUPRC"]
        bvals = baseline_row["values"]["AUPRC"]
        outcomes = [_pairwise_outcome(cvals[idx], bvals[idx]) for idx in range(3)]
        wins = sum(1 for outcome, _ in outcomes if outcome == "W")
        ties = sum(1 for outcome, _ in outcomes if outcome == "T")
        losses = sum(1 for outcome, _ in outcomes if outcome == "L")
        lines.append(
            split_name
            + " & "
            + label
            + " & "
            + outcomes[0][1]
            + " & "
            + outcomes[1][1]
            + " & "
            + outcomes[2][1]
            + " & "
            + f"{wins} / {ties} / {losses}"
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_ablation_table_tex() -> str:
    rows = collect_rows(ABLATIONS_ALL)
    metric_keys = ["AUPRC", "LeadTime@P80", "Analog-Fidelity@5", "TTE-Err@1", "Brier"]
    rank = _best_and_second_methods(rows, metric_keys, DEFAULT_METRIC_DECIMALS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Public ATLASv2 chronological operating-point comparison for TRACER and related support variants. Mean and standard deviation are computed over seeds 7, 13, and 21 when available. Best available value in each metric column is bolded and the second-best value is underlined; lower is better for Brier and TTE-Err@1.}",
        r"\label{tab:atlasv2-public-ablations}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & AUPRC & LeadTime@P80 & AF@5 & TTE-Err@1 & Brier \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in rank["AUPRC"]["best"], underline=row["display_name"] in rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in rank["LeadTime@P80"]["best"], underline=row["display_name"] in rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["TTE-Err@1"], row["std"]["TTE-Err@1"], decimals=2, bold=row["display_name"] in rank["TTE-Err@1"]["best"], underline=row["display_name"] in rank["TTE-Err@1"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in rank["Brier"]["best"], underline=row["display_name"] in rank["Brier"]["second"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_event_extended_table_tex() -> str:
    rows = collect_rows(EVENT_DISJOINT_EXTENDED, split="test_event_disjoint")
    metric_keys = ["AUPRC", "LeadTime@P80", "Analog-Fidelity@5", "TTE-Err@1", "Brier"]
    rank = _best_and_second_methods(rows, metric_keys, DEFAULT_METRIC_DECIMALS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Extended family-held-out event-disjoint comparison with additional parametric and retrieval controls. Mean and standard deviation are computed over seeds 7, 13, and 21. Best available value in each metric column is bolded and the second-best value is underlined; lower is better for Brier and TTE-Err@1.}",
        r"\label{tab:atlasv2-public-event-extended}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & AUPRC & LeadTime@P80 & AF@5 & TTE-Err@1 & Brier \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in rank["AUPRC"]["best"], underline=row["display_name"] in rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in rank["LeadTime@P80"]["best"], underline=row["display_name"] in rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["TTE-Err@1"], row["std"]["TTE-Err@1"], decimals=2, bold=row["display_name"] in rank["TTE-Err@1"]["best"], underline=row["display_name"] in rank["TTE-Err@1"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in rank["Brier"]["best"], underline=row["display_name"] in rank["Brier"]["second"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_event_retrieval_table_tex() -> str:
    rows = collect_rows(EVENT_DISJOINT_RETRIEVAL, split="test_event_disjoint")
    metric_keys = ["AUPRC", "LeadTime@P80", "Analog-Fidelity@5", "TTE-Err@1", "Brier"]
    rank = _best_and_second_methods(rows, metric_keys, DEFAULT_METRIC_DECIMALS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Family-held-out event-disjoint operating-point comparison for TRACER and related support variants. Mean and standard deviation are computed over seeds 7, 13, and 21. Best available value in each metric column is bolded and the second-best value is underlined; lower is better for Brier and TTE-Err@1.}",
        r"\label{tab:atlasv2-public-event-retrieval}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & AUPRC & LeadTime@P80 & AF@5 & TTE-Err@1 & Brier \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in rank["AUPRC"]["best"], underline=row["display_name"] in rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in rank["LeadTime@P80"]["best"], underline=row["display_name"] in rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["TTE-Err@1"], row["std"]["TTE-Err@1"], decimals=2, bold=row["display_name"] in rank["TTE-Err@1"]["best"], underline=row["display_name"] in rank["TTE-Err@1"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in rank["Brier"]["best"], underline=row["display_name"] in rank["Brier"]["second"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_public_benchmark_stats_table_tex() -> str:
    metadata = load_metadata(PUBLIC_DATASET_DIR)
    rows = [
        _split_summary(PUBLIC_DATASET_DIR, split_name)
        for split_name in ("train", "dev", "test", "test_event_disjoint")
    ]
    held_out = ", ".join(metadata.get("event_disjoint_attack_families", [])) or "--"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\caption{Primary public ATLASv2 benchmark statistics. The held-out family set for the event-disjoint split is \texttt{"
        + held_out.replace("_", r"\_")
        + r"}.}",
        r"\label{tab:atlasv2-public-stats}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Split & Windows & Positives & Pos.\ rate & Incidents & Families \\",
        r"\midrule",
    ]
    split_labels = {
        "train": "Train",
        "dev": "Dev",
        "test": "Test",
        "test_event_disjoint": "Event-disjoint",
    }
    for row in rows:
        lines.append(
            split_labels[row["split"]]
            + " & "
            + str(row["windows"])
            + " & "
            + str(row["positives"])
            + " & "
            + _format_rate(float(row["positive_rate"]))
            + " & "
            + str(row["incidents"])
            + " & "
            + str(row["families"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_workbook_probe_stats_table_tex() -> str:
    rows = [_split_summary(WORKBOOK_DATASET_DIR, split_name) for split_name in ("train", "dev", "test")]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\caption{Secondary workbook-derived public probe statistics. This split is chronological-only and is used as supplementary evidence.}",
        r"\label{tab:atlasv2-workbook-stats}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Split & Windows & Positives & Pos.\ rate & Incidents & Families \\",
        r"\midrule",
    ]
    split_labels = {"train": "Train", "dev": "Dev", "test": "Test"}
    for row in rows:
        lines.append(
            split_labels[row["split"]]
            + " & "
            + str(row["windows"])
            + " & "
            + str(row["positives"])
            + " & "
            + _format_rate(float(row["positive_rate"]))
            + " & "
            + str(row["incidents"])
            + " & "
            + str(row["families"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_ait_ads_benchmark_stats_table_tex() -> str:
    metadata = load_metadata(AIT_ADS_DATASET_DIR)
    rows = [
        _split_summary(AIT_ADS_DATASET_DIR, split_name)
        for split_name in ("train", "dev", "test", "test_event_disjoint")
    ]
    held_out = ", ".join(metadata.get("event_disjoint_attack_families", [])) or "--"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\caption{AIT-ADS public benchmark statistics. The held-out scenario set for the event-disjoint split is \texttt{"
        + held_out.replace("_", r"\_")
        + r"}.}",
        r"\label{tab:aitads-public-stats}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Split & Windows & Positives & Pos.\ rate & Incidents & Families \\",
        r"\midrule",
    ]
    split_labels = {
        "train": "Train",
        "dev": "Dev",
        "test": "Test",
        "test_event_disjoint": "Event-disjoint",
    }
    for row in rows:
        lines.append(
            split_labels[row["split"]]
            + " & "
            + str(row["windows"])
            + " & "
            + str(row["positives"])
            + " & "
            + _format_rate(float(row["positive_rate"]))
            + " & "
            + str(row["incidents"])
            + " & "
            + str(row["families"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_ait_ads_results_table_tex() -> str:
    chrono_rows = collect_rows(AIT_ADS_MAIN)
    event_rows = collect_rows(AIT_ADS_EVENT_MAIN, split="test_event_disjoint")
    metric_keys = ["AUPRC", "LeadTime@P80", "Analog-Fidelity@5", "Brier"]
    chrono_rank = _best_and_second_methods(chrono_rows, metric_keys, DEFAULT_METRIC_DECIMALS)
    event_rank = _best_and_second_methods(event_rows, metric_keys, DEFAULT_METRIC_DECIMALS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{AIT-ADS public benchmark results over three public seeds. Best available value in each metric column is bolded and the second-best value is underlined; lower is better for Brier. The later TRACER-family rows report promoted benchmark-specific variants retained from the training-strategy sweep; they are informative as supplementary evidence, but they do not replace the primary ATLASv2 evidence base.}",
        r"\label{tab:aitads-public-main}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Split & Method & AUPRC & LeadTime@P80 & AF@5 & Brier \\",
        r"\midrule",
    ]
    for row in chrono_rows:
        lines.append(
            "Chronological"
            + " & "
            + row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in chrono_rank["AUPRC"]["best"], underline=row["display_name"] in chrono_rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in chrono_rank["LeadTime@P80"]["best"], underline=row["display_name"] in chrono_rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in chrono_rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in chrono_rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in chrono_rank["Brier"]["best"], underline=row["display_name"] in chrono_rank["Brier"]["second"])
            + r" \\"
        )
    lines.append(r"\midrule")
    for row in event_rows:
        lines.append(
            "Event-disjoint"
            + " & "
            + row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in event_rank["AUPRC"]["best"], underline=row["display_name"] in event_rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in event_rank["LeadTime@P80"]["best"], underline=row["display_name"] in event_rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in event_rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in event_rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in event_rank["Brier"]["best"], underline=row["display_name"] in event_rank["Brier"]["second"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_raw_public_benchmark_stats_table_tex() -> str:
    metadata = load_metadata(RAW_PUBLIC_DATASET_DIR)
    rows = [
        _split_summary(RAW_PUBLIC_DATASET_DIR, split_name)
        for split_name in ("train", "dev", "test", "test_event_disjoint")
    ]
    held_out = ", ".join(metadata.get("event_disjoint_attack_families", [])) or "--"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\caption{ATLAS-Raw public benchmark statistics. The held-out family set for the event-disjoint split is \texttt{"
        + held_out.replace("_", r"\_")
        + r"}.}",
        r"\label{tab:atlasraw-public-stats}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Split & Windows & Positives & Pos.\ rate & Incidents & Families \\",
        r"\midrule",
    ]
    split_labels = {
        "train": "Train",
        "dev": "Dev",
        "test": "Test",
        "test_event_disjoint": "Event-disjoint",
    }
    for row in rows:
        lines.append(
            split_labels[row["split"]]
            + " & "
            + str(row["windows"])
            + " & "
            + str(row["positives"])
            + " & "
            + _format_rate(float(row["positive_rate"]))
            + " & "
            + str(row["incidents"])
            + " & "
            + str(row["families"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def build_raw_public_results_table_tex() -> str:
    chrono_rows = collect_rows(RAW_PUBLIC_MAIN)
    event_rows = collect_rows(RAW_PUBLIC_MAIN, split="test_event_disjoint")
    metric_keys = ["AUPRC", "LeadTime@P80", "Analog-Fidelity@5", "TTE-Err@1", "Brier"]
    chrono_rank = _best_and_second_methods(chrono_rows, metric_keys, DEFAULT_METRIC_DECIMALS)
    event_rank = _best_and_second_methods(event_rows, metric_keys, DEFAULT_METRIC_DECIMALS)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{ATLAS-Raw public benchmark results. Mean and standard deviation are computed over seeds 7, 13, and 21. The TRACER row corresponds to the frozen \texttt{tracer\_adaptive} policy on this benchmark, while the ATLASv2-tuned core transfer is shown separately. Best available value in each metric column is bolded and the second-best value is underlined; lower is better for Brier and TTE-Err@1.}",
        r"\label{tab:atlasraw-public-main}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Split & Method & AUPRC & LeadTime@P80 & AF@5 & TTE-Err@1 & Brier \\",
        r"\midrule",
    ]
    for row in chrono_rows:
        lines.append(
            "Chronological"
            + " & "
            + row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in chrono_rank["AUPRC"]["best"], underline=row["display_name"] in chrono_rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in chrono_rank["LeadTime@P80"]["best"], underline=row["display_name"] in chrono_rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in chrono_rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in chrono_rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["TTE-Err@1"], row["std"]["TTE-Err@1"], decimals=2, bold=row["display_name"] in chrono_rank["TTE-Err@1"]["best"], underline=row["display_name"] in chrono_rank["TTE-Err@1"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in chrono_rank["Brier"]["best"], underline=row["display_name"] in chrono_rank["Brier"]["second"])
            + r" \\"
        )
    lines.append(r"\midrule")
    for row in event_rows:
        lines.append(
            "Event-disjoint"
            + " & "
            + row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in event_rank["AUPRC"]["best"], underline=row["display_name"] in event_rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in event_rank["LeadTime@P80"]["best"], underline=row["display_name"] in event_rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in event_rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in event_rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["TTE-Err@1"], row["std"]["TTE-Err@1"], decimals=2, bold=row["display_name"] in event_rank["TTE-Err@1"]["best"], underline=row["display_name"] in event_rank["TTE-Err@1"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in event_rank["Brier"]["best"], underline=row["display_name"] in event_rank["Brier"]["second"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def build_robustness_table_tex() -> str:
    rows = collect_rows(ROBUSTNESS)
    metric_keys = ["AUPRC", "LeadTime@P80", "Analog-Fidelity@5", "TTE-Err@1", "Brier"]
    robustness_decimals = dict(DEFAULT_METRIC_DECIMALS)
    robustness_decimals["TTE-Err@1"] = 1
    rank = _best_and_second_methods(rows, metric_keys, robustness_decimals)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Secondary workbook-derived public probe with expanded parametric and retrieval baselines. This table is supplementary because the workbook split is not family-held-out. Best available value in each metric column is bolded and the second-best value is underlined; lower is better for Brier and TTE-Err@1.}",
        r"\label{tab:atlasv2-public-robustness}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & AUPRC & LeadTime@P80 & AF@5 & TTE-Err@1 & Brier \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            row["display_name"]
            + " & "
            + _format_metric_emph(row["metrics"]["AUPRC"], row["std"]["AUPRC"], bold=row["display_name"] in rank["AUPRC"]["best"], underline=row["display_name"] in rank["AUPRC"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["LeadTime@P80"], row["std"]["LeadTime@P80"], decimals=2, bold=row["display_name"] in rank["LeadTime@P80"]["best"], underline=row["display_name"] in rank["LeadTime@P80"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Analog-Fidelity@5"], row["std"]["Analog-Fidelity@5"], decimals=2, bold=row["display_name"] in rank["Analog-Fidelity@5"]["best"], underline=row["display_name"] in rank["Analog-Fidelity@5"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["TTE-Err@1"], row["std"]["TTE-Err@1"], decimals=1, bold=row["display_name"] in rank["TTE-Err@1"]["best"], underline=row["display_name"] in rank["TTE-Err@1"]["second"])
            + " & "
            + _format_metric_emph(row["metrics"]["Brier"], row["std"]["Brier"], bold=row["display_name"] in rank["Brier"]["best"], underline=row["display_name"] in rank["Brier"]["second"])
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def write_table_files() -> None:
    (FIG_DIR / "tab_public_benchmark_stats.tex").write_text(build_public_benchmark_stats_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_ait_ads_benchmark_stats.tex").write_text(build_ait_ads_benchmark_stats_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_ait_ads_results.tex").write_text(build_ait_ads_results_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_atlas_raw_benchmark_stats.tex").write_text(build_raw_public_benchmark_stats_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_atlas_raw_results.tex").write_text(build_raw_public_results_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_workbook_probe_stats.tex").write_text(build_workbook_probe_stats_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_public_main_results.tex").write_text(build_main_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_public_operating_point_metrics.tex").write_text(build_primary_operating_point_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_public_discrimination_calibration.tex").write_text(build_primary_discrimination_calibration_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_ait_ads_discrimination_calibration.tex").write_text(build_ait_ads_discrimination_calibration_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_atlas_raw_discrimination_calibration.tex").write_text(build_atlas_raw_discrimination_calibration_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_public_uncertainty.tex").write_text(build_uncertainty_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_public_pairwise_consistency.tex").write_text(build_pairwise_consistency_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_public_ablations.tex").write_text(build_ablation_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_public_event_extended.tex").write_text(build_event_extended_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_public_event_retrieval.tex").write_text(build_event_retrieval_table_tex(), encoding="utf-8")
    (FIG_DIR / "tab_public_robustness.tex").write_text(build_robustness_table_tex(), encoding="utf-8")


def main() -> None:
    write_summary_files()
    write_table_files()


if __name__ == "__main__":
    main()
