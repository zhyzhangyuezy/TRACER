from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "outputs" / "results"
FIG_DIR = ROOT / "figures"

SETTINGS = [
    {
        "setting": "ATLASv2 held-out-family",
        "dataset_dir": "data/atlasv2_public",
        "split": "test_event_disjoint",
        "bank_size": 438,
        "embedding_dim": 128,
        "lsh_bits": 6,
    },
    {
        "setting": "AIT-ADS chronology",
        "dataset_dir": "data/ait_ads_public",
        "split": "test",
        "bank_size": 9119,
        "embedding_dim": 128,
        "lsh_bits": 8,
    },
    {
        "setting": "ATLAS-Raw chronology",
        "dataset_dir": "data/atlas_raw_public",
        "split": "test",
        "bank_size": 46156,
        "embedding_dim": 128,
        "lsh_bits": 10,
    },
]


def _split_size(dataset_dir: str, split: str) -> int:
    path = ROOT / dataset_dir / f"{split}.npz"
    if not path.exists():
        return 512
    with np.load(path, allow_pickle=True) as data:
        for key in ("prefix", "x", "features"):
            if key in data:
                return int(data[key].shape[0])
        first = next(iter(data.files))
        return int(data[first].shape[0])


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-6)


def _exact_topk(query: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    similarity = query @ bank.T
    top = np.argpartition(-similarity, kth=min(k - 1, bank.shape[0] - 1), axis=1)[:, :k]
    row = np.arange(query.shape[0])[:, None]
    order = np.argsort(-similarity[row, top], axis=1)
    return top[row, order]


def _hash_codes(x: np.ndarray, planes: np.ndarray) -> np.ndarray:
    bits = (x @ planes.T) >= 0.0
    weights = (1 << np.arange(bits.shape[1], dtype=np.int64))
    return (bits.astype(np.int64) * weights).sum(axis=1)


def _build_lsh_index(bank: np.ndarray, bits: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    planes = rng.normal(size=(bits, bank.shape[1])).astype(np.float32)
    codes = _hash_codes(bank, planes)
    buckets: dict[int, list[int]] = {}
    for idx, code in enumerate(codes.tolist()):
        buckets.setdefault(int(code), []).append(idx)
    return planes, {code: np.asarray(indices, dtype=np.int64) for code, indices in buckets.items()}


def _lsh_topk(query: np.ndarray, bank: np.ndarray, planes: np.ndarray, buckets: dict[int, np.ndarray], k: int) -> tuple[np.ndarray, float, float]:
    codes = _hash_codes(query, planes)
    rows = []
    candidate_sizes = []
    fallback = 0
    for q, code in zip(query, codes.tolist()):
        candidates = buckets.get(int(code))
        if candidates is None or candidates.shape[0] < k:
            # Exact fallback for tiny buckets keeps the approximate index safe on sparse buckets.
            candidates = np.arange(bank.shape[0], dtype=np.int64)
            fallback += 1
        candidate_sizes.append(float(candidates.shape[0]))
        similarity = q @ bank[candidates].T
        local_top = np.argpartition(-similarity, kth=min(k - 1, candidates.shape[0] - 1))[:k]
        rows.append(candidates[local_top[np.argsort(-similarity[local_top])]])
    return np.vstack(rows), float(mean(candidate_sizes)), float(fallback) / max(query.shape[0], 1)


def _recall_at_k(exact: np.ndarray, approx: np.ndarray) -> float:
    recalls = []
    for exact_row, approx_row in zip(exact, approx):
        recalls.append(len(set(exact_row.tolist()) & set(approx_row.tolist())) / max(exact_row.shape[0], 1))
    return 100.0 * float(mean(recalls))


def _bench_setting(spec: dict[str, Any], repeats: int = 30, k: int = 5) -> dict[str, Any]:
    rng = np.random.default_rng(20260417 + int(spec["bank_size"]))
    query_count = _split_size(str(spec["dataset_dir"]), str(spec["split"]))
    bank = _normalize(rng.normal(size=(int(spec["bank_size"]), int(spec["embedding_dim"]))).astype(np.float32))
    query = _normalize(rng.normal(size=(query_count, int(spec["embedding_dim"]))).astype(np.float32))

    exact = _exact_topk(query, bank, k)
    planes, buckets = _build_lsh_index(bank, int(spec["lsh_bits"]), rng)
    approx, candidate_mean, fallback_rate = _lsh_topk(query, bank, planes, buckets, k)
    recall = _recall_at_k(exact, approx)

    exact_times = []
    lsh_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _exact_topk(query, bank, k)
        exact_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        _lsh_topk(query, bank, planes, buckets, k)
        lsh_times.append(time.perf_counter() - start)

    return {
        "setting": spec["setting"],
        "query_count": query_count,
        "bank_size": int(spec["bank_size"]),
        "embedding_dim": int(spec["embedding_dim"]),
        "lsh_bits": int(spec["lsh_bits"]),
        "exact_ms_per_query": 1000.0 * float(mean(exact_times)) / query_count,
        "lsh_ms_per_query": 1000.0 * float(mean(lsh_times)) / query_count,
        "lsh_recall_at_5": recall,
        "lsh_mean_candidates": candidate_mean,
        "lsh_fallback_rate": 100.0 * fallback_rate,
        "repeats": repeats,
    }


def build_audit() -> dict[str, Any]:
    rows = [_bench_setting(spec) for spec in SETTINGS]
    return {
        "audit": "Measured exact-top-k versus LSH lookup latency microbenchmark",
        "rows": rows,
        "note": "The microbenchmark isolates retrieval lookup cost by using random normalized 128-dimensional float32 embeddings with the same bank sizes as the released retrieval-active routes. It measures CPU wall-clock lookup time on this workstation and reports an LSH-style approximate candidate lookup as a lightweight ANN stress test, not as a production index recommendation.",
    }


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Measured latency audit",
        "",
        audit["note"],
        "",
        "| Setting | Q | Bank | Exact ms/query | LSH ms/query | LSH Recall@5 | Candidates | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["rows"]:
        lines.append(
            f"| {row['setting']} | {row['query_count']} | {row['bank_size']} | {row['exact_ms_per_query']:.4f} | {row['lsh_ms_per_query']:.4f} | {row['lsh_recall_at_5']:.1f}% | {row['lsh_mean_candidates']:.1f} | {row['lsh_fallback_rate']:.1f}% |"
        )
    (RESULT_DIR / "measured_latency_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(audit: dict[str, Any]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Measured retrieval lookup latency. The benchmark isolates exact top-$k$ cosine lookup over random normalized 128-dimensional float32 embeddings with the same bank sizes as the released retrieval-active routes, and compares it with a lightweight LSH candidate lookup. Times are CPU wall-clock milliseconds per query on the local workstation.}",
        r"\label{tab:measured-latency-audit}",
        r"\maxtablewidth{",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Setting & Queries & Bank & Exact ms/q & LSH ms/q & LSH R@5 & Fallback \\",
        r"\midrule",
    ]
    for row in audit["rows"]:
        lines.append(
            "{setting} & {query_count} & {bank_size} & ${exact:.4f}$ & ${lsh:.4f}$ & ${recall:.1f}\\%$ & ${fallback:.1f}\\%$ \\\\".format(
                setting=str(row["setting"]).replace("_", r"\_"),
                query_count=int(row["query_count"]),
                bank_size=int(row["bank_size"]),
                exact=float(row["exact_ms_per_query"]),
                lsh=float(row["lsh_ms_per_query"]),
                recall=float(row["lsh_recall_at_5"]),
                fallback=float(row["lsh_fallback_rate"]),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}"]
    (FIG_DIR / "tab_measured_latency_audit.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit = build_audit()
    (RESULT_DIR / "measured_latency_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_markdown(audit)
    write_latex(audit)
    print("Wrote outputs/results/measured_latency_audit.json")
    print("Wrote figures/tab_measured_latency_audit.tex")


if __name__ == "__main__":
    main()
