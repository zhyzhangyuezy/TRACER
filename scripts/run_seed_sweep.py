from __future__ import annotations

import argparse
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.training import run_experiment
from campaign_mem.utils import load_yaml, save_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_manifest(
    output_path: str | Path,
    *,
    configs: list[str],
    seeds: list[int],
    runs: list[dict[str, object]],
    started_at: str,
) -> None:
    save_json(
        output_path,
        {
            "configs": configs,
            "seeds": seeds,
            "started_at": started_at,
            "updated_at": _utc_now(),
            "runs": runs,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-seed sweep over experiment configs.")
    parser.add_argument("--configs", nargs="+", required=True, help="Experiment YAML configs.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21], help="Seeds to run.")
    parser.add_argument(
        "--manifest-output",
        default="outputs/results/public_seed_sweep_manifest.json",
        help="Path to the manifest JSON summarizing the sweep.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a seed run when the target output JSON already exists.",
    )
    parser.add_argument(
        "--output-dir-override",
        default=None,
        help="Optional output directory override for all generated result JSON files.",
    )
    args = parser.parse_args()

    normalized_configs = [str(Path(path)).replace("\\", "/") for path in args.configs]
    started_at = _utc_now()
    manifest_runs: list[dict[str, object]] = []
    _write_manifest(
        args.manifest_output,
        configs=normalized_configs,
        seeds=args.seeds,
        runs=manifest_runs,
        started_at=started_at,
    )
    for config_path_str in args.configs:
        config_path = Path(config_path_str)
        config = load_yaml(config_path)
        base_name = str(config["experiment_name"])
        if args.output_dir_override is not None:
            run_output_dir = str(Path(args.output_dir_override)).replace("\\", "/")
            config.setdefault("output", {})
            config["output"]["dir"] = run_output_dir
        output_dir = Path(config.get("output", {}).get("dir", "outputs/results"))
        for seed in args.seeds:
            run_config = deepcopy(config)
            run_name = f"{base_name}_seed{seed}"
            run_config["experiment_name"] = run_name
            run_config["seed"] = seed
            output_json = output_dir / f"{run_name}.json"
            if args.skip_existing and output_json.exists():
                manifest_runs.append(
                    {
                        "config": str(config_path).replace("\\", "/"),
                        "base_experiment_name": base_name,
                        "seed": seed,
                        "output_json": str(output_json).replace("\\", "/"),
                        "status": "skipped_existing",
                    }
                )
                _write_manifest(
                    args.manifest_output,
                    configs=normalized_configs,
                    seeds=args.seeds,
                    runs=manifest_runs,
                    started_at=started_at,
                )
                print(f"[skip] {run_name}")
                continue

            started = time.time()
            result = run_experiment(run_config)
            manifest_runs.append(
                {
                    "config": str(config_path).replace("\\", "/"),
                    "base_experiment_name": base_name,
                    "seed": seed,
                    "output_json": str(output_json).replace("\\", "/"),
                    "status": "completed",
                    "test_auprc": result.get("test", {}).get("AUPRC"),
                    "test_event_disjoint_auprc": result.get("test_event_disjoint", {}).get("AUPRC"),
                    "duration_sec": round(time.time() - started, 3),
                }
            )
            _write_manifest(
                args.manifest_output,
                configs=normalized_configs,
                seeds=args.seeds,
                runs=manifest_runs,
                started_at=started_at,
            )
            print(f"[done] {run_name}")

    print(f"Manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
