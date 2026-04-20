from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.training import run_experiment
from campaign_mem.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Campaign-MEM experiment from YAML config.")
    parser.add_argument("--config", required=True, help="Experiment YAML config.")
    args = parser.parse_args()
    result = run_experiment(load_yaml(args.config))
    print(f"Experiment: {result['experiment_name']}")
    print(f"Test AUPRC: {result['test']['AUPRC']:.4f}")
    if "Analog-Fidelity@5" in result["test"]:
        print(f"Test AF@5: {result['test']['Analog-Fidelity@5']:.2f}")


if __name__ == "__main__":
    main()
