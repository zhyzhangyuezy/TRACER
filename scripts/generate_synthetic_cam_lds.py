from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data import generate_synthetic_dataset
from campaign_mem.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic Campaign-MEM smoke-test dataset.")
    parser.add_argument("--config", required=True, help="YAML config for synthetic dataset generation.")
    args = parser.parse_args()
    metadata = generate_synthetic_dataset(load_yaml(args.config))
    print(f"Generated dataset at {metadata['dataset_name']} with seed={metadata['seed']}")


if __name__ == "__main__":
    main()
