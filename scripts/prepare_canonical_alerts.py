from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data import prepare_canonical_alert_dataset
from campaign_mem.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical alert events into Campaign-MEM contract.")
    parser.add_argument("--config", required=True, help="YAML config describing canonical alert input.")
    args = parser.parse_args()
    metadata = prepare_canonical_alert_dataset(load_yaml(args.config))
    print(f"Prepared canonical alert dataset: {metadata['dataset_name']}")


if __name__ == "__main__":
    main()
