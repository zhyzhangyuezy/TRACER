from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data import prepare_atlasv2_workbook
from campaign_mem.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ATLASv2 workbook data into Campaign-MEM contract.")
    parser.add_argument("--config", required=True, help="YAML config for workbook preparation.")
    args = parser.parse_args()
    metadata = prepare_atlasv2_workbook(load_yaml(args.config))
    print(f"Prepared {metadata['dataset_name']} from {metadata['source_workbook']}")


if __name__ == "__main__":
    main()
