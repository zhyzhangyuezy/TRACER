from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data import prepare_ait_ads_public
from campaign_mem.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AIT-ADS public benchmark into Campaign-MEM contract.")
    parser.add_argument("--config", required=True, help="YAML config describing AIT-ADS raw input.")
    args = parser.parse_args()
    metadata = prepare_ait_ads_public(load_yaml(args.config))
    print(f"Prepared AIT-ADS dataset: {metadata['dataset_name']}")


if __name__ == "__main__":
    main()
