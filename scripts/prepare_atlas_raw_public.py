from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data import prepare_atlas_raw_public
from campaign_mem.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ATLAS raw security events into Campaign-MEM public benchmark format.")
    parser.add_argument("--config", required=True, help="YAML config for ATLAS raw public benchmark preparation.")
    args = parser.parse_args()
    metadata = prepare_atlas_raw_public(load_yaml(args.config))
    print(f"Prepared {metadata['dataset_name']} from {metadata['raw_logs_dir']}")


if __name__ == "__main__":
    main()
