from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data import normalize_suricata_eve
from campaign_mem.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Suricata EVE alerts to canonical alert CSV.")
    parser.add_argument("--config", required=True, help="YAML config for Suricata normalization.")
    args = parser.parse_args()
    result = normalize_suricata_eve(load_yaml(args.config))
    print(f"Normalized {result['rows']} rows from {result['files']} files -> {result['output_path']}")


if __name__ == "__main__":
    main()
