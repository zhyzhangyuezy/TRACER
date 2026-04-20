from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data import apply_stage_intervals
from campaign_mem.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply stage intervals to canonical alert events.")
    parser.add_argument("--config", required=True, help="YAML config for stage interval labeling.")
    args = parser.parse_args()
    result = apply_stage_intervals(load_yaml(args.config))
    print(f"Labeled {result['labeled_event_rows']} event rows -> {result['output_path']}")


if __name__ == "__main__":
    main()
