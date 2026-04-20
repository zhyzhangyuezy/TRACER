from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from campaign_mem.data import audit_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Campaign-MEM dataset splits and leakage constraints.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing train/dev/test npz splits.")
    parser.add_argument("--output", required=True, help="Path to write audit JSON.")
    args = parser.parse_args()
    report = audit_dataset(dataset_dir=args.dataset_dir, output_path=args.output)
    print(f"incident_leakage_free={report['checks']['incident_leakage_free']}")
    print(f"event_disjoint_family_free={report['checks']['event_disjoint_family_free']}")


if __name__ == "__main__":
    main()
