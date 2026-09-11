"""Create federated partitions from existing processed datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedlegal.config import load_data_config
from fedlegal.data import partition_processed_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a dataset YAML or experiment YAML.")
    args = parser.parse_args()

    config = load_data_config(args.config)
    plan = partition_processed_datasets(config)
    print(
        json.dumps(
            {
                "strategy": plan.strategy,
                "output_dir": plan.output_dir,
                "client_sizes": plan.client_sizes,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
