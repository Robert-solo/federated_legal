"""Preprocess legal datasets and optionally create federated partitions."""

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
from fedlegal.data import partition_processed_datasets, preprocess_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to a dataset YAML or experiment YAML.")
    parser.add_argument(
        "--partition",
        action="store_true",
        help="Create federated partitions after preprocessing.",
    )
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    config = load_data_config(args.config)
    if args.output_root:
        config = config.model_copy(
            update={
                "processed_dir": args.output_root / "processed",
                "partition_dir": args.output_root / "partitions",
            }
        )
    manifests = preprocess_datasets(config, limit_per_source=args.limit_per_source)
    result = {"processed": [manifest.model_dump(mode="json") for manifest in manifests]}

    if args.partition:
        plan = partition_processed_datasets(config)
        result["partition"] = {
            "strategy": plan.strategy,
            "output_dir": plan.output_dir,
            "client_sizes": plan.client_sizes,
        }

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
