"""Validate downloaded legal datasets against provenance and adapter contracts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedlegal.config import load_data_config
from fedlegal.data.adapters import DatasetAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("manifest")
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument(
        "--output",
        default="outputs/data_validation/public_extended_validation.json",
    )
    args = parser.parse_args()

    config = load_data_config(args.config)
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_entries = [
        {"dataset_key": dataset["key"], **split}
        for dataset in manifest.get("datasets", [])
        for split in dataset.get("splits", [])
    ]

    adapter_results = []
    for source in config.datasets:
        adapter = DatasetAdapter(source, config.jurisdictions, config.raw_dir)
        records = list(adapter.iter_load(limit=args.sample_size))
        if not records:
            raise ValueError(f"No normalizable records found for {source.name}:{source.task}")
        adapter_results.append(
            {
                "dataset": source.name,
                "task": source.task,
                "split": source.split,
                "sampled_records": len(records),
                "jurisdictions": sorted({record.jurisdiction for record in records}),
                "labels_present": sum(record.label is not None for record in records),
            }
        )

    report: dict[str, Any] = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "manifest": str(manifest_path),
        "manifest_dataset_count": len(manifest.get("datasets", [])),
        "manifest_split_count": len(split_entries),
        "manifest_bytes": sum(int(entry.get("num_bytes", 0)) for entry in split_entries),
        "configured_sources_validated": len(adapter_results),
        "adapter_results": adapter_results,
        "status": "passed",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
if __name__ == "__main__":
    main()
