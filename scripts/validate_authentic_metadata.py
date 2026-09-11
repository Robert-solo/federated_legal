"""Audit whether a dataset can support an authentic jurisdiction experiment.

The audit is deliberately conservative: it never infers a country, court, or
jurisdiction from document text, filenames, labels, or client assignments.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedlegal.data.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--field", required=True, help="Explicit metadata field, e.g. respondent_state")
    parser.add_argument("--min-per-value", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    values: Counter[str] = Counter()
    total = 0
    missing = 0
    duplicate_ids = 0
    seen_ids: set[str] = set()
    files: list[dict[str, Any]] = []
    for path in args.paths:
        file_total = 0
        file_missing = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                total += 1
                file_total += 1
                value = row.get(args.field)
                if value is None or str(value).strip() == "":
                    missing += 1
                    file_missing += 1
                else:
                    values[str(value).strip()] += 1
                record_id = row.get("id") or row.get("case_id")
                if record_id is not None:
                    if str(record_id) in seen_ids:
                        duplicate_ids += 1
                    seen_ids.add(str(record_id))
        files.append({"path": str(path), "records": file_total, "missing": file_missing})

    under_minimum = {key: count for key, count in values.items() if count < args.min_per_value}
    report = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "field": args.field,
        "paths": files,
        "records": total,
        "missing": missing,
        "missing_rate": (missing / total) if total else None,
        "value_counts": dict(sorted(values.items())),
        "under_minimum": under_minimum,
        "duplicate_ids": duplicate_ids,
        "status": "passed" if total and not missing and not under_minimum and not duplicate_ids else "blocked",
        "interpretation": (
            "Explicit metadata is complete and sufficiently populated."
            if total and not missing and not under_minimum and not duplicate_ids
            else "Do not claim authentic jurisdiction holdout from this artifact."
        ),
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
