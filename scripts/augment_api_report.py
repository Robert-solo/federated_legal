"""Add parsing-coverage metrics to an existing API baseline report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    predictions_path = args.run_dir / "predictions.jsonl"
    report_path = args.run_dir / "report.json"
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    total = len(rows)
    parsed = sum(row.get("predicted_label") is not None and not row.get("error") for row in rows)
    errors = sum(bool(row.get("error")) for row in rows)
    conditional_correct = sum(
        bool(row.get("correct"))
        for row in rows
        if row.get("predicted_label") is not None and not row.get("error")
    )
    report.update(
        parsed=parsed,
        parse_rate=parsed / total if total else 0.0,
        api_errors=errors,
        api_error_rate=errors / total if total else 0.0,
        conditional_correct=conditional_correct,
        conditional_accuracy=conditional_correct / parsed if parsed else 0.0,
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("accuracy", "parsed", "parse_rate", "conditional_accuracy", "api_errors")}, indent=2))


if __name__ == "__main__":
    main()
