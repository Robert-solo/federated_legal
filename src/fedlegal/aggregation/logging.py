"""Aggregation logging utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fedlegal.aggregation.aggregators import AggregationResult


class AggregationLogger:
    """Append conflict-aware aggregation diagnostics as JSONL."""

    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"{run_name}_aggregation.jsonl"

    def log_round(self, round_id: int, result: AggregationResult) -> None:
        """Append one round of aggregation diagnostics."""

        payload = {
            "round_id": round_id,
            "weights": result.weights,
            "conflict_report": asdict(result.conflict_report),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
