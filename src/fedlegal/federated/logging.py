"""Communication logging for federated legal training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommunicationEvent:
    """One client-server communication event."""

    round_id: int
    client_id: str
    direction: str
    payload_type: str
    num_tensors: int
    num_bytes: int
    metrics: dict[str, Any]


class CommunicationLogger:
    """Append-only JSONL logger for federated communication metrics."""

    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / f"{run_name}.jsonl"

    def log(self, event: CommunicationEvent) -> None:
        """Append one communication event."""

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")
