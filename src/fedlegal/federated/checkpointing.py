"""Federated checkpoint management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in minimal environments
    np = None


class CheckpointManager:
    """Save server-round adapter checkpoints and metadata."""

    def __init__(self, checkpoint_dir: str | Path, run_name: str) -> None:
        self.run_dir = Path(checkpoint_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_round(
        self,
        round_id: int,
        parameter_keys: list[str],
        parameters: list[Any],
        metrics: dict[str, Any] | None = None,
    ) -> Path:
        """Save one federated round checkpoint."""

        path = self.run_dir / f"round_{round_id:04d}.npz"
        payload = {key: value for key, value in zip(parameter_keys, parameters, strict=True)}
        if np is not None:
            np.savez_compressed(path, **payload)
        else:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")

        metadata_path = self.run_dir / f"round_{round_id:04d}.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "round_id": round_id,
                    "parameter_keys": parameter_keys,
                    "metrics": metrics or {},
                    "checkpoint": str(path),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        return path
