"""Adapter payload conversion for Flower communication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised in minimal environments
    np = None


@dataclass
class AdapterPayload:
    """PEFT adapter state exchanged between federated clients and the server."""

    tensors: dict[str, Any]

    def to_parameters(self) -> list[Any]:
        """Return tensors in deterministic key order for Flower NumPy clients."""

        return [self.tensors[key] for key in sorted(self.tensors)]

    @classmethod
    def from_parameters(cls, keys: list[str], parameters: list[Any]) -> "AdapterPayload":
        """Rebuild a named adapter payload from ordered parameters."""

        if len(keys) != len(parameters):
            raise ValueError("Adapter key count does not match parameter count.")
        return cls({key: parameter for key, parameter in zip(keys, parameters, strict=True)})

    def nbytes(self) -> int:
        """Total serialized tensor bytes, before protocol overhead."""

        total = 0
        for value in self.tensors.values():
            nbytes = getattr(value, "nbytes", None)
            if nbytes is not None:
                total += int(nbytes)
            else:
                total += len(json.dumps(value).encode("utf-8"))
        return total


def empty_adapter_payload() -> AdapterPayload:
    """Return a deterministic empty payload for dry-run orchestration."""

    if np is not None:
        return AdapterPayload({"adapter.placeholder": np.zeros(1, dtype=np.float32)})
    return AdapterPayload({"adapter.placeholder": [0.0]})
