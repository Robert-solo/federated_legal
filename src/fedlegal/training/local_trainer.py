"""Local Legal LLM adapter trainer boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fedlegal.federated.payload import AdapterPayload, empty_adapter_payload
from fedlegal.training.peft_pipeline import AdapterTrainingPlan


@dataclass(frozen=True)
class LocalTrainingResult:
    """Result of one local federated training step."""

    payload: AdapterPayload
    num_examples: int
    metrics: dict[str, float]


class LocalAdapterTrainer:
    """Boundary for local HuggingFace + PEFT training.

    The class provides parameter import/export and a deterministic dry-run path.
    Full gradient training will be added later with HuggingFace `Trainer` or a
    custom loop. This keeps Flower orchestration testable without downloading LLMs.
    """

    def __init__(
        self,
        plan: AdapterTrainingPlan,
        dataset_path: str | Path | None = None,
        dry_run: bool = True,
        jurisdiction: str = "",
        legal_tradition: str = "",
    ) -> None:
        self.plan = plan
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.dry_run = dry_run
        self.jurisdiction = jurisdiction
        self.legal_tradition = legal_tradition
        self._payload = empty_adapter_payload()

    @property
    def parameter_keys(self) -> list[str]:
        """Adapter parameter keys in deterministic order."""

        return sorted(self._payload.tensors)

    def export_payload(self) -> AdapterPayload:
        """Export current adapter payload."""

        return self._payload

    def import_payload(self, payload: AdapterPayload) -> None:
        """Load server adapter parameters into the local trainer."""

        self._payload = payload

    def fit(self, round_id: int, local_epochs: int) -> LocalTrainingResult:
        """Run local training or a deterministic dry run."""

        if not self.dry_run:
            raise NotImplementedError("Full HuggingFace PEFT training is not implemented yet.")

        num_examples = self._count_examples()
        scale = float(round_id + local_epochs + max(num_examples, 1) * 0.001)
        tensors = {
            key: _add_scalar(value, scale)
            for key, value in self._payload.tensors.items()
        }
        self._payload = AdapterPayload(tensors)
        return LocalTrainingResult(
            payload=self._payload,
            num_examples=num_examples,
            metrics={
                "local_epochs": float(local_epochs),
                "num_examples": float(num_examples),
                "dry_run": 1.0,
                "jurisdiction": self.jurisdiction,
                "legal_tradition": self.legal_tradition,
                "citations": self._synthetic_citations(),
                "reasoning_embedding": self._synthetic_embedding(round_id),
                "verdict_distribution": self._synthetic_verdict_distribution(),
                "contradiction_score": self._synthetic_contradiction_score(),
            },
        )

    def evaluate(self) -> dict[str, float]:
        """Return local evaluation metrics for orchestration smoke tests."""

        return {"num_examples": float(self._count_examples()), "dry_run": 1.0}

    def _count_examples(self) -> int:
        if not self.dataset_path or not self.dataset_path.exists():
            return 0
        with self.dataset_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def _synthetic_citations(self) -> list[str]:
        if "chinese" in self.jurisdiction:
            return ["《民法典》", "第五百条"]
        if "us" in self.jurisdiction:
            return ["42 U.S.C. § 1983"]
        if "european" in self.jurisdiction:
            return ["Article 6"]
        if "contract" in self.jurisdiction:
            return ["Clause 12"]
        return []

    def _synthetic_embedding(self, round_id: int) -> list[float]:
        base = float((sum(ord(char) for char in self.jurisdiction) % 11) + 1)
        return [base / 10.0, float(round_id) / 10.0, len(self.legal_tradition) / 20.0]

    def _synthetic_verdict_distribution(self) -> list[float]:
        bucket = sum(ord(char) for char in self.jurisdiction) % 3
        distributions = ([0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6])
        return list(distributions[bucket])

    def _synthetic_contradiction_score(self) -> float:
        return (sum(ord(char) for char in self.legal_tradition) % 7) / 10.0


def _add_scalar(value: Any, scalar: float) -> Any:
    if hasattr(value, "astype"):
        return value.astype("float32") + scalar
    if isinstance(value, list):
        return [_add_scalar(item, scalar) for item in value]
    return float(value) + scalar
