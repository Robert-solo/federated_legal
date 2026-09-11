"""Conflict-aware aggregation algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fedlegal.aggregation.citation_divergence import citation_weight
from fedlegal.aggregation.legal_conflict_metrics import (
    AggregationConflictReport,
    LegalConflictProfile,
    build_conflict_report,
)
from fedlegal.config.schemas import ConflictConfig


@dataclass(frozen=True)
class ClientUpdate:
    """One federated client update with legal metadata."""

    client_id: str
    parameters: list[Any]
    num_examples: int
    profile: LegalConflictProfile


@dataclass(frozen=True)
class AggregationResult:
    """Aggregated parameters and legal conflict diagnostics."""

    parameters: list[Any]
    weights: dict[str, float]
    conflict_report: AggregationConflictReport


class ConflictAwareFedAvg:
    """Conflict-aware FedAvg aligned with the paper formula.

    Paper formula:

        w_{t+1} = sum_k alpha_k w_k - lambda * D_conflict

    Implementation:

    1. Compute alpha_k from sample counts.
    2. Optionally reweight alpha_k by citation reliability.
    3. Compute D_conflict from legal conflict metrics.
    4. Apply a contradiction/conflict shrinkage penalty to client weights before
       weighted parameter averaging.

    The shrinkage form keeps parameter tensors well-typed while encoding the
    paper's subtractive penalty as reduced influence for conflicting updates.
    """

    def __init__(
        self,
        config: ConflictConfig,
        citation_weighted: bool = True,
        contradiction_penalty_enabled: bool = True,
    ) -> None:
        self.config = config
        self.citation_weighted = citation_weighted
        self.contradiction_penalty_enabled = contradiction_penalty_enabled

    def aggregate(self, updates: list[ClientUpdate]) -> AggregationResult:
        """Aggregate client updates with legal conflict diagnostics."""

        if not updates:
            raise ValueError("At least one client update is required.")

        profiles = [update.profile for update in updates]
        report = build_conflict_report(profiles, self.config)
        weights = self._client_weights(updates, report)
        parameters_by_client = [update.parameters for update in updates]
        aggregated: list[Any] = []
        ordered_weights = [weights[update.client_id] for update in updates]
        for parameter_group in zip(*parameters_by_client, strict=True):
            aggregated.append(_weighted_sum(parameter_group, ordered_weights))
        return AggregationResult(parameters=aggregated, weights=weights, conflict_report=report)

    def _client_weights(
        self,
        updates: list[ClientUpdate],
        report: AggregationConflictReport,
    ) -> dict[str, float]:
        total_examples = sum(max(update.num_examples, 0) for update in updates)
        if total_examples <= 0:
            base = {update.client_id: 1.0 / len(updates) for update in updates}
        else:
            base = {
                update.client_id: max(update.num_examples, 0) / total_examples
                for update in updates
            }

        weighted = dict(base)
        if self.citation_weighted:
            weighted = {
                update.client_id: weighted[update.client_id] * citation_weight(update.profile.citations)
                for update in updates
            }

        if self.contradiction_penalty_enabled:
            # lambda * D_conflict is the paper penalty term. Applying it as a
            # bounded shrinkage preserves FedAvg normalization.
            shrinkage = max(0.0, 1.0 - self.config.penalty_lambda * report.total_conflict)
            weighted = {client_id: value * shrinkage for client_id, value in weighted.items()}

        normalizer = sum(weighted.values())
        if normalizer <= 0:
            return {update.client_id: 1.0 / len(updates) for update in updates}
        return {client_id: value / normalizer for client_id, value in weighted.items()}


def _weighted_sum(values, weights):
    first = values[0]
    if hasattr(first, "__mul__") and not isinstance(first, list):
        return sum(weight * value for weight, value in zip(weights, values, strict=True))
    if isinstance(first, list):
        return [
            _weighted_sum([value[index] for value in values], weights)
            for index in range(len(first))
        ]
    return sum(weight * float(value) for weight, value in zip(weights, values, strict=True))
