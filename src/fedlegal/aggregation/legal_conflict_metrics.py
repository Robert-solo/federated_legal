"""Legal conflict metrics aligned with the paper aggregation objective.

Paper objective:

    w_{t+1} = sum_k alpha_k w_k - lambda * D_conflict

where D_conflict combines citation conflict, reasoning conflict, verdict
conflict, and rule-alignment distance. These utilities estimate D_conflict from
client-side legal metadata without assuming access to raw judicial data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fedlegal.config.schemas import ConflictConfig


@dataclass(frozen=True)
class LegalConflictProfile:
    """Per-client legal conflict evidence used during aggregation."""

    client_id: str
    citations: tuple[str, ...] = ()
    reasoning_embedding: tuple[float, ...] = ()
    verdict_distribution: tuple[float, ...] = ()
    jurisdiction: str = ""
    legal_tradition: str = ""
    contradiction_score: float = 0.0


@dataclass(frozen=True)
class PairwiseConflict:
    """Conflict score between two clients."""

    left_client_id: str
    right_client_id: str
    citation_conflict: float
    reasoning_conflict: float
    verdict_conflict: float
    rule_alignment_distance: float
    contradiction_penalty: float

    @property
    def total(self) -> float:
        """Unweighted pairwise conflict magnitude."""

        return (
            self.citation_conflict
            + self.reasoning_conflict
            + self.verdict_conflict
            + self.rule_alignment_distance
            + self.contradiction_penalty
        ) / 5.0


@dataclass(frozen=True)
class AggregationConflictReport:
    """Aggregated conflict components for one server round."""

    citation_conflict: float
    reasoning_conflict: float
    verdict_conflict: float
    rule_alignment_distance: float
    contradiction_penalty: float
    total_conflict: float
    pairwise: tuple[PairwiseConflict, ...]


def contradiction_penalty(
    left: LegalConflictProfile,
    right: LegalConflictProfile,
) -> float:
    """Estimate contradiction penalty from verdict and explicit contradiction scores.

    Mathematical role: this is a component of D_conflict in
    w_{t+1} = sum_k alpha_k w_k - lambda * D_conflict. Higher values suppress
    the contribution of mutually contradictory client updates.
    """

    verdict_distance = l1_distance(left.verdict_distribution, right.verdict_distribution)
    explicit = abs(left.contradiction_score - right.contradiction_score)
    return clamp01(0.7 * verdict_distance + 0.3 * explicit)


def build_conflict_report(
    profiles: list[LegalConflictProfile],
    config: ConflictConfig,
) -> AggregationConflictReport:
    """Compute a weighted legal conflict report across participating clients."""

    if len(profiles) < 2:
        return AggregationConflictReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ())

    from fedlegal.aggregation.citation_divergence import citation_divergence
    from fedlegal.aggregation.jurisdiction_similarity import jurisdiction_distance

    pairwise: list[PairwiseConflict] = []
    for left_index, left in enumerate(profiles):
        for right in profiles[left_index + 1 :]:
            pairwise.append(
                PairwiseConflict(
                    left_client_id=left.client_id,
                    right_client_id=right.client_id,
                    citation_conflict=citation_divergence(left.citations, right.citations),
                    reasoning_conflict=cosine_distance(
                        left.reasoning_embedding,
                        right.reasoning_embedding,
                    ),
                    verdict_conflict=l1_distance(
                        left.verdict_distribution,
                        right.verdict_distribution,
                    ),
                    rule_alignment_distance=jurisdiction_distance(left, right),
                    contradiction_penalty=contradiction_penalty(left, right),
                )
            )

    citation = _mean(item.citation_conflict for item in pairwise)
    reasoning = _mean(item.reasoning_conflict for item in pairwise)
    verdict = _mean(item.verdict_conflict for item in pairwise)
    rule = _mean(item.rule_alignment_distance for item in pairwise)
    contradiction = _mean(item.contradiction_penalty for item in pairwise)

    # D_conflict = weighted legal divergence used by conflict-aware FedAvg.
    total = clamp01(
        config.citation_weight * citation
        + config.reasoning_weight * reasoning
        + config.verdict_weight * verdict
        + config.rule_alignment_weight * rule
        + contradiction
    )
    return AggregationConflictReport(
        citation_conflict=citation,
        reasoning_conflict=reasoning,
        verdict_conflict=verdict,
        rule_alignment_distance=rule,
        contradiction_penalty=contradiction,
        total_conflict=total,
        pairwise=tuple(pairwise),
    )


def profile_from_metrics(client_id: str, metrics: dict[str, Any]) -> LegalConflictProfile:
    """Build a conflict profile from Flower-style client metrics."""

    return LegalConflictProfile(
        client_id=client_id,
        citations=tuple(str(item) for item in metrics.get("citations", ())),
        reasoning_embedding=tuple(float(item) for item in metrics.get("reasoning_embedding", ())),
        verdict_distribution=tuple(float(item) for item in metrics.get("verdict_distribution", ())),
        jurisdiction=str(metrics.get("jurisdiction", "")),
        legal_tradition=str(metrics.get("legal_tradition", "")),
        contradiction_score=float(metrics.get("contradiction_score", 0.0)),
    )


def cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return 1 - cosine similarity for reasoning embeddings."""

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return clamp01(1.0 - dot / (left_norm * right_norm))


def l1_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return normalized L1 distance for verdict distributions."""

    if not left or not right or len(left) != len(right):
        return 0.0
    return clamp01(sum(abs(a - b) for a, b in zip(left, right, strict=True)) / 2.0)


def clamp01(value: float) -> float:
    """Clamp a numeric score into [0, 1]."""

    return max(0.0, min(1.0, float(value)))


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
