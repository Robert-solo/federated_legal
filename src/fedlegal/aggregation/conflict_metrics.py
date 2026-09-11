"""Legal conflict metrics for aggregation.

The concrete scoring models will be added after dataset and reasoning outputs
are available. The weighted penalty function is included so configs can be
validated against the paper's aggregation objective.
"""

from __future__ import annotations

from dataclasses import dataclass

from fedlegal.config.schemas import ConflictConfig


@dataclass(frozen=True)
class ConflictScores:
    """Conflict dimensions described in `paper.md`."""

    citation_conflict: float
    reasoning_conflict: float
    verdict_conflict: float
    rule_alignment_distance: float


def weighted_conflict_penalty(scores: ConflictScores, config: ConflictConfig) -> float:
    """Compute a weighted legal conflict penalty."""

    return config.penalty_lambda * (
        config.citation_weight * scores.citation_conflict
        + config.reasoning_weight * scores.reasoning_conflict
        + config.verdict_weight * scores.verdict_conflict
        + config.rule_alignment_weight * scores.rule_alignment_distance
    )
