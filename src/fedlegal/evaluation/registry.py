"""Metric registry aligned with `paper.md` and AGENTS instructions."""

REQUIRED_METRICS: tuple[str, ...] = (
    "legal_accuracy",
    "citation_consistency",
    "reasoning_coherence",
    "hallucination_rate",
    "communication_cost",
    "client_drift",
    "cross_jurisdiction_generalization",
    "privacy_leakage_risk",
    "aggregation_stability",
)
