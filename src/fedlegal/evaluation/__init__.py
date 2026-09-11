"""Evaluation metrics required by the research design."""

from fedlegal.evaluation.metrics import (
    citation_accuracy,
    client_drift,
    communication_cost,
    exact_match_f1,
    hallucination_rate,
    legal_accuracy,
    legal_consistency,
    load_examples,
    micro_f1,
)
from fedlegal.evaluation.registry import REQUIRED_METRICS
from fedlegal.evaluation.reporting import EvaluationRunner
from fedlegal.evaluation.schemas import EvaluationExample, EvaluationReport, MetricResult

__all__ = [
    "EvaluationExample",
    "EvaluationReport",
    "EvaluationRunner",
    "MetricResult",
    "REQUIRED_METRICS",
    "citation_accuracy",
    "client_drift",
    "communication_cost",
    "exact_match_f1",
    "hallucination_rate",
    "legal_accuracy",
    "legal_consistency",
    "load_examples",
    "micro_f1",
]
