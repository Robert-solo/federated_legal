"""Evaluation input and output schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationExample(BaseModel):
    """One legal reasoning prediction for evaluation."""

    example_id: str
    predicted_label: str | int | float | list[str] | list[int] | list[float] | None = None
    gold_label: str | int | float | list[str] | list[int] | list[float] | None = None
    predicted_citations: list[str] = Field(default_factory=list)
    gold_citations: list[str] = Field(default_factory=list)
    generated_text: str = ""
    supported_facts: list[str] = Field(default_factory=list)
    legal_consistency_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """Named scalar metric result."""

    name: str
    value: float
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Complete evaluation payload."""

    run_name: str
    metrics: dict[str, MetricResult]
    num_examples: int
    output_dir: str
