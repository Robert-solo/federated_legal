"""Structured legal case representation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LegalCase(BaseModel):
    """Input unit for federated multi-agent legal reasoning."""

    case_id: str
    jurisdiction: str
    facts: list[str] = Field(default_factory=list)
    legal_issues: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    requested_relief: str | None = None
