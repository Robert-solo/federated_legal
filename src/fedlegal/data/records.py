"""Canonical legal dataset record schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LegalDatasetRecord(BaseModel):
    """Normalized record saved under `data/processed`.

    The schema keeps raw legal text local while making downstream federation,
    citation analysis, and reasoning evaluation consistent across datasets.
    """

    record_id: str
    dataset: str
    split: str = "train"
    text: str
    label: str | int | float | list[str] | list[int] | list[float] | None = None
    citations: list[str] = Field(default_factory=list)
    jurisdiction: str
    legal_tradition: str | None = None
    institution_id: str
    institution_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    input_ids: list[int] = Field(default_factory=list)
    attention_mask: list[int] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)


class DatasetManifest(BaseModel):
    """Summary metadata for one processed dataset artifact."""

    name: str
    task: str | None = None
    split: str
    output_path: str
    num_records: int
    jurisdictions: list[str]
    institutions: list[str]
