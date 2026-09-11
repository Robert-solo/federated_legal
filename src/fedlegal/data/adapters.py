"""Dataset-specific adapters for supported legal corpora."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from fedlegal.config.schemas import DatasetSourceConfig, JurisdictionConfig
from fedlegal.data.citations import extract_citations
from fedlegal.data.records import LegalDatasetRecord

SUPPORTED_DATASETS = {
    "CAIL",
    "LexGLUE",
    "CaseHOLD",
    "CUAD",
    "LeCaRDv2",
    "LegalBench",
    "MultiEURLEX",
}

DEFAULT_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "CAIL": ("fact", "facts", "case_text", "text"),
    "LexGLUE": ("text", "context", "question", "answer"),
    "CaseHOLD": ("prompt", "context", "holding", "text"),
    "CUAD": ("text", "clause", "contract_text", "question", "answer"),
    "LeCaRDv2": ("text", "title"),
    "LegalBench": ("text", "contract", "question", "citation"),
    "MultiEURLEX": ("text", "parallel_text"),
}

DEFAULT_LABEL_FIELDS: dict[str, tuple[str, ...]] = {
    "CAIL": ("accusation", "charge", "label", "term_of_imprisonment"),
    "LexGLUE": ("label", "answer"),
    "CaseHOLD": ("label", "answer", "holding"),
    "CUAD": ("label", "answer", "clause_type"),
    "LeCaRDv2": ("label", "score"),
    "LegalBench": ("label", "answer"),
    "MultiEURLEX": ("label", "labels"),
}

DEFAULT_JURISDICTION: dict[str, str] = {
    "CAIL": "chinese_civil_law",
    "LexGLUE": "european_regulatory_law",
    "CaseHOLD": "us_common_law",
    "CUAD": "contract_law",
    "LeCaRDv2": "chinese_criminal_law",
    "LegalBench": "us_common_law",
    "MultiEURLEX": "european_regulatory_law",
}

DEFAULT_TRADITION: dict[str, str] = {
    "CAIL": "civil_law",
    "LexGLUE": "civil_law_regulatory",
    "CaseHOLD": "common_law",
    "CUAD": "mixed_commercial",
    "LeCaRDv2": "civil_law",
    "LegalBench": "common_law",
    "MultiEURLEX": "civil_law_regulatory",
}


class DatasetAdapter:
    """Normalize one supported legal dataset into canonical records."""

    def __init__(
        self,
        source: DatasetSourceConfig,
        jurisdictions: list[JurisdictionConfig],
        raw_dir: Path,
    ) -> None:
        if source.name not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {source.name}")
        self.source = source
        self.jurisdictions = jurisdictions
        self.raw_dir = raw_dir

    def load(self) -> list[LegalDatasetRecord]:
        """Load and normalize records from local raw JSONL/JSON/CSV files."""

        return list(self.iter_load())

    def iter_load(self, limit: int | None = None) -> Iterable[LegalDatasetRecord]:
        """Yield normalized records without loading a full legal corpus into memory."""

        raw_path = self._resolve_raw_path()
        emitted = 0
        for index, raw in enumerate(_load_raw_records(raw_path)):
            record = self.normalize_record(raw, index)
            if record.text:
                yield record
                emitted += 1
                if limit is not None and emitted >= limit:
                    return

    def _resolve_raw_path(self) -> Path:
        if self.source.raw_path:
            return self.source.raw_path

        candidates = []
        if self.source.task:
            candidates.extend(
                [
                    self.raw_dir / self.source.name / self.source.task / f"{self.source.split}.jsonl",
                    self.raw_dir / self.source.name / self.source.task / f"{self.source.split}.json",
                    self.raw_dir / self.source.name / self.source.task / f"{self.source.split}.csv",
                ]
            )
        candidates.extend(
            [
                self.raw_dir / self.source.name / f"{self.source.split}.jsonl",
                self.raw_dir / self.source.name / f"{self.source.split}.json",
                self.raw_dir / self.source.name / f"{self.source.split}.csv",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        joined = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"No raw file found for {self.source.name}. Checked: {joined}")

    def normalize_record(self, raw: dict[str, Any], index: int) -> LegalDatasetRecord:
        """Normalize one source row into the canonical legal record schema."""

        text = _join_text_fields(
            raw,
            self.source.text_fields or list(DEFAULT_TEXT_FIELDS[self.source.name]),
        )
        label = _first_present(raw, [self.source.label_field] if self.source.label_field else [])
        if label is None:
            label = _first_present(raw, DEFAULT_LABEL_FIELDS[self.source.name])

        citation_values: list[str] = []
        for field in self.source.citation_fields:
            value = raw.get(field)
            if value is None:
                continue
            if isinstance(value, list):
                citation_values.extend(str(item) for item in value if item is not None)
            else:
                citation_values.append(str(value))
        citations = extract_citations(text, citation_values)
        jurisdiction = self._jurisdiction(raw)
        institution_id = self._institution_id(jurisdiction)
        dataset_id = self.source.name if not self.source.task else f"{self.source.name}:{self.source.task}"
        record_id = str(raw.get("id") or raw.get("case_id") or f"{dataset_id}-{index:08d}")

        return LegalDatasetRecord(
            record_id=record_id,
            dataset=dataset_id,
            split=self.source.split,
            text=text,
            label=label,
            citations=citations,
            jurisdiction=jurisdiction,
            legal_tradition=self._legal_tradition(jurisdiction),
            institution_id=institution_id,
            institution_type=self.source.institution_type or self._institution_type(institution_id),
            metadata={
                "source_fields": sorted(raw.keys()),
                "raw_path": str(self._resolve_raw_path()),
                "task": self.source.task,
                "authority_scope": self._metadata_value(raw, self.source.authority_scope_field),
                "languages": {
                    field: raw.get(field)
                    for field in self.source.language_fields
                    if raw.get(field) is not None
                },
                "jurisdiction_provenance": {
                    "field": self.source.jurisdiction_field,
                    "explicit": bool(
                        self.source.jurisdiction_field
                        and raw.get(self.source.jurisdiction_field) is not None
                    ),
                    "configured": self.source.jurisdiction is not None,
                },
            },
        )

    def _jurisdiction(self, raw: dict[str, Any]) -> str:
        field_value = (
            raw.get(self.source.jurisdiction_field)
            if self.source.jurisdiction_field
            else None
        )
        value = field_value or self.source.jurisdiction or raw.get("jurisdiction")
        if self.source.require_explicit_jurisdiction and field_value is None:
            raise ValueError(
                f"{self.source.name}:{self.source.task} requires explicit field "
                f"{self.source.jurisdiction_field!r}; record has no value"
            )
        return str(value or DEFAULT_JURISDICTION[self.source.name])

    @staticmethod
    def _metadata_value(raw: dict[str, Any], field: str | None) -> Any | None:
        return raw.get(field) if field else None

    def _institution_id(self, jurisdiction: str) -> str:
        if self.source.institution_id:
            return self.source.institution_id
        for item in self.jurisdictions:
            if item.jurisdiction == jurisdiction and self.source.name in item.dataset_names:
                return item.client_id
        for item in self.jurisdictions:
            if self.source.name in item.dataset_names:
                return item.client_id
        return f"{self.source.name.lower()}_institution"

    def _institution_type(self, institution_id: str) -> str:
        for item in self.jurisdictions:
            if item.client_id == institution_id:
                return item.institution_type
        return self.source.institution_type or "legal_institution"

    def _legal_tradition(self, jurisdiction: str) -> str:
        for item in self.jurisdictions:
            if item.jurisdiction == jurisdiction:
                return item.legal_tradition
        return DEFAULT_TRADITION[self.source.name]


def _load_raw_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            yield from payload
            return
        for key in ("data", "records", "examples"):
            if isinstance(payload, dict) and isinstance(payload.get(key), list):
                yield from payload[key]
                return
        if isinstance(payload, dict):
            yield payload
            return

    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)
        return

    raise ValueError(f"Unsupported raw dataset format: {path}")


def _join_text_fields(raw: dict[str, Any], fields: list[str] | tuple[str, ...]) -> str:
    parts: list[str] = []
    for field in fields:
        value = raw.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item is not None)
        elif isinstance(value, dict):
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        else:
            parts.append(str(value))
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _first_present(raw: dict[str, Any], fields: list[str] | tuple[str, ...]) -> Any | None:
    for field in fields:
        if field and field in raw:
            return raw[field]
    return None
