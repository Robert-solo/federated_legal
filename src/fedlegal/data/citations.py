"""Citation extraction for legal datasets."""

from __future__ import annotations

import re

STATUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d+\s+U\.S\.C\.?\s*§+\s*[\w().-]+", re.IGNORECASE),
    re.compile(r"\b\d+\s+C\.F\.R\.?\s*§+\s*[\w().-]+", re.IGNORECASE),
    re.compile(r"\b[A-Z][A-Za-z .'-]+ v\. [A-Z][A-Za-z .'-]+,\s*\d+[^.;,\n]*\(\d{4}\)"),
    re.compile(r"\bArticle\s+\d+[A-Za-z-]*\b", re.IGNORECASE),
    re.compile(r"\bSection\s+\d+[A-Za-z0-9().-]*\b", re.IGNORECASE),
    re.compile(r"\bClause\s+\d+[A-Za-z0-9().-]*\b", re.IGNORECASE),
    re.compile(r"第[一二三四五六七八九十百千万零〇\d]+条"),
    re.compile(r"《[^》]{2,80}》"),
)


def extract_citations(text: str, extra_values: list[str] | None = None) -> list[str]:
    """Extract likely statute, case, and contract citation strings."""

    citations: set[str] = set()
    for value in extra_values or []:
        cleaned = normalize_citation(value)
        if cleaned:
            citations.add(cleaned)

    for pattern in STATUTE_PATTERNS:
        for match in pattern.findall(text):
            cleaned = normalize_citation(match)
            if cleaned:
                citations.add(cleaned)

    return sorted(citations)


def normalize_citation(value: str) -> str:
    """Normalize whitespace in one citation value."""

    return " ".join(str(value).strip().rstrip(".,;").split())
