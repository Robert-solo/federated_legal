"""Jurisdiction similarity and rule-alignment scoring."""

from __future__ import annotations

from fedlegal.aggregation.legal_conflict_metrics import LegalConflictProfile

LEGAL_TRADITION_SIMILARITY: dict[tuple[str, str], float] = {
    ("civil_law", "civil_law"): 1.0,
    ("common_law", "common_law"): 1.0,
    ("civil_law_regulatory", "civil_law_regulatory"): 1.0,
    ("mixed_commercial", "mixed_commercial"): 1.0,
    ("hybrid", "hybrid"): 1.0,
    ("civil_law", "civil_law_regulatory"): 0.75,
    ("civil_law", "hybrid"): 0.65,
    ("common_law", "hybrid"): 0.7,
    ("common_law", "mixed_commercial"): 0.65,
    ("civil_law_regulatory", "hybrid"): 0.6,
    ("mixed_commercial", "hybrid"): 0.6,
}


def jurisdiction_similarity(left: LegalConflictProfile, right: LegalConflictProfile) -> float:
    """Estimate legal compatibility between two jurisdictions."""

    if left.jurisdiction and left.jurisdiction == right.jurisdiction:
        return 1.0
    key = (left.legal_tradition, right.legal_tradition)
    reverse_key = (right.legal_tradition, left.legal_tradition)
    if key in LEGAL_TRADITION_SIMILARITY:
        return LEGAL_TRADITION_SIMILARITY[key]
    if reverse_key in LEGAL_TRADITION_SIMILARITY:
        return LEGAL_TRADITION_SIMILARITY[reverse_key]
    if left.legal_tradition and left.legal_tradition == right.legal_tradition:
        return 0.85
    return 0.4


def jurisdiction_distance(left: LegalConflictProfile, right: LegalConflictProfile) -> float:
    """Return 1 - rule-alignment score."""

    return 1.0 - jurisdiction_similarity(left, right)
