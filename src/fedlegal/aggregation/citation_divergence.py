"""Citation divergence metrics for legal federated aggregation."""

from __future__ import annotations

import re


def citation_divergence(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Return Jaccard distance over normalized legal citations.

    Citation conflict in the paper measures inconsistency in cited statutes and
    authorities. Jaccard distance gives a bounded proxy:

        D_citation = 1 - |C_i intersection C_j| / |C_i union C_j|
    """

    left_set = normalize_citation_set(left)
    right_set = normalize_citation_set(right)
    if not left_set and not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return 1.0 - len(left_set & right_set) / len(union)


def citation_weight(citations: tuple[str, ...], min_weight: float = 0.5) -> float:
    """Compute a client reliability weight from citation density.

    Clients that provide more legal authorities get a larger aggregation weight,
    while uncited reasoning is not discarded.
    """

    unique_count = len(normalize_citation_set(citations))
    return min(1.0, min_weight + 0.1 * unique_count)


def normalize_citation_set(citations: tuple[str, ...]) -> set[str]:
    """Normalize citation strings for set comparison."""

    normalized: set[str] = set()
    for citation in citations:
        cleaned = re.sub(r"\s+", " ", citation.strip().lower())
        cleaned = cleaned.rstrip(".,;")
        if cleaned:
            normalized.add(cleaned)
    return normalized
