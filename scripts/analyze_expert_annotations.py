"""Summarize returned human authority annotations without inventing missing labels."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

STATES = ("transferable", "authority_local", "unsupported", "unresolved")
REQUIRED = ("issue", "comparable_yes_no", "routing_state", "authority_status_A",
            "authority_status_B", "operative_rule_A", "operative_rule_B",
            "reason_and_pinpoint", "reviewer_id")


def read_sheet(path: Path) -> tuple[dict[str, dict[str, str]], dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    confirmed = {}
    for row in rows:
        if row.get("confirmed_by_human", "").strip().lower() != "yes":
            continue
        missing = [key for key in REQUIRED if not row.get(key, "").strip()]
        if missing:
            raise ValueError(f"{path.name}/{row['item_id']}: missing {missing}")
        if row["routing_state"] not in STATES or row["comparable_yes_no"].lower() not in {"yes", "no"}:
            raise ValueError(f"{path.name}/{row['item_id']}: invalid routing/comparability label")
        if row["item_id"] in confirmed:
            raise ValueError(f"Duplicate item: {row['item_id']}")
        confirmed[row["item_id"]] = row
    return confirmed, {
        "file": path.name, "candidate_count": len(rows), "human_confirmed": len(confirmed),
        "pending": len(rows)-len(confirmed),
        "reviewer_ids": sorted({row["reviewer_id"] for row in confirmed.values()}),
        "routing_counts": dict(Counter(row["routing_state"] for row in confirmed.values())),
        "partition_counts": dict(Counter(row["partition"] for row in confirmed.values())),
    }


def agreement(first: dict, second: dict, field: str, shared: list[str]) -> dict:
    a = [first[key][field] for key in shared]
    b = [second[key][field] for key in shared]
    n = len(shared)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label]*cb[label] for label in set(a+b)) / (n*n)
    return {"n": n, "raw_agreement": observed,
            "cohen_kappa": (observed-expected)/(1-expected) if expected < 1 else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = [read_sheet(path) for path in args.sheets]
    result = {"sheets": [summary for _, summary in loaded], "agreement": None,
              "status": "awaiting_human_annotations"}
    if any(rows for rows, _ in loaded):
        result["status"] = "single_annotator_or_partial_review"
    if len(loaded) == 2:
        first, second = loaded[0][0], loaded[1][0]
        ids_a, ids_b = set(loaded[0][1]["reviewer_ids"]), set(loaded[1][1]["reviewer_ids"])
        shared = sorted(first.keys() & second.keys())
        if shared and len(ids_a) == len(ids_b) == 1 and ids_a.isdisjoint(ids_b):
            result["status"] = "two_annotator_overlap_available"
            result["agreement"] = {
                field: agreement(first, second, field, shared)
                for field in ("routing_state", "comparable_yes_no")
            }
            result["routing_disagreements"] = [key for key in shared if first[key]["routing_state"] != second[key]["routing_state"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
