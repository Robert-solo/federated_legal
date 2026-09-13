"""Prepare leakage-free MultiEURLEX language-client experiments.

This script only constructs and audits manifests. It does not infer a national
jurisdiction from language and does not claim that language clients are courts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_sample(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row.get("celex_id") or row.get("id")))
    if limit is None or len(ordered) <= limit:
        return ordered
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ordered)), limit))
    return [ordered[index] for index in indices]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--languages", nargs="+", default=["en-de", "en-fr", "en-es", "en-pl"])
    parser.add_argument("--holdout", default="en-pl")
    parser.add_argument("--limit-per-client", type=int, default=None)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    if args.holdout not in args.languages:
        raise SystemExit("--holdout must be one of --languages")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source_paths: dict[str, dict[str, Path]] = {}
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for language in args.languages:
        source_paths[language] = {}
        records[language] = {}
        for split in ("train", "validation", "test"):
            path = args.raw_root / language / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            source_paths[language][split] = path
            rows = read_jsonl(path)
            records[language][split] = rows

    split_ids: dict[str, set[str]] = {}
    for language in args.languages:
        for split in ("train", "validation", "test"):
            key = f"{language}:{split}"
            ids = [str(row.get("celex_id") or row.get("id")) for row in records[language][split]]
            split_ids[key] = set(ids)
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate CELEX identifiers in {key}")

    cross_split_overlaps: dict[str, list[str]] = {}
    for language in args.languages:
        splits = ("train", "validation", "test")
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1 :]:
                overlap = sorted(split_ids[f"{language}:{left}"] & split_ids[f"{language}:{right}"])
                if overlap:
                    cross_split_overlaps[f"{language}:{left}|{right}"] = overlap[:20]
    if cross_split_overlaps:
        raise ValueError(f"CELEX leakage detected: {cross_split_overlaps}")

    language_stats: dict[str, Any] = {}
    for language in args.languages:
        client_dir = output / "clients" / language
        selected: dict[str, int] = {}
        labels = Counter()
        for split in ("train", "validation", "test"):
            rows = stable_sample(records[language][split], args.limit_per_client, args.seed)
            write_jsonl(client_dir / f"{split}.jsonl", rows)
            selected[split] = len(rows)
            for row in rows:
                labels.update(str(label) for label in (row.get("label") or []))
        language_stats[language] = {
            "authority_scope": sorted({str(row.get("jurisdiction")) for row in records[language]["train"]}),
            "institution_ids": sorted({str(row.get("institution_id")) for row in records[language]["train"]}),
            "languages": sorted({
                f"{row.get('language_primary')}->{row.get('language_secondary')}"
                for row in records[language]["train"]
            }),
            "source_counts": {split: len(records[language][split]) for split in ("train", "validation", "test")},
            "selected_counts": selected,
            "unique_labels": len(labels),
            "top_labels": labels.most_common(20),
        }

    manifest = {
        "experiment": "multieurlex_language_holdout",
        "status": "prepared",
        "seed": args.seed,
        "languages": args.languages,
        "held_out_language": args.holdout,
        "train_languages": [language for language in args.languages if language != args.holdout],
        "task_scope": "multilingual EU regulatory text classification under a common EU authority",
        "not_supported": [
            "independent national-jurisdiction transfer",
            "respondent-state or court generalization",
            "legal-conflict construct validity",
        ],
        "source_fingerprint": fingerprint([path for language in args.languages for path in source_paths[language].values()]),
        "language_stats": language_stats,
        "cross_split_overlaps": cross_split_overlaps,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
