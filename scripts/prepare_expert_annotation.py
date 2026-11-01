"""Build blank human annotation sheets from verified official court PDFs.

No candidate pair is assigned a legal gold label by this generator.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from itertools import combinations
from pathlib import Path

from pypdf import PdfReader


def prepare(sources_path: Path, output: Path, per_source: int = 8) -> None:
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads(sources_path.read_text(encoding="utf-8"))["sources"]
    verified, failures, excerpts = [], [], []
    for source in sources:
        try:
            path = output / "sources" / f"{source['id']}.pdf"
            path.parent.mkdir(exist_ok=True)
            if not path.exists():
                subprocess.run([
                    "curl", "--fail", "--location", "--silent", "--show-error",
                    "--max-time", "45", "--output", str(path), source["url"],
                ], check=True)
                data = path.read_bytes()
                if not data.startswith(b"%PDF"):
                    raise ValueError("Official URL did not return PDF content")
                path.write_bytes(data)
            reader = PdfReader(path)
            page_texts = [(page.extract_text() or "") for page in reader.pages]
            # Retain page context and identify opinions; human verifies majority/dissent.
            for index, text in enumerate(page_texts[:per_source]):
                excerpts.append({**source, "page": index+1, "text": text})
            verified.append({**source, "download_status": "official_pdf_retrieved", "pages": len(reader.pages)})
        except Exception as exc:
            failures.append({"id": source["id"], "reason": str(exc)})
    pairs = []
    for left, right in combinations(excerpts, 2):
        if left["id"] == right["id"] and abs(left["page"]-right["page"]) > 1:
            continue
        if left["id"] != right["id"] and {left["id"],right["id"]} not in (
            {"johnson2015","welch2016"}, {"johnson2015","borden2021"},
            {"elonis2015","counterman2023"},
            {"chevron1984","loperbright2024"},
        ):
            continue
        pairs.append((left,right))
    rng = random.Random(20261101)
    rng.shuffle(pairs)
    def family_of(source_id: str) -> str:
        if source_id in {"johnson2015", "welch2016", "borden2021"}:
            return "acca"
        if source_id in {"elonis2015", "counterman2023"}:
            return "threats"
        return "administrative_review"
    # Equal family quotas keep the independent pilot test represented.
    selected = []
    for family in ("acca", "threats", "administrative_review"):
        selected.extend([pair for pair in pairs if family_of(pair[0]["id"]) == family][:40])
    rows = []
    for index,(left,right) in enumerate(selected):
        family = family_of(left["id"])
        rows.append({
            "item_id": f"authority-{index:04d}",
            "partition": {"acca":"development", "threats":"calibration", "administrative_review":"test"}[family],
            "source_family": family,
            "target_jurisdiction": "US-federal", "target_date": "2026-11-01",
            "source_A": left["url"], "title_A": left["title"], "date_A": left["date"],
            "page_A": left["page"], "excerpt_A": left["text"],
            "source_B": right["url"], "title_B": right["title"], "date_B": right["date"],
            "page_B": right["page"], "excerpt_B": right["text"],
            "issue": "", "comparable_yes_no": "", "routing_state": "",
            "authority_status_A": "", "authority_status_B": "",
            "operative_rule_A": "", "operative_rule_B": "",
            "reason_and_pinpoint": "", "reviewer_id": "", "confirmed_by_human": "",
        })
    if not rows:
        raise RuntimeError("No official source excerpts available for annotation.")
    for name in ("reviewer_1.csv", "reviewer_2.csv"):
        path = output / name
        if path.exists():
            raise FileExistsError(f"Preserve existing annotations: {path}")
        with path.open("w",newline="",encoding="utf-8-sig") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (output / "source_manifest.json").write_text(json.dumps({
        "scope": "US federal authority and temporal conflict pilot; not cross-national validation",
        "sources": verified, "unavailable_sources": failures,
        "candidate_pairs": len(rows), "human_gold_labels": 0,
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"verified_sources":len(verified),"unavailable_sources":failures,"blank_items":len(rows)}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--sources",type=Path,default=Path("configs/experiments/authority_annotation_sources.json"))
    parser.add_argument("--output",type=Path,default=Path("annotation/authority_v1"))
    args=parser.parse_args()
    prepare(args.sources,args.output)
