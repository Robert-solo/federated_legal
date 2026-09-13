from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_prepare_multieurlex_rejects_split_leakage(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "en-de"
    raw.mkdir(parents=True)
    row = {
        "celex_id": "same",
        "id": "one",
        "label": [1],
        "language_primary": "English",
        "language_secondary": "German",
        "jurisdiction": "european_regulatory_law",
        "institution_id": "institute_eu_en-de",
        "text": "text",
    }
    for split in ("train", "validation", "test"):
        (raw / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_multieurlex_experiment.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--raw-root", str(tmp_path / "raw"), "--output", str(tmp_path / "out"), "--languages", "en-de", "--holdout", "en-de"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "CELEX leakage detected" in completed.stderr
