from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fedlegal.config.schemas import DatasetSourceConfig
from fedlegal.data.adapters import DatasetAdapter


def test_adapter_rejects_missing_required_jurisdiction(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"id": "case-1", "text": "facts", "label": 1}) + "\n")
    source = DatasetSourceConfig(
        name="LexGLUE",
        task="ecthr_a",
        raw_path=path,
        text_fields=["text"],
        label_field="label",
        jurisdiction_field="respondent_state",
        require_explicit_jurisdiction=True,
    )

    with pytest.raises(ValueError, match="requires explicit field 'respondent_state'"):
        DatasetAdapter(source, [], tmp_path).load()


def test_metadata_audit_passes_only_complete_explicit_values(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    rows = [
        {"id": "case-1", "respondent_state": "France"},
        {"id": "case-2", "respondent_state": "Germany"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_authentic_metadata.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(path), "--field", "respondent_state"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["status"] == "passed"
