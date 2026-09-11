import json
from pathlib import Path

import pytest

from fedlegal.aggregation import (
    build_casehold_proxy_manifest,
    load_calibration_manifest,
)
from fedlegal.federated.strategy import (
    _gold_error_gated_pairwise_diagnostics,
    _gold_error_gated_pairwise_exposures,
)


def test_proxy_manifest_records_probe_and_claim_scope(tmp_path: Path) -> None:
    probe = tmp_path / "conflict_probe.jsonl"
    probe.write_text('{"example_id":"probe-1"}\n', encoding="utf-8")
    manifest = build_casehold_proxy_manifest(
        run_dir=tmp_path,
        num_clients=2,
        probe_path=probe,
    )
    path = tmp_path / "conflict_calibration_manifest.json"
    manifest.write(path)

    loaded = load_calibration_manifest(path)

    assert loaded.evidence_level == "automatic_proxy_pilot"
    assert loaded.formal_legal_claims_allowed is False
    assert loaded.client("client_1").authority_compatibility == 1.0
    assert loaded.calibrate_verdict_error(0.25) == 0.25
    with pytest.raises(ValueError, match="expert-adjudicated"):
        load_calibration_manifest(path, require_expert=True)


def test_proxy_manifest_checks_probe_record_count(tmp_path: Path) -> None:
    probe = tmp_path / "conflict_probe.jsonl"
    probe.write_text('{"example_id":"probe-1"}\n', encoding="utf-8")
    manifest = build_casehold_proxy_manifest(
        run_dir=tmp_path,
        num_clients=1,
        probe_path=probe,
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    probe.write_text('{"example_id":"probe-1"}\n{"example_id":"probe-2"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="record count"):
        load_calibration_manifest(path)


def test_proxy_manifest_cannot_be_relabelled_as_formal_evidence(tmp_path: Path) -> None:
    probe = tmp_path / "probe.jsonl"
    probe.write_text('{"example_id":"probe-1"}\n', encoding="utf-8")
    manifest = build_casehold_proxy_manifest(
        run_dir=tmp_path,
        num_clients=1,
        probe_path=probe,
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["formal_legal_claims_allowed"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot release formal legal claims"):
        load_calibration_manifest(path)


def test_pairwise_exposure_does_not_penalize_correct_minority() -> None:
    exposures = _gold_error_gated_pairwise_exposures(
        {
            "correct_minority": (1, 1),
            "wrong_majority_a": (0, 0),
            "wrong_majority_b": (0, 0),
        },
        probe_labels=(1, 1),
    )

    assert exposures["correct_minority"] == 0.0
    assert exposures["wrong_majority_a"] == 0.5
    assert exposures["wrong_majority_b"] == 0.5


def test_pairwise_exposure_requires_matching_public_probe() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        _gold_error_gated_pairwise_exposures(
            {"client_a": (0,), "client_b": (1,)},
            probe_labels=(0, 1),
        )


def test_pairwise_diagnostics_report_complete_probe_coverage() -> None:
    diagnostics = _gold_error_gated_pairwise_diagnostics(
        {
            "client_a": (0, 1),
            "client_b": (1, 1),
            "client_c": (0, 0),
        },
        probe_labels=(0, 1),
        expected_client_ids={"client_a", "client_b", "client_c"},
    )

    assert diagnostics.comparison_counts == {
        "client_a": 4,
        "client_b": 4,
        "client_c": 4,
    }
    assert diagnostics.expected_comparisons_per_client == 4
    assert diagnostics.prediction_client_coverage == 1.0
    assert diagnostics.probe_pair_coverage == 1.0


def test_pairwise_diagnostics_expose_missing_client_coverage() -> None:
    diagnostics = _gold_error_gated_pairwise_diagnostics(
        {"client_a": (0, 1), "client_b": (1, 1)},
        probe_labels=(0, 1),
        expected_client_ids={"client_a", "client_b", "client_c"},
    )

    assert diagnostics.comparison_counts == {"client_a": 2, "client_b": 2}
    assert diagnostics.expected_comparisons_per_client == 4
    assert diagnostics.prediction_client_coverage == pytest.approx(2 / 3)
    assert diagnostics.probe_pair_coverage == pytest.approx(1 / 3)
