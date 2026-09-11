import json
from pathlib import Path

from scripts.validate_real_flower_run import validate_run


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def build_run(run_dir: Path, rounds: int = 2, clients: int = 2) -> None:
    write_json(run_dir / "manifest.json", {"num_clients": clients})
    write_json(run_dir / "resolved_config.json", {"num_clients": clients, "num_rounds": rounds})
    write_json(
        run_dir / "real_experiment_report.json",
        {
            "run_name": "test_run",
            "dataset": "lex_glue/case_hold",
            "model_name": "test/model",
            "num_clients": clients,
            "num_rounds": rounds,
            "server_rounds": rounds,
            "eval_rounds": rounds,
            "real_training": True,
            "communication_bytes": rounds * clients * 10,
            "final_accuracy": 0.4,
        },
    )
    server_rows = [
        {"round": round_id, "num_results": clients, "num_failures": 0}
        for round_id in range(1, rounds + 1)
    ]
    write_jsonl(run_dir / "server_rounds.jsonl", server_rows)
    write_jsonl(run_dir / "server_eval_rounds.jsonl", server_rows)
    client_rows = [
        {
            "client_id": f"client_{client_id}",
            "num_bytes": 10,
            "metrics": {"server_round": round_id},
        }
        for round_id in range(1, rounds + 1)
        for client_id in range(clients)
    ]
    write_jsonl(run_dir / "communication_log.jsonl", client_rows)
    write_jsonl(run_dir / "client_eval_log.jsonl", client_rows)
    for round_id in range(1, rounds + 1):
        (run_dir / f"server_round_{round_id:04d}.npz").write_bytes(b"checkpoint")


def add_conflict_artifacts(
    run_dir: Path,
    *,
    method: str,
    activated: bool,
    rounds: int = 2,
    clients: int = 2,
) -> None:
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    resolved.update({"method": method, "local_personalization_steps": 0})
    write_json(run_dir / "resolved_config.json", resolved)
    report = json.loads(
        (run_dir / "real_experiment_report.json").read_text(encoding="utf-8")
    )
    report.update(
        {
            "method": method,
            "formal_legal_claims_allowed": False,
            "aggregation_rounds": rounds,
            "local_residual_uploaded": False,
        }
    )
    write_json(run_dir / "real_experiment_report.json", report)
    write_json(run_dir / "conflict_calibration_manifest.json", {"evidence_level": "pilot"})
    client_ids = [f"client_{client_id}" for client_id in range(clients)]
    base = {client_id: 1.0 / clients for client_id in client_ids}
    final = dict(base)
    exposures = {client_id: 0.1 for client_id in client_ids}
    if activated:
        exposures[client_ids[-1]] = 0.8
        final[client_ids[0]] += 0.05
        final[client_ids[-1]] -= 0.05
    aggregation_rows = [
        {
            "round": round_id,
            "base_weights": base,
            "candidate_weights": final,
            "floored_weights": final,
            "smoothed_weights": final,
            "final_weights": final,
            "conflict_exposures": exposures,
            "pairwise_comparison_counts": {
                client_id: clients - 1 for client_id in client_ids
            },
            "expected_comparisons_per_client": clients - 1,
            "prediction_client_coverage": 1.0,
            "probe_pair_coverage": 1.0,
        }
        for round_id in range(1, rounds + 1)
    ]
    write_jsonl(run_dir / "aggregation_rounds.jsonl", aggregation_rows)
    communication = [
        {
            "client_id": client_id,
            "num_bytes": 10,
            "uplink_bytes": 6,
            "metrics": {
                "server_round": round_id,
                "model_uplink_bytes": 5,
                "probe_payload_bytes": 1,
            },
        }
        for round_id in range(1, rounds + 1)
        for client_id in client_ids
    ]
    write_jsonl(run_dir / "communication_log.jsonl", communication)


def test_complete_run_passes(tmp_path: Path) -> None:
    build_run(tmp_path)

    result = validate_run(tmp_path, expected_rounds=2, expected_clients=2)

    assert result["status"] == "passed"
    assert result["errors"] == []


def test_duplicate_round_records_do_not_invalidate_complete_run(tmp_path: Path) -> None:
    build_run(tmp_path)
    with (tmp_path / "server_rounds.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"round": 2, "num_results": 2, "num_failures": 0}) + "\n")

    result = validate_run(tmp_path, expected_rounds=2, expected_clients=2)

    assert result["status"] == "passed"


def test_lambda_zero_conflict_run_requires_base_weight_equivalence(tmp_path: Path) -> None:
    build_run(tmp_path)
    add_conflict_artifacts(tmp_path, method="target_cohort", activated=True)

    result = validate_run(
        tmp_path,
        expected_rounds=2,
        expected_clients=2,
        expected_method="target_cohort",
    )

    assert result["status"] == "failed"
    assert any("lambda-zero final weights" in error for error in result["errors"])


def test_flen_conflict_activation_can_be_required(tmp_path: Path) -> None:
    build_run(tmp_path)
    add_conflict_artifacts(tmp_path, method="flen", activated=False)

    result = validate_run(
        tmp_path,
        expected_rounds=2,
        expected_clients=2,
        expected_method="flen",
        require_conflict_activation=True,
    )

    assert result["status"] == "failed"
    assert any("never activated" in error for error in result["errors"])


def test_flen_activated_conflict_run_passes(tmp_path: Path) -> None:
    build_run(tmp_path)
    add_conflict_artifacts(tmp_path, method="flen", activated=True)

    result = validate_run(
        tmp_path,
        expected_rounds=2,
        expected_clients=2,
        expected_method="flen",
        require_conflict_activation=True,
    )

    assert result["status"] == "passed"


def test_conflict_only_uses_activation_gate(tmp_path: Path) -> None:
    build_run(tmp_path)
    add_conflict_artifacts(tmp_path, method="conflict_only", activated=True)

    result = validate_run(
        tmp_path,
        expected_rounds=2,
        expected_clients=2,
        expected_method="conflict_only",
        require_conflict_activation=True,
    )

    assert result["status"] == "passed"


def test_personalization_only_preserves_lambda_zero_weights(tmp_path: Path) -> None:
    build_run(tmp_path)
    add_conflict_artifacts(tmp_path, method="personalization_only", activated=False)

    result = validate_run(
        tmp_path,
        expected_rounds=2,
        expected_clients=2,
        expected_method="personalization_only",
    )

    assert result["status"] == "passed"


def test_guarded_personalization_requires_gate_and_paired_metrics(tmp_path: Path) -> None:
    build_run(tmp_path)
    add_conflict_artifacts(tmp_path, method="personalization_guarded", activated=False)

    result = validate_run(
        tmp_path,
        expected_rounds=2,
        expected_clients=2,
        expected_method="personalization_guarded",
    )

    assert result["status"] == "failed"
    assert any("guarded personalization metrics are missing" in error for error in result["errors"])
    assert any("local_gate_acceptance_rate is invalid" in error for error in result["errors"])
