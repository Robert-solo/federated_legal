"""Check that a Flower/PEFT run is complete enough for analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "manifest.json",
    "resolved_config.json",
    "real_experiment_report.json",
    "server_rounds.jsonl",
    "server_eval_rounds.jsonl",
    "communication_log.jsonl",
    "client_eval_log.jsonl",
)
CONFLICT_METHODS = {
    "target_cohort",
    "conflict_only",
    "personalization_only",
    "personalization_guarded",
    "flen_guarded",
    "flen",
}
ZERO_CONFLICT_METHODS = {
    "target_cohort",
    "personalization_only",
    "personalization_guarded",
}
LOCAL_RESIDUAL_METHODS = {
    "personalization_only",
    "personalization_guarded",
    "flen_guarded",
    "flen",
}
GUARDED_LOCAL_METHODS = {"personalization_guarded", "flen_guarded"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_run(
    run_dir: Path,
    expected_rounds: int,
    expected_clients: int,
    expected_method: str | None = None,
    require_conflict_activation: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    if missing:
        errors.extend(f"missing required file: {name}" for name in missing)
        return {
            "status": "failed",
            "run_dir": str(run_dir),
            "expected_rounds": expected_rounds,
            "expected_clients": expected_clients,
            "errors": errors,
        }

    manifest = read_json(run_dir / "manifest.json")
    resolved = read_json(run_dir / "resolved_config.json")
    report = read_json(run_dir / "real_experiment_report.json")
    server_rounds = read_jsonl(run_dir / "server_rounds.jsonl")
    eval_rounds = read_jsonl(run_dir / "server_eval_rounds.jsonl")
    communication = read_jsonl(run_dir / "communication_log.jsonl")
    client_eval = read_jsonl(run_dir / "client_eval_log.jsonl")

    expected_round_ids = set(range(1, expected_rounds + 1))
    expected_client_ids = {f"client_{client_id}" for client_id in range(expected_clients)}

    scalar_contracts = {
        "manifest.num_clients": (manifest.get("num_clients"), expected_clients),
        "resolved_config.num_clients": (resolved.get("num_clients"), expected_clients),
        "resolved_config.num_rounds": (resolved.get("num_rounds"), expected_rounds),
        "report.num_clients": (report.get("num_clients"), expected_clients),
        "report.num_rounds": (report.get("num_rounds"), expected_rounds),
        "report.server_rounds": (report.get("server_rounds"), expected_rounds),
        "report.eval_rounds": (report.get("eval_rounds"), expected_rounds),
    }
    if expected_method is not None:
        scalar_contracts["resolved_config.method"] = (resolved.get("method"), expected_method)
        scalar_contracts["report.method"] = (report.get("method"), expected_method)
    for name, (actual, expected) in scalar_contracts.items():
        if actual != expected:
            errors.append(f"{name}: expected {expected}, found {actual}")

    if report.get("real_training") is not True:
        errors.append("report.real_training must be true")

    server_ids = {int(row.get("round", -1)) for row in server_rounds}
    eval_ids = {int(row.get("round", -1)) for row in eval_rounds}
    if not expected_round_ids.issubset(server_ids):
        errors.append(f"server rounds are incomplete: {sorted(server_ids)}")
    if not expected_round_ids.issubset(eval_ids):
        errors.append(f"evaluation rounds are incomplete: {sorted(eval_ids)}")

    for label, rows in (("server", server_rounds), ("evaluation", eval_rounds)):
        for row in rows:
            if int(row.get("num_failures", -1)) != 0:
                errors.append(f"{label} round {row.get('round')} has failures")
            if int(row.get("num_results", -1)) != expected_clients:
                errors.append(
                    f"{label} round {row.get('round')} expected {expected_clients} results, "
                    f"found {row.get('num_results')}"
                )

    expected_events = {
        (round_id, client_id)
        for round_id in expected_round_ids
        for client_id in expected_client_ids
    }
    for label, rows in (("training", communication), ("client evaluation", client_eval)):
        observed_events = {
            (int(row.get("metrics", {}).get("server_round", -1)), str(row.get("client_id")))
            for row in rows
        }
        missing_events = expected_events - observed_events
        if missing_events:
            errors.append(f"{label} coverage is incomplete: {len(missing_events)} events missing")

    final_checkpoint = run_dir / f"server_round_{expected_rounds:04d}.npz"
    if not final_checkpoint.is_file():
        errors.append(f"missing final checkpoint: {final_checkpoint.name}")

    communication_bytes = sum(int(row.get("num_bytes", 0)) for row in communication)
    final_accuracy = report.get("final_accuracy")
    if not isinstance(final_accuracy, (int, float)) or not 0.0 <= float(final_accuracy) <= 1.0:
        errors.append(f"report.final_accuracy is invalid: {final_accuracy}")

    method = expected_method or str(resolved.get("method", "fedavg"))
    if method in CONFLICT_METHODS:
        for name in ("conflict_calibration_manifest.json", "aggregation_rounds.jsonl"):
            if not (run_dir / name).is_file():
                errors.append(f"missing FLEN artifact: {name}")
        aggregation_path = run_dir / "aggregation_rounds.jsonl"
        if aggregation_path.is_file():
            aggregation_rounds = read_jsonl(aggregation_path)
            aggregation_ids = {int(row.get("round", -1)) for row in aggregation_rounds}
            if not expected_round_ids.issubset(aggregation_ids):
                errors.append(f"aggregation rounds are incomplete: {sorted(aggregation_ids)}")
            conflict_activated = False
            for row in aggregation_rounds:
                missing_weights = {"base_weights", "final_weights", "conflict_exposures"} - set(row)
                if missing_weights:
                    errors.append(
                        f"aggregation round {row.get('round')} missing weights: "
                        f"{sorted(missing_weights)}"
                    )
                    continue
                final_values = [float(value) for value in row["final_weights"].values()]
                if set(row["final_weights"]) != expected_client_ids:
                    errors.append(f"aggregation round {row.get('round')} has incomplete final weights")
                    continue
                if any(not math.isfinite(value) or value < 0 for value in final_values):
                    errors.append(f"aggregation round {row.get('round')} has invalid final weights")
                if not math.isclose(sum(final_values), 1.0, rel_tol=0.0, abs_tol=1e-6):
                    errors.append(f"aggregation round {row.get('round')} final weights are not normalized")
                base = row["base_weights"]
                final = row["final_weights"]
                if set(base) != expected_client_ids:
                    errors.append(f"aggregation round {row.get('round')} has incomplete base weights")
                    continue
                max_weight_delta = max(
                    abs(float(final[client_id]) - float(base[client_id]))
                    for client_id in expected_client_ids
                )
                exposures = [
                    float(value)
                    for value in row.get("conflict_exposures", {}).values()
                    if value is not None
                ]
                exposure_spread = max(exposures) - min(exposures) if exposures else 0.0
                conflict_activated = conflict_activated or (
                    exposure_spread > 1e-12 and max_weight_delta > 1e-12
                )
                if method in ZERO_CONFLICT_METHODS and max_weight_delta > 1e-6:
                    errors.append(
                        f"aggregation round {row.get('round')} lambda-zero final weights "
                        "do not equal base weights"
                    )
            if (
                method in {"conflict_only", "flen_guarded", "flen"}
                and require_conflict_activation
                and not conflict_activated
            ):
                errors.append(
                    "FLEN conflict mechanism never activated: no round had both non-uniform "
                    "exposure and conflict-adjusted final weights"
                )
        if report.get("formal_legal_claims_allowed") is not False:
            errors.append("automatic proxy screening must disable formal legal claims")
        if (
            method in LOCAL_RESIDUAL_METHODS
            and int(resolved.get("local_personalization_steps", 0)) > 0
        ):
            for client_id in range(expected_clients):
                if not (run_dir / "local_residuals" / f"client_{client_id}.npz").is_file():
                    errors.append(f"missing local residual for client_{client_id}")
            if report.get("local_residual_uploaded") is not False:
                errors.append("FLEN report must confirm local residuals were not uploaded")
        if method in GUARDED_LOCAL_METHODS:
            paired_metrics = [
                row for row in client_eval
                if "personalization_accuracy_gain" in row.get("metrics", {})
            ]
            if not paired_metrics:
                errors.append("guarded personalization metrics are missing")
            gate_rate = report.get("local_gate_acceptance_rate")
            if not isinstance(gate_rate, (int, float)) or not 0.0 <= float(gate_rate) <= 1.0:
                errors.append(
                    f"report.local_gate_acceptance_rate is invalid: {gate_rate}"
                )
            gain = report.get("mean_personalization_accuracy_gain")
            if not isinstance(gain, (int, float)) or not -1.0 <= float(gain) <= 1.0:
                errors.append(
                    f"report.mean_personalization_accuracy_gain is invalid: {gain}"
                )

    return {
        "status": "passed" if not errors else "failed",
        "run_dir": str(run_dir),
        "run_name": report.get("run_name"),
        "dataset": report.get("dataset"),
        "model_name": report.get("model_name"),
        "expected_rounds": expected_rounds,
        "expected_clients": expected_clients,
        "server_rounds": len(server_rounds),
        "evaluation_rounds": len(eval_rounds),
        "training_events": len(communication),
        "client_evaluation_events": len(client_eval),
        "communication_bytes": communication_bytes,
        "final_accuracy": final_accuracy,
        "method": method,
        "require_conflict_activation": require_conflict_activation,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-rounds", type=int, required=True)
    parser.add_argument("--expected-clients", type=int, required=True)
    parser.add_argument("--expected-method")
    parser.add_argument("--require-conflict-activation", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_run(
        args.run_dir,
        args.expected_rounds,
        args.expected_clients,
        args.expected_method,
        args.require_conflict_activation,
    )
    output_path = args.output or args.run_dir / "acceptance.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
