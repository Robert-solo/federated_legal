"""Run the frozen revision protocol with real PEFT clients and Flower strategies.

Clients execute sequentially on one GPU. Flower receives the same NumPyClient
FitRes/EvaluateRes payloads as the network runner; communication is logical bytes.
The independent test set is opened only after the final training round.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import flwr as fl
import numpy as np
import torch
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts import run_real_flower_peft as runtime


SPECS = {
    "fedavg_10": ("fedavg", 10, 0),
    "fedprox_10": ("fedprox", 10, 0),
    "ditto_8_2": ("ditto", 8, 2),
    "unguarded_8_2": ("personalization_only", 8, 2),
    "guarded_8_2": ("personalization_guarded", 8, 2),
    "unguarded_10_2": ("personalization_only", 10, 2),
    "guarded_10_2": ("personalization_guarded", 10, 2),
    "flen_guarded_8_2": ("flen_guarded", 8, 2),
}


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", choices=SPECS, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--model", required=True)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/public/CaseHOLD/case_hold"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/revision_v2/casehold"))
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--validation-limit", type=int, default=80)
    parser.add_argument("--test-limit", type=int, default=1000)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The revision protocol requires a GPU.")
    out = cli.output_root / f"{cli.spec}_seed{cli.seed}"
    out.mkdir(parents=True, exist_ok=False)
    method, shared, local = SPECS[cli.spec]
    args = argparse.Namespace(
        run_name=out.name, method=method, seed=cli.seed,
        model_name=cli.model, dataset_name="lex_glue", dataset_config="case_hold",
        train_split="train", eval_split="validation", num_clients=4,
        num_rounds=cli.rounds, max_train_samples=cli.train_limit,
        max_eval_samples=cli.validation_limit, max_probe_samples=24,
        max_steps=shared, local_personalization_steps=local, adapter_composition="dual",
        local_epochs=1.0, batch_size=1, grad_accum=4, max_length=512,
        partition_strategy="jurisdiction", dirichlet_alpha=0.3,
        fedprox_mu=0.01, ditto_mu=0.01, conflict_lambda=0.2,
        diversity_floor=0.1, weight_smoothing=0.5, sample_count_cap=1000,
        require_expert_calibration=False, client_id=0,
    )
    runtime.load_legal_dataset = lambda name, config, split: Dataset.from_list(
        runtime.read_jsonl(cli.raw_root / f"{split}.jsonl")
    )
    runtime.prepare_data(args, out)
    resolved = runtime.resolved_experiment_config(args) | {
        "protocol": "revision_v2_dual_lora", "spec": cli.spec,
        "execution": "sequential_flower_numpy_clients",
        "test_split": "test", "test_sample_seed": 20261101,
        "test_limit": cli.test_limit, "test_evaluated_after_final_round_only": True,
        "gate_examples_excluded_from_shared_and_private_training": True,
        "expert_legal_conflict_validated": False,
    }
    runtime.write_json(out / "resolved_config.json", resolved)
    runtime.write_json(out / "environment.json", runtime.environment_metadata())
    runtime.write_json(out / "model_cache_manifest.json", {"snapshot_path": cli.model})
    clients = []
    for index in range(4):
        torch.manual_seed(cli.seed + index)
        np.random.seed(cli.seed + index)
        client_args = copy.copy(args)
        client_args.client_id = index
        clients.append(runtime.PeftFlowerClient(client_args, out, "cuda"))
    parameters = clients[0].get_parameters({})
    if method in runtime.CONFLICT_METHODS:
        calibration = runtime.load_calibration_manifest(out / "conflict_calibration_manifest.json")
        from fedlegal.config.schemas import ConflictConfig
        strategy = runtime.CheckpointingFLEN(
            output_dir=out, num_clients=4, target_jurisdiction=calibration.target_jurisdiction,
            calibration_manifest=calibration,
            probe_labels=tuple(int(row["label"]) for row in runtime.read_jsonl(out / "conflict_probe.jsonl")),
            conflict_config=ConflictConfig(
                enabled=True, penalty_lambda=0.2 if method=="flen_guarded" else 0.0,
                citation_weight=0, reasoning_weight=0, verdict_weight=1,
                rule_alignment_weight=0, diversity_floor=0.1, weight_smoothing=0.5,
                sample_count_cap=1000, missing_component_policy="error",
                cohort_policy="transferable_only", log_dir=out,
            ),
        )
    else:
        strategy = runtime.CheckpointingFedAvg(out, 4)
    status = fl.common.Status(code=fl.common.Code.OK, message="")
    started = time.time()
    for round_id in range(1, cli.rounds + 1):
        fits = []
        for index, client in enumerate(clients):
            arrays, n, metrics = client.fit(parameters, {"server_round": round_id})
            fits.append((SimpleNamespace(cid=str(index)), fl.common.FitRes(
                status=status, parameters=fl.common.ndarrays_to_parameters(arrays),
                num_examples=n, metrics=metrics,
            )))
        aggregate, _ = strategy.aggregate_fit(round_id, fits, [])
        parameters = fl.common.parameters_to_ndarrays(aggregate)
        evaluations = []
        for index, client in enumerate(clients):
            loss, n, metrics = client.evaluate(parameters, {"server_round": round_id})
            evaluations.append((SimpleNamespace(cid=str(index)), fl.common.EvaluateRes(
                status=status, loss=loss, num_examples=n, metrics=metrics,
            )))
        _, metrics = strategy.aggregate_evaluate(round_id, evaluations, [])
        print(json.dumps({"round": round_id, "validation": metrics, "seconds": time.time()-started}), flush=True)
    # Freeze and export the shared checkpoint before opening test records.
    clients[0]._load_numpy_state(parameters)
    if clients[0].dual:
        clients[0].dual.activate(local=False)
    clients[0].model.save_pretrained(out / "shared_adapter", selected_adapters=["default"])
    test = runtime.read_jsonl(cli.raw_root / "test.jsonl")
    random.Random(20261101).shuffle(test)
    test = test[:cli.test_limit]
    per_client = []
    predictions = []
    for index, client in enumerate(clients):
        client.eval_records = [runtime.format_casehold(row, i, 4) for i, row in enumerate(test) if i % 4==index]
        _, n, metrics = client.evaluate(parameters, {"server_round": cli.rounds})
        # evaluate() restores shared-training mode; select the tested composition again.
        if client.dual:
            client.dual.activate(local=client.dual.enabled)
        elif method == "ditto":
            client._load_numpy_state(client.ditto_parameters)
        _, labels = client._evaluate_records_with_predictions(client.eval_records)
        for row, label in zip(client.eval_records, labels, strict=True):
            predictions.append({"client": index, "example_id": row["example_id"], "gold": int(row["label"]), "prediction": label})
        per_client.append({"client": index, "examples": n, **metrics})
    total = sum(row["examples"] for row in per_client)
    result = {
        "protocol": resolved["protocol"], "spec": cli.spec, "seed": cli.seed,
        "test_split": "test", "test_examples": total,
        "accuracy": sum(row["accuracy"]*row["examples"] for row in per_client)/total,
        "worst_client_accuracy": min(row["accuracy"] for row in per_client),
        "clients": per_client, "elapsed_seconds": time.time()-started,
        "completed_rounds": cli.rounds,
    }
    runtime.write_json(out / "test_metrics.json", result)
    runtime.write_jsonl(out / "test_predictions.jsonl", predictions)
    runtime.write_json(out / "acceptance.json", {"status": "passed", "completed_rounds": cli.rounds, "independent_test": True})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    run()
