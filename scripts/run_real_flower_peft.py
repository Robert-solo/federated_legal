"""Run a real Flower + HuggingFace + PEFT federated LoRA experiment.

This script is intentionally self-contained for Slurm jobs. It downloads a real
legal dataset from HuggingFace, starts a Flower server, launches Flower clients,
performs local PEFT/LoRA training, and writes checkpoints/logs under
outputs/real_flower_peft.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from fedlegal.aggregation import (
    build_casehold_proxy_manifest,
    load_calibration_manifest,
)
from fedlegal.config.schemas import ConflictConfig
from fedlegal.federated.strategy import ConflictAwareFlowerStrategy


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["prepare-data", "prepare-model", "server", "client", "report", "export-predictions"],
        required=True,
    )
    parser.add_argument("--run-name", default="real_qwen25_casehold_flower")
    parser.add_argument("--output-dir", default="outputs/real_flower_peft")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset-name", default="lex_glue")
    parser.add_argument("--dataset-config", default="case_hold")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--num-rounds", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=400)
    parser.add_argument("--max-eval-samples", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--local-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--partition-strategy", choices=["iid", "label_dirichlet", "jurisdiction"], default="jurisdiction")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method",
        choices=[
            "fedavg",
            "fedprox",
            "target_cohort",
            "conflict_only",
            "personalization_only",
            "personalization_guarded",
            "flen_guarded",
            "flen",
        ],
        default="fedavg",
    )
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--max-probe-samples", type=int, default=24)
    parser.add_argument("--conflict-lambda", type=float, default=0.2)
    parser.add_argument("--diversity-floor", type=float, default=0.1)
    parser.add_argument("--weight-smoothing", type=float, default=0.5)
    parser.add_argument("--sample-count-cap", type=int, default=1000)
    parser.add_argument("--local-personalization-steps", type=int, default=0)
    parser.add_argument("--require-expert-calibration", action="store_true")
    parser.add_argument("--checkpoint-round", type=int, default=None)
    parser.add_argument("--prediction-limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"prepare-data", "prepare-model"}:
        write_json(output_dir / "resolved_config.json", resolved_experiment_config(args))
    if args.mode == "prepare-data":
        prepare_data(args, output_dir)
    elif args.mode == "prepare-model":
        write_json(output_dir / "environment.json", environment_metadata())
        prepare_model(args, output_dir)
    elif args.mode == "server":
        run_server(args, output_dir)
    elif args.mode == "client":
        run_client(args, output_dir)
    elif args.mode == "report":
        write_report(args, output_dir)
    elif args.mode == "export-predictions":
        export_predictions(args, output_dir)


def prepare_data(args: argparse.Namespace, output_dir: Path) -> None:
    train = load_legal_dataset(args.dataset_name, args.dataset_config, args.train_split)
    eval_ds = load_legal_dataset(args.dataset_name, args.dataset_config, args.eval_split)
    selected_train = min(args.max_train_samples, len(train))
    selected_probe = min(args.max_probe_samples, max(0, len(train) - selected_train))
    train_and_probe = train.select(range(selected_train + selected_probe))
    eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    train_records = [
        format_casehold(train_and_probe[index], index, args.num_clients)
        for index in range(selected_train)
    ]
    probe_records = [
        format_casehold(train_and_probe[index], index, args.num_clients)
        for index in range(selected_train, selected_train + selected_probe)
    ]
    eval_records = [format_casehold(row, index, args.num_clients) for index, row in enumerate(eval_ds)]
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "conflict_probe.jsonl", probe_records)
    write_jsonl(output_dir / "eval.jsonl", eval_records)

    shards_dir = output_dir / "client_shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shards = partition_records(
        train_records,
        num_clients=args.num_clients,
        strategy=args.partition_strategy,
        seed=args.seed,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    for client_id, shard in enumerate(shards):
        write_jsonl(shards_dir / f"client_{client_id}.jsonl", shard)

    eval_shards_dir = output_dir / "eval_shards"
    eval_shards_dir.mkdir(parents=True, exist_ok=True)
    for client_id in range(args.num_clients):
        shard = [row for index, row in enumerate(eval_records) if index % args.num_clients == client_id]
        write_jsonl(eval_shards_dir / f"client_{client_id}.jsonl", shard)

    partition_summary = summarize_partition(shards)
    manifest = {
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_records": len(train_records),
        "conflict_probe_records": len(probe_records),
        "eval_records": len(eval_records),
        "num_clients": args.num_clients,
        "model_name": args.model_name,
        "partition_strategy": args.partition_strategy,
        "dirichlet_alpha": args.dirichlet_alpha,
        "seed": args.seed,
        "partition_summary": partition_summary,
    }
    write_json(output_dir / "manifest.json", manifest)
    if args.method in CONFLICT_METHODS:
        if not probe_records:
            raise ValueError("FLEN methods require a non-empty disjoint conflict probe.")
        calibration = build_casehold_proxy_manifest(
            run_dir=output_dir,
            num_clients=args.num_clients,
            probe_path=output_dir / "conflict_probe.jsonl",
        )
        calibration.write(output_dir / "conflict_calibration_manifest.json")


def prepare_model(args: argparse.Namespace, output_dir: Path) -> None:
    """Populate HuggingFace cache before Slurm clients start training.

    This keeps GPU allocations focused on Flower/PEFT work. The model is still
    loaded by each client during training, but the files are already present in
    HF_HOME so compute nodes can run with local cache access.
    """

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_path = snapshot_download(
        repo_id=args.model_name,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.safetensors",
            "*.txt",
            "*.py",
            "tokenizer.*",
            "generation_config.json",
            "merges.txt",
            "vocab.*",
        ],
    )
    manifest = {
        "model_name": args.model_name,
        "snapshot_path": model_path,
        "tokenizer_class": tokenizer.__class__.__name__,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }
    write_json(output_dir / "model_cache_manifest.json", manifest)


def run_server(args: argparse.Namespace, output_dir: Path) -> None:
    if args.method in CONFLICT_METHODS:
        calibration = load_calibration_manifest(
            output_dir / "conflict_calibration_manifest.json",
            require_expert=args.require_expert_calibration,
        )
        strategy = CheckpointingFLEN(
            output_dir=output_dir,
            num_clients=args.num_clients,
            target_jurisdiction=calibration.target_jurisdiction,
            calibration_manifest=calibration,
            probe_labels=tuple(
                int(record["label"])
                for record in read_jsonl(output_dir / calibration.probe_path)
            ),
            conflict_config=ConflictConfig(
                enabled=True,
                penalty_lambda=(
                    0.0 if args.method in ZERO_CONFLICT_METHODS else args.conflict_lambda
                ),
                citation_weight=calibration.component_weights["citation"],
                reasoning_weight=calibration.component_weights["reasoning"],
                verdict_weight=calibration.component_weights["verdict"],
                rule_alignment_weight=calibration.component_weights["rule_alignment"],
                sample_count_cap=args.sample_count_cap,
                diversity_floor=args.diversity_floor,
                weight_smoothing=args.weight_smoothing,
                missing_component_policy="error",
                cohort_policy="transferable_only",
                calibration_manifest=output_dir / "conflict_calibration_manifest.json",
                log_dir=output_dir,
            ),
        )
    else:
        strategy = CheckpointingFedAvg(output_dir=output_dir, num_clients=args.num_clients)
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


def run_client(args: argparse.Namespace, output_dir: Path) -> None:
    client_seed = args.seed + args.client_id
    np.random.seed(client_seed)
    torch.manual_seed(client_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(client_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = PeftFlowerClient(args=args, output_dir=output_dir, device=device)
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


def export_predictions(args: argparse.Namespace, output_dir: Path) -> None:
    """Export prediction-level evidence from an existing server checkpoint."""

    checkpoint_round = args.checkpoint_round or args.num_rounds
    checkpoint_path = output_dir / f"server_round_{checkpoint_round:04d}.npz"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Server checkpoint not found: {checkpoint_path}")

    client_seed = args.seed + args.client_id
    np.random.seed(client_seed)
    torch.manual_seed(client_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(client_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = PeftFlowerClient(args=args, output_dir=output_dir, device=device)
    with np.load(checkpoint_path) as archive:
        parameters = [archive[name] for name in sorted(archive.files)]
    client._load_numpy_state(parameters)

    records = read_jsonl(output_dir / "eval.jsonl")
    if args.prediction_limit > 0:
        records = records[: args.prediction_limit]
    metrics, predictions = client._evaluate_records_with_predictions(records)
    if len(predictions) != len(records):
        raise ValueError(
            f"Prediction count mismatch: {len(predictions)} predictions for {len(records)} records"
        )

    prediction_path = output_dir / f"checkpoint_predictions_round_{checkpoint_round:04d}.jsonl"
    events = []
    for index, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
        gold = int(record["label"])
        events.append(
            {
                "index": index,
                "example_id": record.get("example_id"),
                "gold_label": gold,
                "predicted_label": prediction,
                "correct": prediction == gold,
                "predicted_citations": [],
                "gold_citations": record.get("citations", []),
                "generated_text": str(prediction),
                "supported_facts": [record.get("context", "")[:240]],
                "metadata": {
                    "task": "casehold",
                    "jurisdiction": record.get("jurisdiction", "unknown"),
                    "checkpoint_round": checkpoint_round,
                    "model_name": args.model_name,
                },
                "error": None,
            }
        )
    write_jsonl(prediction_path, events)
    write_json(
        output_dir / f"checkpoint_prediction_report_round_{checkpoint_round:04d}.json",
        {
            "run_name": args.run_name,
            "model_name": args.model_name,
            "checkpoint_round": checkpoint_round,
            "num_examples": len(events),
            "correct": int(metrics["correct"]),
            "accuracy": metrics["accuracy"],
            "eval_loss": metrics["eval_loss"],
            "prediction_path": str(prediction_path),
        },
    )


class PeftFlowerClient(fl.client.NumPyClient):
    def __init__(self, args: argparse.Namespace, output_dir: Path, device: str) -> None:
        self.args = args
        self.output_dir = output_dir
        self.device = device
        self.client_id = args.client_id
        model_path = resolve_model_path(args.model_name, output_dir)
        local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(base, lora).to(device)
        self.keys = sorted(get_peft_model_state_dict(self.model).keys())
        self.dataset = self._load_client_dataset()
        self.local_personalization_dataset = self.dataset
        self.personalization_gate_records: list[dict[str, Any]] = []
        if args.method in GUARDED_LOCAL_METHODS:
            self._prepare_personalization_gate()
        self.eval_records = read_jsonl(self.output_dir / "eval_shards" / f"client_{self.client_id}.jsonl")
        self.probe_records: list[dict[str, Any]] = []
        self.calibration = None
        self.local_residual = [np.zeros_like(array) for array in self._state_to_numpy()]
        if args.method in CONFLICT_METHODS:
            self.calibration = load_calibration_manifest(
                self.output_dir / "conflict_calibration_manifest.json",
                require_expert=args.require_expert_calibration,
            )
            self.probe_records = read_jsonl(self.output_dir / self.calibration.probe_path)
            self._load_local_residual()

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return self._state_to_numpy()

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        round_id = int(config.get("server_round", 0))
        downlink_bytes = int(sum(array.nbytes for array in parameters))
        incoming = [array.copy() for array in parameters]
        self._load_numpy_state(parameters)
        start = time.time()
        loss = self._train_one_round()
        params = self._state_to_numpy()
        probe_metrics = None
        probe_predictions: list[int] = []
        local_loss = None
        local_metrics: dict[str, float] = {}
        if self.calibration is not None:
            probe_metrics, probe_predictions = self._evaluate_records_with_predictions(
                self.probe_records
            )
        if (
            self.args.method in LOCAL_RESIDUAL_METHODS
            and self.args.local_personalization_steps > 0
        ):
            local_loss, local_metrics = self._update_local_residual(params)
        elapsed = time.time() - start
        model_uplink_bytes = int(sum(array.nbytes for array in params))
        probe_packet = json.dumps(probe_predictions) if probe_metrics is not None else ""
        probe_payload_bytes = len(probe_packet.encode("utf-8"))
        uplink_bytes = model_uplink_bytes + probe_payload_bytes
        drift_l2, drift_ratio = compute_client_drift(incoming, params)
        metrics = {
            "server_round": float(round_id),
            "train_loss": float(loss),
            "num_examples": float(len(self.dataset)),
            "uplink_bytes": float(uplink_bytes),
            "model_uplink_bytes": float(model_uplink_bytes),
            "probe_payload_bytes": float(probe_payload_bytes),
            "downlink_bytes": float(downlink_bytes),
            "total_bytes": float(uplink_bytes + downlink_bytes),
            "client_drift_l2": float(drift_l2),
            "client_drift_ratio": float(drift_ratio),
            "elapsed_sec": float(elapsed),
            "method": self.args.method,
        }
        if self.calibration is not None and probe_metrics is not None:
            client_id = f"client_{self.client_id}"
            routing = self.calibration.client(client_id)
            metrics.update(
                {
                    "client_id": client_id,
                    "target_jurisdiction": routing.target_jurisdiction,
                    "authority_compatibility": float(routing.authority_compatibility),
                    "routing_state": routing.routing_state.value,
                    "probe_accuracy": float(probe_metrics["accuracy"]),
                    "probe_examples": float(probe_metrics["num_examples"]),
                    "probe_predictions": probe_packet,
                    "calibration_evidence_level": self.calibration.evidence_level,
                }
            )
        if local_loss is not None:
            metrics["local_personalization_loss"] = float(local_loss)
            metrics["local_residual_l2"] = float(_arrays_l2(self.local_residual))
            metrics.update(local_metrics)
        self._log_client_event(metrics)
        return params, len(self.dataset), metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        personalized = (
            self.args.method in LOCAL_RESIDUAL_METHODS
            and self.args.local_personalization_steps > 0
        )
        shared_metrics = None
        if personalized:
            self._load_numpy_state(parameters)
            shared_metrics = self._evaluate_records(self.eval_records)
            evaluation_parameters = [
                shared + residual
                for shared, residual in zip(parameters, self.local_residual, strict=True)
            ]
        else:
            evaluation_parameters = parameters
        self._load_numpy_state(evaluation_parameters)
        start = time.time()
        eval_metrics = self._evaluate_records(self.eval_records)
        if shared_metrics is not None:
            eval_metrics.update(
                {
                    "shared_accuracy": float(shared_metrics["accuracy"]),
                    "shared_eval_loss": float(shared_metrics["eval_loss"]),
                    "personalized_accuracy": float(eval_metrics["accuracy"]),
                    "personalized_eval_loss": float(eval_metrics["eval_loss"]),
                    "personalization_accuracy_gain": float(
                        eval_metrics["accuracy"] - shared_metrics["accuracy"]
                    ),
                    "personalization_eval_loss_reduction": float(
                        shared_metrics["eval_loss"] - eval_metrics["eval_loss"]
                    ),
                }
            )
        eval_metrics["elapsed_sec"] = time.time() - start
        eval_metrics["server_round"] = float(config.get("server_round", 0))
        eval_metrics["downlink_bytes"] = float(sum(array.nbytes for array in parameters))
        self._log_eval_event(eval_metrics)
        return float(eval_metrics["eval_loss"]), int(eval_metrics["num_examples"]), eval_metrics

    def _load_client_dataset(self) -> Dataset:
        records = read_jsonl(self.output_dir / "client_shards" / f"client_{self.client_id}.jsonl")
        texts = [record["text"] for record in records]
        ds = Dataset.from_dict({"text": texts})

        def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=self.args.max_length,
                padding=False,
            )

        tokenized = ds.map(tokenize, batched=True, remove_columns=["text"])
        return tokenized

    def _prepare_personalization_gate(self) -> None:
        records = read_jsonl(
            self.output_dir / "client_shards" / f"client_{self.client_id}.jsonl"
        )
        if len(records) < 2:
            raise ValueError("Guarded personalization requires at least two client records.")
        gate_count = min(24, max(8, round(0.1 * len(records))), len(records) - 1)
        ordered_indices = sorted(
            range(len(records)),
            key=lambda index: stable_int(
                f"{self.args.seed}:{self.client_id}:{records[index].get('example_id', index)}"
            ),
        )
        gate_indices = set(ordered_indices[:gate_count])
        train_indices = [index for index in range(len(records)) if index not in gate_indices]
        self.personalization_gate_records = [records[index] for index in sorted(gate_indices)]
        self.local_personalization_dataset = self.dataset.select(train_indices)

    def _evaluate_records(self, records: list[dict[str, Any]]) -> dict[str, float]:
        metrics, _ = self._evaluate_records_with_predictions(records)
        return metrics

    def _evaluate_records_with_predictions(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, float], list[int]]:
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        predictions: list[int] = []
        with torch.no_grad():
            for record in records:
                endings = record.get("endings", [])
                label = int(record.get("label", -1))
                if not endings or label < 0 or label >= len(endings):
                    continue
                scores = [self._score_casehold_choice(record, ending) for ending in endings]
                prediction = int(np.argmin(scores))
                predictions.append(prediction)
                correct += int(prediction == label)
                total += 1
                loss_sum += float(scores[label])
        accuracy = correct / total if total else 0.0
        return (
            {
                "accuracy": accuracy,
                "correct": float(correct),
                "num_examples": float(total),
                "eval_loss": loss_sum / total if total else 0.0,
            },
            predictions,
        )

    def _score_casehold_choice(self, record: dict[str, Any], ending: str) -> float:
        prompt = casehold_prompt(record)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(str(ending), add_special_tokens=False).input_ids
        if not answer_ids:
            return float("inf")
        max_prompt_tokens = max(1, self.args.max_length - len(answer_ids))
        prompt_ids = prompt_ids[-max_prompt_tokens:]
        input_ids = torch.tensor([prompt_ids + answer_ids], device=self.device)
        labels = input_ids.clone()
        labels[:, : len(prompt_ids)] = -100
        outputs = self.model(input_ids=input_ids, labels=labels)
        return float(outputs.loss.detach().cpu().item())

    def _train_one_round(
        self,
        *,
        max_steps: int | None = None,
        phase: str = "shared",
        train_dataset: Dataset | None = None,
    ) -> float:
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        train_args = TrainingArguments(
            output_dir=str(
                self.output_dir / "client_outputs" / f"client_{self.client_id}" / phase
            ),
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_accum,
            num_train_epochs=self.args.local_epochs,
            max_steps=self.args.max_steps if max_steps is None else max_steps,
            learning_rate=2e-4,
            seed=self.args.seed + self.client_id,
            data_seed=self.args.seed + self.client_id,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            bf16=torch.cuda.is_available(),
        )
        trainer_cls = FedProxTrainer if self.args.method == "fedprox" else Trainer
        trainer_kwargs: dict[str, Any] = {}
        if self.args.method == "fedprox":
            trainer_kwargs = {
                "global_parameters": {
                    name: parameter.detach().clone()
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                },
                "fedprox_mu": self.args.fedprox_mu,
            }
        trainer = trainer_cls(
            model=self.model,
            args=train_args,
            train_dataset=self.dataset if train_dataset is None else train_dataset,
            data_collator=collator,
            **trainer_kwargs,
        )
        result = trainer.train()
        return float(result.training_loss)

    def _update_local_residual(
        self,
        shared_parameters: list[np.ndarray],
    ) -> tuple[float, dict[str, float]]:
        guarded = self.args.method in GUARDED_LOCAL_METHODS
        shared_gate_metrics = None
        if guarded:
            self._load_numpy_state(shared_parameters)
            shared_gate_metrics = self._evaluate_records(self.personalization_gate_records)
        personalized_start = [
            shared + residual
            for shared, residual in zip(shared_parameters, self.local_residual, strict=True)
        ]
        self._load_numpy_state(personalized_start)
        loss = self._train_one_round(
            max_steps=self.args.local_personalization_steps,
            phase="local",
            train_dataset=(self.local_personalization_dataset if guarded else self.dataset),
        )
        personalized = self._state_to_numpy()
        candidate_residual = [
            local - shared
            for local, shared in zip(personalized, shared_parameters, strict=True)
        ]
        gate_metrics: dict[str, float] = {}
        if guarded and shared_gate_metrics is not None:
            candidate_gate_metrics = self._evaluate_records(self.personalization_gate_records)
            accepted = (
                candidate_gate_metrics["accuracy"] >= shared_gate_metrics["accuracy"]
                and candidate_gate_metrics["eval_loss"] <= shared_gate_metrics["eval_loss"]
            )
            self.local_residual = (
                candidate_residual
                if accepted
                else [np.zeros_like(array) for array in shared_parameters]
            )
            gate_metrics = {
                "local_gate_accepted": float(accepted),
                "local_gate_examples": float(len(self.personalization_gate_records)),
                "local_gate_shared_accuracy": float(shared_gate_metrics["accuracy"]),
                "local_gate_candidate_accuracy": float(candidate_gate_metrics["accuracy"]),
                "local_gate_shared_eval_loss": float(shared_gate_metrics["eval_loss"]),
                "local_gate_candidate_eval_loss": float(candidate_gate_metrics["eval_loss"]),
            }
        else:
            self.local_residual = candidate_residual
        residual_path = self.output_dir / "local_residuals" / f"client_{self.client_id}.npz"
        residual_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            residual_path,
            **{f"param_{index:04d}": value for index, value in enumerate(self.local_residual)},
        )
        self._load_numpy_state(shared_parameters)
        return loss, gate_metrics

    def _load_local_residual(self) -> None:
        residual_path = self.output_dir / "local_residuals" / f"client_{self.client_id}.npz"
        if not residual_path.is_file():
            return
        with np.load(residual_path) as archive:
            loaded = [archive[name] for name in sorted(archive.files)]
        if len(loaded) != len(self.local_residual):
            raise ValueError("Local residual parameter structure does not match the shared adapter.")
        self.local_residual = loaded

    def _state_to_numpy(self) -> list[np.ndarray]:
        state = get_peft_model_state_dict(self.model)
        return [state[key].detach().cpu().float().numpy() for key in self.keys]

    def _load_numpy_state(self, parameters: list[np.ndarray]) -> None:
        state = {
            key: torch.tensor(value, device=self.device, dtype=torch.float32)
            for key, value in zip(self.keys, parameters, strict=True)
        }
        set_peft_model_state_dict(self.model, state)

    def _log_client_event(self, metrics: dict[str, Any]) -> None:
        path = self.output_dir / "communication_log.jsonl"
        event = {
            "client_id": f"client_{self.client_id}",
            "metrics": metrics,
            "uplink_bytes": int(metrics["uplink_bytes"]),
            "downlink_bytes": int(metrics["downlink_bytes"]),
            "num_bytes": int(metrics["total_bytes"]),
            "timestamp": time.time(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _log_eval_event(self, metrics: dict[str, float]) -> None:
        path = self.output_dir / "client_eval_log.jsonl"
        event = {
            "client_id": f"client_{self.client_id}",
            "metrics": metrics,
            "timestamp": time.time(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


class FedProxTrainer(Trainer):
    """Trainer adding the FedProx proximal penalty to trainable LoRA weights."""

    def __init__(self, *args, global_parameters: dict[str, torch.Tensor], fedprox_mu: float, **kwargs):
        super().__init__(*args, **kwargs)
        if fedprox_mu < 0 or not np.isfinite(fedprox_mu):
            raise ValueError("fedprox_mu must be finite and non-negative")
        self.global_parameters = global_parameters
        self.fedprox_mu = fedprox_mu

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        if return_outputs:
            loss, outputs = result
        else:
            loss = result
            outputs = None
        proximal = torch.zeros((), device=loss.device, dtype=loss.dtype)
        for name, parameter in model.named_parameters():
            reference = self.global_parameters.get(name)
            if reference is not None and parameter.requires_grad:
                proximal = proximal + torch.sum((parameter - reference.to(parameter.device)) ** 2)
        loss = loss + 0.5 * self.fedprox_mu * proximal
        return (loss, outputs) if return_outputs else loss


class CheckpointingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, output_dir: Path, num_clients: int) -> None:
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            on_fit_config_fn=lambda server_round: {"server_round": server_round},
            on_evaluate_config_fn=lambda server_round: {"server_round": server_round},
            evaluate_metrics_aggregation_fn=weighted_eval_metrics,
        )
        self.output_dir = output_dir

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        record = {
            "round": server_round,
            "num_results": len(results),
            "num_failures": len(failures),
            "metrics": metrics,
            "timestamp": time.time(),
        }
        with (self.output_dir / "server_rounds.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if aggregated is not None:
            arrays = fl.common.parameters_to_ndarrays(aggregated)
            np.savez_compressed(
                self.output_dir / f"server_round_{server_round:04d}.npz",
                **{f"param_{index:04d}": value for index, value in enumerate(arrays)},
            )
        return aggregated, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        record = {
            "round": server_round,
            "eval_loss": aggregated_loss,
            "num_results": len(results),
            "num_failures": len(failures),
            "metrics": metrics,
            "timestamp": time.time(),
        }
        with (self.output_dir / "server_eval_rounds.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return aggregated_loss, metrics


class CheckpointingFLEN(ConflictAwareFlowerStrategy):
    """Target-conditioned Flower strategy with checkpoints and evaluation logs."""

    def __init__(
        self,
        *,
        output_dir: Path,
        num_clients: int,
        target_jurisdiction: str,
        calibration_manifest,
        probe_labels: tuple[int, ...],
        conflict_config: ConflictConfig,
    ) -> None:
        super().__init__(
            conflict_config=conflict_config,
            calibration_manifest=calibration_manifest,
            target_jurisdiction=target_jurisdiction,
            probe_labels=probe_labels,
            aggregation_log_path=output_dir / "aggregation_rounds.jsonl",
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            on_fit_config_fn=lambda server_round: {"server_round": server_round},
            on_evaluate_config_fn=lambda server_round: {"server_round": server_round},
            evaluate_metrics_aggregation_fn=weighted_eval_metrics,
        )
        self.output_dir = output_dir

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        record = {
            "round": server_round,
            "num_results": len(results),
            "num_failures": len(failures),
            "metrics": metrics,
            "timestamp": time.time(),
        }
        with (self.output_dir / "server_rounds.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if aggregated is not None:
            arrays = fl.common.parameters_to_ndarrays(aggregated)
            np.savez_compressed(
                self.output_dir / f"server_round_{server_round:04d}.npz",
                **{f"param_{index:04d}": value for index, value in enumerate(arrays)},
            )
        return aggregated, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        record = {
            "round": server_round,
            "eval_loss": aggregated_loss,
            "num_results": len(results),
            "num_failures": len(failures),
            "metrics": metrics,
            "timestamp": time.time(),
        }
        with (self.output_dir / "server_eval_rounds.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return aggregated_loss, metrics


def write_report(args: argparse.Namespace, output_dir: Path) -> None:
    communication = read_jsonl(output_dir / "communication_log.jsonl")
    client_eval = read_jsonl(output_dir / "client_eval_log.jsonl")
    rounds = read_jsonl(output_dir / "server_rounds.jsonl")
    eval_rounds = read_jsonl(output_dir / "server_eval_rounds.jsonl")
    total_bytes = sum(int(row.get("num_bytes", 0)) for row in communication)
    uplink_bytes = sum(int(row.get("uplink_bytes", 0)) for row in communication)
    downlink_bytes = sum(int(row.get("downlink_bytes", 0)) for row in communication)
    losses = [
        row["metrics"]["train_loss"]
        for row in communication
        if "metrics" in row and "train_loss" in row["metrics"]
    ]
    drifts = [
        row["metrics"]["client_drift_l2"]
        for row in communication
        if "metrics" in row and "client_drift_l2" in row["metrics"]
    ]
    accuracies = [
        row["metrics"]["accuracy"]
        for row in eval_rounds
        if "metrics" in row and "accuracy" in row["metrics"]
    ]
    report = {
        "run_name": args.run_name,
        "method": args.method,
        "model_name": args.model_name,
        "dataset": f"{args.dataset_name}/{args.dataset_config}",
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "seed": args.seed,
        "partition_strategy": args.partition_strategy,
        "dirichlet_alpha": args.dirichlet_alpha,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_steps": args.max_steps,
        "max_probe_samples": args.max_probe_samples,
        "local_personalization_steps": args.local_personalization_steps,
        "communication_events": len(communication),
        "communication_bytes": total_bytes,
        "uplink_bytes": uplink_bytes,
        "downlink_bytes": downlink_bytes,
        "mean_train_loss": sum(losses) / len(losses) if losses else None,
        "mean_client_drift_l2": sum(drifts) / len(drifts) if drifts else None,
        "final_accuracy": accuracies[-1] if accuracies else None,
        "best_accuracy": max(accuracies) if accuracies else None,
        "server_rounds": len(rounds),
        "eval_rounds": len(eval_rounds),
        "real_training": True,
    }
    if args.method in GUARDED_LOCAL_METHODS:
        gate_values = [
            float(row["metrics"]["local_gate_accepted"])
            for row in communication
            if "local_gate_accepted" in row.get("metrics", {})
        ]
        personalization_gains = [
            float(row["metrics"]["personalization_accuracy_gain"])
            for row in client_eval
            if "personalization_accuracy_gain" in row.get("metrics", {})
        ]
        report.update(
            {
                "local_gate_events": len(gate_values),
                "local_gate_acceptance_rate": (
                    sum(gate_values) / len(gate_values) if gate_values else None
                ),
                "mean_personalization_accuracy_gain": (
                    sum(personalization_gains) / len(personalization_gains)
                    if personalization_gains
                    else None
                ),
            }
        )
    calibration_path = output_dir / "conflict_calibration_manifest.json"
    if calibration_path.is_file():
        calibration = load_calibration_manifest(calibration_path)
        report["calibration_id"] = calibration.calibration_id
        report["calibration_evidence_level"] = calibration.evidence_level
        report["formal_legal_claims_allowed"] = calibration.formal_legal_claims_allowed
        report["aggregation_rounds"] = len(read_jsonl(output_dir / "aggregation_rounds.jsonl"))
        report["local_residual_uploaded"] = False
    write_json(output_dir / "real_experiment_report.json", report)
    lines = [
        f"# Real Federated PEFT Experiment: {args.run_name}",
        "",
        f"- Model: `{args.model_name}`",
        f"- Method: `{args.method}`",
        f"- Dataset: `{args.dataset_name}/{args.dataset_config}`",
        f"- Clients: `{args.num_clients}`",
        f"- Rounds: `{args.num_rounds}`",
        f"- Seed: `{args.seed}`",
        f"- Partition: `{args.partition_strategy}`",
        f"- Communication bytes: `{total_bytes}`",
        f"- Mean train loss: `{report['mean_train_loss']}`",
        f"- Final CaseHOLD accuracy: `{report['final_accuracy']}`",
        f"- Mean client drift L2: `{report['mean_client_drift_l2']}`",
        "",
        "This run used real HuggingFace dataset loading, real PEFT LoRA training, and real Flower client/server communication.",
    ]
    (output_dir / "real_experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_legal_dataset(name: str, config: str, split: str) -> Dataset:
    try:
        return load_dataset(name, config, split=split)
    except Exception:
        return load_dataset("coastalcph/lex_glue", config, split=split)


def resolve_model_path(model_name: str, output_dir: Path) -> str:
    manifest_path = output_dir / "model_cache_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot_path = Path(str(manifest.get("snapshot_path", "")))
        if snapshot_path.exists():
            return str(snapshot_path)
    return model_name


def format_casehold(row: dict[str, Any], index: int, num_clients: int) -> dict[str, Any]:
    raw_endings = row.get("endings")
    if isinstance(raw_endings, list):
        endings = [str(ending) for ending in raw_endings]
    else:
        endings = [str(row.get(f"ending_{i}", "")) for i in range(5)]
    context = str(row.get("context", row.get("prompt", "")))
    label = row.get("label", row.get("answer", ""))
    answer = endings[int(label)] if str(label).isdigit() and int(label) < len(endings) else str(label)
    citations = extract_citations(context)
    jurisdiction = infer_jurisdiction(context, citations, int(label) if str(label).isdigit() else index, num_clients)
    text = (
        "Legal case context:\n"
        f"{context}\n\n"
        "Candidate holdings:\n"
        + "\n".join(f"{i}. {ending}" for i, ending in enumerate(endings) if ending)
        + f"\n\nCorrect legal holding:\n{answer}"
    )
    return {
        "example_id": f"casehold-{index}",
        "text": text,
        "context": context,
        "endings": endings,
        "answer_text": answer,
        "label": str(label),
        "citations": citations,
        "jurisdiction": jurisdiction,
    }


def casehold_prompt(record: dict[str, Any]) -> str:
    endings = record.get("endings", [])
    return (
        "Legal case context:\n"
        f"{record.get('context', '')}\n\n"
        "Candidate holdings:\n"
        + "\n".join(f"{i}. {ending}" for i, ending in enumerate(endings) if ending)
        + "\n\nCorrect legal holding:\n"
    )


def extract_citations(text: str) -> list[str]:
    patterns = [
        r"\b\d+\s+U\.S\.\s+\d+\b",
        r"\b\d+\s+F\.(?:2d|3d|4th|Supp\.?\s?\d*)\s+\d+\b",
        r"\b\d+\s+S\.Ct\.\s+\d+\b",
        r"\b\d+\s+[A-Z][A-Za-z.]*\s?(?:2d|3d)?\s+\d+\b",
    ]
    citations: list[str] = []
    for pattern in patterns:
        citations.extend(re.findall(pattern, text))
    return sorted(set(citations))


def infer_jurisdiction(text: str, citations: list[str], label_or_index: int, num_clients: int) -> str:
    lowered = text.lower()
    circuit = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s+cir(?:cuit|\.)\b", lowered)
    if circuit:
        return f"federal_circuit_{circuit.group(1)}"
    if "supreme court" in lowered or any("U.S." in citation for citation in citations):
        return "us_supreme_court"
    if re.search(r"\bf\.(?:2d|3d|4th|supp)", text, flags=re.IGNORECASE):
        return "us_federal"
    states = {
        "california": "state_california",
        "new york": "state_new_york",
        "texas": "state_texas",
        "florida": "state_florida",
        "illinois": "state_illinois",
        "massachusetts": "state_massachusetts",
    }
    for name, tag in states.items():
        if name in lowered:
            return tag
    text_group = stable_int(text) % max(num_clients * 2, 8)
    return f"pseudo_jurisdiction_{label_or_index % 5}_{text_group}"


def partition_records(
    records: list[dict[str, Any]],
    num_clients: int,
    strategy: str,
    seed: int,
    dirichlet_alpha: float,
) -> list[list[dict[str, Any]]]:
    if strategy == "iid":
        return [[row for index, row in enumerate(records) if index % num_clients == client_id] for client_id in range(num_clients)]
    if strategy == "label_dirichlet":
        return label_dirichlet_partition(records, num_clients, seed, dirichlet_alpha)
    return jurisdiction_partition(records, num_clients)


def label_dirichlet_partition(
    records: list[dict[str, Any]],
    num_clients: int,
    seed: int,
    alpha: float,
) -> list[list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_label[str(row.get("label", ""))].append(row)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    for label_records in by_label.values():
        rng.shuffle(label_records)
        proportions = rng.dirichlet([alpha] * num_clients)
        counts = rng.multinomial(len(label_records), proportions)
        start = 0
        for client_id, count in enumerate(counts):
            shards[client_id].extend(label_records[start : start + count])
            start += count
    return rebalance_empty_shards(shards)


def jurisdiction_partition(records: list[dict[str, Any]], num_clients: int) -> list[list[dict[str, Any]]]:
    by_jurisdiction: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_jurisdiction[str(row.get("jurisdiction", "unknown"))].append(row)
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_clients)]
    for jurisdiction, group in sorted(by_jurisdiction.items()):
        client_id = stable_int(jurisdiction) % num_clients
        shards[client_id].extend(group)
    return rebalance_empty_shards(shards)


def rebalance_empty_shards(shards: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    for client_id, shard in enumerate(shards):
        if shard:
            continue
        donor_id = max(range(len(shards)), key=lambda index: len(shards[index]))
        if len(shards[donor_id]) <= 1:
            continue
        midpoint = len(shards[donor_id]) // 2
        shards[client_id].extend(shards[donor_id][midpoint:])
        del shards[donor_id][midpoint:]
    return shards


def summarize_partition(shards: list[list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for client_id, shard in enumerate(shards):
        labels: dict[str, int] = defaultdict(int)
        jurisdictions: dict[str, int] = defaultdict(int)
        for row in shard:
            labels[str(row.get("label", ""))] += 1
            jurisdictions[str(row.get("jurisdiction", ""))] += 1
        summary[f"client_{client_id}"] = {
            "num_examples": len(shard),
            "labels": dict(sorted(labels.items())),
            "jurisdictions": dict(sorted(jurisdictions.items(), key=lambda item: (-item[1], item[0]))[:10]),
        }
    return summary


def weighted_eval_metrics(metrics: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    total = sum(num_examples for num_examples, _ in metrics)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0.0, "num_examples": 0.0}
    correct = sum(values.get("correct", 0.0) for _, values in metrics)
    eval_loss = sum(num_examples * values.get("eval_loss", 0.0) for num_examples, values in metrics) / total
    return {
        "accuracy": correct / total,
        "correct": correct,
        "num_examples": float(total),
        "eval_loss": eval_loss,
    }


def compute_client_drift(before: list[np.ndarray], after: list[np.ndarray]) -> tuple[float, float]:
    update_sq = 0.0
    base_sq = 0.0
    for old, new in zip(before, after, strict=True):
        delta = new.astype(np.float64) - old.astype(np.float64)
        update_sq += float(np.sum(delta * delta))
        base_sq += float(np.sum(old.astype(np.float64) * old.astype(np.float64)))
    drift = float(np.sqrt(update_sq))
    base = float(np.sqrt(base_sq))
    return drift, drift / (base + 1e-12)


def _arrays_l2(arrays: list[np.ndarray]) -> float:
    squared = sum(float(np.sum(array.astype(np.float64) ** 2)) for array in arrays)
    return float(np.sqrt(squared))


def stable_int(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value))


def resolved_experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_name": args.run_name,
        "method": args.method,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_probe_samples": args.max_probe_samples,
        "max_steps": args.max_steps,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "partition_strategy": args.partition_strategy,
        "dirichlet_alpha": args.dirichlet_alpha,
        "seed": args.seed,
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "learning_rate": 2e-4,
        "fedprox_mu": args.fedprox_mu if args.method == "fedprox" else None,
        "conflict": {
            "enabled": args.method in CONFLICT_METHODS,
            "penalty_lambda": (
                0.0 if args.method in ZERO_CONFLICT_METHODS else args.conflict_lambda
            ),
            "diversity_floor": args.diversity_floor,
            "weight_smoothing": args.weight_smoothing,
            "sample_count_cap": args.sample_count_cap,
            "calibration_manifest": "conflict_calibration_manifest.json",
            "require_expert_calibration": args.require_expert_calibration,
        },
        "local_personalization_steps": args.local_personalization_steps,
        "privacy": {
            "differential_privacy": False,
            "secure_aggregation": False,
        },
        "multi_agent_reasoning": False,
        "conflict_aware_aggregation": args.method in CONFLICT_METHODS,
    }


def environment_metadata() -> dict[str, Any]:
    packages = {}
    for package in ["accelerate", "datasets", "flwr", "numpy", "peft", "torch", "transformers"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
