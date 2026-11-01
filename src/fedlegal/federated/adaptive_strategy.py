"""Flower FedAdam with the same experiment artifacts as the shared baselines."""
from __future__ import annotations

import json
import time
from pathlib import Path

import flwr as fl
import numpy as np


class CheckpointingFedAdam(fl.server.strategy.FedAdam):
    def __init__(self, output_dir: Path, num_clients: int, initial_parameters,
                 evaluate_metrics_aggregation_fn) -> None:
        super().__init__(
            min_fit_clients=num_clients, min_evaluate_clients=num_clients,
            min_available_clients=num_clients, initial_parameters=initial_parameters,
            eta=0.01, eta_l=2e-4, beta_1=0.9, beta_2=0.99, tau=1e-9,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.output_dir = output_dir

    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        record = {"round":server_round,"num_results":len(results),"num_failures":len(failures),
                  "metrics":metrics,"timestamp":time.time(),"aggregation_method":"flower_fedadam"}
        with (self.output_dir/"server_rounds.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(record)+"\n")
        if parameters is not None:
            np.savez_compressed(self.output_dir/f"server_round_{server_round:04d}.npz",
                                **{f"param_{i:04d}":a for i,a in enumerate(fl.common.parameters_to_ndarrays(parameters))})
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        record = {"round":server_round,"eval_loss":loss,"num_results":len(results),
                  "num_failures":len(failures),"metrics":metrics,"timestamp":time.time()}
        with (self.output_dir/"server_eval_rounds.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(record)+"\n")
        return loss, metrics
