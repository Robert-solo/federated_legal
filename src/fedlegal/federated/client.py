"""Flower client for local Legal LLM adapter training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fedlegal.federated.logging import CommunicationEvent, CommunicationLogger
from fedlegal.federated.payload import AdapterPayload
from fedlegal.training.local_trainer import LocalAdapterTrainer


@dataclass
class LegalFlowerClient:
    """Flower-compatible client boundary for local adapter training."""

    client_id: str
    jurisdiction: str
    trainer: LocalAdapterTrainer
    local_epochs: int = 1
    communication_logger: CommunicationLogger | None = None
    payload_type: str = "lora_adapters"

    @property
    def parameter_keys(self) -> list[str]:
        """Adapter parameter keys for ordered Flower payloads."""

        return self.trainer.parameter_keys

    def get_adapter_parameters(self) -> list[Any]:
        """Return PEFT adapter payloads for federated communication."""

        payload = self.trainer.export_payload()
        self._log(round_id=0, direction="client_to_server", payload=payload, metrics={})
        return payload.to_parameters()

    def fit_local_adapter(
        self,
        parameters: list[Any] | None = None,
        round_id: int = 1,
    ) -> tuple[list[Any], int, dict[str, float]]:
        """Run local adapter training for one federated round."""

        if parameters is not None:
            self.trainer.import_payload(AdapterPayload.from_parameters(self.parameter_keys, parameters))

        result = self.trainer.fit(round_id=round_id, local_epochs=self.local_epochs)
        self._log(
            round_id=round_id,
            direction="client_to_server",
            payload=result.payload,
            metrics=result.metrics,
        )
        return result.payload.to_parameters(), result.num_examples, result.metrics

    def evaluate_local_adapter(
        self,
        parameters: list[Any] | None = None,
    ) -> tuple[float, int, dict[str, float]]:
        """Evaluate local adapter state."""

        if parameters is not None:
            self.trainer.import_payload(AdapterPayload.from_parameters(self.parameter_keys, parameters))
        metrics = self.trainer.evaluate()
        return 0.0, int(metrics.get("num_examples", 0.0)), metrics

    def to_numpy_client(self):
        """Create a `flwr.client.NumPyClient` adapter when Flower is installed."""

        try:
            import flwr as fl
        except ImportError as exc:
            raise RuntimeError("Flower is required for runtime client execution.") from exc

        outer = self

        class _NumPyClient(fl.client.NumPyClient):
            def get_parameters(self, config: dict[str, Any]):
                return outer.get_adapter_parameters()

            def fit(self, parameters: list[Any], config: dict[str, Any]):
                round_id = int(config.get("server_round", config.get("round_id", 1)))
                return outer.fit_local_adapter(parameters=parameters, round_id=round_id)

            def evaluate(self, parameters: list[np.ndarray], config: dict[str, Any]):
                return outer.evaluate_local_adapter(parameters)

        return _NumPyClient()

    def _log(
        self,
        round_id: int,
        direction: str,
        payload: AdapterPayload,
        metrics: dict[str, Any],
    ) -> None:
        if self.communication_logger is None:
            return
        self.communication_logger.log(
            CommunicationEvent(
                round_id=round_id,
                client_id=self.client_id,
                direction=direction,
                payload_type=self.payload_type,
                num_tensors=len(payload.tensors),
                num_bytes=payload.nbytes(),
                metrics=metrics,
            )
        )
