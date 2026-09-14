from __future__ import annotations

import torch
import numpy as np
import pytest

from scripts.run_multieurlex_federated import (
    aggregate,
    load_flen_manifest,
    multilabel_metrics,
    proximal_penalty,
    trainable_parameter_snapshot,
)


def test_fedavg_aggregate_is_sample_weighted() -> None:
    states = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]
    result = aggregate(states, [1, 3], "fedavg")
    assert torch.allclose(result["a"], torch.tensor([2.5]))


def test_proxy_flen_aggregate_is_explicitly_diagnostic() -> None:
    states = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]
    result = aggregate(states, [1, 1], "proxy_flen")
    assert torch.isfinite(result["a"]).all()


def test_lambda_zero_recovers_fedavg() -> None:
    states = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]
    assert torch.equal(
        aggregate(states, [1, 3], "lambda0")["a"],
        aggregate(states, [1, 3], "fedavg")["a"],
    )


def test_flen_rejects_nonexpert_manifest(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        '{"status":"passed","calibration_level":"automatic_proxy",'
        '"target_jurisdiction":"en-pl","clients":{"en-de":{}}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expert-adjudicated"):
        load_flen_manifest(path, "en-pl", ["en-de"])


def test_fedprox_penalty_uses_exact_trainable_parameter_names() -> None:
    model = torch.nn.Linear(2, 1)
    reference = trainable_parameter_snapshot(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    assert float(proximal_penalty(model, reference).detach()) == pytest.approx(2.0)


def test_multilabel_metrics_include_macro_f1() -> None:
    probabilities = np.zeros((2, 21), dtype=np.float32)
    labels = np.zeros((2, 21), dtype=np.float32)
    probabilities[:, 0] = [0.9, 0.1]
    labels[:, 0] = [1.0, 0.0]
    metrics = multilabel_metrics(probabilities, labels, 0.5)
    assert metrics["micro_f1"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0 / 21.0)


def test_model_wrapper_exposes_backbone_config() -> None:
    from scripts.run_multieurlex_federated import PeftMultilabelModel

    backbone = torch.nn.Module()
    backbone.config = type("Config", (), {"hidden_size": 4})()
    model = PeftMultilabelModel(backbone, 4)
    assert model.config.hidden_size == 4
