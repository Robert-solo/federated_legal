from __future__ import annotations

import torch
import numpy as np
import pytest

from scripts.run_multieurlex_federated import (
    aggregate,
    aggregate_proxy_flen,
    clip_client_states,
    calibration_metrics,
    gaussian_epsilon,
    load_flen_manifest,
    multilabel_metrics,
    proximal_penalty,
    replace_one_client_sensitivity,
    THRESHOLD_GRID,
    threshold_search,
    trainable_parameter_snapshot,
)


def test_fedavg_aggregate_is_sample_weighted() -> None:
    states = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]
    result = aggregate(states, [1, 3], "fedavg")
    assert torch.allclose(result["a"], torch.tensor([2.5]))


def test_proxy_flen_aggregate_is_explicitly_diagnostic() -> None:
    states = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]
    result, audit = aggregate_proxy_flen(
        states,
        [1, 1],
        ["en-de", "en-fr"],
        [0.1, 0.4],
        "en-pl",
        0.2,
    )
    assert torch.isfinite(result["a"]).all()
    assert audit["calibration_level"] == "automatic_prediction_disagreement"
    assert audit["formal_legal_claims_allowed"] is False
    assert audit["raw_prediction_disagreement"] == pytest.approx(
        {"en-de": 0.1, "en-fr": 0.4}
    )
    assert audit["conflict_exposures"] == pytest.approx(
        {"en-de": 0.0, "en-fr": 1.0}
    )
    assert audit["weights"]["en-de"] > audit["weights"]["en-fr"]


def test_proxy_flen_equal_exposures_preserve_equal_weights() -> None:
    states = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]
    _, audit = aggregate_proxy_flen(
        states,
        [1, 1],
        ["en-de", "en-fr"],
        [0.2, 0.2],
        "en-pl",
        0.2,
    )
    assert audit["conflict_exposures"] == {"en-de": 0.0, "en-fr": 0.0}
    assert audit["weights"] == pytest.approx({"en-de": 0.5, "en-fr": 0.5})


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


def test_threshold_grid_covers_probabilities_below_point_one() -> None:
    assert THRESHOLD_GRID[0] == 0.001
    assert 0.005 in THRESHOLD_GRID
    assert 0.1 in THRESHOLD_GRID
    assert THRESHOLD_GRID[-1] == 0.9


def test_threshold_search_uses_micro_f1_by_default() -> None:
    probabilities = np.zeros((2, 21), dtype=np.float32)
    labels = np.zeros((2, 21), dtype=np.float32)
    probabilities[0, 0] = 0.8
    labels[0, 0] = 1.0
    selected, scores = threshold_search({"source": (probabilities, labels)})
    assert 0.0 < selected < 0.8
    assert {"threshold", "micro_f1", "macro_f1"} == set(scores[0])


def test_model_wrapper_exposes_backbone_config() -> None:
    from scripts.run_multieurlex_federated import PeftMultilabelModel

    backbone = torch.nn.Module()
    backbone.config = type("Config", (), {"hidden_size": 4})()
    model = PeftMultilabelModel(backbone, 4)
    assert model.config.hidden_size == 4


def test_client_update_clipping_bounds_norm() -> None:
    reference = {"a": torch.tensor([0.0, 0.0])}
    states = [{"a": torch.tensor([3.0, 4.0])}]
    clipped, norms, scales = clip_client_states(reference, states, 1.0)
    assert norms == pytest.approx([5.0])
    assert scales == pytest.approx([0.2])
    assert float(clipped[0]["a"].norm()) == pytest.approx(1.0)


def test_gaussian_accountant_improves_with_more_noise() -> None:
    epsilon_small_noise = gaussian_epsilon(0.5, rounds=3, delta=1e-5)
    epsilon_large_noise = gaussian_epsilon(1.0, rounds=3, delta=1e-5)
    assert epsilon_small_noise is not None
    assert epsilon_large_noise is not None
    assert epsilon_large_noise < epsilon_small_noise


def test_replace_one_client_sensitivity_includes_both_neighbor_updates() -> None:
    assert replace_one_client_sensitivity(1.0, [0.2, 0.3, 0.5]) == pytest.approx(1.0)


def test_calibration_detects_overconfident_wrong_predictions():
    labels = np.asarray([[1., 0.], [0., 1.]])
    good = calibration_metrics(labels, labels)
    bad = calibration_metrics(1-labels, labels)
    assert good == {"brier": 0., "ece_15": 0.}
    assert bad == {"brier": 1., "ece_15": 1.}
