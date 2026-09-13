from __future__ import annotations

import torch

from scripts.run_multieurlex_federated import aggregate


def test_fedavg_aggregate_is_sample_weighted() -> None:
    states = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]
    result = aggregate(states, [1, 3], "fedavg")
    assert torch.allclose(result["a"], torch.tensor([2.5]))


def test_flen_aggregate_is_finite() -> None:
    states = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([3.0])}]
    result = aggregate(states, [1, 1], "flen")
    assert torch.isfinite(result["a"]).all()
