from __future__ import annotations

import json

import numpy as np

from scripts.run_membership_inference_attack import (
    balanced_accuracy,
    load_features,
    rank_auc,
    select_threshold,
)


def test_rank_auc_separates_members() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert rank_auc(labels, scores) == 1.0


def test_attack_threshold_uses_calibration_scores() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold = select_threshold(labels, scores)
    assert balanced_accuracy(labels, scores >= threshold) == 1.0


def test_feature_loader_requires_both_classes(tmp_path) -> None:
    path = tmp_path / "features.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "language": "en-de", "member": True, "loss": 0.1}),
                json.dumps({"id": "b", "language": "en-de", "member": False, "loss": 0.9}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(load_features(path)) == 2
