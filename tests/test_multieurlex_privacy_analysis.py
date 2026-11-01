from __future__ import annotations

from scripts.analyze_multieurlex_privacy import analyze


def test_privacy_analysis_requires_complete_condition_seed_matrix() -> None:
    runs = []
    for condition in ("no_privacy", "gaussian_0_5"):
        for seed in (41, 42):
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "acceptance_status": "passed",
                    "micro_f1": 0.3,
                    "macro_f1": 0.2,
                    "loss": 0.7,
                    "attack_auc": 0.5,
                    "attack_balanced_accuracy": 0.5,
                }
            )
    result = analyze(runs, ["no_privacy", "gaussian_0_5"], [41, 42])
    assert result["matrix_complete"] is True
    assert result["acceptance_matrix_complete"] is True
    assert result["completed_run_count"] == 4
    assert result["accepted_run_count"] == 4
