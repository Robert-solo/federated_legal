from __future__ import annotations

from scripts.analyze_multieurlex_comparison import analyze


def test_analysis_requires_complete_matched_matrix() -> None:
    runs = []
    for method, value in (("fedavg", 0.3), ("fedprox", 0.32)):
        for seed in (41, 42, 43):
            runs.append(
                {
                    "method": method,
                    "holdout": "en-pl",
                    "seed": seed,
                    "micro_f1": value,
                    "macro_f1": value - 0.1,
                    "loss": 0.5,
                }
            )
    result = analyze(runs, ["fedavg", "fedprox"], ["en-pl"], [41, 42, 43])
    assert result["matrix_complete"] is True
    assert result["paired_comparisons"][0]["superiority_gate_passed"] is False
