from scripts.analyze_multi_agent_blind_eval import analyze, exact_mcnemar_pvalue


def test_mcnemar_is_one_when_no_discordant_pairs() -> None:
    assert exact_mcnemar_pvalue(0, 0) == 1.0


def test_paired_analysis_counts_outcomes() -> None:
    rows = [
        {"single_correct": True, "multi_correct": True},
        {"single_correct": True, "multi_correct": False},
        {"single_correct": False, "multi_correct": True},
        {"single_correct": False, "multi_correct": True},
    ]
    result = analyze(rows, repetitions=100)
    assert result["paired_accuracy_difference"] == 0.25
    assert result["single_only_correct"] == 1
    assert result["multi_only_correct"] == 2
