from fedlegal.evaluation.legalbench_diagnostic import TASKS, prompt_for, classification_metrics


def test_citation_prompt_includes_candidate_but_not_gold():
    row={"text":"Legal statement.","citation":"Example v. State","label":"Gold should stay hidden"}
    prompt=prompt_for(TASKS["citation_prediction_classification"],row)
    assert "Example v. State" in prompt
    assert row["label"] not in prompt


def test_balanced_accuracy_exposes_constant_label_prediction():
    rows=[{"gold":"Yes","prediction":"Yes","probabilities":[.1,.9]} for _ in range(3)]
    rows.append({"gold":"No","prediction":"Yes","probabilities":[.1,.9]})
    result=classification_metrics(rows,("No","Yes"))
    assert result["accuracy"] == .75
    assert result["balanced_accuracy"] == .5
    assert abs(result["macro_f1"]-3/7) < 1e-10
