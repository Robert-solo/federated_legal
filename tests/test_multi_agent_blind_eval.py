from scripts.run_multi_agent_blind_eval import blind_pair, format_review_output, parse_decision


def test_parse_decision_prefers_json() -> None:
    label, rationale = parse_decision('analysis {"label": 3, "rationale": "supported"}')
    assert label == 3
    assert rationale == "supported"


def test_blind_pair_is_deterministic_and_complete() -> None:
    first_pair, first_key = blind_pair("case-1", "single", "multi", 41)
    second_pair, second_key = blind_pair("case-1", "single", "multi", 41)
    assert first_pair == second_pair
    assert first_key == second_key
    assert set(first_key.values()) == {"single_agent", "typed_multi_agent"}


def test_parse_decision_accepts_direct_option_output() -> None:
    assert parse_decision("0. holding that the evidence was sufficient")[0] == 0
    assert parse_decision("The correct answer is:\n\n1. holding that the stay did not apply")[0] == 1
    assert parse_decision("4")[0] == 4
    assert parse_decision('```json\n{\n  "label": 2,\n  "rationale": "truncated"')[0] == 2


def test_review_output_uses_one_condition_neutral_format() -> None:
    rendered = format_review_output(1, ["first", "second"], "Because the record supports it.")
    assert rendered == (
        "Selected holding 1: second\nReasoning: Because the record supports it."
    )
