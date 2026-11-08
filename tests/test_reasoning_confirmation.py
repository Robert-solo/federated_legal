from scripts.prepare_reasoning_confirmation import select_cases


def test_confirmation_cases_are_disjoint_from_both_earlier_test_uses():
    rows=[{"id":f"row-{index:04d}"} for index in range(2000)]
    selected,excluded=select_cases(rows)
    ids={row["id"] for row in selected}
    assert len(ids)==200
    assert ids.isdisjoint(excluded)
    assert selected==select_cases(rows)[0]
