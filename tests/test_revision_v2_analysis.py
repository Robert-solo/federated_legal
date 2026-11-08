from scripts.analyze_revision_v2 import summarize, fixed_test_shard_accuracy
from scripts.analyze_expert_annotations import agreement


def test_paired_effect_uses_matching_seeds_and_reports_missing_run():
    rows = [
        {"condition":"guarded", "seed":41, "accuracy":.4},
        {"condition":"unguarded", "seed":41, "accuracy":.3},
        {"condition":"unguarded", "seed":42, "accuracy":.9},
    ]
    result = summarize(rows,["guarded","unguarded"],[41,42],["accuracy"],[("guarded","unguarded")])
    assert not result["matrix_complete"]
    assert result["missing_runs"] == [{"condition":"guarded","seed":42}]
    effect = result["paired_contrasts"][0]
    assert effect["paired_seeds"] == [41]
    assert abs(effect["mean"]-.1) < 1e-9
    assert result["completed_runs_excluded_for_poor_utility"] == 0


def test_authority_agreement_handles_single_category_without_artificial_kappa():
    first = {"a":{"routing_state":"transferable"},"b":{"routing_state":"transferable"}}
    result = agreement(first,first,"routing_state",["a","b"])
    assert result["raw_agreement"] == 1
    assert result["cohen_kappa"] is None


def test_worst_client_uses_fixed_test_assignment_across_early_shared_runs():
    rows=[{"example_id":f"casehold-{i}","client":0,"gold":0,
           "prediction":int(i in (0,4))} for i in range(8)]
    assert fixed_test_shard_accuracy(rows)==0
    for row in rows:
        row["client"]=int(row["example_id"].split("-")[-1])%4
    assert fixed_test_shard_accuracy(rows)==0
