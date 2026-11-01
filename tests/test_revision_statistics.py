from scripts.analyze_revision_v2 import sign_flip_pvalue, summarize


def test_five_seed_two_sided_permutation_has_finite_resolution():
    assert sign_flip_pvalue([1,2,3,4,5])==.0625
    assert sign_flip_pvalue([0,0,0])==1.


def test_holm_uses_contrasts_of_same_metric_as_family():
    rows=[{"condition":c,"seed":s,"accuracy":v} for c,v in [("base",0),("one",1),("two",2)] for s in range(5)]
    result=summarize(rows,["base","one","two"],list(range(5)),["accuracy"],[("one","base"),("two","base")])
    assert result["matrix_complete"]
    assert all(r["holm_adjusted_pvalue"]==.125 for r in result["paired_contrasts"])
