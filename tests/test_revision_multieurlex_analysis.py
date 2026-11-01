from scripts.analyze_revision_multieurlex import aggregate_holdouts, METRICS


def test_worst_language_is_minimum_within_each_seed():
    rows=[]
    for seed,values in [(41,[.1,.4]),(42,[.5,.2])]:
        for holdout,value in zip(["de","fr"],values):
            rows.append({"method":"fedavg","holdout":holdout,"seed":seed,
                         **{key:value for key in METRICS}})
    result=aggregate_holdouts(rows,["fedavg"],["de","fr"],[41,42,43])
    assert len(result)==2
    assert [r["worst_holdout_micro_f1"] for r in result]==[.1,.2]
    assert result[0]["micro_f1"]==.25
