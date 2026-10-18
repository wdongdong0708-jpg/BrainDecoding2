import pytest

from braindecoding.unified_sentence_v2_revision_gain import metric_rows, revision_summary, transitions


def test_final_comparison_and_online_repair_use_their_own_initial_outputs():
    group=dict(subject="s",group="g",reference=["x","y","z"])
    result=dict(final=dict(A=["x","y","bad"],C=["x","y","z"]),trace=[dict(arrival=0,C=["bad"]),dict(arrival=1,C=["x","bad"]),dict(arrival=2,C=["x","y","z"])])
    changes=transitions(group,result)
    assert changes["paired_repaired"]==1 and changes["paired_harmed"]==0
    assert changes["rolling_repaired"]==2 and changes["rolling_harmed"]==0
    assert changes["words"]==3 and changes["eligible"]==2
    summary=revision_summary([changes])
    assert summary["paired_repaired_rate"]==1/3 and summary["rolling_repaired_rate"]==1
    assert summary["net_correct_words"]==1
    short=dict(final=dict(A=["bad"],C=["x"]),trace=[dict(arrival=0,C=["x"])])
    one=transitions(dict(subject="s",group="one",reference=["x"]),short)
    assert one["paired_repaired"]==1 and one["rolling_repaired"]==0 and one["eligible"]==0
    short["trace"]=[dict(arrival=1,C=["x"])]
    with pytest.raises(ValueError,match="arrival trace"):
        transitions(dict(subject="s",group="one",reference=["x"]),short)


def test_default_entry_uses_current_report_without_starting_broad_run(monkeypatch):
    from braindecoding import unified_sentence_v2, unified_sentence_v2_complete, unified_sentence_v2_revision_gain
    called=[]
    def forbidden():
        raise AssertionError("Historical broad run must not be the default")
    monkeypatch.setattr(unified_sentence_v2,"run",forbidden)
    monkeypatch.setattr(unified_sentence_v2_revision_gain,"main",lambda:called.append("revision_report"))
    unified_sentence_v2_complete.main()
    assert called==["revision_report"]


def test_two_condition_cache_needs_no_ablation_output_and_rejects_missing_population():
    group=dict(subject="s",group="g",reference=["x"],material="m",block="b",scope="r")
    preview=dict(preview_sha256="frozen",datasets=dict(data=dict(language="fr",groups=dict(evaluation=[group]))))
    result=dict(final=dict(A=["bad"],C=["x"]),trace=[dict(arrival=0,C=["x"])])
    decoded=dict(preview_sha256="frozen",test_evaluated=False,records=[dict(group="g",result=result)])
    rows=metric_rows(preview,"data",decoded)
    assert [r["condition"] for r in rows]==["A","C"]
    assert [r["aligned_word_accuracy"] for r in rows]==[0.,1.]
    decoded["records"]=[]
    with pytest.raises(ValueError,match="population incomplete"):
        metric_rows(preview,"data",decoded)
