import pytest

from braindecoding import unified_sentence_v2_revision_gain_full as full
from braindecoding.unified_sentence_v2_revision_gain_full_results import aggregate_seeds


def result():
    return dict(final=dict(A=["x","bad","z"],C=["x","y","z"]),
                trace=[dict(arrival=0,A=["x"],C=["x"]),
                       dict(arrival=1,A=["x","bad"],C=["x","y"]),
                       dict(arrival=2,A=["x","bad","z"],C=["x","y","z"])])


def test_resume_rejects_other_conditions_and_other_materials():
    m=dict(manifest_sha256="frozen",datasets=dict(data=dict(selected_weight=.1)))
    payload=dict(manifest_sha256="frozen",dataset="data",seed=0,conditions=["A","C"],
                 branch="open",alignment="real",selected_weight=.1,kind="decoded",test_evaluated=False,
                 records=[dict(group="g",result=result())])
    full.validate_chunk(payload,m,"data",0,["g"],"decoded")
    with pytest.raises(ValueError,match="population"):
        full.validate_chunk(payload,m,"data",0,["different"],"decoded")
    for key,value in (("alignment","shift-00"),("branch","closed"),("conditions",["A","B","C"]),
                      ("seed",1),("selected_weight",.5),("manifest_sha256","changed")):
        with pytest.raises(ValueError,match="provenance"):
            full.validate_chunk({**payload,key:value},m,"data",0,["g"],"decoded")


def test_full_aggregation_gives_subjects_and_seeds_equal_weight():
    rows,changes=[],[]
    for seed,subject,values in ((0,"s1",[1,1]),(0,"s2",[0]),(1,"s1",[0]),(1,"s2",[1,1,1])):
        for value in values:
            for c in ("A","C"):
                score=value if c=="C" else 0
                rows.append(dict(seed=seed,subject=subject,condition=c,words=1,
                    sentence_bleu1_mean=score,wer=1-score,bleu1=score,aligned_word_accuracy=score))
            changes.append(dict(seed=seed,subject=subject,words=1,eligible=1,paired_repaired=value,
                                paired_harmed=0,rolling_repaired=value,rolling_harmed=0))
    output=aggregate_seeds(rows,changes)
    assert output["summary"][1]["sentence_bleu1_mean"]==.5
    assert output["summary"][1]["aligned_word_accuracy"]==.5
    assert output["revisions"]["paired_repaired_rate"]==.5
    assert output["revisions"]["words"]==7
    assert output["revisions"]["count_unit"]=="word_position_seed_decisions"
    with pytest.raises(ValueError,match="Subjects differ"):
        aggregate_seeds([r for r in rows if not(r["seed"]==1 and r["subject"]=="s2")],changes)


def test_full_entry_dispatches_to_AC_without_invoking_archived_runner(monkeypatch):
    from braindecoding import unified_sentence_v2, unified_sentence_v2_complete
    called=[]
    def forbidden():
        raise AssertionError("Archived broad runner must stay inactive")
    monkeypatch.setattr(unified_sentence_v2,"run",forbidden)
    monkeypatch.setattr(full,"main",lambda argv:called.append(argv))
    unified_sentence_v2_complete.main(["--full"])
    assert called==[[]]


def test_reuse_checks_real_alignment_and_conflicting_overlap(tmp_path,monkeypatch):
    from braindecoding.unified_sentence_v2 import save
    monkeypatch.setattr(full,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(full,"PARENT_OUT",tmp_path/"parent")
    monkeypatch.setattr(full,"PREVIEW_OUT",tmp_path/"preview")
    group=dict(group="g",subject="s",partition="evaluation",reference=["x","y","z"])
    parent=dict(manifest_sha256="parent",datasets=dict(data=dict(groups=[group])))
    preview=dict(preview_sha256="preview",datasets=dict(data=dict(groups=dict(evaluation=[group]))))
    parent_path=tmp_path/"parent/data/seed-000/open/real/chunk-0000.json"
    save(tmp_path/"parent/data/calibration.json",dict(manifest_sha256="parent",selected_weight=.1))
    payload=dict(manifest_sha256="parent",label="evaluation_open_real_seed-0",test_evaluated=False,
                 records=[dict(group="g",result=result())])
    save(parent_path,payload)
    preview_path=tmp_path/"preview/data/main.json"
    cached=dict(status="completed",seed=0,selected_weight=.1,preview_sha256="preview",
                parent_manifest_sha256="parent",test_evaluated=False,records=payload["records"])
    save(preview_path,cached)
    reused,sources=full.collect_reuse(parent,preview,"data",0,.1)
    assert len(reused)==1 and len(sources)==3
    cached["records"][0]["result"]["final"]["A"][1]="other"
    cached["records"][0]["result"]["trace"][1]["A"][1]="other"
    cached["records"][0]["result"]["trace"][2]["A"][1]="other"
    save(preview_path,cached)
    with pytest.raises(ValueError,match="disagree"):
        full.collect_reuse(parent,preview,"data",0,.1)
    payload["label"]="evaluation_open_shift-00_seed-0"
    save(parent_path,payload)
    with pytest.raises(ValueError,match="provenance"):
        full.collect_reuse(parent,preview,"data",0,.1)
