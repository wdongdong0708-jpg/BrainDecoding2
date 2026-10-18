import copy

import pytest

from braindecoding.unified_sentence_v2_preview import dataset_rows, select_groups


def test_sampling_preserves_components_subjects_and_strata_without_outcomes():
    groups = []
    for partition in ("calibration", "evaluation"):
        for scope in ("chapter-01", "chapter-02", "chapter-03"):
            for block in range(3):
                for material in range(2):
                    for subject in ("s1", "s2"):
                        # One component contains two variants for s1. Both
                        # must be retained with its other subject's variant.
                        for voice in range(2 if subject == "s1" else 1):
                            groups.append(dict(partition=partition, scope=scope,
                                block=f"{partition}-{scope}-{block}",
                                material=f"{partition}-{scope}-{block}-{material}",
                                subject=subject, group=f"{partition}-{scope}-{block}-{material}-{subject}-{voice}",
                                reference=["untouched"], event_ids=[f"e-{partition}-{scope}-{block}-{material}-{subject}-{voice}"]))
    info = dict(groups=groups)
    selected = select_groups(info, "data", "evaluation", 6)
    assert len({g["block"] for g in selected}) == 6
    assert {g["scope"] for g in selected} == {"chapter-01", "chapter-02", "chapter-03"}
    assert {g["subject"] for g in selected} == {"s1", "s2"}
    assert len(selected) == 18
    for material in {g["material"] for g in selected}:
        assert {g["group"] for g in selected if g["material"] == material} == \
            {g["group"] for g in groups if g["material"] == material}
    changed = copy.deepcopy(info)
    for g in changed["groups"]:
        g["reference"] = ["different"] * 100
        g["outcome"] = 1e10
    assert [g["group"] for g in select_groups(changed, "data", "evaluation", 6)] == [g["group"] for g in selected]
    calibration = select_groups(info, "data", "calibration", 5)
    assert not ({g["material"] for g in calibration} & {g["material"] for g in selected})


def test_preview_rejects_partial_population_and_counts_own_initial_repairs():
    group = dict(group="g", subject="s", material="m", block="b", scope="r",
                 reference=["x", "y", "z"])
    frozen = dict(preview_sha256="preview", datasets={"data":dict(language="fr", groups=dict(evaluation=[group]))})
    result = dict(final=dict(A=["bad", "y", "z"], B=["bad", "y", "z"], C=["x", "bad", "z"]),
        trace=[dict(arrival=0, A=["bad"], B=["bad"], C=["bad"]),
               dict(arrival=1, A=["bad", "y"], B=["bad", "y"], C=["x", "y"]),
               dict(arrival=2, A=["bad", "y", "z"], B=["bad", "y", "z"], C=["x", "bad", "z"])])
    decoded = dict(preview_sha256="preview", test_evaluated=False, records=[dict(group="g",result=result)])
    _, repairs, _ = dataset_rows(frozen, "data", decoded)
    row = next(r for r in repairs if r["condition"] == "C")
    assert row["eligible"] == 2 and row["repaired"] == 1 and row["harmed"] == 1
    missing = copy.deepcopy(decoded)
    missing["records"] = []
    with pytest.raises(ValueError, match="population incomplete"):
        dataset_rows(frozen, "data", missing)
    duplicate = copy.deepcopy(decoded)
    duplicate["records"].append(copy.deepcopy(duplicate["records"][0]))
    with pytest.raises(ValueError, match="population incomplete"):
        dataset_rows(frozen, "data", duplicate)
