"""Reblocking must preserve shared stimuli, strata, tails, and the point estimand."""
import copy

import numpy as np
import pytest

from braindecoding.material_block_length_sensitivity_test_v2 import material_order, reblock, interval_class
from braindecoding.unified_sentence_v2_results import paired_block_statistics


pytestmark = pytest.mark.current_protocol


def fixture_info():
    groups, blocks = [], []
    for scope, n in (("chapter-02", 7), ("chapter-09", 3)):
        for start in range(0, n, 5):
            block_id = f"d|{scope}|test-block-{start//5:03d}"
            block = dict(block=block_id, scope=scope, materials=[], groups=[])
            for i in range(start, min(start+5, n)):
                material = f"{scope}-m{i}"
                block["materials"].append(material)
                for subject, voice in (("s1", "f"), ("s2", "m")):
                    block["groups"].append(len(groups))
                    groups.append(dict(group=f"{scope}-{subject}-{i}", subject=subject, voice=voice,
                                       scope=scope, material=material, order=i, block=block_id, atoms=[f"{scope}-line{i}"]))
            blocks.append(block)
    return dict(groups=groups, blocks=blocks)


def test_five_material_blocks_match_original_and_preserve_connected_voices():
    info = fixture_info()
    mapping, _ = reblock("d", info, 5)
    for g in info["groups"]:
        assert mapping[g["group"]] == g["block"]
    for length in (1, 2, 3, 10):
        mapping, rows = reblock("d", info, length)
        assert len(mapping) == 20 and len(rows) == 10
        for i in range(0, len(info["groups"]), 2):
            assert mapping[info["groups"][i]["group"]] == mapping[info["groups"][i+1]["group"]]


def test_partial_tail_is_retained_without_crossing_chapter():
    info = fixture_info()
    mapping, rows = reblock("d", info, 3)
    assert len(set(mapping.values())) == 4  # ceil(7/3) + ceil(3/3)
    tails = [r for r in rows if r["scope"] == "chapter-02" and r["material_order"] == 6]
    assert len(tails) == 1 and tails[0]["block"].endswith("002")
    assert len(set(reblock("d", info, 20)[0].values())) == 2


def test_split_shared_atom_and_duplicate_or_omitted_membership_are_rejected():
    info = fixture_info()
    split = copy.deepcopy(info)
    split["groups"][2]["atoms"] = split["groups"][0]["atoms"]
    with pytest.raises(ValueError, match="shared stimulus"):
        material_order(split)
    duplicated = copy.deepcopy(info)
    duplicated["blocks"][0]["groups"].append(0)
    with pytest.raises(ValueError, match="multiple original blocks"):
        material_order(duplicated)
    omitted = copy.deepcopy(info)
    omitted["blocks"][0]["groups"].pop()
    with pytest.raises(ValueError, match="omit a group"):
        material_order(omitted)


def test_chronological_order_cannot_be_selected_by_result():
    info = fixture_info()
    info["blocks"][0]["materials"][:2] = info["blocks"][0]["materials"][1::-1]
    with pytest.raises(ValueError, match="chronological"):
        material_order(info)


def test_reblocking_keeps_subject_seed_equal_estimand_with_unequal_cell_counts():
    info = fixture_info()
    settings = dict(seed=161803, paired_material_block_sign_flips=100,
                    material_block_bootstrap_repetitions=100)
    for length in (1, 3, 5, 20):
        mapping, _ = reblock("d", info, length)
        rows = [dict(seed=seed, subject=g["subject"], block=mapping[g["group"]], scope=g["scope"],
                     difference=.1 if g["subject"] == "s1" else .3)
                for seed in range(3) for g in info["groups"]]
        # Repeat a group only inside one subject's cell, keeping its mean .1.
        rows.extend([dict(rows[0])] * 4)
        stats = paired_block_statistics(rows, "difference", settings)
        assert stats["subjects"] == 2 and stats["seeds"] == 3
        assert stats["effect"] == pytest.approx(.2)
        assert stats["bootstrap_samples"] == 100
        assert np.isfinite([stats["ci_lower"], stats["ci_upper"]]).all()


@pytest.mark.parametrize("lo,hi,expected", [(0., 1., "includes_zero"), (-1., 0., "includes_zero"),
                                          (.1, 1., "positive"), (-1., -.1, "negative")])
def test_interval_boundary_is_not_reported_as_excluding_zero(lo, hi, expected):
    assert interval_class(lo, hi) == expected
