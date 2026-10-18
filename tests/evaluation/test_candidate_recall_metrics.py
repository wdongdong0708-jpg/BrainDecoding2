"""Checks for target accounting and subject-preserving misalignment controls."""

import numpy as np

from braindecoding.candidate_recall import (
    horizon_masks,
    recall_diagnostic,
    target_ranks,
    within_subject_permutation,
)


def test_last_available_horizon_and_common_targets():
    available, common, last = horizon_masks([[0, 1, 2], [3, 4]], 5)
    assert available[0].tolist() == [True] * 5
    assert available[1].tolist() == [True, True, False, True, False]
    assert available[2].tolist() == [True, False, False, False, False]
    assert common.tolist() == [True, False, False, False, False]
    assert last.tolist() == [2, 1, 0, 1, 0]


def test_recall_and_shuffle_use_same_subject_without_self_matches():
    vocabulary = tuple(f"w{i}" for i in range(6)) + ("dummy1", "dummy2")
    rows = np.asarray([[index, 6, 7] for index in range(6)], dtype=np.int32)
    ids = {delay: rows.copy() for delay in (0, 1, 2)}
    truths = vocabulary[:6]
    subjects = ["s1"] * 3 + ["s2"] * 3
    result = recall_diagnostic(
        ids, truths, vocabulary, truths[:3], [[0, 1, 2], [3, 4, 5]],
        subjects, ks=(1, 2, 3), shuffle_repetitions=5, shuffle_seed=7,
    )
    assert result["common_event_count"] == 2
    assert result["clean"]["final"]["all"]["hits_at_k"]["1"] == 6
    assert result["clean"]["final"]["seen_in_training"]["events"] == 3
    assert result["clean"]["final"]["unseen_but_in_inventory"]["events"] == 3
    assert result["shuffled_within_subject"]["profiles"]["final"]["all"]["recall_at_k_mean"]["3"] == 0.0
    permutation = within_subject_permutation(np.asarray(subjects), np.random.default_rng(11))
    assert np.all(permutation != np.arange(6))
    assert np.array_equal(np.asarray(subjects)[permutation], subjects)


def test_outside_inventory_target_has_zero_rank():
    assert target_ranks([[0, 1], [1, 0]], [0, -1]).tolist() == [1, 0]
