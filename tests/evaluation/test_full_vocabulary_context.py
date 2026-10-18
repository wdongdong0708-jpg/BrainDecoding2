"""Full-lexicon comparisons retain OOVs and pair material blocks across seeds."""
import numpy as np
import pytest

from braindecoding import full_vocabulary_context_test_v2 as worker


def test_full_lexicon_keeps_oov_references_as_errors_in_macro_and_micro():
    words = np.array(["a", "a", "a", "b", "outside"])
    queries = np.array([[1., 0.]] * 5)
    ranks, covered = worker.full_vocab_ranks(queries, words, np.eye(2), ["a", "b"])
    assert np.isinf(ranks[-1]) and np.array_equal(covered, [True]*4+[False])
    scores = worker.score_ranks(ranks, words, np.array(["s1"]*4+["s2"]), covered)
    assert scores["query_count"] == 5 and scores["target_count"] == 3 and scores["oov_queries"] == 1
    assert scores["macro_recall_at_1"] == pytest.approx(1/3)
    assert scores["micro_recall_at_1"] == .6
    assert scores["subject_equal_recall_at_1"] == .375
    assert scores["micro_recall_at_10"] == .8


def test_ranking_uses_existing_midrank_ties_and_rejects_invalid_candidates():
    ranks, _ = worker.full_vocab_ranks([[1., 0.]], ["a"], [[1., 0.], [1., 0.]], ["a", "b"])
    assert ranks.tolist() == [1.5]
    with pytest.raises(ValueError, match="candidate"):
        worker.full_vocab_ranks([[1., 0.]], ["a"], np.eye(2), ["a", "a"])
    with pytest.raises(ValueError, match="vectors"):
        worker.full_vocab_ranks([[np.nan, 0.]], ["a"], np.eye(2), ["a", "b"])


def test_query_masks_expand_candidates_without_changing_original_50_word_query_ids():
    words = np.array(["a", "outside", "b", "a"])
    common = np.array([True, True, False, False])
    masks = worker.population_masks(words, common, ["a", "b"])
    assert masks["all_words"].tolist() == [True]*4
    assert masks["common_words"].tolist() == common.tolist()
    assert masks["strict50_queries"].tolist() == [True, False, True, True]
    assert masks["strict50_common_queries"].tolist() == [True, False, False, False]


def test_paired_bootstrap_counts_shared_materials_instead_of_subject_seed_copies():
    info = dict(blocks=[dict(block="b1", scope="r"), dict(block="b2", scope="r")])
    ids = np.array([f"{subject}-{block}-{word}" for subject in ("s1", "s2")
                    for block in ("b1", "b2") for word in ("a", "b")])
    words = np.array([e.split("-")[-1] for e in ids])
    meta = {e: dict(block=e.split("-")[1]) for e in ids}
    ranks = np.array([[np.full(len(ids), 20.)]*3, [np.ones(len(ids))]*3])
    m = dict(random_seed=161803, repetitions=1000)
    observed, ci = worker.paired_intervals(info, ids, words, ranks, meta, m, 0)
    np.testing.assert_array_equal(observed, [[0.]*4, [1.]*4])
    assert ci["blocks"] == 2 and ci["bootstrap_samples"] == 1000
    np.testing.assert_array_equal(ci["ci_lower"], [1.]*4)
    np.testing.assert_array_equal(ci["ci_upper"], [1.]*4)
    assert ci["type_count_full_samples"] == 1000
