"""公共词表与检索评价边界的行为锁定测试。"""

import numpy as np
import pytest

from braindecoding.evaluation import retrieval, vocabulary


def _embeddings():
    target_words = ["alpha", "beta", "gamma", "alpha"]
    target_embeddings = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
        dtype=np.float32,
    )
    query_embeddings = np.asarray(
        [[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]],
        dtype=np.float32,
    )
    return query_embeddings, target_embeddings, target_words


def test_retrieval_ranks_and_ties_are_unchanged():
    query_embeddings = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    candidate_embeddings = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    candidate_ids = ["a", "b", "c"]
    target_ids = ["a", "c", "a"]

    actual = retrieval.retrieval_ranks(
        query_embeddings,
        target_ids,
        candidate_embeddings,
        candidate_ids,
        chunk_size=1,
    )
    np.testing.assert_array_equal(actual, [1.5, 1.0, 2.0])

    actual_summary = retrieval.summarize_retrieval(
        actual, target_ids, candidate_count=3, top_ks=(1, 2)
    )
    assert actual_summary["micro_recall_at_1"] == pytest.approx(1 / 3)
    assert actual_summary["macro_recall_at_1"] == pytest.approx(0.5)


def test_fixed_vocabulary_predictions_and_metrics_are_unchanged():
    query_embeddings, target_embeddings, target_words = _embeddings()
    candidates = ["beta", "alpha", "gamma"]

    actual_predictions = retrieval.fixed_vocabulary_top1_predictions(
        query_embeddings, target_embeddings, target_words, candidates
    )
    assert actual_predictions[:2] == (
        ["alpha", "beta", "gamma", "alpha"],
        ["alpha", "gamma", "gamma", "beta"],
    )
    np.testing.assert_array_equal(actual_predictions[2], [0, 1, 2, 3])

    actual_metrics = retrieval.fixed_vocabulary_retrieval_metrics(
        query_embeddings,
        target_embeddings,
        target_words,
        candidates,
        vocabulary_name="test3",
    )
    assert actual_metrics["retrieval_acc1_vocab=test3"] == 0.5
    assert actual_metrics["retrieval_acc10_vocab=test3"] == 1.0


def test_direct_candidate_scores_preserve_ranking_and_ties():
    query_embeddings, _, target_words = _embeddings()
    candidate_words = ["alpha", "beta", "gamma"]
    candidate_embeddings = np.eye(3, dtype=np.float32)
    normalized_queries = retrieval.normalize_rows(query_embeddings)
    scores = normalized_queries @ candidate_embeddings.T

    ranks = retrieval.score_matrix_ranks(scores, target_words, candidate_words)
    metrics = retrieval.fixed_vocabulary_score_metrics(
        scores,
        target_words,
        candidate_words,
        candidate_words,
        vocabulary_name="test3",
    )

    np.testing.assert_array_equal(ranks, [1.0, 2.5, 1.0, 2.5])
    assert metrics["retrieval_acc1_vocab=test3"] == 0.5
    assert metrics["retrieval_acc10_vocab=test3"] == 1.0


def test_frequency_vocabulary_is_deterministic_for_duplicates_and_ties():
    words = [" Beta ", "alpha", "BETA", "ALPHA", "gamma", "delta"]
    normalizer = lambda value: str(value).strip().lower()
    expected = ("alpha", "beta", "delta")
    assert vocabulary.build_frequency_vocabulary(
        words, size=3, normalize=normalizer
    ) == expected
    assert vocabulary.build_frequency_vocabulary(
        reversed(words), size=3, normalize=normalizer
    ) == expected


def test_vocabulary_builder_only_reads_explicit_words():
    class ExplicitWords:
        @property
        def test(self):
            raise AssertionError("词表构建器不应读取 test 数据。")

        def __iter__(self):
            return iter(["甲", "乙", "甲"])

    assert vocabulary.build_frequency_vocabulary(ExplicitWords(), 2) == ("甲", "乙")


def test_frozen_vocabulary_validation_and_metadata():
    frozen = vocabulary.validate_frozen_vocabulary(
        [" Alpha ", "beta"],
        expected_size=2,
        normalize=lambda value: str(value).strip().lower(),
    )
    assert frozen == ("alpha", "beta")
    with pytest.raises(ValueError, match="重复"):
        vocabulary.validate_frozen_vocabulary(
            ["Alpha", "alpha"], normalize=lambda value: str(value).lower()
        )

    metadata = vocabulary.build_vocabulary_metadata(
        frozen,
        source_split="train",
        policy="frequency_desc_word_asc",
        normalization="lower_strip",
        word_counts={"alpha": 4, "beta": 3, "unused": 2},
    )
    assert metadata == {
        "vocabulary": ["alpha", "beta"],
        "size": 2,
        "source_split": "train",
        "policy": "frequency_desc_word_asc",
        "normalization": "lower_strip",
        "word_counts": {"alpha": 4, "beta": 3},
    }


def test_word_training_evaluators_reach_the_same_retrieval_implementation():
    from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import train as chinese
    from braindecoding.tasks.word_decoding.pallier2025 import train as pallier

    for module in (chinese, pallier):
        assert (
            module.fixed_vocabulary_retrieval_metrics
            is retrieval.fixed_vocabulary_retrieval_metrics
        )

