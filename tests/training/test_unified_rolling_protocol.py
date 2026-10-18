from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding.unified_protocol_audit import build_training_vocabulary, load_source
from braindecoding.unified_rolling_training import equal_event_horizon_loss, full_vocabulary_metrics
from braindecoding.unified_sentence import first_complete_word, replay_abc, search_words


def test_epoch_weighting_preserves_event_and_horizon_means_and_gradients():
    scale = torch.tensor(2.0, requires_grad=True)
    values = {0: torch.tensor([1., 3., 8.]), 1: torch.tensor([2., 10.]), 2: torch.tensor([7.])}
    totals = {d: len(v) for d, v in values.items()}
    # Deliberately uneven batches and a missing horizon in one batch.
    partitions = [{0: [0, 1], 1: [0], 2: [0]}, {0: [2], 1: [1]}]
    batch_losses = []
    for batch in partitions:
        losses = {d: (scale * values[d][ids]).mean() for d, ids in batch.items()}
        batch_losses.append(equal_event_horizon_loss(losses, {d: len(ids) for d, ids in batch.items()}, totals, 2))
    actual = torch.stack(batch_losses).mean()
    expected = torch.stack([(scale*v).mean() for v in values.values()]).mean()
    assert float(actual.detach()) == pytest.approx(float(expected.detach()))
    actual_grad = torch.autograd.grad(actual, scale)[0]
    expected_grad = torch.autograd.grad(expected, scale)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_vocabulary_counts_material_once_and_rejects_validation_inputs():
    table = pd.DataFrame({"split": ["train"]*4, "material_event": ["m1", "m1", "m2", "m2"],
        "normalized_word": ["猫", "猫", "chat", "chat"]})
    words, counts = build_training_vocabulary(table)
    assert words == ["chat", "猫"]
    assert counts == {"chat": 1, "猫": 1}
    table.loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="train only"):
        build_training_vocabulary(table)
    with pytest.raises(ValueError, match="train/val"):
        load_source("pallier2025", "test")


def test_full_vocab_evaluation_keeps_unseen_reference_as_error_and_subjects_equal():
    ds = SimpleNamespace(embedding_map={"a": np.array([1., 0.]), "b": np.array([0., 1.])},
        table=pd.DataFrame({"event_id": ["e1", "e2", "e3"], "subject_id": ["s1", "s1", "s2"]}))
    encoded = {"predictions": torch.tensor([[1., 0.]]*3), "words": ["a", "outside", "a"],
        "event_ids": ["e1", "e2", "e3"]}
    metrics = full_vocabulary_metrics(encoded, ds, ["a", "b"])
    assert metrics["reference_count"] == 3
    assert metrics["reference_in_vocabulary"] == pytest.approx(2/3)
    assert metrics["selection_score"] == .75


def test_common_horizon_selection_excludes_only_unavailable_positions():
    ds = SimpleNamespace(embedding_map={"a": np.array([1., 0.])},
        table=pd.DataFrame({"event_id": ["e1", "e2"], "subject_id": ["s", "s"]}))
    encoded = {"words": ["a", "outside"], "event_ids": ["e1", "e2"],
        "predictions": torch.tensor([[1., 0.]]*2),
        "revision_predictions": {d: torch.tensor([[1., 0.]]*2) for d in range(3)},
        "revision_availability_masks": {0: torch.tensor([True, True]),
            1: torch.tensor([True, False]), 2: torch.tensor([True, False])}}
    metrics = full_vocabulary_metrics(encoded, ds, ["a"])
    assert metrics["common_count"] == 1
    assert metrics["reference_count"] == 2
    assert metrics["selection_score"] == 1


class ZeroLM:
    def score_extensions(self, prefixes, words):
        return np.zeros(len(words))

    def next_words(self, prefix):
        return ("outside",)


def test_abc_separates_neural_updates_and_respects_unavailable_horizons():
    predictions = {0: np.array([[1., 0.]]*4), 1: np.array([[0., 1.]]*4), 2: np.array([[0., 1.]]*4)}
    result = replay_abc(predictions, ["a", "b"], np.eye(2), ZeroLM(), None,
        candidate_count=2, language_weight=0, open_vocabulary=False)
    assert result["final"]["A"] == result["final"]["B"] == ["a"]*4
    assert result["final"]["C"] == ["b", "b", "b", "a"]
    assert result["trace"][0]["C"] == ["a"]
    assert result["trace"][1]["C"] == ["b", "a"]
    predictions[2][3] = [99., -99.]
    repeated = replay_abc(predictions, ["a", "b"], np.eye(2), ZeroLM(), None,
        candidate_count=2, language_weight=0, open_vocabulary=False)
    assert repeated == result


def test_open_candidates_are_available_to_all_three_conditions_without_reference():
    predictions = {d: np.array([[0., 1.]]*3) for d in range(3)}
    result = replay_abc(predictions, ["inside"], np.array([[1., 0.]]), ZeroLM(),
        lambda words: np.array([[0., 1.]]*len(words)), candidate_count=1,
        language_weight=0, open_vocabulary=True)
    assert all(words == ["outside"]*3 for words in result["final"].values())


def test_language_only_baseline_does_not_use_neural_candidates():
    class LM(ZeroLM):
        def score_extensions(self, prefixes, words):
            return np.array([1. if w == "outside" else 0. for w in words])
    first = search_words(np.array([[999., -999.]]), ["inside"], np.array([[1., 0.]]),
        LM(), None, pure_lm=True)
    second = search_words(np.array([[-999., 999.]]), ["inside"], np.array([[0., 1.]]),
        LM(), None, pure_lm=True)
    assert first == second == ("outside",)


def test_complete_word_proposals_require_boundaries_and_keep_multiple_characters():
    assert first_complete_word("constitutionnellement ensuite", "fr") == "constitutionnellement"
    assert first_complete_word("constitutionnellement", "fr") is None
    assert first_complete_word("porte-", "fr") is None
    assert first_complete_word("玫瑰花正在盛开", "zh") == "玫瑰花"
