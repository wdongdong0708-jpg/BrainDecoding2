"""Checks for bounded sequence revision and paired validation accounting."""

import numpy as np
import pytest

from braindecoding.sequence_decoding import (
    decode_rolling_group,
    evaluate_rolling_sequences,
    fit_bigram_prior,
)


def test_arriving_evidence_changes_recent_word_and_freezes_older_word():
    prior = fit_bigram_prior([["a", "b"], ["a", "b"]], ("a", "b"))
    scores = {
        0: np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]]),
        1: np.array([[0.0, 2.0], [2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]),
    }
    revisions, final = decode_rolling_group(scores, prior, 0.0)
    assert revisions[0].tolist() == [0, 1, 0, 1]
    assert revisions[1][:3].tolist() == [1, 0, 1]
    assert revisions[1][-1] == -1
    assert final.tolist() == [1, 0, 1, 1]


def test_language_prior_can_choose_joint_path_over_greedy():
    prior = fit_bigram_prior([["a", "b"]] * 20, ("a", "b"))
    scores = {
        0: np.array([[0.0, 0.2], [0.0, 0.2]]),
        1: np.array([[0.0, 0.2], [0.0, 0.2]]),
    }
    greedy, _ = decode_rolling_group(scores, prior, 0.0)
    joint, _ = decode_rolling_group(scores, prior, 1.0)
    assert greedy[0].tolist() == [1, 1]
    assert joint[1][0] == 0
    assert joint[0][1] == 1


def test_common_horizon_uses_same_targets_and_rejects_cross_group_indexing():
    prior = fit_bigram_prior([["a", "b", "a"]], ("a", "b"))
    scores = {delay: np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]) for delay in (0, 1, 2)}
    output = evaluate_rolling_sequences(
        scores, ["a", "b", "a"], [[0, 1, 2]], prior, [0.5]
    )
    for block in output.values():
        assert [block["common_horizons"][str(delay)]["events"] for delay in (0, 1, 2)] == [1, 1, 1]
    metrics, details = evaluate_rolling_sequences(
        scores, ["a", "b", "a"], [[0, 1, 2]], prior, [0.5],
        return_details=True,
    )
    assert metrics == output
    assert details["greedy"]["final"].tolist() == ["a", "b", "a"]
    with pytest.raises(ValueError, match="partition"):
        evaluate_rolling_sequences(scores, ["a", "b", "a"], [[0, 1]], prior, [0.5])
