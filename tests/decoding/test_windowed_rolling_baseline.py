import numpy as np

from braindecoding.jayalath_rolling import decode_rolling_group
from braindecoding.decoding.windowed_sequence_baseline import shift_horizon_scores


class FakeScorer:
    def word_ids(self, word, *, initial=False):
        return (ord(word[0]),)

    def score_extensions(self, prefixes, words):
        return np.zeros(len(words))


def test_rolling_uses_future_horizons_only_when_available_and_freezes_old_words():
    # K1 fixes the first word; an attractive K2 score for the final word is
    # unavailable and must never enter the replay.
    ids = {
        0: np.array([[0, 1], [0, 1], [0, 1], [0, 1]]),
        1: np.array([[1, 0], [0, 1], [0, 1], [0, 1]]),
        2: np.array([[1, 0], [0, 1], [0, 1], [1, 0]]),
    }
    probabilities = {delay: np.tile([0.0, -2.0], (4, 1)) for delay in ids}
    result = decode_rolling_group(
        ids, probabilities, ("a", "b"), FakeScorer(),
        beam_width=2, language_weight=0.0, reference=("b", "a", "a", "a"),
    )
    assert result["k0_frozen"] == ("a", "a", "a", "a")
    assert result["k2_rolling"] == ("b", "a", "a", "a")
    assert result["trace"][0]["k2_rolling"] == ["a"]
    assert result["trace"][1]["k2_rolling"][:1] == ["b"]
    assert result["trace"][3]["k2_rolling"][0] == "b"
    assert [step["newly_frozen_word_indices"] for step in result["trace"]] == [
        [], [], [0], [1, 2, 3]
    ]
    assert all(step["k2_decoder_seconds"] >= 0 for step in result["trace"])
    assert result["transitions"][0]["kind"] == "wrong_to_right"


def test_frozen_prefix_is_visible_to_lm_but_cannot_change():
    class ContextScorer(FakeScorer):
        def score_extensions(self, prefixes, words):
            return np.array([3.0 if prefix and word == "b" else 0.0
                             for prefix, word in zip(prefixes, words)])

    from braindecoding.jayalath_sentence import beam_decode

    output = beam_decode(np.array([[0, 1]]), np.array([[0.0, -1.0]]),
                         ("a", "b"), ContextScorer(), beam_width=2,
                         language_weight=1.0, frozen_prefix=("a",))
    assert output == ("a", "b")


def test_unavailable_future_oov_flag_does_not_change_current_display():
    class OpenScorer(FakeScorer):
        def next_word_candidates(self, prefix, width=5):
            return [("outside", -0.1)]

    ids = {delay: np.zeros((2, 1), dtype=int) for delay in range(3)}
    scores = {delay: np.zeros((2, 1)) for delay in range(3)}
    flags = {0: np.array([False, False]),
             1: np.array([False, False]),
             2: np.array([False, True])}
    result = decode_rolling_group(
        ids, scores, ("inside",), OpenScorer(), beam_width=1,
        language_weight=0.025, oov_flags=flags,
    )
    assert result["k2_rolling"] == ("inside", "inside")


def test_shifted_control_reuses_one_subject_preserving_permutation_for_all_horizons():
    subjects = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    scores = {delay: np.arange(8)[:, None] + delay * 10
              for delay in (0, 1, 2)}
    masks = {0: np.ones(8, dtype=bool),
             1: np.array([True, True, False, False] * 2),
             2: np.zeros(8, dtype=bool)}
    shifted, permutation = shift_horizon_scores(scores, subjects, masks, seed=1729)
    assert np.all(permutation != np.arange(8))
    assert np.array_equal(subjects[permutation], subjects)
    assert np.array_equal(masks[1][permutation], masks[1])
    for delay in (0, 1, 2):
        assert np.array_equal(shifted[delay], scores[delay][permutation])
    assert np.array_equal(shifted[2] - shifted[0], np.full((8, 1), 20))
