import numpy as np

from braindecoding.unified_sentence_batch import replay_many, search_many


class LM:
    def next_words_many(self, prefixes):
        return {tuple(p): ("outside",) for p in prefixes}

    def score_extensions(self, prefixes, words):
        return np.array([.2 if p and p[-1] == w else 0. for p, w in zip(prefixes, words)])


def test_pooling_keeps_brain_candidates_and_frozen_prefixes_separate():
    requests = [dict(queries=np.array([[1., 0.]]), prefix=("fixed",), weight=0., open_vocabulary=False),
                dict(queries=np.array([[0., 1.]]), prefix=(), weight=0., open_vocabulary=False)]
    actual = search_many(requests, ["a", "b"], np.eye(2), LM(), None, candidate_count=1)
    assert actual == [("fixed", "a"), ("b",)]


def test_replay_uses_only_available_delays_and_cached_search_is_not_neural_refresh():
    p = {0: np.array([[1., 0.]] * 4), 1: np.array([[0., 1.]] * 4), 2: np.array([[0., 1.]] * 4)}
    # Deliberately poisoned unavailable predictions must never be inspected.
    p[1][-1] = np.nan
    p[2][-2:] = np.nan
    case = dict(predictions=p, weight=0., open_vocabulary=False)
    result = replay_many([case], ["a", "b"], np.eye(2), LM(), None, candidate_count=2)[0]
    assert result["final"] == dict(A=["a"] * 4, B=["a"] * 4, C=["b", "b", "b", "a"])
    assert result["trace"][0]["C"] == ["a"]
    assert result["trace"][1]["C"] == ["b", "a"]


def test_external_words_enter_all_conditions_and_language_control_ignores_brains():
    p = {d: np.array([[0., 1.]] * 3) for d in range(3)}
    embed = lambda words: np.array([[0., 1.]] * len(words))
    case = dict(predictions=p, weight=0., open_vocabulary=True)
    actual = replay_many([case], ["inside"], np.array([[1., 0.]]), LM(), embed, candidate_count=1)[0]
    assert all(words == ["outside"] * 3 for words in actual["final"].values())
    requests = [dict(queries=np.array([[999., -999.]]), weight=1., open_vocabulary=True, pure_lm=True),
                dict(queries=np.array([[-999., 999.]]), weight=1., open_vocabulary=True, pure_lm=True)]
    assert search_many(requests, ["inside"], np.array([[1., 0.]]), LM(), None)[0] == \
           search_many(requests, ["inside"], np.array([[1., 0.]]), LM(), None)[1]
