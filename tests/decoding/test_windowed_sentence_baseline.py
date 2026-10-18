import numpy as np

from braindecoding.jayalath_sentence import (
    beam_decode,
    candidate_rows,
    final_delay_scores,
    grouped_calibration_mask,
    oov_features,
    sentence_metrics,
    training_retrieval_vocabulary,
)


class FakeScorer:
    def word_ids(self, word, *, initial=False):
        return (ord(word[0]),)

    def score_extensions(self, prefixes, words):
        return np.asarray([2.0 if word == "好" else 0.0 for word in words])


def test_training_vocabulary_ignores_unembedded_and_breaks_ties():
    words = ["乙", "甲", "乙", "甲", "丙", "丙", "丁"]
    assert training_retrieval_vocabulary(words, {"甲": 1, "乙": 1, "丁": 1}, 2) == ("乙", "甲")


def test_final_delay_uses_last_available_event():
    values = {k: np.full((4, 2), k, dtype=float) for k in range(3)}
    result = final_delay_scores(values, [[0, 1, 2], [3]])
    assert result[:, 0].tolist() == [2, 1, 0, 0]


def test_beam_language_score_can_change_neural_first_choice():
    ids, log_probs = candidate_rows([[2.0, 1.0], [2.0, 1.0]], 2)
    decoded = beam_decode(ids, log_probs, ("错", "好"), FakeScorer(),
                          beam_width=2, language_weight=1.0)
    assert decoded == ("好", "好")


def test_sentence_metrics_and_repeated_stimulus_split():
    metrics = sentence_metrics([["我", "好"], ["你"]], [["我", "错"], ["你"]],
                               chinese=True)
    assert metrics["wer"] == 1 / 3
    assert metrics["bleu1"] == 2 / 3
    assert metrics["cer"] == 1 / 3
    unequal = sentence_metrics([["a"], ["b", "c", "d"]],
                               [["a"], ["x", "y", "z"]])
    assert unequal["bleu1"] == 1 / 4
    assert unequal["sentence_bleu1_mean"] == 1 / 2
    mask = grouped_calibration_mask([["我", "好"], ["我", "好"], ["你"]])
    assert mask[0] == mask[1]


def test_french_word_boundaries_match_sentence_tokenization():
    from pathlib import Path

    from transformers import GPT2Tokenizer

    from braindecoding.jayalath_sentence import CausalLMScorer

    path = Path(__file__).resolve().parents[2] / "data/cache/derived/models/gpt-fr-cased-small"
    if not (path / "vocab.json").is_file():
        import pytest
        pytest.skip("French tokenizer is not installed")
    tokenizer = GPT2Tokenizer.from_pretrained(path, local_files_only=True)
    scorer = object.__new__(CausalLMScorer)
    scorer.tokenizer = tokenizer
    scorer.word_separator = " "
    scorer._word_cache = {}
    words = ("le", "petit", "prince")
    pieces = sum((scorer.word_ids(word, initial=index == 0)
                  for index, word in enumerate(words)), ())
    assert pieces == tuple(tokenizer.encode(" ".join(words), add_special_tokens=False))


def test_beam_context_keeps_only_recent_words():
    class RecordingScorer(FakeScorer):
        max_context_words = 2

        def __init__(self):
            self.prefixes = []

        def word_ids(self, word, *, initial=False):
            return (ord(word[0]) + (100 if initial else 0),)

        def score_extensions(self, prefixes, words):
            self.prefixes.extend(tuple(prefix) for prefix in prefixes)
            return np.zeros(len(words))

    scorer = RecordingScorer()
    beam_decode(np.zeros((4, 1), dtype=int), np.zeros((4, 1)),
                ("a",), scorer, beam_width=1)
    assert scorer.prefixes == [(), (197,), (197, 97), (197, 97)]


def test_first_subtoken_score_matches_author_search_rule():
    from types import SimpleNamespace

    import torch

    from braindecoding.jayalath_sentence import CausalLMScorer

    class TwoTokenTokenizer:
        bos_token_id = 0
        pad_token_id = 0

        def encode(self, word, add_special_tokens=False):
            return [1, 2]

    class UniformModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 4)))

    tokenizer = TwoTokenTokenizer()
    full = CausalLMScorer(UniformModel(), tokenizer, device="cpu")
    first = CausalLMScorer(UniformModel(), tokenizer, device="cpu",
                           first_subtoken_only=True)
    assert np.isclose(full.score_extensions([()], ["mot"])[0], -2 * np.log(4))
    assert np.isclose(first.score_extensions([()], ["mot"])[0], -np.log(4))


def test_lm_score_cache_preserves_duplicate_extension_scores():
    from types import SimpleNamespace

    import torch

    from braindecoding.jayalath_sentence import CausalLMScorer

    class Tokenizer:
        bos_token_id = 0
        pad_token_id = 0

        def encode(self, word, add_special_tokens=False):
            return [1 if word == "a" else 2]

    class UniformModel(torch.nn.Module):
        def forward(self, input_ids, attention_mask=None):
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 3)))

    scorer = CausalLMScorer(UniformModel(), Tokenizer(), device="cpu",
                            score_cache_size=2)
    first = scorer.score_extensions([(), ()], ["a", "a"])
    second = scorer.score_extensions([(), (), ()], ["a", "b", "a"])
    assert np.allclose(first, [-np.log(3)] * 2)
    assert np.allclose(second, [-np.log(3)] * 3)


def test_detected_oov_can_leave_neural_retrieval_vocabulary():
    class OpenScorer(FakeScorer):
        def next_word_candidates(self, prefix, width=5):
            return [("libre", -0.1)]

    decoded = beam_decode(np.zeros((2, 1), dtype=int),
                          np.zeros((2, 1)), ("mot",), OpenScorer(),
                          beam_width=1, oov_flags=[False, True])
    assert decoded == ("mot", "libre")


def test_oov_detector_can_use_full_retrieval_distribution():
    scores = np.asarray([[0.2, 0.7, 0.1], [0.3, 0.2, 0.5]])
    assert oov_features(scores).shape == (2, 3)
    full = oov_features(scores, include_probabilities=True)
    assert full.shape == (2, 6)
    assert np.allclose(full[:, :3].sum(axis=1), 1)
