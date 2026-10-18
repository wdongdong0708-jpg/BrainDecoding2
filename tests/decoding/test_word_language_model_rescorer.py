from types import SimpleNamespace

import numpy as np
import pytest
import torch

from braindecoding.natural_word_lm import NaturalWordLM, PROMPTS, first_unit, render_words


class BoundaryTokenizer:
    pad_token_id = eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        # One merged token replaces two tokens at a word boundary.
        return [16 if c == "§" else 1 + ord(c) % 15 for c in text.replace("一个", "§")]


class CausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_position_embeddings=1024)

    def forward(self, input_ids, attention_mask=None, logits_to_keep=0, **kwargs):
        state = (input_ids * attention_mask).cumsum(1).float()
        logits = torch.sin(state[:, :, None] * .13 + torch.arange(17)[None, None, :] * .37)
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


def make_lm(language="zh", **settings):
    return NaturalWordLM(CausalModel(), BoundaryTokenizer(), language,
        dict(revision="test", context_tokens=1024, score_batch_size=3, **settings))


def full_logp(lm, words):
    ids = torch.tensor([lm.tokenizer.encode(lm.input_text(words))])
    logits = lm.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1].float()
    targets = ids[:, 1:, None]
    terms = logits.gather(2, targets).squeeze(2) - logits.logsumexp(-1)
    return float(terms.double().sum())


@pytest.mark.parametrize("words,text", [(["alors", "c", "est", "un"], "alors c'est un"),
    (["j", "étais"], "j'étais"), (["l", "aviateur"], "l'aviateur"),
    (["donne", "-moi"], "donne-moi"), (["qu"], "qu"), (["puisqu", "il"], "puisqu'il"),
    (["aujourd", "hui"], "aujourd'hui"), (["quelqu", "un"], "quelqu'un"),
    (["chef", "-d", "œuvre"], "chef-d'œuvre")])
def test_french_surface(words, text):
    assert render_words(words, "fr") == text


@pytest.mark.parametrize("text,prefix,expected", [("c'est un", (), "c"),
    ("'est un", ("c",), "est"), ("donne-moi le", (), "donne"),
    ("-moi le", ("donne",), "-moi"), ("j’étais seul", (), "j"),
    ("aviateur", (), None), ("c'", (), None), ("'œuvre est", ("chef", "-d"), "œuvre")])
def test_french_proposal_units(text, prefix, expected):
    assert first_unit(text, "fr", prefix) == expected


def test_prompt_and_canonical_boundary():
    lm = make_lm()
    assert lm.input_text(["一", "个"]) == PROMPTS["zh"] + "\n\n一个"
    assert lm.prefix_ids(["一", "个"]) == tuple(lm.tokenizer.encode(lm.input_text(["一", "个"])))
    assert lm.prefix_ids(["一"]) + tuple(lm.tokenizer.encode("个")) != lm.prefix_ids(["一", "个"])


@pytest.mark.parametrize("language,prefixes,words", [
    ("zh", [[], ["一"], ["一", "个"], ["很", "长", "的", "一"]], ["一", "个", "人", "个"]),
    ("fr", [[], ["c"], ["j"], ["donne"]], ["alors", "est", "étais", "-moi"])])
def test_joint_probability_difference_with_boundary_rewrite(language, prefixes, words):
    lm = make_lm(language)
    actual = lm.score_extensions(prefixes, words)
    gold = [full_logp(lm, p + [w]) - full_logp(lm, p) for p, w in zip(prefixes, words)]
    np.testing.assert_allclose(actual, gold, rtol=0, atol=1e-6)
    assert np.array_equal(actual, lm.score_extensions(prefixes, words))


def test_cache_eviction_and_identity(tmp_path):
    lm = make_lm(score_cache_size=1)
    assert len(lm.score_extensions([[], [], []], ["我", "你", "他"])) == 3
    path = tmp_path / "cache.pkl"
    lm.save_cache(path)
    make_lm(score_cache_size=1).load_cache(path)
    with pytest.raises(ValueError, match="mismatch"):
        make_lm("fr", score_cache_size=1).load_cache(path)


def test_context_overflow_is_explicit():
    lm = make_lm()
    lm.context_tokens = 1
    with pytest.raises(ValueError, match="context"):
        lm.prefix_ids(["我"])
