"""Natural LM surfaces and exact canonical-string likelihood differences.

This is a new interface. Frozen historical scorers and their caches are untouched.
Evaluation units remain separate from the text shown to the language model.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

PROMPTS = {
    "fr": "Écrivez un paragraphe de récit en français. Donnez seulement le texte, sans titre ni liste.",
    "zh": "请用中文写一段连续的叙述文字。只给出正文，不要标题或列表。",
}
ELISIONS = frozenset("c j l d n m s t qu jusqu lorsqu puisqu quelqu aujourd".split())
INTERFACE_VERSION = "natural_word_lm_v1"


def render_words(words, language):
    words = tuple(str(w) for w in words)
    if any(not w or w != w.strip() or any(c.isspace() for c in w) for w in words):
        raise ValueError("Expected nonempty individual evaluation units")
    if language == "zh":
        return "".join(words)
    if language != "fr":
        raise ValueError("Only fr and zh are supported")
    result = ""
    for i, word in enumerate(words):
        if not i:
            result = word
        elif word.startswith("-"):
            result += word
        elif words[i - 1].lower().lstrip("-") in ELISIONS:
            result += "'" + word
        else:
            result += " " + word
    return result


def first_unit(continuation, language, prefix=()):
    """Return a complete evaluation unit, using generation only as a proposal."""
    text = str(continuation).lstrip(" \t\r\n\"«»“”.,;:!?，。；：！？（）()[]")
    if language == "zh":
        import jieba
        pieces = list(jieba.tokenize(text, mode="default", HMM=False))
        if not pieces:
            return None
        word, start, stop = pieces[0]
        return word if start == 0 and stop < len(text) and any("\u4e00" <= c <= "\u9fff" for c in word) else None
    if language != "fr":
        raise ValueError("Only fr and zh are supported")
    if prefix and str(prefix[-1]).lower().lstrip("-") in ELISIONS:
        text = text.lstrip("'’")
    # The corpus splits c'est -> c, est and donne-moi -> donne, -moi.
    match = re.match(r"-?[^\W\d_]+", text)
    if match is None or match.end() == len(text):
        return None
    if text[match.end()] in "-'’" and match.end() + 1 == len(text):
        return None
    word = match.group().lower()
    return word if all(not c.isalpha() or "LATIN" in unicodedata.name(c, "") for c in word) else None


def common_prefix_length(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


class NaturalWordLM:
    def __init__(self, model, tokenizer, language, settings):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.language, self.settings = language, dict(settings)
        self.device = next(model.parameters()).device
        self.prompt = PROMPTS[language] + "\n\n"
        self.prompt_ids = tuple(tokenizer.encode(self.prompt, add_special_tokens=False))
        self.context_tokens = min(int(settings.get("context_tokens", 32768)), int(model.config.max_position_embeddings))
        self.cache, self.proposal_cache = OrderedDict(), OrderedDict()

    def input_text(self, words):
        return self.prompt + render_words(words, self.language)

    def prefix_ids(self, words, reserve=0):
        ids = tuple(self.tokenizer.encode(self.input_text(words), add_special_tokens=False))
        if not ids or len(ids) + reserve > self.context_tokens:
            raise ValueError("Canonical input exceeds context; silent truncation would change the score")
        return ids

    @torch.inference_mode()
    def _tail_scores(self, requests):
        result = {}
        pending = list(dict.fromkeys(requests))
        batch_size = int(self.settings.get("score_batch_size", 32))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            active = [(seq, cut) for seq, cut in batch if cut < len(seq)]
            result.update({(seq, cut): 0. for seq, cut in batch if cut == len(seq)})
            if not active:
                continue
            length = max(len(seq) for seq, _ in active)
            keep = max(len(seq) - cut for seq, cut in active) + 1
            ids = torch.full((len(active), length), pad, device=self.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, (seq, _) in enumerate(active):
                ids[i, -len(seq):] = torch.tensor(seq, device=self.device)
                mask[i, -len(seq):] = 1
            logits = self.model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(1) - 1).clamp_min(0),
                use_cache=False, logits_to_keep=keep).logits[:, :-1].float()
            for i, (seq, cut) in enumerate(active):
                n = len(seq) - cut
                tail = logits[i, -n:]
                targets = torch.tensor(seq[cut:], device=self.device)[:, None]
                terms = tail.gather(1, targets).squeeze(1) - torch.logsumexp(tail, dim=-1)
                result[(seq, cut)] = float(terms.double().sum())
        return result

    def score_extensions(self, prefixes, words):
        if len(prefixes) != len(words):
            raise ValueError("Prefix and word counts differ")
        keys = [(tuple(p), str(w)) for p, w in zip(prefixes, words)]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        pairs, requests = {}, []
        for key in missing:
            prefix, word = key
            old = self.prefix_ids(prefix)
            new = self.prefix_ids(prefix + (word,))
            cut = common_prefix_length(old, new)
            if cut < 1:
                raise ValueError("Canonical pair must retain a shared conditioning token")
            pairs[key] = ((new, cut), (old, cut))
            requests.extend(pairs[key])
        scores = self._tail_scores(requests)
        resolved = {key: scores[new] - scores[old] for key, (new, old) in pairs.items()}
        if not all(np.isfinite(s) for s in resolved.values()):
            raise ValueError("Non-finite canonical language score")
        # Build output before eviction: one call may exceed the bounded cache.
        output = np.asarray([resolved[k] if k in resolved else self.cache[k] for k in keys], dtype=np.float64)
        self.cache.update(resolved)
        while len(self.cache) > int(self.settings.get("score_cache_size", 200000)):
            self.cache.popitem(last=False)
        return output

    @torch.inference_mode()
    def next_words_many(self, prefixes):
        keys = list(dict.fromkeys(tuple(p) for p in prefixes))
        missing = [p for p in keys if p not in self.proposal_cache]
        width = int(self.settings.get("proposal_count", 5))
        budget = int(self.settings.get("proposal_max_new_tokens", 16))
        batch_size = int(self.settings.get("proposal_batch_size", 16))
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            sequences = [self.prefix_ids(p, budget) for p in batch]
            length = max(map(len, sequences))
            ids = torch.full((len(batch), length), self.tokenizer.eos_token_id, device=self.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(sequences):
                ids[i, -len(seq):] = torch.tensor(seq, device=self.device)
                mask[i, -len(seq):] = 1
            rows = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=budget,
                num_beams=width, num_return_sequences=width, do_sample=False, use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id)
            for i, prefix in enumerate(batch):
                words = [first_unit(self.tokenizer.decode(row[length:], skip_special_tokens=True), self.language, prefix)
                    for row in rows[i * width:(i + 1) * width]]
                self.proposal_cache[prefix] = tuple(dict.fromkeys(w for w in words if w))
        return {p: self.proposal_cache[p] for p in keys}

    def next_words(self, prefix):
        return self.next_words_many([prefix])[tuple(prefix)]

    def cache_identity(self):
        identity = dict(interface=INTERFACE_VERSION, prompt=self.prompt, language=self.language,
            settings=self.settings, dtype=str(next(self.model.parameters()).dtype), torch=torch.__version__)
        return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def save_cache(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(dict(format=INTERFACE_VERSION, identity=self.cache_identity(),
                scores=dict(self.cache), proposals=dict(self.proposal_cache)), handle, protocol=5)
        temporary.replace(path)

    def load_cache(self, path):
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, dict) or value.get("format") != INTERFACE_VERSION or value.get("identity") != self.cache_identity():
            raise ValueError("Language interface/model/prompt cache mismatch")
        if not np.isfinite(list(value["scores"].values())).all():
            raise ValueError("Non-finite cached score")
        self.cache.update(value["scores"])
        self.proposal_cache.update(value["proposals"])

# --- Merged from natural_word_lm_prompt.py ---
"""Explicit prompt selection on top of the verified natural-word interface."""


class PromptedNaturalWordLM(NaturalWordLM):
    def __init__(self, model, tokenizer, language, settings, *, prompt_body):
        if not isinstance(prompt_body, str) or not prompt_body or prompt_body != prompt_body.strip():
            raise ValueError("Prompt body must be nonempty and have no surrounding whitespace")
        super().__init__(model, tokenizer, language, dict(settings, prompt_body=prompt_body))
        self.prompt = prompt_body + "\n\n"
        self.prompt_ids = tuple(tokenizer.encode(self.prompt, add_special_tokens=False))
