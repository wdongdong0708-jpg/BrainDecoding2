"""One complete-word LM scorer and one A/B/C replay for both languages."""
from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F


def first_complete_word(continuation, language):
    """Require look-ahead beyond the word, so a truncated token is never a word."""
    # Punctuation can precede a proposal; it is not emitted as a separate word.
    text = str(continuation).lstrip(" \t\r\n\"«»“”.,;:!?，。；：！？（）()[]")
    if not text:
        return None
    if language == "fr":
        match = re.match(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", text)
        if match is None or match.end() == len(text):
            return None
        word = match.group(0)
        if not all(not c.isalpha() or "LATIN" in unicodedata.name(c, "") for c in word):
            return None
        # Hyphen/apostrophe at truncation can be part of an unfinished word.
        if text[match.end()] in "-'’":
            return None
        return word.lower()
    if language == "zh":
        import jieba
        pieces = list(jieba.tokenize(text, HMM=False))
        if not pieces:
            return None
        word, start, stop = pieces[0]
        if start != 0 or stop == len(text) or not any("\u4e00" <= c <= "\u9fff" for c in word):
            return None
        return word
    raise ValueError("Only declared fr/zh adapters are supported.")


class SharedWordLM:
    """Incremental complete-word tokenization and bounded whole-word prefixes.

    No dataset title or reference text is supplied. A fixed template identifies
    only the language. All candidate subtokens contribute to the score.
    """
    def __init__(self, model, tokenizer, language, settings):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.language = language
        self.settings = settings
        self.device = next(model.parameters()).device
        self.separator = " " if language == "fr" else ""
        name = {"fr": "French", "zh": "Chinese"}[language]
        prompt = settings["language_prompt_template"].format(language=name)
        self.prompt_ids = tuple(tokenizer.encode(prompt, add_special_tokens=False))
        self.context_tokens = min(int(settings["context_tokens"]), int(model.config.max_position_embeddings))
        self.cache = OrderedDict()
        self.proposal_cache = OrderedDict()
        self.word_cache = {}

    def word_ids(self, word, initial=False):
        key = (str(word), bool(initial))
        if key not in self.word_cache:
            self.word_cache[key] = tuple(self.tokenizer.encode(
                ("" if initial else self.separator) + str(word), add_special_tokens=False))
        if not self.word_cache[key]:
            raise ValueError("Candidate word has no tokens.")
        return self.word_cache[key]

    def prefix_ids(self, words, reserve):
        chunks = [self.word_ids(w, initial=(i == 0)) for i, w in enumerate(words)]
        count = len(self.prompt_ids) + sum(map(len, chunks)) + reserve
        while chunks and count > self.context_tokens:
            count -= len(chunks.pop(0))
        if count > self.context_tokens:
            raise ValueError("A complete word exceeds the language model context.")
        return self.prompt_ids + tuple(token for chunk in chunks for token in chunk)

    @torch.inference_mode()
    def score_extensions(self, prefixes, words):
        if len(prefixes) != len(words):
            raise ValueError("Prefix and word counts differ.")
        output = []
        missing = {}
        for i, (prefix, word) in enumerate(zip(prefixes, words)):
            key = (tuple(prefix), str(word))
            output.append(self.cache.get(key))
            if output[-1] is None:
                missing.setdefault(key, []).append(i)
        keys = list(missing)
        batch_size = int(self.settings.get("score_batch_size", 32))
        for start in range(0, len(keys), batch_size):
            batch = keys[start:start+batch_size]
            suffixes = [self.word_ids(word, initial=not prefix) for prefix, word in batch]
            sequences = [self.prefix_ids(prefix, len(suffix)) + suffix
                for (prefix, _), suffix in zip(batch, suffixes)]
            length = max(map(len, sequences))
            pad = self.tokenizer.pad_token_id
            if pad is None:
                pad = self.tokenizer.eos_token_id
            ids = torch.full((len(batch), length), pad, device=self.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, sequence in enumerate(sequences):
                ids[i, -len(sequence):] = torch.tensor(sequence, device=self.device)
                mask[i, -len(sequence):] = 1
            positions = (mask.cumsum(1)-1).clamp_min(0)
            keep = max(map(len, suffixes)) + 1
            logits = self.model(input_ids=ids, attention_mask=mask, position_ids=positions,
                use_cache=False, logits_to_keep=keep).logits[:, :-1].float()
            log_probs = F.log_softmax(logits, dim=-1)
            for i, (key, suffix) in enumerate(zip(batch, suffixes)):
                n = len(suffix)
                score = float(log_probs[i, -n:].gather(1, torch.tensor(suffix, device=self.device)[:, None]).sum())
                self.cache[key] = score
                for slot in missing[key]:
                    output[slot] = score
            while len(self.cache) > int(self.settings.get("score_cache_size", 20000)):
                self.cache.popitem(last=False)
        return np.asarray(output, dtype=np.float64)

    @torch.inference_mode()
    def next_words(self, prefix):
        key = tuple(prefix)
        if key in self.proposal_cache:
            return self.proposal_cache[key]
        width = int(self.settings["proposal_count"])
        budget = int(self.settings["proposal_max_new_tokens"])
        tokens = self.prefix_ids(prefix, budget)
        ids = torch.tensor([tokens], device=self.device)
        # This is a proposal mechanism. Scores are recomputed on complete words;
        # sequence length penalties from generate never enter the final score.
        generated = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=budget, num_beams=width, num_return_sequences=width,
            do_sample=False, use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id)
        candidates = []
        for row in generated:
            text = self.tokenizer.decode(row[len(tokens):], skip_special_tokens=True)
            word = first_complete_word(text, self.language)
            if word and word not in candidates:
                candidates.append(word)
        self.proposal_cache[key] = tuple(candidates)
        while len(self.proposal_cache) > 2000:
            self.proposal_cache.popitem(last=False)
        return tuple(candidates)


def search_words(predictions, vocabulary, bank, scorer, embed_words, *, prefix=(),
                 candidate_count=20, beam_width=5, language_weight=.1,
                 neural_scale=10., open_vocabulary=False, pure_lm=False):
    """Brain normalization domain: full train vocabulary plus new LM proposals."""
    vocabulary = tuple(vocabulary)
    bank = np.asarray(bank, dtype=np.float64)
    index = {w: i for i, w in enumerate(vocabulary)}
    beams = [(tuple(prefix), 0.)]
    for query in predictions:
        neural = None if pure_lm else np.asarray(query, dtype=np.float64) @ bank.T * neural_scale
        if not pure_lm:
            order = np.argsort(-neural, kind="stable")[:candidate_count]
            base = tuple(vocabulary[i] for i in order)
        extensions, prefixes, words, neural_terms = [], [], [], []
        for generated, total in beams:
            proposals = scorer.next_words(generated) if open_vocabulary or pure_lm else ()
            # The LM-only control ranks the full training lexicon itself. It
            # never inherits a shortlist selected by correctly aligned brains.
            candidates = tuple(dict.fromkeys(vocabulary + proposals)) if pure_lm else tuple(dict.fromkeys(base + proposals))
            if not candidates:
                raise RuntimeError("The language-only proposal budget produced no complete word.")
            extra = [w for w in candidates if w not in index]
            novel = {} if pure_lm or not extra else dict(zip(extra,
                np.asarray(embed_words(extra), dtype=np.float64) @ np.asarray(query) * neural_scale))
            if pure_lm:
                normalizer = 0.
            else:
                domain = np.concatenate([neural, np.asarray(list(novel.values()))])
                maximum = float(domain.max())
                normalizer = maximum + float(np.log(np.exp(domain-maximum).sum()))
            for word in candidates:
                score = 0. if pure_lm else (float(neural[index[word]]) if word in index else float(novel[word])) - normalizer
                extensions.append((generated + (word,), total))
                prefixes.append(generated)
                words.append(word)
                neural_terms.append(score)
        lm = scorer.score_extensions(prefixes, words) if language_weight or pure_lm else np.zeros(len(words))
        lm_weight = 1. if pure_lm else language_weight
        scored = [(hypothesis, total + neural_score + lm_weight*float(language_score))
            for (hypothesis,total),neural_score,language_score in zip(extensions,neural_terms,lm)]
        beams = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:beam_width]
    return beams[0][0]


def replay_abc(predictions, vocabulary, bank, scorer, embed_words, **settings):
    if tuple(sorted(predictions)) != (0, 1, 2):
        raise ValueError("Expected K0/K1/K2.")
    count = len(predictions[0])
    if any(np.asarray(p).shape != np.asarray(predictions[0]).shape for p in predictions.values()):
        raise ValueError("Horizon shapes differ.")
    outputs = {condition: () for condition in "ABC"}
    trace = []
    for arrival in range(count):
        for condition in "ABC":
            start = arrival if condition == "A" else max(0, arrival-2)
            prefix = outputs[condition][:start]
            available = np.stack([predictions[arrival-i if condition == "C" else 0][i]
                for i in range(start, arrival+1)])
            result = search_words(available, vocabulary, bank, scorer, embed_words, prefix=prefix, **settings)
            if result[:start] != prefix or len(result) != arrival+1:
                raise RuntimeError("Decoder changed a frozen prefix or word count.")
            outputs[condition] = result
        trace.append({"arrival": arrival, **{c: list(outputs[c]) for c in "ABC"}})
    return {"final": {c: list(outputs[c]) for c in "ABC"}, "trace": trace}

# --- Merged from unified_sentence_batch.py ---
"""Batch the frozen complete-word searches without sharing brain candidates.

Only LM requests and text encoding are pooled. Each search retains its own
prefix, neural normalization domain, beam scores and frozen output positions.
"""

import pickle
from pathlib import Path

import numpy as np
import torch



class BatchedWordLM(SharedWordLM):
    @torch.inference_mode()
    def next_words_many(self, prefixes):
        keys = list(dict.fromkeys(tuple(p) for p in prefixes))
        missing = [p for p in keys if p not in self.proposal_cache]
        batch_size = int(self.settings.get("proposal_batch_size", 16))
        width = int(self.settings["proposal_count"])
        budget = int(self.settings["proposal_max_new_tokens"])
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            sequences = [self.prefix_ids(p, budget) for p in batch]
            length = max(map(len, sequences))
            ids = torch.full((len(batch), length), self.tokenizer.eos_token_id,
                             device=self.device, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(sequences):
                ids[i, -len(seq):] = torch.tensor(seq, device=self.device)
                mask[i, -len(seq):] = 1
            generated = self.model.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=budget,
                num_beams=width, num_return_sequences=width, do_sample=False,
                use_cache=True, pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
            for i, prefix in enumerate(batch):
                candidates = []
                for row in generated[i * width:(i + 1) * width]:
                    word = first_complete_word(
                        self.tokenizer.decode(row[length:], skip_special_tokens=True), self.language)
                    if word and word not in candidates:
                        candidates.append(word)
                self.proposal_cache[prefix] = tuple(candidates)
        return {p: self.proposal_cache[p] for p in keys}

    def next_words(self, prefix):
        return self.next_words_many([prefix])[tuple(prefix)]

    def save_cache(self, path):
        # Cache keys contain generated prefixes only, never reference text.
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump((self.language, dict(self.cache), dict(self.proposal_cache)), handle, protocol=5)
        temporary.replace(path)

    def load_cache(self, path):
        path = Path(path)
        if path.exists():
            with path.open("rb") as handle:
                language, scores, proposals = pickle.load(handle)
            if language != self.language:
                raise ValueError("Language cache mismatch")
            self.cache.update(scores)
            self.proposal_cache.update(proposals)


def search_many(requests, vocabulary, bank, scorer, embed_words, *,
                candidate_count=20, beam_width=5, neural_scale=10.):
    """Requests carry queries, prefix, weight, open flag and optional pure_lm."""
    vocabulary = tuple(vocabulary)
    index = {w: i for i, w in enumerate(vocabulary)}
    bank = np.asarray(bank, dtype=np.float64)
    beams = [[(tuple(r.get("prefix", ())), 0.)] for r in requests]
    counts = [len(r["queries"]) for r in requests]
    if not counts:
        return []
    for step in range(max(counts)):
        active = [i for i, n in enumerate(counts) if n > step]
        proposed_prefixes = [p for i in active if requests[i].get("open_vocabulary")
                            or requests[i].get("pure_lm") for p, _ in beams[i]]
        if hasattr(scorer, "next_words_many"):
            proposals = scorer.next_words_many(proposed_prefixes)
        else:
            proposals = {p: scorer.next_words(p) for p in set(proposed_prefixes)}
        outside = sorted({w for ws in proposals.values() for w in ws if w not in index})
        needs_embeddings = any(not requests[i].get("pure_lm") for i in active)
        extra_vectors = dict(zip(outside, np.asarray(embed_words(outside), dtype=np.float64))) \
            if outside and needs_embeddings else {}
        slots, prefixes, words, neural_terms, totals, owners = [], [], [], [], [], []
        for i in active:
            request = requests[i]
            pure_lm = request.get("pure_lm", False)
            query = np.asarray(request["queries"][step], dtype=np.float64)
            neural = None if pure_lm else query @ bank.T * neural_scale
            base = vocabulary if pure_lm else tuple(vocabulary[j] for j in
                np.argsort(-neural, kind="stable")[:candidate_count])
            for prefix, total in beams[i]:
                suggestions = proposals.get(prefix, ()) if request.get("open_vocabulary") or pure_lm else ()
                candidates = tuple(dict.fromkeys(base + suggestions))
                if not candidates:
                    raise RuntimeError("Empty complete-word candidate set")
                novel = {w: float(extra_vectors[w] @ query * neural_scale)
                         for w in suggestions if w not in index} if not pure_lm else {}
                if pure_lm:
                    normalizer = 0.
                else:
                    domain = np.concatenate([neural, np.asarray(list(novel.values()))])
                    maximum = float(domain.max())
                    normalizer = maximum + float(np.log(np.exp(domain - maximum).sum()))
                for word in candidates:
                    slots.append(prefix + (word,))
                    prefixes.append(prefix)
                    words.append(word)
                    neural_terms.append(0. if pure_lm else
                        (float(neural[index[word]]) if word in index else novel[word]) - normalizer)
                    totals.append(total)
                    owners.append(i)
        scoring = [j for j, i in enumerate(owners) if requests[i].get("pure_lm") or requests[i]["weight"]]
        lm_scores = np.zeros(len(words))
        if scoring:
            lm_scores[scoring] = scorer.score_extensions(
                [prefixes[j] for j in scoring], [words[j] for j in scoring])
        grouped = {i: [] for i in active}
        for j, i in enumerate(owners):
            weight = 1. if requests[i].get("pure_lm") else requests[i]["weight"]
            grouped[i].append((slots[j], totals[j] + neural_terms[j] + weight * float(lm_scores[j])))
        for i in active:
            beams[i] = sorted(grouped[i], key=lambda item: (-item[1], item[0]))[:beam_width]
    return [b[0][0] for b in beams]


def replay_many(cases, vocabulary, bank, scorer, embed_words, *,
                conditions="ABC", progress=None, **search_settings):
    """Replay complete segments together; no case supplies reference words."""
    outputs = [{c: () for c in conditions} for _ in cases]
    traces = [[] for _ in cases]
    counts = [len(case["predictions"][0]) for case in cases]
    for arrival in range(max(counts, default=0)):
        requests, owners = [], []
        for i, case in enumerate(cases):
            if arrival >= counts[i]:
                continue
            for condition in conditions:
                start = arrival if condition == "A" else max(0, arrival - 2)
                queries = np.stack([case["predictions"][arrival - pos if condition == "C" else 0][pos]
                                    for pos in range(start, arrival + 1)])
                requests.append(dict(queries=queries, prefix=outputs[i][condition][:start],
                    weight=case["weight"], open_vocabulary=case["open_vocabulary"],
                    pure_lm=case.get("pure_lm", False)))
                owners.append((i, condition, start))
        results = search_many(requests, vocabulary, bank, scorer, embed_words, **search_settings)
        for result, request, (i, condition, start) in zip(results, requests, owners):
            if result[:start] != request["prefix"] or len(result) != arrival + 1:
                raise RuntimeError("Frozen prefix or known word count changed")
            outputs[i][condition] = result
        for i, n in enumerate(counts):
            if arrival < n:
                traces[i].append(dict(arrival=arrival, **{c: list(outputs[i][c]) for c in conditions}))
        if progress:
            progress(arrival, counts)
    return [dict(final={c: list(words) for c, words in out.items()}, trace=trace)
            for out, trace in zip(outputs, traces)]
