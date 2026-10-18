"""Validation-only sentence decoding with a causal language-model rescorer.

Word positions and group boundaries are supplied by the frozen event protocol.
The retrieval set is a source of neural proposals, not the final output space.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from string import punctuation

import numpy as np
import torch
from torch.nn import functional as F


def training_retrieval_vocabulary(words, embeddings, size):
    """Choose frequent training words with deterministic lexical tie breaking."""
    if size < 1:
        raise ValueError("Retrieval size must be positive.")
    counts = Counter(map(str, words))
    selected = [word for word, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                if word in embeddings]
    if len(selected) < size:
        raise ValueError("Fewer embedded training words than retrieval size.")
    return tuple(selected[:size])


def final_delay_scores(scores_by_delay, groups):
    """Use the last physically available delay for each aligned word."""
    delays = tuple(sorted(scores_by_delay))
    if delays != tuple(range(len(delays))):
        raise ValueError("Expected consecutive delays from K0.")
    arrays = {delay: np.asarray(scores_by_delay[delay]) for delay in delays}
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("Delay score matrices differ in shape.")
    output = np.empty_like(arrays[0])
    seen = np.zeros(shape[0], dtype=bool)
    for group in groups:
        for offset, position in enumerate(group):
            if seen[position]:
                raise ValueError("Groups overlap.")
            seen[position] = True
            delay = min(delays[-1], len(group) - 1 - offset)
            output[position] = arrays[delay][position]
    if not seen.all():
        raise ValueError("Groups do not cover every event.")
    return output


def candidate_rows(scores, width):
    """Return neural top-K IDs and full-retrieval-set log probabilities."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or not np.isfinite(scores).all() or not 1 <= width <= scores.shape[1]:
        raise ValueError("Invalid neural scores or candidate width.")
    ids = np.argsort(-scores, axis=1, kind="stable")[:, :width].astype(np.int32)
    logits = torch.from_numpy(scores)
    log_probs = F.log_softmax(logits, dim=1).numpy()
    return ids, np.take_along_axis(log_probs, ids, axis=1)


@dataclass(frozen=True)
class Hypothesis:
    words: tuple[str, ...]
    token_ids: tuple[int, ...]
    score: float


class CausalLMScorer:
    """Score appended words by teacher forcing a local causal LM in batches."""

    def __init__(self, model, tokenizer, *, device="cuda", batch_size=64,
                 bos_token_id=None, word_separator="", first_subtoken_only=False,
                 max_context_words=None, score_cache_size=0):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        start_id = (bos_token_id if bos_token_id is not None else
                    tokenizer.bos_token_id if tokenizer.bos_token_id is not None else
                    tokenizer.cls_token_id)
        if start_id is None:
            raise ValueError("A causal language model needs a BOS or CLS token.")
        self.bos_token_id = int(start_id)
        padding_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                      else tokenizer.eos_token_id)
        self.pad_token_id = int(padding_id if padding_id is not None else start_id)
        self.word_separator = str(word_separator)
        self.first_subtoken_only = bool(first_subtoken_only)
        self.max_context_words = (None if max_context_words is None
                                  else int(max_context_words))
        self._word_cache = {}
        self.score_cache_size = int(score_cache_size)
        self._score_cache = OrderedDict()
        self._proposal_cache = OrderedDict()
        if self.batch_size < 1 or (self.max_context_words is not None
                                   and self.max_context_words < 1) or self.score_cache_size < 0:
            raise ValueError("Batch size and context size must be positive.")

    def word_ids(self, word, *, initial=False):
        word = str(word)
        key = (word, bool(initial))
        if key not in self._word_cache:
            text = word if initial else self.word_separator + word
            ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
            if not ids:
                raise ValueError(f"Language tokenizer produced no tokens for {word!r}.")
            self._word_cache[key] = ids
        return self._word_cache[key]

    @torch.inference_mode()
    def score_extensions(self, prefixes, words):
        if len(prefixes) != len(words):
            raise ValueError("Prefix and proposal counts differ.")
        if not self.score_cache_size:
            return self._score_extensions_raw(prefixes, words)
        output = np.empty(len(words), dtype=np.float64)
        missing = {}
        for index, (prefix, word) in enumerate(zip(prefixes, words)):
            key = (tuple(prefix), str(word))
            if key in self._score_cache:
                output[index] = self._score_cache[key]
                self._score_cache.move_to_end(key)
            else:
                missing.setdefault(key, []).append(index)
        if missing:
            keys = list(missing)
            values = self._score_extensions_raw(
                [key[0] for key in keys], [key[1] for key in keys])
            for key, value in zip(keys, values):
                for index in missing[key]:
                    output[index] = value
                self._score_cache[key] = float(value)
                if len(self._score_cache) > self.score_cache_size:
                    self._score_cache.popitem(last=False)
        return output

    @torch.inference_mode()
    def _score_extensions_raw(self, prefixes, words):
        output = np.empty(len(words), dtype=np.float64)
        for start in range(0, len(words), self.batch_size):
            end = min(start + self.batch_size, len(words))
            sequences = []
            suffix_starts = []
            for prefix, word in zip(prefixes[start:end], words[start:end]):
                prefix = tuple(prefix)
                suffix = self.word_ids(word, initial=not prefix)
                if self.first_subtoken_only:
                    suffix = suffix[:1]
                sequences.append((self.bos_token_id,) + prefix + suffix)
                suffix_starts.append(1 + len(prefix))
            length = max(map(len, sequences))
            ids = torch.full((len(sequences), length), self.pad_token_id,
                             dtype=torch.long, device=self.device)
            mask = torch.zeros_like(ids)
            for row, sequence in enumerate(sequences):
                ids[row, :len(sequence)] = torch.tensor(sequence, device=self.device)
                mask[row, :len(sequence)] = 1
            logits = self.model(input_ids=ids, attention_mask=mask).logits[:, :-1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            for row, sequence in enumerate(sequences):
                positions = torch.arange(suffix_starts[row], len(sequence), device=self.device)
                if self.first_subtoken_only:
                    positions = positions[:1]
                targets = ids[row, positions]
                output[start + row] = log_probs[row, positions - 1, targets].sum().item()
        return output

    @torch.inference_mode()
    def next_token_topk(self, prefix, width=64):
        ids = torch.tensor([(self.bos_token_id,) + tuple(prefix)],
                           dtype=torch.long, device=self.device)
        logits = self.model(input_ids=ids).logits[0, -1]
        return set(torch.topk(logits, min(width, len(logits))).indices.tolist())

    @torch.inference_mode()
    def next_word_candidates(self, prefix, width=5, scan_width=128):
        """Mirror the author's single-token open-vocabulary OOV proposal."""
        key = (tuple(prefix), int(width), int(scan_width))
        if self.score_cache_size and key in self._proposal_cache:
            self._proposal_cache.move_to_end(key)
            return self._proposal_cache[key]
        proposals = self._next_word_candidates_raw(prefix, width, scan_width)
        if self.score_cache_size:
            self._proposal_cache[key] = proposals
            if len(self._proposal_cache) > self.score_cache_size:
                self._proposal_cache.popitem(last=False)
        return proposals

    @torch.inference_mode()
    def _next_word_candidates_raw(self, prefix, width, scan_width):
        ids = torch.tensor([(self.bos_token_id,) + tuple(prefix)],
                           dtype=torch.long, device=self.device)
        logits = self.model(input_ids=ids).logits[0, -1].float()
        log_probs = F.log_softmax(logits, dim=-1)
        values, indices = torch.topk(log_probs, min(scan_width, len(log_probs)))
        proposals = []
        seen = set()
        for value, index in zip(values.tolist(), indices.tolist()):
            word = self.tokenizer.decode([int(index)]).strip().lower()
            if (word and word not in seen and any(char.isalpha() for char in word)
                    and not any(char in punctuation for char in word)
                    and not any(char.isspace() for char in word)):
                proposals.append((word, float(value)))
                seen.add(word)
            if len(proposals) >= width:
                break
        if not proposals:
            raise ValueError("Language model proposed no valid OOV word.")
        return proposals


def oov_features(scores, *, include_probabilities=False):
    """Prediction-distribution features available without reference words."""
    values = torch.as_tensor(np.asarray(scores), dtype=torch.float32)
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise ValueError("OOV scores must be a finite matrix.")
    probabilities = F.softmax(values, dim=1).numpy()
    sorted_top = np.sort(probabilities, axis=1)[:, -2:]
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-20))).sum(axis=1)
    statistics = np.column_stack((sorted_top[:, -1],
                                  sorted_top[:, -1] - sorted_top[:, -2], entropy))
    return (np.column_stack((probabilities, statistics))
            if include_probabilities else statistics)


def fill_detected_oov(words, flags, scorer, filler_words, *, first_token_width=64,
                      shortlist_size=32):
    """Causal lexicon-constrained LM completion for detector-marked positions.

    The lexicon is independent of validation labels. This is an approximation
    to Jayalath's generative OOV completion, not an open-vocabulary claim.
    """
    if len(words) != len(flags):
        raise ValueError("Word and OOV flag counts differ.")
    filler_words = tuple(map(str, filler_words))
    for word in filler_words:
        scorer.word_ids(word)
    output = []
    prefix = ()
    for word, flag in zip(words, flags):
        replacement = str(word)
        if flag:
            top_tokens = scorer.next_token_topk(prefix, width=first_token_width)
            proposals = [candidate for candidate in filler_words
                         if scorer.word_ids(candidate, initial=not output)[0] in top_tokens][:shortlist_size]
            if not proposals:
                proposals = list(filler_words[:shortlist_size])
            probabilities = scorer.score_extensions([prefix] * len(proposals), proposals)
            # Length normalization avoids always choosing a one-character completion.
            normalized = [
                score if getattr(scorer, "first_subtoken_only", False)
                else score / len(scorer.word_ids(candidate, initial=not output)) ** 0.5
                for score, candidate in zip(probabilities, proposals)
            ]
            replacement = proposals[int(np.argmax(normalized))]
        output.append(replacement)
        prefix = _context_token_ids(output, scorer)
    return tuple(output)


def _context_token_ids(words, scorer):
    limit = getattr(scorer, "max_context_words", None)
    selected = words[-limit:] if limit is not None else words
    return sum((scorer.word_ids(word, initial=index == 0)
                for index, word in enumerate(selected)), ())


def beam_decode(candidate_ids, candidate_log_probs, vocabulary, scorer, *,
                beam_width=5, language_weight=1.5, oov_flags=None,
                frozen_prefix=()):
    """Jayalath-style neural likelihood plus causal LM beam rescoring."""
    candidate_ids = np.asarray(candidate_ids)
    candidate_log_probs = np.asarray(candidate_log_probs)
    vocabulary = tuple(vocabulary)
    if candidate_ids.shape != candidate_log_probs.shape or candidate_ids.ndim != 2:
        raise ValueError("Candidate arrays must be aligned matrices.")
    if beam_width < 1 or language_weight < 0:
        raise ValueError("Invalid beam or language weight.")
    flags = (np.zeros(candidate_ids.shape[0], dtype=bool) if oov_flags is None
             else np.asarray(oov_flags, dtype=bool))
    if flags.shape != (candidate_ids.shape[0],):
        raise ValueError("OOV flags must match sequence positions.")
    frozen_prefix = tuple(map(str, frozen_prefix))
    beams = [Hypothesis(frozen_prefix, _context_token_ids(frozen_prefix, scorer), 0.0)]
    for position, (row_ids, row_scores) in enumerate(zip(candidate_ids, candidate_log_probs)):
        if flags[position]:
            expanded = []
            for beam in beams:
                for word, lm_log_prob in scorer.next_word_candidates(
                        beam.token_ids, width=beam_width):
                    new_words = beam.words + (word,)
                    expanded.append(Hypothesis(
                        new_words, _context_token_ids(new_words, scorer),
                        beam.score + language_weight * lm_log_prob,
                    ))
            beams = sorted(expanded, key=lambda x: (-x.score, x.words))[:beam_width]
            continue
        prefixes, proposals, neural_scores, parent = [], [], [], []
        for index, beam in enumerate(beams):
            for word_id, log_prob in zip(row_ids, row_scores):
                prefixes.append(beam.token_ids)
                proposals.append(str(vocabulary[int(word_id)]))
                neural_scores.append(float(log_prob))
                parent.append(index)
        lm_scores = scorer.score_extensions(prefixes, proposals)
        expanded = []
        for i, word in enumerate(proposals):
            before = beams[parent[i]]
            new_words = before.words + (word,)
            expanded.append(Hypothesis(
                new_words,
                _context_token_ids(new_words, scorer),
                before.score + neural_scores[i] + language_weight * float(lm_scores[i]),
            ))
        beams = sorted(expanded, key=lambda x: (-x.score, x.words))[:beam_width]
    return beams[0].words


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, target in enumerate(reference, 1):
        current = [i]
        for j, proposed in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (target != proposed)))
        previous = current
    return previous[-1]


def sentence_metrics(references, predictions, *, chinese=False):
    """Corpus WER/CER/BLEU-1 and ROUGE-1 F1 with fixed aligned segments."""
    if len(references) != len(predictions) or not references:
        raise ValueError("References and predictions must be nonempty and paired.")
    word_errors = word_total = char_errors = char_total = 0
    clipped = predicted_total = reference_total = 0
    recall_overlaps = 0
    sentence_bleu1 = []
    for reference, prediction in zip(references, predictions):
        reference = tuple(map(str, reference))
        prediction = tuple(map(str, prediction))
        word_errors += edit_distance(reference, prediction)
        word_total += len(reference)
        ref_text = "".join(reference) if chinese else " ".join(reference)
        pred_text = "".join(prediction) if chinese else " ".join(prediction)
        char_errors += edit_distance(ref_text, pred_text)
        char_total += len(ref_text)
        ref_counts, pred_counts = Counter(reference), Counter(prediction)
        overlap = sum(min(count, pred_counts[word]) for word, count in ref_counts.items())
        sentence_precision = overlap / len(prediction) if prediction else 0.0
        sentence_brevity = (min(1.0, np.exp(1 - len(reference) / len(prediction)))
                            if prediction else 0.0)
        sentence_bleu1.append(float(sentence_precision * sentence_brevity))
        clipped += overlap
        recall_overlaps += overlap
        predicted_total += len(prediction)
        reference_total += len(reference)
    precision = clipped / predicted_total if predicted_total else 0.0
    brevity = min(1.0, np.exp(1 - reference_total / predicted_total)) if predicted_total else 0.0
    rouge_precision = recall_overlaps / predicted_total if predicted_total else 0.0
    rouge_recall = recall_overlaps / reference_total if reference_total else 0.0
    rouge_f1 = (2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
                if rouge_precision + rouge_recall else 0.0)
    return {
        "groups": len(references), "reference_words": word_total,
        "wer": word_errors / word_total, "cer": char_errors / char_total,
        "bleu1": float(brevity * precision),
        "sentence_bleu1_mean": float(np.mean(sentence_bleu1)),
        "rouge1_f1": float(rouge_f1),
        "exact_word_accuracy": sum(a == b for ref, pred in zip(references, predictions)
                                   for a, b in zip(ref, pred)) / word_total,
    }


def grouped_calibration_mask(references, fraction=0.2):
    """Keep repeated stimuli together across subjects when splitting groups."""
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must be between zero and one.")
    return np.asarray([
        int.from_bytes(sha256("\x1f".join(map(str, words)).encode()).digest()[:8], "big")
        / 2**64 < fraction
        for words in references
    ], dtype=bool)


def shift_horizon_scores(scores, subject_ids, availability_masks, *, seed=1729):
    """Derange each subject and availability stratum once across all horizons."""
    from braindecoding.evaluation.retrieval_metrics import within_subject_permutation
    subjects = np.asarray(subject_ids, dtype=object)
    last_available = np.zeros(len(subjects), dtype=np.int8)
    for delay, mask in availability_masks.items():
        if len(mask) != len(subjects):
            raise ValueError("Horizon mask length differs from score rows.")
        available = np.asarray(mask, dtype=bool)
        last_available[available] = np.maximum(last_available[available], delay)
    strata = np.asarray([f"{subject}\x1f{last_available[index]}"
                         for index, subject in enumerate(subjects)], dtype=object)
    permutation = within_subject_permutation(strata, np.random.default_rng(seed))
    return {delay: matrix[permutation] for delay, matrix in scores.items()}, permutation

# --- Merged from jayalath_rolling.py ---
"""Causal, bounded-revision display replay for Jayalath-style sentence search."""


from time import perf_counter

import numpy as np



def decode_rolling_group(candidate_ids, candidate_log_probs, vocabulary, scorer, *,
                         beam_width, language_weight, reference=None,
                         oov_flags=None, maximum_delay=2):
    """Replay the frozen K0 and mutable K0/K1/K2 displays at each word arrival.

    Each delay matrix has one row per target word. At arrival t, only the rows
    (target=t-delay, horizon=delay) are read. Older displayed words are frozen.
    """
    delays = tuple(sorted(candidate_ids))
    if delays != tuple(range(maximum_delay + 1)) or delays != tuple(sorted(candidate_log_probs)):
        raise ValueError("Expected aligned consecutive delays from K0.")
    count = len(candidate_ids[0])
    width = candidate_ids[0].shape[1]
    if any(candidate_ids[d].shape != (count, width)
           or candidate_log_probs[d].shape != (count, width) for d in delays):
        raise ValueError("Candidate matrices have inconsistent shapes.")
    if reference is not None and len(reference) != count:
        raise ValueError("Reference length differs from candidate rows.")
    if oov_flags is not None and any(np.asarray(oov_flags[d]).shape != (count,) for d in delays):
        raise ValueError("OOV flags differ from candidate rows.")

    frozen, rolling = (), ()
    trace = []
    transitions = []
    timings = {"k0_freeze_seconds": 0.0, "k2_revision_seconds": 0.0}
    for arrival in range(count):
        start = perf_counter()
        frozen = beam_decode(
            candidate_ids[0][arrival:arrival + 1],
            candidate_log_probs[0][arrival:arrival + 1],
            vocabulary, scorer, beam_width=beam_width,
            language_weight=language_weight,
            frozen_prefix=frozen,
            oov_flags=None if oov_flags is None else oov_flags[0][arrival:arrival + 1],
        )
        freeze_seconds = perf_counter() - start
        timings["k0_freeze_seconds"] += freeze_seconds

        mutable_start = max(0, arrival - maximum_delay)
        prefix = rolling[:mutable_start]
        targets = range(mutable_start, arrival + 1)
        ids = np.stack([candidate_ids[arrival - position][position]
                        for position in targets])
        log_probs = np.stack([candidate_log_probs[arrival - position][position]
                              for position in targets])
        flags = (None if oov_flags is None else
                 np.asarray([oov_flags[arrival - position][position]
                             for position in targets], dtype=bool))
        start = perf_counter()
        updated = beam_decode(
            ids, log_probs, vocabulary, scorer, beam_width=beam_width,
            language_weight=language_weight, frozen_prefix=prefix, oov_flags=flags,
        )
        revision_seconds = perf_counter() - start
        timings["k2_revision_seconds"] += revision_seconds
        if updated[:mutable_start] != prefix or len(updated) != arrival + 1:
            raise RuntimeError("Rolling decoder modified a frozen word.")
        for position in range(mutable_start, arrival):
            before, after = rolling[position], updated[position]
            if before != after:
                transition = {"arrival_index": arrival, "word_index": position,
                              "from": before, "to": after}
                if reference is not None:
                    truth = str(reference[position])
                    transition["kind"] = (
                        "wrong_to_right" if after == truth else
                        "right_to_wrong" if before == truth else "wrong_to_wrong"
                    )
                transitions.append(transition)
        rolling = updated
        newly_frozen = ([arrival - maximum_delay]
                        if arrival >= maximum_delay else [])
        if arrival == count - 1:
            newly_frozen = list(range(max(0, arrival - maximum_delay), count))
        trace.append({"arrival_index": arrival,
                      "k0_frozen": list(frozen), "k2_rolling": list(rolling),
                      "k0_decoder_seconds": freeze_seconds,
                      "k2_decoder_seconds": revision_seconds,
                      "newly_frozen_word_indices": newly_frozen})
    return {"k0_frozen": frozen, "k2_rolling": rolling,
            "trace": trace, "transitions": transitions,
            "decoder_seconds": timings}
