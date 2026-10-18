"""Closed-vocabulary, bounded-revision sequence decoding for word scores.

This is an independent post-processing experiment. The language prior is a
train-split bigram model; it never reads validation words while fitting.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def fit_bigram_prior(groups, vocabulary, smoothing=0.5):
    """Fit log probabilities from ordered training sequences only."""
    vocabulary = tuple(str(word) for word in vocabulary)
    if not vocabulary or len(vocabulary) != len(set(vocabulary)):
        raise ValueError("Vocabulary must be nonempty and unique.")
    if smoothing <= 0:
        raise ValueError("Smoothing must be positive.")
    index = {word: position for position, word in enumerate(vocabulary)}
    count = len(vocabulary)
    starts = np.full(count, smoothing, dtype=np.float64)
    transitions = np.full((count, count), smoothing, dtype=np.float64)
    observed_groups = 0
    for words in groups:
        words = tuple(str(word) for word in words)
        if not words:
            continue
        observed_groups += 1
        previous = None
        for word in words:
            current = index.get(word)
            if current is not None:
                if previous is None:
                    starts[current] += 1
                else:
                    transitions[previous, current] += 1
            # An out-of-vocabulary token breaks a contiguous bigram.
            previous = current
    if not observed_groups:
        raise ValueError("No nonempty training sequences.")
    return {
        "vocabulary": vocabulary,
        "start": np.log(starts / starts.sum()),
        "transition": np.log(transitions / transitions.sum(axis=1, keepdims=True)),
        "training_group_count": observed_groups,
        "smoothing": float(smoothing),
    }


def cosine_word_scores(predictions, candidate_embeddings, scale=10.0):
    """Score every frozen candidate with the same cosine used by retrieval."""
    predictions = np.asarray(predictions, dtype=np.float32)
    candidates = np.asarray(candidate_embeddings, dtype=np.float32)
    if predictions.ndim != 2 or candidates.ndim != 2:
        raise ValueError("Predictions and candidates must be matrices.")
    if predictions.shape[1] != candidates.shape[1]:
        raise ValueError("Prediction/candidate embedding dimensions differ.")
    if not np.isfinite(predictions).all() or not np.isfinite(candidates).all():
        raise ValueError("Embedding matrices contain non-finite values.")
    if scale <= 0:
        raise ValueError("Neural score scale must be positive.")
    query_norm = np.maximum(np.linalg.norm(predictions, axis=1, keepdims=True), 1e-12)
    candidate_norm = np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12)
    return (predictions / query_norm) @ (candidates / candidate_norm).T * scale


def _best_suffix(emissions, prior, language_weight, previous_word):
    """Exact Viterbi search over the mutable suffix, conditioned on a frozen word."""
    count = emissions.shape[1]
    if previous_word is None:
        values = emissions[0] + language_weight * prior["start"]
    else:
        values = emissions[0] + language_weight * prior["transition"][previous_word]
    backpointers = []
    for row in emissions[1:]:
        choices = values[:, None] + language_weight * prior["transition"]
        pointers = np.argmax(choices, axis=0)
        values = choices[pointers, np.arange(count)] + row
        backpointers.append(pointers)
    last = int(np.argmax(values))
    output = [last]
    for pointers in reversed(backpointers):
        last = int(pointers[last])
        output.append(last)
    output.reverse()
    return output


def decode_rolling_group(scores_by_delay, prior, language_weight):
    """At each arrival, revise only the last R words and freeze older words.

    `scores_by_delay[d]` has one candidate-score row for every word. The score
    for word i at arrival t comes from delay min(t-i, R). A prediction made at
    each exact delay is retained separately for paired evaluation.
    """
    delays = tuple(sorted(int(value) for value in scores_by_delay))
    if delays != tuple(range(len(delays))):
        raise ValueError("Rolling delays must be consecutive from zero.")
    if language_weight < 0:
        raise ValueError("Language weight must be nonnegative.")
    matrices = {delay: np.asarray(scores_by_delay[delay], dtype=np.float64) for delay in delays}
    shape = matrices[0].shape
    if len(shape) != 2 or shape[1] != len(prior["vocabulary"]):
        raise ValueError("Candidate score shape disagrees with the vocabulary.")
    if any(matrix.shape != shape or not np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("Rolling score matrices must share a finite shape.")
    maximum_delay = delays[-1]
    revisions = {delay: np.full(shape[0], -1, dtype=np.int64) for delay in delays}
    current = []
    for arrival in range(shape[0]):
        mutable_start = max(0, arrival - maximum_delay)
        frozen = current[:mutable_start]
        rows = [
            matrices[arrival - position][position]
            for position in range(mutable_start, arrival + 1)
        ]
        suffix = _best_suffix(
            np.stack(rows),
            prior,
            language_weight,
            frozen[-1] if frozen else None,
        )
        current = frozen + suffix
        for delay in delays:
            position = arrival - delay
            if position >= 0:
                revisions[delay][position] = current[position]
    return revisions, np.asarray(current, dtype=np.int64)


def ordered_groups(table, group_column, word_column):
    """Group table rows without changing their chronological order."""
    if group_column not in table or word_column not in table:
        raise KeyError("Missing sequence grouping or word column.")
    groups = defaultdict(list)
    for group, word in zip(table[group_column], table[word_column]):
        groups[str(group)].append(str(word))
    return list(groups.values())


def _accuracy(predictions, truths, vocabulary):
    predictions = np.asarray(predictions, dtype=object)
    truths = np.asarray(truths, dtype=object)
    if predictions.shape != truths.shape or predictions.ndim != 1:
        raise ValueError("Prediction and reference token arrays must align.")
    known = np.isin(truths, tuple(vocabulary))
    correct = predictions == truths
    observed = sorted(set(truths[known]))
    return {
        "events": int(len(truths)),
        "vocabulary_coverage": float(np.mean(known)) if len(truths) else None,
        "all_token_accuracy": float(np.mean(correct)) if len(truths) else None,
        "known_token_accuracy": float(np.mean(correct[known])) if known.any() else None,
        "known_token_macro_accuracy": (
            float(np.mean([np.mean(correct[truths == word]) for word in observed]))
            if observed else None
        ),
    }


def evaluate_rolling_sequences(
    scores_by_delay, truths, groups, prior, language_weights, *, return_details=False
):
    """Paired word and final-sequence metrics on the same frozen groups."""
    delays = tuple(sorted(scores_by_delay))
    maximum_delay = max(delays)
    vocabulary = prior["vocabulary"]
    truths = np.asarray(truths, dtype=object)
    rows = {delay: np.asarray(scores_by_delay[delay], dtype=np.float32) for delay in delays}
    if any(value.shape != (len(truths), len(vocabulary)) for value in rows.values()):
        raise ValueError("Score matrices must match references and vocabulary.")
    if sorted(index for group in groups for index in group) != list(range(len(truths))):
        raise ValueError("Sequence groups must partition the reference events exactly.")
    # Only words with all R subsequent events are compared across K0...KR.
    common = np.asarray(
        [
            index
            for group in groups
            for index in (group[:-maximum_delay] if maximum_delay else group)
        ],
        dtype=np.int64,
    )
    if not len(common):
        raise ValueError("No words have the complete revision horizon.")
    methods = {"greedy": None, "language_only": float("inf")}
    for weight in language_weights:
        weight = float(weight)
        if weight <= 0:
            raise ValueError("Sequence language weights must be positive.")
        methods[f"neural_plus_bigram_lambda_{weight:g}"] = weight
    output = {}
    details = {}
    for method, weight in methods.items():
        per_delay = {
            delay: np.full(len(truths), "", dtype=object) for delay in delays
        }
        final = np.full(len(truths), "", dtype=object)
        exact = []
        for group in groups:
            positions = np.asarray(group, dtype=np.int64)
            if method == "greedy":
                local_revisions = {
                    delay: np.argmax(rows[delay][positions], axis=1)
                    for delay in delays
                }
                local_final = np.asarray([
                    local_revisions[min(maximum_delay, len(group) - 1 - index)][index]
                    for index in range(len(group))
                ])
            else:
                local_scores = {
                    delay: (
                        np.zeros_like(rows[delay][positions])
                        if method == "language_only"
                        else rows[delay][positions]
                    )
                    for delay in delays
                }
                local_revisions, local_final = decode_rolling_group(
                    local_scores, prior,
                    1.0 if method == "language_only" else weight,
                )
            for delay in delays:
                per_delay[delay][positions] = np.asarray(vocabulary, dtype=object)[
                    local_revisions[delay]
                ]
            final[positions] = np.asarray(vocabulary, dtype=object)[local_final]
            exact.append(bool(np.array_equal(final[positions], truths[positions])))
        output[method] = {
            "common_horizons": {
                str(delay): _accuracy(per_delay[delay][common], truths[common], vocabulary)
                for delay in delays
            },
            "final_sequence": {
                **_accuracy(final, truths, vocabulary),
                "exact_group_accuracy": float(np.mean(exact)),
                "group_count": len(groups),
            },
        }
        if return_details:
            details[method] = {"final": final, "per_delay": per_delay}
    return (output, details) if return_details else output
