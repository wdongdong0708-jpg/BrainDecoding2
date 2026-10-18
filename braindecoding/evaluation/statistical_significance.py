"""材料块配对置换检验、分层 Bootstrap 置信区间与 Holm 多重检验校正接口。"""
from __future__ import annotations

from braindecoding.evaluation.unified_sentence_v2_results import (
    METRICS,
    holm,
    metric_row,
    paired_block_statistics,
)

# --- Merged from jayalath_significance.py ---
"""Cluster-aware paired chance-baseline analysis for aligned sentence outputs."""


from collections import Counter, defaultdict
from hashlib import sha256

import numpy as np


def clipped_unigram_hits(reference, prediction):
    reference = Counter(map(str, reference))
    prediction = Counter(map(str, prediction))
    return sum(min(count, prediction[word]) for word, count in reference.items())


def cluster_bleu_differences(rows, actual_key, control_key):
    """Aggregate paired BLEU-1 numerator differences by unique stimulus text."""
    clusters = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        reference = tuple(map(str, row["reference"]))
        actual = tuple(map(str, row[actual_key]))
        control = tuple(map(str, row[control_key]))
        if not reference or len(actual) != len(reference) or len(control) != len(reference):
            raise ValueError("Paired outputs must match the known reference word count.")
        key = sha256("\x1f".join(reference).encode()).hexdigest()
        cluster = clusters[key]
        cluster[0] += clipped_unigram_hits(reference, actual)
        cluster[1] += clipped_unigram_hits(reference, control)
        cluster[2] += len(reference)
    return np.asarray(list(clusters.values()), dtype=np.int64)


def cluster_sentence_bleu_differences(rows, actual_key, control_key):
    """Aggregate per-sequence BLEU-1 by repeated stimulus, as in the author code."""
    clusters = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in rows:
        reference = tuple(map(str, row["reference"]))
        actual = tuple(map(str, row[actual_key]))
        control = tuple(map(str, row[control_key]))
        if not reference or len(actual) != len(reference) or len(control) != len(reference):
            raise ValueError("Paired outputs must match the known reference word count.")
        key = sha256("\x1f".join(reference).encode()).hexdigest()
        cluster = clusters[key]
        cluster[0] += clipped_unigram_hits(reference, actual) / len(reference)
        cluster[1] += clipped_unigram_hits(reference, control) / len(reference)
        cluster[2] += 1.0
    return np.asarray(list(clusters.values()), dtype=np.float64)


def paired_cluster_test(cluster_counts, *, permutations=20000, bootstraps=5000,
                        seed=20260923, denominator_label="reference_words"):
    """One-sided paired sign-flip test and two-sided cluster bootstrap CI."""
    counts = np.asarray(cluster_counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 3 or len(counts) < 2:
        raise ValueError("Expected at least two stimulus clusters with [real, control, words].")
    if permutations < 1 or bootstraps < 1 or np.any(counts[:, 2] <= 0):
        raise ValueError("Invalid resampling settings or cluster word counts.")
    effects = counts[:, 0] - counts[:, 1]
    observed = float(effects.sum() / counts[:, 2].sum())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        signs = rng.choice((-1, 1), size=len(effects))
        extreme += int(np.dot(signs, effects) >= effects.sum())
    samples = np.empty(bootstraps, dtype=np.float64)
    for i in range(bootstraps):
        indices = rng.integers(0, len(counts), size=len(counts))
        samples[i] = effects[indices].sum() / counts[indices, 2].sum()
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "unique_stimulus_clusters": len(counts),
        denominator_label: int(counts[:, 2].sum()),
        "real_bleu1": float(counts[:, 0].sum() / counts[:, 2].sum()),
        "control_bleu1": float(counts[:, 1].sum() / counts[:, 2].sum()),
        "delta_bleu1": observed,
        "ci95_cluster_bootstrap": [float(low), float(high)],
        "one_sided_sign_flip_p": float((extreme + 1) / (permutations + 1)),
        "permutations": permutations,
        "bootstraps": bootstraps,
    }


def holm_adjust(p_values):
    """Familywise Holm adjustment in original comparison order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.all((0 <= values) & (values <= 1)):
        raise ValueError("P values must be a 1D probability vector.")
    adjusted = np.empty_like(values)
    current = 0.0
    for rank, index in enumerate(np.argsort(values)):
        current = max(current, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, current)
    return adjusted.tolist()
