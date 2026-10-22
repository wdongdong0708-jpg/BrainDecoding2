"""材料块配对置换检验、分层 Bootstrap 置信区间与 Holm 多重检验校正接口。"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256

import numpy as np


METRICS = ("sentence_bleu1_mean", "wer", "bleu1", "aligned_word_accuracy")


def edit_distance(reference, prediction):
    """计算两个序列之间的标准 Levenshtein 编辑距离。"""
    previous = list(range(len(prediction) + 1))
    for i, target in enumerate(reference, 1):
        current = [i]
        for j, proposed in enumerate(prediction, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (target != proposed),
                )
            )
        previous = current
    return previous[-1]


def sentence_metrics(references, predictions, *, chinese=False):
    """计算语料级与句段级 WER / CER / BLEU-1 / ROUGE-1 F1 与对齐词位准确率。"""
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
        sentence_brevity = (
            min(1.0, np.exp(1 - len(reference) / len(prediction)))
            if prediction
            else 0.0
        )
        sentence_bleu1.append(float(sentence_precision * sentence_brevity))
        clipped += overlap
        recall_overlaps += overlap
        predicted_total += len(prediction)
        reference_total += len(reference)
    precision = clipped / predicted_total if predicted_total else 0.0
    brevity = (
        min(1.0, np.exp(1 - reference_total / predicted_total))
        if predicted_total
        else 0.0
    )
    rouge_precision = recall_overlaps / predicted_total if predicted_total else 0.0
    rouge_recall = recall_overlaps / reference_total if reference_total else 0.0
    rouge_f1 = (
        2 * rouge_precision * rouge_recall / (rouge_precision + rouge_recall)
        if rouge_precision + rouge_recall
        else 0.0
    )
    return {
        "groups": len(references),
        "reference_words": word_total,
        "wer": word_errors / word_total if word_total else 0.0,
        "cer": char_errors / char_total if char_total else 0.0,
        "bleu1": float(brevity * precision),
        "sentence_bleu1_mean": float(np.mean(sentence_bleu1)),
        "rouge1_f1": float(rouge_f1),
        "exact_word_accuracy": (
            sum(
                a == b
                for ref, pred in zip(references, predictions)
                for a, b in zip(ref, pred)
            )
            / word_total
            if word_total
            else 0.0
        ),
    }


def metric_row(reference, prediction, chinese):
    """计算单句段的四项基础指标行。"""
    metrics = sentence_metrics([reference], [prediction], chinese=chinese)
    metrics["aligned_word_accuracy"] = (
        sum(a == b for a, b in zip(reference, prediction)) / len(reference)
        if reference
        else 0.0
    )
    return {k: metrics[k] for k in METRICS}


def holm(pvalues):
    """执行保序 Holm-Bonferroni 多重比较校正并返回 numpy 数组。"""
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.zeros(len(order))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(
            previous, min(1.0, float(pvalues[index]) * (len(order) - rank))
        )
        adjusted[index] = previous
    return adjusted


def paired_block_statistics(rows, metric, statistic_settings):
    """按共享刺激材料块执行配对符号翻转置换检验与分层 Bootstrap 置信区间估计。"""
    blocks = sorted({r["block"] for r in rows})
    cells = sorted({(r["seed"], r["subject"]) for r in rows})
    bidx = {b: i for i, b in enumerate(blocks)}
    cidx = {c: i for i, c in enumerate(cells)}
    sums = np.zeros((len(blocks), len(cells)))
    counts = np.zeros_like(sums)
    scope = {}
    for row in rows:
        i, j = bidx[row["block"]], cidx[(row["seed"], row["subject"])]
        sums[i, j] += row[metric]
        counts[i, j] += 1
        scope[row["block"]] = row["scope"]
    if (counts.sum(0) == 0).any():
        raise ValueError("Empty statistical subject/seed cell")
    seeds = sorted({c[0] for c in cells})
    if len({sum(c[0] == s for c in cells) for s in seeds}) != 1:
        raise ValueError("Inconsistent subjects across seeds")
    contributions = (sums / counts.sum(0)[None, :]).mean(1)
    observed = float(contributions.sum())
    rng = np.random.default_rng(statistic_settings["seed"])
    repetitions = statistic_settings["paired_material_block_sign_flips"]
    if len(blocks) <= 16:
        integers = np.arange(2 ** len(blocks), dtype=np.uint32)
        signs = (
            2
            * (
                (integers[:, None] >> np.arange(len(blocks), dtype=np.uint32)) & 1
            ).astype(float)
            - 1
        )
        null = signs @ contributions
        pvalue = float(np.mean(null >= observed - 1e-12))
        exact = True
    else:
        signs = rng.choice([-1.0, 1.0], (repetitions, len(blocks)))
        null = signs @ contributions
        pvalue = float(
            (1 + np.count_nonzero(null >= observed - 1e-12)) / (repetitions + 1)
        )
        exact = False
    strata = defaultdict(list)
    for block, i in bidx.items():
        strata[scope[block]].append(i)
    draws = []
    bootstrap_reps = statistic_settings["material_block_bootstrap_repetitions"]
    for start in range(0, bootstrap_reps, 1000):
        batch = min(1000, bootstrap_reps - start)
        weights = np.zeros((batch, len(blocks)))
        for indices in strata.values():
            picks = rng.choice(indices, (batch, len(indices)), replace=True)
            for i in range(batch):
                weights[i] += np.bincount(picks[i], minlength=len(blocks))
        denominator = weights @ counts
        numerator = weights @ sums
        valid = (denominator > 0).all(1)
        draws.extend((numerator[valid] / denominator[valid]).mean(1).tolist())
    if len(draws) < bootstrap_reps * 0.95:
        raise ValueError("Too many empty-subject material bootstrap samples")
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return dict(
        effect=observed,
        ci_lower=float(lo),
        ci_upper=float(hi),
        p_one_sided=pvalue,
        blocks=len(blocks),
        subjects=len(cells) // len(seeds),
        seeds=len(seeds),
        sign_flip_exact=exact,
        bootstrap_samples=len(draws),
    )


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


def paired_cluster_test(
    cluster_counts,
    *,
    permutations=20000,
    bootstraps=5000,
    seed=20260923,
    denominator_label="reference_words",
):
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
