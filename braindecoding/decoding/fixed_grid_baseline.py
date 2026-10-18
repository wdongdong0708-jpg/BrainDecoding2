"""固定时间网格诊断；不读取词起点、词数或语言模型。"""
from collections import Counter
import hashlib
import math

import numpy as np


def fixed_groups(start_seconds, end_seconds, recording_origin_seconds,
                 signal_samples, group_windows=5):
    """每次整组滑移 .15 s；组内 .30 s，末两窗只作共同评价的后续输入。"""
    if not (math.isfinite(start_seconds) and math.isfinite(end_seconds)
            and start_seconds < end_seconds and group_windows >= 3):
        raise ValueError("Invalid sentence boundary or fixed group length")
    # All times use integer half-ticks; samples are rounded from absolute time.
    count = int(math.ceil((end_seconds - start_seconds) / .15 - 1e-9))
    origin_ticks = np.arange(count, dtype=np.int64)
    ticks = origin_ticks[:, None] + 2 * np.arange(group_windows, dtype=np.int64)
    starts = np.floor((start_seconds - recording_origin_seconds) * 50
                      + ticks * 7.5 + .5 + 1e-9).astype(np.int64)
    if starts.min() < 0 or starts.max() + 50 > signal_samples:
        raise ValueError("A fixed group requires unavailable continuous signal; no padding allowed")
    scored = (np.arange(group_windows)[None, :] < group_windows - 2) & (ticks < count)
    assert scored.any(axis=1).all() and np.all(np.diff(starts, axis=1) == 15)
    assert np.array_equal(np.unique(ticks[scored]), origin_ticks)
    return dict(ticks=ticks, starts=starts, scored=scored, count=count)


def continuous_gaussian(shape, means, stds, noise_seed, recording):
    """同一记录只有一条随机流；窗口和模型不会进入随机种子。"""
    tag = f"fixed_grid_continuous_gaussian_v1|{noise_seed}|{recording}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(tag).digest()[:16], "little")
    generator = np.random.default_rng(seed)
    stream = generator.standard_normal(shape, dtype=np.float32)
    stream *= np.asarray(stds, dtype=np.float32)[:, None]
    stream += np.asarray(means, dtype=np.float32)[:, None]
    return stream


def input_windows(stream, starts):
    """两条件共用训练时的逐窗 baseline/clamp；连续性在此变换之前定义。"""
    starts = np.asarray(starts, dtype=np.int64)
    if starts.ndim != 1 or not len(starts) or starts.min() < 0 or starts.max() + 50 > stream.shape[1]:
        raise ValueError("Unavailable raw input window")
    windows = np.stack([stream[:, s:s + 50] for s in starts]).astype(np.float32)
    windows -= windows[:, :, :25].mean(axis=-1, keepdims=True)
    np.clip(windows, -5., 5., out=windows)
    if not np.isfinite(windows).all():
        raise ValueError("Nonfinite model input")
    return windows


def neural_probabilities(embeddings, bank):
    """沿用神经相似度乘数 10；完整词库上的 float64 softmax，无 Qwen。"""
    scores = np.asarray(embeddings, dtype=np.float64) @ np.asarray(bank, dtype=np.float64).T
    scores *= 10.
    scores -= scores.max(axis=1, keepdims=True)
    np.exp(scores, out=scores)
    scores /= scores.sum(axis=1, keepdims=True)
    assert np.isfinite(scores).all()
    return scores.astype(np.float32)


def average_same_times(probabilities, ticks, ready_samples, count):
    """只平均同一名义预测时刻，不混合相邻时刻。"""
    probabilities = np.asarray(probabilities)
    ticks = np.asarray(ticks, dtype=np.int64)
    ready_samples = np.asarray(ready_samples, dtype=np.int64)
    if probabilities.ndim != 2 or len(probabilities) != len(ticks) or len(ticks) != len(ready_samples):
        raise ValueError("Inconsistent raw probability metadata")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Invalid raw probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=2e-6, rtol=0):
        raise ValueError("Raw rows must be normalized full-vocabulary probabilities")
    totals = np.zeros((count, probabilities.shape[1]), dtype=np.float64)
    counts = np.zeros(count, dtype=np.int64)
    ready = np.zeros(count, dtype=np.int64)
    np.add.at(totals, ticks, probabilities)
    np.add.at(counts, ticks, 1)
    np.maximum.at(ready, ticks, ready_samples)
    if not counts.min() > 0:
        raise ValueError("Missing prediction time; all stages must share a timeline")
    totals /= counts[:, None]
    return totals, counts, ready


def merge_probabilities(probabilities, ready_samples, *, smoothing):
    """valid 平均池化 (5,3) 或无时间池化，然后 argmax、合并相邻重复。"""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    ready_samples = np.asarray(ready_samples, dtype=np.int64)
    n = len(probabilities)
    if n == 0 or len(ready_samples) != n:
        raise ValueError("Empty or mismatched timeline")
    short_pool = bool(smoothing and n < 5)
    if smoothing:
        spans = [(s, s + 5) for s in range(0, n - 4, 3)] if n >= 5 else [(0, n)]
        values = np.stack([probabilities[a:b].mean(axis=0) for a, b in spans])
        ready = [int(ready_samples[a:b].max()) for a, b in spans]
        # Pooled predictions are assigned to their centre, not their first input.
        centres = [(a + b - 1) / 2 for a, b in spans]
    else:
        spans = [(s, s + 1) for s in range(n)]
        values, ready, centres = probabilities, ready_samples.tolist(), list(range(n))
    ids = values.argmax(axis=1).astype(int).tolist()
    keep = [i for i, value in enumerate(ids) if i == 0 or value != ids[i - 1]]
    return dict(word_ids=[ids[i] for i in keep],
                word_ready_samples=[ready[i] for i in keep],
                word_centre_half_ticks=[centres[i] for i in keep],
                pooled_word_ids=ids, pooled_ready_samples=ready,
                pool_spans=[list(x) for x in spans],
                short_pool=short_pool, output_tokens=len(keep),
                mode="jayalath_valid_pool_5_3" if smoothing else "no_temporal_pool")


def edit_distance(reference, prediction):
    row = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        following = [i]
        for j, right in enumerate(prediction, 1):
            following.append(min(following[-1] + 1, row[j] + 1,
                                 row[j - 1] + (left != right)))
        row = following
    return row[-1]


def sentence_counts(reference, prediction):
    """CER 含词间单空格；不还原省音，不删重音，保持冻结的参考词单位。"""
    if not reference:
        raise ValueError("Empty reference sentence")
    overlap = sum((Counter(reference) & Counter(prediction)).values())
    ref_text, hyp_text = " ".join(reference), " ".join(prediction)
    precision = overlap / len(prediction) if prediction else 0.
    brevity = math.exp(min(0., 1 - len(reference) / len(prediction))) if prediction else 0.
    return dict(reference_words=len(reference), prediction_words=len(prediction),
                word_errors=edit_distance(reference, prediction),
                reference_characters=len(ref_text), character_errors=edit_distance(ref_text, hyp_text),
                unigram_overlap=overlap, sentence_bleu1=precision * brevity)


def aggregate_counts(rows):
    counts = {k: int(sum(row[k] for row in rows)) for k in (
        "reference_words", "prediction_words", "word_errors", "reference_characters",
        "character_errors", "unigram_overlap")}
    ref, hyp = counts["reference_words"], counts["prediction_words"]
    counts.update(wer=counts["word_errors"] / ref,
                  cer=counts["character_errors"] / counts["reference_characters"],
                  corpus_bleu1=(counts["unigram_overlap"] / hyp
                    * math.exp(min(0., 1 - ref / hyp))) if hyp else 0.,
                  sentence_bleu1_mean=float(np.mean([r["sentence_bleu1"] for r in rows])),
                  sentences=len(rows))
    return counts

# --- Merged from fixed_grid_fast_inference.py ---
"""只合并逐查询 GPU 张量组装；模型运算、形状和查询批次不变。"""
import types

import numpy as np
import torch
from torch.nn import functional as F


def plain_context(context):
    for key in ('local_relative_position_bias', 'physical_time_gap_bias',
                'event_time_interaction_bias', 'adaptive_directional_bias_gate',
                'hierarchical_attention_radii', 'query_window_left_radius'):
        assert getattr(context, key, None) is None, key
    assert not context.chunk_summary_enabled and not context.local_attention_heads
    assert not context.training


def encode_rows(context, features, rows, *, causal_override=None):
    """生成与原 GroupedTransformerEncoder 完全相同的 padded/mask。"""
    plain_context(context)
    width = max(map(len, rows))
    lookup = np.zeros((len(rows), width), dtype=np.int64)
    valid = np.zeros_like(lookup, dtype=bool)
    for i, positions in enumerate(rows):
        lookup[i, :len(positions)] = positions
        valid[i, :len(positions)] = True
    lookup = torch.as_tensor(lookup, device=features.device)
    mask = torch.as_tensor(valid, device=features.device)
    padded = features[lookup].masked_fill(~mask[:, :, None], 0.)
    encoded = padded
    kwargs = dict(mask=mask, attn_mask=None, attn_bias=None)
    causal = context.causal_attention if causal_override is None else bool(causal_override)
    if causal:
        kwargs['causal'] = True
    for _ in range(context.shared_passes):
        encoded = context.encoder(encoded, **kwargs)
    return encoded, mask


def vectorized_context(self, features, group_indices, event_times=None, causal_override=None):
    assert event_times is None
    groups = {}
    for index, group in enumerate(group_indices.detach().cpu().tolist()):
        groups.setdefault(group, []).append(index)
    rows = list(groups.values())
    encoded, mask = encode_rows(self, features, rows, causal_override=causal_override)
    order = torch.as_tensor([i for positions in rows for i in positions], device=features.device)
    output = torch.empty_like(features)
    output[order] = encoded[mask]
    return output


@torch.inference_mode()
def vectorized_revisions(self, signals, subject_indices, group_indices):
    assert self.prefix_revision_enabled and not self.training
    assert self.prefix_revision_protocol == 'local_window'
    brain = F.normalize(self.brain_encoder(signals, subject_indices), dim=-1)
    groups = self._ordered_groups(group_indices)
    mapping = {row: (positions, offset) for positions in groups for offset, row in enumerate(positions)}
    outputs = {}
    batch = self.prefix_revision_validation_query_batch_size
    for horizon in self.prefix_revision_horizons:
        delayed = torch.empty_like(brain)
        for left in range(0, len(brain), batch):
            right = min(left + batch, len(brain))
            rows, targets = [], []
            for row in range(left, right):
                positions, offset = mapping[row]
                a, b, target = self._revision_window_bounds(offset, horizon, len(positions))
                rows.append(positions[a:b])
                targets.append(target)
            encoded, _ = encode_rows(self.context_transformer, brain, rows, causal_override=False)
            target_rows = torch.arange(len(rows), device=brain.device)
            target_cols = torch.as_tensor(targets, device=brain.device)
            delayed[left:right] = F.normalize(encoded[target_rows, target_cols], dim=-1)
        outputs[horizon] = delayed
    return outputs


def install(model):
    if model.context_transformer is None:
        return model
    plain_context(model.context_transformer)
    assert model.pre_transformer_trajectory is None
    model.context_transformer.forward = types.MethodType(vectorized_context, model.context_transformer)
    if model.prefix_revision_enabled:
        model.forward_prefix_revisions = types.MethodType(vectorized_revisions, model)
    return model
