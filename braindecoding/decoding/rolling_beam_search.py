"""单遍静态束搜索解码器（S0 / K0 / K1 / K2 / F / Brain-Only / LM-Only）。

核心原则与架构：
1. 闭集词表 Top-K (K=20) 候选检索：利用神经余弦相似度矩阵除以状态温度 T_r 后做局部 log_softmax，
   每个词位仅截取神经概率 Top-20 候选词送入语言模型增量打分。
2. 单遍左到右束搜索 (`decode_segment_static` / `decode_segments_static_batched`)：
   - 直接消费每个词位最终神经表征（如 Word-Only S0、Local r=0/1/2 或 Full-Context F），
     经冻结因果语言模型增量 KV-Cache 从左到右一次定稿输出句子；
   - 同步支持零语言模型纯神经贪心基线 (`*_brain_only`) 与纯语言模型基线 (`LM-Only`)。
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F_nn

from braindecoding.decoding.language_model_scoring import (
    CachedVocabularyLanguageScorer,
    PrefixCacheState,
)


@dataclass
class BeamHypothesis:
    """定稿主干束或静态束搜索中的单条假设。"""

    words: tuple[str, ...]
    word_indices: tuple[int, ...]
    cache_state: PrefixCacheState
    score: float
    neural_score_sum: float
    lm_score_sum: float


def compute_neural_log_probabilities(
    cosine_scores: torch.Tensor | np.ndarray,
    temperature: float = 0.07,
    *,
    top_k: int | None = 20,
    allowed_vocab_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """将余弦相似度矩阵 [L, |V|] 按温度 T_r 缩放，并在每个词位的 Top-K 神经候选集内部做局部 Softmax 归一化。

    通过仅在 Top-K (默认 K=20) 候选词内部计算配分函数 log Z_t = logsumexp_{w in Top-K}(s(w)/T_r)，
    滤除其余 ~2,000 个无关背景词的分母稀释，确保神经对数概率与语言模型对数概率在同一量纲上耦合。
    若传入 `allowed_vocab_indices`（多尺度高频子词表 V_N），则仅在该子词表列内筛选 Top-K 并做局部归一化。
    """
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"温度参数 T_r 必须为正数，收到：{temperature!r}")
    if isinstance(cosine_scores, np.ndarray):
        tensor = torch.from_numpy(cosine_scores).float().clone()
    else:
        tensor = cosine_scores.detach().cpu().float().clone()
    scaled = tensor / temp
    vocab_size = int(scaled.shape[-1]) if scaled.ndim >= 1 else 0

    active_vocab_count = vocab_size
    disallowed_mask: torch.Tensor | None = None
    if allowed_vocab_indices is not None and vocab_size > 0:
        valid_idx = [
            int(i)
            for i in allowed_vocab_indices
            if 0 <= int(i) < vocab_size
        ]
        if 0 < len(valid_idx) < vocab_size:
            active_vocab_count = len(valid_idx)
            allowed_mask = torch.zeros(vocab_size, dtype=torch.bool)
            allowed_mask[valid_idx] = True
            disallowed_mask = ~allowed_mask
            scaled[..., disallowed_mask] = -1e30

    if (
        top_k is not None
        and active_vocab_count > 0
        and 1 <= int(top_k) <= active_vocab_count
    ):
        k_norm = min(int(top_k), active_vocab_count)
        top_vals, _ = torch.topk(scaled, k=k_norm, dim=-1)
        log_z = torch.logsumexp(top_vals, dim=-1, keepdim=True)
        log_probs = scaled - log_z
    else:
        log_probs = F_nn.log_softmax(scaled, dim=-1)

    if disallowed_mask is not None:
        log_probs[..., disallowed_mask] = -1e30
    return log_probs.numpy().astype(np.float64)


def select_top_k_neural_candidates(
    neural_log_probs: np.ndarray,
    top_k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """按神经对数概率截取每个词位的 Top-K (默认 K=20) 候选词索引与对应神经对数概率。

    返回:
        candidate_indices: shape [L, K], int64
        candidate_log_probs: shape [L, K], float64
    """
    scores = np.asarray(neural_log_probs, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"neural_log_probs 必须为二维矩阵 [L, |V|]，收到 shape={scores.shape}")
    vocab_size = scores.shape[1]
    k = min(max(1, int(top_k)), vocab_size)
    top_indices = np.argsort(-scores, axis=1, kind="stable")[:, :k].astype(np.int64)
    top_log_probs = np.take_along_axis(scores, top_indices, axis=1)
    return top_indices, top_log_probs


def _select_top_beam_expansions(
    beams: Sequence[BeamHypothesis],
    candidate_indices: Sequence[int],
    neural_row: np.ndarray,
    lm_rows: Sequence[np.ndarray],
    language_weight: float,
    beam_width: int,
    vocab_tuple: Sequence[str],
) -> list[tuple]:
    """向量化计算 束 × 候选 的综合得分并选出 Top-B 扩展。

    与逐元素构造全部 B×K 个元组后排序的旧实现严格等价（排序键为 (-总分, 词序列)），
    但仅为总分不低于第 B 大值的少量候选构造元组，消除每步数千次 Python 元组分配。
    返回元组结构：(new_total, neural_sum, lm_sum, beam_idx, word_idx, words, word_indices)。
    """
    n_cands = len(candidate_indices)
    if n_cands == 0 or not beams:
        return []
    cand_arr = np.asarray(candidate_indices, dtype=np.int64)
    n_vals = np.asarray(neural_row, dtype=np.float64)[cand_arr]
    lm_mat = np.stack(
        [np.asarray(row, dtype=np.float64)[:n_cands] for row in lm_rows], axis=0
    )
    lam = float(language_weight)
    step_scores = n_vals[None, :] + lam * lm_mat
    beam_scores = np.asarray([b.score for b in beams], dtype=np.float64)
    totals = beam_scores[:, None] + step_scores
    flat_totals = totals.reshape(-1)
    b_width = max(1, int(beam_width))
    if flat_totals.size > b_width:
        threshold = np.partition(flat_totals, flat_totals.size - b_width)[
            flat_totals.size - b_width
        ]
        keep_flat = np.nonzero(flat_totals >= threshold)[0]
    else:
        keep_flat = np.arange(flat_totals.size)

    expanded: list[tuple] = []
    for flat_i in keep_flat.tolist():
        b_idx, c_pos = divmod(int(flat_i), n_cands)
        beam = beams[b_idx]
        w_id = int(cand_arr[c_pos])
        n_lp = float(n_vals[c_pos])
        lm_lp = float(lm_mat[b_idx, c_pos])
        expanded.append(
            (
                float(totals[b_idx, c_pos]),
                beam.neural_score_sum + n_lp,
                beam.lm_score_sum + lm_lp,
                b_idx,
                w_id,
                beam.words + (vocab_tuple[w_id],),
                beam.word_indices + (w_id,),
            )
        )
    expanded.sort(key=lambda item: (-item[0], item[5]))
    return expanded[:b_width]

def decode_segment_static(
    neural_log_probs: np.ndarray | None,
    vocabulary: Sequence[str],
    scorer: CachedVocabularyLanguageScorer,
    *,
    language_weight: float = 0.15,
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_length: int | None = None,
    pure_language_model: bool = False,
) -> dict:
    """执行单状态静态从左到右束搜索（支持 S0, K0, K1, K2, F 与 LM-Only 基线）。"""
    vocab_tuple = tuple(str(w) for w in vocabulary)
    vocab_size = len(vocab_tuple)
    if vocab_size == 0:
        raise ValueError("vocabulary 不能为空。")

    if pure_language_model or neural_log_probs is None:
        if segment_length is None:
            raise ValueError("LM-Only 模式必须提供 segment_length 或 neural_log_probs。")
        seg_len = int(segment_length)
        uniform_lp = -float(np.log(vocab_size))
        scores_matrix = np.full((seg_len, vocab_size), uniform_lp, dtype=np.float64)
        is_lm_only = True
    else:
        scores_matrix = np.asarray(neural_log_probs, dtype=np.float64)
        if scores_matrix.ndim != 2 or scores_matrix.shape[1] != vocab_size:
            raise ValueError(
                f"neural_log_probs 形状 {scores_matrix.shape} 与词表大小 {vocab_size} 不匹配。"
            )
        seg_len = scores_matrix.shape[0]
        is_lm_only = False

    if seg_len == 0:
        return {
            "final_words": [],
            "score": 0.0,
            "elapsed_seconds": 0.0,
        }

    started = time.perf_counter()
    b_width = max(1, int(beam_width))
    k_top = min(max(1, int(candidate_top_k)), vocab_size)
    lam = float(language_weight)

    if not is_lm_only:
        cand_ids_matrix, _ = select_top_k_neural_candidates(
            scores_matrix, top_k=k_top
        )
    else:
        cand_ids_matrix = None

    prompt_state = scorer.init_prompt_state()
    beams: list[BeamHypothesis] = [
        BeamHypothesis(
            words=(),
            word_indices=(),
            cache_state=prompt_state,
            score=0.0,
            neural_score_sum=0.0,
            lm_score_sum=0.0,
        )
    ]

    top5_words_per_pos: list[list[str]] = []
    for pos in range(seg_len):
        beam_states = [b.cache_state for b in beams]
        if is_lm_only:
            cand_lists = [
                scorer.propose_top_k_from_language_model(
                    b.cache_state, top_k=k_top
                ).tolist()
                for b in beams
            ]
        else:
            row_cands = cand_ids_matrix[pos].tolist()
            cand_lists = [row_cands for _ in beams]

        lm_scores_per_beam = scorer.score_candidates_with_cache(
            beam_states, cand_lists
        )

        if cand_lists and lm_scores_per_beam:
            top1_step_pairs = [
                (
                    float(scores_matrix[pos, w_id]) + lam * float(lm_val),
                    vocab_tuple[w_id],
                )
                for w_id, lm_val in zip(cand_lists[0], lm_scores_per_beam[0])
            ]
            top1_step_pairs.sort(key=lambda kv: (-kv[0], kv[1]))
            top5_words_per_pos.append([w for _, w in top1_step_pairs[:5]])
        else:
            top5_words_per_pos.append([])

        expanded: list[
            tuple[float, float, float, int, int, tuple[str, ...], tuple[int, ...]]
        ] = []
        for b_idx, (beam, cands, lm_row) in enumerate(
            zip(beams, cand_lists, lm_scores_per_beam)
        ):
            for c_pos, w_id in enumerate(cands):
                n_lp = float(scores_matrix[pos, w_id])
                lm_lp = float(lm_row[c_pos])
                step_score = n_lp + lam * lm_lp
                new_total = beam.score + step_score
                w_str = vocab_tuple[w_id]
                expanded.append(
                    (
                        new_total,
                        beam.neural_score_sum + n_lp,
                        beam.lm_score_sum + lm_lp,
                        b_idx,
                        int(w_id),
                        beam.words + (w_str,),
                        beam.word_indices + (int(w_id),),
                    )
                )

        expanded.sort(key=lambda item: (-item[0], item[5]))
        selected = expanded[:b_width]

        if pos < seg_len - 1:
            adv_reqs = [
                (beams[item[3]].cache_state, item[4]) for item in selected
            ]
            adv_caches = scorer.advance_prefix_states(adv_reqs)
            beams = [
                BeamHypothesis(
                    words=item[5],
                    word_indices=item[6],
                    cache_state=adv_cache,
                    score=item[0],
                    neural_score_sum=item[1],
                    lm_score_sum=item[2],
                )
                for item, adv_cache in zip(selected, adv_caches)
            ]
        else:
            beams = [
                BeamHypothesis(
                    words=item[5],
                    word_indices=item[6],
                    cache_state=beams[item[3]].cache_state,
                    score=item[0],
                    neural_score_sum=item[1],
                    lm_score_sum=item[2],
                )
                for item in selected
            ]

    winner = beams[0]
    return {
        "final_words": list(winner.words),
        "beam_candidates": [list(b.words) for b in beams],
        "top5_words_per_pos": top5_words_per_pos,
        "score": float(winner.score),
        "neural_score_sum": float(winner.neural_score_sum),
        "lm_score_sum": float(winner.lm_score_sum),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _static_argmax_decode(mat: np.ndarray, vocab_tuple: Sequence[str]) -> dict:
    """λ=0 时静态束搜索退化为逐词位神经 argmax（无需语言模型）。"""
    vocab_size = len(vocab_tuple)
    if mat.shape[0] == 0:
        return {
            "final_words": [],
            "beam_candidates": [],
            "top5_words_per_pos": [],
            "score": 0.0,
            "elapsed_seconds": 0.0,
        }
    top1 = np.argmax(mat, axis=-1)
    top5_idx = select_top_k_neural_candidates(mat, top_k=min(5, vocab_size))[0]
    words = [vocab_tuple[int(i)] for i in top1]
    top5_per_pos = [[vocab_tuple[int(i)] for i in row] for row in top5_idx]
    score_sum = float(sum(mat[r, int(top1[r])] for r in range(mat.shape[0])))
    return {
        "final_words": words,
        "beam_candidates": [words],
        "top5_words_per_pos": top5_per_pos,
        "score": score_sum,
        "neural_score_sum": score_sum,
        "lm_score_sum": 0.0,
        "elapsed_seconds": 0.0,
    }


def decode_segments_static_batched(
    segment_log_probs_list: Sequence[np.ndarray],
    vocabulary: Sequence[str],
    scorer: CachedVocabularyLanguageScorer,
    *,
    language_weight: float | Sequence[float] = 0.15,
    beam_width: int = 5,
    candidate_top_k: int = 20,
) -> list[dict]:
    """跨多个句段按时间步同步批量执行静态束搜索，大幅合并 GPU 前向与 KV-Cache 批处理调用。

    `language_weight` 可为单一标量，也可为与句段一一对应的序列（逐句段 λ），
    从而允许把不同状态/不同 (T_r, λ_r) 网格点的句段合并到同一批次同步解码。
    """
    if not segment_log_probs_list:
        return []
    vocab_tuple = tuple(str(w) for w in vocabulary)
    vocab_size = len(vocab_tuple)
    b_width = max(1, int(beam_width))
    k_top = min(max(1, int(candidate_top_k)), vocab_size)

    num_segs = len(segment_log_probs_list)
    if isinstance(language_weight, (int, float, np.floating, np.integer)):
        lam_by_seg = [float(language_weight)] * num_segs
    else:
        lam_by_seg = [float(x) for x in language_weight]
        if len(lam_by_seg) != num_segs:
            raise ValueError(
                f"language_weight 序列长度 ({len(lam_by_seg)}) 与句段数 ({num_segs}) 不一致。"
            )
    scores_matrices = [
        np.asarray(mat, dtype=np.float64) for mat in segment_log_probs_list
    ]

    zero_lam_idx = [i for i, lam in enumerate(lam_by_seg) if abs(lam) <= 1e-12]
    if zero_lam_idx:
        results_by_idx: dict[int, dict] = {
            i: _static_argmax_decode(scores_matrices[i], vocab_tuple)
            for i in zero_lam_idx
        }
        beam_idx_list = [i for i in range(num_segs) if i not in results_by_idx]
        if beam_idx_list:
            beam_results = decode_segments_static_batched(
                [scores_matrices[i] for i in beam_idx_list],
                vocab_tuple,
                scorer,
                language_weight=[lam_by_seg[i] for i in beam_idx_list],
                beam_width=b_width,
                candidate_top_k=k_top,
            )
            results_by_idx.update(zip(beam_idx_list, beam_results))
        return [results_by_idx[i] for i in range(num_segs)]

    seg_lengths = [mat.shape[0] for mat in scores_matrices]

    started = time.perf_counter()
    cand_matrices = [
        select_top_k_neural_candidates(mat, top_k=k_top)[0]
        if mat.shape[0] > 0
        else np.zeros((0, k_top), dtype=np.int64)
        for mat in scores_matrices
    ]
    prompt_state = scorer.init_prompt_state()
    beams_by_seg: list[list[BeamHypothesis]] = [
        [
            BeamHypothesis(
                words=(),
                word_indices=(),
                cache_state=prompt_state,
                score=0.0,
                neural_score_sum=0.0,
                lm_score_sum=0.0,
            )
        ]
        for _ in range(num_segs)
    ]
    top5_by_seg: list[list[list[str]]] = [[] for _ in range(num_segs)]

    max_len = max(seg_lengths) if seg_lengths else 0
    for pos in range(max_len):
        active_segs = [s_idx for s_idx in range(num_segs) if pos < seg_lengths[s_idx]]
        if not active_segs:
            break

        flat_states: list[PrefixCacheState] = []
        flat_cands: list[list[int]] = []
        seg_slices: list[tuple[int, int, int, list[int]]] = []

        for s_idx in active_segs:
            s_beams = beams_by_seg[s_idx]
            row_cands = cand_matrices[s_idx][pos].tolist()
            start_offset = len(flat_states)
            for b in s_beams:
                flat_states.append(b.cache_state)
                flat_cands.append(row_cands)
            seg_slices.append(
                (s_idx, start_offset, len(flat_states), row_cands)
            )

        flat_lm_scores = scorer.score_candidates_with_cache(
            flat_states, flat_cands
        )

        selected_by_seg: dict[
            int,
            list[
                tuple[
                    float,
                    float,
                    float,
                    int,
                    int,
                    tuple[str, ...],
                    tuple[int, ...],
                ]
            ],
        ] = {}
        flat_adv_reqs: list[tuple[PrefixCacheState, int]] = []
        adv_slices: list[tuple[int, int, int]] = []

        for s_idx, start_off, end_off, row_cands in seg_slices:
            s_beams = beams_by_seg[s_idx]
            s_lm_rows = flat_lm_scores[start_off:end_off]
            s_mat = scores_matrices[s_idx]
            lam = lam_by_seg[s_idx]
            if row_cands and s_lm_rows:
                step_pairs = [
                    (
                        float(s_mat[pos, w_id]) + lam * float(lm_val),
                        vocab_tuple[w_id],
                    )
                    for w_id, lm_val in zip(row_cands, s_lm_rows[0])
                ]
                step_pairs.sort(key=lambda kv: (-kv[0], kv[1]))
                top5_by_seg[s_idx].append([w for _, w in step_pairs[:5]])
            else:
                top5_by_seg[s_idx].append([])

            selected = _select_top_beam_expansions(
                s_beams,
                row_cands,
                s_mat[pos],
                s_lm_rows,
                lam,
                b_width,
                vocab_tuple,
            )
            selected_by_seg[s_idx] = selected

            if pos < seg_lengths[s_idx] - 1:
                adv_start = len(flat_adv_reqs)
                for item in selected:
                    flat_adv_reqs.append(
                        (s_beams[item[3]].cache_state, item[4])
                    )
                adv_slices.append((s_idx, adv_start, len(flat_adv_reqs)))
            else:
                beams_by_seg[s_idx] = [
                    BeamHypothesis(
                        words=item[5],
                        word_indices=item[6],
                        cache_state=s_beams[item[3]].cache_state,
                        score=item[0],
                        neural_score_sum=item[1],
                        lm_score_sum=item[2],
                    )
                    for item in selected
                ]

        if flat_adv_reqs:
            flat_adv_caches = scorer.advance_prefix_states(flat_adv_reqs)
            for s_idx, adv_start, adv_end in adv_slices:
                selected = selected_by_seg[s_idx]
                adv_caches = flat_adv_caches[adv_start:adv_end]
                beams_by_seg[s_idx] = [
                    BeamHypothesis(
                        words=item[5],
                        word_indices=item[6],
                        cache_state=adv_cache,
                        score=item[0],
                        neural_score_sum=item[1],
                        lm_score_sum=item[2],
                    )
                    for item, adv_cache in zip(selected, adv_caches)
                ]

    per_seg_elapsed = round(
        (time.perf_counter() - started) / max(1, num_segs), 4
    )
    outputs: list[dict] = []
    for s_idx in range(num_segs):
        if seg_lengths[s_idx] == 0:
            outputs.append(
                {
                    "final_words": [],
                    "beam_candidates": [],
                    "top5_words_per_pos": [],
                    "score": 0.0,
                    "elapsed_seconds": 0.0,
                }
            )
            continue
        winner = beams_by_seg[s_idx][0]
        outputs.append(
            {
                "final_words": list(winner.words),
                "beam_candidates": [list(b.words) for b in beams_by_seg[s_idx]],
                "top5_words_per_pos": top5_by_seg[s_idx],
                "score": float(winner.score),
                "neural_score_sum": float(winner.neural_score_sum),
                "lm_score_sum": float(winner.lm_score_sum),
                "elapsed_seconds": per_seg_elapsed,
            }
        )
    return outputs

STATIC_DECODE_CONDITIONS: frozenset[str] = frozenset({"S0", "K0", "K1", "K2", "F"})
BRAIN_ONLY_CONDITION_MAP: dict[str, str] = {
    "BRAIN_ONLY": "K2",
    "S3_BRAIN_ONLY": "K2",
    "S0_BRAIN_ONLY": "S0",
    "K0_BRAIN_ONLY": "K0",
    "K1_BRAIN_ONLY": "K1",
    "K2_BRAIN_ONLY": "K2",
    "F_BRAIN_ONLY": "F",
}


def _resolve_brain_only_state(condition: str, *, brain_only: bool = False) -> str | None:
    """将 *_brain_only / Brain-Only 条件名映射至底层状态键（S0/K0/K1/K2/F）。"""
    norm = str(condition or "").strip().upper().replace("-", "_")
    if norm in BRAIN_ONLY_CONDITION_MAP:
        return BRAIN_ONLY_CONDITION_MAP[norm]
    if brain_only:
        return "K2" if norm in ("S3", "S3_CLEAN") else (
            norm if norm in STATIC_DECODE_CONDITIONS else None
        )
    return None


def _resolve_job_vocabulary(
    exported_payload: dict,
    *,
    vocab_size_limit: int | None = None,
    allowed_vocab_indices: Sequence[int] | None = None,
) -> tuple[list[int] | None, set[str] | None, int]:
    """解析单个解码作业的受限候选词表（返回激活词索引、激活词集合与有效词表规模）。"""
    from braindecoding.decoding.neural_score_exporter import (
        resolve_vocabulary_frequency_order,
    )

    vocabulary = list(exported_payload["vocabulary"])
    active_vocab_indices: list[int] | None = None
    active_vocab_set: set[str] | None = None
    if allowed_vocab_indices is not None:
        active_vocab_indices = [
            int(i)
            for i in allowed_vocab_indices
            if 0 <= int(i) < len(vocabulary)
        ]
        if len(active_vocab_indices) >= len(vocabulary):
            active_vocab_indices = None
        else:
            active_vocab_set = {str(vocabulary[i]) for i in active_vocab_indices}
    elif (
        vocab_size_limit is not None
        and 0 < int(vocab_size_limit) < len(vocabulary)
    ):
        freq_indices, freq_words, _ = resolve_vocabulary_frequency_order(
            exported_payload
        )
        n_lim = int(vocab_size_limit)
        active_vocab_indices = freq_indices[:n_lim]
        active_vocab_set = set(freq_words[:n_lim])
    eff_vocab_size = (
        len(active_vocab_indices)
        if active_vocab_indices is not None
        else len(vocabulary)
    )
    return active_vocab_indices, active_vocab_set, eff_vocab_size


def _segment_record_base(seg: dict) -> dict:
    base = {
        "segment_index": seg["segment_index"],
        "group_id": seg["group_id"],
        "subject_id": seg["subject_id"],
        "recording_id": seg["recording_id"],
        "scope": seg["scope"],
        "material_id": str(seg.get("material_id", "")),
        "block_id": seg.get("block_id", seg["scope"]),
        "reference_words": list(seg["reference_words"]),
    }
    for meta_key in (
        "event_ids",
        "onset_seconds",
        "offset_seconds",
        "pos_tags",
        "is_content_word_mask",
    ):
        if meta_key in seg and seg[meta_key] is not None:
            base[meta_key] = list(seg[meta_key])
    return base


def _decode_brain_only_segment_records(
    score_matrix: torch.Tensor | np.ndarray,
    vocabulary: Sequence[str],
    segments: Sequence[dict],
    *,
    active_vocab_indices: Sequence[int] | None = None,
    active_vocab_set: set[str] | None = None,
    top_k_save: int = 5,
) -> tuple[list[dict], float]:
    """零语言模型纯神经（λ = 0）argmax 与 Top-K 词表检索解码，支持全词表与受限子词表。"""
    started = time.perf_counter()
    if isinstance(score_matrix, np.ndarray):
        mat = torch.from_numpy(score_matrix).float()
    else:
        mat = score_matrix.detach().cpu().float()
    if active_vocab_indices is not None:
        idx_list = [int(i) for i in active_vocab_indices]
        idx_tensor = torch.as_tensor(idx_list, dtype=torch.long)
        vocab_view = [str(vocabulary[i]) for i in idx_list]
    else:
        idx_tensor = None
        vocab_view = [str(w) for w in vocabulary]
    n_cand = len(vocab_view)
    k_eff = min(max(1, int(top_k_save)), max(1, n_cand))

    records: list[dict] = []
    for seg in segments:
        t0 = time.perf_counter()
        ev = seg["event_indices"]
        sub = mat[ev] if idx_tensor is None else mat[ev][:, idx_tensor]
        seg_len = int(sub.shape[0])
        if seg_len == 0 or n_cand == 0:
            pred_words: list[str] = []
            top5_rows: list[list[str]] = []
        else:
            _, top_cols = torch.topk(sub, k=k_eff, dim=-1, largest=True, sorted=True)
            top_cols_list = top_cols.tolist()
            pred_words = [vocab_view[int(row[0])] for row in top_cols_list]
            top5_rows = [[vocab_view[int(c)] for c in row] for row in top_cols_list]
        rec = _segment_record_base(seg)
        rec.update(
            {
                "predicted_words": pred_words,
                "beam_candidates": [list(pred_words)] if pred_words else [],
                "top5_words_per_pos": top5_rows,
                "in_vocabulary_mask": _job_in_vocab_mask(seg, active_vocab_set),
                "elapsed_seconds": round(time.perf_counter() - t0, 6),
            }
        )
        records.append(rec)
    return records, time.perf_counter() - started

def decode_dataset_segment_jobs(
    exported_payload: dict,
    scorer: CachedVocabularyLanguageScorer | None,
    hyperparameters: dict,
    jobs: dict[str, dict],
    *,
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_batch_size: int = 32,
    segment_indices: Sequence[int] | None = None,
    partition_filter: str | None = None,
    max_segments: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict[str, dict]:
    """将多个单遍静态解码作业（不同条件 / 对齐方式 / 受限词表）合并为同一批次同步束搜索。

    每个作业为字典：`condition`（S0/K0/K1/K2/F/S3/LM-Only 及 *_brain_only）、
    `alignment`（clean/shifted）、可选 `vocab_size_limit` / `allowed_vocab_indices` / `segment_indices`。
    其中 `S3` 直接使用最终局部上下文状态 `K2` 执行单遍左到右束搜索。
    返回 `{job_key: bundle}`，bundle 结构与 `decode_dataset_segments` 完全一致。
    """
    from tqdm.auto import tqdm

    vocabulary = list(exported_payload["vocabulary"])
    base_segments = [
        seg
        for seg in exported_payload["segments"]
        if partition_filter is None or seg.get("partition") == partition_filter
    ]
    if segment_indices is not None:
        allowed_seg_set = {int(i) for i in segment_indices}
        segments = [
            seg
            for pos, seg in enumerate(base_segments)
            if pos in allowed_seg_set
            or int(seg.get("segment_index", -1)) in allowed_seg_set
        ]
    else:
        segments = base_segments
    if max_segments is not None:
        segments = segments[: int(max_segments)]
    n_segs = len(segments)
    states_cfg = hyperparameters.get("states", hyperparameters)

    def _get_state_params(state_name: str) -> tuple[float, float]:
        entry = states_cfg.get(state_name, {})
        return (
            float(entry.get("temperature", 0.07)),
            float(entry.get("language_weight", 0.15)),
        )

    results: dict[str, dict] = {}
    static_items: list[tuple[int, int, str, np.ndarray, float, int]] = []
    job_meta: dict[str, dict] = {}
    lm_only_jobs: list[str] = []
    brain_only_jobs: list[str] = []

    for job_order, (job_key, job) in enumerate(jobs.items()):
        condition = str(job.get("condition", "S3"))
        cond_upper = condition.upper()
        alignment = str(job.get("alignment", "clean"))
        brain_only_state = _resolve_brain_only_state(
            condition, brain_only=bool(job.get("brain_only", False))
        )
        if brain_only_state is not None:
            active_idx, active_set, eff_vocab_size = _resolve_job_vocabulary(
                exported_payload,
                vocab_size_limit=job.get("vocab_size_limit"),
                allowed_vocab_indices=job.get("allowed_vocab_indices"),
            )
            score_bank = (
                exported_payload["clean_scores"]
                if alignment == "clean"
                else exported_payload["shifted_scores"]
            )
            if brain_only_state not in score_bank:
                raise KeyError(
                    f"得分矩阵中缺少 Brain-Only 状态 {brain_only_state!r} (alignment={alignment!r})"
                )
            job_seg_indices = job.get("segment_indices")
            job_seg_set = (
                {int(i) for i in job_seg_indices}
                if job_seg_indices is not None
                else None
            )
            sel_segments = [
                seg
                for seg_pos, seg in enumerate(segments)
                if job_seg_set is None
                or seg_pos in job_seg_set
                or int(seg.get("segment_index", -1)) in job_seg_set
            ]
            rec_list, bo_elapsed = _decode_brain_only_segment_records(
                score_bank[brain_only_state],
                vocabulary,
                sel_segments,
                active_vocab_indices=active_idx,
                active_vocab_set=active_set,
            )
            n_bo = len(rec_list)
            results[job_key] = {
                "condition": condition,
                "alignment": alignment,
                "vocabulary_size": eff_vocab_size,
                "segment_count": n_bo,
                "total_elapsed_seconds": round(bo_elapsed, 4),
                "mean_seconds_per_segment": round(bo_elapsed / max(1, n_bo), 4),
                "final_sentences": rec_list,
            }
            brain_only_jobs.append(job_key)
            continue
        if scorer is None:
            raise ValueError(
                f"解码条件 {condition!r}（非 Brain-Only）必须提供非空语言模型 scorer。"
            )
        if cond_upper == "LM-ONLY":
            lm_only_jobs.append(job_key)
            continue
        if cond_upper != "S3" and cond_upper not in STATIC_DECODE_CONDITIONS:
            raise ValueError(
                f"不支持的解码条件 {condition!r}，可选：'S3', 'S0', 'K0', 'K1', 'K2', 'F', 'LM-Only' 及 *_brain_only"
            )
        active_idx, active_set, eff_vocab_size = _resolve_job_vocabulary(
            exported_payload,
            vocab_size_limit=job.get("vocab_size_limit"),
            allowed_vocab_indices=job.get("allowed_vocab_indices"),
        )
        eff_top_k = (
            min(int(candidate_top_k), len(active_idx))
            if active_idx is not None
            else int(candidate_top_k)
        )
        score_bank = (
            exported_payload["clean_scores"]
            if alignment == "clean"
            else exported_payload["shifted_scores"]
        )
        kind = "static_s3" if cond_upper == "S3" else "static"
        job_seg_indices = job.get("segment_indices")
        job_seg_set = (
            {int(i) for i in job_seg_indices}
            if job_seg_indices is not None
            else None
        )
        job_meta[job_key] = {
            "condition": condition,
            "alignment": alignment,
            "vocabulary_size": eff_vocab_size,
            "active_set": active_set,
            "kind": kind,
            "records": [None] * n_segs,
            "elapsed": 0.0,
        }

        state_name = "K2" if cond_upper == "S3" else cond_upper
        t_r, lam_r = _get_state_params(state_name)
        full_lp = compute_neural_log_probabilities(
            score_bank[state_name],
            t_r,
            top_k=eff_top_k,
            allowed_vocab_indices=active_idx,
        )
        for seg_pos, seg in enumerate(segments):
            if (
                job_seg_set is not None
                and seg_pos not in job_seg_set
                and int(seg.get("segment_index", -1)) not in job_seg_set
            ):
                continue
            static_items.append(
                (
                    seg_pos,
                    job_order,
                    job_key,
                    full_lp[seg["event_indices"]],
                    lam_r,
                    eff_top_k,
                )
            )

    # 同一句段的不同作业相邻排列，提升同批次前缀共享
    static_items.sort(key=lambda item: (item[5], item[0], item[1]))
    total_items = len(static_items)
    batch_sz = max(1, int(segment_batch_size))
    started = time.perf_counter()
    done_items = 0
    pbar = (
        tqdm(
            total=total_items,
            desc=progress_desc or "S3 Decode [batched jobs]",
            unit="seg",
            leave=True,
        )
        if show_progress and total_items > 0
        else None
    )

    def _tick(count: int) -> None:
        nonlocal done_items
        done_items += count
        if pbar is not None:
            pbar.update(count)
        if progress_callback is not None:
            progress_callback(done_items, total_items, time.perf_counter() - started)

    try:
        start_i = 0
        while start_i < len(static_items):
            top_k_val = static_items[start_i][5]
            end_i = start_i
            while (
                end_i < len(static_items)
                and end_i - start_i < batch_sz
                and static_items[end_i][5] == top_k_val
            ):
                end_i += 1
            chunk = static_items[start_i:end_i]
            decs = decode_segments_static_batched(
                [item[3] for item in chunk],
                vocabulary,
                scorer,
                language_weight=[item[4] for item in chunk],
                beam_width=beam_width,
                candidate_top_k=top_k_val,
            )
            for (seg_pos, _, job_key, _, _, _), dec in zip(chunk, decs):
                meta = job_meta[job_key]
                seg = segments[seg_pos]
                record = _segment_record_base(seg)
                record["predicted_words"] = list(dec["final_words"])
                if meta["kind"] == "static_s3":
                    record["committed_k2_words"] = list(dec["final_words"])
                record["beam_candidates"] = list(dec.get("beam_candidates", []))
                record["top5_words_per_pos"] = list(
                    dec.get("top5_words_per_pos", [])
                )
                record["in_vocabulary_mask"] = _job_in_vocab_mask(
                    seg, meta["active_set"]
                )
                record["elapsed_seconds"] = dec["elapsed_seconds"]
                meta["records"][seg_pos] = record
                meta["elapsed"] += float(dec.get("elapsed_seconds", 0.0))
            _tick(len(chunk))
            start_i = end_i
    finally:
        if pbar is not None:
            pbar.close()

    for job_key in jobs:
        if job_key in brain_only_jobs:
            continue
        if job_key in lm_only_jobs:
            job = jobs[job_key]
            results[job_key] = decode_dataset_segments(
                exported_payload,
                scorer,
                hyperparameters,
                condition=str(job.get("condition", "LM-Only")),
                alignment=str(job.get("alignment", "clean")),
                beam_width=beam_width,
                candidate_top_k=candidate_top_k,
                segment_batch_size=segment_batch_size,
                vocab_size_limit=job.get("vocab_size_limit"),
                allowed_vocab_indices=job.get("allowed_vocab_indices"),
                segment_indices=job.get("segment_indices", segment_indices),
                partition_filter=partition_filter,
                max_segments=max_segments,
            )
            continue
        meta = job_meta[job_key]
        total_elapsed = float(meta["elapsed"])
        valid_records = [r for r in meta["records"] if r is not None]
        job_n_segs = len(valid_records)
        results[job_key] = {
            "condition": meta["condition"],
            "alignment": meta["alignment"],
            "vocabulary_size": meta["vocabulary_size"],
            "segment_count": job_n_segs,
            "total_elapsed_seconds": round(total_elapsed, 4),
            "mean_seconds_per_segment": round(total_elapsed / max(1, job_n_segs), 4),
            "final_sentences": valid_records,
        }
    return results

def _job_in_vocab_mask(seg: dict, active_set: set[str] | None) -> list[bool]:
    if active_set is not None:
        return [str(w) in active_set for w in seg.get("reference_words", [])]
    return list(seg["in_vocabulary_mask"])


def decode_dataset_segments(
    exported_payload: dict,
    scorer: CachedVocabularyLanguageScorer | None,
    hyperparameters: dict,
    *,
    condition: str = "S3",
    alignment: str = "clean",
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_batch_size: int = 32,
    vocab_size_limit: int | None = None,
    allowed_vocab_indices: Sequence[int] | None = None,
    segment_indices: Sequence[int] | None = None,
    partition_filter: str | None = None,
    max_segments: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict:
    """对导出的 .pt 神经得分载荷批量执行指定条件（S3, S0, K0, K1, K2, F, LM-Only）的句段解码。"""
    cond_upper = str(condition).upper()
    if cond_upper != "LM-ONLY":
        return decode_dataset_segment_jobs(
            exported_payload,
            scorer,
            hyperparameters,
            {
                "job": {
                    "condition": condition,
                    "alignment": alignment,
                    "vocab_size_limit": vocab_size_limit,
                    "allowed_vocab_indices": allowed_vocab_indices,
                    "segment_indices": segment_indices,
                }
            },
            beam_width=beam_width,
            candidate_top_k=candidate_top_k,
            segment_batch_size=segment_batch_size,
            segment_indices=segment_indices,
            partition_filter=partition_filter,
            max_segments=max_segments,
            show_progress=show_progress,
            progress_desc=progress_desc or f"S3 Decode [{condition} | {alignment}]",
            progress_callback=progress_callback,
        )["job"]

    from tqdm.auto import tqdm

    vocabulary = list(exported_payload["vocabulary"])
    active_vocab_indices, active_vocab_set, eff_vocab_size = _resolve_job_vocabulary(
        exported_payload,
        vocab_size_limit=vocab_size_limit,
        allowed_vocab_indices=allowed_vocab_indices,
    )
    eff_top_k = (
        min(int(candidate_top_k), len(active_vocab_indices))
        if active_vocab_indices is not None
        else int(candidate_top_k)
    )

    base_segments = [
        seg
        for seg in exported_payload["segments"]
        if partition_filter is None or seg.get("partition") == partition_filter
    ]
    if segment_indices is not None:
        allowed_seg_set = {int(i) for i in segment_indices}
        segments = [
            seg
            for pos, seg in enumerate(base_segments)
            if pos in allowed_seg_set
            or int(seg.get("segment_index", -1)) in allowed_seg_set
        ]
    else:
        segments = base_segments
    if max_segments is not None:
        segments = segments[: int(max_segments)]

    states_cfg = hyperparameters.get("states", hyperparameters)
    k0_entry = states_cfg.get("K0", {})
    lm_weight = float(k0_entry.get("language_weight", 0.15))

    started = time.perf_counter()
    final_sentences: list[dict] = []
    pbar = (
        tqdm(
            total=len(segments),
            desc=progress_desc or f"S3 Decode [{condition} | {alignment}]",
            unit="seg",
            leave=True,
        )
        if show_progress and segments
        else None
    )

    try:
        # LM-Only 与受试者及脑信号无关，单次推进至最大句长 max_len 即可一次生成全部句长 L 的精确解码结果
        length_cache: dict[int, dict] = {
            0: {
                "final_words": [],
                "beam_candidates": [],
                "top5_words_per_pos": [],
                "score": 0.0,
                "elapsed_seconds": 0.0,
            }
        }
        needed_lengths = {
            len(seg["event_indices"])
            for seg in segments
            if len(seg["event_indices"]) > 0
        }
        if needed_lengths and len(vocabulary) > 0:
            max_lm_len = max(needed_lengths)
            vocab_tuple = tuple(str(w) for w in vocabulary)
            vocab_size = max(1, int(eff_vocab_size))
            uniform_lp = -float(np.log(vocab_size))
            lam = 1.0 if lm_weight <= 0 else float(lm_weight)
            b_width = max(1, int(beam_width))
            k_top = min(max(1, int(eff_top_k)), vocab_size)
            lm_started = time.perf_counter()
            prompt_state = scorer.init_prompt_state()
            beams: list[BeamHypothesis] = [
                BeamHypothesis(
                    words=(),
                    word_indices=(),
                    cache_state=prompt_state,
                    score=0.0,
                    neural_score_sum=0.0,
                    lm_score_sum=0.0,
                )
            ]
            top5_words_per_pos: list[list[str]] = []
            for pos in range(max_lm_len):
                beam_states = [b.cache_state for b in beams]
                cand_lists = [
                    scorer.propose_top_k_from_language_model(
                        b.cache_state,
                        top_k=k_top,
                        allowed_vocab_indices=active_vocab_indices,
                    ).tolist()
                    for b in beams
                ]
                lm_scores_per_beam = scorer.score_candidates_with_cache(
                    beam_states, cand_lists
                )
                if cand_lists and lm_scores_per_beam:
                    top1_step_pairs = [
                        (
                            uniform_lp + lam * float(lm_val),
                            vocab_tuple[w_id],
                        )
                        for w_id, lm_val in zip(
                            cand_lists[0], lm_scores_per_beam[0]
                        )
                    ]
                    top1_step_pairs.sort(key=lambda kv: (-kv[0], kv[1]))
                    top5_words_per_pos.append(
                        [w for _, w in top1_step_pairs[:5]]
                    )
                else:
                    top5_words_per_pos.append([])

                expanded: list[
                    tuple[
                        float,
                        float,
                        float,
                        int,
                        int,
                        tuple[str, ...],
                        tuple[int, ...],
                    ]
                ] = []
                for b_idx, (beam, cands, lm_row) in enumerate(
                    zip(beams, cand_lists, lm_scores_per_beam)
                ):
                    for c_pos, w_id in enumerate(cands):
                        lm_lp = float(lm_row[c_pos])
                        step_score = uniform_lp + lam * lm_lp
                        new_total = beam.score + step_score
                        w_str = vocab_tuple[w_id]
                        expanded.append(
                            (
                                new_total,
                                beam.neural_score_sum + uniform_lp,
                                beam.lm_score_sum + lm_lp,
                                b_idx,
                                int(w_id),
                                beam.words + (w_str,),
                                beam.word_indices + (int(w_id),),
                            )
                        )
                expanded.sort(key=lambda item: (-item[0], item[5]))
                selected = expanded[:b_width]
                cur_len = pos + 1
                if cur_len in needed_lengths:
                    winner_item = selected[0]
                    length_cache[cur_len] = {
                        "final_words": list(winner_item[5]),
                        "beam_candidates": [list(item[5]) for item in selected],
                        "top5_words_per_pos": [
                            list(row) for row in top5_words_per_pos
                        ],
                        "score": float(winner_item[0]),
                        "neural_score_sum": float(winner_item[1]),
                        "lm_score_sum": float(winner_item[2]),
                        "elapsed_seconds": round(
                            time.perf_counter() - lm_started, 4
                        ),
                    }
                if pos < max_lm_len - 1:
                    adv_reqs = [
                        (beams[item[3]].cache_state, item[4])
                        for item in selected
                    ]
                    adv_caches = scorer.advance_prefix_states(adv_reqs)
                    beams = [
                        BeamHypothesis(
                            words=item[5],
                            word_indices=item[6],
                            cache_state=adv_cache,
                            score=item[0],
                            neural_score_sum=item[1],
                            lm_score_sum=item[2],
                        )
                        for item, adv_cache in zip(selected, adv_caches)
                    ]

        for idx, seg in enumerate(segments):
            seg_len = len(seg["event_indices"])
            if seg_len not in length_cache:
                length_cache[seg_len] = decode_segment_static(
                    None,
                    vocabulary,
                    scorer,
                    language_weight=1.0 if lm_weight <= 0 else lm_weight,
                    beam_width=beam_width,
                    candidate_top_k=eff_top_k,
                    segment_length=seg_len,
                    pure_language_model=True,
                )
            dec = length_cache[seg_len]
            record = _segment_record_base(seg)
            record.update(
                {
                    "predicted_words": list(dec["final_words"]),
                    "beam_candidates": list(dec.get("beam_candidates", [])),
                    "top5_words_per_pos": list(dec.get("top5_words_per_pos", [])),
                    "in_vocabulary_mask": _job_in_vocab_mask(seg, active_vocab_set),
                    "elapsed_seconds": dec["elapsed_seconds"],
                }
            )
            final_sentences.append(record)
            if pbar is not None:
                pbar.update(1)
            if progress_callback is not None:
                progress_callback(
                    idx + 1, len(segments), time.perf_counter() - started
                )
    finally:
        if pbar is not None:
            pbar.close()

    total_elapsed = time.perf_counter() - started
    mean_sec = total_elapsed / max(1, len(segments))
    return {
        "condition": condition,
        "alignment": alignment,
        "vocabulary_size": eff_vocab_size,
        "segment_count": len(segments),
        "total_elapsed_seconds": round(total_elapsed, 4),
        "mean_seconds_per_segment": round(mean_sec, 4),
        "final_sentences": final_sentences,
    }

from braindecoding.decoding.beam_search_sweeps import (  # noqa: E402
    DEFAULT_VOCABULARY_SCALING_SIZES,
    _append_brain_only_jobs,
    _average_subject_score_rows_agreement_weighted,
    _average_subject_score_rows_rms_preserved,
    _build_train_corpus_words_and_vocab_lookup,
    _coerce_subject_score_tensors,
    _is_full_vocab_scale_token,
    _resolve_l5_closed_set_scales,
    aggregate_multi_subject_payload,
    run_multi_subject_observation_sweep,
    run_vocabulary_scaling_sweep,
)
