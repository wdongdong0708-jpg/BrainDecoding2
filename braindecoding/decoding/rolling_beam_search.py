"""方案一单遍定稿主干 (K2) + 尾部轻量草稿 (K1/K0) 束搜索解码器（Module 3）。

核心原则与架构：
1. 闭集词表 Top-K (K=20) 候选检索：利用神经余弦相似度矩阵除以状态温度 T_r 后做 log_softmax，
   每个词位仅截取神经概率 Top-20 候选词送入语言模型增量打分。
2. 方案一流式解码 (`decode_segment_streaming`)：
   - 定稿主干（Committed Backbone）：保持束宽 B=5，当时间步 t 推进时，前缀 u_{0:t-3} 永久锁定不变；
     仅对词位 t-2（达到终判状态 K2）执行 1 步束扩展，更新主干束 B_{t-2} 并永久定稿输出 w_{t-2}^{(K2)}。
   - 尾部草稿（Draft Tail）：取当前 Top-1 主干前缀的 KV-Cache，对词位 t-1（状态 K1）做 1 步取词得到
     一次回修显示 w_{t-1}^{(K1)}，再对词位 t（状态 K0）做 1 步取词得到初判显示 w_t^{(K0)}。
     草稿状态仅用于屏幕显示与改写裕度统计，用完即弃，绝不污染定稿主干束。
3. 单状态静态束搜索 (`decode_segment_static`)：
   - 支持 S0、K0、K1、K2、F 以及双锚点纯语言基线 `LM-Only`（神经概率置为均匀分布）。
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


def _score_single_step_candidates(
    scorer: CachedVocabularyLanguageScorer,
    prefix_state: PrefixCacheState,
    neural_log_prob_row: np.ndarray,
    primary_candidate_indices: Sequence[int],
    language_weight: float,
    *,
    extra_word_indices: Sequence[int] = (),
) -> tuple[int, float, dict[int, float], dict[int, float]]:
    """在单个前缀状态 `prefix_state` 下对 Top-K 候选词（及待对比的旧屏幕词）打分。

    返回:
        best_word_idx: 在 `primary_candidate_indices` 中综合得分最高的词索引
        best_total_score: 最优候选词的综合得分 s_i^(r)(w | u)
        total_score_by_word_idx: 候选词（含 extra_word_indices）的综合得分字典
        lm_score_by_word_idx: 候选词（含 extra_word_indices）的语言模型对数概率字典
    """
    primary_list = [int(i) for i in primary_candidate_indices]
    all_eval_indices = list(
        dict.fromkeys(primary_list + [int(i) for i in extra_word_indices])
    )
    lm_scores = scorer.score_candidates_with_cache(
        [prefix_state], [all_eval_indices]
    )[0]
    lam = float(language_weight)

    total_by_idx: dict[int, float] = {}
    lm_by_idx: dict[int, float] = {}
    for w_idx, lm_val in zip(all_eval_indices, lm_scores):
        n_val = float(neural_log_prob_row[w_idx])
        lm_f = float(lm_val)
        total_by_idx[w_idx] = n_val + lam * lm_f
        lm_by_idx[w_idx] = lm_f

    best_idx = max(
        primary_list,
        key=lambda idx: (total_by_idx[idx], -idx),
    )
    return best_idx, total_by_idx[best_idx], total_by_idx, lm_by_idx


def decode_segment_streaming(
    neural_log_probs_by_state: dict[str, np.ndarray],
    vocabulary: Sequence[str],
    scorer: CachedVocabularyLanguageScorer,
    *,
    language_weights: dict[str, float],
    beam_width: int = 5,
    candidate_top_k: int = 20,
    reference_words: Sequence[str] | None = None,
) -> dict:
    """执行方案一（单遍定稿主干 K2 + 尾部轻量草稿 K1/K0）流式束搜索解码。

    参数:
        neural_log_probs_by_state: 包含 "K0", "K1", "K2" 的 [L, |V|] 神经对数概率矩阵字典
        vocabulary: 长度 |V| 的训练词表序列
        scorer: `CachedVocabularyLanguageScorer` 实例
        language_weights: 状态自适应语言权重字典 `{"K0": λ_0, "K1": λ_1, "K2": λ_2}`
        beam_width: 定稿主干束宽 B（默认 5）
        candidate_top_k: 每词位神经候选截断数 K（默认 20）
        reference_words: 可选参考词序列（仅用于在线回修转移分类统计，绝不参与解码打分）
    """
    for req_state in ("K0", "K1", "K2"):
        if req_state not in neural_log_probs_by_state:
            raise ValueError(f"流式解码缺少状态 {req_state!r} 的神经对数概率矩阵。")

    k0_log_probs = np.asarray(neural_log_probs_by_state["K0"], dtype=np.float64)
    k1_log_probs = np.asarray(neural_log_probs_by_state["K1"], dtype=np.float64)
    k2_log_probs = np.asarray(neural_log_probs_by_state["K2"], dtype=np.float64)

    seg_len, vocab_size = k0_log_probs.shape
    if k1_log_probs.shape != (seg_len, vocab_size) or k2_log_probs.shape != (
        seg_len,
        vocab_size,
    ):
        raise ValueError("K0, K1, K2 神经对数概率矩阵形状不一致。")
    if len(vocabulary) != vocab_size:
        raise ValueError(
            f"词表长度 ({len(vocabulary)}) 与得分矩阵列数 ({vocab_size}) 不匹配。"
        )
    if seg_len == 0:
        return {
            "final_words": [],
            "committed_k2_words": [],
            "initial_k0_words": [],
            "screen_history": [],
            "position_trajectories": [],
            "elapsed_seconds": 0.0,
        }

    started = time.perf_counter()
    vocab_tuple = tuple(str(w) for w in vocabulary)
    b_width = max(1, int(beam_width))
    k_top = min(max(1, int(candidate_top_k)), vocab_size)

    lam_k0 = float(language_weights.get("K0", 0.2))
    lam_k1 = float(language_weights.get("K1", 0.15))
    lam_k2 = float(language_weights.get("K2", 0.1))

    k0_cand_ids, _ = select_top_k_neural_candidates(k0_log_probs, top_k=k_top)
    k1_cand_ids, _ = select_top_k_neural_candidates(k1_log_probs, top_k=k_top)
    k2_cand_ids, _ = select_top_k_neural_candidates(k2_log_probs, top_k=k_top)

    prompt_state = scorer.init_prompt_state()
    backbone_beams: list[BeamHypothesis] = [
        BeamHypothesis(
            words=(),
            word_indices=(),
            cache_state=prompt_state,
            score=0.0,
            neural_score_sum=0.0,
            lm_score_sum=0.0,
        )
    ]

    # 每个词位 i 的完整生命周期记录（供 C-gated 任意阈值 τ_gate 零开销精确重放）
    position_trajectories: list[dict] = [
        {
            "word_index": pos,
            "reference_word": (
                str(reference_words[pos])
                if reference_words is not None and pos < len(reference_words)
                else None
            ),
            "k0_word": None,
            "k0_word_id": None,
            "k0_score": None,
            "k1_word": None,
            "k1_word_id": None,
            "k1_score": None,
            "k1_old_score": None,
            "k1_delta_score": None,
            "k1_candidate_scores": {},
            "k2_word": None,
            "k2_word_id": None,
            "k2_score": None,
            "k2_old_score": None,
            "k2_delta_score": None,
            "k2_candidate_scores": {},
        }
        for pos in range(seg_len)
    ]

    screen_history: list[dict] = []
    previous_screen_word_ids: list[int] = []

    for arrival_t in range(seg_len):
        step_updates: list[dict] = []

        # ------------------------------------------------------------------
        # 1. 定稿主干 1 步束扩展（当 arrival_t >= 2 时，针对词位 commit_pos = arrival_t - 2）
        # ------------------------------------------------------------------
        if arrival_t >= 2:
            commit_pos = arrival_t - 2
            cands_k2 = k2_cand_ids[commit_pos].tolist()
            old_k0_id = position_trajectories[commit_pos]["k0_word_id"]
            old_k1_id = position_trajectories[commit_pos]["k1_word_id"]
            prev_disp_id = (
                previous_screen_word_ids[commit_pos]
                if commit_pos < len(previous_screen_word_ids)
                else old_k1_id
            )

            # 对当前 B 个主干束批量打分
            beam_states = [b.cache_state for b in backbone_beams]
            # 第 0 个主干束额外评估 old_k0_id 与 old_k1_id，确保任意门控历史词在 K2 下的得分精确可知
            extra_ids = [
                idx
                for idx in (old_k0_id, old_k1_id, prev_disp_id)
                if idx is not None and idx not in cands_k2
            ]
            cand_lists_per_beam = [
                cands_k2 + (extra_ids if b_idx == 0 else [])
                for b_idx in range(len(backbone_beams))
            ]
            lm_scores_per_beam = scorer.score_candidates_with_cache(
                beam_states, cand_lists_per_beam
            )

            # 记录 Top-1 主干前缀下所有候选词及旧词在 K2 状态下的单步得分 s_{t-2}^{(K2)}(w | u)
            top1_k2_step_scores: dict[int, float] = {}
            for w_id, lm_val in zip(cand_lists_per_beam[0], lm_scores_per_beam[0]):
                top1_k2_step_scores[int(w_id)] = float(
                    k2_log_probs[commit_pos, w_id]
                ) + lam_k2 * float(lm_val)

            expanded_candidates: list[
                tuple[float, float, float, int, int, tuple[str, ...], tuple[int, ...]]
            ] = []
            for b_idx, beam in enumerate(backbone_beams):
                lm_row = lm_scores_per_beam[b_idx]
                for c_pos, w_id in enumerate(cands_k2):
                    n_lp = float(k2_log_probs[commit_pos, w_id])
                    lm_lp = float(lm_row[c_pos])
                    step_score = n_lp + lam_k2 * lm_lp
                    new_total = beam.score + step_score
                    w_str = vocab_tuple[w_id]
                    expanded_candidates.append(
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

            expanded_candidates.sort(key=lambda item: (-item[0], item[5]))
            selected = expanded_candidates[:b_width]

            # 仅对胜出的 Top-B 束推进 1 词 KV-Cache
            adv_requests = [
                (backbone_beams[item[3]].cache_state, item[4]) for item in selected
            ]
            advanced_caches = scorer.advance_prefix_states(adv_requests)

            backbone_beams = [
                BeamHypothesis(
                    words=item[5],
                    word_indices=item[6],
                    cache_state=adv_cache,
                    score=item[0],
                    neural_score_sum=item[1],
                    lm_score_sum=item[2],
                )
                for item, adv_cache in zip(selected, advanced_caches)
            ]

            committed_word_id = backbone_beams[0].word_indices[commit_pos]
            committed_word = backbone_beams[0].words[commit_pos]
            new_k2_step_score = top1_k2_step_scores.get(
                committed_word_id,
                float(k2_log_probs[commit_pos, committed_word_id]),
            )
            old_k2_step_score = top1_k2_step_scores.get(
                int(prev_disp_id) if prev_disp_id is not None else committed_word_id,
                new_k2_step_score,
            )
            delta_k2 = float(new_k2_step_score - old_k2_step_score)

            traj_k2 = position_trajectories[commit_pos]
            traj_k2["k2_word"] = committed_word
            traj_k2["k2_word_id"] = int(committed_word_id)
            traj_k2["k2_score"] = float(new_k2_step_score)
            traj_k2["k2_old_score"] = float(old_k2_step_score)
            traj_k2["k2_delta_score"] = delta_k2
            traj_k2["k2_candidate_scores"] = {
                vocab_tuple[k_id]: float(val)
                for k_id, val in top1_k2_step_scores.items()
            }

            prev_word_str = (
                vocab_tuple[int(prev_disp_id)]
                if prev_disp_id is not None
                else committed_word
            )
            step_updates.append(
                {
                    "word_index": commit_pos,
                    "state": "K2",
                    "old_word": prev_word_str,
                    "new_word": committed_word,
                    "changed": bool(prev_word_str != committed_word),
                    "new_score": float(new_k2_step_score),
                    "old_score": float(old_k2_step_score),
                    "delta_score": delta_k2,
                }
            )

        # 当前 Top-1 定稿主干束（对应前缀 u_{0:arrival_t-2}）
        top1_backbone = backbone_beams[0]
        committed_prefix_words = list(top1_backbone.words)
        committed_prefix_ids = list(top1_backbone.word_indices)
        tail_cache_state = top1_backbone.cache_state

        # ------------------------------------------------------------------
        # 2. 尾部草稿延伸：词位 arrival_t - 1 (状态 K1)
        # ------------------------------------------------------------------
        draft_k1_word: str | None = None
        draft_k1_id: int | None = None
        if arrival_t >= 1:
            k1_pos = arrival_t - 1
            cands_k1 = k1_cand_ids[k1_pos].tolist()
            old_k0_id = position_trajectories[k1_pos]["k0_word_id"]
            extra_k1 = [old_k0_id] if old_k0_id is not None else []

            best_k1_id, best_k1_score, k1_scores_map, _ = (
                _score_single_step_candidates(
                    scorer,
                    tail_cache_state,
                    k1_log_probs[k1_pos],
                    cands_k1,
                    lam_k1,
                    extra_word_indices=extra_k1,
                )
            )
            draft_k1_id = int(best_k1_id)
            draft_k1_word = vocab_tuple[draft_k1_id]
            old_k1_score = k1_scores_map.get(
                int(old_k0_id) if old_k0_id is not None else draft_k1_id,
                best_k1_score,
            )
            delta_k1 = float(best_k1_score - old_k1_score)

            traj_k1 = position_trajectories[k1_pos]
            traj_k1["k1_word"] = draft_k1_word
            traj_k1["k1_word_id"] = draft_k1_id
            traj_k1["k1_score"] = float(best_k1_score)
            traj_k1["k1_old_score"] = float(old_k1_score)
            traj_k1["k1_delta_score"] = delta_k1
            traj_k1["k1_candidate_scores"] = {
                vocab_tuple[k_id]: float(val) for k_id, val in k1_scores_map.items()
            }

            old_k0_str = (
                vocab_tuple[int(old_k0_id)]
                if old_k0_id is not None
                else draft_k1_word
            )
            step_updates.append(
                {
                    "word_index": k1_pos,
                    "state": "K1",
                    "old_word": old_k0_str,
                    "new_word": draft_k1_word,
                    "changed": bool(old_k0_str != draft_k1_word),
                    "new_score": float(best_k1_score),
                    "old_score": float(old_k1_score),
                    "delta_score": delta_k1,
                }
            )

            # 临时推进 1 步草稿 KV-Cache 以供词位 arrival_t (状态 K0) 打分
            tail_cache_state = scorer.advance_prefix_states(
                [(tail_cache_state, draft_k1_id)]
            )[0]

        # ------------------------------------------------------------------
        # 3. 尾部草稿延伸：词位 arrival_t (状态 K0 初判显示)
        # ------------------------------------------------------------------
        k0_pos = arrival_t
        cands_k0 = k0_cand_ids[k0_pos].tolist()
        best_k0_id, best_k0_score, k0_scores_map, _ = _score_single_step_candidates(
            scorer,
            tail_cache_state,
            k0_log_probs[k0_pos],
            cands_k0,
            lam_k0,
        )
        draft_k0_id = int(best_k0_id)
        draft_k0_word = vocab_tuple[draft_k0_id]

        traj_k0 = position_trajectories[k0_pos]
        traj_k0["k0_word"] = draft_k0_word
        traj_k0["k0_word_id"] = draft_k0_id
        traj_k0["k0_score"] = float(best_k0_score)
        traj_k0["k0_candidate_scores"] = {
            vocab_tuple[k_id]: float(val) for k_id, val in k0_scores_map.items()
        }

        # 组装当前时刻 t 的完整屏幕词序列：committed(0..t-2) + [w_{t-1}^{(K1)}, w_t^{(K0)}]
        current_screen_words = list(committed_prefix_words)
        current_screen_ids = list(committed_prefix_ids)
        if draft_k1_word is not None and draft_k1_id is not None:
            current_screen_words.append(draft_k1_word)
            current_screen_ids.append(draft_k1_id)
        current_screen_words.append(draft_k0_word)
        current_screen_ids.append(draft_k0_id)

        # 校验定稿主干前缀 100% 未被篡改
        if arrival_t >= 2 and current_screen_words[: arrival_t - 1] != committed_prefix_words:
            raise RuntimeError("Scheme 1 invariant violated: committed K2 prefix was modified.")

        screen_history.append(
            {
                "arrival_step": arrival_t,
                "committed_prefix": committed_prefix_words,
                "draft_k1_word": draft_k1_word,
                "draft_k0_word": draft_k0_word,
                "screen_words": current_screen_words,
                "updates": step_updates,
            }
        )
        previous_screen_word_ids = current_screen_ids

    final_words = list(screen_history[-1]["screen_words"])
    initial_k0_words = [str(t["k0_word"]) for t in position_trajectories]
    committed_k2_words = list(backbone_beams[0].words)
    tail_suffix = final_words[len(committed_k2_words) :]
    beam_candidates = [list(b.words) + tail_suffix for b in backbone_beams]
    top5_words_per_pos: list[list[str]] = []
    for pos, traj in enumerate(position_trajectories):
        cand_map = (
            traj.get("k2_candidate_scores")
            or traj.get("k1_candidate_scores")
            or traj.get("k0_candidate_scores")
            or {}
        )
        if cand_map:
            ranked_w = [
                w
                for w, _ in sorted(
                    cand_map.items(), key=lambda kv: (-float(kv[1]), kv[0])
                )[:5]
            ]
        else:
            ranked_w = [final_words[pos]] if pos < len(final_words) else []
        top5_words_per_pos.append(ranked_w)

    return {
        "final_words": final_words,
        "committed_k2_words": committed_k2_words,
        "initial_k0_words": initial_k0_words,
        "beam_candidates": beam_candidates,
        "top5_words_per_pos": top5_words_per_pos,
        "screen_history": screen_history,
        "position_trajectories": position_trajectories,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


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


def decode_segments_streaming_batched(
    segment_log_probs_by_state_list: Sequence[dict[str, np.ndarray]],
    vocabulary: Sequence[str],
    scorer: CachedVocabularyLanguageScorer,
    *,
    language_weights: dict[str, float],
    beam_width: int = 5,
    candidate_top_k: int = 20,
    reference_words_list: Sequence[Sequence[str] | None] | None = None,
) -> list[dict]:
    """跨多个句段按到达时间步同步批量执行方案一流式解码（K2 定稿主干 + K1/K0 草稿延伸）。"""
    if not segment_log_probs_by_state_list:
        return []
    started = time.perf_counter()
    num_segs = len(segment_log_probs_by_state_list)
    vocab_tuple = tuple(str(w) for w in vocabulary)
    vocab_size = len(vocab_tuple)
    b_width = max(1, int(beam_width))
    k_top = min(max(1, int(candidate_top_k)), vocab_size)

    lam_k0 = float(language_weights.get("K0", 0.2))
    lam_k1 = float(language_weights.get("K1", 0.15))
    lam_k2 = float(language_weights.get("K2", 0.1))

    k0_mats: list[np.ndarray] = []
    k1_mats: list[np.ndarray] = []
    k2_mats: list[np.ndarray] = []
    k0_cands_list: list[np.ndarray] = []
    k1_cands_list: list[np.ndarray] = []
    k2_cands_list: list[np.ndarray] = []
    seg_lengths: list[int] = []

    for seg_dict in segment_log_probs_by_state_list:
        k0_m = np.asarray(seg_dict["K0"], dtype=np.float64)
        k1_m = np.asarray(seg_dict["K1"], dtype=np.float64)
        k2_m = np.asarray(seg_dict["K2"], dtype=np.float64)
        s_len = k0_m.shape[0]
        seg_lengths.append(s_len)
        k0_mats.append(k0_m)
        k1_mats.append(k1_m)
        k2_mats.append(k2_m)
        if s_len > 0:
            k0_cands_list.append(
                select_top_k_neural_candidates(k0_m, top_k=k_top)[0]
            )
            k1_cands_list.append(
                select_top_k_neural_candidates(k1_m, top_k=k_top)[0]
            )
            k2_cands_list.append(
                select_top_k_neural_candidates(k2_m, top_k=k_top)[0]
            )
        else:
            empty_c = np.zeros((0, k_top), dtype=np.int64)
            k0_cands_list.append(empty_c)
            k1_cands_list.append(empty_c)
            k2_cands_list.append(empty_c)

    prompt_state = scorer.init_prompt_state()
    backbone_beams_by_seg: list[list[BeamHypothesis]] = [
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

    trajectories_by_seg: list[list[dict]] = []
    for s_idx, s_len in enumerate(seg_lengths):
        refs = (
            reference_words_list[s_idx]
            if reference_words_list is not None
            and s_idx < len(reference_words_list)
            else None
        )
        trajectories_by_seg.append(
            [
                {
                    "word_index": pos,
                    "reference_word": (
                        str(refs[pos])
                        if refs is not None and pos < len(refs)
                        else None
                    ),
                    "k0_word": None,
                    "k0_word_id": None,
                    "k0_score": None,
                    "k1_word": None,
                    "k1_word_id": None,
                    "k1_score": None,
                    "k1_old_score": None,
                    "k1_delta_score": None,
                    "k1_candidate_scores": {},
                    "k2_word": None,
                    "k2_word_id": None,
                    "k2_score": None,
                    "k2_old_score": None,
                    "k2_delta_score": None,
                    "k2_candidate_scores": {},
                }
                for pos in range(s_len)
            ]
        )

    screen_history_by_seg: list[list[dict]] = [[] for _ in range(num_segs)]
    prev_screen_ids_by_seg: list[list[int]] = [[] for _ in range(num_segs)]
    full_k2_top5_by_seg: list[list[list[str]]] = [[] for _ in range(num_segs)]

    max_len = max(seg_lengths) if seg_lengths else 0
    for arrival_t in range(max_len):
        active_segs = [
            s_idx for s_idx in range(num_segs) if arrival_t < seg_lengths[s_idx]
        ]
        if not active_segs:
            break

        step_updates_by_seg: dict[int, list[dict]] = {
            s_idx: [] for s_idx in active_segs
        }

        # ------------------------------------------------------------------
        # 1. 批量定稿主干 K2 扩展（arrival_t >= 2）
        # ------------------------------------------------------------------
        k2_segs = [s_idx for s_idx in active_segs if arrival_t >= 2]
        if k2_segs:
            commit_pos = arrival_t - 2
            flat_k2_states: list[PrefixCacheState] = []
            flat_k2_cands: list[list[int]] = []
            k2_meta: list[
                tuple[int, int, int, list[int], list[list[int]], int | None]
            ] = []

            for s_idx in k2_segs:
                s_beams = backbone_beams_by_seg[s_idx]
                cands_k2 = k2_cands_list[s_idx][commit_pos].tolist()
                traj_pos = trajectories_by_seg[s_idx][commit_pos]
                old_k0_id = traj_pos["k0_word_id"]
                old_k1_id = traj_pos["k1_word_id"]
                prev_ids = prev_screen_ids_by_seg[s_idx]
                prev_disp_id = (
                    prev_ids[commit_pos]
                    if commit_pos < len(prev_ids)
                    else old_k1_id
                )
                extra_ids = [
                    idx
                    for idx in (old_k0_id, old_k1_id, prev_disp_id)
                    if idx is not None and idx not in cands_k2
                ]
                cand_lists_per_beam = [
                    cands_k2 + (extra_ids if b_idx == 0 else [])
                    for b_idx in range(len(s_beams))
                ]
                start_off = len(flat_k2_states)
                for b, c_list in zip(s_beams, cand_lists_per_beam):
                    flat_k2_states.append(b.cache_state)
                    flat_k2_cands.append(c_list)
                k2_meta.append(
                    (
                        s_idx,
                        start_off,
                        len(flat_k2_states),
                        cands_k2,
                        cand_lists_per_beam,
                        prev_disp_id,
                    )
                )

            flat_k2_lm_scores = scorer.score_candidates_with_cache(
                flat_k2_states, flat_k2_cands
            )

            flat_k2_adv_reqs: list[tuple[PrefixCacheState, int]] = []
            k2_adv_slices: list[
                tuple[
                    int,
                    int,
                    int,
                    list[tuple],
                    dict[int, float],
                    int | None,
                ]
            ] = []

            for (
                s_idx,
                start_off,
                end_off,
                cands_k2,
                cand_lists_per_beam,
                prev_disp_id,
            ) in k2_meta:
                s_beams = backbone_beams_by_seg[s_idx]
                lm_scores_per_beam = flat_k2_lm_scores[start_off:end_off]
                k2_mat = k2_mats[s_idx]

                top1_k2_step_scores: dict[int, float] = {}
                for w_id, lm_val in zip(
                    cand_lists_per_beam[0], lm_scores_per_beam[0]
                ):
                    top1_k2_step_scores[int(w_id)] = float(
                        k2_mat[commit_pos, w_id]
                    ) + lam_k2 * float(lm_val)
                step_pairs_k2 = [
                    (top1_k2_step_scores[int(w_id)], vocab_tuple[int(w_id)])
                    for w_id in cands_k2
                ]
                step_pairs_k2.sort(key=lambda kv: (-kv[0], kv[1]))
                full_k2_top5_by_seg[s_idx].append(
                    [w for _, w in step_pairs_k2[:5]]
                )

                selected = _select_top_beam_expansions(
                    s_beams,
                    cands_k2,
                    k2_mat[commit_pos],
                    lm_scores_per_beam,
                    lam_k2,
                    b_width,
                    vocab_tuple,
                )

                adv_start = len(flat_k2_adv_reqs)
                for item in selected:
                    flat_k2_adv_reqs.append(
                        (s_beams[item[3]].cache_state, item[4])
                    )
                k2_adv_slices.append(
                    (
                        s_idx,
                        adv_start,
                        len(flat_k2_adv_reqs),
                        selected,
                        top1_k2_step_scores,
                        prev_disp_id,
                    )
                )

            flat_k2_adv_caches = scorer.advance_prefix_states(flat_k2_adv_reqs)
            for (
                s_idx,
                adv_start,
                adv_end,
                selected,
                top1_k2_step_scores,
                prev_disp_id,
            ) in k2_adv_slices:
                adv_caches = flat_k2_adv_caches[adv_start:adv_end]
                backbone_beams_by_seg[s_idx] = [
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
                committed_word_id = backbone_beams_by_seg[s_idx][0].word_indices[
                    commit_pos
                ]
                committed_word = backbone_beams_by_seg[s_idx][0].words[
                    commit_pos
                ]
                k2_mat = k2_mats[s_idx]
                new_k2_step_score = top1_k2_step_scores.get(
                    committed_word_id,
                    float(k2_mat[commit_pos, committed_word_id]),
                )
                old_k2_step_score = top1_k2_step_scores.get(
                    int(prev_disp_id)
                    if prev_disp_id is not None
                    else committed_word_id,
                    new_k2_step_score,
                )
                delta_k2 = float(new_k2_step_score - old_k2_step_score)

                traj_k2 = trajectories_by_seg[s_idx][commit_pos]
                traj_k2["k2_word"] = committed_word
                traj_k2["k2_word_id"] = int(committed_word_id)
                traj_k2["k2_score"] = float(new_k2_step_score)
                traj_k2["k2_old_score"] = float(old_k2_step_score)
                traj_k2["k2_delta_score"] = delta_k2
                traj_k2["k2_candidate_scores"] = {
                    vocab_tuple[k_id]: float(val)
                    for k_id, val in top1_k2_step_scores.items()
                }
                prev_word_str = (
                    vocab_tuple[int(prev_disp_id)]
                    if prev_disp_id is not None
                    else committed_word
                )
                step_updates_by_seg[s_idx].append(
                    {
                        "word_index": commit_pos,
                        "state": "K2",
                        "old_word": prev_word_str,
                        "new_word": committed_word,
                        "changed": bool(prev_word_str != committed_word),
                        "new_score": float(new_k2_step_score),
                        "old_score": float(old_k2_step_score),
                        "delta_score": delta_k2,
                    }
                )

        # ------------------------------------------------------------------
        # 2. 批量尾部草稿延伸：词位 arrival_t - 1 (状态 K1)
        # ------------------------------------------------------------------
        tail_cache_by_seg: dict[int, PrefixCacheState] = {
            s_idx: backbone_beams_by_seg[s_idx][0].cache_state
            for s_idx in active_segs
        }
        draft_k1_word_by_seg: dict[int, str | None] = {
            s_idx: None for s_idx in active_segs
        }
        draft_k1_id_by_seg: dict[int, int | None] = {
            s_idx: None for s_idx in active_segs
        }

        k1_segs = [s_idx for s_idx in active_segs if arrival_t >= 1]
        if k1_segs:
            k1_pos = arrival_t - 1
            flat_k1_states: list[PrefixCacheState] = []
            flat_k1_cands: list[list[int]] = []
            k1_meta: list[tuple[int, list[int], list[int], int | None]] = []

            for s_idx in k1_segs:
                cands_k1 = k1_cands_list[s_idx][k1_pos].tolist()
                old_k0_id = trajectories_by_seg[s_idx][k1_pos]["k0_word_id"]
                extra_k1 = [int(old_k0_id)] if old_k0_id is not None else []
                all_eval = list(dict.fromkeys(cands_k1 + extra_k1))
                flat_k1_states.append(tail_cache_by_seg[s_idx])
                flat_k1_cands.append(all_eval)
                k1_meta.append((s_idx, cands_k1, all_eval, old_k0_id))

            flat_k1_lm_scores = scorer.score_candidates_with_cache(
                flat_k1_states, flat_k1_cands
            )

            flat_k1_adv_reqs: list[tuple[PrefixCacheState, int]] = []
            for (s_idx, cands_k1, all_eval, old_k0_id), lm_row in zip(
                k1_meta, flat_k1_lm_scores
            ):
                k1_row_lp = k1_mats[s_idx][k1_pos]
                k1_scores_map: dict[int, float] = {
                    int(w_idx): float(k1_row_lp[w_idx]) + lam_k1 * float(lm_val)
                    for w_idx, lm_val in zip(all_eval, lm_row)
                }
                best_k1_id = int(
                    max(cands_k1, key=lambda idx: (k1_scores_map[idx], -idx))
                )
                best_k1_score = k1_scores_map[best_k1_id]
                draft_k1_id_by_seg[s_idx] = best_k1_id
                draft_k1_word = vocab_tuple[best_k1_id]
                draft_k1_word_by_seg[s_idx] = draft_k1_word

                old_k1_score = k1_scores_map.get(
                    int(old_k0_id) if old_k0_id is not None else best_k1_id,
                    best_k1_score,
                )
                delta_k1 = float(best_k1_score - old_k1_score)

                traj_k1 = trajectories_by_seg[s_idx][k1_pos]
                traj_k1["k1_word"] = draft_k1_word
                traj_k1["k1_word_id"] = best_k1_id
                traj_k1["k1_score"] = float(best_k1_score)
                traj_k1["k1_old_score"] = float(old_k1_score)
                traj_k1["k1_delta_score"] = delta_k1
                traj_k1["k1_candidate_scores"] = {
                    vocab_tuple[k_id]: float(val)
                    for k_id, val in k1_scores_map.items()
                }

                old_k0_str = (
                    vocab_tuple[int(old_k0_id)]
                    if old_k0_id is not None
                    else draft_k1_word
                )
                step_updates_by_seg[s_idx].append(
                    {
                        "word_index": k1_pos,
                        "state": "K1",
                        "old_word": old_k0_str,
                        "new_word": draft_k1_word,
                        "changed": bool(old_k0_str != draft_k1_word),
                        "new_score": float(best_k1_score),
                        "old_score": float(old_k1_score),
                        "delta_score": delta_k1,
                    }
                )
                flat_k1_adv_reqs.append((tail_cache_by_seg[s_idx], best_k1_id))

            flat_k1_adv_caches = scorer.advance_prefix_states(flat_k1_adv_reqs)
            for (s_idx, _, _, _), adv_cache in zip(k1_meta, flat_k1_adv_caches):
                tail_cache_by_seg[s_idx] = adv_cache

        # ------------------------------------------------------------------
        # 3. 批量尾部草稿延伸：词位 arrival_t (状态 K0 初判显示)
        # ------------------------------------------------------------------
        k0_pos = arrival_t
        flat_k0_states = [tail_cache_by_seg[s_idx] for s_idx in active_segs]
        flat_k0_cands = [
            k0_cands_list[s_idx][k0_pos].tolist() for s_idx in active_segs
        ]
        flat_k0_lm_scores = scorer.score_candidates_with_cache(
            flat_k0_states, flat_k0_cands
        )

        for s_idx, cands_k0, lm_row in zip(
            active_segs, flat_k0_cands, flat_k0_lm_scores
        ):
            k0_row_lp = k0_mats[s_idx][k0_pos]
            k0_scores_map = {
                int(w_idx): float(k0_row_lp[w_idx]) + lam_k0 * float(lm_val)
                for w_idx, lm_val in zip(cands_k0, lm_row)
            }
            draft_k0_id = int(
                max(cands_k0, key=lambda idx: (k0_scores_map[idx], -idx))
            )
            best_k0_score = k0_scores_map[draft_k0_id]
            draft_k0_word = vocab_tuple[draft_k0_id]

            traj_k0 = trajectories_by_seg[s_idx][k0_pos]
            traj_k0["k0_word"] = draft_k0_word
            traj_k0["k0_word_id"] = draft_k0_id
            traj_k0["k0_score"] = float(best_k0_score)
            traj_k0["k0_candidate_scores"] = {
                vocab_tuple[k_id]: float(val)
                for k_id, val in k0_scores_map.items()
            }

            top1_backbone = backbone_beams_by_seg[s_idx][0]
            committed_prefix_words = list(top1_backbone.words)
            committed_prefix_ids = list(top1_backbone.word_indices)

            current_screen_words = list(committed_prefix_words)
            current_screen_ids = list(committed_prefix_ids)
            d_k1_w = draft_k1_word_by_seg[s_idx]
            d_k1_id = draft_k1_id_by_seg[s_idx]
            if d_k1_w is not None and d_k1_id is not None:
                current_screen_words.append(d_k1_w)
                current_screen_ids.append(d_k1_id)
            current_screen_words.append(draft_k0_word)
            current_screen_ids.append(draft_k0_id)

            screen_history_by_seg[s_idx].append(
                {
                    "arrival_step": arrival_t,
                    "committed_prefix": committed_prefix_words,
                    "draft_k1_word": d_k1_w,
                    "draft_k0_word": draft_k0_word,
                    "screen_words": current_screen_words,
                    "updates": step_updates_by_seg[s_idx],
                }
            )
            prev_screen_ids_by_seg[s_idx] = current_screen_ids

    committed_online_words_by_seg: list[list[str]] = [
        list(backbone_beams_by_seg[s_idx][0].words)
        if seg_lengths[s_idx] > 0
        else []
        for s_idx in range(num_segs)
    ]

    # 推进最后 1~2 个末位词的 K2 主干束搜索，使单次流式扫描即可同时产出完整句长 K2 静态束搜索结果
    for _tail_step in range(2):
        tail_k2_segs = [
            s_idx
            for s_idx in range(num_segs)
            if seg_lengths[s_idx] > 0
            and len(backbone_beams_by_seg[s_idx][0].words) < seg_lengths[s_idx]
        ]
        if not tail_k2_segs:
            break
        flat_tail_states: list[PrefixCacheState] = []
        flat_tail_cands: list[list[int]] = []
        tail_meta: list[tuple[int, int, int, int, list[int]]] = []
        for s_idx in tail_k2_segs:
            pos = len(backbone_beams_by_seg[s_idx][0].words)
            s_beams = backbone_beams_by_seg[s_idx]
            cands_k2 = k2_cands_list[s_idx][pos].tolist()
            start_off = len(flat_tail_states)
            for b in s_beams:
                flat_tail_states.append(b.cache_state)
                flat_tail_cands.append(cands_k2)
            tail_meta.append(
                (s_idx, pos, start_off, len(flat_tail_states), cands_k2)
            )
        flat_tail_lm = scorer.score_candidates_with_cache(
            flat_tail_states, flat_tail_cands
        )
        flat_tail_adv_reqs: list[tuple[PrefixCacheState, int]] = []
        tail_adv_slices: list[tuple[int, int, int, list[tuple]]] = []
        for s_idx, pos, start_off, end_off, cands_k2 in tail_meta:
            s_beams = backbone_beams_by_seg[s_idx]
            s_lm_rows = flat_tail_lm[start_off:end_off]
            k2_mat = k2_mats[s_idx]
            if cands_k2 and s_lm_rows:
                step_pairs = [
                    (
                        float(k2_mat[pos, w_id]) + lam_k2 * float(lm_val),
                        vocab_tuple[w_id],
                    )
                    for w_id, lm_val in zip(cands_k2, s_lm_rows[0])
                ]
                step_pairs.sort(key=lambda kv: (-kv[0], kv[1]))
                full_k2_top5_by_seg[s_idx].append([w for _, w in step_pairs[:5]])
            else:
                full_k2_top5_by_seg[s_idx].append([])

            expanded: list[tuple] = []
            for b_idx, (beam, lm_row) in enumerate(zip(s_beams, s_lm_rows)):
                for c_pos, w_id in enumerate(cands_k2):
                    n_lp = float(k2_mat[pos, w_id])
                    lm_lp = float(lm_row[c_pos])
                    step_score = n_lp + lam_k2 * lm_lp
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
            if pos < seg_lengths[s_idx] - 1:
                adv_start = len(flat_tail_adv_reqs)
                for item in selected:
                    flat_tail_adv_reqs.append(
                        (s_beams[item[3]].cache_state, item[4])
                    )
                tail_adv_slices.append(
                    (s_idx, adv_start, len(flat_tail_adv_reqs), selected)
                )
            else:
                backbone_beams_by_seg[s_idx] = [
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
        if flat_tail_adv_reqs:
            flat_tail_caches = scorer.advance_prefix_states(flat_tail_adv_reqs)
            for s_idx, adv_start, adv_end, selected in tail_adv_slices:
                adv_caches = flat_tail_caches[adv_start:adv_end]
                backbone_beams_by_seg[s_idx] = [
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
                    "committed_k2_words": [],
                    "full_k2_words": [],
                    "full_k2_beam_candidates": [],
                    "full_k2_top5_words_per_pos": [],
                    "initial_k0_words": [],
                    "beam_candidates": [],
                    "top5_words_per_pos": [],
                    "screen_history": [],
                    "position_trajectories": [],
                    "elapsed_seconds": 0.0,
                }
            )
            continue
        hist = screen_history_by_seg[s_idx]
        trajs = trajectories_by_seg[s_idx]
        final_w = list(hist[-1]["screen_words"])
        committed_w = committed_online_words_by_seg[s_idx]
        full_k2_w = list(backbone_beams_by_seg[s_idx][0].words)
        full_k2_beam_cands = [
            list(b.words) for b in backbone_beams_by_seg[s_idx]
        ]
        tail_suffix = final_w[len(committed_w) :]
        beam_cands = [
            list(b.words)[: len(committed_w)] + tail_suffix
            for b in backbone_beams_by_seg[s_idx]
        ]
        top5_per_pos: list[list[str]] = []
        for pos, traj in enumerate(trajs):
            cand_map = (
                traj.get("k2_candidate_scores")
                or traj.get("k1_candidate_scores")
                or traj.get("k0_candidate_scores")
                or {}
            )
            if cand_map:
                ranked_w = [
                    w
                    for w, _ in sorted(
                        cand_map.items(), key=lambda kv: (-float(kv[1]), kv[0])
                    )[:5]
                ]
            else:
                ranked_w = [final_w[pos]] if pos < len(final_w) else []
            top5_per_pos.append(ranked_w)

        outputs.append(
            {
                "final_words": final_w,
                "committed_k2_words": committed_w,
                "full_k2_words": full_k2_w,
                "full_k2_beam_candidates": full_k2_beam_cands,
                "full_k2_top5_words_per_pos": full_k2_top5_by_seg[s_idx],
                "initial_k0_words": [str(t["k0_word"]) for t in trajs],
                "beam_candidates": beam_cands,
                "top5_words_per_pos": top5_per_pos,
                "screen_history": hist,
                "position_trajectories": trajs,
                "elapsed_seconds": per_seg_elapsed,
            }
        )
    return outputs


STATIC_DECODE_CONDITIONS: frozenset[str] = frozenset({"S0", "K0", "K1", "K2", "F"})


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
    return {
        "segment_index": seg["segment_index"],
        "group_id": seg["group_id"],
        "subject_id": seg["subject_id"],
        "recording_id": seg["recording_id"],
        "scope": seg["scope"],
        "material_id": str(seg.get("material_id", "")),
        "block_id": seg.get("block_id", seg["scope"]),
        "reference_words": list(seg["reference_words"]),
    }


def decode_dataset_segment_jobs(
    exported_payload: dict,
    scorer: CachedVocabularyLanguageScorer,
    hyperparameters: dict,
    jobs: dict[str, dict],
    *,
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_batch_size: int = 32,
    partition_filter: str | None = None,
    max_segments: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict[str, dict]:
    """将多个解码作业（不同条件 / 对齐方式 / 受限词表）合并为同一批次同步束搜索。

    每个作业为字典：`condition`（S0/K0/K1/K2/F/S3/LM-Only）、`alignment`（clean/shifted）、
    可选 `vocab_size_limit` / `allowed_vocab_indices` / `include_screen_history`。
    - 静态作业（S0/K0/K1/K2/F 以及不记录屏幕轨迹的 S3 = 静态 K2）跨作业逐句段 λ 合并为一次
      `decode_segments_static_batched` 批处理；
    - 流式作业（记录屏幕轨迹的 S3）跨对齐方式合并为一次 `decode_segments_streaming_batched` 批处理；
    - 同一句段的不同作业相邻排入同一批次，最大化共享语言模型前缀 KV-Cache 与转移缓存命中；
    - LM-Only 与脑信号无关，仍按句长缓存单独解码。
    返回 `{job_key: bundle}`，bundle 结构与 `decode_dataset_segments` 完全一致。
    """
    from tqdm.auto import tqdm

    vocabulary = list(exported_payload["vocabulary"])
    segments = [
        seg
        for seg in exported_payload["segments"]
        if partition_filter is None or seg.get("partition") == partition_filter
    ]
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
    stream_items: list[tuple[int, int, str, dict[str, np.ndarray], int]] = []
    job_meta: dict[str, dict] = {}
    lm_only_jobs: list[str] = []

    for job_order, (job_key, job) in enumerate(jobs.items()):
        condition = str(job.get("condition", "S3"))
        cond_upper = condition.upper()
        alignment = str(job.get("alignment", "clean"))
        include_screen = bool(job.get("include_screen_history", True))
        if cond_upper == "LM-ONLY":
            lm_only_jobs.append(job_key)
            continue
        if cond_upper != "S3" and cond_upper not in STATIC_DECODE_CONDITIONS:
            raise ValueError(
                f"不支持的解码条件 {condition!r}，可选：'S3', 'S0', 'K0', 'K1', 'K2', 'F', 'LM-Only'"
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
        is_streaming = cond_upper == "S3" and include_screen
        kind = "streaming" if is_streaming else (
            "static_s3" if cond_upper == "S3" else "static"
        )
        job_meta[job_key] = {
            "condition": condition,
            "alignment": alignment,
            "vocabulary_size": eff_vocab_size,
            "active_set": active_set,
            "kind": kind,
            "records": [None] * n_segs,
            "screens": [None] * n_segs,
            "elapsed": 0.0,
        }

        if is_streaming:
            lp_by_state: dict[str, np.ndarray] = {}
            for st in ("K0", "K1", "K2"):
                t_st, _ = _get_state_params(st)
                lp_by_state[st] = compute_neural_log_probabilities(
                    score_bank[st],
                    t_st,
                    top_k=eff_top_k,
                    allowed_vocab_indices=active_idx,
                )
            for seg_pos, seg in enumerate(segments):
                ev = seg["event_indices"]
                stream_items.append(
                    (
                        seg_pos,
                        job_order,
                        job_key,
                        {st: lp_by_state[st][ev] for st in ("K0", "K1", "K2")},
                        eff_top_k,
                    )
                )
        else:
            state_name = "K2" if cond_upper == "S3" else cond_upper
            t_r, lam_r = _get_state_params(state_name)
            full_lp = compute_neural_log_probabilities(
                score_bank[state_name],
                t_r,
                top_k=eff_top_k,
                allowed_vocab_indices=active_idx,
            )
            for seg_pos, seg in enumerate(segments):
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
    stream_items.sort(key=lambda item: (item[4], item[0], item[1]))
    total_items = len(static_items) + len(stream_items)
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

        start_i = 0
        while start_i < len(stream_items):
            top_k_val = stream_items[start_i][4]
            end_i = start_i
            while (
                end_i < len(stream_items)
                and end_i - start_i < batch_sz
                and stream_items[end_i][4] == top_k_val
            ):
                end_i += 1
            chunk = stream_items[start_i:end_i]
            lam_map = {st: _get_state_params(st)[1] for st in ("K0", "K1", "K2")}
            decs = decode_segments_streaming_batched(
                [item[3] for item in chunk],
                vocabulary,
                scorer,
                language_weights=lam_map,
                beam_width=beam_width,
                candidate_top_k=top_k_val,
                reference_words_list=[
                    segments[item[0]]["reference_words"] for item in chunk
                ],
            )
            for (seg_pos, _, job_key, _, _), dec in zip(chunk, decs):
                meta = job_meta[job_key]
                seg = segments[seg_pos]
                seg_iv_mask = _job_in_vocab_mask(seg, meta["active_set"])
                full_k2_words = list(dec.get("full_k2_words", dec["final_words"]))
                record = _segment_record_base(seg)
                record.update(
                    {
                        "predicted_words": full_k2_words,
                        "streaming_screen_words": list(dec["final_words"]),
                        "initial_k0_words": list(dec["initial_k0_words"]),
                        "committed_k2_words": full_k2_words,
                        "beam_candidates": list(
                            dec.get(
                                "full_k2_beam_candidates",
                                dec.get("beam_candidates", []),
                            )
                        ),
                        "top5_words_per_pos": list(
                            dec.get(
                                "full_k2_top5_words_per_pos",
                                dec.get("top5_words_per_pos", []),
                            )
                        ),
                        "in_vocabulary_mask": seg_iv_mask,
                        "elapsed_seconds": dec["elapsed_seconds"],
                    }
                )
                meta["records"][seg_pos] = record
                meta["screens"][seg_pos] = {
                    "segment_index": seg["segment_index"],
                    "group_id": seg["group_id"],
                    "subject_id": seg["subject_id"],
                    "reference_words": list(seg["reference_words"]),
                    "in_vocabulary_mask": seg_iv_mask,
                    "screen_history": dec["screen_history"],
                    "position_trajectories": dec["position_trajectories"],
                }
                meta["elapsed"] += float(dec.get("elapsed_seconds", 0.0))
            _tick(len(chunk))
            start_i = end_i
    finally:
        if pbar is not None:
            pbar.close()

    for job_key in jobs:
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
                partition_filter=partition_filter,
                max_segments=max_segments,
            )
            continue
        meta = job_meta[job_key]
        total_elapsed = float(meta["elapsed"])
        results[job_key] = {
            "condition": meta["condition"],
            "alignment": meta["alignment"],
            "vocabulary_size": meta["vocabulary_size"],
            "segment_count": n_segs,
            "total_elapsed_seconds": round(total_elapsed, 4),
            "mean_seconds_per_segment": round(total_elapsed / max(1, n_segs), 4),
            "final_sentences": [r for r in meta["records"] if r is not None],
            "screen_histories": [s for s in meta["screens"] if s is not None],
        }
    return results


def _job_in_vocab_mask(seg: dict, active_set: set[str] | None) -> list[bool]:
    if active_set is not None:
        return [str(w) in active_set for w in seg.get("reference_words", [])]
    return list(seg["in_vocabulary_mask"])


def decode_dataset_segments(
    exported_payload: dict,
    scorer: CachedVocabularyLanguageScorer,
    hyperparameters: dict,
    *,
    condition: str = "S3",
    alignment: str = "clean",
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_batch_size: int = 32,
    vocab_size_limit: int | None = None,
    allowed_vocab_indices: Sequence[int] | None = None,
    partition_filter: str | None = None,
    max_segments: int | None = None,
    include_screen_history: bool = True,
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
                    "include_screen_history": include_screen_history,
                }
            },
            beam_width=beam_width,
            candidate_top_k=candidate_top_k,
            segment_batch_size=segment_batch_size,
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

    segments = [
        seg
        for seg in exported_payload["segments"]
        if partition_filter is None or seg.get("partition") == partition_filter
    ]
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
        # LM-Only 与受试者及脑信号无关，相同句长 L 仅需解码一次即可复用
        length_cache: dict[int, dict] = {}
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
        "screen_histories": [],
    }


def _average_subject_score_rows_rms_preserved(
    subject_score_matrices: Sequence[torch.Tensor | np.ndarray],
) -> torch.Tensor:
    """跨 k 名受试者的同一测试句段平均神经余弦得分，并执行 RMS 范数保持缩放。

    原因：k 名受试者的单位向量求均值后，非锁相背景噪声抵消导致合成向量模长按 ~1/sqrt(k) 收缩；
    通过按单受试者平均 RMS 重新对齐行模长，既 100% 保持每词位的跨受试者平均候选排序不变，
    又避免将温度 T_r 变相放大 sqrt(k) 倍。
    """
    tensors = [
        m.detach().cpu().float()
        if isinstance(m, torch.Tensor)
        else torch.from_numpy(np.asarray(m, dtype=np.float32))
        for m in subject_score_matrices
    ]
    if len(tensors) == 1:
        return tensors[0]
    stacked = torch.stack(tensors, dim=0)  # [k, L, |V|]
    mean_mat = stacked.mean(dim=0)  # [L, |V|]
    single_rms = torch.sqrt(
        (stacked * stacked).mean(dim=-1, keepdim=True) + 1e-12
    ).mean(dim=0)  # [L, 1]
    avg_rms = torch.sqrt(
        (mean_mat * mean_mat).mean(dim=-1, keepdim=True) + 1e-12
    )  # [L, 1]
    scale = (single_rms / avg_rms).clamp(min=1.0, max=5.0)
    return mean_mat * scale


def aggregate_multi_subject_payload(
    exported_payload: dict,
    k_observations: int,
) -> dict:
    """构造跨 k 名受试者神经观测聚合的合成 .pt 载荷（对标 d'Ascoli 2024 / SimpleB2T 2026）。"""
    k_req = max(1, int(k_observations))
    all_segments = list(exported_payload.get("segments") or [])
    grouped: dict[tuple, list[dict]] = {}
    for seg in all_segments:
        ref_tuple = tuple(str(w) for w in seg.get("reference_words", []))
        key = (
            str(seg.get("scope", "")),
            str(seg.get("material_id", "")),
            ref_tuple,
        )
        grouped.setdefault(key, []).append(seg)

    clean_bank = exported_payload["clean_scores"]
    shifted_bank = exported_payload.get("shifted_scores") or clean_bank
    state_keys = list(clean_bank.keys())

    agg_clean_lists: dict[str, list[torch.Tensor]] = {s: [] for s in state_keys}
    agg_shifted_lists: dict[str, list[torch.Tensor]] = {
        s: [] for s in state_keys
    }
    agg_segments: list[dict] = []
    cursor = 0

    for group_idx, (_, seg_group) in enumerate(grouped.items()):
        seg_group_sorted = sorted(
            seg_group, key=lambda item: str(item.get("subject_id", ""))
        )
        m_avail = len(seg_group_sorted)
        if m_avail == 0:
            continue
        k_eff = min(k_req, m_avail)
        # 采用确定性轮转窗口选取 k_eff 名受试者，确保全体受试者在语料上均匀代表
        chosen = [
            seg_group_sorted[(group_idx + offset) % m_avail]
            for offset in range(k_eff)
        ]
        template = chosen[0]
        seg_len = len(template["event_indices"])
        if seg_len == 0:
            continue

        for s in state_keys:
            c_rows = [
                clean_bank[s][member["event_indices"]] for member in chosen
            ]
            agg_clean_lists[s].append(
                _average_subject_score_rows_rms_preserved(c_rows)
            )
            if s in shifted_bank:
                sh_rows = [
                    shifted_bank[s][member["event_indices"]]
                    for member in chosen
                ]
                agg_shifted_lists[s].append(
                    _average_subject_score_rows_rms_preserved(sh_rows)
                )

        new_indices = list(range(cursor, cursor + seg_len))
        cursor += seg_len
        subj_label = f"avg_k{k_eff}(" + "+".join(
            str(c.get("subject_id", "")) for c in chosen
        ) + ")"
        agg_segments.append(
            {
                "segment_index": len(agg_segments),
                "group_id": f"k{k_eff}::{template.get('group_id', len(agg_segments))}",
                "subject_id": f"multi_subject_k{k_eff}",
                "subject_group_members": [
                    str(c.get("subject_id", "")) for c in chosen
                ],
                "k_observations": int(k_eff),
                "recording_id": str(template.get("recording_id", "")),
                "scope": str(template.get("scope", "")),
                "block_id": str(
                    template.get("block_id", template.get("scope", ""))
                ),
                "material_id": str(template.get("material_id", "")),
                "partition": str(template.get("partition", "evaluation")),
                "event_indices": new_indices,
                "reference_words": list(template["reference_words"]),
                "in_vocabulary_mask": list(template["in_vocabulary_mask"]),
                "subject_combination_label": subj_label,
            }
        )

    vocab_size = len(exported_payload["vocabulary"])
    agg_clean = {
        s: (
            torch.cat(agg_clean_lists[s], dim=0)
            if agg_clean_lists[s]
            else torch.zeros((0, vocab_size), dtype=torch.float32)
        )
        for s in state_keys
    }
    agg_shifted = {
        s: (
            torch.cat(agg_shifted_lists[s], dim=0)
            if agg_shifted_lists[s]
            else agg_clean[s]
        )
        for s in state_keys
    }

    return {
        **exported_payload,
        "k_observations": k_req,
        "segments": agg_segments,
        "clean_scores": agg_clean,
        "shifted_scores": agg_shifted,
    }


def run_multi_subject_observation_sweep(
    exported_payload: dict,
    scorer: CachedVocabularyLanguageScorer,
    hyperparameters: dict,
    *,
    k_values: Sequence[int] = (1, 2, 5, 10),
    k_observations: Sequence[int] | None = None,
    vocab_sizes: Sequence[int] | None = (50, 100, 250),
    condition: str = "S3",
    alignment: str = "clean",
    single_subject_s3_bundle: dict | None = None,
    single_subject_full_vocab_bundles: dict[str, dict] | None = None,
    single_subject_vocab_scaling_bundles: dict[str, dict] | None = None,
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_batch_size: int = 32,
    max_segments: int | None = None,
    show_progress: bool = False,
) -> dict[str, dict]:
    """扫描 k in {1, 2, 5, 10} 跨受试者多观测神经聚合下的序列解码结果（同时支持全词库与 |V| in {50, 100, 250} 受限核心词表）。"""
    from braindecoding.decoding.neural_score_exporter import (
        resolve_vocabulary_frequency_order,
    )

    active_k_values = k_observations if k_observations is not None else k_values
    all_subjects = sorted(
        {
            str(seg.get("subject_id", ""))
            for seg in exported_payload.get("segments", [])
            if seg.get("subject_id")
        }
    )
    max_avail_k = max(1, len(all_subjects))
    vocabulary = [str(w) for w in exported_payload.get("vocabulary") or []]
    full_v_size = len(vocabulary)
    freq_indices, _, _ = (
        resolve_vocabulary_frequency_order(exported_payload)
        if full_v_size > 0
        else ([], [], {})
    )

    eff_vocab_sizes: list[int] = []
    if vocab_sizes is not None and full_v_size > 0:
        for raw_v in vocab_sizes:
            v_int = int(raw_v)
            if v_int <= 0:
                continue
            eff_v = full_v_size if v_int >= full_v_size else v_int
            if eff_v not in eff_vocab_sizes:
                eff_vocab_sizes.append(eff_v)
        if full_v_size not in eff_vocab_sizes:
            eff_vocab_sizes.append(full_v_size)
        eff_vocab_sizes.sort()

    bundles_by_k: dict[str, dict] = {}
    seen_eff_k: dict[int, dict] = {}

    for k_raw in active_k_values:
        k_int = max(1, int(k_raw))
        k_eff = min(k_int, max_avail_k)
        if k_eff in seen_eff_k:
            cached = dict(seen_eff_k[k_eff])
            cached["k_observations"] = k_int
            cached["effective_k_observations"] = k_eff
            bundles_by_k[f"k={k_int}"] = cached
            continue

        agg_payload = (
            exported_payload
            if k_eff == 1
            else aggregate_multi_subject_payload(exported_payload, k_eff)
        )
        available_clean = agg_payload.get("clean_scores") or {}
        available_shifted = agg_payload.get("shifted_scores") or {}

        reuse_bundle: dict | None = None
        if k_int == 1 and single_subject_s3_bundle is not None:
            reuse_bundle = dict(single_subject_s3_bundle)
        elif (
            k_int == 1
            and isinstance(single_subject_full_vocab_bundles, dict)
            and "S3_clean" in single_subject_full_vocab_bundles
        ):
            reuse_bundle = dict(single_subject_full_vocab_bundles["S3_clean"])

        if k_int == 1 and isinstance(single_subject_full_vocab_bundles, dict):
            if reuse_bundle is None:
                reuse_bundle = decode_dataset_segments(
                    agg_payload,
                    scorer,
                    hyperparameters,
                    condition=condition,
                    alignment=alignment,
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    include_screen_history=False,
                    show_progress=show_progress,
                    progress_desc=f"S3 Multi-Subject [k={k_int} (eff={k_eff})]",
                )
            bundle = reuse_bundle
            conds_full: dict[str, dict] = {"S3_clean": dict(bundle)}
            for c_key in ("S0", "K0", "S3_shifted_neg75s", "F", "K1", "K0_shifted_neg75s"):
                if c_key in single_subject_full_vocab_bundles:
                    conds_full[c_key] = dict(single_subject_full_vocab_bundles[c_key])
            if "S0" not in conds_full and "S0_clean" in single_subject_full_vocab_bundles:
                conds_full["S0"] = dict(single_subject_full_vocab_bundles["S0_clean"])
        else:
            full_jobs: dict[str, dict] = {}
            has_s3_clean = all(
                k in available_clean for k in ("K0", "K1", "K2")
            )
            if reuse_bundle is None and has_s3_clean:
                full_jobs["S3_clean"] = {
                    "condition": condition,
                    "alignment": alignment,
                    "include_screen_history": False,
                }
            for st_key in ("S0", "K0"):
                if st_key in available_clean:
                    full_jobs[st_key] = {"condition": st_key, "alignment": "clean"}
            if "K2" in available_shifted:
                full_jobs["S3_shifted_neg75s"] = {
                    "condition": "S3",
                    "alignment": "shifted",
                    "include_screen_history": False,
                }
            full_results = (
                decode_dataset_segment_jobs(
                    agg_payload,
                    scorer,
                    hyperparameters,
                    full_jobs,
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=(
                        f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | full vocab | "
                        f"{len(full_jobs)} conditions batched]"
                    ),
                )
                if full_jobs
                else {}
            )
            if reuse_bundle is not None:
                bundle = reuse_bundle
                conds_full = {"S3_clean": dict(bundle)}
            elif "S3_clean" in full_results:
                bundle = dict(full_results.pop("S3_clean"))
                conds_full = {"S3_clean": dict(bundle)}
            elif full_results:
                first_cond_b = next(iter(full_results.values()))
                bundle = dict(first_cond_b)
                conds_full = {}
            else:
                bundle = {"segment_count": 0, "final_sentences": []}
                conds_full = {}
            conds_full.update(full_results)

        by_vocab: dict[str, dict] = {}
        for eff_v in eff_vocab_sizes:
            v_key = f"V={eff_v}"
            if eff_v == full_v_size:
                by_vocab[v_key] = {
                    "vocabulary_size": int(eff_v),
                    "is_full_vocabulary": True,
                    "conditions": dict(conds_full),
                }
                continue
            if (
                k_eff == 1
                and isinstance(single_subject_vocab_scaling_bundles, dict)
                and v_key in single_subject_vocab_scaling_bundles
            ):
                by_vocab[v_key] = dict(
                    single_subject_vocab_scaling_bundles[v_key]
                )
                continue

            active_indices = freq_indices[:eff_v]
            v_jobs: dict[str, dict] = {}
            if all(k in available_clean for k in ("K0", "K1", "K2")):
                v_jobs["S3_clean"] = {
                    "condition": "S3",
                    "alignment": "clean",
                    "include_screen_history": False,
                    "allowed_vocab_indices": active_indices,
                }
            for st_key in ("S0", "K0"):
                if st_key in available_clean:
                    v_jobs[st_key] = {
                        "condition": st_key,
                        "alignment": "clean",
                        "allowed_vocab_indices": active_indices,
                    }
            if "K2" in available_shifted:
                v_jobs["S3_shifted_neg75s"] = {
                    "condition": "S3",
                    "alignment": "shifted",
                    "include_screen_history": False,
                    "allowed_vocab_indices": active_indices,
                }
            v_conds: dict[str, dict] = (
                decode_dataset_segment_jobs(
                    agg_payload,
                    scorer,
                    hyperparameters,
                    v_jobs,
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=(
                        f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | V={eff_v} | "
                        f"{len(v_jobs)} conditions batched]"
                    ),
                )
                if v_jobs
                else {}
            )
            by_vocab[v_key] = {
                "vocabulary_size": int(eff_v),
                "is_full_vocabulary": False,
                "conditions": v_conds,
            }

        bundle["k_observations"] = k_int
        bundle["effective_k_observations"] = k_eff
        bundle["conditions"] = conds_full
        if by_vocab:
            bundle["by_vocabulary_size"] = by_vocab
        bundles_by_k[f"k={k_int}"] = bundle
        seen_eff_k[k_eff] = bundle

    return bundles_by_k


DEFAULT_VOCABULARY_SCALING_SIZES: tuple[int, ...] = (
    50,
    100,
    250,
    500,
    1000,
    2020,
)


def run_vocabulary_scaling_sweep(
    exported_payload: dict,
    scorer: CachedVocabularyLanguageScorer,
    hyperparameters: dict,
    *,
    vocab_sizes: Sequence[int] = DEFAULT_VOCABULARY_SCALING_SIZES,
    core_vocab_size: int = 250,
    full_vocab_bundles: dict[str, dict] | None = None,
    existing_scaling_bundles: dict[str, dict] | None = None,
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_batch_size: int = 32,
    max_segments: int | None = None,
    show_progress: bool = False,
    on_scale_completed: Callable[[str, dict], None] | None = None,
) -> dict[str, dict]:
    """按训练集词频降序截取前 N 个候选词（N in {50, 100, 250, 500, 1000, |V|}），零训练成本扫描多尺度词表统一束搜索解码表现。"""
    from braindecoding.decoding.neural_score_exporter import (
        resolve_vocabulary_frequency_order,
    )
    from braindecoding.evaluation.sequence_decoding_evaluation import (
        _backfill_intermediate_and_shifted_bundles,
    )

    vocabulary = [str(w) for w in exported_payload.get("vocabulary") or []]
    full_v_size = len(vocabulary)
    if full_v_size == 0:
        return {}

    freq_indices, freq_words, _ = resolve_vocabulary_frequency_order(
        exported_payload
    )
    all_segments = list(exported_payload.get("segments") or [])
    if max_segments is not None:
        all_segments = all_segments[: int(max_segments)]
    expected_seg_count = len(all_segments)
    total_ref_tokens = max(
        1, sum(len(seg.get("reference_words") or []) for seg in all_segments)
    )

    normalized_sizes: list[int] = []
    for raw_n in vocab_sizes:
        n_int = int(raw_n)
        if n_int <= 0:
            continue
        eff_n = full_v_size if n_int >= full_v_size else n_int
        if eff_n not in normalized_sizes:
            normalized_sizes.append(eff_n)
    if (
        core_vocab_size is not None
        and 0 < int(core_vocab_size) <= full_v_size
        and int(core_vocab_size) not in normalized_sizes
    ):
        normalized_sizes.append(int(core_vocab_size))
    if full_v_size not in normalized_sizes:
        normalized_sizes.append(full_v_size)
    normalized_sizes.sort()

    # 优先执行 core_vocab_size (默认 250)，确保高覆盖核心子集主表最快落盘并预热转移缓存
    execution_order = list(normalized_sizes)
    if (
        core_vocab_size is not None
        and int(core_vocab_size) in execution_order
    ):
        c_val = int(core_vocab_size)
        execution_order = [c_val] + [x for x in execution_order if x != c_val]

    scaling_bundles: dict[str, dict] = {}
    if isinstance(existing_scaling_bundles, dict):
        for k_str, entry in existing_scaling_bundles.items():
            if isinstance(entry, dict) and isinstance(
                entry.get("conditions"), dict
            ):
                s3_b = entry["conditions"].get("S3_clean")
                if (
                    isinstance(s3_b, dict)
                    and int(s3_b.get("segment_count", -1)) == expected_seg_count
                ):
                    scaling_bundles[str(k_str)] = entry

    available_clean = exported_payload.get("clean_scores") or {}
    available_shifted = exported_payload.get("shifted_scores") or {}
    ckpt_prov = dict(exported_payload.get("checkpoint_provenance") or {})
    has_real_context = bool(
        ckpt_prov.get("has_context_checkpoint")
        or (
            ckpt_prov.get("context")
            not in (None, "", "not_available", "fallback_from_revision_mature")
        )
    )

    for eff_n in execution_order:
        scale_key = f"V={eff_n}"
        active_indices = freq_indices[:eff_n]
        active_words = freq_words[:eff_n]
        active_set = set(active_words)

        in_vocab_tokens = sum(
            1
            for seg in all_segments
            for w in (seg.get("reference_words") or [])
            if str(w) in active_set
        )
        token_coverage_pct = float(in_vocab_tokens / total_ref_tokens * 100.0)
        segs_100pct = sum(
            1
            for seg in all_segments
            if (seg.get("reference_words") or [])
            and all(str(w) in active_set for w in seg["reference_words"])
        )
        segs_80pct = sum(
            1
            for seg in all_segments
            if (seg.get("reference_words") or [])
            and (
                sum(1 for w in seg["reference_words"] if str(w) in active_set)
                / len(seg["reference_words"])
                >= 0.80 - 1e-9
            )
        )

        max_core_v = max(int(core_vocab_size or 250), 250)
        is_core_benchmark_scale = eff_n <= max_core_v

        existing_entry = scaling_bundles.get(scale_key)
        existing_conds = (
            dict(existing_entry.get("conditions") or {})
            if isinstance(existing_entry, dict)
            else {}
        )
        has_s3_clean_scores = all(
            k in available_clean for k in ("K0", "K1", "K2")
        )
        required_keys: list[str] = []
        if has_s3_clean_scores:
            required_keys.append("S3_clean")
        if "S0" in available_clean:
            required_keys.append("S0")
        if "K0" in available_clean:
            required_keys.append("K0")
        if "K2" in available_shifted:
            required_keys.append("S3_shifted_neg75s")
        if is_core_benchmark_scale or eff_n == full_v_size:
            if "K1" in available_clean:
                required_keys.append("K1")
            if "F" in available_clean:
                required_keys.append("F")
            if "K0" in available_shifted:
                required_keys.append("K0_shifted_neg75s")
            required_keys.append("LM-Only")
        if existing_conds and all(rk in existing_conds for rk in required_keys):
            continue

        if (
            eff_n == full_v_size
            and isinstance(full_vocab_bundles, dict)
            and (
                not has_s3_clean_scores
                or "S3_clean" in full_vocab_bundles
            )
            and bool(full_vocab_bundles)
        ):
            raw_reused: dict[str, dict] = dict(existing_conds)
            for k in (
                "S0",
                "S0_clean",
                "K0",
                "K1",
                "S3_clean",
                "F",
                "F_clean",
                "LM-Only",
                "K0_shifted_neg75s",
                "S3_shifted_neg75s",
            ):
                if k in full_vocab_bundles:
                    raw_reused[k] = full_vocab_bundles[k]
            if "S0_clean" in raw_reused and "S0" not in raw_reused:
                raw_reused["S0"] = raw_reused["S0_clean"]
            if "F_clean" in raw_reused and "F" not in raw_reused:
                raw_reused["F"] = raw_reused["F_clean"]
            if "F" not in raw_reused and "F" in available_clean:
                raw_reused["F"] = decode_dataset_segments(
                    exported_payload,
                    scorer,
                    hyperparameters,
                    condition="F",
                    alignment="clean",
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=f"S3 VocabScale [V={eff_n} | Full-Context]",
                )
            if "LM-Only" not in raw_reused:
                raw_reused["LM-Only"] = decode_dataset_segments(
                    exported_payload,
                    scorer,
                    hyperparameters,
                    condition="LM-Only",
                    alignment="clean",
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=f"S3 VocabScale [V={eff_n} | LM-Only]",
                )
            reused_conds = _backfill_intermediate_and_shifted_bundles(
                raw_reused
            )
            scale_entry = {
                "vocabulary_size": int(eff_n),
                "is_full_vocabulary": True,
                "test_token_coverage_pct": round(token_coverage_pct, 4),
                "core_100pct_segment_count": int(segs_100pct),
                "core_80pct_segment_count": int(segs_80pct),
                "total_segment_count": int(expected_seg_count),
                "conditions": reused_conds,
            }
            scaling_bundles[scale_key] = scale_entry
            if on_scale_completed is not None:
                on_scale_completed(scale_key, scale_entry)
            continue

        cond_map: dict[str, dict] = dict(existing_conds)
        static_states = (
            ("S0", "K0", "K1") if is_core_benchmark_scale else ("S0", "K0")
        )
        scale_jobs: dict[str, dict] = {}
        if has_s3_clean_scores and "S3_clean" not in cond_map:
            scale_jobs["S3_clean"] = {
                "condition": "S3",
                "alignment": "clean",
                "include_screen_history": False,
            }
        for static_state in static_states:
            if static_state in available_clean and static_state not in cond_map:
                scale_jobs[static_state] = {
                    "condition": static_state,
                    "alignment": "clean",
                }
        if is_core_benchmark_scale and "F" in available_clean and "F" not in cond_map:
            scale_jobs["F"] = {"condition": "F", "alignment": "clean"}
        if is_core_benchmark_scale and "LM-Only" not in cond_map:
            scale_jobs["LM-Only"] = {"condition": "LM-Only", "alignment": "clean"}
        if (
            is_core_benchmark_scale
            and "K0" in available_shifted
            and "K0_shifted_neg75s" not in cond_map
        ):
            scale_jobs["K0_shifted_neg75s"] = {
                "condition": "K0",
                "alignment": "shifted",
            }
        if "K2" in available_shifted and "S3_shifted_neg75s" not in cond_map:
            scale_jobs["S3_shifted_neg75s"] = {
                "condition": "S3",
                "alignment": "shifted",
                "include_screen_history": False,
            }
        if scale_jobs:
            for job in scale_jobs.values():
                job["allowed_vocab_indices"] = active_indices
            cond_map.update(
                decode_dataset_segment_jobs(
                    exported_payload,
                    scorer,
                    hyperparameters,
                    scale_jobs,
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=(
                        f"S3 VocabScale [V={eff_n} | {len(scale_jobs)} conditions batched]"
                    ),
                )
            )

        cond_map = _backfill_intermediate_and_shifted_bundles(cond_map)
        scale_entry = {
            "vocabulary_size": int(eff_n),
            "is_full_vocabulary": bool(eff_n == full_v_size),
            "test_token_coverage_pct": round(token_coverage_pct, 4),
            "core_100pct_segment_count": int(segs_100pct),
            "core_80pct_segment_count": int(segs_80pct),
            "total_segment_count": int(expected_seg_count),
            "conditions": cond_map,
        }
        scaling_bundles[scale_key] = scale_entry
        if on_scale_completed is not None:
            on_scale_completed(scale_key, scale_entry)

    # 按词表规模从小到大排序返回
    ordered_result: dict[str, dict] = {}
    for n_val in normalized_sizes:
        k_str = f"V={n_val}"
        if k_str in scaling_bundles:
            ordered_result[k_str] = scaling_bundles[k_str]
    return ordered_result
