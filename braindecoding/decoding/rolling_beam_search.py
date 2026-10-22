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
    temperature: float = 0.1,
) -> np.ndarray:
    """将全词表余弦相似度矩阵 [L, |V|] 按温度 T_r 缩放并计算 log_softmax 对数概率。"""
    temp = float(temperature)
    if temp <= 0.0:
        raise ValueError(f"温度参数 T_r 必须为正数，收到：{temperature!r}")
    if isinstance(cosine_scores, np.ndarray):
        tensor = torch.from_numpy(cosine_scores).float()
    else:
        tensor = cosine_scores.detach().cpu().float()
    log_probs = F_nn.log_softmax(tensor / temp, dim=-1)
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
        best_k0_id, best_k0_score, _, _ = _score_single_step_candidates(
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

    return {
        "final_words": final_words,
        "committed_k2_words": committed_k2_words,
        "initial_k0_words": initial_k0_words,
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
        "score": float(winner.score),
        "neural_score_sum": float(winner.neural_score_sum),
        "lm_score_sum": float(winner.lm_score_sum),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def decode_dataset_segments(
    exported_payload: dict,
    scorer: CachedVocabularyLanguageScorer,
    hyperparameters: dict,
    *,
    condition: str = "S3",
    alignment: str = "clean",
    beam_width: int = 5,
    candidate_top_k: int = 20,
    partition_filter: str | None = None,
    max_segments: int | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> dict:
    """对导出的 .pt 神经得分载荷批量执行指定条件（S3, S0, K0, K1, K2, F, LM-Only）的句段解码。"""
    vocabulary = list(exported_payload["vocabulary"])
    segments = [
        seg
        for seg in exported_payload["segments"]
        if partition_filter is None or seg.get("partition") == partition_filter
    ]
    if max_segments is not None:
        segments = segments[: int(max_segments)]

    score_bank = (
        exported_payload["clean_scores"]
        if alignment == "clean"
        else exported_payload["shifted_scores"]
    )
    states_cfg = hyperparameters.get("states", hyperparameters)

    def _get_state_params(state_name: str) -> tuple[float, float]:
        entry = states_cfg.get(state_name, {})
        temp = float(entry.get("temperature", 0.1))
        lam = float(entry.get("language_weight", 0.15))
        return temp, lam

    cond_upper = str(condition).upper()
    started = time.perf_counter()
    final_sentences: list[dict] = []
    screen_histories: list[dict] = []

    if cond_upper == "LM-ONLY":
        # LM-Only 与受试者及脑信号无关，相同句长 L 仅需解码一次即可复用
        length_cache: dict[int, dict] = {}
        _, lm_weight = _get_state_params("K0")
        for idx, seg in enumerate(segments):
            seg_len = len(seg["event_indices"])
            if seg_len not in length_cache:
                length_cache[seg_len] = decode_segment_static(
                    None,
                    vocabulary,
                    scorer,
                    language_weight=1.0 if lm_weight <= 0 else lm_weight,
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_length=seg_len,
                    pure_language_model=True,
                )
            dec = length_cache[seg_len]
            final_sentences.append(
                {
                    "segment_index": seg["segment_index"],
                    "group_id": seg["group_id"],
                    "subject_id": seg["subject_id"],
                    "recording_id": seg["recording_id"],
                    "scope": seg["scope"],
                    "block_id": seg.get("block_id", seg["scope"]),
                    "reference_words": list(seg["reference_words"]),
                    "predicted_words": list(dec["final_words"]),
                    "in_vocabulary_mask": list(seg["in_vocabulary_mask"]),
                    "elapsed_seconds": dec["elapsed_seconds"],
                }
            )
            if progress_callback is not None:
                progress_callback(idx + 1, len(segments), time.perf_counter() - started)
    elif cond_upper == "S3":
        t_k0, lam_k0 = _get_state_params("K0")
        t_k1, lam_k1 = _get_state_params("K1")
        t_k2, lam_k2 = _get_state_params("K2")

        full_lp_k0 = compute_neural_log_probabilities(score_bank["K0"], t_k0)
        full_lp_k1 = compute_neural_log_probabilities(score_bank["K1"], t_k1)
        full_lp_k2 = compute_neural_log_probabilities(score_bank["K2"], t_k2)
        lam_map = {"K0": lam_k0, "K1": lam_k1, "K2": lam_k2}

        for idx, seg in enumerate(segments):
            ev_idx = seg["event_indices"]
            seg_lp = {
                "K0": full_lp_k0[ev_idx],
                "K1": full_lp_k1[ev_idx],
                "K2": full_lp_k2[ev_idx],
            }
            dec = decode_segment_streaming(
                seg_lp,
                vocabulary,
                scorer,
                language_weights=lam_map,
                beam_width=beam_width,
                candidate_top_k=candidate_top_k,
                reference_words=seg["reference_words"],
            )
            final_sentences.append(
                {
                    "segment_index": seg["segment_index"],
                    "group_id": seg["group_id"],
                    "subject_id": seg["subject_id"],
                    "recording_id": seg["recording_id"],
                    "scope": seg["scope"],
                    "block_id": seg.get("block_id", seg["scope"]),
                    "reference_words": list(seg["reference_words"]),
                    "predicted_words": list(dec["final_words"]),
                    "initial_k0_words": list(dec["initial_k0_words"]),
                    "committed_k2_words": list(dec["committed_k2_words"]),
                    "in_vocabulary_mask": list(seg["in_vocabulary_mask"]),
                    "elapsed_seconds": dec["elapsed_seconds"],
                }
            )
            screen_histories.append(
                {
                    "segment_index": seg["segment_index"],
                    "group_id": seg["group_id"],
                    "subject_id": seg["subject_id"],
                    "reference_words": list(seg["reference_words"]),
                    "in_vocabulary_mask": list(seg["in_vocabulary_mask"]),
                    "screen_history": dec["screen_history"],
                    "position_trajectories": dec["position_trajectories"],
                }
            )
            if progress_callback is not None:
                progress_callback(idx + 1, len(segments), time.perf_counter() - started)
    elif cond_upper in {"S0", "K0", "K1", "K2", "F"}:
        t_r, lam_r = _get_state_params(cond_upper)
        full_lp = compute_neural_log_probabilities(score_bank[cond_upper], t_r)
        for idx, seg in enumerate(segments):
            ev_idx = seg["event_indices"]
            dec = decode_segment_static(
                full_lp[ev_idx],
                vocabulary,
                scorer,
                language_weight=lam_r,
                beam_width=beam_width,
                candidate_top_k=candidate_top_k,
            )
            final_sentences.append(
                {
                    "segment_index": seg["segment_index"],
                    "group_id": seg["group_id"],
                    "subject_id": seg["subject_id"],
                    "recording_id": seg["recording_id"],
                    "scope": seg["scope"],
                    "block_id": seg.get("block_id", seg["scope"]),
                    "reference_words": list(seg["reference_words"]),
                    "predicted_words": list(dec["final_words"]),
                    "in_vocabulary_mask": list(seg["in_vocabulary_mask"]),
                    "elapsed_seconds": dec["elapsed_seconds"],
                }
            )
            if progress_callback is not None:
                progress_callback(idx + 1, len(segments), time.perf_counter() - started)
    else:
        raise ValueError(
            f"不支持的解码条件 {condition!r}，可选：'S3', 'S0', 'K0', 'K1', 'K2', 'F', 'LM-Only'"
        )

    total_elapsed = time.perf_counter() - started
    mean_sec = total_elapsed / max(1, len(segments))
    return {
        "condition": condition,
        "alignment": alignment,
        "segment_count": len(segments),
        "total_elapsed_seconds": round(total_elapsed, 4),
        "mean_seconds_per_segment": round(mean_sec, 4),
        "final_sentences": final_sentences,
        "screen_histories": screen_histories,
    }
