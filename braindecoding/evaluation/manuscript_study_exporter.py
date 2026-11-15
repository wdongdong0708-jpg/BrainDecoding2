"""面向论文正反论证、定性案例与补充诊断的多研究数据导出器（无废弃支线）。

提供以下纯净导出能力（零 C-Gated / tau_gate / F_local 历史包袱）：
1. 基于真实词起始时刻 τ_i 的物理等待时间分布（Local r = 0/1/2 vs Full-Context）；
2. 训练词长先验分布与神经预测分布的 Jensen-Shannon 散度对齐检验（SimpleB2T Eq. 10, Δ_JS bits）与 Top-10 词长误差降幅（ms）；
3. 按训练词频四分位（Q1..Q4）拆解词级检索准确率；
4. 定性案例导出：词级 Top-10 语义/声学邻域候选表、代表句逐词神经 Top-5 矩阵、分位数典型句跨策略解码对照表；
5. 被试间“词级检索 Top-10 vs 句子解码 BLEU-1 / Word Acc”耦合关联、Brain-Only vs Brain+LM 消融对比、合成脉冲与长词窗对照整理。
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from braindecoding.config import PROJECT_ROOT
from braindecoding.data.derived_cache import restore_legacy_columns
from braindecoding.decoding.neural_score_exporter import (
    enrich_segments_from_event_table,
)
from braindecoding.evaluation.sequence_decoding_evaluation import (
    evaluate_six_dimensional_metrics,
    paper_strategy_display_name,
)
from braindecoding.evaluation.statistical_significance import (
    holm_adjust,
    sentence_metrics,
)
from braindecoding.evaluation.word_retrieval_evaluation import (
    _student_t_two_sided_pvalue,
)
from braindecoding.training.training_runtime import save_json


def _write_csv_rows(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> Path | None:
    if not rows:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _to_numpy_2d(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        arr = matrix.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D score matrix, got shape {arr.shape}.")
    return arr


def _softmax_rows(logits: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    inv_t = 1.0 / max(float(temperature), 1e-6)
    scaled = logits.astype(np.float64, copy=False) * inv_t
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_x = np.exp(scaled)
    denom = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x / np.maximum(denom, 1e-12)


def _js_divergence_bits_chunked(
    p: np.ndarray,
    q: np.ndarray,
    perm: np.ndarray | None = None,
    *,
    chunk_size: int = 512,
) -> float:
    """Compute mean row-wise base-2 Jensen-Shannon divergence (bits) with bounded RAM."""
    n_rows = int(p.shape[0])
    if n_rows == 0:
        return 0.0
    total_js = 0.0
    use_f32 = (p.shape[0] * p.shape[1]) > 1_000_000
    dtype = np.float32 if use_f32 else np.float64
    eps = dtype(1e-12)
    for start in range(0, n_rows, chunk_size):
        end = min(n_rows, start + chunk_size)
        p_blk = p[start:end].astype(dtype, copy=False)
        q_idx = perm[start:end] if perm is not None else slice(start, end)
        q_blk = q[q_idx].astype(dtype, copy=False)
        m_blk = dtype(0.5) * (p_blk + q_blk)
        m_safe = np.maximum(m_blk, eps)
        p_safe = np.maximum(p_blk, eps)
        q_safe = np.maximum(q_blk, eps)
        kl_pm = np.sum(
            np.where(p_blk > 0, p_blk * np.log2(p_safe / m_safe), dtype(0.0)),
            axis=1,
            dtype=np.float64,
        )
        kl_qm = np.sum(
            np.where(q_blk > 0, q_blk * np.log2(q_safe / m_safe), dtype(0.0)),
            axis=1,
            dtype=np.float64,
        )
        js_blk = np.clip(0.5 * (kl_pm + kl_qm), 0.0, 1.0)
        total_js += float(np.sum(js_blk))
    return total_js / float(n_rows)


def _pick_first_column(
    columns: Sequence[str], candidates: Sequence[str]
) -> str | None:
    col_set = set(columns)
    for cand in candidates:
        if cand in col_set:
            return cand
    return None


def compute_word_horizon_physical_waits(
    event_table: pd.DataFrame,
    *,
    split: str = "test",
    horizons: Sequence[int | str] = (0, 1, 2, "full"),
    window_seconds: float = 1.0,
    common_horizon_delay: int = 2,
) -> list[dict[str, Any]]:
    """计算给定数据划分下词级前瞻 r 与整句策略的物理输入等待分布（秒）。"""
    if event_table is None or len(event_table) == 0:
        return []

    df = event_table.copy()
    split_col = _pick_first_column(df.columns, ("data_split", "数据划分", "split"))
    if split_col is not None and split:
        df = df[df[split_col].astype(str) == str(split)]

    train_col = _pick_first_column(
        df.columns, ("is_trainable", "是否可训练", "trainable")
    )
    if train_col is not None:
        mask = df[train_col].astype(bool)
        if bool(mask.any()):
            df = df[mask]

    if len(df) == 0:
        return []

    onset_col = _pick_first_column(
        df.columns, ("onset_seconds", "开始时间", "onset", "start")
    )
    if onset_col is None:
        raise ValueError("event_table lacks onset column (onset_seconds / 开始时间).")

    ctx_col = _pick_first_column(
        df.columns,
        (
            "runtime_context_uid",
            "sentence_uid",
            "context_id",
            "上下文编号",
            "recording_id",
            "记录编号",
        ),
    )
    ord_col = _pick_first_column(
        df.columns,
        ("in_recording_index", "记录内序号", "original_seq_index", "序列原编号"),
    )
    sort_cols = [c for c in (ctx_col, onset_col, ord_col) if c is not None]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")

    grouped_onsets: list[np.ndarray] = []
    if ctx_col is None:
        grouped_onsets.append(pd.to_numeric(df[onset_col], errors="coerce").to_numpy(dtype=np.float64))
    else:
        for _, grp in df.groupby(ctx_col, sort=False):
            arr = pd.to_numeric(grp[onset_col], errors="coerce").to_numpy(
                dtype=np.float64
            )
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                grouped_onsets.append(arr)

    delay_guard = max(0, int(common_horizon_delay))
    win_s = float(window_seconds)
    out_rows: list[dict[str, Any]] = []

    for h in horizons:
        h_str = str(h).strip().lower()
        is_full = h_str == "full"
        if is_full:
            r_val: int | None = None
            h_key = "full"
            disp = "Full-Context"
        else:
            r_val = max(0, int(h))
            h_key = f"r={r_val}"
            disp = f"Local (r = {r_val})"

        waits: list[float] = []
        for tau in grouped_onsets:
            n_tok = int(len(tau))
            if n_tok <= delay_guard:
                continue
            limit = n_tok - delay_guard
            last_idx = n_tok - 1
            for pos in range(limit):
                end_pos = last_idx if r_val is None else min(last_idx, pos + r_val)
                waits.append(max(win_s, float((tau[end_pos] + win_s) - tau[pos])))

        if not waits:
            continue
        arr_w = np.asarray(waits, dtype=np.float64)
        out_rows.append(
            {
                "horizon": h_key,
                "paper_display_name": disp,
                "common_horizon_delay": delay_guard,
                "query_count": int(len(arr_w)),
                "mean_wait_seconds": round(float(np.mean(arr_w)), 6),
                "std_wait_seconds": round(
                    float(np.std(arr_w, ddof=1 if len(arr_w) > 1 else 0)), 6
                ),
                "p05_wait_seconds": round(float(np.percentile(arr_w, 5.0)), 6),
                "p25_wait_seconds": round(float(np.percentile(arr_w, 25.0)), 6),
                "median_wait_seconds": round(float(np.median(arr_w)), 6),
                "p75_wait_seconds": round(float(np.percentile(arr_w, 75.0)), 6),
                "p95_wait_seconds": round(float(np.percentile(arr_w, 95.0)), 6),
            }
        )
    return out_rows


def compute_duration_prior_alignment_metrics(
    score_matrix: np.ndarray | torch.Tensor,
    vocabulary: Sequence[str],
    target_words: Sequence[str],
    target_durations_seconds: Sequence[float],
    *,
    train_words: Sequence[str] | None = None,
    train_durations_seconds: Sequence[float] | None = None,
    temperature: float = 0.07,
    n_permutations: int = 200,
    seed: int = 20261115,
) -> dict[str, Any]:
    """计算神经预测分布与词长高斯-词频先验分布的 Δ_JS (bits) 及 Top-10 词长误差降幅（SimpleB2T Eq. 10 & Fig. 2b）。"""
    scores = _to_numpy_2d(score_matrix)
    n_queries, n_vocab = scores.shape
    vocab_list = [str(w) for w in vocabulary]
    if len(vocab_list) != n_vocab:
        raise ValueError(
            f"Vocabulary length ({len(vocab_list)}) != score columns ({n_vocab})."
        )
    if len(target_words) != n_queries or len(target_durations_seconds) != n_queries:
        raise ValueError("target_words and target_durations_seconds must match score rows.")
    if n_queries == 0 or n_vocab == 0:
        raise ValueError("score_matrix and vocabulary must be non-empty.")

    fit_words_raw = target_words if train_words is None else train_words
    fit_durs_raw = (
        target_durations_seconds
        if train_durations_seconds is None
        else train_durations_seconds
    )
    if len(fit_words_raw) != len(fit_durs_raw) or len(fit_words_raw) == 0:
        raise ValueError("train_words and train_durations_seconds must be aligned and non-empty.")

    fit_durs_list = [max(float(d), 1e-3) for d in fit_durs_raw if math.isfinite(float(d)) and float(d) > 0.0]
    if not fit_durs_list:
        raise ValueError("No valid positive durations available to fit duration prior.")
    global_log = np.log(np.asarray(fit_durs_list, dtype=np.float64))
    global_mu = float(np.mean(global_log))
    global_sigma = max(float(np.std(global_log, ddof=1 if len(global_log) > 1 else 0)), 0.05)
    global_mean_ms = float(np.mean(fit_durs_list) * 1000.0)

    by_word_durs: dict[str, list[float]] = {w: [] for w in vocab_list}
    total_fit = 0
    for w_raw, d_raw in zip(fit_words_raw, fit_durs_raw):
        d_f = float(d_raw)
        if not math.isfinite(d_f) or d_f <= 0.0:
            continue
        w_str = str(w_raw)
        if w_str in by_word_durs:
            by_word_durs[w_str].append(max(d_f, 1e-3))
        total_fit += 1

    mu_vec = np.empty(n_vocab, dtype=np.float64)
    sigma_vec = np.empty(n_vocab, dtype=np.float64)
    prior_vec = np.empty(n_vocab, dtype=np.float64)
    mean_dur_ms = np.empty(n_vocab, dtype=np.float64)
    denom_prior = float(total_fit + n_vocab)

    for idx, word in enumerate(vocab_list):
        d_list = by_word_durs.get(word, [])
        cnt = len(d_list)
        prior_vec[idx] = float(cnt + 1.0) / denom_prior
        if cnt == 0:
            mu_vec[idx] = global_mu
            sigma_vec[idx] = global_sigma
            mean_dur_ms[idx] = global_mean_ms
        elif cnt == 1:
            mu_vec[idx] = float(math.log(d_list[0]))
            sigma_vec[idx] = global_sigma
            mean_dur_ms[idx] = float(d_list[0] * 1000.0)
        else:
            arr_d = np.asarray(d_list, dtype=np.float64)
            log_d = np.log(arr_d)
            mu_vec[idx] = float(np.mean(log_d))
            sigma_vec[idx] = max(float(np.std(log_d, ddof=1)), 0.05)
            mean_dur_ms[idx] = float(np.mean(arr_d) * 1000.0)

    eval_durs = np.asarray(
        [max(float(d), 1e-3) for d in target_durations_seconds], dtype=np.float64
    )
    eval_log_d = np.log(eval_durs)[:, None]
    log_gauss = (
        -0.5 * ((eval_log_d - mu_vec[None, :]) / sigma_vec[None, :]) ** 2
        - np.log(sigma_vec[None, :])
    )
    log_q = np.log(prior_vec[None, :] + 1e-12) + log_gauss
    q_dist = _softmax_rows(log_q, temperature=1.0)
    p_dist = _softmax_rows(scores, temperature=float(temperature))

    mean_js_bits = _js_divergence_bits_chunked(p_dist, q_dist, perm=None)

    n_perms = max(1, int(n_permutations))
    rng = np.random.default_rng(int(seed))
    perms = [rng.permutation(n_queries) for _ in range(n_perms)]
    perm_js_vals = [
        _js_divergence_bits_chunked(p_dist, q_dist, perm=perm)
        for perm in perms
    ]
    permuted_js_bits = float(np.mean(perm_js_vals))
    delta_js_bits = float(permuted_js_bits - mean_js_bits)
    perm_p_value = float(
        (1 + sum(1 for pv in perm_js_vals if pv <= mean_js_bits))
        / float(n_perms + 1)
    )

    target_dur_ms = eval_durs * 1000.0
    top1_idx = np.argmax(scores, axis=1)
    top1_mae_ms = float(np.mean(np.abs(mean_dur_ms[top1_idx] - target_dur_ms)))

    k10 = min(10, n_vocab)
    top10_part = np.argpartition(scores, -k10, axis=1)[:, -k10:]
    top10_pred_ms = np.mean(mean_dur_ms[top10_part], axis=1)
    top10_mae_ms = float(np.mean(np.abs(top10_pred_ms - target_dur_ms)))
    permuted_top10_mae_ms = float(
        np.mean([np.mean(np.abs(top10_pred_ms - target_dur_ms[p])) for p in perms])
    )
    top10_reduction_ms = float(permuted_top10_mae_ms - top10_mae_ms)

    vocab_to_idx = {w: idx for idx, w in enumerate(vocab_list)}
    k10_rep = min(10, max(1, n_vocab - 1))
    scores_masked = scores.astype(np.float32, copy=True)
    for row_i, tw in enumerate(target_words):
        t_idx = vocab_to_idx.get(str(tw))
        if t_idx is not None:
            scores_masked[row_i, t_idx] = -np.inf
    rep_part = np.argpartition(scores_masked, -k10_rep, axis=1)[:, -k10_rep:]
    rep_pred_ms = np.mean(mean_dur_ms[rep_part], axis=1)
    rep_mae_ms = float(np.mean(np.abs(rep_pred_ms - target_dur_ms)))
    perm_rep_mae_ms = float(
        np.mean([np.mean(np.abs(rep_pred_ms - target_dur_ms[p])) for p in perms])
    )
    rep_reduction_ms = float(perm_rep_mae_ms - rep_mae_ms)

    return {
        "query_count": int(n_queries),
        "vocabulary_size": int(n_vocab),
        "temperature": float(temperature),
        "n_permutations": int(n_perms),
        "mean_js_bits": round(mean_js_bits, 6),
        "permuted_js_bits": round(permuted_js_bits, 6),
        "delta_js_bits": round(delta_js_bits, 6),
        "permutation_p_value": round(perm_p_value, 6),
        "top1_duration_mae_ms": round(top1_mae_ms, 4),
        "top10_duration_mae_ms": round(top10_mae_ms, 4),
        "permuted_top10_duration_mae_ms": round(permuted_top10_mae_ms, 4),
        "top10_duration_error_reduction_ms": round(top10_reduction_ms, 4),
        "replacement_top10_duration_mae_ms": round(rep_mae_ms, 4),
        "permuted_replacement_top10_duration_mae_ms": round(perm_rep_mae_ms, 4),
        "replacement_top10_duration_error_reduction_ms": round(
            rep_reduction_ms, 4
        ),
    }


def compute_word_frequency_quartile_metrics(
    score_matrix: np.ndarray | torch.Tensor,
    vocabulary: Sequence[str],
    target_words: Sequence[str],
    train_word_counts: Mapping[str, int],
    *,
    subject_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """按训练词频四分位（Q1_rare .. Q4_frequent）拆解词级检索 Top-1 / Top-5 / Top-10 准确率。"""
    scores = _to_numpy_2d(score_matrix)
    n_queries, n_vocab = scores.shape
    vocab_list = [str(w) for w in vocabulary]
    if len(vocab_list) != n_vocab:
        raise ValueError("vocabulary length must match score_matrix columns.")
    if len(target_words) != n_queries:
        raise ValueError("target_words length must match score_matrix rows.")

    vocab_to_idx = {w: idx for idx, w in enumerate(vocab_list)}
    sorted_vocab = sorted(
        vocab_list,
        key=lambda w: (int(train_word_counts.get(w, 0)), w),
    )
    quartile_labels = ("Q1_rare", "Q2_low_mid", "Q3_high_mid", "Q4_frequent")
    chunks = np.array_split(np.arange(len(sorted_vocab), dtype=np.int64), 4)

    valid_rows: list[int] = []
    target_indices: list[int] = []
    valid_words: list[str] = []
    valid_subjs: list[str] = []
    for i, tw in enumerate(target_words):
        t_idx = vocab_to_idx.get(str(tw))
        if t_idx is None:
            continue
        valid_rows.append(i)
        target_indices.append(t_idx)
        valid_words.append(str(tw))
        valid_subjs.append(
            str(subject_ids[i]) if subject_ids is not None else "default"
        )

    if not valid_rows:
        return []

    sub_scores = scores[np.asarray(valid_rows, dtype=np.int64)]
    t_idx_arr = np.asarray(target_indices, dtype=np.int64)
    target_vals = sub_scores[np.arange(len(valid_rows)), t_idx_arr]
    ranks = 1 + np.sum(sub_scores > target_vals[:, None], axis=1)
    valid_word_arr = np.asarray(valid_words, dtype=object)
    valid_subj_arr = np.asarray(valid_subjs, dtype=object)

    rows_out: list[dict[str, Any]] = []
    for q_label, idx_chunk in zip(quartile_labels, chunks):
        q_words = [sorted_vocab[int(i)] for i in idx_chunk]
        q_set = set(q_words)
        counts = [int(train_word_counts.get(w, 0)) for w in q_words]
        mask = np.asarray([w in q_set for w in valid_word_arr], dtype=bool)
        n_q = int(np.sum(mask))
        if n_q == 0:
            rows_out.append(
                {
                    "quartile": q_label,
                    "min_train_count": min(counts) if counts else 0,
                    "max_train_count": max(counts) if counts else 0,
                    "unique_words": int(len(q_set)),
                    "query_count": 0,
                    "top1_acc_pct": 0.0,
                    "top5_acc_pct": 0.0,
                    "top10_acc_pct": 0.0,
                    "subject_macro_top10_pct": 0.0,
                }
            )
            continue
        q_ranks = ranks[mask]
        q_subjs = valid_subj_arr[mask]
        unique_q_words = len(set(valid_word_arr[mask].tolist()))
        top1 = float(np.mean(q_ranks <= 1) * 100.0)
        top5 = float(np.mean(q_ranks <= 5) * 100.0)
        top10 = float(np.mean(q_ranks <= 10) * 100.0)
        subj_top10_vals = [
            float(np.mean(q_ranks[q_subjs == s] <= 10) * 100.0)
            for s in sorted(set(q_subjs.tolist()))
        ]
        subj_macro_top10 = (
            float(np.mean(subj_top10_vals)) if subj_top10_vals else top10
        )
        rows_out.append(
            {
                "quartile": q_label,
                "min_train_count": int(min(counts)) if counts else 0,
                "max_train_count": int(max(counts)) if counts else 0,
                "unique_words": int(unique_q_words),
                "query_count": n_q,
                "top1_acc_pct": round(top1, 4),
                "top5_acc_pct": round(top5, 4),
                "top10_acc_pct": round(top10, 4),
                "subject_macro_top10_pct": round(subj_macro_top10, 4),
            }
        )
    return rows_out


def _extract_payload_score_matrix(
    payload: Mapping[str, Any], state: str, *, alignment: str = "clean"
) -> np.ndarray | None:
    store = payload.get("clean_scores" if alignment == "clean" else "shifted_scores") or {}
    if state not in store:
        return None
    raw = store[state]
    if raw is None:
        return None
    return _to_numpy_2d(raw)


def _extract_segment_row_indices(
    payload: Mapping[str, Any], seg: Mapping[str, Any]
) -> list[int]:
    if isinstance(seg.get("event_indices"), (list, tuple)):
        return [int(i) for i in seg["event_indices"]]
    seg_eids = seg.get("event_ids") or []
    all_eids = payload.get("event_ids") or []
    if seg_eids and all_eids:
        eid_to_idx = {str(eid): idx for idx, eid in enumerate(all_eids)}
        matched = [eid_to_idx[str(eid)] for eid in seg_eids if str(eid) in eid_to_idx]
        if len(matched) == len(seg_eids):
            return matched
    segments = payload.get("segments") or []
    offset = 0
    for s in segments:
        s_len = len(s.get("reference_words") or ())
        if s is seg or (
            s.get("segment_index") == seg.get("segment_index")
            and str(s.get("group_id", "")) == str(seg.get("group_id", ""))
        ):
            return list(range(offset, offset + s_len))
        offset += s_len
    return []


def export_word_top10_neighborhood_examples(
    payload: Mapping[str, Any],
    *,
    conditions: Sequence[str] = ("S0", "K0", "K2", "F"),
    target_words: Sequence[str] | None = None,
    top_k: int = 10,
    max_examples: int = 12,
) -> list[dict[str, Any]]:
    """导出代表性目标词在不同建模视野（Word-Only / Local r=0,2 / Full-Context）下的 Top-K 神经近邻候选。"""
    vocab = [str(w) for w in (payload.get("vocabulary") or [])]
    if not vocab:
        return []
    vocab_to_idx = {w: idx for idx, w in enumerate(vocab)}
    ref_words = [str(w) for w in (payload.get("reference_words") or [])]
    if not ref_words:
        return []

    avail_map = payload.get("availability_masks") or {}
    clean_matrices: dict[str, np.ndarray] = {}
    for cond in conditions:
        mat = _extract_payload_score_matrix(payload, cond, alignment="clean")
        if mat is not None and mat.shape[0] == len(ref_words) and mat.shape[1] == len(vocab):
            clean_matrices[str(cond)] = mat
    if not clean_matrices:
        return []

    # Build normalized vocabulary embeddings if present for semantic cosine reference
    norm_emb: np.ndarray | None = None
    raw_emb = payload.get("vocabulary_embeddings")
    if raw_emb is not None and len(raw_emb) == len(vocab):
        emb_np = _to_numpy_2d(raw_emb).astype(np.float64)
        norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
        norm_emb = emb_np / np.maximum(norms, 1e-8)

    # Identify valid indices intersecting all loaded conditions
    common_mask = np.asarray([w in vocab_to_idx for w in ref_words], dtype=bool)
    for cond in clean_matrices:
        if cond in avail_map and avail_map[cond] is not None:
            m_arr = np.asarray(
                avail_map[cond].cpu().numpy()
                if isinstance(avail_map[cond], torch.Tensor)
                else avail_map[cond],
                dtype=bool,
            )
            if len(m_arr) == len(common_mask):
                common_mask &= m_arr
    valid_indices = np.flatnonzero(common_mask)
    if len(valid_indices) == 0:
        return []

    # Collect per-word row indices and optional content-word hints from segments
    content_words_seen: set[str] = set()
    for seg in payload.get("segments") or []:
        words_s = seg.get("reference_words") or []
        icw_s = seg.get("is_content_word_mask") or []
        for w, flag in zip(words_s, icw_s):
            if bool(flag):
                content_words_seen.add(str(w))

    word_rows: dict[str, list[int]] = {}
    for idx in valid_indices:
        word_rows.setdefault(ref_words[int(idx)], []).append(int(idx))

    if target_words is not None:
        chosen_words = [str(w) for w in target_words if str(w) in word_rows]
    else:
        base_cond = "S0" if "S0" in clean_matrices else next(iter(clean_matrices))
        focus_cond = "K2" if "K2" in clean_matrices else base_cond
        base_mat = clean_matrices[base_cond]
        focus_mat = clean_matrices[focus_cond]
        scored_candidates: list[tuple[int, float, str]] = []
        for w, idxs in word_rows.items():
            t_idx = vocab_to_idx[w]
            idx_arr = np.asarray(idxs, dtype=np.int64)
            mean_base = np.mean(base_mat[idx_arr], axis=0)
            mean_focus = np.mean(focus_mat[idx_arr], axis=0)
            rank_base = 1 + int(np.sum(mean_base > mean_base[t_idx]))
            rank_focus = 1 + int(np.sum(mean_focus > mean_focus[t_idx]))
            is_cw = 1 if (not content_words_seen or w in content_words_seen) else 0
            gain = float(rank_base - rank_focus)
            scored_candidates.append(
                (is_cw, gain - 0.05 * rank_focus, w)
            )
        scored_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        chosen_words = [item[2] for item in scored_candidates[: max(1, int(max_examples))]]

    k_use = max(1, min(int(top_k), len(vocab)))
    examples_out: list[dict[str, Any]] = []
    for word in chosen_words:
        t_idx = vocab_to_idx[word]
        idx_arr = np.asarray(word_rows[word], dtype=np.int64)
        cond_entries: dict[str, Any] = {}
        for cond, mat in clean_matrices.items():
            mean_scores = np.mean(mat[idx_arr], axis=0).astype(np.float64)
            target_rank = 1 + int(np.sum(mean_scores > mean_scores[t_idx]))
            top_order = sorted(
                range(len(vocab)),
                key=lambda j: (-float(mean_scores[j]), vocab[j]),
            )[:k_use]
            cands = []
            for r_i, v_i in enumerate(top_order, start=1):
                cand_obj: dict[str, Any] = {
                    "rank": r_i,
                    "word": vocab[v_i],
                    "score": round(float(mean_scores[v_i]), 6),
                    "is_target": bool(v_i == t_idx),
                }
                if norm_emb is not None:
                    cos_v = float(np.dot(norm_emb[v_i], norm_emb[t_idx]))
                    cand_obj["cosine_to_target"] = round(cos_v, 6)
                cands.append(cand_obj)
            cond_entries[cond] = {
                "condition": cond,
                "paper_display_name": paper_strategy_display_name(cond),
                "target_rank": int(target_rank),
                "target_score": round(float(mean_scores[t_idx]), 6),
                "top_candidates": cands,
            }
        examples_out.append(
            {
                "target_word": word,
                "occurrence_count": int(len(idx_arr)),
                "is_content_word": bool(word in content_words_seen)
                if content_words_seen
                else None,
                "conditions": cond_entries,
            }
        )
    return examples_out


def export_sentence_neural_top5_matrix(
    payload: Mapping[str, Any],
    *,
    segment_index: int | None = None,
    state: str = "K2",
    top_k: int = 5,
    temperature: float = 0.07,
) -> dict[str, Any]:
    """导出单个代表性完整覆盖句子在指定神经状态下的逐词 Top-K 神经候选概率矩阵。"""
    vocab = [str(w) for w in (payload.get("vocabulary") or [])]
    segments = list(payload.get("segments") or [])
    score_mat = _extract_payload_score_matrix(payload, state, alignment="clean")
    if not vocab or not segments or score_mat is None:
        return {}
    vocab_to_idx = {w: idx for idx, w in enumerate(vocab)}

    chosen_seg: Mapping[str, Any] | None = None
    if segment_index is not None:
        for seg in segments:
            if int(seg.get("segment_index", -1)) == int(segment_index):
                chosen_seg = seg
                break
        if chosen_seg is None and 0 <= int(segment_index) < len(segments):
            chosen_seg = segments[int(segment_index)]
    else:
        # Select best 100% in-vocabulary segment with 5 <= L <= 12
        best_key = (-1, -1.0, 0)
        for seg in segments:
            ref_w = [str(w) for w in (seg.get("reference_words") or ())]
            seg_len = len(ref_w)
            in_v = seg.get("in_vocabulary_mask") or [w in vocab_to_idx for w in ref_w]
            all_iv = seg_len > 0 and all(bool(x) for x in in_v)
            row_idxs = _extract_segment_row_indices(payload, seg)
            if len(row_idxs) != seg_len:
                continue
            mat_seg = score_mat[np.asarray(row_idxs, dtype=np.int64)]
            top5_hits = 0
            for pos, tw in enumerate(ref_w):
                t_i = vocab_to_idx.get(tw)
                if t_i is not None and int(np.sum(mat_seg[pos] > mat_seg[pos, t_i])) < 5:
                    top5_hits += 1
            in_len_band = 1 if (5 <= seg_len <= 12) else 0
            cand_key = (
                1 if all_iv else 0,
                in_len_band,
                float(top5_hits / max(1, seg_len)),
                -int(seg.get("segment_index", 0)),
            )
            if chosen_seg is None or cand_key > best_key:
                best_key = cand_key
                chosen_seg = seg

    if chosen_seg is None:
        return {}

    ref_words = [str(w) for w in (chosen_seg.get("reference_words") or ())]
    row_idxs = _extract_segment_row_indices(payload, chosen_seg)
    if len(row_idxs) != len(ref_words) or not ref_words:
        return {}

    seg_scores = score_mat[np.asarray(row_idxs, dtype=np.int64)]
    seg_probs = _softmax_rows(seg_scores, temperature=temperature)
    k_use = max(1, min(int(top_k), len(vocab)))

    positions: list[dict[str, Any]] = []
    for pos, ref_w in enumerate(ref_words):
        t_idx = vocab_to_idx.get(ref_w)
        row_s = seg_scores[pos]
        row_p = seg_probs[pos]
        true_rank = (
            1 + int(np.sum(row_s > row_s[t_idx]))
            if t_idx is not None
            else None
        )
        top_order = sorted(
            range(len(vocab)),
            key=lambda j: (-float(row_s[j]), vocab[j]),
        )[:k_use]
        positions.append(
            {
                "position": int(pos),
                "reference_word": ref_w,
                "true_word_rank": true_rank,
                "hit_top1": bool(true_rank == 1) if true_rank is not None else False,
                "hit_top5": bool(true_rank is not None and true_rank <= 5),
                "top_candidates": [
                    {
                        "rank": r_i,
                        "word": vocab[v_i],
                        "score": round(float(row_s[v_i]), 6),
                        "prob": round(float(row_p[v_i]), 6),
                        "is_target": bool(t_idx is not None and v_i == t_idx),
                    }
                    for r_i, v_i in enumerate(top_order, start=1)
                ],
            }
        )

    return {
        "state": str(state),
        "paper_display_name": paper_strategy_display_name(state),
        "segment_index": int(chosen_seg.get("segment_index", 0)),
        "group_id": str(chosen_seg.get("group_id", "")),
        "subject_id": str(chosen_seg.get("subject_id", "")),
        "reference_words": ref_words,
        "positions": positions,
    }


def _index_bundle_records(
    bundle: Mapping[str, Any],
) -> tuple[dict[tuple[int, str], dict[str, Any]], dict[int, dict[str, Any]]]:
    by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    by_seg_idx: dict[int, dict[str, Any]] = {}
    for pos_i, rec in enumerate(bundle.get("final_sentences") or []):
        if not isinstance(rec, Mapping):
            continue
        seg_idx = int(rec.get("segment_index", pos_i))
        grp_id = str(rec.get("group_id", ""))
        rec_dict = dict(rec)
        by_pair[(seg_idx, grp_id)] = rec_dict
        by_seg_idx.setdefault(seg_idx, rec_dict)
    return by_pair, by_seg_idx


def export_decoded_sentence_quantile_examples(
    decoded_bundles: Mapping[str, Mapping[str, Any]],
    *,
    primary_condition: str = "S3_clean",
    compare_conditions: Sequence[str] = (
        "S0",
        "K0",
        "S3_clean",
        "F",
        "K2_brain_only",
        "LM-Only",
    ),
    quantiles: Sequence[float] = (0.90, 0.75, 0.50, 0.25),
    language: str = "fr",
) -> list[dict[str, Any]]:
    """按主策略句子 BLEU-1 分位数抽取代表性解码句子，并横向对齐所有比较策略的预测。"""
    primary_bundle = decoded_bundles.get(primary_condition)
    if not isinstance(primary_bundle, Mapping):
        fallback = next(
            (k for k in ("S3_clean", "S3", "K2", "K1", "K0", "S0") if k in decoded_bundles),
            None,
        )
        if fallback is None:
            return []
        primary_condition = fallback
        primary_bundle = decoded_bundles[fallback]

    primary_recs = list(primary_bundle.get("final_sentences") or [])
    if not primary_recs:
        return []

    chinese = language == "zh"
    scored_rows: list[tuple[float, float, int, dict[str, Any]]] = []
    for pos_i, rec in enumerate(primary_recs):
        ref_w = [str(w) for w in (rec.get("reference_words") or ())]
        pred_w = [str(w) for w in (rec.get("predicted_words") or ())]
        if not ref_w or len(ref_w) != len(pred_w):
            continue
        sm = sentence_metrics(ref_w, pred_w, chinese=chinese)
        wacc = float(sum(int(a == b) for a, b in zip(ref_w, pred_w)) / len(ref_w))
        seg_idx = int(rec.get("segment_index", pos_i))
        scored_rows.append((float(sm["bleu1"]), wacc, seg_idx, dict(rec)))

    if not scored_rows:
        return []

    scored_rows.sort(key=lambda item: (item[0], item[1], item[2]))
    n_avail = len(scored_rows)
    used_indices: set[int] = set()
    selected: list[tuple[float, int]] = []
    for q in quantiles:
        q_clamped = min(1.0, max(0.0, float(q)))
        target_pos = int(round(q_clamped * (n_avail - 1)))
        if target_pos in used_indices and len(used_indices) < n_avail:
            # Find nearest unused rank preserving ordering
            candidates = sorted(
                (i for i in range(n_avail) if i not in used_indices),
                key=lambda i: (abs(i - target_pos), i),
            )
            target_pos = candidates[0]
        used_indices.add(target_pos)
        selected.append((q_clamped, target_pos))

    cond_lookups: dict[str, tuple[dict[tuple[int, str], dict[str, Any]], dict[int, dict[str, Any]]]] = {}
    for cond in compare_conditions:
        if cond in decoded_bundles and isinstance(decoded_bundles[cond], Mapping):
            cond_lookups[str(cond)] = _index_bundle_records(decoded_bundles[cond])

    examples_out: list[dict[str, Any]] = []
    for q_val, row_pos in selected:
        _, _, seg_idx, p_rec = scored_rows[row_pos]
        grp_id = str(p_rec.get("group_id", ""))
        ref_w = [str(w) for w in (p_rec.get("reference_words") or ())]
        join_sep = "" if chinese else " "
        preds_map: dict[str, Any] = {}
        for cond, (by_pair, by_idx) in cond_lookups.items():
            matched = by_pair.get((seg_idx, grp_id)) or by_idx.get(seg_idx)
            if not matched:
                continue
            pw = [str(w) for w in (matched.get("predicted_words") or ())]
            if len(pw) != len(ref_w):
                continue
            sm_c = sentence_metrics(ref_w, pw, chinese=chinese)
            wacc_c = float(sum(int(a == b) for a, b in zip(ref_w, pw)) / len(ref_w))
            preds_map[cond] = {
                "condition": cond,
                "paper_display_name": paper_strategy_display_name(cond),
                "predicted_words": pw,
                "predicted_text": join_sep.join(pw),
                "sentence_bleu1_pct": round(float(sm_c["bleu1"]) * 100.0, 4),
                "wer_pct": round(float(sm_c["wer"]) * 100.0, 4),
                "word_acc_pct": round(wacc_c * 100.0, 4),
            }
        examples_out.append(
            {
                "quantile": round(q_val, 4),
                "segment_index": int(seg_idx),
                "group_id": grp_id,
                "subject_id": str(p_rec.get("subject_id", "")),
                "material_id": str(p_rec.get("material_id", "")),
                "reference_words": ref_w,
                "reference_text": join_sep.join(ref_w),
                "predictions": preds_map,
            }
        )
    return examples_out


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_r = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_r
        i = j
    return ranks


def _pearson_and_pvalue(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n = int(len(x))
    if n < 2:
        return 0.0, 1.0
    dx = x - np.mean(x)
    dy = y - np.mean(y)
    denom = float(np.sqrt(np.sum(dx * dx) * np.sum(dy * dy)))
    if denom <= 1e-12:
        return 0.0, 1.0
    r_val = float(np.clip(np.sum(dx * dy) / denom, -1.0, 1.0))
    if n < 3:
        return r_val, 1.0
    if abs(r_val) >= 1.0 - 1e-12:
        return r_val, 0.0
    t_abs = abs(r_val) * math.sqrt((n - 2.0) / max(1e-12, 1.0 - r_val * r_val))
    p_val = float(_student_t_two_sided_pvalue(t_abs, float(n - 2)))
    return r_val, p_val


def _extract_word_top10_pct_by_subject(
    word_eval_by_subject: Mapping[str, Any],
) -> dict[str, float]:
    raw_source: Any = word_eval_by_subject
    for nested_key in (
        "per_subject_summary_aggregate",
        "per_subject",
        "by_subject",
        "subject_breakdown",
        "subjects",
    ):
        cand = word_eval_by_subject.get(nested_key)
        if isinstance(cand, Mapping) or (
            isinstance(cand, Sequence) and not isinstance(cand, (str, bytes))
        ):
            raw_source = cand
            break

    items_iter: list[tuple[str, Any]] = []
    if isinstance(raw_source, Mapping):
        items_iter = [(str(k), v) for k, v in raw_source.items()]
    elif isinstance(raw_source, Sequence) and not isinstance(raw_source, (str, bytes)):
        for entry in raw_source:
            if not isinstance(entry, Mapping):
                continue
            sid = entry.get("subject_id") or entry.get("subject") or entry.get("id")
            if sid:
                items_iter.append((str(sid), entry))

    metric_keys = (
        "word_top10_pct",
        "top10_acc_pct",
        "recall_at_10_pct",
        "mean_macro_top10",
        "vocab50_macro_top10",
        "subject_equal_macro_recall_at_10",
        "macro_top10",
        "macro_recall_at_10",
        "mean_full_vocab_macro_top10",
        "full_vocab_macro_top10",
        "full_vocab_top10",
        "micro_recall_at_10",
        "recall_at_10",
        "top10",
    )

    out: dict[str, float] = {}
    for subj, raw_val in items_iter:
        if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
            val = float(raw_val)
            out[str(subj)] = val * 100.0 if val <= 1.0 else val
            continue
        if not isinstance(raw_val, Mapping):
            continue
        chosen: float | None = None
        for key in metric_keys:
            if key in raw_val and raw_val[key] is not None:
                v = float(raw_val[key])
                if key.endswith("_pct") or v > 1.0:
                    chosen = v
                else:
                    chosen = v * 100.0
                break
        if chosen is not None:
            out[str(subj)] = chosen
    return out


def _extract_sentence_metrics_by_subject(
    decoded_bundles_or_report: Mapping[str, Any],
    *,
    condition: str,
    language: str,
) -> dict[str, dict[str, float]]:
    if not decoded_bundles_or_report:
        return {}
    if isinstance(decoded_bundles_or_report.get("final_sentences"), Sequence):
        recs_all = list(decoded_bundles_or_report.get("final_sentences") or [])
    elif (
        condition in decoded_bundles_or_report
        and isinstance(decoded_bundles_or_report[condition], Mapping)
        and "final_sentences" in decoded_bundles_or_report[condition]
    ):
        recs_all = list(decoded_bundles_or_report[condition].get("final_sentences") or [])
    else:
        per_subj = decoded_bundles_or_report.get("per_subject")
        if isinstance(per_subj, Mapping):
            # Report schema could be {cond: {subj: metrics}} or {subj: {cond: metrics}}
            cond_block = per_subj.get(condition)
            if isinstance(cond_block, Mapping):
                return {
                    str(subj): {
                        "bleu1_pct": float(m.get("bleu1_pct", 0.0)),
                        "word_acc_pct": float(m.get("word_acc_pct", 0.0)),
                        "segment_count": int(m.get("segment_count", 0)),
                    }
                    for subj, m in cond_block.items()
                    if isinstance(m, Mapping)
                }
            extracted: dict[str, dict[str, float]] = {}
            for subj, s_map in per_subj.items():
                if isinstance(s_map, Mapping) and isinstance(
                    s_map.get(condition), Mapping
                ):
                    m = s_map[condition]
                    extracted[str(subj)] = {
                        "bleu1_pct": float(m.get("bleu1_pct", 0.0)),
                        "word_acc_pct": float(m.get("word_acc_pct", 0.0)),
                        "segment_count": int(m.get("segment_count", 0)),
                    }
            if extracted:
                return extracted
        elif isinstance(per_subj, Sequence) and not isinstance(per_subj, (str, bytes)):
            extracted = {}
            for item in per_subj:
                if not isinstance(item, Mapping):
                    continue
                item_cond = item.get("condition")
                if item_cond is not None and str(item_cond) != str(condition):
                    continue
                subj_id = item.get("subject_id") or item.get("subject") or item.get("id")
                if not subj_id:
                    continue
                extracted[str(subj_id)] = {
                    "bleu1_pct": float(item.get("bleu1_pct", 0.0)),
                    "word_acc_pct": float(item.get("word_acc_pct", 0.0)),
                    "segment_count": int(item.get("segment_count", 0)),
                }
            if extracted:
                return extracted
        return {}

    by_subj: dict[str, list[dict[str, Any]]] = {}
    for rec in recs_all:
        if isinstance(rec, Mapping):
            by_subj.setdefault(str(rec.get("subject_id", "default")), []).append(
                dict(rec)
            )

    out: dict[str, dict[str, float]] = {}
    for subj, recs in by_subj.items():
        m6 = evaluate_six_dimensional_metrics(
            recs,
            language=language,
            condition=condition,
            compute_paired_stats=False,
        )
        out[subj] = {
            "bleu1_pct": float(m6["bleu1_pct"]),
            "word_acc_pct": float(m6["word_acc_pct"]),
            "segment_count": int(m6["segment_count"]),
        }
    return out


def export_subject_retrieval_sentence_coupling(
    word_eval_by_subject: Mapping[str, Any],
    decoded_bundles_or_report: Mapping[str, Any],
    *,
    condition: str = "S3_clean",
    language: str = "fr",
) -> dict[str, Any]:
    """计算受试者维度上“词级 Top-10 (%) vs 句子解码 BLEU-1 (%) / Word Acc (%)”的相关系数与 OLS 回归。"""
    word_map = _extract_word_top10_pct_by_subject(word_eval_by_subject)
    sent_map = _extract_sentence_metrics_by_subject(
        decoded_bundles_or_report, condition=condition, language=language
    )
    common_subjs = sorted(set(word_map.keys()) & set(sent_map.keys()))
    if not common_subjs:
        return {"condition": condition, "subject_count": 0, "subjects": []}

    rows: list[dict[str, Any]] = []
    for subj in common_subjs:
        rows.append(
            {
                "subject_id": subj,
                "word_top10_pct": round(float(word_map[subj]), 4),
                "sentence_bleu1_pct": round(float(sent_map[subj]["bleu1_pct"]), 4),
                "sentence_word_acc_pct": round(
                    float(sent_map[subj]["word_acc_pct"]), 4
                ),
                "segment_count": int(sent_map[subj].get("segment_count", 0)),
            }
        )

    x = np.asarray([r["word_top10_pct"] for r in rows], dtype=np.float64)
    y_bleu = np.asarray([r["sentence_bleu1_pct"] for r in rows], dtype=np.float64)
    y_wacc = np.asarray([r["sentence_word_acc_pct"] for r in rows], dtype=np.float64)

    r_bleu, p_r_bleu = _pearson_and_pvalue(x, y_bleu)
    rho_bleu, p_rho_bleu = _pearson_and_pvalue(
        _average_ranks(x), _average_ranks(y_bleu)
    )
    r_wacc, p_r_wacc = _pearson_and_pvalue(x, y_wacc)
    rho_wacc, p_rho_wacc = _pearson_and_pvalue(
        _average_ranks(x), _average_ranks(y_wacc)
    )

    var_x = float(np.var(x))
    slope_bleu = (
        float(np.cov(x, y_bleu, ddof=0)[0, 1] / var_x) if var_x > 1e-12 else 0.0
    )
    intercept_bleu = float(np.mean(y_bleu) - slope_bleu * np.mean(x))
    slope_wacc = (
        float(np.cov(x, y_wacc, ddof=0)[0, 1] / var_x) if var_x > 1e-12 else 0.0
    )
    intercept_wacc = float(np.mean(y_wacc) - slope_wacc * np.mean(x))

    return {
        "condition": str(condition),
        "paper_display_name": paper_strategy_display_name(condition),
        "subject_count": len(rows),
        "subjects": rows,
        "pearson_r_bleu1": round(r_bleu, 6),
        "pearson_p_bleu1": round(p_r_bleu, 6),
        "spearman_rho_bleu1": round(rho_bleu, 6),
        "spearman_p_bleu1": round(p_rho_bleu, 6),
        "ols_slope_bleu1": round(slope_bleu, 6),
        "ols_intercept_bleu1": round(intercept_bleu, 6),
        "pearson_r_word_acc": round(r_wacc, 6),
        "pearson_p_word_acc": round(p_r_wacc, 6),
        "spearman_rho_word_acc": round(rho_wacc, 6),
        "spearman_p_word_acc": round(p_rho_wacc, 6),
        "ols_slope_word_acc": round(slope_wacc, 6),
        "ols_intercept_word_acc": round(intercept_wacc, 6),
    }


def _long_window_paper_label(cond: str) -> str:
    mapping = {
        "word_1s": "Word-Only (1.00s)",
        "long_1p27": "Word-Only (1.27s)",
        "long_1p5": "Word-Only (1.50s)",
        "rolling_k0": "Local (r = 0)",
        "rolling_k1": "Local (r = 1)",
        "rolling_k2": "Local (r = 2)",
    }
    return mapping.get(str(cond), paper_strategy_display_name(cond))


def export_synthetic_and_horizon_studies(
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path,
    event_table: pd.DataFrame | None = None,
    dataset: str = "pallier2025",
    split: str = "test",
) -> dict[str, Any]:
    """整理并导出已存档合成脉冲对照（synthetic_timing_v2）、长窗消融（single_word_long_window_test_v2）与物理等待分布。"""
    root = Path(repo_root).resolve() if repo_root is not None else PROJECT_ROOT
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, str] = {}
    summary: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "output_dir": str(out_dir),
    }

    # 1. Synthetic timing v2 (fallback v1)
    synth_candidates = (
        root / "data" / "results" / "studies" / "synthetic_timing_v2" / "results.json",
        root / "data" / "results" / "studies" / "synthetic_timing_v1" / "results.json",
    )
    synth_path = next((p for p in synth_candidates if p.is_file()), None)
    if synth_path is not None:
        raw_synth = json.loads(synth_path.read_text(encoding="utf-8"))
        synth_rows: list[dict[str, Any]] = []
        for item in raw_synth.get("summaries") or []:
            if not isinstance(item, Mapping):
                continue
            ds_item = str(item.get("dataset", ""))
            if dataset and ds_item and ds_item != dataset:
                continue
            h_i = int(item.get("horizon", 0))
            cond_raw = str(item.get("condition", ""))
            cond_label = (
                "Real MEG"
                if cond_raw == "real"
                else f"Synthetic ({cond_raw})"
            )
            synth_rows.append(
                {
                    "dataset": ds_item,
                    "condition": cond_raw,
                    "condition_label": f"{cond_label} | Local (r = {h_i})",
                    "horizon": h_i,
                    "paper_display_name": f"Local (r = {h_i})",
                    "n_seeds": int(item.get("n_seeds", 0)),
                    "mean_top10_pct": round(float(item.get("mean", 0.0)) * 100.0, 4),
                    "sd_top10_pct": round(float(item.get("sample_sd", 0.0)) * 100.0, 4),
                }
            )
        csv_p = _write_csv_rows(out_dir / "synthetic_timing_summary.csv", synth_rows)
        if csv_p is not None:
            exported_files["synthetic_timing_csv"] = str(csv_p)
        summary["synthetic_timing_rows"] = synth_rows

    # 2. Long window comparison
    lw_path = (
        root
        / "data"
        / "results"
        / "studies"
        / "single_word_long_window_test_v2"
        / "summary.csv"
    )
    if lw_path.is_file():
        lw_df = pd.read_csv(lw_path)
        lw_rows: list[dict[str, Any]] = []
        for _, row in lw_df.iterrows():
            ds_item = str(row.get("dataset", ""))
            if dataset and ds_item and ds_item != dataset:
                continue
            cond_raw = str(row.get("condition", ""))
            lw_rows.append(
                {
                    "dataset": ds_item,
                    "condition": cond_raw,
                    "paper_display_name": _long_window_paper_label(cond_raw),
                    "population": str(row.get("population", "")),
                    "candidates": str(row.get("candidates", "")),
                    "candidate_count": int(row.get("candidate_count", 0)),
                    "unique_events": int(row.get("unique_events", 0)),
                    "seeds": int(row.get("seeds", 0)),
                    "subject_equal_top10_pct": round(
                        float(row.get("subject_equal_recall_at_10_mean", 0.0)) * 100.0, 4
                    ),
                    "subject_equal_top10_sd_pct": round(
                        float(row.get("subject_equal_recall_at_10_sd", 0.0)) * 100.0, 4
                    ),
                    "subject_equal_top1_pct": round(
                        float(row.get("subject_equal_recall_at_1_mean", 0.0)) * 100.0, 4
                    ),
                    "subject_equal_top1_sd_pct": round(
                        float(row.get("subject_equal_recall_at_1_sd", 0.0)) * 100.0, 4
                    ),
                    "macro_top10_pct": round(
                        float(row.get("macro_recall_at_10_mean", 0.0)) * 100.0, 4
                    ),
                    "macro_top10_sd_pct": round(
                        float(row.get("macro_recall_at_10_sd", 0.0)) * 100.0, 4
                    ),
                }
            )
        csv_p = _write_csv_rows(out_dir / "long_window_comparison.csv", lw_rows)
        if csv_p is not None:
            exported_files["long_window_csv"] = str(csv_p)
        summary["long_window_rows"] = lw_rows

    # 3. Physical wait distribution from event table
    if event_table is not None and len(event_table) > 0:
        wait_rows = compute_word_horizon_physical_waits(event_table, split=split)
        csv_p = _write_csv_rows(
            out_dir / "physical_wait_distribution.csv", wait_rows
        )
        if csv_p is not None:
            exported_files["physical_wait_csv"] = str(csv_p)
        summary["physical_wait_rows"] = wait_rows

    summary["exported_files"] = exported_files
    manifest_p = out_dir / "synthetic_and_horizon_summary.json"
    save_json(manifest_p, summary)
    exported_files["synthetic_and_horizon_json"] = str(manifest_p)
    return summary


def export_sentence_and_subject_coupling_studies(
    *,
    output_dir: str | Path,
    decoded_bundles: Mapping[str, Mapping[str, Any]] | None = None,
    eval_summary: Mapping[str, Any] | None = None,
    word_eval_summary: Mapping[str, Any] | None = None,
    score_payload: Mapping[str, Any] | None = None,
    event_table: pd.DataFrame | None = None,
    language: str = "fr",
    split: str = "test",
    primary_condition: str = "S3_clean",
    n_permutations: int = 200,
) -> dict[str, Any]:
    """统一导出词级诊断（Δ_JS、词频四分位）、定性案例（Top-10 邻域 / 句内矩阵 / 分位数例句）、Brain-Only 消融与跨被试耦合结果。"""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exported_files: dict[str, str] = {}
    summary: dict[str, Any] = {
        "split": split,
        "language": language,
        "primary_condition": primary_condition,
        "output_dir": str(out_dir),
    }

    # 1. Enrich & inspect score_payload (if provided)
    if isinstance(score_payload, Mapping):
        vocab = [str(w) for w in (score_payload.get("vocabulary") or [])]
        ref_words = [str(w) for w in (score_payload.get("reference_words") or [])]
        subj_ids = (
            [str(s) for s in (score_payload.get("subject_ids") or [])]
            if score_payload.get("subject_ids") is not None
            else None
        )
        train_counts = dict(score_payload.get("vocabulary_train_counts") or {})

        # 1a. Qualitative word Top-10 neighborhoods
        nb_examples = export_word_top10_neighborhood_examples(score_payload)
        if nb_examples:
            nb_json_p = out_dir / "word_top10_neighborhoods.json"
            save_json(nb_json_p, {"examples": nb_examples})
            exported_files["word_top10_neighborhoods_json"] = str(nb_json_p)
            flat_nb_rows: list[dict[str, Any]] = []
            for ex in nb_examples:
                for cond_k, cond_v in (ex.get("conditions") or {}).items():
                    for cand in cond_v.get("top_candidates") or []:
                        flat_nb_rows.append(
                            {
                                "target_word": ex["target_word"],
                                "occurrence_count": ex["occurrence_count"],
                                "condition": cond_k,
                                "paper_display_name": cond_v["paper_display_name"],
                                "target_rank": cond_v["target_rank"],
                                "candidate_rank": cand["rank"],
                                "candidate_word": cand["word"],
                                "score": cand["score"],
                                "cosine_to_target": cand.get(
                                    "cosine_to_target", ""
                                ),
                                "is_target": cand["is_target"],
                            }
                        )
            nb_csv = _write_csv_rows(
                out_dir / "word_top10_neighborhoods.csv", flat_nb_rows
            )
            if nb_csv is not None:
                exported_files["word_top10_neighborhoods_csv"] = str(nb_csv)
            summary["word_top10_neighborhood_count"] = len(nb_examples)

        # 1b. Sentence neural Top-5 matrix
        sent_matrix = export_sentence_neural_top5_matrix(score_payload)
        if sent_matrix:
            sm_json_p = out_dir / "sentence_neural_top5_matrix.json"
            save_json(sm_json_p, sent_matrix)
            exported_files["sentence_neural_top5_json"] = str(sm_json_p)
            sm_flat = [
                {
                    "segment_index": sent_matrix["segment_index"],
                    "state": sent_matrix["state"],
                    "position": p["position"],
                    "reference_word": p["reference_word"],
                    "true_word_rank": p["true_word_rank"],
                    "hit_top1": p["hit_top1"],
                    "hit_top5": p["hit_top5"],
                    "top5_words": " | ".join(
                        f"{c['word']}({c['prob']:.3f})"
                        for c in p["top_candidates"]
                    ),
                }
                for p in sent_matrix.get("positions") or []
            ]
            sm_csv = _write_csv_rows(
                out_dir / "sentence_neural_top5_matrix.csv", sm_flat
            )
            if sm_csv is not None:
                exported_files["sentence_neural_top5_csv"] = str(sm_csv)
            summary["sentence_neural_top5_segment"] = sent_matrix["segment_index"]

        # 1c. Word frequency quartiles
        clean_store = score_payload.get("clean_scores") or {}
        if train_counts and vocab and ref_words and clean_store:
            freq_rows: list[dict[str, Any]] = []
            for cond_k in ("S0", "K0", "K1", "K2", "F"):
                mat = _extract_payload_score_matrix(
                    score_payload, cond_k, alignment="clean"
                )
                if mat is None or mat.shape[0] != len(ref_words):
                    continue
                q_rows = compute_word_frequency_quartile_metrics(
                    mat, vocab, ref_words, train_counts, subject_ids=subj_ids
                )
                for qr in q_rows:
                    freq_rows.append(
                        {
                            "condition": cond_k,
                            "paper_display_name": paper_strategy_display_name(
                                cond_k
                            ),
                            **qr,
                        }
                    )
            fq_csv = _write_csv_rows(
                out_dir / "word_frequency_quartiles.csv", freq_rows
            )
            if fq_csv is not None:
                exported_files["word_frequency_quartiles_csv"] = str(fq_csv)
            summary["word_frequency_quartile_rows"] = freq_rows

        # 1d. Duration-prior alignment (Δ_JS bits)
        durs_by_event_idx: np.ndarray | None = None
        if len(ref_words) > 0:
            raw_durs = score_payload.get("target_durations_seconds")
            if raw_durs is not None and len(raw_durs) == len(ref_words):
                durs_by_event_idx = np.asarray(raw_durs, dtype=np.float64)
            elif event_table is not None and len(event_table) > 0:
                ev_id_col = _pick_first_column(
                    event_table.columns, ("event_id", "事件编号")
                )
                s_col = _pick_first_column(
                    event_table.columns, ("onset_seconds", "开始时间")
                )
                e_col = _pick_first_column(
                    event_table.columns, ("offset_seconds", "结束时间")
                )
                payload_eids = [str(e) for e in (score_payload.get("event_ids") or [])]
                if (
                    ev_id_col is not None
                    and s_col is not None
                    and e_col is not None
                    and len(payload_eids) == len(ref_words)
                ):
                    dur_lookup: dict[str, float] = {}
                    ev_vals = event_table[ev_id_col].astype(str).tolist()
                    st_vals = pd.to_numeric(
                        event_table[s_col], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    en_vals = pd.to_numeric(
                        event_table[e_col], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    for eid_v, st_v, en_v in zip(ev_vals, st_vals, en_vals):
                        if math.isfinite(st_v) and math.isfinite(en_v) and en_v > st_v:
                            dur_lookup[eid_v] = float(en_v - st_v)
                    matched_d = [dur_lookup.get(eid_v, float("nan")) for eid_v in payload_eids]
                    if all(math.isfinite(x) and x > 0.0 for x in matched_d):
                        durs_by_event_idx = np.asarray(matched_d, dtype=np.float64)

        if durs_by_event_idx is None and score_payload.get("segments"):
            # Recover per-event durations from segment onset/offset pairs when available
            dur_buf = np.full(len(ref_words), float("nan"), dtype=np.float64)
            for seg in score_payload.get("segments") or []:
                row_idxs = _extract_segment_row_indices(score_payload, seg)
                ons = seg.get("onset_seconds")
                offs = seg.get("offset_seconds")
                if (
                    isinstance(ons, (list, tuple))
                    and isinstance(offs, (list, tuple))
                    and len(ons) == len(row_idxs)
                    and len(offs) == len(row_idxs)
                ):
                    for r_i, s_t, e_t in zip(row_idxs, ons, offs):
                        d_v = float(e_t) - float(s_t)
                        if 0 <= r_i < len(dur_buf) and math.isfinite(d_v) and d_v > 0.0:
                            dur_buf[r_i] = d_v
            if np.all(np.isfinite(dur_buf) & (dur_buf > 0.0)):
                durs_by_event_idx = dur_buf

        if durs_by_event_idx is not None and len(durs_by_event_idx) == len(ref_words):
            train_w_fit: list[str] | None = None
            train_d_fit: list[float] | None = None
            if event_table is not None and len(event_table) > 0:
                sp_col = _pick_first_column(
                    event_table.columns, ("data_split", "数据划分", "split")
                )
                w_col = _pick_first_column(
                    event_table.columns, ("normalized_word", "标准词", "word", "词")
                )
                s_col = _pick_first_column(
                    event_table.columns, ("onset_seconds", "开始时间")
                )
                e_col = _pick_first_column(
                    event_table.columns, ("offset_seconds", "结束时间")
                )
                if sp_col and w_col and s_col and e_col:
                    tr_df = event_table[event_table[sp_col].astype(str) == "train"]
                    if len(tr_df) > 0:
                        tr_st = pd.to_numeric(tr_df[s_col], errors="coerce").to_numpy(dtype=np.float64)
                        tr_en = pd.to_numeric(tr_df[e_col], errors="coerce").to_numpy(dtype=np.float64)
                        tr_w = tr_df[w_col].astype(str).tolist()
                        tw_clean: list[str] = []
                        td_clean: list[float] = []
                        for w_i, st_i, en_i in zip(tr_w, tr_st, tr_en):
                            if math.isfinite(st_i) and math.isfinite(en_i) and en_i > st_i:
                                tw_clean.append(w_i)
                                td_clean.append(float(en_i - st_i))
                        if tw_clean:
                            train_w_fit = tw_clean
                            train_d_fit = td_clean

            avail_map = score_payload.get("availability_masks") or {}
            js_rows: list[dict[str, Any]] = []
            for align in ("clean", "shifted"):
                for cond_k in ("S0", "K0", "K1", "K2", "F"):
                    mat = _extract_payload_score_matrix(
                        score_payload, cond_k, alignment=align
                    )
                    if mat is None or mat.shape[0] != len(ref_words):
                        continue
                    row_mask = np.ones(len(ref_words), dtype=bool)
                    if cond_k in avail_map and avail_map[cond_k] is not None:
                        am = np.asarray(
                            avail_map[cond_k].cpu().numpy()
                            if isinstance(avail_map[cond_k], torch.Tensor)
                            else avail_map[cond_k],
                            dtype=bool,
                        )
                        if len(am) == len(row_mask):
                            row_mask &= am
                    idx_sel = np.flatnonzero(row_mask)
                    if len(idx_sel) < 2:
                        continue
                    js_m = compute_duration_prior_alignment_metrics(
                        mat[idx_sel],
                        vocab,
                        [ref_words[int(i)] for i in idx_sel],
                        durs_by_event_idx[idx_sel],
                        train_words=train_w_fit,
                        train_durations_seconds=train_d_fit,
                        n_permutations=n_permutations,
                    )
                    disp_name = (
                        paper_strategy_display_name(cond_k)
                        if align == "clean"
                        else f"Shifted {paper_strategy_display_name(cond_k)}"
                    )
                    js_rows.append(
                        {
                            "alignment": align,
                            "condition": cond_k,
                            "paper_display_name": disp_name,
                            **js_m,
                        }
                    )
            if js_rows:
                raw_ps = [float(r["permutation_p_value"]) for r in js_rows]
                for r_item, p_h in zip(js_rows, holm_adjust(raw_ps)):
                    r_item["p_holm"] = round(float(p_h), 6)
                js_csv = _write_csv_rows(
                    out_dir / "duration_prior_alignment.csv", js_rows
                )
                if js_csv is not None:
                    exported_files["duration_prior_alignment_csv"] = str(js_csv)
                summary["duration_prior_alignment_rows"] = js_rows

    # 2. Decoded bundles: Brain-Only ablation + 4-way POS + quantile sentence examples
    if isinstance(decoded_bundles, Mapping) and decoded_bundles:
        ablation_rows: list[dict[str, Any]] = []
        pos4_rows: list[dict[str, Any]] = []
        ordered_keys = (
            "S0",
            "S0_clean",
            "K0",
            "K1",
            "S3_clean",
            "S3",
            "K2",
            "F",
            "F_clean",
            "S0_brain_only",
            "K0_brain_only",
            "K1_brain_only",
            "K2_brain_only",
            "F_brain_only",
            "LM-Only",
            "K0_shifted_neg75s",
            "S3_shifted_neg75s",
        )
        seen_keys: set[str] = set()
        for ck in (*ordered_keys, *decoded_bundles.keys()):
            if ck in seen_keys or ck not in decoded_bundles:
                continue
            seen_keys.add(str(ck))
            bundle = decoded_bundles[ck]
            if not isinstance(bundle, Mapping):
                continue
            recs = list(bundle.get("final_sentences") or [])
            if not recs:
                continue
            m6 = evaluate_six_dimensional_metrics(
                recs, language=language, condition=str(ck), compute_paired_stats=False
            )
            mode = (
                "Brain-Only"
                if str(ck).endswith("_brain_only")
                else ("LM-Only" if str(ck).startswith("LM-Only") else "Brain+LM")
            )
            ablation_rows.append(
                {
                    "condition": str(ck),
                    "paper_display_name": str(m6.get("paper_display_name", ck)),
                    "mode": mode,
                    "segment_count": int(m6["segment_count"]),
                    "total_words": int(m6["total_words"]),
                    "bleu1_pct": round(float(m6["bleu1_pct"]), 4),
                    "corpus_bleu1_pct": round(float(m6["corpus_bleu1_pct"]), 4),
                    "word_acc_pct": round(float(m6["word_acc_pct"]), 4),
                    "in_vocabulary_word_acc_pct": round(
                        float(m6["in_vocabulary_word_acc_pct"]), 4
                    ),
                    "word_acc_top5_pct": round(float(m6["word_acc_top5_pct"]), 4),
                    "wer_pct": round(float(m6["wer_pct"]), 4),
                    "bertscore_f1_pct": round(float(m6["bertscore_f1_pct"]), 4),
                    "mean_wait_windows": m6.get("mean_wait_windows"),
                    "mean_physical_wait_seconds": m6.get(
                        "mean_physical_wait_seconds"
                    ),
                }
            )
            if "noun_word_acc_pct" in m6:
                pos4_rows.append(
                    {
                        "condition": str(ck),
                        "paper_display_name": str(m6.get("paper_display_name", ck)),
                        "noun_words": int(m6.get("noun_words", 0)),
                        "noun_word_acc_pct": m6.get("noun_word_acc_pct", 0.0),
                        "verb_words": int(m6.get("verb_words", 0)),
                        "verb_word_acc_pct": m6.get("verb_word_acc_pct", 0.0),
                        "adj_adv_words": int(m6.get("adj_adv_words", 0)),
                        "adj_adv_word_acc_pct": m6.get(
                            "adj_adv_word_acc_pct", 0.0
                        ),
                        "function_pos_words": int(m6.get("function_pos_words", 0)),
                        "function_pos_word_acc_pct": m6.get(
                            "function_pos_word_acc_pct", 0.0
                        ),
                    }
                )
        abl_csv = _write_csv_rows(
            out_dir / "brain_only_vs_lm_ablation.csv", ablation_rows
        )
        if abl_csv is not None:
            exported_files["brain_only_vs_lm_ablation_csv"] = str(abl_csv)
        summary["brain_only_vs_lm_rows"] = ablation_rows
        if pos4_rows:
            p4_csv = _write_csv_rows(
                out_dir / "pos_four_way_accuracy.csv", pos4_rows
            )
            if p4_csv is not None:
                exported_files["pos_four_way_csv"] = str(p4_csv)
            summary["pos_four_way_rows"] = pos4_rows

        # Quantile decoded sentence examples
        q_examples = export_decoded_sentence_quantile_examples(
            decoded_bundles,
            primary_condition=primary_condition,
            language=language,
        )
        if q_examples:
            q_json_p = out_dir / "decoded_sentence_quantile_examples.json"
            save_json(q_json_p, {"examples": q_examples})
            exported_files["decoded_sentence_examples_json"] = str(q_json_p)
            flat_q_rows: list[dict[str, Any]] = []
            for ex in q_examples:
                for cond_k, pv in (ex.get("predictions") or {}).items():
                    flat_q_rows.append(
                        {
                            "quantile": ex["quantile"],
                            "segment_index": ex["segment_index"],
                            "subject_id": ex["subject_id"],
                            "reference_text": ex["reference_text"],
                            "condition": cond_k,
                            "paper_display_name": pv["paper_display_name"],
                            "predicted_text": pv["predicted_text"],
                            "sentence_bleu1_pct": pv["sentence_bleu1_pct"],
                            "wer_pct": pv["wer_pct"],
                            "word_acc_pct": pv["word_acc_pct"],
                        }
                    )
            q_csv = _write_csv_rows(
                out_dir / "decoded_sentence_quantile_examples.csv", flat_q_rows
            )
            if q_csv is not None:
                exported_files["decoded_sentence_examples_csv"] = str(q_csv)
            summary["decoded_sentence_example_count"] = len(q_examples)

    # 3. Subject retrieval vs sentence decoding coupling
    target_sent_source: Mapping[str, Any] | None = (
        decoded_bundles if decoded_bundles else eval_summary
    )
    if isinstance(word_eval_summary, Mapping) and isinstance(
        target_sent_source, Mapping
    ):
        coupling = export_subject_retrieval_sentence_coupling(
            word_eval_summary,
            target_sent_source,
            condition=primary_condition,
            language=language,
        )
        if coupling.get("subject_count", 0) > 0:
            cp_json_p = out_dir / "subject_retrieval_sentence_coupling.json"
            save_json(cp_json_p, coupling)
            exported_files["subject_coupling_json"] = str(cp_json_p)
            cp_csv = _write_csv_rows(
                out_dir / "subject_retrieval_sentence_coupling.csv",
                coupling["subjects"],
            )
            if cp_csv is not None:
                exported_files["subject_coupling_csv"] = str(cp_csv)
            summary["subject_coupling"] = coupling

    summary["exported_files"] = exported_files
    return summary


def run_manuscript_study_exports(
    *,
    dataset: str = "pallier2025",
    split: str = "test",
    output_dir: str | Path,
    repo_root: str | Path | None = None,
    export_pt: str | Path | None = None,
    decoding_json: str | Path | None = None,
    eval_summary_json: str | Path | None = None,
    word_eval_json: str | Path | None = None,
    event_table_csv: str | Path | None = None,
    primary_condition: str = "S3_clean",
    n_permutations: int = 200,
) -> dict[str, Any]:
    """统一入口：从可用解码包、神经得分缓存、事件表与已存档研究结果一键生成论文导出产物。"""
    root = Path(repo_root).resolve() if repo_root is not None else PROJECT_ROOT
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    language = "fr" if str(dataset) == "pallier2025" else "zh"

    _aux_event_columns = {
        "pos": "词性",
        "is_content_word": "是否实义词",
        "runtime_context_uid": "上下文编号",
        "sentence_uid": "句子编号",
    }
    event_table: pd.DataFrame | None = None
    if event_table_csv is not None and Path(event_table_csv).is_file():
        event_table = restore_legacy_columns(
            pd.read_csv(event_table_csv), _aux_event_columns
        )
    else:
        cand_ev = (
            root
            / "data"
            / "cache"
            / "derived"
            / dataset
            / "events"
            / "event_table.csv"
        )
        if cand_ev.is_file():
            event_table = restore_legacy_columns(
                pd.read_csv(cand_ev), _aux_event_columns
            )

    score_payload: dict[str, Any] | None = None
    if export_pt is not None and Path(export_pt).is_file():
        score_payload = torch.load(
            export_pt, map_location="cpu", weights_only=False
        )
        if (
            isinstance(score_payload, dict)
            and event_table is not None
            and isinstance(score_payload.get("segments"), list)
        ):
            enrich_segments_from_event_table(score_payload["segments"], event_table)

    decoded_bundles: dict[str, Any] | None = None
    if decoding_json is not None and Path(decoding_json).is_file():
        dec_obj = json.loads(Path(decoding_json).read_text(encoding="utf-8"))
        decoded_bundles = dict(dec_obj.get("decoded_bundles") or dec_obj)
        decoded_bundles.pop("F_local", None)
        if event_table is not None:
            for bundle in decoded_bundles.values():
                if isinstance(bundle, Mapping) and isinstance(
                    bundle.get("final_sentences"), list
                ):
                    enrich_segments_from_event_table(
                        bundle["final_sentences"], event_table
                    )

    eval_summary: dict[str, Any] | None = None
    if eval_summary_json is not None and Path(eval_summary_json).is_file():
        eval_summary = json.loads(
            Path(eval_summary_json).read_text(encoding="utf-8")
        )

    word_eval_summary: dict[str, Any] | None = None
    if word_eval_json is not None and Path(word_eval_json).is_file():
        word_eval_summary = json.loads(
            Path(word_eval_json).read_text(encoding="utf-8")
        )

    synth_res = export_synthetic_and_horizon_studies(
        repo_root=root,
        output_dir=out_dir,
        event_table=event_table,
        dataset=dataset,
        split=split,
    )
    sent_res = export_sentence_and_subject_coupling_studies(
        output_dir=out_dir,
        decoded_bundles=decoded_bundles,
        eval_summary=eval_summary,
        word_eval_summary=word_eval_summary,
        score_payload=score_payload,
        event_table=event_table,
        language=language,
        split=split,
        primary_condition=primary_condition,
        n_permutations=n_permutations,
    )

    all_files = {
        **(synth_res.get("exported_files") or {}),
        **(sent_res.get("exported_files") or {}),
    }
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "split": split,
        "language": language,
        "output_dir": str(out_dir),
        "exported_files": all_files,
        "synthetic_and_horizon": synth_res,
        "sentence_and_coupling": sent_res,
    }
    manifest_path = out_dir / "manuscript_export_manifest.json"
    save_json(manifest_path, manifest)
    all_files["manifest_json"] = str(manifest_path)
    return manifest
