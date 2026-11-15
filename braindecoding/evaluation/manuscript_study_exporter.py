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
import torch.nn.functional as F

from braindecoding.config import PROJECT_ROOT, project_path, results_root
from braindecoding.data.derived_cache import (
    canonical_cache_paths,
    restore_legacy_columns,
)
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

    subj_col = _pick_first_column(df.columns, ("subject_id", "受试者"))
    rec_col = _pick_first_column(df.columns, ("recording_id", "记录编号"))
    ctx_col = _pick_first_column(
        df.columns,
        (
            "runtime_context_uid",
            "sentence_uid",
            "context_id",
            "上下文编号",
        ),
    )
    group_cols: list[str] = []
    for c in (subj_col, rec_col, ctx_col):
        if c is not None and c not in group_cols:
            group_cols.append(c)

    ord_col = _pick_first_column(
        df.columns,
        ("in_recording_index", "记录内序号", "original_seq_index", "序列原编号"),
    )
    sort_cols: list[str] = [*group_cols]
    for c in (onset_col, ord_col):
        if c is not None and c not in sort_cols:
            sort_cols.append(c)
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")

    grouped_onsets: list[np.ndarray] = []
    if not group_cols:
        arr = pd.to_numeric(df[onset_col], errors="coerce").to_numpy(dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            grouped_onsets.append(arr)
    else:
        by_arg: Any = group_cols[0] if len(group_cols) == 1 else group_cols
        for _, grp in df.groupby(by_arg, sort=False):
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
        waits_win: list[float] = []
        for tau in grouped_onsets:
            n_tok = int(len(tau))
            if n_tok <= delay_guard:
                continue
            limit = n_tok - delay_guard
            last_idx = n_tok - 1
            for pos in range(limit):
                end_pos = last_idx if r_val is None else min(last_idx, pos + r_val)
                waits.append(max(win_s, float((tau[end_pos] + win_s) - tau[pos])))
                waits_win.append(float(end_pos - pos + 1))

        if not waits:
            continue
        arr_w = np.asarray(waits, dtype=np.float64)
        arr_win = np.asarray(waits_win, dtype=np.float64)
        out_rows.append(
            {
                "horizon": h_key,
                "paper_display_name": disp,
                "common_horizon_delay": delay_guard,
                "query_count": int(len(arr_w)),
                "mean_wait_windows": round(float(np.mean(arr_win)), 6),
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


_PAPER_CONDITION_DISPLAY_OVERRIDES: dict[str, str] = {
    "word_only": "Word-Only",
    "word_1p00s": "Word-Only (1.00s)",
    "word_1s": "Word-Only (1.00s)",
    "word_1p28s": "Word-Only (1.28s)",
    "long_1p27": "Word-Only (1.27s)",
    "word_1p50s": "Word-Only (1.50s)",
    "long_1p5": "Word-Only (1.50s)",
    "word_2p50s": "Word-Only (2.50s)",
    "F_joint_undebiased": "Full-Context (Joint)",
    "full_context_joint": "Full-Context (Joint)",
    "F_freeze_undebiased": "+ Freeze CNN",
    "full_context_freeze_cnn": "+ Freeze CNN",
    "F_debiased": "+ Debias, Full-Context",
    "debiased_full_context": "+ Debias, Full-Context",
    "debiased_local_r2": "+ Debias, Local (r = 2)",
    "K2_debiased": "+ Debias, Local (r = 2)",
    "local_r0": "Local (r = 0)",
    "rolling_k0": "Local (r = 0)",
    "local_r1": "Local (r = 1)",
    "rolling_k1": "Local (r = 1)",
    "local_r2": "Local (r = 2)",
    "rolling_k2": "Local (r = 2)",
}

CANONICAL_PALLIER2025_FIVE_MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "model_id": "word_only",
        "condition_key": "S0",
        "paper_display_name": "Word-Only",
        "exp_tag": "lr4e-4_bs256",
        "word_exp_tag": "lr4e-4_bs256",
        "stage": "word",
        "config_path": "configs/training/pallier2025_word_encoder.yaml",
        "common_horizon_delay": 2,
    },
    {
        "model_id": "full_context_joint",
        "condition_key": "F_joint_undebiased",
        "paper_display_name": "Full-Context (Joint)",
        "exp_tag": "stage1_lr4e-4_bs256_stage2_lr1e-4_bs128",
        "word_exp_tag": "lr4e-4_bs256",
        "stage": "context",
        "config_path": "configs/training/pallier2025_context_transformer.yaml",
        "common_horizon_delay": 2,
    },
    {
        "model_id": "full_context_freeze_cnn",
        "condition_key": "F_freeze_undebiased",
        "paper_display_name": "+ Freeze CNN",
        "exp_tag": "stage1_freeze_bs256_stage2_lr1e-4_bs128",
        "word_exp_tag": "lr4e-4_bs256",
        "stage": "context",
        "config_path": "configs/training/pallier2025_context_transformer.yaml",
        "common_horizon_delay": 2,
    },
    {
        "model_id": "debiased_full_context",
        "condition_key": "F_debiased",
        "paper_display_name": "+ Debias, Full-Context",
        "exp_tag": "lr4e-4_bs256_window_1.0_siamese_debiased_lamda_0.5_history_window",
        "word_exp_tag": "lr4e-4_bs256",
        "stage": "context",
        "config_path": "configs/training/siamese_debiased/pallier2025_siamese_debiased_context.yaml",
        "common_horizon_delay": 2,
    },
    {
        "model_id": "debiased_local_r2",
        "condition_key": "K2",
        "paper_display_name": "+ Debias, Local (r = 2)",
        "exp_tag": "2_lr4e-4_bs256_window_1.0_siamese_debiased_lamda_0.5_history_window",
        "word_exp_tag": "lr4e-4_bs256",
        "stage": "prefix_revision",
        "config_path": "configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml",
        "common_horizon_delay": None,
    },
)

CANONICAL_PALLIER2025_LONG_WINDOW_SPECS: tuple[dict[str, Any], ...] = (
    {
        "condition": "word_1p00s",
        "paper_display_name": "Word-Only (1.00s)",
        "exp_tag": "lr4e-4_bs256",
        "window_seconds": 1.00,
        "window_samples": 50,
    },
    {
        "condition": "word_1p28s",
        "paper_display_name": "Word-Only (1.28s)",
        "exp_tag": "lr4e-4_bs256_win1p28",
        "window_seconds": 1.28,
        "window_samples": 64,
    },
    {
        "condition": "word_1p50s",
        "paper_display_name": "Word-Only (1.50s)",
        "exp_tag": "lr4e-4_bs256_win1p50",
        "window_seconds": 1.50,
        "window_samples": 75,
    },
    {
        "condition": "word_2p50s",
        "paper_display_name": "Word-Only (2.50s)",
        "exp_tag": "lr4e-4_bs256_win2p50",
        "window_seconds": 2.50,
        "window_samples": 125,
    },
)


def _condition_paper_display_name(cond: str) -> str:
    """将内部模型或对照条件代号映射为论文统一术语。"""
    key = str(cond).strip()
    if key in _PAPER_CONDITION_DISPLAY_OVERRIDES:
        return _PAPER_CONDITION_DISPLAY_OVERRIDES[key]
    return paper_strategy_display_name(key)


def export_word_top10_neighborhood_examples(
    payload: Mapping[str, Any],
    *,
    conditions: Sequence[str] = (
        "S0",
        "F_joint_undebiased",
        "F_freeze_undebiased",
        "F_debiased",
        "K0",
        "K1",
        "K2",
        "F",
    ),
    target_words: Sequence[str] | None = None,
    word_mean_duration_ms: Mapping[str, float] | None = None,
    top_k: int = 10,
    max_examples: int = 12,
) -> list[dict[str, Any]]:
    """导出代表性目标词在不同建模视野（Word-Only / Undebiased Full-Context / Debiased Local r=0..2）下的 Top-K 神经近邻候选。"""
    vocab = [str(w) for w in (payload.get("vocabulary") or [])]
    if not vocab:
        return []
    vocab_to_idx = {w: idx for idx, w in enumerate(vocab)}
    ref_words = [str(w) for w in (payload.get("reference_words") or [])]
    if not ref_words:
        return []
    resolved_dur_map: Mapping[str, float] | None = (
        word_mean_duration_ms
        if word_mean_duration_ms is not None
        else payload.get("word_mean_duration_ms")
    )

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

    # Collect per-word row indices and optional content-word hints from segments or payload
    content_words_seen: set[str] = {
        str(w) for w in (payload.get("content_words") or ())
    }
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
        base_conds = [
            c
            for c in ("S0", "F_joint_undebiased", "F_freeze_undebiased")
            if c in clean_matrices
        ]
        if not base_conds:
            base_conds = [next(iter(clean_matrices))]
        focus_cond = "K2" if "K2" in clean_matrices else next(iter(clean_matrices))
        focus_mat = clean_matrices[focus_cond]
        max_occ = max((len(idxs) for idxs in word_rows.values()), default=1)
        min_occ_pref = 5 if max_occ >= 5 else 1
        scored_candidates: list[tuple[int, int, float, str]] = []
        for w, idxs in word_rows.items():
            t_idx = vocab_to_idx[w]
            idx_arr = np.asarray(idxs, dtype=np.int64)
            mean_focus = np.mean(focus_mat[idx_arr], axis=0)
            rank_focus = 1 + int(np.sum(mean_focus > mean_focus[t_idx]))
            base_ranks: list[int] = []
            for bc in base_conds:
                mean_base = np.mean(clean_matrices[bc][idx_arr], axis=0)
                base_ranks.append(1 + int(np.sum(mean_base > mean_base[t_idx])))
            rank_base = float(np.mean(base_ranks))
            has_support = 1 if len(idxs) >= min_occ_pref else 0
            is_cw = 1 if (not content_words_seen or w in content_words_seen) else 0
            gain = float(rank_base - rank_focus)
            scored_candidates.append(
                (has_support, is_cw, gain - 0.05 * rank_focus, w)
            )
        scored_candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
        chosen_words = [item[3] for item in scored_candidates[: max(1, int(max_examples))]]

    def _lookup_dur_ms(word_str: str) -> float | None:
        if resolved_dur_map is None or word_str not in resolved_dur_map:
            return None
        val = float(resolved_dur_map[word_str])
        return round(val, 2) if math.isfinite(val) and val > 0.0 else None

    k_use = max(1, min(int(top_k), len(vocab)))
    examples_out: list[dict[str, Any]] = []
    for word in chosen_words:
        t_idx = vocab_to_idx[word]
        idx_arr = np.asarray(word_rows[word], dtype=np.int64)
        target_dur_ms = _lookup_dur_ms(word)
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
                cand_word = vocab[v_i]
                cand_obj: dict[str, Any] = {
                    "rank": r_i,
                    "word": cand_word,
                    "score": round(float(mean_scores[v_i]), 6),
                    "is_target": bool(v_i == t_idx),
                }
                if norm_emb is not None:
                    cos_v = float(np.dot(norm_emb[v_i], norm_emb[t_idx]))
                    cand_obj["cosine_to_target"] = round(cos_v, 6)
                cand_dur_ms = _lookup_dur_ms(cand_word)
                if cand_dur_ms is not None:
                    cand_obj["candidate_mean_duration_ms"] = cand_dur_ms
                cands.append(cand_obj)
            cond_entries[cond] = {
                "condition": cond,
                "paper_display_name": _condition_paper_display_name(cond),
                "target_rank": int(target_rank),
                "target_score": round(float(mean_scores[t_idx]), 6),
                "top_candidates": cands,
            }
        ex_obj: dict[str, Any] = {
            "target_word": word,
            "occurrence_count": int(len(idx_arr)),
            "is_content_word": bool(word in content_words_seen)
            if content_words_seen
            else None,
            "conditions": cond_entries,
        }
        if target_dur_ms is not None:
            ex_obj["target_mean_duration_ms"] = target_dur_ms
        examples_out.append(ex_obj)
    return examples_out


def _flatten_word_top10_neighborhoods(
    examples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """将 Top-10 近邻示例展开为适合 CSV 导出的平坦行列表。"""
    flat_rows: list[dict[str, Any]] = []
    for ex in examples:
        target_dur = ex.get("target_mean_duration_ms", "")
        for cond_k, cond_v in (ex.get("conditions") or {}).items():
            for cand in cond_v.get("top_candidates") or []:
                flat_rows.append(
                    {
                        "target_word": ex["target_word"],
                        "occurrence_count": ex["occurrence_count"],
                        "target_mean_duration_ms": target_dur,
                        "condition": cond_k,
                        "paper_display_name": cond_v["paper_display_name"],
                        "target_rank": cond_v["target_rank"],
                        "candidate_rank": cand["rank"],
                        "candidate_word": cand["word"],
                        "candidate_mean_duration_ms": cand.get(
                            "candidate_mean_duration_ms", ""
                        ),
                        "score": cand["score"],
                        "cosine_to_target": cand.get("cosine_to_target", ""),
                        "is_target": cand["is_target"],
                    }
                )
    return flat_rows


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
    return _condition_paper_display_name(cond)


def _apply_cache_dir_override(
    cfg: dict[str, Any], dataset: str, cache_dir: str | Path | None
) -> None:
    """将 workflow cfg['cache'] 路径指向用户指定的 local cache 目录（支持根缓存或数据集子目录）。"""
    if cache_dir is None:
        return
    raw_dir = Path(cache_dir).resolve()
    ds_dir = (
        raw_dir
        if (raw_dir / "events").is_dir() or (raw_dir / "meg").is_dir()
        else raw_dir / str(dataset)
    )
    cache_cfg = cfg.setdefault("cache", {})
    for ev_name in ("events.csv", "event_table.csv"):
        ev_cand = ds_dir / "events" / ev_name
        if ev_cand.is_file():
            cache_cfg["event_table"] = str(ev_cand)
            break
    meg_cand = ds_dir / "meg"
    if meg_cand.is_dir():
        cache_cfg["meg_dir"] = str(meg_cand)
    eeg_cand = ds_dir / "eeg"
    if eeg_cand.is_dir():
        cache_cfg["eeg_dir"] = str(eeg_cand)
    emb_cand = ds_dir / "text" / "t5_large_layer_0_5" / "embeddings.npz"
    if emb_cand.is_file():
        cache_cfg["text_embeddings"] = str(emb_cand)


def _fit_train_word_duration_stats(
    event_table: pd.DataFrame | None,
) -> tuple[list[str], list[float], dict[str, float]]:
    """从训练划分拟合词表先验分布所需的词列表、时长秒数及按词均时长（ms）映射。"""
    if event_table is None or len(event_table) == 0:
        return [], [], {}
    sp_col = _pick_first_column(
        event_table.columns, ("data_split", "数据划分", "split")
    )
    w_col = _pick_first_column(
        event_table.columns, ("normalized_word", "标准词", "word", "词")
    )
    s_col = _pick_first_column(event_table.columns, ("onset_seconds", "开始时间"))
    e_col = _pick_first_column(event_table.columns, ("offset_seconds", "结束时间"))
    if w_col is None or s_col is None or e_col is None:
        return [], [], {}
    tr_df = (
        event_table[event_table[sp_col].astype(str) == "train"]
        if sp_col is not None
        else event_table
    )
    if len(tr_df) == 0:
        tr_df = event_table
    tr_st = pd.to_numeric(tr_df[s_col], errors="coerce").to_numpy(dtype=np.float64)
    tr_en = pd.to_numeric(tr_df[e_col], errors="coerce").to_numpy(dtype=np.float64)
    tr_w = tr_df[w_col].astype(str).tolist()
    train_words: list[str] = []
    train_durs: list[float] = []
    dur_accum: dict[str, list[float]] = {}
    for w_i, st_i, en_i in zip(tr_w, tr_st, tr_en):
        if math.isfinite(st_i) and math.isfinite(en_i) and en_i > st_i:
            dur_s = float(en_i - st_i)
            train_words.append(w_i)
            train_durs.append(dur_s)
            dur_accum.setdefault(w_i, []).append(dur_s * 1000.0)
    mean_ms = {w: float(np.mean(vals)) for w, vals in dur_accum.items() if vals}
    return train_words, train_durs, mean_ms


def _locate_seed_checkpoint(
    ds_results_dir: Path,
    exp_tag: str,
    stage: str,
    subset_tag: str,
    seed: int,
) -> Path:
    stage_candidates = (stage, "revision") if stage == "prefix_revision" else (stage,)
    checked: list[Path] = []
    for st in stage_candidates:
        seed_dir = ds_results_dir / exp_tag / st / subset_tag / f"seed-{int(seed):03d}"
        for ckpt_name in ("best.ckpt", "best.pt"):
            cand = seed_dir / ckpt_name
            checked.append(cand)
            if cand.is_file():
                return cand
    raise FileNotFoundError(
        f"未找到阶段 {stage!r} (exp_tag={exp_tag!r}, seed={seed}) 的最佳检查点；"
        f"已检查: {[str(p) for p in checked]}"
    )


def _locate_multi_seed_summary_json(
    ds_results_dir: Path,
    exp_tag: str,
    eval_subdir: str | None,
    stage: str,
    subset_tag: str,
    split: str,
) -> Path | None:
    sub = str(eval_subdir or "").strip().strip("/\\")
    stage_candidates = (stage, "revision") if stage == "prefix_revision" else (stage,)
    roots: list[Path] = []
    for st in stage_candidates:
        if sub:
            roots.append(ds_results_dir / exp_tag / sub / st / subset_tag)
        roots.append(ds_results_dir / exp_tag / st / subset_tag)
    for base in roots:
        for fname in (
            f"multi_seed_evaluation_{split}.json",
            f"multi_seed_evaluation_common_subset_{split}.json",
        ):
            cand = base / fname
            if cand.is_file():
                return cand
    return None


def _locate_seed_encoded_cache(
    ds_results_dir: Path,
    exp_tag: str,
    eval_subdir: str | None,
    stage: str,
    subset_tag: str,
    seed: int,
    split: str,
) -> Path | None:
    sub = str(eval_subdir or "").strip().strip("/\\")
    stage_candidates = (stage, "revision") if stage == "prefix_revision" else (stage,)
    seed_name = f"seed-{int(seed):03d}"
    for st in stage_candidates:
        base_dirs = []
        if sub:
            base_dirs.append(ds_results_dir / exp_tag / sub / st / subset_tag / seed_name)
        base_dirs.append(ds_results_dir / exp_tag / st / subset_tag / seed_name)
        for s_dir in base_dirs:
            for rel in (
                Path("exploratory") / f"subspace_encoded_cache_{split}.pt",
                Path("common_subset") / "exploratory" / f"subspace_encoded_cache_{split}.pt",
            ):
                cand = s_dir / rel
                if cand.is_file():
                    return cand
    return None


def _align_encoded_predictions_to_event_ids(
    encoded: Mapping[str, Any],
    target_event_ids: Sequence[str],
    *,
    horizon_delay: int | None = None,
) -> np.ndarray | None:
    """将已编码预测张量严格按 target_event_ids 顺序对齐并 L2 归一化为 float32 矩阵 [N, d]。"""
    eids = [str(e) for e in (encoded.get("event_ids") or ())]
    if not eids:
        return None
    pred_tensor: torch.Tensor | None = None
    if horizon_delay is not None and isinstance(encoded.get("revision_predictions"), Mapping):
        rev_map = encoded["revision_predictions"]
        pred_tensor = rev_map.get(int(horizon_delay))
        if pred_tensor is None:
            pred_tensor = rev_map.get(str(int(horizon_delay)))
    if pred_tensor is None:
        raw_p = encoded.get("predictions")
        if isinstance(raw_p, torch.Tensor):
            pred_tensor = raw_p
    if pred_tensor is None or len(pred_tensor) != len(eids):
        return None
    eid_to_idx = {eid: idx for idx, eid in enumerate(eids)}
    try:
        gather_idx = [eid_to_idx[str(eid)] for eid in target_event_ids]
    except KeyError:
        return None
    aligned = pred_tensor.detach().cpu().to(dtype=torch.float32)[gather_idx]
    aligned = F.normalize(aligned, p=2.0, dim=-1, eps=1e-8)
    return aligned.numpy()


def _extract_raw_control_metrics_from_seed_result(
    sr: Mapping[str, Any],
    *,
    stage: str,
    horizon_key: str | None = None,
) -> dict[str, dict[str, float]]:
    """从单个 seed_result 提取未经 post-hoc CMS 处理（raw）的 Clean 与 Shifted 核心词级指标。"""
    nc = sr.get("negative_controls") or {}
    nc_methods = nc.get("methods") if isinstance(nc.get("methods"), Mapping) else {}
    raw_nc = (
        nc_methods.get("raw")
        or nc_methods.get("raw_eta0")
        or nc
    )
    clean_ref = (
        raw_nc.get("clean_reference")
        if isinstance(raw_nc.get("clean_reference"), Mapping)
        else {}
    )
    shifted_agg = (
        raw_nc.get("shifted_aggregate")
        if isinstance(raw_nc.get("shifted_aggregate"), Mapping)
        else {}
    )

    if stage in ("revision", "prefix_revision") and horizon_key:
        c_hz = (clean_ref.get("horizons") or {}).get(horizon_key) or {}
        if not c_hz:
            c_hz = (sr.get("revision_horizons") or {}).get(horizon_key) or {}
        s_hz = (shifted_agg.get("horizons") or {}).get(horizon_key) or {}
        clean_out = {
            "vocab50_macro_top10": float(c_hz.get("common_vocab50_macro_top10", 0.0)),
            "vocab50_macro_top1": float(c_hz.get("common_vocab50_macro_top1", 0.0)),
            "full_vocab_macro_top10": float(
                c_hz.get(
                    "common_full_vocab_word_macro_top10",
                    c_hz.get("common_full_vocab_macro_top10", 0.0),
                )
            ),
            "full_vocab_subject_macro_top10": float(
                c_hz.get(
                    "common_full_vocab_subject_macro_top10",
                    c_hz.get("common_full_vocab_macro_top10", 0.0),
                )
            ),
        }
        shifted_out = {
            "vocab50_macro_top10": float(s_hz.get("mean_common_vocab50_macro_top10", 0.0)),
            "vocab50_macro_top1": float(s_hz.get("mean_common_vocab50_macro_top1", 0.0)),
            "full_vocab_macro_top10": float(
                s_hz.get(
                    "mean_common_full_vocab_word_macro_top10",
                    s_hz.get("mean_common_full_vocab_macro_top10", 0.0),
                )
            ),
            "full_vocab_subject_macro_top10": float(
                s_hz.get(
                    "mean_common_full_vocab_subject_macro_top10",
                    s_hz.get("mean_common_full_vocab_macro_top10", 0.0),
                )
            ),
        }
        return {"clean": clean_out, "shifted": shifted_out}

    sweep = sr.get("vocabulary_sweep") or {}
    n50 = sweep.get("N50") or {}
    fv = sweep.get("full_vocab") or {}
    clean_out = {
        "vocab50_macro_top10": float(
            clean_ref.get("vocab50_macro_top10", n50.get("macro_top10", 0.0))
        ),
        "vocab50_macro_top1": float(
            clean_ref.get("vocab50_macro_top1", n50.get("macro_top1", 0.0))
        ),
        "full_vocab_macro_top10": float(
            clean_ref.get(
                "full_vocab_word_macro_top10",
                clean_ref.get(
                    "full_vocab_macro_top10",
                    fv.get("word_macro_top10", fv.get("macro_top10", 0.0)),
                ),
            )
        ),
        "full_vocab_subject_macro_top10": float(
            clean_ref.get(
                "full_vocab_subject_macro_top10",
                fv.get("subject_macro_top10", fv.get("micro_top10", 0.0)),
            )
        ),
    }
    shifted_out = {
        "vocab50_macro_top10": float(shifted_agg.get("mean_vocab50_macro_top10", 0.0)),
        "vocab50_macro_top1": float(shifted_agg.get("mean_vocab50_macro_top1", 0.0)),
        "full_vocab_macro_top10": float(
            shifted_agg.get(
                "mean_full_vocab_word_macro_top10",
                shifted_agg.get("mean_full_vocab_macro_top10", 0.0),
            )
        ),
        "full_vocab_subject_macro_top10": float(
            shifted_agg.get(
                "mean_full_vocab_subject_macro_top10",
                shifted_agg.get("mean_full_vocab_macro_top10", 0.0),
            )
        ),
    }
    return {"clean": clean_out, "shifted": shifted_out}


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    sd = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    return float(np.mean(arr)), sd


def evaluate_five_model_t5_diagnostics(
    *,
    results_dir: str | Path | None = None,
    output_dir: str | Path,
    dataset: str = "pallier2025",
    split: str = "test",
    seeds: Sequence[int] = (0, 1, 2),
    eval_subdir: str = "causal_fast_audit",
    synthetic_conditions: Sequence[str] = ("shared", "independent"),
    synthetic_seed: int = 20261009,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    event_table: pd.DataFrame | None = None,
    cache_dir: str | Path | None = None,
    num_workers: int | None = None,
    device: str | torch.device = "auto",
    n_permutations: int = 200,
    force: bool = False,
) -> dict[str, Any]:
    """基于现有 T5-Large 5 组模型检查点执行 SyntheticTimingDataset 纯前向脉冲评估、5 模型 Δ_JS 先验对齐及 Top-10 近邻诊断。"""
    import copy

    from braindecoding.data.synthetic_timing_dataset import SyntheticTimingDataset
    from braindecoding.evaluation.subspace_coherence_debiasing_evaluation import (
        _CachedEvalDatasetProxy,
    )
    from braindecoding.evaluation.word_retrieval_evaluation import (
        _build_common_horizon_event_filter,
        _compute_encoded_evaluation_metrics,
        _extract_control_summary_row,
        _slice_encoded_dict,
    )
    from braindecoding.models import build_brain_embedding_model
    from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import (
        train as chinese_task,
    )
    from braindecoding.tasks.word_decoding.pallier2025 import train as pallier_task
    from braindecoding.training.training_runtime import choose_device, load_checkpoint
    from braindecoding.training.word_encoder_training import (
        encode_loader,
        make_loader,
        resolve_amp_dtype,
    )
    from braindecoding.training.workflow_runner import (
        build_workflow_config,
        group_column,
    )

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(s) for s in seeds]
    if not seed_list:
        raise ValueError("seeds 不能为空。")

    diag_json_path = out_dir / f"five_model_t5_diagnostics_{split}.json"
    if not force and diag_json_path.is_file():
        try:
            cached = json.loads(diag_json_path.read_text(encoding="utf-8"))
            cached_files = cached.get("exported_files") or {}
            if (
                str(cached.get("dataset")) == str(dataset)
                and str(cached.get("split")) == str(split)
                and list(cached.get("seeds") or []) == seed_list
                and cached_files
                and all(Path(p).is_file() for p in cached_files.values())
            ):
                return cached
        except Exception:
            pass

    if model_specs is not None:
        resolved_specs = [dict(sp) for sp in model_specs]
    elif str(dataset) == "pallier2025":
        resolved_specs = [dict(sp) for sp in CANONICAL_PALLIER2025_FIVE_MODEL_SPECS]
    else:
        raise ValueError(
            f"数据集 {dataset!r} 未提供默认 CANONICAL_FIVE_MODEL_SPECS，请显式传入 model_specs。"
        )

    results_root_dir = (
        Path(results_dir).resolve() if results_dir is not None else results_root()
    )
    ds_results_dir = results_root_dir / str(dataset)

    base_cfg, subset_spec = build_workflow_config(
        dataset,
        stage="word",
        seed=seed_list[0],
        subject="all",
        mode="eval",
        num_workers=num_workers,
    )
    _apply_cache_dir_override(base_cfg, dataset, cache_dir)
    task = pallier_task if str(dataset) == "pallier2025" else chinese_task
    selected_subjects = list(base_cfg["dataset"]["subjects"])
    subset_tag = str(subset_spec["subset_tag"])

    # Load train and eval tables via canonical task loader
    ev_path_str = str(base_cfg["cache"]["event_table"])
    ev_disk_df = event_table if event_table is not None else None
    if Path(ev_path_str).is_file():
        train_table = task.dataset_module.load_event_table(
            ev_path_str,
            split="train",
            subjects=selected_subjects,
            trainable_only=True,
        )
        eval_table_loaded = task.dataset_module.load_event_table(
            ev_path_str,
            split=split,
            subjects=selected_subjects,
            trainable_only=True,
        )
        if ev_disk_df is None:
            ev_disk_df = pd.concat([train_table, eval_table_loaded], ignore_index=True)
    elif ev_disk_df is not None:
        sp_col = _pick_first_column(ev_disk_df.columns, ("data_split", "数据划分", "split"))
        tr_col = _pick_first_column(ev_disk_df.columns, ("is_trainable", "是否可训练"))
        work_df = ev_disk_df.copy()
        if tr_col is not None:
            work_df = work_df[work_df[tr_col].astype(bool)].reset_index(drop=True)
        if sp_col is not None:
            train_table = work_df[work_df[sp_col].astype(str) == "train"].reset_index(drop=True)
            eval_table_loaded = work_df[work_df[sp_col].astype(str) == str(split)].reset_index(drop=True)
        else:
            train_table = work_df
            eval_table_loaded = work_df
    else:
        raise FileNotFoundError(f"未找到事件表缓存：{ev_path_str}")

    if len(eval_table_loaded) == 0:
        raise ValueError(f"数据集 {dataset} 在 split={split!r} 下无可用可训练事件。")

    zero_base_ds = task.build_dataset(base_cfg, eval_table_loaded, zero_signal=True)
    kept_event_ids, event_id_to_block = _build_common_horizon_event_filter(
        base_cfg,
        dataset=str(dataset),
        split=split,
        selected_subjects=selected_subjects,
        common_horizon_delay=2,
        fallback_eval_table=zero_base_ds.table,
    )

    ordered_common_indices: list[int] = [
        idx
        for idx, eid in enumerate(zero_base_ds.table["event_id"].astype(str).tolist())
        if eid in kept_event_ids
    ]
    if not ordered_common_indices:
        raise RuntimeError(
            f"在数据集 {dataset} (split={split}) 上应用 common_horizon_delay=2 后共同有效事件数为 0。"
        )
    common_table = zero_base_ds.table.iloc[ordered_common_indices].reset_index(drop=True)
    ordered_common_eids = common_table["event_id"].astype(str).tolist()
    common_words = common_table["normalized_word"].astype(str).tolist()
    common_eids_arr = np.asarray(ordered_common_eids, dtype=object)

    # Target durations for common queries + training duration prior
    s_col_ev = _pick_first_column(common_table.columns, ("onset_seconds", "开始时间"))
    e_col_ev = _pick_first_column(common_table.columns, ("offset_seconds", "结束时间"))
    if s_col_ev is not None and e_col_ev is not None:
        st_arr = pd.to_numeric(common_table[s_col_ev], errors="coerce").to_numpy(dtype=np.float64)
        en_arr = pd.to_numeric(common_table[e_col_ev], errors="coerce").to_numpy(dtype=np.float64)
        common_durs_s = en_arr - st_arr
    else:
        common_durs_s = np.full(len(ordered_common_eids), float("nan"), dtype=np.float64)

    train_words_fit, train_durs_fit, train_word_mean_ms = _fit_train_word_duration_stats(
        train_table if len(train_table) > 0 else ev_disk_df
    )
    content_words_set: set[str] = set()
    icw_col = _pick_first_column(common_table.columns, ("is_content_word", "是否内容词"))
    if icw_col is not None:
        for w_v, flag_v in zip(common_words, common_table[icw_col].tolist()):
            if bool(flag_v):
                content_words_set.add(str(w_v))

    train_word_col = (
        "normalized_word"
        if "normalized_word" in train_table.columns
        else "标准词"
    )
    vocabulary = [
        w
        for w in sorted(set(train_table[train_word_col].astype(str)))
        if w in zero_base_ds.embedding_map
    ]
    vocab50 = list(
        pallier_task.frozen_vocabulary()
        if str(dataset) == "pallier2025"
        else chinese_task.frozen_vocabulary(str(base_cfg["cache"]["event_table"]))
    )
    emb_dim = int(next(iter(zero_base_ds.embedding_map.values())).shape[0])
    vocab_emb_np = np.stack(
        [
            np.asarray(zero_base_ds.embedding_map[w], dtype=np.float32)
            for w in vocabulary
        ],
        axis=0,
    )
    vocab_norms = np.linalg.norm(vocab_emb_np, axis=1, keepdims=True)
    vocab_emb_norm = (vocab_emb_np / np.maximum(vocab_norms, 1e-8)).astype(np.float32)

    raw_table_dedup = zero_base_ds.table.copy()
    raw_table_dedup["_eid_str"] = raw_table_dedup["event_id"].astype(str)
    dedup_table = raw_table_dedup.drop_duplicates(subset=["_eid_str"]).set_index("_eid_str")

    def _metrics_from_encoded(
        encoded_dict: Mapping[str, Any],
        cfg_m: dict[str, Any],
        *,
        stage_m: str,
        horizon_delay_filter: int | None,
    ) -> dict[str, Any]:
        eids_all = [str(e) for e in encoded_dict["event_ids"]]
        if horizon_delay_filter is not None or stage_m not in ("revision", "prefix_revision"):
            keep_mask = np.asarray([eid in kept_event_ids for eid in eids_all], dtype=bool)
            enc_sub = _slice_encoded_dict(dict(encoded_dict), keep_mask)
        else:
            enc_sub = dict(encoded_dict)
        sub_df = dedup_table.reindex(enc_sub["event_ids"]).reset_index(drop=True)
        sub_df["event_id"] = list(enc_sub["event_ids"])
        if (
            event_id_to_block
            and "sentence_uid" not in sub_df.columns
            and "上下文编号" not in sub_df.columns
        ):
            sub_df["sentence_uid"] = [
                event_id_to_block.get(str(eid), f"blk-{i // 8:04d}")
                for i, eid in enumerate(enc_sub["event_ids"])
            ]
        proxy_ds = _CachedEvalDatasetProxy(
            sub_df,
            zero_base_ds.embedding_map,
            window_seconds=float(cfg_m["dataset"].get("window_seconds", 1.0)),
            window_samples=int(round(float(cfg_m["dataset"].get("window_seconds", 1.0)) * 50.0)),
        )
        metrics_res = _compute_encoded_evaluation_metrics(
            enc_sub,
            proxy_ds,
            vocabulary,
            vocab50,
            cfg_m,
            stage=stage_m,
            split=split,
        )
        return _extract_control_summary_row("synth", "Synthetic", None, metrics_res)

    dev = (
        device
        if isinstance(device, torch.device)
        else choose_device(str(device))
    )
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    synthetic_datasets = {
        str(cond): SyntheticTimingDataset(
            zero_base_ds,
            str(cond),
            split=split,
            evaluation_seed=int(synthetic_seed),
            include_onset_seconds=True,
        )
        for cond in synthetic_conditions
    }

    ledger_rows: list[dict[str, Any]] = []
    synthetic_tidy_rows: list[dict[str, Any]] = []
    duration_alignment_rows: list[dict[str, Any]] = []

    # Retain ensemble Clean score matrices per condition for unified cross-model Top-10 neighborhoods
    ensemble_clean_scores: dict[str, np.ndarray] = {}
    model_summaries: list[dict[str, Any]] = []

    valid_dur_mask = np.isfinite(common_durs_s) & (common_durs_s > 0.0)
    dur_idx = np.flatnonzero(valid_dur_mask)

    for spec in resolved_specs:
        model_id = str(spec["model_id"])
        cond_key = str(spec["condition_key"])
        disp_name = str(spec.get("paper_display_name") or _condition_paper_display_name(cond_key))
        exp_tag = str(spec["exp_tag"])
        word_exp_tag = str(spec.get("word_exp_tag") or exp_tag)
        stage_m = str(spec["stage"])
        cfg_rel = spec.get("config_path")
        horizon_filter = spec.get("common_horizon_delay")

        summary_json_p = _locate_multi_seed_summary_json(
            ds_results_dir, exp_tag, eval_subdir, stage_m, subset_tag, split
        )
        summary_seed_map: dict[int, Mapping[str, Any]] = {}
        if summary_json_p is not None:
            ms_obj = json.loads(summary_json_p.read_text(encoding="utf-8"))
            for sr_item in ms_obj.get("seed_results") or ():
                if isinstance(sr_item, Mapping) and "seed" in sr_item:
                    summary_seed_map[int(sr_item["seed"])] = sr_item

        # Sub-entries to collect: primary model condition + (if prefix_revision) K0 & K1 horizons
        if stage_m in ("revision", "prefix_revision"):
            track_items = [
                ("K2", 2, disp_name, True),
                ("K0", 0, _condition_paper_display_name("K0"), False),
                ("K1", 1, _condition_paper_display_name("K1"), False),
            ]
        else:
            track_items = [(cond_key, None, disp_name, True)]

        per_track_metrics: dict[str, dict[str, list[float]]] = {
            tk: {
                "clean_50w_top10": [],
                "clean_50w_top1": [],
                "clean_fv_top10": [],
                "clean_fv_subj_top10": [],
                "shifted_50w_top10": [],
                "shifted_50w_top1": [],
                "shifted_fv_top10": [],
                "shifted_fv_subj_top10": [],
                **{f"synth_{sc}_50w_top10": [] for sc in synthetic_conditions},
                **{f"synth_{sc}_50w_top1": [] for sc in synthetic_conditions},
                **{f"synth_{sc}_fv_top10": [] for sc in synthetic_conditions},
                **{f"synth_{sc}_fv_subj_top10": [] for sc in synthetic_conditions},
            }
            for tk, _, _, _ in track_items
        }
        per_track_align_scores: dict[tuple[str, str], np.ndarray] = {}
        per_track_align_counts: dict[tuple[str, str], int] = {}
        per_track_seed_js: dict[tuple[str, str], list[float]] = {}
        per_track_seed_mae: dict[tuple[str, str], list[float]] = {}

        for sd in seed_list:
            cfg_m, _ = build_workflow_config(
                dataset,
                stage=stage_m,
                seed=sd,
                subject="all",
                exp_tag=exp_tag,
                word_exp_tag=word_exp_tag,
                config_path=cfg_rel,
                mode="eval",
                num_workers=num_workers,
            )
            _apply_cache_dir_override(cfg_m, dataset, cache_dir)
            ckpt_file = _locate_seed_checkpoint(
                ds_results_dir, exp_tag, stage_m, subset_tag, sd
            )
            checkpoint = load_checkpoint(ckpt_file, map_location="cpu")
            model_cfg = checkpoint.get("model_config", cfg_m["model"])
            ckpt_subjects = list(checkpoint.get("subject_ids", selected_subjects))
            subject_count = max(len(ckpt_subjects), zero_base_ds.subject_count)

            model = build_brain_embedding_model(
                zero_base_ds.channel_count,
                zero_base_ds.channel_positions,
                subject_count,
                model_cfg,
            ).to(dev)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            tc = cfg_m["training"]
            eval_bs = int(tc.get("eval_batch_size", tc.get("batch_size", 512)))
            amp_enabled = bool(tc.get("amp", True)) and dev.type == "cuda"
            amp_dtype = resolve_amp_dtype(tc)

            # 1) Synthetic forward passes ("shared", "independent")
            synth_encoded_by_cond: dict[str, dict[str, Any]] = {}
            for cond in synthetic_conditions:
                synth_loader, _ = make_loader(
                    synthetic_datasets[str(cond)],
                    tc,
                    shuffle=False,
                    group_column=group_column(cfg_m),
                    batch_size=eval_bs,
                )
                enc_s = encode_loader(
                    model,
                    synth_loader,
                    dev,
                    amp=amp_enabled,
                    amp_dtype=amp_dtype,
                    show_progress=False,
                )
                synth_encoded_by_cond[str(cond)] = enc_s
                synth_row = _metrics_from_encoded(
                    enc_s,
                    cfg_m,
                    stage_m=stage_m,
                    horizon_delay_filter=horizon_filter,
                )
                for tk, h_delay, _, _ in track_items:
                    if stage_m in ("revision", "prefix_revision") and h_delay is not None:
                        h_dict = (synth_row.get("horizons") or {}).get(f"K{h_delay}") or {}
                        v50_10 = float(h_dict.get("common_vocab50_macro_top10", 0.0))
                        v50_1 = float(h_dict.get("common_vocab50_macro_top1", 0.0))
                        fv_10 = float(
                            h_dict.get(
                                "common_full_vocab_word_macro_top10",
                                h_dict.get("common_full_vocab_macro_top10", 0.0),
                            )
                        )
                        fv_subj_10 = float(
                            h_dict.get(
                                "common_full_vocab_subject_macro_top10",
                                fv_10,
                            )
                        )
                    else:
                        v50_10 = float(synth_row.get("vocab50_macro_top10", 0.0))
                        v50_1 = float(synth_row.get("vocab50_macro_top1", 0.0))
                        fv_10 = float(
                            synth_row.get(
                                "full_vocab_word_macro_top10",
                                synth_row.get("full_vocab_macro_top10", 0.0),
                            )
                        )
                        fv_subj_10 = float(
                            synth_row.get("full_vocab_subject_macro_top10", fv_10)
                        )
                    per_track_metrics[tk][f"synth_{cond}_50w_top10"].append(v50_10 * 100.0)
                    per_track_metrics[tk][f"synth_{cond}_50w_top1"].append(v50_1 * 100.0)
                    per_track_metrics[tk][f"synth_{cond}_fv_top10"].append(fv_10 * 100.0)
                    per_track_metrics[tk][f"synth_{cond}_fv_subj_top10"].append(fv_subj_10 * 100.0)

                    if str(cond) == "shared":
                        preds_sh = _align_encoded_predictions_to_event_ids(
                            enc_s,
                            ordered_common_eids,
                            horizon_delay=h_delay,
                        )
                        if preds_sh is not None:
                            s_mat = (preds_sh @ vocab_emb_norm.T).astype(np.float32)
                            acc_k = (tk, "synthetic_shared")
                            if acc_k not in per_track_align_scores:
                                per_track_align_scores[acc_k] = s_mat.copy()
                            else:
                                per_track_align_scores[acc_k] += s_mat
                            per_track_align_counts[acc_k] = (
                                per_track_align_counts.get(acc_k, 0) + 1
                            )
                            if len(dur_idx) >= 2 and train_words_fit and train_durs_fit:
                                js_seed = compute_duration_prior_alignment_metrics(
                                    s_mat[dur_idx],
                                    vocabulary,
                                    [common_words[int(i)] for i in dur_idx],
                                    common_durs_s[dur_idx],
                                    train_words=train_words_fit,
                                    train_durations_seconds=train_durs_fit,
                                    n_permutations=0,
                                )
                                per_track_seed_js.setdefault(acc_k, []).append(
                                    float(js_seed["delta_js_bits"])
                                )
                                per_track_seed_mae.setdefault(acc_k, []).append(
                                    float(js_seed["top10_duration_error_reduction_ms"])
                                )
                            del s_mat

            # 2) Clean & Shifted encoded embeddings + Clean/Shifted metrics
            pt_cache_p = _locate_seed_encoded_cache(
                ds_results_dir, exp_tag, eval_subdir, stage_m, subset_tag, sd, split
            )
            clean_enc_for_seed: Mapping[str, Any] | None = None
            shifted_enc_for_seed: Mapping[str, Any] | None = None
            if pt_cache_p is not None:
                pt_bundle = torch.load(pt_cache_p, map_location="cpu", weights_only=False)
                clean_enc_for_seed = pt_bundle.get("clean_encoded")
                sh_list = pt_bundle.get("shifted_encoded_by_shift") or ()
                if sh_list:
                    shifted_enc_for_seed = sh_list[0][3]
                del pt_bundle

            if clean_enc_for_seed is None:
                # Fallback: run live Clean 0s forward if MEG signals exist on local cache
                try:
                    real_eval_ds = task.build_dataset(cfg_m, eval_table_loaded, zero_signal=False)
                    real_loader, _ = make_loader(
                        real_eval_ds,
                        tc,
                        shuffle=False,
                        group_column=group_column(cfg_m),
                        batch_size=eval_bs,
                    )
                    clean_enc_for_seed = encode_loader(
                        model,
                        real_loader,
                        dev,
                        amp=amp_enabled,
                        amp_dtype=amp_dtype,
                        show_progress=False,
                    )
                except Exception:
                    clean_enc_for_seed = None

            del model
            if dev.type == "cuda":
                torch.cuda.empty_cache()

            sr_item = summary_seed_map.get(sd)
            clean_fallback_row: dict[str, Any] | None = None
            if sr_item is None and clean_enc_for_seed is not None:
                clean_fallback_row = _metrics_from_encoded(
                    clean_enc_for_seed,
                    cfg_m,
                    stage_m=stage_m,
                    horizon_delay_filter=horizon_filter,
                )

            for tk, h_delay, _, _ in track_items:
                if sr_item is not None:
                    ext = _extract_raw_control_metrics_from_seed_result(
                        sr_item,
                        stage=stage_m,
                        horizon_key=(f"K{h_delay}" if h_delay is not None else None),
                    )
                    c_m, s_m = ext["clean"], ext["shifted"]
                    per_track_metrics[tk]["clean_50w_top10"].append(c_m["vocab50_macro_top10"] * 100.0)
                    per_track_metrics[tk]["clean_50w_top1"].append(c_m["vocab50_macro_top1"] * 100.0)
                    per_track_metrics[tk]["clean_fv_top10"].append(c_m["full_vocab_macro_top10"] * 100.0)
                    per_track_metrics[tk]["clean_fv_subj_top10"].append(
                        c_m["full_vocab_subject_macro_top10"] * 100.0
                    )
                    per_track_metrics[tk]["shifted_50w_top10"].append(
                        s_m["vocab50_macro_top10"] * 100.0
                    )
                    per_track_metrics[tk]["shifted_50w_top1"].append(
                        s_m["vocab50_macro_top1"] * 100.0
                    )
                    per_track_metrics[tk]["shifted_fv_top10"].append(
                        s_m["full_vocab_macro_top10"] * 100.0
                    )
                    per_track_metrics[tk]["shifted_fv_subj_top10"].append(
                        s_m["full_vocab_subject_macro_top10"] * 100.0
                    )
                elif clean_fallback_row is not None:
                    if stage_m in ("revision", "prefix_revision") and h_delay is not None:
                        h_dict = (clean_fallback_row.get("horizons") or {}).get(f"K{h_delay}") or {}
                        c50_10 = float(h_dict.get("common_vocab50_macro_top10", 0.0))
                        c50_1 = float(h_dict.get("common_vocab50_macro_top1", 0.0))
                        cfv_10 = float(
                            h_dict.get(
                                "common_full_vocab_word_macro_top10",
                                h_dict.get("common_full_vocab_macro_top10", 0.0),
                            )
                        )
                        cfv_s10 = float(
                            h_dict.get("common_full_vocab_subject_macro_top10", cfv_10)
                        )
                    else:
                        c50_10 = float(clean_fallback_row.get("vocab50_macro_top10", 0.0))
                        c50_1 = float(clean_fallback_row.get("vocab50_macro_top1", 0.0))
                        cfv_10 = float(
                            clean_fallback_row.get(
                                "full_vocab_word_macro_top10",
                                clean_fallback_row.get("full_vocab_macro_top10", 0.0),
                            )
                        )
                        cfv_s10 = float(
                            clean_fallback_row.get("full_vocab_subject_macro_top10", cfv_10)
                        )
                    per_track_metrics[tk]["clean_50w_top10"].append(c50_10 * 100.0)
                    per_track_metrics[tk]["clean_50w_top1"].append(c50_1 * 100.0)
                    per_track_metrics[tk]["clean_fv_top10"].append(cfv_10 * 100.0)
                    per_track_metrics[tk]["clean_fv_subj_top10"].append(cfv_s10 * 100.0)

                for align_name, enc_src in (
                    ("clean", clean_enc_for_seed),
                    ("shifted", shifted_enc_for_seed),
                ):
                    if enc_src is None:
                        continue
                    preds_a = _align_encoded_predictions_to_event_ids(
                        enc_src,
                        ordered_common_eids,
                        horizon_delay=h_delay,
                    )
                    if preds_a is None:
                        continue
                    score_a = (preds_a @ vocab_emb_norm.T).astype(np.float32)
                    acc_k = (tk, align_name)
                    if acc_k not in per_track_align_scores:
                        per_track_align_scores[acc_k] = score_a.copy()
                    else:
                        per_track_align_scores[acc_k] += score_a
                    per_track_align_counts[acc_k] = (
                        per_track_align_counts.get(acc_k, 0) + 1
                    )
                    if len(dur_idx) >= 2 and train_words_fit and train_durs_fit:
                        js_seed = compute_duration_prior_alignment_metrics(
                            score_a[dur_idx],
                            vocabulary,
                            [common_words[int(i)] for i in dur_idx],
                            common_durs_s[dur_idx],
                            train_words=train_words_fit,
                            train_durations_seconds=train_durs_fit,
                            n_permutations=0,
                        )
                        per_track_seed_js.setdefault(acc_k, []).append(
                            float(js_seed["delta_js_bits"])
                        )
                        per_track_seed_mae.setdefault(acc_k, []).append(
                            float(js_seed["top10_duration_error_reduction_ms"])
                        )
                    del score_a

        # Aggregate across seeds for each track of this model
        n_common_all = len(ordered_common_eids)
        n50_q = int(sum(1 for w in common_words if w in set(vocab50)))
        for tk, h_delay, tk_disp, is_primary_track in track_items:
            m_store = per_track_metrics[tk]
            c50_m, c50_sd = _mean_sd(m_store["clean_50w_top10"])
            s50_m, s50_sd = _mean_sd(m_store["shifted_50w_top10"])
            gaps_50 = [
                c - s
                for c, s in zip(m_store["clean_50w_top10"], m_store["shifted_50w_top10"])
            ]
            g50_m, g50_sd = _mean_sd(gaps_50)
            c50_1_m, c50_1_sd = _mean_sd(m_store["clean_50w_top1"])
            s50_1_m, s50_1_sd = _mean_sd(m_store["shifted_50w_top1"])
            cfv_m, cfv_sd = _mean_sd(m_store["clean_fv_top10"])
            sfv_m, sfv_sd = _mean_sd(m_store["shifted_fv_top10"])

            if is_primary_track:
                l_row: dict[str, Any] = {
                    "model_id": model_id,
                    "condition_key": tk,
                    "paper_display_name": tk_disp,
                    "text_embedding_model": "t5-large",
                    "split": split,
                    "seed_count": len(seed_list),
                    "common_query_count": n_common_all,
                    "n50_query_count": n50_q,
                    "clean_50w_top10_pct": round(c50_m, 4),
                    "clean_50w_top10_sd_pct": round(c50_sd, 4),
                    "shifted_50w_top10_pct": round(s50_m, 4),
                    "shifted_50w_top10_sd_pct": round(s50_sd, 4),
                    "net_gain_50w_top10_pp": round(g50_m, 4),
                    "net_gain_50w_top10_sd_pp": round(g50_sd, 4),
                    "clean_50w_top1_pct": round(c50_1_m, 4),
                    "clean_50w_top1_sd_pct": round(c50_1_sd, 4),
                    "shifted_50w_top1_pct": round(s50_1_m, 4),
                    "shifted_50w_top1_sd_pct": round(s50_1_sd, 4),
                    "clean_full_vocab_top10_pct": round(cfv_m, 4),
                    "clean_full_vocab_top10_sd_pct": round(cfv_sd, 4),
                    "shifted_full_vocab_top10_pct": round(sfv_m, 4),
                    "shifted_full_vocab_top10_sd_pct": round(sfv_sd, 4),
                }
                for cond in synthetic_conditions:
                    sm10, ss10 = _mean_sd(m_store[f"synth_{cond}_50w_top10"])
                    sm1, ss1 = _mean_sd(m_store[f"synth_{cond}_50w_top1"])
                    sfv10, sfvs10 = _mean_sd(m_store[f"synth_{cond}_fv_top10"])
                    l_row[f"synthetic_{cond}_50w_top10_pct"] = round(sm10, 4)
                    l_row[f"synthetic_{cond}_50w_top10_sd_pct"] = round(ss10, 4)
                    l_row[f"synthetic_{cond}_50w_top1_pct"] = round(sm1, 4)
                    l_row[f"synthetic_{cond}_50w_top1_sd_pct"] = round(ss1, 4)
                    l_row[f"synthetic_{cond}_full_vocab_top10_pct"] = round(sfv10, 4)
                    l_row[f"synthetic_{cond}_full_vocab_top10_sd_pct"] = round(sfvs10, 4)
                ledger_rows.append(l_row)
                model_summaries.append(l_row)

            # Tidy rows for synthetic_timing_summary.csv across Clean, Shifted, and Synthetic conditions
            for cond_tag, cond_lbl, v50_vals, v50_1_vals, fv_vals in (
                ("clean", "Real MEG (Clean 0s)", m_store["clean_50w_top10"], m_store["clean_50w_top1"], m_store["clean_fv_top10"]),
                ("shifted", "Shifted MEG", m_store["shifted_50w_top10"], m_store["shifted_50w_top1"], m_store["shifted_fv_top10"]),
                *[
                    (
                        str(sc),
                        f"Synthetic ({sc})",
                        m_store[f"synth_{sc}_50w_top10"],
                        m_store[f"synth_{sc}_50w_top1"],
                        m_store[f"synth_{sc}_fv_top10"],
                    )
                    for sc in synthetic_conditions
                ],
            ):
                if not v50_vals and not fv_vals:
                    continue
                m10, sd10 = _mean_sd(v50_vals)
                m1, sd1 = _mean_sd(v50_1_vals)
                mfv, sdfv = _mean_sd(fv_vals)
                synthetic_tidy_rows.append(
                    {
                        "dataset": str(dataset),
                        "split": split,
                        "model_id": model_id if is_primary_track else f"{model_id}_{tk.lower()}",
                        "condition_key": tk,
                        "condition": cond_tag,
                        "condition_label": f"{cond_lbl} | {tk_disp}",
                        "horizon": (int(h_delay) if h_delay is not None else -1),
                        "paper_display_name": tk_disp,
                        "text_embedding_model": "t5-large",
                        "n_seeds": len(seed_list),
                        "query_count_n50": n50_q,
                        "query_count_common": n_common_all,
                        "mean_top10_pct": round(m10, 4),
                        "sd_top10_pct": round(sd10, 4),
                        "mean_top1_pct": round(m1, 4),
                        "sd_top1_pct": round(sd1, 4),
                        "full_vocab_top10_pct": round(mfv, 4),
                        "full_vocab_top10_sd_pct": round(sdfv, 4),
                    }
                )

            # Duration-prior alignment across ("clean", "shifted", "synthetic_shared")
            for align_name in ("clean", "shifted", "synthetic_shared"):
                acc_k = (tk, align_name)
                cnt = int(per_track_align_counts.get(acc_k, 0))
                if cnt == 0 or len(dur_idx) < 2 or not (train_words_fit and train_durs_fit):
                    continue
                ens_mat = (per_track_align_scores[acc_k] / float(cnt)).astype(np.float32)
                if align_name == "clean":
                    ensemble_clean_scores[tk] = ens_mat
                js_ens = compute_duration_prior_alignment_metrics(
                    ens_mat[dur_idx],
                    vocabulary,
                    [common_words[int(i)] for i in dur_idx],
                    common_durs_s[dur_idx],
                    train_words=train_words_fit,
                    train_durations_seconds=train_durs_fit,
                    n_permutations=n_permutations,
                )
                _, js_sd = _mean_sd(per_track_seed_js.get(acc_k, []))
                _, mae_sd = _mean_sd(per_track_seed_mae.get(acc_k, []))
                if align_name == "clean":
                    a_disp = tk_disp
                elif align_name == "shifted":
                    a_disp = f"Shifted {tk_disp}"
                else:
                    a_disp = f"Synthetic (shared) {tk_disp}"
                duration_alignment_rows.append(
                    {
                        "alignment": align_name,
                        "condition": tk,
                        "paper_display_name": a_disp,
                        "text_embedding_model": "t5-large",
                        "seed_count": cnt,
                        **js_ens,
                        "delta_js_bits_sd": round(js_sd, 6),
                        "top10_duration_error_reduction_ms_sd": round(mae_sd, 4),
                    }
                )
            # Drop non-clean ensemble buffers eagerly to keep peak RAM low
            for drop_align in ("shifted", "synthetic_shared"):
                per_track_align_scores.pop((tk, drop_align), None)

    exported_files: dict[str, str] = {}
    if ledger_rows:
        p = _write_csv_rows(out_dir / "five_model_debiasing_ledger.csv", ledger_rows)
        if p is not None:
            exported_files["five_model_debiasing_ledger_csv"] = str(p)

    if synthetic_tidy_rows:
        p = _write_csv_rows(out_dir / "synthetic_timing_summary.csv", synthetic_tidy_rows)
        if p is not None:
            exported_files["synthetic_timing_csv"] = str(p)

    if duration_alignment_rows:
        raw_ps = [float(r["permutation_p_value"]) for r in duration_alignment_rows]
        for r_item, p_h in zip(duration_alignment_rows, holm_adjust(raw_ps)):
            r_item["p_holm"] = round(float(p_h), 6)
        p = _write_csv_rows(out_dir / "duration_prior_alignment.csv", duration_alignment_rows)
        if p is not None:
            exported_files["duration_prior_alignment_csv"] = str(p)

    # Cross-model Top-10 French candidate neighborhoods with T5-Large cosines & word durations
    nb_examples: list[dict[str, Any]] = []
    if ensemble_clean_scores:
        nb_payload = {
            "vocabulary": vocabulary,
            "vocabulary_embeddings": vocab_emb_norm,
            "reference_words": common_words,
            "content_words": sorted(content_words_set),
            "clean_scores": ensemble_clean_scores,
            "word_mean_duration_ms": train_word_mean_ms,
        }
        cond_order = [
            c
            for c in (
                "S0",
                "F_joint_undebiased",
                "F_freeze_undebiased",
                "F_debiased",
                "K0",
                "K1",
                "K2",
            )
            if c in ensemble_clean_scores
        ]
        for extra_c in ensemble_clean_scores:
            if extra_c not in cond_order:
                cond_order.append(extra_c)
        nb_examples = export_word_top10_neighborhood_examples(
            nb_payload,
            conditions=cond_order,
            word_mean_duration_ms=train_word_mean_ms,
        )
        if nb_examples:
            nb_json_p = out_dir / "word_top10_neighborhoods.json"
            save_json(nb_json_p, {"text_embedding_model": "t5-large", "examples": nb_examples})
            exported_files["word_top10_neighborhoods_json"] = str(nb_json_p)
            flat_nb_rows = _flatten_word_top10_neighborhoods(nb_examples)
            nb_csv = _write_csv_rows(out_dir / "word_top10_neighborhoods.csv", flat_nb_rows)
            if nb_csv is not None:
                exported_files["word_top10_neighborhoods_csv"] = str(nb_csv)

    diag_result = {
        "schema_version": 1,
        "dataset": str(dataset),
        "split": str(split),
        "seeds": seed_list,
        "text_embedding_model": "t5-large",
        "common_horizon_delay": 2,
        "common_query_count": len(ordered_common_eids),
        "n50_query_count": int(sum(1 for w in common_words if w in set(vocab50))),
        "synthetic_seed": int(synthetic_seed),
        "synthetic_conditions": [str(c) for c in synthetic_conditions],
        "output_dir": str(out_dir),
        "model_summaries": model_summaries,
        "duration_prior_alignment_rows": duration_alignment_rows,
        "word_top10_neighborhood_count": len(nb_examples),
        "exported_files": exported_files,
    }
    save_json(diag_json_path, diag_result)
    exported_files["five_model_diagnostics_json"] = str(diag_json_path)
    save_json(diag_json_path, diag_result)
    return diag_result


def _physical_waits_lookup(
    event_table: pd.DataFrame | None, *, split: str
) -> dict[str, tuple[float, float]]:
    if event_table is None or len(event_table) == 0:
        return {}
    rows = compute_word_horizon_physical_waits(event_table, split=split)
    lookup: dict[str, tuple[float, float]] = {}
    for r in rows:
        lookup[str(r.get("horizon", ""))] = (
            float(r.get("mean_wait_seconds", 0.0)),
            float(r.get("median_wait_seconds", 0.0)),
        )
    return lookup


def collect_t5_long_window_comparison_rows(
    *,
    results_dir: str | Path | None = None,
    dataset: str = "pallier2025",
    split: str = "test",
    eval_subdir: str = "causal_fast_audit",
    subject_scope: str = "all-n10",
    event_table: pd.DataFrame | None = None,
    long_window_specs: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """收集 T5-Large Stage-1 孤立单窗拉宽对照（1.00/1.28/1.50/2.50s）及去偏跨词对照（Local r=0/1/2, Full-Context）。"""
    results_root_dir = (
        Path(results_dir).resolve() if results_dir is not None else results_root()
    )
    ds_results_dir = results_root_dir / str(dataset)
    if not ds_results_dir.is_dir():
        return []

    specs = (
        [dict(s) for s in long_window_specs]
        if long_window_specs is not None
        else [dict(s) for s in CANONICAL_PALLIER2025_LONG_WINDOW_SPECS]
    )
    wait_lookup = _physical_waits_lookup(event_table, split=split)
    rows_out: list[dict[str, Any]] = []

    def _summarize_stage_metrics(
        ms_json: Path,
        *,
        stage: str,
        horizon_key: str | None = None,
    ) -> dict[str, Any] | None:
        obj = json.loads(ms_json.read_text(encoding="utf-8"))
        sr_list = obj.get("seed_results") or []
        if not sr_list:
            return None
        v50_10: list[float] = []
        v50_1: list[float] = []
        fv_10: list[float] = []
        fv_s10: list[float] = []
        sh_v50_10: list[float] = []
        for sr in sr_list:
            ext = _extract_raw_control_metrics_from_seed_result(
                sr, stage=stage, horizon_key=horizon_key
            )
            v50_10.append(ext["clean"]["vocab50_macro_top10"] * 100.0)
            v50_1.append(ext["clean"]["vocab50_macro_top1"] * 100.0)
            fv_10.append(ext["clean"]["full_vocab_macro_top10"] * 100.0)
            fv_s10.append(ext["clean"]["full_vocab_subject_macro_top10"] * 100.0)
            sh_v50_10.append(ext["shifted"]["vocab50_macro_top10"] * 100.0)
        m_v50_10, sd_v50_10 = _mean_sd(v50_10)
        m_v50_1, sd_v50_1 = _mean_sd(v50_1)
        m_fv_10, sd_fv_10 = _mean_sd(fv_10)
        m_fv_s10, sd_fv_s10 = _mean_sd(fv_s10)
        m_sh50_10, sd_sh50_10 = _mean_sd(sh_v50_10)
        first_fv = ((sr_list[0].get("vocabulary_sweep") or {}).get("full_vocab") or {})
        return {
            "seeds": int(len(sr_list)),
            "candidate_count": int(first_fv.get("candidate_count", 0)),
            "unique_events": int(
                first_fv.get(
                    "common_query_count",
                    first_fv.get("query_count", 0),
                )
            ),
            "vocab50_top10_pct": round(m_v50_10, 4),
            "vocab50_top10_sd_pct": round(sd_v50_10, 4),
            "vocab50_top1_pct": round(m_v50_1, 4),
            "vocab50_top1_sd_pct": round(sd_v50_1, 4),
            "shifted_vocab50_top10_pct": round(m_sh50_10, 4),
            "shifted_vocab50_top10_sd_pct": round(sd_sh50_10, 4),
            "full_vocab_word_macro_top10_pct": round(m_fv_10, 4),
            "full_vocab_word_macro_top10_sd_pct": round(sd_fv_10, 4),
            "subject_equal_top10_pct": round(m_fv_s10, 4),
            "subject_equal_top10_sd_pct": round(sd_fv_s10, 4),
            "macro_top10_pct": round(m_fv_10, 4),
            "macro_top10_sd_pct": round(sd_fv_10, 4),
        }

    # 1. Isolated Word-Only window widths (1.00s, 1.28s, 1.50s, 2.50s)
    for sp in specs:
        exp_tag = str(sp["exp_tag"])
        ms_json = _locate_multi_seed_summary_json(
            ds_results_dir, exp_tag, eval_subdir, "word", subject_scope, split
        )
        if ms_json is None:
            continue
        parsed = _summarize_stage_metrics(ms_json, stage="word")
        if parsed is None:
            continue
        win_sec = float(sp["window_seconds"])
        rows_out.append(
            {
                "dataset": str(dataset),
                "split": split,
                "condition": str(sp["condition"]),
                "paper_display_name": str(
                    sp.get("paper_display_name")
                    or _condition_paper_display_name(str(sp["condition"]))
                ),
                "model_family": "isolated_word_only",
                "text_embedding_model": "t5-large",
                "window_seconds": win_sec,
                "mean_wait_seconds": round(win_sec, 4),
                "median_wait_seconds": round(win_sec, 4),
                "population": f"common_horizon_delay=2 ({split})",
                **parsed,
            }
        )

    # 2. Debiased Local (r = 0, 1, 2) from prefix_revision summary
    local_json = _locate_multi_seed_summary_json(
        ds_results_dir,
        "2_lr4e-4_bs256_window_1.0_siamese_debiased_lamda_0.5_history_window",
        eval_subdir,
        "prefix_revision",
        subject_scope,
        split,
    )
    if local_json is not None:
        for h_delay, cond_id in ((0, "local_r0"), (1, "local_r1"), (2, "local_r2")):
            parsed = _summarize_stage_metrics(
                local_json, stage="prefix_revision", horizon_key=f"K{h_delay}"
            )
            if parsed is None:
                continue
            mean_w, med_w = wait_lookup.get(
                f"r={h_delay}", (1.0 + 0.8 * h_delay, 1.0 + 0.8 * h_delay)
            )
            rows_out.append(
                {
                    "dataset": str(dataset),
                    "split": split,
                    "condition": cond_id,
                    "paper_display_name": _condition_paper_display_name(cond_id),
                    "model_family": "debiased_local_cross_word",
                    "text_embedding_model": "t5-large",
                    "window_seconds": 1.0,
                    "mean_wait_seconds": round(mean_w, 4),
                    "median_wait_seconds": round(med_w, 4),
                    "population": f"common_horizon_delay=2 ({split})",
                    **parsed,
                }
            )

    # 3. Debiased Full-Context
    full_ctx_json = _locate_multi_seed_summary_json(
        ds_results_dir,
        "lr4e-4_bs256_window_1.0_siamese_debiased_lamda_0.5_history_window",
        eval_subdir,
        "context",
        subject_scope,
        split,
    )
    if full_ctx_json is not None:
        parsed = _summarize_stage_metrics(full_ctx_json, stage="context")
        if parsed is not None:
            mean_w, med_w = wait_lookup.get("full", (5.33, 5.0))
            rows_out.append(
                {
                    "dataset": str(dataset),
                    "split": split,
                    "condition": "F_debiased",
                    "paper_display_name": _condition_paper_display_name("F_debiased"),
                    "model_family": "debiased_full_context",
                    "text_embedding_model": "t5-large",
                    "window_seconds": 1.0,
                    "mean_wait_seconds": round(mean_w, 4),
                    "median_wait_seconds": round(med_w, 4),
                    "population": f"common_horizon_delay=2 ({split})",
                    **parsed,
                }
            )
    return rows_out


def _materialize_five_model_exports_to_dir(
    diag_res: Mapping[str, Any], out_dir: Path
) -> dict[str, str]:
    """确保 5 模型 T5-Large 诊断产物落在目标导出目录下并返回文件路径字典。"""
    import shutil

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, str] = {}
    for key, src_val in (diag_res.get("exported_files") or {}).items():
        if not src_val:
            continue
        src_p = Path(src_val)
        if not src_p.is_file():
            continue
        dst_p = out_dir / src_p.name
        if src_p.resolve() != dst_p.resolve():
            shutil.copy2(src_p, dst_p)
        exported[str(key)] = str(dst_p)
    return exported


def export_synthetic_and_horizon_studies(
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path,
    event_table: pd.DataFrame | None = None,
    dataset: str = "pallier2025",
    split: str = "test",
    five_model_diagnostics: Mapping[str, Any] | None = None,
    results_dir: str | Path | None = None,
    collect_long_window: bool = False,
    eval_subdir: str = "causal_fast_audit",
    long_window_specs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """整理并导出 T5-Large 5 模型脉冲对照、长词窗对比与物理等待分布（零旧版 BGE-M3 回退）。"""
    del repo_root  # Never silently load legacy BGE-M3 data/results/studies/ files
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_files: dict[str, str] = {}
    summary: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "text_embedding_model": "t5-large",
        "output_dir": str(out_dir),
    }

    # 1. Materialize 5-model T5-Large synthetic & cross-model diagnostics when explicitly provided
    if isinstance(five_model_diagnostics, Mapping):
        mat_files = _materialize_five_model_exports_to_dir(
            five_model_diagnostics, out_dir
        )
        exported_files.update(mat_files)
        summary["five_model_diagnostics"] = {
            k: five_model_diagnostics.get(k)
            for k in (
                "dataset",
                "split",
                "seeds",
                "common_query_count",
                "n50_query_count",
                "synthetic_seed",
                "model_summaries",
                "word_top10_neighborhood_count",
            )
            if k in five_model_diagnostics
        }

    # 2. Collect T5-Large Stage-1 isolated long-window + Local r=0/1/2 + Full-Context comparison
    if collect_long_window:
        lw_rows = collect_t5_long_window_comparison_rows(
            results_dir=results_dir,
            dataset=dataset,
            split=split,
            eval_subdir=eval_subdir,
            event_table=event_table,
            long_window_specs=long_window_specs,
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
    skip_duration_and_neighborhood_exports: bool = False,
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
        if not skip_duration_and_neighborhood_exports:
            nb_examples = export_word_top10_neighborhood_examples(score_payload)
            if nb_examples:
                nb_json_p = out_dir / "word_top10_neighborhoods.json"
                save_json(nb_json_p, {"examples": nb_examples})
                exported_files["word_top10_neighborhoods_json"] = str(nb_json_p)
                flat_nb_rows = _flatten_word_top10_neighborhoods(nb_examples)
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
                    matched_d = [
                        dur_lookup.get(eid_v, float("nan")) for eid_v in payload_eids
                    ]
                    if sum(1 for x in matched_d if math.isfinite(x) and x > 0.0) >= 2:
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
                        if (
                            0 <= r_i < len(dur_buf)
                            and math.isfinite(d_v)
                            and d_v > 0.0
                        ):
                            dur_buf[r_i] = d_v
            if int(np.sum(np.isfinite(dur_buf) & (dur_buf > 0.0))) >= 2:
                durs_by_event_idx = dur_buf

        if (
            not skip_duration_and_neighborhood_exports
            and durs_by_event_idx is not None
            and len(durs_by_event_idx) == len(ref_words)
        ):
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
            valid_dur_mask = np.isfinite(durs_by_event_idx) & (durs_by_event_idx > 0.0)
            js_rows: list[dict[str, Any]] = []
            for align in ("clean", "shifted"):
                for cond_k in ("S0", "K0", "K1", "K2", "F"):
                    mat = _extract_payload_score_matrix(
                        score_payload, cond_k, alignment=align
                    )
                    if mat is None or mat.shape[0] != len(ref_words):
                        continue
                    row_mask = valid_dur_mask.copy()
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
        word_embedding_map: dict[str, np.ndarray] | None = None
        if isinstance(score_payload, Mapping):
            v_words = [str(w) for w in (score_payload.get("vocabulary") or ())]
            v_embs = score_payload.get("vocabulary_embeddings")
            if v_embs is not None and len(v_words) == len(v_embs):
                emb_arr = (
                    v_embs.detach().cpu().numpy()
                    if isinstance(v_embs, torch.Tensor)
                    else np.asarray(v_embs, dtype=np.float32)
                )
                word_embedding_map = {
                    w: emb_arr[idx] for idx, w in enumerate(v_words)
                }

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
                recs,
                language=language,
                word_embedding_map=word_embedding_map,
                condition=str(ck),
                compute_paired_stats=False,
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
    results_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    eval_subdir: str = "causal_fast_audit",
    seeds: Sequence[int] = (0, 1, 2),
    run_five_model_diagnostics: bool = False,
    five_model_diagnostics_json: str | Path | None = None,
    collect_long_window_results: bool = False,
    force_diagnostics: bool = False,
) -> dict[str, Any]:
    """统一入口：从可用解码包、神经得分缓存、事件表与已存档研究结果一键生成论文导出产物。"""
    root = Path(repo_root).resolve() if repo_root is not None else PROJECT_ROOT
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    language = "fr" if str(dataset) == "pallier2025" else "zh"

    _aux_event_columns = {
        "onset_seconds": "开始时间",
        "offset_seconds": "结束时间",
        "in_recording_index": "记录内序号",
        "pos": "词性",
        "is_content_word": "是否内容词",
    }
    event_table: pd.DataFrame | None = None
    candidate_ev_paths: list[Path] = []
    if event_table_csv is not None and Path(event_table_csv).is_file():
        candidate_ev_paths.append(Path(event_table_csv))
    else:
        try:
            candidate_ev_paths.append(
                project_path(canonical_cache_paths(dataset)["event_table"])
            )
        except ValueError:
            pass
        candidate_ev_paths.extend(
            [
                root / "data" / "cache" / "derived" / dataset / "events" / "events.csv",
                root
                / "data"
                / "cache"
                / "derived"
                / dataset
                / "events"
                / "event_table.csv",
            ]
        )
    chosen_ev_path = next((p for p in candidate_ev_paths if p.is_file()), None)
    if chosen_ev_path is not None:
        event_table = restore_legacy_columns(
            pd.read_csv(chosen_ev_path, low_memory=False), _aux_event_columns
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
            enrich_segments_from_event_table(
                score_payload["segments"], event_table, split=split
            )

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
                        bundle["final_sentences"], event_table, split=split
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

    five_model_diag: dict[str, Any] | None = None
    if five_model_diagnostics_json is not None:
        diag_path = Path(five_model_diagnostics_json).resolve()
        if not diag_path.is_file():
            raise FileNotFoundError(
                f"找不到 five_model_diagnostics_json: {diag_path}"
            )
        five_model_diag = json.loads(diag_path.read_text(encoding="utf-8"))
    elif run_five_model_diagnostics:
        five_model_diag = evaluate_five_model_t5_diagnostics(
            results_dir=results_dir,
            output_dir=out_dir,
            dataset=dataset,
            split=split,
            seeds=seeds,
            eval_subdir=eval_subdir,
            event_table=event_table,
            cache_dir=cache_dir,
            n_permutations=n_permutations,
            force=force_diagnostics,
        )

    synth_res = export_synthetic_and_horizon_studies(
        repo_root=root,
        output_dir=out_dir,
        event_table=event_table,
        dataset=dataset,
        split=split,
        five_model_diagnostics=five_model_diag,
        results_dir=results_dir,
        collect_long_window=bool(collect_long_window_results),
        eval_subdir=eval_subdir,
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
        skip_duration_and_neighborhood_exports=(five_model_diag is not None),
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
        "five_model_diagnostics": synth_res.get("five_model_diagnostics"),
        "synthetic_and_horizon": synth_res,
        "sentence_and_coupling": sent_res,
    }
    manifest_path = out_dir / "manuscript_export_manifest.json"
    save_json(manifest_path, manifest)
    all_files["manifest_json"] = str(manifest_path)
    return manifest
