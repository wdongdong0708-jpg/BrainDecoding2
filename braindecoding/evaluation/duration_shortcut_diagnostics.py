"""面向真实词起始 Clean MEG 预测的隐式词长捷径体检套件（零重训纯推理）。

同时覆盖 SimpleB2T 图 2b（同词词长替换误差缩减）、图 2a（词准确率 vs. 词长标准差相关性
与纯词长先验基线相关性）、Clean 词长先验 JS 散度与同词快慢读偏移，以及 SimpleB2T 图 9
（真实词起始、严格零物理重叠的跨句拼接与同词序列零重叠拼接测试），彻底穿透任何
“Clean vs. Shifted 词起始条件开关”。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import Dataset

from braindecoding.data.word_window_dataset import (
    RUNTIME_CONTEXT_COLUMN,
    WordWindowDataset,
)
from braindecoding.evaluation.retrieval_metrics import mean_by_target, normalize_rows
from braindecoding.models import build_brain_embedding_model
from braindecoding.training import meg_word_task
from braindecoding.training.training_runtime import choose_device, load_checkpoint, set_seed
from braindecoding.training.word_encoder_training import (
    encode_loader,
    make_loader,
    resolve_amp_dtype,
)
from braindecoding.training.workflow_runner import (
    build_workflow_config,
    limit_rows,
    verify_cache_ready,
)

SPLICED_CONTEXT_COLUMN = "spliced_context_uid"
MAX_VALID_DURATION_MS = 5000.0
DEFAULT_MIN_NON_OVERLAP_GAP_SECONDS = 10.0


def extract_event_durations_ms(
    table: pd.DataFrame,
    *,
    group_col: str = RUNTIME_CONTEXT_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """从事件表提取每个词事件的真实声学词长（ms）与句内词起始间隔 IOI（ms）。"""
    onset_col = "开始时间" if "开始时间" in table.columns else "onset_seconds"
    offset_col = "结束时间" if "结束时间" in table.columns else "offset_seconds"
    onsets = pd.to_numeric(table[onset_col], errors="coerce").to_numpy(dtype=np.float64)
    offsets = pd.to_numeric(table[offset_col], errors="coerce").to_numpy(dtype=np.float64)
    acoustic_ms = (offsets - onsets) * 1000.0
    valid_acoustic = (
        np.isfinite(acoustic_ms)
        & (acoustic_ms > 0.0)
        & (acoustic_ms <= MAX_VALID_DURATION_MS)
    )
    acoustic_ms = np.where(valid_acoustic, acoustic_ms, np.nan)

    ioi_ms = np.full(len(table), np.nan, dtype=np.float64)
    if group_col in table.columns:
        groups = table[group_col].astype(str).to_numpy()
    elif "记录编号" in table.columns and "上下文编号" in table.columns:
        groups = (
            table["记录编号"].astype(str) + "|" + table["上下文编号"].astype(str)
        ).to_numpy()
    elif "上下文编号" in table.columns:
        groups = table["上下文编号"].astype(str).to_numpy()
    else:
        groups = np.zeros(len(table), dtype=np.int64)

    for idx in range(len(table) - 1):
        if groups[idx] == groups[idx + 1]:
            delta = (onsets[idx + 1] - onsets[idx]) * 1000.0
            if np.isfinite(delta) and 0.0 < delta <= MAX_VALID_DURATION_MS:
                ioi_ms[idx] = delta
    return acoustic_ms, ioi_ms


def extract_prev_inter_onset_interval_ms(
    table: pd.DataFrame,
    *,
    group_col: str = RUNTIME_CONTEXT_COLUMN,
) -> np.ndarray:
    """从事件表提取每个词事件到前一个词的起始时间间隔 prev_ioi = t_i - t_{i-1}（ms；句首为 NaN）。"""
    onset_col = "开始时间" if "开始时间" in table.columns else "onset_seconds"
    onsets = pd.to_numeric(table[onset_col], errors="coerce").to_numpy(dtype=np.float64)
    prev_ioi_ms = np.full(len(table), np.nan, dtype=np.float64)
    if group_col in table.columns:
        groups = table[group_col].astype(str).to_numpy()
    elif "记录编号" in table.columns and "上下文编号" in table.columns:
        groups = (
            table["记录编号"].astype(str) + "|" + table["上下文编号"].astype(str)
        ).to_numpy()
    elif "上下文编号" in table.columns:
        groups = table["上下文编号"].astype(str).to_numpy()
    else:
        groups = np.zeros(len(table), dtype=np.int64)

    for idx in range(1, len(table)):
        if groups[idx] == groups[idx - 1]:
            delta = (onsets[idx] - onsets[idx - 1]) * 1000.0
            if np.isfinite(delta) and 0.0 < delta <= MAX_VALID_DURATION_MS:
                prev_ioi_ms[idx] = delta
    return prev_ioi_ms


def build_train_duration_priors(
    train_table: pd.DataFrame,
    vocabulary: Sequence[str],
    *,
    group_col: str = RUNTIME_CONTEXT_COLUMN,
) -> dict[str, dict[str, Any]]:
    """严格基于训练集统计每个候选词的声学词长、后向 IOI 与前向间隔（prev_ioi）均值和标准差。"""
    word_col = "normalized_word" if "normalized_word" in train_table.columns else "标准词"
    words = train_table[word_col].astype(str).to_numpy()
    acoustic_ms, ioi_ms = extract_event_durations_ms(train_table, group_col=group_col)
    prev_ioi_ms = extract_prev_inter_onset_interval_ms(train_table, group_col=group_col)

    priors: dict[str, dict[str, Any]] = {}
    for key, values in (
        ("acoustic_duration", acoustic_ms),
        ("inter_onset_interval", ioi_ms),
        ("prev_inter_onset_interval", prev_ioi_ms),
    ):
        valid = np.isfinite(values)
        global_mean = float(np.mean(values[valid])) if np.any(valid) else 250.0
        global_std = float(np.std(values[valid])) if np.any(valid) else 100.0
        df = pd.DataFrame({"word": words[valid], "val": values[valid]})
        grouped = df.groupby("word")["val"].agg(["mean", "std", "count"]) if len(df) else None
        mean_map: dict[str, float] = {}
        std_map: dict[str, float] = {}
        count_map: dict[str, int] = {}
        for word in vocabulary:
            w = str(word)
            if grouped is not None and w in grouped.index:
                row = grouped.loc[w]
                mean_map[w] = float(row["mean"])
                std_val = float(row["std"]) if pd.notna(row["std"]) else 0.0
                std_map[w] = std_val if np.isfinite(std_val) else 0.0
                count_map[w] = int(row["count"])
            else:
                mean_map[w] = global_mean
                std_map[w] = global_std
                count_map[w] = 0
        priors[key] = {
            "mean": mean_map,
            "std": std_map,
            "count": count_map,
            "global_mean_ms": global_mean,
            "global_std_ms": global_std,
        }
    return priors


@torch.inference_mode()
def score_embeddings_against_bank(
    query_embeddings: np.ndarray | torch.Tensor,
    query_words: Sequence[str],
    candidate_embeddings: np.ndarray | torch.Tensor,
    candidate_words: Sequence[str],
    *,
    device: torch.device | None = None,
    top_k: int = 10,
    chunk_size: int = 2048,
) -> dict[str, np.ndarray]:
    """在 GPU/CPU 上分块计算余弦相似度、平局中位秩与 Top-K 候选词下标。"""
    q_np = normalize_rows(
        query_embeddings.detach().cpu().numpy()
        if isinstance(query_embeddings, torch.Tensor)
        else np.asarray(query_embeddings, dtype=np.float32)
    )
    c_np = normalize_rows(
        candidate_embeddings.detach().cpu().numpy()
        if isinstance(candidate_embeddings, torch.Tensor)
        else np.asarray(candidate_embeddings, dtype=np.float32)
    )
    n_queries = int(len(q_np))
    n_cands = int(len(c_np))
    k_eff = max(1, min(int(top_k), n_cands))
    cand_index = {str(w): idx for idx, w in enumerate(candidate_words)}
    target_idx = np.asarray(
        [cand_index.get(str(w), -1) for w in query_words], dtype=np.int64
    )
    covered = target_idx >= 0

    ranks = np.full(n_queries, np.nan, dtype=np.float64)
    top_indices = np.empty((n_queries, k_eff), dtype=np.int32)
    if n_queries == 0 or n_cands == 0:
        return {"ranks": ranks, "top_indices": top_indices, "covered": covered}

    run_device = device if device is not None else torch.device("cpu")
    cand_t = torch.from_numpy(c_np).to(device=run_device, dtype=torch.float32)
    cand_t_T = cand_t.t().contiguous()

    for start in range(0, n_queries, int(chunk_size)):
        stop = min(start + int(chunk_size), n_queries)
        q_t = torch.from_numpy(q_np[start:stop]).to(device=run_device, dtype=torch.float32)
        sim = torch.matmul(q_t, cand_t_T)
        top_idx_t = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=True).indices
        top_indices[start:stop] = top_idx_t.cpu().numpy().astype(np.int32, copy=False)

        local_targets = target_idx[start:stop]
        local_cov = local_targets >= 0
        if np.any(local_cov):
            cov_rows = torch.as_tensor(
                np.flatnonzero(local_cov), device=run_device, dtype=torch.long
            )
            cov_cols = torch.as_tensor(
                local_targets[local_cov], device=run_device, dtype=torch.long
            )
            sim_cov = sim.index_select(0, cov_rows)
            target_scores = sim_cov[
                torch.arange(len(cov_rows), device=run_device), cov_cols
            ].unsqueeze(1)
            greater = (sim_cov > (target_scores + 1e-12)).sum(dim=1)
            equal = ((sim_cov - target_scores).abs() <= 1e-12).sum(dim=1)
            chunk_ranks = 1.0 + greater.float() + 0.5 * (equal.float() - 1.0)
            ranks[start:stop][local_cov] = chunk_ranks.cpu().numpy().astype(np.float64)

    return {"ranks": ranks, "top_indices": top_indices, "covered": covered}


def _kde_histogram_distribution(
    values_ms: np.ndarray,
    *,
    grid_max_ms: float = 1500.0,
    n_bins: int = 150,
    sigma_ms: float = 20.0,
) -> np.ndarray:
    """在 [0, grid_max_ms] 上构建高斯核平滑概率分布，用于稳定计算 JS 散度（bits）。"""
    clean = np.asarray(values_ms, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return np.full(n_bins, 1.0 / n_bins, dtype=np.float64)
    clipped = np.clip(clean, 0.0, grid_max_ms)
    edges = np.linspace(0.0, grid_max_ms, n_bins + 1)
    hist, _ = np.histogram(clipped, bins=edges)
    bin_width = grid_max_ms / n_bins
    sigma_bins = max(sigma_ms / bin_width, 1e-3)
    radius = int(math.ceil(4.0 * sigma_bins))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(hist.astype(np.float64), kernel, mode="same") + 1e-12
    return smoothed / smoothed.sum()


def jensen_shannon_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """计算两个离散概率分布的 Jensen-Shannon 散度（单位：bits，范围 [0, 1]）。"""
    p_norm = np.asarray(p, dtype=np.float64)
    q_norm = np.asarray(q, dtype=np.float64)
    p_norm = np.maximum(p_norm, 1e-15)
    q_norm = np.maximum(q_norm, 1e-15)
    p_norm /= p_norm.sum()
    q_norm /= q_norm.sum()
    m = 0.5 * (p_norm + q_norm)
    kl_pm = float(np.sum(p_norm * np.log2(p_norm / m)))
    kl_qm = float(np.sum(q_norm * np.log2(q_norm / m)))
    return max(0.0, 0.5 * (kl_pm + kl_qm))


def compute_same_word_duration_substitution(
    query_words: Sequence[str],
    actual_ms: np.ndarray,
    pred_ms: np.ndarray,
) -> dict[str, Any]:
    """SimpleB2T 图 2b：解析精确计算同词词长替换误差缩减、同词残差相关与快慢读偏移。

    对每个出现次数 n_w >= 2 的词型 w，精确求取同一词型其余 n_w - 1 次真实出现词长的
    平均绝对误差 e_sub(i)，与真实绝对误差 e_actual(i) = |pred(i) - actual(i)| 做差：
        delta_e(i) = e_sub(i) - e_actual(i)  (ms)
    纯神经解码器（即使准确率 100%）条件于词型 w 与单次发音快慢无关，理论期望严格为 0.0 ms；
    依赖重叠窗读取真实词长的模型会呈现显著正值（如 ~8–15 ms）。
    """
    words_arr = np.asarray([str(w) for w in query_words], dtype=object)
    actual_arr = np.asarray(actual_ms, dtype=np.float64)
    pred_arr = np.asarray(pred_ms, dtype=np.float64)
    valid = np.isfinite(actual_arr) & np.isfinite(pred_arr)
    words_v = words_arr[valid]
    actual_v = actual_arr[valid]
    pred_v = pred_arr[valid]

    if len(words_v) == 0:
        return {
            "eligible_queries": 0,
            "eligible_words": 0,
            "actual_mae_ms": float("nan"),
            "substituted_mae_ms": float("nan"),
            "error_reduction_micro_ms": float("nan"),
            "error_reduction_macro_ms": float("nan"),
            "paired_t_stat": float("nan"),
            "paired_t_p": float("nan"),
            "within_word_corr_r": float("nan"),
            "within_word_corr_p": float("nan"),
            "fast_vs_slow_pred_shift_ms": float("nan"),
            "fast_vs_slow_js_bits": float("nan"),
            "marginal_js_bits": float("nan"),
        }

    # 按词型分组做 O(n_w log n_w) 解析精确留一替换误差
    word_to_indices: dict[str, list[int]] = {}
    for idx, w in enumerate(words_v):
        word_to_indices.setdefault(str(w), []).append(idx)

    actual_err_list: list[np.ndarray] = []
    sub_err_list: list[np.ndarray] = []
    delta_list: list[np.ndarray] = []
    macro_deltas: list[float] = []
    actual_resid_list: list[np.ndarray] = []
    pred_resid_list: list[np.ndarray] = []
    fast_preds: list[np.ndarray] = []
    slow_preds: list[np.ndarray] = []
    eligible_words = 0

    for idx_list in word_to_indices.values():
        n_w = len(idx_list)
        if n_w < 2:
            continue
        eligible_words += 1
        idx_arr = np.asarray(idx_list, dtype=np.int64)
        y = actual_v[idx_arr]
        x = pred_v[idx_arr]
        e_act = np.abs(x - y)

        y_sorted = np.sort(y)
        prefix = np.empty(n_w + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(y_sorted, out=prefix[1:])
        total_y = prefix[n_w]

        pos = np.searchsorted(y_sorted, x, side="right")
        sum_abs_all = (pos * x - prefix[pos]) + ((total_y - prefix[pos]) - (n_w - pos) * x)
        e_sub = (sum_abs_all - e_act) / float(n_w - 1)
        delta = e_sub - e_act

        actual_err_list.append(e_act)
        sub_err_list.append(e_sub)
        delta_list.append(delta)
        macro_deltas.append(float(np.mean(delta)))
        actual_resid_list.append(y - float(np.mean(y)))
        pred_resid_list.append(x - float(np.mean(x)))

        if n_w >= 4:
            q_count = max(1, n_w // 4)
            order = np.argsort(y, kind="mergesort")
            fast_preds.append(x[order[:q_count]])
            slow_preds.append(x[order[-q_count:]])

    if not delta_list:
        return {
            "eligible_queries": 0,
            "eligible_words": 0,
            "actual_mae_ms": float("nan"),
            "substituted_mae_ms": float("nan"),
            "error_reduction_micro_ms": float("nan"),
            "error_reduction_macro_ms": float("nan"),
            "paired_t_stat": float("nan"),
            "paired_t_p": float("nan"),
            "within_word_corr_r": float("nan"),
            "within_word_corr_p": float("nan"),
            "fast_vs_slow_pred_shift_ms": float("nan"),
            "fast_vs_slow_js_bits": float("nan"),
            "marginal_js_bits": float("nan"),
        }

    all_act_err = np.concatenate(actual_err_list)
    all_sub_err = np.concatenate(sub_err_list)
    all_delta = np.concatenate(delta_list)
    all_act_resid = np.concatenate(actual_resid_list)
    all_pred_resid = np.concatenate(pred_resid_list)

    if len(all_delta) >= 2 and float(np.std(all_delta)) > 1e-12:
        t_res = stats.ttest_1samp(all_delta, 0.0)
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)
    else:
        t_stat = 0.0
        t_p = 1.0

    if (
        len(all_act_resid) >= 3
        and float(np.std(all_act_resid)) > 1e-12
        and float(np.std(all_pred_resid)) > 1e-12
    ):
        r_res = stats.pearsonr(all_act_resid, all_pred_resid)
        within_r = float(r_res.statistic)
        within_p = float(r_res.pvalue)
    else:
        within_r = 0.0
        within_p = 1.0

    if fast_preds and slow_preds:
        fast_all = np.concatenate(fast_preds)
        slow_all = np.concatenate(slow_preds)
        fast_slow_shift_ms = float(np.mean(slow_all) - np.mean(fast_all))
        fast_slow_js = jensen_shannon_divergence_bits(
            _kde_histogram_distribution(fast_all),
            _kde_histogram_distribution(slow_all),
        )
    else:
        fast_slow_shift_ms = float("nan")
        fast_slow_js = float("nan")

    marginal_js = jensen_shannon_divergence_bits(
        _kde_histogram_distribution(pred_v),
        _kde_histogram_distribution(actual_v),
    )

    return {
        "eligible_queries": int(len(all_delta)),
        "eligible_words": int(eligible_words),
        "actual_mae_ms": float(np.mean(all_act_err)),
        "substituted_mae_ms": float(np.mean(all_sub_err)),
        "error_reduction_micro_ms": float(np.mean(all_delta)),
        "error_reduction_macro_ms": float(np.mean(macro_deltas)),
        "paired_t_stat": t_stat,
        "paired_t_p": t_p,
        "within_word_corr_r": within_r,
        "within_word_corr_p": within_p,
        "fast_vs_slow_pred_shift_ms": fast_slow_shift_ms,
        "fast_vs_slow_js_bits": fast_slow_js,
        "marginal_js_bits": marginal_js,
    }


def compute_duration_oracle_n50(
    query_words: Sequence[str],
    actual_ms: np.ndarray,
    vocab50: Sequence[str],
    train_prior: Mapping[str, Any],
    *,
    min_std_ms: float = 25.0,
) -> dict[str, Any]:
    """构建纯 1 维词长高斯先验基线（零脑信号），返回 N50 总体及逐词 Top-10 准确率。"""
    vocab_list = [str(w) for w in vocab50]
    cand_to_idx = {w: idx for idx, w in enumerate(vocab_list)}
    means = np.asarray(
        [float(train_prior["mean"][w]) for w in vocab_list], dtype=np.float64
    )
    stds = np.maximum(
        np.asarray([float(train_prior["std"][w]) for w in vocab_list], dtype=np.float64),
        float(min_std_ms),
    )

    words_arr = np.asarray([str(w) for w in query_words], dtype=object)
    actual_arr = np.asarray(actual_ms, dtype=np.float64)
    target_idx = np.asarray([cand_to_idx.get(str(w), -1) for w in words_arr], dtype=np.int64)
    valid = (target_idx >= 0) & np.isfinite(actual_arr)
    if not np.any(valid):
        return {
            "query_count": 0,
            "macro_top10": float("nan"),
            "macro_top1": float("nan"),
            "per_word_top10": {},
        }

    x = actual_arr[valid][:, None]
    log_lik = -0.5 * ((x - means[None, :]) / stds[None, :]) ** 2 - np.log(stds[None, :])
    t_idx = target_idx[valid]
    t_scores = log_lik[np.arange(len(t_idx)), t_idx][:, None]
    greater = (log_lik > (t_scores + 1e-12)).sum(axis=1)
    equal = (np.abs(log_lik - t_scores) <= 1e-12).sum(axis=1)
    ranks = 1.0 + greater + 0.5 * (equal - 1.0)

    hits10 = (ranks <= 10.0).astype(np.float64)
    hits1 = (ranks <= 1.0).astype(np.float64)
    v_words = words_arr[valid]
    per_word_top10: dict[str, float] = {}
    for w in vocab_list:
        mask_w = v_words == w
        if np.any(mask_w):
            per_word_top10[w] = float(np.mean(hits10[mask_w]))

    return {
        "query_count": int(len(ranks)),
        "macro_top10": float(mean_by_target(hits10, v_words)),
        "macro_top1": float(mean_by_target(hits1, v_words)),
        "per_word_top10": per_word_top10,
    }


def compute_duration_variability_correlation(
    ranks: np.ndarray,
    query_words: Sequence[str],
    actual_ms: np.ndarray,
    candidate_words: Sequence[str],
    train_prior: Mapping[str, Any],
    *,
    oracle_per_word_top10: Mapping[str, float] | None = None,
    min_word_queries: int = 1,
) -> dict[str, Any]:
    """SimpleB2T 图 2a：计算逐词 Top-10 准确率与词长标准差（SD）及纯词长基线的相关性。"""
    words_arr = np.asarray([str(w) for w in query_words], dtype=object)
    ranks_arr = np.asarray(ranks, dtype=np.float64)
    actual_arr = np.asarray(actual_ms, dtype=np.float64)
    valid = np.isfinite(ranks_arr)
    words_v = words_arr[valid]
    hits10 = (ranks_arr[valid] <= 10.0).astype(np.float64)
    actual_v = actual_arr[valid]

    acc_list: list[float] = []
    train_sd_list: list[float] = []
    eval_sd_list: list[float] = []
    oracle_acc_list: list[float] = []
    per_word_top10: dict[str, float] = {}

    for w in candidate_words:
        w_str = str(w)
        mask_w = words_v == w_str
        count_w = int(np.sum(mask_w))
        if count_w < int(min_word_queries):
            continue
        acc_w = float(np.mean(hits10[mask_w]))
        per_word_top10[w_str] = acc_w
        train_sd = float(train_prior["std"].get(w_str, float("nan")))
        act_w = actual_v[mask_w]
        act_w = act_w[np.isfinite(act_w)]
        eval_sd = float(np.std(act_w)) if len(act_w) >= 2 else float("nan")
        if np.isfinite(train_sd) and train_sd > 0.0:
            acc_list.append(acc_w)
            train_sd_list.append(train_sd)
            eval_sd_list.append(eval_sd)
            if oracle_per_word_top10 is not None and w_str in oracle_per_word_top10:
                oracle_acc_list.append(float(oracle_per_word_top10[w_str]))

    def _safe_corr(x_vals: Sequence[float], y_vals: Sequence[float]) -> tuple[float, float, float, float]:
        x = np.asarray(x_vals, dtype=np.float64)
        y = np.asarray(y_vals, dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if len(x) < 3 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
            return float("nan"), float("nan"), float("nan"), float("nan")
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)

    tr_r, tr_p, tr_rho, tr_rho_p = _safe_corr(train_sd_list, acc_list)
    ev_r, ev_p, ev_rho, ev_rho_p = _safe_corr(eval_sd_list, acc_list)
    or_r, or_p, or_rho, or_rho_p = (
        _safe_corr(oracle_acc_list, acc_list)
        if len(oracle_acc_list) == len(acc_list) and len(oracle_acc_list) >= 3
        else (float("nan"), float("nan"), float("nan"), float("nan"))
    )

    quantile_bins: list[dict[str, float | int]] = []
    if len(train_sd_list) >= 5:
        order = np.argsort(np.asarray(train_sd_list, dtype=np.float64))
        for b_idx, chunk in enumerate(np.array_split(order, 5), start=1):
            quantile_bins.append(
                {
                    "bin": b_idx,
                    "word_count": int(len(chunk)),
                    "mean_train_dur_sd_ms": float(
                        np.mean(np.asarray(train_sd_list)[chunk])
                    ),
                    "mean_top10": float(np.mean(np.asarray(acc_list)[chunk])),
                }
            )

    return {
        "word_count": int(len(acc_list)),
        "corr_acc_vs_train_dur_sd_pearson_r": tr_r,
        "corr_acc_vs_train_dur_sd_pearson_p": tr_p,
        "corr_acc_vs_train_dur_sd_spearman_rho": tr_rho,
        "corr_acc_vs_train_dur_sd_spearman_p": tr_rho_p,
        "corr_acc_vs_eval_dur_sd_pearson_r": ev_r,
        "corr_acc_vs_eval_dur_sd_pearson_p": ev_p,
        "corr_acc_vs_duration_oracle_pearson_r": or_r,
        "corr_acc_vs_duration_oracle_pearson_p": or_p,
        "corr_acc_vs_duration_oracle_spearman_rho": or_rho,
        "corr_acc_vs_duration_oracle_spearman_p": or_rho_p,
        "dur_sd_quantile_bins": quantile_bins,
        "per_word_top10": per_word_top10,
    }


def _are_non_overlapping(
    rec_a: str,
    onset_a: float,
    rec_b: str,
    onset_b: float,
    *,
    min_gap_seconds: float,
) -> bool:
    """判断两个真实词窗是否来自不同录音或间隔不少于 min_gap_seconds（严格零重叠）。"""
    if rec_a != rec_b:
        return True
    return abs(float(onset_a) - float(onset_b)) >= float(min_gap_seconds)


class SplicedRealOnsetDataset(Dataset):
    """基于真实 Clean 词起始 MEG 窗口的只读拼接数据集包装器（零物理重叠体检）。"""

    def __init__(
        self,
        base_dataset: WordWindowDataset,
        donor_indices: Sequence[int],
        group_ids: Sequence[int],
        zero_overlap_mask: Sequence[bool],
        *,
        mode: str,
        min_gap_seconds: float,
        neighbor_radius: int = 2,
    ) -> None:
        self.base_dataset = base_dataset
        self.donor_indices = np.asarray(donor_indices, dtype=np.int64)
        self.group_ids = np.asarray(group_ids, dtype=np.int64)
        self.zero_overlap_mask = np.asarray(zero_overlap_mask, dtype=bool)
        self.mode = str(mode)
        self.min_gap_seconds = float(min_gap_seconds)
        self.neighbor_radius = int(neighbor_radius)

        self.embedding_map = base_dataset.embedding_map
        self.subject_count = base_dataset.subject_count
        self.channel_count = base_dataset.channel_count
        self.channel_positions = base_dataset.channel_positions

        spliced_table = base_dataset.table.iloc[self.donor_indices].reset_index(drop=True).copy()
        spliced_table[SPLICED_CONTEXT_COLUMN] = [
            f"{self.mode}|grp-{int(gid):06d}" for gid in self.group_ids
        ]
        self.table = spliced_table

    def __len__(self) -> int:
        return int(len(self.donor_indices))

    def __getitem__(self, index: int) -> dict[str, Any]:
        donor = int(self.donor_indices[index])
        sample = dict(self.base_dataset[donor])
        sample["sentence_index"] = torch.tensor(int(self.group_ids[index]), dtype=torch.long)
        return sample


def audit_spliced_zero_overlap(
    base_table: pd.DataFrame,
    donor_indices: np.ndarray,
    group_ids: np.ndarray,
    *,
    min_gap_seconds: float = DEFAULT_MIN_NON_OVERLAP_GAP_SECONDS,
    neighbor_radius: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """逐位置审计拼接句中相邻 ±neighbor_radius 词窗的物理重叠状态。"""
    rec_col = "recording_id" if "recording_id" in base_table.columns else "记录编号"
    onset_col = "onset_seconds" if "onset_seconds" in base_table.columns else "开始时间"
    recs = base_table[rec_col].astype(str).to_numpy()
    onsets = pd.to_numeric(base_table[onset_col], errors="coerce").to_numpy(dtype=np.float64)

    n = int(len(donor_indices))
    zero_overlap_mask = np.ones(n, dtype=bool)
    total_adj_pairs = 0
    overlapping_adj_pairs = 0

    # group_ids 按连续块组织
    start = 0
    while start < n:
        gid = group_ids[start]
        stop = start + 1
        while stop < n and group_ids[stop] == gid:
            stop += 1
        for pos in range(start, stop):
            d_i = int(donor_indices[pos])
            for offset in range(-neighbor_radius, neighbor_radius + 1):
                if offset == 0:
                    continue
                nb = pos + offset
                if start <= nb < stop:
                    d_j = int(donor_indices[nb])
                    ok = _are_non_overlapping(
                        recs[d_i],
                        onsets[d_i],
                        recs[d_j],
                        onsets[d_j],
                        min_gap_seconds=min_gap_seconds,
                    )
                    if offset == 1:
                        total_adj_pairs += 1
                        if not ok:
                            overlapping_adj_pairs += 1
                    if not ok:
                        zero_overlap_mask[pos] = False
        start = stop

    # 严格复核 zero_overlap_mask 内部相邻对零违规
    masked_violations = 0
    start = 0
    while start < n:
        gid = group_ids[start]
        stop = start + 1
        while stop < n and group_ids[stop] == gid:
            stop += 1
        for pos in range(start, stop):
            if not zero_overlap_mask[pos]:
                continue
            d_i = int(donor_indices[pos])
            for offset in range(-neighbor_radius, neighbor_radius + 1):
                if offset == 0:
                    continue
                nb = pos + offset
                if start <= nb < stop:
                    d_j = int(donor_indices[nb])
                    if not _are_non_overlapping(
                        recs[d_i],
                        onsets[d_i],
                        recs[d_j],
                        onsets[d_j],
                        min_gap_seconds=min_gap_seconds,
                    ):
                        masked_violations += 1
        start = stop

    summary = {
        "total_tokens": n,
        "zero_overlap_tokens": int(np.sum(zero_overlap_mask)),
        "zero_overlap_token_rate": float(np.mean(zero_overlap_mask)) if n > 0 else 0.0,
        "total_adjacent_pairs": int(total_adj_pairs),
        "overlapping_adjacent_pairs": int(overlapping_adj_pairs),
        "verified_violations_within_mask": int(masked_violations),
        "min_gap_seconds": float(min_gap_seconds),
        "neighbor_radius": int(neighbor_radius),
    }
    return zero_overlap_mask, summary


def build_spliced_real_onset_dataset(
    base_dataset: WordWindowDataset,
    *,
    mode: str,
    seed: int = 0,
    min_gap_seconds: float = DEFAULT_MIN_NON_OVERLAP_GAP_SECONDS,
    neighbor_radius: int = 2,
) -> tuple[SplicedRealOnsetDataset, dict[str, Any]]:
    """构造真实词起始、严格零物理重叠的拼接测试集。

    支持两种模式：
    - ``shuffled``（SimpleB2T 图 9）：保持每个受试者及句长分布不变，双射重排测试词事件，
      强制每句内相邻 ±neighbor_radius 词窗来自不同录音或相隔 >= min_gap_seconds。
    - ``same_word``（保留真实句义与语法，仅消除物理重叠）：保持原句词序与受试者完全不变，
      将每个位置替换为同受试者、同标准词且距相邻 ±neighbor_radius 词窗 >= min_gap_seconds
      （或跨录音）的另一条真实 Clean 词起始 MEG 窗口。
    """
    if mode not in {"shuffled", "same_word"}:
        raise ValueError(f"不支持的拼接模式：{mode!r}，可选 'shuffled' 或 'same_word'")

    table = base_dataset.table
    subj_col = "subject_id" if "subject_id" in table.columns else "受试者"
    rec_col = "recording_id" if "recording_id" in table.columns else "记录编号"
    word_col = "normalized_word" if "normalized_word" in table.columns else "标准词"
    onset_col = "onset_seconds" if "onset_seconds" in table.columns else "开始时间"

    subjs = table[subj_col].astype(str).to_numpy()
    recs = table[rec_col].astype(str).to_numpy()
    words = table[word_col].astype(str).to_numpy()
    onsets = pd.to_numeric(table[onset_col], errors="coerce").to_numpy(dtype=np.float64)
    sent_ids = np.asarray(base_dataset.sentence_indices, dtype=np.int64)

    # 按原句分组（保持句序与每句长度）
    orig_groups: list[tuple[str, list[int]]] = []
    n_total = len(table)
    idx_cursor = 0
    while idx_cursor < n_total:
        sid = sent_ids[idx_cursor]
        stop = idx_cursor + 1
        while stop < n_total and sent_ids[stop] == sid:
            stop += 1
        orig_groups.append((str(subjs[idx_cursor]), list(range(idx_cursor, stop))))
        idx_cursor = stop

    rng = np.random.default_rng(int(seed) + (17 if mode == "shuffled" else 73))
    donor_out: list[int] = []
    group_out: list[int] = []

    if mode == "shuffled":
        # 按受试者构造真实 Clean 词事件双射随机队列
        by_subj: dict[str, list[int]] = {}
        for row_i, subj in enumerate(subjs):
            by_subj.setdefault(str(subj), []).append(row_i)
        queues: dict[str, list[int]] = {}
        for subj, rows in by_subj.items():
            perm = rng.permutation(rows).tolist()
            queues[subj] = perm

        for gid, (subj, rows) in enumerate(orig_groups):
            q = queues[subj]
            chosen: list[int] = []
            for _ in range(len(rows)):
                recent = chosen[-neighbor_radius:] if neighbor_radius > 0 else []
                picked_pos = -1
                search_limit = min(len(q), 64)
                for cand_pos in range(search_limit):
                    cand = q[cand_pos]
                    if all(
                        _are_non_overlapping(
                            recs[cand],
                            onsets[cand],
                            recs[prev],
                            onsets[prev],
                            min_gap_seconds=min_gap_seconds,
                        )
                        for prev in recent
                    ):
                        picked_pos = cand_pos
                        break
                if picked_pos < 0:
                    # 扩大全队列扫描，尽量不遗漏非重叠候选
                    for cand_pos in range(search_limit, len(q)):
                        cand = q[cand_pos]
                        if all(
                            _are_non_overlapping(
                                recs[cand],
                                onsets[cand],
                                recs[prev],
                                onsets[prev],
                                min_gap_seconds=min_gap_seconds,
                            )
                            for prev in recent
                        ):
                            picked_pos = cand_pos
                            break
                if picked_pos < 0:
                    picked_pos = 0
                cand = q.pop(picked_pos)
                chosen.append(cand)
            donor_out.extend(chosen)
            group_out.extend([gid] * len(chosen))
    else:
        # same_word 模式：按 (subject, word) 建真实 Clean 词窗池
        pool: dict[tuple[str, str], np.ndarray] = {}
        tmp_pool: dict[tuple[str, str], list[int]] = {}
        for row_i in range(n_total):
            tmp_pool.setdefault((str(subjs[row_i]), str(words[row_i])), []).append(row_i)
        for key, rows in tmp_pool.items():
            pool[key] = np.asarray(rows, dtype=np.int64)

        for gid, (subj, rows) in enumerate(orig_groups):
            chosen = []
            for orig_idx in rows:
                w = str(words[orig_idx])
                cands = pool[(subj, w)]
                recent = chosen[-neighbor_radius:] if neighbor_radius > 0 else []
                picked: int | None = None
                if len(cands) > 1:
                    # 优先选与原位和前序均零重叠的真实同词供体（随机起始轮询，O(1) 摊还）
                    n_c = len(cands)
                    start_off = int(rng.integers(0, n_c))
                    probe_n = min(n_c, 48)
                    for step in range(probe_n):
                        cand = int(cands[(start_off + step) % n_c])
                        if not _are_non_overlapping(
                            recs[cand],
                            onsets[cand],
                            recs[orig_idx],
                            onsets[orig_idx],
                            min_gap_seconds=min_gap_seconds,
                        ):
                            continue
                        if all(
                            _are_non_overlapping(
                                recs[cand],
                                onsets[cand],
                                recs[prev],
                                onsets[prev],
                                min_gap_seconds=min_gap_seconds,
                            )
                            for prev in recent
                        ):
                            picked = cand
                            break
                    if picked is None:
                        # 退而求其次：只要求与前序已选供体零重叠且尽量异于原位
                        for step in range(probe_n):
                            cand = int(cands[(start_off + step) % n_c])
                            if cand == orig_idx:
                                continue
                            if all(
                                _are_non_overlapping(
                                    recs[cand],
                                    onsets[cand],
                                    recs[prev],
                                    onsets[prev],
                                    min_gap_seconds=min_gap_seconds,
                                )
                                for prev in recent
                            ):
                                picked = cand
                                break
                chosen.append(orig_idx if picked is None else picked)
            donor_out.extend(chosen)
            group_out.extend([gid] * len(chosen))

    donor_arr = np.asarray(donor_out, dtype=np.int64)
    group_arr = np.asarray(group_out, dtype=np.int64)
    zero_mask, audit = audit_spliced_zero_overlap(
        table,
        donor_arr,
        group_arr,
        min_gap_seconds=min_gap_seconds,
        neighbor_radius=neighbor_radius,
    )
    if audit["verified_violations_within_mask"] != 0:
        raise AssertionError(
            f"零重叠掩码审计失败：发现 {audit['verified_violations_within_mask']} 处重叠违规！"
        )

    dataset = SplicedRealOnsetDataset(
        base_dataset,
        donor_arr,
        group_arr,
        zero_mask,
        mode=mode,
        min_gap_seconds=min_gap_seconds,
        neighbor_radius=neighbor_radius,
    )
    return dataset, audit


def _summarize_retrieval_slice(
    ranks: np.ndarray,
    words: Sequence[str],
    covered: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """在指定掩码上计算平衡宏平均/微平均 Top-1/Top-10 与中位秩。"""
    sel = np.asarray(covered, dtype=bool) & np.isfinite(ranks)
    if mask is not None:
        sel = sel & np.asarray(mask, dtype=bool)
    if not np.any(sel):
        return {
            "query_count": 0,
            "macro_top10": float("nan"),
            "micro_top10": float("nan"),
            "macro_top1": float("nan"),
            "micro_top1": float("nan"),
            "median_rank": float("nan"),
        }
    r = ranks[sel]
    w = np.asarray(words, dtype=object)[sel]
    h10 = (r <= 10.0).astype(np.float64)
    h1 = (r <= 1.0).astype(np.float64)
    return {
        "query_count": int(len(r)),
        "macro_top10": float(mean_by_target(h10, w)),
        "micro_top10": float(np.mean(h10)),
        "macro_top1": float(mean_by_target(h1, w)),
        "micro_top1": float(np.mean(h1)),
        "median_rank": float(np.median(r)),
    }


def _evaluate_clean_predictions_diagnostics(
    predictions: torch.Tensor | np.ndarray,
    words: Sequence[str],
    acoustic_ms: np.ndarray,
    ioi_ms: np.ndarray,
    *,
    eval_mask: np.ndarray,
    vocab50: Sequence[str],
    vocab50_bank: np.ndarray,
    full_vocab: Sequence[str],
    full_vocab_bank: np.ndarray,
    train_priors: Mapping[str, Mapping[str, Any]],
    dur_oracle_n50: Mapping[str, Any],
    device: torch.device,
    prev_ioi_ms: np.ndarray | None = None,
) -> dict[str, Any]:
    """对一组 Clean 神经预测向量计算检索精度 + SimpleB2T 图 2a/2b + Clean JS + 前词间隔体检。"""
    mask = np.asarray(eval_mask, dtype=bool)
    preds_sub = (
        predictions[mask]
        if isinstance(predictions, np.ndarray)
        else predictions[torch.as_tensor(mask, dtype=torch.bool)]
    )
    words_sub = [str(w) for w, keep in zip(words, mask) if keep]
    dur_sub = acoustic_ms[mask]
    ioi_sub = ioi_ms[mask]
    prev_ioi_sub = (
        prev_ioi_ms[mask]
        if prev_ioi_ms is not None
        else np.full(len(words_sub), np.nan, dtype=np.float64)
    )

    res_out: dict[str, Any] = {}
    for scope_name, cand_words, cand_bank in (
        ("N50", vocab50, vocab50_bank),
        ("full_vocab", full_vocab, full_vocab_bank),
    ):
        scored = score_embeddings_against_bank(
            preds_sub,
            words_sub,
            cand_bank,
            cand_words,
            device=device,
            top_k=10,
        )
        ranks = scored["ranks"]
        top_idx = scored["top_indices"]
        covered = scored["covered"]
        retrieval = _summarize_retrieval_slice(ranks, words_sub, covered)

        # 仅在候选词表覆盖的查询上做图 2b 与图 2a
        cov_words = [w for w, c in zip(words_sub, covered) if c]
        cov_dur = dur_sub[covered]
        cov_ioi = ioi_sub[covered]
        cov_prev_ioi = prev_ioi_sub[covered]
        cov_top_idx = top_idx[covered]

        dur_prior = train_priors["acoustic_duration"]
        ioi_prior = train_priors["inter_onset_interval"]
        prev_ioi_prior = train_priors.get("prev_inter_onset_interval", ioi_prior)
        cand_dur_vec = np.asarray(
            [float(dur_prior["mean"][str(w)]) for w in cand_words], dtype=np.float64
        )
        cand_ioi_vec = np.asarray(
            [float(ioi_prior["mean"][str(w)]) for w in cand_words], dtype=np.float64
        )
        cand_prev_ioi_vec = np.asarray(
            [float(prev_ioi_prior["mean"][str(w)]) for w in cand_words],
            dtype=np.float64,
        )

        pred_top10_dur = cand_dur_vec[cov_top_idx].mean(axis=1)
        pred_top1_dur = cand_dur_vec[cov_top_idx[:, 0]]
        pred_top10_ioi = cand_ioi_vec[cov_top_idx].mean(axis=1)
        pred_top10_prev_ioi = cand_prev_ioi_vec[cov_top_idx].mean(axis=1)

        fig2b_dur_top10 = compute_same_word_duration_substitution(
            cov_words, cov_dur, pred_top10_dur
        )
        fig2b_dur_top1 = compute_same_word_duration_substitution(
            cov_words, cov_dur, pred_top1_dur
        )
        fig2b_ioi_top10 = compute_same_word_duration_substitution(
            cov_words, cov_ioi, pred_top10_ioi
        )
        fig2b_prev_ioi_top10 = compute_same_word_duration_substitution(
            cov_words, cov_prev_ioi, pred_top10_prev_ioi
        )

        fig2a = compute_duration_variability_correlation(
            ranks,
            words_sub,
            dur_sub,
            cand_words,
            dur_prior,
            oracle_per_word_top10=(
                dur_oracle_n50["per_word_top10"] if scope_name == "N50" else None
            ),
            min_word_queries=(1 if scope_name == "N50" else 10),
        )
        # 剔除冗长全词表逐词字典，仅保留 N50 逐词映射供跨模型相关性核查
        if scope_name != "N50":
            fig2a = {k: v for k, v in fig2a.items() if k != "per_word_top10"}

        res_out[scope_name] = {
            "retrieval": retrieval,
            "fig2b_duration_top10": fig2b_dur_top10,
            "fig2b_duration_top1": fig2b_dur_top1,
            "fig2b_ioi_top10": fig2b_ioi_top10,
            "fig2b_prev_ioi_top10": fig2b_prev_ioi_top10,
            "fig2a_variability": fig2a,
        }
    return res_out


def _extract_horizon_views(encoded: Mapping[str, Any]) -> list[tuple[str, torch.Tensor, np.ndarray]]:
    """统一提取单状态（word/context）或三视野（K0/K1/K2，同交集掩码）的预测张量视图。"""
    n = int(len(encoded["words"]))
    rev_preds = encoded.get("revision_predictions")
    rev_masks = encoded.get("revision_availability_masks")
    if isinstance(rev_preds, Mapping) and rev_preds and isinstance(rev_masks, Mapping):
        common_mask = np.logical_and.reduce(
            [m.detach().cpu().numpy().astype(bool) for m in rev_masks.values()]
        )
        return [
            (f"K{int(delay)}", rev_preds[delay], common_mask)
            for delay in sorted(rev_preds)
        ]
    return [("default", encoded["predictions"], np.ones(n, dtype=bool))]


def _parse_model_specs(raw_specs: Sequence[str]) -> list[tuple[str, Path]]:
    """解析 --model LABEL=CHECKPOINT_PATH 参数列表。"""
    parsed: list[tuple[str, Path]] = []
    seen_labels: set[str] = set()
    for item in raw_specs:
        if "=" not in item:
            raise ValueError(
                f"--model 参数格式必须为 LABEL=CHECKPOINT_PATH，实际收到：{item!r}"
            )
        label, path_str = item.split("=", 1)
        label = label.strip()
        ckpt_path = Path(path_str.strip())
        if not label:
            raise ValueError(f"--model 标签不能为空：{item!r}")
        if label in seen_labels:
            raise ValueError(f"--model 标签重复：{label!r}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"指定的检查点不存在：{ckpt_path}")
        seen_labels.add(label)
        parsed.append((label, ckpt_path))
    if not parsed:
        raise ValueError("至少需要通过 --model 指定 1 个检查点。")
    return parsed


def run_duration_shortcut_diagnostics(
    *,
    dataset: str,
    model_specs: Sequence[tuple[str, Path]],
    split: str = "test",
    seed: int = 0,
    eval_batch_size: int = 512,
    num_workers: int = 0,
    precision: str = "bf16-mixed",
    min_gap_seconds: float = DEFAULT_MIN_NON_OVERLAP_GAP_SECONDS,
    skip_spliced: bool = False,
    max_eval_rows: int | None = None,
    require_cache: bool = True,
    show_progress: bool = False,
) -> dict[str, Any]:
    """对一组既有检查点执行完整 Clean 隐式词长捷径体检并返回结构化报告。"""
    set_seed(seed)
    cfg, subset_spec = build_workflow_config(
        dataset,
        stage="word",
        seed=seed,
        subject="all",
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        precision=precision,
    )
    if require_cache:
        verify_cache_ready(cfg, splits=("train", split))

    selected_subjects = cfg["dataset"]["subjects"]
    train_table = meg_word_task.load_event_table(
        cfg["cache"]["event_table"],
        split="train",
        subjects=selected_subjects,
        trainable_only=True,
    )
    eval_table = meg_word_task.load_event_table(
        cfg["cache"]["event_table"],
        split=split,
        subjects=selected_subjects,
        trainable_only=True,
    )
    if max_eval_rows is not None:
        eval_table = limit_rows(eval_table, int(max_eval_rows))
    if len(eval_table) == 0:
        raise ValueError(f"数据集 {dataset} 在划分 {split!r} 上无可评估事件。")

    # 从首个检查点同步窗口配置（所有同协议检查点共享 1.0s 窗口）
    first_ckpt = load_checkpoint(model_specs[0][1], map_location="cpu")
    ckpt_ds_cfg = (
        first_ckpt.get("config", {}).get("dataset", {})
        if isinstance(first_ckpt.get("config"), dict)
        else {}
    )
    win_sec = first_ckpt.get("window_seconds", ckpt_ds_cfg.get("window_seconds"))
    win_off = first_ckpt.get(
        "window_start_offset_seconds", ckpt_ds_cfg.get("window_start_offset_seconds")
    )
    if win_sec is not None:
        cfg["dataset"]["window_seconds"] = float(win_sec)
    if win_off is not None:
        cfg["dataset"]["window_start_offset_seconds"] = float(win_off)

    # 先构建包含 runtime_context_uid 的 train / eval Dataset 与词表
    clean_eval_ds = meg_word_task.build_dataset(cfg, eval_table)
    train_word_col = (
        "normalized_word" if "normalized_word" in train_table.columns else "标准词"
    )
    full_vocab = sorted(set(train_table[train_word_col].astype(str)))
    vocab50 = list(
        meg_word_task.frozen_vocabulary(
            dataset=dataset,
            train_words=train_table[train_word_col].astype(str).tolist(),
        )
    )
    # 词向量矩阵仅加载一次
    vocab50_bank = np.stack(
        [np.asarray(clean_eval_ds.embedding_map[w], dtype=np.float32) for w in vocab50]
    )
    full_vocab_bank = np.stack(
        [
            np.asarray(clean_eval_ds.embedding_map[w], dtype=np.float32)
            for w in full_vocab
        ]
    )

    # 严格基于训练集建立词长 / IOI 先验，并在 Clean 测试集上算纯词长基线
    train_priors = build_train_duration_priors(
        train_table, full_vocab, group_col=RUNTIME_CONTEXT_COLUMN
    )
    eval_dur_ms, eval_ioi_ms = extract_event_durations_ms(
        clean_eval_ds.table, group_col=RUNTIME_CONTEXT_COLUMN
    )
    eval_prev_ioi_ms = extract_prev_inter_onset_interval_ms(
        clean_eval_ds.table, group_col=RUNTIME_CONTEXT_COLUMN
    )
    eval_words = clean_eval_ds.table["normalized_word"].astype(str).tolist()
    dur_oracle_n50 = compute_duration_oracle_n50(
        eval_words, eval_dur_ms, vocab50, train_priors["acoustic_duration"]
    )
    ioi_oracle_n50 = compute_duration_oracle_n50(
        eval_words, eval_ioi_ms, vocab50, train_priors["inter_onset_interval"]
    )
    prev_ioi_oracle_n50 = compute_duration_oracle_n50(
        eval_words, eval_prev_ioi_ms, vocab50, train_priors["prev_inter_onset_interval"]
    )

    device = choose_device(cfg["training"].get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(cfg["training"])

    tc_eval = dict(cfg["training"])
    tc_eval["eval_batch_size"] = int(eval_batch_size)
    tc_eval["num_workers"] = int(num_workers)
    tc_eval["preserve_complete_groups"] = True
    clean_loader, _ = make_loader(
        clean_eval_ds,
        tc_eval,
        shuffle=False,
        group_column=RUNTIME_CONTEXT_COLUMN,
        batch_size=int(eval_batch_size),
    )

    spliced_bundles: dict[str, tuple[SplicedRealOnsetDataset, Any, dict[str, Any]]] = {}
    if not skip_spliced:
        for splice_mode in ("same_word", "shuffled"):
            sp_ds, sp_audit = build_spliced_real_onset_dataset(
                clean_eval_ds,
                mode=splice_mode,
                seed=seed,
                min_gap_seconds=min_gap_seconds,
                neighbor_radius=2,
            )
            sp_loader, _ = make_loader(
                sp_ds,
                tc_eval,
                shuffle=False,
                group_column=SPLICED_CONTEXT_COLUMN,
                batch_size=int(eval_batch_size),
            )
            spliced_bundles[splice_mode] = (sp_ds, sp_loader, sp_audit)

    models_report: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for label, ckpt_path in model_specs:
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        ckpt_sub_ds = (
            ckpt.get("config", {}).get("dataset", {})
            if isinstance(ckpt.get("config"), dict)
            else {}
        )
        ckpt_win = ckpt.get("window_seconds", ckpt_sub_ds.get("window_seconds"))
        if ckpt_win is not None and not np.isclose(
            float(ckpt_win), float(clean_eval_ds.window_seconds), atol=1e-4
        ):
            raise ValueError(
                f"检查点 {label} 的 window_seconds={ckpt_win} 与首个检查点 "
                f"{clean_eval_ds.window_seconds} 不一致，请分开运行体检。"
            )
        model_cfg = ckpt.get("model_config") or (ckpt.get("config") or {}).get("model")
        if not isinstance(model_cfg, Mapping):
            raise ValueError(f"检查点缺少 model_config：{ckpt_path}")
        ckpt_subjects = list(ckpt.get("subject_ids") or selected_subjects)
        subject_count = max(len(ckpt_subjects), clean_eval_ds.subject_count)

        model = build_brain_embedding_model(
            clean_eval_ds.channel_count,
            clean_eval_ds.channel_positions,
            subject_count,
            model_cfg,
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()

        clean_encoded = encode_loader(
            model,
            clean_loader,
            device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            show_progress=show_progress,
            progress_desc=f"Clean [{label}]",
        )
        horizon_views = _extract_horizon_views(clean_encoded)

        spliced_encoded_by_mode: dict[str, Mapping[str, Any]] = {}
        for splice_mode, (_, sp_loader, _) in spliced_bundles.items():
            spliced_encoded_by_mode[splice_mode] = encode_loader(
                model,
                sp_loader,
                device,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=f"Splice-{splice_mode} [{label}]",
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        per_horizon_report: dict[str, Any] = {}
        for h_name, h_preds, h_common_mask in horizon_views:
            clean_diag = _evaluate_clean_predictions_diagnostics(
                h_preds,
                clean_encoded["words"],
                eval_dur_ms,
                eval_ioi_ms,
                eval_mask=h_common_mask,
                vocab50=vocab50,
                vocab50_bank=vocab50_bank,
                full_vocab=full_vocab,
                full_vocab_bank=full_vocab_bank,
                train_priors=train_priors,
                dur_oracle_n50=dur_oracle_n50,
                device=device,
                prev_ioi_ms=eval_prev_ioi_ms,
            )

            spliced_diag: dict[str, Any] = {}
            for splice_mode, (sp_ds, _, sp_audit) in spliced_bundles.items():
                sp_enc = spliced_encoded_by_mode[splice_mode]
                sp_views = dict(
                    (vn, (vp, vm)) for vn, vp, vm in _extract_horizon_views(sp_enc)
                )
                sp_preds, sp_common_mask = sp_views[h_name]
                strict_mask = sp_common_mask & sp_ds.zero_overlap_mask
                sp_words = sp_enc["words"]

                sp_n50 = score_embeddings_against_bank(
                    sp_preds[torch.as_tensor(strict_mask, dtype=torch.bool)],
                    [w for w, keep in zip(sp_words, strict_mask) if keep],
                    vocab50_bank,
                    vocab50,
                    device=device,
                    top_k=10,
                )
                sp_fv = score_embeddings_against_bank(
                    sp_preds[torch.as_tensor(strict_mask, dtype=torch.bool)],
                    [w for w, keep in zip(sp_words, strict_mask) if keep],
                    full_vocab_bank,
                    full_vocab,
                    device=device,
                    top_k=10,
                )
                sp_words_masked = [w for w, keep in zip(sp_words, strict_mask) if keep]
                spliced_diag[splice_mode] = {
                    "audit": sp_audit,
                    "strict_eval_tokens": int(np.sum(strict_mask)),
                    "N50": _summarize_retrieval_slice(
                        sp_n50["ranks"], sp_words_masked, sp_n50["covered"]
                    ),
                    "full_vocab": _summarize_retrieval_slice(
                        sp_fv["ranks"], sp_words_masked, sp_fv["covered"]
                    ),
                }

            per_horizon_report[h_name] = {
                "clean": clean_diag,
                "spliced": spliced_diag,
            }

            row_label = label if h_name == "default" else f"{label} [{h_name}]"
            n50_clean = clean_diag["N50"]["retrieval"]
            fv_clean = clean_diag["full_vocab"]["retrieval"]
            f2b_n50_dur = clean_diag["N50"]["fig2b_duration_top10"]
            f2b_n50_ioi = clean_diag["N50"]["fig2b_ioi_top10"]
            f2b_n50_prev_ioi = clean_diag["N50"]["fig2b_prev_ioi_top10"]
            f2b_fv_dur = clean_diag["full_vocab"]["fig2b_duration_top10"]
            f2b_fv_ioi = clean_diag["full_vocab"]["fig2b_ioi_top10"]
            f2b_fv_prev_ioi = clean_diag["full_vocab"]["fig2b_prev_ioi_top10"]
            f2a_n50 = clean_diag["N50"]["fig2a_variability"]
            sp_sw = spliced_diag.get("same_word", {})
            sp_sh = spliced_diag.get("shuffled", {})

            summary_rows.append(
                {
                    "model": row_label,
                    "clean_n50_mac10": n50_clean["macro_top10"],
                    "clean_n50_mac1": n50_clean["macro_top1"],
                    "clean_fv_mic10": fv_clean["micro_top10"],
                    "clean_fv_med": fv_clean["median_rank"],
                    "fig2b_n50_dur_err_red_ms": f2b_n50_dur["error_reduction_micro_ms"],
                    "fig2b_n50_ioi_err_red_ms": f2b_n50_ioi["error_reduction_micro_ms"],
                    "fig2b_n50_prev_ioi_err_red_ms": f2b_n50_prev_ioi["error_reduction_micro_ms"],
                    "fig2b_n50_prev_ioi_within_r": f2b_n50_prev_ioi["within_word_corr_r"],
                    "clean_n50_prev_ioi_fast_slow_shift_ms": f2b_n50_prev_ioi["fast_vs_slow_pred_shift_ms"],
                    "fig2b_fv_dur_err_red_ms": f2b_fv_dur["error_reduction_micro_ms"],
                    "fig2b_fv_ioi_err_red_ms": f2b_fv_ioi["error_reduction_micro_ms"],
                    "fig2b_fv_prev_ioi_err_red_ms": f2b_fv_prev_ioi["error_reduction_micro_ms"],
                    "fig2b_fv_prev_ioi_within_r": f2b_fv_prev_ioi["within_word_corr_r"],
                    "clean_fv_prev_ioi_fast_slow_shift_ms": f2b_fv_prev_ioi["fast_vs_slow_pred_shift_ms"],
                    "fig2b_n50_within_r": f2b_n50_dur["within_word_corr_r"],
                    "fig2b_fv_within_r": f2b_fv_dur["within_word_corr_r"],
                    "clean_n50_fast_slow_shift_ms": f2b_n50_dur["fast_vs_slow_pred_shift_ms"],
                    "clean_n50_fast_slow_js_bits": f2b_n50_dur["fast_vs_slow_js_bits"],
                    "clean_fv_fast_slow_shift_ms": f2b_fv_dur["fast_vs_slow_pred_shift_ms"],
                    "clean_fv_fast_slow_js_bits": f2b_fv_dur["fast_vs_slow_js_bits"],
                    "fig2a_n50_acc_vs_dursd_r": f2a_n50["corr_acc_vs_train_dur_sd_pearson_r"],
                    "fig2a_n50_acc_vs_oracle_r": f2a_n50["corr_acc_vs_duration_oracle_pearson_r"],
                    "splice_same_word_n50_mac10": (
                        sp_sw.get("N50", {}).get("macro_top10", float("nan"))
                    ),
                    "splice_same_word_fv_mic10": (
                        sp_sw.get("full_vocab", {}).get("micro_top10", float("nan"))
                    ),
                    "splice_same_word_fv_med": (
                        sp_sw.get("full_vocab", {}).get("median_rank", float("nan"))
                    ),
                    "splice_shuffled_n50_mac10": (
                        sp_sh.get("N50", {}).get("macro_top10", float("nan"))
                    ),
                    "splice_shuffled_fv_mic10": (
                        sp_sh.get("full_vocab", {}).get("micro_top10", float("nan"))
                    ),
                    "splice_shuffled_fv_med": (
                        sp_sh.get("full_vocab", {}).get("median_rank", float("nan"))
                    ),
                }
            )

        models_report[label] = {
            "checkpoint": str(ckpt_path),
            "horizons": per_horizon_report,
        }

    return {
        "dataset": str(dataset),
        "split": str(split),
        "subset_tag": subset_spec["subset_tag"],
        "seed": int(seed),
        "evaluated_events": int(len(clean_eval_ds)),
        "vocab50_size": int(len(vocab50)),
        "full_vocab_size": int(len(full_vocab)),
        "min_gap_seconds": float(min_gap_seconds),
        "duration_oracles_n50": {
            "acoustic_duration": {
                k: v for k, v in dur_oracle_n50.items() if k != "per_word_top10"
            },
            "inter_onset_interval": {
                k: v for k, v in ioi_oracle_n50.items() if k != "per_word_top10"
            },
            "prev_inter_onset_interval": {
                k: v for k, v in prev_ioi_oracle_n50.items() if k != "per_word_top10"
            },
        },
        "spliced_audits": {
            mode_k: bundle[2] for mode_k, bundle in spliced_bundles.items()
        },
        "summary_table": summary_rows,
        "models": models_report,
    }


def _fmt_pct(val: float) -> str:
    return f"{val * 100.0:6.2f}%" if np.isfinite(val) else "    —  "


def _fmt_num(val: float, fmt: str = "+6.2f") -> str:
    return format(val, fmt) if np.isfinite(val) else "   —  "


def format_diagnostics_table(report: Mapping[str, Any]) -> str:
    """将诊断报告渲染为直观的单屏终端摘要表。"""
    lines: list[str] = []
    or_dur = report["duration_oracles_n50"]["acoustic_duration"]
    or_ioi = report["duration_oracles_n50"]["inter_onset_interval"]
    or_prev = report["duration_oracles_n50"].get("prev_inter_onset_interval")
    lines.append("=" * 124)
    lines.append(
        f" Clean Duration-Shortcut Diagnostics | dataset={report['dataset']} | "
        f"split={report['split']} | events={report['evaluated_events']} | "
        f"|V|={report['full_vocab_size']}"
    )
    prev_str = (
        f" | PrevIOI Top10={or_prev['macro_top10']*100:.2f}% (Top1={or_prev['macro_top1']*100:.2f}%)"
        if isinstance(or_prev, Mapping)
        else ""
    )
    lines.append(
        f" Pure 1D Duration Oracle (N50): acoustic_dur Top10={or_dur['macro_top10']*100:.2f}% "
        f"(Top1={or_dur['macro_top1']*100:.2f}%) | IOI Top10={or_ioi['macro_top10']*100:.2f}%{prev_str}"
    )
    for sp_mode, audit in (report.get("spliced_audits") or {}).items():
        lines.append(
            f" Splice [{sp_mode:9s}]: zero-overlap tokens="
            f"{audit['zero_overlap_tokens']}/{audit['total_tokens']} "
            f"({audit['zero_overlap_token_rate']*100:.2f}%), "
            f"verified_violations={audit['verified_violations_within_mask']}"
        )
    lines.append("-" * 124)
    lines.append(" [Part A] N50 Frozen Vocabulary")
    header = (
        f"{'Model / Horizon':24s} | {'Clean N50':>9s} {'Top1':>7s} | "
        f"{'2b Dur':>7s} {'2b IOI':>7s} {'InWd r':>6s} {'F→S ms':>7s} | "
        f"{'2a SD r':>7s} {'Orcl r':>6s} | "
        f"{'SameW N50':>9s} {'Shuf N50':>8s}"
    )
    lines.append(header)
    lines.append("-" * 124)
    for row in report["summary_table"]:
        lines.append(
            f"{str(row['model'])[:24]:24s} | "
            f"{_fmt_pct(row['clean_n50_mac10']):>9s} "
            f"{_fmt_pct(row['clean_n50_mac1']):>7s} | "
            f"{_fmt_num(row['fig2b_n50_dur_err_red_ms'], '+6.2f'):>7s} "
            f"{_fmt_num(row['fig2b_n50_ioi_err_red_ms'], '+6.2f'):>7s} "
            f"{_fmt_num(row['fig2b_n50_within_r'], '+6.3f'):>6s} "
            f"{_fmt_num(row['clean_n50_fast_slow_shift_ms'], '+7.2f'):>7s} | "
            f"{_fmt_num(row['fig2a_n50_acc_vs_dursd_r'], '+7.3f'):>7s} "
            f"{_fmt_num(row['fig2a_n50_acc_vs_oracle_r'], '+6.3f'):>6s} | "
            f"{_fmt_pct(row['splice_same_word_n50_mac10']):>9s} "
            f"{_fmt_pct(row['splice_shuffled_n50_mac10']):>8s}"
        )
    lines.append("-" * 124)
    lines.append(f" [Part B] Full Vocabulary (|V|={report['full_vocab_size']})")
    header_fv = (
        f"{'Model / Horizon':24s} | {'Clean FV':>8s} {'MedR':>6s} | "
        f"{'2b Dur':>7s} {'2b IOI':>7s} {'InWd r':>6s} {'F→S ms':>7s} | "
        f"{'SameW FV':>8s} {'SW Med':>6s} | "
        f"{'Shuf FV':>8s} {'Sh Med':>6s}"
    )
    lines.append(header_fv)
    lines.append("-" * 124)
    for row in report["summary_table"]:
        lines.append(
            f"{str(row['model'])[:24]:24s} | "
            f"{_fmt_pct(row['clean_fv_mic10']):>8s} "
            f"{_fmt_num(row['clean_fv_med'], '6.0f'):>6s} | "
            f"{_fmt_num(row['fig2b_fv_dur_err_red_ms'], '+6.2f'):>7s} "
            f"{_fmt_num(row['fig2b_fv_ioi_err_red_ms'], '+6.2f'):>7s} "
            f"{_fmt_num(row['fig2b_fv_within_r'], '+6.3f'):>6s} "
            f"{_fmt_num(row['clean_fv_fast_slow_shift_ms'], '+7.2f'):>7s} | "
            f"{_fmt_pct(row['splice_same_word_fv_mic10']):>8s} "
            f"{_fmt_num(row['splice_same_word_fv_med'], '6.0f'):>6s} | "
            f"{_fmt_pct(row['splice_shuffled_fv_mic10']):>8s} "
            f"{_fmt_num(row['splice_shuffled_fv_med'], '6.0f'):>6s}"
        )
    if any("fig2b_n50_prev_ioi_err_red_ms" in r for r in report["summary_table"]):
        lines.append("-" * 124)
        lines.append(
            " [Part C] Preceding-Word Interval Check (prev_ioi = t_i - t_{i-1}; audits causal K0 backward overlap leakage)"
        )
        header_prev = (
            f"{'Model / Horizon':24s} | {'N50 Prev2b':>10s} {'PrevInWd r':>10s} {'Prev F→S':>9s} | "
            f"{'FV Prev2b':>10s} {'PrevInWd r':>10s} {'Prev F→S':>9s}"
        )
        lines.append(header_prev)
        lines.append("-" * 124)
        for row in report["summary_table"]:
            lines.append(
                f"{str(row['model'])[:24]:24s} | "
                f"{_fmt_num(row.get('fig2b_n50_prev_ioi_err_red_ms', float('nan')), '+7.2f'):>10s} "
                f"{_fmt_num(row.get('fig2b_n50_prev_ioi_within_r', float('nan')), '+7.3f'):>10s} "
                f"{_fmt_num(row.get('clean_n50_prev_ioi_fast_slow_shift_ms', float('nan')), '+8.2f'):>9s} | "
                f"{_fmt_num(row.get('fig2b_fv_prev_ioi_err_red_ms', float('nan')), '+7.2f'):>10s} "
                f"{_fmt_num(row.get('fig2b_fv_prev_ioi_within_r', float('nan')), '+7.3f'):>10s} "
                f"{_fmt_num(row.get('clean_fv_prev_ioi_fast_slow_shift_ms', float('nan')), '+8.2f'):>9s}"
            )
    lines.append("=" * 124)
    lines.append(
        "判读参考：2b Dur/IOI (ms) ≈ 0–2 ms、InWd r ≈ 0、F→S ms ≈ 0 且 Shuf N50 稳守 Word-Only 附近 => 无条件开关；"
        "若 2b Dur/IOI ≈ 8–15 ms、InWd r >> 0 且 Shuf N50 崩盘 => 实锤 Clean 词长捷径/条件开关。"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diagnose-duration-shortcut",
        description="对既有检查点执行 5 分钟 Clean 隐式词长捷径与条件开关体检（SimpleB2T 图 2a/2b/9 + Clean JS）。",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("libribrain100", "pallier2025_full", "pallier2025", "armeni2022"),
        help="目标 MEG 数据集名称",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="待体检检查点，格式 LABEL=CHECKPOINT_PATH（可重复多次，自动兼容 word/context/prefix_revision）",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="评测数据划分（默认 test）",
    )
    parser.add_argument("--seed", type=int, default=0, help="确定性拼接随机种子（默认 0）")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=512,
        help="前向推理批大小（默认 512）",
    )
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数（默认 0）")
    parser.add_argument(
        "--precision",
        choices=("bf16-mixed", "fp16-mixed", "fp16", "fp32"),
        default="bf16-mixed",
        help="推理混合精度（默认 bf16-mixed）",
    )
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=DEFAULT_MIN_NON_OVERLAP_GAP_SECONDS,
        help="零重叠拼接句中相邻词窗最小时间间隔（秒，默认 10.0）",
    )
    parser.add_argument(
        "--skip-spliced",
        action="store_true",
        help="跳过图 9 零重叠拼接推理（仅算 Clean 图 2a/2b/JS）",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="可选限制评估事件行数（仅用于本地快速冒烟）",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="可选导出完整诊断 JSON 路径",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="禁用 tqdm 进度条",
    )
    return parser


def _sanitize_json_payload(obj: Any) -> Any:
    """将 NaN/Inf 转为 None，保证导出 JSON 严格符合 RFC 8259。"""
    if isinstance(obj, dict):
        return {k: _sanitize_json_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_payload(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    return obj


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model_specs = _parse_model_specs(args.model)
    report = run_duration_shortcut_diagnostics(
        dataset=args.dataset,
        model_specs=model_specs,
        split=args.split,
        seed=int(args.seed),
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        precision=str(args.precision),
        min_gap_seconds=float(args.min_gap_seconds),
        skip_spliced=bool(args.skip_spliced),
        max_eval_rows=args.max_eval_rows,
        require_cache=True,
        show_progress=not bool(args.no_progress),
    )
    print(format_diagnostics_table(report), flush=True)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                _sanitize_json_payload(report),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved full diagnostics JSON -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
