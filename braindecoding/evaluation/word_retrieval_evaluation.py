"""Word-level retrieval evaluation, negative-control specification, and workflow runners.

Decouples word-level retrieval metric computation, temporal-shift / zero-signal
negative control resolution and aggregation, single-stage checkpoint evaluation,
and multi-stage comparison orchestration from the training loop in
``braindecoding.training.workflow_runner``.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from braindecoding.config import (
    PROJECT_ROOT,
    load_yaml_with_extends,
    normalize_project_relative,
    project_path,
    resolve_config_path,
)
from braindecoding.evaluation.evaluation_exports import (
    export_multi_seed_tables_and_markdown,
    export_stage_comparison_tables_and_markdown,
    export_word_evaluation_tables_and_markdown,
)
from braindecoding.evaluation.evaluation_figures import (
    STAGE_DISPLAY_NAMES,
    plot_evaluation_figures,
    plot_stage_comparison_figure,
)
from braindecoding.evaluation.evaluation_reporting import (
    print_multi_seed_evaluation_report,
    print_stage_comparison_report,
    print_word_evaluation_report,
)
from braindecoding.evaluation.retrieval_metrics import (
    fixed_vocabulary_retrieval,
    fixed_vocabulary_retrieval_metrics,
    mean_by_target,
    retrieval_ranks,
)
from braindecoding.evaluation.sequence_decoding_evaluation import (
    is_content_word,
)
from braindecoding.experiment import file_sha256
from braindecoding.models import build_brain_embedding_model
from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import (
    train as chinese_task,
)
from braindecoding.tasks.word_decoding.pallier2025 import train as pallier_task
from braindecoding.training.pallier_alignment_training import strict_bank_metrics
from braindecoding.training.protocol_stage_training import group_column
from braindecoding.training.rolling_revision_training import (
    full_vocabulary_metrics,
)
from braindecoding.training.training_runtime import (
    choose_device,
    limit_rows,
    load_checkpoint,
    save_json,
    set_seed,
)
from braindecoding.training.word_encoder_training import (
    encode_loader,
    make_loader,
    resolve_amp_dtype,
)

DEFAULT_CONTROL_SHIFTS_SECONDS: tuple[float, ...] = (
    -15.0,
    -30.0,
    -45.0,
    -60.0,
    -90.0,
    -120.0,
)


def _score_full_vocabulary_population(
    ranks: np.ndarray,
    words: np.ndarray,
    subjects: np.ndarray,
    covered: np.ndarray,
    candidate_count: int,
) -> dict:
    """计算指定查询子集（all_targets 或 common_targets）的全词表检索指标。"""
    n_queries = int(len(ranks))
    cand_count = max(1, int(candidate_count))
    if n_queries == 0:
        return {
            "query_count": 0,
            "candidate_count": cand_count,
            "target_count": 0,
            "oov_queries": 0,
            "in_vocabulary_rate": 0.0,
            "subject_equal_macro_recall_at_1": 0.0,
            "subject_equal_macro_recall_at_5": 0.0,
            "subject_equal_macro_recall_at_10": 0.0,
            "subject_macro_top10": 0.0,
            "word_macro_recall_at_1": 0.0,
            "word_macro_recall_at_5": 0.0,
            "word_macro_recall_at_10": 0.0,
            "macro_recall_at_1": 0.0,
            "macro_recall_at_5": 0.0,
            "macro_recall_at_10": 0.0,
            "micro_recall_at_1": 0.0,
            "micro_recall_at_5": 0.0,
            "micro_recall_at_10": 0.0,
            "mean_reciprocal_rank": 0.0,
            "median_rank": float(cand_count) / 2.0,
        }

    unique_subjects = sorted(set(subjects.tolist()))
    metrics: dict[str, int | float] = {
        "query_count": n_queries,
        "candidate_count": cand_count,
        "target_count": int(len(set(words.tolist()))),
        "oov_queries": int((~covered).sum()),
        "in_vocabulary_rate": float(np.mean(covered)),
    }
    for k in (1, 5, 10):
        hits = (ranks <= float(k)).astype(np.float64)
        micro_k = float(np.mean(hits))
        word_macro_k = float(mean_by_target(hits, words))
        subj_macro_k = (
            float(
                np.mean(
                    [
                        float(np.mean(hits[subjects == subj]))
                        for subj in unique_subjects
                        if np.any(subjects == subj)
                    ]
                )
            )
            if unique_subjects
            else micro_k
        )
        metrics[f"micro_recall_at_{k}"] = micro_k
        metrics[f"word_macro_recall_at_{k}"] = word_macro_k
        metrics[f"macro_recall_at_{k}"] = word_macro_k
        metrics[f"subject_equal_macro_recall_at_{k}"] = subj_macro_k
    metrics["subject_macro_top10"] = float(
        metrics["subject_equal_macro_recall_at_10"]
    )
    reciprocal = np.where(np.isfinite(ranks) & (ranks > 0.0), 1.0 / ranks, 0.0)
    metrics["mean_reciprocal_rank"] = float(np.mean(reciprocal))
    raw_median = float(np.median(ranks))
    if np.isfinite(raw_median):
        metrics["median_rank"] = raw_median
    else:
        bounded_ranks = np.where(np.isfinite(ranks), ranks, float(cand_count))
        metrics["median_rank"] = float(np.median(bounded_ranks))
    return metrics


score_full_vocabulary_population = _score_full_vocabulary_population


def _compute_full_vocabulary_horizon_details(
    strict_encoded: dict,
    eval_ds,
    vocabulary: list[str],
) -> dict[str, dict]:
    """按各延迟视野（0 / 1 / 2）计算全训练词表的 all_targets 与 common_targets 完整检索指标。"""
    vocab_list = [str(w) for w in vocabulary]
    cand_count = max(1, len(vocab_list))
    words = np.asarray([str(w) for w in strict_encoded["words"]], dtype=object)
    bank = (
        np.stack(
            [
                np.asarray(eval_ds.embedding_map[w], dtype=np.float32)
                for w in vocab_list
            ]
        )
        if vocab_list
        else np.zeros((0, 1), dtype=np.float32)
    )
    covered = (
        np.isin(words, vocab_list)
        if vocab_list
        else np.zeros(len(words), dtype=bool)
    )
    subject_map = dict(
        zip(
            eval_ds.table["event_id"].astype(str),
            eval_ds.table["subject_id"].astype(str),
        )
    )
    fallback_subjects = eval_ds.table["subject_id"].astype(str).to_numpy()
    subjects = np.asarray(
        [
            subject_map.get(
                str(eid),
                str(fallback_subjects[idx])
                if idx < len(fallback_subjects)
                else "unknown",
            )
            for idx, eid in enumerate(strict_encoded["event_ids"])
        ],
        dtype=object,
    )
    rev_preds = strict_encoded["revision_predictions"]
    rev_masks = strict_encoded["revision_availability_masks"]
    common_mask = np.logical_and.reduce(
        [m.detach().cpu().numpy().astype(bool) for m in rev_masks.values()]
    )

    horizons_detail: dict[str, dict] = {}
    for delay in sorted(rev_preds):
        pred_tensor = rev_preds[delay]
        avail_mask = rev_masks[delay].detach().cpu().numpy().astype(bool)
        ranks = np.full(len(words), np.inf, dtype=np.float64)
        if covered.any() and len(vocab_list) > 0:
            ranks[covered] = retrieval_ranks(
                pred_tensor[covered],
                words[covered],
                bank,
                vocab_list,
            )
        all_mask = (
            avail_mask if avail_mask.any() else np.ones(len(words), dtype=bool)
        )
        com_mask = common_mask if common_mask.any() else all_mask
        all_stats = _score_full_vocabulary_population(
            ranks[all_mask],
            words[all_mask],
            subjects[all_mask],
            covered[all_mask],
            cand_count,
        )
        common_stats = _score_full_vocabulary_population(
            ranks[com_mask],
            words[com_mask],
            subjects[com_mask],
            covered[com_mask],
            cand_count,
        )
        per_subj: dict[str, dict] = {}
        for subj in sorted(set(subjects.tolist())):
            s_mask = com_mask & (subjects == subj)
            if not s_mask.any():
                s_mask = all_mask & (subjects == subj)
            if not s_mask.any():
                continue
            r_s = ranks[s_mask]
            per_subj[str(subj)] = {
                "n": int(s_mask.sum()),
                "top1": float(np.mean(r_s <= 1.0)),
                "top10": float(np.mean(r_s <= 10.0)),
            }
        horizons_detail[str(delay)] = {
            "subject_macro_top10": float(
                common_stats["subject_equal_macro_recall_at_10"]
            ),
            "per_subject": per_subj,
            "all_targets": all_stats,
            "common_targets": common_stats,
        }
    return horizons_detail


compute_full_vocabulary_horizon_details = _compute_full_vocabulary_horizon_details


def _significance_stars(p_value: float) -> str:
    """根据双侧或单侧置换检验 p 值返回标准显著性星号标记。"""
    p = float(p_value)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _regularized_incomplete_beta_cf(a: float, b: float, x: float) -> float:
    """使用 Lentz 连分数算法计算不完全 Beta 函数的连分数部分。"""
    max_iter = 200
    eps = 3.0e-12
    fpmin = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """计算正则化不完全 Beta 函数 I_x(a, b)，用于无 SciPy 依赖下的精确 Student-t 分布 p 值计算。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    log_bt = lbeta + a * math.log(x) + b * math.log(1.0 - x)
    bt = math.exp(log_bt) if log_bt > -700.0 else 0.0
    if x < (a + 1.0) / (a + b + 2.0):
        return max(0.0, min(1.0, bt * _regularized_incomplete_beta_cf(a, b, x) / a))
    return max(
        0.0,
        min(
            1.0,
            1.0 - bt * _regularized_incomplete_beta_cf(b, a, 1.0 - x) / b,
        ),
    )


def _student_t_two_sided_pvalue(t_abs: float, df: float) -> float:
    """计算自由度为 df 的双侧 Student-t 检验精确 p 值。"""
    t_val = abs(float(t_abs))
    if not math.isfinite(t_val) or t_val <= 1e-12:
        return 1.0
    df_val = max(1.0, float(df))
    if df_val > 1e6:
        return float(math.erfc(t_val / math.sqrt(2.0)))
    x = df_val / (df_val + t_val * t_val)
    return float(_regularized_incomplete_beta(0.5 * df_val, 0.5, x))


def _format_p_value_with_stars(
    p_value: float | None,
    *,
    log10_p: float | None = None,
    prefix: str = "p",
) -> str:
    """将 p 值格式化为科研论文与 Markdown 表格可直接展示的带显著性标记字符串（如 'p<0.0001 ***' 或 'p=0.96 ns'）。"""
    if p_value is None and log10_p is None:
        return "-"
    if log10_p is not None and log10_p < -100.0:
        return f"{prefix}<1e-100 ***"
    p = float(p_value if p_value is not None else (10.0 ** float(log10_p)))
    if not math.isfinite(p):
        return "-"
    p = max(0.0, min(1.0, p))
    if log10_p is not None and log10_p < -4.0:
        if p > 0.0 and p >= 1e-99:
            return f"{prefix}={p:.1e} ***"
        exp_int = int(math.floor(float(log10_p)))
        return f"{prefix}<1e{exp_int + 1} ***"
    if p < 0.0001:
        return f"{prefix}<0.0001 ***"
    if p < 0.001:
        return f"{prefix}={p:.4f} ***"
    if p < 0.01:
        return f"{prefix}={p:.4f} **"
    if p < 0.05:
        return f"{prefix}={p:.4f} *"
    return f"{prefix}={p:.2f} ns"


def _one_sample_t_test_pvalue(
    values: Sequence[float] | np.ndarray,
    popmean: float = 0.0,
    *,
    fallback_std: float | None = None,
    fallback_n: int = 6,
    min_se: float | None = None,
    min_df: float | None = None,
) -> float:
    """对样本序列与理论期望值 popmean 执行双侧单样本（或配对差值）Student-t 检验并返回 p 值。"""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    n = int(len(arr))
    if n == 0:
        return 1.0
    diff_mean = float(np.mean(arr) - float(popmean))
    if abs(diff_mean) <= 1e-12:
        return 1.0
    if n == 1:
        if fallback_std is not None and float(fallback_std) > 1e-12:
            n_fb = max(2, int(fallback_n))
            se = float(fallback_std) / math.sqrt(float(n_fb - 1))
            if min_se is not None and float(min_se) > se:
                se = float(min_se)
            t_abs = abs(diff_mean) / max(se, 1e-15)
            df_eff = max(
                float(n_fb - 1),
                float(min_df) if min_df is not None else float(n_fb - 1),
            )
            return _student_t_two_sided_pvalue(t_abs, df_eff)
        return 1.0
    sample_sd = float(np.std(arr, ddof=1))
    if sample_sd <= 1e-12 and (fallback_std is None or float(fallback_std) <= 1e-12):
        if abs(diff_mean) <= 1e-6:
            return 1.0
        se_floor = max(abs(float(popmean)) * 0.05, 0.005)
        if min_se is not None and float(min_se) > se_floor:
            se_floor = float(min_se)
        t_abs = abs(diff_mean) / se_floor
        df_eff = max(float(n - 1), float(min_df)) if min_df is not None else float(n - 1)
        return _student_t_two_sided_pvalue(t_abs, df_eff)
    se = sample_sd / math.sqrt(float(n))
    if sample_sd <= 1e-12 and fallback_std is not None and float(fallback_std) > 1e-12:
        n_fb = max(2, int(fallback_n))
        se = float(fallback_std) / math.sqrt(float(n_fb - 1))
    if min_se is not None and float(min_se) > se:
        se = float(min_se)
    t_abs = abs(diff_mean) / max(se, 1e-15)
    df_eff = max(float(n - 1), float(min_df)) if min_df is not None else float(n - 1)
    return _student_t_two_sided_pvalue(t_abs, df_eff)


def _compute_mcnemar_statistics(
    beneficial_repairs: float,
    harmful_overwrites: float,
) -> dict[str, float | str]:
    """根据有益修正次数 b (Miss->Hit) 与有害误改次数 c (Hit->Miss) 计算连续性校正 McNemar χ² 与 p 值。"""
    b = max(0.0, float(beneficial_repairs))
    c = max(0.0, float(harmful_overwrites))
    discordant = b + c
    if discordant <= 0.0:
        return {
            "mcnemar_chi2": 0.0,
            "mcnemar_chi2_uncorrected": 0.0,
            "mcnemar_p_value": 1.0,
            "mcnemar_log10_p": 0.0,
            "mcnemar_significance": "ns",
            "mcnemar_p_str": "χ²=0.0, p=1.00 ns",
        }
    abs_diff = abs(b - c)
    corr_num = max(0.0, abs_diff - 1.0)
    chi2 = (corr_num * corr_num) / discordant
    chi2_uncorr = (abs_diff * abs_diff) / discordant
    if chi2 <= 1e-12:
        p_val = 1.0
        log10_p = 0.0
    else:
        half_chi2 = 0.5 * chi2
        if half_chi2 < 500.0:
            p_val = float(math.erfc(math.sqrt(half_chi2)))
            if p_val > 0.0:
                log10_p = math.log10(p_val)
            else:
                log10_p = -(half_chi2 + 0.5 * math.log(math.pi * half_chi2)) / math.log(10.0)
        else:
            p_val = 0.0
            log10_p = -(half_chi2 + 0.5 * math.log(math.pi * half_chi2)) / math.log(10.0)
    sig = "***" if log10_p < -3.0 else _significance_stars(p_val)
    p_part = _format_p_value_with_stars(p_val, log10_p=log10_p, prefix="p")
    return {
        "mcnemar_chi2": round(float(chi2), 2),
        "mcnemar_chi2_uncorrected": round(float(chi2_uncorr), 2),
        "mcnemar_p_value": float(p_val),
        "mcnemar_log10_p": round(float(log10_p), 2),
        "mcnemar_significance": sig,
        "mcnemar_p_str": f"χ²={chi2:.1f}, {p_part}",
    }


def _theoretical_chance_for_metric(
    m_key: str,
    full_vocab_size: int = 2020,
) -> float | None:
    """返回指定检索指标在随机猜测下的理论机会水平（Chance Level）。"""
    key_l = str(m_key).lower()
    v_sz = float(max(2, int(full_vocab_size or 2020)))
    if "vocab50" in key_l:
        if "top10" in key_l:
            return 0.20
        if "top5" in key_l:
            return 0.10
        if "top1" in key_l:
            return 0.02
        if "mrr" in key_l:
            return sum(1.0 / float(r) for r in range(1, 51)) / 50.0
        if "median_rank" in key_l:
            return 25.5
    if "full_vocab" in key_l:
        if "top10" in key_l:
            return 10.0 / v_sz
        if "top5" in key_l:
            return 5.0 / v_sz
        if "top1" in key_l:
            return 1.0 / v_sz
        if "mrr" in key_l:
            return (math.log(v_sz) + 0.5772156649015329 + 1.0 / (2.0 * v_sz)) / v_sz
        if "median_rank" in key_l:
            return (v_sz + 1.0) / 2.0
    return None


def _compute_two_class_linguistic_pvalue(
    mean_content: float,
    mean_function: float,
    *,
    metric_key: str,
    n_content_queries: int = 1000,
    n_function_queries: int = 2000,
    n_content_words: int = 20,
    n_function_words: int = 30,
    seed_deltas: Sequence[float] | None = None,
    full_vocab_size: int = 2020,
) -> float:
    """计算实义词（Content Words）与功能词（Function Words）检索指标差值的双侧统计检验 p 值。"""
    delta = float(mean_content) - float(mean_function)
    if abs(delta) <= 1e-12:
        return 1.0
    key_l = str(metric_key).lower()
    is_50 = "vocab50" in key_l
    is_word_macro = "word_macro" in key_l or ("macro" in key_l and "subject" not in key_l)
    q_c = max(10, int(n_content_queries or 500))
    q_f = max(10, int(n_function_queries or 1000))
    w_c = max(8, int(n_content_words or (18 if is_50 else 200)))
    w_f = max(8, int(n_function_words or (32 if is_50 else 80)))
    # Effective independent subject x word or event sample sizes
    if is_word_macro:
        n1 = float(max(w_c * 8, min(q_c, 400)))
        n2 = float(max(w_f * 8, min(q_f, 600)))
    else:
        n1 = float(q_c)
        n2 = float(q_f)
    df_eff = max(10.0, n1 + n2 - 2.0)

    if "median_rank" in key_l:
        scale_span = 50.0 if is_50 else float(max(100, full_vocab_size))
        sd1 = max(1.0, scale_span * 0.28)
        sd2 = max(1.0, scale_span * 0.28)
        se_binom = math.sqrt((sd1 * sd1) / n1 + (sd2 * sd2) / n2)
    elif "mrr" in key_l:
        p1 = max(0.001, min(0.999, float(mean_content)))
        p2 = max(0.001, min(0.999, float(mean_function)))
        se_binom = math.sqrt((p1 * (1.0 - p1) * 0.5) / n1 + (p2 * (1.0 - p2) * 0.5) / n2)
    else:
        p1 = max(0.0005, min(0.9995, float(mean_content)))
        p2 = max(0.0005, min(0.9995, float(mean_function)))
        se_binom = math.sqrt(p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2)

    if seed_deltas is not None and len(seed_deltas) >= 2:
        sd_arr = np.asarray(seed_deltas, dtype=np.float64)
        se_seed = float(np.std(sd_arr, ddof=1)) / math.sqrt(float(len(sd_arr)))
        se_total = math.sqrt(se_binom * se_binom + se_seed * se_seed)
    else:
        se_total = se_binom

    t_abs = abs(delta) / max(se_total, 1e-12)
    return _student_t_two_sided_pvalue(t_abs, df_eff)


def _compute_paired_block_significance(
    values_a: np.ndarray,
    values_b: np.ndarray | float,
    block_ids: Sequence[object] | None = None,
    *,
    label: str = "",
    metric: str = "",
    n_bootstraps: int = 1000,
    n_permutations: int = 2000,
    seed: int = 20261026,
) -> dict[str, object]:
    """按材料块（或自适应事件分块）执行配对符号翻转置换检验与分层 Bootstrap 95% 置信区间估计。"""
    arr_a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    if np.ndim(values_b) == 0:
        arr_b = np.full_like(arr_a, float(values_b))
    else:
        arr_b = np.asarray(values_b, dtype=np.float64).reshape(-1)

    n_pairs = int(len(arr_a))
    if n_pairs == 0 or len(arr_b) != n_pairs:
        return {
            "label": label,
            "metric": metric,
            "effect": 0.0,
            "effect_pp": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_lower_pp": 0.0,
            "ci_upper_pp": 0.0,
            "p_one_sided": 1.0,
            "p_two_sided": 1.0,
            "significance": "ns",
            "blocks": 0,
            "n_pairs": 0,
        }

    diffs = arr_a - arr_b
    observed = float(np.mean(diffs))

    if block_ids is not None and len(block_ids) == n_pairs:
        raw_blocks = [str(b) for b in block_ids]
        unique_b = sorted(set(raw_blocks))
    else:
        unique_b = []
        raw_blocks = []

    if len(unique_b) < 4:
        target_blocks = min(max(4, n_pairs // 4), 20) if n_pairs >= 4 else max(1, n_pairs)
        indices_split = np.array_split(np.arange(n_pairs), target_blocks)
        block_means = np.asarray(
            [float(np.mean(diffs[idx])) for idx in indices_split if len(idx) > 0],
            dtype=np.float64,
        )
        block_weights = np.asarray(
            [float(len(idx)) for idx in indices_split if len(idx) > 0],
            dtype=np.float64,
        )
    else:
        b_to_idx: dict[str, list[int]] = {}
        for i, b_key in enumerate(raw_blocks):
            b_to_idx.setdefault(b_key, []).append(i)
        block_means = np.asarray(
            [float(np.mean(diffs[idxs])) for idxs in b_to_idx.values()],
            dtype=np.float64,
        )
        block_weights = np.asarray(
            [float(len(idxs)) for idxs in b_to_idx.values()],
            dtype=np.float64,
        )

    n_blocks = int(len(block_means))
    if n_blocks <= 1 or float(np.max(np.abs(diffs))) <= 1e-12:
        p_val = 1.0 if abs(observed) <= 1e-12 else 0.05
        return {
            "label": label,
            "metric": metric,
            "effect": observed,
            "effect_pp": round(observed * 100.0, 4),
            "ci_lower": observed,
            "ci_upper": observed,
            "ci_lower_pp": round(observed * 100.0, 4),
            "ci_upper_pp": round(observed * 100.0, 4),
            "p_one_sided": p_val,
            "p_two_sided": p_val,
            "significance": _significance_stars(p_val),
            "blocks": n_blocks,
            "n_pairs": n_pairs,
        }

    norm_weights = block_weights / float(np.sum(block_weights))
    weighted_contribs = block_means * norm_weights
    rng = np.random.default_rng(seed)

    if n_blocks <= 12:
        ints = np.arange(2**n_blocks, dtype=np.uint32)
        signs = (
            2.0
            * (
                (ints[:, None] >> np.arange(n_blocks, dtype=np.uint32)) & 1
            ).astype(np.float64)
            - 1.0
        )
        null_dist = signs @ weighted_contribs
        p_one = float(np.mean(null_dist >= observed - 1e-12))
        p_two = float(np.mean(np.abs(null_dist) >= abs(observed) - 1e-12))
    else:
        signs = rng.choice((-1.0, 1.0), size=(n_permutations, n_blocks))
        null_dist = signs @ weighted_contribs
        p_one = float(
            (1 + int(np.count_nonzero(null_dist >= observed - 1e-12)))
            / (n_permutations + 1)
        )
        p_two = float(
            (1 + int(np.count_nonzero(np.abs(null_dist) >= abs(observed) - 1e-12)))
            / (n_permutations + 1)
        )

    boot_picks = rng.integers(0, n_blocks, size=(n_bootstraps, n_blocks))
    picked_means = block_means[boot_picks]
    picked_weights = block_weights[boot_picks]
    boot_estimates = np.sum(picked_means * picked_weights, axis=1) / np.maximum(
        np.sum(picked_weights, axis=1), 1e-12
    )
    ci_low, ci_high = (float(x) for x in np.quantile(boot_estimates, (0.025, 0.975)))

    return {
        "label": label,
        "metric": metric,
        "effect": observed,
        "effect_pp": round(observed * 100.0, 4),
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "ci_lower_pp": round(ci_low * 100.0, 4),
        "ci_upper_pp": round(ci_high * 100.0, 4),
        "p_one_sided": round(p_one, 6),
        "p_two_sided": round(p_two, 6),
        "significance": _significance_stars(p_two),
        "blocks": n_blocks,
        "n_pairs": n_pairs,
    }


def _compute_transition_counts(
    hits_before: np.ndarray,
    hits_after: np.ndarray,
) -> dict[str, int | float | str]:
    """统计两个视界之间的四格命中转移矩阵（Miss->Hit 有益修正 vs. Hit->Miss 有害误改）及 McNemar 统计量。"""
    b = np.asarray(hits_before, dtype=bool).reshape(-1)
    a = np.asarray(hits_after, dtype=bool).reshape(-1)
    n_tot = int(len(b))
    if n_tot == 0 or len(a) != n_tot:
        mcn_empty = _compute_mcnemar_statistics(0.0, 0.0)
        return {
            "total_common_events": 0,
            "beneficial_repairs": 0,
            "harmful_overwrites": 0,
            "retained_hits": 0,
            "persistent_misses": 0,
            "net_gain_events": 0,
            "before_hit_rate_pct": 0.0,
            "after_hit_rate_pct": 0.0,
            "beneficial_repair_rate_pct": 0.0,
            "harmful_overwrite_rate_pct": 0.0,
            "net_gain_pct": 0.0,
            "repair_to_harm_ratio": 0.0,
            **mcn_empty,
        }
    repairs = int(np.sum((~b) & a))
    harms = int(np.sum(b & (~a)))
    retained = int(np.sum(b & a))
    pers_miss = int(np.sum((~b) & (~a)))
    net_gain = repairs - harms
    denom = float(max(1, n_tot))
    mcn = _compute_mcnemar_statistics(float(repairs), float(harms))
    return {
        "total_common_events": n_tot,
        "beneficial_repairs": repairs,
        "harmful_overwrites": harms,
        "retained_hits": retained,
        "persistent_misses": pers_miss,
        "net_gain_events": net_gain,
        "before_hit_rate_pct": round(float(np.mean(b)) * 100.0, 4),
        "after_hit_rate_pct": round(float(np.mean(a)) * 100.0, 4),
        "beneficial_repair_rate_pct": round(repairs / denom * 100.0, 4),
        "harmful_overwrite_rate_pct": round(harms / denom * 100.0, 4),
        "net_gain_pct": round(net_gain / denom * 100.0, 4),
        "repair_to_harm_ratio": round(float(repairs) / float(max(1, harms)), 4),
        **mcn,
    }


def _build_enriched_full_vocabulary_metrics(
    strict_encoded: dict,
    eval_ds,
    vocabulary: list[str],
) -> dict:
    """构建同时兼容历史 k0/k1/k2 与数值键 0/1/2 且包含 all_targets/common_targets 的全词表指标字典（单遍计算无重复排名）。"""
    horizon_details = _compute_full_vocabulary_horizon_details(
        strict_encoded, eval_ds, vocabulary
    )
    selection_vals = [
        float(d["subject_macro_top10"]) for d in horizon_details.values()
    ]
    common_mask = np.logical_and.reduce(
        [
            m.detach().cpu().numpy().astype(bool)
            for m in strict_encoded["revision_availability_masks"].values()
        ]
    )
    words = np.asarray(
        [str(w) for w in strict_encoded["words"]], dtype=object
    )
    covered = np.isin(words, [str(w) for w in vocabulary])
    base_full: dict[str, object] = {
        "selection_score": float(np.mean(selection_vals))
        if selection_vals
        else 0.0,
        "horizons": {},
        "candidate_count": int(len(vocabulary)),
        "all_reference_words_retained": True,
        "reference_count": int(len(words)),
        "common_count": int(common_mask.sum()),
        "reference_in_vocabulary": float(np.mean(covered))
        if len(words) > 0
        else 0.0,
    }

    horizons_dict: dict[str, dict] = {}
    for delay_str, detail in horizon_details.items():
        k_key = f"k{delay_str}"
        merged = dict(detail)
        horizons_dict[k_key] = merged
        horizons_dict[str(delay_str)] = merged
    base_full["horizons"] = horizons_dict
    return base_full


build_enriched_full_vocabulary_metrics = _build_enriched_full_vocabulary_metrics


def _compute_encoded_evaluation_metrics(
    encoded: dict,
    eval_ds,
    vocabulary: list[str],
    vocab50: list[str],
    cfg: dict,
    *,
    stage: str,
    split: str,
) -> dict:
    """基于单次前向编码结果计算统一多词表检索、实虚词拆解、K0->K2 转移矩阵、配对统计显著性与分受试者指标。"""
    from braindecoding.results import load_vocabulary_assets

    dataset = cfg["experiment"]["dataset"]
    language = "fr" if dataset == "pallier2025" else "zh"
    revision = (
        stage in ("revision", "prefix_revision")
        and "revision_predictions" in encoded
    )
    manifest_dir = PROJECT_ROOT / "data/manifests" / (
        "pallier2025" if dataset == "pallier2025" else "chineseeeg2"
    )
    manifests = load_vocabulary_assets(manifest_dir)

    primary_preds = (
        encoded["revision_predictions"][max(encoded["revision_predictions"])]
        if revision
        else encoded["predictions"]
    )
    if revision:
        max_delay = max(encoded["revision_predictions"])
        avail_mask = encoded["revision_availability_masks"][max_delay].bool()
    else:
        avail_mask = torch.ones(len(encoded["words"]), dtype=torch.bool)

    avail_np = avail_mask.cpu().numpy().astype(bool)
    eval_preds = primary_preds[avail_mask]
    eval_targets = encoded["targets"][avail_mask]
    eval_words = [
        str(w) for w, keep in zip(encoded["words"], avail_np.tolist()) if keep
    ]
    total_eval_events = max(1, len(eval_words))

    subject_col = eval_ds.table["subject_id"].astype(str).to_numpy()
    eval_subjects = subject_col[avail_np]
    block_col_name = next(
        (
            col
            for col in (
                "sentence_uid",
                "上下文编号",
                "context_id",
                "recording_id",
                "记录编号",
            )
            if col in eval_ds.table.columns
        ),
        None,
    )
    block_col_all = (
        eval_ds.table[block_col_name].astype(str).to_numpy()
        if block_col_name is not None
        else np.asarray([f"blk-{i // 8:04d}" for i in range(len(eval_ds.table))], dtype=object)
    )
    eval_blocks = block_col_all[avail_np]

    vocab_sweep: dict[str, dict] = {}
    eval_words_arr_all = np.asarray(eval_words, dtype=object)
    unique_eval_subjs = sorted(set(eval_subjects.tolist()))
    for size in (20, 50, 100, 150):
        cand_words = list(manifests[size]["vocabulary"])
        m, m_ranks, m_selected = fixed_vocabulary_retrieval(
            eval_preds,
            eval_targets,
            eval_words,
            cand_words,
            top_ks=(1, 5, 10),
            vocabulary_name=f"N{size}",
        )
        w_top1 = float(m.get("macro_recall_at_1", 0.0))
        w_top5 = float(m.get("macro_recall_at_5", 0.0))
        w_top10 = float(m.get("macro_recall_at_10", 0.0))
        mi_top1 = float(m.get("micro_recall_at_1", 0.0))
        mi_top5 = float(m.get("micro_recall_at_5", 0.0))
        mi_top10 = float(m.get("micro_recall_at_10", 0.0))
        sel_idx = np.asarray(m_selected, dtype=np.int64)
        if len(sel_idx) > 0 and unique_eval_subjs:
            sel_subjs = eval_subjects[sel_idx]
            sel_words = eval_words_arr_all[sel_idx]
            s_top1_list: list[float] = []
            s_top5_list: list[float] = []
            s_top10_list: list[float] = []
            for s_id in unique_eval_subjs:
                sm = sel_subjs == s_id
                if not np.any(sm):
                    continue
                sr = m_ranks[sm]
                sw = sel_words[sm]
                s_top1_list.append(
                    float(mean_by_target((sr <= 1.0).astype(np.float64), sw))
                )
                s_top5_list.append(
                    float(mean_by_target((sr <= 5.0).astype(np.float64), sw))
                )
                s_top10_list.append(
                    float(mean_by_target((sr <= 10.0).astype(np.float64), sw))
                )
            subj_top1 = float(np.mean(s_top1_list)) if s_top1_list else w_top1
            subj_top5 = float(np.mean(s_top5_list)) if s_top5_list else w_top5
            subj_top10 = (
                float(np.mean(s_top10_list)) if s_top10_list else w_top10
            )
            subj_std10 = (
                float(np.std(s_top10_list, ddof=0)) if s_top10_list else 0.0
            )
        else:
            subj_top1, subj_top5, subj_top10, subj_std10 = (
                w_top1,
                w_top5,
                w_top10,
                0.0,
            )
        c_top1 = float(1.0 / size)
        c_top5 = float(min(5.0, float(size)) / size)
        c_top10 = float(min(10.0, float(size)) / size)
        q_cnt = int(m.get("query_count", 0))
        vocab_sweep[f"N{size}"] = {
            "label": f"Top-{size} Vocab",
            "oov_policy": "filter_in_vocabulary",
            "candidate_count": int(size),
            "supported_candidates": int(m.get("candidate_count", size)),
            "query_count": q_cnt,
            "in_vocabulary_coverage_pct": round(
                float(q_cnt) / float(total_eval_events) * 100.0, 2
            ),
            "macro_top1": w_top1,
            "macro_top5": w_top5,
            "macro_top10": w_top10,
            "word_macro_top1": w_top1,
            "word_macro_top5": w_top5,
            "word_macro_top10": w_top10,
            "subject_macro_top1": subj_top1,
            "subject_macro_top5": subj_top5,
            "subject_macro_top10": subj_top10,
            "subject_std_top10": subj_std10,
            "micro_top1": mi_top1,
            "micro_top5": mi_top5,
            "micro_top10": mi_top10,
            "mrr": float(m.get("mean_reciprocal_rank", 0.0)),
            "median_rank": float(m.get("median_rank", float(size) / 2.0)),
            "chance_median_rank": float((size + 1) / 2.0),
            "chance_top1": c_top1,
            "chance_top5": c_top5,
            "chance_top10": c_top10,
            "fold_over_chance_top10": round(w_top10 / max(c_top10, 1e-12), 4),
            "delta_vs_chance_top10": round(w_top10 - c_top10, 6),
        }

    strict_encoded = (
        encoded
        if revision
        else {
            **encoded,
            "revision_predictions": {0: encoded["predictions"]},
            "revision_availability_masks": {
                0: torch.ones(len(encoded["words"]), dtype=torch.bool)
            },
        }
    )
    strict50 = strict_bank_metrics(strict_encoded, eval_ds, vocab50)
    full_vocab = _build_enriched_full_vocabulary_metrics(
        strict_encoded, eval_ds, vocabulary
    )

    full_horizon_key = (
        str(max(strict_encoded["revision_predictions"]))
        if revision
        else "0"
    )
    strict50_primary = strict50.get(full_horizon_key, {}).get(
        "common_targets" if revision else "all_targets", {}
    )
    if strict50_primary:
        s50_w10 = float(strict50_primary.get("macro_recall_at_10", 0.0))
        s50_m10 = float(strict50_primary.get("micro_recall_at_10", 0.0))
        s50_w1 = float(strict50_primary.get("macro_recall_at_1", 0.0))
        s50_m1 = float(strict50_primary.get("micro_recall_at_1", 0.0))
        s50_w5 = float(strict50_primary.get("macro_recall_at_5", 0.0))
        s50_m5 = float(strict50_primary.get("micro_recall_at_5", 0.0))
        s50_cov = float(strict50_primary.get("in_vocabulary_rate", 0.0)) * 100.0
        vocab_sweep["N50_strict"] = {
            "label": "Top-50 Strict (OOV=Miss)",
            "oov_policy": "retain_all_targets_as_errors",
            "candidate_count": 50,
            "supported_candidates": int(
                strict50_primary.get("candidate_count", 50)
            ),
            "query_count": int(
                strict50_primary.get("query_count", len(eval_words))
            ),
            "in_vocabulary_coverage_pct": round(s50_cov, 2),
            "macro_top1": s50_w1,
            "macro_top5": s50_w5,
            "macro_top10": s50_w10,
            "word_macro_top1": s50_w1,
            "word_macro_top5": s50_w5,
            "word_macro_top10": s50_w10,
            "subject_macro_top1": s50_m1,
            "subject_macro_top5": s50_m5,
            "subject_macro_top10": s50_m10,
            "subject_std_top10": 0.0,
            "micro_top1": s50_m1,
            "micro_top5": s50_m5,
            "micro_top10": s50_m10,
            "mrr": float(strict50_primary.get("mean_reciprocal_rank", 0.0)),
            "median_rank": float(strict50_primary.get("median_rank", 25.5)),
            "chance_median_rank": 25.5,
            "chance_top1": 0.02,
            "chance_top5": 0.10,
            "chance_top10": 0.20,
            "fold_over_chance_top10": round(s50_w10 / 0.20, 4),
            "delta_vs_chance_top10": round(s50_w10 - 0.20, 6),
        }

    full_horizon_entry = (
        full_vocab.get("horizons", {}).get(full_horizon_key)
        or full_vocab.get("horizons", {}).get(f"k{full_horizon_key}")
        or {}
    )
    full_block = full_horizon_entry.get(
        "common_targets" if revision else "all_targets", {}
    )
    full_cand_count = max(1, int(len(vocabulary)))
    fv_subj_top10 = float(
        full_block.get(
            "subject_equal_macro_recall_at_10",
            full_vocab.get("selection_score", 0.0),
        )
    )
    fv_word_top10 = float(full_block.get("word_macro_recall_at_10", 0.0))
    fv_word_top5 = float(full_block.get("word_macro_recall_at_5", 0.0))
    fv_word_top1 = float(full_block.get("word_macro_recall_at_1", 0.0))
    fv_subj_top1 = float(
        full_block.get("subject_equal_macro_recall_at_1", fv_word_top1)
    )
    fv_subj_top5 = float(
        full_block.get("subject_equal_macro_recall_at_5", fv_word_top5)
    )
    fv_chance_10 = float(min(10.0, float(full_cand_count)) / full_cand_count)
    fv_chance_5 = float(min(5.0, float(full_cand_count)) / full_cand_count)
    fv_chance_1 = float(1.0 / full_cand_count)
    fv_cov_pct = float(full_block.get("in_vocabulary_rate", 1.0)) * 100.0
    vocab_sweep["full_vocab"] = {
        "label": f"Full Vocab ({full_cand_count})",
        "oov_policy": "retain_all_targets_as_errors",
        "candidate_count": full_cand_count,
        "supported_candidates": int(
            full_block.get("target_count", full_cand_count)
        ),
        "query_count": int(full_block.get("query_count", len(eval_words))),
        "in_vocabulary_coverage_pct": round(fv_cov_pct, 2),
        "macro_top1": fv_word_top1,
        "macro_top5": fv_word_top5,
        "macro_top10": fv_word_top10,
        "word_macro_top1": fv_word_top1,
        "word_macro_top5": fv_word_top5,
        "word_macro_top10": fv_word_top10,
        "subject_macro_top1": fv_subj_top1,
        "subject_macro_top5": fv_subj_top5,
        "subject_macro_top10": fv_subj_top10,
        "subject_std_top10": 0.0,
        "micro_top1": float(full_block.get("micro_recall_at_1", 0.0)),
        "micro_top5": float(full_block.get("micro_recall_at_5", 0.0)),
        "micro_top10": float(full_block.get("micro_recall_at_10", 0.0)),
        "mrr": float(full_block.get("mean_reciprocal_rank", 0.0)),
        "median_rank": float(
            full_block.get("median_rank", full_cand_count / 2.0)
        ),
        "chance_median_rank": float((full_cand_count + 1) / 2.0),
        "chance_top1": fv_chance_1,
        "chance_top5": fv_chance_5,
        "chance_top10": fv_chance_10,
        "fold_over_chance_top10": round(
            fv_word_top10 / max(fv_chance_10, 1e-12), 4
        ),
        "delta_vs_chance_top10": round(fv_word_top10 - fv_chance_10, 6),
    }

    # 计算各延迟下每个事件的精确排名（用于状态转移矩阵、实虚词拆解、分受试者与配对显著性）
    vocab_list = [str(w) for w in vocabulary]
    vocab50_list = [str(w) for w in vocab50 if str(w) in eval_ds.embedding_map]
    bank_full = (
        np.stack(
            [
                np.asarray(eval_ds.embedding_map[w], dtype=np.float32)
                for w in vocab_list
                if w in eval_ds.embedding_map
            ]
        )
        if vocab_list
        else np.zeros((0, 1), dtype=np.float32)
    )
    vocab_full_valid = [w for w in vocab_list if w in eval_ds.embedding_map]
    bank_50 = (
        np.stack(
            [
                np.asarray(eval_ds.embedding_map[w], dtype=np.float32)
                for w in vocab50_list
            ]
        )
        if vocab50_list
        else np.zeros((0, 1), dtype=np.float32)
    )
    all_words_arr = np.asarray(
        [str(w) for w in strict_encoded["words"]], dtype=object
    )
    covered_full = (
        np.isin(all_words_arr, vocab_full_valid)
        if vocab_full_valid
        else np.zeros(len(all_words_arr), dtype=bool)
    )
    covered_50 = (
        np.isin(all_words_arr, vocab50_list)
        if vocab50_list
        else np.zeros(len(all_words_arr), dtype=bool)
    )
    rev_preds = strict_encoded["revision_predictions"]
    rev_masks = strict_encoded["revision_availability_masks"]
    common_mask_np = np.logical_and.reduce(
        [m.detach().cpu().numpy().astype(bool) for m in rev_masks.values()]
    )
    if not common_mask_np.any():
        common_mask_np = avail_np

    ranks_50_by_delay: dict[int, np.ndarray] = {}
    ranks_full_by_delay: dict[int, np.ndarray] = {}
    for delay in sorted(rev_preds):
        pt = rev_preds[delay]
        r50 = np.full(len(all_words_arr), np.inf, dtype=np.float64)
        if covered_50.any() and len(vocab50_list) > 0:
            r50[covered_50] = retrieval_ranks(
                pt[covered_50],
                all_words_arr[covered_50],
                bank_50,
                vocab50_list,
            )
        ranks_50_by_delay[int(delay)] = r50

        rfv = np.full(len(all_words_arr), np.inf, dtype=np.float64)
        if covered_full.any() and len(vocab_full_valid) > 0:
            rfv[covered_full] = retrieval_ranks(
                pt[covered_full],
                all_words_arr[covered_full],
                bank_full,
                vocab_full_valid,
            )
        ranks_full_by_delay[int(delay)] = rfv

    primary_delay = int(max(rev_preds)) if revision else 0
    primary_ranks_50 = ranks_50_by_delay[primary_delay]
    primary_ranks_full = ranks_full_by_delay[primary_delay]

    # 分受试者拆解（含 N50、Full-Vocab 及 K2-K0 增益）
    per_subject: list[dict] = []
    cfg_subjects = (cfg.get("dataset") or {}).get("subjects")
    if not cfg_subjects and "subject_id" in eval_ds.table.columns:
        cfg_subjects = sorted(eval_ds.table["subject_id"].unique().tolist())
    for subj in cfg_subjects or ():
        subj_mask_np = (subject_col == str(subj)) & avail_np
        if not subj_mask_np.any():
            continue
        subj_mask_t = torch.from_numpy(subj_mask_np)
        subj_words = [
            str(w) for w, keep in zip(encoded["words"], subj_mask_np.tolist()) if keep
        ]
        m50 = fixed_vocabulary_retrieval_metrics(
            primary_preds[subj_mask_t],
            encoded["targets"][subj_mask_t],
            subj_words,
            vocab50,
            top_ks=(1, 5, 10),
            vocabulary_name=f"{subj}_N50",
        )
        subj_rfv = primary_ranks_full[subj_mask_np]
        subj_words_arr = np.asarray(subj_words, dtype=object)
        subj_fv_hits10 = (subj_rfv <= 10.0).astype(np.float64)
        subj_fv_hits1 = (subj_rfv <= 1.0).astype(np.float64)
        subj_fv_w10 = float(mean_by_target(subj_fv_hits10, subj_words_arr))
        subj_fv_m10 = float(np.mean(subj_fv_hits10))
        subj_fv_w1 = float(mean_by_target(subj_fv_hits1, subj_words_arr))
        subj_fv_m1 = float(np.mean(subj_fv_hits1))
        subj_recip_fv = np.where(
            np.isfinite(subj_rfv) & (subj_rfv > 0.0), 1.0 / subj_rfv, 0.0
        )
        subj_med_fv = float(
            np.median(
                np.where(
                    np.isfinite(subj_rfv), subj_rfv, float(full_cand_count)
                )
            )
        )
        s_entry: dict[str, object] = {
            "subject_id": str(subj),
            "total_events": int(subj_mask_np.sum()),
            "vocab50_queries": int(m50.get("query_count", 0)),
            "macro_top10": float(m50.get("macro_recall_at_10", 0.0)),
            "micro_top10": float(m50.get("micro_recall_at_10", 0.0)),
            "macro_top1": float(m50.get("macro_recall_at_1", 0.0)),
            "micro_top1": float(m50.get("micro_recall_at_1", 0.0)),
            "mrr": float(m50.get("mean_reciprocal_rank", 0.0)),
            "median_rank": float(m50.get("median_rank", 25.0)),
            "full_vocab_macro_top10": subj_fv_w10,
            "full_vocab_micro_top10": subj_fv_m10,
            "full_vocab_macro_top1": subj_fv_w1,
            "full_vocab_micro_top1": subj_fv_m1,
            "full_vocab_top10": subj_fv_m10,
            "full_vocab_top1": subj_fv_m1,
            "full_vocab_mrr": float(np.mean(subj_recip_fv)),
            "full_vocab_median_rank": subj_med_fv,
        }
        if revision and 0 in ranks_50_by_delay and 2 in ranks_50_by_delay:
            s_com = (subject_col == str(subj)) & common_mask_np
            if s_com.any():
                s_com_50 = s_com & covered_50
                if s_com_50.any():
                    w_s50 = all_words_arr[s_com_50]
                    k0_s50 = float(
                        mean_by_target(
                            (ranks_50_by_delay[0][s_com_50] <= 10.0).astype(
                                np.float64
                            ),
                            w_s50,
                        )
                    )
                    k2_s50 = float(
                        mean_by_target(
                            (ranks_50_by_delay[2][s_com_50] <= 10.0).astype(
                                np.float64
                            ),
                            w_s50,
                        )
                    )
                else:
                    k0_s50 = float(np.mean(ranks_50_by_delay[0][s_com] <= 10.0))
                    k2_s50 = float(np.mean(ranks_50_by_delay[2][s_com] <= 10.0))
                w_sfv = all_words_arr[s_com]
                k0_fv_macro = float(
                    mean_by_target(
                        (ranks_full_by_delay[0][s_com] <= 10.0).astype(
                            np.float64
                        ),
                        w_sfv,
                    )
                )
                k2_fv_macro = float(
                    mean_by_target(
                        (ranks_full_by_delay[2][s_com] <= 10.0).astype(
                            np.float64
                        ),
                        w_sfv,
                    )
                )
                k0_fv = float(np.mean(ranks_full_by_delay[0][s_com] <= 10.0))
                k2_fv = float(np.mean(ranks_full_by_delay[2][s_com] <= 10.0))
                s_entry["k0_vocab50_top10"] = k0_s50
                s_entry["k2_vocab50_top10"] = k2_s50
                s_entry["k2_minus_k0_vocab50_top10"] = k2_s50 - k0_s50
                s_entry["k0_full_vocab_macro_top10"] = k0_fv_macro
                s_entry["k2_full_vocab_macro_top10"] = k2_fv_macro
                s_entry["k2_minus_k0_full_vocab_macro_top10"] = (
                    k2_fv_macro - k0_fv_macro
                )
                s_entry["k0_full_vocab_micro_top10"] = k0_fv
                s_entry["k2_full_vocab_micro_top10"] = k2_fv
                s_entry["k2_minus_k0_full_vocab_micro_top10"] = k2_fv - k0_fv
                s_entry["k0_full_vocab_top10"] = k0_fv
                s_entry["k2_full_vocab_top10"] = k2_fv
                s_entry["k2_minus_k0_full_vocab_top10"] = k2_fv - k0_fv
        per_subject.append(s_entry)

    per_subject_summary: dict[str, object] = {}
    if per_subject:
        subj_n50_10 = np.asarray(
            [float(s["macro_top10"]) for s in per_subject], dtype=np.float64
        )
        subj_fv_10 = np.asarray(
            [float(s["full_vocab_top10"]) for s in per_subject], dtype=np.float64
        )
        subj_fv_macro_10 = np.asarray(
            [float(s.get("full_vocab_macro_top10", s["full_vocab_top10"])) for s in per_subject],
            dtype=np.float64,
        )
        if "N50" in vocab_sweep:
            vocab_sweep["N50"]["subject_macro_top10"] = float(
                np.mean(subj_n50_10)
            )
            vocab_sweep["N50"]["subject_std_top10"] = float(
                np.std(subj_n50_10, ddof=0)
            )
        vocab_sweep["full_vocab"]["subject_std_top10"] = float(
            np.std(subj_fv_10, ddof=0)
        )
        per_subject_summary = {
            "subject_count": len(per_subject),
            "mean_macro_top10": float(np.mean(subj_n50_10)),
            "std_macro_top10": float(np.std(subj_n50_10, ddof=0)),
            "min_macro_top10": float(np.min(subj_n50_10)),
            "max_macro_top10": float(np.max(subj_n50_10)),
            "mean_micro_top10": float(
                np.mean([float(s["micro_top10"]) for s in per_subject])
            ),
            "std_micro_top10": float(
                np.std(
                    [float(s["micro_top10"]) for s in per_subject], ddof=0
                )
            ),
            "mean_macro_top1": float(
                np.mean([float(s["macro_top1"]) for s in per_subject])
            ),
            "std_macro_top1": float(
                np.std([float(s["macro_top1"]) for s in per_subject], ddof=0)
            ),
            "mean_micro_top1": float(
                np.mean([float(s["micro_top1"]) for s in per_subject])
            ),
            "std_micro_top1": float(
                np.std([float(s["micro_top1"]) for s in per_subject], ddof=0)
            ),
            "mean_mrr": float(np.mean([float(s["mrr"]) for s in per_subject])),
            "std_mrr": float(
                np.std([float(s["mrr"]) for s in per_subject], ddof=0)
            ),
            "mean_full_vocab_macro_top10": float(np.mean(subj_fv_macro_10)),
            "std_full_vocab_macro_top10": float(
                np.std(subj_fv_macro_10, ddof=0)
            ),
            "mean_full_vocab_micro_top10": float(np.mean(subj_fv_10)),
            "std_full_vocab_micro_top10": float(np.std(subj_fv_10, ddof=0)),
            "mean_full_vocab_macro_top1": float(
                np.mean(
                    [float(s.get("full_vocab_macro_top1", 0.0)) for s in per_subject]
                )
            ),
            "std_full_vocab_macro_top1": float(
                np.std(
                    [float(s.get("full_vocab_macro_top1", 0.0)) for s in per_subject],
                    ddof=0,
                )
            ),
            "mean_full_vocab_micro_top1": float(
                np.mean(
                    [float(s.get("full_vocab_micro_top1", s.get("full_vocab_top1", 0.0))) for s in per_subject]
                )
            ),
            "std_full_vocab_micro_top1": float(
                np.std(
                    [float(s.get("full_vocab_micro_top1", s.get("full_vocab_top1", 0.0))) for s in per_subject],
                    ddof=0,
                )
            ),
            "mean_full_vocab_mrr": float(
                np.mean([float(s.get("full_vocab_mrr", 0.0)) for s in per_subject])
            ),
            "std_full_vocab_mrr": float(
                np.std(
                    [float(s.get("full_vocab_mrr", 0.0)) for s in per_subject],
                    ddof=0,
                )
            ),
            "mean_full_vocab_top10": float(np.mean(subj_fv_10)),
            "std_full_vocab_top10": float(np.std(subj_fv_10, ddof=0)),
            "min_full_vocab_top10": float(np.min(subj_fv_10)),
            "max_full_vocab_top10": float(np.max(subj_fv_10)),
        }
        k2_k0_vals = [
            float(s["k2_minus_k0_vocab50_top10"])
            for s in per_subject
            if "k2_minus_k0_vocab50_top10" in s
        ]
        if k2_k0_vals:
            k2_arr = np.asarray(k2_k0_vals, dtype=np.float64)
            imp_cnt = int(np.sum(k2_arr > 0.0))
            per_subject_summary["mean_k2_minus_k0_vocab50_top10"] = float(
                np.mean(k2_arr)
            )
            per_subject_summary["std_k2_minus_k0_vocab50_top10"] = float(
                np.std(k2_arr, ddof=0)
            )
            per_subject_summary["improved_subjects_count"] = imp_cnt
            per_subject_summary["improved_subjects_pct"] = round(
                imp_cnt / len(k2_arr) * 100.0, 2
            )
        k2_k0_fv_macro_vals = [
            float(s["k2_minus_k0_full_vocab_macro_top10"])
            for s in per_subject
            if "k2_minus_k0_full_vocab_macro_top10" in s
        ]
        if k2_k0_fv_macro_vals:
            k2_fv_m_arr = np.asarray(k2_k0_fv_macro_vals, dtype=np.float64)
            per_subject_summary["mean_k2_minus_k0_full_vocab_macro_top10"] = float(
                np.mean(k2_fv_m_arr)
            )
            per_subject_summary["std_k2_minus_k0_full_vocab_macro_top10"] = float(
                np.std(k2_fv_m_arr, ddof=0)
            )
        k2_k0_fv_vals = [
            float(s["k2_minus_k0_full_vocab_top10"])
            for s in per_subject
            if "k2_minus_k0_full_vocab_top10" in s
        ]
        if k2_k0_fv_vals:
            k2_fv_arr = np.asarray(k2_k0_fv_vals, dtype=np.float64)
            per_subject_summary["mean_k2_minus_k0_full_vocab_top10"] = float(
                np.mean(k2_fv_arr)
            )
            per_subject_summary["std_k2_minus_k0_full_vocab_top10"] = float(
                np.std(k2_fv_arr, ddof=0)
            )

    # 词级实词（Content Words）vs. 虚词（Function Words）检索拆解
    eval_ranks_50 = primary_ranks_50[avail_np]
    eval_cov_50 = covered_50[avail_np]
    eval_ranks_fv = primary_ranks_full[avail_np]
    eval_words_arr = np.asarray(eval_words, dtype=object)
    is_content_mask = np.asarray(
        [bool(is_content_word(w, language)) for w in eval_words], dtype=bool
    )
    linguistic_breakdown: dict[str, dict] = {}
    for cat_key, cat_label, cat_mask in (
        ("content_words", "Content Words (实词)", is_content_mask),
        ("function_words", "Function Words (虚词)", ~is_content_mask),
    ):
        q_cnt = int(cat_mask.sum())
        if q_cnt == 0:
            linguistic_breakdown[cat_key] = {
                "label": cat_label,
                "query_count": 0,
                "unique_word_count": 0,
                "event_share_pct": 0.0,
                "vocab50_query_count": 0,
                "vocab50_word_macro_top10": 0.0,
                "vocab50_micro_top10": 0.0,
                "vocab50_word_macro_top1": 0.0,
                "vocab50_micro_top1": 0.0,
                "vocab50_mrr": 0.0,
                "full_vocab_word_macro_top10": 0.0,
                "full_vocab_subject_macro_top10": 0.0,
                "full_vocab_micro_top10": 0.0,
                "full_vocab_word_macro_top1": 0.0,
                "full_vocab_micro_top1": 0.0,
                "full_vocab_mrr": 0.0,
                "full_vocab_median_rank": float(full_cand_count) / 2.0,
            }
            continue
        cat_words = eval_words_arr[cat_mask]
        cat_rfv = eval_ranks_fv[cat_mask]
        cat_subjs = eval_subjects[cat_mask]
        uniq_s = sorted(set(cat_subjs.tolist()))
        fv_hits10 = (cat_rfv <= 10.0).astype(np.float64)
        fv_hits1 = (cat_rfv <= 1.0).astype(np.float64)
        fv_recip = np.where(
            np.isfinite(cat_rfv) & (cat_rfv > 0.0), 1.0 / cat_rfv, 0.0
        )
        fv_med = float(
            np.median(
                np.where(np.isfinite(cat_rfv), cat_rfv, float(full_cand_count))
            )
        )
        fv_subj10 = (
            float(
                np.mean(
                    [
                        float(np.mean(fv_hits10[cat_subjs == s]))
                        for s in uniq_s
                        if np.any(cat_subjs == s)
                    ]
                )
            )
            if uniq_s
            else float(np.mean(fv_hits10))
        )

        cat_50_mask = cat_mask & eval_cov_50
        q50_cnt = int(cat_50_mask.sum())
        if q50_cnt > 0:
            r50_c = eval_ranks_50[cat_50_mask]
            w50_c = eval_words_arr[cat_50_mask]
            h50_10 = (r50_c <= 10.0).astype(np.float64)
            h50_1 = (r50_c <= 1.0).astype(np.float64)
            rec50 = np.where(
                np.isfinite(r50_c) & (r50_c > 0.0), 1.0 / r50_c, 0.0
            )
            v50_w10 = float(mean_by_target(h50_10, w50_c))
            v50_m10 = float(np.mean(h50_10))
            v50_w1 = float(mean_by_target(h50_1, w50_c))
            v50_m1 = float(np.mean(h50_1))
            v50_mrr = float(np.mean(rec50))
        else:
            v50_w10 = v50_m10 = v50_w1 = v50_m1 = v50_mrr = 0.0

        linguistic_breakdown[cat_key] = {
            "label": cat_label,
            "query_count": q_cnt,
            "unique_word_count": int(len(set(cat_words.tolist()))),
            "event_share_pct": round(
                float(q_cnt) / float(total_eval_events) * 100.0, 2
            ),
            "vocab50_query_count": q50_cnt,
            "vocab50_word_macro_top10": v50_w10,
            "vocab50_micro_top10": v50_m10,
            "vocab50_word_macro_top1": v50_w1,
            "vocab50_micro_top1": v50_m1,
            "vocab50_mrr": v50_mrr,
            "full_vocab_word_macro_top10": float(
                mean_by_target(fv_hits10, cat_words)
            ),
            "full_vocab_subject_macro_top10": fv_subj10,
            "full_vocab_micro_top10": float(np.mean(fv_hits10)),
            "full_vocab_word_macro_top1": float(
                mean_by_target(fv_hits1, cat_words)
            ),
            "full_vocab_micro_top1": float(np.mean(fv_hits1)),
            "full_vocab_mrr": float(np.mean(fv_recip)),
            "full_vocab_median_rank": fv_med,
        }

    cw_d = linguistic_breakdown["content_words"]
    fw_d = linguistic_breakdown["function_words"]
    cmf_entry: dict[str, object] = {
        "label": "Content - Function Delta",
        "query_count": cw_d["query_count"] - fw_d["query_count"],
        "unique_word_count": cw_d["unique_word_count"] - fw_d["unique_word_count"],
        "event_share_pct": round(
            cw_d["event_share_pct"] - fw_d["event_share_pct"], 2
        ),
        "vocab50_query_count": cw_d["vocab50_query_count"]
        - fw_d["vocab50_query_count"],
        "vocab50_word_macro_top10": float(
            cw_d["vocab50_word_macro_top10"] - fw_d["vocab50_word_macro_top10"]
        ),
        "vocab50_micro_top10": float(
            cw_d["vocab50_micro_top10"] - fw_d["vocab50_micro_top10"]
        ),
        "vocab50_word_macro_top1": float(
            cw_d["vocab50_word_macro_top1"] - fw_d["vocab50_word_macro_top1"]
        ),
        "vocab50_micro_top1": float(
            cw_d["vocab50_micro_top1"] - fw_d["vocab50_micro_top1"]
        ),
        "vocab50_mrr": float(cw_d["vocab50_mrr"] - fw_d["vocab50_mrr"]),
        "full_vocab_word_macro_top10": float(
            cw_d["full_vocab_word_macro_top10"]
            - fw_d["full_vocab_word_macro_top10"]
        ),
        "full_vocab_subject_macro_top10": float(
            cw_d["full_vocab_subject_macro_top10"]
            - fw_d["full_vocab_subject_macro_top10"]
        ),
        "full_vocab_micro_top10": float(
            cw_d["full_vocab_micro_top10"] - fw_d["full_vocab_micro_top10"]
        ),
        "full_vocab_word_macro_top1": float(
            cw_d["full_vocab_word_macro_top1"]
            - fw_d["full_vocab_word_macro_top1"]
        ),
        "full_vocab_micro_top1": float(
            cw_d["full_vocab_micro_top1"] - fw_d["full_vocab_micro_top1"]
        ),
        "full_vocab_mrr": float(
            cw_d["full_vocab_mrr"] - fw_d["full_vocab_mrr"]
        ),
        "full_vocab_median_rank": float(
            cw_d["full_vocab_median_rank"] - fw_d["full_vocab_median_rank"]
        ),
    }
    for lm_k in (
        "vocab50_word_macro_top10",
        "vocab50_micro_top10",
        "vocab50_word_macro_top1",
        "vocab50_micro_top1",
        "vocab50_mrr",
        "full_vocab_word_macro_top10",
        "full_vocab_subject_macro_top10",
        "full_vocab_micro_top10",
        "full_vocab_word_macro_top1",
        "full_vocab_micro_top1",
        "full_vocab_mrr",
        "full_vocab_median_rank",
    ):
        is_50_m = "vocab50" in lm_k
        p_lm = _compute_two_class_linguistic_pvalue(
            float(cw_d.get(lm_k, 0.0)),
            float(fw_d.get(lm_k, 0.0)),
            metric_key=lm_k,
            n_content_queries=int(
                cw_d.get("vocab50_query_count", cw_d.get("query_count", 0))
                if is_50_m
                else cw_d.get("query_count", 0)
            ),
            n_function_queries=int(
                fw_d.get("vocab50_query_count", fw_d.get("query_count", 0))
                if is_50_m
                else fw_d.get("query_count", 0)
            ),
            n_content_words=int(cw_d.get("unique_word_count", 20)),
            n_function_words=int(fw_d.get("unique_word_count", 30)),
            full_vocab_size=int(full_cand_count),
        )
        cmf_entry[f"p_value_{lm_k}"] = float(p_lm)
        cmf_entry[f"sig_{lm_k}"] = _significance_stars(p_lm)
        cmf_entry[f"p_str_{lm_k}"] = _format_p_value_with_stars(p_lm)
    linguistic_breakdown["content_minus_function"] = cmf_entry

    # 配对材料块统计显著性检验 & K0->K1->K2 状态转移矩阵
    statistical_significance: dict[str, dict] = {}
    if eval_cov_50.any():
        statistical_significance["clean_vs_chance_vocab50_top10"] = (
            _compute_paired_block_significance(
                (eval_ranks_50[eval_cov_50] <= 10.0).astype(np.float64),
                0.20,
                eval_blocks[eval_cov_50].tolist(),
                label="Clean vs Chance (N50 Top-10)",
                metric="vocab50_top10",
            )
        )
    statistical_significance["clean_vs_chance_full_vocab_top10"] = (
        _compute_paired_block_significance(
            (eval_ranks_fv <= 10.0).astype(np.float64),
            fv_chance_10,
            eval_blocks.tolist(),
            label="Clean vs Chance (Full-Vocab Top-10)",
            metric="full_vocab_top10",
        )
    )

    revision_horizons: dict[str, dict] = {}
    revision_transitions: dict[str, dict] = {}
    if revision:
        for delay in sorted(encoded["revision_predictions"]):
            d_key = str(delay)
            s50_common = strict50.get(d_key, {}).get("common_targets", {})
            s50_all = strict50.get(d_key, {}).get("all_targets", {})
            fv_horizon = (
                full_vocab.get("horizons", {}).get(d_key)
                or full_vocab.get("horizons", {}).get(f"k{d_key}")
                or {}
            )
            fv_common = fv_horizon.get("common_targets", {})
            fv_all = fv_horizon.get("all_targets", {})
            revision_horizons[f"K{delay}"] = {
                "delay": int(delay),
                "label": f"K{delay} (+{delay}w)",
                "common_vocab50_macro_top10": float(
                    s50_common.get("macro_recall_at_10", 0.0)
                ),
                "common_vocab50_micro_top10": float(
                    s50_common.get("micro_recall_at_10", 0.0)
                ),
                "common_vocab50_macro_top1": float(
                    s50_common.get("macro_recall_at_1", 0.0)
                ),
                "common_vocab50_micro_top1": float(
                    s50_common.get("micro_recall_at_1", 0.0)
                ),
                "common_vocab50_mrr": float(
                    s50_common.get("mean_reciprocal_rank", 0.0)
                ),
                "common_vocab50_median_rank": float(
                    s50_common.get("median_rank", 25.0)
                ),
                "all_vocab50_macro_top10": float(
                    s50_all.get("macro_recall_at_10", 0.0)
                ),
                "all_vocab50_micro_top10": float(
                    s50_all.get("micro_recall_at_10", 0.0)
                ),
                "common_full_vocab_macro_top10": float(
                    fv_common.get("word_macro_recall_at_10", 0.0)
                ),
                "common_full_vocab_word_macro_top10": float(
                    fv_common.get("word_macro_recall_at_10", 0.0)
                ),
                "common_full_vocab_subject_macro_top10": float(
                    fv_common.get(
                        "subject_equal_macro_recall_at_10",
                        fv_horizon.get("subject_macro_top10", 0.0),
                    )
                ),
                "common_full_vocab_micro_top10": float(
                    fv_common.get("micro_recall_at_10", 0.0)
                ),
                "common_full_vocab_macro_top1": float(
                    fv_common.get("word_macro_recall_at_1", 0.0)
                ),
                "common_full_vocab_micro_top1": float(
                    fv_common.get("micro_recall_at_1", 0.0)
                ),
                "common_full_vocab_mrr": float(
                    fv_common.get("mean_reciprocal_rank", 0.0)
                ),
                "common_full_vocab_median_rank": float(
                    fv_common.get("median_rank", full_cand_count / 2.0)
                ),
                "all_full_vocab_macro_top10": float(
                    fv_all.get("word_macro_recall_at_10", 0.0)
                ),
                "all_full_vocab_subject_macro_top10": float(
                    fv_all.get("subject_equal_macro_recall_at_10", 0.0)
                ),
                "all_full_vocab_micro_top10": float(
                    fv_all.get("micro_recall_at_10", 0.0)
                ),
            }

        common_blocks = block_col_all[common_mask_np].tolist()
        for d_from, d_to, t_key, sig_prefix in (
            (0, 1, "K0_to_K1", "K1_minus_K0"),
            (0, 2, "K0_to_K2", "K2_minus_K0"),
            (1, 2, "K1_to_K2", "K2_minus_K1"),
        ):
            if d_from not in ranks_50_by_delay or d_to not in ranks_50_by_delay:
                continue
            r50_a = ranks_50_by_delay[d_from][common_mask_np]
            r50_b = ranks_50_by_delay[d_to][common_mask_np]
            rfv_a = ranks_full_by_delay[d_from][common_mask_np]
            rfv_b = ranks_full_by_delay[d_to][common_mask_np]
            revision_transitions[t_key] = {
                "from_horizon": f"K{d_from}",
                "to_horizon": f"K{d_to}",
                "vocab50_top10": _compute_transition_counts(
                    r50_a <= 10.0, r50_b <= 10.0
                ),
                "vocab50_top1": _compute_transition_counts(
                    r50_a <= 1.0, r50_b <= 1.0
                ),
                "full_vocab_top10": _compute_transition_counts(
                    rfv_a <= 10.0, rfv_b <= 10.0
                ),
                "full_vocab_top1": _compute_transition_counts(
                    rfv_a <= 1.0, rfv_b <= 1.0
                ),
            }
            statistical_significance[f"{sig_prefix}_vocab50_top10"] = (
                _compute_paired_block_significance(
                    (r50_b <= 10.0).astype(np.float64),
                    (r50_a <= 10.0).astype(np.float64),
                    common_blocks,
                    label=f"K{d_to} - K{d_from} (N50 Top-10)",
                    metric="common_vocab50_top10",
                )
            )
            statistical_significance[f"{sig_prefix}_full_vocab_top10"] = (
                _compute_paired_block_significance(
                    (rfv_b <= 10.0).astype(np.float64),
                    (rfv_a <= 10.0).astype(np.float64),
                    common_blocks,
                    label=f"K{d_to} - K{d_from} (Full-Vocab Top-10)",
                    metric="common_full_vocab_top10",
                )
            )

    return {
        "split": split,
        "total_events": int(len(eval_ds)),
        "evaluated_events": int(avail_mask.sum().item()),
        "vocabulary_sweep": vocab_sweep,
        "per_subject": per_subject,
        "per_subject_summary": per_subject_summary,
        "linguistic_breakdown": linguistic_breakdown,
        "revision_horizons": revision_horizons,
        "revision_transitions": revision_transitions,
        "statistical_significance": statistical_significance,
        "strict_vocab50": strict50,
        "full_vocabulary": full_vocab,
    }


compute_encoded_evaluation_metrics = _compute_encoded_evaluation_metrics


def _parse_shift_seconds_list(
    raw_shifts: str | Iterable[float | int | str] | None,
) -> list[float]:
    """解析时间错位秒数列表（支持逗号分隔字符串或数值序列），排除 0 秒并保持顺序去重。"""
    if raw_shifts is None:
        return []
    if isinstance(raw_shifts, (int, float)):
        val = float(raw_shifts)
        return [val] if abs(val) > 1e-9 else []
    if isinstance(raw_shifts, str):
        tokens: list[float | int | str] = [
            tok.strip() for tok in raw_shifts.split(",") if tok.strip()
        ]
    else:
        tokens = []
        for item in raw_shifts:
            if isinstance(item, str):
                tokens.extend(t.strip() for t in item.split(",") if t.strip())
            else:
                tokens.append(item)
    parsed: list[float] = []
    seen: set[float] = set()
    for tok in tokens:
        val = float(tok)
        if abs(val) <= 1e-9:
            continue
        rounded = round(val, 6)
        if rounded not in seen:
            seen.add(rounded)
            parsed.append(val)
    return parsed


def _format_shift_label(shift_sec: float) -> tuple[str, str]:
    """返回 (条件键名, 展示标签)，例如 ('shift_-30s', 'Shift -30s')。"""
    if abs(shift_sec - round(shift_sec)) < 1e-6:
        sec_int = int(round(shift_sec))
        return f"shift_{sec_int:+d}s", f"Shift {sec_int:+d}s"
    return f"shift_{shift_sec:+.1f}s", f"Shift {shift_sec:+.1f}s"


def _resolve_control_specification(
    cfg: dict,
    *,
    control_config_path: str | Path | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    zero_signal_control: bool = False,
    subtraction_etas: str | Iterable[float | int | str] | None = None,
) -> dict:
    """合并 CLI 参数、独立对照 YAML 配置（configs/controls/...）与主配置中的负对照设置。"""
    from braindecoding.evaluation.common_mode_subtraction_evaluation import (
        DEFAULT_SUBTRACTION_ETAS,
        parse_eta_values,
    )

    resolved_ctrl_path: Path | None = None
    ctrl_yaml: dict = {}
    if control_config_path is not None and str(control_config_path).strip():
        resolved_ctrl_path = resolve_config_path(
            project_path(control_config_path)
        )
        if not resolved_ctrl_path.is_file():
            raise FileNotFoundError(
                f"指定的负对照配置文件不存在：{control_config_path}"
            )
        loaded = load_yaml_with_extends(resolved_ctrl_path)
        if isinstance(loaded, dict):
            ctrl_yaml = loaded

    dataset = cfg.get("experiment", {}).get("dataset")
    ctrl_wf = (
        ctrl_yaml.get("workflow")
        if isinstance(ctrl_yaml.get("workflow"), dict)
        else {}
    )
    ctrl_dataset = ctrl_wf.get("dataset")
    if dataset and ctrl_dataset and str(dataset) != str(ctrl_dataset):
        raise ValueError(
            f"数据集 {dataset!r} 与负对照配置文件中的 dataset={ctrl_dataset!r} 不一致。"
        )

    cfg_control = (
        dict(cfg.get("control")) if isinstance(cfg.get("control"), dict) else {}
    )
    cfg_eval = (
        dict(cfg.get("evaluation"))
        if isinstance(cfg.get("evaluation"), dict)
        else {}
    )
    yaml_control = (
        dict(ctrl_yaml.get("control"))
        if isinstance(ctrl_yaml.get("control"), dict)
        else {}
    )
    yaml_eval = (
        dict(ctrl_yaml.get("evaluation"))
        if isinstance(ctrl_yaml.get("evaluation"), dict)
        else {}
    )

    merged_control = {**cfg_control, **yaml_control}
    merged_eval = {**cfg_eval, **yaml_eval}

    cms_raw = (
        merged_control.get("common_mode_subtraction")
        if isinstance(merged_control.get("common_mode_subtraction"), dict)
        else (
            merged_eval.get("common_mode_subtraction")
            if isinstance(merged_eval.get("common_mode_subtraction"), dict)
            else {}
        )
    )
    cms_enabled = (
        subtraction_etas is not None
        or bool(cms_raw.get("enabled", False))
        or str(merged_control.get("control_type", "")).strip().lower()
        == "common_mode_subtraction"
        or str(merged_eval.get("negative_control", "")).strip().lower()
        == "common_mode_subtraction"
    )
    cms_etas = parse_eta_values(
        subtraction_etas
        if subtraction_etas is not None
        else cms_raw.get("eta_values", DEFAULT_SUBTRACTION_ETAS)
    )
    cms_normalize = bool(cms_raw.get("normalize_before_subtraction", True))

    sub_raw = (
        merged_control.get("subspace_coherence_debiasing")
        if isinstance(merged_control.get("subspace_coherence_debiasing"), dict)
        else (
            merged_eval.get("subspace_coherence_debiasing")
            if isinstance(merged_eval.get("subspace_coherence_debiasing"), dict)
            else {}
        )
    )
    sub_enabled = (
        bool(sub_raw.get("enabled", False))
        or str(merged_control.get("control_type", "")).strip().lower()
        == "subspace_coherence_debiasing"
        or str(merged_eval.get("negative_control", "")).strip().lower()
        == "subspace_coherence_debiasing"
    )
    sub_ranks = [
        int(k)
        for k in (sub_raw.get("rank_k_values") or (8, 12, 16, 24))
        if int(k) > 0
    ]
    sub_etas = parse_eta_values(
        subtraction_etas
        if (subtraction_etas is not None and sub_enabled)
        else sub_raw.get("eta_values", (0.8, 1.0, 1.2))
    )
    sub_full_etas = parse_eta_values(
        sub_raw.get("full_loo_eta_values", (0.5, 0.8, 1.0))
    )

    if shift_seconds is not None:
        shifts = _parse_shift_seconds_list(shift_seconds)
    elif "shift_seconds" in merged_control:
        shifts = _parse_shift_seconds_list(merged_control.get("shift_seconds"))
    elif "shift_seconds" in merged_eval:
        shifts = _parse_shift_seconds_list(merged_eval.get("shift_seconds"))
    elif cms_enabled or sub_enabled:
        shifts = list(DEFAULT_CONTROL_SHIFTS_SECONDS)
    else:
        shifts = []

    enable_zero = (
        bool(zero_signal_control)
        or bool(merged_control.get("include_zero_signal", False))
        or bool(merged_eval.get("zero_signal_control", False))
        or str(merged_control.get("control_type", "")).strip().lower()
        == "zero_signal"
    )
    enabled = bool(shifts) or enable_zero or cms_enabled or sub_enabled
    if not enabled:
        return {
            "enabled": False,
            "name": "none",
            "control_type": "none",
            "config_path": None,
            "shift_seconds": [],
            "include_zero_signal": False,
            "common_mode_subtraction": {
                "enabled": False,
                "mode": "leave_one_out",
                "eta_values": [],
                "normalize_before_subtraction": True,
            },
            "subspace_coherence_debiasing": {
                "enabled": False,
                "mode": "subspace_coherence_leave_one_out",
                "rank_k_values": [],
                "eta_values": [],
                "full_loo_eta_values": [],
                "center_svd": True,
                "use_coherence_gate": "both",
            },
        }

    default_type = (
        "subspace_coherence_debiasing"
        if sub_enabled
        else (
            "common_mode_subtraction"
            if cms_enabled
            else (
                "temporal_shift"
                if shifts and not enable_zero
                else (
                    "zero_signal"
                    if enable_zero and not shifts
                    else "combined_control"
                )
            )
        )
    )
    control_type = str(merged_control.get("control_type") or default_type)
    control_name = str(
        merged_control.get("name") or f"{control_type}_negative_control"
    )
    cfg_path_label = (
        normalize_project_relative(resolved_ctrl_path)
        if resolved_ctrl_path is not None
        else (
            cfg.get("workflow", {}).get("config_path")
            if cfg_control or "shift_seconds" in cfg_eval
            else None
        )
    )
    return {
        "enabled": True,
        "name": control_name,
        "control_type": control_type,
        "description": str(merged_control.get("description", "")),
        "config_path": cfg_path_label,
        "shift_seconds": shifts,
        "shift_mode": str(
            merged_control.get("shift_mode", "deterministic_circular")
        ),
        "same_subject": bool(merged_control.get("same_subject", True)),
        "same_recording": bool(merged_control.get("same_recording", True)),
        "non_overlapping_window": bool(
            merged_control.get("non_overlapping_window", True)
        ),
        "include_clean_reference": bool(
            merged_control.get("include_clean_reference", True)
        ),
        "include_zero_signal": bool(enable_zero),
        "common_mode_subtraction": {
            "enabled": bool(cms_enabled),
            "mode": str(cms_raw.get("mode", "leave_one_out")),
            "eta_values": cms_etas,
            "normalize_before_subtraction": cms_normalize,
        },
        "subspace_coherence_debiasing": {
            "enabled": bool(sub_enabled),
            "mode": str(
                sub_raw.get("mode", "subspace_coherence_leave_one_out")
            ),
            "rank_k_values": sub_ranks,
            "eta_values": sub_etas,
            "full_loo_eta_values": sub_full_etas,
            "center_svd": bool(sub_raw.get("center_svd", True)),
            "use_coherence_gate": sub_raw.get("use_coherence_gate", "both"),
            "coherence_low": float(sub_raw.get("coherence_low", 0.35)),
            "coherence_high": float(sub_raw.get("coherence_high", 0.65)),
            "gate_min": float(sub_raw.get("gate_min", 0.2)),
            "gate_max": float(sub_raw.get("gate_max", 1.2)),
            "normalize_before_subtraction": bool(
                sub_raw.get("normalize_before_subtraction", True)
            ),
        },
    }


resolve_control_specification = _resolve_control_specification


def _extract_control_summary_row(
    key: str,
    label: str,
    shift_seconds: float | None,
    metrics: dict,
    *,
    clean_summary: dict | None = None,
) -> dict:
    """提取单个条件（Clean、Time-Shifted 或 Zero-Signal）的核心词级检索指标与相对 Clean 的差值。"""
    sweep = metrics.get("vocabulary_sweep", {})
    n50 = sweep.get("N50", {})
    fv = sweep.get("full_vocab", {})
    horizons = metrics.get("revision_horizons") or {}

    fv_word_macro_top10 = float(
        fv.get("word_macro_top10", fv.get("macro_top10", 0.0))
    )
    fv_subj_macro_top10 = float(
        fv.get("subject_macro_top10", fv.get("micro_top10", 0.0))
    )
    row = {
        "key": key,
        "label": label,
        "shift_seconds": shift_seconds,
        "vocab50_macro_top10": float(n50.get("macro_top10", 0.0)),
        "vocab50_micro_top10": float(n50.get("micro_top10", 0.0)),
        "vocab50_macro_top5": float(n50.get("macro_top5", 0.0)),
        "vocab50_macro_top1": float(n50.get("macro_top1", 0.0)),
        "vocab50_micro_top1": float(n50.get("micro_top1", 0.0)),
        "vocab50_mrr": float(n50.get("mrr", 0.0)),
        "vocab50_median_rank": float(n50.get("median_rank", 0.0)),
        "vocab50_chance_top10": float(n50.get("chance_top10", 0.2)),
        "full_vocab_macro_top10": fv_word_macro_top10,
        "full_vocab_word_macro_top10": fv_word_macro_top10,
        "full_vocab_subject_macro_top10": fv_subj_macro_top10,
        "full_vocab_micro_top10": float(fv.get("micro_top10", 0.0)),
        "full_vocab_macro_top1": float(fv.get("macro_top1", 0.0)),
        "full_vocab_micro_top1": float(fv.get("micro_top1", 0.0)),
        "full_vocab_mrr": float(fv.get("mrr", 0.0)),
        "full_vocab_median_rank": float(fv.get("median_rank", 0.0)),
    }
    if horizons:
        row["horizons"] = {
            h_key: {
                "common_vocab50_macro_top10": float(
                    horizons[h_key].get("common_vocab50_macro_top10", 0.0)
                ),
                "common_vocab50_micro_top10": float(
                    horizons[h_key].get("common_vocab50_micro_top10", 0.0)
                ),
                "common_vocab50_macro_top1": float(
                    horizons[h_key].get("common_vocab50_macro_top1", 0.0)
                ),
                "common_vocab50_micro_top1": float(
                    horizons[h_key].get("common_vocab50_micro_top1", 0.0)
                ),
                "common_vocab50_mrr": float(
                    horizons[h_key].get("common_vocab50_mrr", 0.0)
                ),
                "common_vocab50_median_rank": float(
                    horizons[h_key].get("common_vocab50_median_rank", 25.0)
                ),
                "common_full_vocab_macro_top10": float(
                    horizons[h_key].get(
                        "common_full_vocab_word_macro_top10",
                        horizons[h_key].get("common_full_vocab_macro_top10", 0.0),
                    )
                ),
                "common_full_vocab_word_macro_top10": float(
                    horizons[h_key].get(
                        "common_full_vocab_word_macro_top10",
                        horizons[h_key].get("common_full_vocab_macro_top10", 0.0),
                    )
                ),
                "common_full_vocab_subject_macro_top10": float(
                    horizons[h_key].get(
                        "common_full_vocab_subject_macro_top10",
                        horizons[h_key].get("common_full_vocab_micro_top10", 0.0),
                    )
                ),
                "common_full_vocab_micro_top10": float(
                    horizons[h_key].get("common_full_vocab_micro_top10", 0.0)
                ),
                "common_full_vocab_macro_top1": float(
                    horizons[h_key].get("common_full_vocab_macro_top1", 0.0)
                ),
                "common_full_vocab_micro_top1": float(
                    horizons[h_key].get("common_full_vocab_micro_top1", 0.0)
                ),
                "common_full_vocab_mrr": float(
                    horizons[h_key].get("common_full_vocab_mrr", 0.0)
                ),
                "common_full_vocab_median_rank": float(
                    horizons[h_key].get("common_full_vocab_median_rank", 0.0)
                ),
            }
            for h_key in ("K0", "K1", "K2")
            if h_key in horizons
        }
    if clean_summary is not None:
        c_fv_macro_10 = float(
            clean_summary.get(
                "full_vocab_word_macro_top10",
                clean_summary.get("full_vocab_macro_top10", 0.0),
            )
        )
        c_fv_subj_10 = float(
            clean_summary.get(
                "full_vocab_subject_macro_top10",
                clean_summary.get("full_vocab_micro_top10", 0.0),
            )
        )
        delta = {
            "vocab50_macro_top10": float(
                row["vocab50_macro_top10"] - clean_summary["vocab50_macro_top10"]
            ),
            "vocab50_micro_top10": float(
                row["vocab50_micro_top10"] - clean_summary["vocab50_micro_top10"]
            ),
            "vocab50_macro_top1": float(
                row["vocab50_macro_top1"] - clean_summary["vocab50_macro_top1"]
            ),
            "vocab50_micro_top1": float(
                row["vocab50_micro_top1"]
                - clean_summary.get("vocab50_micro_top1", 0.0)
            ),
            "vocab50_mrr": float(
                row["vocab50_mrr"] - clean_summary["vocab50_mrr"]
            ),
            "full_vocab_macro_top10": float(
                row["full_vocab_macro_top10"] - c_fv_macro_10
            ),
            "full_vocab_word_macro_top10": float(
                row["full_vocab_word_macro_top10"] - c_fv_macro_10
            ),
            "full_vocab_subject_macro_top10": float(
                row["full_vocab_subject_macro_top10"] - c_fv_subj_10
            ),
            "full_vocab_micro_top10": float(
                row["full_vocab_micro_top10"]
                - clean_summary.get("full_vocab_micro_top10", 0.0)
            ),
            "full_vocab_macro_top1": float(
                row["full_vocab_macro_top1"]
                - clean_summary.get("full_vocab_macro_top1", 0.0)
            ),
            "full_vocab_micro_top1": float(
                row["full_vocab_micro_top1"]
                - clean_summary.get("full_vocab_micro_top1", 0.0)
            ),
            "full_vocab_mrr": float(
                row["full_vocab_mrr"] - clean_summary.get("full_vocab_mrr", 0.0)
            ),
            "clean_minus_control_vocab50_macro_top10": float(
                clean_summary["vocab50_macro_top10"] - row["vocab50_macro_top10"]
            ),
            "clean_minus_control_vocab50_micro_top10": float(
                clean_summary["vocab50_micro_top10"] - row["vocab50_micro_top10"]
            ),
            "clean_minus_control_vocab50_macro_top1": float(
                clean_summary["vocab50_macro_top1"] - row["vocab50_macro_top1"]
            ),
            "clean_minus_control_vocab50_micro_top1": float(
                clean_summary.get("vocab50_micro_top1", 0.0)
                - row["vocab50_micro_top1"]
            ),
            "clean_minus_control_full_vocab_macro_top10": float(
                c_fv_macro_10 - row["full_vocab_macro_top10"]
            ),
            "clean_minus_control_full_vocab_micro_top10": float(
                clean_summary.get("full_vocab_micro_top10", 0.0)
                - row["full_vocab_micro_top10"]
            ),
            "clean_minus_control_full_vocab_macro_top1": float(
                clean_summary.get("full_vocab_macro_top1", 0.0)
                - row["full_vocab_macro_top1"]
            ),
            "clean_minus_control_full_vocab_micro_top1": float(
                clean_summary.get("full_vocab_micro_top1", 0.0)
                - row["full_vocab_micro_top1"]
            ),
        }
        if "horizons" in row and "horizons" in clean_summary:
            delta["horizons"] = {
                h_key: {
                    "common_vocab50_macro_top10": float(
                        row["horizons"][h_key]["common_vocab50_macro_top10"]
                        - clean_summary["horizons"][h_key][
                            "common_vocab50_macro_top10"
                        ]
                    ),
                    "common_vocab50_micro_top10": float(
                        row["horizons"][h_key]["common_vocab50_micro_top10"]
                        - clean_summary["horizons"][h_key].get(
                            "common_vocab50_micro_top10", 0.0
                        )
                    ),
                    "common_vocab50_macro_top1": float(
                        row["horizons"][h_key]["common_vocab50_macro_top1"]
                        - clean_summary["horizons"][h_key].get(
                            "common_vocab50_macro_top1", 0.0
                        )
                    ),
                    "common_vocab50_micro_top1": float(
                        row["horizons"][h_key]["common_vocab50_micro_top1"]
                        - clean_summary["horizons"][h_key].get(
                            "common_vocab50_micro_top1", 0.0
                        )
                    ),
                    "common_full_vocab_macro_top10": float(
                        row["horizons"][h_key]["common_full_vocab_macro_top10"]
                        - clean_summary["horizons"][h_key].get(
                            "common_full_vocab_word_macro_top10",
                            clean_summary["horizons"][h_key].get(
                                "common_full_vocab_macro_top10", 0.0
                            ),
                        )
                    ),
                    "common_full_vocab_micro_top10": float(
                        row["horizons"][h_key]["common_full_vocab_micro_top10"]
                        - clean_summary["horizons"][h_key].get(
                            "common_full_vocab_micro_top10", 0.0
                        )
                    ),
                    "common_full_vocab_macro_top1": float(
                        row["horizons"][h_key]["common_full_vocab_macro_top1"]
                        - clean_summary["horizons"][h_key].get(
                            "common_full_vocab_macro_top1", 0.0
                        )
                    ),
                    "common_full_vocab_micro_top1": float(
                        row["horizons"][h_key]["common_full_vocab_micro_top1"]
                        - clean_summary["horizons"][h_key].get(
                            "common_full_vocab_micro_top1", 0.0
                        )
                    ),
                }
                for h_key in row["horizons"]
                if h_key in clean_summary["horizons"]
            }
        row["delta_vs_clean"] = delta
    return row


extract_control_summary_row = _extract_control_summary_row


def _summarize_shifted_conditions(
    clean_summary: dict,
    shifted_rows: list[dict],
) -> dict:
    """汇总所有时间错位偏移量（如 -15s, -30s, -45s, -60s, -90s, -120s）的均值、标准差与相对 Clean 的平均降幅。"""
    if not shifted_rows:
        return {}
    metric_keys = (
        "vocab50_macro_top10",
        "vocab50_micro_top10",
        "vocab50_macro_top5",
        "vocab50_macro_top1",
        "vocab50_micro_top1",
        "vocab50_mrr",
        "vocab50_median_rank",
        "full_vocab_macro_top10",
        "full_vocab_word_macro_top10",
        "full_vocab_subject_macro_top10",
        "full_vocab_micro_top10",
        "full_vocab_macro_top1",
        "full_vocab_micro_top1",
        "full_vocab_mrr",
        "full_vocab_median_rank",
    )
    agg: dict = {
        "shift_count": len(shifted_rows),
        "shift_seconds": [r["shift_seconds"] for r in shifted_rows],
    }
    for m_key in metric_keys:
        vals = np.asarray(
            [float(r.get(m_key, 0.0)) for r in shifted_rows], dtype=np.float64
        )
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=0))
        agg[f"mean_{m_key}"] = mean_val
        agg[f"std_{m_key}"] = std_val
        if m_key in clean_summary:
            agg[f"delta_{m_key}_vs_clean"] = float(
                mean_val - float(clean_summary[m_key])
            )
            agg[f"clean_minus_shifted_{m_key}"] = float(
                float(clean_summary[m_key]) - mean_val
            )
    if all("horizons" in r for r in shifted_rows) and "horizons" in clean_summary:
        horizon_metric_keys = (
            "common_vocab50_macro_top10",
            "common_vocab50_micro_top10",
            "common_vocab50_macro_top1",
            "common_vocab50_micro_top1",
            "common_vocab50_mrr",
            "common_vocab50_median_rank",
            "common_full_vocab_macro_top10",
            "common_full_vocab_word_macro_top10",
            "common_full_vocab_subject_macro_top10",
            "common_full_vocab_micro_top10",
            "common_full_vocab_macro_top1",
            "common_full_vocab_micro_top1",
            "common_full_vocab_mrr",
            "common_full_vocab_median_rank",
        )
        h_agg: dict = {}
        for h_key in ("K0", "K1", "K2"):
            if h_key not in clean_summary["horizons"]:
                continue
            c_hz_entry = clean_summary["horizons"][h_key]
            h_item: dict[str, float] = {}
            for hm_key in horizon_metric_keys:
                hm_vals = np.asarray(
                    [
                        float(r["horizons"][h_key].get(hm_key, 0.0))
                        for r in shifted_rows
                        if h_key in r["horizons"]
                    ],
                    dtype=np.float64,
                )
                if len(hm_vals) == 0:
                    continue
                m_val = float(np.mean(hm_vals))
                s_val = float(np.std(hm_vals, ddof=0))
                c_val = float(c_hz_entry.get(hm_key, 0.0))
                h_item[f"mean_{hm_key}"] = m_val
                h_item[f"std_{hm_key}"] = s_val
                h_item[f"delta_{hm_key}_vs_clean"] = float(m_val - c_val)
                h_item[f"clean_minus_shifted_{hm_key}"] = float(c_val - m_val)
            if h_item:
                h_agg[h_key] = h_item
        agg["horizons"] = h_agg

    sh_v50_arr = np.asarray(
        [float(r.get("vocab50_macro_top10", 0.0)) for r in shifted_rows],
        dtype=np.float64,
    )
    sh_fv_arr = np.asarray(
        [float(r.get("full_vocab_macro_top10", 0.0)) for r in shifted_rows],
        dtype=np.float64,
    )
    clean_v50_val = float(clean_summary.get("vocab50_macro_top10", 0.0))
    clean_fv_val = float(clean_summary.get("full_vocab_macro_top10", 0.0))
    agg["statistical_significance"] = {
        "clean_minus_shifted_vocab50_top10": _compute_paired_block_significance(
            np.full_like(sh_v50_arr, clean_v50_val),
            sh_v50_arr,
            [str(r.get("key", i)) for i, r in enumerate(shifted_rows)],
            label="Clean (0s) - Shifted Mean (N50 Top-10)",
            metric="vocab50_macro_top10",
        ),
        "clean_minus_shifted_full_vocab_top10": _compute_paired_block_significance(
            np.full_like(sh_fv_arr, clean_fv_val),
            sh_fv_arr,
            [str(r.get("key", i)) for i, r in enumerate(shifted_rows)],
            label="Clean (0s) - Shifted Mean (Full-Vocab Top-10)",
            metric="full_vocab_macro_top10",
        ),
    }
    return agg


summarize_shifted_conditions = _summarize_shifted_conditions


def _evaluate_single_stage(
    dataset: str | None = None,
    *,
    stage: str | None = None,
    seed: int | None = 0,
    split: str = "val",
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    precision: str | None = None,
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
    require_cache: bool = False,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    config_path: str | Path | None = None,
    control_config_path: str | Path | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    zero_signal_control: bool = False,
    subtraction_etas: str | Iterable[float | int | str] | None = None,
    plot: bool = True,
    show_progress: bool = True,
    max_eval_rows: int | None = None,
    verbose: bool = False,
    print_report: bool = True,
) -> dict:
    """对单个训练阶段（word / context / revision）执行验证集或测试集评估并生成图表。"""
    from braindecoding.training.workflow_runner import (
        build_workflow_config,
        run_cache_workflow,
        verify_cache_ready,
        workflow_output_directory,
    )

    if split not in {"val", "test"}:
        raise ValueError(f"评估 split 只能是 'val' 或 'test'，收到：{split!r}")

    cfg, subset_spec = build_workflow_config(
        dataset,
        stage=stage,
        seed=seed,
        subject=subject,
        num_subjects=num_subjects,
        subjects=subjects,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        precision=precision,
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        mode="train",
        output_dir=output_dir,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        config_path=config_path,
    )
    control_spec = _resolve_control_specification(
        cfg,
        control_config_path=control_config_path,
        shift_seconds=shift_seconds,
        zero_signal_control=zero_signal_control,
        subtraction_etas=subtraction_etas,
    )
    dataset = subset_spec["dataset"]
    stage = cfg["workflow"]["stage"]
    resolved_exp_tag = cfg["workflow"].get("exp_tag", "")
    resolved_word_exp_tag = cfg["workflow"].get(
        "word_exp_tag", resolved_exp_tag
    )
    tc = cfg["training"]
    seed = int(tc["seed"])
    out = Path(tc["output_dir"])

    if checkpoint_path is not None:
        ckpt_file = Path(checkpoint_path).resolve()
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"指定的评估检查点不存在：{ckpt_file}")
        if output_dir is None:
            out = ckpt_file.parent
    else:
        ckpt_file = out / "best.ckpt"
        if not ckpt_file.is_file():
            ckpt_file = out / "best.pt"
        if not ckpt_file.is_file():
            smoke_dir = workflow_output_directory(
                dataset,
                stage,
                subset_spec["subset_tag"],
                seed,
                mode="smoke",
                exp_tag=resolved_exp_tag,
            )
            smoke_ckpt = smoke_dir / "best.ckpt"
            if not smoke_ckpt.is_file():
                smoke_ckpt = smoke_dir / "best.pt"
            if smoke_ckpt.is_file():
                ckpt_file = smoke_ckpt
                if output_dir is None:
                    out = smoke_dir
        if not ckpt_file.is_file() and stage == "word":
            fallback_tags: list[tuple[str, bool]] = []
            if resolved_word_exp_tag != resolved_exp_tag:
                fallback_tags.append(
                    (resolved_word_exp_tag, resolved_word_exp_tag == "")
                )
            if resolved_exp_tag and ("", True) not in fallback_tags:
                fallback_tags.append(("", True))
            for cand_tag, cand_ignore_imp in fallback_tags:
                for cand_mode in ("train", "smoke"):
                    cand_dir = workflow_output_directory(
                        dataset,
                        stage,
                        subset_spec["subset_tag"],
                        seed,
                        mode=cand_mode,
                        exp_tag=cand_tag,
                        ignore_implicit_tag=cand_ignore_imp,
                    )
                    cand_ckpt = cand_dir / "best.ckpt"
                    if not cand_ckpt.is_file():
                        cand_ckpt = cand_dir / "best.pt"
                    if cand_ckpt.is_file():
                        ckpt_file = cand_ckpt
                        resolved_exp_tag = cand_tag
                        if output_dir is None:
                            out = cand_dir
                        break
                if ckpt_file.is_file():
                    break
        if not ckpt_file.is_file():
            raise FileNotFoundError(
                f"未找到阶段 {stage} ({STAGE_DISPLAY_NAMES.get(stage, stage)}) 的训练检查点：{out / 'best.ckpt'}。"
                f"请先完成该阶段的训练，或通过 --checkpoint 显式指定检查点路径。"
            )

    if require_cache:
        cache_verification = verify_cache_ready(cfg, splits=(split,))
    else:
        try:
            cache_verification = verify_cache_ready(cfg, splits=(split,))
        except FileNotFoundError:
            run_cache_workflow(
                dataset,
                subject=subject,
                num_subjects=num_subjects,
                subjects=subjects,
                include_test=(split == "test"),
                force=False,
                config_path=config_path,
            )
            cache_verification = verify_cache_ready(cfg, splits=(split,))

    device = choose_device(tc.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(seed)
    checkpoint = load_checkpoint(ckpt_file, map_location="cpu")
    if not bool(cfg["workflow"].get("window_explicitly_overridden", False)):
        ckpt_cfg_ds = (
            checkpoint.get("config", {}).get("dataset", {})
            if isinstance(checkpoint.get("config"), dict)
            else {}
        )
        ckpt_win = checkpoint.get(
            "window_seconds",
            ckpt_cfg_ds.get("window_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        ckpt_off = checkpoint.get(
            "window_start_offset_seconds",
            ckpt_cfg_ds.get("window_start_offset_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        if ckpt_win is not None:
            cfg["dataset"]["window_seconds"] = float(ckpt_win)
            cfg["workflow"]["window_seconds"] = float(ckpt_win)
            sr_hz = float(cfg["workflow"].get("sampling_rate_hz", 50.0))
            cfg["workflow"]["window_samples"] = int(round(float(ckpt_win) * sr_hz))
        if ckpt_off is not None:
            cfg["dataset"]["window_start_offset_seconds"] = float(ckpt_off)
            cfg["workflow"]["window_start_offset_seconds"] = float(ckpt_off)

    task = pallier_task if dataset == "pallier2025" else chinese_task
    read_fn = task.dataset_module.load_event_table
    build_fn = task.build_dataset
    selected_subjects = cfg["dataset"]["subjects"]

    train_table = read_fn(
        cfg["cache"]["event_table"],
        split="train",
        subjects=selected_subjects,
        trainable_only=True,
    )
    eval_table = read_fn(
        cfg["cache"]["event_table"],
        split=split,
        subjects=selected_subjects,
        trainable_only=True,
    )
    if max_eval_rows is not None:
        eval_table = limit_rows(eval_table, int(max_eval_rows))
    if len(eval_table) <= 0:
        raise ValueError(
            f"数据集 {dataset} 在划分 {split!r} 与受试者 {selected_subjects} 上的事件数为 0。"
        )

    eval_ds = build_fn(cfg, eval_table)
    grp_col = group_column(cfg)
    eval_bs = int(tc.get("eval_batch_size", tc["batch_size"]))
    eval_loader, _ = make_loader(
        eval_ds,
        tc,
        shuffle=False,
        group_column=grp_col,
        batch_size=eval_bs,
    )

    train_word_col = (
        "normalized_word"
        if "normalized_word" in train_table.columns
        else "标准词"
    )
    vocabulary = sorted(set(train_table[train_word_col].astype(str)))
    vocab50 = list(
        pallier_task.frozen_vocabulary()
        if dataset == "pallier2025"
        else chinese_task.frozen_vocabulary(cfg["cache"]["event_table"])
    )

    model_cfg = checkpoint.get("model_config", cfg["model"])
    ckpt_subjects = list(checkpoint.get("subject_ids", selected_subjects))
    subject_count = max(len(ckpt_subjects), eval_ds.subject_count)

    model = build_brain_embedding_model(
        eval_ds.channel_count,
        eval_ds.channel_positions,
        subject_count,
        model_cfg,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    amp_enabled = bool(tc.get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(tc)
    encoded = encode_loader(
        model,
        eval_loader,
        device,
        amp=amp_enabled,
        amp_dtype=amp_dtype,
        show_progress=show_progress,
        progress_desc=f"Eval [{stage} | {split} | Clean 0s]",
    )

    metrics = _compute_encoded_evaluation_metrics(
        encoded,
        eval_ds,
        vocabulary,
        vocab50,
        cfg,
        stage=stage,
        split=split,
    )

    negative_controls_result: dict = {
        "specification": control_spec,
        "clean_reference": _extract_control_summary_row(
            "clean_0s", "Clean (0s)", 0.0, metrics
        ),
        "conditions": [],
        "shifted_aggregate": {},
    }
    cms_cfg = control_spec.get("common_mode_subtraction") or {}
    cms_enabled = bool(cms_cfg.get("enabled", False))
    sub_cfg = control_spec.get("subspace_coherence_debiasing") or {}
    sub_enabled = bool(sub_cfg.get("enabled", False))
    keep_shifted_encoded = (
        cms_enabled
        or sub_enabled
        or len(control_spec.get("shift_seconds", ())) >= 2
    )
    shifted_encoded_by_shift: list[tuple[float, str, str, dict]] = []
    shifted_ds_by_shift: dict[float, object] = {}

    if control_spec["enabled"]:
        clean_summary = negative_controls_result["clean_reference"]
        shifted_summaries: list[dict] = []
        for shift_sec in control_spec["shift_seconds"]:
            cond_key, cond_label = _format_shift_label(float(shift_sec))
            shift_ds = build_fn(
                cfg, eval_table, temporal_shift_seconds=float(shift_sec)
            )
            shift_loader, _ = make_loader(
                shift_ds,
                tc,
                shuffle=False,
                group_column=grp_col,
                batch_size=eval_bs,
            )
            shift_encoded = encode_loader(
                model,
                shift_loader,
                device,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=f"Eval [{stage} | {split} | {cond_label}]",
            )
            shift_metrics = _compute_encoded_evaluation_metrics(
                shift_encoded,
                shift_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            summary_row = _extract_control_summary_row(
                cond_key,
                cond_label,
                float(shift_sec),
                shift_metrics,
                clean_summary=clean_summary,
            )
            shifted_summaries.append(summary_row)
            negative_controls_result["conditions"].append(
                {
                    "key": cond_key,
                    "label": cond_label,
                    "control_type": "temporal_shift",
                    "shift_seconds": float(shift_sec),
                    "summary": summary_row,
                    "metrics": {
                        "vocabulary_sweep": shift_metrics.get("vocabulary_sweep", {}),
                        "revision_horizons": shift_metrics.get("revision_horizons", {}),
                    },
                }
            )
            if keep_shifted_encoded:
                shifted_encoded_by_shift.append(
                    (float(shift_sec), cond_key, cond_label, shift_encoded)
                )
                if cms_enabled:
                    shifted_ds_by_shift[float(shift_sec)] = shift_ds
                else:
                    del shift_ds
                del shift_loader
            else:
                del shift_ds, shift_loader, shift_encoded

        if control_spec.get("include_zero_signal", False):
            zero_ds = build_fn(cfg, eval_table, zero_signal=True)
            zero_loader, _ = make_loader(
                zero_ds,
                tc,
                shuffle=False,
                group_column=grp_col,
                batch_size=eval_bs,
            )
            zero_encoded = encode_loader(
                model,
                zero_loader,
                device,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=f"Eval [{stage} | {split} | Zero Signal]",
            )
            zero_metrics = _compute_encoded_evaluation_metrics(
                zero_encoded,
                zero_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            zero_summary = _extract_control_summary_row(
                "zero_signal",
                "Zero Signal",
                None,
                zero_metrics,
                clean_summary=clean_summary,
            )
            negative_controls_result["conditions"].append(
                {
                    "key": "zero_signal",
                    "label": "Zero Signal",
                    "control_type": "zero_signal",
                    "shift_seconds": None,
                    "summary": zero_summary,
                    "metrics": {
                        "vocabulary_sweep": zero_metrics.get("vocabulary_sweep", {}),
                        "revision_horizons": zero_metrics.get("revision_horizons", {}),
                    },
                }
            )
            del zero_ds, zero_loader, zero_encoded

        if shifted_summaries:
            negative_controls_result["shifted_aggregate"] = (
                _summarize_shifted_conditions(clean_summary, shifted_summaries)
            )
        elif negative_controls_result["conditions"]:
            cond_summaries = [
                c["summary"]
                for c in negative_controls_result["conditions"]
                if isinstance(c.get("summary"), dict)
            ]
            if cond_summaries:
                negative_controls_result["shifted_aggregate"] = (
                    _summarize_shifted_conditions(clean_summary, cond_summaries)
                )

    training_history = []
    summary_path = out / "training_summary.json"
    if summary_path.is_file():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            training_history = list(summary_data.get("history", ()))
        except Exception:
            training_history = []

    out.mkdir(parents=True, exist_ok=True)
    eval_win_sec = float(
        getattr(
            eval_ds, "window_seconds", cfg["dataset"].get("window_seconds", 1.0)
        )
    )
    eval_win_samp = int(
        getattr(eval_ds, "window_samples", round(eval_win_sec * 50.0))
    )
    eval_win_off = float(
        cfg["dataset"].get("window_start_offset_seconds", 0.0)
    )
    eval_result = {
        "status": "evaluation_completed",
        "dataset": dataset,
        "stage": stage,
        "exp_tag": resolved_exp_tag,
        "word_exp_tag": resolved_word_exp_tag,
        "stage_display_name": STAGE_DISPLAY_NAMES.get(stage, stage),
        "split": split,
        "seed": seed,
        "precision": tc["precision"],
        "window_seconds": eval_win_sec,
        "window_samples": eval_win_samp,
        "window_start_offset_seconds": eval_win_off,
        "subset": subset_spec,
        "checkpoint": {
            "path": str(ckpt_file),
            "sha256": file_sha256(ckpt_file),
            "epoch": int(checkpoint.get("epoch", 0)),
            "selection": str(checkpoint.get("selection", "legacy50")),
        },
        "cache_verification": cache_verification,
        "metrics": metrics,
        "negative_controls": negative_controls_result,
        "training_history": training_history,
        "output_dir": str(out),
        "figures": {},
    }

    print_word_evaluation_report(
        eval_result,
        verbose=verbose,
        output_dir=out,
        print_to_stdout=print_report,
    )

    if plot:
        figs = plot_evaluation_figures(eval_result, out / "figures")
        eval_result["figures"].update(figs)

    if len(shifted_encoded_by_shift) >= 2:
        from braindecoding.evaluation.subspace_coherence_debiasing_evaluation import (
            run_subspace_coherence_debiasing_sweep,
            save_subspace_encoded_cache,
        )

        saved_cache_pt = save_subspace_encoded_cache(
            out,
            split=split,
            checkpoint_sha256=eval_result["checkpoint"]["sha256"],
            dataset=dataset,
            stage=stage,
            seed=seed,
            exp_tag=resolved_exp_tag,
            word_exp_tag=resolved_word_exp_tag,
            eval_ds=eval_ds,
            shifts=[sec for sec, _, _, _ in shifted_encoded_by_shift],
            vocabulary=vocabulary,
            vocab50=vocab50,
            clean_encoded=encoded,
            shifted_encoded_by_shift=shifted_encoded_by_shift,
        )
        if saved_cache_pt is not None:
            eval_result["subspace_encoded_cache_pt"] = str(saved_cache_pt)

        if cms_enabled:
            from braindecoding.evaluation.common_mode_subtraction_evaluation import (
                run_common_mode_subtraction_sweep,
            )

            cms_report = run_common_mode_subtraction_sweep(
                dataset=dataset,
                stage=stage,
                split=split,
                seed=seed,
                exp_tag=resolved_exp_tag,
                word_exp_tag=resolved_word_exp_tag,
                clean_encoded=encoded,
                shifted_encoded_by_shift=shifted_encoded_by_shift,
                eval_ds=eval_ds,
                shifted_ds_by_shift=shifted_ds_by_shift,
                vocabulary=vocabulary,
                vocab50=vocab50,
                cfg=cfg,
                eta_values=cms_cfg.get("eta_values", (0.5, 0.8, 1.0)),
                normalize_inputs=bool(
                    cms_cfg.get("normalize_before_subtraction", True)
                ),
                baseline_clean_summary=negative_controls_result[
                    "clean_reference"
                ],
                baseline_condition_entries=negative_controls_result[
                    "conditions"
                ],
                baseline_shifted_aggregate=negative_controls_result[
                    "shifted_aggregate"
                ],
                compute_metrics_fn=_compute_encoded_evaluation_metrics,
                extract_summary_row_fn=_extract_control_summary_row,
                summarize_shifted_fn=_summarize_shifted_conditions,
                output_dir=out,
                plot=plot,
                print_report=print_report,
            )
            eval_result["exploratory_common_mode_subtraction_json"] = (
                cms_report.get("exploratory_json")
            )
            eval_result["exploratory_common_mode_subtraction_summary"] = {
                "eta_grid": cms_report["eta_grid"],
                "best_eta_by_clean_vocab50_top10": cms_report[
                    "best_eta_by_clean_vocab50_top10"
                ],
                "best_eta_by_clean_minus_shifted_vocab50_top10": cms_report[
                    "best_eta_by_clean_minus_shifted_vocab50_top10"
                ],
            }
            eval_result["figures"].update(cms_report.get("figures", {}))

        if sub_enabled:
            gate_raw = sub_cfg.get("use_coherence_gate", "both")
            if isinstance(gate_raw, bool):
                gate_modes = [False, True]
            else:
                g_str = str(gate_raw).strip().lower()
                if g_str in {"true", "on", "1", "yes"}:
                    gate_modes = [True]
                elif g_str in {"false", "off", "0", "no"}:
                    gate_modes = [False]
                else:
                    gate_modes = [False, True]

            sub_report = run_subspace_coherence_debiasing_sweep(
                dataset=dataset,
                stage=stage,
                split=split,
                seed=seed,
                exp_tag=resolved_exp_tag,
                word_exp_tag=resolved_word_exp_tag,
                clean_encoded=encoded,
                shifted_encoded_by_shift=shifted_encoded_by_shift,
                eval_ds=eval_ds,
                vocabulary=vocabulary,
                vocab50=vocab50,
                eta_values=sub_cfg.get("eta_values", (0.8, 1.0, 1.2)),
                rank_k_values=sub_cfg.get("rank_k_values", (8, 12, 16, 24)),
                coherence_gate_modes=gate_modes,
                full_loo_etas=sub_cfg.get(
                    "full_loo_eta_values", (0.5, 0.8, 1.0)
                ),
                center_svd=bool(sub_cfg.get("center_svd", True)),
                coherence_low=float(sub_cfg.get("coherence_low", 0.35)),
                coherence_high=float(sub_cfg.get("coherence_high", 0.65)),
                gate_min=float(sub_cfg.get("gate_min", 0.2)),
                gate_max=float(sub_cfg.get("gate_max", 1.2)),
                normalize_inputs=bool(
                    sub_cfg.get("normalize_before_subtraction", True)
                ),
                output_dir=out,
                plot=plot,
                print_report=print_report,
                verbose=verbose,
            )
            eval_result["exploratory_subspace_coherence_debiasing_json"] = (
                sub_report.get("exploratory_json")
            )
            eval_result["exploratory_subspace_coherence_debiasing_summary"] = {
                "grid": sub_report.get("grid"),
                "best_by_net_gap_vocab50": (
                    sub_report.get("best_by_net_gap_vocab50") or {}
                ).get("label"),
                "best_by_clean_preservation_vocab50": (
                    sub_report.get("best_by_clean_preservation_vocab50") or {}
                ).get("label"),
            }
            eval_result["figures"].update(sub_report.get("figures", {}))

        shifted_encoded_by_shift.clear()
        shifted_ds_by_shift.clear()

    eval_result["exported_files"] = export_word_evaluation_tables_and_markdown(
        eval_result, out
    )
    eval_json_path = out / f"evaluation_{split}.json"
    save_json(eval_json_path, eval_result)
    eval_result["evaluation_json"] = str(eval_json_path)

    del model, eval_ds, eval_loader, checkpoint, encoded
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return eval_result


evaluate_single_stage = _evaluate_single_stage


def _parse_seed_list(
    raw_seed: int | str | Iterable[int | str] | None,
) -> list[int]:
    """解析单种子或多种子输入（支持整数、逗号分隔字符串如 '0,1,2' 或序列如 [0, 1, 2]）。"""
    if raw_seed is None:
        return [0]
    if isinstance(raw_seed, bool):
        return [int(raw_seed)]
    if isinstance(raw_seed, int):
        return [int(raw_seed)]
    if isinstance(raw_seed, str):
        cleaned = raw_seed.strip().strip("[]()")
        if not cleaned:
            return [0]
        tokens = [
            tok.strip()
            for tok in cleaned.replace(";", ",").replace(" ", ",").split(",")
            if tok.strip()
        ]
    else:
        tokens = []
        for item in raw_seed:
            if isinstance(item, str):
                tokens.extend(
                    tok.strip()
                    for tok in item.strip("[]()")
                    .replace(";", ",")
                    .replace(" ", ",")
                    .split(",")
                    if tok.strip()
                )
            else:
                tokens.append(item)
    seeds: list[int] = []
    for tok in tokens:
        val = int(tok)
        if val not in seeds:
            seeds.append(val)
    return seeds or [0]


_CONTROL_METRIC_REGISTRY: dict[str, dict[str, object]] = {
    "vocab50_macro_top10": {
        "key": "vocab50_macro_top10",
        "horizon_key": "common_vocab50_macro_top10",
        "label": "50-Word Word-Macro Top-10",
        "short_label": "50w Word-Macro Top-10",
        "is_ratio": True,
    },
    "vocab50_micro_top10": {
        "key": "vocab50_micro_top10",
        "horizon_key": "common_vocab50_micro_top10",
        "label": "50-Word Event-Micro Top-10",
        "short_label": "50w Event-Micro Top-10",
        "is_ratio": True,
    },
    "vocab50_subject_macro_top10": {
        "key": "vocab50_subject_macro_top10",
        "horizon_key": "common_vocab50_subject_macro_top10",
        "label": "50-Word Subject-Macro Top-10",
        "short_label": "50w Subj-Macro Top-10",
        "is_ratio": True,
    },
    "vocab50_macro_top5": {
        "key": "vocab50_macro_top5",
        "horizon_key": "common_vocab50_macro_top10",
        "label": "50-Word Word-Macro Top-5",
        "short_label": "50w Word-Macro Top-5",
        "is_ratio": True,
    },
    "vocab50_macro_top1": {
        "key": "vocab50_macro_top1",
        "horizon_key": "common_vocab50_macro_top1",
        "label": "50-Word Word-Macro Top-1",
        "short_label": "50w Word-Macro Top-1",
        "is_ratio": True,
    },
    "vocab50_micro_top1": {
        "key": "vocab50_micro_top1",
        "horizon_key": "common_vocab50_micro_top1",
        "label": "50-Word Event-Micro Top-1",
        "short_label": "50w Event-Micro Top-1",
        "is_ratio": True,
    },
    "vocab50_mrr": {
        "key": "vocab50_mrr",
        "horizon_key": "common_vocab50_mrr",
        "label": "50-Word MRR",
        "short_label": "50w MRR",
        "is_ratio": True,
    },
    "vocab50_median_rank": {
        "key": "vocab50_median_rank",
        "horizon_key": "common_vocab50_median_rank",
        "label": "50-Word Median Rank",
        "short_label": "50w MedRank",
        "is_ratio": False,
    },
    "full_vocab_macro_top10": {
        "key": "full_vocab_macro_top10",
        "horizon_key": "common_full_vocab_macro_top10",
        "label": "Full-Vocab Word-Macro Top-10",
        "short_label": "FullVoc Word-Macro Top-10",
        "is_ratio": True,
    },
    "full_vocab_subject_macro_top10": {
        "key": "full_vocab_subject_macro_top10",
        "horizon_key": "common_full_vocab_subject_macro_top10",
        "label": "Full-Vocab Subject-Macro Top-10",
        "short_label": "FullVoc Subj-Macro Top-10",
        "is_ratio": True,
    },
    "full_vocab_micro_top10": {
        "key": "full_vocab_micro_top10",
        "horizon_key": "common_full_vocab_micro_top10",
        "label": "Full-Vocab Event-Micro Top-10",
        "short_label": "FullVoc Event-Micro Top-10",
        "is_ratio": True,
    },
    "full_vocab_macro_top1": {
        "key": "full_vocab_macro_top1",
        "horizon_key": "common_full_vocab_macro_top1",
        "label": "Full-Vocab Word-Macro Top-1",
        "short_label": "FullVoc Word-Macro Top-1",
        "is_ratio": True,
    },
    "full_vocab_micro_top1": {
        "key": "full_vocab_micro_top1",
        "horizon_key": "common_full_vocab_micro_top1",
        "label": "Full-Vocab Event-Micro Top-1",
        "short_label": "FullVoc Event-Micro Top-1",
        "is_ratio": True,
    },
    "full_vocab_mrr": {
        "key": "full_vocab_mrr",
        "horizon_key": "common_full_vocab_mrr",
        "label": "Full-Vocab MRR",
        "short_label": "FullVoc MRR",
        "is_ratio": True,
    },
    "full_vocab_median_rank": {
        "key": "full_vocab_median_rank",
        "horizon_key": "common_full_vocab_median_rank",
        "label": "Full-Vocab Median Rank",
        "short_label": "FullVoc MedRank",
        "is_ratio": False,
    },
}

_CONTROL_METRIC_ALIASES: dict[str, str] = {
    "50w_top10": "vocab50_macro_top10",
    "n50_top10": "vocab50_macro_top10",
    "50w_macro_top10": "vocab50_macro_top10",
    "50w_word_macro_top10": "vocab50_macro_top10",
    "vocab50_word_macro_top10": "vocab50_macro_top10",
    "n50_macro_top10": "vocab50_macro_top10",
    "vocab50_top10": "vocab50_macro_top10",
    "50w_micro_top10": "vocab50_micro_top10",
    "50w_event_micro_top10": "vocab50_micro_top10",
    "vocab50_event_micro_top10": "vocab50_micro_top10",
    "n50_micro_top10": "vocab50_micro_top10",
    "50w_subject_macro_top10": "vocab50_subject_macro_top10",
    "n50_subject_macro_top10": "vocab50_subject_macro_top10",
    "50w_top5": "vocab50_macro_top5",
    "n50_top5": "vocab50_macro_top5",
    "50w_macro_top5": "vocab50_macro_top5",
    "n50_macro_top5": "vocab50_macro_top5",
    "50w_top1": "vocab50_macro_top1",
    "n50_top1": "vocab50_macro_top1",
    "50w_macro_top1": "vocab50_macro_top1",
    "50w_word_macro_top1": "vocab50_macro_top1",
    "vocab50_word_macro_top1": "vocab50_macro_top1",
    "n50_macro_top1": "vocab50_macro_top1",
    "vocab50_top1": "vocab50_macro_top1",
    "50w_micro_top1": "vocab50_micro_top1",
    "50w_event_micro_top1": "vocab50_micro_top1",
    "vocab50_event_micro_top1": "vocab50_micro_top1",
    "n50_micro_top1": "vocab50_micro_top1",
    "50w_mrr": "vocab50_mrr",
    "n50_mrr": "vocab50_mrr",
    "50w_median_rank": "vocab50_median_rank",
    "n50_median_rank": "vocab50_median_rank",
    "full_top10": "full_vocab_macro_top10",
    "fv_top10": "full_vocab_macro_top10",
    "full_macro_top10": "full_vocab_macro_top10",
    "fv_macro_top10": "full_vocab_macro_top10",
    "word_macro_top10": "full_vocab_macro_top10",
    "full_vocab_word_macro_top10": "full_vocab_macro_top10",
    "full_vocab_top10": "full_vocab_macro_top10",
    "subject_macro_top10": "full_vocab_subject_macro_top10",
    "subj_macro_top10": "full_vocab_subject_macro_top10",
    "full_subject_macro_top10": "full_vocab_subject_macro_top10",
    "full_micro_top10": "full_vocab_micro_top10",
    "full_vocab_event_micro_top10": "full_vocab_micro_top10",
    "fv_micro_top10": "full_vocab_micro_top10",
    "full_top1": "full_vocab_macro_top1",
    "fv_top1": "full_vocab_macro_top1",
    "full_macro_top1": "full_vocab_macro_top1",
    "fv_macro_top1": "full_vocab_macro_top1",
    "word_macro_top1": "full_vocab_macro_top1",
    "full_vocab_word_macro_top1": "full_vocab_macro_top1",
    "full_vocab_top1": "full_vocab_macro_top1",
    "full_micro_top1": "full_vocab_micro_top1",
    "full_vocab_event_micro_top1": "full_vocab_micro_top1",
    "fv_micro_top1": "full_vocab_micro_top1",
    "full_mrr": "full_vocab_mrr",
    "fv_mrr": "full_vocab_mrr",
    "full_median_rank": "full_vocab_median_rank",
    "fv_median_rank": "full_vocab_median_rank",
}


def _resolve_control_metric_spec(
    control_metric: str | Sequence[str] | None = None,
) -> dict[str, object]:
    """解析用户显式指定的单个负对照核心评估指标（若传入多个指标则返回首个主指标）。"""
    specs = _resolve_control_metric_specs(control_metric)
    return specs[0]


def _resolve_control_metric_specs(
    control_metric: str | Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """解析用户显式指定的一个或多个负对照核心评估指标（支持逗号/分号分隔、列表、'all' 及 k0_/k1_/k2_ 视界前缀）。"""
    if control_metric is None:
        raw_tokens = ["vocab50_macro_top10"]
    elif isinstance(control_metric, str):
        cleaned = control_metric.strip()
        if not cleaned or cleaned.lower() == "auto":
            raw_tokens = ["vocab50_macro_top10"]
        else:
            raw_tokens = [
                t.strip()
                for t in cleaned.replace(";", ",").replace("|", ",").split(",")
                if t.strip()
            ]
    elif isinstance(control_metric, Sequence):
        raw_tokens = []
        for item in control_metric:
            if isinstance(item, str):
                raw_tokens.extend(
                    t.strip()
                    for t in item.replace(";", ",").replace("|", ",").split(",")
                    if t.strip()
                )
            elif item is not None:
                raw_tokens.append(str(item).strip())
    else:
        raw_tokens = [str(control_metric).strip()]

    if not raw_tokens:
        raw_tokens = ["vocab50_macro_top10"]

    expanded_tokens: list[str] = []
    for tok in raw_tokens:
        low = tok.lower().replace("-", "_").replace(" ", "_")
        if low in ("all", "all_metrics", "*"):
            expanded_tokens.extend(
                [
                    "vocab50_macro_top10",
                    "vocab50_micro_top10",
                    "vocab50_macro_top1",
                    "vocab50_mrr",
                    "full_vocab_macro_top10",
                    "full_vocab_subject_macro_top10",
                    "full_vocab_micro_top10",
                    "full_vocab_macro_top1",
                    "full_vocab_mrr",
                    "k0_vocab50_macro_top10",
                    "k1_vocab50_macro_top10",
                    "k2_vocab50_macro_top10",
                    "k0_full_vocab_macro_top10",
                    "k1_full_vocab_macro_top10",
                    "k2_full_vocab_macro_top10",
                ]
            )
        elif low in ("auto", "default", ""):
            expanded_tokens.append("vocab50_macro_top10")
        else:
            expanded_tokens.append(low)

    specs: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    for raw in expanded_tokens:
        horizon: str | None = None
        base_raw = raw
        for h_prefix, h_name in (("k0_", "K0"), ("k1_", "K1"), ("k2_", "K2")):
            if base_raw.startswith(h_prefix):
                horizon = h_name
                base_raw = base_raw[len(h_prefix) :]
                break
        if base_raw.startswith("common_"):
            base_raw = base_raw[len("common_") :]

        canonical = _CONTROL_METRIC_ALIASES.get(base_raw, base_raw)
        if canonical not in _CONTROL_METRIC_REGISTRY:
            valid_keys = ", ".join(sorted(_CONTROL_METRIC_REGISTRY.keys()))
            raise ValueError(
                f"不支持的负对照指标 control_metric={raw!r}。"
                f"可选指标（亦支持加 k0_/k1_/k2_ 前缀或传 'all'）：{valid_keys}"
            )
        spec = dict(_CONTROL_METRIC_REGISTRY[canonical])
        spec["horizon"] = horizon
        if horizon is not None:
            spec["requested_key"] = f"{horizon.lower()}_{canonical}"
            spec["label"] = f"{horizon} {spec['label']}"
            spec["short_label"] = f"{horizon} {spec['short_label']}"
        else:
            spec["requested_key"] = canonical
        req_k = str(spec["requested_key"])
        if req_k not in seen_keys:
            seen_keys.add(req_k)
            specs.append(spec)

    return specs or [_resolve_control_metric_specs("vocab50_macro_top10")[0]]


def _parse_control_methods_list(
    control_method: str | Sequence[str] | None = None,
) -> list[str]:
    """解析用户显式指定的一个或多个负对照/去偏方法（支持分号/竖线/逗号分隔与列表，且自动保留 subspace:K=8,eta=0.8,gate=off 内部逗号）。"""
    import re as _re

    if control_method is None:
        return ["auto"]
    raw_segments: list[str] = []
    if isinstance(control_method, str):
        raw_segments = [control_method]
    elif isinstance(control_method, Sequence):
        for item in control_method:
            if item is not None and str(item).strip():
                raw_segments.append(str(item).strip())
    else:
        raw_segments = [str(control_method).strip()]

    methods: list[str] = []
    for seg in raw_segments:
        for semi_part in _re.split(r"[;|]+", seg):
            semi_part = semi_part.strip()
            if not semi_part:
                continue
            comma_parts = [p.strip() for p in semi_part.split(",") if p.strip()]
            merged: list[str] = []
            for part in comma_parts:
                low_p = part.lower().replace(" ", "")
                is_subspace_Continuation = bool(
                    _re.match(
                        r"^(?:eta[=_:]|gate[=_:]|nogate|gated|k[=_:]?\d+|rank_k[=_:]?\d+)",
                        low_p,
                    )
                )
                if (
                    merged
                    and is_subspace_Continuation
                    and (
                        "subspace" in merged[-1].lower()
                        or _re.search(r"\bk[=_:]?\d+", merged[-1].lower())
                    )
                    and not low_p.startswith("subspace")
                    and not low_p.startswith("cms")
                ):
                    merged[-1] = f"{merged[-1]},{part}"
                else:
                    merged.append(part)
            for m_item in merged:
                if m_item and m_item not in methods:
                    methods.append(m_item)

    return methods or ["auto"]


def _extract_clean_and_shifted_metric(
    clean_ref: dict,
    sh_agg: dict,
    m_key: str,
) -> tuple[float, float]:
    """从单种子 clean_reference 与 shifted_aggregate 中提取统一口径的 Clean 与 Shifted 值。"""
    if m_key in ("full_vocab_macro_top10", "full_vocab_word_macro_top10"):
        c_val = float(
            clean_ref.get(
                "full_vocab_word_macro_top10",
                clean_ref.get("full_vocab_macro_top10", 0.0),
            )
        )
        s_val = float(
            sh_agg.get(
                "mean_full_vocab_word_macro_top10",
                sh_agg.get("mean_full_vocab_macro_top10", 0.0),
            )
        )
    elif m_key == "full_vocab_subject_macro_top10":
        c_val = float(
            clean_ref.get(
                "full_vocab_subject_macro_top10",
                clean_ref.get("full_vocab_micro_top10", 0.0),
            )
        )
        s_val = float(
            sh_agg.get(
                "mean_full_vocab_subject_macro_top10",
                sh_agg.get("mean_full_vocab_micro_top10", 0.0),
            )
        )
    elif m_key == "vocab50_subject_macro_top10":
        c_val = float(
            clean_ref.get(
                "vocab50_subject_macro_top10",
                clean_ref.get("vocab50_macro_top10", 0.0),
            )
        )
        s_val = float(
            sh_agg.get(
                "mean_vocab50_subject_macro_top10",
                sh_agg.get("mean_vocab50_macro_top10", 0.0),
            )
        )
    else:
        c_val = float(clean_ref.get(m_key, 0.0))
        s_val = float(sh_agg.get(f"mean_{m_key}", 0.0))
    return c_val, s_val


def _aggregate_control_entries_across_seeds(
    entries_with_seeds: Sequence[tuple[int, dict]],
    *,
    method_id: str,
    method_label: str,
    control_type: str = "temporal_shift",
    eta: float | None = None,
    rank_k: int | None = None,
    use_coherence_gate: bool | None = None,
    baseline_raw_agg: dict | None = None,
    full_vocab_size: int = 2020,
    n_subjects: int = 10,
    subject_k2_k0_stats: Mapping[str, Sequence[float]] | None = None,
) -> dict:
    """跨多个随机种子聚合任意负对照或去偏配置（含全量检索指标、逐移位条件、K0/K1/K2 视界及显著性检验 p 值）。"""
    valid = [
        (int(sd), ent)
        for sd, ent in entries_with_seeds
        if isinstance(ent, dict)
        and isinstance(ent.get("clean_reference"), dict)
        and isinstance(ent.get("shifted_aggregate"), dict)
        and ent.get("shifted_aggregate")
    ]
    if not valid:
        return {"enabled": False, "method_id": method_id, "label": method_label}

    first_ent = valid[0][1]
    first_sh_agg = first_ent.get("shifted_aggregate") or {}
    shift_secs = list(first_sh_agg.get("shift_seconds") or [])
    shift_cnt = int(first_sh_agg.get("shift_count", len(shift_secs)))

    agg: dict[str, object] = {
        "enabled": True,
        "method_id": method_id,
        "active_method_id": method_id,
        "active_control_method": method_id,
        "label": method_label,
        "active_method_label": method_label,
        "active_control_method_label": method_label,
        "control_type": control_type,
        "eta": float(eta) if eta is not None else None,
        "rank_k": int(rank_k) if rank_k is not None else None,
        "use_coherence_gate": (
            bool(use_coherence_gate) if use_coherence_gate is not None else None
        ),
        "seed_count": len(valid),
        "seeds": [sd for sd, _ in valid],
        "shift_count": shift_cnt,
        "shift_seconds": shift_secs,
    }

    metric_keys = (
        "vocab50_macro_top10",
        "vocab50_micro_top10",
        "vocab50_subject_macro_top10",
        "vocab50_macro_top5",
        "vocab50_macro_top1",
        "vocab50_micro_top1",
        "vocab50_mrr",
        "vocab50_median_rank",
        "full_vocab_macro_top10",
        "full_vocab_word_macro_top10",
        "full_vocab_subject_macro_top10",
        "full_vocab_micro_top10",
        "full_vocab_macro_top1",
        "full_vocab_micro_top1",
        "full_vocab_mrr",
        "full_vocab_median_rank",
    )
    per_seed_by_metric: dict[str, list[dict[str, float | int]]] = {}
    for m_key in metric_keys:
        clean_list: list[float] = []
        sh_list: list[float] = []
        seed_rows_m: list[dict[str, float | int]] = []
        for sd, ent in valid:
            c_ref = ent.get("clean_reference") or {}
            s_agg = ent.get("shifted_aggregate") or {}
            c_val, s_val = _extract_clean_and_shifted_metric(c_ref, s_agg, m_key)
            clean_list.append(c_val)
            sh_list.append(s_val)
            seed_rows_m.append(
                {
                    "seed": sd,
                    "clean": c_val,
                    "shifted": s_val,
                    "gap": float(c_val - s_val),
                }
            )
        clean_vals = np.asarray(clean_list, dtype=np.float64)
        sh_vals = np.asarray(sh_list, dtype=np.float64)
        gap_vals = clean_vals - sh_vals
        agg[f"mean_clean_{m_key}"] = float(np.mean(clean_vals))
        agg[f"std_clean_{m_key}"] = float(np.std(clean_vals, ddof=0))
        agg[f"mean_shifted_{m_key}"] = float(np.mean(sh_vals))
        agg[f"std_shifted_{m_key}"] = float(np.std(sh_vals, ddof=0))
        agg[f"mean_gap_{m_key}"] = float(np.mean(gap_vals))
        agg[f"std_gap_{m_key}"] = float(np.std(gap_vals, ddof=0))
        per_seed_by_metric[m_key] = seed_rows_m
    agg["per_seed_metrics"] = per_seed_by_metric

    # Per-condition (individual shifts / zero_signal) multi-seed aggregation
    conds_first = (
        first_ent.get("conditions")
        or first_ent.get("shifted_conditions")
        or []
    )
    conditions_agg: list[dict[str, object]] = []
    for cond_template in conds_first:
        if not isinstance(cond_template, dict):
            continue
        c_key = str(
            cond_template.get("key") or cond_template.get("condition") or ""
        )
        if not c_key:
            continue
        matched_summaries: list[tuple[dict, dict]] = []
        for _, ent in valid:
            c_ref = ent.get("clean_reference") or {}
            ent_conds = (
                ent.get("conditions") or ent.get("shifted_conditions") or ()
            )
            for c_item in ent_conds:
                if not isinstance(c_item, dict):
                    continue
                item_k = str(c_item.get("key") or c_item.get("condition") or "")
                if item_k == c_key:
                    c_sum = c_item.get("summary")
                    if not isinstance(c_sum, dict):
                        c_sum = c_item
                    matched_summaries.append((c_ref, c_sum))
                    break
        if not matched_summaries:
            continue
        cond_entry: dict[str, object] = {
            "key": c_key,
            "condition": c_key,
            "label": str(cond_template.get("label", c_key)),
            "control_type": str(
                cond_template.get("control_type", control_type)
            ),
            "shift_seconds": cond_template.get("shift_seconds"),
            "seed_count": len(matched_summaries),
        }
        for m_key in metric_keys:
            c_arr = np.asarray(
                [
                    float(
                        cs.get(
                            "full_vocab_word_macro_top10",
                            cs.get("full_vocab_macro_top10", 0.0),
                        )
                        if m_key == "full_vocab_macro_top10"
                        else (
                            cs.get(
                                "vocab50_subject_macro_top10",
                                cs.get("vocab50_macro_top10", 0.0),
                            )
                            if m_key == "vocab50_subject_macro_top10"
                            else cs.get(m_key, 0.0)
                        )
                    )
                    for _, cs in matched_summaries
                ],
                dtype=np.float64,
            )
            ref_arr = np.asarray(
                [
                    float(
                        cr.get(
                            "full_vocab_word_macro_top10",
                            cr.get("full_vocab_macro_top10", 0.0),
                        )
                        if m_key == "full_vocab_macro_top10"
                        else (
                            cr.get(
                                "vocab50_subject_macro_top10",
                                cr.get("vocab50_macro_top10", 0.0),
                            )
                            if m_key == "vocab50_subject_macro_top10"
                            else cr.get(m_key, 0.0)
                        )
                    )
                    for cr, _ in matched_summaries
                ],
                dtype=np.float64,
            )
            delta_arr = c_arr - ref_arr
            gap_arr = ref_arr - c_arr
            cond_entry[f"mean_{m_key}"] = float(np.mean(c_arr))
            cond_entry[f"std_{m_key}"] = float(np.std(c_arr, ddof=0))
            cond_entry[f"mean_delta_vs_clean_{m_key}"] = float(
                np.mean(delta_arr)
            )
            cond_entry[f"std_delta_vs_clean_{m_key}"] = float(
                np.std(delta_arr, ddof=0)
            )
            cond_entry[f"mean_gap_{m_key}"] = float(np.mean(gap_arr))
            cond_entry[f"std_gap_{m_key}"] = float(np.std(gap_arr, ddof=0))
        conditions_agg.append(cond_entry)
    agg["conditions_aggregate"] = conditions_agg
    agg["per_shift_aggregate"] = conditions_agg

    # Location 1 statistical p-values:
    # 1) Control Mean vs. Theoretical Chance (1-sample t-test across the M shifted anchors)
    # 2) Net Gap (Clean - Control) paired significance test
    eff_subj_df = float(max(int(n_subjects or 10) - 1, 9))
    eff_shift_df = float(max(len(conditions_agg) - 1, shift_cnt - 1, 5))
    for m_key in metric_keys:
        chance_v = _theoretical_chance_for_metric(
            m_key, full_vocab_size=full_vocab_size
        )
        agg[f"chance_{m_key}"] = chance_v
        cond_sh_vals = [
            float(ce[f"mean_{m_key}"])
            for ce in conditions_agg
            if ce.get(f"mean_{m_key}") is not None
        ]
        cond_gap_vals = [
            float(ce[f"mean_gap_{m_key}"])
            for ce in conditions_agg
            if ce.get(f"mean_gap_{m_key}") is not None
        ]
        seed_rows_m = per_seed_by_metric.get(m_key, [])
        seed_sh_vals = [float(r["shifted"]) for r in seed_rows_m]
        seed_gap_vals = [float(r["gap"]) for r in seed_rows_m]
        sh_std_fb = float(
            first_sh_agg.get(
                f"std_full_vocab_word_macro_top10"
                if m_key == "full_vocab_macro_top10"
                else f"std_{m_key}",
                first_sh_agg.get(f"std_{m_key}", 0.0),
            )
            or 0.0
        )
        se_seed_sh = (
            float(np.std(seed_sh_vals, ddof=1)) / math.sqrt(float(len(seed_sh_vals)))
            if len(seed_sh_vals) >= 2
            else 0.0
        )
        se_seed_gap = (
            float(np.std(seed_gap_vals, ddof=1)) / math.sqrt(float(len(seed_gap_vals)))
            if len(seed_gap_vals) >= 2
            else 0.0
        )
        if chance_v is not None:
            if len(cond_sh_vals) >= 2:
                p_sh_chance = _one_sample_t_test_pvalue(
                    cond_sh_vals,
                    popmean=chance_v,
                    min_se=se_seed_sh,
                    min_df=eff_shift_df,
                )
            else:
                p_sh_chance = _one_sample_t_test_pvalue(
                    seed_sh_vals,
                    popmean=chance_v,
                    fallback_std=sh_std_fb,
                    fallback_n=max(shift_cnt, 6),
                    min_df=eff_shift_df,
                )
            agg[f"p_shifted_vs_chance_{m_key}"] = float(p_sh_chance)
            agg[f"sig_shifted_vs_chance_{m_key}"] = _significance_stars(p_sh_chance)
            agg[f"p_str_shifted_vs_chance_{m_key}"] = _format_p_value_with_stars(
                p_sh_chance
            )
        if len(cond_gap_vals) >= 2:
            p_gap_val = _one_sample_t_test_pvalue(
                cond_gap_vals,
                popmean=0.0,
                min_se=se_seed_gap,
                min_df=max(eff_subj_df, eff_shift_df),
            )
        else:
            p_gap_val = _one_sample_t_test_pvalue(
                seed_gap_vals,
                popmean=0.0,
                fallback_std=sh_std_fb,
                fallback_n=max(shift_cnt, 6),
                min_df=max(eff_subj_df, eff_shift_df),
            )
        agg[f"p_gap_{m_key}"] = float(p_gap_val)
        agg[f"sig_gap_{m_key}"] = _significance_stars(p_gap_val)
        agg[f"p_str_gap_{m_key}"] = _format_p_value_with_stars(p_gap_val)

    # Per-horizon (K0, K1, K2, K1-K0, K2-K0) multi-seed aggregation under control
    horizon_metric_keys = (
        "common_vocab50_macro_top10",
        "common_vocab50_micro_top10",
        "common_vocab50_macro_top1",
        "common_vocab50_micro_top1",
        "common_vocab50_mrr",
        "common_vocab50_median_rank",
        "common_full_vocab_macro_top10",
        "common_full_vocab_word_macro_top10",
        "common_full_vocab_subject_macro_top10",
        "common_full_vocab_micro_top10",
        "common_full_vocab_macro_top1",
        "common_full_vocab_micro_top1",
        "common_full_vocab_mrr",
        "common_full_vocab_median_rank",
    )
    horizons_agg: dict[str, dict] = {}
    for h_key in ("K0", "K1", "K2"):
        h_pairs: list[tuple[int, dict, dict]] = []
        for sd, ent in valid:
            c_hz = ((ent.get("clean_reference") or {}).get("horizons") or {}).get(
                h_key
            )
            s_hz = (
                (ent.get("shifted_aggregate") or {}).get("horizons") or {}
            ).get(h_key)
            if isinstance(c_hz, dict) and isinstance(s_hz, dict):
                h_pairs.append((sd, c_hz, s_hz))
        if not h_pairs:
            continue
        clean_nested: dict[str, object] = {}
        shifted_nested: dict[str, object] = {}
        gap_nested: dict[str, object] = {}
        h_item: dict[str, object] = {
            "horizon": h_key,
            "seed_count": len(h_pairs),
            "per_seed_metrics": {},
            "clean": clean_nested,
            "shifted_mean": shifted_nested,
            "net_gap": gap_nested,
        }
        for hm_key in horizon_metric_keys:
            c_vals_list: list[float] = []
            s_vals_list: list[float] = []
            s_stds_list: list[float] = []
            hm_seed_rows: list[dict[str, float | int]] = []
            for sd, c_hz, s_hz in h_pairs:
                if hm_key in (
                    "common_full_vocab_macro_top10",
                    "common_full_vocab_word_macro_top10",
                ):
                    cv = float(
                        c_hz.get(
                            "common_full_vocab_word_macro_top10",
                            c_hz.get("common_full_vocab_macro_top10", 0.0),
                        )
                    )
                    sv = float(
                        s_hz.get(
                            "mean_common_full_vocab_word_macro_top10",
                            s_hz.get("mean_common_full_vocab_macro_top10", 0.0),
                        )
                    )
                    s_sd = float(
                        s_hz.get(
                            "std_common_full_vocab_word_macro_top10",
                            s_hz.get("std_common_full_vocab_macro_top10", 0.0),
                        )
                    )
                else:
                    cv = float(c_hz.get(hm_key, 0.0))
                    sv = float(s_hz.get(f"mean_{hm_key}", 0.0))
                    s_sd = float(s_hz.get(f"std_{hm_key}", 0.0))
                c_vals_list.append(cv)
                s_vals_list.append(sv)
                s_stds_list.append(s_sd)
                hm_seed_rows.append(
                    {
                        "seed": sd,
                        "clean": cv,
                        "shifted": sv,
                        "gap": float(cv - sv),
                    }
                )
            c_arr = np.asarray(c_vals_list, dtype=np.float64)
            s_arr = np.asarray(s_vals_list, dtype=np.float64)
            g_arr = c_arr - s_arr
            mc, sc = float(np.mean(c_arr)), float(np.std(c_arr, ddof=0))
            ms, ss = float(np.mean(s_arr)), float(np.std(s_arr, ddof=0))
            mg, sg = float(np.mean(g_arr)), float(np.std(g_arr, ddof=0))
            h_item[f"mean_clean_{hm_key}"] = mc
            h_item[f"std_clean_{hm_key}"] = sc
            h_item[f"mean_shifted_{hm_key}"] = ms
            h_item[f"std_shifted_{hm_key}"] = ss
            h_item[f"mean_gap_{hm_key}"] = mg
            h_item[f"std_gap_{hm_key}"] = sg
            clean_nested[f"mean_{hm_key}"] = mc
            clean_nested[f"std_{hm_key}"] = sc
            shifted_nested[f"mean_{hm_key}"] = ms
            shifted_nested[f"std_{hm_key}"] = ss
            gap_nested[f"mean_{hm_key}"] = mg
            gap_nested[f"std_{hm_key}"] = sg
            h_item["per_seed_metrics"][hm_key] = hm_seed_rows
            hm_chance = _theoretical_chance_for_metric(
                hm_key, full_vocab_size=full_vocab_size
            )
            cond_s_vals_h: list[float] = []
            cond_g_vals_h: list[float] = []
            for cond_template in conds_first:
                if not isinstance(cond_template, dict):
                    continue
                sk = str(
                    cond_template.get("key")
                    or cond_template.get("condition")
                    or ""
                )
                if not sk:
                    continue
                per_seed_sh_h: list[float] = []
                per_seed_gp_h: list[float] = []
                for _, ent in valid:
                    c_hz_m = ((ent.get("clean_reference") or {}).get("horizons") or {}).get(h_key) or {}
                    ent_conds = ent.get("conditions") or ent.get("shifted_conditions") or ()
                    for c_item in ent_conds:
                        if isinstance(c_item, dict) and str(c_item.get("key") or c_item.get("condition") or "") == sk:
                            c_sum = c_item.get("summary")
                            if not isinstance(c_sum, dict):
                                c_sum = c_item
                            sh_hz_m = (c_sum.get("horizons") or {}).get(h_key) or {}
                            if sh_hz_m:
                                if hm_key in (
                                    "common_full_vocab_macro_top10",
                                    "common_full_vocab_word_macro_top10",
                                ):
                                    cv_i = float(
                                        c_hz_m.get(
                                            "common_full_vocab_word_macro_top10",
                                            c_hz_m.get("common_full_vocab_macro_top10", mc),
                                        )
                                    )
                                    sv_i = float(
                                        sh_hz_m.get(
                                            "common_full_vocab_word_macro_top10",
                                            sh_hz_m.get("common_full_vocab_macro_top10", 0.0),
                                        )
                                    )
                                else:
                                    cv_i = float(c_hz_m.get(hm_key, mc))
                                    sv_i = float(sh_hz_m.get(hm_key, 0.0))
                                per_seed_sh_h.append(sv_i)
                                per_seed_gp_h.append(cv_i - sv_i)
                            break
                if per_seed_sh_h:
                    cond_s_vals_h.append(float(np.mean(per_seed_sh_h)))
                    cond_g_vals_h.append(float(np.mean(per_seed_gp_h)))
            fb_sd_h = (
                float(np.std(cond_s_vals_h, ddof=1))
                if len(cond_s_vals_h) >= 2
                else (float(np.mean(s_stds_list)) if s_stds_list else 0.0)
            )
            if hm_chance is not None:
                p_h_sh = _one_sample_t_test_pvalue(
                    cond_s_vals_h if len(cond_s_vals_h) >= 2 else s_arr,
                    popmean=hm_chance,
                    fallback_std=fb_sd_h,
                    fallback_n=max(shift_cnt, 6),
                    min_df=eff_shift_df,
                )
                h_item[f"p_shifted_vs_chance_{hm_key}"] = float(p_h_sh)
                h_item[f"p_str_shifted_vs_chance_{hm_key}"] = (
                    _format_p_value_with_stars(p_h_sh)
                )
            p_h_gap = _one_sample_t_test_pvalue(
                cond_g_vals_h if len(cond_g_vals_h) >= 2 else g_arr,
                popmean=0.0,
                fallback_std=fb_sd_h,
                fallback_n=max(shift_cnt, 6),
                min_df=max(eff_subj_df, eff_shift_df),
            )
            h_item[f"p_gap_{hm_key}"] = float(p_h_gap)
            h_item[f"p_str_gap_{hm_key}"] = _format_p_value_with_stars(p_h_gap)
        horizons_agg[h_key] = h_item

    for h_from, h_to, diff_key in (
        ("K0", "K1", "K1_minus_K0"),
        ("K0", "K2", "K2_minus_K0"),
        ("K1", "K2", "K2_minus_K1"),
    ):
        if h_from not in horizons_agg or h_to not in horizons_agg:
            continue
        clean_diff_nested: dict[str, object] = {}
        shifted_diff_nested: dict[str, object] = {}
        gap_diff_nested: dict[str, object] = {}
        diff_item: dict[str, object] = {
            "horizon": diff_key,
            "clean": clean_diff_nested,
            "shifted_mean": shifted_diff_nested,
            "net_gap": gap_diff_nested,
        }
        for hm_key in horizon_metric_keys:
            c_diffs: list[float] = []
            s_diffs: list[float] = []
            g_diffs: list[float] = []
            s_within_stds: list[float] = []
            for _, ent in valid:
                c_hz_map = (
                    (ent.get("clean_reference") or {}).get("horizons") or {}
                )
                s_hz_map = (
                    (ent.get("shifted_aggregate") or {}).get("horizons") or {}
                )
                if h_from not in c_hz_map or h_to not in c_hz_map:
                    continue
                if h_from not in s_hz_map or h_to not in s_hz_map:
                    continue
                if hm_key in (
                    "common_full_vocab_macro_top10",
                    "common_full_vocab_word_macro_top10",
                ):
                    c1 = float(
                        c_hz_map[h_from].get(
                            "common_full_vocab_word_macro_top10",
                            c_hz_map[h_from].get(
                                "common_full_vocab_macro_top10", 0.0
                            ),
                        )
                    )
                    c2 = float(
                        c_hz_map[h_to].get(
                            "common_full_vocab_word_macro_top10",
                            c_hz_map[h_to].get(
                                "common_full_vocab_macro_top10", 0.0
                            ),
                        )
                    )
                    s1 = float(
                        s_hz_map[h_from].get(
                            "mean_common_full_vocab_word_macro_top10",
                            s_hz_map[h_from].get(
                                "mean_common_full_vocab_macro_top10", 0.0
                            ),
                        )
                    )
                    s2 = float(
                        s_hz_map[h_to].get(
                            "mean_common_full_vocab_word_macro_top10",
                            s_hz_map[h_to].get(
                                "mean_common_full_vocab_macro_top10", 0.0
                            ),
                        )
                    )
                    sd1 = float(
                        s_hz_map[h_from].get(
                            "std_common_full_vocab_word_macro_top10",
                            s_hz_map[h_from].get(
                                "std_common_full_vocab_macro_top10", 0.0
                            ),
                        )
                    )
                    sd2 = float(
                        s_hz_map[h_to].get(
                            "std_common_full_vocab_word_macro_top10",
                            s_hz_map[h_to].get(
                                "std_common_full_vocab_macro_top10", 0.0
                            ),
                        )
                    )
                else:
                    c1 = float(c_hz_map[h_from].get(hm_key, 0.0))
                    c2 = float(c_hz_map[h_to].get(hm_key, 0.0))
                    s1 = float(s_hz_map[h_from].get(f"mean_{hm_key}", 0.0))
                    s2 = float(s_hz_map[h_to].get(f"mean_{hm_key}", 0.0))
                    sd1 = float(s_hz_map[h_from].get(f"std_{hm_key}", 0.0))
                    sd2 = float(s_hz_map[h_to].get(f"std_{hm_key}", 0.0))
                c_diffs.append(c2 - c1)
                s_diffs.append(s2 - s1)
                g_diffs.append((c2 - s2) - (c1 - s1))
                s_within_stds.append(0.5 * (sd1 + sd2))

            # Per-shift condition paired differences (K_to - K_from across M=6 shifted anchors)
            cond_s_diffs: list[float] = []
            cond_g_diffs: list[float] = []
            for cond_template in conds_first:
                if not isinstance(cond_template, dict):
                    continue
                c_key = str(
                    cond_template.get("key")
                    or cond_template.get("condition")
                    or ""
                )
                if not c_key:
                    continue
                per_seed_cond_s: list[float] = []
                per_seed_cond_g: list[float] = []
                for _, ent in valid:
                    c_hz_map = (
                        (ent.get("clean_reference") or {}).get("horizons") or {}
                    )
                    if h_from not in c_hz_map or h_to not in c_hz_map:
                        continue
                    ent_conds = (
                        ent.get("conditions")
                        or ent.get("shifted_conditions")
                        or ()
                    )
                    for c_item in ent_conds:
                        if not isinstance(c_item, dict):
                            continue
                        if (
                            str(
                                c_item.get("key")
                                or c_item.get("condition")
                                or ""
                            )
                            == c_key
                        ):
                            c_sum = c_item.get("summary")
                            if not isinstance(c_sum, dict):
                                c_sum = c_item
                            sh_hz_map = c_sum.get("horizons") or {}
                            if h_from in sh_hz_map and h_to in sh_hz_map:
                                if hm_key in (
                                    "common_full_vocab_macro_top10",
                                    "common_full_vocab_word_macro_top10",
                                ):
                                    c1_v = float(
                                        c_hz_map[h_from].get(
                                            "common_full_vocab_word_macro_top10",
                                            c_hz_map[h_from].get(
                                                "common_full_vocab_macro_top10",
                                                0.0,
                                            ),
                                        )
                                    )
                                    c2_v = float(
                                        c_hz_map[h_to].get(
                                            "common_full_vocab_word_macro_top10",
                                            c_hz_map[h_to].get(
                                                "common_full_vocab_macro_top10",
                                                0.0,
                                            ),
                                        )
                                    )
                                    s1_v = float(
                                        sh_hz_map[h_from].get(
                                            "common_full_vocab_word_macro_top10",
                                            sh_hz_map[h_from].get(
                                                "common_full_vocab_macro_top10",
                                                0.0,
                                            ),
                                        )
                                    )
                                    s2_v = float(
                                        sh_hz_map[h_to].get(
                                            "common_full_vocab_word_macro_top10",
                                            sh_hz_map[h_to].get(
                                                "common_full_vocab_macro_top10",
                                                0.0,
                                            ),
                                        )
                                    )
                                else:
                                    c1_v = float(
                                        c_hz_map[h_from].get(hm_key, 0.0)
                                    )
                                    c2_v = float(
                                        c_hz_map[h_to].get(hm_key, 0.0)
                                    )
                                    s1_v = float(
                                        sh_hz_map[h_from].get(hm_key, 0.0)
                                    )
                                    s2_v = float(
                                        sh_hz_map[h_to].get(hm_key, 0.0)
                                    )
                                per_seed_cond_s.append(s2_v - s1_v)
                                per_seed_cond_g.append(
                                    (c2_v - s2_v) - (c1_v - s1_v)
                                )
                            break
                if per_seed_cond_s:
                    cond_s_diffs.append(float(np.mean(per_seed_cond_s)))
                    cond_g_diffs.append(float(np.mean(per_seed_cond_g)))

            if c_diffs:
                ca = np.asarray(c_diffs, dtype=np.float64)
                sa = np.asarray(s_diffs, dtype=np.float64)
                ga = np.asarray(g_diffs, dtype=np.float64)
                mc, sc = float(np.mean(ca)), float(np.std(ca, ddof=0))
                ms, ss = float(np.mean(sa)), float(np.std(sa, ddof=0))
                mg, sg = float(np.mean(ga)), float(np.std(ga, ddof=0))
                diff_item[f"mean_clean_{hm_key}"] = mc
                diff_item[f"std_clean_{hm_key}"] = sc
                diff_item[f"mean_shifted_{hm_key}"] = ms
                diff_item[f"std_shifted_{hm_key}"] = ss
                diff_item[f"mean_gap_{hm_key}"] = mg
                diff_item[f"std_gap_{hm_key}"] = sg
                clean_diff_nested[f"mean_{hm_key}"] = mc
                clean_diff_nested[f"std_{hm_key}"] = sc
                shifted_diff_nested[f"mean_{hm_key}"] = ms
                shifted_diff_nested[f"std_{hm_key}"] = ss
                gap_diff_nested[f"mean_{hm_key}"] = mg
                gap_diff_nested[f"std_{hm_key}"] = sg

                # Location 2 p-values:
                # 1) Clean (0s) gain p-value (using subject-level observations / min_df=n_subjects-1)
                subj_vals_for_hm: Sequence[float] | None = None
                if diff_key == "K2_minus_K0" and isinstance(
                    subject_k2_k0_stats, Mapping
                ):
                    if hm_key == "common_vocab50_macro_top10":
                        subj_vals_for_hm = subject_k2_k0_stats.get(
                            "k2_minus_k0_vocab50_top10"
                        )
                    elif hm_key in (
                        "common_full_vocab_macro_top10",
                        "common_full_vocab_word_macro_top10",
                        "common_full_vocab_subject_macro_top10",
                    ):
                        subj_vals_for_hm = subject_k2_k0_stats.get(
                            "k2_minus_k0_full_vocab_macro_top10"
                        )
                subj_fb_sd = (
                    float(np.std(subj_vals_for_hm, ddof=0))
                    if subj_vals_for_hm and len(subj_vals_for_hm) >= 2
                    else (
                        float(np.mean(s_within_stds))
                        if s_within_stds
                        else 0.0
                    )
                )
                p_clean_gain = _one_sample_t_test_pvalue(
                    ca,
                    popmean=0.0,
                    fallback_std=subj_fb_sd,
                    fallback_n=max(int(n_subjects or 10), 10),
                    min_df=eff_subj_df,
                )
                # 2) Control Mean gain p-value across M=6 shifted anchors & seeds
                se_seed_s_diff = (
                    float(np.std(sa, ddof=1)) / math.sqrt(float(len(sa)))
                    if len(sa) >= 2
                    else 0.0
                )
                fb_sh_sd = (
                    float(np.mean(s_within_stds)) if s_within_stds else 0.0
                )
                if len(cond_s_diffs) >= 2:
                    p_shifted_gain = _one_sample_t_test_pvalue(
                        cond_s_diffs,
                        popmean=0.0,
                        min_se=se_seed_s_diff,
                        min_df=eff_shift_df,
                    )
                else:
                    p_shifted_gain = _one_sample_t_test_pvalue(
                        sa,
                        popmean=0.0,
                        fallback_std=fb_sh_sd,
                        fallback_n=max(shift_cnt, 6),
                        min_df=eff_shift_df,
                    )
                # 3) Net Gap (Clean - Ctrl) gain p-value
                se_seed_g_diff = (
                    float(np.std(ga, ddof=1)) / math.sqrt(float(len(ga)))
                    if len(ga) >= 2
                    else 0.0
                )
                if len(cond_g_diffs) >= 2:
                    p_gap_gain = _one_sample_t_test_pvalue(
                        cond_g_diffs,
                        popmean=0.0,
                        min_se=se_seed_g_diff,
                        min_df=max(eff_subj_df, eff_shift_df),
                    )
                else:
                    p_gap_gain = _one_sample_t_test_pvalue(
                        ga,
                        popmean=0.0,
                        fallback_std=max(subj_fb_sd, fb_sh_sd),
                        fallback_n=max(int(n_subjects or 10), 10),
                        min_df=max(eff_subj_df, eff_shift_df),
                    )

                diff_item[f"p_clean_{hm_key}"] = float(p_clean_gain)
                diff_item[f"sig_clean_{hm_key}"] = _significance_stars(
                    p_clean_gain
                )
                diff_item[f"p_str_clean_{hm_key}"] = _format_p_value_with_stars(
                    p_clean_gain
                )
                clean_diff_nested[f"p_{hm_key}"] = float(p_clean_gain)
                clean_diff_nested[f"sig_{hm_key}"] = _significance_stars(
                    p_clean_gain
                )
                clean_diff_nested[f"p_str_{hm_key}"] = (
                    _format_p_value_with_stars(p_clean_gain)
                )

                diff_item[f"p_shifted_{hm_key}"] = float(p_shifted_gain)
                diff_item[f"sig_shifted_{hm_key}"] = _significance_stars(
                    p_shifted_gain
                )
                diff_item[f"p_str_shifted_{hm_key}"] = (
                    _format_p_value_with_stars(p_shifted_gain)
                )
                shifted_diff_nested[f"p_{hm_key}"] = float(p_shifted_gain)
                shifted_diff_nested[f"sig_{hm_key}"] = _significance_stars(
                    p_shifted_gain
                )
                shifted_diff_nested[f"p_str_{hm_key}"] = (
                    _format_p_value_with_stars(p_shifted_gain)
                )

                diff_item[f"p_gap_{hm_key}"] = float(p_gap_gain)
                diff_item[f"sig_gap_{hm_key}"] = _significance_stars(
                    p_gap_gain
                )
                diff_item[f"p_str_gap_{hm_key}"] = _format_p_value_with_stars(
                    p_gap_gain
                )
                gap_diff_nested[f"p_{hm_key}"] = float(p_gap_gain)
                gap_diff_nested[f"sig_{hm_key}"] = _significance_stars(
                    p_gap_gain
                )
                gap_diff_nested[f"p_str_{hm_key}"] = (
                    _format_p_value_with_stars(p_gap_gain)
                )
        if len(clean_diff_nested) > 0:
            horizons_agg[diff_key] = diff_item
    agg["horizons_aggregate"] = horizons_agg

    if isinstance(baseline_raw_agg, dict) and baseline_raw_agg.get("enabled"):
        delta_vs_raw: dict[str, float] = {}
        for m_key in metric_keys:
            delta_vs_raw[f"mean_delta_clean_{m_key}"] = float(
                float(agg.get(f"mean_clean_{m_key}", 0.0))
                - float(baseline_raw_agg.get(f"mean_clean_{m_key}", 0.0))
            )
            delta_vs_raw[f"mean_delta_shifted_{m_key}"] = float(
                float(agg.get(f"mean_shifted_{m_key}", 0.0))
                - float(baseline_raw_agg.get(f"mean_shifted_{m_key}", 0.0))
            )
            delta_vs_raw[f"mean_delta_gap_{m_key}"] = float(
                float(agg.get(f"mean_gap_{m_key}", 0.0))
                - float(baseline_raw_agg.get(f"mean_gap_{m_key}", 0.0))
            )
        agg["delta_vs_raw_eta0"] = delta_vs_raw

    return agg


def _extract_metric_from_control_agg(
    ctrl_agg: dict,
    metric_spec: dict[str, object],
) -> dict[str, object]:
    """从已聚合的负对照字典中提取指定指标（含可选 K0/K1/K2 视界）的 Clean/Shifted/Gap Mean±SD、显著性 p 值与逐种子明细。"""
    m_key = str(metric_spec["key"])
    hm_key = str(metric_spec["horizon_key"])
    horizon = metric_spec.get("horizon")
    if isinstance(horizon, str) and horizon in (
        ctrl_agg.get("horizons_aggregate") or {}
    ):
        hb = ctrl_agg["horizons_aggregate"][horizon]
        per_seed_list = (hb.get("per_seed_metrics") or {}).get(hm_key, [])
        return {
            "metric_key": str(metric_spec["requested_key"]),
            "base_metric_key": m_key,
            "horizon": horizon,
            "metric_label": str(metric_spec["label"]),
            "short_label": str(metric_spec["short_label"]),
            "is_ratio": bool(metric_spec.get("is_ratio", True)),
            "mean_clean": float(hb.get(f"mean_clean_{hm_key}", 0.0)),
            "std_clean": float(hb.get(f"std_clean_{hm_key}", 0.0)),
            "mean_shifted": float(hb.get(f"mean_shifted_{hm_key}", 0.0)),
            "std_shifted": float(hb.get(f"std_shifted_{hm_key}", 0.0)),
            "mean_gap": float(hb.get(f"mean_gap_{hm_key}", 0.0)),
            "std_gap": float(hb.get(f"std_gap_{hm_key}", 0.0)),
            "chance_value": ctrl_agg.get(f"chance_{m_key}"),
            "p_shifted_vs_chance": hb.get(
                f"p_shifted_vs_chance_{hm_key}",
                ctrl_agg.get(f"p_shifted_vs_chance_{m_key}"),
            ),
            "p_str_shifted_vs_chance": str(
                hb.get(
                    f"p_str_shifted_vs_chance_{hm_key}",
                    ctrl_agg.get(f"p_str_shifted_vs_chance_{m_key}", ""),
                )
                or ""
            ),
            "p_gap": hb.get(f"p_gap_{hm_key}", ctrl_agg.get(f"p_gap_{m_key}")),
            "p_str_gap": str(
                hb.get(
                    f"p_str_gap_{hm_key}",
                    ctrl_agg.get(f"p_str_gap_{m_key}", ""),
                )
                or ""
            ),
            "per_seed": list(per_seed_list),
        }
    per_seed_list = (ctrl_agg.get("per_seed_metrics") or {}).get(m_key, [])
    return {
        "metric_key": str(metric_spec["requested_key"]),
        "base_metric_key": m_key,
        "horizon": None,
        "metric_label": str(metric_spec["label"]),
        "short_label": str(metric_spec["short_label"]),
        "is_ratio": bool(metric_spec.get("is_ratio", True)),
        "mean_clean": float(ctrl_agg.get(f"mean_clean_{m_key}", 0.0)),
        "std_clean": float(ctrl_agg.get(f"std_clean_{m_key}", 0.0)),
        "mean_shifted": float(ctrl_agg.get(f"mean_shifted_{m_key}", 0.0)),
        "std_shifted": float(ctrl_agg.get(f"std_shifted_{m_key}", 0.0)),
        "mean_gap": float(ctrl_agg.get(f"mean_gap_{m_key}", 0.0)),
        "std_gap": float(ctrl_agg.get(f"std_gap_{m_key}", 0.0)),
        "chance_value": ctrl_agg.get(f"chance_{m_key}"),
        "p_shifted_vs_chance": ctrl_agg.get(f"p_shifted_vs_chance_{m_key}"),
        "p_str_shifted_vs_chance": str(
            ctrl_agg.get(f"p_str_shifted_vs_chance_{m_key}", "") or ""
        ),
        "p_gap": ctrl_agg.get(f"p_gap_{m_key}"),
        "p_str_gap": str(ctrl_agg.get(f"p_str_gap_{m_key}", "") or ""),
        "per_seed": list(per_seed_list),
    }


def _select_active_control_method(
    methods_map: dict[str, dict],
    *,
    control_method: str | None = None,
    metric_spec: dict[str, object],
) -> dict:
    """根据用户传入的单个 control_method 选择对应的负对照/去偏聚合结果。"""
    if not methods_map:
        return {"enabled": False}

    raw_sel = str(control_method or "auto").strip()
    norm_sel = raw_sel.lower().replace(" ", "")

    cms_candidates = [
        v
        for k, v in methods_map.items()
        if k.startswith("cms:") and float(v.get("eta") or 0.0) > 1e-9
    ]
    sub_candidates = [
        v
        for k, v in methods_map.items()
        if k.startswith("subspace:") and v.get("rank_k") is not None
    ]

    if not norm_sel or norm_sel in ("auto", "default"):
        if sub_candidates:
            best = max(
                sub_candidates,
                key=lambda x: float(
                    _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
                ),
            )
            return dict(best)
        if cms_candidates:
            best = max(
                cms_candidates,
                key=lambda x: float(
                    _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
                ),
            )
            return dict(best)
        for default_id in ("raw", "zero_signal"):
            if default_id in methods_map:
                return dict(methods_map[default_id])
        return dict(next(iter(methods_map.values())))

    if norm_sel in ("raw", "temporal_shift", "eta=0", "eta=0.0", "eta_0", "raw_eta_0.0"):
        if "raw" in methods_map:
            return dict(methods_map["raw"])

    if norm_sel in ("zero_signal", "zero", "zerosignal"):
        if "zero_signal" in methods_map:
            return dict(methods_map["zero_signal"])

    if norm_sel in methods_map:
        return dict(methods_map[norm_sel])

    for k, v in methods_map.items():
        if k.lower().replace(" ", "") == norm_sel:
            return dict(v)

    if norm_sel in ("cms", "cms:best_gap", "cms_best_gap") and cms_candidates:
        best = max(
            cms_candidates,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
            ),
        )
        return dict(best)
    if norm_sel in ("cms:best_clean", "cms_best_clean") and cms_candidates:
        best = max(
            cms_candidates,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_clean"]
            ),
        )
        return dict(best)

    if (
        norm_sel in ("subspace", "subspace:best_gap", "subspace_best_gap", "best_gap")
        and sub_candidates
    ):
        best = max(
            sub_candidates,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_gap"]
            ),
        )
        return dict(best)
    if (
        norm_sel in ("subspace:best_clean", "subspace_best_clean", "best_clean")
        and sub_candidates
    ):
        raw_base = methods_map.get("raw")
        raw_gap = (
            float(_extract_metric_from_control_agg(raw_base, metric_spec)["mean_gap"])
            if isinstance(raw_base, dict) and raw_base.get("enabled")
            else -1e9
        )
        improving = [
            x
            for x in sub_candidates
            if float(_extract_metric_from_control_agg(x, metric_spec)["mean_gap"])
            > raw_gap
        ]
        pool = improving if improving else sub_candidates
        best = max(
            pool,
            key=lambda x: float(
                _extract_metric_from_control_agg(x, metric_spec)["mean_clean"]
            ),
        )
        return dict(best)

    # Parse eta=... or K=...,eta=...,gate=...
    import re as _re

    k_match = _re.search(r"k[=_:]?(\d+)", norm_sel)
    eta_match = _re.search(r"eta[=_:]?([0-9]+(?:\.[0-9]+)?)", norm_sel)
    gate_val: bool | None = None
    if any(g in norm_sel for g in ("gate=on", "gate=true", "gate_on", "gated")):
        gate_val = True
    elif any(
        g in norm_sel for g in ("gate=off", "gate=false", "gate_off", "nogate")
    ):
        gate_val = False

    if k_match is not None:
        target_k = int(k_match.group(1))
        target_eta = float(eta_match.group(1)) if eta_match else None
        for v in sub_candidates:
            if int(v.get("rank_k") or -1) != target_k:
                continue
            if (
                target_eta is not None
                and abs(float(v.get("eta") or 0.0) - target_eta) > 1e-6
            ):
                continue
            if (
                gate_val is not None
                and bool(v.get("use_coherence_gate")) != gate_val
            ):
                continue
            return dict(v)

    if eta_match is not None and k_match is None:
        target_eta = float(eta_match.group(1))
        if abs(target_eta) <= 1e-9 and "raw" in methods_map:
            return dict(methods_map["raw"])
        for v in cms_candidates:
            if abs(float(v.get("eta") or 0.0) - target_eta) <= 1e-6:
                return dict(v)

    for k, v in methods_map.items():
        if norm_sel in k.lower() or norm_sel in str(v.get("label", "")).lower().replace(" ", ""):
            return dict(v)

    avail = ", ".join(sorted(methods_map.keys()))
    raise ValueError(
        f"未找到匹配的负对照/去偏方法 control_method={control_method!r}。"
        f"当前可用方法：{avail}"
    )


def _select_control_methods_list(
    methods_map: dict[str, dict],
    *,
    control_method: str | Sequence[str] | None = None,
    metric_spec: dict[str, object],
) -> list[dict]:
    """根据用户传入的一个或多个 control_method（含 'all'）返回去重后的负对照/去偏聚合字典列表。"""
    if not methods_map:
        return []
    tokens = _parse_control_methods_list(control_method)
    selected_list: list[dict] = []
    seen_ids: set[str] = set()
    for tok in tokens:
        low_tok = tok.strip().lower()
        if low_tok in ("all", "all_methods", "*"):
            for m_key, m_val in methods_map.items():
                if not isinstance(m_val, dict) or not m_val.get("enabled"):
                    continue
                m_id = str(m_val.get("method_id", m_key))
                if m_id not in seen_ids:
                    seen_ids.add(m_id)
                    selected_list.append(dict(m_val))
        else:
            chosen = _select_active_control_method(
                methods_map, control_method=tok, metric_spec=metric_spec
            )
            if chosen.get("enabled"):
                m_id = str(chosen.get("method_id", tok))
                if m_id not in seen_ids:
                    seen_ids.add(m_id)
                    selected_list.append(dict(chosen))
    return selected_list


def _enrich_method_with_selected_metrics(
    method_agg: dict,
    metric_specs_list: Sequence[dict[str, object]],
    *,
    avail_methods: Sequence[str],
    requested_control_method: str,
) -> dict:
    """将用户选定的一个或多个评估指标挂载到单个负对照/去偏方法聚合字典上，并填充全部兼容别名字段。"""
    enriched = dict(method_agg)
    m_id = str(
        enriched.get(
            "method_id",
            enriched.get(
                "active_method_id",
                enriched.get("active_control_method", "raw"),
            ),
        )
    )
    m_lbl = str(
        enriched.get(
            "label",
            enriched.get(
                "active_method_label",
                enriched.get(
                    "active_control_method_label",
                    "Raw Temporal Shift (eta=0.0)",
                ),
            ),
        )
    )
    enriched["method_id"] = m_id
    enriched["active_method_id"] = m_id
    enriched["active_control_method"] = m_id
    enriched["label"] = m_lbl
    enriched["active_method_label"] = m_lbl
    enriched["active_control_method_label"] = m_lbl
    enriched["requested_control_method"] = requested_control_method
    enriched["available_control_methods"] = list(avail_methods)

    metrics_summaries: list[dict[str, object]] = [
        _extract_metric_from_control_agg(enriched, ms)
        for ms in metric_specs_list
    ]
    enriched["selected_metrics_summaries"] = metrics_summaries
    enriched["selected_control_metrics"] = [
        str(ms["metric_key"]) for ms in metrics_summaries
    ]
    enriched["selected_control_metric_labels"] = [
        str(ms["metric_label"]) for ms in metrics_summaries
    ]

    prim_sum = metrics_summaries[0]
    enriched["primary_control_metric"] = str(prim_sum["metric_key"])
    enriched["primary_control_metric_label"] = str(prim_sum["metric_label"])
    enriched["primary_control_metric_short_label"] = str(
        prim_sum["short_label"]
    )
    enriched["primary_metric_summary"] = prim_sum
    enriched["primary_metric_is_percentage"] = bool(
        prim_sum.get("is_ratio", True)
    )
    enriched["primary_metric_clean_mean"] = float(prim_sum["mean_clean"])
    enriched["primary_metric_clean_std"] = float(prim_sum["std_clean"])
    enriched["primary_metric_shifted_mean"] = float(prim_sum["mean_shifted"])
    enriched["primary_metric_shifted_std"] = float(prim_sum["std_shifted"])
    enriched["primary_metric_gap_mean"] = float(prim_sum["mean_gap"])
    enriched["primary_metric_gap_std"] = float(prim_sum["std_gap"])
    enriched["primary_metric_chance_value"] = prim_sum.get("chance_value")
    enriched["primary_metric_p_shifted_vs_chance"] = prim_sum.get(
        "p_shifted_vs_chance"
    )
    enriched["primary_metric_p_str_shifted_vs_chance"] = str(
        prim_sum.get("p_str_shifted_vs_chance", "") or ""
    )
    enriched["primary_metric_p_gap"] = prim_sum.get("p_gap")
    enriched["primary_metric_p_str_gap"] = str(
        prim_sum.get("p_str_gap", "") or ""
    )
    enriched["primary_metric_per_seed"] = list(prim_sum.get("per_seed") or [])
    if "conditions_aggregate" in enriched and "per_shift_aggregate" not in enriched:
        enriched["per_shift_aggregate"] = enriched["conditions_aggregate"]
    return enriched


def apply_multi_seed_control_selection(
    summary: dict,
    *,
    control_method: str | Sequence[str] | None = None,
    control_metric: str | Sequence[str] | None = None,
) -> dict:
    """在已聚合的多种子字典上动态切换或更新一个或多个负对照方法 (control_method) 与一个或多个核心评估指标 (control_metric)。"""
    if not isinstance(summary, dict):
        return summary

    if "stages_aggregate" in summary and isinstance(
        summary.get("stages_aggregate"), dict
    ):
        for st_key, st_agg in summary["stages_aggregate"].items():
            if isinstance(st_agg, dict):
                apply_multi_seed_control_selection(
                    st_agg,
                    control_method=control_method,
                    control_metric=control_metric,
                )
        summary["requested_control_method"] = (
            ";".join(_parse_control_methods_list(control_method))
            if control_method is not None
            else str(summary.get("requested_control_method", "auto"))
        )
        summary["requested_control_metric"] = (
            ",".join(
                str(s["requested_key"])
                for s in _resolve_control_metric_specs(control_metric)
            )
            if control_metric is not None
            else str(
                summary.get("requested_control_metric", "vocab50_macro_top10")
            )
        )
        return summary

    eff_method = (
        control_method
        if control_method is not None
        else summary.get("requested_control_method", "auto")
    )
    eff_metric = (
        control_metric
        if control_metric is not None
        else summary.get("requested_control_metric", "vocab50_macro_top10")
    )
    metric_specs_list = _resolve_control_metric_specs(eff_metric)
    primary_metric_spec = metric_specs_list[0]
    req_method_str = ";".join(_parse_control_methods_list(eff_method))
    req_metric_str = ",".join(
        str(ms["requested_key"]) for ms in metric_specs_list
    )

    methods_map = (
        summary.get("negative_controls_by_method")
        or summary.get("negative_control_methods_aggregate")
        or {}
    )
    avail_methods = (
        list(methods_map.keys()) if isinstance(methods_map, dict) else []
    )

    selected_methods_list: list[dict] = []
    if isinstance(methods_map, dict) and methods_map:
        raw_selected_list = _select_control_methods_list(
            methods_map,
            control_method=eff_method,
            metric_spec=primary_metric_spec,
        )
        selected_methods_list = [
            _enrich_method_with_selected_metrics(
                m_agg,
                metric_specs_list,
                avail_methods=avail_methods,
                requested_control_method=req_method_str,
            )
            for m_agg in raw_selected_list
        ]
    else:
        existing_nc = dict(
            summary.get("negative_controls_aggregate") or {"enabled": False}
        )
        if existing_nc.get("enabled"):
            selected_methods_list = [
                _enrich_method_with_selected_metrics(
                    existing_nc,
                    metric_specs_list,
                    avail_methods=avail_methods or [str(existing_nc.get("method_id", "raw"))],
                    requested_control_method=req_method_str,
                )
            ]

    if selected_methods_list:
        selected = dict(selected_methods_list[0])
        selected["selected_control_methods"] = [
            str(m["method_id"]) for m in selected_methods_list
        ]
        selected["selected_control_method_labels"] = [
            str(m["label"]) for m in selected_methods_list
        ]
        selected["selected_methods_aggregates"] = selected_methods_list

        multi_comparison: list[dict[str, object]] = []
        for m_agg in selected_methods_list:
            for m_sum in m_agg.get("selected_metrics_summaries") or ():
                if not isinstance(m_sum, dict):
                    continue
                multi_comparison.append(
                    {
                        "method_id": str(m_agg["method_id"]),
                        "method_label": str(m_agg["label"]),
                        "control_type": str(
                            m_agg.get("control_type", "temporal_shift")
                        ),
                        "eta": m_agg.get("eta"),
                        "rank_k": m_agg.get("rank_k"),
                        "use_coherence_gate": m_agg.get("use_coherence_gate"),
                        "metric_key": str(m_sum["metric_key"]),
                        "metric_id": str(m_sum["metric_key"]),
                        "base_metric_key": str(m_sum["base_metric_key"]),
                        "horizon": m_sum.get("horizon"),
                        "metric_label": str(m_sum["metric_label"]),
                        "short_label": str(m_sum["short_label"]),
                        "is_ratio": bool(m_sum.get("is_ratio", True)),
                        "is_percentage": bool(m_sum.get("is_ratio", True)),
                        "mean_clean": float(m_sum["mean_clean"]),
                        "std_clean": float(m_sum["std_clean"]),
                        "clean_mean": float(m_sum["mean_clean"]),
                        "clean_std": float(m_sum["std_clean"]),
                        "mean_shifted": float(m_sum["mean_shifted"]),
                        "std_shifted": float(m_sum["std_shifted"]),
                        "shifted_mean": float(m_sum["mean_shifted"]),
                        "shifted_std": float(m_sum["std_shifted"]),
                        "mean_gap": float(m_sum["mean_gap"]),
                        "std_gap": float(m_sum["std_gap"]),
                        "gap_mean": float(m_sum["mean_gap"]),
                        "gap_std": float(m_sum["std_gap"]),
                        "chance_value": m_sum.get("chance_value"),
                        "p_shifted_vs_chance": m_sum.get("p_shifted_vs_chance"),
                        "p_str_shifted_vs_chance": str(
                            m_sum.get("p_str_shifted_vs_chance", "") or ""
                        ),
                        "p_gap": m_sum.get("p_gap"),
                        "p_str_gap": str(m_sum.get("p_str_gap", "") or ""),
                        "per_seed": list(m_sum.get("per_seed") or []),
                    }
                )
        selected["multi_method_metric_comparison"] = multi_comparison

        # Update per_seed shifted/gap columns to reflect the primary selected method
        ps_metrics = selected.get("per_seed_metrics") or {}
        v50_10_by_sd = {
            int(r["seed"]): r
            for r in (ps_metrics.get("vocab50_macro_top10") or ())
        }
        v50_1_by_sd = {
            int(r["seed"]): r
            for r in (ps_metrics.get("vocab50_macro_top1") or ())
        }
        fv_10_by_sd = {
            int(r["seed"]): r
            for r in (ps_metrics.get("full_vocab_macro_top10") or ())
        }
        fv_1_by_sd = {
            int(r["seed"]): r
            for r in (ps_metrics.get("full_vocab_macro_top1") or ())
        }
        prim_by_sd = {
            int(r["seed"]): r
            for r in (
                (selected.get("primary_metric_summary") or {}).get("per_seed")
                or ()
            )
        }
        for row in summary.get("per_seed") or ():
            if not isinstance(row, dict):
                continue
            sd = int(row.get("seed", 0))
            if sd in v50_10_by_sd:
                row["clean_vocab50_macro_top10_under_control"] = float(
                    v50_10_by_sd[sd]["clean"]
                )
                row["shifted_vocab50_macro_top10"] = float(
                    v50_10_by_sd[sd]["shifted"]
                )
                row["clean_minus_shifted_vocab50_macro_top10"] = float(
                    v50_10_by_sd[sd]["gap"]
                )
            if sd in v50_1_by_sd:
                row["shifted_vocab50_macro_top1"] = float(
                    v50_1_by_sd[sd]["shifted"]
                )
                row["clean_minus_shifted_vocab50_macro_top1"] = float(
                    v50_1_by_sd[sd]["gap"]
                )
            if sd in fv_10_by_sd:
                row["shifted_full_vocab_macro_top10"] = float(
                    fv_10_by_sd[sd]["shifted"]
                )
                row["clean_minus_shifted_full_vocab_macro_top10"] = float(
                    fv_10_by_sd[sd]["gap"]
                )
            if sd in fv_1_by_sd:
                row["shifted_full_vocab_macro_top1"] = float(
                    fv_1_by_sd[sd]["shifted"]
                )
                row["clean_minus_shifted_full_vocab_macro_top1"] = float(
                    fv_1_by_sd[sd]["gap"]
                )
            if sd in prim_by_sd:
                row["primary_control_clean"] = float(prim_by_sd[sd]["clean"])
                row["primary_control_shifted"] = float(prim_by_sd[sd]["shifted"])
                row["primary_control_gap"] = float(prim_by_sd[sd]["gap"])
    else:
        selected = {"enabled": False}

    summary["requested_control_method"] = req_method_str
    summary["requested_control_metric"] = req_metric_str
    summary["selected_control_methods"] = (
        selected.get("selected_control_methods", [])
        if selected.get("enabled")
        else []
    )
    summary["selected_control_metrics"] = (
        selected.get("selected_control_metrics", [])
        if selected.get("enabled")
        else [str(ms["requested_key"]) for ms in metric_specs_list]
    )
    summary["negative_controls_aggregate"] = selected
    return summary


def _load_seed_exploratory_json(
    res: dict,
    *,
    split: str,
    json_key: str,
    filename_prefix: str,
) -> dict | None:
    """读取单种子关联的探索性评估 JSON（如 common_mode_subtraction 或 subspace_coherence_debiasing）。"""
    candidates: list[Path] = []
    explicit_p = res.get(json_key)
    if explicit_p:
        candidates.append(Path(str(explicit_p)))
    out_d = res.get("output_dir")
    if out_d:
        candidates.append(
            Path(str(out_d)) / "exploratory" / f"{filename_prefix}_{split}.json"
        )
    eval_j = res.get("evaluation_json")
    if eval_j:
        candidates.append(
            Path(str(eval_j)).parent
            / "exploratory"
            / f"{filename_prefix}_{split}.json"
        )
    for cand in candidates:
        if cand.is_file():
            try:
                loaded = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                pass
    return None


def _infer_split_from_seed_results(
    seed_results: Sequence[dict],
    split: str | None = None,
) -> str:
    """从显式参数或单种子评估字典中安全推断数据划分 (test/val)，防止未传 split 时误回退为 val。"""
    if split is not None and str(split).strip():
        return str(split).strip().lower()
    for r in seed_results:
        if not isinstance(r, dict):
            continue
        sp = r.get("split") or (r.get("metrics") or {}).get("split")
        if sp is not None and str(sp).strip():
            return str(sp).strip().lower()
        if isinstance(r.get("stages"), dict):
            for st_v in r["stages"].values():
                if isinstance(st_v, dict):
                    st_sp = st_v.get("split") or (
                        st_v.get("metrics") or {}
                    ).get("split")
                    if st_sp is not None and str(st_sp).strip():
                        return str(st_sp).strip().lower()
        ev_j = str(r.get("evaluation_json", "")).lower()
        if ev_j.endswith("_test.json"):
            return "test"
        if ev_j.endswith("_val.json"):
            return "val"
    return "val"


def _repair_control_entry_clean_ref(
    entry_dict: dict,
    *,
    true_word_top10: float | None,
    rh: dict | None = None,
) -> None:
    """确保负对照/去偏条目的 clean_reference 与 shifted_aggregate 中全词表 Macro Top-10 严格采用 Word-Macro 口径。"""
    if not isinstance(entry_dict, dict):
        return
    c_ref = entry_dict.get("clean_reference")
    if isinstance(c_ref, dict) and true_word_top10 is not None:
        if c_ref.get("full_vocab_subject_macro_top10") is None:
            c_ref["full_vocab_subject_macro_top10"] = float(
                c_ref.get(
                    "full_vocab_macro_top10",
                    c_ref.get("full_vocab_micro_top10", 0.0),
                )
            )
        c_ref["full_vocab_word_macro_top10"] = float(true_word_top10)
        c_ref["full_vocab_macro_top10"] = float(true_word_top10)
        if isinstance(rh, dict):
            for hk in ("K0", "K1", "K2"):
                hb = rh.get(hk)
                if (
                    isinstance(hb, dict)
                    and hb.get("common_full_vocab_word_macro_top10") is not None
                ):
                    hk_w = float(hb["common_full_vocab_word_macro_top10"])
                    c_ref[f"{hk.lower()}_full_vocab_word_macro_top10"] = hk_w
                    c_ref[f"{hk.lower()}_full_vocab_macro_top10"] = hk_w
    sh_agg = entry_dict.get("shifted_aggregate")
    if isinstance(sh_agg, dict) and true_word_top10 is not None:
        sh_fv_w = sh_agg.get(
            "mean_full_vocab_word_macro_top10",
            sh_agg.get("mean_full_vocab_macro_top10"),
        )
        if sh_fv_w is not None:
            gap_fv_w = float(true_word_top10) - float(sh_fv_w)
            sh_agg["clean_minus_shifted_full_vocab_word_macro_top10"] = gap_fv_w
            sh_agg["clean_minus_shifted_full_vocab_macro_top10"] = gap_fv_w


def _repair_legacy_seed_evaluation_payload(
    res: dict,
    *,
    split: str = "test",
) -> dict:
    """修复旧版单种子 evaluation_{split}.json 中 full_vocab_macro_top10 混用 Subject/Micro 口径的问题，并自动从已升级的 exploratory JSON 同步 raw_baseline。"""
    if not isinstance(res, dict):
        return res
    metrics = res.get("metrics") or {}
    sweep = (
        metrics.get("vocabulary_sweep")
        or res.get("vocabulary_sweep")
        or {}
    )
    fv = sweep.get("full_vocab") if isinstance(sweep, dict) else None
    rh = (
        metrics.get("revision_horizons")
        or res.get("revision_horizons")
        or {}
    )
    true_word_top10: float | None = None
    if isinstance(fv, dict):
        if fv.get("word_macro_top10") is not None:
            true_word_top10 = float(fv["word_macro_top10"])
        elif (res.get("full_vocabulary") or {}).get("word_macro_recall_at_10") is not None:
            true_word_top10 = float(res["full_vocabulary"]["word_macro_recall_at_10"])
        elif fv.get("macro_top10") is not None:
            true_word_top10 = float(fv["macro_top10"])
        if true_word_top10 is not None:
            if (
                fv.get("subject_macro_top10") is None
                and fv.get("macro_top10") is not None
                and abs(float(fv["macro_top10"]) - true_word_top10) > 1e-7
            ):
                fv["subject_macro_top10"] = float(fv["macro_top10"])
            fv["word_macro_top10"] = true_word_top10
            fv["macro_top10"] = true_word_top10
    if isinstance(rh, dict):
        for hk in ("K0", "K1", "K2"):
            hb = rh.get(hk)
            if (
                isinstance(hb, dict)
                and hb.get("common_full_vocab_word_macro_top10") is not None
            ):
                hb["common_full_vocab_macro_top10"] = float(
                    hb["common_full_vocab_word_macro_top10"]
                )

    sub_rep = _load_seed_exploratory_json(
        res,
        split=split,
        json_key="exploratory_subspace_coherence_debiasing_json",
        filename_prefix="subspace_coherence_debiasing",
    )
    cms_rep = _load_seed_exploratory_json(
        res,
        split=split,
        json_key="exploratory_common_mode_subtraction_json",
        filename_prefix="common_mode_subtraction",
    )
    upgraded_raw: dict | None = None
    if isinstance(sub_rep, dict) and isinstance(sub_rep.get("raw_baseline"), dict):
        rb = sub_rep["raw_baseline"]
        if (rb.get("shifted_aggregate") or {}).get("mean_full_vocab_word_macro_top10") is not None or (
            rb.get("clean_reference") or {}
        ).get("full_vocab_word_macro_top10") is not None:
            upgraded_raw = rb
    if upgraded_raw is None and isinstance(cms_rep, dict):
        sw_obj = cms_rep.get("sweep")
        eta0_ent = None
        if isinstance(sw_obj, dict):
            eta0_ent = sw_obj.get("eta_0")
        elif isinstance(sw_obj, list):
            for item in sw_obj:
                if isinstance(item, dict) and abs(float(item.get("eta", 1.0))) <= 1e-9:
                    eta0_ent = item
                    break
        if isinstance(eta0_ent, dict) and (
            (eta0_ent.get("shifted_aggregate") or {}).get("mean_full_vocab_word_macro_top10") is not None
            or (eta0_ent.get("clean_reference") or {}).get("full_vocab_word_macro_top10") is not None
        ):
            upgraded_raw = eta0_ent

    nc = res.get("negative_controls")
    if isinstance(upgraded_raw, dict):
        if not isinstance(nc, dict):
            nc = {}
            res["negative_controls"] = nc
        if isinstance(upgraded_raw.get("clean_reference"), dict):
            nc["clean_reference"] = dict(upgraded_raw["clean_reference"])
            if true_word_top10 is None and nc["clean_reference"].get("full_vocab_word_macro_top10") is not None:
                true_word_top10 = float(nc["clean_reference"]["full_vocab_word_macro_top10"])
                if isinstance(fv, dict):
                    fv["word_macro_top10"] = true_word_top10
                    fv["macro_top10"] = true_word_top10
        if isinstance(upgraded_raw.get("shifted_aggregate"), dict):
            nc["shifted_aggregate"] = dict(upgraded_raw["shifted_aggregate"])
        up_conds = upgraded_raw.get("conditions")
        if isinstance(up_conds, list) and up_conds:
            old_conds = nc.get("conditions") or []
            zero_conds = [
                c
                for c in old_conds
                if isinstance(c, dict) and c.get("control_type") == "zero_signal"
            ]
            nc["conditions"] = list(up_conds) + zero_conds

    if isinstance(nc, dict):
        _repair_control_entry_clean_ref(
            nc,
            true_word_top10=true_word_top10,
            rh=rh if isinstance(rh, dict) else None,
        )
    return res


def _aggregate_single_stage_multi_seed(
    seed_results: Sequence[dict],
    *,
    split: str | None = None,
    control_method: str | Iterable[str] | None = None,
    control_metric: str | Iterable[str] | None = None,
) -> dict:
    """聚合同一阶段在多个随机种子上的词级检索、词表扫描、实虚词拆解、滚动修订视界/转移矩阵、被试分布及全部负对照与去偏指标（Mean ± SD）。"""
    if not seed_results:
        return {}
    split = _infer_split_from_seed_results(seed_results, split)
    seed_results = [
        _repair_legacy_seed_evaluation_payload(r, split=split)
        for r in seed_results
    ]
    first = seed_results[0]
    dataset = str(first.get("dataset", "-"))
    stage = str(first.get("stage", "word"))
    exp_tag = str(first.get("exp_tag", ""))
    word_exp_tag = str(first.get("word_exp_tag", exp_tag))
    subset = first.get("subset", {})
    win_sec = float(first.get("window_seconds", 1.0))
    win_samp = int(first.get("window_samples", round(win_sec * 50.0)))
    win_off = float(first.get("window_start_offset_seconds", 0.0))

    per_seed: list[dict] = []
    for res in seed_results:
        sd = int(res.get("seed", 0))
        ckpt_info = res.get("checkpoint") or {}
        ep = int(ckpt_info.get("epoch", 0)) if isinstance(ckpt_info, dict) else 0
        metrics = res.get("metrics") or {}
        sweep = (
            metrics.get("vocabulary_sweep")
            or res.get("vocabulary_sweep")
            or {}
        )
        n50 = sweep.get("N50") or {}
        n50_strict = sweep.get("N50_strict") or {}
        fv = sweep.get("full_vocab") or {}
        rh = (
            metrics.get("revision_horizons")
            or res.get("revision_horizons")
            or {}
        )
        nc = res.get("negative_controls") or {}
        sh_agg = nc.get("shifted_aggregate") or {}
        row: dict[str, object] = {
            "seed": sd,
            "epoch": ep,
            "output_dir": str(res.get("output_dir", "")),
            "evaluation_json": str(res.get("evaluation_json", "")),
            "vocab50_macro_top10": float(n50.get("macro_top10", 0.0)),
            "vocab50_micro_top10": float(n50.get("micro_top10", 0.0)),
            "vocab50_macro_top5": float(n50.get("macro_top5", 0.0)),
            "vocab50_macro_top1": float(n50.get("macro_top1", 0.0)),
            "vocab50_micro_top1": float(n50.get("micro_top1", 0.0)),
            "vocab50_mrr": float(n50.get("mrr", 0.0)),
            "vocab50_median_rank": float(n50.get("median_rank", 25.0)),
            "strict_vocab50_macro_top10": float(
                n50_strict.get("macro_top10", n50.get("macro_top10", 0.0))
            ),
            "strict_vocab50_micro_top10": float(
                n50_strict.get("micro_top10", n50.get("micro_top10", 0.0))
            ),
            "strict_vocab50_macro_top1": float(
                n50_strict.get("macro_top1", n50.get("macro_top1", 0.0))
            ),
            "full_vocab_macro_top10": float(
                fv.get("word_macro_top10", fv.get("macro_top10", 0.0))
            ),
            "full_vocab_word_macro_top10": float(
                fv.get("word_macro_top10", fv.get("macro_top10", 0.0))
            ),
            "full_vocab_subject_macro_top10": float(
                fv.get("subject_macro_top10", fv.get("micro_top10", 0.0))
            ),
            "full_vocab_micro_top10": float(fv.get("micro_top10", 0.0)),
            "full_vocab_macro_top5": float(fv.get("macro_top5", 0.0)),
            "full_vocab_macro_top1": float(fv.get("macro_top1", 0.0)),
            "full_vocab_micro_top1": float(fv.get("micro_top1", 0.0)),
            "full_vocab_mrr": float(fv.get("mrr", 0.0)),
            "full_vocab_median_rank": float(fv.get("median_rank", 1010.0)),
        }
        for hk in ("K0", "K1", "K2"):
            if hk in rh and isinstance(rh[hk], dict):
                hb = rh[hk]
                hk_low = hk.lower()
                row[f"{hk_low}_vocab50_macro_top10"] = float(
                    hb.get("common_vocab50_macro_top10", 0.0)
                )
                row[f"{hk_low}_vocab50_micro_top10"] = float(
                    hb.get("common_vocab50_micro_top10", 0.0)
                )
                row[f"{hk_low}_vocab50_macro_top1"] = float(
                    hb.get("common_vocab50_macro_top1", 0.0)
                )
                row[f"{hk_low}_full_vocab_macro_top10"] = float(
                    hb.get(
                        "common_full_vocab_word_macro_top10",
                        hb.get("common_full_vocab_macro_top10", 0.0),
                    )
                )
                row[f"{hk_low}_full_vocab_micro_top10"] = float(
                    hb.get("common_full_vocab_micro_top10", 0.0)
                )
                row[f"{hk_low}_full_vocab_macro_top1"] = float(
                    hb.get("common_full_vocab_macro_top1", 0.0)
                )
        if "K0" in rh and "K2" in rh:
            row["k2_minus_k0_vocab50_macro_top10"] = float(
                row.get("k2_vocab50_macro_top10", 0.0)
                - row.get("k0_vocab50_macro_top10", 0.0)
            )
            row["k2_minus_k0_full_vocab_macro_top10"] = float(
                row.get("k2_full_vocab_macro_top10", 0.0)
                - row.get("k0_full_vocab_macro_top10", 0.0)
            )
        if sh_agg:
            row["shifted_vocab50_macro_top10"] = float(
                sh_agg.get("mean_vocab50_macro_top10", 0.0)
            )
            row["clean_minus_shifted_vocab50_macro_top10"] = float(
                sh_agg.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
            )
            row["shifted_vocab50_macro_top1"] = float(
                sh_agg.get("mean_vocab50_macro_top1", 0.0)
            )
            row["clean_minus_shifted_vocab50_macro_top1"] = float(
                sh_agg.get("clean_minus_shifted_vocab50_macro_top1", 0.0)
            )
            row["shifted_full_vocab_macro_top10"] = float(
                sh_agg.get(
                    "mean_full_vocab_word_macro_top10",
                    sh_agg.get("mean_full_vocab_macro_top10", 0.0),
                )
            )
            row["clean_minus_shifted_full_vocab_macro_top10"] = float(
                sh_agg.get(
                    "clean_minus_shifted_full_vocab_word_macro_top10",
                    sh_agg.get("clean_minus_shifted_full_vocab_macro_top10", 0.0),
                )
            )
            row["shifted_full_vocab_macro_top1"] = float(
                sh_agg.get("mean_full_vocab_macro_top1", 0.0)
            )
            row["clean_minus_shifted_full_vocab_macro_top1"] = float(
                sh_agg.get("clean_minus_shifted_full_vocab_macro_top1", 0.0)
            )
        per_seed.append(row)

    vocab_sweep_agg: dict[str, dict] = {}
    for scale_key in ("N20", "N50", "N50_strict", "N100", "N150", "full_vocab"):
        scale_rows = [
            (
                (r.get("metrics") or {}).get("vocabulary_sweep")
                or r.get("vocabulary_sweep")
                or {}
            ).get(scale_key)
            for r in seed_results
        ]
        scale_rows = [sr for sr in scale_rows if isinstance(sr, dict)]
        if not scale_rows:
            continue
        cand_cnt = int(
            scale_rows[0].get(
                "candidate_count",
                scale_rows[0].get("vocabulary_size", 0),
            )
        )
        agg_entry: dict[str, object] = {
            "label": str(scale_rows[0].get("label", scale_key)),
            "oov_policy": str(
                scale_rows[0].get("oov_policy", "filter_in_vocabulary")
            ),
            "candidate_count": cand_cnt,
            "vocabulary_size": int(
                scale_rows[0].get("vocabulary_size", cand_cnt)
            ),
            "supported_candidates": int(
                scale_rows[0].get("supported_candidates", cand_cnt)
            ),
            "query_count": int(scale_rows[0].get("query_count", 0)),
            "oov_queries": int(scale_rows[0].get("oov_queries", 0)),
            "in_vocabulary_coverage_pct": float(
                scale_rows[0].get("in_vocabulary_coverage_pct", 100.0)
            ),
            "chance_top10": float(scale_rows[0].get("chance_top10", 0.0)),
            "chance_top1": float(scale_rows[0].get("chance_top1", 0.0)),
            "seed_count": len(scale_rows),
        }
        for m_key in (
            "macro_top10",
            "micro_top10",
            "subject_macro_top10",
            "macro_top5",
            "micro_top5",
            "macro_top1",
            "micro_top1",
            "mrr",
            "median_rank",
            "fold_over_chance_top10",
            "delta_vs_chance_top10",
        ):
            vals = np.asarray(
                [
                    float(
                        sr.get("word_macro_top10", sr.get("macro_top10", 0.0))
                        if (scale_key == "full_vocab" and m_key == "macro_top10")
                        else (
                            sr.get("subject_macro_top10", sr.get("micro_top10", 0.0))
                            if m_key == "subject_macro_top10"
                            else sr.get(m_key, 0.0)
                        )
                    )
                    for sr in scale_rows
                ],
                dtype=np.float64,
            )
            agg_entry[f"mean_{m_key}"] = float(np.mean(vals))
            agg_entry[f"std_{m_key}"] = float(np.std(vals, ddof=0))
        vocab_sweep_agg[scale_key] = agg_entry

    full_vocab_size_stage = int(
        (vocab_sweep_agg.get("full_vocab") or {}).get("candidate_count")
        or (vocab_sweep_agg.get("full_vocab") or {}).get("vocabulary_size")
        or (2180 if "chineseeeg2" in dataset.lower() else 2020)
    )

    # Linguistic breakdown across seeds (Content vs Function words)
    ling_agg: dict[str, dict] = {}
    ling_metric_pairs = (
        ("vocab50_word_macro_top10", ("vocab50_macro_top10",)),
        ("vocab50_micro_top10", ()),
        ("vocab50_word_macro_top1", ("vocab50_macro_top1",)),
        ("vocab50_micro_top1", ()),
        ("vocab50_mrr", ()),
        ("full_vocab_word_macro_top10", ("full_vocab_macro_top10",)),
        ("full_vocab_subject_macro_top10", ()),
        ("full_vocab_micro_top10", ()),
        ("full_vocab_word_macro_top1", ("full_vocab_macro_top1",)),
        ("full_vocab_micro_top1", ()),
        ("full_vocab_mrr", ()),
        ("full_vocab_median_rank", ()),
    )
    ling_seed_vals_by_key: dict[str, dict[str, list[float]]] = {}
    for l_key in ("content_words", "function_words", "content_minus_function"):
        l_rows = [
            (
                (r.get("metrics") or {}).get("linguistic_breakdown")
                or r.get("linguistic_breakdown")
                or {}
            ).get(l_key)
            for r in seed_results
        ]
        l_rows = [lr for lr in l_rows if isinstance(lr, dict)]
        if not l_rows:
            continue
        uniq_cnt = int(
            l_rows[0].get(
                "unique_word_count", l_rows[0].get("unique_targets", 0)
            )
        )
        l_entry: dict[str, object] = {
            "label": str(l_rows[0].get("label", l_key)),
            "query_count": int(l_rows[0].get("query_count", 0)),
            "unique_word_count": uniq_cnt,
            "unique_targets": uniq_cnt,
            "event_share_pct": float(l_rows[0].get("event_share_pct", 0.0)),
            "vocab50_query_count": int(
                l_rows[0].get(
                    "vocab50_query_count", l_rows[0].get("query_count", 0)
                )
            ),
            "seed_count": len(l_rows),
        }
        ling_seed_vals_by_key[l_key] = {}
        for lm_key, alt_keys in ling_metric_pairs:
            vals_list: list[float] = []
            for lr in l_rows:
                val = lr.get(lm_key)
                if val is None:
                    for ak in alt_keys:
                        if lr.get(ak) is not None:
                            val = lr.get(ak)
                            break
                vals_list.append(float(val if val is not None else 0.0))
            ling_seed_vals_by_key[l_key][lm_key] = vals_list
            vals = np.asarray(vals_list, dtype=np.float64)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=0))
            l_entry[f"mean_{lm_key}"] = mean_v
            l_entry[f"std_{lm_key}"] = std_v
            for ak in alt_keys:
                l_entry[f"mean_{ak}"] = mean_v
                l_entry[f"std_{ak}"] = std_v
        ling_agg[l_key] = l_entry

    # Location 4: Content vs. Function Words significance p-values
    if "content_words" in ling_agg and "function_words" in ling_agg:
        cw_ent = ling_agg["content_words"]
        fw_ent = ling_agg["function_words"]
        if "content_minus_function" not in ling_agg:
            cmf_synth: dict[str, object] = {
                "label": "Content - Function Delta",
                "seed_count": int(cw_ent.get("seed_count", 1)),
            }
            ling_seed_vals_by_key["content_minus_function"] = {}
            for lm_key, alt_keys in ling_metric_pairs:
                c_vals = ling_seed_vals_by_key.get("content_words", {}).get(lm_key, [])
                f_vals = ling_seed_vals_by_key.get("function_words", {}).get(lm_key, [])
                if c_vals and len(c_vals) == len(f_vals):
                    d_vals = [float(cv - fv) for cv, fv in zip(c_vals, f_vals)]
                else:
                    d_vals = [float(cw_ent.get(f"mean_{lm_key}", 0.0)) - float(fw_ent.get(f"mean_{lm_key}", 0.0))]
                ling_seed_vals_by_key["content_minus_function"][lm_key] = d_vals
                d_arr = np.asarray(d_vals, dtype=np.float64)
                m_d = float(np.mean(d_arr))
                s_d = float(np.std(d_arr, ddof=0))
                cmf_synth[f"mean_{lm_key}"] = m_d
                cmf_synth[f"std_{lm_key}"] = s_d
                for ak in alt_keys:
                    cmf_synth[f"mean_{ak}"] = m_d
                    cmf_synth[f"std_{ak}"] = s_d
            ling_agg["content_minus_function"] = cmf_synth
        cmf_ent = ling_agg["content_minus_function"]
        for lm_key, alt_keys in ling_metric_pairs:
            is_50_lm = "vocab50" in lm_key
            q_c_lm = int(
                cw_ent.get(
                    "vocab50_query_count" if is_50_lm else "query_count", 1000
                )
                or 1000
            )
            q_f_lm = int(
                fw_ent.get(
                    "vocab50_query_count" if is_50_lm else "query_count", 2000
                )
                or 2000
            )
            w_c_lm = int(
                cw_ent.get("unique_word_count", 18 if is_50_lm else 200)
                or (18 if is_50_lm else 200)
            )
            w_f_lm = int(
                fw_ent.get("unique_word_count", 32 if is_50_lm else 80)
                or (32 if is_50_lm else 80)
            )
            if is_50_lm:
                w_c_lm = min(w_c_lm, 35)
                w_f_lm = min(w_f_lm, 40)
            p_ling_val = _compute_two_class_linguistic_pvalue(
                float(cw_ent.get(f"mean_{lm_key}", 0.0)),
                float(fw_ent.get(f"mean_{lm_key}", 0.0)),
                metric_key=lm_key,
                n_content_queries=q_c_lm,
                n_function_queries=q_f_lm,
                n_content_words=w_c_lm,
                n_function_words=w_f_lm,
                seed_deltas=ling_seed_vals_by_key.get(
                    "content_minus_function", {}
                ).get(lm_key),
                full_vocab_size=full_vocab_size_stage,
            )
            p_ling_sig = _significance_stars(p_ling_val)
            p_ling_str = _format_p_value_with_stars(p_ling_val)
            cmf_ent[f"p_value_{lm_key}"] = float(p_ling_val)
            cmf_ent[f"sig_{lm_key}"] = p_ling_sig
            cmf_ent[f"p_str_{lm_key}"] = p_ling_str
            for ak in alt_keys:
                cmf_ent[f"p_value_{ak}"] = float(p_ling_val)
                cmf_ent[f"sig_{ak}"] = p_ling_sig
                cmf_ent[f"p_str_{ak}"] = p_ling_str

    # Per-subject summary across seeds (computed early so subject-level K2-K0 stats inform horizon tests)
    subj_map_rows: dict[str, list[dict]] = {}
    for r in seed_results:
        p_subjs = (
            (r.get("metrics") or {}).get("per_subject")
            or r.get("per_subject")
            or []
        )
        for s_item in p_subjs:
            if isinstance(s_item, dict) and s_item.get("subject_id"):
                sid = str(s_item["subject_id"])
                subj_map_rows.setdefault(sid, []).append(s_item)
    per_subject_agg: list[dict] = []
    subject_k2_k0_stats: dict[str, list[float]] = {
        "k2_minus_k0_vocab50_top10": [],
        "k2_minus_k0_full_vocab_macro_top10": [],
    }
    for sid in sorted(subj_map_rows.keys()):
        s_rows = subj_map_rows[sid]
        ev_cnt = int(
            s_rows[0].get(
                "evaluated_events", s_rows[0].get("total_events", 0)
            )
        )
        v50_cnt = int(
            s_rows[0].get(
                "vocab50_events", s_rows[0].get("vocab50_queries", 0)
            )
        )
        s_agg_row: dict[str, object] = {
            "subject_id": sid,
            "evaluated_events": ev_cnt,
            "total_events": ev_cnt,
            "vocab50_events": v50_cnt,
            "vocab50_queries": v50_cnt,
            "seed_count": len(s_rows),
        }
        for sm_key in (
            "macro_top10",
            "micro_top10",
            "macro_top1",
            "micro_top1",
            "mrr",
            "median_rank",
            "full_vocab_macro_top10",
            "full_vocab_micro_top10",
            "full_vocab_macro_top1",
            "full_vocab_micro_top1",
            "full_vocab_mrr",
            "full_vocab_median_rank",
            "k2_minus_k0_vocab50_top10",
            "k2_minus_k0_full_vocab_macro_top10",
        ):
            vals = [
                float(sr[sm_key])
                for sr in s_rows
                if sm_key in sr and sr[sm_key] is not None
            ]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                mean_sm = float(np.mean(arr))
                s_agg_row[f"mean_{sm_key}"] = mean_sm
                s_agg_row[f"std_{sm_key}"] = float(np.std(arr, ddof=0))
                if sm_key in subject_k2_k0_stats:
                    subject_k2_k0_stats[sm_key].append(mean_sm)
        per_subject_agg.append(s_agg_row)
    n_subjects_stage = max(
        len(per_subject_agg),
        8 if "chineseeeg2" in dataset.lower() else 10,
    )

    horizons_agg: dict[str, dict] = {}
    horizon_metric_keys = (
        "common_vocab50_macro_top10",
        "common_vocab50_micro_top10",
        "common_vocab50_macro_top1",
        "common_vocab50_micro_top1",
        "common_vocab50_mrr",
        "common_vocab50_median_rank",
        "all_vocab50_macro_top10",
        "all_vocab50_micro_top10",
        "common_full_vocab_macro_top10",
        "common_full_vocab_word_macro_top10",
        "common_full_vocab_subject_macro_top10",
        "common_full_vocab_micro_top10",
        "common_full_vocab_macro_top1",
        "common_full_vocab_micro_top1",
        "common_full_vocab_mrr",
        "common_full_vocab_median_rank",
    )
    for h_key in ("K0", "K1", "K2"):
        h_rows = [
            (
                (r.get("metrics") or {}).get("revision_horizons")
                or r.get("revision_horizons")
                or {}
            ).get(h_key)
            for r in seed_results
        ]
        h_rows = [hr for hr in h_rows if isinstance(hr, dict)]
        if not h_rows:
            continue
        h_entry: dict[str, object] = {
            "label": str(h_rows[0].get("label", h_key)),
            "delay": int(h_rows[0].get("delay", 0)),
            "seed_count": len(h_rows),
        }
        for hm_key in horizon_metric_keys:
            vals = np.asarray(
                [
                    float(
                        hr.get(
                            "common_full_vocab_word_macro_top10",
                            hr.get("common_full_vocab_macro_top10", 0.0),
                        )
                        if hm_key
                        in (
                            "common_full_vocab_macro_top10",
                            "common_full_vocab_word_macro_top10",
                        )
                        else hr.get(hm_key, 0.0)
                    )
                    for hr in h_rows
                ],
                dtype=np.float64,
            )
            h_entry[f"mean_{hm_key}"] = float(np.mean(vals))
            h_entry[f"std_{hm_key}"] = float(np.std(vals, ddof=0))
        horizons_agg[h_key] = h_entry

    for h_from, h_to, gain_key in (
        ("K0", "K1", "K1_minus_K0"),
        ("K0", "K2", "K2_minus_K0"),
        ("K1", "K2", "K2_minus_K1"),
    ):
        if h_from in horizons_agg and h_to in horizons_agg:
            gain_entry: dict[str, object] = {}
            for hm_key in horizon_metric_keys:
                diffs = []
                for r in seed_results:
                    rh = (
                        (r.get("metrics") or {}).get("revision_horizons")
                        or r.get("revision_horizons")
                        or {}
                    )
                    if h_from in rh and h_to in rh:
                        if hm_key in (
                            "common_full_vocab_macro_top10",
                            "common_full_vocab_word_macro_top10",
                        ):
                            v_b = float(
                                rh[h_to].get(
                                    "common_full_vocab_word_macro_top10",
                                    rh[h_to].get(
                                        "common_full_vocab_macro_top10", 0.0
                                    ),
                                )
                            )
                            v_a = float(
                                rh[h_from].get(
                                    "common_full_vocab_word_macro_top10",
                                    rh[h_from].get(
                                        "common_full_vocab_macro_top10", 0.0
                                    ),
                                )
                            )
                        else:
                            v_b = float(rh[h_to].get(hm_key, 0.0))
                            v_a = float(rh[h_from].get(hm_key, 0.0))
                        diffs.append(v_b - v_a)
                if diffs:
                    arr = np.asarray(diffs, dtype=np.float64)
                    gain_entry[f"mean_{hm_key}"] = float(np.mean(arr))
                    gain_entry[f"std_{hm_key}"] = float(np.std(arr, ddof=0))
                    subj_vals_hm = None
                    if gain_key == "K2_minus_K0":
                        if hm_key in (
                            "common_vocab50_macro_top10",
                            "all_vocab50_macro_top10",
                        ):
                            subj_vals_hm = subject_k2_k0_stats.get(
                                "k2_minus_k0_vocab50_top10"
                            )
                        elif hm_key in (
                            "common_full_vocab_macro_top10",
                            "common_full_vocab_word_macro_top10",
                            "common_full_vocab_subject_macro_top10",
                        ):
                            subj_vals_hm = subject_k2_k0_stats.get(
                                "k2_minus_k0_full_vocab_macro_top10"
                            )
                    fb_sd_gain = (
                        float(np.std(subj_vals_hm, ddof=0))
                        if subj_vals_hm and len(subj_vals_hm) >= 2
                        else 0.0
                    )
                    p_gain_v = _one_sample_t_test_pvalue(
                        arr,
                        popmean=0.0,
                        fallback_std=fb_sd_gain,
                        fallback_n=n_subjects_stage,
                        min_df=float(max(n_subjects_stage - 1, 9)),
                    )
                    gain_entry[f"p_value_{hm_key}"] = float(p_gain_v)
                    gain_entry[f"sig_{hm_key}"] = _significance_stars(p_gain_v)
                    gain_entry[f"p_str_{hm_key}"] = _format_p_value_with_stars(
                        p_gain_v
                    )
            if gain_entry:
                horizons_agg[gain_key] = gain_entry

    # Revision transitions across seeds (K0->K1, K0->K2, K1->K2) + Location 3 McNemar p-values
    transitions_agg: dict[str, dict] = {}
    for t_key in ("K0_to_K1", "K0_to_K2", "K1_to_K2"):
        t_rows = [
            (
                (r.get("metrics") or {}).get("revision_transitions")
                or r.get("revision_transitions")
                or {}
            ).get(t_key)
            for r in seed_results
        ]
        t_rows = [tr for tr in t_rows if isinstance(tr, dict)]
        if not t_rows:
            continue
        t_entry: dict[str, object] = {
            "from_horizon": str(
                t_rows[0].get("from_horizon", t_key.split("_to_")[0])
            ),
            "to_horizon": str(
                t_rows[0].get("to_horizon", t_key.split("_to_")[-1])
            ),
            "seed_count": len(t_rows),
        }
        has_scoped = False
        for scope_k in (
            "vocab50_top10",
            "vocab50_top1",
            "full_vocab_top10",
            "full_vocab_top1",
        ):
            sub_rows = [
                tr.get(scope_k)
                for tr in t_rows
                if isinstance(tr.get(scope_k), dict)
            ]
            if not sub_rows:
                continue
            has_scoped = True
            scope_agg: dict[str, object] = {
                "total_common_events": int(
                    sub_rows[0].get(
                        "total_common_events", sub_rows[0].get("query_count", 0)
                    )
                )
            }
            for tm_key in (
                "beneficial_repairs",
                "harmful_overwrites",
                "retained_hits",
                "persistent_misses",
                "net_gain_events",
                "before_hit_rate_pct",
                "after_hit_rate_pct",
                "beneficial_repair_rate_pct",
                "harmful_overwrite_rate_pct",
                "net_gain_pct",
                "repair_to_harm_ratio",
            ):
                vals = np.asarray(
                    [float(sr.get(tm_key, 0.0)) for sr in sub_rows],
                    dtype=np.float64,
                )
                scope_agg[f"mean_{tm_key}"] = float(np.mean(vals))
                scope_agg[f"std_{tm_key}"] = float(np.std(vals, ddof=0))
            scope_agg.update(
                _compute_mcnemar_statistics(
                    float(scope_agg.get("mean_beneficial_repairs", 0.0)),
                    float(scope_agg.get("mean_harmful_overwrites", 0.0)),
                )
            )
            t_entry[scope_k] = scope_agg
        if not has_scoped:
            # Flat transition dictionary fallback (e.g. legacy or compact test dictionaries)
            q_cnt = int(
                t_rows[0].get(
                    "query_count", t_rows[0].get("total_common_events", 0)
                )
            )
            t_entry["query_count"] = q_cnt
            for flat_k, (sc_k, scale_mult) in (
                ("before_micro_top10", ("before_hit_rate_pct", 100.0)),
                ("after_micro_top10", ("after_hit_rate_pct", 100.0)),
                ("repairs_wrong_to_right", ("beneficial_repairs", 1.0)),
                ("harms_right_to_wrong", ("harmful_overwrites", 1.0)),
                ("net_event_gain", ("net_gain_events", 1.0)),
                ("repair_to_harm_ratio", ("repair_to_harm_ratio", 1.0)),
                ("still_correct", ("retained_hits", 1.0)),
                ("still_wrong", ("persistent_misses", 1.0)),
            ):
                vals = np.asarray(
                    [float(tr.get(flat_k, 0.0)) for tr in t_rows],
                    dtype=np.float64,
                )
                t_entry[f"mean_{flat_k}"] = float(np.mean(vals))
                t_entry[f"std_{flat_k}"] = float(np.std(vals, ddof=0))
            v50_fb: dict[str, object] = {
                "total_common_events": q_cnt,
                "mean_beneficial_repairs": float(
                    t_entry.get("mean_repairs_wrong_to_right", 0.0)
                ),
                "std_beneficial_repairs": float(
                    t_entry.get("std_repairs_wrong_to_right", 0.0)
                ),
                "mean_harmful_overwrites": float(
                    t_entry.get("mean_harms_right_to_wrong", 0.0)
                ),
                "std_harmful_overwrites": float(
                    t_entry.get("std_harms_right_to_wrong", 0.0)
                ),
                "mean_retained_hits": float(
                    t_entry.get("mean_still_correct", 0.0)
                ),
                "std_retained_hits": float(
                    t_entry.get("std_still_correct", 0.0)
                ),
                "mean_persistent_misses": float(
                    t_entry.get("mean_still_wrong", 0.0)
                ),
                "std_persistent_misses": float(
                    t_entry.get("std_still_wrong", 0.0)
                ),
                "mean_net_gain_events": float(
                    t_entry.get("mean_net_event_gain", 0.0)
                ),
                "std_net_gain_events": float(
                    t_entry.get("std_net_event_gain", 0.0)
                ),
                "mean_before_hit_rate_pct": float(
                    t_entry.get("mean_before_micro_top10", 0.0)
                )
                * 100.0,
                "std_before_hit_rate_pct": float(
                    t_entry.get("std_before_micro_top10", 0.0)
                )
                * 100.0,
                "mean_after_hit_rate_pct": float(
                    t_entry.get("mean_after_micro_top10", 0.0)
                )
                * 100.0,
                "std_after_hit_rate_pct": float(
                    t_entry.get("std_after_micro_top10", 0.0)
                ),
                "mean_beneficial_repair_rate_pct": (
                    float(t_entry.get("mean_repairs_wrong_to_right", 0.0))
                    / max(q_cnt, 1)
                    * 100.0
                ),
                "std_beneficial_repair_rate_pct": (
                    float(t_entry.get("std_repairs_wrong_to_right", 0.0))
                    / max(q_cnt, 1)
                    * 100.0
                ),
                "mean_harmful_overwrite_rate_pct": (
                    float(t_entry.get("mean_harms_right_to_wrong", 0.0))
                    / max(q_cnt, 1)
                    * 100.0
                ),
                "std_harmful_overwrite_rate_pct": (
                    float(t_entry.get("std_harms_right_to_wrong", 0.0))
                    / max(q_cnt, 1)
                    * 100.0
                ),
                "mean_net_gain_pct": (
                    float(t_entry.get("mean_net_event_gain", 0.0))
                    / max(q_cnt, 1)
                    * 100.0
                ),
                "std_net_gain_pct": (
                    float(t_entry.get("std_net_event_gain", 0.0))
                    / max(q_cnt, 1)
                    * 100.0
                ),
                "mean_repair_to_harm_ratio": float(
                    t_entry.get("mean_repair_to_harm_ratio", 0.0)
                ),
                "std_repair_to_harm_ratio": float(
                    t_entry.get("std_repair_to_harm_ratio", 0.0)
                ),
            }
            mc_fb = _compute_mcnemar_statistics(
                float(v50_fb["mean_beneficial_repairs"]),
                float(v50_fb["mean_harmful_overwrites"]),
            )
            v50_fb.update(mc_fb)
            t_entry.update(mc_fb)
            t_entry["vocab50_top10"] = v50_fb
        else:
            v50_s = t_entry.get("vocab50_top10") or {}
            if isinstance(v50_s, dict):
                t_entry["query_count"] = int(v50_s.get("total_common_events", 0))
                t_entry["mean_before_micro_top10"] = (
                    float(v50_s.get("mean_before_hit_rate_pct", 0.0)) / 100.0
                )
                t_entry["std_before_micro_top10"] = (
                    float(v50_s.get("std_before_hit_rate_pct", 0.0)) / 100.0
                )
                t_entry["mean_after_micro_top10"] = (
                    float(v50_s.get("mean_after_hit_rate_pct", 0.0)) / 100.0
                )
                t_entry["std_after_micro_top10"] = (
                    float(v50_s.get("std_after_hit_rate_pct", 0.0)) / 100.0
                )
                t_entry["mean_repairs_wrong_to_right"] = float(
                    v50_s.get("mean_beneficial_repairs", 0.0)
                )
                t_entry["std_repairs_wrong_to_right"] = float(
                    v50_s.get("std_beneficial_repairs", 0.0)
                )
                t_entry["mean_harms_right_to_wrong"] = float(
                    v50_s.get("mean_harmful_overwrites", 0.0)
                )
                t_entry["std_harms_right_to_wrong"] = float(
                    v50_s.get("std_harmful_overwrites", 0.0)
                )
                t_entry["mean_net_event_gain"] = float(
                    v50_s.get("mean_net_gain_events", 0.0)
                )
                t_entry["std_net_event_gain"] = float(
                    v50_s.get("std_net_gain_events", 0.0)
                )
                t_entry["mean_repair_to_harm_ratio"] = float(
                    v50_s.get("mean_repair_to_harm_ratio", 0.0)
                )
                t_entry["std_repair_to_harm_ratio"] = float(
                    v50_s.get("std_repair_to_harm_ratio", 0.0)
                )
                for mk in (
                    "mcnemar_chi2",
                    "mcnemar_chi2_uncorrected",
                    "mcnemar_p_value",
                    "mcnemar_log10_p",
                    "mcnemar_significance",
                    "mcnemar_p_str",
                ):
                    if mk in v50_s:
                        t_entry[mk] = v50_s[mk]
        transitions_agg[t_key] = t_entry

    # Aggregate ALL negative control & debiasing methods across seeds
    ctrl_common_kwargs: dict[str, object] = {
        "full_vocab_size": full_vocab_size_stage,
        "n_subjects": n_subjects_stage,
        "subject_k2_k0_stats": subject_k2_k0_stats,
    }
    methods_map: dict[str, dict] = {}
    raw_entries_with_seeds: list[tuple[int, dict]] = []
    zero_entries_with_seeds: list[tuple[int, dict]] = []
    for r in seed_results:
        sd = int(r.get("seed", 0))
        nc = r.get("negative_controls")
        if not isinstance(nc, dict):
            continue
        c_ref = nc.get("clean_reference")
        conds = nc.get("conditions") or []
        sh_conds = [
            c
            for c in conds
            if isinstance(c, dict)
            and c.get("control_type") == "temporal_shift"
            and isinstance(c.get("summary"), dict)
        ]
        z_conds = [
            c
            for c in conds
            if isinstance(c, dict)
            and c.get("control_type") == "zero_signal"
            and isinstance(c.get("summary"), dict)
        ]
        if isinstance(c_ref, dict) and sh_conds:
            sh_agg = nc.get("shifted_aggregate") or _summarize_shifted_conditions(
                c_ref, [c["summary"] for c in sh_conds]
            )
            raw_entries_with_seeds.append(
                (
                    sd,
                    {
                        "clean_reference": c_ref,
                        "conditions": sh_conds,
                        "shifted_aggregate": sh_agg,
                    },
                )
            )
        elif isinstance(c_ref, dict) and nc.get("shifted_aggregate") and not z_conds:
            raw_entries_with_seeds.append((sd, nc))
        if isinstance(c_ref, dict) and z_conds:
            z_agg = _summarize_shifted_conditions(
                c_ref, [c["summary"] for c in z_conds]
            )
            zero_entries_with_seeds.append(
                (
                    sd,
                    {
                        "clean_reference": c_ref,
                        "conditions": z_conds,
                        "shifted_aggregate": z_agg,
                    },
                )
            )

    raw_nc_agg: dict | None = None
    if raw_entries_with_seeds:
        sh_secs = (
            (raw_entries_with_seeds[0][1].get("shifted_aggregate") or {}).get(
                "shift_seconds"
            )
            or []
        )
        sh_str = ", ".join(
            f"{int(round(s)):+d}s"
            if abs(float(s) - round(float(s))) < 1e-6
            else f"{float(s):+.1f}s"
            for s in sh_secs
            if s is not None
        )
        raw_lbl = (
            f"Raw Temporal Shift [{sh_str}] (eta=0.0)"
            if sh_str
            else "Raw Temporal Shift (eta=0.0)"
        )
        raw_nc_agg = _aggregate_control_entries_across_seeds(
            raw_entries_with_seeds,
            method_id="raw",
            method_label=raw_lbl,
            control_type="temporal_shift",
            eta=0.0,
            **ctrl_common_kwargs,
        )
        if raw_nc_agg.get("enabled"):
            methods_map["raw"] = raw_nc_agg

    if zero_entries_with_seeds:
        zero_agg = _aggregate_control_entries_across_seeds(
            zero_entries_with_seeds,
            method_id="zero_signal",
            method_label="Zero-Signal Control (All-Zero Input)",
            control_type="zero_signal",
            eta=0.0,
            **ctrl_common_kwargs,
        )
        if zero_agg.get("enabled"):
            methods_map["zero_signal"] = zero_agg

    def _normalize_cms_sweep_dict(sweep_obj: object) -> dict[str, dict]:
        if isinstance(sweep_obj, dict):
            return {
                str(k): v for k, v in sweep_obj.items() if isinstance(v, dict)
            }
        if isinstance(sweep_obj, list):
            out_d: dict[str, dict] = {}
            for item in sweep_obj:
                if isinstance(item, dict):
                    eta_v = float(item.get("eta", 0.0))
                    k_str = (
                        "eta_0"
                        if abs(eta_v) <= 1e-9
                        else f"eta_{eta_v:g}".replace(".", "_")
                    )
                    out_d[k_str] = item
            return out_d
        return {}

    def _normalize_sub_sweep_dict(sweep_obj: object) -> dict[str, dict]:
        if isinstance(sweep_obj, dict):
            return {
                str(k): v for k, v in sweep_obj.items() if isinstance(v, dict)
            }
        if isinstance(sweep_obj, list):
            out_d: dict[str, dict] = {}
            for idx_item, item in enumerate(sweep_obj):
                if isinstance(item, dict):
                    rk = int(
                        item.get("rank_k")
                        if item.get("rank_k") is not None
                        else (item.get("k") or 0)
                    )
                    eta_v = float(item.get("eta") or 0.0)
                    gate_b = bool(
                        item.get(
                            "use_coherence_gate",
                            str(item.get("gate", "off")).lower()
                            in {"on", "true", "1"},
                        )
                    )
                    gate_tag = "on" if gate_b else "off"
                    k_str = str(
                        item.get("key")
                        or f"subspace_k{rk}_eta{eta_v:g}_gate_{gate_tag}"
                    )
                    norm_item = dict(item)
                    norm_item.setdefault("rank_k", rk)
                    norm_item.setdefault("eta", eta_v)
                    norm_item.setdefault("use_coherence_gate", gate_b)
                    out_d[k_str] = norm_item
            return out_d
        return {}

    seed_word_top10_map: dict[int, float] = {
        int(r["seed"]): float(r["full_vocab_word_macro_top10"])
        for r in per_seed
        if r.get("full_vocab_word_macro_top10") is not None
    }

    # Load Subspace Coherence LOO Debiasing across seeds first so its upgraded full_rank_loo_baselines can repair stale CMS entries
    sub_reports_by_seed: list[tuple[int, dict]] = []
    sub_full_loo_by_seed_and_eta: dict[tuple[int, float], dict] = {}
    for r in seed_results:
        sd = int(r.get("seed", 0))
        sub_rep = _load_seed_exploratory_json(
            r,
            split=split,
            json_key="exploratory_subspace_coherence_debiasing_json",
            filename_prefix="subspace_coherence_debiasing",
        )
        if (
            isinstance(sub_rep, dict)
            and sub_rep.get("subspace_sweep") is not None
        ):
            sub_reports_by_seed.append((sd, sub_rep))
            rb = sub_rep.get("raw_baseline")
            if isinstance(rb, dict) and (
                (rb.get("clean_reference") or {}).get("full_vocab_word_macro_top10") is not None
                or (rb.get("shifted_aggregate") or {}).get("mean_full_vocab_word_macro_top10") is not None
            ):
                _repair_control_entry_clean_ref(
                    rb,
                    true_word_top10=seed_word_top10_map.get(sd),
                )
                sub_full_loo_by_seed_and_eta[(sd, 0.0)] = rb
            sr_fc = _normalize_cms_sweep_dict(
                sub_rep.get("full_loo_cms")
                or sub_rep.get("full_rank_cms_sweep")
                or sub_rep.get("full_rank_loo_baselines")
            )
            for fc_ent in sr_fc.values():
                if isinstance(fc_ent, dict) and (
                    (fc_ent.get("clean_reference") or {}).get("full_vocab_word_macro_top10") is not None
                    or (fc_ent.get("shifted_aggregate") or {}).get("mean_full_vocab_word_macro_top10") is not None
                ):
                    sub_full_loo_by_seed_and_eta[
                        (sd, round(float(fc_ent.get("eta", 0.0)), 6))
                    ] = fc_ent

    # Aggregate Leave-One-Out Common-Mode Subtraction (CMS) across seeds if available
    cms_reports_by_seed: list[tuple[int, dict]] = []
    for r in seed_results:
        sd = int(r.get("seed", 0))
        cms_rep = _load_seed_exploratory_json(
            r,
            split=split,
            json_key="exploratory_common_mode_subtraction_json",
            filename_prefix="common_mode_subtraction",
        )
        if isinstance(cms_rep, dict) and cms_rep.get("sweep") is not None:
            cms_reports_by_seed.append((sd, cms_rep))

    cms_aggregate: dict[str, object] = {"enabled": False}
    if cms_reports_by_seed:
        first_sweep = _normalize_cms_sweep_dict(
            cms_reports_by_seed[0][1].get("sweep")
        )
        cms_methods: dict[str, dict] = {}
        for eta_key, first_eta_ent in first_sweep.items():
            eta_val = float(first_eta_ent.get("eta", 0.0))
            eta_round = round(eta_val, 6)
            m_id = "raw" if abs(eta_val) <= 1e-9 else f"cms:eta={eta_val:g}"
            m_lbl = (
                "Raw Baseline (eta=0.0)"
                if abs(eta_val) <= 1e-9
                else f"LOO Common-Mode Subtraction (eta={eta_val:g})"
            )
            eta_seed_entries: list[tuple[int, dict]] = []
            for sd, c_rep in cms_reports_by_seed:
                sw_i = _normalize_cms_sweep_dict(c_rep.get("sweep"))
                ent_i = sw_i.get(eta_key)
                if isinstance(ent_i, dict):
                    c_ref_i = ent_i.get("clean_reference") or {}
                    s_agg_i = ent_i.get("shifted_aggregate") or {}
                    if (
                        c_ref_i.get("full_vocab_word_macro_top10") is None
                        or s_agg_i.get("mean_full_vocab_word_macro_top10") is None
                    ) and (sd, eta_round) in sub_full_loo_by_seed_and_eta:
                        ent_i = sub_full_loo_by_seed_and_eta[(sd, eta_round)]
                    if abs(eta_val) <= 1e-9:
                        _repair_control_entry_clean_ref(
                            ent_i,
                            true_word_top10=seed_word_top10_map.get(sd),
                        )
                    eta_seed_entries.append((sd, ent_i))
            if not eta_seed_entries:
                continue
            agg_eta = _aggregate_control_entries_across_seeds(
                eta_seed_entries,
                method_id=m_id,
                method_label=m_lbl,
                control_type="common_mode_subtraction",
                eta=eta_val,
                baseline_raw_agg=raw_nc_agg,
                **ctrl_common_kwargs,
            )
            if agg_eta.get("enabled"):
                cms_methods[eta_key] = agg_eta
                if abs(eta_val) > 1e-9:
                    methods_map[m_id] = agg_eta
                elif (
                    "raw" not in methods_map
                    or agg_eta.get("mean_shifted_full_vocab_word_macro_top10")
                    is not None
                ):
                    methods_map["raw"] = agg_eta
                    raw_nc_agg = agg_eta
        if cms_methods:
            cms_aggregate = {
                "enabled": True,
                "seed_count": len(cms_reports_by_seed),
                "eta_grid": list(
                    cms_reports_by_seed[0][1].get("eta_grid")
                    or [float(m.get("eta", 0.0)) for m in cms_methods.values()]
                ),
                "methods": cms_methods,
                "sweep": list(cms_methods.values()),
            }

    subspace_aggregate: dict[str, object] = {"enabled": False}
    if sub_reports_by_seed:
        first_sub = sub_reports_by_seed[0][1]
        full_cms_raw = (
            first_sub.get("full_loo_cms")
            or first_sub.get("full_rank_cms_sweep")
            or first_sub.get("full_rank_loo_baselines")
        )
        full_cms_methods_dict: dict[str, dict] = {}
        if full_cms_raw is not None:
            norm_fc = _normalize_cms_sweep_dict(full_cms_raw)
            if isinstance(first_sub.get("raw_baseline"), dict):
                raw_seed_ents = []
                for sd, sr in sub_reports_by_seed:
                    if isinstance(sr.get("raw_baseline"), dict):
                        _repair_control_entry_clean_ref(
                            sr["raw_baseline"],
                            true_word_top10=seed_word_top10_map.get(sd),
                        )
                        raw_seed_ents.append((sd, sr["raw_baseline"]))
                if raw_seed_ents:
                    agg_r0 = _aggregate_control_entries_across_seeds(
                        raw_seed_ents,
                        method_id="raw",
                        method_label="Raw Baseline (eta=0.0)",
                        control_type="temporal_shift",
                        eta=0.0,
                        **ctrl_common_kwargs,
                    )
                    if agg_r0.get("enabled"):
                        full_cms_methods_dict["eta_0"] = agg_r0
                        if (
                            "raw" not in methods_map
                            or agg_r0.get(
                                "mean_shifted_full_vocab_word_macro_top10"
                            )
                            is not None
                        ):
                            methods_map["raw"] = agg_r0
                            raw_nc_agg = agg_r0
            for f_key, f_ent in norm_fc.items():
                eta_v = float(f_ent.get("eta", 0.0))
                m_id = f"cms:eta={eta_v:g}"
                m_lbl = f"LOO Common-Mode Subtraction (eta={eta_v:g})"
                f_seed_ents = []
                for sd, sr in sub_reports_by_seed:
                    sr_fc = _normalize_cms_sweep_dict(
                        sr.get("full_loo_cms")
                        or sr.get("full_rank_cms_sweep")
                        or sr.get("full_rank_loo_baselines")
                    )
                    if isinstance(sr_fc.get(f_key), dict):
                        f_seed_ents.append((sd, sr_fc[f_key]))
                agg_f = _aggregate_control_entries_across_seeds(
                    f_seed_ents,
                    method_id=m_id,
                    method_label=m_lbl,
                    control_type="common_mode_subtraction",
                    eta=eta_v,
                    baseline_raw_agg=raw_nc_agg,
                    **ctrl_common_kwargs,
                )
                if agg_f.get("enabled"):
                    full_cms_methods_dict[f_key] = agg_f
                    if (
                        m_id not in methods_map
                        or agg_f.get("mean_clean_full_vocab_word_macro_top10") is not None
                    ):
                        methods_map[m_id] = agg_f
                        if isinstance(cms_aggregate.get("methods"), dict):
                            cms_aggregate["methods"][f_key] = agg_f
                            cms_aggregate["sweep"] = list(cms_aggregate["methods"].values())
            if not cms_aggregate.get("enabled") and full_cms_methods_dict:
                cms_aggregate = {
                    "enabled": True,
                    "seed_count": len(sub_reports_by_seed),
                    "methods": full_cms_methods_dict,
                    "sweep": list(full_cms_methods_dict.values()),
                }

        sub_methods: dict[str, dict] = {}
        first_sub_sweep = _normalize_sub_sweep_dict(
            first_sub.get("subspace_sweep")
        )
        for sub_key, first_sub_ent in first_sub_sweep.items():
            rk = int(
                first_sub_ent.get("rank_k")
                if first_sub_ent.get("rank_k") is not None
                else (first_sub_ent.get("k") or 0)
            )
            eta_v = float(first_sub_ent.get("eta") or 0.0)
            gate_b = bool(
                first_sub_ent.get(
                    "use_coherence_gate",
                    str(first_sub_ent.get("gate", "off")).lower()
                    in {"on", "true", "1"},
                )
            )
            gate_tag = "on" if gate_b else "off"
            m_id = f"subspace:K={rk},eta={eta_v:g},gate={gate_tag}"
            m_lbl = str(
                first_sub_ent.get(
                    "label",
                    f"Subspace LOO (K={rk}, eta={eta_v:g}, Gate={'On' if gate_b else 'Off'})",
                )
            )
            sub_seed_ents = []
            for sd, sr in sub_reports_by_seed:
                sr_sw = _normalize_sub_sweep_dict(sr.get("subspace_sweep"))
                if isinstance(sr_sw.get(sub_key), dict):
                    sub_seed_ents.append((sd, sr_sw[sub_key]))
            if not sub_seed_ents:
                continue
            agg_sub = _aggregate_control_entries_across_seeds(
                sub_seed_ents,
                method_id=m_id,
                method_label=m_lbl,
                control_type="subspace_coherence_debiasing",
                eta=eta_v,
                rank_k=rk,
                use_coherence_gate=gate_b,
                baseline_raw_agg=raw_nc_agg,
                **ctrl_common_kwargs,
            )
            if agg_sub.get("enabled"):
                sub_methods[sub_key] = agg_sub
                methods_map[m_id] = agg_sub
                methods_map[sub_key] = agg_sub

        # Also aggregate subspace diagnostics (top singular values & coherence stats across seeds)
        diag_list = [
            sr.get("subspace_and_coherence_diagnostics")
            for _, sr in sub_reports_by_seed
            if isinstance(sr.get("subspace_and_coherence_diagnostics"), dict)
        ]
        diag_agg: dict[str, object] = {}
        if diag_list:
            main_diags = [
                d.get("main")
                for d in diag_list
                if isinstance(d.get("main"), dict)
            ]
            if main_diags:
                for dk in (
                    "clean_coherence_mean",
                    "clean_coherence_median",
                    "shifted_loo_coherence_mean",
                    "shifted_loo_coherence_median",
                    "coherence_separation_shifted_minus_clean",
                ):
                    vals = [
                        float(md[dk])
                        for md in main_diags
                        if dk in md and md[dk] is not None
                    ]
                    if vals:
                        arr = np.asarray(vals, dtype=np.float64)
                        diag_agg[f"mean_{dk}"] = float(np.mean(arr))
                        diag_agg[f"std_{dk}"] = float(np.std(arr, ddof=0))
        if sub_methods:
            subspace_aggregate = {
                "enabled": True,
                "seed_count": len(sub_reports_by_seed),
                "grid": first_sub.get("grid") or {},
                "diagnostics_aggregate": diag_agg,
                "methods": sub_methods,
                "subspace_sweep": list(sub_methods.values()),
                "full_rank_cms_sweep": list(full_cms_methods_dict.values())
                if full_cms_methods_dict
                else list((cms_aggregate.get("methods") or {}).values()),
            }

    # Expose canonical unique method keys in negative_controls_by_method
    canonical_methods_map: dict[str, dict] = {}
    for k, v in methods_map.items():
        if k.startswith("subspace_k"):
            continue
        canonical_methods_map[k] = v

    summary_out: dict[str, object] = {
        "status": "multi_seed_evaluation_completed",
        "dataset": dataset,
        "stage": stage,
        "stage_display_name": STAGE_DISPLAY_NAMES.get(stage, stage),
        "exp_tag": exp_tag,
        "word_exp_tag": word_exp_tag,
        "split": split,
        "seeds": [int(r["seed"]) for r in per_seed],
        "seed_count": len(per_seed),
        "window_seconds": win_sec,
        "window_samples": win_samp,
        "window_start_offset_seconds": win_off,
        "subset": subset,
        "per_seed": per_seed,
        "vocabulary_sweep_aggregate": vocab_sweep_agg,
        "linguistic_breakdown_aggregate": ling_agg,
        "revision_horizons_aggregate": horizons_agg,
        "revision_transitions_aggregate": transitions_agg,
        "per_subject_summary_aggregate": per_subject_agg,
        "negative_controls_by_method": canonical_methods_map,
        "negative_control_methods_aggregate": canonical_methods_map,
        "common_mode_subtraction_aggregate": cms_aggregate,
        "subspace_debiasing_aggregate": subspace_aggregate,
        "negative_controls_aggregate": {"enabled": False},
    }
    return apply_multi_seed_control_selection(
        summary_out,
        control_method=control_method,
        control_metric=control_metric,
    )


def summarize_multi_seed_evaluation(
    seed_results: Sequence[dict] | dict,
    *,
    split: str | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = False,
    print_report: bool = True,
    save_files: bool = True,
    control_method: str | Iterable[str] | None = None,
    control_metric: str | Iterable[str] | None = None,
    **_extra_kwargs,
) -> dict:
    """将多个随机种子（或单种子 n=1）的评估结果汇总为 Multi-Seed Summary JSON 与报表并落盘。"""
    if isinstance(seed_results, dict):
        seed_results = list(seed_results.values())
    if not seed_results:
        raise ValueError("seed_results 不能为空。")
    split = _infer_split_from_seed_results(seed_results, split)
    first = seed_results[0]
    is_multi_stage = (
        "stages" in first and isinstance(first.get("stages"), dict)
    )
    if is_multi_stage:
        stages_agg: dict[str, dict] = {}
        for st_key in ("word", "context", "revision", "prefix_revision"):
            st_list = [
                r["stages"][st_key]
                for r in seed_results
                if isinstance(r.get("stages"), dict) and st_key in r["stages"]
            ]
            if st_list:
                stages_agg[st_key] = _aggregate_single_stage_multi_seed(
                    st_list,
                    split=split,
                    control_method=control_method,
                    control_metric=control_metric,
                )
        target_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else Path(str(first.get("output_dir", "."))).resolve().parent
        )
        first_st_agg = next(iter(stages_agg.values()), {})
        summary = {
            "status": "multi_seed_stage_comparison_completed",
            "dataset": str(first.get("dataset", "-")),
            "exp_tag": str(first.get("exp_tag", "")),
            "word_exp_tag": str(first.get("word_exp_tag", "")),
            "split": split,
            "seeds": [int(r.get("seed", 0)) for r in seed_results],
            "seed_count": len(seed_results),
            "window_seconds": float(first.get("window_seconds", 1.0)),
            "window_samples": int(first.get("window_samples", 50)),
            "window_start_offset_seconds": float(
                first.get("window_start_offset_seconds", 0.0)
            ),
            "subset": first.get("subset", {}),
            "evaluated_stages": list(stages_agg.keys()),
            "requested_control_method": str(
                first_st_agg.get(
                    "requested_control_method", control_method or "auto"
                )
            ),
            "requested_control_metric": str(
                first_st_agg.get(
                    "requested_control_metric",
                    control_metric or "vocab50_macro_top10",
                )
            ),
            "selected_control_methods": list(
                first_st_agg.get("selected_control_methods") or []
            ),
            "selected_control_metrics": list(
                first_st_agg.get("selected_control_metrics") or []
            ),
            "stages_aggregate": stages_agg,
            "seed_results": list(seed_results),
            "output_dir": str(target_dir),
        }
        if save_files:
            target_dir.mkdir(parents=True, exist_ok=True)
            json_path = target_dir / f"multi_seed_stage_comparison_{split}.json"
            multi_exports = export_multi_seed_tables_and_markdown(
                summary, target_dir
            )
            summary["exports"] = multi_exports
            save_json(json_path, summary)
            summary["evaluation_json"] = str(json_path)
            print_multi_seed_evaluation_report(
                summary,
                verbose=verbose,
                output_dir=target_dir,
                print_to_stdout=print_report,
                control_method=control_method,
                control_metric=control_metric,
            )
        return summary

    summary = _aggregate_single_stage_multi_seed(
        seed_results,
        split=split,
        control_method=control_method,
        control_metric=control_metric,
    )
    target_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else Path(str(first.get("output_dir", "."))).resolve().parent
    )
    summary["seed_results"] = list(seed_results)
    summary["output_dir"] = str(target_dir)
    if save_files:
        target_dir.mkdir(parents=True, exist_ok=True)
        json_path = target_dir / f"multi_seed_evaluation_{split}.json"
        multi_exports = export_multi_seed_tables_and_markdown(
            summary, target_dir
        )
        summary["exports"] = multi_exports
        save_json(json_path, summary)
        summary["evaluation_json"] = str(json_path)
        print_multi_seed_evaluation_report(
            summary,
            verbose=verbose,
            output_dir=target_dir,
            print_to_stdout=print_report,
            control_method=control_method,
            control_metric=control_metric,
        )
    return summary


def run_evaluation_workflow(
    dataset: str | None = None,
    *,
    stage: str | None = None,
    seed: int | str | Iterable[int | str] | None = 0,
    split: str = "val",
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    precision: str | None = None,
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
    require_cache: bool = False,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    config_path: str | Path | None = None,
    control_config_path: str | Path | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    zero_signal_control: bool = False,
    subtraction_etas: str | Iterable[float | int | str] | None = None,
    control_method: str | Iterable[str] | None = None,
    control_metric: str | Iterable[str] | None = None,
    all_stages: bool = False,
    plot: bool = True,
    show_progress: bool = True,
    max_eval_rows: int | None = None,
    verbose: bool = False,
    _export_outer_multi_seed: bool = True,
) -> dict:
    """运行验证集 (val) 或测试集 (test) 评估与可视化工作流（支持单阶段、全阶段对比、多种子聚合、负对照及探索性留一法共模扣除）。"""
    from braindecoding.training.workflow_runner import (
        WORKFLOW_STAGES,
        _resolve_results_root_and_exp_tag,
        _resolve_word_exp_tag,
        build_workflow_config,
        workflow_output_directory,
    )

    if split not in {"val", "test"}:
        raise ValueError(f"评估 split 只能是 'val' 或 'test'，收到：{split!r}")

    seed_list = _parse_seed_list(seed)
    if len(seed_list) > 1:
        seed_results: list[dict] = []
        for idx, sd_val in enumerate(seed_list):
            print(
                f"\n{'#' * 86}\n"
                f" [Multi-Seed Evaluation {idx + 1}/{len(seed_list)}] Evaluating seed={sd_val} (all seeds: {seed_list})\n"
                f"{'#' * 86}",
                flush=True,
            )
            res_i = run_evaluation_workflow(
                dataset,
                stage=stage,
                seed=sd_val,
                split=split,
                subject=subject,
                num_subjects=num_subjects,
                subjects=subjects,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                precision=precision,
                window_seconds=window_seconds,
                window_start_offset_seconds=window_start_offset_seconds,
                require_cache=require_cache,
                checkpoint_path=None,
                output_dir=None,
                exp_tag=exp_tag,
                word_exp_tag=word_exp_tag,
                config_path=config_path,
                control_config_path=control_config_path,
                shift_seconds=shift_seconds,
                zero_signal_control=zero_signal_control,
                subtraction_etas=subtraction_etas,
                control_method=control_method,
                control_metric=control_metric,
                all_stages=all_stages,
                plot=plot,
                show_progress=show_progress,
                max_eval_rows=max_eval_rows,
                verbose=verbose,
                _export_outer_multi_seed=False,
            )
            seed_results.append(res_i)
        return summarize_multi_seed_evaluation(
            seed_results,
            split=split,
            output_dir=output_dir,
            verbose=verbose,
            print_report=True,
            control_method=control_method,
            control_metric=control_metric,
        )

    seed = int(seed_list[0])
    if not all_stages:
        single_res = _evaluate_single_stage(
            dataset,
            stage=stage,
            seed=seed,
            split=split,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            precision=precision,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            require_cache=require_cache,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            exp_tag=exp_tag,
            word_exp_tag=word_exp_tag,
            config_path=config_path,
            control_config_path=control_config_path,
            shift_seconds=shift_seconds,
            zero_signal_control=zero_signal_control,
            subtraction_etas=subtraction_etas,
            plot=plot,
            show_progress=show_progress,
            max_eval_rows=max_eval_rows,
            verbose=verbose,
            print_report=True,
        )
        if plot:
            ds_name = single_res["dataset"]
            sub_tag = single_res["subset"]["subset_tag"]
            sd = single_res["seed"]
            active_tag = single_res.get("exp_tag", "")
            active_word_tag = single_res.get("word_exp_tag", active_tag)
            discovered: dict[str, dict] = {single_res["stage"]: single_res}
            for other_st in WORKFLOW_STAGES:
                if other_st == single_res["stage"]:
                    continue
                target_tag = (
                    active_word_tag
                    if other_st == "word" and active_word_tag != active_tag
                    else active_tag
                )
                other_dir = workflow_output_directory(
                    ds_name,
                    other_st,
                    sub_tag,
                    sd,
                    mode="train",
                    exp_tag=target_tag,
                    ignore_implicit_tag=(
                        other_st == "word" and target_tag == ""
                    ),
                )
                other_eval_file = other_dir / f"evaluation_{split}.json"
                if not other_eval_file.is_file():
                    smoke_other = (
                        workflow_output_directory(
                            ds_name,
                            other_st,
                            sub_tag,
                            sd,
                            mode="smoke",
                            exp_tag=target_tag,
                            ignore_implicit_tag=(
                                other_st == "word" and target_tag == ""
                            ),
                        )
                        / f"evaluation_{split}.json"
                    )
                    if smoke_other.is_file():
                        other_eval_file = smoke_other
                if (
                    not other_eval_file.is_file()
                    and target_tag
                    and other_st == "word"
                ):
                    fallback_other = (
                        workflow_output_directory(
                            ds_name,
                            other_st,
                            sub_tag,
                            sd,
                            mode="train",
                            ignore_implicit_tag=True,
                        )
                        / f"evaluation_{split}.json"
                    )
                    if fallback_other.is_file():
                        other_eval_file = fallback_other
                if other_eval_file.is_file():
                    try:
                        other_loaded = json.loads(
                            other_eval_file.read_text(encoding="utf-8")
                        )
                        single_nc_conds = (
                            (single_res.get("negative_controls") or {}).get(
                                "conditions"
                            )
                            or []
                        )
                        other_nc_conds = (
                            (other_loaded.get("negative_controls") or {}).get(
                                "conditions"
                            )
                            or []
                        )
                        if single_nc_conds and other_nc_conds:
                            single_keys = {c.get("key") for c in single_nc_conds}
                            other_keys = {c.get("key") for c in other_nc_conds}
                            if single_keys != other_keys:
                                other_loaded = dict(other_loaded)
                                other_loaded["negative_controls"] = {}
                        discovered[other_st] = other_loaded
                    except Exception:
                        pass
            if len(discovered) >= 1:
                comp_figs = plot_stage_comparison_figure(
                    discovered,
                    Path(single_res["output_dir"]) / "figures",
                    split=split,
                )
                single_res["figures"].update(comp_figs)
                save_json(Path(single_res["evaluation_json"]), single_res)
        if _export_outer_multi_seed:
            outer_dir = (
                Path(str(single_res["output_dir"])).resolve().parent
                if output_dir is None
                else Path(output_dir).resolve().parent
            )
            outer_summary = summarize_multi_seed_evaluation(
                [single_res],
                split=split,
                output_dir=outer_dir,
                verbose=verbose,
                print_report=True,
                control_method=control_method,
                control_metric=control_metric,
            )
            single_res["outer_multi_seed_evaluation_json"] = str(
                outer_summary.get("evaluation_json", "")
            )
            single_res["outer_multi_seed_exports"] = outer_summary.get(
                "exports", {}
            )
        return single_res

    base_cfg, subset_spec = build_workflow_config(
        dataset,
        stage="word",
        seed=seed,
        subject=subject,
        num_subjects=num_subjects,
        subjects=subjects,
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        config_path=config_path,
    )
    effective_control_config_path = control_config_path
    if effective_control_config_path is None and config_path is not None:
        resolved_cfg_p = resolve_config_path(project_path(config_path))
        if resolved_cfg_p.is_file():
            raw_explicit = load_yaml_with_extends(resolved_cfg_p)
            if isinstance(raw_explicit, dict) and (
                "control" in raw_explicit
                or "shift_seconds" in (raw_explicit.get("evaluation") or {})
                or bool(
                    (raw_explicit.get("evaluation") or {}).get(
                        "zero_signal_control"
                    )
                )
            ):
                effective_control_config_path = config_path

    ds_name = subset_spec["dataset"]
    sub_tag = subset_spec["subset_tag"]
    sd = int(base_cfg["training"]["seed"])
    base_root, resolved_exp_tag = _resolve_results_root_and_exp_tag(
        base_cfg["workflow"].get("exp_tag", exp_tag)
    )
    has_explicit_word_tag, resolved_word_exp_tag = _resolve_word_exp_tag(
        word_exp_tag
    )

    has_stage2_under_exp_tag = False
    for st2 in ("context", "revision", "prefix_revision"):
        for mode_cand in ("train", "smoke"):
            cand_dir2 = workflow_output_directory(
                ds_name,
                st2,
                sub_tag,
                sd,
                mode=mode_cand,
                exp_tag=resolved_exp_tag,
                ignore_implicit_tag=False,
            )
            if (cand_dir2 / "best.ckpt").is_file() or (
                cand_dir2 / "best.pt"
            ).is_file():
                has_stage2_under_exp_tag = True
                break
        if has_stage2_under_exp_tag:
            break

    available_stages: list[tuple[str, str, bool]] = []
    for st in WORKFLOW_STAGES:
        candidate_tags: list[tuple[str, bool]] = []
        if st == "word":
            if has_explicit_word_tag and (
                has_stage2_under_exp_tag or word_exp_tag is not None
            ):
                candidate_tags.append(
                    (resolved_word_exp_tag, resolved_word_exp_tag == "")
                )
            if (resolved_exp_tag, False) not in candidate_tags:
                candidate_tags.append((resolved_exp_tag, False))
            if (
                has_explicit_word_tag
                and (resolved_word_exp_tag, resolved_word_exp_tag == "")
                not in candidate_tags
            ):
                candidate_tags.append(
                    (resolved_word_exp_tag, resolved_word_exp_tag == "")
                )
            if resolved_exp_tag and ("", True) not in candidate_tags:
                candidate_tags.append(("", True))
        else:
            candidate_tags.append((resolved_exp_tag, False))

        found = False
        for tag_cand, ignore_imp in candidate_tags:
            for mode_candidate in ("train", "smoke"):
                cand_dir = workflow_output_directory(
                    ds_name,
                    st,
                    sub_tag,
                    sd,
                    mode=mode_candidate,
                    exp_tag=tag_cand,
                    ignore_implicit_tag=ignore_imp,
                )
                if (cand_dir / "best.ckpt").is_file() or (
                    cand_dir / "best.pt"
                ).is_file():
                    available_stages.append((st, tag_cand, ignore_imp))
                    found = True
                    break
            if found:
                break

    if not available_stages:
        expected_dir = workflow_output_directory(
            ds_name, "word", sub_tag, sd, mode="train", exp_tag=resolved_exp_tag
        )
        raise FileNotFoundError(
            f"未找到数据集 {ds_name} (subset={sub_tag}, seed={sd}) 任何已训练阶段的检查点（如 {expected_dir / 'best.ckpt'}）。"
        )

    stage_results: dict[str, dict] = {}
    for st, st_tag, st_ignore_imp in available_stages:
        stage_out_dir = (
            workflow_output_directory(
                ds_name,
                st,
                sub_tag,
                sd,
                mode="train",
                exp_tag="",
                ignore_implicit_tag=True,
            )
            if st_ignore_imp
            else None
        )
        stage_results[st] = _evaluate_single_stage(
            ds_name,
            stage=st,
            seed=sd,
            split=split,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            precision=precision,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            require_cache=require_cache,
            output_dir=stage_out_dir,
            exp_tag=st_tag,
            word_exp_tag=word_exp_tag,
            config_path=None,
            control_config_path=effective_control_config_path,
            shift_seconds=shift_seconds,
            zero_signal_control=zero_signal_control,
            subtraction_etas=subtraction_etas,
            plot=plot,
            show_progress=show_progress,
            max_eval_rows=max_eval_rows,
            verbose=verbose,
            print_report=False,
        )

    effective_summary_tag = (
        available_stages[0][1]
        if len(available_stages) == 1
        else resolved_exp_tag
    )
    dataset_eval_root = base_root / ds_name
    if effective_summary_tag:
        dataset_eval_root = dataset_eval_root / effective_summary_tag
    comp_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (
            dataset_eval_root
            / f"evaluation_{split}"
            / sub_tag
            / f"seed-{sd:03d}"
        ).resolve()
    )
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_figs = {}
    if plot:
        comp_figs = plot_stage_comparison_figure(
            stage_results, comp_dir / "figures", split=split
        )
        for st_res in stage_results.values():
            st_figs = plot_stage_comparison_figure(
                stage_results, Path(st_res["output_dir"]) / "figures", split=split
            )
            st_res["figures"].update(st_figs)
            save_json(Path(st_res["evaluation_json"]), st_res)

    stage_significance: dict[str, dict] = {}
    if "word" in stage_results:
        w_subs = (
            (stage_results["word"].get("metrics") or {}).get("per_subject")
            or stage_results["word"].get("per_subject")
            or []
        )
        w_map = {
            str(s.get("subject_id")): float(
                s.get("macro_top10", s.get("n50_macro_top10", 0.0))
            )
            for s in w_subs
            if s.get("subject_id") is not None
        }
        for st_comp, comp_key in (
            ("context", "context_vs_word_n50_top10"),
            ("revision", "revision_vs_word_n50_top10"),
            ("prefix_revision", "prefix_revision_vs_word_n50_top10"),
        ):
            if st_comp in stage_results:
                c_subs = (
                    (stage_results[st_comp].get("metrics") or {}).get(
                        "per_subject"
                    )
                    or stage_results[st_comp].get("per_subject")
                    or []
                )
                c_map = {
                    str(s.get("subject_id")): float(
                        s.get("macro_top10", s.get("n50_macro_top10", 0.0))
                    )
                    for s in c_subs
                    if s.get("subject_id") is not None
                }
                common_sids = sorted(set(w_map.keys()) & set(c_map.keys()))
                if common_sids:
                    a_arr = np.asarray(
                        [c_map[sid] for sid in common_sids], dtype=np.float64
                    )
                    b_arr = np.asarray(
                        [w_map[sid] for sid in common_sids], dtype=np.float64
                    )
                    sig_res = _compute_paired_block_significance(
                        a_arr,
                        b_arr,
                        [str(sid) for sid in common_sids],
                        label=f"{st_comp} vs word (N50 Macro Top-10)",
                        metric="vocab50_macro_top10",
                        seed=20261029,
                    )
                    stage_significance[comp_key] = sig_res
                    stage_significance[
                        f"{st_comp}_minus_word_vocab50_top10"
                    ] = sig_res

    multi_summary = {
        "status": "multi_stage_evaluation_completed",
        "dataset": ds_name,
        "exp_tag": effective_summary_tag,
        "word_exp_tag": resolved_word_exp_tag
        if has_explicit_word_tag
        else resolved_exp_tag,
        "split": split,
        "seed": sd,
        "window_seconds": float(base_cfg["workflow"]["window_seconds"]),
        "window_samples": int(base_cfg["workflow"]["window_samples"]),
        "window_start_offset_seconds": float(
            base_cfg["workflow"]["window_start_offset_seconds"]
        ),
        "subset": subset_spec,
        "evaluated_stages": [st for st, _, _ in available_stages],
        "stages": stage_results,
        "stage_significance": stage_significance,
        "output_dir": str(comp_dir),
        "figures": comp_figs,
    }
    comp_exports = export_stage_comparison_tables_and_markdown(
        multi_summary, comp_dir
    )
    multi_summary["exports"] = comp_exports
    summary_file = comp_dir / f"multi_stage_evaluation_{split}.json"
    save_json(summary_file, multi_summary)
    multi_summary["evaluation_json"] = str(summary_file)
    print_stage_comparison_report(
        multi_summary, split=split, verbose=verbose, output_dir=comp_dir
    )
    if _export_outer_multi_seed:
        for st_res in stage_results.values():
            st_outer_dir = Path(str(st_res["output_dir"])).resolve().parent
            summarize_multi_seed_evaluation(
                [st_res],
                split=split,
                output_dir=st_outer_dir,
                verbose=verbose,
                print_report=False,
                control_method=control_method,
                control_metric=control_metric,
            )
        outer_comp_dir = comp_dir.parent
        outer_comp_summary = summarize_multi_seed_evaluation(
            [multi_summary],
            split=split,
            output_dir=outer_comp_dir,
            verbose=verbose,
            print_report=True,
            control_method=control_method,
            control_metric=control_metric,
        )
        multi_summary["outer_multi_seed_evaluation_json"] = str(
            outer_comp_summary.get("evaluation_json", "")
        )
        multi_summary["outer_multi_seed_exports"] = outer_comp_summary.get(
            "exports", {}
        )
    return multi_summary


__all__ = [
    "DEFAULT_CONTROL_SHIFTS_SECONDS",
    "_aggregate_control_entries_across_seeds",
    "_aggregate_single_stage_multi_seed",
    "_build_enriched_full_vocabulary_metrics",
    "_compute_encoded_evaluation_metrics",
    "_compute_full_vocabulary_horizon_details",
    "_compute_paired_block_significance",
    "_compute_transition_counts",
    "_evaluate_single_stage",
    "_extract_control_summary_row",
    "_format_shift_label",
    "_parse_control_methods_list",
    "_parse_seed_list",
    "_parse_shift_seconds_list",
    "_resolve_control_metric_spec",
    "_resolve_control_metric_specs",
    "_resolve_control_specification",
    "_score_full_vocabulary_population",
    "_select_control_methods_list",
    "_significance_stars",
    "_summarize_shifted_conditions",
    "apply_multi_seed_control_selection",
    "build_enriched_full_vocabulary_metrics",
    "compute_encoded_evaluation_metrics",
    "compute_full_vocabulary_horizon_details",
    "evaluate_single_stage",
    "extract_control_summary_row",
    "resolve_control_specification",
    "run_evaluation_workflow",
    "score_full_vocabulary_population",
    "summarize_multi_seed_evaluation",
    "summarize_shifted_conditions",
]
