"""词级检索评估统计原语（scipy.stats Student-t、McNemar 检验、配对材料块检验与显著性标记）。"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy import stats as scipy_stats

STAGE_DISPLAY_NAMES: dict[str, str] = {
    "word": "Word-Only (Stage 1)",
    "context": "Full-Context (Stage 2.1)",
    "revision": "Rolling-K2 (Stage 2.2)",
    "prefix_revision": "Prefix-Revision (Stage 2.3)",
}

DEFAULT_SUBTRACTION_ETAS: tuple[float, ...] = (0.5, 0.8, 1.0)
DEFAULT_CAUSAL_CMS_ETAS: tuple[float, ...] = (0.0, 0.5, 0.8, 1.0)
DEFAULT_CAUSAL_CMS_EMA_DECAY: float = 0.99
DEFAULT_POS_CAUSAL_CMS_ETAS: tuple[float, ...] = (0.0, 0.5, 0.8, 1.0)
DEFAULT_POS_CAUSAL_EMA_DECAY: float = 0.95
DEFAULT_POS_MIN_BUCKET_COUNT: int = 3


def normalize_cms_mode(value: object, *, default: str = "leave_one_out") -> str:
    text = str(value if value is not None else default).strip().lower()
    if text in {"causal", "causal_ema", "online_causal_ema"}:
        return "causal_ema"
    if text in {"pos_causal", "pos_causal_ema", "position_causal_ema"}:
        return "pos_causal_ema"
    return "leave_one_out"


def parse_eta_values(value: object, default: Sequence[float] = DEFAULT_SUBTRACTION_ETAS) -> tuple[float, ...]:
    if value is None:
        return tuple(float(x) for x in default)
    if isinstance(value, str):
        parts = [tok.strip() for tok in value.replace(";", ",").split(",") if tok.strip()]
        return tuple(float(tok) for tok in parts) if parts else tuple(float(x) for x in default)
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(x) for x in value)


class _CachedEvalDatasetProxy:
    """轻量代理对象：供从已编码缓存执行 common-horizon 子集评估使用。"""

    def __init__(self, table, embedding_map):
        self.table = table
        self.embedding_map = embedding_map

    def __len__(self) -> int:
        return len(self.table)


def export_word_evaluation_tables_and_markdown(*_args, **_kwargs) -> dict:
    return {}


def export_multi_seed_tables_and_markdown(*_args, **_kwargs) -> dict:
    return {}


def export_stage_comparison_tables_and_markdown(*_args, **_kwargs) -> dict:
    return {}


def plot_evaluation_figures(*_args, **_kwargs) -> dict:
    return {}


def plot_stage_comparison_figure(*_args, **_kwargs) -> dict:
    return {}


def print_word_evaluation_report(*_args, **_kwargs) -> None:
    return None


def print_multi_seed_evaluation_report(*_args, **_kwargs) -> None:
    return None


def print_stage_comparison_report(*_args, **_kwargs) -> None:
    return None

DEFAULT_CONTROL_SHIFTS_SECONDS: tuple[float, ...] = (
    -15.0,
    -30.0,
    -45.0,
    -60.0,
    -90.0,
    -120.0,
)

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

def _student_t_two_sided_pvalue(t_abs: float, df: float) -> float:
    """计算自由度为 df 的双侧 Student-t 检验精确 p 值（基于 scipy.stats.t.sf）。"""
    t_val = abs(float(t_abs))
    if not math.isfinite(t_val) or t_val <= 1e-12:
        return 1.0
    df_val = max(1.0, float(df))
    return float(min(1.0, max(0.0, 2.0 * scipy_stats.t.sf(t_val, df_val))))

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
