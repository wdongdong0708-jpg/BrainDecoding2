"""Word-level retrieval evaluation, negative-control specification, and workflow runners.

Decouples word-level retrieval metric computation, temporal-shift / zero-signal
negative control resolution and aggregation, single-stage checkpoint evaluation,
and multi-stage comparison orchestration from the training loop in
``braindecoding.training.workflow_runner``.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable, Sequence

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
) -> dict[str, int | float]:
    """统计两个视界之间的四格命中转移矩阵（Miss->Hit 有益修正 vs. Hit->Miss 有害误改）。"""
    b = np.asarray(hits_before, dtype=bool).reshape(-1)
    a = np.asarray(hits_after, dtype=bool).reshape(-1)
    n_tot = int(len(b))
    if n_tot == 0 or len(a) != n_tot:
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
        }
    repairs = int(np.sum((~b) & a))
    harms = int(np.sum(b & (~a)))
    retained = int(np.sum(b & a))
    pers_miss = int(np.sum((~b) & (~a)))
    net_gain = repairs - harms
    denom = float(max(1, n_tot))
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
    revision = stage == "revision" and "revision_predictions" in encoded
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
    for size in (20, 50, 100, 150):
        cand_words = list(manifests[size]["vocabulary"])
        m = fixed_vocabulary_retrieval_metrics(
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
            "subject_macro_top1": mi_top1,
            "subject_macro_top5": mi_top5,
            "subject_macro_top10": w_top10,
            "subject_std_top10": 0.0,
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
            "subject_macro_top10": s50_w10,
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
        "macro_top10": fv_subj_top10,
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
            fv_subj_top10 / max(fv_chance_10, 1e-12), 4
        ),
        "delta_vs_chance_top10": round(fv_subj_top10 - fv_chance_10, 6),
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
            "full_vocab_top10": float(np.mean(subj_rfv <= 10.0)),
            "full_vocab_top1": float(np.mean(subj_rfv <= 1.0)),
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
                k0_fv = float(np.mean(ranks_full_by_delay[0][s_com] <= 10.0))
                k2_fv = float(np.mean(ranks_full_by_delay[2][s_com] <= 10.0))
                s_entry["k0_vocab50_top10"] = k0_s50
                s_entry["k2_vocab50_top10"] = k2_s50
                s_entry["k2_minus_k0_vocab50_top10"] = k2_s50 - k0_s50
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
            "mean_mrr": float(np.mean([float(s["mrr"]) for s in per_subject])),
            "std_mrr": float(
                np.std([float(s["mrr"]) for s in per_subject], ddof=0)
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
    linguistic_breakdown["content_minus_function"] = {
        "label": "Content - Function Delta",
        "query_count": cw_d["query_count"] - fw_d["query_count"],
        "unique_word_count": cw_d["unique_word_count"] - fw_d["unique_word_count"],
        "event_share_pct": round(
            cw_d["event_share_pct"] - fw_d["event_share_pct"], 2
        ),
        "vocab50_word_macro_top10": float(
            cw_d["vocab50_word_macro_top10"] - fw_d["vocab50_word_macro_top10"]
        ),
        "vocab50_micro_top10": float(
            cw_d["vocab50_micro_top10"] - fw_d["vocab50_micro_top10"]
        ),
        "vocab50_word_macro_top1": float(
            cw_d["vocab50_word_macro_top1"] - fw_d["vocab50_word_macro_top1"]
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
        "full_vocab_mrr": float(
            cw_d["full_vocab_mrr"] - fw_d["full_vocab_mrr"]
        ),
        "full_vocab_median_rank": float(
            cw_d["full_vocab_median_rank"] - fw_d["full_vocab_median_rank"]
        ),
    }

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
                "all_vocab50_macro_top10": float(
                    s50_all.get("macro_recall_at_10", 0.0)
                ),
                "common_full_vocab_macro_top10": float(
                    fv_common.get(
                        "subject_equal_macro_recall_at_10",
                        fv_horizon.get("subject_macro_top10", 0.0),
                    )
                ),
                "common_full_vocab_word_macro_top10": float(
                    fv_common.get("word_macro_recall_at_10", 0.0)
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
                    fv_all.get("subject_equal_macro_recall_at_10", 0.0)
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

    if shift_seconds is not None:
        shifts = _parse_shift_seconds_list(shift_seconds)
    elif "shift_seconds" in merged_control:
        shifts = _parse_shift_seconds_list(merged_control.get("shift_seconds"))
    elif "shift_seconds" in merged_eval:
        shifts = _parse_shift_seconds_list(merged_eval.get("shift_seconds"))
    elif cms_enabled:
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
    enabled = bool(shifts) or enable_zero or cms_enabled
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
        }

    default_type = (
        "common_mode_subtraction"
        if cms_enabled
        else (
            "temporal_shift"
            if shifts and not enable_zero
            else ("zero_signal" if enable_zero and not shifts else "combined_control")
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

    row = {
        "key": key,
        "label": label,
        "shift_seconds": shift_seconds,
        "vocab50_macro_top10": float(n50.get("macro_top10", 0.0)),
        "vocab50_micro_top10": float(n50.get("micro_top10", 0.0)),
        "vocab50_macro_top5": float(n50.get("macro_top5", 0.0)),
        "vocab50_macro_top1": float(n50.get("macro_top1", 0.0)),
        "vocab50_mrr": float(n50.get("mrr", 0.0)),
        "vocab50_median_rank": float(n50.get("median_rank", 0.0)),
        "vocab50_chance_top10": float(n50.get("chance_top10", 0.2)),
        "full_vocab_macro_top10": float(fv.get("macro_top10", 0.0)),
        "full_vocab_micro_top10": float(fv.get("micro_top10", 0.0)),
        "full_vocab_macro_top1": float(fv.get("macro_top1", 0.0)),
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
                "common_full_vocab_macro_top10": float(
                    horizons[h_key].get("common_full_vocab_macro_top10", 0.0)
                ),
                "common_full_vocab_micro_top10": float(
                    horizons[h_key].get("common_full_vocab_micro_top10", 0.0)
                ),
                "common_full_vocab_macro_top1": float(
                    horizons[h_key].get("common_full_vocab_macro_top1", 0.0)
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
            "vocab50_mrr": float(
                row["vocab50_mrr"] - clean_summary["vocab50_mrr"]
            ),
            "full_vocab_macro_top10": float(
                row["full_vocab_macro_top10"]
                - clean_summary["full_vocab_macro_top10"]
            ),
            "clean_minus_control_vocab50_macro_top10": float(
                clean_summary["vocab50_macro_top10"] - row["vocab50_macro_top10"]
            ),
            "clean_minus_control_full_vocab_macro_top10": float(
                clean_summary["full_vocab_macro_top10"]
                - row["full_vocab_macro_top10"]
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
                    "common_full_vocab_macro_top10": float(
                        row["horizons"][h_key]["common_full_vocab_macro_top10"]
                        - clean_summary["horizons"][h_key][
                            "common_full_vocab_macro_top10"
                        ]
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
        "vocab50_mrr",
        "vocab50_median_rank",
        "full_vocab_macro_top10",
        "full_vocab_micro_top10",
        "full_vocab_macro_top1",
        "full_vocab_mrr",
        "full_vocab_median_rank",
    )
    agg: dict = {
        "shift_count": len(shifted_rows),
        "shift_seconds": [r["shift_seconds"] for r in shifted_rows],
    }
    for m_key in metric_keys:
        vals = np.asarray(
            [float(r[m_key]) for r in shifted_rows], dtype=np.float64
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
        h_agg: dict = {}
        for h_key in ("K0", "K1", "K2"):
            if h_key not in clean_summary["horizons"]:
                continue
            v50_vals = np.asarray(
                [
                    float(r["horizons"][h_key]["common_vocab50_macro_top10"])
                    for r in shifted_rows
                    if h_key in r["horizons"]
                ],
                dtype=np.float64,
            )
            fv_vals = np.asarray(
                [
                    float(r["horizons"][h_key]["common_full_vocab_macro_top10"])
                    for r in shifted_rows
                    if h_key in r["horizons"]
                ],
                dtype=np.float64,
            )
            if len(v50_vals) > 0:
                m_v50 = float(np.mean(v50_vals))
                m_fv = float(np.mean(fv_vals))
                h_agg[h_key] = {
                    "mean_common_vocab50_macro_top10": m_v50,
                    "std_common_vocab50_macro_top10": float(
                        np.std(v50_vals, ddof=0)
                    ),
                    "delta_common_vocab50_macro_top10_vs_clean": float(
                        m_v50
                        - float(
                            clean_summary["horizons"][h_key][
                                "common_vocab50_macro_top10"
                            ]
                        )
                    ),
                    "mean_common_full_vocab_macro_top10": m_fv,
                    "std_common_full_vocab_macro_top10": float(
                        np.std(fv_vals, ddof=0)
                    ),
                }
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
            if cms_enabled:
                shifted_encoded_by_shift.append(
                    (float(shift_sec), cond_key, cond_label, shift_encoded)
                )
                shifted_ds_by_shift[float(shift_sec)] = shift_ds
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

    if cms_enabled and len(shifted_encoded_by_shift) >= 2:
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
            baseline_clean_summary=negative_controls_result["clean_reference"],
            baseline_condition_entries=negative_controls_result["conditions"],
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
        eval_result["exploratory_common_mode_subtraction_json"] = cms_report.get(
            "exploratory_json"
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


def _aggregate_single_stage_multi_seed(
    seed_results: Sequence[dict],
    *,
    split: str = "val",
) -> dict:
    """聚合同一阶段在多个随机种子上的词级检索、滚动修订视界与负对照指标（Mean ± SD）。"""
    if not seed_results:
        return {}
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
        sweep = metrics.get("vocabulary_sweep") or {}
        n50 = sweep.get("N50") or {}
        fv = sweep.get("full_vocab") or {}
        nc = res.get("negative_controls") or {}
        sh_agg = nc.get("shifted_aggregate") or {}
        row = {
            "seed": sd,
            "epoch": ep,
            "output_dir": str(res.get("output_dir", "")),
            "evaluation_json": str(res.get("evaluation_json", "")),
            "vocab50_macro_top10": float(n50.get("macro_top10", 0.0)),
            "vocab50_micro_top10": float(n50.get("micro_top10", 0.0)),
            "vocab50_macro_top5": float(n50.get("macro_top5", 0.0)),
            "vocab50_macro_top1": float(n50.get("macro_top1", 0.0)),
            "vocab50_mrr": float(n50.get("mrr", 0.0)),
            "vocab50_median_rank": float(n50.get("median_rank", 25.0)),
            "full_vocab_macro_top10": float(fv.get("macro_top10", 0.0)),
            "full_vocab_micro_top10": float(fv.get("micro_top10", 0.0)),
            "full_vocab_macro_top1": float(fv.get("macro_top1", 0.0)),
            "full_vocab_mrr": float(fv.get("mrr", 0.0)),
            "full_vocab_median_rank": float(fv.get("median_rank", 1010.0)),
        }
        if sh_agg:
            row["shifted_vocab50_macro_top10"] = float(
                sh_agg.get("mean_vocab50_macro_top10", 0.0)
            )
            row["clean_minus_shifted_vocab50_macro_top10"] = float(
                sh_agg.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
            )
            row["shifted_full_vocab_macro_top10"] = float(
                sh_agg.get("mean_full_vocab_macro_top10", 0.0)
            )
            row["clean_minus_shifted_full_vocab_macro_top10"] = float(
                sh_agg.get("clean_minus_shifted_full_vocab_macro_top10", 0.0)
            )
        per_seed.append(row)

    vocab_sweep_agg: dict[str, dict] = {}
    for scale_key in ("N20", "N50", "N100", "N150", "full_vocab"):
        scale_rows = [
            (r.get("metrics") or {}).get("vocabulary_sweep", {}).get(scale_key)
            for r in seed_results
        ]
        scale_rows = [sr for sr in scale_rows if isinstance(sr, dict)]
        if not scale_rows:
            continue
        agg_entry: dict[str, object] = {
            "label": str(scale_rows[0].get("label", scale_key)),
            "candidate_count": int(scale_rows[0].get("candidate_count", 0)),
            "query_count": int(scale_rows[0].get("query_count", 0)),
            "seed_count": len(scale_rows),
        }
        for m_key in (
            "macro_top10",
            "micro_top10",
            "macro_top5",
            "macro_top1",
            "mrr",
            "median_rank",
        ):
            vals = np.asarray(
                [float(sr.get(m_key, 0.0)) for sr in scale_rows],
                dtype=np.float64,
            )
            agg_entry[f"mean_{m_key}"] = float(np.mean(vals))
            agg_entry[f"std_{m_key}"] = float(np.std(vals, ddof=0))
        vocab_sweep_agg[scale_key] = agg_entry

    horizons_agg: dict[str, dict] = {}
    for h_key in ("K0", "K1", "K2"):
        h_rows = [
            (r.get("metrics") or {}).get("revision_horizons", {}).get(h_key)
            for r in seed_results
        ]
        h_rows = [hr for hr in h_rows if isinstance(hr, dict)]
        if not h_rows:
            continue
        h_entry: dict[str, object] = {
            "label": str(h_rows[0].get("label", h_key)),
            "delay": int(h_rows[0].get("delay", 0)),
        }
        for hm_key in (
            "common_vocab50_macro_top10",
            "common_vocab50_micro_top10",
            "common_vocab50_macro_top1",
            "common_full_vocab_macro_top10",
            "common_full_vocab_micro_top10",
            "common_full_vocab_macro_top1",
            "common_full_vocab_mrr",
        ):
            vals = np.asarray(
                [float(hr.get(hm_key, 0.0)) for hr in h_rows], dtype=np.float64
            )
            h_entry[f"mean_{hm_key}"] = float(np.mean(vals))
            h_entry[f"std_{hm_key}"] = float(np.std(vals, ddof=0))
        horizons_agg[h_key] = h_entry

    if "K0" in horizons_agg and "K2" in horizons_agg:
        gain_entry: dict[str, float] = {}
        for hm_key in (
            "common_vocab50_macro_top10",
            "common_vocab50_micro_top10",
            "common_vocab50_macro_top1",
            "common_full_vocab_macro_top10",
        ):
            diffs = []
            for r in seed_results:
                rh = (r.get("metrics") or {}).get("revision_horizons") or {}
                if "K0" in rh and "K2" in rh:
                    diffs.append(
                        float(rh["K2"].get(hm_key, 0.0))
                        - float(rh["K0"].get(hm_key, 0.0))
                    )
            if diffs:
                arr = np.asarray(diffs, dtype=np.float64)
                gain_entry[f"mean_{hm_key}"] = float(np.mean(arr))
                gain_entry[f"std_{hm_key}"] = float(np.std(arr, ddof=0))
        if gain_entry:
            horizons_agg["K2_minus_K0"] = gain_entry

    nc_agg: dict[str, object] = {"enabled": False}
    nc_valid = [
        r.get("negative_controls")
        for r in seed_results
        if isinstance(r.get("negative_controls"), dict)
        and (r.get("negative_controls") or {}).get("shifted_aggregate")
    ]
    if nc_valid:
        nc_agg["enabled"] = True
        for m_key in (
            "vocab50_macro_top10",
            "vocab50_micro_top10",
            "vocab50_macro_top1",
            "full_vocab_macro_top10",
        ):
            clean_vals = np.asarray(
                [
                    float(
                        (nc.get("clean_reference") or {}).get(m_key, 0.0)
                    )
                    for nc in nc_valid
                ],
                dtype=np.float64,
            )
            sh_vals = np.asarray(
                [
                    float(
                        (nc.get("shifted_aggregate") or {}).get(
                            f"mean_{m_key}", 0.0
                        )
                    )
                    for nc in nc_valid
                ],
                dtype=np.float64,
            )
            gap_vals = clean_vals - sh_vals
            nc_agg[f"mean_clean_{m_key}"] = float(np.mean(clean_vals))
            nc_agg[f"std_clean_{m_key}"] = float(np.std(clean_vals, ddof=0))
            nc_agg[f"mean_shifted_{m_key}"] = float(np.mean(sh_vals))
            nc_agg[f"std_shifted_{m_key}"] = float(np.std(sh_vals, ddof=0))
            nc_agg[f"mean_gap_{m_key}"] = float(np.mean(gap_vals))
            nc_agg[f"std_gap_{m_key}"] = float(np.std(gap_vals, ddof=0))

    return {
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
        "revision_horizons_aggregate": horizons_agg,
        "negative_controls_aggregate": nc_agg,
    }


def summarize_multi_seed_evaluation(
    seed_results: Sequence[dict] | dict,
    *,
    split: str = "val",
    output_dir: str | Path | None = None,
    verbose: bool = False,
    print_report: bool = True,
    **_extra_kwargs,
) -> dict:
    """将多个随机种子的评估结果汇总为 Multi-Seed Summary JSON 与报表并落盘。"""
    if isinstance(seed_results, dict):
        seed_results = list(seed_results.values())
    if not seed_results:
        raise ValueError("seed_results 不能为空。")
    first = seed_results[0]
    is_multi_stage = (
        "stages" in first and isinstance(first.get("stages"), dict)
    )
    if is_multi_stage:
        stages_agg: dict[str, dict] = {}
        for st_key in ("word", "context", "revision"):
            st_list = [
                r["stages"][st_key]
                for r in seed_results
                if isinstance(r.get("stages"), dict) and st_key in r["stages"]
            ]
            if st_list:
                stages_agg[st_key] = _aggregate_single_stage_multi_seed(
                    st_list, split=split
                )
        target_dir = (
            Path(output_dir).resolve()
            if output_dir is not None
            else Path(str(first.get("output_dir", "."))).resolve().parent
        )
        target_dir.mkdir(parents=True, exist_ok=True)
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
            "stages_aggregate": stages_agg,
            "seed_results": list(seed_results),
            "output_dir": str(target_dir),
        }
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
        )
        return summary

    summary = _aggregate_single_stage_multi_seed(seed_results, split=split)
    target_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else Path(str(first.get("output_dir", "."))).resolve().parent
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    summary["seed_results"] = list(seed_results)
    summary["output_dir"] = str(target_dir)
    json_path = target_dir / f"multi_seed_evaluation_{split}.json"
    multi_exports = export_multi_seed_tables_and_markdown(summary, target_dir)
    summary["exports"] = multi_exports
    save_json(json_path, summary)
    summary["evaluation_json"] = str(json_path)
    print_multi_seed_evaluation_report(
        summary,
        verbose=verbose,
        output_dir=target_dir,
        print_to_stdout=print_report,
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
    all_stages: bool = False,
    plot: bool = True,
    show_progress: bool = True,
    max_eval_rows: int | None = None,
    verbose: bool = False,
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
                all_stages=all_stages,
                plot=plot,
                show_progress=show_progress,
                max_eval_rows=max_eval_rows,
                verbose=verbose,
            )
            seed_results.append(res_i)
        return summarize_multi_seed_evaluation(
            seed_results,
            split=split,
            output_dir=output_dir,
            verbose=verbose,
            print_report=True,
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
    for st2 in ("context", "revision"):
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
        w_subs = stage_results["word"].get("per_subject") or []
        w_map = {
            int(s.get("subject_id", -1)): float(s.get("n50_macro_top10", 0.0))
            for s in w_subs
            if s.get("subject_id") is not None
        }
        for st_comp, comp_key in (
            ("context", "context_vs_word_n50_top10"),
            ("revision", "revision_vs_word_n50_top10"),
        ):
            if st_comp in stage_results:
                c_subs = stage_results[st_comp].get("per_subject") or []
                c_map = {
                    int(s.get("subject_id", -1)): float(
                        s.get("n50_macro_top10", 0.0)
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
                    stage_significance[comp_key] = (
                        _compute_paired_block_significance(
                            a_arr,
                            b_arr,
                            [f"sub-{sid:02d}" for sid in common_sids],
                            seed=20261029,
                        )
                    )

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
    return multi_summary


__all__ = [
    "DEFAULT_CONTROL_SHIFTS_SECONDS",
    "_aggregate_single_stage_multi_seed",
    "_build_enriched_full_vocabulary_metrics",
    "_compute_encoded_evaluation_metrics",
    "_compute_full_vocabulary_horizon_details",
    "_compute_paired_block_significance",
    "_compute_transition_counts",
    "_evaluate_single_stage",
    "_extract_control_summary_row",
    "_format_shift_label",
    "_parse_seed_list",
    "_parse_shift_seconds_list",
    "_resolve_control_specification",
    "_score_full_vocabulary_population",
    "_significance_stars",
    "_summarize_shifted_conditions",
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
