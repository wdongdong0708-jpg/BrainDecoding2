"""全词表与多尺度词级检索指标计算（Word-Macro / Event-Micro / Subject-Macro / Horizon / 词性拆解）。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from braindecoding.config import PROJECT_ROOT
from braindecoding.evaluation.retrieval_metrics import (
    fixed_vocabulary_retrieval,
    fixed_vocabulary_retrieval_metrics,
    full_vocabulary_metrics,
    mean_by_target,
    retrieval_ranks,
    strict_bank_metrics,
)
from braindecoding.evaluation.sequence_decoding_evaluation import is_content_word
from braindecoding.evaluation.word_retrieval_statistics import (
    _compute_paired_block_significance,
    _compute_transition_counts,
    _compute_two_class_linguistic_pvalue,
    _format_p_value_with_stars,
    _significance_stars,
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

    dataset = str(cfg["experiment"]["dataset"])
    language = "fr" if dataset.startswith("pallier2025") else "en"
    revision = (
        stage in ("revision", "prefix_revision")
        and "revision_predictions" in encoded
    )
    manifest_dir = PROJECT_ROOT / "data/manifests" / dataset
    if manifest_dir.is_dir() and (manifest_dir / "vocabulary_N50.json").is_file():
        manifests = load_vocabulary_assets(manifest_dir)
    else:
        from braindecoding.evaluation.vocabulary_coverage import (
            build_frequency_vocabulary,
        )

        ranked_words: list[str] | None = None
        ev_path = (cfg.get("cache") or {}).get("event_table")
        if ev_path and Path(str(ev_path)).is_file():
            try:
                from braindecoding.training import meg_word_task

                tr_tbl = meg_word_task.load_event_table(
                    ev_path,
                    split="train",
                    subjects=(cfg.get("dataset") or {}).get("subjects"),
                    trainable_only=True,
                )
                w_col = "normalized_word" if "normalized_word" in tr_tbl.columns else "标准词"
                if len(tr_tbl) > 0 and w_col in tr_tbl.columns:
                    ranked_words = tr_tbl[w_col].astype(str).tolist()
            except Exception:
                ranked_words = None
        if ranked_words:
            manifests = {
                sz: {
                    "vocabulary": list(
                        build_frequency_vocabulary(ranked_words, min(sz, len(vocabulary)))
                    )
                }
                for sz in (20, 50, 100, 150)
            }
        else:
            seen: set[str] = set()
            ordered_vocab: list[str] = []
            for w in list(vocab50) + list(vocabulary):
                sw = str(w)
                if sw not in seen:
                    seen.add(sw)
                    ordered_vocab.append(sw)
            manifests = {
                sz: {"vocabulary": ordered_vocab[: min(sz, len(ordered_vocab))]}
                for sz in (20, 50, 100, 150)
            }

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
