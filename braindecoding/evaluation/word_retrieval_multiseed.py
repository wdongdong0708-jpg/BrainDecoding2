"""多种子词级检索评估汇总（Mean ± SD、跨种子配对检验与结果 JSON 落盘）。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from braindecoding.evaluation.word_retrieval_common_horizon import (
    _resolve_eval_subdir,
)
from braindecoding.evaluation.word_retrieval_control_selection import (
    apply_multi_seed_control_selection,
)
from braindecoding.evaluation.word_retrieval_controls import (
    _summarize_shifted_conditions,
)
from braindecoding.evaluation.word_retrieval_multiseed_controls import (
    _aggregate_control_entries_across_seeds,
)
from braindecoding.evaluation.word_retrieval_statistics import (
    STAGE_DISPLAY_NAMES,
    _compute_mcnemar_statistics,
    _compute_two_class_linguistic_pvalue,
    _format_p_value_with_stars,
    _one_sample_t_test_pvalue,
    _significance_stars,
    export_multi_seed_tables_and_markdown,
    print_multi_seed_evaluation_report,
)
from braindecoding.training.training_runtime import save_json

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
    is_common_subset = (
        res.get("common_horizon_delay") is not None
        or str(res.get("evaluation_json", "")).endswith(
            f"common_subset_{split}.json"
        )
    )
    sub_infix = "_common_subset" if is_common_subset else ""
    out_d = res.get("output_dir")
    if out_d:
        candidates.append(
            Path(str(out_d))
            / "exploratory"
            / f"{filename_prefix}{sub_infix}_{split}.json"
        )
    eval_j = res.get("evaluation_json")
    if eval_j:
        candidates.append(
            Path(str(eval_j)).parent
            / "exploratory"
            / f"{filename_prefix}{sub_infix}_{split}.json"
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
                    explicit_k = str(
                        item.get("sweep_key") or item.get("method_id") or ""
                    ).strip()
                    if abs(eta_v) <= 1e-9:
                        k_str = "eta_0"
                    elif explicit_k and ":" in explicit_k:
                        k_str = explicit_k
                    else:
                        k_str = f"eta_{eta_v:g}".replace(".", "_")
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
        first_cms_rep = cms_reports_by_seed[0][1]
        cms_mode_agg = str(first_cms_rep.get("cms_mode") or "").strip().lower()
        exp_type_agg = str(first_cms_rep.get("experiment_type") or "").strip().lower()
        is_pos_cms_report = (
            cms_mode_agg == "pos_causal_ema"
            or exp_type_agg == "position_causal_ema_common_mode_subtraction"
        )
        is_causal_cms = (
            is_pos_cms_report
            or cms_mode_agg == "causal_ema"
            or exp_type_agg == "causal_ema_common_mode_subtraction"
        )
        first_sweep = _normalize_cms_sweep_dict(first_cms_rep.get("sweep"))
        cms_methods: dict[str, dict] = {}
        for eta_key, first_eta_ent in first_sweep.items():
            eta_val = float(first_eta_ent.get("eta", 0.0))
            eta_round = round(eta_val, 6)
            variant_i = str(
                first_eta_ent.get("cms_variant")
                or (
                    "pos_causal_ema"
                    if is_pos_cms_report
                    else ("causal_ema" if is_causal_cms else "leave_one_out")
                )
            ).strip().lower()
            if abs(eta_val) <= 1e-9:
                m_id = "raw"
                m_lbl = "Raw Baseline (eta=0.0)"
            elif variant_i == "pos_causal_ema":
                m_id = str(
                    first_eta_ent.get("method_id")
                    or f"pos_causal_cms:eta={eta_val:g}"
                )
                m_lbl = str(
                    first_eta_ent.get("label")
                    or f"Pos-Locked Causal EMA-CMS (eta={eta_val:g})"
                )
            elif variant_i == "causal_ema" or is_causal_cms:
                m_id = str(
                    first_eta_ent.get("method_id")
                    or f"causal_cms:eta={eta_val:g}"
                )
                m_lbl = str(
                    first_eta_ent.get("label")
                    or f"Causal EMA-CMS (eta={eta_val:g})"
                )
            else:
                m_id = f"cms:eta={eta_val:g}"
                m_lbl = f"LOO Common-Mode Subtraction (eta={eta_val:g})"
            eta_seed_entries: list[tuple[int, dict]] = []
            for sd, c_rep in cms_reports_by_seed:
                sw_i = _normalize_cms_sweep_dict(c_rep.get("sweep"))
                ent_i = sw_i.get(eta_key)
                if isinstance(ent_i, dict):
                    c_ref_i = ent_i.get("clean_reference") or {}
                    s_agg_i = ent_i.get("shifted_aggregate") or {}
                    if (
                        not is_causal_cms
                        and (
                            c_ref_i.get("full_vocab_word_macro_top10") is None
                            or s_agg_i.get("mean_full_vocab_word_macro_top10")
                            is None
                        )
                        and (sd, eta_round) in sub_full_loo_by_seed_and_eta
                    ):
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
                control_type=(
                    "position_causal_common_mode_subtraction"
                    if variant_i == "pos_causal_ema"
                    else (
                        "causal_common_mode_subtraction"
                        if (variant_i == "causal_ema" or is_causal_cms)
                        else "common_mode_subtraction"
                    )
                ),
                eta=eta_val,
                baseline_raw_agg=raw_nc_agg,
                **ctrl_common_kwargs,
            )
            if agg_eta.get("enabled"):
                cms_methods[eta_key] = agg_eta
                if abs(eta_val) > 1e-9:
                    methods_map[m_id] = agg_eta
                    alias_id = f"cms:eta={eta_val:g}"
                    if is_causal_cms and alias_id not in methods_map:
                        methods_map[alias_id] = agg_eta
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
                "cms_mode": (
                    "pos_causal_ema"
                    if is_pos_cms_report
                    else ("causal_ema" if is_causal_cms else "leave_one_out")
                ),
                "seed_count": len(cms_reports_by_seed),
                "eta_grid": list(
                    first_cms_rep.get("eta_grid")
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
    if first.get("common_horizon_delay") is not None:
        summary_out["common_horizon_delay"] = int(first["common_horizon_delay"])
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
    common_horizon_delay: int | None = None,
    eval_subdir: str | None = None,
    **_extra_kwargs,
) -> dict:
    """将多个随机种子（或单种子 n=1）的评估结果汇总为 Multi-Seed Summary JSON 与报表并落盘。"""
    if isinstance(seed_results, dict):
        seed_results = list(seed_results.values())
    if not seed_results:
        raise ValueError("seed_results 不能为空。")
    split = _infer_split_from_seed_results(seed_results, split)
    first = seed_results[0]
    eff_common_delay = (
        common_horizon_delay
        if common_horizon_delay is not None
        else first.get("common_horizon_delay")
    )
    has_common_subset = eff_common_delay is not None
    promote_primary = has_common_subset and (
        bool(_resolve_eval_subdir(eval_subdir))
        or all(
            isinstance(r, Mapping)
            and bool(r.get("promoted_common_subset_to_primary"))
            for r in seed_results
        )
    )
    subset_tag_suffix = "_common_subset" if has_common_subset else ""
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
        if has_common_subset:
            summary["common_horizon_delay"] = int(eff_common_delay)
        if promote_primary:
            summary["promoted_common_subset_to_primary"] = True
        if save_files:
            target_dir.mkdir(parents=True, exist_ok=True)
            json_path = (
                target_dir
                / f"multi_seed_stage_comparison{subset_tag_suffix}_{split}.json"
            )
            export_dir = (
                target_dir / "common_subset"
                if has_common_subset
                else target_dir
            )
            export_dir.mkdir(parents=True, exist_ok=True)
            multi_exports = export_multi_seed_tables_and_markdown(
                summary, export_dir
            )
            if promote_primary:
                primary_exports = export_multi_seed_tables_and_markdown(
                    summary, target_dir
                )
                primary_json_path = (
                    target_dir / f"multi_seed_stage_comparison_{split}.json"
                )
                summary["exports"] = primary_exports
                summary["evaluation_json"] = str(primary_json_path)
                save_json(json_path, summary)
                save_json(primary_json_path, summary)
            else:
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
    if has_common_subset:
        summary["common_horizon_delay"] = int(eff_common_delay)
    if promote_primary:
        summary["promoted_common_subset_to_primary"] = True
    target_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else Path(str(first.get("output_dir", "."))).resolve().parent
    )
    summary["seed_results"] = list(seed_results)
    summary["output_dir"] = str(target_dir)
    if save_files:
        target_dir.mkdir(parents=True, exist_ok=True)
        json_path = (
            target_dir
            / f"multi_seed_evaluation{subset_tag_suffix}_{split}.json"
        )
        export_dir = (
            target_dir / "common_subset" if has_common_subset else target_dir
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        multi_exports = export_multi_seed_tables_and_markdown(
            summary, export_dir
        )
        if promote_primary:
            primary_exports = export_multi_seed_tables_and_markdown(
                summary, target_dir
            )
            primary_json_path = (
                target_dir / f"multi_seed_evaluation_{split}.json"
            )
            summary["exports"] = primary_exports
            summary["evaluation_json"] = str(primary_json_path)
            save_json(json_path, summary)
            save_json(primary_json_path, summary)
        else:
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
