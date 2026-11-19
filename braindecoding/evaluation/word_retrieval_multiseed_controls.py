"""多种子负对照条目（Clean / Shifted / Zero-Signal）跨种子统计聚合器。"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from braindecoding.evaluation.word_retrieval_control_selection import (
    _extract_clean_and_shifted_metric,
)
from braindecoding.evaluation.word_retrieval_statistics import (
    _format_p_value_with_stars,
    _one_sample_t_test_pvalue,
    _significance_stars,
    _theoretical_chance_for_metric,
)

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
        and (
            bool(ent.get("shifted_aggregate"))
            or control_type == "causal_common_mode_subtraction"
        )
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
