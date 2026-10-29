"""Unified terminal/Colab executive summary and full breakdown report rendering.

Provides compact (<= 86 columns) single-screen Executive Summaries by default
for terminal and Google Colab readability, full multi-table breakdowns when
``verbose=True`` (and automatically saved to ``.txt`` report files), and
zero-second report rendering from saved JSON files or run directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from braindecoding.evaluation.evaluation_exports import (
    export_common_mode_subtraction_tables_and_markdown,
    export_multi_seed_tables_and_markdown,
    export_sequence_decoding_tables_and_markdown,
    export_stage_comparison_tables_and_markdown,
    export_subspace_coherence_debiasing_tables_and_markdown,
    export_word_evaluation_tables_and_markdown,
)
from braindecoding.evaluation.evaluation_figures import (
    STAGE_DISPLAY_NAMES,
    STAGE_SHORT_LABELS,
)

REPORT_WIDTH = 86


def _fmt_pct(val: object, width: int = 7) -> str:
    if val is None or val == "":
        return "-".rjust(width)
    try:
        return f"{float(val):{width}.2f}"
    except (TypeError, ValueError):
        return "-".rjust(width)


def _fmt_signed_pct(val: object, width: int = 7) -> str:
    if val is None or val == "":
        return "-".rjust(width)
    try:
        return f"{float(val):+{width}.2f}"
    except (TypeError, ValueError):
        return "-".rjust(width)


def _fmt_ratio_pct(val: object, width: int = 7) -> str:
    """Format a 0..1 ratio as a percentage string (e.g. ' 44.50%')."""
    if val is None or val == "":
        return "-".rjust(width)
    try:
        return f"{float(val) * 100.0:{width - 1}.2f}%"
    except (TypeError, ValueError):
        return "-".rjust(width)


def _fmt_signed_ratio_pct(val: object, width: int = 7) -> str:
    """Format a 0..1 signed delta as percentage points (e.g. ' +5.12%')."""
    if val is None or val == "":
        return "-".rjust(width)
    try:
        return f"{float(val) * 100.0:+{width - 1}.2f}%"
    except (TypeError, ValueError):
        return "-".rjust(width)


def _fmt_float(val: object, width: int = 7, decimals: int = 2) -> str:
    if val is None or val == "":
        return "-".rjust(width)
    try:
        return f"{float(val):{width}.{decimals}f}"
    except (TypeError, ValueError):
        return "-".rjust(width)


def _short_method_label(item: Mapping[str, Any]) -> str:
    """Produce a compact <= 26 char label for a Subspace / Full-Rank LOO entry."""
    fam = str(item.get("method_family", ""))
    eta = item.get("eta")
    rank_k = item.get("rank_k")
    gate = item.get("use_coherence_gate")
    if fam == "raw" or item.get("method_id") == "raw_baseline":
        return "Raw Baseline (eta=0)"
    if fam == "full_rank_loo":
        return f"Full-Rank LOO (eta={float(eta or 0):g})"
    if rank_k is not None:
        g_str = "On" if gate else "Off"
        return f"Subspace K={rank_k} η={float(eta or 0):g} G={g_str}"
    lbl = str(item.get("label", item.get("method_id", "-")))
    return lbl[:26]


# ============================================================================
# 1. Word-Level Single-Stage Evaluation Report
# ============================================================================


def _extract_sweep_and_horizons(
    eval_result: Mapping[str, Any],
) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    """Normalize workflow `eval_result['metrics']`, flat metrics, and canonical `vocabularies` schemas."""
    metrics = eval_result.get("metrics")
    if isinstance(metrics, Mapping) and "vocabulary_sweep" in metrics:
        sweep = {
            k: dict(v) if isinstance(v, Mapping) else v
            for k, v in (metrics.get("vocabulary_sweep") or {}).items()
        }
        if isinstance(sweep.get("full_vocab"), dict):
            fv_b = sweep["full_vocab"]
            if fv_b.get("word_macro_top10") is not None:
                fv_b["macro_top10"] = float(fv_b["word_macro_top10"])
        horizons = {
            hk: dict(hv) if isinstance(hv, Mapping) else hv
            for hk, hv in (metrics.get("revision_horizons") or {}).items()
        }
        for hk, hv in horizons.items():
            if isinstance(hv, dict) and hv.get("common_full_vocab_word_macro_top10") is not None:
                hv["common_full_vocab_macro_top10"] = float(hv["common_full_vocab_word_macro_top10"])
        per_subj = list(metrics.get("per_subject") or [])
        return sweep, horizons, per_subj

    sweep = {}
    horizons = {}
    per_subj: list[dict] = []
    if isinstance(metrics, Mapping):
        horizons = dict(metrics.get("revision_horizons") or {})
        if isinstance(metrics.get("per_subject"), list):
            per_subj = list(metrics["per_subject"])
        elif isinstance(metrics.get("per_subject_vocab50"), Mapping):
            for sid, sdata in metrics["per_subject_vocab50"].items():
                if isinstance(sdata, Mapping):
                    per_subj.append(
                        {
                            "subject_id": str(sid),
                            "total_events": int(sdata.get("query_count", 0)),
                            "vocab50_queries": int(sdata.get("query_count", 0)),
                            "macro_top10": float(sdata.get("macro_top10", 0.0)),
                            "micro_top10": float(sdata.get("micro_top10", 0.0)),
                            "macro_top1": float(sdata.get("macro_top1", 0.0)),
                            "median_rank": float(sdata.get("median_rank_macro", 25.0)),
                        }
                    )
        if "vocab_50_macro_top10" in metrics:
            sweep["N50"] = {
                "label": "50-Word (N50)",
                "vocabulary_size": 50,
                "query_count": int(metrics.get("evaluated_events", 0)),
                "macro_top10": float(metrics.get("vocab_50_macro_top10", 0.0)),
                "micro_top10": float(metrics.get("vocab_50_micro_top10", 0.0)),
                "macro_top1": float(metrics.get("vocab_50_macro_top1", 0.0)),
                "mrr": float(metrics.get("vocab_50_mrr_macro", 0.0)),
                "median_rank": float(metrics.get("vocab_50_median_rank_macro", 25.0)),
                "chance_top10": 0.2,
            }
        if "full_vocab_macro_top10" in metrics:
            fv_sz = int(metrics.get("full_vocab_size", 2020))
            sweep["full_vocab"] = {
                "label": f"Full Vocab ({fv_sz})",
                "vocabulary_size": fv_sz,
                "query_count": int(metrics.get("evaluated_events", 0)),
                "macro_top10": float(metrics.get("full_vocab_macro_top10", 0.0)),
                "micro_top10": float(metrics.get("full_vocab_micro_top10", 0.0)),
                "macro_top1": float(metrics.get("full_vocab_macro_top1", 0.0)),
                "mrr": float(metrics.get("full_vocab_mrr_macro", 0.0)),
                "median_rank": float(metrics.get("full_vocab_median_rank_macro", fv_sz / 2.0)),
                "chance_top10": min(10.0, float(fv_sz)) / max(1.0, float(fv_sz)),
            }

    vocabs = eval_result.get("vocabularies") or {}
    if isinstance(vocabs, Mapping):
        for k in ("N20", "N50", "N100", "N150"):
            v = vocabs.get(k)
            if isinstance(v, Mapping):
                ovr = v.get("overall") or {}
                v_sz = int(v.get("vocabulary_size", int(k.lstrip("N"))))
                sweep[k] = {
                    "label": f"{v_sz}-Word ({k})",
                    "vocabulary_size": v_sz,
                    "query_count": int(v.get("in_vocabulary_query_count", 0)),
                    "macro_top10": float(ovr.get("macro_top10_accuracy", 0.0)),
                    "micro_top10": float(ovr.get("micro_top10_accuracy", 0.0)),
                    "macro_top1": float(ovr.get("macro_top1_accuracy", 0.0)),
                    "mrr": float(ovr.get("macro_mrr", 0.0)),
                    "median_rank": float(ovr.get("macro_median_rank", v_sz / 2.0)),
                    "chance_top10": min(10.0, float(v_sz)) / max(1.0, float(v_sz)),
                }
    fv = eval_result.get("full_vocabulary_evaluation")
    if isinstance(fv, Mapping) and isinstance(fv.get("overall"), Mapping):
        ovr = fv["overall"]
        v_sz = int(fv.get("vocabulary_size", 2020))
        sweep["full_vocab"] = {
            "label": f"Full Vocab ({v_sz})",
            "vocabulary_size": v_sz,
            "query_count": int(fv.get("in_vocabulary_query_count", eval_result.get("query_count", 0))),
            "macro_top10": float(ovr.get("macro_top10_accuracy", 0.0)),
            "micro_top10": float(ovr.get("micro_top10_accuracy", 0.0)),
            "macro_top1": float(ovr.get("macro_top1_accuracy", 0.0)),
            "mrr": float(ovr.get("macro_mrr", 0.0)),
            "median_rank": float(ovr.get("macro_median_rank", v_sz / 2.0)),
            "chance_top10": min(10.0, float(v_sz)) / max(1.0, float(v_sz)),
        }
    return sweep, horizons, per_subj


def format_word_evaluation_report(
    eval_result: Mapping[str, Any],
    *,
    verbose: bool = False,
) -> str:
    """Format a single-stage word-level evaluation result dictionary.

    Default (`verbose=False`) produces a single-screen <= 86-column Executive Summary
    highlighting N50 (primary), Full-Vocab, K0/K1/K2 horizons, and Negative Control
    Clean vs. Shifted Mean. `verbose=True` appends N20/N50/N100/N150, per-shift,
    and per-subject breakdowns.
    """
    dataset = str(eval_result.get("dataset", "-"))
    stage = str(eval_result.get("stage", "word"))
    stage_label = STAGE_DISPLAY_NAMES.get(stage, stage)
    split = str(eval_result.get("split", "val")).upper()
    seed = eval_result.get("seed", 0)
    subset = eval_result.get("subset") or eval_result.get("subject_subset") or {}
    subset_tag = (
        subset.get("subset_tag", subset.get("tag", "all"))
        if isinstance(subset, Mapping)
        else str(subset)
    )
    ckpt = eval_result.get("checkpoint") or {}
    ckpt_epoch = (
        ckpt.get("epoch", eval_result.get("checkpoint_epoch", eval_result.get("evaluated_epoch", "-")))
        if isinstance(ckpt, Mapping)
        else eval_result.get("evaluated_epoch", eval_result.get("checkpoint_epoch", "-"))
    )
    win_sec = float(eval_result.get("window_seconds", 1.0))
    metrics = eval_result.get("metrics") or {}
    eval_events = (
        metrics.get("evaluated_events", eval_result.get("query_count", 0))
        if isinstance(metrics, Mapping)
        else eval_result.get("query_count", 0)
    )

    sweep, horizons, per_subject = _extract_sweep_and_horizons(eval_result)

    banner = "=" * REPORT_WIDTH
    rule = "-" * REPORT_WIDTH
    lines: list[str] = [
        banner,
        f"Word Evaluation Summary | {dataset} | {stage_label}"[:REPORT_WIDTH],
        f"Split: {split} | Subset: {subset_tag} | Seed: {seed} | Win: {win_sec:.2f}s | Ep: {ckpt_epoch} | N={eval_events}"[:REPORT_WIDTH],
        banner,
        "[1] Core Retrieval Performance (50-Word Primary Bank & Full Training Vocab)",
    ]
    hdr = (
        f"{'Vocabulary Scale':<22} | {'Queries':>7} | {'MacTop10':>8} | "
        f"{'MicTop10':>8} | {'MacTop1':>8} | {'MRR(%)':>7} | {'MedRank':>7}"
    )
    lines.append(rule)
    lines.append(hdr)
    lines.append(rule)

    primary_keys = (
        ("N20", "N50", "N50_strict", "N100", "N150", "full_vocab")
        if verbose
        else ("N50", "N50_strict", "full_vocab")
    )
    for k in primary_keys:
        if k not in sweep:
            continue
        r = sweep[k]
        lbl = str(r.get("label", k))[:22]
        lines.append(
            f"{lbl:<22} | {int(r.get('query_count', 0)):>7d} | "
            f"{_fmt_ratio_pct(r.get('macro_top10'), 8)} | "
            f"{_fmt_ratio_pct(r.get('micro_top10'), 8)} | "
            f"{_fmt_ratio_pct(r.get('macro_top1'), 8)} | "
            f"{_fmt_ratio_pct(r.get('mrr'), 7)} | "
            f"{_fmt_float(r.get('median_rank'), 7, 1)}"
        )
    lines.append(rule)

    n50_blk = sweep.get("N50") or {}
    fv_blk = sweep.get("full_vocab") or {}
    subj_sum = (
        (metrics.get("per_subject_summary") if isinstance(metrics, Mapping) else None)
        or eval_result.get("per_subject_summary")
        or {}
    )
    if n50_blk or fv_blk:
        n50_ch = float(n50_blk.get("chance_top10", 0.2))
        n50_fd = float(
            n50_blk.get(
                "fold_over_chance_top10",
                float(n50_blk.get("macro_top10", 0.0)) / max(n50_ch, 1e-12),
            )
        )
        fv_ch = float(fv_blk.get("chance_top10", 0.005))
        fv_fd = float(
            fv_blk.get(
                "fold_over_chance_top10",
                float(fv_blk.get("macro_top10", 0.0)) / max(fv_ch, 1e-12),
            )
        )
        subj_str = ""
        sm_raw = subj_sum.get("mean_n50_macro_top10", subj_sum.get("mean_macro_top10")) if isinstance(subj_sum, Mapping) else None
        if sm_raw is not None:
            sm = float(sm_raw) * 100.0
            ssd = float(subj_sum.get("std_n50_macro_top10", subj_sum.get("std_macro_top10", 0.0))) * 100.0
            subj_str = f" | Subj: {sm:.1f}±{ssd:.1f}%"
        lines.append(
            f" * Chance@10: N50={n50_ch*100:.1f}% ({n50_fd:.2f}x) | "
            f"Full={fv_ch*100:.2f}% ({fv_fd:.1f}x){subj_str}"[:REPORT_WIDTH]
        )

    ling = (
        (metrics.get("linguistic_breakdown") if isinstance(metrics, Mapping) else None)
        or eval_result.get("linguistic_breakdown")
        or {}
    )
    if isinstance(ling, Mapping) and ling:
        cw_b = ling.get("content_words") or {}
        fw_b = ling.get("function_words") or {}
        c50 = cw_b.get("vocab50_word_macro_top10")
        f50 = fw_b.get("vocab50_word_macro_top10")
        cfv = cw_b.get("full_vocab_word_macro_top10")
        ffv = fw_b.get("full_vocab_word_macro_top10")
        if c50 is not None and f50 is not None:
            fv_part = (
                f" | Full: {float(cfv)*100:.2f}% / {float(ffv)*100:.2f}%"
                if cfv is not None and ffv is not None
                else ""
            )
            lines.append(
                f" * Content/Func Top10: N50={float(c50)*100:.2f}% / {float(f50)*100:.2f}%{fv_part}"[
                    :REPORT_WIDTH
                ]
            )
        if verbose and (cw_b or fw_b):
            lines.append("\n[1B] Linguistic Word-Class Breakdown (Content vs. Function Words)")
            l_hdr = (
                f"{'Word Class':<24} | {'Events':>7} | {'Types':>6} | "
                f"{'50wTop10':>9} | {'50wTop1':>8} | {'FulTop10':>9} | {'FulMRR':>7}"
            )
            lines.append(rule)
            lines.append(l_hdr)
            lines.append(rule)
            for cls_k, cls_lbl in (
                ("content_words", "Content Words (实词)"),
                ("function_words", "Function Words (虚词)"),
            ):
                cb = ling.get(cls_k)
                if not isinstance(cb, Mapping):
                    continue
                lines.append(
                    f"{cls_lbl[:24]:<24} | {int(cb.get('query_count', 0)):>7d} | "
                    f"{int(cb.get('unique_word_count', 0)):>6d} | "
                    f"{_fmt_ratio_pct(cb.get('vocab50_word_macro_top10'), 9)} | "
                    f"{_fmt_ratio_pct(cb.get('vocab50_word_macro_top1'), 8)} | "
                    f"{_fmt_ratio_pct(cb.get('full_vocab_word_macro_top10'), 9)} | "
                    f"{_fmt_ratio_pct(cb.get('full_vocab_mrr'), 7)}"
                )
            lines.append(rule)

    sig_map = (
        (metrics.get("statistical_significance") if isinstance(metrics, Mapping) else None)
        or eval_result.get("statistical_significance")
        or {}
    )
    transitions = (
        (metrics.get("revision_transitions") if isinstance(metrics, Mapping) else None)
        or eval_result.get("revision_transitions")
        or {}
    )

    if horizons:
        lines.append("\n[2] Rolling Revision Horizons (Common Targets: K0 -> K1 -> K2)")
        h_hdr = (
            f"{'Horizon':<23} | {'50w MacTop10':>12} | {'50w MicTop10':>12} | "
            f"{'50w MacTop1':>11} | {'FullVoc Top10':>13}"
        )
        lines.append(rule)
        lines.append(h_hdr)
        lines.append(rule)
        for h_key in ("K0", "K1", "K2"):
            if h_key not in horizons:
                continue
            h = horizons[h_key]
            lbl = str(h.get("label", h_key))[:23]
            lines.append(
                f"{lbl:<23} | "
                f"{_fmt_ratio_pct(h.get('common_vocab50_macro_top10'), 12)} | "
                f"{_fmt_ratio_pct(h.get('common_vocab50_micro_top10'), 12)} | "
                f"{_fmt_ratio_pct(h.get('common_vocab50_macro_top1'), 11)} | "
                f"{_fmt_ratio_pct(h.get('common_full_vocab_macro_top10'), 13)}"
            )
        if "K0" in horizons and "K2" in horizons:
            k0, k2 = horizons["K0"], horizons["K2"]
            d_m10 = float(k2.get("common_vocab50_macro_top10", 0.0)) - float(k0.get("common_vocab50_macro_top10", 0.0))
            d_u10 = float(k2.get("common_vocab50_micro_top10", 0.0)) - float(k0.get("common_vocab50_micro_top10", 0.0))
            d_m1 = float(k2.get("common_vocab50_macro_top1", 0.0)) - float(k0.get("common_vocab50_macro_top1", 0.0))
            d_fv = float(k2.get("common_full_vocab_macro_top10", 0.0)) - float(k0.get("common_full_vocab_macro_top10", 0.0))
            lines.append(
                f"{'Revision Gain (K2 - K0)':<23} | "
                f"{_fmt_signed_ratio_pct(d_m10, 12)} | "
                f"{_fmt_signed_ratio_pct(d_u10, 12)} | "
                f"{_fmt_signed_ratio_pct(d_m1, 11)} | "
                f"{_fmt_signed_ratio_pct(d_fv, 13)}"
            )
        lines.append(rule)
        sig_k2 = sig_map.get("K2_minus_K0_vocab50_top10") if isinstance(sig_map, Mapping) else None
        if isinstance(sig_k2, Mapping) and sig_k2.get("ci_lower") is not None:
            ci_l = float(sig_k2["ci_lower"]) * 100.0
            ci_u = float(sig_k2["ci_upper"]) * 100.0
            p_v = float(sig_k2.get("p_two_sided", sig_k2.get("p_one_sided", 1.0)))
            st_s = str(sig_k2.get("significance", ""))
            lines.append(
                f" * K2-K0 50w Top10 95% CI: [{ci_l:+.2f}%, {ci_u:+.2f}%] (p={p_v:.4f} {st_s})"[
                    :REPORT_WIDTH
                ]
            )
        t_k02 = (
            (transitions.get("K0_to_K2") or {}).get("vocab50_top10")
            if isinstance(transitions, Mapping)
            else None
        )
        if isinstance(t_k02, Mapping):
            rep = int(t_k02.get("beneficial_repairs", 0))
            hrm = int(t_k02.get("harmful_overwrites", 0))
            rat = float(t_k02.get("repair_to_harm_ratio", 0.0))
            lines.append(
                f" * K0->K2 Transitions: Repairs={rep} (Miss->Hit) | Harms={hrm} (Hit->Miss) | {rat:.2f}x"[
                    :REPORT_WIDTH
                ]
            )
        if verbose and isinstance(transitions, Mapping) and transitions:
            lines.append("\n[2B] Event-Level Revision Transition Matrix (Miss->Hit vs. Hit->Miss)")
            t_hdr = (
                f"{'Transition / Criterion':<28} | {'Events':>7} | {'Repairs':>9} | "
                f"{'Harms':>9} | {'NetGain':>9} | {'Rep/Harm':>8}"
            )
            lines.append(rule)
            lines.append(t_hdr)
            lines.append(rule)
            for tr_k in ("K0_to_K1", "K0_to_K2", "K1_to_K2"):
                tr_b = transitions.get(tr_k)
                if not isinstance(tr_b, Mapping):
                    continue
                for crit_k, crit_lbl in (
                    ("vocab50_top10", f"{tr_k} (50w Top-10)"),
                    ("full_vocab_top10", f"{tr_k} (Full Top-10)"),
                ):
                    cd = tr_b.get(crit_k)
                    if not isinstance(cd, Mapping):
                        continue
                    lines.append(
                        f"{crit_lbl[:28]:<28} | {int(cd.get('total_events', 0)):>7d} | "
                        f"{int(cd.get('beneficial_repairs', 0)):>9d} | "
                        f"{int(cd.get('harmful_overwrites', 0)):>9d} | "
                        f"{int(cd.get('net_gain_events', 0)):>+9d} | "
                        f"{float(cd.get('repair_to_harm_ratio', 0.0)):>7.2f}x"
                    )
            lines.append(rule)

    neg_ctrl = eval_result.get("negative_controls") or eval_result.get("negative_control") or {}
    raw_ctrl_conditions = neg_ctrl.get("conditions") or []
    if isinstance(raw_ctrl_conditions, Mapping):
        ctrl_conditions = [
            {"key": k, **(v if isinstance(v, Mapping) else {})}
            for k, v in raw_ctrl_conditions.items()
        ]
    else:
        ctrl_conditions = list(raw_ctrl_conditions)
    if ctrl_conditions:
        clean_row = neg_ctrl.get("clean_reference") or {}
        shifted_agg = neg_ctrl.get("shifted_aggregate") or {}
        shift_cnt = int(shifted_agg.get("shift_count", shifted_agg.get("num_shifts", len(ctrl_conditions))))
        lines.append(f"\n[3] Negative Control Summary ({split})")
        c_hdr = (
            f"{'Condition':<20} | {'50w MacTop10':>12} | {'vs Clean':>9} | "
            f"{'50w MacTop1':>11} | {'FullTop10':>10} | {'MedRank':>7}"
        )
        lines.append(rule)
        lines.append(c_hdr)
        lines.append(rule)
        c_fv_m10 = clean_row.get("full_vocab_word_macro_top10", clean_row.get("full_vocab_macro_top10"))
        lines.append(
            f"{str(clean_row.get('label', 'Clean (0s)'))[:20]:<20} | "
            f"{_fmt_ratio_pct(clean_row.get('vocab50_macro_top10'), 12)} | "
            f"{'(Ref)':>9} | "
            f"{_fmt_ratio_pct(clean_row.get('vocab50_macro_top1'), 11)} | "
            f"{_fmt_ratio_pct(c_fv_m10, 10)} | "
            f"{_fmt_float(clean_row.get('vocab50_median_rank', clean_row.get('vocab50_median_rank_macro')), 7, 1)}"
        )
        if verbose or shift_cnt <= 1:
            for cond in ctrl_conditions:
                s_row = cond.get("summary") or {}
                d_50 = (s_row.get("delta_vs_clean") or {}).get("vocab50_macro_top10")
                if d_50 is None and clean_row.get("vocab50_macro_top10") is not None and s_row.get("vocab50_macro_top10") is not None:
                    d_50 = float(s_row["vocab50_macro_top10"]) - float(clean_row["vocab50_macro_top10"])
                s_fv_m10 = s_row.get("full_vocab_word_macro_top10", s_row.get("full_vocab_macro_top10"))
                lines.append(
                    f"{str(s_row.get('label', cond.get('key', '-')))[:20]:<20} | "
                    f"{_fmt_ratio_pct(s_row.get('vocab50_macro_top10'), 12)} | "
                    f"{_fmt_signed_ratio_pct(d_50, 9)} | "
                    f"{_fmt_ratio_pct(s_row.get('vocab50_macro_top1'), 11)} | "
                    f"{_fmt_ratio_pct(s_fv_m10, 10)} | "
                    f"{_fmt_float(s_row.get('vocab50_median_rank', s_row.get('vocab50_median_rank_macro')), 7, 1)}"
                )
        if shift_cnt > 1:
            m50 = float(shifted_agg.get("mean_vocab50_macro_top10", 0.0)) * 100.0
            sd50 = float(shifted_agg.get("std_vocab50_macro_top10", 0.0)) * 100.0
            d50 = shifted_agg.get("delta_vocab50_macro_top10_vs_clean")
            gap50 = shifted_agg.get("clean_minus_shifted_vocab50_macro_top10")
            s_agg_fv = shifted_agg.get("mean_full_vocab_word_macro_top10", shifted_agg.get("mean_full_vocab_macro_top10"))
            gap_fv = shifted_agg.get(
                "clean_minus_shifted_full_vocab_word_macro_top10",
                shifted_agg.get("clean_minus_shifted_full_vocab_macro_top10"),
            )
            lines.append(
                f"{'Shifted Mean ± SD':<20} | "
                f"{f'{m50:.2f}±{sd50:.2f}%':>12} | "
                f"{_fmt_signed_ratio_pct(d50, 9)} | "
                f"{_fmt_ratio_pct(shifted_agg.get('mean_vocab50_macro_top1'), 11)} | "
                f"{_fmt_ratio_pct(s_agg_fv, 10)} | "
                f"{_fmt_float(shifted_agg.get('mean_vocab50_median_rank', shifted_agg.get('mean_vocab50_median_rank_macro')), 7, 1)}"
            )
            lines.append(
                f"{'Net Gap (Clean-Shft)':<20} | "
                f"{_fmt_signed_ratio_pct(gap50, 12)} | "
                f"{'-':>9} | "
                f"{_fmt_signed_ratio_pct(shifted_agg.get('clean_minus_shifted_vocab50_macro_top1'), 11)} | "
                f"{_fmt_signed_ratio_pct(gap_fv, 10)} | "
                f"{'-':>7}"
            )
        lines.append(rule)
        nc_sig = (
            (neg_ctrl.get("statistical_significance") or {}).get(
                "clean_vs_shifted_vocab50_top10"
            )
            if isinstance(neg_ctrl.get("statistical_significance"), Mapping)
            else None
        )
        if isinstance(nc_sig, Mapping) and nc_sig.get("ci_lower") is not None:
            ci_l = float(nc_sig["ci_lower"]) * 100.0
            ci_u = float(nc_sig["ci_upper"]) * 100.0
            p_v = float(nc_sig.get("p_two_sided", nc_sig.get("p_one_sided", 1.0)))
            st_s = str(nc_sig.get("significance", ""))
            lines.append(
                f" * Clean-Shift 50w Top10 95% CI: [{ci_l:+.2f}%, {ci_u:+.2f}%] (p={p_v:.4f} {st_s})"[
                    :REPORT_WIDTH
                ]
            )

    if verbose and len(per_subject) >= 1:
        lines.append(f"\n[4] Per-Subject 50-Word Retrieval Breakdown ({split})")
        s_hdr = (
            f"{'Subject':<12} | {'Events':>7} | {'50w Qry':>7} | "
            f"{'MacTop10':>9} | {'MicTop10':>9} | {'MacTop1':>9} | {'MedRank':>8}"
        )
        lines.append(rule)
        lines.append(s_hdr)
        lines.append(rule)
        for s in per_subject:
            lines.append(
                f"{str(s.get('subject_label', s.get('subject_id', '-'))):<12} | "
                f"{int(s.get('total_events', s.get('query_count', 0))):>7d} | "
                f"{int(s.get('vocab50_queries', s.get('n50_query_count', 0))):>7d} | "
                f"{_fmt_ratio_pct(s.get('macro_top10', s.get('n50_macro_top10')), 9)} | "
                f"{_fmt_ratio_pct(s.get('micro_top10', s.get('n50_micro_top10')), 9)} | "
                f"{_fmt_ratio_pct(s.get('macro_top1', s.get('n50_macro_top1')), 9)} | "
                f"{_fmt_float(s.get('median_rank', s.get('n50_median_rank')), 8, 1)}"
            )
        lines.append(rule)
    elif not verbose:
        lines.append(
            f"[Tip] Full tables: evaluation_report_{split.lower()}.md, .txt & tables/*.csv (--verbose)"[
                :REPORT_WIDTH
            ]
        )

    lines.append(banner)
    return "\n".join(lines)


def print_word_evaluation_report(
    eval_result: Mapping[str, Any],
    *,
    verbose: bool = False,
    output_dir: Path | str | None = None,
    print_to_stdout: bool = True,
) -> str:
    """Save full .txt/.md/CSV reports to output_dir and print compact/verbose report to terminal."""
    split = str(eval_result.get("split", "val")).lower()
    target_dir = output_dir or eval_result.get("output_dir")
    if target_dir:
        out_dir_path = Path(str(target_dir))
        out_path = out_dir_path / f"evaluation_report_{split}.txt"
        try:
            out_dir_path.mkdir(parents=True, exist_ok=True)
            full_txt = format_word_evaluation_report(eval_result, verbose=True)
            out_path.write_text(full_txt + "\n", encoding="utf-8")
            export_word_evaluation_tables_and_markdown(eval_result, out_dir_path)
        except OSError:
            pass
    rendered = format_word_evaluation_report(eval_result, verbose=verbose)
    if print_to_stdout:
        print("\n" + rendered, flush=True)
    return rendered


print_evaluation_report = print_word_evaluation_report


# ============================================================================
# 2. Multi-Stage Comparison Report (--all-stages)
# ============================================================================


def format_stage_comparison_report(
    stage_results_or_summary: Mapping[str, Any],
    *,
    split: str = "val",
    verbose: bool = False,
) -> str:
    """Format a unified single-screen comparison across Stage 1, 2.1, and 2.2."""
    if "stages" in stage_results_or_summary and isinstance(
        stage_results_or_summary.get("stages"), Mapping
    ):
        stages_map: Mapping[str, Any] = stage_results_or_summary["stages"]
        split = str(stage_results_or_summary.get("split", split))
    else:
        stages_map = stage_results_or_summary

    first_res = next(
        (v for v in stages_map.values() if isinstance(v, Mapping)), {}
    )
    dataset = str(first_res.get("dataset", "-"))
    seed = first_res.get("seed", 0)
    subset = first_res.get("subset") or first_res.get("subject_subset") or {}
    subset_tag = (
        subset.get("subset_tag", subset.get("tag", "all"))
        if isinstance(subset, Mapping)
        else str(subset)
    )

    banner = "=" * REPORT_WIDTH
    rule = "-" * REPORT_WIDTH
    lines: list[str] = [
        banner,
        f"Multi-Stage Summary | {dataset} ({split.upper()}) | Subset: {subset_tag} | Seed: {seed}"[:REPORT_WIDTH],
        banner,
        "[1] Cross-Stage Retrieval Progression (50-Word Primary Bank & Full Training Vocab)",
    ]
    hdr = (
        f"{'Stage / Horizon':<26} | {'50w Top10':>9} | {'50w Top1':>8} | "
        f"{'FullTop10':>9} | {'FullTop1':>8} | {'Full MRR':>8}"
    )
    lines.append(rule)
    lines.append(hdr)
    lines.append(rule)

    canonical_order = (
        "word",
        "word_encoder",
        "context",
        "context_transformer",
        "revision",
        "rolling_revision",
        "prefix_revision",
        "full_history_revision",
        "full_prefix_revision",
    )
    ordered_keys = [k for k in canonical_order if k in stages_map] + [
        k for k in stages_map if k not in canonical_order
    ]

    s1_n50_t10: float | None = None
    s1_full_t10: float | None = None
    last_n50_t10: float | None = None
    last_full_t10: float | None = None

    for st_key in ordered_keys:
        res = stages_map.get(st_key)
        if not isinstance(res, Mapping):
            continue
        sweep, horizons, _ = _extract_sweep_and_horizons(res)
        short_lbl = STAGE_SHORT_LABELS.get(st_key, st_key).replace("\n", " ")[:26]
        n50 = sweep.get("N50") or {}
        fv = sweep.get("full_vocab") or {}

        n50_t10 = n50.get("macro_top10")
        fv_t10 = fv.get("macro_top10")
        if st_key in ("word", "word_encoder") and s1_n50_t10 is None:
            s1_n50_t10 = float(n50_t10) if n50_t10 is not None else None
            s1_full_t10 = float(fv_t10) if fv_t10 is not None else None
        last_n50_t10 = float(n50_t10) if n50_t10 is not None else last_n50_t10
        last_full_t10 = float(fv_t10) if fv_t10 is not None else last_full_t10

        lines.append(
            f"{short_lbl:<26} | "
            f"{_fmt_ratio_pct(n50_t10, 9)} | "
            f"{_fmt_ratio_pct(n50.get('macro_top1'), 8)} | "
            f"{_fmt_ratio_pct(fv_t10, 9)} | "
            f"{_fmt_ratio_pct(fv.get('macro_top1'), 8)} | "
            f"{_fmt_ratio_pct(fv.get('mrr'), 8)}"
        )

        if horizons:
            for h_key, h_lbl in (
                ("K0", "  ├─ K0 (0.0s Onset)"),
                ("K1", "  ├─ K1 (+1.0s Delay)"),
                ("K2", "  └─ K2 (+2.0s Delay)"),
            ):
                if h_key not in horizons:
                    continue
                hb = horizons[h_key]
                lines.append(
                    f"{h_lbl:<26} | "
                    f"{_fmt_ratio_pct(hb.get('common_vocab50_macro_top10'), 9)} | "
                    f"{_fmt_ratio_pct(hb.get('common_vocab50_macro_top1'), 8)} | "
                    f"{_fmt_ratio_pct(hb.get('common_full_vocab_macro_top10'), 9)} | "
                    f"{_fmt_ratio_pct(hb.get('common_full_vocab_macro_top1'), 8)} | "
                    f"{_fmt_ratio_pct(hb.get('common_full_vocab_mrr'), 8)}"
                )
            if "K0" in horizons and "K2" in horizons:
                k0, k2 = horizons["K0"], horizons["K2"]
                g50_10 = float(k2.get("common_vocab50_macro_top10", 0.0)) - float(k0.get("common_vocab50_macro_top10", 0.0))
                g50_1 = float(k2.get("common_vocab50_macro_top1", 0.0)) - float(k0.get("common_vocab50_macro_top1", 0.0))
                gfv_10 = float(k2.get("common_full_vocab_macro_top10", 0.0)) - float(k0.get("common_full_vocab_macro_top10", 0.0))
                gfv_1 = (
                    float(k2["common_full_vocab_macro_top1"]) - float(k0["common_full_vocab_macro_top1"])
                    if k2.get("common_full_vocab_macro_top1") is not None and k0.get("common_full_vocab_macro_top1") is not None
                    else None
                )
                gfv_mrr = (
                    float(k2["common_full_vocab_mrr"]) - float(k0["common_full_vocab_mrr"])
                    if k2.get("common_full_vocab_mrr") is not None and k0.get("common_full_vocab_mrr") is not None
                    else None
                )
                lines.append(
                    f"{'  Δ Horizon Gain (K2 - K0)':<26} | "
                    f"{_fmt_signed_ratio_pct(g50_10, 9)} | "
                    f"{_fmt_signed_ratio_pct(g50_1, 8)} | "
                    f"{_fmt_signed_ratio_pct(gfv_10, 9)} | "
                    f"{_fmt_signed_ratio_pct(gfv_1, 8)} | "
                    f"{_fmt_signed_ratio_pct(gfv_mrr, 8)}"
                )

    if (
        s1_n50_t10 is not None
        and last_n50_t10 is not None
        and len(ordered_keys) >= 2
    ):
        dn50 = last_n50_t10 - s1_n50_t10
        dfull = (
            (last_full_t10 - s1_full_t10)
            if (s1_full_t10 is not None and last_full_t10 is not None)
            else None
        )
        lines.append(rule)
        lines.append(
            f"{'Net Gain vs Stage 1 (CNN)':<26} | "
            f"{_fmt_signed_ratio_pct(dn50, 9)} | {'-':>8} | "
            f"{_fmt_signed_ratio_pct(dfull, 9)} | {'-':>8} | {'-':>8}"
        )
    lines.append(rule)

    has_ctrl = any(
        isinstance(stages_map.get(k), Mapping)
        and bool(
            (
                (
                    stages_map[k].get("negative_controls")
                    or stages_map[k].get("negative_control")
                    or {}
                ).get("conditions")
            )
        )
        for k in ordered_keys
    )
    if has_ctrl:
        lines.append(
            "\n[2] Cross-Stage Negative Control (Clean 0s vs. Shifted Mean)"
        )
        c_hdr = (
            f"{'Stage':<18} | {'50wClean':>8} | {'50wShift':>8} | {'50w Gap':>8} | "
            f"{'FulClean':>8} | {'FulShift':>8} | {'Ful Gap':>8}"
        )
        lines.append(rule)
        lines.append(c_hdr)
        lines.append(rule)
        for st_key in ordered_keys:
            res = stages_map.get(st_key)
            if not isinstance(res, Mapping):
                continue
            neg_ctrl = res.get("negative_controls") or res.get("negative_control") or {}
            if not neg_ctrl.get("conditions"):
                continue
            c_ref = neg_ctrl.get("clean_reference") or {}
            s_agg = neg_ctrl.get("shifted_aggregate") or {}
            short_lbl = STAGE_SHORT_LABELS.get(st_key, st_key).replace("\n", " ")[:18]
            c_fv_m10 = c_ref.get("full_vocab_word_macro_top10", c_ref.get("full_vocab_macro_top10"))
            s_fv_m10 = s_agg.get("mean_full_vocab_word_macro_top10", s_agg.get("mean_full_vocab_macro_top10"))
            g_fv_m10 = s_agg.get(
                "clean_minus_shifted_full_vocab_word_macro_top10",
                s_agg.get("clean_minus_shifted_full_vocab_macro_top10"),
            )
            lines.append(
                f"{short_lbl:<18} | "
                f"{_fmt_ratio_pct(c_ref.get('vocab50_macro_top10'), 8)} | "
                f"{_fmt_ratio_pct(s_agg.get('mean_vocab50_macro_top10'), 8)} | "
                f"{_fmt_signed_ratio_pct(s_agg.get('clean_minus_shifted_vocab50_macro_top10'), 8)} | "
                f"{_fmt_ratio_pct(c_fv_m10, 8)} | "
                f"{_fmt_ratio_pct(s_fv_m10, 8)} | "
                f"{_fmt_signed_ratio_pct(g_fv_m10, 8)}"
            )
        lines.append(rule)

    if verbose:
        for st_key in ordered_keys:
            res = stages_map.get(st_key)
            if isinstance(res, Mapping):
                lines.append("")
                lines.append(format_word_evaluation_report(res, verbose=True))
    else:
        lines.append(
            f"[Tip] Full tables: stage_comparison_report_{split.lower()}.md, .txt & tables/*.csv"[
                :REPORT_WIDTH
            ]
        )
        lines.append(banner)

    return "\n".join(lines)


def print_stage_comparison_report(
    stage_results_or_summary: Mapping[str, Any],
    *,
    split: str = "val",
    verbose: bool = False,
    output_dir: Path | str | None = None,
) -> str:
    """Print unified multi-stage comparison report and optionally save .txt/.md/CSV files."""
    if output_dir:
        out_dir_path = Path(str(output_dir))
        out_path = out_dir_path / f"stage_comparison_report_{split.lower()}.txt"
        try:
            out_dir_path.mkdir(parents=True, exist_ok=True)
            full_txt = format_stage_comparison_report(
                stage_results_or_summary, split=split, verbose=True
            )
            out_path.write_text(full_txt + "\n", encoding="utf-8")
            export_stage_comparison_tables_and_markdown(
                stage_results_or_summary, out_dir_path, split=split
            )
        except OSError:
            pass
    rendered = format_stage_comparison_report(
        stage_results_or_summary, split=split, verbose=verbose
    )
    print("\n" + rendered, flush=True)
    return rendered


# ============================================================================
# 3. Leave-One-Out Common-Mode Subtraction Report
# ============================================================================


def format_common_mode_subtraction_report(
    report: Mapping[str, Any],
    *,
    verbose: bool = False,
) -> str:
    """Format Leave-One-Out Common-Mode Subtraction sweep report (<= 86 cols)."""
    raw_sweep = report.get("sweep")
    if isinstance(raw_sweep, Mapping):
        sweep_list = [v for v in raw_sweep.values() if isinstance(v, Mapping)]
    elif isinstance(raw_sweep, Sequence):
        sweep_list = [v for v in raw_sweep if isinstance(v, Mapping)]
    else:
        return ""
    if not sweep_list:
        return ""

    dataset = str(report.get("dataset", "-"))
    stage = str(report.get("stage", "-"))
    split = str(report.get("split", "val"))
    seed = report.get("seed", 0)
    f_info = report.get("formula") or {}
    m_count = f_info.get("num_shifted_conditions_M", 6)
    loo_div = f_info.get("shifted_loo_divisor", 5)

    banner = "=" * REPORT_WIDTH
    rule = "-" * REPORT_WIDTH
    lines: list[str] = [
        banner,
        f"LOO Common-Mode Subtraction | {dataset} | {stage} ({split.upper()}) | Seed: {seed}",
        f"Clean Div: 1/{m_count} historical shifts | Shifted LOO Div: 1/{loo_div} shifts",
        banner,
        "[1A] 50-Word Retrieval & Net Neural Gap by Weight η",
    ]
    hdr_50 = (
        f"{'Weight (eta)':<14} | {'50w Clean':>9} | {'50w Shift':>9} | {'50w Gap':>9} | "
        f"{'vs eta=0':>8} | {'50w Top1':>9} | {'ShiftTop1':>9}"
    )
    lines.append(rule)
    lines.append(hdr_50)
    lines.append(rule)

    has_horizons = False
    for item in sweep_list:
        eta = float(item.get("eta", 0.0))
        c_ref = item.get("clean_reference") or {}
        agg = item.get("shifted_aggregate") or {}
        d_eta0 = item.get("delta_vs_eta0")
        if c_ref.get("horizons"):
            has_horizons = True
        eta_lbl = f"eta={eta:.2f} (raw)" if abs(eta) <= 1e-9 else f"eta={eta:.2f}"
        d50_str = (
            "base".rjust(8)
            if d_eta0 is None
            else _fmt_signed_ratio_pct(d_eta0.get("vocab50_macro_top10"), 8)
        )
        lines.append(
            f"{eta_lbl:<14} | "
            f"{_fmt_ratio_pct(c_ref.get('vocab50_macro_top10'), 9)} | "
            f"{_fmt_ratio_pct(agg.get('mean_vocab50_macro_top10'), 9)} | "
            f"{_fmt_signed_ratio_pct(agg.get('clean_minus_shifted_vocab50_macro_top10'), 9)} | "
            f"{d50_str} | "
            f"{_fmt_ratio_pct(c_ref.get('vocab50_macro_top1'), 9)} | "
            f"{_fmt_ratio_pct(agg.get('mean_vocab50_macro_top1'), 9)}"
        )
    lines.append(rule)

    lines.append(
        "\n[1B] Full-Vocabulary Retrieval (Macro Top-10, Macro Top-1 & MRR) by Weight η"
    )
    hdr_full = (
        f"{'Weight (eta)':<14} | {'FullTop10':>9} | {'ShiftT10':>9} | {'Full Gap':>9} | "
        f"{'FullTop1':>9} | {'ShiftTop1':>9} | {'CleanMRR':>8}"
    )
    lines.append(rule)
    lines.append(hdr_full)
    lines.append(rule)
    for item in sweep_list:
        eta = float(item.get("eta", 0.0))
        c_ref = item.get("clean_reference") or {}
        agg = item.get("shifted_aggregate") or {}
        eta_lbl = f"eta={eta:.2f} (raw)" if abs(eta) <= 1e-9 else f"eta={eta:.2f}"
        c_fv_m10 = c_ref.get("full_vocab_word_macro_top10", c_ref.get("full_vocab_macro_top10"))
        s_fv_m10 = agg.get("mean_full_vocab_word_macro_top10", agg.get("mean_full_vocab_macro_top10"))
        g_fv_m10 = (
            float(c_fv_m10) - float(s_fv_m10)
            if c_fv_m10 is not None and s_fv_m10 is not None
            else agg.get("clean_minus_shifted_full_vocab_macro_top10")
        )
        lines.append(
            f"{eta_lbl:<14} | "
            f"{_fmt_ratio_pct(c_fv_m10, 9)} | "
            f"{_fmt_ratio_pct(s_fv_m10, 9)} | "
            f"{_fmt_signed_ratio_pct(g_fv_m10, 9)} | "
            f"{_fmt_ratio_pct(c_ref.get('full_vocab_macro_top1'), 9)} | "
            f"{_fmt_ratio_pct(agg.get('mean_full_vocab_macro_top1'), 9)} | "
            f"{float(c_ref.get('full_vocab_mrr', 0.0)):>8.4f}"
        )
    lines.append(rule)

    if has_horizons:
        h_hdr = (
            f"{'Weight (eta)':<14} | {'K0 Clean/Shift (Gap)':>21} | "
            f"{'K1 Clean/Shift (Gap)':>21} | {'K2 Clean/Shift (Gap)':>21}"
        )
        for sec_tag, sec_title, metric_key in (
            ("[2A]", "Revision Horizons (K0 / K1 / K2) 50-Word Macro Top-10 Across η", "common_vocab50_macro_top10"),
            ("[2B]", "Revision Horizons (K0 / K1 / K2) 50-Word Macro Top-1 Across η", "common_vocab50_macro_top1"),
            ("[2C]", "Revision Horizons (K0 / K1 / K2) Full-Vocab Macro Top-10 Across η", "common_full_vocab_macro_top10"),
            ("[2D]", "Revision Horizons (K0 / K1 / K2) Full-Vocab Macro Top-1 Across η", "common_full_vocab_macro_top1"),
        ):
            lines.append(f"\n{sec_tag} {sec_title}")
            lines.append(rule)
            lines.append(h_hdr)
            lines.append(rule)
            for item in sweep_list:
                eta = float(item.get("eta", 0.0))
                eta_lbl = f"eta={eta:.2f} (raw)" if abs(eta) <= 1e-9 else f"eta={eta:.2f}"
                c_hz = (item.get("clean_reference") or {}).get("horizons") or {}
                s_hz = (item.get("shifted_aggregate") or {}).get("horizons") or {}
                cols: list[str] = []
                for hk in ("K0", "K1", "K2"):
                    if hk in c_hz:
                        raw_ch = (
                            c_hz[hk].get("common_full_vocab_word_macro_top10", c_hz[hk].get(metric_key))
                            if metric_key == "common_full_vocab_macro_top10"
                            else c_hz[hk].get(metric_key)
                        )
                        raw_sh = (
                            (s_hz.get(hk) or {}).get("mean_common_full_vocab_word_macro_top10", (s_hz.get(hk) or {}).get(f"mean_{metric_key}"))
                            if metric_key == "common_full_vocab_macro_top10"
                            else (s_hz.get(hk) or {}).get(f"mean_{metric_key}")
                        )
                        if raw_ch is not None and raw_sh is not None:
                            ch_v = float(raw_ch) * 100.0
                            sh_v = float(raw_sh) * 100.0
                            gp_v = ch_v - sh_v
                            cols.append(f"{ch_v:5.1f}/{sh_v:4.1f}({gp_v:+5.1f}%)")
                        else:
                            cols.append("N/A")
                    else:
                        cols.append("N/A")
                lines.append(
                    f"{eta_lbl:<14} | {cols[0]:>21} | {cols[1]:>21} | {cols[2]:>21}"
                )
            lines.append(rule)

    if verbose and sweep_list and sweep_list[0].get("conditions"):
        lines.append("\n[3] Per-Condition 50-Word Macro Top-10 Breakdown Across η")
        eta_hdrs = " | ".join(
            f"{'eta=' + format(float(it['eta']), '.1f'):>9}" for it in sweep_list
        )
        lines.append(rule)
        lines.append(f"{'Condition':<18} | {eta_hdrs}")
        lines.append(rule)
        clean_vals = " | ".join(
            f"{float(it['clean_reference']['vocab50_macro_top10']) * 100.0:8.2f}%"
            for it in sweep_list
        )
        lines.append(f"{'Clean (0s)':<18} | {clean_vals}")
        for idx, cond in enumerate(sweep_list[0].get("conditions", ())):
            c_lbl = str(cond.get("label", "-"))[:18]
            row_vals = " | ".join(
                f"{float(it['conditions'][idx]['summary']['vocab50_macro_top10']) * 100.0:8.2f}%"
                for it in sweep_list
            )
            lines.append(f"{c_lbl:<18} | {row_vals}")
        lines.append(rule)

    lines.append(banner)
    return "\n".join(lines)


def print_common_mode_subtraction_report(
    report: Mapping[str, Any],
    *,
    verbose: bool = False,
    output_dir: Path | str | None = None,
) -> str:
    """Print Leave-One-Out Common-Mode Subtraction report."""
    rendered = format_common_mode_subtraction_report(report, verbose=verbose)
    if not rendered:
        return ""
    if output_dir:
        split = str(report.get("split", "val")).lower()
        out_path = (
            Path(str(output_dir))
            / "exploratory"
            / f"common_mode_subtraction_report_{split}.txt"
        )
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                format_common_mode_subtraction_report(report, verbose=True) + "\n",
                encoding="utf-8",
            )
            export_common_mode_subtraction_tables_and_markdown(report, output_dir)
        except OSError:
            pass
    print("\n" + rendered, flush=True)
    return rendered


# ============================================================================
# 4. Subspace Coherence LOO Debiasing Report
# ============================================================================


def format_subspace_coherence_debiasing_report(
    report: Mapping[str, Any],
    *,
    verbose: bool = False,
) -> str:
    """Format Scheme 1 (Subspace Projection + Coherence Gate LOO) report (<= 86 cols).

    Preserves compact <= 86-column width while keeping exact test contract strings:
    - '50w Clean Top1'
    - '50w Shift Top1'
    - '3B. Revision Horizons (K0 / K1 / K2) 50-Word Macro Top-1'
    """
    dataset = str(report.get("dataset", "-"))
    stage = str(report.get("stage", "-"))
    split = str(report.get("split", "test"))
    seed = report.get("seed", 0)
    exp_tag = report.get("exp_tag", "") or "(default)"
    f_info = report.get("formula") or {}
    m_count = f_info.get("num_shifted_conditions_M", 6)
    center_svd = bool(f_info.get("center_svd", True))

    exp_tag_short = str(exp_tag)[:26]
    banner = "=" * REPORT_WIDTH
    rule = "-" * REPORT_WIDTH
    lines: list[str] = [
        banner,
        f"Subspace-Projected + Coherence-Gated LOO Debiasing | {dataset} | {stage} ({split.upper()})",
        f"Seed: {seed} | Tag: {exp_tag_short} | CenterSVD: {center_svd} | Shifts: M={m_count} (LOO={m_count - 1})",
        banner,
    ]

    diags = report.get("subspace_and_coherence_diagnostics") or {}
    if isinstance(diags, Mapping) and diags:
        lines.append("1. Shift Common-Mode SVD Energy Spectrum & Coherence Gate Diagnostics")
        d_hdr = (
            f"{'Hzn':<4} | {'EffRk':>5} | {'Top8%':>6} | {'Top12%':>6} | "
            f"{'Top16%':>6} | {'Top24%':>6} | {'||y_sh|| Mean±SD':>17} | {'GateMean':>8}"
        )
        lines.append(rule)
        lines.append(d_hdr)
        lines.append(rule)
        for h_key, d_item in diags.items():
            if not isinstance(d_item, Mapping):
                continue
            cum = d_item.get("cumulative_energy_explained") or {}
            coh = d_item.get("coherence_norm") or {}
            gt = d_item.get("adaptive_gate") or {}
            c_m = float(coh.get("mean", 0.0))
            c_sd = float(coh.get("std", 0.0))
            lines.append(
                f"{str(h_key)[:4]:<4} | "
                f"{float(d_item.get('effective_rank_participation_ratio', 0.0)):>5.1f} | "
                f"{_fmt_ratio_pct(cum.get('top_8'), 6)} | "
                f"{_fmt_ratio_pct(cum.get('top_12'), 6)} | "
                f"{_fmt_ratio_pct(cum.get('top_16'), 6)} | "
                f"{_fmt_ratio_pct(cum.get('top_24'), 6)} | "
                f"{f'{c_m:.3f}±{c_sd:.3f}':>17} | "
                f"{float(gt.get('mean', 1.0)):>8.3f}"
            )
        lines.append(rule)

    ctx_ref = report.get("context_stage_reference")
    if isinstance(ctx_ref, Mapping) and ctx_ref.get("vocab50_macro_top10") is not None:
        lines.append(
            f"[Ref] Stage 2.1 Context: 50w Top10={float(ctx_ref['vocab50_macro_top10'])*100:.2f}% | "
            f"50w Top1={float(ctx_ref.get('vocab50_macro_top1', 0.0))*100:.2f}% | "
            f"FullTop10={float(ctx_ref.get('full_vocab_macro_top10', 0.0))*100:.2f}%"
        )
        lines.append(rule)

    raw_base = report.get("raw_baseline")
    all_rows: list[Mapping[str, Any]] = []
    if isinstance(raw_base, Mapping):
        all_rows.append(raw_base)
    for r in report.get("full_rank_loo_baselines") or ():
        if isinstance(r, Mapping):
            all_rows.append(r)
    for r in report.get("subspace_sweep") or ():
        if isinstance(r, Mapping):
            all_rows.append(r)

    if not all_rows:
        lines.append(banner)
        return "\n".join(lines)

    best_gap = report.get("best_by_net_gap_vocab50")
    best_clean = report.get("best_by_clean_preservation_vocab50")

    if not verbose and len(all_rows) > 10:
        selected_ids: set[str] = set()
        if isinstance(raw_base, Mapping):
            selected_ids.add(str(raw_base.get("method_id", "raw_baseline")))
        for r in (report.get("full_rank_loo_baselines") or ())[:2]:
            if isinstance(r, Mapping):
                selected_ids.add(str(r.get("method_id", "")))
        if isinstance(best_clean, Mapping):
            selected_ids.add(str(best_clean.get("method_id", "")))
        if isinstance(best_gap, Mapping):
            selected_ids.add(str(best_gap.get("method_id", "")))
        sub_sorted = sorted(
            [r for r in (report.get("subspace_sweep") or ()) if isinstance(r, Mapping)],
            key=lambda it: float(
                (it.get("shifted_aggregate") or {}).get(
                    "clean_minus_shifted_vocab50_macro_top10", 0.0
                )
            ),
            reverse=True,
        )
        for r in sub_sorted[:5]:
            selected_ids.add(str(r.get("method_id", "")))
        display_rows = [
            r for r in all_rows if str(r.get("method_id", "")) in selected_ids
        ]
    else:
        display_rows = all_rows

    lines.append(
        "\n2A. Feature-Space Subspace LOO Retrieval Comparison (50w Top-10 & Top-1)"
    )
    m_hdr = (
        f"{'Configuration':<22} | {'50wT10':>7} | {'ShftT10':>7} | {'T10Gap':>7} | "
        f"{'50w Clean Top1':>14} | {'50w Shift Top1':>14}"
    )
    lines.append(rule)
    lines.append(m_hdr)
    lines.append(rule)

    for item in display_rows:
        lbl = _short_method_label(item)[:22]
        c_ref = item.get("clean_reference") or {}
        agg = item.get("shifted_aggregate") or {}
        lines.append(
            f"{lbl:<22} | "
            f"{_fmt_ratio_pct(c_ref.get('vocab50_macro_top10'), 7)} | "
            f"{_fmt_ratio_pct(agg.get('mean_vocab50_macro_top10'), 7)} | "
            f"{_fmt_signed_ratio_pct(agg.get('clean_minus_shifted_vocab50_macro_top10'), 7)} | "
            f"{_fmt_ratio_pct(c_ref.get('vocab50_macro_top1'), 14)} | "
            f"{_fmt_ratio_pct(agg.get('mean_vocab50_macro_top1'), 14)}"
        )
    lines.append(rule)

    lines.append(
        "\n2B. Feature-Space Subspace LOO Full-Vocabulary (Macro Top-10, Top-1 & MRR)"
    )
    f_hdr = (
        f"{'Configuration':<22} | {'FullT10':>7} | {'ShftT10':>7} | {'FullGap':>8} | "
        f"{'FullTop1':>8} | {'ShftTop1':>8} | {'CleanMRR':>8}"
    )
    lines.append(rule)
    lines.append(f_hdr)
    lines.append(rule)
    for item in display_rows:
        lbl = _short_method_label(item)[:22]
        c_ref = item.get("clean_reference") or {}
        agg = item.get("shifted_aggregate") or {}
        c_fv_m10 = c_ref.get("full_vocab_word_macro_top10", c_ref.get("full_vocab_macro_top10"))
        s_fv_m10 = agg.get("mean_full_vocab_word_macro_top10", agg.get("mean_full_vocab_macro_top10"))
        g_fv_m10 = (
            float(c_fv_m10) - float(s_fv_m10)
            if c_fv_m10 is not None and s_fv_m10 is not None
            else agg.get("clean_minus_shifted_full_vocab_macro_top10")
        )
        lines.append(
            f"{lbl:<22} | "
            f"{_fmt_ratio_pct(c_fv_m10, 7)} | "
            f"{_fmt_ratio_pct(s_fv_m10, 7)} | "
            f"{_fmt_signed_ratio_pct(g_fv_m10, 8)} | "
            f"{_fmt_ratio_pct(c_ref.get('full_vocab_macro_top1'), 8)} | "
            f"{_fmt_ratio_pct(agg.get('mean_full_vocab_macro_top1'), 8)} | "
            f"{float(c_ref.get('full_vocab_mrr', 0.0)):>8.4f}"
        )
    lines.append(rule)
    if not verbose and len(all_rows) > len(display_rows):
        lines.append(
            f"  * Showing {len(display_rows)}/{len(all_rows)} key configs. Pass --verbose for full {len(all_rows)}-row grid."
        )

    raw_hz = (
        (raw_base.get("clean_reference") or {}).get("horizons")
        if isinstance(raw_base, Mapping)
        else None
    )
    if isinstance(raw_hz, Mapping) and raw_hz:
        hz_rows = display_rows[:6] if not verbose and len(display_rows) > 6 else display_rows
        h_hdr = (
            f"{'Configuration':<20} | {'K0 Cln/Shf (Gap)':>19} | "
            f"{'K1 Cln/Shf (Gap)':>19} | {'K2 Cln/Shf (Gap)':>19}"
        )
        for sec_title, metric_key in (
            (
                "3A. Revision Horizons (K0 / K1 / K2) 50-Word Macro Top-10 Clean/Shift (Gap)",
                "common_vocab50_macro_top10",
            ),
            (
                "3B. Revision Horizons (K0 / K1 / K2) 50-Word Macro Top-1 Clean vs Strict-LOO Shifted",
                "common_vocab50_macro_top1",
            ),
            (
                "3C. Revision Horizons (K0 / K1 / K2) Full-Vocab Macro Top-10 Clean/Shift (Gap)",
                "common_full_vocab_macro_top10",
            ),
            (
                "3D. Revision Horizons (K0 / K1 / K2) Full-Vocab Macro Top-1 Clean/Shift (Gap)",
                "common_full_vocab_macro_top1",
            ),
        ):
            lines.append(f"\n{sec_title}")
            lines.append(rule)
            lines.append(h_hdr)
            lines.append(rule)
            for item in hz_rows:
                lbl = _short_method_label(item)[:20]
                c_hz = (item.get("clean_reference") or {}).get("horizons") or {}
                s_hz = (item.get("shifted_aggregate") or {}).get("horizons") or {}
                cols: list[str] = []
                for hk in ("K0", "K1", "K2"):
                    if hk in c_hz:
                        raw_ch = (
                            c_hz[hk].get("common_full_vocab_word_macro_top10", c_hz[hk].get(metric_key))
                            if metric_key == "common_full_vocab_macro_top10"
                            else c_hz[hk].get(metric_key)
                        )
                        raw_sh = (
                            (s_hz.get(hk) or {}).get("mean_common_full_vocab_word_macro_top10", (s_hz.get(hk) or {}).get(f"mean_{metric_key}"))
                            if metric_key == "common_full_vocab_macro_top10"
                            else (s_hz.get(hk) or {}).get(f"mean_{metric_key}")
                        )
                        if raw_ch is not None and raw_sh is not None:
                            ch_v = float(raw_ch) * 100.0
                            sh_v = float(raw_sh) * 100.0
                            gp_v = ch_v - sh_v
                            cols.append(f"{ch_v:4.1f}/{sh_v:4.1f}({gp_v:+5.1f}%)")
                        else:
                            cols.append("N/A")
                    else:
                        cols.append("N/A")
                lines.append(
                    f"{lbl:<20} | {cols[0]:>19} | {cols[1]:>19} | {cols[2]:>19}"
                )
            lines.append(rule)

    if isinstance(best_clean, Mapping):
        bc_c = best_clean.get("clean_reference") or {}
        bc_s = best_clean.get("shifted_aggregate") or {}
        lines.append(
            f"[Best Clean] {_short_method_label(best_clean)[:22]}: "
            f"Clean50={_fmt_ratio_pct(bc_c.get('vocab50_macro_top10'), 6)} | "
            f"Shift50={_fmt_ratio_pct(bc_s.get('mean_vocab50_macro_top10'), 6)} | "
            f"Gap={_fmt_signed_ratio_pct(bc_s.get('clean_minus_shifted_vocab50_macro_top10'), 6)}"
        )
    if isinstance(best_gap, Mapping):
        bg_c = best_gap.get("clean_reference") or {}
        bg_s = best_gap.get("shifted_aggregate") or {}
        lines.append(
            f"[Best Gap]   {_short_method_label(best_gap)[:22]}: "
            f"Clean50={_fmt_ratio_pct(bg_c.get('vocab50_macro_top10'), 6)} | "
            f"Shift50={_fmt_ratio_pct(bg_s.get('mean_vocab50_macro_top10'), 6)} | "
            f"Gap={_fmt_signed_ratio_pct(bg_s.get('clean_minus_shifted_vocab50_macro_top10'), 6)}"
        )
    lines.append(banner)
    return "\n".join(lines)


def print_subspace_coherence_debiasing_report(
    report: Mapping[str, Any],
    *,
    verbose: bool = False,
    output_dir: Path | str | None = None,
) -> str:
    """Print Scheme 1 (Subspace Projection + Coherence Gate LOO) report."""
    rendered = format_subspace_coherence_debiasing_report(report, verbose=verbose)
    if not rendered:
        return ""
    if output_dir:
        split = str(report.get("split", "test")).lower()
        out_path = (
            Path(str(output_dir))
            / "exploratory"
            / f"subspace_coherence_debiasing_report_{split}.txt"
        )
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                format_subspace_coherence_debiasing_report(report, verbose=True)
                + "\n",
                encoding="utf-8",
            )
            export_subspace_coherence_debiasing_tables_and_markdown(
                report, output_dir
            )
        except OSError:
            pass
    print("\n" + rendered, flush=True)
    return rendered


# ============================================================================
# 5. S3 Sequence Decoding Executive Summary & Full 6-Table Report
# ============================================================================


def _compute_s3_revision_gain_deltas(
    cond_metrics: Mapping[str, Mapping[str, float | int]],
) -> dict[str, dict[str, float]]:
    gains: dict[str, dict[str, float]] = {}
    metric_keys = (
        "bleu1_pct",
        "corpus_bleu1_pct",
        "beam_oracle_bleu1_pct",
        "word_acc_pct",
        "in_vocabulary_word_acc_pct",
        "word_acc_top5_pct",
        "content_word_acc_pct",
        "function_word_acc_pct",
        "cer_pct",
        "wer_pct",
        "bertscore_f1_pct",
    )
    if "S3_clean" in cond_metrics and "K0" in cond_metrics:
        c_m = cond_metrics["S3_clean"]
        a_m = cond_metrics["K0"]
        gains["clean_C_minus_A_pp"] = {
            mk: round(float(c_m.get(mk, 0.0)) - float(a_m.get(mk, 0.0)), 4)
            for mk in metric_keys
            if mk in c_m and mk in a_m
        }
    if (
        "S3_shifted_neg75s" in cond_metrics
        and "K0_shifted_neg75s" in cond_metrics
    ):
        cs_m = cond_metrics["S3_shifted_neg75s"]
        as_m = cond_metrics["K0_shifted_neg75s"]
        gains["shifted_C_minus_A_pp"] = {
            mk: round(float(cs_m.get(mk, 0.0)) - float(as_m.get(mk, 0.0)), 4)
            for mk in metric_keys
            if mk in cs_m and mk in as_m
        }
    if "S3_clean" in cond_metrics and "S3_shifted_neg75s" in cond_metrics:
        c_m = cond_metrics["S3_clean"]
        cs_m = cond_metrics["S3_shifted_neg75s"]
        gains["clean_minus_shifted_S3_pp"] = {
            mk: round(float(c_m.get(mk, 0.0)) - float(cs_m.get(mk, 0.0)), 4)
            for mk in metric_keys
            if mk in c_m and mk in cs_m
        }
    return gains


def _filter_main_latency_conditions(
    conds_map: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Filter out F_local and online C-Gated entries from main accuracy tables and order by latency tier."""
    preferred_order = (
        "S0",
        "S0_clean",
        "K0",
        "K0_clean",
        "K1",
        "K1_clean",
        "S3_clean",
        "F",
        "F_clean",
        "LM-Only",
        "LM-Only_clean",
        "K0_shifted_neg75s",
        "S3_shifted_neg75s",
    )
    filtered: dict[str, Mapping[str, Any]] = {}
    for k in preferred_order:
        m = conds_map.get(k)
        if not isinstance(m, Mapping):
            continue
        if k == "F_local" or str(k).endswith("_c_gated") or bool(m.get("is_online_screen_gate")):
            continue
        filtered[str(k)] = m
    for k, m in conds_map.items():
        if str(k) in filtered or not isinstance(m, Mapping):
            continue
        if str(k) == "F_local" or str(k).endswith("_c_gated") or bool(m.get("is_online_screen_gate")):
            continue
        filtered[str(k)] = m
    return filtered


def format_sequence_decoding_executive_summary(
    report: Mapping[str, Any],
) -> str:
    """Format a compact <= 86-column Executive Summary for S3 sequence decoding."""
    dataset = str(report.get("dataset", "pallier2025"))
    split = str(report.get("split", "test"))
    seed = report.get("seed", 0)
    exp_tag = report.get("exp_tag") or "(default)"
    cms_eta = report.get("cms_eta", 0.0)

    banner = "=" * REPORT_WIDTH
    rule = "-" * REPORT_WIDTH
    lines: list[str] = [
        banner,
        f"S3 Sequence Decoding Executive Summary | {dataset} ({split.upper()}) | Seed: {seed}",
        f"ExpTag: {exp_tag} | CMS η: {cms_eta}",
        banner,
    ]

    state_hp = report.get("state_hyperparameters")
    if isinstance(state_hp, Mapping) and state_hp:
        hp_parts = [
            f"{st}:(T={state_hp[st].get('temperature','-')},λ={state_hp[st].get('language_weight','-')})"
            for st in ("S0", "K0", "K1", "K2", "F")
            if isinstance(state_hp.get(st), Mapping)
        ]
        if hp_parts:
            lines.append("Calibrated: " + " | ".join(hp_parts))
            lines.append(rule)

    conds = report.get("conditions")
    if isinstance(conds, Mapping) and conds:
        main_conds = _filter_main_latency_conditions(conds)
        first_c = next(iter(main_conds.values()), next(iter(conds.values()), {}))
        segs = first_c.get("segment_count", "-")
        words = first_c.get("total_words", "-")
        lines.append(
            f"[1] Latency-Budget Conditions (Segments={segs}, TotalWords={words})"[:REPORT_WIDTH]
        )
        c_hdr = (
            f"{'Condition':<22} | {'BLEU-1':>7} | {'InVoc%':>7} | "
            f"{'WordAcc':>7} | {'Acc@5':>7} | {'CER(%)':>7} | {'BERT-F1':>7}"
        )
        lines.append(rule)
        lines.append(c_hdr)
        lines.append(rule)
        for c_name, m in main_conds.items():
            lines.append(
                f"{str(c_name)[:22]:<22} | "
                f"{_fmt_pct(m.get('bleu1_pct'), 7)} | "
                f"{_fmt_pct(m.get('in_vocabulary_word_acc_pct'), 7)} | "
                f"{_fmt_pct(m.get('word_acc_pct'), 7)} | "
                f"{_fmt_pct(m.get('word_acc_top5_pct'), 7)} | "
                f"{_fmt_pct(m.get('cer_pct'), 7)} | "
                f"{_fmt_pct(m.get('bertscore_f1_pct'), 7)}"
            )
        lines.append(rule)
        gains = _compute_s3_revision_gain_deltas(main_conds)
        for g_key, g_lbl in (
            ("clean_C_minus_A_pp", "Clean Revision Gain (S3 - K0)"),
            ("clean_minus_shifted_S3_pp", "Net Neural Advantage (Clean - Shift)"),
        ):
            gv = gains.get(g_key)
            if isinstance(gv, Mapping) and gv:
                lines.append(
                    f" * {g_lbl:<31}: "
                    f"ΔBLEU1={_fmt_signed_pct(gv.get('bleu1_pct'), 6)} | "
                    f"ΔInVoc={_fmt_signed_pct(gv.get('in_vocabulary_word_acc_pct'), 6)} | "
                    f"ΔWAcc={_fmt_signed_pct(gv.get('word_acc_pct'), 6)}"
                )
        s3_sig = report.get("statistical_significance") or {}
        if isinstance(s3_sig, Mapping):
            sig_c = s3_sig.get("s3_clean_vs_k0_bleu1")
            if isinstance(sig_c, Mapping) and sig_c.get("ci_lower_pp") is not None:
                lines.append(
                    f" * K2 vs K0 BLEU-1 95% CI: "
                    f"[{float(sig_c['ci_lower_pp']):+.2f}, {float(sig_c['ci_upper_pp']):+.2f}] pp "
                    f"(p={float(sig_c.get('p_two_sided', 1.0)):.4f} {sig_c.get('significance', '')})"[
                        :REPORT_WIDTH
                    ]
                )
            sig_sh = s3_sig.get("s3_clean_vs_shifted_bleu1")
            if isinstance(sig_sh, Mapping) and sig_sh.get("ci_lower_pp") is not None:
                lines.append(
                    f" * Clean vs Shifted BLEU-1 95% CI: "
                    f"[{float(sig_sh['ci_lower_pp']):+.2f}, {float(sig_sh['ci_upper_pp']):+.2f}] pp "
                    f"(p={float(sig_sh.get('p_two_sided', 1.0)):.4f} {sig_sh.get('significance', '')})"[
                        :REPORT_WIDTH
                    ]
                )
        s3_sub_sum = report.get("per_subject_summary") or {}
        if isinstance(s3_sub_sum, Mapping) and s3_sub_sum.get("mean_bleu1_pct") is not None:
            mb = float(s3_sub_sum["mean_bleu1_pct"])
            sb = float(s3_sub_sum.get("std_bleu1_pct", 0.0))
            mw = float(s3_sub_sum.get("mean_word_acc_pct", 0.0))
            sw = float(s3_sub_sum.get("std_word_acc_pct", 0.0))
            lines.append(
                f" * Across-Subject K2: BLEU1={mb:.2f}±{sb:.2f}% | WordAcc={mw:.2f}±{sw:.2f}%"[
                    :REPORT_WIDTH
                ]
            )

    vs = report.get("vocabulary_scaling")
    if isinstance(vs, Mapping) and vs:
        lines.append(
            "\n[2] Core & Multi-Scale Vocabulary Scaling (All Segments vs. >=80% Core)"
        )
        v_hdr = (
            f"{'Scale |V|':<10} | {'TokCov':>6} | {'All K2':>9} | "
            f"{'All K0':>9} | {'All InV%':>8} | {'Core80':>9} | {'CoreInV%':>9}"
        )
        lines.append(rule)
        lines.append(v_hdr)
        lines.append(rule)
        ordered_scales = sorted(
            vs.items(),
            key=lambda kv: int(
                kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
            )
            if isinstance(kv[1], Mapping)
            else 0,
        )
        for s_key, s_entry in ordered_scales:
            if not isinstance(s_entry, Mapping):
                continue
            all_c = s_entry.get("conditions") or s_entry.get("all_segments") or {}
            s3_m = all_c.get("S3_clean") or {}
            k0_m = all_c.get("K0") or {}
            c80_blk = s_entry.get("core_subset_80pct") or {}
            c80_map = (
                c80_blk.get("conditions")
                if isinstance(c80_blk.get("conditions"), Mapping)
                else c80_blk
            ) or {}
            c80_s3 = c80_map.get("S3_clean") or {}
            lines.append(
                f"{str(s_key):<10} | "
                f"{_fmt_pct(s_entry.get('test_token_coverage_pct'), 6)} | "
                f"{_fmt_pct(s3_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_pct(k0_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_pct(s3_m.get('in_vocabulary_word_acc_pct'), 8)} | "
                f"{_fmt_pct(c80_s3.get('bleu1_pct'), 9)} | "
                f"{_fmt_pct(c80_s3.get('in_vocabulary_word_acc_pct'), 9)}"
            )
        lines.append(rule)

    ms = report.get("multi_subject_scaling")
    if isinstance(ms, Mapping) and ms:
        lines.append("\n[3] Multi-Subject Observation Scaling (k = 1..10)")
        m_hdr = (
            f"{'Subjects k':<12} | {'BLEU-1':>8} | {'OraclBLEU':>9} | "
            f"{'InVocAcc':>9} | {'WordAcc':>8} | {'Acc@5':>8} | {'BERT-F1':>8}"
        )
        lines.append(rule)
        lines.append(m_hdr)
        lines.append(rule)
        ordered_ms = sorted(
            ms.items(),
            key=lambda kv: int(
                kv[1].get("k_observations", str(kv[0]).lstrip("kK=_") or 1)
            )
            if isinstance(kv[1], Mapping)
            else 1,
        )
        for k_lbl, m in ordered_ms:
            if not isinstance(m, Mapping):
                continue
            lines.append(
                f"{str(k_lbl):<12} | "
                f"{_fmt_pct(m.get('bleu1_pct'), 8)} | "
                f"{_fmt_pct(m.get('beam_oracle_bleu1_pct'), 9)} | "
                f"{_fmt_pct(m.get('in_vocabulary_word_acc_pct'), 9)} | "
                f"{_fmt_pct(m.get('word_acc_pct'), 8)} | "
                f"{_fmt_pct(m.get('word_acc_top5_pct'), 8)} | "
                f"{_fmt_pct(m.get('bertscore_f1_pct'), 8)}"
            )
        lines.append(rule)

    lines.append(
        f"[Tip] Full reports: s3_evaluation_report_{split}.md, s3_decoded_sentences_{split}.md"[
            :REPORT_WIDTH
        ]
    )
    lines.append(banner)
    return "\n".join(lines)


def _format_s3_conditions_table_full(
    conds_map: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    lines: list[str] = []
    header = (
        f"{'Condition':<25} | {'BLEU-1':>7} | {'CorpBLEU':>8} | {'OraclBLEU':>9} | "
        f"{'WordAcc':>7} | {'InVocAcc':>8} | {'Acc@5':>7} | {'ContAcc':>7} | "
        f"{'FuncAcc':>7} | {'WER(%)':>7} | {'CER(%)':>7} | {'BERT-F1':>7}"
    )
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for cond_name, m in conds_map.items():
        if not isinstance(m, Mapping):
            continue
        lines.append(
            f"{str(cond_name):<25} | "
            f"{_fmt_pct(m.get('bleu1_pct'), 7)} | "
            f"{_fmt_pct(m.get('corpus_bleu1_pct'), 8)} | "
            f"{_fmt_pct(m.get('beam_oracle_bleu1_pct'), 9)} | "
            f"{_fmt_pct(m.get('word_acc_pct'), 7)} | "
            f"{_fmt_pct(m.get('in_vocabulary_word_acc_pct'), 8)} | "
            f"{_fmt_pct(m.get('word_acc_top5_pct'), 7)} | "
            f"{_fmt_pct(m.get('content_word_acc_pct'), 7)} | "
            f"{_fmt_pct(m.get('function_word_acc_pct'), 7)} | "
            f"{_fmt_pct(m.get('wer_pct'), 7)} | "
            f"{_fmt_pct(m.get('cer_pct'), 7)} | "
            f"{_fmt_pct(m.get('bertscore_f1_pct'), 7)}"
        )
    lines.append(sep)
    return lines


def _format_s3_revision_gains_block_full(
    gains: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if not gains:
        return []
    label_map = {
        "clean_C_minus_A_pp": "Clean 滚动修正净增益 (Rolling-K2 - Rolling-K0)",
        "shifted_C_minus_A_pp": "Shifted 错位滚动增益 (K2_shift - K0_shift)",
        "clean_minus_shifted_S3_pp": "真实 vs 错位地面实况净优势 (K2_clean - K2_shift)",
    }
    out: list[str] = []
    for g_key in (
        "clean_C_minus_A_pp",
        "shifted_C_minus_A_pp",
        "clean_minus_shifted_S3_pp",
    ):
        g_val = gains.get(g_key)
        if not isinstance(g_val, Mapping) or not g_val:
            continue
        out.append(
            f"  * {label_map.get(g_key, g_key):<46}: "
            f"ΔBLEU-1={_fmt_signed_pct(g_val.get('bleu1_pct'), 6)} pp | "
            f"ΔWordAcc={_fmt_signed_pct(g_val.get('word_acc_pct'), 6)} pp | "
            f"ΔInVocAcc={_fmt_signed_pct(g_val.get('in_vocabulary_word_acc_pct'), 6)} pp | "
            f"ΔCER={_fmt_signed_pct(g_val.get('cer_pct'), 6)} pp | "
            f"ΔBERT-F1={_fmt_signed_pct(g_val.get('bertscore_f1_pct'), 6)} pp"
        )
    return out


_format_conditions_table = _format_s3_conditions_table_full
_format_revision_gains_block = _format_s3_revision_gains_block_full


def format_sequence_decoding_report(
    report: Mapping[str, Any],
    *,
    verbose: bool = True,
) -> str:
    """Format S3 sequence decoding evaluation report.

    When ``verbose=True`` (default for `.txt` export and backward compatibility),
    returns the full 6-table report (`[表 1]` .. `[表 6]`).
    When ``verbose=False``, returns the compact ``<= 86``-column Executive Summary.
    """
    if not verbose:
        return format_sequence_decoding_executive_summary(report)

    dataset = str(report.get("dataset", "pallier2025"))
    split = str(report.get("split", "test"))
    seed = report.get("seed", 0)
    exp_tag = report.get("exp_tag") or "(default)"
    cms_eta = report.get("cms_eta", 0.0)
    f_mode = report.get("f_condition_mode") or ""

    banner = "=" * 132
    lines: list[str] = [
        banner,
        f"S3 序列解码协议评估汇总报告 | Dataset: {dataset} | Split: {split.upper()} | Seed: {seed} | ExpTag: {exp_tag} | CMS η: {cms_eta}"
        + (f" | F-Mode: {f_mode}" if f_mode else ""),
        banner,
    ]

    state_hp = report.get("state_hyperparameters")
    if isinstance(state_hp, Mapping) and state_hp:
        hp_parts: list[str] = []
        for st_name in ("S0", "K0", "K1", "K2", "F"):
            st_cfg = state_hp.get(st_name)
            if isinstance(st_cfg, Mapping):
                t_val = st_cfg.get("temperature", "-")
                l_val = st_cfg.get("language_weight", "-")
                hp_parts.append(f"{st_name}: (T={t_val}, λ={l_val})")
        if hp_parts:
            lines.append("[状态自适应校准超参] " + " | ".join(hp_parts))
            lines.append("-" * 132)

    conds = report.get("conditions")
    if isinstance(conds, Mapping) and conds:
        main_conds = _filter_main_latency_conditions(conds)
        first_cond = next(iter(main_conds.values()), next(iter(conds.values()), {}))
        tot_segs = first_cond.get("segment_count", "-")
        tot_words = first_cond.get("total_words", "-")
        invoc_rate = first_cond.get("in_vocabulary_rate_pct", None)
        cov_str = f"{float(invoc_rate):.2f}%" if invoc_rate is not None else "-"
        lines.append(
            f"\n[表 1] 全词库主实验结果 — 4 档输入延迟预算统一束搜索对比 (Full Vocabulary | Segments={tot_segs}, TotalWords={tot_words}, TokenCoverage={cov_str})"
        )
        lines.extend(_format_s3_conditions_table_full(main_conds))
        main_gains = _compute_s3_revision_gain_deltas(main_conds)
        lines.extend(_format_s3_revision_gains_block_full(main_gains))

    core_benchmarks = report.get("core_vocabulary_benchmarks")
    core_bench_list: list[Mapping[str, Any]] = []
    if isinstance(core_benchmarks, Mapping) and core_benchmarks:
        ordered_cb = sorted(
            core_benchmarks.items(),
            key=lambda kv: int(
                kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
            )
            if isinstance(kv[1], Mapping)
            else 0,
        )
        core_bench_list = [
            cb for _, cb in ordered_cb if isinstance(cb, Mapping) and cb.get("subsets")
        ]
    else:
        single_cb = report.get("core_vocabulary_benchmark")
        if isinstance(single_cb, Mapping) and single_cb.get("subsets"):
            core_bench_list = [single_cb]

    if core_bench_list:
        lines.append(
            "\n[表 2] 高频核心词表三档基准对比 (Core Vocabulary |V| ∈ {50, 100, 250}，对标 Moses 2021 / SimpleB2T / Jayalath 2025 / d'Ascoli 2025)"
        )
        subset_titles = (
            ("all_segments", "口径 1 [all_segments]: 全部测试句段 (含 In-Vocab Word Acc)"),
            ("core_subset_100pct", "口径 2a [core_subset_100pct]: 100% 词表内核心句 (0% OOV Core Set，直接对标 SimpleB2T)"),
            ("core_subset_80pct", "口径 2b [core_subset_80pct]: >=80% 词表内高覆盖核心句"),
        )
        for cb_entry in core_bench_list:
            v_core = cb_entry.get("vocabulary_size", 250)
            tok_cov_core = cb_entry.get("test_token_coverage_pct")
            cov_core_str = (
                f"{float(tok_cov_core):.2f}%" if tok_cov_core is not None else "-"
            )
            lines.append(
                f"\n  === 核心词表档位 |V| = {v_core} (全集词位覆盖率 = {cov_core_str}) ==="
            )
            subsets_map = cb_entry.get("subsets") or {}
            for sub_key, sub_title in subset_titles:
                sub_block = subsets_map.get(sub_key)
                if not isinstance(sub_block, Mapping):
                    continue
                sub_conds = _filter_main_latency_conditions(
                    sub_block.get("conditions") or {}
                )
                if not sub_conds:
                    continue
                seg_cnt = sub_block.get("segment_count", 0)
                lines.append(f"\n  >>> {sub_title} (Segments = {seg_cnt}) <<<")
                lines.extend(_format_s3_conditions_table_full(sub_conds))
                lines.extend(
                    _format_s3_revision_gains_block_full(
                        sub_block.get("revision_gains") or {}
                    )
                )

    full_core = report.get("full_vocab_core_subsets")
    if isinstance(full_core, Mapping) and full_core.get("subsets"):
        lines.append(
            "\n[表 3] 全词库 (Full Vocabulary) 下的高覆盖核心子集拆解 (100% 零 OOV / >=80% 高覆盖核心集)"
        )
        fc_header = (
            f"{'Subset':<26} | {'Segs':>5} | {'Condition':<18} | {'BLEU-1':>7} | "
            f"{'CorpBLEU':>8} | {'WordAcc':>7} | {'InVocAcc':>8} | {'Acc@5':>7} | "
            f"{'WER(%)':>7} | {'CER(%)':>7} | {'BERT-F1':>7}"
        )
        fc_sep = "-" * len(fc_header)
        lines.append(fc_sep)
        lines.append(fc_header)
        lines.append(fc_sep)
        fc_subsets = full_core.get("subsets") or {}
        for sub_key in (
            "all_segments",
            "core_subset_100pct",
            "core_subset_80pct",
        ):
            sb = fc_subsets.get(sub_key)
            if not isinstance(sb, Mapping):
                continue
            seg_cnt = sb.get("segment_count", 0)
            sb_conds = _filter_main_latency_conditions(sb.get("conditions") or {})
            for c_name, m in sb_conds.items():
                lines.append(
                    f"{sub_key:<26} | {int(seg_cnt):>5d} | {c_name:<18} | "
                    f"{_fmt_pct(m.get('bleu1_pct'), 7)} | "
                    f"{_fmt_pct(m.get('corpus_bleu1_pct'), 8)} | "
                    f"{_fmt_pct(m.get('word_acc_pct'), 7)} | "
                    f"{_fmt_pct(m.get('in_vocabulary_word_acc_pct'), 8)} | "
                    f"{_fmt_pct(m.get('word_acc_top5_pct'), 7)} | "
                    f"{_fmt_pct(m.get('wer_pct'), 7)} | "
                    f"{_fmt_pct(m.get('cer_pct'), 7)} | "
                    f"{_fmt_pct(m.get('bertscore_f1_pct'), 7)}"
                )
            lines.append(fc_sep)

    vocab_scaling = report.get("vocabulary_scaling")
    if isinstance(vocab_scaling, Mapping) and vocab_scaling:
        lines.append(
            "\n[表 4A] 多尺度候选词表缩放曲线 — 全测试句段 (All Segments，对标 Jayalath et al. 2025 Fig. 4)"
        )
        vs_header = (
            f"{'Scale |V|':<10} | {'TokCov%':>7} | {'S3 BLEU-1':>9} | {'K0 BLEU-1':>9} | "
            f"{'S0 BLEU-1':>9} | {'ShiftBLEU':>9} | {'Δ(C-A)BLEU':>10} | {'S3 InVocAcc':>11} | "
            f"{'K0 InVocAcc':>11} | {'S3 WordAcc':>10} | {'S3 CER(%)':>9} | {'S3 BERT-F1':>10}"
        )
        vs_sep = "-" * len(vs_header)
        lines.append(vs_sep)
        lines.append(vs_header)
        lines.append(vs_sep)

        ordered_scales = sorted(
            vocab_scaling.items(),
            key=lambda kv: int(
                kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
            )
            if isinstance(kv[1], Mapping)
            else 0,
        )
        for s_key, s_entry in ordered_scales:
            if not isinstance(s_entry, Mapping):
                continue
            all_c = (
                s_entry.get("conditions")
                or s_entry.get("all_segments")
                or {}
            )
            s3_m = all_c.get("S3_clean") or {}
            k0_m = all_c.get("K0") or {}
            s0_m = all_c.get("S0") or {}
            sh_m = all_c.get("S3_shifted_neg75s") or {}
            raw_gains = (
                s_entry.get("revision_gains")
                or s_entry.get("revision_gains_all_segments")
                or {}
            )
            gains_all = raw_gains.get("clean_C_minus_A_pp") or {}
            lines.append(
                f"{str(s_key):<10} | "
                f"{_fmt_pct(s_entry.get('test_token_coverage_pct'), 7)} | "
                f"{_fmt_pct(s3_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_pct(k0_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_pct(s0_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_pct(sh_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_signed_pct(gains_all.get('bleu1_pct'), 10)} | "
                f"{_fmt_pct(s3_m.get('in_vocabulary_word_acc_pct'), 11)} | "
                f"{_fmt_pct(k0_m.get('in_vocabulary_word_acc_pct'), 11)} | "
                f"{_fmt_pct(s3_m.get('word_acc_pct'), 10)} | "
                f"{_fmt_pct(s3_m.get('cer_pct'), 9)} | "
                f"{_fmt_pct(s3_m.get('bertscore_f1_pct'), 10)}"
            )
        lines.append(vs_sep)

        lines.append(
            "\n[表 4B] 多尺度候选词表缩放曲线 — 高覆盖核心子集 (100% Core [0% OOV] 与 >=80% Core Sentences)"
        )
        vc_header = (
            f"{'Scale |V|':<10} | {'100%Segs':>8} | {'100% S3 BLEU':>12} | {'100% S3 Acc':>11} | "
            f"{'>=80%Segs':>9} | {'>=80% S3 BLEU':>13} | {'>=80% K0 BLEU':>13} | "
            f"{'>=80% ShiftBLEU':>15} | {'>=80% S3 InVocAcc':>17} | {'>=80% S3 Acc@5':>14}"
        )
        vc_sep = "-" * len(vc_header)
        lines.append(vc_sep)
        lines.append(vc_header)
        lines.append(vc_sep)
        for s_key, s_entry in ordered_scales:
            if not isinstance(s_entry, Mapping):
                continue
            c100_block = s_entry.get("core_subset_100pct") or {}
            c100_conds = (
                c100_block.get("conditions")
                if isinstance(c100_block.get("conditions"), Mapping)
                else c100_block
            ) or {}
            c100 = c100_conds.get("S3_clean") or {}
            c80_block = s_entry.get("core_subset_80pct") or {}
            c80_map = (
                c80_block.get("conditions")
                if isinstance(c80_block.get("conditions"), Mapping)
                else c80_block
            ) or {}
            c80_s3 = c80_map.get("S3_clean") or {}
            c80_k0 = c80_map.get("K0") or {}
            c80_sh = c80_map.get("S3_shifted_neg75s") or {}
            lines.append(
                f"{str(s_key):<10} | "
                f"{int(s_entry.get('core_100pct_segment_count', 0)):>8d} | "
                f"{_fmt_pct(c100.get('bleu1_pct'), 12)} | "
                f"{_fmt_pct(c100.get('word_acc_pct'), 11)} | "
                f"{int(s_entry.get('core_80pct_segment_count', 0)):>9d} | "
                f"{_fmt_pct(c80_s3.get('bleu1_pct'), 13)} | "
                f"{_fmt_pct(c80_k0.get('bleu1_pct'), 13)} | "
                f"{_fmt_pct(c80_sh.get('bleu1_pct'), 15)} | "
                f"{_fmt_pct(c80_s3.get('in_vocabulary_word_acc_pct'), 17)} | "
                f"{_fmt_pct(c80_s3.get('word_acc_top5_pct'), 14)}"
            )
        lines.append(vc_sep)

    ms_scaling = report.get("multi_subject_scaling")
    if isinstance(ms_scaling, Mapping) and ms_scaling:
        lines.append(
            "\n[表 5] 跨被试多观测神经聚合曲线 (Multi-Subject Observation Scaling k ∈ {1, 2, 5, 10})"
        )
        ordered_ms = dict(
            sorted(
                ms_scaling.items(),
                key=lambda kv: int(
                    kv[1].get("k_observations", str(kv[0]).lstrip("kK=_") or 1)
                )
                if isinstance(kv[1], Mapping)
                else 1,
            )
        )
        lines.extend(_format_s3_conditions_table_full(ordered_ms))

        has_ms_by_vocab = any(
            isinstance(v, Mapping) and isinstance(v.get("by_vocabulary_size"), Mapping) and v.get("by_vocabulary_size")
            for v in ordered_ms.values()
        )
        if has_ms_by_vocab:
            lines.append(
                "\n[表 5B] 跨被试多观测神经聚合 × 核心词表规模 (k ∈ {1, 2, 5, 10} × |V| ∈ {50, 100, 250, Full})"
            )
            msv_hdr = (
                f"{'Scale |V|':<10} | {'Subjects':<8} | {'All BLEU1':>9} | {'All WAcc':>8} | "
                f"{'All InV%':>8} | {'Core80 BLEU':>11} | {'Core80 Acc':>10} | "
                f"{'100%Segs':>8} | {'Core100 BLEU':>12} | {'Core100 Acc':>11}"
            )
            msv_sep = "-" * len(msv_hdr)
            lines.append(msv_sep)
            lines.append(msv_hdr)
            lines.append(msv_sep)
            vocab_keys_seen: list[str] = []
            for m_entry in ordered_ms.values():
                if not isinstance(m_entry, Mapping):
                    continue
                by_v = m_entry.get("by_vocabulary_size") or {}
                if isinstance(by_v, Mapping):
                    for vk in by_v.keys():
                        if str(vk) not in vocab_keys_seen:
                            vocab_keys_seen.append(str(vk))
            vocab_keys_sorted = sorted(
                vocab_keys_seen,
                key=lambda s: int(str(s).lstrip("vV=_") or 0),
            )
            for vk in vocab_keys_sorted:
                for k_lbl, m_entry in ordered_ms.items():
                    if not isinstance(m_entry, Mapping):
                        continue
                    by_v = m_entry.get("by_vocabulary_size") or {}
                    v_blk = by_v.get(vk) if isinstance(by_v, Mapping) else None
                    if not isinstance(v_blk, Mapping):
                        continue
                    c80 = v_blk.get("core_subset_80pct") or {}
                    c100 = v_blk.get("core_subset_100pct") or {}
                    lines.append(
                        f"{vk:<10} | {str(k_lbl):<8} | "
                        f"{_fmt_pct(v_blk.get('bleu1_pct'), 9)} | "
                        f"{_fmt_pct(v_blk.get('word_acc_pct'), 8)} | "
                        f"{_fmt_pct(v_blk.get('in_vocabulary_word_acc_pct'), 8)} | "
                        f"{_fmt_pct(c80.get('bleu1_pct'), 11)} | "
                        f"{_fmt_pct(c80.get('word_acc_pct'), 10)} | "
                        f"{int(v_blk.get('core_100pct_segment_count', 0)):>8d} | "
                        f"{_fmt_pct(c100.get('bleu1_pct'), 12)} | "
                        f"{_fmt_pct(c100.get('word_acc_pct'), 11)}"
                    )
                lines.append(msv_sep)

    cg_stab = report.get("c_gated_online_stability")
    if isinstance(cg_stab, Mapping) and cg_stab:
        tau_val = cg_stab.get("recommended_tau_gate", 0.5)
        lines.append(
            f"\n[表 6] 在线屏幕防闪烁置信度门控稳定性分析 (Online Screen Anti-Flicker C-Gated | τ_gate = {tau_val})"
        )
        cg_conds: dict[str, Mapping[str, Any]] = {}
        if isinstance(cg_stab.get("S3_clean_ungated"), Mapping):
            cg_conds["Rolling-K2 (Ungated τ=0)"] = cg_stab["S3_clean_ungated"]
        if isinstance(cg_stab.get("S3_clean_c_gated"), Mapping):
            cg_conds[f"C-Gated (τ={tau_val})"] = cg_stab["S3_clean_c_gated"]
        if isinstance(cg_stab.get("S3_shifted_neg75s_c_gated"), Mapping):
            cg_conds[f"C-Gated Shift (τ={tau_val})"] = cg_stab["S3_shifted_neg75s_c_gated"]
        if cg_conds:
            lines.extend(_format_s3_conditions_table_full(cg_conds))

    lines.append("\n[导出产物与可视化图表路径]")
    for path_key in (
        "export_path",
        "calibration_path",
        "decoding_path",
        "evaluation_dir",
    ):
        if report.get(path_key):
            lines.append(f"  - {path_key:<18}: {report[path_key]}")
    figs = report.get("figures")
    if isinstance(figs, Mapping) and figs:
        for fig_k, fig_p in figs.items():
            lines.append(f"  - {fig_k:<28}: {fig_p}")
    lines.append(banner)
    return "\n".join(lines)


def print_sequence_decoding_report(
    report: Mapping[str, Any],
    *,
    verbose: bool = False,
) -> None:
    """Print S3 sequence decoding evaluation report (compact by default, full when verbose=True)."""
    print(format_sequence_decoding_report(report, verbose=verbose), flush=True)


# ============================================================================
# 6. Zero-Second Saved JSON Report Renderer
# ============================================================================


def _entry_needs_exploratory_pt_upgrade(entry_blk: Any) -> bool:
    if not isinstance(entry_blk, Mapping):
        return True
    c_ref = entry_blk.get("clean_reference")
    if not isinstance(c_ref, Mapping):
        return True
    if (
        "full_vocab_word_macro_top10" not in c_ref
        or "full_vocab_macro_top1" not in c_ref
    ):
        return True
    fv_wm = c_ref.get("full_vocab_word_macro_top10")
    fv_m = c_ref.get("full_vocab_macro_top10")
    if (
        fv_wm is not None
        and fv_m is not None
        and abs(float(fv_wm) - float(fv_m)) > 1e-6
    ):
        return True
    sh_agg = entry_blk.get("shifted_aggregate")
    if not isinstance(sh_agg, Mapping):
        return True
    if (
        "mean_full_vocab_word_macro_top10" not in sh_agg
        or "mean_full_vocab_macro_top1" not in sh_agg
    ):
        return True
    sh_wm = sh_agg.get("mean_full_vocab_word_macro_top10")
    sh_m = sh_agg.get("mean_full_vocab_macro_top10")
    if (
        sh_wm is not None
        and sh_m is not None
        and abs(float(sh_wm) - float(sh_m)) > 1e-6
    ):
        return True
    k0_blk = (c_ref.get("horizons") or {}).get("K0")
    if isinstance(k0_blk, Mapping) and k0_blk:
        if (
            "common_vocab50_macro_top1" not in k0_blk
            or "common_full_vocab_word_macro_top10" not in k0_blk
        ):
            return True
    return False


def _needs_exploratory_pt_upgrade(
    payload_dict: Mapping[str, Any],
    *,
    is_cms: bool = False,
) -> bool:
    """判断已保存的 exploratory 子空间去偏或共模扣除 JSON 是否缺失新增的全词表 Word-Macro / Top-1 或视界指标。"""
    if not isinstance(payload_dict, Mapping):
        return True
    if is_cms:
        sweep_list = (
            payload_dict.get("sweep")
            or payload_dict.get("subtraction_sweep")
            or []
        )
        if not isinstance(sweep_list, Sequence) or not sweep_list:
            return True
        for item in sweep_list:
            if _entry_needs_exploratory_pt_upgrade(item):
                return True
        if "baseline_raw_eta0" in payload_dict and _entry_needs_exploratory_pt_upgrade(
            payload_dict.get("baseline_raw_eta0")
        ):
            return True
        return False

    base_blk = payload_dict.get("raw_baseline")
    if _entry_needs_exploratory_pt_upgrade(base_blk):
        return True
    for item in payload_dict.get("full_rank_loo_baselines") or []:
        if _entry_needs_exploratory_pt_upgrade(item):
            return True
    for item in payload_dict.get("subspace_sweep") or []:
        if _entry_needs_exploratory_pt_upgrade(item):
            return True
    return False


def _auto_upgrade_seed_exploratory_files(cdir: Path, split_l: str) -> None:
    """自动检查并零开销升级单种子目录下的 subspace_coherence_debiasing_{split}.json 与 common_mode_subtraction_{split}.json。"""
    pt_cand = cdir / "exploratory" / f"subspace_encoded_cache_{split_l}.pt"
    sub_json_cand = (
        cdir / "exploratory" / f"subspace_coherence_debiasing_{split_l}.json"
    )
    cms_json_cand = (
        cdir / "exploratory" / f"common_mode_subtraction_{split_l}.json"
    )

    sub_raw: dict[str, Any] | None = None
    need_sub_upgrade = not sub_json_cand.is_file()
    if not need_sub_upgrade:
        try:
            loaded_sub = json.loads(sub_json_cand.read_text(encoding="utf-8"))
            if isinstance(loaded_sub, Mapping):
                sub_raw = dict(loaded_sub)
                need_sub_upgrade = _needs_exploratory_pt_upgrade(
                    sub_raw, is_cms=False
                )
            else:
                need_sub_upgrade = True
        except Exception:
            need_sub_upgrade = True

    if need_sub_upgrade and pt_cand.is_file():
        try:
            from braindecoding.evaluation.subspace_coherence_debiasing_evaluation import (
                evaluate_cached_pt_subspace_loo,
            )

            sub_raw = evaluate_cached_pt_subspace_loo(
                pt_cand,
                output_dir=cdir,
                plot=False,
                print_report=False,
                verbose=False,
            )
            need_sub_upgrade = False
        except Exception:
            pass

    need_cms_upgrade = not cms_json_cand.is_file()
    if not need_cms_upgrade:
        try:
            loaded_cms = json.loads(cms_json_cand.read_text(encoding="utf-8"))
            if isinstance(loaded_cms, Mapping):
                need_cms_upgrade = _needs_exploratory_pt_upgrade(
                    loaded_cms, is_cms=True
                )
            else:
                need_cms_upgrade = True
        except Exception:
            need_cms_upgrade = True

    if need_cms_upgrade:
        upgraded_cms = False
        if isinstance(sub_raw, Mapping) and not need_sub_upgrade:
            try:
                from braindecoding.evaluation.subspace_coherence_debiasing_evaluation import (
                    sync_common_mode_subtraction_from_subspace_report,
                )

                synced = sync_common_mode_subtraction_from_subspace_report(
                    sub_raw,
                    cdir,
                    plot=False,
                )
                if isinstance(synced, Mapping):
                    upgraded_cms = True
            except Exception:
                pass
        if not upgraded_cms and pt_cand.is_file():
            try:
                from braindecoding.evaluation.common_mode_subtraction_evaluation import (
                    evaluate_cached_pt_common_mode_subtraction,
                )

                evaluate_cached_pt_common_mode_subtraction(
                    pt_cand,
                    output_dir=cdir,
                    plot=False,
                    print_report=False,
                    verbose=False,
                )
            except Exception:
                pass


def _ensure_enriched_multi_seed_summary(
    summary: Mapping[str, Any],
    *,
    base_dir: Path | str | None = None,
    split: str | None = None,
    force_reaggregate: bool = False,
    control_method: str | Sequence[str] | None = None,
    control_metric: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Ensure a multi-seed summary has full aggregates and apply explicit control method/metric selection."""
    payload = dict(summary)
    try:
        from braindecoding.evaluation.word_retrieval_evaluation import (
            _repair_legacy_seed_evaluation_payload,
            apply_multi_seed_control_selection,
            summarize_multi_seed_evaluation,
        )
    except Exception:
        return payload

    is_multi_stage = bool(payload.get("stages_aggregate"))
    needs_reaggregation = bool(force_reaggregate)
    bdir = Path(str(base_dir)) if base_dir is not None else None

    # Safely determine split without letting a previously-corrupted payload["split"] override explicit split or filename
    ev_json_str = str(payload.get("evaluation_json", "")).lower()
    if split is not None and str(split).strip():
        split_l = str(split).strip().lower()
    elif ev_json_str.endswith("_test.json"):
        split_l = "test"
    elif ev_json_str.endswith("_val.json"):
        split_l = "val"
    elif bdir is not None and bdir.is_dir():
        raw_sp = str(payload.get("split", "")).strip().lower()
        if (
            raw_sp == "val"
            and not list(bdir.glob("seed-*/evaluation_val.json"))
            and list(bdir.glob("seed-*/evaluation_test.json"))
        ):
            split_l = "test"
        elif raw_sp in {"test", "val"}:
            split_l = raw_sp
        elif list(bdir.glob("seed-*/evaluation_test.json")) or (
            bdir / "multi_seed_evaluation_test.json"
        ).is_file():
            split_l = "test"
        else:
            split_l = "val"
    else:
        split_l = str(payload.get("split", "test")).strip().lower()
    payload["split"] = split_l

    def _method_agg_has_metric_discrepancy(m_blk: Any) -> bool:
        if not isinstance(m_blk, Mapping) or not m_blk.get("enabled", True):
            return False
        if (
            "mean_clean_full_vocab_word_macro_top10" not in m_blk
            or "mean_shifted_full_vocab_word_macro_top10" not in m_blk
            or "p_str_shifted_vs_chance_vocab50_macro_top10" not in m_blk
            or "p_str_gap_vocab50_macro_top10" not in m_blk
        ):
            return True
        hz_b = m_blk.get("horizons_aggregate")
        if isinstance(hz_b, Mapping) and isinstance(hz_b.get("K2_minus_K0"), Mapping):
            k20_b = hz_b["K2_minus_K0"]
            if (
                "p_str_shifted_common_vocab50_macro_top10" not in k20_b
                or "p_str_clean_common_vocab50_macro_top10" not in k20_b
                or "p_str_gap_common_vocab50_macro_top10" not in k20_b
            ):
                return True
        c_wm = m_blk.get("mean_clean_full_vocab_word_macro_top10")
        c_m = m_blk.get("mean_clean_full_vocab_macro_top10")
        if (
            c_wm is not None
            and c_m is not None
            and abs(float(c_wm) - float(c_m)) > 1e-6
        ):
            return True
        sh_wm = m_blk.get("mean_shifted_full_vocab_word_macro_top10")
        sh_m = m_blk.get("mean_shifted_full_vocab_macro_top10")
        if (
            sh_wm is not None
            and sh_m is not None
            and abs(float(sh_wm) - float(sh_m)) > 1e-6
        ):
            return True
        for ps in m_blk.get("per_seed") or []:
            if not isinstance(ps, Mapping):
                continue
            ps_c = ps.get("clean_reference")
            if isinstance(ps_c, Mapping):
                if "full_vocab_word_macro_top10" not in ps_c:
                    return True
                ps_c_wm = ps_c.get("full_vocab_word_macro_top10")
                ps_c_m = ps_c.get("full_vocab_macro_top10")
                if (
                    ps_c_wm is not None
                    and ps_c_m is not None
                    and abs(float(ps_c_wm) - float(ps_c_m)) > 1e-6
                ):
                    return True
            ps_sh = ps.get("shifted_aggregate")
            if isinstance(ps_sh, Mapping):
                if "mean_full_vocab_word_macro_top10" not in ps_sh:
                    return True
                ps_sh_wm = ps_sh.get("mean_full_vocab_word_macro_top10")
                ps_sh_m = ps_sh.get("mean_full_vocab_macro_top10")
                if (
                    ps_sh_wm is not None
                    and ps_sh_m is not None
                    and abs(float(ps_sh_wm) - float(ps_sh_m)) > 1e-6
                ):
                    return True
        return False

    def _stage_block_has_metric_discrepancy(st_blk: Mapping[str, Any]) -> bool:
        ling_b = st_blk.get("linguistic_breakdown_aggregate")
        if isinstance(ling_b, Mapping):
            cmf = ling_b.get("content_minus_function")
            if isinstance(cmf, Mapping) and "p_str_vocab50_word_macro_top10" not in cmf:
                return True
        tr_b = st_blk.get("revision_transitions_aggregate")
        if isinstance(tr_b, Mapping) and tr_b:
            k02 = tr_b.get("K0_to_K2")
            if isinstance(k02, Mapping):
                v50_tr = k02.get("vocab50_top10")
                if isinstance(v50_tr, Mapping) and "mcnemar_p_str" not in v50_tr:
                    return True
        hz_st = st_blk.get("revision_horizons_aggregate")
        if isinstance(hz_st, Mapping) and isinstance(hz_st.get("K2_minus_K0"), Mapping):
            if "p_str_common_vocab50_macro_top10" not in hz_st["K2_minus_K0"]:
                return True
        fv_agg_b = (st_blk.get("vocabulary_sweep_aggregate") or {}).get(
            "full_vocab"
        ) or {}
        nc_by_m = st_blk.get("negative_controls_by_method") or {}
        raw_m = (
            nc_by_m.get("raw")
            if isinstance(nc_by_m, Mapping)
            else None
        ) or st_blk.get("negative_controls_aggregate") or {}
        if isinstance(fv_agg_b, Mapping) and isinstance(raw_m, Mapping) and raw_m.get("enabled"):
            if (
                "mean_clean_full_vocab_word_macro_top10" not in raw_m
                or "p_str_shifted_vs_chance_vocab50_macro_top10" not in raw_m
            ):
                return True
            fv_c = fv_agg_b.get("mean_macro_top10")
            nc_c = raw_m.get("mean_clean_full_vocab_macro_top10")
            if (
                fv_c is not None
                and nc_c is not None
                and abs(float(fv_c) - float(nc_c)) > 1e-6
            ):
                return True
        if isinstance(nc_by_m, Mapping):
            for m_blk in nc_by_m.values():
                if _method_agg_has_metric_discrepancy(m_blk):
                    return True
        cms_agg = st_blk.get("common_mode_subtraction_aggregate") or {}
        if isinstance(cms_agg, Mapping) and cms_agg.get("enabled"):
            cms_methods = cms_agg.get("methods")
            if isinstance(cms_methods, Mapping):
                for m_blk in cms_methods.values():
                    if _method_agg_has_metric_discrepancy(m_blk):
                        return True
            elif isinstance(cms_methods, Sequence):
                for m_blk in cms_methods:
                    if _method_agg_has_metric_discrepancy(m_blk):
                        return True
        return False

    if not is_multi_stage:
        if (
            "negative_controls_by_method" not in payload
            or "negative_control_methods_aggregate" not in payload
            or "common_mode_subtraction_aggregate" not in payload
            or "subspace_debiasing_aggregate" not in payload
            or "linguistic_breakdown_aggregate" not in payload
            or "per_subject_summary_aggregate" not in payload
            or _stage_block_has_metric_discrepancy(payload)
        ):
            needs_reaggregation = True
        elif bdir is not None and bdir.is_dir():
            cms_enabled = bool((payload.get("common_mode_subtraction_aggregate") or {}).get("enabled"))
            sub_enabled = bool((payload.get("subspace_debiasing_aggregate") or {}).get("enabled"))
            if not cms_enabled and list(bdir.glob(f"seed-*/exploratory/common_mode_subtraction_{split_l}.json")):
                needs_reaggregation = True
            if not sub_enabled and (
                list(bdir.glob(f"seed-*/exploratory/subspace_coherence_debiasing_{split_l}.json"))
                or list(bdir.glob(f"seed-*/exploratory/subspace_encoded_cache_{split_l}.pt"))
            ):
                needs_reaggregation = True
    else:
        stages_agg = payload.get("stages_aggregate") or {}
        if isinstance(stages_agg, Mapping):
            for st_blk in stages_agg.values():
                if isinstance(st_blk, Mapping) and (
                    "negative_controls_by_method" not in st_blk
                    or "negative_control_methods_aggregate" not in st_blk
                    or _stage_block_has_metric_discrepancy(st_blk)
                ):
                    needs_reaggregation = True
                    break

    if needs_reaggregation:
        seed_results = list(payload.get("seed_results") or [])
        if bdir is not None and bdir.is_dir():
            seeds_l = list(payload.get("seeds") or [])
            fname = (
                f"multi_stage_evaluation_{split_l}.json"
                if is_multi_stage
                else f"evaluation_{split_l}.json"
            )
            candidate_seed_dirs: list[tuple[int, Path]] = []
            seen_dirs: set[Path] = set()
            for sd in seeds_l:
                for cdir in (bdir / f"seed-{int(sd):03d}", bdir / f"seed-{int(sd)}"):
                    if cdir.is_dir() and cdir.resolve() not in seen_dirs:
                        seen_dirs.add(cdir.resolve())
                        candidate_seed_dirs.append((int(sd), cdir))
            for cdir in sorted(bdir.glob("seed-*")):
                if cdir.is_dir() and cdir.resolve() not in seen_dirs:
                    tail = cdir.name.split("seed-", 1)[-1]
                    if tail.isdigit():
                        seen_dirs.add(cdir.resolve())
                        candidate_seed_dirs.append((int(tail), cdir))

            loaded_seeds: list[dict[str, Any]] = []
            for sd_num, cdir in candidate_seed_dirs:
                _auto_upgrade_seed_exploratory_files(cdir, split_l)

                f_cand = cdir / fname
                if f_cand.is_file():
                    try:
                        s_obj = json.loads(f_cand.read_text(encoding="utf-8"))
                        if isinstance(s_obj, Mapping):
                            s_dict = dict(s_obj)
                            s_dict.setdefault("output_dir", str(cdir))
                            s_dict.setdefault("seed", int(sd_num))
                            s_dict["split"] = split_l
                            if not is_multi_stage:
                                s_dict = _repair_legacy_seed_evaluation_payload(
                                    s_dict, split=split_l
                                )
                                if force_reaggregate:
                                    try:
                                        f_cand.write_text(
                                            json.dumps(
                                                s_dict,
                                                ensure_ascii=False,
                                                indent=2,
                                            )
                                            + "\n",
                                            encoding="utf-8",
                                        )
                                        export_word_evaluation_tables_and_markdown(
                                            s_dict, cdir, split=split_l
                                        )
                                    except Exception:
                                        pass
                            loaded_seeds.append(s_dict)
                    except Exception:
                        pass
            if loaded_seeds:
                seed_results = loaded_seeds
        if seed_results:
            try:
                re_agg = summarize_multi_seed_evaluation(
                    seed_results,
                    split=split_l,
                    output_dir=None,
                    print_report=False,
                    save_files=False,
                    control_method=control_method,
                    control_metric=control_metric,
                )
                re_agg["split"] = split_l
                if payload.get("output_dir") and not re_agg.get("output_dir"):
                    re_agg["output_dir"] = payload["output_dir"]
                elif bdir is not None and not re_agg.get("output_dir"):
                    re_agg["output_dir"] = str(bdir)
                re_agg["_was_reaggregated"] = True
                payload = re_agg
            except Exception:
                pass

    if is_multi_stage:
        stages_agg = dict(payload.get("stages_aggregate") or {})
        for st_k, st_v in list(stages_agg.items()):
            if isinstance(st_v, Mapping):
                st_copy = dict(st_v)
                st_copy["split"] = split_l
                apply_multi_seed_control_selection(
                    st_copy,
                    control_method=control_method,
                    control_metric=control_metric,
                )
                stages_agg[st_k] = st_copy
        payload["stages_aggregate"] = stages_agg
        if control_method is not None:
            payload["control_method"] = (
                ",".join(str(x) for x in control_method)
                if isinstance(control_method, (list, tuple))
                else str(control_method)
            )
        if control_metric is not None:
            payload["control_metric"] = (
                ",".join(str(x) for x in control_metric)
                if isinstance(control_metric, (list, tuple))
                else str(control_metric)
            )
    else:
        nc_agg = payload.get("negative_controls_aggregate") or {}
        if (
            control_method is not None
            or control_metric is not None
            or (
                isinstance(nc_agg, Mapping)
                and (
                    "active_control_method" not in nc_agg
                    or "multi_method_metric_comparison" not in nc_agg
                )
            )
        ):
            apply_multi_seed_control_selection(
                payload,
                control_method=control_method,
                control_metric=control_metric,
            )

    return payload


def _cleanup_accidental_val_reports_in_test_dir(st_dir: Path, split_l: str) -> None:
    """若当前为 test 升级且目录下不存在真实 seed-*/evaluation_val.json，清理此前因默认 split='val' 误生成的 _val 报表文件。"""
    if split_l != "test" or not st_dir.is_dir():
        return
    if list(st_dir.glob("seed-*/evaluation_val.json")) or list(
        st_dir.glob("seed-*/multi_stage_evaluation_val.json")
    ):
        return
    for stale_name in (
        "multi_seed_evaluation_val.json",
        "multi_seed_evaluation_report_val.md",
        "multi_seed_evaluation_report_val.txt",
        "multi_seed_stage_comparison_val.json",
        "multi_seed_stage_comparison_report_val.md",
        "multi_seed_stage_comparison_report_val.txt",
    ):
        stale_p = st_dir / stale_name
        if stale_p.is_file():
            try:
                stale_p.unlink()
            except OSError:
                pass


def upgrade_and_refresh_saved_multi_seed_reports(
    target_path: Path | str,
    *,
    split: str = "test",
    control_method: str | Sequence[str] | None = None,
    control_metric: str | Sequence[str] | None = None,
    verbose: bool = False,
    print_to_stdout: bool = True,
) -> list[dict[str, Any]]:
    """Zero-cost upgrade of legacy/saved multi-seed (or single-seed) evaluation JSONs from seed-*/evaluation_{split}.json & exploratory/*.json, refreshing .json, .txt, .md, and .csv."""
    from braindecoding.evaluation.word_retrieval_evaluation import (
        summarize_multi_seed_evaluation,
    )

    src = Path(str(target_path))
    if not src.exists():
        raise FileNotFoundError(f"Target path for multi-seed report upgrade not found: {src}")

    split_l = str(split or "test").strip().lower()
    stage_dirs: list[Path] = []
    direct_json_files: list[Path] = []

    if src.is_file():
        if src.name.startswith("multi_seed_"):
            direct_json_files.append(src)
        elif src.name.startswith("evaluation_") and src.parent.name.startswith("seed-"):
            stage_dirs.append(src.parent.parent)
        else:
            direct_json_files.append(src)
    else:
        # Check if src itself is a stage/subset directory containing multi_seed_*.json or seed-*/
        if list(src.glob(f"multi_seed_*_{split_l}.json")) or list(src.glob(f"seed-*/evaluation_{split_l}.json")):
            stage_dirs.append(src)
        else:
            found_dirs: set[Path] = set()
            for ms_f in sorted(src.rglob(f"multi_seed_*_{split_l}.json")):
                if ms_f.parent.resolve() not in found_dirs:
                    found_dirs.add(ms_f.parent.resolve())
                    stage_dirs.append(ms_f.parent)
            for sd_f in sorted(src.rglob(f"seed-*/evaluation_{split_l}.json")):
                st_dir = sd_f.parent.parent
                if st_dir.resolve() not in found_dirs:
                    found_dirs.add(st_dir.resolve())
                    stage_dirs.append(st_dir)

    upgraded_summaries: list[dict[str, Any]] = []

    for jf in direct_json_files:
        raw_obj = json.loads(jf.read_text(encoding="utf-8"))
        if not isinstance(raw_obj, Mapping):
            continue
        jf_split = (
            "test"
            if jf.name.lower().endswith("_test.json")
            else ("val" if jf.name.lower().endswith("_val.json") else split_l)
        )
        _cleanup_accidental_val_reports_in_test_dir(jf.parent, jf_split)
        enriched = _ensure_enriched_multi_seed_summary(
            raw_obj,
            base_dir=jf.parent,
            split=jf_split,
            force_reaggregate=True,
            control_method=control_method,
            control_metric=control_metric,
        )
        enriched["split"] = jf_split
        enriched.pop("_was_reaggregated", None)
        jf.write_text(json.dumps(enriched, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_multi_seed_evaluation_report(
            enriched,
            verbose=verbose,
            output_dir=jf.parent,
            print_to_stdout=print_to_stdout,
            control_method=control_method,
            control_metric=control_metric,
        )
        upgraded_summaries.append(enriched)

    for st_dir in stage_dirs:
        _cleanup_accidental_val_reports_in_test_dir(st_dir, split_l)
        ms_stage_file = st_dir / f"multi_seed_stage_comparison_{split_l}.json"
        ms_eval_file = st_dir / f"multi_seed_evaluation_{split_l}.json"
        if ms_stage_file.is_file():
            raw_obj = json.loads(ms_stage_file.read_text(encoding="utf-8"))
            enriched = _ensure_enriched_multi_seed_summary(
                raw_obj,
                base_dir=st_dir,
                split=split_l,
                force_reaggregate=True,
                control_method=control_method,
                control_metric=control_metric,
            )
            enriched["split"] = split_l
            enriched.pop("_was_reaggregated", None)
            ms_stage_file.write_text(
                json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print_multi_seed_evaluation_report(
                enriched,
                verbose=verbose,
                output_dir=st_dir,
                print_to_stdout=print_to_stdout,
                control_method=control_method,
                control_metric=control_metric,
            )
            upgraded_summaries.append(enriched)
            continue

        if ms_eval_file.is_file():
            raw_obj = json.loads(ms_eval_file.read_text(encoding="utf-8"))
            enriched = _ensure_enriched_multi_seed_summary(
                raw_obj,
                base_dir=st_dir,
                split=split_l,
                force_reaggregate=True,
                control_method=control_method,
                control_metric=control_metric,
            )
            enriched["split"] = split_l
            enriched.pop("_was_reaggregated", None)
            ms_eval_file.write_text(
                json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print_multi_seed_evaluation_report(
                enriched,
                verbose=verbose,
                output_dir=st_dir,
                print_to_stdout=print_to_stdout,
                control_method=control_method,
                control_metric=control_metric,
            )
            upgraded_summaries.append(enriched)
            continue

        # Fallback: build outer multi_seed_evaluation_{split}.json directly from seed-*/evaluation_{split}.json
        seed_Payloads: list[dict[str, Any]] = []
        for cdir in sorted(st_dir.glob("seed-*")):
            _auto_upgrade_seed_exploratory_files(cdir, split_l)
            f_cand = cdir / f"evaluation_{split_l}.json"
            if f_cand.is_file():
                try:
                    s_obj = json.loads(f_cand.read_text(encoding="utf-8"))
                    if isinstance(s_obj, Mapping):
                        s_dict = dict(s_obj)
                        s_dict.setdefault("output_dir", str(cdir))
                        s_dict["split"] = split_l
                        tail = cdir.name.split("seed-", 1)[-1]
                        if tail.isdigit():
                            s_dict.setdefault("seed", int(tail))
                        seed_Payloads.append(s_dict)
                except Exception:
                    pass
        if seed_Payloads:
            built = summarize_multi_seed_evaluation(
                seed_Payloads,
                split=split_l,
                output_dir=st_dir,
                print_report=False,
                control_method=control_method,
                control_metric=control_metric,
            )
            print_multi_seed_evaluation_report(
                built,
                verbose=verbose,
                output_dir=st_dir,
                print_to_stdout=print_to_stdout,
                control_method=control_method,
                control_metric=control_metric,
            )
            upgraded_summaries.append(built)

    return upgraded_summaries


def render_saved_evaluation_report(
    input_path: Path | str,
    *,
    verbose: bool = False,
    control_method: str | None = None,
    control_metric: str | None = None,
) -> str:
    """Detect and print formatted evaluation report from a saved JSON or directory in 0 seconds."""
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Evaluation report path not found: {src}")

    if src.is_dir():
        candidates = (
            list(sorted(src.glob("s3_evaluation_summary_*.json")))
            + list(sorted(src.glob("multi_seed_stage_comparison_*.json")))
            + list(sorted(src.glob("multi_seed_evaluation_*.json")))
            + list(sorted(src.glob("multi_stage_evaluation_*.json")))
            + list(sorted(src.glob("subspace_coherence_debiasing_*.json")))
            + list(sorted(src.glob("common_mode_subtraction_*.json")))
            + list(sorted(src.glob("evaluation_*.json")))
            + list(sorted(src.rglob("s3_evaluation_summary_*.json")))
            + list(sorted(src.rglob("multi_seed_stage_comparison_*.json")))
            + list(sorted(src.rglob("multi_seed_evaluation_*.json")))
            + list(sorted(src.rglob("multi_stage_evaluation_*.json")))
            + list(sorted(src.rglob("subspace_coherence_debiasing_*.json")))
            + list(sorted(src.rglob("common_mode_subtraction_*.json")))
            + list(sorted(src.rglob("evaluation_*.json")))
        )
        seen: set[Path] = set()
        unique_files: list[Path] = []
        for c in candidates:
            resolved = c.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique_files.append(c)
        if not unique_files:
            raise FileNotFoundError(
                f"No recognized evaluation JSON files found under directory: {src}"
            )
        rendered_blocks: list[str] = []
        for fpath in unique_files:
            rendered_blocks.append(
                render_saved_evaluation_report(
                    fpath,
                    verbose=verbose,
                    control_method=control_method,
                    control_metric=control_metric,
                )
            )
        return "\n\n".join(b for b in rendered_blocks if b)

    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object in {src}")
    payload = dict(payload)
    if src.name.lower().endswith("_test.json"):
        payload["split"] = "test"
    elif src.name.lower().endswith("_val.json"):
        payload["split"] = "val"

    status_str = str(payload.get("status", ""))
    protocol = str(payload.get("protocol", ""))
    exp_type = str(payload.get("experiment_type", ""))
    if (
        status_str
        in (
            "multi_seed_evaluation_completed",
            "multi_seed_stage_comparison_completed",
        )
        or ("per_seed" in payload and "vocabulary_sweep_aggregate" in payload)
        or "stages_aggregate" in payload
    ):
        return print_multi_seed_evaluation_report(
            payload,
            verbose=verbose,
            output_dir=src.parent,
            control_method=control_method,
            control_metric=control_metric,
        )

    if protocol == "sequence_decoding_s3_evaluation" or (
        "conditions" in payload and "vocabulary_scaling" in payload
    ):
        try:
            export_sequence_decoding_tables_and_markdown(payload, src.parent)
        except OSError:
            pass
        rendered = format_sequence_decoding_report(payload, verbose=verbose)
        print(rendered, flush=True)
        return rendered

    if "stages" in payload and isinstance(payload.get("stages"), Mapping):
        split = str(payload.get("split", "val"))
        return print_stage_comparison_report(
            payload, split=split, verbose=verbose, output_dir=src.parent
        )

    if exp_type == "exploratory_subspace_coherence_loo_debiasing" or (
        "subspace_and_coherence_diagnostics" in payload and "subspace_sweep" in payload
    ):
        base_out = src.parent.parent if src.parent.name == "exploratory" else src.parent
        split_s = str(payload.get("split", "test"))
        pt_candidate = (base_out / "exploratory" / f"subspace_encoded_cache_{split_s}.pt")
        if not pt_candidate.exists():
            pt_candidate = src.parent / f"subspace_encoded_cache_{split_s}.pt"
        if _needs_exploratory_pt_upgrade(payload, is_cms=False) and pt_candidate.exists():
            try:
                from braindecoding.evaluation.subspace_coherence_debiasing_evaluation import (
                    evaluate_cached_pt_subspace_loo,
                )

                payload = evaluate_cached_pt_subspace_loo(
                    pt_candidate,
                    output_dir=base_out,
                    plot=False,
                    print_report=False,
                    verbose=verbose,
                )
                return format_subspace_coherence_debiasing_report(payload, verbose=verbose)
            except Exception:
                pass
        return print_subspace_coherence_debiasing_report(
            payload, verbose=verbose, output_dir=base_out
        )

    if exp_type == "exploratory_leave_one_out_common_mode_subtraction" or (
        "eta_grid" in payload and "sweep" in payload
    ):
        base_out = src.parent.parent if src.parent.name == "exploratory" else src.parent
        split_s = str(payload.get("split", "test"))
        pt_candidate = (base_out / "exploratory" / f"subspace_encoded_cache_{split_s}.pt")
        if not pt_candidate.exists():
            pt_candidate = src.parent / f"subspace_encoded_cache_{split_s}.pt"
        if _needs_exploratory_pt_upgrade(payload, is_cms=True) and pt_candidate.exists():
            try:
                from braindecoding.evaluation.subspace_coherence_debiasing_evaluation import (
                    evaluate_cached_pt_common_mode_subtraction,
                )

                payload = evaluate_cached_pt_common_mode_subtraction(
                    pt_candidate,
                    output_dir=base_out,
                    plot=False,
                    print_report=False,
                    verbose=verbose,
                )
                return format_common_mode_subtraction_report(payload, verbose=verbose)
            except Exception:
                pass
        return print_common_mode_subtraction_report(
            payload, verbose=verbose, output_dir=base_out
        )

    if (
        "metrics" in payload
        or "vocabularies" in payload
        or "full_vocabulary_evaluation" in payload
    ):
        return print_word_evaluation_report(
            payload, verbose=verbose, output_dir=src.parent
        )

    raise ValueError(f"Unrecognized evaluation JSON schema in {src}")


def _fmt_mean_sd_pct(
    mean_val: object,
    std_val: object,
    width: int = 12,
    *,
    decimals: int = 2,
) -> str:
    """Format a 0..1 ratio mean ± sd as percentage string (e.g. '45.20±0.51%')."""
    if mean_val is None or mean_val == "":
        return "-".rjust(width)
    try:
        m = float(mean_val) * 100.0
        s = float(std_val or 0.0) * 100.0
        txt = f"{m:.{decimals}f}±{s:.{decimals}f}%"
        if len(txt) > width and decimals > 1:
            txt = f"{m:.1f}±{s:.1f}%"
        return txt.rjust(width)
    except (TypeError, ValueError):
        return "-".rjust(width)


def _fmt_signed_mean_sd_pct(
    mean_val: object,
    std_val: object,
    width: int = 12,
    *,
    decimals: int = 2,
) -> str:
    """Format a 0..1 signed delta mean ± sd as percentage points (e.g. '+5.12±0.34%')."""
    if mean_val is None or mean_val == "":
        return "-".rjust(width)
    try:
        m = float(mean_val) * 100.0
        s = float(std_val or 0.0) * 100.0
        txt = f"{m:+.{decimals}f}±{s:.{decimals}f}%"
        if len(txt) > width and decimals > 1:
            txt = f"{m:+.1f}±{s:.1f}%"
        return txt.rjust(width)
    except (TypeError, ValueError):
        return "-".rjust(width)


def _fmt_mean_sd_num(
    mean_val: object,
    std_val: object,
    width: int = 12,
    *,
    decimals: int = 1,
    signed: bool = False,
) -> str:
    """Format a scalar mean ± sd (e.g. '16.5±1.2' or '+24.5±3.1')."""
    if mean_val is None or mean_val == "":
        return "-".rjust(width)
    try:
        m = float(mean_val)
        s = float(std_val or 0.0)
        fmt_m = f"{m:+.{decimals}f}" if signed else f"{m:.{decimals}f}"
        txt = f"{fmt_m}±{s:.{decimals}f}"
        return txt.rjust(width)
    except (TypeError, ValueError):
        return "-".rjust(width)


def format_multi_seed_evaluation_report(
    summary: Mapping[str, Any],
    *,
    verbose: bool = False,
    control_method: str | None = None,
    control_metric: str | None = None,
) -> str:
    """Format a multi-seed evaluation summary (single-stage or multi-stage)."""
    summary = _ensure_enriched_multi_seed_summary(
        summary,
        base_dir=summary.get("output_dir"),
        control_method=control_method,
        control_metric=control_metric,
    )
    banner = "=" * REPORT_WIDTH
    rule = "-" * REPORT_WIDTH
    dataset = str(summary.get("dataset", "-"))
    split = str(summary.get("split", "val")).upper()
    seeds = list(summary.get("seeds") or [])
    seeds_str = ",".join(str(s) for s in seeds) if seeds else "-"
    subset = summary.get("subset") or {}
    subset_tag = (
        subset.get("subset_tag", "all")
        if isinstance(subset, Mapping)
        else str(subset)
    )
    win_sec = float(summary.get("window_seconds", 1.0))
    exp_tag = str(summary.get("exp_tag", "") or "default")

    # Multi-stage multi-seed comparison schema
    stages_agg = summary.get("stages_aggregate")
    if isinstance(stages_agg, Mapping) and stages_agg:
        lines: list[str] = [
            banner,
            f"Multi-Seed Stage Comparison | {dataset} | Seeds: {seeds_str} (n={len(seeds)})"[:REPORT_WIDTH],
            f"Split: {split} | Subset: {subset_tag} | Win: {win_sec:.2f}s | Tag: {exp_tag}"[:REPORT_WIDTH],
            banner,
            "[1] Multi-Stage Retrieval Summary Across Seeds (Mean ± SD)",
            rule,
            (
                f"{'Stage':<22} | {'50w MacTop10':>13} | {'50w MicTop10':>13} | "
                f"{'50w MacTop1':>13} | {'FullVoc Top10':>13}"
            ),
            rule,
        ]
        for st_key in ("word", "context", "revision", "prefix_revision"):
            if st_key not in stages_agg:
                continue
            st_block = stages_agg[st_key]
            lbl = STAGE_SHORT_LABELS.get(st_key, STAGE_DISPLAY_NAMES.get(st_key, st_key))[:22]
            sw_agg = st_block.get("vocabulary_sweep_aggregate") or {}
            n50_a = sw_agg.get("N50") or {}
            fv_a = sw_agg.get("full_vocab") or {}
            lines.append(
                f"{lbl:<22} | "
                f"{_fmt_mean_sd_pct(n50_a.get('mean_macro_top10'), n50_a.get('std_macro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(n50_a.get('mean_micro_top10'), n50_a.get('std_micro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(n50_a.get('mean_macro_top1'), n50_a.get('std_macro_top1'), 13)} | "
                f"{_fmt_mean_sd_pct(fv_a.get('mean_macro_top10'), fv_a.get('std_macro_top10'), 13)}"
            )
            hz_agg = st_block.get("revision_horizons_aggregate") or {}
            if hz_agg:
                for h_key, h_lbl in (
                    ("K0", "  ├─ K0 (0.0s Onset)"),
                    ("K1", "  ├─ K1 (+1.0s Delay)"),
                    ("K2", "  └─ K2 (+2.0s Delay)"),
                ):
                    if h_key not in hz_agg:
                        continue
                    hb = hz_agg[h_key]
                    lines.append(
                        f"{h_lbl:<22} | "
                        f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_macro_top10'), hb.get('std_common_vocab50_macro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_micro_top10'), hb.get('std_common_vocab50_micro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_macro_top1'), hb.get('std_common_vocab50_macro_top1'), 13)} | "
                        f"{_fmt_mean_sd_pct(hb.get('mean_common_full_vocab_macro_top10'), hb.get('std_common_full_vocab_macro_top10'), 13)}"
                    )
        lines.append(rule)
        if verbose:
            for st_key in ("word", "context", "revision", "prefix_revision"):
                if st_key in stages_agg:
                    lines.append("")
                    lines.append(
                        format_multi_seed_evaluation_report(
                            stages_agg[st_key],
                            verbose=True,
                            control_method=control_method,
                            control_metric=control_metric,
                        )
                    )
        else:
            lines.append("Full multi-seed tables (.txt/.md/.csv) saved; pass --verbose for full breakdown.")
        lines.append(banner)
        return "\n".join(lines)

    # Single-stage multi-seed schema
    stage = str(summary.get("stage", "word"))
    stage_label = STAGE_DISPLAY_NAMES.get(stage, stage)
    per_seed = list(summary.get("per_seed") or [])
    sweep_agg = dict(summary.get("vocabulary_sweep_aggregate") or {})
    n50_agg = sweep_agg.get("N50") or {}
    fv_agg = sweep_agg.get("full_vocab") or {}
    hz_agg = summary.get("revision_horizons_aggregate") or {}
    nc_agg = summary.get("negative_controls_aggregate") or {}

    if not verbose:
        # Compact Single-Screen Executive Summary (<= 86 cols)
        lines = [
            banner,
            f"Multi-Seed Word Evaluation | {dataset} | {stage_label}"[:REPORT_WIDTH],
            f"Split: {split} | Subset: {subset_tag} | Seeds: {seeds_str} (n={len(per_seed)}) | Win: {win_sec:.2f}s"[:REPORT_WIDTH],
            banner,
            "[1] Per-Seed & Multi-Seed Aggregate (50-Word Bank & Full Training Vocab)",
            rule,
            (
                f"{'Seed / Summary':<15} | {'50w MacTop10':>12} | {'50w MicTop10':>12} | "
                f"{'50w MacTop1':>11} | {'FullTop10':>11} | {'50w MRR':>10}"
            ),
            rule,
        ]
        for row in per_seed:
            sd = int(row.get("seed", 0))
            ep = row.get("epoch", "-")
            seed_lbl = f"seed-{sd:03d} (e{ep})"[:15]
            lines.append(
                f"{seed_lbl:<15} | "
                f"{_fmt_ratio_pct(row.get('vocab50_macro_top10'), 12)} | "
                f"{_fmt_ratio_pct(row.get('vocab50_micro_top10'), 12)} | "
                f"{_fmt_ratio_pct(row.get('vocab50_macro_top1'), 11)} | "
                f"{_fmt_ratio_pct(row.get('full_vocab_macro_top10'), 11)} | "
                f"{_fmt_ratio_pct(row.get('vocab50_mrr'), 10)}"
            )
        if per_seed:
            lines.append(rule)
            mean_lbl = f"Mean±SD (n={len(per_seed)})"[:15]
            lines.append(
                f"{mean_lbl:<15} | "
                f"{_fmt_mean_sd_pct(n50_agg.get('mean_macro_top10'), n50_agg.get('std_macro_top10'), 12)} | "
                f"{_fmt_mean_sd_pct(n50_agg.get('mean_micro_top10'), n50_agg.get('std_micro_top10'), 12)} | "
                f"{_fmt_mean_sd_pct(n50_agg.get('mean_macro_top1'), n50_agg.get('std_macro_top1'), 11)} | "
                f"{_fmt_mean_sd_pct(fv_agg.get('mean_macro_top10'), fv_agg.get('std_macro_top10'), 11)} | "
                f"{_fmt_mean_sd_pct(n50_agg.get('mean_mrr'), n50_agg.get('std_mrr'), 10, decimals=1)}"
            )
        lines.append(rule)

        if sweep_agg and len(sweep_agg) > 1:
            lines.append("\n[2] Vocabulary Scale Sweep Across Seeds (Mean ± SD)")
            lines.append(rule)
            lines.append(
                f"{'Vocabulary Scale':<18} | {'MacTop10 (M±SD)':>15} | {'MicTop10 (M±SD)':>15} | "
                f"{'MacTop1 (M±SD)':>14} | {'MRR (M±SD)':>12}"
            )
            lines.append(rule)
            for sk in ("N50", "N50_strict", "full_vocab"):
                if sk not in sweep_agg:
                    continue
                sb = sweep_agg[sk]
                lbl = str(sb.get("label", sk))[:18]
                lines.append(
                    f"{lbl:<18} | "
                    f"{_fmt_mean_sd_pct(sb.get('mean_macro_top10'), sb.get('std_macro_top10'), 15)} | "
                    f"{_fmt_mean_sd_pct(sb.get('mean_micro_top10'), sb.get('std_micro_top10'), 15)} | "
                    f"{_fmt_mean_sd_pct(sb.get('mean_macro_top1'), sb.get('std_macro_top1'), 14)} | "
                    f"{_fmt_mean_sd_pct(sb.get('mean_mrr'), sb.get('std_mrr'), 12)}"
                )
            lines.append(rule)

        if isinstance(hz_agg, Mapping) and hz_agg:
            lines.append("\n[3] Rolling Revision Horizons Across Seeds (Mean ± SD)")
            lines.append(rule)
            lines.append(
                f"{'Horizon':<22} | {'50w MacTop10':>13} | {'50w MicTop10':>13} | "
                f"{'50w MacTop1':>13} | {'FullVoc Top10':>13}"
            )
            lines.append(rule)
            for h_key in ("K0", "K1", "K2"):
                if h_key not in hz_agg:
                    continue
                hb = hz_agg[h_key]
                lbl = str(hb.get("label", h_key))[:22]
                lines.append(
                    f"{lbl:<22} | "
                    f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_macro_top10'), hb.get('std_common_vocab50_macro_top10'), 13)} | "
                    f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_micro_top10'), hb.get('std_common_vocab50_micro_top10'), 13)} | "
                    f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_macro_top1'), hb.get('std_common_vocab50_macro_top1'), 13)} | "
                    f"{_fmt_mean_sd_pct(hb.get('mean_common_full_vocab_macro_top10'), hb.get('std_common_full_vocab_macro_top10'), 13)}"
                )
            for g_key, g_lbl in (
                ("K1_minus_K0", "Step-1 Gain (K1-K0)"),
                ("K2_minus_K0", "Revision Gain (K2-K0)"),
            ):
                if g_key in hz_agg:
                    gb = hz_agg[g_key]
                    lines.append(
                        f"{g_lbl:<22} | "
                        f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_macro_top10'), gb.get('std_common_vocab50_macro_top10'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_micro_top10'), gb.get('std_common_vocab50_micro_top10'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_macro_top1'), gb.get('std_common_vocab50_macro_top1'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_full_vocab_macro_top10'), gb.get('std_common_full_vocab_macro_top10'), 13)}"
                    )
            lines.append(rule)

        if isinstance(nc_agg, Mapping) and nc_agg.get("enabled"):
            act_method_key = str(nc_agg.get("active_control_method", "raw"))
            act_method_lbl = str(nc_agg.get("active_control_method_label", "Raw Un-Debiased Temporal Shift (eta=0.0)"))
            prim_metric_key = str(nc_agg.get("primary_control_metric", "vocab50_macro_top10"))
            prim_metric_lbl = str(nc_agg.get("primary_control_metric_label", "50-Word Word-Macro Top-10"))
            sel_methods = list(nc_agg.get("selected_control_methods") or [act_method_key])
            sel_metrics_info = list(nc_agg.get("selected_control_metrics") or [])
            sel_metric_ids = [
                str(m.get("metric_key") or m.get("metric_id"))
                if isinstance(m, Mapping)
                else str(m)
                for m in sel_metrics_info
                if m
            ] or [prim_metric_key]
            mm_comp = list(nc_agg.get("multi_method_metric_comparison") or [])
            is_pct = bool(nc_agg.get("primary_metric_is_percentage", True))
            p_c = (
                _fmt_mean_sd_pct(nc_agg.get("primary_metric_clean_mean"), nc_agg.get("primary_metric_clean_std"), 1)
                if is_pct
                else _fmt_mean_sd_num(nc_agg.get("primary_metric_clean_mean"), nc_agg.get("primary_metric_clean_std"), 1)
            ).strip()
            p_s = (
                _fmt_mean_sd_pct(nc_agg.get("primary_metric_shifted_mean"), nc_agg.get("primary_metric_shifted_std"), 1)
                if is_pct
                else _fmt_mean_sd_num(nc_agg.get("primary_metric_shifted_mean"), nc_agg.get("primary_metric_shifted_std"), 1)
            ).strip()
            p_g = (
                _fmt_signed_mean_sd_pct(nc_agg.get("primary_metric_gap_mean"), nc_agg.get("primary_metric_gap_std"), 1)
                if is_pct
                else _fmt_mean_sd_num(nc_agg.get("primary_metric_gap_mean"), nc_agg.get("primary_metric_gap_std"), 1, signed=True)
            ).strip()

            lines.append(f"\n[4] Negative Control Multi-Seed Summary ({split}, Mean ± SD)")
            if len(sel_methods) > 1 or len(sel_metric_ids) > 1:
                lines.append(f"Methods: {', '.join(sel_methods)}"[:REPORT_WIDTH])
                lines.append(f"Metrics: {', '.join(sel_metric_ids)}"[:REPORT_WIDTH])
            lines.append(f"Method : [{act_method_key}] {act_method_lbl}"[:REPORT_WIDTH])
            lines.append(
                f"Metric : [{prim_metric_key}] Clean {p_c} | Shft {p_s} | Gap {p_g}"[:REPORT_WIDTH]
            )
            if len(mm_comp) > 1:
                lines.append(rule)
                lines.append(
                    f"{'Method':<22} | {'Metric':<18} | {'Clean (0s)':>12} | {'Shft Mean':>12} | {'Net Gap':>12}"
                )
                lines.append(rule)
                for mm_r in mm_comp:
                    if not isinstance(mm_r, Mapping):
                        continue
                    m_id_s = str(mm_r.get("method_id", "-"))[:22]
                    mt_id_s = str(mm_r.get("metric_key") or mm_r.get("metric_id") or "-")[:18]
                    m_is_pct = bool(mm_r.get("is_ratio", mm_r.get("is_percentage", True)))
                    c_m = mm_r.get("mean_clean", mm_r.get("clean_mean"))
                    c_sd = mm_r.get("std_clean", mm_r.get("clean_std"))
                    s_m = mm_r.get("mean_shifted", mm_r.get("shifted_mean"))
                    s_sd = mm_r.get("std_shifted", mm_r.get("shifted_std"))
                    g_m = mm_r.get("mean_gap", mm_r.get("gap_mean"))
                    g_sd = mm_r.get("std_gap", mm_r.get("gap_std"))
                    mc_s = (
                        _fmt_mean_sd_pct(c_m, c_sd, 12)
                        if m_is_pct
                        else _fmt_mean_sd_num(c_m, c_sd, 12)
                    )
                    ms_s = (
                        _fmt_mean_sd_pct(s_m, s_sd, 12)
                        if m_is_pct
                        else _fmt_mean_sd_num(s_m, s_sd, 12)
                    )
                    mg_s = (
                        _fmt_signed_mean_sd_pct(g_m, g_sd, 12)
                        if m_is_pct
                        else _fmt_mean_sd_num(g_m, g_sd, 12, signed=True)
                    )
                    lines.append(f"{m_id_s:<22} | {mt_id_s:<18} | {mc_s} | {ms_s} | {mg_s}")
            lines.append(rule)
            lines.append(
                f"{'Condition':<22} | {'50w MacTop10':>13} | {'50w MicTop10':>13} | "
                f"{'50w MacTop1':>13} | {'FullVoc Top10':>13}"
            )
            lines.append(rule)
            lines.append(
                f"{'Clean (0s)':<22} | "
                f"{_fmt_mean_sd_pct(nc_agg.get('mean_clean_vocab50_macro_top10'), nc_agg.get('std_clean_vocab50_macro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(nc_agg.get('mean_clean_vocab50_micro_top10'), nc_agg.get('std_clean_vocab50_micro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(nc_agg.get('mean_clean_vocab50_macro_top1'), nc_agg.get('std_clean_vocab50_macro_top1'), 13)} | "
                f"{_fmt_mean_sd_pct(nc_agg.get('mean_clean_full_vocab_macro_top10'), nc_agg.get('std_clean_full_vocab_macro_top10'), 13)}"
            )
            lines.append(
                f"{'Shifted Mean':<22} | "
                f"{_fmt_mean_sd_pct(nc_agg.get('mean_shifted_vocab50_macro_top10'), nc_agg.get('std_shifted_vocab50_macro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(nc_agg.get('mean_shifted_vocab50_micro_top10'), nc_agg.get('std_shifted_vocab50_micro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(nc_agg.get('mean_shifted_vocab50_macro_top1'), nc_agg.get('std_shifted_vocab50_macro_top1'), 13)} | "
                f"{_fmt_mean_sd_pct(nc_agg.get('mean_shifted_full_vocab_macro_top10'), nc_agg.get('std_shifted_full_vocab_macro_top10'), 13)}"
            )
            lines.append(
                f"{'Net Gap (Clean-Shft)':<22} | "
                f"{_fmt_signed_mean_sd_pct(nc_agg.get('mean_gap_vocab50_macro_top10'), nc_agg.get('std_gap_vocab50_macro_top10'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(nc_agg.get('mean_gap_vocab50_micro_top10'), nc_agg.get('std_gap_vocab50_micro_top10'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(nc_agg.get('mean_gap_vocab50_macro_top1'), nc_agg.get('std_gap_vocab50_macro_top1'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(nc_agg.get('mean_gap_full_vocab_macro_top10'), nc_agg.get('std_gap_full_vocab_macro_top10'), 13)}"
            )
            nc_hz = nc_agg.get("horizons_aggregate") or {}
            if isinstance(nc_hz, Mapping) and nc_hz:
                for hk, hlbl in (
                    ("K0", "  ├─ K0 Net Gap"),
                    ("K1", "  ├─ K1 Net Gap"),
                    ("K2", "  └─ K2 Net Gap"),
                ):
                    if hk not in nc_hz:
                        continue
                    h_gap = (nc_hz[hk].get("net_gap") or {}) if isinstance(nc_hz[hk], Mapping) else {}
                    lines.append(
                        f"{hlbl:<22} | "
                        f"{_fmt_signed_mean_sd_pct(h_gap.get('mean_common_vocab50_macro_top10'), h_gap.get('std_common_vocab50_macro_top10'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(h_gap.get('mean_common_vocab50_micro_top10'), h_gap.get('std_common_vocab50_micro_top10'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(h_gap.get('mean_common_vocab50_macro_top1'), h_gap.get('std_common_vocab50_macro_top1'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(h_gap.get('mean_common_full_vocab_macro_top10'), h_gap.get('std_common_full_vocab_macro_top10'), 13)}"
                    )
            lines.append(rule)

        lines.append("Full multi-seed tables (.txt/.md/.csv) saved; pass --verbose for full breakdown.")
        lines.append(banner)
        return "\n".join(lines)

    # =========================================================================
    # Full Breakdown Mode (verbose=True -> written to multi_seed_evaluation_report_{split}.txt)
    # =========================================================================
    wide_banner = "=" * 136
    wide_rule = "-" * 136
    lines = [
        wide_banner,
        f"Multi-Seed Word-Level Retrieval Evaluation Report (Full Breakdown) | {dataset} | {stage_label}",
        (
            f"Split: {split} | Subset: {subset_tag} | Seeds: {seeds_str} (n={len(per_seed)}) | "
            f"Window: {win_sec:.2f}s | Exp Tag: {exp_tag}"
        ),
        wide_banner,
        "[1A] Per-Seed & Multi-Seed Aggregate — 50-Word Frozen Bank (N50, Word-Macro vs Event-Micro vs Subject-Macro)",
        wide_rule,
        (
            f"{'Seed / Summary':<18} | {'Queries':>7} | {'50w MacTop10':>13} | {'50w MicTop10':>13} | "
            f"{'50w SubjTop10':>13} | {'50w MacTop5':>12} | {'50w MacTop1':>12} | {'50w MicTop1':>12} | "
            f"{'50w MRR':>11} | {'50w MedRank':>11}"
        ),
        wide_rule,
    ]
    for row in per_seed:
        sd = int(row.get("seed", 0))
        ep = row.get("epoch", "-")
        seed_lbl = f"seed-{sd:03d} (e{ep})"[:18]
        q_cnt = int(row.get("vocab50_query_count", n50_agg.get("query_count", 0)))
        lines.append(
            f"{seed_lbl:<18} | {q_cnt:>7d} | "
            f"{_fmt_ratio_pct(row.get('vocab50_macro_top10'), 13)} | "
            f"{_fmt_ratio_pct(row.get('vocab50_micro_top10'), 13)} | "
            f"{_fmt_ratio_pct(row.get('vocab50_subject_macro_top10', row.get('vocab50_macro_top10')), 13)} | "
            f"{_fmt_ratio_pct(row.get('vocab50_macro_top5'), 12)} | "
            f"{_fmt_ratio_pct(row.get('vocab50_macro_top1'), 12)} | "
            f"{_fmt_ratio_pct(row.get('vocab50_micro_top1'), 12)} | "
            f"{_fmt_ratio_pct(row.get('vocab50_mrr'), 11)} | "
            f"{_fmt_float(row.get('vocab50_median_rank'), 11, 1)}"
        )
    if per_seed:
        lines.append(wide_rule)
        mean_lbl = f"Mean±SD (n={len(per_seed)})"[:18]
        q_cnt = int(n50_agg.get("query_count", 0))
        lines.append(
            f"{mean_lbl:<18} | {q_cnt:>7d} | "
            f"{_fmt_mean_sd_pct(n50_agg.get('mean_macro_top10'), n50_agg.get('std_macro_top10'), 13)} | "
            f"{_fmt_mean_sd_pct(n50_agg.get('mean_micro_top10'), n50_agg.get('std_micro_top10'), 13)} | "
            f"{_fmt_mean_sd_pct(n50_agg.get('mean_subject_macro_top10'), n50_agg.get('std_subject_macro_top10'), 13)} | "
            f"{_fmt_mean_sd_pct(n50_agg.get('mean_macro_top5'), n50_agg.get('std_macro_top5'), 12)} | "
            f"{_fmt_mean_sd_pct(n50_agg.get('mean_macro_top1'), n50_agg.get('std_macro_top1'), 12)} | "
            f"{_fmt_mean_sd_pct(n50_agg.get('mean_micro_top1'), n50_agg.get('std_micro_top1'), 12)} | "
            f"{_fmt_mean_sd_pct(n50_agg.get('mean_mrr'), n50_agg.get('std_mrr'), 11)} | "
            f"{_fmt_mean_sd_num(n50_agg.get('mean_median_rank'), n50_agg.get('std_median_rank'), 11, decimals=1)}"
        )
    lines.append(wide_rule)

    # [1B] Full Training Vocabulary Per-Seed & Mean±SD
    lines.extend(
        [
            "",
            "[1B] Per-Seed & Multi-Seed Aggregate — Full Training Vocabulary (OOV Queries Retained as Rank |V|+1)",
            wide_rule,
            (
                f"{'Seed / Summary':<18} | {'Vocab|V|':>8} | {'Queries':>7} | {'OOV':>5} | "
                f"{'Full MacTop10':>13} | {'Full MicTop10':>13} | {'Full SubjTop10':>14} | "
                f"{'Full MacTop1':>12} | {'Full MicTop1':>12} | {'Full MRR':>11} | {'Full MedRank':>12}"
            ),
            wide_rule,
        ]
    )
    for row in per_seed:
        sd = int(row.get("seed", 0))
        ep = row.get("epoch", "-")
        seed_lbl = f"seed-{sd:03d} (e{ep})"[:18]
        v_sz = int(row.get("full_vocab_size", fv_agg.get("vocabulary_size", 0)))
        q_cnt = int(row.get("full_vocab_query_count", fv_agg.get("query_count", 0)))
        oov_cnt = int(row.get("full_vocab_oov_queries", fv_agg.get("oov_queries", 0)))
        lines.append(
            f"{seed_lbl:<18} | {v_sz:>8d} | {q_cnt:>7d} | {oov_cnt:>5d} | "
            f"{_fmt_ratio_pct(row.get('full_vocab_macro_top10'), 13)} | "
            f"{_fmt_ratio_pct(row.get('full_vocab_micro_top10'), 13)} | "
            f"{_fmt_ratio_pct(row.get('full_vocab_subject_macro_top10', row.get('full_vocab_macro_top10')), 14)} | "
            f"{_fmt_ratio_pct(row.get('full_vocab_macro_top1'), 12)} | "
            f"{_fmt_ratio_pct(row.get('full_vocab_micro_top1'), 12)} | "
            f"{_fmt_ratio_pct(row.get('full_vocab_mrr'), 11)} | "
            f"{_fmt_float(row.get('full_vocab_median_rank'), 12, 1)}"
        )
    if per_seed:
        lines.append(wide_rule)
        mean_lbl = f"Mean±SD (n={len(per_seed)})"[:18]
        v_sz = int(fv_agg.get("vocabulary_size", 0))
        q_cnt = int(fv_agg.get("query_count", 0))
        oov_cnt = int(fv_agg.get("oov_queries", 0))
        lines.append(
            f"{mean_lbl:<18} | {v_sz:>8d} | {q_cnt:>7d} | {oov_cnt:>5d} | "
            f"{_fmt_mean_sd_pct(fv_agg.get('mean_macro_top10'), fv_agg.get('std_macro_top10'), 13)} | "
            f"{_fmt_mean_sd_pct(fv_agg.get('mean_micro_top10'), fv_agg.get('std_micro_top10'), 13)} | "
            f"{_fmt_mean_sd_pct(fv_agg.get('mean_subject_macro_top10'), fv_agg.get('std_subject_macro_top10'), 14)} | "
            f"{_fmt_mean_sd_pct(fv_agg.get('mean_macro_top1'), fv_agg.get('std_macro_top1'), 12)} | "
            f"{_fmt_mean_sd_pct(fv_agg.get('mean_micro_top1'), fv_agg.get('std_micro_top1'), 12)} | "
            f"{_fmt_mean_sd_pct(fv_agg.get('mean_mrr'), fv_agg.get('std_mrr'), 11)} | "
            f"{_fmt_mean_sd_num(fv_agg.get('mean_median_rank'), fv_agg.get('std_median_rank'), 12, decimals=1)}"
        )
    lines.append(wide_rule)

    # [2A] Multi-Scale Vocabulary Sweep Across Seeds
    if sweep_agg:
        lines.extend(
            [
                "",
                "[2A] Multi-Scale Vocabulary Sweep Across Seeds (Mean ± SD)",
                wide_rule,
                (
                    f"{'Vocabulary Scale':<22} | {'|V|':>5} | {'Queries':>7} | {'MacTop10':>12} | "
                    f"{'MicTop10':>12} | {'SubjTop10':>12} | {'MacTop5':>12} | {'MacTop1':>12} | "
                    f"{'MicTop1':>12} | {'MRR':>11} | {'MedRank':>11} | {'Fold×Ch':>10}"
                ),
                wide_rule,
            ]
        )
        for sk in ("N20", "N50", "N50_strict", "N100", "N150", "full_vocab"):
            if sk not in sweep_agg:
                continue
            sb = sweep_agg[sk]
            lbl = str(sb.get("label", sk))[:22]
            v_sz = int(sb.get("vocabulary_size", 0))
            q_cnt = int(sb.get("query_count", 0))
            fold_str = _fmt_mean_sd_num(
                sb.get("mean_fold_over_chance_top10"),
                sb.get("std_fold_over_chance_top10"),
                10,
                decimals=2,
            )
            lines.append(
                f"{lbl:<22} | {v_sz:>5d} | {q_cnt:>7d} | "
                f"{_fmt_mean_sd_pct(sb.get('mean_macro_top10'), sb.get('std_macro_top10'), 12)} | "
                f"{_fmt_mean_sd_pct(sb.get('mean_micro_top10'), sb.get('std_micro_top10'), 12)} | "
                f"{_fmt_mean_sd_pct(sb.get('mean_subject_macro_top10'), sb.get('std_subject_macro_top10'), 12)} | "
                f"{_fmt_mean_sd_pct(sb.get('mean_macro_top5'), sb.get('std_macro_top5'), 12)} | "
                f"{_fmt_mean_sd_pct(sb.get('mean_macro_top1'), sb.get('std_macro_top1'), 12)} | "
                f"{_fmt_mean_sd_pct(sb.get('mean_micro_top1'), sb.get('std_micro_top1'), 12)} | "
                f"{_fmt_mean_sd_pct(sb.get('mean_mrr'), sb.get('std_mrr'), 11)} | "
                f"{_fmt_mean_sd_num(sb.get('mean_median_rank'), sb.get('std_median_rank'), 11, decimals=1)} | "
                f"{fold_str}"
            )
        lines.append(wide_rule)

    # [2B] Linguistic Breakdown Across Seeds
    ling_agg = summary.get("linguistic_breakdown_aggregate") or {}
    if isinstance(ling_agg, Mapping) and ling_agg:
        lines.extend(
            [
                "",
                "[2B] Linguistic Category Breakdown Across Seeds (Content vs Function Words, Mean ± SD)",
                wide_rule,
                (
                    f"{'Category':<26} | {'Queries':>7} | {'Unique':>6} | {'50w MacTop10':>13} | "
                    f"{'50w MicTop10':>13} | {'50w MacTop1':>12} | "
                    f"{'Full MacTop10':>13} | {'Full MicTop10':>13} | {'Full MacTop1':>12} | {'50w Top10 p-val':>18}"
                ),
                wide_rule,
            ]
        )
        for lk in ("content_words", "function_words", "content_minus_function"):
            lb = ling_agg.get(lk)
            if not isinstance(lb, Mapping) or int(lb.get("query_count", 0)) == 0:
                continue
            is_d_l = lk == "content_minus_function"
            fmt_l = _fmt_signed_mean_sd_pct if is_d_l else _fmt_mean_sd_pct
            lbl = str(lb.get("label", lk))[:26]
            u_cnt = int(lb.get("unique_word_count", lb.get("unique_targets", 0)))
            p_l_str = str(lb.get("p_str_vocab50_word_macro_top10", "-") if is_d_l else "-")[:18]
            lines.append(
                f"{lbl:<26} | {int(lb.get('query_count', 0)):>7d} | {u_cnt:>6d} | "
                f"{fmt_l(lb.get('mean_vocab50_word_macro_top10', lb.get('mean_vocab50_macro_top10')), lb.get('std_vocab50_word_macro_top10', lb.get('std_vocab50_macro_top10')), 13)} | "
                f"{fmt_l(lb.get('mean_vocab50_micro_top10'), lb.get('std_vocab50_micro_top10'), 13)} | "
                f"{fmt_l(lb.get('mean_vocab50_word_macro_top1', lb.get('mean_vocab50_macro_top1')), lb.get('std_vocab50_word_macro_top1', lb.get('std_vocab50_macro_top1')), 12)} | "
                f"{fmt_l(lb.get('mean_full_vocab_word_macro_top10', lb.get('mean_full_vocab_macro_top10')), lb.get('std_full_vocab_word_macro_top10', lb.get('std_full_vocab_macro_top10')), 13)} | "
                f"{fmt_l(lb.get('mean_full_vocab_micro_top10'), lb.get('std_full_vocab_micro_top10'), 13)} | "
                f"{fmt_l(lb.get('mean_full_vocab_word_macro_top1', lb.get('mean_full_vocab_macro_top1')), lb.get('std_full_vocab_word_macro_top1', lb.get('std_full_vocab_macro_top1')), 12)} | "
                f"{p_l_str:>18}"
            )
        lines.append(wide_rule)

    # [3A] Rolling Revision Horizons Across Seeds (50w + Full-Vocab)
    if isinstance(hz_agg, Mapping) and hz_agg:
        lines.extend(
            [
                "",
                "[3A] Rolling Revision Horizons Across Seeds — 50-Word Bank & Full Training Vocab (Mean ± SD)",
                wide_rule,
                (
                    f"{'Horizon / Delta':<24} | {'50w MacTop10':>13} | {'50w MicTop10':>13} | "
                    f"{'50w MacTop1':>12} | {'50w MicTop1':>12} | {'50w MRR':>11} | "
                    f"{'Full MacTop10':>13} | {'Full MicTop10':>13} | {'Full MacTop1':>12} | {'50w Top10 p':>14}"
                ),
                wide_rule,
            ]
        )
        for h_key in ("K0", "K1", "K2"):
            if h_key not in hz_agg:
                continue
            hb = hz_agg[h_key]
            lbl = str(hb.get("label", h_key))[:24]
            lines.append(
                f"{lbl:<24} | "
                f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_macro_top10'), hb.get('std_common_vocab50_macro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_micro_top10'), hb.get('std_common_vocab50_micro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_macro_top1'), hb.get('std_common_vocab50_macro_top1'), 12)} | "
                f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_micro_top1'), hb.get('std_common_vocab50_micro_top1'), 12)} | "
                f"{_fmt_mean_sd_pct(hb.get('mean_common_vocab50_mrr'), hb.get('std_common_vocab50_mrr'), 11)} | "
                f"{_fmt_mean_sd_pct(hb.get('mean_common_full_vocab_macro_top10'), hb.get('std_common_full_vocab_macro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(hb.get('mean_common_full_vocab_micro_top10'), hb.get('std_common_full_vocab_micro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(hb.get('mean_common_full_vocab_macro_top1'), hb.get('std_common_full_vocab_macro_top1'), 12)} | "
                f"{'-':>14}"
            )
        lines.append(wide_rule)
        for g_key, g_lbl in (
            ("K1_minus_K0", "Step-1 Gain (K1 - K0)"),
            ("K2_minus_K0", "Revision Gain (K2 - K0)"),
            ("K2_minus_K1", "Late Step Gain (K2 - K1)"),
        ):
            if g_key not in hz_agg:
                continue
            gb = hz_agg[g_key]
            p_g_str = str(gb.get("p_str_common_vocab50_macro_top10", "-") or "-")[:14]
            lines.append(
                f"{g_lbl:<24} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_macro_top10'), gb.get('std_common_vocab50_macro_top10'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_micro_top10'), gb.get('std_common_vocab50_micro_top10'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_macro_top1'), gb.get('std_common_vocab50_macro_top1'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_micro_top1'), gb.get('std_common_vocab50_micro_top1'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_mrr'), gb.get('std_common_vocab50_mrr'), 11)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_full_vocab_macro_top10'), gb.get('std_common_full_vocab_macro_top10'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_full_vocab_micro_top10'), gb.get('std_common_full_vocab_micro_top10'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_full_vocab_macro_top1'), gb.get('std_common_full_vocab_macro_top1'), 12)} | "
                f"{p_g_str:>14}"
            )
        lines.append(wide_rule)

    # [3B] Rolling Revision Transitions Across Seeds
    tr_agg = summary.get("revision_transitions_aggregate") or {}
    if isinstance(tr_agg, Mapping) and tr_agg:
        lines.extend(
            [
                "",
                "[3B] Rolling Revision Hit/Miss Transitions Across Seeds (Mean ± SD & McNemar Test)",
                wide_rule,
                (
                    f"{'Transition':<10} | {'Metric Scope':<16} | {'Events':>6} | "
                    f"{'Repairs (M->H)':>15} | {'Harms (H->M)':>14} | {'Net Gain':>13} | "
                    f"{'Repair/Harm':>11} | {'McNemar p-value':>26}"
                ),
                wide_rule,
            ]
        )
        for tk in ("K0_to_K1", "K0_to_K2", "K1_to_K2"):
            tb = tr_agg.get(tk)
            if not isinstance(tb, Mapping):
                continue
            lbl = tk.replace("_to_", " -> ")
            for scope_k, scope_lbl in (
                ("vocab50_top10", "50-Word Top-10"),
                ("vocab50_top1", "50-Word Top-1"),
                ("full_vocab_top10", "Full-Vocab Top-10"),
                ("full_vocab_top1", "Full-Vocab Top-1"),
            ):
                sb = tb.get(scope_k)
                if not isinstance(sb, Mapping):
                    continue
                mc_s = str(sb.get("mcnemar_p_str", "-") or "-")[:26]
                lines.append(
                    f"{lbl:<10} | {scope_lbl:<16} | {int(sb.get('total_common_events', 0)):>6d} | "
                    f"{_fmt_mean_sd_num(sb.get('mean_beneficial_repairs'), sb.get('std_beneficial_repairs'), 15, decimals=1)} | "
                    f"{_fmt_mean_sd_num(sb.get('mean_harmful_overwrites'), sb.get('std_harmful_overwrites'), 14, decimals=1)} | "
                    f"{_fmt_mean_sd_num(sb.get('mean_net_gain_events'), sb.get('std_net_gain_events'), 13, decimals=1, signed=True)} | "
                    f"{_fmt_mean_sd_num(sb.get('mean_repair_to_harm_ratio'), sb.get('std_repair_to_harm_ratio'), 10, decimals=2)}x | "
                    f"{mc_s:>26}"
                )
        lines.append(wide_rule)

    # [4A] Negative Control Multi-Seed Summary (Active Method & Multi-Method × Multi-Metric Spotlight)
    if isinstance(nc_agg, Mapping) and nc_agg.get("enabled"):
        act_method_key = str(nc_agg.get("active_control_method", "raw"))
        act_method_lbl = str(
            nc_agg.get("active_control_method_label", "Raw Un-Debiased Temporal Shift (eta=0.0)")
        )
        prim_metric_key = str(nc_agg.get("primary_control_metric", "vocab50_macro_top10"))
        prim_metric_lbl = str(
            nc_agg.get("primary_control_metric_label", "50-Word Word-Macro Top-10")
        )
        sel_methods = list(nc_agg.get("selected_control_methods") or [act_method_key])
        sel_metrics_info = list(nc_agg.get("selected_control_metrics") or [])
        sel_metric_ids = [
            str(m.get("metric_key") or m.get("metric_id"))
            if isinstance(m, Mapping)
            else str(m)
            for m in sel_metrics_info
            if m
        ] or [prim_metric_key]
        mm_comp = list(nc_agg.get("multi_method_metric_comparison") or [])
        sel_methods_agg = nc_agg.get("selected_methods_aggregates") or {}
        avail_methods = list(nc_agg.get("available_control_methods") or [])
        avail_str = ", ".join(avail_methods) if avail_methods else "raw"
        is_pct = bool(nc_agg.get("primary_metric_is_percentage", True))
        p_c = (
            _fmt_mean_sd_pct(nc_agg.get("primary_metric_clean_mean"), nc_agg.get("primary_metric_clean_std"), 1)
            if is_pct
            else _fmt_mean_sd_num(nc_agg.get("primary_metric_clean_mean"), nc_agg.get("primary_metric_clean_std"), 1)
        ).strip()
        p_s = (
            _fmt_mean_sd_pct(nc_agg.get("primary_metric_shifted_mean"), nc_agg.get("primary_metric_shifted_std"), 1)
            if is_pct
            else _fmt_mean_sd_num(nc_agg.get("primary_metric_shifted_mean"), nc_agg.get("primary_metric_shifted_std"), 1)
        ).strip()
        p_g = (
            _fmt_signed_mean_sd_pct(nc_agg.get("primary_metric_gap_mean"), nc_agg.get("primary_metric_gap_std"), 1)
            if is_pct
            else _fmt_mean_sd_num(nc_agg.get("primary_metric_gap_mean"), nc_agg.get("primary_metric_gap_std"), 1, signed=True)
        ).strip()

        prim_sum_d = nc_agg.get("primary_metric_summary") or {}
        p_s_p = str(prim_sum_d.get("p_str_shifted_vs_chance") or "").strip()
        p_g_p = str(prim_sum_d.get("p_str_gap") or "").strip()
        p_s_disp = f"{p_s} (vs. chance: {p_s_p})" if p_s_p else p_s
        p_g_disp = f"{p_g} ({p_g_p})" if p_g_p else p_g

        lines.extend(
            [
                "",
                f"[4A] Negative Control Multi-Seed Summary ({split}, Mean ± SD)",
                wide_rule,
                f"Selected Control Methods (--control-method): {', '.join(sel_methods)}",
                f"Selected Control Metrics (--control-metric): {', '.join(sel_metric_ids)}",
                f"Active Control Method  (--control-method): [{act_method_key}] {act_method_lbl}",
                f"Primary Control Metric (--control-metric): [{prim_metric_key}] {prim_metric_lbl}",
                f"Primary Metric Spotlight                 : Clean (0s) = {p_c} | Shifted Mean = {p_s_disp} | Net Gap (Clean-Shifted) = {p_g_disp}",
                f"Available Control Methods                : {avail_str}",
                wide_rule,
            ]
        )

        if mm_comp:
            lines.extend(
                [
                    "[4A-0] Multi-Method × Multi-Metric Negative Control Comparison (Mean ± SD Across Seeds):",
                    (
                        f"{'Control Method':<24} | {'Control Metric':<24} | {'Clean (0s)':>14} | "
                        f"{'Shifted (vs Chance)':>28} | {'Net Gap (Clean-Shft)':>28}"
                    ),
                    wide_rule,
                ]
            )
            for mm_r in mm_comp:
                if not isinstance(mm_r, Mapping):
                    continue
                m_id_s = str(mm_r.get("method_id", "-"))[:24]
                mt_id_s = str(mm_r.get("metric_key") or mm_r.get("metric_id") or "-")[:24]
                m_is_pct = bool(mm_r.get("is_ratio", mm_r.get("is_percentage", True)))
                c_m = mm_r.get("mean_clean", mm_r.get("clean_mean"))
                c_sd = mm_r.get("std_clean", mm_r.get("clean_std"))
                s_m = mm_r.get("mean_shifted", mm_r.get("shifted_mean"))
                s_sd = mm_r.get("std_shifted", mm_r.get("shifted_std"))
                g_m = mm_r.get("mean_gap", mm_r.get("gap_mean"))
                g_sd = mm_r.get("std_gap", mm_r.get("gap_std"))
                mc_s = (
                    _fmt_mean_sd_pct(c_m, c_sd, 14)
                    if m_is_pct
                    else _fmt_mean_sd_num(c_m, c_sd, 14)
                )
                ms_s = (
                    _fmt_mean_sd_pct(s_m, s_sd, 1)
                    if m_is_pct
                    else _fmt_mean_sd_num(s_m, s_sd, 1)
                ).strip()
                mg_s = (
                    _fmt_signed_mean_sd_pct(g_m, g_sd, 1)
                    if m_is_pct
                    else _fmt_mean_sd_num(g_m, g_sd, 1, signed=True)
                ).strip()
                ps_ch = str(mm_r.get("p_str_shifted_vs_chance") or "").strip()
                pg_st = str(mm_r.get("p_str_gap") or "").strip()
                if ps_ch:
                    ms_s = f"{ms_s} ({ps_ch})"
                if pg_st:
                    mg_s = f"{mg_s} ({pg_st})"
                lines.append(f"{m_id_s:<24} | {mt_id_s:<24} | {mc_s} | {ms_s:>28} | {mg_s:>28}")
            lines.append(wide_rule)

        prim_per_seed = list(nc_agg.get("primary_metric_per_seed") or [])
        if prim_per_seed:
            lines.append(
                f"Per-Seed Breakdown on Primary Control Metric [{prim_metric_key}] ({prim_metric_lbl}):"
            )
            lines.append(
                f"{'Seed / Summary':<20} | {'Clean (0s)':>16} | {'Shifted Mean':>16} | {'Net Gap (Clean-Shft)':>20}"
            )
            lines.append("-" * 80)
            for ps_r in prim_per_seed:
                sd_num = int(ps_r.get("seed", 0))
                c_s = (
                    _fmt_ratio_pct(ps_r.get("clean"), 16)
                    if is_pct
                    else _fmt_float(ps_r.get("clean"), 16, 2)
                )
                sh_s = (
                    _fmt_ratio_pct(ps_r.get("shifted"), 16)
                    if is_pct
                    else _fmt_float(ps_r.get("shifted"), 16, 2)
                )
                g_s = (
                    _fmt_signed_ratio_pct(ps_r.get("gap"), 20)
                    if is_pct
                    else f"{float(ps_r.get('gap', 0.0)):+20.2f}"
                )
                lines.append(f"{f'seed-{sd_num:03d}':<20} | {c_s} | {sh_s} | {g_s}")
            lines.append("-" * 80)
            lines.append(
                f"{f'Mean±SD (n={len(prim_per_seed)})':<20} | {p_c:>16} | {p_s:>16} | {p_g:>20}"
            )
            lines.append(wide_rule)

        methods_to_render: list[tuple[str, Mapping[str, Any]]] = []
        if isinstance(sel_methods_agg, list):
            for m_item in sel_methods_agg:
                if isinstance(m_item, Mapping):
                    m_k = str(m_item.get("method_id") or m_item.get("active_control_method") or "raw")
                    methods_to_render.append((m_k, m_item))
        elif isinstance(sel_methods_agg, Mapping) and sel_methods_agg:
            for m_id in sel_methods:
                if m_id in sel_methods_agg and isinstance(sel_methods_agg[m_id], Mapping):
                    methods_to_render.append((m_id, sel_methods_agg[m_id]))
        if not methods_to_render:
            methods_to_render.append((act_method_key, nc_agg))

        for m_idx, (m_id, m_agg) in enumerate(methods_to_render, start=1):
            m_lbl = str(m_agg.get("active_control_method_label", m_agg.get("label", m_id)))
            hdr_prefix = (
                f"[4A-{m_idx}] Selected Control Method [{m_id}] ({m_lbl}) — "
                if len(methods_to_render) > 1
                else ""
            )
            p_sh_50 = str(m_agg.get("p_str_shifted_vs_chance_vocab50_macro_top10", "-") or "-")[:16]
            p_gp_50 = str(m_agg.get("p_str_gap_vocab50_macro_top10", "-") or "-")[:16]
            lines.extend(
                [
                    f"{hdr_prefix}All 50-Word & Full-Vocabulary Retrieval Metrics Under Control Method [{m_id}]:",
                    (
                        f"{'Condition':<22} | {'50w MacTop10':>13} | {'50w MicTop10':>13} | "
                        f"{'50w MacTop1':>12} | {'50w MRR':>11} | {'Full MacTop10':>13} | "
                        f"{'Full MicTop10':>13} | {'Full MacTop1':>12} | {'50w Top10 p':>16}"
                    ),
                    wide_rule,
                    (
                        f"{'Clean (0s)':<22} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_clean_vocab50_macro_top10'), m_agg.get('std_clean_vocab50_macro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_clean_vocab50_micro_top10'), m_agg.get('std_clean_vocab50_micro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_clean_vocab50_macro_top1'), m_agg.get('std_clean_vocab50_macro_top1'), 12)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_clean_vocab50_mrr'), m_agg.get('std_clean_vocab50_mrr'), 11)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_clean_full_vocab_macro_top10'), m_agg.get('std_clean_full_vocab_macro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_clean_full_vocab_micro_top10'), m_agg.get('std_clean_full_vocab_micro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_clean_full_vocab_macro_top1'), m_agg.get('std_clean_full_vocab_macro_top1'), 12)} | "
                        f"{'-':>16}"
                    ),
                    (
                        f"{'Shifted Mean':<22} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_shifted_vocab50_macro_top10'), m_agg.get('std_shifted_vocab50_macro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_shifted_vocab50_micro_top10'), m_agg.get('std_shifted_vocab50_micro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_shifted_vocab50_macro_top1'), m_agg.get('std_shifted_vocab50_macro_top1'), 12)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_shifted_vocab50_mrr'), m_agg.get('std_shifted_vocab50_mrr'), 11)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_shifted_full_vocab_macro_top10'), m_agg.get('std_shifted_full_vocab_macro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_shifted_full_vocab_micro_top10'), m_agg.get('std_shifted_full_vocab_micro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(m_agg.get('mean_shifted_full_vocab_macro_top1'), m_agg.get('std_shifted_full_vocab_macro_top1'), 12)} | "
                        f"{p_sh_50:>16}"
                    ),
                    (
                        f"{'Net Gap (Clean-Shft)':<22} | "
                        f"{_fmt_signed_mean_sd_pct(m_agg.get('mean_gap_vocab50_macro_top10'), m_agg.get('std_gap_vocab50_macro_top10'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(m_agg.get('mean_gap_vocab50_micro_top10'), m_agg.get('std_gap_vocab50_micro_top10'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(m_agg.get('mean_gap_vocab50_macro_top1'), m_agg.get('std_gap_vocab50_macro_top1'), 12)} | "
                        f"{_fmt_signed_mean_sd_pct(m_agg.get('mean_gap_vocab50_mrr'), m_agg.get('std_gap_vocab50_mrr'), 11)} | "
                        f"{_fmt_signed_mean_sd_pct(m_agg.get('mean_gap_full_vocab_macro_top10'), m_agg.get('std_gap_full_vocab_macro_top10'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(m_agg.get('mean_gap_full_vocab_micro_top10'), m_agg.get('std_gap_full_vocab_micro_top10'), 13)} | "
                        f"{_fmt_signed_mean_sd_pct(m_agg.get('mean_gap_full_vocab_macro_top1'), m_agg.get('std_gap_full_vocab_macro_top1'), 12)} | "
                        f"{p_gp_50:>16}"
                    ),
                    wide_rule,
                ]
            )

            # Per-Shift Conditions under this Control Method
            per_shifts = list(m_agg.get("per_shift_aggregate") or m_agg.get("shifted_per_condition") or m_agg.get("conditions_aggregate") or [])
            if per_shifts:
                lines.append(f"Per-Shift Conditions Across Seeds Under [{m_id}] (Mean ± SD):")
                for sh_row in per_shifts:
                    sh_lbl = f"  {sh_row.get('label', sh_row.get('condition', 'shift'))}"[:22]
                    lines.append(
                        f"{sh_lbl:<22} | "
                        f"{_fmt_mean_sd_pct(sh_row.get('mean_vocab50_macro_top10'), sh_row.get('std_vocab50_macro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(sh_row.get('mean_vocab50_micro_top10'), sh_row.get('std_vocab50_micro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(sh_row.get('mean_vocab50_macro_top1'), sh_row.get('std_vocab50_macro_top1'), 12)} | "
                        f"{_fmt_mean_sd_pct(sh_row.get('mean_vocab50_mrr'), sh_row.get('std_vocab50_mrr'), 11)} | "
                        f"{_fmt_mean_sd_pct(sh_row.get('mean_full_vocab_macro_top10'), sh_row.get('std_full_vocab_macro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(sh_row.get('mean_full_vocab_micro_top10'), sh_row.get('std_full_vocab_micro_top10'), 13)} | "
                        f"{_fmt_mean_sd_pct(sh_row.get('mean_full_vocab_macro_top1'), sh_row.get('std_full_vocab_macro_top1'), 12)} | "
                        f"{'-':>16}"
                    )
                lines.append(wide_rule)

            # Rolling Revision Horizons under this Control Method
            nc_hz = m_agg.get("horizons_aggregate") or {}
            if isinstance(nc_hz, Mapping) and nc_hz:
                lines.extend(
                    [
                        f"Rolling Revision Horizons Under [{m_id}] (K0 / K1 / K2 Clean vs Shifted vs Net Gap):",
                        (
                            f"{'Horizon':<12} | {'Condition':<14} | {'50w MacTop10':>13} | {'50w MicTop10':>13} | "
                            f"{'50w MacTop1':>12} | {'50w MRR':>11} | {'Full MacTop10':>13} | {'Full MicTop10':>13} | {'50w Top10 p':>16}"
                        ),
                        wide_rule,
                    ]
                )
                for hk in ("K0", "K1", "K2", "K1_minus_K0", "K2_minus_K0"):
                    if hk not in nc_hz:
                        continue
                    h_entry = nc_hz[hk]
                    if not isinstance(h_entry, Mapping):
                        continue
                    is_delta_h = "minus" in hk
                    h_disp = hk.replace("_minus_", "-")
                    for prefix_k, cond_lbl in (
                        ("clean", "Clean (0s)"),
                        ("shifted", "Control Mean"),
                        ("gap", "Net Gap"),
                    ):
                        fmt_fn = (
                            _fmt_signed_mean_sd_pct
                            if (is_delta_h or prefix_k == "gap")
                            else _fmt_mean_sd_pct
                        )
                        p_col = "-"
                        if is_delta_h:
                            p_col = str(h_entry.get(f"p_str_{prefix_k}_common_vocab50_macro_top10", "-") or "-")
                        elif prefix_k == "shifted":
                            p_col = str(h_entry.get("p_str_shifted_vs_chance_common_vocab50_macro_top10", "-") or "-")
                        elif prefix_k == "gap":
                            p_col = str(h_entry.get("p_str_gap_common_vocab50_macro_top10", "-") or "-")
                        lines.append(
                            f"{h_disp:<12} | {cond_lbl:<14} | "
                            f"{fmt_fn(h_entry.get(f'mean_{prefix_k}_common_vocab50_macro_top10'), h_entry.get(f'std_{prefix_k}_common_vocab50_macro_top10'), 13)} | "
                            f"{fmt_fn(h_entry.get(f'mean_{prefix_k}_common_vocab50_micro_top10'), h_entry.get(f'std_{prefix_k}_common_vocab50_micro_top10'), 13)} | "
                            f"{fmt_fn(h_entry.get(f'mean_{prefix_k}_common_vocab50_macro_top1'), h_entry.get(f'std_{prefix_k}_common_vocab50_macro_top1'), 12)} | "
                            f"{fmt_fn(h_entry.get(f'mean_{prefix_k}_common_vocab50_mrr'), h_entry.get(f'std_{prefix_k}_common_vocab50_mrr'), 11)} | "
                            f"{fmt_fn(h_entry.get(f'mean_{prefix_k}_common_full_vocab_macro_top10'), h_entry.get(f'std_{prefix_k}_common_full_vocab_macro_top10'), 13)} | "
                            f"{fmt_fn(h_entry.get(f'mean_{prefix_k}_common_full_vocab_micro_top10'), h_entry.get(f'std_{prefix_k}_common_full_vocab_micro_top10'), 13)} | "
                            f"{p_col[:16]:>16}"
                        )
                lines.append(wide_rule)

    # [4B] Exploratory LOO Common-Mode Subtraction (CMS) Multi-Seed Sweep
    cms_agg = summary.get("common_mode_subtraction_aggregate") or {}
    if isinstance(cms_agg, Mapping) and cms_agg.get("enabled"):
        lines.extend(
            [
                "",
                "[4B] Exploratory Leave-One-Out Common-Mode Subtraction (CMS) Multi-Seed Sweep (Mean ± SD)",
                wide_rule,
                (
                    f"{'CMS Method (eta)':<26} | {'50w Clean10':>12} | {'50w Shft10':>12} | {'50w Gap10':>12} | "
                    f"{'50w GapTop1':>12} | {'Full Clean10':>12} | {'Full Shft10':>12} | {'Full Gap10':>12} | {'Full GapTop1':>12}"
                ),
                wide_rule,
            ]
        )
        for c_row in cms_agg.get("sweep") or []:
            if not isinstance(c_row, Mapping):
                continue
            m_lbl = f"CMS LOO (eta={float(c_row.get('eta', 0.0)):.2f})"[:26]
            lines.append(
                f"{m_lbl:<26} | "
                f"{_fmt_mean_sd_pct(c_row.get('mean_clean_vocab50_macro_top10'), c_row.get('std_clean_vocab50_macro_top10'), 12)} | "
                f"{_fmt_mean_sd_pct(c_row.get('mean_shifted_vocab50_macro_top10'), c_row.get('std_shifted_vocab50_macro_top10'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(c_row.get('mean_gap_vocab50_macro_top10'), c_row.get('std_gap_vocab50_macro_top10'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(c_row.get('mean_gap_vocab50_macro_top1'), c_row.get('std_gap_vocab50_macro_top1'), 12)} | "
                f"{_fmt_mean_sd_pct(c_row.get('mean_clean_full_vocab_macro_top10'), c_row.get('std_clean_full_vocab_macro_top10'), 12)} | "
                f"{_fmt_mean_sd_pct(c_row.get('mean_shifted_full_vocab_macro_top10'), c_row.get('std_shifted_full_vocab_macro_top10'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(c_row.get('mean_gap_full_vocab_macro_top10'), c_row.get('std_gap_full_vocab_macro_top10'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(c_row.get('mean_gap_full_vocab_macro_top1'), c_row.get('std_gap_full_vocab_macro_top1'), 12)}"
            )
            c_hz = c_row.get("horizons_aggregate") or {}
            if isinstance(c_hz, Mapping) and c_hz:
                for hk in ("K0", "K1", "K2"):
                    if hk not in c_hz:
                        continue
                    hc = c_hz[hk].get("clean") or {}
                    hs = c_hz[hk].get("shifted_mean") or {}
                    hg = c_hz[hk].get("net_gap") or {}
                    sub_lbl = f"  ├─ {hk}"[:26]
                    lines.append(
                        f"{sub_lbl:<26} | "
                        f"{_fmt_mean_sd_pct(hc.get('mean_common_vocab50_macro_top10'), hc.get('std_common_vocab50_macro_top10'), 12)} | "
                        f"{_fmt_mean_sd_pct(hs.get('mean_common_vocab50_macro_top10'), hs.get('std_common_vocab50_macro_top10'), 12)} | "
                        f"{_fmt_signed_mean_sd_pct(hg.get('mean_common_vocab50_macro_top10'), hg.get('std_common_vocab50_macro_top10'), 12)} | "
                        f"{_fmt_signed_mean_sd_pct(hg.get('mean_common_vocab50_macro_top1'), hg.get('std_common_vocab50_macro_top1'), 12)} | "
                        f"{_fmt_mean_sd_pct(hc.get('mean_common_full_vocab_macro_top10'), hc.get('std_common_full_vocab_macro_top10'), 12)} | "
                        f"{_fmt_mean_sd_pct(hs.get('mean_common_full_vocab_macro_top10'), hs.get('std_common_full_vocab_macro_top10'), 12)} | "
                        f"{_fmt_signed_mean_sd_pct(hg.get('mean_common_full_vocab_macro_top10'), hg.get('std_common_full_vocab_macro_top10'), 12)} | "
                        f"{_fmt_signed_mean_sd_pct(hg.get('mean_common_full_vocab_macro_top1'), hg.get('std_common_full_vocab_macro_top1'), 12)}"
                    )
        lines.append(wide_rule)

    # [4C] Exploratory Subspace Coherence LOO Debiasing Multi-Seed Sweep
    sub_agg = summary.get("subspace_debiasing_aggregate") or {}
    if isinstance(sub_agg, Mapping) and sub_agg.get("enabled"):
        lines.extend(
            [
                "",
                "[4C] Exploratory Subspace Coherence LOO Debiasing Multi-Seed Sweep (Mean ± SD)",
                wide_rule,
                (
                    f"{'Subspace Method':<26} | {'50w Clean10':>12} | {'50w Shft10':>12} | {'50w Gap10':>12} | "
                    f"{'50w GapTop1':>12} | {'Full Clean10':>12} | {'Full Shft10':>12} | {'Full Gap10':>12} | {'Full GapTop1':>12}"
                ),
                wide_rule,
            ]
        )
        all_sub_rows: list[Mapping[str, Any]] = []
        if isinstance(sub_agg.get("raw_baseline"), Mapping):
            all_sub_rows.append(sub_agg["raw_baseline"])
        for fr_r in sub_agg.get("full_rank_cms_sweep") or []:
            if isinstance(fr_r, Mapping):
                all_sub_rows.append(fr_r)
        for sp_r in sub_agg.get("subspace_sweep") or []:
            if isinstance(sp_r, Mapping):
                all_sub_rows.append(sp_r)
        for s_row in all_sub_rows:
            m_lbl = _short_method_label(s_row)[:26]
            lines.append(
                f"{m_lbl:<26} | "
                f"{_fmt_mean_sd_pct(s_row.get('mean_clean_vocab50_macro_top10'), s_row.get('std_clean_vocab50_macro_top10'), 12)} | "
                f"{_fmt_mean_sd_pct(s_row.get('mean_shifted_vocab50_macro_top10'), s_row.get('std_shifted_vocab50_macro_top10'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(s_row.get('mean_gap_vocab50_macro_top10'), s_row.get('std_gap_vocab50_macro_top10'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(s_row.get('mean_gap_vocab50_macro_top1'), s_row.get('std_gap_vocab50_macro_top1'), 12)} | "
                f"{_fmt_mean_sd_pct(s_row.get('mean_clean_full_vocab_macro_top10'), s_row.get('std_clean_full_vocab_macro_top10'), 12)} | "
                f"{_fmt_mean_sd_pct(s_row.get('mean_shifted_full_vocab_macro_top10'), s_row.get('std_shifted_full_vocab_macro_top10'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(s_row.get('mean_gap_full_vocab_macro_top10'), s_row.get('std_gap_full_vocab_macro_top10'), 12)} | "
                f"{_fmt_signed_mean_sd_pct(s_row.get('mean_gap_full_vocab_macro_top1'), s_row.get('std_gap_full_vocab_macro_top1'), 12)}"
            )
            s_hz = s_row.get("horizons_aggregate") or {}
            if isinstance(s_hz, Mapping) and s_hz:
                for hk in ("K0", "K1", "K2"):
                    if hk not in s_hz:
                        continue
                    hc = s_hz[hk].get("clean") or {}
                    hs = s_hz[hk].get("shifted_mean") or {}
                    hg = s_hz[hk].get("net_gap") or {}
                    sub_lbl = f"  ├─ {hk}"[:26]
                    lines.append(
                        f"{sub_lbl:<26} | "
                        f"{_fmt_mean_sd_pct(hc.get('mean_common_vocab50_macro_top10'), hc.get('std_common_vocab50_macro_top10'), 12)} | "
                        f"{_fmt_mean_sd_pct(hs.get('mean_common_vocab50_macro_top10'), hs.get('std_common_vocab50_macro_top10'), 12)} | "
                        f"{_fmt_signed_mean_sd_pct(hg.get('mean_common_vocab50_macro_top10'), hg.get('std_common_vocab50_macro_top10'), 12)} | "
                        f"{_fmt_signed_mean_sd_pct(hg.get('mean_common_vocab50_macro_top1'), hg.get('std_common_vocab50_macro_top1'), 12)} | "
                        f"{_fmt_mean_sd_pct(hc.get('mean_common_full_vocab_macro_top10'), hc.get('std_common_full_vocab_macro_top10'), 12)} | "
                        f"{_fmt_mean_sd_pct(hs.get('mean_common_full_vocab_macro_top10'), hs.get('std_common_full_vocab_macro_top10'), 12)} | "
                        f"{_fmt_signed_mean_sd_pct(hg.get('mean_common_full_vocab_macro_top10'), hg.get('std_common_full_vocab_macro_top10'), 12)} | "
                        f"{_fmt_signed_mean_sd_pct(hg.get('mean_common_full_vocab_macro_top1'), hg.get('std_common_full_vocab_macro_top1'), 12)}"
                    )
        lines.append(wide_rule)

    # [5] Per-Subject Multi-Seed Summary
    subj_agg = list(summary.get("per_subject_summary_aggregate") or [])
    if subj_agg:
        lines.extend(
            [
                "",
                "[5] Per-Subject Multi-Seed Summary (Mean ± SD Across Seeds)",
                wide_rule,
                (
                    f"{'Subject':<12} | {'Events':>6} | {'50w Q':>6} | {'50w MacTop10':>13} | "
                    f"{'50w MicTop10':>13} | {'50w MacTop1':>12} | {'50w MRR':>11} | "
                    f"{'50w MedRank':>11} | {'Full MacTop10':>13} | {'Full MicTop10':>13} | {'Full MacTop1':>12}"
                ),
                wide_rule,
            ]
        )
        for sr in subj_agg:
            if not isinstance(sr, Mapping):
                continue
            sid = str(sr.get("subject_id", "-"))[:12]
            lines.append(
                f"{sid:<12} | {int(sr.get('total_events', 0)):>6d} | {int(sr.get('vocab50_queries', 0)):>6d} | "
                f"{_fmt_mean_sd_pct(sr.get('mean_macro_top10'), sr.get('std_macro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(sr.get('mean_micro_top10'), sr.get('std_micro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(sr.get('mean_macro_top1'), sr.get('std_macro_top1'), 12)} | "
                f"{_fmt_mean_sd_pct(sr.get('mean_mrr'), sr.get('std_mrr'), 11)} | "
                f"{_fmt_mean_sd_num(sr.get('mean_median_rank'), sr.get('std_median_rank'), 11, decimals=1)} | "
                f"{_fmt_mean_sd_pct(sr.get('mean_full_vocab_macro_top10'), sr.get('std_full_vocab_macro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(sr.get('mean_full_vocab_micro_top10'), sr.get('std_full_vocab_micro_top10'), 13)} | "
                f"{_fmt_mean_sd_pct(sr.get('mean_full_vocab_macro_top1'), sr.get('std_full_vocab_macro_top1'), 12)}"
            )
        lines.append(wide_rule)

    lines.append(wide_banner)
    return "\n".join(lines)


def print_multi_seed_evaluation_report(
    summary: Mapping[str, Any],
    *,
    verbose: bool = False,
    output_dir: Path | str | None = None,
    print_to_stdout: bool = True,
    control_method: str | Sequence[str] | None = None,
    control_metric: str | Sequence[str] | None = None,
) -> str:
    """Save full multi-seed .txt/.md/.csv reports to output_dir and print compact/verbose report."""
    target_dir = output_dir or summary.get("output_dir")
    enriched = _ensure_enriched_multi_seed_summary(
        summary,
        base_dir=target_dir,
        control_method=control_method,
        control_metric=control_metric,
    )
    was_reaggregated = bool(enriched.pop("_was_reaggregated", False))
    split = str(enriched.get("split", "val")).lower()
    is_multi_stage = bool(enriched.get("stages_aggregate"))
    report_filename = (
        f"multi_seed_stage_comparison_report_{split}.txt"
        if is_multi_stage
        else f"multi_seed_evaluation_report_{split}.txt"
    )
    json_filename = (
        f"multi_seed_stage_comparison_{split}.json"
        if is_multi_stage
        else f"multi_seed_evaluation_{split}.json"
    )
    if target_dir:
        out_dir_path = Path(str(target_dir))
        out_path = out_dir_path / report_filename
        json_path = out_dir_path / json_filename
        try:
            out_dir_path.mkdir(parents=True, exist_ok=True)
            if (
                json_path.is_file()
                and (was_reaggregated or control_method is not None or control_metric is not None)
            ):
                json_path.write_text(
                    json.dumps(enriched, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            full_txt = format_multi_seed_evaluation_report(
                enriched,
                verbose=True,
                control_method=control_method,
                control_metric=control_metric,
            )
            out_path.write_text(full_txt + "\n", encoding="utf-8")
            export_multi_seed_tables_and_markdown(enriched, out_dir_path)
        except OSError:
            pass
    rendered = format_multi_seed_evaluation_report(
        enriched,
        verbose=verbose,
        control_method=control_method,
        control_metric=control_metric,
    )
    if print_to_stdout:
        print("\n" + rendered, flush=True)
    return rendered


__all__ = [
    "REPORT_WIDTH",
    "format_common_mode_subtraction_report",
    "format_multi_seed_evaluation_report",
    "format_sequence_decoding_executive_summary",
    "format_sequence_decoding_report",
    "format_stage_comparison_report",
    "format_subspace_coherence_debiasing_report",
    "format_word_evaluation_report",
    "print_common_mode_subtraction_report",
    "print_evaluation_report",
    "print_multi_seed_evaluation_report",
    "print_sequence_decoding_report",
    "print_stage_comparison_report",
    "print_subspace_coherence_debiasing_report",
    "print_word_evaluation_report",
    "render_saved_evaluation_report",
    "upgrade_and_refresh_saved_multi_seed_reports",
]
