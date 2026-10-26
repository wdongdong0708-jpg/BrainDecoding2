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
    export_multi_seed_tables_and_markdown,
    export_sequence_decoding_tables_and_markdown,
    export_stage_comparison_tables_and_markdown,
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
        sweep = dict(metrics.get("vocabulary_sweep") or {})
        horizons = dict(metrics.get("revision_horizons") or {})
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
        lines.append(
            f"{str(clean_row.get('label', 'Clean (0s)'))[:20]:<20} | "
            f"{_fmt_ratio_pct(clean_row.get('vocab50_macro_top10'), 12)} | "
            f"{'(Ref)':>9} | "
            f"{_fmt_ratio_pct(clean_row.get('vocab50_macro_top1'), 11)} | "
            f"{_fmt_ratio_pct(clean_row.get('full_vocab_macro_top10'), 10)} | "
            f"{_fmt_float(clean_row.get('vocab50_median_rank', clean_row.get('vocab50_median_rank_macro')), 7, 1)}"
        )
        if verbose or shift_cnt <= 1:
            for cond in ctrl_conditions:
                s_row = cond.get("summary") or {}
                d_50 = (s_row.get("delta_vs_clean") or {}).get("vocab50_macro_top10")
                if d_50 is None and clean_row.get("vocab50_macro_top10") is not None and s_row.get("vocab50_macro_top10") is not None:
                    d_50 = float(s_row["vocab50_macro_top10"]) - float(clean_row["vocab50_macro_top10"])
                lines.append(
                    f"{str(s_row.get('label', cond.get('key', '-')))[:20]:<20} | "
                    f"{_fmt_ratio_pct(s_row.get('vocab50_macro_top10'), 12)} | "
                    f"{_fmt_signed_ratio_pct(d_50, 9)} | "
                    f"{_fmt_ratio_pct(s_row.get('vocab50_macro_top1'), 11)} | "
                    f"{_fmt_ratio_pct(s_row.get('full_vocab_macro_top10'), 10)} | "
                    f"{_fmt_float(s_row.get('vocab50_median_rank', s_row.get('vocab50_median_rank_macro')), 7, 1)}"
                )
        if shift_cnt > 1:
            m50 = float(shifted_agg.get("mean_vocab50_macro_top10", 0.0)) * 100.0
            sd50 = float(shifted_agg.get("std_vocab50_macro_top10", 0.0)) * 100.0
            d50 = shifted_agg.get("delta_vocab50_macro_top10_vs_clean")
            gap50 = shifted_agg.get("clean_minus_shifted_vocab50_macro_top10")
            gap_fv = shifted_agg.get("clean_minus_shifted_full_vocab_macro_top10")
            lines.append(
                f"{'Shifted Mean ± SD':<20} | "
                f"{f'{m50:.2f}±{sd50:.2f}%':>12} | "
                f"{_fmt_signed_ratio_pct(d50, 9)} | "
                f"{_fmt_ratio_pct(shifted_agg.get('mean_vocab50_macro_top1'), 11)} | "
                f"{_fmt_ratio_pct(shifted_agg.get('mean_full_vocab_macro_top10'), 10)} | "
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
                    f"{'-':>8} | {'-':>8}"
                )
            if "K0" in horizons and "K2" in horizons:
                k0, k2 = horizons["K0"], horizons["K2"]
                g50_10 = float(k2.get("common_vocab50_macro_top10", 0.0)) - float(k0.get("common_vocab50_macro_top10", 0.0))
                g50_1 = float(k2.get("common_vocab50_macro_top1", 0.0)) - float(k0.get("common_vocab50_macro_top1", 0.0))
                gfv_10 = float(k2.get("common_full_vocab_macro_top10", 0.0)) - float(k0.get("common_full_vocab_macro_top10", 0.0))
                lines.append(
                    f"{'  Δ Horizon Gain (K2 - K0)':<26} | "
                    f"{_fmt_signed_ratio_pct(g50_10, 9)} | "
                    f"{_fmt_signed_ratio_pct(g50_1, 8)} | "
                    f"{_fmt_signed_ratio_pct(gfv_10, 9)} | "
                    f"{'-':>8} | {'-':>8}"
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
            lines.append(
                f"{short_lbl:<18} | "
                f"{_fmt_ratio_pct(c_ref.get('vocab50_macro_top10'), 8)} | "
                f"{_fmt_ratio_pct(s_agg.get('mean_vocab50_macro_top10'), 8)} | "
                f"{_fmt_signed_ratio_pct(s_agg.get('clean_minus_shifted_vocab50_macro_top10'), 8)} | "
                f"{_fmt_ratio_pct(c_ref.get('full_vocab_macro_top10'), 8)} | "
                f"{_fmt_ratio_pct(s_agg.get('mean_full_vocab_macro_top10'), 8)} | "
                f"{_fmt_signed_ratio_pct(s_agg.get('clean_minus_shifted_full_vocab_macro_top10'), 8)}"
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
        "[1] Overall Retrieval & Net Neural Gap by Weight η",
    ]
    hdr = (
        f"{'Weight (eta)':<14} | {'50w Clean':>9} | {'50w Shift':>9} | {'50w Gap':>9} | "
        f"{'vs eta=0':>8} | {'FullClean':>9} | {'FullShift':>9}"
    )
    lines.append(rule)
    lines.append(hdr)
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
            f"{_fmt_ratio_pct(c_ref.get('full_vocab_macro_top10'), 9)} | "
            f"{_fmt_ratio_pct(agg.get('mean_full_vocab_macro_top10'), 9)}"
        )
    lines.append(rule)

    if has_horizons:
        lines.append(
            "\n[2] Revision Horizons (K0 / K1 / K2) 50-Word Macro Top-10 Across η"
        )
        h_hdr = (
            f"{'Weight (eta)':<14} | {'K0 Clean/Shift (Gap)':>21} | "
            f"{'K1 Clean/Shift (Gap)':>21} | {'K2 Clean/Shift (Gap)':>21}"
        )
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
                    ch_v = float(c_hz[hk].get("common_vocab50_macro_top10", 0.0)) * 100.0
                    sh_v = float((s_hz.get(hk) or {}).get("mean_common_vocab50_macro_top10", 0.0)) * 100.0
                    gp_v = ch_v - sh_v
                    cols.append(f"{ch_v:5.1f}/{sh_v:4.1f}({gp_v:+5.1f}%)")
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

    banner = "=" * REPORT_WIDTH
    rule = "-" * REPORT_WIDTH
    lines: list[str] = [
        banner,
        f"Subspace-Projected + Coherence-Gated LOO Debiasing | {dataset} | {stage} ({split.upper()})",
        f"Seed: {seed} | ExpTag: {exp_tag} | CenteredSVD: {center_svd} | Shifts: M={m_count} (LOO={m_count - 1})",
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
        "\n2. Feature-Space Subspace LOO Retrieval Comparison (50w Top-10 & Top-1)"
    )
    m_hdr = (
        f"{'Configuration':<26} | {'50wTop10':>8} | {'ShiftT10':>8} | {'T10 Gap':>8} | "
        f"{'50w Clean Top1':>14} | {'50w Shift Top1':>14}"
    )
    lines.append(rule)
    lines.append(m_hdr)
    lines.append(rule)

    for item in display_rows:
        lbl = _short_method_label(item)
        c_ref = item.get("clean_reference") or {}
        agg = item.get("shifted_aggregate") or {}
        lines.append(
            f"{lbl:<26} | "
            f"{_fmt_ratio_pct(c_ref.get('vocab50_macro_top10'), 8)} | "
            f"{_fmt_ratio_pct(agg.get('mean_vocab50_macro_top10'), 8)} | "
            f"{_fmt_signed_ratio_pct(agg.get('clean_minus_shifted_vocab50_macro_top10'), 8)} | "
            f"{_fmt_ratio_pct(c_ref.get('vocab50_macro_top1'), 14)} | "
            f"{_fmt_ratio_pct(agg.get('mean_vocab50_macro_top1'), 14)}"
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
        lines.append(
            "\n3A. Revision Horizons (K0 / K1 / K2) 50-Word Macro Top-10 Clean/Shift (Gap)"
        )
        h_hdr = (
            f"{'Configuration':<24} | {'K0 Clean/Shift(Gap)':>18} | "
            f"{'K1 Clean/Shift(Gap)':>18} | {'K2 Clean/Shift(Gap)':>18}"
        )
        lines.append(rule)
        lines.append(h_hdr)
        lines.append(rule)
        for item in hz_rows:
            lbl = _short_method_label(item)[:24]
            c_hz = (item.get("clean_reference") or {}).get("horizons") or {}
            s_hz = (item.get("shifted_aggregate") or {}).get("horizons") or {}
            cols10: list[str] = []
            for hk in ("K0", "K1", "K2"):
                if hk in c_hz:
                    ch_v = float(c_hz[hk].get("common_vocab50_macro_top10", 0.0)) * 100.0
                    sh_v = float((s_hz.get(hk) or {}).get("mean_common_vocab50_macro_top10", 0.0)) * 100.0
                    gp_v = ch_v - sh_v
                    cols10.append(f"{ch_v:4.1f}/{sh_v:4.1f}({gp_v:+5.1f}%)")
                else:
                    cols10.append("N/A")
            lines.append(
                f"{lbl:<24} | {cols10[0]:>18} | {cols10[1]:>18} | {cols10[2]:>18}"
            )
        lines.append(rule)

        lines.append(
            "\n3B. Revision Horizons (K0 / K1 / K2) 50-Word Macro Top-1 Clean vs Strict-LOO Shifted"
        )
        lines.append(rule)
        lines.append(h_hdr)
        lines.append(rule)
        for item in hz_rows:
            lbl = _short_method_label(item)[:24]
            c_hz = (item.get("clean_reference") or {}).get("horizons") or {}
            s_hz = (item.get("shifted_aggregate") or {}).get("horizons") or {}
            cols1: list[str] = []
            for hk in ("K0", "K1", "K2"):
                if hk in c_hz:
                    ch_t1 = float(c_hz[hk].get("common_vocab50_macro_top1", 0.0)) * 100.0
                    sh_t1 = float((s_hz.get(hk) or {}).get("mean_common_vocab50_macro_top1", 0.0)) * 100.0
                    gp_t1 = ch_t1 - sh_t1
                    cols1.append(f"{ch_t1:4.1f}/{sh_t1:4.1f}({gp_t1:+5.1f}%)")
                else:
                    cols1.append("N/A")
            lines.append(
                f"{lbl:<24} | {cols1[0]:>18} | {cols1[1]:>18} | {cols1[2]:>18}"
            )
        lines.append(rule)

    if isinstance(best_clean, Mapping):
        bc_c = best_clean.get("clean_reference") or {}
        bc_s = best_clean.get("shifted_aggregate") or {}
        lines.append(
            f"[Best Clean-Preserving] {_short_method_label(best_clean)}: "
            f"Clean50={_fmt_ratio_pct(bc_c.get('vocab50_macro_top10'), 6)} | "
            f"Shift50={_fmt_ratio_pct(bc_s.get('mean_vocab50_macro_top10'), 6)} | "
            f"Gap={_fmt_signed_ratio_pct(bc_s.get('clean_minus_shifted_vocab50_macro_top10'), 6)}"
        )
    if isinstance(best_gap, Mapping):
        bg_c = best_gap.get("clean_reference") or {}
        bg_s = best_gap.get("shifted_aggregate") or {}
        lines.append(
            f"[Best Net-Neural-Gap]   {_short_method_label(best_gap)}: "
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
        first_c = next(iter(conds.values()), {})
        segs = first_c.get("segment_count", "-")
        words = first_c.get("total_words", "-")
        lines.append(
            f"[1] Full-Vocabulary Conditions (Segments={segs}, TotalWords={words})"[:REPORT_WIDTH]
        )
        c_hdr = (
            f"{'Condition':<22} | {'BLEU-1':>7} | {'InVoc%':>7} | "
            f"{'WordAcc':>7} | {'Acc@5':>7} | {'CER(%)':>7} | {'BERT-F1':>7}"
        )
        lines.append(rule)
        lines.append(c_hdr)
        lines.append(rule)
        for c_name, m in conds.items():
            if not isinstance(m, Mapping):
                continue
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
        gains = _compute_s3_revision_gain_deltas(conds)
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
                    f" * S3 vs K0 BLEU-1 95% CI: "
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
                f" * Across-Subject S3: BLEU1={mb:.2f}±{sb:.2f}% | WordAcc={mw:.2f}±{sw:.2f}%"[
                    :REPORT_WIDTH
                ]
            )

    vs = report.get("vocabulary_scaling")
    if isinstance(vs, Mapping) and vs:
        lines.append(
            "\n[2] Multi-Scale Candidate Vocabulary Scaling (All Segments vs. >=80% Core)"
        )
        v_hdr = (
            f"{'Scale |V|':<10} | {'TokCov':>6} | {'All S3':>9} | "
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
        "clean_C_minus_A_pp": "Clean 滚动修正净增益 (C - A = S3_clean - K0)",
        "shifted_C_minus_A_pp": "Shifted 错位滚动增益 (C_shift - A_shift)",
        "clean_minus_shifted_S3_pp": "真实 vs 错位地面实况净优势 (S3_clean - S3_shift)",
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
    returns the full 6-table report (`[表 1]` .. `[表 5]`).
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
        first_cond = next(iter(conds.values()), {})
        tot_segs = first_cond.get("segment_count", "-")
        tot_words = first_cond.get("total_words", "-")
        invoc_rate = first_cond.get("in_vocabulary_rate_pct", None)
        cov_str = f"{float(invoc_rate):.2f}%" if invoc_rate is not None else "-"
        lines.append(
            f"\n[表 1] 全词库主实验结果 (Full Vocabulary | Segments={tot_segs}, TotalWords={tot_words}, TokenCoverage={cov_str})"
        )
        lines.extend(_format_s3_conditions_table_full(conds))
        main_gains = _compute_s3_revision_gain_deltas(conds)
        lines.extend(_format_s3_revision_gains_block_full(main_gains))

    core_bench = report.get("core_vocabulary_benchmark")
    if isinstance(core_bench, Mapping) and core_bench.get("subsets"):
        v_core = core_bench.get("vocabulary_size", 250)
        tok_cov_core = core_bench.get("test_token_coverage_pct")
        cov_core_str = (
            f"{float(tok_cov_core):.2f}%" if tok_cov_core is not None else "-"
        )
        lines.append(
            f"\n[表 2] 高覆盖核心词表平行基准 (Core Vocabulary |V| = {v_core}, 全集词位覆盖率 = {cov_core_str}，对标 SimpleB2T & Jayalath 2025)"
        )
        subsets_map = core_bench.get("subsets") or {}
        subset_titles = (
            ("all_segments", "口径 1 [all_segments]: 全部测试句段 (含 In-Vocab Word Acc)"),
            ("core_subset_100pct", "口径 2a [core_subset_100pct]: 100% 词表内核心句 (0% OOV Core Set，直接对标 SimpleB2T)"),
            ("core_subset_80pct", "口径 2b [core_subset_80pct]: >=80% 词表内高覆盖核心句"),
            ("expanded_subset_70pct", "口径 2c [expanded_subset_70pct]: >=70% 词表内扩展高覆盖句"),
        )
        for sub_key, sub_title in subset_titles:
            sub_block = subsets_map.get(sub_key)
            if not isinstance(sub_block, Mapping):
                continue
            sub_conds = sub_block.get("conditions") or {}
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
            "\n[表 3] 全词库 (Full Vocabulary) 下的高覆盖核心子集拆解 (100% 零 OOV / >=80% / >=70%)"
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
            "expanded_subset_70pct",
        ):
            sb = fc_subsets.get(sub_key)
            if not isinstance(sb, Mapping):
                continue
            seg_cnt = sb.get("segment_count", 0)
            sb_conds = sb.get("conditions") or {}
            for c_name in (
                "S0",
                "K0",
                "K1",
                "S3_clean",
                "K0_shifted_neg75s",
                "S3_shifted_neg75s",
            ):
                if c_name not in sb_conds:
                    continue
                m = sb_conds[c_name]
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


def render_saved_evaluation_report(
    input_path: Path | str,
    *,
    verbose: bool = False,
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
                render_saved_evaluation_report(fpath, verbose=verbose)
            )
        return "\n\n".join(b for b in rendered_blocks if b)

    payload = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object in {src}")

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
            payload, verbose=verbose, output_dir=src.parent
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
        return print_subspace_coherence_debiasing_report(payload, verbose=verbose)

    if exp_type == "exploratory_leave_one_out_common_mode_subtraction" or (
        "eta_grid" in payload and "sweep" in payload
    ):
        return print_common_mode_subtraction_report(payload, verbose=verbose)

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


def format_multi_seed_evaluation_report(
    summary: Mapping[str, Any],
    *,
    verbose: bool = False,
) -> str:
    """Format a multi-seed evaluation summary (single-stage or multi-stage)."""
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
        for st_key in ("word", "context", "revision"):
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
            for st_key in ("word", "context", "revision"):
                if st_key in stages_agg:
                    lines.append("")
                    lines.append(
                        format_multi_seed_evaluation_report(
                            stages_agg[st_key], verbose=True
                        )
                    )
        lines.append(banner)
        return "\n".join(lines)

    # Single-stage multi-seed schema
    stage = str(summary.get("stage", "word"))
    stage_label = STAGE_DISPLAY_NAMES.get(stage, stage)
    per_seed = list(summary.get("per_seed") or [])
    sweep_agg = dict(summary.get("vocabulary_sweep_aggregate") or {})
    n50_agg = sweep_agg.get("N50") or {}
    fv_agg = sweep_agg.get("full_vocab") or {}

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

    if sweep_agg and (verbose or len(sweep_agg) > 1):
        lines.append("\n[2] Vocabulary Scale Sweep Across Seeds (Mean ± SD)")
        lines.append(rule)
        lines.append(
            f"{'Vocabulary Scale':<18} | {'MacTop10 (M±SD)':>15} | {'MicTop10 (M±SD)':>15} | "
            f"{'MacTop1 (M±SD)':>14} | {'MRR (M±SD)':>12}"
        )
        lines.append(rule)
        scale_keys = (
            ("N20", "N50", "N100", "N150", "full_vocab")
            if verbose
            else ("N50", "full_vocab")
        )
        for sk in scale_keys:
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

    hz_agg = summary.get("revision_horizons_aggregate") or {}
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
        if "K2_minus_K0" in hz_agg:
            gb = hz_agg["K2_minus_K0"]
            lines.append(
                f"{'Revision Gain (K2-K0)':<22} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_macro_top10'), gb.get('std_common_vocab50_macro_top10'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_micro_top10'), gb.get('std_common_vocab50_micro_top10'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_vocab50_macro_top1'), gb.get('std_common_vocab50_macro_top1'), 13)} | "
                f"{_fmt_signed_mean_sd_pct(gb.get('mean_common_full_vocab_macro_top10'), gb.get('std_common_full_vocab_macro_top10'), 13)}"
            )
        lines.append(rule)

    nc_agg = summary.get("negative_controls_aggregate") or {}
    if isinstance(nc_agg, Mapping) and nc_agg.get("enabled"):
        lines.append(f"\n[4] Negative Control Multi-Seed Summary ({split}, Mean ± SD)")
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
        lines.append(rule)

    lines.append(banner)
    return "\n".join(lines)


def print_multi_seed_evaluation_report(
    summary: Mapping[str, Any],
    *,
    verbose: bool = False,
    output_dir: Path | str | None = None,
    print_to_stdout: bool = True,
) -> str:
    """Save full multi-seed .txt report to output_dir and print compact/verbose report."""
    split = str(summary.get("split", "val")).lower()
    target_dir = output_dir or summary.get("output_dir")
    is_multi_stage = bool(summary.get("stages_aggregate"))
    report_filename = (
        f"multi_seed_stage_comparison_report_{split}.txt"
        if is_multi_stage
        else f"multi_seed_evaluation_report_{split}.txt"
    )
    if target_dir:
        out_dir_path = Path(str(target_dir))
        out_path = out_dir_path / report_filename
        try:
            out_dir_path.mkdir(parents=True, exist_ok=True)
            full_txt = format_multi_seed_evaluation_report(summary, verbose=True)
            out_path.write_text(full_txt + "\n", encoding="utf-8")
            export_multi_seed_tables_and_markdown(summary, out_dir_path)
        except OSError:
            pass
    rendered = format_multi_seed_evaluation_report(summary, verbose=verbose)
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
]
