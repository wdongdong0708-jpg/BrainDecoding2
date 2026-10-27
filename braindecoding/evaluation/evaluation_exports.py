"""Human-readable multi-format evaluation exporters (CSV spreadsheets, Markdown reports, and side-by-side decoded sentences).

Automatically exports flat UTF-8-SIG ``.csv`` tables (under ``<output_dir>/tables/``),
structured GitHub-Flavored Markdown ``.md`` reports, and S3 segment-by-segment
decoded sentence comparisons (``s3_decoded_sentences_<split>.md`` and ``.csv``)
so users never have to manually inspect deeply nested JSON files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _pct(val: Any, scale: float = 100.0, digits: int = 2) -> float:
    """将比例或百分比安全转换为保留指定小数位的浮点数。"""
    if val is None or val == "":
        return 0.0
    try:
        return round(float(val) * scale, digits)
    except (TypeError, ValueError):
        return 0.0


def _num(val: Any, digits: int = 4) -> float:
    """将数值安全转换为保留指定小数位的浮点数。"""
    if val is None or val == "":
        return 0.0
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return 0.0


def _fmt_pct(val: Any, scale: float = 100.0, digits: int = 2) -> str:
    """格式化为带 % 的字符串。"""
    if val is None or val == "":
        return "-"
    try:
        return f"{float(val) * scale:.{digits}f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_signed_pct(val: Any, scale: float = 100.0, digits: int = 2) -> str:
    """格式化为带正负号与 pp 的百分点差值字符串。"""
    if val is None or val == "":
        return "-"
    try:
        return f"{float(val) * scale:+.{digits}f} pp"
    except (TypeError, ValueError):
        return "-"


def _write_csv_table(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """以 UTF-8-SIG 编码写入扁平 CSV 表（兼容 Excel 直接双击打开与 Pandas 读取）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return path


def _join_tokens_natural(words: Sequence[str], language: str) -> str:
    """将词序列拼接为自然可读的法语或中文句子字符串。"""
    tokens = [str(w).strip() for w in words if str(w).strip()]
    if not tokens:
        return ""
    if language == "zh":
        return "".join(tokens)
    text = " ".join(tokens)
    for prefix in ("l'", "d'", "j'", "c'", "n'", "s'", "qu'", "m'", "t'"):
        text = text.replace(f"{prefix} ", prefix)
    text = text.replace(" -", "-")
    return text


def export_word_evaluation_tables_and_markdown(
    eval_result: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """将单阶段词级评估结果导出为 tables/*.csv 与 evaluation_report_{split}.md。"""
    target_dir_raw = output_dir or eval_result.get("output_dir")
    if not target_dir_raw:
        return {}
    out_dir = Path(str(target_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    split = str(eval_result.get("split", "val"))
    dataset = str(eval_result.get("dataset", "-"))
    stage = str(eval_result.get("stage", "word"))
    stage_disp = str(eval_result.get("stage_display_name", stage))
    seed = eval_result.get("seed", 0)
    win_sec = float(eval_result.get("window_seconds", 1.0))
    win_off = float(eval_result.get("window_start_offset_seconds", 0.0))
    ckpt = eval_result.get("checkpoint") or {}
    ckpt_epoch = ckpt.get("epoch", "-") if isinstance(ckpt, dict) else "-"
    metrics = eval_result.get("metrics") or {}
    sweep = metrics.get("vocabulary_sweep") or {}
    horizons = metrics.get("revision_horizons") or {}
    transitions = metrics.get("revision_transitions") or {}
    ling = metrics.get("linguistic_breakdown") or {}
    sig_map = metrics.get("statistical_significance") or {}
    per_subj = metrics.get("per_subject") or []
    subj_sum = metrics.get("per_subject_summary") or {}
    nc = eval_result.get("negative_controls") or {}

    exported: dict[str, str] = {}

    # 1. 导出词表规模汇总表 vocabulary_metrics_{split}.csv
    vocab_rows: list[dict[str, Any]] = []
    scale_order = ("N20", "N50", "N50_strict", "N100", "N150", "full_vocab")
    ordered_keys = [k for k in scale_order if k in sweep] + [
        k for k in sweep if k not in scale_order
    ]
    for s_key in ordered_keys:
        entry = sweep.get(s_key)
        if not isinstance(entry, Mapping):
            continue
        c_cnt = int(entry.get("candidate_count", 0))
        q_cnt = int(entry.get("query_count", 0))
        w_top10 = float(entry.get("word_macro_top10", entry.get("macro_top10", 0.0)))
        s_top10 = float(entry.get("subject_macro_top10", entry.get("micro_top10", w_top10)))
        s_std10 = float(entry.get("subject_std_top10", 0.0))
        m_top10 = float(entry.get("micro_top10", 0.0))
        c_top10 = float(
            entry.get("chance_top10", min(10.0, float(max(1, c_cnt))) / max(1, c_cnt))
        )
        fold_10 = w_top10 / max(c_top10, 1e-12)
        delta_10 = w_top10 - c_top10
        w_top5 = float(entry.get("word_macro_top5", entry.get("macro_top5", 0.0)))
        m_top5 = float(entry.get("micro_top5", 0.0))
        w_top1 = float(entry.get("word_macro_top1", entry.get("macro_top1", 0.0)))
        m_top1 = float(entry.get("micro_top1", 0.0))
        c_top1 = float(entry.get("chance_top1", 1.0 / max(1, c_cnt)))
        cov_pct = float(
            entry.get(
                "in_vocabulary_coverage_pct",
                q_cnt / max(1, int(metrics.get("evaluated_events", max(1, q_cnt)))) * 100.0,
            )
        )
        vocab_rows.append(
            {
                "scale_key": s_key,
                "label": str(entry.get("label", s_key)),
                "oov_policy": str(entry.get("oov_policy", "filter_in_vocabulary")),
                "candidate_count": c_cnt,
                "supported_candidates": int(entry.get("supported_candidates", c_cnt)),
                "query_count": q_cnt,
                "in_vocabulary_coverage_pct": _num(cov_pct, 2),
                "macro_top10_pct": _pct(w_top10),
                "micro_top10_pct": _pct(m_top10),
                "subject_macro_top10_pct": _pct(s_top10),
                "subject_std_top10_pct": _pct(s_std10),
                "chance_top10_pct": _pct(c_top10),
                "fold_over_chance_top10": _num(fold_10, 2),
                "delta_vs_chance_top10_pp": _pct(delta_10),
                "macro_top5_pct": _pct(w_top5),
                "micro_top5_pct": _pct(m_top5),
                "macro_top1_pct": _pct(w_top1),
                "micro_top1_pct": _pct(m_top1),
                "chance_top1_pct": _pct(c_top1),
                "mrr": _num(entry.get("mrr", 0.0), 4),
                "median_rank": _num(entry.get("median_rank", 0.0), 1),
                "chance_median_rank": _num(
                    entry.get("chance_median_rank", (c_cnt + 1) / 2.0), 1
                ),
            }
        )
    if vocab_rows:
        v_csv = _write_csv_table(
            tables_dir / f"vocabulary_metrics_{split}.csv",
            list(vocab_rows[0].keys()),
            vocab_rows,
        )
        exported["vocabulary_metrics_csv"] = str(v_csv)

    # 2. 导出实词 vs. 虚词拆解表 linguistic_breakdown_{split}.csv
    if isinstance(ling, Mapping) and ling:
        ling_rows: list[dict[str, Any]] = []
        for cat_key in ("content_words", "function_words", "content_minus_function"):
            cat = ling.get(cat_key)
            if not isinstance(cat, Mapping):
                continue
            ling_rows.append(
                {
                    "category": cat_key,
                    "label": str(cat.get("label", cat_key)),
                    "query_count": int(cat.get("query_count", 0)),
                    "unique_word_count": int(cat.get("unique_word_count", 0)),
                    "event_share_pct": _num(cat.get("event_share_pct", 0.0), 2),
                    "vocab50_macro_top10_pct": _pct(
                        cat.get("vocab50_word_macro_top10", cat.get("vocab50_macro_top10", 0.0))
                    ),
                    "vocab50_micro_top10_pct": _pct(
                        cat.get("vocab50_micro_top10", 0.0)
                    ),
                    "vocab50_macro_top1_pct": _pct(
                        cat.get("vocab50_word_macro_top1", cat.get("vocab50_macro_top1", 0.0))
                    ),
                    "vocab50_micro_top1_pct": _pct(
                        cat.get("vocab50_micro_top1", 0.0)
                    ),
                    "vocab50_mrr": _num(cat.get("vocab50_mrr", 0.0), 4),
                    "full_vocab_macro_top10_pct": _pct(
                        cat.get("full_vocab_word_macro_top10", cat.get("full_vocab_macro_top10", 0.0))
                    ),
                    "full_vocab_micro_top10_pct": _pct(
                        cat.get("full_vocab_micro_top10", 0.0)
                    ),
                    "full_vocab_macro_top1_pct": _pct(
                        cat.get("full_vocab_word_macro_top1", cat.get("full_vocab_macro_top1", 0.0))
                    ),
                    "full_vocab_micro_top1_pct": _pct(
                        cat.get("full_vocab_micro_top1", 0.0)
                    ),
                    "full_vocab_mrr": _num(cat.get("full_vocab_mrr", 0.0), 4),
                    "full_vocab_median_rank": _num(
                        cat.get("full_vocab_median_rank", 0.0), 1
                    ),
                }
            )
        if ling_rows:
            l_csv = _write_csv_table(
                tables_dir / f"linguistic_breakdown_{split}.csv",
                list(ling_rows[0].keys()),
                ling_rows,
            )
            exported["linguistic_breakdown_csv"] = str(l_csv)

    # 3. 导出滚动回修视野与状态转移表 revision_horizons_{split}.csv
    if isinstance(horizons, Mapping) and horizons:
        h_rows: list[dict[str, Any]] = []
        for h_key in ("K0", "K1", "K2"):
            if h_key not in horizons:
                continue
            h = horizons[h_key]
            fv_m10 = h.get(
                "common_full_vocab_word_macro_top10",
                h.get("common_full_vocab_macro_top10", 0.0),
            )
            h_rows.append(
                {
                    "horizon": h_key,
                    "label": str(h.get("label", h_key)),
                    "delay_words": int(h.get("delay", 0)),
                    "common_vocab50_macro_top10_pct": _pct(
                        h.get("common_vocab50_macro_top10", 0.0)
                    ),
                    "common_vocab50_micro_top10_pct": _pct(
                        h.get("common_vocab50_micro_top10", 0.0)
                    ),
                    "common_vocab50_macro_top1_pct": _pct(
                        h.get("common_vocab50_macro_top1", 0.0)
                    ),
                    "common_vocab50_micro_top1_pct": _pct(
                        h.get("common_vocab50_micro_top1", 0.0)
                    ),
                    "common_vocab50_mrr": _num(
                        h.get("common_vocab50_mrr", 0.0), 4
                    ),
                    "common_vocab50_median_rank": _num(
                        h.get("common_vocab50_median_rank", 25.0), 1
                    ),
                    "common_full_vocab_macro_top10_pct": _pct(fv_m10),
                    "common_full_vocab_micro_top10_pct": _pct(
                        h.get("common_full_vocab_micro_top10", 0.0)
                    ),
                    "common_full_vocab_macro_top1_pct": _pct(
                        h.get("common_full_vocab_macro_top1", 0.0)
                    ),
                    "common_full_vocab_micro_top1_pct": _pct(
                        h.get("common_full_vocab_micro_top1", 0.0)
                    ),
                    "common_full_vocab_mrr": _num(
                        h.get("common_full_vocab_mrr", 0.0), 4
                    ),
                    "common_full_vocab_median_rank": _num(
                        h.get("common_full_vocab_median_rank", 0.0), 1
                    ),
                    "beneficial_repairs_vocab50_top10": "",
                    "harmful_overwrites_vocab50_top10": "",
                    "net_gain_events_vocab50_top10": "",
                    "ci95_low_pp": "",
                    "ci95_high_pp": "",
                    "p_value": "",
                    "significance": "",
                }
            )
        for pair_key, from_k, to_k in (
            ("K0_to_K1", "K0", "K1"),
            ("K0_to_K2", "K0", "K2"),
        ):
            if from_k in horizons and to_k in horizons:
                h_from = horizons[from_k]
                h_to = horizons[to_k]
                t_pair = (
                    (transitions.get(pair_key) or {}).get("vocab50_top10") or {}
                    if isinstance(transitions, Mapping)
                    else {}
                )
                sig_entry = (
                    sig_map.get(f"{to_k}_minus_{from_k}_vocab50_top10") or {}
                    if isinstance(sig_map, Mapping)
                    else {}
                )
                fv_from_10 = float(
                    h_from.get(
                        "common_full_vocab_word_macro_top10",
                        h_from.get("common_full_vocab_macro_top10", 0.0),
                    )
                )
                fv_to_10 = float(
                    h_to.get(
                        "common_full_vocab_word_macro_top10",
                        h_to.get("common_full_vocab_macro_top10", 0.0),
                    )
                )
                h_rows.append(
                    {
                        "horizon": f"Delta_{to_k}_vs_{from_k}",
                        "label": f"{to_k} - {from_k} Net Gain",
                        "delay_words": int(h_to.get("delay", 0))
                        - int(h_from.get("delay", 0)),
                        "common_vocab50_macro_top10_pct": _pct(
                            float(h_to.get("common_vocab50_macro_top10", 0.0))
                            - float(h_from.get("common_vocab50_macro_top10", 0.0))
                        ),
                        "common_vocab50_micro_top10_pct": _pct(
                            float(h_to.get("common_vocab50_micro_top10", 0.0))
                            - float(h_from.get("common_vocab50_micro_top10", 0.0))
                        ),
                        "common_vocab50_macro_top1_pct": _pct(
                            float(h_to.get("common_vocab50_macro_top1", 0.0))
                            - float(h_from.get("common_vocab50_macro_top1", 0.0))
                        ),
                        "common_vocab50_micro_top1_pct": _pct(
                            float(h_to.get("common_vocab50_micro_top1", 0.0))
                            - float(h_from.get("common_vocab50_micro_top1", 0.0))
                        ),
                        "common_vocab50_mrr": _num(
                            float(h_to.get("common_vocab50_mrr", 0.0))
                            - float(h_from.get("common_vocab50_mrr", 0.0)),
                            4,
                        ),
                        "common_vocab50_median_rank": _num(
                            float(h_to.get("common_vocab50_median_rank", 25.0))
                            - float(h_from.get("common_vocab50_median_rank", 25.0)),
                            1,
                        ),
                        "common_full_vocab_macro_top10_pct": _pct(
                            fv_to_10 - fv_from_10
                        ),
                        "common_full_vocab_micro_top10_pct": _pct(
                            float(h_to.get("common_full_vocab_micro_top10", 0.0))
                            - float(
                                h_from.get("common_full_vocab_micro_top10", 0.0)
                            )
                        ),
                        "common_full_vocab_macro_top1_pct": _pct(
                            float(h_to.get("common_full_vocab_macro_top1", 0.0))
                            - float(h_from.get("common_full_vocab_macro_top1", 0.0))
                        ),
                        "common_full_vocab_micro_top1_pct": _pct(
                            float(h_to.get("common_full_vocab_micro_top1", 0.0))
                            - float(h_from.get("common_full_vocab_micro_top1", 0.0))
                        ),
                        "common_full_vocab_mrr": _num(
                            float(h_to.get("common_full_vocab_mrr", 0.0))
                            - float(h_from.get("common_full_vocab_mrr", 0.0)),
                            4,
                        ),
                        "common_full_vocab_median_rank": _num(
                            float(h_to.get("common_full_vocab_median_rank", 0.0))
                            - float(
                                h_from.get("common_full_vocab_median_rank", 0.0)
                            ),
                            1,
                        ),
                        "beneficial_repairs_vocab50_top10": t_pair.get(
                            "beneficial_repairs", ""
                        ),
                        "harmful_overwrites_vocab50_top10": t_pair.get(
                            "harmful_overwrites", ""
                        ),
                        "net_gain_events_vocab50_top10": t_pair.get(
                            "net_gain_events", ""
                        ),
                        "ci95_low_pp": _pct(sig_entry.get("ci_lower"))
                        if "ci_lower" in sig_entry
                        else "",
                        "ci95_high_pp": _pct(sig_entry.get("ci_upper"))
                        if "ci_upper" in sig_entry
                        else "",
                        "p_value": _num(sig_entry.get("p_two_sided", sig_entry.get("p_one_sided")), 5)
                        if sig_entry
                        else "",
                        "significance": str(sig_entry.get("significance", "")),
                    }
                )
        if h_rows:
            h_csv = _write_csv_table(
                tables_dir / f"revision_horizons_{split}.csv",
                list(h_rows[0].keys()),
                h_rows,
            )
            exported["revision_horizons_csv"] = str(h_csv)

    # 4. 导出负对照汇总表 negative_controls_{split}.csv
    if isinstance(nc, Mapping) and (nc.get("conditions") or nc.get("shifted_aggregate")):
        nc_rows: list[dict[str, Any]] = []
        clean_ref = nc.get("clean_reference") or {}
        c_fv_m10 = float(
            clean_ref.get(
                "full_vocab_word_macro_top10",
                clean_ref.get("full_vocab_macro_top10", 0.0),
            )
        )
        if clean_ref:
            nc_rows.append(
                {
                    "condition_key": "clean_0s",
                    "label": str(clean_ref.get("label", "Clean (0s)")),
                    "control_type": "clean_reference",
                    "shift_seconds": 0.0,
                    "vocab50_macro_top10_pct": _pct(
                        clean_ref.get("vocab50_macro_top10", 0.0)
                    ),
                    "vocab50_micro_top10_pct": _pct(
                        clean_ref.get("vocab50_micro_top10", 0.0)
                    ),
                    "vocab50_macro_top1_pct": _pct(
                        clean_ref.get("vocab50_macro_top1", 0.0)
                    ),
                    "vocab50_micro_top1_pct": _pct(
                        clean_ref.get("vocab50_micro_top1", 0.0)
                    ),
                    "vocab50_mrr": _num(clean_ref.get("vocab50_mrr", 0.0), 4),
                    "vocab50_median_rank": _num(
                        clean_ref.get("vocab50_median_rank", 0.0), 1
                    ),
                    "full_vocab_macro_top10_pct": _pct(c_fv_m10),
                    "full_vocab_micro_top10_pct": _pct(
                        clean_ref.get("full_vocab_micro_top10", 0.0)
                    ),
                    "full_vocab_macro_top1_pct": _pct(
                        clean_ref.get("full_vocab_macro_top1", 0.0)
                    ),
                    "full_vocab_micro_top1_pct": _pct(
                        clean_ref.get("full_vocab_micro_top1", 0.0)
                    ),
                    "full_vocab_mrr": _num(clean_ref.get("full_vocab_mrr", 0.0), 4),
                    "full_vocab_median_rank": _num(
                        clean_ref.get("full_vocab_median_rank", 0.0), 1
                    ),
                    "delta_vocab50_top10_vs_clean_pp": 0.0,
                    "delta_full_vocab_top10_vs_clean_pp": 0.0,
                    "ci95_low_pp": "",
                    "ci95_high_pp": "",
                    "p_value": "",
                    "significance": "",
                }
            )
        for cond in nc.get("conditions") or ():
            s_row = cond.get("summary") or {}
            d_clean = s_row.get("delta_vs_clean") or {}
            s_fv_m10 = float(
                s_row.get(
                    "full_vocab_word_macro_top10",
                    s_row.get("full_vocab_macro_top10", 0.0),
                )
            )
            nc_rows.append(
                {
                    "condition_key": str(cond.get("key", "")),
                    "label": str(cond.get("label", "")),
                    "control_type": str(cond.get("control_type", "temporal_shift")),
                    "shift_seconds": cond.get("shift_seconds", ""),
                    "vocab50_macro_top10_pct": _pct(
                        s_row.get("vocab50_macro_top10", 0.0)
                    ),
                    "vocab50_micro_top10_pct": _pct(
                        s_row.get("vocab50_micro_top10", 0.0)
                    ),
                    "vocab50_macro_top1_pct": _pct(
                        s_row.get("vocab50_macro_top1", 0.0)
                    ),
                    "vocab50_micro_top1_pct": _pct(
                        s_row.get("vocab50_micro_top1", 0.0)
                    ),
                    "vocab50_mrr": _num(s_row.get("vocab50_mrr", 0.0), 4),
                    "vocab50_median_rank": _num(
                        s_row.get("vocab50_median_rank", 0.0), 1
                    ),
                    "full_vocab_macro_top10_pct": _pct(s_fv_m10),
                    "full_vocab_micro_top10_pct": _pct(
                        s_row.get("full_vocab_micro_top10", 0.0)
                    ),
                    "full_vocab_macro_top1_pct": _pct(
                        s_row.get("full_vocab_macro_top1", 0.0)
                    ),
                    "full_vocab_micro_top1_pct": _pct(
                        s_row.get("full_vocab_micro_top1", 0.0)
                    ),
                    "full_vocab_mrr": _num(s_row.get("full_vocab_mrr", 0.0), 4),
                    "full_vocab_median_rank": _num(
                        s_row.get("full_vocab_median_rank", 0.0), 1
                    ),
                    "delta_vocab50_top10_vs_clean_pp": _pct(
                        d_clean.get("vocab50_macro_top10", 0.0)
                    ),
                    "delta_full_vocab_top10_vs_clean_pp": _pct(
                        d_clean.get(
                            "full_vocab_word_macro_top10",
                            d_clean.get("full_vocab_macro_top10", s_fv_m10 - c_fv_m10),
                        )
                    ),
                    "ci95_low_pp": "",
                    "ci95_high_pp": "",
                    "p_value": "",
                    "significance": "",
                }
            )
        sh_agg = nc.get("shifted_aggregate") or {}
        if sh_agg:
            nc_sig = sh_agg.get("statistical_significance") or {}
            v50_sig = nc_sig.get("clean_minus_shifted_vocab50_top10") or {}
            sh_fv_m10 = float(
                sh_agg.get(
                    "mean_full_vocab_word_macro_top10",
                    sh_agg.get("mean_full_vocab_macro_top10", 0.0),
                )
            )
            nc_rows.append(
                {
                    "condition_key": "shifted_mean",
                    "label": f"Shifted Mean (M={sh_agg.get('shift_count', 0)})",
                    "control_type": "shifted_aggregate",
                    "shift_seconds": "",
                    "vocab50_macro_top10_pct": _pct(
                        sh_agg.get("mean_vocab50_macro_top10", 0.0)
                    ),
                    "vocab50_micro_top10_pct": _pct(
                        sh_agg.get("mean_vocab50_micro_top10", 0.0)
                    ),
                    "vocab50_macro_top1_pct": _pct(
                        sh_agg.get("mean_vocab50_macro_top1", 0.0)
                    ),
                    "vocab50_micro_top1_pct": _pct(
                        sh_agg.get("mean_vocab50_micro_top1", 0.0)
                    ),
                    "vocab50_mrr": _num(sh_agg.get("mean_vocab50_mrr", 0.0), 4),
                    "vocab50_median_rank": _num(
                        sh_agg.get("mean_vocab50_median_rank", 0.0), 1
                    ),
                    "full_vocab_macro_top10_pct": _pct(sh_fv_m10),
                    "full_vocab_micro_top10_pct": _pct(
                        sh_agg.get("mean_full_vocab_micro_top10", 0.0)
                    ),
                    "full_vocab_macro_top1_pct": _pct(
                        sh_agg.get("mean_full_vocab_macro_top1", 0.0)
                    ),
                    "full_vocab_micro_top1_pct": _pct(
                        sh_agg.get("mean_full_vocab_micro_top1", 0.0)
                    ),
                    "full_vocab_mrr": _num(
                        sh_agg.get("mean_full_vocab_mrr", 0.0), 4
                    ),
                    "full_vocab_median_rank": _num(
                        sh_agg.get("mean_full_vocab_median_rank", 0.0), 1
                    ),
                    "delta_vocab50_top10_vs_clean_pp": _pct(
                        sh_agg.get("delta_vocab50_macro_top10_vs_clean", 0.0)
                    ),
                    "delta_full_vocab_top10_vs_clean_pp": _pct(
                        sh_agg.get(
                            "delta_full_vocab_word_macro_top10_vs_clean",
                            sh_agg.get(
                                "delta_full_vocab_macro_top10_vs_clean",
                                sh_fv_m10 - c_fv_m10,
                            ),
                        )
                    ),
                    "ci95_low_pp": _pct(v50_sig.get("ci_lower"))
                    if "ci_lower" in v50_sig
                    else "",
                    "ci95_high_pp": _pct(v50_sig.get("ci_upper"))
                    if "ci_upper" in v50_sig
                    else "",
                    "p_value": _num(
                        v50_sig.get("p_two_sided", v50_sig.get("p_one_sided")), 5
                    )
                    if v50_sig
                    else "",
                    "significance": str(v50_sig.get("significance", "")),
                }
            )
        if nc_rows:
            nc_csv = _write_csv_table(
                tables_dir / f"negative_controls_{split}.csv",
                list(nc_rows[0].keys()),
                nc_rows,
            )
            exported["negative_controls_csv"] = str(nc_csv)

    # 5. 导出受试者级明细与汇总表 per_subject_metrics_{split}.csv
    if isinstance(per_subj, Sequence) and per_subj:
        ps_rows: list[dict[str, Any]] = []
        for s_item in per_subj:
            if not isinstance(s_item, Mapping):
                continue
            ps_rows.append(
                {
                    "subject_id": str(s_item.get("subject_id", "-")),
                    "total_events": int(s_item.get("total_events", 0)),
                    "vocab50_queries": int(s_item.get("vocab50_queries", 0)),
                    "vocab50_macro_top10_pct": _pct(s_item.get("macro_top10", 0.0)),
                    "vocab50_micro_top10_pct": _pct(s_item.get("micro_top10", 0.0)),
                    "vocab50_macro_top1_pct": _pct(s_item.get("macro_top1", 0.0)),
                    "vocab50_micro_top1_pct": _pct(s_item.get("micro_top1", 0.0)),
                    "vocab50_mrr": _num(s_item.get("mrr", 0.0), 4),
                    "vocab50_median_rank": _num(
                        s_item.get("median_rank", 25.0), 1
                    ),
                    "full_vocab_macro_top10_pct": _pct(
                        s_item.get(
                            "full_vocab_macro_top10",
                            s_item.get("full_vocab_top10", 0.0),
                        )
                    ),
                    "full_vocab_micro_top10_pct": _pct(
                        s_item.get(
                            "full_vocab_micro_top10",
                            s_item.get("full_vocab_top10", 0.0),
                        )
                    ),
                    "full_vocab_macro_top1_pct": _pct(
                        s_item.get(
                            "full_vocab_macro_top1",
                            s_item.get("full_vocab_top1", 0.0),
                        )
                    ),
                    "full_vocab_micro_top1_pct": _pct(
                        s_item.get(
                            "full_vocab_micro_top1",
                            s_item.get("full_vocab_top1", 0.0),
                        )
                    ),
                    "full_vocab_mrr": _num(s_item.get("full_vocab_mrr", 0.0), 4),
                    "full_vocab_median_rank": _num(
                        s_item.get("full_vocab_median_rank", 0.0), 1
                    ),
                    "k0_vocab50_top10_pct": _pct(
                        s_item.get("k0_vocab50_top10", 0.0)
                    )
                    if "k0_vocab50_top10" in s_item
                    else "",
                    "k2_vocab50_top10_pct": _pct(
                        s_item.get("k2_vocab50_top10", 0.0)
                    )
                    if "k2_vocab50_top10" in s_item
                    else "",
                    "k2_minus_k0_vocab50_top10_pp": _pct(
                        s_item.get("k2_minus_k0_vocab50_top10", 0.0)
                    )
                    if "k2_minus_k0_vocab50_top10" in s_item
                    else "",
                    "k0_full_vocab_macro_top10_pct": _pct(
                        s_item.get("k0_full_vocab_macro_top10", 0.0)
                    )
                    if "k0_full_vocab_macro_top10" in s_item
                    else "",
                    "k2_full_vocab_macro_top10_pct": _pct(
                        s_item.get("k2_full_vocab_macro_top10", 0.0)
                    )
                    if "k2_full_vocab_macro_top10" in s_item
                    else "",
                    "k2_minus_k0_full_vocab_macro_top10_pp": _pct(
                        s_item.get("k2_minus_k0_full_vocab_macro_top10", 0.0)
                    )
                    if "k2_minus_k0_full_vocab_macro_top10" in s_item
                    else "",
                }
            )
        if ps_rows and len(ps_rows) >= 2:
            numeric_cols = (
                "vocab50_macro_top10_pct",
                "vocab50_micro_top10_pct",
                "vocab50_macro_top1_pct",
                "vocab50_micro_top1_pct",
                "vocab50_mrr",
                "vocab50_median_rank",
                "full_vocab_macro_top10_pct",
                "full_vocab_micro_top10_pct",
                "full_vocab_macro_top1_pct",
                "full_vocab_micro_top1_pct",
                "full_vocab_mrr",
                "full_vocab_median_rank",
            )
            mean_row: dict[str, Any] = {
                "subject_id": "MEAN",
                "total_events": int(
                    sum(int(r["total_events"]) for r in ps_rows)
                ),
                "vocab50_queries": int(
                    sum(int(r["vocab50_queries"]) for r in ps_rows)
                ),
            }
            sd_row: dict[str, Any] = {
                "subject_id": "SD",
                "total_events": "",
                "vocab50_queries": "",
            }
            for col in numeric_cols:
                arr = np.asarray(
                    [float(r[col]) for r in ps_rows if r.get(col) != ""],
                    dtype=np.float64,
                )
                digits = 4 if "mrr" in col else (1 if "rank" in col else 2)
                mean_row[col] = (
                    round(float(np.mean(arr)), digits) if len(arr) else ""
                )
                sd_row[col] = (
                    round(float(np.std(arr, ddof=0)), digits)
                    if len(arr)
                    else ""
                )
            for opt_col in (
                "k0_vocab50_top10_pct",
                "k2_vocab50_top10_pct",
                "k2_minus_k0_vocab50_top10_pp",
                "k0_full_vocab_macro_top10_pct",
                "k2_full_vocab_macro_top10_pct",
                "k2_minus_k0_full_vocab_macro_top10_pp",
            ):
                vals = [
                    float(r[opt_col])
                    for r in ps_rows
                    if r.get(opt_col) not in ("", None)
                ]
                mean_row[opt_col] = (
                    round(float(np.mean(vals)), 2) if vals else ""
                )
                sd_row[opt_col] = (
                    round(float(np.std(vals, ddof=0)), 2) if vals else ""
                )
            ps_rows.append(mean_row)
            ps_rows.append(sd_row)

        if ps_rows:
            ps_csv = _write_csv_table(
                tables_dir / f"per_subject_metrics_{split}.csv",
                list(ps_rows[0].keys()),
                ps_rows,
            )
            exported["per_subject_metrics_csv"] = str(ps_csv)

    # 6. 生成结构化 Markdown 报告 evaluation_report_{split}.md
    md_lines: list[str] = [
        f"# Word-Level Retrieval Evaluation Report (`{dataset}` | `{stage}` | `{split}`)",
        "",
        f"- **Dataset**: `{dataset}` | **Stage**: `{stage_disp}` | **Split**: `{split}` | **Seed**: `{seed}` (Best Epoch: `{ckpt_epoch}`)",
        f"- **Signal Window**: `{win_sec:.2f}s` (Offset: `{win_off:+.2f}s`) | **Evaluated Events**: `{metrics.get('evaluated_events', '-')}` / `{metrics.get('total_events', '-')}`",
        "",
    ]

    n50_d = sweep.get("N50") or {}
    fv_d = sweep.get("full_vocab") or {}
    if n50_d:
        n50_top10 = float(n50_d.get("macro_top10", 0.0))
        n50_chance = float(n50_d.get("chance_top10", 0.20))
        n50_fold = n50_top10 / max(n50_chance, 1e-12)
        fv_top10 = float(
            fv_d.get("word_macro_top10", fv_d.get("macro_top10", 0.0))
        )
        fv_micro10 = float(fv_d.get("micro_top10", 0.0))
        fv_chance = float(fv_d.get("chance_top10", 0.005))
        fv_fold = fv_top10 / max(fv_chance, 1e-12)
        md_lines.extend(
            [
                "> [!IMPORTANT]",
                f"> **Executive Headline**: Top-50 Vocab Macro Top-10 = **{_fmt_pct(n50_top10)}** (`{n50_fold:.2f}x` chance `{_fmt_pct(n50_chance)}`), "
                f"Micro Top-10 = **{_fmt_pct(n50_d.get('micro_top10'))}**, Top-1 = **{_fmt_pct(n50_d.get('macro_top1'))}** | "
                f"Full-Vocab (`|V|={fv_d.get('candidate_count', '-')}`) Macro Top-10 = **{_fmt_pct(fv_top10)}** (`{fv_fold:.2f}x` chance `{_fmt_pct(fv_chance)}`), "
                f"Micro Top-10 = **{_fmt_pct(fv_micro10)}**, Macro Top-1 = **{_fmt_pct(fv_d.get('macro_top1', 0.0))}**.",
                "",
            ]
        )

    if vocab_rows:
        md_lines.extend(
            [
                "## 1. Multi-Scale Vocabulary Retrieval Metrics",
                "",
                "| Scale | OOV Policy | Candidates | Queries (Cov%) | Macro@10 | Micro@10 | Subj-Macro@10 | Chance@10 (Fold) | Macro@1 | Micro@1 | MRR | Median Rank (Chance) |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for vr in vocab_rows:
            md_lines.append(
                f"| **{vr['label']}** (`{vr['scale_key']}`) | `{vr['oov_policy']}` | {vr['candidate_count']} | "
                f"{vr['query_count']} ({vr['in_vocabulary_coverage_pct']:.1f}%) | "
                f"**{vr['macro_top10_pct']:.2f}%** | {vr['micro_top10_pct']:.2f}% | "
                f"{vr['subject_macro_top10_pct']:.2f}% | {vr['chance_top10_pct']:.2f}% (**{vr['fold_over_chance_top10']:.2f}x**) | "
                f"{vr['macro_top1_pct']:.2f}% | {vr['micro_top1_pct']:.2f}% | {vr['mrr']:.4f} | "
                f"{vr['median_rank']:.1f} ({vr['chance_median_rank']:.1f}) |"
            )
        md_lines.append("")

    if isinstance(horizons, Mapping) and horizons:
        md_lines.extend(
            [
                "## 2. Rolling Revision Horizons (`K0 -> K1 -> K2`) & Transition Diagnostics",
                "",
                "| Horizon | Delay | N50 Macro@10 | N50 Micro@10 | N50 Macro@1 | Full-Vocab Macro@10 | Full-Vocab Micro@10 | Full-Vocab Macro@1 | Full-Vocab MRR |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for h_k in ("K0", "K1", "K2"):
            if h_k not in horizons:
                continue
            hd = horizons[h_k]
            hd_fv_m10 = hd.get(
                "common_full_vocab_word_macro_top10",
                hd.get("common_full_vocab_macro_top10"),
            )
            md_lines.append(
                f"| **{hd.get('label', h_k)}** | `+{hd.get('delay', 0)}w` | "
                f"**{_fmt_pct(hd.get('common_vocab50_macro_top10'))}** | {_fmt_pct(hd.get('common_vocab50_micro_top10'))} | "
                f"{_fmt_pct(hd.get('common_vocab50_macro_top1'))} | **{_fmt_pct(hd_fv_m10)}** | "
                f"{_fmt_pct(hd.get('common_full_vocab_micro_top10'))} | "
                f"{_fmt_pct(hd.get('common_full_vocab_macro_top1'))} | "
                f"{_num(hd.get('common_full_vocab_mrr', 0.0), 4):.4f} |"
            )
        if "K0" in horizons and "K2" in horizons:
            k0_d, k2_d = horizons["K0"], horizons["K2"]
            k0_fv_m10 = float(
                k0_d.get(
                    "common_full_vocab_word_macro_top10",
                    k0_d.get("common_full_vocab_macro_top10", 0.0),
                )
            )
            k2_fv_m10 = float(
                k2_d.get(
                    "common_full_vocab_word_macro_top10",
                    k2_d.get("common_full_vocab_macro_top10", 0.0),
                )
            )
            sig_k2 = sig_map.get("K2_minus_K0_vocab50_top10") or {}
            ci_str = (
                f"[{_pct(sig_k2.get('ci_lower')):+.2f}, {_pct(sig_k2.get('ci_upper')):+.2f}] pp "
                f"(p={_num(sig_k2.get('p_two_sided', sig_k2.get('p_one_sided')), 4):.4f} {sig_k2.get('significance', '')})"
                if sig_k2
                else ""
            )
            md_lines.append(
                f"| **Δ (K2 - K0)** | `+2w` | "
                f"**{_fmt_signed_pct(float(k2_d.get('common_vocab50_macro_top10', 0.0)) - float(k0_d.get('common_vocab50_macro_top10', 0.0)))}** {ci_str} | "
                f"{_fmt_signed_pct(float(k2_d.get('common_vocab50_micro_top10', 0.0)) - float(k0_d.get('common_vocab50_micro_top10', 0.0)))} | "
                f"{_fmt_signed_pct(float(k2_d.get('common_vocab50_macro_top1', 0.0)) - float(k0_d.get('common_vocab50_macro_top1', 0.0)))} | "
                f"**{_fmt_signed_pct(k2_fv_m10 - k0_fv_m10)}** | "
                f"{_fmt_signed_pct(float(k2_d.get('common_full_vocab_micro_top10', 0.0)) - float(k0_d.get('common_full_vocab_micro_top10', 0.0)))} | "
                f"{_fmt_signed_pct(float(k2_d.get('common_full_vocab_macro_top1', 0.0)) - float(k0_d.get('common_full_vocab_macro_top1', 0.0)))} | "
                f"{float(k2_d.get('common_full_vocab_mrr', 0.0)) - float(k0_d.get('common_full_vocab_mrr', 0.0)):+.4f} |"
            )
        md_lines.append("")

        if isinstance(transitions, Mapping) and transitions:
            md_lines.extend(
                [
                    "### Event-Level Revision Transition Matrix (`Miss -> Hit` vs. `Hit -> Miss`)",
                    "",
                    "| Transition | Criterion | Common Events | Beneficial Repairs (`Miss->Hit`) | Harmful Overwrites (`Hit->Miss`) | Retained Hits (`Hit->Hit`) | Net Gain Events (Rate) | Repair / Harm Ratio |",
                    "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for t_key, t_label in (
                ("K0_to_K1", "K0 -> K1 (+1w)"),
                ("K0_to_K2", "K0 -> K2 (+2w)"),
                ("K1_to_K2", "K1 -> K2 (+1w)"),
            ):
                t_group = transitions.get(t_key)
                if not isinstance(t_group, Mapping):
                    continue
                for crit_key, crit_label in (
                    ("vocab50_top10", "N50 Top-10"),
                    ("full_vocab_top10", "Full-Vocab Top-10"),
                ):
                    td = t_group.get(crit_key)
                    if not isinstance(td, Mapping):
                        continue
                    md_lines.append(
                        f"| **{t_label}** | `{crit_label}` | {td.get('total_common_events', 0)} | "
                        f"**+{td.get('beneficial_repairs', 0)}** ({_num(td.get('beneficial_repair_rate_pct', 0.0), 2):.2f}%) | "
                        f"-{td.get('harmful_overwrites', 0)} ({_num(td.get('harmful_overwrite_rate_pct', 0.0), 2):.2f}%) | "
                        f"{td.get('retained_hits', 0)} | "
                        f"**{int(td.get('net_gain_events', 0)):+d}** ({_num(td.get('net_gain_pct', 0.0), 2):+.2f}%) | "
                        f"`{_num(td.get('repair_to_harm_ratio', 0.0), 2):.2f}x` |"
                    )
            md_lines.append("")

    if isinstance(ling, Mapping) and ling:
        md_lines.extend(
            [
                "## 3. Content Words vs. Function Words Breakdown",
                "",
                "| Word Class | Events (Share) | Unique Words | N50 Macro@10 | N50 Micro@10 | N50 Macro@1 | Full-Vocab Macro@10 | Full-Vocab Micro@10 | Full-Vocab MRR |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cat_k in ("content_words", "function_words"):
            cd = ling.get(cat_k)
            if not isinstance(cd, Mapping):
                continue
            md_lines.append(
                f"| **{cd.get('label', cat_k)}** | {cd.get('query_count', 0)} ({_num(cd.get('event_share_pct', 0.0), 1):.1f}%) | "
                f"{cd.get('unique_word_count', 0)} | **{_fmt_pct(cd.get('vocab50_word_macro_top10', cd.get('vocab50_macro_top10')))}** | "
                f"{_fmt_pct(cd.get('vocab50_micro_top10'))} | {_fmt_pct(cd.get('vocab50_word_macro_top1', cd.get('vocab50_macro_top1')))} | "
                f"**{_fmt_pct(cd.get('full_vocab_word_macro_top10', cd.get('full_vocab_macro_top10')))}** | {_fmt_pct(cd.get('full_vocab_micro_top10'))} | "
                f"{_num(cd.get('full_vocab_mrr', 0.0), 4):.4f} |"
            )
        md_lines.append("")

    if isinstance(nc, Mapping) and (nc.get("conditions") or nc.get("shifted_aggregate")):
        md_lines.extend(
            [
                "## 4. Negative Controls & Neural Specificity (`Clean vs. Shifted`)",
                "",
                "| Condition | Shift (s) | N50 Macro@10 | Δ vs Clean | N50 Micro@10 | N50 Macro@1 | Full-Vocab Macro@10 | Δ Full-Vocab |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        clean_ref = nc.get("clean_reference") or {}
        if clean_ref:
            c_fv_m10 = clean_ref.get(
                "full_vocab_word_macro_top10",
                clean_ref.get("full_vocab_macro_top10"),
            )
            md_lines.append(
                f"| **Clean (0s)** | `0.0s` | **{_fmt_pct(clean_ref.get('vocab50_macro_top10'))}** | `0.00 pp` | "
                f"{_fmt_pct(clean_ref.get('vocab50_micro_top10'))} | {_fmt_pct(clean_ref.get('vocab50_macro_top1'))} | "
                f"**{_fmt_pct(c_fv_m10)}** | `0.00 pp` |"
            )
        for cond in nc.get("conditions") or ():
            sr = cond.get("summary") or {}
            dc = sr.get("delta_vs_clean") or {}
            s_val = cond.get("shift_seconds")
            s_str = f"`{float(s_val):+.0f}s`" if s_val is not None else "`N/A`"
            sr_fv_m10 = sr.get(
                "full_vocab_word_macro_top10",
                sr.get("full_vocab_macro_top10"),
            )
            dc_fv_m10 = dc.get(
                "full_vocab_word_macro_top10",
                dc.get("full_vocab_macro_top10"),
            )
            md_lines.append(
                f"| {cond.get('label', cond.get('key', '-'))} | {s_str} | "
                f"{_fmt_pct(sr.get('vocab50_macro_top10'))} | {_fmt_signed_pct(dc.get('vocab50_macro_top10'))} | "
                f"{_fmt_pct(sr.get('vocab50_micro_top10'))} | {_fmt_pct(sr.get('vocab50_macro_top1'))} | "
                f"{_fmt_pct(sr_fv_m10)} | {_fmt_signed_pct(dc_fv_m10)} |"
            )
        sh_agg = nc.get("shifted_aggregate") or {}
        if sh_agg:
            nc_sig = (sh_agg.get("statistical_significance") or {}).get(
                "clean_minus_shifted_vocab50_top10"
            ) or {}
            sig_note = (
                f" (95% CI `[{_pct(nc_sig.get('ci_lower')):+.2f}, {_pct(nc_sig.get('ci_upper')):+.2f}] pp`, "
                f"`p={_num(nc_sig.get('p_two_sided', nc_sig.get('p_one_sided')), 4):.4f}` {nc_sig.get('significance', '')})"
                if nc_sig
                else ""
            )
            sh_fv_m10 = sh_agg.get(
                "mean_full_vocab_word_macro_top10",
                sh_agg.get("mean_full_vocab_macro_top10"),
            )
            sh_fv_delta = sh_agg.get(
                "delta_full_vocab_word_macro_top10_vs_clean",
                sh_agg.get("delta_full_vocab_macro_top10_vs_clean"),
            )
            md_lines.append(
                f"| **Shifted Mean ± SD** | `M={sh_agg.get('shift_count', 0)}` | "
                f"**{_fmt_pct(sh_agg.get('mean_vocab50_macro_top10'))} ± {_fmt_pct(sh_agg.get('std_vocab50_macro_top10'))}** | "
                f"**{_fmt_signed_pct(sh_agg.get('delta_vocab50_macro_top10_vs_clean'))}**{sig_note} | "
                f"{_fmt_pct(sh_agg.get('mean_vocab50_micro_top10'))} | {_fmt_pct(sh_agg.get('mean_vocab50_macro_top1'))} | "
                f"**{_fmt_pct(sh_fv_m10)}** | "
                f"{_fmt_signed_pct(sh_fv_delta)} |"
            )
        md_lines.append("")

    if isinstance(per_subj, Sequence) and per_subj:
        has_k2_k0 = any("k2_minus_k0_vocab50_top10" in s for s in per_subj if isinstance(s, Mapping))
        md_lines.append("## 5. Per-Subject Performance (`Mean ± SD`)")
        md_lines.append("")
        if isinstance(subj_sum, Mapping) and subj_sum:
            imp_cnt = subj_sum.get("improved_subjects_count")
            tot_cnt = subj_sum.get("subject_count", len(per_subj))
            imp_str = (
                f" | **Revision Improved Subjects**: `{imp_cnt}/{tot_cnt}` (`{_num(subj_sum.get('improved_subjects_pct', 0.0), 1):.1f}%`)"
                if imp_cnt is not None
                else ""
            )
            md_lines.append(
                f"- **Across-Subject N50 Macro@10**: `{_fmt_pct(subj_sum.get('mean_macro_top10'))} ± {_fmt_pct(subj_sum.get('std_macro_top10'))}` "
                f"(Range: `{_fmt_pct(subj_sum.get('min_macro_top10'))} ~ {_fmt_pct(subj_sum.get('max_macro_top10'))}`) | "
                f"**Full-Vocab Micro@10**: `{_fmt_pct(subj_sum.get('mean_full_vocab_micro_top10', subj_sum.get('mean_full_vocab_top10')))} ± {_fmt_pct(subj_sum.get('std_full_vocab_micro_top10', subj_sum.get('std_full_vocab_top10')))}`{imp_str}"
            )
            md_lines.append("")
        if has_k2_k0:
            md_lines.append(
                "| Subject | Events (N50) | N50 Macro@10 | N50 Micro@10 | N50 Macro@1 | N50 MRR | Full-Vocab Macro@10 | Full-Vocab Micro@10 | K0 N50@10 | K2 N50@10 | Δ (K2 - K0) |"
            )
            md_lines.append(
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            for s in per_subj:
                if not isinstance(s, Mapping):
                    continue
                md_lines.append(
                    f"| `{s.get('subject_id', '-')}` | {s.get('total_events', 0)} ({s.get('vocab50_queries', 0)}) | "
                    f"**{_fmt_pct(s.get('macro_top10'))}** | {_fmt_pct(s.get('micro_top10'))} | "
                    f"{_fmt_pct(s.get('macro_top1'))} | {_num(s.get('mrr', 0.0), 4):.4f} | "
                    f"{_fmt_pct(s.get('full_vocab_macro_top10', s.get('full_vocab_top10', 0.0)))} | "
                    f"{_fmt_pct(s.get('full_vocab_micro_top10', s.get('full_vocab_top10', 0.0)))} | {_fmt_pct(s.get('k0_vocab50_top10'))} | "
                    f"{_fmt_pct(s.get('k2_vocab50_top10'))} | **{_fmt_signed_pct(s.get('k2_minus_k0_vocab50_top10'))}** |"
                )
        else:
            md_lines.append(
                "| Subject | Events (N50) | N50 Macro@10 | N50 Micro@10 | N50 Macro@1 | N50 MRR | N50 MedRank | Full-Vocab Macro@10 | Full-Vocab Micro@10 | Full-Vocab MRR |"
            )
            md_lines.append(
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            )
            for s in per_subj:
                if not isinstance(s, Mapping):
                    continue
                md_lines.append(
                    f"| `{s.get('subject_id', '-')}` | {s.get('total_events', 0)} ({s.get('vocab50_queries', 0)}) | "
                    f"**{_fmt_pct(s.get('macro_top10'))}** | {_fmt_pct(s.get('micro_top10'))} | "
                    f"{_fmt_pct(s.get('macro_top1'))} | {_num(s.get('mrr', 0.0), 4):.4f} | "
                    f"{_num(s.get('median_rank', 25.0), 1):.1f} | "
                    f"**{_fmt_pct(s.get('full_vocab_macro_top10', s.get('full_vocab_top10', 0.0)))}** | "
                    f"{_fmt_pct(s.get('full_vocab_micro_top10', s.get('full_vocab_top10', 0.0)))} | "
                    f"{_num(s.get('full_vocab_mrr', 0.0), 4):.4f} |"
                )
        md_lines.append("")

    md_path = out_dir / f"evaluation_report_{split}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    exported["evaluation_report_md"] = str(md_path)
    return exported


def export_stage_comparison_tables_and_markdown(
    multi_summary: Mapping[str, Any],
    output_dir: str | Path | None = None,
    *,
    split: str | None = None,
) -> dict[str, str]:
    """将三阶段对比评估结果导出为 tables/stage_comparison_{split}.csv 与 stage_comparison_report_{split}.md。"""
    target_dir_raw = output_dir or multi_summary.get("output_dir")
    if not target_dir_raw:
        return {}
    out_dir = Path(str(target_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    stages_map = (
        multi_summary["stages"]
        if "stages" in multi_summary and isinstance(multi_summary["stages"], Mapping)
        else multi_summary
    )
    first_st = next(iter(stages_map.values()), {})
    resolved_split = str(
        split or multi_summary.get("split") or first_st.get("split", "val")
    )
    dataset = str(multi_summary.get("dataset") or first_st.get("dataset", "-"))
    seed = multi_summary.get("seed", first_st.get("seed", 0))
    stage_sig = multi_summary.get("stage_significance") or {}

    rows: list[dict[str, Any]] = []
    word_res = stages_map.get("word")
    w_n50 = (
        (word_res.get("metrics") or {}).get("vocabulary_sweep", {}).get("N50", {})
        if isinstance(word_res, Mapping)
        else {}
    )
    w_fv = (
        (word_res.get("metrics") or {}).get("vocabulary_sweep", {}).get("full_vocab", {})
        if isinstance(word_res, Mapping)
        else {}
    )
    w_fv_m10 = float(w_fv.get("word_macro_top10", w_fv.get("macro_top10", 0.0)))

    for st_key in ("word", "context", "revision", "prefix_revision"):
        if st_key not in stages_map:
            continue
        res = stages_map[st_key]
        if not isinstance(res, Mapping):
            continue
        m = res.get("metrics") or {}
        sw = m.get("vocabulary_sweep") or {}
        n50 = sw.get("N50") or {}
        fv = sw.get("full_vocab") or {}
        fv_m10 = float(fv.get("word_macro_top10", fv.get("macro_top10", 0.0)))
        nc = res.get("negative_controls") or {}
        sh_agg = nc.get("shifted_aggregate") or {}
        sig_vs_w = (
            stage_sig.get(f"{st_key}_minus_word_vocab50_top10")
            or stage_sig.get(f"{st_key}_vs_word_n50_top10")
            or {}
        )
        rows.append(
            {
                "stage": st_key,
                "stage_display_name": str(res.get("stage_display_name", st_key)),
                "best_epoch": (res.get("checkpoint") or {}).get("epoch", 0)
                if isinstance(res.get("checkpoint"), Mapping)
                else 0,
                "vocab50_macro_top10_pct": _pct(n50.get("macro_top10", 0.0)),
                "delta_vs_word_vocab50_top10_pp": _pct(
                    float(n50.get("macro_top10", 0.0))
                    - float(w_n50.get("macro_top10", 0.0))
                )
                if w_n50
                else 0.0,
                "vocab50_micro_top10_pct": _pct(n50.get("micro_top10", 0.0)),
                "vocab50_macro_top1_pct": _pct(n50.get("macro_top1", 0.0)),
                "vocab50_micro_top1_pct": _pct(n50.get("micro_top1", 0.0)),
                "vocab50_mrr": _num(n50.get("mrr", 0.0), 4),
                "vocab50_median_rank": _num(n50.get("median_rank", 25.0), 1),
                "full_vocab_macro_top10_pct": _pct(fv_m10),
                "delta_vs_word_full_vocab_top10_pp": _pct(fv_m10 - w_fv_m10)
                if w_fv
                else 0.0,
                "full_vocab_micro_top10_pct": _pct(fv.get("micro_top10", 0.0)),
                "full_vocab_macro_top1_pct": _pct(fv.get("macro_top1", 0.0)),
                "full_vocab_micro_top1_pct": _pct(fv.get("micro_top1", 0.0)),
                "full_vocab_mrr": _num(fv.get("mrr", 0.0), 4),
                "full_vocab_median_rank": _num(fv.get("median_rank", 0.0), 1),
                "shifted_vocab50_macro_top10_pct": _pct(
                    sh_agg.get("mean_vocab50_macro_top10", 0.0)
                )
                if sh_agg
                else "",
                "clean_minus_shifted_vocab50_top10_pp": _pct(
                    sh_agg.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
                )
                if sh_agg
                else "",
                "ci95_low_vs_word_pp": _pct(sig_vs_w.get("ci_lower"))
                if "ci_lower" in sig_vs_w
                else "",
                "ci95_high_vs_word_pp": _pct(sig_vs_w.get("ci_upper"))
                if "ci_upper" in sig_vs_w
                else "",
                "p_value_vs_word": _num(
                    sig_vs_w.get("p_two_sided", sig_vs_w.get("p_one_sided")), 5
                )
                if sig_vs_w
                else "",
                "significance_vs_word": str(sig_vs_w.get("significance", "")),
            }
        )

    exported: dict[str, str] = {}
    if rows:
        csv_path = _write_csv_table(
            tables_dir / f"stage_comparison_{resolved_split}.csv",
            list(rows[0].keys()),
            rows,
        )
        exported["stage_comparison_csv"] = str(csv_path)

        md_lines = [
            f"# Cross-Stage Comparison Report (`{dataset}` | `{resolved_split}` | `seed={seed}`)",
            "",
            "| Stage | Best Epoch | N50 Macro@10 | Δ vs Stage 1 (95% CI / p) | N50 Micro@10 | N50 Macro@1 | Full-Vocab Macro@10 | Full-Vocab Micro@10 | Δ Full-Vocab | Clean - Shifted |",
            "| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in rows:
            sig_txt = (
                f" (`[{r['ci95_low_vs_word_pp']:+.2f}, {r['ci95_high_vs_word_pp']:+.2f}]`, `p={r['p_value_vs_word']}` {r['significance_vs_word']})"
                if r["p_value_vs_word"] != ""
                else ""
            )
            sh_txt = (
                f"{r['clean_minus_shifted_vocab50_top10_pp']:+.2f} pp"
                if r["clean_minus_shifted_vocab50_top10_pp"] != ""
                else "-"
            )
            md_lines.append(
                f"| **{r['stage_display_name']}** (`{r['stage']}`) | {r['best_epoch']} | "
                f"**{r['vocab50_macro_top10_pct']:.2f}%** | {r['delta_vs_word_vocab50_top10_pp']:+.2f} pp{sig_txt} | "
                f"{r['vocab50_micro_top10_pct']:.2f}% | {r['vocab50_macro_top1_pct']:.2f}% | "
                f"**{r['full_vocab_macro_top10_pct']:.2f}%** | {r['full_vocab_micro_top10_pct']:.2f}% | "
                f"{r['delta_vs_word_full_vocab_top10_pp']:+.2f} pp | {sh_txt} |"
            )
        md_path = out_dir / f"stage_comparison_report_{resolved_split}.md"
        md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        exported["stage_comparison_md"] = str(md_path)
    return exported


def export_multi_seed_tables_and_markdown(
    summary: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """将多种子聚合评估结果导出为 tables/multi_seed_summary_{split}.csv 与 multi_seed_evaluation_report_{split}.md。"""
    target_dir_raw = output_dir or summary.get("output_dir")
    if not target_dir_raw:
        return {}
    out_dir = Path(str(target_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    split = str(summary.get("split", "val"))
    dataset = str(summary.get("dataset", "-"))
    seeds = summary.get("seeds") or []
    exported: dict[str, str] = {}

    if "stages_aggregate" in summary and isinstance(
        summary.get("stages_aggregate"), Mapping
    ):
        st_rows: list[dict[str, Any]] = []
        for st_key, st_agg in summary["stages_aggregate"].items():
            if not isinstance(st_agg, Mapping):
                continue
            vs_agg = st_agg.get("vocabulary_sweep_aggregate") or {}
            n50 = vs_agg.get("N50") or {}
            fv = vs_agg.get("full_vocab") or {}
            nc = st_agg.get("negative_controls_aggregate") or {}
            st_rows.append(
                {
                    "stage": st_key,
                    "seed_count": int(st_agg.get("seed_count", len(seeds))),
                    "mean_vocab50_macro_top10_pct": _pct(
                        n50.get("mean_macro_top10", 0.0)
                    ),
                    "std_vocab50_macro_top10_pct": _pct(
                        n50.get("std_macro_top10", 0.0)
                    ),
                    "mean_vocab50_micro_top10_pct": _pct(
                        n50.get("mean_micro_top10", 0.0)
                    ),
                    "std_vocab50_micro_top10_pct": _pct(
                        n50.get("std_micro_top10", 0.0)
                    ),
                    "mean_vocab50_macro_top1_pct": _pct(
                        n50.get("mean_macro_top1", 0.0)
                    ),
                    "std_vocab50_macro_top1_pct": _pct(
                        n50.get("std_macro_top1", 0.0)
                    ),
                    "mean_vocab50_micro_top1_pct": _pct(
                        n50.get("mean_micro_top1", 0.0)
                    ),
                    "std_vocab50_micro_top1_pct": _pct(
                        n50.get("std_micro_top1", 0.0)
                    ),
                    "mean_full_vocab_macro_top10_pct": _pct(
                        fv.get("mean_macro_top10", 0.0)
                    ),
                    "std_full_vocab_macro_top10_pct": _pct(
                        fv.get("std_macro_top10", 0.0)
                    ),
                    "mean_full_vocab_micro_top10_pct": _pct(
                        fv.get("mean_micro_top10", 0.0)
                    ),
                    "std_full_vocab_micro_top10_pct": _pct(
                        fv.get("std_micro_top10", 0.0)
                    ),
                    "mean_full_vocab_macro_top1_pct": _pct(
                        fv.get("mean_macro_top1", 0.0)
                    ),
                    "std_full_vocab_macro_top1_pct": _pct(
                        fv.get("std_macro_top1", 0.0)
                    ),
                    "mean_full_vocab_micro_top1_pct": _pct(
                        fv.get("mean_micro_top1", 0.0)
                    ),
                    "std_full_vocab_micro_top1_pct": _pct(
                        fv.get("std_micro_top1", 0.0)
                    ),
                    "mean_clean_minus_shifted_vocab50_top10_pp": _pct(
                        nc.get("mean_gap_vocab50_macro_top10", 0.0)
                    )
                    if nc.get("enabled")
                    else "",
                    "std_clean_minus_shifted_vocab50_top10_pp": _pct(
                        nc.get("std_gap_vocab50_macro_top10", 0.0)
                    )
                    if nc.get("enabled")
                    else "",
                }
            )
        if st_rows:
            p_csv = _write_csv_table(
                tables_dir / f"multi_seed_stage_summary_{split}.csv",
                list(st_rows[0].keys()),
                st_rows,
            )
            exported["multi_seed_summary_csv"] = str(p_csv)
    else:
        vs_agg = summary.get("vocabulary_sweep_aggregate") or {}
        scale_rows: list[dict[str, Any]] = []
        for s_key in ("N20", "N50", "N50_strict", "N100", "N150", "full_vocab"):
            sd = vs_agg.get(s_key)
            if not isinstance(sd, Mapping):
                continue
            scale_rows.append(
                {
                    "scale_key": s_key,
                    "label": str(sd.get("label", s_key)),
                    "candidate_count": int(sd.get("candidate_count", 0)),
                    "seed_count": int(sd.get("seed_count", len(seeds))),
                    "mean_macro_top10_pct": _pct(sd.get("mean_macro_top10", 0.0)),
                    "std_macro_top10_pct": _pct(sd.get("std_macro_top10", 0.0)),
                    "mean_micro_top10_pct": _pct(sd.get("mean_micro_top10", 0.0)),
                    "std_micro_top10_pct": _pct(sd.get("std_micro_top10", 0.0)),
                    "mean_macro_top1_pct": _pct(sd.get("mean_macro_top1", 0.0)),
                    "std_macro_top1_pct": _pct(sd.get("std_macro_top1", 0.0)),
                    "mean_micro_top1_pct": _pct(sd.get("mean_micro_top1", 0.0)),
                    "std_micro_top1_pct": _pct(sd.get("std_micro_top1", 0.0)),
                    "mean_mrr": _num(sd.get("mean_mrr", 0.0), 4),
                    "std_mrr": _num(sd.get("std_mrr", 0.0), 4),
                }
            )
        if scale_rows:
            p_csv = _write_csv_table(
                tables_dir / f"multi_seed_summary_{split}.csv",
                list(scale_rows[0].keys()),
                scale_rows,
            )
            exported["multi_seed_summary_csv"] = str(p_csv)

        per_seed = summary.get("per_seed") or []
        if isinstance(per_seed, Sequence) and per_seed:
            seed_rows = [
                {
                    "seed": r.get("seed", 0),
                    "best_epoch": r.get("epoch", 0),
                    "vocab50_macro_top10_pct": _pct(
                        r.get("vocab50_macro_top10", 0.0)
                    ),
                    "vocab50_micro_top10_pct": _pct(
                        r.get("vocab50_micro_top10", 0.0)
                    ),
                    "vocab50_macro_top1_pct": _pct(
                        r.get("vocab50_macro_top1", 0.0)
                    ),
                    "vocab50_micro_top1_pct": _pct(
                        r.get("vocab50_micro_top1", 0.0)
                    ),
                    "vocab50_mrr": _num(r.get("vocab50_mrr", 0.0), 4),
                    "full_vocab_macro_top10_pct": _pct(
                        r.get("full_vocab_macro_top10", 0.0)
                    ),
                    "full_vocab_micro_top10_pct": _pct(
                        r.get("full_vocab_micro_top10", 0.0)
                    ),
                    "full_vocab_macro_top1_pct": _pct(
                        r.get("full_vocab_macro_top1", 0.0)
                    ),
                    "full_vocab_micro_top1_pct": _pct(
                        r.get("full_vocab_micro_top1", 0.0)
                    ),
                    "full_vocab_mrr": _num(r.get("full_vocab_mrr", 0.0), 4),
                }
                for r in per_seed
                if isinstance(r, Mapping)
            ]
            if seed_rows:
                ps_csv = _write_csv_table(
                    tables_dir / f"multi_seed_per_seed_{split}.csv",
                    list(seed_rows[0].keys()),
                    seed_rows,
                )
                exported["multi_seed_per_seed_csv"] = str(ps_csv)

    md_path = out_dir / f"multi_seed_evaluation_report_{split}.md"
    md_lines = [
        f"# Multi-Seed Evaluation Summary (`{dataset}` | `{split}` | Seeds: `{seeds}`)",
        "",
        f"Exported flat CSV tables to `{tables_dir.name}/`.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    exported["multi_seed_report_md"] = str(md_path)
    return exported


def export_sequence_decoding_tables_and_markdown(
    report: Mapping[str, Any],
    output_dir: str | Path | None = None,
    *,
    decoded_bundles: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """将 S3 序列解码评估结果与逐句预测导出为扁平 CSV 表、Markdown 报告及双语逐句可读对照视图。"""
    target_dir_raw = output_dir or report.get("evaluation_dir") or report.get("output_dir")
    if not target_dir_raw:
        return {}
    out_dir = Path(str(target_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    split = str(report.get("split", "test"))
    dataset = str(report.get("dataset", "-"))
    language = str(
        report.get("language", "fr" if dataset == "pallier2025" else "zh")
    )
    conditions = report.get("conditions") or {}
    sig_map = report.get("statistical_significance") or {}
    per_subj = report.get("per_subject") or []
    subj_sum = report.get("per_subject_summary") or {}
    vocab_scaling = report.get("vocabulary_scaling") or {}
    multi_subj = report.get("multi_subject_scaling") or {}
    pareto_map = report.get("confidence_gate_pareto") or {}

    exported: dict[str, str] = {}

    # 1. 导出条件对比表 s3_conditions_summary_{split}.csv（仅含 4 档输入延迟预算与负对照，不含 F_local 与在线屏幕 C-Gated）
    cond_rows: list[dict[str, Any]] = []
    cgated_rows: list[dict[str, Any]] = []
    for cond_name, cm in conditions.items():
        if not isinstance(cm, Mapping):
            continue
        if str(cond_name) == "F_local":
            continue
        row_item = {
            "condition": cond_name,
            "segment_count": int(cm.get("segment_count", 0)),
            "total_words": int(cm.get("total_words", 0)),
            "bleu1_subject_macro_pct": _num(cm.get("bleu1_pct", 0.0), 2),
            "corpus_bleu1_pct": _num(cm.get("corpus_bleu1_pct", 0.0), 2),
            "beam_oracle_bleu1_pct": _num(
                cm.get("beam_oracle_bleu1_pct", 0.0), 2
            ),
            "word_acc_pct": _num(cm.get("word_acc_pct", 0.0), 2),
            "in_vocabulary_word_acc_pct": _num(
                cm.get("in_vocabulary_word_acc_pct", 0.0), 2
            ),
            "word_acc_top5_pct": _num(cm.get("word_acc_top5_pct", 0.0), 2),
            "content_word_acc_pct": _num(
                cm.get("content_word_acc_pct", 0.0), 2
            ),
            "function_word_acc_pct": _num(
                cm.get("function_word_acc_pct", 0.0), 2
            ),
            "wer_pct": _num(cm.get("wer_pct", 0.0), 2),
            "cer_pct": _num(cm.get("cer_pct", 0.0), 2),
            "bertscore_f1_pct": _num(cm.get("bertscore_f1_pct", 0.0), 2),
            "screen_rewrite_rate_pct": _num(
                cm.get("screen_rewrite_rate_pct", 0.0), 2
            )
            if "screen_rewrite_rate_pct" in cm
            else "",
            "recommended_tau_gate": cm.get("recommended_tau_gate", ""),
        }
        if str(cond_name).endswith("_c_gated") or bool(cm.get("is_online_screen_gate")):
            cgated_rows.append(row_item)
        else:
            cond_rows.append(row_item)
    if cond_rows:
        c_csv = _write_csv_table(
            tables_dir / f"s3_conditions_summary_{split}.csv",
            list(cond_rows[0].keys()),
            cond_rows,
        )
        exported["s3_conditions_csv"] = str(c_csv)

    # 1B. 导出三档核心词表基准表 s3_core_vocabulary_benchmarks_{split}.csv
    core_benchmarks = report.get("core_vocabulary_benchmarks") or {}
    if isinstance(core_benchmarks, Mapping) and core_benchmarks:
        cb_rows: list[dict[str, Any]] = []
        for v_key, cb_entry in sorted(
            core_benchmarks.items(),
            key=lambda kv: int(
                kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
            )
            if isinstance(kv[1], Mapping)
            else 0,
        ):
            if not isinstance(cb_entry, Mapping):
                continue
            v_sz = int(cb_entry.get("vocabulary_size", 0))
            tok_cov = _num(cb_entry.get("test_token_coverage_pct", 0.0), 2)
            subsets_map = cb_entry.get("subsets") or {}
            for sub_key in ("all_segments", "core_subset_100pct", "core_subset_80pct"):
                sub_blk = subsets_map.get(sub_key)
                if not isinstance(sub_blk, Mapping):
                    continue
                seg_cnt = int(sub_blk.get("segment_count", 0))
                sub_conds = sub_blk.get("conditions") or {}
                for cond_name, cm in sub_conds.items():
                    if not isinstance(cm, Mapping) or str(cond_name) == "F_local" or str(cond_name).endswith("_c_gated"):
                        continue
                    cb_rows.append(
                        {
                            "scale_key": str(v_key),
                            "vocabulary_size": v_sz,
                            "test_token_coverage_pct": tok_cov,
                            "subset": sub_key,
                            "segment_count": seg_cnt,
                            "condition": str(cond_name),
                            "bleu1_pct": _num(cm.get("bleu1_pct", 0.0), 2),
                            "word_acc_pct": _num(cm.get("word_acc_pct", 0.0), 2),
                            "in_vocabulary_word_acc_pct": _num(
                                cm.get("in_vocabulary_word_acc_pct", 0.0), 2
                            ),
                            "word_acc_top5_pct": _num(
                                cm.get("word_acc_top5_pct", 0.0), 2
                            ),
                            "wer_pct": _num(cm.get("wer_pct", 0.0), 2),
                            "cer_pct": _num(cm.get("cer_pct", 0.0), 2),
                            "bertscore_f1_pct": _num(
                                cm.get("bertscore_f1_pct", 0.0), 2
                            ),
                        }
                    )
        if cb_rows:
            cb_csv = _write_csv_table(
                tables_dir / f"s3_core_vocabulary_benchmarks_{split}.csv",
                list(cb_rows[0].keys()),
                cb_rows,
            )
            exported["s3_core_vocabulary_benchmarks_csv"] = str(cb_csv)

    # 1C. 导出跨被试多观测神经聚合表 s3_multi_subject_scaling_{split}.csv
    if isinstance(multi_subj, Mapping) and multi_subj:
        ms_rows: list[dict[str, Any]] = []
        for k_lbl, m_entry in sorted(
            multi_subj.items(),
            key=lambda kv: int(
                kv[1].get("k_observations", str(kv[0]).lstrip("kK=_") or 1)
            )
            if isinstance(kv[1], Mapping)
            else 1,
        ):
            if not isinstance(m_entry, Mapping):
                continue
            k_obs = int(m_entry.get("k_observations", str(k_lbl).lstrip("kK=_") or 1))
            by_v = m_entry.get("by_vocabulary_size")
            if isinstance(by_v, Mapping) and by_v:
                for v_key, v_blk in sorted(
                    by_v.items(),
                    key=lambda kv: int(
                        kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
                    )
                    if isinstance(kv[1], Mapping)
                    else 0,
                ):
                    if not isinstance(v_blk, Mapping):
                        continue
                    c80 = v_blk.get("core_subset_80pct") or {}
                    c100 = v_blk.get("core_subset_100pct") or {}
                    ms_rows.append(
                        {
                            "k_label": str(k_lbl),
                            "k_observations": k_obs,
                            "scale_key": str(v_key),
                            "vocabulary_size": int(v_blk.get("vocabulary_size", 0)),
                            "all_bleu1_pct": _num(v_blk.get("bleu1_pct", 0.0), 2),
                            "all_word_acc_pct": _num(v_blk.get("word_acc_pct", 0.0), 2),
                            "all_in_vocab_word_acc_pct": _num(
                                v_blk.get("in_vocabulary_word_acc_pct", 0.0), 2
                            ),
                            "core80_segment_count": int(
                                v_blk.get("core_80pct_segment_count", 0)
                            ),
                            "core80_bleu1_pct": _num(c80.get("bleu1_pct", 0.0), 2),
                            "core80_word_acc_pct": _num(c80.get("word_acc_pct", 0.0), 2),
                            "core100_segment_count": int(
                                v_blk.get("core_100pct_segment_count", 0)
                            ),
                            "core100_bleu1_pct": _num(c100.get("bleu1_pct", 0.0), 2),
                            "core100_word_acc_pct": _num(
                                c100.get("word_acc_pct", 0.0), 2
                            ),
                        }
                    )
            else:
                ms_rows.append(
                    {
                        "k_label": str(k_lbl),
                        "k_observations": k_obs,
                        "scale_key": "full",
                        "vocabulary_size": 0,
                        "all_bleu1_pct": _num(m_entry.get("bleu1_pct", 0.0), 2),
                        "all_word_acc_pct": _num(m_entry.get("word_acc_pct", 0.0), 2),
                        "all_in_vocab_word_acc_pct": _num(
                            m_entry.get("in_vocabulary_word_acc_pct", 0.0), 2
                        ),
                        "core80_segment_count": 0,
                        "core80_bleu1_pct": "",
                        "core80_word_acc_pct": "",
                        "core100_segment_count": 0,
                        "core100_bleu1_pct": "",
                        "core100_word_acc_pct": "",
                    }
                )
        if ms_rows:
            ms_csv = _write_csv_table(
                tables_dir / f"s3_multi_subject_scaling_{split}.csv",
                list(ms_rows[0].keys()),
                ms_rows,
            )
            exported["s3_multi_subject_scaling_csv"] = str(ms_csv)

    # 2. 导出受试者级序列解码表 s3_per_subject_{split}.csv
    if isinstance(per_subj, Sequence) and per_subj:
        ps_rows: list[dict[str, Any]] = []
        for s_item in per_subj:
            if not isinstance(s_item, Mapping):
                continue
            ps_rows.append(
                {
                    "subject_id": str(s_item.get("subject_id", "-")),
                    "segment_count": int(s_item.get("segment_count", 0)),
                    "total_words": int(s_item.get("total_words", 0)),
                    "s3_bleu1_pct": _num(s_item.get("bleu1_pct", 0.0), 2),
                    "k0_bleu1_pct": _num(s_item.get("k0_bleu1_pct", 0.0), 2)
                    if "k0_bleu1_pct" in s_item
                    else "",
                    "s3_minus_k0_bleu1_pp": _num(
                        s_item.get("s3_minus_k0_bleu1_pp", 0.0), 2
                    )
                    if "s3_minus_k0_bleu1_pp" in s_item
                    else "",
                    "s3_word_acc_pct": _num(s_item.get("word_acc_pct", 0.0), 2),
                    "k0_word_acc_pct": _num(s_item.get("k0_word_acc_pct", 0.0), 2)
                    if "k0_word_acc_pct" in s_item
                    else "",
                    "s3_minus_k0_word_acc_pp": _num(
                        s_item.get("s3_minus_k0_word_acc_pp", 0.0), 2
                    )
                    if "s3_minus_k0_word_acc_pp" in s_item
                    else "",
                    "content_word_acc_pct": _num(
                        s_item.get("content_word_acc_pct", 0.0), 2
                    ),
                    "function_word_acc_pct": _num(
                        s_item.get("function_word_acc_pct", 0.0), 2
                    ),
                    "wer_pct": _num(s_item.get("wer_pct", 0.0), 2),
                    "cer_pct": _num(s_item.get("cer_pct", 0.0), 2),
                    "bertscore_f1_pct": _num(
                        s_item.get("bertscore_f1_pct", 0.0), 2
                    ),
                }
            )
        if ps_rows and len(ps_rows) >= 2:
            num_cols = (
                "s3_bleu1_pct",
                "s3_word_acc_pct",
                "content_word_acc_pct",
                "function_word_acc_pct",
                "wer_pct",
                "cer_pct",
                "bertscore_f1_pct",
            )
            m_row: dict[str, Any] = {
                "subject_id": "MEAN",
                "segment_count": sum(int(r["segment_count"]) for r in ps_rows),
                "total_words": sum(int(r["total_words"]) for r in ps_rows),
            }
            sd_row: dict[str, Any] = {
                "subject_id": "SD",
                "segment_count": "",
                "total_words": "",
            }
            for col in num_cols:
                vals = [float(r[col]) for r in ps_rows if r.get(col) != ""]
                m_row[col] = round(float(np.mean(vals)), 2) if vals else ""
                sd_row[col] = (
                    round(float(np.std(vals, ddof=0)), 2) if vals else ""
                )
            for opt_c in (
                "k0_bleu1_pct",
                "s3_minus_k0_bleu1_pp",
                "k0_word_acc_pct",
                "s3_minus_k0_word_acc_pp",
            ):
                vals = [
                    float(r[opt_c])
                    for r in ps_rows
                    if r.get(opt_c) not in ("", None)
                ]
                m_row[opt_c] = round(float(np.mean(vals)), 2) if vals else ""
                sd_row[opt_c] = (
                    round(float(np.std(vals, ddof=0)), 2) if vals else ""
                )
            ps_rows.append(m_row)
            ps_rows.append(sd_row)

        if ps_rows:
            s_csv = _write_csv_table(
                tables_dir / f"s3_per_subject_{split}.csv",
                list(ps_rows[0].keys()),
                ps_rows,
            )
            exported["s3_per_subject_csv"] = str(s_csv)

    # 3. 导出多尺度词表表 s3_vocabulary_scaling_{split}.csv
    if isinstance(vocab_scaling, Mapping) and vocab_scaling:
        vs_rows: list[dict[str, Any]] = []
        for scale_key, sc_block in vocab_scaling.items():
            if not isinstance(sc_block, Mapping):
                continue
            conds = sc_block.get("conditions") or {}
            s3_m = conds.get("S3_clean") or {}
            k0_m = conds.get("K0") or {}
            sh_m = conds.get("S3_shifted_neg75s") or {}
            vs_rows.append(
                {
                    "scale_key": scale_key,
                    "vocabulary_size": int(sc_block.get("vocabulary_size", 0)),
                    "test_token_coverage_pct": _num(
                        sc_block.get("test_token_coverage_pct", 0.0), 2
                    ),
                    "s3_bleu1_pct": _num(s3_m.get("bleu1_pct", 0.0), 2),
                    "k0_bleu1_pct": _num(k0_m.get("bleu1_pct", 0.0), 2),
                    "delta_s3_minus_k0_bleu1_pp": _num(
                        float(s3_m.get("bleu1_pct", 0.0))
                        - float(k0_m.get("bleu1_pct", 0.0)),
                        2,
                    ),
                    "s3_word_acc_pct": _num(s3_m.get("word_acc_pct", 0.0), 2),
                    "s3_in_vocab_word_acc_pct": _num(
                        s3_m.get("in_vocabulary_word_acc_pct", 0.0), 2
                    ),
                    "s3_content_acc_pct": _num(
                        s3_m.get("content_word_acc_pct", 0.0), 2
                    ),
                    "s3_bertscore_f1_pct": _num(
                        s3_m.get("bertscore_f1_pct", 0.0), 2
                    ),
                    "shifted_s3_bleu1_pct": _num(sh_m.get("bleu1_pct", 0.0), 2),
                    "clean_minus_shifted_bleu1_pp": _num(
                        float(s3_m.get("bleu1_pct", 0.0))
                        - float(sh_m.get("bleu1_pct", 0.0)),
                        2,
                    ),
                }
            )
        if vs_rows:
            vs_csv = _write_csv_table(
                tables_dir / f"s3_vocabulary_scaling_{split}.csv",
                list(vs_rows[0].keys()),
                vs_rows,
            )
            exported["s3_vocabulary_scaling_csv"] = str(vs_csv)

    # 4. 导出 C-gated 帕累托表 s3_confidence_gate_pareto_{split}.csv
    s3_pareto = (
        pareto_map.get("S3_clean") or pareto_map.get("S3") or {}
        if isinstance(pareto_map, Mapping)
        else {}
    )
    pareto_pts = (
        s3_pareto.get("pareto_curve") or []
        if isinstance(s3_pareto, Mapping)
        else []
    )
    if pareto_pts:
        p_rows = [
            {
                "tau_gate": _num(pt.get("tau_gate", 0.0), 2),
                "screen_rewrite_rate_pct": _num(
                    pt.get("screen_rewrite_rate_pct", 0.0), 2
                ),
                "beneficial_repair_rate_pct": _num(
                    pt.get("beneficial_repair_rate_pct", 0.0), 2
                ),
                "harmful_rewrite_rate_pct": _num(
                    pt.get("harmful_rewrite_rate_pct", 0.0), 2
                ),
                "invalid_flicker_rate_pct": _num(
                    pt.get("invalid_flicker_rate_pct", 0.0), 2
                ),
                "invalid_flicker_filtered_pct": _num(
                    pt.get("invalid_flicker_filtered_pct", 0.0), 2
                ),
                "net_beneficial_gain_pct": _num(
                    pt.get("net_beneficial_gain_pct", 0.0), 2
                ),
                "bleu1_pct": _num(pt.get("bleu1_pct", 0.0), 2),
                "word_acc_pct": _num(pt.get("word_acc_pct", 0.0), 2),
                "content_word_acc_pct": _num(
                    pt.get("content_word_acc_pct", 0.0), 2
                ),
                "bertscore_f1_pct": _num(pt.get("bertscore_f1_pct", 0.0), 2),
                "beneficial_repairs": int(pt.get("beneficial_repairs", 0)),
                "harmful_rewrites": int(pt.get("harmful_rewrites", 0)),
                "triggered_rewrites": int(pt.get("triggered_rewrites", 0)),
            }
            for pt in pareto_pts
            if isinstance(pt, Mapping)
        ]
        if p_rows:
            p_csv = _write_csv_table(
                tables_dir / f"s3_confidence_gate_pareto_{split}.csv",
                list(p_rows[0].keys()),
                p_rows,
            )
            exported["s3_pareto_csv"] = str(p_csv)

    # 5. 导出逐句对照可读表 s3_decoded_sentences_{split}.csv 与 s3_decoded_sentences_{split}.md
    s3_bundle = (
        decoded_bundles.get("S3_clean") or decoded_bundles.get("S3")
        if isinstance(decoded_bundles, Mapping)
        else None
    )
    k0_bundle = (
        decoded_bundles.get("K0")
        if isinstance(decoded_bundles, Mapping)
        else None
    )
    k1_bundle = (
        decoded_bundles.get("K1")
        if isinstance(decoded_bundles, Mapping)
        else None
    )
    if isinstance(s3_bundle, Mapping) and s3_bundle.get("final_sentences"):
        s3_sents = s3_bundle.get("final_sentences") or []
        k0_sents = (
            k0_bundle.get("final_sentences") or []
            if isinstance(k0_bundle, Mapping)
            else []
        )
        k1_sents = (
            k1_bundle.get("final_sentences") or []
            if isinstance(k1_bundle, Mapping)
            else []
        )

        sentence_rows: list[dict[str, Any]] = []
        for idx, rec in enumerate(s3_sents):
            if not isinstance(rec, Mapping):
                continue
            ref_w = [str(w) for w in rec.get("reference_words") or ()]
            s3_w = [str(w) for w in rec.get("predicted_words") or ()]
            k0_w = (
                [str(w) for w in k0_sents[idx].get("predicted_words") or ()]
                if idx < len(k0_sents) and isinstance(k0_sents[idx], Mapping)
                else [str(w) for w in rec.get("initial_k0_words") or s3_w]
            )
            k1_w = (
                [str(w) for w in k1_sents[idx].get("predicted_words") or ()]
                if idx < len(k1_sents) and isinstance(k1_sents[idx], Mapping)
                else s3_w
            )
            iv_mask = rec.get("in_vocabulary_mask") or [True] * len(ref_w)
            n_w = max(1, len(ref_w))
            cov_pct = sum(1 for f in iv_mask if bool(f)) / n_w * 100.0
            k0_hits = sum(int(a == b) for a, b in zip(ref_w, k0_w))
            s3_hits = sum(int(a == b) for a, b in zip(ref_w, s3_w))
            k0_acc = k0_hits / n_w * 100.0
            s3_acc = s3_hits / n_w * 100.0

            repairs: list[str] = []
            harms: list[str] = []
            for pos, r_tok in enumerate(ref_w):
                w0 = k0_w[pos] if pos < len(k0_w) else ""
                w2 = s3_w[pos] if pos < len(s3_w) else ""
                if w0 != r_tok and w2 == r_tok:
                    repairs.append(f"#{pos + 1}:{w0}->{w2}")
                elif w0 == r_tok and w2 != r_tok:
                    harms.append(f"#{pos + 1}:{w0}->{w2}")

            sentence_rows.append(
                {
                    "segment_index": int(rec.get("segment_index", idx)),
                    "subject_id": str(rec.get("subject_id", "-")),
                    "group_id": str(rec.get("group_id", "-")),
                    "word_count": len(ref_w),
                    "in_vocab_coverage_pct": round(cov_pct, 1),
                    "reference_sentence": _join_tokens_natural(ref_w, language),
                    "k0_initial_sentence": _join_tokens_natural(k0_w, language),
                    "k1_step1_sentence": _join_tokens_natural(k1_w, language),
                    "s3_final_sentence": _join_tokens_natural(s3_w, language),
                    "k0_word_acc_pct": round(k0_acc, 2),
                    "s3_word_acc_pct": round(s3_acc, 2),
                    "net_word_acc_gain_pp": round(s3_acc - k0_acc, 2),
                    "beneficial_repairs": len(repairs),
                    "harmful_overwrites": len(harms),
                    "repair_highlights": "; ".join(repairs),
                    "harm_highlights": "; ".join(harms),
                }
            )

        if sentence_rows:
            sent_csv = _write_csv_table(
                tables_dir / f"s3_decoded_sentences_{split}.csv",
                list(sentence_rows[0].keys()),
                sentence_rows,
            )
            exported["s3_decoded_sentences_csv"] = str(sent_csv)

            # 生成人类易读的逐句对照 Markdown 文件
            repaired_top = sorted(
                [r for r in sentence_rows if r["beneficial_repairs"] > r["harmful_overwrites"]],
                key=lambda r: (
                    r["beneficial_repairs"] - r["harmful_overwrites"],
                    r["s3_word_acc_pct"],
                ),
                reverse=True,
            )[:15]
            accurate_top = sorted(
                sentence_rows,
                key=lambda r: (r["s3_word_acc_pct"], r["word_count"]),
                reverse=True,
            )[:15]

            sent_md: list[str] = [
                f"# S3 Decoded Sentences Qualitative Inspection (`{dataset}` | `{split}`)",
                "",
                f"- **Total Segments**: `{len(sentence_rows)}` | **Language**: `{language}` | **Full Spreadsheet**: `tables/s3_decoded_sentences_{split}.csv`",
                "",
            ]
            if repaired_top:
                sent_md.extend(
                    [
                        "## 1. Top Beneficial Rolling Revision Examples (`K0 -> S3 (K2)` Repairs)",
                        "",
                        "| Seg # | Subject | Reference | K0 (Initial Draft) | S3 / K2 (Revised Final) | Word Acc (`K0 -> S3`) | Repaired Positions (`Miss -> Hit`) |",
                        "| ---: | :--- | :--- | :--- | :--- | ---: | :--- |",
                    ]
                )
                for r in repaired_top:
                    sent_md.append(
                        f"| `{r['segment_index']}` | `{r['subject_id']}` | {r['reference_sentence']} | "
                        f"{r['k0_initial_sentence']} | **{r['s3_final_sentence']}** | "
                        f"`{r['k0_word_acc_pct']:.1f}% -> {r['s3_word_acc_pct']:.1f}%` (**{r['net_word_acc_gain_pp']:+.1f} pp**) | "
                        f"`{r['repair_highlights'] or '-'}` |"
                    )
                sent_md.append("")

            if accurate_top:
                sent_md.extend(
                    [
                        "## 2. Top Highest-Accuracy Decoded Segments",
                        "",
                        "| Seg # | Subject | Words (Cov%) | Reference | K0 (Initial) | K1 (+1w) | S3 / K2 (Final) | S3 Word Acc |",
                        "| ---: | :--- | ---: | :--- | :--- | :--- | :--- | ---: |",
                    ]
                )
                for r in accurate_top:
                    sent_md.append(
                        f"| `{r['segment_index']}` | `{r['subject_id']}` | {r['word_count']} ({r['in_vocab_coverage_pct']:.0f}%) | "
                        f"{r['reference_sentence']} | {r['k0_initial_sentence']} | {r['k1_step1_sentence']} | "
                        f"**{r['s3_final_sentence']}** | **{r['s3_word_acc_pct']:.1f}%** |"
                    )
                sent_md.append("")

            max_preview = min(50, len(sentence_rows))
            sent_md.extend(
                [
                    f"## 3. Segment-by-Segment Side-by-Side Comparison (First {max_preview} of {len(sentence_rows)} Segments)",
                    "",
                    "| Seg # | Subject | Reference | K0 (Initial) | S3 / K2 (Final) | Word Acc | Repairs |",
                    "| ---: | :--- | :--- | :--- | :--- | ---: | :--- |",
                ]
            )
            for r in sentence_rows[:max_preview]:
                sent_md.append(
                    f"| `{r['segment_index']}` | `{r['subject_id']}` | {r['reference_sentence']} | "
                    f"{r['k0_initial_sentence']} | {r['s3_final_sentence']} | "
                    f"{r['s3_word_acc_pct']:.1f}% | `{r['repair_highlights'] or '-'}` |"
                )
            sent_md.append("")
            sent_md_path = out_dir / f"s3_decoded_sentences_{split}.md"
            sent_md_path.write_text("\n".join(sent_md) + "\n", encoding="utf-8")
            exported["s3_decoded_sentences_md"] = str(sent_md_path)

    # 6. 生成 S3 结构化 Markdown 总报告 s3_evaluation_report_{split}.md
    md_lines = [
        f"# Stage 3 (S3) Sequence Decoding Evaluation Report (`{dataset}` | `{split}`)",
        "",
    ]
    s3_clean_m = conditions.get("S3_clean") or conditions.get("S3") or {}
    k0_m = conditions.get("K0") or {}
    sh_m = conditions.get("S3_shifted_neg75s") or {}
    if s3_clean_m:
        md_lines.extend(
            [
                "> [!IMPORTANT]",
                f"> **Executive Headline**: `S3_clean` BLEU-1 = **{ _num(s3_clean_m.get('bleu1_pct'), 2):.2f}%** "
                f"(vs `K0` `{_num(k0_m.get('bleu1_pct'), 2):.2f}%`, Δ = `{float(s3_clean_m.get('bleu1_pct', 0.0)) - float(k0_m.get('bleu1_pct', 0.0)):+.2f} pp`; "
                f"vs `Shifted -75s` `{_num(sh_m.get('bleu1_pct'), 2):.2f}%`, Δ = `{float(s3_clean_m.get('bleu1_pct', 0.0)) - float(sh_m.get('bleu1_pct', 0.0)):+.2f} pp`), "
                f"Word Acc = **{_num(s3_clean_m.get('word_acc_pct'), 2):.2f}%** "
                f"(Content: `{_num(s3_clean_m.get('content_word_acc_pct'), 2):.2f}%`, Function: `{_num(s3_clean_m.get('function_word_acc_pct'), 2):.2f}%`), "
                f"BERTScore-F1 = **{_num(s3_clean_m.get('bertscore_f1_pct'), 2):.2f}%**.",
                "",
            ]
        )

    if cond_rows:
        md_lines.extend(
            [
                "## 1. Decoding Conditions Summary",
                "",
                "| Condition | Segments (Words) | BLEU-1 (Subj) | Corpus BLEU-1 | Word Acc (In-Vocab) | Content / Function Acc | WER | BERTScore-F1 |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cr in cond_rows:
            md_lines.append(
                f"| **`{cr['condition']}`** | {cr['segment_count']} ({cr['total_words']}) | "
                f"**{cr['bleu1_subject_macro_pct']:.2f}%** | {cr['corpus_bleu1_pct']:.2f}% | "
                f"**{cr['word_acc_pct']:.2f}%** ({cr['in_vocabulary_word_acc_pct']:.2f}%) | "
                f"{cr['content_word_acc_pct']:.2f}% / {cr['function_word_acc_pct']:.2f}% | "
                f"{cr['wer_pct']:.2f}% | **{cr['bertscore_f1_pct']:.2f}%** |"
            )
        md_lines.append("")

    if isinstance(sig_map, Mapping) and sig_map:
        md_lines.extend(
            [
                "## 2. Paired Material-Block Statistical Significance (`95% CI & Permutation p-value`)",
                "",
                "| Comparison | Metric | Observed Effect (pp) | 95% Bootstrap CI (pp) | Two-Sided p-value | Significance | Blocks / Segments |",
                "| :--- | :--- | ---: | ---: | ---: | :---: | ---: |",
            ]
        )
        for comp_k, sig_d in sig_map.items():
            if not isinstance(sig_d, Mapping):
                continue
            md_lines.append(
                f"| **`{sig_d.get('label', comp_k)}`** | `{sig_d.get('metric', '-')}` | "
                f"**{_num(sig_d.get('effect_pp', 0.0), 2):+.2f} pp** | "
                f"`[{_num(sig_d.get('ci_lower_pp', 0.0), 2):+.2f}, {_num(sig_d.get('ci_upper_pp', 0.0), 2):+.2f}]` | "
                f"`{_num(sig_d.get('p_two_sided', sig_d.get('p_one_sided', 1.0)), 4):.4f}` | "
                f"**{sig_d.get('significance', 'ns')}** | {sig_d.get('blocks', '-')} |"
            )
        md_lines.append("")

    if isinstance(per_subj, Sequence) and per_subj:
        md_lines.extend(
            [
                "## 3. Per-Subject Sequence Decoding Performance (`Mean ± SD`)",
                "",
            ]
        )
        if isinstance(subj_sum, Mapping) and subj_sum:
            md_lines.append(
                f"- **Across-Subject BLEU-1**: `{_num(subj_sum.get('mean_bleu1_pct'), 2):.2f}% ± {_num(subj_sum.get('std_bleu1_pct'), 2):.2f}%` | "
                f"**Word Acc**: `{_num(subj_sum.get('mean_word_acc_pct'), 2):.2f}% ± {_num(subj_sum.get('std_word_acc_pct'), 2):.2f}%` | "
                f"**Improved Subjects (`S3 > K0`)**: `{subj_sum.get('improved_subjects_count', '-')}/{subj_sum.get('subject_count', len(per_subj))}`"
            )
            md_lines.append("")
        md_lines.extend(
            [
                "| Subject | Segments (Words) | S3 BLEU-1 | K0 BLEU-1 | Δ BLEU-1 | S3 Word Acc | Content Acc | Function Acc | BERTScore-F1 |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for s in per_subj:
            if not isinstance(s, Mapping):
                continue
            k0_b = (
                f"{_num(s.get('k0_bleu1_pct'), 2):.2f}%"
                if "k0_bleu1_pct" in s
                else "-"
            )
            d_b = (
                f"**{_num(s.get('s3_minus_k0_bleu1_pp'), 2):+.2f} pp**"
                if "s3_minus_k0_bleu1_pp" in s
                else "-"
            )
            md_lines.append(
                f"| `{s.get('subject_id', '-')}` | {s.get('segment_count', 0)} ({s.get('total_words', 0)}) | "
                f"**{_num(s.get('bleu1_pct'), 2):.2f}%** | {k0_b} | {d_b} | "
                f"**{_num(s.get('word_acc_pct'), 2):.2f}%** | {_num(s.get('content_word_acc_pct'), 2):.2f}% | "
                f"{_num(s.get('function_word_acc_pct'), 2):.2f}% | {_num(s.get('bertscore_f1_pct'), 2):.2f}% |"
            )
        md_lines.append("")

    if isinstance(core_benchmarks, Mapping) and core_benchmarks:
        md_lines.extend(
            [
                "## 4. Core Vocabulary Benchmarks (`|V| ∈ {50, 100, 250}`)",
                "",
                "| Scale `|V|` | Token Coverage | Subset | Segments | Rolling-K2 BLEU-1 | Rolling-K0 BLEU-1 | Rolling-K2 Word Acc | In-Vocab Word Acc | BERTScore-F1 |",
                "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for v_key, cb_entry in sorted(
            core_benchmarks.items(),
            key=lambda kv: int(
                kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
            )
            if isinstance(kv[1], Mapping)
            else 0,
        ):
            if not isinstance(cb_entry, Mapping):
                continue
            tok_cov = _num(cb_entry.get("test_token_coverage_pct", 0.0), 2)
            subsets_map = cb_entry.get("subsets") or {}
            for sub_key in ("all_segments", "core_subset_100pct", "core_subset_80pct"):
                sub_blk = subsets_map.get(sub_key)
                if not isinstance(sub_blk, Mapping):
                    continue
                seg_cnt = int(sub_blk.get("segment_count", 0))
                sub_conds = sub_blk.get("conditions") or {}
                s3_m = sub_conds.get("S3_clean") or {}
                k0_m_sub = sub_conds.get("K0") or {}
                if not s3_m:
                    continue
                md_lines.append(
                    f"| **`{v_key}`** | {tok_cov:.2f}% | `{sub_key}` | {seg_cnt} | "
                    f"**{_num(s3_m.get('bleu1_pct'), 2):.2f}%** | {_num(k0_m_sub.get('bleu1_pct'), 2):.2f}% | "
                    f"**{_num(s3_m.get('word_acc_pct'), 2):.2f}%** | {_num(s3_m.get('in_vocabulary_word_acc_pct'), 2):.2f}% | "
                    f"{_num(s3_m.get('bertscore_f1_pct'), 2):.2f}% |"
                )
        md_lines.append("")

    if isinstance(multi_subj, Mapping) and multi_subj:
        md_lines.extend(
            [
                "## 5. Multi-Subject Observation Scaling (`k ∈ {1, 2, 5, 10}`)",
                "",
                "| Subjects `k` | Scale `|V|` | All BLEU-1 | All Word Acc | In-Vocab Word Acc | `>=80%` Core BLEU-1 (`Word Acc`) | `100%` 0-OOV Core BLEU-1 (`Word Acc`) |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for k_lbl, m_entry in sorted(
            multi_subj.items(),
            key=lambda kv: int(
                kv[1].get("k_observations", str(kv[0]).lstrip("kK=_") or 1)
            )
            if isinstance(kv[1], Mapping)
            else 1,
        ):
            if not isinstance(m_entry, Mapping):
                continue
            by_v = m_entry.get("by_vocabulary_size")
            if isinstance(by_v, Mapping) and by_v:
                for v_key, v_blk in sorted(
                    by_v.items(),
                    key=lambda kv: int(
                        kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
                    )
                    if isinstance(kv[1], Mapping)
                    else 0,
                ):
                    if not isinstance(v_blk, Mapping):
                        continue
                    c80 = v_blk.get("core_subset_80pct") or {}
                    c100 = v_blk.get("core_subset_100pct") or {}
                    c80_str = (
                        f"**{_num(c80.get('bleu1_pct'), 2):.2f}%** (`{_num(c80.get('word_acc_pct'), 2):.2f}%`)"
                        if c80
                        else "-"
                    )
                    c100_str = (
                        f"**{_num(c100.get('bleu1_pct'), 2):.2f}%** (`{_num(c100.get('word_acc_pct'), 2):.2f}%`)"
                        if c100
                        else "-"
                    )
                    md_lines.append(
                        f"| **`{k_lbl}`** | `{v_key}` | "
                        f"**{_num(v_blk.get('bleu1_pct'), 2):.2f}%** | {_num(v_blk.get('word_acc_pct'), 2):.2f}% | "
                        f"{_num(v_blk.get('in_vocabulary_word_acc_pct'), 2):.2f}% | {c80_str} | {c100_str} |"
                    )
            else:
                md_lines.append(
                    f"| **`{k_lbl}`** | `Full` | "
                    f"**{_num(m_entry.get('bleu1_pct'), 2):.2f}%** | {_num(m_entry.get('word_acc_pct'), 2):.2f}% | "
                    f"{_num(m_entry.get('in_vocabulary_word_acc_pct'), 2):.2f}% | - | - |"
                )
        md_lines.append("")

    if cgated_rows:
        md_lines.extend(
            [
                "## 6. Online Screen Anti-Flicker Stability (`C-Gated`)",
                "",
                "| Condition | Recommended `τ_gate` | Screen Rewrite Rate | BLEU-1 (Subj) | Word Acc (In-Vocab) | WER | BERTScore-F1 |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cr in cgated_rows:
            rw_str = (
                f"{float(cr['screen_rewrite_rate_pct']):.2f}%"
                if cr.get("screen_rewrite_rate_pct") not in ("", None)
                else "-"
            )
            md_lines.append(
                f"| **`{cr['condition']}`** | `{cr.get('recommended_tau_gate', '-')}` | "
                f"`{rw_str}` | **{cr['bleu1_subject_macro_pct']:.2f}%** | "
                f"**{cr['word_acc_pct']:.2f}%** ({cr['in_vocabulary_word_acc_pct']:.2f}%) | "
                f"{cr['wer_pct']:.2f}% | **{cr['bertscore_f1_pct']:.2f}%** |"
            )
        md_lines.append("")

    md_path = out_dir / f"s3_evaluation_report_{split}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    exported["s3_report_md"] = str(md_path)
    return exported


def _extract_debiasing_row_metrics(
    c_ref: Mapping[str, Any],
    s_agg: Mapping[str, Any],
) -> dict[str, Any]:
    """提取留一法共模扣除与子空间去偏共用的 50 词、全词表及 K0/K1/K2 统一口径（Word-Macro / Event-Micro）列字典。"""
    c_hz = c_ref.get("horizons") if isinstance(c_ref.get("horizons"), Mapping) else {}
    s_hz = s_agg.get("horizons") if isinstance(s_agg.get("horizons"), Mapping) else {}

    def _gap_pct(gap_key: str, c_key: str, s_key: str) -> float:
        if gap_key in s_agg and s_agg.get(gap_key) is not None:
            return _pct(s_agg.get(gap_key))
        return round(_pct(c_ref.get(c_key)) - _pct(s_agg.get(s_key)), 2)

    def _gap_num(gap_key: str, c_key: str, s_key: str, digits: int = 4) -> float:
        if gap_key in s_agg and s_agg.get(gap_key) is not None:
            return _num(s_agg.get(gap_key), digits)
        return round(_num(c_ref.get(c_key), digits) - _num(s_agg.get(s_key), digits), digits)

    # Unify Full-Vocab Macro Top-10 to Word-Macro (even if reading legacy JSON where full_vocab_macro_top10 was Subject-Macro)
    c_fv_m10 = c_ref.get("full_vocab_word_macro_top10", c_ref.get("full_vocab_macro_top10"))
    s_fv_m10 = s_agg.get("mean_full_vocab_word_macro_top10", s_agg.get("mean_full_vocab_macro_top10"))
    g_fv_m10_raw = s_agg.get(
        "clean_minus_shifted_full_vocab_word_macro_top10",
        s_agg.get("clean_minus_shifted_full_vocab_macro_top10"),
    )
    g_fv_m10 = (
        _pct(g_fv_m10_raw)
        if g_fv_m10_raw is not None and c_ref.get("full_vocab_word_macro_top10") is None
        else round(_pct(c_fv_m10) - _pct(s_fv_m10), 2)
    )

    row: dict[str, Any] = {
        # 50-Word Overall Metrics (Macro = Word-Macro, Micro = Event-Micro)
        "clean_vocab50_macro_top10_pct": _pct(c_ref.get("vocab50_macro_top10")),
        "shifted_mean_vocab50_macro_top10_pct": _pct(s_agg.get("mean_vocab50_macro_top10")),
        "clean_minus_shifted_vocab50_top10_pp": _gap_pct(
            "clean_minus_shifted_vocab50_macro_top10", "vocab50_macro_top10", "mean_vocab50_macro_top10"
        ),
        "clean_vocab50_micro_top10_pct": _pct(c_ref.get("vocab50_micro_top10")),
        "shifted_mean_vocab50_micro_top10_pct": _pct(s_agg.get("mean_vocab50_micro_top10")),
        "clean_minus_shifted_vocab50_micro_top10_pp": _gap_pct(
            "clean_minus_shifted_vocab50_micro_top10", "vocab50_micro_top10", "mean_vocab50_micro_top10"
        ),
        "clean_vocab50_macro_top1_pct": _pct(c_ref.get("vocab50_macro_top1")),
        "shifted_mean_vocab50_macro_top1_pct": _pct(s_agg.get("mean_vocab50_macro_top1")),
        "clean_minus_shifted_vocab50_top1_pp": _gap_pct(
            "clean_minus_shifted_vocab50_macro_top1", "vocab50_macro_top1", "mean_vocab50_macro_top1"
        ),
        "clean_vocab50_micro_top1_pct": _pct(c_ref.get("vocab50_micro_top1")),
        "shifted_mean_vocab50_micro_top1_pct": _pct(s_agg.get("mean_vocab50_micro_top1")),
        "clean_minus_shifted_vocab50_micro_top1_pp": _gap_pct(
            "clean_minus_shifted_vocab50_micro_top1", "vocab50_micro_top1", "mean_vocab50_micro_top1"
        ),
        "clean_vocab50_mrr": _num(c_ref.get("vocab50_mrr"), 4),
        "shifted_mean_vocab50_mrr": _num(s_agg.get("mean_vocab50_mrr"), 4),
        "clean_minus_shifted_vocab50_mrr": _gap_num(
            "clean_minus_shifted_vocab50_mrr", "vocab50_mrr", "mean_vocab50_mrr", 4
        ),
        "clean_vocab50_median_rank": _num(c_ref.get("vocab50_median_rank"), 2),
        "shifted_mean_vocab50_median_rank": _num(s_agg.get("mean_vocab50_median_rank"), 2),
        # Full-Vocabulary Overall Metrics (100% symmetric: Macro = Word-Macro, Micro = Event-Micro)
        "clean_full_vocab_macro_top10_pct": _pct(c_fv_m10),
        "shifted_mean_full_vocab_macro_top10_pct": _pct(s_fv_m10),
        "clean_minus_shifted_full_vocab_top10_pp": g_fv_m10,
        "clean_full_vocab_micro_top10_pct": _pct(c_ref.get("full_vocab_micro_top10")),
        "shifted_mean_full_vocab_micro_top10_pct": _pct(s_agg.get("mean_full_vocab_micro_top10")),
        "clean_minus_shifted_full_vocab_micro_top10_pp": _gap_pct(
            "clean_minus_shifted_full_vocab_micro_top10", "full_vocab_micro_top10", "mean_full_vocab_micro_top10"
        ),
        "clean_full_vocab_macro_top1_pct": _pct(c_ref.get("full_vocab_macro_top1")),
        "shifted_mean_full_vocab_macro_top1_pct": _pct(s_agg.get("mean_full_vocab_macro_top1")),
        "clean_minus_shifted_full_vocab_top1_pp": _gap_pct(
            "clean_minus_shifted_full_vocab_macro_top1", "full_vocab_macro_top1", "mean_full_vocab_macro_top1"
        ),
        "clean_full_vocab_micro_top1_pct": _pct(c_ref.get("full_vocab_micro_top1")),
        "shifted_mean_full_vocab_micro_top1_pct": _pct(s_agg.get("mean_full_vocab_micro_top1")),
        "clean_minus_shifted_full_vocab_micro_top1_pp": _gap_pct(
            "clean_minus_shifted_full_vocab_micro_top1", "full_vocab_micro_top1", "mean_full_vocab_micro_top1"
        ),
        "clean_full_vocab_mrr": _num(c_ref.get("full_vocab_mrr"), 4),
        "shifted_mean_full_vocab_mrr": _num(s_agg.get("mean_full_vocab_mrr"), 4),
        "clean_minus_shifted_full_vocab_mrr": _gap_num(
            "clean_minus_shifted_full_vocab_mrr", "full_vocab_mrr", "mean_full_vocab_mrr", 4
        ),
        "clean_full_vocab_median_rank": _num(c_ref.get("full_vocab_median_rank"), 2),
        "shifted_mean_full_vocab_median_rank": _num(s_agg.get("mean_full_vocab_median_rank"), 2),
    }

    if not c_hz:
        return row

    def _opt_pct(val: Any) -> float | str:
        return "" if val is None else _pct(val)

    def _opt_num(val: Any, digits: int = 4) -> float | str:
        return "" if val is None else _num(val, digits)

    for hk in ("K0", "K1", "K2"):
        pfx = hk.lower()
        ch = c_hz.get(hk) if isinstance(c_hz.get(hk), Mapping) else {}
        sh = s_hz.get(hk) if isinstance(s_hz.get(hk), Mapping) else {}

        def _hz_gap_pct(
            base_k: str,
            c_val_override: Any = None,
            s_val_override: Any = None,
            _ch: Mapping[str, Any] = ch,
            _sh: Mapping[str, Any] = sh,
        ) -> float | str:
            c_raw = c_val_override if c_val_override is not None else _ch.get(base_k)
            s_raw = s_val_override if s_val_override is not None else _sh.get(f"mean_{base_k}")
            if c_raw is None or s_raw is None:
                return ""
            return round(_pct(c_raw) - _pct(s_raw), 2)

        def _hz_gap_num(
            base_k: str,
            digits: int = 4,
            _ch: Mapping[str, Any] = ch,
            _sh: Mapping[str, Any] = sh,
        ) -> float | str:
            c_raw = _ch.get(base_k)
            s_raw = _sh.get(f"mean_{base_k}")
            if c_raw is None or s_raw is None:
                return ""
            return round(_num(c_raw, digits) - _num(s_raw, digits), digits)

        # 50w Horizon Metrics
        row[f"{pfx}_clean_vocab50_macro_top10_pct"] = _opt_pct(ch.get("common_vocab50_macro_top10"))
        row[f"{pfx}_shifted_vocab50_macro_top10_pct"] = _opt_pct(sh.get("mean_common_vocab50_macro_top10"))
        row[f"{pfx}_gap_vocab50_macro_top10_pp"] = _hz_gap_pct("common_vocab50_macro_top10")

        row[f"{pfx}_clean_vocab50_macro_top1_pct"] = _opt_pct(ch.get("common_vocab50_macro_top1"))
        row[f"{pfx}_shifted_vocab50_macro_top1_pct"] = _opt_pct(sh.get("mean_common_vocab50_macro_top1"))
        row[f"{pfx}_gap_vocab50_macro_top1_pp"] = _hz_gap_pct("common_vocab50_macro_top1")

        row[f"{pfx}_clean_vocab50_micro_top10_pct"] = _opt_pct(ch.get("common_vocab50_micro_top10"))
        row[f"{pfx}_shifted_vocab50_micro_top10_pct"] = _opt_pct(sh.get("mean_common_vocab50_micro_top10"))
        row[f"{pfx}_gap_vocab50_micro_top10_pp"] = _hz_gap_pct("common_vocab50_micro_top10")

        row[f"{pfx}_clean_vocab50_micro_top1_pct"] = _opt_pct(ch.get("common_vocab50_micro_top1"))
        row[f"{pfx}_shifted_vocab50_micro_top1_pct"] = _opt_pct(sh.get("mean_common_vocab50_micro_top1"))
        row[f"{pfx}_gap_vocab50_micro_top1_pp"] = _hz_gap_pct("common_vocab50_micro_top1")

        row[f"{pfx}_clean_vocab50_mrr"] = _opt_num(ch.get("common_vocab50_mrr"), 4)
        row[f"{pfx}_shifted_vocab50_mrr"] = _opt_num(sh.get("mean_common_vocab50_mrr"), 4)
        row[f"{pfx}_gap_vocab50_mrr"] = _hz_gap_num("common_vocab50_mrr", 4)

        row[f"{pfx}_clean_vocab50_median_rank"] = _opt_num(ch.get("common_vocab50_median_rank"), 2)
        row[f"{pfx}_shifted_vocab50_median_rank"] = _opt_num(sh.get("mean_common_vocab50_median_rank"), 2)

        # Full-Vocab Horizon Metrics (Macro = Word-Macro)
        ch_fv_m10 = ch.get("common_full_vocab_word_macro_top10", ch.get("common_full_vocab_macro_top10"))
        sh_fv_m10 = sh.get("mean_common_full_vocab_word_macro_top10", sh.get("mean_common_full_vocab_macro_top10"))
        row[f"{pfx}_clean_full_vocab_macro_top10_pct"] = _opt_pct(ch_fv_m10)
        row[f"{pfx}_shifted_full_vocab_macro_top10_pct"] = _opt_pct(sh_fv_m10)
        row[f"{pfx}_gap_full_vocab_macro_top10_pp"] = _hz_gap_pct(
            "common_full_vocab_macro_top10", c_val_override=ch_fv_m10, s_val_override=sh_fv_m10
        )

        row[f"{pfx}_clean_full_vocab_macro_top1_pct"] = _opt_pct(ch.get("common_full_vocab_macro_top1"))
        row[f"{pfx}_shifted_full_vocab_macro_top1_pct"] = _opt_pct(sh.get("mean_common_full_vocab_macro_top1"))
        row[f"{pfx}_gap_full_vocab_macro_top1_pp"] = _hz_gap_pct("common_full_vocab_macro_top1")

        row[f"{pfx}_clean_full_vocab_micro_top10_pct"] = _opt_pct(ch.get("common_full_vocab_micro_top10"))
        row[f"{pfx}_shifted_full_vocab_micro_top10_pct"] = _opt_pct(sh.get("mean_common_full_vocab_micro_top10"))
        row[f"{pfx}_gap_full_vocab_micro_top10_pp"] = _hz_gap_pct("common_full_vocab_micro_top10")

        row[f"{pfx}_clean_full_vocab_micro_top1_pct"] = _opt_pct(ch.get("common_full_vocab_micro_top1"))
        row[f"{pfx}_shifted_full_vocab_micro_top1_pct"] = _opt_pct(sh.get("mean_common_full_vocab_micro_top1"))
        row[f"{pfx}_gap_full_vocab_micro_top1_pp"] = _hz_gap_pct("common_full_vocab_micro_top1")

        row[f"{pfx}_clean_full_vocab_mrr"] = _opt_num(ch.get("common_full_vocab_mrr"), 4)
        row[f"{pfx}_shifted_full_vocab_mrr"] = _opt_num(sh.get("mean_common_full_vocab_mrr"), 4)
        row[f"{pfx}_gap_full_vocab_mrr"] = _hz_gap_num("common_full_vocab_mrr", 4)

        row[f"{pfx}_clean_full_vocab_median_rank"] = _opt_num(ch.get("common_full_vocab_median_rank"), 2)
        row[f"{pfx}_shifted_full_vocab_median_rank"] = _opt_num(sh.get("mean_common_full_vocab_median_rank"), 2)

    def _diff_opt(k2_key: str, k0_key: str) -> float | str:
        v2 = row.get(k2_key)
        v0 = row.get(k0_key)
        if isinstance(v2, (int, float)) and isinstance(v0, (int, float)):
            return round(float(v2) - float(v0), 2)
        return ""

    # K2 - K0 Revision Gains
    row["k2_minus_k0_clean_vocab50_macro_top10_pp"] = _diff_opt(
        "k2_clean_vocab50_macro_top10_pct", "k0_clean_vocab50_macro_top10_pct"
    )
    row["k2_minus_k0_shifted_vocab50_macro_top10_pp"] = _diff_opt(
        "k2_shifted_vocab50_macro_top10_pct", "k0_shifted_vocab50_macro_top10_pct"
    )
    row["k2_minus_k0_clean_vocab50_macro_top1_pp"] = _diff_opt(
        "k2_clean_vocab50_macro_top1_pct", "k0_clean_vocab50_macro_top1_pct"
    )
    row["k2_minus_k0_shifted_vocab50_macro_top1_pp"] = _diff_opt(
        "k2_shifted_vocab50_macro_top1_pct", "k0_shifted_vocab50_macro_top1_pct"
    )
    row["k2_minus_k0_clean_full_vocab_macro_top10_pp"] = _diff_opt(
        "k2_clean_full_vocab_macro_top10_pct", "k0_clean_full_vocab_macro_top10_pct"
    )
    row["k2_minus_k0_shifted_full_vocab_macro_top10_pp"] = _diff_opt(
        "k2_shifted_full_vocab_macro_top10_pct", "k0_shifted_full_vocab_macro_top10_pct"
    )
    row["k2_minus_k0_clean_full_vocab_macro_top1_pp"] = _diff_opt(
        "k2_clean_full_vocab_macro_top1_pct", "k0_clean_full_vocab_macro_top1_pct"
    )
    row["k2_minus_k0_shifted_full_vocab_macro_top1_pp"] = _diff_opt(
        "k2_shifted_full_vocab_macro_top1_pct", "k0_shifted_full_vocab_macro_top1_pct"
    )
    return row


def _extract_debiasing_horizon_rows(
    base_meta: Mapping[str, Any],
    c_ref: Mapping[str, Any],
    s_agg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """展开每个方法在 K0 / K1 / K2 三个回修步上的独立明细行，供 *_horizons_{split}.csv 导出。"""
    c_hz = c_ref.get("horizons") if isinstance(c_ref.get("horizons"), Mapping) else {}
    s_hz = s_agg.get("horizons") if isinstance(s_agg.get("horizons"), Mapping) else {}
    hz_rows: list[dict[str, Any]] = []
    for hk in ("K0", "K1", "K2"):
        ch = c_hz.get(hk) if isinstance(c_hz.get(hk), Mapping) else {}
        sh = s_hz.get(hk) if isinstance(s_hz.get(hk), Mapping) else {}
        if not ch and not sh:
            continue
        row: dict[str, Any] = dict(base_meta)
        row["horizon"] = hk
        for base_k, out_prefix, is_pct in (
            ("common_vocab50_macro_top10", "vocab50_macro_top10", True),
            ("common_vocab50_micro_top10", "vocab50_micro_top10", True),
            ("common_vocab50_macro_top1", "vocab50_macro_top1", True),
            ("common_vocab50_micro_top1", "vocab50_micro_top1", True),
            ("common_vocab50_mrr", "vocab50_mrr", False),
            ("common_vocab50_median_rank", "vocab50_median_rank", False),
            ("common_full_vocab_macro_top10", "full_vocab_macro_top10", True),
            ("common_full_vocab_micro_top10", "full_vocab_micro_top10", True),
            ("common_full_vocab_macro_top1", "full_vocab_macro_top1", True),
            ("common_full_vocab_micro_top1", "full_vocab_micro_top1", True),
            ("common_full_vocab_mrr", "full_vocab_mrr", False),
            ("common_full_vocab_median_rank", "full_vocab_median_rank", False),
        ):
            if base_k == "common_full_vocab_macro_top10":
                c_raw = ch.get("common_full_vocab_word_macro_top10", ch.get("common_full_vocab_macro_top10"))
                s_mean_raw = sh.get(
                    "mean_common_full_vocab_word_macro_top10", sh.get("mean_common_full_vocab_macro_top10")
                )
                s_std_raw = sh.get(
                    "std_common_full_vocab_word_macro_top10", sh.get("std_common_full_vocab_macro_top10")
                )
            else:
                c_raw = ch.get(base_k)
                s_mean_raw = sh.get(f"mean_{base_k}")
                s_std_raw = sh.get(f"std_{base_k}")
            if is_pct:
                if c_raw is None and s_mean_raw is None:
                    row[f"clean_{out_prefix}_pct"] = ""
                    row[f"shifted_mean_{out_prefix}_pct"] = ""
                    row[f"shifted_std_{out_prefix}_pct"] = ""
                    row[f"clean_minus_shifted_{out_prefix}_pp"] = ""
                else:
                    c_val = _pct(c_raw)
                    s_mean = _pct(s_mean_raw)
                    s_std = _pct(s_std_raw)
                    gap_val = round(c_val - s_mean, 2)
                    row[f"clean_{out_prefix}_pct"] = c_val
                    row[f"shifted_mean_{out_prefix}_pct"] = s_mean
                    row[f"shifted_std_{out_prefix}_pct"] = s_std
                    row[f"clean_minus_shifted_{out_prefix}_pp"] = gap_val
            else:
                digits = 2 if "median_rank" in base_k else 4
                if c_raw is None and s_mean_raw is None:
                    row[f"clean_{out_prefix}"] = ""
                    row[f"shifted_mean_{out_prefix}"] = ""
                    row[f"shifted_std_{out_prefix}"] = ""
                    row[f"clean_minus_shifted_{out_prefix}"] = ""
                else:
                    c_val = _num(c_raw, digits)
                    s_mean = _num(s_mean_raw, digits)
                    s_std = _num(s_std_raw, digits)
                    gap_val = round(c_val - s_mean, digits)
                    row[f"clean_{out_prefix}"] = c_val
                    row[f"shifted_mean_{out_prefix}"] = s_mean
                    row[f"shifted_std_{out_prefix}"] = s_std
                    row[f"clean_minus_shifted_{out_prefix}"] = gap_val
        hz_rows.append(row)
    return hz_rows


def _append_debiasing_markdown_tables(
    md_lines: list[str],
    rows: list[dict[str, Any]],
    *,
    section_offset: int = 1,
    include_subspace_meta: bool = False,
) -> None:
    """将 50 词总表、全词表总表（含 Macro Top-1）及 K0/K1/K2 50词与全词表拆解表追加至 Markdown。"""
    s1 = section_offset
    s2 = section_offset + 1
    s3a = f"{section_offset + 2}A"
    s3b = f"{section_offset + 2}B"

    if include_subspace_meta:
        meta_hdr = "| Method Configuration | Rank `K` | Weight `η` | Gate |"
        meta_sep = "| :--- | :---: | ---: | :---: |"

        def _meta_cols(r: Mapping[str, Any]) -> str:
            return f"| **`{r['label']}`** | `{r['rank_k']}` | `{r['eta']:.2f}` | `{r['coherence_gate']}` |"
    else:
        meta_hdr = "| Subtraction Weight (`η`) |"
        meta_sep = "| :--- |"

        def _meta_cols(r: Mapping[str, Any]) -> str:
            return f"| **`{r['label']}`** |"

    def _cell_pct(val: Any, bold: bool = False) -> str:
        if not isinstance(val, (int, float)):
            return "-"
        txt = f"{float(val):.2f}%"
        return f"**{txt}**" if bold else txt

    def _cell_pp(val: Any, bold: bool = False) -> str:
        if not isinstance(val, (int, float)):
            return "-"
        txt = f"{float(val):+.2f} pp"
        return f"**{txt}**" if bold else txt

    def _cell_num(val: Any, digits: int = 4) -> str:
        if not isinstance(val, (int, float)):
            return "-"
        return f"{float(val):.{digits}f}"

    # Table 1: 50-Word Retrieval Summary
    md_lines.extend(
        [
            f"## {s1}. 50-Word Retrieval Summary (Word-Macro & Micro Top-10, Top-1, MRR, Median Rank)",
            "",
            f"{meta_hdr} 50w Clean Macro Top-10 | 50w Shift Macro Top-10 | 50w Top-10 Gap | 50w Clean Macro Top-1 | 50w Shift Macro Top-1 | 50w Top-1 Gap | 50w Clean Micro Top-10 | 50w Shift Micro Top-10 | 50w Clean MRR | 50w Shift MRR | 50w Clean MedRank | 50w Shift MedRank |",
            f"{meta_sep} ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rows:
        md_lines.append(
            f"{_meta_cols(r)} {_cell_pct(r.get('clean_vocab50_macro_top10_pct'), bold=True)} | "
            f"{_cell_pct(r.get('shifted_mean_vocab50_macro_top10_pct'))} | "
            f"{_cell_pp(r.get('clean_minus_shifted_vocab50_top10_pp'), bold=True)} | "
            f"{_cell_pct(r.get('clean_vocab50_macro_top1_pct'), bold=True)} | {_cell_pct(r.get('shifted_mean_vocab50_macro_top1_pct'))} | "
            f"{_cell_pp(r.get('clean_minus_shifted_vocab50_top1_pp'))} | "
            f"{_cell_pct(r.get('clean_vocab50_micro_top10_pct'))} | {_cell_pct(r.get('shifted_mean_vocab50_micro_top10_pct'))} | "
            f"{_cell_num(r.get('clean_vocab50_mrr'), 4)} | {_cell_num(r.get('shifted_mean_vocab50_mrr'), 4)} | "
            f"{_cell_num(r.get('clean_vocab50_median_rank'), 1)} | {_cell_num(r.get('shifted_mean_vocab50_median_rank'), 1)} |"
        )
    md_lines.append("")

    # Table 2: Full-Vocabulary Retrieval Summary (100% symmetric with Table 1: Macro = Word-Macro, Micro = Event-Micro)
    md_lines.extend(
        [
            f"## {s2}. Full-Vocabulary Retrieval Summary (Word-Macro & Micro Top-10, Top-1, MRR, Median Rank)",
            "",
            f"{meta_hdr} Full Clean Macro Top-10 | Full Shift Macro Top-10 | Full Top-10 Gap | Full Clean Macro Top-1 | Full Shift Macro Top-1 | Full Top-1 Gap | Full Clean Micro Top-10 | Full Shift Micro Top-10 | Full Clean MRR | Full Shift MRR | Full Clean MedRank | Full Shift MedRank |",
            f"{meta_sep} ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rows:
        md_lines.append(
            f"{_meta_cols(r)} {_cell_pct(r.get('clean_full_vocab_macro_top10_pct'), bold=True)} | "
            f"{_cell_pct(r.get('shifted_mean_full_vocab_macro_top10_pct'))} | "
            f"{_cell_pp(r.get('clean_minus_shifted_full_vocab_top10_pp'), bold=True)} | "
            f"{_cell_pct(r.get('clean_full_vocab_macro_top1_pct'), bold=True)} | "
            f"{_cell_pct(r.get('shifted_mean_full_vocab_macro_top1_pct'))} | "
            f"{_cell_pp(r.get('clean_minus_shifted_full_vocab_top1_pp'), bold=True)} | "
            f"{_cell_pct(r.get('clean_full_vocab_micro_top10_pct'))} | {_cell_pct(r.get('shifted_mean_full_vocab_micro_top10_pct'))} | "
            f"{_cell_num(r.get('clean_full_vocab_mrr'), 4)} | {_cell_num(r.get('shifted_mean_full_vocab_mrr'), 4)} | "
            f"{_cell_num(r.get('clean_full_vocab_median_rank'), 1)} | {_cell_num(r.get('shifted_mean_full_vocab_median_rank'), 1)} |"
        )
    md_lines.append("")

    has_horizons = bool(rows) and "k0_clean_vocab50_macro_top10_pct" in rows[0] and rows[0].get("k0_clean_vocab50_macro_top10_pct") != ""
    if not has_horizons:
        return

    # Table 3A: Revision Horizons (K0 / K1 / K2) — 50-Word Breakdown
    md_lines.extend(
        [
            f"## {s3a}. Revision Horizons (`K0 / K1 / K2`) — 50-Word Macro Top-10 & Macro Top-1 Breakdown",
            "",
            f"{meta_hdr} K0 50w Clean Top-10 | K0 50w Shift Top-10 | K1 50w Clean Top-10 | K1 50w Shift Top-10 | K2 50w Clean Top-10 | K2 50w Shift Top-10 | K2 50w Gap Top-10 | K2-K0 Clean Top-10 | K0 50w Clean Top-1 | K0 50w Shift Top-1 | K1 50w Clean Top-1 | K1 50w Shift Top-1 | K2 50w Clean Top-1 | K2 50w Shift Top-1 | K2 50w Gap Top-1 | K2-K0 Clean Top-1 |",
            f"{meta_sep} ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rows:
        md_lines.append(
            f"{_meta_cols(r)} {_cell_pct(r.get('k0_clean_vocab50_macro_top10_pct'))} | {_cell_pct(r.get('k0_shifted_vocab50_macro_top10_pct'))} | "
            f"{_cell_pct(r.get('k1_clean_vocab50_macro_top10_pct'))} | {_cell_pct(r.get('k1_shifted_vocab50_macro_top10_pct'))} | "
            f"{_cell_pct(r.get('k2_clean_vocab50_macro_top10_pct'), bold=True)} | {_cell_pct(r.get('k2_shifted_vocab50_macro_top10_pct'))} | "
            f"{_cell_pp(r.get('k2_gap_vocab50_macro_top10_pp'), bold=True)} | {_cell_pp(r.get('k2_minus_k0_clean_vocab50_macro_top10_pp'))} | "
            f"{_cell_pct(r.get('k0_clean_vocab50_macro_top1_pct'))} | {_cell_pct(r.get('k0_shifted_vocab50_macro_top1_pct'))} | "
            f"{_cell_pct(r.get('k1_clean_vocab50_macro_top1_pct'))} | {_cell_pct(r.get('k1_shifted_vocab50_macro_top1_pct'))} | "
            f"{_cell_pct(r.get('k2_clean_vocab50_macro_top1_pct'), bold=True)} | {_cell_pct(r.get('k2_shifted_vocab50_macro_top1_pct'))} | "
            f"{_cell_pp(r.get('k2_gap_vocab50_macro_top1_pp'), bold=True)} | {_cell_pp(r.get('k2_minus_k0_clean_vocab50_macro_top1_pp'))} |"
        )
    md_lines.append("")

    # Table 3B: Revision Horizons (K0 / K1 / K2) — Full-Vocabulary Breakdown
    md_lines.extend(
        [
            f"## {s3b}. Revision Horizons (`K0 / K1 / K2`) — Full-Vocabulary Macro Top-10, Macro Top-1 & MRR Breakdown",
            "",
            f"{meta_hdr} K0 Full Clean Top-10 | K0 Full Shift Top-10 | K1 Full Clean Top-10 | K1 Full Shift Top-10 | K2 Full Clean Top-10 | K2 Full Shift Top-10 | K2 Full Gap Top-10 | K2-K0 Full Clean Top-10 | K0 Full Clean Top-1 | K0 Full Shift Top-1 | K1 Full Clean Top-1 | K1 Full Shift Top-1 | K2 Full Clean Top-1 | K2 Full Shift Top-1 | K2 Full Gap Top-1 | K2-K0 Full Clean Top-1 | K0 Full Clean MRR | K2 Full Clean MRR |",
            f"{meta_sep} ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rows:
        md_lines.append(
            f"{_meta_cols(r)} {_cell_pct(r.get('k0_clean_full_vocab_macro_top10_pct'))} | {_cell_pct(r.get('k0_shifted_full_vocab_macro_top10_pct'))} | "
            f"{_cell_pct(r.get('k1_clean_full_vocab_macro_top10_pct'))} | {_cell_pct(r.get('k1_shifted_full_vocab_macro_top10_pct'))} | "
            f"{_cell_pct(r.get('k2_clean_full_vocab_macro_top10_pct'), bold=True)} | {_cell_pct(r.get('k2_shifted_full_vocab_macro_top10_pct'))} | "
            f"{_cell_pp(r.get('k2_gap_full_vocab_macro_top10_pp'), bold=True)} | {_cell_pp(r.get('k2_minus_k0_clean_full_vocab_macro_top10_pp'))} | "
            f"{_cell_pct(r.get('k0_clean_full_vocab_macro_top1_pct'))} | {_cell_pct(r.get('k0_shifted_full_vocab_macro_top1_pct'))} | "
            f"{_cell_pct(r.get('k1_clean_full_vocab_macro_top1_pct'))} | {_cell_pct(r.get('k1_shifted_full_vocab_macro_top1_pct'))} | "
            f"{_cell_pct(r.get('k2_clean_full_vocab_macro_top1_pct'), bold=True)} | {_cell_pct(r.get('k2_shifted_full_vocab_macro_top1_pct'))} | "
            f"{_cell_pp(r.get('k2_gap_full_vocab_macro_top1_pp'), bold=True)} | {_cell_pp(r.get('k2_minus_k0_clean_full_vocab_macro_top1_pp'))} | "
            f"{_cell_num(r.get('k0_clean_full_vocab_mrr'), 4)} | {_cell_num(r.get('k2_clean_full_vocab_mrr'), 4)} |"
        )
    md_lines.append("")


def export_common_mode_subtraction_tables_and_markdown(
    cms_report: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """将留一法共模扣除（Leave-One-Out Common-Mode Subtraction）评估结果导出为 tables/*.csv 与 exploratory/*.md。"""
    target_dir_raw = output_dir
    if not target_dir_raw:
        json_p = cms_report.get("exploratory_json")
        if json_p:
            p = Path(str(json_p))
            target_dir_raw = p.parent.parent if p.parent.name == "exploratory" else p.parent
    if not target_dir_raw:
        return {}
    out_dir = Path(str(target_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    exploratory_dir = out_dir / "exploratory"
    exploratory_dir.mkdir(parents=True, exist_ok=True)

    split = str(cms_report.get("split", "test"))
    dataset = str(cms_report.get("dataset", "-"))
    stage = str(cms_report.get("stage", "revision"))
    seed = cms_report.get("seed", 0)
    exp_tag = str(cms_report.get("exp_tag", "")) or "default"

    raw_entry = cms_report.get("baseline_raw_eta0") or {}
    sweep_entries = list(
        cms_report.get("sweep") or cms_report.get("subtraction_sweep") or []
    )
    all_entries = (
        ([raw_entry] + sweep_entries)
        if (raw_entry and raw_entry not in sweep_entries)
        else sweep_entries
    )

    exported: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for entry in all_entries:
        if not isinstance(entry, Mapping):
            continue
        c_ref = entry.get("clean_reference") or {}
        s_agg = entry.get("shifted_aggregate") or {}
        base_meta = {
            "eta": _num(entry.get("eta", 0.0), 3),
            "label": str(entry.get("label", f"eta={entry.get('eta', 0.0)}")),
        }
        row = dict(base_meta)
        row.update(_extract_debiasing_row_metrics(c_ref, s_agg))
        rows.append(row)
        horizon_rows.extend(_extract_debiasing_horizon_rows(base_meta, c_ref, s_agg))

    if rows:
        csv_path = _write_csv_table(
            tables_dir / f"common_mode_subtraction_{split}.csv",
            list(rows[0].keys()),
            rows,
        )
        exported["common_mode_subtraction_csv"] = str(csv_path)

    if horizon_rows:
        hz_csv_path = _write_csv_table(
            tables_dir / f"common_mode_subtraction_horizons_{split}.csv",
            list(horizon_rows[0].keys()),
            horizon_rows,
        )
        exported["common_mode_subtraction_horizons_csv"] = str(hz_csv_path)

    md_lines = [
        f"# Leave-One-Out Common-Mode Subtraction Report — `{dataset}` | `{stage}` (`{split.upper()}`)",
        "",
        f"- **Experiment Tag**: `{exp_tag}` | **Seed**: `{seed}`",
        "",
    ]
    _append_debiasing_markdown_tables(
        md_lines,
        rows,
        section_offset=1,
        include_subspace_meta=False,
    )

    md_path = exploratory_dir / f"common_mode_subtraction_{split}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    exported["common_mode_subtraction_md"] = str(md_path)
    return exported


def export_subspace_coherence_debiasing_tables_and_markdown(
    subspace_report: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """将方案一（中心化 SVD 低秩子空间投影 + 严格留一法基底 LOO 去偏）导出为 tables/*.csv 与 exploratory/*.md。"""
    target_dir_raw = output_dir
    if not target_dir_raw:
        json_p = subspace_report.get("exploratory_json")
        if json_p:
            p = Path(str(json_p))
            target_dir_raw = p.parent.parent if p.parent.name == "exploratory" else p.parent
    if not target_dir_raw:
        return {}
    out_dir = Path(str(target_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    exploratory_dir = out_dir / "exploratory"
    exploratory_dir.mkdir(parents=True, exist_ok=True)

    split = str(subspace_report.get("split", "test"))
    dataset = str(subspace_report.get("dataset", "-"))
    stage = str(subspace_report.get("stage", "revision"))
    seed = subspace_report.get("seed", 0)
    exp_tag = str(subspace_report.get("exp_tag", "")) or "default"

    raw_entry = subspace_report.get("raw_baseline") or {}
    full_entries = list(subspace_report.get("full_rank_loo_baselines") or [])
    sub_entries = list(subspace_report.get("subspace_sweep") or [])
    all_entries = ([raw_entry] if raw_entry else []) + full_entries + sub_entries

    exported: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    for entry in all_entries:
        if not isinstance(entry, Mapping) or not entry:
            continue
        c_ref = entry.get("clean_reference") or {}
        s_agg = entry.get("shifted_aggregate") or {}
        rk = entry.get("rank_k")
        base_meta = {
            "method_id": str(entry.get("method_id", "-")),
            "method_family": str(entry.get("method_family", "-")),
            "label": str(entry.get("label", "-")),
            "rank_k": "Full" if rk is None else int(rk),
            "eta": _num(entry.get("eta", 0.0), 3),
            "coherence_gate": "On" if entry.get("use_coherence_gate") else "Off",
        }
        row = dict(base_meta)
        row.update(_extract_debiasing_row_metrics(c_ref, s_agg))
        rows.append(row)
        horizon_rows.extend(_extract_debiasing_horizon_rows(base_meta, c_ref, s_agg))

    if rows:
        csv_path = _write_csv_table(
            tables_dir / f"subspace_debiasing_sweep_{split}.csv",
            list(rows[0].keys()),
            rows,
        )
        exported["subspace_debiasing_sweep_csv"] = str(csv_path)

    if horizon_rows:
        hz_csv_path = _write_csv_table(
            tables_dir / f"subspace_debiasing_horizons_{split}.csv",
            list(horizon_rows[0].keys()),
            horizon_rows,
        )
        exported["subspace_debiasing_horizons_csv"] = str(hz_csv_path)

    diag_map = subspace_report.get("subspace_and_coherence_diagnostics") or {}
    best_gap = subspace_report.get("best_by_net_gap_vocab50") or {}
    best_clean = subspace_report.get("best_by_clean_preservation_vocab50") or {}

    md_lines = [
        f"# Subspace-Projected & Coherence-Gated LOO Debiasing Report — `{dataset}` | `{stage}` (`{split.upper()}`)",
        "",
        f"- **Experiment Tag**: `{exp_tag}` | **Seed**: `{seed}` | **Centered SVD**: `{subspace_report.get('formula', {}).get('center_svd', True)}`",
    ]
    if isinstance(best_gap, Mapping) and best_gap.get("label"):
        bg_c = _pct((best_gap.get("clean_reference") or {}).get("vocab50_macro_top10"))
        bg_s = _pct((best_gap.get("shifted_aggregate") or {}).get("mean_vocab50_macro_top10"))
        bg_d = _pct((best_gap.get("shifted_aggregate") or {}).get("clean_minus_shifted_vocab50_macro_top10"))
        md_lines.append(
            f"- **Best by Net Gap (`Clean - Shifted`)**: **`{best_gap['label']}`** (`Clean={bg_c:.2f}%`, `Shifted={bg_s:.2f}%`, `Gap={bg_d:+.2f} pp`)"
        )
    if isinstance(best_clean, Mapping) and best_clean.get("label"):
        bc_c = _pct((best_clean.get("clean_reference") or {}).get("vocab50_macro_top10"))
        bc_s = _pct((best_clean.get("shifted_aggregate") or {}).get("mean_vocab50_macro_top10"))
        bc_d = _pct((best_clean.get("shifted_aggregate") or {}).get("clean_minus_shifted_vocab50_macro_top10"))
        md_lines.append(
            f"- **Best Clean-Preserving**: **`{best_clean['label']}`** (`Clean={bc_c:.2f}%`, `Shifted={bc_s:.2f}%`, `Gap={bc_d:+.2f} pp`)"
        )
    md_lines.append("")

    section_offset = 1
    if isinstance(diag_map, Mapping) and diag_map:
        md_lines.extend(
            [
                "## 1. Centered SVD Bias Subspace & Coherence Diagnostics",
                "",
                "| Horizon / Key | Effective Rank | Top-4 Energy | Top-8 Energy | Top-16 Energy | Mean Coherence Norm | Random Orth Norm |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for hk, d_info in diag_map.items():
            if not isinstance(d_info, Mapping):
                continue
            cum_e = d_info.get("cumulative_energy_explained") or {}
            c_norm = d_info.get("coherence_norm") or {}
            md_lines.append(
                f"| **`{hk}`** | `{_num(d_info.get('effective_rank', 0.0), 2):.2f}` | "
                f"{_pct(cum_e.get('top_4', 0.0)):.2f}% | **{_pct(cum_e.get('top_8', 0.0)):.2f}%** | "
                f"{_pct(cum_e.get('top_16', 0.0)):.2f}% | `{_num(c_norm.get('mean', 0.0), 3):.3f}` | "
                f"`{_num(d_info.get('theoretical_random_orthogonal_norm', 0.408), 3):.3f}` |"
            )
        md_lines.append("")
        section_offset = 2

    _append_debiasing_markdown_tables(
        md_lines,
        rows,
        section_offset=section_offset,
        include_subspace_meta=True,
    )

    md_path = exploratory_dir / f"subspace_coherence_debiasing_{split}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    exported["subspace_coherence_debiasing_md"] = str(md_path)
    return exported


__all__ = [
    "export_common_mode_subtraction_tables_and_markdown",
    "export_multi_seed_tables_and_markdown",
    "export_sequence_decoding_tables_and_markdown",
    "export_stage_comparison_tables_and_markdown",
    "export_subspace_coherence_debiasing_tables_and_markdown",
    "export_word_evaluation_tables_and_markdown",
]
