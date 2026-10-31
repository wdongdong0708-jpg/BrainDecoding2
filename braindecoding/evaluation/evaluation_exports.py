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


def _append_p_str(
    cell_text: str,
    p_str: Any,
    *,
    bold_val: bool = False,
    prefix: str = "",
) -> str:
    """在表格单元格文本后附加统计检验 p 值标注（如 '**+36.37 ± 0.00 pp** (p<0.0001 ***)'）。"""
    if cell_text == "-":
        return "-"
    base = f"**{cell_text}**" if bold_val and not cell_text.startswith("**") else cell_text
    p_clean = str(p_str or "").strip()
    if not p_clean:
        return base
    if prefix:
        return f"{base} ({prefix}{p_clean})"
    return f"{base} ({p_clean})"


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
                    "p_value_vocab50_macro_top10": cat.get("p_value_vocab50_word_macro_top10", ""),
                    "significance_vocab50_macro_top10": cat.get("sig_vocab50_word_macro_top10", ""),
                    "vocab50_micro_top10_pct": _pct(
                        cat.get("vocab50_micro_top10", 0.0)
                    ),
                    "vocab50_macro_top1_pct": _pct(
                        cat.get("vocab50_word_macro_top1", cat.get("vocab50_macro_top1", 0.0))
                    ),
                    "p_value_vocab50_macro_top1": cat.get("p_value_vocab50_word_macro_top1", ""),
                    "significance_vocab50_macro_top1": cat.get("sig_vocab50_word_macro_top1", ""),
                    "vocab50_micro_top1_pct": _pct(
                        cat.get("vocab50_micro_top1", 0.0)
                    ),
                    "vocab50_mrr": _num(cat.get("vocab50_mrr", 0.0), 4),
                    "full_vocab_macro_top10_pct": _pct(
                        cat.get("full_vocab_word_macro_top10", cat.get("full_vocab_macro_top10", 0.0))
                    ),
                    "p_value_full_vocab_macro_top10": cat.get("p_value_full_vocab_word_macro_top10", ""),
                    "significance_full_vocab_macro_top10": cat.get("sig_full_vocab_word_macro_top10", ""),
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
                f"> **Executive Headline**: 50-Word Word-Macro Top-10 = **{_fmt_pct(n50_top10)}** (`{n50_fold:.2f}x` chance `{_fmt_pct(n50_chance)}`), "
                f"50-Word Event-Micro Top-10 = **{_fmt_pct(n50_d.get('micro_top10'))}**, 50-Word Word-Macro Top-1 = **{_fmt_pct(n50_d.get('macro_top1'))}** | "
                f"Full-Vocab (`|V|={fv_d.get('candidate_count', '-')}`) Word-Macro Top-10 = **{_fmt_pct(fv_top10)}** (`{fv_fold:.2f}x` chance `{_fmt_pct(fv_chance)}`), "
                f"Full-Vocab Event-Micro Top-10 = **{_fmt_pct(fv_micro10)}**, Full-Vocab Word-Macro Top-1 = **{_fmt_pct(fv_d.get('macro_top1', 0.0))}**.",
                "",
            ]
        )

    if vocab_rows:
        md_lines.extend(
            [
                "## 1. Multi-Scale Vocabulary Retrieval Metrics",
                "",
                "| Scale | OOV Policy | Candidates | Queries (Cov%) | Word-Macro Top-10 | Event-Micro Top-10 | Subject-Macro Top-10 | Chance Top-10 (Fold) | Word-Macro Top-1 | Event-Micro Top-1 | MRR | Median Rank (Chance) |",
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
                "| Horizon | Delay | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | Full-Vocab Word-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab Word-Macro Top-1 | Full-Vocab MRR |",
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
                    "| Transition | Criterion | Common Events | Beneficial Repairs (`Miss->Hit`) | Harmful Overwrites (`Hit->Miss`) | Retained Hits (`Hit->Hit`) | Net Gain Events (Rate) | Repair / Harm Ratio | McNemar p-value |",
                    "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
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
                    ("vocab50_top10", "50-Word Top-10"),
                    ("vocab50_top1", "50-Word Top-1"),
                    ("full_vocab_top10", "Full-Vocab Top-10"),
                    ("full_vocab_top1", "Full-Vocab Top-1"),
                ):
                    td = t_group.get(crit_key)
                    if not isinstance(td, Mapping):
                        continue
                    mc_str = str(td.get("mcnemar_p_str", "-") or "-")
                    md_lines.append(
                        f"| **{t_label}** | `{crit_label}` | {td.get('total_common_events', 0)} | "
                        f"**+{td.get('beneficial_repairs', 0)}** ({_num(td.get('beneficial_repair_rate_pct', 0.0), 2):.2f}%) | "
                        f"-{td.get('harmful_overwrites', 0)} ({_num(td.get('harmful_overwrite_rate_pct', 0.0), 2):.2f}%) | "
                        f"{td.get('retained_hits', 0)} | "
                        f"**{int(td.get('net_gain_events', 0)):+d}** ({_num(td.get('net_gain_pct', 0.0), 2):+.2f}%) | "
                        f"`{_num(td.get('repair_to_harm_ratio', 0.0), 2):.2f}x` | "
                        f"`{mc_str}` |"
                    )
            md_lines.append("")

    if isinstance(ling, Mapping) and ling:
        md_lines.extend(
            [
                "## 3. Content Words vs. Function Words Breakdown",
                "",
                "| Word Class | Events (Share) | Unique Words | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | Full-Vocab Word-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab MRR |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for cat_k in ("content_words", "function_words", "content_minus_function"):
            cd = ling.get(cat_k)
            if not isinstance(cd, Mapping):
                continue
            is_d = cat_k == "content_minus_function"
            fmt_fn = _fmt_signed_pct if is_d else _fmt_pct
            v50_m10_txt = _append_p_str(
                fmt_fn(cd.get("vocab50_word_macro_top10", cd.get("vocab50_macro_top10"))),
                cd.get("p_str_vocab50_word_macro_top10") if is_d else "",
                bold_val=True,
            )
            v50_u10_txt = _append_p_str(
                fmt_fn(cd.get("vocab50_micro_top10")),
                cd.get("p_str_vocab50_micro_top10") if is_d else "",
            )
            v50_m1_txt = _append_p_str(
                fmt_fn(cd.get("vocab50_word_macro_top1", cd.get("vocab50_macro_top1"))),
                cd.get("p_str_vocab50_word_macro_top1") if is_d else "",
                bold_val=is_d,
            )
            fv_m10_txt = _append_p_str(
                fmt_fn(cd.get("full_vocab_word_macro_top10", cd.get("full_vocab_macro_top10"))),
                cd.get("p_str_full_vocab_word_macro_top10") if is_d else "",
                bold_val=True,
            )
            fv_u10_txt = _append_p_str(
                fmt_fn(cd.get("full_vocab_micro_top10")),
                cd.get("p_str_full_vocab_micro_top10") if is_d else "",
            )
            mrr_raw = (
                f"{_num(cd.get('full_vocab_mrr', 0.0), 4):+.4f}"
                if is_d
                else f"{_num(cd.get('full_vocab_mrr', 0.0), 4):.4f}"
            )
            fv_mrr_txt = _append_p_str(
                mrr_raw,
                cd.get("p_str_full_vocab_mrr") if is_d else "",
            )
            md_lines.append(
                f"| **{cd.get('label', cat_k)}** | {cd.get('query_count', 0)} ({_num(cd.get('event_share_pct', 0.0), 1):.1f}%) | "
                f"{cd.get('unique_word_count', 0)} | {v50_m10_txt} | "
                f"{v50_u10_txt} | {v50_m1_txt} | "
                f"{fv_m10_txt} | {fv_u10_txt} | "
                f"{fv_mrr_txt} |"
            )
        md_lines.append("")

    if isinstance(nc, Mapping) and (nc.get("conditions") or nc.get("shifted_aggregate")):
        md_lines.extend(
            [
                "## 4. Negative Controls & Neural Specificity (`Clean vs. Shifted`)",
                "",
                "| Condition | Shift (s) | 50-Word Word-Macro Top-10 | Δ vs Clean | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | Full-Vocab Word-Macro Top-10 | Δ Full-Vocab |",
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
                f"- **Across-Subject 50-Word Word-Macro Top-10**: `{_fmt_pct(subj_sum.get('mean_macro_top10'))} ± {_fmt_pct(subj_sum.get('std_macro_top10'))}` "
                f"(Range: `{_fmt_pct(subj_sum.get('min_macro_top10'))} ~ {_fmt_pct(subj_sum.get('max_macro_top10'))}`) | "
                f"**Full-Vocab Event-Micro Top-10**: `{_fmt_pct(subj_sum.get('mean_full_vocab_micro_top10', subj_sum.get('mean_full_vocab_top10')))} ± {_fmt_pct(subj_sum.get('std_full_vocab_micro_top10', subj_sum.get('std_full_vocab_top10')))}`{imp_str}"
            )
            md_lines.append("")
        if has_k2_k0:
            md_lines.append(
                "| Subject | Events (50-Word) | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | 50-Word MRR | Full-Vocab Word-Macro Top-10 | Full-Vocab Event-Micro Top-10 | K0 50-Word Top-10 | K2 50-Word Top-10 | Δ (K2 - K0) |"
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
                "| Subject | Events (50-Word) | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | 50-Word MRR | 50-Word Median Rank | Full-Vocab Word-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab MRR |"
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
            "| Stage | Best Epoch | 50-Word Word-Macro Top-10 | Δ vs Stage 1 (95% CI / p) | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | Full-Vocab Word-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Δ Full-Vocab | Clean - Shifted |",
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


def _fmt_mean_sd_md_pct(
    mean_val: Any,
    std_val: Any,
    *,
    scale: float = 100.0,
    digits: int = 2,
    signed: bool = False,
    suffix: str = "%",
) -> str:
    """将均值与标准差格式化为 Markdown 表格字符串（如 '45.20 ± 0.51%' 或 '+12.34 ± 0.40 pp'）。"""
    if mean_val is None or mean_val == "":
        return "-"
    try:
        m = float(mean_val) * scale
        s = float(std_val or 0.0) * scale
        if signed:
            return f"{m:+.{digits}f} ± {s:.{digits}f}{suffix}"
        return f"{m:.{digits}f} ± {s:.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def _fmt_mean_sd_md_num(
    mean_val: Any,
    std_val: Any,
    *,
    digits: int = 4,
    signed: bool = False,
) -> str:
    """将无量纲或排名均值与标准差格式化为 Markdown 表格字符串（如 '0.2845 ± 0.0041' 或 '12.5 ± 0.5'）。"""
    if mean_val is None or mean_val == "":
        return "-"
    try:
        m = float(mean_val)
        s = float(std_val or 0.0)
        if signed:
            return f"{m:+.{digits}f} ± {s:.{digits}f}"
        return f"{m:.{digits}f} ± {s:.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _build_single_stage_multi_seed_markdown_lines(
    summary: Mapping[str, Any],
    *,
    section_prefix: str = "##",
) -> list[str]:
    """为单个阶段的多种子聚合结果构建完整的 GitHub-Flavored Markdown 报表行。"""
    sub_prefix = "#" + section_prefix
    stage = str(summary.get("stage", "word"))
    stage_disp = str(summary.get("stage_display_name", stage))
    seeds = list(summary.get("seeds") or [])
    n_seeds = int(summary.get("seed_count", len(seeds)))
    per_seed = list(summary.get("per_seed") or [])
    vs_agg = summary.get("vocabulary_sweep_aggregate") or {}
    n50_agg = vs_agg.get("N50") or {}
    n50_strict_agg = vs_agg.get("N50_strict") or n50_agg
    fv_agg = vs_agg.get("full_vocab") or {}
    ling_agg = summary.get("linguistic_breakdown_aggregate") or {}
    hz_agg = summary.get("revision_horizons_aggregate") or {}
    tr_agg = summary.get("revision_transitions_aggregate") or {}
    nc_agg = summary.get("negative_controls_aggregate") or {}
    cms_agg = summary.get("common_mode_subtraction_aggregate") or {}
    sub_agg = summary.get("subspace_debiasing_aggregate") or {}
    subj_agg = list(summary.get("per_subject_summary_aggregate") or [])

    md: list[str] = [
        f"{section_prefix} 1. Per-Seed & Multi-Seed Aggregate Summary — `{stage}` ({stage_disp})",
        "",
        f"{sub_prefix} 1A. 50-Word Frozen Bank (`N50`) (Per-Seed & Mean ± SD)",
        "",
        "| Seed / Summary | Best Epoch | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-5 | 50-Word Word-Macro Top-1 | 50-Word Event-Micro Top-1 | 50-Word MRR | 50-Word Median Rank | Strict-50 Word-Macro Top-10 | Control 50-Word Top-10 | Net Gap 50-Word Top-10 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in per_seed:
        if not isinstance(r, Mapping):
            continue
        sd = int(r.get("seed", 0))
        ep = r.get("epoch", "-")
        sh_50 = (
            _fmt_pct(r.get("shifted_vocab50_macro_top10"))
            if "shifted_vocab50_macro_top10" in r
            else "-"
        )
        gap_50 = (
            _fmt_signed_pct(r.get("clean_minus_shifted_vocab50_macro_top10"))
            if "clean_minus_shifted_vocab50_macro_top10" in r
            else "-"
        )
        md.append(
            f"| `seed-{sd:03d}` | `{ep}` | "
            f"{_fmt_pct(r.get('vocab50_macro_top10'))} | "
            f"{_fmt_pct(r.get('vocab50_micro_top10'))} | "
            f"{_fmt_pct(r.get('vocab50_macro_top5'))} | "
            f"{_fmt_pct(r.get('vocab50_macro_top1'))} | "
            f"{_fmt_pct(r.get('vocab50_micro_top1'))} | "
            f"{_num(r.get('vocab50_mrr', 0.0), 4):.4f} | "
            f"{_num(r.get('vocab50_median_rank', 25.0), 1):.1f} | "
            f"{_fmt_pct(r.get('strict_vocab50_macro_top10', r.get('vocab50_macro_top10')))} | "
            f"{sh_50} | {gap_50} |"
        )
    if n50_agg:
        sh_50_m = (
            _fmt_mean_sd_md_pct(
                nc_agg.get("mean_shifted_vocab50_macro_top10"),
                nc_agg.get("std_shifted_vocab50_macro_top10"),
            )
            if nc_agg.get("enabled")
            else "-"
        )
        gap_50_m = (
            _fmt_mean_sd_md_pct(
                nc_agg.get("mean_gap_vocab50_macro_top10"),
                nc_agg.get("std_gap_vocab50_macro_top10"),
                signed=True,
                suffix=" pp",
            )
            if nc_agg.get("enabled")
            else "-"
        )
        md.append(
            f"| **Mean ± SD (n={n_seeds})** | - | "
            f"**{_fmt_mean_sd_md_pct(n50_agg.get('mean_macro_top10'), n50_agg.get('std_macro_top10'))}** | "
            f"**{_fmt_mean_sd_md_pct(n50_agg.get('mean_micro_top10'), n50_agg.get('std_micro_top10'))}** | "
            f"**{_fmt_mean_sd_md_pct(n50_agg.get('mean_macro_top5'), n50_agg.get('std_macro_top5'))}** | "
            f"**{_fmt_mean_sd_md_pct(n50_agg.get('mean_macro_top1'), n50_agg.get('std_macro_top1'))}** | "
            f"**{_fmt_mean_sd_md_pct(n50_agg.get('mean_micro_top1'), n50_agg.get('std_micro_top1'))}** | "
            f"**{_fmt_mean_sd_md_num(n50_agg.get('mean_mrr'), n50_agg.get('std_mrr'), digits=4)}** | "
            f"**{_fmt_mean_sd_md_num(n50_agg.get('mean_median_rank'), n50_agg.get('std_median_rank'), digits=1)}** | "
            f"**{_fmt_mean_sd_md_pct(n50_strict_agg.get('mean_macro_top10'), n50_strict_agg.get('std_macro_top10'))}** | "
            f"**{sh_50_m}** | **{gap_50_m}** |"
        )

    md.extend(
        [
            "",
            f"{sub_prefix} 1B. Full Training Vocabulary (Per-Seed & Mean ± SD)",
            "",
            "| Seed / Summary | Best Epoch | Full-Vocab Word-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab Subject-Macro Top-10 | Full-Vocab Word-Macro Top-1 | Full-Vocab Event-Micro Top-1 | Full-Vocab MRR | Full-Vocab Median Rank | Control Full-Vocab Top-10 | Net Gap Full-Vocab Top-10 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in per_seed:
        if not isinstance(r, Mapping):
            continue
        sd = int(r.get("seed", 0))
        ep = r.get("epoch", "-")
        sh_fv = (
            _fmt_pct(r.get("shifted_full_vocab_macro_top10"))
            if "shifted_full_vocab_macro_top10" in r
            else "-"
        )
        gap_fv = (
            _fmt_signed_pct(r.get("clean_minus_shifted_full_vocab_macro_top10"))
            if "clean_minus_shifted_full_vocab_macro_top10" in r
            else "-"
        )
        md.append(
            f"| `seed-{sd:03d}` | `{ep}` | "
            f"{_fmt_pct(r.get('full_vocab_macro_top10'))} | "
            f"{_fmt_pct(r.get('full_vocab_micro_top10'))} | "
            f"{_fmt_pct(r.get('full_vocab_subject_macro_top10', r.get('full_vocab_micro_top10')))} | "
            f"{_fmt_pct(r.get('full_vocab_macro_top1'))} | "
            f"{_fmt_pct(r.get('full_vocab_micro_top1'))} | "
            f"{_num(r.get('full_vocab_mrr', 0.0), 4):.4f} | "
            f"{_num(r.get('full_vocab_median_rank', 0.0), 1):.1f} | "
            f"{sh_fv} | {gap_fv} |"
        )
    if fv_agg:
        sh_fv_m = (
            _fmt_mean_sd_md_pct(
                nc_agg.get("mean_shifted_full_vocab_macro_top10"),
                nc_agg.get("std_shifted_full_vocab_macro_top10"),
            )
            if nc_agg.get("enabled")
            else "-"
        )
        gap_fv_m = (
            _fmt_mean_sd_md_pct(
                nc_agg.get("mean_gap_full_vocab_macro_top10"),
                nc_agg.get("std_gap_full_vocab_macro_top10"),
                signed=True,
                suffix=" pp",
            )
            if nc_agg.get("enabled")
            else "-"
        )
        md.append(
            f"| **Mean ± SD (n={n_seeds})** | - | "
            f"**{_fmt_mean_sd_md_pct(fv_agg.get('mean_macro_top10'), fv_agg.get('std_macro_top10'))}** | "
            f"**{_fmt_mean_sd_md_pct(fv_agg.get('mean_micro_top10'), fv_agg.get('std_micro_top10'))}** | "
            f"**{_fmt_mean_sd_md_pct(fv_agg.get('mean_subject_macro_top10', fv_agg.get('mean_micro_top10')), fv_agg.get('std_subject_macro_top10', fv_agg.get('std_micro_top10')))}** | "
            f"**{_fmt_mean_sd_md_pct(fv_agg.get('mean_macro_top1'), fv_agg.get('std_macro_top1'))}** | "
            f"**{_fmt_mean_sd_md_pct(fv_agg.get('mean_micro_top1'), fv_agg.get('std_micro_top1'))}** | "
            f"**{_fmt_mean_sd_md_num(fv_agg.get('mean_mrr'), fv_agg.get('std_mrr'), digits=4)}** | "
            f"**{_fmt_mean_sd_md_num(fv_agg.get('mean_median_rank'), fv_agg.get('std_median_rank'), digits=1)}** | "
            f"**{sh_fv_m}** | **{gap_fv_m}** |"
        )

    if vs_agg or (isinstance(ling_agg, Mapping) and ling_agg):
        md.extend(
            [
                "",
                f"{section_prefix} 2. Multi-Scale Vocabulary Sweep & Linguistic Breakdown (Mean ± SD)",
            ]
        )
    if vs_agg:
        md.extend(
            [
                "",
                f"{sub_prefix} 2A. Multi-Scale Vocabulary Sweep Across Seeds",
                "",
                "| Scale | Candidates | Queries | In-Vocab Cov. | Word-Macro Top-10 | Event-Micro Top-10 | Subject-Macro Top-10 | Word-Macro Top-5 | Word-Macro Top-1 | Event-Micro Top-1 | MRR | Median Rank | Fold / Chance |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for s_key in ("N20", "N50", "N50_strict", "N100", "N150", "full_vocab"):
            sd = vs_agg.get(s_key)
            if not isinstance(sd, Mapping):
                continue
            fold_str = (
                f"{float(sd.get('mean_fold_over_chance_top10', 0.0)):.2f}x"
                if sd.get("mean_fold_over_chance_top10") is not None
                else "-"
            )
            md.append(
                f"| **{sd.get('label', s_key)}** (`{s_key}`) | "
                f"{int(sd.get('candidate_count', 0))} | "
                f"{int(sd.get('query_count', 0))} | "
                f"{float(sd.get('in_vocabulary_coverage_pct', 100.0)):.1f}% | "
                f"{_fmt_mean_sd_md_pct(sd.get('mean_macro_top10'), sd.get('std_macro_top10'))} | "
                f"{_fmt_mean_sd_md_pct(sd.get('mean_micro_top10'), sd.get('std_micro_top10'))} | "
                f"{_fmt_mean_sd_md_pct(sd.get('mean_subject_macro_top10', sd.get('mean_micro_top10')), sd.get('std_subject_macro_top10', sd.get('std_micro_top10')))} | "
                f"{_fmt_mean_sd_md_pct(sd.get('mean_macro_top5'), sd.get('std_macro_top5'))} | "
                f"{_fmt_mean_sd_md_pct(sd.get('mean_macro_top1'), sd.get('std_macro_top1'))} | "
                f"{_fmt_mean_sd_md_pct(sd.get('mean_micro_top1'), sd.get('std_micro_top1'))} | "
                f"{_fmt_mean_sd_md_num(sd.get('mean_mrr'), sd.get('std_mrr'), digits=4)} | "
                f"{_fmt_mean_sd_md_num(sd.get('mean_median_rank'), sd.get('std_median_rank'), digits=1)} | "
                f"{fold_str} |"
            )

    if isinstance(ling_agg, Mapping) and ling_agg:
        md.extend(
            [
                "",
                f"{sub_prefix} 2B. Linguistic Word-Class Breakdown Across Seeds (Mean ± SD)",
                "",
                "| Word Class | Queries | Unique Words | Event Share | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | Full-Vocab Word-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab Word-Macro Top-1 | Full-Vocab MRR | Full-Vocab Median Rank |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for l_key in ("content_words", "function_words", "content_minus_function"):
            lb = ling_agg.get(l_key)
            if not isinstance(lb, Mapping):
                continue
            is_delta = l_key == "content_minus_function"
            suf = " pp" if is_delta else "%"
            c_v50_m10 = _append_p_str(
                _fmt_mean_sd_md_pct(lb.get("mean_vocab50_word_macro_top10"), lb.get("std_vocab50_word_macro_top10"), signed=is_delta, suffix=suf),
                lb.get("p_str_vocab50_word_macro_top10") if is_delta else "",
                bold_val=is_delta,
            )
            c_v50_u10 = _append_p_str(
                _fmt_mean_sd_md_pct(lb.get("mean_vocab50_micro_top10"), lb.get("std_vocab50_micro_top10"), signed=is_delta, suffix=suf),
                lb.get("p_str_vocab50_micro_top10") if is_delta else "",
            )
            c_v50_m1 = _append_p_str(
                _fmt_mean_sd_md_pct(lb.get("mean_vocab50_word_macro_top1"), lb.get("std_vocab50_word_macro_top1"), signed=is_delta, suffix=suf),
                lb.get("p_str_vocab50_word_macro_top1") if is_delta else "",
                bold_val=is_delta,
            )
            c_fv_m10 = _append_p_str(
                _fmt_mean_sd_md_pct(lb.get("mean_full_vocab_word_macro_top10"), lb.get("std_full_vocab_word_macro_top10"), signed=is_delta, suffix=suf),
                lb.get("p_str_full_vocab_word_macro_top10") if is_delta else "",
                bold_val=is_delta,
            )
            c_fv_u10 = _append_p_str(
                _fmt_mean_sd_md_pct(lb.get("mean_full_vocab_micro_top10"), lb.get("std_full_vocab_micro_top10"), signed=is_delta, suffix=suf),
                lb.get("p_str_full_vocab_micro_top10") if is_delta else "",
            )
            c_fv_m1 = _append_p_str(
                _fmt_mean_sd_md_pct(lb.get("mean_full_vocab_word_macro_top1"), lb.get("std_full_vocab_word_macro_top1"), signed=is_delta, suffix=suf),
                lb.get("p_str_full_vocab_word_macro_top1") if is_delta else "",
                bold_val=is_delta,
            )
            c_fv_mrr = _append_p_str(
                _fmt_mean_sd_md_num(lb.get("mean_full_vocab_mrr"), lb.get("std_full_vocab_mrr"), digits=4, signed=is_delta),
                lb.get("p_str_full_vocab_mrr") if is_delta else "",
            )
            c_fv_med = _append_p_str(
                _fmt_mean_sd_md_num(lb.get("mean_full_vocab_median_rank"), lb.get("std_full_vocab_median_rank"), digits=1, signed=is_delta),
                lb.get("p_str_full_vocab_median_rank") if is_delta else "",
            )
            md.append(
                f"| **{lb.get('label', l_key)}** | "
                f"{int(lb.get('query_count', 0))} | "
                f"{int(lb.get('unique_word_count', 0))} | "
                f"{float(lb.get('event_share_pct', 0.0)):.2f}% | "
                f"{c_v50_m10} | "
                f"{c_v50_u10} | "
                f"{c_v50_m1} | "
                f"{c_fv_m10} | "
                f"{c_fv_u10} | "
                f"{c_fv_m1} | "
                f"{c_fv_mrr} | "
                f"{c_fv_med} |"
            )

    if (isinstance(hz_agg, Mapping) and hz_agg) or (
        isinstance(tr_agg, Mapping) and tr_agg
    ):
        md.extend(
            [
                "",
                f"{section_prefix} 3. Rolling Revision Horizons (`K0 / K1 / K2`) & Event Transitions (Mean ± SD)",
            ]
        )
    if isinstance(hz_agg, Mapping) and hz_agg:
        md.extend(
            [
                "",
                f"{sub_prefix} 3A. 50-Word & Full-Vocabulary Horizon Trajectory",
                "",
                "| Horizon | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | 50-Word Event-Micro Top-1 | 50-Word MRR | 50-Word Median Rank | Full-Vocab Word-Macro Top-10 | Full-Vocab Subject-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab Word-Macro Top-1 | Full-Vocab Event-Micro Top-1 | Full-Vocab MRR | Full-Vocab Median Rank |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for h_key, h_default_lbl, is_gain in (
            ("K0", "K0 (+0w Onset)", False),
            ("K1", "K1 (+1w Future)", False),
            ("K2", "K2 (+2w Future)", False),
            ("K1_minus_K0", "Revision Gain (K1 - K0)", True),
            ("K2_minus_K0", "Revision Gain (K2 - K0)", True),
            ("K2_minus_K1", "Step2 Gain (K2 - K1)", True),
        ):
            hb = hz_agg.get(h_key)
            if not isinstance(hb, Mapping):
                continue
            lbl = str(hb.get("label", h_default_lbl))
            suf = " pp" if is_gain else "%"
            md.append(
                f"| **{lbl}** | "
                f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get('mean_common_vocab50_macro_top10'), hb.get('std_common_vocab50_macro_top10'), signed=is_gain, suffix=suf), hb.get('p_str_common_vocab50_macro_top10') if is_gain else '', bold_val=is_gain)} | "
                f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get('mean_common_vocab50_micro_top10'), hb.get('std_common_vocab50_micro_top10'), signed=is_gain, suffix=suf), hb.get('p_str_common_vocab50_micro_top10') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get('mean_common_vocab50_macro_top1'), hb.get('std_common_vocab50_macro_top1'), signed=is_gain, suffix=suf), hb.get('p_str_common_vocab50_macro_top1') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get('mean_common_vocab50_micro_top1'), hb.get('std_common_vocab50_micro_top1'), signed=is_gain, suffix=suf), hb.get('p_str_common_vocab50_micro_top1') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_num(hb.get('mean_common_vocab50_mrr'), hb.get('std_common_vocab50_mrr'), digits=4, signed=is_gain), hb.get('p_str_common_vocab50_mrr') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_num(hb.get('mean_common_vocab50_median_rank'), hb.get('std_common_vocab50_median_rank'), digits=1, signed=is_gain), hb.get('p_str_common_vocab50_median_rank') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get('mean_common_full_vocab_macro_top10'), hb.get('std_common_full_vocab_macro_top10'), signed=is_gain, suffix=suf), hb.get('p_str_common_full_vocab_macro_top10') if is_gain else '', bold_val=is_gain)} | "
                f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get('mean_common_full_vocab_subject_macro_top10', hb.get('mean_common_full_vocab_micro_top10')), hb.get('std_common_full_vocab_subject_macro_top10', hb.get('std_common_full_vocab_micro_top10')), signed=is_gain, suffix=suf), hb.get('p_str_common_full_vocab_subject_macro_top10') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get('mean_common_full_vocab_micro_top10'), hb.get('std_common_full_vocab_micro_top10'), signed=is_gain, suffix=suf), hb.get('p_str_common_full_vocab_micro_top10') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get('mean_common_full_vocab_macro_top1'), hb.get('std_common_full_vocab_macro_top1'), signed=is_gain, suffix=suf), hb.get('p_str_common_full_vocab_macro_top1') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get('mean_common_full_vocab_micro_top1'), hb.get('std_common_full_vocab_micro_top1'), signed=is_gain, suffix=suf), hb.get('p_str_common_full_vocab_micro_top1') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_num(hb.get('mean_common_full_vocab_mrr'), hb.get('std_common_full_vocab_mrr'), digits=4, signed=is_gain), hb.get('p_str_common_full_vocab_mrr') if is_gain else '')} | "
                f"{_append_p_str(_fmt_mean_sd_md_num(hb.get('mean_common_full_vocab_median_rank'), hb.get('std_common_full_vocab_median_rank'), digits=1, signed=is_gain), hb.get('p_str_common_full_vocab_median_rank') if is_gain else '')} |"
            )

    if isinstance(tr_agg, Mapping) and tr_agg:
        md.extend(
            [
                "",
                f"{sub_prefix} 3B. Revision Hit/Miss Transition Matrix Across Seeds (Mean ± SD)",
                "",
                "| Transition | Metric Scope | Common Events | Beneficial Repairs (Miss->Hit) | Harmful Overwrites (Hit->Miss) | Net Hit Gain | Retained Hits | Persistent Misses | Repair / Harm Ratio | McNemar p-value |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for t_key, t_lbl in (
            ("K0_to_K1", "K0 -> K1"),
            ("K0_to_K2", "K0 -> K2"),
            ("K1_to_K2", "K1 -> K2"),
        ):
            tb = tr_agg.get(t_key)
            if not isinstance(tb, Mapping):
                continue
            for scope_k, scope_lbl in (
                ("vocab50_top10", "50-Word Top-10"),
                ("vocab50_top1", "50-Word Top-1"),
                ("full_vocab_top10", "Full-Vocab Top-10"),
                ("full_vocab_top1", "Full-Vocab Top-1"),
            ):
                sb = tb.get(scope_k)
                if not isinstance(sb, Mapping):
                    continue
                mc_str = str(sb.get("mcnemar_p_str", "-") or "-")
                md.append(
                    f"| **{t_lbl}** | {scope_lbl} | "
                    f"{int(sb.get('total_common_events', 0))} | "
                    f"{_fmt_mean_sd_md_num(sb.get('mean_beneficial_repairs'), sb.get('std_beneficial_repairs'), digits=1)} ({_fmt_mean_sd_md_pct(sb.get('mean_beneficial_repair_rate_pct'), sb.get('std_beneficial_repair_rate_pct'), scale=1.0)}) | "
                    f"{_fmt_mean_sd_md_num(sb.get('mean_harmful_overwrites'), sb.get('std_harmful_overwrites'), digits=1)} ({_fmt_mean_sd_md_pct(sb.get('mean_harmful_overwrite_rate_pct'), sb.get('std_harmful_overwrite_rate_pct'), scale=1.0)}) | "
                    f"{_fmt_mean_sd_md_num(sb.get('mean_net_gain_events'), sb.get('std_net_gain_events'), digits=1, signed=True)} ({_fmt_mean_sd_md_pct(sb.get('mean_net_gain_pct'), sb.get('std_net_gain_pct'), scale=1.0, signed=True, suffix=' pp')}) | "
                    f"{_fmt_mean_sd_md_num(sb.get('mean_retained_hits'), sb.get('std_retained_hits'), digits=1)} | "
                    f"{_fmt_mean_sd_md_num(sb.get('mean_persistent_misses'), sb.get('std_persistent_misses'), digits=1)} | "
                    f"{_fmt_mean_sd_md_num(sb.get('mean_repair_to_harm_ratio'), sb.get('std_repair_to_harm_ratio'), digits=2)}x | "
                    f"`{mc_str}` |"
                )

    has_any_control_section = (
        (isinstance(nc_agg, Mapping) and nc_agg.get("enabled"))
        or (isinstance(cms_agg, Mapping) and cms_agg.get("enabled"))
        or (isinstance(sub_agg, Mapping) and sub_agg.get("enabled"))
    )
    if has_any_control_section:
        md.extend(
            [
                "",
                f"{section_prefix} 4. Negative Control & Debiasing Multi-Seed Summary (Mean ± SD)",
            ]
        )

    if isinstance(nc_agg, Mapping) and nc_agg.get("enabled"):
        active_m_id = str(nc_agg.get("active_method_id", "raw"))
        active_m_lbl = str(
            nc_agg.get(
                "active_method_label",
                nc_agg.get("label", "Raw Temporal Shift (eta=0.0)"),
            )
        )
        sel_methods = list(
            nc_agg.get("selected_control_methods") or [active_m_id]
        )
        sel_metrics = list(
            nc_agg.get("selected_control_metrics")
            or [nc_agg.get("primary_control_metric", "vocab50_macro_top10")]
        )
        prim_m_key = str(
            nc_agg.get("primary_control_metric", "vocab50_macro_top10")
        )
        prim_m_lbl = str(
            nc_agg.get("primary_control_metric_label", "50-Word Word-Macro Top-10")
        )
        prim_sum = nc_agg.get("primary_metric_summary") or {}
        avail_methods = list(nc_agg.get("available_control_methods") or [])
        md.extend(
            [
                "",
                f"- **Selected Control / Debiasing Method(s)**: {', '.join(f'`{m}`' for m in sel_methods)} (Primary: `{active_m_id}` — **{active_m_lbl}**)",
                f"- **Selected Control Metric(s)**: {', '.join(f'`{m}`' for m in sel_metrics)} (Primary: `{prim_m_key}` — **{prim_m_lbl}**)",
            ]
        )
        if prim_sum:
            is_r = bool(prim_sum.get("is_ratio", True))
            if is_r:
                c_txt = _fmt_mean_sd_md_pct(
                    prim_sum.get("mean_clean"), prim_sum.get("std_clean")
                )
                s_txt = _fmt_mean_sd_md_pct(
                    prim_sum.get("mean_shifted"), prim_sum.get("std_shifted")
                )
                g_txt = _fmt_mean_sd_md_pct(
                    prim_sum.get("mean_gap"),
                    prim_sum.get("std_gap"),
                    signed=True,
                    suffix=" pp",
                )
            else:
                c_txt = _fmt_mean_sd_md_num(
                    prim_sum.get("mean_clean"),
                    prim_sum.get("std_clean"),
                    digits=2,
                )
                s_txt = _fmt_mean_sd_md_num(
                    prim_sum.get("mean_shifted"),
                    prim_sum.get("std_shifted"),
                    digits=2,
                )
                g_txt = _fmt_mean_sd_md_num(
                    prim_sum.get("mean_gap"),
                    prim_sum.get("std_gap"),
                    digits=2,
                    signed=True,
                )
            s_p_note = str(prim_sum.get("p_str_shifted_vs_chance") or "").strip()
            g_p_note = str(prim_sum.get("p_str_gap") or "").strip()
            s_disp = f"**{s_txt}** (vs. chance: {s_p_note})" if s_p_note else f"**{s_txt}**"
            g_disp = f"**{g_txt}** ({g_p_note})" if g_p_note else f"**{g_txt}**"
            md.append(
                f"  - **Primary Metric Result**: Clean = **{c_txt}** | Control = {s_disp} | Net Gap (Clean - Control) = {g_disp}"
            )
            ps_list = list(prim_sum.get("per_seed") or [])
            if ps_list:
                seed_parts = []
                for ps in ps_list:
                    if not isinstance(ps, Mapping):
                        continue
                    sd_i = int(ps.get("seed", 0))
                    if is_r:
                        seed_parts.append(
                            f"`seed-{sd_i:03d}`: Clean {_fmt_pct(ps.get('clean'))}, Ctrl {_fmt_pct(ps.get('shifted'))}, Gap {_fmt_signed_pct(ps.get('gap'))}"
                        )
                    else:
                        seed_parts.append(
                            f"`seed-{sd_i:03d}`: Clean {float(ps.get('clean', 0.0)):.2f}, Ctrl {float(ps.get('shifted', 0.0)):.2f}, Gap {float(ps.get('gap', 0.0)):+.2f}"
                        )
                if seed_parts:
                    md.append(f"  - **Per-Seed Breakdown**: {' ; '.join(seed_parts)}")
        if avail_methods:
            md.append(
                f"- **Available `--control-method` Selectors**: {', '.join(f'`{m}`' for m in avail_methods)}"
            )

        mm_comp = list(nc_agg.get("multi_method_metric_comparison") or [])
        if mm_comp:
            md.extend(
                [
                    "",
                    f"{sub_prefix} 4A-0. Multi-Method × Multi-Metric Comparison Table (`--control-method` × `--control-metric`)",
                    "",
                    "| Control Method | Metric | Clean (0s) Mean ± SD | Control Mean ± SD (vs. Chance) | Net Gap (Clean - Control) | Per-Seed Breakdown (Clean / Ctrl / Gap) |",
                    "| :--- | :--- | :---: | :---: | :---: | :--- |",
                ]
            )
            for row_mm in mm_comp:
                if not isinstance(row_mm, Mapping):
                    continue
                is_r_mm = bool(row_mm.get("is_ratio", True))
                if is_r_mm:
                    c_m = _fmt_mean_sd_md_pct(
                        row_mm.get("mean_clean"), row_mm.get("std_clean")
                    )
                    s_m = _fmt_mean_sd_md_pct(
                        row_mm.get("mean_shifted"), row_mm.get("std_shifted")
                    )
                    g_m = _fmt_mean_sd_md_pct(
                        row_mm.get("mean_gap"),
                        row_mm.get("std_gap"),
                        signed=True,
                        suffix=" pp",
                    )
                else:
                    c_m = _fmt_mean_sd_md_num(
                        row_mm.get("mean_clean"),
                        row_mm.get("std_clean"),
                        digits=2,
                    )
                    s_m = _fmt_mean_sd_md_num(
                        row_mm.get("mean_shifted"),
                        row_mm.get("std_shifted"),
                        digits=2,
                    )
                    g_m = _fmt_mean_sd_md_num(
                        row_mm.get("mean_gap"),
                        row_mm.get("std_gap"),
                        digits=2,
                        signed=True,
                    )
                s_m_disp = _append_p_str(
                    s_m, row_mm.get("p_str_shifted_vs_chance")
                )
                g_m_disp = _append_p_str(
                    g_m, row_mm.get("p_str_gap"), bold_val=True
                )
                ps_mm = []
                for ps_i in row_mm.get("per_seed") or ():
                    if not isinstance(ps_i, Mapping):
                        continue
                    sd_i = int(ps_i.get("seed", 0))
                    if is_r_mm:
                        ps_mm.append(
                            f"`s{sd_i}`: {_fmt_pct(ps_i.get('clean'))}/{_fmt_pct(ps_i.get('shifted'))}/{_fmt_signed_pct(ps_i.get('gap'))}"
                        )
                    else:
                        ps_mm.append(
                            f"`s{sd_i}`: {float(ps_i.get('clean', 0.0)):.2f}/{float(ps_i.get('shifted', 0.0)):.2f}/{float(ps_i.get('gap', 0.0)):+.2f}"
                        )
                md.append(
                    f"| `{row_mm.get('method_id', '-')}` ({row_mm.get('method_label', '-')}) | "
                    f"**{row_mm.get('metric_label', row_mm.get('metric_key', '-'))}** (`{row_mm.get('metric_key', '-')}`) | "
                    f"{c_m} | {s_m_disp} | {g_m_disp} | {' ; '.join(ps_mm) if ps_mm else '-'} |"
                )

        sel_methods_aggs = list(
            nc_agg.get("selected_methods_aggregates") or [nc_agg]
        )
        for idx_m, m_agg_item in enumerate(sel_methods_aggs):
            if not isinstance(m_agg_item, Mapping):
                continue
            cur_m_id = str(
                m_agg_item.get(
                    "active_method_id", m_agg_item.get("method_id", active_m_id)
                )
            )
            cur_m_lbl = str(
                m_agg_item.get(
                    "active_method_label",
                    m_agg_item.get("label", active_m_lbl),
                )
            )
            hdr_tag = (
                "4A"
                if len(sel_methods_aggs) == 1
                else f"4A.{idx_m + 1}"
            )
            md.extend(
                [
                    "",
                    f"{sub_prefix} {hdr_tag}. Overall Clean vs. Control vs. Net Gap Across All Retrieval Metrics (`{cur_m_id}` — {cur_m_lbl})",
                    "",
                    "| Condition | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | 50-Word Event-Micro Top-1 | 50-Word MRR | 50-Word Median Rank | Full-Vocab Word-Macro Top-10 | Full-Vocab Subject-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab Word-Macro Top-1 | Full-Vocab Event-Micro Top-1 | Full-Vocab MRR | Full-Vocab Median Rank |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                    f"| **Clean (0s)** | "
                    f"{_fmt_mean_sd_md_pct(m_agg_item.get('mean_clean_vocab50_macro_top10'), m_agg_item.get('std_clean_vocab50_macro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(m_agg_item.get('mean_clean_vocab50_micro_top10'), m_agg_item.get('std_clean_vocab50_micro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(m_agg_item.get('mean_clean_vocab50_macro_top1'), m_agg_item.get('std_clean_vocab50_macro_top1'))} | "
                    f"{_fmt_mean_sd_md_pct(m_agg_item.get('mean_clean_vocab50_micro_top1'), m_agg_item.get('std_clean_vocab50_micro_top1'))} | "
                    f"{_fmt_mean_sd_md_num(m_agg_item.get('mean_clean_vocab50_mrr'), m_agg_item.get('std_clean_vocab50_mrr'), digits=4)} | "
                    f"{_fmt_mean_sd_md_num(m_agg_item.get('mean_clean_vocab50_median_rank'), m_agg_item.get('std_clean_vocab50_median_rank'), digits=1)} | "
                    f"{_fmt_mean_sd_md_pct(m_agg_item.get('mean_clean_full_vocab_macro_top10'), m_agg_item.get('std_clean_full_vocab_macro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(m_agg_item.get('mean_clean_full_vocab_subject_macro_top10', m_agg_item.get('mean_clean_full_vocab_micro_top10')), m_agg_item.get('std_clean_full_vocab_subject_macro_top10', m_agg_item.get('std_clean_full_vocab_micro_top10')))} | "
                    f"{_fmt_mean_sd_md_pct(m_agg_item.get('mean_clean_full_vocab_micro_top10'), m_agg_item.get('std_clean_full_vocab_micro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(m_agg_item.get('mean_clean_full_vocab_macro_top1'), m_agg_item.get('std_clean_full_vocab_macro_top1'))} | "
                    f"{_fmt_mean_sd_md_pct(m_agg_item.get('mean_clean_full_vocab_micro_top1'), m_agg_item.get('std_clean_full_vocab_micro_top1'))} | "
                    f"{_fmt_mean_sd_md_num(m_agg_item.get('mean_clean_full_vocab_mrr'), m_agg_item.get('std_clean_full_vocab_mrr'), digits=4)} | "
                    f"{_fmt_mean_sd_md_num(m_agg_item.get('mean_clean_full_vocab_median_rank'), m_agg_item.get('std_clean_full_vocab_median_rank'), digits=1)} |",
                    f"| **Control Mean ({cur_m_id})** | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_shifted_vocab50_macro_top10'), m_agg_item.get('std_shifted_vocab50_macro_top10')), m_agg_item.get('p_str_shifted_vs_chance_vocab50_macro_top10'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_shifted_vocab50_micro_top10'), m_agg_item.get('std_shifted_vocab50_micro_top10')), m_agg_item.get('p_str_shifted_vs_chance_vocab50_micro_top10'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_shifted_vocab50_macro_top1'), m_agg_item.get('std_shifted_vocab50_macro_top1')), m_agg_item.get('p_str_shifted_vs_chance_vocab50_macro_top1'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_shifted_vocab50_micro_top1'), m_agg_item.get('std_shifted_vocab50_micro_top1')), m_agg_item.get('p_str_shifted_vs_chance_vocab50_micro_top1'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_num(m_agg_item.get('mean_shifted_vocab50_mrr'), m_agg_item.get('std_shifted_vocab50_mrr'), digits=4), m_agg_item.get('p_str_shifted_vs_chance_vocab50_mrr'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_num(m_agg_item.get('mean_shifted_vocab50_median_rank'), m_agg_item.get('std_shifted_vocab50_median_rank'), digits=1), m_agg_item.get('p_str_shifted_vs_chance_vocab50_median_rank'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_shifted_full_vocab_macro_top10'), m_agg_item.get('std_shifted_full_vocab_macro_top10')), m_agg_item.get('p_str_shifted_vs_chance_full_vocab_macro_top10'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_shifted_full_vocab_subject_macro_top10', m_agg_item.get('mean_shifted_full_vocab_micro_top10')), m_agg_item.get('std_shifted_full_vocab_subject_macro_top10', m_agg_item.get('std_shifted_full_vocab_micro_top10'))), m_agg_item.get('p_str_shifted_vs_chance_full_vocab_subject_macro_top10'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_shifted_full_vocab_micro_top10'), m_agg_item.get('std_shifted_full_vocab_micro_top10')), m_agg_item.get('p_str_shifted_vs_chance_full_vocab_micro_top10'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_shifted_full_vocab_macro_top1'), m_agg_item.get('std_shifted_full_vocab_macro_top1')), m_agg_item.get('p_str_shifted_vs_chance_full_vocab_macro_top1'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_shifted_full_vocab_micro_top1'), m_agg_item.get('std_shifted_full_vocab_micro_top1')), m_agg_item.get('p_str_shifted_vs_chance_full_vocab_micro_top1'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_num(m_agg_item.get('mean_shifted_full_vocab_mrr'), m_agg_item.get('std_shifted_full_vocab_mrr'), digits=4), m_agg_item.get('p_str_shifted_vs_chance_full_vocab_mrr'))} | "
                    f"{_append_p_str(_fmt_mean_sd_md_num(m_agg_item.get('mean_shifted_full_vocab_median_rank'), m_agg_item.get('std_shifted_full_vocab_median_rank'), digits=1), m_agg_item.get('p_str_shifted_vs_chance_full_vocab_median_rank'))} |",
                    f"| **Net Gap (Clean - Control)** | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_gap_vocab50_macro_top10'), m_agg_item.get('std_gap_vocab50_macro_top10'), signed=True, suffix=' pp'), m_agg_item.get('p_str_gap_vocab50_macro_top10'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_gap_vocab50_micro_top10'), m_agg_item.get('std_gap_vocab50_micro_top10'), signed=True, suffix=' pp'), m_agg_item.get('p_str_gap_vocab50_micro_top10'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_gap_vocab50_macro_top1'), m_agg_item.get('std_gap_vocab50_macro_top1'), signed=True, suffix=' pp'), m_agg_item.get('p_str_gap_vocab50_macro_top1'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_gap_vocab50_micro_top1'), m_agg_item.get('std_gap_vocab50_micro_top1'), signed=True, suffix=' pp'), m_agg_item.get('p_str_gap_vocab50_micro_top1'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_num(m_agg_item.get('mean_gap_vocab50_mrr'), m_agg_item.get('std_gap_vocab50_mrr'), digits=4, signed=True), m_agg_item.get('p_str_gap_vocab50_mrr'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_num(m_agg_item.get('mean_gap_vocab50_median_rank'), m_agg_item.get('std_gap_vocab50_median_rank'), digits=1, signed=True), m_agg_item.get('p_str_gap_vocab50_median_rank'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_gap_full_vocab_macro_top10'), m_agg_item.get('std_gap_full_vocab_macro_top10'), signed=True, suffix=' pp'), m_agg_item.get('p_str_gap_full_vocab_macro_top10'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_gap_full_vocab_subject_macro_top10', m_agg_item.get('mean_gap_full_vocab_micro_top10')), m_agg_item.get('std_gap_full_vocab_subject_macro_top10', m_agg_item.get('std_gap_full_vocab_micro_top10')), signed=True, suffix=' pp'), m_agg_item.get('p_str_gap_full_vocab_subject_macro_top10'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_gap_full_vocab_micro_top10'), m_agg_item.get('std_gap_full_vocab_micro_top10'), signed=True, suffix=' pp'), m_agg_item.get('p_str_gap_full_vocab_micro_top10'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_gap_full_vocab_macro_top1'), m_agg_item.get('std_gap_full_vocab_macro_top1'), signed=True, suffix=' pp'), m_agg_item.get('p_str_gap_full_vocab_macro_top1'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_pct(m_agg_item.get('mean_gap_full_vocab_micro_top1'), m_agg_item.get('std_gap_full_vocab_micro_top1'), signed=True, suffix=' pp'), m_agg_item.get('p_str_gap_full_vocab_micro_top1'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_num(m_agg_item.get('mean_gap_full_vocab_mrr'), m_agg_item.get('std_gap_full_vocab_mrr'), digits=4, signed=True), m_agg_item.get('p_str_gap_full_vocab_mrr'), bold_val=True)} | "
                    f"{_append_p_str(_fmt_mean_sd_md_num(m_agg_item.get('mean_gap_full_vocab_median_rank'), m_agg_item.get('std_gap_full_vocab_median_rank'), digits=1, signed=True), m_agg_item.get('p_str_gap_full_vocab_median_rank'), bold_val=True)} |",
                ]
            )
            for cond_row in m_agg_item.get("conditions_aggregate") or ():
                if not isinstance(cond_row, Mapping):
                    continue
                md.append(
                    f"| `{cond_row.get('label', cond_row.get('key', '-'))}` | "
                    f"{_fmt_mean_sd_md_pct(cond_row.get('mean_vocab50_macro_top10'), cond_row.get('std_vocab50_macro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(cond_row.get('mean_vocab50_micro_top10'), cond_row.get('std_vocab50_micro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(cond_row.get('mean_vocab50_macro_top1'), cond_row.get('std_vocab50_macro_top1'))} | "
                    f"{_fmt_mean_sd_md_pct(cond_row.get('mean_vocab50_micro_top1'), cond_row.get('std_vocab50_micro_top1'))} | "
                    f"{_fmt_mean_sd_md_num(cond_row.get('mean_vocab50_mrr'), cond_row.get('std_vocab50_mrr'), digits=4)} | "
                    f"{_fmt_mean_sd_md_num(cond_row.get('mean_vocab50_median_rank'), cond_row.get('std_vocab50_median_rank'), digits=1)} | "
                    f"{_fmt_mean_sd_md_pct(cond_row.get('mean_full_vocab_macro_top10'), cond_row.get('std_full_vocab_macro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(cond_row.get('mean_full_vocab_subject_macro_top10', cond_row.get('mean_full_vocab_micro_top10')), cond_row.get('std_full_vocab_subject_macro_top10', cond_row.get('std_full_vocab_micro_top10')))} | "
                    f"{_fmt_mean_sd_md_pct(cond_row.get('mean_full_vocab_micro_top10'), cond_row.get('std_full_vocab_micro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(cond_row.get('mean_full_vocab_macro_top1'), cond_row.get('std_full_vocab_macro_top1'))} | "
                    f"{_fmt_mean_sd_md_pct(cond_row.get('mean_full_vocab_micro_top1'), cond_row.get('std_full_vocab_micro_top1'))} | "
                    f"{_fmt_mean_sd_md_num(cond_row.get('mean_full_vocab_mrr'), cond_row.get('std_full_vocab_mrr'), digits=4)} | "
                    f"{_fmt_mean_sd_md_num(cond_row.get('mean_full_vocab_median_rank'), cond_row.get('std_full_vocab_median_rank'), digits=1)} |"
                )

            nc_hz = m_agg_item.get("horizons_aggregate") or {}
            if isinstance(nc_hz, Mapping) and nc_hz:
                md.extend(
                    [
                        "",
                        f"{sub_prefix} {hdr_tag}-H. Rolling Revision Horizons (`K0`, `K1`, `K2`) Under Negative Control (`{cur_m_id}`, Mean ± SD)",
                        "",
                        "| Horizon | Condition | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | 50-Word MRR | Full-Vocab Word-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab Word-Macro Top-1 | Full-Vocab MRR |",
                        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                    ]
                )
                for h_key in ("K0", "K1", "K2", "K2_minus_K0"):
                    hb = nc_hz.get(h_key)
                    if not isinstance(hb, Mapping):
                        continue
                    is_diff_h = h_key == "K2_minus_K0"
                    h_disp = "K2 - K0 Gain" if is_diff_h else h_key
                    for row_kind, prefix_k, is_signed in (
                        ("Clean (0s)", "clean", is_diff_h),
                        ("Control Mean", "shifted", is_diff_h),
                        ("Net Gap (Clean-Ctrl)", "gap", True),
                    ):
                        suf = " pp" if is_signed else "%"
                        p_key_prefix = ""
                        if is_diff_h:
                            p_key_prefix = f"p_str_{prefix_k}_"
                        elif prefix_k == "shifted":
                            p_key_prefix = "p_str_shifted_vs_chance_"
                        elif prefix_k == "gap":
                            p_key_prefix = "p_str_gap_"
                        bold_cell = is_diff_h and prefix_k in ("clean", "gap")
                        md.append(
                            f"| **{h_disp}** | {row_kind} | "
                            f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get(f'mean_{prefix_k}_common_vocab50_macro_top10'), hb.get(f'std_{prefix_k}_common_vocab50_macro_top10'), signed=is_signed, suffix=suf), hb.get(f'{p_key_prefix}common_vocab50_macro_top10') if p_key_prefix else '', bold_val=bold_cell)} | "
                            f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get(f'mean_{prefix_k}_common_vocab50_micro_top10'), hb.get(f'std_{prefix_k}_common_vocab50_micro_top10'), signed=is_signed, suffix=suf), hb.get(f'{p_key_prefix}common_vocab50_micro_top10') if p_key_prefix else '', bold_val=bold_cell)} | "
                            f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get(f'mean_{prefix_k}_common_vocab50_macro_top1'), hb.get(f'std_{prefix_k}_common_vocab50_macro_top1'), signed=is_signed, suffix=suf), hb.get(f'{p_key_prefix}common_vocab50_macro_top1') if p_key_prefix else '', bold_val=bold_cell)} | "
                            f"{_append_p_str(_fmt_mean_sd_md_num(hb.get(f'mean_{prefix_k}_common_vocab50_mrr'), hb.get(f'std_{prefix_k}_common_vocab50_mrr'), digits=4, signed=is_signed), hb.get(f'{p_key_prefix}common_vocab50_mrr') if p_key_prefix else '', bold_val=bold_cell)} | "
                            f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get(f'mean_{prefix_k}_common_full_vocab_macro_top10'), hb.get(f'std_{prefix_k}_common_full_vocab_macro_top10'), signed=is_signed, suffix=suf), hb.get(f'{p_key_prefix}common_full_vocab_macro_top10') if p_key_prefix else '', bold_val=bold_cell)} | "
                            f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get(f'mean_{prefix_k}_common_full_vocab_micro_top10'), hb.get(f'std_{prefix_k}_common_full_vocab_micro_top10'), signed=is_signed, suffix=suf), hb.get(f'{p_key_prefix}common_full_vocab_micro_top10') if p_key_prefix else '', bold_val=bold_cell)} | "
                            f"{_append_p_str(_fmt_mean_sd_md_pct(hb.get(f'mean_{prefix_k}_common_full_vocab_macro_top1'), hb.get(f'std_{prefix_k}_common_full_vocab_macro_top1'), signed=is_signed, suffix=suf), hb.get(f'{p_key_prefix}common_full_vocab_macro_top1') if p_key_prefix else '', bold_val=bold_cell)} | "
                            f"{_append_p_str(_fmt_mean_sd_md_num(hb.get(f'mean_{prefix_k}_common_full_vocab_mrr'), hb.get(f'std_{prefix_k}_common_full_vocab_mrr'), digits=4, signed=is_signed), hb.get(f'{p_key_prefix}common_full_vocab_mrr') if p_key_prefix else '', bold_val=bold_cell)} |"
                        )

    if isinstance(cms_agg, Mapping) and cms_agg.get("enabled"):
        cms_methods = cms_agg.get("methods") or {}
        if isinstance(cms_methods, Mapping) and cms_methods:
            md.extend(
                [
                    "",
                    f"{sub_prefix} 4B. Leave-One-Out Common-Mode Subtraction (CMS) Multi-Seed Sweep (Mean ± SD)",
                    "",
                    "| Method | `eta` | Clean 50-Word Word-Macro Top-10 | Shifted 50-Word Word-Macro Top-10 | Net Gap 50-Word Top-10 | Clean 50-Word Word-Macro Top-1 | Net Gap 50-Word Top-1 | Clean Full-Vocab Word-Macro Top-10 | Shifted Full-Vocab Word-Macro Top-10 | Net Gap Full-Vocab Top-10 | Clean Full-Vocab Word-Macro Top-1 | Net Gap Full-Vocab Top-1 | Clean Full-Vocab MRR |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for m_ent in cms_methods.values():
                if not isinstance(m_ent, Mapping):
                    continue
                md.append(
                    f"| **{m_ent.get('label', m_ent.get('method_id', '-'))}** (`{m_ent.get('method_id', '-')}`) | "
                    f"`{float(m_ent.get('eta') or 0.0):g}` | "
                    f"{_fmt_mean_sd_md_pct(m_ent.get('mean_clean_vocab50_macro_top10'), m_ent.get('std_clean_vocab50_macro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(m_ent.get('mean_shifted_vocab50_macro_top10'), m_ent.get('std_shifted_vocab50_macro_top10'))} | "
                    f"**{_fmt_mean_sd_md_pct(m_ent.get('mean_gap_vocab50_macro_top10'), m_ent.get('std_gap_vocab50_macro_top10'), signed=True, suffix=' pp')}** | "
                    f"{_fmt_mean_sd_md_pct(m_ent.get('mean_clean_vocab50_macro_top1'), m_ent.get('std_clean_vocab50_macro_top1'))} | "
                    f"{_fmt_mean_sd_md_pct(m_ent.get('mean_gap_vocab50_macro_top1'), m_ent.get('std_gap_vocab50_macro_top1'), signed=True, suffix=' pp')} | "
                    f"{_fmt_mean_sd_md_pct(m_ent.get('mean_clean_full_vocab_macro_top10'), m_ent.get('std_clean_full_vocab_macro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(m_ent.get('mean_shifted_full_vocab_macro_top10'), m_ent.get('std_shifted_full_vocab_macro_top10'))} | "
                    f"**{_fmt_mean_sd_md_pct(m_ent.get('mean_gap_full_vocab_macro_top10'), m_ent.get('std_gap_full_vocab_macro_top10'), signed=True, suffix=' pp')}** | "
                    f"{_fmt_mean_sd_md_pct(m_ent.get('mean_clean_full_vocab_macro_top1'), m_ent.get('std_clean_full_vocab_macro_top1'))} | "
                    f"{_fmt_mean_sd_md_pct(m_ent.get('mean_gap_full_vocab_macro_top1'), m_ent.get('std_gap_full_vocab_macro_top1'), signed=True, suffix=' pp')} | "
                    f"{_fmt_mean_sd_md_num(m_ent.get('mean_clean_full_vocab_mrr'), m_ent.get('std_clean_full_vocab_mrr'), digits=4)} |"
                )

            has_cms_hz = any(
                isinstance(m.get("horizons_aggregate"), Mapping)
                and m.get("horizons_aggregate")
                for m in cms_methods.values()
                if isinstance(m, Mapping)
            )
            if has_cms_hz:
                md.extend(
                    [
                        "",
                        f"{sub_prefix} 4B-H. Rolling Revision Horizons (`K0`, `K1`, `K2`) Across `CMS` `eta` Values",
                        "",
                        "| Method | `eta` | Horizon | Clean 50-Word Word-Macro Top-10 | Shifted 50-Word Word-Macro Top-10 | Gap 50-Word Top-10 | Clean 50-Word Word-Macro Top-1 | Gap 50-Word Top-1 | Clean Full-Vocab Word-Macro Top-10 | Shifted Full-Vocab Word-Macro Top-10 | Gap Full-Vocab Top-10 | Clean Full-Vocab Word-Macro Top-1 | Gap Full-Vocab Top-1 |",
                        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                    ]
                )
                for m_ent in cms_methods.values():
                    if not isinstance(m_ent, Mapping):
                        continue
                    hz_map = m_ent.get("horizons_aggregate") or {}
                    for hk in ("K0", "K1", "K2"):
                        hb = hz_map.get(hk)
                        if not isinstance(hb, Mapping):
                            continue
                        md.append(
                            f"| `{m_ent.get('method_id', '-')}` | `{float(m_ent.get('eta') or 0.0):g}` | **{hk}** | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_clean_common_vocab50_macro_top10'), hb.get('std_clean_common_vocab50_macro_top10'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_shifted_common_vocab50_macro_top10'), hb.get('std_shifted_common_vocab50_macro_top10'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_gap_common_vocab50_macro_top10'), hb.get('std_gap_common_vocab50_macro_top10'), signed=True, suffix=' pp')} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_clean_common_vocab50_macro_top1'), hb.get('std_clean_common_vocab50_macro_top1'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_gap_common_vocab50_macro_top1'), hb.get('std_gap_common_vocab50_macro_top1'), signed=True, suffix=' pp')} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_clean_common_full_vocab_macro_top10'), hb.get('std_clean_common_full_vocab_macro_top10'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_shifted_common_full_vocab_macro_top10'), hb.get('std_shifted_common_full_vocab_macro_top10'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_gap_common_full_vocab_macro_top10'), hb.get('std_gap_common_full_vocab_macro_top10'), signed=True, suffix=' pp')} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_clean_common_full_vocab_macro_top1'), hb.get('std_clean_common_full_vocab_macro_top1'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_gap_common_full_vocab_macro_top1'), hb.get('std_gap_common_full_vocab_macro_top1'), signed=True, suffix=' pp')} |"
                        )

    if isinstance(sub_agg, Mapping) and sub_agg.get("enabled"):
        sub_methods = sub_agg.get("methods") or {}
        if isinstance(sub_methods, Mapping) and sub_methods:
            md.extend(
                [
                    "",
                    f"{sub_prefix} 4C. Subspace Coherence LOO Debiasing Multi-Seed Sweep (Mean ± SD)",
                    "",
                    "| Selector ID | Rank `K` | `eta` | Gate | Clean 50-Word Word-Macro Top-10 | Shifted 50-Word Word-Macro Top-10 | Net Gap 50-Word Top-10 | Clean 50-Word Word-Macro Top-1 | Net Gap 50-Word Top-1 | Clean Full-Vocab Word-Macro Top-10 | Shifted Full-Vocab Word-Macro Top-10 | Net Gap Full-Vocab Top-10 | Clean Full-Vocab Word-Macro Top-1 | Net Gap Full-Vocab Top-1 | Clean Full-Vocab MRR |",
                    "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for s_ent in sub_methods.values():
                if not isinstance(s_ent, Mapping):
                    continue
                md.append(
                    f"| `{s_ent.get('method_id', '-')}` | "
                    f"`{s_ent.get('rank_k', '-')}` | "
                    f"`{float(s_ent.get('eta') or 0.0):g}` | "
                    f"`{'On' if s_ent.get('use_coherence_gate') else 'Off'}` | "
                    f"{_fmt_mean_sd_md_pct(s_ent.get('mean_clean_vocab50_macro_top10'), s_ent.get('std_clean_vocab50_macro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(s_ent.get('mean_shifted_vocab50_macro_top10'), s_ent.get('std_shifted_vocab50_macro_top10'))} | "
                    f"**{_fmt_mean_sd_md_pct(s_ent.get('mean_gap_vocab50_macro_top10'), s_ent.get('std_gap_vocab50_macro_top10'), signed=True, suffix=' pp')}** | "
                    f"{_fmt_mean_sd_md_pct(s_ent.get('mean_clean_vocab50_macro_top1'), s_ent.get('std_clean_vocab50_macro_top1'))} | "
                    f"{_fmt_mean_sd_md_pct(s_ent.get('mean_gap_vocab50_macro_top1'), s_ent.get('std_gap_vocab50_macro_top1'), signed=True, suffix=' pp')} | "
                    f"{_fmt_mean_sd_md_pct(s_ent.get('mean_clean_full_vocab_macro_top10'), s_ent.get('std_clean_full_vocab_macro_top10'))} | "
                    f"{_fmt_mean_sd_md_pct(s_ent.get('mean_shifted_full_vocab_macro_top10'), s_ent.get('std_shifted_full_vocab_macro_top10'))} | "
                    f"**{_fmt_mean_sd_md_pct(s_ent.get('mean_gap_full_vocab_macro_top10'), s_ent.get('std_gap_full_vocab_macro_top10'), signed=True, suffix=' pp')}** | "
                    f"{_fmt_mean_sd_md_pct(s_ent.get('mean_clean_full_vocab_macro_top1'), s_ent.get('std_clean_full_vocab_macro_top1'))} | "
                    f"{_fmt_mean_sd_md_pct(s_ent.get('mean_gap_full_vocab_macro_top1'), s_ent.get('std_gap_full_vocab_macro_top1'), signed=True, suffix=' pp')} | "
                    f"{_fmt_mean_sd_md_num(s_ent.get('mean_clean_full_vocab_mrr'), s_ent.get('std_clean_full_vocab_mrr'), digits=4)} |"
                )

            has_sub_hz = any(
                isinstance(m.get("horizons_aggregate"), Mapping)
                and m.get("horizons_aggregate")
                for m in sub_methods.values()
                if isinstance(m, Mapping)
            )
            if has_sub_hz:
                md.extend(
                    [
                        "",
                        f"{sub_prefix} 4C-H. Rolling Revision Horizons (`K0`, `K1`, `K2`) Across Subspace LOO Configurations",
                        "",
                        "| Selector ID | Horizon | Clean 50-Word Word-Macro Top-10 | Shifted 50-Word Word-Macro Top-10 | Gap 50-Word Top-10 | Clean 50-Word Word-Macro Top-1 | Gap 50-Word Top-1 | Clean Full-Vocab Word-Macro Top-10 | Shifted Full-Vocab Word-Macro Top-10 | Gap Full-Vocab Top-10 | Clean Full-Vocab Word-Macro Top-1 | Gap Full-Vocab Top-1 |",
                        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
                    ]
                )
                for s_ent in sub_methods.values():
                    if not isinstance(s_ent, Mapping):
                        continue
                    hz_map = s_ent.get("horizons_aggregate") or {}
                    for hk in ("K0", "K1", "K2"):
                        hb = hz_map.get(hk)
                        if not isinstance(hb, Mapping):
                            continue
                        md.append(
                            f"| `{s_ent.get('method_id', '-')}` | **{hk}** | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_clean_common_vocab50_macro_top10'), hb.get('std_clean_common_vocab50_macro_top10'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_shifted_common_vocab50_macro_top10'), hb.get('std_shifted_common_vocab50_macro_top10'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_gap_common_vocab50_macro_top10'), hb.get('std_gap_common_vocab50_macro_top10'), signed=True, suffix=' pp')} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_clean_common_vocab50_macro_top1'), hb.get('std_clean_common_vocab50_macro_top1'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_gap_common_vocab50_macro_top1'), hb.get('std_gap_common_vocab50_macro_top1'), signed=True, suffix=' pp')} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_clean_common_full_vocab_macro_top10'), hb.get('std_clean_common_full_vocab_macro_top10'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_shifted_common_full_vocab_macro_top10'), hb.get('std_shifted_common_full_vocab_macro_top10'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_gap_common_full_vocab_macro_top10'), hb.get('std_gap_common_full_vocab_macro_top10'), signed=True, suffix=' pp')} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_clean_common_full_vocab_macro_top1'), hb.get('std_clean_common_full_vocab_macro_top1'))} | "
                            f"{_fmt_mean_sd_md_pct(hb.get('mean_gap_common_full_vocab_macro_top1'), hb.get('std_gap_common_full_vocab_macro_top1'), signed=True, suffix=' pp')} |"
                        )

    if subj_agg:
        md.extend(
            [
                "",
                f"{section_prefix} 5. Per-Subject Multi-Seed Summary (Mean ± SD)",
                "",
                "| Subject | Events | 50-Word Events | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | 50-Word MRR | Full-Vocab Word-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab Word-Macro Top-1 | Full-Vocab MRR | K2-K0 50-Word Top-10 | K2-K0 Full-Vocab Top-10 |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for s_row in subj_agg:
            if not isinstance(s_row, Mapping):
                continue
            k2_50_str = (
                _fmt_mean_sd_md_pct(
                    s_row.get("mean_k2_minus_k0_vocab50_top10"),
                    s_row.get("std_k2_minus_k0_vocab50_top10"),
                    signed=True,
                    suffix=" pp",
                )
                if "mean_k2_minus_k0_vocab50_top10" in s_row
                else "-"
            )
            k2_fv_str = (
                _fmt_mean_sd_md_pct(
                    s_row.get("mean_k2_minus_k0_full_vocab_macro_top10"),
                    s_row.get("std_k2_minus_k0_full_vocab_macro_top10"),
                    signed=True,
                    suffix=" pp",
                )
                if "mean_k2_minus_k0_full_vocab_macro_top10" in s_row
                else "-"
            )
            md.append(
                f"| `{s_row.get('subject_id', '-')}` | "
                f"{int(s_row.get('evaluated_events', 0))} | "
                f"{int(s_row.get('vocab50_events', 0))} | "
                f"{_fmt_mean_sd_md_pct(s_row.get('mean_macro_top10'), s_row.get('std_macro_top10'))} | "
                f"{_fmt_mean_sd_md_pct(s_row.get('mean_micro_top10'), s_row.get('std_micro_top10'))} | "
                f"{_fmt_mean_sd_md_pct(s_row.get('mean_macro_top1'), s_row.get('std_macro_top1'))} | "
                f"{_fmt_mean_sd_md_num(s_row.get('mean_mrr'), s_row.get('std_mrr'), digits=4)} | "
                f"{_fmt_mean_sd_md_pct(s_row.get('mean_full_vocab_macro_top10'), s_row.get('std_full_vocab_macro_top10'))} | "
                f"{_fmt_mean_sd_md_pct(s_row.get('mean_full_vocab_micro_top10'), s_row.get('std_full_vocab_micro_top10'))} | "
                f"{_fmt_mean_sd_md_pct(s_row.get('mean_full_vocab_macro_top1'), s_row.get('std_full_vocab_macro_top1'))} | "
                f"{_fmt_mean_sd_md_num(s_row.get('mean_full_vocab_mrr'), s_row.get('std_full_vocab_mrr'), digits=4)} | "
                f"{k2_50_str} | {k2_fv_str} |"
            )

    return md


def export_multi_seed_tables_and_markdown(
    summary: Mapping[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """将多种子聚合评估结果导出为 tables/multi_seed_*.csv 与完整 multi_seed_evaluation_report_{split}.md。"""
    target_dir_raw = output_dir or summary.get("output_dir")
    if not target_dir_raw:
        return {}
    out_dir = Path(str(target_dir_raw))
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_split = str(summary.get("split", "")).strip().lower()
    if not raw_split:
        eval_json_str = str(summary.get("evaluation_json", "")).lower()
        if eval_json_str.endswith("_test.json"):
            raw_split = "test"
        elif eval_json_str.endswith("_val.json"):
            raw_split = "val"
    split = raw_split or "val"
    dataset = str(summary.get("dataset", "-"))
    seeds = list(summary.get("seeds") or [])
    exp_tag = str(summary.get("exp_tag", "") or "default")
    win_sec = float(summary.get("window_seconds", 1.0))
    subset = summary.get("subset") or {}
    subset_tag = (
        subset.get("subset_tag", "all")
        if isinstance(subset, Mapping)
        else str(subset)
    )
    exported: dict[str, str] = {}

    if "stages_aggregate" in summary and isinstance(
        summary.get("stages_aggregate"), Mapping
    ):
        st_rows: list[dict[str, Any]] = []
        md_lines = [
            f"# Multi-Seed Multi-Stage Evaluation Summary (`{dataset}` | `{split.upper()}` | Seeds: `{seeds}`)",
            "",
            f"- **Dataset**: `{dataset}` | **Split**: `{split.upper()}` | **Subset**: `{subset_tag}` | **Seeds**: `{seeds}` (`n={len(seeds)}`)",
            f"- **Window**: `{win_sec:.2f}s` | **Experiment Tag**: `{exp_tag}`",
            f"- **Requested Control Method**: `{summary.get('requested_control_method', 'auto')}` | **Primary Control Metric**: `{summary.get('requested_control_metric', 'vocab50_macro_top10')}`",
            "",
            "## Cross-Stage Multi-Seed Comparison (Mean ± SD)",
            "",
            "| Stage | 50-Word Word-Macro Top-10 | 50-Word Event-Micro Top-10 | 50-Word Word-Macro Top-1 | 50-Word MRR | Full-Vocab Word-Macro Top-10 | Full-Vocab Subject-Macro Top-10 | Full-Vocab Event-Micro Top-10 | Full-Vocab Word-Macro Top-1 | Full-Vocab MRR | Net Gap 50-Word Top-10 | Net Gap Full-Vocab Top-10 |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
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
                    "mean_vocab50_mrr": _num(n50.get("mean_mrr", 0.0), 4),
                    "std_vocab50_mrr": _num(n50.get("std_mrr", 0.0), 4),
                    "mean_full_vocab_macro_top10_pct": _pct(
                        fv.get("mean_macro_top10", 0.0)
                    ),
                    "std_full_vocab_macro_top10_pct": _pct(
                        fv.get("std_macro_top10", 0.0)
                    ),
                    "mean_full_vocab_subject_macro_top10_pct": _pct(
                        fv.get(
                            "mean_subject_macro_top10",
                            fv.get("mean_micro_top10", 0.0),
                        )
                    ),
                    "std_full_vocab_subject_macro_top10_pct": _pct(
                        fv.get(
                            "std_subject_macro_top10",
                            fv.get("std_micro_top10", 0.0),
                        )
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
                    "mean_full_vocab_mrr": _num(fv.get("mean_mrr", 0.0), 4),
                    "std_full_vocab_mrr": _num(fv.get("std_mrr", 0.0), 4),
                    "active_control_method": nc.get("active_method_id", "")
                    if nc.get("enabled")
                    else "",
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
                    "mean_clean_minus_shifted_full_vocab_top10_pp": _pct(
                        nc.get("mean_gap_full_vocab_macro_top10", 0.0)
                    )
                    if nc.get("enabled")
                    else "",
                    "std_clean_minus_shifted_full_vocab_top10_pp": _pct(
                        nc.get("std_gap_full_vocab_macro_top10", 0.0)
                    )
                    if nc.get("enabled")
                    else "",
                }
            )
            md_lines.append(
                f"| **{st_agg.get('stage_display_name', st_key)}** (`{st_key}`) | "
                f"**{_fmt_mean_sd_md_pct(n50.get('mean_macro_top10'), n50.get('std_macro_top10'))}** | "
                f"{_fmt_mean_sd_md_pct(n50.get('mean_micro_top10'), n50.get('std_micro_top10'))} | "
                f"{_fmt_mean_sd_md_pct(n50.get('mean_macro_top1'), n50.get('std_macro_top1'))} | "
                f"{_fmt_mean_sd_md_num(n50.get('mean_mrr'), n50.get('std_mrr'), digits=4)} | "
                f"**{_fmt_mean_sd_md_pct(fv.get('mean_macro_top10'), fv.get('std_macro_top10'))}** | "
                f"{_fmt_mean_sd_md_pct(fv.get('mean_subject_macro_top10', fv.get('mean_micro_top10')), fv.get('std_subject_macro_top10', fv.get('std_micro_top10')))} | "
                f"{_fmt_mean_sd_md_pct(fv.get('mean_micro_top10'), fv.get('std_micro_top10'))} | "
                f"{_fmt_mean_sd_md_pct(fv.get('mean_macro_top1'), fv.get('std_macro_top1'))} | "
                f"{_fmt_mean_sd_md_num(fv.get('mean_mrr'), fv.get('std_mrr'), digits=4)} | "
                f"{_fmt_mean_sd_md_pct(nc.get('mean_gap_vocab50_macro_top10'), nc.get('std_gap_vocab50_macro_top10'), signed=True, suffix=' pp') if nc.get('enabled') else '-'} | "
                f"{_fmt_mean_sd_md_pct(nc.get('mean_gap_full_vocab_macro_top10'), nc.get('std_gap_full_vocab_macro_top10'), signed=True, suffix=' pp') if nc.get('enabled') else '-'} |"
            )

        if st_rows:
            p_csv = _write_csv_table(
                tables_dir / f"multi_seed_stage_summary_{split}.csv",
                list(st_rows[0].keys()),
                st_rows,
            )
            exported["multi_seed_summary_csv"] = str(p_csv)

        for st_key, st_agg in summary["stages_aggregate"].items():
            if not isinstance(st_agg, Mapping):
                continue
            md_lines.extend(
                [
                    "",
                    "---",
                    "",
                    f"## Stage Detail: `{st_key}` — {st_agg.get('stage_display_name', st_key)}",
                    "",
                ]
            )
            md_lines.extend(
                _build_single_stage_multi_seed_markdown_lines(
                    st_agg, section_prefix="###"
                )
            )
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
                    "query_count": int(sd.get("query_count", 0)),
                    "in_vocabulary_coverage_pct": _num(
                        sd.get("in_vocabulary_coverage_pct", 100.0), 2
                    ),
                    "seed_count": int(sd.get("seed_count", len(seeds))),
                    "mean_macro_top10_pct": _pct(sd.get("mean_macro_top10", 0.0)),
                    "std_macro_top10_pct": _pct(sd.get("std_macro_top10", 0.0)),
                    "mean_micro_top10_pct": _pct(sd.get("mean_micro_top10", 0.0)),
                    "std_micro_top10_pct": _pct(sd.get("std_micro_top10", 0.0)),
                    "mean_subject_macro_top10_pct": _pct(
                        sd.get(
                            "mean_subject_macro_top10",
                            sd.get("mean_micro_top10", 0.0),
                        )
                    ),
                    "std_subject_macro_top10_pct": _pct(
                        sd.get(
                            "std_subject_macro_top10",
                            sd.get("std_micro_top10", 0.0),
                        )
                    ),
                    "mean_macro_top5_pct": _pct(sd.get("mean_macro_top5", 0.0)),
                    "std_macro_top5_pct": _pct(sd.get("std_macro_top5", 0.0)),
                    "mean_macro_top1_pct": _pct(sd.get("mean_macro_top1", 0.0)),
                    "std_macro_top1_pct": _pct(sd.get("std_macro_top1", 0.0)),
                    "mean_micro_top1_pct": _pct(sd.get("mean_micro_top1", 0.0)),
                    "std_micro_top1_pct": _pct(sd.get("std_micro_top1", 0.0)),
                    "mean_mrr": _num(sd.get("mean_mrr", 0.0), 4),
                    "std_mrr": _num(sd.get("std_mrr", 0.0), 4),
                    "mean_median_rank": _num(sd.get("mean_median_rank", 0.0), 2),
                    "std_median_rank": _num(sd.get("std_median_rank", 0.0), 2),
                    "mean_fold_over_chance_top10": _num(
                        sd.get("mean_fold_over_chance_top10", 0.0), 3
                    ),
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
            seed_rows: list[dict[str, Any]] = []
            for r in per_seed:
                if not isinstance(r, Mapping):
                    continue
                s_row: dict[str, Any] = {
                    "seed": r.get("seed", 0),
                    "best_epoch": r.get("epoch", 0),
                    "vocab50_macro_top10_pct": _pct(
                        r.get("vocab50_macro_top10", 0.0)
                    ),
                    "vocab50_micro_top10_pct": _pct(
                        r.get("vocab50_micro_top10", 0.0)
                    ),
                    "vocab50_macro_top5_pct": _pct(
                        r.get("vocab50_macro_top5", 0.0)
                    ),
                    "vocab50_macro_top1_pct": _pct(
                        r.get("vocab50_macro_top1", 0.0)
                    ),
                    "vocab50_micro_top1_pct": _pct(
                        r.get("vocab50_micro_top1", 0.0)
                    ),
                    "vocab50_mrr": _num(r.get("vocab50_mrr", 0.0), 4),
                    "vocab50_median_rank": _num(
                        r.get("vocab50_median_rank", 25.0), 2
                    ),
                    "strict_vocab50_macro_top10_pct": _pct(
                        r.get(
                            "strict_vocab50_macro_top10",
                            r.get("vocab50_macro_top10", 0.0),
                        )
                    ),
                    "full_vocab_macro_top10_pct": _pct(
                        r.get("full_vocab_macro_top10", 0.0)
                    ),
                    "full_vocab_subject_macro_top10_pct": _pct(
                        r.get(
                            "full_vocab_subject_macro_top10",
                            r.get("full_vocab_micro_top10", 0.0),
                        )
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
                    "full_vocab_median_rank": _num(
                        r.get("full_vocab_median_rank", 0.0), 2
                    ),
                }
                for hk_low in ("k0", "k1", "k2"):
                    if f"{hk_low}_vocab50_macro_top10" in r:
                        s_row[f"{hk_low}_vocab50_macro_top10_pct"] = _pct(
                            r.get(f"{hk_low}_vocab50_macro_top10", 0.0)
                        )
                        s_row[f"{hk_low}_vocab50_macro_top1_pct"] = _pct(
                            r.get(f"{hk_low}_vocab50_macro_top1", 0.0)
                        )
                        s_row[f"{hk_low}_full_vocab_macro_top10_pct"] = _pct(
                            r.get(f"{hk_low}_full_vocab_macro_top10", 0.0)
                        )
                        s_row[f"{hk_low}_full_vocab_macro_top1_pct"] = _pct(
                            r.get(f"{hk_low}_full_vocab_macro_top1", 0.0)
                        )
                if "shifted_vocab50_macro_top10" in r:
                    s_row["control_vocab50_macro_top10_pct"] = _pct(
                        r.get("shifted_vocab50_macro_top10", 0.0)
                    )
                    s_row["clean_minus_control_vocab50_macro_top10_pp"] = _pct(
                        r.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
                    )
                    s_row["control_full_vocab_macro_top10_pct"] = _pct(
                        r.get("shifted_full_vocab_macro_top10", 0.0)
                    )
                    s_row["clean_minus_control_full_vocab_macro_top10_pp"] = (
                        _pct(
                            r.get(
                                "clean_minus_shifted_full_vocab_macro_top10",
                                0.0,
                            )
                        )
                    )
                seed_rows.append(s_row)
            if seed_rows:
                ps_csv = _write_csv_table(
                    tables_dir / f"multi_seed_per_seed_{split}.csv",
                    list(seed_rows[0].keys()),
                    seed_rows,
                )
                exported["multi_seed_per_seed_csv"] = str(ps_csv)

        # Export multi-seed horizons CSV if present
        hz_agg = summary.get("revision_horizons_aggregate") or {}
        if isinstance(hz_agg, Mapping) and hz_agg:
            hz_csv_rows: list[dict[str, Any]] = []
            for h_key in (
                "K0",
                "K1",
                "K2",
                "K1_minus_K0",
                "K2_minus_K0",
                "K2_minus_K1",
            ):
                hb = hz_agg.get(h_key)
                if not isinstance(hb, Mapping):
                    continue
                hz_csv_rows.append(
                    {
                        "horizon": h_key,
                        "label": str(hb.get("label", h_key)),
                        "mean_vocab50_macro_top10_pct": _pct(
                            hb.get("mean_common_vocab50_macro_top10", 0.0)
                        ),
                        "std_vocab50_macro_top10_pct": _pct(
                            hb.get("std_common_vocab50_macro_top10", 0.0)
                        ),
                        "mean_vocab50_micro_top10_pct": _pct(
                            hb.get("mean_common_vocab50_micro_top10", 0.0)
                        ),
                        "std_vocab50_micro_top10_pct": _pct(
                            hb.get("std_common_vocab50_micro_top10", 0.0)
                        ),
                        "mean_vocab50_macro_top1_pct": _pct(
                            hb.get("mean_common_vocab50_macro_top1", 0.0)
                        ),
                        "std_vocab50_macro_top1_pct": _pct(
                            hb.get("std_common_vocab50_macro_top1", 0.0)
                        ),
                        "mean_vocab50_micro_top1_pct": _pct(
                            hb.get("mean_common_vocab50_micro_top1", 0.0)
                        ),
                        "std_vocab50_micro_top1_pct": _pct(
                            hb.get("std_common_vocab50_micro_top1", 0.0)
                        ),
                        "mean_vocab50_mrr": _num(
                            hb.get("mean_common_vocab50_mrr", 0.0), 4
                        ),
                        "std_vocab50_mrr": _num(
                            hb.get("std_common_vocab50_mrr", 0.0), 4
                        ),
                        "mean_full_vocab_macro_top10_pct": _pct(
                            hb.get("mean_common_full_vocab_macro_top10", 0.0)
                        ),
                        "std_full_vocab_macro_top10_pct": _pct(
                            hb.get("std_common_full_vocab_macro_top10", 0.0)
                        ),
                        "mean_full_vocab_subject_macro_top10_pct": _pct(
                            hb.get(
                                "mean_common_full_vocab_subject_macro_top10",
                                hb.get(
                                    "mean_common_full_vocab_micro_top10", 0.0
                                ),
                            )
                        ),
                        "std_full_vocab_subject_macro_top10_pct": _pct(
                            hb.get(
                                "std_common_full_vocab_subject_macro_top10",
                                hb.get(
                                    "std_common_full_vocab_micro_top10", 0.0
                                ),
                            )
                        ),
                        "mean_full_vocab_micro_top10_pct": _pct(
                            hb.get("mean_common_full_vocab_micro_top10", 0.0)
                        ),
                        "std_full_vocab_micro_top10_pct": _pct(
                            hb.get("std_common_full_vocab_micro_top10", 0.0)
                        ),
                        "mean_full_vocab_macro_top1_pct": _pct(
                            hb.get("mean_common_full_vocab_macro_top1", 0.0)
                        ),
                        "std_full_vocab_macro_top1_pct": _pct(
                            hb.get("std_common_full_vocab_macro_top1", 0.0)
                        ),
                        "mean_full_vocab_micro_top1_pct": _pct(
                            hb.get("mean_common_full_vocab_micro_top1", 0.0)
                        ),
                        "std_full_vocab_micro_top1_pct": _pct(
                            hb.get("std_common_full_vocab_micro_top1", 0.0)
                        ),
                        "mean_full_vocab_mrr": _num(
                            hb.get("mean_common_full_vocab_mrr", 0.0), 4
                        ),
                        "std_full_vocab_mrr": _num(
                            hb.get("std_common_full_vocab_mrr", 0.0), 4
                        ),
                        "p_value_vocab50_macro_top10": hb.get(
                            "p_value_common_vocab50_macro_top10", ""
                        ),
                        "significance_vocab50_macro_top10": hb.get(
                            "sig_common_vocab50_macro_top10", ""
                        ),
                        "p_value_full_vocab_macro_top10": hb.get(
                            "p_value_common_full_vocab_macro_top10", ""
                        ),
                        "significance_full_vocab_macro_top10": hb.get(
                            "sig_common_full_vocab_macro_top10", ""
                        ),
                    }
                )
            if hz_csv_rows:
                hz_csv = _write_csv_table(
                    tables_dir / f"multi_seed_horizons_{split}.csv",
                    list(hz_csv_rows[0].keys()),
                    hz_csv_rows,
                )
                exported["multi_seed_horizons_csv"] = str(hz_csv)

        # Export multi-seed revision transitions CSV if present
        tr_agg = summary.get("revision_transitions_aggregate") or {}
        if isinstance(tr_agg, Mapping) and tr_agg:
            tr_csv_rows: list[dict[str, Any]] = []
            for t_key, t_lbl in (
                ("K0_to_K1", "K0 -> K1"),
                ("K0_to_K2", "K0 -> K2"),
                ("K1_to_K2", "K1 -> K2"),
            ):
                tb = tr_agg.get(t_key)
                if not isinstance(tb, Mapping):
                    continue
                for scope_k, scope_lbl in (
                    ("vocab50_top10", "50-Word Top-10"),
                    ("vocab50_top1", "50-Word Top-1"),
                    ("full_vocab_top10", "Full-Vocab Top-10"),
                    ("full_vocab_top1", "Full-Vocab Top-1"),
                ):
                    sb = tb.get(scope_k)
                    if not isinstance(sb, Mapping):
                        continue
                    tr_csv_rows.append(
                        {
                            "transition": t_key,
                            "transition_label": t_lbl,
                            "metric_scope": scope_k,
                            "metric_scope_label": scope_lbl,
                            "total_common_events": int(
                                sb.get("total_common_events", 0)
                            ),
                            "mean_beneficial_repairs": _num(
                                sb.get("mean_beneficial_repairs", 0.0), 1
                            ),
                            "std_beneficial_repairs": _num(
                                sb.get("std_beneficial_repairs", 0.0), 1
                            ),
                            "mean_harmful_overwrites": _num(
                                sb.get("mean_harmful_overwrites", 0.0), 1
                            ),
                            "std_harmful_overwrites": _num(
                                sb.get("std_harmful_overwrites", 0.0), 1
                            ),
                            "mean_net_gain_events": _num(
                                sb.get("mean_net_gain_events", 0.0), 1
                            ),
                            "mean_net_gain_pct": _num(
                                sb.get("mean_net_gain_pct", 0.0), 2
                            ),
                            "mean_repair_to_harm_ratio": _num(
                                sb.get("mean_repair_to_harm_ratio", 0.0), 2
                            ),
                            "mcnemar_chi2": _num(
                                sb.get("mcnemar_chi2", 0.0), 2
                            ),
                            "mcnemar_p_value": sb.get("mcnemar_p_value", ""),
                            "mcnemar_significance": sb.get(
                                "mcnemar_significance", ""
                            ),
                            "mcnemar_p_str": sb.get("mcnemar_p_str", ""),
                        }
                    )
            if tr_csv_rows:
                tr_csv = _write_csv_table(
                    tables_dir / f"multi_seed_revision_transitions_{split}.csv",
                    list(tr_csv_rows[0].keys()),
                    tr_csv_rows,
                )
                exported["multi_seed_revision_transitions_csv"] = str(tr_csv)

        # Export multi-seed negative controls CSV if enabled
        nc_agg = summary.get("negative_controls_aggregate") or {}
        if isinstance(nc_agg, Mapping) and nc_agg.get("enabled"):
            nc_csv_rows: list[dict[str, Any]] = []
            sel_methods_aggs = list(
                nc_agg.get("selected_methods_aggregates") or [nc_agg]
            )
            for m_agg_item in sel_methods_aggs:
                if not isinstance(m_agg_item, Mapping):
                    continue
                m_id = str(
                    m_agg_item.get(
                        "active_method_id",
                        m_agg_item.get("method_id", "raw"),
                    )
                )
                for row_key, row_lbl, prefix_k in (
                    ("clean_0s", "Clean (0s)", "clean"),
                    ("control_mean", f"Control Mean ({m_id})", "shifted"),
                    ("net_gap", "Net Gap (Clean - Control)", "gap"),
                ):
                    p_50_10 = (
                        m_agg_item.get("p_shifted_vs_chance_vocab50_macro_top10", "")
                        if prefix_k == "shifted"
                        else (
                            m_agg_item.get("p_gap_vocab50_macro_top10", "")
                            if prefix_k == "gap"
                            else ""
                        )
                    )
                    sig_50_10 = (
                        m_agg_item.get("sig_shifted_vs_chance_vocab50_macro_top10", "")
                        if prefix_k == "shifted"
                        else (
                            m_agg_item.get("sig_gap_vocab50_macro_top10", "")
                            if prefix_k == "gap"
                            else ""
                        )
                    )
                    p_fv_10 = (
                        m_agg_item.get("p_shifted_vs_chance_full_vocab_macro_top10", "")
                        if prefix_k == "shifted"
                        else (
                            m_agg_item.get("p_gap_full_vocab_macro_top10", "")
                            if prefix_k == "gap"
                            else ""
                        )
                    )
                    sig_fv_10 = (
                        m_agg_item.get("sig_shifted_vs_chance_full_vocab_macro_top10", "")
                        if prefix_k == "shifted"
                        else (
                            m_agg_item.get("sig_gap_full_vocab_macro_top10", "")
                            if prefix_k == "gap"
                            else ""
                        )
                    )
                    nc_csv_rows.append(
                        {
                            "active_method_id": m_id,
                            "row_type": row_key,
                            "label": row_lbl,
                            "mean_vocab50_macro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_vocab50_macro_top10", 0.0
                                )
                            ),
                            "std_vocab50_macro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"std_{prefix_k}_vocab50_macro_top10", 0.0
                                )
                            ),
                            "p_value_vocab50_macro_top10": p_50_10,
                            "significance_vocab50_macro_top10": sig_50_10,
                            "mean_vocab50_micro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_vocab50_micro_top10", 0.0
                                )
                            ),
                            "std_vocab50_micro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"std_{prefix_k}_vocab50_micro_top10", 0.0
                                )
                            ),
                            "mean_vocab50_macro_top1_pct": _pct(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_vocab50_macro_top1", 0.0
                                )
                            ),
                            "std_vocab50_macro_top1_pct": _pct(
                                m_agg_item.get(
                                    f"std_{prefix_k}_vocab50_macro_top1", 0.0
                                )
                            ),
                            "mean_vocab50_micro_top1_pct": _pct(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_vocab50_micro_top1", 0.0
                                )
                            ),
                            "std_vocab50_micro_top1_pct": _pct(
                                m_agg_item.get(
                                    f"std_{prefix_k}_vocab50_micro_top1", 0.0
                                )
                            ),
                            "mean_vocab50_mrr": _num(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_vocab50_mrr", 0.0
                                ),
                                4,
                            ),
                            "std_vocab50_mrr": _num(
                                m_agg_item.get(
                                    f"std_{prefix_k}_vocab50_mrr", 0.0
                                ),
                                4,
                            ),
                            "mean_full_vocab_macro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_full_vocab_macro_top10",
                                    0.0,
                                )
                            ),
                            "std_full_vocab_macro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"std_{prefix_k}_full_vocab_macro_top10",
                                    0.0,
                                )
                            ),
                            "p_value_full_vocab_macro_top10": p_fv_10,
                            "significance_full_vocab_macro_top10": sig_fv_10,
                            "mean_full_vocab_subject_macro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_full_vocab_subject_macro_top10",
                                    m_agg_item.get(
                                        f"mean_{prefix_k}_full_vocab_micro_top10",
                                        0.0,
                                    ),
                                )
                            ),
                            "std_full_vocab_subject_macro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"std_{prefix_k}_full_vocab_subject_macro_top10",
                                    m_agg_item.get(
                                        f"std_{prefix_k}_full_vocab_micro_top10",
                                        0.0,
                                    ),
                                )
                            ),
                            "mean_full_vocab_micro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_full_vocab_micro_top10",
                                    0.0,
                                )
                            ),
                            "std_full_vocab_micro_top10_pct": _pct(
                                m_agg_item.get(
                                    f"std_{prefix_k}_full_vocab_micro_top10",
                                    0.0,
                                )
                            ),
                            "mean_full_vocab_macro_top1_pct": _pct(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_full_vocab_macro_top1",
                                    0.0,
                                )
                            ),
                            "std_full_vocab_macro_top1_pct": _pct(
                                m_agg_item.get(
                                    f"std_{prefix_k}_full_vocab_macro_top1",
                                    0.0,
                                )
                            ),
                            "mean_full_vocab_micro_top1_pct": _pct(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_full_vocab_micro_top1",
                                    0.0,
                                )
                            ),
                            "std_full_vocab_micro_top1_pct": _pct(
                                m_agg_item.get(
                                    f"std_{prefix_k}_full_vocab_micro_top1",
                                    0.0,
                                )
                            ),
                            "mean_full_vocab_mrr": _num(
                                m_agg_item.get(
                                    f"mean_{prefix_k}_full_vocab_mrr", 0.0
                                ),
                                4,
                            ),
                            "std_full_vocab_mrr": _num(
                                m_agg_item.get(
                                    f"std_{prefix_k}_full_vocab_mrr", 0.0
                                ),
                                4,
                            ),
                        }
                    )
            if nc_csv_rows:
                nc_csv = _write_csv_table(
                    tables_dir / f"multi_seed_negative_controls_{split}.csv",
                    list(nc_csv_rows[0].keys()),
                    nc_csv_rows,
                )
                exported["multi_seed_negative_controls_csv"] = str(nc_csv)

            mm_comp = list(nc_agg.get("multi_method_metric_comparison") or [])
            if mm_comp:
                mm_csv_rows: list[dict[str, Any]] = []
                for row_mm in mm_comp:
                    if not isinstance(row_mm, Mapping):
                        continue
                    is_r_mm = bool(row_mm.get("is_ratio", True))
                    mult = 100.0 if is_r_mm else 1.0
                    dig = 2 if is_r_mm else 4
                    mm_csv_rows.append(
                        {
                            "method_id": str(row_mm.get("method_id", "")),
                            "method_label": str(row_mm.get("method_label", "")),
                            "metric_key": str(row_mm.get("metric_key", "")),
                            "metric_label": str(row_mm.get("metric_label", "")),
                            "is_ratio": is_r_mm,
                            "mean_clean": _num(
                                float(row_mm.get("mean_clean", 0.0)) * mult, dig
                            ),
                            "std_clean": _num(
                                float(row_mm.get("std_clean", 0.0)) * mult, dig
                            ),
                            "mean_control": _num(
                                float(row_mm.get("mean_shifted", 0.0)) * mult,
                                dig,
                            ),
                            "std_control": _num(
                                float(row_mm.get("std_shifted", 0.0)) * mult,
                                dig,
                            ),
                            "p_control_vs_chance": row_mm.get(
                                "p_shifted_vs_chance", ""
                            ),
                            "sig_control_vs_chance": row_mm.get(
                                "sig_shifted_vs_chance", ""
                            ),
                            "mean_net_gap": _num(
                                float(row_mm.get("mean_gap", 0.0)) * mult, dig
                            ),
                            "std_net_gap": _num(
                                float(row_mm.get("std_gap", 0.0)) * mult, dig
                            ),
                            "p_net_gap": row_mm.get("p_gap", ""),
                            "sig_net_gap": row_mm.get("sig_gap", ""),
                        }
                    )
                if mm_csv_rows:
                    mm_csv = _write_csv_table(
                        tables_dir
                        / f"multi_seed_control_comparison_{split}.csv",
                        list(mm_csv_rows[0].keys()),
                        mm_csv_rows,
                    )
                    exported["multi_seed_control_comparison_csv"] = str(mm_csv)

        stage_disp = str(
            summary.get("stage_display_name", summary.get("stage", "word"))
        )
        md_lines = [
            f"# Multi-Seed Evaluation Summary (`{dataset}` | `{stage_disp}` | `{split.upper()}` | Seeds: `{seeds}`)",
            "",
            f"- **Dataset**: `{dataset}` | **Stage**: `{summary.get('stage', 'word')}` ({stage_disp}) | **Split**: `{split.upper()}`",
            f"- **Subset**: `{subset_tag}` | **Seeds**: `{seeds}` (`n={len(seeds)}`) | **Window**: `{win_sec:.2f}s` | **Tag**: `{exp_tag}`",
            f"- **Requested Control Method(s)**: `{summary.get('requested_control_method', 'auto')}` | **Requested Control Metric(s)**: `{summary.get('requested_control_metric', 'vocab50_macro_top10')}`",
            "",
        ]
        md_lines.extend(
            _build_single_stage_multi_seed_markdown_lines(
                summary, section_prefix="##"
            )
        )

    md_path = out_dir / f"multi_seed_evaluation_report_{split}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    exported["multi_seed_report_md"] = str(md_path)
    return exported


_S3_STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    "S0": "Word-Only",
    "S0_clean": "Word-Only",
    "Word-Only": "Word-Only",
    "K0": "Rolling-K0",
    "K0_clean": "Rolling-K0",
    "Rolling-K0": "Rolling-K0",
    "K1": "Rolling-K1",
    "K1_clean": "Rolling-K1",
    "Rolling-K1": "Rolling-K1",
    "S3_clean": "Rolling-K2",
    "S3": "Rolling-K2",
    "K2": "Rolling-K2",
    "Rolling-K2": "Rolling-K2",
    "F": "Full-Context",
    "F_clean": "Full-Context",
    "Full-Context": "Full-Context",
    "LM-Only": "LM-Only",
    "LM-Only_clean": "LM-Only",
    "K0_shifted_neg75s": "Shifted-K0 (-75s)",
    "Shifted-K0 (-75s)": "Shifted-K0 (-75s)",
    "Shifted-K0": "Shifted-K0 (-75s)",
    "S3_shifted_neg75s": "Shifted-K2 (-75s)",
    "Shifted-K2 (-75s)": "Shifted-K2 (-75s)",
    "Shifted-K2": "Shifted-K2 (-75s)",
    "Shifted (-75s)": "Shifted-K2 (-75s)",
    "S3_clean_c_gated": "Rolling-K2 (C-Gated)",
    "S3_shifted_neg75s_c_gated": "Shifted-K2 (-75s, C-Gated)",
}

_S3_STRATEGY_STATE_KEYS: dict[str, str] = {
    "S0": "S0",
    "S0_clean": "S0",
    "Word-Only": "S0",
    "K0": "K0",
    "K0_clean": "K0",
    "Rolling-K0": "K0",
    "K1": "K1",
    "K1_clean": "K1",
    "Rolling-K1": "K1",
    "S3_clean": "K2",
    "S3": "K2",
    "K2": "K2",
    "Rolling-K2": "K2",
    "F": "F",
    "F_clean": "F",
    "Full-Context": "F",
    "LM-Only": "LM-Only",
    "K0_shifted_neg75s": "K0",
    "Shifted-K0 (-75s)": "K0",
    "S3_shifted_neg75s": "K2",
    "Shifted-K2 (-75s)": "K2",
}


def _s3_strategy_name(raw_name: str) -> str:
    return _S3_STRATEGY_DISPLAY_NAMES.get(str(raw_name), str(raw_name))


def _s3_format_lambda_temp(
    cond_key: str,
    state_hp: Mapping[str, Any] | None,
) -> str:
    st_key = _S3_STRATEGY_STATE_KEYS.get(
        str(cond_key), _s3_strategy_name(cond_key)
    )
    if st_key == "LM-Only":
        return "1.00 (-)"
    if not isinstance(state_hp, Mapping) or not state_hp:
        return "-"
    states_map = (
        state_hp.get("states")
        if isinstance(state_hp.get("states"), Mapping)
        else state_hp
    )
    entry = states_map.get(st_key)
    if not isinstance(entry, Mapping):
        if st_key == "S0":
            entry = states_map.get("K0")
        elif st_key == "F":
            entry = states_map.get("K2")
    if not isinstance(entry, Mapping):
        return "-"
    lam = entry.get("language_weight")
    temp = entry.get("temperature")
    if lam is None and temp is None:
        return "-"
    lam_str = f"{float(lam):.2f}" if isinstance(lam, (int, float)) else "-"
    temp_str = f"{float(temp):.2f}" if isinstance(temp, (int, float)) else "-"
    return f"{lam_str} ({temp_str})"


def _s3_resolve_mean_wait_s(cond_key: str, cm: Mapping[str, Any]) -> float:
    disp = _s3_strategy_name(cond_key)
    if disp == "LM-Only":
        return 0.0
    if disp in ("Word-Only", "Rolling-K0", "Shifted-K0 (-75s)"):
        return 1.0
    if "mean_wait_seconds" in cm and isinstance(
        cm.get("mean_wait_seconds"), (int, float)
    ):
        val = float(cm["mean_wait_seconds"])
        if val > 0.0:
            return round(val, 2)
    tot_w = max(1, int(cm.get("total_words", 0)))
    seg_c = max(1, int(cm.get("segment_count", 1)))
    avg_l = max(1.0, tot_w / seg_c)
    if disp == "Rolling-K1":
        return round(1.0 + max(0.0, avg_l - 1.0) / avg_l, 2)
    if disp == "Full-Context":
        return round(1.0 + 0.5 * max(0.0, avg_l - 1.0), 2)
    return round(
        1.0
        + (2.0 * max(0.0, avg_l - 2.0) + (1.0 if avg_l >= 2.0 else 0.0))
        / avg_l,
        2,
    )


def _s3_lookup_cond_metrics(
    conds_map: Mapping[str, Any],
    canonical_disp: str,
) -> Mapping[str, Any] | None:
    alias_groups: dict[str, tuple[str, ...]] = {
        "Word-Only": ("S0", "S0_clean", "Word-Only"),
        "Rolling-K0": ("K0", "K0_clean", "Rolling-K0"),
        "Rolling-K1": ("K1", "K1_clean", "Rolling-K1"),
        "Rolling-K2": ("S3_clean", "S3", "K2", "Rolling-K2"),
        "Full-Context": ("F", "F_clean", "Full-Context"),
        "LM-Only": ("LM-Only", "LM-Only_clean"),
        "Shifted-K0 (-75s)": (
            "K0_shifted_neg75s",
            "Shifted-K0 (-75s)",
            "Shifted-K0",
        ),
        "Shifted-K2 (-75s)": (
            "S3_shifted_neg75s",
            "Shifted-K2 (-75s)",
            "Shifted-K2",
        ),
    }
    for k in alias_groups.get(canonical_disp, (canonical_disp,)):
        if k in conds_map and isinstance(conds_map[k], Mapping):
            return conds_map[k]
    return None


def export_sequence_decoding_tables_and_markdown(
    report: Mapping[str, Any],
    output_dir: str | Path | None = None,
    *,
    decoded_bundles: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, str]:
    """将 S3 序列解码评估结果与逐句预测导出为 6 大标准化模块（GFM Markdown + 扁平 CSV 表 + 双语逐句对照视图）。"""
    import json

    target_dir_raw = (
        output_dir or report.get("evaluation_dir") or report.get("output_dir")
    )
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
    sig_map = (
        report.get("paired_statistical_significance")
        or report.get("statistical_significance")
        or {}
    )
    per_subj = report.get("per_subject") or []
    subj_sum = report.get("per_subject_summary") or {}
    vocab_scaling = dict(report.get("vocabulary_scaling") or {})
    multi_subj = dict(report.get("multi_subject_scaling") or {})
    pareto_map = report.get("confidence_gate_pareto") or {}
    state_hp = report.get("state_hyperparameters")
    if not isinstance(state_hp, Mapping) or not state_hp:
        for cand_p in (
            out_dir / "state_hyperparameter_calibration.json",
            out_dir.parent / "state_hyperparameter_calibration.json",
            out_dir / f"sequence_decoding_{split}.json",
            out_dir.parent / f"sequence_decoding_{split}.json",
        ):
            if cand_p.is_file():
                try:
                    c_obj = json.loads(cand_p.read_text(encoding="utf-8"))
                    if isinstance(c_obj.get("states"), Mapping):
                        state_hp = c_obj["states"]
                        break
                    if isinstance(c_obj.get("state_hyperparameters"), Mapping):
                        state_hp = c_obj["state_hyperparameters"]
                        break
                except Exception:
                    pass

    is_multi_seed = bool(report.get("is_multi_seed_summary"))
    seeds_list = report.get("seeds")
    seed_val = report.get("seed")

    def _fmt_pct_ms(
        m_dict: Mapping[str, Any],
        key: str,
        *,
        bold: bool = False,
        fallback: float = 0.0,
    ) -> str:
        val = _num(m_dict.get(key, fallback), 2)
        std_val = _num(m_dict.get(f"{key}_std", 0.0), 2)
        if is_multi_seed and isinstance(seeds_list, Sequence) and len(seeds_list) > 1 and std_val > 0.0:
            s = f"{val:.2f}±{std_val:.2f}%"
        else:
            s = f"{val:.2f}%"
        return f"**{s}**" if bold else s

    # 确保全词表规模（如 V=2020 或 V=2180）也注册在 core_benchmarks 与 vocab_scaling 中
    core_benchmarks = dict(report.get("core_vocabulary_benchmarks") or {})
    full_vocab_subsets_blk = report.get("full_vocab_core_subsets") or {}
    full_v_sz = int(
        full_vocab_subsets_blk.get("vocabulary_size")
        or max(
            (
                int(str(k).lstrip("vV=_") or 0)
                for k in list(vocab_scaling.keys()) + list(core_benchmarks.keys())
                if str(k).lstrip("vV=_").isdigit()
            ),
            default=2020 if dataset == "pallier2025" else 2180,
        )
    )
    if full_v_sz <= 250:
        full_v_sz = 2020 if dataset == "pallier2025" else 2180
    full_v_key = f"V={full_v_sz}"

    if full_v_key not in core_benchmarks:
        fv_subs = (
            dict(full_vocab_subsets_blk.get("subsets") or {})
            if isinstance(full_vocab_subsets_blk, Mapping)
            else {}
        )
        if not fv_subs and conditions:
            s3_c = conditions.get("S3_clean") or conditions.get("S3") or {}
            fv_subs = {
                "all_segments": {
                    "description": "All evaluation segments",
                    "segment_count": int(s3_c.get("segment_count", 0)),
                    "conditions": conditions,
                }
            }
        if fv_subs:
            s3_c = conditions.get("S3_clean") or conditions.get("S3") or {}
            tot_w = max(1, int(s3_c.get("total_words", 0)))
            iv_w = int(s3_c.get("in_vocabulary_word_count", tot_w))
            core_benchmarks[full_v_key] = {
                "vocabulary_size": full_v_sz,
                "is_full_vocabulary": True,
                "test_token_coverage_pct": round(iv_w / tot_w * 100.0, 2),
                "subsets": fv_subs,
            }

    if full_v_key not in vocab_scaling and full_v_key in core_benchmarks:
        cb_fv = core_benchmarks[full_v_key]
        fv_subs = cb_fv.get("subsets") or {}
        fv_all = fv_subs.get("all_segments") or {}
        fv_c80 = fv_subs.get("core_subset_80pct") or {}
        fv_c100 = fv_subs.get("core_subset_100pct") or {}
        s3_c = (fv_all.get("conditions") or {}).get("S3_clean") or conditions.get("S3_clean") or {}
        vocab_scaling[full_v_key] = {
            "vocabulary_size": full_v_sz,
            "is_full_vocabulary": True,
            "test_token_coverage_pct": _num(cb_fv.get("test_token_coverage_pct", 100.0), 2),
            "total_segment_count": int(fv_all.get("segment_count", s3_c.get("segment_count", 0))),
            "core_100pct_segment_count": int(fv_c100.get("segment_count", 0)),
            "core_80pct_segment_count": int(fv_c80.get("segment_count", 0)),
            "bleu1_pct": _num(s3_c.get("bleu1_pct", 0.0), 2),
            "corpus_bleu1_pct": _num(s3_c.get("corpus_bleu1_pct", 0.0), 2),
            "beam_oracle_bleu1_pct": _num(s3_c.get("beam_oracle_bleu1_pct", 0.0), 2),
            "word_acc_pct": _num(s3_c.get("word_acc_pct", 0.0), 2),
            "in_vocabulary_word_acc_pct": _num(s3_c.get("in_vocabulary_word_acc_pct", 0.0), 2),
            "word_acc_top5_pct": _num(s3_c.get("word_acc_top5_pct", 0.0), 2),
            "content_word_acc_pct": _num(s3_c.get("content_word_acc_pct", 0.0), 2),
            "function_word_acc_pct": _num(s3_c.get("function_word_acc_pct", 0.0), 2),
            "wer_pct": _num(s3_c.get("wer_pct", 0.0), 2),
            "cer_pct": _num(s3_c.get("cer_pct", 0.0), 2),
            "bertscore_f1_pct": _num(s3_c.get("bertscore_f1_pct", 0.0), 2),
            "conditions": fv_all.get("conditions") or conditions,
            "subsets": fv_subs,
            "core_subset_80pct": fv_c80,
            "core_subset_100pct": fv_c100,
        }

    exported: dict[str, str] = {}
    tables_map: dict[str, str] = {}

    # 1. 导出条件对比表 s3_conditions_summary_{split}.csv（统一使用标准策略名）
    cond_rows: list[dict[str, Any]] = []
    cgated_rows: list[dict[str, Any]] = []
    for cond_name, cm in conditions.items():
        if not isinstance(cm, Mapping) or str(cond_name) == "F_local":
            continue
        disp_strategy = _s3_strategy_name(cond_name)
        lam_temp = _s3_format_lambda_temp(cond_name, state_hp)
        wait_s = _s3_resolve_mean_wait_s(cond_name, cm)
        cf_delta = _num(
            cm.get(
                "content_minus_function_acc_pp",
                float(cm.get("content_word_acc_pct", 0.0))
                - float(cm.get("function_word_acc_pct", 0.0)),
            ),
            2,
        )
        cf_p = _num(cm.get("content_vs_function_p_value", 1.0), 4)
        row_item = {
            "strategy": disp_strategy,
            "condition": str(cond_name),
            "lambda_r_T_r": lam_temp,
            "segment_count": int(cm.get("segment_count", 0)),
            "total_words": int(cm.get("total_words", 0)),
            "corpus_bleu1_pct": _num(cm.get("corpus_bleu1_pct", 0.0), 2),
            "bleu1_subject_macro_pct": _num(cm.get("bleu1_pct", 0.0), 2),
            "beam_oracle_bleu1_pct": _num(
                cm.get("beam_oracle_bleu1_pct", 0.0), 2
            ),
            "wer_pct": _num(cm.get("wer_pct", 0.0), 2),
            "cer_pct": _num(cm.get("cer_pct", 0.0), 2),
            "bertscore_f1_pct": _num(cm.get("bertscore_f1_pct", 0.0), 2),
            "mean_wait_seconds": wait_s,
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
            "content_minus_function_acc_pp": cf_delta,
            "content_vs_function_p_value": cf_p,
            "screen_rewrite_rate_pct": _num(
                cm.get("screen_rewrite_rate_pct", 0.0), 2
            )
            if "screen_rewrite_rate_pct" in cm
            else "",
            "rewrite_filtered_pct": _num(
                cm.get("rewrite_filtered_pct", 0.0), 2
            )
            if "rewrite_filtered_pct" in cm
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
        tables_map["s3_conditions_summary"] = str(c_csv)

    # 1B. 导出核心词表主表与词位拆解表 s3_core_vocabulary_benchmarks_{split}.csv
    module1_strategies = (
        "Word-Only",
        "Rolling-K0",
        "Rolling-K1",
        "Rolling-K2",
        "Full-Context",
        "Shifted-K0 (-75s)",
        "Shifted-K2 (-75s)",
    )
    module2_strategies = (
        "LM-Only",
        "Word-Only",
        "Rolling-K0",
        "Rolling-K1",
        "Rolling-K2",
        "Full-Context",
        "Shifted-K0 (-75s)",
        "Shifted-K2 (-75s)",
    )
    cb_rows: list[dict[str, Any]] = []
    if isinstance(core_benchmarks, Mapping) and core_benchmarks:
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
            v_sz = int(
                cb_entry.get("vocabulary_size", str(v_key).lstrip("vV=_") or 0)
            )
            tok_cov = _num(cb_entry.get("test_token_coverage_pct", 0.0), 2)
            subsets_map = cb_entry.get("subsets") or {}
            for sub_key, sub_disp in (
                ("core_subset_100pct", "0% OOV"),
                ("core_subset_80pct", ">=80% In-Vocab"),
                ("all_segments", "All Segs"),
            ):
                sub_blk = subsets_map.get(sub_key)
                if not isinstance(sub_blk, Mapping):
                    continue
                seg_cnt = int(sub_blk.get("segment_count", 0))
                if seg_cnt <= 0:
                    continue
                sub_conds = sub_blk.get("conditions") or {}
                for strat_disp in module2_strategies:
                    cm = _s3_lookup_cond_metrics(sub_conds, strat_disp)
                    if cm is None:
                        continue
                    lam_temp = _s3_format_lambda_temp(strat_disp, state_hp)
                    wait_s = _s3_resolve_mean_wait_s(strat_disp, cm)
                    cf_delta = _num(
                        cm.get(
                            "content_minus_function_acc_pp",
                            float(cm.get("content_word_acc_pct", 0.0))
                            - float(cm.get("function_word_acc_pct", 0.0)),
                        ),
                        2,
                    )
                    cf_p = _num(cm.get("content_vs_function_p_value", 1.0), 4)
                    cb_rows.append(
                        {
                            "scale_key": str(v_key),
                            "vocabulary_size": v_sz,
                            "test_token_coverage_pct": tok_cov,
                            "subset_key": sub_key,
                            "subset": f"{sub_disp} (n={seg_cnt})",
                            "segment_count": seg_cnt,
                            "strategy": strat_disp,
                            "lambda_r_T_r": lam_temp,
                            "corpus_bleu1_pct": _num(
                                cm.get("corpus_bleu1_pct", 0.0), 2
                            ),
                            "bleu1_pct": _num(cm.get("bleu1_pct", 0.0), 2),
                            "beam_oracle_bleu1_pct": _num(
                                cm.get("beam_oracle_bleu1_pct", 0.0), 2
                            ),
                            "wer_pct": _num(cm.get("wer_pct", 0.0), 2),
                            "cer_pct": _num(cm.get("cer_pct", 0.0), 2),
                            "bertscore_f1_pct": _num(
                                cm.get("bertscore_f1_pct", 0.0), 2
                            ),
                            "mean_wait_seconds": wait_s,
                            "word_acc_pct": _num(
                                cm.get("word_acc_pct", 0.0), 2
                            ),
                            "in_vocabulary_word_acc_pct": _num(
                                cm.get("in_vocabulary_word_acc_pct", 0.0), 2
                            ),
                            "word_acc_top5_pct": _num(
                                cm.get("word_acc_top5_pct", 0.0), 2
                            ),
                            "content_word_acc_pct": _num(
                                cm.get("content_word_acc_pct", 0.0), 2
                            ),
                            "function_word_acc_pct": _num(
                                cm.get("function_word_acc_pct", 0.0), 2
                            ),
                            "content_minus_function_acc_pp": cf_delta,
                            "content_vs_function_p_value": cf_p,
                        }
                    )
        if cb_rows:
            cb_csv = _write_csv_table(
                tables_dir / f"s3_core_vocabulary_benchmarks_{split}.csv",
                list(cb_rows[0].keys()),
                cb_rows,
            )
            exported["s3_core_vocabulary_benchmarks_csv"] = str(cb_csv)
            tables_map["s3_core_vocabulary_benchmarks"] = str(cb_csv)

    # 1C. 导出跨被试多观测神经聚合表 s3_multi_subject_scaling_{split}.csv
    ms_rows: list[dict[str, Any]] = []
    if isinstance(multi_subj, Mapping) and multi_subj:
        k1_entry = multi_subj.get("k=1") or {}
        k1_by_v = (
            k1_entry.get("by_vocabulary_size")
            if isinstance(k1_entry, Mapping)
            else {}
        ) or {}
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
            k_obs = int(
                m_entry.get("k_observations", str(k_lbl).lstrip("kK=_") or 1)
            )
            by_v = dict(m_entry.get("by_vocabulary_size") or {})
            if full_v_key not in by_v:
                m_subs = m_entry.get("subsets") or {}
                m_all = m_subs.get("all_segments") or {}
                by_v[full_v_key] = {
                    "vocabulary_size": full_v_sz,
                    "bleu1_pct": m_entry.get("bleu1_pct", 0.0),
                    "corpus_bleu1_pct": m_entry.get("corpus_bleu1_pct", 0.0),
                    "wer_pct": m_entry.get("wer_pct", 0.0),
                    "word_acc_pct": m_entry.get("word_acc_pct", 0.0),
                    "in_vocabulary_word_acc_pct": m_entry.get(
                        "in_vocabulary_word_acc_pct", 0.0
                    ),
                    "conditions": m_all.get("conditions")
                    or {"S3_clean": m_entry},
                    "core_subset_80pct": m_subs.get("core_subset_80pct") or {},
                    "core_subset_100pct": m_subs.get("core_subset_100pct")
                    or {},
                }
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
                v_sz = int(
                    v_blk.get("vocabulary_size", str(v_key).lstrip("vV=_") or 0)
                )
                v_conds = v_blk.get("conditions") or {"S3_clean": v_blk}
                k1_v_blk = (
                    k1_by_v.get(v_key)
                    or vocab_scaling.get(v_key)
                    or {}
                )
                k1_v_conds = (
                    k1_v_blk.get("conditions")
                    if isinstance(k1_v_blk, Mapping)
                    else {}
                ) or {}
                c80 = v_blk.get("core_subset_80pct") or {}
                c100 = v_blk.get("core_subset_100pct") or {}

                for strat_disp in (
                    "Rolling-K2",
                    "Word-Only",
                    "Rolling-K0",
                    "Shifted-K2 (-75s)",
                ):
                    cm = _s3_lookup_cond_metrics(v_conds, strat_disp)
                    if cm is None:
                        continue
                    cm_k1 = _s3_lookup_cond_metrics(k1_v_conds, strat_disp)
                    b_val = _num(cm.get("bleu1_pct", 0.0), 2)
                    cb_val = _num(cm.get("corpus_bleu1_pct", 0.0), 2)
                    w_val = _num(cm.get("wer_pct", 0.0), 2)
                    iv_val = _num(cm.get("in_vocabulary_word_acc_pct", 0.0), 2)
                    delta_b = (
                        round(b_val - _num(cm_k1.get("bleu1_pct", 0.0), 2), 2)
                        if cm_k1 is not None and k_obs > 1
                        else 0.0
                    )
                    delta_w = (
                        round(w_val - _num(cm_k1.get("wer_pct", 0.0), 2), 2)
                        if cm_k1 is not None and k_obs > 1
                        else 0.0
                    )
                    delta_iv = (
                        round(
                            iv_val
                            - _num(
                                cm_k1.get("in_vocabulary_word_acc_pct", 0.0), 2
                            ),
                            2,
                        )
                        if cm_k1 is not None and k_obs > 1
                        else 0.0
                    )
                    p_b_item = (
                        sig_map.get(f"g8_k{k_obs}_vs_k1_V{v_sz}_sent_bleu1")
                        if strat_disp == "Rolling-K2" and k_obs > 1
                        else None
                    )
                    p_iv_item = (
                        sig_map.get(f"g8_k{k_obs}_vs_k1_V{v_sz}_in_vocab_top1")
                        if strat_disp == "Rolling-K2" and k_obs > 1
                        else None
                    )
                    ms_rows.append(
                        {
                            "scale_key": str(v_key),
                            "vocabulary_size": v_sz,
                            "k_label": str(k_lbl),
                            "k_observations": k_obs,
                            "strategy": strat_disp,
                            "corpus_bleu1_pct": cb_val,
                            "all_bleu1_pct": b_val,
                            "wer_pct": w_val,
                            "all_word_acc_pct": _num(
                                cm.get("word_acc_pct", 0.0), 2
                            ),
                            "all_in_vocab_word_acc_pct": iv_val,
                            "delta_bleu1_vs_k1_pp": delta_b if k_obs > 1 else "",
                            "p_bleu1_vs_k1": _num(
                                p_b_item.get("p_two_sided", 1.0), 4
                            )
                            if isinstance(p_b_item, Mapping)
                            else "",
                            "delta_wer_vs_k1_pp": delta_w if k_obs > 1 else "",
                            "delta_in_vocab_vs_k1_pp": delta_iv
                            if k_obs > 1
                            else "",
                            "p_in_vocab_vs_k1": _num(
                                p_iv_item.get("p_two_sided", 1.0), 4
                            )
                            if isinstance(p_iv_item, Mapping)
                            else "",
                            "core80_bleu1_pct": _num(c80.get("bleu1_pct", 0.0), 2)
                            if c80 and strat_disp == "Rolling-K2"
                            else "",
                            "core100_bleu1_pct": _num(
                                c100.get("bleu1_pct", 0.0), 2
                            )
                            if c100 and strat_disp == "Rolling-K2"
                            else "",
                        }
                    )
        if ms_rows:
            ms_csv = _write_csv_table(
                tables_dir / f"s3_multi_subject_scaling_{split}.csv",
                list(ms_rows[0].keys()),
                ms_rows,
            )
            exported["s3_multi_subject_scaling_csv"] = str(ms_csv)
            tables_map["s3_multi_subject_scaling"] = str(ms_csv)

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
                    "rolling_k2_bleu1_pct": _num(s_item.get("bleu1_pct", 0.0), 2),
                    "rolling_k0_bleu1_pct": _num(
                        s_item.get("k0_bleu1_pct", 0.0), 2
                    )
                    if "k0_bleu1_pct" in s_item
                    else "",
                    "k2_minus_k0_bleu1_pp": _num(
                        s_item.get("s3_minus_k0_bleu1_pp", 0.0), 2
                    )
                    if "s3_minus_k0_bleu1_pp" in s_item
                    else "",
                    "rolling_k2_word_acc_pct": _num(
                        s_item.get("word_acc_pct", 0.0), 2
                    ),
                    "rolling_k0_word_acc_pct": _num(
                        s_item.get("k0_word_acc_pct", 0.0), 2
                    )
                    if "k0_word_acc_pct" in s_item
                    else "",
                    "k2_minus_k0_word_acc_pp": _num(
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
                "rolling_k2_bleu1_pct",
                "rolling_k2_word_acc_pct",
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
                "rolling_k0_bleu1_pct",
                "k2_minus_k0_bleu1_pp",
                "rolling_k0_word_acc_pct",
                "k2_minus_k0_word_acc_pp",
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
            tables_map["s3_per_subject"] = str(s_csv)

    # 3. 导出多尺度词表表 s3_vocabulary_scaling_{split}.csv
    vs_rows: list[dict[str, Any]] = []
    if isinstance(vocab_scaling, Mapping) and vocab_scaling:
        for scale_key, sc_block in sorted(
            vocab_scaling.items(),
            key=lambda kv: int(
                kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
            )
            if isinstance(kv[1], Mapping)
            else 0,
        ):
            if not isinstance(sc_block, Mapping):
                continue
            v_sz = int(
                sc_block.get(
                    "vocabulary_size", str(scale_key).lstrip("vV=_") or 0
                )
            )
            tok_cov = _num(sc_block.get("test_token_coverage_pct", 0.0), 2)
            conds = sc_block.get("conditions") or {}
            for strat_disp in (
                "Word-Only",
                "Rolling-K0",
                "Rolling-K1",
                "Rolling-K2",
                "Full-Context",
                "Shifted-K0 (-75s)",
                "Shifted-K2 (-75s)",
            ):
                cm = _s3_lookup_cond_metrics(conds, strat_disp)
                if cm is None:
                    continue
                vs_rows.append(
                    {
                        "scale_key": str(scale_key),
                        "vocabulary_size": v_sz,
                        "test_token_coverage_pct": tok_cov,
                        "strategy": strat_disp,
                        "corpus_bleu1_pct": _num(
                            cm.get("corpus_bleu1_pct", 0.0), 2
                        ),
                        "bleu1_pct": _num(cm.get("bleu1_pct", 0.0), 2),
                        "beam_oracle_bleu1_pct": _num(
                            cm.get("beam_oracle_bleu1_pct", 0.0), 2
                        ),
                        "wer_pct": _num(cm.get("wer_pct", 0.0), 2),
                        "cer_pct": _num(cm.get("cer_pct", 0.0), 2),
                        "word_acc_pct": _num(cm.get("word_acc_pct", 0.0), 2),
                        "in_vocabulary_word_acc_pct": _num(
                            cm.get("in_vocabulary_word_acc_pct", 0.0), 2
                        ),
                        "word_acc_top5_pct": _num(
                            cm.get("word_acc_top5_pct", 0.0), 2
                        ),
                        "content_word_acc_pct": _num(
                            cm.get("content_word_acc_pct", 0.0), 2
                        ),
                        "function_word_acc_pct": _num(
                            cm.get("function_word_acc_pct", 0.0), 2
                        ),
                        "bertscore_f1_pct": _num(
                            cm.get("bertscore_f1_pct", 0.0), 2
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
            tables_map["s3_vocabulary_scaling"] = str(vs_csv)

    # 4. 导出 C-gated 帕累托表 s3_confidence_gate_pareto_{split}.csv（同时包含 Clean 与 Shifted (-75s)）
    p_rows: list[dict[str, Any]] = []
    if isinstance(pareto_map, Mapping):
        for p_key, cond_label in (
            ("S3_clean", "Clean (Rolling-K2)"),
            ("S3_shifted_neg75s", "Shifted (-75s) (Shifted-K2)"),
        ):
            p_blk = (
                pareto_map.get(p_key)
                or (pareto_map.get("S3") if p_key == "S3_clean" else None)
                or {}
            )
            pts = (
                p_blk.get("pareto_curve") or []
                if isinstance(p_blk, Mapping)
                else []
            )
            if not pts:
                continue
            base_rw = float(
                pts[0].get("screen_rewrite_rate_pct", 0.0)
                if isinstance(pts[0], Mapping)
                else 0.0
            )
            for pt in pts:
                if not isinstance(pt, Mapping):
                    continue
                rw = float(pt.get("screen_rewrite_rate_pct", 0.0))
                filt_pct = (
                    float(pt["rewrite_filtered_pct"])
                    if "rewrite_filtered_pct" in pt
                    else (
                        max(0.0, (1.0 - rw / base_rw) * 100.0)
                        if base_rw > 0.0
                        else 0.0
                    )
                )
                p_rows.append(
                    {
                        "input_condition": cond_label,
                        "tau_gate": _num(pt.get("tau_gate", 0.0), 2),
                        "screen_rewrite_rate_pct": _num(rw, 2),
                        "flicker_filtered_pct": _num(filt_pct, 2),
                        "corpus_bleu1_pct": _num(
                            pt.get("corpus_bleu1_pct", pt.get("bleu1_pct", 0.0)),
                            2,
                        ),
                        "bleu1_pct": _num(pt.get("bleu1_pct", 0.0), 2),
                        "wer_pct": _num(pt.get("wer_pct", 0.0), 2),
                        "word_acc_pct": _num(pt.get("word_acc_pct", 0.0), 2),
                        "content_word_acc_pct": _num(
                            pt.get("content_word_acc_pct", 0.0), 2
                        ),
                        "bertscore_f1_pct": _num(
                            pt.get("bertscore_f1_pct", 0.0), 2
                        ),
                        "beneficial_repair_rate_pct": _num(
                            pt.get("beneficial_repair_rate_pct", 0.0), 2
                        ),
                        "harmful_rewrite_rate_pct": _num(
                            pt.get("harmful_rewrite_rate_pct", 0.0), 2
                        ),
                        "invalid_flicker_filtered_pct": _num(
                            pt.get("invalid_flicker_filtered_pct", 0.0), 2
                        ),
                        "beneficial_repairs": int(
                            pt.get("beneficial_repairs", 0)
                        ),
                        "harmful_rewrites": int(pt.get("harmful_rewrites", 0)),
                        "triggered_rewrites": int(
                            pt.get("triggered_rewrites", 0)
                        ),
                    }
                )
    if p_rows:
        p_csv = _write_csv_table(
            tables_dir / f"s3_confidence_gate_pareto_{split}.csv",
            list(p_rows[0].keys()),
            p_rows,
        )
        exported["s3_pareto_csv"] = str(p_csv)
        tables_map["s3_confidence_gate_pareto"] = str(p_csv)

    # 4B. 导出 8 组配对统计检验汇总表 s3_paired_significance_{split}.csv
    sig_rows: list[dict[str, Any]] = []
    if isinstance(sig_map, Mapping) and sig_map:
        ordered_sig_items = sorted(
            [
                (k, v)
                for k, v in sig_map.items()
                if isinstance(v, Mapping) and (k.startswith("g") or "group_id" in v)
            ],
            key=lambda kv: (int(kv[1].get("group_id", 99)), str(kv[0])),
        )
        if not ordered_sig_items:
            ordered_sig_items = [
                (k, v) for k, v in sig_map.items() if isinstance(v, Mapping)
            ]
        for comp_k, sig_d in ordered_sig_items:
            sig_rows.append(
                {
                    "test_key": str(comp_k),
                    "group_id": int(sig_d.get("group_id", 0)),
                    "comparison": str(
                        sig_d.get("comparison", sig_d.get("label", comp_k))
                    ),
                    "scale": str(sig_d.get("scale", f"V={full_v_sz}")),
                    "subset": str(sig_d.get("subset", "All Segs")),
                    "metric": str(sig_d.get("metric", "-")),
                    "lhs_value_pct": _num(sig_d.get("lhs_value", 0.0), 2)
                    if "lhs_value" in sig_d
                    else "",
                    "rhs_value_pct": _num(sig_d.get("rhs_value", 0.0), 2)
                    if "rhs_value" in sig_d
                    else "",
                    "effect_pp": _num(sig_d.get("effect_pp", 0.0), 2),
                    "ci_lower_pp": _num(sig_d.get("ci_lower_pp", 0.0), 2),
                    "ci_upper_pp": _num(sig_d.get("ci_upper_pp", 0.0), 2),
                    "p_two_sided": _num(
                        sig_d.get("p_two_sided", sig_d.get("p_one_sided", 1.0)),
                        4,
                    ),
                    "p_holm": _num(
                        sig_d.get(
                            "p_holm",
                            sig_d.get(
                                "p_two_sided", sig_d.get("p_one_sided", 1.0)
                            ),
                        ),
                        4,
                    ),
                    "significance": str(
                        sig_d.get(
                            "significance_holm",
                            sig_d.get("significance", "ns"),
                        )
                    ),
                    "blocks": int(sig_d.get("blocks", 0)),
                }
            )
        if sig_rows:
            sig_csv = _write_csv_table(
                tables_dir / f"s3_paired_significance_{split}.csv",
                list(sig_rows[0].keys()),
                sig_rows,
            )
            exported["s3_paired_significance_csv"] = str(sig_csv)
            tables_map["s3_paired_significance"] = str(sig_csv)

    # 5. 导出逐句对照可读表 s3_decoded_sentences_{split}.csv 与 s3_decoded_sentences_{split}.md
    repaired_top: list[dict[str, Any]] = list(
        report.get("trajectory_examples") or []
    )
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
            tables_map["s3_decoded_sentences"] = str(sent_csv)

            repaired_top = sorted(
                [
                    r
                    for r in sentence_rows
                    if r["beneficial_repairs"] > r["harmful_overwrites"]
                ],
                key=lambda r: (
                    r["beneficial_repairs"] - r["harmful_overwrites"],
                    r["s3_word_acc_pct"],
                ),
                reverse=True,
            )[:15]
            if isinstance(report, dict):
                report["trajectory_examples"] = repaired_top[:10]
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
                        "## 1. Top Beneficial Rolling Revision Examples (`Rolling-K0 -> Rolling-K2` Repairs)",
                        "",
                        "| Seg # | Subject | Reference | Rolling-K0 (Initial) | Rolling-K2 (Revised Final) | Word Acc (`Rolling-K0 -> Rolling-K2`) | Repaired Positions (`Miss -> Hit`) |",
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
                        "| Seg # | Subject | Words (Cov%) | Reference | Rolling-K0 | Rolling-K1 | Rolling-K2 | Rolling-K2 Word Acc |",
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
                    "| Seg # | Subject | Reference | Rolling-K0 | Rolling-K2 | Word Acc | Repairs |",
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

    # 6. 生成 6 模块标准化 Markdown 总报告 s3_evaluation_report_{split}.md
    seed_hdr = (
        f"Seeds `{list(seeds_list)}` (`n={len(seeds_list)}`)"
        if isinstance(seeds_list, Sequence) and seeds_list
        else (f"Seed `{seed_val}`" if seed_val is not None else "Single-Run")
    )
    md_lines = [
        f"# Stage 3 (S3) Sequence Decoding Evaluation Report (`{dataset}` | `{split}` | {seed_hdr})",
        "",
    ]
    s3_clean_m = conditions.get("S3_clean") or conditions.get("S3") or {}
    s0_m = _s3_lookup_cond_metrics(conditions, "Word-Only") or {}
    k0_m = _s3_lookup_cond_metrics(conditions, "Rolling-K0") or {}
    f_m = _s3_lookup_cond_metrics(conditions, "Full-Context") or {}
    sh_m = _s3_lookup_cond_metrics(conditions, "Shifted-K2 (-75s)") or {}
    if s3_clean_m:
        f_part = (
            f"; vs `Full-Context` `{_num(f_m.get('bleu1_pct'), 2):.2f}%`"
            if f_m
            else ""
        )
        s0_part = (
            f"vs `Word-Only` `{_num(s0_m.get('bleu1_pct'), 2):.2f}%` (`Δ={float(s3_clean_m.get('bleu1_pct', 0.0)) - float(s0_m.get('bleu1_pct', 0.0)):+.2f} pp`); "
            if s0_m
            else ""
        )
        md_lines.extend(
            [
                "> [!IMPORTANT]",
                f"> **Executive Headline (`V={full_v_sz}`)**: `Rolling-K2` Sent BLEU-1 = **{_fmt_pct_ms(s3_clean_m, 'bleu1_pct')}** "
                f"(Oracle `{_num(s3_clean_m.get('beam_oracle_bleu1_pct'), 2):.2f}%`, Corpus BLEU-1 `{_num(s3_clean_m.get('corpus_bleu1_pct'), 2):.2f}%`; "
                f"{s0_part}"
                f"vs `Rolling-K0` `{_num(k0_m.get('bleu1_pct'), 2):.2f}%`, `Δ={float(s3_clean_m.get('bleu1_pct', 0.0)) - float(k0_m.get('bleu1_pct', 0.0)):+.2f} pp`"
                f"{f_part}; "
                f"vs `Shifted-K2 (-75s)` `{_num(sh_m.get('bleu1_pct'), 2):.2f}%`, `Δ={float(s3_clean_m.get('bleu1_pct', 0.0)) - float(sh_m.get('bleu1_pct', 0.0)):+.2f} pp`), "
                f"In-Vocab Top-1 = **{_num(s3_clean_m.get('in_vocabulary_word_acc_pct'), 2):.2f}%** (Top-5 `{_num(s3_clean_m.get('word_acc_top5_pct'), 2):.2f}%`), "
                f"Content / Function Acc = `{_num(s3_clean_m.get('content_word_acc_pct'), 2):.2f}%` / `{_num(s3_clean_m.get('function_word_acc_pct'), 2):.2f}%`, "
                f"WER = `{_num(s3_clean_m.get('wer_pct'), 2):.2f}%`, CER = `{_num(s3_clean_m.get('cer_pct'), 2):.2f}%`, "
                f"BERTScore-F1 = **{_num(s3_clean_m.get('bertscore_f1_pct'), 2):.2f}%**, Mean Wait = `{_s3_resolve_mean_wait_s('Rolling-K2', s3_clean_m):.2f}s`.",
                "",
            ]
        )

    # 辅助函数：收集用于模块 1 与模块 2 的 (scale_key, subset_label, sub_conds)
    ordered_scale_blocks: list[tuple[str, str, Mapping[str, Any]]] = []
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
        v_sz = int(
            cb_entry.get("vocabulary_size", str(v_key).lstrip("vV=_") or 0)
        )
        subsets_map = cb_entry.get("subsets") or {}
        c100_blk = subsets_map.get("core_subset_100pct") or {}
        c80_blk = subsets_map.get("core_subset_80pct") or {}
        all_blk = subsets_map.get("all_segments") or {}
        n100 = int(c100_blk.get("segment_count", 0))
        n80 = int(c80_blk.get("segment_count", 0))
        nall = int(all_blk.get("segment_count", 0))

        if v_sz < full_v_sz:
            if n100 > 0 and isinstance(c100_blk.get("conditions"), Mapping):
                ordered_scale_blocks.append(
                    (str(v_key), f"0% OOV (n={n100})", c100_blk["conditions"])
                )
            elif n80 > 0 and isinstance(c80_blk.get("conditions"), Mapping):
                ordered_scale_blocks.append(
                    (
                        str(v_key),
                        f">=80% In-Vocab (n={n80})",
                        c80_blk["conditions"],
                    )
                )
            if (
                n100 > 0
                and n80 > n100
                and isinstance(c80_blk.get("conditions"), Mapping)
            ):
                ordered_scale_blocks.append(
                    (
                        str(v_key),
                        f">=80% In-Vocab (n={n80})",
                        c80_blk["conditions"],
                    )
                )
            if nall > 0 and isinstance(all_blk.get("conditions"), Mapping):
                ordered_scale_blocks.append(
                    (str(v_key), f"All Segs (n={nall})", all_blk["conditions"])
                )
        else:
            if (
                n100 > 0
                and n100 < nall
                and isinstance(c100_blk.get("conditions"), Mapping)
            ):
                ordered_scale_blocks.append(
                    (str(v_key), f"0% OOV (n={n100})", c100_blk["conditions"])
                )
            if nall > 0 and isinstance(all_blk.get("conditions"), Mapping):
                lbl_all = (
                    f"0% OOV / All (n={nall})"
                    if n100 == nall
                    else f"All Segs (n={nall})"
                )
                ordered_scale_blocks.append(
                    (str(v_key), lbl_all, all_blk["conditions"])
                )

    # 模块 1：核心词表主表数据
    if ordered_scale_blocks:
        md_lines.extend(
            [
                "## Module 1: Core Vocabulary Main Table (`V in {50, 250, 2020}`)",
                "",
                "| Scale | Subset | Strategy | lambda_r (T_r) | Corpus BLEU-1 | Sent BLEU-1 (Oracle) | WER | CER | BERTScore-F1 | Mean Wait (s) |",
                "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for v_key, sub_lbl, sub_conds in ordered_scale_blocks:
            for strat_disp in module1_strategies:
                cm = _s3_lookup_cond_metrics(sub_conds, strat_disp)
                if cm is None:
                    continue
                lam_t = _s3_format_lambda_temp(strat_disp, state_hp)
                c_bleu = _fmt_pct_ms(cm, "corpus_bleu1_pct")
                s_bleu = _fmt_pct_ms(
                    cm, "bleu1_pct", bold=(strat_disp == "Rolling-K2")
                )
                o_bleu = _num(cm.get("beam_oracle_bleu1_pct", 0.0), 2)
                wer_s = _fmt_pct_ms(cm, "wer_pct")
                cer_s = _fmt_pct_ms(cm, "cer_pct")
                bert_s = _fmt_pct_ms(
                    cm, "bertscore_f1_pct", bold=(strat_disp == "Rolling-K2")
                )
                wait_s = _s3_resolve_mean_wait_s(strat_disp, cm)
                md_lines.append(
                    f"| `{v_key}` | `{sub_lbl}` | **`{strat_disp}`** | `{lam_t}` | "
                    f"{c_bleu} | {s_bleu} (`{o_bleu:.2f}%`) | {wer_s} | {cer_s} | {bert_s} | `{wait_s:.2f}` |"
                )
        md_lines.append("")

    # 模块 2：词位准确率与实义词/功能词拆解
    if ordered_scale_blocks:
        md_lines.extend(
            [
                "## Module 2: Word Position Accuracy & Content/Function Breakdown",
                "",
                "| Scale | Subset | Strategy | Word Acc (All) | In-Vocab Top-1 Acc | Position Top-5 Acc | Content Acc | Function Acc | Content - Function Delta (p-value) |",
                "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for v_key, sub_lbl, sub_conds in ordered_scale_blocks:
            for strat_disp in module2_strategies:
                cm = _s3_lookup_cond_metrics(sub_conds, strat_disp)
                if cm is None:
                    continue
                w_acc = _fmt_pct_ms(
                    cm, "word_acc_pct", bold=(strat_disp == "Rolling-K2")
                )
                iv_acc = _fmt_pct_ms(
                    cm,
                    "in_vocabulary_word_acc_pct",
                    bold=(strat_disp == "Rolling-K2"),
                )
                top5_acc = _fmt_pct_ms(cm, "word_acc_top5_pct")
                c_acc = _fmt_pct_ms(cm, "content_word_acc_pct")
                f_acc = _fmt_pct_ms(cm, "function_word_acc_pct")
                cf_d = _num(
                    cm.get(
                        "content_minus_function_acc_pp",
                        float(cm.get("content_word_acc_pct", 0.0))
                        - float(cm.get("function_word_acc_pct", 0.0)),
                    ),
                    2,
                )
                cf_p = _num(cm.get("content_vs_function_p_value", 1.0), 4)
                md_lines.append(
                    f"| `{v_key}` | `{sub_lbl}` | **`{strat_disp}`** | {w_acc} | {iv_acc} | "
                    f"{top5_acc} | {c_acc} | {f_acc} | `{cf_d:+.2f} pp` (`p={cf_p:.4f}`) |"
                )
        md_lines.append("")

    # 模块 3：门控防闪烁帕累托扫描表（Clean 与 Shifted (-75s)）
    if p_rows:
        md_lines.extend(
            [
                "## Module 3: Confidence-Gated Anti-Flicker Pareto Sweep (`C-Gated`)",
                "",
                "| Input Condition | tau_gate | Screen Rewrite Rate | Flicker Filtered (%) | Corpus BLEU-1 | Sent BLEU-1 | WER |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for pr in p_rows:
            tau_v = float(pr["tau_gate"])
            tau_lbl = f"{tau_v:.1f} (Ungated)" if tau_v == 0.0 else f"{tau_v:.1f}"
            md_lines.append(
                f"| **`{pr['input_condition']}`** | `{tau_lbl}` | "
                f"`{float(pr['screen_rewrite_rate_pct']):.2f}%` | "
                f"**{float(pr['flicker_filtered_pct']):.2f}%** | "
                f"{float(pr['corpus_bleu1_pct']):.2f}% | "
                f"**{float(pr['bleu1_pct']):.2f}%** | "
                f"{float(pr['wer_pct']):.2f}% |"
            )
        md_lines.append("")

    # 模块 4：多尺度词表缩放与跨受试者多观测聚合（严禁在表头使用未转义 |V|）
    if vs_rows or ms_rows:
        md_lines.extend(
            [
                "## Module 4: Multi-Scale Vocabulary Scaling & Cross-Subject Aggregation (`V in {50, 100, 250, 2020}`)",
                "",
            ]
        )
        if vs_rows:
            md_lines.extend(
                [
                    "### 4A. Single-Subject Streaming Vocabulary Scaling (`k=1`)",
                    "",
                    "| Scale | Token Coverage | Strategy | Corpus BLEU-1 | Sent BLEU-1 (Oracle) | WER | In-Vocab Top-1 | Position Top-5 Acc |",
                    "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for vr in vs_rows:
                md_lines.append(
                    f"| **`{vr['scale_key']}`** | {float(vr['test_token_coverage_pct']):.2f}% | "
                    f"**`{vr['strategy']}`** | {float(vr['corpus_bleu1_pct']):.2f}% | "
                    f"**{float(vr['bleu1_pct']):.2f}%** (`{float(vr['beam_oracle_bleu1_pct']):.2f}%`) | "
                    f"{float(vr['wer_pct']):.2f}% | **{float(vr['in_vocabulary_word_acc_pct']):.2f}%** | "
                    f"{float(vr['word_acc_top5_pct']):.2f}% |"
                )
            md_lines.append("")

        if ms_rows:
            md_lines.extend(
                [
                    "### 4B. Cross-Subject Multi-Observation Aggregation (`k in {1, 2, 5, 10}`)",
                    "",
                    "| Scale | Subjects k | Strategy | Corpus BLEU-1 | Sent BLEU-1 | WER | In-Vocab Top-1 | Gain vs k=1 (Sent BLEU-1, p-value) | Gain vs k=1 (In-Vocab Top-1, p-value) |",
                    "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
                ]
            )
            for mr in ms_rows:
                k_obs = int(mr["k_observations"])
                if k_obs <= 1:
                    g_bleu_s = "Baseline (`k=1`)"
                    g_iv_s = "Baseline (`k=1`)"
                else:
                    db = mr.get("delta_bleu1_vs_k1_pp")
                    pb = mr.get("p_bleu1_vs_k1")
                    div = mr.get("delta_in_vocab_vs_k1_pp")
                    piv = mr.get("p_in_vocab_vs_k1")
                    g_bleu_s = (
                        f"**{float(db):+.2f} pp** (`p={float(pb):.4f}`)"
                        if db != "" and pb != ""
                        else (f"**{float(db):+.2f} pp**" if db != "" else "-")
                    )
                    g_iv_s = (
                        f"**{float(div):+.2f} pp** (`p={float(piv):.4f}`)"
                        if div != "" and piv != ""
                        else (f"**{float(div):+.2f} pp**" if div != "" else "-")
                    )
                md_lines.append(
                    f"| **`{mr['scale_key']}`** | **`{mr['k_label']}`** | `{mr['strategy']}` | "
                    f"{float(mr['corpus_bleu1_pct']):.2f}% | **{float(mr['all_bleu1_pct']):.2f}%** | "
                    f"{float(mr['wer_pct']):.2f}% | **{float(mr['all_in_vocab_word_acc_pct']):.2f}%** | "
                    f"{g_bleu_s} | {g_iv_s} |"
                )
            md_lines.append("")

    # 模块 5：配对统计检验 p 值汇总表（8 组固定比较 + Holm 校正）
    if sig_rows:
        md_lines.extend(
            [
                "## Module 5: Paired Material-Block Statistical Significance Summary (8 Fixed Groups, 10,000 Resamples + Holm)",
                "",
                "| Group | Comparison | Scale | Subset | Metric | LHS | RHS | Effect Delta (pp) | 95% Block CI (pp) | Two-Sided p | Holm p | Sig |",
                "| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for sr in sig_rows:
            lhs_s = (
                f"{float(sr['lhs_value_pct']):.2f}%"
                if sr.get("lhs_value_pct") != ""
                else "-"
            )
            rhs_s = (
                f"{float(sr['rhs_value_pct']):.2f}%"
                if sr.get("rhs_value_pct") != ""
                else "-"
            )
            grp_s = f"**#{sr['group_id']}**" if int(sr.get("group_id", 0)) > 0 else "-"
            md_lines.append(
                f"| {grp_s} | **`{sr['comparison']}`** | `{sr['scale']}` | `{sr['subset']}` | "
                f"`{sr['metric']}` | {lhs_s} | {rhs_s} | "
                f"**{float(sr['effect_pp']):+.2f} pp** | "
                f"`[{float(sr['ci_lower_pp']):+.2f}, {float(sr['ci_upper_pp']):+.2f}]` | "
                f"`{float(sr['p_two_sided']):.4f}` | `{float(sr['p_holm']):.4f}` | "
                f"**{sr['significance']}** |"
            )
        md_lines.append("")

    # 模块 6：受试者级序列解码表现与代表性滚动修正案例
    if (isinstance(per_subj, Sequence) and per_subj) or repaired_top:
        md_lines.extend(
            [
                "## Module 6: Per-Subject Breakdown & Qualitative Rolling Revision Trajectories",
                "",
            ]
        )
        if isinstance(per_subj, Sequence) and per_subj:
            if isinstance(subj_sum, Mapping) and subj_sum:
                md_lines.append(
                    f"- **Across-Subject Rolling-K2 Sent BLEU-1**: `{_num(subj_sum.get('mean_bleu1_pct'), 2):.2f}% ± {_num(subj_sum.get('std_bleu1_pct'), 2):.2f}%` | "
                    f"**Word Acc**: `{_num(subj_sum.get('mean_word_acc_pct'), 2):.2f}% ± {_num(subj_sum.get('std_word_acc_pct'), 2):.2f}%` | "
                    f"**Improved Subjects (`Rolling-K2 > Rolling-K0`)**: `{subj_sum.get('improved_subjects_count', '-')}/{subj_sum.get('subject_count', len(per_subj))}`"
                )
                md_lines.append("")
            md_lines.extend(
                [
                    "### 6A. Per-Subject Performance (`Rolling-K2` vs `Rolling-K0`)",
                    "",
                    "| Subject | Segments (Words) | Rolling-K2 BLEU-1 | Rolling-K0 BLEU-1 | Delta BLEU-1 | Rolling-K2 Word Acc | Content Acc | Function Acc | BERTScore-F1 |",
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

        if repaired_top:
            md_lines.extend(
                [
                    "### 6B. Representative Rolling Revision Trajectory Examples (`Rolling-K0 -> Rolling-K2`)",
                    "",
                    "| Seg # | Subject | Reference | Rolling-K0 (Draft) | Rolling-K2 (Revised) | Word Acc (`K0 -> K2`) | Repaired Positions |",
                    "| ---: | :--- | :--- | :--- | :--- | ---: | :--- |",
                ]
            )
            for r in repaired_top[:10]:
                md_lines.append(
                    f"| `{r['segment_index']}` | `{r['subject_id']}` | {r['reference_sentence']} | "
                    f"{r['k0_initial_sentence']} | **{r['s3_final_sentence']}** | "
                    f"`{r['k0_word_acc_pct']:.1f}% -> {r['s3_word_acc_pct']:.1f}%` (**{r['net_word_acc_gain_pp']:+.1f} pp**) | "
                    f"`{r['repair_highlights'] or '-'}` |"
                )
            md_lines.append("")

    md_path = out_dir / f"s3_evaluation_report_{split}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    exported["s3_report_md"] = str(md_path)
    exported["markdown"] = str(md_path)
    exported["tables"] = tables_map
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
            f"## {s1}. 50-Word Retrieval Summary (Word-Macro & Event-Micro Top-10, Top-1, MRR, Median Rank)",
            "",
            f"{meta_hdr} 50-Word Clean Word-Macro Top-10 | 50-Word Shift Word-Macro Top-10 | 50-Word Top-10 Gap | 50-Word Clean Word-Macro Top-1 | 50-Word Shift Word-Macro Top-1 | 50-Word Top-1 Gap | 50-Word Clean Event-Micro Top-10 | 50-Word Shift Event-Micro Top-10 | 50-Word Clean MRR | 50-Word Shift MRR | 50-Word Clean Median Rank | 50-Word Shift Median Rank |",
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
            f"## {s2}. Full-Vocabulary Retrieval Summary (Word-Macro & Event-Micro Top-10, Top-1, MRR, Median Rank)",
            "",
            f"{meta_hdr} Full-Vocab Clean Word-Macro Top-10 | Full-Vocab Shift Word-Macro Top-10 | Full-Vocab Top-10 Gap | Full-Vocab Clean Word-Macro Top-1 | Full-Vocab Shift Word-Macro Top-1 | Full-Vocab Top-1 Gap | Full-Vocab Clean Event-Micro Top-10 | Full-Vocab Shift Event-Micro Top-10 | Full-Vocab Clean MRR | Full-Vocab Shift MRR | Full-Vocab Clean Median Rank | Full-Vocab Shift Median Rank |",
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
            f"## {s3a}. Revision Horizons (`K0 / K1 / K2`) — 50-Word Word-Macro Top-10 & Word-Macro Top-1 Breakdown",
            "",
            f"{meta_hdr} K0 50-Word Clean Top-10 | K0 50-Word Shift Top-10 | K1 50-Word Clean Top-10 | K1 50-Word Shift Top-10 | K2 50-Word Clean Top-10 | K2 50-Word Shift Top-10 | K2 50-Word Gap Top-10 | K2-K0 50-Word Clean Top-10 | K0 50-Word Clean Top-1 | K0 50-Word Shift Top-1 | K1 50-Word Clean Top-1 | K1 50-Word Shift Top-1 | K2 50-Word Clean Top-1 | K2 50-Word Shift Top-1 | K2 50-Word Gap Top-1 | K2-K0 50-Word Clean Top-1 |",
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
            f"## {s3b}. Revision Horizons (`K0 / K1 / K2`) — Full-Vocabulary Word-Macro Top-10, Word-Macro Top-1 & MRR Breakdown",
            "",
            f"{meta_hdr} K0 Full-Vocab Clean Top-10 | K0 Full-Vocab Shift Top-10 | K1 Full-Vocab Clean Top-10 | K1 Full-Vocab Shift Top-10 | K2 Full-Vocab Clean Top-10 | K2 Full-Vocab Shift Top-10 | K2 Full-Vocab Gap Top-10 | K2-K0 Full-Vocab Clean Top-10 | K0 Full-Vocab Clean Top-1 | K0 Full-Vocab Shift Top-1 | K1 Full-Vocab Clean Top-1 | K1 Full-Vocab Shift Top-1 | K2 Full-Vocab Clean Top-1 | K2 Full-Vocab Shift Top-1 | K2 Full-Vocab Gap Top-1 | K2-K0 Full-Vocab Clean Top-1 | K0 Full-Vocab Clean MRR | K2 Full-Vocab Clean MRR |",
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


def export_cross_model_comparison_tables_and_markdown(
    comparison_payload: Mapping[str, Any],
    output_dir: Path | str,
) -> dict[str, str]:
    """导出四模型联合对比与 3 处配对统计检验的独立 Markdown 报告及 CSV 表格（multi_seed_cross_model_comparison_{split}.md & tables/*.csv）。"""
    out = Path(output_dir)
    tables_dir = out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, str] = {}

    split = str(comparison_payload.get("split", "test")).strip().lower()
    dataset = str(comparison_payload.get("dataset", "pallier2025"))
    models_summary = list(comparison_payload.get("models_summary") or [])
    hypothesis_tests = list(comparison_payload.get("hypothesis_tests") or [])
    paired_tests_flat = list(comparison_payload.get("paired_tests_flat") or [])

    # 1. Build CSV rows for Four-Model Summary
    summary_csv_rows: list[dict[str, Any]] = []
    for m_rec in models_summary:
        if not isinstance(m_rec, Mapping):
            continue
        m_map = m_rec.get("metrics") or {}
        v50_10 = m_map.get("vocab50_macro_top10") or {}
        v50_1 = m_map.get("vocab50_macro_top1") or {}
        v50_mic10 = m_map.get("vocab50_micro_top10") or {}
        v50_mrr = m_map.get("vocab50_mrr") or {}
        fv_10 = m_map.get("full_vocab_macro_top10") or {}
        fv_1 = m_map.get("full_vocab_macro_top1") or {}
        fv_mic10 = m_map.get("full_vocab_micro_top10") or {}
        fv_mrr = m_map.get("full_vocab_mrr") or {}
        summary_csv_rows.append(
            {
                "slot_key": str(m_rec.get("slot_key", "")),
                "model_label": str(m_rec.get("label", "")),
                "stage": str(m_rec.get("stage", "")),
                "exp_tag": str(m_rec.get("exp_tag", "")),
                "control_method": str(m_rec.get("control_method", "")),
                "seed_count": int(m_rec.get("seed_count", 1)),
                "mean_wait_seconds": _num(m_rec.get("mean_wait_seconds", 1.0), 2),
                "lookahead_wait_seconds": _num(
                    m_rec.get("lookahead_wait_seconds", 0.0), 2
                ),
                "wait_saving_vs_context_pct": _num(
                    m_rec.get("wait_saving_vs_context_pct", 0.0), 2
                ),
                "lookahead_saving_vs_context_pct": _num(
                    m_rec.get("lookahead_saving_vs_context_pct", 0.0), 2
                ),
                "mean_clean_vocab50_macro_top10_pct": _pct(v50_10.get("mean_clean", 0.0)),
                "std_clean_vocab50_macro_top10_pct": _pct(v50_10.get("std_clean", 0.0)),
                "mean_shifted_vocab50_macro_top10_pct": _pct(v50_10.get("mean_shifted", 0.0)),
                "std_shifted_vocab50_macro_top10_pct": _pct(v50_10.get("std_shifted", 0.0)),
                "p_shifted_vs_chance_vocab50_macro_top10": v50_10.get("p_shifted_vs_chance", ""),
                "mean_gap_vocab50_macro_top10_pp": _pct(v50_10.get("mean_gap", 0.0)),
                "std_gap_vocab50_macro_top10_pp": _pct(v50_10.get("std_gap", 0.0)),
                "p_gap_vocab50_macro_top10": v50_10.get("p_gap", ""),
                "mean_clean_vocab50_macro_top1_pct": _pct(v50_1.get("mean_clean", 0.0)),
                "std_clean_vocab50_macro_top1_pct": _pct(v50_1.get("std_clean", 0.0)),
                "mean_shifted_vocab50_macro_top1_pct": _pct(v50_1.get("mean_shifted", 0.0)),
                "std_shifted_vocab50_macro_top1_pct": _pct(v50_1.get("std_shifted", 0.0)),
                "mean_gap_vocab50_macro_top1_pp": _pct(v50_1.get("mean_gap", 0.0)),
                "std_gap_vocab50_macro_top1_pp": _pct(v50_1.get("std_gap", 0.0)),
                "p_gap_vocab50_macro_top1": v50_1.get("p_gap", ""),
                "mean_clean_vocab50_micro_top10_pct": _pct(v50_mic10.get("mean_clean", 0.0)),
                "std_clean_vocab50_micro_top10_pct": _pct(v50_mic10.get("std_clean", 0.0)),
                "mean_clean_vocab50_mrr": _num(v50_mrr.get("mean_clean", 0.0), 4),
                "std_clean_vocab50_mrr": _num(v50_mrr.get("std_clean", 0.0), 4),
                "mean_clean_full_vocab_macro_top10_pct": _pct(fv_10.get("mean_clean", 0.0)),
                "std_clean_full_vocab_macro_top10_pct": _pct(fv_10.get("std_clean", 0.0)),
                "mean_shifted_full_vocab_macro_top10_pct": _pct(fv_10.get("mean_shifted", 0.0)),
                "std_shifted_full_vocab_macro_top10_pct": _pct(fv_10.get("std_shifted", 0.0)),
                "mean_gap_full_vocab_macro_top10_pp": _pct(fv_10.get("mean_gap", 0.0)),
                "std_gap_full_vocab_macro_top10_pp": _pct(fv_10.get("std_gap", 0.0)),
                "p_gap_full_vocab_macro_top10": fv_10.get("p_gap", ""),
                "mean_clean_full_vocab_macro_top1_pct": _pct(fv_1.get("mean_clean", 0.0)),
                "std_clean_full_vocab_macro_top1_pct": _pct(fv_1.get("std_clean", 0.0)),
                "mean_shifted_full_vocab_macro_top1_pct": _pct(fv_1.get("mean_shifted", 0.0)),
                "std_shifted_full_vocab_macro_top1_pct": _pct(fv_1.get("std_shifted", 0.0)),
                "mean_gap_full_vocab_macro_top1_pp": _pct(fv_1.get("mean_gap", 0.0)),
                "std_gap_full_vocab_macro_top1_pp": _pct(fv_1.get("std_gap", 0.0)),
                "p_gap_full_vocab_macro_top1": fv_1.get("p_gap", ""),
                "mean_clean_full_vocab_micro_top10_pct": _pct(fv_mic10.get("mean_clean", 0.0)),
                "std_clean_full_vocab_micro_top10_pct": _pct(fv_mic10.get("std_clean", 0.0)),
                "mean_clean_full_vocab_mrr": _num(fv_mrr.get("mean_clean", 0.0), 4),
                "std_clean_full_vocab_mrr": _num(fv_mrr.get("std_clean", 0.0), 4),
                "mean_gap_full_vocab_mrr": _num(fv_mrr.get("mean_gap", 0.0), 4),
                "std_gap_full_vocab_mrr": _num(fv_mrr.get("std_gap", 0.0), 4),
            }
        )
    if summary_csv_rows:
        sum_csv_path = _write_csv_table(
            tables_dir / f"multi_seed_cross_model_summary_{split}.csv",
            list(summary_csv_rows[0].keys()),
            summary_csv_rows,
        )
        exported["cross_model_summary_csv"] = str(sum_csv_path)

    # 2. Build CSV rows for Paired Statistical Hypothesis Tests
    paired_csv_rows: list[dict[str, Any]] = []
    for r_test in paired_tests_flat:
        if not isinstance(r_test, Mapping):
            continue
        is_r = bool(r_test.get("is_ratio", True))
        paired_csv_rows.append(
            {
                "hypothesis_id": str(r_test.get("hypothesis_id", "")),
                "hypothesis_title": str(r_test.get("hypothesis_title", "")),
                "quantity": str(r_test.get("quantity", "")),
                "quantity_label": str(r_test.get("quantity_label", "")),
                "metric_key": str(r_test.get("metric_key", "")),
                "metric_label": str(r_test.get("metric_label", "")),
                "model_a_key": str(r_test.get("model_a_key", "")),
                "model_a_label": str(r_test.get("model_a_label", "")),
                "model_a_control": str(r_test.get("model_a_control", "")),
                "model_a_mean": (
                    _pct(r_test.get("model_a_mean", 0.0))
                    if is_r
                    else _num(r_test.get("model_a_mean", 0.0), 4)
                ),
                "model_a_std": (
                    _pct(r_test.get("model_a_std", 0.0))
                    if is_r
                    else _num(r_test.get("model_a_std", 0.0), 4)
                ),
                "model_b_key": str(r_test.get("model_b_key", "")),
                "model_b_label": str(r_test.get("model_b_label", "")),
                "model_b_control": str(r_test.get("model_b_control", "")),
                "model_b_mean": (
                    _pct(r_test.get("model_b_mean", 0.0))
                    if is_r
                    else _num(r_test.get("model_b_mean", 0.0), 4)
                ),
                "model_b_std": (
                    _pct(r_test.get("model_b_std", 0.0))
                    if is_r
                    else _num(r_test.get("model_b_std", 0.0), 4)
                ),
                "delta_mean": (
                    _pct(r_test.get("delta_mean", 0.0))
                    if is_r
                    else _num(r_test.get("delta_mean", 0.0), 4)
                ),
                "delta_std": (
                    _pct(r_test.get("delta_std", 0.0))
                    if is_r
                    else _num(r_test.get("delta_std", 0.0), 4)
                ),
                "t_stat": _num(r_test.get("t_stat", 0.0), 4),
                "df": _num(r_test.get("df", 9.0), 1),
                "p_value": r_test.get("p_value", 1.0),
                "significance": str(r_test.get("significance", "ns")),
                "p_str": str(r_test.get("p_str", "")),
                "hypothesis_supported": bool(r_test.get("hypothesis_supported", True)),
                "scientific_verdict": str(r_test.get("scientific_verdict", "")),
            }
        )
    if paired_csv_rows:
        tests_csv_path = _write_csv_table(
            tables_dir / f"multi_seed_cross_model_paired_tests_{split}.csv",
            list(paired_csv_rows[0].keys()),
            paired_csv_rows,
        )
        exported["cross_model_paired_tests_csv"] = str(tests_csv_path)

    # 3. Build Markdown Report: multi_seed_cross_model_comparison_{split}.md
    md_lines: list[str] = [
        f"# Cross-Model Retrieval, Debiasing & Paired Statistical Comparison ({split.upper()}, Mean ± SD)",
        "",
        f"- **Dataset**: `{dataset}` | **Split**: `{split}` | **Models Compared**: `{len(models_summary)}`",
    ]
    for idx_m, m_rec in enumerate(models_summary, start=1):
        if not isinstance(m_rec, Mapping):
            continue
        md_lines.append(
            f"- **Model {idx_m} (`{m_rec.get('slot_key')}`)**: **{m_rec.get('label')}** — "
            f"`stage={m_rec.get('stage')}`, `control={m_rec.get('control_method')}`, "
            f"`seeds={m_rec.get('seeds')}`, `exp_tag={m_rec.get('exp_tag') or 'default'}`"
        )
    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 1. Four-Model Retrieval, Debiasing & Waiting Time Summary (Mean ± SD)",
            "",
            "### 1A. 50-Word Frozen Benchmark & Input Latency Summary",
            "",
            "| Model Architecture | Stage | Control Method | Mean Wait (Lookahead) | 50w Clean Top-10 | 50w Control Top-10 (vs. Chance p) | 50w Net Gap Top-10 (Paired p) | 50w Clean Top-1 | 50w Control Top-1 | 50w Net Gap Top-1 (Paired p) | 50w Clean MRR |",
            "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for m_rec in models_summary:
        if not isinstance(m_rec, Mapping):
            continue
        m_map = m_rec.get("metrics") or {}
        v50_10 = m_map.get("vocab50_macro_top10") or {}
        v50_1 = m_map.get("vocab50_macro_top1") or {}
        v50_mrr = m_map.get("vocab50_mrr") or {}
        wait_s = float(m_rec.get("mean_wait_seconds", 1.0))
        look_s = float(m_rec.get("lookahead_wait_seconds", 0.0))
        save_look = m_rec.get("lookahead_saving_vs_context_pct")
        if save_look is not None and float(save_look) > 0.0:
            wait_cell = f"`{wait_s:.2f}s` (`{look_s:.2f}s`, **-{float(save_look):.1f}%**)"
        else:
            wait_cell = f"`{wait_s:.2f}s` (`{look_s:.2f}s`)"
        md_lines.append(
            f"| **{m_rec.get('label')}** | `{m_rec.get('stage')}` | `{m_rec.get('control_method')}` | {wait_cell} | "
            f"**{_fmt_mean_sd_md_pct(v50_10.get('mean_clean'), v50_10.get('std_clean'))}** | "
            f"{_append_p_str(_fmt_mean_sd_md_pct(v50_10.get('mean_shifted'), v50_10.get('std_shifted')), v50_10.get('p_str_shifted_vs_chance'), prefix='vs. 20%: ')} | "
            f"{_append_p_str(_fmt_mean_sd_md_pct(v50_10.get('mean_gap'), v50_10.get('std_gap'), signed=True, suffix=' pp'), v50_10.get('p_str_gap'), bold_val=True)} | "
            f"**{_fmt_mean_sd_md_pct(v50_1.get('mean_clean'), v50_1.get('std_clean'))}** | "
            f"{_fmt_mean_sd_md_pct(v50_1.get('mean_shifted'), v50_1.get('std_shifted'))} | "
            f"{_append_p_str(_fmt_mean_sd_md_pct(v50_1.get('mean_gap'), v50_1.get('std_gap'), signed=True, suffix=' pp'), v50_1.get('p_str_gap'), bold_val=True)} | "
            f"{_fmt_mean_sd_md_num(v50_mrr.get('mean_clean'), v50_mrr.get('std_clean'), digits=4)} |"
        )

    md_lines.extend(
        [
            "",
            "### 1B. Full-Vocabulary (|V|=2020) Benchmark Summary",
            "",
            "| Model Architecture | Control Method | Full-Vocab Clean Word-Macro Top-10 | Full-Vocab Control Word-Macro Top-10 (vs. Chance p) | Full-Vocab Net Gap Word-Macro Top-10 (Paired p) | Full-Vocab Clean Word-Macro Top-1 | Full-Vocab Control Word-Macro Top-1 | Full-Vocab Net Gap Word-Macro Top-1 (Paired p) | Full-Vocab Clean Sample-Micro Top-10 | Full-Vocab Clean MRR |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for m_rec in models_summary:
        if not isinstance(m_rec, Mapping):
            continue
        m_map = m_rec.get("metrics") or {}
        fv_10 = m_map.get("full_vocab_macro_top10") or {}
        fv_1 = m_map.get("full_vocab_macro_top1") or {}
        fv_mic10 = m_map.get("full_vocab_micro_top10") or {}
        fv_mrr = m_map.get("full_vocab_mrr") or {}
        md_lines.append(
            f"| **{m_rec.get('label')}** | `{m_rec.get('control_method')}` | "
            f"**{_fmt_mean_sd_md_pct(fv_10.get('mean_clean'), fv_10.get('std_clean'))}** | "
            f"{_append_p_str(_fmt_mean_sd_md_pct(fv_10.get('mean_shifted'), fv_10.get('std_shifted')), fv_10.get('p_str_shifted_vs_chance'))} | "
            f"{_append_p_str(_fmt_mean_sd_md_pct(fv_10.get('mean_gap'), fv_10.get('std_gap'), signed=True, suffix=' pp'), fv_10.get('p_str_gap'), bold_val=True)} | "
            f"**{_fmt_mean_sd_md_pct(fv_1.get('mean_clean'), fv_1.get('std_clean'))}** | "
            f"{_fmt_mean_sd_md_pct(fv_1.get('mean_shifted'), fv_1.get('std_shifted'))} | "
            f"{_append_p_str(_fmt_mean_sd_md_pct(fv_1.get('mean_gap'), fv_1.get('std_gap'), signed=True, suffix=' pp'), fv_1.get('p_str_gap'), bold_val=True)} | "
            f"{_fmt_mean_sd_md_pct(fv_mic10.get('mean_clean'), fv_mic10.get('std_clean'))} | "
            f"{_fmt_mean_sd_md_num(fv_mrr.get('mean_clean'), fv_mrr.get('std_clean'), digits=4)} |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Three Key Paired Statistical Hypothesis Tests (Model A vs. Model B)",
            "",
            "### 2A. Unified Overview of the 3 Core Paired Hypothesis Tests",
            "",
            "| Hypothesis Comparison (A vs. B) | Target Quantity | Metric | Model A (Mean ± SD) | Model B (Mean ± SD) | Paired Difference Δ (A - B) | t-stat (df) | Paired p-value | Scientific Conclusion |",
            "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
    )
    for r_test in paired_tests_flat:
        if not isinstance(r_test, Mapping):
            continue
        is_r = bool(r_test.get("is_ratio", True))
        is_gap = str(r_test.get("quantity")) == "net_gap"
        if is_r:
            a_cell = _fmt_mean_sd_md_pct(
                r_test.get("model_a_mean"),
                r_test.get("model_a_std"),
                signed=is_gap,
                suffix=" pp" if is_gap else "%",
            )
            b_cell = _fmt_mean_sd_md_pct(
                r_test.get("model_b_mean"),
                r_test.get("model_b_std"),
                signed=is_gap,
                suffix=" pp" if is_gap else "%",
            )
            d_cell = _fmt_mean_sd_md_pct(
                r_test.get("delta_mean"),
                r_test.get("delta_std"),
                signed=True,
                suffix=" pp",
            )
        else:
            a_cell = _fmt_mean_sd_md_num(
                r_test.get("model_a_mean"),
                r_test.get("model_a_std"),
                digits=4,
                signed=is_gap,
            )
            b_cell = _fmt_mean_sd_md_num(
                r_test.get("model_b_mean"),
                r_test.get("model_b_std"),
                digits=4,
                signed=is_gap,
            )
            d_cell = _fmt_mean_sd_md_num(
                r_test.get("delta_mean"),
                r_test.get("delta_std"),
                digits=4,
                signed=True,
            )
        t_cell = f"`t={float(r_test.get('t_stat', 0.0)):+.2f}` (`df={float(r_test.get('df', 9.0)):.0f}`)"
        md_lines.append(
            f"| **{r_test.get('hypothesis_title')}** | `{r_test.get('quantity_label')}` | **{r_test.get('metric_label')}** | "
            f"{a_cell} | {b_cell} | **{d_cell}** | {t_cell} | **{r_test.get('p_str', '-')}** | {r_test.get('scientific_verdict', '')} |"
        )

    for idx_h, h_block in enumerate(hypothesis_tests, start=1):
        if not isinstance(h_block, Mapping):
            continue
        md_lines.extend(
            [
                "",
                f"### 2.{idx_h + 1} {h_block.get('title')}",
                "",
                f"> {h_block.get('description')}",
                "",
                "| Target Quantity | Metric | Model A (`control`) | Model B (`control`) | Paired Δ (A - B) | t-stat (df) | Paired p-value | Verdict |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |",
            ]
        )
        for r_test in h_block.get("rows") or []:
            if not isinstance(r_test, Mapping):
                continue
            is_r = bool(r_test.get("is_ratio", True))
            is_gap = str(r_test.get("quantity")) == "net_gap"
            if is_r:
                a_cell = _fmt_mean_sd_md_pct(
                    r_test.get("model_a_mean"),
                    r_test.get("model_a_std"),
                    signed=is_gap,
                    suffix=" pp" if is_gap else "%",
                )
                b_cell = _fmt_mean_sd_md_pct(
                    r_test.get("model_b_mean"),
                    r_test.get("model_b_std"),
                    signed=is_gap,
                    suffix=" pp" if is_gap else "%",
                )
                d_cell = _fmt_mean_sd_md_pct(
                    r_test.get("delta_mean"),
                    r_test.get("delta_std"),
                    signed=True,
                    suffix=" pp",
                )
            else:
                a_cell = _fmt_mean_sd_md_num(
                    r_test.get("model_a_mean"),
                    r_test.get("model_a_std"),
                    digits=4,
                    signed=is_gap,
                )
                b_cell = _fmt_mean_sd_md_num(
                    r_test.get("model_b_mean"),
                    r_test.get("model_b_std"),
                    digits=4,
                    signed=is_gap,
                )
                d_cell = _fmt_mean_sd_md_num(
                    r_test.get("delta_mean"),
                    r_test.get("delta_std"),
                    digits=4,
                    signed=True,
                )
            t_cell = f"`t={float(r_test.get('t_stat', 0.0)):+.2f}` (`df={float(r_test.get('df', 9.0)):.0f}`)"
            md_lines.append(
                f"| `{r_test.get('quantity_label')}` | **{r_test.get('metric_label')}** | "
                f"{a_cell} (`{r_test.get('model_a_control')}`) | {b_cell} (`{r_test.get('model_b_control')}`) | "
                f"**{d_cell}** | {t_cell} | **{r_test.get('p_str', '-')}** | {r_test.get('scientific_verdict', '')} |"
            )

    md_path = out / f"multi_seed_cross_model_comparison_{split}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    exported["cross_model_comparison_md"] = str(md_path)
    return exported


__all__ = [
    "export_common_mode_subtraction_tables_and_markdown",
    "export_cross_model_comparison_tables_and_markdown",
    "export_multi_seed_tables_and_markdown",
    "export_sequence_decoding_tables_and_markdown",
    "export_stage_comparison_tables_and_markdown",
    "export_subspace_coherence_debiasing_tables_and_markdown",
    "export_word_evaluation_tables_and_markdown",
]
