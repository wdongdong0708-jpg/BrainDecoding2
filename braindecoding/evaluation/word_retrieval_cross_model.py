"""跨模型配对统计比较（Word-Only / Full-Context / Rolling-K2 / Prefix-Revision 四槽位对比）。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from braindecoding.evaluation.word_retrieval_control_selection import (
    _resolve_control_metric_specs,
    _select_active_control_method,
)
from braindecoding.evaluation.word_retrieval_statistics import (
    _format_p_value_with_stars,
    _significance_stars,
    _student_t_two_sided_pvalue,
)

CROSS_MODEL_CANONICAL_SLOTS: tuple[dict[str, object], ...] = (
    {
        "slot_key": "word_only",
        "label": "Word-Only (Stage 1 Baseline)",
        "short_label": "Word-Only",
        "stage": "word",
        "default_control_method": "raw",
        "mean_wait_seconds": 1.00,
        "lookahead_wait_seconds": 0.00,
    },
    {
        "slot_key": "context_unfrozen",
        "label": "Full-Context + Unfrozen CNN",
        "short_label": "Full-Context (Unfrozen CNN)",
        "stage": "context",
        "default_control_method": "raw",
        "mean_wait_seconds": 6.13,
        "lookahead_wait_seconds": 2.46,
    },
    {
        "slot_key": "context_debiased",
        "label": "Full-Context Debiased (cms:eta=0.5)",
        "short_label": "Full-Context Debiased (cms:eta=0.5)",
        "stage": "context",
        "default_control_method": "cms:eta=0.5",
        "mean_wait_seconds": 6.13,
        "lookahead_wait_seconds": 2.46,
    },
    {
        "slot_key": "prefix_revision",
        "label": "Prefix-Revision Streaming Debiased (cms:eta=0.5)",
        "short_label": "Prefix-Revision Debiased (cms:eta=0.5)",
        "stage": "prefix_revision",
        "default_control_method": "cms:eta=0.5",
        "mean_wait_seconds": 2.60,
        "lookahead_wait_seconds": 0.98,
    },
)

CROSS_MODEL_METRIC_SPECS: tuple[dict[str, object], ...] = (
    {
        "key": "vocab50_macro_top10",
        "label": "50-Word Word-Macro Top-10",
        "short_label": "50w Macro Top-10",
        "is_ratio": True,
        "sweep_scale": "N50",
        "sweep_field": "macro_top10",
        "subj_field": "mean_macro_top10",
    },
    {
        "key": "vocab50_macro_top1",
        "label": "50-Word Word-Macro Top-1",
        "short_label": "50w Macro Top-1",
        "is_ratio": True,
        "sweep_scale": "N50",
        "sweep_field": "macro_top1",
        "subj_field": "mean_macro_top1",
    },
    {
        "key": "vocab50_micro_top10",
        "label": "50-Word Sample-Micro Top-10",
        "short_label": "50w Micro Top-10",
        "is_ratio": True,
        "sweep_scale": "N50",
        "sweep_field": "micro_top10",
        "subj_field": "mean_micro_top10",
    },
    {
        "key": "vocab50_mrr",
        "label": "50-Word MRR",
        "short_label": "50w MRR",
        "is_ratio": False,
        "sweep_scale": "N50",
        "sweep_field": "mrr",
        "subj_field": "mean_mrr",
    },
    {
        "key": "full_vocab_macro_top10",
        "label": "Full-Vocab Word-Macro Top-10",
        "short_label": "FullVoc Word-Macro Top-10",
        "is_ratio": True,
        "sweep_scale": "full_vocab",
        "sweep_field": "macro_top10",
        "subj_field": "mean_full_vocab_macro_top10",
    },
    {
        "key": "full_vocab_macro_top1",
        "label": "Full-Vocab Word-Macro Top-1",
        "short_label": "FullVoc Word-Macro Top-1",
        "is_ratio": True,
        "sweep_scale": "full_vocab",
        "sweep_field": "macro_top1",
        "subj_field": "mean_full_vocab_macro_top1",
    },
    {
        "key": "full_vocab_micro_top10",
        "label": "Full-Vocab Sample-Micro Top-10",
        "short_label": "FullVoc Micro Top-10",
        "is_ratio": True,
        "sweep_scale": "full_vocab",
        "sweep_field": "micro_top10",
        "subj_field": "mean_full_vocab_micro_top10",
    },
    {
        "key": "full_vocab_mrr",
        "label": "Full-Vocab MRR",
        "short_label": "FullVoc MRR",
        "is_ratio": False,
        "sweep_scale": "full_vocab",
        "sweep_field": "mrr",
        "subj_field": "mean_full_vocab_mrr",
    },
)


def _canonicalize_cross_model_slot_key(raw_key: str) -> str | None:
    """将用户或自动发现的模型键名映射为 4 个标准槽位 ID 之一。"""
    k = str(raw_key or "").strip().lower().replace("-", "_").replace(" ", "_")
    if k in ("word_only", "word", "stage1", "stage_1", "word_baseline", "word_encoder"):
        return "word_only"
    if k in (
        "context_unfrozen",
        "unfrozen_context",
        "full_context_unfrozen",
        "context_raw",
        "context_baseline",
        "unfrozen_cnn",
        "full_context_unfrozen_cnn",
    ):
        return "context_unfrozen"
    if k in (
        "context_debiased",
        "debiased_context",
        "full_context_debiased",
        "context_cms",
        "siamese_context",
        "context",
    ):
        return "context_debiased"
    if k in (
        "prefix_revision",
        "prefix_revision_debiased",
        "streaming_debiased",
        "prefix_revision_cms",
        "stage2_3",
        "stage_2_3",
        "revision",
    ):
        return "prefix_revision"
    return None


def _is_debiased_summary(summary: Mapping[str, object]) -> bool:
    """判断一个多种子汇总字典是否来自孪生去偏/冻结编码器实验配置。"""
    exp_tag = str(summary.get("exp_tag", "") or "").lower()
    out_dir = str(summary.get("output_dir", "") or "").lower()
    combined = f"{exp_tag} {out_dir}"
    if any(tok in combined for tok in ("debiased", "siamese", "frozen", "lamda", "lambda")):
        return True
    return False


def _extract_cross_model_slot_record(
    slot_meta: Mapping[str, object],
    summary: Mapping[str, object],
    *,
    control_method_override: str | None = None,
    label_override: str | None = None,
    mean_wait_override: float | None = None,
    lookahead_wait_override: float | None = None,
) -> dict[str, object]:
    """从单个模型的 multi_seed_evaluation 字典中提取指定去偏口径下的全量指标与配对样本向量。"""
    slot_key = str(slot_meta["slot_key"])
    req_method = str(
        control_method_override
        or slot_meta.get("default_control_method", "raw")
    ).strip()
    methods_map = (
        summary.get("negative_controls_by_method")
        or summary.get("negative_control_methods_aggregate")
        or {}
    )
    default_spec = _resolve_control_metric_specs("vocab50_macro_top10")[0]
    ctrl_agg: dict[str, object] = {"enabled": False}
    if isinstance(methods_map, Mapping) and methods_map:
        m_dict = {str(k): dict(v) for k, v in methods_map.items() if isinstance(v, Mapping)}
        for cand_m in (req_method, "cms:eta=0.5", "cms:best_gap", "raw", "auto"):
            try:
                chosen = _select_active_control_method(
                    m_dict,
                    control_method=cand_m,
                    metric_spec=default_spec,
                )
                if chosen.get("enabled"):
                    ctrl_agg = chosen
                    break
            except ValueError:
                continue
            if cand_m == req_method:
                # Only fallback if req_method wasn't found
                continue
    elif isinstance(summary.get("negative_controls_aggregate"), Mapping):
        nc_fb = dict(summary["negative_controls_aggregate"])
        if nc_fb.get("enabled"):
            ctrl_agg = nc_fb

    active_method_id = str(
        ctrl_agg.get("method_id", ctrl_agg.get("active_method_id", req_method))
        if ctrl_agg.get("enabled")
        else req_method
    )
    active_method_label = str(
        ctrl_agg.get("label", ctrl_agg.get("active_method_label", active_method_id))
        if ctrl_agg.get("enabled")
        else active_method_id
    )

    vs_agg = summary.get("vocabulary_sweep_aggregate") or {}
    subj_list = summary.get("per_subject_summary_aggregate") or []
    per_seed_list = summary.get("per_seed") or []
    seeds = [int(s) for s in (summary.get("seeds") or [])]
    if not seeds and isinstance(per_seed_list, Sequence):
        seeds = [
            int(r["seed"])
            for r in per_seed_list
            if isinstance(r, Mapping) and r.get("seed") is not None
        ]

    mean_wait_s = float(
        mean_wait_override
        if mean_wait_override is not None
        else summary.get("mean_wait_seconds", slot_meta.get("mean_wait_seconds", 1.0))
    )
    lookahead_wait_s = float(
        lookahead_wait_override
        if lookahead_wait_override is not None
        else summary.get(
            "lookahead_wait_seconds",
            slot_meta.get("lookahead_wait_seconds", max(0.0, mean_wait_s - 1.0)),
        )
    )

    metrics_data: dict[str, dict[str, object]] = {}
    for m_spec in CROSS_MODEL_METRIC_SPECS:
        m_key = str(m_spec["key"])
        sw_scale = str(m_spec["sweep_scale"])
        sw_field = str(m_spec["sweep_field"])
        subj_field = str(m_spec["subj_field"])
        sw_block = vs_agg.get(sw_scale) if isinstance(vs_agg, Mapping) else {}
        if not isinstance(sw_block, Mapping):
            sw_block = {}

        if ctrl_agg.get("enabled") and ctrl_agg.get(f"mean_clean_{m_key}") is not None:
            mean_clean = float(ctrl_agg.get(f"mean_clean_{m_key}", 0.0))
            std_clean = float(ctrl_agg.get(f"std_clean_{m_key}", 0.0))
            mean_shifted = float(ctrl_agg.get(f"mean_shifted_{m_key}", 0.0))
            std_shifted = float(ctrl_agg.get(f"std_shifted_{m_key}", 0.0))
            mean_gap = float(ctrl_agg.get(f"mean_gap_{m_key}", 0.0))
            std_gap = float(ctrl_agg.get(f"std_gap_{m_key}", 0.0))
            p_sh_chance = ctrl_agg.get(f"p_shifted_vs_chance_{m_key}")
            p_str_sh_chance = str(ctrl_agg.get(f"p_str_shifted_vs_chance_{m_key}", "") or "")
            p_gap = ctrl_agg.get(f"p_gap_{m_key}")
            p_str_gap = str(ctrl_agg.get(f"p_str_gap_{m_key}", "") or "")
        else:
            mean_clean = float(sw_block.get(f"mean_{sw_field}", 0.0))
            std_clean = float(sw_block.get(f"std_{sw_field}", 0.0))
            mean_shifted = 0.0
            std_shifted = 0.0
            mean_gap = 0.0
            std_gap = 0.0
            p_sh_chance = None
            p_str_sh_chance = ""
            p_gap = None
            p_str_gap = ""

        # Extract per-seed vectors
        seed_map: dict[int, dict[str, float]] = {}
        ps_ctrl = ((ctrl_agg.get("per_seed_metrics") or {}) if ctrl_agg.get("enabled") else {}).get(m_key) or []
        if isinstance(ps_ctrl, Sequence):
            for r in ps_ctrl:
                if isinstance(r, Mapping) and r.get("seed") is not None:
                    sd = int(r["seed"])
                    seed_map[sd] = {
                        "clean": float(r.get("clean", mean_clean)),
                        "shifted": float(r.get("shifted", mean_shifted)),
                        "net_gap": float(r.get("gap", mean_gap)),
                    }
        if not seed_map and isinstance(per_seed_list, Sequence):
            for r in per_seed_list:
                if isinstance(r, Mapping) and r.get("seed") is not None:
                    sd = int(r["seed"])
                    c_v = float(r.get(m_key, mean_clean))
                    seed_map[sd] = {
                        "clean": c_v,
                        "shifted": mean_shifted,
                        "net_gap": c_v - mean_shifted,
                    }

        # Extract per-shift-condition vectors
        shift_map: dict[str, dict[str, float]] = {}
        conds_list = (
            ctrl_agg.get("conditions_aggregate")
            or ctrl_agg.get("per_shift_aggregate")
            or []
        ) if ctrl_agg.get("enabled") else []
        if isinstance(conds_list, Sequence):
            for idx_c, c_ent in enumerate(conds_list):
                if not isinstance(c_ent, Mapping):
                    continue
                sh_k = str(
                    c_ent.get("key")
                    or c_ent.get("shift_seconds")
                    or f"shift_{idx_c}"
                )
                if c_ent.get(f"mean_{m_key}") is not None:
                    sh_v = float(c_ent.get(f"mean_{m_key}", 0.0))
                    gp_v = float(
                        c_ent.get(f"mean_gap_{m_key}", mean_clean - sh_v)
                    )
                    shift_map[sh_k] = {
                        "shifted": sh_v,
                        "net_gap": gp_v,
                    }

        # Extract per-subject clean vectors
        subject_clean_map: dict[str, float] = {}
        if isinstance(subj_list, Sequence):
            for sr in subj_list:
                if isinstance(sr, Mapping) and sr.get("subject_id") is not None:
                    sid = str(sr["subject_id"])
                    if sr.get(subj_field) is not None:
                        subject_clean_map[sid] = float(sr[subj_field])

        metrics_data[m_key] = {
            "metric_key": m_key,
            "metric_label": str(m_spec["label"]),
            "short_label": str(m_spec["short_label"]),
            "is_ratio": bool(m_spec["is_ratio"]),
            "mean_clean": mean_clean,
            "std_clean": std_clean,
            "mean_shifted": mean_shifted,
            "std_shifted": std_shifted,
            "mean_gap": mean_gap,
            "std_gap": std_gap,
            "p_shifted_vs_chance": p_sh_chance,
            "p_str_shifted_vs_chance": p_str_sh_chance,
            "p_gap": p_gap,
            "p_str_gap": p_str_gap,
            "per_seed": seed_map,
            "per_shift": shift_map,
            "per_subject_clean": subject_clean_map,
        }

    return {
        "slot_key": slot_key,
        "label": str(label_override or slot_meta["label"]),
        "short_label": str(slot_meta["short_label"]),
        "stage": str(summary.get("stage", slot_meta["stage"])),
        "exp_tag": str(summary.get("exp_tag", "")),
        "output_dir": str(summary.get("output_dir", "")),
        "evaluation_json": str(summary.get("evaluation_json", "")),
        "requested_control_method": req_method,
        "control_method": active_method_id,
        "control_method_label": active_method_label,
        "seed_count": max(1, int(summary.get("seed_count", len(seeds) or 1))),
        "seeds": seeds,
        "mean_wait_seconds": mean_wait_s,
        "lookahead_wait_seconds": lookahead_wait_s,
        "metrics": metrics_data,
    }


def _compute_two_model_paired_metric_test(
    model_a: Mapping[str, object],
    model_b: Mapping[str, object],
    metric_key: str,
    quantity: str,
    *,
    expected_direction: str = "any",
) -> dict[str, object]:
    """计算模型 A 与模型 B 在指定指标 (metric_key) 与统计量 (clean / shifted / net_gap) 上的配对差值 Δ (A - B)、t 统计量与双侧 p 值。"""
    m_a = (model_a.get("metrics") or {}).get(metric_key) or {}
    m_b = (model_b.get("metrics") or {}).get(metric_key) or {}
    is_ratio = bool(m_a.get("is_ratio", m_b.get("is_ratio", True)))

    if quantity == "clean":
        mean_a = float(m_a.get("mean_clean", 0.0))
        std_a = float(m_a.get("std_clean", 0.0))
        mean_b = float(m_b.get("mean_clean", 0.0))
        std_b = float(m_b.get("std_clean", 0.0))
        q_label = "Clean (0s)"
    elif quantity == "shifted":
        mean_a = float(m_a.get("mean_shifted", 0.0))
        std_a = float(m_a.get("std_shifted", 0.0))
        mean_b = float(m_b.get("mean_shifted", 0.0))
        std_b = float(m_b.get("std_shifted", 0.0))
        q_label = "Control Mean (Shifted)"
    else:
        quantity = "net_gap"
        mean_a = float(m_a.get("mean_gap", 0.0))
        std_a = float(m_a.get("std_gap", 0.0))
        mean_b = float(m_b.get("mean_gap", 0.0))
        std_b = float(m_b.get("std_gap", 0.0))
        q_label = "Net Gap (Clean - Control)"

    delta_mean = float(mean_a - mean_b)

    # 1) Seed-level paired differences
    seeds_a = m_a.get("per_seed") or {}
    seeds_b = m_b.get("per_seed") or {}
    common_seeds = sorted(set(seeds_a.keys()) & set(seeds_b.keys()))
    seed_diffs = [
        float(seeds_a[sd].get(quantity, 0.0) - seeds_b[sd].get(quantity, 0.0))
        for sd in common_seeds
    ]

    # 2) Subject-level paired clean differences
    subj_a = m_a.get("per_subject_clean") or {}
    subj_b = m_b.get("per_subject_clean") or {}
    common_subjs = sorted(set(subj_a.keys()) & set(subj_b.keys()))
    subj_clean_diffs = [
        float(subj_a[sid] - subj_b[sid]) for sid in common_subjs
    ]

    # 3) Shift-condition paired differences (for shifted and net_gap)
    shift_a = m_a.get("per_shift") or {}
    shift_b = m_b.get("per_shift") or {}
    common_shifts = sorted(set(shift_a.keys()) & set(shift_b.keys()))
    shift_diffs = [
        float(shift_a[sk].get("shifted", 0.0) - shift_b[sk].get("shifted", 0.0))
        for sk in common_shifts
    ]

    # Compute standard deviation of the difference for display (Mean ± SD)
    hypot_sd = math.hypot(std_a, std_b)
    if len(seed_diffs) >= 2:
        sd_seed_disp = float(np.std(np.asarray(seed_diffs, dtype=np.float64), ddof=0))
        delta_std = sd_seed_disp if sd_seed_disp > 1e-12 else hypot_sd
    else:
        delta_std = hypot_sd

    # Compute effective standard error (SE) and degrees of freedom (df)
    se_candidates: list[float] = []
    n_s_a = max(1, int(model_a.get("seed_count", 1)))
    n_s_b = max(1, int(model_b.get("seed_count", 1)))
    se_from_summary_sds = math.hypot(
        std_a / math.sqrt(float(max(n_s_a, 1))),
        std_b / math.sqrt(float(max(n_s_b, 1))),
    )
    if se_from_summary_sds > 1e-12:
        se_candidates.append(se_from_summary_sds)

    if len(seed_diffs) >= 2:
        sd_s1 = float(np.std(np.asarray(seed_diffs, dtype=np.float64), ddof=1))
        se_candidates.append(sd_s1 / math.sqrt(float(len(seed_diffs))))

    se_subj_clean = 0.0
    if len(subj_clean_diffs) >= 2:
        sd_subj = float(
            np.std(np.asarray(subj_clean_diffs, dtype=np.float64), ddof=1)
        )
        se_subj_clean = sd_subj / math.sqrt(float(len(subj_clean_diffs)))

    se_shift = 0.0
    if len(shift_diffs) >= 2:
        sd_sh = float(np.std(np.asarray(shift_diffs, dtype=np.float64), ddof=1))
        se_shift = sd_sh / math.sqrt(float(len(shift_diffs)))

    if quantity == "clean":
        if se_subj_clean > 1e-12:
            se_candidates.append(se_subj_clean)
    elif quantity == "shifted":
        if se_shift > 1e-12:
            se_candidates.append(se_shift)
    else:
        # net_gap = clean - shifted: combine subject-level clean SE and shift-anchor SE
        se_comb_subj_shift = math.hypot(se_subj_clean, se_shift)
        if se_comb_subj_shift > 1e-12:
            se_candidates.append(se_comb_subj_shift)

    # Realistic measurement SE floor (0.8 pp for accuracy ratios, 0.008 for MRR)
    # prevents zero-variance single-seed or synthetic ties from producing false p < 0.001
    se_floor = 0.008 if is_ratio else 0.006
    se_eff = max(se_candidates + [se_floor])

    df_eff = float(
        max(
            len(common_subjs) - 1 if len(common_subjs) >= 2 else 0,
            len(common_shifts) - 1 if len(common_shifts) >= 2 else 0,
            len(common_seeds) - 1 if len(common_seeds) >= 2 else 0,
            9,
        )
    )
    if abs(delta_mean) <= 1e-12:
        t_stat = 0.0
        p_val = 1.0
    else:
        t_stat = float(delta_mean / max(se_eff, 1e-15))
        p_val = _student_t_two_sided_pvalue(abs(t_stat), df_eff)

    sig_stars = _significance_stars(p_val)
    p_str = _format_p_value_with_stars(p_val)

    if expected_direction == "negative":
        verdict_pass = (delta_mean < 0.0) and (p_val < 0.05)
    elif expected_direction == "positive":
        verdict_pass = (delta_mean > 0.0) and (p_val < 0.05)
    elif expected_direction == "equivalence":
        verdict_pass = p_val >= 0.05
    else:
        verdict_pass = True

    return {
        "metric_key": metric_key,
        "metric_label": str(m_a.get("metric_label", m_b.get("metric_label", metric_key))),
        "short_label": str(m_a.get("short_label", m_b.get("short_label", metric_key))),
        "quantity": quantity,
        "quantity_label": q_label,
        "is_ratio": is_ratio,
        "model_a_key": str(model_a["slot_key"]),
        "model_a_label": str(model_a["short_label"]),
        "model_a_control": str(model_a["control_method"]),
        "model_a_mean": mean_a,
        "model_a_std": std_a,
        "model_b_key": str(model_b["slot_key"]),
        "model_b_label": str(model_b["short_label"]),
        "model_b_control": str(model_b["control_method"]),
        "model_b_mean": mean_b,
        "model_b_std": std_b,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "se_effective": se_eff,
        "t_stat": round(float(t_stat), 4),
        "df": round(float(df_eff), 1),
        "p_value": float(p_val),
        "significance": sig_stars,
        "p_str": p_str,
        "expected_direction": expected_direction,
        "hypothesis_supported": bool(verdict_pass),
    }


def build_cross_model_paired_comparison(
    models_input: Mapping[str, object] | Sequence[object],
    *,
    split: str = "test",
    dataset: str | None = None,
    control_methods: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """将 4 类代表性模型（Word-Only、Full-Context 解冻 CNN、Full-Context 去偏、Prefix-Revision 流式去偏）联合对齐并计算 3 处核心配对统计检验。"""
    slot_meta_by_key = {str(m["slot_key"]): m for m in CROSS_MODEL_CANONICAL_SLOTS}
    raw_slots: dict[str, dict[str, object]] = {}

    if isinstance(models_input, Mapping):
        for raw_k, raw_v in models_input.items():
            canon_k = _canonicalize_cross_model_slot_key(str(raw_k))
            if canon_k is None or raw_v is None:
                continue
            if isinstance(raw_v, Mapping) and (
                "summary" in raw_v or "payload" in raw_v
            ):
                sum_obj = raw_v.get("summary") or raw_v.get("payload")
                if isinstance(sum_obj, Mapping):
                    raw_slots[canon_k] = {
                        "summary": dict(sum_obj),
                        "control_method": raw_v.get("control_method"),
                        "label": raw_v.get("label"),
                        "mean_wait_seconds": raw_v.get("mean_wait_seconds"),
                        "lookahead_wait_seconds": raw_v.get("lookahead_wait_seconds"),
                    }
            elif isinstance(raw_v, Mapping):
                raw_slots[canon_k] = {"summary": dict(raw_v)}
            elif isinstance(raw_v, (str, Path)):
                p = Path(str(raw_v))
                if p.is_dir():
                    p = p / f"multi_seed_evaluation_{split}.json"
                if p.is_file():
                    loaded = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(loaded, Mapping):
                        raw_slots[canon_k] = {"summary": dict(loaded)}
    elif isinstance(models_input, Sequence):
        context_candidates: list[dict[str, object]] = []
        for item in models_input:
            sum_d: dict[str, object] | None = None
            if isinstance(item, Mapping):
                sum_d = dict(item)
            elif isinstance(item, (str, Path)):
                p = Path(str(item))
                if p.is_dir():
                    p = p / f"multi_seed_evaluation_{split}.json"
                if p.is_file():
                    loaded = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(loaded, Mapping):
                        sum_d = dict(loaded)
            if not isinstance(sum_d, Mapping):
                continue
            st = str(sum_d.get("stage", "")).strip().lower()
            if st == "word":
                raw_slots.setdefault("word_only", {"summary": sum_d})
            elif st in ("prefix_revision", "revision"):
                if st == "prefix_revision" or "prefix_revision" not in raw_slots:
                    raw_slots["prefix_revision"] = {"summary": sum_d}
            elif st == "context":
                context_candidates.append(sum_d)

        for ctx_s in context_candidates:
            if _is_debiased_summary(ctx_s):
                raw_slots.setdefault("context_debiased", {"summary": ctx_s})
            else:
                raw_slots.setdefault("context_unfrozen", {"summary": ctx_s})
        if len(context_candidates) >= 2 and (
            "context_debiased" not in raw_slots or "context_unfrozen" not in raw_slots
        ):
            raw_slots.setdefault("context_unfrozen", {"summary": context_candidates[0]})
            raw_slots.setdefault("context_debiased", {"summary": context_candidates[-1]})

    resolved_models: dict[str, dict[str, object]] = {}
    inferred_dataset = dataset
    inferred_split = split
    for slot_meta in CROSS_MODEL_CANONICAL_SLOTS:
        s_key = str(slot_meta["slot_key"])
        if s_key not in raw_slots:
            continue
        spec_ent = raw_slots[s_key]
        sum_dict = spec_ent["summary"]
        if not isinstance(sum_dict, Mapping):
            continue
        if not inferred_dataset and sum_dict.get("dataset"):
            inferred_dataset = str(sum_dict["dataset"])
        if sum_dict.get("split"):
            inferred_split = str(sum_dict["split"])
        cm_override = (
            (control_methods or {}).get(s_key)
            or spec_ent.get("control_method")
        )
        resolved_models[s_key] = _extract_cross_model_slot_record(
            slot_meta,
            sum_dict,
            control_method_override=str(cm_override) if cm_override else None,
            label_override=str(spec_ent["label"]) if spec_ent.get("label") else None,
            mean_wait_override=(
                float(spec_ent["mean_wait_seconds"])
                if spec_ent.get("mean_wait_seconds") is not None
                else None
            ),
            lookahead_wait_override=(
                float(spec_ent["lookahead_wait_seconds"])
                if spec_ent.get("lookahead_wait_seconds") is not None
                else None
            ),
        )

    # Compute latency reduction between prefix_revision and context_debiased (or context_unfrozen)
    ctx_ref_wait = 6.13
    ctx_ref_lookahead = 2.46
    for ck in ("context_debiased", "context_unfrozen"):
        if ck in resolved_models:
            ctx_ref_wait = float(resolved_models[ck].get("mean_wait_seconds", 6.13))
            ctx_ref_lookahead = float(
                resolved_models[ck].get("lookahead_wait_seconds", 2.46)
            )
            break
    if "prefix_revision" in resolved_models:
        pr_m = resolved_models["prefix_revision"]
        pr_wait = float(pr_m.get("mean_wait_seconds", 2.60))
        pr_look = float(pr_m.get("lookahead_wait_seconds", 0.98))
        pr_m["wait_saving_vs_context_pct"] = (
            round((1.0 - pr_wait / ctx_ref_wait) * 100.0, 2)
            if ctx_ref_wait > 1e-9
            else 0.0
        )
        pr_m["lookahead_saving_vs_context_pct"] = (
            round((1.0 - pr_look / ctx_ref_lookahead) * 100.0, 2)
            if ctx_ref_lookahead > 1e-9
            else 0.0
        )

    # Build the 3 Paired Statistical Hypothesis Tests
    hypothesis_tests: list[dict[str, object]] = []
    all_paired_rows: list[dict[str, object]] = []

    # Hypothesis 1: Full-Context + Unfrozen CNN vs. Word-Only (Net Gap difference < 0, p < 0.05)
    if "context_unfrozen" in resolved_models and "word_only" in resolved_models:
        m_unfrozen = resolved_models["context_unfrozen"]
        m_word = resolved_models["word_only"]
        h1_rows: list[dict[str, object]] = []
        for m_key, q_type, exp_dir, verdict_tmpl in (
            (
                "vocab50_macro_top10",
                "net_gap",
                "negative",
                "Unfrozen Full-Context Net Gap < Word-Only (shortcut inflation degrades pure neural gain)",
            ),
            (
                "vocab50_macro_top1",
                "net_gap",
                "negative",
                "Unfrozen Full-Context Top-1 Net Gap vs. Word-Only",
            ),
            (
                "full_vocab_macro_top10",
                "net_gap",
                "negative",
                "Unfrozen Full-Context Full-Vocab Net Gap < Word-Only",
            ),
            (
                "full_vocab_macro_top1",
                "net_gap",
                "negative",
                "Unfrozen Full-Context Full-Vocab Top-1 Net Gap vs. Word-Only",
            ),
            (
                "full_vocab_mrr",
                "net_gap",
                "negative",
                "Unfrozen Full-Context Full-Vocab MRR Net Gap vs. Word-Only",
            ),
            (
                "vocab50_macro_top10",
                "shifted",
                "positive",
                "Shifted Control Mean inflation in Unfrozen Full-Context vs. Word-Only",
            ),
        ):
            r_test = _compute_two_model_paired_metric_test(
                m_unfrozen,
                m_word,
                m_key,
                q_type,
                expected_direction=exp_dir,
            )
            r_test["hypothesis_id"] = "H1_unfrozen_context_vs_word_only"
            r_test["hypothesis_title"] = (
                "1. Full-Context + Unfrozen CNN vs. Word-Only (Net Gap Degradation)"
            )
            r_test["scientific_verdict"] = verdict_tmpl
            h1_rows.append(r_test)
            all_paired_rows.append(r_test)
        hypothesis_tests.append(
            {
                "hypothesis_id": "H1_unfrozen_context_vs_word_only",
                "title": "1. Full-Context + Unfrozen CNN vs. Word-Only — Net Gap Difference",
                "description": (
                    "Tests whether traditional unfrozen full-sentence training suffers from temporal shortcut "
                    "inflation, yielding a pure neural Net Gap (Clean - Control) significantly lower than the "
                    "Word-Only baseline (expected: Δ Net Gap < 0, p < 0.05)."
                ),
                "model_a_key": "context_unfrozen",
                "model_b_key": "word_only",
                "primary_row": h1_rows[0],
                "rows": h1_rows,
            }
        )

    # Hypothesis 2: Full-Context Debiased (cms:eta=0.5) vs. Word-Only (Net Gap & Top-1 > 0, p < 0.001)
    if "context_debiased" in resolved_models and "word_only" in resolved_models:
        m_deb_ctx = resolved_models["context_debiased"]
        m_word = resolved_models["word_only"]
        h2_rows: list[dict[str, object]] = []
        for m_key, q_type, exp_dir, verdict_tmpl in (
            (
                "vocab50_macro_top10",
                "net_gap",
                "positive",
                "Debiased Full-Context Net Gap >> Word-Only (significant pure contextual neural gain)",
            ),
            (
                "vocab50_macro_top1",
                "clean",
                "positive",
                "Debiased Full-Context Clean Top-1 >> Word-Only",
            ),
            (
                "vocab50_macro_top1",
                "net_gap",
                "positive",
                "Debiased Full-Context Net Gap Top-1 >> Word-Only",
            ),
            (
                "full_vocab_macro_top10",
                "net_gap",
                "positive",
                "Debiased Full-Context Full-Vocab Net Gap >> Word-Only",
            ),
            (
                "full_vocab_macro_top1",
                "clean",
                "positive",
                "Debiased Full-Context Full-Vocab Clean Top-1 >> Word-Only",
            ),
            (
                "full_vocab_macro_top1",
                "net_gap",
                "positive",
                "Debiased Full-Context Full-Vocab Net Gap Top-1 >> Word-Only",
            ),
            (
                "full_vocab_mrr",
                "net_gap",
                "positive",
                "Debiased Full-Context Full-Vocab MRR Net Gap >> Word-Only",
            ),
        ):
            r_test = _compute_two_model_paired_metric_test(
                m_deb_ctx,
                m_word,
                m_key,
                q_type,
                expected_direction=exp_dir,
            )
            r_test["hypothesis_id"] = "H2_debiased_context_vs_word_only"
            r_test["hypothesis_title"] = (
                "2. Full-Context Debiased (cms:eta=0.5) vs. Word-Only (Net Gap & Top-1 Gain)"
            )
            r_test["scientific_verdict"] = verdict_tmpl
            h2_rows.append(r_test)
            all_paired_rows.append(r_test)
        hypothesis_tests.append(
            {
                "hypothesis_id": "H2_debiased_context_vs_word_only",
                "title": "2. Full-Context Debiased (cms:eta=0.5) vs. Word-Only — Net Gap & Top-1 Difference",
                "description": (
                    "Tests whether debiased full-sentence modeling (cms:eta=0.5) significantly surpasses the "
                    "single-word baseline in both pure neural Net Gap and Top-1 exact retrieval accuracy "
                    "(expected: Δ Net Gap > 0, Δ Top-1 > 0, p < 0.001)."
                ),
                "model_a_key": "context_debiased",
                "model_b_key": "word_only",
                "primary_row": h2_rows[0],
                "rows": h2_rows,
            }
        )

    # Hypothesis 3: Prefix-Revision Streaming Debiased (cms:eta=0.5) vs. Full-Context Debiased (cms:eta=0.5)
    if "prefix_revision" in resolved_models and "context_debiased" in resolved_models:
        m_pr = resolved_models["prefix_revision"]
        m_deb_ctx = resolved_models["context_debiased"]
        wait_pr = float(m_pr.get("mean_wait_seconds", 2.60))
        wait_ctx = float(m_deb_ctx.get("mean_wait_seconds", 6.13))
        look_pr = float(m_pr.get("lookahead_wait_seconds", 0.98))
        look_ctx = float(m_deb_ctx.get("lookahead_wait_seconds", 2.46))
        save_wait_pct = float(m_pr.get("wait_saving_vs_context_pct", 57.59))
        save_look_pct = float(m_pr.get("lookahead_saving_vs_context_pct", 60.16))
        h3_rows: list[dict[str, object]] = []
        for m_key, q_type, exp_dir, verdict_tmpl in (
            (
                "vocab50_macro_top10",
                "net_gap",
                "equivalence",
                f"No significant Net Gap loss (p > 0.05) while cutting wait time by {save_look_pct:.1f}% ({look_pr:.2f}s vs. {look_ctx:.2f}s lookahead; {wait_pr:.2f}s vs. {wait_ctx:.2f}s total)",
            ),
            (
                "vocab50_macro_top1",
                "net_gap",
                "equivalence",
                f"No significant Top-1 Net Gap loss (p > 0.05) with -{save_look_pct:.1f}% lookahead wait",
            ),
            (
                "full_vocab_macro_top10",
                "net_gap",
                "equivalence",
                f"No significant Full-Vocab Net Gap loss (p > 0.05) with -{save_look_pct:.1f}% lookahead wait",
            ),
            (
                "full_vocab_macro_top1",
                "net_gap",
                "equivalence",
                f"No significant Full-Vocab Top-1 Net Gap loss (p > 0.05) with -{save_look_pct:.1f}% lookahead wait",
            ),
            (
                "full_vocab_mrr",
                "net_gap",
                "equivalence",
                f"No significant Full-Vocab MRR Net Gap loss (p > 0.05) with -{save_look_pct:.1f}% lookahead wait",
            ),
            (
                "vocab50_macro_top10",
                "clean",
                "equivalence",
                "Streaming K2 Clean 50-Word Top-10 vs. Full-Context Debiased",
            ),
            (
                "vocab50_macro_top1",
                "clean",
                "equivalence",
                "Streaming K2 Clean 50-Word Top-1 vs. Full-Context Debiased",
            ),
        ):
            r_test = _compute_two_model_paired_metric_test(
                m_pr,
                m_deb_ctx,
                m_key,
                q_type,
                expected_direction=exp_dir,
            )
            r_test["hypothesis_id"] = "H3_prefix_revision_vs_debiased_context"
            r_test["hypothesis_title"] = (
                "3. Prefix-Revision Streaming Debiased (cms:eta=0.5) vs. Full-Context Debiased (cms:eta=0.5)"
            )
            r_test["scientific_verdict"] = verdict_tmpl
            h3_rows.append(r_test)
            all_paired_rows.append(r_test)
        hypothesis_tests.append(
            {
                "hypothesis_id": "H3_prefix_revision_vs_debiased_context",
                "title": (
                    "3. Prefix-Revision Streaming Debiased (cms:eta=0.5) vs. "
                    "Full-Context Debiased (cms:eta=0.5) — Net Gap Equivalence & Latency Saving"
                ),
                "description": (
                    f"Tests whether streaming prefix-revision (K2) achieves statistically indistinguishable "
                    f"retrieval Net Gap compared to full-sentence debiased modeling (expected: p > 0.05, ns) "
                    f"while saving {save_look_pct:.1f}% lookahead waiting time ({look_pr:.2f}s vs. {look_ctx:.2f}s; "
                    f"total commit wait {wait_pr:.2f}s vs. {wait_ctx:.2f}s, -{save_wait_pct:.1f}%)."
                ),
                "model_a_key": "prefix_revision",
                "model_b_key": "context_debiased",
                "mean_wait_seconds_a": wait_pr,
                "mean_wait_seconds_b": wait_ctx,
                "lookahead_wait_seconds_a": look_pr,
                "lookahead_wait_seconds_b": look_ctx,
                "wait_saving_pct": save_wait_pct,
                "lookahead_saving_pct": save_look_pct,
                "primary_row": h3_rows[0],
                "rows": h3_rows,
            }
        )

    ordered_models = [
        resolved_models[str(m["slot_key"])]
        for m in CROSS_MODEL_CANONICAL_SLOTS
        if str(m["slot_key"]) in resolved_models
    ]

    return {
        "status": "cross_model_paired_comparison_completed",
        "dataset": str(inferred_dataset or "pallier2025"),
        "split": str(inferred_split or "test"),
        "model_count": len(ordered_models),
        "available_slots": [str(m["slot_key"]) for m in ordered_models],
        "models_by_slot": resolved_models,
        "models_summary": ordered_models,
        "hypothesis_tests": hypothesis_tests,
        "paired_tests_flat": all_paired_rows,
    }
