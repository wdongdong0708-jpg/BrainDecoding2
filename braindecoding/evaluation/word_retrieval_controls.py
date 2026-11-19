"""时间错位 / 零信号负对照协议解析与汇总指标提取。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from braindecoding.config import (
    load_yaml_with_extends,
    normalize_project_relative,
    project_path,
    resolve_config_path,
)
from braindecoding.evaluation.word_retrieval_statistics import (
    DEFAULT_CAUSAL_CMS_EMA_DECAY,
    DEFAULT_CAUSAL_CMS_ETAS,
    DEFAULT_CONTROL_SHIFTS_SECONDS,
    DEFAULT_POS_CAUSAL_CMS_ETAS,
    DEFAULT_POS_CAUSAL_EMA_DECAY,
    DEFAULT_POS_MIN_BUCKET_COUNT,
    DEFAULT_SUBTRACTION_ETAS,
    _compute_paired_block_significance,
    normalize_cms_mode,
    parse_eta_values,
)

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
        if isinstance(tok, str) and tok.strip().lower() in {
            "none",
            "off",
            "empty",
            "null",
        }:
            continue
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
    shift_mode: str | None = None,
    preroll_seconds: float | None = None,
    cms_mode: str | None = None,
    cms_ema_decay: float | None = None,
) -> dict:
    """合并 CLI 参数、独立对照 YAML 配置（configs/controls/...）与主配置中的负对照设置。"""
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
    raw_cms_mode = (
        cms_mode
        if cms_mode is not None
        else (
            cms_raw.get("mode")
            or merged_control.get("cms_mode")
            or merged_eval.get("cms_mode")
        )
    )
    resolved_cms_mode = normalize_cms_mode(
        str(raw_cms_mode) if raw_cms_mode is not None else None,
        default="leave_one_out",
    )
    cms_enabled = (
        subtraction_etas is not None
        or cms_mode is not None
        or bool(cms_raw.get("enabled", False))
        or str(merged_control.get("control_type", "")).strip().lower()
        in {
            "common_mode_subtraction",
            "causal_common_mode_subtraction",
            "position_causal_common_mode_subtraction",
        }
        or str(merged_eval.get("negative_control", "")).strip().lower()
        in {
            "common_mode_subtraction",
            "causal_common_mode_subtraction",
            "position_causal_common_mode_subtraction",
        }
    )
    default_etas = (
        DEFAULT_POS_CAUSAL_CMS_ETAS
        if resolved_cms_mode == "pos_causal_ema"
        else (
            DEFAULT_CAUSAL_CMS_ETAS
            if resolved_cms_mode == "causal_ema"
            else DEFAULT_SUBTRACTION_ETAS
        )
    )
    cms_etas = parse_eta_values(
        subtraction_etas
        if subtraction_etas is not None
        else cms_raw.get("eta_values", default_etas),
        default=default_etas,
    )
    cms_normalize = bool(cms_raw.get("normalize_before_subtraction", True))
    resolved_ema_decay = float(
        cms_ema_decay
        if cms_ema_decay is not None
        else cms_raw.get("ema_decay", DEFAULT_CAUSAL_CMS_EMA_DECAY)
    )
    resolved_pos_ema_decay = float(
        cms_raw.get("pos_ema_decay", DEFAULT_POS_CAUSAL_EMA_DECAY)
    )
    resolved_min_bucket_count = max(
        1, int(cms_raw.get("min_bucket_count", DEFAULT_POS_MIN_BUCKET_COUNT))
    )
    include_global_causal_baseline = bool(
        cms_raw.get("include_global_causal_baseline", True)
    )

    sub_raw = (
        merged_control.get("subspace_coherence_debiasing")
        if isinstance(merged_control.get("subspace_coherence_debiasing"), dict)
        else (
            merged_eval.get("subspace_coherence_debiasing")
            if isinstance(merged_eval.get("subspace_coherence_debiasing"), dict)
            else {}
        )
    )
    sub_enabled = (
        bool(sub_raw.get("enabled", False))
        or str(merged_control.get("control_type", "")).strip().lower()
        == "subspace_coherence_debiasing"
        or str(merged_eval.get("negative_control", "")).strip().lower()
        == "subspace_coherence_debiasing"
    )
    sub_ranks = [
        int(k)
        for k in (sub_raw.get("rank_k_values") or (8, 12, 16, 24))
        if int(k) > 0
    ]
    sub_etas = parse_eta_values(
        subtraction_etas
        if (subtraction_etas is not None and sub_enabled)
        else sub_raw.get("eta_values", (0.8, 1.0, 1.2))
    )
    sub_full_etas = parse_eta_values(
        sub_raw.get("full_loo_eta_values", (0.5, 0.8, 1.0))
    )

    if shift_seconds is not None:
        shifts = _parse_shift_seconds_list(shift_seconds)
    elif "shift_seconds" in merged_control:
        shifts = _parse_shift_seconds_list(merged_control.get("shift_seconds"))
    elif "shift_seconds" in merged_eval:
        shifts = _parse_shift_seconds_list(merged_eval.get("shift_seconds"))
    elif sub_enabled or (
        cms_enabled and resolved_cms_mode not in {"causal_ema", "pos_causal_ema"}
    ):
        shifts = list(DEFAULT_CONTROL_SHIFTS_SECONDS)
    else:
        shifts = []

    raw_shift_mode = (
        shift_mode
        if shift_mode is not None
        else (
            merged_control.get("shift_mode")
            or merged_eval.get("shift_mode")
            or (cfg.get("dataset") or {}).get("shift_mode")
            or (
                "preroll"
                if resolved_cms_mode in {"causal_ema", "pos_causal_ema"}
                else "deterministic_circular"
            )
        )
    )
    resolved_shift_mode = str(raw_shift_mode).strip().lower()
    if resolved_shift_mode in {"pre_roll", "causal_preroll"}:
        resolved_shift_mode = "preroll"
    elif resolved_shift_mode in {"circular", ""}:
        resolved_shift_mode = "deterministic_circular"

    resolved_preroll_seconds = float(
        preroll_seconds
        if preroll_seconds is not None
        else merged_control.get(
            "preroll_seconds",
            merged_eval.get(
                "preroll_seconds",
                (cfg.get("dataset") or {}).get("preroll_seconds", 45.0),
            ),
        )
    )

    enable_zero = (
        bool(zero_signal_control)
        or bool(merged_control.get("include_zero_signal", False))
        or bool(merged_eval.get("zero_signal_control", False))
        or str(merged_control.get("control_type", "")).strip().lower()
        == "zero_signal"
    )
    enabled = bool(shifts) or enable_zero or cms_enabled or sub_enabled
    if not enabled:
        return {
            "enabled": False,
            "name": "none",
            "control_type": "none",
            "config_path": None,
            "shift_seconds": [],
            "shift_mode": resolved_shift_mode,
            "preroll_seconds": resolved_preroll_seconds,
            "include_zero_signal": False,
            "common_mode_subtraction": {
                "enabled": False,
                "mode": resolved_cms_mode,
                "ema_decay": resolved_ema_decay,
                "pos_ema_decay": resolved_pos_ema_decay,
                "min_bucket_count": resolved_min_bucket_count,
                "include_global_causal_baseline": include_global_causal_baseline,
                "eta_values": [],
                "normalize_before_subtraction": True,
            },
            "subspace_coherence_debiasing": {
                "enabled": False,
                "mode": "subspace_coherence_leave_one_out",
                "rank_k_values": [],
                "eta_values": [],
                "full_loo_eta_values": [],
                "center_svd": True,
                "use_coherence_gate": "both",
            },
        }

    default_type = (
        "subspace_coherence_debiasing"
        if sub_enabled
        else (
            "position_causal_common_mode_subtraction"
            if (cms_enabled and resolved_cms_mode == "pos_causal_ema")
            else (
                "causal_common_mode_subtraction"
                if (cms_enabled and resolved_cms_mode == "causal_ema")
                else (
                    "common_mode_subtraction"
                    if cms_enabled
                    else (
                        "temporal_shift"
                        if shifts and not enable_zero
                        else (
                            "zero_signal"
                            if enable_zero and not shifts
                            else "combined_control"
                        )
                    )
                )
            )
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
        "shift_mode": resolved_shift_mode,
        "preroll_seconds": resolved_preroll_seconds,
        "same_subject": bool(merged_control.get("same_subject", True)),
        "same_recording": bool(
            merged_control.get(
                "same_recording", resolved_shift_mode != "preroll"
            )
        ),
        "non_overlapping_window": bool(
            merged_control.get("non_overlapping_window", True)
        ),
        "include_clean_reference": bool(
            merged_control.get("include_clean_reference", True)
        ),
        "include_zero_signal": bool(enable_zero),
        "common_mode_subtraction": {
            "enabled": bool(cms_enabled),
            "mode": resolved_cms_mode,
            "ema_decay": resolved_ema_decay,
            "pos_ema_decay": resolved_pos_ema_decay,
            "min_bucket_count": resolved_min_bucket_count,
            "include_global_causal_baseline": include_global_causal_baseline,
            "eta_values": cms_etas,
            "normalize_before_subtraction": cms_normalize,
        },
        "subspace_coherence_debiasing": {
            "enabled": bool(sub_enabled),
            "mode": str(
                sub_raw.get("mode", "subspace_coherence_leave_one_out")
            ),
            "rank_k_values": sub_ranks,
            "eta_values": sub_etas,
            "full_loo_eta_values": sub_full_etas,
            "center_svd": bool(sub_raw.get("center_svd", True)),
            "use_coherence_gate": sub_raw.get("use_coherence_gate", "both"),
            "coherence_low": float(sub_raw.get("coherence_low", 0.35)),
            "coherence_high": float(sub_raw.get("coherence_high", 0.65)),
            "gate_min": float(sub_raw.get("gate_min", 0.2)),
            "gate_max": float(sub_raw.get("gate_max", 1.2)),
            "normalize_before_subtraction": bool(
                sub_raw.get("normalize_before_subtraction", True)
            ),
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

    fv_word_macro_top10 = float(
        fv.get("word_macro_top10", fv.get("macro_top10", 0.0))
    )
    fv_subj_macro_top10 = float(
        fv.get("subject_macro_top10", fv.get("micro_top10", 0.0))
    )
    row = {
        "key": key,
        "label": label,
        "shift_seconds": shift_seconds,
        "vocab50_macro_top10": float(n50.get("macro_top10", 0.0)),
        "vocab50_micro_top10": float(n50.get("micro_top10", 0.0)),
        "vocab50_macro_top5": float(n50.get("macro_top5", 0.0)),
        "vocab50_macro_top1": float(n50.get("macro_top1", 0.0)),
        "vocab50_micro_top1": float(n50.get("micro_top1", 0.0)),
        "vocab50_mrr": float(n50.get("mrr", 0.0)),
        "vocab50_median_rank": float(n50.get("median_rank", 0.0)),
        "vocab50_chance_top10": float(n50.get("chance_top10", 0.2)),
        "full_vocab_macro_top10": fv_word_macro_top10,
        "full_vocab_word_macro_top10": fv_word_macro_top10,
        "full_vocab_subject_macro_top10": fv_subj_macro_top10,
        "full_vocab_micro_top10": float(fv.get("micro_top10", 0.0)),
        "full_vocab_macro_top1": float(fv.get("macro_top1", 0.0)),
        "full_vocab_micro_top1": float(fv.get("micro_top1", 0.0)),
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
                "common_vocab50_micro_top1": float(
                    horizons[h_key].get("common_vocab50_micro_top1", 0.0)
                ),
                "common_vocab50_mrr": float(
                    horizons[h_key].get("common_vocab50_mrr", 0.0)
                ),
                "common_vocab50_median_rank": float(
                    horizons[h_key].get("common_vocab50_median_rank", 25.0)
                ),
                "common_full_vocab_macro_top10": float(
                    horizons[h_key].get(
                        "common_full_vocab_word_macro_top10",
                        horizons[h_key].get("common_full_vocab_macro_top10", 0.0),
                    )
                ),
                "common_full_vocab_word_macro_top10": float(
                    horizons[h_key].get(
                        "common_full_vocab_word_macro_top10",
                        horizons[h_key].get("common_full_vocab_macro_top10", 0.0),
                    )
                ),
                "common_full_vocab_subject_macro_top10": float(
                    horizons[h_key].get(
                        "common_full_vocab_subject_macro_top10",
                        horizons[h_key].get("common_full_vocab_micro_top10", 0.0),
                    )
                ),
                "common_full_vocab_micro_top10": float(
                    horizons[h_key].get("common_full_vocab_micro_top10", 0.0)
                ),
                "common_full_vocab_macro_top1": float(
                    horizons[h_key].get("common_full_vocab_macro_top1", 0.0)
                ),
                "common_full_vocab_micro_top1": float(
                    horizons[h_key].get("common_full_vocab_micro_top1", 0.0)
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
        c_fv_macro_10 = float(
            clean_summary.get(
                "full_vocab_word_macro_top10",
                clean_summary.get("full_vocab_macro_top10", 0.0),
            )
        )
        c_fv_subj_10 = float(
            clean_summary.get(
                "full_vocab_subject_macro_top10",
                clean_summary.get("full_vocab_micro_top10", 0.0),
            )
        )
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
            "vocab50_micro_top1": float(
                row["vocab50_micro_top1"]
                - clean_summary.get("vocab50_micro_top1", 0.0)
            ),
            "vocab50_mrr": float(
                row["vocab50_mrr"] - clean_summary["vocab50_mrr"]
            ),
            "full_vocab_macro_top10": float(
                row["full_vocab_macro_top10"] - c_fv_macro_10
            ),
            "full_vocab_word_macro_top10": float(
                row["full_vocab_word_macro_top10"] - c_fv_macro_10
            ),
            "full_vocab_subject_macro_top10": float(
                row["full_vocab_subject_macro_top10"] - c_fv_subj_10
            ),
            "full_vocab_micro_top10": float(
                row["full_vocab_micro_top10"]
                - clean_summary.get("full_vocab_micro_top10", 0.0)
            ),
            "full_vocab_macro_top1": float(
                row["full_vocab_macro_top1"]
                - clean_summary.get("full_vocab_macro_top1", 0.0)
            ),
            "full_vocab_micro_top1": float(
                row["full_vocab_micro_top1"]
                - clean_summary.get("full_vocab_micro_top1", 0.0)
            ),
            "full_vocab_mrr": float(
                row["full_vocab_mrr"] - clean_summary.get("full_vocab_mrr", 0.0)
            ),
            "clean_minus_control_vocab50_macro_top10": float(
                clean_summary["vocab50_macro_top10"] - row["vocab50_macro_top10"]
            ),
            "clean_minus_control_vocab50_micro_top10": float(
                clean_summary["vocab50_micro_top10"] - row["vocab50_micro_top10"]
            ),
            "clean_minus_control_vocab50_macro_top1": float(
                clean_summary["vocab50_macro_top1"] - row["vocab50_macro_top1"]
            ),
            "clean_minus_control_vocab50_micro_top1": float(
                clean_summary.get("vocab50_micro_top1", 0.0)
                - row["vocab50_micro_top1"]
            ),
            "clean_minus_control_full_vocab_macro_top10": float(
                c_fv_macro_10 - row["full_vocab_macro_top10"]
            ),
            "clean_minus_control_full_vocab_micro_top10": float(
                clean_summary.get("full_vocab_micro_top10", 0.0)
                - row["full_vocab_micro_top10"]
            ),
            "clean_minus_control_full_vocab_macro_top1": float(
                clean_summary.get("full_vocab_macro_top1", 0.0)
                - row["full_vocab_macro_top1"]
            ),
            "clean_minus_control_full_vocab_micro_top1": float(
                clean_summary.get("full_vocab_micro_top1", 0.0)
                - row["full_vocab_micro_top1"]
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
                    "common_vocab50_micro_top10": float(
                        row["horizons"][h_key]["common_vocab50_micro_top10"]
                        - clean_summary["horizons"][h_key].get(
                            "common_vocab50_micro_top10", 0.0
                        )
                    ),
                    "common_vocab50_macro_top1": float(
                        row["horizons"][h_key]["common_vocab50_macro_top1"]
                        - clean_summary["horizons"][h_key].get(
                            "common_vocab50_macro_top1", 0.0
                        )
                    ),
                    "common_vocab50_micro_top1": float(
                        row["horizons"][h_key]["common_vocab50_micro_top1"]
                        - clean_summary["horizons"][h_key].get(
                            "common_vocab50_micro_top1", 0.0
                        )
                    ),
                    "common_full_vocab_macro_top10": float(
                        row["horizons"][h_key]["common_full_vocab_macro_top10"]
                        - clean_summary["horizons"][h_key].get(
                            "common_full_vocab_word_macro_top10",
                            clean_summary["horizons"][h_key].get(
                                "common_full_vocab_macro_top10", 0.0
                            ),
                        )
                    ),
                    "common_full_vocab_micro_top10": float(
                        row["horizons"][h_key]["common_full_vocab_micro_top10"]
                        - clean_summary["horizons"][h_key].get(
                            "common_full_vocab_micro_top10", 0.0
                        )
                    ),
                    "common_full_vocab_macro_top1": float(
                        row["horizons"][h_key]["common_full_vocab_macro_top1"]
                        - clean_summary["horizons"][h_key].get(
                            "common_full_vocab_macro_top1", 0.0
                        )
                    ),
                    "common_full_vocab_micro_top1": float(
                        row["horizons"][h_key]["common_full_vocab_micro_top1"]
                        - clean_summary["horizons"][h_key].get(
                            "common_full_vocab_micro_top1", 0.0
                        )
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
    agg: dict = {
        "shift_count": len(shifted_rows),
        "shift_seconds": [r["shift_seconds"] for r in shifted_rows],
    }
    for m_key in metric_keys:
        vals = np.asarray(
            [float(r.get(m_key, 0.0)) for r in shifted_rows], dtype=np.float64
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
        h_agg: dict = {}
        for h_key in ("K0", "K1", "K2"):
            if h_key not in clean_summary["horizons"]:
                continue
            c_hz_entry = clean_summary["horizons"][h_key]
            h_item: dict[str, float] = {}
            for hm_key in horizon_metric_keys:
                hm_vals = np.asarray(
                    [
                        float(r["horizons"][h_key].get(hm_key, 0.0))
                        for r in shifted_rows
                        if h_key in r["horizons"]
                    ],
                    dtype=np.float64,
                )
                if len(hm_vals) == 0:
                    continue
                m_val = float(np.mean(hm_vals))
                s_val = float(np.std(hm_vals, ddof=0))
                c_val = float(c_hz_entry.get(hm_key, 0.0))
                h_item[f"mean_{hm_key}"] = m_val
                h_item[f"std_{hm_key}"] = s_val
                h_item[f"delta_{hm_key}_vs_clean"] = float(m_val - c_val)
                h_item[f"clean_minus_shifted_{hm_key}"] = float(c_val - m_val)
            if h_item:
                h_agg[h_key] = h_item
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
