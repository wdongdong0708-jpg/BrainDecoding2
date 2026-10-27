"""留一法共模时序偏置扣除（Leave-One-Out Common-Mode Subtraction）探索性评估模块。

本模块完全独立于主训练与标准评估代码，用于在不重新训练模型、不增加额外 GPU 前向推理的前提下，
对每个测试词位 i 的 1 个 Clean (0s) 预测与 M 个纯历史因果时间错位 Shifted 预测（如 -15s, -30s, -45s, -60s, -90s, -120s，M=6）
执行无泄漏的留一法共模扣除（Leave-One-Out Common-Mode Subtraction）：

- Clean (0s)：使用全部 M 个错位条件的均值作为共模时序偏置扣除
    s_tilde_i^(0) = s_i^(0) - eta * (1 / M) * sum_{m=1..M} s_i^(delta_m)
- Shifted (delta_k)：为严格防止自身噪声泄漏（保持负对照公平性），仅使用其余 M-1 个错位条件的均值扣除
    s_tilde_i^(delta_k) = s_i^(delta_k) - eta * (1 / (M - 1)) * sum_{m != k} s_i^(delta_m)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_SUBTRACTION_ETAS: tuple[float, ...] = (0.5, 0.8, 1.0)


def parse_eta_values(
    raw_etas: str | Iterable[float | int | str] | None,
    *,
    default: Iterable[float] = DEFAULT_SUBTRACTION_ETAS,
) -> list[float]:
    """解析共模扣除权重 eta 列表（自动排除 0.0 基线并按升序去重排列，0.0 由评估器自动作为首项参考）。"""
    if raw_etas is None:
        tokens: list[float | int | str] = list(default)
    elif isinstance(raw_etas, (int, float)):
        tokens = [float(raw_etas)]
    elif isinstance(raw_etas, str):
        text = raw_etas.strip()
        if not text:
            tokens = list(default)
        else:
            tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    else:
        tokens = []
        for item in raw_etas:
            if isinstance(item, str):
                tokens.extend(t.strip() for t in item.split(",") if t.strip())
            else:
                tokens.append(item)

    parsed: list[float] = []
    seen: set[float] = set()
    for tok in tokens:
        val = float(tok)
        if val < 0.0:
            raise ValueError(f"共模扣除权重 eta 必须为非负数，收到：{val}")
        if abs(val) <= 1e-9:
            continue
        rounded = round(val, 6)
        if rounded not in seen:
            seen.add(rounded)
            parsed.append(val)
    return parsed if parsed else [float(x) for x in default]


def _l2_normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """对二维预测张量逐行做 L2 归一化（使内积严格等于余弦相似度得分）。"""
    t = tensor.detach().clone().to(dtype=torch.float32)
    return F.normalize(t, p=2.0, dim=-1, eps=1e-12)


def apply_leave_one_out_subtraction_to_tensors(
    clean_tensor: torch.Tensor,
    shifted_tensors: dict[str, torch.Tensor],
    eta: float,
    *,
    normalize_inputs: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """对单层预测张量执行留一法共模时序偏置扣除。

    参数
    ----
    clean_tensor:
        Clean (0s) 预测张量，形状 ``(N, D)``。
    shifted_tensors:
        有序字典 ``{cond_key: tensor}``，每个张量形状 ``(N, D)``，对应 M 个错位条件（要求 M >= 2）。
    eta:
        共模扣除权重（如 0.5, 0.8, 1.0）。
    normalize_inputs:
        扣除前是否先对各条件向量做单位 L2 归一化（默认 True，保证在单位超球面/余弦得分空间线性扣除）。
    """
    shift_keys = list(shifted_tensors.keys())
    num_shifts = len(shift_keys)
    if num_shifts < 2:
        raise ValueError(
            f"留一法共模扣除（Leave-One-Out Common-Mode Subtraction）至少需要 2 个错位条件，"
            f"当前仅有 {num_shifts} 个（推荐 6 个：-15s, -30s, -45s, -60s, -90s, -120s）。"
        )

    clean_base = (
        _l2_normalize_tensor(clean_tensor)
        if normalize_inputs
        else clean_tensor.detach().clone().to(dtype=torch.float32)
    )
    shifted_base: dict[str, torch.Tensor] = {}
    for k in shift_keys:
        st = shifted_tensors[k]
        if st.shape != clean_base.shape:
            raise ValueError(
                f"错位条件 {k!r} 张量形状 {tuple(st.shape)} 与 Clean 张量形状 {tuple(clean_base.shape)} 不一致。"
            )
        shifted_base[k] = (
            _l2_normalize_tensor(st)
            if normalize_inputs
            else st.detach().clone().to(dtype=torch.float32)
        )

    eta_val = float(eta)
    if abs(eta_val) <= 1e-12:
        return clean_base, shifted_base

    # 计算全部 M 个错位条件的总和向量 S_all = sum_{m=1..M} s^(delta_m)
    stacked_shifts = torch.stack([shifted_base[k] for k in shift_keys], dim=0)
    sum_all_shifts = stacked_shifts.sum(dim=0)

    # 1. Clean (0s): 减去全部 M 个错位条件的均值 (1 / M) * S_all
    mean_all_shifts = sum_all_shifts / float(num_shifts)
    debiased_clean = clean_base - eta_val * mean_all_shifts

    # 2. Shifted (delta_k): 留一法，仅减去其余 (M - 1) 个错位条件的均值 (1 / (M - 1)) * (S_all - s^(delta_k))
    denom_loo = float(num_shifts - 1)
    debiased_shifted: dict[str, torch.Tensor] = {}
    for k in shift_keys:
        cur = shifted_base[k]
        mean_other_shifts = (sum_all_shifts - cur) / denom_loo
        debiased_shifted[k] = cur - eta_val * mean_other_shifts

    return debiased_clean, debiased_shifted


def apply_leave_one_out_subtraction_to_encoded(
    clean_encoded: dict,
    shifted_encoded_map: dict[str, dict],
    eta: float,
    *,
    normalize_inputs: bool = True,
) -> tuple[dict, dict[str, dict]]:
    """对 encode_loader 输出的完整字典（含阶段 2.2 的 K0/K1/K2 修订预测）执行留一法共模扣除。"""
    shift_keys = list(shifted_encoded_map.keys())
    clean_preds = clean_encoded["predictions"]
    shifted_preds = {k: shifted_encoded_map[k]["predictions"] for k in shift_keys}

    deb_clean_preds, deb_shifted_preds = apply_leave_one_out_subtraction_to_tensors(
        clean_preds,
        shifted_preds,
        eta,
        normalize_inputs=normalize_inputs,
    )

    new_clean = dict(clean_encoded)
    new_clean["predictions"] = deb_clean_preds

    new_shifted: dict[str, dict] = {}
    for k in shift_keys:
        copied = dict(shifted_encoded_map[k])
        copied["predictions"] = deb_shifted_preds[k]
        new_shifted[k] = copied

    if "revision_predictions" in clean_encoded and isinstance(
        clean_encoded["revision_predictions"], dict
    ):
        clean_rev = dict(clean_encoded["revision_predictions"])
        new_clean_rev: dict = {}
        new_shifted_rev: dict[str, dict] = {k: {} for k in shift_keys}

        for delay, clean_delay_tensor in sorted(clean_rev.items()):
            shifted_delay_tensors = {
                k: shifted_encoded_map[k]["revision_predictions"][delay]
                for k in shift_keys
            }
            deb_c_delay, deb_s_delay = apply_leave_one_out_subtraction_to_tensors(
                clean_delay_tensor,
                shifted_delay_tensors,
                eta,
                normalize_inputs=normalize_inputs,
            )
            new_clean_rev[delay] = deb_c_delay
            for k in shift_keys:
                new_shifted_rev[k][delay] = deb_s_delay[k]

        new_clean["revision_predictions"] = new_clean_rev
        for k in shift_keys:
            new_shifted[k]["revision_predictions"] = new_shifted_rev[k]

    return new_clean, new_shifted


def _build_eta_entry(
    eta: float,
    clean_summary: dict,
    condition_entries: list[dict],
    shifted_aggregate: dict,
    *,
    baseline_entry: dict | None = None,
) -> dict:
    """构建单个 eta 权重下的完整汇总字典（含相对 eta=0 的增益差值）。"""
    num_shifts = len(condition_entries)
    entry = {
        "eta": float(eta),
        "label": "Raw Baseline (eta=0.0)" if abs(eta) <= 1e-9 else f"LOO Subtracted (eta={eta:g})",
        "clean_divisor": int(num_shifts),
        "shifted_loo_divisor": int(max(1, num_shifts - 1)),
        "clean_reference": clean_summary,
        "conditions": condition_entries,
        "shifted_aggregate": shifted_aggregate,
    }
    if baseline_entry is not None:
        base_clean = baseline_entry["clean_reference"]
        base_agg = baseline_entry["shifted_aggregate"]
        delta_clean = {
            "vocab50_macro_top10": float(
                clean_summary.get("vocab50_macro_top10", 0.0)
                - base_clean.get("vocab50_macro_top10", 0.0)
            ),
            "vocab50_micro_top10": float(
                clean_summary.get("vocab50_micro_top10", 0.0)
                - base_clean.get("vocab50_micro_top10", 0.0)
            ),
            "vocab50_macro_top1": float(
                clean_summary.get("vocab50_macro_top1", 0.0)
                - base_clean.get("vocab50_macro_top1", 0.0)
            ),
            "vocab50_mrr": float(
                clean_summary.get("vocab50_mrr", 0.0)
                - base_clean.get("vocab50_mrr", 0.0)
            ),
            "full_vocab_macro_top10": float(
                clean_summary.get("full_vocab_macro_top10", 0.0)
                - base_clean.get("full_vocab_macro_top10", 0.0)
            ),
            "full_vocab_micro_top10": float(
                clean_summary.get("full_vocab_micro_top10", 0.0)
                - base_clean.get("full_vocab_micro_top10", 0.0)
            ),
            "full_vocab_macro_top1": float(
                clean_summary.get("full_vocab_macro_top1", 0.0)
                - base_clean.get("full_vocab_macro_top1", 0.0)
            ),
            "full_vocab_mrr": float(
                clean_summary.get("full_vocab_mrr", 0.0)
                - base_clean.get("full_vocab_mrr", 0.0)
            ),
            "full_vocab_median_rank": float(
                clean_summary.get("full_vocab_median_rank", 0.0)
                - base_clean.get("full_vocab_median_rank", 0.0)
            ),
            "clean_minus_shifted_vocab50_macro_top10": float(
                shifted_aggregate.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
                - base_agg.get("clean_minus_shifted_vocab50_macro_top10", 0.0)
            ),
            "shifted_mean_vocab50_macro_top1": float(
                shifted_aggregate.get("mean_vocab50_macro_top1", 0.0)
                - base_agg.get("mean_vocab50_macro_top1", 0.0)
            ),
            "clean_minus_shifted_vocab50_macro_top1": float(
                shifted_aggregate.get("clean_minus_shifted_vocab50_macro_top1", 0.0)
                - base_agg.get("clean_minus_shifted_vocab50_macro_top1", 0.0)
            ),
            "clean_minus_shifted_full_vocab_macro_top10": float(
                shifted_aggregate.get("clean_minus_shifted_full_vocab_macro_top10", 0.0)
                - base_agg.get("clean_minus_shifted_full_vocab_macro_top10", 0.0)
            ),
            "shifted_mean_full_vocab_macro_top1": float(
                shifted_aggregate.get("mean_full_vocab_macro_top1", 0.0)
                - base_agg.get("mean_full_vocab_macro_top1", 0.0)
            ),
            "clean_minus_shifted_full_vocab_macro_top1": float(
                shifted_aggregate.get("clean_minus_shifted_full_vocab_macro_top1", 0.0)
                - base_agg.get("clean_minus_shifted_full_vocab_macro_top1", 0.0)
            ),
        }
        entry["delta_vs_eta0"] = delta_clean
    return entry


def run_common_mode_subtraction_sweep(
    *,
    dataset: str,
    stage: str,
    split: str,
    seed: int,
    exp_tag: str,
    word_exp_tag: str,
    clean_encoded: dict,
    shifted_encoded_by_shift: list[tuple[float, str, str, dict]],
    eval_ds,
    shifted_ds_by_shift: dict[float, object],
    vocabulary: list[str],
    vocab50: list[str],
    cfg: dict,
    eta_values: Iterable[float] = DEFAULT_SUBTRACTION_ETAS,
    normalize_inputs: bool = True,
    baseline_clean_summary: dict | None = None,
    baseline_condition_entries: list[dict] | None = None,
    baseline_shifted_aggregate: dict | None = None,
    compute_metrics_fn: Callable[..., dict],
    extract_summary_row_fn: Callable[..., dict],
    summarize_shifted_fn: Callable[[dict, list[dict]], dict],
    output_dir: str | Path | None = None,
    plot: bool = True,
    print_report: bool = True,
    verbose: bool = False,
) -> dict:
    """在单次编码缓存上执行留一法共模扣除多权重 (eta) 扫描、终端打印、独立落盘与绘图。"""
    etas = parse_eta_values(eta_values)
    shift_meta = [
        (float(sec), str(key), str(label))
        for sec, key, label, _ in shifted_encoded_by_shift
    ]
    shifted_encoded_map = {
        str(key): enc for _, key, _, enc in shifted_encoded_by_shift
    }
    shift_sec_by_key = {str(key): float(sec) for sec, key, _, _ in shifted_encoded_by_shift}
    shift_label_by_key = {str(key): str(label) for _, key, label, _ in shifted_encoded_by_shift}

    if len(shift_meta) < 2:
        raise ValueError(
            f"留一法共模扣除至少需要 2 个时间错位条件，当前只有 {len(shift_meta)} 个。"
        )

    # 1. 基线 eta = 0.0（直接复用已算好的 Clean 与 Shifted 结果，避免重复计算）
    if (
        baseline_clean_summary is not None
        and baseline_condition_entries is not None
        and baseline_shifted_aggregate is not None
    ):
        eta0_clean = baseline_clean_summary
        eta0_conditions = [
            {
                "key": c["key"],
                "label": c["label"],
                "shift_seconds": c["shift_seconds"],
                "summary": c["summary"],
            }
            for c in baseline_condition_entries
            if c.get("control_type") == "temporal_shift"
        ]
        eta0_agg = baseline_shifted_aggregate
    else:
        raw_clean_metrics = compute_metrics_fn(
            clean_encoded,
            eval_ds,
            vocabulary,
            vocab50,
            cfg,
            stage=stage,
            split=split,
        )
        eta0_clean = extract_summary_row_fn(
            "clean_0s", "Clean (0s)", 0.0, raw_clean_metrics
        )
        eta0_shifted_rows: list[dict] = []
        eta0_conditions = []
        for sec, key, label in shift_meta:
            s_ds = shifted_ds_by_shift.get(sec, eval_ds)
            s_metrics = compute_metrics_fn(
                shifted_encoded_map[key],
                s_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            s_row = extract_summary_row_fn(
                key, label, sec, s_metrics, clean_summary=eta0_clean
            )
            eta0_shifted_rows.append(s_row)
            eta0_conditions.append(
                {
                    "key": key,
                    "label": label,
                    "shift_seconds": sec,
                    "summary": s_row,
                }
            )
        eta0_agg = summarize_shifted_fn(eta0_clean, eta0_shifted_rows)

    eta0_entry = _build_eta_entry(
        0.0, eta0_clean, eta0_conditions, eta0_agg, baseline_entry=None
    )
    sweep_results: list[dict] = [eta0_entry]

    # 2. 扫描每个 eta > 0（纯内存向量运算 + 排名计算，无需重跑神经网络）
    for eta in etas:
        deb_clean_enc, deb_shifted_enc_map = apply_leave_one_out_subtraction_to_encoded(
            clean_encoded,
            shifted_encoded_map,
            eta,
            normalize_inputs=normalize_inputs,
        )
        deb_clean_metrics = compute_metrics_fn(
            deb_clean_enc,
            eval_ds,
            vocabulary,
            vocab50,
            cfg,
            stage=stage,
            split=split,
        )
        deb_clean_row = extract_summary_row_fn(
            "clean_0s", "Clean (0s)", 0.0, deb_clean_metrics
        )

        deb_shifted_rows: list[dict] = []
        deb_conditions: list[dict] = []
        for key in shifted_encoded_map:
            sec = shift_sec_by_key[key]
            label = shift_label_by_key[key]
            s_ds = shifted_ds_by_shift.get(sec, eval_ds)
            deb_s_metrics = compute_metrics_fn(
                deb_shifted_enc_map[key],
                s_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            s_row = extract_summary_row_fn(
                key, label, sec, deb_s_metrics, clean_summary=deb_clean_row
            )
            deb_shifted_rows.append(s_row)
            deb_conditions.append(
                {
                    "key": key,
                    "label": label,
                    "shift_seconds": sec,
                    "summary": s_row,
                }
            )

        deb_agg = summarize_shifted_fn(deb_clean_row, deb_shifted_rows)
        eta_entry = _build_eta_entry(
            eta,
            deb_clean_row,
            deb_conditions,
            deb_agg,
            baseline_entry=eta0_entry,
        )
        sweep_results.append(eta_entry)

    # 选出 50 词库 Clean Macro Top-10 最优与 Clean-Shifted 净差值最优的 eta
    best_by_clean50 = max(
        sweep_results,
        key=lambda item: float(item["clean_reference"].get("vocab50_macro_top10", 0.0)),
    )
    best_by_gap50 = max(
        sweep_results,
        key=lambda item: float(
            item["shifted_aggregate"].get("clean_minus_shifted_vocab50_macro_top10", 0.0)
        ),
    )

    report_payload = {
        "schema_version": 1,
        "experiment_type": "exploratory_leave_one_out_common_mode_subtraction",
        "dataset": dataset,
        "stage": stage,
        "split": split,
        "seed": int(seed),
        "exp_tag": exp_tag,
        "word_exp_tag": word_exp_tag,
        "formula": {
            "clean": "s_tilde_i^(0) = s_i^(0) - eta * (1 / M) * sum_{delta != 0} s_i^(delta)",
            "shifted_leave_one_out": "s_tilde_i^(delta_k) = s_i^(delta_k) - eta * (1 / (M - 1)) * sum_{delta not in {0, delta_k}} s_i^(delta)",
            "num_shifted_conditions_M": len(shift_meta),
            "clean_divisor": len(shift_meta),
            "shifted_loo_divisor": len(shift_meta) - 1,
            "shift_seconds": [sec for sec, _, _ in shift_meta],
            "normalize_before_subtraction": bool(normalize_inputs),
        },
        "eta_grid": [0.0, *etas],
        "best_eta_by_clean_vocab50_top10": float(best_by_clean50["eta"]),
        "best_eta_by_clean_minus_shifted_vocab50_top10": float(best_by_gap50["eta"]),
        "sweep": sweep_results,
        "figures": {},
    }

    if print_report:
        print_common_mode_subtraction_report(
            report_payload,
            verbose=verbose,
            output_dir=output_dir,
        )

    if output_dir is not None:
        from braindecoding.evaluation.evaluation_exports import (
            export_common_mode_subtraction_tables_and_markdown,
        )

        out_path = Path(output_dir)
        exploratory_dir = out_path / "exploratory"
        exploratory_dir.mkdir(parents=True, exist_ok=True)
        report_payload["exported_files"] = (
            export_common_mode_subtraction_tables_and_markdown(
                report_payload, out_path
            )
        )
        json_file = exploratory_dir / f"common_mode_subtraction_{split}.json"
        json_file.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_payload["exploratory_json"] = str(json_file)
        if plot:
            figs = plot_common_mode_subtraction_figures(
                report_payload, out_path / "figures"
            )
            report_payload["figures"].update(figs)
            json_file.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return report_payload


from braindecoding.evaluation.evaluation_figures import (
    plot_common_mode_subtraction_figures,
)
from braindecoding.evaluation.evaluation_reporting import (
    format_common_mode_subtraction_report,
    print_common_mode_subtraction_report,
)

