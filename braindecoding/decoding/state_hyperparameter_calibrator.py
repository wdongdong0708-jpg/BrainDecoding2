"""验证集状态自适应超参数 (T_r, λ_r) 网格校准器（Module 4）。

核心职责：
1. 弃用全局固定语言权重 λ，在验证集上对每个状态 r ∈ {S0, K0, K1, K2, F} 独立网格搜索
   温度参数 T_r 与语言模型权重 λ_r。
2. 验证并约束神经证据越强则语言权重越小的客观规律：
       λ_{K0} >= λ_{K1} >= λ_{K2}
3. 输出完整网格搜索报表与最优超参配置 JSON（`calibrated_hyperparameters.json`）。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from braindecoding.decoding.language_model_scoring import CachedVocabularyLanguageScorer
from braindecoding.decoding.neural_score_exporter import CANONICAL_STATES
from braindecoding.decoding.rolling_beam_search import (
    compute_neural_log_probabilities,
    decode_segment_static,
)
from braindecoding.evaluation.statistical_significance import sentence_metrics
from braindecoding.training.training_runtime import save_json


DEFAULT_TEMPERATURE_GRID: tuple[float, ...] = (0.05, 0.07, 0.10, 0.15, 0.20)
DEFAULT_LANGUAGE_WEIGHT_GRID: tuple[float, ...] = (
    0.0,
    0.05,
    0.10,
    0.15,
    0.25,
    0.40,
    0.60,
    1.00,
)


def _evaluate_decoded_segments(
    segments: Sequence[dict],
    predictions: Sequence[Sequence[str]],
    *,
    language: str,
    metric_name: str = "composite",
) -> dict[str, float]:
    """计算校准集句段的受试者等权 BLEU-1、词位准确率与综合校准得分。"""
    chinese = language == "zh"
    per_subject_bleu: dict[str, list[float]] = defaultdict(list)
    per_subject_acc: dict[str, list[float]] = defaultdict(list)

    for seg, pred in zip(segments, predictions):
        ref = [str(w) for w in seg["reference_words"]]
        hyp = [str(w) for w in pred]
        if not ref:
            continue
        m = sentence_metrics([ref], [hyp], chinese=chinese)
        bleu1 = float(m["sentence_bleu1_mean"])
        acc = float(sum(a == b for a, b in zip(ref, hyp)) / len(ref))
        subj = str(seg.get("subject_id", "default"))
        per_subject_bleu[subj].append(bleu1)
        per_subject_acc[subj].append(acc)

    mean_bleu1 = (
        float(np.mean([np.mean(vals) for vals in per_subject_bleu.values()]))
        if per_subject_bleu
        else 0.0
    )
    mean_acc = (
        float(np.mean([np.mean(vals) for vals in per_subject_acc.values()]))
        if per_subject_acc
        else 0.0
    )
    composite = 0.5 * mean_bleu1 + 0.5 * mean_acc

    if metric_name == "sentence_bleu1_mean":
        primary = mean_bleu1
    elif metric_name == "aligned_word_accuracy":
        primary = mean_acc
    else:
        primary = composite

    return {
        "selection_score": float(primary),
        "sentence_bleu1_mean": float(mean_bleu1),
        "aligned_word_accuracy": float(mean_acc),
        "composite_score": float(composite),
    }


def calibrate_state_hyperparameters(
    exported_val_payload: dict,
    scorer: CachedVocabularyLanguageScorer,
    *,
    states: Sequence[str] = CANONICAL_STATES,
    temperature_grid: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
    language_weight_grid: Sequence[float] = DEFAULT_LANGUAGE_WEIGHT_GRID,
    beam_width: int = 5,
    candidate_top_k: int = 20,
    metric_name: str = "composite",
    enforce_monotonic_lambda: bool = True,
    use_calibration_partition: bool = True,
    max_calibration_segments: int | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """在验证集上按状态独立网格搜索 (T_r, λ_r) 并检验/约束 λ_{K0} >= λ_{K1} >= λ_{K2}。"""
    if exported_val_payload.get("split") not in {"val", None}:
        raise ValueError(
            f"超参数校准仅允许在验证集 ('val') 上执行，收到 split={exported_val_payload.get('split')!r}。"
        )

    started = time.perf_counter()
    vocabulary = list(exported_val_payload["vocabulary"])
    language = str(exported_val_payload.get("language", "fr"))
    all_segments = list(exported_val_payload["segments"])

    if use_calibration_partition:
        cal_segments = [
            seg for seg in all_segments if seg.get("partition") == "calibration"
        ]
        if not cal_segments:
            cal_segments = all_segments
    else:
        cal_segments = all_segments

    if max_calibration_segments is not None:
        cal_segments = cal_segments[: int(max_calibration_segments)]
    if not cal_segments:
        raise ValueError("验证集校准句段列表为空。")

    temp_list = sorted({float(t) for t in temperature_grid if float(t) > 0.0})
    lam_list = sorted({float(l) for l in language_weight_grid if float(l) >= 0.0})
    if not temp_list or not lam_list:
        raise ValueError("temperature_grid 与 language_weight_grid 不能为空。")

    clean_scores = exported_val_payload["clean_scores"]
    state_grids: dict[str, list[dict]] = {}
    unconstrained_best: dict[str, dict] = {}
    best_by_state_and_lambda: dict[str, dict[float, dict]] = {}

    for state in states:
        if state not in clean_scores:
            raise ValueError(f"导出的验证集得分缺少状态 {state!r}。")
        raw_matrix = clean_scores[state]
        grid_rows: list[dict] = []
        best_for_lam: dict[float, dict] = {}

        # 当 λ_r == 0.0 时，Top-1 神经预测与温度 T_r 无关，只需计算一次
        zero_lam_metrics: dict[str, float] | None = None
        if 0.0 in lam_list:
            lp_any = compute_neural_log_probabilities(raw_matrix, temp_list[0])
            top1_idx = np.argmax(lp_any, axis=-1)
            zero_preds = [
                [vocabulary[int(top1_idx[i])] for i in seg["event_indices"]]
                for seg in cal_segments
            ]
            zero_lam_metrics = _evaluate_decoded_segments(
                cal_segments,
                zero_preds,
                language=language,
                metric_name=metric_name,
            )

        for temp in temp_list:
            lp_matrix = compute_neural_log_probabilities(raw_matrix, temp)
            for lam in lam_list:
                if abs(lam) <= 1e-12 and zero_lam_metrics is not None:
                    eval_res = dict(zero_lam_metrics)
                else:
                    preds: list[list[str]] = []
                    for seg in cal_segments:
                        ev_idx = seg["event_indices"]
                        dec = decode_segment_static(
                            lp_matrix[ev_idx],
                            vocabulary,
                            scorer,
                            language_weight=lam,
                            beam_width=beam_width,
                            candidate_top_k=candidate_top_k,
                        )
                        preds.append(dec["final_words"])
                    eval_res = _evaluate_decoded_segments(
                        cal_segments,
                        preds,
                        language=language,
                        metric_name=metric_name,
                    )

                row = {
                    "state": state,
                    "temperature": float(temp),
                    "language_weight": float(lam),
                    **eval_res,
                }
                grid_rows.append(row)

                prev_best_lam = best_for_lam.get(float(lam))
                if prev_best_lam is None or (
                    row["selection_score"],
                    -row["temperature"],
                ) > (
                    prev_best_lam["selection_score"],
                    -prev_best_lam["temperature"],
                ):
                    best_for_lam[float(lam)] = row

        # 无约束最优：按 selection_score 降序，平局优先选择较小 language_weight 与适中温度
        winner = sorted(
            grid_rows,
            key=lambda item: (
                -round(item["selection_score"], 9),
                item["language_weight"],
                item["temperature"],
            ),
        )[0]
        state_grids[state] = grid_rows
        unconstrained_best[state] = dict(winner)
        best_by_state_and_lambda[state] = best_for_lam

    selected_states: dict[str, dict] = {
        state: dict(unconstrained_best[state]) for state in states
    }

    has_k012 = all(k in selected_states for k in ("K0", "K1", "K2"))
    unconstrained_monotonic = True
    if has_k012:
        unconstrained_monotonic = bool(
            unconstrained_best["K0"]["language_weight"]
            >= unconstrained_best["K1"]["language_weight"]
            >= unconstrained_best["K2"]["language_weight"]
        )

    if has_k012 and enforce_monotonic_lambda:
        # 在满足 λ_{K0} >= λ_{K1} >= λ_{K2} 的所有网格三元组中寻找总校准分最高的组合
        # 平局时优先鼓励严格单调递减 (λ_{K0} > λ_{K1} > λ_{K2}) 以体现神经证据递增规律
        best_triple: tuple[dict, dict, dict] | None = None
        best_key: tuple[float, int, float] | None = None

        for l0 in lam_list:
            r0 = best_by_state_and_lambda["K0"][l0]
            for l1 in lam_list:
                if l1 > l0 + 1e-12:
                    continue
                r1 = best_by_state_and_lambda["K1"][l1]
                for l2 in lam_list:
                    if l2 > l1 + 1e-12:
                        continue
                    r2 = best_by_state_and_lambda["K2"][l2]
                    total_score = (
                        r0["selection_score"]
                        + r1["selection_score"]
                        + r2["selection_score"]
                    )
                    strict_steps = int(l0 > l1) + int(l1 > l2)
                    cand_key = (
                        round(total_score, 8),
                        strict_steps,
                        -(l0 + l1 + l2),
                    )
                    if best_key is None or cand_key > best_key:
                        best_key = cand_key
                        best_triple = (dict(r0), dict(r1), dict(r2))

        if best_triple is not None:
            selected_states["K0"] = best_triple[0]
            selected_states["K1"] = best_triple[1]
            selected_states["K2"] = best_triple[2]

    monotonicity_satisfied = True
    if has_k012:
        monotonicity_satisfied = bool(
            selected_states["K0"]["language_weight"]
            >= selected_states["K1"]["language_weight"]
            >= selected_states["K2"]["language_weight"]
        )

    result = {
        "schema_version": 1,
        "protocol": "sequence_decoding_s3_calibration",
        "dataset": exported_val_payload.get("dataset"),
        "language": language,
        "split": "val",
        "seed": exported_val_payload.get("seed", 0),
        "calibration_segments": len(cal_segments),
        "metric": metric_name,
        "temperature_grid": temp_list,
        "language_weight_grid": lam_list,
        "enforce_monotonic_lambda": bool(enforce_monotonic_lambda),
        "unconstrained_monotonic": unconstrained_monotonic,
        "monotonicity_satisfied": monotonicity_satisfied,
        "states": selected_states,
        "unconstrained_states": unconstrained_best,
        "grids": state_grids,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    if output_path is not None:
        save_json(Path(output_path), result)

    return result
