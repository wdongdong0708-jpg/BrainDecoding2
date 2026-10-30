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
    decode_segments_static_batched,
)
from braindecoding.evaluation.statistical_significance import sentence_metrics
from braindecoding.training.training_runtime import save_json


DEFAULT_TEMPERATURE_GRID: tuple[float, ...] = (0.05, 0.07)
DEFAULT_MIN_LANGUAGE_WEIGHT: float = 0.05
DEFAULT_LANGUAGE_WEIGHT_GRID: tuple[float, ...] = (
    0.05,
    0.10,
    0.15,
    0.25,
    0.40,
    0.60,
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


def _stratified_select_segments(
    segments: Sequence[dict],
    max_count: int,
) -> list[dict]:
    """按受试者轮询（Round-Robin）均匀抽样校准句段，保证每个受试者在子集中等权代表。"""
    limit = max(1, int(max_count))
    if len(segments) <= limit:
        return list(segments)
    by_subj: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        by_subj[str(seg.get("subject_id", "default"))].append(seg)
    subj_keys = sorted(by_subj.keys())
    selected: list[dict] = []
    round_idx = 0
    while len(selected) < limit:
        added_in_round = False
        for sk in subj_keys:
            bucket = by_subj[sk]
            if round_idx < len(bucket):
                selected.append(bucket[round_idx])
                added_in_round = True
                if len(selected) >= limit:
                    break
        if not added_in_round:
            break
        round_idx += 1
    return selected


def calibrate_state_hyperparameters(
    exported_val_payload: dict,
    scorer: CachedVocabularyLanguageScorer,
    *,
    states: Sequence[str] = CANONICAL_STATES,
    temperature_grid: Sequence[float] = DEFAULT_TEMPERATURE_GRID,
    language_weight_grid: Sequence[float] = DEFAULT_LANGUAGE_WEIGHT_GRID,
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_batch_size: int = 32,
    metric_name: str = "composite",
    enforce_monotonic_lambda: bool = True,
    shared_language_weight: bool = False,
    min_language_weight: float = DEFAULT_MIN_LANGUAGE_WEIGHT,
    use_calibration_partition: bool = True,
    max_calibration_segments: int | None = None,
    output_path: str | Path | None = None,
    show_progress: bool = False,
) -> dict:
    """在验证集上按状态网格搜索 (T_r, λ_r) 并强制约束 λ_{K0} >= λ_{K1} >= λ_{K2} >= λ_min > 0。"""
    from tqdm.auto import tqdm

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
        cal_segments = _stratified_select_segments(
            cal_segments, int(max_calibration_segments)
        )
    if not cal_segments:
        raise ValueError("验证集校准句段列表为空。")

    temp_list = sorted({float(t) for t in temperature_grid if float(t) > 0.0})
    raw_lam_list = sorted({float(l) for l in language_weight_grid if float(l) >= 0.0})
    if not temp_list or not raw_lam_list:
        raise ValueError("temperature_grid 与 language_weight_grid 不能为空。")

    min_lam = max(0.0, float(min_language_weight))
    positive_lam_list = [l for l in raw_lam_list if l >= min_lam - 1e-12]
    lam_list = positive_lam_list if positive_lam_list else raw_lam_list

    clean_scores = exported_val_payload["clean_scores"]
    state_grids: dict[str, list[dict]] = {}
    unconstrained_best: dict[str, dict] = {}
    best_by_state_and_lambda: dict[str, dict[float, dict]] = {}
    by_state_temp_lam: dict[str, dict[tuple[float, float], dict]] = {}
    batch_sz = max(1, int(segment_batch_size))

    total_grid_evals = len(states) * len(temp_list) * len(lam_list)

    # ------------------------------------------------------------------
    # 阶段 1：规划全部 (state, T_r, λ_r) 网格点（保留 α=T·λ 去重与 F≡K2 复用语义）
    # ------------------------------------------------------------------
    copy_from_k2_states: set[str] = set()
    planned_rows: dict[str, list[tuple[float, float, float]]] = {}
    decode_configs: list[tuple[str, float, float, float]] = []
    lp_by_state_temp: dict[tuple[str, float], np.ndarray] = {}
    zero_metrics_by_state: dict[str, dict[str, float]] = {}
    seen_states: list[str] = []

    for state in states:
        if state not in clean_scores:
            raise ValueError(f"导出的验证集得分缺少状态 {state!r}。")
        raw_matrix = clean_scores[state]
        if (
            state == "F"
            and "K2" in seen_states
            and np.allclose(
                np.asarray(raw_matrix, dtype=np.float32),
                np.asarray(clean_scores["K2"], dtype=np.float32),
                atol=1e-6,
            )
        ):
            copy_from_k2_states.add(state)
            seen_states.append(state)
            continue
        seen_states.append(state)

        alpha_seen: set[float] = set()
        # 当 λ_r == 0.0 时，Top-1 神经预测与温度 T_r 无关，只需计算一次
        if 0.0 in lam_list:
            lp_any = compute_neural_log_probabilities(
                raw_matrix, temp_list[0], top_k=candidate_top_k
            )
            top1_idx = np.argmax(lp_any, axis=-1)
            zero_preds = [
                [vocabulary[int(top1_idx[i])] for i in seg["event_indices"]]
                for seg in cal_segments
            ]
            zero_metrics_by_state[state] = _evaluate_decoded_segments(
                cal_segments,
                zero_preds,
                language=language,
                metric_name=metric_name,
            )
            alpha_seen.add(0.0)

        rows_plan: list[tuple[float, float, float]] = []
        for temp in temp_list:
            lp_by_state_temp[(state, float(temp))] = compute_neural_log_probabilities(
                raw_matrix, temp, top_k=candidate_top_k
            )
            for lam in lam_list:
                alpha_key = round(float(temp) * float(lam), 9)
                rows_plan.append((float(temp), float(lam), alpha_key))
                if alpha_key not in alpha_seen:
                    alpha_seen.add(alpha_key)
                    decode_configs.append(
                        (state, float(temp), float(lam), alpha_key)
                    )
        planned_rows[state] = rows_plan

    # ------------------------------------------------------------------
    # 阶段 2：将全部待解码网格点 × 校准句段合并为跨状态、逐句段 λ 的同步批量束搜索
    #         （同一句段的不同网格点相邻排入同一批次，共享语言模型前缀 KV-Cache）
    # ------------------------------------------------------------------
    decode_items: list[tuple[int, int]] = [
        (cfg_i, seg_i)
        for seg_i in range(len(cal_segments))
        for cfg_i in range(len(decode_configs))
    ]
    preds_by_cfg: list[list[list[str] | None]] = [
        [None] * len(cal_segments) for _ in decode_configs
    ]
    pbar = (
        tqdm(
            total=len(decode_items),
            desc=(
                f"S3 Calibrate [{len(cal_segments)} segs × "
                f"{len(decode_configs)}/{total_grid_evals} grid]"
            ),
            unit="seg",
            leave=True,
        )
        if show_progress and decode_items
        else None
    )
    try:
        for start_i in range(0, len(decode_items), batch_sz):
            chunk = decode_items[start_i : start_i + batch_sz]
            batch_mats = []
            batch_lams = []
            for cfg_i, seg_i in chunk:
                state, temp, lam, _ = decode_configs[cfg_i]
                lp_matrix = lp_by_state_temp[(state, temp)]
                batch_mats.append(lp_matrix[cal_segments[seg_i]["event_indices"]])
                batch_lams.append(lam)
            batch_decs = decode_segments_static_batched(
                batch_mats,
                vocabulary,
                scorer,
                language_weight=batch_lams,
                beam_width=beam_width,
                candidate_top_k=candidate_top_k,
            )
            for (cfg_i, seg_i), dec in zip(chunk, batch_decs):
                preds_by_cfg[cfg_i][seg_i] = list(dec["final_words"])
            if pbar is not None:
                pbar.update(len(chunk))
    finally:
        if pbar is not None:
            pbar.close()

    eval_by_state_alpha: dict[tuple[str, float], dict[str, float]] = {}
    for cfg_i, (state, _, _, alpha_key) in enumerate(decode_configs):
        eval_by_state_alpha[(state, alpha_key)] = _evaluate_decoded_segments(
            cal_segments,
            [p if p is not None else [] for p in preds_by_cfg[cfg_i]],
            language=language,
            metric_name=metric_name,
        )
    for state, zero_metrics in zero_metrics_by_state.items():
        eval_by_state_alpha[(state, 0.0)] = zero_metrics

    # ------------------------------------------------------------------
    # 阶段 3：按原网格顺序回填结果行与各状态最优（选择规则与逐点串行实现完全一致）
    # ------------------------------------------------------------------
    for state in states:
        if state in copy_from_k2_states:
            state_grids["F"] = [{**row, "state": "F"} for row in state_grids["K2"]]
            unconstrained_best["F"] = {**unconstrained_best["K2"], "state": "F"}
            best_by_state_and_lambda["F"] = {
                lam_k: {**row, "state": "F"}
                for lam_k, row in best_by_state_and_lambda["K2"].items()
            }
            by_state_temp_lam["F"] = {
                tl_k: {**row, "state": "F"}
                for tl_k, row in by_state_temp_lam["K2"].items()
            }
            continue

        grid_rows: list[dict] = []
        best_for_lam: dict[float, dict] = {}
        temp_lam_map: dict[tuple[float, float], dict] = {}
        for temp, lam, alpha_key in planned_rows[state]:
            eval_res = dict(eval_by_state_alpha[(state, alpha_key)])
            row = {
                "state": state,
                "temperature": float(temp),
                "language_weight": float(lam),
                **eval_res,
            }
            grid_rows.append(row)
            temp_lam_map[(float(temp), float(lam))] = row
            prev_best_lam = best_for_lam.get(float(lam))
            if prev_best_lam is None or (
                row["selection_score"],
                -row["temperature"],
            ) > (
                prev_best_lam["selection_score"],
                -prev_best_lam["temperature"],
            ):
                best_for_lam[float(lam)] = row

        # 无约束最优：按 selection_score 降序，平局优先选择较小 language_weight 与较小锐利温度
        winner = sorted(
            grid_rows,
            key=lambda item: (
                -round(item["selection_score"], 9),
                item["temperature"],
                item["language_weight"],
            ),
        )[0]
        state_grids[state] = grid_rows
        unconstrained_best[state] = dict(winner)
        best_by_state_and_lambda[state] = best_for_lam
        by_state_temp_lam[state] = temp_lam_map

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
            and unconstrained_best["K0"]["temperature"]
            == unconstrained_best["K1"]["temperature"]
            == unconstrained_best["K2"]["temperature"]
        )

    if shared_language_weight and has_k012:
        # 全状态锁定单一共享 (T*, λ*)：严格控制解码器超参，使各档位差异 100% 源自输入神经表征
        best_shared_pair: tuple[float, float] | None = None
        best_shared_key: tuple[float, float, float, float] | None = None
        for shared_temp in temp_list:
            for shared_lam in lam_list:
                r0 = by_state_temp_lam["K0"][(shared_temp, shared_lam)]
                r1 = by_state_temp_lam["K1"][(shared_temp, shared_lam)]
                r2 = by_state_temp_lam["K2"][(shared_temp, shared_lam)]
                total_score = (
                    r0["selection_score"]
                    + r1["selection_score"]
                    + r2["selection_score"]
                )
                cand_key = (
                    round(total_score, 8),
                    round(r2["selection_score"], 8),
                    -shared_temp,
                    -shared_lam,
                )
                if best_shared_key is None or cand_key > best_shared_key:
                    best_shared_key = cand_key
                    best_shared_pair = (shared_temp, shared_lam)
        if best_shared_pair is not None:
            for st in states:
                if best_shared_pair in by_state_temp_lam.get(st, {}):
                    selected_states[st] = dict(
                        by_state_temp_lam[st][best_shared_pair]
                    )
    elif has_k012 and enforce_monotonic_lambda:
        # 锁定同一锐利温度 T_0 = T_1 = T_2 = T，并在满足 λ_{K0} >= λ_{K1} >= λ_{K2} >= λ_min > 0 的所有网格三元组中寻找总校准分最高的组合
        best_triple: tuple[dict, dict, dict] | None = None
        best_key: tuple[float, float, float, float] | None = None

        for shared_temp in temp_list:
            for l0 in lam_list:
                r0 = by_state_temp_lam["K0"][(shared_temp, l0)]
                for l1 in lam_list:
                    if l1 > l0 + 1e-12:
                        continue
                    r1 = by_state_temp_lam["K1"][(shared_temp, l1)]
                    for l2 in lam_list:
                        if l2 > l1 + 1e-12:
                            continue
                        r2 = by_state_temp_lam["K2"][(shared_temp, l2)]
                        total_score = (
                            r0["selection_score"]
                            + r1["selection_score"]
                            + r2["selection_score"]
                        )
                        cand_key = (
                            round(total_score, 8),
                            round(r2["selection_score"], 8),
                            -shared_temp,
                            -(l0 + l1 + l2),
                        )
                        if best_key is None or cand_key > best_key:
                            best_key = cand_key
                            best_triple = (dict(r0), dict(r1), dict(r2))

        if best_triple is not None:
            selected_states["K0"] = best_triple[0]
            selected_states["K1"] = best_triple[1]
            selected_states["K2"] = best_triple[2]
            ckpt_prov = dict(
                exported_val_payload.get("checkpoint_provenance") or {}
            )
            if (
                "F" in selected_states
                and ckpt_prov.get("context")
                in (None, "", "not_available", "fallback_from_revision_mature")
            ):
                selected_states["F"] = {
                    **best_triple[2],
                    "state": "F",
                }

    monotonicity_satisfied = True
    if has_k012:
        monotonicity_satisfied = bool(
            selected_states["K0"]["language_weight"]
            >= selected_states["K1"]["language_weight"]
            >= selected_states["K2"]["language_weight"]
        )

    cms_meta = dict(exported_val_payload.get("cms_metadata") or {})
    result = {
        "schema_version": 2,
        "protocol": "sequence_decoding_s3_calibration",
        "dataset": exported_val_payload.get("dataset"),
        "language": language,
        "split": "val",
        "seed": exported_val_payload.get("seed", 0),
        "cms_eta": cms_meta.get("subtraction_eta"),
        "calibration_segments": len(cal_segments),
        "metric": metric_name,
        "temperature_grid": temp_list,
        "language_weight_grid": lam_list,
        "min_language_weight": min_lam,
        "shared_language_weight": bool(shared_language_weight),
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
