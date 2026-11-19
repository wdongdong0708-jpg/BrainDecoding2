"""多被试观测聚合（k-scaling）与闭集词表尺度扫描（|V|-scaling）。

从 `rolling_beam_search.py` 拆出，专注跨被试一致性加权聚合与多尺度单遍静态束搜索编排。
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch

from braindecoding.decoding.language_model_scoring import (
    CachedVocabularyLanguageScorer,
)


def _coerce_subject_score_tensors(
    subject_score_matrices: Sequence[torch.Tensor | np.ndarray],
) -> list[torch.Tensor]:
    return [
        m.detach().cpu().float()
        if isinstance(m, torch.Tensor)
        else torch.from_numpy(np.asarray(m, dtype=np.float32))
        for m in subject_score_matrices
    ]


def _average_subject_score_rows_rms_preserved(
    subject_score_matrices: Sequence[torch.Tensor | np.ndarray],
) -> torch.Tensor:
    """跨 k 名受试者的同一测试句段平均神经余弦得分，并执行 RMS 范数保持缩放。

    原因：k 名受试者的单位向量求均值后，非锁相背景噪声抵消导致合成向量模长按 ~1/sqrt(k) 收缩；
    通过按单受试者平均 RMS 重新对齐行模长，既 100% 保持每词位的跨受试者平均候选排序不变，
    又避免将温度 T_r 变相放大 sqrt(k) 倍。
    """
    tensors = _coerce_subject_score_tensors(subject_score_matrices)
    if len(tensors) == 1:
        return tensors[0]
    stacked = torch.stack(tensors, dim=0)  # [k, L, |V|]
    mean_mat = stacked.mean(dim=0)  # [L, |V|]
    single_rms = torch.sqrt(
        (stacked * stacked).mean(dim=-1, keepdim=True) + 1e-12
    ).mean(dim=0)  # [L, 1]
    avg_rms = torch.sqrt(
        (mean_mat * mean_mat).mean(dim=-1, keepdim=True) + 1e-12
    )  # [L, 1]
    scale = (single_rms / avg_rms).clamp(min=1.0, max=5.0)
    return mean_mat * scale


def _average_subject_score_rows_agreement_weighted(
    subject_score_matrices: Sequence[torch.Tensor | np.ndarray],
    *,
    agreement_alpha: float = 2.0,
) -> torch.Tensor:
    """遵循 SimpleB2T 式 (7)-(8) 的跨被试一致性加权神经得分聚合：

        r_i = || (1/k) sum_j s_{i,j} / ||s_{i,j}||_2 ||_2,
        A_i = (1 + alpha * r_i) * u_i,

    其中 u_i 为保持单受试者 RMS 尺度的共识得分向量；跨被试高度一致时放大分布锐度，
    分歧较大时保留平滑不确定性以利语言模型与束搜索消歧（alpha=0 时退化为 RMS 保持均值）。
    """
    tensors = _coerce_subject_score_tensors(subject_score_matrices)
    if len(tensors) == 1:
        return tensors[0]
    stacked = torch.stack(tensors, dim=0)  # [k, L, |V|]
    row_norms = torch.linalg.vector_norm(stacked, dim=-1, keepdim=True).clamp(min=1e-12)
    unit_rows = stacked / row_norms
    agreement_r = torch.linalg.vector_norm(
        unit_rows.mean(dim=0), dim=-1, keepdim=True
    ).clamp(0.0, 1.0)
    consensus_u = _average_subject_score_rows_rms_preserved(tensors)
    return consensus_u * (1.0 + float(agreement_alpha) * agreement_r)


def aggregate_multi_subject_payload(
    exported_payload: dict,
    k_observations: int,
    *,
    aggregation_mode: str = "rms_preserved",
    agreement_alpha: float = 2.0,
) -> dict:
    """构造跨 k 名受试者神经观测聚合的合成 .pt 载荷（对标 d'Ascoli 2024 / SimpleB2T 2026）。"""
    mode = str(aggregation_mode or "rms_preserved").strip().lower()
    if mode not in ("rms_preserved", "agreement_weighted"):
        raise ValueError(
            f"未知的多被试得分聚合模式 {aggregation_mode!r}，可选：'rms_preserved', 'agreement_weighted'"
        )
    alpha_val = float(agreement_alpha)
    k_req = max(1, int(k_observations))
    all_segments = list(exported_payload.get("segments") or [])
    grouped: dict[tuple, list[dict]] = {}
    for seg in all_segments:
        ref_tuple = tuple(str(w) for w in seg.get("reference_words", []))
        key = (
            str(seg.get("scope", "")),
            str(seg.get("material_id", "")),
            ref_tuple,
        )
        grouped.setdefault(key, []).append(seg)

    clean_bank = exported_payload["clean_scores"]
    shifted_bank = exported_payload.get("shifted_scores") or clean_bank
    state_keys = list(clean_bank.keys())

    agg_clean_lists: dict[str, list[torch.Tensor]] = {s: [] for s in state_keys}
    agg_shifted_lists: dict[str, list[torch.Tensor]] = {
        s: [] for s in state_keys
    }
    agg_segments: list[dict] = []
    cursor = 0

    for group_idx, (_, seg_group) in enumerate(grouped.items()):
        seg_group_sorted = sorted(
            seg_group, key=lambda item: str(item.get("subject_id", ""))
        )
        m_avail = len(seg_group_sorted)
        if m_avail == 0:
            continue
        k_eff = min(k_req, m_avail)
        # 采用确定性轮转窗口选取 k_eff 名受试者，确保全体受试者在语料上均匀代表
        chosen = [
            seg_group_sorted[(group_idx + offset) % m_avail]
            for offset in range(k_eff)
        ]
        template = chosen[0]
        seg_len = len(template["event_indices"])
        if seg_len == 0:
            continue

        for s in state_keys:
            c_rows = [
                clean_bank[s][member["event_indices"]] for member in chosen
            ]
            agg_clean_lists[s].append(
                _average_subject_score_rows_agreement_weighted(
                    c_rows, agreement_alpha=alpha_val
                )
                if mode == "agreement_weighted"
                else _average_subject_score_rows_rms_preserved(c_rows)
            )
            if s in shifted_bank:
                sh_rows = [
                    shifted_bank[s][member["event_indices"]]
                    for member in chosen
                ]
                agg_shifted_lists[s].append(
                    _average_subject_score_rows_agreement_weighted(
                        sh_rows, agreement_alpha=alpha_val
                    )
                    if mode == "agreement_weighted"
                    else _average_subject_score_rows_rms_preserved(sh_rows)
                )

        new_indices = list(range(cursor, cursor + seg_len))
        cursor += seg_len
        subj_label = f"avg_k{k_eff}(" + "+".join(
            str(c.get("subject_id", "")) for c in chosen
        ) + ")"
        agg_seg = {
            "segment_index": len(agg_segments),
            "group_id": f"k{k_eff}::{template.get('group_id', len(agg_segments))}",
            "subject_id": f"multi_subject_k{k_eff}",
            "subject_group_members": [
                str(c.get("subject_id", "")) for c in chosen
            ],
            "k_observations": int(k_eff),
            "recording_id": str(template.get("recording_id", "")),
            "scope": str(template.get("scope", "")),
            "block_id": str(
                template.get("block_id", template.get("scope", ""))
            ),
            "material_id": str(template.get("material_id", "")),
            "partition": str(template.get("partition", "evaluation")),
            "event_indices": new_indices,
            "reference_words": list(template["reference_words"]),
            "in_vocabulary_mask": list(template["in_vocabulary_mask"]),
            "subject_combination_label": subj_label,
        }
        for meta_key in (
            "event_ids",
            "onset_seconds",
            "offset_seconds",
            "pos_tags",
            "is_content_word_mask",
        ):
            if meta_key in template and template[meta_key] is not None:
                agg_seg[meta_key] = list(template[meta_key])
        agg_segments.append(agg_seg)

    vocab_size = len(exported_payload["vocabulary"])
    agg_clean = {
        s: (
            torch.cat(agg_clean_lists[s], dim=0)
            if agg_clean_lists[s]
            else torch.zeros((0, vocab_size), dtype=torch.float32)
        )
        for s in state_keys
    }
    agg_shifted = {
        s: (
            torch.cat(agg_shifted_lists[s], dim=0)
            if agg_shifted_lists[s]
            else agg_clean[s]
        )
        for s in state_keys
    }

    return {
        **exported_payload,
        "k_observations": k_req,
        "aggregation_mode": mode,
        "agreement_alpha": alpha_val,
        "segments": agg_segments,
        "clean_scores": agg_clean,
        "shifted_scores": agg_shifted,
    }


def _is_full_vocab_scale_token(scale: Any) -> bool:
    if scale is None:
        return True
    if isinstance(scale, str):
        s = scale.strip().lower()
        if s in ("full", "all", ""):
            return True
        if s.lstrip("-").isdigit():
            return int(s) <= 0
        return True
    return int(scale) <= 0


def _build_train_corpus_words_and_vocab_lookup(
    exported_payload: dict,
) -> tuple[list[str], dict[str, int]]:
    from braindecoding.decoding.neural_score_exporter import (
        resolve_vocabulary_frequency_order,
    )

    vocabulary = [str(w) for w in exported_payload.get("vocabulary") or []]
    if not vocabulary:
        return [], {}
    _, _, full_counts = resolve_vocabulary_frequency_order(exported_payload)
    word_to_vocab_idx: dict[str, int] = {}
    for idx_v, w_v in enumerate(vocabulary):
        norm_v = w_v.strip().lower()
        if norm_v and norm_v not in word_to_vocab_idx:
            word_to_vocab_idx[norm_v] = idx_v

    has_zero_count = any(
        int(full_counts.get(w_v, 0)) <= 0 for w_v in vocabulary
    )
    train_corpus_words: list[str] = []
    for w_v in vocabulary:
        norm_v = w_v.strip().lower()
        if not norm_v:
            continue
        c_val = int(full_counts.get(w_v, 0))
        c_eff = max(1, c_val + 1) if has_zero_count else max(1, c_val)
        train_corpus_words.extend([norm_v] * c_eff)
    return train_corpus_words, word_to_vocab_idx


def _resolve_l5_closed_set_scales(
    exported_payload: dict,
    vocab_sizes: Sequence[int | str] | None,
    *,
    min_sentence_length: int = 5,
    max_segments: int | None = None,
) -> list[dict]:
    from braindecoding.evaluation.vocabulary_coverage import (
        build_l5_sentence_matched_vocabulary,
    )

    all_segments = list(exported_payload.get("segments") or [])
    if not all_segments:
        return []
    train_corpus_words, word_to_vocab_idx = (
        _build_train_corpus_words_and_vocab_lookup(exported_payload)
    )
    if not train_corpus_words:
        return []
    test_sentence_token_lists = [
        [str(w) for w in (seg.get("reference_words") or [])]
        for seg in all_segments
    ]
    eff_min_len = max(1, int(min_sentence_length))
    full_words, full_covered_indices = build_l5_sentence_matched_vocabulary(
        train_words=train_corpus_words,
        test_segments=test_sentence_token_lists,
        size="full",
        min_sentence_length=eff_min_len,
        pad_with_train_freq=False,
    )
    if not full_covered_indices and eff_min_len > 1:
        eff_min_len = 1
        full_words, full_covered_indices = build_l5_sentence_matched_vocabulary(
            train_words=train_corpus_words,
            test_segments=test_sentence_token_lists,
            size="full",
            min_sentence_length=eff_min_len,
            pad_with_train_freq=False,
        )
    full_indices = [
        word_to_vocab_idx[w] for w in full_words if w in word_to_vocab_idx
    ]
    full_v_len = len(full_indices)

    raw_scales: list[int | str] = list(vocab_sizes or (50, 100, "full"))
    if not any(_is_full_vocab_scale_token(s) for s in raw_scales):
        raw_scales.append("full")

    specs_by_size: dict[int, dict] = {}
    for raw_s in raw_scales:
        is_full = _is_full_vocab_scale_token(raw_s)
        target_scale: int | str = (
            "full"
            if is_full
            else int(raw_s.strip() if isinstance(raw_s, str) else raw_s)
        )
        pad_flag = False
        if not is_full:
            scale_int = int(target_scale)
            pad_flag = scale_int == 250 or scale_int > full_v_len
        vocab_words, covered_l5_seg_indices = (
            build_l5_sentence_matched_vocabulary(
                train_words=train_corpus_words,
                test_segments=test_sentence_token_lists,
                size=target_scale,
                min_sentence_length=eff_min_len,
                pad_with_train_freq=pad_flag,
            )
        )
        if max_segments is not None:
            covered_l5_seg_indices = covered_l5_seg_indices[
                : int(max_segments)
            ]
        allowed_indices = [
            word_to_vocab_idx[w] for w in vocab_words if w in word_to_vocab_idx
        ]
        eff_n = len(allowed_indices)
        if eff_n <= 0:
            continue
        is_full_vocab = bool(
            is_full or (eff_n == full_v_len and not pad_flag)
        )
        if eff_n in specs_by_size:
            if is_full_vocab:
                specs_by_size[eff_n]["is_full_vocabulary"] = True
            continue
        specs_by_size[eff_n] = {
            "scale_key": f"V={eff_n}",
            "requested_scale": target_scale,
            "vocabulary_size": int(eff_n),
            "is_full_vocabulary": is_full_vocab,
            "allowed_vocab_indices": allowed_indices,
            "vocab_words": vocab_words,
            "segment_indices": covered_l5_seg_indices,
            "min_sentence_length": int(eff_min_len),
        }

    return [specs_by_size[k] for k in sorted(specs_by_size.keys())]


def _append_brain_only_jobs(
    jobs: dict[str, dict],
    available_clean: dict,
    *,
    states: Sequence[str] = ("S0", "K0", "K2", "F"),
    allowed_vocab_indices: Sequence[int] | None = None,
    existing_map: dict[str, dict] | None = None,
) -> None:
    for st in states:
        job_key = f"{st}_brain_only"
        if (
            st in available_clean
            and (existing_map is None or job_key not in existing_map)
            and job_key not in jobs
        ):
            job: dict = {"condition": job_key, "alignment": "clean"}
            if allowed_vocab_indices is not None:
                job["allowed_vocab_indices"] = allowed_vocab_indices
            jobs[job_key] = job


def run_multi_subject_observation_sweep(
    exported_payload: dict,
    scorer: CachedVocabularyLanguageScorer | None,
    hyperparameters: dict,
    *,
    k_values: Sequence[int] = (1, 2, 5, 10),
    k_observations: Sequence[int] | None = None,
    vocab_sizes: Sequence[int | str] | None = (50, 100, "full"),
    min_sentence_length: int | None = None,
    condition: str = "S3",
    alignment: str = "clean",
    aggregation_mode: str = "agreement_weighted",
    agreement_alpha: float = 2.0,
    include_brain_only: bool = False,
    single_subject_s3_bundle: dict | None = None,
    single_subject_full_vocab_bundles: dict[str, dict] | None = None,
    single_subject_vocab_scaling_bundles: dict[str, dict] | None = None,
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_batch_size: int = 32,
    max_segments: int | None = None,
    show_progress: bool = False,
) -> dict[str, dict]:
    """扫描 k in {1, 2, 5, 10} 跨受试者多观测神经聚合下的序列解码结果（同时支持全词库与 |V| in {50, 100, |V_full|} 受限核心词表）。"""
    from braindecoding.decoding.rolling_beam_search import (
        decode_dataset_segment_jobs,
        decode_dataset_segments,
    )

    from braindecoding.decoding.neural_score_exporter import (
        resolve_vocabulary_frequency_order,
    )

    active_k_values = k_observations if k_observations is not None else k_values
    all_subjects = sorted(
        {
            str(seg.get("subject_id", ""))
            for seg in exported_payload.get("segments", [])
            if seg.get("subject_id")
        }
    )
    max_avail_k = max(1, len(all_subjects))
    vocabulary = [str(w) for w in exported_payload.get("vocabulary") or []]
    full_v_size = len(vocabulary)
    use_l5_closed_set = min_sentence_length is not None or any(
        _is_full_vocab_scale_token(v) for v in (vocab_sizes or ())
    )

    if use_l5_closed_set and full_v_size > 0:
        eff_min_len = int(min_sentence_length) if min_sentence_length is not None else 5
        bundles_by_k: dict[str, dict] = {}
        seen_eff_k: dict[int, dict] = {}

        for k_raw in active_k_values:
            k_int = max(1, int(k_raw))
            k_eff = min(k_int, max_avail_k)
            if k_eff in seen_eff_k:
                cached = dict(seen_eff_k[k_eff])
                cached["k_observations"] = k_int
                cached["effective_k_observations"] = k_eff
                bundles_by_k[f"k={k_int}"] = cached
                continue

            agg_payload = (
                exported_payload
                if k_eff == 1
                else aggregate_multi_subject_payload(
                    exported_payload,
                    k_eff,
                    aggregation_mode=aggregation_mode,
                    agreement_alpha=agreement_alpha,
                )
            )
            available_clean = agg_payload.get("clean_scores") or {}
            available_shifted = agg_payload.get("shifted_scores") or {}
            scale_specs = _resolve_l5_closed_set_scales(
                agg_payload,
                vocab_sizes,
                min_sentence_length=eff_min_len,
                max_segments=max_segments,
            )
            full_spec = next(
                (s for s in scale_specs if s["is_full_vocabulary"]),
                scale_specs[-1] if scale_specs else None,
            )
            full_allowed_indices = (
                full_spec["allowed_vocab_indices"] if full_spec else None
            )
            full_seg_indices = (
                full_spec["segment_indices"] if full_spec else None
            )
            expected_full_segs = (
                len(full_seg_indices)
                if full_seg_indices is not None
                else len(agg_payload.get("segments") or [])
            )

            reuse_bundle: dict | None = None
            if (
                k_int == 1
                and isinstance(single_subject_s3_bundle, dict)
                and int(single_subject_s3_bundle.get("segment_count", -1))
                == expected_full_segs
            ):
                reuse_bundle = dict(single_subject_s3_bundle)
            elif (
                k_int == 1
                and isinstance(single_subject_full_vocab_bundles, dict)
                and isinstance(
                    single_subject_full_vocab_bundles.get("S3_clean"), dict
                )
                and int(
                    single_subject_full_vocab_bundles["S3_clean"].get(
                        "segment_count", -1
                    )
                )
                == expected_full_segs
            ):
                reuse_bundle = dict(
                    single_subject_full_vocab_bundles["S3_clean"]
                )

            if (
                k_int == 1
                and isinstance(single_subject_full_vocab_bundles, dict)
                and reuse_bundle is not None
            ):
                bundle = reuse_bundle
                conds_full: dict[str, dict] = {"S3_clean": dict(bundle)}
                for c_key in (
                    "S0",
                    "K0",
                    "S3_shifted_neg75s",
                    "F",
                    "K1",
                    "K0_shifted_neg75s",
                    "S0_brain_only",
                    "K0_brain_only",
                    "K1_brain_only",
                    "K2_brain_only",
                    "F_brain_only",
                ):
                    if c_key in single_subject_full_vocab_bundles:
                        conds_full[c_key] = dict(
                            single_subject_full_vocab_bundles[c_key]
                        )
                if (
                    "S0" not in conds_full
                    and "S0_clean" in single_subject_full_vocab_bundles
                ):
                    conds_full["S0"] = dict(
                        single_subject_full_vocab_bundles["S0_clean"]
                    )
                extra_k1_jobs: dict[str, dict] = {}
                if "F" not in conds_full and "F" in available_clean:
                    extra_k1_jobs["F"] = {
                        "condition": "F",
                        "alignment": "clean",
                        "allowed_vocab_indices": full_allowed_indices,
                    }
                if include_brain_only:
                    _append_brain_only_jobs(
                        extra_k1_jobs,
                        available_clean,
                        states=("S0", "K0", "K2", "F"),
                        allowed_vocab_indices=full_allowed_indices,
                        existing_map=conds_full,
                    )
                if extra_k1_jobs:
                    conds_full.update(
                        decode_dataset_segment_jobs(
                            agg_payload,
                            scorer,
                            hyperparameters,
                            extra_k1_jobs,
                            beam_width=beam_width,
                            candidate_top_k=candidate_top_k,
                            segment_batch_size=segment_batch_size,
                            segment_indices=full_seg_indices,
                            max_segments=max_segments,
                            show_progress=show_progress,
                            progress_desc=(
                                f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | full L>=5 extra | "
                                f"{len(extra_k1_jobs)} conditions]"
                            ),
                        )
                    )
            else:
                full_jobs: dict[str, dict] = {}
                has_s3_clean = all(
                    k in available_clean for k in ("K0", "K1", "K2")
                )
                if reuse_bundle is None and has_s3_clean:
                    full_jobs["S3_clean"] = {
                        "condition": condition,
                        "alignment": alignment,
                        "allowed_vocab_indices": full_allowed_indices,
                    }
                for st_key in ("S0", "K0", "F"):
                    if st_key in available_clean:
                        full_jobs[st_key] = {
                            "condition": st_key,
                            "alignment": "clean",
                            "allowed_vocab_indices": full_allowed_indices,
                        }
                if include_brain_only:
                    _append_brain_only_jobs(
                        full_jobs,
                        available_clean,
                        states=("S0", "K0", "K2", "F"),
                        allowed_vocab_indices=full_allowed_indices,
                    )
                if "K2" in available_shifted:
                    full_jobs["S3_shifted_neg75s"] = {
                        "condition": "S3",
                        "alignment": "shifted",
                        "allowed_vocab_indices": full_allowed_indices,
                    }
                full_results = (
                    decode_dataset_segment_jobs(
                        agg_payload,
                        scorer,
                        hyperparameters,
                        full_jobs,
                        beam_width=beam_width,
                        candidate_top_k=candidate_top_k,
                        segment_batch_size=segment_batch_size,
                        segment_indices=full_seg_indices,
                        max_segments=max_segments,
                        show_progress=show_progress,
                        progress_desc=(
                            f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | full L>=5 vocab | "
                            f"{len(full_jobs)} conditions batched]"
                        ),
                    )
                    if full_jobs
                    else {}
                )
                if reuse_bundle is not None:
                    bundle = reuse_bundle
                    conds_full = {"S3_clean": dict(bundle)}
                elif "S3_clean" in full_results:
                    bundle = dict(full_results.pop("S3_clean"))
                    conds_full = {"S3_clean": dict(bundle)}
                elif full_results:
                    first_cond_b = next(iter(full_results.values()))
                    bundle = dict(first_cond_b)
                    conds_full = {}
                else:
                    bundle = {"segment_count": 0, "final_sentences": []}
                    conds_full = {}
                conds_full.update(full_results)

            by_vocab: dict[str, dict] = {}
            for spec in scale_specs:
                eff_v = int(spec["vocabulary_size"])
                v_key = spec["scale_key"]
                if spec["is_full_vocabulary"]:
                    by_vocab[v_key] = {
                        "vocabulary_size": eff_v,
                        "is_full_vocabulary": True,
                        "conditions": dict(conds_full),
                    }
                    continue
                if (
                    k_eff == 1
                    and isinstance(single_subject_vocab_scaling_bundles, dict)
                    and v_key in single_subject_vocab_scaling_bundles
                ):
                    reused_v = dict(single_subject_vocab_scaling_bundles[v_key])
                    reused_v_conds = dict(reused_v.get("conditions") or {})
                    extra_v_jobs: dict[str, dict] = {}
                    if "F" not in reused_v_conds and "F" in available_clean:
                        extra_v_jobs["F"] = {
                            "condition": "F",
                            "alignment": "clean",
                            "allowed_vocab_indices": spec["allowed_vocab_indices"],
                        }
                    if include_brain_only:
                        _append_brain_only_jobs(
                            extra_v_jobs,
                            available_clean,
                            states=("S0", "K0", "K2", "F"),
                            allowed_vocab_indices=spec["allowed_vocab_indices"],
                            existing_map=reused_v_conds,
                        )
                    if extra_v_jobs:
                        reused_v_conds.update(
                            decode_dataset_segment_jobs(
                                agg_payload,
                                scorer,
                                hyperparameters,
                                extra_v_jobs,
                                beam_width=beam_width,
                                candidate_top_k=candidate_top_k,
                                segment_batch_size=segment_batch_size,
                                segment_indices=spec["segment_indices"],
                                max_segments=max_segments,
                                show_progress=show_progress,
                                progress_desc=(
                                    f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | V={eff_v} extra]"
                                ),
                            )
                        )
                        reused_v["conditions"] = reused_v_conds
                    by_vocab[v_key] = reused_v
                    continue

                active_indices = spec["allowed_vocab_indices"]
                scale_seg_indices = spec["segment_indices"]
                v_jobs: dict[str, dict] = {}
                if all(k in available_clean for k in ("K0", "K1", "K2")):
                    v_jobs["S3_clean"] = {
                        "condition": "S3",
                        "alignment": "clean",
                        "allowed_vocab_indices": active_indices,
                    }
                for st_key in ("S0", "K0", "F"):
                    if st_key in available_clean:
                        v_jobs[st_key] = {
                            "condition": st_key,
                            "alignment": "clean",
                            "allowed_vocab_indices": active_indices,
                        }
                if include_brain_only:
                    _append_brain_only_jobs(
                        v_jobs,
                        available_clean,
                        states=("S0", "K0", "K2", "F"),
                        allowed_vocab_indices=active_indices,
                    )
                if "K2" in available_shifted:
                    v_jobs["S3_shifted_neg75s"] = {
                        "condition": "S3",
                        "alignment": "shifted",
                        "allowed_vocab_indices": active_indices,
                    }
                v_conds: dict[str, dict] = (
                    decode_dataset_segment_jobs(
                        agg_payload,
                        scorer,
                        hyperparameters,
                        v_jobs,
                        beam_width=beam_width,
                        candidate_top_k=candidate_top_k,
                        segment_batch_size=segment_batch_size,
                        segment_indices=scale_seg_indices,
                        max_segments=max_segments,
                        show_progress=show_progress,
                        progress_desc=(
                            f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | V={eff_v} | "
                            f"{len(v_jobs)} conditions batched]"
                        ),
                    )
                    if v_jobs
                    else {}
                )
                by_vocab[v_key] = {
                    "vocabulary_size": eff_v,
                    "is_full_vocabulary": False,
                    "conditions": v_conds,
                }

            bundle["k_observations"] = k_int
            bundle["effective_k_observations"] = k_eff
            bundle["aggregation_mode"] = str(aggregation_mode)
            bundle["agreement_alpha"] = float(agreement_alpha)
            bundle["conditions"] = conds_full
            if by_vocab:
                bundle["by_vocabulary_size"] = by_vocab
            bundles_by_k[f"k={k_int}"] = bundle
            seen_eff_k[k_eff] = bundle

        return bundles_by_k

    freq_indices, _, _ = (
        resolve_vocabulary_frequency_order(exported_payload)
        if full_v_size > 0
        else ([], [], {})
    )

    eff_vocab_sizes: list[int] = []
    if vocab_sizes is not None and full_v_size > 0:
        for raw_v in vocab_sizes:
            v_int = int(raw_v)
            if v_int <= 0:
                continue
            eff_v = full_v_size if v_int >= full_v_size else v_int
            if eff_v not in eff_vocab_sizes:
                eff_vocab_sizes.append(eff_v)
        if full_v_size not in eff_vocab_sizes:
            eff_vocab_sizes.append(full_v_size)
        eff_vocab_sizes.sort()

    bundles_by_k: dict[str, dict] = {}
    seen_eff_k: dict[int, dict] = {}

    for k_raw in active_k_values:
        k_int = max(1, int(k_raw))
        k_eff = min(k_int, max_avail_k)
        if k_eff in seen_eff_k:
            cached = dict(seen_eff_k[k_eff])
            cached["k_observations"] = k_int
            cached["effective_k_observations"] = k_eff
            bundles_by_k[f"k={k_int}"] = cached
            continue

        agg_payload = (
            exported_payload
            if k_eff == 1
            else aggregate_multi_subject_payload(
                exported_payload,
                k_eff,
                aggregation_mode=aggregation_mode,
                agreement_alpha=agreement_alpha,
            )
        )
        available_clean = agg_payload.get("clean_scores") or {}
        available_shifted = agg_payload.get("shifted_scores") or {}

        reuse_bundle: dict | None = None
        if k_int == 1 and single_subject_s3_bundle is not None:
            reuse_bundle = dict(single_subject_s3_bundle)
        elif (
            k_int == 1
            and isinstance(single_subject_full_vocab_bundles, dict)
            and "S3_clean" in single_subject_full_vocab_bundles
        ):
            reuse_bundle = dict(single_subject_full_vocab_bundles["S3_clean"])

        if k_int == 1 and isinstance(single_subject_full_vocab_bundles, dict):
            if reuse_bundle is None:
                reuse_bundle = decode_dataset_segments(
                    agg_payload,
                    scorer,
                    hyperparameters,
                    condition=condition,
                    alignment=alignment,
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=f"S3 Multi-Subject [k={k_int} (eff={k_eff})]",
                )
            bundle = reuse_bundle
            conds_full: dict[str, dict] = {"S3_clean": dict(bundle)}
            for c_key in (
                "S0",
                "K0",
                "S3_shifted_neg75s",
                "F",
                "K1",
                "K0_shifted_neg75s",
                "S0_brain_only",
                "K0_brain_only",
                "K1_brain_only",
                "K2_brain_only",
                "F_brain_only",
            ):
                if c_key in single_subject_full_vocab_bundles:
                    conds_full[c_key] = dict(single_subject_full_vocab_bundles[c_key])
            if "S0" not in conds_full and "S0_clean" in single_subject_full_vocab_bundles:
                conds_full["S0"] = dict(single_subject_full_vocab_bundles["S0_clean"])
            extra_k1_legacy_jobs: dict[str, dict] = {}
            if "F" not in conds_full and "F" in available_clean:
                extra_k1_legacy_jobs["F"] = {"condition": "F", "alignment": "clean"}
            if include_brain_only:
                _append_brain_only_jobs(
                    extra_k1_legacy_jobs,
                    available_clean,
                    states=("S0", "K0", "K2", "F"),
                    existing_map=conds_full,
                )
            if extra_k1_legacy_jobs:
                conds_full.update(
                    decode_dataset_segment_jobs(
                        agg_payload,
                        scorer,
                        hyperparameters,
                        extra_k1_legacy_jobs,
                        beam_width=beam_width,
                        candidate_top_k=candidate_top_k,
                        segment_batch_size=segment_batch_size,
                        max_segments=max_segments,
                        show_progress=show_progress,
                        progress_desc=(
                            f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | full vocab extra]"
                        ),
                    )
                )
        else:
            full_jobs: dict[str, dict] = {}
            has_s3_clean = all(
                k in available_clean for k in ("K0", "K1", "K2")
            )
            if reuse_bundle is None and has_s3_clean:
                full_jobs["S3_clean"] = {
                    "condition": condition,
                    "alignment": alignment,
                }
            for st_key in ("S0", "K0", "F"):
                if st_key in available_clean:
                    full_jobs[st_key] = {"condition": st_key, "alignment": "clean"}
            if include_brain_only:
                _append_brain_only_jobs(
                    full_jobs,
                    available_clean,
                    states=("S0", "K0", "K2", "F"),
                )
            if "K2" in available_shifted:
                full_jobs["S3_shifted_neg75s"] = {
                    "condition": "S3",
                    "alignment": "shifted",
                }
            full_results = (
                decode_dataset_segment_jobs(
                    agg_payload,
                    scorer,
                    hyperparameters,
                    full_jobs,
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=(
                        f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | full vocab | "
                        f"{len(full_jobs)} conditions batched]"
                    ),
                )
                if full_jobs
                else {}
            )
            if reuse_bundle is not None:
                bundle = reuse_bundle
                conds_full = {"S3_clean": dict(bundle)}
            elif "S3_clean" in full_results:
                bundle = dict(full_results.pop("S3_clean"))
                conds_full = {"S3_clean": dict(bundle)}
            elif full_results:
                first_cond_b = next(iter(full_results.values()))
                bundle = dict(first_cond_b)
                conds_full = {}
            else:
                bundle = {"segment_count": 0, "final_sentences": []}
                conds_full = {}
            conds_full.update(full_results)

        by_vocab: dict[str, dict] = {}
        for eff_v in eff_vocab_sizes:
            v_key = f"V={eff_v}"
            if eff_v == full_v_size:
                by_vocab[v_key] = {
                    "vocabulary_size": int(eff_v),
                    "is_full_vocabulary": True,
                    "conditions": dict(conds_full),
                }
                continue
            if (
                k_eff == 1
                and isinstance(single_subject_vocab_scaling_bundles, dict)
                and v_key in single_subject_vocab_scaling_bundles
            ):
                reused_v = dict(single_subject_vocab_scaling_bundles[v_key])
                reused_v_conds = dict(reused_v.get("conditions") or {})
                extra_v_legacy_jobs: dict[str, dict] = {}
                if "F" not in reused_v_conds and "F" in available_clean:
                    extra_v_legacy_jobs["F"] = {
                        "condition": "F",
                        "alignment": "clean",
                        "allowed_vocab_indices": freq_indices[:eff_v],
                    }
                if include_brain_only:
                    _append_brain_only_jobs(
                        extra_v_legacy_jobs,
                        available_clean,
                        states=("S0", "K0", "K2", "F"),
                        allowed_vocab_indices=freq_indices[:eff_v],
                        existing_map=reused_v_conds,
                    )
                if extra_v_legacy_jobs:
                    reused_v_conds.update(
                        decode_dataset_segment_jobs(
                            agg_payload,
                            scorer,
                            hyperparameters,
                            extra_v_legacy_jobs,
                            beam_width=beam_width,
                            candidate_top_k=candidate_top_k,
                            segment_batch_size=segment_batch_size,
                            max_segments=max_segments,
                            show_progress=show_progress,
                            progress_desc=(
                                f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | V={eff_v} extra]"
                            ),
                        )
                    )
                    reused_v["conditions"] = reused_v_conds
                by_vocab[v_key] = reused_v
                continue

            active_indices = freq_indices[:eff_v]
            v_jobs: dict[str, dict] = {}
            if all(k in available_clean for k in ("K0", "K1", "K2")):
                v_jobs["S3_clean"] = {
                    "condition": "S3",
                    "alignment": "clean",
                    "allowed_vocab_indices": active_indices,
                }
            for st_key in ("S0", "K0", "F"):
                if st_key in available_clean:
                    v_jobs[st_key] = {
                        "condition": st_key,
                        "alignment": "clean",
                        "allowed_vocab_indices": active_indices,
                    }
            if include_brain_only:
                _append_brain_only_jobs(
                    v_jobs,
                    available_clean,
                    states=("S0", "K0", "K2", "F"),
                    allowed_vocab_indices=active_indices,
                )
            if "K2" in available_shifted:
                v_jobs["S3_shifted_neg75s"] = {
                    "condition": "S3",
                    "alignment": "shifted",
                    "allowed_vocab_indices": active_indices,
                }
            v_conds: dict[str, dict] = (
                decode_dataset_segment_jobs(
                    agg_payload,
                    scorer,
                    hyperparameters,
                    v_jobs,
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=(
                        f"S3 Multi-Subject [k={k_int} (eff={k_eff}) | V={eff_v} | "
                        f"{len(v_jobs)} conditions batched]"
                    ),
                )
                if v_jobs
                else {}
            )
            by_vocab[v_key] = {
                "vocabulary_size": int(eff_v),
                "is_full_vocabulary": False,
                "conditions": v_conds,
            }

        bundle["k_observations"] = k_int
        bundle["effective_k_observations"] = k_eff
        bundle["aggregation_mode"] = str(aggregation_mode)
        bundle["agreement_alpha"] = float(agreement_alpha)
        bundle["conditions"] = conds_full
        if by_vocab:
            bundle["by_vocabulary_size"] = by_vocab
        bundles_by_k[f"k={k_int}"] = bundle
        seen_eff_k[k_eff] = bundle

    return bundles_by_k


DEFAULT_VOCABULARY_SCALING_SIZES: tuple[int | str, ...] = (
    50,
    100,
    "full",
)


def run_vocabulary_scaling_sweep(
    exported_payload: dict,
    scorer: CachedVocabularyLanguageScorer | None,
    hyperparameters: dict,
    *,
    vocab_sizes: Sequence[int | str] = DEFAULT_VOCABULARY_SCALING_SIZES,
    core_vocab_size: int | str | None = 250,
    min_sentence_length: int | None = None,
    include_brain_only: bool = False,
    full_vocab_bundles: dict[str, dict] | None = None,
    existing_scaling_bundles: dict[str, dict] | None = None,
    beam_width: int = 5,
    candidate_top_k: int = 20,
    segment_batch_size: int = 32,
    max_segments: int | None = None,
    show_progress: bool = False,
    on_scale_completed: Callable[[str, dict], None] | None = None,
) -> dict[str, dict]:
    """多尺度候选词表统一束搜索解码表现扫描（支持模式一 L>=5 句表同源闭集协议及传统词频截断）。"""
    from braindecoding.decoding.rolling_beam_search import (
        decode_dataset_segment_jobs,
        decode_dataset_segments,
    )

    from braindecoding.decoding.neural_score_exporter import (
        resolve_vocabulary_frequency_order,
    )
    from braindecoding.evaluation.sequence_decoding_evaluation import (
        _normalize_decoded_bundles,
    )

    vocabulary = [str(w) for w in exported_payload.get("vocabulary") or []]
    full_v_size = len(vocabulary)
    if full_v_size == 0:
        return {}

    use_l5_closed_set = min_sentence_length is not None or any(
        _is_full_vocab_scale_token(v) for v in (vocab_sizes or ())
    )
    if use_l5_closed_set:
        eff_min_len = int(min_sentence_length) if min_sentence_length is not None else 5
        scale_specs = _resolve_l5_closed_set_scales(
            exported_payload,
            vocab_sizes,
            min_sentence_length=eff_min_len,
            max_segments=max_segments,
        )
        if not scale_specs:
            return {}

        available_clean = exported_payload.get("clean_scores") or {}
        available_shifted = exported_payload.get("shifted_scores") or {}
        has_s3_clean_scores = all(
            k in available_clean for k in ("K0", "K1", "K2")
        )

        scaling_bundles: dict[str, dict] = {}
        for spec in scale_specs:
            eff_n = int(spec["vocabulary_size"])
            scale_key = spec["scale_key"]
            active_indices = spec["allowed_vocab_indices"]
            seg_indices = spec["segment_indices"]
            n_scale_segs = len(seg_indices)
            is_full_vocab = bool(spec["is_full_vocabulary"])

            existing_entry = (
                existing_scaling_bundles.get(scale_key)
                if isinstance(existing_scaling_bundles, dict)
                else None
            )
            existing_conds: dict[str, dict] = {}
            if isinstance(existing_entry, dict) and isinstance(
                existing_entry.get("conditions"), dict
            ):
                for ck, cb in existing_entry["conditions"].items():
                    if (
                        isinstance(cb, dict)
                        and int(cb.get("segment_count", -1)) == n_scale_segs
                    ):
                        existing_conds[ck] = cb

            required_keys: list[str] = []
            if has_s3_clean_scores:
                required_keys.append("S3_clean")
            if "S0" in available_clean:
                required_keys.append("S0")
            if "K0" in available_clean:
                required_keys.append("K0")
            if "K1" in available_clean:
                required_keys.append("K1")
            if "F" in available_clean:
                required_keys.append("F")
            if "K0" in available_shifted:
                required_keys.append("K0_shifted_neg75s")
            if "K2" in available_shifted:
                required_keys.append("S3_shifted_neg75s")
            required_keys.append("LM-Only")
            if include_brain_only:
                for bo_st in ("S0", "K0", "K1", "K2", "F"):
                    if bo_st in available_clean:
                        required_keys.append(f"{bo_st}_brain_only")

            if existing_conds and all(rk in existing_conds for rk in required_keys):
                reused_conds = _normalize_decoded_bundles(
                    existing_conds
                )
                scale_entry = {
                    "vocabulary_size": eff_n,
                    "is_full_vocabulary": is_full_vocab,
                    "protocol_mode": "l5_sentence_matched_closed_set",
                    "min_sentence_length": int(spec["min_sentence_length"]),
                    "test_token_coverage_pct": 100.0 if n_scale_segs > 0 else 0.0,
                    "core_100pct_segment_count": n_scale_segs,
                    "core_80pct_segment_count": n_scale_segs,
                    "total_segment_count": n_scale_segs,
                    "conditions": reused_conds,
                }
                scaling_bundles[scale_key] = scale_entry
                if on_scale_completed is not None:
                    on_scale_completed(scale_key, scale_entry)
                continue

            if (
                is_full_vocab
                and isinstance(full_vocab_bundles, dict)
                and bool(full_vocab_bundles)
                and (
                    not has_s3_clean_scores
                    or (
                        isinstance(full_vocab_bundles.get("S3_clean"), dict)
                        and int(
                            full_vocab_bundles["S3_clean"].get(
                                "segment_count", -1
                            )
                        )
                        == n_scale_segs
                    )
                )
            ):
                raw_reused: dict[str, dict] = dict(existing_conds)
                for k in (
                    "S0",
                    "S0_clean",
                    "K0",
                    "K1",
                    "S3_clean",
                    "F",
                    "F_clean",
                    "LM-Only",
                    "K0_shifted_neg75s",
                    "S3_shifted_neg75s",
                    "S0_brain_only",
                    "K0_brain_only",
                    "K1_brain_only",
                    "K2_brain_only",
                    "F_brain_only",
                ):
                    if (
                        k in full_vocab_bundles
                        and isinstance(full_vocab_bundles[k], dict)
                        and int(full_vocab_bundles[k].get("segment_count", -1))
                        == n_scale_segs
                    ):
                        raw_reused[k] = full_vocab_bundles[k]
                if "S0_clean" in raw_reused and "S0" not in raw_reused:
                    raw_reused["S0"] = raw_reused["S0_clean"]
                if "F_clean" in raw_reused and "F" not in raw_reused:
                    raw_reused["F"] = raw_reused["F_clean"]
                if "F" not in raw_reused and "F" in available_clean:
                    raw_reused["F"] = decode_dataset_segments(
                        exported_payload,
                        scorer,
                        hyperparameters,
                        condition="F",
                        alignment="clean",
                        beam_width=beam_width,
                        candidate_top_k=candidate_top_k,
                        segment_batch_size=segment_batch_size,
                        allowed_vocab_indices=active_indices,
                        segment_indices=seg_indices,
                        max_segments=max_segments,
                        show_progress=show_progress,
                        progress_desc=f"S3 VocabScale [V={eff_n} | Full-Context]",
                    )
                if "LM-Only" not in raw_reused:
                    raw_reused["LM-Only"] = decode_dataset_segments(
                        exported_payload,
                        scorer,
                        hyperparameters,
                        condition="LM-Only",
                        alignment="clean",
                        beam_width=beam_width,
                        candidate_top_k=candidate_top_k,
                        segment_batch_size=segment_batch_size,
                        allowed_vocab_indices=active_indices,
                        segment_indices=seg_indices,
                        max_segments=max_segments,
                        show_progress=show_progress,
                        progress_desc=f"S3 VocabScale [V={eff_n} | LM-Only]",
                    )
                if include_brain_only:
                    extra_bo_jobs: dict[str, dict] = {}
                    _append_brain_only_jobs(
                        extra_bo_jobs,
                        available_clean,
                        states=("S0", "K0", "K1", "K2", "F"),
                        allowed_vocab_indices=active_indices,
                        existing_map=raw_reused,
                    )
                    if extra_bo_jobs:
                        raw_reused.update(
                            decode_dataset_segment_jobs(
                                exported_payload,
                                scorer,
                                hyperparameters,
                                extra_bo_jobs,
                                beam_width=beam_width,
                                candidate_top_k=candidate_top_k,
                                segment_batch_size=segment_batch_size,
                                segment_indices=seg_indices,
                                max_segments=max_segments,
                                show_progress=show_progress,
                                progress_desc=f"S3 VocabScale [V={eff_n} | Brain-Only]",
                            )
                        )
                reused_conds = _normalize_decoded_bundles(
                    raw_reused
                )
                scale_entry = {
                    "vocabulary_size": eff_n,
                    "is_full_vocabulary": True,
                    "protocol_mode": "l5_sentence_matched_closed_set",
                    "min_sentence_length": int(spec["min_sentence_length"]),
                    "test_token_coverage_pct": 100.0 if n_scale_segs > 0 else 0.0,
                    "core_100pct_segment_count": n_scale_segs,
                    "core_80pct_segment_count": n_scale_segs,
                    "total_segment_count": n_scale_segs,
                    "conditions": reused_conds,
                }
                scaling_bundles[scale_key] = scale_entry
                if on_scale_completed is not None:
                    on_scale_completed(scale_key, scale_entry)
                continue

            cond_map: dict[str, dict] = dict(existing_conds)
            scale_jobs: dict[str, dict] = {}
            if has_s3_clean_scores and "S3_clean" not in cond_map:
                scale_jobs["S3_clean"] = {
                    "condition": "S3",
                    "alignment": "clean",
                }
            for static_state in ("S0", "K0", "K1"):
                if static_state in available_clean and static_state not in cond_map:
                    scale_jobs[static_state] = {
                        "condition": static_state,
                        "alignment": "clean",
                    }
            if "F" in available_clean and "F" not in cond_map:
                scale_jobs["F"] = {"condition": "F", "alignment": "clean"}
            if "LM-Only" not in cond_map:
                scale_jobs["LM-Only"] = {"condition": "LM-Only", "alignment": "clean"}
            if "K0" in available_shifted and "K0_shifted_neg75s" not in cond_map:
                scale_jobs["K0_shifted_neg75s"] = {
                    "condition": "K0",
                    "alignment": "shifted",
                }
            if "K2" in available_shifted and "S3_shifted_neg75s" not in cond_map:
                scale_jobs["S3_shifted_neg75s"] = {
                    "condition": "S3",
                    "alignment": "shifted",
                }
            if include_brain_only:
                _append_brain_only_jobs(
                    scale_jobs,
                    available_clean,
                    states=("S0", "K0", "K1", "K2", "F"),
                    existing_map=cond_map,
                )
            if scale_jobs:
                for job in scale_jobs.values():
                    job["allowed_vocab_indices"] = active_indices
                cond_map.update(
                    decode_dataset_segment_jobs(
                        exported_payload,
                        scorer,
                        hyperparameters,
                        scale_jobs,
                        beam_width=beam_width,
                        candidate_top_k=candidate_top_k,
                        segment_batch_size=segment_batch_size,
                        segment_indices=seg_indices,
                        max_segments=max_segments,
                        show_progress=show_progress,
                        progress_desc=(
                            f"S3 VocabScale [V={eff_n} (0% OOV L>={spec['min_sentence_length']}, n={n_scale_segs}) | "
                            f"{len(scale_jobs)} conditions batched]"
                        ),
                    )
                )

            cond_map = _normalize_decoded_bundles(cond_map)
            scale_entry = {
                "vocabulary_size": eff_n,
                "is_full_vocabulary": is_full_vocab,
                "protocol_mode": "l5_sentence_matched_closed_set",
                "min_sentence_length": int(spec["min_sentence_length"]),
                "test_token_coverage_pct": 100.0 if n_scale_segs > 0 else 0.0,
                "core_100pct_segment_count": n_scale_segs,
                "core_80pct_segment_count": n_scale_segs,
                "total_segment_count": n_scale_segs,
                "conditions": cond_map,
            }
            scaling_bundles[scale_key] = scale_entry
            if on_scale_completed is not None:
                on_scale_completed(scale_key, scale_entry)

        ordered_result: dict[str, dict] = {}
        for spec in scale_specs:
            k_str = spec["scale_key"]
            if k_str in scaling_bundles:
                ordered_result[k_str] = scaling_bundles[k_str]
        return ordered_result

    freq_indices, freq_words, _ = resolve_vocabulary_frequency_order(
        exported_payload
    )
    all_segments = list(exported_payload.get("segments") or [])
    if max_segments is not None:
        all_segments = all_segments[: int(max_segments)]
    expected_seg_count = len(all_segments)
    total_ref_tokens = max(
        1, sum(len(seg.get("reference_words") or []) for seg in all_segments)
    )

    normalized_sizes: list[int] = []
    for raw_n in vocab_sizes:
        n_int = int(raw_n)
        if n_int <= 0:
            continue
        eff_n = full_v_size if n_int >= full_v_size else n_int
        if eff_n not in normalized_sizes:
            normalized_sizes.append(eff_n)
    if (
        core_vocab_size is not None
        and 0 < int(core_vocab_size) <= full_v_size
        and int(core_vocab_size) not in normalized_sizes
    ):
        normalized_sizes.append(int(core_vocab_size))
    if full_v_size not in normalized_sizes:
        normalized_sizes.append(full_v_size)
    normalized_sizes.sort()

    # 优先执行 core_vocab_size (默认 250)，确保高覆盖核心子集主表最快落盘并预热转移缓存
    execution_order = list(normalized_sizes)
    if (
        core_vocab_size is not None
        and int(core_vocab_size) in execution_order
    ):
        c_val = int(core_vocab_size)
        execution_order = [c_val] + [x for x in execution_order if x != c_val]

    scaling_bundles: dict[str, dict] = {}
    if isinstance(existing_scaling_bundles, dict):
        for k_str, entry in existing_scaling_bundles.items():
            if isinstance(entry, dict) and isinstance(
                entry.get("conditions"), dict
            ):
                s3_b = entry["conditions"].get("S3_clean")
                if (
                    isinstance(s3_b, dict)
                    and int(s3_b.get("segment_count", -1)) == expected_seg_count
                ):
                    scaling_bundles[str(k_str)] = entry

    available_clean = exported_payload.get("clean_scores") or {}
    available_shifted = exported_payload.get("shifted_scores") or {}
    ckpt_prov = dict(exported_payload.get("checkpoint_provenance") or {})
    has_real_context = bool(
        ckpt_prov.get("has_context_checkpoint")
        or (
            ckpt_prov.get("context")
            not in (None, "", "not_available", "fallback_from_revision_mature")
        )
    )

    for eff_n in execution_order:
        scale_key = f"V={eff_n}"
        active_indices = freq_indices[:eff_n]
        active_words = freq_words[:eff_n]
        active_set = set(active_words)

        in_vocab_tokens = sum(
            1
            for seg in all_segments
            for w in (seg.get("reference_words") or [])
            if str(w) in active_set
        )
        token_coverage_pct = float(in_vocab_tokens / total_ref_tokens * 100.0)
        segs_100pct = sum(
            1
            for seg in all_segments
            if (seg.get("reference_words") or [])
            and all(str(w) in active_set for w in seg["reference_words"])
        )
        segs_80pct = sum(
            1
            for seg in all_segments
            if (seg.get("reference_words") or [])
            and (
                sum(1 for w in seg["reference_words"] if str(w) in active_set)
                / len(seg["reference_words"])
                >= 0.80 - 1e-9
            )
        )

        max_core_v = max(int(core_vocab_size or 250), 250)
        is_core_benchmark_scale = eff_n <= max_core_v

        existing_entry = scaling_bundles.get(scale_key)
        existing_conds = (
            dict(existing_entry.get("conditions") or {})
            if isinstance(existing_entry, dict)
            else {}
        )
        has_s3_clean_scores = all(
            k in available_clean for k in ("K0", "K1", "K2")
        )
        required_keys: list[str] = []
        if has_s3_clean_scores:
            required_keys.append("S3_clean")
        if "S0" in available_clean:
            required_keys.append("S0")
        if "K0" in available_clean:
            required_keys.append("K0")
        if "K2" in available_shifted:
            required_keys.append("S3_shifted_neg75s")
        bo_states = (
            ("S0", "K0", "K1", "K2", "F")
            if (is_core_benchmark_scale or eff_n == full_v_size)
            else ("S0", "K0", "K2")
        )
        if is_core_benchmark_scale or eff_n == full_v_size:
            if "K1" in available_clean:
                required_keys.append("K1")
            if "F" in available_clean:
                required_keys.append("F")
            if "K0" in available_shifted:
                required_keys.append("K0_shifted_neg75s")
            required_keys.append("LM-Only")
        if include_brain_only:
            for bo_st in bo_states:
                if bo_st in available_clean:
                    required_keys.append(f"{bo_st}_brain_only")
        if existing_conds and all(rk in existing_conds for rk in required_keys):
            continue

        if (
            eff_n == full_v_size
            and isinstance(full_vocab_bundles, dict)
            and (
                not has_s3_clean_scores
                or "S3_clean" in full_vocab_bundles
            )
            and bool(full_vocab_bundles)
        ):
            raw_reused: dict[str, dict] = dict(existing_conds)
            for k in (
                "S0",
                "S0_clean",
                "K0",
                "K1",
                "S3_clean",
                "F",
                "F_clean",
                "LM-Only",
                "K0_shifted_neg75s",
                "S3_shifted_neg75s",
                "S0_brain_only",
                "K0_brain_only",
                "K1_brain_only",
                "K2_brain_only",
                "F_brain_only",
            ):
                if k in full_vocab_bundles:
                    raw_reused[k] = full_vocab_bundles[k]
            if "S0_clean" in raw_reused and "S0" not in raw_reused:
                raw_reused["S0"] = raw_reused["S0_clean"]
            if "F_clean" in raw_reused and "F" not in raw_reused:
                raw_reused["F"] = raw_reused["F_clean"]
            if "F" not in raw_reused and "F" in available_clean:
                raw_reused["F"] = decode_dataset_segments(
                    exported_payload,
                    scorer,
                    hyperparameters,
                    condition="F",
                    alignment="clean",
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=f"S3 VocabScale [V={eff_n} | Full-Context]",
                )
            if "LM-Only" not in raw_reused:
                raw_reused["LM-Only"] = decode_dataset_segments(
                    exported_payload,
                    scorer,
                    hyperparameters,
                    condition="LM-Only",
                    alignment="clean",
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=f"S3 VocabScale [V={eff_n} | LM-Only]",
                )
            if include_brain_only:
                extra_legacy_bo_jobs: dict[str, dict] = {}
                _append_brain_only_jobs(
                    extra_legacy_bo_jobs,
                    available_clean,
                    states=bo_states,
                    existing_map=raw_reused,
                )
                if extra_legacy_bo_jobs:
                    raw_reused.update(
                        decode_dataset_segment_jobs(
                            exported_payload,
                            scorer,
                            hyperparameters,
                            extra_legacy_bo_jobs,
                            beam_width=beam_width,
                            candidate_top_k=candidate_top_k,
                            segment_batch_size=segment_batch_size,
                            max_segments=max_segments,
                            show_progress=show_progress,
                            progress_desc=f"S3 VocabScale [V={eff_n} | Brain-Only]",
                        )
                    )
            reused_conds = _normalize_decoded_bundles(
                raw_reused
            )
            scale_entry = {
                "vocabulary_size": int(eff_n),
                "is_full_vocabulary": True,
                "test_token_coverage_pct": round(token_coverage_pct, 4),
                "core_100pct_segment_count": int(segs_100pct),
                "core_80pct_segment_count": int(segs_80pct),
                "total_segment_count": int(expected_seg_count),
                "conditions": reused_conds,
            }
            scaling_bundles[scale_key] = scale_entry
            if on_scale_completed is not None:
                on_scale_completed(scale_key, scale_entry)
            continue

        cond_map: dict[str, dict] = dict(existing_conds)
        static_states = (
            ("S0", "K0", "K1") if is_core_benchmark_scale else ("S0", "K0")
        )
        scale_jobs: dict[str, dict] = {}
        if has_s3_clean_scores and "S3_clean" not in cond_map:
            scale_jobs["S3_clean"] = {
                "condition": "S3",
                "alignment": "clean",
            }
        for static_state in static_states:
            if static_state in available_clean and static_state not in cond_map:
                scale_jobs[static_state] = {
                    "condition": static_state,
                    "alignment": "clean",
                }
        if is_core_benchmark_scale and "F" in available_clean and "F" not in cond_map:
            scale_jobs["F"] = {"condition": "F", "alignment": "clean"}
        if is_core_benchmark_scale and "LM-Only" not in cond_map:
            scale_jobs["LM-Only"] = {"condition": "LM-Only", "alignment": "clean"}
        if (
            is_core_benchmark_scale
            and "K0" in available_shifted
            and "K0_shifted_neg75s" not in cond_map
        ):
            scale_jobs["K0_shifted_neg75s"] = {
                "condition": "K0",
                "alignment": "shifted",
            }
        if "K2" in available_shifted and "S3_shifted_neg75s" not in cond_map:
            scale_jobs["S3_shifted_neg75s"] = {
                "condition": "S3",
                "alignment": "shifted",
            }
        if include_brain_only:
            _append_brain_only_jobs(
                scale_jobs,
                available_clean,
                states=bo_states,
                existing_map=cond_map,
            )
        if scale_jobs:
            for job in scale_jobs.values():
                job["allowed_vocab_indices"] = active_indices
            cond_map.update(
                decode_dataset_segment_jobs(
                    exported_payload,
                    scorer,
                    hyperparameters,
                    scale_jobs,
                    beam_width=beam_width,
                    candidate_top_k=candidate_top_k,
                    segment_batch_size=segment_batch_size,
                    max_segments=max_segments,
                    show_progress=show_progress,
                    progress_desc=(
                        f"S3 VocabScale [V={eff_n} | {len(scale_jobs)} conditions batched]"
                    ),
                )
            )

        cond_map = _normalize_decoded_bundles(cond_map)
        scale_entry = {
            "vocabulary_size": int(eff_n),
            "is_full_vocabulary": bool(eff_n == full_v_size),
            "test_token_coverage_pct": round(token_coverage_pct, 4),
            "core_100pct_segment_count": int(segs_100pct),
            "core_80pct_segment_count": int(segs_80pct),
            "total_segment_count": int(expected_seg_count),
            "conditions": cond_map,
        }
        scaling_bundles[scale_key] = scale_entry
        if on_scale_completed is not None:
            on_scale_completed(scale_key, scale_entry)

    # 按词表规模从小到大排序返回
    ordered_result: dict[str, dict] = {}
    for n_val in normalized_sizes:
        k_str = f"V={n_val}"
        if k_str in scaling_bundles:
            ordered_result[k_str] = scaling_bundles[k_str]
    return ordered_result
