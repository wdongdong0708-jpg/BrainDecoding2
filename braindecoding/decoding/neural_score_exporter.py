"""S3 序列解码协议神经特征导出器（Module 1）。

提取 S0 (Stage 1 Word)、K0/K1/K2 (Stage 2.2 Revision) 与 F (Stage 2.1 Context) 在全训练词表 V 上的
[N, |V|] 余弦相似度得分矩阵，支持复用训练期一致的 6 个纯历史因果窗口 {-15, -30, -45, -60, -90, -120}s
执行背景共模扣除（Common-Mode Subtraction, CMS），并同步导出未见纯历史因果错位 -75s（Shifted -75s）
负对照得分矩阵，保存为 .pt 文件供后续超参校准与极速束搜索直接消费。
"""

from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.nn import functional as F_nn

from braindecoding.config import (
    PROJECT_ROOT,
    ensure_default_environment,
    load_yaml_with_extends,
    project_path,
    resolve_config_path,
    results_root,
)
from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import train as chinese_task
from braindecoding.tasks.word_decoding.pallier2025 import train as pallier_task
from braindecoding.training.protocol_stage_training import CONFIG_PATH, group_column
from braindecoding.training.training_runtime import (
    choose_device,
    limit_rows,
    load_checkpoint,
    save_json,
    set_seed,
)
from braindecoding.training.word_encoder_training import (
    encode_loader,
    make_loader,
    resolve_amp_dtype,
)
from braindecoding.training.workflow_runner import (
    _infer_dataset_and_stage_from_yaml,
    build_workflow_config,
    run_cache_workflow,
    verify_cache_ready,
    workflow_output_directory,
)


CANONICAL_STATES: tuple[str, ...] = ("S0", "K0", "K1", "K2", "F")
DEFAULT_BACKGROUND_SHIFT_SECONDS: tuple[float, ...] = (
    -15.0,
    -30.0,
    -45.0,
    -60.0,
    -90.0,
    -120.0,
)
DEFAULT_CONTROL_SHIFT_SECONDS: float = -75.0
DEFAULT_SUBTRACTION_ETA: float = 0.8


def _digest_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _l2_normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F_nn.normalize(tensor.float(), p=2.0, dim=-1, eps=eps)


def apply_historical_common_mode_subtraction(
    target_embeddings: torch.Tensor,
    historical_shift_embeddings: Sequence[torch.Tensor],
    *,
    eta: float = DEFAULT_SUBTRACTION_ETA,
) -> torch.Tensor:
    """对目标神经预测向量扣除历史因果错位窗口均值并重新 L2 归一化。

    公式：
        \\tilde{y} = normalize(\\hat{y} - \\eta * mean_{\\delta \\in D_hist}(\\hat{y}_\\delta))
    对 Clean (0s) 与 Shifted (-75s) 对称扣除同一历史背景均值 \\bar{y}_hist。
    """
    norm_target = _l2_normalize_tensor(target_embeddings)
    if not historical_shift_embeddings or abs(float(eta)) <= 1e-12:
        return norm_target
    stacked_hist = torch.stack(
        [_l2_normalize_tensor(item) for item in historical_shift_embeddings],
        dim=0,
    )
    mean_hist = stacked_hist.mean(dim=0)
    debiased = norm_target - float(eta) * mean_hist
    return _l2_normalize_tensor(debiased)


def build_segment_inventory(
    dataset: str,
    table,
    vocabulary: Sequence[str],
    *,
    adjacent_components_per_block: int = 5,
    calibration_fraction: float = 0.2,
    partition_seed: int = 20260928,
) -> list[dict]:
    """从事件表构建句段（Segment）结构、连通刺激材料块（Block）与词表覆盖掩码。"""
    vocab_set = set(str(w) for w in vocabulary)
    settings = load_yaml_with_extends(CONFIG_PATH)
    grp_col = settings["adapters"][dataset]["group_column"]
    working = table.reset_index(drop=True).copy()
    event_col = "event_id" if "event_id" in working.columns else "事件编号"
    subj_col = "subject_id" if "subject_id" in working.columns else "受试者"
    rec_col = "recording_id" if "recording_id" in working.columns else "记录编号"
    word_col = (
        "normalized_word" if "normalized_word" in working.columns else "标准词"
    )
    sent_col = (
        "sentence_uid" if "sentence_uid" in working.columns else "上下文编号"
    )
    if grp_col not in working.columns:
        if grp_col == "runtime_context_uid" and rec_col in working.columns and sent_col in working.columns:
            working["runtime_context_uid"] = (
                working[rec_col].astype(str) + "::" + working[sent_col].astype(str)
            )
        elif "protocol_group" in working.columns:
            grp_col = "protocol_group"
        elif "context_uid" in working.columns:
            grp_col = "context_uid"
        else:
            grp_col = sent_col

    segments: list[dict] = []
    parent: list[int] = []
    atom_owner: dict[str, int] = {}

    def find_root(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    for group_name, rows in working.groupby(grp_col, sort=False):
        seg_idx = len(segments)
        parent.append(seg_idx)
        row_indices = [int(i) for i in rows.index.tolist()]
        if dataset == "pallier2025":
            run_col = "运行编号" if "运行编号" in rows.columns else rec_col
            bids_col = "BIDS事件行号" if "BIDS事件行号" in rows.columns else "event_index"
            scope = str(rows[run_col].iloc[0])
            if bids_col in rows.columns:
                atoms = [f"{scope}|event-{v}" for v in rows[bids_col].tolist()]
                order = int(rows[bids_col].min())
            else:
                atoms = [f"{scope}|row-{r}" for r in row_indices]
                order = int(min(row_indices))
        else:
            if "chapter" in rows.columns:
                scope = f"chapter-{int(rows['chapter'].iloc[0]):02d}"
            elif "章节" in rows.columns:
                scope = f"chapter-{int(rows['章节'].iloc[0]):02d}"
            else:
                scope = str(rows[rec_col].iloc[0])
            src_sent_col = (
                "source_sentence_uid"
                if "source_sentence_uid" in rows.columns
                else ("源句编号" if "源句编号" in rows.columns else None)
            )
            if src_sent_col is not None:
                lines = []
                for val in rows[src_sent_col].astype(str).tolist():
                    match = re.search(r"actual-line-(\d+)", val)
                    lines.append(match.group(1) if match else val)
                atoms = [f"{scope}|source-line-{ln}" for ln in lines]
                numeric_lines = [int(ln) for ln in lines if str(ln).isdigit()]
                order = min(numeric_lines) if numeric_lines else int(min(row_indices))
            else:
                atoms = [f"{scope}|group-{group_name}"]
                order = int(min(row_indices))

        for atom in set(atoms):
            if atom in atom_owner:
                r1, r2 = find_root(seg_idx), find_root(atom_owner[atom])
                parent[max(r1, r2)] = min(r1, r2)
            else:
                atom_owner[atom] = seg_idx

        ref_words = rows[word_col].astype(str).tolist()
        in_vocab = [bool(w in vocab_set) for w in ref_words]
        segments.append(
            {
                "segment_index": seg_idx,
                "group_id": str(group_name),
                "subject_id": str(rows[subj_col].iloc[0]),
                "recording_id": str(rows[rec_col].iloc[0]),
                "scope": scope,
                "order": order,
                "event_indices": row_indices,
                "event_ids": rows[event_col].astype(str).tolist(),
                "reference_words": ref_words,
                "in_vocabulary_mask": in_vocab,
                "_atoms": sorted(set(atoms)),
            }
        )

    components: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(segments)):
        components[find_root(idx)].append(idx)

    materials_by_scope: dict[str, list[dict]] = defaultdict(list)
    for comp_members in components.values():
        atoms = sorted({a for m in comp_members for a in segments[m]["_atoms"]})
        scope = segments[comp_members[0]]["scope"]
        mat_key = _digest_json(atoms)[:20]
        mat_order = min(segments[m]["order"] for m in comp_members)
        for m in comp_members:
            segments[m]["material_id"] = mat_key
            segments[m].pop("_atoms", None)
        materials_by_scope[scope].append(
            {
                "material_id": mat_key,
                "scope": scope,
                "order": mat_order,
                "segments": comp_members,
            }
        )

    width = max(1, int(adjacent_components_per_block))
    for scope, entries in sorted(materials_by_scope.items()):
        entries.sort(key=lambda item: (item["order"], item["material_id"]))
        scope_blocks: list[dict] = []
        for start in range(0, len(entries), width):
            block_id = f"{dataset}|{scope}|block-{start // width:03d}"
            members = entries[start : start + width]
            seg_ids = [s for m in members for s in m["segments"]]
            scope_blocks.append({"block_id": block_id, "segments": seg_ids})

        if len(scope_blocks) >= 2 and 0.0 < float(calibration_fraction) < 1.0:
            rng_seed = int(_digest_json([partition_seed, dataset, scope])[:12], 16)
            rng = np.random.default_rng(rng_seed)
            n_cal = max(
                1,
                min(
                    len(scope_blocks) - 1,
                    int(round(len(scope_blocks) * float(calibration_fraction))),
                ),
            )
            cal_indices = set(rng.choice(len(scope_blocks), n_cal, replace=False).tolist())
        else:
            cal_indices = set(range(len(scope_blocks)))

        for blk_idx, blk in enumerate(scope_blocks):
            part = "calibration" if blk_idx in cal_indices else "evaluation"
            for seg_id in blk["segments"]:
                segments[seg_id]["block_id"] = blk["block_id"]
                segments[seg_id]["partition"] = part

    return segments


def _locate_stage_checkpoint(
    dataset: str,
    stage: str,
    subset_tag: str,
    seed: int,
    *,
    exp_tag: str | None = None,
    fallback_exp_tags: tuple[str | None, ...] = (),
    explicit_checkpoint: str | Path | None = None,
    fallback_checkpoint: str | Path | None = None,
) -> Path | None:
    """定位指定阶段（word / context / revision）的训练检查点文件。"""
    if explicit_checkpoint is not None:
        path = Path(explicit_checkpoint).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"显式指定的 {stage} 阶段检查点不存在：{path}")
        return path

    candidate_tags: list[str | None] = [exp_tag]
    for fb_tag in fallback_exp_tags:
        if fb_tag not in candidate_tags:
            candidate_tags.append(fb_tag)

    for tag_cand in candidate_tags:
        for mode in ("train", "smoke"):
            stage_dir = workflow_output_directory(
                dataset,
                stage,
                subset_tag,
                seed,
                mode=mode,
                exp_tag=tag_cand,
                ignore_implicit_tag=(tag_cand == ""),
            )
            for name in ("best.ckpt", "best.pt", "last.ckpt"):
                candidate = stage_dir / name
                if candidate.is_file():
                    return candidate

    if fallback_checkpoint is not None:
        fb = Path(fallback_checkpoint).resolve()
        if fb.is_file():
            return fb
    return None


def _encode_stage_with_shifts(
    cfg: dict,
    eval_table,
    ckpt_file: Path,
    *,
    stage: str,
    split: str,
    device: torch.device,
    background_shift_seconds: tuple[float, ...],
    control_shift_seconds: float,
    subtraction_eta: float,
    show_progress: bool = True,
) -> dict:
    """加载单个阶段的检查点，提取 Clean (0s) 与 Shifted (-75s)（含 6 窗口历史共模扣除）特征。"""
    from braindecoding.models import build_brain_embedding_model

    dataset = cfg["experiment"]["dataset"]
    tc = cfg["training"]
    task = pallier_task if dataset == "pallier2025" else chinese_task
    build_fn = task.build_dataset
    grp_col = group_column(cfg)
    eval_bs = int(tc.get("eval_batch_size", tc["batch_size"]))

    checkpoint = load_checkpoint(ckpt_file, map_location="cpu")
    if not bool(cfg["workflow"].get("window_explicitly_overridden", False)):
        ckpt_cfg_ds = (
            checkpoint.get("config", {}).get("dataset", {})
            if isinstance(checkpoint.get("config"), dict)
            else {}
        )
        ckpt_win = checkpoint.get(
            "window_seconds",
            ckpt_cfg_ds.get("window_seconds") if isinstance(ckpt_cfg_ds, dict) else None,
        )
        ckpt_off = checkpoint.get(
            "window_start_offset_seconds",
            ckpt_cfg_ds.get("window_start_offset_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        if ckpt_win is not None:
            cfg["dataset"]["window_seconds"] = float(ckpt_win)
        if ckpt_off is not None:
            cfg["dataset"]["window_start_offset_seconds"] = float(ckpt_off)

    clean_ds = build_fn(cfg, eval_table)
    clean_loader, _ = make_loader(
        clean_ds, tc, shuffle=False, group_column=grp_col, batch_size=eval_bs
    )
    selected_subjects = cfg["dataset"]["subjects"]
    model_cfg = checkpoint.get("model_config", cfg["model"])
    ckpt_subjects = list(checkpoint.get("subject_ids", selected_subjects))
    subject_count = max(len(ckpt_subjects), clean_ds.subject_count)

    model = build_brain_embedding_model(
        clean_ds.channel_count,
        clean_ds.channel_positions,
        subject_count,
        model_cfg,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    amp_enabled = bool(tc.get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(tc)
    ds_table = getattr(clean_ds, "table", eval_table)
    event_col = "event_id" if "event_id" in ds_table.columns else "事件编号"
    canonical_event_ids = [
        str(e) for e in ds_table[event_col].astype(str).tolist()
    ]

    def _align_encoded_to_canonical_events(encoded: dict) -> dict:
        actual_ids = [str(e) for e in encoded["event_ids"]]
        if actual_ids == canonical_event_ids:
            return encoded
        if len(actual_ids) != len(canonical_event_ids) or set(actual_ids) != set(
            canonical_event_ids
        ):
            raise ValueError(
                f"Stage {stage} 编码输出的事件集合与 eval_table 不一致："
                f"count={len(actual_ids)} vs {len(canonical_event_ids)}。"
            )
        id_to_pos = {eid: pos for pos, eid in enumerate(actual_ids)}
        perm = torch.tensor(
            [id_to_pos[eid] for eid in canonical_event_ids], dtype=torch.long
        )
        aligned = dict(encoded)
        aligned["event_ids"] = list(canonical_event_ids)
        if "predictions" in aligned and isinstance(
            aligned["predictions"], torch.Tensor
        ):
            aligned["predictions"] = aligned["predictions"][perm]
        if "targets" in aligned and isinstance(aligned["targets"], torch.Tensor):
            aligned["targets"] = aligned["targets"][perm]
        if " words" in aligned or "words" in aligned:
            if "words" in aligned and isinstance(aligned["words"], list):
                aligned["words"] = [aligned["words"][int(i)] for i in perm]
        if "revision_predictions" in aligned and isinstance(
            aligned["revision_predictions"], dict
        ):
            aligned["revision_predictions"] = {
                delay: mat[perm]
                for delay, mat in aligned["revision_predictions"].items()
            }
        if "revision_availability_masks" in aligned and isinstance(
            aligned["revision_availability_masks"], dict
        ):
            aligned["revision_availability_masks"] = {
                delay: mask[perm]
                for delay, mask in aligned["revision_availability_masks"].items()
            }
        return aligned

    def _run_encode(shift_sec: float | None, label: str) -> dict:
        if shift_sec is None or abs(float(shift_sec)) <= 1e-9:
            ds_obj, loader_obj = clean_ds, clean_loader
        else:
            ds_obj = build_fn(
                cfg, eval_table, temporal_shift_seconds=float(shift_sec)
            )
            loader_obj, _ = make_loader(
                ds_obj, tc, shuffle=False, group_column=grp_col, batch_size=eval_bs
            )
        raw_encoded = encode_loader(
            model,
            loader_obj,
            device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            show_progress=show_progress,
            progress_desc=f"S3 Export [{stage} | {split} | {label}]",
        )
        return _align_encoded_to_canonical_events(raw_encoded)

    clean_encoded = _run_encode(0.0, "Clean 0s")
    control_encoded = _run_encode(
        float(control_shift_seconds), f"Shifted {control_shift_seconds:+.0f}s"
    )

    use_cms = bool(background_shift_seconds) and abs(float(subtraction_eta)) > 1e-12
    hist_encoded_list: list[dict] = []
    if use_cms:
        for shift_sec in background_shift_seconds:
            hist_encoded_list.append(
                _run_encode(float(shift_sec), f"CMS Hist {float(shift_sec):+.0f}s")
            )

    del model, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if stage in ("revision", "prefix_revision") and "revision_predictions" in clean_encoded:
        clean_states: dict[str, torch.Tensor] = {}
        shifted_states: dict[str, torch.Tensor] = {}
        avail_masks: dict[str, torch.Tensor] = {}
        for delay in sorted(clean_encoded["revision_predictions"]):
            state_key = f"K{delay}"
            c_raw = clean_encoded["revision_predictions"][delay].cpu()
            s_raw = control_encoded["revision_predictions"][delay].cpu()
            h_raw = [
                h["revision_predictions"][delay].cpu() for h in hist_encoded_list
            ]
            clean_states[state_key] = apply_historical_common_mode_subtraction(
                c_raw, h_raw, eta=subtraction_eta
            )
            shifted_states[state_key] = apply_historical_common_mode_subtraction(
                s_raw, h_raw, eta=subtraction_eta
            )
            avail_masks[state_key] = (
                clean_encoded["revision_availability_masks"][delay].cpu().bool()
            )
    else:
        state_key = "S0" if stage == "word" else "F"
        c_raw = clean_encoded["predictions"].cpu()
        s_raw = control_encoded["predictions"].cpu()
        h_raw = [h["predictions"].cpu() for h in hist_encoded_list]
        clean_states = {
            state_key: apply_historical_common_mode_subtraction(
                c_raw, h_raw, eta=subtraction_eta
            )
        }
        shifted_states = {
            state_key: apply_historical_common_mode_subtraction(
                s_raw, h_raw, eta=subtraction_eta
            )
        }
        avail_masks = {
            state_key: torch.ones(len(clean_encoded["event_ids"]), dtype=torch.bool)
        }

    return {
        "event_ids": [str(e) for e in clean_encoded["event_ids"]],
        "clean_embeddings": clean_states,
        "shifted_embeddings": shifted_states,
        "availability_masks": avail_masks,
        "checkpoint_path": str(ckpt_file),
    }


def export_neural_vocabulary_scores(
    dataset: str = "pallier2025",
    *,
    split: str = "val",
    seed: int = 0,
    subject: str | None = "all",
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    context_exp_tag: str | None = None,
    revision_exp_tag: str | None = None,
    word_checkpoint: str | Path | None = None,
    context_checkpoint: str | Path | None = None,
    revision_checkpoint: str | Path | None = None,
    revision_stage: str = "auto",
    config_path: str | Path | None = None,
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
    background_shift_seconds: Sequence[float] = DEFAULT_BACKGROUND_SHIFT_SECONDS,
    control_shift_seconds: float = DEFAULT_CONTROL_SHIFT_SECONDS,
    subtraction_eta: float = DEFAULT_SUBTRACTION_ETA,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    precision: str | None = None,
    require_cache: bool = False,
    max_eval_rows: int | None = None,
    output_path: str | Path | None = None,
    device: str = "auto",
    show_progress: bool = True,
) -> Path:
    """导出 S0, K0, K1, K2, F 在全训练词表 V 上的 Clean (0s) 与 Shifted (-75s) [N, |V|] 余弦得分 .pt 文件。"""
    ensure_default_environment()
    if split not in {"val", "test"}:
        raise ValueError(f"split 仅支持 'val' 或 'test'，当前为：{split!r}")

    rev_tag = revision_exp_tag if revision_exp_tag is not None else exp_tag
    ctx_tag = context_exp_tag if context_exp_tag is not None else exp_tag
    w_tag = word_exp_tag if word_exp_tag is not None else exp_tag

    rev_stage_mode = str(revision_stage or "auto").strip().lower()
    if rev_stage_mode not in {"auto", "revision", "prefix_revision"}:
        raise ValueError(
            f"revision_stage 仅支持 'auto'、'revision' 或 'prefix_revision'，当前为：{revision_stage!r}"
        )

    matched_rev_cfg_path: Path | None = None
    inferred_rev_stage: str | None = None
    if config_path is not None:
        resolved_cfg_p = resolve_config_path(project_path(config_path))
        if resolved_cfg_p.is_file():
            raw_yaml = load_yaml_with_extends(resolved_cfg_p)
            if isinstance(raw_yaml, dict) and any(
                k in raw_yaml for k in ("workflow", "model", "siamese_debiased", "training")
            ):
                inferred_ds, inferred_stage = _infer_dataset_and_stage_from_yaml(raw_yaml)
                dataset = inferred_ds
                if rev_stage_mode == "auto" and inferred_stage in {"revision", "prefix_revision"}:
                    rev_stage_mode = inferred_stage
                if inferred_stage in {"revision", "prefix_revision"}:
                    matched_rev_cfg_path = resolved_cfg_p
                    inferred_rev_stage = inferred_stage

    initial_rev_stage = (
        "prefix_revision" if rev_stage_mode == "prefix_revision" else "revision"
    )

    rev_cfg, subset_spec = build_workflow_config(
        dataset,
        stage=initial_rev_stage,
        seed=seed,
        subject=subject,
        num_subjects=num_subjects,
        subjects=subjects,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        precision=precision,
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        mode="train",
        exp_tag=rev_tag,
        word_exp_tag=w_tag,
        config_path=(
            matched_rev_cfg_path
            if (matched_rev_cfg_path is not None and inferred_rev_stage == initial_rev_stage)
            else None
        ),
    )
    dataset = subset_spec["dataset"]
    subset_tag = subset_spec["subset_tag"]

    if require_cache:
        verify_cache_ready(rev_cfg, splits=("train", split))
    else:
        try:
            verify_cache_ready(rev_cfg, splits=("train", split))
        except FileNotFoundError:
            run_cache_workflow(
                dataset,
                subject=subject,
                num_subjects=num_subjects,
                subjects=subjects,
                include_test=(split == "test"),
                force=False,
            )
            verify_cache_ready(rev_cfg, splits=("train", split))

    set_seed(int(seed))
    torch_device = choose_device(device)
    if torch_device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    task = pallier_task if dataset == "pallier2025" else chinese_task
    read_fn = task.dataset_module.load_event_table
    selected_subjects = rev_cfg["dataset"]["subjects"]

    train_table = read_fn(
        rev_cfg["cache"]["event_table"],
        split="train",
        subjects=selected_subjects,
        trainable_only=True,
    )
    eval_table = read_fn(
        rev_cfg["cache"]["event_table"],
        split=split,
        subjects=selected_subjects,
        trainable_only=True,
    )
    if max_eval_rows is not None:
        eval_table = limit_rows(eval_table, int(max_eval_rows))
    if len(eval_table) <= 0:
        raise ValueError(
            f"数据集 {dataset} 在 split={split!r} 与受试者 {selected_subjects} 上的样本为空。"
        )

    word_col = "normalized_word" if "normalized_word" in train_table.columns else "标准词"
    vocabulary = sorted(set(train_table[word_col].astype(str).tolist()))
    if dataset == "pallier2025":
        run_c = "运行编号" if "运行编号" in train_table.columns else "recording_id"
        bids_c = "BIDS事件行号" if "BIDS事件行号" in train_table.columns else "event_index"
        dedup_train = (
            train_table.drop_duplicates([run_c, bids_c])
            if (run_c in train_table.columns and bids_c in train_table.columns)
            else train_table
        )
    else:
        voice_c = "音频版本" if "音频版本" in train_table.columns else "voice_version"
        src_w_c = "源词编号" if "源词编号" in train_table.columns else "source_word_id"
        dedup_train = (
            train_table.drop_duplicates([voice_c, src_w_c])
            if (voice_c in train_table.columns and src_w_c in train_table.columns)
            else train_table
        )
    train_word_series = dedup_train[word_col].astype(str).value_counts().to_dict()
    vocabulary_train_counts = {
        str(w): int(train_word_series.get(str(w), 1)) for w in vocabulary
    }
    eval_ds = task.build_dataset(rev_cfg, eval_table)

    vocab_vectors = np.stack(
        [np.asarray(eval_ds.embedding_map[w], dtype=np.float32) for w in vocabulary],
        axis=0,
    )
    vocab_tensor = _l2_normalize_tensor(torch.from_numpy(vocab_vectors))

    resolved_rev_stage = initial_rev_stage
    rev_ckpt = _locate_stage_checkpoint(
        dataset,
        resolved_rev_stage,
        subset_tag,
        seed,
        exp_tag=rev_tag,
        explicit_checkpoint=revision_checkpoint,
    )
    if rev_ckpt is None and rev_stage_mode == "auto":
        alt_ckpt = _locate_stage_checkpoint(
            dataset,
            "prefix_revision",
            subset_tag,
            seed,
            exp_tag=rev_tag,
            explicit_checkpoint=None,
        )
        if alt_ckpt is not None:
            resolved_rev_stage = "prefix_revision"
            rev_ckpt = alt_ckpt
            rev_cfg, _ = build_workflow_config(
                dataset,
                stage="prefix_revision",
                seed=seed,
                subject=subject,
                num_subjects=num_subjects,
                subjects=subjects,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                precision=precision,
                window_seconds=window_seconds,
                window_start_offset_seconds=window_start_offset_seconds,
                mode="train",
                exp_tag=rev_tag,
                word_exp_tag=w_tag,
            )
    warmstart_fallback = rev_cfg["training"].get("pretrained_brain_encoder_checkpoint")
    word_ckpt = _locate_stage_checkpoint(
        dataset,
        "word",
        subset_tag,
        seed,
        exp_tag=w_tag,
        fallback_exp_tags=(rev_tag, ""),
        explicit_checkpoint=word_checkpoint,
        fallback_checkpoint=warmstart_fallback,
    )
    ctx_ckpt = _locate_stage_checkpoint(
        dataset,
        "context",
        subset_tag,
        seed,
        exp_tag=ctx_tag,
        explicit_checkpoint=context_checkpoint,
    )
    if rev_ckpt is None and word_ckpt is None:
        raise FileNotFoundError(
            f"未找到 Stage 2.2/2.3 ({resolved_rev_stage}) 或 Stage 1 (word) 检查点（dataset={dataset}, subset={subset_tag}, seed={seed}）。"
        )

    bg_shifts = tuple(float(s) for s in background_shift_seconds)
    ctrl_shift = float(control_shift_seconds)
    eta_val = float(subtraction_eta)

    started = time.perf_counter()
    if rev_ckpt is not None:
        rev_result = _encode_stage_with_shifts(
            rev_cfg,
            eval_table,
            rev_ckpt,
            stage=resolved_rev_stage,
            split=split,
            device=torch_device,
            background_shift_seconds=bg_shifts,
            control_shift_seconds=ctrl_shift,
            subtraction_eta=eta_val,
            show_progress=show_progress,
        )

        event_ids = rev_result["event_ids"]
        clean_embeddings: dict[str, torch.Tensor] = dict(rev_result["clean_embeddings"])
        shifted_embeddings: dict[str, torch.Tensor] = dict(rev_result["shifted_embeddings"])
        availability_masks: dict[str, torch.Tensor] = dict(rev_result["availability_masks"])
        checkpoint_provenance: dict[str, str] = {
            "revision": rev_result["checkpoint_path"],
            "revision_stage": resolved_rev_stage,
        }

        if word_ckpt is not None:
            word_cfg, _ = build_workflow_config(
                dataset,
                stage="word",
                seed=seed,
                subject=subject,
                num_subjects=num_subjects,
                subjects=subjects,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                precision=precision,
                window_seconds=window_seconds,
                window_start_offset_seconds=window_start_offset_seconds,
                mode="train",
                exp_tag=w_tag,
            )
            word_res = _encode_stage_with_shifts(
                word_cfg,
                eval_table,
                word_ckpt,
                stage="word",
                split=split,
                device=torch_device,
                background_shift_seconds=bg_shifts,
                control_shift_seconds=ctrl_shift,
                subtraction_eta=eta_val,
                show_progress=show_progress,
            )
            if word_res["event_ids"] != event_ids:
                raise ValueError("Stage 1 (word) 与 Stage 2.2 (revision) 的编码事件顺序不一致。")
            clean_embeddings.update(word_res["clean_embeddings"])
            shifted_embeddings.update(word_res["shifted_embeddings"])
            availability_masks.update(word_res["availability_masks"])
            checkpoint_provenance["word"] = word_res["checkpoint_path"]
        else:
            clean_embeddings["S0"] = clean_embeddings["K0"].clone()
            shifted_embeddings["S0"] = shifted_embeddings["K0"].clone()
            availability_masks["S0"] = availability_masks["K0"].clone()
            checkpoint_provenance["word"] = "fallback_from_K0"
    else:
        assert word_ckpt is not None
        word_cfg, _ = build_workflow_config(
            dataset,
            stage="word",
            seed=seed,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            precision=precision,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            mode="train",
            exp_tag=w_tag,
        )
        word_res = _encode_stage_with_shifts(
            word_cfg,
            eval_table,
            word_ckpt,
            stage="word",
            split=split,
            device=torch_device,
            background_shift_seconds=bg_shifts,
            control_shift_seconds=ctrl_shift,
            subtraction_eta=eta_val,
            show_progress=show_progress,
        )
        event_ids = word_res["event_ids"]
        clean_embeddings = dict(word_res["clean_embeddings"])
        shifted_embeddings = dict(word_res["shifted_embeddings"])
        availability_masks = dict(word_res["availability_masks"])
        for h_state in ("K0", "K1", "K2"):
            clean_embeddings[h_state] = clean_embeddings["S0"].clone()
            shifted_embeddings[h_state] = shifted_embeddings["S0"].clone()
            availability_masks[h_state] = availability_masks["S0"].clone()
        checkpoint_provenance = {
            "word": word_res["checkpoint_path"],
            "revision": "fallback_from_word_S0",
        }

    if ctx_ckpt is not None:
        ctx_cfg, _ = build_workflow_config(
            dataset,
            stage="context",
            seed=seed,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            precision=precision,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            mode="train",
            exp_tag=ctx_tag,
            word_exp_tag=w_tag,
        )
        ctx_res = _encode_stage_with_shifts(
            ctx_cfg,
            eval_table,
            ctx_ckpt,
            stage="context",
            split=split,
            device=torch_device,
            background_shift_seconds=bg_shifts,
            control_shift_seconds=ctrl_shift,
            subtraction_eta=eta_val,
            show_progress=show_progress,
        )
        if ctx_res["event_ids"] != event_ids:
            raise ValueError("Stage 2.1 (context) 与 Stage 2.2 (revision) 的编码事件顺序不一致。")
        clean_embeddings.update(ctx_res["clean_embeddings"])
        shifted_embeddings.update(ctx_res["shifted_embeddings"])
        availability_masks.update(ctx_res["availability_masks"])
        checkpoint_provenance["context"] = ctx_res["checkpoint_path"]
        checkpoint_provenance["has_context_checkpoint"] = True
    else:
        # 未单独训练 Stage 2.1 全局上下文检查点时标记为 not_available（保持张量键兼容但禁止下游解码伪条件 F-local）
        clean_embeddings["F"] = clean_embeddings["K2"].clone()
        shifted_embeddings["F"] = shifted_embeddings["K2"].clone()
        availability_masks["F"] = availability_masks["K2"].clone()
        checkpoint_provenance["context"] = "not_available"
        checkpoint_provenance["has_context_checkpoint"] = False

    clean_scores: dict[str, torch.Tensor] = {}
    shifted_scores: dict[str, torch.Tensor] = {}
    for state in CANONICAL_STATES:
        clean_scores[state] = (clean_embeddings[state].float() @ vocab_tensor.T).cpu()
        shifted_scores[state] = (shifted_embeddings[state].float() @ vocab_tensor.T).cpu()

    canonical_table = getattr(eval_ds, "table", eval_table)
    segments = build_segment_inventory(dataset, canonical_table, vocabulary)
    eval_word_col = (
        "normalized_word"
        if "normalized_word" in canonical_table.columns
        else "标准词"
    )
    subj_col = (
        "subject_id" if "subject_id" in canonical_table.columns else "受试者"
    )
    rec_col = (
        "recording_id" if "recording_id" in canonical_table.columns else "记录编号"
    )
    ref_words = canonical_table[eval_word_col].astype(str).tolist()
    vocab_set = set(vocabulary)
    in_vocab_mask = torch.tensor([w in vocab_set for w in ref_words], dtype=torch.bool)
    language = "fr" if dataset == "pallier2025" else "zh"

    if output_path is None:
        base_out = (
            results_root()
            / dataset
            / (rev_tag if rev_tag else "sequence_decoding")
            / "s3_protocol"
            / subset_tag
            / f"seed-{int(seed):03d}"
        )
        out_file = base_out / f"s3_neural_scores_{split}.pt"
    else:
        out_file = Path(output_path).resolve()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": "sequence_decoding_s3",
        "dataset": dataset,
        "language": language,
        "split": split,
        "seed": int(seed),
        "subset": subset_spec,
        "states": list(CANONICAL_STATES),
        "vocabulary": vocabulary,
        "vocabulary_size": len(vocabulary),
        "vocabulary_train_counts": vocabulary_train_counts,
        "vocabulary_embeddings": vocab_tensor.cpu(),
        "event_ids": event_ids,
        "reference_words": ref_words,
        "subject_ids": canonical_table[subj_col].astype(str).tolist(),
        "recording_ids": canonical_table[rec_col].astype(str).tolist(),
        "in_vocabulary_mask": in_vocab_mask,
        "in_vocabulary_rate": float(in_vocab_mask.float().mean().item()),
        "availability_masks": availability_masks,
        "clean_scores": clean_scores,
        "shifted_scores": shifted_scores,
        "segments": segments,
        "cms_metadata": {
            "enabled": bool(bg_shifts) and abs(eta_val) > 1e-12,
            "background_shift_seconds": list(bg_shifts),
            "control_shift_seconds": ctrl_shift,
            "subtraction_eta": eta_val,
        },
        "checkpoints": checkpoint_provenance,
        "checkpoint_provenance": checkpoint_provenance,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    tmp_file = out_file.with_suffix(out_file.suffix + ".tmp")
    torch.save(payload, tmp_file)
    tmp_file.replace(out_file)

    sidecar = out_file.with_suffix(".json")
    save_json(
        sidecar,
        {
            "status": "exported",
            "dataset": dataset,
            "language": language,
            "split": split,
            "seed": int(seed),
            "subset_tag": subset_tag,
            "events": len(event_ids),
            "segments": len(segments),
            "vocabulary_size": len(vocabulary),
            "in_vocabulary_rate": payload["in_vocabulary_rate"],
            "states": list(CANONICAL_STATES),
            "cms_metadata": payload["cms_metadata"],
            "checkpoints": checkpoint_provenance,
            "checkpoint_provenance": checkpoint_provenance,
            "pt_file": str(out_file),
            "elapsed_seconds": payload["elapsed_seconds"],
        },
    )
    return out_file


def resolve_vocabulary_frequency_order(
    exported_payload: dict,
) -> tuple[list[int], list[str], dict[str, int]]:
    """返回按训练集词频降序、字典序升序排列的词表列索引、词列表与训练词频字典。"""
    vocabulary = [str(w) for w in exported_payload.get("vocabulary") or []]
    if not vocabulary:
        return [], [], {}

    dataset = str(exported_payload.get("dataset") or "pallier2025")
    saved_counts = exported_payload.get("vocabulary_train_counts")
    counts: dict[str, int] = {}

    if isinstance(saved_counts, dict) and saved_counts:
        counts = {str(k): int(v) for k, v in saved_counts.items()}
    elif isinstance(saved_counts, (list, tuple)) and len(saved_counts) == len(vocabulary):
        counts = {
            str(vocabulary[i]): int(saved_counts[i])
            for i in range(len(vocabulary))
        }

    if not counts and dataset == "pallier2025":
        manifest_path = project_path(
            "data/manifests/rolling_sentence/pallier2025_train_vocabulary.json"
        )
        if manifest_path.is_file():
            try:
                manifest_obj = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                mat_freqs = manifest_obj.get("material_frequencies") or {}
                if isinstance(mat_freqs, dict) and mat_freqs:
                    counts = {str(k): int(v) for k, v in mat_freqs.items()}
            except Exception:
                counts = {}

    if not counts:
        try:
            task = pallier_task if dataset == "pallier2025" else chinese_task
            from braindecoding.data.derived_cache import canonical_cache_paths

            evt_path = Path(canonical_cache_paths(dataset)["event_table"])
            if evt_path.is_file():
                train_df = task.dataset_module.load_event_table(
                    evt_path, split="train", trainable_only=True
                )
                w_col = (
                    "normalized_word"
                    if "normalized_word" in train_df.columns
                    else "标准词"
                )
                if dataset == "pallier2025":
                    r_col = (
                        "运行编号"
                        if "运行编号" in train_df.columns
                        else "recording_id"
                    )
                    b_col = (
                        "BIDS事件行号"
                        if "BIDS事件行号" in train_df.columns
                        else "event_index"
                    )
                    if r_col in train_df.columns and b_col in train_df.columns:
                        train_df = train_df.drop_duplicates([r_col, b_col])
                else:
                    v_col = (
                        "音频版本"
                        if "音频版本" in train_df.columns
                        else "voice_version"
                    )
                    s_col = (
                        "源词编号"
                        if "源词编号" in train_df.columns
                        else "source_word_id"
                    )
                    if v_col in train_df.columns and s_col in train_df.columns:
                        train_df = train_df.drop_duplicates([v_col, s_col])
                counts = {
                    str(k): int(v)
                    for k, v in train_df[w_col]
                    .astype(str)
                    .value_counts()
                    .to_dict()
                    .items()
                }
        except Exception:
            counts = {}

    if not counts:
        manifest_dir = (
            "pallier2025" if dataset == "pallier2025" else "chineseeeg2"
        )
        n150_path = project_path(
            f"data/manifests/{manifest_dir}/vocabulary_N150.json"
        )
        if n150_path.is_file():
            try:
                n150_obj = json.loads(n150_path.read_text(encoding="utf-8"))
                wc = n150_obj.get("word_counts") or {}
                if isinstance(wc, dict) and wc:
                    counts.update({str(k): int(v) * 1000 for k, v in wc.items()})
                elif isinstance(n150_obj.get("vocabulary"), list):
                    for rank_i, w_str in enumerate(n150_obj["vocabulary"]):
                        counts[str(w_str)] = (200 - rank_i) * 1000
            except Exception:
                pass
        for seg in exported_payload.get("segments") or []:
            for w_str in seg.get("reference_words") or []:
                w_s = str(w_str)
                if w_s not in counts:
                    counts[w_s] = 0
                counts[w_s] += 1

    ranked_indices = sorted(
        range(len(vocabulary)),
        key=lambda idx: (-int(counts.get(vocabulary[idx], 0)), vocabulary[idx]),
    )
    ranked_words = [vocabulary[idx] for idx in ranked_indices]
    full_counts = {w: int(counts.get(w, 0)) for w in vocabulary}
    return ranked_indices, ranked_words, full_counts


def load_neural_vocabulary_scores(path: str | Path) -> dict:
    """加载并校验导出的 S3 神经词表得分 .pt 文件。"""
    pt_path = Path(path).resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"S3 神经特征文件不存在：{pt_path}")
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    required_keys = {
        "dataset",
        "language",
        "split",
        "vocabulary",
        "clean_scores",
        "shifted_scores",
        "segments",
    }
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"S3 神经特征文件缺少必要字段 {sorted(missing)}：{pt_path}")
    return payload
