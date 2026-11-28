"""冻结 Stage-1 CNN 特征预提取、缓存键计算、静态视界表构建与 GPU 驻留打包。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from braindecoding.config import cache_root
from braindecoding.training.debiased_losses import (
    DEFAULT_CAUSAL_PREROLL_SHIFT_SECONDS,
    DEFAULT_SIAMESE_SHIFT_SECONDS,
    _normalize_shift_mode,
)
from braindecoding.training.word_encoder_training import (
    _wrap_progress,
    resolve_amp_dtype,
)

def _extract_single_dataset_cnn_embeddings(
    model,
    dataset_obj,
    *,
    device: torch.device,
    batch_size: int = 512,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    temporal_shift_seconds: float = 0.0,
    shift_mode: str = "deterministic_circular",
    preroll_seconds: float = 45.0,
    show_progress: bool = True,
    progress_desc: str = "Pre-extract CNN",
    out_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """在固定 temporal_shift_seconds 下按原始行序提取 Stage-1 CNN 归一化输出向量 [N, D]。"""
    model.brain_encoder.eval()
    orig_shift_sec = getattr(dataset_obj, "temporal_shift_seconds", 0.0)
    orig_shift_samp = getattr(dataset_obj, "temporal_shift_samples", 0)
    orig_shift_mode = getattr(dataset_obj, "shift_mode", "deterministic_circular")
    orig_preroll_sec = getattr(dataset_obj, "preroll_seconds", 45.0)
    sr_hz = float(getattr(dataset_obj, "sampling_rate_hz", 50.0))
    norm_mode = _normalize_shift_mode(shift_mode) or "deterministic_circular"

    try:
        if hasattr(dataset_obj, "shift_mode"):
            dataset_obj.shift_mode = norm_mode
        if hasattr(dataset_obj, "preroll_seconds"):
            dataset_obj.preroll_seconds = float(preroll_seconds)
        if hasattr(dataset_obj, "temporal_shift_seconds"):
            dataset_obj.temporal_shift_seconds = float(temporal_shift_seconds)
        if hasattr(dataset_obj, "temporal_shift_samples"):
            dataset_obj.temporal_shift_samples = int(
                round(float(temporal_shift_seconds) * sr_hz)
            )
        if (
            abs(float(temporal_shift_seconds)) > 1e-9
            and norm_mode == "preroll"
            and hasattr(dataset_obj, "ensure_preroll_buffers")
        ):
            dataset_obj.ensure_preroll_buffers(
                float(temporal_shift_seconds),
                shift_mode="preroll",
                preroll_seconds=float(preroll_seconds),
            )

        if device.type == "cuda" and hasattr(
            dataset_obj, "extract_cnn_embeddings_gpu"
        ):
            return dataset_obj.extract_cnn_embeddings_gpu(
                model,
                device=device,
                batch_size=int(batch_size),
                amp=bool(amp),
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=progress_desc,
                progress_wrap=_wrap_progress,
                out_tensor=out_tensor,
            )

        loader = DataLoader(
            dataset_obj,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        use_amp = (
            bool(amp)
            and device.type == "cuda"
            and amp_dtype in {torch.float16, torch.bfloat16}
        )
        chunks: list[torch.Tensor] = []
        iterable, pbar = _wrap_progress(
            loader,
            enabled=show_progress,
            desc=progress_desc,
            total=len(loader),
        )
        try:
            for batch in iterable:
                signal_key = "meg" if "meg" in batch else "eeg"
                signals = batch[signal_key].to(device, non_blocking=True)
                subject_indices = batch["subject_index"].to(
                    device, non_blocking=True
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    z = F.normalize(
                        model.brain_encoder(signals, subject_indices), dim=-1
                    )
                chunks.append(z.float().cpu())
        finally:
            if pbar is not None:
                pbar.close()
    finally:
        if hasattr(dataset_obj, "temporal_shift_seconds"):
            dataset_obj.temporal_shift_seconds = orig_shift_sec
        if hasattr(dataset_obj, "temporal_shift_samples"):
            dataset_obj.temporal_shift_samples = orig_shift_samp
        if hasattr(dataset_obj, "shift_mode"):
            dataset_obj.shift_mode = orig_shift_mode
        if hasattr(dataset_obj, "preroll_seconds"):
            dataset_obj.preroll_seconds = orig_preroll_sec

    result = torch.cat(chunks, dim=0)
    if out_tensor is not None:
        out_tensor.copy_(result, non_blocking=False)
        return out_tensor
    return result


def _extract_single_dataset_stage1_features(
    model,
    dataset_obj,
    *,
    device: torch.device,
    batch_size: int = 512,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    temporal_shift_seconds: float = 0.0,
    shift_mode: str = "deterministic_circular",
    preroll_seconds: float = 45.0,
    show_progress: bool = True,
    progress_desc: str = "Pre-extract Stage-1",
    return_pre_head: bool = True,
    return_raw_h_bins: bool = False,
    project_h_bins_fn=None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """单遍提取冻结 Stage-1 的 (z_norm, h_pre, h_bins_or_proj)，支持真实 MEG 与合成脉冲流 GPU 快路径。"""
    model.brain_encoder.eval()
    orig_shift_sec = getattr(dataset_obj, "temporal_shift_seconds", 0.0)
    orig_shift_samp = getattr(dataset_obj, "temporal_shift_samples", 0)
    orig_shift_mode = getattr(dataset_obj, "shift_mode", "deterministic_circular")
    orig_preroll_sec = getattr(dataset_obj, "preroll_seconds", 45.0)
    sr_hz = float(getattr(dataset_obj, "sampling_rate_hz", 50.0))
    norm_mode = _normalize_shift_mode(shift_mode) or "deterministic_circular"

    try:
        if hasattr(dataset_obj, "shift_mode"):
            dataset_obj.shift_mode = norm_mode
        if hasattr(dataset_obj, "preroll_seconds"):
            dataset_obj.preroll_seconds = float(preroll_seconds)
        if hasattr(dataset_obj, "temporal_shift_seconds"):
            dataset_obj.temporal_shift_seconds = float(temporal_shift_seconds)
        if hasattr(dataset_obj, "temporal_shift_samples"):
            dataset_obj.temporal_shift_samples = int(
                round(float(temporal_shift_seconds) * sr_hz)
            )
        if (
            abs(float(temporal_shift_seconds)) > 1e-9
            and norm_mode == "preroll"
            and hasattr(dataset_obj, "ensure_preroll_buffers")
        ):
            dataset_obj.ensure_preroll_buffers(
                float(temporal_shift_seconds),
                shift_mode="preroll",
                preroll_seconds=float(preroll_seconds),
            )

        if device.type == "cuda" and hasattr(
            dataset_obj, "extract_cnn_embeddings_gpu"
        ):
            return dataset_obj.extract_cnn_embeddings_gpu(
                model,
                device=device,
                batch_size=int(batch_size),
                amp=bool(amp),
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=progress_desc,
                progress_wrap=_wrap_progress,
                return_pre_head=bool(return_pre_head),
                return_raw_h_bins=bool(return_raw_h_bins),
                project_h_bins_fn=project_h_bins_fn,
            )

        loader = DataLoader(
            dataset_obj,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        use_amp = (
            bool(amp)
            and device.type == "cuda"
            and amp_dtype in {torch.float16, torch.bfloat16}
        )
        z_chunks: list[torch.Tensor] = []
        pre_chunks: list[torch.Tensor] = []
        bins_chunks: list[torch.Tensor] = []
        has_pre_head = hasattr(model.brain_encoder, "encode_with_pre_head")
        store_dtype = torch.bfloat16 if (use_amp and amp_dtype == torch.bfloat16) else torch.float32
        iterable, pbar = _wrap_progress(
            loader,
            enabled=show_progress,
            desc=progress_desc,
            total=len(loader),
        )
        try:
            with torch.no_grad():
                for batch in iterable:
                    signal_key = "meg" if "meg" in batch else "eeg"
                    signals = batch[signal_key].to(device, non_blocking=True)
                    subject_indices = batch["subject_index"].to(
                        device, non_blocking=True
                    )
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        if has_pre_head:
                            z_raw, h_pre, h_bins = (
                                model.brain_encoder.encode_with_pre_head(
                                    signals, subject_indices
                                )
                            )
                            z = F.normalize(z_raw, dim=-1)
                            if return_raw_h_bins:
                                h_bins_out = h_bins
                            elif project_h_bins_fn is not None:
                                h_bins_out = project_h_bins_fn(h_bins)
                            else:
                                h_bins_out = None
                        else:
                            z = F.normalize(
                                model.brain_encoder(signals, subject_indices),
                                dim=-1,
                            )
                            h_pre = z if return_pre_head else None
                            h_bins_out = None
                    z_chunks.append(z.to(device="cpu", dtype=store_dtype))
                    if return_pre_head and h_pre is not None:
                        pre_chunks.append(h_pre.to(device="cpu", dtype=store_dtype))
                    if h_bins_out is not None:
                        bins_chunks.append(
                            h_bins_out.to(device="cpu", dtype=store_dtype)
                        )
        finally:
            if pbar is not None:
                pbar.close()
    finally:
        if hasattr(dataset_obj, "temporal_shift_seconds"):
            dataset_obj.temporal_shift_seconds = orig_shift_sec
        if hasattr(dataset_obj, "temporal_shift_samples"):
            dataset_obj.temporal_shift_samples = orig_shift_samp
        if hasattr(dataset_obj, "shift_mode"):
            dataset_obj.shift_mode = orig_shift_mode
        if hasattr(dataset_obj, "preroll_seconds"):
            dataset_obj.preroll_seconds = orig_preroll_sec

    z_all = torch.cat(z_chunks, dim=0)
    pre_all = torch.cat(pre_chunks, dim=0) if pre_chunks else None
    bins_all = torch.cat(bins_chunks, dim=0) if bins_chunks else None
    return z_all, pre_all, bins_all


def _extract_static_dataset_metadata(
    dataset_obj,
    device: torch.device,
) -> dict:
    """提取 Dataset 行级静态张量（目标词嵌入、受试者索引、句组索引、词起始秒数等），免除训练期逐样本查表。"""
    onset_seconds: torch.Tensor
    if hasattr(dataset_obj, "_words") and hasattr(
        dataset_obj, "_embedding_tensors"
    ):
        words = list(dataset_obj._words)
        targets = torch.stack(
            [dataset_obj._embedding_tensors[w] for w in words], dim=0
        ).to(device=device, dtype=torch.float32)
        subject_indices = dataset_obj._subject_index_tensors.to(
            device=device, dtype=torch.long
        )
        sentence_indices = dataset_obj._sentence_index_tensors.to(
            device=device, dtype=torch.long
        )
        event_ids = list(dataset_obj._event_ids)
        recording_ids = (
            list(dataset_obj._recording_ids)
            if hasattr(dataset_obj, "_recording_ids")
            else [
                str(dataset_obj[idx].get("recording_id", ""))
                for idx in range(len(dataset_obj))
            ]
        )
        if hasattr(dataset_obj, "_onset_second_tensors"):
            onset_seconds = dataset_obj._onset_second_tensors.to(
                device=device, dtype=torch.float32
            )
        elif hasattr(dataset_obj, "_onset_seconds"):
            onset_seconds = torch.from_numpy(
                np.asarray(dataset_obj._onset_seconds, dtype=np.float32)
            ).to(device=device, dtype=torch.float32)
        elif hasattr(dataset_obj, "table") and "onset_seconds" in dataset_obj.table.columns:
            onset_seconds = torch.from_numpy(
                dataset_obj.table["onset_seconds"].to_numpy(dtype=np.float32)
            ).to(device=device, dtype=torch.float32)
        else:
            onset_seconds = torch.arange(
                len(words), device=device, dtype=torch.float32
            ) * 0.35
    else:
        words = []
        target_list = []
        subj_list = []
        sent_list = []
        onset_list = []
        event_ids = []
        recording_ids = []
        for idx in range(len(dataset_obj)):
            sample = dataset_obj[idx]
            words.append(str(sample["word"]))
            target_list.append(sample["text_embedding"].float())
            subj_list.append(int(sample["subject_index"]))
            sent_list.append(int(sample["sentence_index"]))
            onset_val = sample.get("onset_seconds")
            onset_list.append(
                float(onset_val.item() if isinstance(onset_val, torch.Tensor) else onset_val)
                if onset_val is not None
                else float(idx) * 0.35
            )
            event_ids.append(str(sample["event_id"]))
            recording_ids.append(str(sample.get("recording_id", "")))
        targets = torch.stack(target_list, dim=0).to(
            device=device, dtype=torch.float32
        )
        subject_indices = torch.as_tensor(
            subj_list, device=device, dtype=torch.long
        )
        sentence_indices = torch.as_tensor(
            sent_list, device=device, dtype=torch.long
        )
        onset_seconds = torch.as_tensor(
            onset_list, device=device, dtype=torch.float32
        )

    return {
        "targets": targets,
        "subject_indices": subject_indices,
        "sentence_indices": sentence_indices,
        "onset_seconds": onset_seconds,
        "words": words,
        "event_ids": event_ids,
        "recording_ids": recording_ids,
    }


def _preextracted_cache_path(
    *,
    dataset: str,
    subset_tag: str,
    word_ckpt_sha256: str,
    window_seconds: float,
    window_start_offset_seconds: float,
    shift_seconds: list[float],
    train_len: int,
    val_len: int,
    shift_mode: str = "deterministic_circular",
    preroll_seconds: float = 45.0,
    baseline_seconds: float = 0.5,
    causal_highpass_seconds: float = 0.0,
    causal_highpass_mode: str = "moving_average",
    causal_highpass_hz: float = 0.5,
    causal_highpass_order: int = 2,
    clamp: float = 5.0,
    feature_profile: str | None = None,
) -> Path:
    norm_mode = _normalize_shift_mode(shift_mode) or "deterministic_circular"
    hp_mode = str(causal_highpass_mode or "moving_average").strip().lower()
    sig_payload = {
        "version": "preextracted_frozen_cnn_v1",
        "dataset": str(dataset),
        "subset_tag": str(subset_tag),
        "word_ckpt_sha256": str(word_ckpt_sha256),
        "window_seconds": round(float(window_seconds), 6),
        "window_start_offset_seconds": round(
            float(window_start_offset_seconds), 6
        ),
        "shift_seconds": [round(float(s), 4) for s in shift_seconds],
        "train_len": int(train_len),
        "val_len": int(val_len),
    }
    if norm_mode != "deterministic_circular" and shift_seconds:
        sig_payload["shift_mode"] = norm_mode
        sig_payload["preroll_seconds"] = round(float(preroll_seconds), 4)
    if not np.isclose(float(baseline_seconds), 0.5, atol=1e-6):
        sig_payload["baseline_seconds"] = round(float(baseline_seconds), 6)
    if hp_mode != "moving_average":
        sig_payload["causal_highpass_mode"] = hp_mode
        sig_payload["causal_highpass_hz"] = round(float(causal_highpass_hz), 6)
        sig_payload["causal_highpass_order"] = int(causal_highpass_order)
    elif float(causal_highpass_seconds) > 1e-6:
        sig_payload["causal_highpass_seconds"] = round(
            float(causal_highpass_seconds), 6
        )
    if not np.isclose(float(clamp), 5.0, atol=1e-6):
        sig_payload["clamp"] = round(float(clamp), 6)
    if feature_profile:
        sig_payload["feature_profile"] = str(feature_profile)
    digest = hashlib.sha256(
        json.dumps(sig_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    win_tag = f"win{float(window_seconds):.2f}s".replace(".", "p")
    cache_dir = cache_root() / "preextracted_cnn" / str(dataset)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{subset_tag}_{win_tag}_{digest}.pt"


def _build_static_revision_train_table(
    model,
    sentence_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    """在训练开始前一次性构建全数据集行级局部修订窗口 GPU 静态查表 [N, H, L_max]。

    当 preserve_complete_groups=True 且 cutoff_policy='all' 时，每个训练步无需任何
    CPU/Python 循环即可直接在 GPU 显存中完成 Clean 与 Shifted 分支的 2D 窗口索引。
    """
    if not getattr(model, "prefix_revision_enabled", False):
        return None

    horizons = tuple(int(d) for d in getattr(model, "prefix_revision_horizons", (0, 1, 2)))
    num_horizons = len(horizons)
    history_radius = getattr(model, "prefix_revision_history_radius", 2)
    n_rows = int(len(sentence_indices))

    groups = model._ordered_groups(sentence_indices)
    max_win_len = (
        max((len(g) for g in groups), default=1)
        if history_radius is None
        else int(history_radius) + max(horizons, default=0) + 1
    )
    win_idx_np = np.full((n_rows, num_horizons, max_win_len), n_rows, dtype=np.int64)
    mask_np = np.zeros((n_rows, num_horizons, max_win_len), dtype=np.bool_)
    target_col_np = np.zeros((n_rows, num_horizons), dtype=np.int64)
    valid_delay_np = np.zeros((n_rows, num_horizons), dtype=np.bool_)
    unique_win_np = np.zeros((n_rows, num_horizons), dtype=np.bool_)
    group_pos_np = np.zeros(n_rows, dtype=np.int64)

    for indices in groups:
        g_len = len(indices)
        for cutoff, global_row in enumerate(indices):
            group_pos_np[global_row] = cutoff
            prev_bounds: tuple[int, int] | None = None
            for h_idx, delay in enumerate(horizons):
                target_pos = cutoff - delay
                if target_pos < 0:
                    continue
                start, stop, target_offset = model._revision_window_bounds(
                    target_pos, delay, g_len
                )
                w = indices[start:stop]
                w_len = len(w)
                win_idx_np[global_row, h_idx, :w_len] = w
                mask_np[global_row, h_idx, :w_len] = True
                target_col_np[global_row, h_idx] = int(target_offset)
                valid_delay_np[global_row, h_idx] = True
                if (start, stop) != prev_bounds:
                    unique_win_np[global_row, h_idx] = True
                    prev_bounds = (start, stop)

    idx_dtype = torch.int32 if n_rows < (1 << 31) - 1 else torch.long
    return {
        "row_win_global_idx": torch.from_numpy(win_idx_np).to(
            device=device, dtype=idx_dtype
        ),
        "row_win_mask": torch.from_numpy(mask_np).to(
            device=device, dtype=torch.bool
        ),
        "row_target_col": torch.from_numpy(target_col_np).to(
            device=device, dtype=torch.long
        ),
        "row_valid_delay": torch.from_numpy(valid_delay_np).to(
            device=device, dtype=torch.bool
        ),
        "row_is_unique_win": torch.from_numpy(unique_win_np).to(
            device=device, dtype=torch.bool
        ),
        "row_group_pos": torch.from_numpy(group_pos_np).to(
            device=device, dtype=torch.long
        ),
        "horizons_tensor": torch.as_tensor(
            horizons, device=device, dtype=torch.long
        ),
    }


def _build_static_revision_eval_table(
    model,
    sentence_indices: torch.Tensor,
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor]] | None:
    """在训练开始前一次性构建验证集/评估集各延迟视野的 GPU 静态窗口查表。"""
    if not getattr(model, "prefix_revision_enabled", False):
        return None

    horizons = tuple(int(d) for d in getattr(model, "prefix_revision_horizons", (0, 1, 2)))
    history_radius = getattr(model, "prefix_revision_history_radius", 2)
    n_rows = int(len(sentence_indices))
    groups = model._ordered_groups(sentence_indices)
    max_group_len = max((len(g) for g in groups), default=1)
    avail_masks = model.revision_availability_masks(sentence_indices)

    table_by_delay: dict[int, dict[str, torch.Tensor]] = {}
    for delay in horizons:
        max_w_len = (
            max_group_len
            if history_radius is None
            else int(history_radius) + int(delay) + 1
        )
        win_idx_np = np.full((n_rows, max_w_len), n_rows, dtype=np.int64)
        mask_np = np.zeros((n_rows, max_w_len), dtype=np.bool_)
        target_col_np = np.zeros(n_rows, dtype=np.int64)
        step_offset_np = np.zeros(n_rows, dtype=np.int64)

        for indices in groups:
            g_len = len(indices)
            for position, global_row in enumerate(indices):
                start, stop, target_offset = model._revision_window_bounds(
                    position, delay, g_len
                )
                w = indices[start:stop]
                w_len = len(w)
                win_idx_np[global_row, :w_len] = w
                mask_np[global_row, :w_len] = True
                target_col_np[global_row] = int(target_offset)
                step_offset_np[global_row] = int(stop - 1 - position)

        idx_dtype = torch.int32 if n_rows < (1 << 31) - 1 else torch.long
        table_by_delay[int(delay)] = {
            "win_global_idx": torch.from_numpy(win_idx_np).to(
                device=device, dtype=idx_dtype
            ),
            "win_mask": torch.from_numpy(mask_np).to(
                device=device, dtype=torch.bool
            ),
            "target_col": torch.from_numpy(target_col_np).to(
                device=device, dtype=torch.long
            ),
            "step_offset": torch.from_numpy(step_offset_np).to(
                device=device, dtype=torch.long
            ),
            "avail_mask": avail_masks[int(delay)].to(
                device=device, dtype=torch.bool
            ),
        }
    return table_by_delay


def _alloc_padded_gpu_feature_matrix(
    src_tensor: torch.Tensor,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在目标设备上单次分配带尾部零填充行的连续特征矩阵，返回 (零拷贝切片视图, 填充张量)。"""
    n_rows, feat_dim = int(src_tensor.shape[0]), int(src_tensor.shape[1])
    padded = torch.zeros(
        (n_rows + 1, feat_dim), device=device, dtype=dtype
    )
    padded[:n_rows].copy_(src_tensor, non_blocking=False)
    return padded[:n_rows], padded


def _finalize_gpu_preextracted_bundle(
    model,
    train_ds,
    val_ds,
    train_clean_z: torch.Tensor,
    train_shifted_z: dict[float, torch.Tensor],
    val_clean_z: torch.Tensor,
    device: torch.device,
    *,
    cache_file: Path | None,
    loaded_from_disk: bool,
    elapsed: float,
    feature_dtype: torch.dtype = torch.float32,
    train_clean_self: torch.Tensor | None = None,
    train_shifted_self: dict[float, torch.Tensor] | None = None,
    val_clean_self: torch.Tensor | None = None,
    session_calibrated: bool = False,
    train_meta: dict | None = None,
    val_meta: dict | None = None,
) -> dict:
    """将预提取张量、尾部零填充视图及全数据集静态 GPU 窗口查表常驻于 GPU 显存（零冗余单份存储）。"""
    if train_meta is None:
        train_meta = _extract_static_dataset_metadata(train_ds, device)
    if val_meta is None:
        val_meta = _extract_static_dataset_metadata(val_ds, device)

    train_clean_z, train_clean_z_pad = _alloc_padded_gpu_feature_matrix(
        train_clean_z, device, dtype=feature_dtype
    )
    val_clean_z, val_clean_z_pad = _alloc_padded_gpu_feature_matrix(
        val_clean_z, device, dtype=feature_dtype
    )
    train_shifted_z_gpu: dict[float, torch.Tensor] = {}
    train_shifted_z_pad: dict[float, torch.Tensor] = {}
    for k in list(train_shifted_z.keys()):
        v_src = train_shifted_z.pop(k)
        v_unpad, v_pad = _alloc_padded_gpu_feature_matrix(
            v_src, device, dtype=feature_dtype
        )
        del v_src
        train_shifted_z_gpu[float(k)] = v_unpad
        train_shifted_z_pad[float(k)] = v_pad
    train_shifted_z = train_shifted_z_gpu

    train_clean_self_gpu: torch.Tensor | None = None
    train_clean_self_pad: torch.Tensor | None = None
    val_clean_self_gpu: torch.Tensor | None = None
    val_clean_self_pad: torch.Tensor | None = None
    train_shifted_self_gpu: dict[float, torch.Tensor] = {}
    train_shifted_self_pad: dict[float, torch.Tensor] = {}
    if train_clean_self is not None:
        train_clean_self_gpu, train_clean_self_pad = (
            _alloc_padded_gpu_feature_matrix(
                train_clean_self, device, dtype=feature_dtype
            )
        )
    if val_clean_self is not None:
        val_clean_self_gpu, val_clean_self_pad = (
            _alloc_padded_gpu_feature_matrix(
                val_clean_self, device, dtype=feature_dtype
            )
        )
    if train_shifted_self:
        for k in list(train_shifted_self.keys()):
            s_src = train_shifted_self.pop(k)
            s_unpad, s_pad = _alloc_padded_gpu_feature_matrix(
                s_src, device, dtype=feature_dtype
            )
            del s_src
            train_shifted_self_gpu[float(k)] = s_unpad
            train_shifted_self_pad[float(k)] = s_pad

    train_static_rev = _build_static_revision_train_table(
        model, train_meta["sentence_indices"], device
    )
    val_static_rev = _build_static_revision_eval_table(
        model, val_meta["sentence_indices"], device
    )
    train_meta["clean_z_pad"] = train_clean_z_pad
    if train_clean_self_gpu is not None:
        train_meta["clean_self_all"] = train_clean_self_gpu
        train_meta["clean_self_pad"] = train_clean_self_pad
    val_meta["clean_z_pad"] = val_clean_z_pad
    val_meta["static_revision_table"] = val_static_rev
    if val_clean_self_gpu is not None:
        val_meta["clean_self_all"] = val_clean_self_gpu
        val_meta["clean_self_pad"] = val_clean_self_pad

    total_vram_bytes = (
        train_clean_z_pad.numel() * train_clean_z_pad.element_size()
        + val_clean_z_pad.numel() * val_clean_z_pad.element_size()
        + sum(t.numel() * t.element_size() for t in train_shifted_z_pad.values())
    )
    if train_clean_self_pad is not None:
        total_vram_bytes += (
            train_clean_self_pad.numel() * train_clean_self_pad.element_size()
        )
    if val_clean_self_pad is not None:
        total_vram_bytes += (
            val_clean_self_pad.numel() * val_clean_self_pad.element_size()
        )
    total_vram_bytes += sum(
        t.numel() * t.element_size() for t in train_shifted_self_pad.values()
    )
    vram_mb = total_vram_bytes / (1024.0 * 1024.0)
    src_tag = f"Loaded from {cache_file}" if loaded_from_disk else "Extracted & cached"
    extra_tag = ""
    if train_clean_self_gpu is not None:
        extra_tag += f", self_feat={int(train_clean_self_gpu.shape[-1])}d"
    if session_calibrated:
        extra_tag += ", causal_sess_cal=on"
    print(
        f" [Pre-Extract Cache] {src_tag} in {elapsed:.2f}s | "
        f"Resident in GPU VRAM ({device}): {vram_mb:.1f} MB "
        f"(train={len(train_ds):,}, shifts={len(train_shifted_z)}, val={len(val_ds):,}{extra_tag}; "
        f"raw MEG DataLoader & cnn(x) 100% bypassed)",
        flush=True,
    )
    return {
        "train_clean_z": train_clean_z,
        "train_clean_z_pad": train_clean_z_pad,
        "train_shifted_z": train_shifted_z,
        "train_shifted_z_pad": train_shifted_z_pad,
        "val_clean_z": val_clean_z,
        "val_clean_z_pad": val_clean_z_pad,
        "train_clean_self": train_clean_self_gpu,
        "train_clean_self_pad": train_clean_self_pad,
        "train_shifted_self": train_shifted_self_gpu,
        "train_shifted_self_pad": train_shifted_self_pad,
        "val_clean_self": val_clean_self_gpu,
        "val_clean_self_pad": val_clean_self_pad,
        "session_calibrated": bool(session_calibrated),
        "train_meta": train_meta,
        "val_meta": val_meta,
        "train_static_revision_table": train_static_rev,
        "val_static_revision_table": val_static_rev,
        "cache_file": str(cache_file) if cache_file is not None else None,
        "loaded_from_disk": bool(loaded_from_disk),
        "seconds": float(elapsed),
    }


def _export_stage1_calibration_buffers(ctx) -> dict:
    payload: dict[str, object] = {}
    if ctx is None:
        return payload
    for key in (
        "clean_input_mean",
        "pre_head_mean",
        "h_bins_mean",
        "h_bins_pca_proj",
        "h_bins_pca_scale",
    ):
        val = getattr(ctx, key, None)
        if isinstance(val, torch.Tensor):
            payload[key] = val.detach().float().cpu()
    return payload


@torch.no_grad()
def _restore_stage1_calibration_buffers(ctx, payload: dict | None) -> None:
    if ctx is None or not isinstance(payload, dict):
        return
    if (
        "clean_input_mean" in payload
        and hasattr(ctx, "set_clean_input_mean")
        and not bool(getattr(ctx, "clean_input_mean_ready", torch.tensor(False)).item())
    ):
        ctx.set_clean_input_mean(payload["clean_input_mean"])
    if (
        "pre_head_mean" in payload
        and hasattr(ctx, "set_pre_head_mean")
        and getattr(ctx, "pre_head_mean_ready", None) is not None
        and not bool(ctx.pre_head_mean_ready.item())
    ):
        ctx.set_pre_head_mean(payload["pre_head_mean"])
    if (
        "h_bins_pca_proj" in payload
        and getattr(ctx, "use_h_bins_pca", False)
        and getattr(ctx, "h_bins_pca_ready", None) is not None
        and not bool(ctx.h_bins_pca_ready.item())
    ):
        ctx.h_bins_mean.copy_(
            payload["h_bins_mean"].to(device=ctx.h_bins_mean.device, dtype=torch.float32)
        )
        ctx.h_bins_pca_proj.copy_(
            payload["h_bins_pca_proj"].to(
                device=ctx.h_bins_pca_proj.device, dtype=torch.float32
            )
        )
        ctx.h_bins_pca_scale.copy_(
            payload["h_bins_pca_scale"].to(
                device=ctx.h_bins_pca_scale.device, dtype=torch.float32
            )
        )
        ctx.h_bins_pca_ready.fill_(True)


def _preextract_with_prehead_or_session_calibration(
    model,
    train_ds,
    val_ds,
    *,
    dataset: str,
    subset_tag: str,
    word_ckpt_sha256: str,
    window_seconds: float,
    window_start_offset_seconds: float,
    shifts: list[float],
    norm_shift_mode: str,
    eff_preroll: float,
    eff_baseline: float,
    eff_causal_hp: float,
    eff_hp_mode: str,
    eff_hp_hz: float,
    eff_hp_order: int,
    eff_clamp: float,
    device: torch.device,
    batch_size: int,
    amp: bool,
    amp_dtype: torch.dtype,
    feature_dtype: torch.dtype,
    cache_to_disk: bool,
    show_progress: bool,
) -> dict:
    """支持 pre-head 解坍缩特征（h_pre + 可选 PCA(h_bins)）与在线因果会话校准的预提取快路径。"""
    ctx = getattr(model, "context_transformer", None)
    need_pre_head = bool(getattr(ctx, "uses_pre_head_features", False))
    need_bins_pca = bool(need_pre_head and getattr(ctx, "use_h_bins_pca", False))
    need_sess_cal = bool(getattr(ctx, "session_calibration_enabled", False))
    pca_dim = int(getattr(ctx, "h_bins_pca_dim", 0)) if need_bins_pca else 0
    m0 = float(getattr(ctx, "session_calibration_m0", 50.0))
    lag_gap = int(getattr(ctx, "session_calibration_lag_gap", 2))
    alpha_cal = float(getattr(ctx, "session_calibration_alpha", 1.0))
    profile = (
        f"op3a_pre{int(need_pre_head)}_pca{pca_dim}"
        f"_sc{int(need_sess_cal)}_m0{m0:g}_g{lag_gap}_a{alpha_cal:g}"
    )

    cache_file: Path | None = None
    if cache_to_disk and word_ckpt_sha256:
        cache_file = _preextracted_cache_path(
            dataset=dataset,
            subset_tag=subset_tag,
            word_ckpt_sha256=word_ckpt_sha256,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            shift_seconds=shifts,
            train_len=len(train_ds),
            val_len=len(val_ds),
            shift_mode=norm_shift_mode,
            preroll_seconds=eff_preroll,
            baseline_seconds=eff_baseline,
            causal_highpass_seconds=eff_causal_hp,
            causal_highpass_mode=eff_hp_mode,
            causal_highpass_hz=eff_hp_hz,
            causal_highpass_order=eff_hp_order,
            clamp=eff_clamp,
            feature_profile=profile,
        )
        if cache_file.is_file():
            t0 = time.perf_counter()
            try:
                saved = torch.load(
                    cache_file, map_location="cpu", weights_only=False
                )
                has_self_ok = (not need_pre_head) or (
                    saved.get("train_clean_self") is not None
                    and saved.get("val_clean_self") is not None
                    and len(saved["train_clean_self"]) == len(train_ds)
                    and len(saved["val_clean_self"]) == len(val_ds)
                )
                if (
                    isinstance(saved, dict)
                    and saved.get("train_clean_z") is not None
                    and len(saved["train_clean_z"]) == len(train_ds)
                    and len(saved["val_clean_z"]) == len(val_ds)
                    and has_self_ok
                ):
                    _restore_stage1_calibration_buffers(
                        ctx, saved.get("stage1_buffers")
                    )
                    elapsed = time.perf_counter() - t0
                    t_clean = saved.pop("train_clean_z")
                    t_shifted = saved.pop("train_shifted_z", None) or {}
                    v_clean = saved.pop("val_clean_z")
                    t_self = saved.pop("train_clean_self", None)
                    t_self_sh = saved.pop("train_shifted_self", None) or {}
                    v_self = saved.pop("val_clean_self", None)
                    saved.clear()
                    del saved
                    return _finalize_gpu_preextracted_bundle(
                        model,
                        train_ds,
                        val_ds,
                        t_clean,
                        t_shifted,
                        v_clean,
                        device,
                        cache_file=cache_file,
                        loaded_from_disk=True,
                        elapsed=elapsed,
                        feature_dtype=feature_dtype,
                        train_clean_self=t_self,
                        train_shifted_self=t_self_sh,
                        val_clean_self=v_self,
                        session_calibrated=need_sess_cal,
                    )
            except Exception:
                pass

    t0 = time.perf_counter()
    train_meta = _extract_static_dataset_metadata(train_ds, device)
    val_meta = _extract_static_dataset_metadata(val_ds, device)

    print(
        f" [Pre-Extract Cache] Extracting frozen Stage-1 features ({profile}; "
        f"Clean 0s + {len(shifts)} shifted conditions [{norm_shift_mode}])...",
        flush=True,
    )
    need_raw_bins_on_train = bool(
        need_bins_pca
        and (
            getattr(ctx, "h_bins_pca_ready", None) is None
            or not bool(ctx.h_bins_pca_ready.item())
        )
    )
    train_z_raw, train_h_pre_raw, train_h_bins_obj = (
        _extract_single_dataset_stage1_features(
            model,
            train_ds,
            device=device,
            batch_size=batch_size,
            amp=amp,
            amp_dtype=amp_dtype,
            temporal_shift_seconds=0.0,
            shift_mode=norm_shift_mode,
            preroll_seconds=eff_preroll,
            show_progress=show_progress,
            progress_desc="Pre-extract [Train | Clean 0s + PreHead]",
            return_pre_head=need_pre_head,
            return_raw_h_bins=need_raw_bins_on_train,
            project_h_bins_fn=(
                ctx.project_h_bins
                if (need_bins_pca and not need_raw_bins_on_train)
                else None
            ),
        )
    )

    # 严格仅基于训练集原始特征初始化均值与 Top-K PCA 基（绝不触碰 val/test）
    if (
        hasattr(ctx, "set_clean_input_mean")
        and not bool(getattr(ctx, "clean_input_mean_ready", torch.tensor(False)).item())
    ):
        ctx.set_clean_input_mean(train_z_raw.float().mean(dim=0))
    if (
        need_pre_head
        and train_h_pre_raw is not None
        and hasattr(ctx, "set_pre_head_mean")
        and getattr(ctx, "pre_head_mean_ready", None) is not None
        and not bool(ctx.pre_head_mean_ready.item())
    ):
        ctx.set_pre_head_mean(train_h_pre_raw.float().mean(dim=0))

    train_h_bins_proj: torch.Tensor | None = None
    if need_bins_pca:
        if need_raw_bins_on_train and train_h_bins_obj is not None:
            ctx.fit_h_bins_pca(
                train_h_bins_obj,
                recording_ids=train_meta["recording_ids"],
                subject_indices=train_meta["subject_indices"],
                onset_seconds=train_meta["onset_seconds"],
            )
            # 分块线性投影后立即释放原始 1280-d h_bins 节省内存
            proj_chunks: list[torch.Tensor] = []
            n_tr = int(train_h_bins_obj.shape[0])
            for s_i in range(0, n_tr, 16384):
                e_i = min(s_i + 16384, n_tr)
                proj_chunks.append(
                    ctx.project_h_bins(
                        train_h_bins_obj[s_i:e_i].to(device=device)
                    ).to(device="cpu", dtype=train_z_raw.dtype)
                )
            train_h_bins_proj = torch.cat(proj_chunks, dim=0)
            del train_h_bins_obj, proj_chunks
        else:
            train_h_bins_proj = train_h_bins_obj

    train_clean_z, train_clean_self = ctx.calibrate_stage1_features(
        train_z_raw,
        train_h_pre_raw,
        train_h_bins_proj,
        recording_ids=train_meta["recording_ids"],
        subject_indices=train_meta["subject_indices"],
        onset_seconds=train_meta["onset_seconds"],
    )
    del train_z_raw, train_h_pre_raw, train_h_bins_proj

    val_z_raw, val_h_pre_raw, val_h_bins_proj = (
        _extract_single_dataset_stage1_features(
            model,
            val_ds,
            device=device,
            batch_size=batch_size,
            amp=amp,
            amp_dtype=amp_dtype,
            temporal_shift_seconds=0.0,
            shift_mode=norm_shift_mode,
            preroll_seconds=eff_preroll,
            show_progress=show_progress,
            progress_desc="Pre-extract [Val   | Clean 0s + PreHead]",
            return_pre_head=need_pre_head,
            return_raw_h_bins=False,
            project_h_bins_fn=(ctx.project_h_bins if need_bins_pca else None),
        )
    )
    val_clean_z, val_clean_self = ctx.calibrate_stage1_features(
        val_z_raw,
        val_h_pre_raw,
        val_h_bins_proj,
        recording_ids=val_meta["recording_ids"],
        subject_indices=val_meta["subject_indices"],
        onset_seconds=val_meta["onset_seconds"],
    )
    del val_z_raw, val_h_pre_raw, val_h_bins_proj

    train_shifted_z: dict[float, torch.Tensor] = {}
    train_shifted_self: dict[float, torch.Tensor] = {}
    for shift_sec in shifts:
        f_sec = float(shift_sec)
        shift_label = (
            f"{shift_sec:+.0f}s"
            if abs(shift_sec - round(shift_sec)) < 1e-6
            else f"{shift_sec:+.1f}s"
        )
        sh_z_raw, sh_h_pre_raw, sh_h_bins_proj = (
            _extract_single_dataset_stage1_features(
                model,
                train_ds,
                device=device,
                batch_size=batch_size,
                amp=amp,
                amp_dtype=amp_dtype,
                temporal_shift_seconds=f_sec,
                shift_mode=norm_shift_mode,
                preroll_seconds=eff_preroll,
                show_progress=show_progress,
                progress_desc=f"Pre-extract [Train | Shift {shift_label} + PreHead]",
                return_pre_head=need_pre_head,
                return_raw_h_bins=False,
                project_h_bins_fn=(ctx.project_h_bins if need_bins_pca else None),
            )
        )
        sh_z_cal, sh_self_cal = ctx.calibrate_stage1_features(
            sh_z_raw,
            sh_h_pre_raw,
            sh_h_bins_proj,
            recording_ids=train_meta["recording_ids"],
            subject_indices=train_meta["subject_indices"],
            onset_seconds=train_meta["onset_seconds"],
        )
        train_shifted_z[f_sec] = sh_z_cal
        if sh_self_cal is not None:
            train_shifted_self[f_sec] = sh_self_cal
        del sh_z_raw, sh_h_pre_raw, sh_h_bins_proj

    if cache_file is not None:
        try:
            tmp_file = cache_file.with_suffix(".pt.tmp")
            disk_payload = {
                "word_ckpt_sha256": str(word_ckpt_sha256),
                "shift_mode": norm_shift_mode,
                "preroll_seconds": eff_preroll,
                "feature_profile": profile,
                "stage1_buffers": _export_stage1_calibration_buffers(ctx),
                "train_clean_z": train_clean_z.to(dtype=torch.bfloat16),
                "val_clean_z": val_clean_z.to(dtype=torch.bfloat16),
                "train_shifted_z": {
                    float(k): v.to(dtype=torch.bfloat16)
                    for k, v in train_shifted_z.items()
                },
            }
            if train_clean_self is not None:
                disk_payload["train_clean_self"] = train_clean_self.to(
                    dtype=torch.bfloat16
                )
            if val_clean_self is not None:
                disk_payload["val_clean_self"] = val_clean_self.to(
                    dtype=torch.bfloat16
                )
            if train_shifted_self:
                disk_payload["train_shifted_self"] = {
                    float(k): v.to(dtype=torch.bfloat16)
                    for k, v in train_shifted_self.items()
                }
            torch.save(disk_payload, tmp_file)
            os.replace(tmp_file, cache_file)
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    return _finalize_gpu_preextracted_bundle(
        model,
        train_ds,
        val_ds,
        train_clean_z,
        train_shifted_z,
        val_clean_z,
        device,
        cache_file=cache_file,
        loaded_from_disk=False,
        elapsed=elapsed,
        feature_dtype=feature_dtype,
        train_clean_self=train_clean_self,
        train_shifted_self=train_shifted_self,
        val_clean_self=val_clean_self,
        session_calibrated=need_sess_cal,
        train_meta=train_meta,
        val_meta=val_meta,
    )


def preextract_frozen_cnn_features(
    model,
    train_ds,
    val_ds,
    *,
    dataset: str,
    subset_tag: str,
    word_ckpt_sha256: str,
    window_seconds: float,
    window_start_offset_seconds: float = 0.0,
    shift_seconds: Iterable[float] = (),
    shift_mode: str = "deterministic_circular",
    preroll_seconds: float = 45.0,
    device: torch.device,
    batch_size: int = 512,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    cache_to_disk: bool = True,
    show_progress: bool = True,
) -> dict:
    """一次性提取（或从本地缓存秒级加载）训练集 Clean + Shifted 与验证集 Clean 的冻结 Stage-1 CNN 特征。"""
    shifts = [float(s) for s in shift_seconds]
    norm_shift_mode = _normalize_shift_mode(shift_mode) or "deterministic_circular"
    eff_preroll = float(preroll_seconds)
    eff_baseline = float(getattr(train_ds, "baseline_seconds", 0.5))
    eff_causal_hp = float(getattr(train_ds, "causal_highpass_seconds", 0.0))
    eff_hp_mode = str(
        getattr(train_ds, "causal_highpass_mode", "moving_average") or "moving_average"
    ).strip().lower()
    eff_hp_hz = float(getattr(train_ds, "causal_highpass_hz", 0.5))
    eff_hp_order = int(getattr(train_ds, "causal_highpass_order", 2))
    eff_clamp = float(getattr(train_ds, "clamp", getattr(train_ds, "_clamp", 5.0)))
    feature_dtype = (
        torch.bfloat16
        if (bool(amp) and device.type == "cuda" and amp_dtype == torch.bfloat16)
        else torch.float32
    )
    ctx_mod = getattr(model, "context_transformer", None)
    if ctx_mod is not None and (
        bool(getattr(ctx_mod, "uses_pre_head_features", False))
        or bool(getattr(ctx_mod, "session_calibration_enabled", False))
    ):
        return _preextract_with_prehead_or_session_calibration(
            model,
            train_ds,
            val_ds,
            dataset=dataset,
            subset_tag=subset_tag,
            word_ckpt_sha256=word_ckpt_sha256,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            shifts=shifts,
            norm_shift_mode=norm_shift_mode,
            eff_preroll=eff_preroll,
            eff_baseline=eff_baseline,
            eff_causal_hp=eff_causal_hp,
            eff_hp_mode=eff_hp_mode,
            eff_hp_hz=eff_hp_hz,
            eff_hp_order=eff_hp_order,
            eff_clamp=eff_clamp,
            device=device,
            batch_size=batch_size,
            amp=amp,
            amp_dtype=amp_dtype,
            feature_dtype=feature_dtype,
            cache_to_disk=cache_to_disk,
            show_progress=show_progress,
        )
    cache_file: Path | None = None
    if cache_to_disk and word_ckpt_sha256:
        cache_file = _preextracted_cache_path(
            dataset=dataset,
            subset_tag=subset_tag,
            word_ckpt_sha256=word_ckpt_sha256,
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            shift_seconds=shifts,
            train_len=len(train_ds),
            val_len=len(val_ds),
            shift_mode=norm_shift_mode,
            preroll_seconds=eff_preroll,
            baseline_seconds=eff_baseline,
            causal_highpass_seconds=eff_causal_hp,
            causal_highpass_mode=eff_hp_mode,
            causal_highpass_hz=eff_hp_hz,
            causal_highpass_order=eff_hp_order,
            clamp=eff_clamp,
        )
        if cache_file.is_file():
            t0 = time.perf_counter()
            try:
                saved = torch.load(
                    cache_file, map_location="cpu", weights_only=False
                )
                if (
                    isinstance(saved, dict)
                    and saved.get("train_clean_z") is not None
                    and len(saved["train_clean_z"]) == len(train_ds)
                    and len(saved["val_clean_z"]) == len(val_ds)
                ):
                    elapsed = time.perf_counter() - t0
                    t_clean = saved.pop("train_clean_z")
                    t_shifted = saved.pop("train_shifted_z", None) or {}
                    v_clean = saved.pop("val_clean_z")
                    saved.clear()
                    del saved
                    return _finalize_gpu_preextracted_bundle(
                        model,
                        train_ds,
                        val_ds,
                        t_clean,
                        t_shifted,
                        v_clean,
                        device,
                        cache_file=cache_file,
                        loaded_from_disk=True,
                        elapsed=elapsed,
                        feature_dtype=feature_dtype,
                    )
            except Exception:
                pass

    t0 = time.perf_counter()
    donor_train_clean: torch.Tensor | None = None
    donor_val_clean: torch.Tensor | None = None
    donor_shifted: dict[float, torch.Tensor] = {}
    if cache_to_disk and word_ckpt_sha256:
        donor_candidates = [
            (
                _preextracted_cache_path(
                    dataset=dataset,
                    subset_tag=subset_tag,
                    word_ckpt_sha256=word_ckpt_sha256,
                    window_seconds=window_seconds,
                    window_start_offset_seconds=window_start_offset_seconds,
                    shift_seconds=list(DEFAULT_CAUSAL_PREROLL_SHIFT_SECONDS),
                    train_len=len(train_ds),
                    val_len=len(val_ds),
                    shift_mode=norm_shift_mode,
                    preroll_seconds=eff_preroll,
                    baseline_seconds=eff_baseline,
                    causal_highpass_seconds=eff_causal_hp,
                    causal_highpass_mode=eff_hp_mode,
                    causal_highpass_hz=eff_hp_hz,
                    causal_highpass_order=eff_hp_order,
                    clamp=eff_clamp,
                ),
                norm_shift_mode,
            ),
            (
                _preextracted_cache_path(
                    dataset=dataset,
                    subset_tag=subset_tag,
                    word_ckpt_sha256=word_ckpt_sha256,
                    window_seconds=window_seconds,
                    window_start_offset_seconds=window_start_offset_seconds,
                    shift_seconds=list(DEFAULT_SIAMESE_SHIFT_SECONDS),
                    train_len=len(train_ds),
                    val_len=len(val_ds),
                    shift_mode="deterministic_circular",
                    baseline_seconds=eff_baseline,
                    causal_highpass_seconds=eff_causal_hp,
                    causal_highpass_mode=eff_hp_mode,
                    causal_highpass_hz=eff_hp_hz,
                    causal_highpass_order=eff_hp_order,
                    clamp=eff_clamp,
                ),
                "deterministic_circular",
            ),
            (
                _preextracted_cache_path(
                    dataset=dataset,
                    subset_tag=subset_tag,
                    word_ckpt_sha256=word_ckpt_sha256,
                    window_seconds=window_seconds,
                    window_start_offset_seconds=window_start_offset_seconds,
                    shift_seconds=[],
                    train_len=len(train_ds),
                    val_len=len(val_ds),
                    shift_mode="deterministic_circular",
                    baseline_seconds=eff_baseline,
                    causal_highpass_seconds=eff_causal_hp,
                    causal_highpass_mode=eff_hp_mode,
                    causal_highpass_hz=eff_hp_hz,
                    causal_highpass_order=eff_hp_order,
                    clamp=eff_clamp,
                ),
                "deterministic_circular",
            ),
        ]
        for donor_path, donor_mode in donor_candidates:
            if cache_file is not None and donor_path == cache_file:
                continue
            if not donor_path.is_file():
                continue
            try:
                donor = torch.load(
                    donor_path, map_location="cpu", weights_only=False
                )
                if (
                    isinstance(donor, dict)
                    and donor.get("train_clean_z") is not None
                    and len(donor["train_clean_z"]) == len(train_ds)
                    and len(donor["val_clean_z"]) == len(val_ds)
                ):
                    if donor_train_clean is None:
                        donor_train_clean = donor["train_clean_z"].float()
                        donor_val_clean = donor["val_clean_z"].float()
                    if donor_mode == norm_shift_mode and isinstance(
                        donor.get("train_shifted_z"), dict
                    ):
                        for sk, sv in donor["train_shifted_z"].items():
                            if len(sv) == len(train_ds) and float(sk) not in donor_shifted:
                                donor_shifted[float(sk)] = sv.float()
            except Exception:
                continue

    print(
        f" [Pre-Extract Cache] Pre-extracting frozen Stage-1 CNN features "
        f"(Clean 0s + {len(shifts)} shifted conditions [{norm_shift_mode}], window={window_seconds:.2f}s)...",
        flush=True,
    )
    if donor_train_clean is not None and donor_val_clean is not None:
        train_clean_cpu = donor_train_clean
        val_clean_cpu = donor_val_clean
        print(
            " [Pre-Extract Cache] Reused Clean 0s (Train & Val) from matching Stage-1 cache.",
            flush=True,
        )
    else:
        train_clean_cpu = _extract_single_dataset_cnn_embeddings(
            model,
            train_ds,
            device=device,
            batch_size=batch_size,
            amp=amp,
            amp_dtype=amp_dtype,
            temporal_shift_seconds=0.0,
            shift_mode=norm_shift_mode,
            preroll_seconds=eff_preroll,
            show_progress=show_progress,
            progress_desc="Pre-extract [Train | Clean 0s]",
        )
        val_clean_cpu = _extract_single_dataset_cnn_embeddings(
            model,
            val_ds,
            device=device,
            batch_size=batch_size,
            amp=amp,
            amp_dtype=amp_dtype,
            temporal_shift_seconds=0.0,
            shift_mode=norm_shift_mode,
            preroll_seconds=eff_preroll,
            show_progress=show_progress,
            progress_desc="Pre-extract [Val   | Clean 0s]",
        )

    train_shifted_cpu: dict[float, torch.Tensor] = {}
    for shift_sec in shifts:
        f_sec = float(shift_sec)
        if f_sec in donor_shifted:
            train_shifted_cpu[f_sec] = donor_shifted.pop(f_sec)
            continue
        shift_label = f"{shift_sec:+.0f}s" if abs(shift_sec - round(shift_sec)) < 1e-6 else f"{shift_sec:+.1f}s"
        train_shifted_cpu[f_sec] = _extract_single_dataset_cnn_embeddings(
            model,
            train_ds,
            device=device,
            batch_size=batch_size,
            amp=amp,
            amp_dtype=amp_dtype,
            temporal_shift_seconds=f_sec,
            shift_mode=norm_shift_mode,
            preroll_seconds=eff_preroll,
            show_progress=show_progress,
            progress_desc=f"Pre-extract [Train | Shift {shift_label} ({norm_shift_mode})]",
        )
    donor_shifted.clear()

    if cache_file is not None:
        try:
            tmp_file = cache_file.with_suffix(".pt.tmp")
            torch.save(
                {
                    "word_ckpt_sha256": str(word_ckpt_sha256),
                    "shift_mode": norm_shift_mode,
                    "preroll_seconds": eff_preroll,
                    "train_clean_z": train_clean_cpu.to(dtype=torch.bfloat16),
                    "val_clean_z": val_clean_cpu.to(dtype=torch.bfloat16),
                    "train_shifted_z": {
                        float(k): v.to(dtype=torch.bfloat16)
                        for k, v in train_shifted_cpu.items()
                    },
                },
                tmp_file,
            )
            os.replace(tmp_file, cache_file)
        except Exception:
            pass

    elapsed = time.perf_counter() - t0
    return _finalize_gpu_preextracted_bundle(
        model,
        train_ds,
        val_ds,
        train_clean_cpu,
        train_shifted_cpu,
        val_clean_cpu,
        device,
        cache_file=cache_file,
        loaded_from_disk=False,
        elapsed=elapsed,
        feature_dtype=feature_dtype,
    )


def refresh_preextracted_train_features(
    preextracted_cache: dict,
    model,
    train_ds,
    *,
    shift_seconds: Iterable[float] = (),
    shift_mode: str = "preroll",
    preroll_seconds: float = 45.0,
    device: torch.device,
    batch_size: int = 512,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    show_progress: bool = False,
) -> dict:
    """在每轮刷新合成平滑脉冲流后，原地更新 GPU 驻留的训练集 Clean/Shifted 冻结 CNN 特征（不重提 Val）。"""
    shifts = [float(s) for s in shift_seconds]
    norm_shift_mode = _normalize_shift_mode(shift_mode) or "preroll"
    eff_preroll = float(preroll_seconds)
    ctx_mod = getattr(model, "context_transformer", None)
    need_pre_head = bool(
        ctx_mod is not None and getattr(ctx_mod, "uses_pre_head_features", False)
    )
    need_sess_cal = bool(
        ctx_mod is not None and getattr(ctx_mod, "session_calibration_enabled", False)
    )
    if need_pre_head or need_sess_cal:
        need_bins_pca = bool(
            need_pre_head and getattr(ctx_mod, "use_h_bins_pca", False)
        )
        t_meta = preextracted_cache["train_meta"]
        z_raw, h_pre_raw, h_bins_proj = _extract_single_dataset_stage1_features(
            model,
            train_ds,
            device=device,
            batch_size=batch_size,
            amp=amp,
            amp_dtype=amp_dtype,
            temporal_shift_seconds=0.0,
            shift_mode=norm_shift_mode,
            preroll_seconds=eff_preroll,
            show_progress=show_progress,
            progress_desc="Refresh [Train | Clean 0s + PreHead]",
            return_pre_head=need_pre_head,
            return_raw_h_bins=False,
            project_h_bins_fn=(ctx_mod.project_h_bins if need_bins_pca else None),
        )
        z_cal, self_cal = ctx_mod.calibrate_stage1_features(
            z_raw,
            h_pre_raw,
            h_bins_proj,
            recording_ids=t_meta["recording_ids"],
            subject_indices=t_meta["subject_indices"],
            onset_seconds=t_meta["onset_seconds"],
        )
        preextracted_cache["train_clean_z_pad"][: len(train_ds)].copy_(
            z_cal, non_blocking=False
        )
        if (
            self_cal is not None
            and preextracted_cache.get("train_clean_self_pad") is not None
        ):
            preextracted_cache["train_clean_self_pad"][: len(train_ds)].copy_(
                self_cal, non_blocking=False
            )
        for f_sec in shifts:
            sh_z_raw, sh_h_pre_raw, sh_h_bins_proj = (
                _extract_single_dataset_stage1_features(
                    model,
                    train_ds,
                    device=device,
                    batch_size=batch_size,
                    amp=amp,
                    amp_dtype=amp_dtype,
                    temporal_shift_seconds=f_sec,
                    shift_mode=norm_shift_mode,
                    preroll_seconds=eff_preroll,
                    show_progress=show_progress,
                    progress_desc=f"Refresh [Train | Shift {f_sec:+.0f}s + PreHead]",
                    return_pre_head=need_pre_head,
                    return_raw_h_bins=False,
                    project_h_bins_fn=(
                        ctx_mod.project_h_bins if need_bins_pca else None
                    ),
                )
            )
            sh_z_cal, sh_self_cal = ctx_mod.calibrate_stage1_features(
                sh_z_raw,
                sh_h_pre_raw,
                sh_h_bins_proj,
                recording_ids=t_meta["recording_ids"],
                subject_indices=t_meta["subject_indices"],
                onset_seconds=t_meta["onset_seconds"],
            )
            if f_sec in preextracted_cache["train_shifted_z_pad"]:
                preextracted_cache["train_shifted_z_pad"][f_sec][
                    : len(train_ds)
                ].copy_(sh_z_cal, non_blocking=False)
            else:
                unpad, pad = _alloc_padded_gpu_feature_matrix(
                    sh_z_cal,
                    device,
                    dtype=preextracted_cache["train_clean_z_pad"].dtype,
                )
                preextracted_cache["train_shifted_z"][f_sec] = unpad
                preextracted_cache["train_shifted_z_pad"][f_sec] = pad
            if sh_self_cal is not None:
                self_sh_pad = preextracted_cache.setdefault(
                    "train_shifted_self_pad", {}
                )
                self_sh_unpad = preextracted_cache.setdefault(
                    "train_shifted_self", {}
                )
                if f_sec in self_sh_pad:
                    self_sh_pad[f_sec][: len(train_ds)].copy_(
                        sh_self_cal, non_blocking=False
                    )
                else:
                    s_unpad, s_pad = _alloc_padded_gpu_feature_matrix(
                        sh_self_cal,
                        device,
                        dtype=preextracted_cache["train_clean_z_pad"].dtype,
                    )
                    self_sh_unpad[f_sec] = s_unpad
                    self_sh_pad[f_sec] = s_pad
        return preextracted_cache

    clean_dst = preextracted_cache["train_clean_z_pad"][: len(train_ds)]
    _extract_single_dataset_cnn_embeddings(
        model,
        train_ds,
        device=device,
        batch_size=batch_size,
        amp=amp,
        amp_dtype=amp_dtype,
        temporal_shift_seconds=0.0,
        shift_mode=norm_shift_mode,
        preroll_seconds=eff_preroll,
        show_progress=show_progress,
        progress_desc="Refresh [Train | Clean 0s]",
        out_tensor=clean_dst,
    )

    for f_sec in shifts:
        if f_sec in preextracted_cache["train_shifted_z_pad"]:
            shift_dst = preextracted_cache["train_shifted_z_pad"][f_sec][
                : len(train_ds)
            ]
            _extract_single_dataset_cnn_embeddings(
                model,
                train_ds,
                device=device,
                batch_size=batch_size,
                amp=amp,
                amp_dtype=amp_dtype,
                temporal_shift_seconds=f_sec,
                shift_mode=norm_shift_mode,
                preroll_seconds=eff_preroll,
                show_progress=show_progress,
                progress_desc=f"Refresh [Train | Shift {f_sec:+.0f}s]",
                out_tensor=shift_dst,
            )
        else:
            shifted_cpu = _extract_single_dataset_cnn_embeddings(
                model,
                train_ds,
                device=device,
                batch_size=batch_size,
                amp=amp,
                amp_dtype=amp_dtype,
                temporal_shift_seconds=f_sec,
                shift_mode=norm_shift_mode,
                preroll_seconds=eff_preroll,
                show_progress=show_progress,
                progress_desc=f"Refresh [Train | Shift {f_sec:+.0f}s]",
            )
            unpad, pad = _alloc_padded_gpu_feature_matrix(
                shifted_cpu,
                device,
                dtype=preextracted_cache["train_clean_z_pad"].dtype,
            )
            preextracted_cache["train_shifted_z"][f_sec] = unpad
            preextracted_cache["train_shifted_z_pad"][f_sec] = pad
    return preextracted_cache
