"""孪生共模抵消（Siamese Common-Mode Rejection）+ 错位零化正则（Shift-Null Regularization）与冻结 CNN 特征预提取极速训练模块。

本模块完全独立于主训练循环（word_encoder_training.py）与核心模型定义（embedding.py），
专门用于 Stage 2.1 (context) 与 Stage 2.2 (revision) 冻结 Stage-1 CNN 时的探索性去偏训练与 5–8 倍加速：
1. 预提取冻结 CNN 特征缓存（Pre-Extraction Cache）：
   在训练启动前一次性提取 Clean (0s) 与 6 个时间错位 (-120s, -60s, -30s, +30s, +60s, +120s)
   的 Stage-1 CNN 归一化向量，训练期间无需重复运行 306 通道膨胀卷积网络。
2. 机制 1 — 前向孪生共模扣除（Siamese Common-Mode Subtraction，保持双分支梯度连通不 detach）：
   y_tilde_i^(r) = Norm( y_i^(r, clean) - eta_train * (1 / M) * sum_{m=1..M} y_i^(r, shift_m) )
3. 机制 2 — 错位零化正则（Shift-Null Regularization）：
   L_null^(r) = (1 / B_r) * sum_{i=1..B_r} || (E_V - e_bar) * y_bar_i^(r, shift) ||_2^2
   L_total = (1 / |D_b|) * sum_{r in D_b} ( L_SigLIP(y_tilde^(r), E_target) + lambda_null * L_null^(r) )
"""

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
from braindecoding.training.word_encoder_training import (
    _wrap_progress,
    local_context_internal_mask,
    resolve_amp_dtype,
)

DEFAULT_SIAMESE_SHIFT_SECONDS: tuple[float, ...] = (
    -120.0,
    -60.0,
    -30.0,
    30.0,
    60.0,
    120.0,
)


def _parse_shift_list(
    raw_shifts: str | Iterable[float | int | str] | None,
) -> list[float]:
    if raw_shifts is None:
        return list(DEFAULT_SIAMESE_SHIFT_SECONDS)
    if isinstance(raw_shifts, (int, float)):
        val = float(raw_shifts)
        return [val] if abs(val) > 1e-9 else []
    if isinstance(raw_shifts, str):
        parts = [p.strip() for p in raw_shifts.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in raw_shifts if str(p).strip()]
    parsed: list[float] = []
    for item in parts:
        val = float(item)
        if abs(val) <= 1e-9:
            raise ValueError("孪生错位分支 shift_seconds 不能包含 0。")
        if val not in parsed:
            parsed.append(val)
    return parsed


def _parse_bool_env(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"环境变量 {name}={raw!r} 不是合法布尔值。")


def resolve_siamese_debiased_config(
    cfg: dict,
    *,
    siamese_debiased: bool | None = None,
    eta_train: float | None = None,
    lambda_null: float | None = None,
    shifts_per_step: int | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    preextract_frozen_cnn: bool | None = None,
) -> dict:
    """合并 YAML `siamese_debiased` 配置块、环境变量与 CLI 参数。"""
    raw_block = (
        dict(cfg.get("siamese_debiased"))
        if isinstance(cfg.get("siamese_debiased"), dict)
        else {}
    )
    tc = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}

    env_enabled = _parse_bool_env("BRAINDECODING_SIAMESE_DEBIASED")
    env_preextract = _parse_bool_env("BRAINDECODING_PREEXTRACT_FROZEN_CNN")
    env_eta = os.environ.get("BRAINDECODING_ETA_TRAIN", "").strip()
    env_lambda = os.environ.get("BRAINDECODING_LAMBDA_NULL", "").strip()
    env_m = os.environ.get("BRAINDECODING_SHIFTS_PER_STEP", "").strip()
    env_shifts = os.environ.get("BRAINDECODING_TRAIN_SHIFT_SECONDS", "").strip()

    enabled = (
        bool(siamese_debiased)
        if siamese_debiased is not None
        else (
            env_enabled
            if env_enabled is not None
            else bool(raw_block.get("enabled", False))
        )
    )
    if eta_train is not None or env_eta or lambda_null is not None or env_lambda:
        # 显式传入了 eta_train 或 lambda_null 时自动启用孪生去偏
        if siamese_debiased is not False and env_enabled is not False:
            enabled = True

    resolved_eta = float(
        eta_train
        if eta_train is not None
        else (float(env_eta) if env_eta else raw_block.get("eta_train", 0.8))
    )
    resolved_lambda = float(
        lambda_null
        if lambda_null is not None
        else (
            float(env_lambda)
            if env_lambda
            else raw_block.get("lambda_null", 10.0)
        )
    )
    resolved_shifts = _parse_shift_list(
        shift_seconds
        if shift_seconds is not None
        else (
            env_shifts
            if env_shifts
            else raw_block.get("shift_seconds", DEFAULT_SIAMESE_SHIFT_SECONDS)
        )
    )
    if enabled and not resolved_shifts:
        raise ValueError("启用 siamese_debiased 时至少需要 1 个非零 shift_seconds。")

    default_m = min(2, len(resolved_shifts)) if resolved_shifts else 2
    resolved_m = int(
        shifts_per_step
        if shifts_per_step is not None
        else (
            int(env_m)
            if env_m
            else int(raw_block.get("shifts_per_step", default_m))
        )
    )
    if enabled:
        resolved_m = max(1, min(resolved_m, len(resolved_shifts)))

    resolved_preextract = (
        bool(preextract_frozen_cnn)
        if preextract_frozen_cnn is not None
        else (
            env_preextract
            if env_preextract is not None
            else bool(
                raw_block.get(
                    "preextract_frozen_cnn",
                    tc.get("preextract_frozen_cnn", enabled),
                )
            )
        )
    )
    sync_rng = bool(raw_block.get("synchronize_dropout_rng", True))
    cache_to_disk = bool(raw_block.get("cache_preextracted_to_disk", True))

    return {
        "enabled": bool(enabled),
        "eta_train": resolved_eta,
        "lambda_null": resolved_lambda,
        "shift_seconds": resolved_shifts,
        "shifts_per_step": int(resolved_m),
        "synchronize_dropout_rng": sync_rng,
        "preextract_frozen_cnn": bool(resolved_preextract),
        "cache_preextracted_to_disk": cache_to_disk,
    }


def build_centered_vocabulary_bank(
    dataset_obj,
    vocabulary: Iterable[str],
    device: torch.device,
) -> torch.Tensor:
    """构建中心化训练词表矩阵 E_tilde_V = E_V - e_bar in R^(|V| x D)。

    其中 E_V 的每一行为训练集唯一词的 L2 归一化目标向量，e_bar 为全词库均值向量。
    """
    emb_map = getattr(dataset_obj, "embedding_map", None) or getattr(
        dataset_obj, "word_to_embedding", None
    )
    if not emb_map:
        raise ValueError("Dataset 缺少 embedding_map，无法构建中心化训练词表矩阵。")

    rows = []
    for word in vocabulary:
        w_str = str(word)
        if w_str in emb_map:
            vec = np.asarray(emb_map[w_str], dtype=np.float32)
            rows.append(vec)
    if not rows:
        raise ValueError("训练词表在 embedding_map 中无匹配词项。")

    bank = torch.from_numpy(np.stack(rows, axis=0)).to(
        device=device, dtype=torch.float32
    )
    bank = F.normalize(bank, dim=-1)
    mean_vec = bank.mean(dim=0, keepdim=True)
    centered_bank = bank - mean_vec
    return centered_bank


def compute_shift_null_loss(
    y_shift_mean: torch.Tensor,
    centered_vocab_bank: torch.Tensor,
) -> torch.Tensor:
    """计算错位零化正则项 L_null^(r) = (1 / B) * sum_{i=1}^B || E_tilde_V * y_bar_i^(r, shift) ||_2^2。"""
    if y_shift_mean.numel() == 0:
        return y_shift_mean.new_zeros(())
    queries = y_shift_mean.float()
    bank = centered_vocab_bank.to(device=queries.device, dtype=torch.float32)
    # projections: [B, |V|]
    projections = queries @ bank.T
    # 每个样本在中心化词表空间上的 L2 范数平方，再对 batch 求均值
    return (projections * projections).sum(dim=-1).mean()


@torch.inference_mode()
def _extract_single_dataset_cnn_embeddings(
    model,
    dataset_obj,
    *,
    device: torch.device,
    batch_size: int = 512,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    temporal_shift_seconds: float = 0.0,
    show_progress: bool = True,
    progress_desc: str = "Pre-extract CNN",
) -> torch.Tensor:
    """在固定 temporal_shift_seconds 下按原始行序提取 Stage-1 CNN 归一化输出向量 [N, D]。"""
    model.brain_encoder.eval()
    orig_shift_sec = getattr(dataset_obj, "temporal_shift_seconds", 0.0)
    orig_shift_samp = getattr(dataset_obj, "temporal_shift_samples", 0)
    sr_hz = float(getattr(dataset_obj, "sampling_rate_hz", 50.0))

    try:
        if hasattr(dataset_obj, "temporal_shift_seconds"):
            dataset_obj.temporal_shift_seconds = float(temporal_shift_seconds)
        if hasattr(dataset_obj, "temporal_shift_samples"):
            dataset_obj.temporal_shift_samples = int(
                round(float(temporal_shift_seconds) * sr_hz)
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

    return torch.cat(chunks, dim=0)


def _extract_static_dataset_metadata(
    dataset_obj,
    device: torch.device,
) -> dict:
    """提取 Dataset 行级静态张量（目标词嵌入、受试者索引、句组索引等），免除训练期逐样本查表。"""
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
    else:
        words = []
        target_list = []
        subj_list = []
        sent_list = []
        event_ids = []
        recording_ids = []
        for idx in range(len(dataset_obj)):
            sample = dataset_obj[idx]
            words.append(str(sample["word"]))
            target_list.append(sample["text_embedding"].float())
            subj_list.append(int(sample["subject_index"]))
            sent_list.append(int(sample["sentence_index"]))
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

    return {
        "targets": targets,
        "subject_indices": subject_indices,
        "sentence_indices": sentence_indices,
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
) -> Path:
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
    if getattr(model, "prefix_revision_protocol", "local_window") == "full_prefix":
        return None

    horizons = tuple(int(d) for d in getattr(model, "prefix_revision_horizons", (0, 1, 2)))
    num_horizons = len(horizons)
    history_radius = int(getattr(model, "prefix_revision_history_radius", 2))
    max_win_len = history_radius + max(horizons, default=0) + 1
    n_rows = int(len(sentence_indices))

    groups = model._ordered_groups(sentence_indices)
    win_idx_np = np.full((n_rows, num_horizons, max_win_len), n_rows, dtype=np.int64)
    mask_np = np.zeros((n_rows, num_horizons, max_win_len), dtype=np.bool_)
    target_col_np = np.zeros((n_rows, num_horizons), dtype=np.int64)
    valid_delay_np = np.zeros((n_rows, num_horizons), dtype=np.bool_)
    group_pos_np = np.zeros(n_rows, dtype=np.int64)

    for indices in groups:
        g_len = len(indices)
        for cutoff, global_row in enumerate(indices):
            group_pos_np[global_row] = cutoff
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

    return {
        "row_win_global_idx": torch.from_numpy(win_idx_np).to(
            device=device, dtype=torch.long
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
    if getattr(model, "prefix_revision_protocol", "local_window") == "full_prefix":
        return None

    horizons = tuple(int(d) for d in getattr(model, "prefix_revision_horizons", (0, 1, 2)))
    history_radius = int(getattr(model, "prefix_revision_history_radius", 2))
    n_rows = int(len(sentence_indices))
    groups = model._ordered_groups(sentence_indices)
    avail_masks = model.revision_availability_masks(sentence_indices)

    table_by_delay: dict[int, dict[str, torch.Tensor]] = {}
    for delay in horizons:
        max_w_len = history_radius + int(delay) + 1
        win_idx_np = np.full((n_rows, max_w_len), n_rows, dtype=np.int64)
        mask_np = np.zeros((n_rows, max_w_len), dtype=np.bool_)
        target_col_np = np.zeros(n_rows, dtype=np.int64)

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

        table_by_delay[int(delay)] = {
            "win_global_idx": torch.from_numpy(win_idx_np).to(
                device=device, dtype=torch.long
            ),
            "win_mask": torch.from_numpy(mask_np).to(
                device=device, dtype=torch.bool
            ),
            "target_col": torch.from_numpy(target_col_np).to(
                device=device, dtype=torch.long
            ),
            "avail_mask": avail_masks[int(delay)].to(
                device=device, dtype=torch.bool
            ),
        }
    return table_by_delay


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
) -> dict:
    """将预提取张量、尾部零填充视图及全数据集静态 GPU 窗口查表常驻于 GPU 显存。"""
    train_meta = _extract_static_dataset_metadata(train_ds, device)
    val_meta = _extract_static_dataset_metadata(val_ds, device)

    train_clean_z = train_clean_z.to(device=device, dtype=torch.float32).contiguous()
    val_clean_z = val_clean_z.to(device=device, dtype=torch.float32).contiguous()
    train_shifted_z = {
        float(k): v.to(device=device, dtype=torch.float32).contiguous()
        for k, v in train_shifted_z.items()
    }

    train_clean_z_pad = F.pad(train_clean_z, (0, 0, 0, 1)).contiguous()
    val_clean_z_pad = F.pad(val_clean_z, (0, 0, 0, 1)).contiguous()
    train_shifted_z_pad = {
        float(k): F.pad(v, (0, 0, 0, 1)).contiguous()
        for k, v in train_shifted_z.items()
    }

    train_static_rev = _build_static_revision_train_table(
        model, train_meta["sentence_indices"], device
    )
    val_static_rev = _build_static_revision_eval_table(
        model, val_meta["sentence_indices"], device
    )
    val_meta["clean_z_pad"] = val_clean_z_pad
    val_meta["static_revision_table"] = val_static_rev

    total_vram_bytes = (
        train_clean_z_pad.numel() * train_clean_z_pad.element_size()
        + val_clean_z_pad.numel() * val_clean_z_pad.element_size()
        + sum(t.numel() * t.element_size() for t in train_shifted_z_pad.values())
    )
    vram_mb = total_vram_bytes / (1024.0 * 1024.0)
    src_tag = f"Loaded from {cache_file}" if loaded_from_disk else "Extracted & cached"
    print(
        f" [Pre-Extract Cache] {src_tag} in {elapsed:.2f}s | "
        f"Resident in GPU VRAM ({device}): {vram_mb:.1f} MB "
        f"(train={len(train_ds):,}, shifts={len(train_shifted_z)}, val={len(val_ds):,}; "
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
        "train_meta": train_meta,
        "val_meta": val_meta,
        "train_static_revision_table": train_static_rev,
        "val_static_revision_table": val_static_rev,
        "cache_file": str(cache_file) if cache_file is not None else None,
        "loaded_from_disk": bool(loaded_from_disk),
        "seconds": float(elapsed),
    }


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
    device: torch.device,
    batch_size: int = 512,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    cache_to_disk: bool = True,
    show_progress: bool = True,
) -> dict:
    """一次性提取（或从本地缓存秒级加载）训练集 Clean + Shifted 与验证集 Clean 的冻结 Stage-1 CNN 特征。"""
    shifts = [float(s) for s in shift_seconds]
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
                    return _finalize_gpu_preextracted_bundle(
                        model,
                        train_ds,
                        val_ds,
                        saved["train_clean_z"],
                        saved.get("train_shifted_z") or {},
                        saved["val_clean_z"],
                        device,
                        cache_file=cache_file,
                        loaded_from_disk=True,
                        elapsed=elapsed,
                    )
            except Exception:
                pass

    t0 = time.perf_counter()
    print(
        f" [Pre-Extract Cache] Pre-extracting frozen Stage-1 CNN features "
        f"(Clean 0s + {len(shifts)} shifted conditions, window={window_seconds:.2f}s)...",
        flush=True,
    )
    train_clean_cpu = _extract_single_dataset_cnn_embeddings(
        model,
        train_ds,
        device=device,
        batch_size=batch_size,
        amp=amp,
        amp_dtype=amp_dtype,
        temporal_shift_seconds=0.0,
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
        show_progress=show_progress,
        progress_desc="Pre-extract [Val   | Clean 0s]",
    )
    train_shifted_cpu: dict[float, torch.Tensor] = {}
    for shift_sec in shifts:
        shift_label = f"{shift_sec:+.0f}s" if abs(shift_sec - round(shift_sec)) < 1e-6 else f"{shift_sec:+.1f}s"
        train_shifted_cpu[float(shift_sec)] = _extract_single_dataset_cnn_embeddings(
            model,
            train_ds,
            device=device,
            batch_size=batch_size,
            amp=amp,
            amp_dtype=amp_dtype,
            temporal_shift_seconds=float(shift_sec),
            show_progress=show_progress,
            progress_desc=f"Pre-extract [Train | Shift {shift_label}]",
        )

    if cache_file is not None:
        try:
            tmp_file = cache_file.with_suffix(".pt.tmp")
            torch.save(
                {
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
    )


def _can_use_fast_padded_transformer(model) -> bool:
    """检查模型是否可直接运行零 Python 循环的 2D 张量填充 Transformer 极速路径。"""
    ctx = getattr(model, "context_transformer", None)
    if ctx is None:
        return False
    if getattr(model, "pre_transformer_trajectory", None) is not None:
        return False
    return (
        getattr(ctx, "query_window_left_radius", None) is None
        and not bool(getattr(ctx, "chunk_summary_enabled", False))
        and getattr(ctx, "physical_time_gap_bias", None) is None
        and getattr(ctx, "event_time_interaction_bias", None) is None
        and hasattr(ctx, "encoder")
    )


def _can_use_direct_layer_loop(
    ctx,
    *,
    use_causal_attention: bool,
) -> bool:
    """检查是否可直接执行零冗余同步、支持末层单目标剪枝与多分支共享 Dropout 的原生层循环。"""
    enc = getattr(ctx, "encoder", None)
    if enc is None:
        return False
    return (
        getattr(ctx, "local_relative_position_bias", None) is None
        and int(getattr(ctx, "local_attention_heads", 0)) == 0
        and getattr(ctx, "hierarchical_attention_radii", None) is None
        and int(getattr(ctx, "shared_passes", 1)) == 1
        and not bool(use_causal_attention)
        and not bool(getattr(enc, "residual_attn", False))
        and not bool(getattr(enc, "add_value_residual", False))
        and getattr(enc, "rel_pos", None) is None
    )


def _apply_rotary_pos_emb_fast(
    t: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """在不触发 float32 强制回退的前提下应用部分旋转位置编码（与 x_transformers 逐位等价）。"""
    rot_dim = int(cos.shape[-1])
    t_rot, t_unrot = t[..., :rot_dim], t[..., rot_dim:]
    t_reshaped = t_rot.reshape(*t_rot.shape[:-1], rot_dim // 2, 2)
    t1, t2 = t_reshaped.unbind(dim=-1)
    t_half = torch.stack((-t2, t1), dim=-1).reshape_as(t_rot)
    t_rot = (t_rot * cos) + (t_half * sin)
    return torch.cat((t_rot, t_unrot), dim=-1)


def _apply_siamese_shared_dropout(
    attn: torch.Tensor,
    dropout_module: torch.nn.Module,
    *,
    num_branches: int,
    synchronize_dropout_rng: bool,
) -> torch.Tensor:
    """在单次融合前向中为 Clean 与 M 个 Shifted 分支施加完全相同的注意力/前馈 Dropout 掩码。"""
    p = float(getattr(dropout_module, "p", 0.0))
    if not dropout_module.training or p <= 0.0:
        return attn
    if num_branches <= 1 or not synchronize_dropout_rng:
        return dropout_module(attn)
    branch_g = attn.shape[0] // num_branches
    reshaped = attn.view(num_branches, branch_g, *attn.shape[1:])
    shared_mask = F.dropout(
        torch.ones_like(reshaped[:1]),
        p=p,
        training=True,
        inplace=False,
    )
    return (reshaped * shared_mask).view_as(attn)


def _run_ff_with_shared_dropout(
    block: torch.nn.Module,
    x_norm: torch.Tensor,
    *,
    num_branches: int,
    synchronize_dropout_rng: bool,
) -> torch.Tensor:
    """在前馈层中支持跨孪生分支共享 Dropout 掩码（当 ff_dropout=0 时零开销直通）。"""
    if not block.training or num_branches <= 1 or not synchronize_dropout_rng:
        return block(x_norm)
    ff_seq = getattr(block, "ff", None)
    if ff_seq is None or not any(
        isinstance(m, torch.nn.Dropout) and float(getattr(m, "p", 0.0)) > 0.0
        for m in ff_seq.modules()
    ):
        return block(x_norm)
    h = x_norm
    for mod in ff_seq:
        if isinstance(mod, torch.nn.Sequential):
            for sub in mod:
                if isinstance(sub, torch.nn.Dropout):
                    h = _apply_siamese_shared_dropout(
                        h,
                        sub,
                        num_branches=num_branches,
                        synchronize_dropout_rng=synchronize_dropout_rng,
                    )
                else:
                    h = sub(h)
        elif isinstance(mod, torch.nn.Dropout):
            h = _apply_siamese_shared_dropout(
                h,
                mod,
                num_branches=num_branches,
                synchronize_dropout_rng=synchronize_dropout_rng,
            )
        else:
            h = mod(h)
    return h


def _run_fast_padded_transformer(
    ctx,
    padded: torch.Tensor,
    mask: torch.Tensor,
    *,
    causal_override: bool | None = None,
    target_col_idx: torch.Tensor | None = None,
    num_branches: int = 1,
    synchronize_dropout_rng: bool = True,
) -> torch.Tensor:
    """在已构建好的 [G, L_max, D] 填充张量与 [G, L_max] 掩码上直接执行 Transformer（零 Python 循环）。

    当传入 `target_col_idx`（形状 [G]）且满足直接层循环条件时，第 16 层（末层）仅对
    每个窗口的目标词位计算 Query、输出投影与 FeedForward，省去末层 80% 冗余计算并直接返回 [G, D]。
    """
    use_causal_attention = (
        ctx.causal_attention
        if causal_override is None
        else bool(causal_override)
    )
    if _can_use_direct_layer_loop(ctx, use_causal_attention=use_causal_attention):
        enc = ctx.encoder
        bsz, max_len, dim = padded.shape
        work_dtype = (
            torch.get_autocast_gpu_dtype()
            if (padded.is_cuda and torch.is_autocast_enabled())
            else padded.dtype
        )
        x = padded.to(dtype=work_dtype)
        pos = torch.arange(max_len, device=x.device)
        if enc.rotary_pos_emb is not None:
            freqs, _ = enc.rotary_pos_emb(pos)
            if freqs.ndim == 3:
                freqs = freqs[0]
            cos_full = freqs.cos().to(dtype=work_dtype)[None, None, :, :]
            sin_full = freqs.sin().to(dtype=work_dtype)[None, None, :, :]
            if target_col_idx is not None:
                cos_t = (
                    cos_full[:, :, target_col_idx, :]
                    .squeeze(0)
                    .transpose(0, 1)
                    .unsqueeze(2)
                )
                sin_t = (
                    sin_full[:, :, target_col_idx, :]
                    .squeeze(0)
                    .transpose(0, 1)
                    .unsqueeze(2)
                )
            else:
                cos_t = sin_t = None
        else:
            cos_full = sin_full = cos_t = sin_t = None

        key_mask = mask[:, None, None, :]
        mask_3d = mask.unsqueeze(-1)
        num_layers = len(enc.layers)
        row_arange = (
            torch.arange(bsz, device=x.device, dtype=torch.long)
            if target_col_idx is not None
            else None
        )

        for ind, (layer_type, layer_tuple) in enumerate(
            zip(enc.layer_types, enc.layers)
        ):
            norms, block, residual_fn = layer_tuple
            pre_norm, post_branch_norm, post_main_norm = norms
            is_last_attn = target_col_idx is not None and ind == num_layers - 2
            is_last_ff = target_col_idx is not None and ind == num_layers - 1

            if is_last_attn:
                residual = x[row_arange, target_col_idx].unsqueeze(1)
                x_norm = (
                    pre_norm(x).to(dtype=work_dtype)
                    if pre_norm is not None
                    else x
                )
                q_in = x_norm[row_arange, target_col_idx].unsqueeze(1)
                heads = int(block.heads)
                q = block.to_q(q_in).view(bsz, 1, heads, -1).transpose(1, 2)
                k = block.to_k(x_norm).view(bsz, max_len, heads, -1).transpose(1, 2)
                v = block.to_v(x_norm).view(bsz, max_len, heads, -1).transpose(1, 2)
                if cos_full is not None:
                    q = _apply_rotary_pos_emb_fast(q, cos_t, sin_t)
                    k = _apply_rotary_pos_emb_fast(k, cos_full, sin_full)
                scale = q.shape[-1] ** -0.5
                sim = torch.matmul(q, k.transpose(-1, -2)) * scale
                sim = sim.masked_fill(~key_mask, -torch.finfo(sim.dtype).max)
                attn = F.softmax(sim, dim=-1).to(dtype=work_dtype)
                attn = _apply_siamese_shared_dropout(
                    attn,
                    block.attend.attn_dropout,
                    num_branches=num_branches,
                    synchronize_dropout_rng=synchronize_dropout_rng,
                )
                out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, 1, -1)
                out = block.to_out(out)
                if post_branch_norm is not None:
                    out = post_branch_norm(out)
                x = residual_fn(out, residual).to(dtype=work_dtype)
                if post_main_norm is not None:
                    x = post_main_norm(x).to(dtype=work_dtype)
            elif is_last_ff:
                residual = x
                x_norm = (
                    pre_norm(x).to(dtype=work_dtype)
                    if pre_norm is not None
                    else x
                )
                out = _run_ff_with_shared_dropout(
                    block,
                    x_norm,
                    num_branches=num_branches,
                    synchronize_dropout_rng=synchronize_dropout_rng,
                )
                if post_branch_norm is not None:
                    out = post_branch_norm(out)
                x = residual_fn(out, residual).to(dtype=work_dtype)
                if post_main_norm is not None:
                    x = post_main_norm(x).to(dtype=work_dtype)
            elif layer_type == "a":
                residual = x
                x_norm = (
                    pre_norm(x).to(dtype=work_dtype)
                    if pre_norm is not None
                    else x
                )
                heads = int(block.heads)
                q = block.to_q(x_norm).view(bsz, max_len, heads, -1).transpose(1, 2)
                k = block.to_k(x_norm).view(bsz, max_len, heads, -1).transpose(1, 2)
                v = block.to_v(x_norm).view(bsz, max_len, heads, -1).transpose(1, 2)
                if cos_full is not None:
                    q = _apply_rotary_pos_emb_fast(q, cos_full, sin_full)
                    k = _apply_rotary_pos_emb_fast(k, cos_full, sin_full)
                scale = q.shape[-1] ** -0.5
                sim = torch.matmul(q, k.transpose(-1, -2)) * scale
                sim = sim.masked_fill(~key_mask, -torch.finfo(sim.dtype).max)
                attn = F.softmax(sim, dim=-1).to(dtype=work_dtype)
                attn = _apply_siamese_shared_dropout(
                    attn,
                    block.attend.attn_dropout,
                    num_branches=num_branches,
                    synchronize_dropout_rng=synchronize_dropout_rng,
                )
                out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, max_len, -1)
                out = block.to_out(out) * mask_3d
                if post_branch_norm is not None:
                    out = post_branch_norm(out)
                x = residual_fn(out, residual).to(dtype=work_dtype)
                if post_main_norm is not None:
                    x = post_main_norm(x).to(dtype=work_dtype)
            else:
                residual = x
                x_norm = (
                    pre_norm(x).to(dtype=work_dtype)
                    if pre_norm is not None
                    else x
                )
                out = _run_ff_with_shared_dropout(
                    block,
                    x_norm,
                    num_branches=num_branches,
                    synchronize_dropout_rng=synchronize_dropout_rng,
                )
                if post_branch_norm is not None:
                    out = post_branch_norm(out)
                x = residual_fn(out, residual).to(dtype=work_dtype)
                if post_main_norm is not None:
                    x = post_main_norm(x).to(dtype=work_dtype)

        x = enc.final_norm(x)
        return x.squeeze(1) if target_col_idx is not None else x

    max_length = int(padded.shape[1])
    attention_bias = None
    if ctx.local_relative_position_bias is not None:
        attention_bias = ctx.local_relative_position_bias(
            max_length,
            device=padded.device,
            dtype=padded.dtype,
            heads=(
                ctx.total_attention_heads
                if ctx.local_attention_heads
                else None
            ),
            active_heads=(
                ctx.local_attention_heads
                if ctx.local_attention_heads
                else None
            ),
        )
        if ctx.adaptive_directional_bias_gate is not None:
            directional_gates = ctx.adaptive_directional_bias_gate(padded)
            positions = torch.arange(max_length, device=padded.device)
            relative = positions[None, :] - positions[:, None]
            directional_scale = torch.ones(
                (padded.shape[0], max_length, max_length),
                device=padded.device,
                dtype=padded.dtype,
            )
            directional_scale = torch.where(
                relative[None] < 0,
                directional_gates[:, :, 0, None],
                directional_scale,
            )
            directional_scale = torch.where(
                relative[None] > 0,
                directional_gates[:, :, 1, None],
                directional_scale,
            )
            attention_bias = attention_bias * directional_scale[:, None]
    attention_mask = ctx.local_global_attention_mask(max_length, padded.device)
    encoded = padded
    for _ in range(ctx.shared_passes):
        encoder_kwargs = {
            "mask": mask,
            "attn_mask": attention_mask,
            "attn_bias": attention_bias,
        }
        if use_causal_attention:
            encoder_kwargs["causal"] = True
        encoded = ctx.encoder(encoded, **encoder_kwargs)
    if target_col_idx is not None:
        row_arange = torch.arange(
            encoded.shape[0], device=encoded.device, dtype=torch.long
        )
        return encoded[row_arange, target_col_idx]
    return encoded


def build_padded_revision_window_plan(
    model,
    group_indices: torch.Tensor,
    *,
    cutoff_generator=None,
    cutoff_policy: str = "all",
) -> dict[str, torch.Tensor]:
    """在 CPU NumPy 中一次性构建 2D 局部窗口填充索引矩阵 [G, L_max] 与掩码，彻底消除逐窗口 CUDA 小算子循环。"""
    if not getattr(model, "prefix_revision_enabled", False):
        raise ValueError("模型未启用共享前缀回修。")
    if cutoff_policy not in {"sample_one", "all"}:
        raise ValueError("回修截止位置策略必须是 sample_one 或 all。")

    device = group_indices.device
    batch_len = int(len(group_indices))
    windows: list[list[int]] = []
    target_group_indices: list[int] = []
    target_col_indices: list[int] = []
    revision_rows: list[int] = []
    revision_delays: list[int] = []
    virtual_group = 0

    for indices in model._ordered_groups(group_indices):
        cutoffs = (
            range(len(indices))
            if cutoff_policy == "all"
            else (
                int(
                    torch.randint(
                        0,
                        len(indices),
                        (1,),
                        generator=cutoff_generator,
                    ).item()
                ),
            )
        )
        for cutoff in cutoffs:
            if model.prefix_revision_protocol == "full_prefix":
                win_indices = indices[: cutoff + 1]
                windows.append(win_indices)
                for delay in model.prefix_revision_horizons:
                    target_position = cutoff - delay
                    if target_position < 0:
                        continue
                    target_group_indices.append(virtual_group)
                    target_col_indices.append(target_position)
                    revision_rows.append(indices[target_position])
                    revision_delays.append(delay)
                virtual_group += 1
                continue

            for delay in model.prefix_revision_horizons:
                target_position = cutoff - delay
                if target_position < 0:
                    continue
                start, stop, target_offset = model._revision_window_bounds(
                    target_position, delay, len(indices)
                )
                if stop != cutoff + 1:
                    raise RuntimeError("局部窗口滚动回修训练边界不一致。")
                win_indices = indices[start:stop]
                windows.append(win_indices)
                target_group_indices.append(virtual_group)
                target_col_indices.append(target_offset)
                revision_rows.append(indices[target_position])
                revision_delays.append(delay)
                virtual_group += 1

    num_windows = len(windows)
    max_win_len = max((len(w) for w in windows), default=1)
    # 填充位统一指向 batch_len（即 F.pad(z_batch, (0, 0, 0, 1)) 末尾的全零行）
    win_idx_np = np.full((num_windows, max_win_len), batch_len, dtype=np.int64)
    mask_np = np.zeros((num_windows, max_win_len), dtype=np.bool_)
    for g, w in enumerate(windows):
        w_len = len(w)
        win_idx_np[g, :w_len] = w
        mask_np[g, :w_len] = True

    return {
        "win_idx_2d": torch.from_numpy(win_idx_np).to(
            device=device, dtype=torch.long, non_blocking=True
        ),
        "mask_2d": torch.from_numpy(mask_np).to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
        "target_group_idx": torch.as_tensor(
            target_group_indices, device=device, dtype=torch.long
        ),
        "target_col_idx": torch.as_tensor(
            target_col_indices, device=device, dtype=torch.long
        ),
        "revision_rows": torch.as_tensor(
            revision_rows, device=device, dtype=torch.long
        ),
        "revision_delays": torch.as_tensor(
            revision_delays, device=device, dtype=torch.long
        ),
    }


def build_padded_context_group_plan(
    model,
    group_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """为 Stage 2.1 (context) 构建 2D 句子填充索引矩阵与逆向还原索引。"""
    device = group_indices.device
    batch_len = int(len(group_indices))
    if getattr(model, "context_mode", "grouped") == "singleton":
        group_values = list(range(batch_len))
    else:
        group_values = group_indices.detach().cpu().tolist()

    groups: dict[int, list[int]] = {}
    for row_index, group in enumerate(group_values):
        groups.setdefault(int(group), []).append(row_index)

    group_list = list(groups.values())
    num_groups = len(group_list)
    max_len = max((len(g) for g in group_list), default=1)
    win_idx_np = np.full((num_groups, max_len), batch_len, dtype=np.int64)
    mask_np = np.zeros((num_groups, max_len), dtype=np.bool_)
    row_to_group = np.zeros(batch_len, dtype=np.int64)
    row_to_col = np.zeros(batch_len, dtype=np.int64)

    for g_idx, rows in enumerate(group_list):
        g_len = len(rows)
        win_idx_np[g_idx, :g_len] = rows
        mask_np[g_idx, :g_len] = True
        for col_idx, r in enumerate(rows):
            row_to_group[r] = g_idx
            row_to_col[r] = col_idx

    return {
        "win_idx_2d": torch.from_numpy(win_idx_np).to(
            device=device, dtype=torch.long, non_blocking=True
        ),
        "mask_2d": torch.from_numpy(mask_np).to(
            device=device, dtype=torch.bool, non_blocking=True
        ),
        "row_to_group": torch.from_numpy(row_to_group).to(
            device=device, dtype=torch.long, non_blocking=True
        ),
        "row_to_col": torch.from_numpy(row_to_col).to(
            device=device, dtype=torch.long, non_blocking=True
        ),
    }


def build_revision_window_plan(
    model,
    group_indices: torch.Tensor,
    *,
    cutoff_generator=None,
    cutoff_policy: str = "all",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """为当前 batch 构建局部窗口滚动修订索引计划（Clean 与 Shifted 分支 100% 共享同构时间索引）。"""
    if not getattr(model, "prefix_revision_enabled", False):
        raise ValueError("模型未启用共享前缀回修。")
    if cutoff_policy not in {"sample_one", "all"}:
        raise ValueError("回修截止位置策略必须是 sample_one 或 all。")

    device = group_indices.device
    flat_batch_positions: list[int] = []
    prefix_groups_np: list[np.ndarray] = []
    revision_rows: list[int] = []
    revision_delays: list[int] = []
    prefix_target_locations: list[int] = []
    cursor = 0
    virtual_group = 0

    for indices in model._ordered_groups(group_indices):
        cutoffs = (
            range(len(indices))
            if cutoff_policy == "all"
            else (
                int(
                    torch.randint(
                        0,
                        len(indices),
                        (1,),
                        generator=cutoff_generator,
                    ).item()
                ),
            )
        )
        for cutoff in cutoffs:
            if model.prefix_revision_protocol == "full_prefix":
                win_indices = indices[: cutoff + 1]
                flat_batch_positions.extend(win_indices)
                prefix_groups_np.append(
                    np.full(len(win_indices), virtual_group, dtype=np.int64)
                )
                for delay in model.prefix_revision_horizons:
                    target_position = cutoff - delay
                    if target_position < 0:
                        continue
                    prefix_target_locations.append(cursor + target_position)
                    revision_rows.append(indices[target_position])
                    revision_delays.append(delay)
                cursor += len(win_indices)
                virtual_group += 1
                continue

            for delay in model.prefix_revision_horizons:
                target_position = cutoff - delay
                if target_position < 0:
                    continue
                start, stop, target_offset = model._revision_window_bounds(
                    target_position, delay, len(indices)
                )
                if stop != cutoff + 1:
                    raise RuntimeError("局部窗口滚动回修训练边界不一致。")
                win_indices = indices[start:stop]
                flat_batch_positions.extend(win_indices)
                prefix_groups_np.append(
                    np.full(len(win_indices), virtual_group, dtype=np.int64)
                )
                prefix_target_locations.append(cursor + target_offset)
                revision_rows.append(indices[target_position])
                revision_delays.append(delay)
                cursor += len(win_indices)
                virtual_group += 1

    return (
        torch.as_tensor(flat_batch_positions, device=device, dtype=torch.long),
        torch.from_numpy(np.concatenate(prefix_groups_np, axis=0)).to(
            device=device, dtype=group_indices.dtype
        ),
        torch.as_tensor(
            prefix_target_locations, device=device, dtype=torch.long
        ),
        torch.as_tensor(revision_rows, device=device, dtype=torch.long),
        torch.as_tensor(revision_delays, device=device, dtype=torch.long),
    )


def forward_revision_from_preextracted(
    model,
    clean_z_batch: torch.Tensor | None,
    group_indices: torch.Tensor,
    *,
    shifted_z_batches: list[torch.Tensor] | None = None,
    cutoff_generator=None,
    cutoff_policy: str = "all",
    synchronize_dropout_rng: bool = True,
    padded_plan: dict[str, torch.Tensor] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    """在预提取 CNN 特征上运行共享上下文 Transformer，返回 (y_clean, y_shift_mean, revision_rows, revision_delays)。"""
    if _can_use_fast_padded_transformer(model):
        plan = (
            padded_plan
            if padded_plan is not None
            else build_padded_revision_window_plan(
                model,
                group_indices,
                cutoff_generator=cutoff_generator,
                cutoff_policy=cutoff_policy,
            )
        )
        mask_2d = plan["mask_2d"]
        target_group_idx = plan["target_group_idx"]
        target_col_idx = plan["target_col_idx"]
        revision_rows = plan["revision_rows"]
        revision_delays = plan["revision_delays"]

        if "clean_padded" in plan:
            clean_padded = plan["clean_padded"]
            shifted_padded_list = plan.get("shifted_padded_list")
        else:
            win_idx_2d = plan["win_idx_2d"]
            clean_padded = F.pad(clean_z_batch, (0, 0, 0, 1))[win_idx_2d]
            shifted_padded_list = (
                [F.pad(shift_z, (0, 0, 0, 1))[win_idx_2d] for shift_z in shifted_z_batches]
                if shifted_z_batches
                else None
            )

        g_len = int(clean_padded.shape[0])
        can_prune_last_layer = int(target_col_idx.shape[0]) == g_len

        if not shifted_padded_list:
            clean_out = _run_fast_padded_transformer(
                model.context_transformer,
                clean_padded,
                mask_2d,
                causal_override=False,
                target_col_idx=target_col_idx if can_prune_last_layer else None,
            )
            if can_prune_last_layer:
                y_clean = F.normalize(clean_out, dim=-1)
            else:
                y_clean = F.normalize(
                    clean_out[target_group_idx, target_col_idx], dim=-1
                )
            return y_clean, None, revision_rows, revision_delays

        if _can_use_direct_layer_loop(
            model.context_transformer, use_causal_attention=False
        ):
            num_branches = 1 + len(shifted_padded_list)
            all_padded = torch.cat([clean_padded, *shifted_padded_list], dim=0)
            all_mask = mask_2d.repeat(num_branches, 1)
            all_target_col = (
                target_col_idx.repeat(num_branches)
                if can_prune_last_layer
                else None
            )
            all_out = _run_fast_padded_transformer(
                model.context_transformer,
                all_padded,
                all_mask,
                causal_override=False,
                target_col_idx=all_target_col,
                num_branches=num_branches,
                synchronize_dropout_rng=synchronize_dropout_rng,
            )
            if can_prune_last_layer:
                all_norm = F.normalize(all_out, dim=-1).view(
                    num_branches, g_len, -1
                )
            else:
                all_reshaped = all_out.view(
                    num_branches, g_len, all_out.shape[1], all_out.shape[2]
                )
                all_norm = F.normalize(
                    all_reshaped[:, target_group_idx, target_col_idx], dim=-1
                )
            y_clean = all_norm[0]
            y_shift_mean = all_norm[1:].mean(dim=0)
            return y_clean, y_shift_mean, revision_rows, revision_delays

        cpu_rng_state = (
            torch.get_rng_state() if synchronize_dropout_rng else None
        )
        cuda_device = (
            clean_padded.device.index
            if (synchronize_dropout_rng and clean_padded.is_cuda)
            else None
        )
        cuda_rng_state = (
            torch.cuda.get_rng_state(cuda_device)
            if cuda_device is not None
            else None
        )

        clean_revised = _run_fast_padded_transformer(
            model.context_transformer,
            clean_padded,
            mask_2d,
            causal_override=False,
        )
        y_clean = F.normalize(
            clean_revised[target_group_idx, target_col_idx], dim=-1
        )

        fork_devices = [] if cuda_device is None else [cuda_device]
        y_shift_list: list[torch.Tensor] = []
        for shift_padded in shifted_padded_list:
            if synchronize_dropout_rng and cpu_rng_state is not None:
                with torch.random.fork_rng(devices=fork_devices):
                    torch.set_rng_state(cpu_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state(cuda_rng_state, cuda_device)
                    shift_revised = _run_fast_padded_transformer(
                        model.context_transformer,
                        shift_padded,
                        mask_2d,
                        causal_override=False,
                    )
            else:
                shift_revised = _run_fast_padded_transformer(
                    model.context_transformer,
                    shift_padded,
                    mask_2d,
                    causal_override=False,
                )
            y_shift_m = F.normalize(
                shift_revised[target_group_idx, target_col_idx], dim=-1
            )
            y_shift_list.append(y_shift_m)

        y_shift_mean = torch.stack(y_shift_list, dim=0).mean(dim=0)
        return y_clean, y_shift_mean, revision_rows, revision_delays

    (
        window_positions,
        prefix_group_indices,
        target_locations,
        revision_rows,
        revision_delays,
    ) = build_revision_window_plan(
        model,
        group_indices,
        cutoff_generator=cutoff_generator,
        cutoff_policy=cutoff_policy,
    )

    clean_features = clean_z_batch[window_positions]
    if model.pre_transformer_trajectory is not None:
        clean_features = model.pre_transformer_trajectory(
            clean_features, prefix_group_indices
        )

    cpu_rng_state = torch.get_rng_state() if synchronize_dropout_rng else None
    cuda_device = (
        clean_features.device.index
        if (synchronize_dropout_rng and clean_features.is_cuda)
        else None
    )
    cuda_rng_state = (
        torch.cuda.get_rng_state(cuda_device)
        if cuda_device is not None
        else None
    )

    clean_revised = model.context_transformer(
        clean_features,
        prefix_group_indices,
        causal_override=False,
    )
    y_clean = F.normalize(clean_revised[target_locations], dim=-1)

    if not shifted_z_batches:
        return y_clean, None, revision_rows, revision_delays

    fork_devices = [] if cuda_device is None else [cuda_device]
    y_shift_list: list[torch.Tensor] = []
    for shift_z in shifted_z_batches:
        shift_features = shift_z[window_positions]
        if model.pre_transformer_trajectory is not None:
            shift_features = model.pre_transformer_trajectory(
                shift_features, prefix_group_indices
            )
        if synchronize_dropout_rng and cpu_rng_state is not None:
            with torch.random.fork_rng(devices=fork_devices):
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, cuda_device)
                shift_revised = model.context_transformer(
                    shift_features,
                    prefix_group_indices,
                    causal_override=False,
                )
        else:
            shift_revised = model.context_transformer(
                shift_features,
                prefix_group_indices,
                causal_override=False,
            )
        # 严格保持梯度连通，绝不 detach！
        y_shift_m = F.normalize(shift_revised[target_locations], dim=-1)
        y_shift_list.append(y_shift_m)

    y_shift_mean = torch.stack(y_shift_list, dim=0).mean(dim=0)
    return y_clean, y_shift_mean, revision_rows, revision_delays


def forward_context_from_preextracted(
    model,
    clean_z_batch: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    shifted_z_batches: list[torch.Tensor] | None = None,
    synchronize_dropout_rng: bool = True,
    padded_plan: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """针对 Stage 2.1 (context) 在预提取 CNN 特征上运行共享 Transformer，返回 (y_clean, y_shift_mean)。"""
    if _can_use_fast_padded_transformer(model):
        plan = (
            padded_plan
            if padded_plan is not None
            else build_padded_context_group_plan(model, group_indices)
        )
        win_idx_2d = plan["win_idx_2d"]
        mask_2d = plan["mask_2d"]
        row_to_group = plan["row_to_group"]
        row_to_col = plan["row_to_col"]

        clean_padded = F.pad(clean_z_batch, (0, 0, 0, 1))[win_idx_2d]
        if not shifted_z_batches:
            clean_encoded = _run_fast_padded_transformer(
                model.context_transformer, clean_padded, mask_2d
            )
            y_clean = F.normalize(
                clean_encoded[row_to_group, row_to_col], dim=-1
            )
            return y_clean, None

        if _can_use_direct_layer_loop(
            model.context_transformer,
            use_causal_attention=bool(
                model.context_transformer.causal_attention
            ),
        ):
            shifted_padded_list = [
                F.pad(shift_z, (0, 0, 0, 1))[win_idx_2d]
                for shift_z in shifted_z_batches
            ]
            num_branches = 1 + len(shifted_padded_list)
            g_len = int(clean_padded.shape[0])
            all_padded = torch.cat([clean_padded, *shifted_padded_list], dim=0)
            all_mask = mask_2d.repeat(num_branches, 1)
            all_encoded = _run_fast_padded_transformer(
                model.context_transformer,
                all_padded,
                all_mask,
                num_branches=num_branches,
                synchronize_dropout_rng=synchronize_dropout_rng,
            ).view(num_branches, g_len, clean_padded.shape[1], -1)
            all_norm = F.normalize(
                all_encoded[:, row_to_group, row_to_col], dim=-1
            )
            return all_norm[0], all_norm[1:].mean(dim=0)

        cpu_rng_state = (
            torch.get_rng_state() if synchronize_dropout_rng else None
        )
        cuda_device = (
            clean_padded.device.index
            if (synchronize_dropout_rng and clean_padded.is_cuda)
            else None
        )
        cuda_rng_state = (
            torch.cuda.get_rng_state(cuda_device)
            if cuda_device is not None
            else None
        )

        clean_encoded = _run_fast_padded_transformer(
            model.context_transformer, clean_padded, mask_2d
        )
        y_clean = F.normalize(clean_encoded[row_to_group, row_to_col], dim=-1)

        fork_devices = [] if cuda_device is None else [cuda_device]
        y_shift_list: list[torch.Tensor] = []
        for shift_z in shifted_z_batches:
            shift_padded = F.pad(shift_z, (0, 0, 0, 1))[win_idx_2d]
            if synchronize_dropout_rng and cpu_rng_state is not None:
                with torch.random.fork_rng(devices=fork_devices):
                    torch.set_rng_state(cpu_rng_state)
                    if cuda_rng_state is not None:
                        torch.cuda.set_rng_state(cuda_rng_state, cuda_device)
                    shift_encoded = _run_fast_padded_transformer(
                        model.context_transformer, shift_padded, mask_2d
                    )
            else:
                shift_encoded = _run_fast_padded_transformer(
                    model.context_transformer, shift_padded, mask_2d
                )
            y_shift_list.append(
                F.normalize(shift_encoded[row_to_group, row_to_col], dim=-1)
            )

        y_shift_mean = torch.stack(y_shift_list, dim=0).mean(dim=0)
        return y_clean, y_shift_mean

    clean_input = clean_z_batch
    if model.context_mode == "singleton":
        group_indices = torch.arange(
            len(clean_input), device=clean_input.device, dtype=torch.long
        )
    if model.pre_transformer_trajectory is not None:
        clean_input = model.pre_transformer_trajectory(
            clean_input, group_indices
        )

    cpu_rng_state = torch.get_rng_state() if synchronize_dropout_rng else None
    cuda_device = (
        clean_input.device.index
        if (synchronize_dropout_rng and clean_input.is_cuda)
        else None
    )
    cuda_rng_state = (
        torch.cuda.get_rng_state(cuda_device)
        if cuda_device is not None
        else None
    )

    clean_ctx = model.context_transformer(clean_input, group_indices)
    y_clean = F.normalize(clean_ctx, dim=-1)

    if not shifted_z_batches:
        return y_clean, None

    fork_devices = [] if cuda_device is None else [cuda_device]
    y_shift_list: list[torch.Tensor] = []
    for shift_z in shifted_z_batches:
        shift_input = shift_z
        if model.pre_transformer_trajectory is not None:
            shift_input = model.pre_transformer_trajectory(
                shift_input, group_indices
            )
        if synchronize_dropout_rng and cpu_rng_state is not None:
            with torch.random.fork_rng(devices=fork_devices):
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, cuda_device)
                shift_ctx = model.context_transformer(
                    shift_input, group_indices
                )
        else:
            shift_ctx = model.context_transformer(shift_input, group_indices)
        y_shift_list.append(F.normalize(shift_ctx, dim=-1))

    y_shift_mean = torch.stack(y_shift_list, dim=0).mean(dim=0)
    return y_clean, y_shift_mean


def train_one_epoch_siamese_debiased(
    model,
    loss_module,
    batch_sampler,
    preextracted_cache: dict,
    optimizer,
    scaler,
    device: torch.device,
    training_config: dict,
    siamese_config: dict,
    *,
    centered_vocab_bank: torch.Tensor | None = None,
    epoch_index: int = 0,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[float, int, int, dict]:
    """基于预提取冻结 CNN 特征执行单轮孪生共模抵消 + 错位零化正则训练。"""
    model.train()
    model.brain_encoder.eval()
    loss_module.train()

    amp_dtype = resolve_amp_dtype(training_config)
    use_amp = (
        bool(training_config.get("amp", True))
        and device.type == "cuda"
        and amp_dtype in {torch.float16, torch.bfloat16}
    )

    siamese_enabled = bool(siamese_config.get("enabled", False))
    eta_train = float(siamese_config.get("eta_train", 0.8))
    lambda_null = float(siamese_config.get("lambda_null", 10.0))
    shifts_per_step = int(siamese_config.get("shifts_per_step", 2))
    sync_rng = bool(siamese_config.get("synchronize_dropout_rng", True))

    train_clean_z: torch.Tensor = preextracted_cache["train_clean_z"]
    train_clean_z_pad: torch.Tensor | None = preextracted_cache.get(
        "train_clean_z_pad"
    )
    train_shifted_z: dict[float, torch.Tensor] = preextracted_cache.get(
        "train_shifted_z", {}
    )
    train_shifted_z_pad: dict[float, torch.Tensor] = preextracted_cache.get(
        "train_shifted_z_pad", {}
    )
    shift_keys = sorted(train_shifted_z.keys())
    if siamese_enabled and not shift_keys:
        raise ValueError(
            "启用 siamese_debiased 时 preextracted_cache 缺少 train_shifted_z。"
        )
    if siamese_enabled and lambda_null > 0.0 and centered_vocab_bank is None:
        raise ValueError(
            "启用错位零化正则 (lambda_null > 0) 时必须提供 centered_vocab_bank。"
        )

    train_meta = preextracted_cache["train_meta"]
    targets_all: torch.Tensor = train_meta["targets"]
    sentence_indices_all: torch.Tensor = train_meta["sentence_indices"]
    static_rev_table: dict[str, torch.Tensor] | None = preextracted_cache.get(
        "train_static_revision_table"
    )

    seed = int(training_config.get("seed", 0))
    prefix_generator = torch.Generator(device="cpu")
    prefix_generator.manual_seed(seed * 1_000_033 + int(epoch_index) + 193)
    shift_sample_generator = torch.Generator(device="cpu")
    shift_sample_generator.manual_seed(seed * 1_000_039 + int(epoch_index) + 307)

    prefix_revision_enabled = bool(
        getattr(model, "prefix_revision_enabled", False)
    )
    cutoff_policy = str(training_config.get("revision_cutoff_policy", "all"))
    horizons = (
        tuple(int(d) for d in getattr(model, "prefix_revision_horizons", (0, 1, 2)))
        if prefix_revision_enabled
        else ()
    )
    use_static_rev_table = (
        prefix_revision_enabled
        and static_rev_table is not None
        and train_clean_z_pad is not None
        and cutoff_policy == "all"
        and bool(getattr(batch_sampler, "preserve_complete_groups", False))
        and _can_use_fast_padded_transformer(model)
    )
    trainable_model_params = [
        p for p in model.parameters() if p.requires_grad
    ]

    total_loss_sum = 0.0
    siglip_loss_sum = 0.0
    null_loss_sum = 0.0
    sample_count = 0
    optimizer_updates = 0
    batches_seen = 0
    revision_supervision_count = 0
    revision_delay_counts = {d: 0 for d in horizons}
    revision_delay_siglip_sums = {d: 0.0 for d in horizons}
    revision_delay_null_sums = {d: 0.0 for d in horizons}
    revision_delay_batches = {d: 0 for d in horizons}

    current_lr = (
        float(optimizer.param_groups[0].get("lr", 0.0))
        if getattr(optimizer, "param_groups", None)
        else 0.0
    )
    iterable, pbar = _wrap_progress(
        batch_sampler,
        enabled=show_progress,
        desc=progress_desc or f"Epoch {int(epoch_index) + 1:02d} [siamese]",
        total=len(batch_sampler),
    )

    try:
        for batch_indices in iterable:
            idx_tensor = torch.as_tensor(
                batch_indices, device=device, dtype=torch.long
            )
            batch_targets = targets_all[idx_tensor]
            batch_sentence_indices = sentence_indices_all[idx_tensor]

            chosen_keys: list[float] = []
            if siamese_enabled:
                if shifts_per_step >= len(shift_keys):
                    chosen_keys = shift_keys
                else:
                    perm = torch.randperm(
                        len(shift_keys), generator=shift_sample_generator
                    )[:shifts_per_step].tolist()
                    chosen_keys = [shift_keys[i] for i in perm]

            if use_static_rev_table:
                valid_2d = static_rev_table["row_valid_delay"][idx_tensor]
                win_global_2d = static_rev_table["row_win_global_idx"][
                    idx_tensor
                ][valid_2d]
                mask_2d = static_rev_table["row_win_mask"][idx_tensor][
                    valid_2d
                ]
                target_col_idx = static_rev_table["row_target_col"][
                    idx_tensor
                ][valid_2d]
                horizons_t = static_rev_table["horizons_tensor"]
                b_arange = torch.arange(
                    len(idx_tensor), device=device, dtype=torch.long
                ).unsqueeze(1)
                revision_delays_plan = horizons_t.unsqueeze(0).expand(
                    len(idx_tensor), -1
                )[valid_2d]
                revision_rows_plan = (b_arange - horizons_t.unsqueeze(0))[
                    valid_2d
                ]
                clean_padded = train_clean_z_pad[win_global_2d]
                shifted_padded_list = (
                    [
                        train_shifted_z_pad[k][win_global_2d]
                        for k in chosen_keys
                    ]
                    if siamese_enabled
                    else None
                )
                padded_plan = {
                    "clean_padded": clean_padded,
                    "shifted_padded_list": shifted_padded_list,
                    "mask_2d": mask_2d,
                    "target_group_idx": torch.arange(
                        len(target_col_idx), device=device, dtype=torch.long
                    ),
                    "target_col_idx": target_col_idx,
                    "revision_rows": revision_rows_plan,
                    "revision_delays": revision_delays_plan,
                }
                clean_z_batch = None
                shifted_z_batches = None
            else:
                padded_plan = None
                clean_z_batch = train_clean_z[idx_tensor]
                shifted_z_batches = (
                    [train_shifted_z[k][idx_tensor] for k in chosen_keys]
                    if siamese_enabled
                    else None
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                if prefix_revision_enabled:
                    (
                        y_clean,
                        y_shift_mean,
                        revision_rows,
                        revision_delays,
                    ) = forward_revision_from_preextracted(
                        model,
                        clean_z_batch,
                        batch_sentence_indices,
                        shifted_z_batches=shifted_z_batches,
                        cutoff_generator=prefix_generator,
                        cutoff_policy=cutoff_policy,
                        synchronize_dropout_rng=sync_rng,
                        padded_plan=padded_plan,
                    )
                    if siamese_enabled and y_shift_mean is not None:
                        # 阻断主对比损失向错位分支回传 -eta_train 反向梯度（防止错位分支反向拟合 -E_target 投机），
                        # 错位分支仅受下方 compute_shift_null_loss 的零化正则约束。
                        y_tilde = F.normalize(
                            y_clean - eta_train * y_shift_mean.detach(), dim=-1
                        )
                    else:
                        y_tilde = y_clean

                    horizon_total_losses = []
                    horizon_siglip_losses = []
                    horizon_null_losses = []
                    for delay in horizons:
                        delay_mask = revision_delays == delay
                        d_count = int(delay_mask.sum())
                        if d_count == 0:
                            continue
                        d_siglip = loss_module.forward_rectangular(
                            y_tilde[delay_mask],
                            batch_targets,
                            revision_rows[delay_mask],
                        )
                        if (
                            siamese_enabled
                            and y_shift_mean is not None
                            and lambda_null > 0.0
                            and centered_vocab_bank is not None
                        ):
                            d_null = compute_shift_null_loss(
                                y_shift_mean[delay_mask], centered_vocab_bank
                            )
                        else:
                            d_null = d_siglip * 0.0
                        d_total = d_siglip + lambda_null * d_null
                        horizon_total_losses.append(d_total)
                        horizon_siglip_losses.append(d_siglip)
                        horizon_null_losses.append(d_null)

                        revision_delay_counts[delay] += d_count
                        revision_delay_siglip_sums[delay] += float(
                            d_siglip.detach()
                        )
                        revision_delay_null_sums[delay] += float(
                            d_null.detach()
                        )
                        revision_delay_batches[delay] += 1

                    if not horizon_total_losses:
                        raise RuntimeError("当前 batch 没有可用的滚动回修监督。")
                    loss = torch.stack(horizon_total_losses).mean()
                    batch_siglip = float(
                        torch.stack(horizon_siglip_losses).mean().detach()
                    )
                    batch_null = float(
                        torch.stack(horizon_null_losses).mean().detach()
                    )
                    revision_supervision_count += int(len(y_tilde))
                else:
                    y_clean, y_shift_mean = forward_context_from_preextracted(
                        model,
                        clean_z_batch,
                        batch_sentence_indices,
                        shifted_z_batches=shifted_z_batches,
                        synchronize_dropout_rng=sync_rng,
                    )
                    if siamese_enabled and y_shift_mean is not None:
                        # 阻断主对比损失向错位分支回传 -eta_train 反向梯度，错位分支仅受 L_null 零化约束
                        y_tilde = F.normalize(
                            y_clean - eta_train * y_shift_mean.detach(), dim=-1
                        )
                    else:
                        y_tilde = y_clean
                    d_siglip = loss_module(y_tilde, batch_targets)
                    if (
                        siamese_enabled
                        and y_shift_mean is not None
                        and lambda_null > 0.0
                        and centered_vocab_bank is not None
                    ):
                        d_null = compute_shift_null_loss(
                            y_shift_mean, centered_vocab_bank
                        )
                    else:
                        d_null = d_siglip * 0.0
                    loss = d_siglip + lambda_null * d_null
                    batch_siglip = float(d_siglip.detach())
                    batch_null = float(d_null.detach())

            scaler.scale(loss).backward()
            max_grad_norm = float(training_config.get("max_grad_norm", 0.0))
            if max_grad_norm > 0 and trainable_model_params:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_model_params,
                    max_grad_norm,
                    foreach=True if device.type == "cuda" else None,
                )
            previous_scale = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            batches_seen += 1
            if not scaler.is_enabled() or float(scaler.get_scale()) >= previous_scale:
                optimizer_updates += 1

            bs_len = len(idx_tensor)
            loss_val = float(loss.detach())
            total_loss_sum += loss_val * bs_len
            siglip_loss_sum += batch_siglip * bs_len
            null_loss_sum += batch_null * bs_len
            sample_count += bs_len

            if pbar is not None:
                running_tot = total_loss_sum / max(1, sample_count)
                running_sig = siglip_loss_sum / max(1, sample_count)
                running_nul = null_loss_sum / max(1, sample_count)
                pbar.set_postfix(
                    loss=f"{running_tot:.4f}",
                    siglip=f"{running_sig:.4f}",
                    null=f"{running_nul:.4f}",
                    lr=f"{current_lr:.1e}",
                    refresh=False,
                )
    finally:
        if pbar is not None:
            pbar.close()

    denom = max(1, sample_count)
    diagnostics = {
        "mode": "siamese_debiased" if siamese_enabled else "preextracted_frozen_cnn",
        "siamese_debiased_enabled": siamese_enabled,
        "eta_train": eta_train if siamese_enabled else 0.0,
        "lambda_null": lambda_null if siamese_enabled else 0.0,
        "shifts_per_step": shifts_per_step if siamese_enabled else 0,
        "word_loss": float(siglip_loss_sum / denom),
        "siglip_loss": float(siglip_loss_sum / denom),
        "shift_null_loss": float(null_loss_sum / denom),
        "weighted_shift_null_loss": float((lambda_null * null_loss_sum) / denom)
        if siamese_enabled
        else 0.0,
        "total_loss": float(total_loss_sum / denom),
        "prefix_revision": {
            "enabled": prefix_revision_enabled,
            "supervision_count": int(revision_supervision_count),
            "delay_counts": revision_delay_counts,
            "delay_mean_losses": {
                d: (
                    revision_delay_siglip_sums[d] / revision_delay_batches[d]
                    if revision_delay_batches[d]
                    else None
                )
                for d in revision_delay_siglip_sums
            },
            "delay_mean_null_losses": {
                d: (
                    revision_delay_null_sums[d] / revision_delay_batches[d]
                    if revision_delay_batches[d]
                    else None
                )
                for d in revision_delay_null_sums
            },
        },
    }
    return (
        float(total_loss_sum / denom),
        int(optimizer_updates),
        int(batches_seen),
        diagnostics,
    )


@torch.inference_mode()
def encode_from_preextracted_cnn(
    model,
    batch_sampler,
    clean_z_all: torch.Tensor,
    meta: dict,
    device: torch.device,
    *,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict:
    """利用预提取的验证集/测试集 CNN 向量直接运行上下文 Transformer，生成与 encode_loader 100% 兼容的编码字典。"""
    model.eval()
    use_amp = (
        bool(amp)
        and device.type == "cuda"
        and amp_dtype in {torch.float16, torch.bfloat16}
    )
    prefix_revision_enabled = bool(
        getattr(model, "prefix_revision_enabled", False)
    )
    horizons = (
        tuple(int(d) for d in getattr(model, "prefix_revision_horizons", (0, 1, 2)))
        if prefix_revision_enabled
        else ()
    )

    targets_all: torch.Tensor = meta["targets"]
    subject_indices_all: torch.Tensor = meta["subject_indices"]
    sentence_indices_all: torch.Tensor = meta["sentence_indices"]
    words_all: list[str] = meta["words"]
    event_ids_all: list[str] = meta["event_ids"]
    recording_ids_all: list[str] = meta.get(
        "recording_ids", [""] * len(words_all)
    )
    local_context_radius = int(getattr(model, "local_context_radius", 1))
    clean_z_pad: torch.Tensor | None = meta.get("clean_z_pad")
    static_eval_table: dict[int, dict[str, torch.Tensor]] | None = meta.get(
        "static_revision_table"
    )
    use_static_eval_table = (
        prefix_revision_enabled
        and static_eval_table is not None
        and clean_z_pad is not None
        and bool(getattr(batch_sampler, "preserve_complete_groups", False))
        and _can_use_fast_padded_transformer(model)
    )

    predictions = []
    targets = []
    words: list[str] = []
    event_ids: list[str] = []
    recording_ids: list[str] = []
    subject_indices = []
    sentence_indices = []
    internal_masks = []
    revision_predictions = {d: [] for d in horizons}
    revision_masks = {d: [] for d in horizons}

    iterable, pbar = _wrap_progress(
        batch_sampler,
        enabled=show_progress,
        desc=progress_desc or "Eval [pre-extracted]",
        total=len(batch_sampler),
    )
    try:
        for batch_indices in iterable:
            idx_tensor = torch.as_tensor(
                batch_indices, device=device, dtype=torch.long
            )
            t_batch = targets_all[idx_tensor]
            subj_batch = subject_indices_all[idx_tensor]
            sent_batch = sentence_indices_all[idx_tensor]

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                if prefix_revision_enabled:
                    if use_static_eval_table:
                        delayed: dict[int, torch.Tensor] = {}
                        masks: dict[int, torch.Tensor] = {}
                        for d in horizons:
                            d_tbl = static_eval_table[int(d)]
                            win_idx_2d = d_tbl["win_global_idx"][idx_tensor]
                            mask_2d = d_tbl["win_mask"][idx_tensor]
                            target_col_t = d_tbl["target_col"][idx_tensor]
                            masks[int(d)] = d_tbl["avail_mask"][idx_tensor]
                            revised = _run_fast_padded_transformer(
                                model.context_transformer,
                                clean_z_pad[win_idx_2d],
                                mask_2d,
                                causal_override=False,
                                target_col_idx=target_col_t,
                            )
                            delayed[int(d)] = F.normalize(revised, dim=-1)
                    else:
                        z_batch = clean_z_all[idx_tensor]
                        delayed = _forward_prefix_revisions_from_z(
                            model, z_batch, sent_batch
                        )
                        masks = model.revision_availability_masks(sent_batch)
                    maximum_delay = max(delayed)
                    out_batch = delayed[maximum_delay]
                else:
                    z_batch = clean_z_all[idx_tensor]
                    out_batch, _ = forward_context_from_preextracted(
                        model,
                        z_batch,
                        sent_batch,
                        shifted_z_batches=None,
                        synchronize_dropout_rng=False,
                    )

            predictions.append(out_batch.float().cpu())
            targets.append(t_batch.float().cpu())
            subject_indices.append(subj_batch.cpu())
            sentence_indices.append(sent_batch.cpu())
            internal_masks.append(
                local_context_internal_mask(
                    sent_batch, radius=local_context_radius
                ).cpu()
            )
            for i in batch_indices:
                row_idx = int(i)
                words.append(words_all[row_idx])
                event_ids.append(event_ids_all[row_idx])
                recording_ids.append(recording_ids_all[row_idx])
            if prefix_revision_enabled:
                for d in horizons:
                    revision_predictions[d].append(delayed[d].float().cpu())
                    revision_masks[d].append(masks[d].cpu())
    finally:
        if pbar is not None:
            pbar.close()

    concatenated_internal_masks = torch.cat(internal_masks, dim=0)
    result = {
        "predictions": torch.cat(predictions, dim=0),
        "targets": torch.cat(targets, dim=0),
        "words": words,
        "event_ids": event_ids,
        "recording_ids": recording_ids,
        "subject_indices": torch.cat(subject_indices, dim=0),
        "sentence_indices": torch.cat(sentence_indices, dim=0),
        "local_context_internal_mask": concatenated_internal_masks,
        "three_state_internal_mask": concatenated_internal_masks,
    }
    if prefix_revision_enabled:
        result["revision_predictions"] = {
            d: torch.cat(vals, dim=0)
            for d, vals in revision_predictions.items()
        }
        result["revision_availability_masks"] = {
            d: torch.cat(vals, dim=0) for d, vals in revision_masks.items()
        }
    return result


@torch.inference_mode()
def _forward_prefix_revisions_from_z(
    model,
    brain_embedding: torch.Tensor,
    group_indices: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """与 BrainEmbeddingModel.forward_prefix_revisions 完全等价，但直接接收预提取的 brain_embedding。"""
    outputs: dict[int, torch.Tensor] = {}
    groups = model._ordered_groups(group_indices)
    row_to_group = {}
    for indices in groups:
        for position, row_index in enumerate(indices):
            row_to_group[row_index] = (indices, position)

    batch_size = int(
        getattr(model, "prefix_revision_validation_query_batch_size", 512)
    )
    if _can_use_fast_padded_transformer(model):
        n_rows = int(len(brain_embedding))
        z_pad = F.pad(brain_embedding, (0, 0, 0, 1))
        for delay in model.prefix_revision_horizons:
            delayed = torch.empty_like(brain_embedding)
            for start in range(0, n_rows, batch_size):
                stop = min(start + batch_size, n_rows)
                chunk_len = stop - start
                windows: list[list[int]] = []
                target_cols: list[int] = []
                for row_index in range(start, stop):
                    indices, position = row_to_group[row_index]
                    window_start, window_stop, target_offset = (
                        model._revision_window_bounds(
                            position, delay, len(indices)
                        )
                    )
                    windows.append(indices[window_start:window_stop])
                    target_cols.append(int(target_offset))
                max_w_len = max((len(w) for w in windows), default=1)
                win_idx_np = np.full(
                    (chunk_len, max_w_len), n_rows, dtype=np.int64
                )
                mask_np = np.zeros((chunk_len, max_w_len), dtype=np.bool_)
                for g, w in enumerate(windows):
                    w_len = len(w)
                    win_idx_np[g, :w_len] = w
                    mask_np[g, :w_len] = True
                win_idx_2d = torch.from_numpy(win_idx_np).to(
                    device=brain_embedding.device, dtype=torch.long
                )
                mask_2d = torch.from_numpy(mask_np).to(
                    device=brain_embedding.device, dtype=torch.bool
                )
                target_col_t = torch.as_tensor(
                    target_cols,
                    device=brain_embedding.device,
                    dtype=torch.long,
                )
                revised = _run_fast_padded_transformer(
                    model.context_transformer,
                    z_pad[win_idx_2d],
                    mask_2d,
                    causal_override=False,
                    target_col_idx=target_col_t,
                )
                delayed[start:stop] = F.normalize(revised, dim=-1)
            outputs[int(delay)] = delayed
        return outputs

    for delay in model.prefix_revision_horizons:
        delayed = torch.empty_like(brain_embedding)
        for start in range(0, len(brain_embedding), batch_size):
            stop = min(start + batch_size, len(brain_embedding))
            prefix_parts = []
            prefix_groups = []
            target_locations = []
            cursor = 0
            for virtual_group, row_index in enumerate(range(start, stop)):
                indices, position = row_to_group[row_index]
                window_start, window_stop, target_offset = (
                    model._revision_window_bounds(
                        position, delay, len(indices)
                    )
                )
                selected_indices = torch.as_tensor(
                    indices[window_start:window_stop],
                    device=brain_embedding.device,
                )
                prefix = brain_embedding[selected_indices]
                prefix_parts.append(prefix)
                prefix_groups.append(
                    torch.full(
                        (len(prefix),),
                        virtual_group,
                        device=group_indices.device,
                        dtype=group_indices.dtype,
                    )
                )
                target_locations.append(cursor + target_offset)
                cursor += len(prefix)
            revised = model.context_transformer(
                torch.cat(prefix_parts, dim=0),
                torch.cat(prefix_groups, dim=0),
                causal_override=False,
            )
            delayed[start:stop] = F.normalize(
                revised[
                    torch.as_tensor(
                        target_locations,
                        device=brain_embedding.device,
                        dtype=torch.long,
                    )
                ],
                dim=-1,
            )
        outputs[int(delay)] = delayed
    return outputs
