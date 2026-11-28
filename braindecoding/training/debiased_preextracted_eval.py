"""基于冻结 CNN 预提取特征的快速批量验证编码与训练集因果共模均值 mu0 计算。"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from braindecoding.training.debiased_padded_forward import (
    _can_use_fast_padded_transformer,
    _run_fast_padded_transformer,
)
from braindecoding.training.debiased_preextracted_forward import (
    forward_context_from_preextracted,
)
from braindecoding.training.word_encoder_training import (
    _wrap_progress,
    local_context_internal_mask,
    resolve_amp_dtype,
)

def compute_causal_cms_mu0_from_encoded(
    encoded: dict,
    *,
    source: str = "train",
) -> dict:
    """从训练集或校准集编码字典计算在线 Causal EMA-CMS 初始状态向量 mu_0。"""
    preds = encoded["predictions"].detach().float()
    mu0_pred = preds.mean(dim=0).cpu()
    rev_mu0: dict[int, torch.Tensor] = {}
    rev_preds = encoded.get("revision_predictions")
    rev_masks = encoded.get("revision_availability_masks") or {}
    if isinstance(rev_preds, dict):
        for delay, mat in sorted(rev_preds.items()):
            d_int = int(delay)
            mat_f = mat.detach().float()
            mask = rev_masks.get(delay)
            if mask is None:
                mask = rev_masks.get(d_int)
            if mask is not None and bool(mask.any()):
                valid_mat = mat_f[mask.to(device=mat_f.device, dtype=torch.bool)]
                rev_mu0[d_int] = valid_mat.mean(dim=0).cpu()
            else:
                rev_mu0[d_int] = mat_f.mean(dim=0).cpu()
    return {
        "version": "causal_cms_mu0_v1",
        "source": str(source),
        "event_count": int(len(preds)),
        "predictions": mu0_pred,
        "revision_predictions": rev_mu0,
    }


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
    onset_seconds_all: torch.Tensor | None = meta.get("onset_seconds")
    has_temporal_bias_head = bool(
        getattr(model, "temporal_bias_head", None) is not None
        and float(getattr(model, "temporal_bias_eta", 0.0)) > 0.0
        and onset_seconds_all is not None
    )
    words_all: list[str] = meta["words"]
    event_ids_all: list[str] = meta["event_ids"]
    recording_ids_all: list[str] = meta.get(
        "recording_ids", [""] * len(words_all)
    )
    local_context_radius = int(getattr(model, "local_context_radius", 1))
    clean_z_pad: torch.Tensor | None = meta.get("clean_z_pad")
    clean_self_all: torch.Tensor | None = meta.get("clean_self_all")
    clean_self_pad: torch.Tensor | None = meta.get("clean_self_pad")
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
                        if (
                            getattr(model, "prefix_revision_protocol", "")
                            == "full_prefix"
                            and 0 in static_eval_table
                        ):
                            # Stage 2.3 共享到达前缀去重：每个句段仅运行 L 个唯一前缀 [0..c] 一次（省去 2/3 前向），
                            # 词 i 在视野 r 下的输出直接从唯一前缀 c = min(L-1, i+r) 的恒定下标 target_offset = i 读取！
                            d0_tbl = static_eval_table[0]
                            win_idx_2d = d0_tbl["win_global_idx"][idx_tensor]
                            mask_2d = d0_tbl["win_mask"][idx_tensor]
                            if mask_2d.numel() > 0:
                                batch_max_len = max(
                                    1, int(mask_2d.sum(dim=1).max().item())
                                )
                                if batch_max_len < mask_2d.shape[1]:
                                    win_idx_2d = win_idx_2d[:, :batch_max_len]
                                    mask_2d = mask_2d[:, :batch_max_len]
                            bs_eval = int(len(idx_tensor))
                            b_arange = torch.arange(
                                bs_eval, device=device, dtype=torch.long
                            )
                            group_idx_list = []
                            col_idx_list = []
                            for d in horizons:
                                d_tbl = static_eval_table[int(d)]
                                masks[int(d)] = d_tbl["avail_mask"][idx_tensor]
                                group_idx_list.append(
                                    b_arange + d_tbl["step_offset"][idx_tensor]
                                )
                                col_idx_list.append(
                                    d_tbl["target_col"][idx_tensor]
                                )
                            all_group_idx = torch.cat(group_idx_list, dim=0)
                            all_col_idx = torch.cat(col_idx_list, dim=0)
                            revised_all = _run_fast_padded_transformer(
                                model.context_transformer,
                                clean_z_pad[win_idx_2d],
                                mask_2d,
                                self_padded=(
                                    clean_self_pad[win_idx_2d]
                                    if clean_self_pad is not None
                                    else None
                                ),
                                causal_override=False,
                                target_col_idx=all_col_idx,
                                target_group_idx=all_group_idx,
                            )
                            norm_all = F.normalize(revised_all, dim=-1).view(
                                len(horizons), bs_eval, -1
                            )
                            for h_i, d in enumerate(horizons):
                                delayed[int(d)] = norm_all[h_i]
                        else:
                            for d in horizons:
                                d_tbl = static_eval_table[int(d)]
                                win_idx_2d = d_tbl["win_global_idx"][idx_tensor]
                                mask_2d = d_tbl["win_mask"][idx_tensor]
                                if mask_2d.numel() > 0:
                                    batch_max_len = max(
                                        1, int(mask_2d.sum(dim=1).max().item())
                                    )
                                    if batch_max_len < mask_2d.shape[1]:
                                        win_idx_2d = win_idx_2d[:, :batch_max_len]
                                        mask_2d = mask_2d[:, :batch_max_len]
                                target_col_t = d_tbl["target_col"][idx_tensor]
                                masks[int(d)] = d_tbl["avail_mask"][idx_tensor]
                                revised = _run_fast_padded_transformer(
                                    model.context_transformer,
                                    clean_z_pad[win_idx_2d],
                                    mask_2d,
                                    self_padded=(
                                        clean_self_pad[win_idx_2d]
                                        if clean_self_pad is not None
                                        else None
                                    ),
                                    causal_override=False,
                                    target_col_idx=target_col_t,
                                )
                                delayed[int(d)] = F.normalize(revised, dim=-1)
                    else:
                        z_batch = clean_z_all[idx_tensor]
                        self_batch = (
                            clean_self_all[idx_tensor]
                            if clean_self_all is not None
                            else None
                        )
                        delayed = _forward_prefix_revisions_from_z(
                            model,
                            z_batch,
                            sent_batch,
                            self_features=self_batch,
                        )
                        masks = model.revision_availability_masks(sent_batch)
                    if has_temporal_bias_head and onset_seconds_all is not None:
                        onset_batch = onset_seconds_all[idx_tensor]
                        b_by_delay = (
                            model.temporal_bias_head.predict_group_revisions(
                                onset_batch, sent_batch, horizons
                            )
                        )
                        eta_b = float(model.temporal_bias_eta)
                        for d in horizons:
                            d_int = int(d)
                            delayed[d_int] = F.normalize(
                                delayed[d_int]
                                - eta_b
                                * b_by_delay[d_int].to(
                                    dtype=delayed[d_int].dtype
                                ),
                                dim=-1,
                            )
                    maximum_delay = max(delayed)
                    out_batch = delayed[maximum_delay]
                else:
                    z_batch = clean_z_all[idx_tensor]
                    self_batch = (
                        clean_self_all[idx_tensor]
                        if clean_self_all is not None
                        else None
                    )
                    out_batch, _ = forward_context_from_preextracted(
                        model,
                        z_batch,
                        sent_batch,
                        shifted_z_batches=None,
                        clean_self_batch=self_batch,
                        synchronize_dropout_rng=False,
                    )
                    if has_temporal_bias_head and onset_seconds_all is not None:
                        onset_batch = onset_seconds_all[idx_tensor]
                        b_0 = model.temporal_bias_head.predict_group_revisions(
                            onset_batch, sent_batch, (0,)
                        )[0]
                        out_batch = F.normalize(
                            out_batch
                            - float(model.temporal_bias_eta)
                            * b_0.to(dtype=out_batch.dtype),
                            dim=-1,
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
    *,
    self_features: torch.Tensor | None = None,
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
        self_pad = (
            F.pad(self_features, (0, 0, 0, 1))
            if self_features is not None
            else None
        )
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
                    self_padded=(
                        self_pad[win_idx_2d] if self_pad is not None else None
                    ),
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
