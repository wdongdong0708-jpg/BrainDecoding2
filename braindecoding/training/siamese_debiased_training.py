"""孪生共模抵消（Siamese Debiasing）+ 错位零空间正则训练主循环与统一门面。

各子职责已按单一原则拆分为：
- `debiased_losses.py`：配置解析、中心化词表矩阵及零空间/差分/蒸馏损失；
- `debiased_preextraction.py`：冻结 CNN 特征预提取缓存与静态视界表构建；
- `debiased_padded_forward.py`：定长填充 Transformer 极速内核与窗口规划；
- `debiased_preextracted_forward.py`：Revision / Context 预提取前向分发；
- `debiased_preextracted_eval.py`：预提取快速评估编码与训练集 mu0 计算。
本模块保留主 Epoch 训练循环 `train_one_epoch_siamese_debiased` 并完整重导出上述全部符号。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from braindecoding.training.debiased_losses import (
    DEFAULT_CAUSAL_PREROLL_SHIFT_SECONDS,
    DEFAULT_SIAMESE_SHIFT_SECONDS,
    VALID_DEBIAS_MODES,
    VALID_SHIFT_MODES,
    _normalize_debias_mode,
    _normalize_shift_mode,
    _parse_bool_env,
    _parse_shift_list,
    build_centered_vocabulary_bank,
    compute_delta_shift_null_loss,
    compute_euclidean_cross_horizon_loss,
    compute_norm_damped_debiased_embedding,
    compute_norm_damped_rolling_debiased_embedding,
    compute_scaled_shift_null_loss,
    compute_shift_null_loss,
    compute_temporal_bias_distillation_loss,
    resolve_siamese_debiased_config,
)
from braindecoding.training.debiased_padded_forward import (
    _apply_rotary_pos_emb_fast,
    _apply_siamese_shared_dropout,
    _can_use_direct_layer_loop,
    _can_use_fast_padded_transformer,
    _repeat_target_group_idx,
    _run_fast_padded_transformer,
    _run_ff_with_shared_dropout,
    build_padded_context_group_plan,
    build_padded_revision_window_plan,
    build_revision_window_plan,
)
from braindecoding.training.debiased_preextracted_eval import (
    _forward_prefix_revisions_from_z,
    compute_causal_cms_mu0_from_encoded,
    encode_from_preextracted_cnn,
)
from braindecoding.training.debiased_preextracted_forward import (
    forward_context_from_preextracted,
    forward_revision_from_preextracted,
)
from braindecoding.training.debiased_preextraction import (
    _alloc_padded_gpu_feature_matrix,
    _build_static_revision_eval_table,
    _build_static_revision_train_table,
    _extract_single_dataset_cnn_embeddings,
    _extract_static_dataset_metadata,
    _finalize_gpu_preextracted_bundle,
    _preextracted_cache_path,
    preextract_frozen_cnn_features,
    refresh_preextracted_train_features,
)
from braindecoding.training.word_encoder_training import (
    _wrap_progress,
    resolve_amp_dtype,
)

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
    debias_mode = (
        _normalize_debias_mode(siamese_config.get("debias_mode"))
        or "legacy_siamese"
    )
    shift_mode = (
        _normalize_shift_mode(siamese_config.get("shift_mode"))
        or "deterministic_circular"
    )
    is_causal_tier = debias_mode in {"null_space", "amortized_timing"}
    default_eta = (
        0.0
        if debias_mode == "null_space"
        else (0.50 if debias_mode == "amortized_timing" else 0.8)
    )
    eta_0 = float(
        siamese_config.get(
            "eta_0", siamese_config.get("eta_train", default_eta)
        )
    )
    eta_train = eta_0
    eta_delta = float(
        siamese_config.get(
            "eta_delta", siamese_config.get("eta_delta_train", 0.0)
        )
    )
    lambda_null = float(siamese_config.get("lambda_null", 10.0))
    lambda_delta_null = float(
        siamese_config.get(
            "lambda_delta", siamese_config.get("lambda_delta_null", 0.0)
        )
    )
    lambda_delta = lambda_delta_null
    lambda_distill = float(
        siamese_config.get(
            "lambda_distill", 1.0 if debias_mode == "amortized_timing" else 0.0
        )
    )
    beta_delta_cm = float(
        siamese_config.get("beta_delta_cm", 0.0 if is_causal_tier else 0.50)
    )
    shifts_per_step = int(
        siamese_config.get("shifts_per_step", 1 if is_causal_tier else 2)
    )
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
    if (
        siamese_enabled
        and (
            lambda_null > 0.0
            or (not is_causal_tier and lambda_delta_null > 0.0)
        )
        and centered_vocab_bank is None
    ):
        raise ValueError(
            "启用错位零化正则 (lambda_null > 0 或 lambda_delta_null > 0) 时必须提供 centered_vocab_bank。"
        )

    train_meta = preextracted_cache["train_meta"]
    targets_all: torch.Tensor = train_meta["targets"]
    sentence_indices_all: torch.Tensor = train_meta["sentence_indices"]
    onset_seconds_all: torch.Tensor | None = train_meta.get("onset_seconds")
    if siamese_enabled and debias_mode == "amortized_timing":
        if getattr(model, "temporal_bias_head", None) is None:
            raise ValueError(
                "debias_mode='amortized_timing' 要求模型已挂载 model.temporal_bias_head。"
            )
        if onset_seconds_all is None:
            raise ValueError(
                "debias_mode='amortized_timing' 要求 train_meta 包含 onset_seconds。"
            )
        model.temporal_bias_eta = float(eta_0)
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

    total_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    siglip_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    null_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    delta_null_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    distill_loss_gpu = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = 0
    optimizer_updates = 0
    batches_seen = 0
    revision_supervision_count = 0
    delay_counts_gpu = {
        d: torch.zeros((), device=device, dtype=torch.long) for d in horizons
    }
    delay_siglip_sums_gpu = {
        d: torch.zeros((), device=device, dtype=torch.float64) for d in horizons
    }
    delay_null_sums_gpu = {
        d: torch.zeros((), device=device, dtype=torch.float64) for d in horizons
    }
    delay_delta_null_sums_gpu = {
        d: torch.zeros((), device=device, dtype=torch.float64) for d in horizons
    }
    delay_batches_gpu = {
        d: torch.zeros((), device=device, dtype=torch.long) for d in horizons
    }
    max_horizon = max(horizons, default=0)
    if is_causal_tier:
        compute_null_grad = bool(
            siamese_enabled
            and (
                (lambda_null > 0.0 and centered_vocab_bank is not None)
                or lambda_delta > 0.0
            )
        )
    else:
        compute_null_grad = bool(
            siamese_enabled
            and (
                lambda_null > 0.0
                or lambda_delta_null > 0.0
                or beta_delta_cm > 0.0
            )
            and centered_vocab_bank is not None
        )

    current_lr = (
        float(optimizer.param_groups[0].get("lr", 0.0))
        if getattr(optimizer, "param_groups", None)
        else 0.0
    )
    total_batches = len(batch_sampler)
    iterable, pbar = _wrap_progress(
        batch_sampler,
        enabled=show_progress,
        desc=progress_desc or f"Epoch {int(epoch_index) + 1:02d} [{debias_mode}]",
        total=total_batches,
    )

    try:
        for batch_indices in iterable:
            idx_tensor = torch.as_tensor(
                batch_indices, device=device, dtype=torch.long
            )
            bs_len = int(len(idx_tensor))
            batch_targets = targets_all[idx_tensor]
            batch_sentence_indices = sentence_indices_all[idx_tensor]

            chosen_keys: list[float] = []
            if siamese_enabled:
                perm = torch.randperm(
                    len(shift_keys), generator=shift_sample_generator
                )[: min(shifts_per_step, len(shift_keys))].tolist()
                chosen_keys = [shift_keys[i] for i in perm]

            if use_static_rev_table:
                valid_2d = static_rev_table["row_valid_delay"][idx_tensor]
                unique_col = static_rev_table.get("row_is_unique_win")
                if unique_col is not None:
                    win_sel_2d = unique_col[idx_tensor]
                    target_group_idx = (
                        torch.cumsum(win_sel_2d.view(-1), dim=0).view_as(
                            valid_2d
                        )
                        - 1
                    )[valid_2d]
                else:
                    win_sel_2d = valid_2d
                    target_group_idx = torch.arange(
                        int(valid_2d.sum()), device=device, dtype=torch.long
                    )
                win_global_2d = static_rev_table["row_win_global_idx"][
                    idx_tensor
                ][win_sel_2d]
                mask_2d = static_rev_table["row_win_mask"][idx_tensor][
                    win_sel_2d
                ]
                target_col_idx = static_rev_table["row_target_col"][
                    idx_tensor
                ][valid_2d]
                horizons_t = static_rev_table["horizons_tensor"]
                b_arange = torch.arange(
                    bs_len, device=device, dtype=torch.long
                ).unsqueeze(1)
                revision_delays_plan = horizons_t.unsqueeze(0).expand(
                    bs_len, -1
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
                    "target_group_idx": target_group_idx,
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
                        y_shift_branches,
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
                        compute_null_grad=compute_null_grad,
                        return_shift_branches=True,
                    )
                    k0_has_row: torch.Tensor | None = None
                    k0_shift_mean_by_row: torch.Tensor | None = None
                    k0_shift_branches_by_row: torch.Tensor | None = None
                    b_pred_all: torch.Tensor | None = None
                    if siamese_enabled and y_shift_mean is not None:
                        k0_mask = revision_delays == 0
                        k0_has_row = torch.zeros(
                            bs_len, device=device, dtype=torch.bool
                        )
                        k0_shift_mean_by_row = torch.zeros(
                            (bs_len, y_shift_mean.shape[-1]),
                            device=device,
                            dtype=y_shift_mean.dtype,
                        )
                        if y_shift_branches is not None:
                            k0_shift_branches_by_row = torch.zeros(
                                (
                                    y_shift_branches.shape[0],
                                    bs_len,
                                    y_shift_branches.shape[-1],
                                ),
                                device=device,
                                dtype=y_shift_branches.dtype,
                            )
                        if bool(k0_mask.any()):
                            k0_rows = revision_rows[k0_mask]
                            k0_has_row[k0_rows] = True
                            k0_shift_mean_by_row[k0_rows] = y_shift_mean[
                                k0_mask
                            ].detach()
                            if (
                                k0_shift_branches_by_row is not None
                                and y_shift_branches is not None
                            ):
                                k0_shift_branches_by_row[:, k0_rows, :] = (
                                    y_shift_branches[:, k0_mask, :].detach()
                                )

                        if debias_mode == "null_space":
                            if eta_0 > 0.0:
                                y_tilde = F.normalize(
                                    y_clean - eta_0 * y_shift_mean.detach(),
                                    dim=-1,
                                )
                            else:
                                y_tilde = y_clean
                        elif debias_mode == "amortized_timing":
                            assert onset_seconds_all is not None
                            batch_onset_seconds = onset_seconds_all[idx_tensor]
                            b_pred_all = model.temporal_bias_head.predict_queries(
                                batch_onset_seconds,
                                batch_sentence_indices,
                                revision_rows,
                                revision_delays,
                            )
                            if eta_0 > 0.0:
                                y_tilde = F.normalize(
                                    y_clean
                                    - eta_0
                                    * b_pred_all.detach().to(
                                        dtype=y_clean.dtype
                                    ),
                                    dim=-1,
                                )
                            else:
                                y_tilde = y_clean
                        else:
                            y_shift_k0_all = torch.where(
                                k0_has_row[revision_rows].unsqueeze(-1),
                                k0_shift_mean_by_row[revision_rows],
                                y_shift_mean.detach(),
                            )
                            # 前缀解耦差分共模扣除（eta_0 + eta_delta）+ 雅可比模长阻尼重缩放（Norm-Damped Detach）
                            y_tilde = compute_norm_damped_rolling_debiased_embedding(
                                y_clean,
                                y_shift_mean,
                                eta_0,
                                y_shift_k0=y_shift_k0_all,
                                eta_delta=eta_delta,
                            )
                    else:
                        y_tilde = y_clean

                    horizon_total_losses = []
                    horizon_siglip_losses = []
                    horizon_null_losses = []
                    horizon_delta_null_losses = []
                    horizon_distill_losses = []
                    skip_empty_check = (
                        use_static_rev_table and bs_len > max_horizon * 2
                    )
                    for delay in horizons:
                        delay_mask = revision_delays == delay
                        d_count_t = delay_mask.sum()
                        if not skip_empty_check and int(d_count_t) == 0:
                            continue
                        rows_r = revision_rows[delay_mask]
                        d_siglip = loss_module.forward_rectangular(
                            y_tilde[delay_mask],
                            batch_targets,
                            rows_r,
                        )
                        if is_causal_tier:
                            if (
                                compute_null_grad
                                and y_shift_mean is not None
                                and lambda_null > 0.0
                                and centered_vocab_bank is not None
                            ):
                                d_null = compute_scaled_shift_null_loss(
                                    y_shift_mean[delay_mask],
                                    centered_vocab_bank,
                                    y_shift_branches=(
                                        y_shift_branches[:, delay_mask, :]
                                        if y_shift_branches is not None
                                        else None
                                    ),
                                )
                            else:
                                d_null = d_siglip * 0.0
                            if (
                                compute_null_grad
                                and y_shift_mean is not None
                                and int(delay) > 0
                                and lambda_delta > 0.0
                                and k0_has_row is not None
                                and k0_shift_mean_by_row is not None
                                and bool(k0_has_row[rows_r].any())
                            ):
                                valid_k0_r = k0_has_row[rows_r]
                                y_shift_r_m = y_shift_mean[delay_mask][valid_k0_r]
                                y_shift_k0_m = k0_shift_mean_by_row[
                                    rows_r[valid_k0_r]
                                ]
                                y_shift_r_br = (
                                    y_shift_branches[:, delay_mask, :][
                                        :, valid_k0_r, :
                                    ]
                                    if y_shift_branches is not None
                                    else None
                                )
                                y_shift_k0_br = (
                                    k0_shift_branches_by_row[
                                        :, rows_r[valid_k0_r], :
                                    ]
                                    if k0_shift_branches_by_row is not None
                                    else None
                                )
                                d_delta_null = compute_euclidean_cross_horizon_loss(
                                    y_shift_r_m,
                                    y_shift_k0_m,
                                    y_shift_r_branches=y_shift_r_br,
                                    y_shift_k0_branches=y_shift_k0_br,
                                )
                                d_delta_weighted = lambda_delta * d_delta_null
                            else:
                                d_delta_null = d_siglip * 0.0
                                d_delta_weighted = d_siglip * 0.0
                            if (
                                debias_mode == "amortized_timing"
                                and b_pred_all is not None
                                and y_shift_mean is not None
                                and lambda_distill > 0.0
                            ):
                                d_distill = compute_temporal_bias_distillation_loss(
                                    b_pred_all[delay_mask],
                                    y_shift_mean[delay_mask],
                                )
                                d_distill_weighted = lambda_distill * d_distill
                            else:
                                d_distill = d_siglip * 0.0
                                d_distill_weighted = d_siglip * 0.0
                        else:
                            d_distill = d_siglip * 0.0
                            d_distill_weighted = d_siglip * 0.0
                            if (
                                compute_null_grad
                                and y_shift_mean is not None
                                and lambda_null > 0.0
                            ):
                                d_null = compute_shift_null_loss(
                                    y_shift_mean[delay_mask], centered_vocab_bank
                                )
                            else:
                                d_null = d_siglip * 0.0
                            if (
                                compute_null_grad
                                and y_shift_mean is not None
                                and int(delay) > 0
                                and (
                                    lambda_delta_null > 0.0 or beta_delta_cm > 0.0
                                )
                                and k0_has_row is not None
                                and k0_shift_mean_by_row is not None
                                and bool(k0_has_row[rows_r].any())
                            ):
                                valid_k0_r = k0_has_row[rows_r]
                                y_shift_r_m = y_shift_mean[delay_mask][valid_k0_r]
                                y_shift_k0_m = k0_shift_mean_by_row[
                                    rows_r[valid_k0_r]
                                ]
                                y_shift_r_br = (
                                    y_shift_branches[:, delay_mask, :][
                                        :, valid_k0_r, :
                                    ]
                                    if y_shift_branches is not None
                                    else None
                                )
                                y_shift_k0_br = (
                                    k0_shift_branches_by_row[
                                        :, rows_r[valid_k0_r], :
                                    ]
                                    if k0_shift_branches_by_row is not None
                                    else None
                                )
                                d_delta_vocab, d_delta_cm = (
                                    compute_delta_shift_null_loss(
                                        y_shift_r_m,
                                        y_shift_k0_m,
                                        centered_vocab_bank,
                                        beta_delta_cm=beta_delta_cm,
                                        y_shift_r_branches=y_shift_r_br,
                                        y_shift_k0_branches=y_shift_k0_br,
                                        return_components=True,
                                    )
                                )
                                d_delta_null = d_delta_cm + d_delta_vocab
                                d_delta_weighted = (
                                    lambda_delta_null * d_delta_vocab
                                    + beta_delta_cm * d_delta_cm
                                )
                            else:
                                d_delta_null = d_siglip * 0.0
                                d_delta_weighted = d_siglip * 0.0

                        d_total = (
                            d_siglip
                            + lambda_null * d_null
                            + d_delta_weighted
                            + d_distill_weighted
                        )
                        horizon_total_losses.append(d_total)
                        horizon_siglip_losses.append(d_siglip.detach())
                        horizon_null_losses.append(d_null.detach())
                        horizon_delta_null_losses.append(d_delta_null.detach())
                        horizon_distill_losses.append(d_distill.detach())

                        delay_counts_gpu[delay] += d_count_t
                        delay_siglip_sums_gpu[delay] += d_siglip.detach().to(
                            dtype=torch.float64
                        )
                        delay_null_sums_gpu[delay] += d_null.detach().to(
                            dtype=torch.float64
                        )
                        delay_delta_null_sums_gpu[delay] += (
                            d_delta_null.detach().to(dtype=torch.float64)
                        )
                        delay_batches_gpu[delay] += 1

                    if not horizon_total_losses:
                        raise RuntimeError("当前 batch 没有可用的滚动回修监督。")
                    loss = torch.stack(horizon_total_losses).mean()
                    batch_siglip_t = torch.stack(horizon_siglip_losses).mean()
                    batch_null_t = torch.stack(horizon_null_losses).mean()
                    batch_delta_null_t = torch.stack(
                        horizon_delta_null_losses
                    ).mean()
                    batch_distill_t = torch.stack(horizon_distill_losses).mean()
                    revision_supervision_count += int(y_tilde.shape[0])
                else:
                    (
                        y_clean,
                        y_shift_mean,
                        y_shift_branches,
                    ) = forward_context_from_preextracted(
                        model,
                        clean_z_batch,
                        batch_sentence_indices,
                        shifted_z_batches=shifted_z_batches,
                        synchronize_dropout_rng=sync_rng,
                        compute_null_grad=compute_null_grad,
                        return_shift_branches=True,
                    )
                    b_ctx_pred: torch.Tensor | None = None
                    if siamese_enabled and y_shift_mean is not None:
                        if debias_mode == "null_space":
                            if eta_train > 0.0:
                                y_tilde = F.normalize(
                                    y_clean - eta_train * y_shift_mean.detach(),
                                    dim=-1,
                                )
                            else:
                                y_tilde = y_clean
                        elif debias_mode == "amortized_timing":
                            assert onset_seconds_all is not None
                            batch_onset_seconds = onset_seconds_all[idx_tensor]
                            b_ctx_pred = (
                                model.temporal_bias_head.predict_group_revisions(
                                    batch_onset_seconds,
                                    batch_sentence_indices,
                                    (0,),
                                )[0]
                            )
                            if eta_train > 0.0:
                                y_tilde = F.normalize(
                                    y_clean
                                    - eta_train
                                    * b_ctx_pred.detach().to(dtype=y_clean.dtype),
                                    dim=-1,
                                )
                            else:
                                y_tilde = y_clean
                        else:
                            # 雅可比模长阻尼重缩放（Norm-Damped Detach）
                            y_tilde = compute_norm_damped_debiased_embedding(
                                y_clean, y_shift_mean, eta_train
                            )
                    else:
                        y_tilde = y_clean
                    d_siglip = loss_module(y_tilde, batch_targets)
                    if (
                        compute_null_grad
                        and y_shift_mean is not None
                        and lambda_null > 0.0
                        and centered_vocab_bank is not None
                    ):
                        d_null = (
                            compute_scaled_shift_null_loss(
                                y_shift_mean,
                                centered_vocab_bank,
                                y_shift_branches=y_shift_branches,
                            )
                            if is_causal_tier
                            else compute_shift_null_loss(
                                y_shift_mean, centered_vocab_bank
                            )
                        )
                    else:
                        d_null = d_siglip * 0.0
                    if (
                        debias_mode == "amortized_timing"
                        and b_ctx_pred is not None
                        and y_shift_mean is not None
                        and lambda_distill > 0.0
                    ):
                        d_distill = compute_temporal_bias_distillation_loss(
                            b_ctx_pred, y_shift_mean
                        )
                    else:
                        d_distill = d_siglip * 0.0
                    loss = (
                        d_siglip
                        + lambda_null * d_null
                        + lambda_distill * d_distill
                    )
                    batch_siglip_t = d_siglip.detach()
                    batch_null_t = d_null.detach()
                    batch_delta_null_t = d_null.detach() * 0.0
                    batch_distill_t = d_distill.detach()

            scaler.scale(loss).backward()
            max_grad_norm = float(training_config.get("max_grad_norm", 0.0))
            if max_grad_norm > 0 and trainable_model_params:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    trainable_model_params,
                    max_grad_norm,
                    foreach=True if device.type == "cuda" else None,
                )
            if scaler.is_enabled():
                previous_scale = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                if float(scaler.get_scale()) >= previous_scale:
                    optimizer_updates += 1
            else:
                optimizer.step()
                optimizer_updates += 1
            batches_seen += 1

            total_loss_gpu += loss.detach().to(dtype=torch.float64) * bs_len
            siglip_loss_gpu += batch_siglip_t.to(dtype=torch.float64) * bs_len
            null_loss_gpu += batch_null_t.to(dtype=torch.float64) * bs_len
            delta_null_loss_gpu += (
                batch_delta_null_t.to(dtype=torch.float64) * bs_len
            )
            distill_loss_gpu += (
                batch_distill_t.to(dtype=torch.float64) * bs_len
            )
            sample_count += bs_len

            if pbar is not None and (
                batches_seen == 1
                or batches_seen % 10 == 0
                or batches_seen == total_batches
            ):
                denom_cur = max(1, sample_count)
                running_tot = float(total_loss_gpu.item()) / denom_cur
                running_sig = float(siglip_loss_gpu.item()) / denom_cur
                running_nul = float(null_loss_gpu.item()) / denom_cur
                postfix = {
                    "loss": f"{running_tot:.4f}",
                    "siglip": f"{running_sig:.4f}",
                    "null": f"{running_nul:.4f}",
                }
                if lambda_delta_null > 0.0:
                    postfix["d_null"] = (
                        f"{float(delta_null_loss_gpu.item()) / denom_cur:.4f}"
                    )
                if debias_mode == "amortized_timing" and lambda_distill > 0.0:
                    postfix["dist"] = (
                        f"{float(distill_loss_gpu.item()) / denom_cur:.4f}"
                    )
                postfix["lr"] = f"{current_lr:.1e}"
                pbar.set_postfix(postfix, refresh=False)
    finally:
        if pbar is not None:
            pbar.close()

    denom = max(1, sample_count)
    total_loss_sum = float(total_loss_gpu.item())
    siglip_loss_sum = float(siglip_loss_gpu.item())
    null_loss_sum = float(null_loss_gpu.item())
    delta_null_loss_sum = float(delta_null_loss_gpu.item())
    distill_loss_sum = float(distill_loss_gpu.item())
    revision_delay_counts = {
        d: int(delay_counts_gpu[d].item()) for d in horizons
    }
    revision_delay_siglip_sums = {
        d: float(delay_siglip_sums_gpu[d].item()) for d in horizons
    }
    revision_delay_null_sums = {
        d: float(delay_null_sums_gpu[d].item()) for d in horizons
    }
    revision_delay_delta_null_sums = {
        d: float(delay_delta_null_sums_gpu[d].item()) for d in horizons
    }
    revision_delay_batches = {
        d: int(delay_batches_gpu[d].item()) for d in horizons
    }
    diagnostics = {
        "mode": "siamese_debiased" if siamese_enabled else "preextracted_frozen_cnn",
        "debias_mode": debias_mode if siamese_enabled else "none",
        "shift_mode": shift_mode,
        "siamese_debiased_enabled": siamese_enabled,
        "eta_0": eta_0 if siamese_enabled else 0.0,
        "eta_train": eta_train if siamese_enabled else 0.0,
        "eta_delta": eta_delta if siamese_enabled else 0.0,
        "lambda_null": lambda_null if siamese_enabled else 0.0,
        "lambda_delta": lambda_delta if siamese_enabled else 0.0,
        "lambda_delta_null": lambda_delta_null if siamese_enabled else 0.0,
        "lambda_distill": lambda_distill if siamese_enabled else 0.0,
        "beta_delta_cm": beta_delta_cm if siamese_enabled else 0.0,
        "shifts_per_step": shifts_per_step if siamese_enabled else 0,
        "word_loss": float(siglip_loss_sum / denom),
        "siglip_loss": float(siglip_loss_sum / denom),
        "shift_null_loss": float(null_loss_sum / denom),
        "weighted_shift_null_loss": float((lambda_null * null_loss_sum) / denom)
        if siamese_enabled
        else 0.0,
        "delta_shift_null_loss": float(delta_null_loss_sum / denom),
        "weighted_delta_shift_null_loss": float(
            (lambda_delta_null * delta_null_loss_sum) / denom
        )
        if siamese_enabled
        else 0.0,
        "distill_loss": float(distill_loss_sum / denom),
        "weighted_distill_loss": float(
            (lambda_distill * distill_loss_sum) / denom
        )
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
            "delay_mean_delta_null_losses": {
                d: (
                    revision_delay_delta_null_sums[d] / revision_delay_batches[d]
                    if revision_delay_batches[d]
                    else None
                )
                for d in revision_delay_delta_null_sums
            },
        },
    }
    return (
        float(total_loss_sum / denom),
        int(optimizer_updates),
        int(batches_seen),
        diagnostics,
    )
