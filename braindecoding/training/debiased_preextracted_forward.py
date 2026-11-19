"""冻结 CNN 预提取特征下的 Revision / Prefix-Revision 与 Full-Context 前向分发器。"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from braindecoding.training.debiased_padded_forward import (
    _can_use_direct_layer_loop,
    _can_use_fast_padded_transformer,
    _repeat_target_group_idx,
    _run_fast_padded_transformer,
    build_padded_context_group_plan,
    build_padded_revision_window_plan,
    build_revision_window_plan,
)
from braindecoding.training.word_encoder_training import (
    local_context_internal_mask,
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
    compute_null_grad: bool = True,
    return_shift_branches: bool = False,
):
    """在预提取 CNN 特征上运行共享上下文 Transformer。

    默认返回 (y_clean, y_shift_mean, revision_rows, revision_delays)；
    当 return_shift_branches=True 时返回 (y_clean, y_shift_mean, y_shift_branches, revision_rows, revision_delays)。
    """
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
        q_len = int(target_col_idx.shape[0])
        base_target_group = None if q_len == g_len else target_group_idx

        if not shifted_padded_list:
            clean_out = _run_fast_padded_transformer(
                model.context_transformer,
                clean_padded,
                mask_2d,
                causal_override=False,
                target_col_idx=target_col_idx,
                target_group_idx=base_target_group,
            )
            y_clean = F.normalize(clean_out, dim=-1)
            if return_shift_branches:
                return y_clean, None, None, revision_rows, revision_delays
            return y_clean, None, revision_rows, revision_delays

        if _can_use_direct_layer_loop(
            model.context_transformer, use_causal_attention=False
        ):
            num_shifts = len(shifted_padded_list)
            # 当启用梯度且 M > 1 时，机制 1 (y_tilde) 对 y_shift_mean 施加了 .detach()，
            # 只有机制 2 (L_null / L_delta_null) 对错位分支求导。因此仅需将 1 个随机错位分支与 Clean
            # 合并做带梯度前反向传播（num_branches=2），其余 M-1 个错位分支在 torch.no_grad()
            # 下共享相同 Dropout RNG 前向计算，通过直通均值估计器 (Straight-Through Mean)
            # 保证前向 100% 精确等于 M 分支均值、反向提供 L_null 与 L_delta_null 的无偏梯度估计，消去 5/6 错位反向耗时。
            if torch.is_grad_enabled() and (
                not compute_null_grad or num_shifts > 1
            ):
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
                fork_devices = [] if cuda_device is None else [cuda_device]

                if not compute_null_grad:
                    clean_out = _run_fast_padded_transformer(
                        model.context_transformer,
                        clean_padded,
                        mask_2d,
                        causal_override=False,
                        target_col_idx=target_col_idx,
                        target_group_idx=base_target_group,
                        num_branches=1,
                        synchronize_dropout_rng=synchronize_dropout_rng,
                    )
                    y_clean = F.normalize(clean_out, dim=-1)
                    with torch.no_grad():
                        if synchronize_dropout_rng and cpu_rng_state is not None:
                            with torch.random.fork_rng(devices=fork_devices):
                                torch.set_rng_state(cpu_rng_state)
                                if cuda_rng_state is not None:
                                    torch.cuda.set_rng_state(
                                        cuda_rng_state, cuda_device
                                    )
                                shift_out = _run_fast_padded_transformer(
                                    model.context_transformer,
                                    torch.cat(shifted_padded_list, dim=0),
                                    mask_2d.repeat(num_shifts, 1),
                                    causal_override=False,
                                    target_col_idx=target_col_idx.repeat(
                                        num_shifts
                                    ),
                                    target_group_idx=_repeat_target_group_idx(
                                        base_target_group, num_shifts, g_len
                                    ),
                                    num_branches=num_shifts,
                                    synchronize_dropout_rng=synchronize_dropout_rng,
                                )
                        else:
                            shift_out = _run_fast_padded_transformer(
                                model.context_transformer,
                                torch.cat(shifted_padded_list, dim=0),
                                mask_2d.repeat(num_shifts, 1),
                                causal_override=False,
                                target_col_idx=target_col_idx.repeat(num_shifts),
                                target_group_idx=_repeat_target_group_idx(
                                    base_target_group, num_shifts, g_len
                                ),
                                num_branches=num_shifts,
                                synchronize_dropout_rng=False,
                            )
                        y_shift_branches = F.normalize(shift_out, dim=-1).view(
                            num_shifts, q_len, -1
                        )
                        y_shift_mean = y_shift_branches.mean(dim=0)
                    if return_shift_branches:
                        return (
                            y_clean,
                            y_shift_mean,
                            y_shift_branches,
                            revision_rows,
                            revision_delays,
                        )
                    return y_clean, y_shift_mean, revision_rows, revision_delays

                grad_padded = torch.cat(
                    [clean_padded, shifted_padded_list[0]], dim=0
                )
                grad_out = _run_fast_padded_transformer(
                    model.context_transformer,
                    grad_padded,
                    mask_2d.repeat(2, 1),
                    causal_override=False,
                    target_col_idx=target_col_idx.repeat(2),
                    target_group_idx=_repeat_target_group_idx(
                        base_target_group, 2, g_len
                    ),
                    num_branches=2,
                    synchronize_dropout_rng=synchronize_dropout_rng,
                )
                del grad_padded
                grad_norm = F.normalize(grad_out, dim=-1).view(2, q_len, -1)
                del grad_out
                y_clean = grad_norm[0]
                y_shift_0 = grad_norm[1]

                num_rest = num_shifts - 1
                rest_branches = shifted_padded_list[1:]
                max_seq_len = int(clean_padded.shape[1])
                chunk_b = (
                    2
                    if (num_rest > 2 and num_rest * g_len * max_seq_len > 16384)
                    else num_rest
                )
                with torch.no_grad():
                    rest_norm_chunks: list[torch.Tensor] = []
                    for c_start in range(0, num_rest, chunk_b):
                        c_slice = rest_branches[c_start : c_start + chunk_b]
                        c_num = len(c_slice)
                        c_padded = (
                            c_slice[0]
                            if c_num == 1
                            else torch.cat(c_slice, dim=0)
                        )
                        c_mask = (
                            mask_2d if c_num == 1 else mask_2d.repeat(c_num, 1)
                        )
                        c_tcol = (
                            target_col_idx
                            if c_num == 1
                            else target_col_idx.repeat(c_num)
                        )
                        c_tgrp = _repeat_target_group_idx(
                            base_target_group, c_num, g_len
                        )
                        if synchronize_dropout_rng and cpu_rng_state is not None:
                            with torch.random.fork_rng(devices=fork_devices):
                                torch.set_rng_state(cpu_rng_state)
                                if cuda_rng_state is not None:
                                    torch.cuda.set_rng_state(
                                        cuda_rng_state, cuda_device
                                    )
                                c_out = _run_fast_padded_transformer(
                                    model.context_transformer,
                                    c_padded,
                                    c_mask,
                                    causal_override=False,
                                    target_col_idx=c_tcol,
                                    target_group_idx=c_tgrp,
                                    num_branches=c_num,
                                    synchronize_dropout_rng=synchronize_dropout_rng,
                                )
                        else:
                            c_out = _run_fast_padded_transformer(
                                model.context_transformer,
                                c_padded,
                                c_mask,
                                causal_override=False,
                                target_col_idx=c_tcol,
                                target_group_idx=c_tgrp,
                                num_branches=c_num,
                                synchronize_dropout_rng=False,
                            )
                        del c_padded
                        rest_norm_chunks.append(
                            F.normalize(c_out, dim=-1).view(c_num, q_len, -1)
                        )
                        del c_out
                    rest_norm = (
                        rest_norm_chunks[0]
                        if len(rest_norm_chunks) == 1
                        else torch.cat(rest_norm_chunks, dim=0)
                    )
                    del rest_norm_chunks
                    y_shift_mean_detached = (
                        y_shift_0.detach() + rest_norm.sum(dim=0)
                    ) / float(num_shifts)

                y_shift_mean = y_shift_mean_detached + (
                    y_shift_0 - y_shift_0.detach()
                )
                if return_shift_branches:
                    y_shift_branches = torch.cat(
                        [y_shift_0.unsqueeze(0), rest_norm], dim=0
                    )
                    return (
                        y_clean,
                        y_shift_mean,
                        y_shift_branches,
                        revision_rows,
                        revision_delays,
                    )
                return y_clean, y_shift_mean, revision_rows, revision_delays

            num_branches = 1 + num_shifts
            all_padded = torch.cat([clean_padded, *shifted_padded_list], dim=0)
            all_mask = mask_2d.repeat(num_branches, 1)
            all_target_col = target_col_idx.repeat(num_branches)
            all_target_group = _repeat_target_group_idx(
                base_target_group, num_branches, g_len
            )
            all_out = _run_fast_padded_transformer(
                model.context_transformer,
                all_padded,
                all_mask,
                causal_override=False,
                target_col_idx=all_target_col,
                target_group_idx=all_target_group,
                num_branches=num_branches,
                synchronize_dropout_rng=synchronize_dropout_rng,
            )
            all_norm = F.normalize(all_out, dim=-1).view(
                num_branches, q_len, -1
            )
            y_clean = all_norm[0]
            y_shift_branches = all_norm[1:]
            y_shift_mean = y_shift_branches.mean(dim=0)
            if return_shift_branches:
                return (
                    y_clean,
                    y_shift_mean,
                    y_shift_branches,
                    revision_rows,
                    revision_delays,
                )
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

        y_shift_branches = torch.stack(y_shift_list, dim=0)
        y_shift_mean = y_shift_branches.mean(dim=0)
        if return_shift_branches:
            return (
                y_clean,
                y_shift_mean,
                y_shift_branches,
                revision_rows,
                revision_delays,
            )
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
        if return_shift_branches:
            return y_clean, None, None, revision_rows, revision_delays
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

    y_shift_branches = torch.stack(y_shift_list, dim=0)
    y_shift_mean = y_shift_branches.mean(dim=0)
    if return_shift_branches:
        return (
            y_clean,
            y_shift_mean,
            y_shift_branches,
            revision_rows,
            revision_delays,
        )
    return y_clean, y_shift_mean, revision_rows, revision_delays


def forward_context_from_preextracted(
    model,
    clean_z_batch: torch.Tensor,
    group_indices: torch.Tensor,
    *,
    shifted_z_batches: list[torch.Tensor] | None = None,
    synchronize_dropout_rng: bool = True,
    padded_plan: dict[str, torch.Tensor] | None = None,
    compute_null_grad: bool = True,
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
            num_shifts = len(shifted_padded_list)
            g_len = int(clean_padded.shape[0])
            max_len = int(clean_padded.shape[1])

            if torch.is_grad_enabled() and (
                not compute_null_grad or num_shifts > 1
            ):
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
                fork_devices = [] if cuda_device is None else [cuda_device]

                if not compute_null_grad:
                    clean_encoded = _run_fast_padded_transformer(
                        model.context_transformer,
                        clean_padded,
                        mask_2d,
                        num_branches=1,
                        synchronize_dropout_rng=synchronize_dropout_rng,
                    )
                    y_clean = F.normalize(
                        clean_encoded[row_to_group, row_to_col], dim=-1
                    )
                    with torch.no_grad():
                        if synchronize_dropout_rng and cpu_rng_state is not None:
                            with torch.random.fork_rng(devices=fork_devices):
                                torch.set_rng_state(cpu_rng_state)
                                if cuda_rng_state is not None:
                                    torch.cuda.set_rng_state(
                                        cuda_rng_state, cuda_device
                                    )
                                shift_encoded = _run_fast_padded_transformer(
                                    model.context_transformer,
                                    torch.cat(shifted_padded_list, dim=0),
                                    mask_2d.repeat(num_shifts, 1),
                                    num_branches=num_shifts,
                                    synchronize_dropout_rng=synchronize_dropout_rng,
                                ).view(num_shifts, g_len, max_len, -1)
                        else:
                            shift_encoded = _run_fast_padded_transformer(
                                model.context_transformer,
                                torch.cat(shifted_padded_list, dim=0),
                                mask_2d.repeat(num_shifts, 1),
                                num_branches=num_shifts,
                                synchronize_dropout_rng=False,
                            ).view(num_shifts, g_len, max_len, -1)
                        y_shift_mean = F.normalize(
                            shift_encoded[:, row_to_group, row_to_col], dim=-1
                        ).mean(dim=0)
                    return y_clean, y_shift_mean

                grad_encoded = _run_fast_padded_transformer(
                    model.context_transformer,
                    torch.cat([clean_padded, shifted_padded_list[0]], dim=0),
                    mask_2d.repeat(2, 1),
                    num_branches=2,
                    synchronize_dropout_rng=synchronize_dropout_rng,
                ).view(2, g_len, max_len, -1)
                grad_norm = F.normalize(
                    grad_encoded[:, row_to_group, row_to_col], dim=-1
                )
                del grad_encoded
                y_clean = grad_norm[0]
                y_shift_0 = grad_norm[1]

                num_rest = num_shifts - 1
                rest_branches = shifted_padded_list[1:]
                chunk_b = (
                    2
                    if (num_rest > 2 and num_rest * g_len * max_len > 16384)
                    else num_rest
                )
                with torch.no_grad():
                    rest_norm_chunks: list[torch.Tensor] = []
                    for c_start in range(0, num_rest, chunk_b):
                        c_slice = rest_branches[c_start : c_start + chunk_b]
                        c_num = len(c_slice)
                        c_padded = (
                            c_slice[0]
                            if c_num == 1
                            else torch.cat(c_slice, dim=0)
                        )
                        c_mask = (
                            mask_2d if c_num == 1 else mask_2d.repeat(c_num, 1)
                        )
                        if synchronize_dropout_rng and cpu_rng_state is not None:
                            with torch.random.fork_rng(devices=fork_devices):
                                torch.set_rng_state(cpu_rng_state)
                                if cuda_rng_state is not None:
                                    torch.cuda.set_rng_state(
                                        cuda_rng_state, cuda_device
                                    )
                                c_encoded = _run_fast_padded_transformer(
                                    model.context_transformer,
                                    c_padded,
                                    c_mask,
                                    num_branches=c_num,
                                    synchronize_dropout_rng=synchronize_dropout_rng,
                                ).view(c_num, g_len, max_len, -1)
                        else:
                            c_encoded = _run_fast_padded_transformer(
                                model.context_transformer,
                                c_padded,
                                c_mask,
                                num_branches=c_num,
                                synchronize_dropout_rng=False,
                            ).view(c_num, g_len, max_len, -1)
                        del c_padded
                        rest_norm_chunks.append(
                            F.normalize(
                                c_encoded[:, row_to_group, row_to_col], dim=-1
                            )
                        )
                        del c_encoded
                    rest_norm = (
                        rest_norm_chunks[0]
                        if len(rest_norm_chunks) == 1
                        else torch.cat(rest_norm_chunks, dim=0)
                    )
                    del rest_norm_chunks
                    y_shift_mean_detached = (
                        y_shift_0.detach() + rest_norm.sum(dim=0)
                    ) / float(num_shifts)

                y_shift_mean = y_shift_mean_detached + (
                    y_shift_0 - y_shift_0.detach()
                )
                return y_clean, y_shift_mean

            num_branches = 1 + num_shifts
            all_padded = torch.cat([clean_padded, *shifted_padded_list], dim=0)
            all_mask = mask_2d.repeat(num_branches, 1)
            all_encoded = _run_fast_padded_transformer(
                model.context_transformer,
                all_padded,
                all_mask,
                num_branches=num_branches,
                synchronize_dropout_rng=synchronize_dropout_rng,
            ).view(num_branches, g_len, max_len, -1)
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
