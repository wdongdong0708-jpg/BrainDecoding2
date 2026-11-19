"""基于冻结 CNN 预提取特征的定长填充 Transformer 极速前向与窗口规划器。"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

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
    if not synchronize_dropout_rng:
        return dropout_module(attn)
    if num_branches <= 1:
        shared_mask = F.dropout(
            torch.ones_like(attn),
            p=p,
            training=True,
            inplace=False,
        )
        return attn * shared_mask
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
    if not block.training or not synchronize_dropout_rng:
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


def _repeat_target_group_idx(
    target_group_idx: torch.Tensor | None,
    num_branches: int,
    g_len: int,
) -> torch.Tensor | None:
    """将单分支目标窗口索引 [Q] 扩展为多分支拼接后的目标窗口索引 [num_branches * Q]。"""
    if target_group_idx is None or num_branches <= 1:
        return target_group_idx
    offsets = (
        torch.arange(
            num_branches,
            device=target_group_idx.device,
            dtype=torch.long,
        ).unsqueeze(1)
        * int(g_len)
    )
    return (offsets + target_group_idx.unsqueeze(0)).reshape(-1)


def _run_fast_padded_transformer(
    ctx,
    padded: torch.Tensor,
    mask: torch.Tensor,
    *,
    causal_override: bool | None = None,
    target_col_idx: torch.Tensor | None = None,
    target_group_idx: torch.Tensor | None = None,
    num_branches: int = 1,
    synchronize_dropout_rng: bool = True,
) -> torch.Tensor:
    """在已构建好的 [G, L_max, D] 填充张量与 [G, L_max] 掩码上直接执行 Transformer（零 Python 循环）。

    核心极速优化：
    1. 有效 Token 紧凑打包（Valid-Token Packing）：将非填充位打包为 [N_valid, D] 贯穿全部 16 层，
       使得所有 LayerNorm、融合 QKV 投影、to_out 投影与 4x FeedForward 100% 跳过填充零位；
    2. 融合 QKV / KV 单次大矩阵乘（Fused QKV GEMM）：将每层独立的 to_q/to_k/to_v 合并为单次 GEMM；
    3. 唯一窗口去重与末层单目标剪枝：前 15 层仅在去重后的唯一窗口 [G] 上计算，第 16 层（末层）
       仅对目标词位 [Q] 计算 Query、输出投影与 FeedForward，省去末层 80% 冗余计算并直接返回 [Q, D]。
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
            torch.get_autocast_dtype("cuda")
            if (padded.is_cuda and torch.is_autocast_enabled())
            else padded.dtype
        )
        x_init = padded.to(dtype=work_dtype)
        valid_flat_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        n_valid = int(valid_flat_idx.numel())
        if n_valid == 0:
            out_empty = enc.final_norm(x_init)
            if target_col_idx is not None:
                win_idx = (
                    target_group_idx
                    if target_group_idx is not None
                    else torch.arange(bsz, device=x_init.device, dtype=torch.long)
                )
                return out_empty[win_idx, target_col_idx]
            return out_empty

        pos = torch.arange(max_len, device=x_init.device)
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
        num_layers = len(enc.layers)
        if target_col_idx is not None:
            num_q = int(target_col_idx.shape[0])
            win_idx = (
                target_group_idx
                if target_group_idx is not None
                else torch.arange(bsz, device=x_init.device, dtype=torch.long)
            )
            target_flat_idx = win_idx * max_len + target_col_idx
            flat_to_valid = torch.zeros(
                bsz * max_len, device=x_init.device, dtype=torch.long
            )
            flat_to_valid[valid_flat_idx] = torch.arange(
                n_valid, device=x_init.device, dtype=torch.long
            )
            target_valid_idx = flat_to_valid[target_flat_idx]
        else:
            num_q = 0
            target_valid_idx = None

        xv = x_init.reshape(bsz * max_len, dim).index_select(0, valid_flat_idx)
        x_last = None

        for ind, (layer_type, layer_tuple) in enumerate(
            zip(enc.layer_types, enc.layers)
        ):
            norms, block, residual_fn = layer_tuple
            pre_norm, post_branch_norm, post_main_norm = norms
            is_last_attn = target_col_idx is not None and ind == num_layers - 2
            is_last_ff = target_col_idx is not None and ind == num_layers - 1

            if is_last_attn:
                residual = xv.index_select(0, target_valid_idx).unsqueeze(1)
                xv_norm = (
                    pre_norm(xv).to(dtype=work_dtype)
                    if pre_norm is not None
                    else xv
                )
                q_in = xv_norm.index_select(0, target_valid_idx).unsqueeze(1)
                heads = int(block.heads)
                q = block.to_q(q_in).view(num_q, 1, heads, -1).transpose(1, 2)
                kv_w = torch.cat((block.to_k.weight, block.to_v.weight), dim=0)
                kv_valid = F.linear(xv_norm, kv_w)
                kv_pad = kv_valid.new_zeros((bsz * max_len, kv_valid.shape[-1]))
                kv_pad.index_copy_(0, valid_flat_idx, kv_valid)
                k, v = kv_pad.view(bsz, max_len, 2, heads, -1).unbind(dim=2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                if cos_full is not None:
                    q = _apply_rotary_pos_emb_fast(q, cos_t, sin_t)
                    k = _apply_rotary_pos_emb_fast(k, cos_full, sin_full)
                if target_group_idx is not None:
                    k = k.index_select(0, target_group_idx)
                    v = v.index_select(0, target_group_idx)
                    last_key_mask = key_mask.index_select(0, target_group_idx)
                else:
                    last_key_mask = key_mask
                scale = q.shape[-1] ** -0.5
                sim = torch.matmul(q, k.transpose(-1, -2)) * scale
                sim = sim.masked_fill(
                    ~last_key_mask, -torch.finfo(sim.dtype).max
                )
                attn = F.softmax(sim, dim=-1).to(dtype=work_dtype)
                attn = _apply_siamese_shared_dropout(
                    attn,
                    block.attend.attn_dropout,
                    num_branches=num_branches,
                    synchronize_dropout_rng=synchronize_dropout_rng,
                )
                out = torch.matmul(attn, v).transpose(1, 2).reshape(num_q, 1, -1)
                out = block.to_out(out)
                if post_branch_norm is not None:
                    out = post_branch_norm(out)
                x_last = residual_fn(out, residual).to(dtype=work_dtype)
                if post_main_norm is not None:
                    x_last = post_main_norm(x_last).to(dtype=work_dtype)
            elif is_last_ff:
                residual = x_last
                x_norm = (
                    pre_norm(x_last).to(dtype=work_dtype)
                    if pre_norm is not None
                    else x_last
                )
                out = _run_ff_with_shared_dropout(
                    block,
                    x_norm,
                    num_branches=num_branches,
                    synchronize_dropout_rng=synchronize_dropout_rng,
                )
                if post_branch_norm is not None:
                    out = post_branch_norm(out)
                x_last = residual_fn(out, residual).to(dtype=work_dtype)
                if post_main_norm is not None:
                    x_last = post_main_norm(x_last).to(dtype=work_dtype)
            elif layer_type == "a":
                xv_norm = (
                    pre_norm(xv).to(dtype=work_dtype)
                    if pre_norm is not None
                    else xv
                )
                heads = int(block.heads)
                qkv_w = torch.cat(
                    (block.to_q.weight, block.to_k.weight, block.to_v.weight),
                    dim=0,
                )
                qkv_valid = F.linear(xv_norm, qkv_w)
                qkv_pad = qkv_valid.new_zeros(
                    (bsz * max_len, qkv_valid.shape[-1])
                )
                qkv_pad.index_copy_(0, valid_flat_idx, qkv_valid)
                q, k, v = qkv_pad.view(bsz, max_len, 3, heads, -1).unbind(dim=2)
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
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
                out_pad = (
                    torch.matmul(attn, v)
                    .transpose(1, 2)
                    .reshape(bsz * max_len, -1)
                )
                out_v = block.to_out(out_pad.index_select(0, valid_flat_idx))
                if post_branch_norm is not None:
                    out_v = post_branch_norm(out_v)
                xv = residual_fn(out_v, xv).to(dtype=work_dtype)
                if post_main_norm is not None:
                    xv = post_main_norm(xv).to(dtype=work_dtype)
            else:
                xv_norm = (
                    pre_norm(xv).to(dtype=work_dtype)
                    if pre_norm is not None
                    else xv
                )
                out_v = _run_ff_with_shared_dropout(
                    block,
                    xv_norm,
                    num_branches=num_branches,
                    synchronize_dropout_rng=synchronize_dropout_rng,
                )
                if post_branch_norm is not None:
                    out_v = post_branch_norm(out_v)
                xv = residual_fn(out_v, xv).to(dtype=work_dtype)
                if post_main_norm is not None:
                    xv = post_main_norm(xv).to(dtype=work_dtype)

        if target_col_idx is not None:
            return enc.final_norm(x_last).squeeze(1)
        xv_final = enc.final_norm(xv)
        out_full = xv_final.new_zeros((bsz * max_len, dim))
        out_full.index_copy_(0, valid_flat_idx, xv_final)
        return out_full.view(bsz, max_len, dim)

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
        win_idx = (
            target_group_idx
            if target_group_idx is not None
            else torch.arange(
                encoded.shape[0], device=encoded.device, dtype=torch.long
            )
        )
        return encoded[win_idx, target_col_idx]
    return encoded


def build_padded_revision_window_plan(
    model,
    group_indices: torch.Tensor,
    *,
    cutoff_generator=None,
    cutoff_policy: str = "all",
) -> dict[str, torch.Tensor]:
    """在 CPU NumPy 中一次性构建去重后的 2D 局部窗口填充索引矩阵 [G_unique, L_max] 与掩码。"""
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

            prev_bounds: tuple[int, int] | None = None
            for delay in model.prefix_revision_horizons:
                target_position = cutoff - delay
                if target_position < 0:
                    continue
                start, stop, target_offset = model._revision_window_bounds(
                    target_position, delay, len(indices)
                )
                if stop != cutoff + 1:
                    raise RuntimeError("局部窗口滚动回修训练边界不一致。")
                if (start, stop) != prev_bounds:
                    win_indices = indices[start:stop]
                    windows.append(win_indices)
                    virtual_group += 1
                    prev_bounds = (start, stop)
                target_group_indices.append(virtual_group - 1)
                target_col_indices.append(target_offset)
                revision_rows.append(indices[target_position])
                revision_delays.append(delay)

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
