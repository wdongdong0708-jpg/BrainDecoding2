"""局部相对位置偏置、物理时间间隔偏置与分组上下文/局部窗口修订 Transformer。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class LocalRelativePositionBias(nn.Module):
    """只在给定半径内向注意力 logits 加入可学习相对位置偏置。"""

    MODES = {"undirected", "directed"}

    def __init__(self, max_distance=4, mode="directed"):
        super().__init__()
        self.max_distance = int(max_distance)
        self.mode = str(mode)
        if self.max_distance <= 0:
            raise ValueError("局部相对位置偏置半径必须为正数。")
        if self.mode not in self.MODES:
            raise ValueError(
                "局部相对位置偏置模式必须是 undirected 或 directed。"
            )
        self.past_bias = nn.Parameter(torch.zeros(self.max_distance))
        self.future_bias = (
            nn.Parameter(torch.zeros(self.max_distance))
            if self.mode == "directed"
            else None
        )

    def forward(
        self,
        length,
        device=None,
        dtype=None,
        heads=None,
        active_heads=None,
    ):
        length = int(length)
        if length <= 0:
            raise ValueError("注意力序列长度必须为正数。")
        device = self.past_bias.device if device is None else device
        dtype = self.past_bias.dtype if dtype is None else dtype
        positions = torch.arange(length, device=device)
        relative = positions[None, :] - positions[:, None]
        bias = torch.zeros((length, length), device=device, dtype=dtype)
        for distance in range(1, self.max_distance + 1):
            past = self.past_bias[distance - 1].to(dtype=dtype)
            future = (
                self.future_bias[distance - 1].to(dtype=dtype)
                if self.future_bias is not None
                else past
            )
            bias = bias + past * (relative == -distance)
            bias = bias + future * (relative == distance)
        if active_heads is None:
            return bias[None, None, :, :]
        heads = int(heads)
        active_heads = int(active_heads)
        if not 0 < active_heads < heads:
            raise ValueError("局部偏置头数必须大于 0 且小于总头数。")
        active = (
            torch.arange(heads, device=device) < active_heads
        ).to(dtype=dtype)
        return bias[None, None, :, :] * active[None, :, None, None]


class PhysicalTimeGapBias(nn.Module):
    """按真实事件时间差向注意力 logits 加入有方向的分桶偏置。"""

    MODES = {"directed"}

    def __init__(self, bin_edges_ms=(200, 400, 600, 1000), mode="directed"):
        super().__init__()
        edges = tuple(float(value) for value in bin_edges_ms)
        if not edges or any(value <= 0 for value in edges):
            raise ValueError("物理时间偏置边界必须是非空的正数序列。")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("物理时间偏置边界必须严格递增。")
        self.mode = str(mode)
        if self.mode not in self.MODES:
            raise ValueError("物理时间偏置首版只支持 directed 模式。")
        self.register_buffer(
            "bin_edges_seconds",
            torch.tensor(edges, dtype=torch.float64) / 1000.0,
        )
        self.past_bias = nn.Parameter(torch.zeros(len(edges) + 1))
        self.future_bias = nn.Parameter(torch.zeros(len(edges) + 1))

    @property
    def bin_edges_ms(self):
        return tuple(
            float(value)
            for value in (self.bin_edges_seconds.detach().cpu() * 1000.0)
        )

    def forward(self, event_times, valid_mask=None, dtype=None):
        if event_times is None:
            raise ValueError("物理时间偏置要求每个事件的 onset_seconds。")
        if event_times.ndim != 2:
            raise ValueError("event_times 必须是 [batch, length] 二维张量。")
        if valid_mask is not None and valid_mask.shape != event_times.shape:
            raise ValueError("event_times 与有效位置 mask 的形状必须一致。")
        dtype = self.past_bias.dtype if dtype is None else dtype
        times = event_times.to(dtype=torch.float64)
        delta = times[:, None, :] - times[:, :, None]
        bucket = torch.bucketize(
            delta.abs().contiguous(),
            self.bin_edges_seconds.to(device=times.device),
            right=True,
        )
        bias = torch.zeros(delta.shape, device=times.device, dtype=dtype)
        for bin_index in range(len(self.past_bias)):
            in_bin = bucket == bin_index
            bias = bias + self.past_bias[bin_index].to(dtype=dtype) * (
                in_bin & (delta < 0)
            )
            bias = bias + self.future_bias[bin_index].to(dtype=dtype) * (
                in_bin & (delta > 0)
            )
        if valid_mask is not None:
            valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
            bias = bias * valid_pairs.to(dtype=dtype)
        return bias[:, None, :, :]


class EventTimeInteractionBias(nn.Module):
    """只在指定事件距离上学习事件×真实时间桶的零初始化修正。"""

    def __init__(
        self,
        max_event_distance=2,
        bin_edges_ms=(200, 400, 1000),
    ):
        super().__init__()
        self.max_event_distance = int(max_event_distance)
        if self.max_event_distance <= 0:
            raise ValueError("事件×时间联合偏置距离必须为正数。")
        edges = tuple(float(value) for value in bin_edges_ms)
        if not edges or any(value <= 0 for value in edges):
            raise ValueError("事件×时间联合偏置边界必须是非空的正数序列。")
        if any(right <= left for left, right in zip(edges, edges[1:])):
            raise ValueError("事件×时间联合偏置边界必须严格递增。")
        self.register_buffer(
            "bin_edges_seconds",
            torch.tensor(edges, dtype=torch.float64) / 1000.0,
            persistent=False,
        )
        # 第一维依次表示 -1..-D、+1..+D；单个参数矩阵便于严格审计。
        self.correction = nn.Parameter(
            torch.zeros(2 * self.max_event_distance, len(edges) + 1)
        )

    @property
    def bin_edges_ms(self):
        return tuple(
            float(value)
            for value in (self.bin_edges_seconds.detach().cpu() * 1000.0)
        )

    def forward(self, event_times, valid_mask=None, dtype=None):
        if event_times is None:
            raise ValueError("事件×时间联合偏置要求每个事件的 onset_seconds。")
        if event_times.ndim != 2:
            raise ValueError("event_times 必须是 [batch, length] 二维张量。")
        if valid_mask is not None and valid_mask.shape != event_times.shape:
            raise ValueError("event_times 与有效位置 mask 的形状必须一致。")
        dtype = self.correction.dtype if dtype is None else dtype
        times = event_times.to(dtype=torch.float64)
        delta = times[:, None, :] - times[:, :, None]
        buckets = torch.bucketize(
            delta.abs().contiguous(),
            self.bin_edges_seconds.to(device=times.device),
            right=True,
        )
        length = int(event_times.shape[1])
        positions = torch.arange(length, device=times.device)
        relative = positions[None, :] - positions[:, None]
        bias = torch.zeros(delta.shape, device=times.device, dtype=dtype)
        for distance in range(1, self.max_event_distance + 1):
            past_row = distance - 1
            future_row = self.max_event_distance + distance - 1
            past_pairs = relative == -distance
            future_pairs = relative == distance
            for bin_index in range(self.correction.shape[1]):
                in_bin = buckets == bin_index
                bias = bias + self.correction[past_row, bin_index].to(
                    dtype=dtype
                ) * (in_bin & past_pairs[None])
                bias = bias + self.correction[future_row, bin_index].to(
                    dtype=dtype
                ) * (in_bin & future_pairs[None])
        if valid_mask is not None:
            valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
            bias = bias * valid_pairs.to(dtype=dtype)
        return bias[:, None, :, :]


class AdaptiveDirectionalBiasGate(nn.Module):
    """以当前目标位置表示自适应缩放过去与未来的有向偏置。"""

    def __init__(self, dimension):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, int(dimension)))
        self.bias = nn.Parameter(torch.zeros(2))

    def forward(self, features):
        return 1.0 + torch.tanh(F.linear(features, self.weight, self.bias))


def _apply_soft_bounded_residual(
    base_features: torch.Tensor,
    raw_delta: torch.Tensor,
    max_norm: float,
) -> torch.Tensor:
    """将跨词更新的欧氏范数软限制在 [0, max_norm] 内并叠加回单窗特征归一化。"""
    alpha = float(max_norm)
    base_f = F.normalize(base_features.float(), dim=-1)
    delta_f = raw_delta.float()
    d_norm = delta_f.norm(p=2, dim=-1, keepdim=True)
    bounded = (alpha * torch.tanh(d_norm / max(alpha, 1e-6)) / (d_norm + 1e-6)) * delta_f
    out_f = F.normalize(base_f + bounded, dim=-1)
    return out_f.to(dtype=base_features.dtype)


def _local_attention_window_pre_hook(module, args, kwargs):
    """在不包装或替换原生注意力模块的情况下加入逐层局部窗口。"""
    radius = int(module.local_attention_radius)
    features = args[0]
    length = int(features.shape[-2])
    positions = torch.arange(length, device=features.device)
    relative = positions[None, :] - positions[:, None]
    local_mask = relative.abs() <= radius
    attention_mask = kwargs.get("attn_mask")
    if attention_mask is not None:
        local_mask = attention_mask & local_mask
    kwargs["attn_mask"] = local_mask
    return args, kwargs


class GroupedTransformerEncoder(nn.Module):
    """在每个序列组内独立应用上下文 Transformer。"""

    def __init__(self, dimension, config):
        super().__init__()
        try:
            from x_transformers import Encoder
        except ImportError as exc:
            raise ImportError(
                "The context transformer requires the x-transformers package."
            ) from exc
        heads = int(config.get("heads", 16))
        self.causal_attention = bool(config.get("causal_attention", False))
        window_radius = config.get("query_window_radius")
        left_radius = config.get("query_window_left_radius")
        right_radius = config.get("query_window_right_radius")
        if window_radius is not None and (
            left_radius is not None or right_radius is not None
        ):
            raise ValueError(
                "query_window_radius 不能与左右独立查询窗口同时设置。"
            )
        if (left_radius is None) != (right_radius is None):
            raise ValueError("左右独立查询窗口必须同时设置 left 与 right radius。")
        self.query_window_radius = (
            None if window_radius is None else int(window_radius)
        )
        if self.query_window_radius is not None and self.query_window_radius < 0:
            raise ValueError("query_window_radius 必须是非负整数。")
        if self.query_window_radius is not None:
            left_radius = right_radius = self.query_window_radius
        self.query_window_left_radius = (
            None if left_radius is None else int(left_radius)
        )
        self.query_window_right_radius = (
            None if right_radius is None else int(right_radius)
        )
        if self.query_window_left_radius is not None and (
            self.query_window_left_radius < 0
            or self.query_window_right_radius < 0
        ):
            raise ValueError("查询窗口的 left/right radius 必须是非负整数。")
        self.shared_passes = int(config.get("shared_passes", 1))
        if self.shared_passes <= 0:
            raise ValueError("共享 Transformer 执行次数必须为正数。")
        chunk_config = config.get("chunk_summary_tokens", {})
        self.chunk_summary_enabled = bool(chunk_config.get("enabled", False))
        self.chunk_summary_size = int(chunk_config.get("chunk_size", 4))
        self.chunk_summary_minimum_size = int(
            chunk_config.get("minimum_chunk_size", 3)
        )
        if self.chunk_summary_size <= 0:
            raise ValueError("组块摘要的 chunk_size 必须为正数。")
        if not 1 <= self.chunk_summary_minimum_size <= self.chunk_summary_size:
            raise ValueError(
                "组块摘要的 minimum_chunk_size 必须在 1 和 chunk_size 之间。"
            )
        raw_residual_max_norm = config.get("soft_bounded_residual_max_norm")
        self.soft_bounded_residual_max_norm = (
            None
            if raw_residual_max_norm is None
            else float(raw_residual_max_norm)
        )
        if (
            self.soft_bounded_residual_max_norm is not None
            and self.soft_bounded_residual_max_norm <= 0.0
        ):
            raise ValueError("soft_bounded_residual_max_norm 必须为正数。")
        self.encoder = Encoder(
            dim=int(dimension),
            depth=int(config.get("depth", 16)),
            heads=heads,
            attn_dropout=float(config.get("attention_dropout", 0.1)),
            ff_dropout=float(config.get("feedforward_dropout", 0.0)),
            use_scalenorm=bool(config.get("use_scalenorm", True)),
            use_rmsnorm=bool(config.get("use_rmsnorm", False)),
            rotary_pos_emb=bool(config.get("rotary_position_embedding", True)),
            residual_attn=bool(config.get("residual_attention", False)),
            scale_residual=bool(config.get("scale_residual", True)),
        )
        bias_config = config.get("local_relative_position_bias", {})
        self.local_relative_position_bias = (
            LocalRelativePositionBias(
                max_distance=int(bias_config.get("max_distance", 4)),
                mode=str(bias_config.get("mode", "directed")),
            )
            if bool(bias_config.get("enabled", False))
            else None
        )
        time_bias_config = config.get("physical_time_gap_bias", {})
        self.physical_time_gap_bias = (
            PhysicalTimeGapBias(
                bin_edges_ms=time_bias_config.get(
                    "bin_edges_ms", (200, 400, 600, 1000)
                ),
                mode=str(time_bias_config.get("mode", "directed")),
            )
            if bool(time_bias_config.get("enabled", False))
            else None
        )
        interaction_config = config.get("event_time_interaction_bias", {})
        self.event_time_interaction_bias = (
            EventTimeInteractionBias(
                max_event_distance=int(
                    interaction_config.get("max_event_distance", 2)
                ),
                bin_edges_ms=interaction_config.get(
                    "bin_edges_ms", (200, 400, 1000)
                ),
            )
            if bool(interaction_config.get("enabled", False))
            else None
        )
        if (
            self.physical_time_gap_bias is not None
            and self.event_time_interaction_bias is not None
        ):
            raise ValueError(
                "独立物理时间偏置与事件×时间联合偏置不能同时启用。"
            )
        if (
            self.event_time_interaction_bias is not None
            and (
                self.local_relative_position_bias is None
                or self.local_relative_position_bias.mode != "directed"
            )
        ):
            raise ValueError(
                "事件×时间联合偏置要求保留 directed 事件相对位置偏置。"
            )
        if (
            self.event_time_interaction_bias is not None
            and self.event_time_interaction_bias.max_event_distance
            > self.local_relative_position_bias.max_distance
        ):
            raise ValueError("联合偏置事件距离不能超过事件偏置支持范围。")
        if (
            self.physical_time_gap_bias is not None
            or self.event_time_interaction_bias is not None
        ) and self.chunk_summary_enabled:
            raise ValueError(
                "真实时间偏置首版不与组块摘要 token 组合；摘要 token 没有唯一 onset。"
            )
        adaptive_bias_config = config.get(
            "adaptive_directional_position_bias", {}
        )
        self.adaptive_directional_bias_gate = None
        if bool(adaptive_bias_config.get("enabled", False)):
            if (
                self.local_relative_position_bias is None
                or self.local_relative_position_bias.mode != "directed"
            ):
                raise ValueError(
                    "自适应有方向软偏置要求启用 directed 局部相对位置偏置。"
                )
            self.adaptive_directional_bias_gate = (
                AdaptiveDirectionalBiasGate(dimension)
            )
        partition_config = config.get("local_global_attention", {})
        self.local_attention_heads = 0
        self.local_attention_radius = 0
        self.total_attention_heads = heads
        if bool(partition_config.get("enabled", False)):
            self.local_attention_heads = int(
                partition_config.get("local_heads", heads // 2)
            )
            self.local_attention_radius = int(
                partition_config.get("radius", 4)
            )
            if not 0 < self.local_attention_heads < heads:
                raise ValueError(
                    "局部注意头数必须大于 0 且小于 Transformer 总头数。"
                )
            if self.local_attention_radius <= 0:
                raise ValueError("局部注意半径必须为正数。")
            if self.local_relative_position_bias is None:
                raise ValueError("局部/全局混合注意要求启用局部相对位置偏置。")
            if (
                self.local_relative_position_bias.max_distance
                != self.local_attention_radius
            ):
                raise ValueError("局部注意半径必须与相对位置偏置半径一致。")

        hierarchy_config = config.get("hierarchical_attention", {})
        self.hierarchical_attention_radii = None
        if bool(hierarchy_config.get("enabled", False)):
            if self.local_attention_heads:
                raise ValueError("逐层感受野不能与按头局部/全局分工同时启用。")
            if self.local_relative_position_bias is not None:
                raise ValueError("首版逐层感受野实验不叠加相对位置偏置。")
            stages = hierarchy_config.get("stages", ())
            radii = []
            for stage in stages:
                layer_count = int(stage.get("layers", 0))
                radius = stage.get("radius")
                if layer_count <= 0:
                    raise ValueError("逐层感受野每个阶段的层数必须为正数。")
                if radius == "global" or radius is None:
                    parsed_radius = None
                else:
                    parsed_radius = int(radius)
                    if parsed_radius <= 0:
                        raise ValueError("局部注意半径必须为正数或 global。")
                radii.extend([parsed_radius] * layer_count)
            depth = int(config.get("depth", 16))
            if len(radii) != depth:
                raise ValueError("逐层感受野阶段层数之和必须等于 Transformer 深度。")
            attention_layers = [
                layer[1]
                for layer_type, layer in zip(
                    self.encoder.layer_types, self.encoder.layers
                )
                if layer_type == "a"
            ]
            if len(attention_layers) != depth:
                raise RuntimeError("无法将逐层感受野映射到全部自注意力层。")
            for attention, radius in zip(attention_layers, radii):
                attention.local_attention_radius = radius
                if radius is not None:
                    attention.register_forward_pre_hook(
                        _local_attention_window_pre_hook,
                        with_kwargs=True,
                    )
            self.hierarchical_attention_radii = tuple(radii)
        if self.query_window_left_radius is not None and (
            self.causal_attention
            or self.chunk_summary_enabled
            or self.local_attention_heads
            or self.hierarchical_attention_radii is not None
            or self.local_relative_position_bias is not None
            or self.physical_time_gap_bias is not None
            or self.event_time_interaction_bias is not None
        ):
            raise ValueError("独立查询窗口实验不能叠加其他上下文注意力变体。")

    def _forward_query_windows(self, features, groups):
        """每个目标词独立编码指定左右窗口，层间不传递窗外信息。"""
        left_radius = self.query_window_left_radius
        right_radius = self.query_window_right_radius
        offsets = torch.arange(
            -left_radius, right_radius + 1, device=features.device
        )
        all_rows = []
        all_neighbors = []
        all_masks = []
        for indices in groups.values():
            rows = torch.as_tensor(indices, device=features.device)
            positions = torch.arange(len(indices), device=features.device)
            neighbors = positions[:, None] + offsets[None, :]
            valid = (neighbors >= 0) & (neighbors < len(indices))
            all_rows.append(rows)
            all_neighbors.append(rows[neighbors.clamp(0, len(indices) - 1)])
            all_masks.append(valid)
        row_order = torch.cat(all_rows)
        window_features = features[torch.cat(all_neighbors)]
        window_mask = torch.cat(all_masks)
        encoded = window_features
        for _ in range(self.shared_passes):
            if self.training and torch.is_grad_enabled():
                encoded = checkpoint(
                    lambda value: self.encoder(value, mask=window_mask),
                    encoded,
                    use_reentrant=False,
                )
            else:
                encoded = self.encoder(encoded, mask=window_mask)
        output = torch.empty_like(features)
        output[row_order] = encoded[:, left_radius]
        if self.soft_bounded_residual_max_norm is not None:
            output = _apply_soft_bounded_residual(
                features,
                output - features,
                self.soft_bounded_residual_max_norm,
            )
        return output

    def local_global_attention_mask(self, length, device):
        """前若干头只看局部窗口，其余头保留完整句子注意力。"""
        if self.local_attention_heads == 0:
            return None
        positions = torch.arange(int(length), device=device)
        relative = positions[None, :] - positions[:, None]
        local_mask = relative.abs() <= self.local_attention_radius
        global_mask = torch.ones_like(local_mask)
        return torch.cat(
            (
                local_mask[None].expand(self.local_attention_heads, -1, -1),
                global_mask[None].expand(
                    self.total_attention_heads - self.local_attention_heads,
                    -1,
                    -1,
                ),
            ),
            dim=0,
        )

    def forward(
        self,
        features,
        group_indices,
        event_times=None,
        causal_override=None,
    ):
        if group_indices is None:
            raise ValueError("group_indices is required by the context transformer.")
        requires_event_times = (
            self.physical_time_gap_bias is not None
            or self.event_time_interaction_bias is not None
        )
        if requires_event_times:
            if event_times is None:
                raise ValueError("启用物理时间偏置时必须提供 event_times。")
            if event_times.ndim != 1 or len(event_times) != len(features):
                raise ValueError("event_times 必须与 features 行一一对应。")
            if not torch.isfinite(event_times).all():
                raise ValueError("event_times 包含非有限值。")
        group_values = group_indices.detach().cpu().tolist()
        groups = {}
        for row_index, group in enumerate(group_values):
            groups.setdefault(int(group), []).append(row_index)
        if self.query_window_left_radius is not None:
            return self._forward_query_windows(features, groups)
        augmented_lengths = []
        for indices in groups.values():
            summary_count = 0
            if self.chunk_summary_enabled:
                summary_count = sum(
                    min(
                        self.chunk_summary_size,
                        len(indices) - start,
                    )
                    >= self.chunk_summary_minimum_size
                    for start in range(0, len(indices), self.chunk_summary_size)
                )
            augmented_lengths.append(len(indices) + summary_count)
        max_length = max(augmented_lengths)
        padded = features.new_zeros((len(groups), max_length, features.shape[-1]))
        mask = torch.zeros(
            len(groups), max_length, device=features.device, dtype=torch.bool
        )
        padded_event_times = None
        if requires_event_times:
            padded_event_times = torch.zeros(
                (len(groups), max_length),
                device=features.device,
                dtype=torch.float64,
            )
        ordered_indices = []
        ordered_word_positions = []
        for group_index, indices in enumerate(groups.values()):
            index_tensor = torch.tensor(indices, device=features.device)
            grouped_features = features[index_tensor]
            if padded_event_times is not None:
                grouped_times = event_times[index_tensor].to(dtype=torch.float64)
                if len(grouped_times) > 1 and torch.any(
                    torch.diff(grouped_times) <= 0
                ):
                    raise ValueError("同一句内的 event_times 必须严格递增。")
            if self.chunk_summary_enabled:
                augmented_parts = []
                word_positions = []
                cursor = 0
                for start in range(
                    0, len(indices), self.chunk_summary_size
                ):
                    chunk = grouped_features[
                        start : start + self.chunk_summary_size
                    ]
                    augmented_parts.append(chunk)
                    word_positions.extend(range(cursor, cursor + len(chunk)))
                    cursor += len(chunk)
                    if len(chunk) >= self.chunk_summary_minimum_size:
                        augmented_parts.append(chunk.mean(dim=0, keepdim=True))
                        cursor += 1
                grouped_features = torch.cat(augmented_parts, dim=0)
                word_positions = torch.tensor(
                    word_positions, device=features.device
                )
            else:
                word_positions = torch.arange(
                    len(indices), device=features.device
                )
            padded[group_index, : len(grouped_features)] = grouped_features
            mask[group_index, : len(grouped_features)] = True
            if padded_event_times is not None:
                padded_event_times[group_index, : len(grouped_times)] = grouped_times
            ordered_indices.append(index_tensor)
            ordered_word_positions.append(word_positions)
        attention_bias = None
        if self.local_relative_position_bias is not None:
            attention_bias = self.local_relative_position_bias(
                max_length,
                device=padded.device,
                dtype=padded.dtype,
                heads=(
                    self.total_attention_heads
                    if self.local_attention_heads
                    else None
                ),
                active_heads=(
                    self.local_attention_heads
                    if self.local_attention_heads
                    else None
                ),
            )
            if self.adaptive_directional_bias_gate is not None:
                directional_gates = self.adaptive_directional_bias_gate(
                    padded
                )
                positions = torch.arange(max_length, device=padded.device)
                relative = positions[None, :] - positions[:, None]
                directional_scale = torch.ones(
                    (
                        padded.shape[0],
                        max_length,
                        max_length,
                    ),
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
                attention_bias = (
                    attention_bias * directional_scale[:, None]
                )
        if self.physical_time_gap_bias is not None:
            time_bias = self.physical_time_gap_bias(
                padded_event_times,
                valid_mask=mask,
                dtype=padded.dtype,
            )
            attention_bias = (
                time_bias
                if attention_bias is None
                else attention_bias + time_bias
            )
        if self.event_time_interaction_bias is not None:
            interaction_bias = self.event_time_interaction_bias(
                padded_event_times,
                valid_mask=mask,
                dtype=padded.dtype,
            )
            attention_bias = (
                interaction_bias
                if attention_bias is None
                else attention_bias + interaction_bias
            )
        attention_mask = self.local_global_attention_mask(
            max_length, padded.device
        )
        encoded = padded
        use_causal_attention = (
            self.causal_attention
            if causal_override is None
            else bool(causal_override)
        )
        for _ in range(self.shared_passes):
            encoder_kwargs = {
                "mask": mask,
                "attn_mask": attention_mask,
                "attn_bias": attention_bias,
            }
            if use_causal_attention:
                encoder_kwargs["causal"] = True
            encoded = self.encoder(encoded, **encoder_kwargs)
        output = torch.empty_like(features)
        for group_index, (indices, word_positions) in enumerate(
            zip(ordered_indices, ordered_word_positions)
        ):
            output[indices] = encoded[group_index, word_positions]
        if self.soft_bounded_residual_max_norm is not None:
            output = _apply_soft_bounded_residual(
                features,
                output - features,
                self.soft_bounded_residual_max_norm,
            )
        return output


class SalienceGatedAdditiveEncoder(nn.Module):
    """词内显著性门控加性跨词注意力编码器（物理阻断跨词内积词长捷径与条件开关）。

    所有非线性门控严格在单个词窗内部独立完成：
        psi_j^(h) = sigmoid(w_G^(h)^T z_check_j + b_G^(h)) * (W_V^(h) z_check_j)
    跨词聚合仅通过可学习有向相对词位权重 A_{i,j}^(h) 线性求和（包含 j=i 自项以支持重叠背景
    噪声的线性 Wiener 白化），随后在保持单窗锥轴投影 c_i = z_i^T u_0 恒定的正交球面上做软有界残差更新。
    """

    def __init__(self, dimension: int, config: dict | None = None):
        super().__init__()
        cfg = dict(config or {})
        self.dimension = int(dimension)
        self.heads = int(cfg.get("heads", 16))
        self.head_dimension = int(cfg.get("head_dimension", 32))
        self.inner_dimension = self.heads * self.head_dimension
        self.max_relative_distance = int(cfg.get("max_relative_distance", 24))
        self.residual_max_norm = float(cfg.get("residual_max_norm", 0.65))
        self.ema_momentum = float(cfg.get("ema_momentum", 0.02))
        self.causal_attention = bool(cfg.get("causal_attention", False))

        if self.dimension <= 0 or self.heads <= 0 or self.head_dimension <= 0:
            raise ValueError("SalienceGatedAdditiveEncoder 维度与头数必须为正整数。")
        if self.max_relative_distance <= 0:
            raise ValueError("max_relative_distance 必须为正整数。")
        if self.residual_max_norm <= 0.0:
            raise ValueError("residual_max_norm 必须为正数。")

        # 保持与 GroupedTransformerEncoder 对齐的只读属性契约
        self.query_window_radius = None
        self.query_window_left_radius = None
        self.query_window_right_radius = None
        self.chunk_summary_enabled = False
        self.physical_time_gap_bias = None
        self.event_time_interaction_bias = None
        self.local_relative_position_bias = None
        self.is_salience_gated_additive = True

        self.gate_proj = nn.Linear(self.dimension, self.heads, bias=True)
        self.value_proj = nn.Linear(
            self.dimension, self.inner_dimension, bias=False
        )
        num_offsets = 2 * self.max_relative_distance + 1
        init_logits = torch.zeros(self.heads, num_offsets, dtype=torch.float32)
        offsets = torch.arange(
            -self.max_relative_distance,
            self.max_relative_distance + 1,
            dtype=torch.float32,
        )
        dist_excess = (offsets.abs() - 4.0).clamp(min=0.0)
        init_logits.copy_((-0.25 * dist_excess).unsqueeze(0).expand_as(init_logits))
        self.rel_pos_logits = nn.Parameter(init_logits)

        self.dropout = nn.Dropout(float(cfg.get("attention_dropout", 0.1)))
        self.out_proj = nn.Linear(
            self.inner_dimension, self.dimension, bias=False
        )
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)

        self.register_buffer(
            "clean_input_mean",
            torch.zeros(self.dimension, dtype=torch.float32),
        )
        self.register_buffer(
            "clean_input_mean_ready",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "psi_clean_ema",
            torch.zeros(self.inner_dimension, dtype=torch.float32),
        )
        self.register_buffer(
            "psi_clean_ema_ready",
            torch.tensor(False, dtype=torch.bool),
        )

    @torch.no_grad()
    def set_clean_input_mean(self, mean_vector: torch.Tensor) -> None:
        """从训练集全量冻结 Stage-1 特征设置精确输入均值与语义锥轴。"""
        vec = mean_vector.detach().to(
            device=self.clean_input_mean.device, dtype=torch.float32
        )
        if vec.ndim != 1 or int(vec.numel()) != self.dimension:
            raise ValueError(
                f"clean_input_mean 维度不匹配：期望 [{self.dimension}]，收到 {tuple(vec.shape)}"
            )
        self.clean_input_mean.copy_(vec)
        self.clean_input_mean_ready.fill_(True)

    def _cone_preserving_readout(
        self,
        z_base: torch.Tensor,
        g_raw: torch.Tensor,
        u0_axis: torch.Tensor,
    ) -> torch.Tensor:
        """在保持单窗锥轴分量 c_i = z_i^T u_0 不变的正交子空间球面上合并有界跨词更新。"""
        z_f = F.normalize(z_base.float(), dim=-1)
        u0 = F.normalize(u0_axis.float(), dim=-1)
        c = (z_f * u0).sum(dim=-1, keepdim=True).clamp(-0.999999, 0.999999)
        z_perp = z_f - c * u0

        g_f = g_raw.float()
        g_perp = g_f - (g_f * u0).sum(dim=-1, keepdim=True) * u0
        g_norm = g_perp.norm(p=2, dim=-1, keepdim=True)
        alpha = float(self.residual_max_norm)
        delta_perp = (
            alpha * torch.tanh(g_norm / max(alpha, 1e-6)) / (g_norm + 1e-6)
        ) * g_perp

        q_perp = F.normalize(z_perp + delta_perp, dim=-1)
        r_perp = torch.sqrt((1.0 - c * c).clamp(min=0.0))
        y_f = c * u0 + r_perp * q_perp
        return F.normalize(y_f, dim=-1).to(dtype=z_base.dtype)

    def forward_padded(
        self,
        padded: torch.Tensor,
        mask: torch.Tensor,
        *,
        causal_override: bool | None = None,
        target_col_idx: torch.Tensor | None = None,
        target_group_idx: torch.Tensor | None = None,
        num_branches: int = 1,
        synchronize_dropout_rng: bool = True,
        shared_dropout_fn=None,
    ) -> torch.Tensor:
        """在 [B_tot, L, D] 填充张量上执行零循环词内显著性门控加性聚合。"""
        bsz, max_len, dim = padded.shape
        work_dtype = (
            torch.get_autocast_dtype("cuda")
            if (padded.is_cuda and torch.is_autocast_enabled())
            else padded.dtype
        )
        use_causal = (
            self.causal_attention
            if causal_override is None
            else bool(causal_override)
        )
        valid_flat_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        n_valid = int(valid_flat_idx.numel())
        if n_valid == 0:
            if target_col_idx is not None:
                win_idx = (
                    target_group_idx
                    if target_group_idx is not None
                    else torch.arange(bsz, device=padded.device, dtype=torch.long)
                )
                return padded[win_idx, target_col_idx]
            return padded

        eff_branches = (
            int(num_branches)
            if (int(num_branches) > 1 and bsz % int(num_branches) == 0)
            else 1
        )
        g_per_branch = bsz // eff_branches

        # 1. 分支中心化与单位锥轴提取（Clean 与推理统一锁定训练集全局均值，Shifted 训练分支独立去均值）
        padded_f32 = padded.float()
        mask_f32 = mask.unsqueeze(-1).float()
        if self.training and not bool(self.clean_input_mean_ready.item()):
            with torch.no_grad():
                b0_mask = mask_f32[:g_per_branch]
                b0_cnt = b0_mask.sum().clamp(min=1.0)
                b0_mean = (padded_f32[:g_per_branch] * b0_mask).sum(dim=(0, 1)) / b0_cnt
                self.clean_input_mean.copy_(b0_mean)
                self.clean_input_mean_ready.fill_(True)

        if eff_branches == 1 or not self.training:
            mu_per_row = self.clean_input_mean.view(1, 1, dim)
        else:
            with torch.no_grad():
                br_pad = padded_f32.view(eff_branches, g_per_branch, max_len, dim)
                br_msk = mask_f32.view(eff_branches, g_per_branch, max_len, 1)
                br_cnt = br_msk.sum(dim=(1, 2)).clamp(min=1.0)
                br_means = (br_pad * br_msk).sum(dim=(1, 2)) / br_cnt
                if bool(self.clean_input_mean_ready.item()):
                    br_means[0] = self.clean_input_mean
                mu_per_row = (
                    br_means[:, None, None, :]
                    .expand(eff_branches, g_per_branch, 1, dim)
                    .reshape(bsz, 1, dim)
                )

        zc_all = ((padded_f32 - mu_per_row) * mask_f32).to(dtype=work_dtype)
        zc_valid = zc_all.reshape(bsz * max_len, dim).index_select(0, valid_flat_idx)

        # 2. 纯词内一元显著性门控与线性值映射（仅在有效 Token 上计算）
        gates = torch.sigmoid(self.gate_proj(zc_valid))  # [N_valid, H]
        vals = self.value_proj(zc_valid).view(
            n_valid, self.heads, self.head_dimension
        )
        psi_valid = (gates.unsqueeze(-1) * vals).reshape(
            n_valid, self.inner_dimension
        )

        psi_padded = zc_valid.new_zeros((bsz * max_len, self.inner_dimension))
        psi_padded.index_copy_(0, valid_flat_idx, psi_valid)
        psi_padded = psi_padded.view(bsz, max_len, self.inner_dimension)

        # 3. 句外（Leave-Own-Sentence-Out）均值中心化与推理期 EMA 统计更新
        if self.training:
            psi_br = psi_padded.float().view(
                eff_branches, g_per_branch, max_len, self.inner_dimension
            )
            msk_br = mask_f32.view(eff_branches, g_per_branch, max_len, 1)
            grp_sum = (psi_br * msk_br).sum(dim=2)  # [B_br, G, D_in]
            grp_cnt = msk_br.sum(dim=2)  # [B_br, G, 1]
            br_sum = grp_sum.sum(dim=1, keepdim=True)  # [B_br, 1, D_in]
            br_cnt = grp_cnt.sum(dim=1, keepdim=True)  # [B_br, 1, 1]

            with torch.no_grad():
                clean_cnt = float(br_cnt[0, 0, 0].item())
                if clean_cnt > 0.0:
                    clean_batch_mean = br_sum[0, 0] / clean_cnt
                    if not bool(self.psi_clean_ema_ready.item()):
                        self.psi_clean_ema.copy_(clean_batch_mean)
                        self.psi_clean_ema_ready.fill_(True)
                    else:
                        self.psi_clean_ema.lerp_(
                            clean_batch_mean, float(self.ema_momentum)
                        )

            other_cnt = br_cnt - grp_cnt
            fallback_mean = self.psi_clean_ema.view(1, 1, self.inner_dimension)
            loso_mean = torch.where(
                other_cnt > 0.5,
                (br_sum - grp_sum) / other_cnt.clamp(min=1.0),
                fallback_mean.expand_as(grp_sum),
            )
            psi_centered = (
                (psi_br - loso_mean.unsqueeze(2)) * msk_br
            ).reshape(bsz, max_len, self.inner_dimension).to(dtype=work_dtype)
        else:
            psi_centered = (
                (psi_padded.float() - self.psi_clean_ema.view(1, 1, -1))
                * mask_f32
            ).to(dtype=work_dtype)

        if shared_dropout_fn is not None:
            psi_centered = shared_dropout_fn(
                psi_centered,
                self.dropout,
                num_branches=eff_branches,
                synchronize_dropout_rng=synchronize_dropout_rng,
            )
        else:
            psi_centered = self.dropout(psi_centered)

        # 4. 有向相对词位注意力线性汇聚（零跨词 QK 内积）
        r_max = self.max_relative_distance
        pos_k = torch.arange(max_len, device=padded.device, dtype=torch.long)

        if target_col_idx is not None:
            win_idx = (
                target_group_idx
                if target_group_idx is not None
                else torch.arange(bsz, device=padded.device, dtype=torch.long)
            )
            rel_q = pos_k.unsqueeze(0) - target_col_idx.unsqueeze(1)  # [Q, L]
            rel_idx_q = rel_q.clamp(-r_max, r_max) + r_max
            logits_q = self.rel_pos_logits[:, rel_idx_q].permute(1, 0, 2).float()
            key_valid_q = mask[win_idx].unsqueeze(1)  # [Q, 1, L]
            if use_causal:
                key_valid_q = key_valid_q & (rel_q.unsqueeze(1) <= 0)
            logits_q = logits_q.masked_fill(~key_valid_q, -1e9)
            attn_q = torch.softmax(logits_q, dim=-1).to(dtype=work_dtype)
            any_valid_q = key_valid_q.any(dim=-1, keepdim=True)
            attn_q = torch.where(any_valid_q, attn_q, torch.zeros_like(attn_q))

            psi_q = psi_centered[win_idx].view(
                int(win_idx.shape[0]), max_len, self.heads, self.head_dimension
            )
            ctx_q = torch.einsum("qhl,qlhd->qhd", attn_q, psi_q).reshape(
                int(win_idx.shape[0]), self.inner_dimension
            )
            g_raw = self.out_proj(ctx_q)
            z_base = padded[win_idx, target_col_idx]
            u0_q = (
                self.clean_input_mean.unsqueeze(0)
                if (eff_branches == 1 or not self.training)
                else mu_per_row.squeeze(1)[win_idx]
            )
            return self._cone_preserving_readout(z_base, g_raw, u0_q)

        rel_full = pos_k.unsqueeze(0) - pos_k.unsqueeze(1)  # [L, L]
        rel_idx_full = rel_full.clamp(-r_max, r_max) + r_max
        logits_full = (
            self.rel_pos_logits[:, rel_idx_full].unsqueeze(0).float()
        )  # [1, H, L, L]
        key_valid = mask[:, None, None, :]
        if use_causal:
            key_valid = key_valid & (rel_full[None, None, :, :] <= 0)
        logits_full = logits_full.masked_fill(~key_valid, -1e9)
        attn_full = torch.softmax(logits_full, dim=-1).to(dtype=work_dtype)
        any_valid = key_valid.any(dim=-1, keepdim=True)
        attn_full = torch.where(
            any_valid, attn_full, torch.zeros_like(attn_full)
        )

        psi_heads = psi_centered.view(
            bsz, max_len, self.heads, self.head_dimension
        )
        ctx_full = torch.einsum("bhij,bjhd->bihd", attn_full, psi_heads).reshape(
            bsz, max_len, self.inner_dimension
        )
        g_raw_full = self.out_proj(ctx_full)
        y_full = self._cone_preserving_readout(padded, g_raw_full, mu_per_row)
        return y_full * mask.unsqueeze(-1).to(dtype=y_full.dtype)

    def forward(
        self,
        features: torch.Tensor,
        group_indices: torch.Tensor,
        event_times: torch.Tensor | None = None,
        causal_override: bool | None = None,
    ) -> torch.Tensor:
        if group_indices is None:
            raise ValueError("group_indices is required by the context transformer.")
        group_values = group_indices.detach().cpu().tolist()
        groups: dict[int, list[int]] = {}
        for row_index, group in enumerate(group_values):
            groups.setdefault(int(group), []).append(row_index)
        max_length = max((len(g) for g in groups.values()), default=1)
        padded = features.new_zeros((len(groups), max_length, features.shape[-1]))
        mask = torch.zeros(
            (len(groups), max_length), device=features.device, dtype=torch.bool
        )
        ordered_indices: list[torch.Tensor] = []
        for group_index, indices in enumerate(groups.values()):
            idx_t = torch.as_tensor(
                indices, device=features.device, dtype=torch.long
            )
            g_len = len(indices)
            padded[group_index, :g_len] = features.index_select(0, idx_t)
            mask[group_index, :g_len] = True
            ordered_indices.append(idx_t)
        encoded = self.forward_padded(
            padded,
            mask,
            causal_override=causal_override,
            num_branches=1,
            synchronize_dropout_rng=False,
        )
        output = torch.empty_like(features)
        for group_index, idx_t in enumerate(ordered_indices):
            output[idx_t] = encoded[group_index, : len(idx_t)]
        return output


class SalienceGatedAdditiveV2Encoder(nn.Module):
    """词内显著性门控加性跨词编码器 V2（有符号定分母零汇核 + 独立文本锥轴锁 + 视界条件化）。

    严格保持词内一元非线性门控（零跨词 Q·K 内积或跨词交互 MLP），在 V1 基础上引入：
    1. 有符号定分母零汇核（Signed Fixed-Denominator Zero-Sink Kernel）：
       对全部偏移 \\Delta \\in [-R, R] 与单标量零汇项联合 softmax：
           w_\\Delta^{(h)} = p_{+,\\Delta}^{(h)} - p_{-,\\Delta}^{(h)},  \\sum_\\Delta |w_\\Delta^{(h)}| < 1
       天然支持负权重线性 Wiener 去噪、无边界重归一化长度泄漏，且采用非对称初始化
       （``neg_logits = pos_logits - 2.0``）保证 step-0 满秩非零梯度；
    2. 独立持久化文本锥轴缓冲区 ``text_cone_axis``：与输入中心化均值 ``clean_input_mean`` 解耦，
       即使上游 Stage-1 采用混合全词表 CE 损失，仍物理锁定单窗在 T5-Large 词表锥轴上的投影不变；
    3. 可选视界条件化（``horizon_conditioned=True``，用于第 3 章 ``r \\in \\{0, 1, 2\\}``）：
       按可用前瞻步数 ``avail_r = clamp(last_valid_col - query_col, 0, num_horizons - 1)``
       选择每视界专用核与零初始化低秩残差读出，且 ``forward`` 与 ``forward_padded`` 严格逐位一致。
    """

    def __init__(self, dimension: int, config: dict | None = None):
        super().__init__()
        cfg = dict(config or {})
        self.dimension = int(dimension)
        self.heads = int(cfg.get("heads", 32))
        self.head_dimension = int(cfg.get("head_dimension", 64))
        self.inner_dimension = self.heads * self.head_dimension
        self.max_relative_distance = int(cfg.get("max_relative_distance", 24))
        self.residual_max_norm = float(cfg.get("residual_max_norm", 0.85))
        self.ema_momentum = float(cfg.get("ema_momentum", 0.02))
        self.causal_attention = bool(cfg.get("causal_attention", False))
        self.horizon_conditioned = bool(cfg.get("horizon_conditioned", False))
        self.num_horizons = int(cfg.get("num_horizons", 3))
        self.horizon_readout_rank = int(cfg.get("horizon_readout_rank", 128))

        if self.dimension <= 0 or self.heads <= 0 or self.head_dimension <= 0:
            raise ValueError("SalienceGatedAdditiveV2Encoder 维度与头数必须为正整数。")
        if self.max_relative_distance <= 0:
            raise ValueError("max_relative_distance 必须为正整数。")
        if self.residual_max_norm <= 0.0:
            raise ValueError("residual_max_norm 必须为正数。")
        if self.horizon_conditioned and (
            self.num_horizons <= 0 or self.horizon_readout_rank <= 0
        ):
            raise ValueError("num_horizons 与 horizon_readout_rank 必须为正整数。")

        # 保持与 GroupedTransformerEncoder 对齐的只读属性契约
        self.query_window_radius = None
        self.query_window_left_radius = None
        self.query_window_right_radius = None
        self.chunk_summary_enabled = False
        self.physical_time_gap_bias = None
        self.event_time_interaction_bias = None
        self.local_relative_position_bias = None
        self.is_salience_gated_additive = True

        self.gate_proj = nn.Linear(self.dimension, self.heads, bias=True)
        self.value_proj = nn.Linear(
            self.dimension, self.inner_dimension, bias=False
        )

        num_offsets = 2 * self.max_relative_distance + 1
        offsets = torch.arange(
            -self.max_relative_distance,
            self.max_relative_distance + 1,
            dtype=torch.float32,
        )
        dist_excess = (offsets.abs() - 4.0).clamp(min=0.0)
        base_pos = -0.25 * dist_excess
        base_neg = base_pos - 2.0  # 非对称初始化，严禁 pos == neg 导致 step-0 零梯度

        if self.horizon_conditioned:
            pos_init = (
                base_pos.view(1, 1, num_offsets)
                .expand(self.num_horizons, self.heads, num_offsets)
                .clone()
            )
            neg_init = (
                base_neg.view(1, 1, num_offsets)
                .expand(self.num_horizons, self.heads, num_offsets)
                .clone()
            )
            sink_init = torch.zeros(
                self.num_horizons, self.heads, dtype=torch.float32
            )
        else:
            pos_init = (
                base_pos.view(1, num_offsets)
                .expand(self.heads, num_offsets)
                .clone()
            )
            neg_init = (
                base_neg.view(1, num_offsets)
                .expand(self.heads, num_offsets)
                .clone()
            )
            sink_init = torch.zeros(self.heads, dtype=torch.float32)

        self.pos_logits = nn.Parameter(pos_init)
        self.neg_logits = nn.Parameter(neg_init)
        self.sink_logits = nn.Parameter(sink_init)

        self.dropout = nn.Dropout(float(cfg.get("attention_dropout", 0.1)))
        self.out_proj = nn.Linear(
            self.inner_dimension, self.dimension, bias=False
        )
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)

        if self.horizon_conditioned:
            self.horizon_down = nn.Parameter(
                torch.empty(
                    self.num_horizons,
                    self.horizon_readout_rank,
                    self.inner_dimension,
                    dtype=torch.float32,
                )
            )
            nn.init.normal_(
                self.horizon_down,
                mean=0.0,
                std=1.0 / (float(self.inner_dimension) ** 0.5),
            )
            self.horizon_up = nn.Parameter(
                torch.zeros(
                    self.num_horizons,
                    self.dimension,
                    self.horizon_readout_rank,
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("horizon_down", None)
            self.register_parameter("horizon_up", None)

        self.register_buffer(
            "clean_input_mean",
            torch.zeros(self.dimension, dtype=torch.float32),
        )
        self.register_buffer(
            "clean_input_mean_ready",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "text_cone_axis",
            torch.zeros(self.dimension, dtype=torch.float32),
        )
        self.register_buffer(
            "text_cone_axis_ready",
            torch.tensor(False, dtype=torch.bool),
        )
        self.register_buffer(
            "psi_clean_ema",
            torch.zeros(self.inner_dimension, dtype=torch.float32),
        )
        self.register_buffer(
            "psi_clean_ema_ready",
            torch.tensor(False, dtype=torch.bool),
        )

    @torch.no_grad()
    def set_clean_input_mean(self, mean_vector: torch.Tensor) -> None:
        """从训练集全量冻结 Stage-1 特征设置精确输入中心化均值。"""
        vec = mean_vector.detach().to(
            device=self.clean_input_mean.device, dtype=torch.float32
        )
        if vec.ndim != 1 or int(vec.numel()) != self.dimension:
            raise ValueError(
                f"clean_input_mean 维度不匹配：期望 [{self.dimension}]，收到 {tuple(vec.shape)}"
            )
        self.clean_input_mean.copy_(vec)
        self.clean_input_mean_ready.fill_(True)

    @torch.no_grad()
    def set_text_cone_axis(self, axis_vector: torch.Tensor) -> None:
        """绑定独立文本词表语义锥单位轴（严格来自训练集词表均值，防锥轴漂移或退化）。"""
        vec = axis_vector.detach().to(
            device=self.text_cone_axis.device, dtype=torch.float32
        )
        if vec.ndim != 1 or int(vec.numel()) != self.dimension:
            raise ValueError(
                f"text_cone_axis 维度不匹配：期望 [{self.dimension}]，收到 {tuple(vec.shape)}"
            )
        norm = float(vec.norm(p=2).item())
        if norm <= 1e-8:
            raise ValueError("text_cone_axis 范数过小，无法归一化。")
        self.text_cone_axis.copy_(vec / norm)
        self.text_cone_axis_ready.fill_(True)

    def _compute_signed_weight_table(self) -> torch.Tensor:
        """通过单次联合 softmax 计算有符号定分母零汇权重表 [H, 2R+1] 或 [K, H, 2R+1]。"""
        num_offsets = 2 * self.max_relative_distance + 1
        cat_logits = torch.cat(
            [
                self.pos_logits.float(),
                self.neg_logits.float(),
                self.sink_logits.float().unsqueeze(-1),
            ],
            dim=-1,
        )
        probs = torch.softmax(cat_logits, dim=-1)
        return probs[..., :num_offsets] - probs[..., num_offsets : 2 * num_offsets]

    def _apply_horizon_readout(
        self,
        ctx_flat: torch.Tensor,
        avail_r_flat: torch.Tensor | None,
    ) -> torch.Tensor:
        """共享线性投影 + 可选每视界零初始化低秩残差读出。"""
        g_raw = self.out_proj(ctx_flat)
        if not self.horizon_conditioned or avail_r_flat is None:
            return g_raw
        delta_h = torch.zeros_like(g_raw)
        for k in range(self.num_horizons):
            sel = avail_r_flat == k
            if bool(sel.any()):
                h_mid = F.linear(
                    ctx_flat[sel],
                    self.horizon_down[k].to(dtype=ctx_flat.dtype),
                )
                delta_h[sel] = F.linear(
                    h_mid,
                    self.horizon_up[k].to(dtype=ctx_flat.dtype),
                )
        return g_raw + delta_h

    def _cone_preserving_readout(
        self,
        z_base: torch.Tensor,
        g_raw: torch.Tensor,
        fallback_u0_axis: torch.Tensor,
    ) -> torch.Tensor:
        """在保持单窗沿文本锥轴投影 c_i = z_i^T u_{0,text} 不变的正交球面上合并有界跨词更新。"""
        z_f = F.normalize(z_base.float(), dim=-1)
        if bool(self.text_cone_axis_ready.item()):
            u0 = F.normalize(self.text_cone_axis.float(), dim=-1)
            while u0.ndim < z_f.ndim:
                u0 = u0.unsqueeze(0)
        else:
            u0 = F.normalize(fallback_u0_axis.float(), dim=-1, eps=1e-12)
        c = (z_f * u0).sum(dim=-1, keepdim=True).clamp(-0.999999, 0.999999)
        z_perp = z_f - c * u0

        g_f = g_raw.float()
        g_perp = g_f - (g_f * u0).sum(dim=-1, keepdim=True) * u0
        g_norm = g_perp.norm(p=2, dim=-1, keepdim=True)
        alpha = float(self.residual_max_norm)
        delta_perp = (
            alpha * torch.tanh(g_norm / max(alpha, 1e-6)) / (g_norm + 1e-6)
        ) * g_perp

        q_perp = F.normalize(z_perp + delta_perp, dim=-1)
        r_perp = torch.sqrt((1.0 - c * c).clamp(min=0.0))
        y_f = c * u0 + r_perp * q_perp
        return F.normalize(y_f, dim=-1).to(dtype=z_base.dtype)

    def forward_padded(
        self,
        padded: torch.Tensor,
        mask: torch.Tensor,
        *,
        causal_override: bool | None = None,
        target_col_idx: torch.Tensor | None = None,
        target_group_idx: torch.Tensor | None = None,
        num_branches: int = 1,
        synchronize_dropout_rng: bool = True,
        shared_dropout_fn=None,
    ) -> torch.Tensor:
        """在 [B_tot, L, D] 填充张量上执行零循环词内显著性门控有符号定分母聚合。"""
        bsz, max_len, dim = padded.shape
        work_dtype = (
            torch.get_autocast_dtype("cuda")
            if (padded.is_cuda and torch.is_autocast_enabled())
            else padded.dtype
        )
        use_causal = (
            self.causal_attention
            if causal_override is None
            else bool(causal_override)
        )
        valid_flat_idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        n_valid = int(valid_flat_idx.numel())
        if n_valid == 0:
            if target_col_idx is not None:
                win_idx = (
                    target_group_idx
                    if target_group_idx is not None
                    else torch.arange(bsz, device=padded.device, dtype=torch.long)
                )
                return padded[win_idx, target_col_idx]
            return padded

        eff_branches = (
            int(num_branches)
            if (int(num_branches) > 1 and bsz % int(num_branches) == 0)
            else 1
        )
        g_per_branch = bsz // eff_branches

        # 1. 分支输入去均值（仅用于构造 zc，不混淆独立文本锥轴 text_cone_axis）
        padded_f32 = padded.float()
        mask_f32 = mask.unsqueeze(-1).float()
        if self.training and not bool(self.clean_input_mean_ready.item()):
            with torch.no_grad():
                b0_mask = mask_f32[:g_per_branch]
                b0_cnt = b0_mask.sum().clamp(min=1.0)
                b0_mean = (padded_f32[:g_per_branch] * b0_mask).sum(dim=(0, 1)) / b0_cnt
                self.clean_input_mean.copy_(b0_mean)
                self.clean_input_mean_ready.fill_(True)

        if eff_branches == 1 or not self.training:
            mu_per_row = self.clean_input_mean.view(1, 1, dim)
        else:
            with torch.no_grad():
                br_pad = padded_f32.view(eff_branches, g_per_branch, max_len, dim)
                br_msk = mask_f32.view(eff_branches, g_per_branch, max_len, 1)
                br_cnt = br_msk.sum(dim=(1, 2)).clamp(min=1.0)
                br_means = (br_pad * br_msk).sum(dim=(1, 2)) / br_cnt
                if bool(self.clean_input_mean_ready.item()):
                    br_means[0] = self.clean_input_mean
                mu_per_row = (
                    br_means[:, None, None, :]
                    .expand(eff_branches, g_per_branch, 1, dim)
                    .reshape(bsz, 1, dim)
                )

        zc_all = ((padded_f32 - mu_per_row) * mask_f32).to(dtype=work_dtype)
        zc_valid = zc_all.reshape(bsz * max_len, dim).index_select(0, valid_flat_idx)

        # 2. 纯词内一元显著性门控与线性值映射
        gates = torch.sigmoid(self.gate_proj(zc_valid))  # [N_valid, H]
        vals = self.value_proj(zc_valid).view(
            n_valid, self.heads, self.head_dimension
        )
        psi_valid = (gates.unsqueeze(-1) * vals).reshape(
            n_valid, self.inner_dimension
        )

        psi_padded = zc_valid.new_zeros((bsz * max_len, self.inner_dimension))
        psi_padded.index_copy_(0, valid_flat_idx, psi_valid)
        psi_padded = psi_padded.view(bsz, max_len, self.inner_dimension)

        # 3. 句外（LOSO）均值中心化与推理期 EMA 统计更新
        if self.training:
            psi_br = psi_padded.float().view(
                eff_branches, g_per_branch, max_len, self.inner_dimension
            )
            msk_br = mask_f32.view(eff_branches, g_per_branch, max_len, 1)
            grp_sum = (psi_br * msk_br).sum(dim=2)
            grp_cnt = msk_br.sum(dim=2)
            br_sum = grp_sum.sum(dim=1, keepdim=True)
            br_cnt = grp_cnt.sum(dim=1, keepdim=True)

            with torch.no_grad():
                clean_cnt = float(br_cnt[0, 0, 0].item())
                if clean_cnt > 0.0:
                    clean_batch_mean = br_sum[0, 0] / clean_cnt
                    if not bool(self.psi_clean_ema_ready.item()):
                        self.psi_clean_ema.copy_(clean_batch_mean)
                        self.psi_clean_ema_ready.fill_(True)
                    else:
                        self.psi_clean_ema.lerp_(
                            clean_batch_mean, float(self.ema_momentum)
                        )

            other_cnt = br_cnt - grp_cnt
            fallback_mean = self.psi_clean_ema.view(1, 1, self.inner_dimension)
            loso_mean = torch.where(
                other_cnt > 0.5,
                (br_sum - grp_sum) / other_cnt.clamp(min=1.0),
                fallback_mean.expand_as(grp_sum),
            )
            psi_centered = (
                (psi_br - loso_mean.unsqueeze(2)) * msk_br
            ).reshape(bsz, max_len, self.inner_dimension).to(dtype=work_dtype)
        else:
            psi_centered = (
                (psi_padded.float() - self.psi_clean_ema.view(1, 1, -1))
                * mask_f32
            ).to(dtype=work_dtype)

        if shared_dropout_fn is not None:
            psi_centered = shared_dropout_fn(
                psi_centered,
                self.dropout,
                num_branches=eff_branches,
                synchronize_dropout_rng=synchronize_dropout_rng,
            )
        else:
            psi_centered = self.dropout(psi_centered)

        # 4. 有符号定分母零汇相对位置核汇聚（零边界重归一化泄漏 + 严格 forward/forward_padded 视界一致）
        r_max = self.max_relative_distance
        pos_k = torch.arange(max_len, device=padded.device, dtype=torch.long)
        w_table = self._compute_signed_weight_table().to(dtype=work_dtype)

        if target_col_idx is not None:
            win_idx = (
                target_group_idx
                if target_group_idx is not None
                else torch.arange(bsz, device=padded.device, dtype=torch.long)
            )
            rel_q = pos_k.unsqueeze(0) - target_col_idx.unsqueeze(1)  # [Q, L]
            in_range_q = (rel_q >= -r_max) & (rel_q <= r_max)
            rel_idx_q = rel_q.clamp(-r_max, r_max) + r_max
            key_valid_q = mask[win_idx].unsqueeze(1) & in_range_q.unsqueeze(1)
            if use_causal:
                key_valid_q = key_valid_q & (rel_q.unsqueeze(1) <= 0)

            avail_r_q: torch.Tensor | None = None
            if self.horizon_conditioned:
                if use_causal:
                    avail_r_q = torch.zeros_like(target_col_idx)
                else:
                    last_col_q = mask[win_idx].long().sum(dim=-1) - 1
                    avail_r_q = (last_col_q - target_col_idx).clamp(
                        0, self.num_horizons - 1
                    )
                head_arange = torch.arange(
                    self.heads, device=padded.device, dtype=torch.long
                )
                attn_q = w_table[
                    avail_r_q[:, None, None],
                    head_arange[None, :, None],
                    rel_idx_q[:, None, :],
                ]
            else:
                attn_q = w_table[:, rel_idx_q].permute(1, 0, 2)
            attn_q = attn_q * key_valid_q.to(dtype=work_dtype)

            psi_q = psi_centered[win_idx].view(
                int(win_idx.shape[0]), max_len, self.heads, self.head_dimension
            )
            ctx_q = torch.einsum("qhl,qlhd->qhd", attn_q, psi_q).reshape(
                int(win_idx.shape[0]), self.inner_dimension
            )
            g_raw = self._apply_horizon_readout(ctx_q, avail_r_q)
            z_base = padded[win_idx, target_col_idx]
            u0_q = (
                self.clean_input_mean.unsqueeze(0)
                if (eff_branches == 1 or not self.training)
                else mu_per_row.squeeze(1)[win_idx]
            )
            return self._cone_preserving_readout(z_base, g_raw, u0_q)

        rel_full = pos_k.unsqueeze(0) - pos_k.unsqueeze(1)  # [L, L]
        in_range_full = (rel_full >= -r_max) & (rel_full <= r_max)
        rel_idx_full = rel_full.clamp(-r_max, r_max) + r_max
        key_valid = mask[:, None, None, :] & in_range_full[None, None, :, :]
        if use_causal:
            key_valid = key_valid & (rel_full[None, None, :, :] <= 0)

        avail_r_full: torch.Tensor | None = None
        if self.horizon_conditioned:
            if use_causal:
                avail_r_full = torch.zeros(
                    (bsz, max_len), device=padded.device, dtype=torch.long
                )
            else:
                last_col_g = mask.long().sum(dim=-1, keepdim=True) - 1  # [B, 1]
                avail_r_full = (last_col_g - pos_k.unsqueeze(0)).clamp(
                    0, self.num_horizons - 1
                )
            head_arange = torch.arange(
                self.heads, device=padded.device, dtype=torch.long
            )
            attn_full = w_table[
                avail_r_full[:, None, :, None],
                head_arange[None, :, None, None],
                rel_idx_full[None, None, :, :],
            ]
        else:
            attn_full = w_table[:, rel_idx_full].unsqueeze(0)
        attn_full = attn_full * key_valid.to(dtype=work_dtype)

        psi_heads = psi_centered.view(
            bsz, max_len, self.heads, self.head_dimension
        )
        ctx_full = torch.einsum("bhij,bjhd->bihd", attn_full, psi_heads).reshape(
            bsz * max_len, self.inner_dimension
        )
        g_raw_full = self._apply_horizon_readout(
            ctx_full,
            avail_r_full.reshape(-1) if avail_r_full is not None else None,
        ).view(bsz, max_len, self.dimension)
        y_full = self._cone_preserving_readout(padded, g_raw_full, mu_per_row)
        return y_full * mask.unsqueeze(-1).to(dtype=y_full.dtype)

    def forward(
        self,
        features: torch.Tensor,
        group_indices: torch.Tensor,
        event_times: torch.Tensor | None = None,
        causal_override: bool | None = None,
    ) -> torch.Tensor:
        del event_times
        if group_indices is None:
            raise ValueError("group_indices is required by the context transformer.")
        group_values = group_indices.detach().cpu().tolist()
        groups: dict[int, list[int]] = {}
        for row_index, group in enumerate(group_values):
            groups.setdefault(int(group), []).append(row_index)
        max_length = max((len(g) for g in groups.values()), default=1)
        padded = features.new_zeros((len(groups), max_length, features.shape[-1]))
        mask = torch.zeros(
            (len(groups), max_length), device=features.device, dtype=torch.bool
        )
        ordered_indices: list[torch.Tensor] = []
        for group_index, indices in enumerate(groups.values()):
            idx_t = torch.as_tensor(
                indices, device=features.device, dtype=torch.long
            )
            g_len = len(indices)
            padded[group_index, :g_len] = features.index_select(0, idx_t)
            mask[group_index, :g_len] = True
            ordered_indices.append(idx_t)
        encoded = self.forward_padded(
            padded,
            mask,
            causal_override=causal_override,
            num_branches=1,
            synchronize_dropout_rng=False,
        )
        output = torch.empty_like(features)
        for group_index, idx_t in enumerate(ordered_indices):
            output[idx_t] = encoded[group_index, : len(idx_t)]
        return output


def build_context_encoder(dimension: int, config: dict | None = None) -> nn.Module:
    """根据 model.transformer.architecture 构建跨词编码器（默认保持标准 GroupedTransformerEncoder）。"""
    cfg = dict(config or {})
    arch = str(cfg.get("architecture", "standard")).strip().lower()
    if arch in {"", "standard", "grouped_transformer", "x_transformers"}:
        return GroupedTransformerEncoder(dimension, cfg)
    if arch in {
        "salience_gated_additive",
        "unary_gated_additive",
        "wiener_ag2",
    }:
        return SalienceGatedAdditiveEncoder(dimension, cfg)
    if arch in {
        "salience_gated_additive_v2",
        "unary_gated_additive_v2",
        "signed_zero_sink_additive",
    }:
        return SalienceGatedAdditiveV2Encoder(dimension, cfg)
    raise ValueError(
        f"不支持的 model.transformer.architecture={arch!r}，"
        "可选：'standard' | 'salience_gated_additive' | 'salience_gated_additive_v2'。"
    )
