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
        return output
