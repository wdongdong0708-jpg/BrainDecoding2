"""跨步卷积、时间注意力池化、显式时间滤波器与扩张卷积脑编码器。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from braindecoding.models.spatial import (
    ResidualSubjectLinear,
    SpatialChannelMerger,
    SubjectLinear,
)


class StridedConvEncoder(nn.Module):
    """将固定长度的多通道信号映射为归一化向量。"""

    def __init__(
        self,
        channel_positions,
        subject_count,
        embedding_dimension=768,
        virtual_channels=64,
        fourier_harmonics=8,
        spatial_margin=0.1,
        dropout=0.1,
    ):
        super().__init__()
        self.spatial_attention = SpatialChannelMerger(
            channel_positions,
            output_channels=virtual_channels,
            position_dimension=2 * int(fourier_harmonics) ** 2,
            subject_count=subject_count,
            per_subject=False,
            dropout_radius=0.0,
            projection_margin=spatial_margin,
            frequency_start=1,
            fourier_margin=0.0,
            scale_fourier_coordinates=False,
            initialization_std=0.02,
        )
        self.subject_layer = SubjectLinear(
            subject_count,
            virtual_channels,
            bias=True,
            initialization="identity",
        )
        self.encoder = nn.Sequential(
            nn.Conv1d(virtual_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(16, 256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dimension),
        )

    def forward(self, signals, subject_indices):
        features = self.spatial_attention(signals)
        features = self.subject_layer(features, subject_indices)
        features = self.encoder(features)
        return F.normalize(self.projection(features), dim=-1)


def build_strided_conv_encoder(channel_count, channel_positions, subject_count, config):
    """构建紧凑的跨步卷积信号编码器。"""
    if len(channel_positions) != int(channel_count):
        raise ValueError("Signal channels and channel_positions do not match.")
    return StridedConvEncoder(
        channel_positions=channel_positions,
        subject_count=subject_count,
        embedding_dimension=int(config.get("embedding_dimension", 768)),
        virtual_channels=int(config.get("virtual_channels", 64)),
        fourier_harmonics=int(config.get("fourier_harmonics", 8)),
        spatial_margin=float(config.get("spatial_margin", 0.1)),
        dropout=float(config.get("dropout", 0.1)),
    )


class TemporalAttentionPool(nn.Module):
    """使用学习得到的加性注意力汇聚时间特征序列。"""

    def __init__(self, input_dimension, hidden_dimension=256):
        super().__init__()
        self.key = nn.Linear(int(input_dimension), int(hidden_dimension))
        self.score = nn.Linear(int(hidden_dimension), 1)

    def forward(self, features):
        keys = features.transpose(1, 2)
        scores = self.score(torch.tanh(self.key(keys))).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(1)
        return torch.bmm(weights, keys).squeeze(1)


class ExplicitTemporalFilter(nn.Module):
    """Per-feature FIR filter, optionally with a zero-initialized subject residual."""

    def __init__(self, channels, subject_count, kernel_size, mode="shared"):
        super().__init__()
        channels = int(channels)
        subject_count = int(subject_count)
        kernel_size = int(kernel_size)
        if channels <= 0 or subject_count <= 0:
            raise ValueError("channels and subject_count must be positive.")
        if kernel_size <= 1 or kernel_size % 2 != 1:
            raise ValueError("temporal filter kernel_size must be odd and greater than one.")
        if mode not in {"shared", "subject_residual"}:
            raise ValueError("temporal filter mode must be shared or subject_residual.")
        self.channels = channels
        self.kernel_size = kernel_size
        self.mode = mode
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel_size))
        with torch.no_grad():
            self.weight[:, 0, kernel_size // 2] = 1.0
        if mode == "subject_residual":
            self.subject_residuals = nn.Parameter(
                torch.zeros(subject_count, channels, kernel_size)
            )
        else:
            self.register_parameter("subject_residuals", None)

    def forward(self, features, subject_indices=None):
        if features.ndim != 3 or features.shape[1] != self.channels:
            raise ValueError("temporal filter expects [batch, channels, time].")
        output = F.conv1d(
            features,
            self.weight,
            padding=self.kernel_size // 2,
            groups=self.channels,
        )
        if self.subject_residuals is None:
            return output
        if subject_indices is None:
            return output
        if subject_indices.ndim != 1 or len(subject_indices) != len(features):
            raise ValueError("subject_residual temporal filter requires one index per row.")
        subject_indices = subject_indices.long()
        if subject_indices.numel() and (
            subject_indices.min() < -1
            or subject_indices.max() >= len(self.subject_residuals)
        ):
            raise IndexError("subject index is outside the temporal filter range.")
        # A single convolution per subject keeps the FIR weights tied across rows.
        residual_output = torch.zeros_like(output)
        for subject in subject_indices.unique():
            if subject < 0:
                continue
            rows = (subject_indices == subject).nonzero(as_tuple=True)[0]
            residual = F.conv1d(
                features.index_select(0, rows),
                self.subject_residuals[subject].unsqueeze(1),
                padding=self.kernel_size // 2,
                groups=self.channels,
            )
            residual_output = residual_output.index_add(0, rows, residual)
        return output + residual_output


class DilatedConvEncoder(nn.Module):
    """由空间合并、残差扩张卷积和可选时间汇聚组成的信号编码器。"""

    def __init__(self, channel_positions, subject_count, output_dimension, config):
        super().__init__()
        virtual_channels = int(config.get("merger_channels", 270))
        position_dimension = int(config.get("merger_position_dimension", 2048))
        initial_channels = int(config.get("initial_linear", 512))
        branch_channels = int(config.get("branch_channels", initial_channels))
        hidden_channels = int(config.get("hidden", 160))
        depth = int(config.get("depth", 5))
        kernel_size = int(config.get("kernel_size", 3))
        dilation_growth = int(config.get("dilation_growth", 2))
        dilation_period = config.get("dilation_period", 5)
        dilation_period = None if dilation_period is None else int(dilation_period)
        dropout_input = float(config.get("dropout_input", 0.1))
        convolution_dropout = float(config.get("convolution_dropout", 0.0))
        use_batch_norm = bool(config.get("batch_norm", True))
        use_gelu = bool(config.get("gelu", True))
        use_skip = bool(config.get("skip", True))
        glu_every = int(config.get("glu_every", 2))
        glu_context = int(config.get("glu_context", 1))
        if depth <= 0:
            raise ValueError("depth must be positive.")

        self.channel_merger = SpatialChannelMerger(
            channel_positions,
            output_channels=virtual_channels,
            position_dimension=position_dimension,
            subject_count=subject_count,
            per_subject=bool(config.get("merger_per_subject", True)),
            residual_per_subject=bool(
                config.get("merger_residual_per_subject", False)
            ),
            dropout_radius=float(config.get("merger_dropout", 0.2)),
            frequency_start=0,
            fourier_margin=0.2,
            scale_fourier_coordinates=True,
            per_sample_dropout=bool(config.get("merger_per_sample_dropout", False)),
        )
        self.initial_projection = nn.Conv1d(virtual_channels, initial_channels, 1)
        self.subject_projection_enabled = bool(
            config.get("subject_projection_enabled", True)
        )
        self.subject_projection_per_subject = bool(
            config.get("subject_projection_per_subject", True)
        )
        self.subject_projection_residual_per_subject = bool(
            config.get("subject_projection_residual_per_subject", False)
        )
        if (
            self.subject_projection_per_subject
            and self.subject_projection_residual_per_subject
        ):
            raise ValueError(
                "subject projection cannot be both independent and residual."
            )
        if not self.subject_projection_enabled:
            if (
                self.subject_projection_per_subject
                or self.subject_projection_residual_per_subject
            ):
                raise ValueError(
                    "disabled subject projection cannot have subject-specific parameters."
                )
            if initial_channels != branch_channels:
                raise ValueError(
                    "removing subject projection requires initial_linear and "
                    "branch_channels to match."
                )
            self.subject_projection = None
        elif self.subject_projection_residual_per_subject:
            self.subject_projection = ResidualSubjectLinear(
                subject_count,
                initial_channels,
                output_channels=branch_channels,
                bias=False,
                initialization="normal",
            )
        else:
            self.subject_projection = SubjectLinear(
                subject_count if self.subject_projection_per_subject else 1,
                initial_channels,
                output_channels=branch_channels,
                bias=False,
                initialization="normal",
            )

        temporal_filter_config = config.get("temporal_filter")
        self.temporal_filter = (
            ExplicitTemporalFilter(
                branch_channels,
                subject_count,
                temporal_filter_config.get("kernel_size", 7),
                mode=temporal_filter_config.get("mode", "shared"),
            )
            if temporal_filter_config is not None
            else None
        )

        sizes = [branch_channels]
        sizes.extend([hidden_channels] * max(depth - 1, 0))
        sizes.append(int(output_dimension))
        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.residual = []
        dilation = 1
        activation_type = nn.GELU if use_gelu else nn.ReLU
        for index, (input_channels, next_channels) in enumerate(
            zip(sizes[:-1], sizes[1:])
        ):
            if dilation_period and index % dilation_period == 0:
                dilation = 1
            modules = []
            if index == 0 and dropout_input > 0:
                modules.append(nn.Dropout(dropout_input))
            modules.append(
                nn.Conv1d(
                    input_channels,
                    next_channels,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size // 2) * dilation,
                    dilation=dilation,
                )
            )
            dilation *= dilation_growth
            is_last = index == depth - 1
            if not is_last:
                if use_batch_norm:
                    modules.append(nn.BatchNorm1d(next_channels))
                modules.append(activation_type())
                if convolution_dropout > 0:
                    modules.append(nn.Dropout(convolution_dropout))
            self.blocks.append(nn.Sequential(*modules))
            self.residual.append(use_skip and input_channels == next_channels)
            if glu_every and (index + 1) % glu_every == 0:
                self.gates.append(
                    nn.Sequential(
                        nn.Conv1d(
                            next_channels,
                            2 * next_channels,
                            1 + 2 * glu_context,
                            padding=glu_context,
                        ),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.gates.append(nn.Identity())
        self.temporal_attention = (
            TemporalAttentionPool(
                output_dimension,
                hidden_dimension=int(config.get("temporal_attention_hidden", 256)),
            )
            if bool(config.get("temporal_pooling", True))
            else None
        )

    def encode_temporal(self, signals, subject_indices=None):
        """保留时间轴，返回候选条件读取所需的动态特征。"""
        features = self.channel_merger(signals, subject_indices)
        features = self.initial_projection(features)
        if self.subject_projection is not None:
            projection_subject_indices = (
                subject_indices
                if (
                    self.subject_projection_per_subject
                    or self.subject_projection_residual_per_subject
                )
                else None
            )
            features = self.subject_projection(features, projection_subject_indices)
        if self.temporal_filter is not None:
            features = self.temporal_filter(features, subject_indices)
        for block, gate, residual in zip(self.blocks, self.gates, self.residual):
            previous = features
            features = block(features)
            if residual:
                features = features + previous
            features = gate(features)
        return features

    def forward(self, signals, subject_indices=None):
        features = self.encode_temporal(signals, subject_indices)
        if self.temporal_attention is None:
            raise RuntimeError("当前 DilatedConvEncoder 配置未启用固定时间汇聚。")
        return self.temporal_attention(features)


class OrderedMultiBinAttentionPool(nn.Module):
    """有序多时域分箱槽内注意力汇聚：将 [B, C, T] 划分为 K 个有序时间槽并拼接为 [B, K*C]。"""

    def __init__(
        self,
        input_dimension: int,
        *,
        num_bins: int = 5,
        attention_hidden: int = 128,
    ):
        super().__init__()
        self.input_dimension = int(input_dimension)
        self.num_bins = int(num_bins)
        self.attention_hidden = int(attention_hidden)
        if self.input_dimension <= 0 or self.num_bins <= 0 or self.attention_hidden <= 0:
            raise ValueError(
                "OrderedMultiBinAttentionPool 的维度、分箱数与注意力隐维必须为正整数。"
            )
        self.output_dimension = self.num_bins * self.input_dimension
        self.key_proj = nn.Linear(self.input_dimension, self.attention_hidden)
        self.bin_score_weight = nn.Parameter(
            torch.empty(self.num_bins, self.attention_hidden)
        )
        nn.init.normal_(self.bin_score_weight, mean=0.0, std=0.02)
        self.bin_score_bias = nn.Parameter(torch.zeros(self.num_bins))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != self.input_dimension:
            raise ValueError(
                f"OrderedMultiBinAttentionPool 期望 [B, {self.input_dimension}, T]，收到 {tuple(features.shape)}"
            )
        bsz, channels, total_steps = features.shape
        if total_steps < self.num_bins:
            raise ValueError(
                f"输入时间步数 ({total_steps}) 小于分箱数 ({self.num_bins})。"
            )
        x = features.transpose(1, 2)  # [B, T, C]
        keys = torch.tanh(self.key_proj(x))  # [B, T, H_att]
        if total_steps % self.num_bins == 0:
            bin_len = total_steps // self.num_bins
            keys_bins = keys.view(bsz, self.num_bins, bin_len, self.attention_hidden)
            scores = (
                torch.einsum("bkth,kh->bkt", keys_bins, self.bin_score_weight)
                + self.bin_score_bias[None, :, None]
            )
            weights = torch.softmax(scores, dim=-1)
            x_bins = x.view(bsz, self.num_bins, bin_len, channels)
            pooled = torch.einsum("bkt,bktc->bkc", weights, x_bins)
            return pooled.reshape(bsz, self.output_dimension)

        x_chunks = torch.tensor_split(x, self.num_bins, dim=1)
        key_chunks = torch.tensor_split(keys, self.num_bins, dim=1)
        pooled_list: list[torch.Tensor] = []
        for bin_idx, (x_chunk, key_chunk) in enumerate(zip(x_chunks, key_chunks)):
            scores = (
                torch.matmul(key_chunk, self.bin_score_weight[bin_idx])
                + self.bin_score_bias[bin_idx]
            )
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            pooled_list.append((weights * x_chunk).sum(dim=1))
        return torch.cat(pooled_list, dim=-1)


class PreNormResMLPBlock(nn.Module):
    """Pre-LayerNorm 残差前馈块（仅用于池化后语义投影头，严禁在卷积主干内替代 BatchNorm1d）。"""

    def __init__(self, dimension: int, hidden_dimension: int, *, dropout: float = 0.15):
        super().__init__()
        self.norm = nn.LayerNorm(int(dimension))
        self.fc1 = nn.Linear(int(dimension), int(hidden_dimension))
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_dimension), int(dimension))
        self.drop2 = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop1(self.act(self.fc1(self.norm(x))))
        return x + self.drop2(self.fc2(h))


class MultiBinResConvEncoder(nn.Module):
    """Tier-2A 多时域分箱残差扩张卷积脑编码器（BatchNorm1d 卷积主干 + K 槽注意力池化 + Pre-Norm ResMLP 头）。

    保持与 DilatedConvEncoder 相同的 ``encode_temporal`` 与 ``forward`` 契约，并额外暴露
    ``encode_with_bins(signals, subject_indices) -> (z, h_bins)`` 供 Stage-2 多时域特征接口按需复用。
    """

    def __init__(self, channel_positions, subject_count, output_dimension, config):
        super().__init__()
        cfg = dict(config or {})
        virtual_channels = int(cfg.get("merger_channels", 270))
        position_dimension = int(cfg.get("merger_position_dimension", 2048))
        hidden_channels = int(cfg.get("hidden", 256))
        depth = int(cfg.get("depth", 6))
        kernel_size = int(cfg.get("kernel_size", 3))
        dilation_growth = int(cfg.get("dilation_growth", 2))
        dilation_period = cfg.get("dilation_period", 4)
        dilation_period = None if dilation_period is None else int(dilation_period)
        dropout_input = float(cfg.get("dropout_input", 0.05))
        convolution_dropout = float(cfg.get("convolution_dropout", 0.10))
        use_batch_norm = bool(cfg.get("batch_norm", True))
        use_gelu = bool(cfg.get("gelu", True))
        glu_every = int(cfg.get("glu_every", 2))
        glu_context = int(cfg.get("glu_context", 1))
        num_temporal_bins = int(cfg.get("num_temporal_bins", 5))
        bin_attention_hidden = int(cfg.get("bin_attention_hidden", 128))
        mlp_dim = int(cfg.get("mlp_dim", int(output_dimension)))
        mlp_hidden = int(cfg.get("mlp_hidden", 2 * mlp_dim))
        mlp_depth = int(cfg.get("mlp_depth", 3))
        mlp_dropout = float(cfg.get("mlp_dropout", 0.15))

        if depth <= 0:
            raise ValueError("MultiBinResConvEncoder depth 必须为正整数。")
        if hidden_channels <= 0 or mlp_dim <= 0 or mlp_hidden <= 0 or mlp_depth < 0:
            raise ValueError("MultiBinResConvEncoder 通道数与 MLP 维度配置无效。")

        self.gaussian_noise_std = max(0.0, float(cfg.get("gaussian_noise_std", 0.0)))
        self.channel_dropout = float(cfg.get("channel_dropout", 0.0))
        if not (0.0 <= self.channel_dropout < 1.0):
            raise ValueError("channel_dropout 必须位于 [0, 1) 之间。")

        self.channel_merger = SpatialChannelMerger(
            channel_positions,
            output_channels=virtual_channels,
            position_dimension=position_dimension,
            subject_count=subject_count,
            per_subject=bool(cfg.get("merger_per_subject", True)),
            residual_per_subject=bool(cfg.get("merger_residual_per_subject", False)),
            dropout_radius=float(cfg.get("merger_dropout", 0.2)),
            frequency_start=0,
            fourier_margin=0.2,
            scale_fourier_coordinates=True,
            per_sample_dropout=bool(cfg.get("merger_per_sample_dropout", False)),
        )
        self.initial_projection = nn.Conv1d(virtual_channels, hidden_channels, 1)
        self.subject_projection_enabled = bool(
            cfg.get("subject_projection_enabled", True)
        )
        self.subject_projection_per_subject = bool(
            cfg.get("subject_projection_per_subject", True)
        )
        self.subject_projection_residual_per_subject = bool(
            cfg.get("subject_projection_residual_per_subject", False)
        )
        if (
            self.subject_projection_per_subject
            and self.subject_projection_residual_per_subject
        ):
            raise ValueError(
                "subject projection cannot be both independent and residual."
            )
        if not self.subject_projection_enabled:
            if (
                self.subject_projection_per_subject
                or self.subject_projection_residual_per_subject
            ):
                raise ValueError(
                    "disabled subject projection cannot have subject-specific parameters."
                )
            self.subject_projection = None
        elif self.subject_projection_residual_per_subject:
            self.subject_projection = ResidualSubjectLinear(
                subject_count,
                hidden_channels,
                output_channels=hidden_channels,
                bias=False,
                initialization="normal",
            )
        else:
            self.subject_projection = SubjectLinear(
                subject_count if self.subject_projection_per_subject else 1,
                hidden_channels,
                output_channels=hidden_channels,
                bias=False,
                initialization="normal",
            )

        temporal_filter_config = cfg.get("temporal_filter")
        self.temporal_filter = (
            ExplicitTemporalFilter(
                hidden_channels,
                subject_count,
                temporal_filter_config.get("kernel_size", 7),
                mode=temporal_filter_config.get("mode", "shared"),
            )
            if temporal_filter_config is not None
            else None
        )

        self.blocks = nn.ModuleList()
        self.gates = nn.ModuleList()
        dilation = 1
        activation_type = nn.GELU if use_gelu else nn.ReLU
        for index in range(depth):
            if dilation_period and index % dilation_period == 0:
                dilation = 1
            modules: list[nn.Module] = []
            if index == 0 and dropout_input > 0.0:
                modules.append(nn.Dropout(dropout_input))
            modules.append(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size // 2) * dilation,
                    dilation=dilation,
                )
            )
            if use_batch_norm:
                modules.append(nn.BatchNorm1d(hidden_channels))
            modules.append(activation_type())
            if convolution_dropout > 0.0:
                modules.append(nn.Dropout(convolution_dropout))
            self.blocks.append(nn.Sequential(*modules))
            dilation *= dilation_growth
            if glu_every and (index + 1) % glu_every == 0:
                self.gates.append(
                    nn.Sequential(
                        nn.Conv1d(
                            hidden_channels,
                            2 * hidden_channels,
                            1 + 2 * glu_context,
                            padding=glu_context,
                        ),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.gates.append(nn.Identity())

        self.multi_bin_pool = OrderedMultiBinAttentionPool(
            hidden_channels,
            num_bins=num_temporal_bins,
            attention_hidden=bin_attention_hidden,
        )
        self.bin_feature_dimension = self.multi_bin_pool.output_dimension
        self.head_in = nn.Linear(self.bin_feature_dimension, mlp_dim)
        self.mlp_blocks = nn.ModuleList(
            [
                PreNormResMLPBlock(mlp_dim, mlp_hidden, dropout=mlp_dropout)
                for _ in range(mlp_depth)
            ]
        )
        self.head_norm = nn.LayerNorm(mlp_dim)
        self.head_out = nn.Linear(mlp_dim, int(output_dimension))

    def _apply_train_augmentation(self, signals: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return signals
        x = signals
        if self.channel_dropout > 0.0:
            keep_prob = 1.0 - self.channel_dropout
            mask = (
                torch.rand(
                    (x.shape[0], x.shape[1], 1),
                    device=x.device,
                    dtype=x.dtype,
                )
                < keep_prob
            ).to(dtype=x.dtype)
            x = x * (mask / keep_prob)
        if self.gaussian_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.gaussian_noise_std
        return x

    def encode_temporal(self, signals, subject_indices=None):
        """提取残差扩张卷积时序表征 [B, hidden, T]（零未来信号泄漏）。"""
        x = self._apply_train_augmentation(signals)
        features = self.channel_merger(x, subject_indices)
        features = self.initial_projection(features)
        if self.subject_projection is not None:
            projection_subject_indices = (
                subject_indices
                if (
                    self.subject_projection_per_subject
                    or self.subject_projection_residual_per_subject
                )
                else None
            )
            features = self.subject_projection(features, projection_subject_indices)
        if self.temporal_filter is not None:
            features = self.temporal_filter(features, subject_indices)
        for block, gate in zip(self.blocks, self.gates):
            features = features + block(features)
            features = gate(features)
        return features

    def encode_with_bins(
        self, signals, subject_indices=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """同时返回单窗投影嵌入 z [B, D] 与池化前有序多时域分箱特征 h_bins [B, K*H]。"""
        temporal = self.encode_temporal(signals, subject_indices)
        h_bins = self.multi_bin_pool(temporal)
        h = self.head_in(h_bins)
        for mlp_block in self.mlp_blocks:
            h = mlp_block(h)
        z = self.head_out(self.head_norm(h))
        return z, h_bins

    def forward(self, signals, subject_indices=None):
        z, _ = self.encode_with_bins(signals, subject_indices)
        return z


def build_brain_encoder(
    channel_positions,
    subject_count: int,
    output_dimension: int,
    config: dict | None = None,
) -> nn.Module:
    """根据 model.conv.architecture 构建 Stage-1 脑信号编码器（默认保持 DilatedConvEncoder 零改动）。"""
    cfg = dict(config or {})
    arch = str(cfg.get("architecture", "dilated_conv") or "dilated_conv").strip().lower()
    if arch in {"", "dilated_conv", "default", "legacy"}:
        return DilatedConvEncoder(
            channel_positions,
            subject_count=subject_count,
            output_dimension=output_dimension,
            config=cfg,
        )
    if arch in {"multibin_resconv", "multi_bin_resconv", "resconv_multibin"}:
        return MultiBinResConvEncoder(
            channel_positions,
            subject_count=subject_count,
            output_dimension=output_dimension,
            config=cfg,
        )
    raise ValueError(
        f"不支持的 model.conv.architecture={arch!r}，"
        "可选：'dilated_conv' | 'multibin_resconv'。"
    )
