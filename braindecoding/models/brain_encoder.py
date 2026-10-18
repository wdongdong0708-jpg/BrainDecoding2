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
