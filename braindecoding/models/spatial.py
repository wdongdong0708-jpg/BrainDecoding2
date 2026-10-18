"""空间传感器坐标投影、Fourier 位置编码、通道合并与受试者线性层。"""
from __future__ import annotations

import math

import torch
from torch import nn


def project_scalp_positions(channel_positions, margin=0.1):
    """将三维传感器坐标投影到归一化二维平面。"""
    positions = torch.as_tensor(channel_positions, dtype=torch.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("channel_positions must have shape [channels, 3].")
    if not torch.isfinite(positions).all():
        raise ValueError("channel_positions contains non-finite values.")
    if not 0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5).")

    unit = positions / positions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    radius = torch.acos(unit[:, 2].clamp(-1.0, 1.0))
    azimuth = torch.atan2(unit[:, 1], unit[:, 0])
    projected = torch.stack(
        [radius * torch.cos(azimuth), radius * torch.sin(azimuth)], dim=1
    )
    lower = projected.amin(dim=0, keepdim=True)
    span = (projected.amax(dim=0, keepdim=True) - lower).clamp_min(1e-8)
    return margin + (1.0 - 2.0 * margin) * (projected - lower) / span


class FourierPositionEmbedding(nn.Module):
    """使用傅里叶基编码归一化二维传感器位置。"""

    def __init__(
        self,
        dimension=2048,
        frequency_start=0,
        margin=0.2,
        scale_coordinates=True,
    ):
        super().__init__()
        dimension = int(dimension)
        frequency_count = int(round((dimension // 2) ** 0.5))
        if frequency_count * frequency_count * 2 != dimension:
            raise ValueError("dimension must be twice a square number.")
        if int(frequency_start) < 0:
            raise ValueError("frequency_start must be non-negative.")
        if float(margin) < 0:
            raise ValueError("margin must be non-negative.")
        self.dimension = dimension
        self.frequency_count = frequency_count
        self.frequency_start = int(frequency_start)
        self.margin = float(margin)
        self.scale_coordinates = bool(scale_coordinates)

    def forward(self, positions):
        if positions.shape[-1] != 2:
            raise ValueError("positions must contain two-dimensional coordinates.")
        coordinates = positions
        if self.scale_coordinates:
            coordinates = (coordinates + self.margin) / (1.0 + 2.0 * self.margin)
        frequencies = torch.arange(
            self.frequency_start,
            self.frequency_start + self.frequency_count,
            device=positions.device,
            dtype=positions.dtype,
        )
        frequency_x, frequency_y = torch.meshgrid(
            frequencies, frequencies, indexing="ij"
        )
        phase = 2.0 * math.pi * (
            coordinates[..., :1] * frequency_x.flatten()
            + coordinates[..., 1:] * frequency_y.flatten()
        )
        return torch.cat([phase.cos(), phase.sin()], dim=-1)


class SpatialChannelMerger(nn.Module):
    """学习从传感器到虚拟通道的傅里叶位置注意力映射。

    同一组件既支持全局注意力映射，也支持每名被试各用一张映射。
    传感器位置可以是归一化二维坐标，也可以是三维头皮坐标。
    """

    def __init__(
        self,
        channel_positions,
        output_channels=64,
        position_dimension=128,
        subject_count=1,
        per_subject=False,
        residual_per_subject=False,
        dropout_radius=0.0,
        projection_margin=0.1,
        frequency_start=1,
        fourier_margin=0.0,
        scale_fourier_coordinates=False,
        initialization_std=None,
    ):
        super().__init__()
        positions = torch.as_tensor(channel_positions, dtype=torch.float32)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError("channel_positions must have shape [channels, 2 or 3].")
        if positions.shape[1] == 3:
            positions = project_scalp_positions(positions, margin=projection_margin)
        elif not torch.isfinite(positions).all():
            raise ValueError("channel_positions contains non-finite values.")
        if int(output_channels) <= 0 or int(subject_count) <= 0:
            raise ValueError("output_channels and subject_count must be positive.")
        if float(dropout_radius) < 0:
            raise ValueError("dropout_radius must be non-negative.")

        self.register_buffer("channel_positions", positions, persistent=False)
        embedding = FourierPositionEmbedding(
            dimension=position_dimension,
            frequency_start=frequency_start,
            margin=fourier_margin,
            scale_coordinates=scale_fourier_coordinates,
        )
        self.position_embedding = embedding
        self.register_buffer("basis", embedding(positions))
        self.per_subject = bool(per_subject)
        self.residual_per_subject = bool(residual_per_subject)
        if self.per_subject and self.residual_per_subject:
            raise ValueError(
                "per_subject and residual_per_subject are mutually exclusive."
            )
        self.dropout_radius = float(dropout_radius)

        coefficient_shape = (int(output_channels), int(position_dimension))
        if self.per_subject:
            coefficient_shape = (int(subject_count), *coefficient_shape)
        self.coefficients = nn.Parameter(torch.empty(coefficient_shape))
        standard_deviation = (
            1.0 / math.sqrt(int(position_dimension))
            if initialization_std is None
            else float(initialization_std)
        )
        nn.init.normal_(self.coefficients, mean=0.0, std=standard_deviation)
        if self.residual_per_subject:
            self.subject_residuals = nn.Parameter(
                torch.zeros(
                    int(subject_count),
                    int(output_channels),
                    int(position_dimension),
                )
            )
        else:
            self.register_parameter("subject_residuals", None)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """兼容组件通用化前保存的 ``heads/channel_positions`` 权重键。"""
        legacy_coefficients_key = prefix + "heads"
        coefficients_key = prefix + "coefficients"
        legacy_positions_key = prefix + "channel_positions"
        basis_key = prefix + "basis"

        if (
            legacy_coefficients_key in state_dict
            and coefficients_key not in state_dict
        ):
            state_dict[coefficients_key] = state_dict.pop(legacy_coefficients_key)

        if legacy_positions_key in state_dict and basis_key not in state_dict:
            legacy_positions = state_dict.pop(legacy_positions_key)
            current_positions = self.channel_positions.to(
                device=legacy_positions.device,
                dtype=legacy_positions.dtype,
            )
            if (
                legacy_positions.shape != current_positions.shape
                or not torch.allclose(legacy_positions, current_positions)
            ):
                error_msgs.append(
                    f"{legacy_positions_key} does not match the current dataset "
                    "channel positions."
                )
            if (
                self.position_embedding.frequency_start != 0
                or not self.position_embedding.scale_coordinates
            ):
                error_msgs.append(
                    f"{legacy_positions_key} uses an unsupported legacy Fourier "
                    "configuration."
                )
            frequency_count = self.position_embedding.frequency_count
            frequencies_y = torch.arange(
                frequency_count,
                device=legacy_positions.device,
                dtype=legacy_positions.dtype,
            )
            frequencies_x = frequencies_y[:, None]
            margin = self.position_embedding.margin
            width = 1.0 + 2.0 * margin
            shifted = legacy_positions + margin
            phase_x = 2.0 * math.pi * frequencies_x / width
            phase_y = 2.0 * math.pi * frequencies_y / width
            shifted = shifted[..., None, None, :]
            phase = (
                shifted[..., 0] * phase_x + shifted[..., 1] * phase_y
            ).reshape(*legacy_positions.shape[:-1], -1)
            state_dict[basis_key] = torch.cat(
                [phase.cos(), phase.sin()], dim=-1
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, signals, subject_indices=None):
        batch_size, channel_count, _ = signals.shape
        if channel_count != len(self.channel_positions):
            raise ValueError("Signal channels and channel_positions do not match.")

        if self.per_subject:
            if subject_indices is None:
                subject_indices = torch.zeros(
                    batch_size, device=signals.device, dtype=torch.long
                )
            subject_indices = subject_indices.long()
            if len(subject_indices) != batch_size:
                raise ValueError("subject_indices and batch size do not match.")
            if subject_indices.min() < 0 or subject_indices.max() >= len(
                self.coefficients
            ):
                raise IndexError("subject index is outside the spatial merger range.")
            coefficients = self.coefficients[subject_indices]
            scores = torch.einsum("bod,cd->boc", coefficients, self.basis)
        elif self.residual_per_subject:
            scores = torch.einsum("od,cd->oc", self.coefficients, self.basis)
            scores = scores.unsqueeze(0).expand(batch_size, -1, -1).clone()
            if subject_indices is not None:
                subject_indices = subject_indices.long()
                if len(subject_indices) != batch_size:
                    raise ValueError("subject_indices and batch size do not match.")
                if subject_indices.min() < -1 or subject_indices.max() >= len(
                    self.subject_residuals
                ):
                    raise IndexError(
                        "subject index is outside the residual spatial merger range."
                    )
                known = subject_indices >= 0
                if known.any():
                    residual_scores = torch.einsum(
                        "bod,cd->boc",
                        self.subject_residuals[subject_indices[known]],
                        self.basis,
                    )
                    scores[known] = scores[known] + residual_scores
        else:
            scores = torch.einsum("od,cd->oc", self.coefficients, self.basis)
            scores = scores.unsqueeze(0).expand(batch_size, -1, -1)

        if self.training and self.dropout_radius > 0:
            center = torch.rand(2, device=signals.device, dtype=signals.dtype)
            banned = (
                self.channel_positions.to(dtype=signals.dtype) - center
            ).norm(dim=-1) <= self.dropout_radius
            scores = scores.masked_fill(banned[None, None], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bct,boc->bot", signals, weights)


class SubjectLinear(nn.Module):
    """为每名已知被试应用独立的线性通道变换。"""

    def __init__(
        self,
        subject_count,
        input_channels,
        output_channels=None,
        bias=True,
        initialization="identity",
    ):
        super().__init__()
        subject_count = int(subject_count)
        input_channels = int(input_channels)
        output_channels = (
            input_channels if output_channels is None else int(output_channels)
        )
        if subject_count <= 0 or input_channels <= 0 or output_channels <= 0:
            raise ValueError("subject_count and channel dimensions must be positive.")
        self.weights = nn.Parameter(
            torch.empty(subject_count, output_channels, input_channels)
        )
        if initialization == "identity":
            if input_channels != output_channels:
                raise ValueError("identity initialization requires equal dimensions.")
            with torch.no_grad():
                self.weights.copy_(torch.eye(input_channels).repeat(subject_count, 1, 1))
        elif initialization == "normal":
            nn.init.normal_(
                self.weights, mean=0.0, std=1.0 / math.sqrt(input_channels)
            )
        else:
            raise ValueError("initialization must be 'identity' or 'normal'.")
        if bias:
            self.bias = nn.Parameter(torch.zeros(subject_count, output_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, features, subject_indices=None):
        if subject_indices is None:
            subject_indices = torch.zeros(
                len(features), device=features.device, dtype=torch.long
            )
        subject_indices = subject_indices.long()
        if len(subject_indices) != len(features):
            raise ValueError("subject_indices and batch size do not match.")
        if subject_indices.min() < 0 or subject_indices.max() >= len(self.weights):
            raise IndexError("subject index is outside the subject layer range.")
        output = torch.bmm(self.weights[subject_indices], features)
        if self.bias is not None:
            output = output + self.bias[subject_indices].unsqueeze(-1)
        return output


class ResidualSubjectLinear(nn.Module):
    """共享线性变换加零初始化的受试者残差。

    ``subject_indices == -1``（或 ``None``）只使用共享参数，作为完全未知
    受试者的明确零样本路径。非负索引使用共享参数与对应个人残差之和。
    """

    def __init__(
        self,
        subject_count,
        input_channels,
        output_channels=None,
        bias=True,
        initialization="identity",
    ):
        super().__init__()
        subject_count = int(subject_count)
        input_channels = int(input_channels)
        output_channels = (
            input_channels if output_channels is None else int(output_channels)
        )
        if subject_count <= 0 or input_channels <= 0 or output_channels <= 0:
            raise ValueError("subject_count and channel dimensions must be positive.")
        self.weight = nn.Parameter(torch.empty(output_channels, input_channels))
        if initialization == "identity":
            if input_channels != output_channels:
                raise ValueError("identity initialization requires equal dimensions.")
            with torch.no_grad():
                self.weight.copy_(torch.eye(input_channels))
        elif initialization == "normal":
            nn.init.normal_(
                self.weight, mean=0.0, std=1.0 / math.sqrt(input_channels)
            )
        else:
            raise ValueError("initialization must be 'identity' or 'normal'.")
        self.subject_residuals = nn.Parameter(
            torch.zeros(subject_count, output_channels, input_channels)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_channels))
            self.subject_bias_residuals = nn.Parameter(
                torch.zeros(subject_count, output_channels)
            )
        else:
            self.register_parameter("bias", None)
            self.register_parameter("subject_bias_residuals", None)

    def forward(self, features, subject_indices=None):
        output = torch.einsum("oi,bit->bot", self.weight, features)
        if self.bias is not None:
            output = output + self.bias[None, :, None]
        if subject_indices is None:
            return output
        subject_indices = subject_indices.long()
        if len(subject_indices) != len(features):
            raise ValueError("subject_indices and batch size do not match.")
        if subject_indices.min() < -1 or subject_indices.max() >= len(
            self.subject_residuals
        ):
            raise IndexError("subject index is outside the residual subject layer range.")
        known = subject_indices >= 0
        if known.any():
            residual = torch.bmm(
                self.subject_residuals[subject_indices[known]], features[known]
            )
            if self.subject_bias_residuals is not None:
                residual = residual + self.subject_bias_residuals[
                    subject_indices[known]
                ].unsqueeze(-1)
            output[known] = output[known] + residual
        return output
