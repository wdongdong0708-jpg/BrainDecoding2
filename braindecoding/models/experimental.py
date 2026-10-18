"""三状态/五状态局部轨迹残差与候选条件证据读取等探索性模型变体。"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from braindecoding.models.brain_encoder import DilatedConvEncoder
from braindecoding.models.embedding import (
    BrainEmbeddingModel,
    build_brain_embedding_model,
)


class CandidateConditionedEvidenceReadout(nn.Module):
    """把时间特征与候选文本转换成可正可负的逐时证据。"""

    def __init__(self, feature_dimension, candidate_dimension, evidence_dimension=128):
        super().__init__()
        feature_dimension = int(feature_dimension)
        candidate_dimension = int(candidate_dimension)
        evidence_dimension = int(evidence_dimension)
        if min(feature_dimension, candidate_dimension, evidence_dimension) <= 0:
            raise ValueError("候选证据读取器的维度必须为正数。")
        self.feature_dimension = feature_dimension
        self.candidate_dimension = candidate_dimension
        self.evidence_dimension = evidence_dimension
        # 两条投影都不使用偏置，避免产生候选词独立得分分支。
        self.feature_projection = nn.Linear(
            feature_dimension, evidence_dimension, bias=False
        )
        self.candidate_projection = nn.Linear(
            candidate_dimension, evidence_dimension, bias=False
        )
        self.scale = evidence_dimension**-0.5

    def signed_evidence(self, temporal_features, candidate_embeddings):
        """返回尚未进行时间汇聚的有界正负证据。"""
        if temporal_features.ndim != 3:
            raise ValueError("temporal_features 必须为 batch×feature×time 张量。")
        if candidate_embeddings.ndim != 2:
            raise ValueError("candidate_embeddings 必须为 candidate×feature 张量。")
        if temporal_features.shape[1] != self.feature_dimension:
            raise ValueError("temporal_features 的特征维与读取器配置不一致。")
        if candidate_embeddings.shape[1] != self.candidate_dimension:
            raise ValueError("candidate_embeddings 的特征维与读取器配置不一致。")

        candidates = F.normalize(candidate_embeddings, dim=-1, eps=1e-15)
        keys = self.feature_projection(temporal_features.transpose(1, 2))
        queries = self.candidate_projection(candidates)
        alignment = torch.einsum("btr,cr->bct", keys, queries) * self.scale
        # 非线性必须发生在时间累积之前，否则会退化成固定汇聚后再匹配。
        return torch.tanh(alignment)

    def forward(self, temporal_features, candidate_embeddings, return_evidence=False):
        evidence = self.signed_evidence(
            temporal_features, candidate_embeddings
        ) / temporal_features.shape[-1]
        scores = evidence.sum(dim=-1)
        if return_evidence:
            return scores, evidence
        return scores


class CandidateConditionedAttentionEvidenceReadout(
    CandidateConditionedEvidenceReadout
):
    """用独立候选注意力选择逐时有符号证据，并从均匀汇聚开始。"""

    def __init__(
        self,
        feature_dimension,
        candidate_dimension,
        evidence_dimension=128,
        attention_dimension=128,
        attention_gate_init=0.0,
    ):
        super().__init__(
            feature_dimension=feature_dimension,
            candidate_dimension=candidate_dimension,
            evidence_dimension=evidence_dimension,
        )
        attention_dimension = int(attention_dimension)
        if attention_dimension <= 0:
            raise ValueError("候选注意力维度必须为正数。")
        if float(attention_gate_init) != 0.0:
            raise ValueError("候选注意力门必须从 0 初始化。")
        # 注意力与有符号证据使用独立投影；两者都不引入候选词独立偏置。
        self.attention_dimension = attention_dimension
        # 新增投影不推进全局随机流，保证同 seed 的 V0/attention 配对训练
        # 具有相同主干、证据投影和后续 dropout 随机序列。
        with torch.random.fork_rng(devices=[]):
            self.attention_feature_projection = nn.Linear(
                self.feature_dimension, attention_dimension, bias=False
            )
            self.attention_candidate_projection = nn.Linear(
                self.candidate_dimension, attention_dimension, bias=False
            )
        self.attention_scale = attention_dimension**-0.5
        self.attention_gate_logit = nn.Parameter(
            torch.tensor(float(attention_gate_init))
        )

    @property
    def attention_gate(self):
        """零值给出严格均匀注意力，训练后有界地开放候选选择。"""
        return torch.tanh(self.attention_gate_logit)

    def forward(
        self,
        temporal_features,
        candidate_embeddings,
        return_evidence=False,
        return_attention=False,
    ):
        signed_evidence = self.signed_evidence(
            temporal_features, candidate_embeddings
        )

        candidates = F.normalize(candidate_embeddings, dim=-1, eps=1e-15)
        attention_keys = self.attention_feature_projection(
            temporal_features.transpose(1, 2)
        )
        attention_queries = self.attention_candidate_projection(candidates)
        attention_logits = torch.einsum(
            "btr,cr->bct", attention_keys, attention_queries
        ) * self.attention_scale
        attention_weights = torch.softmax(
            self.attention_gate * attention_logits, dim=-1
        )
        contributions = attention_weights * signed_evidence
        scores = contributions.sum(dim=-1)

        if return_attention:
            return scores, contributions, attention_weights, signed_evidence
        if return_evidence:
            return scores, contributions
        return scores


class ThreeStateTrajectoryReadout(nn.Module):
    """在同一句内用 i-1、i、i+1 构造一个零初始化局部残差。"""

    MODES = {"trajectory", "concat_control", "conditional_trajectory"}

    def __init__(self, dimension, config, mode):
        super().__init__()
        self.dimension = int(dimension)
        self.mode = str(mode)
        if self.mode not in self.MODES:
            raise ValueError(f"未知三状态读取模式：{self.mode}")
        dropout = float(config.get("dropout", 0.1))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("state_3.dropout 必须位于 [0, 1)。")
        initial_gate = float(config.get("residual_gate_init", 0.0))
        if initial_gate != 0.0:
            raise ValueError("三状态残差门必须从 0 初始化。")

        self.condition_network = None
        self.center_projection = None
        self.incoming_projection = None
        self.outgoing_projection = None
        if self.mode == "trajectory":
            projection_dimension = int(config.get("projection_dimension", 256))
            hidden_dimension = int(
                config.get("trajectory_hidden_dimension", 512)
            )
            if projection_dimension <= 0 or hidden_dimension <= 0:
                raise ValueError("三状态轨迹投影与隐藏维度必须为正数。")
            self.state_projection = nn.Linear(
                self.dimension, projection_dimension
            )
            allowed_components = {
                "state",
                "incoming",
                "outgoing",
                "curvature",
            }
            requested_components = tuple(
                str(value)
                for value in config.get(
                    "trajectory_components",
                    ("state", "incoming", "outgoing", "curvature"),
                )
            )
            if (
                not requested_components
                or len(requested_components) != len(set(requested_components))
                or not set(requested_components).issubset(allowed_components)
                or "state" not in requested_components
            ):
                raise ValueError(
                    "state_3.trajectory_components 必须是不重复的有效分量，"
                    "并包含 state。"
                )
            if "curvature" in requested_components and not {
                "incoming",
                "outgoing",
            }.issubset(requested_components):
                raise ValueError(
                    "保留 curvature 时必须同时保留 incoming/outgoing，"
                    "否则曲率会泄漏被消融的方向。"
                )
            self.active_components = requested_components
            input_dimension = 4 * projection_dimension
        elif self.mode == "conditional_trajectory":
            projection_dimension = int(config.get("projection_dimension", 256))
            hidden_dimension = int(
                config.get("trajectory_hidden_dimension", 512)
            )
            condition_hidden_dimension = int(
                config.get("condition_hidden_dimension", projection_dimension)
            )
            if min(
                projection_dimension,
                hidden_dimension,
                condition_hidden_dimension,
            ) <= 0:
                raise ValueError(
                    "三状态条件轨迹投影、条件与隐藏维度必须为正数。"
                )
            self.state_projection = None
            self.center_projection = nn.Linear(
                self.dimension, projection_dimension
            )
            self.incoming_projection = nn.Linear(
                self.dimension, projection_dimension
            )
            self.outgoing_projection = nn.Linear(
                self.dimension, projection_dimension
            )
            self.condition_network = nn.Sequential(
                nn.Linear(2 * projection_dimension, condition_hidden_dimension),
                nn.GELU(),
                nn.Linear(condition_hidden_dimension, projection_dimension),
            )
            nn.init.zeros_(self.condition_network[-1].weight)
            nn.init.zeros_(self.condition_network[-1].bias)
            self.active_components = (
                "state",
                "incoming",
                "outgoing",
                "conditioned_outgoing",
            )
            input_dimension = 4 * projection_dimension
        else:
            hidden_dimension = int(config.get("concat_hidden_dimension", 320))
            if hidden_dimension <= 0:
                raise ValueError("三状态拼接对照隐藏维度必须为正数。")
            self.state_projection = None
            self.active_components = ("state", "incoming", "outgoing")
            input_dimension = 3 * self.dimension

        self.joint_network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, self.dimension),
        )
        self.residual_gate_logit = nn.Parameter(torch.tensor(initial_gate))

    @property
    def residual_gate(self):
        return torch.tanh(self.residual_gate_logit)

    @staticmethod
    def triplet_indices(group_indices):
        """返回严格位于每个组内部的前驱、中心与后继行号。"""
        if group_indices is None:
            raise ValueError("三状态读取需要 group_indices。")
        if group_indices.ndim != 1:
            raise ValueError("group_indices 必须是一维张量。")
        groups = {}
        for row_index, group in enumerate(
            group_indices.detach().cpu().tolist()
        ):
            groups.setdefault(int(group), []).append(row_index)
        previous = []
        center = []
        following = []
        for indices in groups.values():
            if len(indices) < 3:
                continue
            previous.extend(indices[:-2])
            center.extend(indices[1:-1])
            following.extend(indices[2:])
        device = group_indices.device
        return tuple(
            torch.as_tensor(values, device=device, dtype=torch.long)
            for values in (previous, center, following)
        )

    def trajectory_coordinates(self, previous, center, following):
        """显式保留来向、去向与二阶转向量的不同槽位。"""
        if self.state_projection is None:
            raise ValueError("普通拼接对照不定义显式轨迹坐标。")
        h_previous = self.state_projection(previous)
        h_center = self.state_projection(center)
        h_following = self.state_projection(following)
        incoming = h_center - h_previous
        outgoing = h_following - h_center
        curvature = outgoing - incoming
        components = {
            "state": h_center,
            "incoming": incoming,
            "outgoing": outgoing,
            "curvature": curvature,
        }
        return torch.cat(
            tuple(
                components[name]
                if name in self.active_components
                else torch.zeros_like(components[name])
                for name in ("state", "incoming", "outgoing", "curvature")
            ),
            dim=-1,
        )

    def conditional_coordinates(
        self,
        previous,
        center,
        following,
        return_condition=False,
    ):
        """用当前状态与来向变化逐维调制去向变化。"""
        if self.mode != "conditional_trajectory":
            raise ValueError("只有条件轨迹模式定义条件坐标。")
        state = self.center_projection(center)
        incoming = self.incoming_projection(center - previous)
        outgoing = self.outgoing_projection(following - center)
        condition = self.condition_network(torch.cat((state, incoming), dim=-1))
        conditioned_outgoing = outgoing * (1.0 + condition)
        coordinates = torch.cat(
            (state, incoming, outgoing, conditioned_outgoing), dim=-1
        )
        if return_condition:
            return coordinates, condition
        return coordinates

    def forward(self, features, group_indices, return_internal_mask=False):
        if features.ndim != 2 or features.shape[-1] != self.dimension:
            raise ValueError(
                "三状态读取输入必须为 batch×dimension，"
                f"实际为 {tuple(features.shape)}。"
            )
        previous_indices, center_indices, following_indices = (
            self.triplet_indices(group_indices)
        )
        internal_mask = torch.zeros(
            len(features), device=features.device, dtype=torch.bool
        )
        if len(center_indices) == 0:
            if return_internal_mask:
                return features, internal_mask
            return features

        previous = features[previous_indices]
        center = features[center_indices]
        following = features[following_indices]
        if self.mode == "trajectory":
            joint_input = self.trajectory_coordinates(
                previous, center, following
            )
        elif self.mode == "conditional_trajectory":
            joint_input = self.conditional_coordinates(
                previous, center, following
            )
        else:
            joint_input = torch.cat((previous, center, following), dim=-1)
        trajectory = self.joint_network(joint_input)
        updated_center = F.normalize(
            center + self.residual_gate * trajectory, dim=-1
        )
        output = features.clone()
        output[center_indices] = updated_center
        internal_mask[center_indices] = True
        if return_internal_mask:
            return output, internal_mask
        return output


class ThreeStateBrainEmbeddingModel(BrainEmbeddingModel):
    """CNN 主路加一个局部、分组且有方向的三状态读取分支。"""

    def __init__(self, channel_positions, subject_count, config, mode):
        if bool(config.get("use_transformer", False)):
            raise ValueError("三状态模型不能同时启用句级 Transformer。")
        if str(config.get("context_mode", "grouped")) != "grouped":
            raise ValueError("三状态模型只支持 grouped context。")
        super().__init__(channel_positions, subject_count, config)
        output_dimension = int(config.get("embedding_dimension", 1024))
        # 两个配对分支的参数形状不同；隔离初始化对全局随机流的推进，避免
        # 后续 CNN/dropout 仅因建模方式不同而失去同 seed 可比性。
        with torch.random.fork_rng(devices=[]):
            self.state_3_readout = ThreeStateTrajectoryReadout(
                output_dimension,
                config.get("state_3", {}),
                mode=mode,
            )
        self.requires_complete_groups = True
        self.local_context_radius = 1

    def forward(
        self,
        signals,
        subject_indices=None,
        group_indices=None,
        return_brain_embedding=False,
    ):
        brain_embedding = F.normalize(
            self.brain_encoder(signals, subject_indices), dim=-1
        )
        output = self.state_3_readout(brain_embedding, group_indices)
        if return_brain_embedding:
            return output, brain_embedding
        return output


class FiveStateTrajectoryReadout(nn.Module):
    """用 i±1 与 i±2 构造参数严格匹配的五状态局部残差。"""

    MODES = {"multiscale_trajectory", "concat_control"}

    def __init__(self, dimension, config, mode):
        super().__init__()
        self.dimension = int(dimension)
        self.mode = str(mode)
        if self.mode not in self.MODES:
            raise ValueError(f"未知五状态读取模式：{self.mode}")
        projection_dimension = int(config.get("projection_dimension", 256))
        hidden_dimension = int(config.get("hidden_dimension", 512))
        dropout = float(config.get("dropout", 0.1))
        initial_gate = float(config.get("residual_gate_init", 0.0))
        if projection_dimension <= 0 or hidden_dimension <= 0:
            raise ValueError("五状态投影与隐藏维度必须为正数。")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("state_5.dropout 必须位于 [0, 1)。")
        if initial_gate != 0.0:
            raise ValueError("五状态残差门必须从 0 初始化。")

        self.state_projection = nn.Linear(
            self.dimension, projection_dimension
        )
        self.joint_network = nn.Sequential(
            nn.Linear(5 * projection_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, self.dimension),
        )
        self.residual_gate_logit = nn.Parameter(torch.tensor(initial_gate))

    @property
    def residual_gate(self):
        return torch.tanh(self.residual_gate_logit)

    @staticmethod
    def window_indices(group_indices):
        """返回每个组内 i-2、i-1、i、i+1、i+2 的配对行号。"""
        if group_indices is None:
            raise ValueError("五状态读取需要 group_indices。")
        if group_indices.ndim != 1:
            raise ValueError("group_indices 必须是一维张量。")
        groups = {}
        for row_index, group in enumerate(
            group_indices.detach().cpu().tolist()
        ):
            groups.setdefault(int(group), []).append(row_index)
        far_previous = []
        previous = []
        center = []
        following = []
        far_following = []
        for indices in groups.values():
            if len(indices) < 5:
                continue
            far_previous.extend(indices[:-4])
            previous.extend(indices[1:-3])
            center.extend(indices[2:-2])
            following.extend(indices[3:-1])
            far_following.extend(indices[4:])
        device = group_indices.device
        return tuple(
            torch.as_tensor(values, device=device, dtype=torch.long)
            for values in (
                far_previous,
                previous,
                center,
                following,
                far_following,
            )
        )

    def joint_coordinates(
        self,
        far_previous,
        previous,
        center,
        following,
        far_following,
    ):
        """产生多尺度速度坐标或信息等价的普通五状态坐标。"""
        h_far_previous = self.state_projection(far_previous)
        h_previous = self.state_projection(previous)
        h_center = self.state_projection(center)
        h_following = self.state_projection(following)
        h_far_following = self.state_projection(far_following)
        if self.mode == "concat_control":
            return torch.cat(
                (
                    h_far_previous,
                    h_previous,
                    h_center,
                    h_following,
                    h_far_following,
                ),
                dim=-1,
            )
        short_incoming = h_center - h_previous
        short_outgoing = h_following - h_center
        medium_incoming = (h_center - h_far_previous) / 2.0
        medium_outgoing = (h_far_following - h_center) / 2.0
        return torch.cat(
            (
                h_center,
                short_incoming,
                short_outgoing,
                medium_incoming,
                medium_outgoing,
            ),
            dim=-1,
        )

    def forward(self, features, group_indices, return_internal_mask=False):
        if features.ndim != 2 or features.shape[-1] != self.dimension:
            raise ValueError(
                "五状态读取输入必须为 batch×dimension，"
                f"实际为 {tuple(features.shape)}。"
            )
        indices = self.window_indices(group_indices)
        center_indices = indices[2]
        internal_mask = torch.zeros(
            len(features), device=features.device, dtype=torch.bool
        )
        if len(center_indices) == 0:
            if return_internal_mask:
                return features, internal_mask
            return features
        states = tuple(features[index] for index in indices)
        joint_input = self.joint_coordinates(*states)
        trajectory = self.joint_network(joint_input)
        updated_center = F.normalize(
            states[2] + self.residual_gate * trajectory, dim=-1
        )
        output = features.clone()
        output[center_indices] = updated_center
        internal_mask[center_indices] = True
        if return_internal_mask:
            return output, internal_mask
        return output


class FiveStateBrainEmbeddingModel(BrainEmbeddingModel):
    """CNN 主路加一个半径为 2 的五状态局部轨迹分支。"""

    def __init__(self, channel_positions, subject_count, config, mode):
        if bool(config.get("use_transformer", False)):
            raise ValueError("五状态模型不能同时启用句级 Transformer。")
        if str(config.get("context_mode", "grouped")) != "grouped":
            raise ValueError("五状态模型只支持 grouped context。")
        super().__init__(channel_positions, subject_count, config)
        output_dimension = int(config.get("embedding_dimension", 1024))
        # 两个分支形状完全相同；隔离初始化仍可锁住后续训练随机流。
        with torch.random.fork_rng(devices=[]):
            self.state_5_readout = FiveStateTrajectoryReadout(
                output_dimension,
                config.get("state_5", {}),
                mode=mode,
            )
        self.requires_complete_groups = True
        self.local_context_radius = 2

    def forward(
        self,
        signals,
        subject_indices=None,
        group_indices=None,
        return_brain_embedding=False,
    ):
        brain_embedding = F.normalize(
            self.brain_encoder(signals, subject_indices), dim=-1
        )
        output = self.state_5_readout(brain_embedding, group_indices)
        if return_brain_embedding:
            return output, brain_embedding
        return output


class CandidateEvidenceModel(nn.Module):
    """用每个候选词分别读取同一段动态脑特征。"""

    candidate_conditioned = True

    def __init__(self, channel_positions, subject_count, config):
        super().__init__()
        if bool(config.get("use_transformer", False)):
            raise ValueError("候选证据 V0 不允许启用句级 Transformer。")
        output_dimension = int(config.get("embedding_dimension", 1024))
        convolution_config = dict(config.get("conv", config))
        convolution_config["temporal_pooling"] = False
        self.brain_encoder = DilatedConvEncoder(
            channel_positions,
            subject_count=subject_count,
            output_dimension=output_dimension,
            config=convolution_config,
        )
        evidence_config = config.get("candidate_evidence", {})
        readout = str(config.get("readout", "candidate_evidence"))
        self.prefix_supervised = readout == "candidate_evidence_competition_soft"
        readout_arguments = {
            "feature_dimension": output_dimension,
            "candidate_dimension": output_dimension,
            "evidence_dimension": int(
                evidence_config.get("evidence_dimension", 128)
            ),
        }
        if readout in {
            "candidate_evidence",
            "candidate_evidence_competition_soft",
        }:
            self.evidence_readout = CandidateConditionedEvidenceReadout(
                **readout_arguments
            )
        elif readout == "candidate_evidence_attention":
            self.evidence_readout = CandidateConditionedAttentionEvidenceReadout(
                **readout_arguments,
                attention_dimension=int(
                    evidence_config.get("attention_dimension", 128)
                ),
                attention_gate_init=float(
                    evidence_config.get("attention_gate_init", 0.0)
                ),
            )
        else:
            raise ValueError(f"候选证据模型不支持读取机制：{readout}")
        if self.prefix_supervised:
            prefix_samples = tuple(
                int(value)
                for value in evidence_config.get("prefix_samples", ())
            )
            state_normalizer_samples = int(
                evidence_config.get("state_normalizer_samples", 0)
            )
            prefix_loss_weights = tuple(
                float(value)
                for value in evidence_config.get("prefix_loss_weights", ())
            )
            if (
                not prefix_samples
                or any(value <= 0 for value in prefix_samples)
                or tuple(sorted(set(prefix_samples))) != prefix_samples
            ):
                raise ValueError("软竞争候选状态前缀必须严格递增且为正数。")
            if (
                state_normalizer_samples <= 0
                or prefix_samples[-1] != state_normalizer_samples
            ):
                raise ValueError("最后一个候选状态前缀必须等于状态归一化长度。")
            if (
                len(prefix_loss_weights) != len(prefix_samples)
                or any(value < 0 for value in prefix_loss_weights)
                or sum(prefix_loss_weights) <= 0
            ):
                raise ValueError("每个候选状态前缀都必须有有效的非负监督权重。")
            self.prefix_samples = prefix_samples
            self.state_normalizer_samples = state_normalizer_samples
            self.prefix_loss_weights = prefix_loss_weights
        else:
            self.prefix_samples = ()
            self.state_normalizer_samples = None
            self.prefix_loss_weights = ()
        self.context_transformer = None
        self.context_mode = readout

    def prefix_candidate_states(
        self,
        signals,
        candidate_embeddings,
        subject_indices=None,
        group_indices=None,
        return_increments=False,
        return_supervision_scores=False,
    ):
        """从严格输入前缀估计累计状态及尺度统一的监督分数。"""
        del group_indices
        if not self.prefix_supervised:
            raise ValueError("当前候选证据模型没有启用前缀共同监督。")
        if signals.ndim != 3:
            raise ValueError("signals 必须为 batch×channel×time 张量。")
        if signals.shape[-1] != self.state_normalizer_samples:
            raise ValueError(
                "软累积候选状态要求完整训练窗口长度为 "
                f"{self.state_normalizer_samples}，实际为 {signals.shape[-1]}。"
            )

        states_by_prefix = {}
        supervision_by_prefix = {}

        def encode_prefix(prefix):
            temporal_features = self.brain_encoder.encode_temporal(
                signals[..., :prefix], subject_indices
            )
            signed_evidence = self.evidence_readout.signed_evidence(
                temporal_features, candidate_embeddings
            )
            evidence_sum = signed_evidence.sum(dim=-1)
            # 读取器的完整窗口分数是 mean_t(e_t)。固定用完整窗口长度
            # 归一化后，A_k = sum_{t<=k}(e_t)/T，状态强度可随证据累积。
            states_by_prefix[prefix] = (
                evidence_sum / self.state_normalizer_samples
            )
            # SigLIP 的温度和偏置由所有前缀共享。训练时恢复到 [-1, 1]
            # 的共同分数范围，避免短前缀因累计尺度较小而无法形成正 logit。
            supervision_by_prefix[prefix] = evidence_sum / prefix

        if self.training:
            # 完整窗口先运行且是唯一更新 BatchNorm running statistics 的分支。
            # 短前缀仍保留梯度，但复用刚更新的完整窗口统计。
            full_prefix = self.prefix_samples[-1]
            encode_prefix(full_prefix)
            batch_norms = [
                module
                for module in self.brain_encoder.modules()
                if isinstance(
                    module,
                    (
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.SyncBatchNorm,
                    ),
                )
            ]
            original_batch_norm_modes = [module.training for module in batch_norms]
            cuda_devices = []
            if signals.is_cuda:
                cuda_devices = [
                    signals.device.index
                    if signals.device.index is not None
                    else torch.cuda.current_device()
                ]
            try:
                # 辅助前缀不推进全局随机流，尽量保持完整窗口主任务与 V0
                # 的 dropout 随机序列可比。
                with torch.random.fork_rng(devices=cuda_devices):
                    for module in batch_norms:
                        module.eval()
                    for prefix in self.prefix_samples[:-1]:
                        encode_prefix(prefix)
            finally:
                for module, training in zip(
                    batch_norms, original_batch_norm_modes
                ):
                    module.train(training)
        else:
            for prefix in self.prefix_samples:
                encode_prefix(prefix)

        states = torch.stack(
            [states_by_prefix[prefix] for prefix in self.prefix_samples], dim=0
        )
        supervision_scores = torch.stack(
            [
                supervision_by_prefix[prefix]
                for prefix in self.prefix_samples
            ],
            dim=0,
        )
        if not return_increments and not return_supervision_scores:
            return states
        increments = torch.cat(
            [states[:1], states[1:] - states[:-1]], dim=0
        )
        if return_increments and return_supervision_scores:
            return states, increments, supervision_scores
        if return_increments:
            return states, increments
        return states, supervision_scores

    def score_temporal_features(
        self,
        temporal_features,
        candidate_embeddings,
        group_indices=None,
        return_evidence=False,
        return_attention=False,
    ):
        del group_indices
        if return_attention:
            if not isinstance(
                self.evidence_readout,
                CandidateConditionedAttentionEvidenceReadout,
            ):
                raise ValueError("当前读取机制没有独立候选注意力。")
            return self.evidence_readout(
                temporal_features,
                candidate_embeddings,
                return_evidence=return_evidence,
                return_attention=True,
            )
        return self.evidence_readout(
            temporal_features,
            candidate_embeddings,
            return_evidence=return_evidence,
        )

    def forward(
        self,
        signals,
        candidate_embeddings,
        subject_indices=None,
        group_indices=None,
        return_evidence=False,
        return_temporal_features=False,
        return_attention=False,
    ):
        temporal_features = self.brain_encoder.encode_temporal(
            signals, subject_indices
        )
        output = self.score_temporal_features(
            temporal_features,
            candidate_embeddings,
            group_indices,
            return_evidence=return_evidence,
            return_attention=return_attention,
        )
        if return_temporal_features:
            return output, temporal_features
        return output


class CandidateContextResidualModel(BrainEmbeddingModel):
    """在分组上下文分数上叠加候选条件的有符号时间证据。"""

    candidate_conditioned = True

    def __init__(self, channel_positions, subject_count, config):
        if not bool(config.get("use_transformer", False)):
            raise ValueError("候选上下文残差模型必须启用句级 Transformer。")
        if str(config.get("context_mode", "grouped")) != "grouped":
            raise ValueError("候选上下文残差模型只支持 grouped context。")
        super().__init__(channel_positions, subject_count, config)
        if self.brain_encoder.temporal_attention is None:
            raise ValueError("候选上下文残差模型需要保留固定时间汇聚主路。")
        output_dimension = int(config.get("embedding_dimension", 1024))
        evidence_config = config.get("candidate_evidence", {})
        # 新分支不能推进全局随机流，否则同 seed 的 Transformer/dropout
        # 配对比较会因初始化副作用而失去逐步可比性。
        with torch.random.fork_rng(devices=[]):
            self.evidence_readout = CandidateConditionedEvidenceReadout(
                feature_dimension=output_dimension,
                candidate_dimension=output_dimension,
                evidence_dimension=int(
                    evidence_config.get("evidence_dimension", 128)
                ),
            )
        initial_gate = float(evidence_config.get("residual_gate_init", 0.0))
        if initial_gate != 0.0:
            raise ValueError("V1 候选证据残差门必须从 0 初始化。")
        self.evidence_gate_logit = nn.Parameter(torch.tensor(initial_gate))

    @property
    def evidence_gate(self):
        """把新增分支限制为有界残差，并保留精确零初始化。"""
        return torch.tanh(self.evidence_gate_logit)

    def score_temporal_features(
        self,
        temporal_features,
        candidate_embeddings,
        group_indices=None,
        return_evidence=False,
    ):
        if group_indices is None:
            raise ValueError("候选上下文残差评分需要 group_indices。")
        brain_embedding = F.normalize(
            self.brain_encoder.temporal_attention(temporal_features), dim=-1
        )
        context_embedding = self.context_transformer(brain_embedding, group_indices)
        context_embedding = F.normalize(context_embedding, dim=-1)
        normalized_candidates = F.normalize(candidate_embeddings, dim=-1, eps=1e-15)
        context_scores = context_embedding @ normalized_candidates.T
        evidence_scores, evidence = self.evidence_readout(
            temporal_features,
            candidate_embeddings,
            return_evidence=True,
        )
        scores = context_scores + self.evidence_gate * evidence_scores
        if return_evidence:
            return scores, evidence
        return scores

    def forward(
        self,
        signals,
        candidate_embeddings,
        subject_indices=None,
        group_indices=None,
        return_evidence=False,
        return_temporal_features=False,
    ):
        temporal_features = self.brain_encoder.encode_temporal(
            signals, subject_indices
        )
        output = self.score_temporal_features(
            temporal_features,
            candidate_embeddings,
            group_indices,
            return_evidence=return_evidence,
        )
        if return_temporal_features:
            return output, temporal_features
        return output


def build_word_decoding_model(
    channel_count,
    channel_positions,
    subject_count,
    config,
):
    """按读取机制构建静态嵌入或候选条件证据模型。"""
    readout = str(config.get("readout", "brain_embedding"))
    if readout == "brain_embedding":
        return build_brain_embedding_model(
            channel_count,
            channel_positions,
            subject_count,
            config,
        )
    if readout in {
        "three_state_trajectory",
        "three_state_concat_control",
        "three_state_conditional_trajectory",
    }:
        if len(channel_positions) != int(channel_count):
            raise ValueError("Signal channels and channel_positions do not match.")
        return ThreeStateBrainEmbeddingModel(
            channel_positions=channel_positions,
            subject_count=int(subject_count),
            config=config,
            mode={
                "three_state_trajectory": "trajectory",
                "three_state_concat_control": "concat_control",
                "three_state_conditional_trajectory": (
                    "conditional_trajectory"
                ),
            }[readout],
        )
    if readout in {
        "five_state_multiscale_trajectory",
        "five_state_concat_control",
    }:
        if len(channel_positions) != int(channel_count):
            raise ValueError("Signal channels and channel_positions do not match.")
        return FiveStateBrainEmbeddingModel(
            channel_positions=channel_positions,
            subject_count=int(subject_count),
            config=config,
            mode=(
                "multiscale_trajectory"
                if readout == "five_state_multiscale_trajectory"
                else "concat_control"
            ),
        )
    if readout == "candidate_context_residual":
        if len(channel_positions) != int(channel_count):
            raise ValueError("Signal channels and channel_positions do not match.")
        return CandidateContextResidualModel(
            channel_positions=channel_positions,
            subject_count=int(subject_count),
            config=config,
        )
    if readout not in {
        "candidate_evidence",
        "candidate_evidence_attention",
        "candidate_evidence_competition_soft",
    }:
        raise ValueError(f"未知词解码读取机制：{readout}")
    if len(channel_positions) != int(channel_count):
        raise ValueError("Signal channels and channel_positions do not match.")
    return CandidateEvidenceModel(
        channel_positions=channel_positions,
        subject_count=int(subject_count),
        config=config,
    )
