"""神经信号编码与局部窗口滚动修订模型子包。

- spatial: 传感器二维投影、Fourier 位置编码、SpatialChannelMerger、SubjectLinear
- brain_encoder: DilatedConvEncoder、StridedConvEncoder、ExplicitTemporalFilter、TemporalAttentionPool
- transformer: 局部相对位置偏置、物理时间偏置、GroupedTransformerEncoder
- embedding: BrainEmbeddingModel、PreTransformerTrajectoryResidual、RelationDiscriminationHead
- experimental: 三状态/五状态轨迹残差与候选条件证据读取变体
"""
from __future__ import annotations

from braindecoding.models.brain_encoder import (
    DilatedConvEncoder,
    ExplicitTemporalFilter,
    StridedConvEncoder,
    TemporalAttentionPool,
    build_strided_conv_encoder,
)
from braindecoding.models.embedding import (
    BrainEmbeddingModel,
    PreTransformerTrajectoryResidual,
    RelationDiscriminationHead,
    build_brain_embedding_model,
)
from braindecoding.models.experimental import (
    CandidateConditionedAttentionEvidenceReadout,
    CandidateConditionedEvidenceReadout,
    CandidateContextResidualModel,
    CandidateEvidenceModel,
    FiveStateBrainEmbeddingModel,
    FiveStateTrajectoryReadout,
    ThreeStateBrainEmbeddingModel,
    ThreeStateTrajectoryReadout,
    build_word_decoding_model,
)
from braindecoding.models.spatial import (
    FourierPositionEmbedding,
    ResidualSubjectLinear,
    SpatialChannelMerger,
    SubjectLinear,
    project_scalp_positions,
)
from braindecoding.models.transformer import (
    AdaptiveDirectionalBiasGate,
    EventTimeInteractionBias,
    GroupedTransformerEncoder,
    LocalRelativePositionBias,
    PhysicalTimeGapBias,
    _local_attention_window_pre_hook,
)

__all__ = [
    "project_scalp_positions",
    "FourierPositionEmbedding",
    "SpatialChannelMerger",
    "SubjectLinear",
    "ResidualSubjectLinear",
    "StridedConvEncoder",
    "TemporalAttentionPool",
    "CandidateConditionedEvidenceReadout",
    "CandidateConditionedAttentionEvidenceReadout",
    "DilatedConvEncoder",
    "ExplicitTemporalFilter",
    "LocalRelativePositionBias",
    "PhysicalTimeGapBias",
    "EventTimeInteractionBias",
    "AdaptiveDirectionalBiasGate",
    "GroupedTransformerEncoder",
    "PreTransformerTrajectoryResidual",
    "RelationDiscriminationHead",
    "BrainEmbeddingModel",
    "ThreeStateTrajectoryReadout",
    "ThreeStateBrainEmbeddingModel",
    "FiveStateTrajectoryReadout",
    "FiveStateBrainEmbeddingModel",
    "CandidateEvidenceModel",
    "CandidateContextResidualModel",
    "build_strided_conv_encoder",
    "build_brain_embedding_model",
    "build_word_decoding_model",
]
