"""神经信号编码与局部前瞻上下文模型子包。

- spatial: 传感器二维投影、Fourier 位置编码、SpatialChannelMerger、SubjectLinear
- brain_encoder: DilatedConvEncoder、StridedConvEncoder、ExplicitTemporalFilter、TemporalAttentionPool
- transformer: 局部相对位置偏置、物理时间偏置、GroupedTransformerEncoder
- embedding: BrainEmbeddingModel、TemporalGeometryBiasHead 及工厂入口
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
    TemporalGeometryBiasHead,
    build_brain_embedding_model,
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

build_word_decoding_model = build_brain_embedding_model

__all__ = [
    "project_scalp_positions",
    "FourierPositionEmbedding",
    "SpatialChannelMerger",
    "SubjectLinear",
    "ResidualSubjectLinear",
    "StridedConvEncoder",
    "TemporalAttentionPool",
    "DilatedConvEncoder",
    "ExplicitTemporalFilter",
    "LocalRelativePositionBias",
    "PhysicalTimeGapBias",
    "EventTimeInteractionBias",
    "AdaptiveDirectionalBiasGate",
    "GroupedTransformerEncoder",
    "PreTransformerTrajectoryResidual",
    "RelationDiscriminationHead",
    "TemporalGeometryBiasHead",
    "BrainEmbeddingModel",
    "build_strided_conv_encoder",
    "build_brain_embedding_model",
    "build_word_decoding_model",
]
