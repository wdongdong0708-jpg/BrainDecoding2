"""活跃 BrainEmbeddingModel 权重 state_dict 键名与形状冻结回归测试。

确保仓库重构期间绝不改动任何活跃 nn.Module 的参数键名或张量形状，
保证已有 10 被试 Pallier2025 `.pt` 检查点零重训直接加载。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch

from braindecoding.config import load_yaml_with_extends
from braindecoding.models import build_brain_embedding_model


FROZEN_STATE_DICT_CONTRACTS = (
    (
        "word_only",
        "configs/training/pallier2025_word_encoder.yaml",
        43,
        "bf97cc56ead709daa45bab3324c7dbc716570c9f985cbf80a9d79241085b9935",
    ),
    (
        "full_context",
        "configs/word_decoding/pallier2025/all_subjects/main_context.yaml",
        237,
        "e819b375f26581a7e674855f9a298875bc37b1cdef3ae7266585ba7fdd30b164",
    ),
    (
        "rolling_k2",
        "configs/word_decoding/pallier2025/rolling_revision/local_window_revision.yaml",
        237,
        "e819b375f26581a7e674855f9a298875bc37b1cdef3ae7266585ba7fdd30b164",
    ),
    (
        "prefix_revision",
        "configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml",
        237,
        "e819b375f26581a7e674855f9a298875bc37b1cdef3ae7266585ba7fdd30b164",
    ),
    (
        "amortized_timing",
        "configs/training/siamese_debiased/pallier2025_amortized_timing_prefix_revision.yaml",
        247,
        "a4ddd35c0c0e157c7c41dda234e349c50d62b85e3bd660c80f9bc53b8e0dec3a",
    ),
)


@pytest.mark.parametrize(
    ("name", "config_path", "expected_tensor_count", "expected_sha256"),
    FROZEN_STATE_DICT_CONTRACTS,
)
def test_active_model_state_dict_keys_and_shapes_frozen(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    config_path: str,
    expected_tensor_count: int,
    expected_sha256: str,
) -> None:
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    cfg = load_yaml_with_extends(config_path)
    positions = np.zeros((306, 2), dtype=np.float32)
    model = build_brain_embedding_model(
        channel_count=306,
        channel_positions=positions,
        subject_count=10,
        config=cfg["model"],
    )
    signature = {key: list(tensor.shape) for key, tensor in model.state_dict().items()}
    digest = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode("utf-8")
    ).hexdigest()

    assert len(signature) == expected_tensor_count, (
        f"{name} state_dict 张量数量漂移：{len(signature)} != {expected_tensor_count}"
    )
    assert digest == expected_sha256, (
        f"{name} state_dict 键名或形状发生变化（会破坏已有 .pt 权重加载）！"
    )

    # 往返严格加载校验（strict=True）
    reloaded = build_brain_embedding_model(
        channel_count=306,
        channel_positions=positions,
        subject_count=10,
        config=cfg["model"],
    )
    incompatible = reloaded.load_state_dict(model.state_dict(), strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys


def test_word_only_state_dict_loads_into_stage2_brain_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Stage 1 冻结卷积基座权重可直接热启动加载至 Stage 2 模型的 brain_encoder。"""
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    positions = np.zeros((306, 2), dtype=np.float32)
    word_cfg = load_yaml_with_extends("configs/training/pallier2025_word_encoder.yaml")
    stage2_cfg = load_yaml_with_extends(
        "configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml"
    )
    stage1 = build_brain_embedding_model(306, positions, 10, word_cfg["model"])
    stage2 = build_brain_embedding_model(306, positions, 10, stage2_cfg["model"])

    encoder_weights = {
        key.removeprefix("brain_encoder."): tensor.clone()
        for key, tensor in stage1.state_dict().items()
        if key.startswith("brain_encoder.")
    }
    incompatible = stage2.brain_encoder.load_state_dict(encoder_weights, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys
    for key, tensor in encoder_weights.items():
        assert torch.equal(stage2.brain_encoder.state_dict()[key], tensor)
