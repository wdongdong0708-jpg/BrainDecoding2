"""Tier-2 去毒涨点模块单元测试：MultiBinResConvEncoder、邻居专用 SwiGLU、英语 Stage-3 解码与配置契约。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from braindecoding.config import load_yaml_with_extends
from braindecoding.decoding.language_model_scoring import (
    PROMPTS,
    PreTokenizedVocabulary,
    prefix_context_kind,
    render_words,
    word_surface_for_context,
)
from braindecoding.decoding.state_hyperparameter_calibrator import (
    DEFAULT_TEMPERATURE_GRID,
)
from braindecoding.models import (
    MultiBinResConvEncoder,
    SalienceGatedAdditiveV2Encoder,
    build_brain_embedding_model,
)
from braindecoding.optimizers import build_adamw_for_modules
from braindecoding.training.debiased_padded_forward import (
    _apply_siamese_shared_dropout,
)
from braindecoding.training.workflow_runner import build_workflow_config


def test_multibin_resconv_encoder_shapes_bins_batchnorm_and_state_dict_isolation() -> None:
    torch.manual_seed(11)
    positions = np.linspace(0.0, 1.0, 32 * 2, dtype=np.float32).reshape(32, 2)
    model_cfg = {
        "embedding_dimension": 64,
        "use_transformer": False,
        "context_mode": "grouped",
        "conv": {
            "architecture": "multibin_resconv",
            "merger_channels": 24,
            "merger_position_dimension": 32,
            "merger_per_subject": True,
            "merger_dropout": 0.1,
            "merger_per_sample_dropout": False,
            "hidden": 32,
            "depth": 6,
            "kernel_size": 3,
            "dilation_growth": 2,
            "dilation_period": 4,
            "batch_norm": True,
            "glu_every": 2,
            "num_temporal_bins": 5,
            "bin_attention_hidden": 16,
            "mlp_dim": 64,
            "mlp_hidden": 128,
            "mlp_depth": 3,
            "mlp_dropout": 0.1,
            "gaussian_noise_std": 0.05,
            "channel_dropout": 0.05,
        },
    }
    model = build_brain_embedding_model(32, positions, 2, model_cfg)
    assert isinstance(model.brain_encoder, MultiBinResConvEncoder)
    assert model.brain_encoder.bin_feature_dimension == 5 * 32

    # 卷积主干严格使用 BatchNorm1d，严禁出现 GroupNorm 或 LayerNorm
    trunk_modules = list(model.brain_encoder.blocks.modules()) + list(
        model.brain_encoder.gates.modules()
    )
    assert any(isinstance(m, torch.nn.BatchNorm1d) for m in trunk_modules)
    assert not any(
        isinstance(m, (torch.nn.GroupNorm, torch.nn.LayerNorm))
        for m in trunk_modules
    )

    signals_50 = torch.randn(4, 32, 50)
    subjects = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    # 训练期增广生效且全参数 step-0 梯度有限非零
    model.train()
    out_train = model(signals_50, subjects)
    assert out_train.shape == (4, 64)
    assert torch.allclose(out_train.norm(dim=-1), torch.ones(4), atol=1e-5)
    loss = (out_train * torch.randn_like(out_train)).sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"缺少梯度：{name}"
            assert torch.isfinite(param.grad).all(), f"非有限梯度：{name}"
            assert float(param.grad.abs().max().item()) > 0.0, f"零梯度：{name}"

    # 推理期确定性且 encode_with_bins 与 forward 严格一致（同时验证整除 T=50 与非整除 T=64）
    model.eval()
    with torch.no_grad():
        z_50, h_bins_50 = model.brain_encoder.encode_with_bins(signals_50, subjects)
        assert z_50.shape == (4, 64)
        assert h_bins_50.shape == (4, 5 * 32)
        assert torch.allclose(
            model(signals_50, subjects), F.normalize(z_50, dim=-1), atol=1e-6
        )
        assert torch.allclose(
            model(signals_50, subjects), model(signals_50, subjects), atol=1e-6
        )

        signals_64 = torch.randn(4, 32, 64)
        z_64, h_bins_64 = model.brain_encoder.encode_with_bins(signals_64, subjects)
        assert z_64.shape == (4, 64)
        assert h_bins_64.shape == (4, 5 * 32)
        assert torch.isfinite(z_64).all() and torch.isfinite(h_bins_64).all()

    # strict state_dict 往返加载
    reloaded = build_brain_embedding_model(32, positions, 2, model_cfg)
    reloaded.load_state_dict(model.state_dict(), strict=True)
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(
            reloaded(signals_50, subjects), model(signals_50, subjects), atol=1e-6
        )


@pytest.mark.parametrize("horizon_conditioned", (False, True))
def test_salience_v2_neighbor_swiglu_self_mask_parity_and_step0_grad(
    horizon_conditioned: bool,
) -> None:
    torch.manual_seed(23)
    dim = 64

    # 确认 neighbor_swiglu=False 不向 state_dict 引入任何多余键
    base_cfg = {
        "architecture": "salience_gated_additive_v2",
        "heads": 8,
        "head_dimension": 8,
        "max_relative_distance": 12,
        "residual_max_norm": 0.70,
        "attention_dropout": 0.0,
        "horizon_conditioned": horizon_conditioned,
        "neighbor_swiglu": False,
    }
    enc_no_swi = SalienceGatedAdditiveV2Encoder(dim, base_cfg)
    assert not any("neighbor_swiglu" in k for k in enc_no_swi.state_dict().keys())

    cfg = {
        **base_cfg,
        "neighbor_swiglu": True,
        "neighbor_swiglu_heads": 4,
        "neighbor_swiglu_head_dimension": 8,
        "neighbor_swiglu_hidden_dimension": 32,
        "neighbor_swiglu_dropout": 0.30,
    }
    enc = SalienceGatedAdditiveV2Encoder(dim, cfg)
    u0_text = F.normalize(torch.randn(dim), dim=-1)
    enc.set_text_cone_axis(u0_text)

    z = F.normalize(0.5 * u0_text.unsqueeze(0) + 0.5 * torch.randn(10, dim), dim=-1)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    enc.set_clean_input_mean(z.mean(dim=0))

    padded = torch.zeros((3, 4, dim), dtype=torch.float32)
    mask = torch.zeros((3, 4), dtype=torch.bool)
    padded[0, :4] = z[:4]
    mask[0, :4] = True
    padded[1, :3] = z[4:7]
    mask[1, :3] = True
    padded[2, :3] = z[7:10]
    mask[2, :3] = True

    # 1. Siamese 多分支共享 Dropout 前向 + step-0 满秩非零梯度校验
    enc.train()
    multi_padded = torch.cat([padded, padded * 0.9, padded * 0.8], dim=0)
    multi_mask = mask.repeat(3, 1)
    out_train = enc.forward_padded(
        multi_padded,
        multi_mask,
        num_branches=3,
        synchronize_dropout_rng=True,
        shared_dropout_fn=_apply_siamese_shared_dropout,
    )
    target = torch.randn_like(out_train)
    (out_train * target).sum().backward()
    for name in (
        "pos_logits",
        "neg_logits",
        "sink_logits",
        "value_proj.weight",
        "out_proj.weight",
        "neighbor_swiglu_w12.weight",
        "neighbor_swiglu_w3.weight",
        "neighbor_swiglu_out_proj.weight",
        "neighbor_swiglu_pos_logits",
        "neighbor_swiglu_neg_logits",
        "neighbor_swiglu_sink_logits",
    ):
        param = dict(enc.named_parameters())[name]
        assert param.grad is not None, f"缺少 step-0 梯度：{name}"
        assert torch.isfinite(param.grad).all(), f"非有限梯度：{name}"
        assert float(param.grad.abs().max().item()) > 0.0, f"零 step-0 梯度：{name}"

    # 2. 严格屏蔽 Δ=0 验证：当主干线性输出置零时，单字句 (L=1) 邻居 SwiGLU 贡献必须严格为 0
    with torch.no_grad():
        saved_out_w = enc.out_proj.weight.clone()
        saved_h_up = (
            enc.horizon_up.clone() if enc.horizon_up is not None else None
        )
        enc.out_proj.weight.zero_()
        if enc.horizon_up is not None:
            enc.horizon_up.zero_()
        enc.neighbor_swiglu_out_proj.weight.normal_(mean=0.0, std=0.5)
    enc.eval()
    single_z = F.normalize(torch.randn(2, dim), dim=-1)
    single_groups = torch.tensor([0, 1], dtype=torch.long)
    y_single = enc(single_z, single_groups)
    assert torch.allclose(y_single, single_z, atol=1e-5)

    # 恢复主干线性输出，验证 forward == forward_padded == 单目标剪枝 逐位等价与文本锥轴不变性
    with torch.no_grad():
        enc.out_proj.weight.copy_(saved_out_w)
        if saved_h_up is not None:
            enc.horizon_up.copy_(saved_h_up)
            enc.horizon_up.normal_(mean=0.0, std=0.05)

    enc.eval()
    y_grouped = enc(z, groups)
    y_padded = enc.forward_padded(padded, mask, num_branches=1)
    reconstructed = torch.cat(
        [y_padded[0, :4], y_padded[1, :3], y_padded[2, :3]], dim=0
    )
    assert torch.allclose(y_grouped, reconstructed, atol=1e-5)

    target_group = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    target_col = torch.tensor([1, 3, 2, 0], dtype=torch.long)
    y_pruned = enc.forward_padded(
        padded,
        mask,
        target_group_idx=target_group,
        target_col_idx=target_col,
    )
    assert torch.allclose(y_pruned, y_padded[target_group, target_col], atol=1e-5)

    # 文本锥轴投影严格锁定
    assert torch.allclose(
        (y_grouped * u0_text).sum(dim=-1),
        (z * u0_text).sum(dim=-1),
        atol=1e-5,
    )

    # 解耦优化器按 neighbor_swiglu_weight_decay 正确分组，且核 logits 豁免衰减
    opt = build_adamw_for_modules(
        [enc],
        {
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "neighbor_swiglu_weight_decay": 0.04,
            "decoupled_weight_decay": True,
        },
    )
    wd_map = {round(g["weight_decay"], 4): len(g["params"]) for g in opt.param_groups}
    assert set(wd_map.keys()) == {0.0, 0.01, 0.04}
    assert wd_map[0.04] == 3  # w12, w3, out_proj


def test_stage3_english_rendering_pretokenization_and_temperature_grid() -> None:
    assert "en" in PROMPTS and PROMPTS["en"].strip()
    assert render_words(["the", "black", "cat", "'s", "hat"], "en") == "the black cat's hat"
    assert render_words(["well", "-known"], "en") == "well-known"
    assert prefix_context_kind((), "en") == "initial"
    assert prefix_context_kind(("the",), "en") == "continuation"
    assert word_surface_for_context("cat", "initial") == "cat"
    assert word_surface_for_context("cat", "continuation") == " cat"
    assert word_surface_for_context("'s", "continuation") == "'s"

    class _FakeTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) for ch in text]

    vocab_table = PreTokenizedVocabulary(
        ["the", "cat", "sherlock"], _FakeTokenizer(), "en"
    )
    assert set(vocab_table.tokens_by_context.keys()) == {"initial", "continuation"}
    assert DEFAULT_TEMPERATURE_GRID == (0.025, 0.035, 0.05, 0.07, 0.10)


def test_tier2_detoxed_configs_resolve_pure_siglip_and_hyperparams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")

    cfg_w, _ = build_workflow_config(
        "libribrain100",
        seed=0,
        config_path="configs/training/libribrain100_word_encoder_multibin_resconv_butterworth.yaml",
        exp_tag="t2_check",
    )
    assert cfg_w["model"]["conv"]["architecture"] == "multibin_resconv"
    assert cfg_w["dataset"]["causal_highpass_mode"] == "butterworth_iir"
    assert float(cfg_w["dataset"]["clamp"]) == pytest.approx(20.0)
    assert str(cfg_w["loss"]["name"]).lower() == "siglip"

    for cfg_path, expect_hcond, expect_lam_d in (
        (
            "configs/training/siamese_debiased/libribrain100_multibin_salience_v2_neighbor_swiglu_context.yaml",
            False,
            0.0,
        ),
        (
            "configs/training/siamese_debiased/libribrain100_multibin_salience_v2_neighbor_swiglu_prefix_revision.yaml",
            True,
            10.0,
        ),
    ):
        cfg_s2, _ = build_workflow_config(
            "libribrain100",
            seed=0,
            config_path=cfg_path,
            exp_tag="t2_s2_check",
            word_exp_tag="t2_check",
        )
        tr = cfg_s2["model"]["transformer"]
        assert tr["architecture"] == "salience_gated_additive_v2"
        assert tr["neighbor_swiglu"] is True
        assert tr["horizon_conditioned"] is expect_hcond
        assert str(cfg_s2["loss"]["name"]).lower() == "siglip"
        assert int(cfg_s2["training"]["batch_size"]) == 128
        assert float(cfg_s2["training"]["neighbor_swiglu_weight_decay"]) == pytest.approx(0.04)
        siam = cfg_s2["workflow"]["siamese_debiased"]
        assert float(siam["eta_train"]) == pytest.approx(0.0)
        assert float(siam["lambda_null"]) == pytest.approx(60.0)
        assert float(siam["lambda_delta"]) == pytest.approx(expect_lam_d)
        assert float(siam["null_space_excess_tolerance"]) == pytest.approx(0.0)

    s3_cfg = load_yaml_with_extends(
        "configs/decoding/libribrain100_sequence_decoding_protocol.yaml"
    )
    assert int(s3_cfg["decoding"]["candidate_top_k"]) == 50
    assert list(s3_cfg["calibration"]["temperature_grid"]) == [
        0.025,
        0.035,
        0.05,
        0.07,
        0.10,
    ]
    assert float(s3_cfg["control"]["common_mode_subtraction"]["subtraction_eta"]) == 0.0
