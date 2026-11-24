"""词内显著性门控加性注意力、超额相干错位零空间与同词间隔残差正交损失单元测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from braindecoding.models import (
    SalienceGatedAdditiveEncoder,
    build_brain_embedding_model,
)
from braindecoding.training.debiased_losses import (
    CleanSteinRunningLoss,
    build_clean_stein_residual_basis,
    build_vocabulary_gram_sqrt,
    compute_excess_coherent_shift_null_loss,
)
from braindecoding.training.debiased_padded_forward import (
    _can_use_direct_layer_loop,
    _can_use_fast_padded_transformer,
    _run_fast_padded_transformer,
)
from braindecoding.training.workflow_runner import build_workflow_config


def test_salience_gated_additive_cone_preservation_and_padded_equivalence() -> None:
    torch.manual_seed(42)
    dim = 64
    cfg = {
        "architecture": "salience_gated_additive",
        "heads": 8,
        "head_dimension": 8,
        "max_relative_distance": 12,
        "residual_max_norm": 0.65,
        "attention_dropout": 0.0,
    }
    enc = SalienceGatedAdditiveEncoder(dim, cfg)
    u0 = F.normalize(torch.randn(dim), dim=-1)
    z_raw = 0.75 * u0.unsqueeze(0) + 0.35 * torch.randn(11, dim)
    z = F.normalize(z_raw, dim=-1)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)

    enc.set_clean_input_mean(z.mean(dim=0))

    # 当 out_proj 严格置零时，输出必须与单窗特征 z 逐元素一致
    with torch.no_grad():
        saved_w = enc.out_proj.weight.clone()
        enc.out_proj.weight.zero_()
    enc.eval()
    y_zero = enc(z, groups)
    assert torch.allclose(y_zero, z, atol=1e-5)

    with torch.no_grad():
        enc.out_proj.weight.normal_(mean=0.0, std=0.25)
    # 先跑一次训练步以填充 psi_clean_ema
    enc.train()
    padded = torch.zeros((3, 4, dim), dtype=torch.float32)
    mask = torch.zeros((3, 4), dtype=torch.bool)
    padded[0, :4] = z[:4]
    mask[0, :4] = True
    padded[1, :3] = z[4:7]
    mask[1, :3] = True
    padded[2, :4] = z[7:11]
    mask[2, :4] = True
    _ = enc.forward_padded(padded, mask, num_branches=1)

    enc.eval()
    y_grouped = enc(z, groups)
    y_padded = enc.forward_padded(padded, mask, num_branches=1)

    # 验证单窗锥轴分量 c_i = z_i^T u_0 严格恒定且输出为单位范数
    u0_hat = F.normalize(enc.clean_input_mean, dim=-1)
    assert torch.allclose(
        (y_grouped * u0_hat).sum(dim=-1),
        (z * u0_hat).sum(dim=-1),
        atol=1e-5,
    )
    assert torch.allclose(
        y_grouped.norm(dim=-1), torch.ones(len(z)), atol=1e-5
    )

    # 验证 forward(features, groups) 与 forward_padded 及单目标剪枝逐位等价
    reconstructed = torch.cat(
        [y_padded[0, :4], y_padded[1, :3], y_padded[2, :4]], dim=0
    )
    assert torch.allclose(y_grouped, reconstructed, atol=1e-5)

    target_group = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    target_col = torch.tensor([2, 1, 0, 3], dtype=torch.long)
    y_pruned = enc.forward_padded(
        padded,
        mask,
        target_group_idx=target_group,
        target_col_idx=target_col,
    )
    expected_pruned = y_padded[target_group, target_col]
    assert torch.allclose(y_pruned, expected_pruned, atol=1e-5)

    # 验证 strict state_dict 往返加载保留持久缓冲区
    reloaded = SalienceGatedAdditiveEncoder(dim, cfg)
    reloaded.load_state_dict(enc.state_dict(), strict=True)
    reloaded.eval()
    assert torch.allclose(reloaded(z, groups), y_grouped, atol=1e-6)


def test_soft_bounded_standard_transformer_padded_equivalence() -> None:
    torch.manual_seed(7)
    positions = np.zeros((306, 2), dtype=np.float32)
    model_cfg = {
        "embedding_dimension": 64,
        "use_transformer": True,
        "context_mode": "grouped",
        "conv": {"channels": 64, "blocks": 1},
        "transformer": {
            "architecture": "standard",
            "depth": 2,
            "heads": 4,
            "attention_dropout": 0.0,
            "feedforward_dropout": 0.0,
            "soft_bounded_residual_max_norm": 0.65,
        },
    }
    model = build_brain_embedding_model(306, positions, 1, model_cfg)
    model.eval()
    assert _can_use_fast_padded_transformer(model)
    assert _can_use_direct_layer_loop(
        model.context_transformer, use_causal_attention=False
    )

    z = F.normalize(torch.randn(7, 64), dim=-1)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    y_direct = model.context_transformer(z, groups)

    padded = torch.zeros((2, 4, 64), dtype=torch.float32)
    mask = torch.zeros((2, 4), dtype=torch.bool)
    padded[0, :4] = z[:4]
    mask[0, :4] = True
    padded[1, :3] = z[4:7]
    mask[1, :3] = True
    y_fast = _run_fast_padded_transformer(
        model.context_transformer, padded, mask
    )
    y_fast_flat = torch.cat([y_fast[0, :4], y_fast[1, :3]], dim=0)
    assert torch.allclose(y_direct, y_fast_flat, atol=1e-5)

    t_grp = torch.tensor([0, 1], dtype=torch.long)
    t_col = torch.tensor([2, 1], dtype=torch.long)
    y_fast_q = _run_fast_padded_transformer(
        model.context_transformer,
        padded,
        mask,
        target_group_idx=t_grp,
        target_col_idx=t_col,
    )
    assert torch.allclose(y_fast_q, y_fast[t_grp, t_col], atol=1e-5)


def test_excess_coherent_shift_null_loss_zero_incoherent_and_active_coherent() -> None:
    torch.manual_seed(123)
    vocab = F.normalize(torch.randn(200, 32), dim=-1)
    centered_bank = vocab - vocab.mean(dim=0, keepdim=True)
    gram_sqrt = build_vocabulary_gram_sqrt(centered_bank)

    # 验证 Gram 平方根精确等价于 (D / |V|) * ||X @ E_bar^T||_F^2
    x_probe = torch.randn(16, 32)
    lhs = (x_probe @ gram_sqrt).pow(2).sum()
    rhs = (32.0 / 200.0) * (x_probe @ centered_bank.t()).pow(2).sum()
    assert torch.allclose(lhs, rhs, rtol=1e-4, atol=1e-5)

    # 1. 纯独立分支语音噪声（无相干时序偏置）：期望超额损失应为 0 且梯度为 0
    incoherent = torch.randn(5, 256, 32) * 0.2
    incoherent_0 = incoherent[0].clone().requires_grad_(True)
    br_incoh = torch.cat([incoherent_0.unsqueeze(0), incoherent[1:].detach()], dim=0)
    loss_incoh = compute_excess_coherent_shift_null_loss(
        br_incoh,
        centered_bank,
        vocab_gram_sqrt=gram_sqrt,
        excess_tolerance=0.15,
    )
    loss_incoh.backward()
    assert float(loss_incoh.item()) == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(incoherent_0.grad, torch.zeros_like(incoherent_0))

    # 2. 注入跨分支共享相干时序偏置：超额损失严格 > 0 且仅对相干分量产生满幅无偏梯度
    coherent_bias = torch.randn(1, 256, 32) * 0.5
    branch0 = (coherent_bias[0] + 0.05 * torch.randn(256, 32)).requires_grad_(True)
    rest = (coherent_bias + 0.05 * torch.randn(4, 256, 32)).detach()
    br_coh = torch.cat([branch0.unsqueeze(0), rest], dim=0)
    loss_coh = compute_excess_coherent_shift_null_loss(
        br_coh,
        centered_bank,
        vocab_gram_sqrt=gram_sqrt,
        excess_tolerance=0.08,
    )
    loss_coh.backward()
    assert float(loss_coh.item()) > 0.05
    assert float(branch0.grad.norm().item()) > 1e-3


def test_clean_stein_basis_target_orthogonality_and_ema_runner() -> None:
    torch.manual_seed(99)
    vocab_words = ["the", "of", "and", "sherlock", "holmes", "inspector", "door", "water"]
    emb_table = {w: F.normalize(torch.randn(32), dim=-1) for w in vocab_words}
    mean_dur = {
        "the": 0.12,
        "of": 0.13,
        "and": 0.16,
        "door": 0.32,
        "water": 0.36,
        "holmes": 0.45,
        "sherlock": 0.52,
        "inspector": 0.62,
    }
    words = []
    sent_ids = []
    onsets = []
    targets_list = []
    rng = np.random.default_rng(99)
    cur_t = 1.0
    for s in range(60):
        s_len = 8
        for _ in range(s_len):
            w = vocab_words[int(rng.integers(0, len(vocab_words)))]
            words.append(w)
            sent_ids.append(s)
            onsets.append(cur_t)
            targets_list.append(emb_table[w])
            dt = mean_dur[w] * float(np.exp(rng.normal(0.0, 0.25)))
            cur_t += dt
        cur_t += 1.5
    # 加入几个单次生僻词验证 LOO 收缩不归零
    for rare_idx in range(4):
        rw = f"rareword{rare_idx}"
        emb_table[rw] = F.normalize(torch.randn(32), dim=-1)
        words.append(rw)
        sent_ids.append(60)
        onsets.append(cur_t)
        targets_list.append(emb_table[rw])
        cur_t += 0.40

    targets = torch.stack(targets_list, dim=0)
    sent_t = torch.tensor(sent_ids, dtype=torch.long)
    onset_t = torch.tensor(onsets, dtype=torch.float32)

    bundle = build_clean_stein_residual_basis(
        words, targets, sent_t, onset_t, torch.device("cpu")
    )
    basis = bundle["basis"].float()
    assert basis.shape == (len(words), 48)
    assert torch.isfinite(basis).all()

    t_c = targets - targets.mean(dim=0, keepdim=True)
    cross_cov = (t_c.t() @ basis) / float(len(words))
    assert float(cross_cov.abs().max().item()) < 5e-3

    bank = torch.stack(list(emb_table.values()), dim=0)
    gram_sqrt = build_vocabulary_gram_sqrt(bank - bank.mean(dim=0, keepdim=True))
    runner = CleanSteinRunningLoss(bundle, gram_sqrt, ema_gamma=0.05)

    idx = torch.arange(64, dtype=torch.long)
    dy = (basis[idx, :32] * 0.1).requires_grad_(True)
    loss_val = runner(dy, idx)
    loss_val.backward()
    assert float(loss_val.item()) > 0.0
    assert runner.initialized
    assert float(dy.grad.norm().item()) > 0.0


@pytest.mark.parametrize(
    "config_path",
    (
        "configs/training/siamese_debiased/libribrain100_salience_additive_excess_null_context.yaml",
        "configs/training/siamese_debiased/libribrain100_salience_additive_stein_excess_context.yaml",
        "configs/training/siamese_debiased/libribrain100_shallow_residual_stein_excess_context.yaml",
        "configs/training/siamese_debiased/libribrain100_salience_additive_excess_null_prefix_revision.yaml",
    ),
)
def test_new_ab_pilot_configs_resolve_cleanly(
    monkeypatch: pytest.MonkeyPatch, config_path: str
) -> None:
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    cfg, _ = build_workflow_config(
        "libribrain100",
        seed=0,
        config_path=config_path,
        exp_tag="ab_pilot_check",
        word_exp_tag="ch2_real_w10",
    )
    siam = cfg["workflow"]["siamese_debiased"]
    assert siam["enabled"] is True
    assert siam["null_space_excess_coherent"] is True
    assert siam["null_space_increment_mode"] is True
