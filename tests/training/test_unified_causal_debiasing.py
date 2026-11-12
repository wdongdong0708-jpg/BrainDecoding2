from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from braindecoding.losses import build_siglip_loss
from braindecoding.models import TemporalGeometryBiasHead, build_brain_embedding_model
from braindecoding.optimizers import build_adamw_for_modules
from braindecoding.training.siamese_debiased_training import (
    compute_causal_cms_mu0_from_encoded,
    compute_euclidean_cross_horizon_loss,
    compute_scaled_shift_null_loss,
    compute_shift_null_loss,
    compute_temporal_bias_distillation_loss,
    encode_from_preextracted_cnn,
    resolve_siamese_debiased_config,
    train_one_epoch_siamese_debiased,
)
from braindecoding.training.word_encoder_training import SentenceBatchSampler
from braindecoding.training.workflow_runner import build_workflow_config


def _build_tiny_prefix_model(*, with_temporal_bias_head: bool = False, eta: float = 0.5):
    positions = torch.tensor(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=torch.float32
    )
    model_cfg = {
        "embedding_dimension": 32,
        "use_transformer": True,
        "conv": {
            "channels": 16,
            "blocks": 1,
            "dropout": 0.0,
            "merger": False,
            "subject_projection": "per_subject",
        },
        "transformer": {
            "depth": 2,
            "heads": 4,
            "dim_head": 8,
            "mlp_multiplier": 2,
            "dropout": 0.0,
            "causal": False,
            "positional_encoding": "rotary",
        },
        "prefix_revision": {
            "enabled": True,
            "horizons": [0, 1, 2],
            "history_radius": None,
            "validation_query_batch_size": 32,
        },
    }
    if with_temporal_bias_head:
        model_cfg["temporal_bias_head"] = {
            "enabled": True,
            "eta": float(eta),
            "hidden_dimension": 16,
            "window_seconds": 1.0,
        }
    return build_brain_embedding_model(
        channel_count=4,
        channel_positions=positions,
        subject_count=2,
        config=model_cfg,
    )


def test_temporal_geometry_bias_head_zero_neural_input_and_strict_future_causality():
    """检验 1：TemporalGeometryBiasHead 严格不接收脑信号参数，且对未来词到达时刻 (> i+r) 扰动零敏感。"""
    sig_q = inspect.signature(TemporalGeometryBiasHead.predict_queries)
    sig_g = inspect.signature(TemporalGeometryBiasHead.predict_group_revisions)
    for forbidden in ("meg", "eeg", "brain", "signal", "z_cnn", "x"):
        assert forbidden not in sig_q.parameters
        assert forbidden not in sig_g.parameters

    torch.manual_seed(7)
    head = TemporalGeometryBiasHead(
        dimension=32,
        config={"hidden_dimension": 24, "window_seconds": 1.0},
    ).eval()
    # 初始化非零读出层以便严格放大任何潜在未来泄漏
    for p in head.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.2)

    onsets_orig = torch.tensor(
        [0.2, 0.6, 1.1, 1.5, 2.2, 2.8, 10.0, 10.5, 11.2], dtype=torch.float32
    )
    groups = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)

    rev_orig = head.predict_group_revisions(onsets_orig, groups, horizons=(0, 1, 2))

    # 对第一句 (group=0, 长度 6) 的词 i=1，分别验证 r=0 (cutoff=1), r=1 (cutoff=2), r=2 (cutoff=3)：
    # 极端篡改 cutoff 之后 (> 1+r) 的所有未来词起始时刻，验证 b_psi[i=1] 差异严格为 0.0！
    for r in (0, 1, 2):
        cutoff = 1 + r
        onsets_mut = onsets_orig.clone()
        onsets_mut[cutoff + 1 : 6] = onsets_mut[cutoff + 1 : 6] + 75.0
        rev_mut = head.predict_group_revisions(onsets_mut, groups, horizons=(0, 1, 2))
        diff = float((rev_mut[r][1] - rev_orig[r][1]).abs().max().item())
        assert diff == 0.0, f"TemporalGeometryBiasHead leaked future onset at r={r}: diff={diff}"

        q_rows = torch.tensor([1], dtype=torch.long)
        q_delays = torch.tensor([r], dtype=torch.long)
        q_orig = head.predict_queries(onsets_orig, groups, q_rows, q_delays)
        q_mut = head.predict_queries(onsets_mut, groups, q_rows, q_delays)
        assert float((q_mut - q_orig).abs().max().item()) == 0.0
        torch.testing.assert_close(q_orig[0], rev_orig[r][1], atol=1e-6, rtol=1e-6)

    # 验证挂载到 BrainEmbeddingModel 后 forward_prefix_revisions 同时对未来脑电与未来时刻保持因果不变性
    model = _build_tiny_prefix_model(with_temporal_bias_head=True, eta=0.5).eval()
    for p in model.temporal_bias_head.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.15)
    x_orig = torch.randn(6, 4, 50)
    subj = torch.zeros(6, dtype=torch.long)
    grp = torch.zeros(6, dtype=torch.long)
    onsets_6 = torch.tensor([0.2, 0.55, 0.95, 1.4, 2.0, 2.7], dtype=torch.float32)

    with torch.no_grad():
        out_orig = model.forward_prefix_revisions(x_orig, subj, grp, onset_seconds=onsets_6)
        x_mut = x_orig.clone()
        onsets_6_mut = onsets_6.clone()
        # 针对词 i=1, r=1 (可见前缀 [0, 1, 2])，极端篡改索引 3, 4, 5 的脑电与时间戳
        x_mut[3:] = 500.0
        onsets_6_mut[3:] = onsets_6_mut[3:] + 90.0
        out_mut = model.forward_prefix_revisions(x_mut, subj, grp, onset_seconds=onsets_6_mut)

    diff_model = float((out_mut[1][1] - out_orig[1][1]).abs().max().item())
    assert diff_model < 1e-6


def test_scaled_shift_null_loss_dimension_balance_and_euclidean_cross_horizon():
    """检验 2：维度平衡零空间正交惩罚 (D/|V|) 在不同词表规模下保持 O(1) 尺度恒定，欧氏稳定项阻断 K0 反向梯度。"""
    torch.manual_seed(42)
    dim = 128
    queries = F.normalize(torch.randn(64, dim), dim=-1)

    scaled_values: dict[int, float] = {}
    unscaled_values: dict[int, float] = {}
    for vocab_size in (50, 500, 2020):
        raw_bank = F.normalize(torch.randn(vocab_size, dim), dim=-1)
        centered_bank = raw_bank - raw_bank.mean(dim=0, keepdim=True)
        scaled_values[vocab_size] = float(
            compute_scaled_shift_null_loss(queries, centered_bank).item()
        )
        unscaled_values[vocab_size] = float(
            compute_shift_null_loss(queries, centered_bank).item()
        )

    # 未缩放旧公式随 |V| 暴涨 ~40 倍，而缩放后 (D/|V|) 比值稳定在 1.0 附近 (0.80 ~ 1.25)
    assert unscaled_values[2020] / unscaled_values[50] > 30.0
    ratio_scaled = scaled_values[2020] / scaled_values[50]
    assert 0.80 <= ratio_scaled <= 1.25
    assert 0.70 <= scaled_values[2020] <= 1.30

    # 验证单一欧氏跨视界平滑项精确等于 2 * (1 - cos) 且对 K0 stop-gradient
    y_k0 = F.normalize(torch.randn(16, dim), dim=-1).detach().requires_grad_(True)
    y_r = F.normalize(torch.randn(16, dim), dim=-1).detach().requires_grad_(True)
    loss_euclid = compute_euclidean_cross_horizon_loss(y_r, y_k0)
    expected_euclid = (2.0 * (1.0 - F.cosine_similarity(y_r, y_k0.detach(), dim=-1))).mean()
    torch.testing.assert_close(loss_euclid, expected_euclid, atol=1e-6, rtol=1e-6)
    loss_euclid.backward()
    assert y_r.grad is not None and float(y_r.grad.norm().item()) > 0.0
    assert y_k0.grad is None


def test_train_one_epoch_tier_a_null_space_and_tier_b_amortized_timing():
    """检验 3：端到端单轮训练覆盖主推档位 A (null_space) 与保底档位 B (amortized_timing) 及 mu_0 提取。"""
    torch.manual_seed(99)
    device = torch.device("cpu")
    n_samples = 16
    dim = 32

    sentence_indices = np.asarray([0] * 8 + [1] * 8, dtype=np.int64)
    subject_indices = torch.tensor([0] * 8 + [1] * 8, dtype=torch.long)
    onsets = torch.tensor(
        [0.3 * i for i in range(8)] + [10.0 + 0.35 * i for i in range(8)],
        dtype=torch.float32,
    )
    targets = F.normalize(torch.randn(n_samples, dim), dim=-1)
    clean_z = F.normalize(torch.randn(n_samples, dim), dim=-1)
    shifted_10 = F.normalize(torch.randn(n_samples, dim), dim=-1)
    shifted_20 = F.normalize(torch.randn(n_samples, dim), dim=-1)

    preextracted_cache = {
        "train_clean_z": clean_z,
        "train_clean_z_pad": F.pad(clean_z, (0, 0, 0, 1)),
        "train_shifted_z": {-10.0: shifted_10, -20.0: shifted_20},
        "train_shifted_z_pad": {
            -10.0: F.pad(shifted_10, (0, 0, 0, 1)),
            -20.0: F.pad(shifted_20, (0, 0, 0, 1)),
        },
        "train_meta": {
            "targets": targets,
            "subject_indices": subject_indices,
            "sentence_indices": torch.from_numpy(sentence_indices),
            "onset_seconds": onsets,
            "words": [f"w{i % 4}" for i in range(n_samples)],
            "event_ids": [f"sub-{i // 8}:run-01:ev-{i}" for i in range(n_samples)],
            "recording_ids": [f"rec-{i // 8}" for i in range(n_samples)],
        },
    }
    raw_vocab = F.normalize(torch.randn(8, dim), dim=-1)
    centered_vocab_bank = raw_vocab - raw_vocab.mean(dim=0, keepdim=True)
    sampler = SentenceBatchSampler(
        sentence_uids=sentence_indices,
        batch_size=8,
        shuffle=False,
        seed=0,
        preserve_complete_groups=True,
    )

    # ---------------- 档位 A: null_space ----------------
    model_a = _build_tiny_prefix_model(with_temporal_bias_head=False)
    for p in model_a.brain_encoder.parameters():
        p.requires_grad_(False)
    cnn_before = {k: v.detach().clone() for k, v in model_a.brain_encoder.state_dict().items()}
    trf_before = {k: v.detach().clone() for k, v in model_a.context_transformer.state_dict().items()}

    loss_mod_a = build_siglip_loss({"initial_temperature": 10.0, "initial_bias": -10.0})
    opt_a = build_adamw_for_modules(
        [model_a, loss_mod_a], {"learning_rate": 1e-3, "weight_decay": 0.01}
    )
    scaler_a = torch.amp.GradScaler("cuda", enabled=False)
    siam_cfg_a = resolve_siamese_debiased_config(
        {
            "siamese_debiased": {
                "enabled": True,
                "debias_mode": "null_space",
                "shift_mode": "preroll",
                "eta_0": 0.0,
                "lambda_null": 10.0,
                "lambda_delta": 0.45,
                "shift_seconds": [-10.0, -20.0],
                "shifts_per_step": 2,
            }
        }
    )
    loss_val_a, seen_a, updates_a, diag_a = train_one_epoch_siamese_debiased(
        model_a,
        loss_mod_a,
        sampler,
        preextracted_cache,
        opt_a,
        scaler_a,
        device,
        {"seed": 0, "amp": False, "revision_cutoff_policy": "all"},
        siam_cfg_a,
        centered_vocab_bank=centered_vocab_bank,
        epoch_index=0,
    )
    assert np.isfinite(loss_val_a)
    assert seen_a == 2 and updates_a == 2
    assert diag_a["debias_mode"] == "null_space"
    assert diag_a["eta_0"] == 0.0
    assert diag_a["shift_null_loss"] > 0.0
    assert diag_a["delta_shift_null_loss"] > 0.0
    for k, v in model_a.brain_encoder.state_dict().items():
        torch.testing.assert_close(v, cnn_before[k], atol=0.0, rtol=0.0)
    assert any(
        not torch.equal(v, trf_before[k])
        for k, v in model_a.context_transformer.state_dict().items()
    )

    encoded_a = encode_from_preextracted_cnn(
        model_a, sampler, clean_z, preextracted_cache["train_meta"], device, amp=False
    )
    mu0_payload = compute_causal_cms_mu0_from_encoded(encoded_a, source="train")
    assert mu0_payload["version"] == "causal_cms_mu0_v1"
    assert mu0_payload["event_count"] == n_samples
    assert mu0_payload["predictions"].shape == (dim,)
    assert set(mu0_payload["revision_predictions"].keys()) == {0, 1, 2}

    # ---------------- 档位 B: amortized_timing ----------------
    model_b = _build_tiny_prefix_model(with_temporal_bias_head=True, eta=0.5)
    for p in model_b.brain_encoder.parameters():
        p.requires_grad_(False)

    # 先独立验证主干前向对 b_psi 严格施加了 detach (sg(b_psi))，主 SigLIP 损失不会反向更新 temporal_bias_head
    model_b.train()
    model_b.zero_grad(set_to_none=True)
    x_dummy = torch.randn(8, 4, 50)
    rev_queries_b, rev_rows_b, _rev_delays_b = model_b.forward_prefix_revision_training(
        x_dummy,
        torch.zeros(8, dtype=torch.long),
        torch.zeros(8, dtype=torch.long),
        cutoff_policy="all",
        onset_seconds=onsets[:8],
    )
    pure_main_loss = (rev_queries_b * targets[rev_rows_b]).sum()
    pure_main_loss.backward()
    assert any(
        p.grad is not None and float(p.grad.abs().max().item()) > 0.0
        for p in model_b.context_transformer.parameters()
    )
    assert all(
        (p.grad is None or float(p.grad.abs().max().item()) == 0.0)
        for p in model_b.temporal_bias_head.parameters()
    )
    model_b.zero_grad(set_to_none=True)

    # 再验证完整 Tier B 训练循环通过 MSE 蒸馏项正常更新 temporal_bias_head 与 context_transformer
    head_before = {
        k: v.detach().clone() for k, v in model_b.temporal_bias_head.state_dict().items()
    }
    loss_mod_b = build_siglip_loss({"initial_temperature": 10.0, "initial_bias": -10.0})
    opt_b = build_adamw_for_modules(
        [model_b, loss_mod_b], {"learning_rate": 5e-3, "weight_decay": 0.01}
    )
    scaler_b = torch.amp.GradScaler("cuda", enabled=False)
    siam_cfg_b = resolve_siamese_debiased_config(
        {
            "model": {
                "temporal_bias_head": {
                    "enabled": True,
                    "eta": 0.4,
                    "hidden_dimension": 16,
                    "window_seconds": 1.0,
                }
            },
            "siamese_debiased": {
                "enabled": True,
                "debias_mode": "amortized_timing",
                "shift_mode": "preroll",
                "lambda_distill": 0.5,
                "lambda_null": 10.0,
                "lambda_delta": 0.45,
                "shift_seconds": [-10.0, -20.0],
                "shifts_per_step": 2,
            },
        }
    )
    assert pytest.approx(float(siam_cfg_b["eta_0"]), abs=1e-6) == 0.4
    assert int(siam_cfg_b["temporal_bias_hidden_dim"]) == 16
    loss_val_b, seen_b, updates_b, diag_b = train_one_epoch_siamese_debiased(
        model_b,
        loss_mod_b,
        sampler,
        preextracted_cache,
        opt_b,
        scaler_b,
        device,
        {"seed": 0, "amp": False, "revision_cutoff_policy": "all"},
        siam_cfg_b,
        centered_vocab_bank=centered_vocab_bank,
        epoch_index=0,
    )
    assert np.isfinite(loss_val_b)
    assert seen_b == 2 and updates_b == 2
    assert diag_b["debias_mode"] == "amortized_timing"
    assert diag_b["shift_null_loss"] > 0.0
    assert diag_b["delta_shift_null_loss"] > 0.0
    assert diag_b["distill_loss"] > 0.0
    assert any(
        not torch.equal(v, head_before[k])
        for k, v in model_b.temporal_bias_head.state_dict().items()
    )
    # 单独验证 compute_temporal_bias_distillation_loss 对目标施加 detach
    y_sh_leaf = torch.randn(4, dim, requires_grad=True)
    b_leaf = torch.randn(4, dim, requires_grad=True)
    d_loss = compute_temporal_bias_distillation_loss(b_leaf, y_sh_leaf)
    d_loss.backward()
    assert b_leaf.grad is not None and float(b_leaf.grad.norm().item()) > 0.0
    assert y_sh_leaf.grad is None


@pytest.mark.parametrize(
    ("yaml_relpath", "expected_mode"),
    [
        (
            "configs/training/siamese_debiased/pallier2025_causal_null_space_prefix_revision.yaml",
            "null_space",
        ),
        (
            "configs/training/siamese_debiased/pallier2025_amortized_timing_prefix_revision.yaml",
            "amortized_timing",
        ),
        (
            "configs/training/siamese_debiased/chineseeeg2_causal_null_space_prefix_revision.yaml",
            "null_space",
        ),
        (
            "configs/training/siamese_debiased/chineseeeg2_amortized_timing_prefix_revision.yaml",
            "amortized_timing",
        ),
    ],
)
def test_new_causal_yaml_configs_build_workflow_cleanly(
    yaml_relpath: str, expected_mode: str
):
    """确认新增的 4 份法汉因果 YAML 配置无参数遮蔽/量纲陷阱，并严格呈递进消融关系。"""
    import yaml

    root = Path(__file__).resolve().parents[2]
    yaml_path = root / yaml_relpath
    raw_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    raw_siam = raw_yaml.get("siamese_debiased") or {}
    assert "lambda_delta_null" not in raw_siam, (
        f"{yaml_relpath} must not define lambda_delta_null alongside lambda_delta"
    )

    cfg, _meta = build_workflow_config(config_path=str(yaml_path), seed=0)
    siam = cfg["siamese_debiased"]
    assert siam["enabled"] is True
    assert siam["debias_mode"] == expected_mode
    assert siam["shift_mode"] == "preroll"
    assert float(siam["preroll_seconds"]) == 45.0
    assert list(siam["shift_seconds"]) == [-12.0, -19.0, -27.0, -36.0]
    assert int(siam["shifts_per_step"]) == 2
    assert pytest.approx(float(siam["lambda_null"]), abs=1e-6) == 10.0
    assert pytest.approx(float(siam["lambda_delta"]), abs=1e-6) == 0.45
    assert pytest.approx(float(siam["beta_delta_cm"]), abs=1e-6) == 0.0
    if expected_mode == "null_space":
        assert pytest.approx(float(siam["eta_0"]), abs=1e-6) == 0.0
        assert pytest.approx(float(siam["lambda_distill"]), abs=1e-6) == 0.0
        assert "temporal_bias_head" not in (cfg.get("model") or {})
    elif expected_mode == "amortized_timing":
        assert pytest.approx(float(siam["eta_0"]), abs=1e-6) == 0.4
        assert pytest.approx(float(siam["eta_train"]), abs=1e-6) == 0.4
        assert pytest.approx(float(siam["lambda_distill"]), abs=1e-6) == 0.5
        assert int(siam["temporal_bias_hidden_dim"]) == 256
        assert pytest.approx(float(siam["s3_cms_eta"]), abs=1e-6) == 0.0
        tb_cfg = cfg["model"].get("temporal_bias_head") or {}
        assert tb_cfg.get("enabled") is True
        assert pytest.approx(float(tb_cfg.get("eta", 0.0)), abs=1e-6) == 0.40
        assert int(tb_cfg.get("hidden_dimension", 0)) == 256
        assert pytest.approx(float(tb_cfg.get("window_seconds", 0.0)), abs=1e-6) == 1.0
