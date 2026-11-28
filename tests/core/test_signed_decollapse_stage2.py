"""第一阶梯（Op3A-v2 / SignedDeCollapseAdditiveEncoder）纯脑侧解坍缩与在线因果校准单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from braindecoding.evaluation.duration_shortcut_diagnostics import (
    SplicedRealOnsetDataset,
    compute_representation_health_and_noise_anatomy,
    format_diagnostics_table,
)
from braindecoding.losses import build_siglip_loss
from braindecoding.models import (
    MultiBinResConvEncoder,
    SignedDeCollapseAdditiveEncoder,
    apply_causal_session_calibration,
    build_brain_embedding_model,
)
from braindecoding.optimizers import build_adamw_for_modules
from braindecoding.training.debiased_losses import (
    build_centered_vocabulary_bank,
    build_vocabulary_gram_sqrt,
    compute_shifted_lag_decorrelation_loss,
    resolve_siamese_debiased_config,
)
from braindecoding.training.debiased_padded_forward import (
    _apply_siamese_shared_dropout,
)
from braindecoding.training.debiased_preextracted_eval import (
    encode_from_preextracted_cnn,
)
from braindecoding.training.debiased_preextraction import (
    preextract_frozen_cnn_features,
)
from braindecoding.training.siamese_debiased_training import (
    train_one_epoch_siamese_debiased,
)
from braindecoding.training.word_encoder_training import (
    SentenceBatchSampler,
    encode_loader,
)
from braindecoding.training.workflow_runner import build_workflow_config


def _tiny_multibin_conv_cfg(dim: int = 32) -> dict:
    return {
        "architecture": "multibin_resconv",
        "merger_channels": 16,
        "merger_position_dimension": 32,
        "merger_per_subject": True,
        "merger_dropout": 0.0,
        "merger_per_sample_dropout": False,
        "hidden": 16,
        "depth": 2,
        "kernel_size": 3,
        "dilation_growth": 2,
        "dilation_period": 2,
        "batch_norm": True,
        "glu_every": 2,
        "num_temporal_bins": 5,
        "bin_attention_hidden": 8,
        "mlp_dim": dim,
        "mlp_hidden": dim * 2,
        "mlp_depth": 2,
        "mlp_dropout": 0.0,
    }


def test_multibin_encode_with_pre_head_matches_forward_and_bins() -> None:
    torch.manual_seed(7)
    positions = np.linspace(0.0, 1.0, 16 * 2, dtype=np.float32).reshape(16, 2)
    enc = MultiBinResConvEncoder(
        positions, subject_count=2, output_dimension=32, config=_tiny_multibin_conv_cfg(32)
    )
    enc.eval()
    signals = torch.randn(5, 16, 50)
    subjects = torch.tensor([0, 1, 0, 1, 0], dtype=torch.long)

    with torch.no_grad():
        z_ph, h_pre, h_bins = enc.encode_with_pre_head(signals, subjects)
        z_bin, h_bins_2 = enc.encode_with_bins(signals, subjects)
        z_fwd = enc(signals, subjects)

    assert enc.pre_head_dimension == 32
    assert enc.bin_feature_dimension == 5 * 16
    assert z_ph.shape == (5, 32)
    assert h_pre.shape == (5, 32)
    assert h_bins.shape == (5, 80)
    assert torch.allclose(z_ph, z_bin, atol=1e-6)
    assert torch.allclose(z_ph, z_fwd, atol=1e-6)
    assert torch.allclose(h_bins, h_bins_2, atol=1e-6)
    assert torch.allclose(enc.head_out(h_pre), z_ph, atol=1e-6)


def test_causal_session_calibration_strict_causality_shrinkage_and_drift_removal() -> None:
    torch.manual_seed(13)
    dim = 24
    # 两条独立录音（各自拥有恒定头位偏移 + 零均值信号），验证严格因果性、录音隔离与偏移清零
    n_rec = 120
    mu_train = torch.randn(dim) * 0.2
    offset_a = torch.randn(dim) * 1.5
    offset_b = torch.randn(dim) * -1.2
    sig_a = mu_train + offset_a + 0.05 * torch.randn(n_rec, dim)
    sig_b = mu_train + offset_b + 0.05 * torch.randn(n_rec, dim)
    feats = torch.cat([sig_a, sig_b], dim=0)
    recs = ["recA"] * n_rec + ["recB"] * n_rec
    subjs = [0] * (2 * n_rec)
    onsets = list(range(n_rec)) + list(range(n_rec))

    cal = apply_causal_session_calibration(
        feats,
        recording_ids=recs,
        subject_indices=subjs,
        onset_seconds=onsets,
        train_mean=mu_train,
        shrinkage_m0=50.0,
        lag_gap=2,
        alpha=1.0,
        renormalize_unit_sphere=False,
    )

    # 1. 录首 k=0, 1 严格零修改（无 j <= k-2 过去词）
    assert torch.allclose(cal[0], feats[0], atol=1e-6)
    assert torch.allclose(cal[1], feats[1], atol=1e-6)
    assert torch.allclose(cal[n_rec], feats[n_rec], atol=1e-6)
    assert torch.allclose(cal[n_rec + 1], feats[n_rec + 1], atol=1e-6)

    # 2. 严格因果 + 排除 lag-1：大幅扰动 recA[k-1] 与 recA[k+1:] 绝不改变 cal[k]
    k_check = 40
    feats_mut = feats.clone()
    feats_mut[k_check - 1] += 100.0
    feats_mut[k_check + 1 : n_rec] -= 100.0
    feats_mut[n_rec:] += 50.0  # 另一条录音完全隔离
    cal_mut = apply_causal_session_calibration(
        feats_mut,
        recording_ids=recs,
        subject_indices=subjs,
        onset_seconds=onsets,
        train_mean=mu_train,
        shrinkage_m0=50.0,
        lag_gap=2,
        alpha=1.0,
        renormalize_unit_sphere=False,
    )
    assert torch.allclose(cal_mut[k_check], cal[k_check], atol=1e-6)

    # 3. 录音后段偏移被大幅清除（远小于原始恒定偏移）
    raw_err = float((feats[n_rec - 1] - mu_train).norm().item())
    cal_err = float((cal[n_rec - 1] - mu_train).norm().item())
    assert cal_err < 0.4 * raw_err

    # 4. 乱序行按 onset_seconds 稳定排序后结果逐位等价
    perm = torch.randperm(len(feats))
    cal_perm = apply_causal_session_calibration(
        feats[perm],
        recording_ids=[recs[i] for i in perm.tolist()],
        subject_indices=[subjs[i] for i in perm.tolist()],
        onset_seconds=[onsets[i] for i in perm.tolist()],
        train_mean=mu_train,
        shrinkage_m0=50.0,
        lag_gap=2,
        alpha=1.0,
        renormalize_unit_sphere=False,
    )
    assert torch.allclose(cal_perm, cal[perm], atol=1e-5)


@pytest.mark.parametrize("horizon_conditioned", (False, True))
@pytest.mark.parametrize("use_h_bins_pca", (False, True))
def test_signed_decollapse_encoder_linearity_parity_cone_lock_and_step0_grad(
    horizon_conditioned: bool,
    use_h_bins_pca: bool,
) -> None:
    torch.manual_seed(31)
    dim = 32
    bin_dim = 80
    pca_dim = 16
    cfg = {
        "architecture": "signed_decollapse_additive",
        "heads": 8,
        "head_dimension": 8,
        "max_relative_distance": 12,
        "residual_max_norm": 0.75,
        "attention_dropout": 0.1,
        "include_self_offset": False,
        "horizon_conditioned": horizon_conditioned,
        "num_horizons": 3,
        "horizon_readout_rank": 16,
        "self_decollapse_rank": 24,
        "self_decollapse_dropout": 0.35,
        "pre_head_dimension": dim,
        "use_h_bins_pca": use_h_bins_pca,
        "h_bins_pca_dim": pca_dim,
        "bin_feature_dimension": bin_dim,
        "session_calibration": {
            "enabled": True,
            "shrinkage_m0": 50.0,
            "lag_gap": 2,
            "alpha": 1.0,
        },
    }
    enc = SignedDeCollapseAdditiveEncoder(dim, cfg)
    u0_text = F.normalize(torch.randn(dim), dim=-1)
    enc.set_text_cone_axis(u0_text)

    n_rows = 12
    z = F.normalize(0.4 * u0_text.unsqueeze(0) + 0.6 * torch.randn(n_rows, dim), dim=-1)
    h_pre = torch.randn(n_rows, dim)
    h_bins = torch.randn(n_rows, bin_dim)
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    recs = ["r0"] * 6 + ["r1"] * 6
    subjs = [0] * n_rows
    onsets = list(range(6)) + list(range(6))

    enc.set_clean_input_mean(z.mean(dim=0))
    enc.set_pre_head_mean(h_pre.mean(dim=0))
    if use_h_bins_pca:
        enc.fit_h_bins_pca(
            h_bins,
            recording_ids=recs,
            subject_indices=subjs,
            onset_seconds=onsets,
        )
        h_bins_proj = enc.project_h_bins(h_bins)
        assert h_bins_proj.shape == (n_rows, pca_dim)
        # 检查标量 RMS 缩放生效（不破坏非均匀特征值分布）
        assert float(h_bins_proj.pow(2).mean().sqrt().item()) == pytest.approx(1.0, rel=0.25)
    else:
        h_bins_proj = None

    z_cal, self_feat = enc.calibrate_stage1_features(
        z,
        h_pre,
        h_bins_proj,
        recording_ids=recs,
        subject_indices=subjs,
        onset_seconds=onsets,
    )
    assert self_feat is not None
    assert self_feat.shape == (n_rows, dim + (pca_dim if use_h_bins_pca else 0))

    # 1. 验证双极初始化：前一半头偏正权重，后一半头偏负权重
    w_tab = enc._compute_signed_weight_table()
    w_first = w_tab[0] if horizon_conditioned else w_tab
    assert float(w_first[:4].mean().item()) > 0.0
    assert float(w_first[4:].mean().item()) < 0.0

    # 2. Siamese 多分支共享 Dropout + step-0 满秩非零梯度
    padded = z_cal.view(3, 4, dim)
    self_padded = self_feat.view(3, 4, -1)
    mask = torch.ones((3, 4), dtype=torch.bool)
    enc.train()
    out_tr = enc.forward_padded(
        torch.cat([padded, padded * 0.9, padded * 0.8], dim=0),
        mask.repeat(3, 1),
        self_padded=torch.cat([self_padded, self_padded * 0.9, self_padded * 0.8], dim=0),
        num_branches=3,
        synchronize_dropout_rng=True,
        shared_dropout_fn=_apply_siamese_shared_dropout,
    )
    (out_tr * torch.randn_like(out_tr)).sum().backward()
    for name, param in enc.named_parameters():
        if name == "horizon_down":
            # horizon_up 严格零初始化，step-0 对 horizon_down 的梯度天然为 0（step-1 起激活）
            continue
        assert param.grad is not None, f"缺少 step-0 梯度：{name}"
        assert torch.isfinite(param.grad).all(), f"非有限梯度：{name}"
        assert float(param.grad.abs().max().item()) > 0.0, f"零 step-0 梯度：{name}"

    # 3. 验证 Δ=0 严格线性（关闭邻居输出后，g_self 对输入严格线性、无 sigmoid）
    enc.eval()
    d_self = int(self_feat.shape[-1])
    h1 = torch.randn(2, 1, d_self)
    h2 = torch.randn(2, 1, d_self)
    m_single = torch.ones((2, 1, 1), dtype=torch.float32)
    with torch.no_grad():
        mu_s = enc._self_feature_mean_vector(h1.device).view(1, 1, d_self)
        g1 = enc._compute_linear_self_decollapse(
            h1 + mu_s, m_single, eff_branches=1, g_per_branch=2, max_len=1, work_dtype=torch.float32
        )
        g2 = enc._compute_linear_self_decollapse(
            h2 + mu_s, m_single, eff_branches=1, g_per_branch=2, max_len=1, work_dtype=torch.float32
        )
        g12 = enc._compute_linear_self_decollapse(
            0.3 * h1 + 0.7 * h2 + mu_s,
            m_single,
            eff_branches=1,
            g_per_branch=2,
            max_len=1,
            work_dtype=torch.float32,
        )
    assert torch.allclose(g12, 0.3 * g1 + 0.7 * g2, atol=1e-5)

    # 4. 验证 forward == forward_padded == 单目标剪枝 逐位一致 + 文本锥轴严格锁定
    if horizon_conditioned and enc.horizon_up is not None:
        with torch.no_grad():
            enc.horizon_up.normal_(mean=0.0, std=0.05)
    y_fwd = enc(z_cal, groups, self_features=self_feat)
    y_pad = enc.forward_padded(padded, mask, self_padded=self_padded, num_branches=1)
    assert torch.allclose(y_fwd, y_pad.reshape(n_rows, dim), atol=1e-5)

    tg = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    tc = torch.tensor([1, 3, 2, 0], dtype=torch.long)
    y_pruned = enc.forward_padded(
        padded, mask, self_padded=self_padded, target_group_idx=tg, target_col_idx=tc
    )
    assert torch.allclose(y_pruned, y_pad[tg, tc], atol=1e-5)
    assert torch.allclose((y_fwd * u0_text).sum(dim=-1), (z_cal * u0_text).sum(dim=-1), atol=1e-5)

    # 5. 解耦权重衰减分组（self_decollapse 独立分组且核 logits 豁免）
    opt = build_adamw_for_modules(
        [enc],
        {
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "self_decollapse_weight_decay": 0.03,
            "decoupled_weight_decay": True,
        },
    )
    wd_map = {round(g["weight_decay"], 4): len(g["params"]) for g in opt.param_groups}
    assert set(wd_map.keys()) == {0.0, 0.01, 0.03}
    assert wd_map[0.03] == 2  # self_decollapse_down & self_decollapse_up


def test_shifted_lag_decorrelation_loss_behavior_and_straight_through() -> None:
    torch.manual_seed(41)
    # 强 lag-1 相关序列（相邻词高度相似） vs 独立白噪声序列
    sent = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    base = torch.randn(12, 32)
    corr_seq = base.clone()
    for i in range(1, 12):
        if sent[i] == sent[i - 1]:
            corr_seq[i] = 0.85 * corr_seq[i - 1] + 0.15 * base[i]
    corr_seq = F.normalize(corr_seq, dim=-1)
    corr_seq.requires_grad_(True)

    # 模拟多分支直通：分支 0 可导，分支 1..2 no_grad
    y_br = torch.stack(
        [corr_seq, corr_seq.detach() * 0.9 + 0.1, corr_seq.detach() * 0.8 - 0.1],
        dim=0,
    )
    loss_corr = compute_shifted_lag_decorrelation_loss(y_br, sent, max_lag=3)
    assert float(loss_corr.item()) > 0.3
    loss_corr.backward()
    assert corr_seq.grad is not None
    assert torch.isfinite(corr_seq.grad).all()
    assert float(corr_seq.grad.abs().max().item()) > 0.0

    # 反相关/跨句边界不串扰：单字句序列损失严格为 0
    singleton_sent = torch.arange(12, dtype=torch.long)
    loss_single = compute_shifted_lag_decorrelation_loss(
        corr_seq.detach().unsqueeze(0), singleton_sent, max_lag=3
    )
    assert float(loss_single.item()) == pytest.approx(0.0, abs=1e-7)


class _ToyMegDataset(Dataset):
    """用于验证 preextract / train_one_epoch / encode_loader 逐位等价的最小确定性 MEG 数据集。"""

    def __init__(self, *, n_sentences: int = 8, sent_len: int = 6, dim: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.channel_count = 16
        self.subject_count = 2
        self.channel_positions = np.linspace(0.0, 1.0, 16 * 2, dtype=np.float32).reshape(16, 2)
        self.sampling_rate_hz = 50.0
        self.baseline_seconds = 0.0
        self.causal_highpass_seconds = 0.0
        self.causal_highpass_mode = "butterworth_iir"
        self.causal_highpass_hz = 0.5
        self.causal_highpass_order = 2
        self.clamp = 20.0
        self.temporal_shift_seconds = 0.0
        self.temporal_shift_samples = 0
        self.shift_mode = "preroll"
        self.preroll_seconds = 45.0

        vocab = [f"w{i}" for i in range(12)]
        self.embedding_map = {
            w: (v / np.linalg.norm(v)).astype(np.float32)
            for w, v in ((w, rng.standard_normal(dim)) for w in vocab)
        }
        self._embedding_tensors = {
            w: torch.from_numpy(v.copy()) for w, v in self.embedding_map.items()
        }

        n_rows = n_sentences * sent_len
        self._signals_clean = rng.standard_normal((n_rows, 16, 50)).astype(np.float32)
        self._words = [vocab[(i * 5 + i // sent_len) % len(vocab)] for i in range(n_rows)]
        self._event_ids = [f"ev-{i:04d}" for i in range(n_rows)]
        self._recording_ids = [
            "rec-0" if i < n_rows // 2 else "rec-1" for i in range(n_rows)
        ]
        self._subject_index_tensors = torch.tensor(
            [0 if i < n_rows // 2 else 1 for i in range(n_rows)], dtype=torch.long
        )
        self._sentence_index_tensors = torch.tensor(
            [i // sent_len for i in range(n_rows)], dtype=torch.long
        )
        self._onset_second_tensors = torch.tensor(
            [(i % (n_rows // 2)) * 0.4 for i in range(n_rows)], dtype=torch.float32
        )
        self.sentence_indices = self._sentence_index_tensors.numpy()
        self.table = pd.DataFrame(
            {
                "event_id": self._event_ids,
                "normalized_word": self._words,
                "subject_id": [f"sub-{int(s):02d}" for s in self._subject_index_tensors.tolist()],
                "recording_id": self._recording_ids,
                "onset_seconds": self._onset_second_tensors.numpy(),
                "runtime_context_uid": [f"ctx-{int(s)}" for s in self.sentence_indices],
            }
        )

    def __len__(self) -> int:
        return len(self._words)

    def __getitem__(self, idx: int) -> dict:
        shift = int(round(self.temporal_shift_seconds))
        sig = np.roll(self._signals_clean[idx], shift, axis=-1) if shift != 0 else self._signals_clean[idx]
        w = self._words[idx]
        return {
            "meg": torch.from_numpy(sig.copy()),
            "text_embedding": self._embedding_tensors[w],
            "subject_index": self._subject_index_tensors[idx],
            "sentence_index": self._sentence_index_tensors[idx],
            "onset_seconds": self._onset_second_tensors[idx],
            "word": w,
            "event_id": self._event_ids[idx],
            "recording_id": self._recording_ids[idx],
        }


@pytest.mark.parametrize("stage_kind", ("context", "prefix_revision"))
def test_end_to_end_preextract_train_and_encode_loader_exact_parity(
    stage_kind: str,
) -> None:
    torch.manual_seed(53)
    dim = 32
    train_ds = _ToyMegDataset(n_sentences=8, sent_len=5, dim=dim, seed=1)
    val_ds = _ToyMegDataset(n_sentences=6, sent_len=5, dim=dim, seed=2)
    is_rev = stage_kind == "prefix_revision"

    model_cfg = {
        "embedding_dimension": dim,
        "use_transformer": True,
        "context_mode": "grouped",
        "conv": _tiny_multibin_conv_cfg(dim),
        "transformer": {
            "architecture": "signed_decollapse_additive",
            "heads": 8,
            "head_dimension": 8,
            "max_relative_distance": 12,
            "residual_max_norm": 0.75,
            "attention_dropout": 0.1,
            "include_self_offset": False,
            "horizon_conditioned": is_rev,
            "num_horizons": 3,
            "horizon_readout_rank": 16,
            "self_decollapse_rank": 16,
            "self_decollapse_dropout": 0.25,
            "use_h_bins_pca": True,
            "h_bins_pca_dim": 16,
            "session_calibration": {
                "enabled": True,
                "shrinkage_m0": 5.0,
                "lag_gap": 2,
                "alpha": 1.0,
            },
        },
    }
    if is_rev:
        model_cfg["prefix_revision"] = {
            "enabled": True,
            "history_radius": None,
            "horizons": [0, 1, 2],
            "separate_horizon_losses": True,
            "validation_query_batch_size": 32,
        }

    device = torch.device("cpu")
    model = build_brain_embedding_model(
        train_ds.channel_count, train_ds.channel_positions, train_ds.subject_count, model_cfg
    ).to(device)
    for p in model.brain_encoder.parameters():
        p.requires_grad = False

    vocab = sorted(train_ds.embedding_map.keys())
    vocab_embs = torch.stack([train_ds._embedding_tensors[w] for w in vocab], dim=0)
    model.context_transformer.set_text_cone_axis(
        F.normalize(F.normalize(vocab_embs, dim=-1).mean(dim=0), dim=-1)
    )

    bundle = preextract_frozen_cnn_features(
        model,
        train_ds,
        val_ds,
        dataset="libribrain100",
        subset_tag="toy",
        word_ckpt_sha256="",
        window_seconds=1.0,
        shift_seconds=[-10.0, -20.0],
        shift_mode="preroll",
        device=device,
        batch_size=16,
        amp=False,
        cache_to_disk=False,
        show_progress=False,
    )
    assert bundle["session_calibrated"] is True
    assert bundle["train_clean_self"] is not None
    assert bundle["train_clean_self"].shape == (len(train_ds), dim + 16)
    assert bundle["val_clean_self"].shape == (len(val_ds), dim + 16)

    centered_bank = build_centered_vocabulary_bank(train_ds, vocab, device, freq_alpha=0.5)
    bundle["vocab_gram_sqrt"] = build_vocabulary_gram_sqrt(centered_bank)

    train_sampler = SentenceBatchSampler(
        train_ds.table["runtime_context_uid"].tolist(),
        batch_size=20,
        shuffle=True,
        seed=0,
        preserve_complete_groups=True,
    )
    val_sampler = SentenceBatchSampler(
        val_ds.table["runtime_context_uid"].tolist(),
        batch_size=15,
        shuffle=False,
        seed=0,
        preserve_complete_groups=True,
    )
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=0)

    loss_module = build_siglip_loss({"name": "siglip"}).to(device)
    opt = build_adamw_for_modules(
        [model, loss_module],
        {
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "self_decollapse_weight_decay": 0.03,
            "decoupled_weight_decay": True,
        },
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    siamese_cfg = resolve_siamese_debiased_config(
        {
            "siamese_debiased": {
                "enabled": True,
                "debias_mode": "null_space",
                "shift_mode": "preroll",
                "eta_0": 0.0,
                "eta_train": 0.0,
                "lambda_null": 60.0,
                "lambda_delta": 10.0 if is_rev else 0.0,
                "null_space_coherent_mean": True,
                "null_space_excess_coherent": True,
                "null_space_excess_tolerance": 0.0,
                "null_space_increment_mode": True,
                "lambda_shift_lag_decorr": 2.0,
                "shift_lag_decorr_max_lag": 3,
                "shift_seconds": [-10.0, -20.0],
                "shifts_per_step": 2,
            }
        }
    )

    mean_loss, updates, batches, diag = train_one_epoch_siamese_debiased(
        model,
        loss_module,
        train_sampler,
        bundle,
        opt,
        scaler,
        device,
        {"amp": False, "seed": 0, "revision_cutoff_policy": "all"},
        siamese_cfg,
        centered_vocab_bank=centered_bank,
        epoch_index=0,
        show_progress=False,
    )
    assert np.isfinite(mean_loss)
    assert updates == batches and updates > 0
    assert diag["lambda_shift_lag_decorr"] == pytest.approx(2.0)
    assert np.isfinite(diag["shift_lag_loss"])

    # 验证 encode_from_preextracted_cnn 与 encode_loader（走在线会话校准）逐位等价！
    enc_fast = encode_from_preextracted_cnn(
        model,
        val_sampler,
        bundle["val_clean_z"],
        bundle["val_meta"],
        device,
        amp=False,
        show_progress=False,
    )
    enc_loader = encode_loader(
        model, val_loader, device, amp=False, show_progress=False
    )
    assert torch.allclose(enc_fast["predictions"], enc_loader["predictions"], atol=1e-5)
    if is_rev:
        for d in (0, 1, 2):
            assert torch.allclose(
                enc_fast["revision_predictions"][d],
                enc_loader["revision_predictions"][d],
                atol=1e-5,
            )

    # 验证 SplicedRealOnsetDataset 通过 encode_loader 正常运行且无崩溃
    donor_idx = np.arange(len(val_ds) - 1, -1, -1, dtype=np.int64)
    sp_groups = np.repeat(np.arange(len(val_ds) // 5, dtype=np.int64), 5)
    sp_ds = SplicedRealOnsetDataset(
        val_ds,
        donor_idx,
        sp_groups,
        np.ones(len(val_ds), dtype=bool),
        mode="shuffled",
        min_gap_seconds=10.0,
    )
    sp_sampler = SentenceBatchSampler(
        sp_ds.table["spliced_context_uid"].tolist(),
        batch_size=15,
        shuffle=False,
        seed=0,
        preserve_complete_groups=True,
    )
    sp_loader = DataLoader(sp_ds, batch_sampler=sp_sampler, num_workers=0)
    enc_sp = encode_loader(model, sp_loader, device, amp=False, show_progress=False)
    assert enc_sp["predictions"].shape == (len(val_ds), dim)
    assert torch.isfinite(enc_sp["predictions"]).all()


def test_representation_health_diagnostics_and_part_d_formatting() -> None:
    torch.manual_seed(61)
    dim = 32
    n = 160
    words = [f"w{i % 10}" for i in range(n)]
    recs = ["r0"] * (n // 2) + ["r1"] * (n // 2)
    sents = np.repeat(np.arange(16, dtype=np.int64), 10)
    onsets = np.concatenate([np.arange(n // 2) * 2.0, np.arange(n // 2) * 2.0])
    word_vecs = {w: F.normalize(torch.randn(dim), dim=-1) for w in set(words)}
    noise_scale = 0.5 / float(dim**0.5)
    preds = torch.stack(
        [F.normalize(word_vecs[w] + noise_scale * torch.randn(dim), dim=-1) for w in words],
        dim=0,
    )
    health = compute_representation_health_and_noise_anatomy(
        preds, words, recs, sents, onsets, min_gap_seconds=10.0
    )
    assert health["same_word_pair_count"] > 100
    assert health["eff_rank_signal"] > 1.5
    assert health["eff_rank_total"] > health["eff_rank_signal"]
    assert health["r_within_zero_overlap"] > 0.5
    assert np.isfinite(health["lag1_corr"]) and np.isfinite(health["rec30_diff_word_corr"])

    fake_report = {
        "dataset": "libribrain100",
        "split": "test",
        "evaluated_events": n,
        "full_vocab_size": 100,
        "duration_oracles_n50": {
            "acoustic_duration": {"macro_top10": 0.2, "macro_top1": 0.02},
            "inter_onset_interval": {"macro_top10": 0.2, "macro_top1": 0.02},
        },
        "spliced_audits": {},
        "summary_table": [
            {
                "model": "T3_Ch2_Op3Av2",
                "clean_n50_mac10": 0.55,
                "clean_n50_mac1": 0.14,
                "clean_fv_mic10": 0.16,
                "clean_fv_mac10": 0.11,
                "clean_fv_med": 900.0,
                "clean_fv_utop1": 420,
                "eff_rank_signal": health["eff_rank_signal"],
                "eff_rank_total": health["eff_rank_total"],
                "r_within_zero_overlap": health["r_within_zero_overlap"],
                "lag1_corr": health["lag1_corr"],
                "lag2_corr": health["lag2_corr"],
                "lag5_corr": health["lag5_corr"],
                "rec30_diff_word_corr": health["rec30_diff_word_corr"],
                "fig2b_n50_dur_err_red_ms": 1.5,
                "fig2b_n50_ioi_err_red_ms": 1.2,
                "fig2b_n50_within_r": 0.01,
                "clean_n50_fast_slow_shift_ms": 2.0,
                "fig2a_n50_acc_vs_dursd_r": 0.0,
                "fig2a_n50_acc_vs_oracle_r": 0.0,
                "splice_same_word_n50_mac10": 0.50,
                "splice_shuffled_n50_mac10": 0.47,
                "fig2b_fv_dur_err_red_ms": 1.5,
                "fig2b_fv_ioi_err_red_ms": 1.2,
                "fig2b_fv_within_r": 0.01,
                "clean_fv_fast_slow_shift_ms": 2.0,
                "splice_same_word_fv_mic10": 0.10,
                "splice_same_word_fv_med": 1700.0,
                "splice_shuffled_fv_mic10": 0.08,
                "splice_shuffled_fv_med": 2200.0,
            }
        ],
    }
    table_text = format_diagnostics_table(fake_report)
    assert "[Part D] Representation Health & Noise Anatomy" in table_text
    assert "T3_Ch2_Op3Av2" in table_text


def test_ladder1_configs_and_colab_stage_h_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    for stage, path_str, expect_horizon, expect_lam_delta in (
        (
            "context",
            "configs/training/siamese_debiased/libribrain100_multibin_signed_decollapse_context.yaml",
            False,
            0.0,
        ),
        (
            "prefix_revision",
            "configs/training/siamese_debiased/libribrain100_multibin_signed_decollapse_prefix_revision.yaml",
            True,
            10.0,
        ),
    ):
        cfg, _ = build_workflow_config(
            "libribrain100", seed=0, config_path=path_str, exp_tag="t3_check"
        )
        assert cfg["workflow"]["stage"] == stage
        t_cfg = cfg["model"]["transformer"]
        assert t_cfg["architecture"] == "signed_decollapse_additive"
        assert t_cfg["horizon_conditioned"] is expect_horizon
        assert t_cfg["include_self_offset"] is False
        assert int(t_cfg["self_decollapse_rank"]) == 128
        assert t_cfg["use_h_bins_pca"] is True
        assert t_cfg["session_calibration"]["enabled"] is True
        assert float(t_cfg["session_calibration"]["shrinkage_m0"]) == pytest.approx(50.0)
        assert int(cfg["training"]["early_stop_min_epoch"]) == 3
        assert cfg["training"]["decoupled_weight_decay"] is True
        assert float(cfg["training"]["self_decollapse_weight_decay"]) == pytest.approx(0.03)
        s_cfg = cfg["siamese_debiased"]
        assert float(s_cfg["eta_train"]) == pytest.approx(0.0)
        assert float(s_cfg["lambda_null"]) == pytest.approx(60.0)
        assert float(s_cfg["lambda_delta"]) == pytest.approx(expect_lam_delta)
        assert float(s_cfg["lambda_shift_lag_decorr"]) == pytest.approx(2.0)
        assert int(s_cfg["shift_lag_decorr_max_lag"]) == 3

    nb_path = Path("scripts/colab/train_libribrain.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell_ids = {c.get("id") for c in nb["cells"]}
    assert {
        "lb-stage-h-md",
        "lb-h1-ladder1-train",
        "lb-h2-ladder1-synth",
        "lb-h3-ladder1-diag",
    }.issubset(cell_ids)


def test_cuda_bf16_and_synthetic_gpu_prehead_smoke() -> None:
    """在可用 GPU 上校验 bf16 预提取、原地刷新、训练、在线校准推理与合成脉冲流 pre-head 快路径。"""
    from braindecoding.data.synthetic_timing_dataset import SyntheticTimingDataset
    from braindecoding.training.debiased_preextraction import (
        refresh_preextracted_train_features,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = dev.type == "cuda"
    ds_tr = _ToyMegDataset(n_sentences=8, sent_len=5, dim=32, seed=10)
    ds_va = _ToyMegDataset(n_sentences=6, sent_len=5, dim=32, seed=11)
    cfg = {
        "embedding_dimension": 32,
        "use_transformer": True,
        "context_mode": "grouped",
        "conv": _tiny_multibin_conv_cfg(32),
        "transformer": {
            "architecture": "signed_decollapse_additive",
            "heads": 8,
            "head_dimension": 8,
            "max_relative_distance": 12,
            "residual_max_norm": 0.75,
            "attention_dropout": 0.1,
            "include_self_offset": False,
            "horizon_conditioned": True,
            "num_horizons": 3,
            "horizon_readout_rank": 16,
            "self_decollapse_rank": 16,
            "self_decollapse_dropout": 0.25,
            "use_h_bins_pca": True,
            "h_bins_pca_dim": 16,
            "session_calibration": {
                "enabled": True,
                "shrinkage_m0": 5.0,
                "lag_gap": 2,
                "alpha": 1.0,
            },
        },
        "prefix_revision": {
            "enabled": True,
            "history_radius": None,
            "horizons": [0, 1, 2],
            "separate_horizon_losses": True,
            "validation_query_batch_size": 32,
        },
    }
    model = build_brain_embedding_model(16, ds_tr.channel_positions, 2, cfg).to(dev)
    for p in model.brain_encoder.parameters():
        p.requires_grad_(False)
    vocab = sorted(ds_tr.embedding_map.keys())
    bank = build_centered_vocabulary_bank(ds_tr, vocab, dev, freq_alpha=0.5)
    model.context_transformer.set_text_cone_axis(
        F.normalize(
            torch.stack([ds_tr._embedding_tensors[w] for w in vocab]).mean(dim=0),
            dim=-1,
        )
    )
    bundle = preextract_frozen_cnn_features(
        model,
        ds_tr,
        ds_va,
        dataset="libribrain100",
        subset_tag="cuda_smoke",
        word_ckpt_sha256="",
        window_seconds=1.0,
        shift_seconds=[-10.0, -20.0],
        shift_mode="preroll",
        device=dev,
        batch_size=16,
        amp=use_bf16,
        amp_dtype=torch.bfloat16,
        cache_to_disk=False,
        show_progress=False,
    )
    bundle["vocab_gram_sqrt"] = build_vocabulary_gram_sqrt(bank)
    refresh_preextracted_train_features(
        bundle,
        model,
        ds_tr,
        shift_seconds=[-10.0, -20.0],
        shift_mode="preroll",
        device=dev,
        batch_size=16,
        amp=use_bf16,
        amp_dtype=torch.bfloat16,
        show_progress=False,
    )
    ts = SentenceBatchSampler(
        ds_tr.table["runtime_context_uid"].tolist(),
        batch_size=20,
        shuffle=True,
        seed=0,
        preserve_complete_groups=True,
    )
    vs = SentenceBatchSampler(
        ds_va.table["runtime_context_uid"].tolist(),
        batch_size=15,
        shuffle=False,
        seed=0,
        preserve_complete_groups=True,
    )
    loss_mod = build_siglip_loss({"name": "siglip"}).to(dev)
    opt = build_adamw_for_modules(
        [model, loss_mod],
        {
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "self_decollapse_weight_decay": 0.03,
            "decoupled_weight_decay": True,
        },
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    scfg = resolve_siamese_debiased_config(
        {
            "siamese_debiased": {
                "enabled": True,
                "debias_mode": "null_space",
                "shift_mode": "preroll",
                "eta_0": 0.0,
                "eta_train": 0.0,
                "lambda_null": 60.0,
                "lambda_delta": 10.0,
                "null_space_coherent_mean": True,
                "null_space_excess_coherent": True,
                "null_space_excess_tolerance": 0.0,
                "null_space_increment_mode": True,
                "lambda_shift_lag_decorr": 2.0,
                "shift_lag_decorr_max_lag": 3,
                "shift_seconds": [-10.0, -20.0],
                "shifts_per_step": 2,
            }
        }
    )
    loss_v, upd, _, dg = train_one_epoch_siamese_debiased(
        model,
        loss_mod,
        ts,
        bundle,
        opt,
        scaler,
        dev,
        {
            "amp": use_bf16,
            "amp_dtype": "bfloat16",
            "seed": 0,
            "revision_cutoff_policy": "all",
        },
        scfg,
        centered_vocab_bank=bank,
        epoch_index=0,
        show_progress=False,
    )
    assert np.isfinite(loss_v) and upd > 0 and np.isfinite(dg["shift_lag_loss"])

    enc_fast = encode_from_preextracted_cnn(
        model,
        vs,
        bundle["val_clean_z"],
        bundle["val_meta"],
        dev,
        amp=use_bf16,
        amp_dtype=torch.bfloat16,
        show_progress=False,
    )
    enc_ld = encode_loader(
        model,
        DataLoader(ds_va, batch_sampler=vs),
        dev,
        amp=use_bf16,
        amp_dtype=torch.bfloat16,
        show_progress=False,
    )
    max_diff = float((enc_fast["predictions"] - enc_ld["predictions"]).abs().max().item())
    assert max_diff < 2e-2

    # 校验合成脉冲流 GPU pre-head + PCA 提取快路径（支撑 Synth-Shared 真实 CNN 审计）
    ds_tr.signal_key = "meg"
    ds_tr.window_samples = 50
    ds_tr.baseline_samples = 0
    ds_tr.dataset_config = {
        "target_sampling_rate_hz": 50.0,
        "baseline_seconds": 0.0,
        "causal_highpass_mode": "butterworth_iir",
        "causal_highpass_hz": 0.5,
        "causal_highpass_order": 2,
        "clamp": 20.0,
    }
    ds_tr.subject_map = {"sub-00": 0, "sub-01": 1}
    ds_tr.signal_metadata = {
        "rec-0": {"source_first_samp": 0, "source_sampling_rate_hz": 50.0},
        "rec-1": {"source_first_samp": 0, "source_sampling_rate_hz": 50.0},
    }
    ds_tr.table["split"] = "train"
    ds_tr.table["word"] = ds_tr.table["normalized_word"]
    syn = SyntheticTimingDataset(ds_tr, condition="shared", split="train", training_seed=0)
    z_s, pre_s, bins_s = syn.extract_cnn_embeddings_gpu(
        model,
        device=dev,
        batch_size=16,
        amp=use_bf16,
        amp_dtype=torch.bfloat16,
        show_progress=False,
        return_pre_head=True,
        project_h_bins_fn=model.context_transformer.project_h_bins,
    )
    assert z_s.shape == (40, 32)
    assert pre_s is not None and pre_s.shape == (40, 32)
    assert bins_s is not None and bins_s.shape == (40, 16)
    assert torch.isfinite(z_s).all() and torch.isfinite(pre_s).all() and torch.isfinite(bins_s).all()

