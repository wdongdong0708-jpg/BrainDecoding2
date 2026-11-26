"""Tier 1 涨点与系统优化套件单元测试（零捷径物理边界、等价性与配置解析）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from braindecoding.data import text_embeddings as text_data
from braindecoding.data.synthetic_timing_dataset import SyntheticTimingDataset
from braindecoding.data.word_window_dataset import (
    WordWindowDataset,
    apply_causal_butterworth_highpass,
    open_derived_signal,
)
from braindecoding.losses import (
    HybridSigLipFullVocabCELoss,
    LogitAdjustedFullVocabCELoss,
    SigLipLoss,
    build_siglip_loss,
)
from braindecoding.models import (
    SalienceGatedAdditiveV2Encoder,
    SpatialChannelMerger,
    SubjectLinear,
    build_brain_embedding_model,
)
from braindecoding.optimizers import (
    ModelEMA,
    build_adamw_for_modules,
    build_scheduler,
)
from braindecoding.training.debiased_losses import (
    compute_excess_coherent_shift_null_loss,
)
from braindecoding.training.debiased_padded_forward import (
    _can_use_direct_layer_loop,
    _can_use_fast_padded_transformer,
    _run_fast_padded_transformer,
)
from braindecoding.training.debiased_preextraction import _preextracted_cache_path
from braindecoding.training.workflow_runner import build_workflow_config


def _make_toy_meg_dataset(
    tmp_path: Path,
    *,
    dataset_overrides: dict | None = None,
    temporal_shift_seconds: float = 0.0,
    shift_mode: str = "deterministic_circular",
    preroll_buffers: dict[str, np.ndarray] | None = None,
    recording_signal: np.ndarray | None = None,
) -> tuple[WordWindowDataset, np.ndarray]:
    rng = np.random.default_rng(2026)
    channels = 4
    total_samples = 1200  # 24 seconds @ 50Hz
    if recording_signal is None:
        t_axis = np.linspace(0.0, 24.0, total_samples, dtype=np.float32)
        drift = (
            3.5 * np.sin(2.0 * np.pi * 0.1 * t_axis)[None, :]
            + np.linspace(-2.0, 2.0, total_samples, dtype=np.float32)[None, :]
        )
        fast = 0.6 * np.sin(2.0 * np.pi * 3.0 * t_axis)[None, :]
        recording_signal = (
            drift + fast + 0.35 * rng.standard_normal((channels, total_samples))
        ).astype(np.float32)

    meg_dir = tmp_path / "meg"
    rec_dir = meg_dir / "sub-0"
    rec_dir.mkdir(parents=True, exist_ok=True)
    sig_path = rec_dir / "rec-00.npy"
    np.save(sig_path, recording_signal)
    open_derived_signal.cache_clear()

    ch_names = [f"MEG{i:04d}" for i in range(channels)]
    sidecar = {
        "recording_id": "rec-00",
        "output_sampling_rate_hz": 50.0,
        "source_sampling_rate_hz": 1000.0,
        "source_first_samp": 0,
        "dtype": "float32",
        "channel_count": channels,
        "channel_names": ch_names,
    }
    sig_path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")

    words = ["the", "cat", "sat", "on", "mat", "and", "ran", "home"]
    onsets = [0.20, 0.50, 1.20, 2.50, 12.00, 13.50, 18.00, 20.50]
    rows = []
    for idx, (word, onset) in enumerate(zip(words, onsets)):
        rows.append(
            {
                "事件编号": f"ev-{idx:03d}",
                "受试者": "sub-0",
                "记录编号": "rec-00",
                "数据划分": "train",
                "划分单元": "unit-00",
                "材料编号": "mat-00",
                "是否可训练": True,
                "排除原因": "",
                "上下文编号": "sent-0",
                "记录内序号": idx,
                "词": word,
                "标准词": word,
                "开始时间": onset,
                "结束时间": onset + 0.25,
                "持续时间": 0.25,
                "是否内容词": True,
                "是否序列末词": idx == len(words) - 1,
                "信号相对路径": "sub-0/rec-00.npy",
            }
        )
    table = pd.DataFrame(rows)

    text_cfg = {
        "model_name": "toy-text",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "embedding_dimension": 8,
        "word_normalization": "canonical_standard_word",
    }
    emb_path = tmp_path / "text_embeddings.npz"
    sig = text_data.text_embedding_signature(text_cfg)
    vocab = sorted(set(words))
    vecs = rng.standard_normal((len(vocab), 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.savez(emb_path, words=np.asarray(vocab), embeddings=vecs)
    emb_path.with_suffix(".json").write_text(
        json.dumps({"signature": sig}), encoding="utf-8"
    )

    ds_cfg = {
        "target_sampling_rate_hz": 50.0,
        "window_seconds": 1.0,
        "window_start_offset_seconds": 0.0,
        "baseline_seconds": 0.0,
        "causal_highpass_seconds": 0.0,
        "causal_highpass_mode": "butterworth_iir",
        "causal_highpass_hz": 0.5,
        "causal_highpass_order": 2,
        "clamp": 20.0,
        "subjects": ["sub-0"],
    }
    if dataset_overrides:
        ds_cfg.update(dataset_overrides)

    ds = WordWindowDataset(
        table,
        ds_cfg,
        meg_dir,
        emb_path,
        text_cfg,
        temporal_shift_seconds=temporal_shift_seconds,
        shift_mode=shift_mode,
        preroll_seconds=15.0,
        preroll_buffers=preroll_buffers,
    )
    return ds, recording_signal


def test_spatial_merger_factored_gemm_and_subject_linear_s1_parity() -> None:
    torch.manual_seed(0)
    positions = torch.rand(12, 2)
    x = torch.randn(16, 12, 50)
    subj_s1 = torch.zeros(16, dtype=torch.long)

    merger_s1 = SpatialChannelMerger(
        positions,
        output_channels=8,
        position_dimension=32,
        subject_count=1,
        per_subject=True,
        dropout_radius=0.0,
    )
    merger_s1.eval()
    out_fast = merger_s1(x, subj_s1)
    # 参考路径：原始逐样本索引后 einsum
    coeff_ref = merger_s1.coefficients[subj_s1]
    scores_ref = torch.einsum("bod,cd->boc", coeff_ref, merger_s1.basis)
    weights_ref = torch.softmax(scores_ref, dim=-1)
    out_ref = torch.einsum("bct,boc->bot", x, weights_ref)
    assert torch.allclose(out_fast, out_ref, atol=1e-5)

    # 验证 per_sample_dropout=True 对同批样本产生不同空间掩码
    merger_drop = SpatialChannelMerger(
        positions,
        output_channels=8,
        position_dimension=32,
        subject_count=1,
        per_subject=True,
        dropout_radius=0.4,
        per_sample_dropout=True,
    )
    merger_drop.train()
    same_x = torch.randn(1, 12, 10).expand(16, 12, 10).contiguous()
    out_drop = merger_drop(same_x, subj_s1)
    assert torch.isfinite(out_drop).all()
    assert not torch.allclose(out_drop[0], out_drop[1])

    # SubjectLinear S=1 快速路径逐位等价
    sl = SubjectLinear(1, 8, output_channels=10, bias=True, initialization="normal")
    feats = torch.randn(16, 8, 50)
    out_sl_fast = sl(feats, subj_s1)
    out_sl_ref = torch.bmm(sl.weights[subj_s1], feats) + sl.bias[subj_s1].unsqueeze(-1)
    assert torch.allclose(out_sl_fast, out_sl_ref, atol=1e-5)


def test_causal_butterworth_zero_future_leak_preload_and_preroll_parity(
    tmp_path: Path,
) -> None:
    ds_lazy, sig = _make_toy_meg_dataset(
        tmp_path / "lazy",
        dataset_overrides={"preload_filtered_recordings": False},
    )
    ds_ram, _ = _make_toy_meg_dataset(
        tmp_path / "ram",
        dataset_overrides={"preload_filtered_recordings": True},
        recording_signal=sig,
    )
    # 局部结算段切片与全记录 RAM 预滤波必须在 1e-5 内完全一致
    for idx in range(len(ds_lazy)):
        w_lazy = ds_lazy.read_meg(idx)
        w_ram = ds_ram.read_meg(idx)
        assert np.allclose(w_lazy, w_ram, atol=1e-5)

    # 严格因果性：篡改某词窗末尾之后 1 个采样点，该词窗输出必须逐位零变化
    target_idx = 4  # onset = 12.0s (sample 600..650)
    sig_mut = sig.copy()
    sig_mut[:, 650:] += 50.0
    ds_mut, _ = _make_toy_meg_dataset(
        tmp_path / "mut",
        dataset_overrides={"preload_filtered_recordings": True},
        recording_signal=sig_mut,
    )
    assert np.allclose(ds_ram.read_meg(target_idx), ds_mut.read_meg(target_idx), atol=1e-6)

    # 0.1 Hz 慢漂移抑制验证（与 2.0s 滑动均值相比，Butterworth 0.5Hz 压低慢漂移幅度 > 5x）
    t = np.arange(1500, dtype=np.float64) / 50.0
    drift_only = np.sin(2.0 * np.pi * 0.1 * t)[None, :].astype(np.float32)
    bw_out = apply_causal_butterworth_highpass(
        drift_only, sampling_rate_hz=50.0, cutoff_hz=0.5, order=2
    )
    assert float(np.std(bw_out[:, 500:])) < 0.06 * float(np.std(drift_only[:, 500:]))

    # Pre-Roll 跨边界拼接因果连续性：无阶跃伪影且精确匹配整段拼接后 lfilter
    preroll = np.full((4, 1000), 1.25, dtype=np.float32)
    ds_pr, _ = _make_toy_meg_dataset(
        tmp_path / "preroll",
        dataset_overrides={"preload_filtered_recordings": True},
        temporal_shift_seconds=-1.0,
        shift_mode="preroll",
        preroll_buffers={"rec-00": preroll},
        recording_signal=sig,
    )
    # idx=0 onset=0.20s -> start=10, shift=-50 -> raw_start=-40（跨 preroll/recording 边界）
    stitched = np.concatenate([preroll, sig], axis=1)
    ref_full = apply_causal_butterworth_highpass(
        stitched, sampling_rate_hz=50.0, cutoff_hz=0.5, order=2
    )
    ref_win = np.clip(ref_full[:, 1000 - 40 : 1000 - 40 + 50], -20.0, 20.0)
    assert np.allclose(ds_pr.read_meg(0), ref_win, atol=1e-5)


def test_synthetic_timing_dataset_butterworth_and_clamp_parity(tmp_path: Path) -> None:
    base_ds, _ = _make_toy_meg_dataset(
        tmp_path / "synth_base",
        dataset_overrides={
            "causal_highpass_mode": "butterworth_iir",
            "causal_highpass_hz": 0.5,
            "causal_highpass_order": 2,
            "clamp": 20.0,
        },
    )
    synth_ds = SyntheticTimingDataset(base_ds, "shared", split="val", evaluation_seed=77)
    w_cpu = synth_ds[0]["meg"]
    assert w_cpu.shape == (4, 50)
    assert torch.isfinite(w_cpu).all()
    # GPU 批量物化同样施加 Butterworth 高通且输出有限、低频被抑制
    w_gpu_batch = synth_ds.materialize_windows_gpu(device=torch.device("cpu"))
    assert w_gpu_batch.shape == (len(synth_ds), 4, 50)
    assert torch.isfinite(w_gpu_batch).all()


def test_logit_adjusted_and_hybrid_full_vocab_loss_zero_noise_bias() -> None:
    torch.manual_seed(11)
    v_size, dim = 64, 32
    words = [f"w{i}" for i in range(v_size)]
    embs = F.normalize(torch.randn(v_size, dim), dim=-1)
    # 极端 Zipf 词频分布
    counts = [int(1000 / (i + 1)) + 1 for i in range(v_size)]

    loss_ce = build_siglip_loss(
        {
            "name": "logit_adjusted_full_vocab_ce",
            "initial_temperature": 12.0,
            "max_temperature": 18.0,
            "alpha_prior": 1.0,
            "label_smoothing": 0.0,
        }
    )
    assert isinstance(loss_ce, LogitAdjustedFullVocabCELoss)
    loss_ce.set_vocabulary(words, embs, counts)

    # 验证共享纯噪声/常向量（零信息）在训练集词频分布 pi_v 上对 z 的期望梯度严格为零（零词频捷径）
    probs = torch.tensor(counts, dtype=torch.float32)
    probs = probs / probs.sum()
    z0 = torch.zeros(1, dim, requires_grad=True)
    z_const = z0.expand(v_size, dim)
    logits = (
        loss_ce._effective_temperature() * (z_const @ loss_ce.vocab_embeddings.t())
        + loss_ce.log_prior.unsqueeze(0)
    )
    per_class_ce = F.cross_entropy(logits, torch.arange(v_size), reduction="none")
    weighted_ce = (probs * per_class_ce).sum()
    weighted_ce.backward()
    assert float(z0.grad.abs().max().item()) < 1e-6

    # 验证 Hybrid 损失 forward 与 forward_rectangular 及非持久 buffer 契约
    hybrid = build_siglip_loss(
        {
            "name": "hybrid_siglip_full_vocab_ce",
            "siglip_weight": 0.5,
            "ce_weight": 0.5,
            "ce_max_temperature": 22.0,
            "label_smoothing": 0.0,
        }
    )
    assert isinstance(hybrid, HybridSigLipFullVocabCELoss)
    hybrid.set_vocabulary(words, embs, counts)
    # 词表矩阵不得进入 state_dict（防止检查点膨胀）
    assert "full_vocab_ce.vocab_embeddings" not in hybrid.state_dict()
    assert "full_vocab_ce.log_prior" not in hybrid.state_dict()

    ids = torch.tensor([0, 3, 7, 15, 22, 40], dtype=torch.long)
    est = F.normalize(torch.randn(6, dim), dim=-1).requires_grad_(True)
    cand = embs[ids]
    l_full = hybrid(est, cand, target_word_ids=ids)
    assert torch.isfinite(l_full)
    l_full.backward()
    assert est.grad is not None and float(est.grad.abs().sum().item()) > 0.0

    est2 = F.normalize(torch.randn(3, dim), dim=-1).requires_grad_(True)
    pos_idx = torch.tensor([1, 3, 5], dtype=torch.long)
    l_rect = hybrid.forward_rectangular(est2, cand, pos_idx, target_word_ids=ids[pos_idx])
    assert torch.isfinite(l_rect)
    l_rect.backward()
    assert est2.grad is not None and float(est2.grad.abs().sum().item()) > 0.0


def test_salience_v2_asymmetric_init_text_cone_lock_and_horizon_parity() -> None:
    torch.manual_seed(17)
    dim = 64
    cfg = {
        "architecture": "salience_gated_additive_v2",
        "heads": 8,
        "head_dimension": 8,
        "max_relative_distance": 12,
        "residual_max_norm": 0.85,
        "attention_dropout": 0.0,
        "horizon_conditioned": True,
        "num_horizons": 3,
        "horizon_readout_rank": 16,
    }
    enc = SalienceGatedAdditiveV2Encoder(dim, cfg)

    # 1) 验证定分母零汇权重表满足 sum_d |w_d| < 1 且 step-0 全部核参数梯度非零
    w_tab = enc._compute_signed_weight_table()
    assert w_tab.shape == (3, 8, 25)
    assert bool((w_tab.abs().sum(dim=-1) < 1.0).all())

    u0_text = F.normalize(torch.randn(dim), dim=-1)
    enc.set_text_cone_axis(u0_text)

    # 故意构造与 u0_text 几乎正交的输入 z（模拟纯 LA-FV-CE 使 z@u0 -> 0 的极端情形）
    z_raw = torch.randn(11, dim)
    z_raw = z_raw - (z_raw @ u0_text).unsqueeze(-1) * u0_text.unsqueeze(0) + 0.05 * u0_text.unsqueeze(0)
    z = F.normalize(z_raw, dim=-1)
    enc.set_clean_input_mean(z.mean(dim=0))

    padded = torch.zeros((3, 5, dim), dtype=torch.float32)
    mask = torch.zeros((3, 5), dtype=torch.bool)
    padded[0, :4] = z[:4]
    mask[0, :4] = True
    padded[1, :3] = z[4:7]
    mask[1, :3] = True
    padded[2, :4] = z[7:11]
    mask[2, :4] = True

    enc.train()
    out_step0 = enc.forward_padded(padded, mask, num_branches=1)
    target = F.normalize(torch.randn_like(out_step0), dim=-1)
    loss0 = -(out_step0 * target * mask.unsqueeze(-1).float()).sum(dim=-1).mean()
    loss0.backward()
    assert enc.pos_logits.grad is not None and float(enc.pos_logits.grad.abs().max().item()) > 1e-8
    assert enc.neg_logits.grad is not None and float(enc.neg_logits.grad.abs().max().item()) > 1e-8
    assert enc.sink_logits.grad is not None and float(enc.sink_logits.grad.abs().max().item()) > 1e-8
    assert enc.horizon_up.grad is not None and float(enc.horizon_up.grad.abs().max().item()) > 1e-8

    # 2) 验证非零 horizon_up 下独立 text_cone_axis 严格锁定 c_i = z_i^T u_{0,text}
    with torch.no_grad():
        enc.out_proj.weight.normal_(0.0, 0.2)
        enc.horizon_up.normal_(0.0, 0.1)
    enc.eval()
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
    y_grouped = enc(z, groups)
    assert torch.allclose(
        (y_grouped * u0_text).sum(dim=-1),
        (z * u0_text).sum(dim=-1),
        atol=1e-5,
    )
    assert torch.allclose(y_grouped.norm(dim=-1), torch.ones(len(z)), atol=1e-5)

    # 3) 验证 horizon_conditioned=True 时 forward == forward_padded(target_col_idx=None) == forward_padded(target_col_idx=...)
    y_padded_full = enc.forward_padded(padded, mask, num_branches=1)
    reconstructed = torch.cat(
        [y_padded_full[0, :4], y_padded_full[1, :3], y_padded_full[2, :4]], dim=0
    )
    assert torch.allclose(y_grouped, reconstructed, atol=1e-5)

    # 覆盖 avail_r = 0, 1, 2 全部三个视界（含句末词与句中词）
    target_group = torch.tensor([0, 0, 0, 0, 1, 1, 2], dtype=torch.long)
    target_col = torch.tensor([0, 1, 2, 3, 0, 2, 1], dtype=torch.long)
    y_pruned = enc.forward_padded(
        padded,
        mask,
        target_group_idx=target_group,
        target_col_idx=target_col,
    )
    expected_pruned = y_padded_full[target_group, target_col]
    assert torch.allclose(y_pruned, expected_pruned, atol=1e-5)

    # 4) 验证 BrainEmbeddingModel 路由与 strict state_dict 往返加载
    model = build_brain_embedding_model(
        4,
        np.zeros((4, 2), dtype=np.float32),
        1,
        {
            "embedding_dimension": dim,
            "use_transformer": True,
            "context_mode": "grouped",
            "conv": {"merger_channels": 8, "merger_position_dimension": 32, "initial_linear": 16, "hidden": 16, "depth": 2},
            "transformer": cfg,
            "prefix_revision": {"enabled": True, "horizons": [0, 1, 2]},
        },
    )
    assert _can_use_fast_padded_transformer(model)
    assert _can_use_direct_layer_loop(model.context_transformer, use_causal_attention=False)
    model.context_transformer.set_text_cone_axis(u0_text)
    model.eval()
    rev_out = model.forward_prefix_revisions(
        torch.randn(5, 4, 50),
        torch.zeros(5, dtype=torch.long),
        torch.zeros(5, dtype=torch.long),
    )
    assert set(rev_out.keys()) == {0, 1, 2}
    for d, pred in rev_out.items():
        assert pred.shape == (5, dim) and torch.isfinite(pred).all()


def test_decoupled_adamw_step_scheduler_ema_and_zero_sync_excess_loss() -> None:
    torch.manual_seed(3)
    enc = SalienceGatedAdditiveV2Encoder(
        32,
        {
            "architecture": "salience_gated_additive_v2",
            "heads": 4,
            "head_dimension": 8,
            "horizon_conditioned": True,
        },
    )
    loss_mod = build_siglip_loss({"name": "hybrid_siglip_full_vocab_ce"})
    opt = build_adamw_for_modules(
        [enc, loss_mod],
        {
            "learning_rate": 6e-4,
            "weight_decay": 0.05,
            "decoupled_weight_decay": True,
        },
    )
    assert len(opt.param_groups) == 2
    wd_map = {id(p): g["weight_decay"] for g in opt.param_groups for p in g["params"]}
    assert wd_map[id(enc.value_proj.weight)] == pytest.approx(0.05)
    assert wd_map[id(enc.pos_logits)] == 0.0
    assert wd_map[id(enc.neg_logits)] == 0.0
    assert wd_map[id(enc.sink_logits)] == 0.0
    assert wd_map[id(enc.gate_proj.bias)] == 0.0
    assert wd_map[id(loss_mod.siglip.log_temperature)] == 0.0

    # Step warmup + cosine minimum lr
    sched = build_scheduler(
        opt, total_steps=100, warmup_ratio=0.1, minimum_learning_rate=1e-6
    )
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9]
    assert lrs[9] == pytest.approx(6e-4, rel=1e-3)
    assert lrs[-1] >= 1e-6 - 1e-12

    # ModelEMA 无损切换与恢复
    ema = ModelEMA(enc, decay=0.9, trainable_only=True)
    w_init = enc.out_proj.weight.detach().clone()
    with torch.no_grad():
        enc.out_proj.weight.add_(1.0)
    w_after = enc.out_proj.weight.detach().clone()
    ema.update(enc)
    with ema.apply_to(enc):
        assert not torch.allclose(enc.out_proj.weight, w_after)
        assert not torch.allclose(enc.out_proj.weight, w_init)
    assert torch.allclose(enc.out_proj.weight, w_after)

    # 零 host sync 超额相干损失正/零分支梯度行为
    bank = F.normalize(torch.randn(30, 16), dim=-1)
    # 纯相干（所有分支相同）-> excess > 0，有非零梯度
    base_u = torch.randn(1, 8, 16)
    u_coh = base_u.repeat(4, 1, 1).requires_grad_(True)
    loss_pos = compute_excess_coherent_shift_null_loss(u_coh, bank, excess_tolerance=0.0)
    assert float(loss_pos.item()) > 0.0
    loss_pos.backward()
    assert float(u_coh.grad.abs().sum().item()) > 0.0

    # 容忍度极大 -> excess == 0，梯度严格为 0
    u_zero = base_u.repeat(4, 1, 1) + 2.0 * torch.randn(4, 8, 16)
    u_zero.requires_grad_(True)
    loss_zero = compute_excess_coherent_shift_null_loss(u_zero, bank, excess_tolerance=1e6)
    assert float(loss_zero.item()) == 0.0
    loss_zero.backward()
    assert float(u_zero.grad.abs().sum().item()) == 0.0


def test_tier1_cache_hash_isolation_and_yaml_configs() -> None:
    p_default = _preextracted_cache_path(
        dataset="libribrain100",
        subset_tag="all-n01",
        word_ckpt_sha256="abc123",
        window_seconds=1.0,
        window_start_offset_seconds=0.0,
        shift_seconds=[-12.0],
        train_len=100,
        val_len=20,
        shift_mode="preroll",
        baseline_seconds=0.0,
        causal_highpass_seconds=2.0,
    )
    p_bw = _preextracted_cache_path(
        dataset="libribrain100",
        subset_tag="all-n01",
        word_ckpt_sha256="abc123",
        window_seconds=1.0,
        window_start_offset_seconds=0.0,
        shift_seconds=[-12.0],
        train_len=100,
        val_len=20,
        shift_mode="preroll",
        baseline_seconds=0.0,
        causal_highpass_seconds=0.0,
        causal_highpass_mode="butterworth_iir",
        causal_highpass_hz=0.5,
        causal_highpass_order=2,
        clamp=20.0,
    )
    assert p_default != p_bw

    cfg_s1, _ = build_workflow_config(
        config_path="configs/training/libribrain100_word_encoder_tier1_butterworth_hybrid.yaml",
        seed=0,
    )
    assert cfg_s1["dataset"]["causal_highpass_mode"] == "butterworth_iir"
    assert cfg_s1["dataset"]["causal_highpass_hz"] == pytest.approx(0.5)
    assert cfg_s1["dataset"]["clamp"] == pytest.approx(20.0)
    assert cfg_s1["dataset"]["preload_filtered_recordings"] is True
    assert cfg_s1["loss"]["name"] == "hybrid_siglip_full_vocab_ce"
    assert cfg_s1["loss"]["label_smoothing"] == 0.0
    assert cfg_s1["training"]["scheduler"] == "step_cosine"
    assert cfg_s1["training"]["decoupled_weight_decay"] is True
    assert cfg_s1["training"]["ema_decay"] == pytest.approx(0.999)

    cfg_ch2, _ = build_workflow_config(
        config_path="configs/training/siamese_debiased/libribrain100_tier1_salience_v2_context.yaml",
        seed=0,
    )
    assert cfg_ch2["model"]["transformer"]["architecture"] == "salience_gated_additive_v2"
    assert cfg_ch2["model"]["transformer"]["horizon_conditioned"] is False
    assert cfg_ch2["siamese_debiased"]["eta_0"] == 0.0
    assert cfg_ch2["siamese_debiased"]["lambda_null"] == pytest.approx(120.0)

    cfg_ch3, _ = build_workflow_config(
        config_path="configs/training/siamese_debiased/libribrain100_tier1_salience_v2_prefix_revision.yaml",
        seed=0,
    )
    assert cfg_ch3["model"]["transformer"]["architecture"] == "salience_gated_additive_v2"
    assert cfg_ch3["model"]["transformer"]["horizon_conditioned"] is True
    assert cfg_ch3["siamese_debiased"]["eta_0"] == 0.0
    assert cfg_ch3["siamese_debiased"]["lambda_null"] == pytest.approx(120.0)
    assert cfg_ch3["siamese_debiased"]["lambda_delta"] == pytest.approx(20.0)
