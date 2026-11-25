"""方案体系 II 最小改动（2.0s 因果滑动高通、去重叠间隔采样与缓存隔离）单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from braindecoding.data.word_window_dataset import (
    WordWindowDataset,
    _apply_causal_moving_average_highpass,
    open_derived_signal,
)
from braindecoding.training.debiased_preextraction import _preextracted_cache_path
from braindecoding.training.word_encoder_training import (
    MinGapBatchSampler,
    SentenceBatchSampler,
    make_loader,
)
from braindecoding.training.workflow_runner import build_workflow_config


def _reference_full_recording_causal_hp(signal: np.ndarray, hp_samples: int) -> np.ndarray:
    channels, length = signal.shape
    out = np.empty((channels, length), dtype=np.float32)
    sig64 = signal.astype(np.float64)
    for t in range(length):
        left = max(0, t - hp_samples + 1)
        out[:, t] = (sig64[:, t] - sig64[:, left : t + 1].mean(axis=1)).astype(np.float32)
    return out


def _make_toy_meg_dataset(
    tmp_path: Path,
    *,
    baseline_seconds: float,
    causal_highpass_seconds: float,
    temporal_shift_seconds: float = 0.0,
    shift_mode: str = "deterministic_circular",
    preroll_buffers: dict[str, np.ndarray] | None = None,
    recording_signal: np.ndarray | None = None,
) -> tuple[WordWindowDataset, np.ndarray]:
    rng = np.random.default_rng(123)
    channels = 4
    total_samples = 600
    if recording_signal is None:
        # 注入显著低频慢漂移 + 局部波动以检验高通去漂移精确性
        t_axis = np.linspace(0.0, 12.0, total_samples, dtype=np.float32)
        drift = 2.0 * np.sin(0.25 * t_axis)[None, :] + np.linspace(-1.0, 1.5, total_samples, dtype=np.float32)[None, :]
        recording_signal = (drift + 0.3 * rng.standard_normal((channels, total_samples))).astype(np.float32)

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
    onsets = [0.20, 0.50, 0.85, 1.20, 3.10, 3.45, 5.80, 6.20]
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
    from braindecoding.data import text_embeddings as text_data

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
        "subjects": ["sub-0"],
        "target_sampling_rate_hz": 50.0,
        "window_seconds": 1.0,
        "window_start_offset_seconds": 0.0,
        "baseline_seconds": baseline_seconds,
        "causal_highpass_seconds": causal_highpass_seconds,
        "clamp": 5.0,
    }
    ds = WordWindowDataset(
        table,
        ds_cfg,
        meg_dir,
        emb_path,
        text_cfg,
        temporal_shift_seconds=temporal_shift_seconds,
        shift_mode=shift_mode,
        preroll_seconds=3.0,
        preroll_buffers=preroll_buffers,
    )
    return ds, recording_signal


def test_causal_highpass_matches_full_recording_and_zero_future_leak(tmp_path: Path) -> None:
    ds, raw = _make_toy_meg_dataset(
        tmp_path, baseline_seconds=0.0, causal_highpass_seconds=2.0
    )
    ref_full = np.clip(_reference_full_recording_causal_hp(raw, 100), -5.0, 5.0)
    # 覆盖记录开头（t < 100）与记录中段（t >= 100）
    for idx in range(len(ds)):
        start = int(ds._window_starts[idx])
        got = ds.read_meg(idx)
        expected = ref_full[:, start : start + 50]
        assert np.allclose(got, expected, atol=1e-5)

    # 严格因果性：大幅扰动窗口结束后的未来采样点，当前窗口输出必须逐位不变
    perturbed = raw.copy()
    target_idx = 4
    stop = int(ds._window_starts[target_idx]) + 50
    perturbed[:, stop:] += 100.0
    ds_pert, _ = _make_toy_meg_dataset(
        tmp_path / "pert",
        baseline_seconds=0.0,
        causal_highpass_seconds=2.0,
        recording_signal=perturbed,
    )
    assert np.array_equal(ds.read_meg(target_idx), ds_pert.read_meg(target_idx))


def test_causal_highpass_preroll_boundary_continuity(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    preroll = rng.standard_normal((4, 300)).astype(np.float32)
    ds, raw = _make_toy_meg_dataset(
        tmp_path,
        baseline_seconds=0.0,
        causal_highpass_seconds=2.0,
        temporal_shift_seconds=-1.0,
        shift_mode="preroll",
        preroll_buffers={"sub-0": preroll},
    )
    virtual = np.concatenate([preroll, raw], axis=1)
    ref_virtual = np.clip(_reference_full_recording_causal_hp(virtual, 100), -5.0, 5.0)
    # idx=0 (onset=0.20s -> start=10, shifted=-40 跨越 preroll/recording 边界)
    for idx in (0, 1, 2, 5):
        v_start = preroll.shape[1] + int(ds._window_starts[idx]) - 50
        got = ds.read_meg(idx)
        expected = ref_virtual[:, v_start : v_start + 50]
        assert np.allclose(got, expected, atol=1e-5)

    # 独立校验 helper 函数直接调用
    seg = virtual[:, 0:150]
    direct = _apply_causal_moving_average_highpass(
        seg, target_start=99, window_samples=50, highpass_samples=100
    )
    assert direct.shape == (4, 50)


def test_min_gap_batch_sampler_zero_overlap_and_full_coverage() -> None:
    rng = np.random.default_rng(99)
    rec_ids = []
    onsets = []
    for r in range(6):
        t = 0.0
        for _ in range(120):
            rec_ids.append(f"rec-{r:02d}")
            onsets.append(t)
            t += float(rng.uniform(0.12, 0.55))

    min_gap = 2.5
    sampler = MinGapBatchSampler(
        rec_ids, onsets, batch_size=32, min_gap_seconds=min_gap, shuffle=True, seed=0
    )
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        assert len(batches) == len(sampler)
        seen = []
        for batch in batches:
            assert len(batch) >= 2
            assert len(batch) <= 32
            seen.extend(batch)
            for i in range(len(batch)):
                for j in range(i + 1, len(batch)):
                    a, b = batch[i], batch[j]
                    if rec_ids[a] == rec_ids[b]:
                        assert abs(onsets[a] - onsets[b]) >= min_gap - 1e-9
        assert sorted(seen) == list(range(len(rec_ids)))


def test_make_loader_opt_in_min_gap_sampler(tmp_path: Path) -> None:
    ds, _ = _make_toy_meg_dataset(
        tmp_path, baseline_seconds=0.0, causal_highpass_seconds=2.0
    )
    tc_default = {"batch_size": 4, "seed": 0, "num_workers": 0}
    _, s_default = make_loader(ds, tc_default, shuffle=True)
    assert isinstance(s_default, SentenceBatchSampler)

    tc_mingap = {
        "batch_size": 4,
        "seed": 0,
        "num_workers": 0,
        "word_sampling": "min_gap_sampler",
        "min_gap_seconds": 2.5,
    }
    _, s_train = make_loader(ds, tc_mingap, shuffle=True)
    assert isinstance(s_train, MinGapBatchSampler)

    # shuffle=False (val/test) 与 preserve_complete_groups=True (Stage-2) 必须回落 SentenceBatchSampler
    _, s_val = make_loader(ds, tc_mingap, shuffle=False)
    assert isinstance(s_val, SentenceBatchSampler)
    _, s_stage2 = make_loader(
        ds,
        {**tc_mingap, "batch_size": 8, "preserve_complete_groups": True},
        shuffle=True,
    )
    assert isinstance(s_stage2, SentenceBatchSampler)


def test_preextracted_cache_path_and_scheme2_configs() -> None:
    legacy_path = _preextracted_cache_path(
        dataset="libribrain100",
        subset_tag="all-n01",
        word_ckpt_sha256="abc123",
        window_seconds=1.0,
        window_start_offset_seconds=0.0,
        shift_seconds=[-12.0],
        train_len=100,
        val_len=20,
        shift_mode="preroll",
        preroll_seconds=55.0,
    )
    explicit_default_path = _preextracted_cache_path(
        dataset="libribrain100",
        subset_tag="all-n01",
        word_ckpt_sha256="abc123",
        window_seconds=1.0,
        window_start_offset_seconds=0.0,
        shift_seconds=[-12.0],
        train_len=100,
        val_len=20,
        shift_mode="preroll",
        preroll_seconds=55.0,
        baseline_seconds=0.5,
        causal_highpass_seconds=0.0,
    )
    hp_path = _preextracted_cache_path(
        dataset="libribrain100",
        subset_tag="all-n01",
        word_ckpt_sha256="abc123",
        window_seconds=1.0,
        window_start_offset_seconds=0.0,
        shift_seconds=[-12.0],
        train_len=100,
        val_len=20,
        shift_mode="preroll",
        preroll_seconds=55.0,
        baseline_seconds=0.0,
        causal_highpass_seconds=2.0,
    )
    assert legacy_path == explicit_default_path
    assert hp_path != legacy_path

    w_cfg, _ = build_workflow_config(
        "libribrain100",
        stage="word",
        config_path="configs/training/libribrain100_word_encoder_causal_highpass_mingap.yaml",
    )
    assert float(w_cfg["dataset"]["baseline_seconds"]) == 0.0
    assert float(w_cfg["dataset"]["causal_highpass_seconds"]) == 2.0
    assert w_cfg["training"]["word_sampling"] == "min_gap_sampler"
    assert float(w_cfg["training"]["min_gap_seconds"]) == 2.5

    for stage, cfg_rel in (
        (
            "context",
            "configs/training/siamese_debiased/libribrain100_causal_highpass_salience_additive_context.yaml",
        ),
        (
            "prefix_revision",
            "configs/training/siamese_debiased/libribrain100_causal_highpass_salience_additive_prefix_revision.yaml",
        ),
    ):
        cfg_s2, _ = build_workflow_config(
            "libribrain100", stage=stage, config_path=cfg_rel
        )
        assert float(cfg_s2["dataset"]["baseline_seconds"]) == 0.0
        assert float(cfg_s2["dataset"]["causal_highpass_seconds"]) == 2.0
        assert cfg_s2["model"]["transformer"]["architecture"] == "salience_gated_additive"
        assert float(cfg_s2["workflow"]["siamese_debiased"].get("lambda_clean_stein", 0.0)) == 0.0
