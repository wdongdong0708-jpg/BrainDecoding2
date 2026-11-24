"""Scientific controls: shared sample identity, split RNG and unchanged model."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scipy.ndimage import gaussian_filter1d
import torch
from torch import nn
from torch.nn import functional as F
from braindecoding.data.synthetic_timing_dataset import (
    SyntheticTimingDataset,
    overlap_statistics,
    synthetic_stream,
    synthesize_streams_batch_gpu,
    transform_window,
)
from braindecoding.training import synthetic_timing_control as control
from braindecoding.training.debiased_preextraction import (
    _extract_single_dataset_cnn_embeddings,
    refresh_preextracted_train_features,
)

pytestmark = pytest.mark.current_protocol


def fake_base(onsets=(0, .3, .6), *, group_ids=None, subjects=None):
    count = len(onsets)
    subjects = subjects or ["sub-01"] * count
    group_ids = group_ids or ["s1"] * count
    table = pd.DataFrame({"event_id": [f"e{i}" for i in range(count)], "subject_id": subjects,
        "recording_id": [f"r-{s}" for s in subjects], "normalized_word": ["word"] * count,
        "runtime_context_uid": group_ids, "sentence_uid": group_ids,
        "onset_seconds": onsets})
    return SimpleNamespace(table=table, dataset_config={"target_sampling_rate_hz": 50,
        "baseline_seconds": .5, "clamp": 5}, window_samples=50, channel_count=8,
        subject_map={"sub-01": 0, "sub-02": 1},
        sentence_indices=np.asarray([0 if v == "s1" else 1 for v in group_ids]),
        embedding_map={"word": np.zeros(1024, dtype=np.float32)},
        signal_metadata={r: {"source_first_samp": 0, "source_sampling_rate_hz": 1000}
                         for r in table.recording_id.unique()})


def dataset(condition="shared", split="train", **kwargs):
    return SyntheticTimingDataset(fake_base(**kwargs), condition, split=split)


@pytest.mark.parametrize("samples", (50, 172, 936))
def test_sparse_generator_matches_gaussian_smoothing(samples):
    rng = np.random.default_rng(19)
    n = samples + 100
    signal = np.zeros((8, n), dtype=np.float32)
    for sigma in (3, 6, 12):
        count = min(int(rng.poisson(n / 50)), n)
        positions = rng.choice(n, count, replace=False)
        pulses = np.zeros_like(signal)
        pulses[:, positions] = rng.standard_normal((8, count))
        signal += gaussian_filter1d(pulses, sigma=sigma, mode="constant", truncate=4)
    signal /= float(signal.std(dtype=np.float64)) + 1e-6
    expected = signal[:, 50:50 + samples]
    np.testing.assert_allclose(synthetic_stream(samples, 8, 50, np.random.default_rng(19)),
                               expected, rtol=2e-6, atol=2e-6)


def test_fifteen_sample_shift_shares_exact_thirty_five_samples():
    shared, independent = dataset(), dataset("independent")
    np.testing.assert_array_equal(shared.raw_window(0)[:, 15:], shared.raw_window(1)[:, :35])
    assert not np.array_equal(independent.raw_window(0)[:, 15:], independent.raw_window(1)[:, :35])
    # Per-window baseline is a channel-wise constant; it does not erase the
    # shifted feature pattern before clipping. Do not demand equality here.
    a = transform_window(shared.raw_window(0), baseline_samples=25, clamp=None)
    b = transform_window(shared.raw_window(1), baseline_samples=25, clamp=None)
    difference = a[:, 15:] - b[:, :35]
    np.testing.assert_allclose(np.diff(difference, axis=1), 0, atol=5e-7)


def test_overlap_does_not_jump_the_one_second_boundary():
    shared = dataset(onsets=(0, 1, 2))
    assert not np.array_equal(shared.raw_window(0), shared.raw_window(1))
    shared = dataset(onsets=(0, 0, .6))
    np.testing.assert_array_equal(shared.raw_window(0), shared.raw_window(1))


def test_quantize_recording_clock_before_translation():
    # Rounding differences of float onsets would incorrectly round .01 to 0.
    shared = dataset(onsets=(.01, .02, .32))
    np.testing.assert_array_equal(shared.starts, [0, 1, 16])


def test_streams_independent_across_subjects_and_segments():
    shared = dataset(onsets=(0, 0, 0), group_ids=["s1", "s2", "s1"],
                     subjects=["sub-01", "sub-01", "sub-02"])
    assert not np.array_equal(shared.raw_window(0), shared.raw_window(1))
    assert not np.array_equal(shared.raw_window(0), shared.raw_window(2))


@pytest.mark.parametrize("condition", ("shared", "independent"))
def test_train_refresh_and_fixed_evaluation_and_cache_eviction(condition):
    train = dataset(condition)
    first = train.raw_window(0)
    train.set_epoch(1)
    assert not np.array_equal(first, train.raw_window(0))
    train.set_epoch(0)
    np.testing.assert_array_equal(first, train.raw_window(0))
    for split in ("val", "test"):
        a = SyntheticTimingDataset(fake_base(), condition, split=split, training_seed=0, cache_streams=1)
        b = SyntheticTimingDataset(fake_base(), condition, split=split, training_seed=2, cache_streams=1)
        first = a.raw_window(0)
        a.raw_window(1)
        a.set_epoch(19)
        np.testing.assert_array_equal(first, a.raw_window(0))
        np.testing.assert_array_equal(first, b.raw_window(0))
    assert not np.array_equal(dataset(condition, "val").raw_window(0), dataset(condition, "test").raw_window(0))


def test_no_brain_getter_and_no_label_driven_signal():
    base = fake_base()
    def fail(*args):
        raise AssertionError("A real neural sample was accessed")
    base.read_meg = fail
    wrapped = SyntheticTimingDataset(base, "shared", split="val")
    sample = wrapped[0]
    assert sample["meg"].shape == (8, 50) and np.isfinite(sample["meg"].numpy()).all()
    base.table.normalized_word = "changed_label"
    renamed = SyntheticTimingDataset(base, "shared", split="val")
    np.testing.assert_array_equal(wrapped.raw_window(0), renamed.raw_window(0))
    assert "onset_seconds" not in sample


def test_statistics_exclude_segment_boundaries_and_missing_words():
    table = pd.DataFrame({"subject_id": ["s"] * 6, "recording_id": ["r"] * 6,
        "sentence_uid": ["a", "a", "a", "a", "b", "b"], "split": ["train"] * 6,
        "is_trainable": [True, True, False, True, True, True],
        "order": [0, 1, 2, 3, 4, 5], "onset_seconds": [0, .3, .5, .8, 1, 2.2]})
    stats = overlap_statistics(table)
    assert stats["pairs"] == 2 and stats["skipped_nonadjacent_gaps"] == 1
    assert stats["overlap_probability"] == .5
    assert stats["mean_shared_fraction"] == pytest.approx(.35)


def test_conditions_keep_same_cnn_transformer_loss_and_training():
    control.configure_environment()
    a, b = (control.build_config(c, 0, "revision") for c in ("shared", "independent"))
    real = control._original_build_config("pallier2025", 0, "revision")
    assert a["model"] == b["model"] == real["model"]
    assert a["model"]["use_transformer"] and a["model"]["transformer"]["depth"] == 16
    for key in ("dataset", "loss", "text_embedding", "evaluation", "cache"):
        assert a[key] == b[key] == real[key]
    exclude = {"output_dir", "pretrained_brain_encoder_checkpoint", "warm_start_output_group"}
    scientific = lambda cfg: {k: v for k, v in cfg["training"].items() if k not in exclude}
    assert scientific(a) == scientific(b) == scientific(real)
    assert "synthetic_timing_v1_shared" in a["training"]["pretrained_brain_encoder_checkpoint"]
    assert "synthetic_timing_v1_independent" in b["training"]["pretrained_brain_encoder_checkpoint"]


def test_epoch_hook_does_not_change_group_sampler():
    train = dataset()
    sampler = SimpleNamespace(epoch=None, set_epoch=lambda e: setattr(sampler, "epoch", e))
    hook = control.EpochSampler(sampler, train)
    hook.set_epoch(3)
    assert train.epoch == sampler.epoch == 3


def test_training_test_split_guard():
    with pytest.raises(ValueError, match="only train and validation"):
        control.dataset_and_loader({}, "test")


def test_real_and_other_condition_cnn_initialization_are_rejected(monkeypatch):
    monkeypatch.setattr(control, "_original_word_validation", lambda *args: None)
    cfg = {"synthetic_control": {"condition": "shared"}}
    for checkpoint in ({"config": {}}, {"config": {"synthetic_control": {"condition": "independent"}}}):
        with pytest.raises(ValueError, match="real checkpoints are forbidden"):
            control.validate_word_checkpoint(checkpoint, cfg, None, None)
    control.validate_word_checkpoint({"config": cfg}, cfg, None, None)


def test_include_onset_seconds_passthrough():
    onsets = (0.25, 0.75, 1.40)
    ds_default = SyntheticTimingDataset(
        fake_base(onsets=onsets), "shared", split="val", include_onset_seconds=False
    )
    ds_with_onset = SyntheticTimingDataset(
        fake_base(onsets=onsets), "shared", split="val", include_onset_seconds=True
    )
    item_default = ds_default[1]
    item_with_onset = ds_with_onset[1]
    assert "onset_seconds" not in item_default
    assert "onset_seconds" in item_with_onset
    assert item_with_onset["onset_seconds"].dtype == np.float32 or str(item_with_onset["onset_seconds"].dtype) == "torch.float32"
    assert float(item_with_onset["onset_seconds"]) == pytest.approx(0.75, abs=1e-6)
    np.testing.assert_allclose(item_default["meg"].numpy(), item_with_onset["meg"].numpy(), atol=1e-6)


@pytest.mark.parametrize(
    "dev_str",
    ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",),
)
def test_gpu_batched_streams_overlap_epoch_and_label_invariance(dev_str):
    dev = torch.device(dev_str)
    gen = torch.Generator(device=dev)
    gen.manual_seed(20261124)
    mixed = synthesize_streams_batch_gpu(
        [50, 172, 936], channels=16, sfreq=50.0, generator=gen, device=dev
    )
    assert mixed.shape == (3, 16, 936)
    assert torch.isfinite(mixed).all()

    # Shared vs independent sample-sharing invariant
    shared = dataset("shared", "train", onsets=(0.0, 0.3, 0.6))
    indep = dataset("independent", "train", onsets=(0.0, 0.3, 0.6))
    raw_shared = shared.materialize_windows_gpu(device=dev, apply_transform=False)
    raw_indep = indep.materialize_windows_gpu(device=dev, apply_transform=False)
    torch.testing.assert_close(raw_shared[0, :, 15:], raw_shared[1, :, :35], rtol=0, atol=0)
    assert not torch.equal(raw_indep[0, :, 15:], raw_indep[1, :, :35])

    # Transform matches CPU transform_window bit-for-bit
    tr_shared = shared.materialize_windows_gpu(device=dev, apply_transform=True).cpu().numpy()
    expected_tr = np.stack(
        [
            transform_window(
                raw_shared[i].cpu().numpy(),
                baseline_samples=shared.baseline_samples,
                clamp=shared.clamp,
            )
            for i in range(len(shared))
        ],
        axis=0,
    )
    np.testing.assert_allclose(tr_shared, expected_tr, rtol=1e-6, atol=1e-6)

    # Train epoch refresh vs fixed val/test & temporal shift independence
    w_ep0 = shared.materialize_windows_gpu(device=dev, apply_transform=True)
    shared.set_epoch(1)
    w_ep1 = shared.materialize_windows_gpu(device=dev, apply_transform=True)
    assert not torch.equal(w_ep0, w_ep1)
    shared.set_epoch(0)
    torch.testing.assert_close(
        w_ep0, shared.materialize_windows_gpu(device=dev, apply_transform=True), rtol=0, atol=0
    )

    shared.temporal_shift_seconds = -15.0
    w_shift = shared.materialize_windows_gpu(device=dev, apply_transform=True)
    assert not torch.equal(w_ep0, w_shift)
    shared.temporal_shift_seconds = 0.0

    val_a = SyntheticTimingDataset(fake_base(), "shared", split="val", training_seed=0)
    val_b = SyntheticTimingDataset(fake_base(), "shared", split="val", training_seed=7)
    v0 = val_a.materialize_windows_gpu(device=dev, apply_transform=True)
    val_a.set_epoch(9)
    torch.testing.assert_close(v0, val_a.materialize_windows_gpu(device=dev, apply_transform=True), rtol=0, atol=0)
    torch.testing.assert_close(v0, val_b.materialize_windows_gpu(device=dev, apply_transform=True), rtol=0, atol=0)

    # Label invariance
    base_renamed = fake_base()
    base_renamed.table.normalized_word = ["other_a", "other_b", "other_c"]
    val_renamed = SyntheticTimingDataset(base_renamed, "shared", split="val", training_seed=0)
    torch.testing.assert_close(
        v0, val_renamed.materialize_windows_gpu(device=dev, apply_transform=True), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU preextraction fast path")
def test_gpu_cnn_preextraction_and_inplace_refresh():
    dev = torch.device("cuda")

    class DummyBrainEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8 * 50, 32, bias=False)
            self.subj_emb = nn.Embedding(4, 32)

        def forward(self, x, subj_idx):
            return self.proj(x.flatten(1)) + self.subj_emb(subj_idx)

    dummy_model = SimpleNamespace(brain_encoder=DummyBrainEncoder().to(dev).eval())
    onsets = (0.0, 0.3, 0.0, 0.2, 0.5, 0.9)
    groups = ["g_long", "g_long", "g_short", "g_mid", "g_mid", "g_mid"]
    subjects = ["sub-01", "sub-01", "sub-02", "sub-01", "sub-01", "sub-01"]
    ds = SyntheticTimingDataset(
        fake_base(onsets=onsets, group_ids=groups, subjects=subjects),
        "shared",
        split="train",
    )

    # Batch-size invariance & exact match to manual forward on materialized windows
    z_bs2 = ds.extract_cnn_embeddings_gpu(
        dummy_model, device=dev, batch_size=2, amp=False, show_progress=False
    )
    z_bs64 = _extract_single_dataset_cnn_embeddings(
        dummy_model,
        ds,
        device=dev,
        batch_size=64,
        amp=False,
        temporal_shift_seconds=0.0,
        shift_mode="preroll",
        show_progress=False,
    )
    torch.testing.assert_close(z_bs2, z_bs64, rtol=1e-5, atol=1e-5)

    win_all = ds.materialize_windows_gpu(device=dev, apply_transform=True)
    subj_all = torch.as_tensor(ds.subject_indices, device=dev, dtype=torch.long)
    with torch.no_grad():
        expected_z = F.normalize(dummy_model.brain_encoder(win_all, subj_all), dim=-1).float().cpu()
    torch.testing.assert_close(z_bs64, expected_z, rtol=1e-5, atol=1e-5)

    # In-place VRAM refresh updates rows [0:N] and keeps padding row [N] strictly zero
    n_rows = len(ds)
    clean_pad = torch.zeros((n_rows + 1, 32), device=dev, dtype=torch.bfloat16)
    shift_pad = torch.zeros((n_rows + 1, 32), device=dev, dtype=torch.bfloat16)
    cache = {
        "train_clean_z": clean_pad[:n_rows],
        "train_clean_z_pad": clean_pad,
        "train_shifted_z": {-15.0: shift_pad[:n_rows]},
        "train_shifted_z_pad": {-15.0: shift_pad},
    }
    ds.set_epoch(2)
    refresh_preextracted_train_features(
        cache,
        dummy_model,
        ds,
        shift_seconds=(-15.0,),
        shift_mode="preroll",
        device=dev,
        batch_size=3,
        amp=True,
        amp_dtype=torch.bfloat16,
        show_progress=False,
    )
    assert float(clean_pad[n_rows].abs().max().item()) == 0.0
    assert float(shift_pad[n_rows].abs().max().item()) == 0.0
    assert not torch.allclose(cache["train_clean_z"], cache["train_shifted_z"][-15.0])
    assert not torch.allclose(cache["train_clean_z"].float().cpu(), z_bs64, atol=1e-3)


