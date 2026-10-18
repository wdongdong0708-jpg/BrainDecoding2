"""Scientific controls: shared sample identity, split RNG and unchanged model."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scipy.ndimage import gaussian_filter1d
from braindecoding.data.synthetic_timing import SyntheticTimingDataset, overlap_statistics, transform_window, synthetic_stream
from braindecoding import synthetic_timing_control as control

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
