from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding.data.single_word_long_window import SingleWordLongWindowDataset, open_signal, window_spec
from braindecoding.models import build_brain_embedding_model
from braindecoding.single_word_long_window_test_v2 import build_config, timing_rows
from braindecoding.single_word_long_window_test_v2_results import budget_mask, holm, subject_scores


def fake_base(tmp_path, *, meg=False, sample_count=250):
    signal = np.vstack([np.arange(sample_count) / 40, -np.arange(sample_count) / 50]).astype(np.float32)
    path = tmp_path / ("meg.npy" if meg else "eeg.npy")
    np.save(path, signal)
    table = pd.DataFrame(dict(event_id=["e0", "e1", "e2"], normalized_word=["a", "b", "a"],
        recording_id=["r"] * 3, subject_id=["s"] * 3, onset_seconds=[2., 2.26, 2.5],
        window_start_target_sample=[50, 63, 75], vhdr_relpath=["x"] * 3))
    cfg = dict(target_sampling_rate_hz=50., window_start_offset_seconds=0., baseline_seconds=.5, clamp=5.)
    base = SimpleNamespace(table=table, dataset_config=cfg, window_samples=50, channel_count=2,
        embedding_map={"a": np.array([1., 0.]), "b": np.array([0., 1.])},
        sentence_indices=np.zeros(3, dtype=np.int64), subject_to_index={"s": 0},
        recording_cache_paths={"x": path})
    if meg:
        base.signal_paths = {"r": path}
        base.signal_metadata = {"r": dict(source_first_samp=1000, source_sampling_rate_hz=1000)}
        base.subject_indices = np.zeros(3, dtype=np.int64)
    return base, signal


def test_requested_window_quantization_is_explicit():
    assert window_spec(1.27)["samples"] == 64
    assert window_spec(1.27)["actual_seconds"] == 1.28
    assert window_spec(1.5)["samples"] == 75
    with pytest.raises(ValueError):
        window_spec(1.27, 250)


@pytest.mark.parametrize("meg", [False, True])
def test_extension_preserves_all_original_samples_and_labels(tmp_path, meg):
    base, signal = fake_base(tmp_path, meg=meg)
    original = SingleWordLongWindowDataset(base, 1.)
    for seconds in (1.27, 1.5):
        extended = SingleWordLongWindowDataset(base, seconds)
        key = "meg" if meg else "eeg"
        for i in range(len(base.table)):
            assert torch.equal(extended[i][key][:, :50], original[i][key])
            assert extended[i][key].shape == (2, window_spec(seconds)["samples"])
            assert extended[i]["event_id"] == original[i]["event_id"]
        assert np.array_equal(extended.raw_window(0), signal[:, 50:50 + extended.window_samples])
        assert extended.bounds_audit["dropped_events"] == extended.bounds_audit["padded_events"] == 0
    open_signal.cache_clear()


def test_pallier_uses_recording_origin_before_rounding(tmp_path):
    base, _ = fake_base(tmp_path, meg=True)
    data = SingleWordLongWindowDataset(base, 1.27)
    assert data.starts.tolist() == [50, 63, 75]
    base.signal_metadata["r"]["source_first_samp"] = 0
    other = SingleWordLongWindowDataset(base, 1.27)
    assert other.starts.tolist() == [100, 113, 125]
    open_signal.cache_clear()


def test_incomplete_long_window_is_rejected_without_padding(tmp_path):
    base, _ = fake_base(tmp_path, sample_count=130)
    SingleWordLongWindowDataset(base, 1.)
    with pytest.raises(ValueError, match="no row may be silently dropped"):
        SingleWordLongWindowDataset(base, 1.5)
    open_signal.cache_clear()


def test_duration_budget_uses_integer_boundaries_and_real_availability(tmp_path):
    base, _ = fake_base(tmp_path)
    ds = SingleWordLongWindowDataset(base, 1.5)
    info = dict(groups=[dict(event_ids=["e0", "e1", "e2"], subject="s", block="b")])
    rows = timing_rows(ds, info)
    assert rows[0]["common"]
    assert rows[0]["ends"] == {"0": 100, "1": 113, "2": 125}
    assert rows[0]["ends"]["1"] <= rows[0]["start"] + 64
    assert rows[0]["ends"]["2"] <= rows[0]["start"] + 75
    assert rows[1]["ends"]["2"] is None
    assert not rows[1]["common"] and not rows[2]["common"]
    open_signal.cache_clear()


@pytest.mark.parametrize("dataset", ["pallier2025", "chineseeeg2_littleprince"])
def test_long_window_training_retains_word_rules_and_uses_separate_outputs(dataset):
    a = build_config(dataset, 1, "long_1p27")
    b = build_config(dataset, 1, "long_1p5")
    assert not a["model"]["use_transformer"]
    assert a["model"]["context_mode"] == "singleton"
    assert a["model"]["conv"] == b["model"]["conv"]
    assert not a["model"].get("local_window_revision")
    assert a["training"]["batch_size"] == 128 and a["training"]["epochs"] == 30
    assert a["training"]["learning_rate"] == 1e-4 and a["training"]["weight_decay"] == 0
    assert "pretrained_brain_encoder_checkpoint" not in a["training"]
    assert a["training"]["output_dir"] != b["training"]["output_dir"]
    assert a["dataset"]["window_seconds"] == 1.28


def test_independent_cnn_handles_long_lengths_and_ignores_word_groups():
    torch.set_num_threads(1)
    positions = np.array([[.1, .2], [.2, .3]], dtype=np.float32)
    cfg = dict(embedding_dimension=8, use_transformer=False, context_mode="singleton",
        conv=dict(merger_channels=4, merger_position_dimension=8, merger_per_subject=True,
            merger_dropout=0., initial_linear=4, hidden=4, depth=2, kernel_size=3,
            dilation_growth=2, dilation_period=2, dropout_input=0., convolution_dropout=0.,
            batch_norm=True, gelu=True, skip=True, glu_every=2, glu_context=1, temporal_attention_hidden=4))
    model = build_brain_embedding_model(2, positions, 1, cfg).eval()
    for length in (50, 64, 75):
        x = torch.randn(3, 2, length)
        subject = torch.zeros(3, dtype=torch.long)
        with torch.inference_mode():
            a = model(x, subject, torch.tensor([0, 0, 0]))
            b = model(x, subject, torch.tensor([0, 1, 2]))
            shuffled = model(x[[2, 0, 1]], subject, torch.tensor([1, 1, 0]))
        assert a.shape == (3, 8)
        assert torch.allclose(a.norm(dim=-1), torch.ones(3), atol=1e-5)
        assert torch.equal(a, b)
        assert torch.allclose(a[[2, 0, 1]], shuffled, atol=1e-6)
    assert model.context_transformer is None and not model.prefix_revision_enabled


def test_subject_equal_estimator_does_not_reduce_to_event_micro():
    counts = np.array([[9., 1.], [1., 1.]])
    hits = np.zeros((2, 3, 2, 2, 2))
    hits[0, :, 0, 0, 0] = 9
    hits[1, :, 0, :, 1] = 1
    observed = subject_scores(np.ones((1, 2)), counts, hits)
    assert observed.shape == (2, 2, 1)
    assert observed[0, 0, 0] == pytest.approx(.45)
    assert observed[1, 0, 0] == pytest.approx(.5)
    # Event micro would have the opposite direction: 2/12 - 9/12.
    assert observed[1, 0, 0] - observed[0, 0, 0] == pytest.approx(.05)


def test_duration_selection_does_not_use_labels_or_predictions():
    timing = {"a": dict(start=100, common=True, ends={"1": 164, "2": 175}),
              "b": dict(start=100, common=True, ends={"1": 165, "2": 176}),
              "c": dict(start=100, common=False, ends={"1": None, "2": None})}
    assert budget_mask(["a", "b", "c"], timing, 1, 64).tolist() == [True, False, False]
    assert budget_mask(["a", "b", "c"], timing, 2, 75).tolist() == [True, False, False]


def test_holm_primary_family_and_full_rng_recovery(tmp_path, monkeypatch):
    import random
    from braindecoding.single_word_long_window_test_v2 import rng_state, restore_rng
    from braindecoding.training.runtime import load_checkpoint, save_checkpoint
    assert np.allclose(holm([.03, .01, .001, .2]), [.06, .03, .004, .2])
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    state = rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    path = tmp_path / "rng.pt"
    save_checkpoint(path, {"rng": state})
    restore_rng(load_checkpoint(path)["rng"])
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])


def test_full_export_keeps_four_primary_comparisons_and_six_model_rows(tmp_path, monkeypatch):
    from braindecoding import single_word_long_window_test_v2 as study
    from braindecoding import single_word_long_window_test_v2_results as results
    monkeypatch.setattr(study, "REPORT", tmp_path / "reports")
    monkeypatch.setattr(study, "verify_manifest", lambda m: None)
    monkeypatch.setattr(study, "verify_sources", lambda m, **kw: None)
    ids = np.asarray([f"event-{i}" for i in range(8)])
    subjects = np.asarray(["s0", "s1"] * 4)
    words = np.asarray(["a", "b"] * 4)
    meta = {e: dict(block=f"b{i//2}", common=True, subject=subjects[i], word=words[i]) for i, e in enumerate(ids)}
    info = dict(blocks=[dict(block=f"b{i}", scope="run") for i in range(4)],
                vocabulary=["a", "b"] + [f"w{i}" for i in range(18)], vocabulary50=["a", "b"])
    rank_map = {}
    for condition in results.CONDITIONS:
        rank = np.ones((3, 8))
        if condition.startswith("long_"):
            rank[:, ::2] = 20
        rank_map[condition] = rank
    data = dict(info=info, meta=meta, ids=ids, words=words, subjects=subjects, common=np.ones(8, bool),
        covered=np.ones(8, bool), ranks=rank_map, strict_ranks=rank_map, sources={})
    monkeypatch.setattr(results, "load_dataset", lambda *args: data)
    timing = [dict(event_id=e, start=i*50, common=True,
        ends={"1": i*50+60, "2": i*50+70}) for i, e in enumerate(ids)]
    original_read = study.read
    def reader(path):
        if path == study.PARENT:
            return {}
        if path == study.RUNTIME / "timing_by_event.json":
            return {name: timing for name in study.neural.DATASETS}
        return original_read(path)
    monkeypatch.setattr(study, "read", reader)
    m = dict(manifest_sha256="toy", parent_manifest_sha256="parent",
        settings=dict(statistics=dict(random_seed=161803, bootstrap_repetitions=200, sign_flip_repetitions=200)))
    output = results.export(m)
    primary = [r for r in output["comparisons"] if r["population"] == "within_long_window_budget"]
    assert len(output["comparisons"]) == 8 and len(primary) == 4
    assert all(r["top10"]["effect"] == pytest.approx(.5) for r in primary)
    assert all("p_holm" in r["top10"] for r in primary)
    assert len([r for r in output["summaries"] if r["population"] == "common_words"
                and r["candidates"] == "full_training_vocabulary"]) == 12
    assert (study.REPORT / "long_window_comparison.png").is_file()
    assert (study.REPORT / "audit.json").is_file()
