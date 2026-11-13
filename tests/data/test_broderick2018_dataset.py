from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import torch

from braindecoding.config import PROJECT_ROOT
from braindecoding.data import broderick2018_dataset as broderick_mod
from braindecoding.models import SpatialChannelMerger, build_brain_embedding_model


def _write_synthetic_broderick_raw(
    raw_root: Path,
    *,
    subject_dir: str = "Subject1",
    run_idx: int = 1,
    fs: float = 128.0,
    duration_seconds: float = 15.0,
    include_text: bool = True,
) -> Path:
    eeg_dir = raw_root / "Natural Speech" / "EEG" / subject_dir
    priv_dir = raw_root / "private"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    priv_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(100 + run_idx)
    n_samples = int(round(duration_seconds * fs))
    eeg_data = rng.standard_normal((n_samples, 128)).astype(np.float64) * 5.0
    mastoids = rng.standard_normal((n_samples, 2)).astype(np.float64) * 12.0 + 3.0
    mat_path = eeg_dir / f"{subject_dir}_Run{run_idx}.mat"
    scipy.io.savemat(
        str(mat_path),
        {
            "eegData": eeg_data,
            "mastoids": mastoids,
            "fs": np.array([[float(fs)]]),
        },
    )

    words_spec = [
        {"case": "success", "alignedWord": "the", "word": "The", "start": 1.0, "end": 1.2},
        {"case": "success", "alignedWord": "old", "word": "old", "start": 1.25, "end": 1.45},
        {"case": "success", "alignedWord": "<unk>", "word": "foobar", "start": 1.5, "end": 1.7},
        {"case": "not-found-in-audio", "alignedWord": "ghost", "word": "ghost", "start": 1.75, "end": 1.9},
        {"case": "success", "alignedWord": "man", "word": "man", "start": 1.5, "end": 1.8},
        {"case": "success", "alignedWord": "went", "word": "went", "start": 2.4, "end": 2.65},
        {"case": "success", "alignedWord": "out", "word": "out", "start": 2.7, "end": 2.95},
        {"case": "success", "alignedWord": "to", "word": "to", "start": 3.0, "end": 3.15},
        {"case": "success", "alignedWord": "sea", "word": "sea", "start": 3.2, "end": 3.6},
    ]
    align_path = priv_dir / f"align{run_idx}.json"
    align_path.write_text(json.dumps({"words": words_spec}), encoding="utf-8")
    if include_text:
        text_path = priv_dir / f"oldman_run{run_idx}.txt"
        text_path.write_text("The old man! Went out to sea.", encoding="utf-8")
    return raw_root


def test_frozen_run_split_manifest_is_16_2_2_and_validates_hash():
    manifest_path = PROJECT_ROOT / "data/manifests/broderick2018/run_split.json"
    assert manifest_path.is_file()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["strategy"]["salt"] == "broderick2018-main-run-split-v1"
    assignments = payload["assignments"]
    assert len(assignments["train"]) == 16
    assert len(assignments["val"]) == 2
    assert len(assignments["test"]) == 2

    split_map = broderick_mod.load_run_split(manifest_path)
    assert len(split_map) == 20
    assert set(split_map.keys()) == set(broderick_mod.RUNS)
    assert set(split_map.values()) == {"train", "val", "test"}


def test_build_event_table_filters_gentle_align_and_supports_text_and_silence_fallback(
    tmp_path: Path,
):
    raw_with_text = _write_synthetic_broderick_raw(tmp_path / "with_text", include_text=True)
    split_subset = {"run-01": "train"}
    cfg = {
        "root": str(raw_with_text),
        "subjects": ["sub-01"],
        "runs": ["run-01"],
        "allow_subject_subset": True,
        "allow_run_subset": True,
        "split_by_run": split_subset,
        "min_matched_words": 4,
        "max_context_words": 128,
        "silence_sentence_gap_seconds": 0.45,
    }
    table_text, _ = broderick_mod.build_event_table(cfg)
    # 7 条有效词（剔除 <unk> 与 not-found-in-audio）；"The old man." -> sent 0 (3 词), "Went out to sea." -> sent 1 (4 词)
    assert list(table_text["标准词"]) == ["the", "old", "man", "went", "out", "to", "sea"]
    assert len(set(table_text["上下文编号"])) == 2
    assert list(table_text["原始句子编号"]) == [0, 0, 0, 1, 1, 1, 1]

    # 无文本回退：词间静音 >= 0.45s（man end=1.8 -> went start=2.4，静音=0.6s >= 0.45s）同样准确断句
    raw_no_text = _write_synthetic_broderick_raw(tmp_path / "no_text", include_text=False)
    cfg_no_text = {**cfg, "root": str(raw_no_text)}
    table_gap, _ = broderick_mod.build_event_table(cfg_no_text)
    assert list(table_gap["标准词"]) == ["the", "old", "man", "went", "out", "to", "sea"]
    assert list(table_gap["原始句子编号"]) == [0, 0, 0, 1, 1, 1, 1]


def test_mastoid_rereference_contract_switch_and_sampling_rate_guard(tmp_path: Path):
    raw_root = _write_synthetic_broderick_raw(tmp_path / "raw_signal", duration_seconds=20.0)
    paths = broderick_mod.recording_paths(raw_root, "sub-01", "run-01")
    eeg_data, mastoids, fs_hz = broderick_mod._load_mat_eeg_and_mastoids(
        Path(paths["eeg_mat"]), load_mastoids=True
    )
    assert fs_hz == 128.0

    sig_default, meta_default = broderick_mod._process_eeg_array(
        eeg_data,
        source_sampling_rate_hz=fs_hz,
        dataset_config={"mastoid_rereference": False, "mne_n_jobs": 1},
        mastoids=mastoids,
    )
    sig_reref, meta_reref = broderick_mod._process_eeg_array(
        eeg_data,
        source_sampling_rate_hz=fs_hz,
        dataset_config={"mastoid_rereference": True, "mne_n_jobs": 1},
        mastoids=mastoids,
    )
    assert sig_default.shape == sig_reref.shape
    assert sig_default.shape[0] == 128
    assert meta_default["output_sampling_rate_hz"] == 50.0
    assert meta_reref["output_sampling_rate_hz"] == 50.0
    assert not np.allclose(sig_default, sig_reref)

    c_false = broderick_mod.signal_preprocessing_contract({"mastoid_rereference": False})
    c_true = broderick_mod.signal_preprocessing_contract({"mastoid_rereference": True})
    assert c_false["mastoid_rereference"] is False
    assert c_true["mastoid_rereference"] is True
    assert c_false["preprocessing_contract_sha256"] != c_true["preprocessing_contract_sha256"]

    with pytest.raises(ValueError, match="source_sampling_rate_hz"):
        broderick_mod._process_eeg_array(
            eeg_data,
            source_sampling_rate_hz=512.0,
            dataset_config={"mastoid_rereference": False},
        )


def test_biosemi128_positions_pass_through_spatial_channel_merger_without_model_edits(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("BRAINDATA_ROOT", str(tmp_path / "braindata"))
    positions = broderick_mod.biosemi128_channel_positions()
    assert positions.shape == (128, 3)
    assert positions.dtype == np.float32
    assert np.isfinite(positions).all()

    merger = SpatialChannelMerger(
        positions, output_channels=270, position_dimension=2048, subject_count=19
    )
    x = torch.randn(2, 128, 50)
    subject_idx = torch.tensor([0, 18], dtype=torch.long)
    out = merger(x, subject_idx)
    assert out.shape == (2, 270, 50)

    # 通过完整 Word Encoder 构建链路验证与 BioSemi-128 [128, 3] 坐标兼容
    from braindecoding.tasks.word_decoding.broderick2018.train import load_config

    cfg = load_config("configs/word_decoding/broderick2018/all_subjects/main_word.yaml")
    model = build_brain_embedding_model(128, positions, 19, cfg["model"])
    pred = model(x, subject_idx)
    assert pred.shape == (2, 1024)


def test_portable_manifest_path_handles_external_paths():
    external = Path("D:/some_external_drive/Broderick2018/cache/signal.npy")
    result = broderick_mod._portable_manifest_path(external)
    assert isinstance(result, str) and "signal.npy" in result
