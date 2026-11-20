"""PNPL 薄适配器（LibriBrain100 / Pallier2025 / Armeni2022）单元测试与契约验证。"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding.data.derived_cache import canonical_cache_paths, validate_event_product
from braindecoding.data.event_table import CORE_EVENT_COLUMNS, validate_event_table
from braindecoding.data.pnpl_dataset_adapter import (
    DerivedMegWordDataset,
    load_pnpl_event_table,
    materialize_pnpl_dataset,
    open_pnpl_dataset,
    preprocess_continuous_meg,
    resolve_split_and_unit,
)
from braindecoding.data.synthetic_timing_dataset import SyntheticTimingDataset
import braindecoding.data.text_embeddings as text_data


def _write_fake_h5(
    path: Path,
    *,
    channels: int,
    samples: int,
    sfreq: float,
    channel_names: list[str],
    seed: int = 0,
    first_samp: int = 0,
    highpass: float = 0.1,
    lowpass: float = 125.0,
) -> Path:
    rng = np.random.default_rng(seed)
    data = (rng.standard_normal((channels, samples)) * 15.0 + 3.5).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)
        handle.create_dataset("times", data=(np.arange(samples, dtype=np.float32) / float(sfreq)))
        handle.attrs["sample_frequency"] = float(sfreq)
        handle.attrs["highpass_cutoff"] = float(highpass)
        handle.attrs["lowpass_cutoff"] = float(lowpass)
        handle.attrs["channel_names"] = ", ".join(channel_names)
        handle.attrs["first_samp"] = int(first_samp)
    return path


class _FakePnplDataset:
    """离线替身，完整模拟 PNPL 连续 H5 与 events.tsv 访问接口。"""

    def __init__(
        self,
        *,
        data_path: Path,
        run_keys: list[tuple[str, str, str, str]],
        h5_map: dict[tuple[str, str, str, str], Path],
        events_map: dict[tuple[str, str, str, str], Path],
        sfreq: float,
        standardize_enabled: bool = False,
        records: list[Any] | None = None,
    ) -> None:
        self.data_path = str(data_path)
        self.run_keys = list(run_keys)
        self._h5_map = dict(h5_map)
        self._events_map = dict(events_map)
        self.sfreq = float(sfreq)
        self.standardize_enabled = bool(standardize_enabled)
        self._records = records or []

    def get_h5_path(self, subject: str, session: str, task: str, run: str) -> str:
        return str(self._h5_map[(subject, session, task, run)])

    def get_events_path(self, subject: str, session: str, task: str, run: str) -> str:
        return str(self._events_map[(subject, session, task, run)])

    def ensure_file(self, path: str) -> str:
        return path


def test_open_pnpl_dataset_enforces_standardize_false_for_all_three_datasets(tmp_path):
    captured: dict[str, dict] = {}

    class _Recorder:
        def __init__(self, **kwargs):
            captured[self.__class__.__name__] = kwargs

    class LibriBrain100(_Recorder):
        pass

    class Pallier2025(_Recorder):
        pass

    class Armeni2022(_Recorder):
        pass

    fake_mod = SimpleNamespace(
        LibriBrain100=LibriBrain100,
        Pallier2025=Pallier2025,
        Armeni2022=Armeni2022,
    )

    open_pnpl_dataset("libribrain100", tmp_path, download=False, pnpl_datasets_module=fake_mod)
    open_pnpl_dataset(
        "pallier2025",
        tmp_path,
        subjects=["sub-01", "2"],
        runs=["run-01", "7"],
        download=False,
        pnpl_datasets_module=fake_mod,
    )
    open_pnpl_dataset(
        "armeni2022",
        tmp_path,
        subjects=["001"],
        sessions=["001"],
        download=False,
        pnpl_datasets_module=fake_mod,
    )

    for cls_name in ("LibriBrain100", "Pallier2025", "Armeni2022"):
        assert captured[cls_name]["standardize"] is False
        assert captured[cls_name]["clipping_boundary"] is None
        assert hasattr(captured[cls_name]["task"], "collect_samples")

    assert captured["LibriBrain100"]["subjects"] == "deep"
    assert captured["Pallier2025"]["preprocessing"] == "bp+ds"
    assert captured["Pallier2025"]["preprocessing_config"] == {
        "bp": {"l_freq": 0.1, "h_freq": 40.0},
        "ds": {"sfreq": 50.0},
    }
    assert captured["Pallier2025"]["include_subjects"] == ["01", "02"]
    assert captured["Pallier2025"]["include_runs"] == ["01", "07"]
    assert captured["Armeni2022"]["preprocessing"] == "bp+ds"
    assert captured["Armeni2022"]["include_subjects"] == ["001"]
    assert captured["Armeni2022"]["include_sessions"] == ["001"]


def test_open_pnpl_dataset_libribrain100_subject_and_session_task_filters(
    tmp_path, monkeypatch
):
    captured: list[dict] = []

    class LibriBrain100:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    fake_mod = SimpleNamespace(LibriBrain100=LibriBrain100)
    fake_manifest = SimpleNamespace(
        RUN_RECORDS=(
            SimpleNamespace(run_key=("0", "1", "Sherlock1", "1")),
            SimpleNamespace(run_key=("0", "11", "Sherlock1", "2")),
            SimpleNamespace(run_key=("0", "0", "Sherlock9", "1")),
            SimpleNamespace(run_key=("1", "1", "Sherlock1", "1")),
        )
    )
    orig_import = importlib.import_module

    def _fake_import(name: str, package=None):
        if name == "pnpl.datasets.libribrain100.manifest":
            return fake_manifest
        return orig_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import)

    open_pnpl_dataset(
        "libribrain100",
        tmp_path,
        subjects=["sub-0"],
        sessions=["ses-1", "11"],
        tasks=["Sherlock1"],
        download=False,
        pnpl_datasets_module=fake_mod,
    )
    assert captured[-1]["subjects"] == ["0"]
    assert captured[-1]["include_run_keys"] == [
        ("0", "1", "Sherlock1", "1"),
        ("0", "11", "Sherlock1", "2"),
    ]

    open_pnpl_dataset(
        "libribrain100",
        tmp_path,
        subjects="broad",
        tasks=["Sherlock1"],
        download=False,
        pnpl_datasets_module=fake_mod,
    )
    assert captured[-1]["subjects"] == "broad"
    assert captured[-1]["include_run_keys"] == [("1", "1", "Sherlock1", "1")]

    open_pnpl_dataset(
        "libribrain100",
        tmp_path,
        include_run_keys=[("1", "1", "Sherlock1", "1")],
        download=False,
        pnpl_datasets_module=fake_mod,
    )
    assert captured[-1]["subjects"] == "all"
    assert captured[-1]["include_run_keys"] == [("1", "1", "Sherlock1", "1")]


def test_preprocess_continuous_meg_bandpasses_resamples_and_robust_scales():
    rng = np.random.default_rng(42)
    # 250 Hz × 40 s = 10000 samples -> 50 Hz × 40 s = 2000 samples（超过 0.1 Hz FIR 滤长）
    raw_250hz = rng.standard_normal((6, 10000)) * 50.0 + 100.0
    ch_names = [f"MEG011{i + 1}" for i in range(6)]
    scaled, details = preprocess_continuous_meg(
        raw_250hz,
        source_sampling_rate_hz=250.0,
        channel_names=ch_names,
        source_highpass_hz=0.1,
        source_lowpass_hz=125.0,
    )
    assert scaled.dtype == np.float32
    assert scaled.shape == (6, 2000)
    assert details["bandpass_applied"] is True
    assert details["resample_applied"] is True
    assert details["pnpl_standardize"] is False
    assert details["scaler"] == "RobustScaler"
    # RobustScaler 后各通道中位数应归零且 IQR 接近 1
    assert np.allclose(np.median(scaled, axis=1), 0.0, atol=1e-4)
    q75, q25 = np.percentile(scaled, [75, 25], axis=1)
    assert np.allclose(q75 - q25, 1.0, atol=1e-3)


def test_materialize_libribrain100_and_downstream_dataset_and_synthetic_control(tmp_path):
    pnpl_root = tmp_path / "pnpl_raw"
    out_root = tmp_path / "derived_libribrain"
    ch_names = ["MEG0111", "MEG0112", "MEG0113", "MEG0121", "MEG0122", "MEG0123"]

    # 构造 SimpleB2T 表 3 中的 train(ses-1, ses-2) / val(ses-11) / test(ses-7) 会话
    run_specs = [
        ("0", "1", "Sherlock1", "1", "train", "sherlock"),
        ("0", "2", "Sherlock1", "1", "train", "sherlock"),
        ("0", "11", "Sherlock1", "2", "validation", "sherlock"),
        ("0", "7", "Sherlock1", "1", "test", "sherlock"),
    ]
    h5_map = {}
    events_map = {}
    records = []
    for idx, (sub, ses, task, run, part, corpus) in enumerate(run_specs):
        rk = (sub, ses, task, run)
        h5_p = _write_fake_h5(
            pnpl_root / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}.h5",
            channels=len(ch_names),
            samples=250 * 24,  # 24 s @ 250 Hz -> 1200 samples @ 50 Hz
            sfreq=250.0,
            channel_names=ch_names,
            seed=idx + 10,
            lowpass=125.0,
        )
        ev_p = pnpl_root / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv"
        # 包含词行、音素干扰行、以及尾部越界词（22.5 + 3.0 > 24.0）
        ev_df = pd.DataFrame(
            {
                "kind": ["word", "phoneme", "word", "word", "word", "word"],
                "segment": ["holmes", "hh", "looked", "at", "the", "door"],
                "timemeg": [2.0, 2.05, 2.35, 2.75, 4.10, 22.50],
                "duration": [0.30, 0.05, 0.35, 0.15, 0.15, 0.25],
                "sentenceidx": [0, 0, 0, 0, 1, 2],
                "wordidx": [0, -1, 1, 2, 0, 0],
            }
        )
        ev_df.to_csv(ev_p, sep="\t", index=False)
        h5_map[rk] = h5_p
        events_map[rk] = ev_p
        records.append(SimpleNamespace(run_key=rk, partition=part, corpus=corpus))

    fake_ds = _FakePnplDataset(
        data_path=pnpl_root,
        run_keys=[r.run_key for r in records],
        h5_map=h5_map,
        events_map=events_map,
        sfreq=250.0,
        standardize_enabled=False,
        records=records,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = materialize_pnpl_dataset(
            "libribrain100",
            pnpl_root,
            out_root,
            split_policy="simpleb2t",
            pnpl_dataset=fake_ds,
        )
    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert summary["recording_count"] == 4
    assert summary["materialized_recordings"] == 4
    assert summary["reused_recordings"] == 0
    assert summary["event_count"] == 20  # 4 runs × 5 word rows
    assert summary["trainable_event_count"] == 16  # 4 runs × 4 in-bounds words
    assert summary["split_counts"]["train"]["trainable_event_count"] == 8
    assert summary["split_counts"]["val"]["trainable_event_count"] == 4
    assert summary["split_counts"]["test"]["trainable_event_count"] == 4

    # 验证磁盘事件表与 manifest 自摘要完全合法
    manifest = validate_event_product(summary["events_path"], summary["event_manifest_path"])
    assert manifest["dataset"] == "libribrain100"
    table = load_pnpl_event_table(summary["events_path"])
    assert tuple(table.columns[:14]) == CORE_EVENT_COLUMNS
    assert validate_event_table(table) == ()

    # 二次调用应完全复用 50 Hz 缓存
    summary_reused = materialize_pnpl_dataset(
        "libribrain100",
        pnpl_root,
        out_root,
        split_policy="simpleb2t",
        pnpl_dataset=fake_ds,
    )
    assert summary_reused["reused_recordings"] == 4
    assert summary_reused["materialized_recordings"] == 0

    # 构建文本向量缓存并验证 DerivedMegWordDataset + SyntheticTimingDataset 端到端兼容
    text_cfg = {
        "model_name": "t5-large",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "embedding_dimension": 16,
        "word_normalization": "canonical_standard_word",
    }
    words = sorted(set(table["标准词"].astype(str)))
    rng = np.random.default_rng(7)
    emb_path = out_root / "text" / "embeddings.npz"
    text_data.save_text_embedding_cache(
        emb_path, words, rng.standard_normal((len(words), 16)).astype(np.float32), text_cfg
    )

    test_table = load_pnpl_event_table(summary["events_path"], split="test", trainable_only=True)
    train_table = load_pnpl_event_table(summary["events_path"], split="train", trainable_only=True)
    ds_cfg = {
        "subjects": ["sub-0"],
        "target_sampling_rate_hz": 50.0,
        "window_seconds": 1.0,
        "baseline_seconds": 0.5,
        "clamp": 5.0,
    }
    test_ds = DerivedMegWordDataset(
        test_table,
        ds_cfg,
        summary["signals_dir"],
        emb_path,
        text_cfg,
    )
    sample = test_ds[0]
    assert sample["meg"].shape == (6, 50)
    assert sample["text_embedding"].shape == (16,)
    assert float(sample["meg"].abs().max()) <= 5.0 + 1e-6
    assert test_ds.channel_positions.shape == (6, 2)

    # 验证因果 Pre-Roll 错位模式使用独立 train 记录且可正常采样
    test_ds.ensure_preroll_buffers(
        shift_seconds=-10.0,
        shift_mode="preroll",
        preroll_seconds=15.0,
    )
    shifted_sample = test_ds[0]
    assert shifted_sample["meg"].shape == (6, 50)
    assert not torch.equal(sample["meg"], shifted_sample["meg"])

    # 验证现有 SyntheticTimingDataset 联合合成脉冲对照及错位平滑脉冲流分支无缝兼容
    synth_ds = SyntheticTimingDataset(
        test_ds,
        "shared",
        split="test",
        include_onset_seconds=True,
    )
    synth_sample = synth_ds[0]
    assert synth_sample["meg"].shape == (6, 50)
    assert "onset_seconds" in synth_sample
    assert synth_ds.ensure_preroll_buffers(-12.0) == {}
    synth_ds.temporal_shift_seconds = -12.0
    synth_shifted_12 = synth_ds[0]
    synth_ds.temporal_shift_seconds = -19.0
    synth_shifted_19 = synth_ds[0]
    assert not torch.equal(synth_sample["meg"], synth_shifted_12["meg"])
    assert not torch.equal(synth_shifted_12["meg"], synth_shifted_19["meg"])
    assert len(train_table) == 8


def test_materialize_cleanup_raw_deletes_temp_h5_and_reuses_without_h5(tmp_path):
    pnpl_root = tmp_path / "pnpl_raw"
    out_root = tmp_path / "derived_cleanup"
    ch_names = ["MEG0111", "MEG0112", "MEG0113", "MEG0121"]
    rk = ("0", "1", "Sherlock1", "1")
    h5_p = _write_fake_h5(
        pnpl_root / "sub-0_ses-1_task-Sherlock1_run-1.h5",
        channels=4,
        samples=50 * 16,
        sfreq=50.0,
        channel_names=ch_names,
        seed=9,
        highpass=0.1,
        lowpass=40.0,
    )
    ev_p = pnpl_root / "sub-0_ses-1_task-Sherlock1_run-1_events.tsv"
    pd.DataFrame(
        {
            "kind": ["word", "word", "word"],
            "segment": ["holmes", "looked", "up"],
            "timemeg": [2.0, 2.35, 2.75],
            "duration": [0.30, 0.30, 0.20],
            "sentenceidx": [0, 0, 0],
            "wordidx": [0, 1, 2],
        }
    ).to_csv(ev_p, sep="\t", index=False)

    fake_ds = _FakePnplDataset(
        data_path=pnpl_root,
        run_keys=[rk],
        h5_map={rk: h5_p},
        events_map={rk: ev_p},
        sfreq=50.0,
        standardize_enabled=False,
        records=[SimpleNamespace(run_key=rk, partition="train", corpus="sherlock")],
    )
    summary = materialize_pnpl_dataset(
        "libribrain100",
        pnpl_root,
        out_root,
        split_policy="simpleb2t",
        corpus="sherlock",
        cleanup_raw=True,
        pnpl_dataset=fake_ds,
    )
    assert summary["cleanup_raw"] is True
    assert summary["cleaned_raw_files"] == 1
    assert not h5_p.exists()

    # 二次续跑（H5 已被清理）应直接复用已落盘的 50 Hz .npy，不报错不重新读取 H5
    resumed = materialize_pnpl_dataset(
        "libribrain100",
        pnpl_root,
        out_root,
        split_policy="simpleb2t",
        corpus="sherlock",
        cleanup_raw=True,
        pnpl_dataset=fake_ds,
    )
    assert resumed["reused_recordings"] == 1
    assert resumed["materialized_recordings"] == 0


def test_materialize_pallier2025_preserves_first_samp_and_simpleb2t_chapter_split(tmp_path):
    pnpl_root = tmp_path / "pallier_raw"
    out_root = tmp_path / "derived_pallier"
    ch_names = ["MEG0111", "MEG0112", "MEG0113", "MEG0121"]

    run_keys = [
        ("01", "01", "listen", "01"),  # train
        ("01", "01", "listen", "07"),  # val
        ("01", "01", "listen", "08"),  # test
    ]
    h5_map = {}
    events_map = {}
    for idx, rk in enumerate(run_keys):
        sub, ses, task, run = rk
        # 50 Hz 信号，first_samp=500 (@ 50 Hz -> 10.0 s 起点)
        h5_map[rk] = _write_fake_h5(
            pnpl_root / f"sub-{sub}_run-{run}.h5",
            channels=4,
            samples=50 * 20,  # 20 s -> 覆盖 [10.0 s, 30.0 s]
            sfreq=50.0,
            channel_names=ch_names,
            seed=idx + 1,
            first_samp=500,
            highpass=0.1,
            lowpass=40.0,
        )
        ev_p = pnpl_root / f"sub-{sub}_run-{run}_events.tsv"
        # onset=12.0s 对应相对 2.0s（合法）；onset=13.2s 与前词间隔 >0.5s 应自动分新句
        pd.DataFrame(
            {
                "onset": [12.0, 12.3, 13.5],
                "duration": [0.25, 0.20, 0.25],
                "trial_type": ["Word", "Word", "Word"],
                "stimulus": ["Le", "Petit", "Prince"],
            }
        ).to_csv(ev_p, sep="\t", index=False)
        events_map[rk] = ev_p

    fake_ds = _FakePnplDataset(
        data_path=pnpl_root,
        run_keys=run_keys,
        h5_map=h5_map,
        events_map=events_map,
        sfreq=50.0,
        standardize_enabled=False,
    )
    summary = materialize_pnpl_dataset(
        "pallier2025",
        pnpl_root,
        out_root,
        split_policy="simpleb2t",
        pnpl_dataset=fake_ds,
    )
    table = load_pnpl_event_table(summary["events_path"])
    assert summary["split_counts"]["train"]["trainable_event_count"] == 3
    assert summary["split_counts"]["val"]["trainable_event_count"] == 3
    assert summary["split_counts"]["test"]["trainable_event_count"] == 3
    # 验证停顿间隔自动切句：前两词同句，第三词间隔 1.0s 分到下一句
    run01 = table[table["运行编号"].eq("run-01")]
    assert run01["上下文编号"].nunique() == 2
    assert run01["标准词"].tolist() == ["le", "petit", "prince"]


def test_materialize_armeni2022_extracts_word_onsets_and_ctf_split(tmp_path):
    pnpl_root = tmp_path / "armeni_raw"
    out_root = tmp_path / "derived_armeni"
    ch_names = ["MLC11", "MLC12", "MRC11", "MRC12"]
    run_keys = [
        ("001", "001", "compr", "01"),  # train
        ("001", "009", "compr", "01"),  # val
        ("001", "010", "compr", "01"),  # test
    ]
    h5_map = {}
    events_map = {}
    for idx, rk in enumerate(run_keys):
        sub, ses, task, _ = rk
        h5_map[rk] = _write_fake_h5(
            pnpl_root / f"sub-{sub}_ses-{ses}_task-{task}.h5",
            channels=4,
            samples=50 * 15,
            sfreq=50.0,
            channel_names=ch_names,
            seed=idx + 100,
            highpass=0.1,
            lowpass=40.0,
        )
        ev_p = pnpl_root / f"sub-{sub}_ses-{ses}_task-{task}_events.tsv"
        pd.DataFrame(
            {
                "onset": [1.0, 1.05, 1.4, 2.5],
                "duration": [0.3, 0.05, 0.25, 0.3],
                "type": ["word_onset", "phoneme_onset_01", "word_onset", "word_onset"],
                "value": ["Sherlock", "SH", "Holmes", "smiled"],
            }
        ).to_csv(ev_p, sep="\t", index=False)
        events_map[rk] = ev_p

    fake_ds = _FakePnplDataset(
        data_path=pnpl_root,
        run_keys=run_keys,
        h5_map=h5_map,
        events_map=events_map,
        sfreq=50.0,
        standardize_enabled=False,
    )
    summary = materialize_pnpl_dataset(
        "armeni2022",
        pnpl_root,
        out_root,
        split_policy="simpleb2t",
        pnpl_dataset=fake_ds,
    )
    table = load_pnpl_event_table(summary["events_path"])
    assert len(table) == 9  # 仅保留 3 个 word_onset × 3 sessions，过滤 phoneme_onset_01
    assert set(table["标准词"]) == {"sherlock", "holmes", "smiled"}
    assert resolve_split_and_unit("armeni2022", ("001", "009", "compr", "01")) == (
        "val",
        "armeni2022|ses-009",
    )


def test_rejects_pnpl_dataset_with_standardize_enabled(tmp_path):
    fake_ds = _FakePnplDataset(
        data_path=tmp_path,
        run_keys=[("0", "1", "Sherlock1", "1")],
        h5_map={},
        events_map={},
        sfreq=250.0,
        standardize_enabled=True,
    )
    with pytest.raises(RuntimeError, match="standardize=False"):
        materialize_pnpl_dataset("libribrain100", tmp_path, tmp_path / "out", pnpl_dataset=fake_ds)


def test_canonical_cache_paths_supports_pnpl_datasets():
    for name in ("libribrain100", "pallier2025_full", "armeni2022"):
        paths = canonical_cache_paths(name)
        assert paths["event_table"].startswith(f"data/cache/derived/{name}/events/")
        assert paths["meg_dir"].startswith(f"data/cache/derived/{name}/signals/")


def test_real_pnpl_installed_signatures_match_adapter():
    """在安装有 pnpl 的环境（如 pnpl2026）下验证真实类签名与适配器参数完全兼容。"""
    try:
        datasets_mod = importlib.import_module("pnpl.datasets")
    except ImportError:
        pytest.skip("当前 Python 环境未安装 pnpl，跳过真实 pnpl 签名检查。")

    for cls_name, required_params in (
        (
            "LibriBrain100",
            {"data_path", "task", "partition", "subjects", "corpus", "standardize", "clipping_boundary", "download"},
        ),
        (
            "Pallier2025",
            {"data_path", "task", "preprocessing", "preprocessing_config", "standardize", "clipping_boundary", "download"},
        ),
        (
            "Armeni2022",
            {"data_path", "task", "preprocessing", "preprocessing_config", "standardize", "clipping_boundary", "download"},
        ),
    ):
        cls = getattr(datasets_mod, cls_name)
        params = set(inspect.signature(cls.__init__).parameters)
        assert required_params <= params, f"{cls_name} 缺少参数：{required_params - params}"


def test_preprocess_continuous_meg_threaded_mne_n_jobs_matches_single_thread():
    rng = np.random.default_rng(7)
    raw_250hz = rng.standard_normal((8, 10000)) * 20.0
    ch_names = [f"MEG011{i + 1}" for i in range(8)]
    ref, ref_details = preprocess_continuous_meg(
        raw_250hz,
        source_sampling_rate_hz=250.0,
        channel_names=ch_names,
        source_highpass_hz=0.1,
        source_lowpass_hz=125.0,
        mne_n_jobs=1,
    )
    mt, mt_details = preprocess_continuous_meg(
        raw_250hz,
        source_sampling_rate_hz=250.0,
        channel_names=ch_names,
        source_highpass_hz=0.1,
        source_lowpass_hz=125.0,
        mne_n_jobs=0,
    )
    assert ref_details["mne_n_jobs"] == 1
    assert mt_details["mne_n_jobs"] >= 1
    assert np.allclose(ref, mt, atol=1e-5, rtol=1e-5)


def test_materialize_libribrain100_cleanup_raw_streams_one_run_at_a_time_and_zero_redownload_resume(tmp_path):
    pnpl_root = tmp_path / "pnpl_raw"
    out_root = tmp_path / "derived_stream"
    ch_names = ["MEG0111", "MEG0112", "MEG0113", "MEG0121"]
    run_keys = [
        ("0", "1", "Sherlock1", "1"),
        ("0", "2", "Sherlock1", "1"),
    ]
    created_h5s: dict[tuple[str, str, str, str], Path] = {}
    open_calls: list[list[tuple[str, str, str, str]]] = []
    h5_alive_at_open: list[int] = []

    class _StreamingFakeLibriBrain100:
        def __init__(self, **kwargs):
            rks = [tuple(map(str, rk)) for rk in kwargs["include_run_keys"]]
            open_calls.append(rks)
            # 记录本次打开时磁盘上尚存的旧 H5 数量：必须为 0（上一条已清理）
            h5_alive_at_open.append(sum(1 for p in created_h5s.values() if p.exists()))
            self.data_path = str(pnpl_root)
            self.run_keys = rks
            self.sfreq = 50.0
            self.standardize_enabled = False
            self._records = [
                SimpleNamespace(run_key=rk, partition="train", corpus="sherlock")
                for rk in rks
            ]
            self._h5_map = {}
            self._events_map = {}
            for rk in rks:
                sub, ses, task, run = rk
                h5_p = _write_fake_h5(
                    pnpl_root / task / "derivatives" / "serialised" / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_meg.h5",
                    channels=4,
                    samples=50 * 16,
                    sfreq=50.0,
                    channel_names=ch_names,
                    seed=int(ses),
                    highpass=0.1,
                    lowpass=40.0,
                )
                ev_p = pnpl_root / task / "derivatives" / "events" / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv"
                ev_p.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "kind": ["word", "word", "word"],
                        "segment": ["holmes", "walked", "home"],
                        "timemeg": [2.0, 2.4, 2.8],
                        "duration": [0.3, 0.3, 0.2],
                        "sentenceidx": [0, 0, 0],
                        "wordidx": [0, 1, 2],
                    }
                ).to_csv(ev_p, sep="\t", index=False)
                created_h5s[rk] = h5_p
                self._h5_map[rk] = h5_p
                self._events_map[rk] = ev_p

        def get_h5_path(self, subject, session, task, run):
            return str(self._h5_map[(str(subject), str(session), str(task), str(run))])

        def get_events_path(self, subject, session, task, run):
            return str(self._events_map[(str(subject), str(session), str(task), str(run))])

        def close_h5_files(self):
            return None

    fake_mod = SimpleNamespace(LibriBrain100=_StreamingFakeLibriBrain100)
    summary = materialize_pnpl_dataset(
        "libribrain100",
        pnpl_root,
        out_root,
        split_policy="simpleb2t",
        corpus="sherlock",
        include_run_keys=run_keys,
        cleanup_raw=True,
        pnpl_datasets_module=fake_mod,
    )
    assert len(open_calls) == 2
    assert open_calls == [[run_keys[0]], [run_keys[1]]]
    assert h5_alive_at_open == [0, 0]
    assert summary["cleaned_raw_files"] == 2
    assert all(not p.exists() for p in created_h5s.values())

    # 断点续跑：所有 50 Hz .npy 与本地 events.tsv 已存在，应 0 次调用 LibriBrain100（零重下 H5）
    open_calls.clear()
    resumed = materialize_pnpl_dataset(
        "libribrain100",
        pnpl_root,
        out_root,
        split_policy="simpleb2t",
        corpus="sherlock",
        include_run_keys=run_keys,
        cleanup_raw=True,
        pnpl_datasets_module=fake_mod,
    )
    assert open_calls == []
    assert resumed["reused_recordings"] == 2
    assert resumed["materialized_recordings"] == 0
