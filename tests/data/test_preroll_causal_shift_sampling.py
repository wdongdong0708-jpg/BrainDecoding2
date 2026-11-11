from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding.data import chineseeeg2_dataset as ceeg_mod
from braindecoding.data import pallier2025_dataset as pallier_mod
from braindecoding.data.derived_cache import sha256_file
from braindecoding.data.text_embeddings import text_embedding_signature


def _canonical_vectorview_layout_and_channels() -> tuple[Path, list[str]]:
    spec = importlib.util.find_spec("mne")
    assert spec is not None and spec.origin is not None
    layout_path = (
        Path(spec.origin).parent / "channels" / "data" / "layouts" / "Vectorview-all.lout"
    )
    lines = [
        line.split()
        for line in layout_path.read_text(encoding="utf-8").splitlines()[1:]
        if len(line.split()) >= 7
    ]
    names = ["".join(fields[5:]) for fields in lines]
    assert len(names) == pallier_mod.MEG_CHANNEL_COUNT
    return layout_path, names


def _write_pallier_signal_fixture(
    cache_dir: Path,
    *,
    subject: str,
    run: str,
    array: np.ndarray,
    first_samp: int = 0,
) -> tuple[Path, Path]:
    _layout_path, channel_names = _canonical_vectorview_layout_and_channels()
    out_path = cache_dir / subject / f"{subject}_ses-01_task-listen_{run}_meg.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr_f32 = np.asarray(array, dtype=np.float32)
    np.save(out_path, arr_f32, allow_pickle=False)
    rec_id = f"pallier2025|{subject}|{run}"
    sidecar = {
        "schema_version": 1,
        "dataset": "pallier2025",
        "recording_id": rec_id,
        "subject": subject,
        "run": run,
        "recording_signal_status": "valid",
        "event_training_status": "eligible_by_signal_contract",
        "source_first_samp": int(first_samp),
        "source_sampling_rate_hz": 1000.0,
        "source_recording_start_seconds": float(first_samp) / 1000.0,
        "output_first_samp": int(round(first_samp * 50.0 / 1000.0)),
        "output_sampling_rate_hz": 50.0,
        "output_recording_start_seconds": float(first_samp) / 1000.0,
        "output_sample_count": int(arr_f32.shape[1]),
        "channel_count": pallier_mod.MEG_CHANNEL_COUNT,
        "channel_names": channel_names,
        "channel_names_sha256": pallier_mod.CHANNEL_NAMES_SHA256,
        "magnetometer_count": pallier_mod.MAGNETOMETER_COUNT,
        "gradiometer_count": pallier_mod.GRADIOMETER_COUNT,
        "bad_channels": [],
        "shape": list(arr_f32.shape),
        "dtype": "float32",
        "output_sha256": sha256_file(out_path),
        "preprocessing_contract_sha256": "synthetic_test_contract",
    }
    sidecar_path = out_path.with_suffix(".json")
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False), encoding="utf-8")
    return out_path, sidecar_path


def _write_pallier_text_cache(tmp_path: Path, words: list[str]) -> tuple[Path, dict]:
    text_cfg = {
        "model_name": "synthetic-test-bge",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "embedding_dimension": 16,
        "word_normalization": "canonical_standard_word",
    }
    emb_path = tmp_path / "pallier_text_embeddings.npz"
    unique_words = sorted(set(words))
    rng = np.random.default_rng(123)
    vectors = rng.standard_normal((len(unique_words), 16)).astype(np.float32)
    np.savez(
        emb_path,
        words=np.asarray(unique_words, dtype=str),
        embeddings=vectors,
    )
    emb_path.with_suffix(".json").write_text(
        json.dumps({"signature": text_embedding_signature(text_cfg)}),
        encoding="utf-8",
    )
    return emb_path, text_cfg


def _write_pallier_layout(_tmp_path: Path) -> Path:
    layout_path, _ = _canonical_vectorview_layout_and_channels()
    return layout_path


def _make_pallier_rows(
    *,
    subject: str,
    run: str,
    split: str,
    onsets: list[float],
    words: list[str],
) -> pd.DataFrame:
    rec_id = f"pallier2025|{subject}|{run}"
    mat_id = f"pallier2025|{run}"
    n = len(onsets)
    return pd.DataFrame(
        {
            "事件编号": [f"{rec_id}|event-{i:04d}" for i in range(n)],
            "受试者": [subject] * n,
            "记录编号": [rec_id] * n,
            "材料编号": [mat_id] * n,
            "划分单元": [mat_id] * n,
            "记录内序号": list(range(n)),
            "词": words,
            "标准词": words,
            "开始时间": [float(t) for t in onsets],
            "结束时间": [float(t) + 0.25 for t in onsets],
            "上下文编号": [f"{mat_id}|sequence-001"] * n,
            "数据划分": [split] * n,
            "是否可训练": [True] * n,
            "排除原因": [""] * n,
            "会话": ["ses-01"] * n,
            "运行编号": [run] * n,
        }
    )


def _create_ceeg_fixture(
    tmp_path: Path,
    *,
    runs_spec: list[tuple[str, str, np.ndarray]],
) -> tuple[Path, Path, Path, dict, dict]:
    root = tmp_path / "pl"
    eeg_cache_dir = tmp_path / "cc"
    eeg_cache_dir.mkdir(parents=True, exist_ok=True)
    subj = "sub-01"
    ch_names = ["E1", "E2", "E3", "E4"]
    ds_cfg = {
        "root": str(root),
        "subjects": [subj],
        "source_sampling_rate_hz": 100.0,
        "target_sampling_rate_hz": 50.0,
        "window_start_offset_seconds": 0.0,
        "window_seconds": 1.0,
        "source_preprocessed_filter_hz": [1.0, 40.0],
        "additional_filter": None,
        "scaler": "RobustScaler",
        "clamp": None,
    }
    relpaths: dict[str, str] = {}
    cache_files: dict[str, Path] = {}
    eeg_dir = root / "derivatives" / "preprocessed" / subj / "ses-littleprince" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "name": ch_names,
            "x": [0.0, 1.0, 0.0, -1.0],
            "y": [1.0, 0.0, -1.0, 0.0],
            "z": [1.0, 1.0, 1.0, 1.0],
        }
    ).to_csv(
        eeg_dir / f"{subj}_ses-littleprince_space-CapTrak_electrodes.tsv",
        sep="\t",
        index=False,
    )

    for run_id, _split, sig_arr in runs_spec:
        stem = f"{subj}_ses-littleprince_task-lis_run-{run_id}_eeg"
        vhdr = eeg_dir / f"{stem}.vhdr"
        ch_lines = "\n".join(
            f"Ch{idx}={nm},,1.0,uV" for idx, nm in enumerate(ch_names, start=1)
        )
        vhdr.write_text(
            "\n".join(
                [
                    "Brain Vision Data Exchange Header File Version 1.0",
                    "[Common Infos]",
                    f"DataFile={stem}.eeg",
                    "DataFormat=BINARY",
                    "DataOrientation=MULTIPLEXED",
                    f"NumberOfChannels={len(ch_names)}",
                    "SamplingInterval=10000.0",
                    "[Binary Infos]",
                    "BinaryFormat=IEEE_FLOAT_32",
                    "[Channel Infos]",
                    ch_lines,
                ]
            ),
            encoding="utf-8",
        )
        n_src = int(sig_arr.shape[1]) * 2
        vhdr.with_suffix(".eeg").write_bytes(b"\0" * (4 * len(ch_names) * n_src))
        vhdr.with_suffix(".json").write_text(
            json.dumps({"SamplingFrequency": 100.0, "RecordingDuration": float(sig_arr.shape[1]) / 50.0}),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "name": ch_names,
                "type": ["EEG"] * len(ch_names),
                "low_cutoff": [1.0] * len(ch_names),
                "high_cutoff": [40.0] * len(ch_names),
                "sampling_frequency": [100.0] * len(ch_names),
                "status": ["good"] * len(ch_names),
            }
        ).to_csv(
            eeg_dir / f"{stem.replace('_eeg', '')}_channels.tsv",
            sep="\t",
            index=False,
        )
        cache_p = ceeg_mod.处理后记录路径(vhdr, ds_cfg, eeg_cache_dir)
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        arr_f32 = np.asarray(sig_arr, dtype=np.float32)
        np.save(cache_p, arr_f32, allow_pickle=False)
        sig_meta = ceeg_mod._预处理签名(ds_cfg, vhdr)
        cache_p.with_suffix(".json").write_text(
            json.dumps(
                {
                    "signature": sig_meta,
                    "shape": list(arr_f32.shape),
                    "channel_positions": [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
                }
            ),
            encoding="utf-8",
        )
        relpaths[run_id] = vhdr.relative_to(root).as_posix()
        cache_files[run_id] = cache_p

    text_cfg = {
        "model_name": "synthetic-ceeg-bge",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "embedding_dimension": 16,
        "word_normalization": "canonical_standard_word",
    }
    emb_path = tmp_path / "ceeg_text.npz"
    words_vocab = ["小王子", "星球", "玫瑰"]
    np.savez(
        emb_path,
        words=np.asarray(words_vocab, dtype=str),
        embeddings=np.ones((len(words_vocab), 16), dtype=np.float32),
    )
    emb_path.with_suffix(".json").write_text(
        json.dumps({"signature": text_embedding_signature(text_cfg)}),
        encoding="utf-8",
    )
    return root, eeg_cache_dir, emb_path, ds_cfg, {"text_cfg": text_cfg, "relpaths": relpaths, "cache_files": cache_files}


def test_pallier2025_preroll_zero_wrap_and_future_perturbation_invariance(tmp_path: Path):
    """反事实检验 1 & 2 (Pallier2025)：早词 (t=0.2s) 错位绝不环绕至测试集尾部，且未来样本扰动严格零影响。"""
    meg_cache = tmp_path / "meg_50hz"
    layout_path = _write_pallier_layout(tmp_path)
    emb_path, text_cfg = _write_pallier_text_cache(tmp_path, ["le", "petit", "prince"])

    ch = pallier_mod.MEG_CHANNEL_COUNT
    # 两条各 50 秒 (2500 采样点) 的 train 记录，合计 100 秒供 -75s 深移位自动扩容
    t_grid_train1 = np.linspace(0.0, 1.0, 2500, dtype=np.float32)
    t_grid_train2 = np.linspace(1.0, 2.0, 2500, dtype=np.float32)
    train_sig_1 = np.broadcast_to(np.sin(11.0 * t_grid_train1)[None, :], (ch, 2500)).copy()
    train_sig_2 = np.broadcast_to(np.cos(17.0 * t_grid_train2)[None, :], (ch, 2500)).copy()

    # 一条 60 秒 (3000 采样点) 的 test 记录
    t_grid_test = np.linspace(0.0, 3.0, 3000, dtype=np.float32)
    test_sig_orig = np.broadcast_to(np.sin(23.0 * t_grid_test)[None, :], (ch, 3000)).copy()

    _write_pallier_signal_fixture(meg_cache, subject="sub-01", run="run-01", array=train_sig_1)
    _write_pallier_signal_fixture(meg_cache, subject="sub-01", run="run-02", array=train_sig_2)
    test_npy_path, _ = _write_pallier_signal_fixture(
        meg_cache, subject="sub-01", run="run-06", array=test_sig_orig
    )

    train_df_1 = _make_pallier_rows(
        subject="sub-01", run="run-01", split="train", onsets=[1.0, 5.0], words=["le", "petit"]
    )
    train_df_2 = _make_pallier_rows(
        subject="sub-01", run="run-02", split="train", onsets=[1.0, 5.0], words=["petit", "prince"]
    )
    # 测试集包含极早词 t=0.2s (sample=10)、边界跨接词 t=0.5s (sample=25) 与中段词 t=35.0s (sample=1750)
    test_df = _make_pallier_rows(
        subject="sub-01",
        run="run-06",
        split="test",
        onsets=[0.2, 0.5, 35.0],
        words=["le", "petit", "prince"],
    )
    all_events = pd.concat([train_df_1, train_df_2, test_df], ignore_index=True)

    ds_cfg = {
        "subjects": ["sub-01"],
        "layout_path": str(layout_path),
        "window_seconds": 1.0,
        "window_start_offset_seconds": 0.0,
        "baseline_seconds": 0.0,  # 关闭基线均值以直接核验原始切片字节等价性
        "clamp": 1e6,
    }

    # 先验证旧 circular 模式在早词 t=0.2s, shift=-30s 时确实会环绕读取 test 记录尾部未来信号
    legacy_ds = pallier_mod.Pallier2025WordDataset(
        test_df,
        ds_cfg,
        meg_cache,
        emb_path,
        text_cfg,
        temporal_shift_seconds=-30.0,
        shift_mode="deterministic_circular",
        allow_subject_subset=True,
    )
    legacy_win_before = legacy_ds.read_meg(0).copy()

    # 构建 Pre-Roll 模式的数据集覆盖全部 3 条因果切片分支（含 -75s 自动扩容）
    shifts_to_verify = [0.0, -0.5, -10.0, -30.0, -40.0, -75.0]
    preroll_snapshots_before: dict[tuple[float, int], np.ndarray] = {}
    for shift_sec in shifts_to_verify:
        ds = pallier_mod.Pallier2025WordDataset(
            test_df,
            ds_cfg,
            meg_cache,
            emb_path,
            text_cfg,
            temporal_shift_seconds=shift_sec,
            shift_mode="preroll",
            preroll_seconds=45.0,
            preroll_source_table=all_events,
            allow_subject_subset=True,
        )
        if shift_sec == 0.0:
            sample_item = ds[0]
            assert "onset_seconds" in sample_item
            assert pytest.approx(float(sample_item["onset_seconds"].item()), abs=1e-5) == 0.2
        for idx in range(len(test_df)):
            preroll_snapshots_before[(shift_sec, idx)] = ds.read_meg(idx).copy()

    # 精确核验 t=0.2s (sample=10), shift=-30s (-1500 samples) 完全取自 train_sig_2 倒数 [2500-1490 : 2500-1440]
    expected_preroll_slice = train_sig_2[:, 2500 - 1490 : 2500 - 1490 + 50]
    np.testing.assert_allclose(
        preroll_snapshots_before[(-30.0, 0)], expected_preroll_slice, atol=0.0, rtol=0.0
    )
    # 精确核验跨边界拼接分支：t=0.5s (sample=25), shift=-0.5s (-25 samples) -> raw_start=0；
    # 以及 t=0.2s (sample=10), shift=-0.5s (-25 samples) -> raw_start=-15 (前 15 点来自 train_sig_2 末尾，后 35 点来自 test_sig_orig 开头)
    expected_stitch = np.concatenate(
        [train_sig_2[:, -15:], test_sig_orig[:, :35]], axis=1
    )
    np.testing.assert_allclose(
        preroll_snapshots_before[(-0.5, 0)], expected_stitch, atol=0.0, rtol=0.0
    )

    # 旧 circular 模式因为环绕到了 test 尾部 [1460..2950)，当篡改尾部未来信号时必然发生泄漏变化：
    meg_cache_circ = tmp_path / "meg_circ"
    mutated_tail_for_circular = test_sig_orig.copy()
    mutated_tail_for_circular[:, 1400:] = -7777.0
    _write_pallier_signal_fixture(meg_cache_circ, subject="sub-01", run="run-06", array=mutated_tail_for_circular)
    legacy_ds_mut = pallier_mod.Pallier2025WordDataset(
        test_df,
        ds_cfg,
        meg_cache_circ,
        emb_path,
        text_cfg,
        temporal_shift_seconds=-30.0,
        shift_mode="deterministic_circular",
        allow_subject_subset=True,
    )
    legacy_win_after = legacy_ds_mut.read_meg(0)
    assert float(np.max(np.abs(legacy_win_after - legacy_win_before))) > 100.0

    # 反事实检验 A：对 idx=0 (t=0.2s, 窗尾 1.2s -> sample 60)，篡改 sample >= 60 的全部未来信号，逐位移验证 0.0 差异
    meg_cache_idx0 = tmp_path / "meg_idx0"
    strict_future_for_idx0 = test_sig_orig.copy()
    strict_future_for_idx0[:, 60:] = 4321.0
    _write_pallier_signal_fixture(meg_cache_idx0, subject="sub-01", run="run-01", array=train_sig_1)
    _write_pallier_signal_fixture(meg_cache_idx0, subject="sub-01", run="run-02", array=train_sig_2)
    _write_pallier_signal_fixture(meg_cache_idx0, subject="sub-01", run="run-06", array=strict_future_for_idx0)

    for shift_sec in shifts_to_verify:
        ds_after = pallier_mod.Pallier2025WordDataset(
            test_df,
            ds_cfg,
            meg_cache_idx0,
            emb_path,
            text_cfg,
            temporal_shift_seconds=shift_sec,
            shift_mode="preroll",
            preroll_seconds=45.0,
            preroll_source_table=all_events,
            allow_subject_subset=True,
        )
        win_after_0 = ds_after.read_meg(0)
        max_diff_0 = float(np.max(np.abs(win_after_0 - preroll_snapshots_before[(shift_sec, 0)])))
        assert max_diff_0 == 0.0, f"Pallier shift={shift_sec}s idx=0 leaked future signal: diff={max_diff_0}"

    # 反事实检验 B：对全部词 idx in {0, 1, 2}，篡改最后一个词窗结束 (sample >= 1800) 后的全部未来信号，验证零差异
    meg_cache_all = tmp_path / "meg_all"
    mutated_test_sig = test_sig_orig.copy()
    mutated_test_sig[:, 1800:] = 9999.0
    _write_pallier_signal_fixture(meg_cache_all, subject="sub-01", run="run-01", array=train_sig_1)
    _write_pallier_signal_fixture(meg_cache_all, subject="sub-01", run="run-02", array=train_sig_2)
    _write_pallier_signal_fixture(meg_cache_all, subject="sub-01", run="run-06", array=mutated_test_sig)

    for shift_sec in shifts_to_verify:
        ds_after_all = pallier_mod.Pallier2025WordDataset(
            test_df,
            ds_cfg,
            meg_cache_all,
            emb_path,
            text_cfg,
            temporal_shift_seconds=shift_sec,
            shift_mode="preroll",
            preroll_seconds=45.0,
            preroll_source_table=all_events,
            allow_subject_subset=True,
        )
        for idx in range(len(test_df)):
            diff = float(np.max(np.abs(ds_after_all.read_meg(idx) - preroll_snapshots_before[(shift_sec, idx)])))
            assert diff == 0.0, f"Pallier shift={shift_sec}s idx={idx} leaked future signal: diff={diff}"


def test_chineseeeg2_preroll_zero_wrap_and_future_perturbation_invariance(tmp_path: Path):
    """反事实检验 1 & 2 (ChineseEEG2)：全 3 分支无环绕切片与严格未来扰动零敏感。"""
    t1 = np.broadcast_to(np.linspace(1.0, 2.0, 2500, dtype=np.float32)[None, :], (4, 2500)).copy()
    t2 = np.broadcast_to(np.linspace(3.0, 4.0, 2500, dtype=np.float32)[None, :], (4, 2500)).copy()
    test_orig = np.broadcast_to(np.linspace(-1.0, 1.0, 2000, dtype=np.float32)[None, :], (4, 2000)).copy()

    _root, eeg_cache_dir, emb_path, ds_cfg, extra = _create_ceeg_fixture(
        tmp_path,
        runs_spec=[("11", "train", t1), ("12", "train", t2), ("13", "test", test_orig)],
    )
    rel = extra["relpaths"]
    cache_files = extra["cache_files"]
    text_cfg = extra["text_cfg"]

    def _ceeg_rows(run_id: str, split: str, starts: list[int], words: list[str]) -> pd.DataFrame:
        n = len(starts)
        rec_id = f"chineseeeg2|sub-01|run-{run_id}"
        return pd.DataFrame(
            {
                "event_id": [f"{rec_id}|ev-{i}" for i in range(n)],
                "subject_id": ["sub-01"] * n,
                "recording_id": [rec_id] * n,
                "vhdr_relpath": [rel[run_id]] * n,
                "normalized_word": words,
                "sentence_uid": [f"{rec_id}|sent-0"] * n,
                "split": [split] * n,
                "is_trainable": [True] * n,
                "window_start_target_sample": starts,
                "target_sample_count": [50] * n,
                "target_sampling_rate_hz": [50.0] * n,
                "onset_seconds": [float(s) / 50.0 for s in starts],
            }
        )

    tr_df = pd.concat(
        [
            _ceeg_rows("11", "train", [50, 200], ["小王子", "星球"]),
            _ceeg_rows("12", "train", [50, 200], ["星球", "玫瑰"]),
        ],
        ignore_index=True,
    )
    te_df = _ceeg_rows("13", "test", [10, 1200], ["小王子", "玫瑰"])
    full_df = pd.concat([tr_df, te_df], ignore_index=True)

    shifts = [0.0, -0.4, -20.0, -75.0]
    before_map: dict[tuple[float, int], np.ndarray] = {}
    for sh in shifts:
        ds = ceeg_mod.ChineseEEG2LittlePrinceWordDataset(
            te_df,
            ds_cfg,
            eeg_cache_dir,
            emb_path,
            text_cfg,
            temporal_shift_seconds=sh,
            shift_mode="preroll",
            preroll_seconds=45.0,
            preroll_source_table=full_df,
        )
        for idx in range(len(te_df)):
            before_map[(sh, idx)] = ds.read_eeg(idx).copy()
        del ds
    gc.collect()

    # 验证跨边界拼接：start=10 (0.2s), shift=-0.4s (-20 samples) -> raw_start=-10 (前 10 点来自 t2 末尾，后 40 点来自 test_orig 开头)
    expected_stitch = np.concatenate([t2[:, -10:], test_orig[:, :40]], axis=1)
    np.testing.assert_allclose(before_map[(-0.4, 0)], expected_stitch, atol=0.0, rtol=0.0)

    # 极端篡改 test_orig 在 idx=0 窗口 (sample 10 + 50 = 60) 之后的全部未来信号（写入独立缓存子目录避免 Windows mmap 文件锁）
    mutated = test_orig.copy()
    mutated[:, 60:] = -8888.0
    _root_m, eeg_cache_dir_m, emb_path_m, ds_cfg_m, extra_m = _create_ceeg_fixture(
        tmp_path / "m2",
        runs_spec=[("11", "train", t1), ("12", "train", t2), ("13", "test", mutated)],
    )
    rel_m = extra_m["relpaths"]
    tr_df_m = tr_df.copy()
    tr_df_m["vhdr_relpath"] = [rel_m["11"], rel_m["11"], rel_m["12"], rel_m["12"]]
    te_df_m = te_df.copy()
    te_df_m["vhdr_relpath"] = [rel_m["13"], rel_m["13"]]
    full_df_m = pd.concat([tr_df_m, te_df_m], ignore_index=True)

    for sh in shifts:
        ds_mut = ceeg_mod.ChineseEEG2LittlePrinceWordDataset(
            te_df_m,
            ds_cfg_m,
            eeg_cache_dir_m,
            emb_path_m,
            extra_m["text_cfg"],
            temporal_shift_seconds=sh,
            shift_mode="preroll",
            preroll_seconds=45.0,
            preroll_source_table=full_df_m,
        )
        diff_0 = float(np.max(np.abs(ds_mut.read_eeg(0) - before_map[(sh, 0)])))
        assert diff_0 == 0.0, f"ChineseEEG2 shift={sh}s idx=0 leaked future signal: diff={diff_0}"


def test_same_recording_split_guard_truncates_at_train_end_and_refuses_overflow(tmp_path: Path):
    """反事实检验 3：当 train 与 test 共享同一连续记录时，Pre-Roll 严格截断在 train 边界，绝不越界读取 test 区间。"""
    meg_cache = tmp_path / "meg_shared"
    layout_path = _write_pallier_layout(tmp_path)
    emb_path, text_cfg = _write_pallier_text_cache(tmp_path, ["le", "prince"])
    ch = pallier_mod.MEG_CHANNEL_COUNT

    # 同一记录共 4000 点 (80s)：[0, 2500) 为 train (0..50s)，[3000, 4000) 为 test (60..80s)
    shared_sig = np.ones((ch, 4000), dtype=np.float32) * 2.5
    shared_sig[:, 2500:] = -999.0  # 将 train 结束点后的全部样本设为显著毒化值 -999.0
    shared_npy, _ = _write_pallier_signal_fixture(
        meg_cache, subject="sub-01", run="run-01", array=shared_sig
    )

    tr_shared = _make_pallier_rows(
        subject="sub-01", run="run-01", split="train", onsets=[1.0, 49.0], words=["le", "prince"]
    )
    te_shared = _make_pallier_rows(
        subject="sub-01", run="run-01", split="test", onsets=[60.0], words=["le"]
    )
    shared_source_table = pd.concat([tr_shared, te_shared], ignore_index=True)
    ds_cfg = {
        "subjects": ["sub-01"],
        "layout_path": str(layout_path),
        "window_seconds": 1.0,
        "window_start_offset_seconds": 0.0,
        "baseline_seconds": 0.0,
        "clamp": 1e6,
    }

    # 1) 当移位所需样本超出可用 train 区间 (|-3500| + 50 = 3550 > 2500) 时，立即抛 ValueError 拒绝越界偷读 test 毒化区！
    with pytest.raises(ValueError, match="不足以构建所需 Pre-Roll"):
        _ = pallier_mod.Pallier2025WordDataset(
            te_shared,
            ds_cfg,
            meg_cache,
            emb_path,
            text_cfg,
            temporal_shift_seconds=-70.0,
            shift_mode="preroll",
            preroll_seconds=45.0,
            preroll_source_table=shared_source_table,
            allow_subject_subset=True,
        )
    with pytest.raises(ValueError, match="不足以构建所需 Pre-Roll"):
        _ = pallier_mod.Pallier2025WordDataset(
            te_shared,
            ds_cfg,
            meg_cache,
            emb_path,
            text_cfg,
            temporal_shift_seconds=-49.5,  # |-2475| + 50 = 2525 > 2500 -> 严禁越界偷读 2500 之后的毒化区
            shift_mode="preroll",
            preroll_seconds=45.0,
            preroll_source_table=shared_source_table,
            allow_subject_subset=True,
        )

    # 当移位需要的样本 <= 2500 时（例如 preroll_seconds=45.0, shift=-40s -> required=2250 <= 2500），
    # 且早词 start=10 (这里用显式早测试词或检查内部 _preroll_by_recording)，缓冲严格恒等于 2.5：
    ds_valid = pallier_mod.Pallier2025WordDataset(
        te_shared,
        ds_cfg,
        meg_cache,
        emb_path,
        text_cfg,
        temporal_shift_seconds=-40.0,
        shift_mode="preroll",
        preroll_seconds=45.0,
        preroll_source_table=shared_source_table,
        allow_subject_subset=True,
    )
    buf = ds_valid._preroll_by_recording["pallier2025|sub-01|run-01"].copy()
    assert buf.shape == (ch, 2250)
    assert not ds_valid._preroll_by_recording["pallier2025|sub-01|run-01"].flags.writeable
    np.testing.assert_allclose(buf, 2.5, atol=0.0, rtol=0.0)
    del ds_valid
    gc.collect()

    # 篡改 [2500:] 区间后写入独立缓存重新初始化，Pre-Roll 缓冲区 0.0 变化
    meg_cache_2 = tmp_path / "meg_shared_2"
    shared_sig_2 = shared_sig.copy()
    shared_sig_2[:, 2500:] = 123456.0
    _write_pallier_signal_fixture(
        meg_cache_2, subject="sub-01", run="run-01", array=shared_sig_2
    )
    ds_valid_2 = pallier_mod.Pallier2025WordDataset(
        te_shared,
        ds_cfg,
        meg_cache_2,
        emb_path,
        text_cfg,
        temporal_shift_seconds=-40.0,
        shift_mode="preroll",
        preroll_seconds=45.0,
        preroll_source_table=shared_source_table,
        allow_subject_subset=True,
    )
    buf_2 = ds_valid_2._preroll_by_recording["pallier2025|sub-01|run-01"]
    assert float(np.max(np.abs(buf_2 - buf))) == 0.0

    # 2) 单条纯 train 记录自身构建 train Pre-Roll 时，禁止自我尾部环绕
    pure_train_only = tr_shared.copy()
    with pytest.raises(ValueError, match="禁止使用同一记录末尾环绕"):
        _ = pallier_mod.Pallier2025WordDataset(
            pure_train_only,
            ds_cfg,
            meg_cache,
            emb_path,
            text_cfg,
            temporal_shift_seconds=-10.0,
            shift_mode="preroll",
            preroll_seconds=20.0,
            preroll_source_table=pure_train_only,
            allow_subject_subset=True,
        )
