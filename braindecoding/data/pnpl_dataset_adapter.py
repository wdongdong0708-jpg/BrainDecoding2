"""基于牛津 PNPL 工具库的统一 MEG 数据集薄适配器。

调用 ``pnpl``（强制关闭其内部 z-score 标准化 ``standardize=False``）完成
``LibriBrain100``、全量 ``Pallier2025``（法语《小王子》听觉 MEG）与
``Armeni2022`` 的自动拉取、连续 H5 读取与词级事件提取，并将其转存为本仓库
统一的 ``50 Hz + per-recording RobustScaler`` 派生信号缓存与中文规范
``event_table``，同时提供可直接接入现有训练/合成对照/评估管线的
``DerivedMegWordDataset``。
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from braindecoding.config import PROJECT_ROOT, project_path
from braindecoding.data.derived_cache import (
    canonical_cache_paths,
    sha256_file,
    stable_sha256,
    validate_event_product,
    write_event_product,
)
from braindecoding.data.event_table import (
    CORE_EVENT_COLUMNS,
    text_fingerprint,
    validate_event_table,
)
from braindecoding.data.pallier2025_dataset import (
    DEFAULT_PREROLL_SECONDS,
    RUNTIME_CONTEXT_COLUMN,
    VALID_TEMPORAL_SHIFT_MODES,
    add_runtime_context_uid,
    channel_names_sha256,
    recording_onset_to_sample,
)
from braindecoding.data.sensor_coordinates import vectorview_channel_positions
import braindecoding.data.text_embeddings as text_data


SUPPORTED_DATASETS = ("libribrain100", "pallier2025", "armeni2022")

TARGET_SAMPLING_RATE_HZ = 50.0
FILTER_LOW_HZ = 0.1
FILTER_HIGH_HZ = 40.0
WINDOW_START_OFFSET_SECONDS = 0.0
WINDOW_SECONDS = 1.0
ELIGIBILITY_WINDOW_SECONDS = 3.0
BASELINE_SECONDS = 0.5
CLAMP = 5.0
DEFAULT_PAUSE_GAP_SECONDS = 0.5
DEFAULT_MAX_CONTEXT_WORDS = 128

# SimpleB2T (arXiv:2609.40359v1) 表 3：LibriBrain100 0 号受试者会话级划分。
LIBRIBRAIN100_SIMPLEB2T_TRAIN_SESSIONS = frozenset(
    {
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "8",
        "9",
        "10",
        "12",
        "17",
        "19",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
    }
)
LIBRIBRAIN100_SIMPLEB2T_VAL_SESSIONS = frozenset({"11"})
LIBRIBRAIN100_SIMPLEB2T_TEST_SESSIONS = frozenset(
    {"0", "7", "13", "14", "15", "16", "18", "20", "21", "22", "29", "30"}
)

# SimpleB2T 附录 A.4：Pallier2025 按书籍章节划分（1-6 训练 / 7 验证 / 8-9 测试）。
PALLIER2025_SIMPLEB2T_SPLIT_BY_RUN = {
    **{f"{index:02d}": "train" for index in range(1, 7)},
    "07": "val",
    "08": "test",
    "09": "test",
}

# SimpleB2T 附录 A.4 / Jayalath 2025 表 6：Armeni2022 按会话划分（1-8 训练 / 9 验证 / 10 测试）。
ARMENI2022_SIMPLEB2T_SPLIT_BY_SESSION = {
    **{f"{index:03d}": "train" for index in range(1, 9)},
    "009": "val",
    "010": "test",
}

PERSISTED_AUXILIARY_COLUMNS = (
    "会话",
    "任务",
    "运行编号",
    "语料",
    "原事件行号",
    "词性",
    "是否内容词",
    "信号相对路径",
)

PERSISTED_COLUMNS = (*CORE_EVENT_COLUMNS, *PERSISTED_AUXILIARY_COLUMNS)


class _RunDiscoveryTask:
    """最小化 PNPL Task 探针：仅用于触发运行键发现与 H5 拉取，不做词类过滤。"""

    def __init__(self, tmin: float = 0.0, tmax: float = 0.02) -> None:
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self._label_info = {
            "classes": ["_probe"],
            "label_to_id": {"_probe": 0},
            "id_to_label": {0: "_probe"},
            "n_classes": 1,
        }

    @property
    def label_info(self) -> dict[str, Any]:
        return dict(self._label_info)

    def collect_samples(self, dataset: Any) -> list[tuple]:
        run_keys = getattr(dataset, "run_keys", ())
        return [
            (str(sub), str(ses), str(task), str(run), 0.0, "_probe")
            for sub, ses, task, run in run_keys
        ]

    def get_label(self, sample: tuple) -> int:
        del sample
        return 0


def _import_pnpl_datasets():
    try:
        return importlib.import_module("pnpl.datasets")
    except ImportError as exc:
        raise ImportError(
            "使用 PNPL 数据适配器需要安装 `pnpl`（例如在 `pnpl2026` 环境中运行，"
            "或执行 `pip install pnpl`）。"
        ) from exc


def _normalize_dataset_name(dataset_name: str) -> str:
    name = str(dataset_name).strip().lower()
    aliases = {
        "libribrain": "libribrain100",
        "libribrain100": "libribrain100",
        "pallier": "pallier2025",
        "pallier2025": "pallier2025",
        "pallier2025_full": "pallier2025",
        "littleprince": "pallier2025",
        "armeni": "armeni2022",
        "armeni2022": "armeni2022",
    }
    if name not in aliases:
        raise ValueError(
            f"不支持的 PNPL 数据集：{dataset_name!r}，支持范围：{SUPPORTED_DATASETS}"
        )
    return aliases[name]


def _format_subject_id(subject: str) -> str:
    raw = str(subject).strip()
    return raw if raw.startswith("sub-") else f"sub-{raw}"


def _strip_prefix(value: str, prefix: str) -> str:
    text = str(value).strip()
    return text[len(prefix) :] if text.startswith(prefix) else text


def _normalize_token(value: Any, *, preserve_accents: bool = True) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().strip('"').strip("'").strip()
    if not text:
        return ""
    text = text.lower()
    if preserve_accents:
        return text
    return "".join(
        ch for ch in text if ch.isalnum() or ch in {"-", "'"}
    )


def _parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"无法解析布尔值：{value!r}")


def _write_array_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _decode_attr_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8", errors="replace")
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(raw_value, np.ndarray):
        raw_value = raw_value.tolist()
    if isinstance(raw_value, Sequence):
        result = []
        for item in raw_value:
            if isinstance(item, bytes):
                item = item.decode("utf-8", errors="replace")
            token = str(item).strip()
            if token:
                result.append(token)
        return result
    return [str(raw_value).strip()]


def _infer_channel_types(
    channel_names: Sequence[str],
    raw_types: Sequence[str] | None = None,
) -> list[str]:
    valid_mne_types = {"mag", "grad", "eeg", "eog", "ecg", "stim", "misc", "seeg"}
    if raw_types and len(raw_types) == len(channel_names):
        normalized = [str(t).strip().lower() for t in raw_types]
        if all(t in valid_mne_types for t in normalized):
            return normalized
    inferred: list[str] = []
    for name in channel_names:
        clean = str(name).replace(" ", "").upper()
        if clean.startswith("MEG") and len(clean) >= 7:
            inferred.append("mag" if clean.endswith("1") else "grad")
        else:
            inferred.append("mag")
    return inferred


def open_pnpl_dataset(
    dataset_name: str,
    data_path: str | Path,
    *,
    subjects: Any = None,
    corpus: Any = "all",
    sessions: Sequence[str] | None = None,
    runs: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    include_run_keys: Sequence[tuple[str, str, str, str]] | None = None,
    exclude_run_keys: Sequence[tuple[str, str, str, str]] | None = None,
    preprocessing: str | None = None,
    preprocessing_config: Mapping[str, Any] | None = None,
    download: bool = True,
    preload_files: bool = False,
    preload_h5: bool = False,
    create_h5_if_missing: bool = True,
    pnpl_datasets_module: Any = None,
) -> Any:
    """实例化 PNPL 数据集并强制关闭其内置 z-score 标准化。"""
    canonical_name = _normalize_dataset_name(dataset_name)
    datasets_mod = pnpl_datasets_module or _import_pnpl_datasets()
    task = _RunDiscoveryTask(tmin=0.0, tmax=0.02)
    data_dir = str(Path(data_path))

    if canonical_name == "libribrain100":
        cls = getattr(datasets_mod, "LibriBrain100")
        kwargs: dict[str, Any] = {
            "data_path": data_dir,
            "task": task,
            "partition": None,
            "subjects": "deep" if subjects is None else subjects,
            "corpus": corpus,
            "standardize": False,
            "clipping_boundary": None,
            "download": bool(download),
            "preload_files": bool(preload_files),
            "preload_h5": bool(preload_h5),
        }
        if preprocessing is not None:
            kwargs["preprocessing_str"] = preprocessing
        if preprocessing_config is not None:
            kwargs["preprocessing_config"] = dict(preprocessing_config)
        if include_run_keys:
            kwargs["include_run_keys"] = [tuple(map(str, rk)) for rk in include_run_keys]
        if exclude_run_keys:
            kwargs["exclude_run_keys"] = [tuple(map(str, rk)) for rk in exclude_run_keys]
        return cls(**kwargs)

    # Pallier2025 与 Armeni2022 默认复现 d'Ascoli / SimpleB2T 的 0.1-40 Hz + 50 Hz 管线
    default_preproc = "bp+ds" if preprocessing is None else preprocessing
    default_preproc_cfg = (
        {
            "bp": {"l_freq": FILTER_LOW_HZ, "h_freq": FILTER_HIGH_HZ},
            "ds": {"sfreq": TARGET_SAMPLING_RATE_HZ},
        }
        if preprocessing_config is None and default_preproc == "bp+ds"
        else dict(preprocessing_config or {})
    )

    if canonical_name == "pallier2025":
        cls = getattr(datasets_mod, "Pallier2025")
        include_subjects = (
            [_strip_prefix(s, "sub-") for s in subjects]
            if isinstance(subjects, (list, tuple, set))
            else ([_strip_prefix(subjects, "sub-")] if isinstance(subjects, str) and subjects != "all" else None)
        )
        include_runs = (
            [_strip_prefix(r, "run-") for r in runs] if runs else None
        )
        include_sessions = (
            [_strip_prefix(s, "ses-") for s in sessions] if sessions else None
        )
        return cls(
            data_path=data_dir,
            task=task,
            preprocessing=default_preproc,
            preprocessing_config=default_preproc_cfg,
            include_subjects=include_subjects,
            include_sessions=include_sessions,
            include_tasks=list(tasks) if tasks else None,
            include_runs=include_runs,
            include_run_keys=[tuple(map(str, rk)) for rk in include_run_keys]
            if include_run_keys
            else None,
            exclude_run_keys=[tuple(map(str, rk)) for rk in exclude_run_keys]
            if exclude_run_keys
            else None,
            standardize=False,
            clipping_boundary=None,
            create_h5_if_missing=bool(create_h5_if_missing),
            download=bool(download),
            preload_h5=bool(preload_h5),
        )

    cls = getattr(datasets_mod, "Armeni2022")
    include_subjects = (
        [_strip_prefix(s, "sub-") for s in subjects]
        if isinstance(subjects, (list, tuple, set))
        else ([_strip_prefix(subjects, "sub-")] if isinstance(subjects, str) and subjects != "all" else None)
    )
    include_sessions = (
        [_strip_prefix(s, "ses-") for s in sessions] if sessions else None
    )
    return cls(
        data_path=data_dir,
        task=task,
        preprocessing=default_preproc,
        preprocessing_config=default_preproc_cfg,
        include_subjects=include_subjects,
        include_sessions=include_sessions,
        include_tasks=list(tasks) if tasks else None,
        include_run_keys=[tuple(map(str, rk)) for rk in include_run_keys]
        if include_run_keys
        else None,
        exclude_run_keys=[tuple(map(str, rk)) for rk in exclude_run_keys]
        if exclude_run_keys
        else None,
        standardize=False,
        clipping_boundary=None,
        create_h5_if_missing=bool(create_h5_if_missing),
        download=bool(download),
        preload_h5=bool(preload_h5),
    )


def _read_h5_continuous_meg(
    h5_path: str | Path,
    *,
    fallback_sfreq: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """读取 PNPL 连续 H5 文件的原始浮点矩阵与元数据（不做 z-score）。"""
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as handle:
        dataset = handle["data"]
        data = np.asarray(dataset[:], dtype=np.float64)
        file_attrs = dict(handle.attrs)
        ds_attrs = dict(dataset.attrs)

    attrs = {**ds_attrs, **file_attrs}
    if data.ndim != 2:
        raise ValueError(f"连续 MEG H5 数据维度必须为 2D (C, T)，实际为 {data.shape}：{h5_path}")

    sfreq_raw = attrs.get("sample_frequency", attrs.get("sfreq", fallback_sfreq))
    if sfreq_raw is None:
        raise ValueError(f"H5 文件缺少采样率属性 sample_frequency/sfreq：{h5_path}")
    source_sfreq = float(sfreq_raw)
    if source_sfreq <= 0.0:
        raise ValueError(f"H5 采样率必须为正数：{source_sfreq} ({h5_path})")

    channel_names = _decode_attr_list(attrs.get("channel_names"))
    if len(channel_names) != data.shape[0]:
        channel_names = [f"MEG{idx + 1:04d}" for idx in range(data.shape[0])]
    channel_types = _infer_channel_types(
        channel_names,
        _decode_attr_list(attrs.get("channel_types")),
    )

    sensor_xyz = attrs.get("sensor_xyz")
    sensor_xyz_list = (
        np.asarray(sensor_xyz, dtype=np.float32).tolist()
        if sensor_xyz is not None
        else None
    )

    highpass = attrs.get("highpass_cutoff")
    lowpass = attrs.get("lowpass_cutoff")
    first_samp = int(attrs.get("first_samp", 0)) if attrs.get("first_samp") is not None else 0

    meta = {
        "source_h5_path": str(h5_path.resolve()),
        "source_sampling_rate_hz": source_sfreq,
        "source_sample_count": int(data.shape[1]),
        "source_first_samp": first_samp,
        "source_highpass_hz": float(highpass) if highpass is not None else None,
        "source_lowpass_hz": float(lowpass) if lowpass is not None else None,
        "channel_count": int(data.shape[0]),
        "channel_names": channel_names,
        "channel_types": channel_types,
        "channel_names_sha256": channel_names_sha256(channel_names),
        "sensor_xyz": sensor_xyz_list,
    }
    return data, meta


def _inspect_pallier_time_origin(
    pnpl_dataset: Any,
    run_key: tuple[str, str, str, str],
    h5_meta: dict[str, Any],
) -> tuple[int, float, str]:
    """若 Pallier2025 原始或预处理 FIF 可用，懒读 header 保留精确 first_samp 时间原点。"""
    subject, session, task, run = run_key
    candidates: list[Path] = []
    if hasattr(pnpl_dataset, "get_bids_raw_path"):
        try:
            candidates.append(Path(pnpl_dataset.get_bids_raw_path(subject, session, task, run)))
        except Exception:
            pass
    if hasattr(pnpl_dataset, "get_preprocessed_path") and getattr(pnpl_dataset, "preprocessing", None):
        try:
            candidates.append(
                Path(
                    pnpl_dataset.get_preprocessed_path(
                        subject,
                        session,
                        task,
                        run,
                        preprocessing=pnpl_dataset.preprocessing,
                        extension="fif",
                    )
                )
            )
        except Exception:
            pass
    for path in candidates:
        if path.is_file():
            try:
                import mne

                raw = mne.io.read_raw_fif(
                    path, preload=False, allow_maxshield=True, verbose="ERROR"
                )
                try:
                    return int(raw.first_samp), float(raw.info["sfreq"]), f"fif_header:{path.name}"
                finally:
                    raw.close()
            except Exception:
                continue
    return (
        int(h5_meta.get("source_first_samp", 0)),
        float(h5_meta["source_sampling_rate_hz"]),
        "h5_attrs",
    )


def preprocess_continuous_meg(
    data: np.ndarray,
    *,
    source_sampling_rate_hz: float,
    channel_names: Sequence[str],
    channel_types: Sequence[str] | None = None,
    target_sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ,
    filter_low_hz: float = FILTER_LOW_HZ,
    filter_high_hz: float = FILTER_HIGH_HZ,
    source_highpass_hz: float | None = None,
    source_lowpass_hz: float | None = None,
    apply_bandpass: bool = True,
    mne_n_jobs: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """执行 0.1–40 Hz 带通滤波 -> 50 Hz 重采样 -> 按通道 RobustScaler -> float32。

    与 ``pallier2025_dataset._process_picked_meg`` 保持一致：
    - 仅做记录级连续预处理，不把窗口级基线扣除与截断固化到连续缓存中；
    - 当源信号已经处于目标采样率（如 PNPL ``bp+ds`` 产出的 50 Hz H5）时，
      自动跳过会触发 Nyquist 越界的 40 Hz 低通与冗余重采样，直接执行按通道 RobustScaler。
    """
    import mne
    from sklearn.preprocessing import RobustScaler

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError(f"无效的连续 MEG 数组形状：{arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("输入连续 MEG 数组包含 NaN 或 Inf。")

    src_sfreq = float(source_sampling_rate_hz)
    tgt_sfreq = float(target_sampling_rate_hz)
    if src_sfreq <= 0.0 or tgt_sfreq <= 0.0:
        raise ValueError(f"采样率必须为正数：src={src_sfreq}, tgt={tgt_sfreq}")

    ch_names = list(channel_names)
    if len(ch_names) != arr.shape[0]:
        ch_names = [f"MEG{idx + 1:04d}" for idx in range(arr.shape[0])]
    ch_types = _infer_channel_types(ch_names, channel_types)

    already_filtered = (
        source_highpass_hz is not None
        and source_lowpass_hz is not None
        and np.isclose(float(source_highpass_hz), float(filter_low_hz), atol=1e-2)
        and np.isclose(float(source_lowpass_hz), float(filter_high_hz), atol=1e-2)
    )
    nyquist = src_sfreq / 2.0
    should_filter = (
        bool(apply_bandpass)
        and not already_filtered
        and float(filter_high_hz) < nyquist
        and arr.shape[1] > 16
    )
    should_resample = not np.isclose(src_sfreq, tgt_sfreq, rtol=0.0, atol=1e-6)

    if should_filter or should_resample:
        info = mne.create_info(ch_names=ch_names, sfreq=src_sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(arr, info, verbose=False)
        if should_filter:
            raw.filter(
                l_freq=float(filter_low_hz),
                h_freq=float(filter_high_hz),
                n_jobs=int(mne_n_jobs),
                verbose=False,
            )
        if should_resample:
            raw = raw.resample(
                tgt_sfreq,
                n_jobs=int(mne_n_jobs),
                verbose=False,
            )
        processed = np.asarray(raw.get_data(), dtype=np.float64)
        out_sfreq = float(raw.info["sfreq"])
    else:
        processed = arr
        out_sfreq = src_sfreq

    scaled = RobustScaler().fit_transform(processed.T).T.astype(np.float32, copy=False)
    if not np.isfinite(scaled).all():
        raise ValueError("RobustScaler 后的连续 MEG 出现非有限数值。")

    details = {
        "pnpl_standardize": False,
        "bandpass_applied": bool(should_filter),
        "resample_applied": bool(should_resample),
        "filter_low_hz": float(filter_low_hz),
        "filter_high_hz": float(filter_high_hz),
        "source_sampling_rate_hz": src_sfreq,
        "output_sampling_rate_hz": out_sfreq,
        "scaler": "RobustScaler",
        "scaler_scope": "per_channel_per_recording",
        "baseline_applied_in_cache": False,
        "clamp_applied_in_cache": False,
        "output_dtype": "float32",
    }
    return np.ascontiguousarray(scaled), details


def recording_identifier(dataset_name: str, run_key: tuple[str, str, str, str]) -> str:
    canonical = _normalize_dataset_name(dataset_name)
    subject, session, task, run = map(str, run_key)
    sub = _format_subject_id(subject)
    if canonical == "pallier2025":
        return f"pallier2025|{sub}|run-{_strip_prefix(run, 'run-')}"
    if canonical == "armeni2022":
        return f"armeni2022|{sub}|ses-{_strip_prefix(session, 'ses-')}|task-{task}"
    return (
        f"libribrain100|{sub}|ses-{_strip_prefix(session, 'ses-')}|"
        f"task-{task}|run-{_strip_prefix(run, 'run-')}"
    )


def recording_signal_stem(dataset_name: str, run_key: tuple[str, str, str, str]) -> str:
    canonical = _normalize_dataset_name(dataset_name)
    subject, session, task, run = map(str, run_key)
    sub = _format_subject_id(subject)
    ses = _strip_prefix(session, "ses-")
    run_tok = _strip_prefix(run, "run-")
    if canonical == "pallier2025":
        return f"{sub}_ses-{ses}_task-{task}_run-{run_tok}_meg"
    if canonical == "armeni2022":
        return f"{sub}_ses-{ses}_task-{task}_meg"
    return f"{sub}_ses-{ses}_task-{task}_run-{run_tok}_meg"


def resolve_split_and_unit(
    dataset_name: str,
    run_key: tuple[str, str, str, str],
    *,
    split_policy: str = "simpleb2t",
    pnpl_dataset: Any = None,
    split_manifest_path: str | Path | None = None,
) -> tuple[str, str]:
    """返回 ``(split, split_unit)``，保证同一划分单元永不跨 train/val/test 泄漏。"""
    canonical = _normalize_dataset_name(dataset_name)
    policy = str(split_policy).strip().lower()
    subject, session, task, run = map(str, run_key)
    del subject

    if canonical == "libribrain100":
        ses = _strip_prefix(session, "ses-")
        if policy == "simpleb2t":
            unit = f"libribrain100|ses-{ses}"
            if ses in LIBRIBRAIN100_SIMPLEB2T_TRAIN_SESSIONS:
                return "train", unit
            if ses in LIBRIBRAIN100_SIMPLEB2T_VAL_SESSIONS:
                return "val", unit
            if ses in LIBRIBRAIN100_SIMPLEB2T_TEST_SESSIONS:
                return "test", unit
            raise ValueError(
                f"LibriBrain100 会话 ses-{ses} 不在 SimpleB2T 表 3 划分集合内。"
            )
        if policy == "pnpl_official":
            unit = f"libribrain100|task-{task}|ses-{ses}"
            records = getattr(pnpl_dataset, "_records", None) or ()
            for rec in records:
                if tuple(map(str, rec.run_key)) == tuple(map(str, run_key)):
                    part = str(rec.partition).strip().lower()
                    return ("val" if part == "validation" else part), unit
            manifest_mod = importlib.import_module("pnpl.datasets.libribrain100.manifest")
            rec = manifest_mod.get_record(str(run_key[0]), ses, task, _strip_prefix(run, "run-"))
            if rec is None:
                raise ValueError(f"LibriBrain100 官方 manifest 未找到记录：{run_key}")
            part = str(rec.partition).strip().lower()
            return ("val" if part == "validation" else part), unit
        raise ValueError(
            f"LibriBrain100 不支持 split_policy={split_policy!r}，可选：simpleb2t, pnpl_official"
        )

    if canonical == "pallier2025":
        run_num = _strip_prefix(run, "run-").zfill(2)
        unit = f"pallier2025|run-{run_num}"
        if policy == "simpleb2t":
            if run_num not in PALLIER2025_SIMPLEB2T_SPLIT_BY_RUN:
                raise ValueError(f"Pallier2025 未知运行编号：run-{run_num}")
            return PALLIER2025_SIMPLEB2T_SPLIT_BY_RUN[run_num], unit
        if policy == "manifest":
            from braindecoding.data.pallier2025_dataset import load_run_split

            manifest_p = Path(
                split_manifest_path
                or (PROJECT_ROOT / "data/manifests/pallier2025/run_split.json")
            )
            assignments = load_run_split(manifest_p)
            return assignments[f"run-{run_num}"], unit
        raise ValueError(
            f"Pallier2025 不支持 split_policy={split_policy!r}，可选：simpleb2t, manifest"
        )

    # armeni2022
    ses_num = _strip_prefix(session, "ses-").zfill(3)
    unit = f"armeni2022|ses-{ses_num}"
    if policy in {"simpleb2t", "pnpl_official"}:
        if ses_num not in ARMENI2022_SIMPLEB2T_SPLIT_BY_SESSION:
            raise ValueError(f"Armeni2022 未知会话编号：ses-{ses_num}")
        return ARMENI2022_SIMPLEB2T_SPLIT_BY_SESSION[ses_num], unit
    raise ValueError(
        f"Armeni2022 不支持 split_policy={split_policy!r}，可选：simpleb2t, pnpl_official"
    )


def _group_context_indices(
    onsets: np.ndarray,
    ends: np.ndarray,
    *,
    explicit_sentence_ids: Sequence[Any] | None = None,
    pause_gap_seconds: float = DEFAULT_PAUSE_GAP_SECONDS,
    max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS,
) -> list[str]:
    """基于显式句子编号或自然停顿间隔切分局部上下文编号，且不超过 max_context_words。"""
    n = len(onsets)
    if n == 0:
        return []
    max_words = max(1, int(max_context_words))
    use_explicit = False
    parsed_explicit: list[int | None] = []
    if explicit_sentence_ids is not None and len(explicit_sentence_ids) == n:
        for val in explicit_sentence_ids:
            num = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
            if pd.notna(num) and int(num) >= 0:
                parsed_explicit.append(int(num))
            else:
                parsed_explicit.append(None)
        use_explicit = all(item is not None for item in parsed_explicit)

    labels: list[str] = []
    if use_explicit:
        chunk_counter: dict[int, int] = {}
        current_counts: dict[int, int] = {}
        for sent_id in parsed_explicit:
            assert sent_id is not None
            count = current_counts.get(sent_id, 0)
            if count >= max_words:
                chunk_counter[sent_id] = chunk_counter.get(sent_id, 0) + 1
                count = 0
            chunk_idx = chunk_counter.get(sent_id, 0)
            current_counts[sent_id] = count + 1
            if chunk_idx == 0:
                labels.append(f"sent-{sent_id:04d}")
            else:
                labels.append(f"sent-{sent_id:04d}-part-{chunk_idx:02d}")
        return labels

    sent_idx = 0
    in_group = 0
    for idx in range(n):
        if idx > 0:
            gap = float(onsets[idx]) - float(ends[idx - 1])
            if gap >= float(pause_gap_seconds) or in_group >= max_words:
                sent_idx += 1
                in_group = 0
        labels.append(f"sent-{sent_idx:04d}")
        in_group += 1
    return labels


def _resolve_pnpl_events_path(pnpl_dataset: Any, run_key: tuple[str, str, str, str]) -> Path:
    subject, session, task, run = run_key
    events_path = pnpl_dataset.get_events_path(subject, session, task, run)
    if not os.path.exists(events_path) and hasattr(pnpl_dataset, "ensure_file"):
        events_path = pnpl_dataset.ensure_file(events_path)
    path = Path(events_path)
    if not path.is_file():
        raise FileNotFoundError(f"PNPL 词事件表缺失：{path}")
    return path


def _resolve_pnpl_h5_path(pnpl_dataset: Any, run_key: tuple[str, str, str, str]) -> Path:
    subject, session, task, run = run_key
    h5_path = pnpl_dataset.get_h5_path(subject, session, task, run)
    if not os.path.exists(h5_path) and hasattr(pnpl_dataset, "ensure_file"):
        h5_path = pnpl_dataset.ensure_file(h5_path)
    path = Path(h5_path)
    if not path.is_file():
        raise FileNotFoundError(f"PNPL 连续 H5 文件缺失：{path}")
    return path


def _load_optional_pallier_extra_info(
    pnpl_dataset: Any,
    run: str,
) -> pd.DataFrame | None:
    data_path = getattr(pnpl_dataset, "data_path", None)
    if not data_path:
        return None
    run_num = _strip_prefix(run, "run-").zfill(2)
    extra_path = Path(data_path) / "sourcedata" / f"task-listen_run-{run_num}_extra_info.tsv"
    if not extra_path.is_file():
        return None
    header = extra_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    sep = "\t" if header.count("\t") > header.count(",") else ","
    df = pd.read_csv(extra_path, sep=sep)
    if "word" not in df.columns:
        return None
    words = df["word"].map(lambda w: _normalize_token(w, preserve_accents=True))
    return df.loc[words.ne("")].reset_index(drop=True)


def extract_run_word_events(
    dataset_name: str,
    events_path: str | Path,
    *,
    run_key: tuple[str, str, str, str],
    split: str,
    split_unit: str,
    signal_metadata: Mapping[str, Any],
    signal_relpath: str,
    corpus_name: str = "",
    pnpl_dataset: Any = None,
    eligibility_window_seconds: float = ELIGIBILITY_WINDOW_SECONDS,
    window_start_offset_seconds: float = WINDOW_START_OFFSET_SECONDS,
    pause_gap_seconds: float = DEFAULT_PAUSE_GAP_SECONDS,
    max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS,
) -> pd.DataFrame:
    """将单条 PNPL run 的 ``events.tsv`` 转为满足 ``CORE_EVENT_COLUMNS`` 的词事件表。"""
    canonical = _normalize_dataset_name(dataset_name)
    subject, session, task, run = map(str, run_key)
    sub_id = _format_subject_id(subject)
    rec_id = recording_identifier(canonical, run_key)

    df = pd.read_csv(events_path, sep="\t")
    preserve_accents = canonical == "pallier2025"

    if canonical == "libribrain100":
        if "kind" in df.columns:
            word_df = df[df["kind"].astype(str).str.strip().str.lower().eq("word")].copy()
        else:
            word_df = df.copy()
        word_col = "segment" if "segment" in word_df.columns else "word"
        onset_col = "timemeg" if "timemeg" in word_df.columns else "onset"
        duration_col = "duration" if "duration" in word_df.columns else None
        sent_col = "sentenceidx" if "sentenceidx" in word_df.columns else None
        material_id = (
            f"libribrain100|ses-{_strip_prefix(session, 'ses-')}|"
            f"task-{task}|run-{_strip_prefix(run, 'run-')}"
        )
    elif canonical == "pallier2025":
        if "trial_type" in df.columns:
            word_df = df[df["trial_type"].astype(str).str.strip().str.lower().eq("word")].copy()
        else:
            word_df = df.copy()
        word_col = "stimulus" if "stimulus" in word_df.columns else "word"
        onset_col = "onset"
        duration_col = "duration" if "duration" in word_df.columns else None
        sent_col = "sequence_id" if "sequence_id" in word_df.columns else None
        run_num = _strip_prefix(run, "run-").zfill(2)
        material_id = f"pallier2025|run-{run_num}"
    else:  # armeni2022
        if "type" in df.columns:
            type_series = df["type"].astype(str).str.strip().str.lower()
            mask = type_series.str.startswith("word_onset") | type_series.eq("word")
            word_df = df[mask].copy()
        elif "kind" in df.columns:
            word_df = df[df["kind"].astype(str).str.strip().str.lower().eq("word")].copy()
        elif "trial_type" in df.columns:
            word_df = df[df["trial_type"].astype(str).str.strip().str.lower().eq("word")].copy()
        else:
            word_df = df.copy()
        word_col = next(
            (col for col in ("value", "stimulus", "segment", "word") if col in word_df.columns),
            None,
        )
        if word_col is None:
            raise ValueError(f"Armeni2022 事件表缺少词列：{events_path}")
        onset_col = "onset"
        duration_col = "duration" if "duration" in word_df.columns else None
        sent_col = next(
            (col for col in ("sentenceidx", "sentence_id", "sentence") if col in word_df.columns),
            None,
        )
        ses_num = _strip_prefix(session, "ses-").zfill(3)
        material_id = f"armeni2022|ses-{ses_num}|task-{task}"

    if word_col not in word_df.columns or onset_col not in word_df.columns:
        raise ValueError(
            f"{canonical} 事件表缺少必要列 ({word_col}, {onset_col})：{events_path}"
        )

    word_df = word_df.copy()
    word_df["_orig_row"] = word_df.index.astype(int)
    word_df["_onset"] = pd.to_numeric(word_df[onset_col], errors="coerce")
    word_df = word_df[word_df["_onset"].notna() & np.isfinite(word_df["_onset"])].copy()
    if duration_col is not None:
        raw_dur = pd.to_numeric(word_df[duration_col], errors="coerce").fillna(0.0)
        word_df["_duration"] = np.maximum(raw_dur.to_numpy(dtype=np.float64), 0.0)
    else:
        word_df["_duration"] = 0.0

    # 按 onset 严格升序稳定排序，确保记录内与句内词时间戳单调非降。
    word_df = word_df.sort_values(["_onset", "_orig_row"], kind="mergesort").reset_index(
        drop=True
    )

    raw_words = word_df[word_col].fillna("").astype(str).str.strip()
    norm_words = raw_words.map(
        lambda w: _normalize_token(w, preserve_accents=preserve_accents)
    )
    onsets = word_df["_onset"].to_numpy(dtype=np.float64)
    ends = onsets + word_df["_duration"].to_numpy(dtype=np.float64)

    explicit_sents = word_df[sent_col].tolist() if sent_col is not None else None
    pos_values = (
        word_df["pos"].fillna("").astype(str).tolist()
        if "pos" in word_df.columns
        else [""] * len(word_df)
    )
    content_values = (
        [_parse_bool(v) for v in word_df["content_word"]]
        if "content_word" in word_df.columns
        else [False] * len(word_df)
    )

    if canonical == "pallier2025" and explicit_sents is None and pnpl_dataset is not None:
        extra_df = _load_optional_pallier_extra_info(pnpl_dataset, run)
        if extra_df is not None and len(extra_df) == len(word_df):
            extra_norm = extra_df["word"].map(
                lambda w: _normalize_token(w, preserve_accents=True)
            ).tolist()
            if extra_norm == norm_words.tolist():
                if "sequence_id" in extra_df.columns:
                    explicit_sents = extra_df["sequence_id"].tolist()
                if "pos" in extra_df.columns:
                    pos_values = extra_df["pos"].fillna("").astype(str).tolist()
                if "content_word" in extra_df.columns:
                    content_values = [_parse_bool(v) for v in extra_df["content_word"]]

    context_suffixes = _group_context_indices(
        onsets,
        ends,
        explicit_sentence_ids=explicit_sents,
        pause_gap_seconds=pause_gap_seconds,
        max_context_words=max_context_words,
    )

    source_first_samp = int(signal_metadata["source_first_samp"])
    source_sfreq = float(signal_metadata["source_sampling_rate_hz"])
    out_sfreq = float(signal_metadata["output_sampling_rate_hz"])
    out_samples = int(signal_metadata["output_sample_count"])
    elig_samples = int(round(float(eligibility_window_seconds) * out_sfreq))

    trainable_flags: list[bool] = []
    exclusion_reasons: list[str] = []
    for idx in range(len(word_df)):
        reasons: list[str] = []
        if not norm_words.iloc[idx]:
            reasons.append("词文本为空或规范化后为空")
        start_sample = recording_onset_to_sample(
            float(onsets[idx]) + float(window_start_offset_seconds),
            source_first_samp=source_first_samp,
            source_sampling_rate_hz=source_sfreq,
            output_sampling_rate_hz=out_sfreq,
        )
        stop_sample = start_sample + elig_samples
        if start_sample < 0 or stop_sample > out_samples:
            reasons.append(
                f"{eligibility_window_seconds:g} 秒输入窗口超出当前连续 MEG 有效范围"
            )
        trainable_flags.append(len(reasons) == 0)
        exclusion_reasons.append("；".join(reasons))

    result = pd.DataFrame(
        {
            "事件编号": [f"{rec_id}|event-{idx:05d}" for idx in range(len(word_df))],
            "受试者": sub_id,
            "记录编号": rec_id,
            "材料编号": material_id,
            "划分单元": split_unit,
            "记录内序号": np.arange(len(word_df), dtype=int),
            "词": raw_words.to_numpy(),
            "标准词": norm_words.to_numpy(),
            "开始时间": onsets,
            "结束时间": ends,
            "上下文编号": [f"{material_id}|{suffix}" for suffix in context_suffixes],
            "数据划分": str(split),
            "是否可训练": np.asarray(trainable_flags, dtype=bool),
            "排除原因": exclusion_reasons,
            "会话": f"ses-{_strip_prefix(session, 'ses-')}",
            "任务": task,
            "运行编号": f"run-{_strip_prefix(run, 'run-')}",
            "语料": str(corpus_name or task),
            "原事件行号": word_df["_orig_row"].to_numpy(dtype=int),
            "词性": pos_values,
            "是否内容词": np.asarray(content_values, dtype=bool),
            "信号相对路径": str(signal_relpath),
        }
    )
    return result.loc[:, list(PERSISTED_COLUMNS)]


def _reconcile_material_fingerprints(table: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """确保共享同一材料编号的跨受试者可训练记录具有一致文本指纹；若不一致则按记录隔离材料编号。"""
    result = table.copy()
    mismatches: list[dict[str, Any]] = []
    fingerprints_by_material: dict[str, dict[str, str]] = {}
    for (material_id, rec_id), group in result.groupby(
        ["材料编号", "记录编号"], sort=False, dropna=False
    ):
        if not group["是否可训练"].astype(bool).any():
            continue
        words = group.sort_values("记录内序号", kind="stable")["标准词"]
        fingerprints_by_material.setdefault(str(material_id), {})[str(rec_id)] = (
            text_fingerprint(words)
        )

    for material_id, rec_map in fingerprints_by_material.items():
        if len(set(rec_map.values())) <= 1:
            continue
        mismatches.append(
            {
                "original_material_id": material_id,
                "recording_count": len(rec_map),
                "unique_fingerprints": len(set(rec_map.values())),
            }
        )
        mask = result["材料编号"].eq(material_id)
        result.loc[mask, "材料编号"] = (
            result.loc[mask, "记录编号"].astype(str) + "|material"
        )
    return result, mismatches


def _corpus_for_run_key(
    dataset_name: str,
    run_key: tuple[str, str, str, str],
    pnpl_dataset: Any,
) -> str:
    canonical = _normalize_dataset_name(dataset_name)
    if canonical == "libribrain100":
        for rec in getattr(pnpl_dataset, "_records", None) or ():
            if tuple(map(str, rec.run_key)) == tuple(map(str, run_key)):
                return str(rec.corpus)
        return str(run_key[2]).lower()
    if canonical == "pallier2025":
        return "littleprince_listen"
    return "armeni_sherlock"


def _sidecar_matches(
    output_path: Path,
    sidecar_path: Path,
    *,
    source_h5_path: Path,
    target_sampling_rate_hz: float,
    filter_low_hz: float,
    filter_high_hz: float,
) -> dict[str, Any] | None:
    if not output_path.is_file() or not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        stat = source_h5_path.stat()
        preproc = sidecar.get("preprocessing", {})
        if (
            sidecar.get("source_h5_path") != str(source_h5_path.resolve())
            or sidecar.get("source_h5_size_bytes") != int(stat.st_size)
            or sidecar.get("source_h5_mtime_ns") != int(stat.st_mtime_ns)
            or not np.isclose(
                float(sidecar.get("output_sampling_rate_hz", -1.0)),
                float(target_sampling_rate_hz),
            )
            or not np.isclose(float(preproc.get("filter_low_hz", -1.0)), float(filter_low_hz))
            or not np.isclose(float(preproc.get("filter_high_hz", -1.0)), float(filter_high_hz))
            or preproc.get("scaler") != "RobustScaler"
            or preproc.get("pnpl_standardize") is not False
        ):
            return None
        arr = np.load(output_path, mmap_mode="r", allow_pickle=False)
        if (
            arr.dtype != np.float32
            or list(arr.shape) != sidecar.get("shape")
            or sha256_file(output_path) != sidecar.get("output_sha256")
        ):
            return None
        return sidecar
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def materialize_pnpl_dataset(
    dataset_name: str,
    pnpl_data_path: str | Path,
    output_root: str | Path,
    *,
    split_policy: str = "simpleb2t",
    subjects: Any = None,
    corpus: Any = "all",
    sessions: Sequence[str] | None = None,
    runs: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    include_run_keys: Sequence[tuple[str, str, str, str]] | None = None,
    exclude_run_keys: Sequence[tuple[str, str, str, str]] | None = None,
    preprocessing: str | None = None,
    preprocessing_config: Mapping[str, Any] | None = None,
    split_manifest_path: str | Path | None = None,
    target_sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ,
    filter_low_hz: float = FILTER_LOW_HZ,
    filter_high_hz: float = FILTER_HIGH_HZ,
    eligibility_window_seconds: float = ELIGIBILITY_WINDOW_SECONDS,
    pause_gap_seconds: float = DEFAULT_PAUSE_GAP_SECONDS,
    max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS,
    mne_n_jobs: int = 1,
    download: bool = True,
    preload_files: bool = False,
    create_h5_if_missing: bool = True,
    force: bool = False,
    pnpl_dataset: Any = None,
    pnpl_datasets_module: Any = None,
) -> dict[str, Any]:
    """通过 PNPL 拉取连续 MEG H5 与词事件表，并转存为本仓 50 Hz + RobustScaler 缓存与规范事件表。"""
    canonical = _normalize_dataset_name(dataset_name)
    out_root = Path(output_root)
    meg_dir = out_root / "signals" / "meg_50hz"
    events_path = out_root / "events" / "events.csv"
    meg_dir.mkdir(parents=True, exist_ok=True)

    opened_locally = pnpl_dataset is None
    if pnpl_dataset is None:
        pnpl_dataset = open_pnpl_dataset(
            canonical,
            pnpl_data_path,
            subjects=subjects,
            corpus=corpus,
            sessions=sessions,
            runs=runs,
            tasks=tasks,
            include_run_keys=include_run_keys,
            exclude_run_keys=exclude_run_keys,
            preprocessing=preprocessing,
            preprocessing_config=preprocessing_config,
            download=download,
            preload_files=preload_files,
            preload_h5=False,
            create_h5_if_missing=create_h5_if_missing,
            pnpl_datasets_module=pnpl_datasets_module,
        )

    try:
        # 防御性检查：确保 PNPL 内置 z-score 标准化已关闭。
        if getattr(pnpl_dataset, "standardize_enabled", False):
            raise RuntimeError(
                "PNPL 数据集必须以 standardize=False 实例化，避免在 RobustScaler 前混入 z-score。"
            )

        run_keys = [tuple(map(str, rk)) for rk in getattr(pnpl_dataset, "run_keys", ())]
        if not run_keys:
            raise ValueError(f"PNPL 数据集 {canonical} 未解析到任何可用 run_key。")

        event_frames: list[pd.DataFrame] = []
        signal_records: list[dict[str, Any]] = []
        reused_count = 0
        materialized_count = 0

        for index, run_key in enumerate(run_keys, start=1):
            subject, session, task, run = run_key
            sub_id = _format_subject_id(subject)
            rec_id = recording_identifier(canonical, run_key)
            split, split_unit = resolve_split_and_unit(
                canonical,
                run_key,
                split_policy=split_policy,
                pnpl_dataset=pnpl_dataset,
                split_manifest_path=split_manifest_path,
            )
            h5_path = _resolve_pnpl_h5_path(pnpl_dataset, run_key)
            events_tsv_path = _resolve_pnpl_events_path(pnpl_dataset, run_key)

            stem = recording_signal_stem(canonical, run_key)
            output_npy = meg_dir / sub_id / f"{stem}.npy"
            sidecar_json = output_npy.with_suffix(".json")
            signal_relpath = output_npy.relative_to(meg_dir).as_posix()

            reused_sidecar = (
                None
                if force
                else _sidecar_matches(
                    output_npy,
                    sidecar_json,
                    source_h5_path=h5_path,
                    target_sampling_rate_hz=target_sampling_rate_hz,
                    filter_low_hz=filter_low_hz,
                    filter_high_hz=filter_high_hz,
                )
            )

            if reused_sidecar is not None:
                sidecar = reused_sidecar
                reused_count += 1
            else:
                raw_array, h5_meta = _read_h5_continuous_meg(
                    h5_path,
                    fallback_sfreq=getattr(pnpl_dataset, "sfreq", None),
                )
                if canonical == "pallier2025":
                    first_samp, time_origin_sfreq, time_origin_src = (
                        _inspect_pallier_time_origin(pnpl_dataset, run_key, h5_meta)
                    )
                else:
                    first_samp = int(h5_meta["source_first_samp"])
                    time_origin_sfreq = float(h5_meta["source_sampling_rate_hz"])
                    time_origin_src = "h5_attrs"

                scaled_array, preproc_details = preprocess_continuous_meg(
                    raw_array,
                    source_sampling_rate_hz=h5_meta["source_sampling_rate_hz"],
                    channel_names=h5_meta["channel_names"],
                    channel_types=h5_meta["channel_types"],
                    target_sampling_rate_hz=target_sampling_rate_hz,
                    filter_low_hz=filter_low_hz,
                    filter_high_hz=filter_high_hz,
                    source_highpass_hz=h5_meta["source_highpass_hz"],
                    source_lowpass_hz=h5_meta["source_lowpass_hz"],
                    mne_n_jobs=mne_n_jobs,
                )
                _write_array_atomic(output_npy, scaled_array)
                output_sha = sha256_file(output_npy)
                stat = h5_path.stat()
                sidecar = {
                    "schema_version": 1,
                    "dataset": canonical,
                    "recording_id": rec_id,
                    "subject": sub_id,
                    "session": f"ses-{_strip_prefix(session, 'ses-')}",
                    "task": task,
                    "run": f"run-{_strip_prefix(run, 'run-')}",
                    "split": split,
                    "split_unit": split_unit,
                    "source_h5_path": str(h5_path.resolve()),
                    "source_h5_size_bytes": int(stat.st_size),
                    "source_h5_mtime_ns": int(stat.st_mtime_ns),
                    "source_first_samp": int(first_samp),
                    "source_sampling_rate_hz": float(time_origin_sfreq),
                    "source_h5_sampling_rate_hz": float(h5_meta["source_sampling_rate_hz"]),
                    "source_sample_count": int(h5_meta["source_sample_count"]),
                    "time_origin_source": time_origin_src,
                    "output_first_samp": 0,
                    "output_sampling_rate_hz": float(preproc_details["output_sampling_rate_hz"]),
                    "output_sample_count": int(scaled_array.shape[1]),
                    "channel_count": int(scaled_array.shape[0]),
                    "channel_names": h5_meta["channel_names"],
                    "channel_types": h5_meta["channel_types"],
                    "channel_names_sha256": h5_meta["channel_names_sha256"],
                    "sensor_xyz": h5_meta["sensor_xyz"],
                    "shape": list(scaled_array.shape),
                    "dtype": "float32",
                    "output_sha256": output_sha,
                    "signal_relpath": signal_relpath,
                    "preprocessing": preproc_details,
                }
                _write_json_atomic(sidecar_json, sidecar)
                materialized_count += 1

            corpus_name = _corpus_for_run_key(canonical, run_key, pnpl_dataset)
            frame = extract_run_word_events(
                canonical,
                events_tsv_path,
                run_key=run_key,
                split=split,
                split_unit=split_unit,
                signal_metadata=sidecar,
                signal_relpath=signal_relpath,
                corpus_name=corpus_name,
                pnpl_dataset=pnpl_dataset,
                eligibility_window_seconds=eligibility_window_seconds,
                pause_gap_seconds=pause_gap_seconds,
                max_context_words=max_context_words,
            )
            if not frame.empty:
                event_frames.append(frame)

            signal_records.append(
                {
                    "recording_id": rec_id,
                    "subject": sub_id,
                    "split": split,
                    "split_unit": split_unit,
                    "signal_relpath": signal_relpath,
                    "sidecar_relpath": sidecar_json.relative_to(meg_dir).as_posix(),
                    "shape": sidecar["shape"],
                    "output_sha256": sidecar["output_sha256"],
                    "sidecar_sha256": sha256_file(sidecar_json),
                    "channel_count": sidecar["channel_count"],
                    "channel_names_sha256": sidecar["channel_names_sha256"],
                }
            )
            print(
                f"PNPL[{canonical}] {index}/{len(run_keys)}: {rec_id} "
                f"shape={sidecar['shape']} words={len(frame)}",
                flush=True,
            )

        if not event_frames:
            raise ValueError(f"PNPL 数据集 {canonical} 未提取到任何词级事件。")

        full_table = pd.concat(event_frames, ignore_index=True)
        full_table, material_mismatches = _reconcile_material_fingerprints(full_table)
        validate_event_table(full_table)

        written_events, event_manifest_file, event_manifest = write_event_product(
            full_table,
            events_path,
            manifest_kwargs={
                "dataset": canonical,
                "project_root": PROJECT_ROOT,
                "source_raw_roots": [Path(pnpl_data_path)],
                "artifact_dependencies": [],
                "context_definition": {
                    "grouping": "explicit_sentence_id_or_pause_gap",
                    "pause_gap_seconds": float(pause_gap_seconds),
                    "maximum_words": int(max_context_words),
                },
                "window_contract": {
                    "target_sampling_rate_hz": float(target_sampling_rate_hz),
                    "filter_low_hz": float(filter_low_hz),
                    "filter_high_hz": float(filter_high_hz),
                    "window_start_offset_seconds": WINDOW_START_OFFSET_SECONDS,
                    "window_seconds": WINDOW_SECONDS,
                    "eligibility_window_seconds": float(eligibility_window_seconds),
                    "baseline_seconds": BASELINE_SECONDS,
                    "clamp": CLAMP,
                    "scaler": "RobustScaler",
                    "pnpl_standardize": False,
                },
                "source_contract": {
                    "adapter": "braindecoding.data.pnpl_dataset_adapter",
                    "dataset": canonical,
                    "split_policy": split_policy,
                    "run_keys": [list(rk) for rk in run_keys],
                },
                "audit": {
                    "recording_count": len(run_keys),
                    "materialized_recordings": materialized_count,
                    "reused_recordings": reused_count,
                    "material_mismatches": material_mismatches,
                    "pnpl_standardize_disabled": True,
                },
                "builder_sources": [
                    Path(__file__),
                    PROJECT_ROOT / "braindecoding/data/derived_cache.py",
                    PROJECT_ROOT / "braindecoding/data/event_table.py",
                ],
            },
        )
        validate_event_product(written_events, event_manifest_file)

        signals_manifest_payload = {
            "schema_version": 1,
            "dataset": canonical,
            "status": "complete",
            "split_policy": split_policy,
            "recording_count": len(signal_records),
            "target_sampling_rate_hz": float(target_sampling_rate_hz),
            "filter_low_hz": float(filter_low_hz),
            "filter_high_hz": float(filter_high_hz),
            "scaler": "RobustScaler",
            "pnpl_standardize": False,
            "dtype": "float32",
            "event_table_sha256": event_manifest["event_table_sha256"],
            "recordings": sorted(signal_records, key=lambda item: item["recording_id"]),
        }
        signals_manifest_payload["manifest_sha256"] = stable_sha256(signals_manifest_payload)
        signals_manifest_path = out_root / "signals" / "manifest.json"
        _write_json_atomic(signals_manifest_path, signals_manifest_payload)

        return {
            "dataset": canonical,
            "split_policy": split_policy,
            "recording_count": len(signal_records),
            "materialized_recordings": materialized_count,
            "reused_recordings": reused_count,
            "event_count": int(len(full_table)),
            "trainable_event_count": int(full_table["是否可训练"].astype(bool).sum()),
            "split_counts": event_manifest["split_counts"],
            "events_path": str(written_events),
            "event_manifest_path": str(event_manifest_file),
            "signals_dir": str(meg_dir),
            "signals_manifest_path": str(signals_manifest_path),
            "event_table_sha256": event_manifest["event_table_sha256"],
            "signals_manifest_sha256": signals_manifest_payload["manifest_sha256"],
        }
    finally:
        if opened_locally and hasattr(pnpl_dataset, "close_h5_files"):
            try:
                pnpl_dataset.close_h5_files()
            except Exception:
                pass


def load_pnpl_event_table(
    path: str | Path,
    *,
    split: str | None = None,
    subjects: Sequence[str] | None = None,
    trainable_only: bool = False,
) -> pd.DataFrame:
    """读取并校验由 PNPL 适配器落盘的规范中文事件表。"""
    table = pd.read_csv(path, keep_default_na=False)
    for col in ("是否可训练", "是否内容词"):
        if col in table.columns:
            table[col] = table[col].map(_parse_bool)
    validate_event_table(table)
    if split is not None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"无效数据划分：{split!r}")
        table = table[table["数据划分"].eq(split)]
    if subjects is not None:
        wanted = {_format_subject_id(s) for s in subjects}
        table = table[table["受试者"].astype(str).isin(wanted)]
    if trainable_only:
        table = table[table["是否可训练"].astype(bool)]
    return table.reset_index(drop=True)


def _resolve_meg_channel_positions(
    channel_names: Sequence[str],
    *,
    layout_path: str | Path | None = None,
    sensor_xyz: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """按 Vectorview -> CTF275 -> H5 3D 坐标投影 -> 规则网格的顺序解析 2D 传感器位置。"""
    names = [str(name) for name in channel_names]
    try:
        return vectorview_channel_positions(names, layout_path=layout_path)
    except (ImportError, FileNotFoundError, KeyError):
        pass

    # 尝试 MNE CTF275 布局（适配 Armeni2022 等 CTF MEG 系统）
    spec = importlib.util.find_spec("mne")
    if spec is not None and spec.origin is not None:
        ctf_layout = (
            Path(spec.origin).parent / "channels" / "data" / "layouts" / "CTF275.lay"
        )
        if ctf_layout.is_file():
            positions: dict[str, tuple[float, float]] = {}
            for line in ctf_layout.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) >= 6:
                    label = fields[5].split("-")[0].strip()
                    try:
                        positions[label] = (float(fields[1]), float(fields[2]))
                    except ValueError:
                        continue
            mapped = [
                positions.get(name.split("-")[0].replace(" ", ""))
                for name in names
            ]
            if all(pos is not None for pos in mapped):
                coords = np.asarray(mapped, dtype=np.float32)
                lower = coords.min(axis=0, keepdims=True)
                span = coords.max(axis=0, keepdims=True) - lower
                return ((coords - lower) / np.maximum(span, 1e-8)).astype(np.float32)

    if sensor_xyz is not None and len(sensor_xyz) == len(names):
        xyz = np.asarray(sensor_xyz, dtype=np.float32)
        if xyz.ndim == 2 and xyz.shape[1] >= 2 and np.isfinite(xyz[:, :2]).all():
            xy = xyz[:, :2]
            lower = xy.min(axis=0, keepdims=True)
            span = xy.max(axis=0, keepdims=True) - lower
            return ((xy - lower) / np.maximum(span, 1e-8)).astype(np.float32)

    # 兜底：无布局信息时提供确定的 [0, 1]^2 单位圆周排布，确保测试与新设备不崩溃
    angles = np.linspace(0.0, 2.0 * np.pi, num=max(1, len(names)), endpoint=False)
    coords = np.stack([0.5 + 0.5 * np.cos(angles), 0.5 + 0.5 * np.sin(angles)], axis=1)
    return coords[: len(names)].astype(np.float32)


@lru_cache(maxsize=256)
def _open_derived_signal(path: str):
    return np.load(path, mmap_mode="r")


class DerivedMegWordDataset(Dataset):
    """通用 50 Hz 派生 MEG 词级数据集，接口与 ``Pallier2025WordDataset`` 完全对齐。

    可直接接入现有 Stage 1 / Stage 2 错位去偏与局部前瞻训练器，以及
    ``SyntheticTimingDataset`` 联合/独立合成脉冲对照。
    """

    def __init__(
        self,
        table: pd.DataFrame,
        dataset_config: Mapping[str, Any],
        meg_cache_dir: str | Path,
        text_embedding_path: str | Path,
        text_embedding_config: Mapping[str, Any],
        *,
        zero_meg: bool = False,
        temporal_shift_seconds: float = 0.0,
        shift_mode: str | None = None,
        preroll_seconds: float | None = None,
        preroll_source_table: pd.DataFrame | None = None,
        preroll_buffers: dict[str, np.ndarray] | None = None,
    ) -> None:
        source = table.copy()
        missing = sorted(set(CORE_EVENT_COLUMNS) - set(source.columns))
        if missing:
            raise ValueError(f"DerivedMegWordDataset 缺少核心事件字段：{missing}")
        if "信号相对路径" not in source.columns:
            raise ValueError("DerivedMegWordDataset 事件表缺少 '信号相对路径' 字段。")

        source = source[source["是否可训练"].astype(bool)].reset_index(drop=True)
        if source.empty:
            raise ValueError("DerivedMegWordDataset 没有可训练事件。")

        cfg = dict(dataset_config)
        cfg.setdefault("target_sampling_rate_hz", TARGET_SAMPLING_RATE_HZ)
        cfg.setdefault("window_seconds", WINDOW_SECONDS)
        cfg.setdefault("window_start_offset_seconds", WINDOW_START_OFFSET_SECONDS)
        cfg.setdefault("baseline_seconds", BASELINE_SECONDS)
        cfg.setdefault("clamp", CLAMP)
        cfg.setdefault("max_context_words", DEFAULT_MAX_CONTEXT_WORDS)
        self.dataset_config = cfg

        self.table = add_runtime_context_uid(
            source,
            max_context_words=int(cfg["max_context_words"]),
        )
        aliases = {
            "event_id": "事件编号",
            "subject_id": "受试者",
            "recording_id": "记录编号",
            "word": "词",
            "normalized_word": "标准词",
            "sentence_uid": "上下文编号",
            "split": "数据划分",
            "is_trainable": "是否可训练",
            "onset_seconds": "开始时间",
        }
        for target, origin in aliases.items():
            self.table[target] = self.table[origin]

        configured_subjects = tuple(
            _format_subject_id(s)
            for s in (
                cfg.get("subjects")
                or list(dict.fromkeys(self.table["subject_id"].astype(str)))
            )
        )
        observed_subjects = set(self.table["subject_id"].astype(str))
        if not observed_subjects.issubset(set(configured_subjects)):
            raise ValueError("事件表包含未在配置 subjects 中声明的受试者。")

        self.meg_cache_dir = Path(meg_cache_dir)
        self.zero_meg = bool(zero_meg)
        self.sampling_rate_hz = float(cfg["target_sampling_rate_hz"])
        self.window_seconds = float(cfg["window_seconds"])
        self.window_samples = int(round(self.window_seconds * self.sampling_rate_hz))
        self._baseline_samples = int(
            round(float(cfg["baseline_seconds"]) * self.sampling_rate_hz)
        )
        self._clamp = float(cfg["clamp"])
        self._window_offset_seconds = float(cfg["window_start_offset_seconds"])

        self.temporal_shift_seconds = float(temporal_shift_seconds)
        self.temporal_shift_samples = int(
            round(self.temporal_shift_seconds * self.sampling_rate_hz)
        )
        resolved_shift_mode = str(
            shift_mode if shift_mode is not None else cfg.get("shift_mode", "deterministic_circular")
        ).strip().lower()
        if resolved_shift_mode not in VALID_TEMPORAL_SHIFT_MODES:
            raise ValueError(
                f"无效 shift_mode：{resolved_shift_mode!r}，可选 {VALID_TEMPORAL_SHIFT_MODES}"
            )
        self.shift_mode = resolved_shift_mode
        self.preroll_seconds = float(
            preroll_seconds
            if preroll_seconds is not None
            else cfg.get("preroll_seconds", DEFAULT_PREROLL_SECONDS)
        )
        if self.shift_mode == "preroll" and self.temporal_shift_samples > 0:
            raise ValueError("Pre-Roll 因果移位模式仅允许严格过去移位 (<= 0)。")

        self.subject_map = {subj: idx for idx, subj in enumerate(configured_subjects)}
        self.subject_count = len(self.subject_map)
        self.subject_indices = np.asarray(
            [self.subject_map[s] for s in self.table["subject_id"].astype(str)],
            dtype=np.int64,
        )
        contexts = self.table[RUNTIME_CONTEXT_COLUMN].astype(str).tolist()
        context_map = {val: idx for idx, val in enumerate(dict.fromkeys(contexts))}
        self.sentence_indices = np.asarray(
            [context_map[val] for val in contexts], dtype=np.int64
        )

        signature = text_data.text_embedding_signature(text_embedding_config)
        self.embedding_map = text_data.load_text_embedding_cache(
            text_embedding_path, expected_signature=signature
        )
        expected_dim = int(text_embedding_config["embedding_dimension"])
        missing_words = sorted(
            set(self.table["normalized_word"].astype(str)) - set(self.embedding_map)
        )
        if missing_words:
            raise KeyError(f"文本向量缓存缺少词型：{missing_words[:5]}")
        if any(vec.shape != (expected_dim,) for vec in self.embedding_map.values()):
            raise ValueError("文本向量维度与配置不一致。")

        self.signal_paths: dict[str, Path] = {}
        self.signal_metadata: dict[str, dict[str, Any]] = {}
        channel_names: tuple[str, ...] | None = None
        sensor_xyz_ref = None
        for rec_id, relpath in (
            self.table[["recording_id", "信号相对路径"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            rec_key = str(rec_id)
            sig_path = self.meg_cache_dir / str(relpath)
            sidecar_path = sig_path.with_suffix(".json")
            if not sig_path.is_file() or not sidecar_path.is_file():
                raise FileNotFoundError(f"派生 MEG 信号或其 sidecar 缺失：{sig_path}")
            meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not np.isclose(
                float(meta.get("output_sampling_rate_hz", -1.0)),
                self.sampling_rate_hz,
            ):
                raise ValueError(f"信号采样率与配置不一致：{sidecar_path}")
            names_tuple = tuple(meta["channel_names"])
            if channel_names is None:
                channel_names = names_tuple
                sensor_xyz_ref = meta.get("sensor_xyz")
            elif names_tuple != channel_names:
                raise ValueError(f"recording 间通道顺序不一致：{sidecar_path}")
            self.signal_paths[rec_key] = sig_path
            self.signal_metadata[rec_key] = meta

        self.channel_names = list(channel_names or ())
        self.channel_count = len(self.channel_names)
        self.channel_positions = _resolve_meg_channel_positions(
            self.channel_names,
            layout_path=cfg.get("layout_path"),
            sensor_xyz=sensor_xyz_ref,
        )

        self._recording_ids = tuple(self.table["recording_id"].astype(str).tolist())
        self._words = tuple(self.table["normalized_word"].astype(str).tolist())
        self._event_ids = tuple(self.table["event_id"].astype(str).tolist())
        self._onset_seconds = self.table["onset_seconds"].to_numpy(dtype=np.float64, copy=True)
        self._signal_path_strs = tuple(
            str(self.signal_paths[rec_id]) for rec_id in self._recording_ids
        )
        self._window_starts = np.asarray(
            [
                recording_onset_to_sample(
                    float(onset) + self._window_offset_seconds,
                    source_first_samp=self.signal_metadata[rec_id]["source_first_samp"],
                    source_sampling_rate_hz=self.signal_metadata[rec_id][
                        "source_sampling_rate_hz"
                    ],
                    output_sampling_rate_hz=self.signal_metadata[rec_id][
                        "output_sampling_rate_hz"
                    ],
                )
                for rec_id, onset in zip(self._recording_ids, self._onset_seconds)
            ],
            dtype=np.int64,
        )
        self._embedding_tensors = {
            word: torch.from_numpy(np.array(vec, dtype=np.float32, copy=True))
            for word, vec in self.embedding_map.items()
        }
        self._subject_index_tensors = torch.as_tensor(self.subject_indices, dtype=torch.long)
        self._sentence_index_tensors = torch.as_tensor(self.sentence_indices, dtype=torch.long)
        self._onset_second_tensors = torch.as_tensor(self._onset_seconds, dtype=torch.float32)

        self._preroll_by_recording: dict[str, np.ndarray] = {}
        self._preroll_source_table = preroll_source_table
        self._explicit_preroll_buffers = preroll_buffers
        if (
            not self.zero_meg
            and self.shift_mode == "preroll"
            and self.temporal_shift_samples < 0
        ):
            self.ensure_preroll_buffers()

    def ensure_preroll_buffers(
        self,
        shift_seconds: float | None = None,
        *,
        shift_mode: str | None = None,
        preroll_seconds: float | None = None,
    ) -> None:
        if shift_mode is not None:
            mode_norm = str(shift_mode).strip().lower()
            if mode_norm not in VALID_TEMPORAL_SHIFT_MODES:
                raise ValueError(f"无效 shift_mode：{mode_norm!r}")
            self.shift_mode = mode_norm
        if preroll_seconds is not None:
            self.preroll_seconds = float(preroll_seconds)
        if shift_seconds is not None:
            self.temporal_shift_seconds = float(shift_seconds)
            self.temporal_shift_samples = int(
                round(self.temporal_shift_seconds * self.sampling_rate_hz)
            )
        if self.zero_meg or self.shift_mode != "preroll" or self.temporal_shift_samples == 0:
            return
        if self.temporal_shift_samples > 0:
            raise ValueError("Pre-Roll 因果移位模式仅允许严格过去移位 (<= 0)。")

        required = max(
            int(round(self.preroll_seconds * self.sampling_rate_hz)),
            abs(int(self.temporal_shift_samples)) + int(self.window_samples),
        )
        observed_pairs = list(
            self.table[["subject_id", "recording_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if self._explicit_preroll_buffers is not None:
            for subj_raw, rec_raw in observed_pairs:
                subj, rec_id = str(subj_raw), str(rec_raw)
                buf = self._explicit_preroll_buffers.get(
                    rec_id, self._explicit_preroll_buffers.get(subj)
                )
                if buf is None:
                    raise ValueError(f"preroll_buffers 缺少受试者 {subj} ({rec_id})。")
                arr = np.asarray(buf, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] != self.channel_count or arr.shape[1] < required:
                    raise ValueError(f"preroll_buffers[{subj}] 形状或长度不足：{arr.shape}")
                sliced = np.array(arr[:, -required:], dtype=np.float32, copy=True)
                sliced.setflags(write=False)
                self._preroll_by_recording[rec_id] = sliced
            return

        src_table = self._preroll_source_table
        if src_table is None:
            csv_path = self.meg_cache_dir.parent.parent / "events" / "events.csv"
            if csv_path.is_file():
                src_table = load_pnpl_event_table(csv_path, split="train", trainable_only=True)
        if src_table is None or src_table.empty:
            raise ValueError("Pre-Roll 模式需要提供 train 划分事件表或显式 preroll_buffers。")

        split_col = "数据划分" if "数据划分" in src_table.columns else "split"
        subj_col = "受试者" if "受试者" in src_table.columns else "subject_id"
        rec_col = "记录编号" if "记录编号" in src_table.columns else "recording_id"
        train_df = src_table[src_table[split_col].astype(str).eq("train")]
        by_subject: dict[str, list[tuple[str, Path]]] = {}
        for subj_v, rec_v, rel_v in (
            train_df[[subj_col, rec_col, "信号相对路径"]]
            .drop_duplicates()
            .sort_values([subj_col, rec_col], kind="mergesort")
            .itertuples(index=False, name=None)
        ):
            sig_p = self.meg_cache_dir / str(rel_v)
            if sig_p.is_file():
                by_subject.setdefault(str(subj_v), []).append((str(rec_v), sig_p))

        for subj_raw, rec_raw in observed_pairs:
            subj, rec_id = str(subj_raw), str(rec_raw)
            candidates = [
                (r, p) for r, p in by_subject.get(subj, ()) if r != rec_id
            ]
            if not candidates:
                raise ValueError(f"受试者 {subj} 缺少独立于 {rec_id} 的 train Pre-Roll 记录。")
            chunks_rev: list[np.ndarray] = []
            remaining = required
            for _, sig_p in reversed(candidates):
                arr = _open_derived_signal(str(sig_p))
                take = min(remaining, int(arr.shape[1]))
                if take > 0:
                    chunks_rev.append(
                        np.array(arr[:, int(arr.shape[1]) - take :], dtype=np.float32, copy=True)
                    )
                    remaining -= take
                if remaining <= 0:
                    break
            if remaining > 0:
                raise ValueError(f"受试者 {subj} 的 train 记录总长不足以构建 {required} 点 Pre-Roll。")
            buf = (
                chunks_rev[0]
                if len(chunks_rev) == 1
                else np.concatenate(list(reversed(chunks_rev)), axis=1).astype(np.float32)
            )
            buf.setflags(write=False)
            self._preroll_by_recording[rec_id] = buf

    def __len__(self) -> int:
        return len(self.table)

    def read_meg(self, index: int) -> np.ndarray:
        idx = int(index)
        if self.zero_meg:
            return np.zeros((self.channel_count, self.window_samples), dtype=np.float32)
        recording = _open_derived_signal(self._signal_path_strs[idx])
        if recording.ndim != 2 or recording.shape[1] < self.window_samples:
            raise ValueError("词窗口超出连续 MEG 信号长度。")
        start = int(self._window_starts[idx])
        stop = start + self.window_samples
        if start < 0 or stop > recording.shape[1]:
            start = max(0, min(start, int(recording.shape[1]) - self.window_samples))
            stop = start + self.window_samples

        if self.temporal_shift_samples == 0:
            window = recording[:, start:stop].astype(np.float32, copy=True)
        elif self.shift_mode == "preroll":
            raw_start = start + int(self.temporal_shift_samples)
            win_len = int(self.window_samples)
            if raw_start >= 0:
                window = recording[:, raw_start : raw_start + win_len].astype(np.float32, copy=True)
            else:
                preroll = self._preroll_by_recording[self._recording_ids[idx]]
                buf_len = int(preroll.shape[1])
                buf_start = buf_len + raw_start
                if buf_start < 0:
                    raise IndexError(f"Pre-Roll 缓冲区长度不足：raw_start={raw_start}, buf_len={buf_len}")
                if raw_start + win_len <= 0:
                    window = preroll[:, buf_start : buf_start + win_len].astype(np.float32, copy=True)
                else:
                    left_len = -raw_start
                    right_len = win_len - left_len
                    window = np.empty((self.channel_count, win_len), dtype=np.float32)
                    window[:, :left_len] = preroll[:, buf_start:buf_len]
                    window[:, left_len:] = recording[:, :right_len]
        else:
            valid_span = int(recording.shape[1]) - self.window_samples + 1
            if valid_span <= 2 * self.window_samples:
                raise ValueError("连续信号长度不足以执行非重叠时间错位。")
            shifted_start = (start + self.temporal_shift_samples) % valid_span
            if abs(shifted_start - start) < self.window_samples:
                step = self.window_samples if self.temporal_shift_samples > 0 else -self.window_samples
                shifted_start = (shifted_start + step) % valid_span
            window = recording[:, shifted_start : shifted_start + self.window_samples].astype(
                np.float32, copy=True
            )

        effective_baseline = min(self._baseline_samples, int(window.shape[1]))
        if effective_baseline > 0:
            window -= window[:, :effective_baseline].mean(axis=1, keepdims=True)
        np.clip(window, -self._clamp, self._clamp, out=window)
        return window

    def __getitem__(self, index: int) -> dict[str, Any]:
        idx = int(index)
        word = self._words[idx]
        return {
            "meg": torch.from_numpy(self.read_meg(idx)),
            "text_embedding": self._embedding_tensors[word],
            "subject_index": self._subject_index_tensors[idx],
            "sentence_index": self._sentence_index_tensors[idx],
            "onset_seconds": self._onset_second_tensors[idx],
            "word": word,
            "event_id": self._event_ids[idx],
            "recording_id": self._recording_ids[idx],
        }


def _default_output_root(dataset_name: str) -> Path:
    canonical = _normalize_dataset_name(dataset_name)
    key = "pallier2025_full" if canonical == "pallier2025" else canonical
    paths = canonical_cache_paths(key)
    return project_path(paths["event_table"]).parent.parent


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--pnpl-root", required=True, help="PNPL 本地数据与缓存目录")
    parser.add_argument(
        "--output-root",
        default=None,
        help="派生缓存输出目录（默认写入 data/cache/derived/<dataset>/）",
    )
    parser.add_argument(
        "--split-policy",
        default="simpleb2t",
        help="划分策略：libribrain100(simpleb2t/pnpl_official), pallier2025(simpleb2t/manifest), armeni2022(simpleb2t)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="受试者选择器（如 deep / all / 0 / 01 02 / 001）",
    )
    parser.add_argument("--corpus", default="all", help="LibriBrain100 语料筛选")
    parser.add_argument("--sessions", nargs="+", default=None)
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--no-download", action="store_true", help="禁用远程自动拉取")
    parser.add_argument("--preload-files", action="store_true")
    parser.add_argument("--force", action="store_true", help="强制重建已有 50 Hz 缓存")
    parser.add_argument("--mne-n-jobs", type=int, default=1)
    args = parser.parse_args(argv)

    subjects_arg: Any = None
    if args.subjects:
        subjects_arg = args.subjects[0] if len(args.subjects) == 1 and args.subjects[0] in {"all", "deep", "broad"} else args.subjects

    output_root = Path(args.output_root) if args.output_root else _default_output_root(args.dataset)
    summary = materialize_pnpl_dataset(
        args.dataset,
        args.pnpl_root,
        output_root,
        split_policy=args.split_policy,
        subjects=subjects_arg,
        corpus=args.corpus,
        sessions=args.sessions,
        runs=args.runs,
        tasks=args.tasks,
        download=not args.no_download,
        preload_files=args.preload_files,
        force=args.force,
        mne_n_jobs=args.mne_n_jobs,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
