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
import concurrent.futures
import contextlib
from functools import lru_cache
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote
import warnings

import h5py
import numpy as np
import pandas as pd

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
from braindecoding.data.word_window_dataset import (
    BASELINE_SECONDS,
    CLAMP,
    DEFAULT_MAX_CONTEXT_WORDS,
    DEFAULT_PREROLL_SECONDS,
    DerivedMegWordDataset,
    ELIGIBILITY_WINDOW_SECONDS,
    RUNTIME_CONTEXT_COLUMN,
    TARGET_SAMPLING_RATE_HZ,
    VALID_TEMPORAL_SHIFT_MODES,
    WINDOW_SECONDS,
    WINDOW_START_OFFSET_SECONDS,
    WordWindowDataset,
    add_runtime_context_uid,
    channel_names_sha256,
    format_subject_id as _format_subject_id,
    parse_bool_value as _parse_bool,
    recording_onset_to_sample,
)


SUPPORTED_DATASETS = ("libribrain100", "pallier2025_full", "armeni2022")

FILTER_LOW_HZ = 0.1
FILTER_HIGH_HZ = 40.0
DEFAULT_PAUSE_GAP_SECONDS = 0.5

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
        "pallier": "pallier2025_full",
        "pallier2025": "pallier2025_full",
        "pallier2025_full": "pallier2025_full",
        "littleprince": "pallier2025_full",
        "armeni": "armeni2022",
        "armeni2022": "armeni2022",
    }
    if name not in aliases:
        raise ValueError(
            f"不支持的 PNPL 数据集：{dataset_name!r}，支持范围：{SUPPORTED_DATASETS}"
        )
    return aliases[name]


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
        resolved_include_keys = (
            [tuple(map(str, rk)) for rk in include_run_keys]
            if include_run_keys
            else None
        )
        if subjects is None:
            lb_subjects: Any = "all" if resolved_include_keys else "deep"
        elif isinstance(subjects, str):
            low_s = subjects.strip().lower()
            lb_subjects = (
                low_s
                if low_s in {"all", "deep", "broad"}
                else [_strip_prefix(subjects, "sub-")]
            )
        else:
            lb_subjects = [_strip_prefix(str(s), "sub-") for s in subjects]

        if resolved_include_keys is None and (sessions or runs or tasks):
            manifest_mod = importlib.import_module("pnpl.datasets.libribrain100.manifest")
            ses_set = (
                {_strip_prefix(str(s), "ses-") for s in sessions} if sessions else None
            )
            run_set = {_strip_prefix(str(r), "run-") for r in runs} if runs else None
            task_set = {str(t).strip() for t in tasks} if tasks else None
            subj_set = (
                {str(s) for s in lb_subjects}
                if isinstance(lb_subjects, list)
                else ({"0"} if lb_subjects == "deep" else None)
            )
            corpus_set = (
                None
                if corpus is None or (isinstance(corpus, str) and corpus.strip().lower() == "all")
                else (
                    {corpus.strip().lower()}
                    if isinstance(corpus, str)
                    else {str(c).strip().lower() for c in corpus}
                )
            )
            filtered: list[tuple[str, str, str, str]] = []
            for rec in getattr(manifest_mod, "RUN_RECORDS", ()):
                rk = tuple(map(str, rec.run_key))
                if lb_subjects == "broad" and rk[0] == "0":
                    continue
                if subj_set is not None and rk[0] not in subj_set:
                    continue
                if corpus_set is not None:
                    rec_corpus = str(getattr(rec, "corpus", "")).strip().lower()
                    if rec_corpus and rec_corpus not in corpus_set:
                        continue
                if ses_set is not None and rk[1] not in ses_set:
                    continue
                if task_set is not None and rk[2] not in task_set:
                    continue
                if run_set is not None and rk[3] not in run_set:
                    continue
                filtered.append((rk[0], rk[1], rk[2], rk[3]))
            if not filtered:
                raise ValueError(
                    f"LibriBrain100 过滤条件未匹配任何 run_key："
                    f"subjects={lb_subjects}, sessions={sessions}, tasks={tasks}, runs={runs}"
                )
            resolved_include_keys = filtered

        kwargs: dict[str, Any] = {
            "data_path": data_dir,
            "task": task,
            "partition": None,
            "subjects": lb_subjects,
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
        if resolved_include_keys:
            kwargs["include_run_keys"] = resolved_include_keys
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

    if canonical_name == "pallier2025_full":
        cls = getattr(datasets_mod, "Pallier2025")

        def _pad2(val: str, prefix: str) -> str:
            tok = _strip_prefix(str(val), prefix)
            return tok.zfill(2) if tok.isdigit() else tok

        include_subjects = (
            [_pad2(s, "sub-") for s in subjects]
            if isinstance(subjects, (list, tuple, set))
            else ([_pad2(subjects, "sub-")] if isinstance(subjects, str) and subjects != "all" else None)
        )
        include_runs = (
            [_pad2(r, "run-") for r in runs] if runs else None
        )
        include_sessions = (
            [_pad2(s, "ses-") for s in sessions] if sessions else None
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
            include_run_keys=[
                (_pad2(rk[0], "sub-"), _pad2(rk[1], "ses-"), str(rk[2]), _pad2(rk[3], "run-"))
                for rk in include_run_keys
            ]
            if include_run_keys
            else None,
            exclude_run_keys=[
                (_pad2(rk[0], "sub-"), _pad2(rk[1], "ses-"), str(rk[2]), _pad2(rk[3], "run-"))
                for rk in exclude_run_keys
            ]
            if exclude_run_keys
            else None,
            standardize=False,
            clipping_boundary=None,
            create_h5_if_missing=bool(create_h5_if_missing),
            download=bool(download),
            preload_h5=bool(preload_h5),
        )

    cls = getattr(datasets_mod, "Armeni2022")

    def _pad3(val: str, prefix: str) -> str:
        tok = _strip_prefix(str(val), prefix)
        return tok.zfill(3) if tok.isdigit() else tok

    include_subjects = (
        [_pad3(s, "sub-") for s in subjects]
        if isinstance(subjects, (list, tuple, set))
        else ([_pad3(subjects, "sub-")] if isinstance(subjects, str) and subjects != "all" else None)
    )
    include_sessions = (
        [_pad3(s, "ses-") for s in sessions] if sessions else None
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


def _read_raw_fif_continuous_meg(
    fif_path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """直接读取 Pallier2025 原始 BIDS FIF 的全部 MEG 通道（跳过 pnpl 单线程分块 bp+ds 与中间 preproc.fif/H5 落盘）。"""
    import mne

    fif_p = Path(fif_path)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Internal Active Shielding.*",
            category=RuntimeWarning,
        )
        raw = mne.io.read_raw_fif(
            fif_p, preload=False, allow_maxshield=True, verbose="ERROR"
        )
        try:
            raw.pick(picks="meg", exclude=[])
            raw.load_data(verbose="ERROR")
            data = np.asarray(raw.get_data(), dtype=np.float64)
            source_sfreq = float(raw.info["sfreq"])
            first_samp = int(raw.first_samp)
            highpass = raw.info.get("highpass")
            lowpass = raw.info.get("lowpass")
            channel_names = list(raw.ch_names)
            try:
                raw_ch_types = list(raw.get_channel_types())
            except Exception:
                raw_ch_types = None
        finally:
            raw.close()

    if data.ndim != 2 or data.shape[1] == 0:
        raise ValueError(f"原始 FIF MEG 数据维度无效：{data.shape} ({fif_p})")
    channel_types = _infer_channel_types(channel_names, raw_ch_types)
    meta = {
        "source_h5_path": str(fif_p.resolve()),
        "source_sampling_rate_hz": source_sfreq,
        "source_sample_count": int(data.shape[1]),
        "source_first_samp": first_samp,
        "source_highpass_hz": float(highpass) if highpass is not None else None,
        "source_lowpass_hz": float(lowpass) if lowpass is not None else None,
        "channel_count": int(data.shape[0]),
        "channel_names": channel_names,
        "channel_types": channel_types,
        "channel_names_sha256": channel_names_sha256(channel_names),
        "sensor_xyz": None,
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


def _resolve_mne_n_jobs(mne_n_jobs: int | None) -> int:
    """解析 MNE 滤波与重采样线程数（0/负数自动取 min(4, CPU核数)，正数按主机核数封顶）。"""
    cpu_count = max(1, int(os.cpu_count() or 1))
    if mne_n_jobs is None:
        return 1
    requested = int(mne_n_jobs)
    if requested <= 0:
        return max(1, min(4, cpu_count))
    return max(1, min(requested, cpu_count))


@contextlib.contextmanager
def _suppress_nested_download_bars(*, disable_xet: bool = False):
    """外层进度条接管期间静默 Hugging Face / OpenNeuro 子进度条；cleanup_raw 时同步禁用 Xet 块缓存防爆盘。"""
    hf_utils = None
    prev_hf_disabled = None
    try:
        from huggingface_hub import utils as _hf_utils

        hf_utils = _hf_utils
        prev_hf_disabled = bool(hf_utils.are_progress_bars_disabled())
        if not prev_hf_disabled:
            hf_utils.disable_progress_bars()
    except Exception:
        hf_utils = None

    hf_constants = None
    prev_xet_const = None
    had_xet_env = "HF_HUB_DISABLE_XET" in os.environ
    prev_xet_env = os.environ.get("HF_HUB_DISABLE_XET")
    if disable_xet:
        if not had_xet_env:
            os.environ["HF_HUB_DISABLE_XET"] = "1"
        try:
            from huggingface_hub import constants as _hf_constants

            if hasattr(_hf_constants, "HF_HUB_DISABLE_XET"):
                hf_constants = _hf_constants
                prev_xet_const = getattr(hf_constants, "HF_HUB_DISABLE_XET")
                setattr(hf_constants, "HF_HUB_DISABLE_XET", True)
        except Exception:
            hf_constants = None

    had_openneuro_env = "PNPL_OPENNEURO_PROGRESS" in os.environ
    prev_openneuro_env = os.environ.get("PNPL_OPENNEURO_PROGRESS")
    if not had_openneuro_env:
        os.environ["PNPL_OPENNEURO_PROGRESS"] = "0"

    try:
        yield
    finally:
        if hf_utils is not None and prev_hf_disabled is False:
            try:
                hf_utils.enable_progress_bars()
            except Exception:
                pass
        if hf_constants is not None and prev_xet_const is not None:
            try:
                setattr(hf_constants, "HF_HUB_DISABLE_XET", prev_xet_const)
            except Exception:
                pass
        if disable_xet:
            if not had_xet_env:
                os.environ.pop("HF_HUB_DISABLE_XET", None)
            elif prev_xet_env is not None:
                os.environ["HF_HUB_DISABLE_XET"] = prev_xet_env
        if not had_openneuro_env:
            os.environ.pop("PNPL_OPENNEURO_PROGRESS", None)
        elif prev_openneuro_env is not None:
            os.environ["PNPL_OPENNEURO_PROGRESS"] = prev_openneuro_env


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

    resolved_n_jobs = _resolve_mne_n_jobs(mne_n_jobs)
    if should_filter or should_resample:
        info = mne.create_info(ch_names=ch_names, sfreq=src_sfreq, ch_types=ch_types)
        work_arr = arr.copy() if np.shares_memory(arr, data) else arr
        raw = mne.io.RawArray(work_arr, info, verbose=False)
        # 共享内存 threading 后端：scipy.fft / NumPy 释放 GIL，既提速 ~2.5x 又避免 loky 进程复制爆内存
        parallel_ctx: Any = contextlib.nullcontext()
        if resolved_n_jobs > 1:
            try:
                import joblib

                parallel_ctx = joblib.parallel_backend("threading")
            except ImportError:
                parallel_ctx = contextlib.nullcontext()
        with parallel_ctx:
            if should_filter:
                raw.filter(
                    l_freq=float(filter_low_hz),
                    h_freq=float(filter_high_hz),
                    n_jobs=resolved_n_jobs,
                    verbose=False,
                )
            if should_resample:
                raw = raw.resample(
                    tgt_sfreq,
                    n_jobs=resolved_n_jobs,
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
        "mne_n_jobs": int(resolved_n_jobs),
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
    if canonical == "pallier2025_full":
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
    if canonical == "pallier2025_full":
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
            try:
                rec = manifest_mod.get_record((str(run_key[0]), ses, task, _strip_prefix(run, "run-")))
            except KeyError as exc:
                raise ValueError(f"LibriBrain100 官方 manifest 未找到记录：{run_key}") from exc
            part = str(rec.partition).strip().lower()
            return ("val" if part == "validation" else part), unit
        raise ValueError(
            f"LibriBrain100 不支持 split_policy={split_policy!r}，可选：simpleb2t, pnpl_official"
        )

    if canonical == "pallier2025_full":
        run_num = _strip_prefix(run, "run-").zfill(2)
        unit = f"pallier2025|run-{run_num}"
        if policy == "simpleb2t":
            if run_num not in PALLIER2025_SIMPLEB2T_SPLIT_BY_RUN:
                raise ValueError(f"Pallier2025 未知运行编号：run-{run_num}")
            return PALLIER2025_SIMPLEB2T_SPLIT_BY_RUN[run_num], unit
        if policy == "manifest":
            manifest_p = Path(
                split_manifest_path
                or (PROJECT_ROOT / "data/manifests/pallier2025/run_split.json")
            )
            payload = json.loads(manifest_p.read_text(encoding="utf-8"))
            assignments = {
                str(run_id): str(split_name)
                for split_name, run_list in payload["assignments"].items()
                for run_id in run_list
            }
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
    pnpl_root: Path | None = None,
) -> pd.DataFrame | None:
    data_path = getattr(pnpl_dataset, "data_path", None) or (
        str(pnpl_root) if pnpl_root is not None else None
    )
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
    preserve_accents = canonical == "pallier2025_full"

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
    elif canonical == "pallier2025_full":
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

    if canonical == "pallier2025_full" and explicit_sents is None and pnpl_dataset is not None:
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
    if canonical == "pallier2025_full":
        return "littleprince_listen"
    return "armeni_sherlock"


def _sidecar_matches(
    output_path: Path,
    sidecar_path: Path,
    *,
    source_h5_path: Path | None,
    target_sampling_rate_hz: float,
    filter_low_hz: float,
    filter_high_hz: float,
) -> dict[str, Any] | None:
    if not output_path.is_file() or not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        preproc = sidecar.get("preprocessing", {})
        if source_h5_path is not None:
            stat = source_h5_path.stat()
            if (
                sidecar.get("source_h5_path") != str(source_h5_path.resolve())
                or sidecar.get("source_h5_size_bytes") != int(stat.st_size)
                or sidecar.get("source_h5_mtime_ns") != int(stat.st_mtime_ns)
            ):
                return None
        if (
            not np.isclose(
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


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        if os.name == "nt":
            st = os.lstat(path)
            return bool(getattr(st, "st_file_attributes", 0) & 0x0400)
    except OSError:
        return True
    return False


def _safe_unlink_inside_root(candidate: Path, pnpl_root: Path) -> bool:
    """安全删除位于 pnpl_root 内部的临时原始文件（绝不穿越软链/Windows Junction）。"""
    try:
        root_res = pnpl_root.resolve()
        if not candidate.exists() or not candidate.is_file():
            return False
        if _is_reparse_or_symlink(candidate):
            return False
        # 逐级检查从 candidate 到 pnpl_root 之间无符号链接或目录联接
        cur = candidate
        found_root = False
        for _ in range(32):
            if cur.resolve() == root_res:
                found_root = True
                break
            if _is_reparse_or_symlink(cur):
                return False
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
        if not found_root:
            return False
        cand_res = candidate.resolve()
        cand_res.relative_to(root_res)
        candidate.unlink()
        return True
    except (OSError, ValueError):
        return False


def _cleanup_run_raw_files(
    pnpl_dataset: Any,
    run_key: tuple[str, str, str, str],
    pnpl_root: Path,
) -> int:
    """释放单条记录在 pnpl_root 内的临时 H5 / 预处理 FIF / 原始 FIF（Colab 防爆盘）。"""
    if hasattr(pnpl_dataset, "close_h5_files"):
        try:
            pnpl_dataset.close_h5_files()
        except Exception:
            pass

    subject, session, task, run = run_key
    candidates: list[Path] = []
    if hasattr(pnpl_dataset, "get_h5_path"):
        try:
            candidates.append(Path(pnpl_dataset.get_h5_path(subject, session, task, run)))
        except Exception:
            pass
    preproc = getattr(pnpl_dataset, "preprocessing", None)
    if preproc and hasattr(pnpl_dataset, "get_preprocessed_path"):
        try:
            candidates.append(
                Path(
                    pnpl_dataset.get_preprocessed_path(
                        subject, session, task, run, preprocessing=preproc, extension="fif"
                    )
                )
            )
        except Exception:
            pass
    if hasattr(pnpl_dataset, "get_bids_raw_path"):
        try:
            candidates.append(Path(pnpl_dataset.get_bids_raw_path(subject, session, task, run)))
        except Exception:
            pass

    removed = 0
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if _safe_unlink_inside_root(p, pnpl_root):
            removed += 1
    return removed


def _libribrain_local_events_path(
    pnpl_root: Path, run_key: tuple[str, str, str, str]
) -> Path:
    subject, session, task, run = map(str, run_key)
    fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
    return pnpl_root / task / "derivatives" / "events" / fname


def _cleanup_pnpl_local_hf_cache(pnpl_root: Path) -> None:
    """清理 local_dir/.cache/huggingface/download 中残留的未完成分片，避免占用本地盘。"""
    download_cache = pnpl_root / ".cache" / "huggingface" / "download"
    if not download_cache.is_dir() or _is_reparse_or_symlink(download_cache):
        return
    for entry in download_cache.rglob("*.incomplete"):
        _safe_unlink_inside_root(entry, pnpl_root)


def _resolve_libribrain_streaming_records(
    *,
    subjects: Any,
    corpus: Any,
    sessions: Sequence[str] | None,
    runs: Sequence[str] | None,
    tasks: Sequence[str] | None,
    include_run_keys: Sequence[tuple[str, str, str, str]] | None,
    exclude_run_keys: Sequence[tuple[str, str, str, str]] | None,
) -> list[tuple[tuple[str, str, str, str], str]] | None:
    """零下载解析 LibriBrain100 待物化 (run_key, corpus) 列表，避免 __init__ 预拉全部 114 个 H5 爆盘。"""
    excluded = (
        {tuple(map(str, rk)) for rk in exclude_run_keys} if exclude_run_keys else set()
    )
    if include_run_keys:
        get_rec = None
        try:
            manifest_mod = importlib.import_module("pnpl.datasets.libribrain100.manifest")
            get_rec = getattr(manifest_mod, "get_record", None)
        except Exception:
            get_rec = None
        resolved: list[tuple[tuple[str, str, str, str], str]] = []
        for rk_raw in include_run_keys:
            rk = tuple(map(str, rk_raw))
            if rk in excluded:
                continue
            corpus_name = str(rk[2]).lower()
            if get_rec is not None:
                try:
                    rec = get_rec(rk)
                    if getattr(rec, "corpus", None):
                        corpus_name = str(rec.corpus)
                except Exception:
                    pass
            resolved.append((rk, corpus_name))
        return resolved

    try:
        base_mod = importlib.import_module("pnpl.datasets.libribrain100.base")
        normalize_subjects = getattr(base_mod, "normalize_subjects")
        normalize_corpus = getattr(base_mod, "normalize_corpus")
        select_records = getattr(base_mod, "select_records")
    except Exception:
        return None

    if subjects is None:
        lb_subjects: Any = "deep"
    elif isinstance(subjects, str):
        low_s = subjects.strip().lower()
        lb_subjects = (
            low_s
            if low_s in {"all", "deep", "broad"}
            else [_strip_prefix(subjects, "sub-")]
        )
    else:
        lb_subjects = [_strip_prefix(str(s), "sub-") for s in subjects]

    recs = select_records(
        subjects=normalize_subjects(lb_subjects),
        corpus=normalize_corpus(corpus),
        partition=None,
        exclude_run_keys=list(excluded) if excluded else None,
    )
    ses_set = {_strip_prefix(str(s), "ses-") for s in sessions} if sessions else None
    run_set = {_strip_prefix(str(r), "run-") for r in runs} if runs else None
    task_set = {str(t).strip() for t in tasks} if tasks else None

    out: list[tuple[tuple[str, str, str, str], str]] = []
    for rec in recs:
        rk = tuple(map(str, rec.run_key))
        if rk in excluded:
            continue
        if ses_set is not None and _strip_prefix(rk[1], "ses-") not in ses_set:
            continue
        if task_set is not None and rk[2] not in task_set:
            continue
        if run_set is not None and _strip_prefix(rk[3], "run-") not in run_set:
            continue
        out.append((rk, str(getattr(rec, "corpus", "") or rk[2])))
    return out


PALLIER2025_OPENNEURO_S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds007523"


def _pallier_pad2(val: str, prefix: str) -> str:
    tok = _strip_prefix(str(val), prefix)
    return tok.zfill(2) if tok.isdigit() else tok


def _pallier_openneuro_url(rel_path: str) -> str:
    clean = rel_path.lstrip("/")
    encoded = "/".join(quote(seg, safe="") for seg in clean.split("/"))
    return f"{PALLIER2025_OPENNEURO_S3_BASE}/{encoded}"


def _pallier_local_bids_paths(
    pnpl_root: Path, run_key: tuple[str, str, str, str]
) -> tuple[Path, Path, Path]:
    sub, ses, task, run = (
        _pallier_pad2(run_key[0], "sub-"),
        _pallier_pad2(run_key[1], "ses-"),
        str(run_key[2]),
        _pallier_pad2(run_key[3], "run-"),
    )
    meg_dir = pnpl_root / f"sub-{sub}" / f"ses-{ses}" / "meg"
    raw_fif = meg_dir / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_meg.fif"
    events_tsv = meg_dir / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv"
    extra_tsv = pnpl_root / "sourcedata" / f"task-{task}_run-{run}_extra_info.tsv"
    return raw_fif, events_tsv, extra_tsv


def _resolve_pallier_subject_run_keys(
    subject_token: str,
    *,
    sessions: Sequence[str] | None,
    runs: Sequence[str] | None,
    tasks: Sequence[str] | None,
    include_run_keys: Sequence[tuple[str, str, str, str]] | None,
    exclude_run_keys: Sequence[tuple[str, str, str, str]] | None,
) -> list[tuple[str, str, str, str]]:
    sub = _pallier_pad2(subject_token, "sub-")
    excluded = (
        {
            (
                _pallier_pad2(rk[0], "sub-"),
                _pallier_pad2(rk[1], "ses-"),
                str(rk[2]),
                _pallier_pad2(rk[3], "run-"),
            )
            for rk in exclude_run_keys
        }
        if exclude_run_keys
        else set()
    )
    if include_run_keys:
        out = []
        for rk in include_run_keys:
            norm = (
                _pallier_pad2(rk[0], "sub-"),
                _pallier_pad2(rk[1], "ses-"),
                str(rk[2]),
                _pallier_pad2(rk[3], "run-"),
            )
            if norm[0] == sub and norm not in excluded:
                out.append(norm)
        return out

    ses_list = [_pallier_pad2(s, "ses-") for s in sessions] if sessions else ["01"]
    task_list = [str(t).strip() for t in tasks] if tasks else ["listen"]
    run_list = (
        [_pallier_pad2(r, "run-") for r in runs]
        if runs
        else [f"{idx:02d}" for idx in range(1, 10)]
    )
    return [
        (sub, ses, task, run)
        for ses in ses_list
        for task in task_list
        for run in run_list
        if (sub, ses, task, run) not in excluded
    ]


def _download_single_http_file(url: str, dest: Path, *, max_retries: int = 4) -> bool:
    import time
    import requests

    if dest.is_file():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(dest.name + ".tmp")
    for attempt in range(1, max_retries + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            with requests.get(url, stream=True, timeout=120) as resp:
                if resp.status_code in (403, 404):
                    return False
                resp.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            os.replace(tmp_path, dest)
            return True
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt >= max_retries:
                return False
            time.sleep(min(8.0, float(2**attempt)))
    return False


def _prefetch_pallier_subject_from_openneuro(
    pnpl_root: Path,
    run_keys: Sequence[tuple[str, str, str, str]],
) -> None:
    """并发多连接预拉取 Pallier2025 单被试原始 FIF 与 events.tsv（优先 aria2c -x 8 -s 8 -j 4，回退 4 线程 requests）。"""
    download_items: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    for rk in run_keys:
        sub, ses, task, run = rk
        raw_fif, events_tsv, extra_tsv = _pallier_local_bids_paths(pnpl_root, rk)
        candidates = [
            (f"sub-{sub}/ses-{ses}/meg/{raw_fif.name}", raw_fif),
            (f"sub-{sub}/ses-{ses}/meg/{events_tsv.name}", events_tsv),
            (f"sourcedata/{extra_tsv.name}", extra_tsv),
        ]
        for rel_path, local_path in candidates:
            if local_path in seen_paths or local_path.is_file():
                continue
            seen_paths.add(local_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            download_items.append((_pallier_openneuro_url(rel_path), local_path))

    if not download_items:
        return

    aria2_bin = shutil.which("aria2c")
    if aria2_bin:
        sub_tag = run_keys[0][0] if run_keys else "batch"
        list_file = pnpl_root / f".aria2_prefetch_sub-{sub_tag}.txt"
        lines: list[str] = []
        for url, dest in download_items:
            lines.append(url)
            lines.append(f"  dir={dest.parent}")
            lines.append(f"  out={dest.name}")
        list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            subprocess.run(
                [
                    aria2_bin,
                    "--continue=true",
                    "--auto-file-renaming=false",
                    "--allow-overwrite=false",
                    "--max-connection-per-server=8",
                    "--split=8",
                    "--min-split-size=4M",
                    "--max-concurrent-downloads=4",
                    "--max-tries=5",
                    "--retry-wait=2",
                    "--connect-timeout=30",
                    "--timeout=120",
                    "--console-log-level=error",
                    "--summary-interval=0",
                    "--download-result=hide",
                    "-i",
                    str(list_file),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            list_file.unlink(missing_ok=True)
            for _, dest in download_items:
                ctrl = Path(str(dest) + ".aria2")
                if ctrl.exists():
                    ctrl.unlink(missing_ok=True)
                    dest.unlink(missing_ok=True)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(_download_single_http_file, url, dest)
            for url, dest in download_items
        ]
        for fut in futures:
            try:
                fut.result()
            except Exception:
                pass


def _resolve_streaming_subjects(
    canonical: str,
    *,
    subjects: Any,
    include_run_keys: Sequence[tuple[str, str, str, str]] | None,
) -> list[str]:
    if include_run_keys:
        ordered: list[str] = []
        for rk in include_run_keys:
            tok = str(rk[0])
            if tok not in ordered:
                ordered.append(tok)
        return ordered
    if isinstance(subjects, (list, tuple, set)):
        return [str(s) for s in subjects]
    if isinstance(subjects, str) and subjects.strip().lower() != "all":
        return [subjects.strip()]
    if canonical == "pallier2025_full":
        return [f"{idx:02d}" for idx in range(1, 59)]
    return ["001", "002", "003"]


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
    cleanup_raw: bool = False,
    force: bool = False,
    show_progress: bool = True,
    pnpl_dataset: Any = None,
    pnpl_datasets_module: Any = None,
) -> dict[str, Any]:
    """通过 PNPL 拉取连续 MEG H5 与词事件表，并转存为本仓 50 Hz + RobustScaler 缓存与规范事件表。"""
    canonical = _normalize_dataset_name(dataset_name)
    out_root = Path(output_root)
    pnpl_root_path = Path(pnpl_data_path)
    meg_dir = out_root / "signals" / "meg_50hz"
    events_path = out_root / "events" / "events.csv"
    meg_dir.mkdir(parents=True, exist_ok=True)

    tqdm_cls = None
    if show_progress:
        try:
            from tqdm import tqdm as _tqdm_cls

            tqdm_cls = _tqdm_cls
        except ImportError:
            tqdm_cls = None

    # Colab 防爆盘：
    # - LibriBrain100.__init__ 即使 preload_files=False 也会在循环中预先拉完所有选中 run 的 250 Hz H5（114 runs ~180 GB），
    #   开启 cleanup_raw 时按单条 run_key 逐个物化并立即删临时 H5，将峰值盘占压至 <15 GB。
    # - Pallier2025 / Armeni2022 的 PNPL __init__ 会批量下载全部选中被试原始 FIF，
    #   开启 cleanup_raw 时按单个被试逐批物化并及时删除临时 FIF/H5，将峰值盘占压至 <15 GB。
    libribrain_stream_records = (
        _resolve_libribrain_streaming_records(
            subjects=subjects,
            corpus=corpus,
            sessions=sessions,
            runs=runs,
            tasks=tasks,
            include_run_keys=include_run_keys,
            exclude_run_keys=exclude_run_keys,
        )
        if (cleanup_raw and pnpl_dataset is None and canonical == "libribrain100")
        else None
    )
    stream_by_run = bool(libribrain_stream_records)
    stream_by_subject = bool(
        cleanup_raw
        and pnpl_dataset is None
        and canonical in {"pallier2025_full", "armeni2022"}
    )
    # Pallier2025 高速直通模式：多连接并发拉原始 FIF -> 多线程整段 MNE 滤波重采样 -> 写 .npy 即删 FIF，
    # 跳过 pnpl 单连接 requests 下载、单线程 120s 分块 bp+ds 与冗余 preproc.fif/H5 落盘（提速 ~5x 且零刷屏告警）。
    pallier_fast_direct = bool(
        stream_by_subject
        and canonical == "pallier2025_full"
        and pnpl_datasets_module is None
        and preprocessing is None
    )
    subject_batches: list[Any] = (
        [[s] for s in _resolve_streaming_subjects(canonical, subjects=subjects, include_run_keys=include_run_keys)]
        if stream_by_subject
        else [subjects]
    )

    resolved_mne_n_jobs = _resolve_mne_n_jobs(mne_n_jobs)
    event_frames: list[pd.DataFrame] = []
    signal_records: list[dict[str, Any]] = []
    all_run_keys: list[tuple[str, str, str, str]] = []
    reused_count = 0
    materialized_count = 0
    cleaned_raw_count = 0
    total_words_extracted = 0

    def _materialize_run(
        run_key: tuple[str, str, str, str],
        *,
        active_ds: Any,
        fallback_corpus: str = "",
        prematched_sidecar: dict[str, Any] | None = None,
        prematched_events_path: Path | None = None,
        direct_raw_fif_path: Path | None = None,
    ) -> None:
        nonlocal reused_count, materialized_count, cleaned_raw_count, total_words_extracted

        all_run_keys.append(run_key)
        subject, session, task, run = run_key
        sub_id = _format_subject_id(subject)
        rec_id = recording_identifier(canonical, run_key)
        split, split_unit = resolve_split_and_unit(
            canonical,
            run_key,
            split_policy=split_policy,
            pnpl_dataset=active_ds,
            split_manifest_path=split_manifest_path,
        )

        stem = recording_signal_stem(canonical, run_key)
        output_npy = meg_dir / sub_id / f"{stem}.npy"
        sidecar_json = output_npy.with_suffix(".json")
        signal_relpath = output_npy.relative_to(meg_dir).as_posix()

        reused_sidecar = prematched_sidecar
        h5_path: Path | None = None
        if reused_sidecar is None and not force and cleanup_raw:
            # 断点续跑优先校验已有 50 Hz 产物，避免重复下载已清理的临时 H5/FIF
            reused_sidecar = _sidecar_matches(
                output_npy,
                sidecar_json,
                source_h5_path=None,
                target_sampling_rate_hz=target_sampling_rate_hz,
                filter_low_hz=filter_low_hz,
                filter_high_hz=filter_high_hz,
            )

        if reused_sidecar is None:
            if direct_raw_fif_path is not None:
                h5_path = direct_raw_fif_path
            else:
                assert active_ds is not None
                h5_path = _resolve_pnpl_h5_path(active_ds, run_key)
            if not force:
                reused_sidecar = _sidecar_matches(
                    output_npy,
                    sidecar_json,
                    source_h5_path=h5_path,
                    target_sampling_rate_hz=target_sampling_rate_hz,
                    filter_low_hz=filter_low_hz,
                    filter_high_hz=filter_high_hz,
                )

        if prematched_events_path is not None and prematched_events_path.is_file():
            events_tsv_path = prematched_events_path
        else:
            assert active_ds is not None
            events_tsv_path = _resolve_pnpl_events_path(active_ds, run_key)

        if reused_sidecar is not None:
            sidecar = reused_sidecar
            reused_count += 1
        else:
            assert h5_path is not None
            if direct_raw_fif_path is not None:
                raw_array, h5_meta = _read_raw_fif_continuous_meg(direct_raw_fif_path)
                first_samp = int(h5_meta["source_first_samp"])
                time_origin_sfreq = float(h5_meta["source_sampling_rate_hz"])
                time_origin_src = f"fif_header:{direct_raw_fif_path.name}"
            else:
                raw_array, h5_meta = _read_h5_continuous_meg(
                    h5_path,
                    fallback_sfreq=getattr(active_ds, "sfreq", None),
                )
                if canonical == "pallier2025_full":
                    first_samp, time_origin_sfreq, time_origin_src = (
                        _inspect_pallier_time_origin(active_ds, run_key, h5_meta)
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
                mne_n_jobs=resolved_mne_n_jobs,
            )
            del raw_array
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
            if direct_raw_fif_path is not None:
                gc.collect()

        corpus_name = (
            _corpus_for_run_key(canonical, run_key, active_ds)
            if active_ds is not None
            else (fallback_corpus or str(run_key[2]).lower())
        )
        events_ds_context = (
            active_ds
            if active_ds is not None
            else SimpleNamespace(data_path=str(pnpl_root_path))
        )
        frame = extract_run_word_events(
            canonical,
            events_tsv_path,
            run_key=run_key,
            split=split,
            split_unit=split_unit,
            signal_metadata=sidecar,
            signal_relpath=signal_relpath,
            corpus_name=corpus_name,
            pnpl_dataset=events_ds_context,
            eligibility_window_seconds=eligibility_window_seconds,
            pause_gap_seconds=pause_gap_seconds,
            max_context_words=max_context_words,
        )
        if not frame.empty:
            event_frames.append(frame)
            total_words_extracted += int(len(frame))

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
        if cleanup_raw:
            if active_ds is not None:
                cleaned_raw_count += _cleanup_run_raw_files(
                    active_ds, run_key, pnpl_root_path
                )
                _cleanup_pnpl_local_hf_cache(pnpl_root_path)
            elif direct_raw_fif_path is not None:
                if _safe_unlink_inside_root(direct_raw_fif_path, pnpl_root_path):
                    cleaned_raw_count += 1

    subject_pbar = (
        tqdm_cls(
            total=len(subject_batches),
            desc=f"PNPL[{canonical}]",
            unit="sub",
            ncols=92,
            mininterval=2.0,
        )
        if (stream_by_subject and tqdm_cls is not None)
        else None
    )

    with _suppress_nested_download_bars(disable_xet=bool(cleanup_raw)):
        if stream_by_run:
            assert libribrain_stream_records is not None
            run_pbar = (
                tqdm_cls(
                    total=len(libribrain_stream_records),
                    desc=f"PNPL[{canonical}]",
                    unit="run",
                    ncols=92,
                    mininterval=2.0,
                    leave=True,
                )
                if tqdm_cls is not None
                else None
            )
            try:
                for index, (run_key, rec_corpus) in enumerate(libribrain_stream_records, start=1):
                    subject, session, task, run = run_key
                    sub_id = _format_subject_id(subject)
                    rec_id = recording_identifier(canonical, run_key)
                    stem = recording_signal_stem(canonical, run_key)
                    output_npy = meg_dir / sub_id / f"{stem}.npy"
                    sidecar_json = output_npy.with_suffix(".json")
                    local_events = _libribrain_local_events_path(pnpl_root_path, run_key)

                    fast_sidecar = None
                    if not force and local_events.is_file():
                        fast_sidecar = _sidecar_matches(
                            output_npy,
                            sidecar_json,
                            source_h5_path=None,
                            target_sampling_rate_hz=target_sampling_rate_hz,
                            filter_low_hz=filter_low_hz,
                            filter_high_hz=filter_high_hz,
                        )

                    if fast_sidecar is not None:
                        _materialize_run(
                            run_key,
                            active_ds=None,
                            fallback_corpus=rec_corpus,
                            prematched_sidecar=fast_sidecar,
                            prematched_events_path=local_events,
                        )
                    else:
                        active_ds = open_pnpl_dataset(
                            canonical,
                            pnpl_data_path,
                            subjects=subjects,
                            corpus=corpus,
                            include_run_keys=[run_key],
                            exclude_run_keys=exclude_run_keys,
                            preprocessing=preprocessing,
                            preprocessing_config=preprocessing_config,
                            download=download,
                            preload_files=False,
                            preload_h5=False,
                            create_h5_if_missing=create_h5_if_missing,
                            pnpl_datasets_module=pnpl_datasets_module,
                        )
                        try:
                            if getattr(active_ds, "standardize_enabled", False):
                                raise RuntimeError(
                                    "PNPL 数据集必须以 standardize=False 实例化，避免在 RobustScaler 前混入 z-score。"
                                )
                            active_keys = [
                                tuple(map(str, rk))
                                for rk in getattr(active_ds, "run_keys", ())
                            ]
                            for active_rk in active_keys:
                                _materialize_run(
                                    active_rk,
                                    active_ds=active_ds,
                                    fallback_corpus=rec_corpus,
                                )
                        finally:
                            if hasattr(active_ds, "close_h5_files"):
                                try:
                                    active_ds.close_h5_files()
                                except Exception:
                                    pass

                    if run_pbar is not None:
                        run_pbar.set_postfix_str(
                            f"{sub_id}/r{_strip_prefix(run, 'run-')} w={total_words_extracted}",
                            refresh=False,
                        )
                        run_pbar.update(1)
                    elif index == 1 or index % 10 == 0 or index == len(libribrain_stream_records):
                        print(
                            f"PNPL[{canonical}] {index}/{len(libribrain_stream_records)}: {rec_id} "
                            f"words={total_words_extracted} reused={reused_count}",
                            flush=True,
                        )
            finally:
                if run_pbar is not None:
                    run_pbar.close()
        else:
            try:
                for batch_idx, batch_subjects in enumerate(subject_batches, start=1):
                    batch_include_keys = include_run_keys
                    sub_label = (
                        _format_subject_id(str(batch_subjects[0]))
                        if stream_by_subject and batch_subjects
                        else canonical
                    )
                    if stream_by_subject and include_run_keys is not None:
                        want_sub = _strip_prefix(str(batch_subjects[0]), "sub-")
                        batch_include_keys = [
                            rk
                            for rk in include_run_keys
                            if _strip_prefix(str(rk[0]), "sub-").lstrip("0") == want_sub.lstrip("0")
                        ]
                        if not batch_include_keys:
                            if subject_pbar is not None:
                                subject_pbar.update(1)
                            continue

                    if subject_pbar is not None:
                        subject_pbar.set_postfix_str(
                            f"{sub_label} fetch w={total_words_extracted}",
                            refresh=True,
                        )

                    if pallier_fast_direct:
                        sub_run_keys = _resolve_pallier_subject_run_keys(
                            str(batch_subjects[0]),
                            sessions=sessions,
                            runs=runs,
                            tasks=tasks,
                            include_run_keys=batch_include_keys,
                            exclude_run_keys=exclude_run_keys,
                        )
                        if not sub_run_keys:
                            if subject_pbar is not None:
                                subject_pbar.update(1)
                            continue

                        prematched_by_rk: dict[tuple[str, str, str, str], tuple[dict[str, Any], Path]] = {}
                        need_fetch_rks: list[tuple[str, str, str, str]] = []
                        for rk in sub_run_keys:
                            sub_id_rk = _format_subject_id(rk[0])
                            stem_rk = recording_signal_stem(canonical, rk)
                            out_npy_rk = meg_dir / sub_id_rk / f"{stem_rk}.npy"
                            sidecar_rk = out_npy_rk.with_suffix(".json")
                            _, ev_rk, _ = _pallier_local_bids_paths(pnpl_root_path, rk)
                            fast_sc = None
                            if not force and ev_rk.is_file():
                                fast_sc = _sidecar_matches(
                                    out_npy_rk,
                                    sidecar_rk,
                                    source_h5_path=None,
                                    target_sampling_rate_hz=target_sampling_rate_hz,
                                    filter_low_hz=filter_low_hz,
                                    filter_high_hz=filter_high_hz,
                                )
                            if fast_sc is not None:
                                prematched_by_rk[rk] = (fast_sc, ev_rk)
                            else:
                                need_fetch_rks.append(rk)

                        if download and need_fetch_rks:
                            _prefetch_pallier_subject_from_openneuro(
                                pnpl_root_path, need_fetch_rks
                            )

                        ready_rks: list[tuple[tuple[str, str, str, str], Path | None, Path]] = []
                        for rk in sub_run_keys:
                            if rk in prematched_by_rk:
                                ready_rks.append((rk, None, prematched_by_rk[rk][1]))
                                continue
                            raw_fif_rk, ev_rk, _ = _pallier_local_bids_paths(pnpl_root_path, rk)
                            if raw_fif_rk.is_file() and ev_rk.is_file():
                                ready_rks.append((rk, raw_fif_rk, ev_rk))

                        if ready_rks:
                            run_pbar = (
                                tqdm_cls(
                                    total=len(ready_rks),
                                    desc=f"PNPL[{sub_label}]",
                                    unit="run",
                                    ncols=92,
                                    mininterval=2.0,
                                    leave=False,
                                )
                                if tqdm_cls is not None
                                else None
                            )
                            try:
                                for index, (rk, raw_fif_rk, ev_rk) in enumerate(ready_rks, start=1):
                                    sub_id_rk = _format_subject_id(rk[0])
                                    rec_id_rk = recording_identifier(canonical, rk)
                                    if rk in prematched_by_rk:
                                        _materialize_run(
                                            rk,
                                            active_ds=None,
                                            fallback_corpus="littleprince_listen",
                                            prematched_sidecar=prematched_by_rk[rk][0],
                                            prematched_events_path=ev_rk,
                                        )
                                    else:
                                        _materialize_run(
                                            rk,
                                            active_ds=None,
                                            fallback_corpus="littleprince_listen",
                                            prematched_events_path=ev_rk,
                                            direct_raw_fif_path=raw_fif_rk,
                                        )
                                    if run_pbar is not None:
                                        run_pbar.set_postfix_str(
                                            f"{sub_id_rk}/r{_strip_prefix(rk[3], 'run-')} w={total_words_extracted}",
                                            refresh=False,
                                        )
                                        run_pbar.update(1)
                                    elif index == 1 or index == len(ready_rks):
                                        print(
                                            f"PNPL[{canonical}] {sub_label} {index}/{len(ready_rks)}: "
                                            f"{rec_id_rk} words={total_words_extracted} reused={reused_count}",
                                            flush=True,
                                        )
                            finally:
                                if run_pbar is not None:
                                    run_pbar.close()
                                if subject_pbar is not None:
                                    subject_pbar.set_postfix_str(
                                        f"{sub_label} done w={total_words_extracted}",
                                        refresh=False,
                                    )
                                    subject_pbar.update(1)
                            continue

                    opened_locally = pnpl_dataset is None
                    active_ds = pnpl_dataset
                    if active_ds is None:
                        active_ds = open_pnpl_dataset(
                            canonical,
                            pnpl_data_path,
                            subjects=batch_subjects,
                            corpus=corpus,
                            sessions=sessions,
                            runs=runs,
                            tasks=tasks,
                            include_run_keys=batch_include_keys,
                            exclude_run_keys=exclude_run_keys,
                            preprocessing=preprocessing,
                            preprocessing_config=preprocessing_config,
                            download=download,
                            preload_files=preload_files,
                            preload_h5=False,
                            create_h5_if_missing=create_h5_if_missing,
                            pnpl_datasets_module=pnpl_datasets_module,
                        )

                    run_pbar = None
                    try:
                        # 防御性检查：确保 PNPL 内置 z-score 标准化已关闭。
                        if getattr(active_ds, "standardize_enabled", False):
                            raise RuntimeError(
                                "PNPL 数据集必须以 standardize=False 实例化，避免在 RobustScaler 前混入 z-score。"
                            )

                        run_keys = [tuple(map(str, rk)) for rk in getattr(active_ds, "run_keys", ())]
                        if not run_keys:
                            if stream_by_subject:
                                continue
                            raise ValueError(f"PNPL 数据集 {canonical} 未解析到任何可用 run_key。")

                        if tqdm_cls is not None:
                            run_pbar = tqdm_cls(
                                total=len(run_keys),
                                desc=(
                                    f"PNPL[{sub_label}]"
                                    if stream_by_subject
                                    else f"PNPL[{canonical}]"
                                ),
                                unit="run",
                                ncols=92,
                                mininterval=2.0,
                                leave=not stream_by_subject,
                            )

                        for index, run_key in enumerate(run_keys, start=1):
                            subject, _, _, run = run_key
                            sub_id = _format_subject_id(subject)
                            rec_id = recording_identifier(canonical, run_key)
                            _materialize_run(run_key, active_ds=active_ds)
                            if run_pbar is not None:
                                run_pbar.set_postfix_str(
                                    f"{sub_id}/r{_strip_prefix(run, 'run-')} w={total_words_extracted}",
                                    refresh=False,
                                )
                                run_pbar.update(1)
                            elif index == 1 or index % 10 == 0 or index == len(run_keys):
                                print(
                                    f"PNPL[{canonical}] {index}/{len(run_keys)}: {rec_id} "
                                    f"words={total_words_extracted} reused={reused_count}",
                                    flush=True,
                                )
                    finally:
                        if run_pbar is not None:
                            run_pbar.close()
                        if opened_locally and hasattr(active_ds, "close_h5_files"):
                            try:
                                active_ds.close_h5_files()
                            except Exception:
                                pass
                        if subject_pbar is not None:
                            subject_pbar.set_postfix_str(
                                f"{sub_label} done w={total_words_extracted}",
                                refresh=False,
                            )
                            subject_pbar.update(1)
            finally:
                if subject_pbar is not None:
                    subject_pbar.close()

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
            "source_raw_roots": [pnpl_root_path],
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
                "corpus": corpus,
                "run_keys": [list(rk) for rk in all_run_keys],
            },
            "audit": {
                "recording_count": len(all_run_keys),
                "materialized_recordings": materialized_count,
                "reused_recordings": reused_count,
                "cleaned_raw_files": cleaned_raw_count,
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
        "corpus": corpus,
        "mne_n_jobs": int(resolved_mne_n_jobs),
        "cleanup_raw": bool(cleanup_raw),
        "cleaned_raw_files": int(cleaned_raw_count),
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


def _default_output_root(dataset_name: str) -> Path:
    canonical = _normalize_dataset_name(dataset_name)
    paths = canonical_cache_paths(canonical)
    return project_path(paths['event_table']).parent.parent


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
        help="划分策略：libribrain100(simpleb2t/pnpl_official), pallier2025_full(simpleb2t/manifest), armeni2022(simpleb2t)",
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
    parser.add_argument(
        "--cleanup-raw",
        action="store_true",
        help="每物化完一条记录/被试后立即删除临时原始 H5/FIF（Colab 防爆盘）",
    )
    parser.add_argument("--force", action="store_true", help="强制重建已有 50 Hz 缓存")
    parser.add_argument(
        "--mne-n-jobs",
        type=int,
        default=1,
        help="MNE 带通滤波与重采样线程并行数（默认 1；设 0 自动取 min(4, CPU核数)）",
    )
    parser.add_argument("--no-progress", action="store_true", help="禁用 tqdm 实时进度条")
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
        cleanup_raw=args.cleanup_raw,
        force=args.force,
        mne_n_jobs=args.mne_n_jobs,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
