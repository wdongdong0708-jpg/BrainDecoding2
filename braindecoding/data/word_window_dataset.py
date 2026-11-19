"""全仓统一的 50 Hz 派生 MEG 词级时间窗数据集与公共事件/信号辅助模块。

同时原生兼容两类 50 Hz 派生缓存布局：
1. PNPL 统一适配器缓存（``libribrain100`` / ``pallier2025_full`` / ``armeni2022``）：
   事件表携带 ``信号相对路径`` 列，支持从 ``events.csv`` 动态提取有序受试者列表及多设备
   传感器布局（Neuromag Vectorview-306 / CTF-275 / sidecar ``sensor_xyz``）。
2. 旧 10 被试本地 BIDS ``pallier2025`` 派生缓存：
   事件表通过 ``(受试者, 运行编号)`` 定位 ``sub-XX/sub-XX_ses-01_task-listen_run-YY_meg.npy``，
   在配置显式提供 ``subjects`` 时严格维持原有 ``subject_map`` 索引顺序以保证既有 ``.pt`` 权重免重训加载。
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from braindecoding.data.event_table import CORE_EVENT_COLUMNS
from braindecoding.data.sensor_coordinates import vectorview_channel_positions
import braindecoding.data.text_embeddings as text_data


SOURCE_SAMPLING_RATE_HZ = 1000.0
TARGET_SAMPLING_RATE_HZ = 50.0
WINDOW_START_OFFSET_SECONDS = 0.0
WINDOW_SECONDS = 1.0
ELIGIBILITY_WINDOW_SECONDS = 3.0
BASELINE_SECONDS = 0.5
CLAMP = 5.0
DEFAULT_MAX_CONTEXT_WORDS = 128

RUNTIME_CONTEXT_COLUMN = "runtime_context_uid"
RUNTIME_CONTEXT_GROUPING = "recording_id_plus_canonical_context_v1"
VALID_TEMPORAL_SHIFT_MODES = ("deterministic_circular", "preroll")
DEFAULT_PREROLL_SECONDS = 45.0


def format_subject_id(subject: Any) -> str:
    """将受试者标识标准化为 ``sub-*`` 形态。"""
    raw = str(subject).strip()
    return raw if raw.startswith("sub-") else f"sub-{raw}"


def parse_bool_value(value: Any, *, field: str = "bool") -> bool:
    """解析事件表 CSV 中的严格布尔值。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"字段 {field} 包含非法布尔值：{value!r}")


def channel_names_sha256(channel_names: Sequence[str]) -> str:
    """按换行连接规则计算通道顺序摘要。"""
    payload = "\n".join(str(name) for name in channel_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recording_onset_to_sample(
    onset_seconds: float,
    *,
    source_first_samp: int | float,
    source_sampling_rate_hz: float = SOURCE_SAMPLING_RATE_HZ,
    output_sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ,
) -> int:
    """将记录时钟 onset 转换为 50 Hz 连续派生信号的零起点采样下标。"""
    recording_start = float(source_first_samp) / float(source_sampling_rate_hz)
    relative_seconds = float(onset_seconds) - recording_start
    return int(np.round(relative_seconds * float(output_sampling_rate_hz)))


def recording_window_bounds(
    onset_seconds: float,
    *,
    duration_seconds: float,
    source_first_samp: int | float,
    source_sampling_rate_hz: float = SOURCE_SAMPLING_RATE_HZ,
    output_sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ,
) -> tuple[int, int]:
    """返回左闭右开的派生信号窗口样本范围。"""
    start = recording_onset_to_sample(
        onset_seconds,
        source_first_samp=source_first_samp,
        source_sampling_rate_hz=source_sampling_rate_hz,
        output_sampling_rate_hz=output_sampling_rate_hz,
    )
    length = int(np.round(float(duration_seconds) * float(output_sampling_rate_hz)))
    return start, start + length


def extract_ordered_subjects(table: pd.DataFrame) -> tuple[str, ...]:
    """按事件表首次出现顺序提取有序受试者列表（适配任意单/多被试 MEG 缓存）。"""
    subj_col = "受试者" if "受试者" in table.columns else "subject_id"
    if subj_col not in table.columns:
        raise ValueError("事件表缺少受试者列（'受试者' 或 'subject_id'）。")
    return tuple(dict.fromkeys(table[subj_col].astype(str).tolist()))


def add_runtime_context_uid(
    table: pd.DataFrame, max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS
) -> pd.DataFrame:
    """建立受试者记录内的运行时上下文键，并校验绝不跨受试者或跨记录泄漏。"""
    required = {"受试者", "记录编号", "上下文编号", "记录内序号"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"运行时上下文缺少字段：{missing}")
    maximum = int(max_context_words)
    if maximum <= 0:
        raise ValueError("max_context_words 必须为正整数。")

    result = table.copy()
    result[RUNTIME_CONTEXT_COLUMN] = (
        result["记录编号"].astype(str) + "|" + result["上下文编号"].astype(str)
    )
    grouped = result.groupby(RUNTIME_CONTEXT_COLUMN, sort=False)
    if grouped["受试者"].nunique().gt(1).any():
        raise ValueError("运行时上下文跨受试者。")
    if grouped["记录编号"].nunique().gt(1).any():
        raise ValueError("运行时上下文跨 recording。")
    if grouped["上下文编号"].nunique().gt(1).any():
        raise ValueError("运行时上下文混合多个 canonical context。")
    sizes = grouped.size()
    oversized = sizes[sizes.gt(maximum)]
    if not oversized.empty:
        raise ValueError(
            f"运行时上下文超过 max_context_words={maximum}："
            f"{oversized.index[0]}={int(oversized.iloc[0])}"
        )
    for context_uid, rows in grouped:
        order = pd.to_numeric(rows["记录内序号"], errors="coerce")
        if order.isna().any() or (np.diff(order.to_numpy()) <= 0).any():
            raise ValueError(f"运行时上下文记录内序号未严格递增：{context_uid}")
    return result


def runtime_context_statistics(
    table: pd.DataFrame, max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS
) -> dict[str, Any]:
    """汇总受试者记录内运行时上下文的只读统计。"""
    runtime = add_runtime_context_uid(table, max_context_words=max_context_words)
    grouped = runtime.groupby(RUNTIME_CONTEXT_COLUMN, sort=False)
    sizes = grouped.size()
    return {
        "context_count": int(len(sizes)),
        "mean_words": float(sizes.mean()) if len(sizes) else 0.0,
        "median_words": float(sizes.median()) if len(sizes) else 0.0,
        "maximum_words": int(sizes.max()) if len(sizes) else 0,
        "cross_subject_groups": int(grouped["受试者"].nunique().gt(1).sum()),
        "cross_recording_groups": int(grouped["记录编号"].nunique().gt(1).sum()),
        "cross_canonical_context_groups": int(grouped["上下文编号"].nunique().gt(1).sum()),
        "groups_over_maximum": int(sizes.gt(int(max_context_words)).sum()),
    }


def pallier_legacy_signal_paths(
    cache_dir: str | Path, subject: str, run: str, *, session: str = "ses-01"
) -> tuple[Path, Path]:
    """返回旧 10 被试 Pallier2025 派生 MEG 信号及其 sidecar 路径。"""
    stem = f"{subject}_{session}_task-listen_{run}_meg"
    signal_path = Path(cache_dir) / str(subject) / f"{stem}.npy"
    return signal_path, signal_path.with_suffix(".json")


@lru_cache(maxsize=256)
def open_derived_signal(path: str) -> np.ndarray:
    """只读 mmap 打开 50 Hz 连续派生 MEG 信号，避免 DataLoader 反复整文件载入。"""
    return np.load(path, mmap_mode="r")


def resolve_meg_channel_positions(
    channel_names: Sequence[str],
    *,
    layout_path: str | Path | None = None,
    sensor_xyz: Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    """按 Vectorview -> CTF275 -> sidecar 3D 坐标投影 -> 单位圆兜底解析 2D 传感器位置。"""
    names = [str(name) for name in channel_names]
    try:
        return vectorview_channel_positions(names, layout_path=layout_path)
    except (ImportError, FileNotFoundError, KeyError):
        pass

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
                positions.get(name.split("-")[0].replace(" ", "")) for name in names
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

    angles = np.linspace(0.0, 2.0 * np.pi, num=max(1, len(names)), endpoint=False)
    coords = np.stack([0.5 + 0.5 * np.cos(angles), 0.5 + 0.5 * np.sin(angles)], axis=1)
    return coords[: len(names)].astype(np.float32)


def concat_readonly_tail_from_signals(
    signal_paths: Sequence[Path],
    *,
    required_samples: int,
    channel_count: int,
    subject_id: str,
    max_samples_by_path: Mapping[Path, int] | None = None,
) -> np.ndarray:
    """从同受试者 train 连续记录末尾拼接只读 Pre-Roll 校准缓冲 ``[C, required_samples]``。"""
    if required_samples <= 0:
        raise ValueError(f"Pre-Roll 采样点数必须为正整数：{required_samples}")
    chunks_rev: list[np.ndarray] = []
    remaining = int(required_samples)
    for sig_path in reversed(signal_paths):
        arr = open_derived_signal(str(sig_path))
        if arr.ndim != 2 or int(arr.shape[0]) != int(channel_count):
            raise ValueError(
                f"Pre-Roll 信号维度异常：{sig_path} shape={getattr(arr, 'shape', None)}"
            )
        avail = int(arr.shape[1])
        if max_samples_by_path and sig_path in max_samples_by_path:
            avail = min(avail, max(0, int(max_samples_by_path[sig_path])))
        if avail <= 0:
            continue
        take = min(remaining, avail)
        chunks_rev.append(
            np.array(arr[:, avail - take : avail], dtype=np.float32, copy=True)
        )
        remaining -= take
        if remaining <= 0:
            break
    if remaining > 0:
        collected = int(required_samples) - remaining
        raise ValueError(
            f"受试者 {subject_id} 的 train 划分记录总长度 ({collected}) "
            f"不足以构建所需 Pre-Roll 校准缓冲 ({required_samples} 采样点)。"
        )
    buffer = (
        chunks_rev[0]
        if len(chunks_rev) == 1
        else np.concatenate(list(reversed(chunks_rev)), axis=1).astype(
            np.float32, copy=False
        )
    )
    buffer.setflags(write=False)
    return buffer


@lru_cache(maxsize=8)
def _cached_train_recording_rows(
    events_csv_path: str, mtime_ns: int
) -> tuple[tuple[str, str, str, str], ...]:
    """缓存读取 events.csv 中 train 划分的可训练记录清单 ``(subject, rec_id, run, relpath)``。"""
    del mtime_ns
    header = pd.read_csv(events_csv_path, nrows=0).columns.tolist()
    has_rel = "信号相对路径" in header
    has_run = "运行编号" in header
    cols = ["受试者", "记录编号", "数据划分", "是否可训练"]
    if has_rel:
        cols.append("信号相对路径")
    if has_run:
        cols.append("运行编号")
    df = pd.read_csv(events_csv_path, usecols=cols, keep_default_na=False)
    trainable_mask = df["是否可训练"].map(lambda v: parse_bool_value(v, field="是否可训练"))
    train_df = df[df["数据划分"].eq("train") & trainable_mask]
    if not has_rel:
        train_df = train_df.assign(**{"信号相对路径": ""})
    if not has_run:
        train_df = train_df.assign(**{"运行编号": ""})
    dedup = (
        train_df[["受试者", "记录编号", "运行编号", "信号相对路径"]]
        .drop_duplicates()
        .sort_values(["受试者", "运行编号", "记录编号"], kind="mergesort")
    )
    return tuple(
        (str(s), str(r), str(u), str(p))
        for s, r, u, p in dedup.itertuples(index=False, name=None)
    )


class WordWindowDataset(Dataset):
    """全仓唯一 50 Hz 派生 MEG 词级时间窗数据集。

    支持：
    - 任意 ``window_seconds`` 与 ``baseline_seconds``（含 ``0.0`` 零基线模式）切窗与 ``[-clamp, clamp]`` 截断；
    - 确定性圆周错位与严格因果 Pre-Roll 错位（严禁同记录末尾环绕或跨划分未来泄漏）；
    - PNPL 缓存（``信号相对路径`` + 动态受试者）与旧 10 被试 ``pallier2025`` 缓存双模无缝读取。
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
        preroll_buffers: Mapping[str, np.ndarray] | None = None,
        expected_channel_count: int | None = None,
        expected_channel_names_sha256: str | None = None,
        strict_window_bounds: bool = False,
    ) -> None:
        source = table.copy()
        missing = sorted(set(CORE_EVENT_COLUMNS) - set(source.columns))
        if missing:
            raise ValueError(f"WordWindowDataset 缺少核心事件字段：{missing}")
        source = source[source["是否可训练"].astype(bool)].reset_index(drop=True)
        if source.empty:
            raise ValueError("WordWindowDataset 没有可训练事件。")

        cfg = dict(dataset_config or {})
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

        raw_subjects = cfg.get("subjects")
        if raw_subjects and raw_subjects != "auto":
            configured_subjects = tuple(format_subject_id(s) for s in raw_subjects)
        else:
            configured_subjects = extract_ordered_subjects(self.table)
        observed_subjects = set(self.table["subject_id"].astype(str))
        if not observed_subjects.issubset(set(configured_subjects)):
            raise ValueError("事件表包含未在配置 subjects 中声明的受试者。")

        self.meg_cache_dir = Path(meg_cache_dir)
        self.zero_meg = bool(zero_meg)
        self._strict_window_bounds = bool(strict_window_bounds)
        self.sampling_rate_hz = float(cfg["target_sampling_rate_hz"])
        self.window_seconds = float(cfg["window_seconds"])
        self.window_samples = int(round(self.window_seconds * self.sampling_rate_hz))
        self._baseline_samples = int(
            round(float(cfg["baseline_seconds"]) * self.sampling_rate_hz)
        )
        if self.window_samples <= 0:
            raise ValueError(f"窗口样本数必须为正整数：{self.window_samples}")
        if self._baseline_samples < 0:
            raise ValueError(f"baseline 样本数不能为负数：{self._baseline_samples}")
        self._clamp = float(cfg["clamp"])
        self._window_offset_seconds = float(cfg["window_start_offset_seconds"])

        self.temporal_shift_seconds = float(temporal_shift_seconds)
        self.temporal_shift_samples = int(
            round(self.temporal_shift_seconds * self.sampling_rate_hz)
        )
        resolved_shift_mode = str(
            shift_mode
            if shift_mode is not None
            else cfg.get("shift_mode", "deterministic_circular")
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
        if self.preroll_seconds <= 0.0:
            raise ValueError(f"preroll_seconds 必须为正数：{self.preroll_seconds}")
        if self.shift_mode == "preroll" and self.temporal_shift_samples > 0:
            raise ValueError("Pre-Roll 因果移位模式仅允许严格过去移位 (temporal_shift_seconds <= 0)。")

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

        self._has_relpath_column = "信号相对路径" in self.table.columns
        if not self._has_relpath_column and "运行编号" not in self.table.columns:
            raise ValueError("事件表既缺少 '信号相对路径' 也缺少 '运行编号'，无法定位派生 MEG 信号。")

        self.signal_paths: dict[str, Path] = {}
        self.signal_metadata: dict[str, dict[str, Any]] = {}
        channel_names: tuple[str, ...] | None = None
        sensor_xyz_ref = None
        self._load_signal_metadata(
            expected_channel_count=expected_channel_count,
            expected_channel_names_sha256=expected_channel_names_sha256,
        )
        first_meta = next(iter(self.signal_metadata.values()))
        channel_names = tuple(first_meta["channel_names"])
        sensor_xyz_ref = first_meta.get("sensor_xyz")

        self.channel_names = list(channel_names)
        self.channel_count = len(self.channel_names)
        if expected_channel_names_sha256 is not None:
            if channel_names_sha256(self.channel_names) != expected_channel_names_sha256:
                raise ValueError("Dataset 通道顺序 SHA 不一致。")
        self.channel_positions = resolve_meg_channel_positions(
            self.channel_names,
            layout_path=cfg.get("layout_path"),
            sensor_xyz=sensor_xyz_ref,
        )

        self._recording_ids = tuple(self.table["recording_id"].astype(str).tolist())
        self._words = tuple(self.table["normalized_word"].astype(str).tolist())
        self._event_ids = tuple(self.table["event_id"].astype(str).tolist())
        self._onset_seconds = self.table["onset_seconds"].to_numpy(
            dtype=np.float64, copy=True
        )
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
        self._subject_index_tensors = torch.as_tensor(
            self.subject_indices, dtype=torch.long
        )
        self._sentence_index_tensors = torch.as_tensor(
            self.sentence_indices, dtype=torch.long
        )
        self._onset_second_tensors = torch.as_tensor(
            self._onset_seconds, dtype=torch.float32
        )

        self._preroll_by_recording: dict[str, np.ndarray] = {}
        self._default_preroll_buffer: np.ndarray | None = None
        self._preroll_source_table = preroll_source_table
        self._explicit_preroll_buffers = (
            dict(preroll_buffers) if preroll_buffers is not None else None
        )
        if (
            not self.zero_meg
            and self.shift_mode == "preroll"
            and self.temporal_shift_samples < 0
        ):
            self.ensure_preroll_buffers()

    def _resolve_signal_and_sidecar(
        self, subject: str, run: str = "", relpath: str = ""
    ) -> tuple[Path, Path]:
        if relpath:
            sig_path = self.meg_cache_dir / str(relpath)
            return sig_path, sig_path.with_suffix(".json")
        return pallier_legacy_signal_paths(self.meg_cache_dir, subject, run)

    def _load_signal_metadata(
        self,
        *,
        expected_channel_count: int | None,
        expected_channel_names_sha256: str | None,
    ) -> None:
        channel_names_ref: tuple[str, ...] | None = None
        if self._has_relpath_column:
            iter_rows = [
                (str(rec), "", "", str(rel))
                for rec, rel in self.table[["recording_id", "信号相对路径"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            ]
        else:
            iter_rows = [
                (str(rec), str(subj), str(run), "")
                for rec, subj, run in self.table[
                    ["recording_id", "subject_id", "运行编号"]
                ]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            ]

        for rec_id, subj, run, relpath in iter_rows:
            sig_path, sidecar_path = self._resolve_signal_and_sidecar(
                subj, run, relpath
            )
            if not sig_path.is_file() or not sidecar_path.is_file():
                raise FileNotFoundError(f"派生 MEG 信号或其 sidecar 缺失：{sig_path}")
            meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if "recording_id" in meta and meta.get("recording_id") != rec_id:
                raise ValueError(f"signal sidecar 记录编号不一致：{sidecar_path}")
            if not np.isclose(
                float(meta.get("output_sampling_rate_hz", -1.0)),
                self.sampling_rate_hz,
            ):
                raise ValueError(f"信号采样率与配置不一致：{sidecar_path}")
            if meta.get("dtype") not in {None, "float32"}:
                raise ValueError(f"signal sidecar dtype 非 float32：{sidecar_path}")
            if expected_channel_count is not None and int(
                meta.get("channel_count", -1)
            ) != int(expected_channel_count):
                raise ValueError(f"signal sidecar 通道数合同不一致：{sidecar_path}")
            if (
                expected_channel_names_sha256 is not None
                and meta.get("channel_names_sha256") != expected_channel_names_sha256
            ):
                raise ValueError(f"signal sidecar 通道顺序 SHA 不一致：{sidecar_path}")
            names_tuple = tuple(meta["channel_names"])
            if channel_names_ref is None:
                channel_names_ref = names_tuple
            elif names_tuple != channel_names_ref:
                raise ValueError(f"recording 间通道顺序不一致：{sidecar_path}")
            self.signal_paths[rec_id] = sig_path
            self.signal_metadata[rec_id] = meta

    def ensure_preroll_buffers(
        self,
        shift_seconds: float | None = None,
        *,
        shift_mode: str | None = None,
        preroll_seconds: float | None = None,
    ) -> None:
        """动态切换至 Pre-Roll 模式或扩大过去移位幅度时，按需初始化只读前置校准缓冲。"""
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
            raise ValueError("Pre-Roll 因果移位模式仅允许严格过去移位 (temporal_shift_seconds <= 0)。")

        required_samples = max(
            int(round(self.preroll_seconds * self.sampling_rate_hz)),
            abs(int(self.temporal_shift_samples)) + int(self.window_samples),
        )
        if self._preroll_by_recording and all(
            int(buf.shape[1]) >= required_samples
            for buf in self._preroll_by_recording.values()
        ):
            return
        self._init_preroll_buffers(
            required_samples=required_samples,
            preroll_source_table=self._preroll_source_table,
            preroll_buffers=self._explicit_preroll_buffers,
        )

    def _init_preroll_buffers(
        self,
        *,
        required_samples: int,
        preroll_source_table: pd.DataFrame | None,
        preroll_buffers: Mapping[str, np.ndarray] | None,
    ) -> None:
        observed_pairs = list(
            self.table[["subject_id", "recording_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if preroll_buffers is not None:
            for subj_raw, rec_raw in observed_pairs:
                subj, rec_id = str(subj_raw), str(rec_raw)
                buf = preroll_buffers.get(rec_id, preroll_buffers.get(subj))
                if buf is None:
                    raise ValueError(f"preroll_buffers 缺少受试者 {subj} (recording={rec_id})。")
                arr = np.asarray(buf, dtype=np.float32)
                if arr.ndim != 2 or int(arr.shape[0]) != int(self.channel_count):
                    raise ValueError(f"preroll_buffers[{subj}] 形状无效：{arr.shape}")
                if int(arr.shape[1]) < required_samples:
                    raise ValueError(
                        f"preroll_buffers[{subj}] 长度 ({arr.shape[1]}) 小于所需采样点 ({required_samples})。"
                    )
                sliced = np.array(arr[:, -required_samples:], dtype=np.float32, copy=True)
                sliced.setflags(write=False)
                self._preroll_by_recording[rec_id] = sliced
            return

        train_rows: list[tuple[str, str, str, str]] = []
        shared_rec_max_train_end: dict[str, int] = {}
        if preroll_source_table is not None:
            src_df = preroll_source_table
            split_col = "数据划分" if "数据划分" in src_df.columns else "split"
            subj_col = "受试者" if "受试者" in src_df.columns else "subject_id"
            rec_col = "记录编号" if "记录编号" in src_df.columns else "recording_id"
            run_col = "运行编号" if "运行编号" in src_df.columns else ("run" if "run" in src_df.columns else None)
            rel_col = "信号相对路径" if "信号相对路径" in src_df.columns else None
            onset_col = "开始时间" if "开始时间" in src_df.columns else "onset_seconds"
            train_col = "是否可训练" if "是否可训练" in src_df.columns else "is_trainable"
            trainable_mask = (
                src_df[train_col].astype(bool)
                if train_col in src_df.columns
                else pd.Series(True, index=src_df.index)
            )
            mask = src_df[split_col].astype(str).eq("train") & trainable_mask
            work = src_df.loc[mask, [subj_col, rec_col]].copy()
            work["_run"] = src_df.loc[mask, run_col].astype(str) if run_col else ""
            work["_rel"] = src_df.loc[mask, rel_col].astype(str) if rel_col else ""
            dedup = work.drop_duplicates().sort_values(
                [subj_col, "_run", rec_col], kind="mergesort"
            )
            train_rows = [
                (str(s), str(r), str(u), str(p))
                for s, r, u, p in dedup.itertuples(index=False, name=None)
            ]
            # 同记录内混合 train / non-train 时，仅截取严格早于非训练事件的 train 片段，杜绝未来泄漏
            non_train_mask = (~src_df[split_col].astype(str).eq("train")) & trainable_mask
            shared_recs = set(src_df.loc[mask, rec_col].astype(str)) & set(
                src_df.loc[non_train_mask, rec_col].astype(str)
            )
            if shared_recs and onset_col in src_df.columns:
                for s_rec in shared_recs:
                    rec_rows = src_df[src_df[rec_col].astype(str).eq(s_rec) & trainable_mask]
                    s_subj = str(rec_rows[subj_col].iloc[0])
                    s_run = str(rec_rows[run_col].iloc[0]) if run_col else ""
                    s_rel = str(rec_rows[rel_col].iloc[0]) if rel_col else ""
                    _, sidecar_p = self._resolve_signal_and_sidecar(s_subj, s_run, s_rel)
                    sidecar = (
                        json.loads(sidecar_p.read_text(encoding="utf-8"))
                        if sidecar_p.is_file()
                        else self.signal_metadata.get(s_rec, {})
                    )
                    first_samp = int(sidecar.get("source_first_samp", 0))
                    src_rate = float(sidecar.get("source_sampling_rate_hz", SOURCE_SAMPLING_RATE_HZ))
                    out_rate = float(sidecar.get("output_sampling_rate_hz", self.sampling_rate_hz))
                    tr_onsets = rec_rows.loc[
                        rec_rows[split_col].astype(str).eq("train"), onset_col
                    ].astype(float)
                    ntr_onsets = rec_rows.loc[
                        ~rec_rows[split_col].astype(str).eq("train"), onset_col
                    ].astype(float)
                    max_tr_stop = max(
                        recording_onset_to_sample(
                            float(ov) + self._window_offset_seconds,
                            source_first_samp=first_samp,
                            source_sampling_rate_hz=src_rate,
                            output_sampling_rate_hz=out_rate,
                        )
                        + int(self.window_samples)
                        for ov in tr_onsets
                    )
                    min_ntr_start = min(
                        recording_onset_to_sample(
                            float(ov) + self._window_offset_seconds,
                            source_first_samp=first_samp,
                            source_sampling_rate_hz=src_rate,
                            output_sampling_rate_hz=out_rate,
                        )
                        for ov in ntr_onsets
                    )
                    shared_rec_max_train_end[s_rec] = max(0, min(max_tr_stop, min_ntr_start))
        else:
            candidate_csv = self.dataset_config.get("event_table_path")
            csv_path = (
                Path(candidate_csv)
                if candidate_csv
                else (self.meg_cache_dir.parent.parent / "events" / "events.csv")
            )
            if csv_path.is_file():
                train_rows = list(
                    _cached_train_recording_rows(
                        str(csv_path.resolve()), csv_path.stat().st_mtime_ns
                    )
                )
            elif "split" in self.table.columns:
                mask = self.table["split"].astype(str).eq("train")
                work = self.table.loc[mask, ["subject_id", "recording_id"]].copy()
                work["_run"] = (
                    self.table.loc[mask, "运行编号"].astype(str)
                    if "运行编号" in self.table.columns
                    else ""
                )
                work["_rel"] = (
                    self.table.loc[mask, "信号相对路径"].astype(str)
                    if "信号相对路径" in self.table.columns
                    else ""
                )
                dedup = work.drop_duplicates().sort_values(
                    ["subject_id", "_run", "recording_id"], kind="mergesort"
                )
                train_rows = [
                    (str(s), str(r), str(u), str(p))
                    for s, r, u, p in dedup.itertuples(index=False, name=None)
                ]

        by_subject: dict[str, list[tuple[str, Path]]] = {}
        max_samples_by_path: dict[Path, int] = {}
        for subj, rec_id, run, relpath in train_rows:
            sig_path, _ = self._resolve_signal_and_sidecar(subj, run, relpath)
            if sig_path.is_file():
                by_subject.setdefault(subj, []).append((rec_id, sig_path))
                if rec_id in shared_rec_max_train_end:
                    max_samples_by_path[sig_path] = shared_rec_max_train_end[rec_id]

        primary_cache: dict[str, tuple[str, np.ndarray]] = {}
        alt_cache: dict[str, np.ndarray] = {}
        for subj_raw, rec_raw in observed_pairs:
            subj, rec_id = str(subj_raw), str(rec_raw)
            records = by_subject.get(subj, [])
            if not records:
                raise ValueError(f"Pre-Roll 模式未找到受试者 {subj} 的 train 划分校准信号。")
            if subj not in primary_cache:
                last_rec_id = records[-1][0]
                primary_buf = concat_readonly_tail_from_signals(
                    [p for _, p in records],
                    required_samples=required_samples,
                    channel_count=self.channel_count,
                    subject_id=subj,
                    max_samples_by_path=max_samples_by_path or None,
                )
                primary_cache[subj] = (last_rec_id, primary_buf)
            last_rec_id, primary_buf = primary_cache[subj]
            if rec_id == last_rec_id and rec_id not in shared_rec_max_train_end:
                if len(records) >= 2:
                    if subj not in alt_cache:
                        alt_cache[subj] = concat_readonly_tail_from_signals(
                            [p for _, p in records[:-1]],
                            required_samples=required_samples,
                            channel_count=self.channel_count,
                            subject_id=subj,
                            max_samples_by_path=max_samples_by_path or None,
                        )
                    self._preroll_by_recording[rec_id] = alt_cache[subj]
                else:
                    raise ValueError(
                        f"受试者 {subj} 仅有一条未截断 train 记录 ({rec_id})，"
                        "Pre-Roll 模式禁止使用同一记录末尾环绕充当校准缓冲。"
                    )
            else:
                self._preroll_by_recording[rec_id] = primary_buf

    def _get_preroll_for_index(self, idx: int) -> np.ndarray:
        buf = self._preroll_by_recording.get(self._recording_ids[idx])
        if buf is not None:
            return buf
        if self._default_preroll_buffer is not None:
            return self._default_preroll_buffer
        raise RuntimeError(f"Pre-Roll 校准缓冲未初始化 (index={idx})。")

    def __len__(self) -> int:
        return len(self.table)

    def read_meg(self, index: int) -> np.ndarray:
        idx = int(index)
        if self.zero_meg:
            return np.zeros((self.channel_count, self.window_samples), dtype=np.float32)
        recording = open_derived_signal(self._signal_path_strs[idx])
        if recording.ndim != 2 or recording.shape[1] < self.window_samples:
            raise ValueError("词窗口超出连续 MEG 信号长度。")
        start = int(self._window_starts[idx])
        stop = start + self.window_samples
        if start < 0 or stop > recording.shape[1]:
            if (
                self._strict_window_bounds
                and self.window_samples == 50
                and self._window_offset_seconds == 0.0
            ):
                raise ValueError("词窗口超出 canonical continuous signal。")
            start = max(0, min(start, int(recording.shape[1]) - self.window_samples))
            stop = start + self.window_samples

        effective_shift_mode = getattr(self, "shift_mode", "deterministic_circular")
        if self.temporal_shift_samples == 0:
            window = recording[:, start:stop].astype(np.float32, copy=True)
        elif effective_shift_mode == "preroll":
            if self.temporal_shift_samples > 0:
                raise ValueError("Pre-Roll 因果移位模式仅允许严格过去移位 (temporal_shift_seconds <= 0)。")
            raw_start = start + int(self.temporal_shift_samples)
            win_len = int(self.window_samples)
            if raw_start >= 0:
                window = recording[:, raw_start : raw_start + win_len].astype(
                    np.float32, copy=True
                )
            else:
                preroll = self._get_preroll_for_index(idx)
                buf_len = int(preroll.shape[1])
                buf_start = buf_len + raw_start
                if buf_start < 0:
                    raise IndexError(
                        f"Pre-Roll 缓冲区长度不足：raw_start={raw_start}, buf_len={buf_len}"
                    )
                if raw_start + win_len <= 0:
                    window = preroll[:, buf_start : buf_start + win_len].astype(
                        np.float32, copy=True
                    )
                else:
                    left_len = -raw_start
                    right_len = win_len - left_len
                    window = np.empty((self.channel_count, win_len), dtype=np.float32)
                    window[:, :left_len] = preroll[:, buf_start:buf_len]
                    window[:, left_len:] = recording[:, :right_len]
        else:
            valid_span = int(recording.shape[1]) - self.window_samples + 1
            if valid_span <= 2 * self.window_samples:
                raise ValueError("连续信号长度不足以执行非重叠时间错位对照。")
            shifted_start = (start + self.temporal_shift_samples) % valid_span
            if abs(shifted_start - start) < self.window_samples:
                step = (
                    self.window_samples
                    if self.temporal_shift_samples > 0
                    else -self.window_samples
                )
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


DerivedMegWordDataset = WordWindowDataset
