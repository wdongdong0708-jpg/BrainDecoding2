"""Broderick 2018 自然语音 128 导联 EEG 数据集适配器与词级样本构建器。"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from braindecoding.config import PROJECT_ROOT, cache_root
from braindecoding.data import text_embeddings as text_data
from braindecoding.data.derived_cache import (
    CORE_EVENT_COLUMNS,
    canonical_event_table,
    sha256_file,
    stable_sha256,
)
from braindecoding.data.event_table import (
    add_core_event_columns,
    add_record_order,
    align_text_sequences,
    validate_event_table,
)

DATASET_ID = "broderick2018"
RAW_DATASET_NAME = "Broderick2018-NaturalSpeech-EEG"
DRYAD_DOI = "10.5061/dryad.070jc"
CORE_SCOPE = "broderick2018_all_subjects_word_v1"
RUN_SPLIT_SALT = "broderick2018-main-run-split-v1"
SUBJECTS = tuple(f"sub-{index:02d}" for index in range(1, 20))
RUNS = tuple(f"run-{index:02d}" for index in range(1, 21))
EXPECTED_RUN_COUNT = 20
TRAIN_RUN_COUNT = 16
VAL_RUN_COUNT = 2
TEST_RUN_COUNT = 2

EEG_CHANNEL_COUNT = 128
MASTOID_CHANNEL_COUNT = 2
BIOSEMI128_CHANNEL_NAMES = tuple(
    f"{letter}{index}" for letter in "ABCD" for index in range(1, 33)
)
CHANNEL_NAMES_SHA256 = hashlib.sha256(
    "\n".join(BIOSEMI128_CHANNEL_NAMES).encode("utf-8")
).hexdigest()

SOURCE_SAMPLING_RATE_HZ = 128.0
TARGET_SAMPLING_RATE_HZ = 50.0
FILTER_LOW_HZ = 0.1
FILTER_HIGH_HZ = 40.0

WINDOW_CONTRACT_VERSION = "word_1s_support_3s_baseline_0p5_v1"
WINDOW_START_OFFSET_SECONDS = 0.0
WINDOW_SECONDS = 1.0
ELIGIBILITY_WINDOW_SECONDS = 3.0
BASELINE_SECONDS = 0.5
CLAMP = 5.0

DEFAULT_MAX_CONTEXT_WORDS = 128
DEFAULT_SILENCE_GAP_SECONDS = 0.45
DEFAULT_MIN_MATCHED_WORDS = 450

VALID_TEMPORAL_SHIFT_MODES = ("deterministic_circular", "preroll")
DEFAULT_PREROLL_SECONDS = 45.0
RUNTIME_CONTEXT_GROUPING = "recording_id_plus_canonical_context_v1"
RUNTIME_CONTEXT_COLUMN = "runtime_context_uid"
ENGLISH_REFERENCE_TOKENS = (
    "the",
    "sea",
    "old",
    "man",
    "don't",
    "fish",
    "boat",
    "line",
)

PERSISTED_COLUMNS = (
    *CORE_EVENT_COLUMNS,
    "运行编号",
    "时长",
    "窗口开始",
    "窗口结束",
    "对齐词索引",
    "原始句子编号",
    "是否内容词",
)

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])(?:["\'\)\]]+)?\s+')
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")


def _portable_manifest_path(path: str | Path) -> str:
    """将缓存路径规范为相对于仓库根目录或 cache_root 的可移植 POSIX 路径。"""
    resolved = Path(path).resolve()
    for anchor in (PROJECT_ROOT.resolve(), cache_root().resolve()):
        try:
            return resolved.relative_to(anchor).as_posix()
        except ValueError:
            continue
    return Path(path).as_posix()


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json_atomic(path: str | Path, payload: dict) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _write_array_atomic(path: str | Path, array: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp.npy")
    np.save(temporary, np.ascontiguousarray(array))
    os.replace(temporary, target)
    return target


def _with_manifest_sha256(payload: dict) -> dict:
    result = dict(payload)
    result["manifest_sha256"] = stable_sha256(result)
    return result


def _validate_self_hash(payload: dict, *, path: str | Path) -> dict:
    body = dict(payload)
    recorded = body.pop("manifest_sha256", None) or body.pop("artifact_sha256", None)
    if stable_sha256(body) != recorded:
        raise ValueError(f"文件自摘要不一致：{path}")
    return payload


def _bool_value(value, *, field: str = "value") -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"无法解析布尔字段 {field}: {value!r}")


def normalize_english_word(word) -> str:
    """规范化英文词型：仅保留字母数字、连字符 '-' 与撇号 '\'' 并转小写。"""
    if pd.isna(word):
        return ""
    return text_data.normalize_word(str(word))


def channel_names_sha256(channel_names) -> str:
    """按冻结的换行拼接规则计算 128 通道顺序 SHA-256。"""
    payload = "\n".join(str(name) for name in channel_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@lru_cache(maxsize=4)
def _cached_biosemi128_positions(
    channel_names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    """从 MNE standard montage 提取 BioSemi-128 三维头皮坐标。"""
    try:
        import mne
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("读取 BioSemi-128 电极坐标需要安装 mne。") from error

    montage = mne.channels.make_standard_montage("biosemi128")
    positions_map = montage.get_positions()["ch_pos"]
    missing = [name for name in channel_names if name not in positions_map]
    if missing:
        raise ValueError(f"biosemi128 montage 缺少通道：{missing[:5]}")
    coords = np.asarray(
        [positions_map[name] for name in channel_names], dtype=np.float32
    )
    if coords.shape != (len(channel_names), 3) or not np.isfinite(coords).all():
        raise ValueError(f"biosemi128 坐标矩阵无效：{coords.shape}")
    coords.setflags(write=False)
    return (coords,)


def biosemi128_channel_positions(
    channel_names=BIOSEMI128_CHANNEL_NAMES,
) -> np.ndarray:
    """返回 BioSemi-128 通道的三维坐标 [128, 3] (float32)。"""
    names = tuple(str(name) for name in channel_names)
    if len(names) != EEG_CHANNEL_COUNT:
        raise ValueError(
            f"Broderick2018 仅支持 {EEG_CHANNEL_COUNT} 个头皮 EEG 通道，当前为 {len(names)}。"
        )
    return np.array(_cached_biosemi128_positions(names)[0], dtype=np.float32, copy=True)


def subject_to_raw_name(subject: str) -> str:
    """sub-01 -> Subject1."""
    text = str(subject).strip()
    if text.startswith("sub-"):
        return f"Subject{int(text.split('-', 1)[1])}"
    if text.startswith("Subject"):
        return text
    raise ValueError(f"无法识别 Broderick2018 subject：{subject!r}")


def run_to_raw_index(run: str) -> int:
    """run-01 -> 1."""
    text = str(run).strip()
    if text.startswith("run-"):
        return int(text.split("-", 1)[1])
    if text.startswith("Run"):
        return int(text[3:])
    raise ValueError(f"无法识别 Broderick2018 run：{run!r}")


def _resolve_existing_candidate(candidates: tuple[Path, ...]) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def recording_paths(raw_root: str | Path, subject: str, run: str) -> dict[str, Path | None]:
    """解析 Broderick2018 单个 (subject, run) 的原始 .mat、align JSON 与原文 txt 路径。"""
    root = Path(raw_root)
    raw_subject = subject_to_raw_name(subject)
    raw_run = run_to_raw_index(run)
    mat_candidates = (
        root / "Natural Speech" / "EEG" / raw_subject / f"{raw_subject}_Run{raw_run}.mat",
        root / "EEG" / raw_subject / f"{raw_subject}_Run{raw_run}.mat",
        root / raw_subject / f"{raw_subject}_Run{raw_run}.mat",
    )
    align_candidates = (
        root / "private" / f"align{raw_run}.json",
        root / f"align{raw_run}.json",
        root / "Natural Speech" / "private" / f"align{raw_run}.json",
    )
    text_candidates = (
        root / "private" / f"oldman_run{raw_run}.txt",
        root / f"oldman_run{raw_run}.txt",
        root / "Natural Speech" / "private" / f"oldman_run{raw_run}.txt",
    )
    resolved_text = next((path for path in text_candidates if path.is_file()), None)
    return {
        "eeg_mat": _resolve_existing_candidate(mat_candidates),
        "align_json": _resolve_existing_candidate(align_candidates),
        "text_txt": resolved_text,
    }


def discover_subjects(raw_root: str | Path) -> list[str]:
    """按 canonical 顺序发现 raw_root 中存在的 Broderick2018 被试。"""
    root = Path(raw_root)
    found = []
    for subject in SUBJECTS:
        raw_subject = subject_to_raw_name(subject)
        if any(
            path.is_dir()
            for path in (
                root / "Natural Speech" / "EEG" / raw_subject,
                root / "EEG" / raw_subject,
                root / raw_subject,
            )
        ):
            found.append(subject)
    return found


def deterministic_run_split(
    runs=RUNS, *, salt: str = RUN_SPLIT_SALT
) -> tuple[dict[str, str], list[dict]]:
    """根据 run 标识加盐 SHA-256 排序生成 16 Train / 2 Val / 2 Test 确定性划分。"""
    ranked = sorted(
        (
            hashlib.sha256(f"{salt}|{run}".encode("utf-8")).hexdigest(),
            str(run),
        )
        for run in runs
    )
    if len(ranked) != EXPECTED_RUN_COUNT or len({run for _, run in ranked}) != EXPECTED_RUN_COUNT:
        raise ValueError("Broderick2018 主 split 需要 20 个唯一 run。")
    split_by_run = {
        run: (
            "train"
            if rank < TRAIN_RUN_COUNT
            else "val"
            if rank < TRAIN_RUN_COUNT + VAL_RUN_COUNT
            else "test"
        )
        for rank, (_, run) in enumerate(ranked)
    }
    ranking = [
        {"rank": rank, "run": run, "sha256": digest, "split": split_by_run[run]}
        for rank, (digest, run) in enumerate(ranked)
    ]
    return split_by_run, ranking


def build_run_split_manifest(*, salt: str = RUN_SPLIT_SALT) -> dict:
    """构建带自校验摘要的 Broderick2018 Run 级冻结划分清单。"""
    split_by_run, ranking = deterministic_run_split(RUNS, salt=salt)
    assignments = {
        split: [run for run in RUNS if split_by_run[run] == split]
        for split in ("train", "val", "test")
    }
    return _with_manifest_sha256(
        {
            "asset_type": "run_split",
            "created_from_test": False,
            "dataset": DATASET_ID,
            "hash_contract": "sha256(canonical_json_without_manifest_sha256)",
            "scope": CORE_SCOPE,
            "status": "frozen",
            "strategy": {
                "type": "deterministic_salted_sha256_run_sort",
                "salt": str(salt),
                "counts": {
                    "train": TRAIN_RUN_COUNT,
                    "val": VAL_RUN_COUNT,
                    "test": TEST_RUN_COUNT,
                },
            },
            "assignments": assignments,
            "hash_ranking": ranking,
        }
    )


def write_run_split_manifest(
    output_path: str | Path | None = None, *, salt: str = RUN_SPLIT_SALT
) -> Path:
    """原子写入 data/manifests/broderick2018/run_split.json 并执行自校验。"""
    target = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "data/manifests/broderick2018/run_split.json"
    )
    payload = build_run_split_manifest(salt=salt)
    _write_json_atomic(target, payload)
    load_run_split(target)
    return target


def load_run_split(path: str | Path) -> dict[str, str]:
    """读取并严格校验冻结的 Broderick2018 Run 级划分清单。"""
    payload = _validate_self_hash(_read_json(path), path=path)
    if payload.get("dataset") != DATASET_ID:
        raise ValueError(f"split manifest 数据集不是 {DATASET_ID}：{path}")
    assignments = payload.get("assignments", {})
    split_by_run = {
        run: split
        for split in ("train", "val", "test")
        for run in assignments.get(split, ())
    }
    if set(split_by_run) != set(RUNS):
        raise ValueError("Broderick2018 split 必须恰好覆盖 run-01 至 run-20。")
    counts = [len(assignments.get(split, ())) for split in ("train", "val", "test")]
    if counts != [TRAIN_RUN_COUNT, VAL_RUN_COUNT, TEST_RUN_COUNT]:
        raise ValueError(
            "Broderick2018 主 split 必须是 16 train / 2 val / 2 test。"
        )
    expected_split, expected_ranking = deterministic_run_split(
        RUNS, salt=payload["strategy"]["salt"]
    )
    if split_by_run != expected_split or payload.get("hash_ranking") != expected_ranking:
        raise ValueError("Broderick2018 split 与冻结 deterministic 算法不一致。")
    return split_by_run


def word_window_contract(dataset_config: dict | None = None) -> dict:
    """返回 Broderick2018 词级 EEG 窗口与基线扣除合同。"""
    config = dataset_config or {}
    if isinstance(config.get("dataset"), dict) and "window_seconds" not in config:
        config = config["dataset"]
    sampling_rate = float(config.get("target_sampling_rate_hz", TARGET_SAMPLING_RATE_HZ))
    window_seconds = float(config.get("window_seconds", WINDOW_SECONDS))
    baseline_seconds = float(config.get("baseline_seconds", BASELINE_SECONDS))
    contract = {
        "version": str(config.get("window_contract_version", WINDOW_CONTRACT_VERSION)),
        "window_start_offset_seconds": float(
            config.get("window_start_offset_seconds", WINDOW_START_OFFSET_SECONDS)
        ),
        "window_seconds": window_seconds,
        "eligibility_window_seconds": float(
            config.get("eligibility_window_seconds", ELIGIBILITY_WINDOW_SECONDS)
        ),
        "baseline_seconds": baseline_seconds,
        "clamp": float(config.get("clamp", CLAMP)),
        "sampling_rate_hz": sampling_rate,
        "window_samples": int(round(window_seconds * sampling_rate)),
        "baseline_samples": int(round(baseline_seconds * sampling_rate)),
        "continuous_signal_includes_baseline": False,
        "continuous_signal_includes_clamp": False,
    }
    if contract["window_samples"] <= 0:
        raise ValueError(
            f"Broderick2018 窗口样本数必须为正整数，当前为 {contract['window_samples']}。"
        )
    if contract["baseline_samples"] < 0:
        raise ValueError("Broderick2018 baseline 样本数不能为负数。")
    if contract["eligibility_window_seconds"] != 3.0:
        raise ValueError("Broderick2018 正式事件资格窗口必须为 3 秒。")
    return contract


def recording_onset_to_sample(
    onset_seconds: float,
    *,
    source_first_samp: int = 0,
    source_sampling_rate_hz: float = SOURCE_SAMPLING_RATE_HZ,
    output_sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ,
) -> int:
    """将 EEG 记录秒级 onset 转换为 canonical 连续信号的零起点样本下标。"""
    recording_start = float(source_first_samp) / float(source_sampling_rate_hz)
    relative_seconds = float(onset_seconds) - recording_start
    return int(np.round(relative_seconds * float(output_sampling_rate_hz)))


def recording_window_bounds(
    onset_seconds: float,
    *,
    duration_seconds: float,
    source_first_samp: int = 0,
    source_sampling_rate_hz: float = SOURCE_SAMPLING_RATE_HZ,
    output_sampling_rate_hz: float = TARGET_SAMPLING_RATE_HZ,
) -> tuple[int, int]:
    """返回左闭右开的 canonical 窗口样本范围。"""
    start = recording_onset_to_sample(
        onset_seconds,
        source_first_samp=source_first_samp,
        source_sampling_rate_hz=source_sampling_rate_hz,
        output_sampling_rate_hz=output_sampling_rate_hz,
    )
    length = int(np.round(float(duration_seconds) * float(output_sampling_rate_hz)))
    return start, start + length


def extract_word_window(signal, onset_seconds, signal_metadata, dataset_config=None):
    """从连续 EEG 信号切取 1 秒词窗，执行前 0.5 秒基线扣除与 ±5 截断。"""
    contract = word_window_contract(dataset_config)
    start = recording_onset_to_sample(
        float(onset_seconds) + contract["window_start_offset_seconds"],
        source_first_samp=int(signal_metadata.get("source_first_samp", 0)),
        source_sampling_rate_hz=float(
            signal_metadata.get("source_sampling_rate_hz", SOURCE_SAMPLING_RATE_HZ)
        ),
        output_sampling_rate_hz=float(
            signal_metadata.get("output_sampling_rate_hz", TARGET_SAMPLING_RATE_HZ)
        ),
    )
    stop = start + contract["window_samples"]
    continuous = np.asarray(signal)
    if continuous.ndim != 2 or start < 0 or stop > continuous.shape[1]:
        raise ValueError("Broderick2018 词窗口超出 canonical continuous signal 范围。")
    window = continuous[:, start:stop].astype(np.float32, copy=True)
    effective_baseline = min(int(contract["baseline_samples"]), int(window.shape[1]))
    if effective_baseline > 0:
        baseline = window[:, :effective_baseline].mean(axis=1, keepdims=True)
        window -= baseline
    np.clip(window, -contract["clamp"], contract["clamp"], out=window)
    return window


def _inspect_mat_metadata(mat_path: Path) -> tuple[float, int]:
    """读取 .mat 文件中的 sampling rate (fs) 与时间采样点总数 T。"""
    import scipy.io

    try:
        entries = {name: shape for name, shape, _ in scipy.io.whosmat(str(mat_path))}
        if "eegData" not in entries or "fs" not in entries:
            raise ValueError(f"Broderick2018 .mat 缺少 eegData 或 fs：{mat_path}")
        mat = scipy.io.loadmat(str(mat_path), variable_names=["fs"])
        fs_hz = float(np.asarray(mat["fs"]).squeeze())
        eeg_shape = entries["eegData"]
        if len(eeg_shape) != 2 or int(eeg_shape[1]) != EEG_CHANNEL_COUNT:
            raise ValueError(
                f"Broderick2018 eegData 形状应为 [T, 128]，实际为 {eeg_shape} ({mat_path})"
            )
        return fs_hz, int(eeg_shape[0])
    except NotImplementedError:
        import h5py

        with h5py.File(str(mat_path), "r") as handle:
            fs_hz = float(np.asarray(handle["fs"]).squeeze())
            shape = tuple(handle["eegData"].shape)
            if len(shape) != 2:
                raise ValueError(f"Broderick2018 HDF5 eegData 维度无效：{shape}")
            sample_count = shape[1] if shape[0] == EEG_CHANNEL_COUNT else shape[0]
            return fs_hz, int(sample_count)


def _load_mat_eeg_and_mastoids(
    mat_path: Path, *, load_mastoids: bool = False
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """加载 Dryad .mat 中的 eegData [T, 128]、可选 mastoids [T, 2] 与采样率 fs。"""
    import scipy.io

    variables = ["eegData", "fs"] + (["mastoids"] if load_mastoids else [])
    try:
        mat = scipy.io.loadmat(str(mat_path), variable_names=variables)
        fs_hz = float(np.asarray(mat["fs"]).squeeze())
        eeg_data = np.asarray(mat["eegData"], dtype=np.float64)
        mastoids = (
            np.asarray(mat["mastoids"], dtype=np.float64)
            if load_mastoids and "mastoids" in mat
            else None
        )
    except NotImplementedError:
        import h5py

        with h5py.File(str(mat_path), "r") as handle:
            fs_hz = float(np.asarray(handle["fs"]).squeeze())
            raw_eeg = np.asarray(handle["eegData"], dtype=np.float64)
            eeg_data = raw_eeg.T if raw_eeg.shape[0] == EEG_CHANNEL_COUNT else raw_eeg
            mastoids = None
            if load_mastoids and "mastoids" in handle:
                raw_m = np.asarray(handle["mastoids"], dtype=np.float64)
                mastoids = raw_m.T if raw_m.shape[0] == MASTOID_CHANNEL_COUNT else raw_m

    if eeg_data.ndim != 2 or eeg_data.shape[1] != EEG_CHANNEL_COUNT:
        raise ValueError(
            f"Broderick2018 eegData 形状应为 [T, 128]，实际为 {eeg_data.shape} ({mat_path})"
        )
    if load_mastoids:
        if mastoids is None:
            raise ValueError(f"启用 mastoid_rereference=true 但 .mat 缺少 mastoids：{mat_path}")
        if mastoids.shape[0] != eeg_data.shape[0] or mastoids.shape[1] < 1:
            raise ValueError(
                f"Broderick2018 mastoids 形状与 eegData 不匹配：{mastoids.shape} vs {eeg_data.shape}"
            )
    return eeg_data, mastoids, fs_hz


def _parse_gentle_align_words(align_path: Path) -> list[dict]:
    """解析 Gentle align{r}.json，仅保留 case=='success' 且 alignedWord!='<unk>' 词项。"""
    payload = _read_json(align_path)
    raw_words = payload.get("words", []) if isinstance(payload, dict) else list(payload)
    cleaned: list[dict] = []
    for raw_idx, item in enumerate(raw_words):
        if item.get("case") != "success":
            continue
        aligned_word = str(item.get("alignedWord", "")).strip()
        if not aligned_word or aligned_word == "<unk>":
            continue
        word_text = str(item.get("word") or aligned_word).strip()
        normalized = normalize_english_word(word_text)
        if not normalized:
            continue
        onset = float(item["start"])
        offset = float(item["end"])
        cleaned.append(
            {
                "raw_align_index": int(raw_idx),
                "word": word_text,
                "normalized_word": normalized,
                "onset_seconds": onset,
                "offset_seconds": offset,
                "duration_seconds": max(0.0, offset - onset),
            }
        )
    if not cleaned:
        raise ValueError(f"Broderick2018 align JSON 未包含任何有效对齐词：{align_path}")
    return cleaned


def _split_text_to_sentences(text: str) -> list[list[str]]:
    """按英文句末标点切句并提取每句中的标准化词序列。"""
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if not collapsed:
        return []
    sentences: list[list[str]] = []
    for chunk in _SENTENCE_SPLIT_RE.split(collapsed):
        tokens = [
            normalize_english_word(tok)
            for tok in _WORD_TOKEN_RE.findall(chunk)
        ]
        tokens = [tok for tok in tokens if tok]
        if tokens:
            sentences.append(tokens)
    return sentences


def _align_sentence_ids_to_events(
    event_words: list[str],
    sentences: list[list[str]],
    *,
    min_matched_words: int = DEFAULT_MIN_MATCHED_WORDS,
    source_label: str = "",
) -> list[int]:
    """把原文句子编号对齐映射到 Gentle 事件词序列上，并执行最小匹配词数校验。"""
    text_tokens: list[str] = []
    text_sent_ids: list[int] = []
    for sent_id, words in enumerate(sentences):
        for word in words:
            text_tokens.append(word)
            text_sent_ids.append(sent_id)

    alignment = align_text_sequences(text_tokens, event_words)
    if int(alignment["matches"]) < int(min_matched_words):
        raise ValueError(
            f"Broderick2018 原文与 Gentle 对齐匹配词数不足 ({source_label}): "
            f"{alignment['matches']} < {min_matched_words}"
        )

    # 重建带操作的 DP 回溯路径以便精确把句号映射到 event_words 每一项。
    left_n, right_n = len(text_tokens), len(event_words)
    dp = [list(range(right_n + 1))]
    for i, lw in enumerate(text_tokens, start=1):
        prev = dp[-1]
        row = [i] + [0] * right_n
        for j, rw in enumerate(event_words, start=1):
            row[j] = min(
                prev[j] + 1,
                row[j - 1] + 1,
                prev[j - 1] + int(lw != rw),
            )
        dp.append(row)

    mapped: list[int | None] = [None] * right_n
    i, j = left_n, right_n
    while i > 0 or j > 0:
        cur = dp[i][j]
        if i > 0 and j > 0 and text_tokens[i - 1] == event_words[j - 1] and cur == dp[i - 1][j - 1]:
            mapped[j - 1] = text_sent_ids[i - 1]
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and cur == dp[i - 1][j - 1] + 1:
            mapped[j - 1] = text_sent_ids[i - 1]
            i -= 1
            j -= 1
        elif i > 0 and cur == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1

    last_seen = 0
    for idx, value in enumerate(mapped):
        if value is not None:
            last_seen = int(value)
        else:
            mapped[idx] = last_seen
    return [int(v) for v in mapped]


def _chunk_sentence_ids(
    raw_sentence_ids: list[int],
    *,
    run: str,
    max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS,
) -> list[str]:
    """按句子边界划分上下文，并把超长句切分为不超过 max_context_words 的块。"""
    limit = int(max_context_words)
    if limit <= 0:
        raise ValueError("Broderick2018 max_context_words 必须为正整数。")
    context_ids: list[str] = []
    prev_sent = None
    sent_counter = -1
    chunk_idx = 0
    count_in_chunk = 0
    for sent_id in raw_sentence_ids:
        if prev_sent is None or sent_id != prev_sent:
            sent_counter += 1
            chunk_idx = 0
            count_in_chunk = 0
            prev_sent = sent_id
        elif count_in_chunk >= limit:
            chunk_idx += 1
            count_in_chunk = 0
        count_in_chunk += 1
        suffix = f"-c{chunk_idx:02d}" if chunk_idx > 0 else ""
        context_ids.append(f"{DATASET_ID}|{run}|sequence-{sent_counter:04d}{suffix}")
    return context_ids


def _fallback_silence_sentence_ids(
    events: list[dict], *, gap_seconds: float = DEFAULT_SILENCE_GAP_SECONDS
) -> list[int]:
    """在缺失 oldman_run{r}.txt 的极端回退情形下，按词间静音间隔切句。"""
    sent_ids: list[int] = []
    current_sent = 0
    prev_offset = None
    for event in events:
        if prev_offset is not None and (float(event["onset_seconds"]) - prev_offset) >= float(
            gap_seconds
        ):
            current_sent += 1
        sent_ids.append(current_sent)
        prev_offset = float(event["offset_seconds"])
    return sent_ids


def _run_sequence_ids(
    align_path: Path,
    text_path: Path | None,
    *,
    run: str,
    max_context_words: int,
    silence_gap_seconds: float,
    min_matched_words: int,
) -> tuple[list[dict], list[int], list[str], str]:
    events = _parse_gentle_align_words(align_path)
    event_words = [item["normalized_word"] for item in events]
    if text_path is not None and text_path.is_file():
        text_content = text_path.read_text(encoding="utf-8", errors="replace")
        sentences = _split_text_to_sentences(text_content)
        raw_sent_ids = _align_sentence_ids_to_events(
            event_words,
            sentences,
            min_matched_words=min_matched_words,
            source_label=f"{run}:{text_path.name}",
        )
        source_mode = "gentle_align_with_text_sentences"
    else:
        raw_sent_ids = _fallback_silence_sentence_ids(
            events, gap_seconds=silence_gap_seconds
        )
        source_mode = "gentle_align_silence_gap_fallback"
    context_ids = _chunk_sentence_ids(
        raw_sent_ids, run=run, max_context_words=max_context_words
    )
    return events, raw_sent_ids, context_ids, source_mode


def build_event_table(dataset_config: dict) -> tuple[pd.DataFrame, dict]:
    """从 Broderick2018 原始 .mat、Gentle align JSON 与原文构建 canonical 事件表。"""
    from braindecoding.evaluation.sequence_decoding_evaluation import is_content_word

    config = dict(dataset_config)
    raw_root = config.get("raw_root") or config.get("root")
    if not raw_root:
        raise ValueError("Broderick2018 build_event_table 需要配置 dataset.raw_root / dataset.root。")
    split_manifest = Path(
        config.get("split_manifest")
        or config.get("run_split_path")
        or PROJECT_ROOT / "data/manifests/broderick2018/run_split.json"
    )
    split_by_run = load_run_split(split_manifest)
    subjects = tuple(config.get("subjects") or SUBJECTS)
    runs = tuple(config.get("runs") or RUNS)
    max_context_words = int(config.get("max_context_words", DEFAULT_MAX_CONTEXT_WORDS))
    silence_gap = float(
        config.get("silence_sentence_gap_seconds", DEFAULT_SILENCE_GAP_SECONDS)
    )
    min_matched = int(config.get("min_matched_words", DEFAULT_MIN_MATCHED_WORDS))
    expected_source_rate = float(
        config.get("expected_source_sampling_rate_hz", SOURCE_SAMPLING_RATE_HZ)
    )
    contract = word_window_contract(config)

    run_cache: dict[str, tuple[list[dict], list[int], list[str], str]] = {}
    frames: list[pd.DataFrame] = []
    source_modes: dict[str, str] = {}
    recording_stats: list[dict] = []

    for run in runs:
        split = split_by_run[run]
        for subject in subjects:
            paths = recording_paths(raw_root, subject, run)
            eeg_mat = Path(paths["eeg_mat"])
            align_json = Path(paths["align_json"])
            text_txt = paths["text_txt"]
            if not eeg_mat.is_file():
                raise FileNotFoundError(f"Broderick2018 原始 EEG 文件缺失：{eeg_mat}")
            if not align_json.is_file():
                raise FileNotFoundError(f"Broderick2018 align JSON 缺失：{align_json}")

            if run not in run_cache:
                run_cache[run] = _run_sequence_ids(
                    align_json,
                    text_txt,
                    run=run,
                    max_context_words=max_context_words,
                    silence_gap_seconds=silence_gap,
                    min_matched_words=min_matched,
                )
            events, raw_sent_ids, context_ids, mode = run_cache[run]
            source_modes[run] = mode

            fs_hz, num_samples = _inspect_mat_metadata(eeg_mat)
            if abs(float(fs_hz) - expected_source_rate) > 1e-3:
                raise ValueError(
                    f"Broderick2018 .mat fs 异常 ({eeg_mat}): 期望 {expected_source_rate} Hz，实际 {fs_hz} Hz。"
                )
            duration_seconds = float(num_samples) / float(fs_hz)
            rec_id = f"{DATASET_ID}|{subject}|{run}"

            rows = []
            for idx, (ev, raw_sent, ctx_uid) in enumerate(
                zip(events, raw_sent_ids, context_ids)
            ):
                onset = float(ev["onset_seconds"])
                offset = max(onset, float(ev["offset_seconds"]))
                win_start = onset + float(contract["window_start_offset_seconds"])
                win_end = win_start + float(contract["window_seconds"])
                elig_end = win_start + float(contract["eligibility_window_seconds"])
                in_bounds = win_start >= 0.0 and elig_end <= duration_seconds
                trainable = bool(in_bounds and ev["normalized_word"])
                exclusion_reason = (
                    ""
                    if trainable
                    else ("window_out_of_bounds" if not in_bounds else "empty_word")
                )
                rows.append(
                    {
                        "事件编号": f"{rec_id}|{idx:06d}",
                        "受试者": subject,
                        "记录编号": rec_id,
                        "材料编号": f"{DATASET_ID}|{run}",
                        "划分单元": f"{DATASET_ID}|{run}",
                        "记录内序号": idx,
                        "词": ev["word"],
                        "标准词": ev["normalized_word"],
                        "开始时间": onset,
                        "结束时间": offset,
                        "上下文编号": ctx_uid,
                        "数据划分": split,
                        "是否可训练": trainable,
                        "排除原因": exclusion_reason,
                        "运行编号": run,
                        "时长": float(ev["duration_seconds"]),
                        "窗口开始": win_start,
                        "窗口结束": win_end,
                        "对齐词索引": int(ev["raw_align_index"]),
                        "原始句子编号": int(raw_sent),
                        "是否内容词": bool(
                            is_content_word(ev["normalized_word"], "en")
                        ),
                    }
                )
            frames.append(pd.DataFrame(rows))
            recording_stats.append(
                {
                    "recording_id": rec_id,
                    "subject": subject,
                    "run": run,
                    "split": split,
                    "source_sampling_rate_hz": float(fs_hz),
                    "source_sample_count": int(num_samples),
                    "duration_seconds": duration_seconds,
                    "event_count": len(rows),
                    "trainable_count": int(sum(1 for row in rows if row["是否可训练"])),
                }
            )

    table = pd.concat(frames, ignore_index=True)
    # 保证核心字段与持久化顺序一致并执行公共事件表合同校验
    for column in CORE_EVENT_COLUMNS:
        if column not in table.columns:
            raise ValueError(f"Broderick2018 构建事件表缺少核心列：{column}")
    table = table[list(PERSISTED_COLUMNS)].reset_index(drop=True)
    table["是否可训练"] = table["是否可训练"].astype(bool)
    table["排除原因"] = table["排除原因"].astype(str)
    validate_event_table(table)
    audit = {
        "dataset": DATASET_ID,
        "raw_dataset_name": RAW_DATASET_NAME,
        "dryad_doi": DRYAD_DOI,
        "scope": CORE_SCOPE,
        "subject_count": len(subjects),
        "run_count": len(runs),
        "recording_count": len(recording_stats),
        "total_events": int(len(table)),
        "trainable_events": int(table["是否可训练"].astype(bool).sum()),
        "unique_normalized_words": int(table["标准词"].nunique()),
        "sentence_segmentation_modes": source_modes,
        "recordings": recording_stats,
    }
    return table, audit


def load_event_table(
    path: str | Path,
    *,
    split: str | None = None,
    subjects=None,
    trainable_only: bool = False,
) -> pd.DataFrame:
    """读取 Broderick2018 canonical events.csv，校验合同并按可选条件过滤。"""
    df = pd.read_csv(path, keep_default_na=False)
    missing = sorted(set(CORE_EVENT_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Broderick2018 events.csv 缺少核心字段：{missing}")
    df["标准词"] = df["标准词"].map(normalize_english_word)
    df["开始时间"] = pd.to_numeric(df["开始时间"], errors="raise").astype(float)
    df["结束时间"] = pd.to_numeric(df["结束时间"], errors="raise").astype(float)
    for col in ("时长", "窗口开始", "窗口结束"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    df["记录内序号"] = pd.to_numeric(df["记录内序号"], errors="raise").astype(int)
    df["是否可训练"] = (
        df["是否可训练"]
        .map(lambda value: _bool_value(value, field="是否可训练"))
        .astype(bool)
    )
    df["排除原因"] = df["排除原因"].astype(str)
    if "是否内容词" in df.columns:
        df["是否内容词"] = (
            df["是否内容词"]
            .map(lambda value: _bool_value(value, field="是否内容词"))
            .astype(bool)
        )
    if "运行编号" not in df.columns:
        df["运行编号"] = df["记录编号"].astype(str).str.split("|").str[-1]
    validate_event_table(df)
    if split is not None:
        df = df[df["数据划分"].astype(str).eq(str(split))]
    if subjects is not None:
        allowed_subjects = {str(item) for item in subjects}
        df = df[df["受试者"].astype(str).isin(allowed_subjects)]
    if trainable_only:
        df = df[df["是否可训练"].astype(bool)]
    return df.reset_index(drop=True)


def material_word_view(table: pd.DataFrame) -> pd.DataFrame:
    """去除多被试重复，返回每段故事词位置唯一一次的材料级稳定视图。"""
    required = {
        "受试者",
        "运行编号",
        "记录内序号",
        "标准词",
        "数据划分",
        "划分单元",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Broderick2018 材料视图缺少字段：{missing}")

    parts = []
    present_runs = [run for run in RUNS if (table["运行编号"] == run).any()]
    if not present_runs:
        raise ValueError("Broderick2018 事件表没有任何有效运行段落。")
    for run in present_runs:
        run_rows = table[table["运行编号"].eq(run)]
        source_subject = str(sorted(run_rows["受试者"].astype(str).unique())[0])
        source = run_rows[run_rows["受试者"].eq(source_subject)].sort_values(
            "记录内序号", kind="stable"
        )
        expected_orders = source["记录内序号"].astype(int).tolist()
        expected_words = source["标准词"].map(normalize_english_word).tolist()
        for subject in sorted(run_rows["受试者"].astype(str).unique()):
            current = run_rows[run_rows["受试者"].eq(subject)].sort_values(
                "记录内序号", kind="stable"
            )
            if (
                current["记录内序号"].astype(int).tolist() != expected_orders
                or current["标准词"].map(normalize_english_word).tolist() != expected_words
            ):
                raise ValueError(
                    f"同一 Broderick2018 故事段跨被试词序列不一致：{subject}/{run}"
                )
        selected = source.copy()
        selected["_protocol_word"] = selected["标准词"].map(normalize_english_word)
        selected["_material_source_subject"] = source_subject
        parts.append(selected)
    result = pd.concat(parts, ignore_index=True)
    keys = ["运行编号", "记录内序号"]
    if result.duplicated(keys).any():
        raise ValueError("Broderick2018 材料视图含重复词位置。")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def add_runtime_context_uid(
    table: pd.DataFrame, max_context_words: int = DEFAULT_MAX_CONTEXT_WORDS
) -> pd.DataFrame:
    """建立受试者记录内的运行时上下文键，严格禁止跨被试或跨段注意力混合。"""
    required = {"受试者", "记录编号", "上下文编号", "记录内序号"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Broderick2018 运行时上下文缺少字段：{missing}")
    maximum = int(max_context_words)
    if maximum <= 0:
        raise ValueError("Broderick2018 max_context_words 必须为正整数。")

    result = table.copy()
    result[RUNTIME_CONTEXT_COLUMN] = (
        result["记录编号"].astype(str) + "|" + result["上下文编号"].astype(str)
    )
    grouped = result.groupby(RUNTIME_CONTEXT_COLUMN, sort=False)
    if grouped["受试者"].nunique().gt(1).any():
        raise ValueError("Broderick2018 运行时上下文跨受试者。")
    if grouped["记录编号"].nunique().gt(1).any():
        raise ValueError("Broderick2018 运行时上下文跨 recording。")
    if grouped["上下文编号"].nunique().gt(1).any():
        raise ValueError("Broderick2018 运行时上下文混合多个 canonical context。")
    sizes = grouped.size()
    oversized = sizes[sizes.gt(maximum)]
    if not oversized.empty:
        raise ValueError(
            f"Broderick2018 运行时上下文超过 max_context_words={maximum}："
            f"{oversized.index[0]}={int(oversized.iloc[0])}"
        )
    for context_uid, rows in grouped:
        order = pd.to_numeric(rows["记录内序号"], errors="coerce")
        if order.isna().any() or (np.diff(order.to_numpy()) <= 0).any():
            raise ValueError(
                f"Broderick2018 运行时上下文记录内序号未严格递增：{context_uid}"
            )
    return result


def signal_preprocessing_contract(dataset_config: dict | None = None) -> dict:
    """返回包含 mastoid_rereference 布尔开关的 Broderick2018 信号缓存合同签名。"""
    config = dict(dataset_config or {})
    if isinstance(config.get("dataset"), dict):
        config = dict(config["dataset"])
    mastoid_rereference = bool(config.get("mastoid_rereference", False))
    body = {
        "dataset": DATASET_ID,
        "montage": "biosemi128",
        "channel_count": EEG_CHANNEL_COUNT,
        "channel_names": list(BIOSEMI128_CHANNEL_NAMES),
        "channel_names_sha256": CHANNEL_NAMES_SHA256,
        "expected_source_sampling_rate_hz": float(
            config.get("expected_source_sampling_rate_hz", SOURCE_SAMPLING_RATE_HZ)
        ),
        "output_sampling_rate_hz": float(
            config.get("target_sampling_rate_hz", TARGET_SAMPLING_RATE_HZ)
        ),
        "bandpass_hz": [
            float(config.get("filter_low_hz", FILTER_LOW_HZ)),
            float(config.get("filter_high_hz", FILTER_HIGH_HZ)),
        ],
        "scaler": "RobustScaler",
        "mastoid_rereference": mastoid_rereference,
        "dtype": "float32",
    }
    body["preprocessing_contract_sha256"] = stable_sha256(body)
    return body


def _signal_product_paths(cache_dir: str | Path, subject: str, run: str) -> tuple[Path, Path]:
    stem = f"{subject}_{run}_eeg"
    signal_path = Path(cache_dir) / str(subject) / f"{stem}.npy"
    return signal_path, signal_path.with_suffix(".json")


def _process_eeg_array(
    eeg_data: np.ndarray,
    *,
    source_sampling_rate_hz: float,
    dataset_config: dict | None = None,
    mastoids: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """执行 0.1–40 Hz 带通滤波、50 Hz 重采样、可选乳突重参考与逐通道 RobustScaler。"""
    import mne
    from sklearn.preprocessing import RobustScaler

    config = dict(dataset_config or {})
    contract = signal_preprocessing_contract(config)
    expected_rate = float(contract["expected_source_sampling_rate_hz"])
    if abs(float(source_sampling_rate_hz) - expected_rate) > 1e-3:
        raise ValueError(
            f"Broderick2018 source_sampling_rate_hz 应为 {expected_rate} Hz，实际为 {source_sampling_rate_hz} Hz。"
        )
    data_tc = np.asarray(eeg_data, dtype=np.float64)
    if data_tc.ndim != 2 or data_tc.shape[1] != EEG_CHANNEL_COUNT:
        raise ValueError(
            f"Broderick2018 eegData 形状应为 [T, {EEG_CHANNEL_COUNT}]，实际为 {data_tc.shape}。"
        )
    if bool(contract["mastoid_rereference"]):
        if mastoids is None:
            raise ValueError("启用 mastoid_rereference=true 时必须提供 mastoids 数组。")
        ref = np.asarray(mastoids, dtype=np.float64).mean(axis=1, keepdims=True)
        data_tc = data_tc - ref

    # MNE expects [channels, times]; scalar factor 1.0 preserved because RobustScaler normalizes scale.
    data_ct = np.ascontiguousarray(data_tc.T, dtype=np.float64)
    info = mne.create_info(
        list(BIOSEMI128_CHANNEL_NAMES),
        sfreq=float(source_sampling_rate_hz),
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data_ct, info, verbose=False)
    n_jobs = int(config.get("mne_n_jobs", 1))
    low_hz, high_hz = contract["bandpass_hz"]
    raw.filter(low_hz, high_hz, n_jobs=n_jobs, verbose=False)
    raw = raw.resample(
        float(contract["output_sampling_rate_hz"]), n_jobs=n_jobs, verbose=False
    )
    scaled = RobustScaler().fit_transform(raw._data.T).T.astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Broderick2018 预处理后的信号包含非有限值 (NaN/Inf)。")
    meta = {
        "source_first_samp": 0,
        "source_sampling_rate_hz": float(source_sampling_rate_hz),
        "output_first_samp": int(raw.first_samp),
        "output_sampling_rate_hz": float(raw.info["sfreq"]),
        "output_sample_count": int(scaled.shape[1]),
    }
    return scaled, meta


def _valid_signal_product(output_path: Path, expected_source_contract: dict) -> bool:
    sidecar_path = output_path.with_suffix(".json")
    if not output_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        sidecar = _read_json(sidecar_path)
    except Exception:
        return False
    for key, value in expected_source_contract.items():
        if sidecar.get(key) != value:
            return False
    return True


def materialize_recording_signal(
    *,
    raw_root: str | Path,
    subject: str,
    run: str,
    cache_dir: str | Path,
    dataset_config: dict | None = None,
    split: str = "",
    force: bool = False,
) -> Path:
    """将单条 Broderick2018 .mat 信号物化为 50 Hz .npy 与 .json sidecar。"""
    paths = recording_paths(raw_root, subject, run)
    mat_path = Path(paths["eeg_mat"])
    if not mat_path.is_file():
        raise FileNotFoundError(f"Broderick2018 原始 .mat 不存在：{mat_path}")
    contract = signal_preprocessing_contract(dataset_config)
    signal_path, sidecar_path = _signal_product_paths(cache_dir, subject, run)
    rec_id = f"{DATASET_ID}|{subject}|{run}"
    expected_contract = {
        "dataset": DATASET_ID,
        "recording_id": rec_id,
        "channel_count": EEG_CHANNEL_COUNT,
        "channel_names_sha256": CHANNEL_NAMES_SHA256,
        "output_sampling_rate_hz": float(contract["output_sampling_rate_hz"]),
        "dtype": "float32",
        "mastoid_rereference": bool(contract["mastoid_rereference"]),
        "preprocessing_contract_sha256": contract["preprocessing_contract_sha256"],
    }
    if not force and _valid_signal_product(signal_path, expected_contract):
        return signal_path

    eeg_data, mastoids, fs_hz = _load_mat_eeg_and_mastoids(
        mat_path, load_mastoids=bool(contract["mastoid_rereference"])
    )
    scaled, meta = _process_eeg_array(
        eeg_data,
        source_sampling_rate_hz=fs_hz,
        dataset_config=dataset_config,
        mastoids=mastoids,
    )
    _write_array_atomic(signal_path, scaled)
    sidecar = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "recording_id": rec_id,
        "subject": subject,
        "run": run,
        "split": split,
        "source_file": str(mat_path),
        "signal_file": signal_path.name,
        "signal_file_sha256": sha256_file(signal_path),
        "channel_count": EEG_CHANNEL_COUNT,
        "channel_names": list(BIOSEMI128_CHANNEL_NAMES),
        "channel_names_sha256": CHANNEL_NAMES_SHA256,
        "dtype": "float32",
        "mastoid_rereference": bool(contract["mastoid_rereference"]),
        "preprocessing_contract_sha256": contract["preprocessing_contract_sha256"],
        **meta,
    }
    _write_json_atomic(sidecar_path, sidecar)
    return signal_path


def validate_signal_manifest(path: str | Path, *, event_table: pd.DataFrame | None = None) -> dict:
    """校验 Broderick2018 canonical signals/manifest.json 与磁盘文件一致性。"""
    manifest_path = Path(path)
    payload = _validate_self_hash(_read_json(manifest_path), path=manifest_path)
    if payload.get("dataset") != DATASET_ID or payload.get("status") != "complete":
        raise ValueError(f"Broderick2018 signal manifest 状态无效：{manifest_path}")
    if int(payload.get("channel_count", -1)) != EEG_CHANNEL_COUNT:
        raise ValueError("Broderick2018 signal manifest 通道数必须为 128。")
    if payload.get("channel_names_sha256") != CHANNEL_NAMES_SHA256:
        raise ValueError("Broderick2018 signal manifest 通道顺序摘要不一致。")
    records = payload.get("recordings", [])
    if not records:
        raise ValueError("Broderick2018 signal manifest 不能为空。")
    if event_table is not None:
        expected_ids = set(event_table["记录编号"].astype(str).unique())
        recorded_ids = {str(item["recording_id"]) for item in records}
        missing = sorted(expected_ids - recorded_ids)
        if missing:
            raise ValueError(f"Broderick2018 signal manifest 缺少记录：{missing[:5]}")
    return payload


def build_signal_manifest(
    *,
    event_table: pd.DataFrame,
    cache_dir: str | Path,
    dataset_config: dict | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    """扫描已物化 EEG 信号并原子写入 signals/manifest.json。"""
    cache_path = Path(cache_dir)
    if manifest_path is not None:
        target = Path(manifest_path)
    elif cache_path.name == "eeg_50hz":
        target = cache_path.parent / "manifest.json"
    else:
        target = cache_path / "manifest.json"
    contract = signal_preprocessing_contract(dataset_config)
    records = []
    dedup = (
        event_table[["受试者", "记录编号", "运行编号", "数据划分"]]
        .drop_duplicates()
        .sort_values(["受试者", "运行编号"], kind="mergesort")
    )
    for subject, rec_id, run, split in dedup.itertuples(index=False, name=None):
        signal_path, sidecar_path = _signal_product_paths(cache_path, str(subject), str(run))
        if not signal_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"Broderick2018 canonical signal 缺失：{signal_path}")
        sidecar = _read_json(sidecar_path)
        if sidecar.get("preprocessing_contract_sha256") != contract["preprocessing_contract_sha256"]:
            raise ValueError(f"Broderick2018 signal contract 签名不匹配：{sidecar_path}")
        records.append(
            {
                "recording_id": str(rec_id),
                "subject": str(subject),
                "run": str(run),
                "split": str(split),
                "signal_path": _portable_manifest_path(signal_path),
                "sidecar_path": _portable_manifest_path(sidecar_path),
                "signal_file_sha256": str(sidecar["signal_file_sha256"]),
                "sample_count": int(sidecar["output_sample_count"]),
                "source_sampling_rate_hz": float(sidecar["source_sampling_rate_hz"]),
                "output_sampling_rate_hz": float(sidecar["output_sampling_rate_hz"]),
            }
        )
    payload = _with_manifest_sha256(
        {
            "schema_version": 1,
            "dataset": DATASET_ID,
            "status": "complete",
            "channel_count": EEG_CHANNEL_COUNT,
            "channel_names_sha256": CHANNEL_NAMES_SHA256,
            "output_sampling_rate_hz": float(contract["output_sampling_rate_hz"]),
            "mastoid_rereference": bool(contract["mastoid_rereference"]),
            "preprocessing_contract_sha256": contract["preprocessing_contract_sha256"],
            "recording_count": len(records),
            "coverage": {
                "recording_count": len(records),
                "event_count": int(len(event_table)),
                "trainable_event_count": int(event_table["是否可训练"].astype(bool).sum()),
                "complete": True,
            },
            "recordings": records,
        }
    )
    _write_json_atomic(target, payload)
    validate_signal_manifest(target, event_table=event_table)
    return target


def ensure_recording_signals(
    event_table: pd.DataFrame,
    dataset_config: dict,
    cache_dir: str | Path,
    *,
    project_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    force: bool = False,
) -> Path:
    """按事件表批量物化所有 Broderick2018 50 Hz EEG 连续信号并写入 manifest。"""
    del project_root
    raw_root = dataset_config.get("raw_root") or dataset_config.get("root")
    if not raw_root:
        raise ValueError("Broderick2018 ensure_recording_signals 需要 dataset.raw_root / dataset.root。")
    dedup = (
        event_table[["受试者", "运行编号", "数据划分"]]
        .drop_duplicates()
        .sort_values(["受试者", "运行编号"], kind="mergesort")
    )
    for subject, run, split in dedup.itertuples(index=False, name=None):
        materialize_recording_signal(
            raw_root=raw_root,
            subject=str(subject),
            run=str(run),
            cache_dir=cache_dir,
            dataset_config=dataset_config,
            split=str(split),
            force=force,
        )
    return build_signal_manifest(
        event_table=event_table,
        cache_dir=cache_dir,
        dataset_config=dataset_config,
        manifest_path=manifest_path,
    )


def _embedding_content_sha256(words, embeddings) -> str:
    matrix = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(words), ensure_ascii=False).encode("utf-8"))
    digest.update(str(matrix.shape).encode("ascii"))
    digest.update(str(matrix.dtype).encode("ascii"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _reference_t5_embeddings(words, config, tokenizer, model, device) -> np.ndarray:
    inputs = tokenizer(
        list(words),
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    states = outputs.hidden_states
    layer_index = int(float(config["layer_fraction"]) * len(states) - 1e-6)
    hidden = states[layer_index]
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.float().cpu().numpy().astype(np.float32, copy=False)


def validate_text_manifest(path: str | Path, *, expected_config: dict | None = None) -> dict:
    """校验 Broderick2018 canonical text manifest 与内容摘要。"""
    manifest_path = Path(path)
    payload = _validate_self_hash(_read_json(manifest_path), path=manifest_path)
    if payload.get("dataset") != DATASET_ID or payload.get("status") != "complete":
        raise ValueError(f"Broderick2018 text manifest 状态无效：{manifest_path}")
    embedding_path = manifest_path.parent / payload["embedding_file"]
    sidecar_path = manifest_path.parent / payload["sidecar_file"]
    if sha256_file(embedding_path) != payload["embedding_file_sha256"]:
        raise ValueError(f"Broderick2018 text NPZ SHA 漂移：{embedding_path}")
    if sha256_file(sidecar_path) != payload["sidecar_sha256"]:
        raise ValueError(f"Broderick2018 text sidecar SHA 漂移：{sidecar_path}")
    if expected_config is not None:
        sig = text_data.text_embedding_signature(expected_config)
        sidecar = _read_json(sidecar_path)
        if sidecar.get("signature") != sig:
            raise ValueError("Broderick2018 text cache signature 与配置不一致。")
    return payload


def build_text_product(
    event_table: pd.DataFrame,
    config: dict,
    output_path: str | Path,
    *,
    event_table_path: str | Path,
    force: bool = False,
) -> Path:
    """构建 Broderick2018 孤立词型 T5-large 第 12 层均值池化嵌入查找表。"""
    target = Path(output_path)
    manifest_path = target.parent / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = validate_text_manifest(manifest_path, expected_config=config)
        if manifest.get("event_table_sha256") != sha256_file(event_table_path):
            raise ValueError("Broderick2018 text product 事件表合同已变化。")
        return manifest_path

    material_table = material_word_view(event_table)
    words = text_data.prepared_words(material_table["标准词"], config)
    tokenizer, model, device = text_data.load_text_model(config)
    fixture_words = list(ENGLISH_REFERENCE_TOKENS)
    direct = _reference_t5_embeddings(fixture_words, config, tokenizer, model, device)
    shared = text_data.encode_words(
        fixture_words,
        config,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    if not np.allclose(direct, shared, atol=1e-6):
        raise ValueError("Broderick2018 T5 reference fixture 数值门未通过。")

    embeddings = text_data.encode_words(
        words, config, tokenizer=tokenizer, model=model, device=device
    )
    content_digest = _embedding_content_sha256(words, embeddings)
    text_data.save_text_embedding_cache(target, words, embeddings, config)
    sidecar_path = target.with_suffix(".json")
    payload = _with_manifest_sha256(
        {
            "schema_version": 1,
            "dataset": DATASET_ID,
            "status": "complete",
            "model_name": str(config["model_name"]),
            "layer_fraction": float(config["layer_fraction"]),
            "selected_hidden_layer": int(
                float(config["layer_fraction"])
                * (int(getattr(model.config, "num_layers")) + 1)
                - 1e-6
            ),
            "token_aggregation": str(config["token_aggregation"]),
            "embedding_dimension": int(config["embedding_dimension"]),
            "dtype": "float32",
            "normalization": "ascii_alnum_hyphen_apostrophe_lower",
            "word_type_count": len(words),
            "embedding_shape": list(embeddings.shape),
            "embedding_file": target.name,
            "embedding_file_sha256": sha256_file(target),
            "sidecar_file": sidecar_path.name,
            "sidecar_sha256": sha256_file(sidecar_path),
            "content_sha256": content_digest,
            "event_table_sha256": sha256_file(event_table_path),
            "software": {
                "transformers": importlib.metadata.version("transformers"),
                "torch": torch.__version__,
                "numpy": np.__version__,
            },
        }
    )
    _write_json_atomic(manifest_path, payload)
    validate_text_manifest(manifest_path)
    return manifest_path


def build_protocol_assets(event_table_path: str | Path) -> dict[str, dict]:
    """从 Broderick2018 frozen event table 构建候选词表与支持审计清单。"""
    event_path = Path(event_table_path)
    table = load_event_table(event_path)
    material = material_word_view(table)
    material = material[material["_protocol_word"].ne("")].reset_index(drop=True)
    event_digest = sha256_file(event_path)
    generator_digest = sha256_file(Path(__file__))

    train = material[material["数据划分"].eq("train")].reset_index(drop=True)
    counts = Counter(train["_protocol_word"])
    ranked = sorted(counts, key=lambda word: (-int(counts[word]), word))
    source_units = sorted(train["划分单元"].astype(str).unique())

    vocabularies = {}
    documents = {}
    for size in (20, 50, 100, 150):
        vocabulary = ranked[:size]
        if len(vocabulary) != size:
            raise ValueError(f"Broderick2018 train 材料不足以冻结 N={size} 词表。")
        manifest = _with_manifest_sha256(
            {
                "asset_type": "candidate_vocabulary",
                "created_from_test": False,
                "dataset": DATASET_ID,
                "evaluation_support_used_for_selection": False,
                "event_table": _portable_manifest_path(event_path),
                "event_table_sha256": event_digest,
                "frequency_unit": "material_word_occurrence",
                "generator": "braindecoding.data.broderick2018_dataset.build_protocol_assets",
                "generator_sha256": generator_digest,
                "hash_contract": "sha256(canonical_json_without_manifest_sha256)",
                "normalization": "ascii_alnum_hyphen_apostrophe_lower",
                "policy": "frequency_desc_word_asc",
                "scope": CORE_SCOPE,
                "source_split": "train",
                "source_split_units": source_units,
                "source_units": source_units,
                "status": "frozen",
                "train_cutoff_frequency": int(counts[vocabulary[-1]]),
                "train_token_coverage": float(
                    train["_protocol_word"].isin(vocabulary).mean()
                ),
                "vocabulary": vocabulary,
                "vocabulary_size": size,
                "word_counts": {word: int(counts[word]) for word in vocabulary},
            }
        )
        vocabularies[size] = manifest
        documents[f"vocabulary_N{size}.json"] = manifest

    support = {}
    for size, vocabulary_manifest in vocabularies.items():
        vocabulary = list(vocabulary_manifest["vocabulary"])
        vocab_set = set(vocabulary)
        split_records = {}
        for split in ("val", "test"):
            words = material.loc[
                material["数据划分"].eq(split), "_protocol_word"
            ].tolist()
            observed = set(words)
            missing = [word for word in vocabulary if word not in observed]
            split_records[split] = {
                "candidate_count": size,
                "supported_candidate_count": size - len(missing),
                "missing_candidate_count": len(missing),
                "missing_words": missing,
                "support_fraction": (size - len(missing)) / size,
                "token_count": len(words),
                "token_coverage": (
                    float(sum(word in vocab_set for word in words) / len(words))
                    if words
                    else 0.0
                ),
            }
        support[f"N{size}"] = {
            "candidate_manifest_sha256": vocabulary_manifest["manifest_sha256"],
            "splits": split_records,
        }
    documents["vocabulary_support.json"] = _with_manifest_sha256(
        {
            "asset_type": "evaluation_vocabulary_support",
            "dataset": DATASET_ID,
            "generator": "braindecoding.data.broderick2018_dataset.build_protocol_assets",
            "generator_sha256": generator_digest,
            "status": "frozen_support_audit",
            "support_audit_used_for_vocabulary_selection": False,
            "test_labels_used_for_support_audit_only": True,
            "vocabularies": support,
        }
    )
    return documents


def write_protocol_assets(
    event_table_path: str | Path, output_dir: str | Path
) -> list[Path]:
    """原子写入 Broderick2018 机器可读词表与支持审计清单。"""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, payload in sorted(build_protocol_assets(event_table_path).items()):
        path = target_dir / name
        _write_json_atomic(path, payload)
        written.append(path)
    return written


@lru_cache(maxsize=256)
def _open_signal_product(path: str):
    return np.load(path, mmap_mode="r")


@lru_cache(maxsize=8)
def _cached_broderick_train_recordings(
    events_csv_path: str, mtime_ns: int
) -> tuple[tuple[str, str, str], ...]:
    del mtime_ns
    df = pd.read_csv(
        events_csv_path,
        usecols=["受试者", "记录编号", "运行编号", "数据划分", "是否可训练"],
        keep_default_na=False,
    )
    trainable_mask = df["是否可训练"].map(
        lambda value: _bool_value(value, field="是否可训练")
    )
    train_df = df[df["数据划分"].eq("train") & trainable_mask]
    dedup = (
        train_df[["受试者", "记录编号", "运行编号"]]
        .drop_duplicates()
        .sort_values(["受试者", "运行编号", "记录编号"], kind="mergesort")
    )
    return tuple(
        (str(subj), str(rec_id), str(run))
        for subj, rec_id, run in dedup.itertuples(index=False, name=None)
    )


def _concat_readonly_tail_from_signals(
    signal_paths: list[Path],
    *,
    required_samples: int,
    channel_count: int,
    subject_id: str,
    max_samples_by_path: dict[Path, int] | None = None,
) -> np.ndarray:
    """从同被试合法 train 连续信号末端逆序截取 required_samples 只读前置校准缓冲。"""
    if required_samples <= 0:
        return np.empty((channel_count, 0), dtype=np.float32)
    chunks: list[np.ndarray] = []
    collected = 0
    for sig_path in reversed(signal_paths):
        arr = _open_signal_product(str(sig_path))
        cap = (
            min(int(arr.shape[1]), int(max_samples_by_path[sig_path]))
            if max_samples_by_path and sig_path in max_samples_by_path
            else int(arr.shape[1])
        )
        if cap <= 0:
            continue
        need = required_samples - collected
        take = min(need, cap)
        if take > 0:
            chunks.append(np.asarray(arr[:, cap - take : cap], dtype=np.float32))
            collected += take
        if collected >= required_samples:
            break
    if collected < required_samples:
        raise ValueError(
            f"Broderick2018 受试者 {subject_id} 可用 train 连续信号总长度 ({collected}) "
            f"不足以提供 Pre-Roll 前置校准缓冲 ({required_samples})。"
        )
    concatenated = np.concatenate(list(reversed(chunks)), axis=1).astype(
        np.float32, copy=True
    )
    concatenated.setflags(write=False)
    return concatenated


class Broderick2018WordDataset(Dataset):
    """从 Broderick2018 冻结事件、128 导联 EEG 连续信号和 T5 词向量构造 1 秒词样本。"""

    def __init__(
        self,
        table: pd.DataFrame,
        dataset_config: dict,
        signal_cache_dir: str | Path,
        text_embedding_path: str | Path,
        text_embedding_config: dict,
        *,
        zero_eeg: bool = False,
        temporal_shift_seconds: float = 0.0,
        shift_mode: str | None = None,
        preroll_seconds: float | None = None,
        preroll_source_table: pd.DataFrame | None = None,
        preroll_buffers: dict[str, np.ndarray] | None = None,
        allow_subject_subset: bool = False,
    ):
        source = table.copy()
        missing = sorted(set(CORE_EVENT_COLUMNS) - set(source.columns))
        if missing:
            raise ValueError(f"Broderick2018 Dataset 缺少核心事件字段：{missing}")
        source = source[source["是否可训练"].astype(bool)].reset_index(drop=True)
        if source.empty:
            raise ValueError("Broderick2018 Dataset 没有可训练事件。")

        configured_subjects = tuple(
            str(value) for value in dataset_config.get("subjects", SUBJECTS)
        )
        if allow_subject_subset:
            if (
                not configured_subjects
                or tuple(s for s in SUBJECTS if s in configured_subjects)
                != configured_subjects
            ):
                raise ValueError(
                    "Broderick2018 subject subset 必须非空且保持 canonical 顺序。"
                )
        elif configured_subjects != SUBJECTS:
            raise ValueError(
                "Broderick2018 subject 顺序必须固定为 sub-01 至 sub-19。"
            )
        observed_subjects = set(source["受试者"].astype(str))
        if not observed_subjects.issubset(set(configured_subjects)):
            raise ValueError("Broderick2018 事件表含配置外受试者。")

        self.table = add_runtime_context_uid(
            source,
            max_context_words=int(
                dataset_config.get("max_context_words", DEFAULT_MAX_CONTEXT_WORDS)
            ),
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
        if "运行编号" not in self.table.columns:
            self.table["运行编号"] = (
                self.table["recording_id"].astype(str).str.split("|").str[-1]
            )

        self.dataset_config = dict(dataset_config)
        self.eeg_cache_dir = Path(signal_cache_dir)
        self.meg_cache_dir = self.eeg_cache_dir  # 兼容公共数据集属性读取
        self.zero_eeg = bool(zero_eeg)
        self.zero_meg = self.zero_eeg

        contract = word_window_contract(dataset_config)
        self.window_seconds = float(contract["window_seconds"])
        self.window_samples = int(contract["window_samples"])
        self._baseline_samples = int(contract["baseline_samples"])
        self._clamp = float(contract["clamp"])
        self._window_offset_seconds = float(contract["window_start_offset_seconds"])
        self.sampling_rate_hz = float(
            contract.get("sampling_rate_hz", TARGET_SAMPLING_RATE_HZ)
        )
        self.temporal_shift_seconds = float(temporal_shift_seconds)
        self.temporal_shift_samples = int(
            round(self.temporal_shift_seconds * self.sampling_rate_hz)
        )
        resolved_shift_mode = str(
            shift_mode
            if shift_mode is not None
            else self.dataset_config.get("shift_mode", "deterministic_circular")
        ).strip().lower()
        if resolved_shift_mode not in VALID_TEMPORAL_SHIFT_MODES:
            raise ValueError(
                f"Broderick2018 shift_mode 无效：{resolved_shift_mode!r}，可选 {VALID_TEMPORAL_SHIFT_MODES}"
            )
        self.shift_mode = resolved_shift_mode
        self.preroll_seconds = float(
            preroll_seconds
            if preroll_seconds is not None
            else self.dataset_config.get("preroll_seconds", DEFAULT_PREROLL_SECONDS)
        )
        if self.preroll_seconds <= 0.0:
            raise ValueError(f"Broderick2018 preroll_seconds 必须为正数：{self.preroll_seconds}")
        if self.shift_mode == "preroll" and self.temporal_shift_samples > 0:
            raise ValueError(
                "Broderick2018 Pre-Roll 因果移位模式仅允许严格过去移位 (temporal_shift_seconds <= 0)。"
            )

        self.channel_count = EEG_CHANNEL_COUNT
        self.subject_map = {
            subject: index for index, subject in enumerate(configured_subjects)
        }
        self.subject_count = len(self.subject_map)
        self.subject_indices = np.asarray(
            [self.subject_map[value] for value in self.table["subject_id"].astype(str)],
            dtype=np.int64,
        )
        contexts = self.table[RUNTIME_CONTEXT_COLUMN].astype(str).tolist()
        context_map = {value: index for index, value in enumerate(dict.fromkeys(contexts))}
        self.sentence_indices = np.asarray(
            [context_map[value] for value in contexts], dtype=np.int64
        )

        signature = text_data.text_embedding_signature(text_embedding_config)
        self.embedding_map = text_data.load_text_embedding_cache(
            text_embedding_path, expected_signature=signature
        )
        expected_dimension = int(text_embedding_config["embedding_dimension"])
        missing_words = sorted(
            set(self.table["normalized_word"].astype(str)) - set(self.embedding_map)
        )
        if missing_words:
            raise KeyError(f"Broderick2018 文本缓存缺少词型：{missing_words[:5]}")
        if any(value.shape != (expected_dimension,) for value in self.embedding_map.values()):
            raise ValueError("Broderick2018 文本向量维数与配置不一致。")

        self.signal_paths: dict[str, Path] = {}
        self.signal_metadata: dict[str, dict] = {}
        channel_names = None
        for row in self.table[
            ["recording_id", "subject_id", "运行编号"]
        ].drop_duplicates().itertuples(index=False):
            recording_id, subject, run = map(str, row)
            signal_path, sidecar_path = _signal_product_paths(
                self.eeg_cache_dir, subject, run
            )
            if not signal_path.is_file() or not sidecar_path.is_file():
                raise FileNotFoundError(
                    f"Broderick2018 canonical signal product 缺失：{signal_path}"
                )
            metadata = _read_json(sidecar_path)
            if metadata.get("recording_id") != recording_id:
                raise ValueError(
                    f"Broderick2018 signal sidecar 记录编号不一致：{sidecar_path}"
                )
            if (
                int(metadata.get("channel_count", -1)) != EEG_CHANNEL_COUNT
                or metadata.get("channel_names_sha256") != CHANNEL_NAMES_SHA256
                or float(metadata.get("output_sampling_rate_hz", -1))
                != TARGET_SAMPLING_RATE_HZ
                or metadata.get("dtype") != "float32"
            ):
                raise ValueError(f"Broderick2018 signal sidecar 合同不一致：{sidecar_path}")
            current_names = tuple(metadata["channel_names"])
            channel_names = current_names if channel_names is None else channel_names
            if current_names != channel_names:
                raise ValueError("Broderick2018 recording 间通道顺序不一致。")
            self.signal_paths[recording_id] = signal_path
            self.signal_metadata[recording_id] = metadata

        self.channel_names = list(channel_names or BIOSEMI128_CHANNEL_NAMES)
        if channel_names_sha256(self.channel_names) != CHANNEL_NAMES_SHA256:
            raise ValueError("Broderick2018 Dataset 通道顺序 SHA 不一致。")
        self.channel_positions = biosemi128_channel_positions(self.channel_names)

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
                    source_first_samp=int(
                        self.signal_metadata[rec_id].get("source_first_samp", 0)
                    ),
                    source_sampling_rate_hz=float(
                        self.signal_metadata[rec_id].get(
                            "source_sampling_rate_hz", SOURCE_SAMPLING_RATE_HZ
                        )
                    ),
                    output_sampling_rate_hz=float(
                        self.signal_metadata[rec_id].get(
                            "output_sampling_rate_hz", TARGET_SAMPLING_RATE_HZ
                        )
                    ),
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
        self._preroll_source_table = preroll_source_table
        self._explicit_preroll_buffers = preroll_buffers
        if (
            not self.zero_eeg
            and self.shift_mode == "preroll"
            and self.temporal_shift_samples < 0
        ):
            self._init_preroll_buffers(
                preroll_source_table=preroll_source_table,
                preroll_buffers=preroll_buffers,
            )

    def ensure_preroll_buffers(
        self,
        shift_seconds: float | None = None,
        *,
        shift_mode: str | None = None,
        preroll_seconds: float | None = None,
    ) -> None:
        """按需初始化或扩容只读 45 秒前滚缓冲。"""
        if shift_mode is not None:
            mode_norm = str(shift_mode).strip().lower()
            if mode_norm not in VALID_TEMPORAL_SHIFT_MODES:
                raise ValueError(f"Broderick2018 shift_mode 无效：{mode_norm!r}")
            self.shift_mode = mode_norm
        if preroll_seconds is not None:
            self.preroll_seconds = float(preroll_seconds)
        if shift_seconds is not None:
            self.temporal_shift_seconds = float(shift_seconds)
            self.temporal_shift_samples = int(
                round(self.temporal_shift_seconds * self.sampling_rate_hz)
            )
        if self.zero_eeg or self.shift_mode != "preroll" or self.temporal_shift_samples == 0:
            return
        if self.temporal_shift_samples > 0:
            raise ValueError(
                "Broderick2018 Pre-Roll 因果移位模式仅允许严格过去移位 (temporal_shift_seconds <= 0)。"
            )
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
            preroll_source_table=self._preroll_source_table,
            preroll_buffers=self._explicit_preroll_buffers,
        )

    def _init_preroll_buffers(
        self,
        *,
        preroll_source_table: pd.DataFrame | None = None,
        preroll_buffers: dict[str, np.ndarray] | None = None,
    ) -> None:
        required_samples = max(
            int(round(self.preroll_seconds * self.sampling_rate_hz)),
            abs(int(self.temporal_shift_samples)) + int(self.window_samples),
        )
        observed_pairs = list(
            self.table[["subject_id", "recording_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        if preroll_buffers is not None:
            for subj_raw, rec_raw in observed_pairs:
                subj = str(subj_raw)
                rec_id = str(rec_raw)
                buf = preroll_buffers.get(rec_id)
                if buf is None:
                    buf = preroll_buffers.get(subj)
                if buf is None:
                    raise ValueError(
                        f"Broderick2018 preroll_buffers 缺少受试者 {subj} (recording={rec_id}) 的前置校准信号。"
                    )
                arr = np.asarray(buf, dtype=np.float32)
                if arr.ndim != 2 or int(arr.shape[0]) != int(self.channel_count):
                    raise ValueError(
                        f"Broderick2018 preroll_buffers[{subj}] 形状无效：{arr.shape}"
                    )
                if int(arr.shape[1]) < required_samples:
                    raise ValueError(
                        f"Broderick2018 preroll_buffers[{subj}] 长度 ({arr.shape[1]}) 小于所需采样点 ({required_samples})。"
                    )
                sliced = np.array(arr[:, -required_samples:], dtype=np.float32, copy=True)
                sliced.setflags(write=False)
                self._preroll_by_recording[rec_id] = sliced
            return

        train_rows: list[tuple[str, str, str]] = []
        shared_rec_max_train_end: dict[str, int] = {}
        if preroll_source_table is not None:
            src_df = preroll_source_table
            split_col = "数据划分" if "数据划分" in src_df.columns else "split"
            subj_col = "受试者" if "受试者" in src_df.columns else "subject_id"
            rec_col = "记录编号" if "记录编号" in src_df.columns else "recording_id"
            run_col = "运行编号" if "运行编号" in src_df.columns else "run"
            onset_col = "开始时间" if "开始时间" in src_df.columns else "onset_seconds"
            train_col = "是否可训练" if "是否可训练" in src_df.columns else "is_trainable"
            trainable_mask = (
                src_df[train_col].astype(bool)
                if train_col in src_df.columns
                else pd.Series(True, index=src_df.index)
            )
            mask = src_df[split_col].astype(str).eq("train") & trainable_mask
            dedup = (
                src_df.loc[mask, [subj_col, rec_col, run_col]]
                .drop_duplicates()
                .sort_values([subj_col, run_col, rec_col], kind="mergesort")
            )
            train_rows = [
                (str(s), str(r), str(u))
                for s, r, u in dedup.itertuples(index=False, name=None)
            ]
            non_train_mask = (~src_df[split_col].astype(str).eq("train")) & trainable_mask
            shared_recs = set(src_df.loc[mask, rec_col].astype(str)) & set(
                src_df.loc[non_train_mask, rec_col].astype(str)
            )
            if shared_recs and onset_col in src_df.columns:
                for s_rec in shared_recs:
                    rec_rows = src_df[src_df[rec_col].astype(str).eq(s_rec) & trainable_mask]
                    s_subj = str(rec_rows[subj_col].iloc[0])
                    s_run = str(rec_rows[run_col].iloc[0])
                    _, sidecar_p = _signal_product_paths(self.eeg_cache_dir, s_subj, s_run)
                    sidecar = _read_json(sidecar_p) if sidecar_p.is_file() else {}
                    first_samp = int(
                        sidecar.get(
                            "source_first_samp",
                            self.signal_metadata.get(s_rec, {}).get("source_first_samp", 0),
                        )
                    )
                    src_rate = float(
                        sidecar.get(
                            "source_sampling_rate_hz",
                            self.signal_metadata.get(s_rec, {}).get(
                                "source_sampling_rate_hz", SOURCE_SAMPLING_RATE_HZ
                            ),
                        )
                    )
                    out_rate = float(
                        sidecar.get("output_sampling_rate_hz", self.sampling_rate_hz)
                    )
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
                else self.eeg_cache_dir.parent.parent / "events" / "events.csv"
            )
            if csv_path.is_file():
                train_rows = list(
                    _cached_broderick_train_recordings(
                        str(csv_path.resolve()), csv_path.stat().st_mtime_ns
                    )
                )
            elif "split" in self.table.columns:
                mask = self.table["split"].astype(str).eq("train")
                dedup = (
                    self.table.loc[mask, ["subject_id", "recording_id", "运行编号"]]
                    .drop_duplicates()
                    .sort_values(["subject_id", "运行编号", "recording_id"], kind="mergesort")
                )
                train_rows = [
                    (str(s), str(r), str(u))
                    for s, r, u in dedup.itertuples(index=False, name=None)
                ]

        by_subject: dict[str, list[tuple[str, Path]]] = {}
        max_samples_by_path: dict[Path, int] = {}
        for subj, rec_id, run in train_rows:
            sig_path, _ = _signal_product_paths(self.eeg_cache_dir, subj, run)
            if sig_path.is_file():
                by_subject.setdefault(subj, []).append((rec_id, sig_path))
                if rec_id in shared_rec_max_train_end:
                    max_samples_by_path[sig_path] = shared_rec_max_train_end[rec_id]

        primary_cache: dict[str, tuple[str, np.ndarray]] = {}
        alt_cache: dict[str, np.ndarray] = {}
        for subj_raw, rec_raw in observed_pairs:
            subj = str(subj_raw)
            rec_id = str(rec_raw)
            records = by_subject.get(subj, [])
            if not records:
                raise ValueError(
                    f"Broderick2018 Pre-Roll 模式未找到受试者 {subj} 的 train 划分校准信号。"
                )
            if subj not in primary_cache:
                last_rec_id = records[-1][0]
                primary_buf = _concat_readonly_tail_from_signals(
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
                        alt_cache[subj] = _concat_readonly_tail_from_signals(
                            [p for _, p in records[:-1]],
                            required_samples=required_samples,
                            channel_count=self.channel_count,
                            subject_id=subj,
                            max_samples_by_path=max_samples_by_path or None,
                        )
                    self._preroll_by_recording[rec_id] = alt_cache[subj]
                else:
                    raise ValueError(
                        f"Broderick2018 受试者 {subj} 仅有一条未截断 train 记录 ({rec_id})，"
                        "Pre-Roll 模式禁止使用同一记录末尾环绕充当校准缓冲。"
                    )
            else:
                self._preroll_by_recording[rec_id] = primary_buf

    def _get_preroll_for_index(self, idx: int) -> np.ndarray:
        buf = self._preroll_by_recording.get(self._recording_ids[idx])
        if buf is None:
            raise KeyError(
                f"Broderick2018 未找到记录 {self._recording_ids[idx]} 的 Pre-Roll 校准缓冲。"
            )
        return buf

    def __len__(self) -> int:
        return len(self.table)

    def read_eeg(self, index: int) -> np.ndarray:
        idx = int(index)
        if self.zero_eeg:
            return np.zeros((self.channel_count, self.window_samples), dtype=np.float32)

        recording = _open_signal_product(self._signal_path_strs[idx])
        start = int(self._window_starts[idx])
        stop = start + self.window_samples
        if start < 0 or stop > int(recording.shape[1]):
            raise ValueError("Broderick2018 词窗口超出 canonical continuous signal 范围。")

        if self.temporal_shift_samples == 0:
            window = recording[:, start:stop].astype(np.float32, copy=True)
        elif self.shift_mode == "preroll":
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
                        f"Broderick2018 Pre-Roll 缓冲区长度不足：raw_start={raw_start}, buf_len={buf_len}"
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
                raise ValueError("Broderick2018 连续信号长度不足以执行非重叠时间错位对照。")
            shifted_start = (start + self.temporal_shift_samples) % valid_span
            if abs(shifted_start - start) < self.window_samples:
                step = (
                    self.window_samples
                    if self.temporal_shift_samples > 0
                    else -self.window_samples
                )
                shifted_start = (shifted_start + step) % valid_span
            start = int(shifted_start)
            stop = start + self.window_samples
            window = recording[:, start:stop].astype(np.float32, copy=True)

        effective_baseline = min(self._baseline_samples, int(window.shape[1]))
        if effective_baseline > 0:
            baseline = window[:, :effective_baseline].mean(axis=1, keepdims=True)
            window -= baseline
        np.clip(window, -self._clamp, self._clamp, out=window)
        return window

    def read_meg(self, index: int) -> np.ndarray:
        """与历史 read_meg 别名保持接口一致，实际返回头皮 EEG 窗口。"""
        return self.read_eeg(index)

    def __getitem__(self, index: int) -> dict:
        idx = int(index)
        word = self._words[idx]
        return {
            "eeg": torch.from_numpy(self.read_eeg(idx)),
            "text_embedding": self._embedding_tensors[word],
            "subject_index": self._subject_index_tensors[idx],
            "sentence_index": self._sentence_index_tensors[idx],
            "onset_seconds": self._onset_second_tensors[idx],
            "word": word,
            "event_id": self._event_ids[idx],
            "recording_id": self._recording_ids[idx],
        }


__all__ = [
    "BIOSEMI128_CHANNEL_NAMES",
    "CHANNEL_NAMES_SHA256",
    "CORE_SCOPE",
    "DATASET_ID",
    "DEFAULT_PREROLL_SECONDS",
    "DRYAD_DOI",
    "EEG_CHANNEL_COUNT",
    "ENGLISH_REFERENCE_TOKENS",
    "PERSISTED_COLUMNS",
    "RAW_DATASET_NAME",
    "RUNS",
    "RUN_SPLIT_SALT",
    "SOURCE_SAMPLING_RATE_HZ",
    "SUBJECTS",
    "TARGET_SAMPLING_RATE_HZ",
    "VALID_TEMPORAL_SHIFT_MODES",
    "WINDOW_CONTRACT_VERSION",
    "Broderick2018WordDataset",
    "add_runtime_context_uid",
    "biosemi128_channel_positions",
    "build_event_table",
    "build_protocol_assets",
    "build_run_split_manifest",
    "build_signal_manifest",
    "build_text_product",
    "channel_names_sha256",
    "deterministic_run_split",
    "discover_subjects",
    "ensure_recording_signals",
    "extract_word_window",
    "load_event_table",
    "load_run_split",
    "material_word_view",
    "materialize_recording_signal",
    "normalize_english_word",
    "recording_onset_to_sample",
    "recording_paths",
    "recording_window_bounds",
    "signal_preprocessing_contract",
    "subject_to_raw_name",
    "run_to_raw_index",
    "validate_signal_manifest",
    "validate_text_manifest",
    "word_window_contract",
    "write_protocol_assets",
    "write_run_split_manifest",
]
