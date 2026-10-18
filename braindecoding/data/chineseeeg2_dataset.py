"""ChineseEEG2 被动听《小王子》的词级解码数据接口。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from braindecoding.data.text import (
    ensure_text_embedding_cache,
    load_text_embedding_cache,
    normalize_word,
    text_embedding_signature,
)
from braindecoding.data.derived import (
    canonical_event_table as _canonical_event_table,
    restore_legacy_columns,
    signal_cache_directory,
)
from braindecoding.events import add_core_event_columns


CANONICAL_AUXILIARY_COLUMNS = {
    "chapter": "章节",
    "audio_file": "音频文件",
    "material_line": "材料行号",
    "text_start_position": "文本起始位置",
    "text_stop_position": "文本结束位置",
    "audio_start_seconds": "音频开始时间",
    "audio_stop_seconds": "音频结束时间",
    "timestamp_qc_prompt": "时间戳质检提示",
    "timestamp_source": "时间戳来源",
    "run": "运行编号",
    "voice_version": "声音版本",
    "speaker_gender": "朗读者性别",
    "小说": "小说",
    "word_index": "词序号",
    "chapter_word_index": "章内词序号",
    "source_word_id": "原始词编号",
    "vhdr_relpath": "BrainVision头文件相对路径",
    "recording_duration_seconds": "记录时长秒",
    "source_sample_count": "源采样点数",
    "aligned_start_source_sample": "对齐开始源采样点",
    "aligned_stop_source_sample": "对齐结束源采样点",
    "window_start_source_sample": "窗口开始源采样点",
    "window_stop_source_sample": "窗口结束源采样点",
    "eligibility_stop_source_sample": "资格窗口结束源采样点",
    "window_start_target_sample": "窗口开始目标采样点",
    "window_stop_target_sample": "窗口结束目标采样点",
    "window_start_seconds": "窗口开始时间",
    "window_stop_seconds": "窗口结束时间",
    "source_sampling_rate_hz": "源采样率Hz",
    "target_sampling_rate_hz": "目标采样率Hz",
    "target_sample_count": "目标采样点数",
    "window_complete": "窗口完整",
    "alignment_status": "对齐状态",
    "automatic_qc_pass": "自动质检通过",
    "synchronization_status": "同步状态",
    "mapping_status": "映射状态",
    "text_verification_status": "文本核验状态",
    "hardware_delay_corrected": "已修正硬件延迟",
    "manual_excluded": "人工排除",
    "manual_exclusion_reason": "人工排除原因",
    "primary_inclusion": "主分析纳入",
    "source_sentence_uid": "原始上下文编号",
    "context_chunk_index": "上下文分块序号",
    "context_grouping": "上下文定义",
    "context_boundary_reason": "上下文边界原因",
    "in_chineseeeg2_littleprince50": "属于小王子前50词表",
    "vocabulary_rank": "候选词频排名",
    "timing_status": "时间状态",
    "timing_source_clock": "时间来源时钟",
}

_CANONICAL_REDUNDANT_COLUMNS = {
    "aligned_start_seconds",
    "aligned_stop_seconds",
    "split_group_id",
}


def canonical_event_table(event_table: pd.DataFrame) -> pd.DataFrame:
    """生成 ChineseEEG2 中文优先的 canonical 磁盘事件表。"""
    return _canonical_event_table(
        event_table,
        CANONICAL_AUXILIARY_COLUMNS,
        redundant_columns=_CANONICAL_REDUNDANT_COLUMNS,
    )


def restore_event_table_columns(event_table: pd.DataFrame) -> pd.DataFrame:
    """在内存中恢复旧训练消费者使用的英文列。"""
    result = restore_legacy_columns(
        event_table,
        CANONICAL_AUXILIARY_COLUMNS,
        derived_aliases={
            "aligned_start_seconds": "开始时间",
            "aligned_stop_seconds": "结束时间",
        },
    )
    if "split_group_id" not in result and "chapter" in result:
        result["split_group_id"] = result["chapter"].map(
            lambda chapter: f"chapter-{int(chapter):02d}"
        )
    return result


记录名规则 = re.compile(
    r"(?P<subject>sub-\d+)_ses-littleprince_task-lis_run-(?P<run>\d+)_eeg\.vhdr$"
)


def 解析布尔值(value) -> bool:
    """将对齐表中的多种布尔写法统一为真或假。"""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", "nan", ""}:
        return False
    raise ValueError(f"无法解析布尔值：{value!r}")


def 文件摘要(path) -> str:
    """计算文件的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 章节划分映射(split_config) -> dict[int, str]:
    """展开并检查互不重叠的章节级划分。"""
    mapping = {}
    for split, key in (
        ("train", "train_chapters"),
        ("val", "val_chapters"),
        ("test", "test_chapters"),
    ):
        for chapter in split_config[key]:
            chapter = int(chapter)
            if chapter in mapping:
                raise ValueError(f"章节 {chapter} 同时出现在多个数据划分中。")
            mapping[chapter] = split
    return mapping


def 记录路径(dataset_config, subject_id, run) -> Path:
    """根据受试者和运行编号定位预处理 BrainVision 头文件。"""
    root = Path(dataset_config["root"])
    relative = (
        Path(dataset_config["preprocessed_dir"])
        / str(subject_id)
        / "ses-littleprince"
        / "eeg"
        / f"{subject_id}_ses-littleprince_task-lis_run-{int(run)}_eeg.vhdr"
    )
    return root / relative


def 检查记录结构(vhdr_path) -> dict:
    """只读取旁车元数据，检查采样率、通道与电极坐标。"""
    vhdr_path = Path(vhdr_path)
    match = 记录名规则.search(vhdr_path.name)
    if match is None:
        raise ValueError(f"无法解析记录文件名：{vhdr_path.name}")
    metadata_path = vhdr_path.with_suffix(".json")
    data_path = vhdr_path.with_suffix(".eeg")
    channels_path = Path(str(vhdr_path).replace("_eeg.vhdr", "_channels.tsv"))
    electrodes_path = (
        vhdr_path.parent
        / f"{match.group('subject')}_ses-littleprince_space-CapTrak_electrodes.tsv"
    )
    required = (vhdr_path, data_path, metadata_path, channels_path, electrodes_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"记录缺少必要文件：{missing}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    channels = pd.read_csv(channels_path, sep="\t")
    electrodes = pd.read_csv(electrodes_path, sep="\t").set_index("name")
    channel_names = tuple(channels["name"].astype(str))
    missing_positions = sorted(set(channel_names) - set(electrodes.index.astype(str)))
    if missing_positions:
        raise ValueError(f"以下通道缺少电极坐标：{missing_positions[:5]}")
    positions = electrodes.loc[list(channel_names), ["x", "y", "z"]].to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(positions).all():
        raise ValueError(f"电极坐标包含非有限值：{electrodes_path}")
    if not channels["status"].astype(str).str.lower().eq("good").all():
        raise ValueError(f"预处理记录仍包含非 good 通道：{channels_path}")
    if not channels["type"].astype(str).str.upper().eq("EEG").all():
        raise ValueError(f"预处理记录包含非 EEG 通道：{channels_path}")
    channel_sampling_rates = pd.to_numeric(
        channels["sampling_frequency"], errors="raise"
    )
    if not np.allclose(
        channel_sampling_rates,
        float(metadata["SamplingFrequency"]),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(f"通道旁车采样率与记录元数据不一致：{channels_path}")
    filter_hz = (
        float(pd.to_numeric(channels["low_cutoff"], errors="raise").iloc[0]),
        float(pd.to_numeric(channels["high_cutoff"], errors="raise").iloc[0]),
    )
    if not (
        np.allclose(channels["low_cutoff"].astype(float), filter_hz[0])
        and np.allclose(channels["high_cutoff"].astype(float), filter_hz[1])
    ):
        raise ValueError(f"同一记录内的通道滤波合同不一致：{channels_path}")
    bytes_per_sample = 4 * len(channel_names)
    if data_path.stat().st_size % bytes_per_sample:
        raise ValueError(f"脑电二进制文件长度无法整除通道数：{data_path}")
    sample_count = data_path.stat().st_size // bytes_per_sample
    metadata_sample_count = int(
        round(
            float(metadata["SamplingFrequency"])
            * float(metadata["RecordingDuration"])
        )
    )
    if abs(sample_count - metadata_sample_count) > 1:
        raise ValueError(
            f"脑电二进制与旁车时长推算相差超过一个采样点：{data_path}"
        )
    return {
        "subject_id": match.group("subject"),
        "run": int(match.group("run")),
        "sampling_rate_hz": float(metadata["SamplingFrequency"]),
        "recording_duration_seconds": float(metadata["RecordingDuration"]),
        "sample_count": int(sample_count),
        "channel_names": channel_names,
        "channel_positions": positions,
        "channel_count": len(channel_names),
        "preprocessed_filter_hz": filter_hz,
    }


def 训练集高频词(event_table, vocabulary_size) -> tuple[tuple[str, ...], dict[str, int]]:
    """只用训练章节，并让每个声音版本的词事件只计数一次。"""
    train = event_table[
        event_table["split"].eq("train") & event_table["is_trainable"].astype(bool)
    ].copy()
    canonical = (
        train.sort_values(
            ["voice_version", "chapter", "chapter_word_index", "subject_id"]
        )
        .drop_duplicates(["voice_version", "source_word_id"])
        .reset_index(drop=True)
    )
    counts = canonical["normalized_word"].value_counts().to_dict()
    ordered = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    vocabulary = tuple(word for word, _ in ordered[: int(vocabulary_size)])
    if len(vocabulary) != int(vocabulary_size):
        raise ValueError(
            f"训练集只有 {len(vocabulary)} 个有效词，无法冻结 {vocabulary_size} 词候选集。"
        )
    return vocabulary, {word: int(counts[word]) for word in vocabulary}


def 读取实际朗读逐行文本(workbook_path, config) -> dict[tuple[int, int], str]:
    """读取工作簿中的实际阅读文本，保留用于判断语境边界的标点。"""
    lines = pd.read_excel(
        workbook_path,
        sheet_name=config.get("actual_reading_line_sheet_name", "逐行对照"),
    )
    required = {"音频编号", "原表行号", "实际阅读"}
    missing = sorted(required - set(lines.columns))
    if missing:
        raise ValueError(f"实际朗读逐行页缺少字段：{missing}")
    lines = lines[
        pd.to_numeric(lines["音频编号"], errors="raise").between(1, 27)
    ].copy()
    lines["音频编号"] = pd.to_numeric(lines["音频编号"], errors="raise").astype(int)
    lines["原表行号"] = pd.to_numeric(lines["原表行号"], errors="raise").astype(int)
    if lines.duplicated(["音频编号", "原表行号"]).any():
        raise ValueError("实际朗读逐行页包含重复的章节与材料行。")
    texts = {
        (int(row["音频编号"]), int(row["原表行号"])): str(row["实际阅读"])
        for _, row in lines.iterrows()
        if pd.notna(row["实际阅读"])
    }
    if not texts:
        raise ValueError("实际朗读逐行页没有可用文本。")
    return texts


def 构建长度受控语义片段(frame, texts, settings):
    """按实际阅读标点跨行合并，并用词数和时间约束片段长度。"""
    preferred = int(settings["preferred_words"])
    maximum = int(settings["maximum_words"])
    maximum_seconds = float(settings["maximum_seconds"])
    if not 0 < preferred <= maximum or maximum_seconds <= 0:
        raise ValueError("语境词数和时间上限必须为正数，优先长度不能超过上限。")

    frame = frame.copy()
    frame["context_boundary_reason"] = ""
    missing_lines = set()
    for _, recording in frame.groupby(
        ["subject_id", "run", "chapter"], sort=False
    ):
        chunks = []
        pending = []
        # 材料行只用于恢复标点，不再直接充当注意力边界。
        for _, line in recording.groupby("material_line", sort=False):
            first = line.iloc[0]
            key = (int(first["chapter"]), int(first["material_line"]))
            if key not in texts:
                missing_lines.add(key)
                continue
            text = texts[key].rstrip().rstrip("”’」』\"'）)]】").rstrip()
            indices = line.index.tolist()
            start = (
                float(frame.loc[pending[0], "aligned_start_seconds"])
                if pending
                else float(first["aligned_start_seconds"])
            )
            span = float(line.iloc[-1]["aligned_stop_seconds"]) - start
            if pending and (
                len(pending) + len(indices) > maximum or span > maximum_seconds
            ):
                chunks.append((pending, "length_limit_at_clause_boundary"))
                pending = []
            for index in indices:
                start_index = pending[0] if pending else index
                span = float(frame.loc[index, "aligned_stop_seconds"]) - float(
                    frame.loc[start_index, "aligned_start_seconds"]
                )
                if pending and (len(pending) >= maximum or span > maximum_seconds):
                    chunks.append((pending, "hard_length_limit"))
                    pending = []
                pending.append(index)
            strong = text.endswith(tuple("。！？!?；;"))
            soft = text.endswith(tuple("，,、"))
            if strong or (soft and len(pending) >= preferred):
                reason = (
                    "sentence_end"
                    if strong
                    else "clause_end_after_preferred_length"
                )
                chunks.append((pending, reason))
                pending = []
        if pending:
            chunks.append((pending, "recording_end"))
        for number, (indices, reason) in enumerate(chunks):
            first = frame.loc[indices[0]]
            uid = (
                f"{first['subject_id']}|run-{int(first['run'])}"
                f"|chapter-{int(first['chapter'])}|semantic-{number}"
            )
            frame.loc[indices, "sentence_uid"] = uid
            frame.loc[indices, "context_chunk_index"] = number
            frame.loc[indices, "context_boundary_reason"] = reason
    if missing_lines:
        preview = sorted(missing_lines)[:5]
        raise ValueError(f"语境逐行文本缺少词事件对应的材料行：{preview}")
    frame["context_grouping"] = "bounded_semantic_v1"
    return frame


def _构建单一实际朗读事件表(config) -> pd.DataFrame:
    """从一份实际朗读工作簿构建一秒事件。"""
    if config.get("additional_filter") is not None:
        raise ValueError("预处理输入不允许在本任务中重复滤波。")
    workbook_path = Path(config["alignment_path"])
    if not workbook_path.exists():
        raise FileNotFoundError(f"实际朗读工作簿不存在：{workbook_path}")
    context_grouping = str(
        config.get(
            "context_grouping", "actual_reading_row_then_contiguous_chunks"
        )
    )
    supported_groupings = {
        "actual_reading_row_then_contiguous_chunks",
        "bounded_semantic_v1",
    }
    if context_grouping not in supported_groupings:
        raise ValueError(f"不支持的实际朗读语境分组方式：{context_grouping}")
    line_texts = None
    if context_grouping == "bounded_semantic_v1":
        settings = config.get("semantic_context")
        if not isinstance(settings, dict):
            raise ValueError("长度受控语义片段缺少 semantic_context 参数。")
        line_texts = 读取实际朗读逐行文本(workbook_path, settings)
    frame = pd.read_excel(
        workbook_path, sheet_name=config.get("actual_reading_sheet_name", "词级时间戳")
    )
    required = {
        "音频编号", "音频文件", "原表行号", "词",
        "文本起始位置", "文本结束位置",
        "开始时间（秒）", "结束时间（秒）", "质检提示", "时间戳来源",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"实际朗读工作簿缺少字段：{missing}")
    frame = frame[pd.to_numeric(frame["音频编号"], errors="raise").between(1, 27)].copy()
    if len(frame) != int(config.get("expected_source_event_count", len(frame))):
        raise ValueError(f"工作簿词事件数量漂移：实际 {len(frame)}")
    if frame[list(required - {"质检提示"})].isna().any().any():
        raise ValueError("实际朗读词事件包含缺失值。")
    if frame.duplicated(
        ["音频编号", "原表行号", "文本起始位置", "文本结束位置", "词"]
    ).any():
        raise ValueError("实际朗读工作簿包含重复词事件。")

    frame = frame.rename(columns={
        "音频编号": "chapter", "音频文件": "audio_file",
        "原表行号": "material_line", "词": "word",
        "文本起始位置": "text_start_position",
        "文本结束位置": "text_stop_position",
        "开始时间（秒）": "audio_start_seconds",
        "结束时间（秒）": "audio_stop_seconds",
        "质检提示": "timestamp_qc_prompt", "时间戳来源": "timestamp_source",
    })
    frame["chapter"] = frame["chapter"].astype(int)
    frame["run"] = [int(f"1{chapter}") if chapter <= 14 else int(f"2{chapter - 14}")
                    for chapter in frame["chapter"]]
    voice_version = str(config.get("voice_version", "f1"))
    voice_contracts = {
        "f1": ({"sub-01", "sub-02", "sub-03", "sub-04"}, "female"),
        "m1": ({"sub-05", "sub-06", "sub-07", "sub-08"}, "male"),
    }
    if voice_version not in voice_contracts:
        raise ValueError(f"不支持的实际朗读声音版本：{voice_version}")
    allowed_subjects, speaker_gender = voice_contracts[voice_version]
    frame["voice_version"] = voice_version
    frame["speaker_gender"] = speaker_gender
    frame["小说"] = "littleprince"
    frame["word_index"] = frame.groupby("chapter", sort=False).cumcount() + 1
    frame["chapter_word_index"] = frame["word_index"]
    frame["source_word_id"] = [
        f"actual-{voice_version}-c{chapter:02d}-w{word_index:05d}"
        for chapter, word_index in zip(frame["chapter"], frame["word_index"])
    ]
    frame["normalized_word"] = frame["word"].map(normalize_word)
    if frame["normalized_word"].eq("").any():
        raise ValueError("实际朗读词事件包含空标签，无法在不丢弃事件的前提下训练。")
    split_mapping = 章节划分映射(config["split"])
    frame["split"] = frame["chapter"].map(split_mapping)
    if frame["split"].isna().any():
        raise ValueError("实际朗读工作簿包含未分配数据划分的章节。")

    subjects = tuple(sorted(str(value) for value in config["subjects"]))
    if not subjects:
        raise ValueError("实际朗读事件至少需要配置一名受试者。")
    unsupported = sorted(set(subjects) - allowed_subjects)
    if unsupported:
        raise ValueError(
            f"{voice_version} 实际朗读工作簿不能用于合同外受试者：{unsupported}"
        )
    frame = pd.concat(
        [frame.assign(subject_id=subject_id) for subject_id in subjects],
        ignore_index=True,
    )

    schemas = {}
    paths = {}
    pairs = frame[["subject_id", "run"]].drop_duplicates()
    for subject_id, run in pairs.itertuples(index=False, name=None):
        key = (str(subject_id), int(run))
        path = 记录路径(config, *key)
        schemas[key] = 检查记录结构(path)
        paths[key] = path
    reference = next(iter(schemas.values()))
    expected_rate = float(config["source_sampling_rate_hz"])
    expected_filter = tuple(float(value) for value in config["source_preprocessed_filter_hz"])
    for key, schema in schemas.items():
        if not math.isclose(schema["sampling_rate_hz"], expected_rate):
            raise ValueError(f"记录 {key} 的采样率不符合合同。")
        if schema["channel_names"] != reference["channel_names"]:
            raise ValueError(f"记录 {key} 的通道顺序不一致。")
        if not np.allclose(schema["channel_positions"], reference["channel_positions"], rtol=0, atol=1e-7):
            raise ValueError(f"记录 {key} 的电极坐标不一致。")
        if not np.allclose(schema["preprocessed_filter_hz"], expected_filter, rtol=0, atol=1e-8):
            raise ValueError(f"记录 {key} 的滤波合同不一致。")

    keys = list(zip(frame["subject_id"].astype(str), frame["run"].astype(int)))
    frame["recording_id"] = [f"{subject_id}|run-{run}" for subject_id, run in keys]
    frame["vhdr_relpath"] = [
        paths[key].relative_to(Path(config["root"])).as_posix() for key in keys
    ]
    frame["recording_duration_seconds"] = [schemas[key]["recording_duration_seconds"] for key in keys]
    frame["source_sample_count"] = [schemas[key]["sample_count"] for key in keys]
    clock_offset = float(config.get("timestamp_to_eeg_offset_seconds", 0.0))
    frame["aligned_start_seconds"] = pd.to_numeric(frame["audio_start_seconds"], errors="raise") + clock_offset
    frame["aligned_stop_seconds"] = pd.to_numeric(frame["audio_stop_seconds"], errors="raise") + clock_offset
    frame["aligned_start_source_sample"] = np.floor(frame["aligned_start_seconds"] * expected_rate + 0.5).astype(int)
    frame["aligned_stop_source_sample"] = np.floor(frame["aligned_stop_seconds"] * expected_rate + 0.5).astype(int)
    start_offset = float(config.get("window_start_offset_seconds", 0.0))
    window_seconds = float(config["window_seconds"])
    eligibility_seconds = float(config.get("eligibility_window_seconds", window_seconds))
    if not math.isclose(eligibility_seconds, window_seconds, rel_tol=0, abs_tol=1e-12):
        raise ValueError("不丢词一秒方案要求资格窗口与输入窗口同为一秒。")
    target_rate = float(config["target_sampling_rate_hz"])
    start_shift = int(round(start_offset * expected_rate))
    source_window = int(round(window_seconds * expected_rate))
    target_window = int(round(window_seconds * target_rate))
    frame["window_start_source_sample"] = frame["aligned_start_source_sample"] + start_shift
    frame["window_stop_source_sample"] = frame["window_start_source_sample"] + source_window
    frame["eligibility_stop_source_sample"] = frame["window_stop_source_sample"]
    frame["window_start_target_sample"] = np.floor(
        frame["window_start_source_sample"] * target_rate / expected_rate + 0.5
    ).astype(int)
    frame["window_stop_target_sample"] = frame["window_start_target_sample"] + target_window
    frame["window_start_seconds"] = frame["window_start_source_sample"] / expected_rate
    frame["window_stop_seconds"] = frame["window_start_seconds"] + window_seconds
    frame["source_sampling_rate_hz"] = expected_rate
    frame["target_sampling_rate_hz"] = target_rate
    frame["target_sample_count"] = target_window
    frame["window_complete"] = (
        frame["window_start_source_sample"].ge(0)
        & frame["window_stop_source_sample"].le(frame["source_sample_count"])
    )
    excluded_chapters = {
        int(value) for value in config.get("excluded_chapters", ())
    }
    unknown_exclusions = sorted(excluded_chapters - set(split_mapping))
    if unknown_exclusions:
        raise ValueError(f"显式剔除列表包含未知章节：{unknown_exclusions}")
    chapter_excluded = frame["chapter"].isin(excluded_chapters)
    unexpected_incomplete = ~frame["window_complete"] & ~chapter_excluded
    if unexpected_incomplete.any():
        raise ValueError(
            f"有 {int(unexpected_incomplete.sum())} 个未获授权剔除事件的一秒窗口超出 EEG 记录，"
            "为避免静默丢词已停止。"
        )

    frame["alignment_status"] = "actual_reading_text_qwen_estimated_timing"
    frame["automatic_qc_pass"] = frame["timestamp_qc_prompt"].isna()
    frame["synchronization_status"] = "audio_seconds_used_as_eeg_seconds_by_user_request"
    frame["mapping_status"] = "chapter_to_subject_recording_direct"
    frame["text_verification_status"] = "actual_reading_workbook"
    frame["hardware_delay_corrected"] = False
    exclusion_reason = str(
        config.get("excluded_chapter_reason", "配置明确剔除整章")
    )
    frame["manual_excluded"] = chapter_excluded
    frame["manual_exclusion_reason"] = np.where(
        chapter_excluded, exclusion_reason, ""
    )
    frame["primary_inclusion"] = ~chapter_excluded
    frame["is_trainable"] = ~chapter_excluded & frame["window_complete"]
    frame["exclusion_reason"] = np.where(
        chapter_excluded, exclusion_reason, ""
    )
    frame["split_group_id"] = frame["chapter"].map(lambda value: f"chapter-{int(value):02d}")
    frame["source_sentence_uid"] = (
        frame["subject_id"] + "|run-" + frame["run"].astype(str)
        + "|actual-line-" + frame["material_line"].astype(int).astype(str)
    )
    frame = frame.sort_values(["chapter", "word_index", "aligned_start_source_sample"]).reset_index(drop=True)
    frame["context_chunk_index"] = frame.groupby("source_sentence_uid", sort=False).cumcount() // int(config.get("max_context_words", 128))
    frame["sentence_uid"] = frame["source_sentence_uid"] + "|chunk-" + frame["context_chunk_index"].astype(str)
    frame["context_grouping"] = "actual_reading_row_then_contiguous_chunks"
    if context_grouping == "bounded_semantic_v1":
        frame = 构建长度受控语义片段(
            frame, line_texts, config["semantic_context"]
        )
    frame["event_id"] = frame["subject_id"] + "|" + frame["voice_version"] + "|" + frame["source_word_id"]
    if frame["event_id"].duplicated().any():
        raise ValueError("实际朗读事件表包含重复 event_id。")
    vocabulary, _ = 训练集高频词(frame, config["vocabulary_size"])
    rank = {word: index + 1 for index, word in enumerate(vocabulary)}
    frame["in_chineseeeg2_littleprince50"] = frame["normalized_word"].isin(vocabulary)
    frame["vocabulary_rank"] = frame["normalized_word"].map(rank).fillna(0).astype(int)
    frame["timing_status"] = "actual_reading_audio_clock_used_as_eeg_clock"
    frame["timing_source_clock"] = "audio_file_zero_seconds_direct_to_preprocessed_eeg"
    return frame


def 构建实际朗读事件表(config) -> pd.DataFrame:
    """构建单声音或多声音实际朗读事件，并统一冻结训练词表。"""
    sources = config.get("actual_reading_sources")
    if not sources:
        return add_event_contract(_构建单一实际朗读事件表(config))
    if not isinstance(sources, list) or len(sources) < 2:
        raise ValueError("多声音实际朗读合同至少需要两项事件源。")

    frames = []
    reference_schema = None
    for source in sources:
        source_config = dict(config)
        source_config.pop("actual_reading_sources", None)
        source_config.pop("expected_trainable_counts", None)
        source_config.update(source)
        current = _构建单一实际朗读事件表(source_config)
        first_source = Path(config["root"]) / str(current["vhdr_relpath"].iloc[0])
        schema = 检查记录结构(first_source)
        if reference_schema is None:
            reference_schema = schema
        elif (
            schema["channel_names"] != reference_schema["channel_names"]
            or not np.allclose(
                schema["channel_positions"],
                reference_schema["channel_positions"],
                rtol=0,
                atol=1e-7,
            )
        ):
            raise ValueError("不同声音事件源的通道或电极坐标合同不一致。")
        frames.append(current)

    frame = pd.concat(frames, ignore_index=True)
    configured_subjects = tuple(sorted(str(value) for value in config["subjects"]))
    observed_subjects = tuple(sorted(frame["subject_id"].astype(str).unique()))
    if observed_subjects != configured_subjects:
        raise ValueError(
            f"多声音事件源受试者与总合同不一致：{observed_subjects} != {configured_subjects}"
        )
    if frame["event_id"].duplicated().any():
        raise ValueError("多声音实际朗读事件表包含重复 event_id。")

    vocabulary, _ = 训练集高频词(frame, config["vocabulary_size"])
    rank = {word: index + 1 for index, word in enumerate(vocabulary)}
    frame["in_chineseeeg2_littleprince50"] = frame["normalized_word"].isin(
        vocabulary
    )
    frame["vocabulary_rank"] = (
        frame["normalized_word"].map(rank).fillna(0).astype(int)
    )
    frame = frame.sort_values(
        ["voice_version", "chapter", "word_index", "subject_id"]
    ).reset_index(drop=True)
    return add_event_contract(frame)


def add_event_contract(event_table: pd.DataFrame) -> pd.DataFrame:
    """保持现有事件顺序，追加 ChineseEEG2 中文公共字段。"""
    material_ids = pd.Series(
        [
            f"ChineseEEG2|littleprince|chapter-{int(chapter):02d}|voice-{voice}"
            for chapter, voice in zip(
                event_table["chapter"], event_table["voice_version"]
            )
        ],
        index=event_table.index,
    )
    split_units = event_table["chapter"].map(
        lambda chapter: f"ChineseEEG2|littleprince|chapter-{int(chapter):02d}"
    )
    return add_core_event_columns(
        event_table,
        material_ids=material_ids,
        split_units=split_units,
        start_times=event_table["aligned_start_seconds"],
        end_times=event_table["aligned_stop_seconds"],
    )


def 审计事件表(event_table, config) -> dict:
    """验证划分、候选词、窗口与记录结构，并生成可复核摘要。"""
    trainable = event_table[event_table["is_trainable"].astype(bool)].copy()
    split_mapping = 章节划分映射(config["split"])
    expected_split = event_table["chapter"].map(split_mapping)
    if not expected_split.equals(event_table["split"]):
        raise ValueError("事件表的章节与 split 标签不一致。")
    split_chapters = {
        split: sorted(trainable.loc[trainable["split"].eq(split), "chapter"].unique().astype(int).tolist())
        for split in ("train", "val", "test")
    }
    split_sets = [set(value) for value in split_chapters.values()]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("章节划分存在重叠。")

    vocabulary, train_counts = 训练集高频词(
        event_table, config["vocabulary_size"]
    )
    ranked = (
        event_table.loc[event_table["vocabulary_rank"].gt(0), ["normalized_word", "vocabulary_rank"]]
        .drop_duplicates()
        .sort_values("vocabulary_rank")
    )
    if tuple(ranked["normalized_word"]) != vocabulary:
        raise ValueError("事件表中的候选词顺序与训练集词频不一致。")

    expected_counts = config.get("expected_trainable_counts")
    counts = {
        split: int(trainable["split"].eq(split).sum())
        for split in ("train", "val", "test")
    }
    if expected_counts is not None and counts != {
        key: int(value) for key, value in expected_counts.items()
    }:
        raise ValueError(f"有效事件计数漂移：实际 {counts}，预期 {expected_counts}。")
    group_sizes = trainable.groupby(["split", "sentence_uid"]).size()
    source_specs = config.get("actual_reading_sources") or [config]
    source_paths = [Path(source["alignment_path"]).resolve() for source in source_specs]
    source_hashes = {str(path): 文件摘要(path) for path in source_paths}
    return {
        "status": "event_table_validated",
        "context_grouping": config["context_grouping"],
        "semantic_context": config.get("semantic_context"),
        "raw_eeg_loaded": False,
        "test_eeg_loaded": False,
        "alignment_path": (
            str(source_paths[0])
            if len(source_paths) == 1
            else [str(path) for path in source_paths]
        ),
        "alignment_sha256": (
            source_hashes[str(source_paths[0])]
            if len(source_paths) == 1
            else source_hashes
        ),
        "row_count": int(len(event_table)),
        "primary_inclusion_count": int(event_table["primary_inclusion"].sum()),
        "manual_excluded_count": int(event_table["manual_excluded"].sum()),
        "manual_excluded_chapters": sorted(
            event_table.loc[
                event_table["manual_excluded"].astype(bool), "chapter"
            ].unique().astype(int).tolist()
        ),
        "incomplete_window_count": int((~event_table["window_complete"]).sum()),
        "incomplete_window_excluded_count": int(
            (
                ~event_table["window_complete"]
                & event_table["manual_excluded"].astype(bool)
            ).sum()
        ),
        "automatic_qc_pass_count": int(event_table["automatic_qc_pass"].sum()),
        "hardware_delay_corrected_count": int(
            event_table["hardware_delay_corrected"].sum()
        ),
        "alignment_status_counts": {
            str(key): int(value)
            for key, value in event_table["alignment_status"].value_counts().items()
        },
        "synchronization_status_counts": {
            str(key): int(value)
            for key, value in event_table[
                "synchronization_status"
            ].value_counts().items()
        },
        "mapping_status_counts": {
            str(key): int(value)
            for key, value in event_table["mapping_status"].value_counts().items()
        },
        "text_verification_status_counts": {
            str(key): int(value)
            for key, value in event_table[
                "text_verification_status"
            ].value_counts().items()
        },
        "trainable_counts": counts,
        "recording_counts": {
            split: int(trainable.loc[trainable["split"].eq(split), "recording_id"].nunique())
            for split in ("train", "val", "test")
        },
        "subject_count": int(trainable["subject_id"].nunique()),
        "split_unit": "chapter",
        "split_seed": int(config["split"].get("seed", 42)),
        "split_chapters": split_chapters,
        "split_overlap_count": 0,
        "source_sampling_rate_hz": float(config["source_sampling_rate_hz"]),
        "target_sampling_rate_hz": float(config["target_sampling_rate_hz"]),
        "source_preprocessed_filter_hz": [
            float(value) for value in config["source_preprocessed_filter_hz"]
        ],
        "additional_filter": config.get("additional_filter"),
        "window_start_offset_seconds": float(
            config.get("window_start_offset_seconds", 0.0)
        ),
        "window_seconds": float(config["window_seconds"]),
        "eligibility_window_seconds": float(
            config.get("eligibility_window_seconds", config["window_seconds"])
        ),
        "target_sample_count": int(trainable["target_sample_count"].iloc[0]),
        "candidate_provenance": config.get("evaluation_vocabulary_policy", "train_chapters_unique_voice_word_events"),
        "evaluation_vocabulary": list(vocabulary),
        "training_canonical_counts": train_counts,
        "observed_vocabulary_size": {
            split: int(
                trainable.loc[
                    trainable["split"].eq(split)
                    & trainable["in_chineseeeg2_littleprince50"].astype(bool),
                    "normalized_word",
                ].nunique()
            )
            for split in ("train", "val", "test")
        },
        "evaluation_query_counts": {
            split: int(
                (
                    trainable["split"].eq(split)
                    & trainable["in_chineseeeg2_littleprince50"].astype(bool)
                ).sum()
            )
            for split in ("train", "val", "test")
        },
        "maximum_context_group_size": {
            split: int(group_sizes.loc[split].max())
            for split in ("train", "val", "test")
        },
        "exclusion_counts": event_table.loc[
            ~event_table["is_trainable"].astype(bool), "exclusion_reason"
        ].value_counts().to_dict(),
        "timing_status": str(event_table["timing_status"].iloc[0]),
    }


def 保存事件表(event_table, path, audit) -> Path:
    """原子写入事件表及其审计旁车文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    event_table.to_csv(temporary, index=False, encoding="utf-8")
    os.replace(temporary, path)
    audit = dict(audit)
    audit["event_table_sha256"] = 文件摘要(path)
    audit_path = path.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def 载入候选词(event_table_path) -> tuple[str, ...]:
    """从事件表审计文件读取冻结的训练集候选词。"""
    audit_path = Path(event_table_path).with_suffix(".audit.json")
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        manifest_path = Path(event_table_path).parent / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"事件表缺少审计文件或 derived manifest：{event_table_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit = manifest.get("audit", {})
    vocabulary = tuple(str(value) for value in audit["evaluation_vocabulary"])
    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError("冻结候选词包含重复项。")
    return vocabulary


def 载入事件表(path, split=None, trainable_only=True, subjects=None) -> pd.DataFrame:
    """载入事件表，并按需要筛选数据划分和有效事件。"""
    table = restore_event_table_columns(pd.read_csv(path, low_memory=False))
    required = {
        "split",
        "is_trainable",
        "normalized_word",
        "sentence_uid",
        "vhdr_relpath",
        "window_start_target_sample",
        "target_sample_count",
        "subject_id",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"事件表缺少字段：{missing}")
    if trainable_only:
        table = table[table["is_trainable"].map(解析布尔值)]
    if subjects is not None:
        subject_set = {str(value) for value in subjects}
        table = table[table["subject_id"].astype(str).isin(subject_set)]
    if split is not None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"未知数据划分：{split}")
        table = table[table["split"].eq(split)]
    return table.reset_index(drop=True)


def _预处理签名(config, source_path) -> dict:
    source_path = Path(source_path)
    schema = 检查记录结构(source_path)
    data_path = source_path.with_suffix(".eeg")
    return {
        "source_path": str(source_path.resolve()),
        "source_size": source_path.stat().st_size,
        "source_mtime_ns": source_path.stat().st_mtime_ns,
        "data_size": data_path.stat().st_size,
        "data_mtime_ns": data_path.stat().st_mtime_ns,
        "source_sampling_rate_hz": float(config["source_sampling_rate_hz"]),
        "target_sampling_rate_hz": float(config["target_sampling_rate_hz"]),
        "source_is_preprocessed": True,
        "source_preprocessed_filter_hz": [
            float(value) for value in config["source_preprocessed_filter_hz"]
        ],
        "additional_filter": config.get("additional_filter"),
        "scaler": str(config.get("scaler", "RobustScaler")),
        "channel_names": list(schema["channel_names"]),
    }


def _签名摘要(signature) -> str:
    payload = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def 处理后记录路径(source_path, config, cache_dir) -> Path:
    """按源文件和预处理合同生成不会冲突的缓存路径。"""
    signature = _预处理签名(config, source_path)
    directory = signal_cache_directory(
        cache_dir,
        source_path,
        by_subject=bool(config.get("signal_cache_subject_subdirectories", False)),
    )
    return directory / f"{Path(source_path).stem}_{_签名摘要(signature)}.npy"


def _记录缓存有效(cache_path, signature) -> bool:
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    if not cache_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return False
        array = np.load(cache_path, mmap_mode="r")
        return array.shape == tuple(metadata["shape"]) and array.dtype == np.float32
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


class _直接BrainVision记录:
    """直接读取本数据集固定格式的 BrainVision 浮点二进制文件。"""

    def __init__(self, path):
        path = Path(path)
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        section = None
        common = {}
        binary = {}
        channels = {}
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if section == "Common Infos":
                common[key] = value
            elif section == "Binary Infos":
                binary[key] = value
            elif section == "Channel Infos" and key.startswith("Ch"):
                channels[int(key[2:])] = value

        if common.get("DataFormat") != "BINARY":
            raise ValueError(f"只支持二进制 BrainVision 数据：{path}")
        if common.get("DataOrientation") != "MULTIPLEXED":
            raise ValueError(f"只支持按采样点交错存储的 BrainVision 数据：{path}")
        if binary.get("BinaryFormat") != "IEEE_FLOAT_32":
            raise ValueError(f"只支持 IEEE_FLOAT_32 BrainVision 数据：{path}")
        channel_count = int(common["NumberOfChannels"])
        if sorted(channels) != list(range(1, channel_count + 1)):
            raise ValueError(f"BrainVision 通道编号不连续：{path}")

        self.ch_names = []
        scales = []
        unit_scales = {"V": 1.0, "mV": 1e-3, "µV": 1e-6, "μV": 1e-6, "uV": 1e-6}
        for index in range(1, channel_count + 1):
            fields = channels[index].split(",")
            if len(fields) < 4:
                raise ValueError(f"BrainVision 通道定义不完整：Ch{index}")
            self.ch_names.append(fields[0].replace(r"\1", ","))
            resolution = float(fields[2] or 1.0)
            unit = fields[3].strip()
            if unit not in unit_scales:
                raise ValueError(f"不支持的 BrainVision 单位：{unit}")
            scales.append(resolution * unit_scales[unit])

        data_path = path.parent / common["DataFile"]
        bytes_per_sample = 4 * channel_count
        if data_path.stat().st_size % bytes_per_sample:
            raise ValueError(f"BrainVision 二进制文件长度无法整除通道数：{data_path}")
        self.n_times = data_path.stat().st_size // bytes_per_sample
        self.info = {"sfreq": 1e6 / float(common["SamplingInterval"])}
        self._scales = np.asarray(scales, dtype=np.float32)
        self._data = np.memmap(
            data_path,
            dtype="<f4",
            mode="r",
            shape=(self.n_times, channel_count),
        )

    def get_data(self, picks):
        indices = [self.ch_names.index(str(value)) for value in picks]
        data = np.asarray(self._data[:, indices].T, dtype=np.float32)
        return data * self._scales[indices, None]

    def close(self):
        mmap = getattr(self._data, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _读取脑电记录(path):
    """使用可审计的固定格式读取器，避开通用读取器的高启动开销。"""
    return _直接BrainVision记录(path)


def _缩放通道(data, scaler):
    if scaler == "RobustScaler":
        quartiles = np.percentile(data, [25.0, 50.0, 75.0], axis=1)
        center = quartiles[1].astype(np.float32)
        scale = (quartiles[2] - quartiles[0]).astype(np.float32)
    elif scaler == "StandardScaler":
        center = data.mean(axis=1, dtype=np.float64).astype(np.float32)
        scale = data.std(axis=1, dtype=np.float64).astype(np.float32)
    elif str(scaler).lower() in {"none", "null"}:
        return data
    else:
        raise ValueError(f"未知脑电缩放方式：{scaler}")
    scale[scale == 0] = 1.0
    data -= center[:, None]
    data /= scale[:, None]
    return data


def 物化记录缓存(source_path, config, cache_dir, force=False) -> Path:
    """读取预处理 BrainVision 记录，降采样并按通道缩放。"""
    from scipy import signal

    source_path = Path(source_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = _预处理签名(config, source_path)
    cache_path = 处理后记录路径(source_path, config, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not force and _记录缓存有效(cache_path, signature):
        return cache_path

    raw = _读取脑电记录(source_path)
    try:
        source_rate = float(raw.info["sfreq"])
        expected_rate = float(config["source_sampling_rate_hz"])
        if not math.isclose(source_rate, expected_rate):
            raise ValueError(
                f"记录采样率为 {source_rate} Hz，配置要求 {expected_rate} Hz：{source_path}"
            )
        schema = 检查记录结构(source_path)
        if int(raw.n_times) != int(schema["sample_count"]):
            raise ValueError(
                f"记录采样点数与旁车元数据不一致：{raw.n_times} != {schema['sample_count']}"
            )
        channel_names = tuple(signature["channel_names"])
        missing = sorted(set(channel_names) - set(raw.ch_names))
        if missing:
            raise ValueError(f"BrainVision 记录缺少通道：{missing[:5]}")
        target_rate = float(config["target_sampling_rate_hz"])
        divisor = math.gcd(round(source_rate), round(target_rate))
        up = round(target_rate) // divisor
        down = round(source_rate) // divisor
        chunks = []
        chunk_size = int(config.get("materialization_channel_chunk", 32))
        scaler = str(config.get("scaler", "RobustScaler"))
        for start in range(0, len(channel_names), chunk_size):
            names = channel_names[start : start + chunk_size]
            data = raw.get_data(picks=list(names)).astype(np.float32)
            if not math.isclose(source_rate, target_rate):
                data = signal.resample_poly(data, up, down, axis=1).astype(
                    np.float32, copy=False
                )
            chunks.append(_缩放通道(data, scaler))
        output = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    finally:
        raw.close()
    if not np.isfinite(output).all():
        raise ValueError(f"记录预处理后包含非有限值：{source_path}")

    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, output, allow_pickle=False)
    os.replace(temporary, cache_path)
    cache_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "status": "materialized",
                "signature": signature,
                "shape": list(output.shape),
                "dtype": "float32",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return cache_path


def 确保记录缓存(event_table, dataset_config, cache_dir, force=False):
    """仅物化事件表实际引用的记录，默认不会触碰测试记录。"""
    root = Path(dataset_config["root"])
    paths = event_table["vhdr_relpath"].drop_duplicates().tolist()
    outputs = []
    for index, relative_path in enumerate(paths, start=1):
        source_path = root / str(relative_path)
        print(f"脑电缓存 {index}/{len(paths)}：{source_path.name}")
        outputs.append(
            物化记录缓存(source_path, dataset_config, cache_dir, force=force)
        )
    return outputs


@lru_cache(maxsize=16)
def _打开处理后记录(path: str):
    return np.load(path, mmap_mode="r")


class ChineseEEG2LittlePrinceWordDataset(Dataset):
    """将词起点后的固定脑电窗口与冻结文本向量配对。"""

    def __init__(
        self,
        event_table,
        dataset_config,
        eeg_cache_dir,
        embedding_cache_path,
        text_embedding_config,
        zero_eeg=False,
    ):
        self.table = event_table.reset_index(drop=True).copy()
        self.dataset_config = dict(dataset_config)
        self.eeg_cache_dir = Path(eeg_cache_dir)
        self.zero_eeg = bool(zero_eeg)
        self.embedding_map = load_text_embedding_cache(
            embedding_cache_path,
            expected_signature=text_embedding_signature(text_embedding_config),
        )
        missing = sorted(
            set(self.table["normalized_word"].astype(str)) - set(self.embedding_map)
        )
        if missing:
            raise ValueError(f"文本向量缓存缺少 {len(missing)} 个词，例如 {missing[:5]}。")

        root = Path(self.dataset_config["root"])
        first_source = root / str(self.table["vhdr_relpath"].iloc[0])
        schema = 检查记录结构(first_source)
        self.channel_names = schema["channel_names"]
        self.channel_positions = schema["channel_positions"]
        self.channel_count = schema["channel_count"]
        self.window_samples = int(self.table["target_sample_count"].iloc[0])
        self.clamp = self.dataset_config.get("clamp", 5.0)

        self.subject_ids = tuple(sorted(str(value) for value in self.dataset_config["subjects"]))
        self.subject_to_index = {
            subject_id: index for index, subject_id in enumerate(self.subject_ids)
        }
        self.subject_count = len(self.subject_ids)
        unknown_subjects = sorted(
            set(self.table["subject_id"].astype(str)) - set(self.subject_ids)
        )
        if unknown_subjects:
            raise ValueError(f"事件表包含未知受试者：{unknown_subjects}")

        sentence_uids = self.table["sentence_uid"].astype(str).tolist()
        sentence_to_index = {
            value: index for index, value in enumerate(dict.fromkeys(sentence_uids))
        }
        self.sentence_indices = np.asarray(
            [sentence_to_index[value] for value in sentence_uids], dtype=np.int64
        )
        self.recording_cache_paths = {}
        for relative_path in self.table["vhdr_relpath"].drop_duplicates():
            source_path = root / str(relative_path)
            cache_path = 处理后记录路径(
                source_path, self.dataset_config, self.eeg_cache_dir
            )
            signature = _预处理签名(self.dataset_config, source_path)
            if not self.zero_eeg and not _记录缓存有效(cache_path, signature):
                raise FileNotFoundError(
                    f"脑电缓存尚未物化：{cache_path}。请先运行 prepare --with-eeg-cache。"
                )
            self.recording_cache_paths[str(relative_path)] = cache_path

    def __len__(self):
        return len(self.table)

    def 读取脑电(self, index) -> np.ndarray:
        row = self.table.iloc[int(index)]
        if self.zero_eeg:
            return np.zeros(
                (self.channel_count, self.window_samples), dtype=np.float32
            )
        cache_path = self.recording_cache_paths[str(row["vhdr_relpath"])]
        recording = _打开处理后记录(str(cache_path))
        start = int(row["window_start_target_sample"])
        stop = start + self.window_samples
        if start < 0 or stop > recording.shape[1]:
            raise IndexError(
                f"事件窗口越界：{row['event_id']} [{start}, {stop}) / {recording.shape[1]}"
            )
        window = np.array(recording[:, start:stop], dtype=np.float32, copy=True)
        if self.clamp is not None:
            np.clip(window, -float(self.clamp), float(self.clamp), out=window)
        return window

    def __getitem__(self, index):
        row = self.table.iloc[int(index)]
        word = str(row["normalized_word"])
        subject_id = str(row["subject_id"])
        return {
            "eeg": torch.from_numpy(self.读取脑电(index)),
            "text_embedding": torch.from_numpy(
                np.array(self.embedding_map[word], dtype=np.float32, copy=True)
            ),
            "subject_index": torch.tensor(
                self.subject_to_index[subject_id], dtype=torch.long
            ),
            "sentence_index": torch.tensor(
                self.sentence_indices[int(index)], dtype=torch.long
            ),
            "word": word,
            "event_id": str(row["event_id"]),
            "recording_id": str(row["recording_id"]),
        }


# 标准英文接口别名（与 pallier2025 保持一致，同时保留中文函数名向后兼容）
file_sha256 = 文件摘要
parse_bool = 解析布尔值
chapter_split_mapping = 章节划分映射
validate_recording_structure = 检查记录结构
recording_paths = 记录路径
read_actual_reading_lines = 读取实际朗读逐行文本
build_bounded_semantic_segments = 构建长度受控语义片段
build_event_table = 构建实际朗读事件表
train_top_vocabulary = 训练集高频词
save_event_table = 保存事件表
load_event_table = 载入事件表
load_candidate_vocabulary = 载入候选词
frozen_vocabulary = 载入候选词
audit_event_table = 审计事件表
processed_recording_path = 处理后记录路径
materialize_recording_signal = 物化记录缓存
ensure_recording_signals = 确保记录缓存
