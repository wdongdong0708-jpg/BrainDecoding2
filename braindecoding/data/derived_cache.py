"""可重建数据产品的事件表落盘与清单工具。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import re
from pathlib import Path

import pandas as pd

from braindecoding.events import CORE_EVENT_COLUMNS, validate_event_table


CORE_LEGACY_COLUMNS = {
    "event_id": "事件编号",
    "subject_id": "受试者",
    "recording_id": "记录编号",
    "word": "词",
    "normalized_word": "标准词",
    "sentence_uid": "上下文编号",
    "split": "数据划分",
    "is_trainable": "是否可训练",
    "exclusion_reason": "排除原因",
}


def canonical_cache_paths(dataset: str) -> dict[str, str]:
    """返回 canonical 实验唯一允许使用的 derived 数据位置。"""
    dataset = str(dataset)
    if dataset == "chineseeeg2_littleprince":
        return {
            "event_table": "data/cache/derived/chineseeeg2_littleprince/events/events.csv",
            "eeg_dir": "data/cache/derived/chineseeeg2_littleprince/signals/eeg_50hz",
            "text_embeddings": (
                "data/cache/derived/chineseeeg2_littleprince/text/"
                "mengzi_t5_layer_0_5/embeddings.npz"
            ),
        }
    if dataset == "pallier2025":
        return {
            "event_table": "data/cache/derived/pallier2025/events/events.csv",
            "meg_dir": "data/cache/derived/pallier2025/signals/meg_50hz",
            "text_embeddings": (
                "data/cache/derived/pallier2025/text/t5_large_layer_0_5/embeddings.npz"
            ),
        }
    raise ValueError(f"没有为数据集定义 canonical derived 路径：{dataset}")


def signal_cache_directory(cache_dir, source_path, by_subject=False) -> Path:
    """按配置把 canonical 信号缓存放入受试者子目录。"""
    cache_dir = Path(cache_dir)
    if not by_subject:
        return cache_dir
    candidates = [
        match.group(0)
        for part in Path(source_path).parts
        for match in [re.search(r"sub-\d+", part)]
        if match is not None
    ]
    if not candidates:
        raise ValueError(f"无法从源记录路径解析受试者：{source_path}")
    return cache_dir / candidates[-1]


def sha256_file(path) -> str:
    """流式计算文件 SHA-256，不修改源文件。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_manifest_path(event_path) -> Path:
    """根据事件表文件名返回同目录 manifest 路径。"""
    event_path = Path(event_path)
    suffix = event_path.stem.removeprefix("events")
    name = f"manifest{suffix}.json" if suffix else "manifest.json"
    return event_path.parent / name


def stable_sha256(value) -> str:
    """计算 JSON 可序列化对象的稳定 SHA-256。"""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_event_table(
    table: pd.DataFrame,
    auxiliary_columns: dict[str, str],
    *,
    redundant_columns=(),
) -> pd.DataFrame:
    """生成中文优先的磁盘事件表，不保留核心字段的英文别名。"""
    validate_event_table(table)
    redundant = set(redundant_columns) | set(CORE_LEGACY_COLUMNS)
    remaining = [
        column
        for column in table.columns
        if column not in CORE_EVENT_COLUMNS and column not in redundant
    ]
    missing_mappings = sorted(
        column for column in remaining if column not in auxiliary_columns
    )
    if missing_mappings:
        raise ValueError(f"事件表存在未定义的持久化字段：{missing_mappings}")

    renamed = table[remaining].rename(columns=auxiliary_columns)
    duplicates = sorted(set(CORE_EVENT_COLUMNS) & set(renamed.columns))
    if duplicates:
        raise ValueError(f"辅助字段与核心中文字段重名：{duplicates}")
    if renamed.columns.duplicated().any():
        duplicate_names = sorted(
            set(renamed.columns[renamed.columns.duplicated()].astype(str))
        )
        raise ValueError(f"辅助中文字段重名：{duplicate_names}")
    result = pd.concat(
        [table.loc[:, list(CORE_EVENT_COLUMNS)].copy(), renamed], axis=1
    )
    if any(column in result.columns for column in CORE_LEGACY_COLUMNS):
        raise RuntimeError("canonical 事件表仍包含核心英文字段。")
    return result


def restore_legacy_columns(
    table: pd.DataFrame,
    auxiliary_columns: dict[str, str],
    *,
    derived_aliases: dict[str, str] | None = None,
) -> pd.DataFrame:
    """读取中文 canonical 表时，仅在内存中恢复旧消费者需要的英文列。"""
    if not set(CORE_EVENT_COLUMNS).issubset(table.columns):
        return table
    result = table.copy()
    for legacy_column, chinese_column in CORE_LEGACY_COLUMNS.items():
        if legacy_column not in result.columns:
            result[legacy_column] = result[chinese_column]
    for legacy_column, chinese_column in auxiliary_columns.items():
        if legacy_column not in result.columns and chinese_column in result.columns:
            result[legacy_column] = result[chinese_column]
    for legacy_column, chinese_column in (derived_aliases or {}).items():
        if legacy_column not in result.columns:
            result[legacy_column] = result[chinese_column]
    return result


def git_commit(project_root) -> str | None:
    """读取当前 Git commit；仓库不可用时返回空值。"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_tracked_dirty(project_root) -> bool | None:
    """只检查 tracked 文件是否有未提交变化。"""
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def build_event_manifest(
    table: pd.DataFrame,
    *,
    dataset: str,
    event_path,
    project_root,
    source_raw_roots,
    artifact_dependencies,
    context_definition,
    window_contract,
    source_contract,
    audit,
    builder_sources=(),
    additional_fields=None,
) -> dict:
    """为已经落盘的 canonical 事件表构建稳定 manifest。"""
    validate_event_table(table)
    event_path = Path(event_path)
    trainable = table["是否可训练"].astype(bool)
    split_counts = {
        split: {
            "event_count": int(table["数据划分"].eq(split).sum()),
            "trainable_event_count": int(
                (table["数据划分"].eq(split) & trainable).sum()
            ),
        }
        for split in ("train", "val", "test")
    }
    manifest = {
        "schema_version": 1,
        "dataset": str(dataset),
        "subjects": list(dict.fromkeys(table["受试者"].astype(str))),
        "subject_order": list(dict.fromkeys(table["受试者"].astype(str))),
        "material_count": int(table["材料编号"].nunique()),
        "recording_count": int(table["记录编号"].nunique()),
        "event_count": int(len(table)),
        "trainable_event_count": int(trainable.sum()),
        "split_counts": split_counts,
        "columns": list(table.columns),
        "source_raw_roots": [str(Path(value)) for value in source_raw_roots],
        "artifact_dependencies": list(artifact_dependencies),
        "context_definition": context_definition,
        "window_contract": window_contract,
        "builder_git_commit": git_commit(project_root),
        "builder_git_dirty": git_tracked_dirty(project_root),
        "builder_sources": [
            {
                "path": Path(path)
                .resolve()
                .relative_to(Path(project_root).resolve())
                .as_posix(),
                "sha256": sha256_file(path),
            }
            for path in builder_sources
        ],
        "event_table": event_path.name,
        "event_table_sha256": sha256_file(event_path),
        "source_contract_sha256": stable_sha256(source_contract),
        "audit": audit,
    }
    additional_fields = dict(additional_fields or {})
    overlap = sorted(set(manifest) & set(additional_fields))
    if overlap:
        raise ValueError(f"事件 manifest 扩展字段覆盖保留字段：{overlap}")
    manifest.update(additional_fields)
    manifest["manifest_sha256"] = stable_sha256(manifest)
    return manifest


def write_event_product(
    table: pd.DataFrame,
    event_path,
    *,
    manifest_kwargs,
) -> tuple[Path, Path, dict]:
    """原子写入 canonical 事件 CSV 和同目录 manifest。"""
    event_path = Path(event_path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = event_path.with_suffix(event_path.suffix + ".tmp")
    table.to_csv(temporary_path, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary_path, event_path)
    manifest = build_event_manifest(
        table,
        event_path=event_path,
        **manifest_kwargs,
    )
    manifest_path = event_manifest_path(event_path)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return event_path, manifest_path, manifest


def validate_event_product(event_path, manifest_path=None) -> dict:
    """验证事件文件与 manifest 的摘要和自摘要。"""
    event_path = Path(event_path)
    manifest_path = Path(manifest_path or event_manifest_path(event_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(event_path) != manifest.get("event_table_sha256"):
        raise ValueError(f"事件表 SHA 与 manifest 不一致：{event_path}")
    manifest_without_sha = dict(manifest)
    recorded_sha = manifest_without_sha.pop("manifest_sha256", None)
    if stable_sha256(manifest_without_sha) != recorded_sha:
        raise ValueError(f"manifest 自摘要不一致：{manifest_path}")
    return manifest
