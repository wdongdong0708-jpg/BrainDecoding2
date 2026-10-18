"""词级数据集共享的中文事件合同。"""

from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype


CORE_EVENT_COLUMNS = (
    "事件编号",
    "受试者",
    "记录编号",
    "材料编号",
    "划分单元",
    "记录内序号",
    "词",
    "标准词",
    "开始时间",
    "结束时间",
    "上下文编号",
    "数据划分",
    "是否可训练",
    "排除原因",
)


_DIRECT_COLUMN_MAPPING = {
    "事件编号": "event_id",
    "受试者": "subject_id",
    "记录编号": "recording_id",
    "词": "word",
    "标准词": "normalized_word",
    "上下文编号": "sentence_uid",
    "数据划分": "split",
    "是否可训练": "is_trainable",
    "排除原因": "exclusion_reason",
}


def text_fingerprint(words) -> str:
    """按给定顺序计算标准词序列的稳定 SHA-256 摘要。"""
    payload = json.dumps(
        [str(word) for word in words],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compare_text_sequences(first_words, second_words) -> dict:
    """比较两组标准词，并返回长度、差异位置和内容指纹。"""
    first = tuple(str(word) for word in first_words)
    second = tuple(str(word) for word in second_words)
    first_difference = next(
        (index for index, pair in enumerate(zip(first, second)) if pair[0] != pair[1]),
        None,
    )
    if first_difference is None and len(first) != len(second):
        first_difference = min(len(first), len(second))
    difference_count = sum(left != right for left, right in zip(first, second))
    difference_count += abs(len(first) - len(second))
    return {
        "identical": first == second,
        "first_length": len(first),
        "second_length": len(second),
        "difference_count": difference_count,
        "first_difference_index": first_difference,
        "first_fingerprint": text_fingerprint(first),
        "second_fingerprint": text_fingerprint(second),
    }


def align_text_sequences(left_words, right_words) -> dict:
    """用词级 Levenshtein 对齐统计替换、插入和删除。"""
    left = tuple(str(word) for word in left_words)
    right = tuple(str(word) for word in right_words)
    distances = [list(range(len(right) + 1))]
    for left_index, left_word in enumerate(left, start=1):
        row = [left_index] + [0] * len(right)
        previous = distances[-1]
        for right_index, right_word in enumerate(right, start=1):
            row[right_index] = min(
                previous[right_index] + 1,
                row[right_index - 1] + 1,
                previous[right_index - 1] + int(left_word != right_word),
            )
        distances.append(row)

    operations = []
    left_index = len(left)
    right_index = len(right)
    while left_index or right_index:
        current = distances[left_index][right_index]
        if (
            left_index
            and right_index
            and left[left_index - 1] == right[right_index - 1]
            and current == distances[left_index - 1][right_index - 1]
        ):
            operations.append("match")
            left_index -= 1
            right_index -= 1
        elif (
            left_index
            and right_index
            and current == distances[left_index - 1][right_index - 1] + 1
        ):
            # 并列最短路径优先替换，使操作计数保持确定。
            operations.append("substitution")
            left_index -= 1
            right_index -= 1
        elif left_index and current == distances[left_index - 1][right_index] + 1:
            operations.append("deletion")
            left_index -= 1
        else:
            operations.append("insertion")
            right_index -= 1
    operations.reverse()

    first_left_index = None
    first_right_index = None
    left_cursor = 0
    right_cursor = 0
    for operation in operations:
        if operation != "match" and first_left_index is None:
            first_left_index = left_cursor
            first_right_index = right_cursor
        if operation in {"match", "substitution", "deletion"}:
            left_cursor += 1
        if operation in {"match", "substitution", "insertion"}:
            right_cursor += 1

    edit_distance = distances[-1][-1]
    return {
        "left_length": len(left),
        "right_length": len(right),
        "equal": left == right,
        "matches": operations.count("match"),
        "substitutions": operations.count("substitution"),
        "insertions": operations.count("insertion"),
        "deletions": operations.count("deletion"),
        "edit_distance": edit_distance,
        "normalized_edit_distance": edit_distance / max(len(left), len(right), 1),
        "first_edit_left_index": first_left_index,
        "first_edit_right_index": first_right_index,
        "left_fingerprint": text_fingerprint(left),
        "right_fingerprint": text_fingerprint(right),
    }


def add_record_order(
    table: pd.DataFrame,
    recording_column: str = "记录编号",
    output_column: str = "记录内序号",
) -> pd.DataFrame:
    """保持现有行序，为每条记录增加从零开始的连续事件序号。"""
    if recording_column not in table:
        raise ValueError(f"事件表缺少记录字段：{recording_column}")
    result = table.copy()
    result[output_column] = (
        result.groupby(recording_column, sort=False, dropna=False).cumcount().astype(int)
    )
    return result


def add_core_event_columns(
    table: pd.DataFrame,
    *,
    material_ids,
    split_units,
    start_times,
    end_times,
) -> pd.DataFrame:
    """保留全部旧列和行序，在内存中追加中文公共事件合同。"""
    missing = sorted(set(_DIRECT_COLUMN_MAPPING.values()) - set(table.columns))
    if missing:
        raise ValueError(f"旧事件表缺少生成中文合同所需字段：{missing}")

    result = table.copy()
    for core_column, legacy_column in _DIRECT_COLUMN_MAPPING.items():
        result[core_column] = result[legacy_column]
    result["材料编号"] = material_ids
    result["划分单元"] = split_units
    result["开始时间"] = start_times
    result["结束时间"] = end_times
    result = add_record_order(result)
    validate_event_table(result)
    return result


def validate_event_table(table: pd.DataFrame) -> tuple[str, ...]:
    """验证中文核心合同；历史空排除原因为警告，其余违约为错误。"""
    missing = sorted(set(CORE_EVENT_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"事件表缺少中文核心字段：{missing}")

    identifiers = table["事件编号"]
    if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
        raise ValueError("事件编号不能为空。")
    if identifiers.duplicated().any():
        raise ValueError("事件编号必须全局唯一。")

    for column in ("受试者", "记录编号", "材料编号", "划分单元", "上下文编号"):
        values = table[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{column}不能为空。")

    splits = table["数据划分"]
    if splits.isna().any() or not set(splits.astype(str)).issubset(
        {"train", "val", "test"}
    ):
        raise ValueError("数据划分只允许 train/val/test。")
    split_counts = table.groupby("划分单元", dropna=False)["数据划分"].nunique(
        dropna=False
    )
    if split_counts.gt(1).any():
        raise ValueError("同一个划分单元不能属于多个数据划分。")

    order = pd.to_numeric(table["记录内序号"], errors="coerce")
    if order.isna().any() or not np.equal(order, np.floor(order)).all():
        raise ValueError("记录内序号必须是整数。")
    ordered = table.assign(_record_order=order.astype(int))
    for recording_id, group in ordered.groupby("记录编号", sort=False, dropna=False):
        actual = group["_record_order"].tolist()
        if actual != list(range(len(group))):
            raise ValueError(
                f"记录 {recording_id} 的记录内序号必须从 0 开始严格连续。"
            )

    starts = pd.to_numeric(table["开始时间"], errors="coerce")
    stops = pd.to_numeric(table["结束时间"], errors="coerce")
    if starts.isna().any() or stops.isna().any():
        raise ValueError("开始时间和结束时间必须是有效秒数。")
    if not np.isfinite(starts).all() or not np.isfinite(stops).all():
        raise ValueError("开始时间和结束时间必须是有限秒数。")
    if starts.gt(stops).any():
        raise ValueError("开始时间不能晚于结束时间。")

    trainable = table["是否可训练"]
    if not is_bool_dtype(trainable.dtype):
        raise ValueError("是否可训练必须保持布尔语义。")

    material_fingerprints = []
    for keys, group in table.groupby(
        ["材料编号", "受试者", "记录编号"], sort=False, dropna=False
    ):
        # 完整排除的 recording 没有可用于材料合同的神经事件；保留其原始
        # annotation 供审计，但不让已冻结的显式排除阻断其他可训练记录。
        if not group["是否可训练"].astype(bool).any():
            continue
        words = group.sort_values("记录内序号", kind="stable")["标准词"]
        material_fingerprints.append((str(keys[0]), text_fingerprint(words)))
    fingerprint_table = pd.DataFrame(
        material_fingerprints, columns=["材料编号", "文本指纹"]
    )
    fingerprint_counts = (
        fingerprint_table.groupby("材料编号")["文本指纹"].nunique()
        if not fingerprint_table.empty
        else pd.Series(dtype=int)
    )
    inconsistent_materials = sorted(
        fingerprint_counts[fingerprint_counts.gt(1)].index.astype(str)
    )
    if inconsistent_materials:
        raise ValueError(
            f"材料编号对应多个文本指纹：{inconsistent_materials[:10]}"
        )

    empty_reason = table["排除原因"].isna() | table["排除原因"].astype(
        str
    ).str.strip().eq("")
    missing_reason_count = int((~trainable.astype(bool) & empty_reason).sum())
    messages = []
    if missing_reason_count:
        message = (
            f"发现 {missing_reason_count} 个不可训练事件没有排除原因；"
            "已保留历史值，未自动填写。"
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        messages.append(message)
    return tuple(messages)
