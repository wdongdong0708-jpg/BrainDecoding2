"""候选词表及语言参考分布的纯函数工具。"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping


def _normalize_word(value, normalize):
    word = normalize(value) if normalize is not None else str(value).strip()
    return str(word).strip()


def build_frequency_vocabulary(
    words: Iterable,
    size: int,
    normalize: Callable | None = None,
):
    """只统计调用方显式传入的词，按频次降序、词典序升序生成词表。"""
    if int(size) != size or int(size) <= 0:
        raise ValueError("词表大小必须是正整数。")
    counts = Counter()
    for value in words:
        word = _normalize_word(value, normalize)
        if word:
            counts[word] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(word for word, _ in ranked[: int(size)])


def validate_frozen_vocabulary(
    vocabulary: Iterable,
    expected_size: int | None = None,
    normalize: Callable | None = None,
):
    """验证冻结词表非空、无重复，并保留调用方给定的顺序。"""
    words = tuple(_normalize_word(value, normalize) for value in vocabulary)
    if not words:
        raise ValueError("冻结词表不能为空。")
    if any(not word for word in words):
        raise ValueError("冻结词表不能包含空词。")
    if len(words) != len(set(words)):
        raise ValueError("冻结词表包含重复词。")
    if expected_size is not None and len(words) != int(expected_size):
        raise ValueError(
            f"冻结词表大小不一致：期望 {int(expected_size)}，实际 {len(words)}。"
        )
    return words


def build_vocabulary_metadata(
    vocabulary: Iterable,
    source_split: str,
    policy: str,
    normalization: str,
    word_counts: Mapping | None = None,
):
    """构造可写入 JSON 的冻结词表来源说明，不修改 checkpoint。"""
    words = validate_frozen_vocabulary(vocabulary)
    counts = word_counts or {}
    return {
        "vocabulary": list(words),
        "size": int(len(words)),
        "source_split": str(source_split),
        "policy": str(policy),
        "normalization": str(normalization),
        "word_counts": {word: int(counts.get(word, 0)) for word in words},
    }
