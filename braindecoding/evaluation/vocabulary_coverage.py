"""候选词表及语言参考分布的纯函数工具。"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Union


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


def build_l5_sentence_matched_vocabulary(
    train_words: Sequence[str] | Mapping[str, int],
    test_segments: Sequence[Sequence[str] | Mapping],
    size: Union[int, str, None] = "full",
    min_sentence_length: int = 5,
    pad_with_train_freq: bool = False,
) -> tuple[list[str], list[int]]:
    """模式一（路径 B1）：构造满足句长 L >= min_sentence_length 且训练可见的句表同源闭集词表 V。

    参数:
        train_words: 训练集所有词 token 列表或训练词频映射（用于统计训练词频并判定训练可见性）。
        test_segments: 测试集每个片段的词 token 列表（或含 target_words 的片段字典）。
        size:
            - "full"（或 0 / None）：直接取所有满足 L >= 5 且全词在训练集出现过的测试长句的词汇并集 V_full（约 120~140 词）；
            - int（如 50, 100, 250）：通过贪心集合覆盖在 L >= 5 长句集内选取紧凑闭集词表。
        min_sentence_length: 流式滚动物理最小句长准入（R=2 时 L >= 2R + 1 = 5）。
        pad_with_train_freq:
            - 当 size > |V_full|（如 size=250）时，是否用训练集最高频词补齐至 size。

    返回:
        vocab_list: 排序后的闭集候选词表 V
        covered_seg_indices: 在该词表 V 下满足 L >= 5 且严格 0% OOV 的测试片段索引列表
    """
    if isinstance(train_words, Mapping):
        train_counts: Counter = Counter()
        for w, c in train_words.items():
            norm_w = str(w).strip().lower()
            if norm_w and int(c) > 0:
                train_counts[norm_w] += int(c)
    else:
        train_counts = Counter(
            str(w).strip().lower()
            for w in train_words
            if w is not None and str(w).strip()
        )

    # 1. 筛选满足 L >= min_sentence_length 且所有词均在训练词库中的标准连续长句
    valid_segments: dict[int, list[str]] = {}
    for idx, seg in enumerate(test_segments):
        raw_tokens = seg.get("target_words", []) if isinstance(seg, Mapping) else seg
        tokens = [
            str(w).strip().lower()
            for w in raw_tokens
            if w is not None and str(w).strip()
        ]
        if len(tokens) < int(min_sentence_length):
            continue
        if all(w in train_counts for w in tokens):
            valid_segments[idx] = tokens

    # 2. 若请求基准全词表 V_full（size == "full" 或 None 或 <= 0）
    full_l5_vocab_set = {w for tokens in valid_segments.values() for w in tokens}
    is_full_size = False
    if size is None:
        is_full_size = True
    elif isinstance(size, str):
        stripped = size.strip().lower()
        if stripped in ("full", "all", "") or not stripped.lstrip("-").isdigit():
            is_full_size = True
        else:
            is_full_size = int(stripped) <= 0
    elif int(size) <= 0:
        is_full_size = True

    if is_full_size:
        vocab_list = sorted(
            full_l5_vocab_set, key=lambda w: (-train_counts.get(w, 0), w)
        )
        covered_indices = sorted(valid_segments.keys())
        return vocab_list, covered_indices

    # 3. 若请求指定大小的子词表（如 size = 50 或 100），使用贪心集合覆盖最大化 0% OOV 长句数量
    target_size = int(size.strip() if isinstance(size, str) else size)
    if target_size >= len(full_l5_vocab_set):
        vocab_set = set(full_l5_vocab_set)
        if pad_with_train_freq and target_size > len(vocab_set):
            for word, _ in sorted(train_counts.items(), key=lambda x: (-x[1], x[0])):
                if len(vocab_set) >= target_size:
                    break
                vocab_set.add(word)
        vocab_list = sorted(vocab_set, key=lambda w: (-train_counts.get(w, 0), w))[
            :target_size
        ]
        covered_indices = sorted(valid_segments.keys())
        return vocab_list, covered_indices

    vocab_set: set[str] = set()
    selected_seg_ids: set[int] = set()

    while len(vocab_set) < target_size:
        best_seg_id = -1
        best_score = (-1.0, -1.0)
        for idx, tokens in valid_segments.items():
            if idx in selected_seg_ids:
                continue
            new_words = set(tokens) - vocab_set
            if len(vocab_set) + len(new_words) > target_size:
                continue
            # 贪心准则：每新增 1 个词表词所覆盖的句内总词位数最高，次选句内平均训练词频最高
            coverage_efficiency = len(tokens) / (len(new_words) + 0.05)
            mean_freq = sum(train_counts[w] for w in tokens) / len(tokens)
            score = (coverage_efficiency, mean_freq)
            if score > best_score:
                best_score = score
                best_seg_id = idx

        if best_seg_id == -1:
            break
        selected_seg_ids.add(best_seg_id)
        vocab_set.update(valid_segments[best_seg_id])

    # 若贪心选取后仍有少量剩余词表名额（如差 1~3 个词满 50/100），用其余长句中的高频词补满
    if len(vocab_set) < target_size:
        remaining_l5_words = sorted(
            full_l5_vocab_set - vocab_set,
            key=lambda w: (-train_counts.get(w, 0), w),
        )
        for word in remaining_l5_words:
            if len(vocab_set) >= target_size:
                break
            vocab_set.add(word)

    vocab_list = sorted(vocab_set, key=lambda w: (-train_counts.get(w, 0), w))[
        :target_size
    ]
    vocab_lookup = set(vocab_list)
    covered_indices = [
        idx
        for idx, tokens in valid_segments.items()
        if all(w in vocab_lookup for w in tokens)
    ]
    return vocab_list, covered_indices
