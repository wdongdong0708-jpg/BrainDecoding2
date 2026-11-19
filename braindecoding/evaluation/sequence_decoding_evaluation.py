"""S3 序列解码协议多维语义评价指标与配对显著性分析器。

核心功能：
1. 同步输出 6 维语义与词位评价指标（全部统一以百分比 % 呈现）：
   - 字面精确匹配：BLEU-1 (%), WER (%), CER (%)
   - 词位准确率：Word Acc (%)，并拆解为 实义词（Content Words）Acc (%) 与 功能词（Function Words）Acc (%)
   - 连续语义相似度：BERTScore (%)（Precision / Recall / F1）
2. 覆盖四级延迟预算（Word-Only / Local r=0,1,2 / Full-Context）、Brain-Only / LM-Only 基线、
   L>=5 闭集与多尺度词表扫描、跨被试聚合（k-scaling）及配对材料块置换/自助检验。
"""


from __future__ import annotations

import copy
from collections import Counter, defaultdict
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F_nn

from braindecoding.evaluation.statistical_significance import (
    edit_distance,
    holm_adjust,
    paired_block_statistics,
    sentence_metrics,
)
from braindecoding.training.training_runtime import save_json



STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    "S0": "Word-Only",
    "S0_clean": "Word-Only",
    "Word-Only": "Word-Only",
    "K0": "Rolling-K0",
    "K0_clean": "Rolling-K0",
    "Rolling-K0": "Rolling-K0",
    "K1": "Rolling-K1",
    "K1_clean": "Rolling-K1",
    "Rolling-K1": "Rolling-K1",
    "S3_clean": "Rolling-K2",
    "S3": "Rolling-K2",
    "K2": "Rolling-K2",
    "Rolling-K2": "Rolling-K2",
    "F": "Full-Context",
    "F_clean": "Full-Context",
    "Full-Context": "Full-Context",
    "LM-Only": "LM-Only",
    "LM-Only_clean": "LM-Only",
    "K0_shifted_neg75s": "Shifted-K0 (-75s)",
    "Shifted-K0 (-75s)": "Shifted-K0 (-75s)",
    "S3_shifted_neg75s": "Shifted-K2 (-75s)",
    "Shifted-K2 (-75s)": "Shifted-K2 (-75s)",
    "S0_brain_only": "Word-Only (Brain-Only)",
    "Word-Only (Brain-Only)": "Word-Only (Brain-Only)",
    "K0_brain_only": "Rolling-K0 (Brain-Only)",
    "Rolling-K0 (Brain-Only)": "Rolling-K0 (Brain-Only)",
    "Local (r = 0, Brain-Only)": "Rolling-K0 (Brain-Only)",
    "K1_brain_only": "Rolling-K1 (Brain-Only)",
    "Rolling-K1 (Brain-Only)": "Rolling-K1 (Brain-Only)",
    "Local (r = 1, Brain-Only)": "Rolling-K1 (Brain-Only)",
    "K2_brain_only": "Rolling-K2 (Brain-Only)",
    "Rolling-K2 (Brain-Only)": "Rolling-K2 (Brain-Only)",
    "Local (r = 2, Brain-Only)": "Rolling-K2 (Brain-Only)",
    "F_brain_only": "Full-Context (Brain-Only)",
    "Full-Context (Brain-Only)": "Full-Context (Brain-Only)",
}

PAPER_STRATEGY_DISPLAY_NAMES: dict[str, str] = {
    "S0": "Word-Only",
    "S0_clean": "Word-Only",
    "Word-Only": "Word-Only",
    "K0": "Local (r = 0)",
    "K0_clean": "Local (r = 0)",
    "Rolling-K0": "Local (r = 0)",
    "Local (r = 0)": "Local (r = 0)",
    "K1": "Local (r = 1)",
    "K1_clean": "Local (r = 1)",
    "Rolling-K1": "Local (r = 1)",
    "Local (r = 1)": "Local (r = 1)",
    "S3_clean": "Local (r = 2)",
    "S3": "Local (r = 2)",
    "K2": "Local (r = 2)",
    "Rolling-K2": "Local (r = 2)",
    "Local (r = 2)": "Local (r = 2)",
    "F": "Full-Context",
    "F_clean": "Full-Context",
    "Full-Context": "Full-Context",
    "LM-Only": "LM-Only",
    "LM-Only_clean": "LM-Only",
    "K0_shifted_neg75s": "Shifted Local (r = 0, -75s)",
    "Shifted-K0 (-75s)": "Shifted Local (r = 0, -75s)",
    "Shifted Local (r = 0, -75s)": "Shifted Local (r = 0, -75s)",
    "S3_shifted_neg75s": "Shifted Local (r = 2, -75s)",
    "Shifted-K2 (-75s)": "Shifted Local (r = 2, -75s)",
    "Shifted Local (r = 2, -75s)": "Shifted Local (r = 2, -75s)",
    "S0_brain_only": "Word-Only (Brain-Only)",
    "Word-Only (Brain-Only)": "Word-Only (Brain-Only)",
    "K0_brain_only": "Local (r = 0, Brain-Only)",
    "Rolling-K0 (Brain-Only)": "Local (r = 0, Brain-Only)",
    "Local (r = 0, Brain-Only)": "Local (r = 0, Brain-Only)",
    "K1_brain_only": "Local (r = 1, Brain-Only)",
    "Rolling-K1 (Brain-Only)": "Local (r = 1, Brain-Only)",
    "Local (r = 1, Brain-Only)": "Local (r = 1, Brain-Only)",
    "K2_brain_only": "Local (r = 2, Brain-Only)",
    "Rolling-K2 (Brain-Only)": "Local (r = 2, Brain-Only)",
    "Local (r = 2, Brain-Only)": "Local (r = 2, Brain-Only)",
    "F_brain_only": "Full-Context (Brain-Only)",
    "Full-Context (Brain-Only)": "Full-Context (Brain-Only)",
}

STRATEGY_STATE_KEYS: dict[str, str] = {
    "S0": "S0",
    "S0_clean": "S0",
    "Word-Only": "S0",
    "S0_brain_only": "S0",
    "Word-Only (Brain-Only)": "S0",
    "K0": "K0",
    "K0_clean": "K0",
    "Rolling-K0": "K0",
    "Local (r = 0)": "K0",
    "K0_brain_only": "K0",
    "Rolling-K0 (Brain-Only)": "K0",
    "Local (r = 0, Brain-Only)": "K0",
    "K1": "K1",
    "K1_clean": "K1",
    "Rolling-K1": "K1",
    "Local (r = 1)": "K1",
    "K1_brain_only": "K1",
    "Rolling-K1 (Brain-Only)": "K1",
    "Local (r = 1, Brain-Only)": "K1",
    "S3_clean": "K2",
    "S3": "K2",
    "K2": "K2",
    "Rolling-K2": "K2",
    "Local (r = 2)": "K2",
    "K2_brain_only": "K2",
    "Rolling-K2 (Brain-Only)": "K2",
    "Local (r = 2, Brain-Only)": "K2",
    "F": "F",
    "F_clean": "F",
    "Full-Context": "F",
    "F_brain_only": "F",
    "Full-Context (Brain-Only)": "F",
    "LM-Only": "LM-Only",
    "LM-Only_clean": "LM-Only",
    "K0_shifted_neg75s": "K0",
    "Shifted-K0 (-75s)": "K0",
    "Shifted Local (r = 0, -75s)": "K0",
    "S3_shifted_neg75s": "K2",
    "Shifted-K2 (-75s)": "K2",
    "Shifted Local (r = 2, -75s)": "K2",
}


def canonical_strategy_display_name(condition_key: str) -> str:
    """将内部解码条件代号映射为标准化报告策略名称。"""
    raw = str(condition_key).strip()
    return STRATEGY_DISPLAY_NAMES.get(raw, raw)


def paper_strategy_display_name(condition: str) -> str:
    """将内部解码条件代号映射为论文统一术语（如 Local (r = 0/1/2)）。"""
    raw = str(condition).strip()
    return PAPER_STRATEGY_DISPLAY_NAMES.get(raw, raw)


def _resolve_wait_strategy_state(condition: str | None) -> str:
    raw = str(condition or "").strip()
    if raw in STRATEGY_STATE_KEYS:
        return STRATEGY_STATE_KEYS[raw]
    disp = canonical_strategy_display_name(raw)
    if disp in STRATEGY_STATE_KEYS:
        return STRATEGY_STATE_KEYS[disp]
    p_disp = paper_strategy_display_name(raw)
    if p_disp in STRATEGY_STATE_KEYS:
        return STRATEGY_STATE_KEYS[p_disp]
    if "LM-Only" in raw or "LM-Only" in disp:
        return "LM-Only"
    if "Full-Context" in raw or "Full-Context" in disp:
        return "F"
    if "K2" in raw or "r = 2" in raw or "S3" in raw:
        return "K2"
    if "K1" in raw or "r = 1" in raw:
        return "K1"
    return "S0"


def compute_mean_wait_seconds(
    segment_records: Sequence[Mapping[str, Any]],
    condition: str | None = None,
) -> float:
    """计算给定策略在句段集合上的词均词窗等待时间（等效 1.0s 窗步数，含自身窗）。"""
    state_key = _resolve_wait_strategy_state(condition)
    if state_key == "LM-Only":
        return 0.0
    if state_key in ("S0", "K0"):
        return 1.0
    lengths = [
        len(rec.get("reference_words") or rec.get("predicted_words") or ())
        for rec in segment_records
    ]
    lengths = [L for L in lengths if L > 0]
    if not lengths:
        if state_key == "K1":
            return 1.80
        if state_key == "K2":
            return 2.60
        if state_key == "F":
            return 4.00
        return 1.0
    total_tokens = sum(lengths)
    if state_key == "K1":
        wait_sum = sum(
            sum(1.0 + min(1, L - 1 - pos) for pos in range(L))
            for L in lengths
        )
        return round(float(wait_sum / total_tokens), 4)
    if state_key == "K2":
        wait_sum = sum(
            sum(1.0 + min(2, L - 1 - pos) for pos in range(L))
            for L in lengths
        )
        return round(float(wait_sum / total_tokens), 4)
    if state_key == "F":
        wait_sum = sum(
            sum(1.0 + (L - 1 - pos) for pos in range(L))
            for L in lengths
        )
        return round(float(wait_sum / total_tokens), 4)
    return 1.0


def compute_mean_physical_wait_seconds(
    records: Sequence[Mapping[str, Any]],
    strategy_or_state: str | None,
    *,
    window_seconds: float = 1.0,
) -> float | None:
    """基于真实词起始时刻 τ_i 计算词均物理输入等待时间（秒）：(τ_{min(L-1, i+r)} + T_win) - τ_i。"""
    state_key = _resolve_wait_strategy_state(strategy_or_state)
    if state_key == "LM-Only":
        return 0.0
    if state_key in ("S0", "K0"):
        return round(float(window_seconds), 4)
    lookahead_r: int | None
    if state_key == "K1":
        lookahead_r = 1
    elif state_key == "K2":
        lookahead_r = 2
    elif state_key == "F":
        lookahead_r = None
    else:
        return round(float(window_seconds), 4)

    wait_sum = 0.0
    token_count = 0
    for rec in records:
        ref_words = rec.get("reference_words") or rec.get("predicted_words") or ()
        seg_len = len(ref_words)
        onsets = rec.get("onset_seconds")
        if seg_len <= 0:
            if not isinstance(onsets, (list, tuple)) or len(onsets) == 0:
                continue
            seg_len = len(onsets)
        if not isinstance(onsets, (list, tuple)) or len(onsets) != seg_len:
            return None
        try:
            tau = [float(x) for x in onsets]
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(t) for t in tau):
            return None
        last_idx = seg_len - 1
        for pos in range(seg_len):
            end_pos = (
                last_idx
                if lookahead_r is None
                else min(last_idx, pos + lookahead_r)
            )
            wait_sum += max(
                float(window_seconds),
                (tau[end_pos] + float(window_seconds)) - tau[pos],
            )
            token_count += 1
    if token_count <= 0:
        return None
    return round(float(wait_sum / token_count), 4)


def _extract_canonical_material_id(
    rec: Mapping[str, Any], fallback_idx: int = 0
) -> str:
    """提取跨受试者共享的规范材料段标识（剥离 k*:: 与 sub-XX 前缀，保留 run-06:0 / chapter-02:0）。"""
    mat_id = str(rec.get("material_id") or "").strip()
    scope_id = str(rec.get("scope") or "").strip()
    if mat_id and scope_id and not mat_id.startswith(scope_id):
        return f"{scope_id}:{mat_id}"
    if mat_id:
        return mat_id
    gid = str(rec.get("group_id") or "").strip()
    if gid:
        gid_clean = re.sub(r"^k\d+::", "", gid, flags=re.IGNORECASE).strip()
        gid_clean = re.sub(
            r"^(?:sub-\d+|s\d+)(?:[_-]|:+)?",
            "",
            gid_clean,
            flags=re.IGNORECASE,
        ).strip()
        parts = [p.strip() for p in gid_clean.split(":") if p.strip()]
        if len(parts) >= 2 and parts[0].isdigit():
            tail = ":".join(parts[1:])
            return f"{scope_id}:{tail}" if scope_id else tail
        if gid_clean:
            if (
                scope_id
                and ":" not in gid_clean
                and not gid_clean.startswith(scope_id)
            ):
                return f"{scope_id}:{gid_clean}"
            return gid_clean
    ref = tuple(str(w) for w in (rec.get("reference_words") or ()))
    if ref:
        return f"{scope_id}:{'_'.join(ref[:6])}_{len(ref)}"
    return f"seg_{fallback_idx:04d}"


def _assign_material_blocks(
    records: Sequence[Mapping[str, Any]],
    *,
    segments_per_block: int = 1,
    block_size: int | None = None,
) -> list[str]:
    """按独立材料长句（跨被试对齐，默认 block_size=1 即每个独立筛选长句为 1 个统计块）分配 block_id；若子集材料段过少则退化为单记录块以保证统计自由度。"""
    mat_ids = [
        _extract_canonical_material_id(rec, idx)
        for idx, rec in enumerate(records)
    ]
    unique_mats: list[str] = []
    mat_to_ord: dict[str, int] = {}
    for m in mat_ids:
        if m not in mat_to_ord:
            mat_to_ord[m] = len(unique_mats)
            unique_mats.append(m)
    m_count = len(unique_mats)
    eff_block_size = (
        int(block_size) if block_size is not None else int(segments_per_block)
    )
    step = max(1, eff_block_size)
    if step > 1 and m_count >= step * 6:
        return [f"block_{mat_to_ord[m] // step:04d}" for m in mat_ids]
    if m_count >= 4:
        return [f"block_{mat_to_ord[m]:04d}" for m in mat_ids]
    return [f"seg_{idx:04d}" for idx in range(len(records))]


def _paired_block_permutation_and_bootstrap(
    block_diffs: np.ndarray,
    block_weights: np.ndarray,
    *,
    n_resamples: int = 10000,
    seed: int = 20261031,
) -> dict[str, float | int]:
    """对块级差值向量与块权重执行 10,000 次分层块自举（95% CI）与双侧符号翻转置换检验。"""
    diffs = np.asarray(block_diffs, dtype=np.float64)
    weights = np.asarray(block_weights, dtype=np.float64)
    valid = np.isfinite(diffs) & (weights > 0.0)
    diffs = diffs[valid]
    weights = weights[valid]
    n_blocks = int(len(diffs))
    if n_blocks == 0:
        return {
            "effect_pp": 0.0,
            "ci_lower_pp": 0.0,
            "ci_upper_pp": 0.0,
            "p_one_sided": 1.0,
            "p_two_sided": 1.0,
            "blocks": 0,
        }
    total_w = max(float(np.sum(weights)), 1e-12)
    observed = float(np.sum(diffs * weights) / total_w)
    if n_blocks <= 1 or float(np.max(np.abs(diffs))) <= 1e-12:
        p_val = 1.0 if abs(observed) <= 1e-12 else 0.05
        return {
            "effect_pp": observed,
            "ci_lower_pp": observed,
            "ci_upper_pp": observed,
            "p_one_sided": p_val,
            "p_two_sided": p_val,
            "blocks": n_blocks,
        }
    norm_weights = weights / total_w
    weighted_contribs = diffs * norm_weights
    rng = np.random.default_rng(int(seed))
    if n_blocks <= 14:
        ints = np.arange(2**n_blocks, dtype=np.uint32)
        signs = (
            2.0
            * (
                (ints[:, None] >> np.arange(n_blocks, dtype=np.uint32)) & 1
            ).astype(np.float64)
            - 1.0
        )
        null_dist = signs @ weighted_contribs
        if observed >= 0.0:
            p_one = float(np.mean(null_dist >= observed - 1e-12))
        else:
            p_one = float(np.mean(null_dist <= observed + 1e-12))
        p_two = float(np.mean(np.abs(null_dist) >= abs(observed) - 1e-12))
    else:
        signs = rng.choice((-1.0, 1.0), size=(int(n_resamples), n_blocks))
        null_dist = signs @ weighted_contribs
        if observed >= 0.0:
            ext_one = int(np.count_nonzero(null_dist >= observed - 1e-12))
        else:
            ext_one = int(np.count_nonzero(null_dist <= observed + 1e-12))
        ext_two = int(
            np.count_nonzero(np.abs(null_dist) >= abs(observed) - 1e-12)
        )
        p_one = float((1 + ext_one) / (int(n_resamples) + 1))
        p_two = float((1 + ext_two) / (int(n_resamples) + 1))

    boot_picks = rng.integers(0, n_blocks, size=(int(n_resamples), n_blocks))
    picked_diffs = diffs[boot_picks]
    picked_weights = weights[boot_picks]
    boot_estimates = np.sum(picked_diffs * picked_weights, axis=1) / np.maximum(
        np.sum(picked_weights, axis=1), 1e-12
    )
    ci_low, ci_high = (
        float(x) for x in np.quantile(boot_estimates, (0.025, 0.975))
    )
    return {
        "effect_pp": observed,
        "ci_lower_pp": ci_low,
        "ci_upper_pp": ci_high,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "blocks": n_blocks,
    }

# 法语闭集功能词表（冠词、代词、介词、连词、助动词、否定与高频功能副词、省音功能词）
FRENCH_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "le", "la", "les", "un", "une", "du", "des", "au", "aux",
        "de", "à", "a", "en", "dans", "sur", "sous", "pour", "par", "avec",
        "sans", "chez", "vers", "entre", "contre", "depuis", "pendant",
        "avant", "après", "devant", "derrière", "dès", "selon", "malgré",
        "et", "ou", "où", "mais", "donc", "or", "ni", "car", "comme", "si",
        "que", "qui", "quoi", "dont", "quand", "lorsque", "puisque", "quoique",
        "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
        "me", "te", "se", "lui", "leur", "leurs", "moi", "toi", "soi", "eux",
        "y", "en", "ce", "cet", "cette", "ces", "cela", "ça", "ceci",
        "celui", "celle", "ceux", "celles", "-ci", "-là",
        "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses",
        "notre", "votre", "nos", "vos", "mien", "tien", "sien",
        "c", "j", "l", "d", "n", "m", "s", "t", "qu", "jusqu", "lorsqu", "puisqu", "quelqu",
        "ne", "pas", "plus", "jamais", "rien", "personne", "aucun", "aucune",
        "très", "bien", "trop", "peu", "assez", "aussi", "encore", "déjà", "toujours",
        "tout", "tous", "toute", "toutes", "même", "autre", "autres", "quel", "quelle", "quels", "quelles",
        "être", "suis", "es", "est", "sommes", "êtes", "sont", "étais", "était", "étaient", "fut", "sera", "serait", "été", "étant",
        "avoir", "ai", "as", "avons", "avez", "ont", "avais", "avait", "avaient", "eut", "aura", "aurait", "eu", "ayant",
        "-je", "-tu", "-il", "-elle", "-on", "-nous", "-vous", "-ils", "-elles",
        "-t-il", "-t-elle", "-t-on", "-t-ils", "-t-elles",
        "-le", "-la", "-les", "-lui", "-leur", "-moi", "-toi", "-t", "-y", "-en",
    }
)

# 中文闭集功能词表（助词、代词、介词、连词、语气词、副词、系词/助动词、通用量词与方位词）
CHINESE_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "的", "地", "得", "了", "着", "过", "之", "所",
        "吗", "呢", "吧", "啊", "呀", "啦", "呐", "哇", "嘛", "罢了", "而已",
        "我", "你", "您", "他", "她", "它", "我们", "你们", "他们", "她们", "它们", "咱们", "自己", "大家", "别人",
        "这", "那", "这个", "那个", "这些", "那些", "这里", "那里", "这儿", "那儿", "这么", "那么", "这样", "那样",
        "谁", "什么", "哪", "哪个", "哪些", "哪里", "哪儿", "怎么", "怎样", "怎么样", "多少", "几",
        "在", "从", "自", "打", "向", "往", "朝", "对", "对于", "关于", "至于", "替", "给", "为", "为了",
        "把", "被", "叫", "让", "由", "归", "管", "靠", "比", "跟", "和", "与", "同", "及", "以及",
        "或", "或者", "还是", "但", "但是", "可", "可是", "却", "然而", "不过", "只是",
        "而", "而且", "并且", "况且", "何况", "不但", "不仅", "不光",
        "因为", "由于", "所以", "因此", "因而", "以致",
        "如果", "假如", "假若", "要是", "倘若", "只要", "只有", "除非", "无论", "不管", "不论",
        "虽然", "尽管", "固然", "即使", "就算", "哪怕", "既然", "既", "便", "就", "才", "都", "也", "还", "又", "再",
        "不", "没", "没有", "别", "莫", "勿", "未", "非",
        "很", "太", "更", "最", "挺", "怪", "非常", "十分", "格外", "稍微", "略微", "更加", "越发", "尤其",
        "已经", "已", "曾经", "曾", "正在", "正", "在", "将", "将要", "就要", "立刻", "马上", "忽然", "突然",
        "常常", "往往", "一直", "总是", "老是", "仍然", "仍旧", "依然", "果然", "居然", "竟然", "简直", "到底", "究竟",
        "几乎", "差不多", "大概", "大约", "也许", "或许", "恐怕", "大概", "必定", "一定", "肯定", "当然",
        "只", "光", "单", "仅仅", "总共", "一共", "统统", "一齐", "一起", "一同", "互相", "彼此",
        "是", "有", "在", "像", "好比", "如同", "能", "能够", "会", "可以", "可能", "应该", "应当", "得", "必须", "愿意", "敢", "肯",
        "个", "位", "只", "条", "件", "种", "类", "些", "点", "次", "回", "遍", "趟", "下", "番", "群", "伙", "双", "对", "副", "张", "座", "棵", "朵", "颗", "粒",
        "上", "下", "左", "右", "前", "后", "里", "外", "中", "内", "间", "旁", "以上", "以下", "以前", "以后", "之内", "之外", "之间", "之中",
    }
)


def is_function_word(word: str, language: str) -> bool:
    """判断给定词是否为功能词（Function Word）。"""
    clean = str(word).strip()
    if not clean:
        return True
    if language == "zh":
        return clean in CHINESE_FUNCTION_WORDS
    lowered = clean.lower()
    if lowered in FRENCH_FUNCTION_WORDS or lowered.lstrip("-") in FRENCH_FUNCTION_WORDS:
        return True
    return False


def is_content_word(word: str, language: str) -> bool:
    """判断给定词是否为实义词（Content Word）。"""
    return not is_function_word(word, language)


_POS4_NOUN_TAGS: frozenset[str] = frozenset(
    {"NC", "NPP", "N", "NOM", "NOUN", "PROPN", "NN", "NNS", "NNP", "NNPS", "PROPER_NOUN", "SUBST"}
)
_POS4_VERB_TAGS: frozenset[str] = frozenset(
    {"V", "VINF", "VPP", "VPR", "VS", "VIMP", "VER", "VERB", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"}
)
_POS4_ADJ_ADV_TAGS: frozenset[str] = frozenset(
    {"ADJ", "ADJWH", "ADV", "ADVWH", "A", "ADJECTIVE", "ADVERB", "JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB"}
)
_POS4_FUNCTION_TAGS: frozenset[str] = frozenset(
    {
        "DET", "DETWH", "ART", "ADP", "P", "P+D", "P+PRO", "PRE", "PREP",
        "PRON", "PRO", "PROREL", "PROWH", "CLS", "CLO", "CLR",
        "CON", "KON", "CC", "CS", "CCONJ", "SCONJ", "PART", "CONJ",
        "PREF", "INTJ", "I", "NUM", "IN", "DT", "PDT", "PRP", "PRP$",
        "WP", "WP$", "WDT", "TO", "EX", "RP", "UH", "FW", "LIA",
    }
)


def classify_part_of_speech_four_way(
    pos_tag: Any, is_content_word: Any = None
) -> str | None:
    """将词性标签归一化为 4 类（noun / verb / adj_adv / function）。"""
    raw = str(pos_tag or "").strip().upper()
    if raw in {"PUNCT", "PONCT", "SYM", "SPACE"}:
        return None
    if raw in {"", "NONE", "NAN", "NULL"}:
        return "function" if is_content_word is False else None
    base = raw.split(":", 1)[0]
    if base in _POS4_NOUN_TAGS or base.startswith(("NN", "NC", "NPP")):
        return "noun"
    if base in _POS4_VERB_TAGS or base.startswith(("VB", "VINF", "VPP", "VPR", "VIMP")):
        return "verb"
    if base == "AUX":
        return "verb" if is_content_word is True else "function"
    if base in _POS4_ADJ_ADV_TAGS or base.startswith(("JJ", "RB", "ADJ", "ADV")):
        return "adj_adv"
    if base in _POS4_FUNCTION_TAGS:
        return "function"
    if is_content_word is False:
        return "function"
    return None


def _character_ngram_vector(word: str, dim: int = 256) -> np.ndarray:
    """为未登录词生成确定性字符 n-gram 单位向量（仅在缺少预计算词向量时作为 BERTScore 兜底）。"""
    text = f"<{str(word).strip().lower()}>"
    vec = np.zeros(dim, dtype=np.float32)
    for n in (1, 2, 3):
        for i in range(max(1, len(text) - n + 1)):
            gram = text[i : i + n]
            h = hash(gram)
            idx = h % dim
            sign = 1.0 if (h // dim) % 2 == 0 else -1.0
            vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-8)


_EDIT_DISTANCE_PAIR_CACHE: dict[
    tuple[bool, tuple[str, ...], tuple[str, ...]], tuple[int, int, int, int]
] = {}
_BERTSCORE_PAIR_CACHE: dict[
    tuple[int, tuple[str, ...], tuple[str, ...]], tuple[float, float, float]
] = {}
_MAX_EVAL_PAIR_CACHE_ENTRIES = 250000


def _cached_edit_distance_pair(
    ref_tuple: tuple[str, ...],
    pred_tuple: tuple[str, ...],
    *,
    chinese: bool,
) -> tuple[int, int, int, int]:
    """计算并缓存单句段 (ref_tuple, pred_tuple) 的词级与字符级 Levenshtein 编辑距离。"""
    key = (chinese, ref_tuple, pred_tuple)
    cached = _EDIT_DISTANCE_PAIR_CACHE.get(key)
    if cached is not None:
        return cached
    if ref_tuple == pred_tuple:
        ref_text = "".join(ref_tuple) if chinese else " ".join(ref_tuple)
        res = (0, len(ref_tuple), 0, len(ref_text))
    else:
        word_err = int(edit_distance(ref_tuple, pred_tuple))
        ref_text = "".join(ref_tuple) if chinese else " ".join(ref_tuple)
        pred_text = "".join(pred_tuple) if chinese else " ".join(pred_tuple)
        char_err = int(edit_distance(ref_text, pred_text))
        res = (word_err, len(ref_tuple), char_err, len(ref_text))
    if len(_EDIT_DISTANCE_PAIR_CACHE) >= _MAX_EVAL_PAIR_CACHE_ENTRIES:
        _EDIT_DISTANCE_PAIR_CACHE.clear()
    _EDIT_DISTANCE_PAIR_CACHE[key] = res
    return res


def _unigram_overlap_and_bleu1(
    ref_counts: Counter[str],
    ref_len: int,
    pred_seq: Sequence[str],
) -> tuple[int, float]:
    """零编辑距离开销计算单句段 unigram 截断重合数与 sentence BLEU-1。"""
    pred_len = len(pred_seq)
    if pred_len == 0:
        return 0, 0.0
    pred_counts = Counter(pred_seq)
    overlap = sum(
        min(count, pred_counts[word]) for word, count in ref_counts.items()
    )
    precision = overlap / pred_len
    brevity = (
        1.0
        if pred_len >= ref_len
        else float(min(1.0, np.exp(1.0 - ref_len / pred_len)))
    )
    return int(overlap), float(precision * brevity)


def compute_bertscore_percentage(
    references: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
    *,
    word_embedding_map: Mapping[str, np.ndarray | torch.Tensor] | None = None,
) -> dict[str, float]:
    """计算参考序列与预测序列之间的连续语义相似度 BERTScore（Precision / Recall / F1 %）。

    对于每对句子 (r, \\hat{w})：
      - 提取每个词的 L2 归一化稠密语义向量 e(r_i), e(\\hat{w}_j)；
      - 计算余弦相似度矩阵 M_{i,j} = <e(r_i), e(\\hat{w}_j)>；
      - Recall = (1/|r|) \\sum_i \\max_j M_{i,j}
      - Precision = (1/|\\hat{w}|) \\sum_j \\max_i M_{i,j}
      - F1 = 2 * P * R / (P + R)
    """
    if len(references) != len(predictions) or not references:
        raise ValueError("references 与 predictions 必须非空且长度一致。")

    norm_cache: dict[str, np.ndarray] = {}
    fallback_dim = 256
    map_id = id(word_embedding_map) if word_embedding_map is not None else 0
    if word_embedding_map:
        first_val = next(iter(word_embedding_map.values()))
        if isinstance(first_val, torch.Tensor):
            fallback_dim = int(first_val.numel())
        else:
            fallback_dim = int(np.asarray(first_val).size)

    def _get_vec(word: str) -> np.ndarray:
        w = str(word)
        if w in norm_cache:
            return norm_cache[w]
        if word_embedding_map is not None and w in word_embedding_map:
            raw = word_embedding_map[w]
            arr = (
                raw.detach().cpu().numpy().astype(np.float32)
                if isinstance(raw, torch.Tensor)
                else np.asarray(raw, dtype=np.float32)
            ).reshape(-1)
            norm = float(np.linalg.norm(arr))
            vec = arr / max(norm, 1e-8)
        else:
            vec = _character_ngram_vector(w, dim=fallback_dim)
        norm_cache[w] = vec
        return vec

    p_list: list[float] = []
    r_list: list[float] = []
    f1_list: list[float] = []

    for ref_seq, pred_seq in zip(references, predictions):
        ref_tuple = tuple(str(w) for w in ref_seq)
        pred_tuple = tuple(str(w) for w in pred_seq)
        if not ref_tuple or not pred_tuple:
            p_list.append(0.0)
            r_list.append(0.0)
            f1_list.append(0.0)
            continue

        pair_key = (map_id, ref_tuple, pred_tuple)
        cached_bs = _BERTSCORE_PAIR_CACHE.get(pair_key)
        if cached_bs is not None:
            p_pct, r_pct, f1_pct = cached_bs
            p_list.append(p_pct)
            r_list.append(r_pct)
            f1_list.append(f1_pct)
            continue

        ref_mat = np.stack([_get_vec(w) for w in ref_tuple], axis=0)
        pred_mat = np.stack([_get_vec(w) for w in pred_tuple], axis=0)
        sim_matrix = np.clip(ref_mat @ pred_mat.T, -1.0, 1.0)

        recall = float(np.mean(np.max(sim_matrix, axis=1)))
        precision = float(np.mean(np.max(sim_matrix, axis=0)))
        if precision + recall > 1e-12:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0

        p_pct = precision * 100.0
        r_pct = recall * 100.0
        f1_pct = f1 * 100.0
        if len(_BERTSCORE_PAIR_CACHE) >= _MAX_EVAL_PAIR_CACHE_ENTRIES:
            _BERTSCORE_PAIR_CACHE.clear()
        _BERTSCORE_PAIR_CACHE[pair_key] = (p_pct, r_pct, f1_pct)
        p_list.append(p_pct)
        r_list.append(r_pct)
        f1_list.append(f1_pct)

    return {
        "bertscore_precision_pct": float(np.mean(p_list)),
        "bertscore_recall_pct": float(np.mean(r_list)),
        "bertscore_f1_pct": float(np.mean(f1_list)),
    }


def evaluate_six_dimensional_metrics(
    segment_records: Sequence[dict],
    *,
    language: str,
    word_embedding_map: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    condition: str | None = None,
    compute_paired_stats: bool = True,
) -> dict[str, float | int | str | None]:
    """同步计算 S3 序列解码协议规定的 6 维核心评价指标：

    1. BLEU-1 (%)：含句段均值 `bleu1_pct` 与语料级 `corpus_bleu1_pct`
    2. WER (%)：词错误率 `wer_pct`
    3. CER (%)：字符/汉字错误率 `cer_pct`
    4. Word Acc (%)：对齐词位总准确率 `word_acc_pct`
    5. 实义词/功能词拆解：`content_word_acc_pct`、`function_word_acc_pct` 及差值与配对 p 值
    6. BERTScore (%)：`bertscore_f1_pct`（及 Precision / Recall）
    """
    if not segment_records:
        raise ValueError("segment_records 不能为空。")

    chinese = language == "zh"
    references = [
        [str(w) for w in rec["reference_words"]] for rec in segment_records
    ]
    predictions = [
        [str(w) for w in rec["predicted_words"]] for rec in segment_records
    ]
    block_ids = (
        _assign_material_blocks(segment_records)
        if compute_paired_stats
        else []
    )

    total_words = 0
    total_correct = 0
    total_top5_correct = 0
    content_total = 0
    content_correct = 0
    function_total = 0
    function_correct = 0
    in_vocab_total = 0
    in_vocab_correct = 0
    in_vocab_top5_correct = 0
    oracle_correct = 0
    oracle_in_vocab_correct = 0

    word_errors_sum = 0
    char_errors_sum = 0
    char_total_sum = 0
    corpus_clipped_sum = 0
    oracle_clipped_sum = 0
    sentence_bleu1_all: list[float] = []
    oracle_sentence_bleu1_all: list[float] = []

    per_subject_rows: dict[str, list[dict]] = defaultdict(list)
    block_content_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    pos4_totals: dict[str, int] = {"noun": 0, "verb": 0, "adj_adv": 0, "function": 0}
    pos4_correct: dict[str, int] = {"noun": 0, "verb": 0, "adj_adv": 0, "function": 0}

    for rec_idx, (rec, ref_seq, pred_seq) in enumerate(
        zip(segment_records, references, predictions)
    ):
        seg_len = len(ref_seq)
        if seg_len != len(pred_seq):
            raise ValueError(
                f"参考词序列长度 ({seg_len}) 与预测词序列长度 ({len(pred_seq)}) 不一致。"
            )
        ref_tuple = tuple(ref_seq)
        pred_tuple = tuple(pred_seq)
        w_err, _, c_err, c_len = _cached_edit_distance_pair(
            ref_tuple, pred_tuple, chinese=chinese
        )
        word_errors_sum += w_err
        char_errors_sum += c_err
        char_total_sum += c_len

        ref_counts = Counter(ref_tuple)
        seg_overlap, seg_bleu1 = _unigram_overlap_and_bleu1(
            ref_counts, seg_len, pred_tuple
        )
        corpus_clipped_sum += seg_overlap
        sentence_bleu1_all.append(seg_bleu1)

        in_vocab_mask = rec.get("in_vocabulary_mask") or [True] * seg_len
        top5_per_pos = rec.get("top5_words_per_pos") or []
        beam_cands_raw = rec.get("beam_candidates") or []

        # 确定 Top-B (Beam=5) Oracle 句段预测（直接用 O(L) unigram 重合度计算，零编辑距离开销）
        valid_beam_cands = [
            [str(w) for w in cand]
            for cand in beam_cands_raw
            if isinstance(cand, (list, tuple)) and len(cand) == seg_len
        ]
        if not valid_beam_cands:
            valid_beam_cands = [pred_seq]
        elif pred_seq not in valid_beam_cands:
            valid_beam_cands = [pred_seq] + valid_beam_cands

        best_oracle_seq = pred_seq
        best_oracle_key = (-1.0, -1)
        best_oracle_overlap = seg_overlap
        best_oracle_bleu1 = seg_bleu1
        for cand_seq in valid_beam_cands:
            c_hits = sum(int(a == b) for a, b in zip(ref_seq, cand_seq))
            if cand_seq == pred_seq:
                c_overlap, c_bleu = seg_overlap, seg_bleu1
            else:
                c_overlap, c_bleu = _unigram_overlap_and_bleu1(
                    ref_counts, seg_len, cand_seq
                )
            cand_key = (
                round(0.5 * c_bleu + 0.5 * (c_hits / max(1, seg_len)), 8),
                c_hits,
            )
            if cand_key > best_oracle_key:
                best_oracle_key = cand_key
                best_oracle_seq = cand_seq
                best_oracle_overlap = c_overlap
                best_oracle_bleu1 = c_bleu

        oracle_clipped_sum += best_oracle_overlap
        oracle_sentence_bleu1_all.append(best_oracle_bleu1)

        seg_hits = 0
        seg_c_tot = seg_c_hit = seg_f_tot = seg_f_hit = 0
        seg_pos_tags = rec.get("pos_tags")
        seg_icw_mask = rec.get("is_content_word_mask")
        for pos, (r_w, p_w) in enumerate(zip(ref_seq, pred_seq)):
            hit = int(r_w == p_w)
            seg_hits += hit
            ora_w = best_oracle_seq[pos] if pos < len(best_oracle_seq) else p_w
            ora_hit = int(r_w == ora_w)
            pos_cands: set[str] = {p_w, ora_w}
            if pos < len(top5_per_pos) and isinstance(
                top5_per_pos[pos], (list, tuple)
            ):
                for cw in top5_per_pos[pos][:5]:
                    pos_cands.add(str(cw))
            for b_seq in valid_beam_cands[:5]:
                if pos < len(b_seq):
                    pos_cands.add(str(b_seq[pos]))
            top5_hit = int(r_w in pos_cands)

            total_words += 1
            total_correct += hit
            total_top5_correct += top5_hit
            oracle_correct += ora_hit
            if pos < len(in_vocab_mask) and bool(in_vocab_mask[pos]):
                in_vocab_total += 1
                in_vocab_correct += hit
                in_vocab_top5_correct += top5_hit
                oracle_in_vocab_correct += ora_hit
            if is_content_word(r_w, language):
                content_total += 1
                content_correct += hit
                seg_c_tot += 1
                seg_c_hit += hit
            else:
                function_total += 1
                function_correct += hit
                seg_f_tot += 1
                seg_f_hit += hit
            if isinstance(seg_pos_tags, (list, tuple)) and pos < len(seg_pos_tags):
                icw_val = (
                    bool(seg_icw_mask[pos])
                    if isinstance(seg_icw_mask, (list, tuple))
                    and pos < len(seg_icw_mask)
                    and seg_icw_mask[pos] is not None
                    else None
                )
                pos4_cls = classify_part_of_speech_four_way(
                    seg_pos_tags[pos], is_content_word=icw_val
                )
                if pos4_cls in pos4_totals:
                    pos4_totals[pos4_cls] += 1
                    pos4_correct[pos4_cls] += hit

        if compute_paired_stats:
            b_key = block_ids[rec_idx]
            b_stat = block_content_stats[b_key]
            b_stat[0] += seg_c_hit
            b_stat[1] += seg_c_tot
            b_stat[2] += seg_f_hit
            b_stat[3] += seg_f_tot

        subj = str(rec.get("subject_id", "default"))
        per_subject_rows[subj].append(
            {
                "words": seg_len,
                "sentence_bleu1": float(seg_bleu1),
                "oracle_sentence_bleu1": float(best_oracle_bleu1),
                "wer": float(w_err / seg_len) if seg_len > 0 else 0.0,
                "cer": float(c_err / c_len) if c_len > 0 else 0.0,
                "word_acc": float(seg_hits / seg_len) if seg_len > 0 else 0.0,
                "content_total": seg_c_tot,
                "content_correct": seg_c_hit,
                "function_total": seg_f_tot,
                "function_correct": seg_f_hit,
            }
        )

    base = {
        "sentence_bleu1_mean": (
            float(np.mean(sentence_bleu1_all)) if sentence_bleu1_all else 0.0
        ),
        "bleu1": (
            float(corpus_clipped_sum / total_words) if total_words > 0 else 0.0
        ),
        "wer": (
            float(word_errors_sum / total_words) if total_words > 0 else 0.0
        ),
        "cer": (
            float(char_errors_sum / char_total_sum)
            if char_total_sum > 0
            else 0.0
        ),
    }
    oracle_base = {
        "sentence_bleu1_mean": (
            float(np.mean(oracle_sentence_bleu1_all))
            if oracle_sentence_bleu1_all
            else 0.0
        ),
        "bleu1": (
            float(oracle_clipped_sum / total_words) if total_words > 0 else 0.0
        ),
    }
    bert_scores = compute_bertscore_percentage(
        references, predictions, word_embedding_map=word_embedding_map
    )

    # 计算受试者等权宏平均 BLEU-1 (%) 与 Word Acc (%)
    subj_bleu1_list = [
        float(np.mean([r["sentence_bleu1"] for r in rows])) * 100.0
        for rows in per_subject_rows.values()
    ]
    subj_oracle_bleu1_list = [
        float(np.mean([r["oracle_sentence_bleu1"] for r in rows])) * 100.0
        for rows in per_subject_rows.values()
    ]
    subj_word_acc_list = [
        float(
            np.average(
                [r["word_acc"] for r in rows],
                weights=[max(1, r["words"]) for r in rows],
            )
        )
        * 100.0
        for rows in per_subject_rows.values()
    ]

    word_acc_pct = (
        float(total_correct / total_words * 100.0) if total_words > 0 else 0.0
    )
    word_acc_top5_pct = (
        float(total_top5_correct / total_words * 100.0)
        if total_words > 0
        else 0.0
    )
    content_acc_pct = (
        float(content_correct / content_total * 100.0)
        if content_total > 0
        else 0.0
    )
    function_acc_pct = (
        float(function_correct / function_total * 100.0)
        if function_total > 0
        else 0.0
    )
    in_vocab_acc_pct = (
        float(in_vocab_correct / in_vocab_total * 100.0)
        if in_vocab_total > 0
        else 0.0
    )
    in_vocab_top5_acc_pct = (
        float(in_vocab_top5_correct / in_vocab_total * 100.0)
        if in_vocab_total > 0
        else 0.0
    )
    oracle_word_acc_pct = (
        float(oracle_correct / total_words * 100.0) if total_words > 0 else 0.0
    )
    oracle_in_vocab_acc_pct = (
        float(oracle_in_vocab_correct / in_vocab_total * 100.0)
        if in_vocab_total > 0
        else 0.0
    )

    cf_delta_pp = float(content_acc_pct - function_acc_pct)
    if compute_paired_stats:
        cf_diffs: list[float] = []
        cf_weights: list[float] = []
        for c_hit, c_tot, f_hit, f_tot in block_content_stats.values():
            if c_tot > 0 and f_tot > 0:
                cf_diffs.append((c_hit / c_tot - f_hit / f_tot) * 100.0)
                cf_weights.append(float(c_tot + f_tot))
        cf_stat = _paired_block_permutation_and_bootstrap(
            np.asarray(cf_diffs, dtype=np.float64),
            np.asarray(cf_weights, dtype=np.float64),
            n_resamples=10000,
            seed=20261077,
        )
        cf_p_two = float(cf_stat["p_two_sided"])
    else:
        cf_p_two = 1.0
    mean_wait_s = compute_mean_wait_seconds(segment_records, condition=condition)
    mean_phys_wait_s = compute_mean_physical_wait_seconds(
        segment_records, condition
    )

    res_dict: dict[str, float | int | str | None] = {
        "segment_count": len(segment_records),
        "total_words": int(total_words),
        "content_word_count": int(content_total),
        "function_word_count": int(function_total),
        "in_vocabulary_word_count": int(in_vocab_total),
        "mean_wait_seconds": float(mean_wait_s),
        "mean_wait_windows": float(mean_wait_s),
        "mean_physical_wait_seconds": mean_phys_wait_s,
        "bleu1_pct": float(np.mean(subj_bleu1_list))
        if subj_bleu1_list
        else float(base["sentence_bleu1_mean"] * 100.0),
        "sentence_bleu1_pct": float(base["sentence_bleu1_mean"] * 100.0),
        "corpus_bleu1_pct": float(base["bleu1"] * 100.0),
        "beam_oracle_bleu1_pct": float(np.mean(subj_oracle_bleu1_list))
        if subj_oracle_bleu1_list
        else float(oracle_base["sentence_bleu1_mean"] * 100.0),
        "beam_oracle_corpus_bleu1_pct": float(oracle_base["bleu1"] * 100.0),
        "wer_pct": float(base["wer"] * 100.0),
        "cer_pct": float(base["cer"] * 100.0),
        "word_acc_pct": word_acc_pct,
        "in_vocabulary_word_acc_pct": in_vocab_acc_pct,
        "word_acc_top5_pct": word_acc_top5_pct,
        "in_vocabulary_word_acc_top5_pct": in_vocab_top5_acc_pct,
        "beam_oracle_word_acc_pct": oracle_word_acc_pct,
        "beam_oracle_in_vocabulary_word_acc_pct": oracle_in_vocab_acc_pct,
        "subject_equal_word_acc_pct": float(np.mean(subj_word_acc_list))
        if subj_word_acc_list
        else word_acc_pct,
        "content_word_acc_pct": content_acc_pct,
        "function_word_acc_pct": function_acc_pct,
        "content_minus_function_acc_pp": round(cf_delta_pp, 4),
        "content_vs_function_p_value": round(cf_p_two, 6),
        "content_vs_function_significance": _s3_significance_stars(cf_p_two),
        "bertscore_f1_pct": float(bert_scores["bertscore_f1_pct"]),
        "bertscore_precision_pct": float(bert_scores["bertscore_precision_pct"]),
        "bertscore_recall_pct": float(bert_scores["bertscore_recall_pct"]),
    }
    pos4_total_classified = sum(pos4_totals.values())
    if pos4_total_classified > 0:
        for cls_key, out_prefix in (
            ("noun", "noun"),
            ("verb", "verb"),
            ("adj_adv", "adj_adv"),
            ("function", "function_pos"),
        ):
            cls_tot = int(pos4_totals[cls_key])
            cls_cor = int(pos4_correct[cls_key])
            res_dict[f"{out_prefix}_words"] = cls_tot
            res_dict[f"{out_prefix}_word_acc_pct"] = (
                round(float(cls_cor / cls_tot * 100.0), 4)
                if cls_tot > 0
                else 0.0
            )
    if condition:
        res_dict["strategy_display_name"] = canonical_strategy_display_name(
            condition
        )
        res_dict["paper_display_name"] = paper_strategy_display_name(condition)
    return res_dict


def _copy_optional_segment_metadata(
    dst: dict[str, Any], *sources: Mapping[str, Any]
) -> dict[str, Any]:
    for key in (
        "event_ids",
        "onset_seconds",
        "offset_seconds",
        "pos_tags",
        "is_content_word_mask",
    ):
        for src in sources:
            val = src.get(key)
            if val is not None:
                dst[key] = list(val) if isinstance(val, (list, tuple)) else val
                break
    return dst


def _normalize_decoded_bundles(
    decoded_bundles: Mapping[str, dict],
) -> dict[str, dict]:
    """规范化解码条件键别名（将 S0_clean / F_clean 同步映射至 S0 / F）。"""
    bundles: dict[str, dict] = dict(decoded_bundles)
    if "S0" not in bundles and "S0_clean" in bundles:
        bundles["S0"] = bundles["S0_clean"]
    if "F" not in bundles and "F_clean" in bundles:
        bundles["F"] = bundles["F_clean"]
    return bundles


def _filter_records_by_vocabulary_coverage(
    records: Sequence[Mapping],
    min_coverage_fraction: float,
) -> list[dict]:
    """筛选词表覆盖率 >= min_coverage_fraction 的高覆盖核心句段（对标 SimpleB2T Core / Expanded 子集）。"""
    threshold = float(min_coverage_fraction) - 1e-9
    filtered: list[dict] = []
    for rec in records:
        ref = rec.get("reference_words") or []
        if not ref:
            continue
        iv_mask = rec.get("in_vocabulary_mask")
        if iv_mask is None or len(iv_mask) != len(ref):
            cov = 1.0
        else:
            cov = float(sum(1 for flag in iv_mask if bool(flag)) / len(ref))
        if cov >= threshold:
            filtered.append(dict(rec))
    return filtered


def _compute_revision_gain_deltas(
    cond_metrics: Mapping[str, Mapping[str, float | int]],
) -> dict[str, dict[str, float]]:
    """计算指定条件集合下的 Clean (S3_clean - K0) 与 Shifted (S3_shifted - K0_shifted) 滚动修正净增益（百分点 pp）。"""
    gains: dict[str, dict[str, float]] = {}
    metric_keys = (
        "bleu1_pct",
        "corpus_bleu1_pct",
        "beam_oracle_bleu1_pct",
        "word_acc_pct",
        "in_vocabulary_word_acc_pct",
        "word_acc_top5_pct",
        "content_word_acc_pct",
        "function_word_acc_pct",
        "cer_pct",
        "wer_pct",
        "bertscore_f1_pct",
    )
    if "S3_clean" in cond_metrics and "K0" in cond_metrics:
        c_m = cond_metrics["S3_clean"]
        a_m = cond_metrics["K0"]
        gains["clean_C_minus_A_pp"] = {
            mk: round(float(c_m.get(mk, 0.0)) - float(a_m.get(mk, 0.0)), 4)
            for mk in metric_keys
            if mk in c_m and mk in a_m
        }
    if (
        "S3_shifted_neg75s" in cond_metrics
        and "K0_shifted_neg75s" in cond_metrics
    ):
        cs_m = cond_metrics["S3_shifted_neg75s"]
        as_m = cond_metrics["K0_shifted_neg75s"]
        gains["shifted_C_minus_A_pp"] = {
            mk: round(float(cs_m.get(mk, 0.0)) - float(as_m.get(mk, 0.0)), 4)
            for mk in metric_keys
            if mk in cs_m and mk in as_m
        }
    if "S3_clean" in cond_metrics and "S3_shifted_neg75s" in cond_metrics:
        c_m = cond_metrics["S3_clean"]
        cs_m = cond_metrics["S3_shifted_neg75s"]
        gains["clean_minus_shifted_S3_pp"] = {
            mk: round(float(c_m.get(mk, 0.0)) - float(cs_m.get(mk, 0.0)), 4)
            for mk in metric_keys
            if mk in c_m and mk in cs_m
        }
    return gains


def evaluate_high_coverage_core_subsets(
    bundles_map: Mapping[str, dict],
    *,
    language: str,
    word_embedding_map: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    vocabulary_size: int | None = None,
) -> dict[str, Any]:
    """评估特定词表规模（如 Core Vocab |V|=250 或 Full Vocab |V|=2020）下的口径 1（In-Vocab 全句段）与口径 2（高覆盖核心句 Core Sentences: 100%, >=80%, >=70%）。"""
    expanded = _normalize_decoded_bundles(bundles_map)
    subset_specs = (
        ("all_segments", 0.0, "All Test Segments (In-Vocab Token Evaluation)"),
        (
            "core_subset_100pct",
            1.0,
            "Core Sentences (100% Words in V_N, 0% OOV, SimpleB2T Core Set)",
        ),
        (
            "core_subset_80pct",
            0.80,
            "High-Coverage Core Sentences (>=80% Words in V_N)",
        ),
        (
            "expanded_subset_70pct",
            0.70,
            "Expanded High-Coverage Sentences (>=70% Words in V_N)",
        ),
    )
    canonical_conds = (
        "LM-Only",
        "S0",
        "K0",
        "K1",
        "S3_clean",
        "F",
        "K0_shifted_neg75s",
        "S3_shifted_neg75s",
        "S0_brain_only",
        "K0_brain_only",
        "K1_brain_only",
        "K2_brain_only",
        "F_brain_only",
    )

    subsets_report: dict[str, Any] = {}
    for subset_key, min_cov, desc in subset_specs:
        cond_results: dict[str, dict] = {}
        seg_count_observed = 0
        for cond_name in canonical_conds:
            if cond_name not in expanded:
                continue
            raw_recs = expanded[cond_name].get("final_sentences") or []
            sub_recs = (
                list(raw_recs)
                if min_cov <= 0.0
                else _filter_records_by_vocabulary_coverage(raw_recs, min_cov)
            )
            if not sub_recs:
                continue
            seg_count_observed = max(seg_count_observed, len(sub_recs))
            cond_results[cond_name] = evaluate_six_dimensional_metrics(
                sub_recs,
                language=language,
                word_embedding_map=word_embedding_map,
                condition=cond_name,
            )
        if cond_results:
            subsets_report[subset_key] = {
                "description": desc,
                "min_vocabulary_coverage_fraction": float(min_cov),
                "segment_count": int(seg_count_observed),
                "conditions": cond_results,
                "revision_gains": _compute_revision_gain_deltas(cond_results),
            }

    return {
        "vocabulary_size": vocabulary_size,
        "subsets": subsets_report,
    }


def evaluate_vocabulary_scaling_summary(
    vocabulary_scaling_bundles: Mapping[str, dict],
    *,
    language: str,
    word_embedding_map: Mapping[str, np.ndarray | torch.Tensor] | None = None,
) -> dict[str, dict]:
    """汇总多尺度候选词表 |V| in {50, 100, 250, 500, 1000, 2020} 下的 6 维指标、高覆盖核心句表现与滚动修正净增益（对标 Jayalath 2025 Fig 4）。"""
    summary_by_scale: dict[str, dict] = {}
    ordered_entries = sorted(
        vocabulary_scaling_bundles.items(),
        key=lambda kv: int(
            kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
        ),
    )

    for scale_key, entry in ordered_entries:
        if not isinstance(entry, Mapping):
            continue
        v_size = int(
            entry.get("vocabulary_size", str(scale_key).lstrip("vV=_") or 0)
        )
        raw_conds = entry.get("conditions") or {}
        if not isinstance(raw_conds, Mapping) or not raw_conds:
            continue

        subset_eval = evaluate_high_coverage_core_subsets(
            raw_conds,
            language=language,
            word_embedding_map=word_embedding_map,
            vocabulary_size=v_size,
        )
        subsets_map = subset_eval.get("subsets", {})
        all_seg_block = subsets_map.get("all_segments", {})
        core80_block = subsets_map.get("core_subset_80pct", {})
        core100_block = subsets_map.get("core_subset_100pct", {})

        all_conds = all_seg_block.get("conditions", {})
        s3_m = all_conds.get("S3_clean", {})
        c80_s3 = (core80_block.get("conditions") or {}).get("S3_clean", {})
        c100_s3 = (core100_block.get("conditions") or {}).get("S3_clean", {})
        top_metric_keys = (
            "bleu1_pct",
            "corpus_bleu1_pct",
            "beam_oracle_bleu1_pct",
            "word_acc_pct",
            "in_vocabulary_word_acc_pct",
            "word_acc_top5_pct",
            "content_word_acc_pct",
            "function_word_acc_pct",
            "wer_pct",
            "cer_pct",
            "bertscore_f1_pct",
        )
        if core80_block and c80_s3:
            core80_block = {
                **{mk: c80_s3[mk] for mk in top_metric_keys if mk in c80_s3},
                **core80_block,
            }
        if core100_block and c100_s3:
            core100_block = {
                **{mk: c100_s3[mk] for mk in top_metric_keys if mk in c100_s3},
                **core100_block,
            }
        tok_cov = float(
            entry.get(
                "test_token_coverage_pct",
                s3_m.get("in_vocabulary_rate_pct", 0.0),
            )
        )

        summary_by_scale[f"V={v_size}"] = {
            "vocabulary_size": v_size,
            "is_full_vocabulary": bool(entry.get("is_full_vocabulary", False)),
            "test_token_coverage_pct": round(tok_cov, 4),
            "total_segment_count": int(
                entry.get(
                    "total_segment_count",
                    all_seg_block.get("segment_count", 0),
                )
            ),
            "core_100pct_segment_count": int(
                entry.get(
                    "core_100pct_segment_count",
                    core100_block.get("segment_count", 0),
                )
            ),
            "core_80pct_segment_count": int(
                entry.get(
                    "core_80pct_segment_count",
                    core80_block.get("segment_count", 0),
                )
            ),
            **{mk: s3_m[mk] for mk in top_metric_keys if mk in s3_m},
            "conditions": all_conds,
            "revision_gains": all_seg_block.get("revision_gains", {}),
            "subsets": subsets_map,
            "core_subset_80pct": core80_block,
            "core_subset_100pct": core100_block,
        }

    return summary_by_scale


def _s3_significance_stars(p_value: float) -> str:
    p = float(p_value)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _compute_s3_per_subject_breakdown(
    expanded_bundles: Mapping[str, dict],
    *,
    language: str,
    word_embedding_map: Mapping[str, np.ndarray | torch.Tensor] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """计算 S3_clean（及与 K0 对比）的逐被试 6 维指标明细与跨被试 Mean ± SD 汇总。"""
    s3_bundle = expanded_bundles.get("S3_clean") or expanded_bundles.get("S3")
    if not s3_bundle or not s3_bundle.get("final_sentences"):
        return [], {}
    s3_sents = s3_bundle.get("final_sentences") or []
    k0_sents = (expanded_bundles.get("K0") or {}).get("final_sentences") or []

    by_subj_s3: dict[str, list[dict]] = defaultdict(list)
    by_subj_k0: dict[str, list[dict]] = defaultdict(list)
    for rec in s3_sents:
        sid = str(rec.get("subject_id", "default"))
        by_subj_s3[sid].append(rec)
    for rec in k0_sents:
        sid = str(rec.get("subject_id", "default"))
        by_subj_k0[sid].append(rec)

    per_subject: list[dict[str, Any]] = []
    for sid in sorted(by_subj_s3.keys()):
        s3_m = evaluate_six_dimensional_metrics(
            by_subj_s3[sid],
            language=language,
            word_embedding_map=word_embedding_map,
        )
        row: dict[str, Any] = {
            "subject_id": sid,
            "segment_count": int(s3_m["segment_count"]),
            "total_words": int(s3_m["total_words"]),
            "bleu1_pct": round(float(s3_m["bleu1_pct"]), 4),
            "word_acc_pct": round(float(s3_m["word_acc_pct"]), 4),
            "in_vocabulary_word_acc_pct": round(
                float(s3_m["in_vocabulary_word_acc_pct"]), 4
            ),
            "content_word_acc_pct": round(
                float(s3_m["content_word_acc_pct"]), 4
            ),
            "function_word_acc_pct": round(
                float(s3_m["function_word_acc_pct"]), 4
            ),
            "wer_pct": round(float(s3_m["wer_pct"]), 4),
            "cer_pct": round(float(s3_m["cer_pct"]), 4),
            "bertscore_f1_pct": round(float(s3_m["bertscore_f1_pct"]), 4),
        }
        if sid in by_subj_k0 and by_subj_k0[sid]:
            k0_m = evaluate_six_dimensional_metrics(
                by_subj_k0[sid],
                language=language,
                word_embedding_map=word_embedding_map,
            )
            row["k0_bleu1_pct"] = round(float(k0_m["bleu1_pct"]), 4)
            row["k0_word_acc_pct"] = round(float(k0_m["word_acc_pct"]), 4)
            row["s3_minus_k0_bleu1_pp"] = round(
                float(s3_m["bleu1_pct"]) - float(k0_m["bleu1_pct"]), 4
            )
            row["s3_minus_k0_word_acc_pp"] = round(
                float(s3_m["word_acc_pct"]) - float(k0_m["word_acc_pct"]), 4
            )
        per_subject.append(row)

    summary: dict[str, Any] = {}
    if per_subject:
        b_vals = [float(r["bleu1_pct"]) for r in per_subject]
        w_vals = [float(r["word_acc_pct"]) for r in per_subject]
        summary = {
            "subject_count": len(per_subject),
            "mean_bleu1_pct": round(float(np.mean(b_vals)), 4),
            "std_bleu1_pct": round(float(np.std(b_vals, ddof=0)), 4),
            "min_bleu1_pct": round(float(np.min(b_vals)), 4),
            "max_bleu1_pct": round(float(np.max(b_vals)), 4),
            "mean_word_acc_pct": round(float(np.mean(w_vals)), 4),
            "std_word_acc_pct": round(float(np.std(w_vals, ddof=0)), 4),
            "min_word_acc_pct": round(float(np.min(w_vals)), 4),
            "max_word_acc_pct": round(float(np.max(w_vals)), 4),
        }
        d_b = [
            float(r["s3_minus_k0_bleu1_pp"])
            for r in per_subject
            if "s3_minus_k0_bleu1_pp" in r
        ]
        if d_b:
            summary["mean_s3_minus_k0_bleu1_pp"] = round(
                float(np.mean(d_b)), 4
            )
            summary["std_s3_minus_k0_bleu1_pp"] = round(
                float(np.std(d_b, ddof=0)), 4
            )
            summary["improved_subjects_count"] = int(
                sum(1 for v in d_b if v > 0.0)
            )
    return per_subject, summary


def _block_stats_for_records(
    records: Sequence[Mapping[str, Any]],
    *,
    language: str,
    mat_to_block: Mapping[str, str] | None = None,
    segments_per_block: int = 1,
    block_size: int | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """计算单个解码条件句段列表在每个独立长句材料块（默认 block_size=1）上的分子/分母汇总，供跨条件或跨 k 配对块检验精确对齐使用。"""
    chinese = language == "zh"
    eff_block_size = (
        int(block_size) if block_size is not None else int(segments_per_block)
    )
    step = max(1, eff_block_size)
    if mat_to_block is None:
        mat_ids = [
            _extract_canonical_material_id(rec, idx)
            for idx, rec in enumerate(records)
        ]
        unique_mats: list[str] = []
        m_ord: dict[str, int] = {}
        for m in mat_ids:
            if m not in m_ord:
                m_ord[m] = len(unique_mats)
                unique_mats.append(m)
        m_cnt = len(unique_mats)
        if step > 1 and m_cnt >= step * 6:
            resolved_map = {
                m: f"block_{m_ord[m] // step:04d}" for m in unique_mats
            }
        elif m_cnt >= 4:
            resolved_map = {m: f"block_{m_ord[m]:04d}" for m in unique_mats}
        else:
            resolved_map = {
                _extract_canonical_material_id(rec, idx): f"seg_{idx:04d}"
                for idx, rec in enumerate(records)
            }
    else:
        resolved_map = dict(mat_to_block)

    blocks: dict[str, dict[str, float]] = {}
    for idx, rec in enumerate(records):
        ref = [str(w) for w in (rec.get("reference_words") or ())]
        pred = [str(w) for w in (rec.get("predicted_words") or ())]
        if not ref or len(ref) != len(pred):
            continue
        seg_len = len(ref)
        mat_id = _extract_canonical_material_id(rec, idx)
        b_id = resolved_map.get(mat_id, f"block_{idx // step:04d}")
        b = blocks.setdefault(
            b_id,
            {
                "seg_count": 0.0,
                "sent_bleu1_sum": 0.0,
                "overlap_sum": 0.0,
                "word_err_sum": 0.0,
                "total_words": 0.0,
                "word_hits": 0.0,
                "in_vocab_hits": 0.0,
                "in_vocab_words": 0.0,
                "content_hits": 0.0,
                "content_words": 0.0,
                "function_hits": 0.0,
                "function_words": 0.0,
            },
        )
        ref_tuple = tuple(ref)
        pred_tuple = tuple(pred)
        w_err, _, _, _ = _cached_edit_distance_pair(
            ref_tuple, pred_tuple, chinese=chinese
        )
        overlap, seg_bleu1 = _unigram_overlap_and_bleu1(
            Counter(ref_tuple), seg_len, pred_tuple
        )
        iv_mask = rec.get("in_vocabulary_mask") or [True] * seg_len

        b["seg_count"] += 1.0
        b["sent_bleu1_sum"] += float(seg_bleu1) * 100.0
        b["overlap_sum"] += float(overlap) * 100.0
        b["word_err_sum"] += float(w_err) * 100.0
        b["total_words"] += float(seg_len)

        for pos, (r_w, p_w) in enumerate(zip(ref, pred)):
            hit = float(r_w == p_w)
            b["word_hits"] += hit * 100.0
            if pos < len(iv_mask) and bool(iv_mask[pos]):
                b["in_vocab_words"] += 1.0
                b["in_vocab_hits"] += hit * 100.0
            if is_content_word(r_w, language):
                b["content_words"] += 1.0
                b["content_hits"] += hit * 100.0
            else:
                b["function_words"] += 1.0
                b["function_hits"] += hit * 100.0
    return blocks, resolved_map


def _paired_test_from_block_maps(
    blocks_a: Mapping[str, Mapping[str, float]],
    blocks_b: Mapping[str, Mapping[str, float]],
    metric_code: str,
    *,
    n_resamples: int = 10000,
    seed: int = 20261031,
) -> dict[str, Any]:
    """基于两个对齐块字典计算指定指标（corpus_bleu1 / sent_bleu1 / wer / word_acc / in_vocab_top1 / content_acc / content_minus_function）的块自举 CI 与置换检验。"""
    common_keys = [k for k in blocks_a.keys() if k in blocks_b]
    diffs: list[float] = []
    weights: list[float] = []
    num_a_sum = 0.0
    den_a_sum = 0.0
    num_b_sum = 0.0
    den_b_sum = 0.0

    for k in common_keys:
        ba = blocks_a[k]
        bb = blocks_b[k]
        if metric_code == "corpus_bleu1":
            na, da = float(ba["overlap_sum"]), float(ba["total_words"])
            nb, db = float(bb["overlap_sum"]), float(bb["total_words"])
        elif metric_code in ("sent_bleu1", "bleu1"):
            na, da = float(ba["sent_bleu1_sum"]), float(ba["seg_count"])
            nb, db = float(bb["sent_bleu1_sum"]), float(bb["seg_count"])
        elif metric_code == "wer":
            na, da = float(ba["word_err_sum"]), float(ba["total_words"])
            nb, db = float(bb["word_err_sum"]), float(bb["total_words"])
        elif metric_code == "word_acc":
            na, da = float(ba["word_hits"]), float(ba["total_words"])
            nb, db = float(bb["word_hits"]), float(bb["total_words"])
        elif metric_code == "in_vocab_top1":
            na, da = float(ba["in_vocab_hits"]), float(ba["in_vocab_words"])
            nb, db = float(bb["in_vocab_hits"]), float(bb["in_vocab_words"])
        elif metric_code == "content_acc":
            na, da = float(ba["content_hits"]), float(ba["content_words"])
            nb, db = float(bb["content_hits"]), float(bb["content_words"])
        elif metric_code == "content_minus_function":
            na, da = float(ba["content_hits"]), float(ba["content_words"])
            nb, db = float(ba["function_hits"]), float(ba["function_words"])
        else:
            raise ValueError(f"未知配对检验指标：{metric_code!r}")

        if da <= 0.0 or db <= 0.0:
            continue
        ma = na / da
        mb = nb / db
        diffs.append(ma - mb)
        weights.append(da)
        num_a_sum += na
        den_a_sum += da
        num_b_sum += nb
        den_b_sum += db

    stat = _paired_block_permutation_and_bootstrap(
        np.asarray(diffs, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        n_resamples=n_resamples,
        seed=seed,
    )
    lhs_val = float(num_a_sum / den_a_sum) if den_a_sum > 0.0 else 0.0
    rhs_val = float(num_b_sum / den_b_sum) if den_b_sum > 0.0 else 0.0
    obs_eff = float(lhs_val - rhs_val)
    return {
        "lhs_value": round(lhs_val, 4),
        "rhs_value": round(rhs_val, 4),
        "effect_pp": round(obs_eff, 4),
        "ci_lower_pp": round(float(stat["ci_lower_pp"]), 4),
        "ci_upper_pp": round(float(stat["ci_upper_pp"]), 4),
        "p_one_sided": round(float(stat["p_one_sided"]), 6),
        "p_two_sided": round(float(stat["p_two_sided"]), 6),
        "significance": _s3_significance_stars(float(stat["p_two_sided"])),
        "blocks": int(stat["blocks"]),
    }


def _compute_s3_paired_significance(
    expanded_bundles: Mapping[str, dict],
    *,
    language: str,
    vocabulary_scaling_bundles: Mapping[str, dict] | None = None,
    multi_subject_bundles: Mapping[str | int, dict] | None = None,
    block_size: int = 1,
) -> dict[str, dict[str, Any]]:
    """计算全部 8 组固定配对统计检验（默认 block_size=1 按独立长句成块、10,000 次分层块自举与双侧符号翻转检验 + Holm 多重比较校正）。"""
    s3_bundle = expanded_bundles.get("S3_clean") or expanded_bundles.get("S3")
    if not s3_bundle or not s3_bundle.get("final_sentences"):
        return {}

    full_v_size = int(
        s3_bundle.get("vocabulary_size")
        or max(
            (
                int(str(k).lstrip("vV=_") or 0)
                for k in (vocabulary_scaling_bundles or {}).keys()
                if str(k).lstrip("vV=_").isdigit()
            ),
            default=2020,
        )
    )

    scale_conds_map: dict[int, dict[str, dict]] = {
        full_v_size: dict(expanded_bundles)
    }
    if isinstance(vocabulary_scaling_bundles, Mapping):
        for sk, s_entry in vocabulary_scaling_bundles.items():
            if not isinstance(s_entry, Mapping):
                continue
            v_int = int(
                s_entry.get("vocabulary_size")
                or (str(sk).lstrip("vV=_") if str(sk).lstrip("vV=_").isdigit() else 0)
            )
            conds = s_entry.get("conditions")
            if v_int > 0 and isinstance(conds, Mapping) and conds:
                scale_conds_map[v_int] = _normalize_decoded_bundles(
                    conds
                )

    def _get_scale_conds(v_target: int) -> tuple[int, dict[str, dict]]:
        if v_target in scale_conds_map:
            return v_target, scale_conds_map[v_target]
        if v_target >= 2000 and full_v_size in scale_conds_map:
            return full_v_size, scale_conds_map[full_v_size]
        return v_target, {}

    sig_results: dict[str, dict[str, Any]] = {}

    # 兼容旧版 4 个核心键（Full Vocab All Segs）
    s3_blocks_full, full_mat_map = _block_stats_for_records(
        s3_bundle.get("final_sentences") or [],
        language=language,
        block_size=block_size,
    )
    for other_key, label_prefix, key_prefix, base_seed in (
        ("K0", "Rolling-K2 vs Rolling-K0 (Revision Gain)", "s3_clean_vs_k0", 20261031),
        (
            "S3_shifted_neg75s",
            "Rolling-K2 vs Shifted-K2 (-75s) (Temporal Alignment)",
            "s3_clean_vs_shifted",
            20261032,
        ),
    ):
        other_b = expanded_bundles.get(other_key)
        if not other_b or not other_b.get("final_sentences"):
            continue
        o_blocks, _ = _block_stats_for_records(
            other_b.get("final_sentences") or [],
            language=language,
            mat_to_block=full_mat_map,
            block_size=block_size,
        )
        for m_label, m_suf, m_code, s_off in (
            ("BLEU-1 (%)", "bleu1", "sent_bleu1", 0),
            ("Word Acc (%)", "word_acc", "word_acc", 10),
        ):
            res = _paired_test_from_block_maps(
                s3_blocks_full,
                o_blocks,
                m_code,
                n_resamples=10000,
                seed=base_seed + s_off,
            )
            sig_results[f"{key_prefix}_{m_suf}"] = {
                "group_id": 4 if other_key == "K0" else 6,
                "label": label_prefix,
                "comparison": (
                    "Rolling-K2 vs Rolling-K0 (Clean)"
                    if other_key == "K0"
                    else "Rolling-K2 vs Shifted-K2 (-75s)"
                ),
                "scale": f"V={full_v_size}",
                "subset": "All Segs",
                "metric": m_label,
                **res,
            }

    # 8 组标准化配对统计检验（Module 5）
    all_avail_scales = tuple(sorted(set(scale_conds_map.keys()) | {50, 100, 250, full_v_size}))
    group_specs: list[tuple[int, str, str, str, tuple[int, ...], tuple[tuple[str, str], ...]]] = [
        (
            1,
            "Rolling-K2 vs Word-Only",
            "S3_clean",
            "S0",
            all_avail_scales,
            (
                ("Corpus BLEU-1", "corpus_bleu1"),
                ("Sent BLEU-1", "sent_bleu1"),
                ("WER", "wer"),
            ),
        ),
        (
            2,
            "Rolling-K0 vs Word-Only",
            "K0",
            "S0",
            all_avail_scales,
            (
                ("Corpus BLEU-1", "corpus_bleu1"),
                ("In-Vocab Top-1", "in_vocab_top1"),
            ),
        ),
        (
            3,
            "Rolling-K2 vs Full-Context",
            "S3_clean",
            "F",
            all_avail_scales,
            (
                ("Corpus BLEU-1", "corpus_bleu1"),
                ("WER", "wer"),
            ),
        ),
        (
            4,
            "Rolling-K2 vs Rolling-K0 (Clean)",
            "S3_clean",
            "K0",
            all_avail_scales,
            (
                ("Corpus BLEU-1", "corpus_bleu1"),
                ("In-Vocab Top-1", "in_vocab_top1"),
            ),
        ),
        (
            5,
            "Shifted-K2 vs Shifted-K0 (-75s)",
            "S3_shifted_neg75s",
            "K0_shifted_neg75s",
            all_avail_scales,
            (
                ("Corpus BLEU-1", "corpus_bleu1"),
                ("Sent BLEU-1", "sent_bleu1"),
            ),
        ),
        (
            6,
            "Rolling-K2 vs Shifted-K2 (-75s)",
            "S3_clean",
            "S3_shifted_neg75s",
            all_avail_scales,
            (
                ("Corpus BLEU-1", "corpus_bleu1"),
                ("In-Vocab Top-1", "in_vocab_top1"),
            ),
        ),
    ]

    seed_counter = 20262000
    for grp_id, comp_label, cond_a, cond_b, scales, metrics_spec in group_specs:
        seen_eff_v: set[int] = set()
        for raw_v in scales:
            eff_v, conds_v = _get_scale_conds(int(raw_v))
            if eff_v in seen_eff_v or not conds_v:
                continue
            seen_eff_v.add(eff_v)
            b_a = conds_v.get(cond_a)
            b_b = conds_v.get(cond_b)
            if not b_a or not b_b:
                continue
            recs_a_all = b_a.get("final_sentences") or []
            recs_b_all = b_b.get("final_sentences") or []
            if not recs_a_all or len(recs_a_all) != len(recs_b_all):
                continue

            # 若为受限词表（V=50 / V=250），同时检验 All Segs 与高覆盖核心子集（0% OOV 或 >=80%），主表展示以 All Segs（及核心子集补充）为准
            blocks_a_all, mat_map_all = _block_stats_for_records(
                recs_a_all, language=language, block_size=block_size
            )
            blocks_b_all, _ = _block_stats_for_records(
                recs_b_all,
                language=language,
                mat_to_block=mat_map_all,
                block_size=block_size,
            )
            subset_label = (
                "0% OOV / All Segs" if eff_v == full_v_size else "All Segs"
            )
            for m_disp, m_code in metrics_spec:
                seed_counter += 1
                res = _paired_test_from_block_maps(
                    blocks_a_all,
                    blocks_b_all,
                    m_code,
                    n_resamples=10000,
                    seed=seed_counter,
                )
                entry_key = f"g{grp_id}_V{eff_v}_all_{m_code}"
                sig_results[entry_key] = {
                    "group_id": grp_id,
                    "label": f"{comp_label} (V={eff_v})",
                    "comparison": comp_label,
                    "scale": f"V={eff_v}",
                    "subset": subset_label,
                    "metric": m_disp,
                    **res,
                }

            if eff_v < full_v_size:
                recs_a_100 = _filter_records_by_vocabulary_coverage(
                    recs_a_all, 1.0
                )
                recs_b_100 = _filter_records_by_vocabulary_coverage(
                    recs_b_all, 1.0
                )
                if len(recs_a_100) >= 10 and len(recs_a_100) == len(recs_b_100):
                    core_recs_a, core_recs_b, core_sub_lbl = (
                        recs_a_100,
                        recs_b_100,
                        "0% OOV",
                    )
                else:
                    recs_a_80 = _filter_records_by_vocabulary_coverage(
                        recs_a_all, 0.80
                    )
                    recs_b_80 = _filter_records_by_vocabulary_coverage(
                        recs_b_all, 0.80
                    )
                    if len(recs_a_80) >= 4 and len(recs_a_80) == len(recs_b_80):
                        core_recs_a, core_recs_b, core_sub_lbl = (
                            recs_a_80,
                            recs_b_80,
                            ">=80% In-Vocab",
                        )
                    else:
                        core_recs_a, core_recs_b, core_sub_lbl = (
                            recs_a_100,
                            recs_b_100,
                            "0% OOV",
                        )
                if core_recs_a and len(core_recs_a) == len(core_recs_b):
                    ba_core, cmap_core = _block_stats_for_records(
                        core_recs_a, language=language, block_size=block_size
                    )
                    bb_core, _ = _block_stats_for_records(
                        core_recs_b,
                        language=language,
                        mat_to_block=cmap_core,
                        block_size=block_size,
                    )
                    for m_disp, m_code in metrics_spec:
                        seed_counter += 1
                        res_c = _paired_test_from_block_maps(
                            ba_core,
                            bb_core,
                            m_code,
                            n_resamples=10000,
                            seed=seed_counter,
                        )
                        c_key = f"g{grp_id}_V{eff_v}_core_{m_code}"
                        sig_results[c_key] = {
                            "group_id": grp_id,
                            "label": f"{comp_label} (V={eff_v}, {core_sub_lbl})",
                            "comparison": comp_label,
                            "scale": f"V={eff_v}",
                            "subset": core_sub_lbl,
                            "metric": m_disp,
                            **res_c,
                        }

    # Group 7: Rolling-K2 Content vs Word-Only Content & Rolling-K2 Content vs Function (|V|=250, 0% OOV & All Segs)
    eff_250, conds_250 = _get_scale_conds(250)
    if not conds_250:
        eff_250, conds_250 = full_v_size, dict(expanded_bundles)
    s3_250 = (conds_250.get("S3_clean") or {}).get("final_sentences") or []
    s0_250 = (conds_250.get("S0") or {}).get("final_sentences") or []
    if s3_250:
        s3_250_100 = _filter_records_by_vocabulary_coverage(s3_250, 1.0)
        s0_250_100 = (
            _filter_records_by_vocabulary_coverage(s0_250, 1.0)
            if s0_250
            else []
        )
        for sub_tag, sub_lbl, r_s3, r_s0 in (
            ("core100", "0% OOV", s3_250_100, s0_250_100),
            ("all", "All Segs", s3_250, s0_250),
        ):
            if not r_s3:
                continue
            b_s3, m_250 = _block_stats_for_records(
                r_s3, language=language, block_size=block_size
            )
            if r_s0 and len(r_s0) == len(r_s3):
                b_s0, _ = _block_stats_for_records(
                    r_s0,
                    language=language,
                    mat_to_block=m_250,
                    block_size=block_size,
                )
                seed_counter += 1
                res_c_vs_s0 = _paired_test_from_block_maps(
                    b_s3,
                    b_s0,
                    "content_acc",
                    n_resamples=10000,
                    seed=seed_counter,
                )
                sig_results[f"g7_V{eff_250}_{sub_tag}_content_acc"] = {
                    "group_id": 7,
                    "label": f"Rolling-K2 Content vs Word-Only Content (V={eff_250}, {sub_lbl})",
                    "comparison": "Rolling-K2 Content vs Word-Only Content",
                    "scale": f"V={eff_250}",
                    "subset": sub_lbl,
                    "metric": "Content Acc",
                    **res_c_vs_s0,
                }
            seed_counter += 1
            res_c_vs_f = _paired_test_from_block_maps(
                b_s3,
                b_s3,
                "content_minus_function",
                n_resamples=10000,
                seed=seed_counter,
            )
            sig_results[f"g7_V{eff_250}_{sub_tag}_content_minus_function"] = {
                "group_id": 7,
                "label": f"Rolling-K2 Content vs Function (V={eff_250}, {sub_lbl})",
                "comparison": "Rolling-K2 Content vs Function",
                "scale": f"V={eff_250}",
                "subset": sub_lbl,
                "metric": "Content - Function",
                **res_c_vs_f,
            }

    # Group 8: Multi-Subject k=5 / k=10 (and k=2) vs k=1 across all available |V| scales
    if isinstance(multi_subject_bundles, Mapping) and multi_subject_bundles:
        k1_bundle = multi_subject_bundles.get("k=1") or multi_subject_bundles.get(1)
        for k_target in (2, 5, 10):
            kt_bundle = (
                multi_subject_bundles.get(f"k={k_target}")
                or multi_subject_bundles.get(k_target)
            )
            if not isinstance(kt_bundle, Mapping):
                continue
            by_v_t_keys = {
                int(str(vk).lstrip("vV=_"))
                for vk in (kt_bundle.get("by_vocabulary_size") or {}).keys()
                if str(vk).lstrip("vV=_").isdigit()
            }
            g8_v_targets = sorted(
                {50, 100, 250, full_v_size}
                | set(scale_conds_map.keys())
                | by_v_t_keys
            )
            for v_target in g8_v_targets:
                v_key = f"V={v_target}"
                # 提取 k=1 在 v_target 下的 S3_clean 句段
                recs_k1_v: list[dict] = []
                if isinstance(k1_bundle, Mapping):
                    by_v_1 = k1_bundle.get("by_vocabulary_size") or {}
                    if v_key in by_v_1 and isinstance(by_v_1[v_key], Mapping):
                        c_map_1 = by_v_1[v_key].get("conditions") if isinstance(by_v_1[v_key].get("conditions"), Mapping) else by_v_1[v_key]
                        recs_k1_v = (
                            (c_map_1.get("S3_clean") or {})
                            .get("final_sentences")
                            or []
                        )
                    elif v_target == full_v_size:
                        recs_k1_v = k1_bundle.get("final_sentences") or []
                if not recs_k1_v:
                    _, c_v = _get_scale_conds(v_target)
                    recs_k1_v = (c_v.get("S3_clean") or {}).get(
                        "final_sentences"
                    ) or []

                # 提取 k=k_target 在 v_target 下的 S3_clean 句段
                recs_kt_v: list[dict] = []
                by_v_t = kt_bundle.get("by_vocabulary_size") or {}
                if v_key in by_v_t and isinstance(by_v_t[v_key], Mapping):
                    c_map_t = by_v_t[v_key].get("conditions") if isinstance(by_v_t[v_key].get("conditions"), Mapping) else by_v_t[v_key]
                    recs_kt_v = (
                        (c_map_t.get("S3_clean") or {})
                        .get("final_sentences")
                        or []
                    )
                elif v_target == full_v_size:
                    recs_kt_v = kt_bundle.get("final_sentences") or []

                if not recs_k1_v or not recs_kt_v:
                    continue
                b_k1, mat_map_k1 = _block_stats_for_records(
                    recs_k1_v, language=language, block_size=block_size
                )
                b_kt, _ = _block_stats_for_records(
                    recs_kt_v,
                    language=language,
                    mat_to_block=mat_map_k1,
                    block_size=block_size,
                )
                for m_disp, m_code in (
                    ("Sent BLEU-1", "sent_bleu1"),
                    ("WER", "wer"),
                    ("In-Vocab Top-1", "in_vocab_top1"),
                ):
                    seed_counter += 1
                    res_ms = _paired_test_from_block_maps(
                        b_kt,
                        b_k1,
                        m_code,
                        n_resamples=10000,
                        seed=seed_counter,
                    )
                    sig_results[f"g8_k{k_target}_vs_k1_V{v_target}_{m_code}"] = {
                        "group_id": 8,
                        "label": f"Multi-Subject k={k_target} vs k=1 (V={v_target})",
                        "comparison": f"Multi-Subject k={k_target} vs k=1",
                        "scale": f"V={v_target}",
                        "subset": "All Segs",
                        "metric": m_disp,
                        "k_observations": int(k_target),
                        **res_ms,
                    }

    # 按检验组（group_id 1..8）内部执行 Holm 多重比较校正
    by_group: dict[int, list[str]] = defaultdict(list)
    for k_name, item in sig_results.items():
        if k_name.startswith("g") and isinstance(item.get("group_id"), int):
            by_group[int(item["group_id"])].append(k_name)
    for grp_id, keys_in_grp in by_group.items():
        raw_ps = [float(sig_results[k]["p_two_sided"]) for k in keys_in_grp]
        adj_ps = holm_adjust(raw_ps)
        for k_name, p_adj in zip(keys_in_grp, adj_ps):
            sig_results[k_name]["p_holm"] = round(float(p_adj), 6)
            sig_results[k_name]["significance_holm"] = _s3_significance_stars(
                float(p_adj)
            )
    for k_name, item in sig_results.items():
        if "p_holm" not in item:
            p2 = float(item.get("p_two_sided", 1.0))
            item["p_holm"] = round(p2, 6)
            item["significance_holm"] = _s3_significance_stars(p2)

    return sig_results



def evaluate_and_export_sequence_decoding(
    decoded_bundles: Mapping[str, dict],
    *,
    dataset: str,
    language: str,
    split: str,
    output_dir: str | Path,
    word_embedding_map: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    multi_subject_bundles: Mapping[str | int, dict] | None = None,
    vocabulary_scaling_bundles: Mapping[str, dict] | None = None,
    core_vocab_size: int | str = "full",
    core_vocab_sizes: Sequence[int | str] = (50, 100, "full"),
    state_hyperparameters: Mapping[str, Any] | None = None,
    seed: int | None = None,
    exp_tag: str | None = None,
    cms_eta: float | None = None,
    f_condition_mode: str | None = None,
    plot: bool = True,
) -> dict:
    """汇总四级延迟预算阶梯（Word-Only, Rolling-K0, Rolling-K1, Rolling-K2, Full-Context 及对照）、|V| 多尺度闭集词表与 k 跨被试聚合评估。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expanded_bundles = _normalize_decoded_bundles(
        decoded_bundles
    )
    condition_summaries: dict[str, dict] = {}
    figure_paths: dict[str, str] = {}
    multi_subject_summaries: dict[str, dict] = {}
    vocabulary_scaling_summaries: dict[str, dict] = {}
    core_vocabulary_benchmark: dict[str, Any] = {}
    core_vocabulary_benchmarks: dict[str, Any] = {}

    for key, bundle in expanded_bundles.items():
        records = bundle["final_sentences"]
        metrics_6d = evaluate_six_dimensional_metrics(
            records,
            language=language,
            word_embedding_map=word_embedding_map,
            condition=key,
        )
        metrics_6d["mean_seconds_per_segment"] = float(
            bundle.get("mean_seconds_per_segment", 0.0)
        )
        condition_summaries[key] = metrics_6d


    canonical_order = (
        "S0",
        "S0_clean",
        "K0",
        "K1",
        "S3_clean",
        "F",
        "F_clean",
        "LM-Only",
        "LM-Only_clean",
        "K0_shifted_neg75s",
        "S3_shifted_neg75s",
        "S0_brain_only",
        "K0_brain_only",
        "K1_brain_only",
        "K2_brain_only",
        "F_brain_only",
    )
    ordered_summaries: dict[str, dict] = {}
    for k in canonical_order:
        if k in condition_summaries:
            ordered_summaries[k] = condition_summaries[k]
    for k, v in condition_summaries.items():
        if k not in ordered_summaries:
            ordered_summaries[k] = v
    condition_summaries = ordered_summaries

    full_vocab_sz = int(
        expanded_bundles.get("S3_clean", {}).get("vocabulary_size")
        or max(
            (
                int(str(k).lstrip("vV=_") or 0)
                for k in (vocabulary_scaling_bundles or {}).keys()
                if str(k).lstrip("vV=_").isdigit()
            ),
            default=2020,
        )
    )
    full_vocab_core_subsets = evaluate_high_coverage_core_subsets(
        expanded_bundles,
        language=language,
        word_embedding_map=word_embedding_map,
        vocabulary_size=full_vocab_sz,
    )
    full_v_key = f"V={full_vocab_sz}"

    if vocabulary_scaling_bundles:
        vocabulary_scaling_summaries = evaluate_vocabulary_scaling_summary(
            vocabulary_scaling_bundles,
            language=language,
            word_embedding_map=word_embedding_map,
        )

        def _resolve_scale_int(val: Any) -> int:
            if val is None:
                return full_vocab_sz
            if isinstance(val, str):
                s = val.strip().lower()
                if s in ("full", "all", "") or not s.lstrip("-").isdigit():
                    return full_vocab_sz
                v_i = int(s)
                return full_vocab_sz if v_i <= 0 else v_i
            v_i = int(val)
            return full_vocab_sz if v_i <= 0 else v_i

        target_core_sizes = sorted(
            {_resolve_scale_int(v) for v in (core_vocab_sizes or ())}
            | ({_resolve_scale_int(core_vocab_size)} if core_vocab_size is not None else set())
            | {
                int(s_val["vocabulary_size"])
                for s_val in vocabulary_scaling_summaries.values()
                if isinstance(s_val, Mapping) and int(s_val.get("vocabulary_size", 0)) > 0
            }
        )
        for v_sz in target_core_sizes:
            v_key = f"V={v_sz}"
            if v_key in vocabulary_scaling_bundles:
                v_entry = vocabulary_scaling_bundles[v_key]
                scaled_sum = vocabulary_scaling_summaries.get(v_key)
                if isinstance(scaled_sum, Mapping) and isinstance(
                    scaled_sum.get("subsets"), Mapping
                ):
                    v_bench = {
                        "vocabulary_size": int(v_sz),
                        "is_full_vocabulary": bool(
                            v_entry.get("is_full_vocabulary", v_sz == full_vocab_sz)
                        ),
                        "subsets": scaled_sum["subsets"],
                    }
                else:
                    v_bench = evaluate_high_coverage_core_subsets(
                        v_entry.get("conditions") or {},
                        language=language,
                        word_embedding_map=word_embedding_map,
                        vocabulary_size=int(v_sz),
                    )
                    v_bench["is_full_vocabulary"] = bool(
                        v_entry.get("is_full_vocabulary", v_sz == full_vocab_sz)
                    )
                v_bench["test_token_coverage_pct"] = v_entry.get(
                    "test_token_coverage_pct"
                )
                core_vocabulary_benchmarks[v_key] = v_bench

    if isinstance(full_vocab_core_subsets.get("subsets"), Mapping) and full_vocab_core_subsets["subsets"]:
        s3_fv = condition_summaries.get("S3_clean") or {}
        fv_tot_w = max(1, int(s3_fv.get("total_words", 0)))
        fv_iv_w = int(s3_fv.get("in_vocabulary_word_count", fv_tot_w))
        fv_cov_pct = round(fv_iv_w / fv_tot_w * 100.0, 4)
        fv_subsets = full_vocab_core_subsets["subsets"]
        fv_all_blk = fv_subsets.get("all_segments") or {}
        fv_c80_blk = fv_subsets.get("core_subset_80pct") or {}
        fv_c100_blk = fv_subsets.get("core_subset_100pct") or {}
        if full_v_key not in core_vocabulary_benchmarks:
            core_vocabulary_benchmarks[full_v_key] = {
                "vocabulary_size": full_vocab_sz,
                "is_full_vocabulary": True,
                "test_token_coverage_pct": fv_cov_pct,
                "subsets": fv_subsets,
            }
        if full_v_key not in vocabulary_scaling_summaries:
            vocabulary_scaling_summaries[full_v_key] = {
                "vocabulary_size": full_vocab_sz,
                "is_full_vocabulary": True,
                "test_token_coverage_pct": fv_cov_pct,
                "total_segment_count": int(fv_all_blk.get("segment_count", 0)),
                "core_100pct_segment_count": int(
                    fv_c100_blk.get("segment_count", 0)
                ),
                "core_80pct_segment_count": int(
                    fv_c80_blk.get("segment_count", 0)
                ),
                "bleu1_pct": float(s3_fv.get("bleu1_pct", 0.0)),
                "corpus_bleu1_pct": float(s3_fv.get("corpus_bleu1_pct", 0.0)),
                "beam_oracle_bleu1_pct": float(
                    s3_fv.get("beam_oracle_bleu1_pct", 0.0)
                ),
                "word_acc_pct": float(s3_fv.get("word_acc_pct", 0.0)),
                "in_vocabulary_word_acc_pct": float(
                    s3_fv.get("in_vocabulary_word_acc_pct", 0.0)
                ),
                "word_acc_top5_pct": float(
                    s3_fv.get("word_acc_top5_pct", 0.0)
                ),
                "content_word_acc_pct": float(
                    s3_fv.get("content_word_acc_pct", 0.0)
                ),
                "function_word_acc_pct": float(
                    s3_fv.get("function_word_acc_pct", 0.0)
                ),
                "wer_pct": float(s3_fv.get("wer_pct", 0.0)),
                "cer_pct": float(s3_fv.get("cer_pct", 0.0)),
                "bertscore_f1_pct": float(s3_fv.get("bertscore_f1_pct", 0.0)),
                "conditions": fv_all_blk.get("conditions", {}),
                "revision_gains": fv_all_blk.get("revision_gains", {}),
                "subsets": fv_subsets,
                "core_subset_80pct": fv_c80_blk,
                "core_subset_100pct": fv_c100_blk,
            }

    if core_vocabulary_benchmarks:
        if (
            isinstance(core_vocab_size, str)
            and core_vocab_size.strip().lower() in ("full", "all", "")
        ) or core_vocab_size is None:
            core_key = full_v_key
        elif isinstance(core_vocab_size, str) and core_vocab_size.strip().lstrip("-").isdigit():
            core_key = f"V={int(core_vocab_size.strip())}"
        else:
            core_key = f"V={int(core_vocab_size)}"
        if core_key in core_vocabulary_benchmarks:
            core_vocabulary_benchmark = core_vocabulary_benchmarks[core_key]
        else:
            first_k = sorted(
                core_vocabulary_benchmarks.keys(),
                key=lambda s: int(s.lstrip("vV=")),
            )[-1]
            core_vocabulary_benchmark = core_vocabulary_benchmarks[first_k]

    if multi_subject_bundles:
        for raw_key, ms_bundle in multi_subject_bundles.items():
            ms_records = ms_bundle.get("final_sentences") or []
            if not ms_records:
                continue
            k_obs = int(
                ms_bundle.get(
                    "k_observations",
                    str(raw_key).lstrip("kK=_"),
                )
            )
            ms_metrics = evaluate_six_dimensional_metrics(
                ms_records,
                language=language,
                word_embedding_map=word_embedding_map,
                condition="S3_clean",
            )
            ms_metrics["k_observations"] = k_obs
            ms_metrics["effective_k_observations"] = int(
                ms_bundle.get("effective_k_observations", k_obs)
            )
            ms_metrics["mean_seconds_per_segment"] = float(
                ms_bundle.get("mean_seconds_per_segment", 0.0)
            )
            ms_conds_map = ms_bundle.get("conditions")
            if isinstance(ms_conds_map, Mapping) and ms_conds_map:
                ms_subsets = evaluate_high_coverage_core_subsets(
                    ms_conds_map,
                    language=language,
                    word_embedding_map=word_embedding_map,
                    vocabulary_size=full_vocab_sz,
                )
            else:
                ms_subsets = evaluate_high_coverage_core_subsets(
                    {"S3_clean": ms_bundle},
                    language=language,
                    word_embedding_map=word_embedding_map,
                    vocabulary_size=full_vocab_sz,
                )
            ms_metrics["subsets"] = ms_subsets.get("subsets", {})

            ms_by_vocab_eval: dict[str, dict] = {}
            ms_by_vocab = ms_bundle.get("by_vocabulary_size")
            if isinstance(ms_by_vocab, Mapping) and ms_by_vocab:
                ms_by_vocab_eval = evaluate_vocabulary_scaling_summary(
                    ms_by_vocab,
                    language=language,
                    word_embedding_map=word_embedding_map,
                )
            if full_v_key not in ms_by_vocab_eval and ms_metrics["subsets"]:
                ms_all_blk = ms_metrics["subsets"].get("all_segments") or {}
                ms_c80_blk = ms_metrics["subsets"].get("core_subset_80pct") or {}
                ms_c100_blk = ms_metrics["subsets"].get("core_subset_100pct") or {}
                ms_tot_w = max(1, int(ms_metrics.get("total_words", 0)))
                ms_iv_w = int(ms_metrics.get("in_vocabulary_word_count", ms_tot_w))
                ms_by_vocab_eval[full_v_key] = {
                    "vocabulary_size": full_vocab_sz,
                    "is_full_vocabulary": True,
                    "test_token_coverage_pct": round(
                        ms_iv_w / ms_tot_w * 100.0, 4
                    ),
                    "total_segment_count": int(
                        ms_all_blk.get("segment_count", 0)
                    ),
                    "core_100pct_segment_count": int(
                        ms_c100_blk.get("segment_count", 0)
                    ),
                    "core_80pct_segment_count": int(
                        ms_c80_blk.get("segment_count", 0)
                    ),
                    "bleu1_pct": float(ms_metrics.get("bleu1_pct", 0.0)),
                    "corpus_bleu1_pct": float(
                        ms_metrics.get("corpus_bleu1_pct", 0.0)
                    ),
                    "beam_oracle_bleu1_pct": float(
                        ms_metrics.get("beam_oracle_bleu1_pct", 0.0)
                    ),
                    "word_acc_pct": float(ms_metrics.get("word_acc_pct", 0.0)),
                    "in_vocabulary_word_acc_pct": float(
                        ms_metrics.get("in_vocabulary_word_acc_pct", 0.0)
                    ),
                    "word_acc_top5_pct": float(
                        ms_metrics.get("word_acc_top5_pct", 0.0)
                    ),
                    "content_word_acc_pct": float(
                        ms_metrics.get("content_word_acc_pct", 0.0)
                    ),
                    "function_word_acc_pct": float(
                        ms_metrics.get("function_word_acc_pct", 0.0)
                    ),
                    "wer_pct": float(ms_metrics.get("wer_pct", 0.0)),
                    "cer_pct": float(ms_metrics.get("cer_pct", 0.0)),
                    "bertscore_f1_pct": float(
                        ms_metrics.get("bertscore_f1_pct", 0.0)
                    ),
                    "conditions": ms_all_blk.get("conditions", {}),
                    "revision_gains": ms_all_blk.get("revision_gains", {}),
                    "subsets": ms_metrics["subsets"],
                    "core_subset_80pct": ms_c80_blk,
                    "core_subset_100pct": ms_c100_blk,
                }
            if ms_by_vocab_eval:
                ms_metrics["by_vocabulary_size"] = ms_by_vocab_eval
            multi_subject_summaries[f"k={k_obs}"] = ms_metrics

        # 为 k=1 补齐来自 vocabulary_scaling_summaries 的策略指标，并计算 k>1 相对 k=1 的增益与配对块检验 p 值
        k1_sum = multi_subject_summaries.get("k=1")
        if isinstance(k1_sum, dict) and vocabulary_scaling_summaries:
            by_v_k1 = k1_sum.setdefault("by_vocabulary_size", {})
            for vk, vs_entry in vocabulary_scaling_summaries.items():
                if vk not in by_v_k1:
                    by_v_k1[vk] = copy.deepcopy(vs_entry)
                else:
                    k1_conds = by_v_k1[vk].setdefault("conditions", {})
                    for ck, cm in (vs_entry.get("conditions") or {}).items():
                        if ck not in k1_conds:
                            k1_conds[ck] = copy.deepcopy(cm)

    per_subject, per_subject_summary = _compute_s3_per_subject_breakdown(
        expanded_bundles,
        language=language,
        word_embedding_map=word_embedding_map,
    )
    statistical_significance = _compute_s3_paired_significance(
        expanded_bundles,
        language=language,
        vocabulary_scaling_bundles=vocabulary_scaling_bundles,
        multi_subject_bundles=multi_subject_bundles,
    )
    revision_gains = _compute_revision_gain_deltas(condition_summaries)

    report: dict[str, Any] = {
        "schema_version": 3,
        "protocol": "sequence_decoding_s3_evaluation",
        "dataset": dataset,
        "language": language,
        "split": split,
        "conditions": condition_summaries,
        "revision_gains": revision_gains,
        "statistical_significance": statistical_significance,
        "paired_statistical_significance": statistical_significance,
        "per_subject": per_subject,
        "per_subject_summary": per_subject_summary,
        "core_vocabulary_benchmark": core_vocabulary_benchmark,
        "core_vocabulary_benchmarks": core_vocabulary_benchmarks,
        "full_vocab_core_subsets": full_vocab_core_subsets,
        "vocabulary_scaling": vocabulary_scaling_summaries,
        "multi_subject_scaling": multi_subject_summaries,
        "figures": figure_paths,
    }
    if state_hyperparameters is not None:
        report["state_hyperparameters"] = dict(state_hyperparameters)
    if seed is not None:
        report["seed"] = int(seed)
    if exp_tag is not None:
        report["exp_tag"] = exp_tag
    if cms_eta is not None:
        report["cms_eta"] = float(cms_eta)
    if f_condition_mode is not None:
        report["f_condition_mode"] = str(f_condition_mode)

    report["exports"] = {}
    report["tables"] = {}
    summary_path = out_dir / f"s3_evaluation_summary_{split}.json"
    save_json(summary_path, report)
    report["report_path"] = str(summary_path)
    return report


def _aggregate_numeric_dicts_across_seeds(
    items: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """递归聚合多个随机种子的数值字典，保留原字段名作为跨种子均值，并附加 _std / _min / _max。"""
    if not items:
        return {}
    if len(items) == 1:
        out = copy.deepcopy(dict(items[0]))
        for k, v in list(out.items()):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{k}_std"] = 0.0
            elif isinstance(v, Mapping):
                out[k] = _aggregate_numeric_dicts_across_seeds([v])
            elif (
                isinstance(v, list)
                and v
                and all(isinstance(x, Mapping) for x in v)
            ):
                out[k] = [
                    _aggregate_numeric_dicts_across_seeds([x]) for x in v
                ]
        return out

    result: dict[str, Any] = {}
    all_keys: list[str] = []
    for it in items:
        for k in it.keys():
            if k not in all_keys:
                all_keys.append(k)

    for k in all_keys:
        vals = [it[k] for it in items if k in it and it[k] is not None]
        if not vals:
            continue
        first = vals[0]
        if isinstance(first, bool):
            result[k] = bool(first)
        elif isinstance(first, (int, float)):
            f_vals = [
                float(x)
                for x in vals
                if isinstance(x, (int, float)) and not isinstance(x, bool)
            ]
            mean_v = float(np.mean(f_vals))
            std_v = float(np.std(f_vals, ddof=1 if len(f_vals) > 1 else 0))
            if all(isinstance(x, int) for x in vals) and k.endswith(
                ("_count", "total_words", "_words", "blocks", "k_observations", "vocabulary_size")
            ):
                result[k] = int(round(mean_v))
            else:
                result[k] = round(mean_v, 6)
            result[f"{k}_std"] = round(std_v, 6)
            result[f"{k}_min"] = round(float(np.min(f_vals)), 6)
            result[f"{k}_max"] = round(float(np.max(f_vals)), 6)
        elif isinstance(first, Mapping):
            sub_maps = [x for x in vals if isinstance(x, Mapping)]
            result[k] = _aggregate_numeric_dicts_across_seeds(sub_maps)
        elif (
            isinstance(first, list)
            and first
            and all(isinstance(x, Mapping) for x in first)
        ):
            min_len = min(
                len(x) for x in vals if isinstance(x, list)
            )
            result[k] = [
                _aggregate_numeric_dicts_across_seeds(
                    [x[idx] for x in vals if isinstance(x, list) and idx < len(x)]
                )
                for idx in range(min_len)
            ]
        else:
            result[k] = copy.deepcopy(first)
    return result


def summarize_and_export_multi_seed_sequence_decoding(
    sequence_decoding_root: str | Path,
    *,
    split: str = "test",
    dataset: str | None = None,
    exp_tag: str | None = None,
    seed_reports: Sequence[Mapping[str, Any]] | None = None,
    seed_summaries: Sequence[Mapping[str, Any]] | None = None,
    plot: bool = True,
    verbose: bool = False,
    print_report: bool = False,
) -> dict[str, Any]:
    """跨所有可用随机种子（包含单种子 n=1 场景）聚合 S3 序列解码评估结果，并在外部 sequence_decoding/ 根目录导出汇总 JSON、Markdown、TXT 与 CSV 表。"""
    import json

    seq_root = Path(sequence_decoding_root).resolve()
    seq_root.mkdir(parents=True, exist_ok=True)

    by_seed: dict[int, dict[str, Any]] = {}
    provided_reports = seed_reports if seed_reports is not None else seed_summaries
    if provided_reports:
        for rep in provided_reports:
            if isinstance(rep, Mapping) and (
                "conditions" in rep or "vocabulary_scaling" in rep
            ):
                s_val = int(rep.get("seed", 0))
                by_seed[s_val] = dict(rep)

    for child in sorted(seq_root.iterdir()):
        if not child.is_dir() or not child.name.startswith("seed-"):
            continue
        suffix = child.name.split("seed-", 1)[1]
        if not suffix.isdigit():
            continue
        s_int = int(suffix)
        summary_file = (
            child / f"evaluation_{split}" / f"s3_evaluation_summary_{split}.json"
        )
        if not summary_file.is_file():
            summary_file = child / f"s3_evaluation_summary_{split}.json"
        dec_file = child / f"sequence_decoding_{split}.json"
        cal_file = child / "state_hyperparameter_calibration.json"
        if summary_file.is_file():
            try:
                loaded_rep = json.loads(summary_file.read_text(encoding="utf-8"))
                if isinstance(loaded_rep, dict):
                    loaded_rep.setdefault("seed", s_int)
                    if "state_hyperparameters" not in loaded_rep:
                        if dec_file.is_file():
                            dec_obj = json.loads(
                                dec_file.read_text(encoding="utf-8")
                            )
                            if dec_obj.get("state_hyperparameters"):
                                loaded_rep["state_hyperparameters"] = dec_obj[
                                    "state_hyperparameters"
                                ]
                        elif cal_file.is_file():
                            cal_obj = json.loads(
                                cal_file.read_text(encoding="utf-8")
                            )
                            if cal_obj.get("states"):
                                loaded_rep["state_hyperparameters"] = cal_obj[
                                    "states"
                                ]
                    if s_int not in by_seed:
                        by_seed[s_int] = loaded_rep
            except Exception:
                pass

    if not by_seed:
        return {}

    ordered_seeds = sorted(by_seed.keys())
    reports_list = [by_seed[s] for s in ordered_seeds]
    first_rep = reports_list[0]
    ds_name = str(dataset or first_rep.get("dataset") or "pallier2025")
    lang = str(
        first_rep.get("language") or ("fr" if ds_name.startswith("pallier2025") else "en")
    )
    resolved_tag = exp_tag if exp_tag is not None else first_rep.get("exp_tag")

    agg_report = _aggregate_numeric_dicts_across_seeds(reports_list)
    agg_report["schema_version"] = 3
    agg_report["protocol"] = "multi_seed_sequence_decoding_s3_evaluation"
    agg_report["is_multi_seed_summary"] = True
    agg_report["dataset"] = ds_name
    agg_report["language"] = lang
    agg_report["split"] = split
    agg_report["seeds"] = ordered_seeds
    agg_report["seed_count"] = len(ordered_seeds)
    if resolved_tag is not None:
        agg_report["exp_tag"] = resolved_tag

    agg_report["exports"] = {}
    agg_report["tables"] = {}
    agg_report["figures"] = {}

    multi_json_path = seq_root / f"multi_seed_sequence_decoding_{split}.json"
    summary_json_path = seq_root / f"s3_evaluation_summary_{split}.json"
    save_json(multi_json_path, agg_report)
    save_json(summary_json_path, agg_report)

    agg_report["multi_seed_json_path"] = str(multi_json_path)
    agg_report["report_path"] = str(multi_json_path)
    if print_report:
        print(format_sequence_decoding_report(agg_report, verbose=verbose), flush=True)
    return agg_report


def format_sequence_decoding_report(
    report: Mapping[str, Any], *, verbose: bool = False
) -> str:
    """紧凑输出序列解码评测摘要（用于 CLI 终端反馈）。"""
    del verbose
    ds = report.get("dataset", "unknown")
    split = report.get("split", "val")
    lines = [f"[Sequence Decoding] dataset={ds} split={split}"]
    for cond_name, entry in (report.get("conditions") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        wer = entry.get("wer")
        wa = entry.get("word_accuracy")
        if wer is not None or wa is not None:
            lines.append(
                f"  - {cond_name}: WER={wer:.4f}" if isinstance(wer, (int, float))
                else f"  - {cond_name}: word_acc={wa}"
            )
    return "\n".join(lines)


def print_sequence_decoding_report(
    report: Mapping[str, Any], *, verbose: bool = False
) -> None:
    print(format_sequence_decoding_report(report, verbose=verbose), flush=True)

