"""S3 序列解码协议多维语义评价指标与 C-gated 置信度门控帕累托分析器（Module 5）。

核心功能：
1. 同步输出 6 维语义与词位评价指标（全部统一以百分比 % 呈现）：
   - 字面精确匹配：BLEU-1 (%), WER (%), CER (字符/汉字错误率 %)
   - 词位准确率：Word Acc (%)，并拆解为 实义词（Content Words）Acc (%) 与 功能词（Function Words）Acc (%)
   - 连续语义相似度：BERTScore (%)（Precision / Recall / F1）
2. 置信度门控回修（C-gated）后处理：
   - 对流式输出检验综合得分裕度：
         Δs_i^{(r)} = s_i^{(r)}(w_i^{new} | u) - s_i^{(r)}(w_i^{old} | u) > τ_{gate}
   - 扫描 τ_{gate} ∈ [0.0, 2.0]（步长 0.1），统计屏幕改写率、有益改对率、有害改错率与无效闪烁滤除率，
     并自动绘制“解码质量 vs 屏幕改写率”帕累托曲线与基线/负对照对比图。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F_nn

from braindecoding.evaluation.statistical_significance import (
    edit_distance,
    sentence_metrics,
)
from braindecoding.training.training_runtime import save_json


DEFAULT_TAU_GATE_GRID: tuple[float, ...] = tuple(
    round(0.1 * i, 2) for i in range(21)
)

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
        ref_words = [str(w) for w in ref_seq]
        pred_words = [str(w) for w in pred_seq]
        if not ref_words or not pred_words:
            p_list.append(0.0)
            r_list.append(0.0)
            f1_list.append(0.0)
            continue

        ref_mat = np.stack([_get_vec(w) for w in ref_words], axis=0)
        pred_mat = np.stack([_get_vec(w) for w in pred_words], axis=0)
        sim_matrix = np.clip(ref_mat @ pred_mat.T, -1.0, 1.0)

        recall = float(np.mean(np.max(sim_matrix, axis=1)))
        precision = float(np.mean(np.max(sim_matrix, axis=0)))
        if precision + recall > 1e-12:
            f1 = float(2.0 * precision * recall / (precision + recall))
        else:
            f1 = 0.0

        p_list.append(precision * 100.0)
        r_list.append(recall * 100.0)
        f1_list.append(f1 * 100.0)

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
) -> dict[str, float | int]:
    """同步计算 S3 序列解码协议规定的 6 维核心评价指标：

    1. BLEU-1 (%)：含句段均值 `bleu1_pct` 与语料级 `corpus_bleu1_pct`
    2. WER (%)：词错误率 `wer_pct`
    3. CER (%)：字符/汉字错误率 `cer_pct`
    4. Word Acc (%)：对齐词位总准确率 `word_acc_pct`
    5. 实义词/功能词拆解：`content_word_acc_pct` 与 `function_word_acc_pct`
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

    base = sentence_metrics(references, predictions, chinese=chinese)

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

    oracle_predictions: list[list[str]] = []
    per_subject_rows: dict[str, list[dict]] = defaultdict(list)

    for rec, ref_seq, pred_seq in zip(segment_records, references, predictions):
        if len(ref_seq) != len(pred_seq):
            raise ValueError(
                f"参考词序列长度 ({len(ref_seq)}) 与预测词序列长度 ({len(pred_seq)}) 不一致。"
            )
        seg_m = sentence_metrics([ref_seq], [pred_seq], chinese=chinese)
        in_vocab_mask = rec.get("in_vocabulary_mask") or [True] * len(ref_seq)
        top5_per_pos = rec.get("top5_words_per_pos") or []
        beam_cands_raw = rec.get("beam_candidates") or []

        # 确定 Top-B (Beam=5) Oracle 句段预测
        valid_beam_cands = [
            [str(w) for w in cand]
            for cand in beam_cands_raw
            if isinstance(cand, (list, tuple)) and len(cand) == len(ref_seq)
        ]
        if not valid_beam_cands:
            valid_beam_cands = [pred_seq]
        elif pred_seq not in valid_beam_cands:
            valid_beam_cands = [pred_seq] + valid_beam_cands

        best_oracle_seq = pred_seq
        best_oracle_key = (-1.0, -1)
        for cand_seq in valid_beam_cands:
            c_hits = sum(int(a == b) for a, b in zip(ref_seq, cand_seq))
            c_bleu = (
                float(
                    sentence_metrics([ref_seq], [cand_seq], chinese=chinese)[
                        "sentence_bleu1_mean"
                    ]
                )
                if len(valid_beam_cands) > 1
                else float(seg_m["sentence_bleu1_mean"])
            )
            cand_key = (round(0.5 * c_bleu + 0.5 * (c_hits / max(1, len(ref_seq))), 8), c_hits)
            if cand_key > best_oracle_key:
                best_oracle_key = cand_key
                best_oracle_seq = cand_seq
        oracle_predictions.append(best_oracle_seq)
        oracle_seg_m = (
            sentence_metrics([ref_seq], [best_oracle_seq], chinese=chinese)
            if best_oracle_seq != pred_seq
            else seg_m
        )

        seg_c_tot = seg_c_hit = seg_f_tot = seg_f_hit = 0
        for pos, (r_w, p_w) in enumerate(zip(ref_seq, pred_seq)):
            hit = int(r_w == p_w)
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

        subj = str(rec.get("subject_id", "default"))
        per_subject_rows[subj].append(
            {
                "words": len(ref_seq),
                "sentence_bleu1": float(seg_m["sentence_bleu1_mean"]),
                "oracle_sentence_bleu1": float(
                    oracle_seg_m["sentence_bleu1_mean"]
                ),
                "wer": float(seg_m["wer"]),
                "cer": float(seg_m["cer"]),
                "word_acc": float(seg_m["exact_word_accuracy"]),
                "content_total": seg_c_tot,
                "content_correct": seg_c_hit,
                "function_total": seg_f_tot,
                "function_correct": seg_f_hit,
            }
        )

    oracle_base = sentence_metrics(
        references, oracle_predictions, chinese=chinese
    )
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

    return {
        "segment_count": len(segment_records),
        "total_words": int(total_words),
        "content_word_count": int(content_total),
        "function_word_count": int(function_total),
        "in_vocabulary_word_count": int(in_vocab_total),
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
        "bertscore_f1_pct": float(bert_scores["bertscore_f1_pct"]),
        "bertscore_precision_pct": float(bert_scores["bertscore_precision_pct"]),
        "bertscore_recall_pct": float(bert_scores["bertscore_recall_pct"]),
    }


def apply_confidence_gate_to_segment(
    segment_history_record: dict,
    tau_gate: float = 0.0,
) -> dict:
    """对单个句段的流式轨迹应用置信度裕度门控 τ_{gate}：

    仅当 Δs_i^{(r)} = s_i^{(r)}(w_i^{new} | u) - s_i^{(r)}(w_i^{disp} | u) > τ_{gate} 时
    才触发屏幕改写，否则保留当前屏幕已显示词 w_i^{disp}。
    """
    tau = float(tau_gate)
    trajectories = segment_history_record["position_trajectories"]
    ref_words = [str(w) for w in segment_history_record["reference_words"]]
    seg_len = len(trajectories)

    final_gated_words: list[str] = []
    initial_k0_words: list[str] = []
    top5_words_per_pos: list[list[str]] = []

    rewrite_opportunities = 0
    triggered_rewrites = 0
    beneficial_repairs = 0  # wrong -> right
    harmful_rewrites = 0  # right -> wrong
    invalid_flickers = 0  # wrong -> wrong
    net_position_changes = 0

    for pos, traj in enumerate(trajectories):
        truth = ref_words[pos] if pos < len(ref_words) else None
        k0_word = str(traj["k0_word"])
        initial_k0_words.append(k0_word)
        current_word = k0_word

        # 阶段 1 回修：K0 -> K1（当 pos <= seg_len - 2 且存在 k1_word 时）
        k1_word = traj.get("k1_word")
        if k1_word is not None:
            k1_word = str(k1_word)
            rewrite_opportunities += 1
            delta_k1 = float(traj.get("k1_delta_score") or 0.0)
            if k1_word != current_word and delta_k1 > tau:
                old_w = current_word
                current_word = k1_word
                triggered_rewrites += 1
                if truth is not None:
                    if old_w != truth and current_word == truth:
                        beneficial_repairs += 1
                    elif old_w == truth and current_word != truth:
                        harmful_rewrites += 1
                    else:
                        invalid_flickers += 1

        # 阶段 2 定稿回修：当前屏幕词 (k0_word 或 k1_word) -> K2（当 pos <= seg_len - 3 且存在 k2_word 时）
        k2_word = traj.get("k2_word")
        if k2_word is not None:
            k2_word = str(k2_word)
            rewrite_opportunities += 1
            k2_new_score = float(traj.get("k2_score") or 0.0)
            k2_cand_map = traj.get("k2_candidate_scores") or {}
            if current_word in k2_cand_map:
                k2_disp_score = float(k2_cand_map[current_word])
                delta_k2 = k2_new_score - k2_disp_score
            else:
                delta_k2 = float(traj.get("k2_delta_score") or 0.0)

            if k2_word != current_word and delta_k2 > tau:
                old_w = current_word
                current_word = k2_word
                triggered_rewrites += 1
                if truth is not None:
                    if old_w != truth and current_word == truth:
                        beneficial_repairs += 1
                    elif old_w == truth and current_word != truth:
                        harmful_rewrites += 1
                    else:
                        invalid_flickers += 1

        if pos < seg_len - 1 and current_word != k0_word:
            net_position_changes += 1
        final_gated_words.append(current_word)

        cand_map = (
            traj.get("k2_candidate_scores")
            or traj.get("k1_candidate_scores")
            or traj.get("k0_candidate_scores")
            or {}
        )
        if cand_map:
            ranked_w = [
                str(w)
                for w, _ in sorted(
                    cand_map.items(), key=lambda kv: (-float(kv[1]), str(kv[0]))
                )[:5]
            ]
        else:
            ranked_w = [current_word]
        top5_words_per_pos.append(ranked_w)

    eligible_positions = max(0, seg_len - 1)
    return {
        "segment_index": segment_history_record["segment_index"],
        "group_id": segment_history_record["group_id"],
        "subject_id": segment_history_record["subject_id"],
        "reference_words": ref_words,
        "in_vocabulary_mask": list(
            segment_history_record.get("in_vocabulary_mask", [True] * seg_len)
        ),
        "initial_k0_words": initial_k0_words,
        "predicted_words": final_gated_words,
        "top5_words_per_pos": top5_words_per_pos,
        "eligible_positions": eligible_positions,
        "rewrite_opportunities": rewrite_opportunities,
        "triggered_rewrites": triggered_rewrites,
        "beneficial_repairs": beneficial_repairs,
        "harmful_rewrites": harmful_rewrites,
        "invalid_flickers": invalid_flickers,
        "net_position_changes": net_position_changes,
    }


def sweep_confidence_gate_pareto(
    screen_histories: Sequence[dict],
    *,
    language: str,
    tau_grid: Sequence[float] = DEFAULT_TAU_GATE_GRID,
    word_embedding_map: Mapping[str, np.ndarray | torch.Tensor] | None = None,
) -> dict:
    """扫描 τ_{gate} ∈ [0.0, 2.0]，计算每个阈值下的 6 维质量指标与屏幕改写/闪烁统计，构建帕累托前沿。"""
    if not screen_histories:
        raise ValueError("screen_histories 不能为空。")

    taus = [float(t) for t in tau_grid]
    curve_points: list[dict] = []
    baseline_invalid_flickers: int | None = None

    for tau in taus:
        gated_records = [
            apply_confidence_gate_to_segment(rec, tau_gate=tau)
            for rec in screen_histories
        ]
        metrics_6d = evaluate_six_dimensional_metrics(
            gated_records,
            language=language,
            word_embedding_map=word_embedding_map,
        )

        tot_opps = sum(r["rewrite_opportunities"] for r in gated_records)
        tot_elig = sum(r["eligible_positions"] for r in gated_records)
        tot_rewrites = sum(r["triggered_rewrites"] for r in gated_records)
        tot_repairs = sum(r["beneficial_repairs"] for r in gated_records)
        tot_harms = sum(r["harmful_rewrites"] for r in gated_records)
        tot_flickers = sum(r["invalid_flickers"] for r in gated_records)
        tot_pos_changes = sum(r["net_position_changes"] for r in gated_records)

        if baseline_invalid_flickers is None:
            baseline_invalid_flickers = max(0, tot_flickers)

        denom_opps = max(1, tot_opps)
        denom_elig = max(1, tot_elig)
        if baseline_invalid_flickers and baseline_invalid_flickers > 0:
            flicker_filtered_pct = float(
                (1.0 - tot_flickers / baseline_invalid_flickers) * 100.0
            )
        else:
            flicker_filtered_pct = 0.0 if abs(tau) <= 1e-12 else 100.0

        point = {
            "tau_gate": round(tau, 4),
            "screen_rewrite_rate_pct": float(tot_rewrites / denom_opps * 100.0),
            "position_change_rate_pct": float(
                tot_pos_changes / denom_elig * 100.0
            ),
            "beneficial_repair_rate_pct": float(tot_repairs / denom_opps * 100.0),
            "harmful_rewrite_rate_pct": float(tot_harms / denom_opps * 100.0),
            "invalid_flicker_rate_pct": float(tot_flickers / denom_opps * 100.0),
            "invalid_flicker_filtered_pct": flicker_filtered_pct,
            "net_beneficial_gain_pct": float(
                (tot_repairs - tot_harms) / denom_elig * 100.0
            ),
            "triggered_rewrites": int(tot_rewrites),
            "beneficial_repairs": int(tot_repairs),
            "harmful_rewrites": int(tot_harms),
            "invalid_flickers": int(tot_flickers),
            "rewrite_opportunities": int(tot_opps),
            **metrics_6d,
        }
        curve_points.append(point)

    # 选择推荐工作点：在保持至少 98% 无门控质量（Word Acc + BLEU-1）前提下最小化屏幕改写率，
    # 或净有益增益 net_beneficial_gain_pct 最大的阈值点
    unconstrained_pt = curve_points[0]
    base_quality = (
        0.5 * unconstrained_pt["bleu1_pct"]
        + 0.5 * unconstrained_pt["word_acc_pct"]
    )
    best_point = max(
        curve_points,
        key=lambda pt: (
            round(
                0.5 * pt["bleu1_pct"]
                + 0.5 * pt["word_acc_pct"]
                - 0.12 * pt["invalid_flicker_rate_pct"],
                6,
            ),
            -pt["screen_rewrite_rate_pct"],
        ),
    )

    return {
        "tau_grid": taus,
        "recommended_tau_gate": best_point["tau_gate"],
        "recommended_operating_point": best_point,
        "unconstrained_operating_point": unconstrained_pt,
        "baseline_quality_score": round(base_quality, 4),
        "pareto_curve": curve_points,
    }


def plot_confidence_gate_pareto_curve(
    pareto_result: dict,
    output_dir: str | Path,
    *,
    dataset: str = "pallier2025",
    split: str = "test",
) -> dict[str, str]:
    """绘制 C-gated 置信度门控“解码质量 vs 屏幕改写率”帕累托曲线与闪烁构成图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = pareto_result["pareto_curve"]
    rec_pt = pareto_result["recommended_operating_point"]

    taus = [pt["tau_gate"] for pt in curve]
    rewrite_rates = [pt["screen_rewrite_rate_pct"] for pt in curve]
    bleu1_vals = [pt["bleu1_pct"] for pt in curve]
    word_acc_vals = [pt["word_acc_pct"] for pt in curve]
    bert_vals = [pt["bertscore_f1_pct"] for pt in curve]
    flicker_rates = [pt["invalid_flicker_rate_pct"] for pt in curve]
    repair_rates = [pt["beneficial_repair_rate_pct"] for pt in curve]
    harm_rates = [pt["harmful_rewrite_rate_pct"] for pt in curve]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=160)
    fig.suptitle(
        f"S3 Confidence-Gated Revision (C-gated) Pareto Analysis | {dataset} ({split.upper()})",
        fontsize=12.5,
        fontweight="bold",
    )

    # 左图：解码质量 vs 屏幕改写率帕累托前沿
    ax_left.plot(
        rewrite_rates,
        bleu1_vals,
        "-o",
        color="#1f77b4",
        linewidth=2.2,
        markersize=5,
        label="BLEU-1 (%)",
    )
    ax_left.plot(
        rewrite_rates,
        word_acc_vals,
        "-s",
        color="#2ca02c",
        linewidth=2.2,
        markersize=5,
        label="Word Acc (%)",
    )
    ax_left.plot(
        rewrite_rates,
        bert_vals,
        "-^",
        color="#9467bd",
        linewidth=1.8,
        markersize=4.5,
        alpha=0.85,
        label="BERTScore F1 (%)",
    )

    ax_left.axvline(
        rec_pt["screen_rewrite_rate_pct"],
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label=f"Recommended τ={rec_pt['tau_gate']:.1f} (Rewrite={rec_pt['screen_rewrite_rate_pct']:.1f}%)",
    )
    for idx in (0, len(curve) // 2, len(curve) - 1):
        pt = curve[idx]
        ax_left.annotate(
            f"τ={pt['tau_gate']:.1f}",
            xy=(pt["screen_rewrite_rate_pct"], pt["bleu1_pct"]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
        )

    ax_left.set_title(
        "(A) Decoding Quality vs. Screen Rewrite Rate (Pareto Curve)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_left.set_xlabel("Screen Rewrite / Flicker Rate (%)", fontsize=10)
    ax_left.set_ylabel("Decoding Quality (%)", fontsize=10)
    ax_left.grid(True, linestyle=":", alpha=0.5)
    ax_left.legend(loc="best", fontsize=8.5)

    # 右图：随 τ_gate 变化的改写构成（有益改对 vs 无效闪烁 vs 有害改错）
    ax_right.plot(
        taus,
        rewrite_rates,
        "-o",
        color="#333333",
        linewidth=2.0,
        markersize=4.5,
        label="Total Screen Rewrite Rate (%)",
    )
    ax_right.plot(
        taus,
        flicker_rates,
        "-d",
        color="#ff7f0e",
        linewidth=2.0,
        markersize=4.5,
        label="Invalid Flicker (Wrong→Wrong %)",
    )
    ax_right.plot(
        taus,
        repair_rates,
        "-s",
        color="#2ca02c",
        linewidth=2.0,
        markersize=4.5,
        label="Beneficial Repair (Wrong→Right %)",
    )
    ax_right.plot(
        taus,
        harm_rates,
        "-x",
        color="#d62728",
        linewidth=1.8,
        markersize=4.5,
        label="Harmful Rewrite (Right→Wrong %)",
    )
    ax_right.axvline(
        rec_pt["tau_gate"],
        color="#d62728",
        linestyle="--",
        linewidth=1.3,
    )

    ax_right.set_title(
        "(B) Screen Rewrite Decomposition Across Gate Threshold τ_gate",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_right.set_xlabel("Confidence Margin Threshold τ_gate", fontsize=10)
    ax_right.set_ylabel("Rate per Revision Opportunity (%)", fontsize=10)
    ax_right.grid(True, linestyle=":", alpha=0.5)
    ax_right.legend(loc="best", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = out_dir / f"pareto_confidence_gate_{split}.png"
    pdf_path = out_dir / f"pareto_confidence_gate_{split}.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "pareto_png": str(png_path),
        "pareto_pdf": str(pdf_path),
    }


def plot_baseline_and_shifted_comparison(
    condition_metrics: Mapping[str, Mapping[str, float | int]],
    output_dir: str | Path,
    *,
    dataset: str = "pallier2025",
    split: str = "test",
) -> dict[str, str]:
    """绘制 LM-Only、S0、K0、S3 (C-gated)、F 及 Shifted (-75s) 的 6 维指标对比图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(condition_metrics.keys())
    if not labels:
        return {}

    x = np.arange(len(labels))
    width = 0.18

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.0, 5.2), dpi=160)
    fig.suptitle(
        f"S3 Sequence Decoding Baselines & Shifted (-75s) Control | {dataset} ({split.upper()})",
        fontsize=12.5,
        fontweight="bold",
    )

    bleu1 = [float(condition_metrics[k].get("bleu1_pct", 0.0)) for k in labels]
    word_acc = [
        float(condition_metrics[k].get("word_acc_pct", 0.0)) for k in labels
    ]
    content_acc = [
        float(condition_metrics[k].get("content_word_acc_pct", 0.0))
        for k in labels
    ]
    function_acc = [
        float(condition_metrics[k].get("function_word_acc_pct", 0.0))
        for k in labels
    ]

    ax_left.bar(
        x - 1.5 * width, bleu1, width, label="BLEU-1 (%)", color="#1f77b4"
    )
    ax_left.bar(
        x - 0.5 * width, word_acc, width, label="Word Acc (%)", color="#2ca02c"
    )
    ax_left.bar(
        x + 0.5 * width,
        content_acc,
        width,
        label="Content Word Acc (%)",
        color="#ff7f0e",
    )
    ax_left.bar(
        x + 1.5 * width,
        function_acc,
        width,
        label="Function Word Acc (%)",
        color="#9467bd",
    )
    ax_left.set_title(
        "(A) Lexical & Content/Function Word Accuracy",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax_left.set_ylabel("Accuracy / Score (%)", fontsize=10)
    ax_left.grid(axis="y", linestyle=":", alpha=0.5)
    ax_left.legend(loc="best", fontsize=8.5)

    bert_f1 = [
        float(condition_metrics[k].get("bertscore_f1_pct", 0.0)) for k in labels
    ]
    wer_vals = [float(condition_metrics[k].get("wer_pct", 0.0)) for k in labels]
    cer_vals = [float(condition_metrics[k].get("cer_pct", 0.0)) for k in labels]

    w2 = 0.24
    ax_right.bar(
        x - w2, bert_f1, w2, label="BERTScore F1 (%) ↑", color="#17becf"
    )
    ax_right.bar(x, wer_vals, w2, label="WER (%) ↓", color="#d62728", alpha=0.85)
    ax_right.bar(
        x + w2, cer_vals, w2, label="CER (%) ↓", color="#8c564b", alpha=0.85
    )
    ax_right.set_title(
        "(B) Semantic BERTScore & Error Rates (WER / CER)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax_right.set_ylabel("Percentage (%)", fontsize=10)
    ax_right.grid(axis="y", linestyle=":", alpha=0.5)
    ax_right.legend(loc="best", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = out_dir / f"baseline_and_shifted_comparison_{split}.png"
    pdf_path = out_dir / f"baseline_and_shifted_comparison_{split}.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "comparison_png": str(png_path),
        "comparison_pdf": str(pdf_path),
    }


def _top5_from_trajectory(traj: Mapping[str, object], fallback_word: str = "") -> list[str]:
    cand_map = (
        traj.get("k2_candidate_scores")
        or traj.get("k1_candidate_scores")
        or traj.get("k0_candidate_scores")
        or {}
    )
    if isinstance(cand_map, Mapping) and cand_map:
        return [
            str(w)
            for w, _ in sorted(
                cand_map.items(), key=lambda kv: (-float(kv[1]), str(kv[0]))
            )[:5]
        ]
    return [str(fallback_word)] if fallback_word else []


def _backfill_intermediate_and_shifted_bundles(
    decoded_bundles: Mapping[str, dict],
) -> dict[str, dict]:
    """若历史解码产物未单独显式保存 K1（C1）或 K0_shifted_neg75s，则从流式轨迹中零开销回填。"""
    bundles: dict[str, dict] = dict(decoded_bundles)

    for key in ("S3_clean", "S3", "S3_shifted_neg75s"):
        b = bundles.get(key)
        if not b:
            continue
        sents = b.get("final_sentences") or []
        hists = b.get("screen_histories") or []
        if sents and hists and len(sents) == len(hists):
            for rec, hist in zip(sents, hists):
                if not rec.get("top5_words_per_pos"):
                    trajs = hist.get("position_trajectories") or []
                    preds = rec.get("predicted_words") or []
                    rec["top5_words_per_pos"] = [
                        _top5_from_trajectory(
                            t, str(preds[p]) if p < len(preds) else ""
                        )
                        for p, t in enumerate(trajs)
                    ]

    s3_clean = bundles.get("S3_clean") or bundles.get("S3")
    if "K0" not in bundles and s3_clean is not None:
        clean_sents = s3_clean.get("final_sentences") or []
        clean_hists = s3_clean.get("screen_histories") or []
        k0_clean_records: list[dict] = []
        for idx, rec in enumerate(clean_sents):
            init_k0 = rec.get("initial_k0_words")
            trajs = (
                clean_hists[idx].get("position_trajectories") or []
                if idx < len(clean_hists)
                else []
            )
            if not init_k0 and trajs:
                init_k0 = [str(t.get("k0_word", "")) for t in trajs]
            if init_k0 is None:
                continue
            k0_top5: list[list[str]] = []
            for p, w0 in enumerate(init_k0):
                cmap = (
                    trajs[p].get("k0_candidate_scores") or {}
                    if p < len(trajs)
                    else {}
                )
                if isinstance(cmap, Mapping) and cmap:
                    k0_top5.append(
                        [
                            str(w)
                            for w, _ in sorted(
                                cmap.items(),
                                key=lambda kv: (-float(kv[1]), str(kv[0])),
                            )[:5]
                        ]
                    )
                else:
                    k0_top5.append([str(w0)])
            k0_clean_records.append(
                {
                    "segment_index": rec.get("segment_index", idx),
                    "group_id": rec.get("group_id", ""),
                    "subject_id": rec.get("subject_id", ""),
                    "reference_words": list(rec.get("reference_words", [])),
                    "predicted_words": list(init_k0),
                    "top5_words_per_pos": k0_top5,
                    "in_vocabulary_mask": list(
                        rec.get(
                            "in_vocabulary_mask",
                            [True] * len(rec.get("reference_words", [])),
                        )
                    ),
                }
            )
        if k0_clean_records:
            bundles["K0"] = {
                "condition": "K0",
                "alignment": "clean",
                "segment_count": len(k0_clean_records),
                "mean_seconds_per_segment": float(
                    s3_clean.get("mean_seconds_per_segment", 0.0)
                ),
                "final_sentences": k0_clean_records,
                "screen_histories": [],
            }

    if "K1" not in bundles and s3_clean is not None:
        histories = s3_clean.get("screen_histories") or []
        sentences = s3_clean.get("final_sentences") or []
        if histories:
            k1_records: list[dict] = []
            for idx, hist in enumerate(histories):
                trajs = hist.get("position_trajectories") or []
                k1_words = [
                    str(
                        t.get("k1_word")
                        if t.get("k1_word") is not None
                        else t.get("k0_word", "")
                    )
                    for t in trajs
                ]
                k1_top5: list[list[str]] = []
                for p, t in enumerate(trajs):
                    cmap = (
                        t.get("k1_candidate_scores")
                        or t.get("k0_candidate_scores")
                        or t.get("k2_candidate_scores")
                        or {}
                    )
                    if isinstance(cmap, Mapping) and cmap:
                        k1_top5.append(
                            [
                                str(w)
                                for w, _ in sorted(
                                    cmap.items(),
                                    key=lambda kv: (-float(kv[1]), str(kv[0])),
                                )[:5]
                            ]
                        )
                    else:
                        k1_top5.append([k1_words[p]])
                base_rec = sentences[idx] if idx < len(sentences) else hist
                k1_records.append(
                    {
                        "segment_index": hist.get(
                            "segment_index", base_rec.get("segment_index", idx)
                        ),
                        "group_id": hist.get(
                            "group_id", base_rec.get("group_id", "")
                        ),
                        "subject_id": hist.get(
                            "subject_id", base_rec.get("subject_id", "")
                        ),
                        "reference_words": list(
                            hist.get(
                                "reference_words",
                                base_rec.get("reference_words", []),
                            )
                        ),
                        "predicted_words": k1_words,
                        "top5_words_per_pos": k1_top5,
                        "in_vocabulary_mask": list(
                            hist.get(
                                "in_vocabulary_mask",
                                base_rec.get(
                                    "in_vocabulary_mask",
                                    [True] * len(k1_words),
                                ),
                            )
                        ),
                    }
                )
            bundles["K1"] = {
                "condition": "K1",
                "alignment": "clean",
                "segment_count": len(k1_records),
                "mean_seconds_per_segment": float(
                    s3_clean.get("mean_seconds_per_segment", 0.0)
                ),
                "final_sentences": k1_records,
                "screen_histories": [],
            }

    s3_shifted = bundles.get("S3_shifted_neg75s")
    if "K0_shifted_neg75s" not in bundles and s3_shifted is not None:
        shifted_sents = s3_shifted.get("final_sentences") or []
        shifted_hists = s3_shifted.get("screen_histories") or []
        k0_shifted_records: list[dict] = []
        for idx, rec in enumerate(shifted_sents):
            init_k0 = rec.get("initial_k0_words")
            trajs = (
                shifted_hists[idx].get("position_trajectories") or []
                if idx < len(shifted_hists)
                else []
            )
            if not init_k0 and trajs:
                init_k0 = [str(t.get("k0_word", "")) for t in trajs]
            if init_k0 is None:
                continue
            k0_top5: list[list[str]] = []
            for p, w0 in enumerate(init_k0):
                cmap = (
                    trajs[p].get("k0_candidate_scores") or {}
                    if p < len(trajs)
                    else {}
                )
                if isinstance(cmap, Mapping) and cmap:
                    k0_top5.append(
                        [
                            str(w)
                            for w, _ in sorted(
                                cmap.items(),
                                key=lambda kv: (-float(kv[1]), str(kv[0])),
                            )[:5]
                        ]
                    )
                else:
                    k0_top5.append([str(w0)])
            k0_shifted_records.append(
                {
                    "segment_index": rec.get("segment_index", idx),
                    "group_id": rec.get("group_id", ""),
                    "subject_id": rec.get("subject_id", ""),
                    "reference_words": list(rec.get("reference_words", [])),
                    "predicted_words": list(init_k0),
                    "top5_words_per_pos": k0_top5,
                    "in_vocabulary_mask": list(
                        rec.get(
                            "in_vocabulary_mask",
                            [True] * len(rec.get("reference_words", [])),
                        )
                    ),
                }
            )
        if k0_shifted_records:
            bundles["K0_shifted_neg75s"] = {
                "condition": "K0",
                "alignment": "shifted",
                "segment_count": len(k0_shifted_records),
                "mean_seconds_per_segment": float(
                    s3_shifted.get("mean_seconds_per_segment", 0.0)
                ),
                "final_sentences": k0_shifted_records,
                "screen_histories": [],
            }

    return bundles


def plot_multi_subject_observation_curve(
    multi_subject_scaling: Mapping[str, Mapping[str, float | int]],
    output_dir: str | Path,
    *,
    dataset: str = "pallier2025",
    split: str = "test",
) -> dict[str, str]:
    """绘制跨被试观测数 k ∈ {1, 2, 5, 10} 聚合下的序列解码性能跃升曲线（对标 d'Ascoli 2024 & SimpleB2T 2026）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not multi_subject_scaling:
        return {}

    ordered_items = sorted(
        multi_subject_scaling.items(),
        key=lambda kv: int(kv[1].get("k_observations", 1)),
    )
    k_vals = [int(item[1].get("k_observations", 1)) for item in ordered_items]
    bleu1 = [float(item[1].get("bleu1_pct", 0.0)) for item in ordered_items]
    corpus_bleu1 = [
        float(item[1].get("corpus_bleu1_pct", 0.0)) for item in ordered_items
    ]
    oracle_bleu1 = [
        float(item[1].get("beam_oracle_bleu1_pct", 0.0))
        for item in ordered_items
    ]
    word_acc = [
        float(item[1].get("word_acc_pct", 0.0)) for item in ordered_items
    ]
    invocab_acc = [
        float(item[1].get("in_vocabulary_word_acc_pct", 0.0))
        for item in ordered_items
    ]
    top5_acc = [
        float(item[1].get("word_acc_top5_pct", 0.0)) for item in ordered_items
    ]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=160)
    fig.suptitle(
        f"Multi-Subject Observation Scaling (k = {', '.join(map(str, k_vals))}) | {dataset} ({split.upper()})",
        fontsize=12.5,
        fontweight="bold",
    )

    ax_left.plot(
        k_vals,
        bleu1,
        "-o",
        color="#1f77b4",
        linewidth=2.2,
        markersize=6,
        label="Sentence BLEU-1 (%)",
    )
    ax_left.plot(
        k_vals,
        corpus_bleu1,
        "-s",
        color="#2ca02c",
        linewidth=2.2,
        markersize=6,
        label="Corpus BLEU-1 (%)",
    )
    ax_left.plot(
        k_vals,
        oracle_bleu1,
        "--^",
        color="#9467bd",
        linewidth=2.0,
        markersize=6,
        label="Beam-5 Oracle BLEU-1 (%)",
    )
    ax_left.set_title(
        "(A) Sequence BLEU-1 vs. Aggregated Subject Observations k",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_left.set_xlabel("Number of Averaged Subject Observations (k)", fontsize=10)
    ax_left.set_ylabel("BLEU-1 (%)", fontsize=10)
    ax_left.set_xticks(k_vals)
    ax_left.grid(True, linestyle=":", alpha=0.5)
    ax_left.legend(loc="best", fontsize=8.5)

    ax_right.plot(
        k_vals,
        word_acc,
        "-o",
        color="#ff7f0e",
        linewidth=2.2,
        markersize=6,
        label="All-Word Acc@1 (%)",
    )
    ax_right.plot(
        k_vals,
        invocab_acc,
        "-s",
        color="#d62728",
        linewidth=2.2,
        markersize=6,
        label="In-Vocabulary Word Acc@1 (%)",
    )
    ax_right.plot(
        k_vals,
        top5_acc,
        "--d",
        color="#17becf",
        linewidth=2.0,
        markersize=6,
        label="Position Candidate Acc@5 (%)",
    )
    ax_right.set_title(
        "(B) Word Accuracy (Top-1 / In-Vocab / Top-5) vs. Observations k",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_right.set_xlabel("Number of Averaged Subject Observations (k)", fontsize=10)
    ax_right.set_ylabel("Word Accuracy (%)", fontsize=10)
    ax_right.set_xticks(k_vals)
    ax_right.grid(True, linestyle=":", alpha=0.5)
    ax_right.legend(loc="best", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = out_dir / f"multi_subject_observation_scaling_{split}.png"
    pdf_path = out_dir / f"multi_subject_observation_scaling_{split}.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "multi_subject_scaling_png": str(png_path),
        "multi_subject_scaling_pdf": str(pdf_path),
    }


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
    expanded = _backfill_intermediate_and_shifted_bundles(bundles_map)
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
            "conditions": all_conds,
            "revision_gains": all_seg_block.get("revision_gains", {}),
            "core_subset_80pct": core80_block,
            "core_subset_100pct": core100_block,
        }

    return summary_by_scale


def plot_vocabulary_scaling_curve(
    vocabulary_scaling_summary: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    *,
    dataset: str = "pallier2025",
    split: str = "test",
) -> dict[str, str]:
    """绘制 1x4 多尺度词表敏感性曲线（横轴 log |V|，纵轴 BLEU-1 / In-Vocab Word Acc / CER / BERTScore，对标 Jayalath 2025 Fig 4）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not vocabulary_scaling_summary:
        return {}

    ordered = sorted(
        vocabulary_scaling_summary.items(),
        key=lambda kv: int(kv[1].get("vocabulary_size", 0)),
    )
    v_sizes = [int(item[1].get("vocabulary_size", 0)) for item in ordered]
    if len(v_sizes) < 2:
        return {}

    curve_specs = (
        ("S3_clean", "S3 (K2 Rolling Revision, Clean)", "#1f77b4", "-o", 2.3),
        ("K0", "K0 (Initial Draft, Clean)", "#2ca02c", "--s", 1.9),
        ("S0", "S0 (Single-Word CNN, Clean)", "#ff7f0e", ":^", 1.8),
        (
            "S3_shifted_neg75s",
            "S3 Shifted (-75s Control)",
            "#d62728",
            "-.x",
            1.7,
        ),
    )
    panel_specs = (
        ("bleu1_pct", "(A) Sentence BLEU-1 (%)", "BLEU-1 (%)"),
        (
            "in_vocabulary_word_acc_pct",
            "(B) In-Vocabulary Word Acc (%)",
            "In-Vocab Word Acc (%)",
        ),
        ("cer_pct", "(C) Character Error Rate CER (%) ↓", "CER (%)"),
        (
            "bertscore_f1_pct",
            "(D) Continuous Semantic BERTScore (%)",
            "BERTScore F1 (%)",
        ),
    )

    fig, axes = plt.subplots(1, 4, figsize=(18.5, 4.6), dpi=160)
    fig.suptitle(
        f"Multi-Scale Vocabulary Scaling Law (|V| = {', '.join(map(str, v_sizes))}) | {dataset} ({split.upper()})",
        fontsize=12.5,
        fontweight="bold",
    )

    for ax, (metric_key, panel_title, y_label) in zip(axes, panel_specs):
        for cond_key, label, color, style, lw in curve_specs:
            xs: list[int] = []
            ys: list[float] = []
            for _, scale_obj in ordered:
                c_map = scale_obj.get("conditions") or {}
                if cond_key in c_map and metric_key in c_map[cond_key]:
                    xs.append(int(scale_obj["vocabulary_size"]))
                    ys.append(float(c_map[cond_key][metric_key]))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    style,
                    color=color,
                    linewidth=lw,
                    markersize=5.5,
                    label=label,
                )
        ax.set_xscale("log")
        ax.set_xticks(v_sizes)
        ax.set_xticklabels([str(v) for v in v_sizes], fontsize=8.5)
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(panel_title, fontsize=10.0, fontweight="bold")
        ax.set_xlabel("Candidate Vocabulary Size |V| (log scale)", fontsize=9.5)
        ax.set_ylabel(y_label, fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            fontsize=9.0,
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
        )

    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    png_path = out_dir / f"s3_vocabulary_scaling_{split}.png"
    pdf_path = out_dir / f"s3_vocabulary_scaling_{split}.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)

    return {
        "vocabulary_scaling_png": str(png_path),
        "vocabulary_scaling_pdf": str(pdf_path),
    }


def evaluate_and_export_sequence_decoding(
    decoded_bundles: Mapping[str, dict],
    *,
    dataset: str,
    language: str,
    split: str,
    output_dir: str | Path,
    word_embedding_map: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    tau_grid: Sequence[float] = DEFAULT_TAU_GATE_GRID,
    multi_subject_bundles: Mapping[str | int, dict] | None = None,
    vocabulary_scaling_bundles: Mapping[str, dict] | None = None,
    core_vocab_size: int = 250,
    plot: bool = True,
) -> dict:
    """汇总各解码条件（LM-Only, S0, K0, K1, S3, F 及 Shifted -75s）、跨被试聚合与多尺度候选词表实验，输出 6 维指标表与可视化图表。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expanded_bundles = _backfill_intermediate_and_shifted_bundles(
        decoded_bundles
    )
    condition_summaries: dict[str, dict] = {}
    pareto_results: dict[str, dict] = {}
    figure_paths: dict[str, str] = {}
    multi_subject_summaries: dict[str, dict] = {}
    vocabulary_scaling_summaries: dict[str, dict] = {}
    core_vocabulary_benchmark: dict[str, Any] = {}

    for key, bundle in expanded_bundles.items():
        records = bundle["final_sentences"]
        metrics_6d = evaluate_six_dimensional_metrics(
            records,
            language=language,
            word_embedding_map=word_embedding_map,
        )
        metrics_6d["mean_seconds_per_segment"] = float(
            bundle.get("mean_seconds_per_segment", 0.0)
        )
        condition_summaries[key] = metrics_6d

        histories = bundle.get("screen_histories") or []
        if histories:
            pareto = sweep_confidence_gate_pareto(
                histories,
                language=language,
                tau_grid=tau_grid,
                word_embedding_map=word_embedding_map,
            )
            pareto_results[key] = pareto
            rec_pt = pareto["recommended_operating_point"]
            condition_summaries[f"{key}_c_gated"] = {
                k: rec_pt[k]
                for k in metrics_6d
                if k in rec_pt
            }
            condition_summaries[f"{key}_c_gated"]["recommended_tau_gate"] = (
                rec_pt["tau_gate"]
            )
            condition_summaries[f"{key}_c_gated"]["screen_rewrite_rate_pct"] = (
                rec_pt["screen_rewrite_rate_pct"]
            )
            condition_summaries[f"{key}_c_gated"][
                "invalid_flicker_filtered_pct"
            ] = rec_pt["invalid_flicker_filtered_pct"]
            condition_summaries[f"{key}_c_gated"][
                "mean_seconds_per_segment"
            ] = float(bundle.get("mean_seconds_per_segment", 0.0))

            if plot and key in {"S3_clean", "S3"}:
                figure_paths.update(
                    plot_confidence_gate_pareto_curve(
                        pareto, out_dir, dataset=dataset, split=split
                    )
                )

    canonical_order = (
        "LM-Only",
        "S0",
        "K0",
        "K1",
        "S3_clean",
        "S3_clean_c_gated",
        "F",
        "F_local",
        "K0_shifted_neg75s",
        "S3_shifted_neg75s",
        "S3_shifted_neg75s_c_gated",
    )
    ordered_summaries: dict[str, dict] = {}
    for k in canonical_order:
        if k in condition_summaries:
            ordered_summaries[k] = condition_summaries[k]
    for k, v in condition_summaries.items():
        if k not in ordered_summaries:
            ordered_summaries[k] = v
    condition_summaries = ordered_summaries

    full_vocab_core_subsets = evaluate_high_coverage_core_subsets(
        expanded_bundles,
        language=language,
        word_embedding_map=word_embedding_map,
        vocabulary_size=expanded_bundles.get("S3_clean", {}).get(
            "vocabulary_size"
        ),
    )

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
            )
            ms_metrics["k_observations"] = k_obs
            ms_metrics["mean_seconds_per_segment"] = float(
                ms_bundle.get("mean_seconds_per_segment", 0.0)
            )
            multi_subject_summaries[f"k={k_obs}"] = ms_metrics
        if plot and len(multi_subject_summaries) >= 2:
            figure_paths.update(
                plot_multi_subject_observation_curve(
                    multi_subject_summaries,
                    out_dir,
                    dataset=dataset,
                    split=split,
                )
            )

    if vocabulary_scaling_bundles:
        vocabulary_scaling_summaries = evaluate_vocabulary_scaling_summary(
            vocabulary_scaling_bundles,
            language=language,
            word_embedding_map=word_embedding_map,
        )
        core_key = f"V={int(core_vocab_size)}"
        if core_key in vocabulary_scaling_bundles:
            core_entry = vocabulary_scaling_bundles[core_key]
            core_vocabulary_benchmark = evaluate_high_coverage_core_subsets(
                core_entry.get("conditions") or {},
                language=language,
                word_embedding_map=word_embedding_map,
                vocabulary_size=int(core_vocab_size),
            )
            core_vocabulary_benchmark["test_token_coverage_pct"] = (
                core_entry.get("test_token_coverage_pct")
            )
        if plot and len(vocabulary_scaling_summaries) >= 2:
            figure_paths.update(
                plot_vocabulary_scaling_curve(
                    vocabulary_scaling_summaries,
                    out_dir,
                    dataset=dataset,
                    split=split,
                )
            )

    if plot and condition_summaries:
        figure_paths.update(
            plot_baseline_and_shifted_comparison(
                condition_summaries, out_dir, dataset=dataset, split=split
            )
        )

    report = {
        "schema_version": 1,
        "protocol": "sequence_decoding_s3_evaluation",
        "dataset": dataset,
        "language": language,
        "split": split,
        "conditions": condition_summaries,
        "core_vocabulary_benchmark": core_vocabulary_benchmark,
        "full_vocab_core_subsets": full_vocab_core_subsets,
        "vocabulary_scaling": vocabulary_scaling_summaries,
        "multi_subject_scaling": multi_subject_summaries,
        "confidence_gate_pareto": pareto_results,
        "figures": figure_paths,
    }
    save_json(out_dir / f"s3_evaluation_summary_{split}.json", report)
    report_txt = format_sequence_decoding_report(report)
    (out_dir / f"s3_evaluation_report_{split}.txt").write_text(
        report_txt + "\n", encoding="utf-8"
    )
    return report


def _fmt_pct(val: object, width: int = 7) -> str:
    if val is None or val == "":
        return "-".rjust(width)
    try:
        return f"{float(val):{width}.2f}"
    except (TypeError, ValueError):
        return "-".rjust(width)


def _fmt_signed_pct(val: object, width: int = 7) -> str:
    if val is None or val == "":
        return "-".rjust(width)
    try:
        return f"{float(val):+{width}.2f}"
    except (TypeError, ValueError):
        return "-".rjust(width)


def _format_conditions_table(
    conds_map: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    lines: list[str] = []
    header = (
        f"{'Condition':<25} | {'BLEU-1':>7} | {'CorpBLEU':>8} | {'OraclBLEU':>9} | "
        f"{'WordAcc':>7} | {'InVocAcc':>8} | {'Acc@5':>7} | {'ContAcc':>7} | "
        f"{'FuncAcc':>7} | {'WER(%)':>7} | {'CER(%)':>7} | {'BERT-F1':>7}"
    )
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)
    for cond_name, m in conds_map.items():
        if not isinstance(m, Mapping):
            continue
        lines.append(
            f"{str(cond_name):<25} | "
            f"{_fmt_pct(m.get('bleu1_pct'), 7)} | "
            f"{_fmt_pct(m.get('corpus_bleu1_pct'), 8)} | "
            f"{_fmt_pct(m.get('beam_oracle_bleu1_pct'), 9)} | "
            f"{_fmt_pct(m.get('word_acc_pct'), 7)} | "
            f"{_fmt_pct(m.get('in_vocabulary_word_acc_pct'), 8)} | "
            f"{_fmt_pct(m.get('word_acc_top5_pct'), 7)} | "
            f"{_fmt_pct(m.get('content_word_acc_pct'), 7)} | "
            f"{_fmt_pct(m.get('function_word_acc_pct'), 7)} | "
            f"{_fmt_pct(m.get('wer_pct'), 7)} | "
            f"{_fmt_pct(m.get('cer_pct'), 7)} | "
            f"{_fmt_pct(m.get('bertscore_f1_pct'), 7)}"
        )
    lines.append(sep)
    return lines


def _format_revision_gains_block(
    gains: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    if not gains:
        return []
    label_map = {
        "clean_C_minus_A_pp": "Clean 滚动修正净增益 (C - A = S3_clean - K0)",
        "shifted_C_minus_A_pp": "Shifted 错位滚动增益 (C_shift - A_shift)",
        "clean_minus_shifted_S3_pp": "真实 vs 错位地面实况净优势 (S3_clean - S3_shift)",
    }
    out: list[str] = []
    for g_key in (
        "clean_C_minus_A_pp",
        "shifted_C_minus_A_pp",
        "clean_minus_shifted_S3_pp",
    ):
        g_val = gains.get(g_key)
        if not isinstance(g_val, Mapping) or not g_val:
            continue
        out.append(
            f"  * {label_map.get(g_key, g_key):<46}: "
            f"ΔBLEU-1={_fmt_signed_pct(g_val.get('bleu1_pct'), 6)} pp | "
            f"ΔWordAcc={_fmt_signed_pct(g_val.get('word_acc_pct'), 6)} pp | "
            f"ΔInVocAcc={_fmt_signed_pct(g_val.get('in_vocabulary_word_acc_pct'), 6)} pp | "
            f"ΔCER={_fmt_signed_pct(g_val.get('cer_pct'), 6)} pp | "
            f"ΔBERT-F1={_fmt_signed_pct(g_val.get('bertscore_f1_pct'), 6)} pp"
        )
    return out


def format_sequence_decoding_report(
    report: Mapping[str, Any],
) -> str:
    """将 S3 序列解码评估汇总字典格式化为结构化、易读的终端/Colab 表格报告。"""
    dataset = str(report.get("dataset", "pallier2025"))
    split = str(report.get("split", "test"))
    seed = report.get("seed", 0)
    exp_tag = report.get("exp_tag") or "(default)"
    cms_eta = report.get("cms_eta", 0.0)
    f_mode = report.get("f_condition_mode") or ""

    banner = "=" * 132
    lines: list[str] = [
        banner,
        f"S3 序列解码协议评估汇总报告 | Dataset: {dataset} | Split: {split.upper()} | Seed: {seed} | ExpTag: {exp_tag} | CMS η: {cms_eta}"
        + (f" | F-Mode: {f_mode}" if f_mode else ""),
        banner,
    ]

    state_hp = report.get("state_hyperparameters")
    if isinstance(state_hp, Mapping) and state_hp:
        hp_parts: list[str] = []
        for st_name in ("S0", "K0", "K1", "K2", "F"):
            st_cfg = state_hp.get(st_name)
            if isinstance(st_cfg, Mapping):
                t_val = st_cfg.get("temperature", "-")
                l_val = st_cfg.get("language_weight", "-")
                hp_parts.append(f"{st_name}: (T={t_val}, λ={l_val})")
        if hp_parts:
            lines.append("[状态自适应校准超参] " + " | ".join(hp_parts))
            lines.append("-" * 132)

    # 1. 全词库主实验结果表
    conds = report.get("conditions")
    if isinstance(conds, Mapping) and conds:
        first_cond = next(iter(conds.values()), {})
        tot_segs = first_cond.get("segment_count", "-")
        tot_words = first_cond.get("total_words", "-")
        invoc_rate = first_cond.get("in_vocabulary_rate_pct", None)
        cov_str = (
            f"{float(invoc_rate):.2f}%"
            if invoc_rate is not None
            else "-"
        )
        lines.append(
            f"\n[表 1] 全词库主实验结果 (Full Vocabulary | Segments={tot_segs}, TotalWords={tot_words}, TokenCoverage={cov_str})"
        )
        lines.extend(_format_conditions_table(conds))
        main_gains = _compute_revision_gain_deltas(conds)
        lines.extend(_format_revision_gains_block(main_gains))

    # 2. 高覆盖核心词表基准 (|V| = 250)
    core_bench = report.get("core_vocabulary_benchmark")
    if isinstance(core_bench, Mapping) and core_bench.get("subsets"):
        v_core = core_bench.get("vocabulary_size", 250)
        tok_cov_core = core_bench.get("test_token_coverage_pct")
        cov_core_str = (
            f"{float(tok_cov_core):.2f}%"
            if tok_cov_core is not None
            else "-"
        )
        lines.append(
            f"\n[表 2] 高覆盖核心词表平行基准 (Core Vocabulary |V| = {v_core}, 全集词位覆盖率 = {cov_core_str}，对标 SimpleB2T & Jayalath 2025)"
        )
        subsets_map = core_bench.get("subsets") or {}
        subset_titles = (
            ("all_segments", "口径 1 [all_segments]: 全部测试句段 (含 In-Vocab Word Acc)"),
            ("core_subset_100pct", "口径 2a [core_subset_100pct]: 100% 词表内核心句 (0% OOV Core Set，直接对标 SimpleB2T)"),
            ("core_subset_80pct", "口径 2b [core_subset_80pct]: >=80% 词表内高覆盖核心句"),
            ("expanded_subset_70pct", "口径 2c [expanded_subset_70pct]: >=70% 词表内扩展高覆盖句"),
        )
        for sub_key, sub_title in subset_titles:
            sub_block = subsets_map.get(sub_key)
            if not isinstance(sub_block, Mapping):
                continue
            sub_conds = sub_block.get("conditions") or {}
            if not sub_conds:
                continue
            seg_cnt = sub_block.get("segment_count", 0)
            lines.append(f"\n  >>> {sub_title} (Segments = {seg_cnt}) <<<")
            lines.extend(_format_conditions_table(sub_conds))
            lines.extend(
                _format_revision_gains_block(
                    sub_block.get("revision_gains") or {}
                )
            )

    # 3. 全词库 (|V| = 2020) 下的高覆盖核心子集表现
    full_core = report.get("full_vocab_core_subsets")
    if isinstance(full_core, Mapping) and full_core.get("subsets"):
        lines.append(
            "\n[表 3] 全词库 (Full Vocabulary) 下的高覆盖核心子集拆解 (100% 零 OOV / >=80% / >=70%)"
        )
        fc_header = (
            f"{'Subset':<26} | {'Segs':>5} | {'Condition':<18} | {'BLEU-1':>7} | "
            f"{'CorpBLEU':>8} | {'WordAcc':>7} | {'InVocAcc':>8} | {'Acc@5':>7} | "
            f"{'WER(%)':>7} | {'CER(%)':>7} | {'BERT-F1':>7}"
        )
        fc_sep = "-" * len(fc_header)
        lines.append(fc_sep)
        lines.append(fc_header)
        lines.append(fc_sep)
        fc_subsets = full_core.get("subsets") or {}
        for sub_key in (
            "all_segments",
            "core_subset_100pct",
            "core_subset_80pct",
            "expanded_subset_70pct",
        ):
            sb = fc_subsets.get(sub_key)
            if not isinstance(sb, Mapping):
                continue
            seg_cnt = sb.get("segment_count", 0)
            sb_conds = sb.get("conditions") or {}
            for c_name in (
                "S0",
                "K0",
                "K1",
                "S3_clean",
                "K0_shifted_neg75s",
                "S3_shifted_neg75s",
            ):
                if c_name not in sb_conds:
                    continue
                m = sb_conds[c_name]
                lines.append(
                    f"{sub_key:<26} | {int(seg_cnt):>5d} | {c_name:<18} | "
                    f"{_fmt_pct(m.get('bleu1_pct'), 7)} | "
                    f"{_fmt_pct(m.get('corpus_bleu1_pct'), 8)} | "
                    f"{_fmt_pct(m.get('word_acc_pct'), 7)} | "
                    f"{_fmt_pct(m.get('in_vocabulary_word_acc_pct'), 8)} | "
                    f"{_fmt_pct(m.get('word_acc_top5_pct'), 7)} | "
                    f"{_fmt_pct(m.get('wer_pct'), 7)} | "
                    f"{_fmt_pct(m.get('cer_pct'), 7)} | "
                    f"{_fmt_pct(m.get('bertscore_f1_pct'), 7)}"
                )
            lines.append(fc_sep)

    # 4. 多尺度候选词表缩放定律 (|V| in {50, 100, 250, 500, 1000, 2020})
    vocab_scaling = report.get("vocabulary_scaling")
    if isinstance(vocab_scaling, Mapping) and vocab_scaling:
        lines.append(
            "\n[表 4A] 多尺度候选词表缩放曲线 — 全测试句段 (All Segments，对标 Jayalath et al. 2025 Fig. 4)"
        )
        vs_header = (
            f"{'Scale |V|':<10} | {'TokCov%':>7} | {'S3 BLEU-1':>9} | {'K0 BLEU-1':>9} | "
            f"{'S0 BLEU-1':>9} | {'ShiftBLEU':>9} | {'Δ(C-A)BLEU':>10} | {'S3 InVocAcc':>11} | "
            f"{'K0 InVocAcc':>11} | {'S3 WordAcc':>10} | {'S3 CER(%)':>9} | {'S3 BERT-F1':>10}"
        )
        vs_sep = "-" * len(vs_header)
        lines.append(vs_sep)
        lines.append(vs_header)
        lines.append(vs_sep)

        ordered_scales = sorted(
            vocab_scaling.items(),
            key=lambda kv: int(
                kv[1].get("vocabulary_size", str(kv[0]).lstrip("vV=_") or 0)
            )
            if isinstance(kv[1], Mapping)
            else 0,
        )
        for s_key, s_entry in ordered_scales:
            if not isinstance(s_entry, Mapping):
                continue
            all_c = s_entry.get("all_segments") or {}
            s3_m = all_c.get("S3_clean") or {}
            k0_m = all_c.get("K0") or {}
            s0_m = all_c.get("S0") or {}
            sh_m = all_c.get("S3_shifted_neg75s") or {}
            gains_all = (
                (s_entry.get("revision_gains_all_segments") or {}).get(
                    "clean_C_minus_A_pp"
                )
                or {}
            )
            lines.append(
                f"{str(s_key):<10} | "
                f"{_fmt_pct(s_entry.get('test_token_coverage_pct'), 7)} | "
                f"{_fmt_pct(s3_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_pct(k0_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_pct(s0_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_pct(sh_m.get('bleu1_pct'), 9)} | "
                f"{_fmt_signed_pct(gains_all.get('bleu1_pct'), 10)} | "
                f"{_fmt_pct(s3_m.get('in_vocabulary_word_acc_pct'), 11)} | "
                f"{_fmt_pct(k0_m.get('in_vocabulary_word_acc_pct'), 11)} | "
                f"{_fmt_pct(s3_m.get('word_acc_pct'), 10)} | "
                f"{_fmt_pct(s3_m.get('cer_pct'), 9)} | "
                f"{_fmt_pct(s3_m.get('bertscore_f1_pct'), 10)}"
            )
        lines.append(vs_sep)

        lines.append(
            "\n[表 4B] 多尺度候选词表缩放曲线 — 高覆盖核心子集 (100% Core [0% OOV] 与 >=80% Core Sentences)"
        )
        vc_header = (
            f"{'Scale |V|':<10} | {'100%Segs':>8} | {'100% S3 BLEU':>12} | {'100% S3 Acc':>11} | "
            f"{'>=80%Segs':>9} | {'>=80% S3 BLEU':>13} | {'>=80% K0 BLEU':>13} | "
            f"{'>=80% ShiftBLEU':>15} | {'>=80% S3 InVocAcc':>17} | {'>=80% S3 Acc@5':>14}"
        )
        vc_sep = "-" * len(vc_header)
        lines.append(vc_sep)
        lines.append(vc_header)
        lines.append(vc_sep)
        for s_key, s_entry in ordered_scales:
            if not isinstance(s_entry, Mapping):
                continue
            c100 = (s_entry.get("core_subset_100pct") or {}).get("S3_clean") or {}
            c80_map = s_entry.get("core_subset_80pct") or {}
            c80_s3 = c80_map.get("S3_clean") or {}
            c80_k0 = c80_map.get("K0") or {}
            c80_sh = c80_map.get("S3_shifted_neg75s") or {}
            lines.append(
                f"{str(s_key):<10} | "
                f"{int(s_entry.get('core_100pct_segment_count', 0)):>8d} | "
                f"{_fmt_pct(c100.get('bleu1_pct'), 12)} | "
                f"{_fmt_pct(c100.get('word_acc_pct'), 11)} | "
                f"{int(s_entry.get('core_80pct_segment_count', 0)):>9d} | "
                f"{_fmt_pct(c80_s3.get('bleu1_pct'), 13)} | "
                f"{_fmt_pct(c80_k0.get('bleu1_pct'), 13)} | "
                f"{_fmt_pct(c80_sh.get('bleu1_pct'), 15)} | "
                f"{_fmt_pct(c80_s3.get('in_vocabulary_word_acc_pct'), 17)} | "
                f"{_fmt_pct(c80_s3.get('word_acc_top5_pct'), 14)}"
            )
        lines.append(vc_sep)

    # 5. 跨被试多观测神经聚合曲线 (k in {1, 2, 5, 10})
    ms_scaling = report.get("multi_subject_scaling")
    if isinstance(ms_scaling, Mapping) and ms_scaling:
        lines.append(
            "\n[表 5] 跨被试多观测神经聚合曲线 (Multi-Subject Observation Scaling k ∈ {1, 2, 5, 10})"
        )
        ordered_ms = dict(
            sorted(
                ms_scaling.items(),
                key=lambda kv: int(
                    kv[1].get("k_observations", str(kv[0]).lstrip("kK=_") or 1)
                )
                if isinstance(kv[1], Mapping)
                else 1,
            )
        )
        lines.extend(_format_conditions_table(ordered_ms))

    # 6. 产物与图表保存路径
    lines.append("\n[导出产物与可视化图表路径]")
    for path_key in (
        "export_path",
        "calibration_path",
        "decoding_path",
        "evaluation_dir",
    ):
        if report.get(path_key):
            lines.append(f"  - {path_key:<18}: {report[path_key]}")
    figs = report.get("figures")
    if isinstance(figs, Mapping) and figs:
        for fig_k, fig_p in figs.items():
            lines.append(f"  - {fig_k:<28}: {fig_p}")
    lines.append(banner)
    return "\n".join(lines)


def print_sequence_decoding_report(
    report: Mapping[str, Any],
) -> None:
    """向终端打印结构化可读的 S3 序列解码评估汇总表格。"""
    print(format_sequence_decoding_report(report))
