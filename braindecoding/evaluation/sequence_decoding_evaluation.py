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
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F_nn

from braindecoding.evaluation.evaluation_exports import (
    export_sequence_decoding_tables_and_markdown,
)
from braindecoding.evaluation.statistical_significance import (
    edit_distance,
    paired_block_statistics,
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


from braindecoding.evaluation.evaluation_figures import (
    plot_baseline_and_shifted_comparison,
    plot_confidence_gate_pareto_curve,
    plot_multi_subject_observation_curve,
    plot_vocabulary_scaling_curve,
)



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


def _compute_s3_paired_significance(
    expanded_bundles: Mapping[str, dict],
    *,
    language: str,
) -> dict[str, dict[str, Any]]:
    """计算 S3_clean vs K0 与 S3_clean vs S3_shifted_neg75s 的配对材料块 95% CI 与置换检验 p 值。"""
    s3_bundle = expanded_bundles.get("S3_clean") or expanded_bundles.get("S3")
    if not s3_bundle or not s3_bundle.get("final_sentences"):
        return {}
    chinese = language == "zh"
    s3_sents = s3_bundle.get("final_sentences") or []

    def _segment_scores(records: Sequence[Mapping]) -> tuple[list[float], list[float], list[str], list[float]]:
        bleu_list: list[float] = []
        acc_list: list[float] = []
        blocks: list[str] = []
        weights: list[float] = []
        for idx, rec in enumerate(records):
            ref = [str(w) for w in rec.get("reference_words") or ()]
            pred = [str(w) for w in rec.get("predicted_words") or ()]
            if not ref or len(ref) != len(pred):
                continue
            sm = sentence_metrics([ref], [pred], chinese=chinese)
            bleu_list.append(float(sm["sentence_bleu1_mean"]) * 100.0)
            acc_list.append(float(sm["exact_word_accuracy"]) * 100.0)
            gid = str(rec.get("group_id") or "")
            sid = str(rec.get("subject_id") or "0")
            blk = gid.split(":")[0] if ":" in gid and gid.split(":")[0] else (gid or f"seg_{idx // 5}")
            blocks.append(blk)
            weights.append(float(max(1, len(ref))))
        return bleu_list, acc_list, blocks, weights

    s3_bleu, s3_acc, s3_blocks, s3_weights = _segment_scores(s3_sents)
    if len(s3_bleu) < 2:
        return {}

    sig_results: dict[str, dict[str, Any]] = {}
    comparisons = (
        ("K0", "S3_clean vs K0 (Revision Gain)", "s3_clean_vs_k0", 20261031),
        (
            "S3_shifted_neg75s",
            "S3_clean vs Shifted -75s (Temporal Alignment)",
            "s3_clean_vs_shifted",
            20261032,
        ),
    )
    for other_key, label_prefix, key_prefix, base_seed in comparisons:
        other_bundle = expanded_bundles.get(other_key)
        if not other_bundle or not other_bundle.get("final_sentences"):
            continue
        o_bleu, o_acc, _, _ = _segment_scores(
            other_bundle.get("final_sentences") or []
        )
        if len(o_bleu) != len(s3_bleu):
            continue
        effective_blocks = (
            s3_blocks
            if len(set(s3_blocks)) >= 2
            else [f"seg_{i}" for i in range(len(s3_bleu))]
        )
        for metric_label, m_suffix, a_vals, b_vals, w_vals in (
            ("BLEU-1 (%)", "bleu1", s3_bleu, o_bleu, [1.0] * len(s3_bleu)),
            ("Word Acc (%)", "word_acc", s3_acc, o_acc, s3_weights),
        ):
            arr_a = np.asarray(a_vals, dtype=np.float64)
            arr_b = np.asarray(b_vals, dtype=np.float64)
            arr_w = np.asarray(w_vals, dtype=np.float64)
            diffs = arr_a - arr_b
            b_to_idx: dict[str, list[int]] = {}
            for idx_item, b_key in enumerate(effective_blocks):
                b_to_idx.setdefault(str(b_key), []).append(idx_item)
            block_means = np.asarray(
                [
                    float(
                        np.sum(diffs[idxs] * arr_w[idxs])
                        / max(float(np.sum(arr_w[idxs])), 1e-12)
                    )
                    for idxs in b_to_idx.values()
                ],
                dtype=np.float64,
            )
            block_weights = np.asarray(
                [float(np.sum(arr_w[idxs])) for idxs in b_to_idx.values()],
                dtype=np.float64,
            )
            n_blocks = int(len(block_means))
            total_w = max(float(np.sum(block_weights)), 1e-12)
            observed = float(np.sum(block_means * block_weights) / total_w)
            if n_blocks <= 1 or float(np.max(np.abs(diffs))) <= 1e-12:
                p_one = 1.0 if abs(observed) <= 1e-12 else 0.05
                p_two = p_one
                ci_low = observed
                ci_high = observed
            else:
                norm_weights = block_weights / total_w
                weighted_contribs = block_means * norm_weights
                rng = np.random.default_rng(
                    base_seed + (0 if m_suffix == "bleu1" else 10)
                )
                if n_blocks <= 12:
                    ints = np.arange(2**n_blocks, dtype=np.uint32)
                    signs = (
                        2.0
                        * (
                            (
                                ints[:, None]
                                >> np.arange(n_blocks, dtype=np.uint32)
                            )
                            & 1
                        ).astype(np.float64)
                        - 1.0
                    )
                    null_dist = signs @ weighted_contribs
                    p_one = float(np.mean(null_dist >= observed - 1e-12))
                    p_two = float(
                        np.mean(np.abs(null_dist) >= abs(observed) - 1e-12)
                    )
                else:
                    signs = rng.choice((-1.0, 1.0), size=(2000, n_blocks))
                    null_dist = signs @ weighted_contribs
                    p_one = float(
                        (1 + int(np.count_nonzero(null_dist >= observed - 1e-12)))
                        / 2001.0
                    )
                    p_two = float(
                        (
                            1
                            + int(
                                np.count_nonzero(
                                    np.abs(null_dist) >= abs(observed) - 1e-12
                                )
                            )
                        )
                        / 2001.0
                    )
                boot_picks = rng.integers(0, n_blocks, size=(2000, n_blocks))
                picked_means = block_means[boot_picks]
                picked_weights = block_weights[boot_picks]
                boot_estimates = np.sum(
                    picked_means * picked_weights, axis=1
                ) / np.maximum(np.sum(picked_weights, axis=1), 1e-12)
                ci_low, ci_high = (
                    float(x)
                    for x in np.quantile(boot_estimates, (0.025, 0.975))
                )
            sig_results[f"{key_prefix}_{m_suffix}"] = {
                "label": label_prefix,
                "metric": metric_label,
                "effect_pp": round(float(observed), 4),
                "ci_lower_pp": round(float(ci_low), 4),
                "ci_upper_pp": round(float(ci_high), 4),
                "p_one_sided": round(float(p_one), 6),
                "p_two_sided": round(float(p_two), 6),
                "significance": _s3_significance_stars(p_two),
                "blocks": n_blocks,
            }
    return sig_results


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

    per_subject, per_subject_summary = _compute_s3_per_subject_breakdown(
        expanded_bundles,
        language=language,
        word_embedding_map=word_embedding_map,
    )
    statistical_significance = _compute_s3_paired_significance(
        expanded_bundles,
        language=language,
    )
    revision_gains = _compute_revision_gain_deltas(condition_summaries)

    report = {
        "schema_version": 1,
        "protocol": "sequence_decoding_s3_evaluation",
        "dataset": dataset,
        "language": language,
        "split": split,
        "conditions": condition_summaries,
        "revision_gains": revision_gains,
        "statistical_significance": statistical_significance,
        "per_subject": per_subject,
        "per_subject_summary": per_subject_summary,
        "core_vocabulary_benchmark": core_vocabulary_benchmark,
        "full_vocab_core_subsets": full_vocab_core_subsets,
        "vocabulary_scaling": vocabulary_scaling_summaries,
        "multi_subject_scaling": multi_subject_summaries,
        "confidence_gate_pareto": pareto_results,
        "figures": figure_paths,
    }
    s3_exports = export_sequence_decoding_tables_and_markdown(
        report,
        out_dir,
        decoded_bundles=expanded_bundles,
    )
    report["exports"] = s3_exports
    save_json(out_dir / f"s3_evaluation_summary_{split}.json", report)
    report_txt = format_sequence_decoding_report(report, verbose=True)
    (out_dir / f"s3_evaluation_report_{split}.txt").write_text(
        report_txt + "\n", encoding="utf-8"
    )
    return report


from braindecoding.evaluation.evaluation_reporting import (
    _format_conditions_table,
    _format_revision_gains_block,
    format_sequence_decoding_executive_summary,
    format_sequence_decoding_report,
    print_sequence_decoding_report,
)

