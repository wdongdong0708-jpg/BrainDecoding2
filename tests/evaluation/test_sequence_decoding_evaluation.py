"""S3 序列解码协议多维语义评估与 C-gated 门控帕累托分析（Module 5）单元测试。

覆盖：
1. 6 维核心评价指标 `evaluate_six_dimensional_metrics`：
   - 字面精确匹配：BLEU-1 (%), WER (%), CER (%)
   - 词位准确率：Word Acc (%)，拆解为 实义词（Content Words）Acc (%) 与 功能词（Function Words）Acc (%)
   - 连续语义相似度：BERTScore (%) (Precision / Recall / F1)
2. 置信度门控回修 `apply_confidence_gate_to_segment` 与 `sweep_confidence_gate_pareto`：
   - 验证得分裕度公式 Δs_i^{(r)} = s_i^{(r)}(w_i^{new} | u) - s_i^{(r)}(w_i^{old} | u) > τ_{gate}
   - 验证扫描 τ_{gate} ∈ [0.0, 2.0] 时低裕度无效闪烁被滤除、屏幕改写率单调非增
3. 完整评估导出与帕累托图表生成 `evaluate_and_export_sequence_decoding`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from braindecoding.evaluation.sequence_decoding_evaluation import (
    DEFAULT_TAU_GATE_GRID,
    apply_confidence_gate_to_segment,
    compute_bertscore_percentage,
    evaluate_and_export_sequence_decoding,
    evaluate_six_dimensional_metrics,
    is_content_word,
    is_function_word,
    sweep_confidence_gate_pareto,
)


def test_content_and_function_word_classification() -> None:
    """验证法语与中文实义词/功能词分类器。"""
    # 法语功能词与实义词
    assert is_function_word("le", "fr") is True
    assert is_function_word("dans", "fr") is True
    assert is_function_word("-t-il", "fr") is True
    assert is_content_word("prince", "fr") is True
    assert is_content_word("planète", "fr") is True

    # 中文功能词与实义词
    assert is_function_word("的", "zh") is True
    assert is_function_word("在", "zh") is True
    assert is_function_word("我们", "zh") is True
    assert is_content_word("王子", "zh") is True
    assert is_content_word("星球", "zh") is True


def test_six_dimensional_metrics_computation() -> None:
    """验证 6 维评价指标（BLEU-1, WER, CER, Word Acc + 实义/功能词拆解, BERTScore）精确性。"""
    word_embeddings = {
        "le": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "la": np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
        "petit": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "grand": np.array([0.0, 0.6, 0.8, 0.0], dtype=np.float32),
        "prince": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "parle": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }

    records = [
        {
            "segment_index": 0,
            "group_id": "g0",
            "subject_id": "sub-01",
            # "le" (功能词), "petit" (实义词), "prince" (实义词), "parle" (实义词)
            "reference_words": ["le", "petit", "prince", "parle"],
            # 命中 "le", "prince", "parle"；错 1 个实义词 ("petit" -> "grand")
            "predicted_words": ["le", "grand", "prince", "parle"],
            "in_vocabulary_mask": [True, True, True, True],
        }
    ]

    metrics = evaluate_six_dimensional_metrics(
        records, language="fr", word_embedding_map=word_embeddings
    )

    assert metrics["total_words"] == 4
    assert metrics["function_word_count"] == 1
    assert metrics["content_word_count"] == 3
    assert metrics["word_acc_pct"] == pytest.approx(75.0)
    assert metrics["function_word_acc_pct"] == pytest.approx(100.0)
    assert metrics["content_word_acc_pct"] == pytest.approx(200.0 / 3.0)
    assert metrics["wer_pct"] == pytest.approx(25.0)
    assert 0.0 < metrics["cer_pct"] < 50.0
    assert metrics["bleu1_pct"] == pytest.approx(75.0)
    assert metrics["bertscore_f1_pct"] > 75.0

    perfect_bert = compute_bertscore_percentage(
        [["le", "petit", "prince"]],
        [["le", "petit", "prince"]],
        word_embedding_map=word_embeddings,
    )
    assert perfect_bert["bertscore_f1_pct"] == pytest.approx(100.0)


def test_confidence_gate_pareto_sweep_and_export(tmp_path: Path) -> None:
    """验证 C-gated 门控扫描 τ_gate ∈ [0, 2.0] 能滤除低裕度无效闪烁并生成帕累托曲线与导出报表。"""
    stream_record = {
        "segment_index": 0,
        "group_id": "zh-seg-0",
        "subject_id": "sub-01",
        "recording_id": "sub-01",
        "scope": "chapter-02",
        "block_id": "chapter-02|block-000",
        "reference_words": ["王子", "说", "你好"],
        "in_vocabulary_mask": [True, True, True],
        "position_trajectories": [
            {
                "word_index": 0,
                "reference_word": "王子",
                "k0_word": "狐狸",
                "k0_word_id": 0,
                "k0_score": -2.0,
                "k1_word": "星球",
                "k1_word_id": 1,
                "k1_score": -1.85,
                "k1_old_score": -2.0,
                "k1_delta_score": 0.15,
                "k1_candidate_scores": {"狐狸": -2.0, "星球": -1.85, "王子": -2.1},
                "k2_word": "王子",
                "k2_word_id": 2,
                "k2_score": -0.40,
                "k2_old_score": -1.85,
                "k2_delta_score": 1.45,
                "k2_candidate_scores": {"狐狸": -1.90, "星球": -1.85, "王子": -0.40},
            },
            {
                "word_index": 1,
                "reference_word": "说",
                "k0_word": "说",
                "k0_word_id": 3,
                "k0_score": -1.0,
                "k1_word": "看",
                "k1_word_id": 4,
                "k1_score": -0.75,
                "k1_old_score": -1.0,
                "k1_delta_score": 0.25,
                "k1_candidate_scores": {"说": -1.0, "看": -0.75},
                "k2_word": "看",
                "k2_word_id": 4,
                "k2_score": -0.70,
                "k2_old_score": -0.75,
                "k2_delta_score": 0.05,
                "k2_candidate_scores": {"说": -0.80, "看": -0.70},
            },
            {
                "word_index": 2,
                "reference_word": "你好",
                "k0_word": "你好",
                "k0_word_id": 5,
                "k0_score": -0.5,
                "k1_word": None,
                "k2_word": None,
            },
        ],
    }

    # τ_gate = 0.0（无门控）：触发词位 0 的 K0->K1 (闪烁)、K1->K2 (修正) 以及词位 1 的 K0->K1 (有害改错)
    ungated = apply_confidence_gate_to_segment(stream_record, tau_gate=0.0)
    assert ungated["triggered_rewrites"] == 3
    assert ungated["invalid_flickers"] == 1
    assert ungated["harmful_rewrites"] == 1
    assert ungated["beneficial_repairs"] == 1
    assert ungated["predicted_words"] == ["王子", "看", "你好"]

    # τ_gate = 0.5（中度门控）：滤除 Δs=0.15 的闪烁与 Δs=0.25 的误改，仅保留词位 0 的 K2 强修正 (Δs=1.50 > 0.5)
    gated_mid = apply_confidence_gate_to_segment(stream_record, tau_gate=0.5)
    assert gated_mid["triggered_rewrites"] == 1
    assert gated_mid["invalid_flickers"] == 0
    assert gated_mid["harmful_rewrites"] == 0
    assert gated_mid["beneficial_repairs"] == 1
    assert gated_mid["predicted_words"] == ["王子", "说", "你好"]

    # 扫描完整 τ_gate ∈ [0.0, 2.0] 帕累托网格
    pareto = sweep_confidence_gate_pareto(
        [stream_record],
        language="zh",
        tau_grid=DEFAULT_TAU_GATE_GRID,
    )
    points = pareto["pareto_curve"]
    assert len(points) == len(DEFAULT_TAU_GATE_GRID)
    rewrite_rates = [pt["screen_rewrite_rate_pct"] for pt in points]
    assert all(
        rewrite_rates[i] >= rewrite_rates[i + 1] - 1e-9
        for i in range(len(rewrite_rates) - 1)
    )

    # 验证完整导出函数生成汇总 JSON 与两张 PDF/PNG 评估图
    decoded_bundles = {
        "S3_clean": {
            "condition": "S3",
            "alignment": "clean",
            "mean_seconds_per_segment": 0.02,
            "final_sentences": [
                {
                    "segment_index": 0,
                    "group_id": "zh-seg-0",
                    "subject_id": "sub-01",
                    "recording_id": "sub-01",
                    "scope": "chapter-02",
                    "block_id": "chapter-02|block-000",
                    "reference_words": ["王子", "说", "你好"],
                    "predicted_words": ["王子", "看", "你好"],
                    "in_vocabulary_mask": [True, True, True],
                    "elapsed_seconds": 0.02,
                }
            ],
            "screen_histories": [stream_record],
        },
        "S0_clean": {
            "condition": "S0",
            "alignment": "clean",
            "mean_seconds_per_segment": 0.01,
            "final_sentences": [
                {
                    "segment_index": 0,
                    "group_id": "zh-seg-0",
                    "subject_id": "sub-01",
                    "recording_id": "sub-01",
                    "scope": "chapter-02",
                    "block_id": "chapter-02|block-000",
                    "reference_words": ["王子", "说", "你好"],
                    "predicted_words": ["狐狸", "说", "你好"],
                    "in_vocabulary_mask": [True, True, True],
                    "elapsed_seconds": 0.01,
                }
            ],
            "screen_histories": [],
        },
        "LM-Only_clean": {
            "condition": "LM-Only",
            "alignment": "clean",
            "mean_seconds_per_segment": 0.01,
            "final_sentences": [
                {
                    "segment_index": 0,
                    "group_id": "zh-seg-0",
                    "subject_id": "sub-01",
                    "recording_id": "sub-01",
                    "scope": "chapter-02",
                    "block_id": "chapter-02|block-000",
                    "reference_words": ["王子", "说", "你好"],
                    "predicted_words": ["星球", "看", "你好"],
                    "in_vocabulary_mask": [True, True, True],
                    "elapsed_seconds": 0.01,
                }
            ],
            "screen_histories": [],
        },
        "S3_shifted_neg75s": {
            "condition": "S3",
            "alignment": "shifted",
            "mean_seconds_per_segment": 0.02,
            "final_sentences": [
                {
                    "segment_index": 0,
                    "group_id": "zh-seg-0",
                    "subject_id": "sub-01",
                    "recording_id": "sub-01",
                    "scope": "chapter-02",
                    "block_id": "chapter-02|block-000",
                    "reference_words": ["王子", "说", "你好"],
                    "predicted_words": ["星球", "看", "狐狸"],
                    "initial_k0_words": ["狐狸", "看", "狐狸"],
                    "in_vocabulary_mask": [True, True, True],
                    "elapsed_seconds": 0.02,
                }
            ],
            "screen_histories": [],
        },
    }

    report = evaluate_and_export_sequence_decoding(
        decoded_bundles,
        dataset="chineseeeg2_littleprince",
        language="zh",
        split="val",
        output_dir=tmp_path / "eval_out",
        tau_grid=DEFAULT_TAU_GATE_GRID,
        plot=True,
    )
    assert "K1" in report["conditions"]
    assert "K0_shifted_neg75s" in report["conditions"]
    assert "S3_clean" in report["conditions"]
    assert "S3_clean_c_gated" in report["conditions"]
    assert (tmp_path / "eval_out" / "s3_evaluation_summary_val.json").exists()
    assert (tmp_path / "eval_out" / "pareto_confidence_gate_val.pdf").exists()
    assert (tmp_path / "eval_out" / "baseline_and_shifted_comparison_val.pdf").exists()
