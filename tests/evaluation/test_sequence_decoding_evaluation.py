"""S3 序列解码协议多维语义评估与配对统计显著性单元测试。

覆盖：
1. 6 维核心评价指标 `evaluate_six_dimensional_metrics`：
   - 字面精确匹配：BLEU-1 (%), WER (%), CER (%)
   - 词位准确率：Word Acc (%)，拆解为 实义词（Content Words）Acc (%) 与 功能词（Function Words）Acc (%)
   - 连续语义相似度：BERTScore (%) (Precision / Recall / F1)
2. 多尺度词表评估、跨被试聚合与完整评估导出 `evaluate_and_export_sequence_decoding`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from braindecoding.evaluation.sequence_decoding_evaluation import (
    compute_bertscore_percentage,
    evaluate_and_export_sequence_decoding,
    evaluate_six_dimensional_metrics,
    is_content_word,
    is_function_word,
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
            "in_vocabulary_mask": [True, False, True, True],
            "top5_words_per_pos": [
                ["le"],
                ["grand", "petit"],
                ["prince"],
                ["parle"],
            ],
            "beam_candidates": [
                ["le", "grand", "prince", "parle"],
                ["le", "petit", "prince", "parle"],
            ],
        }
    ]

    metrics = evaluate_six_dimensional_metrics(
        records, language="fr", word_embedding_map=word_embeddings
    )

    assert metrics["total_words"] == 4
    assert metrics["function_word_count"] == 1
    assert metrics["content_word_count"] == 3
    assert metrics["word_acc_pct"] == pytest.approx(75.0)
    assert metrics["in_vocabulary_word_acc_pct"] == pytest.approx(100.0)
    assert metrics["word_acc_top5_pct"] == pytest.approx(100.0)
    assert metrics["beam_oracle_bleu1_pct"] == pytest.approx(100.0)
    assert metrics["beam_oracle_word_acc_pct"] == pytest.approx(100.0)
    assert metrics["corpus_bleu1_pct"] == pytest.approx(75.0)
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


def test_evaluate_and_export_sequence_decoding_with_multi_subject_scaling(
    tmp_path: Path,
) -> None:
    """验证完整单遍静态解码评估导出及跨被试聚合（k=1 vs k=5）汇总。"""
    decoded_bundles = {
        "S3_clean": {
            "condition": "S3",
            "alignment": "clean",
            "mean_seconds_per_segment": 0.02,
            "final_sentences": [
                {
                    "segment_index": 0,
                    "group_id": "sub-01_run-06:0",
                    "subject_id": "sub-01",
                    "recording_id": "sub-01_run-06",
                    "scope": "run-06",
                    "block_id": "run-06|block-000",
                    "reference_words": ["le", "petit", "prince"],
                    "predicted_words": ["le", "petit", "prince"],
                    "in_vocabulary_mask": [True, True, True],
                    "elapsed_seconds": 0.02,
                }
            ],
        },
        "S0_clean": {
            "condition": "S0",
            "alignment": "clean",
            "mean_seconds_per_segment": 0.01,
            "final_sentences": [
                {
                    "segment_index": 0,
                    "group_id": "sub-01_run-06:0",
                    "subject_id": "sub-01",
                    "recording_id": "sub-01_run-06",
                    "scope": "run-06",
                    "block_id": "run-06|block-000",
                    "reference_words": ["le", "petit", "prince"],
                    "predicted_words": ["le", "grand", "prince"],
                    "in_vocabulary_mask": [True, True, True],
                    "elapsed_seconds": 0.01,
                }
            ],
        },
        "LM-Only_clean": {
            "condition": "LM-Only",
            "alignment": "clean",
            "mean_seconds_per_segment": 0.01,
            "final_sentences": [
                {
                    "segment_index": 0,
                    "group_id": "sub-01_run-06:0",
                    "subject_id": "sub-01",
                    "recording_id": "sub-01_run-06",
                    "scope": "run-06",
                    "block_id": "run-06|block-000",
                    "reference_words": ["le", "petit", "prince"],
                    "predicted_words": ["la", "grand", "renard"],
                    "in_vocabulary_mask": [True, True, True],
                    "elapsed_seconds": 0.01,
                }
            ],
        },
        "S3_shifted_neg75s": {
            "condition": "S3",
            "alignment": "shifted",
            "mean_seconds_per_segment": 0.02,
            "final_sentences": [
                {
                    "segment_index": 0,
                    "group_id": "sub-01_run-06:0",
                    "subject_id": "sub-01",
                    "recording_id": "sub-01_run-06",
                    "scope": "run-06",
                    "block_id": "run-06|block-000",
                    "reference_words": ["le", "petit", "prince"],
                    "predicted_words": ["la", "grand", "renard"],
                    "in_vocabulary_mask": [True, True, True],
                    "elapsed_seconds": 0.02,
                }
            ],
        },
    }

    multi_subject_bundles = {
        "k1": {
            "k_observations": 1,
            "mean_seconds_per_segment": 0.02,
            "final_sentences": decoded_bundles["S0_clean"]["final_sentences"],
        },
        "k5": {
            "k_observations": 5,
            "mean_seconds_per_segment": 0.02,
            "final_sentences": decoded_bundles["S3_clean"]["final_sentences"],
        },
    }

    report = evaluate_and_export_sequence_decoding(
        decoded_bundles,
        dataset="pallier2025",
        language="fr",
        split="val",
        output_dir=tmp_path / "eval_out",
        multi_subject_bundles=multi_subject_bundles,
        plot=False,
    )
    assert "S0" in report["conditions"]
    assert "S3_clean" in report["conditions"]
    assert "S3_shifted_neg75s" in report["conditions"]
    assert "k=1" in report["multi_subject_scaling"]
    assert "k=5" in report["multi_subject_scaling"]
    assert (
        report["multi_subject_scaling"]["k=5"]["word_acc_pct"]
        >= report["multi_subject_scaling"]["k=1"]["word_acc_pct"]
    )
    assert (tmp_path / "eval_out" / "s3_evaluation_summary_val.json").exists()


def test_core_vocabulary_subsets_and_scaling_curve_export(tmp_path: Path) -> None:
    """验证高覆盖核心词表（口径1全集 + 口径2 100%/80%/70% 子集）与多尺度词表敏感性曲线导出。"""
    from braindecoding.evaluation.sequence_decoding_evaluation import (
        evaluate_high_coverage_core_subsets,
    )

    # 构造 3 个句段：
    # - seg 0: 4/4 (100%) 词在词表内
    # - seg 1: 4/5 (80%) 词在词表内
    # - seg 2: 2/4 (50%) 词在词表内
    seg_records = [
        {
            "segment_index": 0,
            "group_id": "g0",
            "subject_id": "sub-01",
            "reference_words": ["le", "petit", "prince", "parle"],
            "predicted_words": ["le", "petit", "prince", "parle"],
            "in_vocabulary_mask": [True, True, True, True],
        },
        {
            "segment_index": 1,
            "group_id": "g1",
            "subject_id": "sub-01",
            "reference_words": ["le", "petit", "prince", "dessine", "mouton"],
            "predicted_words": ["le", "petit", "prince", "dessine", "prince"],
            "in_vocabulary_mask": [True, True, True, True, False],
        },
        {
            "segment_index": 2,
            "group_id": "g2",
            "subject_id": "sub-01",
            "reference_words": ["le", "renard", "apprivoise", "rose"],
            "predicted_words": ["le", "prince", "parle", "petit"],
            "in_vocabulary_mask": [True, False, False, True],
        },
    ]

    subsets = evaluate_high_coverage_core_subsets(
        {"S3_clean": {"final_sentences": seg_records}},
        language="fr",
        vocabulary_size=250,
    )
    assert subsets["vocabulary_size"] == 250
    sub_map = subsets["subsets"]
    assert sub_map["all_segments"]["segment_count"] == 3
    assert sub_map["core_subset_100pct"]["segment_count"] == 1
    assert sub_map["core_subset_100pct"]["conditions"]["S3_clean"]["word_acc_pct"] == pytest.approx(100.0)
    assert sub_map["core_subset_80pct"]["segment_count"] == 2
    assert sub_map["expanded_subset_70pct"]["segment_count"] == 2
    assert (
        sub_map["core_subset_100pct"]["conditions"]["S3_clean"]["bleu1_pct"]
        > sub_map["all_segments"]["conditions"]["S3_clean"]["bleu1_pct"]
    )

    vocabulary_scaling_bundles = {
        "V=50": {
            "vocabulary_size": 50,
            "test_token_coverage_pct": 50.0,
            "conditions": {
                "S3_clean": {"final_sentences": seg_records[:2]},
                "S3_shifted_neg75s": {"final_sentences": [seg_records[2]]},
            },
        },
        "V=250": {
            "vocabulary_size": 250,
            "test_token_coverage_pct": 76.92,
            "conditions": {
                "S3_clean": {"final_sentences": seg_records},
                "S3_shifted_neg75s": {"final_sentences": [seg_records[2]]},
            },
        },
    }

    report = evaluate_and_export_sequence_decoding(
        {"S3_clean": {"final_sentences": seg_records}},
        dataset="pallier2025",
        language="fr",
        split="test",
        output_dir=tmp_path / "scaling_eval_out",
        vocabulary_scaling_bundles=vocabulary_scaling_bundles,
        core_vocab_size=250,
        plot=True,
    )
    assert report["core_vocabulary_benchmark"]["vocabulary_size"] == 250
    assert "V=50" in report["core_vocabulary_benchmarks"]
    assert "V=250" in report["core_vocabulary_benchmarks"]
    assert (
        report["core_vocabulary_benchmark"]["subsets"]["core_subset_100pct"][
            "segment_count"
        ]
        == 1
    )
    assert "V=50" in report["vocabulary_scaling"]
    assert "V=250" in report["vocabulary_scaling"]
    assert (tmp_path / "scaling_eval_out" / "s3_evaluation_summary_test.json").exists()


def test_multi_subject_group8_canonical_material_alignment_without_material_id() -> None:
    """验证缺失 material_id 字段的旧解码记录中，k=1 (sub-01_run-06:i) 与 k>1 (k2::sub-01_run-06:i) 能100%对齐材料块。"""
    from braindecoding.evaluation.sequence_decoding_evaluation import (
        _compute_s3_paired_significance,
        _extract_canonical_material_id,
    )

    assert _extract_canonical_material_id({"group_id": "sub-01_run-06:12"}) == "run-06:12"
    assert _extract_canonical_material_id({"group_id": "k2::sub-01_run-06:12"}) == "run-06:12"
    assert _extract_canonical_material_id({"group_id": "k10::sub-02_run-06:0"}) == "run-06:0"

    # 构造 10 个测试材料：
    # k=1 包含 sub-01 与 sub-02 两个被试（20 条记录），k=2 包含聚合后的 10 条记录，且均省略 material_id
    k1_records = []
    k2_records = []
    for mat_idx in range(10):
        ref_words = ["le", "petit", "prince", "parle"]
        for sub_id in ("sub-01", "sub-02"):
            k1_records.append(
                {
                    "segment_index": len(k1_records),
                    "group_id": f"{sub_id}_run-06:{mat_idx}",
                    "subject_id": sub_id,
                    "reference_words": ref_words,
                    "predicted_words": ["le", "prince", "parle", "petit"],
                    "in_vocabulary_mask": [True, True, True, True],
                }
            )
        k2_records.append(
            {
                "segment_index": mat_idx,
                "group_id": f"k2::sub-01_run-06:{mat_idx}",
                "subject_id": "multi_k2",
                "reference_words": ref_words,
                "predicted_words": ["le", "petit", "prince", "parle"],
                "in_vocabulary_mask": [True, True, True, True],
            }
        )

    s3_bundle = {"final_sentences": k1_records}
    ms_bundles = {
        "k=2": {
            "k_observations": 2,
            "conditions": {
                "S3_clean": {"final_sentences": k2_records},
            },
            "by_vocabulary_size": {
                "V=250": {
                    "vocabulary_size": 250,
                    "conditions": {
                        "S3_clean": {"final_sentences": k2_records},
                    },
                }
            },
        }
    }
    vs_bundles = {
        "V=250": {
            "vocabulary_size": 250,
            "conditions": {
                "S3_clean": {"final_sentences": k1_records},
            },
        }
    }

    res = _compute_s3_paired_significance(
        {"S3_clean": s3_bundle},
        vocabulary_scaling_bundles=vs_bundles,
        multi_subject_bundles=ms_bundles,
        language="fr",
    )
    group8_rows = [
        r for r in res.values() if isinstance(r, dict) and int(r.get("group_id", 0)) == 8
    ]
    assert len(group8_rows) >= 2
    for row in group8_rows:
        assert row["blocks"] == 10
        if row["metric"] == "Sent BLEU-1":
            assert row["lhs_value"] == pytest.approx(100.0)
            assert row["rhs_value"] == pytest.approx(100.0)
            assert row["effect_pp"] == pytest.approx(0.0)
        elif row["metric"] == "WER":
            assert row["lhs_value"] == pytest.approx(0.0)
            assert row["rhs_value"] == pytest.approx(50.0)
            assert row["effect_pp"] == pytest.approx(-50.0)


def test_mode_one_l5_closed_set_evaluation_and_single_0pct_oov_reporting(
    tmp_path: Path,
) -> None:
    """验证模式一（路径 B1）下 |V| in {50, 100, |V_full|} (如 |V_full|=130 <= 250) 不被误改写为 2020，且报表每个词表尺度仅输出唯一 0% OOV 口径。"""
    seg_records_full = [
        {
            "segment_index": 0,
            "group_id": "sub-01_run-06:0",
            "material_id": "run-06:0",
            "subject_id": "sub-01",
            "reference_words": ["le", "petit", "prince", "parle", "bien"],
            "predicted_words": ["le", "petit", "prince", "parle", "tres"],
            "in_vocabulary_mask": [True, True, True, True, True],
        },
        {
            "segment_index": 1,
            "group_id": "sub-01_run-06:1",
            "material_id": "run-06:1",
            "subject_id": "sub-01",
            "reference_words": ["le", "renard", "regarde", "la", "rose"],
            "predicted_words": ["le", "renard", "regarde", "le", "prince"],
            "in_vocabulary_mask": [True, True, True, True, True],
        },
    ]
    seg_records_50 = [
        {
            "segment_index": 0,
            "group_id": "sub-01_run-06:0",
            "material_id": "run-06:0",
            "subject_id": "sub-01",
            "reference_words": ["le", "petit", "prince", "parle", "bien"],
            "predicted_words": ["le", "petit", "prince", "parle", "bien"],
            "in_vocabulary_mask": [True, True, True, True, True],
        }
    ]

    vocabulary_scaling_bundles = {
        "V=50": {
            "vocabulary_size": 50,
            "is_full_vocabulary": False,
            "test_token_coverage_pct": 100.0,
            "conditions": {
                "S3_clean": {"final_sentences": seg_records_50},
            },
        },
        "V=100": {
            "vocabulary_size": 100,
            "is_full_vocabulary": False,
            "test_token_coverage_pct": 100.0,
            "conditions": {
                "S3_clean": {"final_sentences": seg_records_full},
            },
        },
        "V=130": {
            "vocabulary_size": 130,
            "is_full_vocabulary": True,
            "test_token_coverage_pct": 100.0,
            "conditions": {
                "S3_clean": {"final_sentences": seg_records_full},
            },
        },
    }

    out_dir = tmp_path / "mode1_eval_out"
    report = evaluate_and_export_sequence_decoding(
        {"S3_clean": {"final_sentences": seg_records_full}},
        dataset="pallier2025",
        language="fr",
        split="test",
        output_dir=out_dir,
        vocabulary_scaling_bundles=vocabulary_scaling_bundles,
        core_vocab_size="full",
        core_vocab_sizes=(50, 100, "full"),
        plot=False,
    )

    assert report["full_vocab_core_subsets"]["vocabulary_size"] == 130
    assert set(report["core_vocabulary_benchmarks"].keys()) == {
        "V=50",
        "V=100",
        "V=130",
    }
    assert "V=2020" not in report["core_vocabulary_benchmarks"]
    assert "V=2020" not in report["vocabulary_scaling"]
    assert (out_dir / "s3_evaluation_summary_test.json").exists()


def test_paper_nomenclature_physical_wait_four_way_pos_and_brain_only() -> None:
    """验证论文标准命名、事件物理等待时长、4 类词性细化与 *_brain_only 评估聚合。"""
    from braindecoding.evaluation.sequence_decoding_evaluation import (
        _aggregate_numeric_dicts_across_seeds,
        classify_part_of_speech_four_way,
        compute_mean_physical_wait_seconds,
        paper_strategy_display_name,
    )

    assert paper_strategy_display_name("S0") == "Word-Only"
    assert paper_strategy_display_name("K0") == "Local (r = 0)"
    assert paper_strategy_display_name("K1") == "Local (r = 1)"
    assert paper_strategy_display_name("K2") == "Local (r = 2)"
    assert paper_strategy_display_name("S3_clean") == "Local (r = 2)"
    assert paper_strategy_display_name("F") == "Full-Context"
    assert paper_strategy_display_name("K2_brain_only") == "Local (r = 2, Brain-Only)"

    assert classify_part_of_speech_four_way("NC") == "noun"
    assert classify_part_of_speech_four_way("NPP") == "noun"
    assert classify_part_of_speech_four_way("V") == "verb"
    assert classify_part_of_speech_four_way("ADJ") == "adj_adv"
    assert classify_part_of_speech_four_way("ADV") == "adj_adv"
    assert classify_part_of_speech_four_way("DET") == "function"
    assert classify_part_of_speech_four_way("NOUN") == "noun"
    assert classify_part_of_speech_four_way("VERB") == "verb"
    assert classify_part_of_speech_four_way("ADP") == "function"
    assert classify_part_of_speech_four_way(None, is_content_word=False) == "function"

    recs = [
        {
            "segment_index": 0,
            "reference_words": ["le", "petit", "prince", "parle"],
            "predicted_words": ["le", "petit", "prince", "parle"],
            "onset_seconds": [0.0, 0.4, 0.9, 1.5],
            "pos_tags": ["DET", "ADJ", "NC", "V"],
        }
    ]
    assert compute_mean_physical_wait_seconds(recs, "S0") == pytest.approx(1.0)
    assert compute_mean_physical_wait_seconds(recs, "K0") == pytest.approx(1.0)
    assert compute_mean_physical_wait_seconds(recs, "K2") == pytest.approx(
        (1.9 + 2.1 + 1.6 + 1.0) / 4.0
    )
    assert compute_mean_physical_wait_seconds(recs, "F") == pytest.approx(
        (2.5 + 2.1 + 1.6 + 1.0) / 4.0
    )

    m6 = evaluate_six_dimensional_metrics(
        recs,
        language="fr",
        condition="K2_brain_only",
        compute_paired_stats=False,
    )
    assert m6["paper_display_name"] == "Local (r = 2, Brain-Only)"
    assert m6["mean_physical_wait_seconds"] == pytest.approx((1.9 + 2.1 + 1.6 + 1.0) / 4.0)
    assert m6["noun_word_acc_pct"] == pytest.approx(100.0)
    assert m6["verb_word_acc_pct"] == pytest.approx(100.0)
    assert m6["adj_adv_word_acc_pct"] == pytest.approx(100.0)
    assert m6["function_pos_word_acc_pct"] == pytest.approx(100.0)
    assert m6["noun_words"] == 1
    assert m6["verb_words"] == 1
    assert m6["adj_adv_words"] == 1
    assert m6["function_pos_words"] == 1

    agg = _aggregate_numeric_dicts_across_seeds(
        [
            {"bleu1_pct": 18.0, "noun_words": 10},
            {"bleu1_pct": 20.0, "noun_words": 12},
        ]
    )
    assert agg["noun_words"] == 11



