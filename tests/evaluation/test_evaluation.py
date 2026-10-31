"""公共词表与检索评价边界的行为锁定测试。"""

import numpy as np
import pytest

from braindecoding.evaluation import retrieval, vocabulary


def _embeddings():
    target_words = ["alpha", "beta", "gamma", "alpha"]
    target_embeddings = np.asarray(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]],
        dtype=np.float32,
    )
    query_embeddings = np.asarray(
        [[1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]],
        dtype=np.float32,
    )
    return query_embeddings, target_embeddings, target_words


def test_retrieval_ranks_and_ties_are_unchanged():
    query_embeddings = np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    candidate_embeddings = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    candidate_ids = ["a", "b", "c"]
    target_ids = ["a", "c", "a"]

    actual = retrieval.retrieval_ranks(
        query_embeddings,
        target_ids,
        candidate_embeddings,
        candidate_ids,
        chunk_size=1,
    )
    np.testing.assert_array_equal(actual, [1.5, 1.0, 2.0])

    actual_summary = retrieval.summarize_retrieval(
        actual, target_ids, candidate_count=3, top_ks=(1, 2)
    )
    assert actual_summary["micro_recall_at_1"] == pytest.approx(1 / 3)
    assert actual_summary["macro_recall_at_1"] == pytest.approx(0.5)


def test_fixed_vocabulary_predictions_and_metrics_are_unchanged():
    query_embeddings, target_embeddings, target_words = _embeddings()
    candidates = ["beta", "alpha", "gamma"]

    actual_predictions = retrieval.fixed_vocabulary_top1_predictions(
        query_embeddings, target_embeddings, target_words, candidates
    )
    assert actual_predictions[:2] == (
        ["alpha", "beta", "gamma", "alpha"],
        ["alpha", "gamma", "gamma", "beta"],
    )
    np.testing.assert_array_equal(actual_predictions[2], [0, 1, 2, 3])

    actual_metrics = retrieval.fixed_vocabulary_retrieval_metrics(
        query_embeddings,
        target_embeddings,
        target_words,
        candidates,
        vocabulary_name="test3",
    )
    assert actual_metrics["retrieval_acc1_vocab=test3"] == 0.5
    assert actual_metrics["retrieval_acc10_vocab=test3"] == 1.0


def test_direct_candidate_scores_preserve_ranking_and_ties():
    query_embeddings, _, target_words = _embeddings()
    candidate_words = ["alpha", "beta", "gamma"]
    candidate_embeddings = np.eye(3, dtype=np.float32)
    normalized_queries = retrieval.normalize_rows(query_embeddings)
    scores = normalized_queries @ candidate_embeddings.T

    ranks = retrieval.score_matrix_ranks(scores, target_words, candidate_words)
    metrics = retrieval.fixed_vocabulary_score_metrics(
        scores,
        target_words,
        candidate_words,
        candidate_words,
        vocabulary_name="test3",
    )

    np.testing.assert_array_equal(ranks, [1.0, 2.5, 1.0, 2.5])
    assert metrics["retrieval_acc1_vocab=test3"] == 0.5
    assert metrics["retrieval_acc10_vocab=test3"] == 1.0


def test_frequency_vocabulary_is_deterministic_for_duplicates_and_ties():
    words = [" Beta ", "alpha", "BETA", "ALPHA", "gamma", "delta"]
    normalizer = lambda value: str(value).strip().lower()
    expected = ("alpha", "beta", "delta")
    assert vocabulary.build_frequency_vocabulary(
        words, size=3, normalize=normalizer
    ) == expected
    assert vocabulary.build_frequency_vocabulary(
        reversed(words), size=3, normalize=normalizer
    ) == expected


def test_vocabulary_builder_only_reads_explicit_words():
    class ExplicitWords:
        @property
        def test(self):
            raise AssertionError("词表构建器不应读取 test 数据。")

        def __iter__(self):
            return iter(["甲", "乙", "甲"])

    assert vocabulary.build_frequency_vocabulary(ExplicitWords(), 2) == ("甲", "乙")


def test_frozen_vocabulary_validation_and_metadata():
    frozen = vocabulary.validate_frozen_vocabulary(
        [" Alpha ", "beta"],
        expected_size=2,
        normalize=lambda value: str(value).strip().lower(),
    )
    assert frozen == ("alpha", "beta")
    with pytest.raises(ValueError, match="重复"):
        vocabulary.validate_frozen_vocabulary(
            ["Alpha", "alpha"], normalize=lambda value: str(value).lower()
        )

    metadata = vocabulary.build_vocabulary_metadata(
        frozen,
        source_split="train",
        policy="frequency_desc_word_asc",
        normalization="lower_strip",
        word_counts={"alpha": 4, "beta": 3, "unused": 2},
    )
    assert metadata == {
        "vocabulary": ["alpha", "beta"],
        "size": 2,
        "source_split": "train",
        "policy": "frequency_desc_word_asc",
        "normalization": "lower_strip",
        "word_counts": {"alpha": 4, "beta": 3},
    }


def test_word_training_evaluators_reach_the_same_retrieval_implementation():
    from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import train as chinese
    from braindecoding.tasks.word_decoding.pallier2025 import train as pallier

    for module in (chinese, pallier):
        assert (
            module.fixed_vocabulary_retrieval_metrics
            is retrieval.fixed_vocabulary_retrieval_metrics
        )


def _make_synthetic_multi_seed_model_summary(
    *,
    stage: str,
    exp_tag: str,
    method_id: str,
    v50_clean10: float,
    v50_shift10: float,
    v50_clean1: float,
    v50_shift1: float,
    fv_clean10: float,
    fv_shift10: float,
    fv_clean1: float,
    fv_shift1: float,
) -> dict:
    seeds = [0, 1, 2]
    offsets = [-0.005, 0.0, 0.005]
    v50_gap10 = v50_clean10 - v50_shift10
    v50_gap1 = v50_clean1 - v50_shift1
    fv_gap10 = fv_clean10 - fv_shift10
    fv_gap1 = fv_clean1 - fv_shift1

    per_seed_metrics = {
        "vocab50_macro_top10": [
            {
                "seed": s,
                "clean": v50_clean10 + off,
                "shifted": v50_shift10 + off * 0.4,
                "gap": (v50_clean10 + off) - (v50_shift10 + off * 0.4),
            }
            for s, off in zip(seeds, offsets)
        ],
        "vocab50_macro_top1": [
            {
                "seed": s,
                "clean": v50_clean1 + off,
                "shifted": v50_shift1,
                "gap": (v50_clean1 + off) - v50_shift1,
            }
            for s, off in zip(seeds, offsets)
        ],
        "full_vocab_macro_top10": [
            {
                "seed": s,
                "clean": fv_clean10 + off,
                "shifted": fv_shift10,
                "gap": (fv_clean10 + off) - fv_shift10,
            }
            for s, off in zip(seeds, offsets)
        ],
        "full_vocab_macro_top1": [
            {
                "seed": s,
                "clean": fv_clean1 + off * 0.5,
                "shifted": fv_shift1,
                "gap": (fv_clean1 + off * 0.5) - fv_shift1,
            }
            for s, off in zip(seeds, offsets)
        ],
        "full_vocab_mrr": [
            {
                "seed": s,
                "clean": fv_clean1 * 1.4,
                "shifted": fv_shift1 * 2.0,
                "gap": fv_clean1 * 1.4 - fv_shift1 * 2.0,
            }
            for s in seeds
        ],
    }
    shift_anchors = [-15.0, -30.0, -45.0, -60.0, -90.0, -120.0]
    sh_jitters = [-0.004, -0.002, 0.0, 0.001, 0.002, 0.003]
    conditions_aggregate = [
        {
            "key": f"shift_{int(abs(sec))}s",
            "shift_seconds": sec,
            "mean_vocab50_macro_top10": v50_shift10 + jit,
            "mean_gap_vocab50_macro_top10": v50_clean10 - (v50_shift10 + jit),
            "mean_vocab50_macro_top1": v50_shift1 + jit * 0.2,
            "mean_gap_vocab50_macro_top1": v50_clean1 - (v50_shift1 + jit * 0.2),
            "mean_full_vocab_macro_top10": fv_shift10 + jit * 0.3,
            "mean_gap_full_vocab_macro_top10": fv_clean10 - (fv_shift10 + jit * 0.3),
            "mean_full_vocab_macro_top1": fv_shift1,
            "mean_gap_full_vocab_macro_top1": fv_clean1 - fv_shift1,
            "mean_full_vocab_mrr": fv_shift1 * 2.0,
            "mean_gap_full_vocab_mrr": fv_clean1 * 1.4 - fv_shift1 * 2.0,
        }
        for sec, jit in zip(shift_anchors, sh_jitters)
    ]
    ctrl_entry = {
        "enabled": True,
        "method_id": method_id,
        "label": method_id,
        "mean_clean_vocab50_macro_top10": v50_clean10,
        "std_clean_vocab50_macro_top10": 0.005,
        "mean_shifted_vocab50_macro_top10": v50_shift10,
        "std_shifted_vocab50_macro_top10": 0.004,
        "mean_gap_vocab50_macro_top10": v50_gap10,
        "std_gap_vocab50_macro_top10": 0.005,
        "p_shifted_vs_chance_vocab50_macro_top10": 0.96 if abs(v50_shift10 - 0.20) < 0.01 else 0.0001,
        "p_str_shifted_vs_chance_vocab50_macro_top10": (
            "p=0.96 ns" if abs(v50_shift10 - 0.20) < 0.01 else "p<0.0001 ***"
        ),
        "p_gap_vocab50_macro_top10": 1e-5,
        "p_str_gap_vocab50_macro_top10": "p<0.0001 ***",
        "mean_clean_vocab50_macro_top1": v50_clean1,
        "std_clean_vocab50_macro_top1": 0.004,
        "mean_shifted_vocab50_macro_top1": v50_shift1,
        "std_shifted_vocab50_macro_top1": 0.002,
        "mean_gap_vocab50_macro_top1": v50_gap1,
        "std_gap_vocab50_macro_top1": 0.004,
        "p_gap_vocab50_macro_top1": 1e-5,
        "p_str_gap_vocab50_macro_top1": "p<0.0001 ***",
        "mean_clean_full_vocab_macro_top10": fv_clean10,
        "std_clean_full_vocab_macro_top10": 0.004,
        "mean_shifted_full_vocab_macro_top10": fv_shift10,
        "std_shifted_full_vocab_macro_top10": 0.002,
        "mean_gap_full_vocab_macro_top10": fv_gap10,
        "std_gap_full_vocab_macro_top10": 0.004,
        "p_gap_full_vocab_macro_top10": 1e-5,
        "p_str_gap_full_vocab_macro_top10": "p<0.0001 ***",
        "mean_clean_full_vocab_macro_top1": fv_clean1,
        "std_clean_full_vocab_macro_top1": 0.003,
        "mean_shifted_full_vocab_macro_top1": fv_shift1,
        "std_shifted_full_vocab_macro_top1": 0.001,
        "mean_gap_full_vocab_macro_top1": fv_gap1,
        "std_gap_full_vocab_macro_top1": 0.003,
        "p_gap_full_vocab_macro_top1": 1e-5,
        "p_str_gap_full_vocab_macro_top1": "p<0.0001 ***",
        "mean_clean_full_vocab_mrr": fv_clean1 * 1.4,
        "std_clean_full_vocab_mrr": 0.003,
        "mean_shifted_full_vocab_mrr": fv_shift1 * 2.0,
        "std_shifted_full_vocab_mrr": 0.001,
        "mean_gap_full_vocab_mrr": fv_clean1 * 1.4 - fv_shift1 * 2.0,
        "std_gap_full_vocab_mrr": 0.003,
        "per_seed_metrics": per_seed_metrics,
        "conditions_aggregate": conditions_aggregate,
    }
    subj_rows = [
        {
            "subject_id": f"sub-{i:02d}",
            "mean_macro_top10": v50_clean10 + (i - 5.5) * 0.004,
            "mean_macro_top1": v50_clean1 + (i - 5.5) * 0.003,
            "mean_micro_top10": v50_clean10 + (i - 5.5) * 0.004,
            "mean_mrr": v50_clean1 * 1.5,
            "mean_full_vocab_macro_top10": fv_clean10 + (i - 5.5) * 0.003,
            "mean_full_vocab_macro_top1": fv_clean1 + (i - 5.5) * 0.002,
            "mean_full_vocab_micro_top10": fv_clean10 + (i - 5.5) * 0.003,
            "mean_full_vocab_mrr": fv_clean1 * 1.4,
        }
        for i in range(1, 11)
    ]
    return {
        "status": "multi_seed_evaluation_completed",
        "dataset": "pallier2025",
        "stage": stage,
        "exp_tag": exp_tag,
        "split": "test",
        "seeds": seeds,
        "seed_count": 3,
        "per_seed": [{"seed": s} for s in seeds],
        "vocabulary_sweep_aggregate": {
            "N50": {
                "mean_macro_top10": v50_clean10,
                "std_macro_top10": 0.005,
                "mean_macro_top1": v50_clean1,
                "std_macro_top1": 0.004,
            },
            "full_vocab": {
                "mean_macro_top10": fv_clean10,
                "std_macro_top10": 0.004,
                "mean_macro_top1": fv_clean1,
                "std_macro_top1": 0.003,
            },
        },
        "per_subject_summary_aggregate": subj_rows,
        "negative_controls_by_method": {method_id: ctrl_entry},
        "negative_controls_aggregate": ctrl_entry,
    }


def test_cross_model_paired_comparison_and_standalone_export(tmp_path):
    from braindecoding.evaluation.evaluation_reporting import (
        export_and_print_cross_model_comparison_report,
    )

    models_map = {
        "word_only": _make_synthetic_multi_seed_model_summary(
            stage="word",
            exp_tag="lr4e-4_bs256",
            method_id="raw",
            v50_clean10=0.5100,
            v50_shift10=0.2000,
            v50_clean1=0.1450,
            v50_shift1=0.0200,
            fv_clean10=0.2550,
            fv_shift10=0.0050,
            fv_clean1=0.0550,
            fv_shift1=0.0005,
        ),
        "context_unfrozen": _make_synthetic_multi_seed_model_summary(
            stage="context",
            exp_tag="lr4e-4_bs256",
            method_id="raw",
            v50_clean10=0.5750,
            v50_shift10=0.3550,
            v50_clean1=0.1900,
            v50_shift1=0.1050,
            fv_clean10=0.2900,
            fv_shift10=0.1150,
            fv_clean1=0.0750,
            fv_shift1=0.0400,
        ),
        "context_debiased": _make_synthetic_multi_seed_model_summary(
            stage="context",
            exp_tag="lr4e-4_bs256_window_1.0_siamese_debiased_lamda_0.5_history_window",
            method_id="cms:eta=0.5",
            v50_clean10=0.5720,
            v50_shift10=0.2002,
            v50_clean1=0.2620,
            v50_shift1=0.0201,
            fv_clean10=0.3680,
            fv_shift10=0.0050,
            fv_clean1=0.1360,
            fv_shift1=0.0005,
        ),
        "prefix_revision": _make_synthetic_multi_seed_model_summary(
            stage="prefix_revision",
            exp_tag="2_lr4e-4_bs256_window_1.0_siamese_debiased_lamda_0.5_history_window",
            method_id="cms:eta=0.5",
            v50_clean10=0.5635,
            v50_shift10=0.1998,
            v50_clean1=0.2560,
            v50_shift1=0.0200,
            fv_clean10=0.3610,
            fv_shift10=0.0050,
            fv_clean1=0.1310,
            fv_shift1=0.0005,
        ),
    }

    res = export_and_print_cross_model_comparison_report(
        models_map,
        output_dir=tmp_path,
        split="test",
        print_to_stdout=False,
    )
    assert res["model_count"] == 4
    htests = {h["hypothesis_id"]: h for h in res["hypothesis_tests"]}
    assert set(htests.keys()) == {
        "H1_unfrozen_context_vs_word_only",
        "H2_debiased_context_vs_word_only",
        "H3_prefix_revision_vs_debiased_context",
    }

    # Test 1: Unfrozen CNN vs Word-Only -> Net Gap delta < 0 and p < 0.05
    h1_prim = htests["H1_unfrozen_context_vs_word_only"]["primary_row"]
    assert h1_prim["delta_mean"] < 0.0
    assert h1_prim["p_value"] < 0.05
    assert h1_prim["hypothesis_supported"] is True

    # Test 2: Debiased Context vs Word-Only -> Net Gap & Top-1 delta > 0 and p < 0.001
    h2_rows = {
        (r["metric_key"], r["quantity"]): r
        for r in htests["H2_debiased_context_vs_word_only"]["rows"]
    }
    assert h2_rows[("vocab50_macro_top10", "net_gap")]["delta_mean"] > 0.0
    assert h2_rows[("vocab50_macro_top10", "net_gap")]["p_value"] < 0.001
    assert h2_rows[("vocab50_macro_top1", "clean")]["delta_mean"] > 0.0
    assert h2_rows[("vocab50_macro_top1", "clean")]["p_value"] < 0.001

    # Test 3: Prefix-Revision vs Debiased Context -> p > 0.05 (no significant loss) + >50% wait reduction
    h3 = htests["H3_prefix_revision_vs_debiased_context"]
    assert h3["lookahead_saving_pct"] > 50.0
    assert h3["primary_row"]["p_value"] > 0.05
    assert h3["primary_row"]["significance"] == "ns"
    assert h3["primary_row"]["hypothesis_supported"] is True

    # Check exported standalone files
    md_file = tmp_path / "multi_seed_cross_model_comparison_test.md"
    txt_file = tmp_path / "multi_seed_cross_model_comparison_report_test.txt"
    json_file = tmp_path / "multi_seed_cross_model_comparison_test.json"
    sum_csv = tmp_path / "tables" / "multi_seed_cross_model_summary_test.csv"
    pair_csv = tmp_path / "tables" / "multi_seed_cross_model_paired_tests_test.csv"
    for f in (md_file, txt_file, json_file, sum_csv, pair_csv):
        assert f.is_file()
        assert f.stat().st_size > 0


