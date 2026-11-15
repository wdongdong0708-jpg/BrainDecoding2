"""论文重构多研究导出器与诊断指标（manuscript_study_exporter & export_manuscript_studies）单元测试。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding.config import PROJECT_ROOT
import braindecoding.evaluation as ev_pkg
from braindecoding.evaluation.manuscript_study_exporter import (
    compute_duration_prior_alignment_metrics,
    compute_word_frequency_quartile_metrics,
    compute_word_horizon_physical_waits,
    export_decoded_sentence_quantile_examples,
    export_sentence_and_subject_coupling_studies,
    export_sentence_neural_top5_matrix,
    export_subject_retrieval_sentence_coupling,
    export_synthetic_and_horizon_studies,
    export_word_top10_neighborhood_examples,
    run_manuscript_study_exports,
)


def test_lazy_exports_and_subpackage_alias_resolution() -> None:
    """验证 braindecoding.evaluation 延迟导出与包别名解析全部可用。"""
    expected_symbols = (
        "compute_word_horizon_physical_waits",
        "compute_duration_prior_alignment_metrics",
        "compute_word_frequency_quartile_metrics",
        "export_word_top10_neighborhood_examples",
        "export_sentence_neural_top5_matrix",
        "export_decoded_sentence_quantile_examples",
        "export_subject_retrieval_sentence_coupling",
        "evaluate_five_model_t5_diagnostics",
        "collect_t5_long_window_comparison_rows",
        "export_synthetic_and_horizon_studies",
        "export_sentence_and_subject_coupling_studies",
        "run_manuscript_study_exports",
    )
    for sym in expected_symbols:
        assert hasattr(ev_pkg, sym), f"Missing lazy symbol {sym!r}"

    import braindecoding.manuscript_study_exporter as alias_mod

    assert alias_mod.run_manuscript_study_exports is run_manuscript_study_exports


def test_compute_word_horizon_physical_waits() -> None:
    """验证各局部跨词视野 r=0/1/2 与 Full-Context 物理等待时长。"""
    df = pd.DataFrame(
        {
            "split": ["test"] * 5,
            "runtime_context_uid": ["ctx-1"] * 5,
            "in_recording_index": [0, 1, 2, 3, 4],
            "onset_seconds": [0.0, 0.5, 1.0, 1.5, 2.0],
        }
    )
    rows = compute_word_horizon_physical_waits(
        df, split="test", horizons=(0, 1, 2, "full"), window_seconds=1.0, common_horizon_delay=2
    )
    by_h = {r["horizon"]: r for r in rows}
    assert by_h["r=0"]["paper_display_name"] == "Local (r = 0)"
    assert by_h["r=0"]["query_count"] == 3
    assert by_h["r=0"]["mean_wait_seconds"] == pytest.approx(1.0)
    assert by_h["r=1"]["mean_wait_seconds"] == pytest.approx(1.5)
    assert by_h["r=2"]["mean_wait_seconds"] == pytest.approx(2.0)
    # Full-Context over first 3 positions (2.0+1.0-0.0=3.0, 2.5, 2.0) -> mean 2.5
    assert by_h["full"]["paper_display_name"] == "Full-Context"
    assert by_h["full"]["mean_wait_seconds"] == pytest.approx(2.5)


def test_compute_duration_prior_alignment_metrics() -> None:
    """验证模型输出符合词长先验时 Δ_JS > 0 且高于均匀预测。"""
    vocab = ["un", "le", "deux", "prince", "planete", "astronome"]
    vocab_durs = [0.15, 0.15, 0.25, 0.45, 0.55, 0.70]
    train_words = vocab * 4
    train_durs = vocab_durs * 4

    # Test queries mirror vocabulary durations (5 repeats each)
    target_words = vocab * 3
    target_durs = vocab_durs * 3
    n_q = len(target_words)

    aligned_scores = np.zeros((n_q, len(vocab)), dtype=np.float32)
    for i, dur in enumerate(target_durs):
        # Sharply favor words matching the true query duration (short vs long)
        for j, vd in enumerate(vocab_durs):
            aligned_scores[i, j] = -abs(dur - vd) * 3.0

    uniform_scores = np.zeros((n_q, len(vocab)), dtype=np.float32)

    res_aligned = compute_duration_prior_alignment_metrics(
        aligned_scores,
        vocab,
        target_words,
        target_durs,
        train_words=train_words,
        train_durations_seconds=train_durs,
        n_permutations=40,
    )
    res_uniform = compute_duration_prior_alignment_metrics(
        uniform_scores,
        vocab,
        target_words,
        target_durs,
        train_words=train_words,
        train_durations_seconds=train_durs,
        n_permutations=40,
    )
    assert res_aligned["delta_js_bits"] > 0.0
    assert res_aligned["delta_js_bits"] > res_uniform["delta_js_bits"]
    assert res_aligned["top10_duration_error_reduction_ms"] >= res_uniform["top10_duration_error_reduction_ms"]


def test_compute_word_frequency_quartile_metrics() -> None:
    """验证训练词频 Q1-Q4 四分位分层指标。"""
    vocab = ["rare1", "rare2", "mid1", "mid2", "hi1", "hi2", "freq1", "freq2"]
    train_counts = {w: (i + 1) * 10 for i, w in enumerate(vocab)}
    scores = np.zeros((4, len(vocab)), dtype=np.float32)
    # rare1 is rank 1; mid1 is rank 1; hi1 rank 2; freq1 rank 1
    scores[0, 0] = 5.0
    scores[1, 2] = 5.0
    scores[2, 5] = 6.0
    scores[2, 4] = 4.0
    scores[3, 6] = 5.0

    q_rows = compute_word_frequency_quartile_metrics(
        scores,
        vocab,
        ["rare1", "mid1", "hi1", "freq1"],
        train_counts,
        subject_ids=["sub-01", "sub-01", "sub-02", "sub-02"],
    )
    assert [r["quartile"] for r in q_rows] == [
        "Q1_rare",
        "Q2_low_mid",
        "Q3_high_mid",
        "Q4_frequent",
    ]
    assert q_rows[0]["top1_acc_pct"] == pytest.approx(100.0)
    assert q_rows[2]["top1_acc_pct"] == pytest.approx(0.0)
    assert q_rows[2]["top5_acc_pct"] == pytest.approx(100.0)


def _build_fixture_artifacts(tmp_path: Path) -> dict[str, Path]:
    vocab = ["le", "petit", "prince", "parle", "vite", "bien"]
    vocab_emb = torch.eye(len(vocab), 8, dtype=torch.float32)
    segments = [
        {
            "segment_index": 0,
            "group_id": "sub-01_run-01:0",
            "subject_id": "sub-01",
            "partition": "evaluation",
            "event_indices": [0, 1, 2, 3],
            "reference_words": ["le", "petit", "prince", "parle"],
            "onset_seconds": [0.0, 0.4, 0.9, 1.5],
            "offset_seconds": [0.2, 0.7, 1.3, 1.9],
            "pos_tags": ["DET", "ADJ", "NC", "V"],
            "is_content_word_mask": [False, True, True, True],
        },
        {
            "segment_index": 1,
            "group_id": "sub-02_run-01:1",
            "subject_id": "sub-02",
            "partition": "evaluation",
            "event_indices": [4, 5],
            "reference_words": ["parle", "vite"],
            "onset_seconds": [3.0, 3.6],
            "offset_seconds": [3.4, 4.0],
            "pos_tags": ["V", "ADV"],
            "is_content_word_mask": [True, True],
        },
        {
            "segment_index": 2,
            "group_id": "sub-03_run-01:2",
            "subject_id": "sub-03",
            "partition": "evaluation",
            "event_indices": [6, 7],
            "reference_words": ["prince", "bien"],
            "onset_seconds": [5.0, 5.6],
            "offset_seconds": [5.4, 5.9],
            "pos_tags": ["NC", "ADV"],
            "is_content_word_mask": [True, True],
        },
    ]
    ref_words = ["le", "petit", "prince", "parle", "parle", "vite", "prince", "bien"]
    subj_ids = ["sub-01"] * 4 + ["sub-02"] * 2 + ["sub-03"] * 2
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}

    s0_mat = torch.zeros((len(ref_words), len(vocab)), dtype=torch.float32)
    k2_mat = torch.zeros((len(ref_words), len(vocab)), dtype=torch.float32)
    for r_i, rw in enumerate(ref_words):
        col = vocab_to_idx[rw]
        s0_mat[r_i, (col + 1) % len(vocab)] = 1.5
        k2_mat[r_i, col] = 3.0

    payload = {
        "dataset": "pallier2025",
        "language": "fr",
        "split": "test",
        "vocabulary": vocab,
        "vocabulary_embeddings": vocab_emb,
        "vocabulary_train_counts": {w: (i + 1) * 20 for i, w in enumerate(vocab)},
        "reference_words": ref_words,
        "subject_ids": subj_ids,
        "target_durations_seconds": [0.2, 0.3, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3],
        "segments": segments,
        "clean_scores": {
            "S0": s0_mat,
            "K0": s0_mat,
            "K1": (s0_mat + k2_mat) * 0.5,
            "K2": k2_mat,
            "F": k2_mat * 0.9,
        },
        "shifted_scores": {"K0": s0_mat, "K1": s0_mat, "K2": s0_mat, "F": s0_mat},
    }
    pt_path = tmp_path / "neural_vocabulary_scores_test.pt"
    torch.save(payload, pt_path)

    def _mk_records(preds_list: list[list[str]]) -> list[dict]:
        return [
            {
                **seg,
                "predicted_words": pred,
            }
            for seg, pred in zip(segments, preds_list)
        ]

    decoded_bundles = {
        "S0": {"final_sentences": _mk_records([["petit", "petit", "bien", "vite"], ["bien", "bien"], ["bien", "bien"]])},
        "K0": {"final_sentences": _mk_records([["le", "vite", "bien", "parle"], ["parle", "bien"], ["prince", "vite"]])},
        "S3_clean": {"final_sentences": _mk_records([["le", "petit", "prince", "parle"], ["parle", "bien"], ["bien", "bien"]])},
        "F": {"final_sentences": _mk_records([["le", "petit", "bien", "parle"], ["parle", "vite"], ["prince", "bien"]])},
        "K2_brain_only": {"final_sentences": _mk_records([["le", "petit", "prince", "parle"], ["parle", "vite"], ["prince", "bien"]])},
        "LM-Only": {"final_sentences": _mk_records([["le", "vite", "vite", "vite"], ["vite", "vite"], ["vite", "vite"]])},
    }
    dec_path = tmp_path / "sequence_decoding_test.json"
    dec_path.write_text(json.dumps({"decoded_bundles": decoded_bundles}), encoding="utf-8")

    word_eval = {
        "per_subject": {
            "sub-01": {"vocab50_macro_top10": 0.72},
            "sub-02": {"vocab50_macro_top10": 0.58},
            "sub-03": {"vocab50_macro_top10": 0.41},
        }
    }
    word_eval_path = tmp_path / "multi_seed_evaluation_test.json"
    word_eval_path.write_text(json.dumps(word_eval), encoding="utf-8")

    eval_summary = {
        "per_subject": [
            {"subject_id": "sub-01", "bleu1_pct": 85.0, "word_acc_pct": 90.0, "segment_count": 1},
            {"subject_id": "sub-02", "bleu1_pct": 55.0, "word_acc_pct": 50.0, "segment_count": 1},
            {"subject_id": "sub-03", "bleu1_pct": 15.0, "word_acc_pct": 10.0, "segment_count": 1},
        ]
    }
    eval_summary_path = tmp_path / "s3_evaluation_summary_test.json"
    eval_summary_path.write_text(json.dumps(eval_summary), encoding="utf-8")

    event_df = pd.DataFrame(
        {
            "split": ["test"] * 8,
            "runtime_context_uid": ["ctx-1"] * 4 + ["ctx-2"] * 4,
            "in_recording_index": [0, 1, 2, 3, 0, 1, 2, 3],
            "onset_seconds": [0.0, 0.4, 0.9, 1.5, 3.0, 3.6, 5.0, 5.6],
            "offset_seconds": [0.2, 0.7, 1.3, 1.9, 3.4, 4.0, 5.4, 5.9],
            "pos": ["DET", "ADJ", "NC", "V", "V", "ADV", "NC", "ADV"],
            "is_content_word": [False, True, True, True, True, True, True, True],
        }
    )
    ev_path = tmp_path / "event_table.csv"
    event_df.to_csv(ev_path, index=False)

    return {
        "pt_path": pt_path,
        "dec_path": dec_path,
        "word_eval_path": word_eval_path,
        "eval_summary_path": eval_summary_path,
        "ev_path": ev_path,
    }


def test_qualitative_examples_matrices_and_subject_coupling(tmp_path: Path) -> None:
    """验证词邻域、逐词 Top-5 矩阵、分位数例句与跨被试耦合（含 dict 与 list 形式 per_subject）。"""
    arts = _build_fixture_artifacts(tmp_path)
    payload = torch.load(arts["pt_path"], map_location="cpu", weights_only=False)
    decoded_bundles = json.loads(arts["dec_path"].read_text(encoding="utf-8"))["decoded_bundles"]
    word_eval = json.loads(arts["word_eval_path"].read_text(encoding="utf-8"))
    eval_summary = json.loads(arts["eval_summary_path"].read_text(encoding="utf-8"))

    nb = export_word_top10_neighborhood_examples(payload, top_k=5, max_examples=3)
    assert len(nb) >= 1
    assert "K2" in nb[0]["conditions"]
    assert nb[0]["conditions"]["K2"]["paper_display_name"] == "Local (r = 2)"

    mat = export_sentence_neural_top5_matrix(payload, segment_index=0, state="K2", top_k=5)
    assert mat["state"] == "K2"
    assert len(mat["positions"]) == 4
    for pos in mat["positions"]:
        probs = [c["prob"] for c in pos["top_candidates"]]
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert sum(probs) <= 1.0 + 1e-6

    q_exs = export_decoded_sentence_quantile_examples(
        decoded_bundles, primary_condition="S3_clean", quantiles=(0.90, 0.50, 0.10), language="fr"
    )
    assert len(q_exs) == 3
    assert "K2_brain_only" in q_exs[0]["predictions"]

    coup_from_bundles = export_subject_retrieval_sentence_coupling(
        word_eval, decoded_bundles, condition="S3_clean", language="fr"
    )
    assert coup_from_bundles["subject_count"] == 3
    assert coup_from_bundles["pearson_r_bleu1"] > 0.5

    coup_from_report = export_subject_retrieval_sentence_coupling(
        word_eval, eval_summary, condition="S3_clean", language="fr"
    )
    assert coup_from_report["subject_count"] == 3
    assert coup_from_report["pearson_r_bleu1"] > 0.9

    word_eval_agg_list = {
        "per_subject_summary_aggregate": [
            {"subject_id": "sub-01", "mean_macro_top10": 0.72},
            {"subject_id": "sub-02", "mean_macro_top10": 0.58},
            {"subject_id": "sub-03", "mean_macro_top10": 0.41},
        ]
    }
    coup_from_agg_list = export_subject_retrieval_sentence_coupling(
        word_eval_agg_list, eval_summary, condition="S3_clean", language="fr"
    )
    assert coup_from_agg_list["subject_count"] == 3
    assert coup_from_agg_list["pearson_r_bleu1"] == pytest.approx(
        coup_from_report["pearson_r_bleu1"]
    )


def test_full_exporter_pipeline_and_cli_to_tmp_path(tmp_path: Path) -> None:
    """验证汇总导出函数及 scripts/export_manuscript_studies.py 端到端 CLI 落盘。"""
    arts = _build_fixture_artifacts(tmp_path)
    out_dir_py = tmp_path / "out_py"
    fake_repo = tmp_path / "fake_repo"
    synth_dir = fake_repo / "data" / "results" / "studies" / "synthetic_timing_v2"
    lw_dir = fake_repo / "data" / "results" / "studies" / "single_word_long_window_test_v2"
    synth_dir.mkdir(parents=True)
    lw_dir.mkdir(parents=True)
    (synth_dir / "results.json").write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "dataset": "pallier2025",
                        "condition": "shared",
                        "horizon": 2,
                        "n_seeds": 3,
                        "mean": 0.21,
                        "sample_sd": 0.01,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "dataset": "pallier2025",
                "condition": "word_1s",
                "population": "all",
                "candidates": "all",
                "candidate_count": 50,
                "unique_events": 100,
                "seeds": 3,
                "subject_equal_recall_at_10_mean": 0.40,
                "subject_equal_recall_at_10_sd": 0.01,
                "subject_equal_recall_at_1_mean": 0.06,
                "subject_equal_recall_at_1_sd": 0.005,
                "macro_recall_at_10_mean": 0.39,
                "macro_recall_at_10_sd": 0.01,
            }
        ]
    ).to_csv(lw_dir / "summary.csv", index=False)

    ev_df = pd.read_csv(arts["ev_path"])
    synth_res = export_synthetic_and_horizon_studies(
        repo_root=fake_repo, output_dir=out_dir_py, event_table=ev_df, dataset="pallier2025", split="test"
    )
    # 严禁回退加载旧版 BGE-M3 synthetic_timing_v2 / single_word_long_window_test_v2
    assert "synthetic_timing_csv" not in synth_res["exported_files"]
    assert "long_window_csv" not in synth_res["exported_files"]
    assert Path(synth_res["exported_files"]["physical_wait_csv"]).is_file()

    payload = torch.load(arts["pt_path"], map_location="cpu", weights_only=False)
    decoded_bundles = json.loads(arts["dec_path"].read_text(encoding="utf-8"))["decoded_bundles"]
    word_eval = json.loads(arts["word_eval_path"].read_text(encoding="utf-8"))
    sent_res = export_sentence_and_subject_coupling_studies(
        output_dir=out_dir_py,
        decoded_bundles=decoded_bundles,
        word_eval_summary=word_eval,
        score_payload=payload,
        event_table=ev_df,
        language="fr",
        split="test",
        n_permutations=25,
    )
    assert Path(sent_res["exported_files"]["duration_prior_alignment_csv"]).is_file()
    assert Path(sent_res["exported_files"]["brain_only_vs_lm_ablation_csv"]).is_file()
    assert Path(sent_res["exported_files"]["subject_coupling_csv"]).is_file()

    out_dir_cli = tmp_path / "out_cli"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_manuscript_studies.py"),
        "--dataset",
        "pallier2025",
        "--split",
        "test",
        "--output-dir",
        str(out_dir_cli),
        "--repo-root",
        str(fake_repo),
        "--export-pt",
        str(arts["pt_path"]),
        "--decoding-json",
        str(arts["dec_path"]),
        "--eval-summary-json",
        str(arts["eval_summary_path"]),
        "--word-eval-json",
        str(arts["word_eval_path"]),
        "--event-table-csv",
        str(arts["ev_path"]),
        "--n-permutations",
        "25",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout_obj = json.loads(proc.stdout)
    assert stdout_obj["status"] == "completed"
    manifest_p = out_dir_cli / "manuscript_export_manifest.json"
    assert manifest_p.is_file()
    for fpath in stdout_obj["exported_files"].values():
        p = Path(fpath)
        assert p.is_file() and p.stat().st_size > 0
        assert out_dir_cli.resolve() in p.resolve().parents


def test_chinese_canonical_events_cache_dir_and_multi_subject_enrichment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回归测试：多被试共享上下文编号分组、BRAINDECODING_CACHE_DIR 解析、是否内容词映射与旧缓存回填。"""
    from braindecoding.decoding.neural_score_exporter import (
        enrich_segments_from_event_table,
    )

    cache_dir = tmp_path / "local_cache"
    ev_dir = cache_dir / "pallier2025" / "events"
    ev_dir.mkdir(parents=True)
    monkeypatch.setenv("BRAINDECODING_CACHE_DIR", str(cache_dir))

    # Build canonical Chinese events.csv with train + test splits and 2 subjects sharing the same 上下文编号
    rows: list[dict[str, object]] = []
    train_words = [("le", 0.0, 0.2, "DET", False), ("petit", 0.3, 0.6, "ADJ", True), ("prince", 0.7, 1.1, "NC", True), ("parle", 1.2, 1.6, "V", True)]
    test_words = [("le", 10.0, 10.2, "DET", False), ("petit", 10.3, 10.6, "ADJ", True), ("prince", 10.7, 11.1, "NC", True), ("parle", 11.2, 11.6, "V", True)]
    for subj, rec_tag in (("sub-01", "pallier2025|sub-01|run-06"), ("sub-02", "pallier2025|sub-02|run-06")):
        for idx, (w, st, en, pos, is_c) in enumerate(train_words):
            rows.append(
                {
                    "事件编号": f"{rec_tag}|tr-{idx}",
                    "受试者": subj,
                    "记录编号": f"pallier2025|{subj}|run-01",
                    "材料编号": "mat-tr",
                    "划分单元": "unit-tr",
                    "记录内序号": idx,
                    "词": w,
                    "标准词": w,
                    "开始时间": st,
                    "结束时间": en,
                    "上下文编号": "pallier2025|run-01|sequence-000",
                    "数据划分": "train",
                    "是否可训练": True,
                    "排除原因": "",
                    "词性": pos,
                    "是否内容词": is_c,
                }
            )
        for idx, (w, st, en, pos, is_c) in enumerate(test_words):
            # Offset sub-02 clock by +1000s to catch any cross-subject merging bug in physical wait computation
            clk = 1000.0 if subj == "sub-02" else 0.0
            rows.append(
                {
                    "事件编号": f"{rec_tag}|te-{idx}",
                    "受试者": subj,
                    "记录编号": rec_tag,
                    "材料编号": "mat-te",
                    "划分单元": "unit-te",
                    "记录内序号": idx,
                    "词": w,
                    "标准词": w,
                    "开始时间": st + clk,
                    "结束时间": en + clk,
                    "上下文编号": "pallier2025|run-06|sequence-000",
                    "数据划分": "test",
                    "是否可训练": True,
                    "排除原因": "",
                    "词性": pos,
                    "是否内容词": is_c,
                }
            )
    ev_df_disk = pd.DataFrame(rows)
    ev_df_disk.to_csv(ev_dir / "events.csv", index=False)

    # 1. Verify cross-subject wait grouping does not merge sub-01 and sub-02 (mean full wait = 2.0s, NOT > 1000s)
    wait_rows = compute_word_horizon_physical_waits(
        ev_df_disk, split="test", horizons=(0, 1, 2, "full"), window_seconds=1.0, common_horizon_delay=2
    )
    by_h = {r["horizon"]: r for r in wait_rows}
    assert by_h["r=0"]["query_count"] == 4
    assert by_h["r=0"]["mean_wait_windows"] == pytest.approx(1.0)
    assert by_h["r=0"]["mean_wait_seconds"] == pytest.approx(1.0)
    assert by_h["full"]["mean_wait_windows"] == pytest.approx(3.5)
    assert by_h["full"]["mean_wait_seconds"] == pytest.approx(2.05)

    # 2. Verify k2:: multi-subject group_id enrichment works even without event_ids/event_indices
    ms_segs = [
        {
            "segment_index": 0,
            "group_id": "k2::pallier2025|sub-01|run-06::pallier2025|run-06|sequence-000",
            "subject_id": "multi_subject_k2",
            "recording_id": "pallier2025|sub-01|run-06",
            "reference_words": ["le", "petit", "prince", "parle"],
        }
    ]
    enrich_segments_from_event_table(ms_segs, ev_df_disk, split="test")
    assert ms_segs[0]["onset_seconds"] == pytest.approx([10.0, 10.3, 10.7, 11.2])
    assert ms_segs[0]["pos_tags"] == ["DET", "ADJ", "NC", "V"]
    assert ms_segs[0]["is_content_word_mask"] == [False, True, True, True]

    # 3. Verify run_manuscript_study_exports discovers $BRAINDECODING_CACHE_DIR/pallier2025/events/events.csv automatically
    # with an UN-ENRICHED .pt and an UN-ENRICHED sequence_decoding_test.json
    vocab = ["le", "petit", "prince", "parle"]
    vocab_emb = torch.eye(4, dtype=torch.float32)
    ref_words = ["le", "petit", "prince", "parle", "le", "petit", "prince", "parle"]
    subj_ids = ["sub-01"] * 4 + ["sub-02"] * 4
    pt_segments = [
        {
            "segment_index": 0,
            "group_id": "pallier2025|sub-01|run-06::pallier2025|run-06|sequence-000",
            "subject_id": "sub-01",
            "recording_id": "pallier2025|sub-01|run-06",
            "event_indices": [0, 1, 2, 3],
            "reference_words": ["le", "petit", "prince", "parle"],
        },
        {
            "segment_index": 1,
            "group_id": "pallier2025|sub-02|run-06::pallier2025|run-06|sequence-000",
            "subject_id": "sub-02",
            "recording_id": "pallier2025|sub-02|run-06",
            "event_indices": [4, 5, 6, 7],
            "reference_words": ["le", "petit", "prince", "parle"],
        },
    ]
    k2_scores = torch.cat([torch.eye(4, dtype=torch.float32), torch.eye(4, dtype=torch.float32)], dim=0) * 3.0
    s0_scores = torch.zeros((8, 4), dtype=torch.float32)
    pt_path = tmp_path / "unenriched.pt"
    torch.save(
        {
            "dataset": "pallier2025",
            "language": "fr",
            "split": "test",
            "vocabulary": vocab,
            "vocabulary_embeddings": vocab_emb,
            "vocabulary_train_counts": {w: 10 for w in vocab},
            "reference_words": ref_words,
            "subject_ids": subj_ids,
            "segments": pt_segments,
            "clean_scores": {"S0": s0_scores, "K0": s0_scores, "K1": k2_scores * 0.5, "K2": k2_scores},
            "shifted_scores": {"K0": s0_scores, "K1": s0_scores, "K2": s0_scores},
        },
        pt_path,
    )
    dec_path = tmp_path / "unenriched_dec.json"
    dec_path.write_text(
        json.dumps(
            {
                "decoded_bundles": {
                    "S3_clean": {
                        "final_sentences": [
                            {
                                "segment_index": 0,
                                "group_id": "pallier2025|sub-01|run-06::pallier2025|run-06|sequence-000",
                                "subject_id": "sub-01",
                                "recording_id": "pallier2025|sub-01|run-06",
                                "reference_words": ["le", "petit", "prince", "parle"],
                                "predicted_words": ["le", "petit", "prince", "parle"],
                            },
                            {
                                "segment_index": 1,
                                "group_id": "pallier2025|sub-02|run-06::pallier2025|run-06|sequence-000",
                                "subject_id": "sub-02",
                                "recording_id": "pallier2025|sub-02|run-06",
                                "reference_words": ["le", "petit", "prince", "parle"],
                                "predicted_words": ["le", "petit", "prince", "parle"],
                            },
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    out_auto = tmp_path / "auto_out"
    manifest = run_manuscript_study_exports(
        dataset="pallier2025",
        split="test",
        output_dir=out_auto,
        repo_root=tmp_path / "empty_repo",
        export_pt=pt_path,
        decoding_json=dec_path,
        n_permutations=15,
    )
    exp = manifest["exported_files"]
    assert Path(exp["physical_wait_csv"]).is_file()
    assert Path(exp["duration_prior_alignment_csv"]).is_file()
    assert Path(exp["pos_four_way_csv"]).is_file()
    abl_df = pd.read_csv(exp["brain_only_vs_lm_ablation_csv"])
    assert float(abl_df["bertscore_f1_pct"].iloc[0]) > 0.0
    assert pd.notna(abl_df["mean_physical_wait_seconds"].iloc[0])


def test_t5_diag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """端到端验证 T5-Large 5 模型脉冲对照、Δ_JS 先验对齐、1.00–2.50s 长窗对照、缓存命中与非覆盖 CLI 导出。"""
    import copy

    import braindecoding.data.pallier2025_dataset as pallier_ds_mod
    from braindecoding.data.text import save_text_embedding_cache
    from braindecoding.evaluation.manuscript_study_exporter import (
        CANONICAL_PALLIER2025_FIVE_MODEL_SPECS,
        evaluate_five_model_t5_diagnostics,
    )
    from braindecoding.models import build_brain_embedding_model
    from braindecoding.tasks.word_decoding.pallier2025 import train as pallier_train
    from braindecoding.training.workflow_runner import build_workflow_config

    cache_dir = tmp_path / "c"
    results_dir = tmp_path / "r"
    ds_cache_dir = cache_dir / "pallier2025"
    ev_dir = ds_cache_dir / "events"
    meg_dir = ds_cache_dir / "meg"
    text_path = ds_cache_dir / "text" / "t5_large_layer_0_5" / "embeddings.npz"
    ev_dir.mkdir(parents=True)
    text_path.parent.mkdir(parents=True)

    monkeypatch.setenv("BRAINDECODING_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(
        pallier_ds_mod,
        "channel_names_sha256",
        lambda names: pallier_ds_mod.CHANNEL_NAMES_SHA256,
    )
    monkeypatch.setattr(
        pallier_ds_mod,
        "vectorview_channel_positions",
        lambda names, path=None: np.zeros((len(names), 2), dtype=np.float32),
    )

    all_vocab50 = list(pallier_train.frozen_vocabulary())
    vocab_words = list(all_vocab50[:6])
    all_embs = np.zeros((len(all_vocab50), 1024), dtype=np.float32)
    for idx in range(len(all_vocab50)):
        all_embs[idx, idx] = 1.0
    text_cfg = {
        "model_name": "t5-large",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "embedding_dimension": 1024,
        "word_normalization": "canonical_standard_word",
    }
    save_text_embedding_cache(text_path, all_vocab50, all_embs, text_cfg)

    channel_names = [f"MEG{index:04d}" for index in range(pallier_ds_mod.MEG_CHANNEL_COUNT)]
    rng = np.random.default_rng(12345)
    ev_rows: list[dict[str, object]] = []
    pos_tags = ["DET", "NC", "V", "ADJ", "ADV", "PRE"]
    durs_s = [0.15, 0.35, 0.45, 0.55, 0.65, 0.25]
    onsets_s = [2.0, 2.4, 2.9, 3.5, 4.2]

    for subj in ("sub-01", "sub-02"):
        for split_name, run_name in (("train", "run-01"), ("test", "run-06")):
            rec_id = f"pallier2025|{subj}|{run_name}"
            sig_p, side_p = pallier_ds_mod._signal_product_paths(meg_dir, subj, run_name)
            sig_p.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                sig_p,
                rng.normal(0.0, 0.5, size=(pallier_ds_mod.MEG_CHANNEL_COUNT, 800)).astype(
                    np.float32
                ),
            )
            side_p.write_text(
                json.dumps(
                    {
                        "recording_id": rec_id,
                        "source_first_samp": 1000,
                        "source_sampling_rate_hz": 1000.0,
                        "output_sampling_rate_hz": 50.0,
                        "channel_count": pallier_ds_mod.MEG_CHANNEL_COUNT,
                        "channel_names": channel_names,
                        "channel_names_sha256": pallier_ds_mod.CHANNEL_NAMES_SHA256,
                        "dtype": "float32",
                    }
                ),
                encoding="utf-8",
            )
            for idx in range(5):
                w_idx = (idx + (0 if split_name == "train" else 1)) % 6
                w = vocab_words[w_idx]
                st = onsets_s[idx]
                en = st + durs_s[w_idx]
                ev_rows.append(
                    {
                        "事件编号": f"{rec_id}|{split_name}-{idx}",
                        "受试者": subj,
                        "记录编号": rec_id,
                        "运行编号": run_name,
                        "材料编号": f"mat-{split_name}",
                        "划分单元": f"unit-{split_name}",
                        "记录内序号": idx,
                        "词": w,
                        "标准词": w,
                        "开始时间": st,
                        "结束时间": en,
                        "上下文编号": f"pallier2025|{run_name}|sequence-000",
                        "数据划分": split_name,
                        "是否可训练": True,
                        "排除原因": "",
                        "词性": pos_tags[w_idx],
                        "是否内容词": w_idx >= 1,
                    }
                )
    ev_df = pd.DataFrame(ev_rows)
    ev_csv_path = ev_dir / "events.csv"
    ev_df.to_csv(ev_csv_path, index=False)

    # Identify ordered common test event_ids for cached subspace_encoded_cache_test.pt
    test_ev = ev_df[ev_df["数据划分"] == "test"].copy()
    test_ev_ids = test_ev["事件编号"].astype(str).tolist()
    test_words_all = test_ev["标准词"].astype(str).tolist()
    common_mask = test_ev["记录内序号"].to_numpy() <= 2
    common_eids = [e for e, m in zip(test_ev_ids, common_mask) if m]
    common_words_lst = [w for w, m in zip(test_words_all, common_mask) if m]

    def _tiny_model_cfg(stage_name: str, cfg_path_rel: str) -> dict[str, object]:
        cfg_m, _ = build_workflow_config(
            "pallier2025",
            stage=stage_name,
            seed=0,
            subject="all",
            config_path=cfg_path_rel,
            mode="eval",
        )
        m_cfg = copy.deepcopy(cfg_m["model"])
        m_cfg["conv"].update(
            {
                "merger_channels": 8,
                "merger_position_dimension": 32,
                "initial_linear": 16,
                "hidden": 16,
                "depth": 1,
                "temporal_attention_hidden": 16,
            }
        )
        m_cfg["transformer"].update({"depth": 1, "heads": 2})
        if stage_name == "prefix_revision":
            lwr = dict(m_cfg.get("local_window_revision") or {})
            lwr["validation_query_batch_size"] = 8
            lwr["horizons"] = [0, 1, 2]
            m_cfg["local_window_revision"] = lwr
        return m_cfg

    def _mk_seed_summary_entry(
        seed_val: int, *, stage_name: str, c50_10: float, s50_10: float, fv_10: float
    ) -> dict[str, object]:
        if stage_name in ("revision", "prefix_revision"):
            clean_hz = {
                f"K{h}": {
                    "common_vocab50_macro_top10": c50_10 - 0.02 * (2 - h),
                    "common_vocab50_macro_top1": 0.12,
                    "common_full_vocab_word_macro_top10": fv_10 - 0.01 * (2 - h),
                    "common_full_vocab_subject_macro_top10": fv_10 - 0.02 * (2 - h),
                }
                for h in (0, 1, 2)
            }
            sh_hz = {
                f"K{h}": {
                    "mean_common_vocab50_macro_top10": s50_10,
                    "mean_common_vocab50_macro_top1": 0.05,
                    "mean_common_full_vocab_word_macro_top10": s50_10 * 0.5,
                    "mean_common_full_vocab_subject_macro_top10": s50_10 * 0.5,
                }
                for h in (0, 1, 2)
            }
            return {
                "seed": seed_val,
                "vocabulary_sweep": {"full_vocab": {"candidate_count": 6}},
                "negative_controls": {
                    "methods": {
                        "raw": {
                            "clean_reference": {"horizons": clean_hz},
                            "shifted_aggregate": {"horizons": sh_hz},
                        }
                    }
                },
            }
        return {
            "seed": seed_val,
            "vocabulary_sweep": {"full_vocab": {"candidate_count": 6}},
            "negative_controls": {
                "methods": {
                    "raw": {
                        "clean_reference": {
                            "vocab50_macro_top10": c50_10,
                            "vocab50_macro_top1": 0.10,
                            "full_vocab_word_macro_top10": fv_10,
                            "full_vocab_subject_macro_top10": fv_10,
                        },
                        "shifted_aggregate": {
                            "mean_vocab50_macro_top10": s50_10,
                            "mean_vocab50_macro_top1": 0.05,
                            "mean_full_vocab_word_macro_top10": s50_10 * 0.5,
                            "mean_full_vocab_subject_macro_top10": s50_10 * 0.5,
                        },
                    }
                }
            },
        }

    dummy_pos = np.zeros((pallier_ds_mod.MEG_CHANNEL_COUNT, 2), dtype=np.float32)
    base_pred = torch.zeros((len(test_ev_ids), 1024), dtype=torch.float32)
    for row_i, w in enumerate(test_words_all):
        base_pred[row_i, vocab_words.index(w)] = 1.0
    sh_pred = torch.zeros((len(test_ev_ids), 1024), dtype=torch.float32)
    sh_pred[:, 0] = 1.0

    # Save checkpoints + causal_fast_audit summaries + encoded caches for all 5 canonical models
    for idx_sp, sp in enumerate(CANONICAL_PALLIER2025_FIVE_MODEL_SPECS):
        stage_m = str(sp["stage"])
        exp_tag = str(sp["exp_tag"])
        m_cfg = _tiny_model_cfg(stage_m, str(sp["config_path"]))
        seed_entries = []
        for sd in (0, 1):
            torch.manual_seed(100 + idx_sp * 10 + sd)
            tiny_model = build_brain_embedding_model(
                pallier_ds_mod.MEG_CHANNEL_COUNT, dummy_pos, 10, m_cfg
            )
            ckpt_dir = results_dir / "pallier2025" / exp_tag / stage_m / "all-n10" / f"seed-{sd:03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state": tiny_model.state_dict(), "model_config": m_cfg},
                ckpt_dir / "best.ckpt",
            )
            seed_entries.append(
                _mk_seed_summary_entry(
                    sd,
                    stage_name=stage_m,
                    c50_10=0.40 + 0.04 * idx_sp,
                    s50_10=0.20 + (0.25 if "undebiased" in sp["condition_key"] else 0.02),
                    fv_10=0.25 + 0.03 * idx_sp,
                )
            )
            # Save encoded cache with Clean + Shifted so duration_prior_alignment covers all 3 alignments
            enc_dir = (
                results_dir
                / "pallier2025"
                / exp_tag
                / "cfa"
                / stage_m
                / "all-n10"
                / f"seed-{sd:03d}"
                / "exploratory"
            )
            enc_dir.mkdir(parents=True, exist_ok=True)
            if stage_m in ("revision", "prefix_revision"):
                c_enc = {
                    "event_ids": list(common_eids),
                    "words": list(common_words_lst),
                    "revision_predictions": {
                        h: base_pred[common_mask].clone() for h in (0, 1, 2)
                    },
                }
                s_enc = {
                    "event_ids": list(common_eids),
                    "words": list(common_words_lst),
                    "revision_predictions": {
                        h: sh_pred[common_mask].clone() for h in (0, 1, 2)
                    },
                }
            else:
                c_enc = {
                    "event_ids": list(test_ev_ids),
                    "words": list(test_words_all),
                    "predictions": base_pred.clone(),
                }
                s_enc = {
                    "event_ids": list(test_ev_ids),
                    "words": list(test_words_all),
                    "predictions": sh_pred.clone(),
                }
            torch.save(
                {
                    "clean_encoded": c_enc,
                    "shifted_encoded_by_shift": [(-10.0, "preroll", None, s_enc)],
                },
                enc_dir / "subspace_encoded_cache_test.pt",
            )

        ms_dir = results_dir / "pallier2025" / exp_tag / "cfa" / stage_m / "all-n10"
        ms_dir.mkdir(parents=True, exist_ok=True)
        (ms_dir / "multi_seed_evaluation_test.json").write_text(
            json.dumps({"seed_results": seed_entries}), encoding="utf-8"
        )

    # Save cfa summaries for 1.28 / 1.50 / 2.50s isolated long-window Stage-1 controls
    for lw_idx, lw_tag in enumerate(
        ("lr4e-4_bs256_win1p28", "lr4e-4_bs256_win1p50", "lr4e-4_bs256_win2p50"), start=1
    ):
        lw_ms_dir = results_dir / "pallier2025" / lw_tag / "cfa" / "word" / "all-n10"
        lw_ms_dir.mkdir(parents=True, exist_ok=True)
        lw_ms_dir.joinpath("multi_seed_evaluation_test.json").write_text(
            json.dumps(
                {
                    "seed_results": [
                        _mk_seed_summary_entry(
                            sd,
                            stage_name="word",
                            c50_10=0.41 + 0.01 * lw_idx,
                            s50_10=0.20,
                            fv_10=0.24 + 0.01 * lw_idx,
                        )
                        for sd in (0, 1)
                    ]
                }
            ),
            encoding="utf-8",
        )

    diag_out = tmp_path / "d"
    diag_res = evaluate_five_model_t5_diagnostics(
        results_dir=results_dir,
        output_dir=diag_out,
        dataset="pallier2025",
        split="test",
        seeds=(0, 1),
        eval_subdir="cfa",
        cache_dir=cache_dir,
        device="cpu",
        n_permutations=10,
    )
    diag_files = diag_res["exported_files"]
    for req_k in (
        "five_model_debiasing_ledger_csv",
        "synthetic_timing_csv",
        "duration_prior_alignment_csv",
        "word_top10_neighborhoods_json",
        "word_top10_neighborhoods_csv",
        "five_model_diagnostics_json",
    ):
        assert Path(diag_files[req_k]).is_file(), f"Missing {req_k}"

    ledger_df = pd.read_csv(diag_files["five_model_debiasing_ledger_csv"])
    assert set(ledger_df["condition_key"]) == {
        "S0",
        "F_joint_undebiased",
        "F_freeze_undebiased",
        "F_debiased",
        "K2",
    }
    assert (ledger_df["text_embedding_model"] == "t5-large").all()

    synth_df = pd.read_csv(diag_files["synthetic_timing_csv"])
    assert set(synth_df["condition_key"]) == {
        "S0",
        "F_joint_undebiased",
        "F_freeze_undebiased",
        "F_debiased",
        "K0",
        "K1",
        "K2",
    }
    assert {"clean", "shifted", "shared", "independent"} <= set(synth_df["condition"])

    js_df = pd.read_csv(diag_files["duration_prior_alignment_csv"])
    assert set(js_df["alignment"]) == {"clean", "shifted", "synthetic_shared"}
    assert "p_holm" in js_df.columns
    assert len(js_df[js_df["alignment"] == "clean"]) == 7

    nb_obj = json.loads(Path(diag_files["word_top10_neighborhoods_json"]).read_text(encoding="utf-8"))
    assert nb_obj["text_embedding_model"] == "t5-large"
    assert len(nb_obj["examples"]) >= 1
    first_nb = nb_obj["examples"][0]
    assert "target_mean_duration_ms" in first_nb
    assert "F_joint_undebiased" in first_nb["conditions"]
    assert "K2" in first_nb["conditions"]
    nb_csv_df = pd.read_csv(diag_files["word_top10_neighborhoods_csv"])
    for col in ("target_mean_duration_ms", "candidate_mean_duration_ms", "cosine_to_target"):
        assert col in nb_csv_df.columns

    # Cache hit check when force=False
    cached_diag = evaluate_five_model_t5_diagnostics(
        results_dir=results_dir,
        output_dir=diag_out,
        dataset="pallier2025",
        split="test",
        seeds=(0, 1),
        eval_subdir="cfa",
        cache_dir=cache_dir,
        device="cpu",
        n_permutations=10,
        force=False,
    )
    assert cached_diag["common_query_count"] == diag_res["common_query_count"]

    # CLI archive run reusing --five-model-diagnostics-json + --collect-long-window-results alongside --export-pt
    s_dir = tmp_path / "s"
    s_dir.mkdir(parents=True, exist_ok=True)
    arts = _build_fixture_artifacts(s_dir)
    archive_dir = tmp_path / "a"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "export_manuscript_studies.py"),
        "--dataset",
        "pallier2025",
        "--split",
        "test",
        "--output-dir",
        str(archive_dir),
        "--results-dir",
        str(results_dir),
        "--eval-subdir",
        "cfa",
        "--five-model-diagnostics-json",
        diag_files["five_model_diagnostics_json"],
        "--collect-long-window-results",
        "--export-pt",
        str(arts["pt_path"]),
        "--decoding-json",
        str(arts["dec_path"]),
        "--event-table-csv",
        str(ev_csv_path),
        "--n-permutations",
        "10",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    cli_res = json.loads(proc.stdout)
    assert cli_res["status"] == "completed"
    cli_files = cli_res["exported_files"]
    lw_df = pd.read_csv(cli_files["long_window_csv"])
    assert set(lw_df["condition"]) == {
        "word_1p00s",
        "word_1p28s",
        "word_1p50s",
        "word_2p50s",
        "local_r0",
        "local_r1",
        "local_r2",
        "F_debiased",
    }
    assert (lw_df["text_embedding_model"] == "t5-large").all()

    # Ensure single-model export_pt did NOT clobber 5-model neighborhood/JS files
    archived_nb = json.loads(
        (archive_dir / "word_top10_neighborhoods.json").read_text(encoding="utf-8")
    )
    assert "F_joint_undebiased" in archived_nb["examples"][0]["conditions"]
    archived_js = pd.read_csv(archive_dir / "duration_prior_alignment.csv")
    assert set(archived_js["alignment"]) == {"clean", "shifted", "synthetic_shared"}
    assert Path(cli_files["sentence_neural_top5_json"]).is_file()
    assert Path(cli_files["brain_only_vs_lm_ablation_csv"]).is_file()
