"""本地小样本联调与 Google Colab 双模工作流单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from braindecoding import cli
from braindecoding.config import (
    PROJECT_ROOT,
    cache_root,
    data_root,
    ensure_default_environment,
    project_path,
    results_root,
)
from braindecoding.training.workflow_runner import (
    build_workflow_config,
    normalize_workflow_stage,
    plot_evaluation_figures,
    plot_stage_comparison_figure,
    resolve_subject_subset,
    write_subjects_manifest,
)
from scripts import cache_archive, colab_setup


def test_environment_variable_decoupling(tmp_path, monkeypatch):
    custom_data = tmp_path / "studies_links"
    custom_cache = tmp_path / "local_cache"
    custom_results = tmp_path / "runs"
    monkeypatch.setenv("BRAINDECODING_DATA_DIR", str(custom_data))
    monkeypatch.setenv("BRAINDECODING_CACHE_DIR", str(custom_cache))
    monkeypatch.setenv("BRAINDECODING_RESULTS_DIR", str(custom_results))
    monkeypatch.delenv("BRAINDATA_ROOT", raising=False)

    ensure_default_environment()
    assert data_root() == custom_data.resolve()
    assert cache_root() == custom_cache.resolve()
    assert results_root() == custom_results.resolve()
    assert (
        project_path("data/cache/derived/chineseeeg2_littleprince/events/event_table.parquet")
        == (custom_cache / "chineseeeg2_littleprince/events/event_table.parquet").resolve()
    )


def test_flexible_subject_subset_resolution():
    all_spec = resolve_subject_subset("chineseeeg2_littleprince", subject="all")
    assert all_spec["is_full_dataset"] is True
    assert all_spec["subject_count"] == 8
    assert all_spec["subset_tag"] == "all-n08"

    n1_spec = resolve_subject_subset("chineseeeg2_littleprince", num_subjects=1)
    assert n1_spec["is_full_dataset"] is False
    assert n1_spec["subjects"] == ["sub-01"]
    assert n1_spec["subset_tag"] == "sub-01"

    explicit_spec = resolve_subject_subset(
        "pallier2025", subjects="sub-03,sub-01,2"
    )
    assert explicit_spec["subjects"] == ["sub-01", "sub-02", "sub-03"]
    assert explicit_spec["subset_tag"].startswith("subset-n03-")

    with pytest.raises(ValueError, match="互斥"):
        resolve_subject_subset(
            "chineseeeg2_littleprince", num_subjects=1, subjects="sub-01,sub-02"
        )


def test_subjects_json_isolation_prevents_collision(tmp_path):
    spec1 = resolve_subject_subset("chineseeeg2_littleprince", num_subjects=1)
    spec2 = resolve_subject_subset("chineseeeg2_littleprince", num_subjects=2)

    out_dir = tmp_path / "run_dir"
    subjects_path = write_subjects_manifest(
        out_dir, spec1, stage="word", mode="smoke", seed=0
    )
    saved = json.loads(subjects_path.read_text(encoding="utf-8"))
    assert saved["subjects"] == ["sub-01"]

    with pytest.raises(ValueError, match="拒绝混用目录"):
        write_subjects_manifest(
            out_dir, spec2, stage="word", mode="smoke", seed=0
        )


def test_workflow_config_assigns_isolated_output_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINDECODING_RESULTS_DIR", str(tmp_path / "runs"))
    cfg_sub1, _ = build_workflow_config(
        "chineseeeg2_littleprince",
        stage="word",
        seed=0,
        num_subjects=1,
        precision="bf16-mixed",
        mode="smoke",
    )
    cfg_sub2, _ = build_workflow_config(
        "chineseeeg2_littleprince",
        stage="word",
        seed=0,
        num_subjects=2,
        precision="bf16-mixed",
        mode="smoke",
    )
    assert cfg_sub1["training"]["output_dir"] != cfg_sub2["training"]["output_dir"]
    assert cfg_sub1["training"]["amp_dtype"] == "bfloat16"
    assert cfg_sub1["training"]["precision"] == "bf16-mixed"


def test_cache_archive_pack_and_unpack_roundtrip(tmp_path):
    src_cache = tmp_path / "local_cache"
    ds_dir = src_cache / "chineseeeg2_littleprince" / "signals"
    ds_dir.mkdir(parents=True)
    sample_file = ds_dir / "sub-01.npy"
    sample_file.write_bytes(b"fake-npy-payload-12345")
    sidecar_file = ds_dir / "sub-01.json"
    sidecar_file.write_text('{"recording_id": "sub-01"}', encoding="utf-8")

    archive_dir = tmp_path / "archives"
    pack_summary = cache_archive.pack_cache(
        cache_dir=src_cache, archive_dir=archive_dir, modality="eeg"
    )
    tar_path = Path(pack_summary["archive"])
    assert tar_path.is_file()
    assert pack_summary["file_count"] == 2

    dst_cache = tmp_path / "unpacked_cache"
    unpack_summary = cache_archive.unpack_cache(
        archive=tar_path, cache_dir=dst_cache, modality="eeg", skip_existing=True
    )
    assert unpack_summary["extracted_files"] >= 2
    assert (
        dst_cache / "chineseeeg2_littleprince" / "signals" / "sub-01.npy"
    ).read_bytes() == b"fake-npy-payload-12345"

    skip_summary = cache_archive.unpack_cache(
        archive=tar_path, cache_dir=dst_cache, modality="eeg", skip_existing=True
    )
    assert skip_summary["skipped_files"] >= 2


def test_colab_setup_constraints_and_drive_symlink_guard(tmp_path):
    venv_dir = tmp_path / "proj-venv"
    constraints = colab_setup.export_runtime_constraints(venv_dir)
    assert constraints.is_file()
    content = constraints.read_text(encoding="utf-8")
    assert "torch==" in content

    with pytest.raises(ValueError, match="Errno 95"):
        colab_setup.setup_studies_links(
            tmp_path, links_dir="/content/drive/MyDrive/forbidden_links"
        )


def test_colab_notebooks_follow_streaming_and_symlink_rules():
    notebooks = [
        PROJECT_ROOT / "scripts" / "colab" / "colab_drive_init.ipynb",
        PROJECT_ROOT / "scripts" / "colab" / "colab_cpu_cache.ipynb",
        PROJECT_ROOT / "scripts" / "colab" / "colab_gpu_train.ipynb",
        PROJECT_ROOT / "scripts" / "colab" / "colab_gpu_evaluation.ipynb",
    ]
    for nb_path in notebooks:
        assert nb_path.is_file(), f"缺少 Notebook: {nb_path}"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        code_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "code"
        )
        assert "subprocess.run(" not in code_text
        assert "/content/studies_links" in code_text
        assert "/content/local_cache" in code_text
        assert "/content/proj-venv" in code_text
        assert "BRAINDECODING_DATA_DIR" in code_text
        assert "BRAINDECODING_CACHE_DIR" in code_text
        assert "BRAINDECODING_RESULTS_DIR" in code_text


def test_cli_parses_smoke_cache_train_and_eval_run_subcommands():
    parser = cli.build_parser()
    smoke_args = parser.parse_args(
        [
            "smoke",
            "--dataset",
            "chineseeeg2_littleprince",
            "--stage",
            "2.1",
            "--num-subjects",
            "1",
            "--epochs",
            "2",
            "--precision",
            "bf16-mixed",
            "--require-cache",
        ]
    )
    assert smoke_args.command == "smoke"
    assert normalize_workflow_stage(smoke_args.stage) == "context"
    assert smoke_args.num_subjects == 1
    assert smoke_args.epochs == 2
    assert smoke_args.precision == "bf16-mixed"
    assert smoke_args.require_cache is True

    train_args = parser.parse_args(
        [
            "train",
            "--dataset",
            "pallier2025",
            "--stage",
            "context",
            "--subjects",
            "sub-01,sub-02",
            "--batch-size",
            "256",
            "--eval-batch-size",
            "512",
            "--lr",
            "0.0004",
            "--num-workers",
            "0",
            "--require-cache",
            "--resume",
            "auto",
        ]
    )
    assert train_args.command == "train"
    assert train_args.stage == "context"
    assert train_args.subjects == "sub-01,sub-02"
    assert train_args.batch_size == 256
    assert train_args.eval_batch_size == 512
    assert train_args.learning_rate == pytest.approx(0.0004)
    assert train_args.num_workers == 0
    assert train_args.resume == "auto"

    eval_args = parser.parse_args(
        [
            "eval-run",
            "--dataset",
            "pallier2025",
            "--split",
            "test",
            "--all-stages",
            "--seed",
            "0",
            "--require-cache",
        ]
    )
    assert eval_args.command == "eval-run"
    assert eval_args.split == "test"
    assert eval_args.all_stages is True
    assert eval_args.require_cache is True


def test_workflow_config_loads_all_six_stage_yaml_files():
    for expected_ds, expected_stage, rel_yaml in (
        ("pallier2025", "word", "configs/training/pallier2025_word_encoder.yaml"),
        ("pallier2025", "context", "configs/training/pallier2025_context_transformer.yaml"),
        ("pallier2025", "revision", "configs/training/pallier2025_rolling_revision.yaml"),
        ("chineseeeg2_littleprince", "word", "configs/training/chineseeeg2_word_encoder.yaml"),
        ("chineseeeg2_littleprince", "context", "configs/training/chineseeeg2_context_transformer.yaml"),
        ("chineseeeg2_littleprince", "revision", "configs/training/chineseeeg2_rolling_revision.yaml"),
    ):
        cfg, subset = build_workflow_config(
            config_path=rel_yaml, seed=1, mode="train"
        )
        assert subset["dataset"] == expected_ds
        assert cfg["workflow"]["stage"] == expected_stage
        assert cfg["workflow"]["config_path"] == rel_yaml
        assert int(cfg["training"]["batch_size"]) > 0
        assert int(cfg["training"]["eval_batch_size"]) >= int(cfg["training"]["batch_size"])
        assert float(cfg["training"]["learning_rate"]) > 0.0
        assert cfg["training"]["precision"] == "bf16-mixed"
        assert cfg["model"]["use_transformer"] is (expected_stage in ("context", "revision"))
        if expected_ds == "pallier2025":
            assert "t5_large_layer_0_5/embeddings.npz" in cfg["cache"]["text_embeddings"].replace("\\", "/")
            assert cfg["text_embedding"]["model_name"] == "t5-large"
            assert cfg["text_embedding"]["layer_fraction"] == pytest.approx(0.5)
            assert "revision" not in cfg["text_embedding"]
            cfg_no_explicit, _ = build_workflow_config(
                expected_ds, stage=expected_stage, seed=1, mode="train"
            )
            assert cfg_no_explicit["text_embedding"]["model_name"] == "t5-large"
            assert "t5_large_layer_0_5/embeddings.npz" in cfg_no_explicit["cache"]["text_embeddings"].replace("\\", "/")
        if expected_stage == "word":
            assert "local_window_revision" not in cfg["model"]
            assert "pretrained_brain_encoder_checkpoint" not in cfg["training"]
        elif expected_stage == "context":
            assert "local_window_revision" not in cfg["model"]
            assert cfg["training"]["freeze_brain_encoder_epochs"] >= 0
            assert cfg["training"]["preserve_complete_groups"] is True
            assert "pretrained_brain_encoder_checkpoint" in cfg["training"]
            assert "revision_cutoff_policy" not in cfg["training"]
        elif expected_stage == "revision":
            assert cfg["model"]["local_window_revision"]["enabled"] is True
            assert cfg["training"]["freeze_brain_encoder_epochs"] >= 0
            assert cfg["training"]["preserve_complete_groups"] is True
            assert cfg["training"]["revision_cutoff_policy"] == "all"
            assert "pretrained_brain_encoder_checkpoint" in cfg["training"]


def test_word_exp_tag_cross_experiment_warmstart(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("BRAINDECODING_RESULTS_DIR", str(runs_dir))
    monkeypatch.setenv("BRAINDECODING_EXP_TAG", "ctx_lr2e-4_bs256")
    monkeypatch.setenv("BRAINDECODING_WORD_EXP_TAG", "word_lr1e-4_bs128")

    cfg, _ = build_workflow_config(
        "pallier2025",
        stage="context",
        seed=0,
        mode="train",
    )
    out_dir = Path(cfg["training"]["output_dir"]).as_posix()
    warm_ckpt = Path(cfg["training"]["pretrained_brain_encoder_checkpoint"]).as_posix()
    assert out_dir.endswith("runs/pallier2025/ctx_lr2e-4_bs256/context/all-n10/seed-000")
    assert warm_ckpt.endswith("runs/pallier2025/word_lr1e-4_bs128/word/all-n10/seed-000/best.ckpt")


def test_plot_evaluation_and_stage_comparison_figures(tmp_path):
    dummy_sweep = {
        "N20": {"label": "Top-20 Vocab", "macro_top10": 0.72, "micro_top10": 0.75, "macro_top5": 0.52, "macro_top1": 0.24, "mrr": 0.41, "chance_top10": 0.50},
        "N50": {"label": "Top-50 Vocab", "macro_top10": 0.45, "micro_top10": 0.48, "macro_top5": 0.30, "macro_top1": 0.12, "mrr": 0.25, "chance_top10": 0.20},
        "N100": {"label": "Top-100 Vocab", "macro_top10": 0.29, "micro_top10": 0.32, "macro_top5": 0.18, "macro_top1": 0.07, "mrr": 0.16, "chance_top10": 0.10},
        "N150": {"label": "Top-150 Vocab", "macro_top10": 0.22, "micro_top10": 0.25, "macro_top5": 0.13, "macro_top1": 0.05, "mrr": 0.12, "chance_top10": 10.0 / 150.0},
        "full_vocab": {"label": "Full Vocab (2020)", "macro_top10": 0.06, "micro_top10": 0.09, "macro_top5": 0.03, "macro_top1": 0.01, "mrr": 0.03, "chance_top10": 10.0 / 2020.0},
    }
    eval_res = {
        "dataset": "pallier2025",
        "stage": "context",
        "split": "val",
        "seed": 0,
        "subset": {"subset_tag": "all-n10", "subject_count": 10, "total_canonical_subjects": 10},
        "checkpoint": {"path": "best.ckpt", "epoch": 12},
        "metrics": {
            "vocabulary_sweep": dummy_sweep,
            "per_subject": [
                {"subject_id": "sub-01", "macro_top10": 0.46, "macro_top1": 0.13},
                {"subject_id": "sub-02", "macro_top10": 0.44, "macro_top1": 0.11},
            ],
            "revision_horizons": {},
        },
        "training_history": [
            {"epoch": 1, "train_loss": 5.5, "selection_score_legacy50": 0.28, "selection_score_full_vocab": 0.02},
            {"epoch": 2, "train_loss": 4.8, "selection_score_legacy50": 0.45, "selection_score_full_vocab": 0.06},
        ],
    }
    figs = plot_evaluation_figures(eval_res, tmp_path / "figures")
    assert Path(figs["overview_png"]).is_file()
    assert Path(figs["overview_pdf"]).is_file()

    comp_figs = plot_stage_comparison_figure(
        {"word": eval_res, "context": eval_res, "revision": eval_res},
        tmp_path / "figures",
        split="val",
    )
    assert Path(comp_figs["stage_comparison_png"]).is_file()
    assert Path(comp_figs["stage_comparison_pdf"]).is_file()


def test_decoupled_control_configs_and_temporal_shift_specification():
    from braindecoding.training.workflow_runner import (
        _extract_control_summary_row,
        _resolve_control_specification,
        _summarize_shifted_conditions,
    )

    temporal_shift_files = [
        "configs/controls/temporal_shift/temporal_shift_control.yaml",
        "configs/controls/temporal_shift/pallier2025_temporal_shift_control.yaml",
        "configs/controls/temporal_shift/chineseeeg2_temporal_shift_control.yaml",
    ]
    for rel_path in temporal_shift_files:
        full_p = PROJECT_ROOT / rel_path
        assert full_p.is_file(), f"缺少负对照配置: {rel_path}"
        raw_text = full_p.read_text(encoding="utf-8")
        assert "word_decoding" not in raw_text.replace("configs/word_decoding/", "")

    cfg_pal, sub_pal = build_workflow_config(
        config_path="configs/controls/temporal_shift/pallier2025_temporal_shift_control.yaml",
        stage="revision",
        seed=0,
    )
    assert sub_pal["dataset"] == "pallier2025"
    assert cfg_pal["workflow"]["stage"] == "revision"
    spec_pal = _resolve_control_specification(cfg_pal)
    assert spec_pal["enabled"] is True
    assert spec_pal["control_type"] == "temporal_shift"
    assert spec_pal["shift_seconds"] == [-120.0, -60.0, -30.0, 30.0, 60.0, 120.0]
    assert spec_pal["same_subject"] is True
    assert spec_pal["same_recording"] is True

    cfg_cn, sub_cn = build_workflow_config(
        "chineseeeg2_littleprince", stage="word", seed=0
    )
    assert sub_cn["dataset"] == "chineseeeg2_littleprince"
    spec_cn = _resolve_control_specification(
        cfg_cn,
        control_config_path="configs/controls/temporal_shift/temporal_shift_control.yaml",
    )
    assert spec_cn["enabled"] is True
    assert spec_cn["shift_seconds"] == [-120.0, -60.0, -30.0, 30.0, 60.0, 120.0]

    spec_zero = _resolve_control_specification(
        cfg_cn,
        control_config_path="configs/controls/zero_signal/zero_signal_control.yaml",
    )
    assert spec_zero["enabled"] is True
    assert spec_zero["control_type"] == "zero_signal"
    assert spec_zero["include_zero_signal"] is True
    assert spec_zero["shift_seconds"] == []

    parser = cli.build_parser()
    eval_ctrl_args = parser.parse_args(
        [
            "eval-run",
            "--dataset",
            "pallier2025",
            "--split",
            "test",
            "--all-stages",
            "--control-config",
            "configs/controls/temporal_shift/temporal_shift_control.yaml",
            "--shift-seconds",
            "-120,-60,-30,30,60,120",
        ]
    )
    assert (
        eval_ctrl_args.control_config
        == "configs/controls/temporal_shift/temporal_shift_control.yaml"
    )
    assert eval_ctrl_args.shift_seconds == "-120,-60,-30,30,60,120"

    clean_metrics = {
        "vocabulary_sweep": {
            "N50": {
                "macro_top10": 0.45,
                "micro_top10": 0.48,
                "macro_top5": 0.30,
                "macro_top1": 0.12,
                "mrr": 0.25,
                "median_rank": 11.0,
                "chance_top10": 0.20,
            },
            "full_vocab": {
                "macro_top10": 0.06,
                "micro_top10": 0.09,
                "macro_top5": 0.03,
                "macro_top1": 0.01,
                "mrr": 0.03,
                "median_rank": 350.0,
            },
        },
        "revision_horizons": {
            "K0": {"common_vocab50_macro_top10": 0.40, "common_full_vocab_macro_top10": 0.05},
            "K1": {"common_vocab50_macro_top10": 0.43, "common_full_vocab_macro_top10": 0.055},
            "K2": {"common_vocab50_macro_top10": 0.45, "common_full_vocab_macro_top10": 0.06},
        },
    }
    shifted_metrics = {
        "vocabulary_sweep": {
            "N50": {
                "macro_top10": 0.20,
                "micro_top10": 0.20,
                "macro_top5": 0.10,
                "macro_top1": 0.02,
                "mrr": 0.09,
                "median_rank": 25.5,
                "chance_top10": 0.20,
            },
            "full_vocab": {
                "macro_top10": 0.005,
                "micro_top10": 0.005,
                "macro_top5": 0.002,
                "macro_top1": 0.0005,
                "mrr": 0.003,
                "median_rank": 1010.0,
            },
        },
        "revision_horizons": {
            "K0": {"common_vocab50_macro_top10": 0.20, "common_full_vocab_macro_top10": 0.005},
            "K1": {"common_vocab50_macro_top10": 0.20, "common_full_vocab_macro_top10": 0.005},
            "K2": {"common_vocab50_macro_top10": 0.20, "common_full_vocab_macro_top10": 0.005},
        },
    }
    clean_row = _extract_control_summary_row("clean_0s", "Clean (0s)", 0.0, clean_metrics)
    shifted_rows = [
        _extract_control_summary_row(
            f"shift_{int(s):+d}s",
            f"Shift {int(s):+d}s",
            s,
            shifted_metrics,
            clean_summary=clean_row,
        )
        for s in (-120.0, -60.0, -30.0, 30.0, 60.0, 120.0)
    ]
    agg = _summarize_shifted_conditions(clean_row, shifted_rows)
    assert agg["shift_count"] == 6
    assert agg["mean_vocab50_macro_top10"] == pytest.approx(0.20)
    assert agg["delta_vocab50_macro_top10_vs_clean"] == pytest.approx(-0.25)
    assert agg["clean_minus_shifted_vocab50_macro_top10"] == pytest.approx(0.25)


def test_plot_negative_control_figures(tmp_path):
    from braindecoding.training.workflow_runner import (
        _extract_control_summary_row,
        _summarize_shifted_conditions,
    )

    dummy_sweep = {
        "N20": {"label": "Top-20 Vocab", "macro_top10": 0.72, "micro_top10": 0.75, "macro_top5": 0.52, "macro_top1": 0.24, "mrr": 0.41, "chance_top10": 0.50},
        "N50": {"label": "Top-50 Vocab", "macro_top10": 0.45, "micro_top10": 0.48, "macro_top5": 0.30, "macro_top1": 0.12, "mrr": 0.25, "median_rank": 11.0, "chance_top10": 0.20},
        "full_vocab": {"label": "Full Vocab (2020)", "macro_top10": 0.06, "micro_top10": 0.09, "macro_top5": 0.03, "macro_top1": 0.01, "mrr": 0.03, "median_rank": 350.0, "chance_top10": 10.0 / 2020.0},
    }
    shifted_sweep = {
        "N50": {"label": "Top-50 Vocab", "macro_top10": 0.20, "micro_top10": 0.21, "macro_top5": 0.10, "macro_top1": 0.02, "mrr": 0.09, "median_rank": 25.0, "chance_top10": 0.20},
        "full_vocab": {"label": "Full Vocab (2020)", "macro_top10": 0.005, "micro_top10": 0.005, "macro_top5": 0.002, "macro_top1": 0.0005, "mrr": 0.003, "median_rank": 1010.0, "chance_top10": 10.0 / 2020.0},
    }
    clean_metrics = {"vocabulary_sweep": dummy_sweep, "revision_horizons": {}}
    shift_metrics = {"vocabulary_sweep": shifted_sweep, "revision_horizons": {}}
    clean_row = _extract_control_summary_row("clean_0s", "Clean (0s)", 0.0, clean_metrics)
    conditions = []
    shifted_rows = []
    for s in (-120.0, -60.0, -30.0, 30.0, 60.0, 120.0):
        key = f"shift_{int(s):+d}s"
        label = f"Shift {int(s):+d}s"
        row = _extract_control_summary_row(key, label, s, shift_metrics, clean_summary=clean_row)
        shifted_rows.append(row)
        conditions.append(
            {
                "key": key,
                "label": label,
                "control_type": "temporal_shift",
                "shift_seconds": s,
                "summary": row,
                "metrics": shift_metrics,
            }
        )
    eval_res = {
        "dataset": "pallier2025",
        "stage": "word",
        "split": "test",
        "seed": 0,
        "subset": {"subset_tag": "all-n10", "subject_count": 10, "total_canonical_subjects": 10},
        "checkpoint": {"path": "best.ckpt", "epoch": 12},
        "metrics": {
            "vocabulary_sweep": dummy_sweep,
            "per_subject": [{"subject_id": "sub-01", "macro_top10": 0.45, "macro_top1": 0.12}],
            "revision_horizons": {},
        },
        "negative_controls": {
            "specification": {
                "enabled": True,
                "control_type": "temporal_shift",
                "shift_seconds": [-120.0, -60.0, -30.0, 30.0, 60.0, 120.0],
            },
            "clean_reference": clean_row,
            "conditions": conditions,
            "shifted_aggregate": _summarize_shifted_conditions(clean_row, shifted_rows),
        },
        "training_history": [],
    }
    figs = plot_evaluation_figures(eval_res, tmp_path / "figures")
    assert Path(figs["overview_png"]).is_file()
    assert Path(figs["negative_control_png"]).is_file()
    assert Path(figs["negative_control_pdf"]).is_file()

    comp_figs = plot_stage_comparison_figure(
        {"word": eval_res, "context": eval_res, "revision": eval_res},
        tmp_path / "figures",
        split="test",
    )
    assert Path(comp_figs["stage_comparison_png"]).is_file()
    assert Path(comp_figs["stage_control_comparison_png"]).is_file()
    assert Path(comp_figs["stage_control_comparison_pdf"]).is_file()



