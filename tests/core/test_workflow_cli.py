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
        project_path("data/cache/derived/pallier2025/events/events.csv")
        == (custom_cache / "pallier2025/events/events.csv").resolve()
    )


def test_flexible_subject_subset_resolution(tmp_path, monkeypatch):
    all_spec = resolve_subject_subset("pallier2025", subject="all")
    assert all_spec["is_full_dataset"] is True
    assert all_spec["subject_count"] == 10
    assert all_spec["subset_tag"] == "all-n10"

    n1_spec = resolve_subject_subset("pallier2025", num_subjects=1)
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
            "pallier2025", num_subjects=1, subjects="sub-01,sub-02"
        )

    # Dynamic subject discovery from materialized PNPL events.csv
    custom_cache = tmp_path / "pnpl_cache"
    lb_events = custom_cache / "libribrain100" / "events" / "events.csv"
    lb_events.parent.mkdir(parents=True, exist_ok=True)
    lb_events.write_text("受试者\nsub-0\nsub-0\n", encoding="utf-8")
    ar_events = custom_cache / "armeni2022" / "events" / "events.csv"
    ar_events.parent.mkdir(parents=True, exist_ok=True)
    ar_events.write_text("受试者\nsub-001\nsub-002\nsub-003\n", encoding="utf-8")
    monkeypatch.setenv("BRAINDECODING_CACHE_DIR", str(custom_cache))

    lb_spec = resolve_subject_subset("libribrain100", subject="all")
    assert lb_spec["subjects"] == ["sub-0"]
    assert lb_spec["subset_tag"] == "all-n01"
    ar_spec = resolve_subject_subset("armeni2022", num_subjects=2)
    assert ar_spec["subjects"] == ["sub-001", "sub-002"]
    assert ar_spec["is_full_dataset"] is False


def test_subjects_json_isolation_prevents_collision(tmp_path):
    spec1 = resolve_subject_subset("pallier2025", num_subjects=1)
    spec2 = resolve_subject_subset("pallier2025", num_subjects=2)

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
        "pallier2025",
        stage="word",
        seed=0,
        num_subjects=1,
        precision="bf16-mixed",
        mode="smoke",
    )
    cfg_sub2, _ = build_workflow_config(
        "pallier2025",
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
    ds_dir = src_cache / "pallier2025" / "signals"
    ds_dir.mkdir(parents=True)
    sample_file = ds_dir / "sub-01.npy"
    sample_file.write_bytes(b"fake-npy-payload-12345")
    sidecar_file = ds_dir / "sub-01.json"
    sidecar_file.write_text('{"recording_id": "sub-01"}', encoding="utf-8")

    archive_dir = tmp_path / "archives"
    pack_summary = cache_archive.pack_cache(
        cache_dir=src_cache, archive_dir=archive_dir, modality="meg"
    )
    tar_path = Path(pack_summary["archive"])
    assert tar_path.is_file()
    assert pack_summary["file_count"] == 2

    dst_cache = tmp_path / "unpacked_cache"
    unpack_summary = cache_archive.unpack_cache(
        archive=tar_path, cache_dir=dst_cache, modality="meg", skip_existing=True
    )
    assert unpack_summary["extracted_files"] >= 2
    assert (
        dst_cache / "pallier2025" / "signals" / "sub-01.npy"
    ).read_bytes() == b"fake-npy-payload-12345"

    skip_summary = cache_archive.unpack_cache(
        archive=tar_path, cache_dir=dst_cache, modality="meg", skip_existing=True
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




def test_cli_parses_smoke_cache_train_and_eval_run_subcommands():
    parser = cli.build_parser()
    smoke_args = parser.parse_args(
        [
            "smoke",
            "--dataset",
            "libribrain100",
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


def test_workflow_config_loads_all_pallier_stage_yaml_files():
    for expected_ds, expected_stage, rel_yaml in (
        ("pallier2025", "word", "configs/training/pallier2025_word_encoder.yaml"),
        ("pallier2025", "context", "configs/training/pallier2025_context_transformer.yaml"),
        ("pallier2025", "revision", "configs/training/pallier2025_rolling_revision.yaml"),
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




def test_decoupled_control_configs_and_temporal_shift_specification():
    from braindecoding.training.workflow_runner import (
        _extract_control_summary_row,
        _resolve_control_specification,
        _summarize_shifted_conditions,
    )

    temporal_shift_files = [
        "configs/controls/temporal_shift/temporal_shift_control.yaml",
        "configs/controls/temporal_shift/pallier2025_temporal_shift_control.yaml",
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
    assert spec_pal["shift_seconds"] == [-15.0, -30.0, -45.0, -60.0, -90.0, -120.0]
    assert spec_pal["same_subject"] is True
    assert spec_pal["same_recording"] is True

    cfg_lb, sub_lb = build_workflow_config(
        "libribrain100", stage="word", seed=0
    )
    assert sub_lb["dataset"] == "libribrain100"
    spec_lb = _resolve_control_specification(
        cfg_lb,
        control_config_path="configs/controls/temporal_shift/temporal_shift_control.yaml",
    )
    assert spec_lb["enabled"] is True
    assert spec_lb["shift_seconds"] == [-15.0, -30.0, -45.0, -60.0, -90.0, -120.0]

    spec_zero = _resolve_control_specification(
        cfg_lb,
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




def test_compute_encoded_evaluation_metrics_full_vocab_and_k0_k1_k2_horizons():
    import types
    import numpy as np
    import pandas as pd
    import torch
    from braindecoding.results import load_vocabulary_assets
    from braindecoding.training.workflow_runner import (
        _compute_encoded_evaluation_metrics,
    )

    manifests = load_vocabulary_assets(PROJECT_ROOT / "data/manifests/pallier2025")
    vocab50 = list(manifests[50]["vocabulary"])
    vocab150 = list(manifests[150]["vocabulary"])
    extra_words = [f"extra_word_{i:03d}" for i in range(50)]
    full_vocabulary = sorted(set(vocab150 + extra_words))

    rng = np.random.default_rng(42)
    dim = 64
    embedding_map = {}
    for w in full_vocabulary:
        vec = rng.standard_normal(dim).astype(np.float32)
        vec /= np.linalg.norm(vec)
        embedding_map[w] = vec

    n_events = 24
    event_words = [vocab50[i % 12] for i in range(n_events)]
    event_ids = [f"evt-{i:03d}" for i in range(n_events)]
    subject_ids = ["sub-01" if i < 12 else "sub-02" for i in range(n_events)]

    class _DummyDataset:
        def __init__(self, emb_map, tbl):
            self.embedding_map = emb_map
            self.table = tbl

        def __len__(self):
            return len(self.table)

    dummy_ds = _DummyDataset(
        embedding_map,
        pd.DataFrame({"event_id": event_ids, "subject_id": subject_ids}),
    )
    targets = torch.tensor(
        np.stack([embedding_map[w] for w in event_words]), dtype=torch.float32
    )
    noise_k0 = torch.tensor(
        rng.standard_normal((n_events, dim)).astype(np.float32)
    )
    preds_k0 = torch.nn.functional.normalize(0.35 * targets + 0.65 * noise_k0, dim=1)
    preds_k1 = torch.nn.functional.normalize(0.75 * targets + 0.25 * noise_k0, dim=1)
    preds_k2 = torch.nn.functional.normalize(targets.clone(), dim=1)

    mask_k0 = torch.ones(n_events, dtype=torch.bool)
    mask_k1 = torch.ones(n_events, dtype=torch.bool)
    mask_k2 = torch.ones(n_events, dtype=torch.bool)
    mask_k1[[11, 23]] = False
    mask_k2[[10, 11, 22, 23]] = False

    encoded_revision = {
        "predictions": preds_k2,
        "targets": targets,
        "words": event_words,
        "event_ids": event_ids,
        "revision_predictions": {0: preds_k0, 1: preds_k1, 2: preds_k2},
        "revision_availability_masks": {0: mask_k0, 1: mask_k1, 2: mask_k2},
    }
    cfg = {
        "experiment": {"dataset": "pallier2025"},
        "dataset": {"subjects": ["sub-01", "sub-02"]},
    }

    rev_metrics = _compute_encoded_evaluation_metrics(
        encoded_revision,
        dummy_ds,
        full_vocabulary,
        vocab50,
        cfg,
        stage="revision",
        split="test",
    )

    fv_sweep = rev_metrics["vocabulary_sweep"]["full_vocab"]
    assert fv_sweep["candidate_count"] == len(full_vocabulary)
    assert fv_sweep["query_count"] == int(mask_k2.sum().item())
    assert fv_sweep["macro_top10"] == pytest.approx(1.0)
    assert fv_sweep["micro_top10"] == pytest.approx(1.0)
    assert fv_sweep["macro_top5"] == pytest.approx(1.0)
    assert fv_sweep["macro_top1"] == pytest.approx(1.0)
    assert fv_sweep["mrr"] == pytest.approx(1.0)
    assert fv_sweep["median_rank"] == pytest.approx(1.0)

    horizons = rev_metrics["revision_horizons"]
    assert set(horizons.keys()) == {"K0", "K1", "K2"}
    assert horizons["K0"]["common_full_vocab_macro_top10"] > 0.0
    assert (
        horizons["K1"]["common_full_vocab_macro_top10"]
        >= horizons["K0"]["common_full_vocab_macro_top10"]
    )
    assert horizons["K2"]["common_full_vocab_macro_top10"] == pytest.approx(1.0)
    assert horizons["K2"]["common_full_vocab_micro_top10"] == pytest.approx(1.0)
    assert horizons["K2"]["common_full_vocab_macro_top1"] == pytest.approx(1.0)
    assert horizons["K2"]["common_full_vocab_mrr"] == pytest.approx(1.0)
    assert horizons["K2"]["common_full_vocab_median_rank"] == pytest.approx(1.0)

    raw_fv_horizons = rev_metrics["full_vocabulary"]["horizons"]
    for h_key in ("0", "1", "2", "k0", "k1", "k2"):
        assert h_key in raw_fv_horizons
        assert "common_targets" in raw_fv_horizons[h_key]
        assert "all_targets" in raw_fv_horizons[h_key]

    encoded_word = {
        "predictions": preds_k2,
        "targets": targets,
        "words": event_words,
        "event_ids": event_ids,
    }
    word_metrics = _compute_encoded_evaluation_metrics(
        encoded_word,
        dummy_ds,
        full_vocabulary,
        vocab50,
        cfg,
        stage="word",
        split="test",
    )
    word_fv_sweep = word_metrics["vocabulary_sweep"]["full_vocab"]
    assert word_fv_sweep["query_count"] == n_events
    assert word_fv_sweep["macro_top10"] == pytest.approx(1.0)
    assert word_fv_sweep["micro_top10"] == pytest.approx(1.0)
    assert word_fv_sweep["macro_top1"] == pytest.approx(1.0)
    assert word_fv_sweep["mrr"] == pytest.approx(1.0)
    assert word_fv_sweep["median_rank"] == pytest.approx(1.0)


def test_configurable_window_seconds_and_offset(monkeypatch):
    import numpy as np
    from braindecoding.data.pallier2025_dataset import (
        extract_word_window,
        word_window_contract,
    )

    # 1. Window seconds configured in YAML training configs
    cfg_default, _ = build_workflow_config(
        config_path="configs/training/pallier2025_word_encoder.yaml",
        seed=0,
    )
    expected_yaml_win = float(cfg_default["workflow"]["window_seconds"])
    assert expected_yaml_win > 0.0
    assert cfg_default["dataset"]["window_seconds"] == pytest.approx(expected_yaml_win)
    assert cfg_default["dataset"].get("window_start_offset_seconds", 0.0) == pytest.approx(0.0)
    assert cfg_default["workflow"]["window_samples"] == int(round(expected_yaml_win * 50.0))

    # 2. Explicit override via build_workflow_config arguments
    cfg_150, _ = build_workflow_config(
        config_path="configs/training/pallier2025_word_encoder.yaml",
        seed=0,
        window_seconds=1.5,
        window_start_offset_seconds=0.1,
    )
    assert cfg_150["dataset"]["window_seconds"] == pytest.approx(1.5)
    assert cfg_150["dataset"]["window_start_offset_seconds"] == pytest.approx(0.1)
    assert cfg_150["workflow"]["window_seconds"] == pytest.approx(1.5)
    assert cfg_150["workflow"]["window_samples"] == 75
    assert cfg_150["workflow"]["window_start_offset_seconds"] == pytest.approx(0.1)

    # 3. Override via environment variables
    monkeypatch.setenv("BRAINDECODING_WINDOW_SECONDS", "0.5")
    monkeypatch.setenv("BRAINDECODING_WINDOW_START_OFFSET_SECONDS", "-0.04")
    cfg_env, _ = build_workflow_config(
        config_path="configs/training/pallier2025_word_encoder.yaml",
        seed=0,
    )
    assert cfg_env["dataset"]["window_seconds"] == pytest.approx(0.5)
    assert cfg_env["dataset"]["window_start_offset_seconds"] == pytest.approx(-0.04)
    assert cfg_env["workflow"]["window_samples"] == 25
    monkeypatch.delenv("BRAINDECODING_WINDOW_SECONDS", raising=False)
    monkeypatch.delenv("BRAINDECODING_WINDOW_START_OFFSET_SECONDS", raising=False)

    # 4. CLI argument parsing for train, smoke, and eval-run
    parser = cli.build_parser()
    train_args = parser.parse_args(
        [
            "train",
            "--config",
            "configs/training/pallier2025_word_encoder.yaml",
            "--window-seconds",
            "1.25",
            "--window-start-offset-seconds",
            "-0.1",
        ]
    )
    assert train_args.window_seconds == pytest.approx(1.25)
    assert train_args.window_start_offset_seconds == pytest.approx(-0.1)

    eval_args = parser.parse_args(
        [
            "eval-run",
            "--dataset",
            "pallier2025",
            "--window-seconds",
            "0.75",
        ]
    )
    assert eval_args.window_seconds == pytest.approx(0.75)

    # 5. Dynamic window slicing on 50Hz continuous signal (0.5s -> 25 samples, 1.5s -> 75 samples)
    c_short = word_window_contract({"window_seconds": 0.5})
    assert c_short["window_samples"] == 25
    assert c_short["baseline_samples"] == 25

    c_long = word_window_contract(cfg_150["dataset"])
    assert c_long["window_samples"] == 75
    assert c_long["baseline_samples"] == 25

    fake_signal = np.ones((4, 200), dtype=np.float32)
    meta = {
        "source_first_samp": 0,
        "source_sampling_rate_hz": 1000.0,
        "output_sampling_rate_hz": 50.0,
    }
    win_short = extract_word_window(
        fake_signal, 0.5, meta, dataset_config={"window_seconds": 0.5}
    )
    assert win_short.shape == (4, 25)
    win_long = extract_word_window(
        fake_signal, 0.5, meta, dataset_config={"window_seconds": 1.5}
    )
    assert win_long.shape == (4, 75)




def test_siamese_common_mode_rejection_and_shift_null_training(tmp_path):
    import numpy as np
    import torch
    from braindecoding.losses import build_siglip_loss
    from braindecoding.models import build_brain_embedding_model
    from braindecoding.optimizers import build_adamw_for_modules
    from braindecoding.training.siamese_debiased_training import (
        build_centered_vocabulary_bank,
        compute_norm_damped_debiased_embedding,
        compute_shift_null_loss,
        encode_from_preextracted_cnn,
        forward_revision_from_preextracted,
        preextract_frozen_cnn_features,
        resolve_siamese_debiased_config,
        train_one_epoch_siamese_debiased,
    )
    from braindecoding.training.word_encoder_training import (
        SentenceBatchSampler,
        encode_loader,
    )

    # 1. Verify decoupled YAML configs under configs/training/siamese_debiased/
    siamese_configs = [
        (
            "configs/training/siamese_debiased/pallier2025_siamese_debiased_revision.yaml",
            "pallier2025",
            "revision",
        ),
        (
            "configs/training/siamese_debiased/pallier2025_siamese_debiased_context.yaml",
            "pallier2025",
            "context",
        ),
    ]
    for rel_path, expected_ds, expected_stage in siamese_configs:
        full_p = PROJECT_ROOT / rel_path
        assert full_p.is_file(), f"缺少孪生去偏训练配置: {rel_path}"
        cfg, sub = build_workflow_config(config_path=rel_path, seed=0)
        assert sub["dataset"] == expected_ds
        assert cfg["workflow"]["stage"] == expected_stage
        assert float(cfg["workflow"]["window_seconds"]) > 0.0
        assert cfg["training"]["learning_rate"] == pytest.approx(1e-4)
        assert cfg["training"]["weight_decay"] == pytest.approx(0.01)
        assert cfg["training"]["freeze_brain_encoder_epochs"] >= cfg["training"]["epochs"]
        siam = cfg["workflow"]["siamese_debiased"]
        assert siam["enabled"] is True
        assert 0.0 < float(siam["eta_train"]) <= 1.5
        assert float(siam["lambda_null"]) >= 0.0
        assert 1 <= int(siam["shifts_per_step"]) <= len(siam["shift_seconds"])
        assert siam["shift_seconds"] == [-15.0, -30.0, -45.0, -60.0, -90.0, -120.0]
        assert siam["preextract_frozen_cnn"] is True

    # 2. Verify CLI argument parsing & override in build_workflow_config
    parser = cli.build_parser()
    train_args = parser.parse_args(
        [
            "train",
            "--config",
            "configs/training/pallier2025_rolling_revision.yaml",
            "--siamese-debiased",
            "--eta-train",
            "0.5",
            "--lambda-null",
            "12.0",
            "--shifts-per-step",
            "6",
            "--preextract-frozen-cnn",
        ]
    )
    assert train_args.siamese_debiased is True
    assert train_args.eta_train == pytest.approx(0.5)
    assert train_args.lambda_null == pytest.approx(12.0)
    assert train_args.shifts_per_step == 6
    assert train_args.preextract_frozen_cnn is True

    cfg_override, _ = build_workflow_config(
        config_path="configs/training/pallier2025_rolling_revision.yaml",
        seed=0,
        eta_train=0.5,
        lambda_null=12.0,
        shifts_per_step=6,
    )
    assert cfg_override["workflow"]["siamese_debiased"]["enabled"] is True
    assert cfg_override["workflow"]["siamese_debiased"]["eta_train"] == pytest.approx(0.5)
    assert cfg_override["workflow"]["siamese_debiased"]["lambda_null"] == pytest.approx(12.0)
    assert cfg_override["workflow"]["siamese_debiased"]["shifts_per_step"] == 6

    # Reject stage=word with explicit siamese_debiased=True
    with pytest.raises(ValueError, match="仅适用于 context"):
        build_workflow_config(
            config_path="configs/training/pallier2025_word_encoder.yaml",
            siamese_debiased=True,
        )

    # Implicit leftover eta_train / lambda_null / shifts_per_step on stage=word must be safely ignored
    cfg_word_ignore, _ = build_workflow_config(
        config_path="configs/training/pallier2025_word_encoder.yaml",
        seed=1,
        eta_train=0.8,
        lambda_null=10.0,
        shifts_per_step=6,
    )
    assert cfg_word_ignore["workflow"]["stage"] == "word"
    assert cfg_word_ignore["workflow"]["siamese_debiased"]["enabled"] is False
    assert cfg_word_ignore["workflow"]["siamese_debiased"]["preextract_frozen_cnn"] is False

    # Multi-seed CLI parsing (--seed 0,1,2 and --seeds 0,1,2)
    multiseed_args = parser.parse_args(
        [
            "train",
            "--config",
            "configs/training/pallier2025_word_encoder.yaml",
            "--seed",
            "0,1,2",
        ]
    )
    assert multiseed_args.seed == [0, 1, 2]
    assert cli._resolve_seed_list(multiseed_args) == [0, 1, 2]
    singleseed_args = parser.parse_args(
        [
            "train",
            "--config",
            "configs/training/pallier2025_word_encoder.yaml",
            "--seed",
            "1",
        ]
    )
    assert singleseed_args.seed == 1
    assert cli._resolve_seed_list(singleseed_args) == [1]

    # 3. Verify centered vocabulary bank & shift-null loss properties
    dim = 32
    vocab = [f"w_{i:02d}" for i in range(16)]
    rng = np.random.default_rng(7)
    emb_map = {}
    for w in vocab:
        v = rng.standard_normal(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        emb_map[w] = v

    class _DummySignalDataset(torch.utils.data.Dataset):
        def __init__(self, n_rows: int):
            self.embedding_map = emb_map
            self.channel_count = 8
            self.window_samples = 25
            self.sampling_rate_hz = 50.0
            self.temporal_shift_seconds = 0.0
            self.temporal_shift_samples = 0
            self._words = tuple(vocab[i % len(vocab)] for i in range(n_rows))
            self._event_ids = tuple(f"ev-{i:03d}" for i in range(n_rows))
            self._embedding_tensors = {
                w: torch.from_numpy(vec) for w, vec in emb_map.items()
            }
            self._subject_index_tensors = torch.tensor(
                [i % 2 for i in range(n_rows)], dtype=torch.long
            )
            # 2 sentences of 6 words each
            self._sentence_index_tensors = torch.tensor(
                [i // 6 for i in range(n_rows)], dtype=torch.long
            )
            self.sentence_uids = [f"sent-{i // 6}" for i in range(n_rows)]
            # Deterministic synthetic continuous signal per row + shift
            self._base_signals = torch.randn(
                n_rows, self.channel_count, self.window_samples,
                generator=torch.Generator().manual_seed(99),
            )

        def __len__(self):
            return len(self._words)

        def __getitem__(self, idx):
            i = int(idx)
            w = self._words[i]
            shift_offset = int(self.temporal_shift_samples)
            shifted_idx = (i + shift_offset) % len(self._words)
            sig = self._base_signals[shifted_idx]
            return {
                "meg": sig.clone(),
                "text_embedding": self._embedding_tensors[w],
                "subject_index": self._subject_index_tensors[i],
                "sentence_index": self._sentence_index_tensors[i],
                "word": w,
                "event_id": self._event_ids[i],
                "recording_id": "rec-01",
            }

    device = torch.device("cpu")
    train_ds = _DummySignalDataset(12)
    val_ds = _DummySignalDataset(6)

    centered_bank = build_centered_vocabulary_bank(train_ds, vocab, device)
    assert centered_bank.shape == (len(vocab), dim)
    # Column-wise mean of centered vocabulary bank must be strictly 0
    assert torch.allclose(
        centered_bank.mean(dim=0), torch.zeros(dim), atol=1e-6
    )
    zero_null = compute_shift_null_loss(torch.zeros(4, dim), centered_bank)
    assert float(zero_null) == pytest.approx(0.0, abs=1e-7)
    nonzero_null = compute_shift_null_loss(centered_bank[:4], centered_bank)
    assert float(nonzero_null) > 0.0

    # 4. Build miniature revision model and verify dual-branch gradient flow + synchronized dropout RNG
    model_cfg = {
        "embedding_dimension": dim,
        "use_transformer": True,
        "context_mode": "grouped",
        "conv": {
            "merger_channels": 8,
            "merger_position_dimension": 8,
            "merger_per_subject": False,
            "merger_dropout": 0.0,
            "initial_linear": 16,
            "hidden": 16,
            "depth": 2,
            "kernel_size": 3,
            "dilation_growth": 2,
            "dilation_period": 2,
            "dropout_input": 0.0,
            "convolution_dropout": 0.0,
            "batch_norm": True,
            "gelu": True,
            "skip": True,
            "glu_every": 0,
            "glu_context": 0,
            "temporal_attention_hidden": 16,
        },
        "transformer": {
            "depth": 2,
            "heads": 4,
            "attention_dropout": 0.2,
            "feedforward_dropout": 0.2,
            "use_scalenorm": True,
            "use_rmsnorm": False,
            "rotary_position_embedding": True,
            "residual_attention": False,
            "scale_residual": True,
        },
        "local_window_revision": {
            "enabled": True,
            "history_radius": 2,
            "horizons": [0, 1, 2],
            "validation_query_batch_size": 32,
        },
    }
    ch_pos = np.linspace(0.1, 0.9, 8 * 2, dtype=np.float32).reshape(8, 2)
    model = build_brain_embedding_model(8, ch_pos, 2, model_cfg).to(device)
    loss_module = build_siglip_loss(
        {
            "name": "SigLip",
            "initial_temperature": 10.0,
            "initial_bias": -10.0,
            "identical_candidates_threshold": 0.999,
        }
    ).to(device)

    # Freeze brain_encoder
    for p in model.brain_encoder.parameters():
        p.requires_grad_(False)
    model.brain_encoder.eval()
    model.context_transformer.train()

    # Verify synchronized dropout RNG: if clean_z == shift_z, y_clean must equal y_shift_mean even with dropout=0.2 in train() mode!
    dummy_z = torch.nn.functional.normalize(torch.randn(12, dim), dim=-1)
    sent_idx = train_ds._sentence_index_tensors
    y_clean, y_shift_mean, rev_rows, rev_delays = forward_revision_from_preextracted(
        model,
        dummy_z,
        sent_idx,
        shifted_z_batches=[dummy_z.clone(), dummy_z.clone()],
        cutoff_policy="all",
        synchronize_dropout_rng=True,
    )
    assert y_shift_mean is not None
    assert y_clean.requires_grad is True
    assert y_shift_mean.requires_grad is True
    assert y_shift_mean.grad_fn is not None
    assert torch.allclose(y_clean, y_shift_mean, atol=1e-6)

    # Verify Norm-Damped Detach (compute_norm_damped_debiased_embedding):
    # 1) Forward value is 100% identical to standard L2 normalization of (y_clean - eta * y_shift_mean.detach());
    # 2) Blocks main-loss gradients into y_shift_mean while compute_shift_null_loss still differentiates y_shift_mean;
    # 3) Cancels the 1 / ||u||_2 Jacobian amplification so backward gradient is pure tangent-space projection (I - y_tilde * y_tilde^T).
    y_tilde_damped = compute_norm_damped_debiased_embedding(
        y_clean, y_shift_mean, eta_train=0.8
    )
    u_ref = y_clean - 0.8 * y_shift_mean.detach()
    y_tilde_ref = u_ref / u_ref.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
    assert torch.allclose(y_tilde_damped, y_tilde_ref, atol=1e-7)

    grad_from_main = torch.autograd.grad(
        y_tilde_damped.sum(),
        y_shift_mean,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert grad_from_main is None

    upstream_g = torch.randn_like(y_tilde_damped)
    grad_clean_damped = torch.autograd.grad(
        (y_tilde_damped * upstream_g).sum(),
        y_clean,
        retain_graph=True,
    )[0]
    # Pure tangent projection: g - (g . y_tilde) * y_tilde (without 1 / ||u||_2 amplification)
    expected_tangent_grad = upstream_g - (
        (upstream_g * y_tilde_ref.detach()).sum(dim=-1, keepdim=True)
        * y_tilde_ref.detach()
    )
    assert torch.allclose(grad_clean_damped, expected_tangent_grad, atol=1e-5)

    null_loss_check = compute_shift_null_loss(y_shift_mean, centered_bank)
    grad_from_null = torch.autograd.grad(
        null_loss_check, y_shift_mean, retain_graph=False
    )[0]
    assert grad_from_null is not None
    assert float(grad_from_null.abs().max()) > 0.0

    # 5. Pre-extract frozen CNN features (Clean + 6 Shifted) and verify disk cache reload
    shifts = [-15.0, -30.0, -45.0, -60.0, -90.0, -120.0]
    pre_cache = preextract_frozen_cnn_features(
        model,
        train_ds,
        val_ds,
        dataset="pallier2025",
        subset_tag="unit-test",
        word_ckpt_sha256="",
        window_seconds=0.5,
        shift_seconds=shifts,
        device=device,
        batch_size=8,
        amp=False,
        cache_to_disk=False,
        show_progress=False,
    )
    assert pre_cache["train_clean_z"].shape == (12, dim)
    assert pre_cache["val_clean_z"].shape == (6, dim)
    assert len(pre_cache["train_shifted_z"]) == 6
    assert train_ds.temporal_shift_seconds == 0.0
    assert train_ds.temporal_shift_samples == 0

    # 6. Verify encode_from_preextracted_cnn matches encode_loader bit-for-bit on validation set
    val_sampler = SentenceBatchSampler(
        val_ds.sentence_uids,
        batch_size=6,
        shuffle=False,
        seed=0,
        preserve_complete_groups=True,
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_sampler=val_sampler)
    enc_standard = encode_loader(
        model, val_loader, device, amp=False, show_progress=False
    )
    enc_fast = encode_from_preextracted_cnn(
        model,
        val_sampler,
        pre_cache["val_clean_z"],
        pre_cache["val_meta"],
        device,
        amp=False,
        show_progress=False,
    )
    assert torch.allclose(
        enc_standard["predictions"], enc_fast["predictions"], atol=1e-5
    )
    for d in (0, 1, 2):
        assert torch.allclose(
            enc_standard["revision_predictions"][d],
            enc_fast["revision_predictions"][d],
            atol=1e-5,
        )

    # 7. Run 1 epoch of train_one_epoch_siamese_debiased and verify brain_encoder remains untouched while Transformer updates
    train_sampler = SentenceBatchSampler(
        train_ds.sentence_uids,
        batch_size=6,
        shuffle=True,
        seed=0,
        preserve_complete_groups=True,
    )
    tc = {
        "seed": 0,
        "epochs": 2,
        "batch_size": 6,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "amp": False,
        "amp_dtype": "float32",
        "revision_cutoff_policy": "all",
    }
    siamese_cfg = resolve_siamese_debiased_config(
        {},
        siamese_debiased=True,
        eta_train=0.8,
        lambda_null=10.0,
        shifts_per_step=2,
    )
    optimizer = build_adamw_for_modules([model, loss_module], tc)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    cnn_before = {
        k: v.detach().clone() for k, v in model.brain_encoder.state_dict().items()
    }
    tf_before = {
        k: v.detach().clone()
        for k, v in model.context_transformer.state_dict().items()
    }

    loss_val, updates, batches, diag = train_one_epoch_siamese_debiased(
        model,
        loss_module,
        train_sampler,
        pre_cache,
        optimizer,
        scaler,
        device,
        tc,
        siamese_cfg,
        centered_vocab_bank=centered_bank,
        epoch_index=0,
        show_progress=False,
    )
    assert np.isfinite(loss_val)
    assert updates == 2
    assert batches == 2
    assert diag["siamese_debiased_enabled"] is True
    assert diag["eta_train"] == pytest.approx(0.8)
    assert diag["lambda_null"] == pytest.approx(10.0)
    assert diag["siglip_loss"] > 0.0
    assert diag["shift_null_loss"] >= 0.0
    assert set(diag["prefix_revision"]["delay_counts"].keys()) == {0, 1, 2}

    # Brain encoder weights must be 100% unchanged, Transformer weights must have updated
    for k, v in model.brain_encoder.state_dict().items():
        assert torch.equal(v, cnn_before[k])
    tf_changed = any(
        not torch.allclose(v, tf_before[k])
        for k, v in model.context_transformer.state_dict().items()
    )
    assert tf_changed is True

    # 8. Verify circular modulo wrapping (and zero clamping) for negative timestamps in the first 0..120s of a recording
    from braindecoding.data.pallier2025_dataset import Pallier2025WordDataset

    rec_samples = 15000  # 300 seconds @ 50Hz
    win_samples = 50     # 1.0 second @ 50Hz
    valid_span = rec_samples - win_samples + 1
    # Linear ramp signal so every starting sample index has a unique mean/offset signature
    ramp_signal_path = tmp_path / "ramp_recording.npy"
    ramp_array = np.broadcast_to(
        np.arange(rec_samples, dtype=np.float32)[None, :], (4, rec_samples)
    ).copy()
    np.save(ramp_signal_path, ramp_array)

    # Test an early word event at t_onset = 2.0s (start = 100 samples), where t_idx + shift_idx < 0 for all 6 negative shifts
    early_start_sample = 100
    pal_dummy = object.__new__(Pallier2025WordDataset)
    pal_dummy.zero_meg = False
    pal_dummy.channel_count = 4
    pal_dummy.window_samples = win_samples
    pal_dummy._baseline_samples = 0
    pal_dummy._clamp = 1e9
    pal_dummy._window_offset_seconds = 0.0
    pal_dummy._strict_window_bounds = True
    pal_dummy.sampling_rate_hz = 50.0
    pal_dummy._signal_path_strs = (str(ramp_signal_path),)
    pal_dummy._window_starts = np.asarray([early_start_sample], dtype=np.int64)

    pal_first_samples = []
    for delta_sec in (-15.0, -30.0, -45.0, -60.0, -90.0, -120.0):
        shift_samp = int(round(delta_sec * 50.0))
        assert early_start_sample + shift_samp < 0
        expected_wrapped_start = (early_start_sample + shift_samp) % valid_span
        assert expected_wrapped_start > 0

        pal_dummy.temporal_shift_seconds = delta_sec
        pal_dummy.temporal_shift_samples = shift_samp
        pal_win = pal_dummy.read_meg(0)
        assert pal_win.shape == (4, win_samples)
        pal_first_samples.append(float(pal_win[0, 0]))
        assert float(pal_win[0, 0]) == pytest.approx(float(expected_wrapped_start))

    # All 6 negative shifts must wrap to 6 strictly distinct, non-overlapping positions (never clamped to 0)
    assert len(set(pal_first_samples)) == 6
    assert 0.0 not in pal_first_samples
    for a, b in zip(pal_first_samples, pal_first_samples[1:]):
        assert abs(a - b) >= win_samples






def test_stage1_standalone_and_multi_seed_training_and_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    """Verify Stage 1 standalone training/evaluation, WORD_EXP_TAG fallback, and multi-seed aggregation."""
    import numpy as np
    from braindecoding.decoding.neural_score_exporter import _locate_stage_checkpoint
    from braindecoding.evaluation.word_retrieval_evaluation import (
        summarize_multi_seed_evaluation,
    )
    import scripts.decode_sentences as decode_sentences_script

    # 1. Stage 1 (word) ignores implicit leftover Siamese parameters (eta_train, lambda_null, shifts_per_step)
    cfg_word, _ = build_workflow_config(
        config_path="configs/training/pallier2025_word_encoder.yaml",
        seed=0,
        eta_train=0.8,
        lambda_null=0.5,
        shifts_per_step=6,
    )
    assert cfg_word["workflow"]["stage"] == "word"
    assert cfg_word.get("siamese_debiased", {}).get("enabled", False) is False

    # Explicit siamese_debiased=True on Stage 1 still raises ValueError
    with pytest.raises(ValueError, match="仅适用于 context"):
        build_workflow_config(
            config_path="configs/training/pallier2025_word_encoder.yaml",
            seed=0,
            siamese_debiased=True,
        )

    # 2. Multi-seed CLI parsing across train, eval-run, and decode-sentences
    parser = cli.build_parser()
    for subcmd in ("train", "smoke", "eval-run", "decode-sentences"):
        parsed = parser.parse_args(
            [subcmd, "--dataset", "pallier2025", "--seed", "0,1,2"]
        )
        assert parsed.seed == [0, 1, 2]
        assert cli._resolve_seed_list(parsed) == [0, 1, 2]

    ds_parser = decode_sentences_script.build_parser()
    ds_parsed = ds_parser.parse_args(
        ["--dataset", "pallier2025", "--seeds", "0,1,2"]
    )
    assert ds_parsed.seed == [0, 1, 2]

    # 3. _locate_stage_checkpoint falls back from EXP_TAG to WORD_EXP_TAG / untagged for Stage 1
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("BRAINDECODING_RESULTS_DIR", str(runs_dir))
    word_tag_ckpt = (
        runs_dir
        / "pallier2025"
        / "lr4e-4_bs256"
        / "word"
        / "all-n10"
        / "seed-001"
        / "best.ckpt"
    )
    word_tag_ckpt.parent.mkdir(parents=True, exist_ok=True)
    word_tag_ckpt.write_bytes(b"dummy-word-ckpt")

    located_ckpt = _locate_stage_checkpoint(
        "pallier2025",
        "word",
        "all-n10",
        1,
        exp_tag="lr4e-4_bs256",
        fallback_exp_tags=("siamese_stage2_tag", ""),
    )
    assert located_ckpt == word_tag_ckpt.resolve()

    # 4. Multi-seed single-stage (Stage 1 word) aggregation & <=86-col report formatting
    ms_dir = tmp_path / "ms_word_dir"
    ms_dir.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict] = []
    for s_idx, top10_val in ((0, 0.410), (1, 0.430), (2, 0.420)):
        seed_out = ms_dir / f"seed-{s_idx:03d}"
        seed_out.mkdir(parents=True, exist_ok=True)
        seed_results.append(
            {
                "dataset": "pallier2025",
                "stage": "word",
                "split": "test",
                "seed": s_idx,
                "exp_tag": "lr4e-4_bs256",
                "output_dir": str(seed_out),
                "subset": {
                    "subset_tag": "all-n10",
                    "subject_count": 10,
                    "total_canonical_subjects": 10,
                },
                "checkpoint": {"path": str(seed_out / "best.ckpt"), "epoch": 15},
                "window_seconds": 1.0,
                "window_samples": 50,
                "selection_metric": "pallier2025_50_macro",
                "selection_score": top10_val,
                "metrics": {
                    "vocabulary_sweep": {
                        "N20": {
                            "label": "Top-20 Vocab",
                            "candidate_count": 20,
                            "query_count": 1000,
                            "chance_top10": 0.50,
                            "macro_top10": top10_val + 0.25,
                            "micro_top10": top10_val + 0.26,
                            "macro_top5": 0.50,
                            "macro_top1": 0.22,
                            "mrr": 0.38,
                            "median_rank": 6.0,
                        },
                        "N50": {
                            "label": "Top-50 Vocab",
                            "candidate_count": 50,
                            "query_count": 1000,
                            "chance_top10": 0.20,
                            "macro_top10": top10_val,
                            "micro_top10": top10_val + 0.01,
                            "macro_top5": 0.280,
                            "macro_top1": 0.120,
                            "mrr": 0.215,
                            "median_rank": 13.0,
                        },
                        "full_vocab": {
                            "label": "Full Vocab (2020)",
                            "candidate_count": 2020,
                            "query_count": 1000,
                            "chance_top10": 10.0 / 2020.0,
                            "macro_top10": 0.085,
                            "micro_top10": 0.092,
                            "macro_top5": 0.045,
                            "macro_top1": 0.018,
                            "mrr": 0.042,
                            "median_rank": 180.0,
                        },
                    },
                },
                "negative_controls": {
                    "clean_reference": {
                        "vocab50_macro_top10": top10_val,
                        "vocab50_micro_top10": top10_val + 0.01,
                        "vocab50_macro_top1": 0.120,
                        "full_vocab_macro_top10": 0.085,
                    },
                    "shifted_aggregate": {
                        "mean_vocab50_macro_top10": 0.205,
                        "mean_vocab50_micro_top10": 0.210,
                        "mean_vocab50_macro_top1": 0.021,
                        "mean_full_vocab_macro_top10": 0.005,
                    },
                },
            }
        )

    ms_summary = summarize_multi_seed_evaluation(
        seed_results,
        split="test",
        output_dir=ms_dir,
        print_report=False,
    )
    assert ms_summary["seed_count"] == 3
    assert ms_summary["seeds"] == [0, 1, 2]
    n50_agg = ms_summary["vocabulary_sweep_aggregate"]["N50"]
    assert n50_agg["mean_macro_top10"] == pytest.approx(0.420)
    assert n50_agg["std_macro_top10"] == pytest.approx(float(np.std([0.41, 0.43, 0.42], ddof=0)))


def test_stage23_prefix_revision_configs_cli_and_static_tables():
    from braindecoding.training.siamese_debiased_training import (
        _build_static_revision_eval_table,
        _build_static_revision_train_table,
    )
    from braindecoding.models import build_word_decoding_model
    import numpy as np
    import torch

    for alias in ("2.3", "stage2.3", "prefix_revision", "full_history_revision"):
        assert normalize_workflow_stage(alias) == "prefix_revision"

    parser = cli.build_parser()
    args = parser.parse_args(
        ["train", "--dataset", "pallier2025", "--stage", "2.3", "--seed", "0"]
    )
    assert normalize_workflow_stage(args.stage) == "prefix_revision"

    stage23_configs = [
        ("pallier2025", "configs/training/pallier2025_prefix_revision.yaml", False),
        ("pallier2025", "configs/training/siamese_debiased/pallier2025_siamese_debiased_prefix_revision.yaml", True),
    ]
    for expected_ds, rel_yaml, expect_siamese in stage23_configs:
        cfg, subset = build_workflow_config(config_path=rel_yaml, seed=0, mode="train")
        assert subset["dataset"] == expected_ds
        assert cfg["workflow"]["stage"] == "prefix_revision"
        assert cfg["model"]["use_transformer"] is True
        assert cfg["model"]["local_window_revision"]["enabled"] is True
        assert cfg["model"]["local_window_revision"]["history_radius"] is None
        assert cfg["model"]["local_window_revision"]["horizons"] == [0, 1, 2]
        assert cfg["training"]["revision_cutoff_policy"] == "all"
        assert "pretrained_brain_encoder_checkpoint" in cfg["training"]
        assert bool(cfg.get("siamese_debiased", {}).get("enabled", False)) is expect_siamese

    positions = np.stack([np.linspace(0, 1, 6), np.linspace(1, 0, 6)], axis=1).astype("float32")
    small_cfg = {
        "embedding_dimension": 8,
        "use_transformer": True,
        "context_mode": "grouped",
        "conv": {
            "merger_channels": 4,
            "merger_position_dimension": 8,
            "merger_dropout": 0,
            "initial_linear": 6,
            "hidden": 4,
            "depth": 2,
            "kernel_size": 3,
            "dilation_growth": 2,
            "dilation_period": 2,
            "dropout_input": 0,
            "batch_norm": False,
            "gelu": True,
            "skip": True,
            "glu_every": 0,
            "temporal_attention_hidden": 4,
        },
        "transformer": {"depth": 1, "heads": 2, "attention_dropout": 0.0, "feedforward_dropout": 0.0},
        "local_window_revision": {"enabled": True, "history_radius": None, "horizons": [0, 1, 2]},
    }
    model = build_word_decoding_model(6, positions, 1, small_cfg).eval()
    sentence_indices = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)
    train_tbl = _build_static_revision_train_table(model, sentence_indices, torch.device("cpu"))
    assert train_tbl is not None
    # Group 0 (len 4) has 4 unique arrival cutoffs + Group 1 (len 2) has 2 unique arrival cutoffs = 6 unique windows
    assert int(train_tbl["row_is_unique_win"].sum().item()) == 6
    # Total (i, r) pairs: Group 0 has 4(r=0)+3(r=1)+2(r=2)=9; Group 1 has 2(r=0)+1(r=1)+0(r=2)=3 -> total 12
    assert int(train_tbl["row_valid_delay"].sum().item()) == 12
    assert train_tbl["row_win_global_idx"].shape == (6, 3, 4)

    eval_tbl = _build_static_revision_eval_table(model, sentence_indices, torch.device("cpu"))
    assert eval_tbl is not None
    assert set(eval_tbl.keys()) == {0, 1, 2}
    assert eval_tbl[2]["win_global_idx"].shape == (6, 4)




def test_common_horizon_delay_subset_filter_and_cli_arg() -> None:
    import pandas as pd
    from braindecoding.cli import build_parser
    from braindecoding.evaluation.word_retrieval_evaluation import (
        _build_common_horizon_event_filter,
    )

    parser = build_parser()
    parsed = parser.parse_args(
        [
            "eval-run",
            "--dataset",
            "pallier2025",
            "--stage",
            "word",
            "--common-horizon-delay",
            "2",
        ]
    )
    assert parsed.common_horizon_delay == 2

    df = pd.DataFrame(
        {
            "event_id": ["e0", "e1", "e2", "e3", "e4", "e5"],
            "recording_id": ["r0", "r0", "r0", "r0", "r0", "r0"],
            "context_id": ["s0", "s0", "s0", "s0", "s1", "s1"],
            "normalized_word": ["le", "petit", "prince", "parle", "un", "mot"],
        }
    )
    kept_ids, _ = _build_common_horizon_event_filter(
        {},
        dataset="custom_toy",
        split="test",
        selected_subjects=("sub-01",),
        common_horizon_delay=2,
        fallback_eval_table=df,
    )
    assert kept_ids == {"e0", "e1"}

    parsed_report = parser.parse_args(
        [
            "report",
            "--dataset",
            "pallier2025",
            "--exp-tag",
            "lr4e-4_bs256",
            "--stage",
            "word",
            "--seeds",
            "0,1,2",
            "--split",
            "test",
            "--word-exp-tag",
            "lr4e-4_bs256",
        ]
    )
    assert parsed_report.dataset == "pallier2025"
    assert parsed_report.exp_tag == "lr4e-4_bs256"
    assert parsed_report.stage == "word"
    assert parsed_report.seed == [0, 1, 2]
    assert parsed_report.split == "test"
    assert parsed_report.word_exp_tag == "lr4e-4_bs256"


def test_cli_cache_parses_pnpl_flags_and_loads_pnpl_stage_yaml_configs():
    parser = cli.build_parser()
    cache_args = parser.parse_args(
        [
            "cache",
            "--dataset",
            "libribrain100",
            "--pnpl-root",
            "/tmp/pnpl_raw",
            "--split-policy",
            "simpleb2t",
            "--sessions",
            "1,11,0",
            "--runs",
            "1,2",
            "--tasks",
            "Sherlock1,Sherlock2",
            "--include-run-key",
            "0,1,Sherlock1,1",
            "--include-run-key",
            "0,11,Sherlock1,2",
            "--include-test",
            "--no-download",
        ]
    )
    assert cache_args.command == "cache"
    assert cache_args.dataset == "libribrain100"
    assert cache_args.pnpl_root == "/tmp/pnpl_raw"
    assert cache_args.split_policy == "simpleb2t"
    assert cache_args.sessions == "1,11,0"
    assert cache_args.runs == "1,2"
    assert cache_args.tasks == "Sherlock1,Sherlock2"
    assert cache_args.include_run_key == ["0,1,Sherlock1,1", "0,11,Sherlock1,2"]
    assert cache_args.include_test is True
    assert cache_args.no_download is True

    pnpl_configs = [
        ("libribrain100", "word", "configs/training/libribrain100_word_encoder.yaml"),
        ("libribrain100", "context", "configs/training/libribrain100_context_transformer.yaml"),
        ("libribrain100", "prefix_revision", "configs/training/libribrain100_prefix_revision.yaml"),
        ("libribrain100", "context", "configs/training/siamese_debiased/libribrain100_siamese_debiased_context.yaml"),
        ("libribrain100", "prefix_revision", "configs/training/siamese_debiased/libribrain100_causal_null_space_prefix_revision.yaml"),
        ("pallier2025_full", "word", "configs/training/pallier2025_full_word_encoder.yaml"),
        ("pallier2025_full", "context", "configs/training/pallier2025_full_context_transformer.yaml"),
        ("pallier2025_full", "prefix_revision", "configs/training/pallier2025_full_prefix_revision.yaml"),
        ("pallier2025_full", "context", "configs/training/siamese_debiased/pallier2025_full_siamese_debiased_context.yaml"),
        ("pallier2025_full", "prefix_revision", "configs/training/siamese_debiased/pallier2025_full_causal_null_space_prefix_revision.yaml"),
    ]
    for expected_ds, expected_stage, rel_yaml in pnpl_configs:
        cfg, subset = build_workflow_config(config_path=rel_yaml, seed=0, mode="smoke")
        assert subset["dataset"] == expected_ds
        assert cfg["workflow"]["stage"] == expected_stage
        assert cfg["text_embedding"]["model_name"] == "t5-large"
        assert f"/{expected_ds}/events/events.csv" in cfg["cache"]["event_table"].replace("\\", "/")
        assert f"/{expected_ds}/signals" in cfg["cache"]["meg_dir"].replace("\\", "/")


def test_run_cache_workflow_wires_materialize_pnpl_dataset(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd

    import braindecoding.data.pnpl_dataset_adapter as pnpl_adapter
    import braindecoding.data.text_embeddings as text_data
    import braindecoding.training.workflow_runner as wr

    cache_dir = tmp_path / "derived"
    results_dir = tmp_path / "runs"
    raw_dir = tmp_path / "raw_lb"
    raw_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BRAINDECODING_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("BRAINDECODING_RESULTS_DIR", str(results_dir))

    captured: dict = {}

    def _fake_materialize(
        dataset,
        pnpl_root,
        output_root,
        **kwargs,
    ):
        captured["dataset"] = dataset
        captured["pnpl_root"] = Path(pnpl_root)
        captured["output_root"] = Path(output_root)
        captured["kwargs"] = dict(kwargs)
        out_root = Path(output_root)
        ev_path = out_root / "events" / "events.csv"
        sig_dir = out_root / "signals" / "meg_50hz"
        ev_path.parent.mkdir(parents=True, exist_ok=True)
        sig_dir.mkdir(parents=True, exist_ok=True)
        rel_sig = "sub-0_ses-1_task-Sherlock1_run-1.npy"
        np.save(sig_dir / rel_sig, np.zeros((4, 200), dtype=np.float32))
        (sig_dir / "sub-0_ses-1_task-Sherlock1_run-1.json").write_text(
            json.dumps({"recording_id": "rec0"}), encoding="utf-8"
        )
        rows = []
        for idx, split in enumerate(("train", "val", "test")):
            rows.append(
                {
                    "事件编号": f"e{idx}",
                    "受试者": "sub-0",
                    "记录编号": f"rec{idx}",
                    "材料编号": f"Sherlock1|run-{idx}",
                    "划分单元": f"libribrain100|ses-{idx}",
                    "记录内序号": 0,
                    "词": "Holmes",
                    "标准词": "holmes",
                    "开始时间": 1.0,
                    "结束时间": 1.2,
                    "上下文编号": f"ctx{idx}",
                    "数据划分": split,
                    "是否可训练": True,
                    "排除原因": "",
                    "信号相对路径": rel_sig,
                }
            )
        pd.DataFrame(rows).to_csv(ev_path, index=False, encoding="utf-8")
        return {
            "dataset": dataset,
            "recording_count": 1,
            "event_count": 3,
            "trainable_event_count": 3,
            "events_path": str(ev_path),
        }

    def _fake_ensure_text(words, text_cfg, cache_path):
        cache_p = Path(cache_path)
        unique_words = sorted({str(w) for w in words})
        vecs = np.ones((len(unique_words), 8), dtype=np.float32)
        text_data.save_text_embedding_cache(cache_p, unique_words, vecs, text_cfg)
        return cache_p

    monkeypatch.setattr(pnpl_adapter, "materialize_pnpl_dataset", _fake_materialize)
    monkeypatch.setattr(text_data, "ensure_text_embedding_cache", _fake_ensure_text)

    report = wr.run_cache_workflow(
        "libribrain100",
        pnpl_root=raw_dir,
        split_policy="simpleb2t",
        include_run_keys=["0,1,Sherlock1,1"],
        include_test=True,
        download=False,
    )
    assert report["status"] == "cache_ready"
    assert report["dataset"] == "libribrain100"
    assert report["subset"]["subjects"] == ["sub-0"]
    assert captured["dataset"] == "libribrain100"
    assert captured["pnpl_root"] == raw_dir.resolve()
    assert captured["kwargs"]["split_policy"] == "simpleb2t"
    assert captured["kwargs"]["include_run_keys"] == [("0", "1", "Sherlock1", "1")]
    assert captured["kwargs"]["subjects"] is None
    assert captured["kwargs"]["download"] is False
    assert report["pnpl_materialization"]["recording_count"] == 1
    assert report["cache_verification"]["verified_event_rows"] == 3







