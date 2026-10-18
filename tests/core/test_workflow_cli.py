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


def test_cli_parses_smoke_cache_and_train_subcommands():
    parser = cli.build_parser()
    smoke_args = parser.parse_args(
        [
            "smoke",
            "--dataset",
            "chineseeeg2_littleprince",
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
    assert smoke_args.num_subjects == 1
    assert smoke_args.epochs == 2
    assert smoke_args.precision == "bf16-mixed"
    assert smoke_args.require_cache is True

    train_args = parser.parse_args(
        [
            "train",
            "--dataset",
            "pallier2025",
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
    assert train_args.subjects == "sub-01,sub-02"
    assert train_args.batch_size == 256
    assert train_args.eval_batch_size == 512
    assert train_args.learning_rate == pytest.approx(0.0004)
    assert train_args.num_workers == 0
    assert train_args.resume == "auto"

    cfg_train, _ = build_workflow_config(
        "pallier2025", stage="word", seed=0, mode="train"
    )
    assert cfg_train["training"]["batch_size"] == 256
    assert cfg_train["training"]["eval_batch_size"] == 512
    assert cfg_train["training"]["learning_rate"] == pytest.approx(0.0004)

