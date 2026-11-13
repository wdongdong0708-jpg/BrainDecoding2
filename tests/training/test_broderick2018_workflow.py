from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from braindecoding.evaluation.sequence_decoding_evaluation import (
    dataset_language,
    is_content_word,
    is_function_word,
)
from braindecoding.training.workflow_runner import build_workflow_config
from scripts import cache_archive


BRODERICK_TRAINING_CONFIGS = (
    "configs/training/broderick2018_word_encoder.yaml",
    "configs/training/broderick2018_context_transformer.yaml",
    "configs/training/broderick2018_rolling_revision.yaml",
    "configs/training/broderick2018_prefix_revision.yaml",
    "configs/training/siamese_debiased/broderick2018_causal_null_space_prefix_revision.yaml",
    "configs/training/siamese_debiased/broderick2018_siamese_debiased_context.yaml",
    "configs/training/siamese_debiased/broderick2018_siamese_debiased_prefix_revision.yaml",
)


@pytest.mark.parametrize("config_rel", BRODERICK_TRAINING_CONFIGS)
def test_broderick_training_configs_resolve_all_n19_and_preroll_policy(config_rel: str):
    cfg, subset = build_workflow_config(config_path=config_rel, seed=0)
    assert cfg["experiment"]["dataset"] == "broderick2018"
    assert subset["subset_tag"] == "all-n19"
    assert subset["subject_count"] == 19
    assert int(cfg["dataset"]["expected_eeg_channel_count"]) == 128
    assert bool(cfg["dataset"]["mastoid_rereference"]) is False
    if "siamese_debiased/" in config_rel:
        sd = cfg["siamese_debiased"]
        assert sd["enabled"] is True
        assert sd["shift_mode"] == "preroll"
        assert float(sd["preroll_seconds"]) == 45.0
        assert [float(v) for v in sd["shift_seconds"]] == [-12.0, -19.0, -27.0, -36.0]
        assert int(sd["shifts_per_step"]) == 2


def test_exp_tag_and_word_exp_tag_precedence_cli_over_env_over_yaml(monkeypatch):
    config_rel = "configs/training/broderick2018_context_transformer.yaml"
    monkeypatch.setenv("BRAINDECODING_EXP_TAG", "env_exp_v1")
    monkeypatch.setenv("BRAINDECODING_WORD_EXP_TAG", "env_word_v1")

    cfg_env, _ = build_workflow_config(config_path=config_rel, seed=0)
    assert cfg_env["workflow"]["exp_tag"] == "env_exp_v1"
    assert cfg_env["workflow"]["word_exp_tag"] == "env_word_v1"
    assert "env_exp_v1" in Path(cfg_env["training"]["output_dir"]).as_posix()

    cfg_cli, _ = build_workflow_config(
        config_path=config_rel,
        seed=0,
        exp_tag="cli_exp_override",
        word_exp_tag="cli_word_override",
    )
    assert cfg_cli["workflow"]["exp_tag"] == "cli_exp_override"
    assert cfg_cli["workflow"]["word_exp_tag"] == "cli_word_override"
    assert "cli_exp_override" in Path(cfg_cli["training"]["output_dir"]).as_posix()


def test_english_function_vs_content_word_classification():
    assert dataset_language("broderick2018") == "en"
    assert dataset_language("pallier2025") == "fr"
    assert dataset_language("chineseeeg2_littleprince") == "zh"

    for func_w in ("the", "of", "and", "to", "don't", "it's", "was"):
        assert is_function_word(func_w, "en") is True
        assert is_content_word(func_w, "en") is False
    for content_w in ("sea", "fisherman", "marlin", "shark", "harbor", "skiff"):
        assert is_function_word(content_w, "en") is False
        assert is_content_word(content_w, "en") is True


def test_cache_archive_isolates_broderick2018_tar(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    bro_events = cache_dir / "broderick2018" / "events" / "events.csv"
    bro_sig = cache_dir / "broderick2018" / "signals" / "eeg_50hz" / "sub-01" / "sub-01_run-01_eeg.npy"
    pallier_sig = cache_dir / "pallier2025" / "signals" / "meg_50hz" / "sub-01_meg.npy"
    bro_events.parent.mkdir(parents=True, exist_ok=True)
    bro_sig.parent.mkdir(parents=True, exist_ok=True)
    pallier_sig.parent.mkdir(parents=True, exist_ok=True)
    bro_events.write_text("col\n1\n", encoding="utf-8")
    np.save(bro_sig, np.zeros((2, 4), dtype=np.float32))
    np.save(pallier_sig, np.zeros((2, 4), dtype=np.float32))

    archive_dir = tmp_path / "archives"
    resolved = cache_archive.resolve_archive_path(None, archive_dir=archive_dir, modality="broderick2018")
    assert resolved.name == "features_broderick2018.tar"
    assert cache_archive.resolve_archive_path(None, archive_dir=archive_dir, modality="meg").name == "features_meg.tar"
    assert cache_archive.resolve_archive_path(None, archive_dir=archive_dir, modality="eeg").name == "features_eeg.tar"

    pack_summary = cache_archive.pack_cache(
        cache_dir=cache_dir, archive_dir=archive_dir, modality="broderick2018"
    )
    assert pack_summary["file_count"] == 2
    assert Path(pack_summary["archive"]).name == "features_broderick2018.tar"

    unpack_dir = tmp_path / "unpacked"
    unpack_summary = cache_archive.unpack_cache(
        cache_dir=unpack_dir, archive_dir=archive_dir, modality="broderick2018"
    )
    assert unpack_summary["extracted_files"] == 3  # _archive_manifest.json + 2 broderick2018 files
    assert (unpack_dir / "broderick2018" / "events" / "events.csv").is_file()
    assert not (unpack_dir / "pallier2025").exists()
    assert (unpack_dir / ".unpacked_broderick2018.json").is_file()
