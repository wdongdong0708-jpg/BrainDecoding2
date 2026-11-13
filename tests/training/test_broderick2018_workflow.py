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
    "configs/training/siamese_debiased/broderick2018_amortized_timing_prefix_revision.yaml",
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
    if "amortized_timing" in config_rel:
        assert cfg["workflow"]["stage"] == "prefix_revision"
        assert cfg["siamese_debiased"]["debias_mode"] == "amortized_timing"
        assert cfg["model"]["temporal_bias_head"]["enabled"] is True


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


def test_broderick_subset_cache_workflow_and_archive_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import shutil
    import tarfile

    import numpy as np
    from scipy.io import savemat

    from braindecoding.data import text_embeddings as text_data
    from braindecoding.training import workflow_runner
    from scripts import cache_archive

    raw_root = tmp_path / "data" / "Broderick2018_NaturalSpeech"
    priv_dir = raw_root / "private"
    priv_dir.mkdir(parents=True, exist_ok=True)
    cache_base = tmp_path / "cache" / "derived"
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # 160 unique pure-alphabetic English content words, repeated to 460 tokens per run (> 450 min_matched_words, supports N<=150 vocab)
    vocab_pool = [
        f"word{chr(97 + (idx // 26))}{chr(97 + (idx % 26))}" for idx in range(160)
    ]
    tokens = [vocab_pool[i % len(vocab_pool)] for i in range(460)]
    sentence_text = " ".join(
        tok + ("." if (idx + 1) % 23 == 0 else "") for idx, tok in enumerate(tokens)
    )
    align_words = []
    for idx, tok in enumerate(tokens):
        start = round(1.0 + idx * 0.2, 3)
        align_words.append(
            {
                "case": "success",
                "word": tok,
                "alignedWord": tok,
                "start": start,
                "end": round(start + 0.15, 3),
            }
        )
    for run_num in range(1, 21):
        (priv_dir / f"oldman_run{run_num}.txt").write_text(
            sentence_text + "\n", encoding="utf-8"
        )
        (priv_dir / f"align{run_num}.json").write_text(
            json.dumps({"words": align_words}), encoding="utf-8"
        )

    rng = np.random.default_rng(7)
    # Minimal 96 s @ 128 Hz EEG per recording for Subject1 & Subject2
    n_samples = 12288
    sample_mat_template = {
        "fs": np.array([[128.0]], dtype=np.float64),
        "eegData": rng.standard_normal((n_samples, 128), dtype=np.float32),
        "mastoids": rng.standard_normal((n_samples, 2), dtype=np.float32),
    }
    for subj_num in (1, 2):
        subj_dir = raw_root / "Natural Speech" / "EEG" / f"Subject{subj_num}"
        subj_dir.mkdir(parents=True, exist_ok=True)
        for run_num in range(1, 21):
            savemat(
                subj_dir / f"Subject{subj_num}_Run{run_num}.mat",
                sample_mat_template,
            )

    monkeypatch.setenv("BRAINDATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("BRAINDECODING_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAINDECODING_CACHE_DIR", str(cache_base))
    monkeypatch.setenv("BRAINDECODING_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BRAINDECODING_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setattr(text_data, "load_text_model", lambda config: (None, None, "cpu"))
    monkeypatch.setattr(
        text_data,
        "encode_words",
        lambda words, config, **kwargs: np.ones((len(list(words)), 1024), dtype=np.float32),
    )

    report = workflow_runner.run_cache_workflow(
        "broderick2018",
        subjects="sub-01,sub-02",
        include_test=True,
    )
    assert report["status"] == "cache_ready"
    assert report["dataset"] == "broderick2018"

    ds_cache = cache_base / "broderick2018"
    assert (ds_cache / "events" / "events.csv").is_file()
    assert (ds_cache / "events" / "manifest.json").is_file()
    assert (ds_cache / "signals" / "manifest.json").is_file()
    assert not (ds_cache / "signals" / "eeg_50hz" / "manifest.json").is_file()
    assert (ds_cache / "text" / "t5_large_layer_0_5" / "embeddings.npz").is_file()
    assert (ds_cache / "events" / "manifests" / "vocabulary_N50.json").is_file()

    event_manifest = json.loads((ds_cache / "events" / "manifest.json").read_text(encoding="utf-8"))
    assert event_manifest["allow_subject_subset"] is True

    pack_info = cache_archive.pack_cache(
        cache_dir=cache_base,
        archive_dir=archive_dir,
        modality="broderick2018",
    )
    archive_path = Path(pack_info["archive"])
    assert archive_path.name == "features_broderick2018.tar"
    assert archive_path.is_file()

    with tarfile.open(archive_path, "r:") as tf:
        members = {m.name.replace("\\", "/") for m in tf.getmembers() if m.isfile()}
    assert "broderick2018/events/events.csv" in members
    assert "broderick2018/events/manifests/vocabulary_N50.json" in members
    assert "broderick2018/signals/manifest.json" in members
    assert "broderick2018/text/t5_large_layer_0_5/embeddings.npz" in members

    shutil.rmtree(cache_base)
    cache_archive.unpack_cache(cache_dir=cache_base, archive=archive_path)
    cfg, _ = workflow_runner.build_workflow_config(
        "broderick2018", stage="word", subjects="sub-01,sub-02", mode="cache"
    )
    verified = workflow_runner.verify_cache_ready(cfg, splits=("train", "val", "test"))
    assert verified["dataset"] == "broderick2018"


def test_colab_broderick_and_cpu_cache_notebooks_valid_json_and_python_syntax() -> None:
    import ast
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    dl_nb = repo_root / "scripts/colab/colab_broderick2018_download.ipynb"
    cache_nb = repo_root / "scripts/colab/colab_cpu_cache.ipynb"

    for nb_path in (dl_nb, cache_nb):
        payload = json.loads(nb_path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4
        assert len(payload["cells"]) >= 4
        for cell in payload["cells"]:
            if cell["cell_type"] == "code":
                src = "".join(cell["source"])
                ast.parse(src)

    dl_text = dl_nb.read_text(encoding="utf-8")
    for token in (
        "3461482755",
        "5b827db6986b8c8eb22e2c45534b4693",
        "1UAegPighc2t48CWpfBhjAdsZg1VlJ-mZ",
        "Broderick2018_NaturalSpeech",
        "DRYAD_PRESIGNED_URL",
        "DRYAD_OAUTH_CLIENT_ID",
    ):
        assert token in dl_text

    cache_text = cache_nb.read_text(encoding="utf-8")
    for token in (
        "broderick2018",
        "pallier2025",
        "chineseeeg2_littleprince",
        "features_broderick2018.tar",
        "features_meg.tar",
        "features_eeg.tar",
        "STAGE_RAW_TO_LOCAL_SSD",
    ):
        assert token in cache_text
    cache_code = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(cache_text).get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert "dataset_manifest.json" not in cache_code
    assert 'ds_derived / "manifest.json"' in cache_code


def test_colab_broderick_gpu_train_and_evaluation_notebooks_contract() -> None:
    import ast
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    train_nb = repo_root / "scripts/colab/colab_gpu_train_broderick.ipynb"
    eval_nb = repo_root / "scripts/colab/colab_gpu_evaluation_broderick.ipynb"

    shared_tokens = (
        "Broderick-EEG",
        "broderick2018",
        "features_broderick2018.tar",
        "/content/studies_links",
        "/content/local_cache",
        "/content/proj-venv",
        "--require-cache",
        "causal_common_mode_subtraction_control.yaml",
        "broderick2018_word_encoder.yaml",
        "broderick2018_context_transformer.yaml",
        "broderick2018_siamese_debiased_context.yaml",
        "broderick2018_causal_null_space_prefix_revision.yaml",
    )

    for nb_path in (train_nb, eval_nb):
        assert nb_path.is_file(), f"Missing notebook: {nb_path}"
        raw_text = nb_path.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
        assert payload["nbformat"] == 4
        assert len(payload["cells"]) >= 6

        code_parts: list[str] = []
        for cell in payload["cells"]:
            if cell["cell_type"] == "code":
                src = "".join(cell.get("source", []))
                ast.parse(src)
                assert "subprocess.run(" not in src
                assert not any(line.lstrip().startswith("!") for line in src.splitlines())
                code_parts.append(src)
        code_text = "\n".join(code_parts)
        for token in shared_tokens:
            assert token in code_text, f"{nb_path.name} missing token: {token}"

    train_code = "\n".join(
        "".join(c.get("source", []))
        for c in json.loads(train_nb.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    )
    assert "broderick2018_amortized_timing_prefix_revision.yaml" in train_code
    assert "AUTO_TRAIN_MISSING_STAGE1" in train_code

    eval_code = "\n".join(
        "".join(c.get("source", []))
        for c in json.loads(eval_nb.read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    )
    for token in (
        "resolve_and_export_cross_model_comparison",
        "upgrade_and_refresh_saved_multi_seed_reports",
        "common_mode_subtraction_evaluation",
    ):
        assert token in eval_code
    assert "decode-sentences" not in eval_code
    assert "Qwen3-8B" not in eval_code

