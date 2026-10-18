import copy
import importlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from braindecoding.evaluation.retrieval import (
    fixed_vocabulary_retrieval_metrics,
)
from braindecoding.experiment import (
    build_run_manifest,
    environment_provenance,
    git_tracked_dirty,
    initialize_run_directory,
    resolved_config_sha256,
    scientific_config_sha256,
)
from braindecoding.results import (
    PRIMARY_VOCABULARY_SIZES,
    build_audit_summary,
    build_evaluation_result,
    build_training_summary,
    canonical_retrieval,
    canonical_vocabulary_result,
    evaluate_vocabulary_manifests,
    write_audit_summary,
    write_evaluation_result,
)
import braindecoding.results as result_schema


def _config(dataset="fixture", budget="updates", output_dir=None):
    training = {
        "seed": 0,
        "device": "cpu",
        "num_workers": 0,
        "learning_rate": 1e-3,
        "output_dir": str(output_dir or Path("outputs") / dataset),
    }
    if budget == "updates":
        training.update(max_updates=100, epochs=20)
    else:
        training.update(epochs=12)
    return {
        "experiment": {
            "task": "word_decoding",
            "dataset": dataset,
            "subject_scope": "sub01",
            "id": "main_word",
            "category": "main",
        },
        "dataset": {
            "root": "D:/dataset/example",
            "subjects": ["sub-01"],
            "window_seconds": 1.0,
        },
        "model": {"hidden_dim": 8},
        "training": training,
    }


def _key_shape(value):
    if isinstance(value, dict):
        return {key: _key_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def _evaluation(config, vocabulary_result):
    return build_evaluation_result(
        config,
        split="val",
        query_count=6,
        event_table_sha256="e" * 64,
        subjects=["sub-01"],
        checkpoint={
            "path": "best.pt",
            "sha256": "c" * 64,
            "epoch": 3,
            "update": 75,
        },
        vocabulary_results={20: vocabulary_result},
    )


def test_evaluation_schema_is_identical_for_three_word_datasets():
    block = canonical_vocabulary_result(
        20,
        manifest_sha256="v" * 64,
        retrieval_metrics={
            "query_count": 6,
            "vocabulary_size": 20,
            "observed_vocabulary_size": 20,
            "missing_vocabulary_words": [],
            "micro_recall_at_1": 0.25,
            "micro_recall_at_10": 0.75,
            "macro_recall_at_1": 0.2,
            "macro_recall_at_10": 0.7,
            "median_rank": 4.0,
            "mean_reciprocal_rank": 0.4,
        },
    )
    payloads = [
        _evaluation(_config(dataset), block)
        for dataset in (
            "chineseeeg2_littleprince",
            "pallier2025",
        )
    ]
    assert all(payload["schema_version"] == 1 for payload in payloads)
    assert _key_shape(payloads[0]) == _key_shape(payloads[1])


def test_three_evaluators_share_the_same_canonical_serializer():
    modules = (
        "braindecoding.tasks.word_decoding.chineseeeg2_littleprince.evaluate",
        "braindecoding.tasks.word_decoding.pallier2025.evaluate",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert (
            module.canonical_evaluation_from_legacy
            is result_schema.canonical_evaluation_from_legacy
        )


def test_three_trainers_share_the_same_canonical_training_summary_builder():
    modules = (
        "braindecoding.tasks.word_decoding.chineseeeg2_littleprince.train",
        "braindecoding.tasks.word_decoding.pallier2025.train",
    )
    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module.build_training_summary is result_schema.build_training_summary


def test_four_vocabularies_share_one_validation_file_and_no_test_is_created(tmp_path):
    config = _config(output_dir=tmp_path / "seed-000")
    payload = _evaluation(
        config,
        canonical_vocabulary_result(20, manifest_sha256="v" * 64),
    )
    path = write_evaluation_result(config, payload)

    assert path == tmp_path / "seed-000" / "evaluation" / "val.json"
    assert set(json.loads(path.read_text(encoding="utf-8"))["vocabularies"]) == {
        "20",
        "50",
        "100",
        "150",
    }
    assert not (path.parent / "test.json").exists()


def test_canonical_retrieval_values_equal_existing_public_metrics():
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    words = ["甲", "乙", "甲"]
    metrics = fixed_vocabulary_retrieval_metrics(
        queries,
        targets,
        words,
        ["甲", "乙"],
        top_ks=(1, 10),
    )

    result = canonical_retrieval(metrics)

    assert result == {
        "top1": metrics["micro_recall_at_1"],
        "top10": metrics["micro_recall_at_10"],
        "macro_top1": metrics["macro_recall_at_1"],
        "macro_top10": metrics["macro_recall_at_10"],
        "median_rank": metrics["median_rank"],
        "mrr": metrics["mean_reciprocal_rank"],
    }


def test_canonical_vocabulary_result_keeps_internal_word_metrics_separate():
    metrics = {
        "vocabulary_size": 50,
        "observed_vocabulary_size": 47,
        "missing_vocabulary_words": ["x", "y", "z"],
        "micro_recall_at_10": 0.60,
        "macro_recall_at_10": 0.48,
        "internal_only_query_count": 1189,
        "internal_only_context_radius": 2,
        "internal_only_vocabulary_size": 50,
        "internal_only_observed_vocabulary_size": 47,
        "internal_only_missing_vocabulary_words": ["x", "y", "z"],
        "internal_only_micro_recall_at_10": 0.61,
        "internal_only_macro_recall_at_10": 0.45,
        "internal_only_median_rank": 7.0,
        "internal_only_mean_reciprocal_rank": 0.35,
    }

    block = canonical_vocabulary_result(50, retrieval_metrics=metrics)

    assert block["retrieval"]["macro_top10"] == 0.48
    assert block["internal_only"]["query_count"] == 1189
    assert block["internal_only"]["context_radius"] == 2
    assert block["internal_only"]["retrieval"] == {
        "top1": None,
        "top10": 0.61,
        "macro_top1": None,
        "macro_top10": 0.45,
        "median_rank": 7.0,
        "mrr": 0.35,
    }
    assert block["internal_only"]["support"] == block["support"]


def test_four_vocabulary_blocks_reuse_existing_retrieval_numbers():
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
    targets = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    words = ["甲", "乙", "甲"]
    manifests = {}
    for size in PRIMARY_VOCABULARY_SIZES:
        vocabulary = ["甲", "乙"] + [f"缺失{index}" for index in range(size - 2)]
        manifests[size] = {
            "vocabulary": vocabulary,
            "manifest_sha256": str(size) * 64,
        }
    results = evaluate_vocabulary_manifests(
        queries,
        targets,
        words,
        manifests,
    )
    expected = fixed_vocabulary_retrieval_metrics(
        queries,
        targets,
        words,
        manifests[20]["vocabulary"],
        top_ks=(1, 10),
        vocabulary_name="frozen_N20",
    )
    assert set(results) == set(PRIMARY_VOCABULARY_SIZES)
    assert results[20]["retrieval"] == canonical_retrieval(expected)
def test_missing_candidate_support_is_explicit():
    block = canonical_vocabulary_result(
        20,
        manifest_sha256="v" * 64,
        retrieval_metrics={
            "vocabulary_size": 20,
            "observed_vocabulary_size": 18,
            "missing_vocabulary_words": ["甲", "乙"],
        },
    )
    assert block["support"] == {
        "supported_candidate_count": 18,
        "missing_candidate_count": 2,
        "missing_words": ["甲", "乙"],
    }


@pytest.mark.parametrize(
    "budget,expected_type,expected_value,completed_updates",
        (("updates", "updates", 100, 82), ("epochs", "epochs", 12, 82)),
)
def test_training_summary_represents_real_budget_types(
    tmp_path,
    budget,
    expected_type,
    expected_value,
    completed_updates,
):
    config = _config(budget=budget, output_dir=tmp_path / budget)
    legacy = {
        "status": "training_completed",
        "selection_metric": "macro_recall_at_10",
        "best_score": 0.4,
        "best_epoch": 2,
        "history": [{"epoch": 1}, {"epoch": 2, "optimizer_updates": 75}],
        "optimizer_updates": completed_updates,
        "target_updates": 100 if budget == "updates" else None,
        "test_status": "locked_not_evaluated",
    }

    result = build_training_summary(config, legacy)

    assert result["schema_version"] == 1
    assert result["training"]["budget"] == {
        "type": expected_type,
        "value": expected_value,
    }
    assert result["training"]["completed_epochs"] == 2
    assert result["training"]["completed_updates"] == completed_updates
    assert result["selection"]["best_update"] == 75
    assert set(result["checkpoint"]) == {"best", "last"}


def test_run_manifest_has_schema_environment_and_two_config_hashes():
    config = _config()
    manifest = build_run_manifest(config, ["python", "train.py"])
    environment = manifest["environment"]
    assert manifest["schema_version"] == 1
    assert manifest["experiment"]["dataset"] == "fixture"
    assert manifest["resolved_config_sha256"] == resolved_config_sha256(config)
    assert manifest["scientific_config_sha256"] == scientific_config_sha256(config)
    assert isinstance(manifest["git_dirty"], bool)
    assert set(environment) == {
        "python_version",
        "torch_version",
        "numpy_version",
        "cuda_available",
        "cuda_version",
        "cudnn_version",
        "gpu_name",
        "mne_version",
        "transformers_version",
    }
    assert environment_provenance() == environment


def test_git_dirty_ignores_untracked_and_ignored_files(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("stable\n", encoding="utf-8")
    subprocess.run(["git", "add", ".gitignore", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)

    (tmp_path / "untracked.txt").write_text("local\n", encoding="utf-8")
    ignored = tmp_path / "ignored" / "artifact.bin"
    ignored.parent.mkdir()
    ignored.write_bytes(b"local")
    assert git_tracked_dirty(tmp_path) is False

    tracked.write_text("changed\n", encoding="utf-8")
    assert git_tracked_dirty(tmp_path) is True


def test_canonical_run_records_tracked_dirty_without_blocking(monkeypatch, tmp_path):
    config = _config(output_dir=tmp_path / "unused")
    monkeypatch.setattr("braindecoding.experiment.git_tracked_dirty", lambda *_: True)
    output_dir, manifest = initialize_run_directory(
        config,
        "train",
        output_root=tmp_path,
    )
    assert output_dir.is_dir()
    assert manifest["git_dirty"] is True


def test_scientific_hash_ignores_operational_paths_but_not_science(monkeypatch):
    left = _config(output_dir="D:/runs/one")
    left["text_embedding"] = {"model_name": "D:/models/mengzi-t5-base"}
    left["dataset"]["alignment_path"] = (
        "outputs/女声一小王子时间戳/alignment.xlsx"
    )
    right = copy.deepcopy(left)
    right["training"]["output_dir"] = "E:/other/runs/two"
    right["training"]["pretrained_brain_encoder_checkpoint"] = "E:/runs/best.pt"
    left["training"]["pretrained_brain_encoder_checkpoint"] = "D:/runs/best.pt"
    right["dataset"]["root"] = "E:/neuro/example"
    right["text_embedding"]["model_name"] = "E:/models/mengzi-t5-base"
    right["dataset"]["alignment_path"] = (
        "artifacts/alignments/chineseeeg2_littleprince/f1/alignment.xlsx"
    )
    right["provenance"] = {"host": "machine-b"}
    left["provenance"] = {"host": "machine-a"}

    assert resolved_config_sha256(left) != resolved_config_sha256(right)
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv("BRAINDECODING_MODEL_ROOT", "D:/models")
    left_scientific_sha = scientific_config_sha256(left)
    monkeypatch.setenv("BRAINDATA_ROOT", "E:/neuro")
    monkeypatch.setenv("BRAINDECODING_MODEL_ROOT", "E:/models")
    assert left_scientific_sha == scientific_config_sha256(right)

    changed = copy.deepcopy(left)
    changed["dataset"]["window_seconds"] = 3.0
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv("BRAINDECODING_MODEL_ROOT", "D:/models")
    assert scientific_config_sha256(left) != scientific_config_sha256(changed)


def test_audit_summary_uses_fixed_controls_and_marks_missing_as_not_run():
    summary = build_audit_summary(
        _config(),
        checkpoint={"path": "best.pt", "sha256": "c" * 64},
        query_set={"split": "val", "sha256": "q" * 64, "query_count": 10},
        controls={"clean": {"status": "completed", "top1": 0.2}},
    )
    assert summary["schema_version"] == 1
    assert summary["controls"]["clean"]["top1"] == 0.2
    assert summary["controls"]["temporal_shift"] == {"status": "not_run"}
    assert summary["controls"]["donor_following"] == {"status": "not_run"}


def test_audit_summary_uses_canonical_validation_path(tmp_path):
    config = _config(output_dir=tmp_path / "seed-000")
    summary = build_audit_summary(
        config,
        checkpoint={"path": "best.pt", "sha256": "c" * 64},
        query_set={"split": "val", "sha256": "q" * 64, "query_count": 10},
        controls={},
    )
    path = write_audit_summary(config, summary)
    assert path == tmp_path / "seed-000/audits/validation/summary.json"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1
