"""CPU contracts for the accepted v2 worker; no GPU training is started."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from braindecoding import unified_rolling_v2_training as worker
from braindecoding.training.word import train_one_epoch

pytestmark = pytest.mark.current_protocol


@pytest.fixture(autouse=True)
def roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv("BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models")


@pytest.mark.parametrize("dataset", worker.DATASETS)
@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize("stage", worker.STAGES)
def test_resolved_rules_and_stage_sampler(dataset, seed, stage):
    cfg = worker.build_config(dataset, seed, stage)
    training = cfg["training"]
    assert training["seed"] == seed
    assert training["epochs"] == 30 and training["patience"] == 10
    assert training["learning_rate"] == 1e-4 and training["weight_decay"] == 0
    assert training["amp"] is True and training["batch_size"] == 128
    assert cfg["model"]["transformer"]["depth"] == 16
    assert cfg["model"]["transformer"]["heads"] == 16
    assert cfg["text_embedding"]["model_name"] == "BAAI/bge-m3"
    assert cfg["model"]["use_transformer"] == (stage == "revision")
    assert "loss_normalization" not in training
    assert worker.train_one_epoch is train_one_epoch
    assert f"seed-{seed:03d}" in str(worker.output_dir(dataset, seed, stage))
    if stage == "revision":
        assert training["preserve_complete_groups"] is True
        assert training["revision_cutoff_policy"] == "all"
        assert training["freeze_brain_encoder_epochs"] == 1
        assert training["warm_start_seed"] == seed
        assert str(worker.output_dir(dataset, seed, "word") / "best.pt") == training["pretrained_brain_encoder_checkpoint"]
    else:
        assert not training.get("preserve_complete_groups", False)
        assert worker.group_column(cfg) == "sentence_uid"
        assert "pretrained_brain_encoder_checkpoint" not in training


@pytest.mark.parametrize("dataset", worker.DATASETS)
@pytest.mark.parametrize("stage", worker.STAGES)
def test_matches_accepted_reference_model_loss_optimizer(dataset, stage):
    settings = worker.config()
    entry = next(item for item in settings["execution"]["reuse_runs"]
                 if item["dataset"] == dataset and item["stage"] == stage)
    reference_path = worker.project_path(entry["path"]) / "resolved_config.yaml"
    if not reference_path.exists():
        pytest.skip("Accepted development run is not present on this machine.")
    reference = worker.load_yaml_with_extends(reference_path)
    cfg = worker.build_config(dataset, 1, stage)
    for key in ("model", "loss", "text_embedding", "evaluation"):
        assert cfg[key] == reference[key]
    for key in worker.COMMON_TRAINING_KEYS:
        assert cfg["training"][key] == reference["training"][key]
    for key in ("epochs", "top_ks", "preserve_complete_groups", "revision_cutoff_policy", "freeze_brain_encoder_epochs"):
        assert cfg["training"].get(key) == reference["training"].get(key)


def _word_contract():
    cfg = worker.build_config("pallier2025", 1, "revision")
    train = SimpleNamespace(channel_names=["a", "b"], channel_positions=np.zeros((2, 2)))
    checkpoint = {
        "config": {"experiment": {"dataset": "pallier2025"},
                   "dataset": {"subjects": cfg["dataset"]["subjects"]}},
        "stage": "word", "seed": 1, "selection": "legacy50", "test_evaluated": False,
        "model_config": copy.deepcopy(cfg["model"]), "text_embedding_config": cfg["text_embedding"],
        "embedding_sha256": "expected", "channel_names": train.channel_names,
    }
    checkpoint["model_config"]["use_transformer"] = False
    return checkpoint, cfg, train


def test_legacy_pallier_subject_identity_fallback():
    checkpoint, cfg, train = _word_contract()
    worker.validate_word_checkpoint(checkpoint, cfg, train, "expected")


@pytest.mark.parametrize("change", ("seed", "dataset", "embedding", "selection", "subjects", "stage", "channels", "transformer"))
def test_rejects_incompatible_word_initialization(change):
    checkpoint, cfg, train = _word_contract()
    if change == "dataset":
        checkpoint["config"]["experiment"]["dataset"] = "chineseeeg2_littleprince"
    elif change == "embedding":
        checkpoint["embedding_sha256"] = "other"
    elif change == "subjects":
        checkpoint["config"]["dataset"]["subjects"] = ["other"]
    elif change == "channels":
        checkpoint["channel_names"] = ["b", "a"]
    elif change == "transformer":
        checkpoint["model_config"]["use_transformer"] = True
    else:
        checkpoint[change] = {"seed": 0, "selection": "full_vocab", "stage": "revision"}[change]
    with pytest.raises(ValueError, match="initialization contract"):
        worker.validate_word_checkpoint(checkpoint, cfg, train, "expected")


@pytest.mark.parametrize("change", ("status", "test", "loss", "capacity"))
def test_protocol_guard_blocks_retired_or_changed_rules(change):
    settings = worker.config()
    if change == "status":
        settings["protocol"]["status"] = "retired"
    elif change == "test":
        settings["protocol"]["splits_allowed"].append("test")
    elif change == "loss":
        settings["training"]["loss_normalization"] = "global_event_horizon"
    else:
        settings["model"]["transformer"]["depth"] = 4
    with pytest.raises(ValueError):
        worker.assert_ready(settings)


def test_test_split_is_rejected_before_loading():
    with pytest.raises(ValueError, match="Only train and validation"):
        worker.dataset_and_loader({}, "test")
