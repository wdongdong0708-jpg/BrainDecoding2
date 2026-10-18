from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding import preflight
from braindecoding.catalog import (
    discover_experiment_configs,
    discover_runnable_experiment_configs,
    load_experiment,
)
from braindecoding.config import PROJECT_ROOT
from braindecoding.data import pallier2025
from braindecoding.data.text import save_text_embedding_cache
from braindecoding.experiment import run_directory, scientific_config_sha256
from braindecoding.models import BrainEmbeddingModel
from braindecoding.tasks.word_decoding.pallier2025 import evaluate, train
from braindecoding.training.word import make_loader


@pytest.fixture(autouse=True)
def configured_root(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")


def _pallier_selectors():
    return {
        record["selector"]
        for record in discover_experiment_configs()
        if record["identity"]["dataset"] == "pallier2025"
    }


def test_catalog_finds_only_pallier_v2_main_experiments():
    assert _pallier_selectors() == {
        "pallier2025/sub01-10/main_word",
        "pallier2025/sub01-10/main_context_warmstart",
    }


def test_catalog_discovers_runnable_pallier_experiments():
    selectors = {
        record["selector"]
        for record in discover_runnable_experiment_configs()
        if record["identity"]["dataset"] == "pallier2025"
    }
    assert selectors >= _pallier_selectors()


@pytest.mark.parametrize(
    ("selector", "use_transformer", "warm_start"),
    (
        ("pallier2025/sub01-10/main_word", False, False),
        ("pallier2025/sub01-10/main_context_warmstart", True, True),
    ),
)
def test_pallier_resolved_scientific_contract(selector, use_transformer, warm_start):
    _, config = load_experiment(selector)
    assert config["experiment"]["subject_scope"] == "sub01-10"
    assert tuple(config["dataset"]["subjects"]) == pallier2025.SUBJECTS
    assert config["model"]["use_transformer"] is use_transformer
    assert config["model"]["embedding_dimension"] == 1024
    assert config["model"]["transformer"]["depth"] == 16
    assert config["model"]["transformer"]["heads"] == 16
    assert config["dataset"]["max_context_words"] == 128
    assert config["dataset"]["window_seconds"] == 1.0
    assert config["dataset"]["eligibility_window_seconds"] == 3.0
    assert config["dataset"]["baseline_seconds"] == 0.5
    assert config["dataset"]["clamp"] == 5.0
    assert config["training"]["epochs"] == 30
    assert config["training"]["patience"] == 10
    for key in (
        "max_updates",
        "minimum_updates_before_early_stopping",
        "scheduler_total_updates",
    ):
        assert key not in config["training"]
    assert config["evaluation"]["selection_metric"] == (
        "retrieval_acc10_vocab=pallier2025_50_macro"
    )
    assert config["evaluation"]["vocabulary_sizes"] == [20, 50, 100, 150]
    if warm_start:
        assert config["training"]["warm_start_from"] == "main_word"
        assert config["training"]["pretrained_brain_encoder_checkpoint"].replace(
            "\\", "/"
        ).endswith(
            "outputs/word_decoding/pallier2025/sub01-10/main_word/seed-000/best.pt"
        )
        assert config["training"]["freeze_brain_encoder_epochs"] == 1
        assert "freeze_brain_encoder_updates" not in config["training"]
    else:
        for key in (
            "warm_start_from",
            "pretrained_brain_encoder_checkpoint",
            "freeze_brain_encoder_updates",
        ):
            assert key not in config["training"]
    if use_transformer:
        assert config["dataset"]["runtime_context_grouping"] == (
            pallier2025.RUNTIME_CONTEXT_GROUPING
        )
    else:
        assert "runtime_context_grouping" not in config["dataset"]
    assert run_directory(config) == (
        PROJECT_ROOT
        / "outputs"
        / "word_decoding"
        / "pallier2025"
        / "sub01-10"
        / config["experiment"]["id"]
        / "seed-000"
    )


def _event_rows():
    common = {
        "材料编号": "pallier2025|run-01",
        "划分单元": "pallier2025|run-01",
        "记录内序号": 0,
        "词": "Mot",
        "标准词": "mot",
        "开始时间": 1.0,
        "结束时间": 1.2,
        "上下文编号": "pallier2025|run-01|sequence-001",
        "数据划分": "train",
        "是否可训练": True,
        "排除原因": "",
        "运行编号": "run-01",
    }
    valid = {
        **common,
        "事件编号": "pallier2025|sub-10|run-01|word-0000",
        "受试者": "sub-10",
        "记录编号": "pallier2025|sub-10|run-01",
    }
    excluded = {
        **common,
        "事件编号": "pallier2025|sub-09|run-03|word-0000",
        "受试者": "sub-09",
        "记录编号": "pallier2025|sub-09|run-03",
        "运行编号": "run-03",
        "是否可训练": False,
        "排除原因": "冻结 QC 排除",
    }
    return pd.DataFrame([valid, excluded])


def test_word_dataset_uses_canonical_window_subject_and_context_contract(
    tmp_path, monkeypatch
):
    meg_dir = tmp_path / "meg"
    signal_path, sidecar_path = pallier2025._signal_product_paths(
        meg_dir, "sub-10", "run-01"
    )
    signal_path.parent.mkdir(parents=True)
    continuous = np.tile(np.arange(60, dtype=np.float32), (306, 1))
    np.save(signal_path, continuous)
    channel_names = [f"MEG{index:04d}" for index in range(306)]
    sidecar_path.write_text(
        json.dumps(
            {
                "recording_id": "pallier2025|sub-10|run-01",
                "source_first_samp": 1000,
                "source_sampling_rate_hz": 1000.0,
                "output_sampling_rate_hz": 50.0,
                "channel_count": 306,
                "channel_names": channel_names,
                "channel_names_sha256": pallier2025.CHANNEL_NAMES_SHA256,
                "dtype": "float32",
            }
        ),
        encoding="utf-8",
    )
    text_path = tmp_path / "embeddings.npz"
    text_config = {
        "model_name": "t5-large",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "embedding_dimension": 1024,
        "word_normalization": "canonical_standard_word",
    }
    save_text_embedding_cache(
        text_path,
        ["mot"],
        np.ones((1, 1024), dtype=np.float32),
        text_config,
    )
    monkeypatch.setattr(
        pallier2025, "channel_names_sha256", lambda names: pallier2025.CHANNEL_NAMES_SHA256
    )
    monkeypatch.setattr(
        pallier2025,
        "vectorview_channel_positions",
        lambda names, path=None: np.zeros((len(names), 2), dtype=np.float32),
    )
    dataset = pallier2025.Pallier2025WordDataset(
        _event_rows(),
        {
            "subjects": list(pallier2025.SUBJECTS),
            "target_sampling_rate_hz": 50.0,
            "window_seconds": 1.0,
            "eligibility_window_seconds": 3.0,
            "baseline_seconds": 0.5,
            "clamp": 5.0,
        },
        meg_dir,
        text_path,
        text_config,
    )
    assert len(dataset) == 1
    item = dataset[0]
    assert tuple(item["meg"].shape) == (306, 50)
    assert item["meg"].dtype.is_floating_point
    assert float(item["meg"].min()) == -5.0
    assert float(item["meg"].max()) == 5.0
    assert int(item["subject_index"]) == 9
    assert int(item["sentence_index"]) == 0
    assert dataset.table.loc[0, "sentence_uid"] == dataset.table.loc[0, "上下文编号"]
    assert dataset.table.loc[0, pallier2025.RUNTIME_CONTEXT_COLUMN] == (
        "pallier2025|sub-10|run-01|pallier2025|run-01|sequence-001"
    )
    assert item["text_embedding"].shape == (1024,)
    assert "sub-09" not in dataset.table["subject_id"].tolist()
    assert "mot" not in train.frozen_vocabulary(150)

    subset_config = dict(dataset.dataset_config, subjects=["sub-10"])
    subset = pallier2025.Pallier2025WordDataset(
        _event_rows(),
        subset_config,
        meg_dir,
        text_path,
        text_config,
        allow_subject_subset=True,
    )
    assert subset.subject_map == {"sub-10": 0}
    assert int(subset[0]["subject_index"]) == 0


def _runtime_context_rows():
    rows = []
    for subject in ("sub-01", "sub-02"):
        for order in range(3):
            rows.append(
                {
                    "受试者": subject,
                    "记录编号": f"pallier2025|{subject}|run-07",
                    "上下文编号": "pallier2025|run-07|sequence-001",
                    "记录内序号": order,
                }
            )
    return pd.DataFrame(rows)


def test_runtime_context_uid_is_recording_local_and_sampler_uses_it():
    runtime = pallier2025.add_runtime_context_uid(_runtime_context_rows())
    assert runtime["上下文编号"].nunique() == 1
    assert runtime[pallier2025.RUNTIME_CONTEXT_COLUMN].nunique() == 2
    grouped = runtime.groupby(pallier2025.RUNTIME_CONTEXT_COLUMN, sort=False)
    assert grouped["受试者"].nunique().max() == 1
    assert grouped["记录编号"].nunique().max() == 1

    class _Dataset:
        table = runtime.assign(sentence_uid=runtime["上下文编号"])

        def __len__(self):
            return len(self.table)

        def __getitem__(self, index):
            return int(index)

    _, sampler = make_loader(
        _Dataset(),
        {"batch_size": 128, "num_workers": 0, "seed": 0},
        shuffle=False,
        group_column=pallier2025.RUNTIME_CONTEXT_COLUMN,
    )
    assert [len(group) for group in sampler.groups] == [3, 3]


def test_runtime_context_rejects_non_increasing_order_and_oversized_groups():
    wrong_order = _runtime_context_rows().iloc[[1, 0]].copy()
    with pytest.raises(ValueError, match="未严格递增"):
        pallier2025.add_runtime_context_uid(wrong_order)
    oversized = pd.concat([_runtime_context_rows().iloc[:1]] * 129, ignore_index=True)
    oversized["记录内序号"] = np.arange(len(oversized))
    with pytest.raises(ValueError, match="超过 max_context_words=128"):
        pallier2025.add_runtime_context_uid(oversized)


def test_main_word_forward_is_bitwise_independent_of_group_indices():
    torch.manual_seed(7)
    model = BrainEmbeddingModel(
        channel_positions=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        subject_count=2,
        config={
            "embedding_dimension": 8,
            "use_transformer": False,
            "conv": {
                "merger_channels": 2,
                "merger_position_dimension": 8,
                "merger_per_subject": True,
                "merger_dropout": 0.0,
                "initial_linear": 4,
                "hidden": 4,
                "depth": 1,
                "kernel_size": 3,
                "dropout_input": 0.0,
                "convolution_dropout": 0.0,
                "batch_norm": False,
                "skip": False,
                "glu_every": 0,
                "temporal_attention_hidden": 4,
            },
        },
    ).eval()
    signals = torch.randn(4, 2, 12)
    subjects = torch.tensor([0, 0, 1, 1])
    canonical_groups = torch.zeros(4, dtype=torch.long)
    runtime_groups = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        before = model(signals, subjects, canonical_groups)
        after = model(signals, subjects, runtime_groups)
    assert torch.equal(before, after)


def test_real_train_and_validation_runtime_context_statistics():
    event_path = PROJECT_ROOT / "data/cache/derived/pallier2025/events/events.csv"
    expected = {
        "train": (11721, 9.836191451241362, 8.0, 54),
        "val": (1900, 8.926315789473684, 8.0, 30),
    }
    for split, values in expected.items():
        table = pallier2025.load_event_table(
            event_path,
            split=split,
            subjects=pallier2025.SUBJECTS,
            trainable_only=True,
        )
        stats = pallier2025.runtime_context_statistics(table, max_context_words=128)
        assert stats["context_count"] == values[0]
        assert stats["mean_words"] == pytest.approx(values[1])
        assert stats["median_words"] == values[2]
        assert stats["maximum_words"] == values[3]
        assert stats["cross_subject_groups"] == 0
        assert stats["cross_recording_groups"] == 0
        assert stats["groups_over_maximum"] == 0


def test_validation_query_set_and_frozen_data_assets_do_not_drift():
    event_path = PROJECT_ROOT / "data/cache/derived/pallier2025/events/events.csv"
    table = pallier2025.load_event_table(event_path)
    validation = table[
        table["数据划分"].eq("val") & table["是否可训练"].astype(bool)
    ]
    query_bytes = ("\n".join(validation["事件编号"].astype(str)) + "\n").encode(
        "utf-8"
    )
    assert len(validation) == 16960
    assert hashlib.sha256(query_bytes).hexdigest() == (
        "32660833db3943f7655a7d2c30b0b1e12178d40923306e68a46fb52326d7157f"
    )
    assert pallier2025.sha256_file(event_path) == preflight.PALLIER_EVENT_SHA256


def test_only_context_model_uses_runtime_sampler_grouping():
    _, word = load_experiment("pallier2025/sub01-10/main_word")
    _, warmstart = load_experiment(
        "pallier2025/sub01-10/main_context_warmstart"
    )
    assert train.loader_group_column(word) == "sentence_uid"
    assert train.loader_group_column(warmstart) == pallier2025.RUNTIME_CONTEXT_COLUMN
    assert warmstart["dataset"]["runtime_context_grouping"] == (
        pallier2025.RUNTIME_CONTEXT_GROUPING
    )


def test_training_vocabulary_is_not_limited_to_candidate_manifests():
    root = PROJECT_ROOT / "data" / "manifests" / "pallier2025"
    vocabularies = {
        size: json.loads((root / f"vocabulary_N{size}.json").read_text(encoding="utf-8"))
        for size in (20, 50, 100, 150)
    }
    assert len(train.frozen_vocabulary(50)) == 50
    _, config = load_experiment("pallier2025/sub01-10/main_word")
    assert config["dataset"]["training_vocabulary_policy"] == (
        "all_trainable_train_events_with_text_embedding"
    )
    assert all(payload["source_split"] == "train" for payload in vocabularies.values())


def test_pallier_preflight_contract_is_valid_for_all_initialization_conditions(
    tmp_path,
):
    event_manifest = json.loads(
        (PROJECT_ROOT / "data/cache/derived/pallier2025/events/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for selector in (
        "pallier2025/sub01-10/main_word",
        "pallier2025/sub01-10/main_context_warmstart",
    ):
        _, config = load_experiment(selector, output_root=tmp_path)
        contract = preflight._pallier_contract_status(config, event_manifest)
        assert contract["status"] == "valid"
        assert all(contract["checks"].values())
        assert contract["neural_arrays_loaded"] is False
        assert contract["model_loaded"] is False


def test_pallier_scientific_hashes_are_frozen():
    expected = {
        "pallier2025/sub01-10/main_word": "4ac0ff59831c4e15647806b2bd584935a9314534e3a31cde837af51309d5d242",
        "pallier2025/sub01-10/main_context_warmstart": "76ad3d30633d0f6aa3e59641d0f5295ac1ea7d415e85d3d896dd2080e238db70",
    }
    for selector, digest in expected.items():
        _, config = load_experiment(selector)
        assert scientific_config_sha256(config) == digest


def test_checkpoint_payload_carries_frozen_dataset_contract():
    class _State:
        @staticmethod
        def state_dict():
            return {"weight": torch.zeros(1)}

    _, config = load_experiment("pallier2025/sub01-10/main_word")
    payload = train.checkpoint_payload(
        _State(),
        _State(),
        config,
        2,
        {"retrieval_acc10_vocab=pallier2025_50_macro": 0.0},
        [f"MEG{index:04d}" for index in range(306)],
        np.zeros((306, 2), dtype=np.float32),
        optimizer_updates=100,
    )
    contract = payload["dataset_contract"]
    assert contract["event_table_sha256"] == preflight.PALLIER_EVENT_SHA256
    assert contract["recording_count"] == 90
    assert contract["channel_names_sha256"] == preflight.PALLIER_CHANNEL_SHA256
    assert contract["text_content_sha256"] == preflight.PALLIER_TEXT_CONTENT_SHA256
    assert contract["window_seconds"] == 1.0
    assert contract["eligibility_window_seconds"] == 3.0
    assert contract["baseline_seconds"] == 0.5
    assert set(contract["vocabulary_manifest_sha256"]) == {"20", "50", "100", "150"}


def test_pallier_warm_start_loads_only_main_word_brain_encoder(tmp_path):
    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.brain_encoder = torch.nn.Linear(2, 2)
            self.context_transformer = torch.nn.Linear(2, 2)

    _, word_config = load_experiment("pallier2025/sub01-10/main_word")
    _, context_config = load_experiment(
        "pallier2025/sub01-10/main_context_warmstart"
    )
    channel_names = [f"MEG{index:04d}" for index in range(306)]
    channel_positions = np.zeros((306, 2), dtype=np.float32)
    torch.manual_seed(1)
    source = _TinyModel()
    payload = train.checkpoint_payload(
        source,
        torch.nn.Linear(1, 1),
        word_config,
        5,
        {"retrieval_acc10_vocab=pallier2025_50_macro": 0.25},
        channel_names,
        channel_positions,
        optimizer_updates=5902,
    )
    checkpoint = tmp_path / "best.pt"
    torch.save(payload, checkpoint)

    torch.manual_seed(2)
    target = _TinyModel()
    transformer_before = {
        key: value.detach().clone()
        for key, value in target.context_transformer.state_dict().items()
    }
    audit = train.load_pretrained_brain_encoder(
        target,
        checkpoint,
        context_config,
        channel_names,
        channel_positions,
    )

    for key, value in source.brain_encoder.state_dict().items():
        torch.testing.assert_close(target.brain_encoder.state_dict()[key], value)
    for key, value in transformer_before.items():
        torch.testing.assert_close(target.context_transformer.state_dict()[key], value)
    assert audit["method"] == "cnn_only_best_checkpoint_brain_encoder_only"
    assert audit["source_epoch"] == 5
    assert audit["source_update"] == 5902
    assert audit["loaded_loss_state"] is False
    assert audit["loaded_transformer_state"] is False


def test_pallier_evaluator_has_no_test_option():
    with pytest.raises(SystemExit):
        evaluate.parse_args(["--split", "test"])


def test_derived_data_contains_no_checkpoint():
    assert not list((PROJECT_ROOT / "data/cache/derived/pallier2025").rglob("*.pt"))


def test_cross_subject_context_run_is_frozen_as_implementation_diagnostic():
    payload = json.loads(
        (
            PROJECT_ROOT
            / "data/manifests/pallier2025/implementation_diagnostics.json"
        ).read_text(encoding="utf-8")
    )
    diagnostic = payload["diagnostics"][0]
    assert diagnostic["status"] == "implementation_diagnostic"
    assert diagnostic["scientific_result_eligible"] is False
    assert diagnostic["reason"] == "cross_subject_context_mixing"
    assert diagnostic["main_word_affected"] is False
