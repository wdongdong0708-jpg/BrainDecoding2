"""历史基线配置兼容性，以及公共训练与只读报告层的行为合同。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from braindecoding import cli
from braindecoding.catalog import load_experiment
from braindecoding.optimizers import (
    build_adamw_for_modules,
    build_cosine_annealing_scheduler,
)
from braindecoding.results import render_run_report, write_run_report
from braindecoding.training.word import (
    brain_encoder_frozen_for_epoch,
    train_one_epoch,
    validation_patience_exhausted,
)


@pytest.fixture(autouse=True)
def configured_roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )


LEGACY_BASELINE_SCOPES = (
    "chineseeeg2_littleprince/sub01-08",
    "pallier2025/sub01-10",
)


@pytest.mark.legacy_compatibility
@pytest.mark.parametrize("scope", LEGACY_BASELINE_SCOPES)
@pytest.mark.parametrize("experiment", ("main_word", "main_context_warmstart"))
def test_legacy_baseline_uses_epoch_lifecycle(scope, experiment):
    _, config = load_experiment(f"{scope}/{experiment}")
    training = config["training"]
    assert training["epochs"] == 30
    assert training["patience"] == 10
    assert training["seed"] == 0
    assert config["model"]["use_transformer"] is (experiment == "main_context_warmstart")
    for key in (
        "max_updates", "minimum_updates_before_early_stopping", "scheduler_total_updates",
    ):
        assert key not in training


@pytest.mark.legacy_compatibility
@pytest.mark.parametrize("scope", LEGACY_BASELINE_SCOPES)
def test_legacy_context_uses_one_epoch_brain_only_warm_start(scope):
    _, config = load_experiment(f"{scope}/main_context_warmstart")
    training = config["training"]
    assert training["warm_start_from"] == "main_word"
    assert training["freeze_brain_encoder_epochs"] == 1
    assert "freeze_brain_encoder_updates" not in training
    assert config["experiment"]["id"] == "main_context_warmstart"


def test_epoch_scheduler_and_patience_contract():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = build_cosine_annealing_scheduler(optimizer, 30, 0.0)
    assert scheduler.T_max == 30
    assert brain_encoder_frozen_for_epoch(0, 1) is True
    assert brain_encoder_frozen_for_epoch(1, 1) is False
    assert validation_patience_exhausted(9, 10) is False
    assert validation_patience_exhausted(10, 10) is True


def test_optimizer_keeps_brain_parameters_across_epoch_freeze():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.brain_encoder = torch.nn.Linear(2, 2)
            self.context_transformer = torch.nn.Linear(2, 2)

    model = TinyModel()
    loss_parameter = torch.nn.Linear(2, 1)
    optimizer = build_adamw_for_modules(
        [model, loss_parameter], {"learning_rate": 1e-4, "weight_decay": 0.0}
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    brain_ids = {id(parameter) for parameter in model.brain_encoder.parameters()}
    assert brain_ids <= optimizer_ids
    model.brain_encoder.requires_grad_(False)
    model.brain_encoder.eval()
    assert brain_ids <= optimizer_ids
    model.brain_encoder.requires_grad_(True)
    model.brain_encoder.train()
    assert brain_ids <= optimizer_ids


def test_successful_optimizer_updates_are_counted_without_driving_scheduler():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.brain_encoder = self.linear

        def forward(self, meg, subject_indices, sentence_indices):
            return self.linear(meg)

    model = TinyModel()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = {
        "meg": torch.ones(3, 2),
        "text_embedding": torch.zeros(3, 2),
        "subject_index": torch.zeros(3, dtype=torch.long),
        "sentence_index": torch.zeros(3, dtype=torch.long),
    }
    _, updates, batches = train_one_epoch(
        model,
        loss,
        [batch, batch],
        optimizer,
        scaler,
        torch.device("cpu"),
        {"amp": False, "max_grad_norm": 0.0},
        return_stats=True,
    )
    assert updates == batches == 2


def test_local_window_revision_averages_three_independent_siglip_losses():
    class TinyRevisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
            self.brain_encoder = torch.nn.Identity()
            self.prefix_revision_enabled = True
            self.prefix_revision_horizons = (0, 1, 2)
            self.prefix_revision_separate_horizon_losses = True

        def forward_prefix_revision_training(
            self, meg, subject_indices, sentence_indices, cutoff_generator=None
        ):
            del meg, subject_indices, sentence_indices, cutoff_generator
            estimates = self.embedding[None].expand(6, -1)
            rows = torch.tensor([0, 1, 2, 3, 4, 5])
            delays = torch.tensor([0, 0, 1, 1, 2, 2])
            return estimates, rows, delays

    class RecordingRectangularLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.query_counts = []

        def forward_rectangular(self, estimates, candidates, rows):
            self.query_counts.append(len(estimates))
            return (estimates - candidates[rows]).square().mean()

    model = TinyRevisionModel()
    loss = RecordingRectangularLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = {
        "meg": torch.ones(6, 2),
        "text_embedding": torch.arange(12, dtype=torch.float32).reshape(6, 2),
        "subject_index": torch.zeros(6, dtype=torch.long),
        "sentence_index": torch.zeros(6, dtype=torch.long),
    }

    expected = torch.stack(
        [
            (
                model.embedding.detach()[None].expand(2, -1)
                - batch["text_embedding"][start : start + 2]
            )
            .square()
            .mean()
            for start in (0, 2, 4)
        ]
    ).mean()
    mean_loss, updates, batches, diagnostics = train_one_epoch(
        model,
        loss,
        [batch],
        optimizer,
        scaler,
        torch.device("cpu"),
        {"amp": False, "max_grad_norm": 0.0},
        return_stats=True,
        return_diagnostics=True,
    )

    assert updates == batches == 1
    assert mean_loss == pytest.approx(float(expected))
    assert loss.query_counts == [2, 2, 2]
    assert diagnostics["prefix_revision_delay_counts"] == {0: 2, 1: 2, 2: 2}
    assert set(diagnostics["prefix_revision_delay_losses"]) == {0, 1, 2}


def _report_config(output):
    return {
        "experiment": {
            "task": "word_decoding",
            "dataset": "fixture",
            "subject_scope": "sub01",
            "id": "main_context_warmstart",
            "category": "main",
        },
        "dataset": {"subjects": ["sub-01"], "window_seconds": 1.0},
        "model": {"use_transformer": True},
        "training": {
            "seed": 0,
            "epochs": 30,
            "warm_start_from": "main_word",
            "freeze_brain_encoder_epochs": 1,
            "output_dir": str(output),
        },
    }


def _write_report_fixtures(output):
    output.mkdir(parents=True)
    (output / "evaluation").mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({"scientific_config_sha256": "s" * 64}), encoding="utf-8"
    )
    (output / "training_summary.json").write_text(
        json.dumps(
            {
                "training": {
                    "budget": {"type": "epochs", "value": 30},
                    "completed_epochs": 12,
                    "completed_updates": 240,
                },
                "selection": {
                    "metric": "retrieval_acc10_vocab=fixture50_macro",
                    "best_value": 0.4,
                    "best_epoch": 2,
                    "best_update": 40,
                },
                "details": {
                    "stop_reason": "early_stopping",
                    "final_learning_rate": 0.0,
                },
                "test_status": "locked_not_evaluated",
            }
        ),
        encoding="utf-8",
    )
    vocabularies = {
        str(size): {
            "manifest_sha256": str(size) * 16,
            "support": {"supported_candidate_count": size},
            "retrieval": {
                "top1": 0.1,
                "top10": 0.2,
                "macro_top1": 0.3,
                "macro_top10": 0.4,
                "median_rank": 3.0,
                "mrr": 0.5,
            },
        }
        for size in (20, 50)
    }
    (output / "evaluation" / "val.json").write_text(
        json.dumps(
            {
                "data": {"event_table_sha256": "e" * 64},
                "checkpoint": {"sha256": "c" * 64},
                "vocabularies": vocabularies,
            }
        ),
        encoding="utf-8",
    )


def test_report_is_read_only_and_missing_assets_are_compact(tmp_path, monkeypatch):
    output = tmp_path / "seed-000"
    config = _report_config(output)
    _write_report_fixtures(output)
    before = {
        path: path.read_bytes()
        for path in output.rglob("*.json")
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("model loaded"))
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("neural data loaded"))
    markdown = render_run_report(config)
    assert "规范 JSON 文件仍是唯一机器真源" in markdown
    assert "| 100 | — |" in markdown
    assert "| 审计 | 未运行 |" in markdown
    assert "## 训练" in markdown
    assert "## 验证集结果" in markdown
    assert "## 验证集审计" in markdown
    assert "## 来源与指纹" in markdown
    assert "early_stopping" not in markdown
    assert "missing_true_class_support" not in markdown
    assert len(markdown) < 10_000
    assert before == {path: path.read_bytes() for path in output.rglob("*.json")}


def test_report_write_only_creates_summary_md(tmp_path):
    output = tmp_path / "seed-000"
    config = _report_config(output)
    _write_report_fixtures(output)
    path = write_run_report(config)
    assert path == output / "summary.md"
    assert path.is_file()
    assert "请勿手工编辑本文件" in path.read_text(encoding="utf-8")


def test_cli_report_prints_without_writing(tmp_path, monkeypatch, capsys):
    output = tmp_path / "seed-000"
    config = _report_config(output)
    _write_report_fixtures(output)
    record = {"identity": config["experiment"]}
    monkeypatch.setattr(cli, "resolve_selector", lambda selector: record)
    monkeypatch.setattr(cli, "_load_task_config", lambda value: (config, None, None))
    assert cli.main(["report", "fixture/sub01/main_context_warmstart"]) == 0
    assert "# 实验" in capsys.readouterr().out
    assert not (output / "summary.md").exists()


def test_cli_report_write_confirmation_uses_chinese(tmp_path, monkeypatch, capsys):
    output = tmp_path / "seed-000"
    config = _report_config(output)
    _write_report_fixtures(output)
    record = {"identity": config["experiment"]}
    monkeypatch.setattr(cli, "resolve_selector", lambda selector: record)
    monkeypatch.setattr(cli, "_load_task_config", lambda value: (config, None, None))
    assert cli.main(["report", "fixture/sub01/main_context_warmstart", "--write"]) == 0
    terminal = capsys.readouterr().out
    assert "已写入：" in terminal
    assert "written:" not in terminal
