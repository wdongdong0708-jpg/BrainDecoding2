import ast
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from braindecoding.training import runtime, word
from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import (
    train as chineseeeg2_training,
)
from braindecoding.tasks.word_decoding.pallier2025 import train as legacy_training


class _TinyWordDataset(Dataset):
    """只提供公共批处理函数需要的最小数据合同。"""

    def __init__(self):
        self.table = pd.DataFrame(
            {"sentence_uid": ["第一组", "第一组", "第二组", "第三组"]}
        )

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        return {
            "meg": torch.tensor([float(index), float(index + 1)]),
            "text_embedding": torch.tensor([float(index + 2), float(index + 3)]),
            "subject_index": torch.tensor(0),
            "sentence_index": torch.tensor(index),
            "word": f"词{index}",
            "event_id": f"事件{index}",
            "recording_id": f"记录{index // 2}",
        }


class _TinyBrainModel(nn.Module):
    """复现公共训练函数签名的最小线性模型。"""

    def __init__(self):
        super().__init__()
        self.brain_encoder = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.brain_encoder.weight.copy_(torch.eye(2))

    def forward(self, meg, subject_indices, sentence_indices):
        del subject_indices, sentence_indices
        return self.brain_encoder(meg)


def test_set_seed_repeats_python_numpy_and_torch_randomness():
    """同一随机种子必须同时复现三个随机数源。"""
    legacy_training.set_seed(1729)
    first = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )
    legacy_training.set_seed(1729)
    second = (
        random.random(),
        np.random.random(4),
        torch.rand(4),
    )

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_choose_device_keeps_auto_and_explicit_cpu_semantics(monkeypatch):
    """设备选择的默认值、返回类型和 CUDA 错误保持不变。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert legacy_training.choose_device("auto") == torch.device("cpu")
    assert legacy_training.choose_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="配置要求 CUDA"):
        legacy_training.choose_device("cuda")


def test_event_times_are_moved_only_for_models_that_require_them():
    batch = {"onset_seconds": torch.tensor([0.1, 0.35], dtype=torch.float64)}
    ordinary = type("OrdinaryModel", (), {"requires_event_times": False})()
    timed = type("TimedModel", (), {"requires_event_times": True})()

    assert word.event_times_for_model(ordinary, batch, torch.device("cpu")) is None
    moved = word.event_times_for_model(timed, batch, torch.device("cpu"))
    torch.testing.assert_close(moved, batch["onset_seconds"])
    assert moved.dtype == torch.float64

    with pytest.raises(KeyError, match="onset_seconds"):
        word.event_times_for_model(timed, {}, torch.device("cpu"))


def test_sentence_batch_sampler_locks_chunking_and_batch_boundaries():
    """超长组连续切块，且每个事件恰好出现一次。"""
    sentence_uids = ["甲"] * 5 + ["乙"] * 2 + ["丙"] * 3
    sampler = word.SentenceBatchSampler(
        sentence_uids, batch_size=3, shuffle=False, seed=11
    )
    batches = [list(batch) for batch in sampler]

    assert batches == [[0, 1, 2], [3, 4], [5, 6], [7, 8, 9]]
    assert len(sampler) == len(batches)
    assert all(len(batch) <= 3 for batch in batches)
    assert sorted(index for batch in batches for index in batch) == list(range(10))


def test_sentence_batch_sampler_locks_seed_and_epoch_order():
    """shuffle 顺序只由 seed 加 epoch 决定。"""
    sentence_uids = ["甲", "甲", "乙", "乙", "丙", "丙", "丁", "丁"]
    sampler = word.SentenceBatchSampler(
        sentence_uids, batch_size=2, shuffle=True, seed=7
    )

    assert [list(batch) for batch in sampler] == [
        [0, 1],
        [4, 5],
        [2, 3],
        [6, 7],
    ]
    sampler.set_epoch(1)
    assert [list(batch) for batch in sampler] == [
        [6, 7],
        [2, 3],
        [4, 5],
        [0, 1],
    ]


def test_sentence_batch_sampler_can_require_complete_context_groups():
    sampler = word.SentenceBatchSampler(
        ["甲"] * 3 + ["乙"] * 2,
        batch_size=5,
        shuffle=False,
        seed=0,
        preserve_complete_groups=True,
    )
    assert [list(batch) for batch in sampler] == [[0, 1, 2, 3, 4]]
    with pytest.raises(ValueError, match="不能跨批次"):
        word.SentenceBatchSampler(
            ["甲"] * 6,
            batch_size=5,
            shuffle=False,
            seed=0,
            preserve_complete_groups=True,
        )


def test_local_context_internal_mask_respects_radius():
    groups = torch.tensor([0] * 7 + [1] * 4)
    radius_one = word.local_context_internal_mask(groups, radius=1)
    radius_two = word.local_context_internal_mask(groups, radius=2)

    assert radius_one.tolist() == [
        False, True, True, True, True, True, False,
        False, True, True, False,
    ]
    assert radius_two.tolist() == [
        False, False, True, True, True, False, False,
        False, False, False, False,
    ]


def test_runtime_serialization_and_cpu_state_dict_keep_existing_contract(tmp_path):
    """JSON、checkpoint 和 CPU 权重快照保持现有读写语义。"""
    json_path = tmp_path / "summary.json"
    legacy_training.save_json(json_path, {"状态": "完成", "数值": 3})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "状态": "完成",
        "数值": 3,
    }

    model = nn.Linear(2, 2)
    snapshot = legacy_training.cpu_state_dict(model)
    expected = {key: value.clone() for key, value in snapshot.items()}
    with torch.no_grad():
        model.weight.add_(10)
    for key in expected:
        assert snapshot[key].device.type == "cpu"
        assert snapshot[key].grad_fn is None
        torch.testing.assert_close(snapshot[key], expected[key], rtol=0, atol=0)

    checkpoint_path = tmp_path / "checkpoint.pt"
    payload = {
        "format_version": 1,
        "model_state": snapshot,
        "metrics": {"score": 0.25},
    }
    legacy_training.save_checkpoint(checkpoint_path, payload)
    loaded = legacy_training.load_checkpoint(checkpoint_path)

    assert set(loaded) == set(payload)
    assert loaded["format_version"] == 1
    assert loaded["metrics"] == {"score": 0.25}
    assert not Path(str(checkpoint_path) + ".tmp").exists()
    restored = nn.Linear(2, 2)
    restored.load_state_dict(loaded["model_state"], strict=True)
    for key, value in snapshot.items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert legacy_training.parameter_count(restored) == 6


def test_make_loader_and_encode_loader_preserve_grouped_order():
    """公共 DataLoader 和编码结果保持组内顺序及元数据顺序。"""
    dataset = _TinyWordDataset()
    loader, sampler = legacy_training.make_loader(
        dataset,
        {"batch_size": 2, "seed": 13, "num_workers": 0},
        shuffle=False,
    )

    assert [list(batch) for batch in sampler] == [[0, 1], [2, 3]]
    model = _TinyBrainModel()
    encoded = legacy_training.encode_loader(
        model, loader, torch.device("cpu"), amp=False
    )

    expected_predictions = torch.stack(
        [dataset[index]["meg"] for index in range(len(dataset))]
    )
    expected_targets = torch.stack(
        [dataset[index]["text_embedding"] for index in range(len(dataset))]
    )
    torch.testing.assert_close(encoded["predictions"], expected_predictions)
    torch.testing.assert_close(encoded["targets"], expected_targets)
    assert encoded["words"] == [f"词{index}" for index in range(4)]
    assert encoded["event_ids"] == [f"事件{index}" for index in range(4)]
    assert encoded["recording_ids"] == ["记录0", "记录0", "记录1", "记录1"]


def test_train_one_epoch_keeps_single_batch_update_semantics():
    """锁定公共 epoch 训练函数的一次 CPU 参数更新。"""
    model = _TinyBrainModel()
    loss_module = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    batch = {
        "meg": torch.eye(2),
        "text_embedding": torch.zeros(2, 2),
        "subject_index": torch.zeros(2, dtype=torch.long),
        "sentence_index": torch.arange(2),
    }

    loss = legacy_training.train_one_epoch(
        model,
        loss_module,
        [batch],
        optimizer,
        scaler,
        torch.device("cpu"),
        {"amp": False, "max_grad_norm": 0.0},
    )

    assert loss == pytest.approx(0.5)
    torch.testing.assert_close(
        model.brain_encoder.weight,
        0.95 * torch.eye(2),
        rtol=0,
        atol=1e-7,
    )


def test_three_training_entries_use_the_public_training_modules():
    """活动数据集入口都复用新的公共实现，而不是互相导入。"""
    runtime_names = (
        "set_seed",
        "choose_device",
        "save_json",
        "limit_rows",
        "cpu_state_dict",
        "save_checkpoint",
        "load_checkpoint",
        "parameter_count",
    )
    for training in (legacy_training, chineseeeg2_training):
        for name in runtime_names:
            assert getattr(training, name) is getattr(runtime, name)
        assert training.make_loader is word.make_loader

    for name in ("move_batch", "train_one_epoch", "encode_loader"):
        assert getattr(legacy_training, name) is getattr(word, name)


def test_other_datasets_do_not_import_libribrain_training_as_a_library():
    """静态锁定数据集训练入口之间不存在反向依赖。"""
    project_root = Path(__file__).resolve().parents[2]
    paths = (
        project_root / "braindecoding/tasks/word_decoding/pallier2025/train.py",
        project_root
        / "braindecoding/tasks/word_decoding/chineseeeg2_littleprince/train.py",
    )
    forbidden = "braindecoding.tasks.word_decoding.libribrain100.train"
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert forbidden not in imported_modules

