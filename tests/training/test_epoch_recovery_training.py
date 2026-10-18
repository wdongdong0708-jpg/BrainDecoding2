"""Check that interruption preserves learning and validation model selection."""
import copy
import random

import numpy as np
import pytest
import torch

from braindecoding.experiment import file_sha256
from braindecoding.training.epoch_recovery import (
    capture_random_state, process_matches, protect_committed_best,
    restore_boundary, restore_random_state, save_boundary,
)
from braindecoding.training.runtime import save_checkpoint, set_seed

pytestmark = pytest.mark.current_protocol


def modules():
    model = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Dropout(.25), torch.nn.Linear(8, 2))
    loss = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(loss.parameters()), lr=.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, loss, optimizer, scheduler, scaler


def update(items):
    model, loss, optimizer, scheduler, _ = items
    x = torch.randn(7, 5) + random.random() + np.random.normal()
    optimizer.zero_grad()
    loss(model(x)).square().mean().backward()
    optimizer.step()
    scheduler.step()


def test_interrupted_adam_dropout_randomness_and_schedule_are_identical(tmp_path):
    set_seed(7)
    items = modules()
    update(items)
    save_boundary(tmp_path, *items, identity={"config": "same"}, state={"next_epoch": 1}, best_hashes={})
    update(items)
    update(items)
    expected = [copy.deepcopy(x.state_dict()) for x in items[:2]]
    expected_lr = items[2].param_groups[0]["lr"]
    expected_random = (random.random(), np.random.normal(), torch.rand(5))
    set_seed(111)
    resumed = modules()
    state, hashes = restore_boundary(tmp_path, *resumed, identity={"config": "same"})
    assert state == {"next_epoch": 1} and hashes == {}
    update(resumed)
    update(resumed)
    for module, expected_state in zip(resumed[:2], expected):
        for key, tensor in module.state_dict().items():
            torch.testing.assert_close(tensor, expected_state[key], rtol=0, atol=0)
    assert resumed[2].param_groups[0]["lr"] == expected_lr
    assert random.random() == expected_random[0]
    assert np.random.normal() == expected_random[1]
    assert torch.equal(torch.rand(5), expected_random[2])


def test_uncommitted_best_model_rolls_back_with_selection_history(tmp_path):
    items = modules()
    save_checkpoint(tmp_path / "best.pt", {"epoch": 1, "weight": torch.ones(2)})
    hashes = {"best.pt": file_sha256(tmp_path / "best.pt")}
    save_boundary(tmp_path, *items, identity="frozen", state={"next_epoch": 1, "best_epoch": 1}, best_hashes=hashes)
    protect_committed_best(tmp_path, "best.pt", hashes)
    save_checkpoint(tmp_path / "best.pt", {"epoch": 2, "weight": torch.zeros(2)})
    state, restored = restore_boundary(tmp_path, *modules(), identity="frozen")
    assert state["best_epoch"] == 1
    assert restored == hashes
    assert file_sha256(tmp_path / "best.pt") == hashes["best.pt"]


def test_changed_identity_and_model_only_checkpoint_cannot_resume(tmp_path):
    items = modules()
    save_boundary(tmp_path, *items, identity="A", state={}, best_hashes={})
    with pytest.raises(ValueError, match="identical frozen inputs"):
        restore_boundary(tmp_path, *modules(), identity="B")
    save_checkpoint(tmp_path / "resume.pt", {"model_state": items[0].state_dict()})
    with pytest.raises(ValueError, match="full epoch checkpoint"):
        restore_boundary(tmp_path, *modules(), identity="A")


def test_missing_rollback_and_pid_reuse_are_rejected(tmp_path):
    items = modules()
    save_checkpoint(tmp_path / "best.pt", {"epoch": 1})
    hashes = {"best.pt": file_sha256(tmp_path / "best.pt")}
    save_boundary(tmp_path, *items, identity="A", state={}, best_hashes=hashes)
    save_checkpoint(tmp_path / "best.pt", {"epoch": 2})
    with pytest.raises(ValueError, match="No valid committed best"):
        restore_boundary(tmp_path, *modules(), identity="A")
    import os
    import psutil
    created = psutil.Process().create_time()
    assert process_matches({"pid": os.getpid(), "created": created})
    assert not process_matches({"pid": os.getpid(), "created": created - 1})
    assert not process_matches({"pid": -1, "created": created})


def test_random_state_payload_uses_restricted_loader_types():
    state = capture_random_state()
    restore_random_state(state)
    assert isinstance(state["numpy"][1], list)
