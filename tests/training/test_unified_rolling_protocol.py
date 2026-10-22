from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from braindecoding.unified_protocol_audit import build_training_vocabulary, load_source
from braindecoding.unified_rolling_training import equal_event_horizon_loss, full_vocabulary_metrics


def test_epoch_weighting_preserves_event_and_horizon_means_and_gradients():
    scale = torch.tensor(2.0, requires_grad=True)
    values = {0: torch.tensor([1., 3., 8.]), 1: torch.tensor([2., 10.]), 2: torch.tensor([7.])}
    totals = {d: len(v) for d, v in values.items()}
    # Deliberately uneven batches and a missing horizon in one batch.
    partitions = [{0: [0, 1], 1: [0], 2: [0]}, {0: [2], 1: [1]}]
    batch_losses = []
    for batch in partitions:
        losses = {d: (scale * values[d][ids]).mean() for d, ids in batch.items()}
        batch_losses.append(equal_event_horizon_loss(losses, {d: len(ids) for d, ids in batch.items()}, totals, 2))
    actual = torch.stack(batch_losses).mean()
    expected = torch.stack([(scale*v).mean() for v in values.values()]).mean()
    assert float(actual.detach()) == pytest.approx(float(expected.detach()))
    actual_grad = torch.autograd.grad(actual, scale)[0]
    expected_grad = torch.autograd.grad(expected, scale)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_vocabulary_counts_material_once_and_rejects_validation_inputs():
    table = pd.DataFrame({"split": ["train"]*4, "material_event": ["m1", "m1", "m2", "m2"],
        "normalized_word": ["猫", "猫", "chat", "chat"]})
    words, counts = build_training_vocabulary(table)
    assert words == ["chat", "猫"]
    assert counts == {"chat": 1, "猫": 1}
    table.loc[0, "split"] = "val"
    with pytest.raises(ValueError, match="train only"):
        build_training_vocabulary(table)
    with pytest.raises(ValueError, match="train/val"):
        load_source("pallier2025", "test")


def test_full_vocab_evaluation_keeps_unseen_reference_as_error_and_subjects_equal():
    ds = SimpleNamespace(embedding_map={"a": np.array([1., 0.]), "b": np.array([0., 1.])},
        table=pd.DataFrame({"event_id": ["e1", "e2", "e3"], "subject_id": ["s1", "s1", "s2"]}))
    encoded = {"predictions": torch.tensor([[1., 0.]]*3), "words": ["a", "outside", "a"],
        "event_ids": ["e1", "e2", "e3"]}
    metrics = full_vocabulary_metrics(encoded, ds, ["a", "b"])
    assert metrics["reference_count"] == 3
    assert metrics["reference_in_vocabulary"] == pytest.approx(2/3)
    assert metrics["selection_score"] == .75


def test_common_horizon_selection_excludes_only_unavailable_positions():
    ds = SimpleNamespace(embedding_map={"a": np.array([1., 0.])},
        table=pd.DataFrame({"event_id": ["e1", "e2"], "subject_id": ["s", "s"]}))
    encoded = {"words": ["a", "outside"], "event_ids": ["e1", "e2"],
        "predictions": torch.tensor([[1., 0.]]*2),
        "revision_predictions": {d: torch.tensor([[1., 0.]]*2) for d in range(3)},
        "revision_availability_masks": {0: torch.tensor([True, True]),
            1: torch.tensor([True, False]), 2: torch.tensor([True, False])}}
    metrics = full_vocabulary_metrics(encoded, ds, ["a"])
    assert metrics["common_count"] == 1
    assert metrics["reference_count"] == 2
    assert metrics["selection_score"] == 1

