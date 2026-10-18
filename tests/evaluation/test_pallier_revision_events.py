"""Checks for paired rolling-revision event accounting."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from braindecoding.pallier_revision_event_exports import count_transitions
from scripts.paper.rolling_revision_event_panel import load_data


def test_count_transitions_partitions_same_events():
    result = count_transitions(
        np.array([False, False, True, True]),
        np.array([True, False, False, True]),
    )
    assert result == {
        "eligible_word_events": 4, "wrong_to_right": 1,
        "right_to_wrong": 1, "right_to_right": 1,
        "wrong_to_wrong": 1,
    }


def test_count_transitions_rejects_unpaired_inputs():
    with pytest.raises(ValueError, match="aligned"):
        count_transitions(np.array([True]), np.array([True, False]))


def test_panel_loader_checks_count_and_metric_consistency(tmp_path):
    path = tmp_path / "transitions.csv"
    rows = []
    for seed in (0, 1):
        rows.extend([
            {"dataset": "Pallier2025", "split": "val", "vocabulary_size": 50,
             "top_k": 10, "target_set": "common_delay2", "seed": seed,
             "from_delay": 0, "to_delay": 1, "eligible_word_events": 4,
             "wrong_to_right": 1, "right_to_wrong": 1,
             "right_to_right": 1, "wrong_to_wrong": 1,
             "macro_before": 0.5, "macro_after": 0.5,
             "micro_before": 0.5, "micro_after": 0.5},
            {"dataset": "Pallier2025", "split": "val", "vocabulary_size": 50,
             "top_k": 10, "target_set": "common_delay2", "seed": seed,
             "from_delay": 1, "to_delay": 2, "eligible_word_events": 4,
             "wrong_to_right": 1, "right_to_wrong": 0,
             "right_to_right": 2, "wrong_to_wrong": 1,
             "macro_before": 0.5, "macro_after": 0.75,
             "micro_before": 0.5, "micro_after": 0.75},
        ])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    metadata, curves, steps = load_data(path)
    assert metadata["dataset"] == "Pallier2025"
    assert curves[0] == {0: 0.5, 1: 0.5, 2: 0.75}
    assert steps[(1, 2)][1]["wrong_to_right"] == 1
    rows[0]["wrong_to_right"] = 2
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="partition"):
        load_data(path)
