"""Checks for paired rolling-revision event accounting."""

from __future__ import annotations

import numpy as np
import pytest

from braindecoding.pallier_revision_event_exports import count_transitions


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
