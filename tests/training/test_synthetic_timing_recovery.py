import pandas as pd
import pytest

from braindecoding.synthetic_timing_recovery import smoke_segment

pytestmark = pytest.mark.current_protocol


def test_smoke_follows_logical_neighbors_in_interleaved_event_table():
    table = pd.DataFrame({"sentence_uid": ["s1", "s2"] * 4,
        "normalized_word": ["candidate", "x", "a", "b", "c", "d", "e", "f"],
        "event_id": [f"e{i}" for i in range(8)]})
    result = smoke_segment(table, ["candidate"], "sentence_uid")
    assert result.event_id.tolist() == ["e0", "e2", "e4"]
    assert result.sentence_uid.nunique() == 1


def test_smoke_does_not_cross_segment_boundary_for_future_words():
    table = pd.DataFrame({"sentence_uid": ["s1", "s1", "s2", "s2", "s2"],
                          "normalized_word": ["candidate", "x", "a", "b", "c"]})
    with pytest.raises(ValueError, match="No suitable"):
        smoke_segment(table, ["candidate"], "sentence_uid")
