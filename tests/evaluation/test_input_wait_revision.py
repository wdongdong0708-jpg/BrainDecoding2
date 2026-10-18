"""Timing endpoints, frozen schedules, and independent timing denominators."""
import numpy as np
import pandas as pd
import pytest

from braindecoding.c1_input_wait_test_v2 import describe, pallier_bounds, timing_for_group, verify_schedule


pytestmark = pytest.mark.current_protocol


def test_overlapping_windows_do_not_double_count_one_second():
    timing = timing_for_group([10., 10.2, 10.9], [11., 11.2, 11.9], 1)
    np.testing.assert_allclose(timing["extra"], [.2, .7, 0.])
    np.testing.assert_allclose(timing["final"], [1.2, 1.7, 1.])
    np.testing.assert_array_equal(timing["windows"], [1, 1, 0])
    c = timing_for_group([10., 10.2, 10.9], [11., 11.2, 11.9], 2)
    np.testing.assert_allclose(c["extra"], [.9, .7, 0.])
    np.testing.assert_array_equal(c["freeze"], [2, 2, 2])


def test_pause_is_not_capped_to_one_second_and_singleton_does_not_wait():
    timing = timing_for_group([0., 6.16], [1., 7.16], 1)
    assert timing["extra"][0] == pytest.approx(6.16)
    assert timing["final"][0] == pytest.approx(7.16)
    singleton = timing_for_group([0.], [1.], 1)
    assert singleton["extra"].tolist() == [0.]
    assert singleton["windows"].tolist() == [0]


def test_simultaneous_windows_still_count_one_revision_opportunity():
    timing = timing_for_group([0., 0., .2], [1., 1., 1.2], 1)
    assert timing["windows"].tolist() == [1, 1, 0]
    assert timing["extra"][0] == 0


@pytest.mark.parametrize("onsets,ends", [([1., 0.], [2., 1.]), ([0., 1.], [2., 1.5]),
                                         ([float('nan')], [1.]), ([2.], [1.])])
def test_invalid_timelines_are_not_silently_sorted(onsets, ends):
    with pytest.raises(ValueError):
        timing_for_group(onsets, ends, 1)


def test_bids_recording_origin_and_half_open_stop_are_used():
    clock = dict(source_first_samp=66000, source_sampling_rate_hz=1000., output_sampling_rate_hz=50.,
                 output_first_samp=3300, source_recording_start_seconds=66.,
                 output_recording_start_seconds=66., output_sample_count=2000)
    cfg = dict(source_sampling_rate_hz=1000., target_sampling_rate_hz=50.,
               window_start_offset_seconds=0., window_seconds=1.)
    starts, stops, ends = pallier_bounds([66.175, 66.545], clock, cfg)
    np.testing.assert_array_equal(starts, [9, 27])
    np.testing.assert_array_equal(stops, [59, 77])
    np.testing.assert_allclose(ends, [67.18, 67.54])


def fixture_schedule():
    groups = dict(d=dict(groups=[dict(group="g", subject="s", event_ids=["e0", "e1"])]))
    rows = []
    for condition, maximum in (("C1", 1), ("C", 2)):
        for seed in (0, 1, 2):
            for i, event in enumerate(("e0", "e1")):
                j = min(i+maximum, 1)
                rows.append(dict(dataset="d", condition=condition, seed=seed, event_id=event, group="g", subject="s",
                                 position=i, first_arrival=i, freeze_arrival=j, wait_subsequent_word_windows=j-i,
                                 allowed_revision_opportunities=j-i, actual_text_changes=j-i))
    return groups, pd.DataFrame(rows)


def test_every_seed_schedule_must_cover_every_word_without_duplicates():
    groups, latency = fixture_schedule()
    assert verify_schedule(latency, groups)["word_position_seed_condition_decisions"] == 12
    for altered in (latency.iloc[:-1], pd.concat([latency, latency.iloc[:1]])):
        with pytest.raises(ValueError):
            verify_schedule(altered, groups)
    latency.loc[0, "freeze_arrival"] = 0
    with pytest.raises(ValueError, match="freeze_arrival"):
        verify_schedule(latency, groups)


def test_pooled_and_subject_equal_means_are_explicitly_distinct():
    frame = pd.DataFrame(dict(subject=["a", "a", "b"], wait=[0., 0., 3.]))
    stats = describe(frame, "wait")
    assert stats["n"] == 3
    assert stats["mean_seconds"] == 1.
    assert stats["subject_equal_mean_seconds"] == 1.5
    assert stats["p95_seconds"] == pytest.approx(2.7)
