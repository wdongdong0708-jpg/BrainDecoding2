import numpy as np
import pandas as pd

from braindecoding.unified_sentence_v2 import canonical_group_rows, shifted_mapping
from braindecoding.unified_sentence_v2_results import holm, paired_block_statistics


def test_different_voice_segments_sharing_source_lines_stay_in_one_material_component():
    rows = []
    for subject, segments in (("s1", ((1, 2), (3,))), ("s2", ((1,), (2, 3)))):
        for segment, lines in enumerate(segments):
            for line in lines:
                rows.append(dict(protocol_group=f"{subject}-{segment}", subject_id=subject,
                    recording_id=subject, chapter=3, source_sentence_uid=f"{subject}|actual-line-{line}",
                    event_id=f"{subject}-{line}", normalized_word=str(line)))
    groups, materials = canonical_group_rows("chineseeeg2_littleprince", pd.DataFrame(rows))
    assert len(materials) == 1
    assert len({g["material"] for g in groups}) == 1
    assert [g["reference"] for g in groups] == [["1", "2"], ["3"], ["1"], ["2", "3"]]


def test_shift_preserves_subject_recording_availability_and_uses_no_reference():
    groups = []
    for subject in ("s1", "s2"):
        for recording in ("r1", "r2"):
            for segment in range(4):
                groups.append(dict(subject=subject, recording=recording,
                    event_ids=[f"{subject}-{recording}-{segment}-{i}" for i in range(4)]))
    events = [e for g in groups for e in g["event_ids"]]
    info = dict(groups=groups)
    cfg = dict(controls=dict(shifted_seed=123, shifted_offset_fraction=[.25, .75]))
    mapping = shifted_mapping(info, events, 0, cfg)
    classes = {e: (g["subject"], g["recording"], min(2, 3-i))
               for g in groups for i, e in enumerate(g["event_ids"])}
    assert len(set(mapping)) == len(events)
    assert all(i != j and classes[events[i]] == classes[events[j]] for i, j in enumerate(mapping))
    assert np.array_equal(mapping, shifted_mapping(info, events, 0, cfg))


def test_material_sign_flips_do_not_count_repeated_subjects_or_seeds_as_independent_blocks():
    rows = [dict(seed=seed, subject=subject, block=f"b{block}", scope="r", difference=.25)
            for seed in range(3) for subject in ("s1", "s2") for block in range(4)]
    cfg = dict(seed=1, paired_material_block_sign_flips=10000, material_block_bootstrap_repetitions=1000)
    result = paired_block_statistics(rows, "difference", cfg)
    assert result["effect"] == result["ci_lower"] == result["ci_upper"] == .25
    assert result["blocks"] == 4 and result["p_one_sided"] == 1/16
    assert np.allclose(holm([.01, .04, .2, .001]), [.03, .08, .2, .004])
