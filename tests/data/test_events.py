import numpy as np
import pandas as pd
import pytest
from torch.utils.data import Dataset

from braindecoding.events import (
    CORE_EVENT_COLUMNS,
    add_core_event_columns,
    align_text_sequences,
    compare_text_sequences,
    text_fingerprint,
    validate_event_table,
)
from braindecoding.training.word import make_loader
from braindecoding.data import chineseeeg2 as ChineseEEG2


EXPECTED_CORE_COLUMNS = (
    "事件编号",
    "受试者",
    "记录编号",
    "材料编号",
    "划分单元",
    "记录内序号",
    "词",
    "标准词",
    "开始时间",
    "结束时间",
    "上下文编号",
    "数据划分",
    "是否可训练",
    "排除原因",
)


def _base_table():
    """返回包含交错记录的最小旧事件表。"""
    return pd.DataFrame(
        {
            "event_id": ["e0", "e1", "e2", "e3", "e4"],
            "subject_id": ["sub-01", "sub-02", "sub-01", "sub-02", "sub-01"],
            "recording_id": ["record-a", "record-b", "record-a", "record-b", "record-a"],
            "word": ["甲", "乙", "丙", "丁", "戊"],
            "normalized_word": ["甲", "乙", "丙", "丁", "戊"],
            "sentence_uid": ["a", "b", "a", "b", "a"],
            "split": ["train", "val", "train", "val", "train"],
            "is_trainable": [True, True, False, True, True],
            "exclusion_reason": ["", "", "测试排除", "", ""],
        }
    )


def test_core_event_columns_are_the_frozen_public_contract():
    assert CORE_EVENT_COLUMNS == EXPECTED_CORE_COLUMNS


@pytest.mark.parametrize(
    ("module", "extra", "expected_material", "expected_start", "expected_stop"),
    [
        (
            ChineseEEG2,
            {
                "chapter": [1, 2, 1, 2, 1],
                "aligned_start_seconds": [0.1, 0.2, 0.3, 0.4, 0.5],
                "aligned_stop_seconds": [0.2, 0.3, 0.4, 0.5, 0.6],
                "voice_version": ["f1", "m1", "f1", "m1", "f1"],
            },
            [
                "ChineseEEG2|littleprince|chapter-01|voice-f1",
                "ChineseEEG2|littleprince|chapter-02|voice-m1",
                "ChineseEEG2|littleprince|chapter-01|voice-f1",
                "ChineseEEG2|littleprince|chapter-02|voice-m1",
                "ChineseEEG2|littleprince|chapter-01|voice-f1",
            ],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.2, 0.3, 0.4, 0.5, 0.6],
        ),
    ],
)
def test_dataset_contract_addition_preserves_old_columns_and_row_order(
    module, extra, expected_material, expected_start, expected_stop
):
    table = _base_table().assign(**extra)
    before = table.copy(deep=True)

    result = module.add_event_contract(table)

    pd.testing.assert_frame_equal(result.loc[:, before.columns], before)
    assert len(result) == len(before)
    assert result.index.equals(before.index)
    assert result["event_id"].tolist() == before["event_id"].tolist()
    assert result["split"].tolist() == before["split"].tolist()
    assert result["sentence_uid"].tolist() == before["sentence_uid"].tolist()
    assert result["材料编号"].tolist() == expected_material
    if module is ChineseEEG2:
        assert result["划分单元"].tolist() == [
            "ChineseEEG2|littleprince|chapter-01",
            "ChineseEEG2|littleprince|chapter-02",
            "ChineseEEG2|littleprince|chapter-01",
            "ChineseEEG2|littleprince|chapter-02",
            "ChineseEEG2|littleprince|chapter-01",
        ]
    else:
        assert result["划分单元"].tolist() == expected_material
    assert result["开始时间"].tolist() == pytest.approx(expected_start)
    assert result["结束时间"].tolist() == pytest.approx(expected_stop)
    assert result["记录内序号"].tolist() == [0, 0, 1, 1, 2]
    assert tuple(result.loc[:, CORE_EVENT_COLUMNS].columns) == CORE_EVENT_COLUMNS
    assert validate_event_table(result) == ()


def test_core_columns_copy_the_old_contract_row_by_row():
    table = _base_table()
    result = add_core_event_columns(
        table,
        material_ids=["material-a", "material-b", "material-a", "material-b", "material-a"],
        split_units=["material-a", "material-b", "material-a", "material-b", "material-a"],
        start_times=[0.1, 0.2, 0.3, 0.4, 0.5],
        end_times=[0.2, 0.3, 0.4, 0.5, 0.6],
    )

    direct = {
        "事件编号": "event_id",
        "受试者": "subject_id",
        "记录编号": "recording_id",
        "词": "word",
        "标准词": "normalized_word",
        "上下文编号": "sentence_uid",
        "数据划分": "split",
        "是否可训练": "is_trainable",
        "排除原因": "exclusion_reason",
    }
    for core, legacy in direct.items():
        pd.testing.assert_series_equal(
            result[core], table[legacy], check_names=False, check_dtype=True
        )


class _OrderDataset(Dataset):
    """只返回行号，用于锁定新增列前后的 loader 顺序。"""

    def __init__(self, table):
        self.table = table

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        return {"row": index}


def _loader_rows(table):
    loader, sampler = make_loader(
        _OrderDataset(table),
        {"batch_size": 3, "seed": 17, "num_workers": 0},
        shuffle=False,
    )
    return [list(batch) for batch in sampler], [
        int(value) for batch in loader for value in batch["row"]
    ]


def test_new_columns_do_not_change_sampler_groups_or_loader_order():
    table = _base_table()
    result = add_core_event_columns(
        table,
        material_ids=["m1", "m2", "m1", "m2", "m1"],
        split_units=["m1", "m2", "m1", "m2", "m1"],
        start_times=[0, 0, 1, 1, 2],
        end_times=[1, 1, 2, 2, 3],
    )

    assert _loader_rows(result) == _loader_rows(table)


def test_validation_rejects_strict_contract_violations():
    table = add_core_event_columns(
        _base_table(),
        material_ids=["m1", "m2", "m1", "m2", "m1"],
        split_units=["m1", "m2", "m1", "m2", "m1"],
        start_times=[0, 0, 1, 1, 2],
        end_times=[1, 1, 2, 2, 3],
    )

    duplicate = table.copy()
    duplicate.loc[1, "事件编号"] = duplicate.loc[0, "事件编号"]
    with pytest.raises(ValueError, match="事件编号.*唯一"):
        validate_event_table(duplicate)

    crossed = table.copy()
    crossed.loc[1, "划分单元"] = "m1"
    with pytest.raises(ValueError, match="划分单元.*数据划分"):
        validate_event_table(crossed)

    discontinuous = table.copy()
    discontinuous.loc[2, "记录内序号"] = 2
    with pytest.raises(ValueError, match="记录内序号"):
        validate_event_table(discontinuous)

    reversed_time = table.copy()
    reversed_time.loc[0, "结束时间"] = -1
    with pytest.raises(ValueError, match="开始时间.*结束时间"):
        validate_event_table(reversed_time)

    unknown_split = table.copy()
    unknown_split.loc[0, "数据划分"] = "validation"
    with pytest.raises(ValueError, match="train/val/test"):
        validate_event_table(unknown_split)


def test_validation_warns_but_does_not_fill_an_empty_exclusion_reason():
    table = add_core_event_columns(
        _base_table(),
        material_ids=["m1", "m2", "m1", "m2", "m1"],
        split_units=["m1", "m2", "m1", "m2", "m1"],
        start_times=[0, 0, 1, 1, 2],
        end_times=[1, 1, 2, 2, 3],
    )
    table.loc[2, "排除原因"] = ""

    with pytest.warns(UserWarning, match="不可训练事件.*排除原因"):
        messages = validate_event_table(table)

    assert len(messages) == 1
    assert table.loc[2, "排除原因"] == ""


def test_text_fingerprint_and_voice_comparison_are_stable():
    f1 = ["小", "王子", "回来"]
    m1 = ["小", "王子", "回来"]
    changed = ["小", "公主", "回来", "了"]

    assert text_fingerprint(f1) == text_fingerprint(tuple(f1))
    same = compare_text_sequences(f1, m1)
    assert same == {
        "identical": True,
        "first_length": 3,
        "second_length": 3,
        "difference_count": 0,
        "first_difference_index": None,
        "first_fingerprint": text_fingerprint(f1),
        "second_fingerprint": text_fingerprint(m1),
    }

    comparison = compare_text_sequences(f1, changed)
    assert comparison["identical"] is False
    assert comparison["first_length"] == 3
    assert comparison["second_length"] == 4
    assert comparison["difference_count"] == 2
    assert comparison["first_difference_index"] == 1
    assert comparison["first_fingerprint"] != comparison["second_fingerprint"]


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            ["小", "王子", "回来"],
            ["小", "王子", "回来"],
            dict(matches=3, substitutions=0, insertions=0, deletions=0, distance=0),
        ),
        (
            ["小", "王子", "回来"],
            ["小", "公主", "回来"],
            dict(matches=2, substitutions=1, insertions=0, deletions=0, distance=1),
        ),
        (
            ["小", "王子", "回来"],
            ["小", "王子", "已经", "回来"],
            dict(matches=3, substitutions=0, insertions=1, deletions=0, distance=1),
        ),
        (
            ["小", "王子", "已经", "回来"],
            ["小", "王子", "回来"],
            dict(matches=3, substitutions=0, insertions=0, deletions=1, distance=1),
        ),
        (
            ["小", "王子", "回来"],
            ["昨天", "小", "王子", "回来"],
            dict(matches=3, substitutions=0, insertions=1, deletions=0, distance=1),
        ),
    ],
)
def test_word_level_alignment_counts_minimal_edits(left, right, expected):
    result = align_text_sequences(left, right)

    assert result["equal"] is (expected["distance"] == 0)
    assert result["matches"] == expected["matches"]
    assert result["substitutions"] == expected["substitutions"]
    assert result["insertions"] == expected["insertions"]
    assert result["deletions"] == expected["deletions"]
    assert result["edit_distance"] == expected["distance"]
    assert result["normalized_edit_distance"] == pytest.approx(
        expected["distance"] / max(len(left), len(right), 1)
    )
    if expected["distance"] == 0:
        assert result["first_edit_left_index"] is None
        assert result["first_edit_right_index"] is None
    else:
        assert result["first_edit_left_index"] is not None
        assert result["first_edit_right_index"] is not None
    assert result["left_fingerprint"] == text_fingerprint(left)
    assert result["right_fingerprint"] == text_fingerprint(right)


def test_leading_insertion_is_not_miscounted_as_many_substitutions():
    result = align_text_sequences(
        ["小", "王子", "回来"], ["昨天", "小", "王子", "回来"]
    )

    assert result["matches"] == 3
    assert result["substitutions"] == 0
    assert result["insertions"] == 1
    assert result["edit_distance"] == 1
    assert result["first_edit_left_index"] == 0
    assert result["first_edit_right_index"] == 0


def test_chineseeeg2_voice_versions_have_distinct_materials_and_shared_split_unit():
    table = pd.DataFrame(
        {
            "event_id": [f"event-{index}" for index in range(8)],
            "subject_id": [
                "sub-01", "sub-01", "sub-02", "sub-02",
                "sub-05", "sub-05", "sub-06", "sub-06",
            ],
            "recording_id": [
                "f1-a", "f1-a", "f1-b", "f1-b",
                "m1-a", "m1-a", "m1-b", "m1-b",
            ],
            "chapter": [3] * 8,
            "voice_version": ["speaker-A"] * 4 + ["speaker-B"] * 4,
            "word": ["小", "王子"] * 4,
            "normalized_word": ["小", "王子"] * 4,
            "aligned_start_seconds": [0.0, 1.0] * 4,
            "aligned_stop_seconds": [0.5, 1.5] * 4,
            "sentence_uid": ["f1-a"] * 2 + ["f1-b"] * 2
            + ["m1-a"] * 2 + ["m1-b"] * 2,
            "split": ["train"] * 8,
            "is_trainable": [True] * 8,
            "exclusion_reason": [""] * 8,
        }
    )

    result = ChineseEEG2.add_event_contract(table)

    assert result["材料编号"].nunique() == 2
    assert result["划分单元"].nunique() == 1
    assert result.groupby("voice_version")["材料编号"].nunique().eq(1).all()
    assert result.groupby("voice_version")["材料编号"].first().to_dict() == {
        "speaker-A": "ChineseEEG2|littleprince|chapter-03|voice-speaker-A",
        "speaker-B": "ChineseEEG2|littleprince|chapter-03|voice-speaker-B",
    }
    assert result["划分单元"].iat[0] == "ChineseEEG2|littleprince|chapter-03"
    assert result["split"].tolist() == table["split"].tolist()
    assert result["event_id"].tolist() == table["event_id"].tolist()


def test_validation_rejects_multiple_fingerprints_for_one_material():
    table = pd.DataFrame(
        {
            "事件编号": ["a0", "a1", "b0", "b1"],
            "受试者": ["sub-01", "sub-01", "sub-02", "sub-02"],
            "记录编号": ["record-a", "record-a", "record-b", "record-b"],
            "材料编号": ["material"] * 4,
            "划分单元": ["unit"] * 4,
            "记录内序号": [0, 1, 0, 1],
            "词": ["小", "王子", "小", "公主"],
            "标准词": ["小", "王子", "小", "公主"],
            "开始时间": [0.0, 1.0, 0.0, 1.0],
            "结束时间": [0.5, 1.5, 0.5, 1.5],
            "上下文编号": ["a", "a", "b", "b"],
            "数据划分": ["train"] * 4,
            "是否可训练": [True] * 4,
            "排除原因": [""] * 4,
        }
    )

    with pytest.raises(ValueError, match="材料编号.*多个文本指纹"):
        validate_event_table(table)


def test_same_material_across_subjects_has_the_same_text_fingerprint():
    table = pd.DataFrame(
        {
            "材料编号": ["story-01"] * 6,
            "受试者": ["sub-01"] * 3 + ["sub-02"] * 3,
            "记录内序号": [0, 1, 2, 0, 1, 2],
            "标准词": ["我", "爱", "你", "我", "爱", "你"],
        }
    )
    fingerprints = {
        subject: text_fingerprint(frame.sort_values("记录内序号")["标准词"])
        for subject, frame in table.groupby("受试者", sort=False)
    }

    assert len(set(fingerprints.values())) == 1

