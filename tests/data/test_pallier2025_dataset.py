from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from braindecoding.data import pallier2025
from braindecoding.data import build as build_derived_data
from braindecoding.events import CORE_EVENT_COLUMNS, validate_event_table


def _events(words):
    return pd.DataFrame(
        {
            "onset": [10.0 + index for index in range(len(words))],
            "duration": [0.25] * len(words),
            "trial_type": ["Word"] * len(words),
            "stimulus": words,
        }
    )


def _extra(words, *, onset_start=100.0):
    return pd.DataFrame(
        {
            "word": words,
            "onset": [onset_start + index for index in range(len(words))],
            "sequence_id": [4] * len(words),
            "is_last_word": [False] * (len(words) - 1) + [True],
            "pos": ["NC"] * len(words),
            "content_word": [True] * len(words),
            "n_closing": [1] * len(words),
            "_标准词": [pallier2025.normalize_french_word(word) for word in words],
        }
    )


def _header():
    return {"first_time_seconds": 0.0, "last_time_seconds": 1000.0}


def test_pallier_subject_discovery_and_ninety_recording_metadata(tmp_path):
    for subject in pallier2025.SUBJECTS:
        for run in pallier2025.RUNS:
            paths = pallier2025.recording_paths(tmp_path, subject, run)
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch(exist_ok=True)
    assert pallier2025.discover_subjects(tmp_path) == list(pallier2025.SUBJECTS)
    records = pallier2025.discover_recordings(tmp_path)
    assert len(records) == 90
    assert records[0]["subject"] == "sub-01"
    assert records[-1]["subject"] == "sub-10"


def test_pallier_config_uses_word_decoding_canonical_location(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    root = Path(__file__).resolve().parents[2]
    expected = root / "configs/word_decoding/pallier2025/base.yaml"
    assert build_derived_data._config_path("pallier2025") == expected
    assert expected.is_file()
    assert not (root / "configs/data/pallier2025.yaml").exists()
    config = build_derived_data.load_build_config("pallier2025")
    dataset = config["dataset"]
    assert dataset["window_start_offset_seconds"] == 0.0
    assert dataset["window_seconds"] == 1.0
    assert dataset["eligibility_window_seconds"] == 3.0
    assert dataset["baseline_seconds"] == 0.5
    assert dataset["clamp"] == 5.0
    assert config["text_embedding"] == {
        "model_name": "t5-large",
        "layer_fraction": 0.5,
        "token_aggregation": "mean",
        "embedding_dimension": 1024,
        "batch_size": 32,
        "device": "cpu",
        "local_files_only": True,
        "word_normalization": "canonical_standard_word",
    }


def test_frozen_run_split_is_seven_one_one_and_run_level():
    root = Path(__file__).resolve().parents[2]
    path = root / "data/manifests/pallier2025/run_split.json"
    split = pallier2025.load_run_split(path)
    assert [sum(value == name for value in split.values()) for name in ("train", "val", "test")] == [7, 1, 1]
    assert split == {
        "run-01": "train",
        "run-02": "train",
        "run-03": "train",
        "run-04": "train",
        "run-05": "train",
        "run-08": "train",
        "run-09": "train",
        "run-07": "val",
        "run-06": "test",
    }
    payload = json.loads(path.read_text(encoding="utf-8"))
    generated, ranking = pallier2025.deterministic_run_split()
    assert generated == split
    assert ranking == payload["hash_ranking"]
    assert payload["strategy"]["uses_word_labels"] is False
    assert payload["strategy"]["uses_vocabulary_support"] is False


@pytest.mark.parametrize(
    ("source", "expected"),
    [("  Étoile  ", "étoile"), ("J'avais", "j'avais"), ("arc-en-ciel", "arc-en-ciel")],
)
def test_french_normalization_preserves_accents_and_tokenization(source, expected):
    assert pallier2025.normalize_french_word(source) == expected


def test_frozen_word_window_contract_and_window_extraction():
    contract = pallier2025.word_window_contract()
    assert contract == {
        "version": "word_1s_support_3s_baseline_0p5_v1",
        "window_start_offset_seconds": 0.0,
        "window_seconds": 1.0,
        "eligibility_window_seconds": 3.0,
        "baseline_seconds": 0.5,
        "clamp": 5.0,
        "sampling_rate_hz": 50.0,
        "window_samples": 50,
        "baseline_samples": 25,
        "continuous_signal_includes_baseline": False,
        "continuous_signal_includes_clamp": False,
    }
    continuous = np.tile(np.arange(100, dtype=np.float32), (2, 1))
    before = continuous.copy()
    window = pallier2025.extract_word_window(
        continuous,
        0.0,
        {
            "source_first_samp": 0,
            "source_sampling_rate_hz": 1000.0,
            "output_sampling_rate_hz": 50.0,
        },
    )
    assert window.shape == (2, 50)
    assert window.dtype == np.float32
    assert np.allclose(window[:, :25].mean(axis=1), 0.0)
    assert float(window.min()) == -5.0
    assert float(window.max()) == 5.0
    assert np.array_equal(continuous, before)


def test_extra_info_blank_tokens_are_removed_before_alignment(tmp_path):
    path = tmp_path / "extra.tsv"
    path.write_text(
        "word,sequence_id,is_last_word,pos,content_word,n_closing,onset\n"
        "Le,1,False,DET,False,1,999\n"
        ",1,False,PUNCT,False,0,1000\n"
        "Prince,1,True,NC,True,2,1001\n",
        encoding="utf-8",
    )
    extra, blank_count = pallier2025._read_extra_info(path)
    assert blank_count == 1
    aligned, audit, excluded = pallier2025.align_recording_metadata(
        _events(["Le", "Prince"]),
        extra,
        subject="sub-01",
        run="run-01",
        blank_token_count=blank_count,
    )
    assert aligned["_标准词"].tolist() == ["le", "prince"]
    assert audit["edit_distance"] == 0
    assert excluded is False


def test_canonical_rows_use_bids_onset_and_fixed_chinese_columns():
    events = _events(["Étoile", "Prince"])
    extra = _extra(["Étoile", "Prince"], onset_start=900.0)
    table = pallier2025._recording_table(
        events,
        extra,
        subject="sub-01",
        run="run-07",
        split="val",
        header=_header(),
        recording_excluded=False,
        exclusion_reason="",
    )
    assert tuple(table.columns[:14]) == CORE_EVENT_COLUMNS
    assert table.columns.tolist() == list(pallier2025.PERSISTED_COLUMNS)
    assert table["开始时间"].tolist() == events["onset"].tolist()
    assert table["结束时间"].tolist() == [10.25, 11.25]
    assert "onset" not in table.columns
    assert table["上下文编号"].tolist() == [
        "pallier2025|run-07|sequence-004",
        "pallier2025|run-07|sequence-004",
    ]
    assert table["记录内序号"].tolist() == [0, 1]
    assert table["数据划分"].eq("val").all()
    validate_event_table(table)


def test_unapproved_extra_info_edit_blocks_build():
    with pytest.raises(ValueError, match="未经批准"):
        pallier2025.align_recording_metadata(
            _events(["un", "mot"]),
            _extra(["un", "autre"]),
            subject="sub-02",
            run="run-04",
            blank_token_count=0,
        )


def test_explicit_qc_mapping_preserves_bids_rows_but_excludes_recording():
    events = _events(["jour", "toujours", "le"])
    extra = _extra(["le", "jour", "toujours"])
    artifact = {
        "recording": {"subject": "sub-09", "run": "run-03"},
        "decision": "exclude_recording",
        "metadata_mapping_overrides": [
            {"bids_event_row": 2, "extra_word_index": 0}
        ],
    }
    aligned, audit, excluded = pallier2025.align_recording_metadata(
        events,
        extra,
        subject="sub-09",
        run="run-03",
        blank_token_count=0,
        qc_artifact=artifact,
    )
    assert aligned["_标准词"].tolist() == ["jour", "toujours", "le"]
    assert audit["edit_distance"] == 2
    assert excluded is True
    table = pallier2025._recording_table(
        events,
        aligned,
        subject="sub-09",
        run="run-03",
        split="train",
        header=_header(),
        recording_excluded=True,
        exclusion_reason="冻结 QC 排除",
    )
    assert table["词"].tolist() == events["stimulus"].tolist()
    assert not table["是否可训练"].any()
    assert table["排除原因"].eq("冻结 QC 排除").all()


def test_material_validation_ignores_only_fully_excluded_recording():
    first = pallier2025._recording_table(
        _events(["un", "mot"]),
        _extra(["un", "mot"]),
        subject="sub-01",
        run="run-01",
        split="train",
        header=_header(),
        recording_excluded=False,
        exclusion_reason="",
    )
    excluded = pallier2025._recording_table(
        _events(["autre", "texte"]),
        _extra(["autre", "texte"]),
        subject="sub-02",
        run="run-01",
        split="train",
        header=_header(),
        recording_excluded=True,
        exclusion_reason="冻结 QC 排除",
    )
    validate_event_table(pd.concat([first, excluded], ignore_index=True))
    partially_excluded = excluded.copy()
    partially_excluded.loc[0, "是否可训练"] = True
    partially_excluded.loc[0, "排除原因"] = ""
    with pytest.raises(ValueError, match="多个文本指纹"):
        validate_event_table(pd.concat([first, partially_excluded], ignore_index=True))


def test_pallier_data_module_has_no_checkpoint_or_model_evaluation_code():
    source = Path(pallier2025.__file__).read_text(encoding="utf-8")
    assert "torch.load" not in source
    assert "torch.save" not in source
    assert "preload=True" not in source


def test_local_pallier_event_product_and_cli_check(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    event_path = root / "data/cache/derived/pallier2025/events/events.csv"
    if not event_path.exists():
        pytest.skip("本机尚未物化 Pallier2025 阶段 2A 事件表。")
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    table = pallier2025.load_event_table(event_path)
    assert len(table) == 152560
    assert tuple(table.columns[:14]) == CORE_EVENT_COLUMNS
    assert table["记录编号"].nunique() == 90
    assert table["材料编号"].nunique() == 9
    bad = table[table["记录编号"].eq("pallier2025|sub-09|run-03")]
    assert len(bad) == 1860
    assert not bad["是否可训练"].any()
    event_manifest = build_derived_data.validate_event_product(event_path)
    assert event_manifest["event_count"] == 152560
    assert event_manifest["trainable_event_count"] == 150700
    assert len(event_manifest["audit"]["alignment_audits"]) == 90
    assert event_manifest["audit"]["alignment_summary"] == {
        "exact_recordings": 89,
        "approved_mismatch_recordings": 1,
        "total_substitutions": 0,
        "total_insertions": 2,
        "total_deletions": 2,
    }
    assert event_manifest["run_split"]["assignments"] == {
        "train": ["run-01", "run-02", "run-03", "run-04", "run-05", "run-08", "run-09"],
        "val": ["run-07"],
        "test": ["run-06"],
    }
    result = build_derived_data.check_dataset("pallier2025")
    assert result["status"] == "complete"
    assert result["components"]["events"] == "complete"
    assert result["components"]["signals"] == "complete"
    assert result["components"]["text"] == "complete"
    assert result["signal_coverage"] == {
        "expected_recordings": 90,
        "signal_products": 90,
        "missing": 0,
        "extra": 0,
        "channel_count": 306,
        "trainable_windows_in_range": True,
    }
    assert result["text_contract"]["word_count"] == 2426
    assert result["test_model_evaluation_performed"] is False


def test_local_pallier_event_table_did_not_drift_in_stage_2c():
    root = Path(__file__).resolve().parents[2]
    path = root / "data/cache/derived/pallier2025/events/events.csv"
    assert build_derived_data.sha256_file(path) == (
        "3cd1237fd138f3d827f8297cb7081243ce635ebb896f05241da8ccc5614a43fc"
    )


def test_material_view_counts_each_story_position_once_and_ignores_qc_recording():
    root = Path(__file__).resolve().parents[2]
    table = pallier2025.load_event_table(
        root / "data/cache/derived/pallier2025/events/events.csv"
    )
    material = pallier2025.material_word_view(table)
    assert len(material) == 15256
    assert not material.duplicated(["运行编号", "BIDS事件行号"]).any()
    assert material.groupby("运行编号").size().to_dict() == {
        "run-01": 1614,
        "run-02": 1768,
        "run-03": 1860,
        "run-04": 1641,
        "run-05": 1520,
        "run-06": 1845,
        "run-07": 1696,
        "run-08": 1516,
        "run-09": 1796,
    }
    run03 = material[material["运行编号"].eq("run-03")]
    assert run03["_material_source_subject"].eq("sub-01").all()


def test_pallier_protocol_assets_are_train_only_stable_and_split_independent():
    root = Path(__file__).resolve().parents[2]
    event_path = root / "data/cache/derived/pallier2025/events/events.csv"
    first = pallier2025.build_protocol_assets(event_path)
    second = pallier2025.build_protocol_assets(event_path)
    assert first == second
    for size in (20, 50, 100, 150):
        vocabulary = first[f"vocabulary_N{size}.json"]
        assert vocabulary["source_split"] == "train"
        assert vocabulary["created_from_test"] is False
        assert vocabulary["frequency_unit"] == "material_word_occurrence"
        assert vocabulary["source_split_units"] == [
            "pallier2025|run-01",
            "pallier2025|run-02",
            "pallier2025|run-03",
            "pallier2025|run-04",
            "pallier2025|run-05",
            "pallier2025|run-08",
            "pallier2025|run-09",
        ]
        assert len(vocabulary["vocabulary"]) == size
        ranked = [
            (word, vocabulary["word_counts"][word])
            for word in vocabulary["vocabulary"]
        ]
        assert ranked == sorted(ranked, key=lambda item: (-item[1], item[0]))
    support = first["vocabulary_support.json"]
    assert support["asset_type"] == "evaluation_vocabulary_support"
    assert support["support_audit_used_for_vocabulary_selection"] is False


def test_local_pallier_text_product_is_reference_gated_and_reproducible():
    root = Path(__file__).resolve().parents[2]
    manifest = pallier2025.validate_text_manifest(
        root / "data/cache/derived/pallier2025/text/t5_large_layer_0_5/manifest.json"
    )
    assert manifest["word_type_count"] == 2426
    assert manifest["embedding_shape"] == [2426, 1024]
    assert manifest["dtype"] == "float32"
    assert manifest["reference_fixture"]["tokens"] == list(
        pallier2025.FRENCH_REFERENCE_TOKENS
    )
    assert manifest["reference_fixture"]["bitwise_equal"] is True
    assert manifest["reference_fixture"]["different_element_count"] == 0
    assert manifest["canonical_rebuild"]["bitwise_equal"] is True
    assert manifest["test_model_evaluation"] == "not_run"
    assert manifest["test_predictions_generated"] is False
    assert manifest["test_metrics_inspected"] is False


def test_pallier_active_experiments_include_scratch_and_warmstart_context():
    root = Path(__file__).resolve().parents[2]
    assert not list((root / "data/cache/derived/pallier2025").rglob("*.pt"))
    config_root = root / "configs/word_decoding/pallier2025"
    assert sorted(path.name for path in config_root.glob("*.yaml")) == ["base.yaml"]
    assert [path.name for path in config_root.rglob("main_word.yaml")] == [
        "main_word.yaml"
    ]
    assert [path.name for path in config_root.rglob("main_context.yaml")] == [
        "main_context.yaml"
    ]
    assert [
        path.name for path in config_root.rglob("main_context_warmstart.yaml")
    ] == ["main_context_warmstart.yaml"]
