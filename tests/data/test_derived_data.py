from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from braindecoding.config import load_yaml_with_extends
from braindecoding.data import chineseeeg2
from braindecoding.data.derived import (
    CORE_LEGACY_COLUMNS,
    canonical_cache_paths,
    canonical_event_table,
    event_manifest_path,
    restore_legacy_columns,
    signal_cache_directory,
    validate_event_product,
    write_event_product,
)
from braindecoding.events import CORE_EVENT_COLUMNS
from braindecoding.experiment import resolve_experiment_config
from braindecoding.data import build as build_derived_data


def _event_table():
    table = pd.DataFrame(
        {
            "事件编号": ["e0", "e1"],
            "受试者": ["sub-01", "sub-01"],
            "记录编号": ["r0", "r0"],
            "材料编号": ["m0", "m0"],
            "划分单元": ["u0", "u0"],
            "记录内序号": [0, 1],
            "词": ["甲", "乙"],
            "标准词": ["甲", "乙"],
            "开始时间": [0.0, 2.0],
            "结束时间": [0.5, 2.5],
            "上下文编号": ["c0", "c0"],
            "数据划分": ["train", "train"],
            "是否可训练": [True, True],
            "排除原因": ["", ""],
            "event_id": ["e0", "e1"],
            "subject_id": ["sub-01", "sub-01"],
            "recording_id": ["r0", "r0"],
            "word": ["甲", "乙"],
            "normalized_word": ["甲", "乙"],
            "sentence_uid": ["c0", "c0"],
            "split": ["train", "train"],
            "is_trainable": [True, True],
            "exclusion_reason": ["", ""],
            "run": [1, 1],
        }
    )
    return table


def test_canonical_event_table_is_chinese_first_without_core_duplicates():
    source = _event_table()
    result = canonical_event_table(source, {"run": "运行编号"})
    assert tuple(result.columns[:14]) == CORE_EVENT_COLUMNS
    assert not set(CORE_LEGACY_COLUMNS).intersection(result.columns)
    assert list(result["事件编号"]) == list(source["event_id"])
    assert list(result["数据划分"]) == list(source["split"])


def test_legacy_columns_are_restored_in_memory_only():
    canonical = canonical_event_table(_event_table(), {"run": "运行编号"})
    restored = restore_legacy_columns(canonical, {"run": "运行编号"})
    for legacy, chinese in CORE_LEGACY_COLUMNS.items():
        assert restored[legacy].equals(restored[chinese])
    assert restored["run"].tolist() == [1, 1]
    assert "event_id" not in canonical


def test_unmapped_persisted_column_is_rejected():
    source = _event_table().assign(unknown_builder_value=1)
    with pytest.raises(ValueError, match="未定义的持久化字段"):
        canonical_event_table(source, {"run": "运行编号"})


def test_event_manifest_and_hash_are_stable(tmp_path):
    table = canonical_event_table(_event_table(), {"run": "运行编号"})
    event_path = tmp_path / "events.csv"
    kwargs = {
        "dataset": "fixture",
        "project_root": Path(__file__).resolve().parents[2],
        "source_raw_roots": ["X:/raw"],
        "artifact_dependencies": [],
        "context_definition": {"grouping": "fixture"},
        "window_contract": {"window_seconds": 1.0},
        "source_contract": {"split": "frozen"},
        "audit": {"test_brain_signal_opened": False},
    }
    _, manifest_path, first = write_event_product(
        table, event_path, manifest_kwargs=kwargs
    )
    first_bytes = event_path.read_bytes()
    _, _, second = write_event_product(table, event_path, manifest_kwargs=kwargs)
    assert event_path.read_bytes() == first_bytes
    assert first["event_table_sha256"] == second["event_table_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert validate_event_product(event_path, manifest_path) == second


def test_scoped_event_table_gets_scoped_manifest_name(tmp_path):
    path = tmp_path / "events_sub01-06.csv"
    assert event_manifest_path(path) == tmp_path / "manifest_sub01-06.json"
    assert event_manifest_path(tmp_path / "events.csv") == tmp_path / "manifest.json"


def test_signal_cache_subject_directory_is_explicit():
    path = signal_cache_directory(
        "data/cache/derived/example/signals/eeg_50hz",
        "derivatives/preprocessed/sub-03/run.vhdr",
        by_subject=True,
    )
    assert path.as_posix().endswith("eeg_50hz/sub-03")
    with pytest.raises(ValueError, match="无法从源记录路径解析受试者"):
        signal_cache_directory("data/cache/derived/signals", "run.vhdr", by_subject=True)


@pytest.mark.parametrize(
    ("dataset", "expected_event"),
    [
        ("chineseeeg2_littleprince", "events/events.csv"),
        ("pallier2025", "events/events.csv"),
    ],
)
def test_canonical_cache_paths_never_use_task_local_cache(dataset, expected_event):
    paths = canonical_cache_paths(dataset)
    assert paths["event_table"].replace("\\", "/").endswith(expected_event)
    assert all(str(value).startswith("data/cache/derived/") for value in paths.values())
    assert all("tasks/word_decoding" not in str(value) for value in paths.values())


def test_all_canonical_word_configs_resolve_to_derived(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "configs/word_decoding").rglob("*.yaml"))
    runnable = []
    for path in paths:
        config = load_yaml_with_extends(path)
        if "experiment" not in config:
            continue
        runnable.append(path)
        resolved = resolve_experiment_config(config)
        assert all(
            str(value).replace("\\", "/").startswith("data/cache/derived/")
            for value in resolved["cache"].values()
        ), path
    assert runnable


def test_removed_word_configs_and_task_local_caches_are_absent():
    root = Path(__file__).resolve().parents[2]
    assert not list((root / "configs").glob("ChineseEEG2_LittlePrince*.yaml"))
    assert not list((root / "configs").glob("SMN4Lang*.yaml"))
    assert not list((root / "configs").glob("LibriBrain100*.yaml"))
    for task in ("ChineseEEG2_LittlePrince", "SMN4Lang", "LibriBrain100"):
        assert not (root / "tasks" / "word_decoding" / task / "cache").exists()


def test_build_entry_has_no_legacy_cache_fallback():
    root = Path(__file__).resolve().parents[2]
    source = (root / "braindecoding/data/dataset_builder.py").read_text(encoding="utf-8")
    assert "tasks/word_decoding" not in source
    assert "best.pt" not in source
    assert "last.pt" not in source


def test_local_canonical_manifests_when_materialized():
    root = Path(__file__).resolve().parents[2]
    products = {
        "chineseeeg2_littleprince": "events.csv",
    }
    for dataset, filename in products.items():
        path = root / "derived" / dataset / "events" / filename
        if not path.exists():
            continue
        columns = pd.read_csv(path, nrows=0).columns
        assert tuple(columns[:14]) == CORE_EVENT_COLUMNS
        assert not set(CORE_LEGACY_COLUMNS).intersection(columns)
        manifest = validate_event_product(path)
        assert manifest["dataset"] == dataset
        assert manifest["event_count"] > 0


def _require_local_products(*paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        pytest.skip(f"本机尚未物化 integration 数据：{missing}")


def test_local_chineseeeg2_event_product_is_complete_and_loadable():
    root = Path(__file__).resolve().parents[2]
    canonical_path = root / "data/cache/derived/chineseeeg2_littleprince/events/events.csv"
    _require_local_products(canonical_path)
    manifest = validate_event_product(canonical_path)
    restored = chineseeeg2.载入事件表(canonical_path, trainable_only=False)
    assert len(restored) == manifest["event_count"]
    assert manifest["subjects"] == [f"sub-{index:02d}" for index in range(1, 9)]
    assert restored["event_id"].tolist() == restored["事件编号"].tolist()
    assert restored["split"].tolist() == restored["数据划分"].tolist()


def test_derived_tree_contains_no_model_checkpoint():
    root = Path(__file__).resolve().parents[2] / "derived"
    if not root.exists():
        return
    assert not list(root.rglob("*.pt"))


def test_dataset_manifest_validation_requires_complete_components(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(build_derived_data, "PROJECT_ROOT", tmp_path)
    component_paths = []
    for name in ("events", "signals", "text"):
        path = tmp_path / f"{name}.json"
        path.write_text(f'{{"component": "{name}"}}\n', encoding="utf-8")
        component_paths.append((name, path))
    payload = {
        "schema_version": 1,
        "dataset": "fixture",
        "status": "complete",
        "components": {
            name: {
                "status": "complete",
                "manifest": path.name,
                "sha256": build_derived_data.sha256_file(path),
            }
            for name, path in component_paths
        },
    }
    payload["manifest_sha256"] = build_derived_data.stable_sha256(payload)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    assert build_derived_data.validate_dataset_manifest(manifest_path)[
        "status"
    ] == "complete"
    payload["components"]["text"]["status"] = "incomplete"
    payload["manifest_sha256"] = build_derived_data.stable_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="组件未完成"):
        build_derived_data.validate_dataset_manifest(manifest_path)


def test_check_cli_is_read_only(monkeypatch, capsys):
    checked = []
    monkeypatch.setattr(
        build_derived_data,
        "check_dataset",
        lambda dataset: checked.append(dataset)
        or {"dataset": dataset, "status": "complete"},
    )
    monkeypatch.setattr(
        build_derived_data,
        "build_dataset",
        lambda *args, **kwargs: pytest.fail("--check 不应进入构建路径"),
    )
    build_derived_data.main(["--dataset", "chineseeeg2_littleprince", "--check"])
    assert checked == ["chineseeeg2_littleprince"]
    assert '"status": "complete"' in capsys.readouterr().out


def test_check_cli_rejects_build_flags():
    with pytest.raises(SystemExit):
        build_derived_data.main(
            ["--dataset", "chineseeeg2_littleprince", "--check", "--signals"]
        )


@pytest.mark.parametrize(
    ("dataset", "expected_recordings", "expected_channels"),
    [
        ("chineseeeg2_littleprince", 216, 128),
    ],
)
def test_local_complete_derived_dataset_passes_common_check(
    dataset, expected_recordings, expected_channels, monkeypatch
):
    root = Path(__file__).resolve().parents[2]
    _require_local_products(root / "derived" / dataset / "manifest.json")
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )
    result = build_derived_data.check_dataset(dataset)
    assert result["status"] == "complete"
    assert result["signal_coverage"] == {
        "expected_recordings": expected_recordings,
        "signal_products": expected_recordings,
        "missing": 0,
        "extra": 0,
        "channel_count": expected_channels,
        "trainable_windows_in_range": True,
    }
    assert result["test_model_evaluation_performed"] is False
