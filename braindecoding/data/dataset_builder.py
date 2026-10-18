"""从第三方原始数据和长期 artifacts 构建 canonical derived 数据。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends, normalize_project_relative, project_path
from braindecoding.data import chineseeeg2, pallier2025
from braindecoding.data.derived import (
    canonical_cache_paths,
    event_manifest_path,
    sha256_file,
    stable_sha256,
    validate_event_product,
    write_event_product,
)
from braindecoding.experiment import resolve_experiment_config


DATASETS = (
    "chineseeeg2_littleprince",
    "pallier2025",
)


def _config_path(dataset: str) -> Path:
    if dataset == "chineseeeg2_littleprince":
        return PROJECT_ROOT / "configs/word_decoding/chineseeeg2_littleprince/sub01-08/main_context_warmstart.yaml"
    if dataset == "pallier2025":
        return PROJECT_ROOT / "configs/word_decoding/pallier2025/base.yaml"
    raise ValueError(f"未知数据集：{dataset}")


def load_build_config(dataset: str) -> dict:
    """读取数据集的冻结构建配置，并解析所有项目内路径。"""
    loaded = load_yaml_with_extends(_config_path(dataset))
    config = loaded if dataset == "pallier2025" else resolve_experiment_config(loaded)
    config["cache"].update(canonical_cache_paths(dataset))
    for key in ("event_table", "eeg_dir", "meg_dir", "text_embeddings"):
        if key in config["cache"]:
            config["cache"][key] = str(project_path(config["cache"][key]))
    for key in ("alignment_path", "layout_path"):
        if config["dataset"].get(key):
            config["dataset"][key] = str(project_path(config["dataset"][key]))
    for source in config["dataset"].get("actual_reading_sources", ()):
        if source.get("alignment_path"):
            source["alignment_path"] = str(project_path(source["alignment_path"]))
    for key in ("split_manifest", "qc_artifact"):
        if config["dataset"].get(key):
            config["dataset"][key] = str(project_path(config["dataset"][key]))
    return config


def _artifact_dependencies(dataset: str, config: dict) -> list[dict]:
    if dataset == "pallier2025":
        path = Path(config["dataset"]["qc_artifact"])
        return [
            {
                "role": "immutable_annotation_qc",
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        ]
    if dataset != "chineseeeg2_littleprince":
        return []
    sources = config["dataset"].get("actual_reading_sources") or [config["dataset"]]
    dependencies = []
    for source in sources:
        path = Path(source["alignment_path"])
        dependencies.append(
            {
                "role": "actual_reading_alignment",
                "voice_version": source.get("voice_version"),
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return dependencies


def _window_contract(config: dict) -> dict:
    dataset = config["dataset"]
    keys = (
        "source_sampling_rate_hz",
        "target_sampling_rate_hz",
        "window_start_offset_seconds",
        "window_seconds",
        "eligibility_window_seconds",
        "baseline_seconds",
        "source_preprocessed_filter_hz",
        "additional_filter",
        "filter_low_hz",
        "filter_high_hz",
        "filter_order",
        "scaler",
        "clamp",
    )
    return {key: dataset.get(key) for key in keys if key in dataset}


def _context_definition(config: dict) -> dict:
    dataset = config["dataset"]
    result = {
        "grouping": dataset.get("context_grouping"),
        "maximum_words": dataset.get("max_context_words"),
        "semantic_context": dataset.get("semantic_context"),
    }
    if dataset.get("id") == "pallier2025":
        result["identifier"] = "pallier2025|run-YY|sequence-ZZZ"
    return result


def _source_contract(dataset: str, config: dict) -> dict:
    return {
        "dataset_config": config["dataset"],
        "artifact_dependencies": _artifact_dependencies(dataset, config),
    }


def _build_events(dataset: str, config: dict):
    dataset_config = config["dataset"]
    if dataset == "chineseeeg2_littleprince":
        legacy_table = chineseeeg2.build_event_table(dataset_config)
        audit = chineseeeg2.audit_event_table(legacy_table, dataset_config)
        table = chineseeeg2.canonical_event_table(legacy_table)
    elif dataset == "pallier2025":
        table, audit = pallier2025.build_event_table(dataset_config)
    else:
        raise ValueError(f"未知数据集：{dataset}")

    event_path = Path(config["cache"]["event_table"])
    return write_event_product(
        table,
        event_path,
        manifest_kwargs={
            "dataset": dataset,
            "project_root": PROJECT_ROOT,
            "source_raw_roots": [dataset_config["root"]],
            "artifact_dependencies": _artifact_dependencies(dataset, config),
            "context_definition": _context_definition(config),
            "window_contract": _window_contract(config),
            "source_contract": _source_contract(dataset, config),
            "audit": audit,
            "additional_fields": (
                {
                    "raw_dataset_name": pallier2025.RAW_DATASET_NAME,
                    "openneuro_id": pallier2025.OPENNEURO_ID,
                    "canonical_scope": pallier2025.CORE_SCOPE,
                    "run_split": {
                        "manifest": Path(dataset_config["split_manifest"])
                        .relative_to(PROJECT_ROOT)
                        .as_posix(),
                        "sha256": sha256_file(dataset_config["split_manifest"]),
                        "assignments": {
                            split: [
                                run
                                for run, value in pallier2025.load_run_split(
                                    dataset_config["split_manifest"]
                                ).items()
                                if value == split
                            ]
                            for split in ("train", "val", "test")
                        },
                    },
                    "normalization": dataset_config["normalization"],
                    "qc_decision": audit["qc_decision"],
                    "alignment_fingerprints": [
                        {
                            "subject": item["subject"],
                            "run": item["run"],
                            "bids": item["left_fingerprint"],
                            "extra_info": item["right_fingerprint"],
                            "edit_distance": item["edit_distance"],
                        }
                        for item in audit["alignment_audits"]
                    ],
                }
                if dataset == "pallier2025"
                else None
            ),
            "builder_sources": [
                Path(__file__),
                Path(chineseeeg2.__file__)
                if dataset == "chineseeeg2_littleprince"
                else Path(pallier2025.__file__),
                PROJECT_ROOT / "braindecoding/data/derived.py",
                PROJECT_ROOT / "braindecoding/events.py",
            ],
        },
    )


def _load_events(dataset: str, config: dict):
    path = Path(config["cache"]["event_table"])
    if not path.exists():
        raise FileNotFoundError(f"canonical 事件表不存在，请先运行 --events：{path}")
    validate_event_product(path)
    if dataset == "chineseeeg2_littleprince":
        return chineseeeg2.load_event_table(path, trainable_only=False)
    if dataset == "pallier2025":
        return pallier2025.load_event_table(path)
    raise ValueError(f"未知数据集：{dataset}")


def _write_step_manifest(dataset: str, config: dict, step: str, files) -> Path:
    provenance_dir = PROJECT_ROOT / "data/cache/derived" / dataset / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    file_records = []
    for path in sorted({Path(value) for value in files}):
        file_records.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "step": step,
        "event_table_sha256": sha256_file(config["cache"]["event_table"]),
        "contract_sha256": stable_sha256(
            config["text_embedding"] if step == "text" else config["dataset"]
        ),
        "files": file_records,
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    path = provenance_dir / f"{step}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_step_manifest(path)
    return path


def validate_step_manifest(path) -> dict:
    """验证信号或文本产品清单及其中每个文件。"""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded_sha = payload.get("manifest_sha256")
    body = dict(payload)
    body.pop("manifest_sha256", None)
    if stable_sha256(body) != recorded_sha:
        raise ValueError(f"derived 步骤 manifest 自摘要不一致：{path}")
    for record in payload.get("files", ()):
        product = PROJECT_ROOT / record["path"]
        if not product.exists():
            raise FileNotFoundError(f"derived 产品缺失：{product}")
        if product.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"derived 产品大小漂移：{product}")
        if sha256_file(product) != record["sha256"]:
            raise ValueError(f"derived 产品 SHA 漂移：{product}")
    return payload


def _write_json_atomic(path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def _signal_layout(dataset: str, config: dict):
    if dataset == "chineseeeg2_littleprince":
        return chineseeeg2.处理后记录路径, "vhdr_relpath", "eeg_dir"
    raise ValueError(f"未知数据集：{dataset}")


def _validate_signal_coverage(dataset: str, config: dict, table) -> dict:
    """验证每条 recording 恰好对应一个信号产品，且可训练窗口均在范围内。"""
    path_builder, relative_column, cache_key = _signal_layout(dataset, config)
    root = Path(config["dataset"]["root"])
    cache_dir = Path(config["cache"][cache_key])
    records = table[
        ["subject_id", "recording_id", relative_column]
    ].drop_duplicates()
    if records["recording_id"].duplicated().any():
        raise ValueError("同一记录编号对应多个原始信号文件。")

    expected = {}
    channel_counts = set()
    sample_counts = {}
    for row in records.itertuples(index=False):
        relative_path = str(getattr(row, relative_column))
        source_path = root / relative_path
        output_path = path_builder(
            source_path, config["dataset"], cache_dir
        )
        expected[str(row.recording_id)] = output_path.resolve()
        if not output_path.exists() or not output_path.with_suffix(".json").exists():
            continue
        metadata = json.loads(
            output_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        array = np.load(output_path, mmap_mode="r")
        if array.dtype != np.float32 or list(array.shape) != metadata.get("shape"):
            raise ValueError(f"信号产品 shape/dtype 与 sidecar 不一致：{output_path}")
        channel_counts.add(int(array.shape[0]))
        sample_counts[str(row.recording_id)] = int(array.shape[1])

    actual = {path.resolve() for path in cache_dir.rglob("*.npy")}
    expected_paths = set(expected.values())
    missing = sorted(str(path) for path in expected_paths - actual)
    extra = sorted(str(path) for path in actual - expected_paths)
    if missing or extra:
        raise ValueError(
            f"信号覆盖不完整：missing={len(missing)}, extra={len(extra)}"
        )
    if len(channel_counts) != 1:
        raise ValueError(f"信号产品通道数不一致：{sorted(channel_counts)}")

    trainable = table[table["is_trainable"].astype(bool)]
    for recording_id, rows in trainable.groupby("recording_id", sort=False):
        sample_count = sample_counts[str(recording_id)]
        starts = rows["window_start_target_sample"].astype(int)
        stops = rows["window_stop_target_sample"].astype(int)
        if starts.lt(0).any() or stops.gt(sample_count).any():
            raise ValueError(f"可训练窗口超出信号范围：{recording_id}")

    return {
        "expected_recordings": int(len(expected_paths)),
        "signal_products": int(len(actual)),
        "missing": 0,
        "extra": 0,
        "channel_count": int(next(iter(channel_counts))),
        "trainable_windows_in_range": True,
    }


def _validate_text_contract(dataset: str, config: dict, table) -> dict:
    path = Path(config["cache"]["text_embeddings"])
    if dataset == "pallier2025":
        manifest = pallier2025.validate_text_manifest(
            path.parent / "manifest.json",
            expected_config=config["text_embedding"],
        )
        expected_words = pallier2025.material_word_view(table)
        expected_words = sorted(
            set(expected_words["标准词"].map(pallier2025.normalize_french_word))
            - {""}
        )
        with np.load(path, allow_pickle=False) as payload:
            words = payload["words"].astype(str).tolist()
        if words != expected_words:
            raise ValueError("Pallier2025 canonical text 必须覆盖全部非空材料标准词型。")
        return {
            "word_count": len(words),
            "shape": manifest["embedding_shape"],
            "dtype": manifest["dtype"],
            "all_splits_deterministic_text_materialization": True,
            "content_sha256": manifest["content_sha256"],
        }
    metadata_path = path.with_suffix(".json")
    if not path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"canonical 文本产品不存在：{path}")
    selected = table[
        table["split"].isin(["train", "val"])
        & table["is_trainable"].astype(bool)
    ]
    expected_words = sorted(
        set(selected["normalized_word"].astype(str)) - {""}
    )
    with np.load(path, allow_pickle=False) as payload:
        words = payload["words"].astype(str).tolist()
        embeddings = payload["embeddings"]
        if words != expected_words:
            raise ValueError(f"{dataset} canonical 文本词序或词集合漂移。")
        if embeddings.dtype != np.float32 or embeddings.ndim != 2:
            raise ValueError(f"{dataset} canonical 文本矩阵 dtype/shape 无效。")
        shape = list(embeddings.shape)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("word_count") != len(words):
        raise ValueError(f"{dataset} 文本 sidecar 词数不一致。")
    return {
        "word_count": len(words),
        "shape": shape,
        "dtype": "float32",
        "train_val_only": True,
    }


def _build_signals(dataset: str, config: dict, table, *, force=False) -> Path:
    dataset_config = config["dataset"]
    if dataset == "pallier2025":
        provenance_dir = PROJECT_ROOT / "data/cache/derived/pallier2025/provenance"
        pallier2025.run_reference_gate(dataset_config, provenance_dir)
        pallier2025.build_time_origin_audit(dataset_config, provenance_dir)
        timing_records = []
        manifest_path = pallier2025.ensure_recording_signals(
            table,
            dataset_config,
            config["cache"]["meg_dir"],
            project_root=PROJECT_ROOT,
            force=force,
            timing_records=timing_records,
        )
        benchmark_path = provenance_dir / "materialization_benchmark.json"
        payload = {
            "schema_version": 1,
            "dataset": dataset,
            "recording_count": len(timing_records),
            "records": timing_records,
        }
        payload["manifest_sha256"] = stable_sha256(payload)
        _write_json_atomic(benchmark_path, payload)
        return manifest_path
    if dataset == "chineseeeg2_littleprince":
        paths = chineseeeg2.ensure_recording_signals(
            table, dataset_config, config["cache"]["eeg_dir"], force=force
        )
    else:
        raise ValueError(f"未知数据集：{dataset}")
    sidecars = [Path(path).with_suffix(".json") for path in paths]
    return _write_step_manifest(
        dataset, config, "signals", [*paths, *sidecars]
    )


def _build_text(dataset: str, config: dict, table, *, force=False) -> Path:
    path = Path(config["cache"]["text_embeddings"])
    if dataset == "pallier2025":
        manifest = pallier2025.build_text_product(
            table,
            config["text_embedding"],
            path,
            event_table_path=config["cache"]["event_table"],
            force=force,
        )
        pallier2025.write_protocol_assets(
            config["cache"]["event_table"],
            PROJECT_ROOT / "data/manifests/pallier2025",
        )
        return manifest
    # 与既有训练入口一致：首轮只物化 train+val 可训练词；test-only 词不参与
    # 当前缓存的词集合或批次组成，正式 test 解锁后再显式增量生成。
    selected = table[
        table["split"].isin(["train", "val"])
        & table["is_trainable"].astype(bool)
    ].reset_index(drop=True)
    if dataset == "chineseeeg2_littleprince":
        result = chineseeeg2.ensure_text_embedding_cache(
            selected["normalized_word"], config["text_embedding"], path
        )
    else:
        raise ValueError(f"未知数据集：{dataset}")
    return _write_step_manifest(
        dataset,
        config,
        "text",
        [result, Path(result).with_suffix(".json")],
    )


def _component_manifest_paths(dataset: str, config: dict) -> dict[str, Path]:
    signal_path = (
        PROJECT_ROOT / "data/cache/derived/pallier2025/signals/manifest.json"
        if dataset == "pallier2025"
        else PROJECT_ROOT / "data/cache/derived" / dataset / "provenance" / "signals.json"
    )
    return {
        "events": event_manifest_path(config["cache"]["event_table"]),
        "signals": signal_path,
        "text": (
            Path(config["cache"]["text_embeddings"]).parent / "manifest.json"
            if dataset == "pallier2025"
            else PROJECT_ROOT / "data/cache/derived" / dataset / "provenance" / "text.json"
        ),
    }


def _dataset_manifest_path(dataset: str) -> Path:
    return PROJECT_ROOT / "data/cache/derived" / dataset / "manifest.json"


def _validate_components(dataset: str, config: dict, table) -> dict:
    paths = _component_manifest_paths(dataset, config)
    event_payload = validate_event_product(
        config["cache"]["event_table"], paths["events"]
    )
    if dataset == "pallier2025":
        signal_payload = pallier2025.validate_signal_manifest(
            paths["signals"], event_table=table
        )
        text_payload = pallier2025.validate_text_manifest(
            paths["text"], expected_config=config["text_embedding"]
        )
        coverage = signal_payload["coverage"]
    else:
        signal_payload = validate_step_manifest(paths["signals"])
        text_payload = validate_step_manifest(paths["text"])
        coverage = _validate_signal_coverage(dataset, config, table)
    text_contract = _validate_text_contract(dataset, config, table)
    return {
        "events": event_payload,
        "signals": signal_payload,
        "text": text_payload,
        "signal_coverage": coverage,
        "text_contract": text_contract,
    }


def _write_dataset_manifest(dataset: str, config: dict, table) -> Path:
    """仅在 events/signals/text 全部验证通过后写数据集总索引。"""
    paths = _component_manifest_paths(dataset, config)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"derived 组件 manifest 尚未齐全：{missing}")
    validated = _validate_components(dataset, config, table)
    components = {}
    for name, path in paths.items():
        components[name] = {
            "status": "complete",
            "manifest": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
    components["events"]["event_table_sha256"] = validated["events"][
        "event_table_sha256"
    ]
    components["signals"]["coverage"] = validated["signal_coverage"]
    components["text"]["contract"] = validated["text_contract"]
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "status": "complete",
        "components": components,
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    path = _dataset_manifest_path(dataset)
    _write_json_atomic(path, payload)
    validate_dataset_manifest(path)
    return path


def _write_partial_dataset_manifest(dataset: str, config: dict) -> Path:
    """为尚未完成全部组件的数据集写诚实的状态索引。"""
    event_path = Path(config["cache"]["event_table"])
    event_manifest = event_manifest_path(event_path)
    event_payload = validate_event_product(event_path, event_manifest)
    signal_manifest = _component_manifest_paths(dataset, config)["signals"]
    signal_component = {"status": "missing"}
    if signal_manifest.exists():
        if dataset != "pallier2025":
            raise ValueError("partial dataset manifest 目前只支持 Pallier2025。")
        signal_payload = pallier2025.validate_signal_manifest(signal_manifest)
        signal_component = {
            "status": "complete",
            "manifest": signal_manifest.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": sha256_file(signal_manifest),
            "coverage": signal_payload["coverage"],
        }
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "status": "incomplete",
        "components": {
            "events": {
                "status": "complete",
                "manifest": event_manifest.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(event_manifest),
                "event_table_sha256": event_payload["event_table_sha256"],
            },
            "signals": signal_component,
            "text": {"status": "missing"},
        },
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    path = _dataset_manifest_path(dataset)
    _write_json_atomic(path, payload)
    validate_dataset_manifest(path, require_complete=False)
    return path


def validate_dataset_manifest(path, *, require_complete=True) -> dict:
    """验证数据集总索引；不重新构建任何 derived 产品。"""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    recorded_sha = body.pop("manifest_sha256", None)
    if stable_sha256(body) != recorded_sha:
        raise ValueError(f"数据集 manifest 自摘要不一致：{path}")
    if require_complete and payload.get("status") != "complete":
        raise ValueError(f"数据集 derived 状态不是 complete：{path}")
    for component, record in payload.get("components", {}).items():
        if record.get("status") != "complete":
            if require_complete:
                raise ValueError(f"derived 组件未完成：{component}")
            if record.get("status") != "missing":
                raise ValueError(f"derived 组件状态无效：{component}")
            continue
        if "manifest" not in record or "sha256" not in record:
            raise ValueError(f"derived 组件未完成：{component}")
        manifest_path = PROJECT_ROOT / normalize_project_relative(str(record["manifest"]))
        if sha256_file(manifest_path) != record["sha256"]:
            raise ValueError(f"derived 组件 manifest SHA 漂移：{manifest_path}")
    return payload


def check_dataset(dataset: str) -> dict:
    """只读验证一个数据集的完整 canonical derived 合同。"""
    config = load_build_config(dataset)
    table = _load_events(dataset, config)
    if dataset == "pallier2025":
        event_payload = validate_event_product(config["cache"]["event_table"])
        dataset_manifest = validate_dataset_manifest(
            _dataset_manifest_path(dataset), require_complete=False
        )
        signal_coverage = None
        text_contract = None
        signal_record = dataset_manifest["components"]["signals"]
        if signal_record["status"] == "complete":
            signal_payload = pallier2025.validate_signal_manifest(
                PROJECT_ROOT / normalize_project_relative(str(signal_record["manifest"])), event_table=table
            )
            signal_coverage = signal_payload["coverage"]
        text_record = dataset_manifest["components"]["text"]
        if text_record["status"] == "complete":
            pallier2025.validate_text_manifest(
                PROJECT_ROOT / normalize_project_relative(str(text_record["manifest"])),
                expected_config=config["text_embedding"],
            )
            text_contract = _validate_text_contract(dataset, config, table)
        return {
            "dataset": dataset,
            "status": dataset_manifest["status"],
            "components": {
                name: record["status"]
                for name, record in dataset_manifest["components"].items()
            },
            "event_count": int(len(table)),
            "event_table_sha256": event_payload["event_table_sha256"],
            "signal_coverage": signal_coverage,
            "text_contract": text_contract,
            "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
            "test_model_evaluation_performed": False,
        }
    validated = _validate_components(dataset, config, table)
    dataset_manifest = validate_dataset_manifest(_dataset_manifest_path(dataset))
    return {
        "dataset": dataset,
        "status": dataset_manifest["status"],
        "event_count": int(len(table)),
        "event_table_sha256": validated["events"]["event_table_sha256"],
        "signal_coverage": validated["signal_coverage"],
        "text_contract": validated["text_contract"],
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "test_model_evaluation_performed": False,
    }


def build_dataset(
    dataset: str,
    *,
    events=False,
    signals=False,
    text=False,
    force=False,
) -> dict:
    """按 events→signals→text 顺序构建一个数据集。"""
    config = load_build_config(dataset)
    result = {"dataset": dataset}
    if events:
        event_path = Path(config["cache"]["event_table"])
        manifest_path = event_manifest_path(event_path)
        expected_contract_sha = stable_sha256(_source_contract(dataset, config))
        if event_path.exists() and manifest_path.exists() and not force:
            manifest = validate_event_product(event_path, manifest_path)
            if manifest.get("source_contract_sha256") != expected_contract_sha:
                raise ValueError(
                    "derived 事件来源合同已变化；请审查后显式使用 --force 重建。"
                )
            result["events_reused"] = True
        else:
            event_path, manifest_path, manifest = _build_events(dataset, config)
            result["events_reused"] = False
        result["events"] = str(event_path)
        result["event_manifest"] = str(manifest_path)
        result["event_table_sha256"] = manifest["event_table_sha256"]
    table = None
    if signals or text:
        table = _load_events(dataset, config)
    if signals:
        result["signals_manifest"] = str(
            _build_signals(dataset, config, table, force=force)
        )
    if text:
        result["text_manifest"] = str(
            _build_text(dataset, config, table, force=force)
        )
    component_paths = _component_manifest_paths(dataset, config)
    if all(path.exists() for path in component_paths.values()):
        if table is None:
            table = _load_events(dataset, config)
        result["dataset_manifest"] = str(
            _write_dataset_manifest(dataset, config, table)
        )
    elif dataset == "pallier2025" and Path(config["cache"]["event_table"]).exists():
        result["dataset_manifest"] = str(
            _write_partial_dataset_manifest(dataset, config)
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append", required=True)
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--signals", action="store_true")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="只读验证已有 derived 产品"
    )
    parser.add_argument("--force", action="store_true", help="显式重建已有 derived 产品")
    args = parser.parse_args(argv)
    build_requested = args.events or args.signals or args.text or args.all
    if args.check and (build_requested or args.force):
        parser.error("--check 不能与构建参数或 --force 同时使用。")
    if not (build_requested or args.check):
        parser.error("必须指定 --events、--signals、--text、--all 或 --check。")
    for dataset in args.dataset:
        if args.check:
            result = check_dataset(dataset)
        else:
            result = build_dataset(
                dataset,
                events=args.events or args.all,
                signals=args.signals or args.all,
                text=args.text or args.all,
                force=args.force,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
