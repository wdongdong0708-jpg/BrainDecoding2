"""历史词级配置的只读兼容检查；当前协议由各自专用入口验证。"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

from braindecoding.catalog import discover_runnable_experiment_configs
from braindecoding.config import PROJECT_ROOT, project_path, load_yaml_with_extends
from braindecoding.data.build import check_dataset
from braindecoding.data.derived import stable_sha256
from braindecoding.experiment import (
    experiment_identity,
    git_tracked_dirty,
    resolve_experiment_config,
    resolved_config_sha256,
    run_directory,
    scientific_config_sha256,
    warm_start_checkpoint,
)
EXPECTED_SUBJECTS = {
    "chineseeeg2_littleprince": tuple(f"sub-{index:02d}" for index in range(1, 9)),
    "pallier2025": tuple(f"sub-{index:02d}" for index in range(1, 11)),
}

_LEGACY_RECORDS = discover_runnable_experiment_configs()
LEGACY_CONFIGS = tuple(
    record["relative_config"]
    for record in _LEGACY_RECORDS
    if record["identity"]["dataset"] in EXPECTED_SUBJECTS
)
EXCLUDED_LEGACY_CONFIGS = tuple(
    {
        "config": record["relative_config"],
        "dataset": record["identity"]["dataset"],
        "reason": "unsupported_dataset_contract",
    }
    for record in _LEGACY_RECORDS
    if record["identity"]["dataset"] not in EXPECTED_SUBJECTS
)
# 兼容旧调用方的导入名；该集合不代表当前研究计划或可启动的作业。
ACTIVE_CONFIGS = LEGACY_CONFIGS

PALLIER_EVENT_SHA256 = (
    "3cd1237fd138f3d827f8297cb7081243ce635ebb896f05241da8ccc5614a43fc"
)
PALLIER_CHANNEL_SHA256 = (
    "807ff5cf39a18398f004221d596241033e9c55c91f031842069120d10dbf9fb8"
)
PALLIER_TEXT_CONTENT_SHA256 = (
    "605867af6026349dc405db9264689feb97c0288fcce26092790c641d90c8f4f1"
)

_LEGACY_TOKENS = (
    "tasks/word_decoding/ChineseEEG2_LittlePrince/cache",
    "tasks/word_decoding/SMN4Lang/cache",
    "tasks/word_decoding/LibriBrain100/cache",
    "configs/ChineseEEG2_LittlePrince",
    "configs/SMN4Lang",
    "configs/LibriBrain100",
    "outputs/ChineseEEG2_LittlePrince",
    "outputs/SMN4Lang",
    "outputs/LibriBrain100",
)


def _load_resolved_config(relative_path: str, output_root=None) -> dict:
    config = load_yaml_with_extends(PROJECT_ROOT / relative_path)
    return resolve_experiment_config(config, output_root=output_root)


def _stable_config_hashes(relative_path: str, output_root=None) -> dict:
    first = _load_resolved_config(relative_path, output_root=output_root)
    second = _load_resolved_config(relative_path, output_root=output_root)
    scientific_first = scientific_config_sha256(first)
    resolved_first = resolved_config_sha256(first)
    return {
        "config": first,
        "scientific_config_sha256": scientific_first,
        "scientific_config_sha256_stable": (
            scientific_first == scientific_config_sha256(second)
        ),
        "resolved_config_sha256": resolved_first,
        "resolved_config_sha256_stable": (
            resolved_first == resolved_config_sha256(second)
        ),
    }


def _dataset_manifest(dataset: str) -> dict:
    path = PROJECT_ROOT / "data" / "cache" / "derived" / dataset / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _output_status(config: dict, output_root=None) -> dict:
    path = run_directory(config, output_root=output_root)
    manifest_path = path / "run_manifest.json"
    result = {
        "path": str(path),
        "exists": path.exists(),
        "status": "available",
        "collision": False,
    }
    if not path.exists():
        return result
    if not manifest_path.is_file():
        return {**result, "status": "collision_missing_manifest", "collision": True}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result["existing_run_status"] = manifest.get("status")
    if manifest.get("resolved_config_sha256") != resolved_config_sha256(config):
        return {**result, "status": "collision_config_sha_mismatch", "collision": True}
    if manifest.get("status") == "completed":
        return {**result, "status": "collision_completed_run", "collision": True}
    return {**result, "status": "resume_required", "collision": True}


def _dependency_status(config: dict, active_identities: set[tuple], output_root=None) -> dict:
    checkpoint = warm_start_checkpoint(config, output_root=output_root)
    if checkpoint is None:
        return {"status": "not_required", "checkpoint": None}
    source = copy.deepcopy(config)
    source["experiment"]["id"] = config["training"]["warm_start_from"]
    source_identity = experiment_identity(source)
    identity_key = tuple(source_identity.values())
    if identity_key not in active_identities:
        return {
            "status": "invalid_upstream_identity",
            "checkpoint": str(checkpoint),
            "upstream_identity": source_identity,
        }
    return {
        "status": "ready" if checkpoint.is_file() else "waiting_for_upstream",
        "checkpoint": str(checkpoint),
        "upstream_identity": source_identity,
    }


def _legacy_dependency_status(relative_path: str, config: dict) -> dict:
    leaf_path = PROJECT_ROOT / relative_path
    leaf = leaf_path.read_text(encoding="utf-8").replace("\\", "/")
    base_path = leaf_path.parent.parent / "base.yaml"
    base = base_path.read_text(encoding="utf-8").replace("\\", "/")
    serialized = json.dumps(config, ensure_ascii=False).replace("\\", "/")
    matches = [token for token in _LEGACY_TOKENS if token in leaf or token in base or token in serialized]
    return {
        "status": "none" if not matches else "found",
        "legacy_tokens": matches,
        "extends": "../base.yaml" if "extends: ../base.yaml" in leaf else None,
        "base_has_extends": "extends:" in base,
    }


def _subject_status(dataset: str, config: dict, event_manifest: dict) -> dict:
    expected = EXPECTED_SUBJECTS[dataset]
    event_subjects = tuple(event_manifest.get("subjects", ()))
    config_subjects = tuple(config["dataset"].get("subjects", event_subjects))
    return {
        "expected": list(expected),
        "config": list(config_subjects),
        "event_manifest": list(event_subjects),
        "matches": config_subjects == expected and event_subjects == expected,
    }


def _embedded_hash_valid(payload: dict, field="manifest_sha256") -> bool:
    recorded = payload.get(field)
    body = dict(payload)
    body.pop(field, None)
    return recorded is not None and stable_sha256(body) == recorded


def _pallier_contract_status(config: dict, event_manifest: dict) -> dict:
    """只读验证 Stage 2A--2C 冻结合同，不加载 signal array 或模型。"""
    derived_root = PROJECT_ROOT / "data" / "cache" / "derived" / "pallier2025"
    signal_path = derived_root / "signals" / "manifest.json"
    text_path = derived_root / "text" / "t5_large_layer_0_5" / "manifest.json"
    asset_root = PROJECT_ROOT / "data" / "manifests" / "pallier2025"
    qc_path = (
        PROJECT_ROOT
        / "data/cache/artifacts"
        / "annotations"
        / "pallier2025"
        / "sub-09_run-03_qc.json"
    )
    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    text = json.loads(text_path.read_text(encoding="utf-8"))
    split = json.loads((asset_root / "run_split.json").read_text(encoding="utf-8"))
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    vocabulary_hashes = {}
    vocabulary_valid = True
    for size in (20, 50, 100, 150):
        payload = json.loads(
            (asset_root / f"vocabulary_N{size}.json").read_text(encoding="utf-8")
        )
        vocabulary_hashes[str(size)] = payload.get("manifest_sha256")
        vocabulary_valid = vocabulary_valid and (
            _embedded_hash_valid(payload)
            and len(payload.get("vocabulary", ())) == size
            and payload.get("source_split") == "train"
            and payload.get("created_from_test") is False
        )

    dataset = config["dataset"]
    experiment_id = config.get("experiment", {}).get("id")
    expected_transformer = experiment_id in {
        "main_context",
        "main_context_warmstart",
        "main_context_causal_transformer",
        "subject_independent_context_warmstart",
    } or experiment_id in {
        *(f"local_window_r{radius}" for radius in (0, 1, 2, 4, 8)),
        "local_window_m2_p0",
        "local_window_m2_p1",
        "local_window_rolling_revision_k2_d2",
    }
    warm_start_sources = {
        "main_context_warmstart": "main_word",
        "main_context_causal_transformer": "main_word",
        "subject_independent_context_warmstart": "subject_independent_word",
        **{f"local_window_r{radius}": "main_word" for radius in (0, 1, 2, 4, 8)},
        "local_window_m2_p0": "main_word",
        "local_window_m2_p1": "main_word",
        "local_window_rolling_revision_k2_d2": "main_word",
    }
    expected_warm_start_source = warm_start_sources.get(experiment_id)
    warm_start_expected = expected_warm_start_source is not None
    training = config.get("training", {})
    convolution = config.get("model", {}).get("conv", {})
    subject_independent = experiment_id.startswith("subject_independent_")
    canonical_cache = config.get("cache", {})
    checks = {
        "event_table_sha256": event_manifest.get("event_table_sha256")
        == PALLIER_EVENT_SHA256,
        "event_count": event_manifest.get("event_count") == 152560,
        "trainable_event_count": event_manifest.get("trainable_event_count")
        == 150700,
        "subjects": tuple(event_manifest.get("subject_order", ()))
        == EXPECTED_SUBJECTS["pallier2025"],
        "recordings": signal.get("recording_count") == 90,
        "signal_status": signal.get("status") == "complete",
        "channel_count": signal.get("channel_count") == 306,
        "channel_order": signal.get("channel_names_sha256")
        == PALLIER_CHANNEL_SHA256,
        "sampling_rate": float(signal.get("target_sampling_rate_hz", -1)) == 50.0,
        "signal_dtype": signal.get("dtype") == "float32",
        "window_samples": float(dataset.get("window_seconds", -1)) * 50 == 50,
        "baseline_samples": float(dataset.get("baseline_seconds", -1)) * 50 == 25,
        "eligibility_window": float(
            dataset.get("eligibility_window_seconds", -1)
        )
        == 3.0,
        "clamp": float(dataset.get("clamp", -1)) == 5.0,
        "text_status": text.get("status") == "complete",
        "text_shape": text.get("embedding_shape") == [2426, 1024],
        "text_content": text.get("content_sha256")
        == PALLIER_TEXT_CONTENT_SHA256,
        "vocabularies": vocabulary_valid,
        "run_split": split.get("assignments")
        == {
            "train": [
                "run-01",
                "run-02",
                "run-03",
                "run-04",
                "run-05",
                "run-08",
                "run-09",
            ],
            "val": ["run-07"],
            "test": ["run-06"],
        },
        "qc_artifact": qc.get("decision") == "exclude_recording"
        and qc.get("recording", {}).get("subject") == "sub-09"
        and qc.get("recording", {}).get("run") == "run-03",
        "model_condition": bool(config.get("model", {}).get("use_transformer"))
        is expected_transformer,
        "context_mode": config.get("model", {}).get("context_mode") == "grouped",
        "runtime_context_grouping": (
            not expected_transformer
            or dataset.get("runtime_context_grouping")
            == "recording_id_plus_canonical_context_v1"
        ),
        "embedding_dimension": config.get("model", {}).get(
            "embedding_dimension"
        )
        == 1024,
        "transformer_geometry": config.get("model", {})
        .get("transformer", {})
        .get("heads")
        == 16
        and config.get("model", {}).get("transformer", {}).get("depth") == 16,
        "subject_conditioning": (
            convolution.get("merger_per_subject") is False
            and convolution.get("subject_projection_per_subject") is False
            if subject_independent
            else convolution.get("merger_per_subject") is True
            and convolution.get("subject_projection_per_subject", True) is True
        ),
        "initialization_contract": (
            training.get("warm_start_from") == expected_warm_start_source
            and bool(training.get("pretrained_brain_encoder_checkpoint"))
            and training.get("freeze_brain_encoder_updates") in (None, 0, "")
            if warm_start_expected
            else all(
                training.get(key) in (None, 0, "")
                for key in (
                    "warm_start_from",
                    "pretrained_brain_encoder_checkpoint",
                    "freeze_brain_encoder_updates",
                )
            )
        ),
        "selection_metric": config.get("evaluation", {}).get("selection_metric")
        == (
            "revision_common_delay2_"
            "retrieval_acc10_vocab=pallier2025_50_macro"
            if experiment_id == "local_window_rolling_revision_k2_d2"
            else "retrieval_acc10_vocab=pallier2025_50_macro"
        ),
        "canonical_event_path": str(canonical_cache.get("event_table", ""))
        .replace("\\", "/")
        .endswith("data/cache/derived/pallier2025/events/events.csv"),
        "canonical_signal_path": str(canonical_cache.get("meg_dir", ""))
        .replace("\\", "/")
        .endswith("data/cache/derived/pallier2025/signals/meg_50hz"),
        "canonical_text_path": str(canonical_cache.get("text_embeddings", ""))
        .replace("\\", "/")
        .endswith(
            "data/cache/derived/pallier2025/text/t5_large_layer_0_5/embeddings.npz"
        ),
    }
    return {
        "status": "valid" if all(checks.values()) else "invalid",
        "checks": checks,
        "vocabulary_manifest_sha256": vocabulary_hashes,
        "neural_arrays_loaded": False,
        "model_loaded": False,
    }


def _production_legacy_references() -> list[dict]:
    """扫描历史适配器与公共实现中的旧根目录依赖。"""
    paths = [
        path
        for path in sorted((PROJECT_ROOT / "braindecoding").rglob("*.py"))
        if path.resolve() != Path(__file__).resolve()
    ]
    matches = []
    for path in paths:
        text = path.read_text(encoding="utf-8").replace("\\", "/")
        for token in _LEGACY_TOKENS:
            if token in text:
                matches.append(
                    {
                        "path": path.relative_to(PROJECT_ROOT).as_posix(),
                        "token": token,
                    }
                )
    return matches


def build_preflight_report(*, output_root=None, config_paths=None) -> dict:
    """检查已定义合同的历史配置；可收窄范围且不创建运行目录。"""
    selected_configs = (
        tuple(LEGACY_CONFIGS)
        if config_paths is None
        else tuple(dict.fromkeys(str(path) for path in config_paths))
    )
    if not selected_configs:
        raise ValueError("preflight 至少需要一个实验配置。")
    unsupported = sorted(
        set(selected_configs) & {item["config"] for item in EXCLUDED_LEGACY_CONFIGS}
    )
    if unsupported:
        raise ValueError(f"历史检查器尚未定义这些数据集的合同：{unsupported}")
    unknown = sorted(set(selected_configs) - set(LEGACY_CONFIGS))
    if unknown:
        raise ValueError(
            f"配置不在历史词级检查范围内：{unknown}。"
            "当前主协议与补充实验使用各自专用检查入口。"
        )
    git_dirty = git_tracked_dirty()
    loaded_all = {
        path: _stable_config_hashes(path, output_root=output_root)
        for path in LEGACY_CONFIGS
    }
    all_identities = {
        path: experiment_identity(record["config"])
        for path, record in loaded_all.items()
    }
    loaded = {path: loaded_all[path] for path in selected_configs}
    identities = {
        path: experiment_identity(record["config"])
        for path, record in loaded.items()
    }
    derived_checks = {
        dataset: check_dataset(dataset)
        for dataset in sorted(
            {identity["dataset"] for identity in identities.values()}
        )
    }
    identity_keys = [tuple(identity.values()) for identity in identities.values()]
    identities_unique = len(identity_keys) == len(set(identity_keys))
    active_identity_set = {
        tuple(identity.values()) for identity in all_identities.values()
    }
    production_legacy_references = _production_legacy_references()

    experiments = []
    for relative_path in selected_configs:
        hash_record = loaded[relative_path]
        config = hash_record.pop("config")
        identity = identities[relative_path]
        dataset = identity["dataset"]
        dataset_manifest = _dataset_manifest(dataset)
        components = dataset_manifest["components"]
        event_manifest_path = project_path(components["events"]["manifest"])
        event_manifest = json.loads(event_manifest_path.read_text(encoding="utf-8"))
        output = _output_status(config, output_root=output_root)
        dependency = _dependency_status(
            config, active_identity_set, output_root=output_root
        )
        legacy = _legacy_dependency_status(relative_path, config)
        subjects = _subject_status(dataset, config, event_manifest)
        pallier_contract = (
            _pallier_contract_status(config, event_manifest)
            if dataset == "pallier2025"
            else None
        )
        default_split = config.get("evaluation", {}).get("default_split")
        reasons = []
        if not identities_unique:
            reasons.append("duplicate_experiment_identity")
        if derived_checks[dataset]["status"] != "complete":
            reasons.append("derived_incomplete")
        if not subjects["matches"]:
            reasons.append("subject_order_mismatch")
        if (
            legacy["status"] != "none"
            or legacy["base_has_extends"]
            or legacy["extends"] != "../base.yaml"
            or production_legacy_references
        ):
            reasons.append("legacy_dependency")
        if not hash_record["scientific_config_sha256_stable"]:
            reasons.append("scientific_config_sha_unstable")
        if not hash_record["resolved_config_sha256_stable"]:
            reasons.append("resolved_config_sha_unstable")
        if output["collision"]:
            reasons.append(output["status"])
        if default_split != "val":
            reasons.append("default_evaluation_split_is_not_val")
        if dependency["status"] == "invalid_upstream_identity":
            reasons.append("invalid_upstream_identity")
        if dependency["status"] == "waiting_for_upstream":
            reasons.append("waiting_for_upstream")
        if pallier_contract is not None and pallier_contract["status"] != "valid":
            reasons.append("pallier_contract_mismatch")
        if (
            dataset == "chineseeeg2_littleprince"
            and identity["experiment_id"]
            in {
                "main_context",
                "main_context_warmstart",
                "main_context_causal_transformer",
            }
            and config["dataset"].get("context_grouping") != "bounded_semantic_v1"
        ):
            reasons.append("main_context_is_not_semantic_v1")

        blockers = [reason for reason in reasons if reason != "waiting_for_upstream"]
        if blockers:
            ready: bool | str = False
        elif dependency["status"] == "waiting_for_upstream":
            ready = "waiting_for_upstream"
        else:
            ready = True
        experiments.append(
            {
                "identity": identity,
                "config": relative_path,
                **hash_record,
                "derived_status": derived_checks[dataset]["status"],
                "event_status": {
                    "status": components["events"]["status"],
                    "manifest": components["events"]["manifest"],
                    "manifest_sha256": components["events"]["sha256"],
                    "event_table_sha256": components["events"]["event_table_sha256"],
                },
                "signal_status": {
                    "status": components["signals"]["status"],
                    "manifest": components["signals"]["manifest"],
                    "manifest_sha256": components["signals"]["sha256"],
                    "coverage": derived_checks[dataset]["signal_coverage"],
                },
                "text_status": {
                    "status": components["text"]["status"],
                    "manifest": components["text"]["manifest"],
                    "manifest_sha256": components["text"]["sha256"],
                    "contract": derived_checks[dataset]["text_contract"],
                },
                "subject_scope": subjects,
                "dataset_contract": pallier_contract,
                "output_path": output,
                "dependency_status": dependency,
                "test_guard_status": {
                    "status": "validation_only" if default_split == "val" else "invalid",
                    "default_split": default_split,
                    "test_model_evaluation_run": False,
                },
                "legacy_dependency": legacy,
                "git_status": "dirty" if git_dirty else "clean",
                "ready": ready,
                "reasons": reasons,
            }
        )

    report = {
        "schema_version": 1,
        "scope": {
            "kind": "all_legacy_supported" if config_paths is None else "selected",
            "configs": list(selected_configs),
        },
        "status": "ready" if all(item["ready"] is True for item in experiments) else "not_ready",
        "git": {
            "tracked_dirty": git_dirty,
            "status": "dirty" if git_dirty else "clean",
            "not_ready_reason": None,
        },
        "identity_unique": identities_unique,
        "production_legacy_references": production_legacy_references,
        "derived_checks": derived_checks,
        "experiments": experiments,
        "side_effects": {
            "run_directories_created": False,
            "checkpoints_generated": False,
            "model_evaluation_run": False,
            "test_model_evaluation_run": False,
        },
    }
    if config_paths is None:
        report["scope"]["excluded_configs"] = list(EXCLUDED_LEGACY_CONFIGS)
    report["report_sha256"] = stable_sha256(report)
    return report


def write_report(report: dict, path: Path) -> None:
    """稳定写入 preflight JSON；目标不在 outputs 中。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiments/preflight_report.json",
    )
    args = parser.parse_args(argv)
    report = build_preflight_report()
    write_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
