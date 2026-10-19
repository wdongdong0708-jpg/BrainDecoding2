"""Pallier2025（OpenNeuro ds007523）词事件构建工具。"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import time
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from braindecoding.data.derived import sha256_file, stable_sha256
import braindecoding.data.text as text_data
from braindecoding.data.sensors import vectorview_channel_positions
from braindecoding.events import (
    CORE_EVENT_COLUMNS,
    align_text_sequences,
    text_fingerprint,
    validate_event_table,
)


DATASET_ID = "pallier2025"
RAW_DATASET_NAME = "LittlePrince_MEG_French_Listen_Pallier2025"
OPENNEURO_ID = "ds007523"
SESSION = "ses-01"
SUBJECTS = tuple(f"sub-{index:02d}" for index in range(1, 11))
RUNS = tuple(f"run-{index:02d}" for index in range(1, 10))
CORE_SCOPE = "sub01-10"
ELIGIBILITY_WINDOW_SECONDS = 3.0
WINDOW_START_OFFSET_SECONDS = 0.0
WINDOW_SECONDS = 1.0
BASELINE_SECONDS = 0.5
CLAMP = 5.0
WINDOW_CONTRACT_VERSION = "word_1s_support_3s_baseline_0p5_v1"
RUN_SPLIT_SALT = "pallier2025-main-run-split-v1"
DASCOLI_REFERENCE_COMMIT = "e1262ee36aa2fbaa6965e5bfb893f6aaee0dd693"
SIGNAL_MATERIALIZATION_VERSION = 1
SOURCE_SAMPLING_RATE_HZ = 1000.0
TARGET_SAMPLING_RATE_HZ = 50.0
FILTER_LOW_HZ = 0.1
FILTER_HIGH_HZ = 40.0
MEG_CHANNEL_COUNT = 306
MAGNETOMETER_COUNT = 102
GRADIOMETER_COUNT = 204
CHANNEL_NAMES_SHA256 = (
    "807ff5cf39a18398f004221d596241033e9c55c91f031842069120d10dbf9fb8"
)
RUNTIME_CONTEXT_COLUMN = "runtime_context_uid"
RUNTIME_CONTEXT_GROUPING = "recording_id_plus_canonical_context_v1"
FRENCH_REFERENCE_TOKENS = (
    "lorsque",
    "cinquième",
    "grâce",
    "ça",
    "j",
    "l",
    "-là",
    "1909",
)

PERSISTED_COLUMNS = (
    *CORE_EVENT_COLUMNS,
    "会话",
    "运行编号",
    "BIDS事件行号",
    "序列原编号",
    "是否序列末词",
    "词性",
    "是否内容词",
    "右括号数量",
)


def normalize_french_word(word) -> str:
    """去除首尾空白并转小写；保留法语重音和原始 tokenization。"""
    if pd.isna(word):
        return ""
    return str(word).strip().lower()


def word_window_contract(dataset_config: dict | None = None) -> dict:
    """返回词级神经信号窗口与 3 秒资格窗口合同（默认 1.0 秒，支持通过 window_seconds 切换）。"""
    config = dataset_config or {}
    if isinstance(config.get("dataset"), dict) and "window_seconds" not in config:
        config = config["dataset"]
    sampling_rate = float(
        config.get("target_sampling_rate_hz", TARGET_SAMPLING_RATE_HZ)
    )
    window_seconds = float(config.get("window_seconds", WINDOW_SECONDS))
    baseline_seconds = float(config.get("baseline_seconds", BASELINE_SECONDS))
    contract = {
        "version": str(
            config.get("window_contract_version", WINDOW_CONTRACT_VERSION)
        ),
        "window_start_offset_seconds": float(
            config.get("window_start_offset_seconds", WINDOW_START_OFFSET_SECONDS)
        ),
        "window_seconds": window_seconds,
        "eligibility_window_seconds": float(
            config.get("eligibility_window_seconds", ELIGIBILITY_WINDOW_SECONDS)
        ),
        "baseline_seconds": baseline_seconds,
        "clamp": float(config.get("clamp", CLAMP)),
        "sampling_rate_hz": sampling_rate,
        "window_samples": int(round(window_seconds * sampling_rate)),
        "baseline_samples": int(round(baseline_seconds * sampling_rate)),
        "continuous_signal_includes_baseline": False,
        "continuous_signal_includes_clamp": False,
    }
    if contract["window_samples"] <= 0:
        raise ValueError(
            f"Pallier2025 窗口样本数必须为正整数，当前为 {contract['window_samples']} "
            f"(window_seconds={window_seconds})。"
        )
    if contract["baseline_samples"] < 0:
        raise ValueError("Pallier2025 baseline 样本数不能为负数。")
    if contract["eligibility_window_seconds"] != 3.0:
        raise ValueError("Pallier2025 正式事件资格窗口必须为 3 秒。")
    return contract


def extract_word_window(signal, onset_seconds, signal_metadata, dataset_config=None):
    """从 continuous product 提取指定时长窗口，再做 baseline 与 clamp。"""
    contract = word_window_contract(dataset_config)
    start = recording_onset_to_sample(
        float(onset_seconds) + contract["window_start_offset_seconds"],
        source_first_samp=signal_metadata["source_first_samp"],
        source_sampling_rate_hz=signal_metadata["source_sampling_rate_hz"],
        output_sampling_rate_hz=signal_metadata["output_sampling_rate_hz"],
    )
    stop = start + contract["window_samples"]
    continuous = np.asarray(signal)
    if continuous.ndim != 2 or start < 0 or stop > continuous.shape[1]:
        raise ValueError("Pallier2025 词窗口超出 canonical continuous signal。")
    window = continuous[:, start:stop].astype(np.float32, copy=True)
    effective_baseline = min(int(contract["baseline_samples"]), int(window.shape[1]))
    if effective_baseline > 0:
        baseline = window[:, :effective_baseline].mean(
            axis=1, keepdims=True
        )
        window -= baseline
    np.clip(window, -contract["clamp"], contract["clamp"], out=window)
    return window


def material_word_view(table: pd.DataFrame) -> pd.DataFrame:
    """去除受试者重复，返回每个故事材料词位置唯一一次的稳定视图。"""
    required = {
        "受试者",
        "运行编号",
        "BIDS事件行号",
        "标准词",
        "数据划分",
        "划分单元",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Pallier2025 材料视图缺少字段：{missing}")

    parts = []
    for run in RUNS:
        run_rows = table[table["运行编号"].eq(run)]
        if run_rows.empty:
            raise ValueError(f"Pallier2025 事件表缺少材料：{run}")
        source_subject = next(
            subject
            for subject in SUBJECTS
            if not (subject == "sub-09" and run == "run-03")
        )
        source = run_rows[run_rows["受试者"].eq(source_subject)].sort_values(
            "BIDS事件行号", kind="stable"
        )
        expected_rows = source["BIDS事件行号"].astype(int).tolist()
        expected_words = source["标准词"].map(normalize_french_word).tolist()
        for subject in SUBJECTS:
            if subject == "sub-09" and run == "run-03":
                continue
            current = run_rows[run_rows["受试者"].eq(subject)].sort_values(
                "BIDS事件行号", kind="stable"
            )
            if (
                current["BIDS事件行号"].astype(int).tolist() != expected_rows
                or current["标准词"].map(normalize_french_word).tolist()
                != expected_words
            ):
                raise ValueError(f"同一 Pallier2025 材料跨受试者文本不一致：{subject}/{run}")
        selected = source.copy()
        selected["_protocol_word"] = selected["标准词"].map(normalize_french_word)
        selected["_material_source_subject"] = source_subject
        parts.append(selected)
    result = pd.concat(parts, ignore_index=True)
    keys = ["运行编号", "BIDS事件行号"]
    if result.duplicated(keys).any():
        raise ValueError("Pallier2025 材料视图含重复材料词位置。")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _embedding_content_sha256(words, embeddings) -> str:
    """计算与 NPZ 容器元数据无关的词序和矩阵内容摘要。"""
    matrix = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(words), ensure_ascii=False).encode("utf-8"))
    digest.update(str(matrix.shape).encode("ascii"))
    digest.update(str(matrix.dtype).encode("ascii"))
    digest.update(matrix.tobytes(order="C"))
    return digest.hexdigest()


def _reference_t5_embeddings(words, config, tokenizer, model, device) -> np.ndarray:
    """直接逐行复现 T5 层选择和 attention-mask mean，作为数值门。"""
    import torch

    inputs = tokenizer(
        list(words),
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)
    states = outputs.hidden_states
    layer_index = int(float(config["layer_fraction"]) * len(states) - 1e-6)
    hidden = states[layer_index]
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.float().cpu().numpy().astype(np.float32, copy=False)


def validate_text_manifest(path, *, expected_config=None) -> dict:
    """验证 Pallier2025 canonical text manifest 与科学内容摘要。"""
    path = Path(path)
    payload = _validate_self_hash(_read_json(path), path=path)
    if payload.get("dataset") != DATASET_ID or payload.get("status") != "complete":
        raise ValueError(f"Pallier2025 text manifest 状态无效：{path}")
    embedding_path = path.parent / payload["embedding_file"]
    sidecar_path = path.parent / payload["sidecar_file"]
    if sha256_file(embedding_path) != payload["embedding_file_sha256"]:
        raise ValueError(f"Pallier2025 text NPZ SHA 漂移：{embedding_path}")
    if sha256_file(sidecar_path) != payload["sidecar_sha256"]:
        raise ValueError(f"Pallier2025 text sidecar SHA 漂移：{sidecar_path}")
    with np.load(embedding_path, allow_pickle=False) as product:
        words = product["words"].astype(str).tolist()
        embeddings = np.asarray(product["embeddings"])
    if embeddings.dtype != np.float32 or len(words) != len(set(words)):
        raise ValueError("Pallier2025 text dtype 或词型索引无效。")
    if _embedding_content_sha256(words, embeddings) != payload["content_sha256"]:
        raise ValueError("Pallier2025 text 内容 SHA 漂移。")
    if list(embeddings.shape) != payload["embedding_shape"]:
        raise ValueError("Pallier2025 text shape 漂移。")
    if expected_config is not None:
        sidecar = _read_json(sidecar_path)
        expected_signature = text_data.text_embedding_signature(expected_config)
        if sidecar.get("signature") != expected_signature:
            raise ValueError("Pallier2025 text cache signature 与配置不一致。")
    return payload


def build_text_product(event_table, config, output_path, *, event_table_path, force=False):
    """通过 reference gate 和双重重建生成 canonical T5-large 文本产品。"""
    output_path = Path(output_path)
    manifest_path = output_path.parent / "manifest.json"
    if manifest_path.exists() and not force:
        manifest = validate_text_manifest(
            manifest_path, expected_config=config
        )
        if manifest.get("event_table_sha256") != sha256_file(event_table_path):
            raise ValueError("Pallier2025 text product 的事件表合同已经变化。")
        return manifest_path

    words = text_data.prepared_words(event_table["标准词"], config)
    tokenizer, model, device = text_data.load_text_model(config)
    fixture_words = list(FRENCH_REFERENCE_TOKENS)
    direct = _reference_t5_embeddings(
        fixture_words, config, tokenizer, model, device
    )
    shared = text_data.encode_words(
        fixture_words,
        config,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    difference = np.abs(direct - shared)
    if not np.array_equal(direct, shared):
        raise ValueError("Pallier2025 T5 reference fixture 未达到逐元素一致，停止全量物化。")

    rebuild_a = text_data.encode_words(
        words, config, tokenizer=tokenizer, model=model, device=device
    )
    rebuild_b = text_data.encode_words(
        words, config, tokenizer=tokenizer, model=model, device=device
    )
    content_a = _embedding_content_sha256(words, rebuild_a)
    content_b = _embedding_content_sha256(words, rebuild_b)
    if not np.array_equal(rebuild_a, rebuild_b) or content_a != content_b:
        raise ValueError("Pallier2025 canonical T5 重建自身不稳定，停止保存正式产品。")

    text_data.save_text_embedding_cache(output_path, words, rebuild_a, config)
    rebuild_b_path = output_path.with_name(output_path.stem + ".rebuild_b.npz")
    text_data.save_text_embedding_cache(rebuild_b_path, words, rebuild_b, config)
    file_a = sha256_file(output_path)
    file_b = sha256_file(rebuild_b_path)
    rebuild_b_path.unlink()
    rebuild_b_path.with_suffix(".json").unlink()
    sidecar_path = output_path.with_suffix(".json")

    import torch

    payload = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "status": "complete",
        "model_name": str(config["model_name"]),
        "model_identity": str(
            getattr(model.config, "_name_or_path", config["model_name"])
        ),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "layer_fraction": float(config["layer_fraction"]),
        "selected_hidden_layer": int(
            float(config["layer_fraction"])
            * (int(getattr(model.config, "num_layers")) + 1)
            - 1e-6
        ),
        "token_aggregation": str(config["token_aggregation"]),
        "embedding_dimension": int(config["embedding_dimension"]),
        "dtype": "float32",
        "normalization": "strip_lower_preserve_french_accents_and_tokenization",
        "word_type_count": len(words),
        "embedding_shape": list(rebuild_a.shape),
        "embedding_file": output_path.name,
        "embedding_file_sha256": file_a,
        "sidecar_file": sidecar_path.name,
        "sidecar_sha256": sha256_file(sidecar_path),
        "content_sha256": content_a,
        "event_table_sha256": sha256_file(event_table_path),
        "reference_fixture": {
            "tokens": fixture_words,
            "shape": list(direct.shape),
            "different_element_count": int(np.count_nonzero(difference)),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "bitwise_equal": True,
            "reference_content_sha256": _embedding_content_sha256(
                fixture_words, direct
            ),
            "shared_content_sha256": _embedding_content_sha256(
                fixture_words, shared
            ),
        },
        "canonical_rebuild": {
            "word_order_stable": True,
            "shape_stable": True,
            "bitwise_equal": True,
            "content_sha256_a": content_a,
            "content_sha256_b": content_b,
            "container_file_sha256_a": file_a,
            "container_file_sha256_b": file_b,
            "container_file_stable": file_a == file_b,
        },
        "software": {
            "transformers": importlib.metadata.version("transformers"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "tokenizer_identity": type(tokenizer).__name__,
        "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "model_class": type(model).__name__,
        "local_files_only": bool(config.get("local_files_only", False)),
        "test_model_evaluation": "not_run",
        "test_predictions_generated": False,
        "test_metrics_inspected": False,
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    _write_json_atomic(manifest_path, payload)
    validate_text_manifest(manifest_path)
    return manifest_path


def _with_manifest_sha256(payload: dict) -> dict:
    result = dict(payload)
    result["manifest_sha256"] = stable_sha256(result)
    return result


def build_protocol_assets(event_table_path) -> dict[str, dict]:
    """从 frozen event table 构造 Pallier 词表、reference 与支持审计。"""
    event_table_path = Path(event_table_path)
    table = load_event_table(event_table_path)
    material = material_word_view(table)
    material = material[material["_protocol_word"].ne("")].reset_index(drop=True)
    event_digest = sha256_file(event_table_path)
    generator_digest = sha256_file(Path(__file__))

    train = material[material["数据划分"].eq("train")].reset_index(drop=True)
    counts = Counter(train["_protocol_word"])
    ranked = sorted(counts, key=lambda word: (-int(counts[word]), word))
    source_units = sorted(train["划分单元"].astype(str).unique())
    vocabularies = {}
    documents = {}
    for size in (20, 50, 100, 150):
        vocabulary = ranked[:size]
        if len(vocabulary) != size:
            raise ValueError(f"Pallier2025 train 材料不足以冻结 N={size} 词表。")
        manifest = _with_manifest_sha256(
            {
                "asset_type": "candidate_vocabulary",
                "created_from_test": False,
                "dataset": DATASET_ID,
                "evaluation_support_used_for_selection": False,
                "event_table": "data/cache/derived/pallier2025/events/events.csv",
                "event_table_sha256": event_digest,
                "frequency_unit": "material_word_occurrence",
                "generator": "braindecoding.data.pallier2025.build_protocol_assets",
                "generator_sha256": generator_digest,
                "hash_contract": "sha256(canonical_json_without_manifest_sha256)",
                "normalization": "strip_lower_preserve_french_accents_and_tokenization",
                "policy": "frequency_desc_word_asc",
                "scope": CORE_SCOPE,
                "source_split": "train",
                "source_split_units": source_units,
                "source_units": source_units,
                "status": "frozen",
                "train_cutoff_frequency": int(counts[vocabulary[-1]]),
                "train_token_coverage": float(
                    train["_protocol_word"].isin(vocabulary).mean()
                ),
                "vocabulary": vocabulary,
                "vocabulary_size": size,
                "word_counts": {word: int(counts[word]) for word in vocabulary},
            }
        )
        vocabularies[size] = manifest
        documents[f"vocabulary_N{size}.json"] = manifest

    support = {}
    for size, vocabulary_manifest in vocabularies.items():
        vocabulary = list(vocabulary_manifest["vocabulary"])
        split_records = {}
        for split in ("val", "test"):
            words = material.loc[
                material["数据划分"].eq(split), "_protocol_word"
            ].tolist()
            observed = set(words)
            missing = [word for word in vocabulary if word not in observed]
            split_records[split] = {
                "candidate_count": size,
                "supported_candidate_count": size - len(missing),
                "missing_candidate_count": len(missing),
                "missing_words": missing,
                "support_fraction": (size - len(missing)) / size,
                "token_count": len(words),
                "token_coverage": float(
                    sum(word in set(vocabulary) for word in words) / len(words)
                ),
            }
        support[f"N{size}"] = {
            "candidate_manifest_sha256": vocabulary_manifest["manifest_sha256"],
            "splits": split_records,
        }
    documents["vocabulary_support.json"] = _with_manifest_sha256(
        {
            "asset_type": "evaluation_vocabulary_support",
            "dataset": DATASET_ID,
            "generator": "braindecoding.data.pallier2025.build_protocol_assets",
            "generator_sha256": generator_digest,
            "status": "frozen_support_audit",
            "support_audit_used_for_vocabulary_selection": False,
            "test_labels_used_for_support_audit_only": True,
            "test_model_evaluation": "not_run",
            "test_predictions_generated": False,
            "test_metrics_inspected": False,
            "vocabularies": support,
        }
    )

    return documents


def write_protocol_assets(event_table_path, output_dir) -> list[Path]:
    """原子写入 Pallier2025 独立的机器可读协议资产。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, payload in sorted(build_protocol_assets(event_table_path).items()):
        path = output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        paths.append(path)
    return paths


def _read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_self_hash(payload: dict, *, path) -> dict:
    body = dict(payload)
    recorded = body.pop("manifest_sha256", None) or body.pop(
        "artifact_sha256", None
    )
    if stable_sha256(body) != recorded:
        raise ValueError(f"文件自摘要不一致：{path}")
    return payload


def deterministic_run_split(runs=RUNS, *, salt=RUN_SPLIT_SALT):
    """只根据 run 标识的加盐 SHA-256 排名产生 7/1/1 split。"""
    ranked = sorted(
        (
            hashlib.sha256(f"{salt}|{run}".encode("utf-8")).hexdigest(),
            str(run),
        )
        for run in runs
    )
    if len(ranked) != 9 or len({run for _, run in ranked}) != 9:
        raise ValueError("Pallier2025 主 split 需要九个唯一 run。")
    split_by_run = {
        run: "train" if rank < 7 else "val" if rank == 7 else "test"
        for rank, (_, run) in enumerate(ranked)
    }
    ranking = [
        {"rank": rank, "run": run, "sha256": digest}
        for rank, (digest, run) in enumerate(ranked)
    ]
    return split_by_run, ranking


def load_run_split(path) -> dict[str, str]:
    """读取并验证冻结的 run-level split manifest。"""
    payload = _validate_self_hash(_read_json(path), path=path)
    if payload.get("dataset") != DATASET_ID:
        raise ValueError(f"split manifest 数据集不是 {DATASET_ID}：{path}")
    assignments = payload.get("assignments", {})
    split_by_run = {
        run: split
        for split in ("train", "val", "test")
        for run in assignments.get(split, ())
    }
    if set(split_by_run) != set(RUNS):
        raise ValueError("Pallier2025 split 必须恰好覆盖 run-01 至 run-09。")
    if [len(assignments.get(split, ())) for split in ("train", "val", "test")] != [
        7,
        1,
        1,
    ]:
        raise ValueError("Pallier2025 主 split 必须是 7 train / 1 val / 1 test。")
    expected_split, expected_ranking = deterministic_run_split(
        RUNS, salt=payload["strategy"]["salt"]
    )
    if split_by_run != expected_split or payload.get("hash_ranking") != expected_ranking:
        raise ValueError("Pallier2025 split 与冻结的 deterministic 算法不一致。")
    return split_by_run


def discover_subjects(raw_root) -> list[str]:
    """按 BIDS 名称排序发现实际受试者，不根据数量猜测编号。"""
    root = Path(raw_root)
    return sorted(
        path.name
        for path in root.glob("sub-*")
        if path.is_dir() and path.name[4:].isdigit()
    )


def recording_paths(raw_root, subject: str, run: str) -> dict[str, Path]:
    """返回一条 BIDS recording 及其必需 sidecar 路径。"""
    root = Path(raw_root)
    stem = f"{subject}_{SESSION}_task-listen_{run}"
    meg_dir = root / subject / SESSION / "meg"
    return {
        "meg": meg_dir / f"{stem}_meg.fif",
        "events": meg_dir / f"{stem}_events.tsv",
        "channels": meg_dir / f"{stem}_channels.tsv",
        "meg_json": meg_dir / f"{stem}_meg.json",
        "coordsystem": meg_dir / f"{subject}_{SESSION}_coordsystem.json",
        "extra_info": root / "sourcedata" / f"task-listen_{run}_extra_info.tsv",
    }


def discover_recordings(raw_root, subjects=SUBJECTS, runs=RUNS) -> list[dict]:
    """验证 10×9 recording 的 BIDS metadata 文件完整性。"""
    records = []
    for subject in subjects:
        for run in runs:
            paths = recording_paths(raw_root, subject, run)
            missing = [name for name, path in paths.items() if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"{subject}/{run} 缺少 BIDS 文件：{missing}"
                )
            records.append({"subject": subject, "run": run, "paths": paths})
    return records


def read_recording_header(meg_path) -> dict:
    """只读取 FIF header；不载入或预处理 MEG 数值。"""
    try:
        import mne
    except ImportError as error:  # pragma: no cover - 环境错误分支
        raise RuntimeError("读取 Pallier2025 FIF header 需要安装 MNE。") from error

    raw = mne.io.read_raw_fif(
        meg_path,
        preload=False,
        allow_maxshield=True,
        verbose="ERROR",
    )
    try:
        sfreq = float(raw.info["sfreq"])
        return {
            "sampling_rate_hz": sfreq,
            "first_sample": int(raw.first_samp),
            "last_sample": int(raw.last_samp),
            "first_time_seconds": float(raw.first_samp / sfreq),
            "last_time_seconds": float(raw.last_samp / sfreq),
            "sample_count": int(raw.n_times),
            "channel_count": int(len(raw.ch_names)),
            "channel_names_sha256": stable_sha256(list(raw.ch_names)),
        }
    finally:
        raw.close()


def _read_extra_info(path) -> tuple[pd.DataFrame, int]:
    # 文件扩展名是 TSV，但该数据发布版本实际使用逗号分隔。
    header = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    separator = "\t" if header.count("\t") > header.count(",") else ","
    table = pd.read_csv(path, sep=separator)
    if "word" not in table:
        raise ValueError(f"extra_info 缺少 word 字段：{path}")
    words = table["word"].map(normalize_french_word)
    blank_count = int(words.eq("").sum())
    filtered = table.loc[words.ne("")].copy().reset_index(drop=False)
    filtered["_标准词"] = words.loc[words.ne("")].to_numpy()
    filtered["_extra_word_index"] = np.arange(len(filtered), dtype=int)
    return filtered, blank_count


def _bool_value(value, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"extra_info {field} 不是布尔值：{value!r}")


def load_qc_artifact(path, raw_root) -> dict:
    """验证不可变 QC artifact 及其原始来源摘要。"""
    payload = _validate_self_hash(_read_json(path), path=path)
    if payload.get("dataset") != DATASET_ID:
        raise ValueError(f"QC artifact 数据集不是 {DATASET_ID}：{path}")
    for source in payload.get("source_files", ()):
        source_path = Path(raw_root) / source["path"]
        if sha256_file(source_path).lower() != str(source["sha256"]).lower():
            raise ValueError(f"QC artifact 来源 SHA 漂移：{source_path}")
    return payload


def _exact_alignment_summary(left_words, right_words) -> dict:
    return {
        "left_length": len(left_words),
        "right_length": len(right_words),
        "equal": True,
        "matches": len(left_words),
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
        "edit_distance": 0,
        "normalized_edit_distance": 0.0,
        "first_edit_left_index": None,
        "first_edit_right_index": None,
        "left_fingerprint": text_fingerprint(left_words),
        "right_fingerprint": text_fingerprint(right_words),
    }


def _alignment_mapping(
    bids_words: list[str],
    extra_words: list[str],
    overrides: dict[int, int],
) -> list[int]:
    """由确定性 sequence alignment 构造 BIDS row 到 extra word 的映射。"""
    mapping: dict[int, int] = {}
    matcher = SequenceMatcher(None, bids_words, extra_words, autojunk=False)
    for operation, left_start, left_stop, right_start, right_stop in matcher.get_opcodes():
        if operation == "equal":
            for left_index, right_index in zip(
                range(left_start, left_stop), range(right_start, right_stop)
            ):
                mapping[left_index] = right_index
    mapping.update(overrides)
    if set(mapping) != set(range(len(bids_words))):
        missing = sorted(set(range(len(bids_words))) - set(mapping))
        raise ValueError(f"extra_info 对齐后仍有未映射的 BIDS event：{missing[:10]}")
    values = [mapping[index] for index in range(len(bids_words))]
    if len(set(values)) != len(extra_words):
        raise ValueError("extra_info 对齐不是一一映射。")
    for left_index, right_index in enumerate(values):
        if bids_words[left_index] != extra_words[right_index]:
            raise ValueError(
                f"extra_info 映射词不一致：BIDS row {left_index} -> extra {right_index}"
            )
    return values


def align_recording_metadata(
    bids_events: pd.DataFrame,
    extra_info: pd.DataFrame,
    *,
    subject: str,
    run: str,
    blank_token_count: int,
    qc_artifact: dict | None = None,
) -> tuple[pd.DataFrame, dict, bool]:
    """按词序列对齐 BIDS events 与 extra_info；绝不按行号直接 join。"""
    required_events = {"onset", "duration", "trial_type", "stimulus"}
    required_extra = {
        "sequence_id",
        "is_last_word",
        "pos",
        "content_word",
        "n_closing",
        "_标准词",
    }
    if missing := sorted(required_events - set(bids_events)):
        raise ValueError(f"BIDS events 缺少字段：{missing}")
    if missing := sorted(required_extra - set(extra_info)):
        raise ValueError(f"extra_info 缺少字段：{missing}")

    bids_words = bids_events["stimulus"].map(normalize_french_word).tolist()
    extra_words = extra_info["_标准词"].astype(str).tolist()
    if bids_words == extra_words:
        summary = _exact_alignment_summary(bids_words, extra_words)
        mapping = list(range(len(bids_words)))
        recording_excluded = False
        decision = "exact"
    else:
        summary = align_text_sequences(bids_words, extra_words)
        expected = None if qc_artifact is None else qc_artifact.get("recording")
        if not expected or (
            expected.get("subject"), expected.get("run")
        ) != (subject, run):
            raise ValueError(
                f"{subject}/{run} 出现未经批准的 extra_info 对齐差异："
                f"edit_distance={summary['edit_distance']}"
            )
        if qc_artifact.get("decision") != "exclude_recording":
            raise ValueError(f"{subject}/{run} QC artifact 没有冻结排除决定。")
        overrides = {
            int(item["bids_event_row"]): int(item["extra_word_index"])
            for item in qc_artifact.get("metadata_mapping_overrides", ())
        }
        mapping = _alignment_mapping(bids_words, extra_words, overrides)
        recording_excluded = True
        decision = "approved_mismatch_recording_excluded"

    aligned = extra_info.iloc[mapping].reset_index(drop=True)
    audit = {
        "subject": subject,
        "run": run,
        "bids_event_count": int(len(bids_events)),
        "extra_info_row_count": int(len(extra_info) + blank_token_count),
        "extra_info_blank_token_count": int(blank_token_count),
        "extra_info_word_count": int(len(extra_info)),
        "decision": decision,
        **summary,
    }
    return aligned, audit, recording_excluded


def _format_sequence_id(value) -> tuple[int, str]:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or float(numeric) != int(numeric):
        raise ValueError(f"sequence_id 不是整数：{value!r}")
    original = int(numeric)
    return original, f"sequence-{original:03d}"


def _recording_table(
    bids_events: pd.DataFrame,
    aligned_extra: pd.DataFrame,
    *,
    subject: str,
    run: str,
    split: str,
    header: dict,
    recording_excluded: bool,
    exclusion_reason: str,
) -> pd.DataFrame:
    if len(bids_events) != len(aligned_extra):
        raise ValueError(f"{subject}/{run} BIDS 与 extra_info 对齐长度不一致。")
    result = pd.DataFrame(index=np.arange(len(bids_events)))
    words = bids_events["stimulus"].astype(str)
    normalized = words.map(normalize_french_word)
    starts = pd.to_numeric(bids_events["onset"], errors="coerce")
    durations = pd.to_numeric(bids_events["duration"], errors="coerce")
    if starts.isna().any() or durations.isna().any():
        raise ValueError(f"{subject}/{run} BIDS onset/duration 存在非数值。")

    recording_id = f"{DATASET_ID}|{subject}|{run}"
    material_id = f"{DATASET_ID}|{run}"
    sequence_values = [_format_sequence_id(value) for value in aligned_extra["sequence_id"]]
    valid_text = normalized.ne("") & bids_events["trial_type"].astype(str).eq("Word")
    valid_alignment = pd.Series(not recording_excluded, index=result.index)
    window_eligible = starts.ge(float(header["first_time_seconds"])) & (
        starts + ELIGIBILITY_WINDOW_SECONDS
    ).le(float(header["last_time_seconds"]))
    trainable = valid_text & valid_alignment & window_eligible

    reasons = []
    for row_index in result.index:
        row_reasons = []
        if not valid_text.iloc[row_index]:
            row_reasons.append("BIDS Word 文本为空或 trial_type 不是 Word")
        if recording_excluded:
            row_reasons.append(exclusion_reason)
        if not window_eligible.iloc[row_index]:
            row_reasons.append("3 秒输入窗口超出当前 recording 的有效采样范围")
        reasons.append("；".join(row_reasons))

    result["事件编号"] = [
        f"{recording_id}|event-{index:04d}" for index in result.index
    ]
    result["受试者"] = subject
    result["记录编号"] = recording_id
    result["材料编号"] = material_id
    result["划分单元"] = material_id
    result["记录内序号"] = result.index.astype(int)
    result["词"] = words.to_numpy()
    result["标准词"] = normalized.to_numpy()
    result["开始时间"] = starts.to_numpy(dtype=float)
    result["结束时间"] = (starts + durations).to_numpy(dtype=float)
    result["上下文编号"] = [
        f"{DATASET_ID}|{run}|{sequence_name}"
        for _, sequence_name in sequence_values
    ]
    result["数据划分"] = split
    result["是否可训练"] = trainable.to_numpy(dtype=bool)
    result["排除原因"] = reasons
    result["会话"] = SESSION
    result["运行编号"] = run
    result["BIDS事件行号"] = result.index.astype(int)
    result["序列原编号"] = [value for value, _ in sequence_values]
    result["是否序列末词"] = [
        _bool_value(value, field="is_last_word")
        for value in aligned_extra["is_last_word"]
    ]
    result["词性"] = aligned_extra["pos"].fillna("").astype(str).to_numpy()
    result["是否内容词"] = [
        _bool_value(value, field="content_word")
        for value in aligned_extra["content_word"]
    ]
    closing = pd.to_numeric(aligned_extra["n_closing"], errors="coerce")
    if closing.isna().any():
        raise ValueError(f"{subject}/{run} extra_info n_closing 存在非数值。")
    result["右括号数量"] = closing.astype(int).to_numpy()
    return result.loc[:, list(PERSISTED_COLUMNS)]


def build_event_table(dataset_config: dict) -> tuple[pd.DataFrame, dict]:
    """只读 BIDS events/header 与 extra_info，构建 canonical 中文事件表。"""
    root = Path(dataset_config["root"])
    subjects = list(dataset_config.get("subjects", SUBJECTS))
    runs = list(dataset_config.get("runs", RUNS))
    discovered = discover_subjects(root)
    if discovered != subjects:
        raise ValueError(
            f"Pallier2025 受试者集合或顺序漂移：expected={subjects}, actual={discovered}"
        )
    records = discover_recordings(root, subjects, runs)
    split_by_run = load_run_split(dataset_config["split_manifest"])
    qc_artifact = load_qc_artifact(dataset_config["qc_artifact"], root)
    qc_key = (
        qc_artifact["recording"]["subject"],
        qc_artifact["recording"]["run"],
    )

    extra_by_run = {}
    rows = []
    alignments = []
    recording_metadata = []
    exclusion_reason = str(qc_artifact["application"]["exclusion_reason"])
    for record in records:
        subject = record["subject"]
        run = record["run"]
        paths = record["paths"]
        if run not in extra_by_run:
            extra_by_run[run] = _read_extra_info(paths["extra_info"])
        extra_info, blank_count = extra_by_run[run]
        bids_events = pd.read_csv(paths["events"], sep="\t")
        current_artifact = qc_artifact if (subject, run) == qc_key else None
        aligned, alignment, recording_excluded = align_recording_metadata(
            bids_events,
            extra_info,
            subject=subject,
            run=run,
            blank_token_count=blank_count,
            qc_artifact=current_artifact,
        )
        header = read_recording_header(paths["meg"])
        rows.append(
            _recording_table(
                bids_events,
                aligned,
                subject=subject,
                run=run,
                split=split_by_run[run],
                header=header,
                recording_excluded=recording_excluded,
                exclusion_reason=exclusion_reason,
            )
        )
        alignment["events_sha256"] = sha256_file(paths["events"])
        alignment["extra_info_sha256"] = sha256_file(paths["extra_info"])
        alignments.append(alignment)
        recording_metadata.append(
            {
                "recording_id": f"{DATASET_ID}|{subject}|{run}",
                **header,
            }
        )

    table = pd.concat(rows, ignore_index=True)
    validate_event_table(table)

    trainable = table[table["是否可训练"]]
    material_fingerprints = (
        trainable.sort_values(["受试者", "运行编号", "记录内序号"], kind="stable")
        .groupby(["材料编号", "受试者"], sort=False)["标准词"]
        .agg(text_fingerprint)
    )
    inconsistent = material_fingerprints.groupby(level=0).nunique()
    if inconsistent.gt(1).any():
        raise ValueError(
            "Pallier2025 同一 run 的标准词序列在可训练受试者间不一致。"
        )

    contexts = table.drop_duplicates(["材料编号", "上下文编号"])
    context_sizes = (
        table.drop_duplicates(["材料编号", "记录内序号"])
        .groupby("上下文编号", sort=False)
        .size()
    )
    audit = {
        "recording_count": len(records),
        "alignment_audits": alignments,
        "alignment_summary": {
            "exact_recordings": sum(item["equal"] for item in alignments),
            "approved_mismatch_recordings": sum(not item["equal"] for item in alignments),
            "total_substitutions": sum(item["substitutions"] for item in alignments),
            "total_insertions": sum(item["insertions"] for item in alignments),
            "total_deletions": sum(item["deletions"] for item in alignments),
        },
        "material_text_consistency": {
            "same_run_across_trainable_subjects": True,
            "excluded_recordings": [f"{DATASET_ID}|{qc_key[0]}|{qc_key[1]}"],
        },
        "context_statistics": {
            "context_count": int(contexts["上下文编号"].nunique()),
            "mean_words": float(context_sizes.mean()),
            "median_words": float(context_sizes.median()),
            "maximum_words": int(context_sizes.max()),
        },
        "recording_metadata": recording_metadata,
        "qc_decision": {
            "artifact_id": qc_artifact["artifact_id"],
            "artifact_sha256": qc_artifact["artifact_sha256"],
            "recording": qc_artifact["recording"],
            "decision": qc_artifact["decision"],
            "correction_applied_to_bids_fields": False,
        },
        "bids_onset_is_only_neural_time": True,
        "extra_info_onset_persisted": False,
        "signal_materialization_performed": False,
        "model_evaluation_performed": False,
    }
    return table, audit


def load_event_table(
    path,
    *,
    split=None,
    subjects=None,
    trainable_only=False,
) -> pd.DataFrame:
    """读取并验证 Pallier2025 canonical 事件表，可显式筛选训练消费者范围。"""
    table = pd.read_csv(path, keep_default_na=False)
    for column in ("是否可训练", "是否序列末词", "是否内容词"):
        if column in table:
            table[column] = table[column].map(
                lambda value: _bool_value(value, field=column)
            )
    validate_event_table(table)
    if split is not None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Pallier2025 数据划分无效：{split}")
        table = table[table["数据划分"].eq(split)]
    if subjects is not None:
        requested = tuple(str(value) for value in subjects)
        unknown = sorted(set(requested) - set(SUBJECTS))
        if unknown:
            raise ValueError(f"Pallier2025 配置含未知受试者：{unknown}")
        table = table[table["受试者"].isin(requested)]
    if trainable_only:
        table = table[table["是否可训练"].astype(bool)]
    return table.reset_index(drop=True)


def _signal_product_paths(cache_dir, subject, run) -> tuple[Path, Path]:
    """返回一条 canonical Pall signal 及其 sidecar 路径。"""
    stem = f"{subject}_{SESSION}_task-listen_{run}_meg"
    signal_path = Path(cache_dir) / str(subject) / f"{stem}.npy"
    return signal_path, signal_path.with_suffix(".json")


def add_runtime_context_uid(table, max_context_words=128):
    """建立受试者记录内的运行时上下文键，并验证序列边界。

    Canonical ``上下文编号`` 保持材料级语言上下文身份；模型运行时额外把
    ``记录编号`` 纳入分组，避免同一材料在不同受试者间发生注意力交互。
    """
    required = {"受试者", "记录编号", "上下文编号", "记录内序号"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Pallier2025 运行时上下文缺少字段：{missing}")
    maximum = int(max_context_words)
    if maximum <= 0:
        raise ValueError("Pallier2025 max_context_words 必须为正整数。")

    result = table.copy()
    result[RUNTIME_CONTEXT_COLUMN] = (
        result["记录编号"].astype(str)
        + "|"
        + result["上下文编号"].astype(str)
    )
    grouped = result.groupby(RUNTIME_CONTEXT_COLUMN, sort=False)
    if grouped["受试者"].nunique().gt(1).any():
        raise ValueError("Pallier2025 运行时上下文跨受试者。")
    if grouped["记录编号"].nunique().gt(1).any():
        raise ValueError("Pallier2025 运行时上下文跨 recording。")
    if grouped["上下文编号"].nunique().gt(1).any():
        raise ValueError("Pallier2025 运行时上下文混合多个 canonical context。")
    sizes = grouped.size()
    oversized = sizes[sizes.gt(maximum)]
    if not oversized.empty:
        raise ValueError(
            f"Pallier2025 运行时上下文超过 max_context_words={maximum}："
            f"{oversized.index[0]}={int(oversized.iloc[0])}"
        )
    for context_uid, rows in grouped:
        order = pd.to_numeric(rows["记录内序号"], errors="coerce")
        if order.isna().any() or (np.diff(order.to_numpy()) <= 0).any():
            raise ValueError(
                f"Pallier2025 运行时上下文记录内序号未严格递增：{context_uid}"
            )
    return result


def runtime_context_statistics(table, max_context_words=128) -> dict:
    """汇总 Pallier 受试者记录内运行时上下文的只读统计。"""
    runtime = add_runtime_context_uid(table, max_context_words=max_context_words)
    grouped = runtime.groupby(RUNTIME_CONTEXT_COLUMN, sort=False)
    sizes = grouped.size()
    return {
        "context_count": int(len(sizes)),
        "mean_words": float(sizes.mean()) if len(sizes) else 0.0,
        "median_words": float(sizes.median()) if len(sizes) else 0.0,
        "maximum_words": int(sizes.max()) if len(sizes) else 0,
        "cross_subject_groups": int(grouped["受试者"].nunique().gt(1).sum()),
        "cross_recording_groups": int(
            grouped["记录编号"].nunique().gt(1).sum()
        ),
        "cross_canonical_context_groups": int(
            grouped["上下文编号"].nunique().gt(1).sum()
        ),
        "groups_over_maximum": int(sizes.gt(int(max_context_words)).sum()),
    }


@lru_cache(maxsize=256)
def _open_signal_product(path: str):
    """只读映射 canonical continuous signal，避免反复载入完整记录。"""
    return np.load(path, mmap_mode="r")


class Pallier2025WordDataset(Dataset):
    """从冻结事件、连续信号和 T5-large 词向量构造 1 秒词样本。"""

    def __init__(
        self,
        table,
        dataset_config,
        meg_cache_dir,
        text_embedding_path,
        text_embedding_config,
        *,
        zero_meg=False,
        temporal_shift_seconds: float = 0.0,
        allow_subject_subset=False,
    ):
        source = table.copy()
        missing = sorted(set(CORE_EVENT_COLUMNS) - set(source.columns))
        if missing:
            raise ValueError(f"Pallier2025 Dataset 缺少核心事件字段：{missing}")
        # Dataset 永远不暴露冻结 QC 已排除或窗口不完整的事件。
        source = source[source["是否可训练"].astype(bool)].reset_index(drop=True)
        if source.empty:
            raise ValueError("Pallier2025 Dataset 没有可训练事件。")

        configured_subjects = tuple(str(value) for value in dataset_config["subjects"])
        if allow_subject_subset:
            if (
                not configured_subjects
                or tuple(subject for subject in SUBJECTS if subject in configured_subjects)
                != configured_subjects
            ):
                raise ValueError(
                    "Pallier2025 subject subset 必须非空且保持 canonical 顺序。"
                )
        elif configured_subjects != SUBJECTS:
            raise ValueError(
                "Pallier2025 subject 顺序必须固定为 sub-01 至 sub-10。"
            )
        observed_subjects = set(source["受试者"].astype(str))
        if not observed_subjects.issubset(set(configured_subjects)):
            raise ValueError("Pallier2025 事件表含配置外受试者。")

        # Canonical 上下文身份不变；运行时分组严格限制在一条受试者 recording 内。
        self.table = add_runtime_context_uid(
            source,
            max_context_words=int(dataset_config.get("max_context_words", 128)),
        )
        # 仅在内存中恢复训练批处理所需英文 key；不污染 canonical CSV。
        aliases = {
            "event_id": "事件编号",
            "subject_id": "受试者",
            "recording_id": "记录编号",
            "word": "词",
            "normalized_word": "标准词",
            "sentence_uid": "上下文编号",
            "split": "数据划分",
            "is_trainable": "是否可训练",
            "onset_seconds": "开始时间",
        }
        for target, origin in aliases.items():
            self.table[target] = self.table[origin]

        self.dataset_config = dict(dataset_config)
        self.meg_cache_dir = Path(meg_cache_dir)
        self.zero_meg = bool(zero_meg)
        contract = word_window_contract(dataset_config)
        self.window_seconds = float(contract["window_seconds"])
        self.window_samples = int(contract["window_samples"])
        self._baseline_samples = int(contract["baseline_samples"])
        self._clamp = float(contract["clamp"])
        self._window_offset_seconds = float(contract["window_start_offset_seconds"])
        self.sampling_rate_hz = float(
            contract.get("sampling_rate_hz", TARGET_SAMPLING_RATE_HZ)
        )
        self.temporal_shift_seconds = float(temporal_shift_seconds)
        self.temporal_shift_samples = int(
            round(self.temporal_shift_seconds * self.sampling_rate_hz)
        )
        self.channel_count = MEG_CHANNEL_COUNT
        self.subject_map = {
            subject: index for index, subject in enumerate(configured_subjects)
        }
        self.subject_count = len(self.subject_map)
        self.subject_indices = np.asarray(
            [self.subject_map[value] for value in self.table["subject_id"].astype(str)],
            dtype=np.int64,
        )
        contexts = self.table[RUNTIME_CONTEXT_COLUMN].astype(str).tolist()
        context_map = {
            value: index for index, value in enumerate(dict.fromkeys(contexts))
        }
        self.sentence_indices = np.asarray(
            [context_map[value] for value in contexts], dtype=np.int64
        )

        signature = text_data.text_embedding_signature(text_embedding_config)
        self.embedding_map = text_data.load_text_embedding_cache(
            text_embedding_path, expected_signature=signature
        )
        expected_dimension = int(text_embedding_config["embedding_dimension"])
        missing_words = sorted(
            set(self.table["normalized_word"].astype(str)) - set(self.embedding_map)
        )
        if missing_words:
            raise KeyError(f"Pallier2025 文本缓存缺少词型：{missing_words[:5]}")
        if any(
            value.shape != (expected_dimension,)
            for value in self.embedding_map.values()
        ):
            raise ValueError("Pallier2025 文本向量维数与配置不一致。")

        self.signal_paths = {}
        self.signal_metadata = {}
        channel_names = None
        for row in self.table[
            ["recording_id", "subject_id", "运行编号"]
        ].drop_duplicates().itertuples(index=False):
            recording_id, subject, run = map(str, row)
            signal_path, sidecar_path = _signal_product_paths(
                self.meg_cache_dir, subject, run
            )
            if not signal_path.is_file() or not sidecar_path.is_file():
                raise FileNotFoundError(
                    f"Pallier2025 canonical signal product 缺失：{signal_path}"
                )
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if metadata.get("recording_id") != recording_id:
                raise ValueError(
                    f"Pallier2025 signal sidecar 记录编号不一致：{sidecar_path}"
                )
            if (
                int(metadata.get("channel_count", -1)) != MEG_CHANNEL_COUNT
                or metadata.get("channel_names_sha256") != CHANNEL_NAMES_SHA256
                or float(metadata.get("output_sampling_rate_hz", -1))
                != TARGET_SAMPLING_RATE_HZ
                or metadata.get("dtype") != "float32"
            ):
                raise ValueError(f"Pallier2025 signal sidecar 合同不一致：{sidecar_path}")
            current_names = tuple(metadata["channel_names"])
            channel_names = current_names if channel_names is None else channel_names
            if current_names != channel_names:
                raise ValueError("Pallier2025 recording 间通道顺序不一致。")
            self.signal_paths[recording_id] = signal_path
            self.signal_metadata[recording_id] = metadata

        self.channel_names = list(channel_names or ())
        if channel_names_sha256(self.channel_names) != CHANNEL_NAMES_SHA256:
            raise ValueError("Pallier2025 Dataset 通道顺序 SHA 不一致。")
        self.channel_positions = vectorview_channel_positions(
            self.channel_names, self.dataset_config.get("layout_path")
        )

        # 预提取行级索引数组，避免每个样本在 DataLoader 中重复调用 DataFrame.iloc
        self._recording_ids = tuple(self.table["recording_id"].astype(str).tolist())
        self._words = tuple(self.table["normalized_word"].astype(str).tolist())
        self._event_ids = tuple(self.table["event_id"].astype(str).tolist())
        self._onset_seconds = self.table["onset_seconds"].to_numpy(
            dtype=np.float64, copy=True
        )
        self._signal_path_strs = tuple(
            str(self.signal_paths[rec_id]) for rec_id in self._recording_ids
        )
        self._window_starts = np.asarray(
            [
                recording_onset_to_sample(
                    float(onset) + self._window_offset_seconds,
                    source_first_samp=self.signal_metadata[rec_id]["source_first_samp"],
                    source_sampling_rate_hz=self.signal_metadata[rec_id][
                        "source_sampling_rate_hz"
                    ],
                    output_sampling_rate_hz=self.signal_metadata[rec_id][
                        "output_sampling_rate_hz"
                    ],
                )
                for rec_id, onset in zip(self._recording_ids, self._onset_seconds)
            ],
            dtype=np.int64,
        )
        self._embedding_tensors = {
            word: torch.from_numpy(np.array(vec, dtype=np.float32, copy=True))
            for word, vec in self.embedding_map.items()
        }
        self._subject_index_tensors = torch.as_tensor(
            self.subject_indices, dtype=torch.long
        )
        self._sentence_index_tensors = torch.as_tensor(
            self.sentence_indices, dtype=torch.long
        )

    def __len__(self):
        return len(self.table)

    def read_meg(self, index) -> np.ndarray:
        idx = int(index)
        if self.zero_meg:
            return np.zeros(
                (self.channel_count, self.window_samples), dtype=np.float32
            )
        recording = _open_signal_product(self._signal_path_strs[idx])
        if recording.ndim != 2 or recording.shape[1] < self.window_samples:
            raise ValueError("Pallier2025 词窗口超出 canonical continuous signal。")
        start = int(self._window_starts[idx])
        stop = start + self.window_samples
        if start < 0 or stop > recording.shape[1]:
            if self.window_samples == 50 and self._window_offset_seconds == 0.0:
                raise ValueError("Pallier2025 词窗口超出 canonical continuous signal。")
            start = max(0, min(start, int(recording.shape[1]) - self.window_samples))
            stop = start + self.window_samples
        if self.temporal_shift_samples != 0:
            valid_span = int(recording.shape[1]) - self.window_samples + 1
            if valid_span <= 2 * self.window_samples:
                raise ValueError(
                    "Pallier2025 连续信号长度不足以执行非重叠时间错位对照。"
                )
            shifted_start = (start + self.temporal_shift_samples) % valid_span
            if abs(shifted_start - start) < self.window_samples:
                step = (
                    self.window_samples
                    if self.temporal_shift_samples > 0
                    else -self.window_samples
                )
                shifted_start = (shifted_start + step) % valid_span
            start = int(shifted_start)
            stop = start + self.window_samples
        window = recording[:, start:stop].astype(np.float32, copy=True)
        effective_baseline = min(self._baseline_samples, int(window.shape[1]))
        if effective_baseline > 0:
            baseline = window[:, :effective_baseline].mean(
                axis=1, keepdims=True
            )
            window -= baseline
        np.clip(window, -self._clamp, self._clamp, out=window)
        return window

    def __getitem__(self, index):
        idx = int(index)
        word = self._words[idx]
        return {
            "meg": torch.from_numpy(self.read_meg(idx)),
            "text_embedding": self._embedding_tensors[word],
            "subject_index": self._subject_index_tensors[idx],
            "sentence_index": self._sentence_index_tensors[idx],
            "word": word,
            "event_id": self._event_ids[idx],
            "recording_id": self._recording_ids[idx],
        }


def channel_names_sha256(channel_names) -> str:
    """按审计冻结的换行连接规则计算 306 通道顺序摘要。"""
    payload = "\n".join(str(name) for name in channel_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def recording_onset_to_sample(
    onset_seconds,
    *,
    source_first_samp,
    source_sampling_rate_hz=SOURCE_SAMPLING_RATE_HZ,
    output_sampling_rate_hz=TARGET_SAMPLING_RATE_HZ,
) -> int:
    """将 BIDS 记录时钟 onset 转为 canonical 连续信号的零起点样本。"""
    recording_start = float(source_first_samp) / float(source_sampling_rate_hz)
    relative_seconds = float(onset_seconds) - recording_start
    return int(np.round(relative_seconds * float(output_sampling_rate_hz)))


def recording_window_bounds(
    onset_seconds,
    *,
    duration_seconds,
    source_first_samp,
    source_sampling_rate_hz=SOURCE_SAMPLING_RATE_HZ,
    output_sampling_rate_hz=TARGET_SAMPLING_RATE_HZ,
) -> tuple[int, int]:
    """返回左闭右开的 canonical 窗口样本范围。"""
    start = recording_onset_to_sample(
        onset_seconds,
        source_first_samp=source_first_samp,
        source_sampling_rate_hz=source_sampling_rate_hz,
        output_sampling_rate_hz=output_sampling_rate_hz,
    )
    length = int(np.round(float(duration_seconds) * output_sampling_rate_hz))
    return start, start + length


def preprocessing_contract(dataset_config: dict | None = None) -> dict:
    """返回 Pallier2025 冻结的连续 MEG 数值合同。"""
    config = dataset_config or {}
    try:
        import mne
        import scipy
        import sklearn
    except ImportError as error:  # pragma: no cover - 环境错误分支
        raise RuntimeError("Pallier2025 信号预处理需要 MNE、SciPy 和 scikit-learn。") from error
    return {
        "materialization_version": SIGNAL_MATERIALIZATION_VERSION,
        "reference": {
            "repository": "dAscoli-et-al-2023/dascoli-word-decoding",
            "commit": DASCOLI_REFERENCE_COMMIT,
            "method": "Meg._get_preprocessed_data",
        },
        "operation_order": [
            "mne_read_raw_fif_allow_maxshield",
            "mne_pick_meg",
            "mne_filter_0.1_40_hz",
            "mne_resample_50_hz",
            "sklearn_robust_scaler_per_channel_full_recording",
            "float32",
        ],
        "source_sampling_rate_hz": float(
            config.get("source_sampling_rate_hz", SOURCE_SAMPLING_RATE_HZ)
        ),
        "target_sampling_rate_hz": float(
            config.get("target_sampling_rate_hz", TARGET_SAMPLING_RATE_HZ)
        ),
        "filter": {
            "l_freq_hz": float(config.get("filter_low_hz", FILTER_LOW_HZ)),
            "h_freq_hz": float(config.get("filter_high_hz", FILTER_HIGH_HZ)),
            "implementation": "mne.io.Raw.filter",
            "method": "fir",
            "phase": "zero",
            "fir_window": "hamming",
            "fir_design": "firwin",
            "pad": "reflect_limited",
        },
        "resample": {
            "implementation": "mne.io.Raw.resample",
            "method": "fft",
            "npad": "auto",
            "window": "auto",
            "pad": "auto",
        },
        "scaler": {
            "implementation": "sklearn.preprocessing.RobustScaler",
            "fit_scope": "each_channel_over_full_recording",
            "transpose_contract": "fit_transform(data.T).T",
        },
        "channel_contract": {
            "selection": "meg",
            "channel_count": MEG_CHANNEL_COUNT,
            "magnetometer_count": MAGNETOMETER_COUNT,
            "gradiometer_count": GRADIOMETER_COUNT,
            "channel_names_sha256": CHANNEL_NAMES_SHA256,
            "bad_channels_removed": False,
        },
        "output_dtype": "float32",
        "notch_filter": False,
        "sss_or_maxfilter": False,
        "ica": False,
        "artifact_removal": False,
        "baseline_correction": False,
        "clamp": None,
        "software": {
            "mne": mne.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }


def _validate_picked_meg(raw, source_path) -> dict:
    import mne

    names = list(raw.ch_names)
    types = raw.get_channel_types()
    magnetometers = sum(value == "mag" for value in types)
    gradiometers = sum(value == "grad" for value in types)
    digest = channel_names_sha256(names)
    if len(names) != MEG_CHANNEL_COUNT:
        raise ValueError(f"Pallier2025 MEG 通道数不是 306：{source_path}")
    if (magnetometers, gradiometers) != (
        MAGNETOMETER_COUNT,
        GRADIOMETER_COUNT,
    ):
        raise ValueError(
            f"Pallier2025 MEG 通道类型漂移：mag={magnetometers}, grad={gradiometers}"
        )
    if digest != CHANNEL_NAMES_SHA256:
        raise ValueError(f"Pallier2025 MEG 通道名称或顺序漂移：{source_path}")
    return {
        "channel_names": names,
        "channel_names_sha256": digest,
        "channel_count": len(names),
        "magnetometer_count": magnetometers,
        "gradiometer_count": gradiometers,
        "bad_channels": [name for name in raw.info.get("bads", ()) if name in names],
        "channel_types": {
            "mag": magnetometers,
            "grad": gradiometers,
        },
        "mne_channel_type_sha256": stable_sha256(types),
        "mne_version": mne.__version__,
    }


def _open_picked_meg(source_path):
    try:
        import mne
    except ImportError as error:  # pragma: no cover - 环境错误分支
        raise RuntimeError("Pallier2025 信号预处理需要安装 MNE。") from error
    raw = mne.io.read_raw_fif(
        source_path,
        preload=False,
        allow_maxshield=True,
        verbose="ERROR",
    )
    source_rate = float(raw.info["sfreq"])
    if not np.isclose(source_rate, SOURCE_SAMPLING_RATE_HZ):
        raw.close()
        raise ValueError(f"Pallier2025 原始采样率不是 1000 Hz：{source_path}")
    raw.pick(("meg",), verbose=False)
    channel_metadata = _validate_picked_meg(raw, source_path)
    metadata = {
        "source_first_samp": int(raw.first_samp),
        "source_last_samp": int(raw.last_samp),
        "source_sample_count": int(raw.n_times),
        "source_sampling_rate_hz": source_rate,
        "source_recording_start_seconds": float(raw.first_samp / source_rate),
        **channel_metadata,
    }
    return raw, metadata


def _source_contract(source_path, source_metadata: dict, contract: dict) -> dict:
    source_path = Path(source_path)
    stat = source_path.stat()
    identity = {
        "source_path": str(source_path.resolve()),
        "source_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "source_first_samp": source_metadata["source_first_samp"],
        "source_last_samp": source_metadata["source_last_samp"],
        "source_sample_count": source_metadata["source_sample_count"],
        "source_sampling_rate_hz": source_metadata["source_sampling_rate_hz"],
        "channel_names_sha256": source_metadata["channel_names_sha256"],
        "preprocessing_contract_sha256": stable_sha256(contract),
    }
    return {
        **identity,
        "source_fingerprint_strategy": "stat_and_fif_header_sha256",
        "source_fingerprint_sha256": stable_sha256(identity),
    }


def _process_picked_meg(raw, dataset_config: dict, timings=None):
    """按冻结顺序处理已选择的 306 通道；不含任何窗口级操作。"""
    from sklearn.preprocessing import RobustScaler

    n_jobs = int(dataset_config.get("mne_n_jobs", -1))
    started = time.perf_counter()
    raw.load_data()
    if timings is not None:
        timings["load_data_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    raw.filter(
        float(dataset_config.get("filter_low_hz", FILTER_LOW_HZ)),
        float(dataset_config.get("filter_high_hz", FILTER_HIGH_HZ)),
        n_jobs=n_jobs,
        verbose=False,
    )
    if timings is not None:
        timings["mne_filter_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    raw = raw.resample(
        float(dataset_config.get("target_sampling_rate_hz", TARGET_SAMPLING_RATE_HZ)),
        n_jobs=n_jobs,
        verbose=False,
    )
    if timings is not None:
        timings["mne_resample_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    raw._data = RobustScaler().fit_transform(raw._data.T).T
    if timings is not None:
        timings["robust_scaler_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    result = raw._data.astype(np.float32)
    if timings is not None:
        timings["float32_seconds"] = time.perf_counter() - started
    return result, raw


def canonical_preprocess_recording(source_path, dataset_config: dict, timings=None):
    """执行正式 Pallier2025 连续信号预处理并返回数组与时间元数据。"""
    total_started = time.perf_counter()
    started = time.perf_counter()
    raw, source_metadata = _open_picked_meg(source_path)
    if timings is not None:
        timings["read_and_pick_seconds"] = time.perf_counter() - started
    try:
        data, processed = _process_picked_meg(raw, dataset_config, timings)
        output_metadata = {
            "output_first_samp": int(processed.first_samp),
            "output_sampling_rate_hz": float(processed.info["sfreq"]),
            "output_recording_start_seconds": float(
                processed.first_samp / processed.info["sfreq"]
            ),
            "output_sample_count": int(processed.n_times),
        }
    finally:
        raw.close()
    if timings is not None:
        timings["total_preprocessing_seconds"] = time.perf_counter() - total_started
    return data, {**source_metadata, **output_metadata}


def reference_preprocess_recording(source_path, dataset_config: dict, timings=None):
    """逐行复现 d'Ascoli reference；不调用 canonical 处理 helper。"""
    try:
        import mne
        from sklearn.preprocessing import RobustScaler
    except ImportError as error:  # pragma: no cover - 环境错误分支
        raise RuntimeError("Pallier2025 reference 预处理依赖不完整。") from error

    total_started = time.perf_counter()
    started = time.perf_counter()
    raw = mne.io.read_raw_fif(
        source_path,
        preload=False,
        allow_maxshield=True,
        verbose="ERROR",
    )
    source_rate = float(raw.info["sfreq"])
    if not np.isclose(source_rate, SOURCE_SAMPLING_RATE_HZ):
        raw.close()
        raise ValueError(f"Pallier2025 原始采样率不是 1000 Hz：{source_path}")
    raw = raw.pick(("meg",), verbose=False)
    channel_metadata = _validate_picked_meg(raw, source_path)
    source_metadata = {
        "source_first_samp": int(raw.first_samp),
        "source_last_samp": int(raw.last_samp),
        "source_sample_count": int(raw.n_times),
        "source_sampling_rate_hz": source_rate,
        "source_recording_start_seconds": float(raw.first_samp / source_rate),
        **channel_metadata,
    }
    if timings is not None:
        timings["read_and_pick_seconds"] = time.perf_counter() - started
    try:
        n_jobs = int(dataset_config.get("mne_n_jobs", -1))
        started = time.perf_counter()
        raw.load_data()
        if timings is not None:
            timings["load_data_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        raw.filter(
            float(dataset_config.get("filter_low_hz", FILTER_LOW_HZ)),
            float(dataset_config.get("filter_high_hz", FILTER_HIGH_HZ)),
            n_jobs=n_jobs,
            verbose=False,
        )
        if timings is not None:
            timings["mne_filter_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        raw = raw.resample(
            float(dataset_config.get("target_sampling_rate_hz", TARGET_SAMPLING_RATE_HZ)),
            n_jobs=n_jobs,
            verbose=False,
        )
        if timings is not None:
            timings["mne_resample_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        raw._data = RobustScaler().fit_transform(raw._data.T).T
        if timings is not None:
            timings["robust_scaler_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        data = raw._data.astype(np.float32)
        if timings is not None:
            timings["float32_seconds"] = time.perf_counter() - started
        output_metadata = {
            "output_first_samp": int(raw.first_samp),
            "output_sampling_rate_hz": float(raw.info["sfreq"]),
            "output_recording_start_seconds": float(raw.first_samp / raw.info["sfreq"]),
            "output_sample_count": int(raw.n_times),
        }
    finally:
        raw.close()
    if timings is not None:
        timings["total_preprocessing_seconds"] = time.perf_counter() - total_started
    return data, {**source_metadata, **output_metadata}


def compare_preprocessed_recordings(reference, canonical) -> dict:
    """逐值比较 reference 与 canonical 连续信号。"""
    left = np.asarray(reference)
    right = np.asarray(canonical)
    same_shape = left.shape == right.shape
    if same_shape:
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        different = int(np.count_nonzero(left != right))
        max_error = float(difference.max(initial=0.0))
        mean_error = float(difference.mean()) if difference.size else 0.0
    else:
        different = None
        max_error = None
        mean_error = None
    return {
        "shape_equal": same_shape,
        "reference_shape": list(left.shape),
        "canonical_shape": list(right.shape),
        "reference_dtype": str(left.dtype),
        "canonical_dtype": str(right.dtype),
        "dtype_equal": left.dtype == right.dtype,
        "bitwise_equal": bool(same_shape and np.array_equal(left, right)),
        "different_element_count": different,
        "max_abs_error": max_error,
        "mean_abs_error": mean_error,
        "reference_statistics": {
            "min": float(left.min()),
            "max": float(left.max()),
            "mean": float(left.mean()),
            "std": float(left.std()),
        },
        "canonical_statistics": {
            "min": float(right.min()),
            "max": float(right.max()),
            "mean": float(right.mean()),
            "std": float(right.std()),
        },
        "reference_array_sha256": hashlib.sha256(left.tobytes(order="C")).hexdigest(),
        "canonical_array_sha256": hashlib.sha256(right.tobytes(order="C")).hexdigest(),
    }


def run_reference_gate(dataset_config: dict, provenance_dir) -> dict:
    """用 sub-01/run-01 执行独立 reference/canonical bitwise 闸门。"""
    source_path = recording_paths(
        dataset_config["root"], "sub-01", "run-01"
    )["meg"]
    path = Path(provenance_dir) / "preprocessing_reference.json"
    contract = preprocessing_contract(dataset_config)
    if path.exists():
        existing = _read_json(path)
        body = dict(existing)
        recorded_hash = body.pop("manifest_sha256", None)
        source = existing.get("source_contract", {})
        stat = source_path.stat()
        if (
            stable_sha256(body) == recorded_hash
            and existing.get("preprocessing_contract_sha256")
            == stable_sha256(contract)
            and source.get("source_size_bytes") == int(stat.st_size)
            and source.get("source_mtime_ns") == int(stat.st_mtime_ns)
            and existing.get("comparison", {}).get("bitwise_equal") is True
            and existing.get("metadata_equal") is True
        ):
            return existing
    reference_timings = {}
    canonical_timings = {}
    reference, reference_metadata = reference_preprocess_recording(
        source_path, dataset_config, reference_timings
    )
    canonical, canonical_metadata = canonical_preprocess_recording(
        source_path, dataset_config, canonical_timings
    )
    comparison = compare_preprocessed_recordings(reference, canonical)
    metadata_fields = (
        "source_first_samp",
        "source_sampling_rate_hz",
        "source_recording_start_seconds",
        "output_first_samp",
        "output_sampling_rate_hz",
        "output_recording_start_seconds",
        "output_sample_count",
        "channel_names_sha256",
    )
    metadata_equal = all(
        reference_metadata[field] == canonical_metadata[field]
        for field in metadata_fields
    )
    if not comparison["bitwise_equal"] or not metadata_equal:
        raise RuntimeError(
            "Pallier2025 reference 与 canonical 预处理未逐值等价；禁止开始全量物化。"
        )
    source_contract = _source_contract(source_path, reference_metadata, contract)
    payload = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "recording_id": f"{DATASET_ID}|sub-01|run-01",
        "source_path": str(source_path),
        "reference_commit": DASCOLI_REFERENCE_COMMIT,
        "preprocessing_contract": contract,
        "preprocessing_contract_sha256": stable_sha256(contract),
        "source_contract": source_contract,
        "channel_names_sha256": reference_metadata["channel_names_sha256"],
        "reference_metadata": {
            field: reference_metadata[field] for field in metadata_fields
        },
        "canonical_metadata": {
            field: canonical_metadata[field] for field in metadata_fields
        },
        "metadata_equal": metadata_equal,
        "comparison": comparison,
        "timings": {
            "reference": reference_timings,
            "canonical": canonical_timings,
        },
        "model_or_checkpoint_loaded": False,
        "model_evaluation_performed": False,
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    _write_json_atomic(path, payload)
    return payload


def build_time_origin_audit(dataset_config: dict, provenance_dir) -> dict:
    """用 first_samp 小/中/大的三条 train recording 核验时间转换。"""
    import mne

    root = Path(dataset_config["root"])
    split_by_run = load_run_split(dataset_config["split_manifest"])
    candidates = []
    for subject in dataset_config.get("subjects", SUBJECTS):
        for run in dataset_config.get("runs", RUNS):
            if split_by_run[run] != "train":
                continue
            paths = recording_paths(root, subject, run)
            header = read_recording_header(paths["meg"])
            candidates.append((header["first_sample"], subject, run, paths))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    selected = [candidates[0], candidates[len(candidates) // 2], candidates[-1]]
    examples = []
    for _, subject, run, paths in selected:
        raw = mne.io.read_raw_fif(
            paths["meg"],
            preload=False,
            allow_maxshield=True,
            verbose="ERROR",
        )
        try:
            source_rate = float(raw.info["sfreq"])
            source_start = float(raw.first_samp / source_rate)
            events = pd.read_csv(paths["events"], sep="\t")
            indices = sorted({0, len(events) // 2, len(events) - 1})
            onset_examples = []
            for row_index in indices:
                onset = float(events.iloc[row_index]["onset"])
                relative = onset - source_start
                official_source_index = int(
                    raw.time_as_index(relative, use_rounding=True)[0]
                )
                computed_source_index = int(np.round(relative * source_rate))
                output_index = recording_onset_to_sample(
                    onset,
                    source_first_samp=raw.first_samp,
                    source_sampling_rate_hz=source_rate,
                    output_sampling_rate_hz=TARGET_SAMPLING_RATE_HZ,
                )
                if official_source_index != computed_source_index:
                    raise ValueError(
                        f"MNE 时间转换不一致：{subject}/{run}/row-{row_index}"
                    )
                onset_examples.append(
                    {
                        "bids_event_row": row_index,
                        "bids_onset_seconds": onset,
                        "recording_relative_seconds": relative,
                        "mne_source_sample_index": official_source_index,
                        "canonical_output_sample_index": output_index,
                    }
                )
            examples.append(
                {
                    "subject": subject,
                    "run": run,
                    "recording_id": f"{DATASET_ID}|{subject}|{run}",
                    "source_first_samp": int(raw.first_samp),
                    "source_sampling_rate_hz": source_rate,
                    "source_recording_start_seconds": source_start,
                    "examples": onset_examples,
                }
            )
        finally:
            raw.close()
    qc = load_qc_artifact(dataset_config["qc_artifact"], root)
    qc_header = qc["evidence"]["raw_header"]
    negative_examples = []
    for item in qc["evidence"]["anomalous_words"]:
        onset = float(item["expected_recording_onset_seconds"])
        output_index = recording_onset_to_sample(
            onset,
            source_first_samp=int(qc_header["first_sample"]),
            source_sampling_rate_hz=float(qc_header["sampling_rate_hz"]),
        )
        if output_index >= 0:
            raise ValueError("sub-09/run-03 QC 异常词不再早于 recording 起点。")
        negative_examples.append(
            {
                "word": item["word"],
                "inferred_onset_seconds": onset,
                "canonical_output_sample_index": output_index,
            }
        )
    payload = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "conversion": (
            "round((bids_onset_seconds - source_first_samp / "
            "source_sampling_rate_hz) * output_sampling_rate_hz)"
        ),
        "mne_time_as_index_agrees": True,
        "train_recording_examples": examples,
        "sub-09_run-03_negative_boundary": {
            "recording_excluded_from_training": True,
            "source_first_samp": int(qc_header["first_sample"]),
            "source_recording_start_seconds": float(qc_header["first_time_seconds"]),
            "anomalous_words": negative_examples,
        },
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    path = Path(provenance_dir) / "time_origin_audit.json"
    _write_json_atomic(path, payload)
    return payload


def processed_recording_path(source_path, cache_dir) -> Path:
    """返回 canonical Pallier2025 recording 信号路径。"""
    source_path = Path(source_path)
    subject = next(
        (part for part in source_path.parts if part.startswith("sub-")), None
    )
    if subject is None:
        raise ValueError(f"无法从 Pallier2025 源路径识别受试者：{source_path}")
    return Path(cache_dir) / subject / f"{source_path.stem}.npy"


def _sidecar_path(output_path) -> Path:
    return Path(output_path).with_suffix(".json")


def _valid_signal_product(output_path, expected_source_contract: dict) -> bool:
    output_path = Path(output_path)
    sidecar_path = _sidecar_path(output_path)
    if not output_path.is_file() or not sidecar_path.is_file():
        return False
    try:
        sidecar = _read_json(sidecar_path)
        if sidecar.get("source_contract") != expected_source_contract:
            return False
        array = np.load(output_path, mmap_mode="r", allow_pickle=False)
        expected_samples = int(
            np.round(
                expected_source_contract["source_sample_count"]
                * TARGET_SAMPLING_RATE_HZ
                / expected_source_contract["source_sampling_rate_hz"]
            )
        )
        return (
            list(array.shape) == sidecar.get("shape")
            and list(array.shape) == [MEG_CHANNEL_COUNT, expected_samples]
            and array.dtype == np.float32
            and sidecar.get("dtype") == "float32"
            and sha256_file(output_path) == sidecar.get("output_sha256")
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _write_array_atomic(path, array) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def materialize_recording_signal(
    source_path,
    dataset_config: dict,
    cache_dir,
    *,
    subject: str,
    run: str,
    force=False,
    timings=None,
) -> Path:
    """原子物化一条 recording；仅在完整来源与数值合同匹配时复用。"""
    total_started = time.perf_counter()
    source_path = Path(source_path)
    contract = preprocessing_contract(dataset_config)
    started = time.perf_counter()
    raw, source_metadata = _open_picked_meg(source_path)
    if timings is not None:
        timings["read_and_pick_seconds"] = time.perf_counter() - started
    expected_source = _source_contract(source_path, source_metadata, contract)
    output_path = processed_recording_path(source_path, cache_dir)
    if not force and _valid_signal_product(output_path, expected_source):
        raw.close()
        if timings is not None:
            timings.update(
                status="reused",
                total_seconds=time.perf_counter() - total_started,
            )
        return output_path
    if (
        not force
        and output_path.exists()
        and _sidecar_path(output_path).exists()
    ):
        raw.close()
        raise ValueError(f"已有 Pallier2025 信号产品与冻结合同不一致：{output_path}")
    try:
        data, processed = _process_picked_meg(raw, dataset_config, timings)
        output_metadata = {
            "output_first_samp": int(processed.first_samp),
            "output_sampling_rate_hz": float(processed.info["sfreq"]),
            "output_recording_start_seconds": float(
                processed.first_samp / processed.info["sfreq"]
            ),
            "output_sample_count": int(processed.n_times),
        }
    finally:
        raw.close()
    if not np.isfinite(data).all():
        raise ValueError(f"Pallier2025 预处理结果存在非有限数值：{source_path}")
    started = time.perf_counter()
    _write_array_atomic(output_path, data)
    output_sha = sha256_file(output_path)
    if timings is not None:
        timings["npy_write_and_sha_seconds"] = time.perf_counter() - started
    recording_id = f"{DATASET_ID}|{subject}|{run}"
    sidecar = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "recording_id": recording_id,
        "subject": subject,
        "run": run,
        "recording_signal_status": "valid",
        "event_training_status": (
            "excluded_by_annotation_qc"
            if (subject, run) == ("sub-09", "run-03")
            else "eligible_by_signal_contract"
        ),
        "source_contract": expected_source,
        "source_first_samp": source_metadata["source_first_samp"],
        "source_sampling_rate_hz": source_metadata["source_sampling_rate_hz"],
        "source_recording_start_seconds": source_metadata[
            "source_recording_start_seconds"
        ],
        **output_metadata,
        "channel_count": source_metadata["channel_count"],
        "channel_names": source_metadata["channel_names"],
        "channel_names_sha256": source_metadata["channel_names_sha256"],
        "magnetometer_count": source_metadata["magnetometer_count"],
        "gradiometer_count": source_metadata["gradiometer_count"],
        "bad_channels": source_metadata["bad_channels"],
        "shape": list(data.shape),
        "dtype": "float32",
        "output_sha256": output_sha,
        "preprocessing_contract_sha256": stable_sha256(contract),
    }
    started = time.perf_counter()
    _write_json_atomic(_sidecar_path(output_path), sidecar)
    if timings is not None:
        timings["sidecar_write_seconds"] = time.perf_counter() - started
        timings.update(
            status="materialized",
            total_seconds=time.perf_counter() - total_started,
        )
    if not _valid_signal_product(output_path, expected_source):
        raise RuntimeError(f"Pallier2025 信号写入后验证失败：{output_path}")
    return output_path


def validate_signal_manifest(path, *, event_table=None) -> dict:
    """只读验证 90 条 Pallier2025 signal products 与总 manifest。"""
    path = Path(path)
    payload = _read_json(path)
    body = dict(payload)
    recorded_hash = body.pop("manifest_sha256", None)
    if stable_sha256(body) != recorded_hash:
        raise ValueError(f"Pallier2025 signals manifest 自摘要不一致：{path}")
    if payload.get("status") != "complete" or payload.get("recording_count") != 90:
        raise ValueError("Pallier2025 signals 只有 90/90 时才能标记 complete。")
    if payload.get("channel_names_sha256") != CHANNEL_NAMES_SHA256:
        raise ValueError("Pallier2025 signals manifest 通道合同漂移。")
    dataset_dir = path.parents[1]

    def _resolve_rec_path(rel_str: str) -> Path:
        norm = str(rel_str).replace("\\", "/")
        for prefix in ("data/cache/derived/pallier2025/", "derived/pallier2025/"):
            if norm.startswith(prefix):
                cand = dataset_dir / norm[len(prefix):]
                if cand.exists():
                    return cand
        for parent in path.parents:
            cand = parent / norm
            if cand.exists():
                return cand
        return dataset_dir / norm

    seen = set()
    for record in payload.get("recordings", ()):
        recording_id = str(record["recording_id"])
        if recording_id in seen:
            raise ValueError(f"Pallier2025 signals manifest recording 重复：{recording_id}")
        seen.add(recording_id)
        output = _resolve_rec_path(record["output_path"])
        sidecar = _resolve_rec_path(record["sidecar_path"])
        if sha256_file(output) != record["output_sha256"]:
            raise ValueError(f"Pallier2025 signal SHA 漂移：{output}")
        if sha256_file(sidecar) != record["sidecar_sha256"]:
            raise ValueError(f"Pallier2025 sidecar SHA 漂移：{sidecar}")
    if len(seen) != 90:
        raise ValueError(f"Pallier2025 signals manifest 仅含 {len(seen)}/90 条记录。")
    if event_table is not None:
        expected = set(event_table["记录编号"].astype(str).unique())
        if seen != expected:
            raise ValueError(
                f"Pallier2025 event→signal 覆盖漂移：missing={len(expected-seen)}, "
                f"extra={len(seen-expected)}"
            )
    return payload


def _validate_signal_coverage(event_table, recording_sidecars: dict[str, dict]) -> dict:
    expected = set(event_table["记录编号"].astype(str).unique())
    actual = set(recording_sidecars)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"Pallier2025 signal 覆盖不完整：missing={len(missing)}, extra={len(extra)}"
        )
    for recording_id, rows in event_table[event_table["是否可训练"]].groupby(
        "记录编号", sort=False
    ):
        sidecar = recording_sidecars[str(recording_id)]
        for onset in rows["开始时间"].astype(float):
            start, stop = recording_window_bounds(
                onset,
                duration_seconds=ELIGIBILITY_WINDOW_SECONDS,
                source_first_samp=sidecar["source_first_samp"],
                source_sampling_rate_hz=sidecar["source_sampling_rate_hz"],
                output_sampling_rate_hz=sidecar["output_sampling_rate_hz"],
            )
            if start < 0 or stop > int(sidecar["output_sample_count"]):
                raise ValueError(f"可训练事件窗口超出 canonical signal：{recording_id}")
    return {
        "expected_recordings": len(expected),
        "signal_products": len(actual),
        "missing": 0,
        "extra": 0,
        "channel_count": MEG_CHANNEL_COUNT,
        "trainable_windows_in_range": True,
    }


def build_signal_manifest(
    event_table,
    dataset_config: dict,
    cache_dir,
    output_paths,
    *,
    project_root,
) -> Path:
    """验证 90/90 后生成唯一 complete signals manifest。"""
    project_root = Path(project_root)
    output_paths = [Path(path) for path in output_paths]
    actual_paths = set(Path(cache_dir).rglob("*.npy"))
    declared_paths = set(output_paths)
    if actual_paths != declared_paths:
        raise ValueError(
            "Pallier2025 signal 文件集合不等于事件表声明集合："
            f"missing={len(declared_paths-actual_paths)}, "
            f"extra={len(actual_paths-declared_paths)}"
        )
    sidecars = {}
    records = []
    for output in output_paths:
        sidecar_path = _sidecar_path(output)
        sidecar = _read_json(sidecar_path)
        recording_id = str(sidecar["recording_id"])
        sidecars[recording_id] = sidecar
        records.append(
            {
                "recording_id": recording_id,
                "subject": sidecar["subject"],
                "run": sidecar["run"],
                "recording_signal_status": sidecar["recording_signal_status"],
                "event_training_status": sidecar["event_training_status"],
                "source_path": sidecar["source_contract"]["source_path"],
                "source_size_bytes": sidecar["source_contract"]["source_size_bytes"],
                "source_fingerprint_sha256": sidecar["source_contract"][
                    "source_fingerprint_sha256"
                ],
                "output_path": output.relative_to(project_root).as_posix(),
                "shape": sidecar["shape"],
                "output_sha256": sidecar["output_sha256"],
                "sidecar_path": sidecar_path.relative_to(project_root).as_posix(),
                "sidecar_sha256": sha256_file(sidecar_path),
            }
        )
    coverage = _validate_signal_coverage(event_table, sidecars)
    if len(records) != 90 or coverage["signal_products"] != 90:
        raise ValueError("Pallier2025 signals 未达到 90/90，不生成 complete manifest。")
    contract = preprocessing_contract(dataset_config)
    payload = {
        "schema_version": 1,
        "dataset": DATASET_ID,
        "status": "complete",
        "subjects": list(dataset_config.get("subjects", SUBJECTS)),
        "recording_count": 90,
        "channel_count": MEG_CHANNEL_COUNT,
        "magnetometer_count": MAGNETOMETER_COUNT,
        "gradiometer_count": GRADIOMETER_COUNT,
        "channel_names_sha256": CHANNEL_NAMES_SHA256,
        "source_sampling_rate_hz": SOURCE_SAMPLING_RATE_HZ,
        "target_sampling_rate_hz": TARGET_SAMPLING_RATE_HZ,
        "dtype": "float32",
        "preprocessing_contract": contract,
        "preprocessing_contract_sha256": stable_sha256(contract),
        "event_table_sha256": sha256_file(dataset_config["event_table"]),
        "coverage": coverage,
        "test_neural_data_status": "raw_accessed_for_deterministic_preprocessing_only",
        "test_model_evaluation": "not_run",
        "test_predictions_generated": False,
        "test_metrics_inspected": False,
        "recordings": sorted(records, key=lambda item: item["recording_id"]),
    }
    payload["manifest_sha256"] = stable_sha256(payload)
    path = Path(cache_dir).parent / "manifest.json"
    _write_json_atomic(path, payload)
    validate_signal_manifest(path, event_table=event_table)
    return path


def ensure_recording_signals(
    event_table,
    dataset_config: dict,
    cache_dir,
    *,
    project_root,
    force=False,
    timing_records=None,
) -> Path:
    """按受试者/run 稳定顺序物化 90 条连续 MEG 并生成总 manifest。"""
    root = Path(dataset_config["root"])
    recordings = event_table[["受试者", "运行编号", "记录编号"]].drop_duplicates()
    outputs = []
    for index, row in enumerate(recordings.itertuples(index=False), start=1):
        subject = str(row[0])
        run = str(row[1])
        source_path = recording_paths(root, subject, run)["meg"]
        timings = {}
        print(f"Pallier2025 MEG {index}/90: {subject}/{run}", flush=True)
        output = materialize_recording_signal(
            source_path,
            dataset_config,
            cache_dir,
            subject=subject,
            run=run,
            force=force,
            timings=timings,
        )
        outputs.append(output)
        if timing_records is not None:
            timing_records.append(
                {
                    "recording_id": str(row[2]),
                    "source": str(source_path),
                    **timings,
                }
            )
    return build_signal_manifest(
        event_table,
        {**dataset_config, "event_table": str(Path(project_root) / "data/cache/derived/pallier2025/events/events.csv")},
        cache_dir,
        outputs,
        project_root=project_root,
    )
