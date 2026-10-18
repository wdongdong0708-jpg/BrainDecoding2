"""从已冻结事件表生成论文协议资产，不读取脑信号。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from braindecoding.config import PROJECT_ROOT


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "manifests"
DEFAULT_CHINESE_EVENT_TABLE = (
    PROJECT_ROOT
    / "derived"
    / "chineseeeg2_littleprince"
    / "events"
    / "events.csv"
)
PRIMARY_VOCABULARY_SIZES = (20, 50, 100, 150)
CHINESE_SUBJECTS = tuple(f"sub-{index:02d}" for index in range(1, 9))


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    """使用稳定键序和 UTF-8 生成可重复写入的 JSON。"""
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def manifest_sha256(manifest: dict) -> str:
    """计算去掉自指字段后的规范 JSON 摘要。"""
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def with_manifest_sha256(manifest: dict) -> dict:
    """返回带稳定自校验摘要的 manifest 副本。"""
    result = dict(manifest)
    result["manifest_sha256"] = manifest_sha256(result)
    return result


def validate_manifest_sha256(manifest: dict) -> None:
    """验证 manifest 自校验摘要。"""
    expected = str(manifest.get("manifest_sha256", ""))
    actual = manifest_sha256(manifest)
    if not expected or expected != actual:
        raise ValueError(
            f"manifest SHA-256 不一致：期望 {expected or '<missing>'}，实际 {actual}"
        )


def _clean_word(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _bool_series(values: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise ValueError(f"无法解析是否可训练字段：{unknown[:5]}")
    return normalized.map(mapping).astype(bool)


def _require_columns(table: pd.DataFrame, columns) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"事件表缺少协议资产所需字段：{missing}")


def validate_split_units(table: pd.DataFrame) -> None:
    """验证一个材料划分单元不会跨越 train/val/test。"""
    _require_columns(table, ("_split_unit", "split"))
    allowed = {"train", "val", "test"}
    observed = set(table["split"].astype(str))
    if not observed.issubset(allowed):
        raise ValueError(f"事件表含未知数据划分：{sorted(observed - allowed)}")
    split_counts = table.groupby("_split_unit", sort=False)["split"].nunique()
    crossed = sorted(split_counts[split_counts.gt(1)].index.astype(str))
    if crossed:
        raise ValueError(f"划分单元跨 split：{crossed[:10]}")


def canonical_words(
    table: pd.DataFrame,
    *,
    deduplicate_by,
    order_by,
    split: str | None = None,
    trainable_only: bool = False,
) -> pd.DataFrame:
    """按数据集合同去除受试者重复，保留一份材料词序列。"""
    required = {
        "normalized_word",
        *deduplicate_by,
        *order_by,
    }
    if split is not None:
        required.add("split")
    if trainable_only:
        required.add("is_trainable")
    _require_columns(table, required)

    selected = table
    if split is not None:
        selected = selected[selected["split"].eq(split)]
    if trainable_only:
        selected = selected[_bool_series(selected["is_trainable"])]
    selected = selected.copy()
    selected["_protocol_word"] = selected["normalized_word"].map(_clean_word)

    conflicts = selected.groupby(list(deduplicate_by), dropna=False)[
        "_protocol_word"
    ].nunique(dropna=False)
    conflicts = conflicts[conflicts.gt(1)]
    if not conflicts.empty:
        raise ValueError(
            f"同一材料词位置对应多个标准词：{list(conflicts.index[:5])}"
        )

    return (
        selected.sort_values(list(order_by), kind="stable")
        .drop_duplicates(list(deduplicate_by), keep="first")
        .reset_index(drop=True)
    )


def ranked_word_counts(words) -> tuple[list[str], dict[str, int]]:
    """按频次降序、词典序升序返回全部非空标准词。"""
    counts = Counter(_clean_word(word) for word in words)
    counts.pop("", None)
    ranked = sorted(counts, key=lambda word: (-int(counts[word]), word))
    return ranked, {word: int(counts[word]) for word in ranked}


def build_vocabulary_manifest(
    *,
    dataset: str,
    size: int,
    canonical_train_words: pd.DataFrame,
    source_units,
    event_table_path: Path,
    event_table_sha256: str,
    generator_sha256: str,
    status: str,
    normalization: str,
    scope: str,
) -> dict:
    """只从显式传入的训练材料生成候选词表 manifest。"""
    if not canonical_train_words["split"].eq("train").all():
        raise ValueError("候选词表输入包含非 train 事件。")
    ranked, all_counts = ranked_word_counts(canonical_train_words["_protocol_word"])
    vocabulary = ranked[: int(size)]
    if len(vocabulary) != int(size):
        raise ValueError(
            f"{dataset} 训练材料只有 {len(vocabulary)} 个非空标准词，无法生成 N={size}。"
        )
    manifest = {
        "asset_type": "candidate_vocabulary",
        "created_from_test": False,
        "dataset": dataset,
        "evaluation_support_used_for_selection": False,
        "event_table": _project_relative(event_table_path),
        "event_table_sha256": event_table_sha256,
        "generator": "braindecoding.protocols",
        "generator_sha256": generator_sha256,
        "hash_contract": "sha256(canonical_json_without_manifest_sha256)",
        "normalization": normalization,
        "policy": "frequency_desc_lexical_tiebreak",
        "scope": scope,
        "source_split": "train",
        "source_units": list(source_units),
        "status": status,
        "vocabulary": vocabulary,
        "vocabulary_size": int(size),
        "word_counts": {word: all_counts[word] for word in vocabulary},
    }
    return with_manifest_sha256(manifest)


def build_vocabulary_support_manifest(
    *,
    dataset: str,
    table: pd.DataFrame,
    vocabularies: dict[int, dict],
    generator_sha256: str,
    status: str,
    test_status: str,
) -> dict:
    """独立记录评价词类支持；结果不得用于改变冻结候选词。"""
    trainable = table[_bool_series(table["is_trainable"])].copy()
    support = {}
    for size, vocabulary_manifest in vocabularies.items():
        vocabulary = list(vocabulary_manifest["vocabulary"])
        split_support = {}
        for split in ("val", "test"):
            observed = {
                _clean_word(value)
                for value in trainable.loc[
                    trainable["split"].eq(split), "normalized_word"
                ]
            }
            observed.discard("")
            missing = [word for word in vocabulary if word not in observed]
            split_support[split] = {
                "candidate_count": len(vocabulary),
                "missing_candidate_count": len(missing),
                "missing_words": missing,
                "support_fraction": (len(vocabulary) - len(missing)) / len(vocabulary),
                "supported_candidate_count": len(vocabulary) - len(missing),
            }
        support[f"N{size}"] = {
            "candidate_manifest_sha256": vocabulary_manifest["manifest_sha256"],
            "splits": split_support,
        }
    return with_manifest_sha256(
        {
            "asset_type": "evaluation_vocabulary_support",
            "dataset": dataset,
            "generator": "braindecoding.protocols",
            "generator_sha256": generator_sha256,
            "status": status,
            "support_audit_used_for_vocabulary_selection": False,
            "test_status": test_status,
            "vocabularies": support,
        }
    )


def _project_relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _split_unit_lists(table: pd.DataFrame, *, trainable_only=False) -> dict:
    selected = table
    if trainable_only:
        selected = selected[_bool_series(selected["is_trainable"])]
    return {
        split: sorted(
            selected.loc[selected["split"].eq(split), "_split_unit"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        for split in ("train", "val", "test")
    }


def _prepare_chinese(table: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        table,
        (
            "chapter",
            "voice_version",
            "source_word_id",
            "chapter_word_index",
            "subject_id",
            "split",
            "is_trainable",
            "normalized_word",
        ),
    )
    result = table.copy()
    result["_split_unit"] = result["chapter"].map(
        lambda chapter: f"ChineseEEG2|littleprince|chapter-{int(chapter):02d}"
    )
    result["_material_unit"] = [
        f"ChineseEEG2|littleprince|chapter-{int(chapter):02d}|voice-{voice}"
        for chapter, voice in zip(result["chapter"], result["voice_version"])
    ]
    validate_split_units(result)
    return result


def _chinese_documents(
    event_table_path: Path, generator_sha256: str
) -> dict[str, dict]:
    table = _prepare_chinese(
        chineseeeg2.载入事件表(event_table_path, trainable_only=False)
    )
    subjects = tuple(sorted(table["subject_id"].astype(str).unique()))
    if subjects != CHINESE_SUBJECTS:
        raise ValueError(
            f"ChineseEEG2 正式受试者顺序不一致：期望 {CHINESE_SUBJECTS}，实际 {subjects}"
        )
    event_digest = file_sha256(event_table_path)
    canonical_train = canonical_words(
        table,
        deduplicate_by=("voice_version", "source_word_id"),
        order_by=(
            "voice_version",
            "chapter",
            "chapter_word_index",
            "subject_id",
        ),
        split="train",
        trainable_only=True,
    )
    source_units = sorted(canonical_train["_split_unit"].drop_duplicates())
    vocabularies = {
        size: build_vocabulary_manifest(
            dataset="ChineseEEG2",
            size=size,
            canonical_train_words=canonical_train,
            source_units=source_units,
            event_table_path=event_table_path,
            event_table_sha256=event_digest,
            generator_sha256=generator_sha256,
            status="frozen",
            normalization="event_table.normalized_word",
            scope="formal_eight_subject_passive_listening",
        )
        for size in PRIMARY_VOCABULARY_SIZES
    }
    unit_counts = table.groupby("_split_unit").agg(
        row_count=("_split_unit", "size"),
        trainable_event_count=("is_trainable", lambda values: int(_bool_series(values).sum())),
    )
    exclusions = {}
    for chapter in (14, 27):
        unit = f"ChineseEEG2|littleprince|chapter-{chapter:02d}"
        exclusions[unit] = {
            "configured_split": str(table.loc[table["_split_unit"].eq(unit), "split"].iloc[0]),
            "reason": "sub-05_recording_truncated_shared_eight_subject_exclusion",
            "row_count": int(unit_counts.loc[unit, "row_count"]),
            "trainable_event_count": int(unit_counts.loc[unit, "trainable_event_count"]),
        }
    split_manifest = with_manifest_sha256(
        {
            "asset_type": "split_and_subject_manifest",
            "confirmatory_status": "exploratory_or_replication_only",
            "context_contract": {
                "neural_context": "bounded_semantic_v1",
                "row": "historical_structure_prior_control",
                "word": {"model.use_transformer": False},
            },
            "dataset": "ChineseEEG2",
            "event_table": _project_relative(event_table_path),
            "event_table_sha256": event_digest,
            "generator": "braindecoding.protocols",
            "generator_sha256": generator_sha256,
            "shared_excluded_units": exclusions,
            "split_units": _split_unit_lists(table),
            "status": "frozen",
            "subject_order": list(subjects),
            "subjects": list(subjects),
            "test_status": "previously_used_for_exploratory_analysis",
            "trainable_split_units": _split_unit_lists(table, trainable_only=True),
        }
    )
    support = build_vocabulary_support_manifest(
        dataset="ChineseEEG2",
        table=table,
        vocabularies=vocabularies,
        generator_sha256=generator_sha256,
        status="frozen_support_audit",
        test_status="previously_used_for_exploratory_analysis",
    )
    documents = {
        "chineseeeg2/split_manifest.json": split_manifest,
        "chineseeeg2/vocabulary_support.json": support,
    }
    for size, manifest in vocabularies.items():
        documents[f"chineseeeg2/vocabulary_N{size}.json"] = manifest
    return documents


def _conditions_document(generator_sha256: str) -> dict:
    donor_protocol = {
        "common_support_query_set": True,
        "cross_recording_primary": False,
        "donor_word_must_differ_from_target": True,
        "donor_window_must_not_overlap_target": True,
        "same_recording": True,
        "same_subject": True,
        "same_window_contract": True,
        "seeds": list(range(20)),
        "target_and_context_position_unchanged": True,
    }
    temporal_protocol = {
        "common_support_queries": True,
        "offset_selection": "train_val_only",
        "parameter_status": "parameters_not_frozen",
        "same_recording": True,
        "same_subject": True,
        "shift": "deterministic_circular",
        "source_window_must_not_overlap_target": True,
        "target_unchanged": True,
    }
    return with_manifest_sha256(
        {
            "asset_type": "experiment_conditions",
            "candidate_vocabulary_sizes": {
                "primary": list(PRIMARY_VOCABULARY_SIZES),
                "supplementary_exploratory": [200, 300, 500],
                "test_support_may_change_sizes": False,
            },
            "conditions": {
                "neural_context.clean": {"status": "implemented"},
                "neural_context.donor_following": {"status": "exploratory"},
                "neural_context.donor_swap": {
                    "protocol": donor_protocol,
                    "status": "exploratory",
                },
                "neural_context.structure_only": {"status": "exploratory"},
                "neural_context.temporal_shift": {
                    "protocol": temporal_protocol,
                    "status": "missing",
                },
                "word.clean": {"status": "implemented"},
                "word.donor_swap": {
                    "protocol": donor_protocol,
                    "status": "missing",
                },
                "word.temporal_shift": {
                    "protocol": temporal_protocol,
                    "status": "missing",
                },
            },
            "datasets": {
                "ChineseEEG2": {
                    "confirmatory_status": "exploratory_or_replication_only",
                    "neural_context": "bounded_semantic_v1",
                    "row_context_role": "historical_structure_prior_control",
                    "test_status": "previously_used_for_exploratory_analysis",
                    "word": {"model.use_transformer": False},
                },
            },
            "generator": "braindecoding.protocols",
            "generator_sha256": generator_sha256,
            "protocol_status": (
                "machine_checkable_assets_frozen_without_test_model_evaluation"
            ),
        }
    )


def build_documents(
    chinese_event_table: Path = DEFAULT_CHINESE_EVENT_TABLE,
) -> dict[str, dict]:
    """构造全部协议文档；这里只读取事件/文本表，不读取 EEG/MEG。"""
    generator_digest = file_sha256(Path(__file__))
    documents = {}
    documents.update(_chinese_documents(Path(chinese_event_table), generator_digest))
    documents["conditions.json"] = _conditions_document(generator_digest)
    return documents


def write_or_check_documents(
    documents: dict[str, dict], output_root: Path, *, check: bool
) -> list[Path]:
    """稳定写入资产，或逐字节核对现有资产。"""
    output_root = Path(output_root)
    paths = []
    mismatches = []
    for relative_path, document in sorted(documents.items()):
        path = output_root / relative_path
        expected = json_bytes(document)
        paths.append(path)
        if check:
            if not path.exists() or path.read_bytes() != expected:
                mismatches.append(relative_path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    if mismatches:
        raise ValueError(f"协议资产与当前输入不一致：{mismatches}")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只核对，不改写资产")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--chinese-event-table", type=Path, default=DEFAULT_CHINESE_EVENT_TABLE
    )
    args = parser.parse_args(argv)
    documents = build_documents(args.chinese_event_table)
    paths = write_or_check_documents(documents, args.output_root, check=args.check)
    action = "已核对" if args.check else "已生成"
    print(f"{action} {len(paths)} 个协议资产：{args.output_root}")
    return 0

# --- Merged from catalog.py ---
"""发现历史词级配置及其继承关系；当前协议导航见 protocol_catalog。"""


from difflib import get_close_matches
from pathlib import Path

import yaml

from braindecoding.config import PROJECT_ROOT, deep_merge, load_yaml_with_extends, resolve_config_path
from braindecoding.experiment import experiment_identity, resolve_experiment_config


RUNNABLE_EXPERIMENT_CATEGORIES = (
    "main",
    "ablation",
    "scaling",
    "development",
)


def experiment_selector(identity: dict) -> str:
    """把实验身份转换为用户使用的稳定 selector。"""
    return "/".join(
        (
            identity["dataset"],
            identity["subject_scope"],
            identity["experiment_id"],
        )
    )


def _load_raw_config(path: Path, ancestors=()) -> dict:
    """只为发现实验读取继承关系，不展开机器环境变量。"""
    path = resolve_config_path(path).resolve()
    if path in ancestors:
        raise ValueError(f"配置继承出现循环：{path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(str(parent))
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return deep_merge(_load_raw_config(parent_path, (*ancestors, path)), raw)


def discover_experiment_configs(config_root=None, *, categories=("main",)) -> list[dict]:
    """按类别从 canonical 配置树发现实验，不硬编码实验清单。"""
    root = Path(config_root or PROJECT_ROOT / "configs" / "word_decoding")
    categories = frozenset(str(category) for category in categories)
    records = []
    for path in sorted(root.rglob("*.yaml")):
        raw = _load_raw_config(path)
        experiment = raw.get("experiment")
        if not isinstance(experiment, dict) or experiment.get("category") not in categories:
            continue
        identity = experiment_identity(raw)
        resolved_path = path.resolve()
        try:
            relative_config = resolved_path.relative_to(
                PROJECT_ROOT.resolve()
            ).as_posix()
        except ValueError:
            relative_config = resolved_path.relative_to(root.resolve()).as_posix()
        records.append(
            {
                "selector": experiment_selector(identity),
                "identity": identity,
                "config_path": resolved_path,
                "relative_config": relative_config,
                "raw_config": raw,
            }
        )
    selectors = [record["selector"] for record in records]
    if len(selectors) != len(set(selectors)):
        raise ValueError("canonical 配置中存在重复实验 selector。")
    return records


def discover_runnable_experiment_configs(config_root=None) -> list[dict]:
    """兼容旧配置查询；返回的配置不代表当前 CLI 允许启动的实验。"""
    return discover_experiment_configs(
        config_root,
        categories=RUNNABLE_EXPERIMENT_CATEGORIES,
    )


def resolve_selector(selector: str, config_root=None) -> dict:
    """解析用户 selector；不存在时给出最接近的有效候选。"""
    records = discover_runnable_experiment_configs(config_root)
    by_selector = {record["selector"]: record for record in records}
    if selector in by_selector:
        return by_selector[selector]
    candidates = sorted(by_selector)
    nearby = get_close_matches(selector, candidates, n=3, cutoff=0.25)
    hint = nearby or candidates
    rendered = "\n  ".join(hint) if hint else "<无>"
    raise ValueError(f"未知实验：{selector}\n可用或相近候选：\n  {rendered}")


def load_experiment(selector: str, config_root=None, output_root=None) -> tuple[dict, dict]:
    """载入 selector 对应的完整 canonical 配置。"""
    record = resolve_selector(selector, config_root)
    config = load_yaml_with_extends(record["config_path"])
    return record, resolve_experiment_config(config, output_root=output_root)

# --- Merged from protocol_catalog.py ---
"""当前协议的只读导航；不改变冻结配置，也不调度训练或评价。"""


from difflib import get_close_matches
import json
from pathlib import Path

import yaml

from braindecoding.config import PROJECT_ROOT


def repository_path(relative: str, root: Path = PROJECT_ROOT) -> Path:
    """索引中的路径必须位于仓库内。"""
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError(f"协议索引路径超出仓库：{relative}")
    if not path.exists() and root == PROJECT_ROOT.resolve():
        resolved = resolve_config_path(path).resolve()
        if resolved.exists():
            return resolved
    return path


def discover_protocols(index_path=None, *, root=PROJECT_ROOT) -> list[dict]:
    """读取显式协议索引，避免把配置目录自动当成可运行实验目录。"""
    root = Path(root).resolve()
    path = Path(index_path) if index_path else root / "data/manifests/protocol_index.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("协议索引需要 version: 1。")
    records = payload.get("protocols")
    if not isinstance(records, list) or not records:
        raise ValueError("协议索引 protocols 必须是非空列表。")
    identifiers = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("协议索引条目必须是映射。")
        identifier = record.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError(f"协议索引 id 缺失或重复：{identifier}")
        identifiers.add(identifier)
        if record.get("role") not in {"primary", "supplement"}:
            raise ValueError(f"未知协议角色：{identifier}")
        for field in ("title", "protocol", "config"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"协议 {identifier} 缺少 {field}。")
        for field in ("datasets", "entrypoints", "reports", "manifests",
                      "related_protocols", "retained_dependencies"):
            values = record.get(field, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError(f"协议 {identifier} 的 {field} 必须是字符串列表。")
            if field in {"datasets", "entrypoints", "reports"} and not values:
                raise ValueError(f"协议 {identifier} 缺少 {field}。")
        sources = record.get("status_sources", [])
        if not isinstance(sources, list):
            raise ValueError(f"协议 {identifier} 的 status_sources 必须是列表。")
        names = set()
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get("name"), str):
                raise ValueError(f"协议 {identifier} 的状态源缺少 name。")
            if not source["name"] or source["name"] in names:
                raise ValueError(f"协议 {identifier} 的状态源 name 缺失或重复。")
            names.add(source["name"])
            paths = source.get("paths")
            if not isinstance(paths, list) or not paths or any(
                not isinstance(value, str) or not value for value in paths
            ):
                raise ValueError(f"协议 {identifier} 的状态源需要 paths。")
            for relative in paths:
                repository_path(relative, root)
        required_paths = [record["protocol"], record["config"],
                          *record.get("manifests", []), *record.get("related_protocols", [])]
        for relative in required_paths:
            if not repository_path(relative, root).is_file():
                raise FileNotFoundError(f"协议 {identifier} 引用的文件不存在：{relative}")
        for field in ("reports", "retained_dependencies"):
            for relative in record.get(field, []):
                repository_path(relative, root)
    primary = [record["id"] for record in records if record["role"] == "primary"]
    if primary != [payload.get("primary")]:
        raise ValueError("协议索引必须有一个与 primary 一致的主协议。")
    for record in records:
        if record["role"] == "supplement" and record.get("parent") != primary[0]:
            raise ValueError(f"补充协议 {record['id']} 必须引用当前主协议。")
    return records


def resolve_protocol(identifier: str) -> dict:
    records = discover_protocols()
    by_id = {record["id"]: record for record in records}
    if identifier in by_id:
        return by_id[identifier]
    nearby = get_close_matches(identifier, by_id, n=3, cutoff=0.25)
    hint = ", ".join(nearby or by_id)
    raise ValueError(f"未知当前协议：{identifier}。可用或相近候选：{hint}")


def protocol_status(record: dict, *, root=PROJECT_ROOT) -> list[dict]:
    """读取已保存的状态快照；缺少文件不意味着实验尚未运行。"""
    states = []
    for source in record.get("status_sources", []):
        state = {"name": source["name"], "status": "not_recorded", "path": None,
                 "snapshot": None}
        for relative in source["paths"]:
            path = repository_path(relative, root)
            if not path.is_file():
                continue
            state["path"] = path
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(snapshot, dict):
                    raise ValueError("状态文件必须是 JSON 对象。")
                state["snapshot"] = snapshot
                state["status"] = str(snapshot.get("state", snapshot.get("status", "unknown")))
            except (OSError, ValueError) as error:
                state.update(status="unreadable", error=str(error))
            break
        states.append(state)
    return states


if __name__ == "__main__":
    raise SystemExit(main())
