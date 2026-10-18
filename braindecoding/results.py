"""Canonical 运行的统一结果 JSON 结构。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from braindecoding.experiment import (
    evaluation_output_path,
    experiment_identity,
    file_sha256,
)


SCHEMA_VERSION = 1
PRIMARY_VOCABULARY_SIZES = (20, 50, 100, 150)
AUDIT_CONTROLS = (
    "clean",
    "temporal_shift",
    "donor_swap",
    "structure_only",
    "donor_following",
)


def canonical_retrieval(metrics):
    """把现有公共 retrieval 字段无数值变换地映射到稳定名称。"""
    metrics = metrics or {}
    return {
        "top1": metrics.get("micro_recall_at_1"),
        "top10": metrics.get("micro_recall_at_10"),
        "macro_top1": metrics.get("macro_recall_at_1"),
        "macro_top10": metrics.get("macro_recall_at_10"),
        "median_rank": metrics.get("median_rank"),
        "mrr": metrics.get("mean_reciprocal_rank"),
    }


def canonical_vocabulary_result(
    candidate_count,
    *,
    manifest_sha256=None,
    retrieval_metrics=None,
):
    """构造单个冻结候选词表的公共评价结果块。"""
    candidate_count = int(candidate_count)
    metrics = retrieval_metrics or {}
    missing_words = list(metrics.get("missing_vocabulary_words", ()))
    supported_count = metrics.get("observed_vocabulary_size")
    if supported_count is None and retrieval_metrics is not None:
        supported_count = candidate_count - len(missing_words)
    support = {
        "supported_candidate_count": (
            int(supported_count) if supported_count is not None else None
        ),
        "missing_candidate_count": (
            len(missing_words) if retrieval_metrics is not None else None
        ),
        "missing_words": missing_words,
    }

    result = {
        "candidate_count": candidate_count,
        "manifest_sha256": manifest_sha256,
        "support": support,
        "retrieval": canonical_retrieval(metrics),
    }
    internal_prefix = "internal_only_"
    internal_metrics = {
        key[len(internal_prefix) :]: value
        for key, value in metrics.items()
        if key.startswith(internal_prefix)
    }
    if internal_metrics:
        internal_result = canonical_vocabulary_result(
            candidate_count,
            manifest_sha256=manifest_sha256,
            retrieval_metrics=internal_metrics,
        )
        internal_result["query_count"] = int(
            internal_metrics.get("query_count", 0)
        )
        if internal_metrics.get("context_radius") is not None:
            internal_result["context_radius"] = int(
                internal_metrics["context_radius"]
            )
        result["internal_only"] = internal_result
    if retrieval_metrics is None:
        result["status"] = "not_run"
    return result


def load_vocabulary_assets(directory):
    """从调用方指定目录读取四个冻结词表。"""
    directory = Path(directory)
    manifests = {}
    for size in PRIMARY_VOCABULARY_SIZES:
        path = directory / f"vocabulary_N{size}.json"
        if not path.is_file():
            raise FileNotFoundError(f"找不到冻结候选词表：{path}")
        manifests[size] = json.loads(path.read_text(encoding="utf-8"))
    return manifests


def evaluate_vocabulary_manifests(
    predictions,
    targets,
    words,
    vocabulary_manifests,
):
    """用现有公共数学一次生成四个冻结词表的 canonical 结果块。"""
    from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval_metrics

    results = {}
    for size in PRIMARY_VOCABULARY_SIZES:
        manifest = vocabulary_manifests.get(size)
        if manifest is None:
            raise ValueError(f"缺少 N{size} 冻结候选词表 manifest。")
        vocabulary = manifest["vocabulary"]
        if len(vocabulary) != size:
            raise ValueError(f"N{size} manifest 的实际词数不一致。")
        metrics = fixed_vocabulary_retrieval_metrics(
            predictions,
            targets,
            words,
            vocabulary,
            top_ks=(1, 10),
            vocabulary_name=f"frozen_N{size}",
        )
        results[size] = canonical_vocabulary_result(
            size,
            manifest_sha256=manifest.get("manifest_sha256"),
            retrieval_metrics=metrics,
        )
    return results


def build_evaluation_result(
    config,
    *,
    split,
    query_count,
    event_table_sha256,
    subjects,
    checkpoint,
    vocabulary_results,
    status="evaluation_completed",
):
    """构造跨数据集一致的 canonical evaluation JSON。"""
    if split not in {"val", "test"}:
        raise ValueError("评价 split 只能是 val 或 test。")
    supplied = {int(size): copy.deepcopy(value) for size, value in vocabulary_results.items()}
    vocabularies = {}
    for size in PRIMARY_VOCABULARY_SIZES:
        block = supplied.get(size, canonical_vocabulary_result(size))
        if int(block.get("candidate_count", -1)) != size:
            raise ValueError(f"词表结果块与 N{size} 的候选数量不一致。")
        vocabularies[str(size)] = block
    unknown = sorted(set(supplied) - set(PRIMARY_VOCABULARY_SIZES))
    if unknown:
        raise ValueError(f"canonical 主评价不支持词表规模：{unknown}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "experiment": experiment_identity(config),
        "data": {
            "split": split,
            "query_count": int(query_count),
            "event_table_sha256": event_table_sha256,
            "subjects": [str(subject) for subject in subjects],
        },
        "checkpoint": copy.deepcopy(checkpoint),
        "vocabularies": vocabularies,
    }


def canonical_evaluation_from_legacy(
    config,
    legacy_summary,
    *,
    event_table_path,
    subjects,
    checkpoint_path,
    checkpoint,
    vocabulary_manifest_sha256=None,
    vocabulary_results=None,
):
    """把未改数学的现有评价结果封装成 canonical 文件结构。"""
    metrics = legacy_summary["metrics"]
    candidate_count = int(metrics.get("vocabulary_size", 0))
    if candidate_count not in PRIMARY_VOCABULARY_SIZES:
        raise ValueError(f"评价候选词表规模不属于冻结主集合：{candidate_count}")
    if vocabulary_results is None:
        block = canonical_vocabulary_result(
            candidate_count,
            manifest_sha256=vocabulary_manifest_sha256,
            retrieval_metrics=metrics,
        )
        vocabulary_results = {candidate_count: block}
    return build_evaluation_result(
        config,
        split=legacy_summary["split"],
        query_count=legacy_summary["query_rows"],
        event_table_sha256=file_sha256(event_table_path),
        subjects=subjects,
        checkpoint={
            "path": Path(checkpoint_path).name,
            "sha256": file_sha256(checkpoint_path),
            "epoch": (
                int(checkpoint["epoch"])
                if checkpoint.get("epoch") is not None
                else None
            ),
            "update": (
                int(checkpoint["optimizer_updates"])
                if checkpoint.get("optimizer_updates") is not None
                else None
            ),
        },
        vocabulary_results=vocabulary_results,
        status=legacy_summary["status"],
    )


def write_evaluation_result(config, result, *, smoke=False):
    """按 canonical/legacy 路径规则保存一个明确 split 的评价结果。"""
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("canonical evaluation 缺少受支持的 schema_version。")
    split = result.get("data", {}).get("split")
    path = evaluation_output_path(config, split, smoke=smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _best_update(legacy_summary):
    best_epoch = legacy_summary.get("best_epoch")
    for record in legacy_summary.get("history", ()):  # 保留原 epoch/update 语义。
        if record.get("epoch") == best_epoch:
            value = record.get("optimizer_updates")
            return int(value) if value is not None else None
    return None


def _checkpoint_result(output_dir, name):
    path = Path(output_dir) / name
    return {"path": name, "sha256": file_sha256(path)}


def build_training_summary(config, legacy_summary):
    """在不改训练循环的前提下封装 update/epoch 两种预算。"""
    training_config = config.get("training", {})
    target_updates = legacy_summary.get("target_updates")
    if target_updates is None:
        target_updates = training_config.get("max_updates")
    completed_updates = legacy_summary.get("optimizer_updates")
    completed_updates = (
        int(completed_updates) if completed_updates is not None else None
    )
    best_update = _best_update(legacy_summary)
    if target_updates is not None:
        budget = {"type": "updates", "value": int(target_updates)}
    else:
        budget = {"type": "epochs", "value": int(training_config["epochs"])}
    history = legacy_summary.get("history", ())
    output_dir = training_config.get("output_dir", ".")
    best_epoch = legacy_summary.get("best_epoch")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment_identity(config),
        "status": legacy_summary.get("status"),
        "training": {
            "budget": budget,
            "completed_epochs": int(len(history)),
            "completed_updates": completed_updates,
        },
        "selection": {
            "split": "val",
            "metric": legacy_summary.get("selection_metric"),
            "best_value": legacy_summary.get("best_score"),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_update": best_update,
        },
        "checkpoint": {
            "best": _checkpoint_result(output_dir, "best.pt"),
            "last": (
                _checkpoint_result(output_dir, "last.pt")
                if training_config.get("save_last_checkpoint", True)
                else None
            ),
        },
        "test_status": legacy_summary.get("test_status", "locked_not_evaluated"),
        "details": copy.deepcopy(legacy_summary),
    }


def build_audit_summary(config, *, checkpoint, query_set, controls):
    """构造固定 control 名称、未运行项显式标记的 validation audit 摘要。"""
    if query_set.get("split") != "val":
        raise ValueError("canonical audit summary 本轮只允许 validation query set。")
    unknown = sorted(set(controls) - set(AUDIT_CONTROLS))
    if unknown:
        raise ValueError(f"未知 audit control：{unknown}")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment_identity(config),
        "checkpoint": copy.deepcopy(checkpoint),
        "query_set": copy.deepcopy(query_set),
        "controls": {
            name: copy.deepcopy(controls.get(name, {"status": "not_run"}))
            for name in AUDIT_CONTROLS
        },
    }


def write_audit_summary(config, result):
    """把公共审计摘要写入当前 canonical run 的 validation 目录。"""
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("canonical audit summary 缺少受支持的 schema_version。")
    path = (
        Path(config["training"]["output_dir"])
        / "audits"
        / "validation"
        / "summary.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


_REPORT_NOTICE = (
    "> 自动生成的人类可读视图。\n"
    "> 规范 JSON 文件仍是唯一机器真源。\n"
    "> 请勿手工编辑本文件。"
)


_HUMAN_TEXT = {
    "completed": "已完成",
    "early_stopping": "早停",
    "epochs": "轮次",
    "failed": "失败",
    "locked_not_evaluated": "已锁定，未评价",
    "maximum_epochs_reached": "达到最大训练轮次",
    "missing_true_class_support": "缺少真实类别支持",
    "not available": "不可用",
    "not_run": "未运行",
    "running": "运行中",
    "update_budget_reached": "达到更新预算",
    "updates": "更新",
}


def _load_json_if_present(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6f}"
    return _HUMAN_TEXT.get(str(value), str(value))


def render_run_report(config):
    """只读取 canonical JSON，生成固定使用中文的紧凑 Markdown 报告。"""
    output = Path(config["training"]["output_dir"])
    manifest = _load_json_if_present(output / "run_manifest.json") or {}
    training = _load_json_if_present(output / "training_summary.json")
    evaluation = _load_json_if_present(output / "evaluation" / "val.json")
    audit = _load_json_if_present(output / "audits" / "validation" / "summary.json")
    identity = experiment_identity(config)
    selector = "/".join(
        (
            identity["dataset"],
            identity["subject_scope"],
            identity["experiment_id"],
        )
    )
    checkpoint = (evaluation or {}).get("checkpoint", {})
    lines = [
        _REPORT_NOTICE,
        "",
        "# 实验",
        "",
        f"- 实验选择器：`{selector}`",
        f"- 数据集：`{identity['dataset']}`",
        f"- 受试者范围：`{identity['subject_scope']}`",
        f"- 随机种子：`{identity['seed']}`",
        f"- 模型条件：`{'神经上下文' if config['model'].get('use_transformer') else '词级'}`",
        f"- 检查点 SHA-256：`{checkpoint.get('sha256') or '—'}`",
        "",
        "## 训练",
        "",
        "| 字段 | 值 |",
        "|---|---|",
    ]
    if training is None:
        lines.append("| 状态 | 未运行 |")
    else:
        block = training.get("training", {})
        selection = training.get("selection", {})
        details = training.get("details", {})
        budget = block.get("budget", {})
        lines.extend(
            [
                f"| 训练预算 | {_display(budget.get('value'))} {_display(budget.get('type'))} |",
                f"| 已完成轮次 | {_display(block.get('completed_epochs'))} |",
                f"| 已完成更新数 | {_display(block.get('completed_updates'))} |",
                f"| 最佳轮次 | {_display(selection.get('best_epoch'))} |",
                f"| 最佳更新数 | {_display(selection.get('best_update'))} |",
                f"| 检查点选择指标 | {_display(selection.get('metric'))} |",
                f"| 最佳验证值 | {_display(selection.get('best_value'))} |",
                f"| 停止原因 | {_display(details.get('stop_reason'))} |",
                f"| 最终学习率 | {_display(details.get('final_learning_rate'))} |",
            ]
        )
        if config["training"].get("warm_start_from"):
            lines.extend(
                [
                    f"| 预热来源 | {config['training']['warm_start_from']} |",
                    f"| 脑编码器冻结轮次 | {_display(config['training'].get('freeze_brain_encoder_epochs'))} |",
                ]
            )

    lines.extend(
        [
            "",
            "## 验证集结果",
            "",
            "| N | 支持类别 | Macro Top-1 | Macro Top-10 | Top-1 | Top-10 | 中位排名 | MRR |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    vocabularies = (evaluation or {}).get("vocabularies", {})
    for size in PRIMARY_VOCABULARY_SIZES:
        block = vocabularies.get(str(size), {})
        retrieval = block.get("retrieval", {})
        support = block.get("support", {})
        supported = support.get("supported_candidate_count")
        support_text = f"{supported}/{size}" if supported is not None else "—"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(size),
                    support_text,
                    _display(retrieval.get("macro_top1")),
                    _display(retrieval.get("macro_top10")),
                    _display(retrieval.get("top1")),
                    _display(retrieval.get("top10")),
                    _display(retrieval.get("median_rank")),
                    _display(retrieval.get("mrr")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 验证集审计",
            "",
            "| 对照条件 | Macro Top-1 | Macro Top-10 | 中位排名 | MRR |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    controls = (audit or {}).get("controls", {})
    if not controls:
        lines.append("| 审计 | 未运行 | — | — | — |")
    else:
        control_labels = {
            "clean": "干净条件",
            "temporal_shift": "时间错位",
            "structure_only": "结构基线（神经特征全零）",
        }
        for name in ("clean", "temporal_shift", "structure_only"):
            control = controls.get(name, {})
            block = control.get("results_by_vocabulary", {}).get("N50", {})
            lines.append(
                f"| {control_labels[name]} | {_display(block.get('macro_recall_at_1'))} | "
                f"{_display(block.get('macro_recall_at_10'))} | "
                f"{_display(block.get('median_rank'))} | "
                f"{_display(block.get('mean_reciprocal_rank'))} |"
            )
        donor = controls.get("donor_swap", {}).get("aggregate", {}).get("N50", {})
        donor_retrieval = donor.get("retrieval", {})
        def aggregate_text(name):
            value = donor_retrieval.get(name, {})
            if value.get("mean") is None:
                return "—"
            return (
                f"{value['mean']:.6f} ± {value.get('std', 0.0):.6f} "
                f"[{value.get('ci95_low', value['mean']):.6f}, "
                f"{value.get('ci95_high', value['mean']):.6f}]"
            )
        lines.append(
            f"| 供体替换均值 | {aggregate_text('macro_top1')} | "
            f"{aggregate_text('macro_top10')} | {aggregate_text('median_rank')} | "
            f"{aggregate_text('mrr')} |"
        )

    event_sha = (evaluation or {}).get("data", {}).get("event_table_sha256")
    scientific_sha = manifest.get("scientific_config_sha256")
    vocab_sha = None
    if "50" in vocabularies:
        vocab_sha = vocabularies["50"].get("manifest_sha256")
    lines.extend(
        [
            "",
            "## 来源与指纹",
            "",
            f"- 科学配置 SHA-256：`{scientific_sha or '—'}`",
            f"- 事件表 SHA-256：`{event_sha or '—'}`",
            f"- 检查点 SHA-256：`{checkpoint.get('sha256') or '—'}`",
            f"- N50 词表 manifest SHA-256：`{vocab_sha or '—'}`",
            f"- 测试集状态：`{_display((training or {}).get('test_status', 'locked_not_evaluated'))}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_run_report(config):
    """把中文只读派生视图写到当前 run 的 summary.md。"""
    path = Path(config["training"]["output_dir"]) / "summary.md"
    if not path.parent.is_dir():
        raise FileNotFoundError(f"canonical run 不存在：{path.parent}")
    path.write_text(render_run_report(config), encoding="utf-8")
    return path

# --- Merged from reporting.py ---
"""仓库级报告、导出和瞬时缓存的统一路径约定。"""


from pathlib import Path

from braindecoding.config import PROJECT_ROOT


REPORTS_ROOT = PROJECT_ROOT / "data" / "results"
REPORT_STUDIES_ROOT = REPORTS_ROOT / "studies"
REPORT_AUDITS_ROOT = REPORTS_ROOT / "audits"
REPORT_EXPORTS_ROOT = REPORTS_ROOT / "exports"
REPORT_PROTOCOLS_ROOT = REPORTS_ROOT / "protocols"

CACHE_ROOT = PROJECT_ROOT / ".cache"
AUDIT_CACHE_ROOT = CACHE_ROOT / "audits"
RUNTIME_CACHE_ROOT = CACHE_ROOT / "runs"


def _safe_segment(value: str, *, label: str) -> str:
    """拒绝绝对路径和跨目录片段，保持报告层级可预测。"""
    segment = str(value).strip()
    if (
        not segment
        or segment in {".", ".."}
        or any(character in segment for character in ("/", "\\", ":"))
    ):
        raise ValueError(f"{label} 必须是单个非空路径片段：{value!r}")
    return segment


def study_report_dir(dataset: str | None, study: str) -> Path:
    """返回研究目录；共同研究传入 ``dataset=None``，不产生目录。"""
    study = _safe_segment(study, label="study")
    if dataset is None:
        return REPORT_STUDIES_ROOT / study
    return REPORT_STUDIES_ROOT / _safe_segment(dataset, label="dataset") / study


def runtime_cache_dir(job: str) -> Path:
    """返回后台任务日志和 PID 的唯一合法根目录。"""
    return RUNTIME_CACHE_ROOT / _safe_segment(job, label="job")

# --- Merged from report_layout.py ---
"""整理旧报告图表，或仅重画已完成报告；不启动训练、解码或统计导出。"""


import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
import re
import shutil
import time

from braindecoding.config import PROJECT_ROOT


MEDIA_SUFFIXES = {".png", ".pdf", ".svg"}
RENDERERS = {
    "revision_gain": ("unified_sentence_v2_revision_gain", "results.json",
                      "completed_revision_gain_preview", "unified_sentence_v2_revision_gain"),
    "revision_gain_full": ("unified_sentence_v2_revision_gain_full", "results.json",
                           "completed_full_revision_gain", "unified_sentence_v2_revision_gain_full_results"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"路径超出仓库：{path}")
    return resolved


def _records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        for child in value.values():
            yield from _records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records(child)


def normalize_legacy_figures(root: Path = PROJECT_ROOT, *, replace: bool = False) -> dict:
    """将 reports/figures/{study} 归入 study，保留原字节和来源记录。

    所有冲突、路径和图表哈希先检查，再移动；replace 仅供显式重画使用。
    修改的 Markdown 和原始 sources.json 备份到 .cache/audits/report_layout。
    """
    root = root.resolve()
    legacy = _inside(root, root / "reports/figures")
    studies = _inside(root, root / "reports/studies")
    moves, updates, locations = [], {}, {}
    if legacy.exists():
        for folder in sorted(legacy.iterdir()):
            if not folder.is_dir() or folder.is_symlink():
                raise ValueError(f"旧图表目录中存在非研究目录：{folder}")
            destination = _inside(root, studies / folder.name)
            if not destination.is_dir():
                raise ValueError(f"图表没有对应的研究目录：{folder.name}")
            for source in sorted(folder.iterdir()):
                _inside(root, source)
                if not source.is_file() or source.is_symlink() or source.suffix not in MEDIA_SUFFIXES | {".json"}:
                    raise ValueError(f"旧图表目录中存在未知产物：{source}")
                target = _inside(root, destination / source.name)
                if target.exists() and (not replace or not target.is_file()):
                    raise FileExistsError(f"目标已存在，未修改任何文件：{target}")
                moves.append((source, target))
                locations[source.relative_to(root).as_posix()] = target.relative_to(root).as_posix()

    for source, target in moves:
        if source.suffix != ".json":
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        for record in _records(payload):
            original = record["path"]
            if not isinstance(original, str) or not isinstance(record["sha256"], str):
                raise ValueError(f"来源记录的路径或哈希无效：{source}")
            if original.startswith("reports/figures/") and original not in locations:
                raise ValueError(f"来源记录引用了未找到的图表：{original}")
            if original in locations:
                referenced = _inside(root, root / original)
                if _sha256(referenced) != record["sha256"]:
                    raise ValueError(f"图表与来源哈希不符：{original}")
                record.setdefault("original_path", original)
                record["path"] = locations[original]
        updates[target] = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    # 冻结生成器也可能只重写 Markdown；已有标准图表可以直接修复链接。
    if studies.is_dir():
        for folder in sorted(studies.iterdir()):
            if not folder.is_dir():
                continue
            prefix = f"../../figures/{folder.name}/"
            for path in sorted(folder.glob("*.md")):
                _inside(root, path)
                original = path.read_bytes()
                changed = original.replace(("](" + prefix).encode("utf-8"), b"](")
                if changed != original:
                    for match in re.finditer(re.escape("](" + prefix) + r"([^\s)]+)\)", original.decode("utf-8")):
                        target = _inside(root, folder / match.group(1).split("#", 1)[0])
                        if not target.is_file() and target not in {new for _, new in moves}:
                            raise ValueError(f"报告引用的图表未找到：{target}")
                    updates[path] = changed

    if not moves and not updates and not legacy.exists():
        return {"moved_files": [], "updated_files": [], "audit": None}

    audit = _inside(root, root / f".cache/audits/report_layout/{time.time_ns()}")
    audit.mkdir(parents=True)
    originals = (
        {source for source, _ in moves}
        | {target for _, target in moves if target.exists()}
        | {path for path in updates if path.exists()}
    )
    for path in sorted(originals):
        backup = audit / "originals" / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, backup)

    moved = []
    for source, target in moves:
        digest = _sha256(source)
        source.replace(target)
        moved.append({"from": source.relative_to(root).as_posix(),
                      "to": target.relative_to(root).as_posix(), "original_sha256": digest})
    changed = []
    for path, content in updates.items():
        before = _sha256(path)
        path.write_bytes(content)
        changed.append({"path": path.relative_to(root).as_posix(),
                        "before_sha256": before, "after_sha256": _sha256(path)})
    if legacy.exists():
        for folder in legacy.iterdir():
            folder.rmdir()
        legacy.rmdir()
    result = {"moved_files": moved, "updated_files": changed,
              "audit": audit.relative_to(root).as_posix()}
    (audit / "migration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def check_layout(root: Path = PROJECT_ROOT) -> list[str]:
    """只读检查目录、旧图表链接及图表来源中的文件哈希。"""
    root = root.resolve()
    rep_rel = "data/results" if (root / "data/results").exists() and not (root / "reports").exists() else "reports"
    studies = root / f"{rep_rel}/studies"
    problems = []
    for path in (root / f"{rep_rel}/figures", studies / "figures"):
        if path.exists():
            problems.append(f"公共图表目录仍存在：{path.relative_to(root).as_posix()}")
    if not studies.is_dir():
        return problems + ["缺少 reports/studies"]
    for path in studies.iterdir():
        if path.is_file():
            problems.append(f"研究根目录中有散落文件：{path.relative_to(root).as_posix()}")
    for path in studies.rglob("*.md"):
        if "../../figures/" in path.read_text(encoding="utf-8"):
            problems.append(f"仍引用旧图表路径：{path.relative_to(root).as_posix()}")
    for path in studies.rglob("*sources.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for record in _records(payload):
                if not isinstance(record["path"], str) or not isinstance(record["sha256"], str):
                    raise ValueError("来源记录的路径或哈希无效")
                if record["path"].startswith(("reports/figures/", "reports/studies/")):
                    target = _inside(root, root / record["path"])
                    if not target.is_file() or _sha256(target) != record["sha256"]:
                        problems.append(f"图表来源无法核对：{record['path']}")
        except (ValueError, OSError, TypeError) as error:
            problems.append(f"图表来源无法读取：{path.relative_to(root).as_posix()}：{error}")
    return problems


def render_figures(name: str, dataset: str | None = None, *, root: Path = PROJECT_ROOT) -> dict:
    """只调用固定的 render 函数，完成后整理旧生成器的输出。"""
    study, filename, status, module_name = RENDERERS[name]
    if name == "dataset_revision_gain_full":
        if dataset not in {"pallier2025", "chineseeeg2_littleprince"}:
            raise ValueError("单数据集图表需要指定 --dataset")
        filename = f"{dataset}_completed.json"
    elif dataset is not None:
        raise ValueError("--dataset 仅用于 dataset_revision_gain_full")
    path = root / "reports/studies" / study / filename
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != status:
        raise ValueError(f"只重画已完成的匹配报告：{path}")
    normalize_legacy_figures(root)
    legacy = _inside(root.resolve(), root / "reports/figures" / study)
    legacy.mkdir(parents=True)
    module = import_module(f"braindecoding.{module_name}")
    if name in {"preview", "preview_chance"}:
        module.render()
    elif name == "dataset_revision_gain_full":
        module.render(report, path)
    else:
        module.render(report)
    return normalize_legacy_figures(root, replace=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="只读检查目录和图表来源")
    action.add_argument("--normalize", action="store_true", help="整理冻结旧生成器的已有输出")
    action.add_argument("--render", choices=RENDERERS, help="从已有已完成报告重画，不重算实验")
    parser.add_argument("--dataset", choices=("pallier2025", "chineseeeg2_littleprince"))
    args = parser.parse_args(argv)
    if args.dataset and not args.render:
        parser.error("--dataset 需要 --render")
    if args.check:
        problems = check_layout()
        print(json.dumps({"status": "needs_cleanup" if problems else "clean", "problems": problems}, ensure_ascii=False, indent=2))
        return int(bool(problems))
    result = render_figures(args.render, args.dataset) if args.render else normalize_legacy_figures()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
