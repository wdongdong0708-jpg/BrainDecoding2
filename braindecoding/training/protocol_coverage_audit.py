"""Read-only train/validation inventory for the unified rolling protocol."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends, project_path
from braindecoding.data import chineseeeg2, pallier2025
from braindecoding.data.text import load_text_embedding_cache
from braindecoding.reporting import REPORT_AUDITS_ROOT
from braindecoding.training.word import SentenceBatchSampler


DATASETS = {
    "pallier2025": "configs/word_decoding/pallier2025/rolling_revision/local_window_revision.yaml",
    "chineseeeg2_littleprince": "configs/word_decoding/chineseeeg2_littleprince/rolling_revision/full_coverage_local_window_revision.yaml",
}
MANIFEST_ROOT = PROJECT_ROOT / "data/manifests/rolling_sentence"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_source(dataset, split):
    """Never provide a test data access path in the development workflow."""
    if split not in {"train", "val"}:
        raise ValueError("The unified development protocol only permits train/val.")
    config = load_yaml_with_extends(PROJECT_ROOT / DATASETS[dataset])
    path = project_path(config["cache"]["event_table"])
    if dataset == "pallier2025":
        table = pallier2025.load_event_table(
            path, split=split, subjects=config["dataset"]["subjects"], trainable_only=True
        )
        table = pallier2025.add_runtime_context_uid(table)
        table["protocol_group"] = table[pallier2025.RUNTIME_CONTEXT_COLUMN]
        table["material_event"] = table["运行编号"].astype(str) + "|" + table["BIDS事件行号"].astype(str)
    else:
        table = chineseeeg2.load_event_table(path, split=split, subjects=config["dataset"]["subjects"])
        table["protocol_group"] = table["sentence_uid"]
        table["material_event"] = table["voice_version"].astype(str) + "|" + table["source_word_id"].astype(str)
    for target, source in {
        "event_id": "事件编号", "subject_id": "受试者", "recording_id": "记录编号",
        "normalized_word": "标准词", "split": "数据划分", "onset_seconds": "开始时间",
    }.items():
        table[target] = table[source]
    if table.event_id.duplicated().any() or not table.split.eq(split).all():
        raise ValueError("Duplicate event IDs or split leakage.")
    return config, table


def build_training_vocabulary(table):
    if not table["split"].eq("train").all():
        raise ValueError("Vocabulary construction must use train only.")
    if table.groupby("material_event")["normalized_word"].nunique().gt(1).any():
        raise ValueError("The same material position has inconsistent words.")
    material = table.drop_duplicates("material_event")
    counts = material.normalized_word.astype(str).value_counts().to_dict()
    if "" in counts:
        raise ValueError("Empty training word.")
    words = sorted(counts)
    return words, {word: int(counts[word]) for word in words}


def coverage_inventory(table, *, seed=0):
    groups = table.groupby("protocol_group", sort=False)
    sizes = groups.size().to_numpy()
    availability = {str(d): int(np.maximum(sizes - d, 0).sum()) for d in range(3)}
    if groups.subject_id.nunique().gt(1).any() or groups.recording_id.nunique().gt(1).any():
        raise ValueError("Context groups cross subject or recording boundaries.")
    sampler = SentenceBatchSampler(table.protocol_group, 128, True, seed, preserve_complete_groups=True)
    batches = list(sampler)
    indices = np.concatenate(batches)
    if len(indices) != len(table) or len(np.unique(indices)) != len(table):
        raise ValueError("Training sampler duplicates or omits events.")
    gap_count = 0
    for _, rows in groups:
        delta = np.diff(rows["记录内序号"].to_numpy(dtype=int))
        if np.any(delta <= 0):
            raise ValueError("Context event order is not strictly increasing.")
        gap_count += int(np.count_nonzero(delta != 1))
    return {
        "events": len(table), "subjects": int(table.subject_id.nunique()),
        "material_events": int(table.material_event.nunique()),
        "groups": int(len(sizes)), "group_length_quantiles": dict(zip(
            ["min", "p50", "p95", "max"], np.quantile(sizes, [0, .5, .95, 1]).tolist()
        )),
        "available_target_delay_pairs": availability,
        "all_policy_unique_pairs": sum(availability.values()),
        "all_policy_pair_coverage": {str(d): 1.0 for d in range(3)},
        "within_group_event_order_gaps": gap_count,
        "sampler_unique_events": int(len(np.unique(indices))),
        "sampler_duplicate_events": int(len(indices) - len(np.unique(indices))),
        "batches_seed0_epoch0": len(batches),
        "batch_size_quantiles": np.quantile([len(b) for b in batches], [0, .5, 1]).tolist(),
        "count_basis": "structural enumeration; optimizer execution checked separately",
    }


def audit_dataset(dataset):
    config, train = load_source(dataset, "train")
    _, val = load_source(dataset, "val")
    if set(train.event_id) & set(val.event_id) or set(train.material_event) & set(val.material_event):
        raise ValueError("Train/validation event or material overlap.")
    words, counts = build_training_vocabulary(train)
    embeddings = load_text_embedding_cache(project_path(config["cache"]["text_embeddings"]))
    missing = sorted(set(words) - set(embeddings))
    invalid = [word for word in words if word in embeddings and (
        not np.isfinite(embeddings[word]).all() or np.linalg.norm(embeddings[word]) <= 0
    )]
    coverage = float(val.normalized_word.isin(words).mean())
    manifest = {
        "schema_version": 1, "dataset": dataset, "status": "train_vocabulary_inventory",
        "source_split": "train", "policy": "all_eligible_training_words_lexical_order",
        "source_event_table_sha256": sha256(project_path(config["cache"]["event_table"])),
        "material_deduplication": "run+BIDS_row" if dataset == "pallier2025" else "voice+source_word_id",
        "vocabulary": words, "material_frequencies": counts,
        "vocabulary_sha256": hashlib.sha256(json.dumps(words, ensure_ascii=False).encode()).hexdigest(),
        "test_metrics_inspected": False,
    }
    write_json(MANIFEST_ROOT / f"{dataset}_train_vocabulary.json", manifest)
    report = {
        "dataset": dataset, "train": coverage_inventory(train), "val": coverage_inventory(val),
        "vocabulary_size": len(words), "training_embedding_missing": missing,
        "training_embedding_invalid": invalid, "validation_token_coverage": coverage,
        "validation_unseen_types": int(val.loc[~val.normalized_word.isin(words), "normalized_word"].nunique()),
        "legacy_model": config["model"], "legacy_text_model": config["text_embedding"],
        "legacy_cutoff_policy": config["training"].get("revision_cutoff_policy", "sample_one"),
        "test_signals_opened": False, "test_predictions_generated": False,
    }
    out = REPORT_AUDITS_ROOT / "unified_rolling_v1" / dataset
    write_json(out / "inventory.json", report)
    for split, table in [("train", train), ("val", val)]:
        group_rows = []
        for group, rows in table.groupby("protocol_group", sort=False):
            n = len(rows)
            group_rows.append({"group": group, "subject": rows.subject_id.iloc[0],
                "recording": rows.recording_id.iloc[0], "words": n,
                **{f"k{d}_available": max(n-d, 0) for d in range(3)}})
        pd.DataFrame(group_rows).to_csv(out / f"{split}_coverage.csv", index=False, encoding="utf-8-sig")
    return report


def run_audit():
    reports = [audit_dataset(dataset) for dataset in DATASETS]
    gpu = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    resources = {"platform": platform.platform(), "torch": torch.__version__,
        "gpu": None if gpu is None else {"name": gpu.name, "total_bytes": gpu.total_memory},
        "workspace_free_bytes": shutil.disk_usage(PROJECT_ROOT).free,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()}
    write_json(REPORT_AUDITS_ROOT / "unified_rolling_v1/resources.json", resources)
    lines = ["# 统一滚动协议：数据与资源盘点", "", "状态：开发盘点；未打开测试信号或生成测试预测。", "",
        "| 数据集 | 训练事件 | 全覆盖 K0/K1/K2 对数 | 训练词库 | 验证词覆盖率 | 组内序号断档 |",
        "|---|---:|---|---:|---:|---:|"]
    for r in reports:
        t = r["train"]
        lines.append(f"| {r['dataset']} | {t['events']:,} | {' / '.join(str(n) for n in t['available_target_delay_pairs'].values())} | {r['vocabulary_size']:,} | {r['validation_token_coverage']:.2%} | {t['within_group_event_order_gaps']} |")
    lines += ["", "覆盖对数是按现有分组枚举的结构检查；尚不代表新模型已训练。词库只由训练材料构建，验证覆盖率仅作诊断。", "",
        "## 需进入共同实现的事项", "", "- 组内断档必须检查并按统一规则切开，不能跨被排除的词建立邻居。",
        "- 旧训练入口仍绑定各自 50 词选模指标；统一的完整词库选模需单独接入。",
        "- 共同文本表示将写入新缓存，不覆盖旧 T5/Mengzi 数据产品。",
        "- 全覆盖预算以有效目标—延迟对为准，分母按边界条件分别计算。",
        "", f"设备：{resources['gpu']}。工作区可用磁盘：{resources['workspace_free_bytes']/2**30:.1f} GiB。"]
    path = REPORT_AUDITS_ROOT / "unified_rolling_v1/summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(path), "datasets": [{"dataset": r["dataset"], "vocabulary_size": r["vocabulary_size"],
        "validation_token_coverage": r["validation_token_coverage"], "train": r["train"]} for r in reports]}


if __name__ == "__main__":
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))
