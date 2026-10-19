"""本地小样本极速联调 + 云端 CPU 预热 / GPU 全量训练 双模工作流执行器。

提供统一的 CLI 后端实现（``smoke``、``cache``、``train``）：
- 环境变量解耦：``BRAINDECODING_DATA_DIR``、``BRAINDECODING_CACHE_DIR``、``BRAINDECODING_RESULTS_DIR``
- 灵活受试者子集选择：``--subject all``、``--num-subjects N``、``--subjects id1,id2,...``
- 动态子集样本数校验（移除写死的全量样本数硬编码断言）
- 子集专属输出目录与 ``subjects.json`` 防覆盖隔离
- ``--require-cache`` 零现场重算保障与 ``--resume auto`` (``last.ckpt``) 断点续训
- ``bf16-mixed`` / ``fp16`` / ``fp32`` 混合精度与 GPU 峰值显存记录
"""

from __future__ import annotations

import copy
import gc
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

from braindecoding.config import (
    PROJECT_ROOT,
    cache_root,
    data_root,
    ensure_default_environment,
    load_yaml_with_extends,
    model_root,
    normalize_project_relative,
    project_path,
    resolve_config_path,
    results_root,
)
from braindecoding.data import chineseeeg2_dataset as chineseeeg2
from braindecoding.data import pallier2025_dataset as pallier2025
from braindecoding.data import text_embeddings as text_data
from braindecoding.data.derived_cache import sha256_file, stable_sha256
from braindecoding.evaluation.retrieval_metrics import fixed_vocabulary_retrieval_metrics
from braindecoding.experiment import file_sha256
from braindecoding.losses import build_siglip_loss
from braindecoding.models import build_brain_embedding_model
from braindecoding.optimizers import (
    build_adamw_for_modules,
    build_cosine_annealing_scheduler,
)
from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import train as chinese_task
from braindecoding.tasks.word_decoding.pallier2025 import train as pallier_task
from braindecoding.training.pallier_alignment_training import strict_bank_metrics
from braindecoding.training.protocol_stage_training import (
    COMMON_TRAINING_KEYS,
    CONFIG_PATH,
    DATASETS,
    STAGES,
    group_column,
)
from braindecoding.training.rolling_revision_training import full_vocabulary_metrics
from braindecoding.training.training_runtime import (
    capture_random_state,
    choose_device,
    cpu_state_dict,
    limit_rows,
    load_checkpoint,
    restore_random_state,
    save_checkpoint,
    save_json,
    set_seed,
)
from braindecoding.training.word_encoder_training import (
    encode_loader,
    make_loader,
    resolve_amp_dtype,
    train_one_epoch,
)


CANONICAL_SUBJECTS: dict[str, tuple[str, ...]] = {
    "chineseeeg2_littleprince": tuple(f"sub-{index:02d}" for index in range(1, 9)),
    "pallier2025": tuple(pallier2025.SUBJECTS),
}

WORKFLOW_STAGES: tuple[str, ...] = ("word", "context", "revision")

STAGE_DISPLAY_NAMES: dict[str, str] = {
    "word": "Stage 1: Word Encoder (CNN)",
    "context": "Stage 2.1: Context Transformer (CNN + 16L Full Window)",
    "revision": "Stage 2.2: Rolling Revision (CNN + 16L Local Window K0/K1/K2)",
}

_STAGE_ALIASES: dict[str, str] = {
    "1": "word",
    "stage1": "word",
    "stage_1": "word",
    "word": "word",
    "word_encoder": "word",
    "2.1": "context",
    "stage2.1": "context",
    "stage_2_1": "context",
    "context": "context",
    "context_transformer": "context",
    "transformer": "context",
    "2.2": "revision",
    "stage2.2": "revision",
    "stage_2_2": "revision",
    "revision": "revision",
    "rolling_revision": "revision",
}

DEFAULT_STAGE_CONFIGS: dict[tuple[str, str], str] = {
    ("pallier2025", "word"): "configs/training/pallier2025_word_encoder.yaml",
    ("pallier2025", "context"): "configs/training/pallier2025_context_transformer.yaml",
    ("pallier2025", "revision"): "configs/training/pallier2025_rolling_revision.yaml",
    ("chineseeeg2_littleprince", "word"): "configs/training/chineseeeg2_word_encoder.yaml",
    ("chineseeeg2_littleprince", "context"): "configs/training/chineseeeg2_context_transformer.yaml",
    ("chineseeeg2_littleprince", "revision"): "configs/training/chineseeeg2_rolling_revision.yaml",
}


def normalize_workflow_stage(stage: str | None) -> str:
    """将用户输入或配置中的阶段标识标准化为 'word' | 'context' | 'revision'。"""
    if stage is None:
        return "word"
    token = str(stage).strip().lower()
    if token in _STAGE_ALIASES:
        return _STAGE_ALIASES[token]
    raise ValueError(
        f"不支持的训练阶段 {stage!r}，可选：{WORKFLOW_STAGES}（或别名 1 / 2.1 / 2.2）"
    )


def canonical_subjects_for_dataset(dataset: str) -> tuple[str, ...]:
    """返回指定数据集的标准受试者有序元组。"""
    if dataset not in CANONICAL_SUBJECTS:
        raise ValueError(
            f"不支持的数据集 {dataset!r}，可选值：{sorted(CANONICAL_SUBJECTS)}"
        )
    return CANONICAL_SUBJECTS[dataset]


def _normalize_subject_token(token: str, canonical: tuple[str, ...]) -> str:
    text = str(token).strip()
    if not text:
        raise ValueError("受试者标识不能为空。")
    if text in canonical:
        return text
    if text.isdigit():
        candidate = f"sub-{int(text):02d}"
        if candidate in canonical:
            return candidate
    lowered = text.lower()
    if lowered.startswith("sub") and lowered[3:].lstrip("-_").isdigit():
        candidate = f"sub-{int(lowered[3:].lstrip('-_')):02d}"
        if candidate in canonical:
            return candidate
    raise ValueError(
        f"未知受试者标识 {token!r}，合法范围：{canonical[0]}..{canonical[-1]}"
    )


def resolve_subject_subset(
    dataset: str,
    *,
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
) -> dict:
    """解析受试者子集参数，返回标准有序受试者列表、独立子集标签与元数据。"""
    canonical = canonical_subjects_for_dataset(dataset)
    explicit_modes = sum(
        [
            bool(subject and str(subject).strip().lower() != "all"),
            num_subjects is not None,
            subjects is not None,
        ]
    )
    if explicit_modes > 1:
        raise ValueError(
            "--subject、--num-subjects 与 --subjects 互斥，请仅指定其中一种子集方式。"
        )

    if subjects is not None:
        if isinstance(subjects, str):
            raw_tokens = [item.strip() for item in subjects.split(",") if item.strip()]
        else:
            raw_tokens = [
                part.strip()
                for item in subjects
                for part in str(item).split(",")
                if part.strip()
            ]
        if not raw_tokens:
            raise ValueError("--subjects 至少需要提供一个有效的受试者 ID。")
        if len(raw_tokens) == 1 and raw_tokens[0].lower() == "all":
            selected = list(canonical)
            selection_mode = "all"
        else:
            normalized_set = {
                _normalize_subject_token(tok, canonical) for tok in raw_tokens
            }
            selected = [subj for subj in canonical if subj in normalized_set]
            selection_mode = "explicit_subjects"
    elif num_subjects is not None:
        count = int(num_subjects)
        if count < 1 or count > len(canonical):
            raise ValueError(
                f"--num-subjects 必须介于 1 和 {len(canonical)} 之间，当前为 {count}。"
            )
        selected = list(canonical[:count])
        selection_mode = "num_subjects"
    else:
        subject_text = str(subject or "all").strip()
        if subject_text.lower() == "all":
            selected = list(canonical)
            selection_mode = "all"
        else:
            raw_tokens = [
                item.strip() for item in subject_text.split(",") if item.strip()
            ]
            normalized_set = {
                _normalize_subject_token(tok, canonical) for tok in raw_tokens
            }
            selected = [subj for subj in canonical if subj in normalized_set]
            selection_mode = "single_subject" if len(selected) == 1 else "explicit_subjects"

    is_full = tuple(selected) == canonical
    subjects_sha = stable_sha256({"dataset": dataset, "subjects": selected})
    if is_full:
        subset_tag = f"all-n{len(selected):02d}"
    elif len(selected) == 1:
        subset_tag = selected[0]
    else:
        subset_tag = f"subset-n{len(selected):02d}-{subjects_sha[:8]}"

    return {
        "dataset": dataset,
        "subjects": selected,
        "subject_count": len(selected),
        "total_canonical_subjects": len(canonical),
        "is_full_dataset": is_full,
        "selection_mode": selection_mode,
        "subset_tag": subset_tag,
        "subjects_sha256": subjects_sha,
    }


def _normalize_precision(precision: str | None) -> tuple[str, str, bool]:
    """返回 (标准精度标识, amp_dtype名称, amp是否开启)。"""
    text = str(precision or "bf16-mixed").strip().lower()
    if text in {"bf16-mixed", "bf16_mixed", "bf16", "bfloat16"}:
        return "bf16-mixed", "bfloat16", True
    if text in {"fp16", "fp16-mixed", "fp16_mixed", "float16", "16-mixed", "16"}:
        return "fp16-mixed", "float16", True
    if text in {"fp32", "float32", "32", "32-true"}:
        return "fp32", "float32", False
    raise ValueError(
        f"不支持的精度模式 {precision!r}，可选：bf16-mixed、fp16-mixed、fp32"
    )


def _resolve_results_root_and_exp_tag(
    exp_tag: str | None = None,
    *,
    ignore_implicit_tag: bool = False,
) -> tuple[Path, str]:
    """解析实验输出根目录与超参实验标签，确保目录层级为 <results_root>/<dataset>/<exp_tag>/<stage>/..."""
    base_root = results_root()
    implicit_tag = ""
    if base_root.parent.name == "runs" and base_root.name not in {
        "runs",
        "studies",
        "exports",
        *DATASETS,
    }:
        implicit_tag = base_root.name
        base_root = base_root.parent
    if ignore_implicit_tag:
        return base_root, ""

    raw = (
        exp_tag
        if exp_tag is not None
        else (os.environ.get("BRAINDECODING_EXP_TAG", "") or implicit_tag)
    )
    text = str(raw or "").strip().strip("/\\")
    if not text or text.lower() in {"none", "default"}:
        return base_root, ""
    if ".." in Path(text).parts:
        raise ValueError(f"exp_tag 不允许包含父目录跳转 '..'：{exp_tag!r}")
    return base_root, text


def _resolve_word_exp_tag(
    word_exp_tag: str | None = None,
    *,
    yaml_word_exp_tag: str | None = None,
) -> tuple[bool, str]:
    """解析阶段 2 (context / revision) 热启动所用的阶段 1 (word) 实验标签。

    返回 ``(是否显式指定了阶段1标签, 规范化后的阶段1标签)``。
    当填入 ``'default'`` / ``'root'`` / ``'none'`` 时，显式指向无标签根目录 ``runs/<dataset>/word/...``。
    """
    raw = (
        word_exp_tag
        if word_exp_tag is not None
        else (
            yaml_word_exp_tag
            if yaml_word_exp_tag is not None
            else os.environ.get("BRAINDECODING_WORD_EXP_TAG", "")
        )
    )
    text = str(raw or "").strip().strip("/\\")
    if not text:
        return False, ""
    if text.lower() in {"default", "root", "none"}:
        return True, ""
    if ".." in Path(text).parts:
        raise ValueError(f"word_exp_tag 不允许包含父目录跳转 '..'：{word_exp_tag!r}")
    return True, text


def workflow_output_directory(
    dataset: str,
    stage: str,
    subset_tag: str,
    seed: int,
    *,
    mode: str = "train",
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    ignore_implicit_tag: bool = False,
) -> Path:
    """计算受试者子集与超参实验标签隔离的实验输出目录（<dataset>/[<exp_tag>/]<stage>/<subset_tag>/seed-<seed>）。"""
    if output_dir is not None:
        return Path(output_dir).resolve()
    stage = normalize_workflow_stage(stage)
    stage_dir = f"smoke_{stage}" if mode == "smoke" else stage
    base_root, resolved_tag = _resolve_results_root_and_exp_tag(
        exp_tag, ignore_implicit_tag=ignore_implicit_tag
    )
    dataset_root = base_root / dataset
    if resolved_tag:
        dataset_root = dataset_root / resolved_tag
    return (
        dataset_root
        / stage_dir
        / subset_tag
        / f"seed-{int(seed):03d}"
    ).resolve()


def _infer_dataset_and_stage_from_yaml(
    raw_cfg: dict,
    *,
    dataset: str | None = None,
    stage: str | None = None,
) -> tuple[str, str]:
    """从 YAML 配置推断或核验 dataset 与 stage（支持 word / context / revision）。"""
    wf_block = raw_cfg.get("workflow") if isinstance(raw_cfg.get("workflow"), dict) else {}
    exp = raw_cfg.get("experiment") if isinstance(raw_cfg.get("experiment"), dict) else {}
    ds_block = raw_cfg.get("dataset") if isinstance(raw_cfg.get("dataset"), dict) else {}
    model_block = raw_cfg.get("model") if isinstance(raw_cfg.get("model"), dict) else {}

    yaml_dataset = wf_block.get("dataset") or exp.get("dataset") or ds_block.get("name")
    raw_yaml_stage = wf_block.get("stage") or exp.get("stage")
    yaml_stage = (
        normalize_workflow_stage(raw_yaml_stage)
        if raw_yaml_stage is not None
        else None
    )
    if yaml_stage is None and "use_transformer" in model_block:
        if not bool(model_block.get("use_transformer")):
            yaml_stage = "word"
        else:
            lwr = model_block.get("local_window_revision")
            pr = model_block.get("prefix_revision")
            has_revision = (
                (isinstance(lwr, dict) and bool(lwr.get("enabled", False)))
                or (isinstance(pr, dict) and bool(pr.get("enabled", False)))
            )
            yaml_stage = "revision" if has_revision else "context"

    norm_explicit_stage = (
        normalize_workflow_stage(stage) if stage is not None else None
    )
    if dataset is not None and yaml_dataset and dataset != yaml_dataset:
        raise ValueError(
            f"--dataset={dataset!r} 与配置文件中的 dataset={yaml_dataset!r} 不一致。"
        )
    if (
        norm_explicit_stage is not None
        and raw_yaml_stage is not None
        and norm_explicit_stage != yaml_stage
    ):
        raise ValueError(
            f"--stage={stage!r} 与配置文件中的 stage={raw_yaml_stage!r} 不一致。"
        )

    final_dataset = str(dataset or yaml_dataset or "pallier2025")
    final_stage = str(norm_explicit_stage or yaml_stage or "word")
    return final_dataset, final_stage


def build_workflow_config(
    dataset: str | None = None,
    *,
    stage: str | None = None,
    seed: int | None = 0,
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    learning_rate: float | None = None,
    num_workers: int | None = None,
    precision: str | None = None,
    mode: str = "train",
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    word_checkpoint: str | Path | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    config_path: str | Path | None = None,
) -> tuple[dict, dict]:
    """构建支持灵活受试者子集、自定义 YAML 配置与环境变量路径解耦的训练配置。"""
    ensure_default_environment()
    explicit_cfg_path: Path | None = None
    explicit_yaml: dict | None = None
    if config_path is not None:
        explicit_cfg_path = resolve_config_path(project_path(config_path))
        if not explicit_cfg_path.is_file():
            raise FileNotFoundError(f"指定的配置文件不存在：{config_path}")
        explicit_yaml = load_yaml_with_extends(explicit_cfg_path)
        dataset, stage = _infer_dataset_and_stage_from_yaml(
            explicit_yaml, dataset=dataset, stage=stage
        )
    else:
        dataset = str(dataset or "pallier2025")
        stage = normalize_workflow_stage(stage)

    if dataset not in DATASETS:
        raise ValueError(f"不支持的数据集 {dataset!r}，可选：{DATASETS}")
    stage = normalize_workflow_stage(stage)

    if seed is None and explicit_yaml is not None:
        seed = explicit_yaml.get("training", {}).get("seed", 0)
    if seed is None:
        seed = 0
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"随机种子必须是非负整数：{seed!r}")
    seed = int(seed)

    subset_spec = resolve_subject_subset(
        dataset, subject=subject, num_subjects=num_subjects, subjects=subjects
    )

    settings = load_yaml_with_extends(CONFIG_PATH)
    adapter = settings["adapters"][dataset]
    loader = (
        pallier_task.load_config
        if dataset == "pallier2025"
        else chinese_task.load_config
    )
    default_source_rel = DEFAULT_STAGE_CONFIGS.get(
        (dataset, stage),
        adapter.get(f"{stage}_source_config", adapter["word_source_config"]),
    )
    active_config_path = (
        explicit_cfg_path
        if explicit_cfg_path is not None
        else resolve_config_path(project_path(default_source_rel))
    )
    active_yaml = (
        explicit_yaml
        if explicit_yaml is not None
        else load_yaml_with_extends(active_config_path)
    )
    cfg = loader(active_config_path)
    yaml_tc = dict(cfg.get("training", {}))

    yaml_exp_tag = None
    yaml_word_exp_tag = None
    if isinstance(active_yaml, dict):
        yaml_exp_tag = (
            (active_yaml.get("workflow") or {}).get("exp_tag")
            if isinstance(active_yaml.get("workflow"), dict)
            else None
        ) or (
            (active_yaml.get("experiment") or {}).get("exp_tag")
            if isinstance(active_yaml.get("experiment"), dict)
            else None
        )
        yaml_word_exp_tag = (
            (
                (active_yaml.get("workflow") or {}).get("word_exp_tag")
                if isinstance(active_yaml.get("workflow"), dict)
                else None
            )
            or (
                (active_yaml.get("experiment") or {}).get("word_exp_tag")
                if isinstance(active_yaml.get("experiment"), dict)
                else None
            )
            or yaml_tc.get("word_exp_tag")
            or yaml_tc.get("warm_start_exp_tag")
        )
    _, resolved_exp_tag = _resolve_results_root_and_exp_tag(
        exp_tag if exp_tag is not None else yaml_exp_tag
    )
    has_explicit_word_tag, resolved_word_exp_tag = _resolve_word_exp_tag(
        word_exp_tag, yaml_word_exp_tag=yaml_word_exp_tag
    )

    effective_precision = (
        precision
        if precision is not None
        else str(yaml_tc.get("precision", "bf16-mixed"))
    )
    precision_label, amp_dtype_name, amp_enabled = _normalize_precision(
        effective_precision
    )

    cfg["experiment"]["output_group"] = (
        active_yaml.get("experiment", {}).get("output_group")
        if active_yaml is not None
        and active_yaml.get("experiment", {}).get("output_group")
        else settings["execution"]["output_group"]
    )
    if active_yaml is None or "model" not in active_yaml:
        cfg["model"] = copy.deepcopy(settings["model"])
    cfg["model"]["use_transformer"] = stage in ("context", "revision")
    if stage in ("word", "context"):
        cfg["model"].pop("local_window_revision", None)
    if active_yaml is None or "text_embedding" not in active_yaml:
        cfg["text_embedding"] = copy.deepcopy(settings["text_embedding"])
    if active_yaml is None or "loss" not in active_yaml:
        cfg["loss"] = copy.deepcopy(settings["loss"])
    if active_yaml is None or not active_yaml.get("cache", {}).get("text_embeddings"):
        cfg["cache"]["text_embeddings"] = str(project_path(adapter["embedding_cache"]))

    text_cache_norm = str(cfg["cache"].get("text_embeddings", "")).replace("\\", "/")
    if "t5_large_layer_0_5" in text_cache_norm or str(
        cfg.get("text_embedding", {}).get("model_name", "")
    ) == "t5-large":
        te = dict(cfg.get("text_embedding", {}))
        is_t5 = str(te.get("model_name", "")) == "t5-large"
        cfg["text_embedding"] = {
            "model_name": "t5-large",
            "layer_fraction": float(te.get("layer_fraction", 0.5)) if is_t5 else 0.5,
            "token_aggregation": str(te.get("token_aggregation", "mean")) if is_t5 else "mean",
            "embedding_dimension": int(te.get("embedding_dimension", 1024)),
            "batch_size": int(te.get("batch_size", 32)) if is_t5 else 32,
            "device": str(te.get("device", "cpu")) if is_t5 else "cpu",
            "local_files_only": bool(te.get("local_files_only", True)),
            "word_normalization": str(
                te.get("word_normalization", "canonical_standard_word")
            ),
        }

    cfg["dataset"]["subjects"] = list(subset_spec["subjects"])
    cfg["dataset"]["allow_subject_subset"] = True
    if not subset_spec["is_full_dataset"]:
        cfg["subject_generalization"] = {
            "enabled": True,
            "subset_tag": subset_spec["subset_tag"],
            "subjects": list(subset_spec["subjects"]),
        }

    tc = cfg["training"]
    common = settings["training"]
    for key in COMMON_TRAINING_KEYS:
        if key not in yaml_tc:
            tc[key] = copy.deepcopy(common[key])
    yaml_epochs = yaml_tc.get("epochs")
    default_epochs = (
        2
        if mode == "smoke"
        else int(
            yaml_epochs
            if yaml_epochs is not None
            else common.get(f"{stage}_epochs", common.get("revision_epochs", 30))
        )
    )
    tc.update(
        seed=seed,
        epochs=int(epochs) if epochs is not None else default_epochs,
        top_ks=list(yaml_tc.get("top_ks", [1, 10])),
        amp=amp_enabled,
        amp_dtype=amp_dtype_name,
        precision=precision_label,
    )
    if batch_size is not None:
        tc["batch_size"] = int(batch_size)
    elif "batch_size" in yaml_tc and explicit_yaml is not None:
        tc["batch_size"] = int(yaml_tc["batch_size"])
    elif mode == "train":
        tc["batch_size"] = int(yaml_tc.get("batch_size", 256))

    if eval_batch_size is not None:
        tc["eval_batch_size"] = int(eval_batch_size)
    elif "eval_batch_size" in yaml_tc and explicit_yaml is not None:
        tc["eval_batch_size"] = int(yaml_tc["eval_batch_size"])
    else:
        tc["eval_batch_size"] = max(
            int(tc["batch_size"]), 512 if mode == "train" else 256
        )
    tc["validation_query_batch_size"] = max(
        int(tc.get("validation_query_batch_size", 256)),
        int(tc["eval_batch_size"]),
    )
    if learning_rate is not None:
        tc["learning_rate"] = float(learning_rate)
    elif "learning_rate" in yaml_tc and explicit_yaml is not None:
        tc["learning_rate"] = float(yaml_tc["learning_rate"])
    elif mode == "train" and int(tc["batch_size"]) >= 256:
        tc["learning_rate"] = 0.0004
    if num_workers is not None:
        tc["num_workers"] = int(num_workers)
    if max_train_rows is not None:
        tc["max_train_rows"] = int(max_train_rows)
    if max_val_rows is not None:
        tc["max_val_rows"] = int(max_val_rows)

    for key in (
        "loss_normalization",
        "preserve_complete_groups",
        "revision_cutoff_policy",
        "freeze_brain_encoder_epochs",
        "warm_start_from",
        "warm_start_seed",
        "warm_start_output_group",
        "pretrained_brain_encoder_checkpoint",
        "save_last_checkpoint",
    ):
        tc.pop(key, None)

    effective_word_exp_tag = (
        resolved_word_exp_tag if has_explicit_word_tag else resolved_exp_tag
    )
    if stage in ("context", "revision"):
        tc.update(
            preserve_complete_groups=bool(yaml_tc.get("preserve_complete_groups", True)),
            freeze_brain_encoder_epochs=int(
                yaml_tc.get("freeze_brain_encoder_epochs", common["freeze_brain_encoder_epochs"])
            ),
            save_last_checkpoint=True,
            warm_start_from=str(yaml_tc.get("warm_start_from", "main_word")),
            warm_start_seed=seed,
            warm_start_output_group=cfg["experiment"]["output_group"],
            warm_start_exp_tag=effective_word_exp_tag,
        )
        if stage == "revision":
            tc["revision_cutoff_policy"] = str(
                yaml_tc.get("revision_cutoff_policy", "all")
            )
        if word_checkpoint:
            tc["pretrained_brain_encoder_checkpoint"] = str(
                project_path(word_checkpoint)
            )
        elif has_explicit_word_tag:
            explicit_word_dir = workflow_output_directory(
                dataset,
                "word",
                subset_spec["subset_tag"],
                seed,
                mode=mode,
                exp_tag=resolved_word_exp_tag,
                ignore_implicit_tag=(resolved_word_exp_tag == ""),
            )
            candidate_ckpt = explicit_word_dir / "best.ckpt"
            if not candidate_ckpt.exists():
                b_pt = explicit_word_dir / "best.pt"
                if b_pt.exists():
                    candidate_ckpt = b_pt
            tc["pretrained_brain_encoder_checkpoint"] = str(candidate_ckpt)
        else:
            default_word_dir = workflow_output_directory(
                dataset,
                "word",
                subset_spec["subset_tag"],
                seed,
                mode=mode,
                exp_tag=resolved_exp_tag,
            )
            candidate_ckpt = default_word_dir / "best.ckpt"
            if not candidate_ckpt.exists():
                b_pt = default_word_dir / "best.pt"
                if b_pt.exists():
                    candidate_ckpt = b_pt
            if not candidate_ckpt.exists() and resolved_exp_tag:
                fallback_word_dir = workflow_output_directory(
                    dataset,
                    "word",
                    subset_spec["subset_tag"],
                    seed,
                    mode=mode,
                    ignore_implicit_tag=True,
                )
                fallback_ckpt = fallback_word_dir / "best.ckpt"
                if not fallback_ckpt.exists():
                    fallback_ckpt = fallback_word_dir / "best.pt"
                if fallback_ckpt.exists():
                    candidate_ckpt = fallback_ckpt
                    tc["warm_start_exp_tag"] = ""
            tc["pretrained_brain_encoder_checkpoint"] = str(candidate_ckpt)

    cfg["evaluation"]["selection_metric"] = (
        cfg.get("evaluation", {}).get("selection_metric")
        or adapter.get(f"main_selection_metric_{stage}", adapter["main_selection_metric_word"])
    )
    cfg["evaluation"]["default_split"] = "val"
    out_dir = workflow_output_directory(
        dataset,
        stage,
        subset_spec["subset_tag"],
        seed,
        mode=mode,
        output_dir=output_dir,
        exp_tag=resolved_exp_tag,
    )
    tc["output_dir"] = str(out_dir)
    cfg["workflow"] = {
        "mode": mode,
        "stage": stage,
        "exp_tag": resolved_exp_tag,
        "word_exp_tag": tc.get("warm_start_exp_tag", effective_word_exp_tag),
        "config_path": normalize_project_relative(active_config_path),
        "precision": precision_label,
        "amp_dtype": amp_dtype_name,
        "subset": subset_spec,
        "environment": {
            "BRAINDECODING_DATA_DIR": str(data_root()),
            "BRAINDECODING_CACHE_DIR": str(cache_root()),
            "BRAINDECODING_RESULTS_DIR": str(results_root()),
            "BRAINDECODING_EXP_TAG": resolved_exp_tag,
            "BRAINDECODING_WORD_EXP_TAG": resolved_word_exp_tag if has_explicit_word_tag else "",
            "BRAINDECODING_MODEL_DIR": str(model_root()),
        },
    }
    return cfg, subset_spec


def verify_cache_ready(cfg: dict, *, splits: tuple[str, ...] = ("train", "val")) -> dict:
    """校验所选受试者子集的事件表、文本嵌入与神经信号缓存是否全部就绪。

    当启用 ``--require-cache`` 时，若任何缓存文件缺失或签名不匹配，立即抛出
    ``FileNotFoundError``，绝不在 GPU 训练实例上现场读取原始信号重算。
    """
    dataset = cfg["experiment"]["dataset"]
    subjects = list(cfg["dataset"]["subjects"])
    event_path = Path(cfg["cache"]["event_table"])
    if not event_path.is_file():
        raise FileNotFoundError(
            f"[require-cache] 事件表缓存缺失：{event_path}。"
            f"请先运行 `python -m braindecoding cache --dataset {dataset}` 或解包缓存归档。"
        )

    text_path = Path(cfg["cache"]["text_embeddings"])
    text_sidecar = text_path.with_suffix(".json")
    if not text_path.is_file() or not text_sidecar.is_file():
        raise FileNotFoundError(
            f"[require-cache] 文本向量缓存缺失：{text_path}。"
            f"请先运行 `python -m braindecoding cache --dataset {dataset}`。"
        )
    expected_sig = text_data.text_embedding_signature(cfg["text_embedding"])
    try:
        embedding_map = text_data.load_text_embedding_cache(
            text_path, expected_signature=expected_sig
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"[require-cache] 文本向量缓存校验失败（{text_path}）：{exc}"
        ) from exc

    if dataset == "chineseeeg2_littleprince":
        signal_dir = Path(cfg["cache"]["eeg_dir"])
        frames = [
            chineseeeg2.load_event_table(
                event_path, split=sp, subjects=subjects, trainable_only=True
            )
            for sp in splits
        ]
        table = (
            frames[0]
            if len(frames) == 1
            else chineseeeg2.pd.concat(frames, ignore_index=True)
        )
        missing_words = sorted(
            set(table["normalized_word"].astype(str)) - set(embedding_map)
        )
        if missing_words:
            raise FileNotFoundError(
                f"[require-cache] 文本向量缓存缺少所选子集词型：{missing_words[:5]}"
            )
        root = Path(cfg["dataset"]["root"])
        recordings = (
            table[["recording_id", "vhdr_relpath"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
        checked_signals = []
        for recording_id, vhdr_relpath in recordings:
            source_path = root / str(vhdr_relpath)
            located = chineseeeg2.locate_recording_signal(
                source_path, cfg["dataset"], signal_dir
            )
            if located is None:
                raise FileNotFoundError(
                    f"[require-cache] ChineseEEG-2 信号缓存未命中（{recording_id}, {source_path}）。"
                    "请检查 BRAINDECODING_DATA_DIR 软链接路径与 BRAINDECODING_CACHE_DIR 是否已解包。"
                )
            output_path, _ = located
            checked_signals.append(str(output_path))
    else:
        signal_dir = Path(cfg["cache"]["meg_dir"])
        frames = [
            pallier2025.load_event_table(
                event_path, split=sp, subjects=subjects, trainable_only=True
            )
            for sp in splits
        ]
        table = (
            frames[0]
            if len(frames) == 1
            else pallier2025.pd.concat(frames, ignore_index=True)
        )
        missing_words = sorted(
            set(table["标准词"].astype(str)) - set(embedding_map)
        )
        if missing_words:
            raise FileNotFoundError(
                f"[require-cache] 文本向量缓存缺少所选子集词型：{missing_words[:5]}"
            )
        recordings = (
            table[["记录编号", "受试者", "运行编号"]]
            .drop_duplicates()
            .itertuples(index=False)
        )
        checked_signals = []
        for recording_id, subj, run in recordings:
            signal_path, sidecar_path = pallier2025._signal_product_paths(
                signal_dir, str(subj), str(run)
            )
            if not signal_path.is_file() or not sidecar_path.is_file():
                raise FileNotFoundError(
                    f"[require-cache] Pallier2025 信号缓存未命中（{recording_id}: {signal_path}）。"
                    "请先运行 cache 预热或解包缓存归档。"
                )
            checked_signals.append(str(signal_path))

    return {
        "dataset": dataset,
        "subjects": subjects,
        "splits": list(splits),
        "event_table": str(event_path),
        "text_embeddings": str(text_path),
        "signal_dir": str(signal_dir),
        "verified_recording_count": len(checked_signals),
        "verified_event_rows": int(len(table)),
    }


def _load_subset_datasets_and_loaders(cfg: dict):
    """加载所选受试者子集的 train 与 val 数据集及 DataLoader。"""
    dataset = cfg["experiment"]["dataset"]
    task = pallier_task if dataset == "pallier2025" else chinese_task
    read_fn = task.dataset_module.load_event_table
    build_fn = task.build_dataset
    subjects = cfg["dataset"]["subjects"]

    train_table = read_fn(
        cfg["cache"]["event_table"],
        split="train",
        subjects=subjects,
        trainable_only=True,
    )
    val_table = read_fn(
        cfg["cache"]["event_table"],
        split="val",
        subjects=subjects,
        trainable_only=True,
    )
    train_table = limit_rows(train_table, cfg["training"].get("max_train_rows"))
    val_table = limit_rows(val_table, cfg["training"].get("max_val_rows"))

    train_ds = build_fn(cfg, train_table)
    val_ds = build_fn(cfg, val_table)
    grp_col = group_column(cfg)
    train_loader, train_sampler = make_loader(
        train_ds, cfg["training"], shuffle=True, group_column=grp_col
    )
    eval_bs = int(
        cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"])
    )
    val_loader, _ = make_loader(
        val_ds,
        cfg["training"],
        shuffle=False,
        group_column=grp_col,
        batch_size=eval_bs,
    )
    return train_ds, train_loader, train_sampler, val_ds, val_loader


def validate_subset_population(cfg: dict, train_ds, val_ds) -> tuple:
    """根据所选受试者子集动态校验样本与词表契约（替代全量样本数硬编码校验）。"""
    dataset = cfg["experiment"]["dataset"]
    expected_subjects = list(cfg["dataset"]["subjects"])
    if len(train_ds) <= 0 or len(val_ds) <= 0:
        raise ValueError(
            f"所选受试者子集 {expected_subjects} 的训练或验证样本为空："
            f"train={len(train_ds)}, val={len(val_ds)}"
        )

    observed_train_subjects = sorted(set(train_ds.table["subject_id"].astype(str)))
    observed_val_subjects = sorted(set(val_ds.table["subject_id"].astype(str)))
    if not set(observed_train_subjects).issubset(set(expected_subjects)):
        raise ValueError(
            f"训练集包含未选受试者：{observed_train_subjects} vs {expected_subjects}"
        )
    if not set(observed_val_subjects).issubset(set(expected_subjects)):
        raise ValueError(
            f"验证集包含未选受试者：{observed_val_subjects} vs {expected_subjects}"
        )
    if cfg["training"].get("max_train_rows") is None and sorted(expected_subjects) != observed_train_subjects:
        missing = sorted(set(expected_subjects) - set(observed_train_subjects))
        raise ValueError(f"训练集缺少所选受试者数据：{missing}")
    if cfg["training"].get("max_val_rows") is None and sorted(expected_subjects) != observed_val_subjects:
        missing = sorted(set(expected_subjects) - set(observed_val_subjects))
        raise ValueError(f"验证集缺少所选受试者数据：{missing}")

    if set(train_ds.table["event_id"]) & set(val_ds.table["event_id"]):
        raise ValueError("训练集与验证集存在重叠 event_id。")

    settings = load_yaml_with_extends(CONFIG_PATH)
    adapter = settings["adapters"][dataset]
    context_column = adapter["group_column"]
    for ds in (train_ds, val_ds):
        groups = ds.table.groupby(context_column, sort=False)
        if (
            groups["subject_id"].nunique().gt(1).any()
            or groups["recording_id"].nunique().gt(1).any()
        ):
            raise ValueError("上下文分组跨越了受试者或记录边界。")

    vocabulary = sorted(set(train_ds.table["normalized_word"].astype(str)))
    vocab50 = list(
        pallier_task.frozen_vocabulary()
        if dataset == "pallier2025"
        else chinese_task.frozen_vocabulary(cfg["cache"]["event_table"])
    )
    if len(vocab50) != 50 or len(set(vocab50)) != 50:
        raise ValueError("冻结 50 词评测候选词表异常。")

    common_mask = (
        val_ds.table.groupby(context_column, sort=False)
        .cumcount(ascending=False)
        .to_numpy()
        >= 2
    )
    selected_mask = val_ds.table["normalized_word"].isin(vocab50).to_numpy()
    observed = {
        "subset_subjects": expected_subjects,
        "subject_count": len(expected_subjects),
        "train_rows": int(len(train_ds)),
        "val_rows": int(len(val_ds)),
        "train_vocabulary_size": int(len(vocabulary)),
        "all_vocab50_queries": int(selected_mask.sum()),
        "common_vocab50_queries": int((selected_mask & common_mask).sum()),
        "common_rows": int(common_mask.sum()),
        "legacy_word_candidates": int(
            val_ds.table.loc[selected_mask, "normalized_word"].nunique()
        ),
        "legacy_common_candidates": int(
            val_ds.table.loc[selected_mask & common_mask, "normalized_word"].nunique()
        ),
    }
    if observed["all_vocab50_queries"] <= 0:
        raise ValueError("验证集在 50 词评测词表上的有效查询数为 0。")

    sizes = train_ds.table.groupby(context_column, sort=False).size().to_numpy()
    expected_counts = {
        delay: int(np.maximum(sizes - delay, 0).sum()) for delay in range(3)
    }
    return vocabulary, vocab50, expected_counts, observed


def _evaluate_subset_validation(
    model,
    val_loader,
    val_ds,
    vocabulary: list[str],
    vocab50: list[str],
    cfg: dict,
    *,
    revision: bool,
    device: torch.device,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict:
    """在所选受试者子集验证集上计算检索指标（支持 bf16-mixed / fp16 / fp32）。"""
    dataset = cfg["experiment"]["dataset"]
    amp_enabled = bool(cfg["training"].get("amp", True))
    amp_dtype = resolve_amp_dtype(cfg["training"])
    encoded = encode_loader(
        model,
        val_loader,
        device,
        amp=amp_enabled,
        amp_dtype=amp_dtype,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    if len(encoded["event_ids"]) != len(val_ds):
        raise ValueError(
            f"验证集编码事件数不匹配：{len(encoded['event_ids'])} vs {len(val_ds)}"
        )
    predictions = encoded.get("revision_predictions", {0: encoded["predictions"]})
    if not all(torch.isfinite(matrix).all() for matrix in predictions.values()):
        raise FloatingPointError("验证集预测包含非有限数值（NaN/Inf）。")

    top_ks = tuple(cfg["training"].get("top_ks", (1, 10)))
    vocab_name = (
        pallier_task.SELECTION_VOCABULARY_NAME
        if dataset == "pallier2025"
        else "chineseeeg2_littleprince50"
    )
    legacy_metrics = fixed_vocabulary_retrieval_metrics(
        encoded["predictions"],
        encoded["targets"],
        encoded["words"],
        vocab50,
        top_ks=top_ks,
        vocabulary_name=vocab_name,
    )
    if revision and "revision_predictions" in encoded:
        availability = encoded["revision_availability_masks"]
        common_mask = torch.ones(len(encoded["words"]), dtype=torch.bool)
        for mask in availability.values():
            common_mask &= mask.bool()
        legacy_metrics["revision_common_event_count"] = int(common_mask.sum().item())
        for delay, delay_preds in sorted(encoded["revision_predictions"].items()):
            delay_mask = availability[delay].bool()
            delay_words = [
                word for word, keep in zip(encoded["words"], delay_mask.tolist()) if keep
            ]
            if delay_words:
                delay_m = fixed_vocabulary_retrieval_metrics(
                    delay_preds[delay_mask],
                    encoded["targets"][delay_mask],
                    delay_words,
                    vocab50,
                    top_ks=top_ks,
                    vocabulary_name=f"{vocab_name}_delay{delay}",
                )
                legacy_metrics.update(delay_m)
            common_words = [
                word
                for word, keep in zip(encoded["words"], common_mask.tolist())
                if keep
            ]
            if common_words:
                common_m = fixed_vocabulary_retrieval_metrics(
                    delay_preds[common_mask],
                    encoded["targets"][common_mask],
                    common_words,
                    vocab50,
                    top_ks=top_ks,
                    vocabulary_name=f"{vocab_name}_common_delay{delay}",
                )
                legacy_metrics.update(common_m)
                legacy_metrics[f"revision_common_delay{delay}_candidate_count"] = int(
                    common_m["candidate_count"]
                )

    strict_encoded = (
        encoded
        if revision
        else {
            **encoded,
            "revision_predictions": {0: encoded["predictions"]},
            "revision_availability_masks": {
                0: torch.ones(len(encoded["words"]), dtype=torch.bool)
            },
        }
    )
    strict = strict_bank_metrics(strict_encoded, val_ds, vocab50)
    full_vocab = full_vocabulary_metrics(encoded, val_ds, vocabulary)
    result = {
        "legacy": legacy_metrics,
        "strict_vocab50": strict,
        "full_vocabulary": full_vocab,
    }
    if revision and "2" in strict and "0" in strict:
        result["strict_common_k2_minus_k0"] = (
            strict["2"]["common_targets"]["macro_recall_at_10"]
            - strict["0"]["common_targets"]["macro_recall_at_10"]
        )
    return result


def write_subjects_manifest(output_dir: str | Path, subset_spec: dict, *, stage: str, mode: str, seed: int) -> Path:
    """在实验目录中写入或校验 subjects.json，防止不同受试者子集互相覆盖权重。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    subjects_path = out / "subjects.json"
    payload = {
        "schema_version": 1,
        "dataset": subset_spec["dataset"],
        "stage": stage,
        "mode": mode,
        "seed": int(seed),
        "selection_mode": subset_spec["selection_mode"],
        "subset_tag": subset_spec["subset_tag"],
        "is_full_dataset": subset_spec["is_full_dataset"],
        "subject_count": subset_spec["subject_count"],
        "total_canonical_subjects": subset_spec["total_canonical_subjects"],
        "subjects": list(subset_spec["subjects"]),
        "subjects_sha256": subset_spec["subjects_sha256"],
    }
    if subjects_path.is_file():
        existing = json.loads(subjects_path.read_text(encoding="utf-8"))
        if existing.get("subjects") != payload["subjects"] or existing.get("dataset") != payload["dataset"]:
            raise ValueError(
                f"输出目录 {out} 已绑定受试者子集 {existing.get('subjects')}，"
                f"与当前请求的 {payload['subjects']} 不一致；拒绝混用目录。"
            )
    save_json(subjects_path, payload)
    return subjects_path


def run_cache_workflow(
    dataset: str | None = None,
    *,
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    include_test: bool = False,
    force: bool = False,
    config_path: str | Path | None = None,
) -> dict:
    """预热并校验指定数据集与受试者子集的事件表、神经信号与文本嵌入缓存。"""
    ensure_default_environment()
    if config_path is not None and dataset is None:
        raw_cfg = load_yaml_with_extends(resolve_config_path(project_path(config_path)))
        inferred_ds, _ = _infer_dataset_and_stage_from_yaml(raw_cfg)
        datasets = [inferred_ds]
    elif str(dataset or "pallier2025").lower() == "all":
        datasets = list(DATASETS)
    else:
        datasets = [str(dataset or "pallier2025")]
    reports = {}
    for ds_name in datasets:
        cfg, subset_spec = build_workflow_config(
            ds_name,
            stage=None if config_path is not None else "word",
            seed=0,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            mode="cache",
            config_path=config_path if len(datasets) == 1 else None,
        )
        selected_subjects = subset_spec["subjects"]
        splits = ("train", "val", "test") if include_test else ("train", "val")
        event_path = Path(cfg["cache"]["event_table"])

        if ds_name == "chineseeeg2_littleprince":
            if force or not event_path.is_file():
                event_table = chineseeeg2.build_event_table(cfg["dataset"])
                audit = chineseeeg2.audit_event_table(event_table, cfg["dataset"])
                chineseeeg2.save_event_table(event_table, event_path, audit)
            frames = [
                chineseeeg2.load_event_table(
                    event_path,
                    split=sp,
                    subjects=selected_subjects,
                    trainable_only=True,
                )
                for sp in splits
            ]
            subset_table = chineseeeg2.pd.concat(frames, ignore_index=True)
            all_trainable = chineseeeg2.load_event_table(
                event_path, trainable_only=True
            )
            text_words = (
                all_trainable["normalized_word"]
                if len(all_trainable) > 0
                else subset_table["normalized_word"]
            )
            if force and Path(cfg["cache"]["text_embeddings"]).is_file():
                Path(cfg["cache"]["text_embeddings"]).unlink(missing_ok=True)
            text_data.ensure_text_embedding_cache(
                text_words,
                cfg["text_embedding"],
                cfg["cache"]["text_embeddings"],
            )
            chineseeeg2.ensure_recording_signals(
                subset_table,
                cfg["dataset"],
                cfg["cache"]["eeg_dir"],
                force=force,
            )
        else:
            if force or not event_path.is_file():
                from braindecoding.data.dataset_builder import build_dataset

                build_dataset("pallier2025", steps=["events"], force=force)
            frames = [
                pallier2025.load_event_table(
                    event_path,
                    split=sp,
                    subjects=selected_subjects,
                    trainable_only=True,
                )
                for sp in splits
            ]
            subset_table = pallier2025.pd.concat(frames, ignore_index=True)
            all_trainable = pallier2025.load_event_table(
                event_path, trainable_only=True
            )
            text_words = (
                all_trainable["标准词"]
                if len(all_trainable) > 0
                else subset_table["标准词"]
            )
            if force and Path(cfg["cache"]["text_embeddings"]).is_file():
                Path(cfg["cache"]["text_embeddings"]).unlink(missing_ok=True)
            text_data.ensure_text_embedding_cache(
                text_words,
                cfg["text_embedding"],
                cfg["cache"]["text_embeddings"],
            )
            raw_root = Path(cfg["dataset"]["root"])
            recordings = (
                subset_table[["受试者", "运行编号", "记录编号"]]
                .drop_duplicates()
                .itertuples(index=False)
            )
            for subj, run, _ in recordings:
                signal_path, sidecar_path = pallier2025._signal_product_paths(
                    cfg["cache"]["meg_dir"], str(subj), str(run)
                )
                if not force and signal_path.is_file() and sidecar_path.is_file():
                    continue
                src_meg = pallier2025.recording_paths(raw_root, str(subj), str(run))[
                    "meg"
                ]
                pallier2025.materialize_recording_signal(
                    src_meg,
                    cfg["dataset"],
                    cfg["cache"]["meg_dir"],
                    subject=str(subj),
                    run=str(run),
                    force=force,
                )

        verified = verify_cache_ready(cfg, splits=splits)
        cache_out_dir = (
            results_root() / ds_name / "cache" / subset_spec["subset_tag"]
        )
        subjects_file = write_subjects_manifest(
            cache_out_dir, subset_spec, stage="cache", mode="cache", seed=0
        )
        report = {
            "status": "cache_ready",
            "dataset": ds_name,
            "config_path": cfg["workflow"]["config_path"],
            "subset": subset_spec,
            "subjects_file": str(subjects_file),
            "cache_verification": verified,
            "cache_root": str(cache_root()),
            "data_root": str(data_root()),
        }
        save_json(cache_out_dir / "cache_summary.json", report)
        reports[ds_name] = report

    return reports[datasets[0]] if len(datasets) == 1 else {"status": "cache_ready", "datasets": reports}


def run_training_workflow(
    dataset: str | None = None,
    *,
    stage: str | None = None,
    seed: int | None = 0,
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    eval_batch_size: int | None = None,
    learning_rate: float | None = None,
    num_workers: int | None = None,
    precision: str | None = None,
    require_cache: bool = False,
    resume: str | None = None,
    mode: str = "train",
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    word_checkpoint: str | Path | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    config_path: str | Path | None = None,
) -> dict:
    """运行 smoke 冒烟测试或 train 正式训练工作流。"""
    cfg, subset_spec = build_workflow_config(
        dataset,
        stage=stage,
        seed=seed,
        subject=subject,
        num_subjects=num_subjects,
        subjects=subjects,
        epochs=epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        precision=precision,
        mode=mode,
        output_dir=output_dir,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        word_checkpoint=word_checkpoint,
        max_train_rows=max_train_rows,
        max_val_rows=max_val_rows,
        config_path=config_path,
    )
    dataset = subset_spec["dataset"]
    stage = cfg["workflow"]["stage"]
    tc = cfg["training"]
    seed = int(tc["seed"])
    out = Path(tc["output_dir"])
    revision = stage == "revision"
    warm_start_stage = stage in ("context", "revision")

    if require_cache:
        cache_verification = verify_cache_ready(cfg, splits=("train", "val"))
    else:
        try:
            cache_verification = verify_cache_ready(cfg, splits=("train", "val"))
        except FileNotFoundError:
            run_cache_workflow(
                dataset,
                subject=subject,
                num_subjects=num_subjects,
                subjects=subjects,
                include_test=False,
                force=False,
                config_path=config_path,
            )
            cache_verification = verify_cache_ready(cfg, splits=("train", "val"))

    device = choose_device(tc.get("device", "auto"))
    if device.type != "cuda":
        raise RuntimeError("训练与 GPU 冒烟测试要求 CUDA 设备可用。")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    resume_mode = str(resume or "").strip()
    resume_ckpt_path: Path | None = None
    if resume_mode.lower() == "auto":
        candidate = out / "last.ckpt"
        if candidate.is_file():
            resume_ckpt_path = candidate
    elif resume_mode and resume_mode.lower() not in {"none", "false"}:
        candidate = Path(resume_mode).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"指定的恢复检查点不存在：{candidate}")
        resume_ckpt_path = candidate
    elif out.exists() and (out / "training_summary.json").is_file() and not overwrite:
        existing_summary = json.loads(
            (out / "training_summary.json").read_text(encoding="utf-8")
        )
        if existing_summary.get("status") == "completed":
            raise FileExistsError(
                f"输出目录已完成训练：{out}。如需续训请传 `--resume auto`，或指定新子集/新目录。"
            )

    out.mkdir(parents=True, exist_ok=True)
    subjects_file = write_subjects_manifest(
        out, subset_spec, stage=stage, mode=mode, seed=seed
    )
    (out / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    set_seed(seed)
    cpu_threads = min(8, max(2, (os.cpu_count() or 4) // 2))
    torch.set_num_threads(cpu_threads)
    train_ds, train_loader, train_sampler, val_ds, val_loader = (
        _load_subset_datasets_and_loaders(cfg)
    )
    vocabulary, vocab50, expected_counts, population = validate_subset_population(
        cfg, train_ds, val_ds
    )

    model = build_brain_embedding_model(
        train_ds.channel_count,
        train_ds.channel_positions,
        train_ds.subject_count,
        cfg["model"],
    ).to(device)
    loss_module = build_siglip_loss(cfg["loss"]).to(device)
    embedding_hash = file_sha256(cfg["cache"]["text_embeddings"])

    initialization = None
    if warm_start_stage and resume_ckpt_path is None:
        source_ckpt = Path(tc["pretrained_brain_encoder_checkpoint"])
        if source_ckpt.is_file():
            checkpoint = load_checkpoint(source_ckpt)
            encoder_state = {
                key.removeprefix("brain_encoder."): value
                for key, value in checkpoint["model_state"].items()
                if key.startswith("brain_encoder.")
            }
            if not encoder_state:
                raise ValueError(f"预训练词检查点缺少 brain_encoder 权重：{source_ckpt}")
            model.brain_encoder.load_state_dict(encoder_state, strict=True)
            initialization = {
                "path": str(source_ckpt),
                "sha256": file_sha256(source_ckpt),
                "epoch": int(checkpoint.get("epoch", 0)),
            }
            del checkpoint
        elif mode != "smoke":
            raise FileNotFoundError(
                f"{stage} 阶段需要已完成同受试者子集的 word 检查点：{source_ckpt}"
            )

    optimizer = build_adamw_for_modules([model, loss_module], tc)
    scheduler = build_cosine_annealing_scheduler(optimizer, tc["epochs"])
    amp_dtype = resolve_amp_dtype(tc)
    scaler_enabled = (
        bool(tc.get("amp", True))
        and device.type == "cuda"
        and amp_dtype == torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    start_epoch = 0
    history: list[dict] = []
    best = {"legacy50": -float("inf"), "full_vocab": -float("inf")}
    best_rows: dict[str, dict] = {}
    stale_epochs = 0
    total_updates = 0
    resumed_from = None

    if resume_ckpt_path is not None:
        ckpt = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
        if (
            ckpt.get("dataset") != dataset
            or ckpt.get("stage") != stage
            or int(ckpt.get("seed", -1)) != int(seed)
            or list(ckpt.get("subject_ids", ())) != list(subset_spec["subjects"])
        ):
            raise ValueError(
                f"last.ckpt 身份或受试者子集不匹配：{resume_ckpt_path}"
            )
        model.load_state_dict(ckpt["model_state"], strict=True)
        loss_module.load_state_dict(ckpt["loss_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state"])
        if "random_state" in ckpt:
            restore_random_state(ckpt["random_state"])
        start_epoch = int(ckpt.get("completed_epochs", ckpt.get("epoch", 0)))
        history = list(ckpt.get("history", ()))
        best = dict(ckpt.get("best_scores", best))
        best_rows = dict(ckpt.get("best_rows", best_rows))
        stale_epochs = int(ckpt.get("stale_epochs", 0))
        total_updates = int(ckpt.get("optimizer_updates", 0))
        resumed_from = str(resume_ckpt_path)

    torch.cuda.reset_peak_memory_stats(device)
    selection_metric = cfg["evaluation"]["selection_metric"]
    max_epochs = int(tc["epochs"])
    patience = int(tc.get("patience", 6))
    gpu_name = torch.cuda.get_device_name(device)
    gpu_total_gb = torch.cuda.get_device_properties(device).total_memory / (1024.0**3)

    sep = "=" * 96
    thin_sep = "-" * 96
    stage_label = STAGE_DISPLAY_NAMES.get(stage, stage)
    print(
        f"\n{sep}\n"
        f" BrainDecoding Training  |  dataset={dataset}  |  stage={stage} ({stage_label})  |  mode={mode}\n"
        f"{thin_sep}\n"
        f" Config File : {cfg['workflow']['config_path']}\n"
        f" GPU / AMP   : {gpu_name} ({gpu_total_gb:.1f} GB)  |  precision={tc['precision']} (TF32=on)\n"
        f" Subset      : {subset_spec['subset_tag']} ({subset_spec['subject_count']}/{subset_spec['total_canonical_subjects']} subjects)  |  seed={seed}\n"
        f" DataLoader  : train={len(train_ds):,} rows ({len(train_loader)} batches @ bs={tc['batch_size']})"
        f"  |  val={len(val_ds):,} rows (eval_bs={tc.get('eval_batch_size', tc['batch_size'])}, workers={tc.get('num_workers', 0)})\n"
        f" Optimizer   : AdamW (lr={float(tc['learning_rate']):.2e}, wd={float(tc.get('weight_decay', 1e-4)):.1e})"
        f"  |  epochs={max_epochs} (patience={patience})\n"
        f" Output Dir  : {out}",
        flush=True,
    )
    if initialization:
        print(
            f" Warm-Start  : {initialization['path']} (epoch {initialization['epoch']})",
            flush=True,
        )
    if resumed_from:
        print(
            f" Resumed     : {resumed_from} (starting from epoch {start_epoch + 1}/{max_epochs})",
            flush=True,
        )
    print(
        f"{sep}\n"
        f" {'Epoch':^7} | {'Train Loss':^10} | {'Val Top10(50)':^13} | {'Val Top10(Full)':^15} | {'LR':^9} | {'VRAM (Alloc/Res)':^18} | {'Time':^7} | Status\n"
        f"{'-' * 9}+{'-' * 12}+{'-' * 15}+{'-' * 17}+{'-' * 11}+{'-' * 20}+{'-' * 9}+{'-' * 18}",
        flush=True,
    )

    for epoch in range(start_epoch, max_epochs):
        epoch_started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        frozen = warm_start_stage and epoch < int(tc.get("freeze_brain_encoder_epochs", 0))
        pallier_task.set_brain_encoder_trainable(model, not frozen)
        epoch_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, epoch_updates, batches_seen, diagnostics = train_one_epoch(
            model,
            loss_module,
            train_loader,
            optimizer,
            scaler,
            device,
            tc,
            freeze_brain_encoder=frozen,
            return_stats=True,
            return_diagnostics=True,
            epoch_index=epoch,
            show_progress=show_progress,
            progress_desc=f"Epoch {epoch + 1:02d}/{max_epochs:02d} [train]",
        )
        if not np.isfinite(train_loss):
            raise RuntimeError(f"Epoch {epoch + 1} 训练损失出现非有限值：{train_loss}")
        total_updates += int(epoch_updates)

        val_metrics = _evaluate_subset_validation(
            model,
            val_loader,
            val_ds,
            vocabulary,
            vocab50,
            cfg,
            revision=revision,
            device=device,
            show_progress=show_progress,
            progress_desc=f"Epoch {epoch + 1:02d}/{max_epochs:02d} [val]  ",
        )
        scheduler.step()

        peak_alloc_mb = float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        )
        peak_reserved_mb = float(
            torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        )
        epoch_seconds = time.perf_counter() - epoch_started

        legacy_score = float(
            val_metrics["legacy"].get(
                selection_metric,
                val_metrics["legacy"].get("macro_recall_at_10", 0.0),
            )
        )
        full_vocab_score = float(val_metrics["full_vocabulary"]["selection_score"])

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "learning_rate": epoch_lr,
            "seconds": float(epoch_seconds),
            "optimizer_updates": int(total_updates),
            "epoch_optimizer_updates": int(epoch_updates),
            "batches": int(batches_seen),
            "amp_skipped_updates": int(batches_seen - epoch_updates),
            "brain_encoder_frozen": bool(frozen),
            "precision": tc["precision"],
            "amp_dtype": tc["amp_dtype"],
            "peak_memory_allocated_mb": round(peak_alloc_mb, 2),
            "peak_memory_reserved_mb": round(peak_reserved_mb, 2),
            "selection_score_legacy50": legacy_score,
            "selection_score_full_vocab": full_vocab_score,
            "training_diagnostics": diagnostics,
            "validation": val_metrics,
        }
        history.append(row)

        scores = {"legacy50": legacy_score, "full_vocab": full_vocab_score}
        stale_epochs += 1
        improved_tags: list[str] = []
        for selection, score in scores.items():
            if score > best[selection]:
                best[selection] = score
                best_rows[selection] = row
                if selection == "legacy50":
                    stale_epochs = 0
                    improved_tags.append("50")
                else:
                    improved_tags.append("full")
                best_payload = {
                    "dataset": dataset,
                    "stage": stage,
                    "seed": int(seed),
                    "epoch": epoch + 1,
                    "selection": selection,
                    "precision": tc["precision"],
                    "amp_dtype": tc["amp_dtype"],
                    "model_state": cpu_state_dict(model),
                    "loss_state": cpu_state_dict(loss_module),
                    "model_config": cfg["model"],
                    "loss_config": cfg["loss"],
                    "text_embedding_config": cfg["text_embedding"],
                    "embedding_sha256": embedding_hash,
                    "config": cfg,
                    "channel_names": list(train_ds.channel_names),
                    "channel_positions": np.asarray(
                        train_ds.channel_positions
                    ).tolist(),
                    "subject_ids": list(subset_spec["subjects"]),
                    "subset": subset_spec,
                    "metrics": val_metrics,
                    "initialization": initialization,
                    "test_evaluated": False,
                }
                if selection == "legacy50":
                    save_checkpoint(out / "best.ckpt", best_payload)
                    save_checkpoint(out / "best.pt", best_payload)
                else:
                    save_checkpoint(out / "best_full_vocab.pt", best_payload)

        last_payload = {
            "format": "braindecoding_workflow_last_ckpt_v1",
            "dataset": dataset,
            "stage": stage,
            "mode": mode,
            "seed": int(seed),
            "epoch": epoch + 1,
            "completed_epochs": epoch + 1,
            "precision": tc["precision"],
            "amp_dtype": tc["amp_dtype"],
            "subject_ids": list(subset_spec["subjects"]),
            "subset": subset_spec,
            "model_state": cpu_state_dict(model),
            "loss_state": cpu_state_dict(loss_module),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "random_state": capture_random_state(),
            "history": history,
            "best_scores": best,
            "best_rows": best_rows,
            "stale_epochs": stale_epochs,
            "optimizer_updates": total_updates,
            "config": cfg,
            "test_evaluated": False,
        }
        save_checkpoint(out / "last.ckpt", last_payload)

        if improved_tags:
            status_str = f"* best({','.join(improved_tags)})"
        else:
            status_str = f"stale {stale_epochs}/{patience}"
        if frozen:
            status_str += " [frozen]"

        epoch_col = f"{epoch + 1:02d}/{max_epochs:02d}"
        vram_col = f"{peak_alloc_mb:6.0f} / {peak_reserved_mb:5.0f} MB"
        print(
            f" {epoch_col:^7} | {train_loss:10.4f} | {legacy_score * 100.0:11.2f}%  | {full_vocab_score * 100.0:13.2f}%  |"
            f" {epoch_lr:9.2e} | {vram_col:^18} | {epoch_seconds:6.1f}s | {status_str}",
            flush=True,
        )
        if stale_epochs >= patience:
            break

    peak_alloc_mb = float(
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    )
    peak_reserved_mb = float(
        torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
    )
    stop_reason = (
        "patience_exhausted"
        if stale_epochs >= patience
        else "maximum_epochs"
    )
    best50_epoch = best_rows.get("legacy50", {}).get("epoch", "-")
    best_full_epoch = best_rows.get("full_vocab", {}).get("epoch", "-")
    print(
        f"{thin_sep}\n"
        f" Completed ({stop_reason}, epochs={len(history)}/{max_epochs})\n"
        f" Best Val Top10 (50-word)   : {best['legacy50'] * 100.0:6.2f}% (epoch {best50_epoch}) -> {out / 'best.ckpt'}\n"
        f" Best Val Top10 (Full-Vocab): {best['full_vocab'] * 100.0:6.2f}% (epoch {best_full_epoch}) -> {out / 'best_full_vocab.pt'}\n"
        f" Peak GPU Memory            : {peak_alloc_mb:.1f} MB allocated / {peak_reserved_mb:.1f} MB reserved\n"
        f"{sep}\n",
        flush=True,
    )
    summary = {
        "status": "completed",
        "mode": mode,
        "dataset": dataset,
        "stage": stage,
        "seed": int(seed),
        "precision": tc["precision"],
        "amp_dtype": tc["amp_dtype"],
        "subset": subset_spec,
        "subjects_file": str(subjects_file),
        "population": population,
        "cache_verification": cache_verification,
        "resumed_from": resumed_from,
        "completed_epochs": len(history),
        "stop_reason": stop_reason,
        "best_scores": best,
        "best": best_rows,
        "peak_memory_allocated_mb": round(peak_alloc_mb, 2),
        "peak_memory_reserved_mb": round(peak_reserved_mb, 2),
        "checkpoints": {
            "last": str(out / "last.ckpt"),
            "best": str(out / "best.ckpt"),
            "best_pt": str(out / "best.pt"),
        },
        "output_dir": str(out),
        "test_evaluated": False,
        "history": history,
    }
    save_json(out / "training_summary.json", summary)
    save_json(
        out / "metrics.json",
        {
            "dataset": dataset,
            "stage": stage,
            "mode": mode,
            "seed": int(seed),
            "precision": tc["precision"],
            "subset_tag": subset_spec["subset_tag"],
            "subjects": subset_spec["subjects"],
            "completed_epochs": len(history),
            "best_scores": best,
            "peak_memory_allocated_mb": round(peak_alloc_mb, 2),
            "peak_memory_reserved_mb": round(peak_reserved_mb, 2),
            "final_epoch": history[-1] if history else None,
        },
    )
    save_json(
        out / "run_manifest.json",
        {
            "status": "completed",
            "mode": mode,
            "dataset": dataset,
            "stage": stage,
            "seed": int(seed),
            "precision": tc["precision"],
            "subset": subset_spec,
            "population": population,
            "completed_epochs": len(history),
            "stop_reason": stop_reason,
            "peak_memory_allocated_mb": round(peak_alloc_mb, 2),
            "peak_memory_reserved_mb": round(peak_reserved_mb, 2),
            "test_evaluated": False,
        },
    )
    save_json(
        out / "status.json",
        {
            "status": "completed",
            "mode": mode,
            "dataset": dataset,
            "stage": stage,
            "seed": int(seed),
            "subset_tag": subset_spec["subset_tag"],
            "completed_epochs": len(history),
            "peak_memory_allocated_mb": round(peak_alloc_mb, 2),
            "peak_memory_reserved_mb": round(peak_reserved_mb, 2),
            "output_dir": str(out),
        },
    )
    del (
        model,
        loss_module,
        optimizer,
        scheduler,
        scaler,
        train_ds,
        val_ds,
        train_loader,
        val_loader,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def _compute_encoded_evaluation_metrics(
    encoded: dict,
    eval_ds,
    vocabulary: list[str],
    vocab50: list[str],
    cfg: dict,
    *,
    stage: str,
    split: str,
) -> dict:
    """基于单次前向编码结果计算多词表规模检索、分受试者拆解与滚动修订各延迟指标。"""
    from braindecoding.results import load_vocabulary_assets

    dataset = cfg["experiment"]["dataset"]
    revision = stage == "revision" and "revision_predictions" in encoded
    manifest_dir = PROJECT_ROOT / "data/manifests" / (
        "pallier2025" if dataset == "pallier2025" else "chineseeeg2"
    )
    manifests = load_vocabulary_assets(manifest_dir)

    primary_preds = (
        encoded["revision_predictions"][max(encoded["revision_predictions"])]
        if revision
        else encoded["predictions"]
    )
    if revision:
        max_delay = max(encoded["revision_predictions"])
        avail_mask = encoded["revision_availability_masks"][max_delay].bool()
    else:
        avail_mask = torch.ones(len(encoded["words"]), dtype=torch.bool)

    eval_preds = primary_preds[avail_mask]
    eval_targets = encoded["targets"][avail_mask]
    eval_words = [
        w for w, keep in zip(encoded["words"], avail_mask.tolist()) if keep
    ]

    vocab_Sweep: dict[str, dict] = {}
    for size in (20, 50, 100, 150):
        cand_words = list(manifests[size]["vocabulary"])
        m = fixed_vocabulary_retrieval_metrics(
            eval_preds,
            eval_targets,
            eval_words,
            cand_words,
            top_ks=(1, 5, 10),
            vocabulary_name=f"N{size}",
        )
        vocab_Sweep[f"N{size}"] = {
            "label": f"Top-{size} Vocab",
            "candidate_count": int(size),
            "supported_candidates": int(m.get("candidate_count", size)),
            "query_count": int(m.get("query_count", 0)),
            "macro_top1": float(m.get("macro_recall_at_1", 0.0)),
            "macro_top5": float(m.get("macro_recall_at_5", 0.0)),
            "macro_top10": float(m.get("macro_recall_at_10", 0.0)),
            "micro_top1": float(m.get("micro_recall_at_1", 0.0)),
            "micro_top5": float(m.get("micro_recall_at_5", 0.0)),
            "micro_top10": float(m.get("micro_recall_at_10", 0.0)),
            "mrr": float(m.get("mean_reciprocal_rank", 0.0)),
            "median_rank": float(m.get("median_rank", float(size) / 2.0)),
            "chance_top1": float(1.0 / size),
            "chance_top10": float(min(10.0, float(size)) / size),
        }

    strict_encoded = (
        encoded
        if revision
        else {
            **encoded,
            "revision_predictions": {0: encoded["predictions"]},
            "revision_availability_masks": {
                0: torch.ones(len(encoded["words"]), dtype=torch.bool)
            },
        }
    )
    strict50 = strict_bank_metrics(strict_encoded, eval_ds, vocab50)
    full_vocab = full_vocabulary_metrics(encoded, eval_ds, vocabulary)

    full_horizon_key = "2" if (revision and "2" in full_vocab.get("horizons", {})) else "0"
    full_block = full_vocab.get("horizons", {}).get(full_horizon_key, {}).get(
        "common_targets" if revision else "all_targets", {}
    )
    full_cand_count = max(1, int(len(vocabulary)))
    vocab_Sweep["full_vocab"] = {
        "label": f"Full Vocab ({full_cand_count})",
        "candidate_count": full_cand_count,
        "supported_candidates": full_cand_count,
        "query_count": int(full_block.get("query_count", len(eval_words))),
        "macro_top1": float(full_block.get("word_macro_recall_at_1", 0.0)),
        "macro_top5": float(full_block.get("word_macro_recall_at_5", 0.0)),
        "macro_top10": float(
            full_block.get(
                "subject_equal_macro_recall_at_10",
                full_vocab.get("selection_score", 0.0),
            )
        ),
        "micro_top1": float(full_block.get("micro_recall_at_1", 0.0)),
        "micro_top5": float(full_block.get("micro_recall_at_5", 0.0)),
        "micro_top10": float(full_block.get("micro_recall_at_10", 0.0)),
        "mrr": float(full_block.get("mean_reciprocal_rank", 0.0)),
        "median_rank": float(full_block.get("median_rank", full_cand_count / 2.0)),
        "chance_top1": float(1.0 / full_cand_count),
        "chance_top10": float(min(10.0, float(full_cand_count)) / full_cand_count),
    }

    subject_col = eval_ds.table["subject_id"].astype(str).to_numpy()
    avail_np = avail_mask.cpu().numpy()
    per_subject: list[dict] = []
    for subj in cfg["dataset"]["subjects"]:
        subj_mask_np = (subject_col == str(subj)) & avail_np
        if not subj_mask_np.any():
            continue
        subj_mask_t = torch.from_numpy(subj_mask_np)
        subj_words = [
            w for w, keep in zip(encoded["words"], subj_mask_np.tolist()) if keep
        ]
        m50 = fixed_vocabulary_retrieval_metrics(
            primary_preds[subj_mask_t],
            encoded["targets"][subj_mask_t],
            subj_words,
            vocab50,
            top_ks=(1, 5, 10),
            vocabulary_name=f"{subj}_N50",
        )
        per_subject.append(
            {
                "subject_id": str(subj),
                "total_events": int(subj_mask_np.sum()),
                "vocab50_queries": int(m50.get("query_count", 0)),
                "macro_top10": float(m50.get("macro_recall_at_10", 0.0)),
                "micro_top10": float(m50.get("micro_recall_at_10", 0.0)),
                "macro_top1": float(m50.get("macro_recall_at_1", 0.0)),
                "micro_top1": float(m50.get("micro_recall_at_1", 0.0)),
                "mrr": float(m50.get("mean_reciprocal_rank", 0.0)),
                "median_rank": float(m50.get("median_rank", 25.0)),
            }
        )

    revision_horizons: dict[str, dict] = {}
    if revision:
        for delay in sorted(encoded["revision_predictions"]):
            d_key = str(delay)
            s50_common = strict50.get(d_key, {}).get("common_targets", {})
            s50_all = strict50.get(d_key, {}).get("all_targets", {})
            fv_common = (
                full_vocab.get("horizons", {})
                .get(d_key, {})
                .get("common_targets", {})
            )
            revision_horizons[f"K{delay}"] = {
                "delay": int(delay),
                "label": f"K{delay} (+{delay}w)",
                "common_vocab50_macro_top10": float(
                    s50_common.get("macro_recall_at_10", 0.0)
                ),
                "common_vocab50_micro_top10": float(
                    s50_common.get("micro_recall_at_10", 0.0)
                ),
                "common_vocab50_macro_top1": float(
                    s50_common.get("macro_recall_at_1", 0.0)
                ),
                "all_vocab50_macro_top10": float(
                    s50_all.get("macro_recall_at_10", 0.0)
                ),
                "common_full_vocab_macro_top10": float(
                    fv_common.get("subject_equal_macro_recall_at_10", 0.0)
                ),
                "common_full_vocab_micro_top10": float(
                    fv_common.get("micro_recall_at_10", 0.0)
                ),
            }

    return {
        "split": split,
        "total_events": int(len(eval_ds)),
        "evaluated_events": int(avail_mask.sum().item()),
        "vocabulary_sweep": vocab_Sweep,
        "per_subject": per_subject,
        "revision_horizons": revision_horizons,
        "strict_vocab50": strict50,
        "full_vocabulary": full_vocab,
    }


def _print_evaluation_report(eval_result: dict) -> None:
    """在终端或 Notebook 输出格式化的评估指标报表。"""
    sep = "=" * 96
    thin_sep = "-" * 96
    dataset = eval_result["dataset"]
    stage = eval_result["stage"]
    stage_label = STAGE_DISPLAY_NAMES.get(stage, stage)
    split = eval_result["split"].upper()
    subset = eval_result["subset"]
    ckpt_info = eval_result["checkpoint"]
    sweep = eval_result["metrics"]["vocabulary_sweep"]

    print(
        f"\n{sep}\n"
        f" BrainDecoding Evaluation  |  dataset={dataset}  |  split={split}  |  {stage_label}\n"
        f"{thin_sep}\n"
        f" Checkpoint  : {ckpt_info['path']} (epoch {ckpt_info['epoch']})\n"
        f" Subset      : {subset['subset_tag']} ({subset['subject_count']}/{subset['total_canonical_subjects']} subjects)  |  seed={eval_result['seed']}\n"
        f" Split Rows  : {eval_result['metrics']['total_events']:,} total events ({eval_result['metrics']['evaluated_events']:,} evaluated)\n"
        f"{sep}\n"
        f" {'Vocabulary Scale':<22} | {'Queries':>8} | {'Macro Top10':>11} | {'Micro Top10':>11} | {'Macro Top1':>10} | {'MRR':>8} | {'Med Rank':>8} | {'Chance Top10':>12}\n"
        f"{'-' * 24}+{'-' * 10}+{'-' * 13}+{'-' * 13}+{'-' * 12}+{'-' * 10}+{'-' * 10}+{'-' * 14}",
        flush=True,
    )
    for key in ("N20", "N50", "N100", "N150", "full_vocab"):
        if key not in sweep:
            continue
        row = sweep[key]
        print(
            f" {row['label']:<22} | {row['query_count']:8,d} | {row['macro_top10'] * 100.0:10.2f}% | {row['micro_top10'] * 100.0:10.2f}% |"
            f" {row['macro_top1'] * 100.0:9.2f}% | {row['mrr'] * 100.0:7.2f}% | {row['median_rank']:8.1f} | {row['chance_top10'] * 100.0:11.2f}%",
            flush=True,
        )

    horizons = eval_result["metrics"].get("revision_horizons") or {}
    if horizons:
        print(
            f"{thin_sep}\n"
            f" Rolling Revision Horizons (Common Targets across K0 / K1 / K2):\n"
            f" {'Horizon':<16} | {'50-Word Macro Top10':>20} | {'50-Word Micro Top10':>20} | {'50-Word Macro Top1':>19} | {'Full-Vocab Top10':>17}\n"
            f"{'-' * 18}+{'-' * 22}+{'-' * 22}+{'-' * 21}+{'-' * 19}",
            flush=True,
        )
        for h_key in ("K0", "K1", "K2"):
            if h_key not in horizons:
                continue
            h = horizons[h_key]
            print(
                f" {h['label']:<16} | {h['common_vocab50_macro_top10'] * 100.0:19.2f}% | {h['common_vocab50_micro_top10'] * 100.0:19.2f}% |"
                f" {h['common_vocab50_macro_top1'] * 100.0:18.2f}% | {h['common_full_vocab_macro_top10'] * 100.0:16.2f}%",
                flush=True,
            )

    per_subject = eval_result["metrics"].get("per_subject") or []
    if len(per_subject) > 1:
        print(
            f"{thin_sep}\n"
            f" Per-Subject 50-Word Retrieval Breakdown ({split}):\n"
            f" {'Subject':<12} | {'Events':>8} | {'50w Queries':>11} | {'Macro Top10':>12} | {'Micro Top10':>12} | {'Macro Top1':>11} | {'Median Rank':>11}\n"
            f"{'-' * 14}+{'-' * 10}+{'-' * 13}+{'-' * 14}+{'-' * 14}+{'-' * 13}+{'-' * 13}",
            flush=True,
        )
        for s in per_subject:
            print(
                f" {s['subject_id']:<12} | {s['total_events']:8,d} | {s['vocab50_queries']:11,d} |"
                f" {s['macro_top10'] * 100.0:11.2f}% | {s['micro_top10'] * 100.0:11.2f}% |"
                f" {s['macro_top1'] * 100.0:10.2f}% | {s['median_rank']:11.1f}",
                flush=True,
            )
    print(f"{sep}\n", flush=True)


def plot_evaluation_figures(
    eval_result: dict,
    figures_dir: str | Path,
) -> dict[str, str]:
    """生成单阶段 2x2 综合评估看板图（PNG 与 PDF），返回生成的文件路径字典。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = eval_result["dataset"]
    stage = eval_result["stage"]
    stage_title = STAGE_DISPLAY_NAMES.get(stage, stage)
    split = eval_result["split"].upper()
    subset_tag = eval_result["subset"]["subset_tag"]
    seed = eval_result["seed"]
    metrics = eval_result["metrics"]
    sweep = metrics["vocabulary_sweep"]
    per_subject = metrics.get("per_subject") or []
    horizons = metrics.get("revision_horizons") or {}
    history = eval_result.get("training_history") or []

    primary_color = "#1f77b4" if dataset == "pallier2025" else "#6a3d9a"
    secondary_color = "#ff7f0e"
    accent_color = "#2ca02c"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5), dpi=160)
    fig.suptitle(
        f"BrainDecoding {split} Evaluation Dashboard  |  {dataset}  |  {stage_title}\n"
        f"Subset: {subset_tag} (seed={seed}, checkpoint epoch={eval_result['checkpoint']['epoch']})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Retrieval Accuracy Across Vocabulary Sizes
    ax_a = axes[0, 0]
    keys = [k for k in ("N20", "N50", "N100", "N150", "full_vocab") if k in sweep]
    x = np.arange(len(keys))
    width = 0.26
    macro_top10 = [sweep[k]["macro_top10"] * 100.0 for k in keys]
    micro_top10 = [sweep[k]["micro_top10"] * 100.0 for k in keys]
    macro_top1 = [sweep[k]["macro_top1"] * 100.0 for k in keys]
    chance_top10 = [sweep[k]["chance_top10"] * 100.0 for k in keys]
    labels = [sweep[k]["label"].replace(" (", "\n(") for k in keys]

    bars1 = ax_a.bar(x - width, macro_top10, width, label="Macro Top-10 (%)", color=primary_color, alpha=0.9)
    bars2 = ax_a.bar(x, micro_top10, width, label="Micro Top-10 (%)", color=secondary_color, alpha=0.85)
    bars3 = ax_a.bar(x + width, macro_top1, width, label="Macro Top-1 (%)", color=accent_color, alpha=0.85)
    ax_a.plot(x, chance_top10, "k--o", linewidth=1.4, markersize=4, label="Random Chance (Top-10)")

    for bar in bars1:
        h = bar.get_height()
        ax_a.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    ax_a.set_title(f"(A) Retrieval Accuracy by Vocabulary Size ({split})", fontsize=11, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=9)
    ax_a.set_ylabel("Accuracy (%)", fontsize=10)
    ax_a.set_ylim(0, max(max(macro_top10 + micro_top10 + chance_top10, default=50.0) * 1.18, 20.0))
    ax_a.grid(axis="y", linestyle=":", alpha=0.5)
    ax_a.legend(loc="upper right", fontsize=8.5)

    # Panel B: Per-Subject 50-Word Decoding Accuracy
    ax_b = axes[0, 1]
    if per_subject:
        subj_ids = [s["subject_id"] for s in per_subject]
        s_top10 = [s["macro_top10"] * 100.0 for s in per_subject]
        s_top1 = [s["macro_top1"] * 100.0 for s in per_subject]
        sx = np.arange(len(subj_ids))
        sw = 0.36
        b_s10 = ax_b.bar(sx - sw / 2, s_top10, sw, label="50-Word Macro Top-10 (%)", color=primary_color, alpha=0.88)
        ax_b.bar(sx + sw / 2, s_top1, sw, label="50-Word Macro Top-1 (%)", color=accent_color, alpha=0.85)
        mean_top10 = float(np.mean(s_top10))
        ax_b.axhline(mean_top10, color=primary_color, linestyle="-.", linewidth=1.3, label=f"Mean Top-10 ({mean_top10:.1f}%)")
        ax_b.axhline(20.0, color="gray", linestyle="--", linewidth=1.2, label="Chance Top-10 (20.0%)")
        for bar in b_s10:
            h = bar.get_height()
            ax_b.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        ax_b.set_xticks(sx)
        ax_b.set_xticklabels(subj_ids, rotation=25 if len(subj_ids) > 6 else 0, fontsize=9)
        ax_b.set_ylim(0, max(max(s_top10, default=30.0) * 1.2, 35.0))
        ax_b.legend(loc="upper right", fontsize=8.5)
    else:
        ax_b.text(0.5, 0.5, "No per-subject data available", ha="center", va="center", transform=ax_b.transAxes)
    ax_b.set_title(f"(B) Per-Subject 50-Word Retrieval Accuracy ({split})", fontsize=11, fontweight="bold")
    ax_b.set_ylabel("Accuracy (%)", fontsize=10)
    ax_b.grid(axis="y", linestyle=":", alpha=0.5)

    # Panel C: Training Convergence Curve (if training_summary.json history exists)
    ax_c = axes[1, 0]
    if history:
        ep_nums = [int(r["epoch"]) for r in history]
        tr_loss = [float(r["train_loss"]) for r in history]
        val_50 = [float(r.get("selection_score_legacy50", 0.0)) * 100.0 for r in history]
        val_full = [float(r.get("selection_score_full_vocab", 0.0)) * 100.0 for r in history]

        ax_c_right = ax_c.twinx()
        l1 = ax_c.plot(ep_nums, tr_loss, "-o", color="#d62728", linewidth=1.8, markersize=4, label="Train Loss")
        l2 = ax_c_right.plot(ep_nums, val_50, "-s", color=primary_color, linewidth=2.0, markersize=4.5, label="Val 50-Word Top-10 (%)")
        l3 = ax_c_right.plot(ep_nums, val_full, "--^", color=accent_color, linewidth=1.6, markersize=4, label="Val Full-Vocab Top-10 (%)")

        best_ep = int(eval_result["checkpoint"].get("epoch", 0))
        if best_ep in ep_nums:
            ax_c.axvline(best_ep, color="black", linestyle=":", linewidth=1.3, label=f"Best Epoch ({best_ep})")

        ax_c.set_xlabel("Epoch", fontsize=10)
        ax_c.set_ylabel("Train Loss (SigLIP)", color="#d62728", fontsize=10)
        ax_c_right.set_ylabel("Validation Top-10 Accuracy (%)", color=primary_color, fontsize=10)
        lines = l1 + l2 + l3
        labels_c = [line.get_label() for line in lines]
        ax_c.legend(lines, labels_c, loc="center right", fontsize=8.5)
        ax_c.grid(True, linestyle=":", alpha=0.5)
    else:
        ax_c.text(
            0.5,
            0.5,
            "Training history not found in output directory\n(Checkpoint evaluated standalone)",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax_c.transAxes,
        )
    ax_c.set_title("(C) Training Loss & Validation Accuracy Progression", fontsize=11, fontweight="bold")

    # Panel D: Rolling Revision Horizons (for stage 2.2) OR Detailed Metric Summary (for stage 1 / 2.1)
    ax_d = axes[1, 1]
    if horizons and all(k in horizons for k in ("K0", "K1", "K2")):
        h_keys = ["K0", "K1", "K2"]
        hx = np.arange(len(h_keys))
        hw = 0.34
        h_50 = [horizons[k]["common_vocab50_macro_top10"] * 100.0 for k in h_keys]
        h_full = [horizons[k]["common_full_vocab_macro_top10"] * 100.0 for k in h_keys]
        h_labels = [horizons[k]["label"] for k in h_keys]

        b_h1 = ax_d.bar(hx - hw / 2, h_50, hw, label="50-Word Common Macro Top-10 (%)", color=primary_color, alpha=0.9)
        b_h2 = ax_d.bar(hx + hw / 2, h_full, hw, label="Full-Vocab Common Top-10 (%)", color=secondary_color, alpha=0.88)
        for bar in list(b_h1) + list(b_h2):
            h = bar.get_height()
            ax_d.annotate(
                f"{h:.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )
        delta_50 = h_50[-1] - h_50[0]
        ax_d.set_title(
            f"(D) Stage 2.2 Rolling Revision Gain: K2 - K0 = {delta_50:+.2f}% (50-Word)",
            fontsize=11,
            fontweight="bold",
        )
        ax_d.set_xticks(hx)
        ax_d.set_xticklabels(h_labels, fontsize=9.5)
        ax_d.set_ylabel("Macro Top-10 Accuracy (%)", fontsize=10)
        ax_d.set_ylim(0, max(max(h_50 + h_full, default=30.0) * 1.22, 25.0))
        ax_d.legend(loc="upper left", fontsize=8.5)
        ax_d.grid(axis="y", linestyle=":", alpha=0.5)
    else:
        n50 = sweep.get("N50", {})
        fv = sweep.get("full_vocab", {})
        metric_names = ["Macro Top-10", "Micro Top-10", "Macro Top-5", "Macro Top-1", "MRR"]
        vals_50 = [
            n50.get("macro_top10", 0.0) * 100.0,
            n50.get("micro_top10", 0.0) * 100.0,
            n50.get("macro_top5", 0.0) * 100.0,
            n50.get("macro_top1", 0.0) * 100.0,
            n50.get("mrr", 0.0) * 100.0,
        ]
        vals_fv = [
            fv.get("macro_top10", 0.0) * 100.0,
            fv.get("micro_top10", 0.0) * 100.0,
            fv.get("macro_top5", 0.0) * 100.0,
            fv.get("macro_top1", 0.0) * 100.0,
            fv.get("mrr", 0.0) * 100.0,
        ]
        mx = np.arange(len(metric_names))
        mw = 0.35
        bm1 = ax_d.bar(mx - mw / 2, vals_50, mw, label="50-Word Candidate Pool", color=primary_color, alpha=0.9)
        bm2 = ax_d.bar(mx + mw / 2, vals_fv, mw, label="Full Training Vocabulary", color=secondary_color, alpha=0.88)
        for bar in list(bm1) + list(bm2):
            h = bar.get_height()
            ax_d.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax_d.set_title(f"(D) Detailed Retrieval Profile: 50-Word vs Full Vocabulary ({split})", fontsize=11, fontweight="bold")
        ax_d.set_xticks(mx)
        ax_d.set_xticklabels(metric_names, fontsize=9)
        ax_d.set_ylabel("Score (%)", fontsize=10)
        ax_d.set_ylim(0, max(max(vals_50 + vals_fv, default=30.0) * 1.22, 25.0))
        ax_d.legend(loc="upper right", fontsize=8.5)
        ax_d.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png_path = out_dir / f"evaluation_{eval_result['split']}_overview.png"
    pdf_path = out_dir / f"evaluation_{eval_result['split']}_overview.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "overview_png": str(png_path),
        "overview_pdf": str(pdf_path),
    }


def plot_stage_comparison_figure(
    stage_results: dict[str, dict],
    figures_dir: str | Path,
    *,
    split: str = "val",
) -> dict[str, str]:
    """生成阶段 1 (word)、阶段 2.1 (context) 与阶段 2.2 (revision) 的横向对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ordered_stages = [s for s in WORKFLOW_STAGES if s in stage_results]
    if not ordered_stages:
        return {}

    first_res = stage_results[ordered_stages[0]]
    dataset = first_res["dataset"]
    subset_tag = first_res["subset"]["subset_tag"]
    seed = first_res["seed"]
    split_upper = split.upper()

    stage_colors = {
        "word": "#4e79a7",
        "context": "#f28e2b",
        "revision": "#59a14f",
    }
    short_labels = {
        "word": "Stage 1: Word\n(CNN Only)",
        "context": "Stage 2.1: Context\n(CNN + 16L Full Window)",
        "revision": "Stage 2.2: Revision\n(CNN + 16L Local K2)",
    }

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.5, 5.8), dpi=160)
    fig.suptitle(
        f"BrainDecoding Multi-Stage Comparison ({split_upper})  |  dataset={dataset}  |  subset={subset_tag} (seed={seed})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    # Left panel: 50-Word & Full-Vocab Core Metrics by Stage
    metric_defs = [
        ("N50", "macro_top10", "50-Word\nMacro Top-10"),
        ("N50", "micro_top10", "50-Word\nMicro Top-10"),
        ("N50", "macro_top1", "50-Word\nMacro Top-1"),
        ("full_vocab", "macro_top10", "Full-Vocab\nMacro Top-10"),
    ]
    mx = np.arange(len(metric_defs))
    bar_w = 0.75 / max(1, len(ordered_stages))

    for idx, st in enumerate(ordered_stages):
        res = stage_results[st]
        sw = res["metrics"]["vocabulary_sweep"]
        vals = [
            float(sw.get(v_key, {}).get(m_key, 0.0)) * 100.0
            for v_key, m_key, _ in metric_defs
        ]
        offset = (idx - (len(ordered_stages) - 1) / 2.0) * bar_w
        bars = ax_left.bar(
            mx + offset,
            vals,
            bar_w,
            label=STAGE_DISPLAY_NAMES.get(st, st),
            color=stage_colors.get(st, "#333333"),
            alpha=0.9,
        )
        for bar in bars:
            h = bar.get_height()
            ax_left.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax_left.set_title(f"(A) Stage Comparison on Core Retrieval Metrics ({split_upper})", fontsize=11, fontweight="bold")
    ax_left.set_xticks(mx)
    ax_left.set_xticklabels([label for _, _, label in metric_defs], fontsize=9.5)
    ax_left.set_ylabel("Accuracy (%)", fontsize=10)
    ax_left.set_ylim(0, max(40.0, ax_left.get_ylim()[1] * 1.18))
    ax_left.grid(axis="y", linestyle=":", alpha=0.5)
    ax_left.legend(loc="upper right", fontsize=8.5)

    # Right panel: Scaling Curve Across Vocabulary Sizes (N=20, 50, 100, 150, Full Vocab)
    scale_keys = ["N20", "N50", "N100", "N150", "full_vocab"]
    sx = np.arange(len(scale_keys))
    scale_labels = ["N=20", "N=50", "N=100", "N=150", "Full Vocab"]

    for st in ordered_stages:
        res = stage_results[st]
        sw = res["metrics"]["vocabulary_sweep"]
        y_vals = [float(sw.get(k, {}).get("macro_top10", 0.0)) * 100.0 for k in scale_keys]
        ax_right.plot(
            sx,
            y_vals,
            "-o",
            linewidth=2.2,
            markersize=6,
            color=stage_colors.get(st, "#333333"),
            label=short_labels.get(st, st).replace("\n", " "),
        )
        for x_i, y_v in zip(sx, y_vals):
            ax_right.annotate(
                f"{y_v:.1f}%",
                xy=(x_i, y_v),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    first_sw = first_res["metrics"]["vocabulary_sweep"]
    chance_curve = [float(first_sw.get(k, {}).get("chance_top10", 0.0)) * 100.0 for k in scale_keys]
    ax_right.plot(sx, chance_curve, "k--", linewidth=1.3, label="Random Chance (Top-10)")

    ax_right.set_title(f"(B) Macro Top-10 Scaling Across Vocabulary Sizes ({split_upper})", fontsize=11, fontweight="bold")
    ax_right.set_xticks(sx)
    ax_right.set_xticklabels(scale_labels, fontsize=9.5)
    ax_right.set_ylabel("Macro Top-10 Accuracy (%)", fontsize=10)
    ax_right.set_ylim(0, max(60.0, ax_right.get_ylim()[1] * 1.15))
    ax_right.grid(True, linestyle=":", alpha=0.5)
    ax_right.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = out_dir / f"stage_comparison_{split}.png"
    pdf_path = out_dir / f"stage_comparison_{split}.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "stage_comparison_png": str(png_path),
        "stage_comparison_pdf": str(pdf_path),
    }


def _evaluate_single_stage(
    dataset: str | None = None,
    *,
    stage: str | None = None,
    seed: int | None = 0,
    split: str = "val",
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    precision: str | None = None,
    require_cache: bool = False,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    config_path: str | Path | None = None,
    plot: bool = True,
    show_progress: bool = True,
    max_eval_rows: int | None = None,
) -> dict:
    """对单个训练阶段（word / context / revision）执行验证集或测试集评估并生成图表。"""
    if split not in {"val", "test"}:
        raise ValueError(f"评估 split 只能是 'val' 或 'test'，收到：{split!r}")

    cfg, subset_spec = build_workflow_config(
        dataset,
        stage=stage,
        seed=seed,
        subject=subject,
        num_subjects=num_subjects,
        subjects=subjects,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        precision=precision,
        mode="train",
        output_dir=output_dir,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        config_path=config_path,
    )
    dataset = subset_spec["dataset"]
    stage = cfg["workflow"]["stage"]
    resolved_exp_tag = cfg["workflow"].get("exp_tag", "")
    resolved_word_exp_tag = cfg["workflow"].get("word_exp_tag", resolved_exp_tag)
    tc = cfg["training"]
    seed = int(tc["seed"])
    out = Path(tc["output_dir"])

    if checkpoint_path is not None:
        ckpt_file = Path(checkpoint_path).resolve()
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"指定的评估检查点不存在：{ckpt_file}")
        if output_dir is None:
            out = ckpt_file.parent
    else:
        ckpt_file = out / "best.ckpt"
        if not ckpt_file.is_file():
            ckpt_file = out / "best.pt"
        if not ckpt_file.is_file():
            smoke_dir = workflow_output_directory(
                dataset,
                stage,
                subset_spec["subset_tag"],
                seed,
                mode="smoke",
                exp_tag=resolved_exp_tag,
            )
            smoke_ckpt = smoke_dir / "best.ckpt"
            if not smoke_ckpt.is_file():
                smoke_ckpt = smoke_dir / "best.pt"
            if smoke_ckpt.is_file():
                ckpt_file = smoke_ckpt
                if output_dir is None:
                    out = smoke_dir
        if not ckpt_file.is_file():
            raise FileNotFoundError(
                f"未找到阶段 {stage} ({STAGE_DISPLAY_NAMES.get(stage, stage)}) 的训练检查点：{out / 'best.ckpt'}。"
                f"请先完成该阶段的训练，或通过 --checkpoint 显式指定检查点路径。"
            )

    if require_cache:
        cache_verification = verify_cache_ready(cfg, splits=(split,))
    else:
        try:
            cache_verification = verify_cache_ready(cfg, splits=(split,))
        except FileNotFoundError:
            run_cache_workflow(
                dataset,
                subject=subject,
                num_subjects=num_subjects,
                subjects=subjects,
                include_test=(split == "test"),
                force=False,
                config_path=config_path,
            )
            cache_verification = verify_cache_ready(cfg, splits=(split,))

    device = choose_device(tc.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    set_seed(seed)
    task = pallier_task if dataset == "pallier2025" else chinese_task
    read_fn = task.dataset_module.load_event_table
    build_fn = task.build_dataset
    selected_subjects = cfg["dataset"]["subjects"]

    train_table = read_fn(
        cfg["cache"]["event_table"],
        split="train",
        subjects=selected_subjects,
        trainable_only=True,
    )
    eval_table = read_fn(
        cfg["cache"]["event_table"],
        split=split,
        subjects=selected_subjects,
        trainable_only=True,
    )
    if max_eval_rows is not None:
        eval_table = limit_rows(eval_table, int(max_eval_rows))
    if len(eval_table) <= 0:
        raise ValueError(f"数据集 {dataset} 在划分 {split!r} 与受试者 {selected_subjects} 上的事件数为 0。")

    eval_ds = build_fn(cfg, eval_table)
    grp_col = group_column(cfg)
    eval_bs = int(tc.get("eval_batch_size", tc["batch_size"]))
    eval_loader, _ = make_loader(
        eval_ds,
        tc,
        shuffle=False,
        group_column=grp_col,
        batch_size=eval_bs,
    )

    train_word_col = (
        "normalized_word" if "normalized_word" in train_table.columns else "标准词"
    )
    vocabulary = sorted(set(train_table[train_word_col].astype(str)))
    vocab50 = list(
        pallier_task.frozen_vocabulary()
        if dataset == "pallier2025"
        else chinese_task.frozen_vocabulary(cfg["cache"]["event_table"])
    )

    checkpoint = load_checkpoint(ckpt_file, map_location="cpu")
    model_cfg = checkpoint.get("model_config", cfg["model"])
    ckpt_subjects = list(checkpoint.get("subject_ids", selected_subjects))
    subject_count = max(len(ckpt_subjects), eval_ds.subject_count)

    model = build_brain_embedding_model(
        eval_ds.channel_count,
        eval_ds.channel_positions,
        subject_count,
        model_cfg,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    amp_enabled = bool(tc.get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(tc)
    encoded = encode_loader(
        model,
        eval_loader,
        device,
        amp=amp_enabled,
        amp_dtype=amp_dtype,
        show_progress=show_progress,
        progress_desc=f"Eval [{stage} | {split}]",
    )

    metrics = _compute_encoded_evaluation_metrics(
        encoded,
        eval_ds,
        vocabulary,
        vocab50,
        cfg,
        stage=stage,
        split=split,
    )

    training_history = []
    summary_path = out / "training_summary.json"
    if summary_path.is_file():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            training_history = list(summary_data.get("history", ()))
        except Exception:
            training_history = []

    out.mkdir(parents=True, exist_ok=True)
    eval_result = {
        "status": "evaluation_completed",
        "dataset": dataset,
        "stage": stage,
        "exp_tag": resolved_exp_tag,
        "word_exp_tag": resolved_word_exp_tag,
        "stage_display_name": STAGE_DISPLAY_NAMES.get(stage, stage),
        "split": split,
        "seed": seed,
        "precision": tc["precision"],
        "subset": subset_spec,
        "checkpoint": {
            "path": str(ckpt_file),
            "sha256": file_sha256(ckpt_file),
            "epoch": int(checkpoint.get("epoch", 0)),
            "selection": str(checkpoint.get("selection", "legacy50")),
        },
        "cache_verification": cache_verification,
        "metrics": metrics,
        "training_history": training_history,
        "output_dir": str(out),
        "figures": {},
    }

    _print_evaluation_report(eval_result)

    if plot:
        figs = plot_evaluation_figures(eval_result, out / "figures")
        eval_result["figures"].update(figs)

    eval_json_path = out / f"evaluation_{split}.json"
    save_json(eval_json_path, eval_result)
    eval_result["evaluation_json"] = str(eval_json_path)

    del model, eval_ds, eval_loader, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return eval_result


def run_evaluation_workflow(
    dataset: str | None = None,
    *,
    stage: str | None = None,
    seed: int | None = 0,
    split: str = "val",
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    eval_batch_size: int | None = None,
    num_workers: int | None = None,
    precision: str | None = None,
    require_cache: bool = False,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    config_path: str | Path | None = None,
    all_stages: bool = False,
    plot: bool = True,
    show_progress: bool = True,
    max_eval_rows: int | None = None,
) -> dict:
    """运行验证集 (val) 或测试集 (test) 评估与可视化工作流（支持单阶段或全阶段对比）。"""
    if split not in {"val", "test"}:
        raise ValueError(f"评估 split 只能是 'val' 或 'test'，收到：{split!r}")

    if not all_stages:
        single_res = _evaluate_single_stage(
            dataset,
            stage=stage,
            seed=seed,
            split=split,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            precision=precision,
            require_cache=require_cache,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            exp_tag=exp_tag,
            word_exp_tag=word_exp_tag,
            config_path=config_path,
            plot=plot,
            show_progress=show_progress,
            max_eval_rows=max_eval_rows,
        )
        if plot:
            ds_name = single_res["dataset"]
            sub_tag = single_res["subset"]["subset_tag"]
            sd = single_res["seed"]
            active_tag = single_res.get("exp_tag", "")
            active_word_tag = single_res.get("word_exp_tag", active_tag)
            discovered: dict[str, dict] = {single_res["stage"]: single_res}
            for other_st in WORKFLOW_STAGES:
                if other_st == single_res["stage"]:
                    continue
                target_tag = (
                    active_word_tag
                    if other_st == "word" and active_word_tag != active_tag
                    else active_tag
                )
                other_dir = workflow_output_directory(
                    ds_name,
                    other_st,
                    sub_tag,
                    sd,
                    mode="train",
                    exp_tag=target_tag,
                    ignore_implicit_tag=(other_st == "word" and target_tag == ""),
                )
                other_eval_file = other_dir / f"evaluation_{split}.json"
                if not other_eval_file.is_file():
                    smoke_other = workflow_output_directory(
                        ds_name,
                        other_st,
                        sub_tag,
                        sd,
                        mode="smoke",
                        exp_tag=target_tag,
                        ignore_implicit_tag=(other_st == "word" and target_tag == ""),
                    ) / f"evaluation_{split}.json"
                    if smoke_other.is_file():
                        other_eval_file = smoke_other
                if not other_eval_file.is_file() and target_tag and other_st == "word":
                    fallback_other = workflow_output_directory(
                        ds_name, other_st, sub_tag, sd, mode="train", ignore_implicit_tag=True
                    ) / f"evaluation_{split}.json"
                    if fallback_other.is_file():
                        other_eval_file = fallback_other
                if other_eval_file.is_file():
                    try:
                        discovered[other_st] = json.loads(
                            other_eval_file.read_text(encoding="utf-8")
                        )
                    except Exception:
                        pass
            if len(discovered) >= 1:
                comp_figs = plot_stage_comparison_figure(
                    discovered,
                    Path(single_res["output_dir"]) / "figures",
                    split=split,
                )
                single_res["figures"].update(comp_figs)
                save_json(Path(single_res["evaluation_json"]), single_res)
        return single_res

    base_cfg, subset_spec = build_workflow_config(
        dataset,
        stage="word",
        seed=seed,
        subject=subject,
        num_subjects=num_subjects,
        subjects=subjects,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        config_path=config_path,
    )
    ds_name = subset_spec["dataset"]
    sub_tag = subset_spec["subset_tag"]
    sd = int(base_cfg["training"]["seed"])
    base_root, resolved_exp_tag = _resolve_results_root_and_exp_tag(
        base_cfg["workflow"].get("exp_tag", exp_tag)
    )
    has_explicit_word_tag, resolved_word_exp_tag = _resolve_word_exp_tag(word_exp_tag)

    available_stages: list[tuple[str, str, bool]] = []
    for st in WORKFLOW_STAGES:
        candidate_tags: list[tuple[str, bool]] = []
        if st == "word" and has_explicit_word_tag:
            candidate_tags.append((resolved_word_exp_tag, resolved_word_exp_tag == ""))
        else:
            candidate_tags.append((resolved_exp_tag, False))
            if st == "word" and resolved_exp_tag:
                candidate_tags.append(("", True))

        found = False
        for tag_cand, ignore_imp in candidate_tags:
            for mode_candidate in ("train", "smoke"):
                cand_dir = workflow_output_directory(
                    ds_name,
                    st,
                    sub_tag,
                    sd,
                    mode=mode_candidate,
                    exp_tag=tag_cand,
                    ignore_implicit_tag=ignore_imp,
                )
                if (cand_dir / "best.ckpt").is_file() or (cand_dir / "best.pt").is_file():
                    available_stages.append((st, tag_cand, ignore_imp))
                    found = True
                    break
            if found:
                break

    if not available_stages:
        expected_dir = workflow_output_directory(
            ds_name, "word", sub_tag, sd, mode="train", exp_tag=resolved_exp_tag
        )
        raise FileNotFoundError(
            f"未找到数据集 {ds_name} (subset={sub_tag}, seed={sd}) 任何已训练阶段的检查点（如 {expected_dir / 'best.ckpt'}）。"
        )

    stage_results: dict[str, dict] = {}
    for st, st_tag, st_ignore_imp in available_stages:
        stage_out_dir = (
            workflow_output_directory(
                ds_name,
                st,
                sub_tag,
                sd,
                mode="train",
                exp_tag="",
                ignore_implicit_tag=True,
            )
            if st_ignore_imp
            else None
        )
        stage_results[st] = _evaluate_single_stage(
            ds_name,
            stage=st,
            seed=sd,
            split=split,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            precision=precision,
            require_cache=require_cache,
            output_dir=stage_out_dir,
            exp_tag=st_tag,
            word_exp_tag=word_exp_tag,
            config_path=None,
            plot=plot,
            show_progress=show_progress,
            max_eval_rows=max_eval_rows,
        )

    dataset_eval_root = base_root / ds_name
    if resolved_exp_tag:
        dataset_eval_root = dataset_eval_root / resolved_exp_tag
    comp_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (
            dataset_eval_root
            / f"evaluation_{split}"
            / sub_tag
            / f"seed-{sd:03d}"
        ).resolve()
    )
    comp_dir.mkdir(parents=True, exist_ok=True)
    comp_figs = {}
    if plot:
        comp_figs = plot_stage_comparison_figure(
            stage_results, comp_dir / "figures", split=split
        )
        for st_res in stage_results.values():
            st_figs = plot_stage_comparison_figure(
                stage_results, Path(st_res["output_dir"]) / "figures", split=split
            )
            st_res["figures"].update(st_figs)
            save_json(Path(st_res["evaluation_json"]), st_res)

    multi_summary = {
        "status": "multi_stage_evaluation_completed",
        "dataset": ds_name,
        "exp_tag": resolved_exp_tag,
        "word_exp_tag": resolved_word_exp_tag if has_explicit_word_tag else resolved_exp_tag,
        "split": split,
        "seed": sd,
        "subset": subset_spec,
        "evaluated_stages": [st for st, _, _ in available_stages],
        "stages": stage_results,
        "output_dir": str(comp_dir),
        "figures": comp_figs,
    }
    summary_file = comp_dir / f"multi_stage_evaluation_{split}.json"
    save_json(summary_file, multi_summary)
    multi_summary["evaluation_json"] = str(summary_file)
    return multi_summary

