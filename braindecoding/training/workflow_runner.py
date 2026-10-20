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
from braindecoding.evaluation.retrieval_metrics import (
    fixed_vocabulary_retrieval_metrics,
    mean_by_target,
    retrieval_ranks,
)
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


def _resolve_window_settings(
    active_yaml: dict | None,
    cfg: dict,
    *,
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
    explicit_yaml: dict | None = None,
) -> dict:
    """解析神经信号窗口长度（秒）与起点偏移（秒），优先级：CLI/显式参数 > 环境变量 > YAML 配置 > 默认值(1.0s, 0.0s)。"""
    ds_cfg = cfg.setdefault("dataset", {})
    wf_yaml = (
        active_yaml.get("workflow")
        if isinstance(active_yaml, dict) and isinstance(active_yaml.get("workflow"), dict)
        else {}
    )
    ds_yaml = (
        active_yaml.get("dataset")
        if isinstance(active_yaml, dict) and isinstance(active_yaml.get("dataset"), dict)
        else {}
    )
    exp_wf_yaml = (
        explicit_yaml.get("workflow")
        if isinstance(explicit_yaml, dict)
        and isinstance(explicit_yaml.get("workflow"), dict)
        else {}
    )
    exp_ds_yaml = (
        explicit_yaml.get("dataset")
        if isinstance(explicit_yaml, dict)
        and isinstance(explicit_yaml.get("dataset"), dict)
        else {}
    )

    env_win = os.environ.get("BRAINDECODING_WINDOW_SECONDS", "").strip()
    env_off = os.environ.get("BRAINDECODING_WINDOW_START_OFFSET_SECONDS", "").strip()
    has_env_win = bool(env_win and env_win.lower() not in {"none", "default"})
    has_env_off = bool(env_off and env_off.lower() not in {"none", "default"})

    yaml_win = (
        wf_yaml.get("window_seconds")
        if wf_yaml.get("window_seconds") is not None
        else (
            ds_yaml.get("window_seconds")
            if ds_yaml.get("window_seconds") is not None
            else ds_cfg.get("window_seconds")
        )
    )
    yaml_off = (
        wf_yaml.get("window_start_offset_seconds")
        if wf_yaml.get("window_start_offset_seconds") is not None
        else (
            ds_yaml.get("window_start_offset_seconds")
            if ds_yaml.get("window_start_offset_seconds") is not None
            else ds_cfg.get("window_start_offset_seconds")
        )
    )

    if window_seconds is not None:
        resolved_win = float(window_seconds)
    elif has_env_win:
        resolved_win = float(env_win)
    elif yaml_win is not None:
        resolved_win = float(yaml_win)
    else:
        resolved_win = 1.0

    if window_start_offset_seconds is not None:
        resolved_off = float(window_start_offset_seconds)
    elif has_env_off:
        resolved_off = float(env_off)
    elif yaml_off is not None:
        resolved_off = float(yaml_off)
    else:
        resolved_off = 0.0

    sampling_rate_hz = float(ds_cfg.get("target_sampling_rate_hz", 50.0))
    window_samples = int(round(resolved_win * sampling_rate_hz))
    if resolved_win <= 0.0 or window_samples <= 0:
        raise ValueError(
            f"window_seconds 必须为正数且对应至少 1 个采样点，当前 window_seconds={resolved_win} "
            f"(window_samples={window_samples} @ {sampling_rate_hz}Hz)。"
        )

    exp_yaml_win = (
        exp_wf_yaml.get("window_seconds")
        if exp_wf_yaml.get("window_seconds") is not None
        else exp_ds_yaml.get("window_seconds")
    )
    exp_yaml_off = (
        exp_wf_yaml.get("window_start_offset_seconds")
        if exp_wf_yaml.get("window_start_offset_seconds") is not None
        else exp_ds_yaml.get("window_start_offset_seconds")
    )
    explicit_override = (
        window_seconds is not None
        or window_start_offset_seconds is not None
        or has_env_win
        or has_env_off
        or (exp_yaml_win is not None and abs(float(exp_yaml_win) - 1.0) > 1e-9)
        or (exp_yaml_off is not None and abs(float(exp_yaml_off) - 0.0) > 1e-9)
    )

    ds_cfg["window_seconds"] = resolved_win
    ds_cfg["window_start_offset_seconds"] = resolved_off
    return {
        "window_seconds": resolved_win,
        "window_start_offset_seconds": resolved_off,
        "window_samples": window_samples,
        "sampling_rate_hz": sampling_rate_hz,
        "explicit_override": bool(explicit_override),
    }


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
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
    mode: str = "train",
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    word_checkpoint: str | Path | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    config_path: str | Path | None = None,
    siamese_debiased: bool | None = None,
    eta_train: float | None = None,
    lambda_null: float | None = None,
    shifts_per_step: int | None = None,
    preextract_frozen_cnn: bool | None = None,
) -> tuple[dict, dict]:
    """构建支持灵活受试者子集、可切换窗口长度、自定义 YAML 配置与环境变量路径解耦的训练配置。"""
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
    default_config_path = resolve_config_path(project_path(default_source_rel))
    active_config_path = (
        explicit_cfg_path
        if explicit_cfg_path is not None
        else default_config_path
    )
    active_yaml = (
        explicit_yaml
        if explicit_yaml is not None
        else load_yaml_with_extends(active_config_path)
    )
    is_decoupled_overlay = explicit_yaml is not None and (
        "dataset" not in explicit_yaml or "cache" not in explicit_yaml
    )
    if is_decoupled_overlay:
        base_stage_yaml = load_yaml_with_extends(default_config_path)
        cfg = loader(default_config_path)
        for overlay_key in (
            "control",
            "evaluation",
            "workflow",
            "experiment",
            "training",
            "dataset",
            "siamese_debiased",
        ):
            if isinstance(explicit_yaml.get(overlay_key), dict):
                cfg.setdefault(overlay_key, {})
                if isinstance(cfg[overlay_key], dict):
                    cfg[overlay_key].update(
                        copy.deepcopy(explicit_yaml[overlay_key])
                    )
                else:
                    cfg[overlay_key] = copy.deepcopy(explicit_yaml[overlay_key])
        merged_active_yaml = copy.deepcopy(base_stage_yaml)
        for k, v in explicit_yaml.items():
            if isinstance(v, dict) and isinstance(merged_active_yaml.get(k), dict):
                merged_active_yaml[k].update(copy.deepcopy(v))
            else:
                merged_active_yaml[k] = copy.deepcopy(v)
        active_yaml = merged_active_yaml
    else:
        cfg = loader(active_config_path)
        if isinstance(active_yaml, dict) and isinstance(
            active_yaml.get("control"), dict
        ):
            cfg["control"] = copy.deepcopy(active_yaml["control"])
        if isinstance(active_yaml, dict) and isinstance(
            active_yaml.get("siamese_debiased"), dict
        ):
            cfg["siamese_debiased"] = copy.deepcopy(
                active_yaml["siamese_debiased"]
            )
    yaml_tc = dict(cfg.get("training", {}))

    window_info = _resolve_window_settings(
        active_yaml,
        cfg,
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        explicit_yaml=explicit_yaml,
    )

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
        else str(
            (active_yaml.get("workflow") or {}).get("precision")
            if isinstance(active_yaml, dict)
            and isinstance(active_yaml.get("workflow"), dict)
            and (active_yaml.get("workflow") or {}).get("precision")
            else yaml_tc.get("precision", "bf16-mixed")
        )
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

    from braindecoding.training.siamese_debiased_training import (
        resolve_siamese_debiased_config,
    )

    siamese_spec = resolve_siamese_debiased_config(
        cfg,
        siamese_debiased=siamese_debiased,
        eta_train=eta_train,
        lambda_null=lambda_null,
        shifts_per_step=shifts_per_step,
        preextract_frozen_cnn=preextract_frozen_cnn,
    )
    if (
        siamese_spec["enabled"] or siamese_spec["preextract_frozen_cnn"]
    ) and stage not in ("context", "revision"):
        raise ValueError(
            "孪生共模抵消与冻结 CNN 预提取仅适用于 context (Stage 2.1) 或 revision (Stage 2.2) 阶段。"
        )
    if (
        siamese_spec["enabled"] or siamese_spec["preextract_frozen_cnn"]
    ) and stage in ("context", "revision"):
        tc["freeze_brain_encoder_epochs"] = max(
            int(tc.get("freeze_brain_encoder_epochs", 0)),
            int(tc["epochs"]),
        )
    if (
        siamese_spec["enabled"]
        or siamese_spec["preextract_frozen_cnn"]
        or "siamese_debiased" in cfg
    ):
        cfg["siamese_debiased"] = siamese_spec

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
        "window_seconds": window_info["window_seconds"],
        "window_start_offset_seconds": window_info["window_start_offset_seconds"],
        "window_samples": window_info["window_samples"],
        "sampling_rate_hz": window_info["sampling_rate_hz"],
        "window_explicitly_overridden": window_info["explicit_override"],
        "siamese_debiased": siamese_spec,
        "subset": subset_spec,
        "environment": {
            "BRAINDECODING_DATA_DIR": str(data_root()),
            "BRAINDECODING_CACHE_DIR": str(cache_root()),
            "BRAINDECODING_RESULTS_DIR": str(results_root()),
            "BRAINDECODING_EXP_TAG": resolved_exp_tag,
            "BRAINDECODING_WORD_EXP_TAG": resolved_word_exp_tag if has_explicit_word_tag else "",
            "BRAINDECODING_WINDOW_SECONDS": str(window_info["window_seconds"]),
            "BRAINDECODING_WINDOW_START_OFFSET_SECONDS": str(window_info["window_start_offset_seconds"]),
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


def _score_full_vocabulary_population(
    ranks: np.ndarray,
    words: np.ndarray,
    subjects: np.ndarray,
    covered: np.ndarray,
    candidate_count: int,
) -> dict:
    """计算指定查询子集（all_targets 或 common_targets）的全词表检索指标。"""
    n_queries = int(len(ranks))
    cand_count = max(1, int(candidate_count))
    if n_queries == 0:
        return {
            "query_count": 0,
            "candidate_count": cand_count,
            "target_count": 0,
            "oov_queries": 0,
            "in_vocabulary_rate": 0.0,
            "subject_equal_macro_recall_at_1": 0.0,
            "subject_equal_macro_recall_at_5": 0.0,
            "subject_equal_macro_recall_at_10": 0.0,
            "subject_macro_top10": 0.0,
            "word_macro_recall_at_1": 0.0,
            "word_macro_recall_at_5": 0.0,
            "word_macro_recall_at_10": 0.0,
            "macro_recall_at_1": 0.0,
            "macro_recall_at_5": 0.0,
            "macro_recall_at_10": 0.0,
            "micro_recall_at_1": 0.0,
            "micro_recall_at_5": 0.0,
            "micro_recall_at_10": 0.0,
            "mean_reciprocal_rank": 0.0,
            "median_rank": float(cand_count) / 2.0,
        }

    unique_subjects = sorted(set(subjects.tolist()))
    metrics: dict[str, int | float] = {
        "query_count": n_queries,
        "candidate_count": cand_count,
        "target_count": int(len(set(words.tolist()))),
        "oov_queries": int((~covered).sum()),
        "in_vocabulary_rate": float(np.mean(covered)),
    }
    for k in (1, 5, 10):
        hits = (ranks <= float(k)).astype(np.float64)
        micro_k = float(np.mean(hits))
        word_macro_k = float(mean_by_target(hits, words))
        subj_macro_k = (
            float(
                np.mean(
                    [
                        float(np.mean(hits[subjects == subj]))
                        for subj in unique_subjects
                        if np.any(subjects == subj)
                    ]
                )
            )
            if unique_subjects
            else micro_k
        )
        metrics[f"micro_recall_at_{k}"] = micro_k
        metrics[f"word_macro_recall_at_{k}"] = word_macro_k
        metrics[f"macro_recall_at_{k}"] = word_macro_k
        metrics[f"subject_equal_macro_recall_at_{k}"] = subj_macro_k
    metrics["subject_macro_top10"] = float(
        metrics["subject_equal_macro_recall_at_10"]
    )
    reciprocal = np.where(np.isfinite(ranks) & (ranks > 0.0), 1.0 / ranks, 0.0)
    metrics["mean_reciprocal_rank"] = float(np.mean(reciprocal))
    raw_median = float(np.median(ranks))
    if np.isfinite(raw_median):
        metrics["median_rank"] = raw_median
    else:
        bounded_ranks = np.where(np.isfinite(ranks), ranks, float(cand_count))
        metrics["median_rank"] = float(np.median(bounded_ranks))
    return metrics


def _compute_full_vocabulary_horizon_details(
    strict_encoded: dict,
    eval_ds,
    vocabulary: list[str],
) -> dict[str, dict]:
    """按各延迟视野（0 / 1 / 2）计算全训练词表的 all_targets 与 common_targets 完整检索指标。"""
    vocab_list = [str(w) for w in vocabulary]
    cand_count = max(1, len(vocab_list))
    words = np.asarray([str(w) for w in strict_encoded["words"]], dtype=object)
    bank = (
        np.stack(
            [np.asarray(eval_ds.embedding_map[w], dtype=np.float32) for w in vocab_list]
        )
        if vocab_list
        else np.zeros((0, 1), dtype=np.float32)
    )
    covered = (
        np.isin(words, vocab_list)
        if vocab_list
        else np.zeros(len(words), dtype=bool)
    )
    subject_map = dict(
        zip(
            eval_ds.table["event_id"].astype(str),
            eval_ds.table["subject_id"].astype(str),
        )
    )
    fallback_subjects = eval_ds.table["subject_id"].astype(str).to_numpy()
    subjects = np.asarray(
        [
            subject_map.get(
                str(eid),
                str(fallback_subjects[idx])
                if idx < len(fallback_subjects)
                else "unknown",
            )
            for idx, eid in enumerate(strict_encoded["event_ids"])
        ],
        dtype=object,
    )
    rev_preds = strict_encoded["revision_predictions"]
    rev_masks = strict_encoded["revision_availability_masks"]
    common_mask = np.logical_and.reduce(
        [m.detach().cpu().numpy().astype(bool) for m in rev_masks.values()]
    )

    horizons_detail: dict[str, dict] = {}
    for delay in sorted(rev_preds):
        pred_tensor = rev_preds[delay]
        avail_mask = rev_masks[delay].detach().cpu().numpy().astype(bool)
        ranks = np.full(len(words), np.inf, dtype=np.float64)
        if covered.any() and len(vocab_list) > 0:
            ranks[covered] = retrieval_ranks(
                pred_tensor[covered],
                words[covered],
                bank,
                vocab_list,
            )
        all_mask = avail_mask if avail_mask.any() else np.ones(len(words), dtype=bool)
        com_mask = common_mask if common_mask.any() else all_mask
        all_stats = _score_full_vocabulary_population(
            ranks[all_mask],
            words[all_mask],
            subjects[all_mask],
            covered[all_mask],
            cand_count,
        )
        common_stats = _score_full_vocabulary_population(
            ranks[com_mask],
            words[com_mask],
            subjects[com_mask],
            covered[com_mask],
            cand_count,
        )
        per_subj: dict[str, dict] = {}
        for subj in sorted(set(subjects.tolist())):
            s_mask = com_mask & (subjects == subj)
            if not s_mask.any():
                s_mask = all_mask & (subjects == subj)
            if not s_mask.any():
                continue
            r_s = ranks[s_mask]
            per_subj[str(subj)] = {
                "n": int(s_mask.sum()),
                "top1": float(np.mean(r_s <= 1.0)),
                "top10": float(np.mean(r_s <= 10.0)),
            }
        horizons_detail[str(delay)] = {
            "subject_macro_top10": float(
                common_stats["subject_equal_macro_recall_at_10"]
            ),
            "per_subject": per_subj,
            "all_targets": all_stats,
            "common_targets": common_stats,
        }
    return horizons_detail


def _build_enriched_full_vocabulary_metrics(
    strict_encoded: dict,
    eval_ds,
    vocabulary: list[str],
) -> dict:
    """构建同时兼容历史 k0/k1/k2 与数值键 0/1/2 且包含 all_targets/common_targets 的全词表指标字典。"""
    horizon_details = _compute_full_vocabulary_horizon_details(
        strict_encoded, eval_ds, vocabulary
    )
    try:
        base_full = full_vocabulary_metrics(strict_encoded, eval_ds, vocabulary)
    except ValueError:
        selection_vals = [
            float(d["subject_macro_top10"]) for d in horizon_details.values()
        ]
        common_mask = np.logical_and.reduce(
            [
                m.detach().cpu().numpy().astype(bool)
                for m in strict_encoded["revision_availability_masks"].values()
            ]
        )
        words = np.asarray([str(w) for w in strict_encoded["words"]], dtype=object)
        covered = np.isin(words, [str(w) for w in vocabulary])
        base_full = {
            "selection_score": float(np.mean(selection_vals))
            if selection_vals
            else 0.0,
            "horizons": {},
            "candidate_count": int(len(vocabulary)),
            "all_reference_words_retained": True,
            "reference_count": int(len(words)),
            "common_count": int(common_mask.sum()),
            "reference_in_vocabulary": float(np.mean(covered))
            if len(words) > 0
            else 0.0,
        }

    horizons_dict = dict(base_full.get("horizons") or {})
    for delay_str, detail in horizon_details.items():
        k_key = f"k{delay_str}"
        existing_k = dict(horizons_dict.get(k_key) or {})
        merged = {**detail, **existing_k}
        merged["all_targets"] = detail["all_targets"]
        merged["common_targets"] = detail["common_targets"]
        if "subject_macro_top10" in existing_k:
            merged["common_targets"] = {
                **detail["common_targets"],
                "subject_equal_macro_recall_at_10": float(
                    existing_k["subject_macro_top10"]
                ),
                "subject_macro_top10": float(existing_k["subject_macro_top10"]),
            }
        horizons_dict[k_key] = merged
        horizons_dict[str(delay_str)] = merged
    base_full["horizons"] = horizons_dict
    return base_full


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
    preextracted_cache: dict | None = None,
) -> dict:
    """在所选受试者子集验证集上计算检索指标（支持 bf16-mixed / fp16 / fp32）。"""
    dataset = cfg["experiment"]["dataset"]
    amp_enabled = bool(cfg["training"].get("amp", True))
    amp_dtype = resolve_amp_dtype(cfg["training"])
    if preextracted_cache is not None:
        from braindecoding.training.siamese_debiased_training import (
            encode_from_preextracted_cnn,
        )

        encoded = encode_from_preextracted_cnn(
            model,
            val_loader.batch_sampler,
            preextracted_cache["val_clean_z"],
            preextracted_cache["val_meta"],
            device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    else:
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
    full_vocab = _build_enriched_full_vocabulary_metrics(
        strict_encoded, val_ds, vocabulary
    )
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
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
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
    siamese_debiased: bool | None = None,
    eta_train: float | None = None,
    lambda_null: float | None = None,
    shifts_per_step: int | None = None,
    preextract_frozen_cnn: bool | None = None,
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
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        mode=mode,
        output_dir=output_dir,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        word_checkpoint=word_checkpoint,
        max_train_rows=max_train_rows,
        max_val_rows=max_val_rows,
        config_path=config_path,
        siamese_debiased=siamese_debiased,
        eta_train=eta_train,
        lambda_null=lambda_null,
        shifts_per_step=shifts_per_step,
        preextract_frozen_cnn=preextract_frozen_cnn,
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
    win_sec = float(getattr(train_ds, "window_seconds", cfg["dataset"].get("window_seconds", 1.0)))
    win_samp = int(getattr(train_ds, "window_samples", round(win_sec * 50.0)))
    sr_hz = float(getattr(train_ds, "sampling_rate_hz", cfg["dataset"].get("target_sampling_rate_hz", 50.0)))
    win_off = float(cfg["dataset"].get("window_start_offset_seconds", 0.0))
    print(
        f"\n{sep}\n"
        f" BrainDecoding Training  |  dataset={dataset}  |  stage={stage} ({stage_label})  |  mode={mode}\n"
        f"{thin_sep}\n"
        f" Config File : {cfg['workflow']['config_path']}\n"
        f" GPU / AMP   : {gpu_name} ({gpu_total_gb:.1f} GB)  |  precision={tc['precision']} (TF32=on)\n"
        f" Subset      : {subset_spec['subset_tag']} ({subset_spec['subject_count']}/{subset_spec['total_canonical_subjects']} subjects)  |  seed={seed}\n"
        f" Window      : {win_sec:.2f}s ({win_samp} samples @ {sr_hz:.0f}Hz, start_offset={win_off:+.2f}s)\n"
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

    siamese_spec = dict(cfg["workflow"].get("siamese_debiased") or {})
    siamese_enabled = bool(siamese_spec.get("enabled", False))
    all_epochs_frozen = (
        warm_start_stage
        and int(tc.get("freeze_brain_encoder_epochs", 0)) >= max_epochs
    )
    use_preextracted_cnn = bool(
        all_epochs_frozen
        and (siamese_enabled or bool(siamese_spec.get("preextract_frozen_cnn", False)))
    )
    if siamese_enabled:
        print(
            f" Siamese Deb : enabled (eta_train={float(siamese_spec.get('eta_train', 0.8)):.2f}, "
            f"lambda_null={float(siamese_spec.get('lambda_null', 10.0)):.2f}, "
            f"M={int(siamese_spec.get('shifts_per_step', 2))}/{len(siamese_spec.get('shift_seconds', ()))} shifts, "
            f"preextract_cnn={use_preextracted_cnn})",
            flush=True,
        )
    elif use_preextracted_cnn:
        print(
            " Pre-Extract : Frozen Stage-1 CNN feature cache enabled (Clean 0s)",
            flush=True,
        )

    preextracted_cache: dict | None = None
    centered_vocab_bank: torch.Tensor | None = None
    if use_preextracted_cnn and start_epoch < max_epochs:
        from braindecoding.training.siamese_debiased_training import (
            build_centered_vocabulary_bank,
            preextract_frozen_cnn_features,
            train_one_epoch_siamese_debiased,
        )

        pallier_task.set_brain_encoder_trainable(model, False)
        word_ckpt_sha = ""
        if initialization and initialization.get("sha256"):
            word_ckpt_sha = str(initialization["sha256"])
        else:
            src_ckpt_str = str(tc.get("pretrained_brain_encoder_checkpoint", ""))
            if src_ckpt_str and Path(src_ckpt_str).is_file():
                word_ckpt_sha = str(file_sha256(src_ckpt_str) or "")
        preextracted_cache = preextract_frozen_cnn_features(
            model,
            train_ds,
            val_ds,
            dataset=dataset,
            subset_tag=subset_spec["subset_tag"],
            word_ckpt_sha256=word_ckpt_sha,
            window_seconds=win_sec,
            window_start_offset_seconds=win_off,
            shift_seconds=(
                siamese_spec.get("shift_seconds", ()) if siamese_enabled else ()
            ),
            device=device,
            batch_size=int(tc.get("eval_batch_size", tc["batch_size"])),
            amp=bool(tc.get("amp", True)),
            amp_dtype=amp_dtype,
            cache_to_disk=bool(
                siamese_spec.get("cache_preextracted_to_disk", True)
                and tc.get("max_train_rows") is None
                and tc.get("max_val_rows") is None
            ),
            show_progress=show_progress,
        )
        if siamese_enabled and float(siamese_spec.get("lambda_null", 0.0)) > 0.0:
            centered_vocab_bank = build_centered_vocabulary_bank(
                train_ds, vocabulary, device
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

        if preextracted_cache is not None and frozen:
            train_loss, epoch_updates, batches_seen, diagnostics = (
                train_one_epoch_siamese_debiased(
                    model,
                    loss_module,
                    train_sampler,
                    preextracted_cache,
                    optimizer,
                    scaler,
                    device,
                    tc,
                    siamese_spec,
                    centered_vocab_bank=centered_vocab_bank,
                    epoch_index=epoch,
                    show_progress=show_progress,
                    progress_desc=(
                        f"Epoch {epoch + 1:02d}/{max_epochs:02d} [siamese]"
                        if siamese_enabled
                        else f"Epoch {epoch + 1:02d}/{max_epochs:02d} [cached]"
                    ),
                )
            )
        else:
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
            preextracted_cache=(
                preextracted_cache if (preextracted_cache is not None and frozen) else None
            ),
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
                    "window_seconds": win_sec,
                    "window_samples": win_samp,
                    "window_start_offset_seconds": win_off,
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
            "window_seconds": win_sec,
            "window_samples": win_samp,
            "window_start_offset_seconds": win_off,
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
        if siamese_enabled and isinstance(diagnostics, dict) and "shift_null_loss" in diagnostics:
            status_str += (
                f" [sig={float(diagnostics.get('siglip_loss', train_loss)):.3f},"
                f"null={float(diagnostics.get('shift_null_loss', 0.0)):.4f}]"
            )

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
        "window_seconds": win_sec,
        "window_samples": win_samp,
        "window_start_offset_seconds": win_off,
        "siamese_debiased": siamese_spec,
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
            "window_seconds": win_sec,
            "window_samples": win_samp,
            "window_start_offset_seconds": win_off,
            "siamese_debiased": siamese_spec,
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
            "window_seconds": win_sec,
            "window_samples": win_samp,
            "window_start_offset_seconds": win_off,
            "siamese_debiased": siamese_spec,
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
        preextracted_cache,
        centered_vocab_bank,
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
    full_vocab = _build_enriched_full_vocabulary_metrics(
        strict_encoded, eval_ds, vocabulary
    )

    full_horizon_key = (
        str(max(strict_encoded["revision_predictions"]))
        if revision
        else "0"
    )
    full_horizon_entry = (
        full_vocab.get("horizons", {}).get(full_horizon_key)
        or full_vocab.get("horizons", {}).get(f"k{full_horizon_key}")
        or {}
    )
    full_block = full_horizon_entry.get(
        "common_targets" if revision else "all_targets", {}
    )
    full_cand_count = max(1, int(len(vocabulary)))
    vocab_Sweep["full_vocab"] = {
        "label": f"Full Vocab ({full_cand_count})",
        "candidate_count": full_cand_count,
        "supported_candidates": int(
            full_block.get("target_count", full_cand_count)
        ),
        "query_count": int(full_block.get("query_count", len(eval_words))),
        "macro_top1": float(full_block.get("word_macro_recall_at_1", 0.0)),
        "macro_top5": float(full_block.get("word_macro_recall_at_5", 0.0)),
        "macro_top10": float(
            full_block.get(
                "subject_equal_macro_recall_at_10",
                full_vocab.get("selection_score", 0.0),
            )
        ),
        "word_macro_top10": float(
            full_block.get("word_macro_recall_at_10", 0.0)
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
            fv_horizon = (
                full_vocab.get("horizons", {}).get(d_key)
                or full_vocab.get("horizons", {}).get(f"k{d_key}")
                or {}
            )
            fv_common = fv_horizon.get("common_targets", {})
            fv_all = fv_horizon.get("all_targets", {})
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
                    fv_common.get(
                        "subject_equal_macro_recall_at_10",
                        fv_horizon.get("subject_macro_top10", 0.0),
                    )
                ),
                "common_full_vocab_word_macro_top10": float(
                    fv_common.get("word_macro_recall_at_10", 0.0)
                ),
                "common_full_vocab_micro_top10": float(
                    fv_common.get("micro_recall_at_10", 0.0)
                ),
                "common_full_vocab_macro_top1": float(
                    fv_common.get("word_macro_recall_at_1", 0.0)
                ),
                "common_full_vocab_micro_top1": float(
                    fv_common.get("micro_recall_at_1", 0.0)
                ),
                "common_full_vocab_mrr": float(
                    fv_common.get("mean_reciprocal_rank", 0.0)
                ),
                "common_full_vocab_median_rank": float(
                    fv_common.get("median_rank", full_cand_count / 2.0)
                ),
                "all_full_vocab_macro_top10": float(
                    fv_all.get("subject_equal_macro_recall_at_10", 0.0)
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


def _parse_shift_seconds_list(
    raw_shifts: str | Iterable[float | int | str] | None,
) -> list[float]:
    """解析时间错位秒数列表（支持逗号分隔字符串或数值序列），排除 0 秒并保持顺序去重。"""
    if raw_shifts is None:
        return []
    if isinstance(raw_shifts, (int, float)):
        val = float(raw_shifts)
        return [val] if abs(val) > 1e-9 else []
    if isinstance(raw_shifts, str):
        tokens: list[float | int | str] = [
            tok.strip() for tok in raw_shifts.split(",") if tok.strip()
        ]
    else:
        tokens = []
        for item in raw_shifts:
            if isinstance(item, str):
                tokens.extend(t.strip() for t in item.split(",") if t.strip())
            else:
                tokens.append(item)
    parsed: list[float] = []
    seen: set[float] = set()
    for tok in tokens:
        val = float(tok)
        if abs(val) <= 1e-9:
            continue
        rounded = round(val, 6)
        if rounded not in seen:
            seen.add(rounded)
            parsed.append(val)
    return parsed


def _format_shift_label(shift_sec: float) -> tuple[str, str]:
    """返回 (条件键名, 展示标签)，例如 ('shift_-30s', 'Shift -30s')。"""
    if abs(shift_sec - round(shift_sec)) < 1e-6:
        sec_int = int(round(shift_sec))
        return f"shift_{sec_int:+d}s", f"Shift {sec_int:+d}s"
    return f"shift_{shift_sec:+.1f}s", f"Shift {shift_sec:+.1f}s"


def _resolve_control_specification(
    cfg: dict,
    *,
    control_config_path: str | Path | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    zero_signal_control: bool = False,
    subtraction_etas: str | Iterable[float | int | str] | None = None,
) -> dict:
    """合并 CLI 参数、独立对照 YAML 配置（configs/controls/...）与主配置中的负对照设置。"""
    from braindecoding.evaluation.common_mode_subtraction_evaluation import (
        DEFAULT_SUBTRACTION_ETAS,
        parse_eta_values,
    )

    resolved_ctrl_path: Path | None = None
    ctrl_yaml: dict = {}
    if control_config_path is not None and str(control_config_path).strip():
        resolved_ctrl_path = resolve_config_path(project_path(control_config_path))
        if not resolved_ctrl_path.is_file():
            raise FileNotFoundError(
                f"指定的负对照配置文件不存在：{control_config_path}"
            )
        loaded = load_yaml_with_extends(resolved_ctrl_path)
        if isinstance(loaded, dict):
            ctrl_yaml = loaded

    dataset = cfg.get("experiment", {}).get("dataset")
    ctrl_wf = (
        ctrl_yaml.get("workflow")
        if isinstance(ctrl_yaml.get("workflow"), dict)
        else {}
    )
    ctrl_dataset = ctrl_wf.get("dataset")
    if dataset and ctrl_dataset and str(dataset) != str(ctrl_dataset):
        raise ValueError(
            f"数据集 {dataset!r} 与负对照配置文件中的 dataset={ctrl_dataset!r} 不一致。"
        )

    cfg_control = (
        dict(cfg.get("control")) if isinstance(cfg.get("control"), dict) else {}
    )
    cfg_eval = (
        dict(cfg.get("evaluation"))
        if isinstance(cfg.get("evaluation"), dict)
        else {}
    )
    yaml_control = (
        dict(ctrl_yaml.get("control"))
        if isinstance(ctrl_yaml.get("control"), dict)
        else {}
    )
    yaml_eval = (
        dict(ctrl_yaml.get("evaluation"))
        if isinstance(ctrl_yaml.get("evaluation"), dict)
        else {}
    )

    merged_control = {**cfg_control, **yaml_control}
    merged_eval = {**cfg_eval, **yaml_eval}

    cms_raw = (
        merged_control.get("common_mode_subtraction")
        if isinstance(merged_control.get("common_mode_subtraction"), dict)
        else (
            merged_eval.get("common_mode_subtraction")
            if isinstance(merged_eval.get("common_mode_subtraction"), dict)
            else {}
        )
    )
    cms_enabled = (
        subtraction_etas is not None
        or bool(cms_raw.get("enabled", False))
        or str(merged_control.get("control_type", "")).strip().lower()
        == "common_mode_subtraction"
        or str(merged_eval.get("negative_control", "")).strip().lower()
        == "common_mode_subtraction"
    )
    cms_etas = parse_eta_values(
        subtraction_etas
        if subtraction_etas is not None
        else cms_raw.get("eta_values", DEFAULT_SUBTRACTION_ETAS)
    )
    cms_normalize = bool(cms_raw.get("normalize_before_subtraction", True))

    if shift_seconds is not None:
        shifts = _parse_shift_seconds_list(shift_seconds)
    elif "shift_seconds" in merged_control:
        shifts = _parse_shift_seconds_list(merged_control.get("shift_seconds"))
    elif "shift_seconds" in merged_eval:
        shifts = _parse_shift_seconds_list(merged_eval.get("shift_seconds"))
    elif cms_enabled:
        shifts = [-120.0, -60.0, -30.0, 30.0, 60.0, 120.0]
    else:
        shifts = []

    enable_zero = (
        bool(zero_signal_control)
        or bool(merged_control.get("include_zero_signal", False))
        or bool(merged_eval.get("zero_signal_control", False))
        or str(merged_control.get("control_type", "")).strip().lower()
        == "zero_signal"
    )
    enabled = bool(shifts) or enable_zero or cms_enabled
    if not enabled:
        return {
            "enabled": False,
            "name": "none",
            "control_type": "none",
            "config_path": None,
            "shift_seconds": [],
            "include_zero_signal": False,
            "common_mode_subtraction": {
                "enabled": False,
                "mode": "leave_one_out",
                "eta_values": [],
                "normalize_before_subtraction": True,
            },
        }

    default_type = (
        "common_mode_subtraction"
        if cms_enabled
        else (
            "temporal_shift"
            if shifts and not enable_zero
            else ("zero_signal" if enable_zero and not shifts else "combined_control")
        )
    )
    control_type = str(merged_control.get("control_type") or default_type)
    control_name = str(
        merged_control.get("name") or f"{control_type}_negative_control"
    )
    cfg_path_label = (
        normalize_project_relative(resolved_ctrl_path)
        if resolved_ctrl_path is not None
        else (
            cfg.get("workflow", {}).get("config_path")
            if cfg_control or "shift_seconds" in cfg_eval
            else None
        )
    )
    return {
        "enabled": True,
        "name": control_name,
        "control_type": control_type,
        "description": str(merged_control.get("description", "")),
        "config_path": cfg_path_label,
        "shift_seconds": shifts,
        "shift_mode": str(
            merged_control.get("shift_mode", "deterministic_circular")
        ),
        "same_subject": bool(merged_control.get("same_subject", True)),
        "same_recording": bool(merged_control.get("same_recording", True)),
        "non_overlapping_window": bool(
            merged_control.get("non_overlapping_window", True)
        ),
        "include_clean_reference": bool(
            merged_control.get("include_clean_reference", True)
        ),
        "include_zero_signal": bool(enable_zero),
        "common_mode_subtraction": {
            "enabled": bool(cms_enabled),
            "mode": str(cms_raw.get("mode", "leave_one_out")),
            "eta_values": cms_etas,
            "normalize_before_subtraction": cms_normalize,
        },
    }


def _extract_control_summary_row(
    key: str,
    label: str,
    shift_seconds: float | None,
    metrics: dict,
    *,
    clean_summary: dict | None = None,
) -> dict:
    """提取单个条件（Clean、Time-Shifted 或 Zero-Signal）的核心词级检索指标与相对 Clean 的差值。"""
    sweep = metrics.get("vocabulary_sweep", {})
    n50 = sweep.get("N50", {})
    fv = sweep.get("full_vocab", {})
    horizons = metrics.get("revision_horizons") or {}

    row = {
        "key": key,
        "label": label,
        "shift_seconds": shift_seconds,
        "vocab50_macro_top10": float(n50.get("macro_top10", 0.0)),
        "vocab50_micro_top10": float(n50.get("micro_top10", 0.0)),
        "vocab50_macro_top5": float(n50.get("macro_top5", 0.0)),
        "vocab50_macro_top1": float(n50.get("macro_top1", 0.0)),
        "vocab50_mrr": float(n50.get("mrr", 0.0)),
        "vocab50_median_rank": float(n50.get("median_rank", 0.0)),
        "vocab50_chance_top10": float(n50.get("chance_top10", 0.2)),
        "full_vocab_macro_top10": float(fv.get("macro_top10", 0.0)),
        "full_vocab_micro_top10": float(fv.get("micro_top10", 0.0)),
        "full_vocab_macro_top1": float(fv.get("macro_top1", 0.0)),
        "full_vocab_mrr": float(fv.get("mrr", 0.0)),
        "full_vocab_median_rank": float(fv.get("median_rank", 0.0)),
    }
    if horizons:
        row["horizons"] = {
            h_key: {
                "common_vocab50_macro_top10": float(
                    horizons[h_key].get("common_vocab50_macro_top10", 0.0)
                ),
                "common_vocab50_micro_top10": float(
                    horizons[h_key].get("common_vocab50_micro_top10", 0.0)
                ),
                "common_vocab50_macro_top1": float(
                    horizons[h_key].get("common_vocab50_macro_top1", 0.0)
                ),
                "common_full_vocab_macro_top10": float(
                    horizons[h_key].get("common_full_vocab_macro_top10", 0.0)
                ),
                "common_full_vocab_micro_top10": float(
                    horizons[h_key].get("common_full_vocab_micro_top10", 0.0)
                ),
                "common_full_vocab_macro_top1": float(
                    horizons[h_key].get("common_full_vocab_macro_top1", 0.0)
                ),
                "common_full_vocab_mrr": float(
                    horizons[h_key].get("common_full_vocab_mrr", 0.0)
                ),
                "common_full_vocab_median_rank": float(
                    horizons[h_key].get("common_full_vocab_median_rank", 0.0)
                ),
            }
            for h_key in ("K0", "K1", "K2")
            if h_key in horizons
        }
    if clean_summary is not None:
        delta = {
            "vocab50_macro_top10": float(
                row["vocab50_macro_top10"] - clean_summary["vocab50_macro_top10"]
            ),
            "vocab50_micro_top10": float(
                row["vocab50_micro_top10"] - clean_summary["vocab50_micro_top10"]
            ),
            "vocab50_macro_top1": float(
                row["vocab50_macro_top1"] - clean_summary["vocab50_macro_top1"]
            ),
            "vocab50_mrr": float(
                row["vocab50_mrr"] - clean_summary["vocab50_mrr"]
            ),
            "full_vocab_macro_top10": float(
                row["full_vocab_macro_top10"]
                - clean_summary["full_vocab_macro_top10"]
            ),
            "clean_minus_control_vocab50_macro_top10": float(
                clean_summary["vocab50_macro_top10"] - row["vocab50_macro_top10"]
            ),
            "clean_minus_control_full_vocab_macro_top10": float(
                clean_summary["full_vocab_macro_top10"]
                - row["full_vocab_macro_top10"]
            ),
        }
        if "horizons" in row and "horizons" in clean_summary:
            delta["horizons"] = {
                h_key: {
                    "common_vocab50_macro_top10": float(
                        row["horizons"][h_key]["common_vocab50_macro_top10"]
                        - clean_summary["horizons"][h_key][
                            "common_vocab50_macro_top10"
                        ]
                    ),
                    "common_full_vocab_macro_top10": float(
                        row["horizons"][h_key]["common_full_vocab_macro_top10"]
                        - clean_summary["horizons"][h_key][
                            "common_full_vocab_macro_top10"
                        ]
                    ),
                }
                for h_key in row["horizons"]
                if h_key in clean_summary["horizons"]
            }
        row["delta_vs_clean"] = delta
    return row


def _summarize_shifted_conditions(
    clean_summary: dict,
    shifted_rows: list[dict],
) -> dict:
    """汇总所有时间错位偏移量（如 ±30s, ±60s, ±120s）的均值、标准差与相对 Clean 的平均降幅。"""
    if not shifted_rows:
        return {}
    metric_keys = (
        "vocab50_macro_top10",
        "vocab50_micro_top10",
        "vocab50_macro_top5",
        "vocab50_macro_top1",
        "vocab50_mrr",
        "vocab50_median_rank",
        "full_vocab_macro_top10",
        "full_vocab_micro_top10",
        "full_vocab_macro_top1",
        "full_vocab_mrr",
        "full_vocab_median_rank",
    )
    agg: dict = {
        "shift_count": len(shifted_rows),
        "shift_seconds": [r["shift_seconds"] for r in shifted_rows],
    }
    for m_key in metric_keys:
        vals = np.asarray([float(r[m_key]) for r in shifted_rows], dtype=np.float64)
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=0))
        agg[f"mean_{m_key}"] = mean_val
        agg[f"std_{m_key}"] = std_val
        if m_key in clean_summary:
            agg[f"delta_{m_key}_vs_clean"] = float(mean_val - float(clean_summary[m_key]))
            agg[f"clean_minus_shifted_{m_key}"] = float(
                float(clean_summary[m_key]) - mean_val
            )
    if all("horizons" in r for r in shifted_rows) and "horizons" in clean_summary:
        h_agg: dict = {}
        for h_key in ("K0", "K1", "K2"):
            if h_key not in clean_summary["horizons"]:
                continue
            v50_vals = np.asarray(
                [
                    float(r["horizons"][h_key]["common_vocab50_macro_top10"])
                    for r in shifted_rows
                    if h_key in r["horizons"]
                ],
                dtype=np.float64,
            )
            fv_vals = np.asarray(
                [
                    float(r["horizons"][h_key]["common_full_vocab_macro_top10"])
                    for r in shifted_rows
                    if h_key in r["horizons"]
                ],
                dtype=np.float64,
            )
            if len(v50_vals) > 0:
                m_v50 = float(np.mean(v50_vals))
                m_fv = float(np.mean(fv_vals))
                h_agg[h_key] = {
                    "mean_common_vocab50_macro_top10": m_v50,
                    "std_common_vocab50_macro_top10": float(np.std(v50_vals, ddof=0)),
                    "delta_common_vocab50_macro_top10_vs_clean": float(
                        m_v50
                        - float(
                            clean_summary["horizons"][h_key][
                                "common_vocab50_macro_top10"
                            ]
                        )
                    ),
                    "mean_common_full_vocab_macro_top10": m_fv,
                    "std_common_full_vocab_macro_top10": float(
                        np.std(fv_vals, ddof=0)
                    ),
                }
        agg["horizons"] = h_agg
    return agg


def _print_evaluation_report(eval_result: dict) -> None:
    """在终端或 Notebook 输出格式化的评估指标报表（含时间错位/全零信号负对照表）。"""
    sep = "=" * 96
    thin_sep = "-" * 96
    dataset = eval_result["dataset"]
    stage = eval_result["stage"]
    stage_label = STAGE_DISPLAY_NAMES.get(stage, stage)
    split = eval_result["split"].upper()
    subset = eval_result["subset"]
    ckpt_info = eval_result["checkpoint"]
    sweep = eval_result["metrics"]["vocabulary_sweep"]
    win_sec = float(eval_result.get("window_seconds", 1.0))
    win_samp = int(eval_result.get("window_samples", round(win_sec * 50.0)))
    win_off = float(eval_result.get("window_start_offset_seconds", 0.0))

    print(
        f"\n{sep}\n"
        f" BrainDecoding Evaluation  |  dataset={dataset}  |  split={split}  |  {stage_label}\n"
        f"{thin_sep}\n"
        f" Checkpoint  : {ckpt_info['path']} (epoch {ckpt_info['epoch']})\n"
        f" Subset      : {subset['subset_tag']} ({subset['subject_count']}/{subset['total_canonical_subjects']} subjects)  |  seed={eval_result['seed']}\n"
        f" Window      : {win_sec:.2f}s ({win_samp} samples @ 50Hz, start_offset={win_off:+.2f}s)\n"
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

    neg_ctrl = eval_result.get("negative_controls") or {}
    ctrl_conditions = neg_ctrl.get("conditions") or []
    if ctrl_conditions:
        clean_row = neg_ctrl["clean_reference"]
        ctrl_spec = neg_ctrl.get("specification") or {}
        cfg_note = (
            f" (config: {ctrl_spec['config_path']})"
            if ctrl_spec.get("config_path")
            else ""
        )
        print(
            f"{thin_sep}\n"
            f" Negative Control Comparison ({split}){cfg_note}:\n"
            f" {'Condition':<18} | {'50w Macro Top10':>15} | {'Delta vs Clean':>14} | {'50w Micro Top10':>15} | {'50w Macro Top1':>14} | {'FullVocab Top10':>15} | {'Med Rank':>8}\n"
            f"{'-' * 20}+{'-' * 17}+{'-' * 16}+{'-' * 17}+{'-' * 16}+{'-' * 17}+{'-' * 10}",
            flush=True,
        )
        print(
            f" {clean_row['label']:<18} | {clean_row['vocab50_macro_top10'] * 100.0:14.2f}% | {'(Reference)':>14} |"
            f" {clean_row['vocab50_micro_top10'] * 100.0:14.2f}% | {clean_row['vocab50_macro_top1'] * 100.0:13.2f}% |"
            f" {clean_row['full_vocab_macro_top10'] * 100.0:14.2f}% | {clean_row['vocab50_median_rank']:8.1f}",
            flush=True,
        )
        for cond in ctrl_conditions:
            s_row = cond["summary"]
            d_50 = s_row["delta_vs_clean"]["vocab50_macro_top10"] * 100.0
            print(
                f" {s_row['label']:<18} | {s_row['vocab50_macro_top10'] * 100.0:14.2f}% | {d_50:+13.2f}% |"
                f" {s_row['vocab50_micro_top10'] * 100.0:14.2f}% | {s_row['vocab50_macro_top1'] * 100.0:13.2f}% |"
                f" {s_row['full_vocab_macro_top10'] * 100.0:14.2f}% | {s_row['vocab50_median_rank']:8.1f}",
                flush=True,
            )
        shifted_agg = neg_ctrl.get("shifted_aggregate") or {}
        if shifted_agg.get("shift_count", 0) > 1:
            m50 = shifted_agg["mean_vocab50_macro_top10"] * 100.0
            sd50 = shifted_agg["std_vocab50_macro_top10"] * 100.0
            d50 = shifted_agg["delta_vocab50_macro_top10_vs_clean"] * 100.0
            mic50 = shifted_agg["mean_vocab50_micro_top10"] * 100.0
            top1_50 = shifted_agg["mean_vocab50_macro_top1"] * 100.0
            fv10 = shifted_agg["mean_full_vocab_macro_top10"] * 100.0
            med_r = shifted_agg["mean_vocab50_median_rank"]
            print(
                f"{'-' * 20}+{'-' * 17}+{'-' * 16}+{'-' * 17}+{'-' * 16}+{'-' * 17}+{'-' * 10}\n"
                f" {'Shifted Mean +- SD':<18} | {f'{m50:.2f}+-{sd50:.2f}%':>15} | {d50:+13.2f}% |"
                f" {mic50:14.2f}% | {top1_50:13.2f}% | {fv10:14.2f}% | {med_r:8.1f}",
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


def plot_negative_control_figures(
    eval_result: dict,
    figures_dir: str | Path,
) -> dict[str, str]:
    """绘制单阶段时间错位（±30s, ±60s, ±120s）/ 全零信号负对照对比图（PNG 与 PDF）。"""
    neg_ctrl = eval_result.get("negative_controls") or {}
    ctrl_conditions = neg_ctrl.get("conditions") or []
    if not ctrl_conditions:
        return {}

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
    clean_row = neg_ctrl["clean_reference"]

    # Sort conditions so negative shifts come before Clean (0s) and positive shifts come after
    neg_shifts = sorted(
        [
            c["summary"]
            for c in ctrl_conditions
            if c["summary"].get("shift_seconds") is not None
            and float(c["summary"]["shift_seconds"]) < 0
        ],
        key=lambda r: float(r["shift_seconds"]),
    )
    pos_shifts = sorted(
        [
            c["summary"]
            for c in ctrl_conditions
            if c["summary"].get("shift_seconds") is not None
            and float(c["summary"]["shift_seconds"]) > 0
        ],
        key=lambda r: float(r["shift_seconds"]),
    )
    other_ctrls = [
        c["summary"]
        for c in ctrl_conditions
        if c["summary"].get("shift_seconds") is None
    ]
    ordered_rows = [*neg_shifts, clean_row, *pos_shifts, *other_ctrls]
    clean_idx = len(neg_shifts)

    labels = [r["label"].replace("Shift ", "") for r in ordered_rows]
    x = np.arange(len(ordered_rows))
    v50_macro10 = [r["vocab50_macro_top10"] * 100.0 for r in ordered_rows]
    v50_micro10 = [r["vocab50_micro_top10"] * 100.0 for r in ordered_rows]
    v50_macro1 = [r["vocab50_macro_top1"] * 100.0 for r in ordered_rows]
    fv_macro10 = [r["full_vocab_macro_top10"] * 100.0 for r in ordered_rows]
    chance_50 = float(clean_row.get("vocab50_chance_top10", 0.2)) * 100.0

    primary_color = "#1f77b4" if dataset == "pallier2025" else "#6a3d9a"
    ctrl_color = "#b0bec5"
    highlight_color = "#d62728"

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=160)
    fig.suptitle(
        f"Negative Control Evaluation ({split})  |  {dataset}  |  {stage_title}  |  subset={subset_tag} (seed={seed})",
        fontsize=12.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: 50-Word Macro Top-10 & Micro Top-10 Across Temporal Shifts vs Clean (0s)
    bar_colors = [
        primary_color if idx == clean_idx else ctrl_color
        for idx in range(len(ordered_rows))
    ]
    bars = ax_left.bar(
        x,
        v50_macro10,
        width=0.56,
        color=bar_colors,
        edgecolor="#263238",
        linewidth=0.9,
        alpha=0.92,
        label="50-Word Macro Top-10 (%)",
    )
    ax_left.plot(
        x,
        v50_micro10,
        "-s",
        color="#ff7f0e",
        linewidth=1.8,
        markersize=5.5,
        label="50-Word Micro Top-10 (%)",
    )
    ax_left.plot(
        x,
        v50_macro1,
        "--^",
        color="#2ca02c",
        linewidth=1.6,
        markersize=5.0,
        label="50-Word Macro Top-1 (%)",
    )
    ax_left.axhline(
        chance_50,
        color="black",
        linestyle=":",
        linewidth=1.4,
        label=f"Random Chance Top-10 ({chance_50:.1f}%)",
    )
    for idx, bar in enumerate(bars):
        h = bar.get_height()
        ax_left.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if idx == clean_idx else "normal",
            color=primary_color if idx == clean_idx else "#37474f",
        )
    ax_left.set_title(
        f"(A) 50-Word Retrieval Accuracy: Clean (0s) vs Shifted Signals ({split})",
        fontsize=11,
        fontweight="bold",
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=9.5)
    ax_left.set_ylabel("Accuracy (%)", fontsize=10)
    ax_left.set_ylim(0, max(max(v50_macro10 + v50_micro10, default=35.0) * 1.24, 30.0))
    ax_left.grid(axis="y", linestyle=":", alpha=0.5)
    ax_left.legend(loc="upper right", fontsize=8.5)

    # Panel B: If revision stage with K0/K1/K2 horizons, plot K0/K1/K2 across shifts; otherwise plot Full-Vocab + Delta Drop
    has_horizons = all("horizons" in r for r in ordered_rows)
    if has_horizons and stage == "revision":
        h_colors = {"K0": "#4e79a7", "K1": "#f28e2b", "K2": "#59a14f"}
        for h_key in ("K0", "K1", "K2"):
            h_vals = [
                r["horizons"].get(h_key, {}).get("common_vocab50_macro_top10", 0.0)
                * 100.0
                for r in ordered_rows
            ]
            ax_right.plot(
                x,
                h_vals,
                "-o",
                linewidth=2.1,
                markersize=6,
                color=h_colors[h_key],
                label=f"{h_key} Common 50-Word Macro Top-10 (%)",
            )
        ax_right.axhline(
            chance_50,
            color="black",
            linestyle=":",
            linewidth=1.3,
            label=f"Chance Top-10 ({chance_50:.1f}%)",
        )
        ax_right.axvline(
            clean_idx,
            color=highlight_color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.6,
        )
        ax_right.set_title(
            f"(B) Rolling Revision Horizons (K0 / K1 / K2) Across Signal Shifts ({split})",
            fontsize=11,
            fontweight="bold",
        )
        ax_right.set_xticks(x)
        ax_right.set_xticklabels(labels, fontsize=9.5)
        ax_right.set_ylabel("50-Word Macro Top-10 (%)", fontsize=10)
        ax_right.set_ylim(0, max(ax_right.get_ylim()[1] * 1.18, 30.0))
        ax_right.grid(True, linestyle=":", alpha=0.5)
        ax_right.legend(loc="upper right", fontsize=8.5)
    else:
        ctrl_only = [c["summary"] for c in ctrl_conditions]
        cx = np.arange(len(ctrl_only))
        c_labels = [r["label"].replace("Shift ", "") for r in ctrl_only]
        drops_50 = [
            r["delta_vs_clean"]["vocab50_macro_top10"] * 100.0 for r in ctrl_only
        ]
        drops_fv = [
            r["delta_vs_clean"]["full_vocab_macro_top10"] * 100.0 for r in ctrl_only
        ]
        bw = 0.36
        b1 = ax_right.bar(
            cx - bw / 2,
            drops_50,
            bw,
            color="#e15759",
            alpha=0.9,
            label="Delta 50-Word Macro Top-10 (Control - Clean)",
        )
        b2 = ax_right.bar(
            cx + bw / 2,
            drops_fv,
            bw,
            color="#76b7b2",
            alpha=0.9,
            label="Delta Full-Vocab Macro Top-10 (Control - Clean)",
        )
        ax_right.axhline(0.0, color="black", linewidth=1.2)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            va = "top" if h < 0 else "bottom"
            y_off = -11 if h < 0 else 3
            ax_right.annotate(
                f"{h:+.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, y_off),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )
        ax_right.set_title(
            f"(B) Retrieval Accuracy Drop Under Negative Controls ({split})",
            fontsize=11,
            fontweight="bold",
        )
        ax_right.set_xticks(cx)
        ax_right.set_xticklabels(c_labels, fontsize=9.5)
        ax_right.set_ylabel("Accuracy Change vs Clean (%)", fontsize=10)
        y_min = min(min(drops_50 + drops_fv, default=-10.0) * 1.28, -5.0)
        y_max = max(max(drops_50 + drops_fv, default=2.0) * 1.25, 5.0)
        ax_right.set_ylim(y_min, y_max)
        ax_right.grid(axis="y", linestyle=":", alpha=0.5)
        ax_right.legend(loc="lower right", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    split_lower = eval_result["split"]
    png_path = out_dir / f"negative_control_{split_lower}.png"
    pdf_path = out_dir / f"negative_control_{split_lower}.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "negative_control_png": str(png_path),
        "negative_control_pdf": str(pdf_path),
    }


def plot_stage_control_comparison_figure(
    stage_results: dict[str, dict],
    figures_dir: str | Path,
    *,
    split: str = "val",
) -> dict[str, str]:
    """绘制多阶段（Stage 1 / 2.1 / 2.2）在 Clean (0s) 与各时间错位负对照（±30s, ±60s, ±120s）下的对比图。"""
    ordered_stages = [
        s
        for s in WORKFLOW_STAGES
        if s in stage_results
        and (stage_results[s].get("negative_controls") or {}).get("conditions")
    ]
    if not ordered_stages:
        return {}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_res = stage_results[ordered_stages[0]]
    dataset = first_res["dataset"]
    subset_tag = first_res["subset"]["subset_tag"]
    seed = first_res["seed"]
    split_upper = split.upper()

    first_nc = first_res["negative_controls"]
    first_conds = first_nc["conditions"]
    neg_shifts = sorted(
        [
            c["summary"]
            for c in first_conds
            if c["summary"].get("shift_seconds") is not None
            and float(c["summary"]["shift_seconds"]) < 0
        ],
        key=lambda r: float(r["shift_seconds"]),
    )
    pos_shifts = sorted(
        [
            c["summary"]
            for c in first_conds
            if c["summary"].get("shift_seconds") is not None
            and float(c["summary"]["shift_seconds"]) > 0
        ],
        key=lambda r: float(r["shift_seconds"]),
    )
    other_ctrls = [
        c["summary"]
        for c in first_conds
        if c["summary"].get("shift_seconds") is None
    ]
    ordered_keys = [
        *(r["key"] for r in neg_shifts),
        "clean_0s",
        *(r["key"] for r in pos_shifts),
        *(r["key"] for r in other_ctrls),
    ]
    x_labels = [
        *(r["label"].replace("Shift ", "") for r in neg_shifts),
        "Clean (0s)",
        *(r["label"].replace("Shift ", "") for r in pos_shifts),
        *(r["label"] for r in other_ctrls),
    ]
    x = np.arange(len(ordered_keys))

    stage_colors = {
        "word": "#4e79a7",
        "context": "#f28e2b",
        "revision": "#59a14f",
    }

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15.0, 5.8), dpi=160)
    fig.suptitle(
        f"Multi-Stage Negative Control Comparison ({split_upper})  |  dataset={dataset}  |  subset={subset_tag} (seed={seed})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    for st in ordered_stages:
        nc = stage_results[st]["negative_controls"]
        row_map = {"clean_0s": nc["clean_reference"]}
        for c in nc["conditions"]:
            row_map[c["key"]] = c["summary"]
        y_50 = [
            float(row_map.get(k, {}).get("vocab50_macro_top10", 0.0)) * 100.0
            for k in ordered_keys
        ]
        y_fv = [
            float(row_map.get(k, {}).get("full_vocab_macro_top10", 0.0)) * 100.0
            for k in ordered_keys
        ]
        ax_left.plot(
            x,
            y_50,
            "-o",
            linewidth=2.2,
            markersize=6,
            color=stage_colors.get(st, "#333333"),
            label=STAGE_DISPLAY_NAMES.get(st, st),
        )
        ax_right.plot(
            x,
            y_fv,
            "-s",
            linewidth=2.2,
            markersize=6,
            color=stage_colors.get(st, "#333333"),
            label=STAGE_DISPLAY_NAMES.get(st, st),
        )

    chance_50 = float(first_nc["clean_reference"].get("vocab50_chance_top10", 0.2)) * 100.0
    ax_left.axhline(
        chance_50,
        color="black",
        linestyle=":",
        linewidth=1.3,
        label=f"Random Chance ({chance_50:.1f}%)",
    )
    ax_left.set_title(
        f"(A) 50-Word Macro Top-10 Across Temporal Shifts ({split_upper})",
        fontsize=11,
        fontweight="bold",
    )
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(x_labels, fontsize=9.5)
    ax_left.set_ylabel("50-Word Macro Top-10 (%)", fontsize=10)
    ax_left.set_ylim(0, max(ax_left.get_ylim()[1] * 1.18, 35.0))
    ax_left.grid(True, linestyle=":", alpha=0.5)
    ax_left.legend(loc="upper right", fontsize=8.5)

    ax_right.set_title(
        f"(B) Full-Vocabulary Macro Top-10 Across Temporal Shifts ({split_upper})",
        fontsize=11,
        fontweight="bold",
    )
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(x_labels, fontsize=9.5)
    ax_right.set_ylabel("Full-Vocabulary Macro Top-10 (%)", fontsize=10)
    ax_right.set_ylim(0, max(ax_right.get_ylim()[1] * 1.22, 8.0))
    ax_right.grid(True, linestyle=":", alpha=0.5)
    ax_right.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png_path = out_dir / f"stage_control_comparison_{split}.png"
    pdf_path = out_dir / f"stage_control_comparison_{split}.pdf"
    fig.savefig(png_path, dpi=160)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "stage_control_comparison_png": str(png_path),
        "stage_control_comparison_pdf": str(pdf_path),
    }


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
    result_paths = {
        "overview_png": str(png_path),
        "overview_pdf": str(pdf_path),
    }
    result_paths.update(plot_negative_control_figures(eval_result, out_dir))
    return result_paths


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
    result_paths = {
        "stage_comparison_png": str(png_path),
        "stage_comparison_pdf": str(pdf_path),
    }
    result_paths.update(
        plot_stage_control_comparison_figure(stage_results, out_dir, split=split)
    )
    return result_paths


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
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
    require_cache: bool = False,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    config_path: str | Path | None = None,
    control_config_path: str | Path | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    zero_signal_control: bool = False,
    subtraction_etas: str | Iterable[float | int | str] | None = None,
    plot: bool = True,
    show_progress: bool = True,
    max_eval_rows: int | None = None,
) -> dict:
    """对单个训练阶段（word / context / revision）执行验证集或测试集评估（含可选负对照与探索性留一法共模扣除）并生成图表。"""
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
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        mode="train",
        output_dir=output_dir,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        config_path=config_path,
    )
    control_spec = _resolve_control_specification(
        cfg,
        control_config_path=control_config_path,
        shift_seconds=shift_seconds,
        zero_signal_control=zero_signal_control,
        subtraction_etas=subtraction_etas,
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
    checkpoint = load_checkpoint(ckpt_file, map_location="cpu")
    if not bool(cfg["workflow"].get("window_explicitly_overridden", False)):
        ckpt_cfg_ds = (
            checkpoint.get("config", {}).get("dataset", {})
            if isinstance(checkpoint.get("config"), dict)
            else {}
        )
        ckpt_win = checkpoint.get(
            "window_seconds",
            ckpt_cfg_ds.get("window_seconds") if isinstance(ckpt_cfg_ds, dict) else None,
        )
        ckpt_off = checkpoint.get(
            "window_start_offset_seconds",
            ckpt_cfg_ds.get("window_start_offset_seconds")
            if isinstance(ckpt_cfg_ds, dict)
            else None,
        )
        if ckpt_win is not None:
            cfg["dataset"]["window_seconds"] = float(ckpt_win)
            cfg["workflow"]["window_seconds"] = float(ckpt_win)
            sr_hz = float(cfg["workflow"].get("sampling_rate_hz", 50.0))
            cfg["workflow"]["window_samples"] = int(round(float(ckpt_win) * sr_hz))
        if ckpt_off is not None:
            cfg["dataset"]["window_start_offset_seconds"] = float(ckpt_off)
            cfg["workflow"]["window_start_offset_seconds"] = float(ckpt_off)

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
        progress_desc=f"Eval [{stage} | {split} | Clean 0s]",
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

    negative_controls_result: dict = {
        "specification": control_spec,
        "clean_reference": _extract_control_summary_row(
            "clean_0s", "Clean (0s)", 0.0, metrics
        ),
        "conditions": [],
        "shifted_aggregate": {},
    }
    cms_cfg = control_spec.get("common_mode_subtraction") or {}
    cms_enabled = bool(cms_cfg.get("enabled", False))
    shifted_encoded_by_shift: list[tuple[float, str, str, dict]] = []
    shifted_ds_by_shift: dict[float, object] = {}

    if control_spec["enabled"]:
        clean_summary = negative_controls_result["clean_reference"]
        shifted_summaries: list[dict] = []
        for shift_sec in control_spec["shift_seconds"]:
            cond_key, cond_label = _format_shift_label(float(shift_sec))
            shift_ds = build_fn(
                cfg, eval_table, temporal_shift_seconds=float(shift_sec)
            )
            shift_loader, _ = make_loader(
                shift_ds,
                tc,
                shuffle=False,
                group_column=grp_col,
                batch_size=eval_bs,
            )
            shift_encoded = encode_loader(
                model,
                shift_loader,
                device,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=f"Eval [{stage} | {split} | {cond_label}]",
            )
            shift_metrics = _compute_encoded_evaluation_metrics(
                shift_encoded,
                shift_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            summary_row = _extract_control_summary_row(
                cond_key,
                cond_label,
                float(shift_sec),
                shift_metrics,
                clean_summary=clean_summary,
            )
            shifted_summaries.append(summary_row)
            negative_controls_result["conditions"].append(
                {
                    "key": cond_key,
                    "label": cond_label,
                    "control_type": "temporal_shift",
                    "shift_seconds": float(shift_sec),
                    "summary": summary_row,
                    "metrics": shift_metrics,
                }
            )
            if cms_enabled:
                shifted_encoded_by_shift.append(
                    (float(shift_sec), cond_key, cond_label, shift_encoded)
                )
                shifted_ds_by_shift[float(shift_sec)] = shift_ds
                del shift_loader
            else:
                del shift_ds, shift_loader, shift_encoded

        if control_spec.get("include_zero_signal", False):
            zero_ds = build_fn(cfg, eval_table, zero_signal=True)
            zero_loader, _ = make_loader(
                zero_ds,
                tc,
                shuffle=False,
                group_column=grp_col,
                batch_size=eval_bs,
            )
            zero_encoded = encode_loader(
                model,
                zero_loader,
                device,
                amp=amp_enabled,
                amp_dtype=amp_dtype,
                show_progress=show_progress,
                progress_desc=f"Eval [{stage} | {split} | Zero Signal]",
            )
            zero_metrics = _compute_encoded_evaluation_metrics(
                zero_encoded,
                zero_ds,
                vocabulary,
                vocab50,
                cfg,
                stage=stage,
                split=split,
            )
            zero_summary = _extract_control_summary_row(
                "zero_signal",
                "Zero Signal",
                None,
                zero_metrics,
                clean_summary=clean_summary,
            )
            negative_controls_result["conditions"].append(
                {
                    "key": "zero_signal",
                    "label": "Zero Signal",
                    "control_type": "zero_signal",
                    "shift_seconds": None,
                    "summary": zero_summary,
                    "metrics": zero_metrics,
                }
            )
            del zero_ds, zero_loader, zero_encoded

        if shifted_summaries:
            negative_controls_result["shifted_aggregate"] = (
                _summarize_shifted_conditions(clean_summary, shifted_summaries)
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
    eval_win_sec = float(
        getattr(eval_ds, "window_seconds", cfg["dataset"].get("window_seconds", 1.0))
    )
    eval_win_samp = int(
        getattr(eval_ds, "window_samples", round(eval_win_sec * 50.0))
    )
    eval_win_off = float(cfg["dataset"].get("window_start_offset_seconds", 0.0))
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
        "window_seconds": eval_win_sec,
        "window_samples": eval_win_samp,
        "window_start_offset_seconds": eval_win_off,
        "subset": subset_spec,
        "checkpoint": {
            "path": str(ckpt_file),
            "sha256": file_sha256(ckpt_file),
            "epoch": int(checkpoint.get("epoch", 0)),
            "selection": str(checkpoint.get("selection", "legacy50")),
        },
        "cache_verification": cache_verification,
        "metrics": metrics,
        "negative_controls": negative_controls_result,
        "training_history": training_history,
        "output_dir": str(out),
        "figures": {},
    }

    _print_evaluation_report(eval_result)

    if plot:
        figs = plot_evaluation_figures(eval_result, out / "figures")
        eval_result["figures"].update(figs)

    if cms_enabled and len(shifted_encoded_by_shift) >= 2:
        from braindecoding.evaluation.common_mode_subtraction_evaluation import (
            run_common_mode_subtraction_sweep,
        )

        cms_report = run_common_mode_subtraction_sweep(
            dataset=dataset,
            stage=stage,
            split=split,
            seed=seed,
            exp_tag=resolved_exp_tag,
            word_exp_tag=resolved_word_exp_tag,
            clean_encoded=encoded,
            shifted_encoded_by_shift=shifted_encoded_by_shift,
            eval_ds=eval_ds,
            shifted_ds_by_shift=shifted_ds_by_shift,
            vocabulary=vocabulary,
            vocab50=vocab50,
            cfg=cfg,
            eta_values=cms_cfg.get("eta_values", (0.5, 0.8, 1.0)),
            normalize_inputs=bool(
                cms_cfg.get("normalize_before_subtraction", True)
            ),
            baseline_clean_summary=negative_controls_result["clean_reference"],
            baseline_condition_entries=negative_controls_result["conditions"],
            baseline_shifted_aggregate=negative_controls_result["shifted_aggregate"],
            compute_metrics_fn=_compute_encoded_evaluation_metrics,
            extract_summary_row_fn=_extract_control_summary_row,
            summarize_shifted_fn=_summarize_shifted_conditions,
            output_dir=out,
            plot=plot,
            print_report=True,
        )
        eval_result["exploratory_common_mode_subtraction_json"] = cms_report.get(
            "exploratory_json"
        )
        eval_result["exploratory_common_mode_subtraction_summary"] = {
            "eta_grid": cms_report["eta_grid"],
            "best_eta_by_clean_vocab50_top10": cms_report[
                "best_eta_by_clean_vocab50_top10"
            ],
            "best_eta_by_clean_minus_shifted_vocab50_top10": cms_report[
                "best_eta_by_clean_minus_shifted_vocab50_top10"
            ],
        }
        eval_result["figures"].update(cms_report.get("figures", {}))
        shifted_encoded_by_shift.clear()
        shifted_ds_by_shift.clear()

    eval_json_path = out / f"evaluation_{split}.json"
    save_json(eval_json_path, eval_result)
    eval_result["evaluation_json"] = str(eval_json_path)

    del model, eval_ds, eval_loader, checkpoint, encoded
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
    window_seconds: float | None = None,
    window_start_offset_seconds: float | None = None,
    require_cache: bool = False,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    exp_tag: str | None = None,
    word_exp_tag: str | None = None,
    config_path: str | Path | None = None,
    control_config_path: str | Path | None = None,
    shift_seconds: str | Iterable[float | int | str] | None = None,
    zero_signal_control: bool = False,
    subtraction_etas: str | Iterable[float | int | str] | None = None,
    all_stages: bool = False,
    plot: bool = True,
    show_progress: bool = True,
    max_eval_rows: int | None = None,
) -> dict:
    """运行验证集 (val) 或测试集 (test) 评估与可视化工作流（支持单阶段、全阶段对比、负对照及探索性留一法共模扣除）。"""
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
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            require_cache=require_cache,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            exp_tag=exp_tag,
            word_exp_tag=word_exp_tag,
            config_path=config_path,
            control_config_path=control_config_path,
            shift_seconds=shift_seconds,
            zero_signal_control=zero_signal_control,
            subtraction_etas=subtraction_etas,
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
        window_seconds=window_seconds,
        window_start_offset_seconds=window_start_offset_seconds,
        exp_tag=exp_tag,
        word_exp_tag=word_exp_tag,
        config_path=config_path,
    )
    effective_control_config_path = control_config_path
    if effective_control_config_path is None and config_path is not None:
        resolved_cfg_p = resolve_config_path(project_path(config_path))
        if resolved_cfg_p.is_file():
            raw_explicit = load_yaml_with_extends(resolved_cfg_p)
            if isinstance(raw_explicit, dict) and (
                "control" in raw_explicit
                or "shift_seconds" in (raw_explicit.get("evaluation") or {})
                or bool((raw_explicit.get("evaluation") or {}).get("zero_signal_control"))
            ):
                effective_control_config_path = config_path

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
            window_seconds=window_seconds,
            window_start_offset_seconds=window_start_offset_seconds,
            require_cache=require_cache,
            output_dir=stage_out_dir,
            exp_tag=st_tag,
            word_exp_tag=word_exp_tag,
            config_path=None,
            control_config_path=effective_control_config_path,
            shift_seconds=shift_seconds,
            zero_signal_control=zero_signal_control,
            subtraction_etas=subtraction_etas,
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
        "window_seconds": float(base_cfg["workflow"]["window_seconds"]),
        "window_samples": int(base_cfg["workflow"]["window_samples"]),
        "window_start_offset_seconds": float(base_cfg["workflow"]["window_start_offset_seconds"]),
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


