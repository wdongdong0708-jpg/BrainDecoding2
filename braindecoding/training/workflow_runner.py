"""本地小样本极速联调 + 云端 CPU 预热 / GPU 全量训练 双模工作流执行器。

提供统一的 CLI 后端实现（``smoke``、``cache``、``train``）：
- 环境变量解耦：``BRAINDECODING_DATA_DIR``、``BRAINDECODING_CACHE_DIR``、``BRAINDECODING_RESULTS_DIR``
- 灵活受试者子集选择：``--subject all``、``--num-subjects N``、``--subjects id1,id2,...``
- 动态子集样本数校验（移除写死的全量样本数硬编码断言）
- 子集专属输出目录与 ``subjects.json`` 防覆盖隔离
- ``--require-cache`` 零现场重算保障与 ``--resume auto`` (``last.ckpt``) 断点续训
- ``bf16-mixed`` / ``fp16`` / ``fp32`` 混合精度与 GPU 峰值显存记录

评估计算、终端报表与可视化图表已解耦至 ``braindecoding.evaluation``：
- ``braindecoding.evaluation.word_retrieval_evaluation``
- ``braindecoding.evaluation.evaluation_reporting``
- ``braindecoding.evaluation.evaluation_figures``
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
from braindecoding.evaluation.evaluation_figures import (
    STAGE_DISPLAY_NAMES,
    plot_evaluation_figures,
    plot_negative_control_figures,
    plot_stage_comparison_figure,
    plot_stage_control_comparison_figure,
)
from braindecoding.evaluation.evaluation_reporting import (
    print_word_evaluation_report as _print_evaluation_report,
)
from braindecoding.evaluation.retrieval_metrics import (
    fixed_vocabulary_retrieval_metrics,
    mean_by_target,
    retrieval_ranks,
)
from braindecoding.evaluation.word_retrieval_evaluation import (
    _build_enriched_full_vocabulary_metrics,
    _compute_encoded_evaluation_metrics,
    _compute_full_vocabulary_horizon_details,
    _evaluate_single_stage,
    _extract_control_summary_row,
    _format_shift_label,
    _parse_shift_seconds_list,
    _resolve_control_specification,
    _score_full_vocabulary_population,
    _summarize_shifted_conditions,
    run_evaluation_workflow,
    summarize_multi_seed_evaluation,
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

WORKFLOW_STAGES: tuple[str, ...] = (
    "word",
    "context",
    "revision",
    "prefix_revision",
)

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
    "2.3": "prefix_revision",
    "stage2.3": "prefix_revision",
    "stage_2_3": "prefix_revision",
    "stage-2.3": "prefix_revision",
    "prefix_revision": "prefix_revision",
    "full_history_revision": "prefix_revision",
    "full_prefix_revision": "prefix_revision",
}

DEFAULT_STAGE_CONFIGS: dict[tuple[str, str], str] = {
    ("pallier2025", "word"): "configs/training/pallier2025_word_encoder.yaml",
    ("pallier2025", "context"): "configs/training/pallier2025_context_transformer.yaml",
    ("pallier2025", "revision"): "configs/training/pallier2025_rolling_revision.yaml",
    ("pallier2025", "prefix_revision"): "configs/training/pallier2025_prefix_revision.yaml",
    ("chineseeeg2_littleprince", "word"): "configs/training/chineseeeg2_word_encoder.yaml",
    ("chineseeeg2_littleprince", "context"): "configs/training/chineseeeg2_context_transformer.yaml",
    ("chineseeeg2_littleprince", "revision"): "configs/training/chineseeeg2_rolling_revision.yaml",
    ("chineseeeg2_littleprince", "prefix_revision"): "configs/training/chineseeeg2_prefix_revision.yaml",
}


def normalize_workflow_stage(stage: str | None) -> str:
    """将用户输入或配置中的阶段标识标准化为 'word' | 'context' | 'revision' | 'prefix_revision'。"""
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
            has_lwr = isinstance(lwr, dict) and bool(lwr.get("enabled", False))
            has_pr = isinstance(pr, dict) and bool(pr.get("enabled", False))
            has_revision = has_lwr or has_pr
            if not has_revision:
                yaml_stage = "context"
            else:
                is_full_history_lwr = (
                    has_lwr
                    and "history_radius" in lwr
                    and (
                        lwr.get("history_radius") is None
                        or str(lwr.get("history_radius")).strip().lower()
                        in {"none", "null", "full", "all"}
                    )
                )
                yaml_stage = (
                    "prefix_revision"
                    if (has_pr or is_full_history_lwr)
                    else "revision"
                )

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
    eta_0: float | None = None,
    eta_delta: float | None = None,
    lambda_null: float | None = None,
    lambda_delta_null: float | None = None,
    beta_delta_cm: float | None = None,
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
    adapter_stage = "revision" if stage == "prefix_revision" else stage
    default_source_rel = DEFAULT_STAGE_CONFIGS.get(
        (dataset, stage),
        adapter.get(f"{adapter_stage}_source_config", adapter["word_source_config"]),
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
    cfg["model"]["use_transformer"] = stage in (
        "context",
        "revision",
        "prefix_revision",
    )
    if stage in ("word", "context"):
        cfg["model"].pop("local_window_revision", None)
        cfg["model"].pop("prefix_revision", None)
    elif stage == "prefix_revision":
        if isinstance(cfg["model"].get("prefix_revision"), dict) and bool(
            cfg["model"]["prefix_revision"].get("enabled", False)
        ):
            cfg["model"].pop("local_window_revision", None)
            cfg["model"]["prefix_revision"].setdefault("horizons", [0, 1, 2])
            cfg["model"]["prefix_revision"].setdefault(
                "separate_horizon_losses", True
            )
        else:
            lwr_cfg = cfg["model"].setdefault("local_window_revision", {})
            lwr_cfg["enabled"] = True
            lwr_cfg["history_radius"] = None
            lwr_cfg.setdefault("horizons", [0, 1, 2])
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
            else common.get(
                f"{adapter_stage}_epochs", common.get("revision_epochs", 30)
            )
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
    if stage in ("context", "revision", "prefix_revision"):
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
        if stage in ("revision", "prefix_revision"):
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

    raw_siam_cfg = (
        cfg.get("siamese_debiased")
        if isinstance(cfg.get("siamese_debiased"), dict)
        else {}
    )
    yaml_siam_enabled = bool(
        raw_siam_cfg.get("enabled", False)
        or raw_siam_cfg.get("preextract_frozen_cnn", False)
        or tc.get("preextract_frozen_cnn", False)
    )
    ignore_implicit_siamese_on_word = (
        stage not in ("context", "revision", "prefix_revision")
        and siamese_debiased is not True
        and preextract_frozen_cnn is not True
        and not yaml_siam_enabled
    )
    siamese_spec = resolve_siamese_debiased_config(
        cfg,
        siamese_debiased=False if ignore_implicit_siamese_on_word else siamese_debiased,
        eta_train=None if ignore_implicit_siamese_on_word else eta_train,
        eta_0=None if ignore_implicit_siamese_on_word else eta_0,
        eta_delta=None if ignore_implicit_siamese_on_word else eta_delta,
        lambda_null=None if ignore_implicit_siamese_on_word else lambda_null,
        lambda_delta_null=None if ignore_implicit_siamese_on_word else lambda_delta_null,
        beta_delta_cm=None if ignore_implicit_siamese_on_word else beta_delta_cm,
        shifts_per_step=None if ignore_implicit_siamese_on_word else shifts_per_step,
        preextract_frozen_cnn=(
            False if ignore_implicit_siamese_on_word else preextract_frozen_cnn
        ),
    )
    if (
        siamese_spec["enabled"] or siamese_spec["preextract_frozen_cnn"]
    ) and stage not in ("context", "revision", "prefix_revision"):
        raise ValueError(
            "孪生共模抵消与冻结 CNN 预提取仅适用于 context (Stage 2.1)、revision (Stage 2.2) 或 prefix_revision (Stage 2.3) 阶段。"
        )
    if (
        siamese_spec["enabled"] or siamese_spec["preextract_frozen_cnn"]
    ) and stage in ("context", "revision", "prefix_revision"):
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
        or adapter.get(
            f"main_selection_metric_{adapter_stage}",
            adapter["main_selection_metric_word"],
        )
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
    eta_0: float | None = None,
    eta_delta: float | None = None,
    lambda_null: float | None = None,
    lambda_delta_null: float | None = None,
    beta_delta_cm: float | None = None,
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
        eta_0=eta_0,
        eta_delta=eta_delta,
        lambda_null=lambda_null,
        lambda_delta_null=lambda_delta_null,
        beta_delta_cm=beta_delta_cm,
        shifts_per_step=shifts_per_step,
        preextract_frozen_cnn=preextract_frozen_cnn,
    )
    dataset = subset_spec["dataset"]
    stage = cfg["workflow"]["stage"]
    tc = cfg["training"]
    seed = int(tc["seed"])
    out = Path(tc["output_dir"])
    revision = stage in ("revision", "prefix_revision")
    warm_start_stage = stage in ("context", "revision", "prefix_revision")

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
    siamese_spec = dict(cfg["workflow"].get("siamese_debiased") or {})
    siamese_enabled = bool(siamese_spec.get("enabled", False))
    raw_early_stop_metric = str(
        tc.get(
            "early_stop_metric",
            "full_vocab" if siamese_enabled else "legacy50",
        )
    ).strip().lower()
    early_stop_target = (
        "full_vocab"
        if raw_early_stop_metric in {"full_vocab", "full_vocabulary", "full"}
        else "legacy50"
    )
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
        f"  |  epochs={max_epochs} (patience={patience}, early_stop={early_stop_target})\n"
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

    all_epochs_frozen = (
        warm_start_stage
        and int(tc.get("freeze_brain_encoder_epochs", 0)) >= max_epochs
    )
    use_preextracted_cnn = bool(
        all_epochs_frozen
        and (siamese_enabled or bool(siamese_spec.get("preextract_frozen_cnn", False)))
    )
    if siamese_enabled:
        siam_eta0 = float(
            siamese_spec.get("eta_0", siamese_spec.get("eta_train", 0.8))
        )
        siam_eta_d = float(siamese_spec.get("eta_delta", 0.0))
        siam_lam = float(siamese_spec.get("lambda_null", 10.0))
        siam_lam_d = float(siamese_spec.get("lambda_delta_null", 0.0))
        siam_beta_d = float(siamese_spec.get("beta_delta_cm", 6.0))
        delta_msg = (
            f"eta_delta={siam_eta_d:.2f}, lambda_delta_null={siam_lam_d:.2f}, beta_delta_cm={siam_beta_d:.2f}, "
            if (siam_eta_d > 0.0 or siam_lam_d > 0.0 or siam_beta_d > 0.0)
            else ""
        )
        print(
            f" Siamese Deb : enabled (eta_0={siam_eta0:.2f}, lambda_null={siam_lam:.2f}, {delta_msg}"
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
        if siamese_enabled and (
            float(siamese_spec.get("lambda_null", 0.0)) > 0.0
            or float(siamese_spec.get("lambda_delta_null", 0.0)) > 0.0
            or float(siamese_spec.get("beta_delta_cm", 0.0)) > 0.0
        ):
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
                if selection == early_stop_target:
                    stale_epochs = 0
                if selection == "legacy50":
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
                if selection == early_stop_target:
                    save_checkpoint(out / "best.ckpt", best_payload)
                    save_checkpoint(out / "best.pt", best_payload)
                if selection == "legacy50":
                    save_checkpoint(out / "best_legacy50.pt", best_payload)
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
            dn_val = float(diagnostics.get("delta_shift_null_loss", 0.0))
            if (
                dn_val > 0.0
                or float(siamese_spec.get("lambda_delta_null", 0.0)) > 0.0
                or float(siamese_spec.get("beta_delta_cm", 0.0)) > 0.0
            ):
                status_str += (
                    f" [sig={float(diagnostics.get('siglip_loss', train_loss)):.3f},"
                    f"null={float(diagnostics.get('shift_null_loss', 0.0)):.4f},"
                    f"d_null={dn_val:.4f}]"
                )
            else:
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
    best50_path = out / (
        "best.ckpt" if early_stop_target == "legacy50" else "best_legacy50.pt"
    )
    best_full_path = out / (
        "best.ckpt" if early_stop_target == "full_vocab" else "best_full_vocab.pt"
    )
    print(
        f"{thin_sep}\n"
        f" Completed ({stop_reason}, epochs={len(history)}/{max_epochs}, early_stop={early_stop_target})\n"
        f" Best Val Top10 (50-word)   : {best['legacy50'] * 100.0:6.2f}% (epoch {best50_epoch}) -> {best50_path}\n"
        f" Best Val Top10 (Full-Vocab): {best['full_vocab'] * 100.0:6.2f}% (epoch {best_full_epoch}) -> {best_full_path}\n"
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
