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
    project_path,
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


def workflow_output_directory(
    dataset: str,
    stage: str,
    subset_tag: str,
    seed: int,
    *,
    mode: str = "train",
    output_dir: str | Path | None = None,
) -> Path:
    """计算受试者子集隔离的实验输出目录。"""
    if output_dir is not None:
        return Path(output_dir).resolve()
    stage_dir = f"smoke_{stage}" if mode == "smoke" else stage
    return (
        results_root()
        / dataset
        / stage_dir
        / subset_tag
        / f"seed-{int(seed):03d}"
    ).resolve()


def build_workflow_config(
    dataset: str,
    *,
    stage: str = "word",
    seed: int = 0,
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    precision: str = "bf16-mixed",
    mode: str = "train",
    output_dir: str | Path | None = None,
    word_checkpoint: str | Path | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
) -> tuple[dict, dict]:
    """构建支持灵活受试者子集与环境变量路径解耦的训练配置。"""
    ensure_default_environment()
    if dataset not in DATASETS:
        raise ValueError(f"不支持的数据集 {dataset!r}，可选：{DATASETS}")
    if stage not in STAGES:
        raise ValueError(f"不支持的训练阶段 {stage!r}，可选：{STAGES}")
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError(f"随机种子必须是非负整数：{seed!r}")
    seed = int(seed)

    subset_spec = resolve_subject_subset(
        dataset, subject=subject, num_subjects=num_subjects, subjects=subjects
    )
    precision_label, amp_dtype_name, amp_enabled = _normalize_precision(precision)

    settings = load_yaml_with_extends(CONFIG_PATH)
    adapter = settings["adapters"][dataset]
    loader = (
        pallier_task.load_config
        if dataset == "pallier2025"
        else chinese_task.load_config
    )
    cfg = loader(project_path(adapter[f"{stage}_source_config"]))
    cfg["experiment"]["output_group"] = settings["execution"]["output_group"]
    cfg["model"] = copy.deepcopy(settings["model"])
    cfg["model"]["use_transformer"] = stage == "revision"
    if stage == "word":
        cfg["model"].pop("local_window_revision", None)
    cfg["text_embedding"] = copy.deepcopy(settings["text_embedding"])
    cfg["loss"] = copy.deepcopy(settings["loss"])
    cfg["cache"]["text_embeddings"] = str(project_path(adapter["embedding_cache"]))

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
        tc[key] = copy.deepcopy(common[key])
    default_epochs = 2 if mode == "smoke" else int(common[f"{stage}_epochs"])
    tc.update(
        seed=seed,
        epochs=int(epochs) if epochs is not None else default_epochs,
        top_ks=[1, 10],
        amp=amp_enabled,
        amp_dtype=amp_dtype_name,
        precision=precision_label,
    )
    if batch_size is not None:
        tc["batch_size"] = int(batch_size)
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

    if stage == "revision":
        tc.update(
            preserve_complete_groups=True,
            revision_cutoff_policy="all",
            freeze_brain_encoder_epochs=int(common["freeze_brain_encoder_epochs"]),
            save_last_checkpoint=True,
            warm_start_from="main_word",
            warm_start_seed=seed,
            warm_start_output_group=settings["execution"]["output_group"],
        )
        if word_checkpoint:
            tc["pretrained_brain_encoder_checkpoint"] = str(
                project_path(word_checkpoint)
            )
        else:
            default_word_dir = workflow_output_directory(
                dataset,
                "word",
                subset_spec["subset_tag"],
                seed,
                mode=mode,
            )
            candidate_ckpt = default_word_dir / "best.ckpt"
            if not candidate_ckpt.exists():
                candidate_ckpt = default_word_dir / "best.pt"
            tc["pretrained_brain_encoder_checkpoint"] = str(candidate_ckpt)

    cfg["evaluation"]["selection_metric"] = adapter[
        f"main_selection_metric_{stage}"
    ]
    cfg["evaluation"]["default_split"] = "val"
    out_dir = workflow_output_directory(
        dataset,
        stage,
        subset_spec["subset_tag"],
        seed,
        mode=mode,
        output_dir=output_dir,
    )
    tc["output_dir"] = str(out_dir)
    cfg["workflow"] = {
        "mode": mode,
        "stage": stage,
        "precision": precision_label,
        "amp_dtype": amp_dtype_name,
        "subset": subset_spec,
        "environment": {
            "BRAINDECODING_DATA_DIR": str(data_root()),
            "BRAINDECODING_CACHE_DIR": str(cache_root()),
            "BRAINDECODING_RESULTS_DIR": str(results_root()),
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
    val_loader, _ = make_loader(
        val_ds, cfg["training"], shuffle=False, group_column=grp_col
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
    dataset: str,
    *,
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    include_test: bool = False,
    force: bool = False,
) -> dict:
    """预热并校验指定数据集与受试者子集的事件表、神经信号与文本嵌入缓存。"""
    ensure_default_environment()
    datasets = list(DATASETS) if str(dataset).lower() == "all" else [dataset]
    reports = {}
    for ds_name in datasets:
        cfg, subset_spec = build_workflow_config(
            ds_name,
            stage="word",
            seed=0,
            subject=subject,
            num_subjects=num_subjects,
            subjects=subjects,
            mode="cache",
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
            text_data.ensure_text_embedding_cache(
                text_words,
                cfg["text_embedding"],
                cfg["cache"]["text_embeddings"],
                force=force,
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
            text_data.ensure_text_embedding_cache(
                text_words,
                cfg["text_embedding"],
                cfg["cache"]["text_embeddings"],
                force=force,
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
    dataset: str,
    *,
    stage: str = "word",
    seed: int = 0,
    subject: str | None = None,
    num_subjects: int | None = None,
    subjects: str | Iterable[str] | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    precision: str = "bf16-mixed",
    require_cache: bool = False,
    resume: str | None = None,
    mode: str = "train",
    output_dir: str | Path | None = None,
    word_checkpoint: str | Path | None = None,
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    overwrite: bool = False,
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
        precision=precision,
        mode=mode,
        output_dir=output_dir,
        word_checkpoint=word_checkpoint,
        max_train_rows=max_train_rows,
        max_val_rows=max_val_rows,
    )
    tc = cfg["training"]
    out = Path(tc["output_dir"])
    revision = stage == "revision"

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
            )
            cache_verification = verify_cache_ready(cfg, splits=("train", "val"))

    device = choose_device(tc.get("device", "auto"))
    if device.type != "cuda":
        raise RuntimeError("训练与 GPU 冒烟测试要求 CUDA 设备可用。")

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
    torch.set_num_threads(2)
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
    if revision and resume_ckpt_path is None:
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
                f"revision 阶段需要已完成同受试者子集的 word 检查点：{source_ckpt}"
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
        print(
            f"[workflow] Resumed from {resume_ckpt_path} at epoch {start_epoch}/{tc['epochs']}",
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats(device)
    selection_metric = cfg["evaluation"]["selection_metric"]

    for epoch in range(start_epoch, int(tc["epochs"])):
        epoch_started = time.perf_counter()
        train_sampler.set_epoch(epoch)
        frozen = revision and epoch < int(tc.get("freeze_brain_encoder_epochs", 0))
        pallier_task.set_brain_encoder_trainable(model, not frozen)

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
        for selection, score in scores.items():
            if score > best[selection]:
                best[selection] = score
                best_rows[selection] = row
                if selection == "legacy50":
                    stale_epochs = 0
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

        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "dataset": dataset,
                    "stage": stage,
                    "mode": mode,
                    "subset_tag": subset_spec["subset_tag"],
                    "epoch": epoch + 1,
                    "max_epochs": int(tc["epochs"]),
                    "train_loss": round(float(train_loss), 5),
                    "val_legacy50_score": round(legacy_score, 5),
                    "val_full_vocab_score": round(full_vocab_score, 5),
                    "peak_memory_allocated_mb": round(peak_alloc_mb, 2),
                    "peak_memory_reserved_mb": round(peak_reserved_mb, 2),
                    "seconds": round(float(epoch_seconds), 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stale_epochs >= int(tc.get("patience", 6)):
            break

    peak_alloc_mb = float(
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
    )
    peak_reserved_mb = float(
        torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
    )
    stop_reason = (
        "patience_exhausted"
        if stale_epochs >= int(tc.get("patience", 6))
        else "maximum_epochs"
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
