"""ChineseEEG2《小王子》50 高频词闭集解码训练入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends, project_path


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "word_decoding"
    / "chineseeeg2_littleprince"
    / "sub01-08"
    / "main_context_warmstart.yaml"
)
from braindecoding.experiment import (
    initialize_run_directory,
    resolve_experiment_config,
    update_run_status,
)
from braindecoding.results import build_training_summary
from braindecoding.training.runtime import (
    choose_device,
    cpu_state_dict,
    limit_rows,
    load_checkpoint,
    parameter_count,
    save_checkpoint,
    save_json,
    set_seed,
)
from braindecoding.training.word import (
    brain_encoder_frozen_for_epoch,
    encode_loader as encode_word_loader,
    make_loader,
    train_one_epoch as train_word_epoch,
    validation_patience_exhausted,
)
from braindecoding.data import chineseeeg2 as dataset_module
from braindecoding.losses import build_siglip_loss
from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval_metrics
from braindecoding.models import build_brain_embedding_model
from braindecoding.optimizers import (
    build_adamw_for_modules,
    build_cosine_annealing_scheduler,
)


def 载入配置(path=None):
    """载入单一任务配置，并展开项目内的输出与缓存路径。"""
    config_path = project_path(path or DEFAULT_CONFIG)
    config = load_yaml_with_extends(config_path)
    config = resolve_experiment_config(config)
    for key in ("event_table", "eeg_dir", "text_embeddings"):
        config["cache"][key] = str(project_path(config["cache"][key]))
    config["training"]["output_dir"] = str(
        project_path(config["training"]["output_dir"])
    )
    checkpoint = config["training"].get("pretrained_brain_encoder_checkpoint")
    if checkpoint:
        config["training"]["pretrained_brain_encoder_checkpoint"] = str(
            project_path(checkpoint)
        )
    alignment_path = config["dataset"].get("alignment_path")
    if alignment_path:
        config["dataset"]["alignment_path"] = str(project_path(alignment_path))
    for source in config["dataset"].get("actual_reading_sources", ()):
        source_alignment_path = source.get("alignment_path")
        if source_alignment_path:
            source["alignment_path"] = str(project_path(source_alignment_path))
    return config


def 准备事件表(
    config,
    with_eeg_cache=False,
    with_text_embeddings=False,
    include_test=False,
    force_cache=False,
):
    """先完成元数据审计，再按明确授权物化所选数据。"""
    if config["dataset"].get("event_source_type") != "actual_reading_workbook":
        raise ValueError("《小王子》任务只接受 actual_reading 工作簿事件。")
    event_table = dataset_module.构建实际朗读事件表(config["dataset"])
    audit = dataset_module.审计事件表(event_table, config["dataset"])
    event_path = dataset_module.保存事件表(
        event_table, config["cache"]["event_table"], audit
    )
    selected_splits = ["train", "val"] + (["test"] if include_test else [])
    selected = event_table[
        event_table["split"].isin(selected_splits)
        & event_table["is_trainable"].astype(bool)
    ].reset_index(drop=True)
    if with_text_embeddings:
        dataset_module.ensure_text_embedding_cache(
            selected["normalized_word"],
            config["text_embedding"],
            config["cache"]["text_embeddings"],
        )
    if with_eeg_cache:
        dataset_module.确保记录缓存(
            selected,
            config["dataset"],
            config["cache"]["eeg_dir"],
            force=force_cache,
        )
    return {
        "status": "event_table_prepared",
        "event_table": str(event_path),
        "audit": audit,
        "materialized_splits": selected_splits,
        "text_embeddings_materialized": bool(with_text_embeddings),
        "eeg_materialized": bool(with_eeg_cache),
        "test_eeg_opened": bool(with_eeg_cache and include_test),
    }


def 确保事件表(config):
    """事件表缺失时只构建元数据，不物化任何脑电记录。"""
    path = Path(config["cache"]["event_table"])
    if not path.exists():
        准备事件表(config)
    cached = dataset_module.载入事件表(path, trainable_only=False)
    grouping = cached["context_grouping"]
    if not grouping.eq(config["dataset"]["context_grouping"]).all():
        raise ValueError("事件缓存的语境分组方式与配置不一致。")
    semantic_context = config["dataset"].get("semantic_context")
    if semantic_context is not None:
        audit_path = path.with_suffix(".audit.json")
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            manifest_path = path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = manifest.get("audit", {})
        if audit.get("semantic_context") != semantic_context:
            raise ValueError("事件缓存的语义片段参数与配置不一致。")
    return path


def 物化所选输入(config, table, force_cache=False):
    """只物化传入事件表所需的文本向量与脑电记录。"""
    dataset_module.ensure_text_embedding_cache(
        table["normalized_word"],
        config["text_embedding"],
        config["cache"]["text_embeddings"],
    )
    dataset_module.确保记录缓存(
        table,
        config["dataset"],
        config["cache"]["eeg_dir"],
        force=force_cache,
    )


def 构建数据集(config, table, zero_eeg=False, zero_signal=False):
    zero_eeg = bool(zero_eeg or zero_signal)
    return _构建数据集_impl(config, table, zero_eeg=zero_eeg)


def _构建数据集_impl(config, table, zero_eeg=False):
    """构建与当前配置绑定的词事件数据集。"""
    return dataset_module.ChineseEEG2LittlePrinceWordDataset(
        table,
        config["dataset"],
        config["cache"]["eeg_dir"],
        config["cache"]["text_embeddings"],
        config["text_embedding"],
        zero_eeg=zero_eeg,
    )


def 设置脑编码器可训练(model, trainable):
    """冻结时同步固定脑编码器中的随机失活和批归一化状态。"""
    trainable = bool(trainable)
    for parameter in model.brain_encoder.parameters():
        parameter.requires_grad_(trainable)
    model.brain_encoder.train(trainable)


def 文件摘要(path):
    """计算预训练检查点的 SHA-256 摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 载入预训练脑编码器(
    model,
    checkpoint_path,
    config,
    channel_names,
    channel_positions,
    subject_ids,
    vocabulary,
):
    """只载入同一数据合同 CNN-only 检查点中的脑编码器。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "word_decoding/ChineseEEG2_LittlePrince":
        raise ValueError("预训练检查点不属于 ChineseEEG2《小王子》任务。")
    if bool(checkpoint.get("model_config", {}).get("use_transformer", True)):
        raise ValueError("预训练脑编码器必须来自 CNN-only 训练。")
    if checkpoint.get("text_embedding_config") != config["text_embedding"]:
        raise ValueError("预训练 CNN 与目标模型的文本向量合同不同。")

    source_contract = checkpoint.get("dataset_contract", {})
    checked_fields = (
        "split",
        "window_start_offset_seconds",
        "window_seconds",
        "eligibility_window_seconds",
        "source_sampling_rate_hz",
        "target_sampling_rate_hz",
        "training_vocabulary_policy",
        "evaluation_vocabulary_policy",
        "event_source_type",
        "timestamp_to_eeg_offset_seconds",
    )
    for field in checked_fields:
        if source_contract.get(field) != config["dataset"].get(field):
            raise ValueError(f"预训练 CNN 的数据合同字段不一致：{field}")
    if tuple(source_contract.get("evaluation_vocabulary", ())) != tuple(vocabulary):
        raise ValueError("预训练 CNN 与目标模型的冻结候选词不同。")
    if tuple(checkpoint.get("channel_names", ())) != tuple(channel_names):
        raise ValueError("预训练 CNN 与目标数据的脑电通道顺序不同。")
    source_positions = np.asarray(checkpoint.get("channel_positions", ()))
    target_positions = np.asarray(channel_positions)
    if source_positions.shape != target_positions.shape or not np.allclose(
        source_positions,
        target_positions,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("预训练 CNN 与目标数据的电极坐标不同。")
    if tuple(checkpoint.get("subject_ids", ())) != tuple(subject_ids):
        raise ValueError("预训练 CNN 与目标模型的受试者索引合同不同。")

    prefix = "brain_encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError("预训练检查点不包含脑编码器权重。")
    model.brain_encoder.load_state_dict(encoder_state, strict=True)
    metrics = checkpoint.get("metrics", {})
    return {
        "method": "cnn_only_best_checkpoint_brain_encoder_only",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": 文件摘要(checkpoint_path),
        "source_epoch": int(checkpoint["epoch"]),
        "source_optimizer_updates": checkpoint.get("optimizer_updates"),
        "source_macro_recall_at_10": metrics.get(
            "retrieval_acc10_vocab=chineseeeg2_littleprince50_macro"
        ),
        "loaded_loss_state": False,
        "loaded_transformer_state": False,
    }


def 移动批次(batch, device):
    """把模型输入张量移动到目标设备。"""
    return (
        batch["eeg"].to(device, non_blocking=True),
        batch["text_embedding"].to(device, non_blocking=True),
        batch["subject_index"].to(device, non_blocking=True),
        batch["sentence_index"].to(device, non_blocking=True),
    )


def 训练一轮(
    model,
    loss_module,
    loader,
    optimizer,
    scaler,
    device,
    config,
    maximum_updates=None,
    scheduler=None,
    starting_optimizer_updates=0,
    freeze_brain_encoder_updates=0,
    freeze_brain_encoder=None,
):
    """训练一轮；legacy 模式可继续限制成功 optimizer update 数。"""
    model.train()
    loss_module.train()
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    loss_sum = 0.0
    sample_count = 0
    update_count = 0
    frozen_update_count = 0
    joint_update_count = 0
    batches_seen = 0
    previous_frozen_state = None
    for batch in loader:
        if maximum_updates is not None and update_count >= int(maximum_updates):
            break
        encoder_is_frozen = (
            bool(freeze_brain_encoder)
            if freeze_brain_encoder is not None
            else int(starting_optimizer_updates) + update_count
            < int(freeze_brain_encoder_updates)
        )
        if encoder_is_frozen != previous_frozen_state:
            设置脑编码器可训练(model, not encoder_is_frozen)
            previous_frozen_state = encoder_is_frozen
        eeg, targets, subject_indices, sentence_indices = 移动批次(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            estimates = model(eeg, subject_indices, sentence_indices)
            loss = loss_module(estimates, targets)
        scaler.scale(loss).backward()
        max_grad_norm = float(config.get("max_grad_norm", 0.0))
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        previous_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        update_succeeded = not use_amp or float(scaler.get_scale()) >= previous_scale
        if update_succeeded:
            update_count += 1
            if encoder_is_frozen:
                frozen_update_count += 1
            else:
                joint_update_count += 1
            if scheduler is not None:
                scheduler.step()
        loss_sum += float(loss.detach()) * len(eeg)
        sample_count += len(eeg)
        batches_seen += 1
    if sample_count == 0:
        raise RuntimeError("这一轮没有读取任何训练批次。")
    return (
        loss_sum / sample_count,
        update_count,
        batches_seen,
        frozen_update_count,
        joint_update_count,
    )


@torch.inference_mode()
def 编码数据(model, loader, device, amp=True, subject_index_map=None):
    """编码一个完整数据划分，保留检索评估需要的标识。"""
    if bool(getattr(model, "prefix_revision_enabled", False)):
        return encode_word_loader(
            model,
            loader,
            device,
            amp=amp,
            subject_index_map=subject_index_map,
        )
    model.eval()
    predictions = []
    targets = []
    words = []
    event_ids = []
    recording_ids = []
    use_amp = bool(amp) and device.type == "cuda"
    for batch in loader:
        eeg, text_embedding, subject_indices, sentence_indices = 移动批次(
            batch, device
        )
        if subject_index_map is not None:
            from braindecoding.subject_generalization import remap_subject_indices

            subject_indices = remap_subject_indices(
                subject_indices, subject_index_map
            )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            estimate = model(eeg, subject_indices, sentence_indices)
        predictions.append(estimate.float().cpu())
        targets.append(text_embedding.float().cpu())
        words.extend(str(value) for value in batch["word"])
        event_ids.extend(str(value) for value in batch["event_id"])
        recording_ids.extend(str(value) for value in batch["recording_id"])
    return {
        "predictions": torch.cat(predictions),
        "targets": torch.cat(targets),
        "words": words,
        "event_ids": event_ids,
        "recording_ids": recording_ids,
    }


def 评估数据(
    model,
    loader,
    device,
    vocabulary,
    top_ks=(1, 10),
    amp=True,
    subject_index_map=None,
):
    """按冻结训练词表计算闭集检索指标。"""
    encoded = 编码数据(
        model,
        loader,
        device,
        amp=amp,
        subject_index_map=subject_index_map,
    )
    metrics = fixed_vocabulary_retrieval_metrics(
        encoded["predictions"],
        encoded["targets"],
        encoded["words"],
        vocabulary,
        top_ks=top_ks,
        vocabulary_name="chineseeeg2_littleprince50",
    )
    if "revision_predictions" in encoded:
        availability = encoded["revision_availability_masks"]
        maximum_delay = max(encoded["revision_predictions"])
        common_mask = availability[maximum_delay].numpy().astype(bool)
        if not common_mask.any():
            raise ValueError("验证集没有同时支持全部滚动回修延迟的目标词。")
        metrics["revision_common_event_count"] = int(common_mask.sum())
        words = np.asarray(encoded["words"])
        for delay, predictions in encoded["revision_predictions"].items():
            delay_metrics = fixed_vocabulary_retrieval_metrics(
                predictions,
                encoded["targets"],
                encoded["words"],
                vocabulary,
                top_ks=top_ks,
                vocabulary_name="chineseeeg2_littleprince50",
            )
            metrics.update(
                {
                    f"revision_delay{delay}_{key}": value
                    for key, value in delay_metrics.items()
                    if isinstance(value, (int, float))
                }
            )
            common_metrics = fixed_vocabulary_retrieval_metrics(
                predictions[common_mask],
                encoded["targets"][common_mask],
                words[common_mask].tolist(),
                vocabulary,
                top_ks=top_ks,
                vocabulary_name="chineseeeg2_littleprince50",
            )
            metrics.update(
                {
                    f"revision_common_delay{delay}_{key}": value
                    for key, value in common_metrics.items()
                    if isinstance(value, (int, float))
                }
            )
    return metrics, encoded


def 检查点内容(
    model,
    loss_module,
    config,
    epoch,
    metrics,
    vocabulary,
    dataset,
    optimizer_updates,
    optimizer=None,
    scheduler=None,
    initialization_audit=None,
    training_phase=None,
):
    """生成包含完整数据合同且可安全复核的检查点。"""
    payload = {
        "format_version": 1,
        "task": "word_decoding/ChineseEEG2_LittlePrince",
        "epoch": int(epoch),
        "optimizer_updates": int(optimizer_updates),
        "model_state": model.state_dict(),
        "loss_state": loss_module.state_dict(),
        "model_config": config["model"],
        "loss_config": config["loss"],
        "text_embedding_config": config["text_embedding"],
        "metrics": metrics,
        "channel_names": list(dataset.channel_names),
        "channel_positions": np.asarray(dataset.channel_positions).tolist(),
        "subject_ids": list(dataset.subject_ids),
        "dataset_contract": {
            "subject_generalization": config.get("subject_generalization"),
            "context_grouping": config["dataset"]["context_grouping"],
            "semantic_context": config["dataset"].get("semantic_context"),
            "root": config["dataset"]["root"],
            "alignment_path": config["dataset"]["alignment_path"],
            "actual_reading_sources": config["dataset"].get(
                "actual_reading_sources"
            ),
            "subjects": config["dataset"]["subjects"],
            "excluded_chapters": config["dataset"].get("excluded_chapters", []),
            "split": config["dataset"]["split"],
            "window_start_offset_seconds": config["dataset"].get(
                "window_start_offset_seconds", 0.0
            ),
            "window_seconds": config["dataset"]["window_seconds"],
            "eligibility_window_seconds": config["dataset"].get(
                "eligibility_window_seconds", config["dataset"]["window_seconds"]
            ),
            "source_sampling_rate_hz": config["dataset"][
                "source_sampling_rate_hz"
            ],
            "target_sampling_rate_hz": config["dataset"][
                "target_sampling_rate_hz"
            ],
            "training_vocabulary_policy": config["dataset"][
                "training_vocabulary_policy"
            ],
            "evaluation_vocabulary_policy": config["dataset"][
                "evaluation_vocabulary_policy"
            ],
            "event_source_type": config["dataset"]["event_source_type"],
            "timestamp_to_eeg_offset_seconds": config["dataset"].get("timestamp_to_eeg_offset_seconds"),
            "evaluation_vocabulary": list(vocabulary),
            "test_eeg_opened": False,
        },
        "provenance": config.get("provenance", {}),
    }
    if initialization_audit is not None:
        payload["initialization_audit"] = initialization_audit
    if training_phase is not None:
        payload["training_phase"] = training_phase
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    return payload


def 执行训练(config, smoke=False, save=True, force_cache=False):
    """在训练与验证集上训练；active v2 使用 epoch/patience 生命周期。"""
    training_config = dict(config["training"])
    staged_training = bool(
        training_config.get("pretrained_brain_encoder_checkpoint")
    )
    if int(config["model"]["embedding_dimension"]) != int(
        config["text_embedding"]["embedding_dimension"]
    ):
        raise ValueError("模型输出维数与文本目标维数不一致。")
    if smoke:
        training_config["epochs"] = 2 if staged_training else 1
        training_config["patience"] = training_config["epochs"]
        smoke_batch_limit = (
            int(training_config.get("smoke_rows", 64))
            if training_config.get("preserve_complete_groups", False)
            else 16
        )
        training_config["batch_size"] = min(
            int(training_config["batch_size"]), smoke_batch_limit
        )
        if staged_training:
            training_config["freeze_brain_encoder_epochs"] = 1

    max_updates = training_config.get("max_updates")
    if max_updates is not None:
        max_updates = int(max_updates)
    minimum_updates = int(
        training_config.get("minimum_updates_before_early_stopping", 0)
    )
    scheduler_horizon = int(
        training_config.get(
            "scheduler_total_updates",
            max_updates if max_updates is not None else training_config["epochs"],
        )
    )
    if max_updates is not None and (
        max_updates <= 0 or minimum_updates < 0 or minimum_updates > max_updates
    ):
        raise ValueError("优化器更新预算无效。")
    if max_updates is not None and scheduler_horizon < max_updates:
        raise ValueError("余弦调度总步数不能小于训练更新预算。")
    freeze_brain_encoder_updates = int(
        training_config.get("freeze_brain_encoder_updates", 0)
    )
    if freeze_brain_encoder_updates < 0:
        raise ValueError("冻结 CNN 的更新次数不能为负数。")
    freeze_brain_encoder_epochs = int(
        training_config.get("freeze_brain_encoder_epochs", 0)
    )
    if freeze_brain_encoder_epochs < 0:
        raise ValueError("冻结 brain encoder 的 epoch 数不能为负数。")
    if staged_training and max_updates is not None:
        if not bool(config["model"].get("use_transformer", True)):
            raise ValueError("CNN 预热的第二阶段必须启用 Transformer。")
        if not 0 < freeze_brain_encoder_updates < max_updates:
            raise ValueError("第二阶段必须先冻结 CNN，再留出联合训练更新。")
    elif staged_training:
        if not 0 < freeze_brain_encoder_epochs < int(training_config["epochs"]):
            raise ValueError("warm-start context 的冻结轮数必须大于零且小于总轮数。")
        if freeze_brain_encoder_updates:
            raise ValueError("epoch 生命周期不能同时使用 update 冻结字段。")
    elif freeze_brain_encoder_updates or freeze_brain_encoder_epochs:
        raise ValueError("没有预训练 CNN 检查点时不能设置冻结阶段。")

    set_seed(int(training_config["seed"]))
    device = choose_device(training_config.get("device", "auto"))
    event_path = 确保事件表(config)
    subjects = config["dataset"].get("subjects")
    train_table = dataset_module.载入事件表(
        event_path, split="train", subjects=subjects
    )
    protocol = config.get("subject_generalization")
    if protocol:
        from braindecoding.subject_generalization import split_inner_train_validation

        train_table, val_table = split_inner_train_validation(
            train_table, protocol
        )
    else:
        val_table = dataset_module.载入事件表(
            event_path, split="val", subjects=subjects
        )
    if smoke:
        maximum = int(training_config.get("smoke_rows", 64))
        if protocol:
            from braindecoding.subject_generalization import limit_rows_by_subject

            train_table = limit_rows_by_subject(train_table, maximum, subjects)
            val_table = limit_rows_by_subject(val_table, maximum, subjects)
        else:
            train_table = limit_rows(train_table, maximum)
            val_table = limit_rows(val_table, maximum)

    物化所选输入(
        config,
        pd.concat([train_table, val_table], ignore_index=True),
        force_cache=force_cache,
    )
    train_dataset = 构建数据集(config, train_table)
    val_dataset = 构建数据集(config, val_table)
    train_loader, train_sampler = make_loader(
        train_dataset, training_config, shuffle=True
    )
    val_loader, _ = make_loader(val_dataset, training_config, shuffle=False)
    vocabulary = dataset_module.载入候选词(event_path)

    model = build_brain_embedding_model(
        train_dataset.channel_count,
        train_dataset.channel_positions,
        train_dataset.subject_count,
        config["model"],
    ).to(device)
    loss_module = build_siglip_loss(config["loss"]).to(device)
    initialization_audit = None
    if staged_training:
        initialization_audit = 载入预训练脑编码器(
            model,
            training_config["pretrained_brain_encoder_checkpoint"],
            config,
            train_dataset.channel_names,
            train_dataset.channel_positions,
            train_dataset.subject_ids,
            vocabulary,
        )
    optimizer = build_adamw_for_modules([model, loss_module], training_config)
    scheduler = build_cosine_annealing_scheduler(
        optimizer,
        scheduler_horizon,
        minimum_learning_rate=float(
            training_config.get("minimum_learning_rate", 0.0)
        ),
    )
    use_amp = bool(training_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    top_ks = tuple(int(value) for value in training_config.get("top_ks", [1, 10]))
    base_selection_metric = (
        f"retrieval_acc{max(top_ks)}_vocab="
        "chineseeeg2_littleprince50_macro"
    )
    rolling_revision = (
        bool(getattr(model, "prefix_revision_enabled", False))
        and getattr(model, "prefix_revision_protocol", None) == "local_window"
    )
    selection_metric = base_selection_metric
    if rolling_revision:
        selection_metric = (
            f"revision_common_delay{max(model.prefix_revision_horizons)}_"
            f"{base_selection_metric}"
        )
    configured_selection_metric = config.get("evaluation", {}).get(
        "selection_metric"
    )
    if (
        configured_selection_metric is not None
        and configured_selection_metric != selection_metric
    ):
        raise ValueError("ChineseEEG2 validation checkpoint selection metric 合同漂移。")
    output_dir = Path(training_config["output_dir"])
    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

    best_score = -float("inf")
    best_epoch = None
    best_metrics = None
    best_model_state = None
    best_loss_state = None
    epochs_without_improvement = 0
    optimizer_updates = 0
    frozen_encoder_updates = 0
    joint_updates = 0
    stop_reason = "maximum_epochs_reached"
    history = []
    for epoch in range(int(training_config["epochs"])):
        train_sampler.set_epoch(epoch)
        encoder_is_frozen = (
            staged_training
            and max_updates is None
            and brain_encoder_frozen_for_epoch(epoch, freeze_brain_encoder_epochs)
        )
        if max_updates is None:
            设置脑编码器可训练(model, not encoder_is_frozen)
        train_diagnostics = None
        if rolling_revision:
            if max_updates is not None:
                raise ValueError("局部窗口滚动回修当前只支持 epoch 训练生命周期。")
            (
                train_loss,
                epoch_updates,
                batches_seen,
                train_diagnostics,
            ) = train_word_epoch(
                model,
                loss_module,
                train_loader,
                optimizer,
                scaler,
                device,
                training_config,
                freeze_brain_encoder=encoder_is_frozen,
                return_stats=True,
                return_diagnostics=True,
                epoch_index=epoch,
            )
            epoch_frozen_updates = epoch_updates if encoder_is_frozen else 0
            epoch_joint_updates = 0 if encoder_is_frozen else epoch_updates
        else:
            (
                train_loss,
                epoch_updates,
                batches_seen,
                epoch_frozen_updates,
                epoch_joint_updates,
            ) = 训练一轮(
                model,
                loss_module,
                train_loader,
                optimizer,
                scaler,
                device,
                training_config,
                (
                    max_updates - optimizer_updates
                    if max_updates is not None
                    else None
                ),
                scheduler if max_updates is not None else None,
                starting_optimizer_updates=optimizer_updates,
                freeze_brain_encoder_updates=freeze_brain_encoder_updates,
                freeze_brain_encoder=(
                    encoder_is_frozen if max_updates is None else None
                ),
            )
        optimizer_updates += epoch_updates
        frozen_encoder_updates += epoch_frozen_updates
        joint_updates += epoch_joint_updates
        validation_metrics, _ = 评估数据(
            model,
            val_loader,
            device,
            vocabulary,
            top_ks=top_ks,
            amp=training_config.get("amp", True),
        )
        if max_updates is None:
            scheduler.step()
        score = float(validation_metrics[selection_metric])
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_optimizer_updates": int(epoch_updates),
            "optimizer_updates": int(optimizer_updates),
            "batches_seen": int(batches_seen),
            "frozen_encoder_optimizer_updates": int(epoch_frozen_updates),
            "joint_optimizer_updates": int(epoch_joint_updates),
            "cumulative_optimizer_updates": int(optimizer_updates),
            "brain_encoder_frozen": bool(
                encoder_is_frozen
                if max_updates is None
                else optimizer_updates < freeze_brain_encoder_updates
            ),
            "training_phase_at_epoch_end": (
                "transformer_only"
                if (
                    encoder_is_frozen
                    if max_updates is None
                    else optimizer_updates < freeze_brain_encoder_updates
                )
                else "joint"
            ),
            "train_prefix_revision_loss": (
                train_diagnostics["prefix_revision_loss"]
                if train_diagnostics is not None
                else None
            ),
            "train_prefix_revision_supervision_count": (
                train_diagnostics["prefix_revision_supervision_count"]
                if train_diagnostics is not None
                else 0
            ),
            "train_prefix_revision_delay_counts": (
                train_diagnostics["prefix_revision_delay_counts"]
                if train_diagnostics is not None
                else {}
            ),
            "train_prefix_revision_delay_losses": (
                train_diagnostics["prefix_revision_delay_losses"]
                if train_diagnostics is not None
                else {}
            ),
            **validation_metrics,
        }
        history.append(epoch_record)
        print(
            f"第 {epoch + 1} 轮：loss={train_loss:.5f}，"
            f"验证集宏平均 Top-{max(top_ks)}={score:.6f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = validation_metrics
            epochs_without_improvement = 0
            if save:
                save_checkpoint(
                    output_dir / "best.pt",
                    检查点内容(
                        model,
                        loss_module,
                        config,
                        epoch + 1,
                        validation_metrics,
                        vocabulary,
                        train_dataset,
                        optimizer_updates,
                        initialization_audit=initialization_audit,
                        training_phase={
                            "freeze_brain_encoder_epochs": freeze_brain_encoder_epochs,
                            "frozen_encoder_optimizer_updates": frozen_encoder_updates,
                            "joint_optimizer_updates": joint_updates,
                        },
                    ),
                )
            else:
                best_model_state = cpu_state_dict(model)
                best_loss_state = cpu_state_dict(loss_module)
        else:
            epochs_without_improvement += 1

        if save and bool(training_config.get("save_last_checkpoint", True)):
            save_checkpoint(
                output_dir / "last.pt",
                检查点内容(
                    model,
                    loss_module,
                    config,
                    epoch + 1,
                    validation_metrics,
                    vocabulary,
                    train_dataset,
                    optimizer_updates,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    initialization_audit=initialization_audit,
                    training_phase={
                        "freeze_brain_encoder_epochs": freeze_brain_encoder_epochs,
                        "frozen_encoder_optimizer_updates": frozen_encoder_updates,
                        "joint_optimizer_updates": joint_updates,
                    },
                ),
            )
        if max_updates is not None and optimizer_updates >= max_updates:
            stop_reason = "update_budget_reached"
            break
        if validation_patience_exhausted(
            epochs_without_improvement, training_config["patience"]
        ):
            stop_reason = "early_stopping"
            break

    if best_epoch is None:
        raise RuntimeError("训练没有产生有效的验证指标。")
    if max_updates is not None and optimizer_updates < minimum_updates:
        raise RuntimeError(
            f"训练仅完成 {optimizer_updates} 次更新，低于要求的 {minimum_updates} 次。"
        )
    if not save and best_model_state is not None:
        model.load_state_dict(best_model_state)
        loss_module.load_state_dict(best_loss_state)

    summary = {
        "status": "smoke_completed" if smoke else "training_completed",
        "device": str(device),
        "test_eeg_opened": False,
        "event_table": str(event_path),
        "train_rows": len(train_dataset),
        "validation_rows": len(val_dataset),
        "eeg_shape": [train_dataset.channel_count, train_dataset.window_samples],
        "subject_count": train_dataset.subject_count,
        "model_parameter_count": parameter_count(model),
        "loss_parameter_count": parameter_count(loss_module),
        "initialization_audit": initialization_audit,
        "freeze_brain_encoder_updates": freeze_brain_encoder_updates,
        "freeze_brain_encoder_epochs": freeze_brain_encoder_epochs,
        "frozen_encoder_optimizer_updates": int(frozen_encoder_updates),
        "joint_optimizer_updates": int(joint_updates),
        "rolling_revision": (
            {
                "enabled": True,
                "protocol": "local_window_physical_truncation",
                "history_radius": int(model.prefix_revision_history_radius),
                "horizons": list(model.prefix_revision_horizons),
                "loss": "mean_of_independent_horizon_siglip_losses",
                "cutoff_policy": training_config.get(
                    "revision_cutoff_policy", "sample_one"
                ),
                "negative_bank": "all_text_targets_in_original_batch",
                "shared_transformer": True,
                "parameter_count_added": 0,
            }
            if rolling_revision
            else None
        ),
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_validation": best_metrics,
        "optimizer_updates": int(optimizer_updates),
        "target_updates": max_updates,
        "scheduler_total_updates": scheduler_horizon,
        "batch_size_limit": int(training_config["batch_size"]),
        "estimated_batches_per_epoch": len(train_loader),
        "stop_reason": stop_reason,
        "epochs_without_improvement": int(epochs_without_improvement),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "history": history,
        "test_status": "locked_not_evaluated",
    }
    if save:
        saved_summary = (
            build_training_summary(config, summary)
            if "experiment" in config
            else summary
        )
        save_json(output_dir / "training_summary.json", saved_summary)
    return summary


def 打印训练摘要(summary):
    """输出适合命令行查看的简要训练状态。"""
    print("ChineseEEG2《小王子》训练完成")
    print(f"  设备：{summary['device']}")
    print(f"  训练/验证事件：{summary['train_rows']} / {summary['validation_rows']}")
    if summary.get("initialization_audit") is not None:
        print(
            "  分阶段更新："
            f"冻结 CNN {summary['frozen_encoder_optimizer_updates']}，"
            f"联合训练 {summary['joint_optimizer_updates']}"
        )
    print(
        f"  最佳轮次：{summary['best_epoch']}，"
        f"{summary['selection_metric']}={summary['best_score']:.6f}"
    )
    print(
        f"  优化器更新：{summary['optimizer_updates']} / "
        f"{summary['target_updates'] or 'epoch-scheduled'}"
    )
    print("  测试集：已锁定，未评估")


def 解析参数(argv=None):
    """解析准备、训练和缓存物化参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--prepare", action="store_true", help="构建并审计事件表")
    parser.add_argument(
        "--with-eeg-cache", action="store_true", help="物化所选脑电记录缓存"
    )
    parser.add_argument(
        "--with-text-embeddings", action="store_true", help="物化所选文本向量"
    )
    parser.add_argument(
        "--include-test", action="store_true", help="明确允许准备测试集输入"
    )
    parser.add_argument("--force-cache", action="store_true", help="强制重建脑电缓存")
    parser.add_argument("--smoke", action="store_true", help="执行一次小规模真实训练")
    parser.add_argument("--no-save", action="store_true", help="不写入模型检查点")
    return parser.parse_args(argv)


def main(argv=None):
    """执行命令行请求。"""
    args = 解析参数(argv)
    config = 载入配置(args.config)
    if args.prepare:
        summary = 准备事件表(
            config,
            with_eeg_cache=args.with_eeg_cache,
            with_text_embeddings=args.with_text_embeddings,
            include_test=args.include_test,
            force_cache=args.force_cache,
        )
        print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
        return
    run_output = None
    if "experiment" in config and not args.no_save:
        command_args = list(argv) if argv is not None else sys.argv[1:]
        run_output, _ = initialize_run_directory(
            config,
            [sys.executable, str(Path(__file__).resolve()), *command_args],
        )
    try:
        summary = 执行训练(
            config,
            smoke=args.smoke,
            save=not args.no_save,
            force_cache=args.force_cache,
        )
    except BaseException:
        if run_output is not None:
            update_run_status(run_output, "failed")
        raise
    if run_output is not None:
        update_run_status(run_output, "completed")
    打印训练摘要(summary)


# 标准英文接口别名（与 pallier2025.train 保持一致，同时保留中文函数名向后兼容）
load_config = 载入配置
prepare_event_table = 准备事件表
ensure_event_table = 确保事件表
ensure_canonical_inputs = 物化所选输入
build_dataset = 构建数据集
set_brain_encoder_trainable = 设置脑编码器可训练
file_sha256 = 文件摘要
load_pretrained_brain_encoder = 载入预训练脑编码器
move_batch_to_device = 移动批次
train_one_epoch = 训练一轮
encode_loader = 编码数据
evaluate_loader = 评估数据
checkpoint_payload = 检查点内容
run_training = 执行训练
print_training_summary = 打印训练摘要
parse_args = 解析参数


def loader_group_column(config) -> str:
    return "sentence_uid"


def frozen_vocabulary(event_table_path=None):
    if event_table_path is None:
        cfg = load_config()
        event_table_path = cfg["cache"]["event_table"]
    return tuple(dataset_module.load_candidate_vocabulary(event_table_path))


if __name__ == "__main__":
    main()
