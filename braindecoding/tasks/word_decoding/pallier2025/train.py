"""Pallier2025 多受试者词解码训练入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends, project_path
from braindecoding.data import pallier2025 as dataset_module
from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval_metrics
from braindecoding.experiment import file_sha256, resolve_experiment_config
from braindecoding.losses import build_siglip_loss
from braindecoding.models import build_brain_embedding_model
from braindecoding.optimizers import (
    build_adamw_for_modules,
    build_cosine_annealing_scheduler,
)
from braindecoding.results import build_training_summary, load_vocabulary_assets
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
    encode_loader,
    make_loader,
    move_batch,
    train_one_epoch,
    validation_patience_exhausted,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "word_decoding"
    / "pallier2025"
    / "sub01-10"
    / "main_word.yaml"
)
VOCABULARY_ASSET_DIRECTORY = (
    PROJECT_ROOT / "data" / "manifests" / "pallier2025"
)
SELECTION_VOCABULARY_SIZE = 50
SELECTION_VOCABULARY_NAME = "pallier2025_50"


def load_config(path=None):
    """载入 canonical Pallier 实验配置并解析 derived 路径。"""
    config_path = project_path(path or DEFAULT_CONFIG)
    config = resolve_experiment_config(load_yaml_with_extends(config_path))
    for key in ("event_table", "meg_dir", "text_embeddings"):
        config["cache"][key] = str(project_path(config["cache"][key]))
    for key in ("split_manifest", "qc_artifact", "layout_path"):
        if config["dataset"].get(key):
            config["dataset"][key] = str(project_path(config["dataset"][key]))
    config["training"]["output_dir"] = str(
        project_path(config["training"]["output_dir"])
    )
    return config


def ensure_canonical_inputs(config):
    """只验证冻结数据产品；训练入口不得回退或重新构建旧缓存。"""
    from braindecoding.data.build import check_dataset

    status = check_dataset("pallier2025")
    if status["status"] != "complete":
        raise RuntimeError("Pallier2025 canonical derived data 尚未完整。")
    for key in ("event_table", "meg_dir", "text_embeddings"):
        if not Path(config["cache"][key]).exists():
            raise FileNotFoundError(f"Pallier2025 canonical 数据产品缺失：{key}")
    return Path(config["cache"]["event_table"])


def build_dataset(
    config,
    table,
    zero_meg=False,
    zero_signal=False,
    temporal_shift_seconds: float = 0.0,
):
    return dataset_module.Pallier2025WordDataset(
        table,
        config["dataset"],
        config["cache"]["meg_dir"],
        config["cache"]["text_embeddings"],
        config["text_embedding"],
        zero_meg=bool(zero_meg or zero_signal),
        temporal_shift_seconds=float(temporal_shift_seconds),
        allow_subject_subset=bool(
            config.get("subject_generalization")
            or config["dataset"].get("allow_subject_subset")
        ),
    )


def set_brain_encoder_trainable(model, trainable):
    """冻结时同步固定 brain encoder 的 BN 与 dropout 状态。"""
    trainable = bool(trainable)
    for parameter in model.brain_encoder.parameters():
        parameter.requires_grad_(trainable)
    model.brain_encoder.train(trainable)


def loader_group_column(config) -> str:
    """上下文模型使用 recording-local key；word 模型保留原批次组织。"""
    if not bool(config["model"].get("use_transformer", False)):
        return "sentence_uid"
    expected = dataset_module.RUNTIME_CONTEXT_GROUPING
    actual = config["dataset"].get("runtime_context_grouping")
    if actual != expected:
        raise ValueError(
            f"Pallier2025 runtime context 合同不一致：预期 {expected}，实际 {actual}"
        )
    return dataset_module.RUNTIME_CONTEXT_COLUMN


def frozen_vocabulary(size=SELECTION_VOCABULARY_SIZE):
    manifests = load_vocabulary_assets(VOCABULARY_ASSET_DIRECTORY)
    return tuple(manifests[int(size)]["vocabulary"])


def load_pretrained_brain_encoder(
    model,
    checkpoint_path,
    config,
    channel_names,
    channel_positions,
):
    """只载入同一 Pallier 合同 CNN-only 检查点中的脑编码器。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "word_decoding/Pallier2025":
        raise ValueError(f"不是 Pallier2025 词解码检查点：{checkpoint_path}")
    if checkpoint.get("dataset") != "pallier2025":
        raise ValueError(f"Pallier2025 检查点数据集身份不一致：{checkpoint_path}")

    source_model = checkpoint.get("model_config", {})
    if bool(source_model.get("use_transformer", True)):
        raise ValueError("Pallier2025 warm-start 必须来自 CNN-only 检查点。")
    if source_model.get("embedding_dimension") != config["model"].get(
        "embedding_dimension"
    ) or source_model.get("conv") != config["model"].get("conv"):
        raise ValueError("Pallier2025 warm-start 的脑编码器结构合同不一致。")
    if checkpoint.get("text_embedding_config") != config["text_embedding"]:
        raise ValueError("Pallier2025 warm-start 的文本表示合同不一致。")

    event_manifest = json.loads(
        (PROJECT_ROOT / "data/cache/derived/pallier2025/events/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    text_manifest = json.loads(
        (
            PROJECT_ROOT
            / "data/cache/derived/pallier2025/text/t5_large_layer_0_5/manifest.json"
        ).read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (VOCABULARY_ASSET_DIRECTORY / "run_split.json").read_text(encoding="utf-8")
    )
    source_dataset = checkpoint.get("dataset_contract", {})
    target_protocol = config.get("subject_generalization")
    source_protocol = source_dataset.get("subject_generalization")
    if target_protocol is not None or source_protocol is not None:
        if target_protocol is None or source_protocol is None:
            raise ValueError("Pallier2025 warm-start 受试者泛化合同缺失。")
        if source_protocol.get("architecture") != "word" or target_protocol.get(
            "architecture"
        ) != "context":
            raise ValueError("Pallier2025 warm-start 必须从相应 word 折进入 context。")
        for field in (
            "schema_version",
            "model_mode",
            "fold",
            "held_out_subject",
            "inner_val_units",
            "inner_split_salt",
            "inner_validation_fraction",
            "outer_split",
            "training_subjects",
            "evaluation_subjects",
            "test_data_opened",
        ):
            if source_protocol.get(field) != target_protocol.get(field):
                raise ValueError(
                    f"Pallier2025 warm-start 受试者泛化合同不一致：{field}"
                )
    expected_contract = {
        "dataset": "pallier2025",
        "subjects": list(config["dataset"]["subjects"]),
        "subject_order": list(config["dataset"]["subjects"]),
        "runs": list(config["dataset"]["runs"]),
        "split": split_manifest["assignments"],
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "event_table_sha256": event_manifest["event_table_sha256"],
        "window_contract_version": config["dataset"]["window_contract_version"],
        "window_start_offset_seconds": config["dataset"][
            "window_start_offset_seconds"
        ],
        "window_seconds": config["dataset"]["window_seconds"],
        "eligibility_window_seconds": config["dataset"][
            "eligibility_window_seconds"
        ],
        "baseline_seconds": config["dataset"]["baseline_seconds"],
        "clamp": config["dataset"]["clamp"],
        "target_sampling_rate_hz": config["dataset"]["target_sampling_rate_hz"],
        "channel_names_sha256": config["dataset"]["channel_names_sha256"],
        "training_vocabulary_policy": config["dataset"][
            "training_vocabulary_policy"
        ],
        "evaluation_vocabulary_policy": config["dataset"][
            "evaluation_vocabulary_policy"
        ],
        "text_content_sha256": text_manifest["content_sha256"],
    }
    for field, expected in expected_contract.items():
        if source_dataset.get(field) != expected:
            raise ValueError(f"Pallier2025 warm-start 数据合同不一致：{field}")

    if tuple(checkpoint.get("channel_names", ())) != tuple(channel_names):
        raise ValueError("Pallier2025 warm-start 的 MEG 通道顺序不一致。")
    source_positions = np.asarray(checkpoint.get("channel_positions", ()))
    target_positions = np.asarray(channel_positions)
    if source_positions.shape != target_positions.shape or not np.allclose(
        source_positions,
        target_positions,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("Pallier2025 warm-start 的通道位置合同不一致。")

    prefix = "brain_encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError("Pallier2025 warm-start 检查点不包含 brain_encoder 参数。")
    model.brain_encoder.load_state_dict(encoder_state, strict=True)
    source_metrics = checkpoint.get("metrics", {})
    return {
        "method": "cnn_only_best_checkpoint_brain_encoder_only",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_epoch": int(checkpoint["epoch"]),
        "source_update": int(checkpoint.get("optimizer_updates", 0)),
        "source_macro_r_at_10": source_metrics.get(
            "retrieval_acc10_vocab=pallier2025_50_macro"
        ),
        "loaded_loss_state": False,
        "loaded_transformer_state": False,
    }


def evaluate_loader(
    model,
    loader,
    device,
    top_ks=(1, 10),
    amp=True,
    subject_index_map=None,
):
    encoded = encode_loader(
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
        frozen_vocabulary(),
        top_ks=top_ks,
        vocabulary_name=SELECTION_VOCABULARY_NAME,
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
                frozen_vocabulary(),
                top_ks=top_ks,
                vocabulary_name=SELECTION_VOCABULARY_NAME,
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
                frozen_vocabulary(),
                top_ks=top_ks,
                vocabulary_name=SELECTION_VOCABULARY_NAME,
            )
            metrics.update(
                {
                    f"revision_common_delay{delay}_{key}": value
                    for key, value in common_metrics.items()
                    if isinstance(value, (int, float))
                }
            )
    return metrics, encoded


def train_one_epoch_with_update_budget(
    model,
    loss_module,
    loader,
    optimizer,
    scaler,
    device,
    config,
    maximum_optimizer_updates,
    *,
    scheduler,
):
    """执行固定数量的成功 optimizer updates，并保持 AMP overflow 语义。"""
    model.train()
    loss_module.train()
    use_amp = bool(config.get("amp", True)) and device.type == "cuda"
    loss_sum = 0.0
    sample_count = 0
    optimizer_updates = 0
    batches_seen = 0
    for batch in loader:
        if optimizer_updates >= int(maximum_optimizer_updates):
            break
        meg, targets, subject_indices, sentence_indices = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            estimates = model(meg, subject_indices, sentence_indices)
            loss = loss_module(estimates, targets)
        scaler.scale(loss).backward()
        max_grad_norm = float(config.get("max_grad_norm", 0.0))
        if max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        previous_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        step_succeeded = not use_amp or float(scaler.get_scale()) >= previous_scale
        if step_succeeded:
            optimizer_updates += 1
            scheduler.step()
        loss_sum += float(loss.detach()) * len(meg)
        sample_count += len(meg)
        batches_seen += 1
    if sample_count == 0:
        raise RuntimeError("未读取任何 Pallier2025 训练 batch。")
    return loss_sum / sample_count, optimizer_updates, batches_seen


def checkpoint_payload(
    model,
    loss_module,
    config,
    epoch,
    metrics,
    channel_names,
    channel_positions,
    *,
    optimizer_updates,
    optimizer=None,
    scheduler=None,
    initialization_audit=None,
    training_phase=None,
):
    """构造包含完整 Pallier 科学合同的 checkpoint payload。"""
    event_manifest = json.loads(
        (PROJECT_ROOT / "data/cache/derived/pallier2025/events/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    signal_manifest = json.loads(
        (PROJECT_ROOT / "data/cache/derived/pallier2025/signals/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    text_manifest = json.loads(
        (
            PROJECT_ROOT
            / "data/cache/derived/pallier2025/text/t5_large_layer_0_5/manifest.json"
        ).read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (VOCABULARY_ASSET_DIRECTORY / "run_split.json").read_text(encoding="utf-8")
    )
    vocabulary_manifests = load_vocabulary_assets(VOCABULARY_ASSET_DIRECTORY)
    payload = {
        "format_version": 1,
        "task": "word_decoding/Pallier2025",
        "dataset": "pallier2025",
        "epoch": int(epoch),
        "optimizer_updates": int(optimizer_updates),
        "model_state": model.state_dict(),
        "loss_state": loss_module.state_dict(),
        "model_config": config["model"],
        "loss_config": config["loss"],
        "text_embedding_config": config["text_embedding"],
        "metrics": metrics,
        "channel_names": list(channel_names),
        "channel_positions": np.asarray(channel_positions).tolist(),
        "dataset_contract": {
            "subject_generalization": config.get("subject_generalization"),
            "dataset": "pallier2025",
            "subjects": list(config["dataset"]["subjects"]),
            "subject_order": list(config["dataset"]["subjects"]),
            "runs": list(config["dataset"]["runs"]),
            "split_manifest": config["dataset"]["split_manifest"],
            "split": split_manifest["assignments"],
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "event_table_sha256": event_manifest["event_table_sha256"],
            "window_contract_version": config["dataset"][
                "window_contract_version"
            ],
            "window_start_offset_seconds": config["dataset"][
                "window_start_offset_seconds"
            ],
            "window_seconds": config["dataset"]["window_seconds"],
            "eligibility_window_seconds": config["dataset"][
                "eligibility_window_seconds"
            ],
            "baseline_seconds": config["dataset"]["baseline_seconds"],
            "clamp": config["dataset"]["clamp"],
            "target_sampling_rate_hz": config["dataset"][
                "target_sampling_rate_hz"
            ],
            "channel_names_sha256": config["dataset"]["channel_names_sha256"],
            "recording_count": signal_manifest["recording_count"],
            "context_grouping": config["dataset"]["context_grouping"],
            "runtime_context_grouping": config["dataset"].get(
                "runtime_context_grouping",
                "not_used_by_word_model",
            ),
            "max_context_words": config["dataset"]["max_context_words"],
            "training_vocabulary_policy": config["dataset"][
                "training_vocabulary_policy"
            ],
            "evaluation_vocabulary_policy": config["dataset"][
                "evaluation_vocabulary_policy"
            ],
            "text_content_sha256": text_manifest["content_sha256"],
            "vocabulary_manifest_sha256": {
                str(size): manifest["manifest_sha256"]
                for size, manifest in vocabulary_manifests.items()
            },
        },
        "provenance": config.get("provenance", {}),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if initialization_audit is not None:
        payload["initialization_audit"] = initialization_audit
    if training_phase is not None:
        payload["training_phase"] = training_phase
    return payload


def run_training(config, smoke=False, save=True, force_cache=False):
    """训练 Pallier word/context；只消费 train 与 validation canonical 数据。"""
    del force_cache  # Pallier canonical 训练不允许重建或回退数据缓存。
    training_config = dict(config["training"])
    warm_start = bool(training_config.get("pretrained_brain_encoder_checkpoint"))
    allowed_warm_starts = {
        "main_context_warmstart": "main_word",
        "main_context_causal_transformer": "main_word",
        "subject_independent_context_warmstart": "subject_independent_word",
        **{f"local_window_r{radius}": "main_word" for radius in (0, 1, 2, 4, 8)},
        "local_window_m2_p0": "main_word",
        "local_window_m2_p1": "main_word",
        "local_window_rolling_revision_k2_d2": "main_word",
    }
    expected_warm_start = allowed_warm_starts.get(config["experiment"]["id"])
    protocol = config.get("subject_generalization")
    if protocol is not None:
        mode = protocol.get("model_mode")
        suffix = "_smoke" if config["experiment"]["id"].endswith("_smoke") else ""
        if (
            mode in {"shared", "independent", "residual", "no_projection"}
            and protocol.get("architecture") == "context"
            and config["experiment"]["id"] == f"{mode}_context{suffix}"
        ):
            expected_warm_start = f"{mode}_word{suffix}"
    if warm_start and training_config.get("warm_start_from") != expected_warm_start:
        raise ValueError("Pallier2025 context 必须从同一实验族的 CNN-only 检查点 warm-start。")
    if warm_start and not bool(config["model"].get("use_transformer", False)):
        raise ValueError("Pallier2025 warm-start 只用于启用 Transformer 的 context 条件。")
    if int(config["model"]["embedding_dimension"]) != int(
        config["text_embedding"]["embedding_dimension"]
    ):
        raise ValueError("Pallier2025 model 与 T5 target 维数不一致。")
    if smoke:
        training_config["epochs"] = 2 if warm_start else 1
        training_config["patience"] = training_config["epochs"]
        smoke_batch_limit = (
            int(training_config.get("smoke_rows", 64))
            if bool(training_config.get("preserve_complete_groups", False))
            else 16
        )
        training_config["batch_size"] = min(
            int(training_config["batch_size"]), smoke_batch_limit
        )
        if training_config.get("max_updates") is not None:
            training_config["max_updates"] = 2 if warm_start else 1
            training_config["minimum_updates_before_early_stopping"] = training_config["max_updates"]
        elif warm_start:
            training_config["freeze_brain_encoder_epochs"] = 1

    set_seed(int(training_config["seed"]))
    device = choose_device(training_config.get("device", "auto"))
    event_path = ensure_canonical_inputs(config)
    subjects = config["dataset"]["subjects"]
    train_table = dataset_module.load_event_table(
        event_path,
        split="train",
        subjects=subjects,
        trainable_only=True,
    )
    protocol = config.get("subject_generalization")
    if protocol:
        from braindecoding.subject_generalization import split_inner_train_validation

        train_table, val_table = split_inner_train_validation(
            train_table, protocol
        )
    else:
        val_table = dataset_module.load_event_table(
            event_path,
            split="val",
            subjects=subjects,
            trainable_only=True,
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

    train_dataset = build_dataset(config, train_table)
    val_dataset = build_dataset(config, val_table)
    group_column = loader_group_column(config)
    train_loader, train_sampler = make_loader(
        train_dataset,
        training_config,
        shuffle=True,
        group_column=group_column,
    )
    val_loader, _ = make_loader(
        val_dataset,
        training_config,
        shuffle=False,
        group_column=group_column,
    )
    model = build_brain_embedding_model(
        train_dataset.channel_count,
        train_dataset.channel_positions,
        train_dataset.subject_count,
        config["model"],
    ).to(device)
    loss_module = build_siglip_loss(config["loss"]).to(device)
    initialization_audit = None
    if warm_start:
        initialization_audit = load_pretrained_brain_encoder(
            model,
            training_config["pretrained_brain_encoder_checkpoint"],
            config,
            train_dataset.channel_names,
            train_dataset.channel_positions,
        )
    optimizer = build_adamw_for_modules([model, loss_module], training_config)

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
    if max_updates is not None and not 0 < minimum_updates <= max_updates <= scheduler_horizon:
        raise ValueError("Pallier2025 update budget / scheduler 合同无效。")
    freeze_brain_encoder_updates = int(
        training_config.get("freeze_brain_encoder_updates", 0)
    )
    freeze_brain_encoder_epochs = int(
        training_config.get("freeze_brain_encoder_epochs", 0)
    )
    if warm_start and max_updates is None:
        if freeze_brain_encoder_updates:
            raise ValueError("epoch 生命周期不能同时使用 update 冻结字段。")
        if not 0 < freeze_brain_encoder_epochs < int(training_config["epochs"]):
            raise ValueError("Pallier warm-start 冻结轮数必须大于零且小于总轮数。")
    elif warm_start and max_updates is not None:
        if not 0 < freeze_brain_encoder_updates < max_updates:
            raise ValueError("legacy warm-start 需要有效的 update 冻结阶段。")
    elif freeze_brain_encoder_updates or freeze_brain_encoder_epochs:
        raise ValueError("没有 warm-start 检查点时不能冻结 brain encoder。")
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
        f"retrieval_acc{max(top_ks)}_vocab={SELECTION_VOCABULARY_NAME}_macro"
    )
    selection_metric = base_selection_metric
    if (
        bool(getattr(model, "prefix_revision_enabled", False))
        and getattr(model, "prefix_revision_protocol", None) == "local_window"
    ):
        selection_metric = (
            f"revision_common_delay{max(model.prefix_revision_horizons)}_"
            f"{base_selection_metric}"
        )
    if config.get("evaluation", {}).get("selection_metric") != selection_metric:
        raise ValueError("Pallier2025 validation checkpoint selection metric 合同漂移。")
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
        encoder_is_frozen = warm_start and brain_encoder_frozen_for_epoch(
            epoch, freeze_brain_encoder_epochs
        )
        if max_updates is None:
            set_brain_encoder_trainable(model, not encoder_is_frozen)
            (
                train_loss,
                epoch_updates,
                batches_seen,
                train_diagnostics,
            ) = train_one_epoch(
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
            remaining_updates = max_updates - optimizer_updates
            train_loss, epoch_updates, batches_seen = train_one_epoch_with_update_budget(
                model,
                loss_module,
                train_loader,
                optimizer,
                scaler,
                device,
                training_config,
                remaining_updates,
                scheduler=scheduler,
            )
            epoch_frozen_updates = 0
            epoch_joint_updates = epoch_updates
        optimizer_updates += epoch_updates
        frozen_encoder_updates += epoch_frozen_updates
        joint_updates += epoch_joint_updates
        validation_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            top_ks=top_ks,
            amp=training_config.get("amp", True),
        )
        if max_updates is None:
            scheduler.step()
        score = float(validation_metrics[selection_metric])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_optimizer_updates": int(epoch_updates),
                "optimizer_updates": int(optimizer_updates),
                "cumulative_optimizer_updates": int(optimizer_updates),
                "batches_seen": int(batches_seen),
                "brain_encoder_frozen": bool(encoder_is_frozen),
                "training_phase_at_epoch_end": (
                    "transformer_only" if encoder_is_frozen else "joint"
                ),
                "train_prefix_revision_loss": (
                    train_diagnostics["prefix_revision_loss"]
                    if max_updates is None
                    else None
                ),
                "train_prefix_revision_supervision_count": (
                    train_diagnostics["prefix_revision_supervision_count"]
                    if max_updates is None
                    else 0
                ),
                "train_prefix_revision_delay_counts": (
                    train_diagnostics["prefix_revision_delay_counts"]
                    if max_updates is None
                    else {}
                ),
                "train_prefix_revision_delay_losses": (
                    train_diagnostics["prefix_revision_delay_losses"]
                    if max_updates is None
                    else {}
                ),
                **validation_metrics,
            }
        )
        print(
            f"Epoch {epoch + 1}: loss={train_loss:.5f}, "
            f"val macro Top-{max(top_ks)}={score:.6f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = validation_metrics
            epochs_without_improvement = 0
            if save:
                save_checkpoint(
                    output_dir / "best.pt",
                    checkpoint_payload(
                        model,
                        loss_module,
                        config,
                        epoch + 1,
                        validation_metrics,
                        train_dataset.channel_names,
                        train_dataset.channel_positions,
                        optimizer_updates=optimizer_updates,
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
                checkpoint_payload(
                    model,
                    loss_module,
                    config,
                    epoch + 1,
                    validation_metrics,
                    train_dataset.channel_names,
                    train_dataset.channel_positions,
                    optimizer_updates=optimizer_updates,
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
        raise RuntimeError("Pallier2025 训练未产生有效 validation metric。")
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
        "test_meg_opened": False,
        "event_table": str(event_path),
        "train_rows": len(train_dataset),
        "validation_rows": len(val_dataset),
        "meg_shape": [train_dataset.channel_count, train_dataset.window_samples],
        "model_parameter_count": parameter_count(model),
        "loss_parameter_count": parameter_count(loss_module),
        "initialization_audit": initialization_audit,
        "freeze_brain_encoder_updates": freeze_brain_encoder_updates,
        "freeze_brain_encoder_epochs": freeze_brain_encoder_epochs,
        "frozen_encoder_optimizer_updates": int(frozen_encoder_updates),
        "joint_optimizer_updates": int(joint_updates),
        "text_embedding_contract": config["text_embedding"],
        "rolling_revision": (
            {
                "enabled": True,
                "protocol": "local_window_physical_truncation",
                "history_radius": int(model.prefix_revision_history_radius),
                "horizons": list(model.prefix_revision_horizons),
                "loss": "mean_of_independent_horizon_siglip_losses",
                "negative_bank": "all_text_targets_in_original_batch",
                "shared_transformer": True,
                "parameter_count_added": 0,
            }
            if bool(getattr(model, "prefix_revision_enabled", False))
            and getattr(model, "prefix_revision_protocol", None)
            == "local_window"
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


def print_training_summary(summary):
    print("Pallier2025 training completed")
    print(f"  device: {summary['device']}")
    print(f"  train/val: {summary['train_rows']} / {summary['validation_rows']}")
    print(
        f"  best epoch: {summary['best_epoch']}, "
        f"{summary['selection_metric']}: {summary['best_score']:.6f}"
    )
    print(
        f"  optimizer updates: {summary['optimizer_updates']} / "
        f"{summary['target_updates'] or 'epoch-scheduled'}"
    )
    print("  test: locked_not_evaluated")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    summary = run_training(
        load_config(args.config), smoke=args.smoke, save=not args.no_save
    )
    print_training_summary(summary)


if __name__ == "__main__":
    main()
