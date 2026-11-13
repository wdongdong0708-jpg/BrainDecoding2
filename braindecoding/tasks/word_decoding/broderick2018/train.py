"""Broderick 2018 自然语音 128 导联 EEG 词级训练与验证入口。"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import torch

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends, project_path
from braindecoding.data import broderick2018 as dataset_module
from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval_metrics
from braindecoding.experiment import resolve_experiment_config
from braindecoding.losses import build_siglip_loss
from braindecoding.models import build_brain_embedding_model
from braindecoding.optimizers import (
    build_adamw_for_modules,
    build_cosine_annealing_scheduler,
)
from braindecoding.results import build_training_summary, load_vocabulary_assets
from braindecoding.tasks.word_decoding.pallier2025 import train as pallier_train
from braindecoding.training.runtime import (
    choose_device,
    limit_rows,
    save_json,
    set_seed,
)
from braindecoding.training.word import (
    brain_encoder_frozen_for_epoch,
    encode_loader,
    make_loader,
    train_one_epoch,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/word_decoding/broderick2018/all_subjects/main_word.yaml"
)
VOCABULARY_ASSET_DIRECTORY = PROJECT_ROOT / "data/manifests/broderick2018"
SELECTION_VOCABULARY_SIZE = 50
SELECTION_VOCABULARY_NAME = "broderick2018_50"

# 复用与模态无关的模型构建、冻结与 epoch 步进工具
set_brain_encoder_trainable = pallier_train.set_brain_encoder_trainable
load_pretrained_brain_encoder = pallier_train.load_pretrained_brain_encoder
train_one_epoch_with_update_budget = pallier_train.train_one_epoch_with_update_budget


def load_config(path: str | Path | None = None) -> dict:
    """加载 Broderick2018 训练配置并解析工作区与外部缓存路径。"""
    config_path = project_path(path or DEFAULT_CONFIG)
    config = resolve_experiment_config(load_yaml_with_extends(config_path))
    for key in ("event_table", "eeg_dir", "text_embeddings"):
        if key in config.get("cache", {}):
            config["cache"][key] = str(project_path(config["cache"][key]))
    for key in ("split_manifest", "run_split_path", "root", "raw_root", "layout_path"):
        if config.get("dataset", {}).get(key):
            raw_val = str(config["dataset"][key])
            if "${" not in raw_val:
                config["dataset"][key] = str(project_path(raw_val))
    config["training"]["output_dir"] = str(
        project_path(config["training"]["output_dir"])
    )
    return config


def ensure_canonical_inputs(config: dict) -> Path:
    """校验 Broderick2018 canonical derived 缓存产品已完整就绪。"""
    from braindecoding.data.build import check_dataset

    status = check_dataset("broderick2018")
    if status["status"] != "complete":
        raise RuntimeError("Broderick2018 canonical derived data 尚未完整。")
    for key in ("event_table", "eeg_dir", "text_embeddings"):
        target = config.get("cache", {}).get(key)
        if not target or not Path(target).exists():
            raise FileNotFoundError(f"Broderick2018 canonical 数据产品缺失：{key}")
    return Path(config["cache"]["event_table"])


def build_dataset(
    config: dict,
    table,
    zero_meg: bool = False,
    zero_signal: bool = False,
    temporal_shift_seconds: float = 0.0,
    *,
    shift_mode: str | None = None,
    preroll_seconds: float | None = None,
    preroll_source_table=None,
    preroll_buffers=None,
):
    """根据事件表和配置构造 Broderick2018WordDataset（支持 45 秒 Pre-Roll 错位）。"""
    dataset_cfg = dict(config["dataset"])
    cache_cfg = config.get("cache") or {}
    if "event_table" in cache_cfg and "event_table_path" not in dataset_cfg:
        dataset_cfg["event_table_path"] = str(cache_cfg["event_table"])
    siamese_cfg = config.get("siamese_debiased") or {}
    effective_shift_mode = (
        shift_mode
        if shift_mode is not None
        else dataset_cfg.get("shift_mode", siamese_cfg.get("shift_mode"))
    )
    effective_preroll_seconds = (
        preroll_seconds
        if preroll_seconds is not None
        else dataset_cfg.get("preroll_seconds", siamese_cfg.get("preroll_seconds"))
    )
    signal_dir = cache_cfg.get("eeg_dir") or cache_cfg.get("meg_dir")
    return dataset_module.Broderick2018WordDataset(
        table,
        dataset_cfg,
        signal_dir,
        cache_cfg["text_embeddings"],
        config["text_embedding"],
        zero_eeg=bool(zero_meg or zero_signal),
        temporal_shift_seconds=float(temporal_shift_seconds),
        shift_mode=effective_shift_mode,
        preroll_seconds=effective_preroll_seconds,
        preroll_source_table=preroll_source_table,
        preroll_buffers=preroll_buffers,
        allow_subject_subset=bool(
            config.get("subject_generalization")
            or config["dataset"].get("allow_subject_subset")
        ),
    )


def loader_group_column(config: dict) -> str:
    """上下文模型按受试者记录内上下文键分组；单窗词模型按 sentence_uid 分组。"""
    if not bool(config["model"].get("use_transformer", False)):
        return "sentence_uid"
    expected = dataset_module.RUNTIME_CONTEXT_GROUPING
    actual = config["dataset"].get("runtime_context_grouping")
    if actual != expected:
        raise ValueError(
            f"Broderick2018 runtime context 合同不一致：预期 {expected}，实际 {actual}"
        )
    return dataset_module.RUNTIME_CONTEXT_COLUMN


def frozen_vocabulary(
    event_table_path: str | Path | None = None,
    size: int = SELECTION_VOCABULARY_SIZE,
) -> tuple[str, ...]:
    """返回 Broderick2018 冻结评估词表；未预写清单时从 train 事件表动态生成。"""
    target_size = int(size)
    manifest_file = VOCABULARY_ASSET_DIRECTORY / f"vocabulary_N{target_size}.json"
    if manifest_file.is_file():
        manifests = load_vocabulary_assets(VOCABULARY_ASSET_DIRECTORY)
        if target_size in manifests:
            return tuple(manifests[target_size]["vocabulary"])
    if event_table_path is not None:
        cache_manifest_dir = Path(event_table_path).parent / "manifests"
        cache_manifest_file = cache_manifest_dir / f"vocabulary_N{target_size}.json"
        if cache_manifest_file.is_file():
            manifests = load_vocabulary_assets(cache_manifest_dir)
            if target_size in manifests:
                return tuple(manifests[target_size]["vocabulary"])
    if event_table_path is None:
        raise FileNotFoundError(
            f"未找到 Broderick2018 N={target_size} 词表清单，且未提供 event_table_path：{manifest_file}"
        )
    table = dataset_module.load_event_table(event_table_path)
    material = dataset_module.material_word_view(table)
    train_words = material.loc[
        material["数据划分"].eq("train") & material["_protocol_word"].ne(""),
        "_protocol_word",
    ]
    counts = Counter(train_words)
    ranked = sorted(counts, key=lambda word: (-int(counts[word]), word))
    return tuple(ranked[:target_size])


def evaluate_loader(
    model,
    loader,
    device,
    top_ks=(1, 10),
    amp=True,
    subject_index_map=None,
    *,
    event_table_path: str | Path | None = None,
):
    """在 Broderick2018 冻结 N=50（及滚动修订多视界）上评估模型验证性能。"""
    encoded = encode_loader(
        model,
        loader,
        device,
        amp=amp,
        subject_index_map=subject_index_map,
    )
    vocab = frozen_vocabulary(event_table_path=event_table_path)
    metrics = fixed_vocabulary_retrieval_metrics(
        encoded["predictions"],
        encoded["targets"],
        encoded["words"],
        vocab,
        top_ks=top_ks,
        vocabulary_name=SELECTION_VOCABULARY_NAME,
    )
    if "revision_predictions" in encoded:
        availability = encoded["revision_availability_masks"]
        maximum_delay = max(encoded["revision_predictions"])
        common_mask = availability[maximum_delay].numpy().astype(bool)
        if not common_mask.any():
            raise ValueError("Broderick2018 验证集没有同时支持全部滚动回修延迟的目标词。")
        metrics["revision_common_event_count"] = int(common_mask.sum())
        words = np.asarray(encoded["words"])
        for delay, predictions in encoded["revision_predictions"].items():
            delay_metrics = fixed_vocabulary_retrieval_metrics(
                predictions,
                encoded["targets"],
                encoded["words"],
                vocab,
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
                vocab,
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


def _load_optional_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def checkpoint_payload(
    model,
    loss_module,
    config: dict,
    epoch: int,
    metrics: dict,
    channel_names,
    channel_positions,
    *,
    optimizer_updates: int,
    optimizer=None,
    scheduler=None,
    initialization_audit=None,
    training_phase=None,
) -> dict:
    """构造含 Broderick2018 数据与预处理合同签名的 checkpoint payload。"""
    event_path = Path(config["cache"]["event_table"])
    derived_root = event_path.parent.parent
    event_manifest = _load_optional_json(event_path.parent / "manifest.json")
    signal_manifest = _load_optional_json(derived_root / "signals/manifest.json")
    text_manifest = _load_optional_json(
        Path(config["cache"]["text_embeddings"]).parent / "manifest.json"
    )
    split_manifest_path = Path(
        config["dataset"].get("split_manifest")
        or config["dataset"].get("run_split_path")
        or VOCABULARY_ASSET_DIRECTORY / "run_split.json"
    )
    split_manifest = _load_optional_json(split_manifest_path)
    vocab_manifests = (
        load_vocabulary_assets(VOCABULARY_ASSET_DIRECTORY)
        if VOCABULARY_ASSET_DIRECTORY.is_dir()
        and any(VOCABULARY_ASSET_DIRECTORY.glob("vocabulary_N*.json"))
        else {}
    )
    payload = {
        "format_version": 1,
        "task": "word_decoding/Broderick2018",
        "dataset": "broderick2018",
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
            "dataset": "broderick2018",
            "subjects": list(config["dataset"]["subjects"]),
            "subject_order": list(config["dataset"]["subjects"]),
            "runs": list(config["dataset"]["runs"]),
            "split_manifest": config["dataset"].get("split_manifest"),
            "split": split_manifest.get("assignments"),
            "split_manifest_sha256": split_manifest.get("manifest_sha256"),
            "event_table_sha256": event_manifest.get("event_table_sha256"),
            "window_contract_version": config["dataset"]["window_contract_version"],
            "window_start_offset_seconds": config["dataset"]["window_start_offset_seconds"],
            "window_seconds": config["dataset"]["window_seconds"],
            "eligibility_window_seconds": config["dataset"]["eligibility_window_seconds"],
            "baseline_seconds": config["dataset"]["baseline_seconds"],
            "clamp": config["dataset"]["clamp"],
            "target_sampling_rate_hz": config["dataset"]["target_sampling_rate_hz"],
            "channel_names_sha256": config["dataset"]["channel_names_sha256"],
            "mastoid_rereference": bool(
                config["dataset"].get("mastoid_rereference", False)
            ),
            "recording_count": signal_manifest.get("recording_count"),
            "context_grouping": config["dataset"].get("context_grouping"),
            "runtime_context_grouping": config["dataset"].get(
                "runtime_context_grouping", "not_used_by_word_model"
            ),
            "max_context_words": config["dataset"]["max_context_words"],
            "text_content_sha256": text_manifest.get("content_sha256"),
            "vocabulary_manifest_sha256": {
                str(size): manifest["manifest_sha256"]
                for size, manifest in vocab_manifests.items()
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


def run_training(config: dict, smoke: bool = False, save: bool = True, force_cache: bool = False) -> dict:
    """训练 Broderick2018 EEG word/context 模型；仅消费 train 与 validation。"""
    del force_cache
    training_config = dict(config["training"])
    warm_start = bool(training_config.get("pretrained_brain_encoder_checkpoint"))
    if warm_start and not bool(config["model"].get("use_transformer", False)):
        raise ValueError("Broderick2018 warm-start 只用于启用 Transformer 的 context 条件。")
    if int(config["model"]["embedding_dimension"]) != int(
        config["text_embedding"]["embedding_dimension"]
    ):
        raise ValueError("Broderick2018 model 与 T5 target 维数不一致。")
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

    set_seed(int(training_config["seed"]))
    device = choose_device(training_config.get("device", "auto"))
    event_path = ensure_canonical_inputs(config)
    subjects = config["dataset"]["subjects"]
    train_table = dataset_module.load_event_table(
        event_path, split="train", subjects=subjects, trainable_only=True
    )
    val_table = dataset_module.load_event_table(
        event_path, split="val", subjects=subjects, trainable_only=True
    )
    if smoke:
        maximum = int(training_config.get("smoke_rows", 64))
        train_table = limit_rows(train_table, maximum)
        val_table = limit_rows(val_table, maximum)

    train_dataset = build_dataset(config, train_table, preroll_source_table=train_table)
    val_dataset = build_dataset(config, val_table, preroll_source_table=train_table)
    group_column = loader_group_column(config)
    train_loader, train_sampler = make_loader(
        train_dataset, training_config, shuffle=True, group_column=group_column
    )
    val_loader, _ = make_loader(
        val_dataset, training_config, shuffle=False, group_column=group_column
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
    epochs = int(training_config["epochs"])
    freeze_brain_encoder_epochs = int(
        training_config.get("freeze_brain_encoder_epochs", 0)
    )
    scheduler = build_cosine_annealing_scheduler(
        optimizer,
        epochs,
        minimum_learning_rate=float(training_config.get("minimum_learning_rate", 0.0)),
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
    stop_reason = "maximum_epochs_reached"
    history = []

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        encoder_is_frozen = warm_start and brain_encoder_frozen_for_epoch(
            epoch, freeze_brain_encoder_epochs
        )
        set_brain_encoder_trainable(model, not encoder_is_frozen)
        train_loss, epoch_updates, batches_seen, _ = train_one_epoch(
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
        optimizer_updates += epoch_updates
        validation_metrics, _ = evaluate_loader(
            model,
            val_loader,
            device,
            top_ks=top_ks,
            amp=training_config.get("amp", True),
            event_table_path=event_path,
        )
        scheduler.step()
        score = float(validation_metrics[selection_metric])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "optimizer_updates": int(optimizer_updates),
                "batches_seen": int(batches_seen),
                "validation_score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = dict(validation_metrics)
            best_model_state = copy.deepcopy(model.state_dict())
            best_loss_state = copy.deepcopy(loss_module.state_dict())
            epochs_without_improvement = 0
            if save:
                torch.save(
                    checkpoint_payload(
                        model,
                        loss_module,
                        config,
                        epoch + 1,
                        best_metrics,
                        train_dataset.channel_names,
                        train_dataset.channel_positions,
                        optimizer_updates=optimizer_updates,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        initialization_audit=initialization_audit,
                        training_phase="transformer_only" if encoder_is_frozen else "joint",
                    ),
                    output_dir / "best.pt",
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(training_config.get("patience", epochs)):
                stop_reason = "early_stopping_patience_reached"
                break

    if best_model_state is not None and best_loss_state is not None:
        model.load_state_dict(best_model_state)
        loss_module.load_state_dict(best_loss_state)

    summary = {
        "status": "smoke_completed" if smoke else "completed",
        "dataset": "broderick2018",
        "best_epoch": best_epoch,
        "best_score": best_score,
        "selection_metric": selection_metric,
        "stop_reason": stop_reason,
        "optimizer_updates": int(optimizer_updates),
        "metrics": best_metrics or {},
        "history": history,
    }
    if save:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return summary


train_model = run_training


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def print_training_summary(summary: dict) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    summary = run_training(config, smoke=args.smoke, save=not args.no_save)
    print_training_summary(summary)


__all__ = [
    "DEFAULT_CONFIG",
    "SELECTION_VOCABULARY_NAME",
    "SELECTION_VOCABULARY_SIZE",
    "VOCABULARY_ASSET_DIRECTORY",
    "build_dataset",
    "checkpoint_payload",
    "ensure_canonical_inputs",
    "evaluate_loader",
    "frozen_vocabulary",
    "load_config",
    "load_pretrained_brain_encoder",
    "loader_group_column",
    "main",
    "parse_args",
    "print_training_summary",
    "run_training",
    "set_brain_encoder_trainable",
    "train_model",
    "train_one_epoch_with_update_budget",
]
