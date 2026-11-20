"""统一 MEG 词级解码任务公共辅助模块（取代已删除的 ``braindecoding/tasks/word_decoding/``）。

同时无缝支撑：
- 旧 10 被试本地 BIDS ``pallier2025`` 缓存（严格维持既有通道顺序 SHA 与受试者顺序）；
- 统一 PNPL 适配器缓存（``libribrain100`` / ``pallier2025_full`` / ``armeni2022``）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends, project_path
from braindecoding.data import pallier2025_dataset as dataset_module
from braindecoding.data.event_table import validate_event_table
from braindecoding.data.word_window_dataset import (
    RUNTIME_CONTEXT_COLUMN,
    RUNTIME_CONTEXT_GROUPING,
    WordWindowDataset,
    format_subject_id,
    parse_bool_value,
)
from braindecoding.evaluation.retrieval_metrics import fixed_vocabulary_retrieval_metrics
from braindecoding.evaluation.vocabulary_coverage import build_frequency_vocabulary
from braindecoding.experiment import file_sha256, resolve_experiment_config
from braindecoding.results import load_vocabulary_assets
from braindecoding.training.training_runtime import load_checkpoint
from braindecoding.training.word_encoder_training import encode_loader


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "word_decoding" / "pallier2025" / "all_subjects" / "main_word.yaml"
)
VOCABULARY_ASSET_DIRECTORY = PROJECT_ROOT / "data" / "manifests" / "pallier2025"
SELECTION_VOCABULARY_SIZE = 50
SELECTION_VOCABULARY_NAME = "pallier2025_50"


def load_config(path=None) -> dict:
    """载入实验配置并解析派生数据与输出路径。"""
    config_path = project_path(path or DEFAULT_CONFIG)
    config = resolve_experiment_config(load_yaml_with_extends(config_path))
    for key in ("event_table", "meg_dir", "text_embeddings"):
        if key in config.get("cache", {}):
            config["cache"][key] = str(project_path(config["cache"][key]))
    for key in ("split_manifest", "qc_artifact", "layout_path"):
        if config.get("dataset", {}).get(key):
            config["dataset"][key] = str(project_path(config["dataset"][key]))
    if "output_dir" in config.get("training", {}):
        config["training"]["output_dir"] = str(
            project_path(config["training"]["output_dir"])
        )
    return config


def load_event_table(
    path: str | Path,
    *,
    split: str | None = None,
    subjects: Sequence[str] | None = None,
    trainable_only: bool = False,
) -> pd.DataFrame:
    """统一加载旧 10 被试 Pallier2025 或 PNPL 派生 ``events.csv`` 事件表。"""
    table = pd.read_csv(path, keep_default_na=False)
    for col in ("是否可训练", "是否内容词", "是否序列末词"):
        if col in table.columns:
            table[col] = table[col].map(lambda v, f=col: parse_bool_value(v, field=f))
    validate_event_table(table)
    if split is not None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"无效数据划分：{split!r}")
        table = table[table["数据划分"].astype(str).eq(split)]
    if subjects is not None and subjects != "auto":
        wanted = {format_subject_id(s) for s in subjects}
        table = table[table["受试者"].astype(str).isin(wanted)]
    if trainable_only:
        table = table[table["是否可训练"].astype(bool)]
    return table.reset_index(drop=True)


def build_dataset(
    config: Mapping[str, Any],
    table: pd.DataFrame,
    zero_meg: bool = False,
    zero_signal: bool = False,
    temporal_shift_seconds: float = 0.0,
    *,
    shift_mode: str | None = None,
    preroll_seconds: float | None = None,
    preroll_source_table: pd.DataFrame | None = None,
    preroll_buffers: Mapping[str, np.ndarray] | None = None,
) -> WordWindowDataset:
    """统一实例化 ``WordWindowDataset``（自动兼容旧 10 被试 Pallier2025 与 PNPL 缓存）。"""
    dataset_cfg = dict(config.get("dataset") or {})
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
    dataset_name = str(config.get("experiment", {}).get("dataset") or dataset_cfg.get("id") or "")
    synthetic_condition = str(dataset_cfg.get("synthetic_timing") or "").strip().lower()
    if synthetic_condition and synthetic_condition not in {"shared", "independent"}:
        raise ValueError(
            f"不支持的 synthetic_timing={synthetic_condition!r}，可选：shared, independent"
        )
    use_synthetic = bool(synthetic_condition) and not bool(zero_meg or zero_signal)
    effective_zero_meg = bool(zero_meg or zero_signal or use_synthetic)
    is_legacy_pallier = (
        dataset_name == "pallier2025" and "信号相对路径" not in table.columns
    )
    if is_legacy_pallier:
        base_ds: WordWindowDataset = dataset_module.Pallier2025WordDataset(
            table,
            dataset_cfg,
            cache_cfg["meg_dir"],
            cache_cfg["text_embeddings"],
            config["text_embedding"],
            zero_meg=effective_zero_meg,
            temporal_shift_seconds=0.0 if use_synthetic else float(temporal_shift_seconds),
            shift_mode=effective_shift_mode,
            preroll_seconds=effective_preroll_seconds,
            preroll_source_table=preroll_source_table,
            preroll_buffers=preroll_buffers,
            allow_subject_subset=bool(
                config.get("subject_generalization")
                or dataset_cfg.get("allow_subject_subset")
            ),
        )
    else:
        base_ds = WordWindowDataset(
            table,
            dataset_cfg,
            cache_cfg["meg_dir"],
            cache_cfg["text_embeddings"],
            config["text_embedding"],
            zero_meg=effective_zero_meg,
            temporal_shift_seconds=0.0 if use_synthetic else float(temporal_shift_seconds),
            shift_mode=effective_shift_mode,
            preroll_seconds=effective_preroll_seconds,
            preroll_source_table=preroll_source_table,
            preroll_buffers=preroll_buffers,
        )
    if not use_synthetic:
        return base_ds

    from braindecoding.data.synthetic_timing_dataset import SyntheticTimingDataset

    split_col = "数据划分" if "数据划分" in table.columns else ("split" if "split" in table.columns else None)
    if split_col is not None and len(table) > 0:
        resolved_split = str(table[split_col].iloc[0]).strip().lower()
    else:
        resolved_split = "train"
    if resolved_split not in {"train", "val", "test"}:
        resolved_split = "train"
    train_seed = int((config.get("training") or {}).get("seed", 0))
    eval_seed = int(dataset_cfg.get("synthetic_evaluation_seed", 20261008))
    return SyntheticTimingDataset(
        base_ds,
        synthetic_condition,
        split=resolved_split,
        training_seed=train_seed,
        evaluation_seed=eval_seed,
        temporal_shift_seconds=float(temporal_shift_seconds),
        shift_mode=str(effective_shift_mode or "preroll"),
        preroll_seconds=float(effective_preroll_seconds or 45.0),
        include_onset_seconds=bool(getattr(base_ds, "include_onset_seconds", False)),
    )


def set_brain_encoder_trainable(model, trainable: bool) -> None:
    """冻结时同步固定 brain encoder 的 BN 与 dropout 状态。"""
    flag = bool(trainable)
    for parameter in model.brain_encoder.parameters():
        parameter.requires_grad_(flag)
    model.brain_encoder.train(flag)


def loader_group_column(config: Mapping[str, Any]) -> str:
    """上下文模型使用 recording-local 键防跨记录交互；word 模型使用 sentence_uid。"""
    if not bool(config["model"].get("use_transformer", False)):
        return "sentence_uid"
    actual = config.get("dataset", {}).get("runtime_context_grouping")
    if actual is not None and actual != RUNTIME_CONTEXT_GROUPING:
        raise ValueError(
            f"runtime context 合同不一致：预期 {RUNTIME_CONTEXT_GROUPING}，实际 {actual}"
        )
    return RUNTIME_CONTEXT_COLUMN


def frozen_vocabulary(
    size: int = SELECTION_VOCABULARY_SIZE,
    *,
    dataset: str = "pallier2025",
    train_words: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """返回冻结评估词表；对无预设 manifest 的 PNPL 数据集基于 train 词频确定性构建。"""
    if dataset == "pallier2025" and (VOCABULARY_ASSET_DIRECTORY / f"vocabulary_N{int(size)}.json").is_file():
        manifests = load_vocabulary_assets(VOCABULARY_ASSET_DIRECTORY)
        return tuple(manifests[int(size)]["vocabulary"])
    if train_words is not None:
        return tuple(build_frequency_vocabulary(train_words, int(size)))
    manifests = load_vocabulary_assets(VOCABULARY_ASSET_DIRECTORY)
    return tuple(manifests[int(size)]["vocabulary"])


def load_pretrained_brain_encoder(
    model,
    checkpoint_path,
    config,
    channel_names,
    channel_positions,
) -> dict[str, Any]:
    """仅加载同数据合同 CNN-only 检查点中的 ``brain_encoder.*`` 参数用于 Stage 2 热启动。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    source_model = checkpoint.get("model_config", {})
    if bool(source_model.get("use_transformer", True)):
        raise ValueError("warm-start 必须来自 CNN-only (Word-Only) 检查点。")
    if source_model.get("embedding_dimension") != config["model"].get(
        "embedding_dimension"
    ) or source_model.get("conv") != config["model"].get("conv"):
        raise ValueError("warm-start 脑编码器结构合同不一致。")
    if checkpoint.get("text_embedding_config") != config["text_embedding"]:
        raise ValueError("warm-start 文本表示合同不一致。")
    if tuple(checkpoint.get("channel_names", ())) != tuple(channel_names):
        raise ValueError("warm-start MEG 通道顺序不一致。")
    source_positions = np.asarray(checkpoint.get("channel_positions", ()))
    target_positions = np.asarray(channel_positions)
    if source_positions.shape != target_positions.shape or not np.allclose(
        source_positions, target_positions, rtol=0.0, atol=1e-7
    ):
        raise ValueError("warm-start 通道位置合同不一致。")

    prefix = "brain_encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in checkpoint["model_state"].items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise ValueError("warm-start 检查点不包含 brain_encoder 参数。")
    model.brain_encoder.load_state_dict(encoder_state, strict=True)
    dataset_name = str(config.get("experiment", {}).get("dataset") or "pallier2025")
    source_metrics = checkpoint.get("metrics", {})
    source_macro_r10 = source_metrics.get(
        f"retrieval_acc10_vocab={dataset_name}_50_macro",
        source_metrics.get("retrieval_acc10_vocab=pallier2025_50_macro"),
    )
    return {
        "method": "cnn_only_best_checkpoint_brain_encoder_only",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", 0)),
        "source_update": int(checkpoint.get("optimizer_updates", 0)),
        "source_macro_r_at_10": source_macro_r10,
        "loaded_loss_state": False,
        "loaded_transformer_state": False,
    }


def evaluate_loader(
    model,
    loader,
    device,
    top_ks=(1, 10),
    amp: bool = True,
    subject_index_map=None,
    vocabulary: Sequence[str] | None = None,
    vocabulary_name: str = SELECTION_VOCABULARY_NAME,
):
    """对 DataLoader 执行前向编码并计算冻结词表及多视界检索指标。"""
    vocab = tuple(vocabulary) if vocabulary is not None else frozen_vocabulary()
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
        vocab,
        top_ks=top_ks,
        vocabulary_name=vocabulary_name,
    )
    if "revision_predictions" in encoded:
        availability = encoded["revision_availability_masks"]
        maximum_delay = max(encoded["revision_predictions"])
        common_mask = availability[maximum_delay].numpy().astype(bool)
        if not common_mask.any():
            raise ValueError("验证集没有同时支持全部滚动视界的目标词。")
        metrics["revision_common_event_count"] = int(common_mask.sum())
        words = np.asarray(encoded["words"])
        for delay, predictions in encoded["revision_predictions"].items():
            delay_metrics = fixed_vocabulary_retrieval_metrics(
                predictions,
                encoded["targets"],
                encoded["words"],
                vocab,
                top_ks=top_ks,
                vocabulary_name=vocabulary_name,
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
                vocabulary_name=vocabulary_name,
            )
            metrics.update(
                {
                    f"revision_common_delay{delay}_{key}": value
                    for key, value in common_metrics.items()
                    if isinstance(value, (int, float))
                }
            )
    return metrics, encoded
