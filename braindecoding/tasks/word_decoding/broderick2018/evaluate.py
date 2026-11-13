"""Broderick 2018 自然语音 128 导联 EEG 词级解码验证入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from braindecoding.models import build_brain_embedding_model
from braindecoding.results import (
    canonical_evaluation_from_legacy,
    evaluate_vocabulary_manifests,
    load_vocabulary_assets,
    write_evaluation_result,
)
from braindecoding.tasks.word_decoding.broderick2018 import train as training
from braindecoding.training.runtime import load_checkpoint


def evaluate_checkpoint(
    config: dict,
    checkpoint_path: str | Path,
    split: str = "val",
    smoke: bool = False,
    zero_meg_control: bool = False,
    save: bool = True,
    force_cache: bool = False,
) -> dict:
    """加载 Broderick2018 checkpoint 并在指定数据划分上评估词级检索指标。"""
    del force_cache
    if split not in {"val", "test"}:
        raise ValueError(f"Broderick2018 evaluate_checkpoint 仅支持 val/test：{split}")
    training.set_seed(int(config["training"]["seed"]))
    device = training.choose_device(config["training"].get("device", "auto"))
    event_path = training.ensure_canonical_inputs(config)
    table = training.dataset_module.load_event_table(
        event_path,
        split=split,
        subjects=config["dataset"]["subjects"],
        trainable_only=True,
    )
    if smoke:
        table = training.limit_rows(
            table, int(config["training"].get("smoke_rows", 64))
        )
    train_table = training.dataset_module.load_event_table(
        event_path,
        split="train",
        subjects=config["dataset"]["subjects"],
        trainable_only=True,
    )
    dataset = training.build_dataset(config, table, preroll_source_table=train_table)
    group_column = training.loader_group_column(config)
    loader, _ = training.make_loader(
        dataset,
        config["training"],
        shuffle=False,
        group_column=group_column,
    )

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("dataset") not in (None, "broderick2018"):
        raise ValueError(f"不是 Broderick2018 检查点：{checkpoint_path}")
    if tuple(checkpoint.get("channel_names", ())) != tuple(dataset.channel_names):
        raise ValueError("检查点与 Broderick2018 通道顺序不一致。")
    model = build_brain_embedding_model(
        dataset.channel_count,
        dataset.channel_positions,
        dataset.subject_count,
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    top_ks = tuple(int(value) for value in config["training"].get("top_ks", [1, 10]))
    metrics, encoded = training.evaluate_loader(
        model,
        loader,
        device,
        top_ks=top_ks,
        amp=config["training"].get("amp", True),
        event_table_path=event_path,
    )
    summary = {
        "status": "smoke_evaluation_completed" if smoke else "evaluation_completed",
        "dataset": "broderick2018",
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "query_rows": len(dataset),
        "metrics": metrics,
        "controls": {},
    }
    if zero_meg_control:
        zero_dataset = training.build_dataset(
            config, table, zero_signal=True, preroll_source_table=train_table
        )
        zero_loader, _ = training.make_loader(
            zero_dataset,
            config["training"],
            shuffle=False,
            group_column=group_column,
        )
        zero_metrics, _ = training.evaluate_loader(
            model,
            zero_loader,
            device,
            top_ks=top_ks,
            amp=config["training"].get("amp", True),
            event_table_path=event_path,
        )
        summary["controls"]["zero_eeg"] = zero_metrics

    if save and training.VOCABULARY_ASSET_DIRECTORY.is_dir():
        manifests = load_vocabulary_assets(training.VOCABULARY_ASSET_DIRECTORY)
        if manifests:
            vocabulary_results = evaluate_vocabulary_manifests(
                encoded["predictions"],
                encoded["targets"],
                encoded["words"],
                manifests,
            )
            canonical = canonical_evaluation_from_legacy(
                config,
                summary,
                event_table_path=event_path,
                subjects=table["受试者"].astype(str).drop_duplicates().tolist(),
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                vocabulary_results=vocabulary_results,
            )
            write_evaluation_result(config, canonical, smoke=smoke)
    return summary


def print_evaluation_summary(summary: dict) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(training.DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zero-eeg", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = training.load_config(args.config)
    summary = evaluate_checkpoint(
        config,
        args.checkpoint,
        split=args.split,
        smoke=args.smoke,
        zero_meg_control=args.zero_eeg,
        save=not args.no_save,
    )
    print_evaluation_summary(summary)


__all__ = [
    "evaluate_checkpoint",
    "main",
    "parse_args",
    "print_evaluation_summary",
]
