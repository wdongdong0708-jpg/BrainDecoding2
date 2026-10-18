"""Pallier2025 canonical validation-only 评价入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import train as training

from braindecoding.models import build_brain_embedding_model
from braindecoding.results import (
    canonical_evaluation_from_legacy,
    evaluate_vocabulary_manifests,
    load_vocabulary_assets,
    write_evaluation_result,
)
from braindecoding.training.runtime import load_checkpoint


def evaluate_checkpoint(
    config,
    checkpoint_path,
    split="val",
    smoke=False,
    zero_meg_control=False,
    save=True,
    force_cache=False,
):
    """只允许 validation；Pallier test model evaluation 入口尚未开放。"""
    del force_cache
    if split != "val":
        raise ValueError("Pallier2025 canonical evaluator 仅允许 split='val'。")
    training.set_seed(int(config["training"]["seed"]))
    device = training.choose_device(config["training"].get("device", "auto"))
    event_path = training.ensure_canonical_inputs(config)
    table = training.dataset_module.load_event_table(
        event_path,
        split="val",
        subjects=config["dataset"]["subjects"],
        trainable_only=True,
    )
    if smoke:
        table = training.limit_rows(
            table, int(config["training"].get("smoke_rows", 64))
        )
    dataset = training.build_dataset(config, table)
    group_column = training.loader_group_column(config)
    loader, _ = training.make_loader(
        dataset,
        config["training"],
        shuffle=False,
        group_column=group_column,
    )

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint.get("task") != "word_decoding/Pallier2025":
        raise ValueError(f"不是 Pallier2025 检查点：{checkpoint_path}")
    if checkpoint.get("text_embedding_config") != config["text_embedding"]:
        raise ValueError("检查点与 Pallier2025 文本表示合同不一致。")
    if tuple(checkpoint.get("channel_names", ())) != tuple(dataset.channel_names):
        raise ValueError("检查点与 Pallier2025 通道顺序不一致。")
    source_dataset = checkpoint.get("dataset_contract", {})
    from braindecoding.subject_generalization import evaluation_contract

    subject_contract = evaluation_contract(
        checkpoint, config, tuple(dataset.subject_map)
    )
    if (
        subject_contract is None
        and tuple(source_dataset.get("subject_order", ()))
        != tuple(config["dataset"]["subjects"])
    ):
        raise ValueError("检查点与 Pallier2025 受试者顺序不一致。")
    if bool(config["model"].get("use_transformer", False)):
        expected_runtime_context = config["dataset"].get(
            "runtime_context_grouping"
        )
        if source_dataset.get("runtime_context_grouping") != expected_runtime_context:
            raise ValueError(
                "Pallier2025 context checkpoint 的运行时上下文合同不兼容；"
                "旧 cross-subject checkpoint 仅可作为 implementation diagnostic。"
            )

    model = build_brain_embedding_model(
        dataset.channel_count,
        dataset.channel_positions,
        (
            subject_contract["model_subject_count"]
            if subject_contract is not None
            else dataset.subject_count
        ),
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
        subject_index_map=(
            subject_contract["subject_index_map"]
            if subject_contract is not None
            else None
        ),
    )
    summary = {
        "status": "smoke_evaluation_completed" if smoke else "evaluation_completed",
        "split": "val",
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "query_rows": len(dataset),
        "test_meg_opened": False,
        "metrics": metrics,
        "controls": {},
    }
    if "revision_predictions" in encoded:
        maximum_delay = max(encoded["revision_predictions"])
        common_mask = encoded["revision_availability_masks"][
            maximum_delay
        ].numpy().astype(bool)
        summary["rolling_revision"] = {
            "history_radius": int(model.prefix_revision_history_radius),
            "horizons": list(model.prefix_revision_horizons),
            "common_event_count": int(common_mask.sum()),
        }
    if zero_meg_control:
        zero_dataset = training.build_dataset(config, table, zero_meg=True)
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
            subject_index_map=(
                subject_contract["subject_index_map"]
                if subject_contract is not None
                else None
            ),
        )
        summary["controls"]["zero_meg"] = zero_metrics

    if save:
        manifests = load_vocabulary_assets(
            training.VOCABULARY_ASSET_DIRECTORY
        )
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
        if "revision_predictions" in encoded:
            maximum_delay = max(encoded["revision_predictions"])
            common_mask = encoded["revision_availability_masks"][
                maximum_delay
            ].numpy().astype(bool)
            words = np.asarray(encoded["words"])
            horizons = {}
            for delay, predictions in encoded["revision_predictions"].items():
                horizons[str(delay)] = {
                    "relative_window": [
                        -int(model.prefix_revision_history_radius),
                        int(delay),
                    ],
                    "all_targets": {
                        "event_count": len(encoded["words"]),
                        "nominal_delay_available_event_count": int(
                            encoded["revision_availability_masks"][delay]
                            .sum()
                            .item()
                        ),
                        "vocabularies": evaluate_vocabulary_manifests(
                            predictions,
                            encoded["targets"],
                            encoded["words"],
                            manifests,
                        ),
                    },
                    "common_targets": {
                        "event_count": int(common_mask.sum()),
                        "vocabularies": evaluate_vocabulary_manifests(
                            predictions[common_mask],
                            encoded["targets"][common_mask],
                            words[common_mask].tolist(),
                            manifests,
                        ),
                    },
                }
            canonical["rolling_revision"] = {
                "protocol": "local_window_physical_truncation",
                "history_radius": int(model.prefix_revision_history_radius),
                "maximum_delay": int(maximum_delay),
                "shared_transformer": True,
                "common_target_policy": (
                    "target_has_all_future_events_through_maximum_delay"
                ),
                "horizons": horizons,
            }
        write_evaluation_result(config, canonical, smoke=smoke)
    return summary


def print_evaluation_summary(summary):
    metrics = summary["metrics"]
    key = "retrieval_acc10_vocab=pallier2025_50_macro"
    print("Pallier2025 validation evaluation completed")
    print(f"  queries: {summary['query_rows']}")
    print(f"  macro Top-10: {metrics[key]:.6f}")
    if summary.get("rolling_revision"):
        for delay in summary["rolling_revision"]["horizons"]:
            revision_key = f"revision_common_delay{delay}_{key}"
            print(
                f"  K={delay} common-target macro Top-10: "
                f"{metrics[revision_key]:.6f}"
            )
    print("  test: locked_not_evaluated")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(training.DEFAULT_CONFIG))
    parser.add_argument("--checkpoint")
    parser.add_argument("--zero-meg-control", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = training.load_config(args.config)
    checkpoint = args.checkpoint or str(
        Path(config["training"]["output_dir"]) / "best.pt"
    )
    summary = evaluate_checkpoint(
        config,
        checkpoint,
        split="val",
        smoke=args.smoke,
        zero_meg_control=args.zero_meg_control,
        save=not args.no_save,
    )
    print_evaluation_summary(summary)


if __name__ == "__main__":
    main()
