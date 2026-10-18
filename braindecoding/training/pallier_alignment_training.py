"""One isolated Pallier coverage trial, with legacy and strict-bank validation.

The training math comes from the existing Pallier helpers. Only the cutoff
coverage changes relative to the old anchor; this is not the entire BGE v1.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from braindecoding.config import PROJECT_ROOT, project_path
from braindecoding.evaluation.retrieval import retrieval_ranks, summarize_retrieval
from braindecoding.experiment import file_sha256, initialize_run_directory, update_run_status
from braindecoding.reporting import runtime_cache_dir, study_report_dir
from braindecoding.tasks.word_decoding.pallier2025 import train as legacy
from braindecoding.training.runtime import load_checkpoint, save_checkpoint, save_json, set_seed
from braindecoding.training.word import encode_loader, make_loader, train_one_epoch
from braindecoding.unified_rolling_training import full_vocabulary_metrics

CONFIG = PROJECT_ROOT / "configs/training/full_coverage_revision_alignment.yaml"
RUNTIME = runtime_cache_dir("pallier_seed0_coverage_alignment_v1")
REPORT = study_report_dir("pallier2025", "baseline_alignment_coverage_v1")


def status(**fields):
    path = RUNTIME / "status.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update(fields, pid=os.getpid(), updated_unix=time.time(), test_evaluated=False)
    temporary = path.with_suffix(".tmp")
    save_json(temporary, current)
    temporary.replace(path)
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def strict_bank_metrics(encoded, dataset, vocabulary):
    """Keep all supplied candidates even when a word has no common query."""
    words = np.asarray(encoded["words"])
    vocabulary = list(vocabulary)
    bank = np.stack([dataset.embedding_map[w] for w in vocabulary])
    eligible = np.isin(words, vocabulary)
    common = np.logical_and.reduce([
        mask.numpy().astype(bool) for mask in encoded["revision_availability_masks"].values()
    ])
    subject_map = dict(zip(dataset.table.event_id, dataset.table.subject_id))
    subjects = np.asarray([subject_map[e] for e in encoded["event_ids"]])
    output = {}
    for delay, predictions in encoded["revision_predictions"].items():
        ranks = retrieval_ranks(predictions[eligible], words[eligible], bank, vocabulary)
        per_delay = {}
        for name, mask in (("all_targets", np.ones(eligible.sum(), dtype=bool)),
                           ("common_targets", common[eligible])):
            selected_words = words[eligible][mask]
            selected_subjects = subjects[eligible][mask]
            selected_ranks = ranks[mask]
            item = summarize_retrieval(selected_ranks, selected_words, len(vocabulary), top_ks=(1, 10))
            item["per_subject"] = {
                str(subject): summarize_retrieval(selected_ranks[selected_subjects == subject],
                    selected_words[selected_subjects == subject], len(vocabulary), top_ks=(1, 10))
                for subject in sorted(set(selected_subjects))
            }
            per_delay[name] = item
        output[str(delay)] = per_delay
    return output


def gate_result(metrics, baseline, maximum_drop):
    score = metrics["2"]["all_targets"]["macro_recall_at_10"]
    gain = (metrics["2"]["common_targets"]["macro_recall_at_10"]
            - metrics["0"]["common_targets"]["macro_recall_at_10"])
    threshold = baseline - maximum_drop
    return {"passed": bool(score >= threshold and gain > 0), "k2_macro_top10": score,
            "old_k2_macro_top10": baseline, "minimum_k2_macro_top10": threshold,
            "difference_percentage_points": 100 * (score - baseline),
            "common_k2_minus_k0": gain, "full_queue_started": False}


def dataset_and_loader(cfg, split):
    if split not in ("train", "val"):
        raise ValueError("This development trial only allows train and val.")
    table = legacy.dataset_module.load_event_table(cfg["cache"]["event_table"],
        split=split, subjects=cfg["dataset"]["subjects"], trainable_only=True)
    dataset = legacy.build_dataset(cfg, table)
    loader, sampler = make_loader(dataset, cfg["training"], shuffle=split == "train",
        group_column=legacy.loader_group_column(cfg))
    return dataset, loader, sampler


def model_for(dataset, cfg):
    return legacy.build_brain_embedding_model(dataset.channel_count, dataset.channel_positions,
        dataset.subject_count, cfg["model"]).cuda()


def validate(model, loader, dataset, vocabulary):
    metrics, encoded = legacy.evaluate_loader(model, loader, torch.device("cuda"), amp=True)
    strict = strict_bank_metrics(encoded, dataset, legacy.frozen_vocabulary())
    full = full_vocabulary_metrics(encoded, dataset, vocabulary)
    return {"legacy": metrics, "strict_vocab50": strict, "full_vocabulary": full}, encoded


class ProgressLoader:
    """Observe completed batches without changing sampler or random state."""
    def __init__(self, loader, epoch):
        self.loader, self.epoch = loader, epoch

    def __iter__(self):
        for batch_number, batch in enumerate(self.loader, 1):
            yield batch
            if batch_number == 1 or batch_number % 25 == 0:
                status(status="training", epoch=self.epoch, completed_batches=batch_number)


def run():
    cfg = legacy.load_config(CONFIG)
    trial = cfg["alignment_trial"]
    if trial["status"] != "authorized_single_seed_only" or cfg["training"]["seed"] != 0:
        raise ValueError("Only the authorized Pallier seed 0 trial may run.")
    if trial["full_queue_authorized"] or trial["test_evaluation_allowed"]:
        raise ValueError("This entry point cannot authorize the full queue or test evaluation.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this trial.")
    torch.set_num_threads(2)
    out = Path(cfg["training"]["output_dir"])
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite a trial: {out}")
    baseline_dir = project_path(trial["baseline_directory"])
    old_cfg = legacy.load_config(baseline_dir / "resolved_config.yaml")
    for section in ("model", "text_embedding", "loss", "dataset", "evaluation", "cache"):
        if cfg[section] != old_cfg[section]:
            raise ValueError(f"Unexpected change from the old anchor: {section}")
    ignored_training = {"output_dir", "revision_cutoff_policy", "warm_start_seed"}
    if ({k: v for k, v in cfg["training"].items() if k not in ignored_training}
            != {k: v for k, v in old_cfg["training"].items() if k not in ignored_training}):
        raise ValueError("Training factors other than coverage changed.")
    if cfg["training"]["revision_cutoff_policy"] != "all":
        raise ValueError("This trial requires all cutoffs.")
    legacy.ensure_canonical_inputs(cfg)
    old_result = json.loads((baseline_dir / "evaluation/val.json").read_text(encoding="utf-8"))
    old_horizons = old_result["rolling_revision"]["horizons"]
    old_score = old_horizons["2"]["all_targets"]["vocabularies"]["50"]["retrieval"]["macro_top10"]
    vocabulary_manifest = PROJECT_ROOT / "data/manifests/rolling_sentence/pallier2025_train_vocabulary.json"
    vocabulary = json.loads(vocabulary_manifest.read_text(encoding="utf-8"))["vocabulary"]
    val, val_loader, _ = dataset_and_loader(cfg, "val")
    train, train_loader, sampler = dataset_and_loader(cfg, "train")
    if len(val) != trial["expected_validation_rows"] or len(train) != 115290:
        raise ValueError("Unexpected data split size.")
    if sorted(set(train.table.normalized_word)) != vocabulary:
        raise ValueError("The full vocabulary does not match training words.")
    if set(train.table.event_id) & set(val.table.event_id):
        raise ValueError("Train and validation overlap.")
    sizes = train.table.groupby(legacy.loader_group_column(cfg), sort=False).size().to_numpy()
    expected_counts = {d: int(np.maximum(sizes - d, 0).sum()) for d in range(3)}
    out, _ = initialize_run_directory(cfg, sys.argv,
        protocol_manifests=[CONFIG], vocabulary_manifests=[vocabulary_manifest])
    sources = [CONFIG, Path(__file__), PROJECT_ROOT / "braindecoding/training/word.py",
        PROJECT_ROOT / "braindecoding/models.py", PROJECT_ROOT / "braindecoding/losses.py",
        PROJECT_ROOT / "braindecoding/tasks/word_decoding/pallier2025/train.py",
        PROJECT_ROOT / "braindecoding/training/unified_rolling_training.py"]
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in sources if Path(p).is_file()}
    provenance = {"config": cfg, "source_sha256": source_hashes,
        "old_checkpoint_sha256": file_sha256(baseline_dir / "best.pt"),
        "old_evaluation_sha256": file_sha256(baseline_dir / "evaluation/val.json"),
        "warm_start_sha256": file_sha256(cfg["training"]["pretrained_brain_encoder_checkpoint"]),
        "text_cache_sha256": file_sha256(cfg["cache"]["text_embeddings"]),
        "expected_supervision_counts": expected_counts, "train_rows": len(train), "val_rows": len(val),
        "maximum_drop_fixed_before_training": trial["maximum_drop"], "test_evaluated": False}
    save_json(out / "trial_manifest.json", provenance)
    status(status="validating_old_anchor", output_dir=str(out), dataset="pallier2025", seed=0,
        started_unix=time.time(), max_epochs=cfg["training"]["epochs"], completed_epochs=0)
    anchor = model_for(val, cfg)
    checkpoint = load_checkpoint(baseline_dir / "best.pt")
    if checkpoint["model_config"] != cfg["model"] or checkpoint["channel_names"] != val.channel_names:
        raise ValueError("Old anchor model or channel mismatch.")
    anchor.load_state_dict(checkpoint["model_state"], strict=True)
    anchor_metrics, anchor_encoded = validate(anchor, val_loader, val, vocabulary)
    reference_ids = list(anchor_encoded["event_ids"])
    for d in range(3):
        for subset, key in (("all_targets", "expected_all_vocab50_queries"),
                            ("common_targets", "expected_common_vocab50_queries")):
            m = anchor_metrics["strict_vocab50"][str(d)][subset]
            if m["query_count"] != trial[key] or m["candidate_count"] != 50:
                raise ValueError("Anchor query/candidate counts changed.")
        observed = anchor_metrics["strict_vocab50"][str(d)]["all_targets"]["macro_recall_at_10"]
        expected = old_horizons[str(d)]["all_targets"]["vocabularies"]["50"]["retrieval"]["macro_top10"]
        if abs(observed - expected) > trial["baseline_reproduction_tolerance"]:
            raise ValueError(f"Old K{d} could not be reproduced: {observed} versus {expected}")
    save_json(out / "evaluation/old_anchor_strict50.json", anchor_metrics)
    save_json(out / "evaluation/validation_event_ids.json", reference_ids)
    del anchor, checkpoint, anchor_encoded
    gc.collect()
    torch.cuda.empty_cache()
    # Anchor validation must not consume the training seed's random sequence.
    set_seed(0)
    model = model_for(train, cfg)
    loss_module = legacy.build_siglip_loss(cfg["loss"]).cuda()
    initialization = legacy.load_pretrained_brain_encoder(model,
        cfg["training"]["pretrained_brain_encoder_checkpoint"], cfg,
        train.channel_names, train.channel_positions)
    optimizer = legacy.build_adamw_for_modules([model, loss_module], cfg["training"])
    scheduler = legacy.build_cosine_annealing_scheduler(optimizer, cfg["training"]["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history, best = [], {"legacy50": -float("inf"), "full_vocab": -float("inf")}
    best_rows, stale, updates = {}, 0, 0
    for epoch in range(cfg["training"]["epochs"]):
        for path, expected in source_hashes.items():
            if file_sha256(PROJECT_ROOT / path) != expected:
                raise RuntimeError(f"Trial source changed during training: {path}")
        started = time.perf_counter()
        sampler.set_epoch(epoch)
        frozen = epoch < cfg["training"]["freeze_brain_encoder_epochs"]
        legacy.set_brain_encoder_trainable(model, not frozen)
        status(status="training", epoch=epoch + 1, completed_batches=0,
            brain_encoder_frozen=frozen, old_anchor_reproduced=True)
        train_loss, epoch_updates, batches, diagnostics = train_one_epoch(model, loss_module,
            ProgressLoader(train_loader, epoch + 1), optimizer, scaler, torch.device("cuda"),
            cfg["training"], freeze_brain_encoder=frozen, return_stats=True,
            return_diagnostics=True, epoch_index=epoch)
        if not np.isfinite(train_loss) or diagnostics["prefix_revision_delay_counts"] != expected_counts:
            raise RuntimeError("Non-finite loss or incomplete supervision coverage.")
        updates += epoch_updates
        status(status="validating", epoch=epoch + 1, completed_batches=batches,
            epoch_optimizer_updates=epoch_updates)
        metrics, encoded = validate(model, val_loader, val, vocabulary)
        if encoded["event_ids"] != reference_ids:
            raise ValueError("Validation event identities/order changed.")
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": train_loss,
            "optimizer_updates": updates, "epoch_optimizer_updates": epoch_updates,
            "batches": batches, "amp_skipped_updates": batches - epoch_updates,
            "seconds": time.perf_counter() - started, "brain_encoder_frozen": frozen,
            "learning_rate": optimizer.param_groups[0]["lr"], "training_diagnostics": diagnostics,
            "validation": metrics, "gate": gate_result(metrics["strict_vocab50"], old_score, trial["maximum_drop"])}
        history.append(row)
        scores = {"legacy50": metrics["legacy"][cfg["evaluation"]["selection_metric"]],
            "full_vocab": metrics["full_vocabulary"]["selection_score"]}
        stale += 1
        for selection, score in scores.items():
            if score > best[selection]:
                best[selection], best_rows[selection] = score, row
                if selection == "legacy50":
                    stale = 0
                payload = legacy.checkpoint_payload(model, loss_module, cfg, epoch + 1,
                    metrics["legacy"], train.channel_names, train.channel_positions,
                    optimizer_updates=updates, initialization_audit=initialization)
                payload["alignment_validation"] = metrics
                payload["alignment_selection"] = selection
                save_checkpoint(out / ("best.pt" if selection == "legacy50" else "best_full_vocab.pt"), payload)
        save_json(out / "progress.json", {"history": history, "best": best_rows})
        status(status="epoch_complete", completed_epochs=epoch + 1, best_epoch=best_rows["legacy50"]["epoch"],
            best_gate=best_rows["legacy50"]["gate"], epoch_seconds=row["seconds"])
        del encoded
        if stale >= cfg["training"]["patience"]:
            break
    model.load_state_dict(load_checkpoint(out / "best.pt")["model_state"], strict=True)
    status(status="final_validation", completed_epochs=len(history))
    metrics, encoded = validate(model, val_loader, val, vocabulary)
    gate = gate_result(metrics["strict_vocab50"], old_score, trial["maximum_drop"])
    save_json(out / "evaluation/val_alignment.json", metrics)
    np.savez_compressed(out / "evaluation/val_predictions.npz",
        event_ids=np.asarray(encoded["event_ids"]), words=np.asarray(encoded["words"]),
        **{f"k{d}": v.numpy() for d, v in encoded["revision_predictions"].items()},
        **{f"available_k{d}": v.numpy() for d, v in encoded["revision_availability_masks"].items()})
    result = {"status": "completed", "dataset": "pallier2025", "seed": 0,
        "changed_training_factor": trial["changed_training_factor"], "gate": gate,
        "best": best_rows, "history": history, "test_evaluated": False,
        "full_protocol_validated": False, "output_dir": str(out)}
    save_json(out / "training_summary.json", result)
    save_json(REPORT / "seed-000.json", result)
    lines = ["# Pallier seed 0：全覆盖单因素验证", "",
        "保留旧 16 层/16 头、T5、CNN 初始化、损失与优化，仅改变截止位置覆盖。",
        "主检查点按原共同查询 K2 指标选模；严格 50 候选用于准入评价。",
        "", "| 延迟 | 旧 Macro Top-10 | 本次 Macro Top-10 | 差值（百分点） |", "|---|---:|---:|---:|"]
    for d in range(3):
        old = old_horizons[str(d)]["all_targets"]["vocabularies"]["50"]["retrieval"]["macro_top10"]
        new = metrics["strict_vocab50"][str(d)]["all_targets"]["macro_recall_at_10"]
        lines.append(f"| K{d} | {old:.2%} | {new:.2%} | {100*(new-old):+.2f} |")
    lines += ["", f"单种子门槛：{'通过' if gate['passed'] else '未通过'}。",
        f"共同词位严格 50 候选 K2−K0：{100*gate['common_k2_minus_k0']:.2f} 个百分点。",
        "本结果只检验覆盖变更，不代表整个 BGE 统一协议已获验证。完整队列未启动。"]
    (REPORT / "seed-000.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_run_status(out, "completed")
    status(status="completed", gate=gate, report=str(REPORT / "seed-000.md"))


def main():
    try:
        run()
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
