"""Pallier seed 0 BGE word warm-start followed by 16-layer coverage revision."""
from __future__ import annotations

import copy
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from braindecoding.config import PROJECT_ROOT, project_path
from braindecoding.data.text import text_embedding_signature
from braindecoding.experiment import file_sha256, run_directory
from braindecoding.pallier_alignment_trial import strict_bank_metrics, gate_result
from braindecoding.reporting import runtime_cache_dir, study_report_dir
from braindecoding.tasks.word_decoding.pallier2025 import train as legacy
from braindecoding.training.runtime import load_checkpoint, save_checkpoint, set_seed, cpu_state_dict
from braindecoding.training.word import make_loader, train_one_epoch
from braindecoding.unified_rolling_training import full_vocabulary_metrics

CONFIG = PROJECT_ROOT / "configs/unified_rolling/pallier_seed0_bge_alignment_v1.yaml"
RUNTIME = runtime_cache_dir("pallier_seed0_bge_alignment_v1")
REPORT = study_report_dir("pallier2025", "baseline_alignment_bge_v1")
_STATUS = {}


def save_json(path, payload):
    """Atomic JSON with bounded retries for Windows readers holding the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    for attempt in range(8):
        try:
            temp.write_text(content, encoding="utf-8")
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(0.05 * 2**attempt, 1.0))


def status(**fields):
    path = RUNTIME / "status.json"
    _STATUS.update(fields, pid=os.getpid(), updated_unix=time.time(), test_evaluated=False)
    try:
        save_json(path, _STATUS)
    except OSError as error:
        # Telemetry is not a training dependency. Retain the newest state in
        # memory and retry it on the next update, while keeping stdout live.
        print(f"Status write deferred; training continues: {error}", file=sys.stderr, flush=True)
    print(json.dumps(fields, ensure_ascii=False), flush=True)


def configs():
    revision = legacy.load_config(CONFIG)
    trial = revision["alignment_trial"]
    word = legacy.load_config(project_path(trial["word_source_config"]))
    word["experiment"]["output_group"] = revision["experiment"]["output_group"]
    word["text_embedding"] = copy.deepcopy(revision["text_embedding"])
    word["alignment_trial"] = copy.deepcopy(trial)
    word["provenance"]["text_embedding"] = "bge_m3_final_cls_normalized"
    for cfg in (word, revision):
        # The canonical legacy resolver always injects its T5 cache; this
        # isolated entry explicitly overrides it and records the real path.
        cfg["cache"]["text_embeddings"] = str(project_path(trial["embedding_cache"]))
        cfg["training"]["output_dir"] = str(run_directory(cfg))
    revision["training"]["pretrained_brain_encoder_checkpoint"] = str(run_directory(word) / "best.pt")
    return word, revision


class ProgressLoader:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for i, batch in enumerate(self.loader, 1):
            yield batch
            if i == 1 or i % 25 == 0:
                status(status="training", completed_batches=i)


def validate(model, loader, dataset, vocabulary, revision):
    metrics, encoded = legacy.evaluate_loader(model, loader, torch.device("cuda"), amp=True)
    result = {"legacy": metrics, "full_vocabulary": full_vocabulary_metrics(encoded, dataset, vocabulary)}
    if revision:
        result["strict_vocab50"] = strict_bank_metrics(encoded, dataset, legacy.frozen_vocabulary())
        if any(result["strict_vocab50"][str(d)]["all_targets"]["query_count"] != 8260
               or result["strict_vocab50"][str(d)]["common_targets"]["query_count"] != 6760
               for d in range(3)):
            raise ValueError("Revision evaluation query population changed.")
    elif metrics["candidate_count"] != 50 or metrics["query_count"] != 8260:
        raise ValueError("Word evaluation population changed.")
    return result, encoded


def run_stage(cfg, stage, frozen_hashes, old_score, t5_score):
    revision = stage == "revision"
    tc, out = cfg["training"], Path(cfg["training"]["output_dir"])
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    for path, expected in frozen_hashes.items():
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"Source changed: {path}")
    set_seed(0)
    datasets, loaders, samplers = {}, {}, {}
    for split in ("train", "val"):
        table = legacy.dataset_module.load_event_table(cfg["cache"]["event_table"], split=split,
            subjects=cfg["dataset"]["subjects"], trainable_only=True)
        ds = legacy.build_dataset(cfg, table)
        loader, sampler = make_loader(ds, tc, shuffle=split == "train", group_column=legacy.loader_group_column(cfg))
        datasets[split], loaders[split], samplers[split] = ds, loader, sampler
    train, val = datasets["train"], datasets["val"]
    if (len(train), len(val)) != (115290, 16960) or set(train.table.event_id) & set(val.table.event_id):
        raise ValueError("Data split mismatch.")
    vocabulary = sorted(set(train.table.normalized_word))
    if len(vocabulary) != 2020:
        raise ValueError("Training vocabulary changed.")
    sizes = train.table.groupby(legacy.loader_group_column(cfg), sort=False).size().to_numpy()
    expected_counts = {d: int(np.maximum(sizes-d, 0).sum()) for d in range(3)}
    model = legacy.build_brain_embedding_model(train.channel_count, train.channel_positions,
        train.subject_count, cfg["model"]).cuda()
    loss_module = legacy.build_siglip_loss(cfg["loss"]).cuda()
    initialization = None
    if revision:
        source = Path(tc["pretrained_brain_encoder_checkpoint"])
        checkpoint = load_checkpoint(source)
        if (checkpoint["stage"] != "word" or checkpoint["seed"] != 0
                or checkpoint["text_embedding_config"] != cfg["text_embedding"]
                or checkpoint["model_config"]["conv"] != cfg["model"]["conv"]
                or checkpoint["channel_names"] != train.channel_names
                or checkpoint["embedding_sha256"] != file_sha256(cfg["cache"]["text_embeddings"])):
            raise ValueError("BGE warm-start contract mismatch.")
        model.brain_encoder.load_state_dict({k.removeprefix("brain_encoder."):v
            for k,v in checkpoint["model_state"].items() if k.startswith("brain_encoder.")}, strict=True)
        initialization = {"path":str(source), "sha256":file_sha256(source), "epoch":checkpoint["epoch"]}
        del checkpoint
    optimizer = legacy.build_adamw_for_modules([model, loss_module], tc)
    scheduler = legacy.build_cosine_annealing_scheduler(optimizer, tc["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    out.mkdir(parents=True)
    embedding_hash = file_sha256(cfg["cache"]["text_embeddings"])
    manifest = {"status":"running", "config":cfg, "source_sha256":frozen_hashes,
        "embedding_sha256":embedding_hash, "initialization":initialization,
        "event_table_sha256":file_sha256(cfg["cache"]["event_table"]),
        "text_signature":text_embedding_signature(cfg["text_embedding"]),
        "stage":stage, "seed":0, "test_evaluated":False}
    save_json(out / "run_manifest.json", manifest)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    history, best, best_rows, stale, updates = [], {"legacy50":-float("inf"), "full_vocab":-float("inf")}, {}, 0, 0
    for epoch in range(tc["epochs"]):
        for path, expected in frozen_hashes.items():
            if file_sha256(PROJECT_ROOT / path) != expected:
                raise RuntimeError(f"Source changed during training: {path}")
        started = time.perf_counter()
        samplers["train"].set_epoch(epoch)
        frozen = revision and epoch < tc["freeze_brain_encoder_epochs"]
        legacy.set_brain_encoder_trainable(model, not frozen)
        status(status="training", stage=stage, epoch=epoch+1, completed_epochs=epoch,
            completed_batches=0, output_dir=str(out), max_epochs=tc["epochs"], best_gate=None,
            brain_encoder_frozen=frozen)
        loss, epoch_updates, batches, diag = train_one_epoch(model, loss_module,
            ProgressLoader(loaders["train"]), optimizer, scaler, torch.device("cuda"), tc,
            freeze_brain_encoder=frozen, return_stats=True, return_diagnostics=True, epoch_index=epoch)
        if not np.isfinite(loss) or (revision and diag["prefix_revision_delay_counts"] != expected_counts):
            raise RuntimeError("Non-finite loss or incomplete revision coverage.")
        updates += epoch_updates
        status(status="validating", completed_batches=batches)
        metrics, encoded = validate(model, loaders["val"], val, vocabulary, revision)
        if set(encoded["event_ids"]) != set(val.table.event_id):
            raise ValueError("Validation event identities changed.")
        scheduler.step()
        row = {"epoch":epoch+1, "train_loss":loss, "seconds":time.perf_counter()-started,
            "optimizer_updates":updates, "amp_skipped_updates":batches-epoch_updates,
            "training_diagnostics":diag, "validation":metrics}
        if revision:
            row["gate"] = gate_result(metrics["strict_vocab50"], old_score, cfg["alignment_trial"]["maximum_drop"])
            row["difference_vs_t5_coverage_pp"] = 100*(row["gate"]["k2_macro_top10"]-t5_score)
        history.append(row)
        scores = {"legacy50":metrics["legacy"][cfg["evaluation"]["selection_metric"]],
            "full_vocab":metrics["full_vocabulary"]["selection_score"]}
        stale += 1
        for selection, score in scores.items():
            if score > best[selection]:
                best[selection], best_rows[selection] = score, row
                if selection == "legacy50":
                    stale = 0
                save_checkpoint(out / ("best.pt" if selection == "legacy50" else "best_full_vocab.pt"),
                    {"model_state":cpu_state_dict(model), "loss_state":cpu_state_dict(loss_module),
                    "model_config":cfg["model"], "text_embedding_config":cfg["text_embedding"],
                    "embedding_sha256":embedding_hash, "config":cfg, "stage":stage, "seed":0,
                    "epoch":epoch+1, "channel_names":train.channel_names, "metrics":metrics,
                    "selection":selection, "test_evaluated":False})
        save_json(out / "progress.json", {"history":history, "best":best_rows})
        status(status="epoch_complete", completed_epochs=epoch+1,
            best_epoch=best_rows["legacy50"]["epoch"], best_selection_score=best["legacy50"],
            best_gate=best_rows["legacy50"].get("gate"), epoch_seconds=row["seconds"])
        del encoded
        if stale >= tc["patience"]:
            break
    summary = {"status":"completed", "stage":stage, "best":best_rows, "history":history,
        "test_evaluated":False, "full_queue_started":False, "output_dir":str(out)}
    save_json(out / "training_summary.json", summary)
    save_json(REPORT / f"seed-000_{stage}.json", summary)
    manifest["status"] = "completed"
    save_json(out / "run_manifest.json", manifest)
    del model, loss_module, optimizer, scheduler, scaler, datasets, loaders, train, val
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    global CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    CONFIG = project_path(args.config)
    try:
        word, revision = configs()
        if revision["alignment_trial"]["status"] != "authorized_single_seed_only":
            raise ValueError("Single-seed authorization required.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")
        torch.set_num_threads(2)
        # Compare scientific settings against the accepted T5 coverage trial.
        ref = project_path(revision["alignment_trial"]["coverage_reference_directory"])
        t5_cfg = legacy.load_config(ref / "resolved_config.yaml")
        for key in ("model", "loss", "dataset", "evaluation"):
            if revision[key] != t5_cfg[key]:
                raise ValueError(f"Unexpected non-text revision change: {key}")
        ignored = {"output_dir", "warm_start_output_group", "pretrained_brain_encoder_checkpoint"}
        if ({k:v for k,v in revision["training"].items() if k not in ignored}
                != {k:v for k,v in t5_cfg["training"].items() if k not in ignored}):
            raise ValueError("Revision optimization changed.")
        t5_summary = json.loads((ref / "training_summary.json").read_text(encoding="utf-8"))
        t5_score = t5_summary["best"]["legacy50"]["gate"]["k2_macro_top10"]
        old_dir = project_path(revision["alignment_trial"]["baseline_directory"])
        old = json.loads((old_dir / "evaluation/val.json").read_text(encoding="utf-8"))
        old_score = old["rolling_revision"]["horizons"]["2"]["all_targets"]["vocabularies"]["50"]["retrieval"]["macro_top10"]
        sources = [CONFIG, Path(__file__), PROJECT_ROOT / "braindecoding/training/pallier_alignment_trial.py",
            PROJECT_ROOT / "braindecoding/training/word.py", PROJECT_ROOT / "braindecoding/models.py",
            PROJECT_ROOT / "braindecoding/losses.py", PROJECT_ROOT / "braindecoding/training/unified_rolling_training.py",
            PROJECT_ROOT / "braindecoding/tasks/word_decoding/pallier2025/train.py",
            PROJECT_ROOT / "configs/unified_rolling/pallier_seed0_coverage_alignment_v1.yaml",
            PROJECT_ROOT / "configs/word_decoding/pallier2025/base.yaml"]
        hashes = {str(p.relative_to(PROJECT_ROOT)):file_sha256(p) for p in sources if Path(p).is_file()}
        status(status="preparing", started_unix=time.time(), dataset="pallier2025", seed=0,
            stages=["word", "revision"], completed_stages=[])
        run_stage(word, "word", hashes, old_score, t5_score)
        status(status="stage_complete", completed_stages=["word"])
        result = run_stage(revision, "revision", hashes, old_score, t5_score)
        status(status="completed", completed_stages=["word", "revision"],
            best_gate=result["best"]["legacy50"]["gate"], report=str(REPORT / "seed-000_revision.json"))
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
