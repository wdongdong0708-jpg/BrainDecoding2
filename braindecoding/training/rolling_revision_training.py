"""Shared training and resource checks for both unified-protocol datasets.

The legacy task entry points keep their frozen contracts. This explicit v1
development entry point uses a shared loss, vocabulary and model-selection rule.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from braindecoding.config import PROJECT_ROOT, load_yaml_with_extends, project_path
from braindecoding.data import chineseeeg2, pallier2025
from braindecoding.data.text import encode_words, load_text_model, save_text_embedding_cache
from braindecoding.losses import build_siglip_loss
from braindecoding.models import build_brain_embedding_model
from braindecoding.optimizers import build_adamw_for_modules, build_cosine_annealing_scheduler
from braindecoding.reporting import REPORT_AUDITS_ROOT
from braindecoding.training.runtime import cpu_state_dict, load_checkpoint, save_checkpoint, set_seed
from braindecoding.training.word import SentenceBatchSampler, encode_loader, move_batch
from braindecoding.unified_protocol_audit import DATASETS, MANIFEST_ROOT, load_source, sha256, write_json

CONFIG_PATH = PROJECT_ROOT / "configs/training/legacy_rolling_revision_protocol.yaml"
TRAINING_SOURCES = (
    "braindecoding/training/unified_rolling_training.py", "braindecoding/training/unified_protocol_audit.py",
    "braindecoding/config.py", "braindecoding/models.py", "braindecoding/losses.py",
    "braindecoding/optimizers.py", "braindecoding/training/runtime.py", "braindecoding/training/word.py",
    "braindecoding/data/text.py", "braindecoding/data/pallier2025.py", "braindecoding/data/chineseeeg2.py",
    "braindecoding/data/sensors.py",
)


def config():
    return load_yaml_with_extends(CONFIG_PATH)


def embedding_path(dataset):
    return PROJECT_ROOT / "data/cache/derived" / dataset / "text/unified_bge_m3_v1/embeddings.npz"


def prepare_embeddings():
    cfg = config()["text_embedding"]
    inventories = {}
    for dataset in DATASETS:
        words = set()
        for split in ("train", "val"):
            _, table = load_source(dataset, split)
            words.update(table.normalized_word.astype(str))
        inventories[dataset] = sorted(words)
    words = sorted(set().union(*map(set, inventories.values())))
    tokenizer, model, device = load_text_model(cfg)
    matrix = encode_words(words, cfg, tokenizer=tokenizer, model=model, device=device)
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5):
        raise RuntimeError("Shared embeddings are non-finite or not normalized.")
    index = {word: i for i, word in enumerate(words)}
    result = {}
    for dataset, vocabulary in inventories.items():
        path = embedding_path(dataset)
        save_text_embedding_cache(path, vocabulary, matrix[[index[w] for w in vocabulary]], cfg)
        result[dataset] = {"words": len(vocabulary), "path": str(path), "sha256": sha256(path)}
    write_json(REPORT_AUDITS_ROOT / "unified_rolling_v1/shared_embeddings.json", result)
    return result


def choose_smoke_groups(table, per_subject=3):
    chosen = []
    for _, rows in table.groupby("subject_id", sort=True):
        # Select complete groups; never truncate an individual group's future.
        candidates = rows.groupby("protocol_group", sort=False).size()
        candidates = candidates[candidates >= 5]
        chosen.extend(candidates.index[:per_subject])
    return table[table.protocol_group.isin(chosen)].reset_index(drop=True)


def make_dataset(dataset, split, *, smoke=False):
    source, table = load_source(dataset, split)
    # Canonical EEG products are stored under per-subject directories, as in
    # resolve_experiment_config; loading the source YAML alone does not inject it.
    source["dataset"]["signal_cache_subject_subdirectories"] = True
    if smoke:
        table = choose_smoke_groups(table)
    # Audit rejects gaps until an explicit common split-at-gap rule is implemented.
    for _, rows in table.groupby("protocol_group", sort=False):
        if np.any(np.diff(rows["记录内序号"].to_numpy(dtype=int)) != 1):
            raise ValueError("Excluded-event gap needs an explicit group boundary.")
    cfg = config()
    if dataset == "pallier2025":
        result = pallier2025.Pallier2025WordDataset(
            table, source["dataset"], project_path(source["cache"]["meg_dir"]),
            embedding_path(dataset), cfg["text_embedding"])
    else:
        result = chineseeeg2.ChineseEEG2LittlePrinceWordDataset(
            table, source["dataset"], project_path(source["cache"]["eeg_dir"]),
            embedding_path(dataset), cfg["text_embedding"])
    result.table["protocol_group"] = table.protocol_group.to_numpy()
    return result


def make_epoch_loader(dataset, cfg, *, epoch=0, shuffle=True):
    sampler = SentenceBatchSampler(dataset.table.protocol_group, cfg["batch_size"],
        shuffle, cfg["seed"], preserve_complete_groups=True)
    sampler.set_epoch(epoch)
    # Materialize the actual shuffled packing; __len__ based on unshuffled groups
    # can differ by one or more batches and is not an objective normalizer.
    batches = list(sampler)
    return DataLoader(dataset, batch_sampler=batches, num_workers=0, pin_memory=True)


def equal_event_horizon_loss(losses, counts, totals, batch_count):
    """Mean across actual batches equals the whole-epoch equal-horizon objective."""
    if set(totals) != {0, 1, 2} or any(v <= 0 for v in totals.values()):
        raise ValueError("All three horizons need a positive epoch denominator.")
    terms = [losses[d] * (batch_count * counts[d] / totals[d]) for d in losses]
    return torch.stack(terms).sum() / 3


def train_epoch(model, loss_module, ds, optimizer, cfg, *, epoch, revision, frozen):
    loader = make_epoch_loader(ds, cfg, epoch=epoch)
    sizes = ds.table.groupby("protocol_group").size().to_numpy()
    totals = {d: int(np.maximum(sizes-d, 0).sum()) for d in range(3)}
    model.train()
    model.brain_encoder.requires_grad_(not frozen)
    if frozen:
        model.brain_encoder.eval()
    loss_module.train()
    visited = {d: set() for d in (range(3) if revision else [0])}
    observations = {d: 0 for d in visited}
    objective = 0.0
    started = time.perf_counter()
    for batch in loader:
        signal, targets, subject, group = move_batch(batch, torch.device("cuda"))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=getattr(torch, cfg["amp_dtype"]), enabled=cfg["amp"]):
            if revision:
                estimates, rows, delays = model.forward_prefix_revision_training(
                    signal, subject, group, cutoff_policy="all")
                losses, counts = {}, {}
                for d in range(3):
                    mask = delays == d
                    count = int(mask.sum())
                    if count:
                        losses[d] = loss_module.forward_rectangular(estimates[mask], targets, rows[mask])
                        counts[d] = count
                    for row in rows[mask].detach().cpu().tolist():
                        visited[d].add(batch["event_id"][row])
                    observations[d] += count
                loss = equal_event_horizon_loss(losses, counts, totals, len(loader))
            else:
                estimates = model(signal, subject, group)
                loss = loss_module(estimates, targets) * (len(loader) * len(signal) / len(ds))
                visited[0].update(batch["event_id"])
                observations[0] += len(signal)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        # BF16 avoids loss scaling and silently skipped AMP optimizer updates.
        finite = [torch.isfinite(p.grad).all() for module in (model, loss_module)
            for p in module.parameters() if p.grad is not None]
        if not bool(torch.stack(finite).all()):
            raise FloatingPointError("Non-finite gradient; no optimizer update was performed.")
        optimizer.step()
        objective += float(loss.detach())
    torch.cuda.synchronize()
    for d in visited:
        if len(visited[d]) != totals[d] or observations[d] != totals[d]:
            raise RuntimeError(f"Coverage mismatch at K{d}.")
    return {"loss": objective / len(loader), "optimizer_updates": len(loader),
        "seconds": time.perf_counter()-started,
        "available_targets": {str(d): totals[d] for d in visited},
        "supervision_counts": {str(d): observations[d] for d in visited},
        "unique_targets": {str(d): len(visited[d]) for d in visited},
        "duplicate_targets": {str(d): observations[d]-len(visited[d]) for d in visited},
        "coverage": {str(d): len(visited[d])/totals[d] for d in visited},
        "amp_dtype": "bfloat16", "skipped_optimizer_updates": 0}


def full_vocabulary_metrics(encoded, ds, vocabulary):
    bank = F.normalize(torch.tensor(np.stack([ds.embedding_map[w] for w in vocabulary]), dtype=torch.float32), dim=1)
    index = {w: i for i, w in enumerate(vocabulary)}
    targets = np.array([index.get(w, -1) for w in encoded["words"]])
    subject_map = dict(zip(ds.table.event_id, ds.table.subject_id))
    subjects = np.array([subject_map[event] for event in encoded["event_ids"]])
    predictions = encoded.get("revision_predictions", {0: encoded["predictions"]})
    masks = encoded.get("revision_availability_masks", {0: torch.ones(len(targets), dtype=torch.bool)})
    common = np.logical_and.reduce([mask.numpy() for mask in masks.values()])
    result = {}
    selection = []
    for delay, matrix in predictions.items():
        scores = F.normalize(matrix.float(), dim=1) @ bank.T
        top = scores.topk(min(10, len(vocabulary)), dim=1).indices.numpy()
        correct10 = (top == targets[:, None]).any(axis=1)
        correct1 = top[:, 0] == targets
        per_subject = {}
        for subject in sorted(set(subjects)):
            use = common & (subjects == subject)
            if not use.any():
                raise ValueError("No common-horizon validation positions for a subject.")
            per_subject[str(subject)] = {"n": int(use.sum()), "top1": float(correct1[use].mean()),
                "top10": float(correct10[use].mean())}
        macro = float(np.mean([v["top10"] for v in per_subject.values()]))
        selection.append(macro)
        result[f"k{delay}"] = {"subject_macro_top10": macro, "per_subject": per_subject}
    return {"selection_score": float(np.mean(selection)), "horizons": result,
        "candidate_count": len(vocabulary), "all_reference_words_retained": True,
        "reference_count": len(targets), "common_count": int(common.sum()),
        "reference_in_vocabulary": float((targets >= 0).mean())}


def output_dir(dataset, stage, seed, smoke=False):
    group = "unified_rolling_v1_smoke" if smoke else "unified_rolling_v1"
    scope = config()["adapters"][dataset]["subject_scope"]
    return PROJECT_ROOT / f"outputs/word_decoding/{dataset}/{group}/{scope}/{stage}/seed-{seed:03d}"


def causal_input_check(model, ds, cfg):
    model.eval()
    for batch in make_epoch_loader(ds, cfg, shuffle=False):
        group = batch["sentence_index"]
        candidate = next((value for value in torch.unique(group) if int((group==value).sum()) >= 4), None)
        if candidate is not None:
            break
    else:
        raise ValueError("The physical-window check needs a group with >=4 words.")
    signal, _, subject, group = move_batch(batch, torch.device("cuda"))
    ids = torch.where(group == int(candidate))[0]
    if len(ids) < 4:
        raise ValueError("The physical-window check needs a group with >=4 words.")
    signal, subject, group = signal[ids], subject[ids], group[ids]
    before = model.forward_prefix_revisions(signal, subject, group)
    changed = signal.clone()
    changed[3:] += torch.randn_like(changed[3:]) * 5
    after = model.forward_prefix_revisions(changed, subject, group)
    difference = float((before[0][2]-after[0][2]).abs().max())
    if difference > 1e-6:
        raise RuntimeError("Future signal perturbed K0.")
    return {"k0_max_difference_after_unavailable_future_perturbation": difference}


def run_training(dataset, stage, seed=0, *, smoke=False):
    cfg = config()
    if not smoke and cfg["protocol"]["status"] != "ready_for_development_training":
        raise RuntimeError("Complete smoke checks and record the development-training gate first.")
    out = output_dir(dataset, stage, seed, smoke)
    if (out / "summary.json").exists():
        raise FileExistsError(f"Do not overwrite an existing unified run: {out}")
    if not smoke and (out / "run_manifest.json").exists():
        raise FileExistsError(f"Incomplete run needs an explicit recovery decision: {out}")
    out.mkdir(parents=True, exist_ok=True)
    tc = copy.deepcopy(cfg["training"])
    tc["seed"] = seed
    set_seed(seed)
    train = make_dataset(dataset, "train", smoke=smoke)
    val = make_dataset(dataset, "val", smoke=smoke)
    manifest_path = MANIFEST_ROOT / f"{dataset}_train_vocabulary.json"
    vocabulary = json.loads(manifest_path.read_text(encoding="utf-8"))["vocabulary"]
    model_cfg = copy.deepcopy(cfg["model"])
    revision = stage == "revision"
    if not revision:
        model_cfg["use_transformer"] = False
        model_cfg.pop("local_window_revision")
    model = build_brain_embedding_model(train.channel_count, train.channel_positions, train.subject_count, model_cfg).cuda()
    loss_module = build_siglip_loss(cfg["loss"]).cuda()
    initialization = None
    if revision:
        source = output_dir(dataset, "word", seed, smoke) / "best.pt"
        checkpoint = load_checkpoint(source)
        if (checkpoint["dataset"], checkpoint["seed"], checkpoint["smoke"]) != (dataset, seed, smoke):
            raise ValueError("Warm-start dataset, seed or smoke status mismatch.")
        if checkpoint["protocol_config"]["text_embedding"] != cfg["text_embedding"]:
            raise ValueError("Word-model text target mismatch.")
        model.brain_encoder.load_state_dict({k.removeprefix("brain_encoder."): v
            for k, v in checkpoint["model_state"].items() if k.startswith("brain_encoder.")}, strict=True)
        initialization = {"path": str(source), "sha256": sha256(source), "source_seed": seed}
    optimizer = build_adamw_for_modules([model, loss_module], tc)
    epochs = (2 if revision else 1) if smoke else int(tc[f"{stage}_epochs"])
    scheduler = build_cosine_annealing_scheduler(optimizer, epochs)
    val_loader = make_epoch_loader(val, tc, shuffle=False)
    provenance = {"config": cfg, "config_sha256": sha256(CONFIG_PATH), "dataset": dataset,
        "stage": stage, "seed": seed, "smoke": smoke, "initialization": initialization,
        "vocabulary_manifest_sha256": sha256(manifest_path),
        "embedding_sha256": sha256(embedding_path(dataset)),
        "source_sha256": {p: sha256(PROJECT_ROOT/p) for p in TRAINING_SOURCES if (PROJECT_ROOT/p).is_file()},
        "source_dataset_config": load_source(dataset, "train")[0],
        "channel_names": train.channel_names,
        "test_evaluated": False}
    write_json(out / "run_manifest.json", provenance)
    history, best = [], -float("inf")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        frozen = revision and epoch < tc["freeze_brain_encoder_epochs"]
        stats = train_epoch(model, loss_module, train, optimizer, tc, epoch=epoch, revision=revision, frozen=frozen)
        validation_start = time.perf_counter()
        encoded = encode_loader(model, val_loader, torch.device("cuda"), amp=False)
        metrics = full_vocabulary_metrics(encoded, val, vocabulary)
        stats.update(epoch=epoch+1, brain_encoder_frozen=frozen, validation=metrics,
            validation_seconds=time.perf_counter()-validation_start)
        history.append(stats)
        if metrics["selection_score"] > best:
            best = metrics["selection_score"]
            save_checkpoint(out / "best.pt", {"model_state": cpu_state_dict(model),
                "loss_state": cpu_state_dict(loss_module), "model_config": model_cfg,
                "protocol_config": cfg, "dataset": dataset, "seed": seed,
                "epoch": epoch+1, "selection_score": best, "channel_names": train.channel_names,
                "test_evaluated": False, "smoke": smoke})
        scheduler.step()
        write_json(out / "progress.json", {"history": history, "status": "running"})
        print(json.dumps({"dataset":dataset,"stage":stage,"epoch":epoch+1,
            "seconds":round(stats["seconds"],2),"selection":metrics["selection_score"],
            "coverage":stats["coverage"]},ensure_ascii=False),flush=True)
    check = causal_input_check(model, train, tc) if revision else {}
    report = {"status": "smoke_complete" if smoke else "development_training_complete",
        "dataset": dataset, "stage": stage, "seed": seed, "train_rows": len(train), "val_rows": len(val),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "history": history, "best_score": best, "causal_input_check": check,
        "test_evaluated": False, "smoke_scores_are_not_performance_evidence": smoke}
    write_json(out / "summary.json", report)
    write_json(out / "progress.json", {"status": report["status"], "completed_epochs": epochs})
    del model, optimizer, loss_module
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "smoke", "train"])
    parser.add_argument("--dataset", choices=list(DATASETS), default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=["word", "revision", "both"], default="both")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare_embeddings(), ensure_ascii=False, indent=2))
        return
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    for dataset in datasets:
        for stage in (["word", "revision"] if args.stage == "both" else [args.stage]):
            run_training(dataset, stage, args.seed, smoke=args.action == "smoke")


if __name__ == "__main__":
    main()
