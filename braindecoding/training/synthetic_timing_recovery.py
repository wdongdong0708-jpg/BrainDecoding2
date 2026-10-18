"""Continue frozen synthetic controls and add Chinese EEG without rerunning completed work."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
import yaml

from braindecoding import synthetic_timing_control as old
from braindecoding import unified_rolling_v2_training as neural
from braindecoding.config import PROJECT_ROOT as ROOT
from braindecoding.data.synthetic_timing import CONDITIONS, SyntheticTimingDataset
from braindecoding.data.text import text_embedding_signature
from braindecoding.experiment import file_sha256
from braindecoding.pallier_alignment_trial import strict_bank_metrics
from braindecoding.reporting import study_report_dir
from braindecoding.training import epoch_recovery as recovery
from braindecoding.training.runtime import cpu_state_dict, load_checkpoint, save_checkpoint, set_seed
from braindecoding.training.word import encode_loader, make_loader, train_one_epoch

CONFIG = ROOT / "configs/unified_rolling/synthetic_timing_v2.yaml"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = PROTOCOL.with_suffix(".json")
REPORT = study_report_dir(None, "synthetic_timing_v2")
RUNTIME = ROOT / ".cache/runs/synthetic_timing_v2"
DATASETS = neural.DATASETS
read, save, digest = old.read, old.save, old.digest


def settings():
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p = spec["protocol"]
    if (p["datasets"] != list(DATASETS) or p["conditions"] != list(CONDITIONS)
            or p["seeds"] != [0, 1, 2] or p["stages"] != ["word", "revision"]
            or p["history_radius"] != 2 or p["horizons"] != [0, 1, 2]
            or p["test_selection_allowed"] or p["language_model_allowed"]
            or p["expand_context_automatically"]
            or spec["synthetic_from"] != str(old.CONFIG.relative_to(ROOT)).replace("\\", "/")
            or spec["execution"]["resume_boundary"] != "epoch"):
        raise ValueError("The authorized synthetic recovery contract changed")
    old.settings()
    return spec


def configure_environment():
    old.configure_environment()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def old_completed(condition, seed, stage):
    cfg = old.build_config(condition, seed, stage)
    path = Path(cfg["training"]["output_dir"])
    summary = path / "training_summary.json"
    return path if summary.exists() and read(summary).get("status") == "completed" else None


def build_config(dataset, condition, seed, stage):
    if dataset not in DATASETS or condition not in CONDITIONS or seed not in (0, 1, 2) or stage not in ("word", "revision"):
        raise ValueError("Unknown synthetic training job")
    if dataset == "pallier2025" and old_completed(condition, seed, stage) is not None:
        return old.build_config(condition, seed, stage)
    cfg = neural.build_config(dataset, seed, stage)
    group = f"{settings()['execution']['output_group_prefix']}_{condition}"
    cfg["experiment"]["output_group"] = group
    out = ROOT / "outputs/word_decoding" / dataset / group / cfg["experiment"]["subject_scope"]
    cfg["training"]["output_dir"] = str(out / cfg["experiment"]["id"] / f"seed-{seed:03d}")
    if stage == "revision":
        word = build_config(dataset, condition, seed, "word")
        cfg["training"]["warm_start_output_group"] = word["experiment"]["output_group"]
        cfg["training"]["pretrained_brain_encoder_checkpoint"] = str(Path(word["training"]["output_dir"]) / "best.pt")
    cfg["synthetic_control"] = {**old.settings()["synthetic"], "condition": condition,
                               "stream_grouping": "subject_recording_language_segment"}
    cfg["provenance"]["neural_input"] = f"synthetic_{condition}_no_neural_samples"
    return cfg


def dataset_for(cfg, split, table=None):
    dataset = cfg["experiment"]["dataset"]
    if split not in ("train", "val", "test"):
        raise ValueError("Unknown split")
    task = neural.pallier if dataset == "pallier2025" else neural.chinese
    reader = task.dataset_module.load_event_table
    if table is None:
        table = reader(cfg["cache"]["event_table"], split=split,
                       subjects=cfg["dataset"]["subjects"], trainable_only=True)
    resolved = copy.deepcopy(cfg)
    if split == "test":
        resolved["cache"]["text_embeddings"] = str(ROOT / "data/cache/derived" / dataset / "text/unified_heldout_test_v2/embeddings.npz")
    base = task.build_dataset(resolved, table, zero_signal=True)
    spec = cfg["synthetic_control"]
    return SyntheticTimingDataset(base, spec["condition"], split=split,
        training_seed=cfg["training"]["seed"],
        evaluation_seed=spec["validation_seed" if split == "val" else "test_seed"])


def dataset_and_loader(cfg, split):
    if split not in ("train", "val"):
        raise ValueError("Training cannot load test")
    dataset = dataset_for(cfg, split)
    loader, sampler = make_loader(dataset, cfg["training"], shuffle=split == "train",
                                  group_column=neural.group_column(cfg))
    return dataset, loader, old.EpochSampler(sampler, dataset)


def fixed_sources():
    return (CONFIG, PROTOCOL, Path(__file__), Path(recovery.__file__),
            old.CONFIG, old.PROTOCOL, old.MANIFEST, Path(old.__file__),
            ROOT / "braindecoding/data/synthetic_timing.py")


def source_hashes(cfg, adapter):
    hashes = neural.source_hashes(cfg, adapter)
    for path in (*fixed_sources(), MANIFEST):
        hashes[str(path.resolve())] = file_sha256(path)
    return hashes


def verify_manifest():
    m = read(MANIFEST)
    if digest({k: v for k, v in m.items() if k != "manifest_sha256"}) != m["manifest_sha256"]:
        raise ValueError("Recovery manifest digest changed")
    neural.check_hashes(m["source_sha256"])
    return m


def job_for(dataset, condition, seed, stage):
    return next(j for j in verify_manifest()["jobs"] if (j["dataset"], j["condition"], j["seed"], j["stage"]) == (dataset, condition, seed, stage))


def validate_completed(job):
    out = Path(job["output_dir"])
    summary = read(out / "training_summary.json")
    manifest = read(out / "run_manifest.json")
    if summary["status"] != "completed" or manifest["status"] != "completed" or manifest["config"] != build_config(
            job["dataset"], job["condition"], job["seed"], job["stage"]):
        raise ValueError(f"Completed synthetic run identity differs: {out}")
    neural.check_hashes(manifest["source_sha256"])
    expected = manifest.get("primary_checkpoint_sha256", job.get("checkpoint_sha256"))
    if expected and file_sha256(out / "best.pt") != expected:
        raise ValueError(f"Completed checkpoint changed: {out}")
    return summary


def audit_dataset(dataset, cfg):
    checks = []
    for indices in dataset.table.groupby(neural.group_column(cfg), sort=False).indices.values():
        for left, right in zip(indices[:-1], indices[1:]):
            offset = int(dataset.starts[right] - dataset.starts[left])
            if 0 < offset < dataset.window_samples:
                equal = bool(np.array_equal(dataset.raw_window(left)[:, offset:], dataset.raw_window(right)[:, :-offset]))
                if equal != (dataset.condition == "shared"):
                    raise AssertionError("Overlap sample identity is incorrect")
                checks.append({"left_event": dataset.event_ids[left], "sample_offset": offset,
                               "overlap_samples": dataset.window_samples - offset, "raw_overlap_equal": equal})
                break
        if len(checks) == 20:
            break
    if len(checks) != 20:
        raise AssertionError("Insufficient overlap identity checks")
    before = dataset.raw_window(0)
    dataset.set_epoch(1)
    changed = not np.array_equal(before, dataset.raw_window(0))
    if changed != (dataset.split == "train"):
        raise AssertionError("Training refresh / fixed evaluation mismatch")
    return {"dataset": cfg["experiment"]["dataset"], "condition": dataset.condition,
            "split": dataset.split, "checks": checks, "epoch_changed_signal": changed, "passed": True}


def prepare():
    if MANIFEST.exists():
        return verify_manifest()
    parent = old.verify_manifest()
    spec, hashes = settings(), {}
    REPORT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    jobs, audits, populations, test_population = [], [], {}, {}
    real = pd.read_csv(ROOT / spec["execution"]["real_results"])
    for dataset in DATASETS:
        for condition in CONDITIONS:
            cfg = build_config(dataset, condition, 0, "revision")
            train, val = dataset_for(cfg, "train"), dataset_for(cfg, "val")
            _, candidates, _, population = neural.populations(cfg, train, val, neural.config()["adapters"][dataset])
            if dataset in populations and populations[dataset] != population:
                raise ValueError("Synthetic conditions changed population")
            populations[dataset] = population
            audits.extend([audit_dataset(train, cfg), audit_dataset(val, cfg)])
            source_cfg = copy.deepcopy(cfg)
            source_cfg["training"].pop("pretrained_brain_encoder_checkpoint", None)
            hashes.update(neural.source_hashes(source_cfg, neural.config()["adapters"][dataset]))
            test = dataset_for(cfg, "test")
            common = test.table.groupby(neural.group_column(cfg), sort=False).cumcount(ascending=False).to_numpy() >= 2
            selected = test.table.normalized_word.isin(candidates).to_numpy()
            counts = {"rows": len(test), "all_targets": int(selected.sum()),
                      "common_targets": int((common & selected).sum()), "candidate_count": len(candidates),
                      "event_ids_sha256": digest(sorted(test.event_ids)), "candidates_sha256": digest(candidates)}
            for name in ("all_targets", "common_targets"):
                values = real.loc[real.dataset.eq(dataset) & real.population.eq(name)]
                if len(values) != 9 or values.query_count.nunique() != 1 or int(values.query_count.iloc[0]) != counts[name] or not values.candidate_count.eq(50).all():
                    raise ValueError("Synthetic test population differs from frozen Real")
            if dataset in test_population and counts != test_population[dataset]:
                raise ValueError("Conditions changed held-out targets")
            test_population[dataset] = counts
            audits.append(audit_dataset(test, cfg))
            del train, val, test
            gc.collect()
        for seed in (0, 1, 2):
            for condition in CONDITIONS:
                for stage in ("word", "revision"):
                    cfg = build_config(dataset, condition, seed, stage)
                    out = Path(cfg["training"]["output_dir"])
                    reuse = dataset == "pallier2025" and old_completed(condition, seed, stage) is not None
                    job = {"dataset": dataset, "condition": condition, "seed": seed, "stage": stage,
                           "output_dir": str(out), "reuse": reuse}
                    if reuse:
                        validate_completed(job)
                        for path in (out / "best.pt", out / "training_summary.json", out / "run_manifest.json"):
                            if Path(path).is_file():
                                hashes[str(path.resolve())] = file_sha256(path)
                        job["checkpoint_sha256"] = hashes[str((out / "best.pt").resolve())]
                        result = out / "evaluation/synthetic_test.json"
                        if stage == "revision" and result.exists():
                            record = read(result)
                            if record["manifest_sha256"] != parent["manifest_sha256"] or record["checkpoint_sha256"] != job["checkpoint_sha256"]:
                                raise ValueError("Reused test result identity differs")
                            hashes[str(result.resolve())] = file_sha256(result)
                    elif out.exists():
                        raise FileExistsError(f"Unfrozen output already exists: {out}")
                    jobs.append(job)
        for suffix in ("npz", "json"):
            path = ROOT / "data/cache/derived" / dataset / f"text/unified_heldout_test_v2/embeddings.{suffix}"
            hashes[str(path.resolve())] = file_sha256(path)
    references = read(ROOT / spec["execution"]["existing_training_results"])
    references = [r for r in references["completed"] if r["stage"] == "revision"]
    if len(references) != 6:
        raise ValueError("Both datasets need three frozen Real references")
    for r in references:
        if file_sha256(r["checkpoint"]) != r["checkpoint_sha256"]:
            raise ValueError("Frozen Real checkpoint changed")
        hashes[str(Path(r["checkpoint"]).resolve())] = r["checkpoint_sha256"]
    for path in (*fixed_sources(), ROOT / spec["execution"]["real_results"],
                 ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"):
        hashes[str(path.resolve())] = file_sha256(path)
    failed = Path(old.build_config("shared", 2, "revision")["training"]["output_dir"])
    failed_status = read(failed / "status.json")
    if psutil.pid_exists(failed_status["pid"]):
        raise RuntimeError("Original interrupted worker is still alive; do not duplicate it")
    diagnosis = {"observed_unix": time.time(), "prior_output_dir": str(failed),
        "prior_worker_status": failed_status, "prior_queue_status": read(old.RUNTIME / "status.json"),
        "completed_epochs": len(read(failed / "progress.json")["history"]),
        "python_traceback_recorded": "Traceback (most recent call last)" in (old.RUNTIME / "shared_seed2_revision.log").read_text(encoding="utf-8"),
        "full_resume_checkpoint_existed": (failed / "resume.pt").exists(),
        "cause": "unclean_process_exit_without_recorded_python_exception; external cause unconfirmed",
        "action": "restart_only_interrupted_revision_from_completed_matching_synthetic_CNN",
        "user_scope": "2026-10-10: fix third seed and add chineseeeg2_littleprince",
        "evidence_sha256": {str(p.resolve()): file_sha256(p) for p in (
            failed / "status.json", failed / "progress.json", old.RUNTIME / "queue.log", old.RUNTIME / "shared_seed2_revision.log")}}
    save(REPORT / "recovery_diagnosis.json", diagnosis)
    save(REPORT / "window_audit.json", {"passed": True, "audits": audits, "identity_checked_before_baseline": True})
    save(REPORT / "overlap.json", read(old.REPORT / "overlap.json"))
    m = {"protocol": spec["protocol"]["id"], "settings": spec, "frozen_unix": time.time(),
         "parent_manifest_sha256": parent["manifest_sha256"], "jobs": jobs,
         "reused_training_stages": sum(j["reuse"] for j in jobs), "total_training_stages": len(jobs),
         "new_training_stages": sum(not j["reuse"] for j in jobs), "pipeline_count": 12,
         "population": populations, "test_population": test_population, "real_references": references,
         "source_sha256": hashes, "recovery": diagnosis,
         "test_selection_allowed": False, "test_evaluation_authorized": True}
    m["manifest_sha256"] = digest(m)
    save(MANIFEST, m)
    export_report()
    return m


def run_stage(dataset, condition, seed, stage):
    job = job_for(dataset, condition, seed, stage)
    if job["reuse"]:
        return validate_completed(job)
    neural.assert_ready(neural.config())
    cfg = build_config(dataset, condition, seed, stage)
    adapter, revision = neural.config()["adapters"][dataset], stage == "revision"
    tc, out = cfg["training"], Path(job["output_dir"])
    if (out / "training_summary.json").exists():
        return validate_completed(job)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    hashes = source_hashes(cfg, adapter)
    neural.check_hashes(hashes)
    if out.exists():
        previous = read(out / "run_manifest.json")
        if previous["config"] != cfg or previous["source_sha256"] != hashes:
            raise ValueError("Partial run does not match the frozen recovery job")
        if recovery.process_matches(previous.get("worker_identity", {})):
            raise RuntimeError("Another worker still owns this run")
    out.mkdir(parents=True, exist_ok=True)
    identity = {"config": cfg, "source_sha256": hashes}
    status = neural.RunStatus(out, dataset, seed, stage)
    manifest = {"status": "preparing", "config": cfg, "source_sha256": hashes,
        "dataset": dataset, "stage": stage, "seed": seed, "condition": condition,
        "execution_protocol": "synthetic_timing_v2", "test_evaluated": False,
        "worker_identity": {"pid": os.getpid(), "created": psutil.Process().create_time()}}
    save(out / "run_manifest.json", manifest)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        status(status="preparing")
        set_seed(seed)
        torch.set_num_threads(2)
        train, train_loader, sampler = dataset_and_loader(cfg, "train")
        val, val_loader, _ = dataset_and_loader(cfg, "val")
        vocabulary, vocab50, expected_counts, population = neural.populations(cfg, train, val, adapter)
        model = neural.build_brain_embedding_model(train.channel_count, train.channel_positions, train.subject_count, cfg["model"]).cuda()
        loss_module = neural.build_siglip_loss(cfg["loss"]).cuda()
        embedding_hash, initialization = file_sha256(cfg["cache"]["text_embeddings"]), None
        if revision:
            source = Path(tc["pretrained_brain_encoder_checkpoint"])
            for name in ("run_manifest.json", "training_summary.json"):
                record = read(source.parent / name)
                if record["status"] != "completed" or record["stage"] != "word":
                    raise ValueError("Revision requires a completed matching synthetic CNN")
            checkpoint = load_checkpoint(source)
            old.validate_word_checkpoint(checkpoint, cfg, train, embedding_hash)
            model.brain_encoder.load_state_dict({key.removeprefix("brain_encoder."): value
                for key, value in checkpoint["model_state"].items() if key.startswith("brain_encoder.")}, strict=True)
            initialization = {"path": str(source), "sha256": file_sha256(source), "epoch": checkpoint["epoch"]}
            del checkpoint
        optimizer = neural.build_adamw_for_modules([model, loss_module], tc)
        scheduler = neural.build_cosine_annealing_scheduler(optimizer, tc["epochs"])
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        manifest.update(status="running", embedding_sha256=embedding_hash, initialization=initialization,
            text_signature=text_embedding_signature(cfg["text_embedding"]), population=population,
            vocabulary50=vocab50, expected_supervision_counts=expected_counts if revision else None,
            train_rows=len(train), validation_rows=len(val), group_column=neural.group_column(cfg))
        state = {"history": [], "best": {"legacy50": -float("inf"), "full_vocab": -float("inf")},
                 "best_rows": {}, "stale": 0, "updates": 0, "next_epoch": 0}
        best_hashes = {}
        if (out / "resume.pt").exists():
            state, best_hashes = recovery.restore_boundary(out, model, loss_module, optimizer, scheduler, scaler, identity=identity)
            manifest["resumed_after_epoch"] = state["next_epoch"]
            status(status="resumed", completed_epochs=state["next_epoch"])
        else:
            manifest["resumed_after_epoch"] = 0
        save(out / "run_manifest.json", manifest)
        history, best, best_rows = state["history"], state["best"], state["best_rows"]
        stale, updates = state["stale"], state["updates"]
        epochs = range(state["next_epoch"], tc["epochs"]) if stale < tc["patience"] else []
        for epoch in epochs:
            neural.check_hashes(hashes)
            started = time.perf_counter()
            sampler.set_epoch(epoch)
            frozen = revision and epoch < tc["freeze_brain_encoder_epochs"]
            neural.pallier.set_brain_encoder_trainable(model, not frozen)
            status(status="training", epoch=epoch + 1, completed_epochs=epoch, completed_batches=0,
                   max_epochs=tc["epochs"], brain_encoder_frozen=frozen)
            loss, epoch_updates, batches, diagnostics = train_one_epoch(model, loss_module,
                neural.ProgressLoader(train_loader, status), optimizer, scaler, torch.device("cuda"), tc,
                freeze_brain_encoder=frozen, return_stats=True, return_diagnostics=True, epoch_index=epoch)
            if not np.isfinite(loss) or (revision and diagnostics["prefix_revision_delay_counts"] != expected_counts):
                raise RuntimeError("Non-finite loss or incomplete revision supervision")
            updates += epoch_updates
            status(status="validating", completed_batches=batches)
            metrics = neural.validate(model, val_loader, val, vocabulary, vocab50, cfg, adapter, revision)
            scheduler.step()
            row = {"epoch": epoch + 1, "train_loss": loss, "seconds": time.perf_counter() - started,
                   "optimizer_updates": updates, "epoch_optimizer_updates": epoch_updates, "batches": batches,
                   "amp_skipped_updates": batches - epoch_updates, "brain_encoder_frozen": frozen,
                   "training_diagnostics": diagnostics, "validation": metrics}
            history.append(row)
            scores = {"legacy50": metrics["legacy"][cfg["evaluation"]["selection_metric"]],
                      "full_vocab": metrics["full_vocabulary"]["selection_score"]}
            stale += 1
            for selection, score in scores.items():
                if not np.isfinite(score):
                    raise FloatingPointError("Non-finite validation selection score")
                if score > best[selection]:
                    best[selection], best_rows[selection] = score, row
                    if selection == "legacy50":
                        stale = 0
                    name = "best.pt" if selection == "legacy50" else "best_full_vocab.pt"
                    recovery.protect_committed_best(out, name, best_hashes)
                    save_checkpoint(out / name, {
                        "dataset": dataset, "model_state": cpu_state_dict(model), "loss_state": cpu_state_dict(loss_module),
                        "model_config": cfg["model"], "loss_config": cfg["loss"], "text_embedding_config": cfg["text_embedding"],
                        "embedding_sha256": embedding_hash, "config": cfg, "stage": stage, "seed": seed,
                        "epoch": epoch + 1, "channel_names": train.channel_names,
                        "channel_positions": np.asarray(train.channel_positions).tolist(),
                        "subject_ids": list(getattr(train, "subject_ids", cfg["dataset"]["subjects"])),
                        "metrics": metrics, "selection": selection, "initialization": initialization, "test_evaluated": False})
                    best_hashes[name] = file_sha256(out / name)
            status(status="saving_recovery", completed_epochs=epoch)
            recovery.save_boundary(out, model, loss_module, optimizer, scheduler, scaler, identity=identity,
                state={"history": history, "best": best, "best_rows": best_rows, "stale": stale,
                       "updates": updates, "next_epoch": epoch + 1}, best_hashes=best_hashes)
            save(out / "progress.json", {"history": history, "best": best_rows})
            status(status="epoch_complete", completed_epochs=epoch + 1, best_epoch=best_rows["legacy50"]["epoch"],
                   best_selection_score=best["legacy50"], epoch_seconds=row["seconds"], recovery_boundary_epoch=epoch + 1)
            if stale >= tc["patience"]:
                break
        neural.check_hashes(hashes)
        recovery.restore_committed_best(out, best_hashes)
        summary = {"status": "completed", "dataset": dataset, "seed": seed, "condition": condition, "stage": stage,
                   "best": best_rows, "history": history, "completed_epochs": len(history),
                   "stop_reason": "patience_exhausted" if stale >= tc["patience"] else "maximum_epochs",
                   "test_evaluated": False, "output_dir": str(out), "initialization": initialization}
        save(out / "training_summary.json", summary)
        save(REPORT / dataset / condition / f"seed-{seed:03d}_{stage}.json", summary)
        manifest.update(status="completed", stop_reason=summary["stop_reason"], completed_epochs=len(history),
                        primary_checkpoint_sha256=best_hashes["best.pt"])
        save(out / "run_manifest.json", manifest)
        status(status="completed", completed_epochs=len(history), stop_reason=summary["stop_reason"])
        if not settings()["execution"]["keep_completed_resume_checkpoint"]:
            (out / "resume.pt").unlink(missing_ok=True)
        return summary
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}")
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(out / "run_manifest.json", manifest)
        raise


def smoke_segment(table, candidates, group_column):
    # Event tables may interleave subjects; logical neighbors need not occupy
    # adjacent dataframe rows. Follow the same groups as the rolling sampler.
    selected = table.normalized_word.isin(candidates).to_numpy()
    for indices in table.groupby(group_column, sort=False).indices.values():
        for position, index in enumerate(indices[:-2]):
            if selected[index]:
                return table.iloc[indices[position:position + 3]].copy()
    raise ValueError("No suitable three-word smoke segment")


def smoke():
    verify_manifest()
    torch.set_num_threads(2)
    records = []
    dataset_name = "chineseeeg2_littleprince"
    for condition in CONDITIONS:
        cfg = build_config(dataset_name, condition, 0, "revision")
        table = neural.chinese.dataset_module.载入事件表(cfg["cache"]["event_table"], split="train",
            subjects=cfg["dataset"]["subjects"], trainable_only=True)
        candidates = neural.chinese.dataset_module.载入候选词(cfg["cache"]["event_table"])
        table = smoke_segment(table, candidates, neural.group_column(cfg))
        dataset = dataset_for(cfg, "train", table)
        loader, _ = make_loader(dataset, cfg["training"], shuffle=False, group_column=neural.group_column(cfg))
        cnn_state = None
        for stage in ("word", "revision"):
            set_seed(0)
            stage_cfg = build_config(dataset_name, condition, 0, stage)
            model = neural.build_brain_embedding_model(dataset.channel_count, dataset.channel_positions,
                                                       dataset.subject_count, stage_cfg["model"])
            if cnn_state is not None:
                model.brain_encoder.load_state_dict(cnn_state, strict=True)
            loss_module = neural.build_siglip_loss(stage_cfg["loss"])
            optimizer = neural.build_adamw_for_modules([model, loss_module], stage_cfg["training"])
            frozen = stage == "revision"
            neural.pallier.set_brain_encoder_trainable(model, not frozen)
            started = time.perf_counter()
            loss, updates, _, diagnostics = train_one_epoch(model, loss_module, loader, optimizer,
                torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"),
                {**stage_cfg["training"], "amp": False}, freeze_brain_encoder=frozen,
                return_stats=True, return_diagnostics=True)
            if not np.isfinite(loss) or updates != 1:
                raise AssertionError("Chinese synthetic smoke update failed")
            encoded = encode_loader(model, loader, torch.device("cpu"), amp=False)
            if stage == "revision":
                strict = strict_bank_metrics(encoded, dataset, candidates)
                if diagnostics["prefix_revision_delay_counts"] != {0: 3, 1: 2, 2: 1} or set(strict) != {"0", "1", "2"} or any(
                        v["common_targets"]["query_count"] != 1 or v["common_targets"]["candidate_count"] != 50 for v in strict.values()):
                    raise AssertionError("Chinese rolling supervision or candidate bank changed")
            else:
                cnn_state = cpu_state_dict(model.brain_encoder)
            records.append({"dataset": dataset_name, "condition": condition, "stage": stage, "updates": updates,
                "train_loss": loss, "seconds": time.perf_counter() - started,
                "parameters": sum(p.numel() for p in model.parameters()), "passed": True})
            del model, loss_module, optimizer, encoded
            gc.collect()
        del cnn_state, dataset, loader
        gc.collect()
    result = {"passed": True, "device": "cpu", "records": records, "full_model_capacity": True,
              "formal_performance_estimate": False, "pallier_v1_smoke_reused": read(old.REPORT / "smoke.json")["passed"]}
    save(REPORT / "smoke.json", result)
    return result


def evaluate_test(dataset_name, condition, seed):
    m = verify_manifest()
    job = job_for(dataset_name, condition, seed, "revision")
    validate_completed(job)
    cfg = build_config(dataset_name, condition, seed, "revision")
    out = Path(job["output_dir"])
    path = out / "evaluation/synthetic_test.json"
    checkpoint_hash = file_sha256(out / "best.pt")
    expected_manifest = m["parent_manifest_sha256"] if job["reuse"] else m["manifest_sha256"]
    if path.exists():
        record = read(path)
        if record["manifest_sha256"] != expected_manifest or record["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Existing synthetic test evaluation identity changed")
        return record
    checkpoint = load_checkpoint(out / "best.pt")
    summary = read(out / "training_summary.json")
    if checkpoint["config"] != cfg or checkpoint["epoch"] != summary["best"]["legacy50"]["epoch"] or summary["test_evaluated"] is not False:
        raise ValueError("Test requires the original validation-selected primary model")
    torch.set_num_threads(2)
    dataset = dataset_for(cfg, "test")
    loader, _ = make_loader(dataset, cfg["training"], shuffle=False, group_column=neural.group_column(cfg))
    model = neural.build_brain_embedding_model(dataset.channel_count, dataset.channel_positions, dataset.subject_count, cfg["model"]).cuda()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    candidates = (neural.pallier.frozen_vocabulary() if dataset_name == "pallier2025"
                  else neural.chinese.dataset_module.载入候选词(cfg["cache"]["event_table"]))
    encoded = encode_loader(model, loader, torch.device("cuda"), amp=True)
    strict = strict_bank_metrics(encoded, dataset, candidates)
    expected = m["test_population"][dataset_name]
    if (len(encoded["event_ids"]) != expected["rows"] or digest(sorted(encoded["event_ids"])) != expected["event_ids_sha256"]
            or digest(list(candidates)) != expected["candidates_sha256"] or set(strict) != {"0", "1", "2"}
            or any(v["common_targets"]["query_count"] != expected["common_targets"]
                   or v["all_targets"]["query_count"] != expected["all_targets"]
                   or v["common_targets"]["candidate_count"] != 50 for v in strict.values())):
        raise ValueError("Synthetic test target/candidate population differs from the frozen Real")
    result = {"dataset": dataset_name, "condition": condition, "seed": seed, "split": "test", "test_evaluated": True,
        "manifest_sha256": expected_manifest, "checkpoint_sha256": checkpoint_hash,
        "test_embedding_sha256": file_sha256(ROOT / "data/cache/derived" / dataset_name / "text/unified_heldout_test_v2/embeddings.npz"),
        "event_ids_sha256": digest(sorted(encoded["event_ids"])), "strict_vocab50": strict, "best_epoch": checkpoint["epoch"]}
    save(path, result)
    return result


def export_report():
    m = read(MANIFEST)
    REPORT.mkdir(parents=True, exist_ok=True)
    real = pd.read_csv(ROOT / settings()["execution"]["real_results"])
    rows = [{"dataset": r.dataset, "condition": "real", "seed": int(r.seed), "horizon": int(r.horizon),
             "population": r.population, "macro_top10": float(r.macro_recall_at_10),
             "query_count": int(r.query_count), "candidate_count": int(r.candidate_count)} for r in real.itertuples()]
    for job in m["jobs"]:
        path = Path(job["output_dir"]) / "evaluation/synthetic_test.json"
        if job["stage"] == "revision" and path.exists():
            record = read(path)
            for h, values in record["strict_vocab50"].items():
                for population, value in values.items():
                    rows.append({"dataset": job["dataset"], "condition": job["condition"], "seed": job["seed"],
                        "horizon": int(h), "population": population, "macro_top10": value["macro_recall_at_10"],
                        "query_count": value["query_count"], "candidate_count": value["candidate_count"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORT / "test_by_seed.csv", index=False)
    common = frame[frame.population.eq("common_targets")]
    lines = ["# 词级时间捷径控制：两数据集", "", "固定 50 词候选集，宏平均 Top-10；表格比较 K0/K1/K2 共同目标。",
        "所有选模均使用验证集，复用已完成的真实信号和合成实验。以下结果随串行队列更新；不足三个种子时仅为中间结果。",
        "标准差为样本标准差，随机 Top-10 参考为 20%。", ""]
    summaries, differences, dataset_status = [], {}, {}
    labels = {"real": "真实信号（复用）", "shared": "合成重叠", "independent": "合成独立"}
    for dataset in DATASETS:
        lines += [f"## {dataset}", "", f"共同目标数：{m['test_population'][dataset]['common_targets']:,}；两个条件均保留原 CNN 与 Transformer。", "",
            "| 输入 | 已完成种子 | K0 | K1 | K2 | K2−K0 |", "|---|---:|---:|---:|---:|---:|"]
        part = common[common.dataset.eq(dataset)]
        for condition in ("real", *CONDITIONS):
            subset = part[part.condition.eq(condition)]
            n, scores = subset.seed.nunique(), []
            for h in (0, 1, 2):
                values = subset.loc[subset.horizon.eq(h), "macro_top10"].to_numpy()
                mean, sd = (float(values.mean()) if len(values) else None), (float(values.std(ddof=1)) if len(values) > 1 else None)
                scores.append("—" if mean is None else f"{100 * mean:.2f}%" + (f" ± {100 * sd:.2f}" if sd is not None else "（单种子）"))
                summaries.append({"dataset": dataset, "condition": condition, "horizon": h, "n_seeds": len(values), "mean": mean, "sample_sd": sd})
            pivot = subset.pivot(index="seed", columns="horizon", values="macro_top10")
            delta = pivot[2] - pivot[0] if n else pd.Series(dtype=float)
            gain = "—" if not n else f"{100 * delta.mean():.2f} pp" + (f" ± {100 * delta.std(ddof=1):.2f}" if n > 1 else "")
            lines.append(f"| {labels[condition]} | {n}/3 | {' | '.join(scores)} | {gain} |")
        done = all(part[part.condition.eq(c)].seed.nunique() == 3 for c in CONDITIONS)
        dataset_status[dataset] = "completed" if done else "pending_synthetic_training"
        if done:
            a = part[part.condition.eq("shared")].pivot(index="seed", columns="horizon", values="macro_top10")
            b = part[part.condition.eq("independent")].pivot(index="seed", columns="horizon", values="macro_top10")
            difference = a - b
            differences[dataset] = difference.reset_index().to_dict("records")
            lines += ["", "Shared−Independent 配对差：" + "，".join(
                f"K{h} {100 * difference[h].mean():.2f} ± {100 * difference[h].std(ddof=1):.2f} pp" for h in (0, 1, 2)) + "。"]
        o = read(REPORT / "overlap.json")[dataset]["one_second"]
        lines += ["", f"实际相邻词 {o['pairs']:,} 对：1 秒窗口重叠率 {100 * o['overlap_probability']:.2f}%，平均共享比例 {100 * o['mean_shared_fraction']:.2f}%。", ""]
    lines += ["Shared−Independent 表征本控制中可利用的共享窗口信息；Real−Shared 不定义为纯神经贡献。",
        "Independent 高于 20% 还可能来自位置、片段结构或其他先验，需要在完成控制后分析。",
        "", "本次修复复用九个完成的 Pallier 训练阶段，只重跑缺少完整恢复状态的 Shared 种子 2 滚动阶段，另加入中文 EEG 六组实验。",
        "新训练每轮原子保存完整恢复状态。训练按 epoch 重生成，验证和测试的合成信号固定且不随训练种子变化。",
        "", "实验 B 与语言模型解码均未启动。"]
    result = {"status": "completed" if all(s == "completed" for s in dataset_status.values()) else "pending_synthetic_training",
              "datasets": dataset_status, "split": "test", "population": "common_targets", "summaries": summaries,
              "shared_minus_independent_by_seed": differences, "manifest_sha256": m["manifest_sha256"],
              "context_experiment_B_started": False, "language_model_started": False}
    save(REPORT / "results.json", result)
    (REPORT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def acquire_lock(path):
    if path.exists():
        previous = read(path)
        if recovery.process_matches(previous):
            raise RuntimeError("An active synthetic queue already owns the lock")
        archive = path.parent / "stale_locks"
        archive.mkdir(exist_ok=True)
        path.rename(archive / f"{path.name}.{time.time_ns()}.json")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, stream)


def gpu_ready():
    # The original gate respects the paused beam-search task and the same GPU budget.
    return old.gpu_ready()


def worker_arguments(job, action):
    args = [sys.executable, "-X", "utf8", "-u", "-m", "braindecoding.synthetic_timing_recovery", action,
            "--dataset", job["dataset"], "--condition", job["condition"], "--seed", str(job["seed"])]
    if action == "--worker":
        args += ["--stage", job["stage"]]
    return args


def run_queue():
    m = verify_manifest()
    if not read(REPORT / "smoke.json")["passed"]:
        raise RuntimeError("Chinese smoke must pass before training")
    acquire_lock(RUNTIME / "queue.lock")
    state = {"state": "running", "pid": os.getpid(), "created": psutil.Process().create_time(),
             "started_unix": time.time(), "manifest_sha256": m["manifest_sha256"],
             "completed_stages": m["reused_training_stages"], "reused_stages": m["reused_training_stages"],
             "total_stages": m["total_training_stages"], "new_training_stages": m["new_training_stages"]}
    completed = set()
    try:
        for job in m["jobs"]:
            if job["reuse"]:
                validate_completed(job)
                completed.add(job["output_dir"])
        for job in m["jobs"]:
            if job["reuse"]:
                continue
            verify_manifest()
            out = Path(job["output_dir"])
            if (out / "training_summary.json").exists():
                validate_completed(job)
            else:
                attempts = 0
                while True:
                    ready, reason = gpu_ready()
                    if not ready:
                        state.update(state="waiting_for_gpu", current=job, reason=reason, updated_unix=time.time())
                        save(RUNTIME / "status.json", state)
                        time.sleep(settings()["execution"]["poll_seconds"])
                        continue
                    log_path = RUNTIME / f"{job['dataset']}_{job['condition']}_seed{job['seed']}_{job['stage']}.log"
                    with log_path.open("a", encoding="utf-8") as log:
                        process = subprocess.Popen(worker_arguments(job, "--worker"), cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        state.update(state="training", current=job, worker_pid=process.pid,
                            worker_created=psutil.Process(process.pid).create_time(), attempts=attempts + 1, updated_unix=time.time())
                        save(RUNTIME / "status.json", state)
                        code = process.wait()
                    if code == 0:
                        validate_completed(job)
                        break
                    worker_manifest = read(out / "run_manifest.json") if (out / "run_manifest.json").exists() else {}
                    # Explicit scientific/training exceptions need inspection. Only a
                    # process killed without a persisted exception is auto-retried.
                    retry = worker_manifest.get("status") != "failed" and attempts < settings()["execution"]["automatic_unclean_exit_retries"]
                    save(RUNTIME / f"exit_{time.time_ns()}.json", {"job": job, "returncode": code,
                        "worker_manifest_status": worker_manifest.get("status"), "retry": retry,
                        "resume_available": (out / "resume.pt").exists()})
                    if not retry:
                        raise RuntimeError(f"Worker exited {code}; details: {log_path}")
                    attempts += 1
                    state.update(state="recovering_unclean_exit", updated_unix=time.time())
                    save(RUNTIME / "status.json", state)
                    time.sleep(5)
            completed.add(job["output_dir"])
            state["completed_stages"] = len(completed)
            if job["stage"] == "revision":
                state.update(state="evaluating_test", current=job, updated_unix=time.time())
                save(RUNTIME / "status.json", state)
                log_path = RUNTIME / f"{job['dataset']}_{job['condition']}_seed{job['seed']}_test.log"
                with log_path.open("a", encoding="utf-8") as log:
                    subprocess.run(worker_arguments(job, "--test"), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                export_report()
            save(RUNTIME / "status.json", {**state, "updated_unix": time.time()})
        export_report()
        state.update(state="completed", updated_unix=time.time())
        save(RUNTIME / "status.json", state)
        return state
    except BaseException as error:
        state.update(state="failed", error=f"{type(error).__name__}: {error}", updated_unix=time.time())
        save(RUNTIME / "status.json", state)
        raise
    finally:
        (RUNTIME / "queue.lock").unlink(missing_ok=True)


def launch():
    verify_manifest()
    if not read(REPORT / "smoke.json")["passed"]:
        raise RuntimeError("Smoke must pass before launch")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    acquire_lock(RUNTIME / "launch.lock")
    try:
        path = RUNTIME / "launch.json"
        if path.exists() and recovery.process_matches(read(path)):
            raise RuntimeError("Synthetic queue already launched")
        if (RUNTIME / "queue.lock").exists() and recovery.process_matches(read(RUNTIME / "queue.lock")):
            raise RuntimeError("Synthetic queue is already running")
        startup = RUNTIME / "startup.json"
        startup.unlink(missing_ok=True)
        if os.name == "nt":
            task_name = "Codex_BrainDecoding_SyntheticTimingV2"
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            executable = Path(sys.executable).with_name("pythonw.exe")
            if not executable.exists():
                raise FileNotFoundError(executable)
            entry = Path(__file__).resolve()
            arguments = subprocess.list2cmdline(["-X", "utf8", "-u", str(entry), "--queue-entry"])
            command = f"""
$ErrorActionPreference = 'Stop'
$svc = New-Object -ComObject Schedule.Service
$svc.Connect()
$folder = $svc.GetFolder('\\')
$taskName = {quote(task_name)}
$prior = $null
try {{ $prior = $folder.GetTask($taskName) }} catch [System.IO.FileNotFoundException] {{}}
if ($prior -and $prior.State -eq 4) {{ throw 'Independent synthetic task is already running' }}
if ($prior -and $prior.Definition.RegistrationInfo.Description -ne 'BrainDecoding synthetic_timing_v2 immediate queue') {{ throw 'Task name belongs to another task' }}
$task = $svc.NewTask(0)
$task.RegistrationInfo.Description = 'BrainDecoding synthetic_timing_v2 immediate queue'
$task.Settings.Enabled = $true
$task.Settings.Hidden = $true
$task.Settings.AllowDemandStart = $true
$task.Settings.ExecutionTimeLimit = 'PT0S'
$task.Settings.DisallowStartIfOnBatteries = $false
$task.Settings.StopIfGoingOnBatteries = $false
$task.Settings.MultipleInstances = 2
$task.Principal.UserId = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$task.Principal.LogonType = 3
$task.Principal.RunLevel = 0
$action = $task.Actions.Create(0)
$action.Path = {quote(executable)}
$action.Arguments = {quote(arguments)}
$action.WorkingDirectory = {quote(ROOT)}
$registered = $folder.RegisterTaskDefinition($taskName, $task, 6, $task.Principal.UserId, $null, 3, $null)
$null = $registered.Run($null)
"""
            subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", command], check=True)
        else:
            task_name = None
            with (RUNTIME / "queue.log").open("a", encoding="utf-8") as log:
                subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--queue-entry"],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True, close_fds=True)
        deadline = time.monotonic() + 30
        while not startup.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Independent task did not acknowledge startup; inspect the task before retrying")
            time.sleep(.25)
        result = {**read(startup), "state": "launched", "windows_task": task_name,
                  "requested_unix": time.time()}
        save(path, result)
        return result
    finally:
        (RUNTIME / "launch.lock").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("prepare", "smoke", "worker", "test", "queue", "queue-entry", "launch", "report", "status"):
        actions.add_argument("--" + action, action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--stage", choices=("word", "revision"))
    args = parser.parse_args()
    configure_environment()
    if args.prepare:
        m = prepare()
        result = {"prepared": True, "reused_stages": m["reused_training_stages"], "new_stages": m["new_training_stages"], "manifest_sha256": m["manifest_sha256"]}
    elif args.smoke:
        result = smoke()
    elif args.worker or args.test:
        if not args.dataset or not args.condition or (args.worker and not args.stage):
            parser.error("Worker/test requires dataset, condition and (worker only) stage")
        result = (run_stage(args.dataset, args.condition, args.seed, args.stage) if args.worker
                  else evaluate_test(args.dataset, args.condition, args.seed))
        result = {"completed": True, "dataset": args.dataset, "condition": args.condition, "seed": args.seed}
    elif args.queue:
        result = run_queue()
    elif getattr(args, "queue_entry", False):
        os.chdir(ROOT)
        RUNTIME.mkdir(parents=True, exist_ok=True)
        process = psutil.Process()
        payload = {"pid": process.pid, "created": process.create_time(), "started_unix": time.time(),
                   "entry": str(Path(__file__).resolve()),
                   "parent_chain": [{"pid": p.pid, "name": p.name()} for p in process.parents()]}
        save(RUNTIME / "startup.json", payload)
        with (RUNTIME / "queue.log").open("a", encoding="utf-8", buffering=1) as log:
            sys.stdout = sys.stderr = log
            result = run_queue()
    elif args.launch:
        result = launch()
    elif args.report:
        verify_manifest()
        result = export_report()
    else:
        result = read(RUNTIME / "status.json") if (RUNTIME / "status.json").exists() else {"state": "not_launched"}
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
