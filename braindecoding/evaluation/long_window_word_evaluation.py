"""Train and test independent word CNNs with 64/75-sample continuous cuts."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from braindecoding import unified_rolling_v2_training as neural
from braindecoding import unified_heldout_test_v2 as heldout
from braindecoding.config import PROJECT_ROOT as ROOT
from braindecoding.data.single_word_long_window import SingleWordLongWindowDataset, window_spec
from braindecoding.training.runtime import cpu_state_dict, load_checkpoint, save_checkpoint, set_seed
from braindecoding.training.word import encode_loader, train_one_epoch
from braindecoding.unified_sentence_v2 import digest, read, save, sha256


ID = "single_word_long_window_test_v2"
CONFIG = ROOT / f"configs/unified_rolling/{ID}.yaml"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
PARENT = ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"
RUNTIME = ROOT / f".cache/runs/{ID}"
REPORT = ROOT / f"data/results/studies/{ID}"
OUT = ROOT / f"outputs/heldout_test/{ID}"
WINDOWS = {"long_1p27": 1.27, "long_1p5": 1.5}


def native(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(native(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.025 * (attempt + 1), .2))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure_environment():
    parent = read(PARENT)
    cfg = next(j["config"] for j in parent["checkpoints"] if j["dataset"] == "pallier2025")
    os.environ.setdefault("BRAINDATA_ROOT", str(Path(cfg["dataset"]["root"]).parent))
    from braindecoding.config import DEFAULT_MODEL_ROOT
    os.environ.setdefault("BRAINDECODING_MODEL_ROOT", DEFAULT_MODEL_ROOT)


def settings():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p = cfg["protocol"]
    require(p["id"] == ID and p["datasets"] == list(neural.DATASETS) and p["seeds"] == [0, 1, 2]
            and p["train_only_word_models"] and not p["cross_word_transformer_allowed"]
            and not p["language_model_allowed"] and not p["test_selection_allowed"], "Long-window scope changed")
    for condition, seconds in WINDOWS.items():
        spec = window_spec(seconds)
        require(all(cfg["windows"][condition][k] == spec[k] for k in ("requested_seconds", "samples", "actual_seconds")),
                "Long-window sampling contract changed")
    return cfg


def build_config(dataset, seed, condition):
    configure_environment()
    require(dataset in neural.DATASETS and seed in (0, 1, 2) and condition in WINDOWS, "Unknown long-window job")
    cfg = neural.build_config(dataset, seed, "word")
    spec = window_spec(WINDOWS[condition])
    cfg["experiment"].update(output_group=ID, id=f"single_word_{condition}")
    cfg["model"]["use_transformer"] = False
    cfg["model"]["context_mode"] = "singleton"
    require(not cfg["model"].get("local_window_revision"), "Word CNN contains revision settings")
    cfg["dataset"]["window_seconds"] = spec["actual_seconds"]
    cfg["single_word_window"] = {**spec, "canonical_source_window_seconds": 1.,
                                  "cross_word_transformer": False, "language_model": False}
    cfg["training"]["output_dir"] = str(ROOT / "outputs/word_decoding" / dataset / ID /
        cfg["experiment"]["subject_scope"] / cfg["experiment"]["id"] / f"seed-{seed:03d}")
    cfg["unified_protocol"]["supplement"] = ID
    cfg["provenance"].update(long_window_study=ID, neural_input="one_continuous_word_onset_window")
    return cfg


def canonical_base(cfg, split):
    require(split in ("train", "val", "test"), "Unsupported split")
    task = neural.pallier if cfg["experiment"]["dataset"] == "pallier2025" else neural.chinese
    loader = task.dataset_module.load_event_table
    builder = task.build_dataset
    original = copy.deepcopy(cfg)
    original["dataset"]["window_seconds"] = 1.
    if split == "test":
        original["cache"]["text_embeddings"] = str(heldout.target_cache(original))
    table = loader(original["cache"]["event_table"], split=split,
                   subjects=original["dataset"]["subjects"], trainable_only=True)
    base = builder(original, table)
    require(base.table.split.eq(split).all() and not base.table.event_id.duplicated().any(), "Source split or IDs changed")
    return base


def dataset_and_loader(cfg, split):
    base = canonical_base(cfg, split)
    dataset = SingleWordLongWindowDataset(base, cfg["single_word_window"]["requested_seconds"])
    loader, sampler = neural.make_loader(dataset, cfg["training"], shuffle=split == "train",
                                         group_column=neural.group_column(cfg))
    return dataset, loader, sampler


def timing_rows(dataset, info):
    lookup = {event: i for i, event in enumerate(dataset.event_ids)}
    expected = [e for g in info["groups"] for e in g["event_ids"]]
    require(len(lookup) == len(expected) and set(lookup) == set(expected), "Frozen test population changed")
    result = []
    for group in info["groups"]:
        starts = np.asarray([dataset.starts[lookup[event]] for event in group["event_ids"]])
        require((np.diff(starts) >= 0).all(), "Onsets within a frozen group are not chronological")
        for pos, event in enumerate(group["event_ids"]):
            start = int(dataset.starts[lookup[event]])
            ends = {str(h): int(dataset.starts[lookup[group["event_ids"][pos + h]]]) + 50
                    if pos + h < len(group["event_ids"]) else None for h in (0, 1, 2)}
            result.append(dict(event_id=event, start=start, ends=ends, common=pos + 2 < len(group["event_ids"]),
                               subject=group["subject"], block=group["block"]))
    return result


def preflight():
    configure_environment()
    parent = read(PARENT)
    result, timing = {}, {}
    for name in neural.DATASETS:
        cfg = build_config(name, 0, "long_1p5")
        rows = []
        for split in ("train", "val", "test"):
            base = canonical_base(cfg, split)
            datasets = {seconds: SingleWordLongWindowDataset(base, seconds) for seconds in (1., 1.27, 1.5)}
            expected = cfg["dataset"]["expected_trainable_counts"][split]
            require(len(base) == expected, "Original event population changed")
            # Include every recording as well as the table ends.
            samples = {0, len(base) // 2, len(base) - 1}
            samples.update(base.table.groupby("recording_id", sort=False).head(1).index.tolist())
            key = "meg" if name == "pallier2025" else "eeg"
            for i in sorted(samples):
                original = base.read_signal(i)
                for dataset in datasets.values():
                    require(np.array_equal(dataset[i][key].numpy()[:, :50], original), "Original 1 s prefix was modified")
            rows.append(dict(split=split, rows=len(base), bounds={str(s): d.bounds_audit for s, d in datasets.items()},
                             prefix_fidelity=dict(checked_events=len(samples), exact=True)))
            print(json.dumps(dict(dataset=name, split=split, rows=len(base), long_windows_valid=True)), flush=True)
            if split == "test":
                timing[name] = timing_rows(datasets[1.5], parent["datasets"][name])
                eligible = [row for row in timing[name] if row["common"]]
                for horizon, length in ((1, 64), (2, 75)):
                    elapsed = [(r["ends"][str(horizon)] - r["start"]) / 50 for r in eligible]
                    print(json.dumps(dict(dataset=name, horizon=horizon, common_n=len(eligible),
                        coverage_mean_s=float(np.mean(elapsed)), coverage_median_s=float(np.median(elapsed)),
                        within_budget_n=sum(r["ends"][str(horizon)] <= r["start"] + length for r in eligible))), flush=True)
        result[name] = rows
    RUNTIME.mkdir(parents=True, exist_ok=True)
    save(RUNTIME / "preflight.json", result)
    save(RUNTIME / "timing_by_event.json", timing)
    return result


def verify_manifest(m):
    require(digest({k: v for k, v in m.items() if k != "manifest_sha256"}) == m["manifest_sha256"], "Manifest identity changed")


def verify_sources(m, *, full=False):
    for path, expected in m["source_sha256"].items():
        if str(path).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        source = Path(path)
        signature = m["source_stat"][path]
        require(source.is_file() and source.stat().st_size == signature["bytes"]
                and source.stat().st_mtime_ns == signature["mtime_ns"], f"Frozen source changed: {path}")
        if full or path in m["always_hash_sources"]:
            require(sha256(source) == expected, f"Frozen source hash changed: {path}")


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        verify_manifest(m)
        verify_sources(m, full=True)
        return m
    spec = settings()
    configure_environment()
    parent = read(PARENT)
    verify_manifest(parent)
    require((RUNTIME / "preflight.json").exists() and (RUNTIME / "gpu_smoke.json").exists(), "Preflight and GPU smoke required")
    pre = read(RUNTIME / "preflight.json")
    smoke = read(RUNTIME / "gpu_smoke.json")
    require(smoke["status"] == "passed" and all(row["optimizer_updates"] > 0 for row in smoke["rows"]),
            "Smoke must include successful optimizer updates in both datasets")
    source_paths = {CONFIG, PROTOCOL, PARENT, Path(__file__),
        ROOT / "braindecoding/data/single_word_long_window.py",
        ROOT / "braindecoding/evaluation/single_word_long_window_test_v2_results.py",
        ROOT / "braindecoding/evaluation/full_vocabulary_context_test_v2.py",
        ROOT / "braindecoding/evaluation/word_paired_statistics_test_v2.py",
        ROOT / "tests/test_single_word_long_window_test_v2.py",
        RUNTIME / "preflight.json", RUNTIME / "timing_by_event.json", RUNTIME / "gpu_smoke.json"}
    science = {str(p.resolve()) for p in source_paths if p.suffix in (".py", ".yaml", ".md")}
    queue_path = ROOT / "data/results/protocols/unified_rolling_v2_queue_results.json"
    queue = read(queue_path)
    require(queue["status"] == "completed", "Original neural training incomplete")
    source_paths.add(queue_path)
    jobs, references, rolling = [], [], []
    for dataset in neural.DATASETS:
        target = Path(parent["datasets"][dataset]["evaluation_target_cache"])
        require(sha256(target) == parent["source_sha256"][str(target.resolve())], "Frozen test target vectors changed")
        source_paths.update((target, target.with_suffix(".json")))
        base_cfg = build_config(dataset, 0, "long_1p5")
        inherited = neural.source_hashes(base_cfg, neural.config()["adapters"][dataset])
        source_paths.update(Path(p) for p in inherited)
        science.update(p for p in inherited if Path(p).suffix in (".py", ".yaml"))
        for row in pre[dataset]:
            require(row["prefix_fidelity"]["exact"], "Original signal prefix differs")
            for audit in row["bounds"].values():
                require(audit["out_of_bounds"] == audit["dropped_events"] == audit["padded_events"] == 0, "Long-window population differs")
                for path in audit["signal_shapes"]:
                    source_paths.add(Path(path))
                    source_paths.add(Path(path).with_suffix(".json"))
        for seed in (0, 1, 2):
            old = next(j for j in queue["completed"] if (j["dataset"], j["seed"], j["stage"]) == (dataset, seed, "word"))
            checkpoint = Path(old["checkpoint"])
            require(sha256(checkpoint) == old["checkpoint_sha256"], "Original word checkpoint changed")
            cp = load_checkpoint(checkpoint)
            require(cp["stage"] == "word" and cp["seed"] == seed and not cp["model_config"]["use_transformer"]
                    and cp["model_config"]["conv"] == base_cfg["model"]["conv"], "Original word CNN is incompatible")
            require(cp["text_embedding_config"] == base_cfg["text_embedding"]
                    and cp["config"]["dataset"]["window_seconds"] == 1., "Original word target or window differs")
            references.append(dict(dataset=dataset, seed=seed, condition="word_1s", checkpoint=str(checkpoint),
                                   checkpoint_sha256=old["checkpoint_sha256"], config=cp["config"], epoch=cp["epoch"]))
            source_paths.add(checkpoint)
            for name in ("run_manifest.json", "training_summary.json", "resolved_config.yaml"):
                source_paths.add(checkpoint.parent / name)
            root = heldout.seed_root(dataset, seed)
            prediction, sidecar = root / "neural_predictions.npz", root / "word_metrics.json"
            side = read(sidecar)
            source = next(j for j in parent["checkpoints"] if (j["dataset"], j["seed"]) == (dataset, seed))
            require(side["manifest_sha256"] == parent["manifest_sha256"] and side["prediction_sha256"] == sha256(prediction)
                    and side["checkpoint_sha256"] == source["checkpoint_sha256"], "Rolling prediction provenance changed")
            rolling.append(dict(dataset=dataset, seed=seed, prediction=str(prediction), sidecar=str(sidecar)))
            source_paths.update((prediction, sidecar, Path(source["checkpoint"])))
            for condition in WINDOWS:
                cfg = build_config(dataset, seed, condition)
                jobs.append(dict(dataset=dataset, seed=seed, condition=condition, config=cfg,
                                 config_sha256=digest(cfg), output_dir=cfg["training"]["output_dir"]))
    source_hashes, signatures = {}, {}
    for path in sorted(source_paths):
        source = str(path.resolve())
        source_hashes[source] = sha256(path)
        signatures[source] = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    m = dict(protocol=ID, frozen_unix=time.time(), parent_manifest_sha256=parent["manifest_sha256"],
             settings=spec, jobs=jobs, references=references, rolling=rolling,
             source_sha256=source_hashes, source_stat=signatures, always_hash_sources=sorted(science),
             preflight=pre, training_jobs=12, initialization="from_scratch_same_seed",
             test_selection_allowed=False, language_model_allowed=False)
    m["dependency_versions"] = dict(torch=str(torch.__version__), numpy=np.__version__, yaml=yaml.__version__)
    m["manifest_sha256"] = digest(m)
    verify_sources(m)
    write_json(MANIFEST, m)
    print(json.dumps(dict(stage="frozen", manifest_sha256=m["manifest_sha256"], training_jobs=12,
                          source_files=len(source_hashes))), flush=True)
    return m


def gpu_ready(m, callback=None):
    require(torch.cuda.is_available(), "CUDA required; no CPU training fallback")
    policy = m["settings"]["execution"]
    while True:
        free, total = torch.cuda.mem_get_info()
        if free >= policy["minimum_free_gpu_memory_gib"] * 1024**3:
            break
        if callback:
            callback(status="waiting_gpu_memory", free_gib=free / 1024**3)
        time.sleep(10)
    torch.cuda.set_per_process_memory_fraction(policy["gpu_memory_limit_gib"] * 1024**3 / total)


def rng_state():
    array = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(name=array[0], keys=array[1].tolist(),
        position=int(array[2]), has_gauss=int(array[3]), cached_gaussian=float(array[4])),
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n["name"], np.asarray(n["keys"], dtype=np.uint32), n["position"], n["has_gauss"], n["cached_gaussian"]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def completed_training(m, job):
    root = Path(job["output_dir"])
    summary = read(root / "training_summary.json")
    require(summary["status"] == "completed" and summary["manifest_sha256"] == m["manifest_sha256"]
            and summary["config_sha256"] == job["config_sha256"], "Completed long-window training differs")
    for name, value in summary["checkpoints"].items():
        require(sha256(root / name) == value["sha256"], "Completed long-window checkpoint changed")
    return summary


def train_job(m, job):
    verify_sources(m)
    cfg, out = job["config"], Path(job["output_dir"])
    summary_path, manifest_path, resume_path = out / "training_summary.json", out / "run_manifest.json", out / "epoch_latest.pt"
    if summary_path.exists():
        return completed_training(m, job)
    if out.exists():
        require(manifest_path.exists(), "Unknown incomplete directory must not be overwritten")
        previous = read(manifest_path)
        require(previous["manifest_sha256"] == m["manifest_sha256"] and previous["config_sha256"] == job["config_sha256"],
                "Resume source or configuration differs")
        require(resume_path.exists(), "Incomplete first epoch has no exact recovery state")
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(dataset=job["dataset"], seed=job["seed"], condition=job["condition"], stage="word",
                    manifest_sha256=m["manifest_sha256"], config_sha256=job["config_sha256"])
    class Status:
        def __init__(self):
            self.fields = {**identity, "pid": os.getpid(), "started_unix": time.time()}
        def __call__(self, **fields):
            self.fields.update(fields, updated_unix=time.time())
            write_json(out / "status.json", self.fields)
            print(json.dumps(fields), flush=True)
    status = Status()
    run_manifest = {**identity, "status": "preparing", "config": cfg, "test_evaluated": False}
    write_json(manifest_path, run_manifest)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        gpu_ready(m, status)
        torch.set_num_threads(m["settings"]["execution"]["cpu_threads"])
        set_seed(job["seed"])
        train, train_loader, sampler = dataset_and_loader(cfg, "train")
        val, val_loader, _ = dataset_and_loader(cfg, "val")
        adapter = neural.config()["adapters"][job["dataset"]]
        vocabulary, vocab50, _, observed = neural.populations(cfg, train, val, adapter)
        model = neural.build_brain_embedding_model(train.channel_count, train.channel_positions, train.subject_count, cfg["model"]).cuda()
        require(model.context_transformer is None and not model.prefix_revision_enabled, "CNN has cross-word context")
        loss_module = neural.build_siglip_loss(cfg["loss"]).cuda()
        tc = cfg["training"]
        optimizer = neural.build_adamw_for_modules([model, loss_module], tc)
        scheduler = neural.build_cosine_annealing_scheduler(optimizer, tc["epochs"])
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        history, best, best_rows, best_paths = [], {"legacy50": -float("inf"), "full_vocab": -float("inf")}, {}, {}
        stale, updates, start_epoch = 0, 0, 0
        if resume_path.exists():
            state = load_checkpoint(resume_path)
            require(state["manifest_sha256"] == m["manifest_sha256"] and state["config_sha256"] == job["config_sha256"],
                    "Epoch recovery state differs")
            model.load_state_dict(state["model_state"], strict=True)
            loss_module.load_state_dict(state["loss_state"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            history, best, best_rows, best_paths = state["history"], state["best"], state["best_rows"], state["best_paths"]
            stale, updates, start_epoch = state["stale"], state["updates"], state["completed_epochs"]
            restore_rng(state["rng"])
            del state
            status(status="resumed_exact_epoch", completed_epochs=start_epoch)
        run_manifest.update(status="training", train_rows=len(train), validation_rows=len(val), population=observed,
            vocabulary_size=len(vocabulary), parameters=sum(p.numel() for p in model.parameters()),
            input_window=cfg["single_word_window"], cross_word_transformer=False, initialization="from_scratch",
            train_bounds=train.bounds_audit, validation_bounds=val.bounds_audit)
        write_json(manifest_path, run_manifest)
        embedding_hash = sha256(cfg["cache"]["text_embeddings"])
        for epoch in range(start_epoch, tc["epochs"]):
            if stale >= tc["patience"]:
                break
            verify_sources(m)
            started = time.perf_counter()
            sampler.set_epoch(epoch)
            status(status="training", epoch=epoch + 1, completed_epochs=epoch, completed_batches=0)
            loss, epoch_updates, batches, diagnostics = train_one_epoch(model, loss_module,
                neural.ProgressLoader(train_loader, status), optimizer, scaler, torch.device("cuda"), tc,
                return_stats=True, return_diagnostics=True, epoch_index=epoch)
            require(np.isfinite(loss), "Non-finite word training loss")
            updates += epoch_updates
            status(status="validating", completed_batches=batches)
            metrics = native(neural.validate(model, val_loader, val, vocabulary, vocab50, cfg, adapter, False))
            scheduler.step()
            row = native(dict(epoch=epoch + 1, train_loss=loss, seconds=time.perf_counter() - started,
                optimizer_updates=updates, epoch_optimizer_updates=epoch_updates, batches=batches,
                amp_skipped_updates=batches - epoch_updates, training_diagnostics=diagnostics, validation=metrics))
            history.append(row)
            scores = dict(legacy50=metrics["legacy"][cfg["evaluation"]["selection_metric"]],
                          full_vocab=metrics["full_vocabulary"]["selection_score"])
            stale += 1
            for selection, score in scores.items():
                require(np.isfinite(score), "Non-finite validation selection score")
                if score > best[selection]:
                    best[selection], best_rows[selection] = score, row
                    if selection == "legacy50":
                        stale = 0
                    path = out / "selected" / f"{selection}-epoch-{epoch + 1:03d}.pt"
                    payload = {**identity, "dataset": job["dataset"], "stage": "word", "seed": job["seed"],
                        "model_state": cpu_state_dict(model), "loss_state": cpu_state_dict(loss_module),
                        "model_config": cfg["model"], "loss_config": cfg["loss"], "text_embedding_config": cfg["text_embedding"],
                        "embedding_sha256": embedding_hash, "config": cfg, "epoch": epoch + 1,
                        "channel_names": train.channel_names, "channel_positions": np.asarray(train.channel_positions).tolist(),
                        "subject_ids": list(getattr(train, "subject_ids", cfg["dataset"]["subjects"])),
                        "metrics": metrics, "selection": selection, "test_evaluated": False}
                    save_checkpoint(path, payload)
                    best_paths[selection] = str(path)
            save_checkpoint(resume_path, {**identity, "model_state": cpu_state_dict(model), "loss_state": cpu_state_dict(loss_module),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "rng": rng_state(), "history": history, "best": best, "best_rows": best_rows, "best_paths": best_paths,
                "stale": stale, "updates": updates, "completed_epochs": epoch + 1})
            write_json(out / "progress.json", dict(history=history, best=best_rows))
            status(status="epoch_complete", completed_epochs=epoch + 1, best_epoch=best_rows["legacy50"]["epoch"],
                   best_selection_score=best["legacy50"], epoch_seconds=row["seconds"],
                   peak_memory_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
        checkpoints = {}
        for selection, name in (("legacy50", "best.pt"), ("full_vocab", "best_full_vocab.pt")):
            destination = out / name
            shutil.copyfile(best_paths[selection], destination)
            checkpoints[name] = dict(sha256=sha256(destination), epoch=best_rows[selection]["epoch"], selection=selection)
        summary = {**identity, "status": "completed", "best": best_rows, "history": history,
            "completed_epochs": len(history), "stop_reason": "patience_exhausted" if stale >= tc["patience"] else "maximum_epochs",
            "test_evaluated": False, "initialization": "from_scratch", "checkpoints": checkpoints,
            "gpu_memory_limit_gib": m["settings"]["execution"]["gpu_memory_limit_gib"]}
        write_json(summary_path, summary)
        run_manifest.update(status="completed", completed_epochs=len(history), checkpoints=checkpoints)
        write_json(manifest_path, run_manifest)
        status(status="completed", completed_epochs=len(history))
        del model, loss_module, optimizer, scheduler, scaler, train, val, train_loader, val_loader
        return summary
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}")
        run_manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, run_manifest)
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def prediction_root(job):
    return OUT / job["dataset"] / f"seed-{job['seed']:03d}" / job["condition"]


def verify_prediction(m, job, checkpoint_hash):
    root = prediction_root(job)
    side = read(root / "metrics.json")
    require(side["status"] == "completed" and side["manifest_sha256"] == m["manifest_sha256"]
            and side["parent_manifest_sha256"] == m["parent_manifest_sha256"]
            and side["dataset"] == job["dataset"] and side["seed"] == job["seed"]
            and side["condition"] == job["condition"] and side["checkpoint_sha256"] == checkpoint_hash
            and side["prediction_sha256"] == sha256(root / "predictions.npz"), "Saved single-word prediction differs")
    return side


def encode_job(m, job):
    verify_sources(m)
    reference = job["condition"] == "word_1s"
    if reference:
        checkpoint_path, expected = Path(job["checkpoint"]), job["checkpoint_sha256"]
    else:
        summary = completed_training(m, job)
        checkpoint_path = Path(job["output_dir"]) / "best.pt"
        expected = summary["checkpoints"]["best.pt"]["sha256"]
    require(sha256(checkpoint_path) == expected, "Validation-selected word checkpoint changed")
    root = prediction_root(job)
    destination, sidecar = root / "predictions.npz", root / "metrics.json"
    if destination.exists() and sidecar.exists():
        return verify_prediction(m, job, expected)
    require(not destination.exists() and not sidecar.exists(), "Incomplete prediction export must not be overwritten")
    gpu_ready(m)
    torch.set_num_threads(m["settings"]["execution"]["cpu_threads"])
    set_seed(job["seed"])
    cp = load_checkpoint(checkpoint_path)
    cfg = copy.deepcopy(cp["config"])
    if reference:
        cfg["single_word_window"] = {**window_spec(1.), "canonical_source_window_seconds": 1.}
    require(cp["stage"] == "word" and cp["seed"] == job["seed"] and not cp["model_config"]["use_transformer"], "Expected independent word CNN")
    ds, loader, _ = dataset_and_loader(cfg, "test")
    model = neural.build_brain_embedding_model(ds.channel_count, ds.channel_positions, ds.subject_count, cp["model_config"]).cuda().eval()
    model.load_state_dict(cp["model_state"], strict=True)
    started = time.perf_counter()
    encoded = encode_loader(model, loader, torch.device("cuda"), amp=True)
    parent = read(PARENT)
    timing = timing_rows(ds, parent["datasets"][job["dataset"]])
    meta = {r["event_id"]: r for r in timing}
    events = encoded["event_ids"]
    require(len(events) == len(set(events)) == len(meta) and set(events) == set(meta), "Test encoding population differs")
    queries = encoded["predictions"].numpy()
    require(queries.shape == (len(events), 1024) and np.isfinite(queries).all()
            and np.allclose(np.linalg.norm(queries, axis=1), 1., atol=.005), "Invalid single-word test vectors")
    require(not encoded.get("revision_predictions"), "Word encoding used revision context")
    root.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, event_ids=np.asarray(events), words=np.asarray(encoded["words"]),
            predictions=queries, common_mask=np.asarray([meta[e]["common"] for e in events]),
            start_samples=np.asarray([meta[e]["start"] for e in events], dtype=np.int64))
    os.replace(temporary, destination)
    side = dict(status="completed", dataset=job["dataset"], seed=job["seed"], condition=job["condition"],
        split="test", test_evaluated=True, manifest_sha256=m["manifest_sha256"],
        parent_manifest_sha256=m["parent_manifest_sha256"], checkpoint=str(checkpoint_path),
        checkpoint_sha256=expected, checkpoint_epoch=cp["epoch"], selection="legacy50",
        prediction_sha256=sha256(destination), input_window=cfg["single_word_window"],
        query_count=len(events), bounds_audit=ds.bounds_audit, cross_word_transformer=False,
        neural_encoding_seconds=time.perf_counter() - started)
    write_json(sidecar, side)
    del model, ds, loader, cp, encoded
    gc.collect()
    torch.cuda.empty_cache()
    return side


def queue_status(m, **fields):
    path = RUNTIME / "status.json"
    state = read(path) if path.exists() else {}
    state.update(fields, pid=os.getpid(), updated_unix=time.time(), manifest_sha256=m["manifest_sha256"], total_training_jobs=12)
    write_json(path, state)
    lines = ["# 单词长窗对照", "", f"状态：{state.get('state')}。已完成 {state.get('completed_training_jobs', 0)} / 12 项训练。", "",
        "两个数据集、三个种子；64 样本（请求 1.27 秒、实际 1.28 秒）和 75 样本（1.5 秒）的独立 CNN。",
        "完整训练词库检索；沿用原验证选模，测试集不选超参数。", "",
        f"当前：{state.get('current', '')}", f"阶段：{state.get('stage', '')}"]
    current = state.get("current", {})
    if isinstance(current, dict) and current.get("output_dir"):
        worker = Path(current["output_dir"]) / "status.json"
        if worker.exists():
            info = read(worker)
            lines += [f"Epoch：{info.get('epoch', '')}；批次：{info.get('completed_batches', '')}；最近状态：{info.get('status', '')}。"]
    if state.get("error"):
        lines += ["", f"错误：{state['error']}"]
    if state.get("state") == "completed":
        lines += ["", "[完整结果](results.md)"]
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        (REPORT / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except PermissionError:
        pass  # Durable status remains required; a locked display is optional.


def run():
    import msvcrt
    m = freeze()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "worker.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        tasks = m["jobs"] + m["references"]
        completed = 0
        queue_status(m, state="running", stage="starting", started_unix=time.time(), completed_training_jobs=0, error=None)
        try:
            for number, job in enumerate(tasks):
                reference = job["condition"] == "word_1s"
                log_root = RUNTIME / "logs"
                log_root.mkdir(exist_ok=True)
                stem = f"{job['dataset']}_{job['condition']}_seed-{job['seed']:03d}"
                queue_status(m, state="running", stage="reference_encoding" if reference else "word_training_and_encoding",
                             current={k: job[k] for k in ("dataset", "seed", "condition", "output_dir") if k in job},
                             completed_training_jobs=completed, completed_tasks=number)
                command = [sys.executable, "-m", "braindecoding.single_word_long_window_test_v2", "worker",
                           "--dataset", job["dataset"], "--seed", str(job["seed"]), "--condition", job["condition"]]
                env = os.environ.copy()
                env.update(PYTHONUTF8="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
                with (log_root / f"{stem}.stdout.log").open("a", encoding="utf-8") as stdout, \
                     (log_root / f"{stem}.stderr.log").open("a", encoding="utf-8") as stderr:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                               creationflags=subprocess.CREATE_NO_WINDOW)
                    queue_status(m, worker_pid=process.pid)
                    last_update = 0.
                    while process.poll() is None:
                        if time.monotonic() - last_update >= 30:
                            queue_status(m, worker_pid=process.pid)
                            last_update = time.monotonic()
                        time.sleep(1)
                require(process.returncode == 0, f"Worker failed ({process.returncode}); inspect {stem}.stderr.log")
                if reference:
                    expected = job["checkpoint_sha256"]
                else:
                    summary = completed_training(m, job)
                    expected = summary["checkpoints"]["best.pt"]["sha256"]
                    completed += 1
                verify_prediction(m, job, expected)
                queue_status(m, completed_training_jobs=completed, completed_tasks=number + 1, worker_pid=None)
            verify_sources(m, full=True)
            queue_status(m, stage="paired_statistics", current=None)
            export(m)
            queue_status(m, state="completed", stage="completed", completed_unix=time.time(),
                         completed_training_jobs=12, completed_tasks=18, GPU_workers_exited=True)
        except BaseException as error:
            queue_status(m, state="failed", stage="failed", error=f"{type(error).__name__}: {error}")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "prepare", "run", "worker", "export"))
    parser.add_argument("--dataset", choices=neural.DATASETS)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--condition", choices=(*WINDOWS, "word_1s"))
    args = parser.parse_args(argv)
    if args.command == "preflight":
        preflight()
    elif args.command == "prepare":
        freeze()
    elif args.command == "run":
        run()
    elif args.command == "worker":
        m = read(MANIFEST)
        verify_manifest(m)
        configure_environment()
        items = m["references"] if args.condition == "word_1s" else m["jobs"]
        job = next(j for j in items if (j["dataset"], j["seed"], j["condition"]) == (args.dataset, args.seed, args.condition))
        if args.condition != "word_1s":
            train_job(m, job)
        encode_job(m, job)
    else:
        export(freeze())

# --- Merged from single_word_long_window_test_v2_results.py ---
"""Paired, duration-budget-controlled full-lexicon retrieval; no model forward."""

from collections import defaultdict
import os
from pathlib import Path
import time

import numpy as np

from braindecoding.full_vocabulary_context_test_v2 import full_vocab_ranks, score_ranks
from braindecoding.unified_sentence_v2_artifacts import csv_write
from braindecoding.word_paired_statistics_test_v2 import bootstrap_weights, metadata
study = sys.modules[__name__]


CONDITIONS = ("word_1s", "long_1p27", "long_1p5", "rolling_k0", "rolling_k1", "rolling_k2")
NAMES = dict(pallier2025="Pallier2025", chineseeeg2_littleprince="ChineseEEG-2")


def budget_mask(ids, timing, horizon, samples):
    rows = [timing[event] for event in ids]
    return np.asarray([r["common"] and r["ends"][str(horizon)] is not None
                       and r["ends"][str(horizon)] <= r["start"] + samples for r in rows], bool)


def subject_scores(weights, counts, hits):
    """Hits: condition x seed x cutoff x block x subject; share one draw."""
    denominator = np.asarray(weights) @ counts
    study.require((denominator > 0).all(), "A material draw omitted a subject")
    numerator = np.einsum("rb,cgkbu->cgkru", weights, hits, optimize=True)
    return (numerator / denominator[None, None, None, :, :]).mean(axis=-1).mean(axis=1)


def paired_statistics(info, ids, subjects, ranks, meta, policy, index):
    """Ranks: two conditions x three seeds x same unique word-event population."""
    blocks = [b["block"] for b in info["blocks"]]
    bi = {b: i for i, b in enumerate(blocks)}
    names = sorted(set(subjects))
    si = {s: i for i, s in enumerate(names)}
    bq = np.asarray([bi[meta[e]["block"]] for e in ids])
    sq = np.asarray([si[s] for s in subjects])
    counts = np.zeros((len(blocks), len(names)), np.float64)
    np.add.at(counts, (bq, sq), 1.)
    hits = np.zeros((2, 3, 2, len(blocks), len(names)), np.float64)
    for condition in range(2):
        for seed in range(3):
            for cutoff_index, cutoff in enumerate((10, 1)):
                np.add.at(hits[condition, seed, cutoff_index], (bq, sq), ranks[condition, seed] <= cutoff)
    observed = subject_scores(np.ones((1, len(blocks))), counts, hits)[:, :, 0]
    params = dict(random_seed=policy["random_seed"], repetitions=policy["bootstrap_repetitions"])
    weights = bootstrap_weights(info, params, index)
    scores = subject_scores(weights, counts, hits)
    differences = scores[1] - scores[0]
    lower, upper = np.quantile(differences, (.025, .975), axis=1)
    contributions = (((hits[1] - hits[0]).mean(axis=0)) / counts.sum(axis=0)[None, None, :]).mean(axis=-1)
    # One sign per material block is shared by subjects, seeds and cutoffs.
    rng = np.random.default_rng(np.random.SeedSequence([policy["random_seed"], index, 271828]))
    signs = rng.choice((-1., 1.), size=(policy["sign_flip_repetitions"], len(blocks)))
    null = contributions @ signs.T
    effect = observed[1] - observed[0]
    study.require(np.allclose(contributions.sum(axis=1), effect, atol=1e-12, rtol=0), "Paired subject weighting differs")
    pvalues = (1 + (np.abs(null) >= np.abs(effect[:, None]) - 1e-15).sum(axis=1)) / (1 + len(signs))
    return dict(unique_events=len(ids), subjects=len(names), blocks=len(blocks), nonempty_blocks=int((counts.sum(axis=1) > 0).sum()),
        bootstrap_repetitions=len(weights), sign_flip_repetitions=len(signs),
        top10=dict(left=float(observed[0, 0]), right=float(observed[1, 0]), effect=float(effect[0]),
                   ci_lower=float(lower[0]), ci_upper=float(upper[0]), p_raw=float(pvalues[0])),
        top1=dict(left=float(observed[0, 1]), right=float(observed[1, 1]), effect=float(effect[1]),
                  ci_lower=float(lower[1]), ci_upper=float(upper[1]), p_raw=float(pvalues[1])))


def holm(values):
    values = np.asarray(values, np.float64)
    order = np.argsort(values, kind="stable")
    adjusted = np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values))))
    result = np.empty_like(values)
    result[order] = np.minimum(1., adjusted)
    return result


def load_dataset(m, parent, dataset):
    info = parent["datasets"][dataset]
    meta = metadata(info)
    ids = np.asarray(sorted(meta))
    words = np.asarray([meta[e]["word"] for e in ids])
    subjects = np.asarray([meta[e]["subject"] for e in ids])
    common = np.asarray([meta[e]["common"] for e in ids])
    with np.load(info["embedding_cache"], allow_pickle=False) as cache:
        index = {w: i for i, w in enumerate(cache["words"].astype(str))}
        bank = cache["embeddings"][[index[w] for w in info["vocabulary"]]]
    bank50 = bank[[info["vocabulary"].index(w) for w in info["vocabulary50"]]]
    all_ranks, strict_ranks, sources = {}, {}, {}
    for condition in CONDITIONS:
        ranks, diagnostic = [], []
        for seed in range(3):
            if condition.startswith("rolling_"):
                job = next(j for j in m["rolling"] if (j["dataset"], j["seed"]) == (dataset, seed))
                path = Path(job["prediction"])
                horizon = int(condition[-1])
                with np.load(path, allow_pickle=False) as archive:
                    events = archive["event_ids"].astype(str)
                    query = archive[f"k{horizon}"]
                    masks = archive["mask2"]
                study.require(np.array_equal(masks, [meta[e]["common"] for e in events]), "Rolling common mask changed")
            else:
                items = m["references"] if condition == "word_1s" else m["jobs"]
                job = next(j for j in items if (j["dataset"], j["seed"], j["condition"]) == (dataset, seed, condition))
                if condition == "word_1s":
                    expected = job["checkpoint_sha256"]
                else:
                    expected = study.completed_training(m, job)["checkpoints"]["best.pt"]["sha256"]
                side = study.verify_prediction(m, job, expected)
                path = study.prediction_root(job) / "predictions.npz"
                with np.load(path, allow_pickle=False) as archive:
                    events, query = archive["event_ids"].astype(str), archive["predictions"]
                    study.require(np.array_equal(archive["words"], [meta[e]["word"] for e in events])
                        and np.array_equal(archive["common_mask"], [meta[e]["common"] for e in events]), "Single-word targets changed")
                for source in (path.with_name("metrics.json"), Path(side["checkpoint"])):
                    sources[str(source)] = study.sha256(source)
            study.require(len(events) == len(set(events)) == len(meta) and set(events) == set(meta), "Test event population differs")
            order = np.argsort(events)
            study.require(np.array_equal(events[order], ids), "Test event alignment differs")
            query = query[order]
            full, covered = full_vocab_ranks(query, words, bank, info["vocabulary"])
            selected = np.isin(words, info["vocabulary50"])
            strict = np.full(len(ids), np.inf)
            from braindecoding.evaluation.retrieval import retrieval_ranks
            strict[selected] = retrieval_ranks(query[selected], words[selected], bank50, info["vocabulary50"])
            ranks.append(full)
            diagnostic.append(strict)
            sources[str(path)] = study.sha256(path)
        all_ranks[condition] = np.stack(ranks)
        strict_ranks[condition] = np.stack(diagnostic)
    return dict(info=info, meta=meta, ids=ids, words=words, subjects=subjects, common=common,
                covered=covered, ranks=all_ranks, strict_ranks=strict_ranks, sources=sources)


def figure(comparisons):
    os.environ.setdefault("MPLCONFIGDIR", str(study.ROOT / ".cache/matplotlib" / study.ID))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in comparisons if r["population"] == "within_long_window_budget"]
    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    for i, row in enumerate(rows):
        value = row["top10"]
        y = len(rows) - i - 1
        color = "#206d9a" if row["dataset"] == "pallier2025" else "#c16d32"
        ax.plot([100 * value["ci_lower"], 100 * value["ci_upper"]], [y, y], color=color, linewidth=2.2)
        ax.scatter([100 * value["effect"]], [y], color=color, s=46, zorder=3)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(rows)), [f"{NAMES[r['dataset']]}: K{r['horizon']} - {r['actual_seconds']:.2f} s CNN  (n={r['unique_events']:,})" for r in reversed(rows)])
    ax.set_xlabel("Full-training-vocabulary Top-10 difference (percentage points)")
    ax.set_title("Cross-word model vs single-word long window\nPaired material-block 95% CI; rolling input ends within long-window budget", fontsize=11)
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    for extension in ("png", "svg"):
        fig.savefig(study.REPORT / f"long_window_comparison.{extension}", dpi=180)
    plt.close(fig)


def export(m):
    from threadpoolctl import threadpool_limits
    started = time.perf_counter()
    study.verify_manifest(m)
    study.verify_sources(m, full=True)
    parent = study.read(study.PARENT)
    timings = study.read(study.RUNTIME / "timing_by_event.json")
    rows, subject_rows, comparisons, summaries, sources, population_rows = [], [], [], [], {}, []
    with threadpool_limits(limits=2):
        for index, dataset in enumerate(study.neural.DATASETS):
            data = load_dataset(m, parent, dataset)
            info, ids, words, subjects = data["info"], data["ids"], data["words"], data["subjects"]
            timing = {r["event_id"]: r for r in timings[dataset]}
            masks = dict(common_words=data["common"],
                within_budget_k1=budget_mask(ids, timing, 1, 64), within_budget_k2=budget_mask(ids, timing, 2, 75))
            for population, mask in masks.items():
                elapsed = {f"k{h}": [(timing[e]["ends"][str(h)] - timing[e]["start"]) / 50 for e in ids[mask]] for h in (1, 2)}
                population_rows.append(dict(dataset=dataset, population=population, unique_events=int(mask.sum()),
                    subjects=len(set(subjects[mask])), full_vocabulary=len(info["vocabulary"]),
                    k1_mean_coverage_s=float(np.mean(elapsed["k1"])), k2_mean_coverage_s=float(np.mean(elapsed["k2"]))))
                for condition in CONDITIONS:
                    for seed in range(3):
                        for candidates in ("full_training_vocabulary", "strict50"):
                            use = mask if candidates == "full_training_vocabulary" else mask & np.isin(words, info["vocabulary50"])
                            ranks = data["ranks"] if candidates == "full_training_vocabulary" else data["strict_ranks"]
                            score = score_ranks(ranks[condition][seed, use], words[use], subjects[use], data["covered"][use])
                            base = dict(dataset=dataset, condition=condition, seed=seed, population=population,
                                        candidates=candidates, candidate_count=len(info["vocabulary"]) if candidates == "full_training_vocabulary" else 50)
                            rows.append({**base, **score})
                            for subject in sorted(set(subjects[use])):
                                selected = use & (subjects == subject)
                                value = score_ranks(ranks[condition][seed, selected], words[selected], subjects[selected], data["covered"][selected])
                                subject_rows.append({**base, "subject": subject, **value})
            for horizon, condition, samples in ((1, "long_1p27", 64), (2, "long_1p5", 75)):
                for population, mask in (("common_words", masks["common_words"]),
                                          ("within_long_window_budget", masks[f"within_budget_k{horizon}"])):
                    ranks = np.stack([data["ranks"][condition][:, mask], data["ranks"][f"rolling_k{horizon}"][:, mask]])
                    stats = paired_statistics(info, ids[mask], subjects[mask], ranks, data["meta"], m["settings"]["statistics"],
                                              index * 10 + horizon * 2 + (population == "within_long_window_budget"))
                    comparisons.append(dict(dataset=dataset, horizon=horizon, left=condition, right=f"rolling_k{horizon}",
                        actual_seconds=samples / 50, population=population, **stats))
            rank_path = study.REPORT / f"{dataset}_retrieval_ranks.npz"
            study.REPORT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(rank_path, event_ids=ids, words=words, subjects=subjects,
                common_mask=masks["common_words"], within_budget_k1=masks["within_budget_k1"], within_budget_k2=masks["within_budget_k2"],
                conditions=np.asarray(CONDITIONS), full_ranks=np.stack([data["ranks"][c] for c in CONDITIONS]),
                strict50_ranks=np.stack([data["strict_ranks"][c] for c in CONDITIONS]))
            sources.update(data["sources"])
            sources[str(rank_path)] = study.sha256(rank_path)
    primary = [r for r in comparisons if r["population"] == "within_long_window_budget"]
    study.require(len(primary) == 4, "Primary comparison family incomplete")
    for row, pvalue in zip(primary, holm([r["top10"]["p_raw"] for r in primary])):
        row["top10"]["p_holm"] = float(pvalue)
        row["top10"]["significant_holm_05"] = bool(pvalue < .05)
    metrics = ("subject_equal_recall_at_10", "subject_equal_recall_at_1", "macro_recall_at_10", "macro_recall_at_1", "micro_recall_at_10", "micro_recall_at_1")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["condition"], row["population"], row["candidates"])].append(row)
    for (dataset, condition, population, candidates), values in grouped.items():
        item = dict(dataset=dataset, condition=condition, population=population, candidates=candidates,
                    candidate_count=values[0]["candidate_count"], unique_events=values[0]["query_count"], seeds=3)
        for metric in metrics:
            item[metric + "_mean"] = float(np.mean([v[metric] for v in values]))
            item[metric + "_sd"] = float(np.std([v[metric] for v in values], ddof=1))
        summaries.append(item)
    study.REPORT.mkdir(parents=True, exist_ok=True)
    for name, values in (("by_seed", rows), ("by_subject", subject_rows), ("summary", summaries), ("populations", population_rows)):
        path = study.REPORT / f"{name}.csv"
        csv_write(path, values)
        sources[str(path)] = study.sha256(path)
    figure(comparisons)
    for extension in ("png", "svg"):
        path = study.REPORT / f"long_window_comparison.{extension}"
        sources[str(path)] = study.sha256(path)
    result = dict(status="completed", protocol=study.ID, manifest_sha256=m["manifest_sha256"],
        parent_manifest_sha256=m["parent_manifest_sha256"], split="test", test_evaluated=True,
        selection="validation_legacy50_best_pt", primary_metric="full_vocabulary_subject_equal_top10",
        primary_population="within_long_window_budget", seeds=[0, 1, 2], trained_word_models=12,
        language_model_used=False, cross_word_transformer_in_long_models=False, comparisons=comparisons,
        summaries=summaries, populations=population_rows, source_sha256=sources,
        cpu_export_seconds=time.perf_counter() - started, completed_unix=time.time())
    study.write_json(study.REPORT / "results.json", result)
    lines = ["# 单词长窗对照结果", "", "两个数据集、三个种子；12 个从头训练的长窗 CNN。1.27 秒请求实际为 1.28 秒（64 样本），1.5 秒为 75 样本。",
        "原 1 秒 CNN、K0/K1/K2 为经核验的同种子参照；本实验不使用语言模型，不以测试成绩选模。", "",
        "主比较限制在 K0/K1/K2 共同可用、且对应 K1/K2 最后词窗结束不晚于长窗结束的词位。候选为完整冻结训练词库，库外目标保留为错误。",
        "Top-10 先受试者等权、再三种子平均；CI 为 10,000 次配对材料块 bootstrap，四项主比较用块符号翻转和 Holm 校正。正差值表示跨词模型更好。", "",
        "| 数据集 | 比较 | 独立词位 | 长窗 Top-10 | K1/K2 Top-10 | 差值（百分点） | 95% CI | Holm p |",
        "|---|---|---:|---:|---:|---:|---|---:|"]
    for row in primary:
        v = row["top10"]
        lines.append(f"| {NAMES[row['dataset']]} | K{row['horizon']} − {row['actual_seconds']:.2f} s | {row['unique_events']:,} | {100*v['left']:.2f}% | {100*v['right']:.2f}% | {100*v['effect']:+.2f} | [{100*v['ci_lower']:+.2f}, {100*v['ci_upper']:+.2f}] | {v['p_holm']:.4f} |")
    lines += ["", "## 全共同词位：完整训练词库", "", "以下六种模型始终在同一 K0/K1/K2 共同人口上评价；数字为三种子均值。", "",
        "| 数据集 | 模型 | 独立词位 | 受试者等权 Top-1 | 受试者等权 Top-10 | 词类型宏 Top-10 | 事件微 Top-10 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        if row["population"] == "common_words" and row["candidates"] == "full_training_vocabulary":
            lines.append(f"| {NAMES[row['dataset']]} | {row['condition']} | {row['unique_events']:,} | {100*row['subject_equal_recall_at_1_mean']:.2f}% | {100*row['subject_equal_recall_at_10_mean']:.2f}% | {100*row['macro_recall_at_10_mean']:.2f}% | {100*row['micro_recall_at_10_mean']:.2f}% |")
    lines += ["", "## 时长与解释", "", "固定长窗近似对应典型观测时长，K1/K2 的实际时长随词距变化。全共同词位的均值：", "",
        "| 数据集 | K1 从目标词起点至最后窗结束 | K2 从目标词起点至最后窗结束 |", "|---|---:|---:|"]
    for row in population_rows:
        if row["population"] == "common_words":
            lines.append(f"| {NAMES[row['dataset']]} | {row['k1_mean_coverage_s']:.3f} s | {row['k2_mean_coverage_s']:.3f} s |")
    lines += ["", "主比较使用真实整数样本边界，避免 K1/K2 单纯观察更久的解释。K1/K2 还使用历史词窗，网络容量和训练目标也与 CNN 不同；正结果支持跨词建模在延长观测之外提供增益，不能独立证明唯一机制。",
        "区间条件于已观察受试者与三个训练种子；材料块在 recording/chapter 内抽样，所有模型和种子共享同一材料抽样。", "",
        "[逐种子指标](by_seed.csv) · [完整汇总](summary.csv) · [人口和时长](populations.csv) · [完整 JSON](results.json) · [来源审计](audit.json)"]
    (study.REPORT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    study.verify_sources(m, full=True)
    study.write_json(study.REPORT / "audit.json", dict(status="passed", manifest_sha256=m["manifest_sha256"],
        input_sources_unchanged=True, new_training_jobs=12, reused_word_checkpoints=6, reused_rolling_prediction_jobs=6,
        identical_population_per_comparison=True, oov_retained_as_errors=True, test_checkpoint_selection=False,
        source_sha256=sources, results_sha256=study.sha256(study.REPORT / "results.json")))
    print("Full long-window comparison exported", flush=True)
    return result


if __name__ == "__main__":
    main()
