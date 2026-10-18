"""Frozen, serial context/delay baselines extending completed v2 training."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from braindecoding import unified_rolling_v2_training as base
from braindecoding.config import PROJECT_ROOT, project_path
from braindecoding.evaluation.retrieval import fixed_vocabulary_retrieval_metrics
from braindecoding.experiment import file_sha256
from braindecoding.training.runtime import cpu_state_dict, load_checkpoint, save_checkpoint, set_seed
from braindecoding.training.word import encode_loader, train_one_epoch
from braindecoding.unified_rolling_v2_queue import verify_completed

CONFIG = PROJECT_ROOT / "configs/unified_rolling/context_baselines_v2.yaml"
PROTOCOL = PROJECT_ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
RUNTIME = PROJECT_ROOT / ".cache/runs/unified_context_baselines_v2"
REPORT = PROJECT_ROOT / "data/results/studies/unified_context_baselines_v2"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def settings():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def word_checkpoint(dataset, seed):
    for entry in base.config()["execution"]["reuse_runs"]:
        if (entry["dataset"], entry["seed"], entry["stage"]) == (dataset, seed, "word"):
            return project_path(entry["path"]) / "best.pt"
    return base.output_dir(dataset, seed, "word") / "best.pt"


def build_config(dataset, seed, condition):
    spec = settings()
    name = spec["protocol"]["aliases"].get(condition, condition)
    window = spec["protocol"]["conditions"][name]
    cfg = base.build_config(dataset, seed, "revision", str(word_checkpoint(dataset, seed)))
    cfg["model"].pop("local_window_revision")
    if window is not None:
        cfg["model"]["transformer"].update(query_window_left_radius=window[0], query_window_right_radius=window[1])
    cfg["training"].pop("revision_cutoff_policy")
    cfg["experiment"]["output_group"] = spec["protocol"]["id"]
    cfg["training"]["output_dir"] = str(project_path(spec["execution"]["output_root"]) / dataset / f"seed-{seed:03d}" / name)
    cfg["evaluation"]["selection_metric"] = "legacy_common_macro_top10"
    cfg["unified_protocol"] = {"id": spec["protocol"]["id"], "parent": "unified_rolling_v2",
        "stage": "context", "condition": name, "test_evaluated": False,
        "primary_checkpoint": "best.pt", "secondary_checkpoint": None,
        "loss_normalization": "one_state_batch_siglip", "amp_dtype": "float16"}
    cfg["provenance"]["unified_protocol"] = spec["protocol"]["id"]
    return cfg


def common_mask(table, event_ids, group_column):
    if table.event_id.duplicated().any() or len(event_ids) != len(set(event_ids)):
        raise ValueError("Duplicate event IDs invalidate paired evaluation.")
    if len(event_ids) != len(table) or set(event_ids) != set(table.event_id):
        raise ValueError("Validation event population changed.")
    mask = table.groupby(group_column, sort=False).cumcount(ascending=False).ge(2)
    lookup = dict(zip(table.event_id, mask))
    return torch.tensor([lookup[event] for event in event_ids], dtype=torch.bool)


def metrics_from_encoded(encoded, dataset, vocabulary, vocab50, cfg, adapter):
    common = common_mask(dataset.table, encoded["event_ids"], adapter["group_column"])
    if not torch.isfinite(encoded["predictions"]).all():
        raise FloatingPointError("Non-finite validation predictions.")
    if int(common.sum()) != adapter["expected_common_rows"]:
        raise ValueError("Common event count changed.")
    evaluation = {**encoded, "revision_predictions": {0: encoded["predictions"]},
                  "revision_availability_masks": {0: common}}
    strict = base.strict_bank_metrics(evaluation, dataset, vocab50)["0"]
    for population, expected in (("all_targets", adapter["expected_all_vocab50_queries"]),
                                 ("common_targets", adapter["expected_common_vocab50_queries"])):
        if strict[population]["candidate_count"] != 50 or strict[population]["query_count"] != expected:
            raise ValueError("Strict-50 population changed.")
    legacy = fixed_vocabulary_retrieval_metrics(encoded["predictions"][common], encoded["targets"][common],
        np.asarray(encoded["words"])[common.numpy()], vocab50)
    if legacy["candidate_count"] != adapter["expected_legacy_common_candidates"]:
        raise ValueError("Legacy common selection candidate count changed.")
    return {"legacy_common": legacy, "selection_score": legacy["macro_recall_at_10"],
            "strict_vocab50": strict,
            "full_vocabulary_common": base.full_vocabulary_metrics(evaluation, dataset, vocabulary)}


class CoverageLoader:
    """Observe actual consumed IDs; never alter batches or consume RNG."""
    def __init__(self, loader, expected_ids, status):
        self.loader, self.expected, self.status = loader, set(expected_ids), status
        self.seen = set()

    def __iter__(self):
        for number, batch in enumerate(self.loader, 1):
            ids = list(batch["event_id"])
            if len(ids) != len(set(ids)) or self.seen.intersection(ids):
                raise ValueError("A training target was repeated within one epoch.")
            self.seen.update(ids)
            yield batch
            if number == 1 or number % 25 == 0:
                self.status(status="training", completed_batches=number, consumed_targets=len(self.seen))
        if self.seen != self.expected:
            raise ValueError("Training target coverage changed.")


def frozen_inputs(cfg):
    hashes = base.source_hashes(cfg, base.config()["adapters"][cfg["experiment"]["dataset"]])
    for path in (CONFIG, PROTOCOL, Path(__file__),
                 PROJECT_ROOT / "braindecoding/training/unified_context_baselines_v2_report.py",
                 PROJECT_ROOT / "braindecoding/training/unified_context_baselines_v2_preflight.py"):
        hashes[str(path.resolve())] = file_sha256(path)
    return hashes


def freeze():
    base.assert_ready(base.config())
    path = RUNTIME / "plan.json"
    if path.exists():
        plan = read(path)
        base.check_hashes(plan["source_sha256"])
        return plan
    spec = settings()
    preflight = read(REPORT / "preflight.json")
    if (preflight["status"] != "passed" or preflight["worker_sha256"] != file_sha256(Path(__file__))
            or preflight["config_sha256"] != file_sha256(CONFIG)
            or set(preflight["datasets"]) != set(spec["protocol"]["datasets"])):
        raise ValueError("GPU preflight is missing or stale.")
    hashes, jobs, words = {}, [], []
    for seed in spec["protocol"]["seeds"]:
        for dataset in spec["protocol"]["datasets"]:
            source = word_checkpoint(dataset, seed)
            words.append(verify_completed(source.parent, dataset, seed, "word", deep=True))
            for condition in spec["protocol"]["conditions"]:
                cfg = build_config(dataset, seed, condition)
                if condition == "local_r2":
                    if cfg != build_config(dataset, seed, "independent_k2"):
                        raise ValueError("K2 and radius 2 are not exact aliases.")
                    hashes.update(frozen_inputs(cfg))
                jobs.append({"dataset": dataset, "seed": seed, "condition": condition,
                             "output_dir": cfg["training"]["output_dir"], "config": cfg})
    if len(jobs) != 48:
        raise ValueError("Expected exactly 48 unique trainings.")
    priority = {"local_r2": 0, "independent_k0": 1, "independent_k1": 2}
    jobs.sort(key=lambda job: (0 if job["condition"] in priority else 1, job["seed"],
        spec["protocol"]["datasets"].index(job["dataset"]),
        list(spec["protocol"]["conditions"]).index(job["condition"])))
    plan = {"protocol": spec["protocol"], "created_unix": time.time(), "jobs": jobs,
            "word_initializations": words, "source_sha256": hashes, "preflight": preflight,
            "test_evaluated": False}
    save(path, plan)
    save(REPORT / "frozen_plan.json", plan)
    return plan


def ensure_space(path):
    parent = Path(path)
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free / 1024**3
    required = settings()["execution"]["minimum_free_gib"]
    if free < required:
        raise OSError(f"Insufficient disk space: {free:.2f} GiB free; need {required:.1f} GiB before starting a run.")
    return free


def verify_finished(job, plan):
    out = Path(job["output_dir"])
    summary, manifest = read(out / "training_summary.json"), read(out / "run_manifest.json")
    if (summary["status"] != "completed" or manifest["status"] != "completed"
            or manifest["config"] != job["config"] or summary["test_evaluated"] is not False
            or manifest["plan_sha256"] != file_sha256(RUNTIME / "plan.json")
            or summary["checkpoint_sha256"] != file_sha256(out / "best.pt")
            or summary["predictions_sha256"] != file_sha256(out / "best_validation.npz")):
        raise ValueError(f"Incomplete or incompatible completed run: {out}")
    return summary


def run_job(index):
    plan = read(RUNTIME / "plan.json")
    base.check_hashes(plan["source_sha256"])
    job = plan["jobs"][index]
    cfg, dataset, seed = job["config"], job["dataset"], job["seed"]
    out, tc = Path(job["output_dir"]), cfg["training"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {out}")
    ensure_space(out)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    out.mkdir(parents=True)
    status = base.RunStatus(out, dataset, seed, job["condition"])
    manifest = {**job, "status": "preparing", "plan_sha256": file_sha256(RUNTIME / "plan.json"),
                "source_sha256": plan["source_sha256"], "test_evaluated": False}
    save(out / "run_manifest.json", manifest)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    try:
        status(status="preparing")
        set_seed(seed)
        torch.set_num_threads(2)
        train, train_loader, sampler = base.dataset_and_loader(cfg, "train")
        val, val_loader, _ = base.dataset_and_loader(cfg, "val")
        adapter = base.config()["adapters"][dataset]
        vocabulary, vocab50, _, population = base.populations(cfg, train, val, adapter)
        model = base.build_brain_embedding_model(train.channel_count, train.channel_positions,
                    train.subject_count, cfg["model"]).cuda()
        loss_module = base.build_siglip_loss(cfg["loss"]).cuda()
        embedding_hash = file_sha256(cfg["cache"]["text_embeddings"])
        source = Path(tc["pretrained_brain_encoder_checkpoint"])
        cp = load_checkpoint(source)
        base.validate_word_checkpoint(cp, cfg, train, embedding_hash)
        model.brain_encoder.load_state_dict({key.removeprefix("brain_encoder."): value
            for key, value in cp["model_state"].items() if key.startswith("brain_encoder.")}, strict=True)
        initialization = {"path": str(source), "sha256": file_sha256(source), "epoch": cp["epoch"]}
        del cp
        optimizer = base.build_adamw_for_modules([model, loss_module], tc)
        scheduler = base.build_cosine_annealing_scheduler(optimizer, tc["epochs"])
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        manifest.update(status="running", initialization=initialization, population=population,
                        embedding_sha256=embedding_hash, vocabulary50=vocab50,
                        train_rows=len(train), validation_rows=len(val), group_column=base.group_column(cfg))
        save(out / "run_manifest.json", manifest)
        history, best_row, best_score, stale, updates = [], None, -float("inf"), 0, 0
        for epoch in range(tc["epochs"]):
            base.check_hashes(plan["source_sha256"])
            started = time.perf_counter()
            sampler.set_epoch(epoch)
            frozen = epoch < tc["freeze_brain_encoder_epochs"]
            base.pallier.set_brain_encoder_trainable(model, not frozen)
            status(status="training", epoch=epoch + 1, completed_epochs=epoch, completed_batches=0,
                   max_epochs=tc["epochs"], brain_encoder_frozen=frozen)
            loader = CoverageLoader(train_loader, train.table.event_id, status)
            loss, epoch_updates, batches, diagnostics = train_one_epoch(model, loss_module, loader,
                optimizer, scaler, torch.device("cuda"), tc, freeze_brain_encoder=frozen,
                return_stats=True, return_diagnostics=True, epoch_index=epoch)
            if not np.isfinite(loss) or not epoch_updates:
                raise FloatingPointError("Non-finite loss or no successful optimizer update.")
            updates += epoch_updates
            status(status="validating", completed_batches=batches)
            encoded = encode_loader(model, val_loader, torch.device("cuda"), amp=True)
            metrics = metrics_from_encoded(encoded, val, vocabulary, vocab50, cfg, adapter)
            scheduler.step()
            row = {"epoch": epoch + 1, "train_loss": loss, "seconds": time.perf_counter() - started,
                   "optimizer_updates": updates, "epoch_optimizer_updates": epoch_updates, "batches": batches,
                   "amp_skipped_updates": batches - epoch_updates, "brain_encoder_frozen": frozen,
                   "training_targets": len(loader.seen), "training_diagnostics": diagnostics, "validation": metrics}
            history.append(row)
            score = metrics["selection_score"]
            if not np.isfinite(score):
                raise FloatingPointError("Non-finite selection score.")
            stale += 1
            if score > best_score:
                best_score, best_row, stale = score, row, 0
                save_checkpoint(out / "best.pt", {"dataset": dataset, "seed": seed, "stage": "context",
                    "condition": job["condition"], "epoch": epoch + 1, "config": cfg,
                    "model_state": cpu_state_dict(model), "loss_state": cpu_state_dict(loss_module),
                    "model_config": cfg["model"], "loss_config": cfg["loss"], "text_embedding_config": cfg["text_embedding"],
                    "embedding_sha256": embedding_hash, "selection": "legacy_common_macro_top10", "metrics": metrics,
                    "initialization": initialization, "channel_names": train.channel_names,
                    "channel_positions": np.asarray(train.channel_positions).tolist(),
                    "subject_ids": list(getattr(train, "subject_ids", cfg["dataset"]["subjects"])), "test_evaluated": False})
                # Save full-precision outputs for later paired analyses without another GPU pass.
                with (out / "best_validation.npz.tmp").open("wb") as stream:
                    np.savez_compressed(stream, predictions=encoded["predictions"].numpy(),
                        event_ids=np.asarray(encoded["event_ids"]), words=np.asarray(encoded["words"]),
                        common_mask=common_mask(val.table, encoded["event_ids"], adapter["group_column"]).numpy(),
                        epoch=np.asarray(epoch + 1))
                os.replace(out / "best_validation.npz.tmp", out / "best_validation.npz")
            save(out / "progress.json", {"history": history, "best": best_row})
            status(status="epoch_complete", completed_epochs=epoch + 1, best_epoch=best_row["epoch"],
                   best_selection_score=best_score, epoch_seconds=row["seconds"])
            del encoded
            if stale >= tc["patience"]:
                break
        base.check_hashes(plan["source_sha256"])
        summary = {"status": "completed", "dataset": dataset, "seed": seed, "condition": job["condition"],
            "best": best_row, "history": history, "completed_epochs": len(history), "test_evaluated": False,
            "stop_reason": "patience_exhausted" if stale >= tc["patience"] else "maximum_epochs",
            "checkpoint_sha256": file_sha256(out / "best.pt"),
            "predictions_sha256": file_sha256(out / "best_validation.npz"), "initialization": initialization}
        save(out / "training_summary.json", summary)
        manifest.update(status="completed", completed_epochs=len(history))
        save(out / "run_manifest.json", manifest)
        status(status="completed", completed_epochs=len(history))
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}")
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(out / "run_manifest.json", manifest)
        raise


def queue():
    import psutil
    plan = freeze()
    lock = RUNTIME / "queue.lock"
    if lock.exists():
        owner = read(lock)
        if psutil.pid_exists(owner["pid"]) and abs(psutil.Process(owner["pid"]).create_time() - owner["created"]) < 1:
            raise RuntimeError("An experiment queue is already active.")
        # A dead coordinator may have left a live training child.
        for process in psutil.process_iter(["pid", "cmdline"]):
            command = process.info["cmdline"] or []
            if "braindecoding.unified_context_baselines_v2" in command and "worker" in command:
                raise RuntimeError("An existing training child is still active.")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, stream)
    completed = []
    try:
        for index, job in enumerate(plan["jobs"]):
            base.check_hashes(plan["source_sha256"])
            out = Path(job["output_dir"])
            if out.exists():
                verify_finished(job, plan)
                completed.append(index)
                continue
            try:
                free = ensure_space(out)
            except OSError as error:
                save(RUNTIME / "status.json", {"status": "waiting_for_disk_space", "error": str(error),
                    "completed_jobs": len(completed), "planned_jobs": len(plan["jobs"]), "next_job": job,
                    "updated_unix": time.time(), "test_evaluated": False})
                report()
                return
            status = {"status": "running", "pid": os.getpid(), "job_index": index,
                "dataset": job["dataset"], "seed": job["seed"], "condition": job["condition"],
                "completed_jobs": len(completed), "planned_jobs": len(plan["jobs"]), "free_gib": free,
                "updated_unix": time.time(), "test_evaluated": False}
            save(RUNTIME / "status.json", status)
            report()
            with (RUNTIME / f"job-{index:02d}.stdout.log").open("a", encoding="utf-8") as stdout, \
                 (RUNTIME / f"job-{index:02d}.stderr.log").open("a", encoding="utf-8") as stderr:
                process = subprocess.Popen([sys.executable, "-m", "braindecoding.unified_context_baselines_v2",
                    "worker", "--index", str(index)], cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                save(RUNTIME / "status.json", {**status, "worker_pid": process.pid})
                code = process.wait()
            if code:
                raise RuntimeError(f"Training job {index} failed with exit code {code}; see job-{index:02d}.stderr.log")
            verify_finished(job, plan)
            completed.append(index)
            report()
        save(RUNTIME / "status.json", {"status": "completed", "completed_jobs": len(completed),
            "planned_jobs": len(plan["jobs"]), "updated_unix": time.time(), "test_evaluated": False})
        report()
    except BaseException as error:
        save(RUNTIME / "status.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
            "completed_jobs": len(completed), "updated_unix": time.time(), "test_evaluated": False})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "queue", "worker", "report"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps({"jobs": len(freeze()["jobs"])}))
    elif args.command == "queue":
        queue()
    elif args.command == "worker":
        if args.index is None or not 0 <= args.index < 48:
            parser.error("worker requires --index in 0..47")
        run_job(args.index)
    else:
        report()

# --- Merged from unified_context_baselines_v2_report.py ---
"""Export completed context baselines and pair them with frozen rolling results."""

import csv
import shutil
import time
from pathlib import Path

import numpy as np


METRICS = ("macro_recall_at_1", "macro_recall_at_10", "micro_recall_at_1", "micro_recall_at_10")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report():
    REPORT.mkdir(parents=True, exist_ok=True)
    plan = read(RUNTIME / "plan.json")
    rows, slots = [], []
    for job in plan["jobs"]:
        out = Path(job["output_dir"])
        identity = {key: job[key] for key in ("dataset", "seed", "condition")}
        if (out / "training_summary.json").exists() and read(out / "run_manifest.json")["status"] == "completed":
            result = verify_finished(job, plan)
            slots.append({**identity, "status": "completed", "completed_epochs": result["completed_epochs"],
                          "best_epoch": result["best"]["epoch"], "checkpoint": str(out / "best.pt")})
            for population, values in result["best"]["validation"]["strict_vocab50"].items():
                rows.append({**identity, "population": population, "best_epoch": result["best"]["epoch"],
                    "checkpoint": str(out / "best.pt"), **{key: values[key] for key in
                    ("candidate_count", "query_count", "target_count", *METRICS)}})
        else:
            state = read(out / "status.json") if (out / "status.json").exists() else {}
            slots.append({**identity, "status": state.get("status", "pending"),
                          "completed_epochs": state.get("completed_epochs", 0)})
    write_csv(REPORT / "run_status.csv", slots,
              ["dataset", "seed", "condition", "status", "completed_epochs", "best_epoch", "checkpoint"])
    write_csv(REPORT / "context_by_seed.csv", rows,
              ["dataset", "seed", "condition", "population", "best_epoch", "candidate_count", "query_count",
               "target_count", *METRICS, "checkpoint"])
    summary = []
    for dataset in plan["protocol"]["datasets"]:
        for condition in plan["protocol"]["conditions"]:
            for population in ("all_targets", "common_targets"):
                group = [r for r in rows if (r["dataset"], r["condition"], r["population"]) == (dataset, condition, population)]
                item = {"dataset": dataset, "condition": condition, "population": population, "n_seeds": len(group)}
                for metric in METRICS:
                    values = [r[metric] for r in group]
                    item[metric + "_mean"] = float(np.mean(values)) if values else None
                    item[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                summary.append(item)
    write_csv(REPORT / "context_summary.csv", summary, list(summary[0]))
    sources_path = PROJECT_ROOT / "data/results/exports/unified_rolling_v2_paper_sources.json"
    sources = read(sources_path)
    if sources["included_completed_jobs"] != 12 or sources["pending_jobs"] or sources["test_evaluated"]:
        raise ValueError("Rolling comparison requires all 12 completed v2 jobs.")
    exported = next(e for e in sources["exports"] if e["path"].endswith("revision_by_seed.csv"))
    rolling_path = PROJECT_ROOT / exported["path"]
    if file_sha256(rolling_path) != exported["sha256"]:
        raise ValueError("Rolling export does not match provenance.")
    shutil.copyfile(rolling_path, REPORT / "rolling_by_seed.csv")
    with rolling_path.open(encoding="utf-8-sig", newline="") as stream:
        rolling = list(csv.DictReader(stream))
    comparisons, gains = [], []
    aliases = {"independent_k0": "K0", "independent_k1": "K1", "local_r2": "K2"}
    for row in rows:
        if row["population"] != "common_targets" or row["condition"] not in aliases:
            continue
        horizon = aliases[row["condition"]]
        roll = next(r for r in rolling if (r["dataset"], int(r["seed"]), r["population"], r["horizon"]) ==
                    (row["dataset"], row["seed"], "common_targets", horizon))
        if int(roll["query_count"]) != row["query_count"] or int(roll["candidate_count"]) != 50:
            raise ValueError("Independent and rolling populations differ.")
        item = {"dataset": row["dataset"], "seed": row["seed"], "horizon": horizon,
                "query_count": row["query_count"], "independent_checkpoint": row["checkpoint"],
                "rolling_checkpoint": roll["checkpoint"]}
        for metric in METRICS:
            item["independent_" + metric] = row[metric]
            item["rolling_" + metric] = float(roll[metric])
            item["rolling_minus_independent_" + metric] = float(roll[metric]) - row[metric]
        comparisons.append(item)
    for dataset in plan["protocol"]["datasets"]:
        for seed in plan["protocol"]["seeds"]:
            group = {r["horizon"]: r for r in comparisons if r["dataset"] == dataset and r["seed"] == seed}
            if "K0" in group and "K2" in group:
                item = {"dataset": dataset, "seed": seed, "query_count": group["K0"]["query_count"]}
                for kind in ("independent", "rolling"):
                    for metric in METRICS:
                        item[kind + "_k2_minus_k0_" + metric] = group["K2"][kind + "_" + metric] - group["K0"][kind + "_" + metric]
                gains.append(item)
    write_csv(REPORT / "independent_vs_rolling_by_seed.csv", comparisons,
              list(comparisons[0]) if comparisons else ["dataset", "seed", "horizon"])
    write_csv(REPORT / "delay_gain_by_seed.csv", gains, list(gains[0]) if gains else ["dataset", "seed"])
    completed = sum(s["status"] == "completed" for s in slots)
    sentence = PROJECT_ROOT / "data/results/studies/unified_sentence_v2_revision_gain_full/results.md"
    audit = {"completed_jobs": completed, "planned_jobs": 48, "test_evaluated": False,
             "updated_unix": time.time(), "plan_sha256": file_sha256(RUNTIME / "plan.json"),
             "rolling_export": {"path": str(rolling_path), "sha256": file_sha256(rolling_path)},
             "sentence_report": {"path": str(sentence), "sha256": file_sha256(sentence)},
             "aggregation": "completed seeds only; sample SD; alias K2 is not an independent replicate",
             "manuscript_modified": False}
    save(REPORT / "report.json", audit)
    text = ["# 统一 v2 补充实验进度", "", f"已完成 {completed}/48 次训练。仅使用训练集与验证集。", "",
            "局部窗口：0、1、2、4、8、完整片段。独立延迟：历史 2，未来 0、1、2；独立 K2 引用半径 2 的同一次训练。", "",
            "已有滚动模型：两数据集、三个种子全部完成，严格 50 候选结果已核对并复制至本目录。", "",
            "全量句子 A/C：两数据集、三个种子全部完成，结果见 ../unified_sentence_v2_revision_gain_full/results.md。", "",
            "## 已完成模型（严格 50 候选，共同查询）", "",
            "| 数据集 | 条件 | 种子数 | Macro Top-1 | Macro Top-10 |", "|---|---|---:|---:|---:|"]
    for row in summary:
        if row["population"] == "common_targets" and row["n_seeds"]:
            text.append(f"| {row['dataset']} | {row['condition']} | {row['n_seeds']} | {row['macro_recall_at_1_mean']:.2%} | {row['macro_recall_at_10_mean']:.2%} |")
    text.extend(["", "均值只纳入完成的种子，样本标准差见 context_summary.csv。中间最优轮次不作为完成结果。", "",
                 "逐项状态见 run_status.csv；独立与滚动的同种子配对结果见 independent_vs_rolling_by_seed.csv；延迟增益见 delay_gain_by_seed.csv。"])
    (REPORT / "progress.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if completed == 48:
        plot(summary, comparisons)
    return audit


def plot(summary, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["local_r0", "local_r1", "local_r2", "local_r4", "local_r8", "full_context"]
    datasets = [("pallier2025", "Pallier2025"), ("chineseeeg2_littleprince", "ChineseEEG-2")]
    for population in ("all_targets", "common_targets"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True, constrained_layout=True)
        for ax, (dataset, label) in zip(axes, datasets):
            group = {r["condition"]: r for r in summary if r["dataset"] == dataset and r["population"] == population}
            for metric, name in (("macro_recall_at_1", "Top-1"), ("macro_recall_at_10", "Top-10")):
                ax.errorbar(range(6), [100 * group[c][metric + "_mean"] for c in order],
                            yerr=[100 * group[c][metric + "_sd"] for c in order], marker="o", capsize=3, label=name)
            ax.set(title=label, xticks=range(6), xticklabels=["0", "1", "2", "4", "8", "Full"], xlabel="Context radius")
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Word-type macro recall (%)")
        axes[1].legend(frameon=False)
        for ext in ("png", "pdf"):
            fig.savefig(REPORT / f"context_radius_{population}.{ext}", dpi=200)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True, constrained_layout=True)
    for ax, (dataset, label) in zip(axes, datasets):
        for kind, name in (("independent", "Independent"), ("rolling", "Rolling")):
            values = [[100 * r[kind + "_macro_recall_at_10"] for r in comparisons
                       if r["dataset"] == dataset and r["horizon"] == f"K{k}"] for k in range(3)]
            if any(len(v) != 3 for v in values):
                raise ValueError("Final plot requires all three paired seeds.")
            ax.errorbar(range(3), np.mean(values, axis=1), yerr=np.std(values, axis=1, ddof=1), marker="o", capsize=3, label=name)
        ax.set(title=label, xticks=range(3), xticklabels=["K0", "K1", "K2"], xlabel="Delay")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Common-query macro Top-10 (%)")
    axes[1].legend(frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(REPORT / f"independent_vs_rolling.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    report()


if __name__ == "__main__":
    main()
