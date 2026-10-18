"""Verified, idempotent queue for the accepted 16/16 BGE protocol.

Arm once, then launch-if-ready may be called by the thread heartbeat. No worker
starts until the external Chinese seed-0 run completes and exits. Existing
results are referenced in place; partial or incompatible runs are never replaced.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from braindecoding.config import PROJECT_ROOT, project_path, load_yaml_with_extends
from braindecoding.experiment import file_sha256
from braindecoding.pallier_bge_trial import save_json
from braindecoding.reporting import runtime_cache_dir, REPORT_PROTOCOLS_ROOT
from braindecoding.training.runtime import load_checkpoint
from braindecoding import unified_rolling_v2_training as worker

RUNTIME = runtime_cache_dir("unified_rolling_v2")
ENTRY = PROJECT_ROOT / "scripts/replications/run_unified_rolling_v2.py"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    return value.replace("\\", "/") if isinstance(value, str) else value


def scientific_rules(cfg, stage):
    """Ignore run locations, but retain all effective scientific settings."""
    tc = cfg["training"]
    keys = ("seed", "batch_size", "epochs", "patience", "learning_rate", "weight_decay",
            "minimum_learning_rate", "max_grad_norm", "amp", "top_ks")
    training = {key: tc[key] for key in keys}
    training["amp_dtype"] = tc.get("amp_dtype", "float16")
    training["preserve_complete_groups"] = tc.get("preserve_complete_groups", False)
    if stage == "revision":
        training["revision_cutoff_policy"] = tc.get("revision_cutoff_policy", "sample_one")
        training["freeze_brain_encoder_epochs"] = tc.get("freeze_brain_encoder_epochs", 0)
    normalization = tc.get("loss_normalization", "per_batch_independent_horizon_mean")
    if normalization != "per_batch_independent_horizon_mean":
        raise ValueError(f"Unsupported loss normalization for reuse: {normalization}")
    return portable({
        "dataset": cfg["dataset"], "model": cfg["model"], "loss": cfg["loss"],
        "text_embedding": cfg["text_embedding"], "training": training,
        "selection_metric": cfg["evaluation"]["selection_metric"],
        "cache": {k: cfg["cache"][k] for k in ("event_table", "text_embeddings")},
    })


def hash_from_manifest(manifest, relative):
    relative = str(relative).replace("\\", "/")
    for path, digest in manifest.get("source_sha256", {}).items():
        if project_path(path).resolve() == project_path(relative).resolve():
            return digest
    return None


def verify_completed(path, dataset, seed, stage, *, word_checkpoint=None, deep=True):
    path = project_path(path)
    required = ("run_manifest.json", "training_summary.json", "resolved_config.yaml", "best.pt")
    if any(not (path / name).is_file() for name in required):
        raise ValueError(f"Incomplete run; required artifacts missing: {path}")
    manifest, summary = read_json(path / required[0]), read_json(path / required[1])
    if manifest.get("status") != "completed" or summary.get("status") != "completed":
        raise ValueError(f"Run is not completed: {path}")
    actual = load_yaml_with_extends(path / "resolved_config.yaml")
    expected = worker.build_config(dataset, seed, stage, word_checkpoint=word_checkpoint)
    if (actual["experiment"]["dataset"] != dataset or actual["training"]["seed"] != seed
            or manifest.get("stage") != stage or manifest.get("seed") != seed):
        raise ValueError(f"Run identity mismatch: {path}")
    if portable(actual) != portable(manifest["config"]):
        raise ValueError(f"Resolved config differs from recorded manifest: {path}")
    if scientific_rules(actual, stage) != scientific_rules(expected, stage):
        raise ValueError(f"Completed run uses different scientific rules: {path}")
    if manifest.get("test_evaluated") is not False or summary.get("test_evaluated") is not False:
        raise ValueError(f"Missing explicit train/val-only provenance: {path}")
    for name, digest in manifest.get("source_sha256", {}).items():
        if digest is None or file_sha256(project_path(name)) != digest:
            raise ValueError(f"Recorded source/input changed: {name}")
    event = Path(actual["cache"]["event_table"])
    event_relative = event.relative_to(PROJECT_ROOT).as_posix()
    event_digest = manifest.get("event_table_sha256") or hash_from_manifest(manifest, event_relative)
    if not event_digest or event_digest != file_sha256(event):
        raise ValueError(f"Event table hash mismatch: {path}")
    embedding_digest = file_sha256(actual["cache"]["text_embeddings"])
    if not embedding_digest or manifest["embedding_sha256"] != embedding_digest:
        raise ValueError(f"BGE embedding hash mismatch: {path}")
    best = summary["best"]["legacy50"]
    history = summary["history"]
    if not history or best["epoch"] not in [row["epoch"] for row in history]:
        raise ValueError(f"Main checkpoint not supported by completed epoch history: {path}")
    if stage == "revision":
        if word_checkpoint is None:
            raise ValueError("Revision reuse requires the verified same-seed word checkpoint.")
        init = manifest["initialization"]
        if (Path(init["path"]).resolve() != Path(word_checkpoint).resolve()
                or init["sha256"] != file_sha256(word_checkpoint)):
            raise ValueError(f"Revision warm-start provenance mismatch: {path}")
    checkpoint_digest = file_sha256(path / "best.pt")
    if deep:
        cp = load_checkpoint(path / "best.pt")
        if (cp.get("stage"), cp.get("seed"), cp.get("epoch"), cp.get("selection")) != (
                stage, seed, best["epoch"], "legacy50"):
            raise ValueError(f"Main checkpoint identity/selection mismatch: {path}")
        if (portable(cp["config"]) != portable(actual) or cp["model_config"] != actual["model"]
                or cp["text_embedding_config"] != actual["text_embedding"]
                or cp["embedding_sha256"] != embedding_digest or cp.get("test_evaluated") is not False):
            raise ValueError(f"Checkpoint scientific contract mismatch: {path}")
        if not cp.get("model_state") or not cp.get("channel_names"):
            raise ValueError(f"Incomplete checkpoint payload: {path}")
        metric = actual["evaluation"]["selection_metric"]
        if cp["metrics"]["legacy"][metric] != best["validation"]["legacy"][metric]:
            raise ValueError(f"Checkpoint score differs from selected history row: {path}")
        del cp
        gc.collect()
    return {
        "dataset": dataset, "seed": seed, "stage": stage, "path": str(path),
        "checkpoint": str(path / "best.pt"), "checkpoint_sha256": checkpoint_digest,
        "best_epoch": best["epoch"], "completed_epochs": len(history),
        "verification": "config_inputs_manifest_and_checkpoint" if deep else "config_inputs_and_manifest",
        "test_evaluated": False,
    }


def process_matches(pid, *, created=None, module=None):
    if not pid:
        return False
    try:
        process = psutil.Process(int(pid))
        if created is not None and abs(process.create_time() - created) > 1:
            return False
        if module is not None and module not in " ".join(process.cmdline()):
            return False
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied as error:
        raise RuntimeError(f"Cannot verify process {pid}; refusing a concurrent launch.") from error


def external_ready(cfg):
    rule = cfg["execution"]["external_wait"]
    state = read_json(project_path(rule["runtime_status"]))
    if state.get("dataset") != rule["dataset"] or state.get("seed") != rule["seed"]:
        raise ValueError("External training identity changed.")
    if state.get("status") in ("failed", "paused", "paused_by_user", "stopped", "terminated"):
        raise RuntimeError(f"External training did not finish successfully: {state.get('status')}")
    active = process_matches(state.get("pid"), module=rule["expected_module"])
    complete = (state.get("status") == "completed"
                and set(rule["required_stages"]).issubset(state.get("completed_stages", [])))
    if not complete:
        if not active:
            raise RuntimeError("External training has no live process and no successful completion.")
        return False
    return not active


def plan(*, deep=False):
    cfg = worker.config()
    worker.assert_ready(cfg)
    reuse = {(r["dataset"], r["seed"], r["stage"]): project_path(r["path"])
             for r in cfg["execution"]["reuse_runs"]}
    rows, words = [], {}
    for seed in cfg["protocol"]["seeds"]:
        for dataset in cfg["adapters"]:
            for stage in ("word", "revision"):
                key = (dataset, seed, stage)
                path = reuse.get(key, worker.output_dir(dataset, seed, stage))
                word = words.get((dataset, seed))
                row = {"dataset": dataset, "seed": seed, "stage": stage, "path": str(path)}
                if path.exists():
                    complete = ((path / "training_summary.json").is_file()
                                and read_json(path / "run_manifest.json").get("status") == "completed")
                    if not complete:
                        wait = cfg["execution"]["external_wait"]
                        if key == (wait["dataset"], wait["seed"], "revision") and key in reuse:
                            row["action"] = "wait_for_current_run"
                        else:
                            raise ValueError(f"Incomplete existing run requires recovery; refusing overwrite: {path}")
                    else:
                        row.update(verify_completed(path, dataset, seed, stage, word_checkpoint=word, deep=deep))
                        row["action"] = "reuse_completed"
                elif key in reuse:
                    raise FileNotFoundError(f"Declared reusable run is missing: {path}")
                else:
                    row["action"] = "train_new"
                if stage == "word":
                    words[(dataset, seed)] = path / "best.pt"
                rows.append(row)
    if len(rows) != 12:
        raise ValueError("The authorized queue must contain exactly 12 dataset/seed/stage entries.")
    return {"protocol": cfg["protocol"]["id"], "total_jobs": len(rows), "jobs": rows,
            "reusable_completed": sum(r["action"] == "reuse_completed" for r in rows),
            "waiting_current": sum(r["action"] == "wait_for_current_run" for r in rows),
            "new_training": sum(r["action"] == "train_new" for r in rows), "test_evaluated": False}


def freeze_sources():
    cfg = worker.config()
    paths = {Path(__file__), worker.CONFIG_PATH, ENTRY,
             PROJECT_ROOT / "braindecoding/training/unified_rolling_v2_training.py"}
    # These are the already-verified scientific dependencies of accepted runs.
    for item in cfg["execution"]["reuse_runs"]:
        manifest = read_json(project_path(item["path"]) / "run_manifest.json")
        paths.update(project_path(p) for p in manifest["source_sha256"])
    # Freeze the new worker's full dependency set too. Future word checkpoints
    # are validated at the stage boundary, after they have been generated.
    for dataset, adapter in cfg["adapters"].items():
        for stage in ("word", "revision"):
            resolved = worker.build_config(dataset, 1, stage)
            resolved["training"].pop("pretrained_brain_encoder_checkpoint", None)
            paths.update(Path(p) for p in worker.source_hashes(resolved, adapter))
    paths.add(PROJECT_ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md")
    result = {p.relative_to(PROJECT_ROOT).as_posix(): file_sha256(p) for p in paths}
    if any(digest is None for digest in result.values()):
        raise FileNotFoundError("A source needed to freeze this queue is missing.")
    return result


def check_armed():
    armed = read_json(RUNTIME / "armed.json")
    if armed["authorization"] != "start_remaining_jobs_after_chinese_seed0_completes":
        raise ValueError("Queue is not armed for this request.")
    for path, expected in armed["source_sha256"].items():
        if str(path).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        if file_sha256(PROJECT_ROOT / path) != expected:
            raise RuntimeError(f"Armed queue source changed: {path}")
    return armed


@contextlib.contextmanager
def lock(name):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    path = RUNTIME / name
    current = {"pid": os.getpid(), "created": psutil.Process().create_time()}
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        previous = read_json(path)
        if process_matches(previous["pid"], created=previous["created"]):
            raise RuntimeError(f"Queue operation already running: {name}")
        # Never remove a stale lock automatically: two contenders could remove
        # each other's newly acquired lock. Recovery requires inspection.
        raise RuntimeError(f"Stale queue lock requires inspection: {path}")
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(current, stream)
    try:
        yield
    finally:
        if path.exists() and read_json(path) == current:
            path.unlink()


def arm():
    with lock("launch.lock"):
        if (RUNTIME / "launch.json").exists():
            raise RuntimeError("A queue launch is already recorded; inspect it before rearming.")
        current = plan(deep=True)
        record = {"authorization": "start_remaining_jobs_after_chinese_seed0_completes",
                  "armed_unix": time.time(), "source_sha256": freeze_sources(), "plan": current}
        save_json(RUNTIME / "armed.json", record)
        save_json(RUNTIME / "plan.json", current)
        save_json(RUNTIME / "status.json", {"status": "armed_waiting_for_chinese_seed0",
                  "total_jobs": 12, "reused_completed": current["reusable_completed"],
                  "new_training": current["new_training"], "test_evaluated": False,
                  "updated_unix": time.time()})
        save_json(REPORT_PROTOCOLS_ROOT / "unified_rolling_v2_queue_plan.json", current)
        return {"status": "armed", "reused_completed": current["reusable_completed"],
                "waiting_current": current["waiting_current"], "new_training": current["new_training"]}


def launch_if_ready():
    with lock("launch.lock"):
        check_armed()
        launched = RUNTIME / "launch.json"
        if launched.exists():
            info = read_json(launched)
            state = read_json(RUNTIME / "status.json")
            if state.get("status") == "completed":
                return {"status": "already_completed"}
            if process_matches(info.get("pid"), created=info.get("created")):
                return {"status": "already_running", "pid": info["pid"]}
            # A durable launch intent survives failure to persist the child PID.
            queue_lock = RUNTIME / "queue.lock"
            if queue_lock.exists():
                owner = read_json(queue_lock)
                if process_matches(owner["pid"], created=owner["created"]):
                    return {"status": "already_running", "pid": owner["pid"]}
            raise RuntimeError("The queue was already launched and stopped; no automatic restart.")
        queue_lock = RUNTIME / "queue.lock"
        if queue_lock.exists():
            owner = read_json(queue_lock)
            if process_matches(owner["pid"], created=owner["created"]):
                return {"status": "already_running", "pid": owner["pid"]}
        if not external_ready(worker.config()):
            return {"status": "waiting_for_chinese_seed0"}
        current = plan(deep=True)
        if current["waiting_current"]:
            raise RuntimeError("External completion is not backed by completed run artifacts.")
        # Record intent before spawning: an uncertain launch must never auto-retry.
        save_json(launched, {"status": "launch_requested", "launched_unix": time.time()})
        with (RUNTIME / "stdout.log").open("a", encoding="utf-8") as out, (RUNTIME / "stderr.log").open("a", encoding="utf-8") as err:
            process = subprocess.Popen([sys.executable, "-u", str(ENTRY), "--run"],
                cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            created = psutil.Process(process.pid).create_time()
        except psutil.NoSuchProcess as error:
            raise RuntimeError("Queue exited immediately after launch; inspect stderr.log.") from error
        info = {"status": "launched", "pid": process.pid, "created": created,
                "launched_unix": time.time(), "command": [sys.executable, "-u", str(ENTRY), "--run"]}
        save_json(launched, info)
        return {"status": "launched", "pid": process.pid, "new_training": current["new_training"],
                "reused_completed": current["reusable_completed"]}


def run():
    with lock("queue.lock"):
        check_armed()
        if not external_ready(worker.config()):
            raise RuntimeError("Chinese seed-0 training must finish and exit first.")
        initial = plan(deep=True)
        state = {"status": "running", "pid": os.getpid(), "started_unix": time.time(),
                 "total_jobs": 12, "completed": [], "current": None, "test_evaluated": False}
        child = None
        words = {}
        try:
            for row in initial["jobs"]:
                check_armed()
                dataset, seed, stage = row["dataset"], row["seed"], row["stage"]
                word = words.get((dataset, seed))
                state["current"] = row
                state["updated_unix"] = time.time()
                save_json(RUNTIME / "status.json", state)
                if row["action"] == "train_new":
                    args = [sys.executable, "-u", "-m", "braindecoding.unified_rolling_v2_training",
                            "--dataset", dataset, "--seed", str(seed), "--stage", stage]
                    if stage == "revision":
                        args += ["--word-checkpoint", str(word)]
                    name = f"{dataset}_seed-{seed:03d}_{stage}"
                    with (RUNTIME / f"{name}.stdout.log").open("w", encoding="utf-8") as out, (RUNTIME / f"{name}.stderr.log").open("w", encoding="utf-8") as err:
                        child = subprocess.Popen(args, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                            stdout=out, stderr=err, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                        state["child_pid"] = child.pid
                        while True:
                            state["updated_unix"] = time.time()
                            save_json(RUNTIME / "status.json", state)
                            try:
                                code = child.wait(timeout=30)
                                break
                            except subprocess.TimeoutExpired:
                                continue
                    if code:
                        raise RuntimeError(f"Worker exited with status {code}: {name}")
                    child = None
                record = verify_completed(row["path"], dataset, seed, stage, word_checkpoint=word, deep=True)
                record["action"] = "trained" if row["action"] == "train_new" else "reused"
                if stage == "word":
                    words[(dataset, seed)] = Path(record["checkpoint"])
                state["completed"].append(record)
                state["child_pid"] = None
                state["updated_unix"] = time.time()
                save_json(RUNTIME / "status.json", state)
                print(json.dumps(record, ensure_ascii=False), flush=True)
            state.update(status="completed", current=None, completed_unix=time.time())
            save_json(REPORT_PROTOCOLS_ROOT / "unified_rolling_v2_queue_results.json", state)
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
            state.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            state["updated_unix"] = time.time()
            save_json(RUNTIME / "status.json", state)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--plan", action="store_true")
    commands.add_argument("--arm", action="store_true")
    commands.add_argument("--launch-if-ready", action="store_true")
    commands.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.plan:
        result = plan(deep=True)
    elif args.arm:
        result = arm()
    elif args.launch_if_ready:
        result = launch_if_ready()
    else:
        run()
        return
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
