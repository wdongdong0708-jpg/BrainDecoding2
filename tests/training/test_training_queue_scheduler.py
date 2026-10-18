"""Queue safety contracts with temporary artifacts and no GPU processes."""
import copy
import json
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import yaml

from braindecoding import unified_rolling_v2_queue as queue

pytestmark = pytest.mark.current_protocol


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def external(tmp_path, monkeypatch):
    path = tmp_path / "external.json"
    cfg = {"execution": {"external_wait": {"runtime_status": str(path),
        "dataset": "chineseeeg2_littleprince", "seed": 0,
        "expected_module": "training", "required_stages": ["word", "revision"]}}}
    state = {"dataset": "chineseeeg2_littleprince", "seed": 0,
             "pid": 42, "status": "training", "completed_stages": ["word"]}
    return path, cfg, state


@pytest.mark.parametrize("completed,alive,ready", [(False, True, False), (True, True, False), (True, False, True)])
def test_waits_for_both_completion_and_process_exit(external, monkeypatch, completed, alive, ready):
    path, cfg, state = external
    if completed:
        state.update(status="completed", completed_stages=["word", "revision"])
    write(path, state)
    monkeypatch.setattr(queue, "process_matches", lambda *a, **k: alive)
    assert queue.external_ready(cfg) is ready


@pytest.mark.parametrize("status", ["failed", "paused", "training"])
def test_failed_or_abandoned_external_run_blocks_queue(external, monkeypatch, status):
    path, cfg, state = external
    state["status"] = status
    write(path, state)
    monkeypatch.setattr(queue, "process_matches", lambda *a, **k: False)
    with pytest.raises(RuntimeError):
        queue.external_ready(cfg)


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "RUNTIME", tmp_path)
    monkeypatch.setattr(queue, "lock", lambda name: nullcontext())
    monkeypatch.setattr(queue, "check_armed", lambda: {})
    monkeypatch.setattr(queue.worker, "config", lambda: {})
    monkeypatch.setattr(queue, "external_ready", lambda cfg: True)
    monkeypatch.setattr(queue, "plan", lambda **kwargs: {"waiting_current": 0, "new_training": 8, "reusable_completed": 4})
    monkeypatch.setattr(queue.psutil, "Process", lambda pid: SimpleNamespace(create_time=lambda: 100.0))
    calls = []
    def popen(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(pid=123)
    monkeypatch.setattr(queue.subprocess, "Popen", popen)
    monkeypatch.setattr(queue, "process_matches", lambda pid, **kwargs: pid == 123)
    write(tmp_path / "status.json", {"status": "running"})
    return tmp_path, calls


def test_repeated_launch_starts_exactly_one_process(launcher):
    _, calls = launcher
    assert queue.launch_if_ready()["status"] == "launched"
    assert queue.launch_if_ready()["status"] == "already_running"
    assert len(calls) == 1


def test_no_launch_while_external_training_runs(launcher, monkeypatch):
    path, calls = launcher
    monkeypatch.setattr(queue, "external_ready", lambda cfg: False)
    assert queue.launch_if_ready()["status"] == "waiting_for_chinese_seed0"
    assert not calls and not (path / "launch.json").exists()


def test_no_automatic_restart_after_exit_or_uncertain_launch(launcher):
    path, calls = launcher
    write(path / "launch.json", {"status": "launch_requested"})
    with pytest.raises(RuntimeError, match="no automatic restart"):
        queue.launch_if_ready()
    assert not calls


def test_live_queue_lock_prevents_duplicate_without_launch_record(launcher):
    path, calls = launcher
    write(path / "queue.lock", {"pid": 123, "created": 100.0})
    assert queue.launch_if_ready()["status"] == "already_running"
    assert not calls


def test_intent_is_durable_before_spawning(launcher, monkeypatch):
    path, calls = launcher
    def failed_spawn(*args, **kwargs):
        assert (path / "launch.json").exists()
        raise OSError("spawn failure")
    monkeypatch.setattr(queue.subprocess, "Popen", failed_spawn)
    with pytest.raises(OSError):
        queue.launch_if_ready()
    with pytest.raises(RuntimeError, match="no automatic restart"):
        queue.launch_if_ready()


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(queue, "project_path", lambda p: Path(p) if Path(p).is_absolute() else tmp_path / p)
    event, embedding = tmp_path / "events.csv", tmp_path / "embeddings.npz"
    event.write_bytes(b"event fixture")
    embedding.write_bytes(b"embedding fixture")
    out = tmp_path / "word"
    out.mkdir()
    (out / "best.pt").write_bytes(b"checkpoint fixture")
    cfg = {"experiment": {"dataset": "fixture"}, "dataset": {"subjects": ["01"]},
           "model": {"depth": 16}, "loss": {"name": "SigLip"}, "text_embedding": {"model": "BGE"},
           "training": dict(seed=0, batch_size=128, epochs=30, patience=10, learning_rate=1e-4,
                            weight_decay=0, minimum_learning_rate=0, max_grad_norm=0, amp=True, top_ks=[1, 10]),
           "evaluation": {"selection_metric": "score"},
           "cache": {"event_table": str(event), "text_embeddings": str(embedding)}}
    manifest = {"status": "completed", "stage": "word", "seed": 0, "config": cfg,
                "source_sha256": {str(event): queue.file_sha256(event)},
                "embedding_sha256": queue.file_sha256(embedding), "test_evaluated": False}
    row = {"epoch": 2, "validation": {"legacy": {"score": 0.3}}}
    summary = {"status": "completed", "history": [row], "best": {"legacy50": row}, "test_evaluated": False}
    checkpoint = {"stage": "word", "seed": 0, "epoch": 2, "selection": "legacy50", "config": cfg,
                  "model_config": cfg["model"], "text_embedding_config": cfg["text_embedding"],
                  "embedding_sha256": manifest["embedding_sha256"], "test_evaluated": False,
                  "model_state": {"fixture": 1}, "channel_names": ["a"], "metrics": row["validation"]}
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    write(out / "run_manifest.json", manifest)
    write(out / "training_summary.json", summary)
    monkeypatch.setattr(queue.worker, "build_config", lambda *a, **k: copy.deepcopy(cfg))
    monkeypatch.setattr(queue, "load_checkpoint", lambda path: checkpoint)
    return out, cfg, manifest, checkpoint


def test_completed_compatible_checkpoint_reused(artifacts):
    out, _, _, _ = artifacts
    result = queue.verify_completed(out, "fixture", 0, "word")
    assert result["best_epoch"] == 2
    assert result["verification"] == "config_inputs_manifest_and_checkpoint"


@pytest.mark.parametrize("change", ["missing_summary", "running_manifest", "wrong_seed", "wrong_embedding", "wrong_capacity", "wrong_score"])
def test_incomplete_or_incompatible_result_is_not_reused(artifacts, change):
    out, cfg, manifest, checkpoint = artifacts
    if change == "missing_summary":
        (out / "training_summary.json").unlink()
    elif change == "running_manifest":
        manifest["status"] = "running"
        write(out / "run_manifest.json", manifest)
    elif change == "wrong_seed":
        checkpoint["seed"] = 1
    elif change == "wrong_embedding":
        checkpoint["embedding_sha256"] = "different"
    elif change == "wrong_capacity":
        bad_cfg = copy.deepcopy(cfg)
        bad_cfg["model"]["depth"] = 4
        manifest["config"] = bad_cfg
        write(out / "run_manifest.json", manifest)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")
    else:
        checkpoint["metrics"] = {"legacy": {"score": 0.4}}
    with pytest.raises(ValueError):
        queue.verify_completed(out, "fixture", 0, "word")


def test_stale_lock_is_not_removed_automatically(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "RUNTIME", tmp_path)
    monkeypatch.setattr(queue, "process_matches", lambda *a, **k: False)
    write(tmp_path / "launch.lock", {"pid": 42, "created": 1.0})
    with pytest.raises(RuntimeError, match="Stale queue lock"):
        with queue.lock("launch.lock"):
            pytest.fail("Stale lock was acquired")
    assert (tmp_path / "launch.lock").exists()
