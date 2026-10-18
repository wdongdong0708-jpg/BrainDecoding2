import json
from pathlib import Path

import pytest

from braindecoding import single_revision_c1_test_v2_io_guard as guard


pytestmark = pytest.mark.current_protocol


def locked_error():
    error = PermissionError("Simulated Windows file lock")
    error.winerror = 5
    return error


def test_atomic_replace_retries_transient_lock_and_keeps_same_paths(monkeypatch, tmp_path):
    source, target = tmp_path / "result.json.tmp", tmp_path / "result.json"
    calls = []
    def replace(path, destination):
        calls.append((path, destination))
        if len(calls) < 3:
            raise locked_error()
        return destination
    monkeypatch.setattr(guard, "ORIGINAL_REPLACE", replace)
    monkeypatch.setattr(guard.time, "sleep", lambda _: None)
    assert guard.atomic_replace_with_retry(source, target) == target
    assert calls == [(source, target)]*3


def test_non_temporary_rename_is_not_retried(monkeypatch, tmp_path):
    calls = []
    def replace(*args):
        calls.append(args)
        raise locked_error()
    monkeypatch.setattr(guard, "ORIGINAL_REPLACE", replace)
    with pytest.raises(PermissionError):
        guard.atomic_replace_with_retry(tmp_path / "original.json", tmp_path / "result.json")
    assert len(calls) == 1


def configure_status(monkeypatch, tmp_path):
    runtime, report = tmp_path / "runtime", tmp_path / "report"
    runtime.mkdir()
    monkeypatch.setattr(guard.stable, "RUNTIME", runtime)
    monkeypatch.setattr(guard.stable, "REPORT", report)
    monkeypatch.setattr(guard, "LAST_PROGRESS", dict(time=0.0, state=None, dataset=None))
    monkeypatch.setattr(guard, "IO_MANIFEST_SHA", "frozen-io")
    return runtime, report, dict(manifest_sha256="frozen-science", total_segment_seed_units=8244,
                                 adopted_segment_seed_units=4232)


def test_locked_progress_page_does_not_discard_saved_state(monkeypatch, tmp_path, capsys):
    runtime, report, manifest = configure_status(monkeypatch, tmp_path)
    original = Path.replace
    def replace(path, target):
        if path.name == "progress.md.tmp":
            raise locked_error()
        return original(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    guard.status(manifest, state="running", stage="chunk_complete", error=None,
                 dataset="pallier2025", completed_segment_seed_units=4248)
    state = json.loads((runtime / "status.json").read_text())
    assert state["state"] == "running" and state["completed_segment_seed_units"] == 4248
    assert state["execution_io_manifest_sha256"] == "frozen-io"
    assert "optional_progress_page_write" in capsys.readouterr().err
    assert (runtime / "progress_warnings.jsonl").exists()


def test_failed_experiment_state_write_still_aborts(monkeypatch, tmp_path):
    _, _, manifest = configure_status(monkeypatch, tmp_path)
    def replace(*_):
        raise locked_error()
    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(PermissionError):
        guard.status(manifest, state="running", completed_segment_seed_units=4248)


def test_progress_is_throttled_but_saved_state_always_advances(monkeypatch, tmp_path):
    runtime, report, manifest = configure_status(monkeypatch, tmp_path)
    monkeypatch.setattr(guard.time, "time", lambda: 100.0)
    guard.status(manifest, state="running", dataset="pallier2025", completed_segment_seed_units=4232)
    first = (report / "progress.md").read_text(encoding="utf-8")
    guard.status(manifest, state="running", dataset="pallier2025", completed_segment_seed_units=4248)
    assert (report / "progress.md").read_text(encoding="utf-8") == first
    assert json.loads((runtime / "status.json").read_text())["completed_segment_seed_units"] == 4248
    monkeypatch.setattr(guard.time, "time", lambda: 106.0)
    guard.status(manifest, state="running", dataset="pallier2025", completed_segment_seed_units=4264)
    assert "4,264" in (report / "progress.md").read_text(encoding="utf-8")
