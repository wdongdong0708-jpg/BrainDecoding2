"""当前协议范围、来源导航和只读状态查询的合同。"""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from braindecoding import cli
from braindecoding.config import PROJECT_ROOT
from braindecoding.protocol_catalog import (
    discover_protocols,
    protocol_status,
    resolve_protocol,
)

pytestmark = pytest.mark.current_protocol


@pytest.fixture
def index_fixture(tmp_path):
    (tmp_path / "protocol.md").write_text("frozen protocol", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("seed: 0\n", encoding="utf-8")
    record = {
        "id": "current", "role": "primary", "title": "Current protocol",
        "datasets": ["dataset"], "protocol": "protocol.md", "config": "config.yaml",
        "entrypoints": ["python -m worker --help"], "reports": ["reports/results.md"],
        "status_sources": [{"name": "queue", "paths": [".cache/status.json", "reports/status.json"]}],
    }
    payload = {"version": 1, "primary": "current", "protocols": [record]}
    path = tmp_path / "index.yaml"
    return tmp_path, path, payload


def write_index(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_current_scope_is_one_primary_and_two_named_supplements():
    records = discover_protocols()
    assert {record["id"]: record["role"] for record in records} == {
        "unified_protocol_v2": "primary",
        "synthetic_timing_v1": "supplement",
        "synthetic_timing_v2": "supplement",
        "fixed_k0_rolling_search_test_v2": "supplement",
        "single_word_long_window_test_v2": "supplement",
        "pure_language_baseline_test_v2": "supplement",
        "f_local_sentence_test_v2": "supplement",
        "pure_language_prompt_sensitivity_test_v2": "supplement",
        "neural_prompt_sensitivity_test_v2": "supplement",
    }
    primary = resolve_protocol("unified_protocol_v2")
    assert primary["protocol"] == "data/manifests/rolling_sentence/unified_protocol_v2.md"
    assert primary["config"] == "configs/training/rolling_revision_protocol.yaml"
    assert {dataset for record in records for dataset in record["datasets"]} == {
        "pallier2025", "chineseeeg2_littleprince",
    }
    with pytest.raises(ValueError, match="相近候选"):
        resolve_protocol("synthetic_timing_v")


@pytest.mark.parametrize("mutation", ("duplicate", "parent", "missing", "outside", "primary"))
def test_invalid_index_cannot_silently_expand_or_break_scope(index_fixture, mutation):
    root, path, payload = index_fixture
    record = payload["protocols"][0]
    if mutation == "duplicate":
        payload["protocols"].append(copy.deepcopy(record))
    elif mutation == "parent":
        extra = dict(record, id="supplement", role="supplement", parent="retired")
        payload["protocols"].append(extra)
    elif mutation == "missing":
        record["protocol"] = "missing.md"
    elif mutation == "outside":
        record["status_sources"][0]["paths"] = ["../outside/status.json"]
    else:
        payload["primary"] = "retired"
    write_index(path, payload)
    with pytest.raises((ValueError, FileNotFoundError)):
        discover_protocols(path, root=root)


def test_missing_outputs_do_not_create_directories_or_imply_not_started(index_fixture):
    root, path, payload = index_fixture
    write_index(path, payload)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    record = discover_protocols(path, root=root)[0]
    state = protocol_status(record, root=root)[0]
    assert state["status"] == "not_recorded"
    assert state["path"] is None
    assert not (root / "outputs").exists()
    assert not (root / ".cache").exists()
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_runtime_state_overrides_completed_report_and_is_never_rewritten(index_fixture):
    root, _, payload = index_fixture
    runtime = root / ".cache/status.json"
    report = root / "reports/status.json"
    for path in (runtime, report):
        path.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text(json.dumps({"state": "paused", "completed_stages": 9}), encoding="utf-8")
    report.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    before = runtime.read_bytes(), report.read_bytes()
    record = payload["protocols"][0]
    state = protocol_status(record, root=root)[0]
    assert state["status"] == "paused"
    assert state["path"] == runtime
    assert state["snapshot"]["completed_stages"] == 9
    assert (runtime.read_bytes(), report.read_bytes()) == before
    runtime.unlink()
    state = protocol_status(record, root=root)[0]
    assert state["status"] == "completed"
    assert state["path"] == report


def test_corrupt_runtime_is_reported_without_falling_back_to_success(index_fixture):
    root, _, payload = index_fixture
    runtime = root / ".cache/status.json"
    report = root / "reports/status.json"
    for path in (runtime, report):
        path.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_text("{interrupted JSON", encoding="utf-8")
    report.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    state = protocol_status(payload["protocols"][0], root=root)[0]
    assert state["status"] == "unreadable"
    assert state["path"] == runtime
    assert state["error"]
    assert runtime.read_text(encoding="utf-8") == "{interrupted JSON"


@pytest.mark.parametrize("identifier", (
    "unified_protocol_v2", "synthetic_timing_v1", "fixed_k0_rolling_search_test_v2",
))
def test_current_queries_do_not_load_tasks_or_create_outputs(identifier, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_task_config", lambda record: pytest.fail("加载了训练任务"))
    monkeypatch.setattr(cli, "protocol_status", lambda record: protocol_status(record, root=tmp_path))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["show", identifier]) == 0
    assert cli.main(["status", identifier]) == 0
    assert cli.main(["report", identifier]) == 0
    assert identifier in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
    assert cli.main(["report", identifier, "--write"]) == 2
    assert list(tmp_path.iterdir()) == []


def test_current_cli_queries_do_not_import_torch_in_a_fresh_process(tmp_path):
    code = (
        "import sys; from braindecoding.cli import main; "
        "assert main(['list']) == 0; assert main(['status']) == 0; "
        "assert main(['show', 'unified_protocol_v2']) == 0; "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=tmp_path,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("path, ignored", (
    ("docs/new_guide.md", False),
    ("experiments/diagnostics/example/README.md", False),
    ("experiments/diagnostics/example/check.py", False),
    ("data/manifests/protocol_index.yaml", False),
    (".cache/runs/example/stdout.log", True),
    ("outputs/word_decoding/example/best.pt", True),
))
def test_sources_are_visible_to_git_and_runtime_products_remain_ignored(path, ignored):
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", path], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == (0 if ignored else 1), result.stderr
