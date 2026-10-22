"""稳定 CLI 与最终工程边界测试。"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from braindecoding import cli
from braindecoding.catalog import (
    discover_experiment_configs,
    discover_runnable_experiment_configs,
    resolve_selector,
)
from braindecoding.config import PROJECT_ROOT


SELECTORS = {
    "chineseeeg2_littleprince/sub01-08/main_word",
    "chineseeeg2_littleprince/sub01-08/main_context_warmstart",
    "pallier2025/sub01-10/main_word",
    "pallier2025/sub01-10/main_context_warmstart",
}

DEVELOPMENT_SELECTORS = {
    "pallier2025/sub01-10/local_window_rolling_revision_k2_d2",
    "chineseeeg2_littleprince/sub01-08/local_window_rolling_revision_k2_d2",
    "chineseeeg2_littleprince/sub01-08/local_window_rolling_revision_k2_d2_full_coverage",
}


@pytest.fixture(autouse=True)
def configured_roots(monkeypatch):
    monkeypatch.setenv("BRAINDATA_ROOT", "D:/dataset")
    monkeypatch.setenv(
        "BRAINDECODING_MODEL_ROOT", "D:/code/dascoli-word-decoding/models"
    )


def test_root_legacy_business_modules_and_directories_are_absent():
    for name in (
        "datasets",
        "tasks",
        "models.py",
        "losses.py",
        "optimizers.py",
        "metrics.py",
        "run.py",
    ):
        assert not (PROJECT_ROOT / name).exists()


def test_package_production_code_has_no_legacy_root_imports():
    forbidden = {
        "datasets",
        "tasks",
        "models",
        "losses",
        "optimizers",
        "metrics",
    }
    for path in (PROJECT_ROOT / "braindecoding").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden, path
            elif isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in forbidden for alias in node.names), path


def test_default_list_contains_only_current_protocols(monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "discover_runnable_experiment_configs",
        lambda: pytest.fail("默认列表不应扫描历史词级配置"),
    )
    assert cli.main(["list"]) == 0
    output = capsys.readouterr().out
    assert {line.split()[0] for line in output.splitlines()[1:]} == {
        record["id"] for record in cli.discover_protocols()
    }
    assert "smn4lang" not in output
    assert "libribrain100" not in output
    assert "main_word" not in output


def test_explicit_legacy_list_preserves_configuration_queries(capsys):
    assert SELECTORS <= {record["selector"] for record in discover_experiment_configs()}
    records = discover_runnable_experiment_configs()
    selectors = {record["selector"] for record in records}
    assert SELECTORS | DEVELOPMENT_SELECTORS <= selectors
    assert len(selectors) == len(records)
    assert cli.main(["list", "--legacy"]) == 0
    captured = capsys.readouterr()
    assert "历史配置查询" in captured.err
    assert "chineseeeg2_littleprince" in captured.out
    assert "pallier2025" in captured.out


def test_selector_resolution_and_nearby_error():
    selector = "chineseeeg2_littleprince/sub01-08/main_context_warmstart"
    assert resolve_selector(selector)["selector"] == selector
    with pytest.raises(ValueError, match="相近候选"):
        resolve_selector("chineseeeg2_littleprince/sub01-08/main_contex")


def test_show_and_status_do_not_create_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_directory", lambda config: tmp_path / "never-created")
    selector = "pallier2025/sub01-10/main_word"
    assert cli.main(["show", selector]) == 0
    assert cli.main(["status", selector]) == 0
    assert not (tmp_path / "never-created").exists()
    assert "test: locked_not_evaluated" in capsys.readouterr().out


def test_status_marks_implementation_diagnostic_without_changing_run_manifest(
    tmp_path, monkeypatch
):
    output = tmp_path / "main_context" / "seed-000"
    output.mkdir(parents=True)
    (output / "run_manifest.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )
    (output / "implementation_diagnostic.json").write_text(
        json.dumps(
            {
                "status": "implementation_diagnostic",
                "reason": "cross_subject_context_mixing",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "run_directory", lambda config: output)
    record = resolve_selector("pallier2025/sub01-10/main_context_warmstart")
    state = cli._run_status(record)
    assert state["status"] == "implementation_diagnostic"
    assert state["manifest"] == {"status": "completed"}
    assert state["diagnostic"]["reason"] == "cross_subject_context_mixing"


def test_preflight_does_not_create_run_directory(tmp_path, monkeypatch):
    import braindecoding.preflight as preflight

    selector = "pallier2025/sub01-10/main_word"
    record = resolve_selector(selector)
    monkeypatch.setattr(
        preflight,
        "build_preflight_report",
        lambda **kwargs: {
            "experiments": [
                {
                    "config": record["relative_config"],
                    "ready": True,
                    "reasons": [],
                }
            ]
        },
    )
    monkeypatch.chdir(tmp_path)
    assert cli.main(["preflight", selector]) == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", ("run", "evaluate", "audit"))
@pytest.mark.parametrize("selector", (
    "chineseeeg2_littleprince/sub01-08/main_context_warmstart", "unified_protocol_v2",
))
def test_retired_execution_stops_before_loading_tasks_or_touching_outputs(
    command, selector, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        cli, "_load_task_config", lambda record: pytest.fail("加载了训练任务"),
    )
    monkeypatch.setattr(
        cli, "resolve_selector", lambda value: pytest.fail("旧执行入口解析了历史配置"),
    )
    sentinel = tmp_path / "best.pt"
    sentinel.write_bytes(b"existing checkpoint")
    monkeypatch.chdir(tmp_path)
    assert cli.main([command, selector]) == 2
    assert "入口已停用" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"existing checkpoint"


def test_cli_has_no_test_evaluation_option():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "evaluate",
                "pallier2025/sub01-10/main_word",
                "--split",
                "test",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "audit",
                "pallier2025/sub01-10/main_word",
                "--split",
                "test",
            ]
        )


@pytest.mark.parametrize("argv", (
    ["audit", "pallier2025/sub01-10/main_context_warmstart", "--mapping-only", "--device", "cuda"],
    ["data", "build", "smn4lang", "--all"],
    ["data", "build", "libribrain100", "--force"],
))
def test_legacy_execution_does_not_import_workers(argv, tmp_path, monkeypatch, capsys):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name not in {
            "braindecoding.audit.runner", "braindecoding.subject_generalization_runner",
            "braindecoding.data.build",
        }, "停用的入口不应导入执行器"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.chdir(tmp_path)
    assert cli.main(argv) == 2
    assert "入口已停用" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_data_check_reuses_package_builder(monkeypatch, capsys):
    import braindecoding.data.build as build

    monkeypatch.setattr(
        build,
        "check_dataset",
        lambda dataset: {"dataset": dataset, "status": "complete"},
    )
    assert cli.main(["data", "check", "chineseeeg2_littleprince"]) == 0
    assert '"status": "complete"' in capsys.readouterr().out


def test_python_module_cli_works_from_non_project_cwd(tmp_path):
    environment = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-m", "braindecoding", "list"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "chineseeeg2_littleprince" in result.stdout


def test_installed_console_script_lists_experiments_from_non_project_cwd(tmp_path):
    executable = Path(sys.executable).parent / (
        "Scripts/brain-decoding.exe" if sys.platform == "win32" else "brain-decoding"
    )
    assert executable.is_file(), "请先执行 pip install -e ."
    result = subprocess.run(
        [executable, "list"],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "unified_protocol_v2" in result.stdout
    assert "synthetic_timing_v1" in result.stdout
    assert "sequence_decoding_protocol" in result.stdout
    assert "smn4lang" not in result.stdout
