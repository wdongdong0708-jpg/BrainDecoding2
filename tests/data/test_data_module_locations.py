import ast
import importlib

import pytest

from braindecoding.config import PROJECT_ROOT

MODULES = (
    ("word_window_dataset", "WordWindowDataset"),
    ("pallier2025_dataset", "Pallier2025WordDataset"),
    ("pnpl_dataset_adapter", "materialize_pnpl_dataset"),
)


@pytest.mark.parametrize(("module_name", "symbol_name"), MODULES)
def test_official_dataset_modules_live_in_braindecoding(module_name, symbol_name):
    module = importlib.import_module(f"braindecoding.data.{module_name}")
    assert getattr(module, symbol_name).__module__ == (
        f"braindecoding.data.{module_name}"
    )


def test_pnpl_adapter_has_no_reverse_import_to_pallier2025_dataset():
    path = PROJECT_ROOT / "braindecoding/data/pnpl_dataset_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "pallier2025_dataset" not in (node.module or ""), path
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "pallier2025_dataset" not in alias.name, path


def test_root_datasets_directory_is_removed():
    assert not (PROJECT_ROOT / "datasets").exists()


def test_production_code_does_not_import_root_dataset_modules():
    for path in (PROJECT_ROOT / "braindecoding").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "datasets", path
                assert not (node.module or "").startswith("datasets."), path
            elif isinstance(node, ast.Import):
                assert all(
                    alias.name != "datasets"
                    and not alias.name.startswith("datasets.")
                    for alias in node.names
                ), path
