import ast
import importlib
from pathlib import Path

import pytest


from braindecoding.config import PROJECT_ROOT
MODULES = (
    ("chineseeeg2_dataset", "ChineseEEG2LittlePrinceWordDataset"),
    ("pallier2025_dataset", "Pallier2025WordDataset"),
)


@pytest.mark.parametrize(("module_name", "class_name"), MODULES)
def test_official_dataset_modules_live_in_braindecoding(module_name, class_name):
    module = importlib.import_module(f"braindecoding.data.{module_name}")
    if class_name is not None:
        assert getattr(module, class_name).__module__ == (
            f"braindecoding.data.{module_name}"
        )


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
