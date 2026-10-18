"""仓库目录结构、JSON/Markdown 治理与文件命名规范自动化守卫测试。"""

from __future__ import annotations

from pathlib import Path
import re

from braindecoding.config import PROJECT_ROOT


ALLOWED_TOP_LEVEL_DIRS = {
    "braindecoding",
    "configs",
    "data",
    "paper",
    "scripts",
    "tests",
}

FORBIDDEN_NAME_PATTERNS = (
    re.compile(r".*_v[0-9]+(?:_.*|$)"),
    re.compile(r".*_test_v[0-9]+(?:_.*|$)"),
    re.compile(r".*_(?:stable|patch|temp|tmp|new|old|copy|backup)$"),
    re.compile(r".*_io_guard$"),
)

FORBIDDEN_ABBREVIATION_TOKENS = {
    "c1",
    "f",
    "k0",
    "lm",
    "bge",
    "ctc",
    "eval",
    "fig1",
    "fig2",
    "fig3",
    "fig4",
}


def test_top_level_directories_strictly_limited_to_six() -> None:
    """顶层业务目录必须严格收敛为 6 个标准目录，防止目录膨胀。"""
    actual_dirs = {
        p.name
        for p in PROJECT_ROOT.iterdir()
        if p.is_dir() and not p.name.startswith(".") and not p.name.endswith(".egg-info")
    }
    assert actual_dirs == ALLOWED_TOP_LEVEL_DIRS, (
        f"顶层目录超出 6 个标准目录规范: 实际={sorted(actual_dirs)}, 允许={sorted(ALLOWED_TOP_LEVEL_DIRS)}"
    )


def test_markdown_files_strictly_governed() -> None:
    """全仓库仅允许保留根目录 README.md 与主实验协议契约文档，禁止到处散落 .md 笔记或报告。"""
    md_files = {
        p.relative_to(PROJECT_ROOT).as_posix()
        for p in PROJECT_ROOT.rglob("*.md")
        if not any(part.startswith(".") or part.endswith(".egg-info") for part in p.relative_to(PROJECT_ROOT).parts)
    }
    allowed_md = {
        "README.md",
        "AGENTS.md",
        "data/manifests/rolling_sentence/unified_protocol_v2.md",
    }
    assert md_files <= allowed_md and "README.md" in md_files, (
        f"检测到违规的 Markdown 文件: {sorted(md_files - allowed_md)}"
    )


def test_json_files_confined_Strictly_to_data_directory() -> None:
    """禁止在 braindecoding/、configs/、scripts/、paper/、tests/ 及根目录存放 .json 文件。"""
    json_outside_data = [
        p.relative_to(PROJECT_ROOT).as_posix()
        for p in PROJECT_ROOT.rglob("*.json")
        if not any(part.startswith(".") or part.endswith(".egg-info") for part in p.relative_to(PROJECT_ROOT).parts)
        and p.relative_to(PROJECT_ROOT).parts[0] != "data"
    ]
    assert json_outside_data == [], f"检测到 data/ 目录之外的违规 JSON 文件: {json_outside_data}"


def test_data_results_studies_contains_only_clean_structured_data() -> None:
    """data/results/studies/ 下仅允许存放结构化 .json 与 .csv 数据文件，严禁混放图片、Markdown、日志或空目录。"""
    studies_dir = PROJECT_ROOT / "data" / "results" / "studies"
    assert studies_dir.is_dir()
    for path in studies_dir.rglob("*"):
        if path.is_dir():
            assert any(path.iterdir()), f"studies 目录下不应存在空目录: {path.relative_to(PROJECT_ROOT)}"
        else:
            assert path.suffix in {".json", ".csv"}, (
                f"实验结果目录仅允许包含 .json 或 .csv 文件，发现违规文件: {path.relative_to(PROJECT_ROOT)}"
            )
            assert not path.name.endswith("sources.json"), (
                f"实验结果目录禁止包含图表溯源碎片文件: {path.relative_to(PROJECT_ROOT)}"
            )


def test_python_filenames_are_self_explanatory_without_abbreviations_or_versions() -> None:
    """所有 Python 模块、脚本与测试文件名必须使用完整英文词汇，严禁版本号后缀与内部黑话缩写。"""
    scan_roots = (
        PROJECT_ROOT / "braindecoding",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
    )
    violations: list[str] = []
    for root in scan_roots:
        for py_file in root.rglob("*.py"):
            rel = py_file.relative_to(PROJECT_ROOT).as_posix()
            stem = py_file.stem
            if stem == "__init__":
                continue
            if stem != stem.lower():
                violations.append(f"{rel}: 文件名必须全小写 snake_case")
            for pattern in FORBIDDEN_NAME_PATTERNS:
                if pattern.match(stem):
                    violations.append(f"{rel}: 禁止包含版本号或临时补丁后缀 ({pattern.pattern})")
            tokens = set(stem.split("_"))
            bad_tokens = tokens & FORBIDDEN_ABBREVIATION_TOKENS
            if bad_tokens:
                violations.append(f"{rel}: 禁止使用不清晰的缩写词 {sorted(bad_tokens)}")
    assert violations == [], "Python 文件命名规范检查未通过:\n" + "\n".join(violations)
