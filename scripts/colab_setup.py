"""防崩版 Google Colab 环境初始化与软链接桥接脚本。

核心特性：
1. 规避 Colab Python 3.13+ 缺少 ``ensurepip`` 报错：
   绝不直接调用 ``venv.EnvBuilder(with_pip=True)``；优先使用
   ``virtualenv --system-site-packages /content/proj-venv``，若系统未装
   ``virtualenv`` 则先通过系统 ``pip`` 安装，并提供 ``with_pip=False`` 兜底。
2. 保护云端原生 PyTorch / CUDA：
   读取当前系统环境的 ``torch`` / ``torchvision`` / ``torchaudio`` / ``triton``
   版本并写入 ``/content/proj-venv/runtime-constraints.txt``，安装依赖时强制携带
   ``-c /content/proj-venv/runtime-constraints.txt``，最后执行
   ``pip install -e . --no-deps``。
3. 解决 Google Drive FUSE 不支持软链接 (``Errno 95``) 与缓存哈希一致性：
   绝不在 ``/content/drive/...`` 内部调用 ``os.symlink``；统一在 Colab 本地盘
   ``/content/studies_links`` 下创建指向 Google Drive 原始数据目录的软链接桥。
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import venv
from pathlib import Path


PROTECTED_RUNTIME_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "triton",
)

PROJECT_DEPENDENCIES = (
    "numpy>=1.26",
    "pandas>=2.2",
    "pyarrow>=16.0",
    "scipy>=1.13",
    "scikit-learn>=1.5",
    "PyYAML>=6.0",
    "openpyxl>=3.1",
    "jieba==0.42.1",
    "transformers>=4.44",
    "huggingface-hub>=0.24",
    "safetensors>=0.4",
    "sentencepiece>=0.2",
    "mne>=1.7",
    "psutil>=5.9",
    "matplotlib>=3.8",
)


def _run_streaming(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    """流式无缓冲执行子进程命令，实时打印进度与报错堆栈。"""
    print(f"[colab_setup] $ {' '.join(str(part) for part in cmd)}", flush=True)
    process = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"命令执行失败（退出码 {return_code}）：{' '.join(map(str, cmd))}")


def venv_python_path(venv_dir: str | Path) -> Path:
    venv_path = Path(venv_dir)
    if os.name == "nt":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def export_runtime_constraints(
    venv_dir: str | Path,
    packages: tuple[str, ...] = PROTECTED_RUNTIME_PACKAGES,
) -> Path:
    """导出当前系统的 PyTorch/CUDA 相关包版本到 runtime-constraints.txt。"""
    venv_path = Path(venv_dir)
    venv_path.mkdir(parents=True, exist_ok=True)
    constraints_path = venv_path / "runtime-constraints.txt"
    lines: list[str] = []
    for pkg in packages:
        try:
            version = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
        lines.append(f"{pkg}=={version}")
    constraints_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(
        f"[colab_setup] 已导出运行时版本约束 -> {constraints_path}: {lines}",
        flush=True,
    )
    return constraints_path


def create_colab_venv(
    venv_dir: str | Path = "/content/proj-venv",
    *,
    system_python: str | Path = sys.executable,
) -> Path:
    """创建复用系统 site-packages 的虚拟环境，规避 Python 3.13+ ensurepip 缺失。"""
    venv_path = Path(venv_dir).resolve()
    target_py = venv_python_path(venv_path)
    if target_py.is_file():
        print(f"[colab_setup] 复用已存在的虚拟环境：{target_py}", flush=True)
        return target_py

    sys_py = str(system_python)
    has_virtualenv = (
        subprocess.call(
            [sys_py, "-m", "virtualenv", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        == 0
    )
    if not has_virtualenv:
        print("[colab_setup] 正在通过系统 pip 安装 virtualenv ...", flush=True)
        install_rc = subprocess.call([sys_py, "-m", "pip", "install", "-q", "virtualenv"])
        has_virtualenv = install_rc == 0

    if has_virtualenv:
        _run_streaming(
            [
                sys_py,
                "-m",
                "virtualenv",
                "--system-site-packages",
                str(venv_path),
            ]
        )
    else:
        print(
            "[colab_setup] 回退至 venv.EnvBuilder(system_site_packages=True, with_pip=False) ...",
            flush=True,
        )
        builder = venv.EnvBuilder(
            system_site_packages=True,
            clear=False,
            symlinks=(os.name != "nt"),
            with_pip=False,
        )
        builder.create(str(venv_path))

    if not target_py.is_file():
        raise FileNotFoundError(f"虚拟环境 Python 可执行文件未生成：{target_py}")
    return target_py


def install_project_dependencies(
    venv_dir: str | Path = "/content/proj-venv",
    project_root: str | Path | None = None,
    *,
    constraints_path: str | Path | None = None,
) -> dict:
    """在保护原生 PyTorch/CUDA 约束下安装项目依赖与 editable 包。"""
    repo_root = (
        Path(project_root).resolve()
        if project_root
        else Path(__file__).resolve().parents[1]
    )
    venv_path = Path(venv_dir).resolve()
    target_py = venv_python_path(venv_path)
    if not target_py.is_file():
        target_py = create_colab_venv(venv_path)

    c_path = (
        Path(constraints_path).resolve()
        if constraints_path
        else export_runtime_constraints(venv_path)
    )

    pip_Prefix = [str(target_py), "-u", "-m", "pip", "install"]
    constraint_args = ["-c", str(c_path)] if c_path.is_file() and c_path.stat().st_size > 0 else []

    _run_streaming(
        [*pip_Prefix, *constraint_args, *PROJECT_DEPENDENCIES],
        cwd=repo_root,
    )
    _run_streaming(
        [*pip_Prefix, "-e", ".", "--no-deps"],
        cwd=repo_root,
    )
    return {
        "venv_dir": str(venv_path),
        "venv_python": str(target_py),
        "constraints_file": str(c_path),
        "project_root": str(repo_root),
    }


def setup_studies_links(
    drive_data_dir: str | Path,
    links_dir: str | Path = "/content/studies_links",
) -> dict:
    """在本地高速盘创建指向云盘原始数据子目录的软链接桥，绝不在 Drive FUSE 内建软链接。"""
    source_root = Path(drive_data_dir).resolve()
    links_root = Path(links_dir)
    normalized_links = str(links_dir).replace("\\", "/")
    if normalized_links.startswith("/content/drive") or "/content/drive/" in normalized_links:
        raise ValueError(
            f"严禁在 Google Drive FUSE 挂载路径内创建软链接：{links_root}（会触发 Errno 95）。"
            "请将 links_dir 设为本地盘路径（如 /content/studies_links）。"
        )
    if not source_root.is_dir():
        raise FileNotFoundError(f"云盘原始数据目录不存在：{source_root}")

    links_root.mkdir(parents=True, exist_ok=True)
    created_links: list[dict] = []
    for child in sorted(source_root.iterdir()):
        link_path = links_root / child.name
        if link_path.is_symlink() or link_path.exists():
            if link_path.is_symlink():
                link_path.unlink()
            else:
                continue
        os.symlink(child, link_path, target_is_directory=child.is_dir())
        created_links.append({"name": child.name, "target": str(child), "link": str(link_path)})

    summary = {
        "drive_data_dir": str(source_root),
        "links_dir": str(links_root),
        "linked_entries": created_links,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="防崩版 Google Colab 环境初始化与软链接桥接工具"
    )
    parser.add_argument(
        "--venv-dir",
        default="/content/proj-venv",
        help="虚拟环境目录（默认：/content/proj-venv）",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="项目代码根目录（默认自动推导）",
    )
    parser.add_argument(
        "--drive-data-dir",
        default=None,
        help="可选：Google Drive 原始数据目录，用于创建 /content/studies_links 软链接桥",
    )
    parser.add_argument(
        "--links-dir",
        default="/content/studies_links",
        help="本地软链接桥目录（默认：/content/studies_links）",
    )
    parser.add_argument(
        "--local-cache-dir",
        default="/content/local_cache",
        help="本地高速缓存目录（默认：/content/local_cache）",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="仅创建虚拟环境与导出 constraints，跳过 pip 依赖安装",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    Path(args.local_cache_dir).mkdir(parents=True, exist_ok=True)
    create_colab_venv(args.venv_dir)
    c_path = export_runtime_constraints(args.venv_dir)
    if not args.skip_install:
        install_project_dependencies(
            args.venv_dir,
            args.project_root,
            constraints_path=c_path,
        )
    if args.drive_data_dir:
        setup_studies_links(args.drive_data_dir, args.links_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
