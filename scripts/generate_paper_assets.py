"""一键重生成论文正文与附录的全部 LaTeX 表格与主图。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-only", action="store_true", help="仅导出 paper/tables/*.tex")
    parser.add_argument("--figures-only", action="store_true", help="仅重绘 paper/figures/*")
    args = parser.parse_args(argv)
    steps = []
    if not args.figures_only:
        steps.append(ROOT / "scripts/paper/unify_manuscript_tables.py")
    if not args.tables_only:
        steps.append(ROOT / "scripts/paper/polish_main_assets.py")
    for script in steps:
        print(f"[generate_paper_assets] Running {script.relative_to(ROOT).as_posix()}...", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=str(ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
