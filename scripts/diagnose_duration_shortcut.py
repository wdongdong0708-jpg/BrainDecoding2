"""薄层 CLI 入口：转发至 braindecoding.evaluation.duration_shortcut_diagnostics。"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from braindecoding.evaluation.duration_shortcut_diagnostics import (  # noqa: E402
    build_parser,
    format_diagnostics_table,
    main,
    run_duration_shortcut_diagnostics,
)

__all__ = [
    "build_parser",
    "format_diagnostics_table",
    "main",
    "run_duration_shortcut_diagnostics",
]

if __name__ == "__main__":
    raise SystemExit(main())
