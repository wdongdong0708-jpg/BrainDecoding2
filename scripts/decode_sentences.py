"""序列解码协议（Stage 3 / S3）薄层命令行入口。

业务逻辑全部位于 `braindecoding/decoding/sentence_decoding_pipeline.py`。
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from braindecoding.decoding.sentence_decoding_pipeline import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    _load_or_init_language_scorer,
    build_parser,
    main,
    print_pipeline_summary,
    resolve_default_paths,
    run_sequence_decoding_pipeline,
)

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "_load_or_init_language_scorer",
    "build_parser",
    "main",
    "print_pipeline_summary",
    "resolve_default_paths",
    "run_sequence_decoding_pipeline",
]

if __name__ == "__main__":
    raise SystemExit(main())
