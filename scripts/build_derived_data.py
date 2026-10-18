"""统一派生数据缓存构建入口（Pallier2025 与 ChineseEEG-2）。"""
from __future__ import annotations

from braindecoding.cli import main

if __name__ == "__main__":
    import sys
    raise SystemExit(main(["prepare-data", *sys.argv[1:]]))
