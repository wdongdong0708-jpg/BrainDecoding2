"""统一两阶段（词级预训练 + 局部窗口修订 Transformer）训练入口。"""
from __future__ import annotations

from braindecoding.training.rolling import main

if __name__ == "__main__":
    raise SystemExit(main())
