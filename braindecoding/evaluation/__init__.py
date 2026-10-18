"""braindecoding.evaluation 子包。"""
from __future__ import annotations

import importlib
import sys


def __getattr__(name: str):
    from braindecoding import _MERGED_MODULE_ALIASES, _SUBPACKAGE_MAP
    if name in _SUBPACKAGE_MAP:
        target_sub = _SUBPACKAGE_MAP[name]
        mod = importlib.import_module(f"braindecoding.{target_sub}.{name}")
        globals()[name] = mod
        return mod
    if name in _MERGED_MODULE_ALIASES:
        target_sub, target_stem = _MERGED_MODULE_ALIASES[name]
        canonical = f"braindecoding.{target_sub}.{target_stem}" if target_sub else f"braindecoding.{target_stem}"
        mod = importlib.import_module(canonical)
        sys.modules[f"braindecoding.evaluation.{name}"] = mod
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'braindecoding.evaluation' has no attribute {name!r}")
