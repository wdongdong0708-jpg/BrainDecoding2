"""跨数据集复用的传感器布局工具。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def vectorview_channel_positions(channel_names, layout_path=None) -> np.ndarray:
    """从 MNE 内置布局文件中解析 Neuromag Vectorview 通道位置。"""
    if layout_path is None:
        spec = importlib.util.find_spec("mne")
        if spec is None or spec.origin is None:
            raise ImportError(
                "需要安装 mne，或在配置中提供 Vectorview-all.lout 的 layout_path。"
            )
        layout_path = (
            Path(spec.origin).parent
            / "channels"
            / "data"
            / "layouts"
            / "Vectorview-all.lout"
        )
    layout_path = Path(layout_path)
    if not layout_path.exists():
        raise FileNotFoundError(f"找不到 Vectorview layout：{layout_path}")
    positions = {}
    lines = layout_path.read_text(encoding="utf-8").splitlines()[1:]
    for line in lines:
        fields = line.split()
        if len(fields) < 7:
            continue
        name = "".join(fields[5:])
        positions[name] = (float(fields[1]), float(fields[2]))
    try:
        output = np.asarray(
            [positions[str(name).replace(" ", "")] for name in channel_names],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise KeyError(f"Vectorview layout 缺少通道 {exc.args[0]}。") from exc
    lower = output.min(axis=0, keepdims=True)
    span = output.max(axis=0, keepdims=True) - lower
    output = (output - lower) / np.maximum(span, 1e-8)
    return output.astype(np.float32)
