"""与具体数据集无关的训练运行时工具。"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    """同步设置 Python、NumPy 和 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name):
    """按现有设备策略选择 PyTorch 设备。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前环境不可用。")
    return device


def save_json(path, payload):
    """以 UTF-8 格式保存可复核的 JSON 结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def limit_rows(table, maximum):
    """保持原顺序截取小规模调试数据。"""
    if maximum is None or len(table) <= int(maximum):
        return table.reset_index(drop=True)
    return table.iloc[: int(maximum)].reset_index(drop=True)


def cpu_state_dict(module):
    """复制一份与原模块解耦的 CPU 状态字典。"""
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def save_checkpoint(path, payload):
    """通过临时文件原子替换保存检查点，并容忍短暂的 Windows 占用。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    delays = (0.25, 0.5, 1.0, 2.0, 4.0)
    for attempt, delay in enumerate(delays, start=1):
        try:
            os.replace(temporary_path, path)
            return
        except PermissionError:
            if attempt == len(delays):
                raise
            time.sleep(delay)


def load_checkpoint(path, map_location="cpu"):
    """使用 PyTorch 受限加载器读取仅含状态的检查点格式。"""
    return torch.load(path, map_location=map_location, weights_only=True)


def parameter_count(module):
    """返回模块的参数总数。"""
    return int(sum(parameter.numel() for parameter in module.parameters()))

# --- Merged from epoch_recovery.py ---
"""Transactional epoch checkpoints for an interrupted synthetic experiment.

Model-only best checkpoints cannot resume AdamW. A boundary commits optimizer,
schedule, AMP and random states together with the selection history. A rollback
copy keeps the last committed best model safe while the next epoch is pending.
"""

import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import psutil
import torch

from braindecoding.experiment import file_sha256

FORMAT = "synthetic_epoch_boundary_v1"
BEST_FILES = ("best.pt", "best_full_vocab.pt")


def process_matches(record):
    """A reused PID must not turn a stale lock into a live training process."""
    try:
        return abs(psutil.Process(int(record["pid"])).create_time() - float(record["created"])) < 0.01
    except (KeyError, ValueError, TypeError, psutil.NoSuchProcess):
        return False
    except psutil.AccessDenied:
        return True


def capture_random_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_random_state(state):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".copy_tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def rollback_path(out, name):
    return Path(out) / (name + ".rollback")


def protect_committed_best(out, name, committed_hashes):
    """Call before replacing a best model, preserving the committed boundary."""
    expected = committed_hashes.get(name)
    if expected is None:
        return
    source = Path(out) / name
    if not source.exists() or file_sha256(source) != expected:
        raise ValueError(f"Best checkpoint differs from the committed epoch: {source}")
    atomic_copy(source, rollback_path(out, name))


def restore_committed_best(out, hashes):
    for name, expected in hashes.items():
        destination = Path(out) / name
        if destination.exists() and file_sha256(destination) == expected:
            continue
        backup = rollback_path(out, name)
        if not backup.exists() or file_sha256(backup) != expected:
            raise ValueError(f"No valid committed best checkpoint remains: {destination}")
        atomic_copy(backup, destination)


def save_boundary(out, model, loss_module, optimizer, scheduler, scaler, *, identity,
                  state, best_hashes):
    # Saving is synchronous: live GPU tensors are safe here and avoid cloning
    # several GB of optimizer state into this machine's limited CPU memory.
    payload = {
        "format": FORMAT, "identity": identity, "state": state,
        "best_sha256": best_hashes, "random_state": capture_random_state(),
        "model": model.state_dict(), "loss": loss_module.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "saved_unix": time.time(),
    }
    save_checkpoint(Path(out) / "resume.pt", payload)
    for name in BEST_FILES:
        rollback_path(out, name).unlink(missing_ok=True)


def restore_boundary(out, model, loss_module, optimizer, scheduler, scaler, *, identity):
    payload = load_checkpoint(Path(out) / "resume.pt")
    if payload.get("format") != FORMAT or payload.get("identity") != identity:
        raise ValueError("Recovery requires a full epoch checkpoint with identical frozen inputs")
    restore_committed_best(out, payload["best_sha256"])
    model.load_state_dict(payload["model"], strict=True)
    loss_module.load_state_dict(payload["loss"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    scaler.load_state_dict(payload["scaler"])
    # Restore after module/optimizer construction, which can consume randomness.
    restore_random_state(payload["random_state"])
    return payload["state"], payload["best_sha256"]
