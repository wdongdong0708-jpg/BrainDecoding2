"""项目配置读取与路径解析。"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRAINDATA_ROOT = "D:/dataset"
DEFAULT_MODEL_ROOT = "D:/code/dascoli-word-decoding/models"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "derived"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "data" / "cache" / "runs"
_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ENV_FALLBACKS: dict[str, tuple[str, ...]] = {
    "BRAINDATA_ROOT": ("BRAINDECODING_DATA_DIR",),
    "BRAINDECODING_DATA_DIR": ("BRAINDATA_ROOT",),
    "BRAINDECODING_MODEL_ROOT": ("BRAINDECODING_MODEL_DIR",),
    "BRAINDECODING_MODEL_DIR": ("BRAINDECODING_MODEL_ROOT",),
}


def expand_environment_variables(value):
    """递归展开配置值中的 ``${ENV_NAME}`` 环境变量。"""
    if isinstance(value, dict):
        return {
            key: expand_environment_variables(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [expand_environment_variables(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match):
        name = match.group(1)
        if name in os.environ:
            return os.environ[name]
        for fallback in _ENV_FALLBACKS.get(name, ()):
            if fallback in os.environ:
                return os.environ[fallback]
        raise ValueError(f"配置引用了未设置的环境变量：{name}")

    return _ENVIRONMENT_VARIABLE.sub(replace, value)


_LEGACY_PATH_PREFIX_MAP: tuple[tuple[str, str], ...] = (
    ("experiments/manifests/", "data/manifests/"),
    ("experiments/protocol_index.yaml", "data/manifests/protocol_index.yaml"),
    ("reports/", "data/results/"),
    ("derived/", "data/cache/derived/"),
    ("artifacts/", "data/cache/artifacts/"),
)


def normalize_project_relative(value: str) -> str:
    """将历史仓库相对路径映射为统一 data/ 下的标准相对路径。"""
    normalized = value.replace("\\", "/")
    for old_prefix, new_prefix in _LEGACY_PATH_PREFIX_MAP:
        if normalized == old_prefix.rstrip("/"):
            return new_prefix.rstrip("/")
        if normalized.startswith(old_prefix):
            return new_prefix + normalized[len(old_prefix):]
    return value


def project_path(value):
    """以项目根目录为基准解析相对路径，绝对路径保持不变。

    当设置了 ``BRAINDECODING_CACHE_DIR`` 时，``data/cache/derived`` 下的相对路径
    将自动解析到该环境变量指定的缓存根目录。
    """
    expanded_value = expand_environment_variables(str(value))
    path = Path(expanded_value)
    if path.is_absolute():
        return path
    normalized = normalize_project_relative(expanded_value)
    posix_rel = normalized.replace("\\", "/")
    cache_env = os.environ.get("BRAINDECODING_CACHE_DIR")
    if cache_env and (
        posix_rel == "data/cache/derived"
        or posix_rel.startswith("data/cache/derived/")
    ):
        suffix = posix_rel[len("data/cache/derived") :].lstrip("/")
        base = Path(cache_env)
        if not base.is_absolute():
            base = PROJECT_ROOT / base
        return base / suffix if suffix else base
    return PROJECT_ROOT / normalized


def data_root() -> Path:
    """返回原始数据集根目录（优先读取 ``BRAINDECODING_DATA_DIR``，兼容 ``BRAINDATA_ROOT``）。"""
    raw = (
        os.environ.get("BRAINDECODING_DATA_DIR")
        or os.environ.get("BRAINDATA_ROOT")
        or DEFAULT_BRAINDATA_ROOT
    )
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def cache_root() -> Path:
    """返回预处理派生特征缓存根目录（由 ``BRAINDECODING_CACHE_DIR`` 驱动）。"""
    return project_path("data/cache/derived")


def results_root() -> Path:
    """返回训练 Checkpoint 与日志输出根目录（由 ``BRAINDECODING_RESULTS_DIR`` 驱动）。"""
    raw = os.environ.get("BRAINDECODING_RESULTS_DIR")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else PROJECT_ROOT / path
    return DEFAULT_RESULTS_ROOT


def model_root() -> Path:
    """返回本地预训练语言/嵌入模型根目录。"""
    raw = (
        os.environ.get("BRAINDECODING_MODEL_DIR")
        or os.environ.get("BRAINDECODING_MODEL_ROOT")
        or DEFAULT_MODEL_ROOT
    )
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def deep_merge(base, override):
    """递归合并字典；子配置中的列表和标量直接覆盖父配置。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_mapping(config_path, ancestors=()):
    config_path = Path(config_path).resolve()
    if config_path in ancestors:
        raise ValueError(f"配置继承出现循环：{config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    parent = config.pop("extends", None)
    if parent is None:
        return config
    parent_path = Path(expand_environment_variables(str(parent)))
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent_config = _load_yaml_mapping(parent_path, (*ancestors, config_path))
    return deep_merge(parent_config, config)


def load_yaml_with_extends(path):
    """读取 YAML，递归展开 ``extends``，并展开最终配置中的环境变量。"""
    config_path = Path(expand_environment_variables(str(path))).resolve()
    config = _load_yaml_mapping(config_path)
    return expand_environment_variables(config)


def ensure_default_environment(
    *,
    data_root: str | Path = DEFAULT_BRAINDATA_ROOT,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    cache_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, str]:
    """为独立运行的协议与工作流入口设置默认数据、缓存、结果与模型根目录（若外部未显式设置）。"""
    resolved_data = str(
        os.environ.get("BRAINDECODING_DATA_DIR")
        or os.environ.get("BRAINDATA_ROOT")
        or data_root
    )
    resolved_model = str(
        os.environ.get("BRAINDECODING_MODEL_DIR")
        or os.environ.get("BRAINDECODING_MODEL_ROOT")
        or model_root
    )
    os.environ.setdefault("BRAINDECODING_DATA_DIR", resolved_data)
    os.environ.setdefault("BRAINDATA_ROOT", resolved_data)
    os.environ.setdefault("BRAINDECODING_MODEL_ROOT", resolved_model)
    if cache_dir is not None:
        os.environ.setdefault("BRAINDECODING_CACHE_DIR", str(cache_dir))
    if results_dir is not None:
        os.environ.setdefault("BRAINDECODING_RESULTS_DIR", str(results_dir))
    return {
        "BRAINDECODING_DATA_DIR": os.environ["BRAINDECODING_DATA_DIR"],
        "BRAINDECODING_CACHE_DIR": str(
            os.environ.get("BRAINDECODING_CACHE_DIR", DEFAULT_CACHE_ROOT)
        ),
        "BRAINDECODING_RESULTS_DIR": str(
            os.environ.get("BRAINDECODING_RESULTS_DIR", DEFAULT_RESULTS_ROOT)
        ),
        "BRAINDECODING_MODEL_ROOT": os.environ["BRAINDECODING_MODEL_ROOT"],
    }
