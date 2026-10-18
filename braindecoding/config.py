"""项目配置读取与路径解析。"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRAINDATA_ROOT = "D:/dataset"
DEFAULT_MODEL_ROOT = "D:/code/dascoli-word-decoding/models"
_ENVIRONMENT_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


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
        if name not in os.environ:
            raise ValueError(f"配置引用了未设置的环境变量：{name}")
        return os.environ[name]

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
    """以项目根目录为基准解析相对路径，绝对路径保持不变。"""
    expanded_value = expand_environment_variables(str(value))
    path = Path(expanded_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / normalize_project_relative(expanded_value)



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
    data_root: str = DEFAULT_BRAINDATA_ROOT,
    model_root: str = DEFAULT_MODEL_ROOT,
) -> None:
    """为独立运行的协议入口设置默认数据与本地模型根目录（若外部未显式设置）。"""
    os.environ.setdefault("BRAINDATA_ROOT", str(data_root))
    os.environ.setdefault("BRAINDECODING_MODEL_ROOT", str(model_root))
