"""实验身份、运行目录与可复现清单工具。"""

from __future__ import annotations

import copy
import hashlib
from importlib import metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path

import yaml

from braindecoding.config import PROJECT_ROOT, project_path


EXPERIMENT_CATEGORIES = (
    "main",
    "ablation",
    "scaling",
    "development",
    "historical_diagnostic",
)
_IDENTITY_KEYS = ("task", "dataset", "subject_scope", "id")
_OPTIONAL_PATH_COMPONENTS = ("output_group",)
_RUN_STATUSES = ("running", "failed", "completed")
_IDENTITY_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_PROJECT_PATH_PREFIXES = (
    "artifacts/",
    "configs/",
    "data/",
    "derived/",
    "experiments/",
    "outputs/",
    "references/",
    "tasks/",
)


def experiment_identity(config):
    """验证并返回稳定的实验五元组。"""
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("canonical 配置缺少 experiment 身份。")
    missing = [key for key in _IDENTITY_KEYS if not experiment.get(key)]
    if missing:
        raise ValueError(f"experiment 身份缺少字段：{', '.join(missing)}")
    invalid = [
        key
        for key in _IDENTITY_KEYS
        if not _IDENTITY_COMPONENT.fullmatch(str(experiment[key]))
    ]
    if invalid:
        raise ValueError(
            "experiment 身份字段只能使用小写字母、数字、下划线和连字符："
            + ", ".join(invalid)
        )
    invalid_optional = [
        key
        for key in _OPTIONAL_PATH_COMPONENTS
        if experiment.get(key)
        and not _IDENTITY_COMPONENT.fullmatch(str(experiment[key]))
    ]
    if invalid_optional:
        raise ValueError(
            "experiment 可选路径字段只能使用小写字母、数字、下划线和连字符："
            + ", ".join(invalid_optional)
        )
    category = experiment.get("category")
    if category not in EXPERIMENT_CATEGORIES:
        raise ValueError(f"experiment.category 不受支持：{category}")
    training = config.get("training")
    if not isinstance(training, dict) or "seed" not in training:
        raise ValueError("canonical 配置必须显式提供 training.seed。")
    raw_seed = training["seed"]
    if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
        raise ValueError("training.seed 必须为非负整数。")
    seed = raw_seed
    if seed < 0:
        raise ValueError("training.seed 必须为非负整数。")
    return {
        "task": str(experiment["task"]),
        "dataset": str(experiment["dataset"]),
        "subject_scope": str(experiment["subject_scope"]),
        "experiment_id": str(experiment["id"]),
        "seed": seed,
    }


def run_directory(config, output_root=None):
    """仅由实验五元组推导单次运行目录。"""
    identity = experiment_identity(config)
    root = Path(output_root) if output_root is not None else PROJECT_ROOT / "outputs"
    path = root / identity["task"] / identity["dataset"]
    output_group = config["experiment"].get("output_group")
    if output_group:
        path = path / str(output_group)
    return (
        path
        / identity["subject_scope"]
        / identity["experiment_id"]
        / f"seed-{identity['seed']:03d}"
    )


def warm_start_checkpoint(config, output_root=None):
    """解析同任务、数据集、受试者范围下的 warm-start 检查点。"""
    training = config.get("training", {})
    warm_start_from = training.get("warm_start_from")
    if not warm_start_from:
        return None
    source_config = copy.deepcopy(config)
    source_config["experiment"]["id"] = str(warm_start_from)
    source_seed = training.get("warm_start_seed")
    if source_seed is not None:
        if (
            isinstance(source_seed, bool)
            or not isinstance(source_seed, int)
            or source_seed < 0
        ):
            raise ValueError("training.warm_start_seed 必须为非负整数。")
        source_config["training"]["seed"] = source_seed
    # 隔离输出组中的新实验通常仍需从主实验预热。显式的 null 表示来源
    # 位于默认输出树，而不是沿用目标实验的 output_group。
    if "warm_start_output_group" in training:
        source_group = training["warm_start_output_group"]
        if source_group is None:
            source_config["experiment"].pop("output_group", None)
        else:
            source_config["experiment"]["output_group"] = str(source_group)
    return run_directory(source_config, output_root=output_root) / "best.pt"


def evaluation_output_path(config, split, *, smoke=False):
    """返回评价文件位置；legacy 配置继续使用原文件名。"""
    output_dir = Path(config["training"]["output_dir"])
    suffix = "_smoke" if smoke else ""
    if "experiment" in config:
        return output_dir / "evaluation" / f"{split}{suffix}.json"
    return output_dir / f"evaluation_{split}{suffix}.json"


def resolve_experiment_config(config, output_root=None):
    """向 canonical 配置注入派生路径；旧配置保持原样。"""
    resolved = copy.deepcopy(config)
    if "experiment" not in resolved:
        return resolved
    output_dir = str(run_directory(resolved, output_root=output_root))
    resolved.setdefault("training", {})["output_dir"] = output_dir
    dataset_name = experiment_identity(resolved)["dataset"]
    if dataset_name in {
        "chineseeeg2_littleprince",
        "smn4lang",
        "libribrain100",
        "pallier2025",
    }:
        # 只有 canonical 词级配置改用 derived；legacy 配置仍保留原缓存路径。
        from braindecoding.data.derived import canonical_cache_paths

        resolved.setdefault("cache", {}).update(canonical_cache_paths(dataset_name))
        if dataset_name in {
            "chineseeeg2_littleprince",
            "smn4lang",
            "pallier2025",
        }:
            resolved.setdefault("dataset", {})[
                "signal_cache_subject_subdirectories"
            ] = True
    # resolved config 应保存实际使用的项目资源路径。此前 ChineseEEG2/Pallier
    # task loader 会额外展开这些字段，而 preflight 只经过本函数，造成同一配置
    # 在运行与预检阶段得到不同 resolved SHA。
    dataset = resolved.setdefault("dataset", {})
    for key in ("alignment_path", "split_manifest", "qc_artifact", "layout_path"):
        value = dataset.get(key)
        if value:
            dataset[key] = str(project_path(value))
    for source in dataset.get("actual_reading_sources", ()):
        value = source.get("alignment_path")
        if value:
            source["alignment_path"] = str(project_path(value))
    if "closed_set_diagnostic" in resolved:
        resolved["closed_set_diagnostic"]["output_dir"] = output_dir
    checkpoint = warm_start_checkpoint(resolved, output_root=output_root)
    if checkpoint is not None:
        resolved["training"]["pretrained_brain_encoder_checkpoint"] = str(
            checkpoint
        )
    return resolved


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolved_config_sha256(config):
    """计算与字典键顺序无关的 resolved config SHA-256。"""
    payload = json.dumps(
        _json_ready(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _legacy_portable_rel(rel: str) -> str:
    for new_p, old_p in (
        ("data/manifests/protocol_index.yaml", "experiments/protocol_index.yaml"),
        ("data/manifests/", "experiments/manifests/"),
        ("data/cache/artifacts/", "artifacts/"),
        ("data/cache/derived/", "derived/"),
        ("data/results/", "reports/"),
    ):
        if rel == new_p.rstrip("/"):
            return old_p.rstrip("/")
        if rel.startswith(new_p):
            return old_p + rel[len(new_p):]
    return rel


def _portable_path(value, key=None):
    """把机器绝对路径归一为项目或环境变量下的稳定身份。"""
    text = str(value).replace("\\", "/")
    path = Path(value)
    if key and (
        key.endswith("_path")
        or key.endswith("_dir")
        or key in {"event_table", "text_embeddings"}
    ):
        return f"<RESOURCE>/{path.name}"
    roots = [("PROJECT_ROOT", PROJECT_ROOT)]
    for name in ("BRAINDATA_ROOT", "BRAINDECODING_MODEL_ROOT"):
        root = os.environ.get(name)
        if root:
            roots.append((name, Path(root)))
    if path.is_absolute():
        for name, root in roots:
            try:
                relative = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            if name == "PROJECT_ROOT":
                relative = _legacy_portable_rel(relative)
            return f"${{{name}}}/{relative}" if relative != "." else f"${{{name}}}"
        return f"<ABSOLUTE_PATH>/{path.name}"
    if text.startswith(_PROJECT_PATH_PREFIXES):
        return f"${{PROJECT_ROOT}}/{_legacy_portable_rel(text)}"
    return value


def _scientific_config(value, path=()):
    """移除运行位置与机器 provenance，同时保留科学参数。"""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key = str(key)
            if not path and key in {"experiment", "provenance"}:
                continue
            if path == ("training",) and key in {
                "device",
                "num_workers",
                "output_dir",
                "pretrained_brain_encoder_checkpoint",
            }:
                continue
            result[key] = _scientific_config(item, (*path, key))
        return result
    if isinstance(value, (list, tuple)):
        return [_scientific_config(item, path) for item in value]
    if isinstance(value, Path):
        return _portable_path(value, path[-1] if path else None)
    if isinstance(value, str):
        return _portable_path(value, path[-1] if path else None)
    return value


def scientific_config_sha256(config):
    """计算排除运行目录、机器路径和纯 provenance 的科学配置摘要。"""
    return resolved_config_sha256(_scientific_config(config))


def file_sha256(path):
    """计算文件 SHA-256；文件不存在时返回 ``None``。"""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(project_root=PROJECT_ROOT):
    safe_root = str(Path(project_root).resolve())
    environment = os.environ.copy()
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    result = subprocess.run(
        ["git", "-c", f"safe.directory={safe_root}", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_tracked_dirty(project_root=PROJECT_ROOT):
    """只检查已跟踪文件的 staged/unstaged 修改，忽略本地未跟踪资产。"""
    safe_root = str(Path(project_root).resolve())
    environment = os.environ.copy()
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={safe_root}",
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(f"无法检查 Git 工作树状态：{result.stderr.strip()}")
    return bool(result.stdout.strip())


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_provenance():
    """读取版本和 CUDA 元数据，不下载依赖或访问实验数据。"""
    torch_version = _package_version("torch")
    cuda_available = False
    cuda_version = None
    cudnn_version = None
    gpu_name = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except (ImportError, OSError, RuntimeError):
        pass
    return {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "numpy_version": _package_version("numpy"),
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_name": gpu_name,
        "mne_version": _package_version("mne"),
        "transformers_version": _package_version("transformers"),
    }


def _path_record(path):
    path = Path(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def build_run_manifest(
    config,
    command,
    status="running",
    protocol_manifests=(),
    vocabulary_manifests=(),
):
    """构造不主动读取测试数据的单次运行清单。"""
    if status not in _RUN_STATUSES:
        raise ValueError(f"运行状态不受支持：{status}")
    identity = experiment_identity(config)
    event_table = config.get("cache", {}).get("event_table")
    if event_table is None:
        event_table = config.get("training", {}).get("event_table")
    return {
        "schema_version": 1,
        "experiment": identity,
        "resolved_config_sha256": resolved_config_sha256(config),
        "scientific_config_sha256": scientific_config_sha256(config),
        "git_commit": _git_commit(),
        "git_dirty": git_tracked_dirty(),
        "environment": environment_provenance(),
        "command": command if isinstance(command, str) else list(command),
        "seed": identity["seed"],
        "event_table": _path_record(event_table) if event_table else None,
        "protocol_manifests": [
            _path_record(path) for path in protocol_manifests
        ],
        "vocabulary_manifests": [
            _path_record(path) for path in vocabulary_manifests
        ],
        "status": status,
    }


def initialize_run_directory(
    config,
    command,
    *,
    resume=False,
    output_root=None,
    protocol_manifests=(),
    vocabulary_manifests=(),
    allow_dirty=False,
):
    """建立标准运行资产，并拒绝含糊覆盖已有目录。

    ``allow_dirty`` 暂时保留为兼容参数。工作树状态仍写入 provenance，
    但当前不再阻断 canonical 运行。
    """
    resolved = resolve_experiment_config(config, output_root=output_root)
    output_dir = run_directory(resolved, output_root=output_root)
    manifest_path = output_dir / "run_manifest.json"
    config_hash = resolved_config_sha256(resolved)

    if output_dir.exists():
        if not manifest_path.is_file():
            raise FileExistsError(
                f"运行目录已存在但缺少 run_manifest.json，拒绝覆盖：{output_dir}"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("resolved_config_sha256") != config_hash:
            raise FileExistsError(f"运行目录中的配置 SHA 不同，拒绝覆盖：{output_dir}")
        if existing.get("status") == "completed":
            raise FileExistsError(f"运行已经完成，拒绝再次覆盖：{output_dir}")
        if not resume:
            raise FileExistsError(
                f"已有 {existing.get('status')} 运行；只有显式 resume 才可继续：{output_dir}"
            )
    else:
        output_dir.mkdir(parents=True)

    manifest = build_run_manifest(
        resolved,
        command,
        status="running",
        protocol_manifests=protocol_manifests,
        vocabulary_manifests=vocabulary_manifests,
    )
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "evaluation").mkdir(exist_ok=True)
    (output_dir / "audits" / "validation").mkdir(parents=True, exist_ok=True)
    return output_dir, manifest


def update_run_status(output_dir, status):
    """更新已有运行清单状态，不改写其他 provenance。"""
    if status not in _RUN_STATUSES:
        raise ValueError(f"运行状态不受支持：{status}")
    manifest_path = Path(output_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到运行清单：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def require(condition: bool, message: str) -> None:
    """科学协议断言辅助函数：条件不满足时抛出 ValueError。"""
    if not condition:
        raise ValueError(message)


def read_json(path: Path | str):
    """读取 UTF-8 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json_atomic(path: Path | str, payload, *, allow_nan: bool = True) -> Path:
    """使用临时文件原子替换写入 UTF-8 JSON。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def json_digest(value) -> str:
    """计算任意 JSON 可序列化结构的稳定 SHA-256 摘要。"""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def source_key(path: Path | str, root: Path = PROJECT_ROOT) -> str:
    """将仓库内路径规范化为相对 POSIX 路径，仓库外路径保持绝对字符串。"""
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    return (
        resolved.relative_to(root_resolved).as_posix()
        if resolved.is_relative_to(root_resolved)
        else str(resolved)
    )
