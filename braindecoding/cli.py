"""BrainDecoding 的稳定命令行入口。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from braindecoding.catalog import (
    discover_runnable_experiment_configs,
    resolve_selector,
)
from braindecoding.config import PROJECT_ROOT
from braindecoding.experiment import (
    run_directory,
    warm_start_checkpoint,
)
from braindecoding.protocol_catalog import (
    discover_protocols,
    protocol_status,
    repository_path,
    resolve_protocol,
)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _run_status(record: dict) -> dict:
    """只读判断一个 canonical run 当前处于什么状态。"""
    config = record["raw_config"]
    output = run_directory(config)
    manifest = _read_json(output / "run_manifest.json")
    diagnostic = _read_json(output / "implementation_diagnostic.json")
    dependency = warm_start_checkpoint(config)
    if diagnostic is not None and diagnostic.get("status") == "implementation_diagnostic":
        status = "implementation_diagnostic"
    elif manifest is not None:
        status = manifest.get("status")
        if status not in {"running", "failed", "completed"}:
            status = "failed"
    elif output.exists():
        status = "failed"
    elif dependency is not None and not dependency.is_file():
        status = "waiting_for_upstream"
    else:
        status = "not_started"
    return {
        "status": status,
        "run_path": output,
        "manifest": manifest,
        "diagnostic": diagnostic,
        "dependency": dependency,
        "best_checkpoint": output / "best.pt",
        "validation": output / "evaluation" / "val.json",
        "audit": output / "audits" / "validation" / "summary.json",
    }


def _legacy_notice() -> None:
    print("历史配置查询：这些配置不属于当前默认实验列表，旧执行入口已停用。", file=sys.stderr)


def _print_list(legacy: bool = False) -> int:
    if not legacy:
        print("protocol                            role       datasets                               recorded status")
        for record in discover_protocols():
            states = " ".join(
                f"{state['name']}={state['status']}" for state in protocol_status(record)
            )
            print(
                f"{record['id']:<35} {record['role']:<10} "
                f"{','.join(record['datasets']):<38} {states}"
            )
        return 0
    _legacy_notice()
    print("dataset                         subject scope  experiment    status")
    for record in discover_runnable_experiment_configs():
        identity = record["identity"]
        status = _run_status(record)["status"]
        print(
            f"{identity['dataset']:<31} "
            f"{identity['subject_scope']:<14} "
            f"{identity['experiment_id']:<13} {status}"
        )
    return 0


def _show_protocol(identifier: str, resolved: bool = False) -> int:
    record = resolve_protocol(identifier)
    if resolved:
        print(yaml.safe_dump(record, allow_unicode=True, sort_keys=False))
        return 0
    print(f"protocol: {record['id']}")
    print(f"title: {record['title']}")
    print(f"role: {record['role']}")
    print(f"datasets: {', '.join(record['datasets'])}")
    print(f"document: {repository_path(record['protocol'])}")
    print(f"config: {repository_path(record['config'])}")
    for command in record["entrypoints"]:
        print(f"entrypoint help: {command}")
    for relative in record.get("manifests", []) + record.get("related_protocols", []):
        print(f"frozen reference: {repository_path(relative)}")
    for relative in record["reports"]:
        print(f"report: {repository_path(relative)}")
    return 0


def _retired_execution(command: str, identifier: str | None = None) -> int:
    print(f"通用执行入口已停用：brain-decoding {command}。", file=sys.stderr)
    print("请按当前协议使用专用入口；不会启动训练、评价或审计。", file=sys.stderr)
    if identifier and "/" not in identifier:
        record = resolve_protocol(identifier)
        for entrypoint in record["entrypoints"]:
            print(f"  {entrypoint}", file=sys.stderr)
    else:
        print("  brain-decoding list", file=sys.stderr)
        print("  brain-decoding show unified_protocol_v2", file=sys.stderr)
    return 2


def _task_modules(dataset: str):
    """延迟导入数据集任务，list/status 不加载训练依赖。"""
    if dataset == "chineseeeg2_littleprince":
        from braindecoding.tasks.word_decoding.chineseeeg2_littleprince import (
            evaluate,
            train,
        )

        return train, evaluate
    if dataset == "pallier2025":
        from braindecoding.tasks.word_decoding.pallier2025 import evaluate, train

        return train, evaluate
    raise ValueError(f"当前 CLI 不支持数据集：{dataset}")


def _load_task_config(record: dict) -> tuple[dict, object, object]:
    train, evaluate = _task_modules(record["identity"]["dataset"])
    loader = getattr(train, "载入配置", None) or train.load_config
    return loader(record["config_path"]), train, evaluate


def _derived_status(dataset: str) -> str:
    manifest = _read_json(PROJECT_ROOT / "derived" / dataset / "manifest.json")
    return manifest.get("status", "missing") if manifest else "missing"


def _show(selector: str, resolved: bool) -> int:
    if "/" not in selector:
        return _show_protocol(selector, resolved)
    _legacy_notice()
    record = resolve_selector(selector)
    config, _, _ = _load_task_config(record)
    if resolved:
        print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
        return 0
    identity = record["identity"]
    dataset = config["dataset"]
    training = config["training"]
    model = config["model"]
    budget = (
        f"{training['max_updates']} updates"
        if training.get("max_updates") is not None
        else f"{training['epochs']} epochs"
    )
    dependency = warm_start_checkpoint(config)
    print(f"identity: {record['selector']}")
    print(f"subjects: {', '.join(map(str, dataset.get('subjects', ())))}")
    print(f"derived status: {_derived_status(identity['dataset'])}")
    print(
        "window: "
        f"{dataset.get('window_start_offset_seconds', 0.0)} + "
        f"{dataset.get('window_seconds')} seconds"
    )
    print(
        "model: "
        f"use_transformer={model.get('use_transformer')}, "
        f"context_mode={model.get('context_mode')}"
    )
    print(f"training budget: {budget}")
    print(f"seed: {identity['seed']}")
    print(f"output: {run_directory(config)}")
    print(f"dependency: {dependency if dependency is not None else 'none'}")
    return 0


def _preflight(selector: str) -> int:
    if "/" not in selector:
        resolve_protocol(selector)
        raise ValueError(
            "当前协议的来源与运行检查由专用入口执行；"
            f"请先查看 brain-decoding show {selector}。"
        )
    _legacy_notice()
    from braindecoding.preflight import build_preflight_report

    record = resolve_selector(selector)
    report = build_preflight_report(config_paths=(record["relative_config"],))
    item = next(
        value for value in report["experiments"] if value["config"] == record["relative_config"]
    )
    print(json.dumps(item, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if item["ready"] in {True, "waiting_for_upstream"} else 1


def _print_protocol_status(identifier: str | None) -> int:
    records = [resolve_protocol(identifier)] if identifier else discover_protocols()
    print("保存的状态快照；不检查进程存活，也不据此确认完整结果。")
    for record in records:
        print(f"{record['id']}:")
        for state in protocol_status(record):
            print(f"  {state['name']}: {state['status']}")
            if identifier:
                print(f"    source: {state['path'] or 'not_recorded'}")
                snapshot = state["snapshot"] or {}
                for key in ("stage", "current", "completed_stages", "total_stages",
                            "completed_segment_seed_units", "total_segment_seed_units",
                            "updated_unix", "pause_reason", "worker_exited"):
                    if key in snapshot:
                        value = json.dumps(snapshot[key], ensure_ascii=False)
                        print(f"    {key}: {value}")
                if state.get("error"):
                    print(f"    error: {state['error']}")
    return 0


def _print_status(selector: str | None, legacy: bool = False) -> int:
    if not legacy and (not selector or "/" not in selector):
        return _print_protocol_status(selector)
    if selector and "/" not in selector:
        raise ValueError("--legacy 查询需要 dataset/subject_scope/experiment_id。")
    _legacy_notice()
    records = (
        [resolve_selector(selector)]
        if selector
        else discover_runnable_experiment_configs()
    )
    for record in records:
        state = _run_status(record)
        print(f"{record['selector']}: {state['status']}")
        if selector:
            print(f"  run path: {state['run_path']}")
            print(f"  best checkpoint: {'present' if state['best_checkpoint'].is_file() else 'missing'}")
            print(f"  validation: {'present' if state['validation'].is_file() else 'not_run'}")
            print(f"  audit: {'present' if state['audit'].is_file() else 'not_run'}")
            print("  test: locked_not_evaluated")
            if state["diagnostic"] is not None:
                print(f"  diagnostic reason: {state['diagnostic'].get('reason')}")
    return 0


def _data_check(dataset: str) -> int:
    from braindecoding.data.build import check_dataset

    print(json.dumps(check_dataset(dataset), ensure_ascii=False, indent=2))
    return 0


def _data_build(args) -> int:
    datasets = {dataset for record in discover_protocols() for dataset in record["datasets"]}
    if args.dataset not in datasets:
        raise ValueError(
            f"历史数据集的构建入口已停用：{args.dataset}。"
            f"当前数据集：{', '.join(sorted(datasets))}。"
        )
    from braindecoding.data.build import build_dataset

    selected = args.events or args.signals or args.text or args.all
    result = build_dataset(
        args.dataset,
        events=args.events or not selected or args.all,
        signals=args.signals or not selected or args.all,
        text=args.text or not selected or args.all,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _report(selector: str, write: bool) -> int:
    """只读取 canonical JSON，显示或写出派生的人类可读视图。"""
    if "/" not in selector:
        record = resolve_protocol(selector)
        if write:
            raise ValueError("当前协议报告由专用生成器维护；请按对应冻结协议操作。")
        for relative in record["reports"]:
            path = repository_path(relative)
            print(f"{'present' if path.exists() else 'not_recorded'}: {path}")
        return 0
    _legacy_notice()
    from braindecoding.results import render_run_report, write_run_report

    record = resolve_selector(selector)
    config, _, _ = _load_task_config(record)
    markdown = render_run_report(config)
    print(markdown)
    if write:
        print(f"已写入：{write_run_report(config)}")
    return 0


def _add_subject_subset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--subject",
        default="all",
        help="受试者选择：'all' 或单个受试者 ID（如 sub-01）",
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=None,
        help="按标准顺序自动选取前 N 个有效受试者",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="逗号分隔的受试者列表（如 sub-01,sub-02）",
    )


def _add_training_workflow_args(
    parser: argparse.ArgumentParser, *, default_epochs: int | None = None
) -> None:
    parser._negative_number_matcher = re.compile(
        r"^-\d+(?:\.\d+)?(?:,\s*[+-]?\d+(?:\.\d+)?)*$"
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        default=None,
        help="YAML 配置文件路径（如 configs/training/pallier2025_word_encoder.yaml）",
    )
    parser.add_argument(
        "--dataset",
        choices=("chineseeeg2_littleprince", "pallier2025"),
        default=None,
        help="目标数据集（指定 --config 时自动从 YAML 读取；未指定时默认 chineseeeg2_littleprince）",
    )
    parser.add_argument(
        "--stage",
        choices=("word", "context", "revision", "1", "2.1", "2.2"),
        default=None,
        help="训练阶段：word/1（阶段1：单窗脑编码器）、context/2.1（阶段2.1：脑编码器+16层Transformer）或 revision/2.2（阶段2.2：16层局部窗口滚动修订）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认：0）")
    _add_subject_subset_args(parser)
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_epochs,
        help="训练轮数（smoke 默认 2，train 默认从 YAML 读取）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="可选覆写训练批大小（默认从 YAML 读取，L4 GPU 默认为 256）",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="可选覆写验证与检索批大小（默认从 YAML 读取，L4 GPU 默认为 512）",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        dest="learning_rate",
        type=float,
        default=None,
        help="可选覆写初始学习率（默认从 YAML 读取，bs>=256 默认为 4e-4）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader 工作进程数（默认 0 使用进程内缓存零拷贝）",
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=("bf16-mixed", "fp16-mixed", "fp16", "fp32"),
        help="混合精度模式（默认从 YAML 读取或使用 bf16-mixed）",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="可选切换单字神经信号窗口长度（秒，如 0.5、1.0、1.25、1.5、2.0；默认从 YAML 读取，标准为 1.0 秒 = 50 采样点 @ 50Hz）",
    )
    parser.add_argument(
        "--window-start-offset-seconds",
        type=float,
        default=None,
        help="可选切换相对词起始时刻的窗口起点偏移（秒，默认 0.0）",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="要求必须命中预处理特征缓存，缺失时立即报错（禁止在 GPU 上现场重算原始数据）",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="断点续训策略：'auto' 自动从 last.ckpt 恢复，或传入具体检查点路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选显式指定输出目录（默认按数据集/实验标签/阶段/受试者子集自动隔离）",
    )
    parser.add_argument(
        "--exp-tag",
        default=None,
        help="可选超参数实验标签（如 lr1e-4_bs128），输出路径为 <results_dir>/<dataset>/<exp_tag>/<stage>/...",
    )
    parser.add_argument(
        "--word-exp-tag",
        default=None,
        help="context (2.1) / revision (2.2) 阶段可选指定从哪个阶段 1 实验标签加载 word 检查点（如 lr1e-4_bs128；填 default 加载根目录 word）",
    )
    parser.add_argument(
        "--word-checkpoint",
        default=None,
        help="context (2.1) / revision (2.2) 阶段可选显式指定预训练 word 检查点路径",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="可选限制训练事件行数上限（用于超快冒烟联调）",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        help="可选限制验证事件行数上限（用于超快冒烟联调）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已完成的同子集实验输出目录",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="禁用 tqdm 实时进度条（仅输出每轮汇总表格）",
    )


def _add_evaluation_workflow_args(parser: argparse.ArgumentParser) -> None:
    parser._negative_number_matcher = re.compile(
        r"^-\d+(?:\.\d+)?(?:,\s*[+-]?\d+(?:\.\d+)?)*$"
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        default=None,
        help="YAML 配置文件路径（如 configs/training/pallier2025_context_transformer.yaml）",
    )
    parser.add_argument(
        "--dataset",
        choices=("chineseeeg2_littleprince", "pallier2025"),
        default=None,
        help="目标数据集（指定 --config 时自动从 YAML 读取；未指定时默认 chineseeeg2_littleprince）",
    )
    parser.add_argument(
        "--stage",
        choices=("word", "context", "revision", "1", "2.1", "2.2"),
        default=None,
        help="要评估的阶段：word/1、context/2.1 或 revision/2.2",
    )
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="自动扫描并评估当前数据集/受试者子集下已训练完成的全部阶段（word / context / revision）并生成多阶段对比图",
    )
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="val",
        help="评估划分：val（验证集）或 test（留出测试集）",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认：0）")
    _add_subject_subset_args(parser)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="可选显式指定待评估的 best.ckpt / best.pt 路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="可选显式指定评估结果与图表输出目录",
    )
    parser.add_argument(
        "--exp-tag",
        default=None,
        help="可选超参数实验标签（如 lr1e-4_bs128），自动定位 <results_dir>/<dataset>/<exp_tag>/<stage>/...",
    )
    parser.add_argument(
        "--word-exp-tag",
        default=None,
        help="跨阶段对比评估时可选指定阶段 1 (word) 所在的实验标签（如 lr1e-4_bs128；填 default 定位根目录 word）",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="评估前向批大小（默认从 YAML 读取，L4 GPU 默认为 512）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader 工作进程数（默认 0）",
    )
    parser.add_argument(
        "--precision",
        default=None,
        choices=("bf16-mixed", "fp16-mixed", "fp16", "fp32"),
        help="混合精度模式（默认 bf16-mixed）",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="可选覆写评估时切取的单字神经信号窗口长度（秒；未指定时自动沿用检查点或 YAML 配置）",
    )
    parser.add_argument(
        "--window-start-offset-seconds",
        type=float,
        default=None,
        help="可选覆写相对词起始时刻的窗口起点偏移（秒；未指定时自动沿用检查点或 YAML 配置）",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="要求必须命中预处理特征缓存，缺失时立即报错",
    )
    parser.add_argument(
        "--control-config",
        default=None,
        help="可选传入独立解耦的负对照 YAML 配置（如 configs/controls/temporal_shift/temporal_shift_control.yaml）",
    )
    parser.add_argument(
        "--shift-seconds",
        default=None,
        help="可选逗号分隔的时间错位秒数列表（如 -120,-60,-30,30,60,120），以真实错位脑磁/脑电信号作为词级检索负对照",
    )
    parser.add_argument(
        "--zero-signal-control",
        action="store_true",
        help="同时评估全零脑磁/脑电信号负对照（Zero-Signal Control）",
    )
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=None,
        help="可选限制评估事件行数上限（用于快速联调）",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="禁用可视化图表生成",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="禁用 tqdm 实时进度条",
    )


def _workflow_cache(args) -> int:
    from braindecoding.training.workflow_runner import run_cache_workflow

    dataset = args.dataset
    if dataset is None and args.config is None:
        dataset = "chineseeeg2_littleprince"
    report = run_cache_workflow(
        dataset,
        subject=args.subject,
        num_subjects=args.num_subjects,
        subjects=args.subjects,
        include_test=args.include_test,
        force=args.force,
        config_path=args.config,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _workflow_train(args, *, mode: str) -> int:
    from braindecoding.training.workflow_runner import run_training_workflow

    dataset = args.dataset
    if dataset is None and args.config is None:
        dataset = "chineseeeg2_littleprince"
    run_training_workflow(
        dataset,
        stage=args.stage,
        seed=args.seed,
        subject=args.subject,
        num_subjects=args.num_subjects,
        subjects=args.subjects,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        precision=args.precision,
        window_seconds=getattr(args, "window_seconds", None),
        window_start_offset_seconds=getattr(
            args, "window_start_offset_seconds", None
        ),
        require_cache=args.require_cache,
        resume=args.resume,
        mode=mode,
        output_dir=args.output_dir,
        exp_tag=getattr(args, "exp_tag", None),
        word_exp_tag=getattr(args, "word_exp_tag", None),
        word_checkpoint=args.word_checkpoint,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        overwrite=args.overwrite,
        show_progress=not bool(getattr(args, "no_progress", False)),
        config_path=args.config,
    )
    return 0


def _workflow_eval(args) -> int:
    from braindecoding.config import (
        load_yaml_with_extends,
        project_path,
        resolve_config_path,
    )
    from braindecoding.training.workflow_runner import run_evaluation_workflow

    dataset = args.dataset
    control_cfg = getattr(args, "control_config", None)
    if dataset is None and args.config is None:
        if control_cfg is not None:
            resolved_ctrl = resolve_config_path(project_path(control_cfg))
            if resolved_ctrl.is_file():
                raw_ctrl = load_yaml_with_extends(resolved_ctrl)
                if isinstance(raw_ctrl, dict):
                    dataset = (raw_ctrl.get("workflow") or {}).get("dataset")
        if dataset is None:
            dataset = "chineseeeg2_littleprince"
    run_evaluation_workflow(
        dataset,
        stage=args.stage,
        seed=args.seed,
        split=args.split,
        subject=args.subject,
        num_subjects=args.num_subjects,
        subjects=args.subjects,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        precision=args.precision,
        window_seconds=getattr(args, "window_seconds", None),
        window_start_offset_seconds=getattr(
            args, "window_start_offset_seconds", None
        ),
        require_cache=args.require_cache,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        exp_tag=getattr(args, "exp_tag", None),
        word_exp_tag=getattr(args, "word_exp_tag", None),
        config_path=args.config,
        control_config_path=control_cfg,
        shift_seconds=getattr(args, "shift_seconds", None),
        zero_signal_control=bool(getattr(args, "zero_signal_control", False)),
        all_stages=args.all_stages,
        plot=not bool(getattr(args, "no_plot", False)),
        show_progress=not bool(getattr(args, "no_progress", False)),
        max_eval_rows=args.max_eval_rows,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-decoding")
    parser._negative_number_matcher = re.compile(
        r"^-\d+(?:\.\d+)?(?:,\s*[+-]?\d+(?:\.\d+)?)*$"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="列出主协议与补充实验")
    listing.add_argument("--legacy", action="store_true", help="只读列出历史词级配置")

    data = commands.add_parser("data", help="构建或检查 canonical derived 数据")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_check = data_commands.add_parser("check", help="只读验证 derived 数据")
    data_check.add_argument("dataset")
    data_build = data_commands.add_parser("build", help="构建 derived 数据")
    data_build.add_argument("dataset")
    data_build.add_argument("--events", action="store_true")
    data_build.add_argument("--signals", action="store_true")
    data_build.add_argument("--text", action="store_true")
    data_build.add_argument("--all", action="store_true")
    data_build.add_argument("--force", action="store_true")

    cache_cmd = commands.add_parser(
        "cache", help="预热并校验指定数据集与受试者子集的预处理特征缓存"
    )
    cache_cmd.add_argument(
        "--config",
        "-c",
        dest="config",
        default=None,
        help="可选传入 YAML 配置文件（如 configs/training/pallier2025_word_encoder.yaml）",
    )
    cache_cmd.add_argument(
        "--dataset",
        choices=("chineseeeg2_littleprince", "pallier2025", "all"),
        default=None,
        help="要预热缓存的数据集（未传 --config 时默认 chineseeeg2_littleprince，支持 all）",
    )
    _add_subject_subset_args(cache_cmd)
    cache_cmd.add_argument(
        "--include-test",
        action="store_true",
        help="同时预热留出测试集信号缓存",
    )
    cache_cmd.add_argument(
        "--force",
        action="store_true",
        help="强制重新计算并覆盖已有缓存",
    )

    smoke_cmd = commands.add_parser(
        "smoke", help="本地小样本极速冒烟测试（默认 2 Epoch + bf16-mixed）"
    )
    _add_training_workflow_args(smoke_cmd, default_epochs=2)

    train_cmd = commands.add_parser(
        "train", help="支持灵活受试者子集、YAML 配置文件、缓存校验与断点续训的正式训练入口"
    )
    _add_training_workflow_args(train_cmd, default_epochs=None)

    eval_run_cmd = commands.add_parser(
        "eval-run", help="评估已训练检查点（支持 val 验证集与 test 测试集、单阶段或全阶段对比及可视化绘图）"
    )
    _add_evaluation_workflow_args(eval_run_cmd)

    show = commands.add_parser("show", help="显示当前协议或显式指定的历史配置")
    show.add_argument("experiment")
    show.add_argument("--resolved", action="store_true", help="显示协议索引元数据或历史完整配置")
    preflight = commands.add_parser("preflight", help="只读检查显式指定的历史词级配置")
    preflight.add_argument("experiment")
    run = commands.add_parser("run", help="已停用；显示当前专用入口")
    run.add_argument("experiment")
    evaluate = commands.add_parser("evaluate", help="已停用；显示当前专用入口")
    evaluate.add_argument("experiment")
    status = commands.add_parser("status", help="查看当前协议保存的状态")
    status.add_argument("experiment", nargs="?")
    status.add_argument("--legacy", action="store_true", help="只读查看历史词级运行状态")
    audit = commands.add_parser("audit", help="已停用；显示当前专用入口")
    audit.add_argument("experiment")
    audit.add_argument("--mapping-only", action="store_true")
    audit.add_argument("--device", default="auto")
    report = commands.add_parser("report", help="查看当前报告位置或历史运行报告")
    report.add_argument("experiment")
    report.add_argument("--write", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            return _print_list(args.legacy)
        if args.command == "data":
            return _data_check(args.dataset) if args.data_command == "check" else _data_build(args)
        if args.command == "cache":
            return _workflow_cache(args)
        if args.command == "smoke":
            return _workflow_train(args, mode="smoke")
        if args.command == "train":
            return _workflow_train(args, mode="train")
        if args.command == "eval-run":
            return _workflow_eval(args)
        if args.command == "show":
            return _show(args.experiment, args.resolved)
        if args.command == "preflight":
            return _preflight(args.experiment)
        if args.command == "run":
            return _retired_execution("run", args.experiment)
        if args.command == "evaluate":
            return _retired_execution("evaluate", args.experiment)
        if args.command == "status":
            return _print_status(args.experiment, args.legacy)
        if args.command == "audit":
            return _retired_execution("audit", args.experiment)
        if args.command == "report":
            return _report(args.experiment, args.write)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    raise AssertionError(f"未处理的命令：{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
