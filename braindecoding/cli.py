"""BrainDecoding 的稳定命令行入口。"""

from __future__ import annotations

import argparse
import json
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brain-decoding")
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
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    raise AssertionError(f"未处理的命令：{args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
