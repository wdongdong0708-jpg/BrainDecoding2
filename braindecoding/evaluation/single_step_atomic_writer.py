"""Operational Windows I/O recovery; the frozen C1 decoding remains unchanged."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from braindecoding import single_revision_c1_test_v2_stable as stable


ROOT = stable.ROOT
MANIFEST = ROOT / "data/manifests/rolling_sentence/single_revision_c1_test_v2_io_guard.json"
ORIGINAL_REPLACE = Path.replace
IO_MANIFEST_SHA = None
LAST_PROGRESS = {"time": 0.0, "state": None, "dataset": None}


def atomic_replace_with_retry(path, target):
    """Retry transient Windows locks only for atomic temporary-file replacement."""
    for attempt in range(12):
        try:
            return ORIGINAL_REPLACE(path, target)
        except OSError as error:
            if (path.suffix != ".tmp" or getattr(error, "winerror", None) not in (5, 32, 33)
                    or attempt == 11):
                raise
            time.sleep(min(0.025*(attempt+1), 0.2))


def status(m, **fields):
    """Persist resumable state; refresh the optional progress page less frequently."""
    path = stable.RUNTIME / "status.json"
    state = stable.read(path) if path.exists() else {}
    now = time.time()
    state.update(pid=os.getpid(), updated_unix=now, manifest_sha256=m["manifest_sha256"],
                 execution_io_manifest_sha256=IO_MANIFEST_SHA,
                 total_segment_seed_units=m["total_segment_seed_units"],
                 adopted_segment_seed_units=m["adopted_segment_seed_units"], **fields)
    stable.save(path, state)  # A persistent state-write failure must remain fatal.
    refresh = (now-LAST_PROGRESS["time"] >= 5 or state.get("state") != LAST_PROGRESS["state"]
               or state.get("dataset") != LAST_PROGRESS["dataset"])
    if refresh:
        LAST_PROGRESS.update(time=now, state=state.get("state"), dataset=state.get("dataset"))
        lines = ["# C1：一次回修、最多等待一个后续词窗", "",
                 f"状态：{state.get('state')}；已保存 {state.get('completed_segment_seed_units', 0):,} / {m['total_segment_seed_units']:,} 个片段×种子。", "",
                 f"当前：{state.get('dataset', '')}，种子 {state.get('seed', '')}，分块 {state.get('chunk', '')}，到达 {state.get('arrival', '')}。", "",
                 f"保留并迁入原有 {m['adopted_segment_seed_units']:,} 个单元，记录内容不变。到达 0/1 共享原 C 的同一 K0/K1 阶段；从到达 2 开始独立执行 C1。",
                 "一次回修、冻结前缀、最多等待一窗及初始一致性校验均保留。全部任务完成后自动统计并释放 GPU。", "",
                 "进度页面每五秒最多刷新一次；短暂文件锁定经过重试仍未解除时只记录展示警告。实验状态与计算结果仍要求成功保存。",
                 "[故障诊断与恢复验证](recovery_validation.md)"]
        if state.get("error"):
            lines += ["", f"错误：{state['error']}"]
        if state.get("state") == "completed":
            lines += ["", "[完整结果](results.md)"]
        try:
            stable.REPORT.mkdir(parents=True, exist_ok=True)
            temporary = stable.REPORT / "progress.md.tmp"
            temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
            temporary.replace(stable.REPORT / "progress.md")
        except OSError as error:
            warning = dict(kind="optional_progress_page_write", unix=now, error=str(error))
            print(json.dumps(warning, ensure_ascii=False), file=sys.stderr, flush=True)
            try:
                with (stable.RUNTIME / "progress_warnings.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(warning, ensure_ascii=False)+"\n")
            except OSError:
                pass  # The warning was already sent to the durable process stderr log.
    print(json.dumps({k: state[k] for k in ("state", "stage", "dataset", "seed", "chunk", "arrival",
                                          "completed_segment_seed_units") if k in state}), flush=True)


def freeze_io():
    m = stable.freeze()
    if MANIFEST.exists():
        io = stable.read(MANIFEST)
        stable.base.verify_identity(io)
        stable.base.verify_hashes(io["source_sha256"])
        stable.require(io["scientific_execution_manifest_sha256"] == m["manifest_sha256"],
                       "I/O guard scientific execution changed")
        return io
    prior = stable.read(stable.RUNTIME / "status.json")
    stable.require(prior["state"] == "failed" and "progress.md" in prior["error"],
                   "The optional progress-page failure must be recorded before this recovery")
    snapshot = stable.RUNTIME / "windows_progress_failure.json"
    stable.save(snapshot, prior)
    files = (Path(__file__), ROOT / "tests/test_single_revision_c1_test_v2_io_guard.py",
             stable.MANIFEST, snapshot, stable.RUNTIME / "stderr.log")
    sources = {stable.base.source_key(path): stable.sha256(path) for path in files}
    io = dict(protocol="single_revision_c1_test_v2_io_guard", frozen_unix=time.time(),
              scientific_execution_manifest_sha256=m["manifest_sha256"],
              scientific_protocol_manifest_sha256=m["scientific_protocol_manifest_sha256"],
              source_sha256=sources,
              changes=dict(atomic_temporary_replace_retry_windows_errors=[5, 32, 33],
                           maximum_replace_attempts=12, optional_progress_refresh_seconds=5,
                           optional_progress_failure_is_nonfatal=True,
                           experiment_state_and_result_writes_required=True,
                           neural_inputs_search_and_statistics_unchanged=True))
    io["manifest_sha256"] = stable.base.digest(io)
    stable.save(MANIFEST, io)
    return io


def run():
    global IO_MANIFEST_SHA
    io = freeze_io()
    IO_MANIFEST_SHA = io["manifest_sha256"]
    prior_provenance = stable.provenance
    def provenance(m, job, **kwargs):
        return dict(**prior_provenance(m, job, **kwargs), execution_io_manifest_sha256=IO_MANIFEST_SHA)
    stable.provenance = provenance
    stable.status = status
    # Only replacement retries change; all payloads and scientific functions are frozen.
    Path.replace = atomic_replace_with_retry
    try:
        stable.run()
    finally:
        Path.replace = ORIGINAL_REPLACE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.command == "prepare":
        io = freeze_io()
        print(json.dumps(dict(stage="frozen_io_guard", manifest_sha256=io["manifest_sha256"])), flush=True)
    else:
        run()


if __name__ == "__main__":
    main()
