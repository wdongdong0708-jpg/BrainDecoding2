"""Preserve completed B records and share only its authenticated first A/K0 output."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil
import torch

from braindecoding import fixed_k0_rolling_search_test_v2 as base
from braindecoding.single_revision_c1_test_v2_io_guard import atomic_replace_with_retry
from braindecoding.unified_sentence_batch import search_many


ROOT = base.ROOT
ID = "fixed_k0_rolling_search_test_v2_stable"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
OUT = ROOT / f"outputs/heldout_test/{ID}"
REPORT = ROOT / f"data/results/studies/{ID}"
RUNTIME = ROOT / f".cache/runs/{ID}"
require, read, save, sha256 = base.require, base.read, base.save, base.sha256
LAST_PROGRESS = {"time": 0., "stage": None, "state": None}


def shared_initial(groups, ac_records):
    """Extract only arrival zero generated A words; references/future outputs are unused."""
    require(len(groups) == len(ac_records), "B shared-start population differs")
    initial = []
    for group, record in zip(groups, ac_records):
        require(group["group"] == record["group"], "B shared-start group differs")
        step = record["result"]["trace"][0]
        require(step["arrival"] == 0 and len(step["A"]) == 1,
                "Only the actual first A arrival may be reused")
        initial.append(dict(arrival=0, words=list(step["A"])))
    return initial


def replay_shared_initial(cases, initial, vocabulary, bank, scorer, embed_words, *, progress=None, **settings):
    require(len(cases) == len(initial), "B shared-start population differs")
    counts, outputs, traces = [], [], []
    for case, entry in zip(cases, initial):
        require(set(case) == {"predictions", "weight", "open_vocabulary"},
                "B case contains undeclared information")
        require(set(case["predictions"]) == {0}, "B must use K0 only")
        k0 = np.asarray(case["predictions"][0])
        require(k0.ndim == 2 and len(k0) > 0 and np.isfinite(k0).all(), "B K0 shape or values differ")
        require(set(entry) == {"arrival", "words"} and entry["arrival"] == 0,
                "B shared input contains undeclared or future information")
        require(len(entry["words"]) == 1 and all(isinstance(w, str) and w for w in entry["words"]),
                "B shared first output differs")
        counts.append(len(k0))
        outputs.append(tuple(entry["words"]))
        traces.append([dict(arrival=0, B=list(entry["words"]))])
    for arrival in range(1, max(counts, default=0)):
        active = [i for i, count in enumerate(counts) if arrival < count]
        start = max(0, arrival-2)
        requests = [dict(queries=cases[i]["predictions"][0][start:arrival+1],
                         prefix=outputs[i][:start], weight=cases[i]["weight"],
                         open_vocabulary=cases[i]["open_vocabulary"]) for i in active]
        results = search_many(requests, vocabulary, bank, scorer, embed_words, **settings)
        require(len(results) == len(active), "B resumed search population differs")
        for i, request, result in zip(active, requests, results):
            require(tuple(result[:start]) == request["prefix"] and len(result) == arrival+1,
                    "B changed a frozen prefix or known word count")
            outputs[i] = tuple(result)
            traces[i].append(dict(arrival=arrival, B=list(result)))
        if progress:
            progress(arrival, counts)
    return [dict(final=dict(B=list(words)), trace=trace) for words, trace in zip(outputs, traces)]


def provenance(m, job, *, initial_stage="authenticated_shared_arrival_0"):
    return dict(manifest_sha256=m["manifest_sha256"], dataset=job["dataset"], seed=job["seed"],
                conditions=["B"], split="test", test_evaluated=True, neural_representation="K0_only",
                branch="open", alignment="real", selected_weight=.1, execution_id=ID,
                scientific_protocol_manifest_sha256=m["scientific_protocol_manifest_sha256"],
                shared_initial_arrivals_maximum=1, initial_stage=initial_stage)


def validate_chunk(chunk, m, job, groups, ac_records):
    base.validate_b_chunk(chunk, m, job, groups)
    require(chunk.get("execution_id") == ID
            and chunk.get("scientific_protocol_manifest_sha256") == m["scientific_protocol_manifest_sha256"]
            and chunk.get("shared_initial_arrivals_maximum") == 1
            and chunk.get("initial_stage") in ("authenticated_shared_arrival_0", "equivalent_existing_replay",
                                                "mixed_validated_sources"),
            "Stable B execution provenance mismatch")
    base.validate_initial_agreement(chunk["records"], ac_records)


def adopt(path, expected, old_m, m, job, groups, ac_records):
    require(sha256(path) == expected, "Legacy B chunk changed before adoption")
    old = read(path)
    base.validate_b_chunk(old, old_m, job, groups)
    base.validate_initial_agreement(old["records"], ac_records)
    chunk = dict(**provenance(m, job, initial_stage="equivalent_existing_replay"),
                 seconds=old.get("seconds"), records=old["records"],
                 adopted_from=dict(path=base.source_key(path), sha256=expected,
                                   manifest_sha256=old_m["manifest_sha256"],
                                   records_sha256=base.digest(old["records"])))
    validate_chunk(chunk, m, job, groups, ac_records)
    return chunk


def require_no_other_worker():
    modules = {"braindecoding.fixed_k0_rolling_search_test_v2", f"braindecoding.{ID}",
               "braindecoding.c1_b_queue"}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        if process.pid == os.getpid() or not (process.info["name"] or "").lower().startswith("python"):
            continue
        cmdline = process.info["cmdline"] or []
        if any(module in cmdline for module in modules) and any(arg in cmdline for arg in ("run", "run-b")):
            raise ValueError(f"Another B worker is active: PID {process.pid}")


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        base.verify_identity(m)
        base.verify_hashes(m["source_sha256"])
        return m
    require_no_other_worker()
    old_m, parent = base.freeze(), read(base.PARENT)
    prior = read(base.RUNTIME / "status.json")
    require(prior["state"] == "failed" and "B and A disagree" in prior["error"],
            "The original first-arrival failure must be recorded before recovery")
    diagnostic = RUNTIME / "initial_diagnostic.json"
    evidence = read(diagnostic)
    require(evidence["manifest_sha256"] == old_m["manifest_sha256"] and evidence["arrival"] == 0
            and evidence["before_mismatches"] > 0 and evidence["saved_experiment_outputs_changed"] is False,
            "Authenticated first-arrival diagnostic is missing")
    sources = dict(old_m["source_sha256"])
    for path in (base.MANIFEST, PROTOCOL, Path(__file__), ROOT / f"tests/test_{ID}.py",
                 ROOT / "braindecoding/evaluation/single_revision_c1_test_v2_io_guard.py",
                 RUNTIME / "diagnose_initial.py", diagnostic):
        sources[base.source_key(path)] = sha256(path)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    snapshot = RUNTIME / "prior_failure.json"
    save(snapshot, prior)
    sources[base.source_key(snapshot)] = sha256(snapshot)
    jobs, adopted_units = copy.deepcopy(old_m["jobs"]), 0
    width = old_m["settings"]["execution"]["segment_batch_size"]
    for job in jobs:
        info = parent["datasets"][job["dataset"]]
        ac = base.original_ac(parent, job["dataset"], job["seed"], ROOT / job["original_ac"])
        job["legacy_chunks"] = []
        old_root = base.OUT / job["dataset"] / f"seed-{job['seed']:03d}"
        for number, path in enumerate(sorted((old_root / "chunks").glob("chunk-*.json"))):
            require(path.name == f"chunk-{number:04d}.json", "Legacy B chunks are not a contiguous prefix")
            start = number*width
            groups = info["groups"][start:start+width]
            chunk = read(path)
            base.validate_b_chunk(chunk, old_m, job, groups)
            base.validate_initial_agreement(chunk["records"], ac["records"][start:start+len(groups)])
            checksum = sha256(path)
            sources[base.source_key(path)] = checksum
            job["legacy_chunks"].append(dict(number=number, path=base.source_key(path), sha256=checksum))
            adopted_units += len(chunk["records"])
        if (old_root / "main.json").exists():
            sources[base.source_key(old_root / "main.json")] = sha256(old_root / "main.json")
    require(adopted_units == prior["completed_segment_seed_units"] == 7676,
            "Preserved B coverage differs from the failed worker")
    for dataset, info in parent["datasets"].items():
        dataset_jobs = [j for j in jobs if j["dataset"] == dataset]
        incomplete = any(len(j["legacy_chunks"])*width < len(info["groups"]) for j in dataset_jobs)
        if not incomplete:
            continue
        cached = base.OUT / dataset / "lm_cache.pkl"
        require(cached.exists(), "Resume language cache is missing")
        bootstrap = RUNTIME / "bootstrap" / f"{dataset}.pkl"
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, bootstrap)
        require(sha256(bootstrap) == sha256(cached), "Language cache snapshot differs")
        sources[base.source_key(bootstrap)] = sha256(bootstrap)
        sources[base.source_key(cached)] = sha256(cached)
        for job in dataset_jobs:
            job["bootstrap_cache"] = base.source_key(bootstrap)
    recovery = dict(method="reuse_authenticated_arrival_0_before_policy_divergence", use_reference_words=False,
                    use_A_future_arrivals=False, use_A_final=False, use_C=False, use_K1_K2=False,
                    first_independent_arrival=1, preserved_record_content=True,
                    diagnostic=base.source_key(diagnostic), missing_units=old_m["total_segment_seed_units"]-adopted_units)
    m = dict(protocol=ID, scientific_protocol_id=base.ID,
             scientific_protocol_manifest_sha256=old_m["manifest_sha256"],
             parent_manifest_sha256=old_m["parent_manifest_sha256"], frozen_unix=time.time(),
             settings=copy.deepcopy(old_m["settings"]), decoder=copy.deepcopy(old_m["decoder"]),
             evidence=old_m["evidence"], total_segment_seed_units=old_m["total_segment_seed_units"],
             adopted_segment_seed_units=adopted_units, jobs=jobs, source_sha256=sources,
             neural_input="K0_only", recovery=recovery)
    m["manifest_sha256"] = base.digest(m)
    base.verify_hashes(sources)
    save(MANIFEST, m)
    return m


def status(m, **fields):
    path = RUNTIME / "status.json"
    state = read(path) if path.exists() else {}
    now = time.time()
    state.update(pid=os.getpid(), process_created_unix=psutil.Process().create_time(), updated_unix=now,
                 manifest_sha256=m["manifest_sha256"], total_segment_seed_units=m["total_segment_seed_units"],
                 scientific_protocol_manifest_sha256=m["scientific_protocol_manifest_sha256"],
                 adopted_segment_seed_units=m["adopted_segment_seed_units"], **fields)
    save(path, state)
    refresh = now-LAST_PROGRESS["time"] >= 5 or any(state.get(k) != LAST_PROGRESS[k] for k in ("state", "stage"))
    if refresh:
        LAST_PROGRESS.update(time=now, state=state.get("state"), stage=state.get("stage"))
        lines = ["# 固定 K0 滚动重搜 B：恢复运行", "",
                 f"状态：{state.get('state')}；已保存 {state.get('completed_segment_seed_units', 0):,} / {m['total_segment_seed_units']:,} 个片段×种子。", "",
                 f"当前：{state.get('dataset', '')}，种子 {state.get('seed', '')}，分块 {state.get('chunk', '')}，到达 {state.get('arrival', '')}。", "",
                 f"保留原有 {m['adopted_segment_seed_units']:,} 个单元的完整记录；仅补剩余 {m['recovery']['missing_units']:,} 个。",
                 "仅共享原 A 的首次 K0 生成预测；从第二个词开始独立执行 B。始终 K0，最多重搜最近三个词，初始一致性与冻结前缀校验均保留。",
                 "冻结模型、搜索参数与统计协议不变。全部完成后导出 A→B、B→C 对照结果并释放 GPU。", "",
                 "[故障诊断与恢复验证](recovery_validation.md)"]
        if state.get("error"):
            lines += ["", f"错误：{state['error']}"]
        if state.get("state") == "completed":
            lines += ["", "[完整结果](results.md)"]
        try:
            REPORT.mkdir(parents=True, exist_ok=True)
            temporary = REPORT / "progress.md.tmp"
            temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
            atomic_replace_with_retry(temporary, REPORT / "progress.md")
        except OSError as error:
            print(json.dumps(dict(kind="optional_progress_write", error=str(error)), ensure_ascii=False), file=sys.stderr, flush=True)
    print(json.dumps({k: state[k] for k in ("state", "stage", "dataset", "seed", "chunk", "arrival",
                                          "completed_segment_seed_units") if k in state}), flush=True)


def verify_preserved(m):
    count = 0
    for job in m["jobs"]:
        for source in job["legacy_chunks"]:
            old_path = ROOT / source["path"]
            require(sha256(old_path) == source["sha256"], "Legacy B source changed during recovery")
            path = OUT / job["dataset"] / f"seed-{job['seed']:03d}" / "chunks" / f"chunk-{source['number']:04d}.json"
            old, new = read(old_path), read(path)
            require(old["records"] == new["records"]
                    and new["adopted_from"]["records_sha256"] == base.digest(old["records"]),
                    "Legacy B records changed during adoption")
            count += len(old["records"])
    require(count == m["adopted_segment_seed_units"], "Adopted B population differs")
    return count


def export(m, parent):
    from braindecoding import fixed_k0_rolling_search_test_v2_results as reporting
    preserved = verify_preserved(m)
    for job in m["jobs"]:
        info = parent["datasets"][job["dataset"]]
        payload = read(OUT / job["dataset"] / f"seed-{job['seed']:03d}" / "main.json")
        ac = base.original_ac(parent, job["dataset"], job["seed"], ROOT / job["original_ac"])
        validate_chunk(payload, m, job, info["groups"], ac["records"])
    previous = {key: getattr(reporting, key) for key in ("ID", "OUT", "REPORT")}
    try:
        reporting.ID, reporting.OUT, reporting.REPORT = ID, OUT, REPORT
        report = reporting.export(m, parent)
        report.update(scientific_protocol_manifest_sha256=m["scientific_protocol_manifest_sha256"],
                      execution_recovery=m["recovery"], preserved_segment_seed_units=preserved)
        save(REPORT / "results.json", report)
        summary = ROOT / "data/results/exports" / f"{ID}_summary.json"
        save(summary, report)
        markdown = REPORT / "results.md"
        title, remainder = markdown.read_text(encoding="utf-8").split("\n", 1)
        note = (f"\n执行修复：仅共享原 A 的首次 K0 生成预测；从第二个词独立进行 B 的最近三词重搜。"
                f"保留原有 {preserved:,} 个单元的全部记录，只补 {m['recovery']['missing_units']:,} 个。"
                "未使用参考词、后续 A/C 输出、K1/K2；模型、搜索和统计参数均未改变。\n")
        markdown.write_text(title+"\n"+note+remainder, encoding="utf-8")
        audit = read(REPORT / "audit.json")
        audit.update(results_sha256=sha256(summary), markdown_sha256=sha256(markdown),
                     scientific_protocol_manifest_sha256=m["scientific_protocol_manifest_sha256"],
                     execution_recovery=m["recovery"], preserved_segment_seed_units=preserved,
                     initial_agreement_verified=True, K0_only_verified=True, frozen_prefix_verified=True)
        save(REPORT / "audit.json", audit)
        return report
    finally:
        for key, value in previous.items():
            setattr(reporting, key, value)


@contextmanager
def worker_lock(path):
    import msvcrt
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        yield


def run():
    require_no_other_worker()
    m, parent, old_m = freeze(), read(base.PARENT), read(base.MANIFEST)
    previous = read(RUNTIME / "status.json") if (RUNTIME / "status.json").exists() else {}
    require(previous.get("state") != "completed", "Stable B is already complete")
    with worker_lock(base.RUNTIME / "worker.lock"), worker_lock(RUNTIME / "worker.lock"):
        original_replace = Path.replace
        Path.replace = atomic_replace_with_retry
        scorer = embed = bank = None
        completed, done = 0, []
        torch.set_num_threads(4)
        base.set_seed(0)
        status(m, state="running", stage="adopting_existing_results", error=None,
               completed_segment_seed_units=m["adopted_segment_seed_units"],
               started_unix=time.time(), completed_datasets=[], worker_exited=False,
               GPU_models_released=False, dataset=None, seed=None, chunk=None, arrival=None)
        try:
            for dataset in m["settings"]["protocol"]["datasets"]:
                info = parent["datasets"][dataset]
                cache_path = OUT / dataset / "lm_cache.pkl"
                for job in (job for job in m["jobs"] if job["dataset"] == dataset):
                    k0, ids = base.load_k0(ROOT / job["prediction"])
                    ac = read(ROOT / job["original_ac"])
                    root = OUT / dataset / f"seed-{job['seed']:03d}"
                    width, groups = m["settings"]["execution"]["segment_batch_size"], info["groups"]
                    legacy = {item["number"]: item for item in job["legacy_chunks"]}
                    records, sources = [], []
                    for start in range(0, len(groups), width):
                        number, batch = start//width, groups[start:start+width]
                        originals = ac["records"][start:start+len(batch)]
                        path = root / "chunks" / f"chunk-{number:04d}.json"
                        if path.exists():
                            chunk = read(path)
                            validate_chunk(chunk, m, job, batch, originals)
                        elif number in legacy:
                            source = legacy[number]
                            chunk = adopt(ROOT / source["path"], source["sha256"], old_m, m, job, batch, originals)
                            save(path, chunk)
                        else:
                            if scorer is None:
                                status(m, stage="loading_sentence_models", dataset=dataset, seed=job["seed"],
                                       chunk=number, arrival=None, completed_segment_seed_units=completed)
                                scorer, embed, bank = base.load_lm(parent, dataset)
                                scorer.load_cache(cache_path if cache_path.exists() else ROOT / job["bootstrap_cache"])
                            started, last = time.perf_counter(), [0.]
                            def progress(arrival, counts):
                                now = time.perf_counter()
                                if now-last[0] >= m["settings"]["execution"]["progress_interval_seconds"]:
                                    status(m, stage="B_decoding", dataset=dataset, seed=job["seed"], chunk=number,
                                           arrival=arrival, longest_segment=max(counts), chunk_seconds=now-started,
                                           completed_segment_seed_units=completed)
                                    last[0] = now
                            cases = base.k0_cases(batch, k0, ids)
                            d = m["decoder"]
                            results = replay_shared_initial(cases, shared_initial(batch, originals),
                                                            info["vocabulary"], bank, scorer, embed, progress=progress,
                                                            candidate_count=d["candidates"], beam_width=d["beam_width"],
                                                            neural_scale=d["neural_scale"])
                            chunk = dict(**provenance(m, job), seconds=time.perf_counter()-started,
                                         records=[dict(group=g["group"], result=result) for g,result in zip(batch,results)])
                            validate_chunk(chunk, m, job, batch, originals)
                            save(path, chunk)
                            scorer.save_cache(cache_path)
                        records.extend(chunk["records"])
                        sources.append(dict(path=base.source_key(path), sha256=sha256(path)))
                        completed += len(batch)
                        status(m, stage="chunk_complete" if scorer is not None else "adopting_existing_results",
                               dataset=dataset, seed=job["seed"], chunk=number, arrival=None,
                               completed_segment_seed_units=max(completed, m["adopted_segment_seed_units"]))
                    payload = dict(**provenance(m, job, initial_stage="mixed_validated_sources"),
                                   status="completed", records=records, sources=sources)
                    validate_chunk(payload, m, job, groups, ac["records"])
                    save(root / "main.json", payload)
                    del k0, ids, ac
                scorer = embed = bank = None
                gc.collect()
                torch.cuda.empty_cache()
                done.append(dataset)
                status(m, stage="dataset_complete", completed_datasets=list(done),
                       completed_segment_seed_units=max(completed, m["adopted_segment_seed_units"]))
            require(completed == m["total_segment_seed_units"], "B completion population differs")
            status(m, stage="exporting", completed_segment_seed_units=completed, GPU_models_released=True)
            export(m, parent)
            status(m, state="completed", stage="completed", completed_segment_seed_units=completed,
                   completed_unix=time.time(), GPU_models_released=True)
        except BaseException as error:
            scorer = embed = bank = None
            gc.collect()
            torch.cuda.empty_cache()
            status(m, state="failed", stage="failed", error=f"{type(error).__name__}: {error}",
                   completed_segment_seed_units=max(completed, m["adopted_segment_seed_units"]),
                   GPU_models_released=True)
            raise
        finally:
            Path.replace = original_replace


def check():
    state = read(RUNTIME / "status.json")
    alive = False
    try:
        process = psutil.Process(state["pid"])
        alive = (f"braindecoding.{ID}" in process.cmdline()
                 and abs(process.create_time()-state["process_created_unix"]) < 1.)
    except psutil.NoSuchProcess:
        pass
    state = dict(state, worker_alive=alive, worker_exited=not alive)
    if not alive and state["state"] not in ("completed", "failed"):
        state.update(state="failed", error="B worker exited before successful completion; do not restart automatically")
    if state["state"] == "completed" and not alive:
        m = freeze()
        audit, report = read(REPORT / "audit.json"), read(REPORT / "results.json")
        require(audit["status"] == "verified_complete" and audit["manifest_sha256"] == m["manifest_sha256"]
                and report["segment_seed_units"] == audit["segment_seed_units"] == m["total_segment_seed_units"]
                and audit["original_AC_reproduced"] and audit["initial_agreement_verified"]
                and sha256(REPORT / "results.json") == audit["results_sha256"]
                and sha256(REPORT / "results.md") == audit["markdown_sha256"], "Stable B final audit differs")
        verify_preserved(m)
    print(json.dumps(state), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "check"))
    args = parser.parse_args()
    if args.command == "prepare":
        m = freeze()
        print(json.dumps(dict(manifest_sha256=m["manifest_sha256"], preserved=m["adopted_segment_seed_units"],
                              remaining=m["recovery"]["missing_units"])), flush=True)
    elif args.command == "run":
        run()
    else:
        check()


if __name__ == "__main__":
    main()
