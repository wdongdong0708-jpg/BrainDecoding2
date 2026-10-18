"""Resume C1 with authenticated shared arrivals 0/1, then independent K1 replay."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from braindecoding import single_revision_c1_test_v2 as base
from braindecoding.unified_sentence_batch import search_many


ROOT = base.ROOT
ID = "single_revision_c1_test_v2_stable"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
OUT = ROOT / f"outputs/heldout_test/{ID}"
REPORT = ROOT / f"data/results/studies/{ID}"
RUNTIME = ROOT / f".cache/runs/{ID}"
require, read, save, sha256 = base.require, base.read, base.save, base.sha256


def shared_initial(groups, ac_records):
    """Only generated words from arrivals 0/1 cross this boundary."""
    require(len(groups) == len(ac_records), "Shared initial population differs")
    output = []
    for group, record in zip(groups, ac_records):
        require(group["group"] == record["group"], "Shared initial group differs")
        count = len(group["event_ids"])
        traces = record["result"]["trace"]
        entries = []
        for arrival in range(min(2, count)):
            step = traces[arrival]
            require(step["arrival"] == arrival, "Shared initial arrival order differs")
            entries.append(dict(arrival=arrival, words=list(step["C"])))
        output.append(entries)
    return output


def replay_shared_initial(cases, initial, vocabulary, bank, scorer, embed_words, *, progress=None, **settings):
    require(len(cases) == len(initial), "C1 shared-start population differs")
    counts, outputs, traces = [], [], []
    for case, entries in zip(cases, initial):
        require(set(case["predictions"]) == {0, 1}, "C1 requires only K0/K1")
        k0, k1 = (np.asarray(case["predictions"][h]) for h in (0, 1))
        require(k0.ndim == 2 and k0.shape == k1.shape and len(k0) > 0, "C1 horizon shapes differ")
        count = len(k0)
        require(len(entries) == min(count, 2), "Only actual initial arrivals 0/1 may be reused")
        copied = []
        for arrival, entry in enumerate(entries):
            require(set(entry) == {"arrival", "words"} and entry["arrival"] == arrival,
                    "Shared initial entry contains undeclared or future information")
            require(len(entry["words"]) == arrival+1 and all(isinstance(w, str) and w for w in entry["words"]),
                    "Shared initial words differ")
            copied.append(dict(arrival=arrival, C1=list(entry["words"])))
        counts.append(count)
        traces.append(copied)
        outputs.append(tuple(copied[-1]["C1"]))
    for arrival in range(2, max(counts, default=0)):
        active = [i for i, n in enumerate(counts) if arrival < n]
        start = arrival-1
        requests = [dict(queries=np.stack([cases[i]["predictions"][1][start],
                                          cases[i]["predictions"][0][arrival]]),
                         prefix=outputs[i][:start], weight=cases[i]["weight"],
                         open_vocabulary=cases[i]["open_vocabulary"]) for i in active]
        results = search_many(requests, vocabulary, bank, scorer, embed_words, **settings)
        require(len(results) == len(active), "C1 resumed search population differs")
        for i, request, result in zip(active, requests, results):
            require(tuple(result[:start]) == request["prefix"] and len(result) == arrival+1,
                    "C1 changed a frozen prefix or known word count")
            outputs[i] = tuple(result)
            traces[i].append(dict(arrival=arrival, C1=list(result)))
        if progress:
            progress(arrival, counts)
    return [dict(final=dict(C1=list(words)), trace=trace) for words, trace in zip(outputs, traces)]


def provenance(m, job, *, adopted=False):
    return dict(**base.provenance(m, job), execution_id=ID,
                scientific_protocol_manifest_sha256=m["scientific_protocol_manifest_sha256"],
                shared_initial_arrivals_maximum=2,
                initial_stage="equivalent_existing_replay" if adopted else "authenticated_shared_arrivals")


def validate_chunk(chunk, m, job, groups, ac_records):
    base.validate_c1_chunk(chunk, m, job, groups)
    require(chunk.get("execution_id") == ID
            and chunk.get("scientific_protocol_manifest_sha256") == m["scientific_protocol_manifest_sha256"]
            and chunk.get("shared_initial_arrivals_maximum") == 2
            and chunk.get("initial_stage") in ("equivalent_existing_replay", "authenticated_shared_arrivals"),
            "Stable C1 execution provenance mismatch")
    base.validate_initial_agreement(chunk["records"], ac_records)


def adopt(path, expected, old_m, m, job, groups, ac_records):
    require(sha256(path) == expected, "Legacy C1 chunk changed before adoption")
    old = read(path)
    base.validate_c1_chunk(old, old_m, job, groups)
    base.validate_initial_agreement(old["records"], ac_records)
    chunk = dict(**provenance(m, job, adopted=True), seconds=old.get("seconds"), records=old["records"],
                 adopted_from=dict(path=base.source_key(path), sha256=expected,
                                   manifest_sha256=old_m["manifest_sha256"],
                                   records_sha256=base.digest(old["records"])))
    validate_chunk(chunk, m, job, groups, ac_records)
    return chunk


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        base.verify_identity(m)
        base.verify_hashes(m["source_sha256"])
        return m
    old_m = base.freeze()
    parent = read(base.PARENT)
    old_status = read(base.RUNTIME / "status.json")
    require(old_status["state"] == "failed", "Old C1 worker must have stopped before recovery")
    sources = dict(old_m["source_sha256"])
    for path in (base.MANIFEST, PROTOCOL, Path(__file__), ROOT / f"tests/test_{ID}.py"):
        sources[base.source_key(path)] = sha256(path)
    diagnostic = base.RUNTIME / "recovery/chunk-0022-two-arrival-diagnostic.json"
    require(diagnostic.exists(), "Numerical mismatch diagnostic is missing")
    sources[base.source_key(diagnostic)] = sha256(diagnostic)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    failed_copy = RUNTIME / "prior_failure.json"
    save(failed_copy, old_status)
    sources[base.source_key(failed_copy)] = sha256(failed_copy)
    jobs, adopted_units = copy.deepcopy(old_m["jobs"]), 0
    for dataset, info in parent["datasets"].items():
        cached = base.OUT / dataset / "lm_cache.pkl"
        if not cached.exists():
            cached = ROOT / next(j["parent_cache"] for j in jobs if j["dataset"] == dataset)
        snapshot = RUNTIME / "bootstrap" / f"{dataset}.pkl"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, snapshot)
        require(sha256(snapshot) == sha256(cached), "Language cache snapshot changed")
        sources[base.source_key(snapshot)] = sha256(snapshot)
        for job in (j for j in jobs if j["dataset"] == dataset):
            ac = base.original_ac(parent, dataset, job["seed"], ROOT / job["original_ac"])
            job.update(bootstrap_cache=base.source_key(snapshot), legacy_chunks=[])
            old_root = base.OUT / dataset / f"seed-{job['seed']:03d}" / "chunks"
            width = old_m["settings"]["execution"]["segment_batch_size"]
            for number, path in enumerate(sorted(old_root.glob("chunk-*.json"))):
                require(path.name == f"chunk-{number:04d}.json", "Legacy C1 chunks are not a contiguous prefix")
                start = number*width
                groups = info["groups"][start:start+width]
                chunk = read(path)
                base.validate_c1_chunk(chunk, old_m, job, groups)
                base.validate_initial_agreement(chunk["records"], ac["records"][start:start+len(groups)])
                checksum = sha256(path)
                sources[base.source_key(path)] = checksum
                job["legacy_chunks"].append(dict(number=number, path=base.source_key(path), sha256=checksum))
                adopted_units += len(chunk["records"])
    require(adopted_units == old_status["completed_segment_seed_units"], "Preserved C1 coverage differs")
    base.verify_hashes(sources)
    m = dict(protocol=ID, scientific_protocol_id=base.ID,
             scientific_protocol_manifest_sha256=old_m["manifest_sha256"],
             parent_manifest_sha256=old_m["parent_manifest_sha256"], frozen_unix=time.time(),
             settings=copy.deepcopy(old_m["settings"]), decoder=copy.deepcopy(old_m["decoder"]),
             evidence=old_m["evidence"], total_segment_seed_units=old_m["total_segment_seed_units"],
             adopted_segment_seed_units=adopted_units, jobs=jobs, source_sha256=sources,
             recovery=dict(method="reuse_authenticated_arrivals_0_1_before_policy_divergence",
                           use_reference_words=False, use_C_final=False, use_K2=False,
                           first_independent_arrival=2, preserved_record_content=True,
                           diagnostic=base.source_key(diagnostic)))
    m["manifest_sha256"] = base.digest(m)
    save(MANIFEST, m)
    print(json.dumps(dict(stage="frozen_recovery", manifest_sha256=m["manifest_sha256"],
                          preserved_units=adopted_units, total=m["total_segment_seed_units"])), flush=True)
    return m


def status(m, **fields):
    path = RUNTIME / "status.json"
    state = read(path) if path.exists() else {}
    state.update(pid=os.getpid(), updated_unix=time.time(), manifest_sha256=m["manifest_sha256"],
                 total_segment_seed_units=m["total_segment_seed_units"],
                 adopted_segment_seed_units=m["adopted_segment_seed_units"], **fields)
    save(path, state)
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = ["# C1 稳定执行：一次回修、最多一个后续词窗", "",
             f"状态：{state.get('state')}；已保存 {state.get('completed_segment_seed_units', 0):,} / {m['total_segment_seed_units']:,} 个片段×种子。", "",
             f"当前：{state.get('dataset', '')}，种子 {state.get('seed', '')}，分块 {state.get('chunk', '')}，到达 {state.get('arrival', '')}。", "",
             f"保留并迁入原有 {m['adopted_segment_seed_units']:,} 个单元，记录内容不变。到达 0/1 复用原 C 的同一 K0/K1 阶段；从到达 2 开始独立执行 C1。",
             "单次回修、冻结前缀、最多等待一窗及初始一致性校验均保留。完整六个任务完成后自动统计并释放 GPU。"]
    if state.get("error"):
        lines += ["", f"错误：{state['error']}"]
    if state.get("state") == "completed":
        lines += ["", "[完整结果](results.md)"]
    temporary = REPORT / "progress.md.tmp"
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temporary.replace(REPORT / "progress.md")
    print(json.dumps({k: state[k] for k in ("state", "stage", "dataset", "seed", "chunk", "arrival",
                                          "completed_segment_seed_units") if k in state}), flush=True)


def export(m, parent):
    from braindecoding import single_revision_c1_test_v2_results as reporting
    for job in m["jobs"]:
        info = parent["datasets"][job["dataset"]]
        chunk = read(OUT / job["dataset"] / f"seed-{job['seed']:03d}" / "main.json")
        ac = base.original_ac(parent, job["dataset"], job["seed"], ROOT / job["original_ac"])
        validate_chunk(chunk, m, job, info["groups"], ac["records"])
    original_globals = {k: getattr(reporting, k) for k in ("ID", "OUT", "REPORT")}
    try:
        reporting.ID, reporting.OUT, reporting.REPORT = ID, OUT, REPORT
        report = reporting.export(m, parent)
        report.update(scientific_protocol_manifest_sha256=m["scientific_protocol_manifest_sha256"],
                      execution_recovery=m["recovery"], adopted_segment_seed_units=m["adopted_segment_seed_units"])
        save(REPORT / "results.json", report)
        summary = ROOT / "data/results/exports" / f"{ID}_summary.json"
        save(summary, report)
        markdown = REPORT / "results.md"
        text = markdown.read_text(encoding="utf-8")
        title, remainder = text.split("\n", 1)
        note = (f"\n执行修复：共享策略分岔前的到达 0/1，仅复用原 C 的 K0/K1 生成输出；从到达 2 独立执行 C1。"
                f"保留 {m['adopted_segment_seed_units']:,} 个旧单元的记录内容。未使用参考词、原 C 最终输出或 K2。\n")
        markdown.write_text(title+"\n"+note+remainder, encoding="utf-8")
        audit_path = REPORT / "audit.json"
        audit = read(audit_path)
        audit.update(results_sha256=sha256(summary), markdown_sha256=sha256(markdown),
                     scientific_protocol_manifest_sha256=m["scientific_protocol_manifest_sha256"],
                     execution_recovery=m["recovery"], adopted_segment_seed_units=m["adopted_segment_seed_units"])
        save(audit_path, audit)
        return report
    finally:
        for key, value in original_globals.items():
            setattr(reporting, key, value)


def run():
    m, parent, old_m = freeze(), read(base.PARENT), read(base.MANIFEST)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (RUNTIME / "worker.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        torch.set_num_threads(4)
        base.set_seed(0)
        status(m, state="running", stage="starting", error=None, completed_segment_seed_units=0,
               started_unix=time.time(), completed_datasets=[])
        completed, done = 0, []
        try:
            for dataset in m["settings"]["protocol"]["datasets"]:
                info = parent["datasets"][dataset]
                scorer = embed = bank = None
                cache_path = OUT / dataset / "lm_cache.pkl"
                for job in (j for j in m["jobs"] if j["dataset"] == dataset):
                    predictions, ids, mask = base.load_k01(ROOT / job["prediction"])
                    ac = read(ROOT / job["original_ac"])
                    root = OUT / dataset / f"seed-{job['seed']:03d}"
                    legacy = {item["number"]: item for item in job["legacy_chunks"]}
                    records, sources = [], []
                    width = m["settings"]["execution"]["segment_batch_size"]
                    groups = info["groups"]
                    for start in range(0, len(groups), width):
                        group_batch, number = groups[start:start+width], start // width
                        ac_batch = ac["records"][start:start+len(group_batch)]
                        path = root / "chunks" / f"chunk-{number:04d}.json"
                        if path.exists():
                            chunk = read(path)
                        elif number in legacy:
                            item = legacy[number]
                            chunk = adopt(ROOT / item["path"], item["sha256"], old_m, m, job, group_batch, ac_batch)
                            save(path, chunk)
                        else:
                            if scorer is None:
                                status(m, stage="loading_sentence_models", dataset=dataset, seed=job["seed"], chunk=number,
                                       arrival=None, completed_segment_seed_units=completed)
                                scorer, embed, bank = base.load_lm(parent, dataset)
                                scorer.load_cache(cache_path if cache_path.exists() else ROOT / job["bootstrap_cache"])
                            started, last = time.perf_counter(), [0.]
                            def progress(arrival, counts):
                                now = time.perf_counter()
                                if now-last[0] >= m["settings"]["execution"]["progress_interval_seconds"]:
                                    status(m, stage="C1_decoding", dataset=dataset, seed=job["seed"], chunk=number,
                                           arrival=arrival, completed_segment_seed_units=completed,
                                           longest_segment=max(counts), chunk_seconds=now-started)
                                    last[0] = now
                            cases = base.c1_cases(group_batch, predictions, ids, mask)
                            d = m["decoder"]
                            output = replay_shared_initial(cases, shared_initial(group_batch, ac_batch),
                                info["vocabulary"], bank, scorer, embed, progress=progress,
                                candidate_count=d["candidates"], beam_width=d["beam_width"], neural_scale=d["neural_scale"])
                            chunk = dict(**provenance(m, job), seconds=time.perf_counter()-started,
                                         records=[dict(group=g["group"], result=r) for g, r in zip(group_batch, output)])
                            validate_chunk(chunk, m, job, group_batch, ac_batch)
                            save(path, chunk)
                            scorer.save_cache(cache_path)
                        validate_chunk(chunk, m, job, group_batch, ac_batch)
                        records.extend(chunk["records"])
                        sources.append(dict(path=base.source_key(path), sha256=sha256(path)))
                        completed += len(group_batch)
                        status(m, stage="chunk_complete", dataset=dataset, seed=job["seed"], chunk=number,
                               arrival=None, completed_segment_seed_units=completed)
                    payload = dict(**provenance(m, job), status="completed", records=records, sources=sources,
                                   adopted_segment_seed_units=sum(len(read(ROOT / x['path'])['records']) for x in job['legacy_chunks']))
                    validate_chunk(payload, m, job, groups, ac["records"])
                    save(root / "main.json", payload)
                    del predictions, ids, mask, ac
                del scorer, embed, bank
                gc.collect()
                torch.cuda.empty_cache()
                done.append(dataset)
                status(m, stage="dataset_complete", completed_datasets=done, completed_segment_seed_units=completed)
            require(completed == m["total_segment_seed_units"], "Recovered C1 population differs")
            status(m, stage="exporting", completed_segment_seed_units=completed)
            export(m, parent)
            status(m, state="completed", stage="completed", completed_unix=time.time(), GPU_models_released=True)
        except BaseException as error:
            status(m, state="failed", stage="failed", error=f"{type(error).__name__}: {error}")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "export"))
    args = parser.parse_args(argv)
    if args.command == "prepare":
        freeze()
    elif args.command == "run":
        run()
    else:
        export(freeze(), read(base.PARENT))


if __name__ == "__main__":
    main()
