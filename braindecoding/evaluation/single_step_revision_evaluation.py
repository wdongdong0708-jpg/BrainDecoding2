"""C1: K0 first output, one K1 revision at the next arrival, then freeze."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from braindecoding.fixed_k0_rolling_search_test_v2 import (
    ROOT, PARENT, ORIGINAL_SUMMARY, PARENT_OUT, original_ac,
    require, source_key, verify_hashes, verify_identity,
)
from braindecoding.training.runtime import set_seed
from braindecoding.unified_sentence_batch import search_many
from braindecoding.unified_sentence_v2 import digest, load_lm, read, save, sha256


ID = "single_revision_c1_test_v2"
CONFIG = ROOT / "configs/evaluation/single_step_revision_evaluation.yaml"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
OUT = ROOT / f"outputs/heldout_test/{ID}"
REPORT = ROOT / f"data/results/studies/{ID}"
RUNTIME = ROOT / f".cache/runs/{ID}"


def load_k01(path):
    """Never read K2, including on resume or a cache-validation pass."""
    with np.load(path, allow_pickle=False) as archive:
        return {h: archive[f"k{h}"] for h in (0, 1)}, archive["event_ids"], archive["mask1"]


def c1_cases(groups, predictions, event_ids, mask1, *, weight=.1):
    require(set(predictions) == {0, 1}, "C1 requires only K0/K1")
    ids = list(map(str, event_ids))
    require(len(set(ids)) == len(ids), "Duplicate neural events")
    require(all(np.asarray(q).ndim == 2 and len(q) == len(ids)
                and np.isfinite(q).all() for q in predictions.values()), "Invalid neural vectors")
    require(predictions[0].shape == predictions[1].shape and np.asarray(mask1).shape == (len(ids),),
            "C1 horizon or mask shapes differ")
    lookup = {event: i for i, event in enumerate(ids)}
    cases = []
    for group in groups:
        require(group["event_ids"] and all(e in lookup for e in group["event_ids"]),
                "C1 group is absent from neural cache")
        selected = [lookup[e] for e in group["event_ids"]]
        require(np.asarray(mask1)[selected[:-1]].all(), "K1 unavailable inside a C1 segment")
        cases.append(dict(predictions={h: predictions[h][selected] for h in (0, 1)},
                          weight=weight, open_vocabulary=True))
    return cases


def replay_c1_many(cases, vocabulary, bank, scorer, embed_words, *, progress=None, **search_settings):
    """Two-word searches, each earlier output permanently frozen after K1."""
    counts = []
    for case in cases:
        require(set(case["predictions"]) == {0, 1}, "C1 requires only K0/K1")
        k0, k1 = (np.asarray(case["predictions"][h]) for h in (0, 1))
        require(k0.ndim == 2 and k0.shape == k1.shape and len(k0) > 0,
                "C1 horizon shapes differ or empty segment")
        counts.append(len(k0))
    outputs, traces = [() for _ in cases], [[] for _ in cases]
    for arrival in range(max(counts, default=0)):
        active = [i for i, count in enumerate(counts) if arrival < count]
        start = max(0, arrival - 1)
        requests = [dict(
            queries=np.stack([cases[i]["predictions"][arrival-pos][pos]
                              for pos in range(start, arrival+1)]),
            prefix=outputs[i][:start], weight=cases[i]["weight"],
            open_vocabulary=cases[i]["open_vocabulary"],
        ) for i in active]
        results = search_many(requests, vocabulary, bank, scorer, embed_words, **search_settings)
        require(len(results) == len(active), "C1 search population differs")
        for i, request, result in zip(active, requests, results):
            require(tuple(result[:start]) == request["prefix"] and len(result) == arrival+1,
                    "C1 changed a frozen prefix or known word count")
            outputs[i] = tuple(result)
            traces[i].append(dict(arrival=arrival, C1=list(result)))
        if progress:
            progress(arrival, counts)
    return [dict(final=dict(C1=list(words)), trace=trace) for words, trace in zip(outputs, traces)]


def validate_c1_result(group, result):
    count = len(group["event_ids"])
    require(count > 0 and set(result) == {"final", "trace"}, "C1 output schema differs")
    require(set(result["final"]) == {"C1"} and len(result["final"]["C1"]) == count,
            "C1 final population differs")
    require(len(result["trace"]) == count, "C1 arrival population changed or extra end flush")
    previous = []
    changes = np.zeros(count, dtype=int)
    for arrival, step in enumerate(result["trace"]):
        require(set(step) == {"arrival", "C1"} and step["arrival"] == arrival,
                "C1 arrival order or condition changed")
        words = step["C1"]
        require(len(words) == arrival+1 and all(isinstance(w, str) and w for w in words),
                "C1 trace word count or words changed")
        start = max(0, arrival-1)
        require(words[:start] == previous[:start], "C1 changed a frozen prefix")
        for position in range(arrival):
            changes[position] += words[position] != previous[position]
        previous = words
    require(previous == result["final"]["C1"], "C1 final differs from last arrival")
    require((changes <= 1).all(), "C1 revised a word more than once")


def validate_c1_chunk(chunk, m, job, groups):
    expected = dict(manifest_sha256=m["manifest_sha256"], dataset=job["dataset"], seed=job["seed"],
                    conditions=["C1"], split="test", test_evaluated=True, neural_representation="K0_K1_only",
                    maximum_revision_opportunities_per_word=1, maximum_wait_subsequent_word_windows=1,
                    revision_words=2, branch="open", alignment="real", selected_weight=.1)
    require(all(chunk.get(k) == v for k, v in expected.items()), "C1 chunk provenance mismatch")
    require([r["group"] for r in chunk["records"]] == [g["group"] for g in groups],
            "C1 chunk population incomplete, duplicated or reordered")
    for group, record in zip(groups, chunk["records"]):
        validate_c1_result(group, record["result"])


def validate_initial_agreement(records, ac_records):
    require(len(records) == len(ac_records), "C1/A/C initial population differs")
    for new, old in zip(records, ac_records):
        require(new["group"] == old["group"], "C1/A/C groups differ")
        trace, ac_trace = new["result"]["trace"], old["result"]["trace"]
        require(trace[0]["C1"] == ac_trace[0]["A"] == ac_trace[0]["C"],
                "C1/A/C initial output disagrees")
        if len(trace) > 1:
            require(trace[1]["C1"] == ac_trace[1]["C"], "C1/C first K1 revision disagrees")


def provenance(m, job):
    return dict(manifest_sha256=m["manifest_sha256"], dataset=job["dataset"], seed=job["seed"],
                conditions=["C1"], split="test", test_evaluated=True, neural_representation="K0_K1_only",
                maximum_revision_opportunities_per_word=1, maximum_wait_subsequent_word_windows=1,
                revision_words=2, branch="open", alignment="real", selected_weight=.1)


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        verify_identity(m)
        verify_hashes(m["source_sha256"])
        return m
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p, d = cfg["protocol"], cfg["decoding"]
    require(p["split"] == "test" and p["seeds"] == [0, 1, 2]
            and p["datasets"] == ["pallier2025", "chineseeeg2_littleprince"]
            and not p["training_allowed"] and not p["recalibration_allowed"], "C1 protocol changed")
    require(d["compute_conditions"] == ["C1"] and d["report_conditions"] == ["A", "C1", "C"]
            and d["maximum_revision_opportunities_per_word"] == d["maximum_wait_subsequent_word_windows"] == 1
            and d["revision_words"] == 2 and d["read_K2"] is False
            and d["language_weight"] == .1 and d["branch"] == "open" and d["alignment"] == "real"
            and d["end_of_segment"] == "freeze_last_word_at_K0_without_extra_arrival", "C1 timing contract changed")
    require(p["parent"] == source_key(PARENT), "C1 parent changed")
    parent, original = read(PARENT), read(ORIGINAL_SUMMARY)
    verify_identity(parent)
    require(original["status"] == "completed_heldout_test"
            and original["manifest_sha256"] == parent["manifest_sha256"], "Original A/C summary changed")
    require(cfg["execution"]["segment_batch_size"] == parent["decoder"]["segment_batch_size"],
            "C1 batch size differs from original")
    sources = {source_key(path): expected for path, expected in parent["source_sha256"].items()}
    for path in (CONFIG, PROTOCOL, PARENT, ORIGINAL_SUMMARY, Path(__file__),
                 ROOT / f"braindecoding/evaluation/{ID}_results.py", ROOT / "braindecoding/evaluation/fixed_k0_rolling_search_test_v2.py",
                 ROOT / "braindecoding/evaluation/fixed_k0_rolling_search_test_v2_results.py",
                 ROOT / "braindecoding/decoding/jayalath_sentence.py", ROOT / "braindecoding/decoding/unified_sentence_v2_artifacts.py",
                 ROOT / f"tests/test_{ID}.py"):
        sources[source_key(path)] = sha256(path)
    from huggingface_hub import snapshot_download
    text = parent["neural_settings"]["text_embedding"]
    snapshot = Path(snapshot_download(text["model_name"], revision=text["revision"], local_files_only=True))
    for path in snapshot.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".txt", ".model", ".safetensors", ".bin"):
            sources[source_key(path)] = sha256(path)
    jobs = []
    for dataset in p["datasets"]:
        info = parent["datasets"][dataset]
        expected = [e for g in info["groups"] for e in g["event_ids"]]
        require(len(expected) == len(set(expected)) == info["counts"]["events"], "Frozen events changed")
        require(len(info["groups"]) == info["counts"]["groups"] and info["selected_weight"] == .1,
                "Frozen groups or weight changed")
        cache = PARENT_OUT / dataset / "lm_cache.pkl"
        require(cache.exists(), "Original language cache is missing")
        sources[source_key(cache)] = sha256(cache)
        for seed in p["seeds"]:
            root = PARENT_OUT / dataset / f"seed-{seed:03d}"
            prediction, side_path, ac_path = root / "neural_predictions.npz", root / "word_metrics.json", root / "main.json"
            side = read(side_path)
            checkpoint = next(j for j in parent["checkpoints"] if (j["dataset"], j["seed"]) == (dataset, seed))
            require(side["dataset"] == dataset and side["seed"] == seed and side["split"] == "test"
                    and side["test_evaluated"] and side["manifest_sha256"] == parent["manifest_sha256"]
                    and side["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
                    and side["prediction_sha256"] == sha256(prediction), "K0/K1 cache provenance changed")
            predictions, ids, mask = load_k01(prediction)
            require(len(ids) == len(expected) and set(map(str, ids)) == set(expected), "Neural events differ")
            for h in (0, 1):
                require(predictions[h].shape == (len(expected), text["embedding_dimension"])
                        and np.isfinite(predictions[h]).all()
                        and np.allclose(np.linalg.norm(predictions[h], axis=1), 1., atol=.005),
                        "C1 neural embeddings invalid")
            c1_cases(info["groups"], predictions, ids, mask)
            ac = original_ac(parent, dataset, seed, ac_path)
            for path in (prediction, side_path, ac_path):
                if Path(path).is_file():
                    sources[source_key(path)] = sha256(path)
            for item in ac["sources"]:
                sources[item["path"]] = item["sha256"]
            jobs.append(dict(dataset=dataset, seed=seed, prediction=source_key(prediction),
                             original_ac=source_key(ac_path), parent_cache=source_key(cache)))
            del predictions, ids, mask, ac
    verify_hashes(sources)
    decoder = copy.deepcopy(parent["decoder"])
    decoder.update(revision_words=2, conditions=["C1"])
    m = dict(protocol=ID, frozen_unix=time.time(), settings=cfg, decoder=decoder, jobs=jobs,
             parent_manifest_sha256=parent["manifest_sha256"], source_sha256=sources,
             total_segment_seed_units=sum(i["counts"]["groups"] for i in parent["datasets"].values())*3,
             neural_input="K0_K1_only", maximum_wait_subsequent_word_windows=1,
             maximum_revision_opportunities_per_word=1, evidence=cfg["statistics"]["evidence"])
    m["manifest_sha256"] = digest(m)
    save(MANIFEST, m)
    print(json.dumps(dict(stage="frozen", manifest_sha256=m["manifest_sha256"], jobs=len(jobs),
                          total=m["total_segment_seed_units"])), flush=True)
    return m


def status(m, **fields):
    path = RUNTIME / "status.json"
    state = read(path) if path.exists() else {}
    state.update(pid=os.getpid(), updated_unix=time.time(), manifest_sha256=m["manifest_sha256"],
                 total_segment_seed_units=m["total_segment_seed_units"], **fields)
    save(path, state)
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = ["# C1：一次回修、最多一个后续词窗", "",
             f"状态：{state.get('state')}；已保存 {state.get('completed_segment_seed_units', 0):,} / {m['total_segment_seed_units']:,} 个片段×种子。", "",
             f"当前：{state.get('dataset', '')}，种子 {state.get('seed', '')}，分块 {state.get('chunk', '')}，到达词位 {state.get('arrival', '')}。", "",
             "C1 首次使用 K0；下一词到达时用 K1 回修一次后冻结。末词保持 K0，无额外结束回放。",
             "完整六个任务完成后报告 A、C1、C，配对材料块区间及一次回修保留的增益比例。"]
    if state.get("error"):
        lines += ["", f"错误：{state['error']}"]
    if state.get("state") == "completed":
        lines += ["", "[完整结果](results.md)"]
    temporary = REPORT / "progress.md.tmp"
    temporary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    temporary.replace(REPORT / "progress.md")
    print(json.dumps({k: state[k] for k in ("state", "stage", "dataset", "seed", "chunk", "arrival",
                                          "completed_segment_seed_units") if k in state}), flush=True)


def run():
    m, parent = freeze(), read(PARENT)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (RUNTIME / "worker.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        torch.set_num_threads(4)
        set_seed(0)
        status(m, state="running", stage="starting", started_unix=time.time(), error=None,
               completed_segment_seed_units=0, completed_datasets=[])
        try:
            completed, done = 0, []
            for dataset in m["settings"]["protocol"]["datasets"]:
                info = parent["datasets"][dataset]
                scorer = embed = bank = None
                cache_path = OUT / dataset / "lm_cache.pkl"
                for job in (j for j in m["jobs"] if j["dataset"] == dataset):
                    predictions, ids, mask = load_k01(ROOT / job["prediction"])
                    ac = read(ROOT / job["original_ac"])
                    root = OUT / dataset / f"seed-{job['seed']:03d}"
                    records, sources = [], []
                    width = m["settings"]["execution"]["segment_batch_size"]
                    groups = info["groups"]
                    for start in range(0, len(groups), width):
                        batch, number = groups[start:start+width], start // width
                        path = root / "chunks" / f"chunk-{number:04d}.json"
                        if path.exists():
                            chunk = read(path)
                            validate_c1_chunk(chunk, m, job, batch)
                        else:
                            if scorer is None:
                                status(m, stage="loading_sentence_models", dataset=dataset, seed=job["seed"],
                                       chunk=number, arrival=None, completed_segment_seed_units=completed)
                                scorer, embed, bank = load_lm(parent, dataset)
                                scorer.load_cache(cache_path if cache_path.exists() else ROOT / job["parent_cache"])
                            started, last = time.perf_counter(), [0.]
                            def progress(arrival, counts):
                                now = time.perf_counter()
                                if now-last[0] >= m["settings"]["execution"]["progress_interval_seconds"]:
                                    status(m, stage="C1_decoding", dataset=dataset, seed=job["seed"], chunk=number,
                                           arrival=arrival, longest_segment=max(counts), chunk_seconds=now-started,
                                           completed_segment_seed_units=completed)
                                    last[0] = now
                            cases = c1_cases(batch, predictions, ids, mask)
                            d = m["decoder"]
                            results = replay_c1_many(cases, info["vocabulary"], bank, scorer, embed,
                                                    progress=progress, candidate_count=d["candidates"],
                                                    beam_width=d["beam_width"], neural_scale=d["neural_scale"])
                            chunk = dict(**provenance(m, job), seconds=time.perf_counter()-started,
                                         records=[dict(group=g["group"], result=r) for g, r in zip(batch, results)])
                            validate_c1_chunk(chunk, m, job, batch)
                            validate_initial_agreement(chunk["records"], ac["records"][start:start+len(batch)])
                            save(path, chunk)
                            scorer.save_cache(cache_path)
                        validate_initial_agreement(chunk["records"], ac["records"][start:start+len(batch)])
                        records.extend(chunk["records"])
                        sources.append(dict(path=source_key(path), sha256=sha256(path)))
                        completed += len(batch)
                        status(m, stage="chunk_complete", dataset=dataset, seed=job["seed"], chunk=number,
                               arrival=None, completed_segment_seed_units=completed)
                    payload = dict(**provenance(m, job), status="completed", records=records, sources=sources)
                    validate_c1_chunk(payload, m, job, groups)
                    save(root / "main.json", payload)
                    del predictions, ids, mask, ac
                del scorer, embed, bank
                gc.collect()
                torch.cuda.empty_cache()
                done.append(dataset)
                status(m, stage="dataset_complete", completed_datasets=done, completed_segment_seed_units=completed)
            require(completed == m["total_segment_seed_units"], "C1 completion population differs")
            status(m, stage="exporting", completed_segment_seed_units=completed)
            export(m, parent)
            status(m, state="completed", stage="completed", completed_segment_seed_units=completed,
                   completed_unix=time.time(), GPU_models_released=True)
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
        export(freeze(), read(PARENT))

# --- Merged from single_revision_c1_test_v2_results.py ---
"""Paired A/C1/C accuracy, repair and structural latency on frozen tests."""

from collections import defaultdict
import time

import numpy as np

from braindecoding.fixed_k0_rolling_search_test_v2_results import (
    summarize, gain_decomposition, paired_statistics, change_rows, summarize_changes,
)
from braindecoding.unified_sentence_v2_artifacts import csv_write
from braindecoding.unified_sentence_v2_results import METRICS, holm, metric_row


CONDITIONS = ("A", "C1", "C")
NAMES = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}


def renamed_condition(row):
    return {**row, "condition": "C1" if row["condition"] == "B" else row["condition"]}


def summarize_c1(rows):
    # Reuse the authenticated subject/seed and word-pooling implementation.
    canonical = [{**row, "condition": "B" if row["condition"] == "C1" else row["condition"]} for row in rows]
    aggregate = summarize(canonical)
    gains = gain_decomposition(aggregate["summary"])
    statistics_rows = canonical
    aggregate = {key: [renamed_condition(row) for row in values] for key, values in aggregate.items()}
    renamed_gains = {metric: dict(
        one_window_gain_C1_minus_A=g["search_gain_B_minus_A"],
        second_window_gain_C_minus_C1=g["neural_update_gain_C_minus_B"],
        total_gain_C_minus_A=g["total_gain_C_minus_A"],
        one_window_retained_share=g["search_share"],
        second_window_increment_share=g["neural_update_share"],
        gain_direction=g["gain_direction"], share_interpretation=g["share_interpretation"],
    ) for metric, g in gains.items()}
    return aggregate, renamed_gains, statistics_rows


def c1_change_rows(group, ac_result, c1_result, identity):
    canonical = dict(final=dict(B=c1_result["final"]["C1"]),
                     trace=[dict(arrival=s["arrival"], B=s["C1"]) for s in c1_result["trace"]])
    mapping = {"A_to_B": "A_to_C1", "B_to_C": "C1_to_C", "B_first_to_final": "C1_first_to_final"}
    return [{**r, "comparison": mapping.get(r["comparison"], r["comparison"])}
            for r in change_rows(group, ac_result, canonical, identity)]


def latency_rows(group, ac_result, c1_result, identity):
    count = len(group["event_ids"])
    rows = []
    for condition, result, maximum in (("C1", c1_result, 1), ("C", ac_result, 2)):
        for position, event in enumerate(group["event_ids"]):
            freeze_at = min(position+maximum, count-1)
            first = result["trace"][position][condition][position]
            final = result["final"][condition][position]
            changes = sum(result["trace"][arrival][condition][position]
                          != result["trace"][arrival-1][condition][position]
                          for arrival in range(position+1, count))
            require(all(result["trace"][arrival][condition][position] == final
                        for arrival in range(freeze_at, count)), "Output changed after scheduled freeze")
            require(changes <= freeze_at-position <= maximum, "Revision/wait exceeds condition budget")
            rows.append(dict(**identity, condition=condition, event_id=event, position=position,
                             first_arrival=position, freeze_arrival=freeze_at,
                             wait_subsequent_word_windows=freeze_at-position,
                             allowed_revision_opportunities=freeze_at-position,
                             actual_text_changes=changes, first_prediction=first, final_prediction=final,
                             reference=group["reference"][position]))
    return rows


def summarize_latency(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[row["condition"]].append(row)
    return [dict(condition=condition, word_position_seed_decisions=len(group),
                 max_wait_subsequent_word_windows=max(r["wait_subsequent_word_windows"] for r in group),
                 max_allowed_revision_opportunities=max(r["allowed_revision_opportunities"] for r in group),
                 max_actual_text_changes=max(r["actual_text_changes"] for r in group),
                 wait_0=sum(r["wait_subsequent_word_windows"] == 0 for r in group),
                 wait_1=sum(r["wait_subsequent_word_windows"] == 1 for r in group),
                 wait_2=sum(r["wait_subsequent_word_windows"] == 2 for r in group),
                 mean_wait_subsequent_word_windows=float(np.mean([r["wait_subsequent_word_windows"] for r in group])),
                 latency_unit="subsequent_word_windows_not_seconds")
            for condition, group in sorted(cells.items())]


def export(m, parent):
    verify_hashes(m["source_sha256"])
    original = read(ORIGINAL_SUMMARY)
    datasets, all_rows, all_subjects, all_seeds, all_gains = {}, [], [], [], []
    all_changes, all_change_subjects, all_latency, sources = [], [], [], []
    for dataset in m["settings"]["protocol"]["datasets"]:
        info = parent["datasets"][dataset]
        rows, changes, latencies = [], [], []
        jobs = [j for j in m["jobs"] if j["dataset"] == dataset]
        require(sorted(j["seed"] for j in jobs) == [0, 1, 2], "Missing C1 seed")
        for job in jobs:
            path = OUT / dataset / f"seed-{job['seed']:03d}" / "main.json"
            new = read(path)
            require(new["status"] == "completed", "Cannot export incomplete C1 seed")
            validate_c1_chunk(new, m, job, info["groups"])
            reconstructed = []
            for item in new["sources"]:
                chunk_path = ROOT / item["path"]
                require(sha256(chunk_path) == item["sha256"], "Completed C1 chunk changed")
                chunk = read(chunk_path)
                start = len(reconstructed)
                validate_c1_chunk(chunk, m, job, info["groups"][start:start+len(chunk["records"])])
                reconstructed.extend(chunk["records"])
            require(reconstructed == new["records"], "C1 chunks differ from saved output")
            ac = original_ac(parent, dataset, job["seed"], ROOT / job["original_ac"])
            validate_initial_agreement(new["records"], ac["records"])
            for group, old, record in zip(info["groups"], ac["records"], new["records"]):
                identity = dict(dataset=dataset, seed=job["seed"], subject=group["subject"], group=group["group"],
                                material=group["material"], block=group["block"], scope=group["scope"])
                for condition in CONDITIONS:
                    prediction = (record["result"]["final"]["C1"] if condition == "C1"
                                  else old["result"]["final"][condition])
                    rows.append(dict(**identity, condition=condition, words=len(group["reference"]),
                                     **metric_row(group["reference"], prediction, info["language"] == "zh")))
                changes.extend(c1_change_rows(group, old["result"], record["result"], identity))
                latencies.extend(latency_rows(group, old["result"], record["result"], identity))
            sources.append(dict(path=source_key(path), sha256=sha256(path), chunks=new["sources"]))
        aggregate, gains, canonical = summarize_c1(rows)
        for field in ("summary", "by_seed"):
            keys = ("condition",) if field == "summary" else ("seed", "condition")
            actual = {tuple(row[k] for k in keys): row for row in aggregate[field]}
            for expected in original["datasets"][dataset][field]:
                value = actual[tuple(expected[k] for k in keys)]
                require(all(np.isclose(value[k], expected[k], rtol=0, atol=1e-12) for k in METRICS),
                        "Original A/C metrics no longer reproduced")
        raw_statistics = paired_statistics(canonical, m["settings"]["statistics"])
        statistics = {"C1_minus_A": raw_statistics["B_minus_A"], "C_minus_C1": raw_statistics["C_minus_B"]}
        for name, key in (("C1_minus_A", "one_window_gain_C1_minus_A"),
                          ("C_minus_C1", "second_window_gain_C_minus_C1")):
            require(np.isclose(statistics[name]["effect"], gains["sentence_bleu1_mean"][key],
                               rtol=0, atol=1e-12), "C1 paired effect differs from aggregate")
        transitions, change_subjects = summarize_changes(changes)
        transitions_by_name = {r["comparison"]: r for r in transitions}
        require(transitions_by_name["A_to_C1"]["net_correct_words"]
                + transitions_by_name["C1_to_C"]["net_correct_words"]
                == transitions_by_name["A_to_C"]["net_correct_words"], "Correct-word gains do not add up")
        units = info["counts"]["groups"]*3
        require(len(rows) == units*3 and len(changes) == units*5
                and len(latencies) == info["counts"]["events"]*3*2, "Dataset C1 coverage differs")
        datasets[dataset] = dict(counts=info["counts"], **aggregate, gains=gains,
                                supplementary_statistics=statistics, transitions=transitions,
                                latency=summarize_latency(latencies),
                                original_C_minus_A_statistics=original["datasets"][dataset]["primary_statistics"])
        for seed in (0, 1, 2):
            _, seed_gains, _ = summarize_c1([row for row in rows if row["seed"] == seed])
            all_gains.extend(dict(dataset=dataset, seed=seed, metric=metric, **gain)
                             for metric, gain in seed_gains.items())
        all_rows.extend(rows)
        all_changes.extend(changes)
        all_latency.extend(latencies)
        all_subjects.extend(dict(dataset=dataset, **row) for row in aggregate["by_subject_seed"])
        all_seeds.extend(dict(dataset=dataset, **row) for row in aggregate["by_seed"])
        all_change_subjects.extend(dict(dataset=dataset, **row) for row in change_subjects)
    require(len(all_rows) == m["total_segment_seed_units"]*3
            and len(all_changes) == m["total_segment_seed_units"]*5, "Complete C1 coverage differs")
    stats = [s for info in datasets.values() for s in info["supplementary_statistics"].values()]
    require(len(stats) == 4, "C1 requires four supplementary comparisons")
    for statistic, adjusted in zip(stats, holm([s["p_one_sided"] for s in stats])):
        statistic.update(p_holm=float(adjusted), family="two_datasets_by_two_C1_supplementary_comparisons")
    exports = {}
    for suffix, records in (("by_group", all_rows), ("by_subject_seed", all_subjects), ("by_seed", all_seeds),
                            ("gain_by_seed", all_gains), ("transitions_by_group", all_changes),
                            ("transitions_by_subject_seed", all_change_subjects), ("latency_by_word", all_latency)):
        path = ROOT / "data/results/exports" / f"{ID}_{suffix}.csv"
        csv_write(path, records)
        exports[source_key(path)] = sha256(path)
    report = dict(status="completed", protocol=ID, manifest_sha256=m["manifest_sha256"],
                  parent_manifest_sha256=m["parent_manifest_sha256"], split="test", test_evaluated=True,
                  conditions=list(CONDITIONS), computed_conditions=["C1"], seeds=[0, 1, 2],
                  segment_seed_units=m["total_segment_seed_units"], evidence=m["evidence"], datasets=datasets,
                  original_AC_reproduced=True, original_AC_unchanged=True, neural_models_unchanged=True,
                  C1_neural_input="K0_K1_only", maximum_revision_opportunities_per_word=1,
                  maximum_wait_subsequent_word_windows=1, sources=sources, exports=exports, completed_unix=time.time(),
                  interpretation="C1-A: one-window policy gain; C-C1: added second-window policy gain; neural horizon, search window and generated prefix interact.")
    verify_hashes(m["source_sha256"])
    for source in sources:
        require(sha256(ROOT / source["path"]) == source["sha256"], "C1 output changed during export")
        verify_hashes({item["path"]: item["sha256"] for item in source["chunks"]})
    save(REPORT / "results.json", report)
    summary_path = ROOT / "data/results/exports" / f"{ID}_summary.json"
    save(summary_path, report)
    write_markdown(report)
    save(REPORT / "audit.json", dict(status="verified_complete", manifest_sha256=m["manifest_sha256"],
         results_sha256=sha256(summary_path), markdown_sha256=sha256(REPORT / "results.md"),
         exports=exports, sources=sources, source_sha256=m["source_sha256"],
         segment_seed_units=m["total_segment_seed_units"], original_AC_reproduced=True,
         maximum_wait_verified=True, maximum_revision_count_verified=True))
    return report


def write_markdown(report):
    lines = ["# C1：一次回修、最多一个后续词窗——完整测试结果", "",
             f"两个数据集、三个种子、全部 {report['segment_seed_units']:,} 个片段×种子已完成。C1 独立重放两词窗口；原 A/C 汇总及逐种子指标逐项复现。", "",
             "| 数据集 | A BLEU-1（%） | C1 BLEU-1（%） | C BLEU-1（%） | C1−A（百分点） | C−C1（百分点） | 一窗保留总增益 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, info in report["datasets"].items():
        v = {r["condition"]: r for r in info["summary"]}
        g = info["gains"]["sentence_bleu1_mean"]
        share = "不定义" if g["one_window_retained_share"] is None else f"{g['one_window_retained_share']*100:.1f}%"
        lines.append(f"| {NAMES[name]} | {v['A']['sentence_bleu1_mean']*100:.2f} | {v['C1']['sentence_bleu1_mean']*100:.2f} | {v['C']['sentence_bleu1_mean']*100:.2f} | {g['one_window_gain_C1_minus_A']*100:+.2f} | {g['second_window_gain_C_minus_C1']*100:+.2f} | {share} |")
    lines += ["", "| 数据集 | 比较 | 95% 配对材料块区间（百分点） | 补充检验 Holm p |",
              "|---|---|---|---:|"]
    for name, info in report["datasets"].items():
        for comparison, stat in info["supplementary_statistics"].items():
            lines.append(f"| {NAMES[name]} | {comparison} | [{stat['ci_lower']*100:+.2f}, {stat['ci_upper']*100:+.2f}] | {stat['p_holm']:.6f} |")
    lines += ["", "主指标先受试者内平均片段，再对受试者及种子等权；材料块在记录／章节内分层重采样 10,000 次。"
              "四项新增检验共同 Holm 校正，种子不作为独立受试者。原 A/C 主检验未改变。", "",
              "C1 首次用 K0，下一词到达时用 K1 回修一次后冻结；末词无后续窗、保持 K0。"
              "原 C 允许两次回修、最多等待两个后续词窗。占比为带符号的描述性差值分解，"
              "不截断至 0–100%；神经时域、搜索窗口、候选与自身生成前缀交互仍然存在。", "",
              "| 数据集 | 策略 | WER（%） | 对齐词准确率（%） | 按词数加权 BLEU-1（%） |",
              "|---|---|---:|---:|---:|"]
    for name, info in report["datasets"].items():
        for row in info["summary"]:
            lines.append(f"| {NAMES[name]} | {row['condition']} | {row['wer']*100:.2f} | {row['aligned_word_accuracy']*100:.2f} | {row['bleu1']*100:.2f} |")
    lines += ["", "| 数据集 | 种子 | A BLEU-1（%） | C1 BLEU-1（%） | C BLEU-1（%） |",
              "|---|---:|---:|---:|---:|"]
    for name, info in report["datasets"].items():
        for seed in report["seeds"]:
            v = {r["condition"]: r for r in info["by_seed"] if r["seed"] == seed}
            lines.append(f"| {NAMES[name]} | {seed} | {v['A']['sentence_bleu1_mean']*100:.2f} | {v['C1']['sentence_bleu1_mean']*100:.2f} | {v['C']['sentence_bleu1_mean']*100:.2f} |")
    lines += ["", "| 数据集 | 词位变化 | 可评价决策 | 改对 | 误改 | 净增正确词 | 改对率（%） | 误改率（%） |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, info in report["datasets"].items():
        for row in info["transitions"]:
            lines.append(f"| {NAMES[name]} | {row['comparison']} | {row['eligible']:,} | {row['repaired']:,} | {row['harmed']:,} | {row['net_correct_words']:+,} | {row['repaired_rate']*100:.2f} | {row['harmed_rate']*100:.2f} |")
    lines += ["", "计数单位为词位×种子决策；比例先受试者内合并词位，再对受试者、种子等权。在线首次→最终修正排除片段末词。", "",
              "| 数据集 | 策略 | 最大等待窗 | 最大回修机会 | 最大实际文本修改次数 | 无等待词位 | 等待一窗词位 | 等待两窗词位 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, info in report["datasets"].items():
        for row in info["latency"]:
            lines.append(f"| {NAMES[name]} | {row['condition']} | {row['max_wait_subsequent_word_windows']} | {row['max_allowed_revision_opportunities']} | {row['max_actual_text_changes']} | {row['wait_0']:,} | {row['wait_1']:,} | {row['wait_2']:,} |")
    lines += ["", "等待单位为后续词窗，非秒数；无等待词位为片段末词。逐词位首次预测、最终预测、冻结到达、等待窗数和实际修改次数见 latency_by_word CSV。", "",
              f"冻结清单 SHA-256：`{report['manifest_sha256']}`。", ""]
    (REPORT / "results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
