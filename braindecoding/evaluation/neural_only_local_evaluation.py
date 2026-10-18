"""Inference-only F-local: mature shared rolling vectors, original F search."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from braindecoding.config import PROJECT_ROOT as ROOT
from braindecoding.unified_sentence_v2 import digest, read, sha256, load_lm
from braindecoding.unified_sentence_batch import search_many
from braindecoding.fixed_k0_rolling_search_test_v2 import original_ac
from braindecoding.full_context_sentence_test_v2 import validate_chunk as validate_f_chunk

ID = "f_local_sentence_test_v2"
CONDITIONS = ("C", "F-local", "F")
CONFIG = ROOT / f"configs/unified_rolling/{ID}.yaml"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
PARENT = ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"
FULL = ROOT / "data/manifests/rolling_sentence/full_context_sentence_test_v2.json"
F_SUMMARY = ROOT / "data/results/studies/full_context_sentence_test_v2/results.json"
F_WAIT = ROOT / "data/results/exports/full_context_sentence_test_v2_input_wait_by_event.csv"
OUT = ROOT / f"outputs/heldout_test/{ID}"
REPORT = ROOT / f"data/results/studies/{ID}"
EXPORT = ROOT / "data/results/exports"
RUNTIME = ROOT / f".cache/runs/{ID}"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_key(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(12):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(.25)


def verify_identity(m):
    require(digest({k: v for k, v in m.items() if k != "manifest_sha256"}) == m["manifest_sha256"],
            "Manifest identity differs")


def verify_sources(m, full=False):
    verify_identity(m)
    for name, expected in m["source_sha256"].items():
        if str(name).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        path = ROOT / name
        stamp = m["source_stat"][name]
        actual = path.stat()
        require(actual.st_size == stamp["bytes"] and actual.st_mtime_ns == stamp["mtime_ns"],
                f"Frozen source metadata changed: {name}")
        if full or path.suffix in (".py", ".yaml", ".md"):
            require(sha256(path) == expected, f"Frozen source changed: {name}")


def config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p, d, r = cfg["protocol"], cfg["decoding"], cfg["representation"]
    require(p["id"] == ID and p["split"] == "test" and p["seeds"] == [0, 1, 2], "Wrong F-local population")
    require(p["datasets"] == ["pallier2025", "chineseeeg2_littleprince"], "F-local datasets differ")
    require(not any(p[k] for k in ("training_allowed", "recalibration_allowed", "selection_allowed")),
            "F-local must not train, recalibrate or select")
    require(p["parent"] == source_key(PARENT) and p["full_context"] == source_key(FULL), "Parents differ")
    require(r == dict(source="existing_C_shared_rolling_predictions", horizon="min_2_remaining_words_in_same_segment",
                      segment_tail="K1_then_K0"), "Shared final horizon changed")
    require(d["compute_conditions"] == ["F-local"] and d["report_conditions"] == list(CONDITIONS)
            and d["language_weight"] == .1 and d["branch"] == "open" and d["alignment"] == "real"
            and d["search"] == "single_full_segment_beam_from_empty_generated_prefix", "Search contract changed")
    require(cfg["statistics"]["significance_tests"] is False and cfg["statistics"]["material_block_size"] == 5,
            "Supplementary confidence interval contract changed")
    return cfg


def decoder_input(parent, dataset):
    """The model loader receives no references or material metadata."""
    info = parent["datasets"][dataset]
    return dict(neural_settings=copy.deepcopy(parent["neural_settings"]),
                settings=dict(decoding=copy.deepcopy(parent["settings"]["decoding"])),
                datasets={dataset: {k: copy.deepcopy(info[k]) for k in ("language", "vocabulary", "embedding_cache")}})


def load_neural(path):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"event_ids", "k0", "k1", "k2", "mask0", "mask1", "mask2"},
                "Unexpected shared neural archive")
        ids = list(map(str, archive["event_ids"]))
        require(len(ids) == len(set(ids)), "Duplicate neural event identities")
        predictions = {h: archive[f"k{h}"].copy() for h in range(3)}
        masks = {h: archive[f"mask{h}"].copy() for h in range(3)}
    return predictions, masks, {e: i for i, e in enumerate(ids)}


def final_queries(group, predictions, masks, lookup):
    ids = [lookup[e] for e in group["event_ids"]]
    n = len(ids)
    require(n > 0 and len(ids) == len(set(ids)), "Empty or duplicate segment")
    positions = np.arange(n)
    for h in range(3):
        require(np.array_equal(masks[h][ids], positions + h < n), "Horizon mask crosses segment boundary")
    horizons = np.minimum(2, n - 1 - positions)
    queries = np.stack([predictions[int(h)][idx] for h, idx in zip(horizons, ids)])
    require(queries.ndim == 2 and queries.shape == (n, 1024) and np.isfinite(queries).all(),
            "Non-finite or malformed final shared vectors")
    require(np.allclose(np.linalg.norm(queries, axis=1), 1., atol=.005, rtol=0), "Unnormalized shared vectors")
    return queries, horizons.tolist()


def audit_neural(path, info):
    predictions, masks, lookup = load_neural(path)
    expected = [e for g in info["groups"] for e in g["event_ids"]]
    require(len(expected) == len(set(expected)) == info["counts"]["events"] and set(expected) == set(lookup),
            "Neural cache does not match the entire frozen test population")
    for h in range(3):
        require(predictions[h].shape == (len(expected), 1024) and masks[h].shape == (len(expected),)
                and masks[h].dtype == np.bool_, "Shared neural archive dimensions differ")
    counts, checksum = [0, 0, 0], hashlib.sha256()
    for group in info["groups"]:
        q, horizons = final_queries(group, predictions, masks, lookup)
        checksum.update(np.ascontiguousarray(q).tobytes())
        for h in horizons:
            counts[h] += 1
    return dict(events=len(expected), groups=len(info["groups"]), final_horizon_counts=counts,
                final_queries_sha256=checksum.hexdigest(), mask_segment_agreement=True,
                all_selected_vectors_finite_and_normalized=True)


def validate_cache(path, language):
    with Path(path).open("rb") as handle:
        actual, scores, proposals = pickle.load(handle)
    require(actual == language and isinstance(scores, dict) and isinstance(proposals, dict), "Language cache differs")
    require(all(np.isfinite(v) for v in scores.values()), "Language cache contains non-finite word scores")
    return dict(language=actual, score_entries=len(scores), proposal_entries=len(proposals))


def snapshot_cache(dataset, language):
    origin = ROOT / f"outputs/heldout_test/full_context_sentence_test_v2/{dataset}/lm_cache.pkl"
    target = OUT / "inputs" / dataset / "f_language_cache.pkl"
    expected = sha256(origin)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".copying")
        shutil.copyfile(origin, temporary)
        temporary.replace(target)
    require(sha256(target) == expected, "Private F cache snapshot differs from origin")
    return dict(path=source_key(target), sha256=expected, origin=source_key(origin),
                origin_sha256=expected, **validate_cache(target, language))


def original_f(parent, full, dataset, seed, path):
    info = parent["datasets"][dataset]
    payload = read(path)
    require(payload["status"] == "completed", "Original F is incomplete")
    validate_f_chunk(payload, full, dict(dataset=dataset, seed=seed), info["groups"])
    reconstructed = []
    for source in payload["sources"]:
        chunk = ROOT / source["path"]
        require(sha256(chunk) == source["sha256"], "Original F chunk changed")
        value = read(chunk)
        start = len(reconstructed)
        validate_f_chunk(value, full, dict(dataset=dataset, seed=seed),
                         info["groups"][start:start + len(value["records"])])
        reconstructed.extend(value["records"])
    require(reconstructed == payload["records"], "Original F atomic outputs differ")
    return payload


def code_paths():
    return [Path(__file__), ROOT / f"braindecoding/evaluation/{ID}_results.py", CONFIG, PROTOCOL,
            ROOT / f"tests/test_{ID}.py", ROOT / "braindecoding/evaluation/fixed_k0_rolling_search_test_v2.py",
            ROOT / "braindecoding/evaluation/full_context_sentence_test_v2.py", ROOT / "braindecoding/evaluation/unified_sentence_v2_results.py",
            ROOT / "braindecoding/decoding/unified_sentence_v2_artifacts.py", ROOT / "braindecoding/decoding/jayalath_sentence.py"]


def freeze():
    if MANIFEST.exists():
        value = read(MANIFEST)
        verify_sources(value, full=True)
        return value
    cfg, parent, full, summary = config(), read(PARENT), read(FULL), read(F_SUMMARY)
    verify_identity(parent)
    verify_identity(full)
    require(full["parent_manifest_sha256"] == parent["manifest_sha256"] and summary["status"] == "completed"
            and summary["manifest_sha256"] == full["manifest_sha256"], "Original C/F provenance differs")
    for key in ("candidates", "beam_width", "neural_scale", "proposal_count", "proposal_max_new_tokens",
                "proposal_batch_size", "score_batch_size", "score_cache_size", "segment_batch_size"):
        require(cfg["decoding"][key] == full["settings"]["decoder"][key] == parent["settings"]["decoding"][key],
                f"F search parameter differs: {key}")
    require(cfg["execution"]["segment_batch_size"] == full["settings"]["segment_batch_size"] == 16,
            "F segment batching differs")
    smoke = read(RUNTIME / "gpu_smoke.json")
    require(smoke["status"] == "passed" and smoke["source_sha256"] == {source_key(p): sha256(p) for p in code_paths()},
            "Real-model engineering checks are absent or stale")
    sources = {source_key(p): expected for p, expected in full["source_sha256"].items()}

    def bind(path, expected=None):
        name = source_key(path)
        actual = sha256(ROOT / name)
        if expected is not None:
            require(actual == expected, f"Input provenance changed: {name}")
        if name in sources:
            require(sources[name] == actual, f"Original F source changed: {name}")
        sources[name] = actual
        return name

    for path in code_paths() + [PARENT, FULL, F_SUMMARY, F_WAIT, RUNTIME / "gpu_smoke.json"]:
        bind(path)
    original_summary = ROOT / "data/results/exports/unified_heldout_test_v2_sentence_summary.json"
    bind(original_summary)
    caches, jobs = {}, []
    for dataset in cfg["protocol"]["datasets"]:
        info = parent["datasets"][dataset]
        require(info["selected_weight"] == .1 and len(info["groups"]) == info["counts"]["groups"],
                "Frozen groups or language weight differ")
        caches[dataset] = snapshot_cache(dataset, info["language"])
        bind(caches[dataset]["path"], caches[dataset]["sha256"])
        bind(caches[dataset]["origin"], caches[dataset]["origin_sha256"])
        bind(info["embedding_cache"])
        for seed in cfg["protocol"]["seeds"]:
            cp = next(j for j in parent["checkpoints"] if (j["dataset"], j["seed"]) == (dataset, seed))
            folder = ROOT / f"outputs/heldout_test/unified_heldout_test_v2/{dataset}/seed-{seed:03d}"
            prediction, metrics = folder / "neural_predictions.npz", folder / "word_metrics.json"
            side = read(metrics)
            require(side["dataset"] == dataset and side["seed"] == seed and side["split"] == "test"
                    and side["test_evaluated"] and side["manifest_sha256"] == parent["manifest_sha256"]
                    and side["checkpoint_sha256"] == cp["checkpoint_sha256"], "Shared neural cache provenance differs")
            bind(prediction, side["prediction_sha256"])
            bind(cp["checkpoint"], cp["checkpoint_sha256"])
            bind(metrics)
            event_table = bind(cp["config"]["cache"]["event_table"])
            ac_path = folder / "main.json"
            f_path = ROOT / f"outputs/heldout_test/full_context_sentence_test_v2/{dataset}/seed-{seed:03d}/main.json"
            ac = original_ac(parent, dataset, seed, ac_path)
            f = original_f(parent, full, dataset, seed, f_path)
            for payload in (ac, f):
                for source in payload["sources"]:
                    bind(source["path"], source["sha256"])
            jobs.append(dict(dataset=dataset, seed=seed, prediction=source_key(prediction),
                prediction_sha256=side["prediction_sha256"], checkpoint=source_key(cp["checkpoint"]),
                checkpoint_sha256=cp["checkpoint_sha256"], original_ac=bind(ac_path), original_f=bind(f_path),
                event_table=event_table, neural_audit=audit_neural(prediction, info)))
    for name, expected in sources.items():
        require(sha256(ROOT / name) == expected, f"Frozen source changed: {name}")
    value = dict(protocol=ID, authorization="explicit_user_requested_F_local_shared_K2_whole_segment_search",
                 frozen_unix=time.time(), settings=cfg, parent_manifest_sha256=parent["manifest_sha256"],
                 full_manifest_sha256=full["manifest_sha256"], jobs=jobs, initial_language_cache=caches,
                 source_sha256=sources, source_stat={name: dict(bytes=(ROOT / name).stat().st_size,
                     mtime_ns=(ROOT / name).stat().st_mtime_ns) for name in sources},
                 total_segment_seed_units=sum(len(i["groups"]) for i in parent["datasets"].values()) * 3,
                 original_C_F_read_only=True, inference_only=True)
    value["manifest_sha256"] = digest(value)
    write_json(MANIFEST, value)
    print(json.dumps(dict(stage="frozen", manifest_sha256=value["manifest_sha256"],
                          total=value["total_segment_seed_units"], source_files=len(sources))), flush=True)
    return value


class CheckedScorer:
    def __init__(self, scorer):
        self.scorer = scorer

    def __getattr__(self, name):
        return getattr(self.scorer, name)

    def score_extensions(self, prefixes, words):
        values = self.scorer.score_extensions(prefixes, words)
        require(np.asarray(values).shape == (len(words),) and np.isfinite(values).all(), "Non-finite LM word score")
        return values


def models(parent, dataset):
    scorer, embed, bank = load_lm(decoder_input(parent, dataset), dataset)
    require(bank.shape == (len(parent["datasets"][dataset]["vocabulary"]), 1024) and np.isfinite(bank).all(),
            "Malformed training embedding bank")

    def checked_embed(words):
        values = embed(words)
        require(values.shape == (len(words), 1024) and np.isfinite(values).all(), "Non-finite proposed word embeddings")
        return values

    return CheckedScorer(scorer), checked_embed, bank


def decode(groups, predictions, masks, lookup, info, scorer, embed, bank, settings):
    selected = [final_queries(g, predictions, masks, lookup) for g in groups]
    requests = [dict(queries=q, prefix=(), weight=settings["language_weight"], open_vocabulary=True) for q, _ in selected]
    answers = search_many(requests, info["vocabulary"], bank, scorer, embed,
                          candidate_count=settings["candidates"], beam_width=settings["beam_width"],
                          neural_scale=settings["neural_scale"])
    require(len(answers) == len(groups), "Full-segment search population differs")
    records = []
    for group, answer, (_, horizons) in zip(groups, answers, selected):
        require(len(answer) == len(group["event_ids"]) and all(isinstance(w, str) and w for w in answer),
                "Full-segment search changed known word count")
        records.append(dict(group=group["group"], prediction=list(answer), horizons=horizons))
    return records


def gpu_ready(policy, update=None):
    while True:
        free, total = torch.cuda.mem_get_info()
        if free >= policy["minimum_free_gpu_memory_gib"] * 1024**3:
            break
        if update:
            update(stage="waiting_gpu_memory", free_gpu_memory_gib=free / 1024**3)
        time.sleep(15)
    torch.cuda.set_per_process_memory_fraction(policy["gpu_memory_limit_gib"] * 1024**3 / total)


def smoke():
    cfg, parent = config(), read(PARENT)
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.manual_seed(0)
    np.random.seed(0)
    gpu_ready(cfg["execution"])
    records = []
    for dataset in cfg["protocol"]["datasets"]:
        info = parent["datasets"][dataset]
        candidates = [g for g in info["groups"] if 3 <= len(g["event_ids"]) <= 4]
        require(bool(candidates), "No bounded real-model smoke segment")
        group = candidates[0]
        arrays = load_neural(ROOT / f"outputs/heldout_test/unified_heldout_test_v2/{dataset}/seed-000/neural_predictions.npz")
        snapshot = snapshot_cache(dataset, info["language"])
        scorer, embed, bank = models(parent, dataset)
        scorer.load_cache(ROOT / snapshot["path"])
        started = time.perf_counter()
        answer = decode([group], *arrays, info, scorer, embed, bank, cfg["decoding"])[0]
        records.append(dict(dataset=dataset, group=group["group"], words=len(answer["prediction"]),
                            horizons=answer["horizons"], prediction=answer["prediction"],
                            seconds=time.perf_counter() - started,
                            peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
                            empty_generated_prefix=True, reference_supplied_to_decoder=False))
        del scorer, embed, bank, arrays
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps(dict(stage="gpu_smoke", dataset=dataset, status="passed", words=len(answer["prediction"]))), flush=True)
    value = dict(status="passed", datasets=records, source_sha256={source_key(p): sha256(p) for p in code_paths()},
                 neural_training=False, existing_models_only=True, completed_unix=time.time())
    write_json(RUNTIME / "gpu_smoke.json", value)
    return value


def plan(m, parent, dataset):
    width = m["settings"]["execution"]["segment_batch_size"]
    groups = parent["datasets"][dataset]["groups"]
    tasks = []
    for job in [j for j in m["jobs"] if j["dataset"] == dataset]:
        for start in range(0, len(groups), width):
            number = start // width
            path = OUT / dataset / f"seed-{job['seed']:03d}" / "chunks" / f"chunk-{number:04d}.json"
            tasks.append((job, number, groups[start:start + width], path))
    return tasks


def validate_chunk(value, m, job, groups):
    expected = dict(manifest_sha256=m["manifest_sha256"], dataset=job["dataset"], seed=job["seed"],
                    condition="F-local", split="test", checkpoint_sha256=job["checkpoint_sha256"],
                    prediction_sha256=job["prediction_sha256"], input_availability="wait_for_last_word_1s_window",
                    search="single_full_segment_beam_from_empty_generated_prefix", selected_weight=.1)
    require(all(value.get(k) == v for k, v in expected.items()), "F-local chunk provenance differs")
    require([r["group"] for r in value["records"]] == [g["group"] for g in groups], "F-local chunk is reordered or incomplete")
    for row, group in zip(value["records"], groups):
        n = len(group["event_ids"])
        require(len(row["prediction"]) == n and all(isinstance(w, str) and w for w in row["prediction"]),
                "F-local word count differs")
        require(row["horizons"] == np.minimum(2, n - 1 - np.arange(n)).tolist(), "F-local tail horizons differ")


def committed_prefix(tasks, m, dataset):
    chunks, missing = [], False
    previous, cache = None, m["initial_language_cache"][dataset]
    for unit, (job, number, groups, path) in enumerate(tasks):
        if not path.exists():
            missing = True
            continue
        require(not missing, "Completed F-local chunks have a gap")
        value = read(path)
        validate_chunk(value, m, job, groups)
        require(value["unit"] == unit and value["previous_chunk_sha256"] == previous
                and value["cache_before_sha256"] == cache["sha256"], "Committed decoder state chain differs")
        previous, cache = sha256(path), value["cache_after"]
        chunks.append(value)
    require(sha256(ROOT / cache["path"]) == cache["sha256"], "Last committed language cache is absent or changed")
    expected_paths = {str(t[3].resolve()) for t in tasks}
    actual_paths = {str(p.resolve()) for p in (OUT / dataset).glob("seed-*/chunks/chunk-*.json")}
    require(actual_paths.issubset(expected_paths), "Undeclared F-local chunks exist")
    return chunks, cache, previous


def save_cache(scorer, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(12):
        try:
            scorer.save_cache(path)
            return dict(path=source_key(path), sha256=sha256(path))
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(.25)


def prune_cache(folder, unit):
    folder = folder.resolve()
    require(folder.is_relative_to(OUT), "Cache retention path left the experiment")
    for path in folder.glob("unit-*.pkl"):
        number = int(path.stem.removeprefix("unit-"))
        if number < unit - 1:
            require(path.resolve().parent == folder, "Cache retention target differs")
            path.unlink(missing_ok=True)


def worker(m, dataset):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=m["settings"]["execution"]["torch_threads"])
    torch.set_num_threads(m["settings"]["execution"]["torch_threads"])
    torch.manual_seed(0)
    np.random.seed(0)
    verify_sources(m)
    parent = read(PARENT)
    info, tasks = parent["datasets"][dataset], plan(m, parent, dataset)
    chunks, cache, previous = committed_prefix(tasks, m, dataset)
    saved = len(chunks)
    completed = sum(len(t[2]) for t in tasks[:saved])
    worker_status = RUNTIME / dataset / "status.json"

    def update(**fields):
        value = read(worker_status) if worker_status.exists() else {}
        value.update(state="running", pid=os.getpid(), dataset=dataset, manifest_sha256=m["manifest_sha256"],
                     completed_segment_seed_units=completed, total_segment_seed_units=len(info["groups"]) * 3,
                     updated_unix=time.time())
        value.update(fields)
        write_json(worker_status, value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    scorer = embed = bank = arrays = None
    try:
        update(stage="starting", completed_chunks=saved)
        if saved < len(tasks):
            gpu_ready(m["settings"]["execution"], update)
            update(stage="loading_models")
            scorer, embed, bank = models(parent, dataset)
            validate_cache(ROOT / cache["path"], info["language"])
            scorer.load_cache(ROOT / cache["path"])
        array_seed = None
        for unit, (job, number, groups, path) in enumerate(tasks):
            if unit < saved:
                continue
            if array_seed != job["seed"]:
                arrays = load_neural(ROOT / job["prediction"])
                array_seed = job["seed"]
            update(stage="full_segment_search", seed=job["seed"], chunk=number, unit=unit)
            started = time.perf_counter()
            records = decode(groups, *arrays, info, scorer, embed, bank, m["settings"]["decoding"])
            seconds = time.perf_counter() - started
            cache_after = save_cache(scorer, OUT / dataset / "cache_snapshots" / f"unit-{unit:04d}.pkl")
            value = dict(manifest_sha256=m["manifest_sha256"], dataset=dataset, seed=job["seed"],
                         condition="F-local", split="test", selected_weight=.1, unit=unit,
                         checkpoint_sha256=job["checkpoint_sha256"], prediction_sha256=job["prediction_sha256"],
                         input_availability="wait_for_last_word_1s_window",
                         search="single_full_segment_beam_from_empty_generated_prefix", records=records,
                         seconds=seconds, previous_chunk_sha256=previous, cache_before_sha256=cache["sha256"],
                         cache_after=cache_after, reference_supplied_to_decoder=False,
                         peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3)
            validate_chunk(value, m, job, groups)
            write_json(path, value)
            chunks.append(value)
            previous, cache = sha256(path), cache_after
            completed += len(groups)
            prune_cache(OUT / dataset / "cache_snapshots", unit)
            update(stage="chunk_complete", completed_chunks=unit + 1, chunk_seconds=seconds,
                   peak_gpu_memory_gib=value["peak_gpu_memory_gib"])
        for job in [j for j in m["jobs"] if j["dataset"] == dataset]:
            selected = [(t, c) for t, c in zip(tasks, chunks) if t[0]["seed"] == job["seed"]]
            records = [r for _, c in selected for r in c["records"]]
            main = dict(chunks[-1])
            main.update(status="completed", seed=job["seed"], checkpoint_sha256=job["checkpoint_sha256"],
                        prediction_sha256=job["prediction_sha256"], records=records,
                        sources=[dict(path=source_key(t[3]), sha256=sha256(t[3])) for t, _ in selected])
            # A seed summary has no individual chunk's state transaction.
            for key in ("unit", "previous_chunk_sha256", "cache_before_sha256", "cache_after", "seconds"):
                main.pop(key, None)
            validate_chunk(main, m, job, info["groups"])
            write_json(OUT / dataset / f"seed-{job['seed']:03d}" / "main.json", main)
        verify_sources(m)
        write_json(OUT / dataset / "completion.json", dict(status="completed", dataset=dataset,
                   manifest_sha256=m["manifest_sha256"], completed_segment_seed_units=completed,
                   final_language_cache=cache, completed_unix=time.time()))
        update(stage="completed", state="completed", completed_unix=time.time())
    except BaseException as error:
        update(stage="failed", state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        del scorer, embed, bank, arrays
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def queue_status(m, **fields):
    path = RUNTIME / "status.json"
    value = read(path) if path.exists() else {}
    value.update(pid=os.getpid(), manifest_sha256=m["manifest_sha256"], updated_unix=time.time(),
                 total_segment_seed_units=m["total_segment_seed_units"], **fields)
    write_json(path, value)
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = ["# F-local 进度", "", f"状态：{value.get('state', '')}；已提交 {value.get('completed_segment_seed_units', 0):,} / {m['total_segment_seed_units']:,} 个片段×种子。",
             "", f"当前：{value.get('dataset', '')}，种子 {value.get('seed', '')}，分块 {value.get('chunk', '')}。",
             "", "沿用 C 的最终可用共享 K2/K1/K0，等待并搜索方式与 F 相同。全部六组完成后自动汇总配对材料块置信区间。"]
    if value.get("error"):
        lines += ["", f"错误：{value['error']}"]
    if value.get("state") == "completed":
        lines += ["", "[完整结果](results.md)"]
    (REPORT / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run():
    import msvcrt
    m = freeze()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "queue.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0"); handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        completed = 0
        process = None
        queue_status(m, state="running", stage="starting", started_unix=time.time(), worker_pid=None,
                     completed_segment_seed_units=0, error=None, GPU_workers_exited=False)
        try:
            for dataset in m["settings"]["protocol"]["datasets"]:
                log_path = RUNTIME / dataset / "worker.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("ab") as log:
                    command = [sys.executable, "-u", "-m", "braindecoding.f_local_sentence_test_v2", "worker", "--dataset", dataset]
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    queue_status(m, state="running", stage="dataset_worker", dataset=dataset, worker_pid=process.pid)
                    observed = None
                    while process.poll() is None:
                        status_path = RUNTIME / dataset / "status.json"
                        progress = read(status_path) if status_path.exists() else {}
                        if progress.get("updated_unix") != observed:
                            observed = progress.get("updated_unix")
                            queue_status(m, state="running", stage=progress.get("stage", "starting_worker"),
                                dataset=dataset, worker_pid=process.pid, seed=progress.get("seed"), chunk=progress.get("chunk"),
                                completed_segment_seed_units=completed + progress.get("completed_segment_seed_units", 0),
                                last_chunk_seconds=progress.get("chunk_seconds"))
                        time.sleep(5)
                    require(process.returncode == 0, f"Dataset worker failed: {dataset}; see {source_key(log_path)}")
                done = read(OUT / dataset / "completion.json")
                require(done["status"] == "completed" and done["manifest_sha256"] == m["manifest_sha256"],
                        "Dataset completion identity differs")
                completed += done["completed_segment_seed_units"]
                queue_status(m, state="running", stage="dataset_completed", dataset=dataset, worker_pid=None,
                             completed_segment_seed_units=completed)
            require(completed == m["total_segment_seed_units"], "Full F-local population is incomplete")
            verify_sources(m, full=True)
            queue_status(m, state="running", stage="exporting", worker_pid=None, GPU_workers_exited=True)
            export(m)
            queue_status(m, state="completed", stage="completed", completed_unix=time.time(), GPU_workers_exited=True,
                         worker_pid=None, completed_segment_seed_units=completed)
        except BaseException as error:
            exited = process is None or process.poll() is not None
            queue_status(m, state="failed", stage="failed", error=f"{type(error).__name__}: {error}",
                         worker_pid=None if exited else process.pid, GPU_workers_exited=exited)
            raise


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "prepare", "run", "worker", "export"))
    parser.add_argument("--dataset", choices=("pallier2025", "chineseeeg2_littleprince"))
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "prepare":
        freeze()
    elif args.command == "run":
        run()
    elif args.command == "worker":
        require(args.dataset is not None, "A dataset worker requires --dataset")
        worker(read(MANIFEST), args.dataset)
    else:
        m = read(MANIFEST)
        verify_sources(m, full=True)
        export(m)

# --- Merged from f_local_sentence_test_v2_results.py ---
"""Complete C / F-local / F comparison with shared material-block draws."""

from collections import defaultdict
import csv
import time

import numpy as np

study = sys.modules[__name__]
from braindecoding.full_context_sentence_test_v2 import summarize
from braindecoding.unified_sentence_v2_artifacts import csv_write
from braindecoding.unified_sentence_v2_results import METRICS, metric_row


def block_arrays(rows, info):
    blocks = [b["block"] for b in info["blocks"]]
    subjects = sorted({g["subject"] for g in info["groups"]})
    bidx = {b: i for i, b in enumerate(blocks)}
    uidx = {u: i for i, u in enumerate(subjects)}
    cidx = {c: i for i, c in enumerate(study.CONDITIONS)}
    counts = np.zeros((len(METRICS), len(blocks), len(subjects)))
    values = np.zeros((len(study.CONDITIONS), 3, len(METRICS), len(blocks), len(subjects)))
    meta = {g["group"]: g for g in info["groups"]}
    expected = {(s, g["group"], c) for s in range(3) for g in info["groups"] for c in study.CONDITIONS}
    seen = set()
    for row in rows:
        key = row["seed"], row["group"], row["condition"]
        study.require(key in expected and key not in seen, "Duplicate or undeclared metric population")
        seen.add(key)
        group = meta[row["group"]]
        study.require(all(row[k] == group[k] for k in ("subject", "block", "scope"))
                      and row["words"] == len(group["event_ids"]), "Material pairing differs")
        b, u = bidx[row["block"]], uidx[row["subject"]]
        for metric, name in enumerate(METRICS):
            weight = 1. if name == "sentence_bleu1_mean" else row["words"]
            if row["condition"] == study.CONDITIONS[0] and row["seed"] == 0:
                counts[metric, b, u] += weight
            values[cidx[row["condition"]], row["seed"], metric, b, u] += weight * row[name]
    study.require(seen == expected and (counts.sum(axis=1) > 0).all(), "Incomplete metric population")
    return counts, values


def weighted_scores(weights, counts, values):
    output = []
    for metric in range(len(METRICS)):
        denominator = weights @ counts[metric]
        study.require((denominator > 0).all(), "Material bootstrap omitted a subject")
        numerator = np.einsum("rb,csbu->csru", weights, values[:, :, metric], optimize=True)
        output.append((numerator / denominator[None, None, :, :]).mean(axis=-1).mean(axis=1))
    return np.stack(output, axis=1)


def bootstrap_weights(info, policy, dataset_index):
    strata = defaultdict(list)
    for i, block in enumerate(info["blocks"]):
        strata[block["scope"]].append(i)
    rng = np.random.default_rng(np.random.SeedSequence([policy["seed"], dataset_index, 314159]))
    weights = np.zeros((policy["material_block_bootstrap_repetitions"], len(info["blocks"])))
    for indices in strata.values():
        picks = rng.choice(indices, (len(weights), len(indices)), replace=True)
        for i, row in enumerate(picks):
            weights[i] += np.bincount(row, minlength=len(info["blocks"]))
    return weights


def paired_statistics(rows, info, policy, dataset_index):
    counts, values = block_arrays(rows, info)
    observed = weighted_scores(np.ones((1, len(info["blocks"]))), counts, values)[:, :, 0]
    draws = weighted_scores(bootstrap_weights(info, policy, dataset_index), counts, values)
    comparisons = []
    for later, earlier in (("F-local", "C"), ("F", "F-local"), ("F", "C")):
        l, e = study.CONDITIONS.index(later), study.CONDITIONS.index(earlier)
        item = dict(comparison=f"{later}_minus_{earlier}", later=later, earlier=earlier,
                    blocks=len(info["blocks"]), subjects=len(counts[0, 0]), seeds=3,
                    bootstrap_repetitions=draws.shape[-1], supplementary=True, metrics={})
        for metric, name in enumerate(METRICS):
            sign = -1. if name == "wer" else 1.
            delta = sign * (draws[l, metric] - draws[e, metric])
            lo, hi = np.quantile(delta, (.025, .975))
            item["metrics"][name] = dict(later=float(observed[l, metric]), earlier=float(observed[e, metric]),
                improvement=float(sign * (observed[l, metric] - observed[e, metric])),
                ci_lower=float(lo), ci_upper=float(hi),
                direction="error_reduction" if name == "wer" else "increase")
        comparisons.append(item)
    c, local, f = (study.CONDITIONS.index(k) for k in ("C", "F-local", "F"))
    decomposition = []
    for metric, name in enumerate(METRICS):
        sign = -1. if name == "wer" else 1.
        first = sign * (observed[local, metric] - observed[c, metric])
        second = sign * (observed[f, metric] - observed[local, metric])
        total = sign * (observed[f, metric] - observed[c, metric])
        study.require(np.isclose(first + second, total, rtol=0, atol=1e-12)
                      and np.allclose((draws[local, metric] - draws[c, metric])
                                      + (draws[f, metric] - draws[local, metric]),
                                      draws[f, metric] - draws[c, metric], rtol=0, atol=1e-12),
                      "Paired decomposition is not additive")
        decomposition.append(dict(metric=name, waiting_mature_shared_input_and_search_increment=float(first),
            full_context_representation_increment=float(second), total_F_vs_C_improvement=float(total),
            descriptive_first_step_share=float(first / total) if abs(total) > 1e-12 else None,
            descriptive_representation_step_share=float(second / total) if abs(total) > 1e-12 else None,
            interpretation="comparison_path_decomposition_not_unique_additive_causal_attribution"))
    return observed, comparisons, decomposition


def read_local_main(m, job, groups):
    path = study.OUT / job["dataset"] / f"seed-{job['seed']:03d}" / "main.json"
    value = study.read(path)
    study.require(value["status"] == "completed", "Cannot export partial F-local results")
    study.validate_chunk(value, m, job, groups)
    reconstructed = []
    for source in value["sources"]:
        chunk_path = study.ROOT / source["path"]
        study.require(study.sha256(chunk_path) == source["sha256"], "F-local chunk changed")
        chunk = study.read(chunk_path)
        start = len(reconstructed)
        study.validate_chunk(chunk, m, job, groups[start:start + len(chunk["records"])])
        reconstructed.extend(chunk["records"])
    study.require(reconstructed == value["records"], "F-local summary differs from atomic chunks")
    return path, value


def timing(parent):
    with study.F_WAIT.open(encoding="utf-8-sig", newline="") as handle:
        original = list(csv.DictReader(handle))
    expected = {(ds, e): g for ds, info in parent["datasets"].items() for g in info["groups"] for e in g["event_ids"]}
    seen, rows = set(), []
    for row in original:
        key = row["dataset"], row["event_id"]
        study.require(key in expected and key not in seen, "F wait population differs")
        seen.add(key)
        group = expected[key]
        study.require(all(row[k] == str(group[k]) for k in ("group", "subject", "block", "scope")), "F wait material differs")
        row["common50"] = row["common50"].lower() == "true"
        for k in ("c_input_wait_s", "f_input_wait_s"):
            row[k] = float(row[k])
            study.require(np.isfinite(row[k]) and row[k] >= 1 - 1e-6, "Invalid input waiting time")
        row["f_local_input_wait_s"] = row["f_input_wait_s"]
        rows.append(row)
    study.require(seen == set(expected), "Incomplete F wait population")
    summary = []
    for dataset, info in parent["datasets"].items():
        all_members = [r for r in rows if r["dataset"] == dataset]
        study.require(len(all_members) == info["counts"]["events"], "Wait count differs")
        for population in ("all_events", "common50"):
            members = all_members if population == "all_events" else [r for r in all_members if r["common50"]]
            if population == "common50":
                study.require(len(members) == info["counts"]["strict50_common"], "Common wait count differs")
            c = np.array([r["c_input_wait_s"] for r in members])
            for condition, column in (("C", "c_input_wait_s"), ("F-local", "f_local_input_wait_s"), ("F", "f_input_wait_s")):
                values = np.array([r[column] for r in members])
                summary.append(dict(dataset=dataset, population=population, condition=condition, queries=len(members),
                    mean_input_wait_s=float(values.mean()), median_input_wait_s=float(np.median(values)),
                    p90_input_wait_s=float(np.quantile(values, .9)), mean_extra_wait_vs_C_s=float((values - c).mean()),
                    aggregation="query_weighted_no_seed_repetition", excludes="preprocessing_compute_scheduling_display"))
    return rows, summary


def export(m):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=4)
    study.verify_sources(m)
    parent, full = study.read(study.PARENT), study.read(study.FULL)
    original = study.read(study.F_SUMMARY)
    all_rows, all_subjects, all_seeds, all_pairs, datasets, sources = [], [], [], [], {}, []
    for index, (dataset, info) in enumerate(parent["datasets"].items()):
        groups, rows = info["groups"], []
        done = study.read(study.OUT / dataset / "completion.json")
        study.require(done["status"] == "completed" and done["manifest_sha256"] == m["manifest_sha256"]
                      and done["completed_segment_seed_units"] == len(groups) * 3, "Dataset is incomplete")
        study.committed_prefix(study.plan(m, parent, dataset), m, dataset)
        for job in [j for j in m["jobs"] if j["dataset"] == dataset]:
            path, local = read_local_main(m, job, groups)
            ac = study.original_ac(parent, dataset, job["seed"], study.ROOT / job["original_ac"])
            f = study.original_f(parent, full, dataset, job["seed"], study.ROOT / job["original_f"])
            for group, a, l, full_row in zip(groups, ac["records"], local["records"], f["records"]):
                predictions = {"C": a["result"]["final"]["C"], "F-local": l["prediction"], "F": full_row["prediction"]}
                for condition in study.CONDITIONS:
                    rows.append(dict(dataset=dataset, seed=job["seed"], subject=group["subject"], group=group["group"],
                        condition=condition, material=group["material"], block=group["block"], scope=group["scope"],
                        words=len(group["event_ids"]), **metric_row(group["reference"], predictions[condition], info["language"] == "zh")))
            sources.append(dict(path=study.source_key(path), sha256=study.sha256(path)))
        means, seeds, subjects = summarize(rows)
        lookup = {r["condition"]: r for r in means}
        for condition in ("C", "F"):
            expected = next(r for r in original["datasets"][dataset]["summary"] if r["condition"] == condition)
            for name in METRICS:
                study.require(np.isclose(lookup[condition][name], expected[name], rtol=0, atol=1e-12),
                              f"Original C/F metric was not reproduced: {dataset} {condition} {name}")
        observed, comparisons, decomposition = paired_statistics(rows, info, m["settings"]["statistics"], index)
        for c, condition in enumerate(study.CONDITIONS):
            for k, name in enumerate(METRICS):
                study.require(np.isclose(observed[c, k], lookup[condition][name], rtol=0, atol=1e-12), "Bootstrap weights differ from summary")
        datasets[dataset] = dict(counts=info["counts"], summary=[lookup[c] for c in study.CONDITIONS],
                                by_seed=seeds, comparisons=comparisons, decomposition=decomposition,
                                original_C_F_metrics_reproduced=True)
        all_rows.extend(rows)
        all_subjects.extend(dict(dataset=dataset, **r) for r in subjects)
        all_seeds.extend(dict(dataset=dataset, **r) for r in seeds)
        all_pairs.extend(dict(dataset=dataset, comparison=p["comparison"], metric=name, blocks=p["blocks"],
                              subjects=p["subjects"], seeds=3, **values) for p in comparisons for name, values in p["metrics"].items())
    study.require(len(all_rows) == m["total_segment_seed_units"] * 3, "Complete export population differs")
    waits, time_summary = timing(parent)
    exports = {}
    for suffix, rows in (("by_group", all_rows), ("by_subject_seed", all_subjects), ("by_seed", all_seeds),
                         ("comparisons", all_pairs), ("input_wait_by_event", waits), ("input_wait_summary", time_summary)):
        path = study.EXPORT / f"{study.ID}_{suffix}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        csv_write(path, rows)
        exports[study.source_key(path)] = study.sha256(path)
    result = dict(status="completed", protocol=study.ID, split="test", manifest_sha256=m["manifest_sha256"],
                  conditions=list(study.CONDITIONS), datasets=datasets, timing_summary=time_summary,
                  total_segment_seed_units=m["total_segment_seed_units"], inference_only=True,
                  original_C_F_read_only=True, original_C_F_metrics_reproduced=True,
                  input_wait_F_local_equals_F_per_event=True, new_significance_tests=False,
                  interpretation="C_to_F_local_waiting_mature_shared_input_and_unfrozen_search;F_local_to_F_representation",
                  decomposition="descriptive_comparison_path_not_unique_causal_contribution", sources=sources,
                  exports=exports, completed_unix=time.time())
    study.write_json(study.REPORT / "results.json", result)
    summary_path = study.EXPORT / f"{study.ID}_summary.json"
    study.write_json(summary_path, result)
    lines = ["# C / F-local / F 全量比较", "", "两个数据集、三个种子、8,244个片段×种子全部完成。原C/F只读复用，四项指标逐值复现。", "",
             "| 数据集 | 条件 | 平均片段 BLEU-1（%） | WER（%） | 对齐词准确率（%） |", "|---|---|---:|---:|---:|"]
    for dataset, info in datasets.items():
        for row in info["summary"]:
            lines.append(f"| {dataset} | {row['condition']} | {100*row['sentence_bleu1_mean']:.2f} | {100*row['wer']:.2f} | {100*row['aligned_word_accuracy']:.2f} |")
    lines += ["", "| 数据集 | 比较 | BLEU-1 增量（百分点） | 配对材料块95%区间 | WER减少（百分点） | 配对材料块95%区间 |",
              "|---|---|---:|---|---:|---|"]
    for dataset, info in datasets.items():
        for p in info["comparisons"]:
            b, w = p["metrics"]["sentence_bleu1_mean"], p["metrics"]["wer"]
            lines.append(f"| {dataset} | {p['later']} − {p['earlier']} | {100*b['improvement']:+.2f} | [{100*b['ci_lower']:+.2f}, {100*b['ci_upper']:+.2f}] | {100*w['improvement']:+.2f} | [{100*w['ci_lower']:+.2f}, {100*w['ci_upper']:+.2f}] |")
    lines += ["", "共享局部表示直接来自C同一检查点：内部词位K2，末两词K1/K0。F-local与F每词位输入等待完全一致。",
              "", "C→F-local 包含完整片段等待、从搜索起点使用成熟共享表示及解除前缀冻结。F-local→F在相同等待与搜索下比较神经表示。两步之和等于F−C，但候选和生成前缀的交互仍在，比例只描述这条比较路径，不能解释为唯一因果贡献。",
              "", "置信区间采用长度5材料块、记录/章节内分层10,000次bootstrap，同一抽样保留全部受试者和种子。不增加显著性检验。标称等待不含预处理或计算。"]
    report_path = study.REPORT / "results.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    study.write_json(study.REPORT / "audit.json", dict(status="verified_complete", manifest_sha256=m["manifest_sha256"],
                     result_sha256=study.sha256(summary_path), report_sha256=study.sha256(report_path), exports=exports,
                     source_sha256=m["source_sha256"], total_segment_seed_units=m["total_segment_seed_units"],
                     original_C_F_metrics_reproduced=True, complete_material_population=True,
                     shared_bootstrap_decomposition_verified=True, GPU_dataset_workers_exited=True))
    return result


if __name__ == "__main__":
    main()
