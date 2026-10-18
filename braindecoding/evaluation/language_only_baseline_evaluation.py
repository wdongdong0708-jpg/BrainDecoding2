"""Neural-free complete-word LM controls, with the original known boundaries."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from braindecoding.config import PROJECT_ROOT as ROOT
from braindecoding.unified_sentence_batch import BatchedWordLM, search_many
from braindecoding.unified_sentence_v2 import digest, read, sha256


ID = "pure_language_baseline_test_v2"
DATASETS = ("pallier2025", "chineseeeg2_littleprince")
CONDITIONS = ("LM_A", "LM_C1", "LM_R")
MATCH = {"A": "LM_A", "B": "LM_R", "C1": "LM_C1", "C": "LM_R"}
DELAYS = {"LM_A": 0, "LM_C1": 1, "LM_R": 2}
CONFIG = ROOT / "configs/evaluation/language_only_baseline_evaluation.yaml"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
PARENT = ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"
REPORT = ROOT / f"data/results/studies/{ID}"
OUT = ROOT / f"outputs/heldout_test/{ID}"
RUNTIME = ROOT / f".cache/runs/{ID}"
OLD_REPORTS = {
    "AC": ROOT / "data/results/exports/unified_heldout_test_v2_sentence_summary.json",
    "B": ROOT / "data/results/studies/fixed_k0_rolling_search_test_v2_stable/results.json",
    "C1": ROOT / "data/results/studies/single_revision_c1_test_v2_stable/results.json",
}


class FiniteWordLM(BatchedWordLM):
    def score_extensions(self, prefixes, words):
        scores = super().score_extensions(prefixes, words)
        require(np.isfinite(scores).all(), "Non-finite complete-word language scores")
        return scores


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.025 * (attempt + 1), .2))


def verify_identity(value):
    require(digest({k: v for k, v in value.items() if k != "manifest_sha256"}) == value["manifest_sha256"],
            "Frozen manifest identity differs")


def settings():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p, d = cfg["protocol"], cfg["decoding"]
    require(p["id"] == ID and p["datasets"] == list(DATASETS) and p["split"] == "test"
            and p["reference_neural_seeds"] == [0, 1, 2]
            and not any(p[k] for k in ("training_allowed", "recalibration_allowed", "neural_input_allowed", "reference_prompt_allowed")),
            "Language-only scope changed")
    require(d["conditions"] == list(CONDITIONS) and d["matched_controls"] == MATCH
            and d["revision_words"] == {c: h + 1 for c, h in DELAYS.items()}
            and not d["inherit_neural_shortlist"] and d["language_weight"] == 1., "Language-only search changed")
    return cfg


def no_embeddings(words):
    raise AssertionError("Pure language search must never encode candidate words")


def advance(outputs, arrival, vocabulary, scorer, decoder):
    """Only lengths, self-generated prefixes, lexicon and LM enter this API."""
    require(set(outputs) == set(CONDITIONS), "Language policies differ")
    requests, starts = [], []
    for condition in CONDITIONS:
        require(len(outputs[condition]) == arrival, "Language arrival count differs")
        start = max(0, arrival - DELAYS[condition])
        starts.append(start)
        requests.append(dict(queries=np.empty((arrival - start + 1, 0)), prefix=tuple(outputs[condition][:start]),
                             weight=1., open_vocabulary=True, pure_lm=True))
    # There is no neural dimension, embedding bank, checkpoint, or reference.
    results = search_many(requests, vocabulary, np.empty((len(vocabulary), 0)), scorer, no_embeddings,
        candidate_count=decoder["candidates"], beam_width=decoder["beam_width"], neural_scale=decoder["neural_scale"])
    row = {"arrival": arrival}
    for condition, start, request, result in zip(CONDITIONS, starts, requests, results):
        require(tuple(result[:start]) == request["prefix"] and len(result) == arrival + 1,
                "Language search changed a frozen prefix or word count")
        row[condition] = list(result)
    require(arrival != 0 or row["LM_A"] == row["LM_C1"] == row["LM_R"], "Language initial outputs disagree")
    require(arrival > 1 or row["LM_C1"] == row["LM_R"], "Language first one-window search disagrees")
    return row


def validate_trace(trace):
    previous = {c: [] for c in CONDITIONS}
    for arrival, row in enumerate(trace):
        require(set(row) == {"arrival", *CONDITIONS} and row["arrival"] == arrival, "Language trace schema differs")
        for condition in CONDITIONS:
            words = row[condition]
            require(len(words) == arrival + 1 and all(isinstance(w, str) and w for w in words), "Invalid language words")
            start = max(0, arrival - DELAYS[condition])
            require(words[:start] == previous[condition][:start], "Language policy revised a frozen word")
            previous[condition] = words
        require(arrival != 0 or row["LM_A"] == row["LM_C1"] == row["LM_R"], "Language initial mismatch")
        require(arrival > 1 or row["LM_C1"] == row["LM_R"], "Language first revision mismatch")


def crop_result(trace, count):
    require(0 < count <= len(trace), "Unsupported segment length")
    shortened = copy.deepcopy(trace[:count])
    validate_trace(shortened)
    return dict(final={c: list(shortened[-1][c]) for c in CONDITIONS}, trace=shortened)


def replay(count, vocabulary, scorer, decoder):
    outputs, trace = {c: [] for c in CONDITIONS}, []
    for arrival in range(count):
        row = advance(outputs, arrival, vocabulary, scorer, decoder)
        trace.append(row)
        outputs = {c: row[c] for c in CONDITIONS}
    return crop_result(trace, count)


def reference_inventory(parent):
    """Authenticate existing textual predictions without opening neural arrays."""
    from braindecoding.fixed_k0_rolling_search_test_v2 import original_ac, validate_b_result
    from braindecoding.single_revision_c1_test_v2 import validate_c1_result
    old = {name: read(path) for name, path in OLD_REPORTS.items()}
    require(old["AC"]["status"] == "completed_heldout_test"
            and old["AC"]["manifest_sha256"] == parent["manifest_sha256"], "Original A/C summary differs")
    for name in ("B", "C1"):
        require(old[name]["status"] == "completed" and old[name]["parent_manifest_sha256"] == parent["manifest_sha256"],
                f"Completed {name} source differs")
    result, paths = [], set(OLD_REPORTS.values())
    for dataset in DATASETS:
        groups = parent["datasets"][dataset]["groups"]
        for seed in range(3):
            ac_path = ROOT / f"outputs/heldout_test/unified_heldout_test_v2/{dataset}/seed-{seed:03d}/main.json"
            ac = original_ac(parent, dataset, seed, ac_path)
            paths.add(ac_path)
            for item in ac["sources"]:
                paths.add(ROOT / item["path"])
            entry = dict(dataset=dataset, seed=seed, AC=str(ac_path))
            for name, validator in (("B", validate_b_result), ("C1", validate_c1_result)):
                candidates = [x for x in old[name]["sources"] if f"/{dataset}/seed-{seed:03d}/main.json" in x["path"]]
                require(len(candidates) == 1, f"Missing {name} source")
                source = candidates[0]
                path = ROOT / source["path"]
                require(sha256(path) == source["sha256"], f"Completed {name} text changed")
                payload = read(path)
                require(payload["status"] == "completed" and payload["manifest_sha256"] == old[name]["manifest_sha256"]
                    and payload["conditions"] == [name] and payload["selected_weight"] == .1,
                    f"Completed {name} metadata differs")
                require([x["group"] for x in payload["records"]] == [g["group"] for g in groups], f"{name} groups differ")
                for group, record in zip(groups, payload["records"]):
                    validator(group, record["result"])
                for chunk in payload["sources"]:
                    cp = ROOT / chunk["path"]
                    require(sha256(cp) == chunk["sha256"], f"Completed {name} chunk changed")
                    paths.add(cp)
                entry[name] = str(path)
                paths.add(path)
            result.append(entry)
    return result, paths


def verify_sources(m, full=False):
    verify_identity(m)
    for path, expected in m["source_sha256"].items():
        if str(path).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        file = Path(path)
        stat = m["source_stat"][path]
        require(file.is_file() and file.stat().st_size == stat["bytes"] and file.stat().st_mtime_ns == stat["mtime_ns"],
                f"Frozen language source changed: {path}")
        if full or path in m["always_hash_sources"]:
            require(sha256(file) == expected, f"Frozen language source hash changed: {path}")


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        verify_sources(m, full=True)
        return m
    cfg, parent = settings(), read(PARENT)
    verify_identity(parent)
    require(parent["settings"]["sentence"]["language_weight"] == .1, "Original selected LM weight differs")
    for key in ("beam_width", "proposal_count", "proposal_max_new_tokens", "proposal_batch_size", "score_batch_size", "score_cache_size"):
        require(cfg["decoding"][key] == parent["decoder"][key], f"Original language budget differs: {key}")
    smoke = read(RUNTIME / "gpu_smoke.json")
    require(smoke["status"] == "passed" and {r["dataset"] for r in smoke["rows"]} == set(DATASETS), "Real language smoke required")
    refs, source_paths = reference_inventory(parent)
    source_paths.update((CONFIG, PROTOCOL, PARENT, Path(__file__), ROOT / "braindecoding/evaluation/pure_language_baseline_test_v2_results.py",
        ROOT / "braindecoding/decoding/unified_sentence.py", ROOT / "braindecoding/decoding/unified_sentence_batch.py",
        ROOT / "braindecoding/decoding/unified_sentence_v2.py", ROOT / "braindecoding/evaluation/unified_sentence_v2_results.py",
        ROOT / "braindecoding/decoding/unified_sentence_v2_artifacts.py", ROOT / "braindecoding/decoding/jayalath_sentence.py",
        ROOT / "tests/test_pure_language_baseline_test_v2.py", RUNTIME / "gpu_smoke.json"))
    lm = copy.deepcopy(parent["neural_settings"]["language_model"])
    lm.update({k: parent["decoder"][k] for k in ("proposal_count", "proposal_max_new_tokens", "proposal_batch_size", "score_batch_size", "score_cache_size")})
    model_root = ROOT / lm["local_path"]
    for file in model_root.iterdir():
        if file.is_file():
            require(str(file.resolve()) in parent["source_sha256"] and sha256(file) == parent["source_sha256"][str(file.resolve())],
                    "Qwen file differs from original test")
            source_paths.add(file)
    datasets = {}
    for dataset in DATASETS:
        info = parent["datasets"][dataset]
        groups = [dict(**{k: g[k] for k in ("group", "subject", "material", "block", "scope")}, count=len(g["event_ids"]))
                  for g in info["groups"]]
        require(all(g["count"] == len(original["reference"]) for g, original in zip(groups, info["groups"])), "Word counts differ")
        datasets[dataset] = dict(language=info["language"], vocabulary=list(info["vocabulary"]), groups=groups,
            maximum_words=max(g["count"] for g in groups), unique_lengths=sorted({g["count"] for g in groups}))
    sources, signatures = {}, {}
    for path in sorted(source_paths):
        file = str(path.resolve())
        sources[file] = sha256(path)
        signatures[file] = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    m = dict(protocol=ID, frozen_unix=time.time(), settings=cfg, parent_manifest_sha256=parent["manifest_sha256"],
        language_model=lm, decoder=parent["decoder"], datasets=datasets, references=refs,
        source_sha256=sources, source_stat=signatures,
        always_hash_sources=[p for p in sources if Path(p).suffix in (".py", ".yaml", ".md")],
        neural_input_allowed=False, text_embedding_model_loaded=False, reference_text_in_decoder=False,
        computed_lm_seed=0, paired_neural_seeds=[0, 1, 2], dependency_versions=dict(torch=str(torch.__version__), numpy=np.__version__,
            **{k: importlib.metadata.version(k) for k in ("transformers", "tokenizers", "jieba", "PyYAML")}))
    m["manifest_sha256"] = digest(m)
    verify_sources(m)
    write_json(MANIFEST, m)
    print(json.dumps(dict(stage="frozen", manifest_sha256=m["manifest_sha256"], languages=2,
                         policies=3, source_files=len(sources))), flush=True)
    return m


def gpu_ready(policy, status=None):
    require(torch.cuda.is_available(), "CUDA is required for the original language model")
    while True:
        free, total = torch.cuda.mem_get_info()
        if free >= policy["minimum_free_gpu_memory_gib"] * 1024**3:
            break
        if status:
            status(status="waiting_gpu_memory", free_gib=free / 1024**3)
        time.sleep(10)
    torch.cuda.set_per_process_memory_fraction(policy["gpu_memory_limit_gib"] * 1024**3 / total)


def load_model(lm):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = ROOT / lm["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16).cuda().eval()
    return model, tokenizer


def smoke():
    cfg, parent = settings(), read(PARENT)
    gpu_ready(cfg["execution"])
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    torch.manual_seed(0)
    lm = copy.deepcopy(parent["neural_settings"]["language_model"])
    lm.update({k: parent["decoder"][k] for k in ("proposal_count", "proposal_max_new_tokens", "proposal_batch_size", "score_batch_size", "score_cache_size")})
    model, tokenizer = load_model(lm)
    rows = []
    for dataset in DATASETS:
        info = parent["datasets"][dataset]
        scorer = FiniteWordLM(model, tokenizer, info["language"], lm)
        started = time.perf_counter()
        result = replay(1, info["vocabulary"], scorer, parent["decoder"])
        require(len(scorer.cache) >= len(info["vocabulary"]), "Smoke did not score the full lexicon")
        rows.append(dict(dataset=dataset, seconds=time.perf_counter() - started, result=result,
            vocabulary=len(info["vocabulary"]), score_cache_entries=len(scorer.cache),
            peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3, neural_array_reads=0, text_embedding_calls=0))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    value = dict(status="passed", rows=rows, language_model=lm, exact_original_scoring=True)
    write_json(RUNTIME / "gpu_smoke.json", value)
    del model, tokenizer, scorer
    gc.collect()
    torch.cuda.empty_cache()
    return value


def completed_prediction(m, dataset):
    path = OUT / dataset / "main.json"
    value = read(path)
    require(value["status"] == "completed" and value["manifest_sha256"] == m["manifest_sha256"]
            and value["dataset"] == dataset and value["conditions"] == list(CONDITIONS)
            and value["neural_input_allowed"] is False and value["reference_text_in_decoder"] is False,
            "Pure language output provenance differs")
    groups = m["datasets"][dataset]["groups"]
    require([r["group"] for r in value["records"]] == [g["group"] for g in groups], "Language group population differs")
    reconstructed = []
    previous = None
    for source in value["sources"]:
        file = Path(source["path"])
        require(sha256(file) == source["sha256"], "Language arrival changed")
        row = read(file)
        require(row["manifest_sha256"] == m["manifest_sha256"] and row["previous_arrival_sha256"] == previous,
                "Language arrival chain differs")
        require(sha256(row["cache"]["path"]) == row["cache"]["sha256"], "Language cache snapshot changed")
        reconstructed.append(row["result"])
        previous = source["sha256"]
    validate_trace(reconstructed)
    require(len(reconstructed) == m["datasets"][dataset]["maximum_words"], "Language replay is incomplete")
    for group, record in zip(groups, value["records"]):
        require(record["result"] == crop_result(reconstructed, group["count"]), "Language length reuse differs")
    return value


def worker(m, dataset):
    verify_sources(m)
    info, root = m["datasets"][dataset], OUT / dataset
    if (root / "main.json").exists():
        return completed_prediction(m, dataset)
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(manifest_sha256=m["manifest_sha256"], dataset=dataset, language=info["language"],
                    conditions=list(CONDITIONS), neural_input_allowed=False, reference_text_in_decoder=False)
    def status(**fields):
        value = read(root / "status.json") if (root / "status.json").exists() else {}
        value.update(identity, **fields, pid=os.getpid(), updated_unix=time.time())
        write_json(root / "status.json", value)
        print(json.dumps(fields), flush=True)
    existing = sorted((root / "arrivals").glob("arrival-*.json"))
    trace, sources, latest_cache, previous = [], [], None, None
    for arrival, file in enumerate(existing):
        require(file.name == f"arrival-{arrival:03d}.json", "Language resume arrivals are not contiguous")
        payload = read(file)
        require(all(payload[k] == v for k, v in identity.items()) and payload["previous_arrival_sha256"] == previous,
                "Language resume provenance differs")
        require(payload["result"]["arrival"] == arrival, "Language resume order changed")
        latest_cache = payload["cache"]
        require(sha256(latest_cache["path"]) == latest_cache["sha256"], "Exact language resume cache differs")
        trace.append(payload["result"])
        previous = sha256(file)
        sources.append(dict(path=str(file), sha256=previous))
    validate_trace(trace)
    require(len(trace) <= info["maximum_words"], "Language resume exceeds maximum length")
    status(status="loading_language_model", completed_arrivals=len(trace), maximum_arrivals=info["maximum_words"])
    gpu_ready(m["settings"]["execution"], status)
    torch.set_num_threads(m["settings"]["execution"]["cpu_threads"])
    torch.manual_seed(0)
    model, tokenizer = load_model(m["language_model"])
    scorer = FiniteWordLM(model, tokenizer, info["language"], m["language_model"])
    if latest_cache:
        scorer.load_cache(latest_cache["path"])
    outputs = {c: trace[-1][c] if trace else [] for c in CONDITIONS}
    try:
        for arrival in range(len(trace), info["maximum_words"]):
            verify_sources(m)
            status(status="language_decoding", arrival=arrival, completed_arrivals=len(trace))
            started = time.perf_counter()
            row = advance(outputs, arrival, info["vocabulary"], scorer, m["decoder"])
            candidate = trace + [row]
            validate_trace(candidate)
            cache_path = root / "cache_snapshots" / f"arrival-{arrival:03d}-{time.time_ns()}.pkl"
            cache_path.parent.mkdir(exist_ok=True)
            scorer.save_cache(cache_path)
            cache = dict(path=str(cache_path), sha256=sha256(cache_path))
            destination = root / "arrivals" / f"arrival-{arrival:03d}.json"
            require(not destination.exists(), "A committed language arrival must not be overwritten")
            payload = {**identity, "result": row, "previous_arrival_sha256": previous, "cache": cache,
                "seconds": time.perf_counter() - started, "score_cache_entries": len(scorer.cache),
                "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 1024**3}
            write_json(destination, payload)
            previous = sha256(destination)
            sources.append(dict(path=str(destination), sha256=previous))
            trace, outputs = candidate, {c: row[c] for c in CONDITIONS}
            status(status="arrival_complete", completed_arrivals=len(trace), arrival_seconds=payload["seconds"],
                   peak_gpu_memory_gib=payload["peak_gpu_memory_gib"])
        value = {**identity, "status": "completed", "split": "test", "known_word_count_preserved": True,
            "no_neural_shortlist": True, "computed_lm_seed": 0, "fresh_language_cache": True,
            "sources": sources, "records": [dict(group=g["group"], result=crop_result(trace, g["count"])) for g in info["groups"]]}
        write_json(root / "main.json", value)
        completed_prediction(m, dataset)
        status(status="completed", completed_arrivals=len(trace))
        return value
    except BaseException as error:
        status(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        del model, tokenizer, scorer
        gc.collect()
        torch.cuda.empty_cache()


def queue_status(m, **fields):
    path = RUNTIME / "status.json"
    value = read(path) if path.exists() else {}
    value.update(fields, manifest_sha256=m["manifest_sha256"], pid=os.getpid(), updated_unix=time.time())
    write_json(path, value)
    lines = ["# 纯语言模型对照", "", f"状态：{value.get('state')}；完成 {value.get('completed_datasets', 0)} / 2 个数据集。", "",
        "移除神经评分与神经候选筛选；保留原词数、片段边界、Qwen 模型、提示和束搜索规则。",
        "LM_A 对应 A；LM_C1 对应 C1；LM_R 对应 B/C。", ""]
    if value.get("current"):
        current = OUT / value["current"] / "status.json"
        if current.exists():
            state = read(current)
            lines += [f"当前：{value['current']}；{state.get('status')}；完成 {state.get('completed_arrivals', 0)} / {state.get('maximum_arrivals', '')} 次到达。"]
    if value.get("error"):
        lines += ["", value["error"]]
    if value.get("state") == "completed":
        lines += ["", "[完整结果](results.md)"]
    REPORT.mkdir(parents=True, exist_ok=True)
    try:
        (REPORT / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except PermissionError:
        pass


def run():
    import msvcrt
    m = freeze()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / "worker.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        queue_status(m, state="running", completed_datasets=0, started_unix=time.time(), error=None)
        try:
            for i, dataset in enumerate(DATASETS):
                queue_status(m, stage="language_decoding", current=dataset)
                logs = RUNTIME / "logs"
                logs.mkdir(exist_ok=True)
                env = os.environ.copy()
                env.update(PYTHONUTF8="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
                command = [sys.executable, "-m", "braindecoding.pure_language_baseline_test_v2", "worker", "--dataset", dataset]
                with (logs / f"{dataset}.stdout.log").open("a", encoding="utf-8") as stdout, \
                     (logs / f"{dataset}.stderr.log").open("a", encoding="utf-8") as stderr:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                               creationflags=subprocess.CREATE_NO_WINDOW)
                    queue_status(m, worker_pid=process.pid)
                    last = 0.
                    while process.poll() is None:
                        if time.monotonic() - last >= 30:
                            queue_status(m, worker_pid=process.pid)
                            last = time.monotonic()
                        time.sleep(1)
                require(process.returncode == 0, f"Language worker failed ({process.returncode}): {dataset}")
                completed_prediction(m, dataset)
                queue_status(m, completed_datasets=i + 1, worker_pid=None)
            queue_status(m, stage="paired_statistics", current=None)
            export(m)
            queue_status(m, state="completed", stage="completed", completed_unix=time.time(), GPU_workers_exited=True)
        except BaseException as error:
            queue_status(m, state="failed", stage="failed", error=f"{type(error).__name__}: {error}")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "prepare", "run", "worker", "export"))
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        smoke()
    elif args.command == "prepare":
        freeze()
    elif args.command == "run":
        run()
    elif args.command == "worker":
        require(args.dataset is not None, "Dataset is required")
        worker(read(MANIFEST), args.dataset)
    else:
        export(freeze())

# --- Merged from pure_language_baseline_test_v2_results.py ---
"""Paired text metrics against search-matched language-only controls."""

from collections import defaultdict
import time

import numpy as np

study = sys.modules[__name__]
from braindecoding.unified_sentence_v2_results import METRICS, holm, metric_row
from braindecoding.unified_sentence_v2_artifacts import csv_write


ALL_CONDITIONS = (*study.CONDITIONS, *study.MATCH)
NAMES = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}


def summarize(rows):
    cells = defaultdict(list)
    for row in rows:
        study.require(row["condition"] in ALL_CONDITIONS and row["words"] > 0, "Unknown or empty text metric")
        cells[row["seed"], row["subject"], row["condition"]].append(row)
    subjects = []
    for (seed, subject, condition), members in sorted(cells.items()):
        words = sum(r["words"] for r in members)
        values = {k: float(np.mean([r[k] for r in members])) if k == "sentence_bleu1_mean" else
                  float(sum(r[k] * r["words"] for r in members) / words) for k in METRICS}
        subjects.append(dict(seed=seed, subject=subject, condition=condition, segments=len(members), words=words, **values))
    seed_ids = sorted({r["seed"] for r in rows})
    subject_sets = [{r["subject"] for r in subjects if r["seed"] == seed and r["condition"] == condition}
                    for seed in seed_ids for condition in ALL_CONDITIONS]
    study.require(subject_sets and all(s and s == subject_sets[0] for s in subject_sets), "Text subjects or conditions differ")
    by_seed = [dict(seed=seed, condition=condition,
        **{k: float(np.mean([r[k] for r in subjects if r["seed"] == seed and r["condition"] == condition])) for k in METRICS})
        for seed in seed_ids for condition in ALL_CONDITIONS]
    summary = [dict(condition=condition,
        **{k: float(np.mean([r[k] for r in by_seed if r["condition"] == condition])) for k in METRICS})
        for condition in ALL_CONDITIONS]
    return dict(summary=summary, by_seed=by_seed, by_subject_seed=subjects)


def block_arrays(rows, info):
    blocks = [b["block"] for b in info["blocks"]]
    subjects = sorted({g["subject"] for g in info["groups"]})
    bidx, uidx = {b: i for i, b in enumerate(blocks)}, {u: i for i, u in enumerate(subjects)}
    cidx = {c: i for i, c in enumerate(ALL_CONDITIONS)}
    counts = np.zeros((len(METRICS), len(blocks), len(subjects)))
    values = np.zeros((len(ALL_CONDITIONS), 3, len(METRICS), len(blocks), len(subjects)))
    expected = {(seed, g["group"], c) for seed in range(3) for g in info["groups"] for c in ALL_CONDITIONS}
    seen = set()
    meta = {g["group"]: g for g in info["groups"]}
    for row in rows:
        key = row["seed"], row["group"], row["condition"]
        study.require(key in expected and key not in seen, "Text metric population is duplicate or undeclared")
        seen.add(key)
        group = meta[row["group"]]
        study.require(all(row[k] == group[k] for k in ("subject", "scope", "block"))
                      and row["words"] == len(group["event_ids"]), "Text metric material pairing differs")
        b, u = bidx[row["block"]], uidx[row["subject"]]
        for metric, name in enumerate(METRICS):
            weight = 1. if name == "sentence_bleu1_mean" else row["words"]
            if row["condition"] == ALL_CONDITIONS[0] and row["seed"] == 0:
                counts[metric, b, u] += weight
            values[cidx[row["condition"]], row["seed"], metric, b, u] += weight * row[name]
    study.require(seen == expected and (counts.sum(axis=1) > 0).all(), "Text metric population is incomplete")
    return counts, values


def weighted_scores(weights, counts, values):
    """One material draw is shared by all seeds, subjects and conditions."""
    output = []
    for metric in range(len(METRICS)):
        denominator = weights @ counts[metric]
        study.require((denominator > 0).all(), "A material draw omitted a subject")
        numerator = np.einsum("rb,cgbu->cgru", weights, values[:, :, metric], optimize=True)
        output.append((numerator / denominator[None, None, :, :]).mean(axis=-1).mean(axis=1))
    return np.stack(output, axis=1)


def material_weights(info, policy, index):
    strata = defaultdict(list)
    for i, block in enumerate(info["blocks"]):
        strata[block["scope"]].append(i)
    rng = np.random.default_rng(np.random.SeedSequence([policy["seed"], index, 314159]))
    weights = np.zeros((policy["material_block_bootstrap_repetitions"], len(info["blocks"])))
    for indices in strata.values():
        picks = rng.choice(indices, (len(weights), len(indices)), replace=True)
        for i, row in enumerate(picks):
            weights[i] += np.bincount(row, minlength=len(info["blocks"]))
    return weights


def paired_statistics(rows, info, policy, index):
    counts, values = block_arrays(rows, info)
    observed = weighted_scores(np.ones((1, len(info["blocks"]))), counts, values)[:, :, 0]
    draws = weighted_scores(material_weights(info, policy, index), counts, values)
    rng = np.random.default_rng(np.random.SeedSequence([policy["seed"], index, 271828]))
    block_count = len(info["blocks"])
    exact = block_count <= 16
    if exact:
        integers = np.arange(2**block_count, dtype=np.uint32)
        signs = 2 * ((integers[:, None] >> np.arange(block_count, dtype=np.uint32)) & 1).astype(float) - 1
    else:
        signs = rng.choice((-1., 1.), (policy["paired_material_block_sign_flips"], block_count))
    comparisons = []
    for condition, control in study.MATCH.items():
        real, lm = ALL_CONDITIONS.index(condition), ALL_CONDITIONS.index(control)
        item = dict(condition=condition, control=control, comparison=f"{condition}_minus_{control}",
            groups=len(info["groups"]), subjects=len(counts[0, 0]), blocks=block_count, neural_seeds=3,
            unique_language_trajectories=1, bootstrap_repetitions=draws.shape[-1], sign_flip_repetitions=len(signs),
            sign_flip_exact=exact, metrics={})
        for metric, name in enumerate(METRICS):
            sign = -1. if name == "wer" else 1.
            effect = sign * (observed[real, metric] - observed[lm, metric])
            delta = sign * (draws[real, metric] - draws[lm, metric])
            low, high = np.quantile(delta, (.025, .975))
            contributions = (sign * (values[real, :, metric] - values[lm, :, metric]).mean(axis=0)
                             / counts[metric].sum(axis=0)[None, :]).mean(axis=-1)
            study.require(np.isclose(contributions.sum(), effect, atol=1e-12, rtol=0), "Text paired weighting differs")
            null = signs @ contributions
            hits = (np.abs(null) >= abs(effect) - 1e-12).sum()
            pvalue = float(hits / len(signs) if exact else (1 + hits) / (1 + len(signs)))
            item["metrics"][name] = dict(real=float(observed[real, metric]), language_only=float(observed[lm, metric]),
                improvement=float(effect), ci_lower=float(low), ci_upper=float(high), p_two_sided=pvalue,
                improvement_direction="error_reduction" if sign < 0 else "increase")
        comparisons.append(item)
    decomposition = []
    a, la = ALL_CONDITIONS.index("A"), ALL_CONDITIONS.index("LM_A")
    for condition in ("B", "C1", "C"):
        real, lm = ALL_CONDITIONS.index(condition), ALL_CONDITIONS.index(study.MATCH[condition])
        for metric, name in enumerate(METRICS):
            sign = -1. if name == "wer" else 1.
            neural_gain = sign * (observed[real, metric] - observed[a, metric])
            lm_gain = sign * (observed[lm, metric] - observed[la, metric])
            delta = sign * ((draws[real, metric] - draws[a, metric]) - (draws[lm, metric] - draws[la, metric]))
            low, high = np.quantile(delta, (.025, .975))
            decomposition.append(dict(condition=condition, metric=name, real_gain_relative_A=float(neural_gain),
                pure_language_gain_relative_LM_A=float(lm_gain), difference_in_differences=float(neural_gain - lm_gain),
                ci_lower=float(low), ci_upper=float(high), interpretation="descriptive_search_matched_difference_not_additive_causal_share"))
    return comparisons, decomposition


def load_rows(m, parent, dataset):
    info = parent["datasets"][dataset]
    lm = study.completed_prediction(m, dataset)
    lm_records = {r["group"]: r["result"] for r in lm["records"]}
    rows = []
    for reference in [r for r in m["references"] if r["dataset"] == dataset]:
        payloads = {key: study.read(reference[key]) for key in ("AC", "B", "C1")}
        for key, payload in payloads.items():
            study.require([r["group"] for r in payload["records"]] == [g["group"] for g in info["groups"]], f"{key} groups changed")
        lookups = {k: {r["group"]: r["result"] for r in p["records"]} for k, p in payloads.items()}
        for group in info["groups"]:
            identity = dict(dataset=dataset, seed=reference["seed"], subject=group["subject"], group=group["group"],
                            material=group["material"], block=group["block"], scope=group["scope"], words=len(group["reference"]))
            final = {**lm_records[group["group"]]["final"], **lookups["AC"][group["group"]]["final"],
                     **lookups["B"][group["group"]]["final"], **lookups["C1"][group["group"]]["final"]}
            study.require(set(final) == set(ALL_CONDITIONS), "Text comparison policies differ")
            for condition in ALL_CONDITIONS:
                prediction = final[condition]
                study.require(len(prediction) == identity["words"], "Known word count was not preserved")
                rows.append(dict(**identity, condition=condition,
                                 **metric_row(group["reference"], prediction, info["language"] == "zh")))
    return rows


def export(m):
    from threadpoolctl import threadpool_limits
    started = time.perf_counter()
    study.verify_sources(m, full=True)
    parent = study.read(study.PARENT)
    old = {key: study.read(path) for key, path in study.OLD_REPORTS.items()}
    all_rows, all_seed_rows, all_subject_rows, summary_rows, comparisons, decompositions = [], [], [], [], [], []
    datasets = {}
    with threadpool_limits(limits=2):
        for index, dataset in enumerate(study.DATASETS):
            rows = load_rows(m, parent, dataset)
            aggregate = summarize(rows)
            current = {r["condition"]: r for r in aggregate["summary"]}
            for source_name, conditions in (("AC", ("A", "C")), ("B", ("B",)), ("C1", ("C1",))):
                expected = {r["condition"]: r for r in old[source_name]["datasets"][dataset]["summary"]}
                for condition in conditions:
                    study.require(all(np.isclose(current[condition][metric], expected[condition][metric], atol=1e-12, rtol=0)
                                      for metric in METRICS), f"Original {dataset} {condition} metric changed")
            paired, decomposition = paired_statistics(rows, parent["datasets"][dataset], m["settings"]["statistics"], index)
            for row in paired:
                for metric, value in row["metrics"].items():
                    study.require(np.isclose(value["real"], current[row["condition"]][metric], atol=1e-12, rtol=0)
                        and np.isclose(value["language_only"], current[row["control"]][metric], atol=1e-12, rtol=0), "Paired metric differs from summary")
            datasets[dataset] = dict(counts=parent["datasets"][dataset]["counts"], **aggregate)
            comparisons.extend(dict(dataset=dataset, **r) for r in paired)
            decompositions.extend(dict(dataset=dataset, **r) for r in decomposition)
            all_rows.extend(rows)
            all_seed_rows.extend(dict(dataset=dataset, **r) for r in aggregate["by_seed"])
            all_subject_rows.extend(dict(dataset=dataset, **r) for r in aggregate["by_subject_seed"])
            summary_rows.extend(dict(dataset=dataset, **r) for r in aggregate["summary"])
    study.require(len(comparisons) == 8, "Matched comparison family is incomplete")
    for row, pvalue in zip(comparisons, holm([r["metrics"]["sentence_bleu1_mean"]["p_two_sided"] for r in comparisons])):
        row["metrics"]["sentence_bleu1_mean"].update(p_holm=float(pvalue), significant_holm_05=bool(pvalue < .05))
    sources = {}
    study.REPORT.mkdir(parents=True, exist_ok=True)
    for name, values in (("by_group", all_rows), ("by_seed", all_seed_rows), ("by_subject_seed", all_subject_rows),
                         ("summary", summary_rows), ("decomposition", decompositions)):
        file = study.REPORT / f"{name}.csv"
        csv_write(file, values)
        sources[str(file)] = study.sha256(file)
    for dataset in study.DATASETS:
        file = study.OUT / dataset / "main.json"
        sources[str(file)] = study.sha256(file)
    result = dict(status="completed", protocol=study.ID, split="test", test_evaluated=True,
        manifest_sha256=m["manifest_sha256"], parent_manifest_sha256=m["parent_manifest_sha256"],
        conditions=list(ALL_CONDITIONS), matched_controls=study.MATCH, computed_lm_seed=0, paired_neural_seeds=[0, 1, 2],
        deterministic_language_runs_per_dataset=1, language_policies_per_dataset=3,
        independent_language_replicates_per_policy=1, neural_input_in_baseline=False, neural_shortlist_in_baseline=False,
        reference_text_in_prompt=False, datasets=datasets, comparisons=comparisons, decomposition=decompositions,
        original_A_B_C1_C_metrics_reproduced=True, source_sha256=sources,
        elapsed_cpu_seconds=time.perf_counter() - started, completed_unix=time.time())
    study.write_json(study.REPORT / "results.json", result)
    write_markdown(result)
    study.verify_sources(m, full=True)
    study.write_json(study.REPORT / "audit.json", dict(status="passed", manifest_sha256=m["manifest_sha256"],
        results_sha256=study.sha256(study.REPORT / "results.json"), original_outputs_unchanged=True,
        original_metrics_reproduced=True, identical_known_word_counts_and_boundaries=True,
        full_training_vocabulary_scored_by_LM=True, baseline_neural_array_reads=0, baseline_embedding_model_loads=0,
        reference_words_enter_decoder=False, paired_material_draws_shared_by_conditions_subjects_and_seeds=True,
        deterministic_language_predictions_not_counted_as_independent_seeds=True, source_sha256=sources))
    print("Pure-language comparisons exported", flush=True)
    return result


def write_markdown(result):
    lines = ["# 纯语言模型基线：正式测试结果", "",
        "两个数据集全部测试片段。纯语言模型移除神经评分与神经候选筛选，完整训练词库加语言建议；模型、提示、词数、边界与原协议一致。",
        "LM_A 对应 A；LM_C1 对应 C1；LM_R 对应 B/C。纯语言对照只计算一次，配对已有三个神经种子。", "",
        "BLEU-1 为受试者内片段等权、再受试者与神经种子等权；WER 按受试者内词位合并后等权。库外目标保留。", "",
        "| 数据集 | 条件 | 平均片段 BLEU-1 | WER | 语料 BLEU-1 | 对齐词准确率 |",
        "|---|---|---:|---:|---:|---:|"]
    for dataset, info in result["datasets"].items():
        for row in info["summary"]:
            lines.append(f"| {NAMES[dataset]} | {row['condition']} | {100*row['sentence_bleu1_mean']:.2f}% | {100*row['wer']:.2f}% | {100*row['bleu1']:.2f}% | {100*row['aligned_word_accuracy']:.2f}% |")
    lines += ["", "## 相对匹配语言对照的增量", "",
        "正 BLEU 差值或正 WER 减少均表示神经策略更好。95% CI 为 10,000 次配对材料块 bootstrap；BLEU-1 八项双侧块检验共同 Holm 校正。", "",
        "| 数据集 | 比较 | BLEU-1 增加（百分点） | 95% CI | Holm p | WER 减少（百分点） | WER 95% CI |",
        "|---|---|---:|---|---:|---:|---|"]
    for row in result["comparisons"]:
        b, w = row["metrics"]["sentence_bleu1_mean"], row["metrics"]["wer"]
        lines.append(f"| {NAMES[row['dataset']]} | {row['condition']} − {row['control']} | {100*b['improvement']:+.2f} | [{100*b['ci_lower']:+.2f}, {100*b['ci_upper']:+.2f}] | {b['p_holm']:.6f} | {100*w['improvement']:+.2f} | [{100*w['ci_lower']:+.2f}, {100*w['ci_upper']:+.2f}] |")
    lines += ["", "## 策略增益的描述性配对分解", "",
        "| 数据集 | 策略 | 真实策略相对 A 增益 | 纯语言相应搜索相对 LM_A 增益 | 二者差值 | 差值 95% CI |",
        "|---|---|---:|---:|---:|---|"]
    for row in result["decomposition"]:
        if row["metric"] == "sentence_bleu1_mean":
            lines.append(f"| {NAMES[row['dataset']]} | {row['condition']} | {100*row['real_gain_relative_A']:+.2f} | {100*row['pure_language_gain_relative_LM_A']:+.2f} | {100*row['difference_in_differences']:+.2f} | [{100*row['ci_lower']:+.2f}, {100*row['ci_upper']:+.2f}] |")
    lines += ["", "以上单位为 BLEU-1 百分点。差值量化相同搜索规则下神经通路的额外优势；语言前缀、候选和搜索仍有交互，不解释为纯神经信息的可加贡献百分比。",
        "未声明等效界限，不把不显著当作等效。纯语言结果只能评价当前提示和已知词数/边界下的语言先验，不能证明材料未出现在预训练语料中，也不能排除神经生成前缀触发预训练记忆的交互。", "",
        "[完整 JSON](results.json) · [逐种子指标](by_seed.csv) · [逐片段指标](by_group.csv) · [来源审计](audit.json)"]
    (study.REPORT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
