"""Re-decode frozen neural A/C with the completed LM control's exact prompt."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from braindecoding.config import PROJECT_ROOT as ROOT
from braindecoding.fixed_k0_rolling_search_test_v2 import original_ac
from braindecoding.f_local_sentence_test_v2 import (
    audit_neural, gpu_ready, load_neural, source_key, verify_identity, write_json,
)
from braindecoding.pure_language_prompt_sensitivity_test_v2 import FiniteWordLM
from braindecoding.unified_sentence_batch import replay_many
from braindecoding.unified_sentence_v2 import digest, load_lm, read, sha256

ID = "neural_prompt_sensitivity_test_v2"
DATASETS = ("pallier2025", "chineseeeg2_littleprince")
CONDITIONS = ("A", "C")
MATCH = {"A": "LM_A", "C": "LM_R"}
CONFIG = ROOT / f"configs/unified_rolling/{ID}.yaml"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
PARENT = ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"
LM_MANIFEST = PROTOCOL.with_name("pure_language_prompt_sensitivity_test_v2.json")
LM_REPORT = ROOT / "data/results/studies/pure_language_prompt_sensitivity_test_v2"
OLD_SUMMARY = ROOT / "data/results/exports/unified_heldout_test_v2_sentence_summary.json"
OUT = ROOT / f"outputs/heldout_test/{ID}"
REPORT = ROOT / f"data/results/studies/{ID}"
RUNTIME = ROOT / f".cache/runs/{ID}"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p, d, s = cfg["protocol"], cfg["decoding"], cfg["statistics"]
    require(p["id"] == ID and p["split"] == "test" and p["seeds"] == [0, 1, 2]
        and p["datasets"] == list(DATASETS), "Neural prompt population changed")
    require(not any(p[k] for k in ("training_allowed", "recalibration_allowed", "prompt_selection_allowed")),
        "No training, recalibration or prompt selection is allowed")
    require(p["parent"] == source_key(PARENT) and p["language_control"] == source_key(LM_MANIFEST),
        "Prompt experiment parent differs")
    require(d["conditions"] == list(CONDITIONS) and d["matched_controls"] == MATCH
        and d["language_weight"] == .1 and d["branch"] == "open" and d["alignment"] == "real"
        and d["cache"] == "fresh_private_prompt_bound_cache", "Neural A/C search changed")
    require(s["comparisons"] == ["C_minus_LM_R", "A_minus_LM_A", "C_minus_A"]
        and s["material_block_size"] == 5 and s["multiplicity"] == "holm"
        and s["alternative"] == "two_sided" and s["language_replicates_per_policy"] == 1
        and not s["secondary_significance_tests"], "Same-prompt inference contract changed")
    return cfg


def verify_sources(m, full=False):
    verify_identity(m)
    for name, expected in m["source_sha256"].items():
        if str(name).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        path = ROOT / name
        stamp = m["source_stat"][name]
        require(path.is_file() and path.stat().st_size == stamp["bytes"]
            and path.stat().st_mtime_ns == stamp["mtime_ns"], f"Frozen source metadata differs: {name}")
        if full or path.suffix in (".py", ".yaml", ".md"):
            require(sha256(path) == expected, f"Frozen source changed: {name}")


def prompt_inventory(parent):
    lm = read(LM_MANIFEST)
    verify_identity(lm)
    result, audit = read(LM_REPORT / "results.json"), read(LM_REPORT / "audit.json")
    require(lm["parent_manifest_sha256"] == parent["manifest_sha256"]
        and result["status"] == "completed" and audit["status"] == "verified_complete"
        and audit["manifest_sha256"] == result["manifest_sha256"] == lm["manifest_sha256"]
        and sha256(LM_REPORT / "results.json") == audit["result_sha256"], "Completed same-prompt LM source differs")
    require(not lm["reference_text_in_decoder"] and not lm["neural_input_allowed"]
        and not result["test_metrics_used_for_prompt_selection"], "LM source used references for prompt selection")
    diagnostic = read(ROOT / ".cache/runs/pure_language_prompt_sensitivity_test_v2/gpu_smoke.json")
    prompts = {}
    for dataset in DATASETS:
        info, original = lm["datasets"][dataset], parent["datasets"][dataset]
        require(info["language"] == original["language"] and info["vocabulary"] == original["vocabulary"]
            and info["prompt_id"] == "native_prose", "LM prompt or vocabulary differs")
        require([g["group"] for g in info["groups"]] == [g["group"] for g in original["groups"]]
            and [g["count"] for g in info["groups"]] == [len(g["event_ids"]) for g in original["groups"]],
            "Same-prompt known boundaries or word counts differ")
        template = info["prompt_template"]
        selected = next(r for r in diagnostic["rows"] if r["dataset"] == dataset)
        require(digest(template) == info["prompt_template_sha256"]
            and selected["prompt_template"] == template, "Exact LM prompt identity differs")
        prompts[dataset] = dict(prompt_id=info["prompt_id"], prompt_template=template,
            prompt_template_sha256=info["prompt_template_sha256"], prompt_ids=selected["prompt_ids"])
    return lm, prompts


def decoder_input(parent, dataset, prompt):
    info = parent["datasets"][dataset]
    value = dict(neural_settings=copy.deepcopy(parent["neural_settings"]),
        settings=dict(decoding=copy.deepcopy(parent["settings"]["decoding"])),
        datasets={dataset: {k: copy.deepcopy(info[k]) for k in ("language", "vocabulary", "embedding_cache")}})
    value["neural_settings"]["language_model"]["language_prompt_template"] = prompt["prompt_template"]
    return value


def cache_identity(language, prompt_ids, lm):
    return digest(dict(language=language, prompt_ids=list(prompt_ids), revision=lm["revision"],
        score=lm["score"], context_tokens=lm["context_tokens"], score_batch_size=lm["score_batch_size"],
        proposal_count=lm["proposal_count"], proposal_max_new_tokens=lm["proposal_max_new_tokens"]))


def models(parent, dataset, prompt):
    raw, embed, bank = load_lm(decoder_input(parent, dataset, prompt), dataset)
    scorer = FiniteWordLM(raw.model, raw.tokenizer, raw.language, raw.settings)
    require(list(scorer.prompt_ids) == prompt["prompt_ids"], "Neural scorer prompt tokens differ from pure LM")
    identity = cache_identity(scorer.language, scorer.prompt_ids, dict(raw.settings, context_tokens=scorer.context_tokens))
    require(scorer.cache_identity() == identity, "Private neural cache identity differs")
    require(bank.shape == (len(parent["datasets"][dataset]["vocabulary"]), 1024)
        and np.isfinite(bank).all() and np.allclose(np.linalg.norm(bank, axis=1), 1., atol=.005),
        "Frozen training embedding bank is malformed")

    def checked_embed(words):
        vectors = embed(words)
        require(vectors.shape == (len(words), 1024) and np.isfinite(vectors).all()
            and np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=.005), "Invalid language-proposed embedding")
        return vectors

    return scorer, checked_embed, bank


def cases(groups, predictions, masks, lookup, weight=.1):
    """Only the frozen vectors and segment event identities enter replay."""
    output = []
    require(weight == .1, "Language weight changed")
    for group in groups:
        ids = [lookup[e] for e in group["event_ids"]]
        n = len(ids)
        require(n > 0 and len(ids) == len(set(ids)), "Empty or duplicate neural segment")
        for h in range(3):
            expected = np.arange(n) + h < n
            require(np.array_equal(masks[h][ids], expected), "Horizon mask crosses segment boundary")
            valid = predictions[h][np.asarray(ids)[expected]]
            require(valid.shape == (int(expected.sum()), 1024) and np.isfinite(valid).all()
                and np.allclose(np.linalg.norm(valid, axis=1), 1., atol=.005), "Invalid available neural vector")
        output.append(dict(predictions={h: predictions[h][ids] for h in range(3)},
            weight=weight, open_vocabulary=True))
    return output


def validate_result(group, result):
    n = len(group["event_ids"])
    require(n > 0 and set(result) == {"final", "trace"} and set(result["final"]) == set(CONDITIONS)
        and len(result["trace"]) == n, "Neural result schema or population differs")
    previous = {c: [] for c in CONDITIONS}
    for arrival, row in enumerate(result["trace"]):
        require(set(row) == {"arrival", *CONDITIONS} and row["arrival"] == arrival, "Arrival order differs")
        for condition in CONDITIONS:
            words = row[condition]
            start = arrival if condition == "A" else max(0, arrival - 2)
            require(len(words) == arrival + 1 and all(isinstance(w, str) and w for w in words)
                and words[:start] == previous[condition][:start], "Invalid words or revised frozen prefix")
            previous[condition] = words
        require(arrival != 0 or row["A"] == row["C"], "Initial A/C neural search differs")
    require(previous == result["final"], "Final text differs from the last arrival")


def decode(groups, predictions, masks, lookup, info, scorer, embed, bank, settings, progress=None):
    inputs = cases(groups, predictions, masks, lookup, settings["language_weight"])
    answers = replay_many(inputs, info["vocabulary"], bank, scorer, embed, conditions="AC", progress=progress,
        candidate_count=settings["candidates"], beam_width=settings["beam_width"], neural_scale=settings["neural_scale"])
    require(len(answers) == len(groups), "A/C replay population differs")
    records = []
    for group, answer in zip(groups, answers):
        validate_result(group, answer)
        records.append(dict(group=group["group"], result=answer))
    return records


def code_paths():
    return [Path(__file__), ROOT / f"braindecoding/evaluation/{ID}_results.py", CONFIG, PROTOCOL,
        ROOT / f"tests/test_{ID}.py", ROOT / "braindecoding/evaluation/f_local_sentence_test_v2.py",
        ROOT / "braindecoding/evaluation/fixed_k0_rolling_search_test_v2.py",
        ROOT / "braindecoding/evaluation/pure_language_prompt_sensitivity_test_v2.py",
        ROOT / "braindecoding/evaluation/pure_language_baseline_test_v2_results.py",
        ROOT / "braindecoding/evaluation/unified_sentence_v2_results.py",
        ROOT / "braindecoding/decoding/unified_sentence_v2_artifacts.py", ROOT / "braindecoding/decoding/jayalath_sentence.py"]


def smoke():
    cfg, parent = config(), read(PARENT)
    lm, prompts = prompt_inventory(parent)
    torch.set_num_threads(cfg["execution"]["torch_threads"])
    torch.manual_seed(0)
    np.random.seed(0)
    gpu_ready(cfg["execution"])
    rows = []
    for dataset in DATASETS:
        info = parent["datasets"][dataset]
        groups = [next(g for g in info["groups"] if len(g["event_ids"]) == n) for n in (3, 4)]
        arrays = load_neural(ROOT / f"outputs/heldout_test/unified_heldout_test_v2/{dataset}/seed-000/neural_predictions.npz")
        scorer, embed, bank = models(parent, dataset, prompts[dataset])
        require(not scorer.cache and not scorer.proposal_cache, "Engineering search did not start with empty cache")
        started = time.perf_counter()
        records = decode(groups, *arrays, info, scorer, embed, bank, cfg["decoding"])
        seconds = time.perf_counter() - started
        cache = RUNTIME / "smoke" / dataset / "prompt_bound_cache.pkl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        scorer.save_cache(cache)
        replay_scorer = FiniteWordLM(scorer.model, scorer.tokenizer, scorer.language, scorer.settings)
        replay_scorer.load_cache(cache)
        require(decode(groups, *arrays, info, replay_scorer, embed, bank, cfg["decoding"]) == records,
            "Exact prompt-bound cached replay differs")
        rows.append(dict(dataset=dataset, groups=[g["group"] for g in groups], words=sum(len(g["event_ids"]) for g in groups),
            records=records, seconds=seconds, cache_identity=scorer.cache_identity(), prompt_ids=list(scorer.prompt_ids),
            prompt_template_sha256=prompts[dataset]["prompt_template_sha256"], exact_cached_replay=True,
            reference_supplied_to_decoder=False, peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3))
        print(json.dumps(dict(stage="gpu_smoke", dataset=dataset, status="passed", seconds=seconds)), flush=True)
        del replay_scorer, scorer, embed, bank, arrays
        gc.collect()
        torch.cuda.empty_cache()
    value = dict(status="passed", rows=rows, changed_factor="fixed_language_prompt_only",
        language_manifest_sha256=lm["manifest_sha256"], source_sha256={source_key(p): sha256(p) for p in code_paths()},
        checkpoint_selection=False, hyperparameter_selection=False, completed_unix=time.time())
    write_json(RUNTIME / "gpu_smoke.json", value)
    return value


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        verify_sources(m, full=True)
        return m
    from braindecoding import pure_language_prompt_sensitivity_test_v2 as language
    cfg, parent = config(), read(PARENT)
    verify_identity(parent)
    lm, prompts = prompt_inventory(parent)
    language.verify_sources(lm, full=True)
    old = read(OLD_SUMMARY)
    require(old["status"] == "completed_heldout_test" and old["manifest_sha256"] == parent["manifest_sha256"],
        "Original A/C summary differs")
    for key in ("candidates", "beam_width", "neural_scale", "proposal_count", "proposal_max_new_tokens",
        "proposal_batch_size", "score_batch_size", "score_cache_size", "segment_batch_size"):
        require(cfg["decoding"][key] == parent["settings"]["decoding"][key], f"Frozen decoder setting differs: {key}")
    for key in ("model_name", "revision", "local_path", "dtype", "context_tokens", "score"):
        require(lm["language_model"][key] == parent["neural_settings"]["language_model"][key], f"LM model/score differs: {key}")
    require(cfg["execution"]["segment_batch_size"] == parent["decoder"]["segment_batch_size"], "Batch layout differs")
    engineering = read(RUNTIME / "gpu_smoke.json")
    require(engineering["status"] == "passed" and engineering["language_manifest_sha256"] == lm["manifest_sha256"]
        and engineering["source_sha256"] == {source_key(p): sha256(p) for p in code_paths()}, "Engineering checks are absent or stale")
    sources = {source_key(p): expected for p, expected in parent["source_sha256"].items()}

    def bind(path, expected=None):
        name = source_key(path)
        actual = sha256(ROOT / name)
        require(expected is None or expected == actual, f"Source provenance differs: {name}")
        require(name not in sources or sources[name] == actual, f"Parent source changed: {name}")
        sources[name] = actual
        return name

    for path in code_paths() + [PARENT, LM_MANIFEST, OLD_SUMMARY, LM_REPORT / "results.json", LM_REPORT / "audit.json",
        ROOT / ".cache/runs/pure_language_prompt_sensitivity_test_v2/gpu_smoke.json", RUNTIME / "gpu_smoke.json"]:
        bind(path)
    from huggingface_hub import snapshot_download
    text = parent["neural_settings"]["text_embedding"]
    snapshot = Path(snapshot_download(text["model_name"], revision=text["revision"], local_files_only=True))
    for path in snapshot.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".txt", ".model", ".safetensors", ".bin"):
            bind(path)
    jobs, initial = [], {}
    for dataset in DATASETS:
        info = parent["datasets"][dataset]
        require(info["selected_weight"] == .1, "Original selected language weight differs")
        bind(info["embedding_cache"])
        pure = language.completed_prediction(lm, dataset)
        bind(language.OUT / dataset / "main.json")
        for source in pure["sources"]:
            bind(source["path"], source["sha256"])
            arrival = read(source["path"])
            bind(arrival["cache"]["path"], arrival["cache"]["sha256"])
        row = next(r for r in engineering["rows"] if r["dataset"] == dataset)
        empty = OUT / "inputs" / dataset / "empty_prompt_bound_cache.pkl"
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty_value = dict(format="prompt_bound_lm_v1", identity=row["cache_identity"], scores={}, proposals={})
        if empty.exists():
            with empty.open("rb") as handle:
                require(pickle.load(handle) == empty_value, "Unexpected initial cache already exists")
        else:
            with empty.open("wb") as handle:
                pickle.dump(empty_value, handle, protocol=5)
        initial[dataset] = dict(path=bind(empty), sha256=sha256(empty), identity=row["cache_identity"])
        for seed in cfg["protocol"]["seeds"]:
            cp = next(j for j in parent["checkpoints"] if (j["dataset"], j["seed"]) == (dataset, seed))
            folder = ROOT / f"outputs/heldout_test/unified_heldout_test_v2/{dataset}/seed-{seed:03d}"
            prediction, side_path, ac_path = folder / "neural_predictions.npz", folder / "word_metrics.json", folder / "main.json"
            side = read(side_path)
            require(side["manifest_sha256"] == parent["manifest_sha256"] and side["split"] == "test"
                and side["dataset"] == dataset and side["seed"] == seed and side["test_evaluated"]
                and side["checkpoint_sha256"] == cp["checkpoint_sha256"], "Frozen neural cache provenance differs")
            bind(prediction, side["prediction_sha256"])
            bind(cp["checkpoint"], cp["checkpoint_sha256"])
            bind(side_path)
            ac = original_ac(parent, dataset, seed, ac_path)
            for source in ac["sources"]:
                bind(source["path"], source["sha256"])
            jobs.append(dict(dataset=dataset, seed=seed, prediction=source_key(prediction),
                prediction_sha256=side["prediction_sha256"], checkpoint=source_key(cp["checkpoint"]),
                checkpoint_sha256=cp["checkpoint_sha256"], original_ac=bind(ac_path), neural_audit=audit_neural(prediction, info)))
    for name, expected in sources.items():
        require(sha256(ROOT / name) == expected, f"Frozen source changed: {name}")
    m = dict(protocol=ID, authorization="explicit_user_same_narrative_prompt_A_C_lambda_point_one_fixed_checkpoints",
        frozen_unix=time.time(), settings=cfg, parent_manifest_sha256=parent["manifest_sha256"],
        language_manifest_sha256=lm["manifest_sha256"], prompts=prompts, jobs=jobs, initial_language_cache=initial,
        source_sha256=sources, source_stat={n: dict(bytes=(ROOT / n).stat().st_size,
            mtime_ns=(ROOT / n).stat().st_mtime_ns) for n in sources},
        total_segment_seed_units=sum(info["counts"]["groups"] for info in parent["datasets"].values()) * 3,
        changed_factor="fixed_language_prompt_only", neural_training=False, checkpoint_selection=False,
        reference_text_in_decoder=False, language_replicates_per_policy=1)
    m["manifest_sha256"] = digest(m)
    write_json(MANIFEST, m)
    print(json.dumps(dict(stage="frozen", manifest_sha256=m["manifest_sha256"], jobs=len(jobs), source_files=len(sources))), flush=True)
    return m


def plan(m, parent, dataset):
    width = m["settings"]["execution"]["segment_batch_size"]
    groups = parent["datasets"][dataset]["groups"]
    return [(job, start // width, groups[start:start + width],
        OUT / dataset / f"seed-{job['seed']:03d}" / "chunks" / f"chunk-{start // width:04d}.json")
        for job in m["jobs"] if job["dataset"] == dataset for start in range(0, len(groups), width)]


def validate_chunk(value, m, job, groups):
    expected = dict(manifest_sha256=m["manifest_sha256"], dataset=job["dataset"], seed=job["seed"], conditions=list(CONDITIONS),
        split="test", checkpoint_sha256=job["checkpoint_sha256"], prediction_sha256=job["prediction_sha256"],
        prompt_template_sha256=m["prompts"][job["dataset"]]["prompt_template_sha256"],
        branch="open", alignment="real", selected_weight=.1, reference_text_in_decoder=False)
    require(all(value.get(k) == v for k, v in expected.items()), "Neural prompt chunk provenance differs")
    require([r["group"] for r in value["records"]] == [g["group"] for g in groups], "Neural chunk is incomplete or reordered")
    for row, group in zip(value["records"], groups):
        require(set(row) == {"group", "result"}, "Unexpected neural record metadata")
        validate_result(group, row["result"])


def committed_prefix(tasks, m, dataset):
    chunks, missing = [], False
    previous, cache = None, m["initial_language_cache"][dataset]
    for unit, (job, number, groups, path) in enumerate(tasks):
        if not path.exists():
            missing = True
            continue
        require(not missing, "Committed neural chunks have a gap")
        value = read(path)
        validate_chunk(value, m, job, groups)
        require(value["unit"] == unit and value["previous_chunk_sha256"] == previous
            and value["cache_before_sha256"] == cache["sha256"], "Committed search state chain differs")
        previous, cache = sha256(path), value["cache_after"]
        require(cache["identity"] == m["initial_language_cache"][dataset]["identity"], "Committed cache prompt identity differs")
        chunks.append(value)
    require(sha256(ROOT / cache["path"]) == cache["sha256"], "Last committed private cache differs")
    expected = {str(t[3].resolve()) for t in tasks}
    actual = {str(p.resolve()) for p in (OUT / dataset).glob("seed-*/chunks/chunk-*.json")}
    require(actual.issubset(expected), "Undeclared chunks exist")
    return chunks, cache, previous


def prune_cache(dataset, unit):
    folder = (OUT / dataset / "cache_snapshots").resolve()
    require(folder.is_relative_to(OUT.resolve()), "Cache retention path left the experiment")
    for path in folder.glob("unit-*.pkl"):
        if int(path.stem.removeprefix("unit-")) < unit - 1:
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
    saved, completed = len(chunks), sum(len(t[2]) for t in tasks[:len(chunks)])
    path_status = RUNTIME / dataset / "status.json"

    def update(**fields):
        value = read(path_status) if path_status.exists() else {}
        value.update(state="running", pid=os.getpid(), dataset=dataset, manifest_sha256=m["manifest_sha256"],
            completed_segment_seed_units=completed, total_segment_seed_units=len(info["groups"]) * 3,
            updated_unix=time.time())
        value.update(fields)
        write_json(path_status, value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    scorer = embed = bank = arrays = None
    try:
        update(stage="starting", completed_chunks=saved)
        if saved < len(tasks):
            gpu_ready(m["settings"]["execution"], update)
            update(stage="loading_models")
            scorer, embed, bank = models(parent, dataset, m["prompts"][dataset])
            require(scorer.cache_identity() == cache["identity"], "Worker prompt-bound cache settings differ")
            scorer.load_cache(ROOT / cache["path"])
        array_seed = None
        for unit, (job, number, groups, path) in enumerate(tasks):
            if unit < saved:
                continue
            if array_seed != job["seed"]:
                arrays = load_neural(ROOT / job["prediction"])
                array_seed = job["seed"]
            update(stage="rolling_search", seed=job["seed"], chunk=number, unit=unit, arrival=0)
            started, last = time.perf_counter(), [0.]

            def progress(arrival, counts):
                if time.monotonic() - last[0] >= 30 or arrival + 1 == max(counts):
                    last[0] = time.monotonic()
                    update(stage="rolling_search", arrival=arrival + 1, maximum_arrivals=max(counts))

            records = decode(groups, *arrays, info, scorer, embed, bank, m["settings"]["decoding"], progress)
            seconds = time.perf_counter() - started
            snapshot = OUT / dataset / "cache_snapshots" / f"unit-{unit:04d}.pkl"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            scorer.save_cache(snapshot)
            cache_after = dict(path=source_key(snapshot), sha256=sha256(snapshot), identity=scorer.cache_identity())
            value = dict(manifest_sha256=m["manifest_sha256"], dataset=dataset, seed=job["seed"], conditions=list(CONDITIONS),
                split="test", branch="open", alignment="real", selected_weight=.1, unit=unit,
                checkpoint_sha256=job["checkpoint_sha256"], prediction_sha256=job["prediction_sha256"],
                prompt_template_sha256=m["prompts"][dataset]["prompt_template_sha256"], records=records,
                previous_chunk_sha256=previous, cache_before_sha256=cache["sha256"], cache_after=cache_after,
                reference_text_in_decoder=False, seconds=seconds,
                peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3)
            validate_chunk(value, m, job, groups)
            write_json(path, value)
            chunks.append(value)
            previous, cache = sha256(path), cache_after
            completed += len(groups)
            prune_cache(dataset, unit)
            update(stage="chunk_complete", completed_chunks=unit + 1, chunk_seconds=seconds,
                peak_gpu_memory_gib=value["peak_gpu_memory_gib"])
        for job in [j for j in m["jobs"] if j["dataset"] == dataset]:
            selected = [(t, c) for t, c in zip(tasks, chunks) if t[0]["seed"] == job["seed"]]
            main = dict(selected[-1][1])
            main.update(status="completed", records=[r for _, c in selected for r in c["records"]],
                sources=[dict(path=source_key(t[3]), sha256=sha256(t[3])) for t, _ in selected])
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
    lines = ["# 同叙述提示 A/C 重解码进度", "",
        f"状态：{value.get('state', '')}；已提交 {value.get('completed_segment_seed_units', 0):,} / {m['total_segment_seed_units']:,} 个片段×种子。", "",
        f"当前：{value.get('dataset', '')}，种子 {value.get('seed', '')}，分块 {value.get('chunk', '')}，到达 {value.get('arrival', '')}。", "",
        "两个数据集、三个原检查点，λ=0.1。使用已完成纯语言补跑的准确叙述提示。完成后自动统计 C−LM_R、A−LM_A、C−A 的 BLEU-1、WER 和 Acc，以及新旧提示下滚动收益的配对变化。"]
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
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        completed, process = 0, None
        queue_status(m, state="running", stage="starting", started_unix=time.time(), worker_pid=None,
            completed_segment_seed_units=0, error=None, GPU_workers_exited=False)
        try:
            for dataset in DATASETS:
                log_path = RUNTIME / dataset / "worker.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("ab") as log:
                    process = subprocess.Popen([sys.executable, "-u", "-m", f"braindecoding.{ID}", "worker", "--dataset", dataset],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    queue_status(m, state="running", stage="dataset_worker", dataset=dataset, worker_pid=process.pid)
                    observed = None
                    while process.poll() is None:
                        status_path = RUNTIME / dataset / "status.json"
                        progress = read(status_path) if status_path.exists() else {}
                        if progress.get("updated_unix") != observed:
                            observed = progress.get("updated_unix")
                            queue_status(m, state="running", stage=progress.get("stage", "starting_worker"), dataset=dataset,
                                worker_pid=process.pid, seed=progress.get("seed"), chunk=progress.get("chunk"),
                                arrival=progress.get("arrival"), maximum_arrivals=progress.get("maximum_arrivals"),
                                completed_segment_seed_units=completed + progress.get("completed_segment_seed_units", 0),
                                last_chunk_seconds=progress.get("chunk_seconds"))
                        time.sleep(5)
                    require(process.returncode == 0, f"Dataset worker failed: {dataset}; see {source_key(log_path)}")
                done = read(OUT / dataset / "completion.json")
                require(done["status"] == "completed" and done["manifest_sha256"] == m["manifest_sha256"], "Dataset completion differs")
                completed += done["completed_segment_seed_units"]
                queue_status(m, state="running", stage="dataset_completed", dataset=dataset, worker_pid=None,
                    completed_segment_seed_units=completed)
            require(completed == m["total_segment_seed_units"], "Full A/C population is incomplete")
            verify_sources(m, full=True)
            queue_status(m, state="running", stage="exporting", worker_pid=None, GPU_workers_exited=True)
            export(m)
            queue_status(m, state="completed", stage="completed", completed_unix=time.time(), worker_pid=None,
                completed_segment_seed_units=completed, GPU_workers_exited=True)
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
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args()
    if args.command == "smoke":
        smoke()
    elif args.command == "prepare":
        freeze()
    elif args.command == "run":
        run()
    elif args.command == "worker":
        require(args.dataset is not None, "Worker dataset is required")
        worker(read(MANIFEST), args.dataset)
    else:
        export(read(MANIFEST))

# --- Merged from neural_prompt_sensitivity_test_v2_results.py ---
"""Same-prompt neural/LM contrasts without counting LM as three replicates."""

from collections import defaultdict
import time

import numpy as np

study = sys.modules[__name__]
from braindecoding import pure_language_prompt_sensitivity_test_v2 as language
from braindecoding.pure_language_baseline_test_v2_results import material_weights
from braindecoding.unified_sentence_v2_artifacts import csv_write
from braindecoding.unified_sentence_v2_results import METRICS, holm, metric_row

SEEDS = (0, 1, 2)
LANGUAGE_CONDITIONS = ("LM_A", "LM_R")
NEURAL_CONDITIONS = ("A", "C", "A_original", "C_original")
ALL_CONDITIONS = ("A", "C", "LM_A", "LM_R", "A_original", "C_original")
PAIRS = (("C", "LM_R"), ("A", "LM_A"), ("C", "A"))
NAMES = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}


def condition_seeds(condition):
    study.require(condition in ALL_CONDITIONS, "Unknown comparison condition")
    return (None,) if condition in LANGUAGE_CONDITIONS else SEEDS


def summarize(rows):
    cells = defaultdict(list)
    for row in rows:
        study.require(row["seed"] in condition_seeds(row["condition"]) and row["words"] > 0
            and all(np.isfinite(row[k]) for k in METRICS), "Invalid same-prompt metric row")
        cells[row["condition"], row["seed"], row["subject"]].append(row)
    subjects = []
    for (condition, seed, subject), members in sorted(cells.items()):
        words = sum(r["words"] for r in members)
        values = {k: float(np.mean([r[k] for r in members])) if k == "sentence_bleu1_mean" else
            float(sum(r[k] * r["words"] for r in members) / words) for k in METRICS}
        subjects.append(dict(condition=condition, seed=seed, subject=subject, groups=len(members), words=words, **values))
    sets = [{r["subject"] for r in subjects if r["condition"] == c and r["seed"] == seed}
        for c in ALL_CONDITIONS for seed in condition_seeds(c)]
    study.require(sets and all(s and s == sets[0] for s in sets), "Comparison subjects or seeds differ")
    by_seed = [dict(condition=c, seed=seed,
        **{k: float(np.mean([r[k] for r in subjects if r["condition"] == c and r["seed"] == seed])) for k in METRICS})
        for c in ALL_CONDITIONS for seed in condition_seeds(c)]
    summary = [dict(condition=c, independent_neural_seeds=3 if c in NEURAL_CONDITIONS else 0,
        independent_language_trajectories=1 if c in LANGUAGE_CONDITIONS else 0,
        **{k: float(np.mean([r[k] for r in by_seed if r["condition"] == c])) for k in METRICS}) for c in ALL_CONDITIONS]
    return dict(summary=summary, by_seed=by_seed, by_subject_seed=subjects)


def block_arrays(rows, info):
    blocks = [b["block"] for b in info["blocks"]]
    subjects = sorted({g["subject"] for g in info["groups"]})
    bidx, uidx = {b: i for i, b in enumerate(blocks)}, {u: i for i, u in enumerate(subjects)}
    cidx = {c: i for i, c in enumerate(ALL_CONDITIONS)}
    counts = np.zeros((len(METRICS), len(blocks), len(subjects)))
    values = np.zeros((len(ALL_CONDITIONS), len(METRICS), len(blocks), len(subjects)))
    meta = {g["group"]: g for g in info["groups"]}
    expected = {(c, seed, g) for c in ALL_CONDITIONS for seed in condition_seeds(c) for g in meta}
    seen = set()
    for row in rows:
        key = row["condition"], row["seed"], row["group"]
        study.require(key in expected and key not in seen, "Duplicate or undeclared same-prompt observation")
        seen.add(key)
        group = meta[row["group"]]
        study.require(all(row[k] == group[k] for k in ("subject", "block", "scope"))
            and row["words"] == len(group["event_ids"]), "Paired material or word count differs")
        b, u = bidx[row["block"]], uidx[row["subject"]]
        replicates = len(condition_seeds(row["condition"]))
        for metric, name in enumerate(METRICS):
            weight = 1. if name == "sentence_bleu1_mean" else row["words"]
            if row["condition"] == "A" and row["seed"] == 0:
                counts[metric, b, u] += weight
            values[cidx[row["condition"]], metric, b, u] += weight * row[name] / replicates
    study.require(seen == expected and (counts.sum(axis=1) > 0).all(), "Incomplete same-prompt population")
    return counts, values


def weighted_scores(weights, counts, values):
    output = []
    for metric in range(len(METRICS)):
        denominator = weights @ counts[metric]
        study.require((denominator > 0).all(), "A material draw omitted a subject")
        numerator = np.einsum("rb,cbu->cru", weights, values[:, metric], optimize=True)
        output.append((numerator / denominator[None, :, :]).mean(axis=-1))
    return np.stack(output, axis=1)


def paired_statistics(rows, info, policy, index):
    counts, values = block_arrays(rows, info)
    observed = weighted_scores(np.ones((1, len(info["blocks"]))), counts, values)[..., 0]
    draws = weighted_scores(material_weights(info, policy, index), counts, values)
    blocks = len(info["blocks"])
    exact = blocks <= 16
    if exact:
        numbers = np.arange(2**blocks, dtype=np.uint32)
        signs = 2 * ((numbers[:, None] >> np.arange(blocks, dtype=np.uint32)) & 1).astype(float) - 1
    else:
        rng = np.random.default_rng(np.random.SeedSequence([policy["seed"], index, 271828]))
        signs = rng.choice((-1., 1.), (policy["paired_material_block_sign_flips"], blocks))
    pairs = []
    for condition, control in PAIRS:
        a, b = ALL_CONDITIONS.index(condition), ALL_CONDITIONS.index(control)
        item = dict(comparison=f"{condition}_minus_{control}", condition=condition, control=control,
            groups=len(info["groups"]), subjects=len(counts[0, 0]), blocks=blocks, neural_seeds=3,
            independent_language_trajectories=1 if control in LANGUAGE_CONDITIONS else 0,
            bootstrap_repetitions=draws.shape[-1], sign_flip_repetitions=len(signs), sign_flip_exact=exact, metrics={})
        for metric, name in enumerate(METRICS):
            sign = -1. if name == "wer" else 1.
            effect = sign * (observed[a, metric] - observed[b, metric])
            low, high = np.quantile(sign * (draws[a, metric] - draws[b, metric]), (.025, .975))
            result = dict(condition_value=float(observed[a, metric]), control_value=float(observed[b, metric]),
                improvement=float(effect), ci_lower=float(low), ci_upper=float(high),
                improvement_direction="error_reduction" if name == "wer" else "increase")
            if name == "sentence_bleu1_mean":
                contributions = (sign * (values[a, metric] - values[b, metric])
                    / counts[metric].sum(axis=0)[None, :]).mean(axis=-1)
                study.require(np.isclose(contributions.sum(), effect, atol=1e-12, rtol=0), "Sign-flip block weights differ")
                hits = int((np.abs(signs @ contributions) >= abs(effect) - 1e-12).sum())
                result["p_two_sided"] = float(hits / len(signs) if exact else (1 + hits) / (1 + len(signs)))
            item["metrics"][name] = result
        pairs.append(item)
    changes = []
    expressions = {
        "A_new_minus_original": {"A": 1, "A_original": -1},
        "C_new_minus_original": {"C": 1, "C_original": -1},
        "C_minus_A_original_prompt": {"C_original": 1, "A_original": -1},
        "rolling_gain_new_minus_original": {"C": 1, "A": -1, "C_original": -1, "A_original": 1},
    }
    for comparison, coefficients in expressions.items():
        result = dict(comparison=comparison, metrics={}, interpretation="paired_interval_only_no_extra_significance_test")
        for metric, name in enumerate(METRICS):
            sign = -1. if name == "wer" else 1.
            value = sign * sum(coef * observed[ALL_CONDITIONS.index(c), metric] for c, coef in coefficients.items())
            distribution = sign * sum(coef * draws[ALL_CONDITIONS.index(c), metric] for c, coef in coefficients.items())
            low, high = np.quantile(distribution, (.025, .975))
            result["metrics"][name] = dict(improvement=float(value), ci_lower=float(low), ci_upper=float(high),
                improvement_direction="error_reduction" if name == "wer" else "increase")
        changes.append(result)
    return observed, pairs, changes


def completed_neural(m, parent, job):
    groups = parent["datasets"][job["dataset"]]["groups"]
    path = study.OUT / job["dataset"] / f"seed-{job['seed']:03d}" / "main.json"
    value = study.read(path)
    study.require(value["status"] == "completed", "Neural prompt output is incomplete")
    study.validate_chunk(value, m, job, groups)
    reconstructed = []
    for source in value["sources"]:
        file = study.ROOT / source["path"]
        study.require(study.sha256(file) == source["sha256"], "Committed neural chunk changed")
        part = study.read(file)
        start = len(reconstructed)
        study.validate_chunk(part, m, job, groups[start:start + len(part["records"])])
        reconstructed.extend(part["records"])
    study.require(reconstructed == value["records"], "Neural main differs from its atomic chunks")
    return value


def load_rows(m, parent, lm, dataset):
    info, rows = parent["datasets"][dataset], []
    pure = language.completed_prediction(lm, dataset)
    for group, record in zip(info["groups"], pure["records"]):
        identity = dict(dataset=dataset, subject=group["subject"], group=group["group"], material=group["material"],
            block=group["block"], scope=group["scope"], words=len(group["event_ids"]))
        for condition in LANGUAGE_CONDITIONS:
            prediction = record["result"]["final"][condition]
            study.require(len(prediction) == identity["words"], "Language baseline word count differs")
            rows.append(dict(**identity, seed=None, condition=condition,
                **metric_row(group["reference"], prediction, info["language"] == "zh")))
    for job in [j for j in m["jobs"] if j["dataset"] == dataset]:
        new = completed_neural(m, parent, job)
        old = study.original_ac(parent, dataset, job["seed"], study.ROOT / job["original_ac"])
        for group, current, previous in zip(info["groups"], new["records"], old["records"]):
            identity = dict(dataset=dataset, seed=job["seed"], subject=group["subject"], group=group["group"],
                material=group["material"], block=group["block"], scope=group["scope"], words=len(group["event_ids"]))
            for suffix, record in (("", current), ("_original", previous)):
                for condition in study.CONDITIONS:
                    rows.append(dict(**identity, condition=condition + suffix,
                        **metric_row(group["reference"], record["result"]["final"][condition], info["language"] == "zh")))
    return rows


def export(m):
    from threadpoolctl import threadpool_limits
    started = time.perf_counter()
    status = study.read(study.RUNTIME / "status.json")
    study.require(status.get("GPU_workers_exited") is True and status.get("worker_pid") is None,
        "Dataset GPU workers must exit before final export")
    study.verify_sources(m, full=True)
    parent, lm = study.read(study.PARENT), study.read(study.LM_MANIFEST)
    study.require(lm["manifest_sha256"] == m["language_manifest_sha256"], "Same-prompt LM identity differs")
    old_summary, lm_summary = study.read(study.OLD_SUMMARY), study.read(study.LM_REPORT / "results.json")
    datasets, pairs, prompt_changes = {}, [], []
    all_rows, seed_rows, subject_rows, summary_rows, sources = [], [], [], [], {}
    with threadpool_limits(limits=4):
        for index, dataset in enumerate(study.DATASETS):
            info = parent["datasets"][dataset]
            done = study.read(study.OUT / dataset / "completion.json")
            study.require(done["status"] == "completed" and done["manifest_sha256"] == m["manifest_sha256"]
                and done["completed_segment_seed_units"] == len(info["groups"]) * 3, "Dataset output is incomplete")
            study.committed_prefix(study.plan(m, parent, dataset), m, dataset)
            rows = load_rows(m, parent, lm, dataset)
            aggregate = summarize(rows)
            lookup = {r["condition"]: r for r in aggregate["summary"]}
            for source, conditions, suffix in ((old_summary, study.CONDITIONS, "_original"),
                (lm_summary, LANGUAGE_CONDITIONS, "")):
                expected = {r["condition"]: r for r in source["datasets"][dataset]["summary"]
                    if r.get("prompt_version", "selected_prompt") == "selected_prompt"}
                for c in conditions:
                    study.require(all(np.isclose(lookup[c + suffix][k], expected[c][k], rtol=0, atol=1e-12) for k in METRICS),
                        f"Original or matched-language metric not reproduced: {dataset} {c}")
            observed, comparisons, changes = paired_statistics(rows, info, m["settings"]["statistics"], index)
            for c, condition in enumerate(ALL_CONDITIONS):
                study.require(all(np.isclose(observed[c, k], lookup[condition][name], atol=1e-12, rtol=0)
                    for k, name in enumerate(METRICS)), "Bootstrap weighting differs from direct summary")
            datasets[dataset] = dict(counts=info["counts"], prompt=m["prompts"][dataset], **aggregate,
                original_A_C_metrics_reproduced=True, same_prompt_language_metrics_reproduced=True)
            pairs.extend(dict(dataset=dataset, **r) for r in comparisons)
            prompt_changes.extend(dict(dataset=dataset, **r) for r in changes)
            all_rows.extend(rows)
            seed_rows.extend(dict(dataset=dataset, **r) for r in aggregate["by_seed"])
            subject_rows.extend(dict(dataset=dataset, **r) for r in aggregate["by_subject_seed"])
            summary_rows.extend(dict(dataset=dataset, **r) for r in aggregate["summary"])
            for job in [j for j in m["jobs"] if j["dataset"] == dataset]:
                path = study.OUT / dataset / f"seed-{job['seed']:03d}" / "main.json"
                sources[study.source_key(path)] = study.sha256(path)
    study.require(len(pairs) == 6, "The six same-prompt BLEU comparisons are incomplete")
    corrected = holm([r["metrics"]["sentence_bleu1_mean"]["p_two_sided"] for r in pairs])
    for row, value in zip(pairs, corrected):
        row["metrics"]["sentence_bleu1_mean"].update(p_holm=float(value), significant_holm_05=bool(value < .05))
    pair_rows = [dict(dataset=r["dataset"], comparison=r["comparison"], metric=name,
        condition_value=v["condition_value"], control_value=v["control_value"], improvement=v["improvement"],
        ci_lower=v["ci_lower"], ci_upper=v["ci_upper"], p_two_sided=v.get("p_two_sided"), p_holm=v.get("p_holm"),
        improvement_direction=v["improvement_direction"]) for r in pairs for name, v in r["metrics"].items()]
    change_rows = [dict(dataset=r["dataset"], comparison=r["comparison"], metric=name, **v)
        for r in prompt_changes for name, v in r["metrics"].items()]
    study.REPORT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("by_group", all_rows), ("by_seed", seed_rows), ("by_subject_seed", subject_rows),
        ("summary", summary_rows), ("comparisons", pair_rows), ("prompt_changes", change_rows)):
        path = study.REPORT / f"{name}.csv"
        csv_write(path, rows)
        sources[study.source_key(path)] = study.sha256(path)
    result = dict(status="completed", protocol=study.ID, split="test", test_evaluated=True,
        manifest_sha256=m["manifest_sha256"], parent_manifest_sha256=m["parent_manifest_sha256"],
        language_manifest_sha256=m["language_manifest_sha256"], changed_factor="fixed_language_prompt_only",
        checkpoints_unchanged=True, neural_training=False, language_weight=.1, neural_seeds=list(SEEDS),
        language_replicates_per_policy=1, neural_and_language_prompts_identical=True,
        language_baselines_not_duplicated_as_seeds=True, all_OOV_reference_words_retained=True,
        datasets=datasets, comparisons=pairs, prompt_changes=prompt_changes, source_sha256=sources,
        total_segment_seed_units=m["total_segment_seed_units"], GPU_workers_exited=True,
        interpretation="same_prompt_system_contrast_including_candidate_and_generated_prefix_interactions_not_unique_neural_share",
        elapsed_cpu_seconds=time.perf_counter() - started, completed_unix=time.time())
    study.write_json(study.REPORT / "results.json", result)
    write_markdown(result)
    study.verify_sources(m, full=True)
    study.write_json(study.REPORT / "audit.json", dict(status="verified_complete", manifest_sha256=m["manifest_sha256"],
        result_sha256=study.sha256(study.REPORT / "results.json"), report_sha256=study.sha256(study.REPORT / "results.md"),
        source_sha256=m["source_sha256"], exports=sources, checkpoints_unchanged=True,
        exact_prompt_template_and_tokens_verified=True, language_weight=.1, complete_test_population=True,
        original_A_C_and_same_prompt_LM_metrics_reproduced=True, one_language_trajectory_per_policy=True,
        shared_material_resampling=True, primary_BLEU_Holm_family_size=6, GPU_dataset_workers_exited=True))
    return result


def write_markdown(result):
    lines = ["# 同一叙述提示：A/C 与纯语言对照", "",
        "两个数据集全部测试片段、三个原神经检查点。λ=0.1，模型、神经向量、词库、评分和搜索预算不变；提示与已完成的纯语言补跑逐字符和 token 一致。", "",
        "BLEU-1 为受试者内片段等权、再受试者和神经种子等权；WER 与 Acc 为受试者内参考词位合并、再受试者和种子等权。Acc 是标注词位上的完全匹配。纯语言每策略一份轨迹。", "",
        "| 数据集 | 条件 | BLEU-1 (%) | WER (%) | Acc (%) |", "|---|---|---:|---:|---:|"]
    for dataset, info in result["datasets"].items():
        for row in info["summary"]:
            lines.append(f"| {NAMES[dataset]} | {row['condition']} | {100*row['sentence_bleu1_mean']:.2f} | {100*row['wer']:.2f} | {100*row['aligned_word_accuracy']:.2f} |")
    lines += ["", "## 同提示的配对差值", "",
        "差值单位为百分点；每个单元格给出增益及配对 95% CI。WER 正值表示错误减少。5 材料块、记录/章节内分层 bootstrap 10,000 次；所有受试者和种子共享抽样。六项主 BLEU 双侧检验共同 Holm 校正，WER 和 Acc 只报告区间。", "",
        "| 数据集 | 比较 | BLEU-1 增益 [95% CI] | Holm p | WER 减少 [95% CI] | Acc 增益 [95% CI] |",
        "|---|---|---|---:|---|---|"]
    def cell(value):
        return f"{100*value['improvement']:+.2f} [{100*value['ci_lower']:+.2f}, {100*value['ci_upper']:+.2f}]"
    for row in result["comparisons"]:
        b, w, a = (row["metrics"][k] for k in ("sentence_bleu1_mean", "wer", "aligned_word_accuracy"))
        lines.append(f"| {NAMES[row['dataset']]} | {row['condition']} − {row['control']} | {cell(b)} | {b['p_holm']:.6f} | {cell(w)} | {cell(a)} |")
    lines += ["", "## 原提示的成对参考", "", "以下仅报告配对区间，不增加显著性检验。rolling_gain_new_minus_original 为新提示 C−A 减去旧提示 C−A。", "",
        "| 数据集 | 比较 | BLEU-1 增益 [95% CI] | WER 减少 [95% CI] | Acc 增益 [95% CI] |", "|---|---|---|---|---|"]
    for row in result["prompt_changes"]:
        lines.append(f"| {NAMES[row['dataset']]} | {row['comparison']} | " + " | ".join(cell(row["metrics"][k])
            for k in ("sentence_bleu1_mean", "wer", "aligned_word_accuracy")) + " |")
    lines += ["", "同提示神经减纯语言是系统层面的匹配对照，仍包含神经候选约束和生成前缀交互，不能换算为唯一神经贡献比例。新提示 C−A 可检验在该提示下是否仍有滚动收益；提示交互区间跨零不证明提示完全无影响。本组不能排除预训练材料记忆。", "",
        "[完整结果](results.json) · [逐片段指标](by_group.csv) · [配对比较](comparisons.csv) · [来源审计](audit.json)"]
    (study.REPORT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
