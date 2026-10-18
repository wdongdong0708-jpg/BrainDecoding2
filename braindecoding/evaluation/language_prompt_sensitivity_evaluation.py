"""Neural-free complete-word LM controls, with the original known boundaries."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import pickle
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


ID = "pure_language_prompt_sensitivity_test_v2"
DATASETS = ("pallier2025", "chineseeeg2_littleprince")
CONDITIONS = ("LM_A", "LM_C1", "LM_R")
MATCH = {"A": "LM_A", "B": "LM_R", "C1": "LM_C1", "C": "LM_R"}
DELAYS = {"LM_A": 0, "LM_C1": 1, "LM_R": 2}
CONFIG = ROOT / f"configs/unified_rolling/{ID}.yaml"
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
    """A private cache is bound to the exact prompt as well as the model."""
    def cache_identity(self):
        return digest(dict(language=self.language, prompt_ids=list(self.prompt_ids),
            revision=self.settings["revision"], score=self.settings["score"],
            context_tokens=self.context_tokens, score_batch_size=self.settings["score_batch_size"],
            proposal_count=self.settings["proposal_count"],
            proposal_max_new_tokens=self.settings["proposal_max_new_tokens"]))

    def save_cache(self, path):
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(dict(format="prompt_bound_lm_v1", identity=self.cache_identity(),
                scores=dict(self.cache), proposals=dict(self.proposal_cache)), handle, protocol=5)
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(min(.025 * (attempt + 1), .2))

    def load_cache(self, path):
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
        require(isinstance(value, dict) and value.get("format") == "prompt_bound_lm_v1"
            and value.get("identity") == self.cache_identity(), "Prompt/model cache mismatch")
        require(np.isfinite(list(value["scores"].values())).all(), "Non-finite cached language score")
        self.cache.update(value["scores"])
        self.proposal_cache.update(value["proposals"])

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
    pilot = cfg["prompt_diagnostic"]
    require(pilot["selection"] == "first_predeclared_prompt_passing_generation_checks"
        and pilot["word_count"] == 12 and not pilot["use_test_references"]
        and pilot["candidate_ids"] == ["native_prose", "generic_prose", "story_opening"]
        and all(set(pilot["templates"][key]) == {"fr", "zh"} for key in pilot["candidate_ids"]),
        "Prompt diagnostic policy differs")
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
    require(smoke["code_sha256"] == sha256(__file__) and smoke["config_sha256"] == sha256(CONFIG)
        and smoke["policy"] == cfg["prompt_diagnostic"] and smoke["selection_uses_test_metrics"] is False,
        "Diagnostic provenance or selection policy differs")
    from braindecoding import pure_language_baseline_test_v2 as original
    old_manifest = read(original.MANIFEST)
    original.verify_sources(old_manifest, full=True)
    refs, source_paths = reference_inventory(parent)
    source_paths.update((original.MANIFEST, original.REPORT / "results.json",
        original.REPORT / "audit.json", ROOT / "braindecoding/evaluation/pure_language_baseline_test_v2.py",
        ROOT / "braindecoding/evaluation/pure_language_baseline_test_v2_results.py"))
    for dataset in DATASETS:
        original.completed_prediction(old_manifest, dataset)
        source_paths.add(original.OUT / dataset / "main.json")
    for entry in smoke["rows"]:
        for probe in entry["probes"]:
            source_paths.add(RUNTIME / "diagnostic" / f"{entry['dataset']}-{probe['prompt_id']}.json")
    source_paths.update((CONFIG, PROTOCOL, PARENT, Path(__file__), ROOT / "braindecoding/evaluation/pure_language_prompt_sensitivity_test_v2_results.py",
        ROOT / "braindecoding/decoding/unified_sentence.py", ROOT / "braindecoding/decoding/unified_sentence_batch.py",
        ROOT / "braindecoding/decoding/unified_sentence_v2.py", ROOT / "braindecoding/evaluation/unified_sentence_v2_results.py",
        ROOT / "braindecoding/decoding/unified_sentence_v2_artifacts.py", ROOT / "braindecoding/decoding/jayalath_sentence.py",
        ROOT / "tests/test_pure_language_prompt_sensitivity_test_v2.py", RUNTIME / "gpu_smoke.json"))
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
        selected = next(r for r in smoke["rows"] if r["dataset"] == dataset)
        datasets[dataset] = dict(language=info["language"], vocabulary=list(info["vocabulary"]), groups=groups,
            prompt_id=selected["selected_prompt_id"], prompt_template=selected["prompt_template"],
            prompt_template_sha256=digest(selected["prompt_template"]),
            independent_first_three_arrivals=selected["independent_first_three_arrivals"],
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
        original_language_manifest_sha256=old_manifest["manifest_sha256"],
        changed_factor="fixed_language_prompt_only", formal_comparisons="new_prompt_minus_original_prompt_language_only",
        neural_comparisons="descriptive_cross_prompt_only_no_causal_attribution",
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


def generation_diagnostics(words, policy):
    counts = Counter(words)
    longest, run, previous = 0, 0, None
    for word in words:
        run = run + 1 if word == previous else 1
        longest = max(longest, run)
        previous = word
    value = dict(words=len(words), unique_words=len(counts),
        digit_words=sum(any(c.isdigit() for c in w) for w in words),
        largest_word_fraction=max(counts.values(), default=0) / max(len(words), 1),
        longest_identical_run=longest)
    value["passed"] = bool(value["words"] == policy["word_count"]
        and value["unique_words"] >= policy["minimum_unique_words"]
        and value["digit_words"] <= policy["maximum_digit_words"]
        and value["largest_word_fraction"] <= policy["maximum_word_fraction"]
        and value["longest_identical_run"] <= policy["maximum_identical_run"])
    return value


def select_prompt(probes, policy):
    for candidate in policy["candidate_ids"]:
        found = [r for r in probes if r["prompt_id"] == candidate]
        require(len(found) <= 1, "Duplicate prompt probe")
        if found and all(found[0]["diagnostics"][c]["passed"] for c in CONDITIONS):
            return found[0]
    raise ValueError("All predeclared prompts failed generation diagnostics; full test was not started")


def language_settings(m, dataset):
    lm = copy.deepcopy(m["language_model"])
    lm["language_prompt_template"] = m["datasets"][dataset]["prompt_template"]
    return lm


def smoke():
    """Select without test references or BLEU: generated text and train lexicon only."""
    from braindecoding import pure_language_baseline_test_v2 as original
    cfg, previous = settings(), read(original.MANIFEST)
    verify_identity(previous)
    code_hash, config_hash = sha256(__file__), sha256(CONFIG)
    lm = copy.deepcopy(previous["language_model"])
    for path, expected in previous["source_sha256"].items():
        if str((ROOT / lm["local_path"]).resolve()) in path:
            require(sha256(path) == expected, "Original language model changed")
    gpu_ready(cfg["execution"])
    torch.set_num_threads(cfg["execution"]["cpu_threads"])
    torch.manual_seed(0)
    model, tokenizer = load_model(lm)
    rows = []
    policy = cfg["prompt_diagnostic"]
    try:
        for dataset in DATASETS:
            info = previous["datasets"][dataset]
            probes = []
            for candidate in policy["candidate_ids"]:
                template = policy["templates"][candidate][info["language"]]
                settings_now = copy.deepcopy(lm)
                settings_now["language_prompt_template"] = template
                scorer = FiniteWordLM(model, tokenizer, info["language"], settings_now)
                started = time.perf_counter()
                outputs, trace = {c: [] for c in CONDITIONS}, []
                for arrival in range(policy["word_count"]):
                    row = advance(outputs, arrival, info["vocabulary"], scorer, previous["decoder"])
                    trace.append(row)
                    outputs = {c: row[c] for c in CONDITIONS}
                    print(json.dumps(dict(dataset=dataset, prompt_id=candidate,
                        completed_diagnostic_arrivals=len(trace)), ensure_ascii=False), flush=True)
                validate_trace(trace)
                probe = dict(dataset=dataset, prompt_id=candidate, prompt_template=template,
                    prompt_ids=list(scorer.prompt_ids), result=crop_result(trace, policy["word_count"]),
                    diagnostics={c: generation_diagnostics(outputs[c], policy) for c in CONDITIONS},
                    seconds=time.perf_counter() - started, vocabulary=len(info["vocabulary"]),
                    score_cache_entries=len(scorer.cache),
                    peak_gpu_memory_gib=torch.cuda.max_memory_allocated() / 1024**3,
                    test_reference_reads_for_selection=0, neural_array_reads=0, text_embedding_calls=0)
                probes.append(probe)
                write_json(RUNTIME / "diagnostic" / f"{dataset}-{candidate}.json", probe)
                print(json.dumps(dict(dataset=dataset, prompt_id=candidate, diagnostics=probe["diagnostics"],
                    final=outputs), ensure_ascii=False), flush=True)
                if all(probe["diagnostics"][c]["passed"] for c in CONDITIONS):
                    break
                del scorer
            selected = select_prompt(probes, policy)
            rows.append(dict(dataset=dataset, selected_prompt_id=selected["prompt_id"],
                prompt_template=selected["prompt_template"], prompt_ids=selected["prompt_ids"],
                independent_first_three_arrivals=selected["result"]["trace"][:3], probes=probes))
        require(code_hash == sha256(__file__) and config_hash == sha256(CONFIG),
            "Prompt diagnostic code/config changed during execution")
        value = dict(status="passed", rows=rows, language_model=lm, exact_original_scoring=True,
            changed_factor="fixed_language_prompt_only", selection_uses_test_metrics=False,
            policy=policy, code_sha256=code_hash, config_sha256=config_hash,
            original_manifest_sha256=previous["manifest_sha256"])
        write_json(RUNTIME / "gpu_smoke.json", value)
        return value
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


def completed_prediction(m, dataset):
    path = OUT / dataset / "main.json"
    value = read(path)
    require(value["status"] == "completed" and value["manifest_sha256"] == m["manifest_sha256"]
            and value["dataset"] == dataset and value["conditions"] == list(CONDITIONS)
            and value["neural_input_allowed"] is False and value["reference_text_in_decoder"] is False,
            "Pure language output provenance differs")
    info = m["datasets"][dataset]
    require(value["prompt_id"] == info["prompt_id"]
        and value["prompt_template_sha256"] == info["prompt_template_sha256"], "Language prompt provenance differs")
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
        require(row["result"]["arrival"] == len(reconstructed)
            and row["prompt_id"] == info["prompt_id"]
            and row["prompt_template_sha256"] == info["prompt_template_sha256"], "Language arrival prompt/order differs")
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
                    conditions=list(CONDITIONS), neural_input_allowed=False, reference_text_in_decoder=False,
                    prompt_id=info["prompt_id"], prompt_template_sha256=info["prompt_template_sha256"])
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
    scorer = FiniteWordLM(model, tokenizer, info["language"], language_settings(m, dataset))
    if latest_cache:
        scorer.load_cache(latest_cache["path"])
    outputs = {c: trace[-1][c] if trace else [] for c in CONDITIONS}
    try:
        for arrival in range(len(trace), info["maximum_words"]):
            verify_sources(m)
            status(status="language_decoding", arrival=arrival, completed_arrivals=len(trace))
            started = time.perf_counter()
            row = advance(outputs, arrival, info["vocabulary"], scorer, m["decoder"])
            if arrival < 3:
                require(row == info["independent_first_three_arrivals"][arrival],
                    "Fresh full-test search differs from independent prompt diagnostic")
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
        "新提示的纯语言敏感性对照；原词数、片段边界、词库、Qwen 模型、评分和束搜索规则一致。",
        "三种搜索策略均补跑；新提示在训练词库生成检查中固定，未按测试指标选择。",
        "神经策略使用原提示，跨提示的差值仅作描述性参考。", ""]
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
                command = [sys.executable, "-m", "braindecoding.pure_language_prompt_sensitivity_test_v2", "worker", "--dataset", dataset]
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
    parser.add_argument("command", choices=("smoke", "calibrate", "prepare", "run", "worker", "export"))
    parser.add_argument("--dataset", choices=DATASETS)
    args = parser.parse_args(argv)
    if args.command in ("smoke", "calibrate"):
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

# --- Merged from pure_language_prompt_sensitivity_test_v2_results.py ---
"""Paired old/new language prompts; neural levels are descriptive references."""

from collections import Counter, defaultdict
import time

import numpy as np

study = sys.modules[__name__]
from braindecoding import pure_language_baseline_test_v2 as original
from braindecoding.unified_sentence_v2_results import METRICS, holm, metric_row
from braindecoding.unified_sentence_v2_artifacts import csv_write

VERSIONS = ("original_prompt", "selected_prompt")
NAMES = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}


def summarize(rows):
    cells = defaultdict(list)
    for row in rows:
        study.require(row["prompt_version"] in VERSIONS and row["condition"] in study.CONDITIONS
            and row["words"] > 0 and all(np.isfinite(row[k]) for k in METRICS), "Invalid language metric")
        cells[row["prompt_version"], row["subject"], row["condition"]].append(row)
    subjects = []
    for (version, subject, condition), members in sorted(cells.items()):
        words = sum(r["words"] for r in members)
        values = {k: float(np.mean([r[k] for r in members])) if k == "sentence_bleu1_mean" else
            float(sum(r[k] * r["words"] for r in members) / words) for k in METRICS}
        subjects.append(dict(prompt_version=version, subject=subject, condition=condition,
            groups=len(members), words=words, **values))
    sets = [{r["subject"] for r in subjects if r["prompt_version"] == v and r["condition"] == c}
        for v in VERSIONS for c in study.CONDITIONS]
    study.require(sets and all(s and s == sets[0] for s in sets), "Prompt comparison subjects differ")
    summary = [dict(prompt_version=v, condition=c,
        **{k: float(np.mean([r[k] for r in subjects if r["prompt_version"] == v
            and r["condition"] == c])) for k in METRICS}) for v in VERSIONS for c in study.CONDITIONS]
    return dict(summary=summary, by_subject=subjects)


def block_arrays(rows, info):
    blocks = [b["block"] for b in info["blocks"]]
    subjects = sorted({g["subject"] for g in info["groups"]})
    bidx, uidx = {b: i for i, b in enumerate(blocks)}, {u: i for i, u in enumerate(subjects)}
    counts = np.zeros((len(METRICS), len(blocks), len(subjects)))
    values = np.zeros((2, len(study.CONDITIONS), len(METRICS), len(blocks), len(subjects)))
    meta = {g["group"]: g for g in info["groups"]}
    expected = {(v, c, g) for v in VERSIONS for c in study.CONDITIONS for g in meta}
    seen = set()
    for row in rows:
        key = row["prompt_version"], row["condition"], row["group"]
        study.require(key in expected and key not in seen, "Duplicate or undeclared prompt observation")
        seen.add(key)
        group = meta[row["group"]]
        study.require(all(row[k] == group[k] for k in ("subject", "scope", "block"))
            and row["words"] == len(group["event_ids"]), "Prompt comparison material differs")
        v, c = VERSIONS.index(row["prompt_version"]), study.CONDITIONS.index(row["condition"])
        b, u = bidx[row["block"]], uidx[row["subject"]]
        for metric, name in enumerate(METRICS):
            weight = 1. if name == "sentence_bleu1_mean" else row["words"]
            if v == 0 and c == 0:
                counts[metric, b, u] += weight
            values[v, c, metric, b, u] += weight * row[name]
    study.require(seen == expected and (counts.sum(axis=1) > 0).all(), "Incomplete prompt comparison population")
    return counts, values


def weighted_scores(weights, counts, values):
    output = []
    for metric in range(len(METRICS)):
        denominator = weights @ counts[metric]
        study.require((denominator > 0).all(), "A material draw omitted a subject")
        numerator = np.einsum("rb,vcbu->vcru", weights, values[:, :, metric], optimize=True)
        output.append((numerator / denominator[None, None, :, :]).mean(axis=-1))
    return np.stack(output, axis=2)


def paired_statistics(rows, info, policy, index):
    from braindecoding.pure_language_baseline_test_v2_results import material_weights
    counts, values = block_arrays(rows, info)
    observed = weighted_scores(np.ones((1, len(info["blocks"]))), counts, values)[..., 0]
    draws = weighted_scores(material_weights(info, policy, index), counts, values)
    rng = np.random.default_rng(np.random.SeedSequence([policy["seed"], index, 271828]))
    blocks = len(info["blocks"])
    exact = blocks <= 16
    if exact:
        numbers = np.arange(2**blocks, dtype=np.uint32)
        signs = 2 * ((numbers[:, None] >> np.arange(blocks, dtype=np.uint32)) & 1).astype(float) - 1
    else:
        signs = rng.choice((-1., 1.), (policy["paired_material_block_sign_flips"], blocks))
    comparisons = []
    for c, condition in enumerate(study.CONDITIONS):
        row = dict(condition=condition, comparison=f"{condition}_new_prompt_minus_original_prompt",
            groups=len(info["groups"]), subjects=len(counts[0, 0]), blocks=blocks,
            independent_language_replicates_per_prompt=1, bootstrap_repetitions=draws.shape[-1],
            sign_flip_repetitions=len(signs), sign_flip_exact=exact, metrics={})
        for metric, name in enumerate(METRICS):
            sign = -1. if name == "wer" else 1.
            effect = sign * (observed[1, c, metric] - observed[0, c, metric])
            delta = sign * (draws[1, c, metric] - draws[0, c, metric])
            low, high = np.quantile(delta, (.025, .975))
            contributions = (sign * (values[1, c, metric] - values[0, c, metric])
                / counts[metric].sum(axis=0)[None, :]).mean(axis=-1)
            study.require(np.isclose(contributions.sum(), effect, atol=1e-12, rtol=0), "Prompt block weighting differs")
            hits = (np.abs(signs @ contributions) >= abs(effect) - 1e-12).sum()
            p = float(hits / len(signs) if exact else (1 + hits) / (1 + len(signs)))
            row["metrics"][name] = dict(original=float(observed[0, c, metric]), selected=float(observed[1, c, metric]),
                improvement=float(effect), ci_lower=float(low), ci_upper=float(high), p_two_sided=p,
                improvement_direction="error_reduction" if sign < 0 else "increase")
        comparisons.append(row)
    within = []
    for c, condition in enumerate(study.CONDITIONS[1:], 1):
        for metric, name in enumerate(METRICS):
            sign = -1. if name == "wer" else 1.
            delta = sign * (draws[1, c, metric] - draws[1, 0, metric])
            low, high = np.quantile(delta, (.025, .975))
            within.append(dict(condition=condition, control="LM_A", metric=name,
                improvement=float(sign * (observed[1, c, metric] - observed[1, 0, metric])),
                ci_lower=float(low), ci_upper=float(high), interpretation="within_selected_prompt_search_comparison"))
    return comparisons, within


def load_rows(m, parent, dataset):
    old_m = study.read(original.MANIFEST)
    study.require(old_m["manifest_sha256"] == m["original_language_manifest_sha256"], "Original LM identity differs")
    payloads = {"original_prompt": original.completed_prediction(old_m, dataset),
        "selected_prompt": study.completed_prediction(m, dataset)}
    predictions = {v: {r["group"]: r["result"]["final"] for r in p["records"]} for v, p in payloads.items()}
    info = parent["datasets"][dataset]
    expected = {g["group"] for g in info["groups"]}
    study.require(all(set(p) == expected for p in predictions.values()), "Prompt group population differs")
    rows, output_counts = [], {v: {c: Counter() for c in study.CONDITIONS} for v in VERSIONS}
    for group in info["groups"]:
        identity = dict(dataset=dataset, subject=group["subject"], group=group["group"], material=group["material"],
            block=group["block"], scope=group["scope"], words=len(group["reference"]))
        for version in VERSIONS:
            final = predictions[version][group["group"]]
            study.require(set(final) == set(study.CONDITIONS), "Prompt conditions differ")
            for condition in study.CONDITIONS:
                prediction = final[condition]
                study.require(len(prediction) == identity["words"], "Prompt known word count differs")
                rows.append(dict(**identity, prompt_version=version, condition=condition,
                    **metric_row(group["reference"], prediction, info["language"] == "zh")))
                output_counts[version][condition].update(prediction)
    counts = {v: {c: dict(total_words=sum(n.values()), unique_words=len(n),
        digit_words=sum(k for word, k in n.items() if any(ch.isdigit() for ch in word)),
        most_common=n.most_common(10)) for c, n in columns.items()} for v, columns in output_counts.items()}
    return rows, counts


def export(m):
    from threadpoolctl import threadpool_limits
    started = time.perf_counter()
    study.verify_sources(m, full=True)
    original.verify_sources(study.read(original.MANIFEST), full=True)
    parent = study.read(study.PARENT)
    old_result = study.read(original.REPORT / "results.json")
    all_rows, subject_rows, summaries, comparisons, within, descriptive = [], [], [], [], [], []
    datasets = {}
    with threadpool_limits(limits=2):
        for index, dataset in enumerate(study.DATASETS):
            rows, output_counts = load_rows(m, parent, dataset)
            aggregate = summarize(rows)
            current = {(r["prompt_version"], r["condition"]): r for r in aggregate["summary"]}
            previous = {r["condition"]: r for r in old_result["datasets"][dataset]["summary"]}
            for condition in study.CONDITIONS:
                study.require(all(np.isclose(current["original_prompt", condition][k], previous[condition][k],
                    atol=1e-12, rtol=0) for k in METRICS), "Original pure-language metrics changed")
            paired, local = paired_statistics(rows, parent["datasets"][dataset], m["settings"]["statistics"], index)
            for item in paired:
                for metric, values in item["metrics"].items():
                    study.require(np.isclose(values["selected"], current["selected_prompt", item["condition"]][metric],
                        atol=1e-12, rtol=0), "Prompt paired metric differs from summary")
            neural = [dict(r) for r in old_result["datasets"][dataset]["summary"] if r["condition"] in study.MATCH]
            for row in neural:
                condition, control = row["condition"], study.MATCH[row["condition"]]
                for metric in METRICS:
                    sign = -1. if metric == "wer" else 1.
                    descriptive.append(dict(dataset=dataset, neural_condition=condition, language_control=control,
                        metric=metric, neural_original_prompt=row[metric], language_selected_prompt=current["selected_prompt", control][metric],
                        difference=float(sign * (row[metric] - current["selected_prompt", control][metric])),
                        interpretation="cross_prompt_descriptive_not_matched_neural_ablation", causal_attribution_allowed=False))
            datasets[dataset] = dict(counts=parent["datasets"][dataset]["counts"], **aggregate,
                selected_prompt={k: m["datasets"][dataset][k] for k in ("prompt_id", "prompt_template", "prompt_template_sha256")},
                output_diagnostics=output_counts, neural_original_prompt_reference=neural)
            all_rows.extend(rows)
            subject_rows.extend(dict(dataset=dataset, **r) for r in aggregate["by_subject"])
            summaries.extend(dict(dataset=dataset, **r) for r in aggregate["summary"])
            comparisons.extend(dict(dataset=dataset, **r) for r in paired)
            within.extend(dict(dataset=dataset, **r) for r in local)
    study.require(len(comparisons) == 6, "Prompt comparison family differs")
    for row, p in zip(comparisons, holm([r["metrics"]["sentence_bleu1_mean"]["p_two_sided"] for r in comparisons])):
        row["metrics"]["sentence_bleu1_mean"].update(p_holm=float(p), significant_holm_05=bool(p < .05))
    study.REPORT.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, values in (("by_group", all_rows), ("by_subject", subject_rows), ("summary", summaries),
        ("within_prompt_search", within), ("cross_prompt_neural_descriptive", descriptive)):
        file = study.REPORT / f"{name}.csv"
        csv_write(file, values)
        sources[str(file)] = study.sha256(file)
    for dataset in study.DATASETS:
        file = study.OUT / dataset / "main.json"
        sources[str(file)] = study.sha256(file)
    result = dict(status="completed", protocol=study.ID, split="test", test_evaluated=True,
        manifest_sha256=m["manifest_sha256"], parent_manifest_sha256=m["parent_manifest_sha256"],
        original_language_manifest_sha256=m["original_language_manifest_sha256"], changed_factor="fixed_language_prompt_only",
        original_pure_language_metrics_reproduced=True, independent_language_replicates_per_prompt=1,
        neural_input_in_baseline=False, reference_text_in_prompt=False, test_metrics_used_for_prompt_selection=False,
        neural_reference_has_same_prompt=False, neural_reference_comparisons="descriptive_cross_prompt_only",
        datasets=datasets, prompt_comparisons=comparisons, within_prompt_search=within,
        cross_prompt_neural_descriptive=descriptive, source_sha256=sources,
        elapsed_cpu_seconds=time.perf_counter() - started, completed_unix=time.time())
    study.write_json(study.REPORT / "results.json", result)
    write_markdown(result)
    study.verify_sources(m, full=True)
    original.verify_sources(study.read(original.MANIFEST), full=True)
    pre = study.read(study.RUNTIME / "pre_diagnostic_policy.json")
    diagnostic = study.read(study.RUNTIME / "gpu_smoke.json")
    study.require(pre["policy"] == diagnostic["policy"]
        and pre["code_sha256"] == diagnostic["code_sha256"]
        and pre["config_sha256"] == diagnostic["config_sha256"]
        and pre["protocol_sha256"] == study.sha256(study.PROTOCOL), "Pre-diagnostic policy provenance differs")
    study.write_json(study.REPORT / "audit.json", dict(status="verified_complete", manifest_sha256=m["manifest_sha256"],
        result_sha256=study.sha256(study.REPORT / "results.json"), report_sha256=study.sha256(study.REPORT / "results.md"),
        source_sha256=sources, pre_diagnostic_policy_sha256=study.sha256(study.RUNTIME / "pre_diagnostic_policy.json"),
        original_sources_unchanged=True, original_pure_language_metrics_reproduced=True,
        identical_word_count_boundaries_vocabulary_and_search=True, prompt_bound_private_cache=True,
        test_metrics_used_for_prompt_selection=False, reference_words_enter_decoder=False,
        baseline_neural_array_reads=0, independent_language_replicates_per_prompt=1,
        neural_references_cross_prompt_only=True))
    print("Prompt-sensitivity language results exported", flush=True)
    return result


def write_markdown(result):
    lines = ["# 纯语言基线：固定提示敏感性补跑", "",
        "两个数据集全部测试片段，三种搜索策略各一个确定语言轨迹。只改变固定语言提示；完整训练词库、数字候选、词数、边界、模型、评分和搜索预算沿用原实验。",
        "提示在训练词库无参考生成检查中选定，未按测试指标选择。表中 BLEU-1 为受试者内片段等权再受试者等权，WER 为受试者内词位合并后等权。", "",
        "| 数据集 | 提示 | 策略 | BLEU-1 | WER | 语料 BLEU-1 | 对齐词准确率 |",
        "|---|---|---|---:|---:|---:|---:|"]
    for dataset, info in result["datasets"].items():
        for row in info["summary"]:
            version = "原提示" if row["prompt_version"] == "original_prompt" else "新提示"
            lines.append(f"| {NAMES[dataset]} | {version} | {row['condition']} | {100*row['sentence_bleu1_mean']:.2f}% | {100*row['wer']:.2f}% | {100*row['bleu1']:.2f}% | {100*row['aligned_word_accuracy']:.2f}% |")
    lines += ["", "## 新提示相对原提示", "",
        "5 材料块配对 bootstrap 10,000 次；六项双侧 BLEU 检验共同 Holm 校正。WER 正差值表示错误减少。", "",
        "| 数据集 | 策略 | BLEU 增加（百分点） | 95% CI | Holm p | WER 减少（百分点） | WER 95% CI |",
        "|---|---|---:|---|---:|---:|---|"]
    for row in result["prompt_comparisons"]:
        b, w = row["metrics"]["sentence_bleu1_mean"], row["metrics"]["wer"]
        lines.append(f"| {NAMES[row['dataset']]} | {row['condition']} | {100*b['improvement']:+.2f} | [{100*b['ci_lower']:+.2f}, {100*b['ci_upper']:+.2f}] | {b['p_holm']:.6f} | {100*w['improvement']:+.2f} | [{100*w['ci_lower']:+.2f}, {100*w['ci_upper']:+.2f}] |")
    lines += ["", "## 固定的提示与输出检查", ""]
    for dataset, info in result["datasets"].items():
        prompt = info["selected_prompt"]
        lines += [f"### {NAMES[dataset]}：{prompt['prompt_id']}", "", "```text", prompt["prompt_template"].rstrip(), "```", "",
            "| 策略 | 输出词位 | 不同输出词 | 含数字词位 | 最常见的五个词 |", "|---|---:|---:|---:|---|"]
        for condition in study.CONDITIONS:
            d = info["output_diagnostics"]["selected_prompt"][condition]
            common = "; ".join(f"{word}: {n}" for word, n in d["most_common"][:5])
            lines.append(f"| {condition} | {d['total_words']} | {d['unique_words']} | {d['digit_words']} | {common} |")
    lines += ["", "## 新提示内的搜索差值", "",
        "以下为配对 95% CI，不另增显著性检验。", "",
        "| 数据集 | 策略 − LM_A | BLEU 增加（百分点） | 95% CI |", "|---|---|---:|---|"]
    for row in result["within_prompt_search"]:
        if row["metric"] == "sentence_bleu1_mean":
            lines.append(f"| {NAMES[row['dataset']]} | {row['condition']} | {100*row['improvement']:+.2f} | [{100*row['ci_lower']:+.2f}, {100*row['ci_upper']:+.2f}] |")
    lines += ["", "## 原神经策略的跨提示描述性参考", "",
        "神经策略保留原提示，因此下面差值不是同提示神经消融，不能解释为因果增量或神经贡献比例。", "",
        "| 数据集 | 原神经策略 | 新语言策略 | 神经 BLEU | 新语言 BLEU | 差值（百分点） |", "|---|---|---|---:|---:|---:|"]
    for row in result["cross_prompt_neural_descriptive"]:
        if row["metric"] == "sentence_bleu1_mean":
            lines.append(f"| {NAMES[row['dataset']]} | {row['neural_condition']} | {row['language_control']} | {100*row['neural_original_prompt']:.2f}% | {100*row['language_selected_prompt']:.2f}% | {100*row['difference']:+.2f} |")
    lines += ["", "这组量化固定提示敏感性；不能证明材料不在预训练语料中，也不能排除神经生成前缀触发预训练记忆。若采用新提示作为主实验提示，需同步重解码神经策略。", "",
        "[完整结果](results.json) · [逐片段指标](by_group.csv) · [逐受试者指标](by_subject.csv) · [来源审计](audit.json)"]
    (study.REPORT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
