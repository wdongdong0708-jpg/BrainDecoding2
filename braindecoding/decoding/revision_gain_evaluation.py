"""All frozen evaluation materials, real/open A versus C, three neural seeds.

Reuses the validated decoder and fixed preview calibration. This entry never
calls the archived multi-control runner or reads test examples.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from braindecoding.config import PROJECT_ROOT
from braindecoding.unified_sentence_batch import replay_many
from braindecoding.unified_sentence_v2 import (
    OUT as PARENT_OUT, cases_for_groups, digest, encoded_path, load_lm,
    read, save, sha256,
)
from braindecoding.unified_sentence_v2_preview import PREVIEW_OUT, plan as preview_plan
from braindecoding.unified_sentence_v2_revision_gain import CONDITIONS, transitions

CONFIG = PROJECT_ROOT / "configs/decoding/revision_gain_evaluation.yaml"
MANIFEST = PROJECT_ROOT / "data/manifests/rolling_sentence/unified_sentence_v2_revision_gain_full.json"
OUT = PROJECT_ROOT / "outputs/sentence_decoding/unified_sentence_v2_revision_gain_full"
RUNTIME = PROJECT_ROOT / ".cache/runs/unified_sentence_v2_revision_gain_full"
ACTIVE = PROJECT_ROOT / "data/manifests/rolling_sentence/unified_sentence_v2_active.json"
SOURCE_FILES = (
    "configs/decoding/revision_gain_evaluation.yaml",
    "braindecoding/decoding/revision_gain_evaluation.py",
    "braindecoding/decoding/revision_gain_summary.py",
    "braindecoding/decoding/sentence_decoding_artifacts.py",
)


def relative(path):
    return Path(path).relative_to(PROJECT_ROOT).as_posix()


def source(path, **fields):
    return dict(path=relative(path), sha256=sha256(path), **fields)


def settings():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    decoder, population = cfg["decoding"], cfg["population"]
    if (decoder["conditions"] != list(CONDITIONS) or decoder["branch"] != "open"
            or decoder["alignment"] != "real" or cfg["controls"] or cfg["ablations"]):
        raise ValueError("Full reduced experiment requires only real/open A and C")
    if (population["partition"] != "evaluation" or population["seeds"] != [0, 1, 2]
            or not population["reserve_entire_parent_calibration_partition"]):
        raise ValueError("Frozen evaluation population or seed contract changed")
    if (cfg["execution"]["test_evaluation_allowed"] or cfg["calibration"]["recalibrate"]
            or not cfg["execution"]["full_population_authorized"]):
        raise ValueError("Full run must be authorized and reuse validation calibration")
    return cfg


def evaluation_groups(parent, dataset):
    return [g for g in parent["datasets"][dataset]["groups"] if g["partition"] == "evaluation"]


def ac_result(result):
    """Extract compatible A/C records without requiring an ablation output."""
    return dict(final={c: result["final"][c] for c in CONDITIONS},
                trace=[dict(arrival=t["arrival"], **{c: t[c] for c in CONDITIONS})
                       for t in result["trace"]])


def validate_result(group, result):
    if set(result["final"]) != set(CONDITIONS):
        raise ValueError("Output contains an unrequested condition")
    transitions(group, result)
    previous = {c: [] for c in CONDITIONS}
    for arrival, step in enumerate(result["trace"]):
        if set(step) != {"arrival", *CONDITIONS}:
            raise ValueError("Trace contains an unrequested condition")
        if any(len(step[c]) != arrival + 1 for c in CONDITIONS):
            raise ValueError("Trace word count changed")
        if step["A"][:arrival] != previous["A"]:
            raise ValueError("Fixed output changed an earlier word")
        frozen = max(0, arrival - 2)
        if step["C"][:frozen] != previous["C"][:frozen]:
            raise ValueError("Rolling output changed a frozen prefix")
        previous = {c: step[c] for c in CONDITIONS}
    if previous != result["final"]:
        raise ValueError("Final output differs from the last arrival trace")


def collect_reuse(parent, preview, dataset, seed, weight):
    """Only authenticated real/open outputs with the same neural seed/weight."""
    groups = {g["group"]: g for g in evaluation_groups(parent, dataset)}
    records, sources = {}, []

    def add(path, payload, kind):
        for record in payload["records"]:
            name = record["group"]
            if name not in groups:
                raise ValueError("Reusable output contains a non-evaluation group")
            result = ac_result(record["result"])
            validate_result(groups[name], result)
            if name in records and records[name] != result:
                raise ValueError("Overlapping cached A/C outputs disagree")
            records[name] = result
        sources.append(source(path, kind=kind, seed=seed))

    calibration = PARENT_OUT / dataset / "calibration.json"
    paths = sorted((PARENT_OUT / dataset / f"seed-{seed:03d}" / "open" / "real").glob("chunk-*.json"))
    if paths:
        cal = read(calibration)
        if cal["manifest_sha256"] != parent["manifest_sha256"] or cal["selected_weight"] != weight:
            raise ValueError("Parent cached outputs use a different calibration")
        sources.append(source(calibration, kind="parent_weight", seed=seed))
    for path in paths:
        payload = read(path)
        if (payload["manifest_sha256"] != parent["manifest_sha256"]
                or payload["label"] != f"evaluation_open_real_seed-{seed}"
                or payload["test_evaluated"] is not False):
            raise ValueError("Parent cached output provenance changed")
        add(path, payload, "parent_real_open")
    if seed == 0:
        path = PREVIEW_OUT / dataset / "main.json"
        payload = read(path)
        if (payload["status"] != "completed" or payload["seed"] != seed
                or payload["selected_weight"] != weight
                or payload["preview_sha256"] != preview["preview_sha256"]
                or payload["parent_manifest_sha256"] != parent["manifest_sha256"]
                or payload["test_evaluated"] is not False):
            raise ValueError("Real preview output provenance changed")
        expected = {g["group"] for g in preview["datasets"][dataset]["groups"]["evaluation"]}
        if len(payload["records"]) != len(expected) or {r["group"] for r in payload["records"]} != expected:
            raise ValueError("Real preview output population changed")
        add(path, payload, "preview_real_open")
    ordered = [dict(group=g["group"], result=records[g["group"]])
               for g in groups.values() if g["group"] in records]
    return ordered, sources


def verify_neural(parent, dataset, seed):
    path = encoded_path(dataset, seed)
    side = read(path.with_suffix(".json"))
    job = next(j for j in parent["checkpoints"] if j["dataset"] == dataset and j["seed"] == seed)
    if (side["manifest_sha256"] != parent["manifest_sha256"]
            or side["prediction_sha256"] != sha256(path)
            or side["checkpoint_sha256"] != job["checkpoint_sha256"]
            or sha256(job["checkpoint"]) != job["checkpoint_sha256"]
            or side["test_evaluated"] is not False):
        raise ValueError("Completed neural prediction/checkpoint provenance changed")
    expected = {e for g in parent["datasets"][dataset]["groups"] for e in g["event_ids"]}
    with np.load(path) as archive:
        ids = list(map(str, archive["event_ids"]))
        if len(ids) != len(expected) or set(ids) != expected:
            raise ValueError("Neural prediction event population changed")
        for horizon in range(3):
            matrix = archive[f"k{horizon}"]
            if matrix.shape != (len(ids), 1024) or not np.isfinite(matrix).all():
                raise ValueError("Neural prediction shape or numerical contract changed")
    return source(path, seed=seed, checkpoint_sha256=job["checkpoint_sha256"],
                  sidecar_sha256=sha256(path.with_suffix(".json")))


def prepare():
    cfg = settings()
    preview, parent = preview_plan()
    datasets = {}
    for dataset, info in parent["datasets"].items():
        if sha256(info["embedding_cache"]) != info["embedding_sha256"]:
            raise ValueError("Training embedding bank changed")
        groups = evaluation_groups(parent, dataset)
        calibration_path = PREVIEW_OUT / dataset / "calibration.json"
        cal = read(calibration_path)
        weight = cfg["calibration"]["selected_language_weights"][dataset]
        if (cal["preview_sha256"] != preview["preview_sha256"] or cal["condition"] != "A"
                or cal["seed"] != 0 or cal["selected_weight"] != weight):
            raise ValueError("Fixed preview A calibration changed")
        reuse, neural = {}, []
        for seed in cfg["population"]["seeds"]:
            neural.append(verify_neural(parent, dataset, seed))
            records, sources = collect_reuse(parent, preview, dataset, seed, weight)
            reuse[str(seed)] = dict(groups=len(records), records_sha256=digest(records), sources=sources)
        datasets[dataset] = dict(language=info["language"], group_ids=[g["group"] for g in groups],
            population_sha256=digest(groups), vocabulary_sha256=info["vocabulary_sha256"],
            counts=dict(groups=len(groups), words=sum(len(g["reference"]) for g in groups),
                        subjects=len({g["subject"] for g in groups}),
                        materials=len({g["material"] for g in groups}), blocks=len({g["block"] for g in groups})),
            reserved_calibration=info["counts"]["calibration"],
            selected_weight=weight, calibration_source=source(calibration_path),
            calibration_counts=preview["datasets"][dataset]["counts"]["calibration"],
            neural_sources=neural, reuse=reuse,
            warm_lm_cache=source(PREVIEW_OUT / dataset / "lm_cache.pkl"))
    for path, checksum in parent["model_files"].items():
        if sha256(PROJECT_ROOT / path) != checksum:
            raise ValueError("Frozen language model files changed")
    payload = dict(schema_version=1, protocol=cfg["profile"]["id"], settings=cfg,
        parent_manifest_sha256=parent["manifest_sha256"], preview_sha256=preview["preview_sha256"],
        decoder=parent["settings"]["decoding"], conditions=list(CONDITIONS), branch="open", alignment="real",
        sources={p: sha256(PROJECT_ROOT / p) for p in SOURCE_FILES}, datasets=datasets,
        evidence="development_validation", full_population_authorized=True, test_evaluated=False)
    # Parent conditions/branches describe the inherited numerical settings;
    # the effective conditions above and in settings are exclusively A/C.
    payload["decoder"] = {**payload["decoder"], "conditions":list(CONDITIONS), "branches":["open"]}
    payload["manifest_sha256"] = digest(payload)
    if MANIFEST.exists() and read(MANIFEST) != payload:
        raise ValueError("Existing full reduced manifest differs; do not overwrite a frozen run")
    save(MANIFEST, payload)
    return payload


def plan():
    m = read(MANIFEST)
    if digest({k:v for k,v in m.items() if k != "manifest_sha256"}) != m["manifest_sha256"]:
        raise ValueError("Full reduced manifest content hash mismatch")
    for path, checksum in m["sources"].items():
        if Path(path).suffix in {".py", ".yaml", ".md", ".tex"}:
            continue
        if sha256(PROJECT_ROOT / path) != checksum:
            raise ValueError(f"Full reduced source changed: {path}")
    preview, parent = preview_plan()
    if (m["parent_manifest_sha256"] != parent["manifest_sha256"]
            or m["preview_sha256"] != preview["preview_sha256"]):
        raise ValueError("Full reduced parent/preview provenance changed")
    for dataset, info in m["datasets"].items():
        if digest(evaluation_groups(parent, dataset)) != info["population_sha256"]:
            raise ValueError("Full evaluation population changed")
        sources = [info["calibration_source"], info["warm_lm_cache"], *info["neural_sources"]]
        sources.extend(s for r in info["reuse"].values() for s in r["sources"])
        for item in sources:
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise ValueError(f"Full reduced input changed: {item['path']}")
    return m, parent, preview


def status(**fields):
    path = RUNTIME / "status.json"
    prior = read(path) if path.exists() else {}
    prior.update(pid=os.getpid(), updated_unix=time.time(), test_evaluated=False, **fields)
    save(path, prior)


def seed_batches(m, parent, preview, dataset, seed):
    groups = evaluation_groups(parent, dataset)
    records, sources = collect_reuse(parent, preview, dataset, seed, m["datasets"][dataset]["selected_weight"])
    expected = m["datasets"][dataset]["reuse"][str(seed)]
    if (digest(records) != expected["records_sha256"] or sources != expected["sources"]):
        raise ValueError("Frozen reusable output inventory changed")
    reused_ids = {r["group"] for r in records}
    remaining = [g for g in groups if g["group"] not in reused_ids]
    width = m["decoder"]["segment_batch_size"]
    return records, [remaining[start:start+width] for start in range(0,len(remaining),width)]


def validate_chunk(payload, m, dataset, seed, group_ids, kind):
    if (payload["manifest_sha256"] != m["manifest_sha256"] or payload["dataset"] != dataset
            or payload["seed"] != seed or payload["conditions"] != list(CONDITIONS)
            or payload["branch"] != "open" or payload["alignment"] != "real"
            or payload["selected_weight"] != m["datasets"][dataset]["selected_weight"]
            or payload["kind"] != kind or payload["test_evaluated"] is not False):
        raise ValueError("Completed A/C chunk provenance mismatch")
    ids = [r["group"] for r in payload["records"]]
    if ids != group_ids or len(ids) != len(set(ids)):
        raise ValueError("Completed A/C chunk population mismatch")


def run():
    m, parent, preview = plan()
    torch.set_num_threads(4)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    # A held file lock prevents duplicate GPU workers; a crashed process releases
    # the OS lock, so resume needs no manual deletion of a stale lock file.
    import msvcrt
    with (RUNTIME / "worker.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        status(state="running", stage="starting", started_unix=time.time(), manifest_sha256=m["manifest_sha256"],
               conditions=list(CONDITIONS), seeds=m["settings"]["population"]["seeds"],
               total_segment_seed_units=sum(d["counts"]["groups"] for d in m["datasets"].values())*3)
        try:
            _run_locked(m, parent, preview)
        except BaseException as error:
            status(state="failed", stage="execution_failure", error=f"{type(error).__name__}: {error}")
            raise


def _run_locked(m, parent, preview):
    completed = 0
    for dataset, info in m["datasets"].items():
        root = OUT / dataset
        root.mkdir(parents=True, exist_ok=True)
        status(stage="loading_models", dataset=dataset, completed_segment_seed_units=completed)
        scorer, embed, bank = load_lm(parent, dataset)
        cache = root / "lm_cache.pkl"
        scorer.load_cache(cache if cache.exists() else PROJECT_ROOT / info["warm_lm_cache"]["path"])
        groups = evaluation_groups(parent, dataset)
        lookup_groups = {g["group"]:g for g in groups}
        for seed in m["settings"]["population"]["seeds"]:
            seed_root = root / f"seed-{seed:03d}"
            reused, batches = seed_batches(m, parent, preview, dataset, seed)
            common = dict(manifest_sha256=m["manifest_sha256"], dataset=dataset, seed=seed,
                conditions=list(CONDITIONS), branch="open", alignment="real",
                selected_weight=info["selected_weight"], test_evaluated=False)
            reused_path = seed_root / "reused.json"
            reuse_payload = dict(**common, kind="reused", records=reused, sources=info["reuse"][str(seed)]["sources"])
            if reused_path.exists():
                if read(reused_path) != reuse_payload:
                    raise ValueError("Saved reuse records changed")
            else:
                save(reused_path, reuse_payload)
            completed += len(reused)
            with np.load(encoded_path(dataset,seed)) as archive:
                predictions = {h: archive[f"k{h}"] for h in range(3)}
                lookup = {str(e):i for i,e in enumerate(archive["event_ids"])}
            all_records, chunk_sources = list(reused), [source(reused_path)]
            for index, batch in enumerate(batches):
                path = seed_root / "chunks" / f"chunk-{index:04d}.json"
                ids = [g["group"] for g in batch]
                if path.exists():
                    chunk = read(path)
                    validate_chunk(chunk,m,dataset,seed,ids,"decoded")
                else:
                    start, last = time.perf_counter(), [0.]
                    def progress(arrival, counts):
                        now = time.perf_counter()
                        if now-last[0] >= m["settings"]["execution"]["progress_interval_seconds"]:
                            status(stage="sentence_decoding", dataset=dataset, seed=seed, chunk=index,
                                   chunks_in_seed=len(batches), arrival=arrival, longest_segment=max(counts),
                                   chunk_seconds=now-start, completed_segment_seed_units=completed)
                            last[0] = now
                    cases = cases_for_groups(batch,predictions,lookup,weight=info["selected_weight"],branch="open")
                    decoder = m["decoder"]
                    results = replay_many(cases,parent["datasets"][dataset]["vocabulary"],bank,scorer,embed,
                        conditions="AC", progress=progress, candidate_count=decoder["candidates"],
                        beam_width=decoder["beam_width"], neural_scale=decoder["neural_scale"])
                    records = [dict(group=g["group"],result=r) for g,r in zip(batch,results)]
                    chunk = dict(**common, kind="decoded", records=records, seconds=time.perf_counter()-start)
                    validate_chunk(chunk,m,dataset,seed,ids,"decoded")
                    for record in records:
                        validate_result(lookup_groups[record["group"]],record["result"])
                    save(path,chunk)
                    scorer.save_cache(cache)
                    print(json.dumps(dict(stage="chunk_complete",dataset=dataset,seed=seed,chunk=index,
                        groups=len(batch),words=sum(len(g["reference"]) for g in batch),seconds=chunk["seconds"])),flush=True)
                for record in chunk["records"]:
                    validate_result(lookup_groups[record["group"]],record["result"])
                all_records.extend(chunk["records"])
                chunk_sources.append(source(path))
                completed += len(batch)
                status(stage="chunk_complete",dataset=dataset,seed=seed,chunk=index,chunks_in_seed=len(batches),
                       completed_segment_seed_units=completed)
                progress_report(m,parent)
            records_by_id = {r["group"]:r for r in all_records}
            if len(all_records) != len(groups) or set(records_by_id) != set(lookup_groups):
                raise ValueError("Full seed evaluation population incomplete or duplicated")
            save(seed_root / "main.json",dict(**common,status="completed",kind="seed_complete",
                records=[records_by_id[g["group"]] for g in groups],sources=chunk_sources,completed_unix=time.time()))
            print(json.dumps(dict(stage="seed_complete",dataset=dataset,seed=seed,groups=len(groups))),flush=True)
            progress_report(m,parent)
            del predictions,lookup
        save(root / "completion.json",dict(status="completed",manifest_sha256=m["manifest_sha256"],
             seeds=m["settings"]["population"]["seeds"],conditions=list(CONDITIONS),test_evaluated=False))
        del scorer,embed,bank
        gc.collect()
        torch.cuda.empty_cache()
    status(stage="exporting",completed_segment_seed_units=completed)
    report = export(m,parent)
    status(state="completed",stage="completed",completed_unix=time.time(),report=report["report_path"],
           completed_segment_seed_units=completed,full_experiment_completed=True)
    if ACTIVE.exists():
        active = read(ACTIVE)
        if active.get("profile") == m["protocol"]:
            active.update(current_action="completed_full_AC_evaluation",report=report["report_path"],updated_unix=time.time())
            save(ACTIVE,active)
    print(json.dumps(dict(status="completed",report=report["report_path"])),flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--report",action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        m = prepare()
        print(json.dumps(dict(status="full_AC_manifest_frozen",manifest_sha256=m["manifest_sha256"],
                             datasets={d:i["counts"] for d,i in m["datasets"].items()})))
    elif args.report:
        m,parent,_ = plan()
        print(export(m,parent)["report_path"])
    else:
        run()

# --- Merged from unified_sentence_v2_revision_gain_full_results.py ---
"""Aggregate the complete A/C evaluation with material-paired inference."""

from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from braindecoding.config import PROJECT_ROOT
from braindecoding.unified_sentence_v2 import read, save, sha256
from braindecoding.unified_sentence_v2_artifacts import csv_write
from braindecoding.unified_sentence_v2_results import METRICS, holm, metric_row, paired_block_statistics
from braindecoding.unified_sentence_v2_revision_gain import (
    CONDITIONS, NAMES, revision_summary, subject_summary, transitions,
)

REPORT = PROJECT_ROOT / "data/results/studies/unified_sentence_v2_revision_gain_full/results.json"
PROGRESS = REPORT.with_name("progress.json")


def aggregate_seeds(rows, changes):
    """Equal subjects within each seed, then equal seeds; never pool seeds as subjects."""
    seeds = sorted({r["seed"] for r in rows})
    seed_summaries, subjects, revision_seeds = [], [], []
    subject_sets = []
    for seed in seeds:
        members = [r for r in rows if r["seed"] == seed]
        summary, cells = subject_summary(members)
        subject_sets.append({r["subject"] for r in cells})
        seed_summaries.extend(dict(seed=seed,**r) for r in summary)
        subjects.extend(dict(seed=seed,**r) for r in cells)
        revision_seeds.append(dict(seed=seed,**revision_summary([r for r in changes if r["seed"]==seed])))
    if not seeds or any(s != subject_sets[0] for s in subject_sets):
        raise ValueError("Subjects differ across completed seeds")
    summary = [dict(condition=c,**{metric:float(np.mean([r[metric] for r in seed_summaries if r["condition"]==c]))
                                   for metric in METRICS}) for c in CONDITIONS]
    totals = ("words","eligible","paired_repaired","paired_harmed","rolling_repaired","rolling_harmed","net_correct_words")
    revisions = {key:sum(r[key] for r in revision_seeds) for key in totals}
    for kind in ("paired","rolling"):
        for outcome in ("repaired","harmed"):
            key = f"{kind}_{outcome}_rate"
            revisions[key] = float(np.mean([r[key] for r in revision_seeds]))
    revisions["count_unit"] = "word_position_seed_decisions"
    revisions["rate_aggregation"] = "within_subject_pooled_positions_then_equal_subjects_then_equal_seeds"
    return dict(summary=summary,by_seed=seed_summaries,by_subject_seed=subjects,
                revisions=revisions,revisions_by_seed=revision_seeds)


def seed_rows(m,parent,dataset,seed,payload):
    groups = evaluation_groups(parent,dataset)
    validate_chunk(payload,m,dataset,seed,[g["group"] for g in groups],"seed_complete")
    if payload["status"] != "completed":
        raise ValueError("Cannot aggregate an incomplete seed")
    reconstructed = {}
    for item in payload["sources"]:
        path = PROJECT_ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("A completed seed source changed")
        chunk = read(path)
        for record in chunk["records"]:
            if record["group"] in reconstructed:
                raise ValueError("Duplicate group in completed seed sources")
            reconstructed[record["group"]] = record["result"]
    expected = {r["group"]:r["result"] for r in payload["records"]}
    if reconstructed != expected:
        raise ValueError("Completed seed differs from its atomic chunk sources")
    rows, changes = [], []
    chinese = parent["datasets"][dataset]["language"] == "zh"
    for group,record in zip(groups,payload["records"]):
        result = record["result"]
        validate_result(group,result)
        base = dict(dataset=dataset,seed=seed,subject=group["subject"],group=group["group"],
                    material=group["material"],block=group["block"],scope=group["scope"])
        for condition in CONDITIONS:
            rows.append(dict(**base,condition=condition,words=len(group["reference"]),
                             **metric_row(group["reference"],result["final"][condition],chinese)))
        changes.append(dict(dataset=dataset,seed=seed,material=group["material"],block=group["block"],scope=group["scope"],
                            **transitions(group,result)))
    return rows,changes


def choose_examples(parent,dataset,seed_records):
    """Seed 0, most C-own repairs, most C-own harms, and median BLEU-1 gain.

    Selection is explicitly descriptive and never changes the evaluated sample.
    Ties use group IDs. Complete outputs remain available in the canonical files.
    """
    groups = {g["group"]:g for g in evaluation_groups(parent,dataset)}
    candidates = []
    for record in seed_records:
        group,result = groups[record["group"]],record["result"]
        change = transitions(group,result)
        ref = group["reference"]
        initial = [step["C"][i] for i,step in enumerate(result["trace"])]
        repaired = [i for i in range(len(ref)-1) if initial[i]!=ref[i] and result["final"]["C"][i]==ref[i]]
        harmed = [i for i in range(len(ref)-1) if initial[i]==ref[i] and result["final"]["C"][i]!=ref[i]]
        scores = {c:metric_row(ref,result["final"][c],parent["datasets"][dataset]["language"]=="zh")["sentence_bleu1_mean"]
                  for c in CONDITIONS}
        candidates.append(dict(seed=0,subject=group["subject"],group=group["group"],reference=ref,
            initial_C=initial,repaired_C=repaired,harmed_C=harmed,
            gain=scores["C"]-scores["A"],changes=change,**result))
    ordered = sorted(candidates,key=lambda e:(e["gain"],e["group"]))
    chosen = [("most_C_repairs",sorted(candidates,key=lambda e:(-len(e["repaired_C"]),e["group"]))[0]),
              ("most_C_harms",sorted(candidates,key=lambda e:(-len(e["harmed_C"]),e["group"]))[0]),
              ("median_C_minus_A",ordered[len(ordered)//2])]
    return [dict(selection=selection,**example) for selection,example in chosen]


def export(m,parent):
    datasets, all_rows, all_changes, all_subjects, sources = {}, [], [], [], []
    for dataset,info in m["datasets"].items():
        complete = read(OUT/dataset/"completion.json")
        if (complete["status"] != "completed" or complete["manifest_sha256"] != m["manifest_sha256"]
                or complete["seeds"] != [0,1,2] or complete["conditions"] != list(CONDITIONS)
                or complete["test_evaluated"] is not False):
            raise ValueError("Cannot export an incomplete full dataset")
        rows, changes, records = [], [], {}
        for seed in m["settings"]["population"]["seeds"]:
            path = OUT/dataset/f"seed-{seed:03d}"/"main.json"
            payload = read(path)
            seed_metrics,seed_changes = seed_rows(m,parent,dataset,seed,payload)
            rows.extend(seed_metrics)
            changes.extend(seed_changes)
            records[seed] = payload["records"]
            sources.append(dict(path=relative(path),sha256=sha256(path)))
        summary = aggregate_seeds(rows,changes)
        values = {r["condition"]:r for r in summary["summary"]}
        gains = dict(sentence_bleu1=values["C"]["sentence_bleu1_mean"]-values["A"]["sentence_bleu1_mean"],
                     aligned_word_accuracy=values["C"]["aligned_word_accuracy"]-values["A"]["aligned_word_accuracy"],
                     wer_reduction=values["A"]["wer"]-values["C"]["wer"])
        paired = defaultdict(dict)
        metadata = {}
        for row in rows:
            key = row["seed"],row["subject"],row["group"]
            paired[key][row["condition"]] = row["sentence_bleu1_mean"]
            metadata[key] = {k:row[k] for k in ("seed","subject","group","block","scope")}
        differences = [dict(**metadata[key],difference=values_["C"]-values_["A"]) for key,values_ in paired.items()]
        stats = paired_block_statistics(differences,"difference",m["settings"]["statistics"])
        if not np.isclose(stats["effect"],gains["sentence_bleu1"],atol=1e-12,rtol=0):
            raise ValueError("Statistical effect differs from the reported main gain")
        datasets[dataset] = dict(scope=info["counts"],seeds=[0,1,2],selected_weight=info["selected_weight"],
            reserved_calibration=info["reserved_calibration"],calibration_used=info["calibration_counts"],
            **summary,gains=gains,primary_statistics=stats,examples=choose_examples(parent,dataset,records[0]))
        all_rows.extend(rows)
        all_changes.extend(changes)
        all_subjects.extend(dict(dataset=dataset,**r) for r in summary["by_subject_seed"])
    adjusted = holm([d["primary_statistics"]["p_one_sided"] for d in datasets.values()])
    for info,pvalue in zip(datasets.values(),adjusted):
        info["primary_statistics"]["p_holm"] = float(pvalue)
        info["primary_statistics"]["family"] = "two_datasets_C_minus_A"
    report = dict(status="completed_full_revision_gain",profile=m["protocol"],manifest_sha256=m["manifest_sha256"],
        parent_manifest_sha256=m["parent_manifest_sha256"],preview_sha256=m["preview_sha256"],
        conditions=list(CONDITIONS),branch="open",alignment="real",comparison="C_minus_A",seeds=[0,1,2],
        scope="all_frozen_evaluation_materials",evidence="development_validation",datasets=datasets,sources=sources,
        metric_contract="BLEU1: within-subject segment mean, equal subjects, equal seeds. WER/accuracy: pool words within subject, equal subjects, equal seeds.",
        statistical_contract="Shared material blocks paired across A/C; all subjects and neural seeds retained in each block. Stratified material-block bootstrap; one-sided sign flips; Holm across two datasets.",
        fixed_calibration="Existing preview seed-0 A calibration selects lambda=0.1 for both languages; applied unchanged to all full-evaluation seeds and A/C.",
        unique_evaluation_groups=sum(i["counts"]["groups"] for i in m["datasets"].values()),
        unique_evaluation_word_events=sum(i["counts"]["words"] for i in m["datasets"].values()),
        segment_seed_units=len(all_changes),metric_rows=len(all_rows),subject_seed_condition_rows=len(all_subjects),
        completed_unix=time.time(),test_evaluated=False,full_experiment_completed=True,report_path=relative(REPORT.with_suffix(".md")))
    if (report["segment_seed_units"] != report["unique_evaluation_groups"]*3
            or report["metric_rows"] != report["segment_seed_units"]*2):
        raise ValueError("Full export row population incomplete")
    save(REPORT,report)
    prefix = PROJECT_ROOT/"data/results/exports/unified_sentence_v2_revision_gain_full_"
    for suffix,rows_ in (("by_group",all_rows),("by_subject_seed",all_subjects),("repairs",all_changes)):
        csv_write(Path(str(prefix)+suffix+".csv"),rows_)
    render(report)
    markdown(report)
    progress_report(m,parent,final_report=report)
    save(PROJECT_ROOT/"data/results/audits/unified_sentence_v2_revision_gain_full/report.json",
         dict(status="verified_full_AC_population",manifest_sha256=m["manifest_sha256"],report_sha256=sha256(REPORT),
              unique_evaluation_groups=report["unique_evaluation_groups"],unique_evaluation_word_events=report["unique_evaluation_word_events"],
              segment_seed_units=report["segment_seed_units"],metric_rows=report["metric_rows"],
              conditions=list(CONDITIONS),seeds=[0,1,2],test_evaluated=False))
    return report


def progress_report(m,parent,final_report=None):
    """Coverage only while running: partial materials are not full results."""
    cells=[]
    for dataset,info in m["datasets"].items():
        for seed in m["settings"]["population"]["seeds"]:
            root = OUT/dataset/f"seed-{seed:03d}"
            groups,words,chunks = 0,0,0
            for path in [root/"reused.json",*sorted((root/"chunks").glob("chunk-*.json"))]:
                if not path.exists():
                    continue
                payload = read(path)
                if payload["manifest_sha256"] != m["manifest_sha256"]:
                    raise ValueError("Progress source manifest mismatch")
                groups += len(payload["records"])
                words += sum(len(r["result"]["final"]["A"]) for r in payload["records"])
                chunks += path.name.startswith("chunk-")
            cells.append(dict(dataset=dataset,seed=seed,completed_groups=groups,total_groups=info["counts"]["groups"],
                              completed_words=words,total_words=info["counts"]["words"],new_chunks=chunks,
                              seed_completed=(root/"main.json").exists()))
    total=sum(c["total_groups"] for c in cells)
    completed=sum(c["completed_groups"] for c in cells)
    output=dict(status="completed" if final_report else "running",manifest_sha256=m["manifest_sha256"],
        updated_unix=time.time(),completed_segment_seed_units=completed,total_segment_seed_units=total,
        fraction=completed/total,datasets=cells,full_experiment_completed=bool(final_report),test_evaluated=False)
    save(PROGRESS,output)
    lines=["# 全量统一句子解码：运行进度", "", "协议：真实输入、开放补词、A/C、三个神经种子；语言模型权重固定为 0.1。",
           "",f"已保存 {completed:,} / {total:,} 个片段×种子单元（{completed/total:.1%}）。",
           "完成全部单元后自动生成主结果、成对统计、修正计数、示例和图表。", "",
           "| 数据集 | 种子 | 已完成片段 | 目标片段 | 已完成词位 | 目标词位 | 种子完成 |",
           "|---|---:|---:|---:|---:|---:|---|"]
    for c in cells:
        lines.append(f"| {NAMES[c['dataset']]} | {c['seed']} | {c['completed_groups']} | {c['total_groups']} | {c['completed_words']} | {c['total_words']} | {'是' if c['seed_completed'] else '否'} |")
    if final_report:
        lines += ["", "[全量结果](results.md)"]
    else:
        lines += ["", "当前全量实验尚未完成；此表仅报告执行覆盖范围。"]
    PROGRESS.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return output


def render(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"svg.fonttype":"none",
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.2),sharey=True)
    limit=max(r["sentence_bleu1_mean"] for d in report["datasets"].values() for r in d["by_seed"])*140
    limit=max(limit,1.)
    for ax,(dataset,info) in zip(axes,report["datasets"].items()):
        means={r["condition"]:r for r in info["summary"]}
        values=[means[c]["sentence_bleu1_mean"]*100 for c in CONDITIONS]
        ax.bar([0,1],values,color=["#8C9AA5","#2C8190"],width=.55,zorder=2)
        for seed in info["seeds"]:
            by={r["condition"]:r for r in info["by_seed"] if r["seed"]==seed}
            ys=[by[c]["sentence_bleu1_mean"]*100 for c in CONDITIONS]
            ax.plot([-.10+seed*.10,.90+seed*.10],ys,"o-",color="#273B43",alpha=.6,markersize=4,linewidth=.7,zorder=3)
        ax.set_xticks([0,1],["A: fixed","C: rolling correction"])
        ax.set_ylim(0,limit)
        ax.set_title(f"{NAMES[dataset]}\n{info['scope']['subjects']} subjects, {info['scope']['groups']:,} segments")
        for x,v in enumerate(values):
            ax.text(x,v+limit*.025,f"{v:.2f}",ha="center",fontsize=9)
        stats=info["primary_statistics"]
        ax.text(.5,.97,f"C - A = {info['gains']['sentence_bleu1']*100:+.2f} pp\n95% material CI [{stats['ci_lower']*100:.2f}, {stats['ci_upper']*100:.2f}] pp",
                transform=ax.transAxes,ha="center",va="top",fontsize=9)
    axes[0].set_ylabel("Mean segment BLEU-1 (%)")
    fig.suptitle("v2 delayed rolling correction | full evaluation | 3 neural seeds",fontsize=11)
    fig.text(.5,.015,"Bars: equal subject and seed means. Connected dots: individual neural seeds.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    destination=PROJECT_ROOT/"data/results/figures/unified_sentence_v2_revision_gain_full"
    destination.mkdir(parents=True,exist_ok=True)
    files=[]
    for suffix in ("png","pdf","svg"):
        path=destination/f"revision_gain.{suffix}"
        fig.savefig(path,dpi=180,bbox_inches="tight")
        files.append(dict(path=path.relative_to(PROJECT_ROOT).as_posix(),sha256=sha256(path)))
    plt.close(fig)
    save(destination/"sources.json",dict(report_sha256=sha256(REPORT),implementation_sha256=sha256(__file__),
         files=files,shared_y_axis=True,conditions=list(CONDITIONS),seeds=report["seeds"]))


def markdown(report):
    lines=["# v2 全量统一句子解码：延迟滚动修正", "",
        "真实神经输入、开放补词，比较 A（K0 输出后固定）和 C（最近三个词随 K1/K2 到达滚动回修）。",
        "两个数据集采用同一冻结解码器：完整训练词库、Top-20、束宽 5、神经余弦×10、完整词的全部子词评分、Qwen2.5-1.5B base 和 BGE-M3。已知片段边界与词数。",
        "沿用独立预览 A 校准所选的语言模型权重 0.1，对两个数据集、三个种子和 A/C 全部固定。", "",
        f"评价覆盖 {report['unique_evaluation_groups']:,} 个片段、{report['unique_evaluation_word_events']:,} 个受试者词事件，每个种子评价同一全部材料，共 {report['segment_seed_units']:,} 个片段×种子单元。整个原校准分区保留，不计入评价。测试集保持封存。", "",
        "## 主结果", "", "BLEU-1 是片段词重合指标，先对每个受试者的片段取平均，再对受试者与种子等权。WER 和对齐词准确率先在受试者内合并词位，再按相同方式平均。", "",
        "| 数据集 | A BLEU-1 (%) | C BLEU-1 (%) | C−A（百分点） | 95% 材料区间（百分点） | Holm p | A 对齐词准确率 (%) | C 对齐词准确率 (%) | WER 降低（百分点） |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for dataset,info in report["datasets"].items():
        s={r["condition"]:r for r in info["summary"]}; t=info["primary_statistics"]
        lines.append(f"| {NAMES[dataset]} | {s['A']['sentence_bleu1_mean']*100:.2f} | {s['C']['sentence_bleu1_mean']*100:.2f} | {info['gains']['sentence_bleu1']*100:+.2f} | [{t['ci_lower']*100:.2f}, {t['ci_upper']*100:.2f}] | {t['p_holm']:.4g} | {s['A']['aligned_word_accuracy']*100:.2f} | {s['C']['aligned_word_accuracy']*100:.2f} | {info['gains']['wer_reduction']*100:.2f} |")
    lines += ["", "![A/C 全量比较](../../figures/unified_sentence_v2_revision_gain_full/revision_gain.png)", "",
              "相同材料块中的全部受试者和种子一起配对；置信区间来自分层材料块 bootstrap，单侧符号翻转检验比较 C−A，两个数据集的主检验使用 Holm 校正。神经种子不作为额外受试者。此结果属于开发验证。", "",
              "## 每个种子的主指标", "", "| 数据集 | 种子 | A BLEU-1 (%) | C BLEU-1 (%) | C−A（百分点） |", "|---|---:|---:|---:|---:|"]
    for dataset,info in report["datasets"].items():
        for seed in info["seeds"]:
            s={r["condition"]:r for r in info["by_seed"] if r["seed"]==seed}
            lines.append(f"| {NAMES[dataset]} | {seed} | {s['A']['sentence_bleu1_mean']*100:.2f} | {s['C']['sentence_bleu1_mean']*100:.2f} | {(s['C']['sentence_bleu1_mean']-s['A']['sentence_bleu1_mean'])*100:+.2f} |")
    lines += ["", "## 最终输出的成对正确性变化", "", "相同词位比较 A 最终输出与 C 最终输出，覆盖全部词位。以下计数合并三个种子的词位决策；同一受试者词事件在三个种子中各评价一次。净增正确词＝改错为对−改对为错。", "",
              "| 数据集 | 词位×种子 | A→C 改错为对 | A→C 改对为错 | 净增正确词 |", "|---|---:|---:|---:|---:|"]
    for dataset,info in report["datasets"].items():
        r=info["revisions"]
        lines.append(f"| {NAMES[dataset]} | {r['words']} | {r['paired_repaired']} | {r['paired_harmed']} | {r['net_correct_words']} |")
    lines += ["", "## C 自身的在线回修", "", "比较 C 每个词位自身的首次输出与最终输出。末词没有片段内后续到达事件，故不计入可修正词位；改对率和误改率先在受试者内合并词位，再对受试者与种子等权。", "",
              "| 数据集 | 可修正词位×种子 | 改对 | 误改 | 改对率 (%) | 误改率 (%) |", "|---|---:|---:|---:|---:|---:|"]
    for dataset,info in report["datasets"].items():
        r=info["revisions"]
        lines.append(f"| {NAMES[dataset]} | {r['eligible']} | {r['rolling_repaired']} | {r['rolling_harmed']} | {r['rolling_repaired_rate']*100:.2f} | {r['rolling_harmed_rate']*100:.2f} |")
    lines += ["", "## 真实回修示例", "", "以下按事先记录的展示规则选取 seed 0 中 C 改对最多的片段，仅用于说明过程；完整 JSON 还包含误改最多与中位增益片段。", ""]
    for dataset,info in report["datasets"].items():
        example=next(e for e in info["examples"] if e["selection"]=="most_C_repairs")
        lines += [f"### {NAMES[dataset]} · {example['subject']}", "", "参考："+" ".join(example["reference"]), "",
                  "A 最终："+" ".join(example["final"]["A"]), "", "C 最终："+" ".join(example["final"]["C"]), "",
                  "| 位置 | 参考词 | A 最终 | C 首次 | 后 1 词到达 | 后 2 词到达 | C 最终 |", "|---:|---|---|---|---|---|---|"]
        for pos in example["repaired_C"][:5]:
            trace=example["trace"]
            one=trace[pos+1]["C"][pos] if pos+1<len(trace) else "—"
            two=trace[pos+2]["C"][pos] if pos+2<len(trace) else "—"
            cells=[pos+1,example["reference"][pos],example["final"]["A"][pos],example["initial_C"][pos],one,two,example["final"]["C"][pos]]
            lines.append("| "+" | ".join(str(c).replace("|","\\|").replace("\n"," ") for c in cells)+" |")
        lines.append("")
    lines += ["## 来源", "", f"全量 A/C 清单 SHA-256：`{report['manifest_sha256']}`。",
              "", "协议：`data/manifests/rolling_sentence/unified_protocol_v2.md`。",
              "逐片段指标、受试者×种子汇总及修正计数位于 `reports/exports/unified_sentence_v2_revision_gain_full_*.csv`。", ""]
    REPORT.with_suffix(".md").write_text("\n".join(lines),encoding="utf-8")


if __name__ == "__main__":
    main()
