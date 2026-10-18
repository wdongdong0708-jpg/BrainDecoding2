"""Add only fixed-K0 rolling search B to the completed held-out A/C experiment."""
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

from braindecoding.config import PROJECT_ROOT as ROOT
from braindecoding.training.runtime import set_seed
from braindecoding.unified_sentence_batch import replay_many
from braindecoding.unified_sentence_v2 import digest, load_lm, read, save, sha256
from braindecoding.unified_heldout_test_v2 import validate_chunk as validate_ac_chunk


ID = "fixed_k0_rolling_search_test_v2"
CONFIG = ROOT / "configs/evaluation/candidate_pool_search_evaluation.yaml"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
PARENT = ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"
ORIGINAL_SUMMARY = ROOT / "data/results/exports/unified_heldout_test_v2_sentence_summary.json"
PARENT_OUT = ROOT / "outputs/heldout_test/unified_heldout_test_v2"
OUT = ROOT / f"outputs/heldout_test/{ID}"
REPORT = ROOT / f"data/results/studies/{ID}"
RUNTIME = ROOT / f".cache/runs/{ID}"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_key(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def verify_identity(value):
    require(digest({k: v for k, v in value.items() if k != "manifest_sha256"})
            == value["manifest_sha256"], "Frozen manifest identity changed")


def verify_hashes(sources):
    for path, expected in sources.items():
        if str(path).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        require(sha256(ROOT / path) == expected, f"Frozen source changed: {path}")


def k0_cases(groups, k0, event_ids, *, weight=.1):
    """No horizon other than zero, or reference text, can enter B replay."""
    event_ids = list(map(str, event_ids))
    queries = np.asarray(k0)
    require(queries.ndim == 2 and len(queries) == len(event_ids), "K0 shape differs from events")
    require(len(set(event_ids)) == len(event_ids), "Duplicate neural events")
    require(np.isfinite(queries).all(), "Non-finite K0 predictions")
    lookup = {e: i for i, e in enumerate(event_ids)}
    cases = []
    for group in groups:
        require(group["event_ids"] and all(e in lookup for e in group["event_ids"]),
                "B group is empty or absent from K0 cache")
        ids = [lookup[e] for e in group["event_ids"]]
        cases.append(dict(predictions={0: queries[ids]}, weight=weight, open_vocabulary=True))
    return cases


def load_k0(path):
    # In particular, do not construct the usual {0, 1, 2} prediction dictionary.
    with np.load(path, allow_pickle=False) as archive:
        return archive["k0"], archive["event_ids"]


def validate_b_result(group, result):
    count = len(group["event_ids"])
    require(count > 0 and set(result) == {"final", "trace"}, "B output schema differs")
    require(set(result["final"]) == {"B"} and len(result["final"]["B"]) == count,
            "B final condition or known word count changed")
    require(len(result["trace"]) == count, "B arrival population changed")
    previous = []
    for arrival, step in enumerate(result["trace"]):
        require(set(step) == {"arrival", "B"} and step["arrival"] == arrival,
                "B trace conditions or arrival order changed")
        words = step["B"]
        require(len(words) == arrival + 1 and all(isinstance(w, str) and w for w in words),
                "B trace word count or words changed")
        start = max(0, arrival - 2)
        require(words[:start] == previous[:start], "B changed a frozen prefix")
        previous = words
    require(previous == result["final"]["B"], "B final output differs from last arrival")


def validate_b_chunk(chunk, m, job, groups):
    expected = dict(manifest_sha256=m["manifest_sha256"], dataset=job["dataset"], seed=job["seed"],
                    conditions=["B"], split="test", test_evaluated=True,
                    neural_representation="K0_only", branch="open", alignment="real", selected_weight=.1)
    require(all(chunk.get(k) == v for k, v in expected.items()), "B chunk provenance mismatch")
    require([r["group"] for r in chunk["records"]] == [g["group"] for g in groups],
            "B chunk population is incomplete, duplicated or reordered")
    for group, record in zip(groups, chunk["records"]):
        validate_b_result(group, record["result"])


def validate_initial_agreement(b_records, ac_records):
    require(len(b_records) == len(ac_records), "B/A initial comparison population differs")
    for b, ac in zip(b_records, ac_records):
        require(b["group"] == ac["group"]
                and b["result"]["trace"][0]["B"] == ac["result"]["trace"][0]["A"],
                "B and A disagree before any rolling search: check decoder equivalence")


def original_ac(parent, dataset, seed, path):
    info, payload = parent["datasets"][dataset], read(path)
    require(payload["status"] == "completed", "Original A/C is incomplete")
    validate_ac_chunk(payload, parent, dataset, seed, info["groups"])
    reconstructed = []
    for source in payload["sources"]:
        chunk_path = ROOT / source["path"]
        require(sha256(chunk_path) == source["sha256"], "Original A/C chunk changed")
        chunk = read(chunk_path)
        start = len(reconstructed)
        validate_ac_chunk(chunk, parent, dataset, seed,
                          info["groups"][start:start + len(chunk["records"])])
        reconstructed.extend(chunk["records"])
    require(reconstructed == payload["records"], "Original A/C atomic chunks differ from saved outputs")
    return payload


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        verify_identity(m)
        verify_hashes(m["source_sha256"])
        return m
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(cfg["protocol"]["split"] == "test" and cfg["protocol"]["seeds"] == [0, 1, 2],
            "B supplement requires the entire three-seed test population")
    require(not cfg["protocol"]["training_allowed"] and not cfg["protocol"]["recalibration_allowed"],
            "B must not train or recalibrate")
    require(cfg["decoding"]["compute_conditions"] == ["B"]
            and cfg["decoding"]["report_conditions"] == ["A", "B", "C"]
            and cfg["decoding"]["language_weight"] == .1
            and cfg["decoding"]["revision_words"] == 3
            and cfg["decoding"]["read_K1_K2"] is False
            and cfg["decoding"]["branch"] == "open" and cfg["decoding"]["alignment"] == "real",
            "Fixed-K0 rolling B decoder contract changed")
    require(cfg["protocol"]["parent"] == source_key(PARENT), "B parent changed")
    parent, summary = read(PARENT), read(ORIGINAL_SUMMARY)
    verify_identity(parent)
    require(summary["status"] == "completed_heldout_test"
            and summary["manifest_sha256"] == parent["manifest_sha256"], "Original A/C summary changed")
    require(set(parent["datasets"]) == set(cfg["protocol"]["datasets"]), "B datasets differ from original A/C")
    require(parent["decoder"]["revision_words"] == 3, "Parent rolling window changed")
    require(cfg["execution"]["segment_batch_size"] == parent["settings"]["execution"]["segment_batch_size"],
            "B batch layout must match original A/C")
    sources = {source_key(path): expected for path, expected in parent["source_sha256"].items()}
    for path in (CONFIG, PROTOCOL, PARENT, ORIGINAL_SUMMARY, Path(__file__),
                 ROOT / f"braindecoding/evaluation/{ID}_results.py", ROOT / "braindecoding/decoding/jayalath_sentence.py",
                 ROOT / "braindecoding/decoding/unified_sentence_v2_artifacts.py"):
        sources[source_key(path)] = sha256(path)
    # Bind the local text model used to score any newly proposed outside word.
    from huggingface_hub import snapshot_download
    text_cfg = parent["neural_settings"]["text_embedding"]
    snapshot = Path(snapshot_download(text_cfg["model_name"], revision=text_cfg["revision"], local_files_only=True))
    for path in snapshot.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".txt", ".model", ".safetensors", ".bin"):
            sources[source_key(path)] = sha256(path)
    jobs = []
    for dataset in cfg["protocol"]["datasets"]:
        info = parent["datasets"][dataset]
        expected = [e for g in info["groups"] for e in g["event_ids"]]
        require(len(expected) == len(set(expected)) == info["counts"]["events"], "Frozen event population changed")
        require(len({g["group"] for g in info["groups"]}) == len(info["groups"])
                and all(len(g["reference"]) == len(g["event_ids"]) for g in info["groups"]),
                "Frozen groups are duplicated or reference alignment changed")
        require(len(info["groups"]) == info["counts"]["groups"] and info["selected_weight"] == .1,
                "Frozen group population or language weight changed")
        cache = PARENT_OUT / dataset / "lm_cache.pkl"
        if cache.exists():
            sources[source_key(cache)] = sha256(cache)
        for seed in cfg["protocol"]["seeds"]:
            root = PARENT_OUT / dataset / f"seed-{seed:03d}"
            neural_path, side_path, ac_path = root / "neural_predictions.npz", root / "word_metrics.json", root / "main.json"
            side = read(side_path)
            checkpoint = next(j for j in parent["checkpoints"] if (j["dataset"], j["seed"]) == (dataset, seed))
            require(side["dataset"] == dataset and side["seed"] == seed and side["split"] == "test"
                    and side["test_evaluated"] and side["manifest_sha256"] == parent["manifest_sha256"]
                    and side["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
                    and side["prediction_sha256"] == sha256(neural_path), "K0 neural cache provenance changed")
            k0, ids = load_k0(neural_path)
            require(len(ids) == len(expected) and set(map(str, ids)) == set(expected), "K0 events differ from A/C")
            require(k0.shape == (len(expected), text_cfg["embedding_dimension"])
                    and np.isfinite(k0).all() and np.allclose(np.linalg.norm(k0, axis=1), 1., atol=.005),
                    "K0 vectors are not finite normalized frozen embeddings")
            ac = original_ac(parent, dataset, seed, ac_path)
            for path in (neural_path, side_path, ac_path):
                if Path(path).is_file():
                    sources[source_key(path)] = sha256(path)
            for item in ac["sources"]:
                sources[item["path"]] = item["sha256"]
            jobs.append(dict(dataset=dataset, seed=seed, prediction=source_key(neural_path),
                             prediction_sha256=side["prediction_sha256"], original_ac=source_key(ac_path),
                             parent_cache=source_key(cache) if cache.exists() else None))
            del k0, ids, ac
    verify_hashes(sources)
    m = dict(protocol=ID, frozen_unix=time.time(), settings=cfg, decoder=copy.deepcopy(parent["decoder"]),
             parent_manifest_sha256=parent["manifest_sha256"], jobs=jobs, source_sha256=sources,
             total_segment_seed_units=sum(i["counts"]["groups"] for i in parent["datasets"].values()) * 3,
             neural_input="K0_only", evidence=cfg["statistics"]["evidence"])
    m["manifest_sha256"] = digest(m)
    save(MANIFEST, m)
    print(json.dumps(dict(stage="frozen", jobs=len(jobs), total=m["total_segment_seed_units"],
                          manifest_sha256=m["manifest_sha256"])), flush=True)
    return m


def status(m, **fields):
    path = RUNTIME / "status.json"
    previous = read(path) if path.exists() else {}
    previous.update(pid=os.getpid(), updated_unix=time.time(), manifest_sha256=m["manifest_sha256"],
                    total_segment_seed_units=m["total_segment_seed_units"], **fields)
    save(path, previous)
    completed = previous.get("completed_segment_seed_units", 0)
    lines = ["# 固定 K0 滚动文本重搜 B：补充实验进度", "",
             f"状态：{previous.get('state', 'running')}；已保存 {completed:,} / {m['total_segment_seed_units']:,} 个片段×种子。", "",
             f"当前：{previous.get('dataset', '')}，种子 {previous.get('seed', '')}，分块 {previous.get('chunk', '')}，到达词位 {previous.get('arrival', '')}。", "",
             "B 始终使用 K0，每次重新搜索最近三个词。全部完成后自动汇总 A→B 与 B→C 的增益及材料块配对区间。",
             "当前只报告覆盖进度，完整结果写入同目录 results.md。", "",
             f"更新时间（Unix）：{previous['updated_unix']:.0f}。"]
    if previous.get("error"):
        lines += ["", f"错误：{previous['error']}"]
    if previous.get("state") == "completed":
        lines += ["", "[完整结果](results.md)"]
    REPORT.mkdir(parents=True, exist_ok=True)
    temporary = REPORT / "progress.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(REPORT / "progress.md")
    print(json.dumps({k: previous[k] for k in ("state", "stage", "dataset", "seed", "chunk", "arrival",
                                             "completed_segment_seed_units") if k in previous}), flush=True)


def run():
    m = freeze()
    parent = read(PARENT)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (RUNTIME / "worker.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0"); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        torch.set_num_threads(4)
        set_seed(0)
        status(m, state="running", stage="starting", started_unix=time.time(), error=None,
               completed_segment_seed_units=0)
        try:
            completed = 0
            for dataset in m["settings"]["protocol"]["datasets"]:
                info = parent["datasets"][dataset]
                scorer = embed = bank = None
                cache_path = OUT / dataset / "lm_cache.pkl"
                for job in (j for j in m["jobs"] if j["dataset"] == dataset):
                    k0, ids = load_k0(ROOT / job["prediction"])
                    ac = read(ROOT / job["original_ac"])
                    root = OUT / dataset / f"seed-{job['seed']:03d}"
                    records, sources = [], []
                    width = m["settings"]["execution"]["segment_batch_size"]
                    groups = info["groups"]
                    for start in range(0, len(groups), width):
                        batch, number = groups[start:start + width], start // width
                        path = root / "chunks" / f"chunk-{number:04d}.json"
                        if path.exists():
                            chunk = read(path)
                            validate_b_chunk(chunk, m, job, batch)
                        else:
                            if scorer is None:
                                status(m, stage="loading_sentence_models", dataset=dataset, seed=job["seed"], chunk=number)
                                scorer, embed, bank = load_lm(parent, dataset)
                                if cache_path.exists():
                                    scorer.load_cache(cache_path)
                                elif job["parent_cache"]:
                                    scorer.load_cache(ROOT / job["parent_cache"])
                            started, last = time.perf_counter(), [0.]
                            def progress(arrival, counts):
                                now = time.perf_counter()
                                if now - last[0] >= m["settings"]["execution"]["progress_interval_seconds"]:
                                    status(m, stage="B_decoding", dataset=dataset, seed=job["seed"], chunk=number,
                                           arrival=arrival, longest_segment=max(counts), chunk_seconds=now-started,
                                           completed_segment_seed_units=completed)
                                    last[0] = now
                            cases = k0_cases(batch, k0, ids)
                            d = m["decoder"]
                            results = replay_many(cases, info["vocabulary"], bank, scorer, embed,
                                                  conditions="B", progress=progress, candidate_count=d["candidates"],
                                                  beam_width=d["beam_width"], neural_scale=d["neural_scale"])
                            chunk = dict(manifest_sha256=m["manifest_sha256"], dataset=dataset, seed=job["seed"],
                                         conditions=["B"], split="test", test_evaluated=True,
                                         neural_representation="K0_only", branch="open", alignment="real", selected_weight=.1,
                                         seconds=time.perf_counter()-started,
                                         records=[dict(group=g["group"], result=r) for g, r in zip(batch, results)])
                            validate_b_chunk(chunk, m, job, batch)
                            validate_initial_agreement(chunk["records"], ac["records"][start:start + len(batch)])
                            save(path, chunk)
                            scorer.save_cache(cache_path)
                        validate_initial_agreement(chunk["records"], ac["records"][start:start + len(batch)])
                        records.extend(chunk["records"])
                        sources.append(dict(path=source_key(path), sha256=sha256(path)))
                        completed += len(batch)
                        status(m, state="running", stage="chunk_complete", dataset=dataset, seed=job["seed"],
                               chunk=number, completed_segment_seed_units=completed)
                    payload = dict(status="completed", manifest_sha256=m["manifest_sha256"], dataset=dataset,
                                   seed=job["seed"], conditions=["B"], split="test", test_evaluated=True,
                                   neural_representation="K0_only", branch="open", alignment="real", selected_weight=.1,
                                   records=records, sources=sources)
                    validate_b_chunk(payload, m, job, groups)
                    save(root / "main.json", payload)
                    del k0, ids, ac
                del scorer, embed, bank
                gc.collect()
                torch.cuda.empty_cache()
            require(completed == m["total_segment_seed_units"], "B completion population differs")
            status(m, stage="exporting", completed_segment_seed_units=completed)
            export(m, parent)
            status(m, state="completed", stage="completed", completed_segment_seed_units=completed,
                   completed_unix=time.time())
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

# --- Merged from fixed_k0_rolling_search_test_v2_results.py ---
"""Paired A/B/C gain decomposition, retaining the original A/C test results."""

from collections import defaultdict
import time

import numpy as np

from braindecoding.unified_sentence_v2_artifacts import csv_write
from braindecoding.unified_sentence_v2_results import METRICS, holm, metric_row, paired_block_statistics


CONDITIONS = ("A", "B", "C")
PAIRS = (("B_minus_A", "A", "B"), ("C_minus_B", "B", "C"), ("C_minus_A", "A", "C"))
NAMES = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}


def summarize(rows):
    """Pool words within subjects for WER/accuracy; average segments for BLEU-1."""
    require(bool(rows), "Cannot summarize an incomplete B experiment")
    cells = defaultdict(list)
    for row in rows:
        require(row["condition"] in CONDITIONS and row["words"] > 0, "Unknown condition or empty segment")
        cells[(row["seed"], row["subject"], row["condition"])].append(row)
    subjects = []
    for (seed, subject, condition), group in sorted(cells.items()):
        words = sum(r["words"] for r in group)
        values = {metric: (float(np.mean([r[metric] for r in group])) if metric == "sentence_bleu1_mean"
                           else float(sum(r[metric] * r["words"] for r in group) / words)) for metric in METRICS}
        subjects.append(dict(seed=seed, subject=subject, condition=condition, segments=len(group), words=words, **values))
    seed_ids = sorted({r["seed"] for r in rows})
    subject_sets = [{r["subject"] for r in subjects if r["seed"] == seed and r["condition"] == condition}
                    for seed in seed_ids for condition in CONDITIONS]
    require(all(s and s == subject_sets[0] for s in subject_sets), "Subjects differ across seeds or A/B/C")
    by_seed = []
    for seed in seed_ids:
        for condition in CONDITIONS:
            group = [r for r in subjects if r["seed"] == seed and r["condition"] == condition]
            by_seed.append(dict(seed=seed, condition=condition,
                                **{k: float(np.mean([r[k] for r in group])) for k in METRICS}))
    summary = [dict(condition=c, **{k: float(np.mean([r[k] for r in by_seed if r["condition"] == c]))
                                   for k in METRICS}) for c in CONDITIONS]
    return dict(summary=summary, by_seed=by_seed, by_subject_seed=subjects)


def gain_decomposition(summary):
    """Signed differences; WER reductions are positive; shares are never clipped."""
    lookup = {r["condition"]: r for r in summary}
    require(set(lookup) == set(CONDITIONS), "Gain decomposition needs A/B/C")
    output = {}
    for metric in METRICS:
        sign = -1. if metric == "wer" else 1.
        search = sign * (lookup["B"][metric] - lookup["A"][metric])
        neural = sign * (lookup["C"][metric] - lookup["B"][metric])
        total = sign * (lookup["C"][metric] - lookup["A"][metric])
        require(np.isclose(search + neural, total, rtol=0, atol=1e-12), "A/B/C gains do not add up")
        output[metric] = dict(search_gain_B_minus_A=search, neural_update_gain_C_minus_B=neural,
                              total_gain_C_minus_A=total, gain_direction="reduction" if sign < 0 else "increase",
                              search_share=search / total if total > 1e-12 else None,
                              neural_update_share=neural / total if total > 1e-12 else None,
                              share_interpretation="descriptive_signed_difference_decomposition")
    return output


def paired_statistics(rows, cfg):
    pairs = defaultdict(dict)
    for row in rows:
        key = row["seed"], row["subject"], row["group"]
        require(row["condition"] not in pairs[key], "Duplicate A/B/C group metric")
        pairs[key][row["condition"]] = row
    require(all(set(pair) == set(CONDITIONS) for pair in pairs.values()), "Unpaired A/B/C population")
    statistics = {}
    for name, left, right in PAIRS[:2]:
        differences = []
        for pair in pairs.values():
            l, r = pair[left], pair[right]
            require(all(l[k] == r[k] for k in ("seed", "subject", "group", "block", "scope", "words")),
                    "A/B/C material pairing changed")
            differences.append(dict(**{k: r[k] for k in ("seed", "subject", "group", "block", "scope")},
                                    difference=r["sentence_bleu1_mean"] - l["sentence_bleu1_mean"]))
        statistics[name] = paired_block_statistics(differences, "difference", cfg)
    return statistics


def transition_counts(reference, left, right):
    require(len(reference) == len(left) == len(right), "Transition population differs")
    repaired = sum(a != t and b == t for t, a, b in zip(reference, left, right))
    harmed = sum(a == t and b != t for t, a, b in zip(reference, left, right))
    return dict(eligible=len(reference), changed=sum(a != b for a, b in zip(left, right)),
                repaired=repaired, harmed=harmed, net_correct_words=repaired-harmed)


def change_rows(group, ac_result, b_result, identity):
    reference = group["reference"]
    final = {**ac_result["final"], "B": b_result["final"]["B"]}
    output = [dict(**identity, comparison=f"{left}_to_{right}",
                   **transition_counts(reference, final[left], final[right])) for _, left, right in PAIRS]
    for condition, result in (("B", b_result), ("C", ac_result)):
        first = [step[condition][i] for i, step in enumerate(result["trace"])]
        output.append(dict(**identity, comparison=f"{condition}_first_to_final",
                           **transition_counts(reference[:-1], first[:-1], final[condition][:-1])))
    return output


def summarize_changes(rows):
    cells = defaultdict(list)
    for row in rows:
        cells[(row["comparison"], row["seed"], row["subject"])].append(row)
    subjects = []
    for (comparison, seed, subject), group in sorted(cells.items()):
        counts = {k: sum(r[k] for r in group) for k in ("eligible", "changed", "repaired", "harmed", "net_correct_words")}
        rates = {f"{k}_rate": counts[k] / counts["eligible"] if counts["eligible"] else 0.
                 for k in ("changed", "repaired", "harmed", "net_correct_words")}
        subjects.append(dict(comparison=comparison, seed=seed, subject=subject, **counts, **rates))
    output = []
    for comparison in sorted({r["comparison"] for r in rows}):
        group = [r for r in subjects if r["comparison"] == comparison]
        seeds = sorted({r["seed"] for r in group})
        counts = {k: sum(r[k] for r in group) for k in ("eligible", "changed", "repaired", "harmed", "net_correct_words")}
        rates = {k: float(np.mean([np.mean([r[k] for r in group if r["seed"] == s]) for s in seeds]))
                 for k in ("changed_rate", "repaired_rate", "harmed_rate", "net_correct_words_rate")}
        output.append(dict(comparison=comparison, **counts, **rates, count_unit="word_position_seed_decisions",
                           rate_aggregation="within_subject_pooled_words_then_equal_subjects_then_equal_seeds"))
    return output, subjects


def export(m, parent):
    verify_hashes(m["source_sha256"])
    original = read(ORIGINAL_SUMMARY)
    datasets, all_rows, all_changes, all_subjects, all_change_subjects = {}, [], [], [], []
    all_seeds, all_gain_seeds, sources = [], [], []
    for dataset in m["settings"]["protocol"]["datasets"]:
        info = parent["datasets"][dataset]
        rows, changes = [], []
        for job in (j for j in m["jobs"] if j["dataset"] == dataset):
            path = OUT / dataset / f"seed-{job['seed']:03d}" / "main.json"
            b = read(path)
            require(b["status"] == "completed", "Cannot export incomplete B seed")
            validate_b_chunk(b, m, job, info["groups"])
            reconstructed = []
            for item in b["sources"]:
                chunk_path = ROOT / item["path"]
                require(sha256(chunk_path) == item["sha256"], "A completed B source changed")
                chunk = read(chunk_path)
                start = len(reconstructed)
                validate_b_chunk(chunk, m, job, info["groups"][start:start + len(chunk["records"])])
                reconstructed.extend(chunk["records"])
            require(reconstructed == b["records"], "B atomic chunks differ from saved outputs")
            ac = original_ac(parent, dataset, job["seed"], ROOT / job["original_ac"])
            validate_initial_agreement(b["records"], ac["records"])
            for group, old, new in zip(info["groups"], ac["records"], b["records"]):
                identity = dict(dataset=dataset, seed=job["seed"], subject=group["subject"], group=group["group"],
                                material=group["material"], block=group["block"], scope=group["scope"])
                for condition in CONDITIONS:
                    prediction = new["result"]["final"]["B"] if condition == "B" else old["result"]["final"][condition]
                    rows.append(dict(**identity, condition=condition, words=len(group["reference"]),
                                     **metric_row(group["reference"], prediction, info["language"] == "zh")))
                changes.extend(change_rows(group, old["result"], new["result"], identity))
            sources.append(dict(path=source_key(path), sha256=sha256(path)))
        aggregate = summarize(rows)
        lookup = {r["condition"]: r for r in aggregate["summary"]}
        for expected in original["datasets"][dataset]["summary"]:
            for metric in METRICS:
                require(np.isclose(lookup[expected["condition"]][metric], expected[metric], rtol=0, atol=1e-12),
                        f"Original A/C metric no longer reproduced: {dataset}, {expected['condition']}, {metric}")
        gains = gain_decomposition(aggregate["summary"])
        statistics = paired_statistics(rows, m["settings"]["statistics"])
        for name, key in (("B_minus_A", "search_gain_B_minus_A"), ("C_minus_B", "neural_update_gain_C_minus_B")):
            require(np.isclose(statistics[name]["effect"], gains["sentence_bleu1_mean"][key], atol=1e-12),
                    "Paired material effect differs from subject/seed aggregate")
        change_summary, change_subjects = summarize_changes(changes)
        by_change = {r["comparison"]: r for r in change_summary}
        require(by_change["A_to_B"]["net_correct_words"] + by_change["B_to_C"]["net_correct_words"]
                == by_change["A_to_C"]["net_correct_words"], "Net correct-word decomposition differs")
        datasets[dataset] = dict(counts=info["counts"], **aggregate, gains=gains, supplementary_statistics=statistics,
                                  original_C_minus_A_statistics=dict(original["datasets"][dataset]["primary_statistics"]),
                                  transitions=change_summary)
        for seed in m["settings"]["protocol"]["seeds"]:
            values = gain_decomposition([r for r in aggregate["by_seed"] if r["seed"] == seed])
            all_gain_seeds.extend(dict(dataset=dataset, seed=seed, metric=metric, **gain) for metric, gain in values.items())
        all_rows.extend(rows)
        all_changes.extend(changes)
        all_subjects.extend(dict(dataset=dataset, **r) for r in aggregate["by_subject_seed"])
        all_change_subjects.extend(dict(dataset=dataset, **r) for r in change_subjects)
        all_seeds.extend(dict(dataset=dataset, **r) for r in aggregate["by_seed"])
    stats = [stat for info in datasets.values() for stat in info["supplementary_statistics"].values()]
    require(len(stats) == 4, "Expected four supplementary comparisons")
    for stat, p in zip(stats, holm([s["p_one_sided"] for s in stats])):
        stat.update(p_holm=float(p), family="two_datasets_by_two_supplementary_comparisons")
    require(len(all_rows) == m["total_segment_seed_units"] * 3
            and len(all_changes) == m["total_segment_seed_units"] * 5, "A/B/C full coverage differs")
    exports = {}
    for suffix, records in (("by_group", all_rows), ("by_subject_seed", all_subjects), ("by_seed", all_seeds),
                            ("gain_by_seed", all_gain_seeds), ("transitions_by_group", all_changes),
                            ("transitions_by_subject_seed", all_change_subjects)):
        path = ROOT / "data/results/exports" / f"{ID}_{suffix}.csv"
        csv_write(path, records)
        exports[source_key(path)] = sha256(path)
    report = dict(status="completed", protocol=ID, manifest_sha256=m["manifest_sha256"],
                  parent_manifest_sha256=m["parent_manifest_sha256"], split="test", test_evaluated=True,
                  conditions=list(CONDITIONS), computed_conditions=["B"], seeds=[0, 1, 2],
                  segment_seed_units=m["total_segment_seed_units"], evidence=m["evidence"], datasets=datasets,
                  original_AC_reproduced=True, original_AC_unchanged=True, neural_models_unchanged=True,
                  B_neural_input="K0_only", sources=sources, exports=exports, completed_unix=time.time(),
                  interpretation="B-A: K0 rolling-search gain; C-B: added neural updates under rolling search; generated prefix and candidates interact.")
    verify_hashes(m["source_sha256"])
    save(REPORT / "results.json", report)
    summary_path = ROOT / "data/results/exports" / f"{ID}_summary.json"
    save(summary_path, report)
    save(REPORT / "audit.json", dict(status="verified_complete", manifest_sha256=m["manifest_sha256"],
                                     results_sha256=sha256(summary_path), exports=exports, sources=sources,
                                     segment_seed_units=m["total_segment_seed_units"], original_AC_reproduced=True,
                                     source_sha256=m["source_sha256"]))
    write_markdown(report)
    return report


def write_markdown(report):
    lines = ["# 固定 K0 滚动文本重搜：A/B/C 完整测试结果", "",
             f"两个数据集、三个种子、全部 {report['segment_seed_units']:,} 个片段×种子均已完成。A/C 原结果逐项复核；新增计算仅为 B。", "",
             "| 数据集 | A BLEU-1 (%) | B BLEU-1 (%) | C BLEU-1 (%) | 重搜 B−A（百分点） | 神经更新 C−B（百分点） | 总增益 C−A（百分点） |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, info in report["datasets"].items():
        v = {r["condition"]: r for r in info["summary"]}
        g = info["gains"]["sentence_bleu1_mean"]
        lines.append(f"| {NAMES[dataset]} | {v['A']['sentence_bleu1_mean']*100:.2f} | {v['B']['sentence_bleu1_mean']*100:.2f} | {v['C']['sentence_bleu1_mean']*100:.2f} | {g['search_gain_B_minus_A']*100:+.2f} | {g['neural_update_gain_C_minus_B']*100:+.2f} | {g['total_gain_C_minus_A']*100:+.2f} |")
    lines += ["", "| 数据集 | 增益段 | 95% 配对材料块区间（百分点） | 占原总增益 | 补充检验 Holm p |",
              "|---|---|---|---:|---:|"]
    for dataset, info in report["datasets"].items():
        gains = info["gains"]["sentence_bleu1_mean"]
        for comparison, label, share_key in (("B_minus_A", "A→B：固定 K0 重搜", "search_share"),
                                             ("C_minus_B", "B→C：允许神经更新", "neural_update_share")):
            s = info["supplementary_statistics"][comparison]
            share = "不定义" if gains[share_key] is None else f"{gains[share_key]*100:.1f}%"
            lines.append(f"| {NAMES[dataset]} | {label} | [{s['ci_lower']*100:+.2f}, {s['ci_upper']*100:+.2f}] | {share} | {s['p_holm']:.6f} |")
    lines += ["", "主指标先受试者内平均片段，再对受试者及种子等权。区间按共享材料块、在记录／章节内分层重采样，保留块内全部受试者和种子。四个新增检验共同作 Holm 校正，属于看过原 A/C 后的补充消融。原 A/C 主检验未修改。", "",
              "占比为带符号的描述性差值分解；总增益非正或近零时不定义，不截断负数或超过100%的值。B−A 包括滚动搜索路径和自身生成前缀的变化；C−B 是同一滚动策略中允许神经更新的增量，候选与前缀交互仍存在。", "",
              "| 数据集 | 策略 | WER (%) | 对齐词准确率 (%) |", "|---|---|---:|---:|"]
    for dataset, info in report["datasets"].items():
        for row in info["summary"]:
            lines.append(f"| {NAMES[dataset]} | {row['condition']} | {row['wer']*100:.2f} | {row['aligned_word_accuracy']*100:.2f} |")
    lines += ["", "| 数据集 | 词位变化 | 可评价决策 | 改对 | 误改 | 净增正确词 | 改对率 (%) | 误改率 (%) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for dataset, info in report["datasets"].items():
        for row in info["transitions"]:
            lines.append(f"| {NAMES[dataset]} | {row['comparison']} | {row['eligible']:,} | {row['repaired']:,} | {row['harmed']:,} | {row['net_correct_words']:+,} | {row['repaired_rate']*100:.2f} | {row['harmed_rate']*100:.2f} |")
    lines += ["", "计数单位为词位×种子决策；比例先受试者内合并词位，再对受试者和种子等权。在线首次→最终修正排除片段末词。逐片段、受试者×种子、逐种子增益与修正记录见正式 CSV 导出。", "",
              f"冻结清单 SHA-256：`{report['manifest_sha256']}`。", ""]
    (REPORT / "results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
