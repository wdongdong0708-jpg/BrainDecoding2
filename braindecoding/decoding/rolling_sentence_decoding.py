"""Frozen validation-only sentence experiment for completed neural protocol v2.

Prepare material partitions first. Encode all three primary revision models,
calibrate only frozen A outputs, then replay real and shifted A/B/C together.
Completed atomic chunks can be reused only under the identical manifest.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
import yaml

from braindecoding.config import PROJECT_ROOT, project_path
from braindecoding.data.text import encode_words, load_text_embedding_cache, load_text_model
from braindecoding.models import build_brain_embedding_model
from braindecoding.training.runtime import load_checkpoint
from braindecoding.training.word import encode_loader
from braindecoding.unified_protocol_audit import load_source, sha256
from braindecoding.unified_rolling_v2_training import config as neural_config, dataset_and_loader
from braindecoding.unified_sentence_batch import BatchedWordLM, replay_many


CONFIG = PROJECT_ROOT / "configs/decoding/rolling_sentence_decoding.yaml"
MANIFEST = PROJECT_ROOT / "data/manifests/rolling_sentence/unified_sentence_v2.json"
OUT = PROJECT_ROOT / "outputs/sentence_decoding/unified_sentence_v2"
RUNTIME = PROJECT_ROOT / ".cache/runs/unified_sentence_v2"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def settings():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def canonical_group_rows(dataset, table):
    """Connect segments sharing stimulus positions/lines, including voices.

    Group boundaries and references are retained exactly. Connectivity is used
    only for partition and statistical blocks, never as decoder context.
    """
    groups, parent, owner = [], [], {}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for name, rows in table.groupby("protocol_group", sort=False):
        i = len(groups)
        parent.append(i)
        if dataset == "pallier2025":
            scope = str(rows["运行编号"].iloc[0])
            atoms = [f"{scope}|event-{v}" for v in rows["BIDS事件行号"]]
            order = int(rows["BIDS事件行号"].min())
        else:
            scope = f"chapter-{int(rows.chapter.iloc[0]):02d}"
            lines = [re.search(r"actual-line-(\d+)", str(v)).group(1)
                     for v in rows.source_sentence_uid]
            atoms = [f"{scope}|source-line-{v}" for v in lines]
            order = min(map(int, lines))
        for atom in set(atoms):
            if atom in owner:
                a, b = root(i), root(owner[atom])
                parent[max(a, b)] = min(a, b)
            else:
                owner[atom] = i
        groups.append(dict(group=str(name), subject=str(rows.subject_id.iloc[0]),
            recording=str(rows.recording_id.iloc[0]), scope=scope, order=order,
            event_ids=rows.event_id.astype(str).tolist(), reference=rows.normalized_word.astype(str).tolist(),
            atoms=sorted(set(atoms))))
    components = defaultdict(list)
    for i in range(len(groups)):
        components[root(i)].append(i)
    material = []
    for members in components.values():
        atoms = sorted({a for i in members for a in groups[i]["atoms"]})
        scopes = {groups[i]["scope"] for i in members}
        if len(scopes) != 1:
            raise ValueError("A material component spans recordings/chapters")
        key = digest(atoms)[:20]
        material.append(dict(material=key, scope=scopes.pop(),
                             order=min(groups[i]["order"] for i in members), groups=members))
        for i in members:
            groups[i]["material"] = key
    return groups, material


def prepare():
    cfg, nc = settings(), neural_config()
    if cfg["protocol"]["test_evaluation_allowed"] or nc["protocol"]["test_evaluation_allowed"]:
        raise ValueError("Test evaluation is outside this experiment")
    queue = read(PROJECT_ROOT / "data/results/protocols/unified_rolling_v2_queue_results.json")
    if queue["status"] != "completed" or len(queue["completed"]) != 12:
        raise ValueError("All 12 v2 training jobs must be complete")
    checkpoints = [j for j in queue["completed"] if j["stage"] == "revision"]
    dataset_manifests = {}
    for dataset, adapter in nc["adapters"].items():
        source_cfg, table = load_source(dataset, "val")
        groups, materials = canonical_group_rows(dataset, table)
        by_scope = defaultdict(list)
        for material in materials:
            by_scope[material["scope"]].append(material)
        blocks = []
        for scope, entries in sorted(by_scope.items()):
            entries.sort(key=lambda v: (v["order"], v["material"]))
            width = cfg["partition"]["adjacent_components_per_block"]
            scope_blocks = []
            for start in range(0, len(entries), width):
                key = f"{dataset}|{scope}|block-{start // width:03d}"
                members = entries[start:start + width]
                ids = [i for m in members for i in m["groups"]]
                scope_blocks.append(dict(block=key, scope=scope, materials=[m["material"] for m in members], groups=ids))
            rng = np.random.default_rng(int(digest([cfg["partition"]["seed"], dataset, scope])[:12], 16))
            n_cal = max(1, round(len(scope_blocks) * cfg["partition"]["calibration_fraction"]))
            if n_cal >= len(scope_blocks):
                raise ValueError("Insufficient material blocks for disjoint calibration/evaluation")
            calibration = set(rng.choice(len(scope_blocks), n_cal, replace=False).tolist())
            for i, block in enumerate(scope_blocks):
                part = "calibration" if i in calibration else "evaluation"
                block["partition"] = part
                for j in block["groups"]:
                    groups[j].update(block=block["block"], partition=part)
                blocks.append(block)
        train_vocabulary = read(project_path(adapter["vocabulary_manifest"]))["vocabulary"]
        atom_partitions = defaultdict(set)
        for group in groups:
            for atom in group.pop("atoms"):
                atom_partitions[atom].add(group["partition"])
        if any(len(parts) != 1 for parts in atom_partitions.values()):
            raise ValueError("Stimulus leakage across calibration/evaluation")
        dataset_manifests[dataset] = dict(language=adapter["language"],
            event_table=str(project_path(source_cfg["cache"]["event_table"])),
            event_table_sha256=sha256(project_path(source_cfg["cache"]["event_table"])),
            embedding_cache=str(project_path(adapter["embedding_cache"])),
            embedding_sha256=sha256(project_path(adapter["embedding_cache"])),
            vocabulary=train_vocabulary, vocabulary_sha256=digest(train_vocabulary),
            groups=groups, blocks=blocks, material_components=len(materials),
            counts={part: dict(groups=sum(g["partition"] == part for g in groups),
                              words=sum(len(g["event_ids"]) for g in groups if g["partition"] == part),
                              blocks=sum(b["partition"] == part for b in blocks))
                    for part in ("calibration", "evaluation")})
    sources = [CONFIG, Path(__file__), PROJECT_ROOT / "braindecoding/decoding/unified_sentence.py",
               PROJECT_ROOT / "braindecoding/decoding/unified_sentence_batch.py",
               PROJECT_ROOT / "braindecoding/evaluation/unified_sentence_v2_results.py"]
    payload = dict(schema_version=1, protocol="unified_sentence_v2", settings=cfg,
        neural_settings=nc, checkpoints=checkpoints, datasets=dataset_manifests,
        sources={p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in sources},
        model_files={p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in
                     (PROJECT_ROOT / nc["language_model"]["local_path"]).glob("*") if p.is_file()},
        evidence="development_validation", test_evaluated=False)
    payload["manifest_sha256"] = digest(payload)
    if MANIFEST.exists() and read(MANIFEST) != payload:
        raise ValueError("Frozen manifest differs; use a new protocol/output group rather than overwrite")
    save(MANIFEST, payload)
    print(json.dumps(dict(status="manifest_frozen", manifest_sha256=payload["manifest_sha256"],
        datasets={d: v["counts"] for d, v in dataset_manifests.items()}), ensure_ascii=False), flush=True)
    return payload


def manifest():
    result = read(MANIFEST)
    if digest({k: v for k, v in result.items() if k != "manifest_sha256"}) != result["manifest_sha256"]:
        raise ValueError("Manifest content hash mismatch")
    for p, checksum in result["sources"].items():
        if str(p).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        if sha256(PROJECT_ROOT / p) != checksum:
            raise ValueError(f"Frozen source changed: {p}")
    return result


def status(**fields):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    path = RUNTIME / "status.json"
    prior = read(path) if path.exists() else {}
    prior.update(pid=os.getpid(), updated_unix=time.time(), test_evaluated=False, **fields)
    save(path, prior)


def encoded_path(dataset, seed):
    return OUT / dataset / f"seed-{seed:03d}" / "neural_predictions.npz"


def encode_all(m):
    for dataset, info in m["datasets"].items():
        for job in (j for j in m["checkpoints"] if j["dataset"] == dataset):
            seed, checkpoint_path = job["seed"], Path(job["checkpoint"])
            destination = encoded_path(dataset, seed)
            sidecar = destination.with_suffix(".json")
            if destination.exists() and sidecar.exists():
                old = read(sidecar)
                if old["manifest_sha256"] == m["manifest_sha256"] and old["prediction_sha256"] == sha256(destination):
                    continue
                raise ValueError("Prediction cache does not match frozen manifest")
            status(state="running", stage="neural_encoding", dataset=dataset, seed=seed)
            if sha256(checkpoint_path) != job["checkpoint_sha256"]:
                raise ValueError("Primary checkpoint changed")
            cfg = yaml.safe_load((Path(job["path"]) / "resolved_config.yaml").read_text(encoding="utf-8"))
            ds, loader, _ = dataset_and_loader(cfg, "val")
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint["model_config"] != cfg["model"] or checkpoint["stage"] != "revision":
                raise ValueError("Primary revision checkpoint mismatch")
            model = build_brain_embedding_model(ds.channel_count, ds.channel_positions,
                ds.subject_count, checkpoint["model_config"]).cuda().eval()
            model.load_state_dict(checkpoint["model_state"], strict=True)
            started = time.perf_counter()
            encoded = encode_loader(model, loader, torch.device("cuda"), amp=True)
            expected = {e for g in info["groups"] for e in g["event_ids"]}
            if len(encoded["event_ids"]) != len(expected) or set(encoded["event_ids"]) != expected:
                raise ValueError("Encoded event population differs")
            lookup = {e: i for i, e in enumerate(encoded["event_ids"])}
            for group in info["groups"]:
                ids = [lookup[e] for e in group["event_ids"]]
                for d in range(3):
                    observed = encoded["revision_availability_masks"][d][ids].numpy()
                    if not np.array_equal(observed, np.arange(len(ids)) + d < len(ids)):
                        raise ValueError("Encoded availability differs from known segment boundary")
            arrays = dict(event_ids=np.asarray(encoded["event_ids"]))
            for d, values in encoded["revision_predictions"].items():
                matrix = values.numpy()
                if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1., atol=.005):
                    raise ValueError("Predictions are not finite normalized cosine embeddings")
                arrays[f"k{d}"] = matrix
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.with_suffix(".tmp").open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            destination.with_suffix(".tmp").replace(destination)
            save(sidecar, dict(manifest_sha256=m["manifest_sha256"], prediction_sha256=sha256(destination),
                checkpoint_sha256=job["checkpoint_sha256"], events=len(expected),
                seconds=time.perf_counter() - started, test_evaluated=False))
            print(json.dumps(dict(stage="encoded", dataset=dataset, seed=seed, events=len(expected),
                                  seconds=time.perf_counter()-started)), flush=True)
            del model, ds, loader, checkpoint, encoded
            gc.collect()
            torch.cuda.empty_cache()


def shifted_mapping(info, event_ids, repetition, cfg):
    lookup = {e: i for i, e in enumerate(event_ids)}
    classes = defaultdict(list)
    for group in info["groups"]:
        for position, event in enumerate(group["event_ids"]):
            available = min(2, len(group["event_ids"]) - 1 - position)
            classes[(group["subject"], group["recording"], available)].append(lookup[event])
    mapping = np.arange(len(event_ids))
    for key, ids in sorted(classes.items()):
        if len(ids) < 2:
            raise ValueError("No nonidentity donor in an availability stratum")
        ids = np.asarray(ids)
        seed = int(digest([cfg["controls"]["shifted_seed"], repetition, key])[:12], 16)
        rng = np.random.default_rng(seed)
        lo, hi = cfg["controls"]["shifted_offset_fraction"]
        offsets = np.arange(max(1, int(np.ceil(len(ids) * lo))),
                            min(len(ids) - 1, int(np.floor(len(ids) * hi))) + 1)
        offset = int(rng.choice(offsets))
        mapping[ids] = np.roll(ids, offset)
    if np.any(mapping == np.arange(len(mapping))) or len(np.unique(mapping)) != len(mapping):
        raise ValueError("Shift mapping is not a complete nonidentity permutation")
    return mapping


def score_mean(records, groups, condition="A"):
    from braindecoding.jayalath_sentence import sentence_metrics
    per_subject = defaultdict(list)
    group_lookup = {g["group"]: g for g in groups}
    for record in records:
        group = group_lookup[record["group"]]
        bleu = sentence_metrics([group["reference"]], [record["result"]["final"][condition]])["sentence_bleu1_mean"]
        per_subject[group["subject"]].append(bleu)
    return float(np.mean([np.mean(v) for v in per_subject.values()]))


def load_lm(m, dataset):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = copy.deepcopy(m["neural_settings"]["language_model"])
    cfg.update({k: m["settings"]["decoding"][k] for k in
                ("proposal_count", "proposal_max_new_tokens", "proposal_batch_size", "score_batch_size", "score_cache_size")})
    path = PROJECT_ROOT / cfg["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float16).cuda().eval()
    scorer = BatchedWordLM(model, tokenizer, m["datasets"][dataset]["language"], cfg)
    embeddings = load_text_embedding_cache(m["datasets"][dataset]["embedding_cache"])
    text_tokenizer, text_model, device = load_text_model(m["neural_settings"]["text_embedding"])

    def embed(words):
        missing = list(dict.fromkeys(w for w in words if w not in embeddings))
        if missing:
            vectors = encode_words(missing, m["neural_settings"]["text_embedding"],
                tokenizer=text_tokenizer, model=text_model, device=device)
            embeddings.update(zip(missing, vectors))
        return np.stack([embeddings[w] for w in words])

    bank = np.stack([embeddings[w] for w in m["datasets"][dataset]["vocabulary"]])
    return scorer, embed, bank


def cases_for_groups(groups, predictions, lookup, *, weight, branch, mapping=None):
    cases = []
    for group in groups:
        ids = np.array([lookup[e] for e in group["event_ids"]])
        if mapping is not None:
            ids = mapping[ids]
        cases.append(dict(predictions={d: predictions[d][ids] for d in range(3)},
                          weight=weight, open_vocabulary=branch == "open"))
    return cases


def decode_chunk(m, dataset, scorer, embed, bank, groups, cases, path, label, conditions="ABC"):
    if path.exists():
        old = read(path)
        if old["manifest_sha256"] != m["manifest_sha256"] or old["label"] != label:
            raise ValueError("Completed decode chunk provenance mismatch")
        return old["records"]
    started, last_progress = time.perf_counter(), [0.]

    def progress(arrival, counts):
        now = time.perf_counter()
        if now - last_progress[0] >= 30:
            status(state="running", stage="sentence_decoding", dataset=dataset, label=label,
                   chunk=str(path.relative_to(PROJECT_ROOT)), arrival=arrival, longest_segment=max(counts),
                   chunk_seconds=now-started)
            last_progress[0] = now

    ds = m["settings"]["decoding"]
    results = replay_many(cases, m["datasets"][dataset]["vocabulary"], bank, scorer, embed,
        conditions=conditions, progress=progress, candidate_count=ds["candidates"],
        beam_width=ds["beam_width"], neural_scale=ds["neural_scale"])
    records = [dict(group=g["group"], result=result) for g, result in zip(groups, results)]
    save(path, dict(manifest_sha256=m["manifest_sha256"], label=label, records=records,
                    seconds=time.perf_counter()-started, test_evaluated=False))
    print(json.dumps(dict(stage="chunk_complete", dataset=dataset, label=label, groups=len(groups),
                          words=sum(len(g["event_ids"]) for g in groups), seconds=time.perf_counter()-started)), flush=True)
    return records


def run_dataset(m, dataset):
    cfg, info = m["settings"], m["datasets"][dataset]
    scorer, embed, bank = load_lm(m, dataset)
    root = OUT / dataset
    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / "lm_cache.pkl"
    scorer.load_cache(cache_path)
    all_predictions = {}
    for seed in cfg["protocol"]["seeds"]:
        archive = np.load(encoded_path(dataset, seed))
        all_predictions[seed] = ({d: archive[f"k{d}"] for d in range(3)},
                                 {str(e): i for i, e in enumerate(archive["event_ids"])}, archive["event_ids"])
    width = cfg["decoding"]["segment_batch_size"]
    groups = [g for g in info["groups"] if g["partition"] == "calibration"]
    calibration_path = root / "calibration.json"
    if calibration_path.exists():
        calibration = read(calibration_path)
        if calibration["manifest_sha256"] != m["manifest_sha256"]:
            raise ValueError("Calibration provenance mismatch")
        weight = calibration["selected_weight"]
    else:
        candidates = []
        for weight in cfg["calibration"]["language_weight_grid"]:
            scores = []
            for seed, (predictions, lookup, _) in all_predictions.items():
                records = []
                for start in range(0, len(groups), width):
                    batch = groups[start:start + width]
                    path = root / "calibration_chunks" / f"lambda-{weight:g}_seed-{seed:03d}_chunk-{start//width:04d}.json"
                    cases = cases_for_groups(batch, predictions, lookup, weight=weight, branch="open")
                    records.extend(decode_chunk(m, dataset, scorer, embed, bank, batch, cases, path,
                        f"calibration_A_lambda-{weight:g}_seed-{seed}", conditions="A"))
                    scorer.save_cache(cache_path)
                scores.append(score_mean(records, groups))
            candidates.append(dict(weight=weight, seed_scores=scores, mean=float(np.mean(scores))))
        winner = sorted(candidates, key=lambda v: (-v["mean"], v["weight"]))[0]
        weight = winner["weight"]
        save(calibration_path, dict(manifest_sha256=m["manifest_sha256"], selected_weight=weight,
            grid=candidates, condition="A", branch="open", metric=cfg["calibration"]["metric"], test_evaluated=False))
        print(json.dumps(dict(stage="calibrated", dataset=dataset, selected_weight=weight, grid=candidates)), flush=True)
    groups = [g for g in info["groups"] if g["partition"] == "evaluation"]
    for seed, (predictions, lookup, events) in all_predictions.items():
        for repetition in range(cfg["controls"]["shifted_repetitions"]):
            mapping = shifted_mapping(info, events, repetition, cfg)
            save(root / f"seed-{seed:03d}" / f"shift-{repetition:02d}.json", dict(
                manifest_sha256=m["manifest_sha256"], event_ids=list(map(str, events)),
                donor_event_ids=list(map(str, events[mapping])), mapping_sha256=digest(mapping.tolist()),
                same_donor_all_horizons=True, reference_words_used=False))
        for branch in cfg["decoding"]["branches"]:
            for alignment in ["real"] + [f"shift-{i:02d}" for i in range(cfg["controls"]["shifted_repetitions"])]:
                mapping = None if alignment == "real" else shifted_mapping(info, events, int(alignment[-2:]), cfg)
                for start in range(0, len(groups), width):
                    batch = groups[start:start + width]
                    cases = cases_for_groups(batch, predictions, lookup, weight=weight, branch=branch, mapping=mapping)
                    path = root / f"seed-{seed:03d}" / branch / alignment / f"chunk-{start//width:04d}.json"
                    decode_chunk(m, dataset, scorer, embed, bank, batch, cases, path,
                                 f"evaluation_{branch}_{alignment}_seed-{seed}")
                    scorer.save_cache(cache_path)
    # One language-only bounded replay suffices for each possible known length;
    # it is independent of subject, seed and reference words.
    pure_path = root / "pure_language.json"
    if not pure_path.exists():
        maximum = max(len(g["event_ids"]) for g in groups)
        zeros = {d: np.zeros((maximum, bank.shape[1])) for d in range(3)}
        case = dict(predictions=zeros, weight=1., open_vocabulary=True, pure_lm=True)
        result = replay_many([case], info["vocabulary"], bank, scorer, None, conditions="C",
            candidate_count=cfg["decoding"]["candidates"], beam_width=cfg["decoding"]["beam_width"],
            neural_scale=cfg["decoding"]["neural_scale"])[0]
        save(pure_path, dict(manifest_sha256=m["manifest_sha256"], result=result,
                            no_brain_shortlist=True, reference_words_used=False, test_evaluated=False))
    save(root / "completion.json", dict(status="completed", manifest_sha256=m["manifest_sha256"],
        selected_weight=weight, test_evaluated=False, completed_unix=time.time()))
    del scorer, embed, bank, all_predictions
    gc.collect()
    torch.cuda.empty_cache()


def run():
    m = manifest()
    torch.set_num_threads(4)
    status(state="running", stage="starting", manifest_sha256=m["manifest_sha256"])
    try:
        encode_all(m)
        for dataset in m["datasets"]:
            run_dataset(m, dataset)
        result = export(m)
        status(state="completed", stage="completed", result=result, completed_unix=time.time())
    except BaseException as error:
        status(state="failed", error=f"{type(error).__name__}: {error}")
        raise


def main():
    from braindecoding.config import ensure_default_environment
    ensure_default_environment()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--encode", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.encode:
        encode_all(manifest())
    else:
        run()

# --- Merged from unified_sentence_v2_results.py ---
"""Report the complete, frozen v2 sentence experiment from atomic outputs."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from braindecoding.config import PROJECT_ROOT
from braindecoding.jayalath_sentence import sentence_metrics


METRICS = ("sentence_bleu1_mean", "wer", "bleu1", "aligned_word_accuracy")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric_row(reference, prediction, chinese):
    metrics = sentence_metrics([reference], [prediction], chinese=chinese)
    metrics["aligned_word_accuracy"] = sum(a == b for a, b in zip(reference, prediction)) / len(reference)
    return {k: metrics[k] for k in METRICS}


def uniform_baseline(info, cfg):
    vocabulary = info["vocabulary"]
    index = {w: i for i, w in enumerate(vocabulary)}
    repetitions = cfg["controls"]["uniform_random_repetitions"]
    by_subject = defaultdict(list)
    covered, total = 0, 0
    for group in info["groups"]:
        if group["partition"] != "evaluation":
            continue
        reference = group["reference"]
        # A declared auxiliary uniform-word baseline, with no LM or brain.
        rng = np.random.default_rng(int(digest([cfg["controls"]["uniform_random_seed"], group["group"]])[:12], 16))
        draws = rng.integers(len(vocabulary), size=(repetitions, len(reference)))
        overlap = np.zeros(repetitions)
        for word, count in Counter(reference).items():
            if word in index:
                overlap += np.minimum((draws == index[word]).sum(1), count)
        by_subject[group["subject"]].append(overlap / len(reference))
        covered += sum(w in index for w in reference)
        total += len(reference)
    distribution = np.mean([np.mean(values, axis=0) for values in by_subject.values()], axis=0)
    return dict(sentence_bleu1_mean=float(distribution.mean()),
        simulation_lower=float(np.quantile(distribution, .025)),
        simulation_upper=float(np.quantile(distribution, .975)), repetitions=repetitions,
        aligned_word_accuracy_token_expected=covered / total / len(vocabulary),
        candidate_count=len(vocabulary), no_language_model=True, no_brain=True)


def holm(pvalues):
    order = np.argsort(pvalues, kind="stable")
    adjusted = np.zeros(len(order))
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., float(pvalues[index]) * (len(order) - rank)))
        adjusted[index] = previous
    return adjusted


def paired_block_statistics(rows, metric, statistic_settings):
    """Flip whole shared-stimulus blocks, retaining all subjects and seeds.

    Bootstrap material blocks within recording/chapter strata and recompute
    subject and seed means. A neural seed is never treated as a new subject.
    """
    blocks = sorted({r["block"] for r in rows})
    cells = sorted({(r["seed"], r["subject"]) for r in rows})
    bidx, cidx = {b: i for i, b in enumerate(blocks)}, {c: i for i, c in enumerate(cells)}
    sums = np.zeros((len(blocks), len(cells)))
    counts = np.zeros_like(sums)
    scope = {}
    for row in rows:
        i, j = bidx[row["block"]], cidx[(row["seed"], row["subject"])]
        sums[i, j] += row[metric]
        counts[i, j] += 1
        scope[row["block"]] = row["scope"]
    if (counts.sum(0) == 0).any():
        raise ValueError("Empty statistical subject/seed cell")
    # Equal numbers of subjects for every seed are required for this contract.
    seeds = sorted({c[0] for c in cells})
    if len({sum(c[0] == s for c in cells) for s in seeds}) != 1:
        raise ValueError("Inconsistent subjects across seeds")
    contributions = (sums / counts.sum(0)[None, :]).mean(1)
    observed = float(contributions.sum())
    rng = np.random.default_rng(statistic_settings["seed"])
    repetitions = statistic_settings["paired_material_block_sign_flips"]
    if len(blocks) <= 16:
        integers = np.arange(2 ** len(blocks), dtype=np.uint32)
        signs = 2 * ((integers[:, None] >> np.arange(len(blocks), dtype=np.uint32)) & 1).astype(float) - 1
        null = signs @ contributions
        pvalue = float(np.mean(null >= observed - 1e-12))
        exact = True
    else:
        signs = rng.choice([-1., 1.], (repetitions, len(blocks)))
        null = signs @ contributions
        pvalue = float((1 + np.count_nonzero(null >= observed - 1e-12)) / (repetitions + 1))
        exact = False
    strata = defaultdict(list)
    for block, i in bidx.items():
        strata[scope[block]].append(i)
    draws = []
    bootstrap_reps = statistic_settings["material_block_bootstrap_repetitions"]
    for start in range(0, bootstrap_reps, 1000):
        batch = min(1000, bootstrap_reps - start)
        weights = np.zeros((batch, len(blocks)))
        for indices in strata.values():
            picks = rng.choice(indices, (batch, len(indices)), replace=True)
            for i in range(batch):
                weights[i] += np.bincount(picks[i], minlength=len(blocks))
        denominator = weights @ counts
        numerator = weights @ sums
        valid = (denominator > 0).all(1)
        draws.extend((numerator[valid] / denominator[valid]).mean(1).tolist())
    if len(draws) < bootstrap_reps * .95:
        raise ValueError("Too many empty-subject material bootstrap samples")
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(effect=observed, ci_lower=float(lo), ci_upper=float(hi),
                p_one_sided=pvalue, blocks=len(blocks), subjects=len(cells)//len(seeds),
                seeds=len(seeds), sign_flip_exact=exact, bootstrap_samples=len(draws))


def export(manifest):
    cfg = manifest["settings"]
    all_rows, repair_rows, records, provenance = [], [], {}, []
    comparisons, random_baselines = [], {}
    for dataset, info in manifest["datasets"].items():
        root = OUT / dataset
        random_baselines[dataset] = uniform_baseline(info, cfg)
        completion = read(root / "completion.json")
        if completion["status"] != "completed" or completion["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Cannot export incomplete or mismatched experiment")
        groups = {g["group"]: g for g in info["groups"] if g["partition"] == "evaluation"}
        pure = read(root / "pure_language.json")
        if pure["manifest_sha256"] != manifest["manifest_sha256"] or not pure["no_brain_shortlist"]:
            raise ValueError("Language control is not independent")
        pure_by_length = {t["arrival"] + 1: t["C"] for t in pure["result"]["trace"]}
        seed_outcomes = {}
        for seed in cfg["protocol"]["seeds"]:
            for branch in cfg["decoding"]["branches"]:
                alignments = ["real"] + [f"shift-{i:02d}" for i in range(cfg["controls"]["shifted_repetitions"])]
                for alignment in alignments:
                    decoded = {}
                    for path in sorted((root / f"seed-{seed:03d}" / branch / alignment).glob("chunk-*.json")):
                        chunk = read(path)
                        if chunk["manifest_sha256"] != manifest["manifest_sha256"] or chunk["test_evaluated"]:
                            raise ValueError("Decode chunk provenance mismatch")
                        provenance.append(dict(path=path.relative_to(PROJECT_ROOT).as_posix(), sha256=sha256(path)))
                        for record in chunk["records"]:
                            if record["group"] in decoded:
                                raise ValueError("Duplicate decoded group")
                            decoded[record["group"]] = record["result"]
                    if set(decoded) != set(groups):
                        raise ValueError("Evaluation material population incomplete")
                    for name, result in decoded.items():
                        group = groups[name]
                        seed_outcomes[(seed, branch, alignment, name)] = result
                        reference = group["reference"]
                        for condition in "ABC":
                            prediction = result["final"][condition]
                            row = dict(dataset=dataset, seed=seed, branch=branch, alignment=alignment,
                                condition=condition, group=name, subject=group["subject"], material=group["material"],
                                block=group["block"], scope=group["scope"], words=len(reference),
                                **metric_row(reference, prediction, info["language"] == "zh"))
                            all_rows.append(row)
                        if alignment == "real":
                            for condition in "BC":
                                repaired = harmed = changed = eligible = 0
                                for position, truth in enumerate(reference):
                                    initial = result["trace"][position][condition][position]
                                    final = result["final"][condition][position]
                                    if position < len(reference) - 1:
                                        eligible += 1
                                        changed += initial != final
                                        repaired += initial != truth and final == truth
                                        harmed += initial == truth and final != truth
                                repair_rows.append(dict(dataset=dataset, seed=seed, branch=branch, condition=condition,
                                    group=name, subject=group["subject"], block=group["block"],
                                    eligible=eligible, changed=changed, repaired=repaired, harmed=harmed,
                                    repair_rate=repaired / eligible if eligible else 0.,
                                    harm_rate=harmed / eligible if eligible else 0.))
            for name, group in groups.items():
                row = dict(dataset=dataset, seed=seed, branch="open", alignment="pure_language",
                    condition="C", group=name, subject=group["subject"], material=group["material"],
                    block=group["block"], scope=group["scope"], words=len(group["reference"]),
                    **metric_row(group["reference"], pure_by_length[len(group["reference"])], info["language"] == "zh"))
                all_rows.append(row)
        records[dataset] = seed_outcomes
        # All three primary comparisons are fixed before decoding; six tests
        # across the two datasets share one Holm family.
        lookup = {(r["seed"], r["group"], r["branch"], r["alignment"], r["condition"]): r
                  for r in all_rows if r["dataset"] == dataset}
        for comparison in cfg["statistics"]["primary_comparisons"]:
            differences = []
            for seed in cfg["protocol"]["seeds"]:
                for name, group in groups.items():
                    real = lookup[(seed, name, "open", "real", "C")]
                    if comparison == "C_minus_B":
                        controls = [lookup[(seed, name, "open", "real", "B")]]
                    elif comparison == "C_minus_shifted_C":
                        controls = [lookup[(seed, name, "open", f"shift-{i:02d}", "C")]
                                    for i in range(cfg["controls"]["shifted_repetitions"])]
                    else:
                        controls = [lookup[(seed, name, "open", "pure_language", "C")]]
                    differences.append(dict(seed=seed, subject=group["subject"], block=group["block"],
                        scope=group["scope"], difference=real["sentence_bleu1_mean"] -
                        float(np.mean([c["sentence_bleu1_mean"] for c in controls]))))
            comparisons.append(dict(dataset=dataset, comparison=comparison,
                metric="subject_equal_sentence_bleu1_then_seed_equal",
                **paired_block_statistics(differences, "difference", cfg["statistics"])))
    adjusted = holm([r["p_one_sided"] for r in comparisons])
    for row, value in zip(comparisons, adjusted):
        row.update(p_holm=float(value), significant_holm_05=bool(value < .05))

    by_subject = defaultdict(list)
    for row in all_rows:
        key = tuple(row[k] for k in ("dataset", "seed", "branch", "alignment", "condition", "subject"))
        by_subject[key].append(row)
    subject_rows = []
    for key, members in by_subject.items():
        record = dict(zip(("dataset", "seed", "branch", "alignment", "condition", "subject"), key))
        record.update(groups=len(members), words=sum(r["words"] for r in members))
        for metric in METRICS:
            if metric == "sentence_bleu1_mean":
                record[metric] = float(np.mean([r[metric] for r in members]))
            else:
                record[metric] = float(np.average([r[metric] for r in members], weights=[r["words"] for r in members]))
        subject_rows.append(record)
    by_condition = defaultdict(list)
    for row in subject_rows:
        key = tuple(row[k] for k in ("dataset", "branch", "alignment", "condition"))
        by_condition[key].append(row)
    summary = []
    for key, members in by_condition.items():
        record = dict(zip(("dataset", "branch", "alignment", "condition"), key))
        record.update(seeds=len({r["seed"] for r in members}), subjects=len({r["subject"] for r in members}))
        for metric in METRICS:
            seed_values = [np.mean([r[metric] for r in members if r["seed"] == seed])
                           for seed in cfg["protocol"]["seeds"]]
            record[metric + "_mean"] = float(np.mean(seed_values))
            record[metric + "_seed_sd"] = float(np.std(seed_values, ddof=1))
        summary.append(record)
    exports = {}
    prefix = PROJECT_ROOT / "data/results/exports/unified_sentence_v2_"
    for name, rows in (("by_group", all_rows), ("by_subject", subject_rows),
                       ("summary", summary), ("statistics", comparisons), ("repairs", repair_rows)):
        path = Path(str(prefix) + name + ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        exports[name] = dict(path=path.relative_to(PROJECT_ROOT).as_posix(), sha256=sha256(path), rows=len(rows))
    save(Path(str(prefix) + "sources.json"), dict(manifest_sha256=manifest["manifest_sha256"],
        evidence="development_validation", test_evaluated=False, sources=provenance, exports=exports,
        metrics_contract="Mean sentence BLEU1 within subject, then equal subjects and seeds; WER, corpus BLEU1 and aligned accuracy pool word counts within subject before subject/seed averaging; all outputs have known reference word counts"))
    report = PROJECT_ROOT / "data/results/studies/unified_sentence_v2/results.json"
    save(report, dict(status="completed", manifest_sha256=manifest["manifest_sha256"],
        calibration={d: read(OUT / d / "calibration.json") for d in manifest["datasets"]},
        summary=summary, primary_statistics=comparisons, uniform_random_auxiliary=random_baselines,
        test_evaluated=False))
    return str(report.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
