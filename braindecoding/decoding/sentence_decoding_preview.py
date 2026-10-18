"""A fixed, small real-brain A/B/C preview, with the census deferred.

The parent scientific decoder is unchanged. Sample complete stimulus
components before decoding, retain every subject/voice, and calibrate on
disjoint components. This is descriptive development evidence, not the
complete experiment or its six-test statistical family.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from braindecoding.config import PROJECT_ROOT
from braindecoding.unified_sentence_v2 import (
    OUT, RUNTIME, cases_for_groups, digest, encoded_path, load_lm,
    manifest, read, save, score_mean, sha256,
)
from braindecoding.unified_sentence_batch import replay_many
from braindecoding.unified_sentence_v2_results import METRICS, metric_row
from braindecoding.unified_sentence_v2_artifacts import csv_write


PLAN = PROJECT_ROOT / "data/manifests/rolling_sentence/unified_sentence_v2_preview.json"
PREVIEW_OUT = OUT.parent / "unified_sentence_v2_preview"
PREVIEW_RUNTIME = RUNTIME.parent / "unified_sentence_v2_preview"
REPORT = PROJECT_ROOT / "data/results/studies/unified_sentence_v2_preview/results.json"
SEED = 0
SAMPLING_SEED = 20260928
COUNTS = {"calibration": 5, "evaluation": 6}


def select_groups(info, dataset, partition, count):
    """Round-robin strata, distinct blocks, one complete component per block.

    Selection uses identifiers and original material order only, never
    reference words, decoder predictions, length, or any measured outcome.
    """
    eligible = [g for g in info["groups"] if g["partition"] == partition]
    by_block, by_scope = defaultdict(list), defaultdict(list)
    for group in eligible:
        by_block[group["block"]].append(group)
    for block, groups in by_block.items():
        by_scope[groups[0]["scope"]].append(block)
    if count < len(by_scope) or count > len(by_block):
        raise ValueError("Preview size must cover all strata with distinct blocks")
    rng = np.random.default_rng(int(digest([SAMPLING_SEED, "main_preview", dataset, partition])[:12], 16))
    queues = {scope: list(rng.permutation(sorted(blocks))) for scope, blocks in sorted(by_scope.items())}
    blocks = []
    while len(blocks) < count:
        for scope in queues:
            if queues[scope] and len(blocks) < count:
                blocks.append(queues[scope].pop())
    materials = []
    for block in blocks:
        choices = sorted({g["material"] for g in by_block[block]})
        materials.append(choices[int(rng.integers(len(choices)))])
    order = {material: i for i, material in enumerate(materials)}
    selected = sorted([g for g in eligible if g["material"] in order],
                      key=lambda g: (order[g["material"]], g["subject"], g["group"]))
    expected_subjects = {g["subject"] for g in eligible}
    if {g["subject"] for g in selected} != expected_subjects:
        raise ValueError("Preview failed to cover every configured subject")
    return selected


def prepare():
    parent = manifest()
    datasets = {}
    for dataset, info in parent["datasets"].items():
        selected = {part: select_groups(info, dataset, part, n) for part, n in COUNTS.items()}
        if {g["material"] for g in selected["calibration"]} & {g["material"] for g in selected["evaluation"]}:
            raise ValueError("Preview calibration/evaluation material overlap")
        datasets[dataset] = dict(
            language=info["language"],
            groups=selected,
            counts={part: dict(segments=len(groups), words=sum(len(g["event_ids"]) for g in groups),
                               subjects=len({g["subject"] for g in groups}),
                               materials=len({g["material"] for g in groups}),
                               blocks=len({g["block"] for g in groups}),
                               scopes=sorted({g["scope"] for g in groups}))
                    for part, groups in selected.items()},
        )
    payload = dict(schema_version=1, protocol="unified_sentence_v2_preview",
        parent_manifest_sha256=parent["manifest_sha256"], evidence="descriptive_development_preview",
        seeds=[SEED], sampling_seed=SAMPLING_SEED, datasets=datasets,
        decoder="unchanged_frozen_parent_decoder", branch="open", alignment="real", conditions=list("ABC"),
        calibration=dict(condition="A", branch="open", seeds=[SEED],
            language_weight_grid=parent["settings"]["calibration"]["language_weight_grid"],
            metric=parent["settings"]["calibration"]["metric"], tie_break="smaller_weight",
            scope="five_complete_material_components_from_distinct_calibration_blocks"),
        selection_reads_outcomes=False, complete_components_all_subjects_and_voices=True,
        full_experiment_deferred_until_user_decision=True, significance_claim_allowed=False,
        chance_controls_deferred=True, test_evaluated=False,
        sources={Path(__file__).relative_to(PROJECT_ROOT).as_posix(): sha256(Path(__file__))})
    payload["preview_sha256"] = digest(payload)
    if PLAN.exists() and read(PLAN) != payload:
        raise ValueError("Preview was already frozen with a different population or implementation")
    save(PLAN, payload)
    print(json.dumps(dict(stage="preview_frozen", preview_sha256=payload["preview_sha256"],
                         datasets={d: i["counts"] for d, i in datasets.items()})), flush=True)
    return payload


def plan():
    frozen = read(PLAN)
    parent = manifest()
    if frozen["parent_manifest_sha256"] != parent["manifest_sha256"]:
        raise ValueError("Preview parent manifest changed")
    if digest({k: v for k, v in frozen.items() if k != "preview_sha256"}) != frozen["preview_sha256"]:
        raise ValueError("Preview content hash changed")
    for path, checksum in frozen["sources"].items():
        if sha256(PROJECT_ROOT / path) != checksum:
            raise ValueError(f"Frozen preview implementation changed: {path}")
    return frozen, parent


def progress(**fields):
    path = PREVIEW_RUNTIME / "status.json"
    prior = read(path) if path.exists() else {}
    prior.update(pid=os.getpid(), updated_unix=time.time(), test_evaluated=False, **fields)
    save(path, prior)


def existing_parent_records(parent, dataset, groups, *, calibration_weight=None, evaluation_weight=None):
    """Import only compatible completed parent outputs, never partial files."""
    if calibration_weight is not None:
        pattern = f"calibration_chunks/lambda-{calibration_weight:g}_seed-{SEED:03d}_chunk-*.json"
        expected_label = f"calibration_A_lambda-{calibration_weight:g}_seed-{SEED}"
    else:
        calibration = OUT / dataset / "calibration.json"
        if not calibration.exists() or read(calibration)["selected_weight"] != evaluation_weight:
            return {}, []
        pattern = f"seed-{SEED:03d}/open/real/chunk-*.json"
        expected_label = f"evaluation_open_real_seed-{SEED}"
    wanted = {g["group"] for g in groups}
    records, sources = {}, []
    for path in sorted((OUT / dataset).glob(pattern)):
        chunk = read(path)
        if chunk["manifest_sha256"] != parent["manifest_sha256"] or chunk["label"] != expected_label or chunk["test_evaluated"]:
            raise ValueError("Parent chunk provenance mismatch")
        relevant = [r for r in chunk["records"] if r["group"] in wanted]
        if relevant:
            sources.append(dict(path=path.relative_to(PROJECT_ROOT).as_posix(), sha256=sha256(path)))
            for record in relevant:
                if record["group"] in records:
                    raise ValueError("Duplicate parent group")
                records[record["group"]] = record
    return records, sources


def decode_batch(frozen, parent, dataset, scorer, embed, bank, groups, predictions, lookup,
                 weight, path, label, conditions):
    if path.exists():
        chunk = read(path)
        if chunk["preview_sha256"] != frozen["preview_sha256"] or chunk["label"] != label or chunk["weight"] != weight:
            raise ValueError("Preview chunk provenance mismatch")
        if {r["group"] for r in chunk["records"]} != {g["group"] for g in groups}:
            raise ValueError("Preview chunk population mismatch")
        return chunk["records"]
    start = time.perf_counter()
    last = [0.]

    def callback(arrival, counts):
        now = time.perf_counter()
        if now - last[0] >= 30:
            progress(state="running", stage="calibration" if conditions == "A" else "main_decode",
                     dataset=dataset, seed=SEED, label=label, arrival=arrival,
                     longest_segment=max(counts), chunk_seconds=now-start)
            last[0] = now

    cases = cases_for_groups(groups, predictions, lookup, weight=weight, branch="open")
    ds = parent["settings"]["decoding"]
    results = replay_many(cases, parent["datasets"][dataset]["vocabulary"], bank, scorer, embed,
        conditions=conditions, progress=callback, candidate_count=ds["candidates"],
        beam_width=ds["beam_width"], neural_scale=ds["neural_scale"])
    records = [dict(group=g["group"], result=r) for g, r in zip(groups, results)]
    save(path, dict(preview_sha256=frozen["preview_sha256"], parent_manifest_sha256=parent["manifest_sha256"],
        label=label, weight=weight, records=records, seconds=time.perf_counter()-start, test_evaluated=False))
    print(json.dumps(dict(stage="preview_chunk", dataset=dataset, label=label,
                         segments=len(groups), seconds=time.perf_counter()-start)), flush=True)
    return records


def run_dataset(frozen, parent, dataset):
    groups = frozen["datasets"][dataset]["groups"]
    root = PREVIEW_OUT / dataset
    main_path = root / "main.json"
    if main_path.exists():
        if read(main_path)["preview_sha256"] != frozen["preview_sha256"]:
            raise ValueError("Completed preview provenance mismatch")
        return
    progress(state="running", stage="loading_models", dataset=dataset, seed=SEED,
             preview_sha256=frozen["preview_sha256"])
    scorer, embed, bank = load_lm(parent, dataset)
    cache_path = root / "lm_cache.pkl"
    scorer.load_cache(cache_path if cache_path.exists() else OUT / dataset / "lm_cache.pkl")
    archive_path = encoded_path(dataset, SEED)
    sidecar = read(archive_path.with_suffix(".json"))
    if sidecar["manifest_sha256"] != parent["manifest_sha256"] or sidecar["prediction_sha256"] != sha256(archive_path):
        raise ValueError("Neural prediction cache provenance mismatch")
    with np.load(archive_path) as archive:
        predictions = {d: archive[f"k{d}"] for d in range(3)}
        lookup = {str(e): i for i, e in enumerate(archive["event_ids"])}
    width = parent["settings"]["decoding"]["segment_batch_size"]
    calibration_path = root / "calibration.json"
    if calibration_path.exists():
        calibration = read(calibration_path)
        if calibration["preview_sha256"] != frozen["preview_sha256"]:
            raise ValueError("Preview calibration provenance mismatch")
    else:
        candidates, sources = [], []
        for weight in frozen["calibration"]["language_weight_grid"]:
            cached, reused = existing_parent_records(parent, dataset, groups["calibration"], calibration_weight=weight)
            sources.extend(reused)
            remaining = [g for g in groups["calibration"] if g["group"] not in cached]
            records = list(cached.values())
            for start in range(0, len(remaining), width):
                batch = remaining[start:start+width]
                records.extend(decode_batch(frozen, parent, dataset, scorer, embed, bank, batch,
                    predictions, lookup, weight,
                    root / "calibration_chunks" / f"lambda-{weight:g}_chunk-{start//width:04d}.json",
                    f"preview_calibration_A_lambda-{weight:g}", "A"))
                scorer.save_cache(cache_path)
            if {r["group"] for r in records} != {g["group"] for g in groups["calibration"]}:
                raise ValueError("Preview calibration population incomplete")
            candidates.append(dict(weight=weight, score=score_mean(records, groups["calibration"])))
        selected = sorted(candidates, key=lambda row: (-row["score"], row["weight"]))[0]["weight"]
        calibration = dict(preview_sha256=frozen["preview_sha256"], selected_weight=selected,
                           seed=SEED, condition="A", grid=candidates, reused_parent_sources=sources)
        save(calibration_path, calibration)
    weight = calibration["selected_weight"]
    print(json.dumps(dict(stage="preview_calibrated", dataset=dataset, weight=weight)), flush=True)
    cached, sources = existing_parent_records(parent, dataset, groups["evaluation"], evaluation_weight=weight)
    remaining = [g for g in groups["evaluation"] if g["group"] not in cached]
    records = list(cached.values())
    for start in range(0, len(remaining), width):
        batch = remaining[start:start+width]
        records.extend(decode_batch(frozen, parent, dataset, scorer, embed, bank, batch,
            predictions, lookup, weight, root / "main_chunks" / f"chunk-{start//width:04d}.json",
            "preview_open_real_ABC_seed-0", "ABC"))
        scorer.save_cache(cache_path)
    if len(records) != len(groups["evaluation"]) or {r["group"] for r in records} != {g["group"] for g in groups["evaluation"]}:
        raise ValueError("Preview evaluation population incomplete")
    save(main_path, dict(status="completed", preview_sha256=frozen["preview_sha256"],
        parent_manifest_sha256=parent["manifest_sha256"], selected_weight=weight, seed=SEED,
        records=records, reused_parent_sources=sources, neural_source=dict(path=str(archive_path.relative_to(PROJECT_ROOT)),
        sha256=sidecar["prediction_sha256"]), test_evaluated=False))
    del scorer, embed, bank, predictions
    gc.collect()
    torch.cuda.empty_cache()


def dataset_rows(frozen, dataset, decoded):
    expected = {g["group"]: g for g in frozen["datasets"][dataset]["groups"]["evaluation"]}
    records = decoded["records"]
    if len(records) != len(expected) or {r["group"] for r in records} != set(expected):
        raise ValueError("Preview report population incomplete or duplicated")
    if decoded["preview_sha256"] != frozen["preview_sha256"] or decoded["test_evaluated"]:
        raise ValueError("Preview report provenance mismatch")
    rows, repairs, examples = [], [], []
    chinese = frozen["datasets"][dataset]["language"] == "zh"
    for record in records:
        group = expected[record["group"]]
        reference, result = group["reference"], record["result"]
        n = len(reference)
        if len(result["trace"]) != n or any(len(result["final"][c]) != n for c in "ABC"):
            raise ValueError("Preview changed known word count or omitted arrivals")
        for c in "ABC":
            rows.append(dict(dataset=dataset, seed=SEED, subject=group["subject"], group=group["group"],
                material=group["material"], block=group["block"], scope=group["scope"], condition=c, words=n,
                **metric_row(reference, result["final"][c], chinese)))
        record_example = dict(group=group["group"], subject=group["subject"], reference=reference,
                              final=result["final"], trace=result["trace"])
        for c in "BC":
            initial = [result["trace"][pos][c][pos] for pos in range(n)]
            fixed = [i for i in range(n-1) if initial[i] != reference[i] and result["final"][c][i] == reference[i]]
            broken = [i for i in range(n-1) if initial[i] == reference[i] and result["final"][c][i] != reference[i]]
            changed = sum(initial[i] != result["final"][c][i] for i in range(n-1))
            repairs.append(dict(dataset=dataset, seed=SEED, subject=group["subject"], group=group["group"],
                condition=c, eligible=n-1, changed=changed, repaired=len(fixed), harmed=len(broken)))
            record_example[f"initial_{c}"] = initial
            record_example[f"repaired_{c}"] = fixed
            record_example[f"harmed_{c}"] = broken
        examples.append(record_example)
    return rows, repairs, examples


def export(frozen, *, require_all=False):
    all_rows, all_repairs, datasets, provenance = [], [], {}, []
    for dataset, info in frozen["datasets"].items():
        path = PREVIEW_OUT / dataset / "main.json"
        if not path.exists():
            if require_all:
                raise ValueError("Both datasets must finish the fixed preview")
            continue
        decoded = read(path)
        rows, repairs, examples = dataset_rows(frozen, dataset, decoded)
        all_rows.extend(rows)
        all_repairs.extend(repairs)
        by_subject = []
        for subject in sorted({r["subject"] for r in rows}):
            for condition in "ABC":
                selected = [r for r in rows if r["subject"] == subject and r["condition"] == condition]
                row = dict(dataset=dataset, subject=subject, condition=condition, seed=SEED,
                           segments=len(selected), words=sum(r["words"] for r in selected))
                for metric in METRICS:
                    row[metric] = float(np.mean([r[metric] for r in selected])) if metric == "sentence_bleu1_mean" else \
                        float(np.average([r[metric] for r in selected], weights=[r["words"] for r in selected]))
                by_subject.append(row)
        summary = []
        for condition in "ABC":
            selected = [r for r in by_subject if r["condition"] == condition]
            summary.append(dict(dataset=dataset, condition=condition,
                **{metric: float(np.mean([r[metric] for r in selected])) for metric in METRICS}))
        get = {r["condition"]: r for r in summary}
        repair_summary = []
        for condition in "BC":
            subject_rates = []
            for subject in sorted({r["subject"] for r in repairs}):
                selected = [r for r in repairs if r["subject"] == subject and r["condition"] == condition]
                eligible = sum(r["eligible"] for r in selected)
                subject_rates.append({k: sum(r[k] for r in selected)/eligible if eligible else 0.
                                      for k in ("repaired", "harmed", "changed")})
            repair_summary.append(dict(condition=condition,
                **{f"{k}_rate": float(np.mean([r[k] for r in subject_rates])) for k in ("repaired", "harmed", "changed")},
                eligible=sum(r["eligible"] for r in repairs if r["condition"] == condition),
                repaired=sum(r["repaired"] for r in repairs if r["condition"] == condition),
                harmed=sum(r["harmed"] for r in repairs if r["condition"] == condition)))
        first = sorted(examples, key=lambda r: r["group"])[0]
        fixed = sorted(examples, key=lambda r: (-len(r["repaired_C"]), len(r["harmed_C"]), r["group"]))[0]
        broken = sorted(examples, key=lambda r: (-len(r["harmed_C"]), len(r["repaired_C"]), r["group"]))[0]
        datasets[dataset] = dict(scope=info["counts"], selected_weight=decoded["selected_weight"],
            calibration=read(PREVIEW_OUT / dataset / "calibration.json"), summary=summary,
            C_minus_B_bleu1=get["C"]["sentence_bleu1_mean"]-get["B"]["sentence_bleu1_mean"],
            B_minus_A_bleu1=get["B"]["sentence_bleu1_mean"]-get["A"]["sentence_bleu1_mean"],
            C_minus_B_aligned_accuracy=get["C"]["aligned_word_accuracy"]-get["B"]["aligned_word_accuracy"],
            by_subject=by_subject, repairs=repair_summary,
            examples=[dict(selection="first_group", **first), dict(selection="most_C_repairs", **fixed),
                      dict(selection="most_C_harms", **broken)], examples_selection_for_illustration_only=True)
        provenance.append(dict(path=path.relative_to(PROJECT_ROOT).as_posix(), sha256=sha256(path)))
    if not datasets:
        return None
    output = dict(status="completed_preview" if len(datasets) == len(frozen["datasets"]) else "partial_preview",
        preview_sha256=frozen["preview_sha256"], parent_manifest_sha256=frozen["parent_manifest_sha256"],
        evidence=frozen["evidence"], seeds=frozen["seeds"], branch="open", alignment="real",
        datasets=datasets, full_experiment_completed=False, full_experiment_deferred_until_user_decision=True,
        chance_controls_completed=False, significance_claim_allowed=False, test_evaluated=False,
        metrics_contract="Mean sentence BLEU1 within subject then equal subjects; WER, corpus BLEU1 and aligned accuracy pool words within subject before equal subjects; single seed 0.",
        sources=provenance)
    save(REPORT, output)
    prefix = PROJECT_ROOT / "data/results/exports/unified_sentence_v2_preview_"
    csv_write(Path(str(prefix)+"by_group.csv"), all_rows)
    csv_write(Path(str(prefix)+"repairs.csv"), all_repairs)
    csv_write(Path(str(prefix)+"summary.csv"), [r for d in datasets.values() for r in d["summary"]])
    csv_write(Path(str(prefix)+"by_subject.csv"), [r for d in datasets.values() for r in d["by_subject"]])
    figures(output)
    markdown(output)
    return output


def figures(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    destination = PROJECT_ROOT / "data/results/figures/unified_sentence_v2_preview"
    destination.mkdir(parents=True, exist_ok=True)
    datasets = list(report["datasets"])
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 3.8), squeeze=False)
    names = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}
    for ax, dataset in zip(axes[0], datasets):
        info = report["datasets"][dataset]
        values = [r["sentence_bleu1_mean"]*100 for r in info["summary"]]
        ax.bar(range(3), values, color=["#8C9AA5", "#DA9F5E", "#2C8190"], width=.6)
        ax.set_xticks(range(3), ["A: fixed", "B: re-search", "C: refreshed"])
        ax.set_ylabel("Mean sentence BLEU-1 (%)")
        ax.set_ylim(0, max(max(values)*1.35, 1))
        for i,value in enumerate(values):
            ax.text(i, value+max(values)*.035, f"{value:.2f}", ha="center")
        scope = info["scope"]["evaluation"]
        ax.set_title(f"{names.get(dataset,dataset)}\n{scope['subjects']} subjects, {scope['materials']} materials, seed 0")
        ax.text(.5, .95, f"C - B = {info['C_minus_B_bleu1']*100:+.2f} percentage points",
                transform=ax.transAxes, ha="center", va="top", fontsize=9)
    fig.suptitle("Fixed material preview | open vocabulary | descriptive results", fontsize=11)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination / f"main_preview.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown(report):
    names = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}
    lines = ["# 统一句子解码：快速主实验预览", "",
        "固定 seed 0、开放补词、真实对齐脑信号。抽样规则在解码前冻结；每个材料单元保留全部受试者和朗读版本。",
        "每个数据集在五个独立校准材料单元上用 A 选择语言权重，在六个评价材料单元上比较 A/B/C。",
        "A：输出后固定；B：最近三词重新搜索，沿用 K0；C：最近三词重新搜索，使用当前可用 K1/K2。", "",
        "这是单种子、材料子集的描述性验证。尚未完成错位和纯语言对照，不能据此宣称显著超过机会基线；全量实验等待用户决定。", "",
        "| 数据集 | 受试者 | 片段 | 词位 | A BLEU-1 | B BLEU-1 | C BLEU-1 | C−B（百分点） |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset, info in report["datasets"].items():
        scope = info["scope"]["evaluation"]
        values = {r["condition"]:r["sentence_bleu1_mean"]*100 for r in info["summary"]}
        lines.append(f"| {names.get(dataset,dataset)} | {scope['subjects']} | {scope['segments']} | {scope['words']} | {values['A']:.2f}% | {values['B']:.2f}% | {values['C']:.2f}% | {info['C_minus_B_bleu1']*100:+.2f} |")
    lines.extend(["", "主指标先在每名受试者内按片段平均，再对受试者等权平均。BLEU-1 统计词重合，不代表整句完全正确。", "",
        "![主实验预览](../../figures/unified_sentence_v2_preview/main_preview.png)", ""])
    for dataset, info in report["datasets"].items():
        lines.extend([f"## {names.get(dataset,dataset)}", "", f"固定校准子集所选语言权重：{info['selected_weight']:g}。", "",
            "| 条件 | 可修正词位 | 改对 | 改错 | 改对率（受试者等权） | 改错率（受试者等权） |",
            "|---|---:|---:|---:|---:|---:|"])
        for row in info["repairs"]:
            lines.append(f"| {row['condition']} | {row['eligible']} | {row['repaired']} | {row['harmed']} | {row['repaired_rate']*100:.2f}% | {row['harmed_rate']*100:.2f}% |")
        lines.extend(["", "以下例子按预先固定的展示规则选取，不能代替总体统计；改对、改错位置按条件自身的首次输出与末态比较。", ""])
        for example in info["examples"]:
            sep = " " if dataset == "pallier2025" else " / "
            lines.extend([f"### {example['selection']} · {example['subject']}", "", f"材料：{example['group']}", "",
                          f"参考：{sep.join(example['reference'])}", ""])
            for c in "ABC":
                lines.extend([f"{c}：{sep.join(example['final'][c])}", ""])
            lines.extend([f"C 改对位置（从 1 开始）：{[i+1 for i in example['repaired_C']]}；改错位置：{[i+1 for i in example['harmed_C']]}。", ""])
    (REPORT.parent / "results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run():
    frozen, parent = plan()
    # The old census must be deferred, not competing for GPU memory/time.
    import psutil
    old = read(RUNTIME / "status.json")
    try:
        p = psutil.Process(old["pid"])
        if "braindecoding.unified_sentence_v2_complete" in " ".join(p.cmdline()):
            raise RuntimeError("Stop the deferred census worker before running the preview")
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    if not (RUNTIME / "deferred_by_user.json").exists():
        raise ValueError("Full run must be explicitly deferred by the user first")
    torch.set_num_threads(4)
    PREVIEW_RUNTIME.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (PREVIEW_RUNTIME / "experiment.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        started = time.time()
        try:
            progress(state="running", stage="starting", started_unix=started,
                     preview_sha256=frozen["preview_sha256"])
            for dataset in frozen["datasets"]:
                run_dataset(frozen, parent, dataset)
                result = export(frozen)
                print(json.dumps(dict(stage="preview_dataset_ready", dataset=dataset,
                                     result=str(REPORT.relative_to(PROJECT_ROOT)))), flush=True)
            result = export(frozen, require_all=True)
            progress(state="completed", stage="completed_preview", completed_unix=time.time(),
                     seconds=time.time()-started, full_experiment_deferred=True)
        except BaseException as error:
            progress(state="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    from braindecoding.config import ensure_default_environment
    ensure_default_environment()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.run:
        run()
    else:
        frozen, _ = plan()
        export(frozen, require_all=True)


if __name__ == "__main__":
    main()
