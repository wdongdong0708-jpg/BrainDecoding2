"""CPU-only full-training-lexicon retrieval for saved local-r2/full-context tests."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from braindecoding.config import PROJECT_ROOT as ROOT
from braindecoding.data.text import load_text_embedding_cache
from braindecoding.evaluation.retrieval import mean_by_target, retrieval_ranks, summarize_retrieval
from braindecoding.unified_sentence_v2 import digest, read, save, sha256
from braindecoding.unified_sentence_v2_artifacts import csv_write
from braindecoding.word_paired_statistics_test_v2 import metadata, weighted_scores, bootstrap_weights


ID = "full_vocabulary_context_test_v2"
PROTOCOL = ROOT / "data/manifests/rolling_sentence/unified_protocol_v2.md"
MANIFEST = ROOT / f"data/manifests/rolling_sentence/{ID}.json"
PARENT = ROOT / "data/manifests/rolling_sentence/heldout_test_v2.json"
CONTEXT = PROTOCOL.with_name("context_heldout_test_v2.json")
OUT = ROOT / "outputs/heldout_test/unified_context_heldout_test_v2"
REPORT = ROOT / f"data/results/studies/{ID}"
EXPORT = ROOT / "data/results/exports"
RUNTIME = ROOT / f".cache/runs/{ID}"
METRICS = ("macro_recall_at_10", "macro_recall_at_1", "micro_recall_at_10", "micro_recall_at_1")
CONDITIONS = ("full_context", "local_r2")
POPULATIONS = ("all_words", "common_words", "strict50_queries", "strict50_common_queries")
NAMES = {"pallier2025": "Pallier2025", "chineseeeg2_littleprince": "ChineseEEG-2"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_identity(m):
    require(digest({k: v for k, v in m.items() if k != "manifest_sha256"}) == m["manifest_sha256"],
            "Manifest identity changed")


def key(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def check_sources(m):
    for path, expected in m["source_sha256"].items():
        if str(path).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        require(sha256(ROOT / path) == expected, f"Frozen source changed: {path}")


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        verify_identity(m)
        check_sources(m)
        return m
    parent, context = read(PARENT), read(CONTEXT)
    verify_identity(parent)
    verify_identity(context)
    require(context["test_material_manifest_sha256"] == parent["manifest_sha256"], "Context test population differs")
    sources, jobs = {}, []
    for path in (PROTOCOL, Path(__file__), PARENT, CONTEXT,
                 ROOT / "braindecoding/evaluation/retrieval.py", ROOT / "braindecoding/evaluation/word_paired_statistics_test_v2.py",
                 ROOT / "braindecoding/data/text.py", ROOT / "braindecoding/decoding/unified_sentence_v2.py",
                 EXPORT / "unified_context_heldout_test_v2_context_by_seed.csv"):
        sources[key(path)] = sha256(path)
    for dataset, info in parent["datasets"].items():
        require(len(info["vocabulary"]) == (2020 if dataset == "pallier2025" else 2180), "Full train lexicon changed")
        for path in (Path(info["embedding_cache"]), Path(info["embedding_cache"]).with_suffix(".json")):
            expected = parent["source_sha256"].get(str(path.resolve()))
            require(expected is not None and sha256(path) == expected, "Frozen training embeddings changed")
            sources[key(path)] = expected
        for seed in (0, 1, 2):
            for condition in CONDITIONS:
                job = next(j for j in context["jobs"] if (j["dataset"], j["seed"], j["condition"]) == (dataset, seed, condition))
                path = OUT / dataset / f"seed-{seed:03d}" / condition / "metrics.json"
                side = read(path)
                prediction = path.with_name("predictions.npz")
                require(side["status"] == "completed" and side["split"] == "test" and side["test_evaluated"]
                        and side["manifest_sha256"] == context["manifest_sha256"]
                        and side["dataset"] == dataset and side["seed"] == seed and side["condition"] == condition
                        and side["checkpoint_sha256"] == job["checkpoint_sha256"]
                        and side["prediction_sha256"] == sha256(prediction), "Saved context prediction provenance changed")
                require(sha256(job["checkpoint"]) == job["checkpoint_sha256"], "Validation-selected checkpoint changed")
                for p in (path, prediction, Path(job["checkpoint"])):
                    sources[key(p)] = sha256(p)
                jobs.append(dict(dataset=dataset, seed=seed, condition=condition,
                                 prediction=key(prediction), metrics=key(path)))
    require(len(jobs) == 12, "Need all twelve local-r2/full-context prediction files")
    m = dict(protocol=ID, frozen_unix=time.time(), parent_manifest_sha256=parent["manifest_sha256"],
             context_manifest_sha256=context["manifest_sha256"], jobs=jobs, source_sha256=sources,
             seeds=[0, 1, 2], conditions=list(CONDITIONS), populations=list(POPULATIONS),
             random_seed=161803, repetitions=10000, statistics="post_results_paired_material_block_intervals_no_p_tests",
             candidates="full_frozen_training_vocabulary", oov_policy="retain_all_targets_as_errors",
             metric_contract="word_type_macro_and_event_micro_plus_separate_subject_equal_metrics",
             computation="CPU_saved_embeddings_only")
    m["manifest_sha256"] = digest(m)
    check_sources(m)
    save(MANIFEST, m)
    print("Frozen complete-vocabulary comparison before retrieval", flush=True)
    return m


def full_vocab_ranks(queries, words, bank, vocabulary):
    require(len(vocabulary) > 0 and len(set(vocabulary)) == len(vocabulary), "Invalid candidate vocabulary")
    queries, bank = np.asarray(queries, np.float32), np.asarray(bank, np.float32)
    words = np.asarray(words).astype(str)
    require(queries.ndim == bank.ndim == 2 and queries.shape[1] == bank.shape[1]
            and len(queries) == len(words) and len(bank) == len(vocabulary)
            and np.isfinite(queries).all() and np.isfinite(bank).all(), "Invalid retrieval vectors")
    covered = np.isin(words, vocabulary)
    ranks = np.full(len(words), np.inf)
    if covered.any():
        ranks[covered] = retrieval_ranks(queries[covered], words[covered], bank, vocabulary)
    return ranks, covered


def population_masks(words, common, vocab50):
    selected = np.isin(words, vocab50)
    return dict(all_words=np.ones(len(words), bool), common_words=common,
                strict50_queries=selected, strict50_common_queries=selected & common)


def score_ranks(ranks, words, subjects, covered):
    require(len(ranks) == len(words) == len(subjects) == len(covered) and len(ranks) > 0, "Empty or mismatched queries")
    values = dict(query_count=len(words), target_count=len(set(words)), oov_queries=int((~covered).sum()),
                  in_vocabulary_rate=float(covered.mean()))
    for k in (1, 10):
        hit = ranks <= k
        values[f"macro_recall_at_{k}"] = mean_by_target(hit, words)
        values[f"micro_recall_at_{k}"] = float(hit.mean())
        values[f"subject_equal_recall_at_{k}"] = float(np.mean([hit[subjects == s].mean() for s in np.unique(subjects)]))
    return values


def load_verified(job, info, meta, bank, vocabulary):
    with np.load(ROOT / job["prediction"], allow_pickle=False) as archive:
        ids, words = archive["event_ids"].astype(str), archive["words"].astype(str)
        queries, common = archive["predictions"], archive["common_mask"]
    require(len(ids) == len(set(ids)) == len(meta) and set(ids) == set(meta), "Saved test event population differs")
    require(np.array_equal(words, [meta[e]["word"] for e in ids])
            and np.array_equal(common, [meta[e]["common"] for e in ids]), "Saved target or common-event alignment differs")
    require(queries.shape == (len(ids), 1024)
            and np.allclose(np.linalg.norm(queries, axis=1), 1., atol=.005), "Saved predictions are not normalized 1024-dimensional vectors")
    # Independently reproduce all four original 50-word metrics for both query masks.
    vindex = {w: i for i, w in enumerate(vocabulary)}
    bank50 = bank[[vindex[w] for w in info["vocabulary50"]]]
    side = read(ROOT / job["metrics"])
    for population, mask in (("all_targets", np.isin(words, info["vocabulary50"])),
                             ("common_targets", np.isin(words, info["vocabulary50"]) & common)):
        old = summarize_retrieval(retrieval_ranks(queries[mask], words[mask], bank50, info["vocabulary50"]),
                                  words[mask], 50, top_ks=(1, 10))
        for metric in ("query_count", "target_count", "candidate_count", *METRICS):
            require(np.isclose(old[metric], side["strict_vocab50"][population][metric], atol=1e-12, rtol=0),
                    f"Original strict-50 result not reproduced: {job}, {population}, {metric}")
    ranks, covered = full_vocab_ranks(queries, words, bank, vocabulary)
    order = np.argsort(ids)
    return ids[order], words[order], common[order], ranks[order], covered[order]


def paired_intervals(info, ids, words, ranks, meta, m, index):
    """Use one shared material draw for both conditions and all three neural seeds."""
    blocks = [b["block"] for b in info["blocks"]]
    bi = {b: i for i, b in enumerate(blocks)}
    types, wi = np.unique(words, return_inverse=True)
    bq = np.array([bi[meta[e]["block"]] for e in ids])
    counts = np.zeros((len(blocks), len(types)))
    np.add.at(counts, (bq, wi), 1)
    hits = np.zeros((2, 3, 2, *counts.shape))
    for condition in range(2):
        for seed in range(3):
            for cutoff_index, cutoff in enumerate((10, 1)):
                np.add.at(hits[condition, seed, cutoff_index], (bq, wi), ranks[condition, seed] <= cutoff)
    observed, _ = weighted_scores(np.ones((1, len(blocks))), counts, hits)
    weights = bootstrap_weights(info, m, index)
    differences, represented = [], []
    # Keep temporary arrays bounded even when the test contains hundreds of rare types.
    for start in range(0, len(weights), 128):
        values, covered_types = weighted_scores(weights[start:start+128], counts, hits)
        differences.append(values[1] - values[0])
        represented.extend(covered_types.tolist())
    differences = np.concatenate(differences, axis=1)
    lo, hi = np.quantile(differences, [.025, .975], axis=1)
    return observed[:, :, 0], dict(ci_lower=lo, ci_upper=hi, blocks=len(blocks),
        bootstrap_samples=len(weights), type_count_min=min(represented),
        type_count_mean=float(np.mean(represented)), type_count_full_samples=represented.count(len(types)))


def run(m):
    check_sources(m)
    parent = read(PARENT)
    rows, subjects_rows, summary, difference_rows = [], [], [], []
    index = 0
    for dataset, info in parent["datasets"].items():
        print(f"Retrieving saved {dataset} vectors against {len(info['vocabulary'])} training words", flush=True)
        meta = metadata(info)
        embeddings = load_text_embedding_cache(info["embedding_cache"])
        vocabulary = info["vocabulary"]
        bank = np.stack([embeddings[w] for w in vocabulary])
        all_ranks, canonical = [], None
        for condition in CONDITIONS:
            seed_ranks = []
            for seed in (0, 1, 2):
                job = next(j for j in m["jobs"] if (j["dataset"], j["seed"], j["condition"]) == (dataset, seed, condition))
                ids, words, common, ranks, covered = load_verified(job, info, meta, bank, vocabulary)
                if canonical is None:
                    canonical = (ids, words, common, covered)
                require(all(np.array_equal(a, b) for a, b in zip(canonical, (ids, words, common, covered))),
                        "Local-r2/full-context seeds use different queries")
                seed_ranks.append(ranks)
                subjects = np.array([meta[e]["subject"] for e in ids])
                for population, mask in population_masks(words, common, info["vocabulary50"]).items():
                    values = score_ranks(ranks[mask], words[mask], subjects[mask], covered[mask])
                    rows.append(dict(dataset=dataset, condition=condition, seed=seed, population=population,
                                     candidate_count=len(vocabulary), **values))
                    for subject in np.unique(subjects):
                        sm = mask & (subjects == subject)
                        subjects_rows.append(dict(dataset=dataset, condition=condition, seed=seed, population=population,
                                                  subject=subject, candidate_count=len(vocabulary),
                                                  **score_ranks(ranks[sm], words[sm], subjects[sm], covered[sm])))
                print(f"Verified/retrieved {dataset} {condition} seed {seed}", flush=True)
            all_ranks.append(seed_ranks)
        all_ranks = np.asarray(all_ranks)
        ids, words, common, covered = canonical
        legacy = {c: [] for c in CONDITIONS}
        for condition in CONDITIONS:
            for seed in (0, 1, 2):
                job = next(j for j in m["jobs"] if (j["dataset"], j["seed"], j["condition"]) == (dataset, seed, condition))
                legacy[condition].append(read(ROOT / job["metrics"])["strict_vocab50"])
        for population, mask in population_masks(words, common, info["vocabulary50"]).items():
            observed, ci = paired_intervals(info, ids[mask], words[mask], all_ranks[:, :, mask], meta, m, index)
            index += 1
            by_condition = {c: [r for r in rows if (r["dataset"], r["condition"], r["population"]) == (dataset, c, population)]
                            for c in CONDITIONS}
            for mi, metric in enumerate(METRICS):
                scores = np.array([[r[metric] for r in by_condition[c]] for c in CONDITIONS])
                require(np.allclose(scores.mean(1), observed[:, mi], rtol=0, atol=1e-12), "Material aggregation differs from full metrics")
                effects = scores[1] - scores[0]
                legacy_population = "common_targets" if "common" in population else "all_targets"
                original_scores = np.array([[r[legacy_population][metric] for r in legacy[c]] for c in CONDITIONS])
                original_effect = float((original_scores[1]-original_scores[0]).mean())
                summary.append(dict(dataset=dataset, population=population, metric=metric, candidate_count=len(vocabulary),
                    query_count=int(mask.sum()), target_count=len(set(words[mask])), oov_queries=int((~covered[mask]).sum()),
                    local_r2_mean=float(scores[1].mean()), local_r2_sd=float(scores[1].std(ddof=1)),
                    full_context_mean=float(scores[0].mean()), full_context_sd=float(scores[0].std(ddof=1)),
                    effect=float(effects.mean()), ci_lower=float(ci["ci_lower"][mi]), ci_upper=float(ci["ci_upper"][mi]),
                    positive_seeds=int((effects > 1e-12).sum()), negative_seeds=int((effects < -1e-12).sum()),
                    seeds=3, subjects=info["counts"]["subjects"], blocks=ci["blocks"], bootstrap_samples=ci["bootstrap_samples"],
                    strict50_reference_population=legacy_population,
                    strict50_local_r2_mean=float(original_scores[1].mean()),
                    strict50_full_context_mean=float(original_scores[0].mean()), strict50_effect=original_effect,
                    same_direction_as_strict50=bool(np.sign(effects.mean()) == np.sign(original_effect)),
                    bootstrap_type_count_min=ci["type_count_min"], bootstrap_type_count_mean=ci["type_count_mean"],
                    complete_type_bootstrap_samples=ci["type_count_full_samples"]))
                difference_rows.extend(dict(dataset=dataset, population=population, metric=metric, seed=seed,
                    local_r2=float(scores[1, seed]), full_context=float(scores[0, seed]), effect=float(effects[seed])) for seed in (0, 1, 2))
        del all_ranks, embeddings, bank
    exports = {}
    for suffix, values in (("by_seed", rows), ("by_subject_seed", subjects_rows), ("summary", summary),
                           ("difference_by_seed", difference_rows)):
        path = EXPORT / f"{ID}_{suffix}.csv"
        csv_write(path, values)
        exports[key(path)] = sha256(path)
    report = dict(status="verified_complete", protocol=ID, manifest_sha256=m["manifest_sha256"],
                  split="test", conditions=list(CONDITIONS), seeds=[0, 1, 2], populations=list(POPULATIONS),
                  jobs=12, summary=summary, by_seed=rows, difference_by_seed=difference_rows,
                  original_strict50_reproduced=True, frozen_inputs_unchanged=True,
                  oov_policy=m["oov_policy"], candidate_policy=m["candidates"],
                  supplementary_intervals=True, new_p_tests=False, exports=exports, completed_unix=time.time())
    check_sources(m)
    save(REPORT / "report.json", report)
    save(EXPORT / f"{ID}_summary.json", report)
    save(REPORT / "audit.json", dict(status="verified_complete", manifest_sha256=m["manifest_sha256"],
        report_sha256=sha256(REPORT / "report.json"), exports=exports, source_sha256=m["source_sha256"],
        original_strict50_reproduced=True, full_candidate_counts={d: len(i["vocabulary"]) for d, i in parent["datasets"].items()}))
    write_markdown(report, parent)
    print("Verified complete: all twelve model/seed retrievals and four paired populations", flush=True)
    return report


def write_markdown(report, parent):
    labels = dict(all_words="全部测试词", common_words="全部共同查询词",
                  strict50_queries="原50词全查询", strict50_common_queries="原50词共同查询")
    lines = ["# 完整训练词库检索对照：局部半径 2 与完整上下文", "",
             "两个数据集 × 三种子 × 两模型的既存测试预测已全部重算。候选严格为训练词库；词库外目标保留计错。", "",
             "Macro 指真实词类型等权；Micro 指查询事件等权；表中为三个种子的等权均值。", ""]
    for population in POPULATIONS:
        lines += [f"## {labels[population]}", "",
            "| 数据集 | 候选/查询/真实类型/OOV查询 | 指标 | 局部半径2 (%) | 完整上下文 (%) | 局部−完整（pp） | 逐项95%材料区间（pp） | 同方向种子 |",
            "|---|---|---|---:|---:|---:|---|---:|"]
        for r in report["summary"]:
            if r["population"] != population:
                continue
            metric = r["metric"].replace("macro_recall_at_", "Macro Top-").replace("micro_recall_at_", "Micro Top-")
            lines.append(f"| {NAMES[r['dataset']]} | {r['candidate_count']}/{r['query_count']}/{r['target_count']}/{r['oov_queries']} | {metric} | {r['local_r2_mean']*100:.2f} | {r['full_context_mean']*100:.2f} | {r['effect']*100:+.2f} | [{r['ci_lower']*100:+.2f}, {r['ci_upper']*100:+.2f}] | {r['positive_seeds']}/3 |")
    lines += ["", "## 完整词库的逐种子 Macro Top-10", "",
              "| 数据集 | 查询人口 | seed 0 局部−完整（pp） | seed 1 | seed 2 |", "|---|---|---:|---:|---:|"]
    for dataset in parent["datasets"]:
        for population in ("all_words", "strict50_queries"):
            effects = [r["effect"]*100 for r in report["difference_by_seed"]
                       if (r["dataset"], r["population"], r["metric"]) == (dataset, population, "macro_recall_at_10")]
            lines.append(f"| {NAMES[dataset]} | {labels[population]} | {effects[0]:+.2f} | {effects[1]:+.2f} | {effects[2]:+.2f} |")
    lines += ["", "## 与原50词评价的差异方向", "",
              "| 数据集 | 查询人口 | 指标 | 原50词局部−完整（pp） | 完整词库局部−完整（pp） | 方向一致 |",
              "|---|---|---|---:|---:|---|"]
    for r in report["summary"]:
        if r["population"] in ("all_words", "strict50_queries"):
            metric = r["metric"].replace("macro_recall_at_", "Macro Top-").replace("micro_recall_at_", "Micro Top-")
            lines.append(f"| {NAMES[r['dataset']]} | {labels[r['population']]} | {metric} | {r['strict50_effect']*100:+.2f} | {r['effect']*100:+.2f} | {'是' if r['same_direction_as_strict50'] else '否'} |")
    lines += ["", "候选向量、事件人口、共同查询掩码、目标词及模型来源均核验；原50词四项指标逐种子重算一致。",
              "95%区间是结果后的逐项配对材料块bootstrap区间，不属于新的确认性检验，未新增p值。共享刺激内的受试者及种子整体重采样；三个种子不当作独立材料。稀有真实类型在重采样中可能缺失，Macro按该样本出现的类型计算，完整类型覆盖计数见JSON/CSV。",
              "受试者等权Top-1/Top-10单独列于逐种子CSV；它与词类型Macro含义不同。", "",
              f"冻结清单SHA-256：`{report['manifest_sha256']}`。", ""]
    (REPORT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    # Limit CPU linear algebra while the priority B decoder keeps the GPU.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        run(freeze())


if __name__ == "__main__":
    main()
