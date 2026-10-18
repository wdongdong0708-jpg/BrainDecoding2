"""Post-results paired word statistics, without inference or model selection."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from braindecoding.evaluation.retrieval import retrieval_ranks, summarize_retrieval

from braindecoding.config import PROJECT_ROOT as ROOT
PROTOCOL = ROOT / 'data/manifests/rolling_sentence/unified_protocol_v2.md'
MANIFEST = PROTOCOL.with_suffix('.json')
PARENT = ROOT / 'data/manifests/rolling_sentence/heldout_test_v2.json'
CONTEXT = PARENT.with_name('context_heldout_test_v2.json')
EXPORT = ROOT / 'data/results/exports'
REPORT = ROOT / 'data/results/studies/word_paired_statistics_test_v2'
METRICS = ('macro_recall_at_10', 'macro_recall_at_1', 'micro_recall_at_10', 'micro_recall_at_1')
CONTRASTS = (
    dict(id='local_r2_minus_full', population='all_targets', left='full_context', right='local_r2'),
    dict(id='rolling_k2_minus_k0', population='common_targets', left='rolling_k0', right='rolling_k2'),
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    # Keep exactly the existing frozen-manifest identity contract.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_identity(m):
    if digest({k: v for k, v in m.items() if k != 'manifest_sha256'}) != m['manifest_sha256']:
        raise ValueError('Manifest identity changed')


def context_root(dataset, seed, condition):
    return ROOT / 'outputs/heldout_test/unified_context_heldout_test_v2' / dataset / f'seed-{seed:03d}' / condition


def rolling_root(dataset, seed):
    return ROOT / 'outputs/heldout_test/unified_heldout_test_v2' / dataset / f'seed-{seed:03d}'


def freeze():
    parent, context = read(PARENT), read(CONTEXT)
    for m in (parent, context):
        verify_identity(m)
    if context['test_material_manifest_sha256'] != parent['manifest_sha256']:
        raise ValueError('Material population differs')
    if MANIFEST.exists():
        m = read(MANIFEST)
        verify_identity(m)
        return m
    paths = [PROTOCOL, Path(__file__), PARENT, CONTEXT, ROOT / 'braindecoding/evaluation/retrieval.py',
             EXPORT / 'unified_heldout_test_v2_word_by_seed.csv',
             EXPORT / 'unified_context_heldout_test_v2_context_by_seed.csv',
             EXPORT / 'unified_heldout_test_v2_sentence_summary.json',
             EXPORT / 'full_context_sentence_test_v2_summary.json']
    for dataset, info in parent['datasets'].items():
        paths += [Path(info['evaluation_target_cache']), Path(info['evaluation_target_cache']).with_suffix('.json')]
        for seed in (0, 1, 2):
            r = rolling_root(dataset, seed)
            paths += [r / 'neural_predictions.npz', r / 'word_metrics.json']
            for condition in ('local_r2', 'full_context'):
                r = context_root(dataset, seed, condition)
                paths += [r / 'predictions.npz', r / 'metrics.json']
    m = dict(protocol='word_paired_statistics_test_v2', analysis_status='post_results_supplementary',
             training_allowed=False, inference_allowed=False, selection_allowed=False,
             frozen_unix=time.time(), parent_manifest_sha256=parent['manifest_sha256'],
             context_manifest_sha256=context['manifest_sha256'], contrasts=list(CONTRASTS),
             primary_metric='macro_recall_at_10', secondary_metrics=list(METRICS[1:]),
             seeds=[0, 1, 2], repetitions=10000, random_seed=20261005,
             interval='paired_stratified_material_block_percentile_95_pointwise',
             macro_missing_types='average_over_identical_present_types_in_both_conditions',
             complete_type_bootstrap='sensitivity_only', p_tests=False,
             uncertainty='conditional_on_observed_subjects_seeds_and_recording_chapter_strata',
             source_sha256={str(p.resolve()): sha(p) for p in paths})
    m['manifest_sha256'] = digest(m)
    save(MANIFEST, m)
    print('Frozen supplementary protocol before ranks/statistics', flush=True)
    return m


def check_sources(m):
    for path, expected in m['source_sha256'].items():
        if str(path).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        if sha(path) != expected:
            raise ValueError(f'Frozen source changed: {path}')


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def metadata(info):
    result = {}
    for group in info['groups']:
        for pos, (event, word) in enumerate(zip(group['event_ids'], group['reference'])):
            if event in result:
                raise ValueError('Duplicate frozen event')
            result[event] = dict(word=word, subject=group['subject'], material=group['material'],
                                 block=group['block'], scope=group['scope'], common=pos+2 < len(group['reference']))
    if len(result) != info['counts']['events']:
        raise ValueError('Event metadata coverage changed')
    return result


def ranks_for(dataset, seed, condition, population, info, parent, context, meta, byseed):
    rolling = condition.startswith('rolling_')
    if rolling:
        root = rolling_root(dataset, seed)
        side = read(root / 'word_metrics.json')
        h = int(condition[-1])
        values = side['strict_vocab50'][str(h)][population]
        prediction = root / 'neural_predictions.npz'
        expected_manifest = parent['manifest_sha256']
        expected_csv = next(r for r in byseed['rolling'] if (r['dataset'], int(r['seed']), int(r['horizon']), r['population'])
                            == (dataset, seed, h, population))
    else:
        root = context_root(dataset, seed, condition)
        side = read(root / 'metrics.json')
        values = side['strict_vocab50'][population]
        prediction = root / 'predictions.npz'
        expected_manifest = context['manifest_sha256']
        job = next(j for j in context['jobs'] if (j['dataset'], j['seed'], j['condition']) == (dataset, seed, condition))
        if side['checkpoint_sha256'] != job['checkpoint_sha256']:
            raise ValueError('Context checkpoint identity differs')
        expected_csv = next(r for r in byseed['context'] if (r['dataset'], int(r['seed']), r['condition'], r['population'])
                            == (dataset, seed, condition, population))
    if side['manifest_sha256'] != expected_manifest or side['prediction_sha256'] != sha(prediction):
        raise ValueError('Saved prediction provenance changed')
    if rolling:
        job = next(j for j in parent['checkpoints'] if (j['dataset'], j['seed']) == (dataset, seed))
        if side['checkpoint_sha256'] != job['checkpoint_sha256']:
            raise ValueError('Rolling checkpoint identity differs')
    with np.load(prediction, allow_pickle=False) as z:
        events = z['event_ids'].astype(str)
        if len(events) != len(meta) or len(set(events)) != len(events) or set(events) != set(meta):
            raise ValueError('Query population differs')
        words = np.asarray([meta[e]['word'] for e in events])
        common = np.asarray([meta[e]['common'] for e in events])
        if not np.array_equal(z['mask2'] if rolling else z['common_mask'], common):
            raise ValueError('Common population mask differs')
        if not rolling and not np.array_equal(z['words'], words):
            raise ValueError('Saved targets differ from frozen manifest')
        selected = np.isin(words, info['vocabulary50']) & (common if population == 'common_targets' else True)
        q = z[f'k{h}'] if rolling else z['predictions']
        q = q[selected].copy()
        ids, words = events[selected], words[selected]
    cache = Path(info['evaluation_target_cache'])
    with np.load(cache, allow_pickle=False) as z:
        word_index = {w: i for i, w in enumerate(z['words'].astype(str))}
        bank = z['embeddings'][[word_index[w] for w in info['vocabulary50']]]
    ranks = retrieval_ranks(q, words, bank, info['vocabulary50'])
    score = summarize_retrieval(ranks, words, 50, top_ks=(1, 10))
    for key in ('query_count', 'target_count', 'candidate_count', *METRICS):
        if not np.isclose(score[key], values[key], atol=1e-12, rtol=0) or not np.isclose(score[key], float(expected_csv[key]), atol=1e-12, rtol=0):
            raise ValueError(f'Recomputed metric differs: {dataset} {seed} {condition} {key}')
    order = np.argsort(ids)
    return ids[order], words[order], ranks[order], score


def weighted_scores(weights, counts, hits):
    """Weights: draws x blocks; hits: condition x seed x cutoff x block x word."""
    denominator = weights @ counts
    present = denominator > 0
    type_counts = present.sum(axis=1)
    if np.any(type_counts == 0):
        raise ValueError('Empty material sample')
    numerator = np.einsum('rb,cskbq->cskrq', weights, hits, optimize=True)
    ratios = np.divide(numerator, denominator[None, None, None, :, :],
                       out=np.zeros_like(numerator), where=present[None, None, None, :, :])
    macro = (ratios.sum(axis=-1) / type_counts).mean(axis=1)
    micro = (numerator.sum(axis=-1) / denominator.sum(axis=-1)).mean(axis=1)
    # METRICS order: macro@10, macro@1, micro@10, micro@1.
    scores = np.stack([macro[:, 0], macro[:, 1], micro[:, 0], micro[:, 1]], axis=1)
    return scores, type_counts


def bootstrap_weights(info, m, index):
    blocks = [b['block'] for b in info['blocks']]
    strata = defaultdict(list)
    for i, b in enumerate(info['blocks']):
        strata[b['scope']].append(i)
    rng = np.random.default_rng(np.random.SeedSequence([m['random_seed'], index]))
    weights = np.zeros((m['repetitions'], len(blocks)), dtype=np.int32)
    for scope, positions in sorted(strata.items()):
        draws = rng.choice(positions, size=(m['repetitions'], len(positions)), replace=True)
        for i in range(m['repetitions']):
            weights[i] += np.bincount(draws[i], minlength=len(blocks))
    return weights


def run(m):
    check_sources(m)
    parent, context = read(PARENT), read(CONTEXT)
    byseed = dict(rolling=csv_rows(EXPORT / 'unified_heldout_test_v2_word_by_seed.csv'),
                  context=csv_rows(EXPORT / 'unified_context_heldout_test_v2_context_by_seed.csv'))
    summary, seedrows, supportrows, bootrows, omissionrows = [], [], [], [], []
    index = 0
    for dataset, info in parent['datasets'].items():
        meta = metadata(info)
        for contrast in m['contrasts']:
            print(dataset, contrast['id'], flush=True)
            ids0 = words0 = None
            ranks = []
            expected_scores = []
            for condition in (contrast['left'], contrast['right']):
                cranks, cscore = [], []
                for seed in m['seeds']:
                    ids, words, rank, score = ranks_for(dataset, seed, condition, contrast['population'], info, parent, context, meta, byseed)
                    if ids0 is None:
                        ids0, words0 = ids, words
                    if not np.array_equal(ids, ids0) or not np.array_equal(words, words0):
                        raise ValueError('Paired event/word mismatch')
                    cranks.append(rank)
                    cscore.append([score[k] for k in METRICS])
                ranks.append(cranks)
                expected_scores.append(cscore)
            ranks, expected_scores = np.asarray(ranks), np.asarray(expected_scores)
            blocks = [b['block'] for b in info['blocks']]
            bi = {b: i for i, b in enumerate(blocks)}
            types, wi = np.unique(words0, return_inverse=True)
            bq = np.asarray([bi[meta[e]['block']] for e in ids0])
            counts = np.zeros((len(blocks), len(types)))
            np.add.at(counts, (bq, wi), 1)
            hits = np.zeros((2, 3, 2, *counts.shape))
            for c in range(2):
                for seed in range(3):
                    for k, cutoff in enumerate((10, 1)):
                        np.add.at(hits[c, seed, k], (bq, wi), ranks[c, seed] <= cutoff)
            observed, _ = weighted_scores(np.ones((1, len(blocks))), counts, hits)
            if not np.allclose(observed[:, :, 0], expected_scores.mean(axis=1), atol=1e-12, rtol=0):
                raise ValueError('Word aggregation estimand differs')
            weights = bootstrap_weights(info, m, index)
            boot, represented = weighted_scores(weights, counts, hits)
            diffs = boot[1] - boot[0]
            complete = represented == len(types)
            omissions, _ = weighted_scores(np.ones((len(blocks), len(blocks))) - np.eye(len(blocks)), counts, hits)
            leaveout = omissions[1] - omissions[0]
            identity = dict(dataset=dataset, contrast=contrast['id'], population=contrast['population'])
            for mi, metric in enumerate(METRICS):
                lo, hi = np.quantile(diffs[mi], [.025, .975])
                clo, chi = np.quantile(diffs[mi, complete], [.025, .975]) if complete.any() else (np.nan, np.nan)
                summary.append(dict(**identity, metric=metric, left=contrast['left'], right=contrast['right'],
                    left_mean=float(observed[0, mi, 0]), right_mean=float(observed[1, mi, 0]),
                    effect=float(observed[1, mi, 0]-observed[0, mi, 0]), ci_lower=float(lo), ci_upper=float(hi),
                    bootstrap_samples=len(weights), query_count=len(ids0), target_count=len(types),
                    candidate_count=50, blocks=len(blocks), subjects=info['counts']['subjects'], seeds=3,
                    bootstrap_type_count_min=int(represented.min()), bootstrap_type_count_mean=float(represented.mean()),
                    complete_type_bootstrap_samples=int(complete.sum()), complete_type_ci_lower=float(clo), complete_type_ci_upper=float(chi),
                    leave_one_block_effect_min=float(leaveout[mi].min()), leave_one_block_effect_max=float(leaveout[mi].max())))
                for seed in range(3):
                    seedrows.append(dict(**identity, metric=metric, seed=seed,
                        left=contrast['left'], right=contrast['right'], left_score=float(expected_scores[0, seed, mi]),
                        right_score=float(expected_scores[1, seed, mi]), effect=float(expected_scores[1, seed, mi]-expected_scores[0, seed, mi])))
                for b, effect in zip(blocks, leaveout[mi]):
                    omissionrows.append(dict(**identity, metric=metric, omitted_block=b, effect=float(effect)))
            for word, coverage, n in zip(types, (counts > 0).sum(axis=0), counts.sum(axis=0)):
                supportrows.append(dict(**identity, word=word, query_count=int(n), supporting_blocks=int(coverage)))
            for i in range(len(weights)):
                bootrows.append(dict(**identity, replicate=i, represented_types=int(represented[i]),
                                     **{k: float(diffs[j, i]) for j, k in enumerate(METRICS)}))
            index += 1
    exports = {}
    for name, rows in [('summary', summary), ('by_seed', seedrows), ('word_support', supportrows),
                       ('bootstrap', bootrows), ('leave_one_block_out', omissionrows)]:
        path = EXPORT / f'word_paired_statistics_test_v2_{name}.csv'
        write_csv(path, rows)
        exports[path.relative_to(ROOT).as_posix()] = sha(path)
    report = dict(status='verified_complete', analysis_status=m['analysis_status'], manifest_sha256=m['manifest_sha256'],
                  primary_metric=m['primary_metric'], comparisons=4, summary=summary, exports=exports,
                  original_seed_metrics_reproduced=True, frozen_inputs_unchanged=True,
                  all_bootstrap_samples_retained=True, original_sentence_AC_tests_unchanged=True,
                  p_tests=False, completed_unix=time.time())
    check_sources(m)
    save(REPORT / 'report.json', report)
    save(EXPORT / 'word_paired_statistics_test_v2_summary.json', report)
    lines = ['# 词级核心结论的补充配对统计', '',
             '本分析在结果之后固定，不属于原预设句子检验族；没有新增 p 值。', '',
             '| 数据集 | 比较 | Macro Top-10 差 (pp) | 逐项95%材料块区间 (pp) | 查询/真实类型/材料块 |',
             '|---|---|---:|---|---|']
    for r in summary:
        if r['metric'] == m['primary_metric']:
            lines.append(f"| {r['dataset']} | {r['contrast']} | {r['effect']*100:+.2f} | [{r['ci_lower']*100:.2f}, {r['ci_upper']*100:.2f}] | {r['query_count']}/{r['target_count']}/{r['blocks']} |")
    lines += ['', '区间以既有受试者、训练种子和记录/章节为条件。Macro 分母在重采样中可能因缺少真实类型而变化；',
              '候选词保持50。类型覆盖、保留全部类型的敏感性区间、逐块删除范围及四项完整指标见正式导出。']
    (REPORT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze-only', action='store_true')
    args = parser.parse_args()
    manifest = freeze()
    if not args.freeze_only:
        run(manifest)
