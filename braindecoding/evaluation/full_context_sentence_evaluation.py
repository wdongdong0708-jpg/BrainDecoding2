"""Frozen inference-only F baseline, paired with the existing held-out A/C."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from braindecoding.config import PROJECT_ROOT as ROOT
from braindecoding.unified_sentence_v2 import digest, read, save, sha256, load_lm
from braindecoding.unified_sentence_batch import search_many, replay_many
from braindecoding.unified_sentence_v2_results import METRICS, metric_row, paired_block_statistics
from braindecoding.unified_sentence_v2_artifacts import csv_write

ID = 'full_context_sentence_test_v2'
PROTOCOL = ROOT / f'data/manifests/rolling_sentence/{ID}.md'
MANIFEST = PROTOCOL.with_suffix('.json')
OUT = ROOT / f'outputs/heldout_test/{ID}'
REPORT = ROOT / f'data/results/studies/{ID}'
EXPORT = ROOT / 'data/results/exports'
RUNTIME = ROOT / f'.cache/runs/{ID}'
PARENT = ROOT / 'data/manifests/rolling_sentence/heldout_test_v2.json'
CONTEXT = ROOT / 'data/manifests/rolling_sentence/context_heldout_test_v2.json'


def verify_identity(m):
    assert digest({k: v for k, v in m.items() if k != 'manifest_sha256'}) == m['manifest_sha256']


def verify_hashes(sources):
    for path, expected in sources.items():
        if str(path).endswith((".py", ".yaml", ".yml", ".md", ".tex")):
            continue
        if sha256(ROOT / path) != expected:
            raise ValueError(f'Frozen source changed: {path}')


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def source_key(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        verify_identity(m)
        verify_hashes(m['source_sha256'])
        return m
    parent, context = read(PARENT), read(CONTEXT)
    verify_identity(parent)
    verify_identity(context)
    assert context['test_material_manifest_sha256'] == parent['manifest_sha256']
    original = ROOT / 'data/results/exports/unified_heldout_test_v2_sentence_summary.json'
    summary = read(original)
    assert summary['status'] == 'completed_heldout_test'
    assert summary['manifest_sha256'] == parent['manifest_sha256']
    sources = {rel(p): h for p, h in parent['source_sha256'].items()}
    for path in (PROTOCOL, Path(__file__), PARENT, CONTEXT, original,
                 ROOT / 'braindecoding/evaluation/unified_sentence_v2_results.py',
                 ROOT / 'braindecoding/decoding/unified_sentence_v2_artifacts.py',
                 ROOT / 'braindecoding/decoding/jayalath_sentence.py'):
        sources[rel(path)] = sha256(path)
    # Pin the cached BGE files actually used for new out-of-vocabulary proposals.
    from huggingface_hub import snapshot_download
    text_config = parent['neural_settings']['text_embedding']
    text_snapshot = Path(snapshot_download(text_config['model_name'], revision=text_config['revision'], local_files_only=True))
    for path in text_snapshot.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.txt', '.model', '.safetensors', '.bin'):
            sources[source_key(path)] = sha256(path)
    jobs = []
    for dataset, info in parent['datasets'].items():
        expected = [e for g in info['groups'] for e in g['event_ids']]
        assert len(expected) == info['counts']['events'] and len(set(expected)) == len(expected)
        source_job = next(j for j in parent['checkpoints'] if j['dataset'] == dataset)
        event_path = Path(source_job['config']['cache']['event_table'])
        sources[rel(event_path)] = sha256(event_path)
        lm_cache = ROOT / f'outputs/heldout_test/unified_heldout_test_v2/{dataset}/lm_cache.pkl'
        if lm_cache.exists():
            sources[rel(lm_cache)] = sha256(lm_cache)
        for seed in (0, 1, 2):
            job = next(j for j in context['jobs'] if (j['dataset'], j['seed'], j['condition']) == (dataset, seed, 'full_context'))
            full_root = ROOT / f'outputs/heldout_test/unified_context_heldout_test_v2/{dataset}/seed-{seed:03d}/full_context'
            metrics, pred = full_root / 'metrics.json', full_root / 'predictions.npz'
            values = read(metrics)
            assert values['status'] == 'completed' and values['split'] == 'test' and values['test_evaluated']
            assert values['manifest_sha256'] == context['manifest_sha256']
            assert values['checkpoint_sha256'] == job['checkpoint_sha256']
            assert sha256(pred) == values['prediction_sha256']
            with np.load(pred, allow_pickle=False) as archive:
                ids = list(map(str, archive['event_ids']))
                assert len(ids) == len(expected) and set(ids) == set(expected)
                lookup = {e: i for i, e in enumerate(ids)}
                queries = archive['predictions']
                assert queries.shape == (len(expected), 1024) and np.isfinite(queries).all()
                assert np.allclose(np.linalg.norm(queries, axis=1), 1., atol=.005)
                for group in info['groups']:
                    assert [str(archive['words'][lookup[e]]) for e in group['event_ids']] == group['reference']
            original_root = ROOT / f'outputs/heldout_test/unified_heldout_test_v2/{dataset}/seed-{seed:03d}'
            ac = original_root / 'main.json'
            saved = read(ac)
            assert saved['manifest_sha256'] == parent['manifest_sha256'] and saved['status'] == 'completed'
            assert [r['group'] for r in saved['records']] == [g['group'] for g in info['groups']]
            for item in saved['sources']:
                assert sha256(ROOT / item['path']) == item['sha256']
            for p in (metrics, pred, ac, Path(job['checkpoint']), original_root / 'neural_predictions.npz'):
                sources[rel(p)] = sha256(p)
            jobs.append(dict(dataset=dataset, seed=seed, prediction=rel(pred), prediction_sha256=values['prediction_sha256'],
                             checkpoint=rel(job['checkpoint']), checkpoint_sha256=job['checkpoint_sha256'],
                             original_ac=rel(ac), event_table=rel(event_path), parent_cache=rel(lm_cache) if lm_cache.exists() else None))
    verify_hashes(sources)
    settings = dict(condition='F', training_allowed=False, recalibration_allowed=False,
                    selection_allowed=False, split='test', language_weight=.1,
                    decoder=copy.deepcopy(parent['decoder']), segment_batch_size=16,
                    neural_input='all_existing_full_context_predictions_within_frozen_segment',
                    search='single_full_segment_beam_from_empty_generated_prefix',
                    statistics='additional_C_minus_F_material_block_bootstrap_95_CI_no_new_p_tests',
                    evidence='extension_after_original_AC_results',
                    latency='nominal_onset_plus_1s_excludes_preprocessing_and_compute')
    m = dict(protocol=ID, authorization_date='2026-10-05', authorization='explicit_user_start_full_context_sentence_comparison',
             frozen_unix=time.time(), parent_manifest_sha256=parent['manifest_sha256'],
             context_manifest_sha256=context['manifest_sha256'], settings=settings, jobs=jobs, source_sha256=sources,
             total_segment_seed_units=sum(len(v['groups']) for v in parent['datasets'].values()) * 3)
    m['manifest_sha256'] = digest(m)
    save(MANIFEST, m)
    print(json.dumps(dict(stage='frozen', jobs=len(jobs), total=m['total_segment_seed_units'], manifest_sha256=m['manifest_sha256'])), flush=True)
    return m


def status(m, **fields):
    path = RUNTIME / 'status.json'
    previous = read(path) if path.exists() else {}
    previous.update(pid=os.getpid(), updated_unix=time.time(), manifest_sha256=m['manifest_sha256'],
                    total_segment_seed_units=m['total_segment_seed_units'], **fields)
    save(path, previous)
    REPORT.mkdir(parents=True, exist_ok=True)
    completed = previous.get('completed_segment_seed_units', 0)
    lines = ['# 完整上下文句子比较进度', '',
             f"状态：{previous.get('state', 'running')}；已保存 {completed:,} / {m['total_segment_seed_units']:,} 个片段×种子。", '',
             f"当前：{previous.get('dataset', '')}，种子 {previous.get('seed', '')}，分块 {previous.get('chunk', '')}，词位 {previous.get('position', '')}。", '',
             'F 等完整片段词窗后搜索整段；沿用原 A/C 的模型、候选、评分与束宽设置。',
             'A/C 只读引用；不重新训练、校准或修改原结果。部分进度不作为完整科学结论。', '',
             f"更新时间（Unix）：{previous['updated_unix']:.0f}。"]
    if previous.get('error'):
        lines += ['', f"错误：{previous['error']}"]
    if previous.get('state') == 'completed':
        lines += ['', '[完整结果](results.md)']
    (REPORT / 'progress.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: previous[k] for k in ('state', 'stage', 'dataset', 'seed', 'chunk', 'position',
                                             'completed_segment_seed_units') if k in previous}), flush=True)


def validate_chunk(chunk, m, job, groups):
    assert chunk['manifest_sha256'] == m['manifest_sha256']
    assert chunk['dataset'] == job['dataset'] and chunk['seed'] == job['seed']
    assert chunk['condition'] == 'F' and chunk['split'] == 'test'
    assert [r['group'] for r in chunk['records']] == [g['group'] for g in groups]
    for group, record in zip(groups, chunk['records']):
        assert len(record['prediction']) == len(group['event_ids'])
        assert all(isinstance(w, str) and w for w in record['prediction'])


def summarize(rows):
    cells, subjects, seeds = defaultdict(list), [], []
    conditions = sorted({r['condition'] for r in rows})
    for row in rows:
        cells[(row['seed'], row['subject'], row['condition'])].append(row)
    for (seed, subject, condition), group in sorted(cells.items()):
        words = sum(r['words'] for r in group)
        values = {k: (float(np.mean([r[k] for r in group])) if k == 'sentence_bleu1_mean'
                      else sum(r[k] * r['words'] for r in group) / words) for k in METRICS}
        subjects.append(dict(seed=seed, subject=subject, condition=condition, segments=len(group), words=words, **values))
    for seed in (0, 1, 2):
        for condition in conditions:
            group = [r for r in subjects if r['seed'] == seed and r['condition'] == condition]
            assert group
            seeds.append(dict(seed=seed, condition=condition, **{k: float(np.mean([r[k] for r in group])) for k in METRICS}))
    means = [dict(condition=c, **{k: float(np.mean([r[k] for r in seeds if r['condition'] == c])) for k in METRICS}) for c in conditions]
    return means, seeds, subjects


def export(m, parent):
    all_rows, all_subjects, timing, sources, datasets = [], [], [], [], {}
    original = read(ROOT / 'data/results/exports/unified_heldout_test_v2_sentence_summary.json')
    for dataset, info in parent['datasets'].items():
        rows = []
        groups = info['groups']
        jobs = [j for j in m['jobs'] if j['dataset'] == dataset]
        event_table = pd.read_csv(ROOT / jobs[0]['event_table'], encoding='utf-8-sig')
        onset = dict(zip(event_table['事件编号'].astype(str), pd.to_numeric(event_table['开始时间'])))
        for group in groups:
            stamps = np.asarray([onset[e] for e in group['event_ids']], float)
            assert np.isfinite(stamps).all() and (np.diff(stamps) >= -1e-6).all()
            for i, event in enumerate(group['event_ids']):
                timing.append(dict(dataset=dataset, subject=group['subject'], group=group['group'], event_id=event,
                    material=group['material'], block=group['block'], scope=group['scope'],
                    word=group['reference'][i], position=i, common50=(i+2 < len(stamps) and group['reference'][i] in info['vocabulary50']),
                    k0_input_wait_s=1., k1_input_wait_s=float(stamps[min(i+1,len(stamps)-1)]-stamps[i]+1),
                    c_input_wait_s=float(stamps[min(i+2,len(stamps)-1)]-stamps[i]+1),
                    f_input_wait_s=float(stamps[-1]-stamps[i]+1)))
        for job in jobs:
            path = OUT / dataset / f"seed-{job['seed']:03d}" / 'main.json'
            completed = read(path)
            validate_chunk(completed, m, job, groups)
            assert completed['status'] == 'completed'
            reconstructed = []
            for source in completed['sources']:
                assert sha256(ROOT / source['path']) == source['sha256']
                reconstructed.extend(read(ROOT / source['path'])['records'])
            assert reconstructed == completed['records']
            ac = read(ROOT / job['original_ac'])
            for group, new, old in zip(groups, completed['records'], ac['records']):
                for condition in ('A', 'C', 'F'):
                    prediction = new['prediction'] if condition == 'F' else old['result']['final'][condition]
                    rows.append(dict(dataset=dataset, seed=job['seed'], subject=group['subject'], condition=condition,
                        group=group['group'], material=group['material'], block=group['block'], scope=group['scope'],
                        words=len(group['reference']), **metric_row(group['reference'], prediction, info['language'] == 'zh')))
            sources.append(dict(path=rel(path), sha256=sha256(path)))
        means, seeds, subjects = summarize(rows)
        lookup = {r['condition']: r for r in means}
        for condition in ('A', 'C'):
            expected = next(r for r in original['datasets'][dataset]['summary'] if r['condition'] == condition)
            for metric in METRICS:
                assert np.isclose(lookup[condition][metric], expected[metric], rtol=0, atol=1e-12), (dataset, condition, metric)
        pairs = defaultdict(dict)
        for row in rows:
            pairs[(row['seed'], row['subject'], row['group'])][row['condition']] = row
        differences = [dict(**{k: pair['C'][k] for k in ('seed','subject','group','block','scope')},
                            difference=pair['C']['sentence_bleu1_mean']-pair['F']['sentence_bleu1_mean']) for pair in pairs.values()]
        statistics = paired_block_statistics(differences, 'difference', parent['settings']['statistics'])
        statistics = {k: v for k, v in statistics.items() if not k.startswith('p_') and k != 'sign_flip_exact'}
        assert np.isclose(statistics['effect'], lookup['C']['sentence_bleu1_mean']-lookup['F']['sentence_bleu1_mean'], atol=1e-12)
        datasets[dataset] = dict(counts=info['counts'], summary=means, by_seed=seeds, by_subject_seed=subjects,
                                  additional_C_minus_F=statistics)
        all_rows.extend(rows)
        all_subjects.extend(dict(dataset=dataset, **r) for r in subjects)
    assert len(all_rows) == m['total_segment_seed_units'] * 3
    times = pd.DataFrame(timing)
    time_summary = []
    for dataset, info in parent['datasets'].items():
        for population in ('all_events', 'common50'):
            group = times[times.dataset == dataset]
            if population == 'common50':
                group = group[group.common50]
                assert len(group) == info['counts']['strict50_common']
            else:
                assert len(group) == info['counts']['events']
            for strategy, column in [('K0','k0_input_wait_s'),('K1','k1_input_wait_s'),('C_K2','c_input_wait_s'),('F','f_input_wait_s')]:
                values = group[column].to_numpy()
                time_summary.append(dict(dataset=dataset, population=population, strategy=strategy, queries=len(group),
                    mean_input_wait_s=float(values.mean()), median_input_wait_s=float(np.median(values)),
                    p90_input_wait_s=float(np.quantile(values,.9)), mean_extra_wait_after_k0_s=float(values.mean()-1),
                    mean_saved_vs_f_s=float((group.f_input_wait_s-group[column]).mean()),
                    aggregation='query_weighted_no_seed_repetition', excludes='alignment_preprocessing_compute_display'))
    reports = [('by_group', all_rows), ('by_subject_seed', all_subjects), ('input_wait_by_event', timing), ('input_wait_summary', time_summary)]
    outputs = {}
    for suffix, rows in reports:
        path = EXPORT / f'{ID}_{suffix}.csv'
        csv_write(path, rows)
        outputs[rel(path)] = sha256(path)
    result = dict(status='completed', protocol=ID, split='test', manifest_sha256=m['manifest_sha256'],
                  conditions=['A','C','F'], datasets=datasets, timing_summary=time_summary, sources=sources,
                  exports=outputs, original_AC_unchanged=True, comparison='additional_strategy_C_minus_F',
                  completed_unix=time.time(), inference_only=True, new_significance_tests=False)
    save(REPORT / 'results.json', result)
    summary_path = EXPORT / f'{ID}_summary.json'
    save(summary_path, result)
    save(REPORT / 'audit.json', dict(status='verified_complete', manifest_sha256=m['manifest_sha256'],
                                   result_sha256=sha256(summary_path), exports=outputs,
                                   total_segment_seed_units=m['total_segment_seed_units'], source_sha256=m['source_sha256']))
    lines = ['# 完整上下文与滚动句子解码比较', '', '全部六组已完成；A/C只读复用，F为补充比较。', '',
             '| 数据集 | 条件 | 平均片段 BLEU-1（%） | WER（%） | 对齐词准确率（%） |', '|---|---|---:|---:|---:|']
    for dataset, info in datasets.items():
        for row in info['summary']:
            lines.append(f"| {dataset} | {row['condition']} | {row['sentence_bleu1_mean']*100:.2f} | {row['wer']*100:.2f} | {row['aligned_word_accuracy']*100:.2f} |")
        stat = info['additional_C_minus_F']
        lines += ['', f"{dataset} 的 C−F BLEU-1：{stat['effect']*100:+.2f} 个百分点，配对材料块95%区间 [{stat['ci_lower']*100:+.2f}, {stat['ci_upper']*100:+.2f}]。", '']
    lines += ['输入等待是词起点后1秒窗完整可用的标称时间，不含预处理或计算。原 A/C 主检验未修改。']
    (REPORT / 'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run():
    m = freeze()
    parent = read(PARENT)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (RUNTIME / 'worker.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        torch.set_num_threads(4)
        torch.manual_seed(0)
        np.random.seed(0)
        status(m, state='running', stage='starting', completed_segment_seed_units=0, started_unix=time.time(), error=None)
        try:
            completed_count = 0
            for dataset, info in parent['datasets'].items():
                scorer = embed = bank = None
                cache_path = OUT / dataset / 'lm_cache.pkl'
                for job in [j for j in m['jobs'] if j['dataset'] == dataset]:
                    with np.load(ROOT / job['prediction'], allow_pickle=False) as archive:
                        values = archive['predictions']
                        lookup = {str(e): i for i, e in enumerate(archive['event_ids'])}
                    records, sources = [], []
                    groups = info['groups']
                    width = m['settings']['segment_batch_size']
                    for start in range(0, len(groups), width):
                        batch = groups[start:start+width]
                        number = start // width
                        path = OUT / dataset / f"seed-{job['seed']:03d}" / 'chunks' / f'chunk-{number:04d}.json'
                        if path.exists():
                            chunk = read(path)
                            validate_chunk(chunk, m, job, batch)
                        else:
                            if scorer is None:
                                status(m, state='running', stage='loading_models', dataset=dataset, seed=job['seed'], chunk=number)
                                scorer, embed, bank = load_lm(parent, dataset)
                                source_cache = cache_path if cache_path.exists() else ROOT / job['parent_cache'] if job['parent_cache'] else None
                                if source_cache is not None:
                                    scorer.load_cache(source_cache)
                            if not (OUT / dataset / 'compatibility_check.json').exists():
                                first = batch[0]
                                ac = next(r for r in read(ROOT / job['original_ac'])['records'] if r['group'] == first['group'])
                                with np.load(ROOT / f"outputs/heldout_test/unified_heldout_test_v2/{dataset}/seed-{job['seed']:03d}/neural_predictions.npz") as original:
                                    original_index = {str(e): i for i,e in enumerate(original['event_ids'])}
                                    q = original['k0'][[original_index[e] for e in first['event_ids']]]
                                inherited = replay_many([dict(predictions={0:q},weight=.1,open_vocabulary=True)], info['vocabulary'], bank, scorer, embed,
                                    conditions='A', candidate_count=20, beam_width=5, neural_scale=10.)[0]
                                assert inherited['final']['A'] == ac['result']['final']['A'], 'Inherited decoder failed to reproduce original A.'
                                save(OUT / dataset / 'compatibility_check.json', dict(manifest_sha256=m['manifest_sha256'], seed=job['seed'],
                                     group=first['group'], original_A_reproduced_exactly=True, reference_not_supplied_to_decoder=True))
                            started = time.perf_counter()
                            status(m, state='running', stage='full_context_search', dataset=dataset, seed=job['seed'], chunk=number,
                                   completed_segment_seed_units=completed_count)
                            requests = [dict(queries=values[[lookup[e] for e in g['event_ids']]], prefix=(), weight=.1, open_vocabulary=True) for g in batch]
                            # Reuse the frozen search implementation; requests never contain references.
                            predictions = search_many(requests, info['vocabulary'], bank, scorer, embed,
                                candidate_count=20, beam_width=5, neural_scale=10.)
                            chunk = dict(manifest_sha256=m['manifest_sha256'], dataset=dataset, seed=job['seed'], condition='F',
                                split='test', records=[dict(group=g['group'], prediction=list(p)) for g,p in zip(batch,predictions)],
                                seconds=time.perf_counter()-started, input_availability='wait_for_last_word_1s_window')
                            validate_chunk(chunk, m, job, batch)
                            save(path, chunk)
                            scorer.save_cache(cache_path)
                        records.extend(chunk['records'])
                        sources.append(dict(path=rel(path), sha256=sha256(path)))
                        completed_count += len(batch)
                        status(m, state='running', stage='chunk_complete', dataset=dataset, seed=job['seed'], chunk=number,
                               completed_segment_seed_units=completed_count)
                    save(path.parent.parent / 'main.json', dict(status='completed', manifest_sha256=m['manifest_sha256'], dataset=dataset,
                         seed=job['seed'], condition='F', split='test', records=records, sources=sources))
                    del values
                del scorer, embed, bank
                gc.collect()
                torch.cuda.empty_cache()
            verify_hashes(m['source_sha256'])
            status(m, state='running', stage='exporting', completed_segment_seed_units=completed_count)
            export(m, parent)
            status(m, state='completed', stage='completed', completed_unix=time.time(), completed_segment_seed_units=completed_count)
        except BaseException as error:
            status(m, state='failed', stage='failed', error=f'{type(error).__name__}: {error}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','run'])
    args = parser.parse_args()
    freeze() if args.command == 'prepare' else run()
