"""Inference-only test extension for the 48 validation-selected context models."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch

from braindecoding import unified_context_baselines_v2 as development
from braindecoding import unified_heldout_test_v2 as heldout
from braindecoding.unified_sentence_v2 import read, save, digest, sha256
from braindecoding.unified_sentence_v2_artifacts import csv_write
from braindecoding.training.runtime import load_checkpoint, set_seed
from braindecoding.training.word import make_loader, encode_loader

ROOT = heldout.PROJECT_ROOT
PROTOCOL = ROOT / 'data/manifests/rolling_sentence/unified_protocol_v2.md'
MANIFEST = PROTOCOL.with_suffix('.json')
OUT = ROOT / 'outputs/heldout_test/unified_context_heldout_test_v2'
REPORT = ROOT / 'data/results/studies/unified_context_heldout_test_v2'
RUNTIME = ROOT / '.cache/runs/unified_context_heldout_test_v2'
EXPORT = ROOT / 'data/results/exports'
METRICS = ('macro_recall_at_1', 'macro_recall_at_10', 'micro_recall_at_1', 'micro_recall_at_10')
CONDITIONS = ('local_r0', 'local_r1', 'local_r2', 'local_r4', 'local_r8', 'full_context', 'independent_k0', 'independent_k1')


def freeze():
    if MANIFEST.exists():
        m = read(MANIFEST)
        if digest({k: v for k, v in m.items() if k != 'manifest_sha256'}) != m['manifest_sha256']:
            raise ValueError('Context test manifest identity changed')
        return m
    parent = heldout.manifest()
    plan_path = development.RUNTIME / 'plan.json'
    plan = read(plan_path)
    training_report = read(development.REPORT / 'report.json')
    if training_report['completed_jobs'] != 48 or sha256(plan_path) != training_report['plan_sha256']:
        raise ValueError('Need all 48 completed development jobs')
    jobs = []
    hashes = {str(p.resolve()): sha256(p) for p in [PROTOCOL, Path(__file__), plan_path, heldout.MANIFEST,
              development.CONFIG, Path(development.__file__), Path(heldout.__file__),
              ROOT / 'braindecoding/training/word.py', ROOT / 'braindecoding/models.py',
              ROOT / 'braindecoding/training/pallier_alignment_trial.py',
              ROOT / 'data/results/studies/unified_heldout_test_v2/word_results.json',
              EXPORT / 'unified_heldout_test_v2_word_by_seed.csv']}
    for job in plan['jobs']:
        out = Path(job['output_dir'])
        summary = read(out / 'training_summary.json')
        run = read(out / 'run_manifest.json')
        if (summary['status'] != 'completed' or summary['test_evaluated'] is not False
                or run['status'] != 'completed' or run['config'] != job['config']
                or run['plan_sha256'] != training_report['plan_sha256']):
            raise ValueError('Development job identity or selection provenance changed')
        jobs.append({k: job[k] for k in ('dataset', 'seed', 'condition', 'config')})
        jobs[-1].update(checkpoint=str(out / 'best.pt'), checkpoint_sha256=summary['checkpoint_sha256'],
                        best_epoch=summary['best']['epoch'], training_summary=str(out / 'training_summary.json'))
        hashes[str((out / 'training_summary.json').resolve())] = sha256(out / 'training_summary.json')
    expected = {(d, s, c) for d in parent['datasets'] for s in (0, 1, 2) for c in CONDITIONS}
    if {(j['dataset'], j['seed'], j['condition']) for j in jobs} != expected or len(jobs) != 48:
        raise ValueError('Context test job population changed')
    jobs.sort(key=lambda j: (list(parent['datasets']).index(j['dataset']), j['seed'], CONDITIONS.index(j['condition'])))
    for data in parent['datasets'].values():
        for path in [Path(data['evaluation_target_cache']), Path(data['evaluation_target_cache']).with_suffix('.json')]:
            hashes[str(path.resolve())] = sha256(path)
    m = dict(protocol='unified_context_heldout_test_v2', authorization_date='2026-10-03', split='test',
             training_allowed=False, selection_allowed=False, frozen_unix=time.time(),
             test_material_manifest_sha256=parent['manifest_sha256'], training_plan_sha256=sha256(plan_path),
             aliases={'independent_k2': 'local_r2'}, jobs=jobs, source_sha256=hashes,
             counts={d: v['counts'] for d, v in parent['datasets'].items()})
    m['manifest_sha256'] = digest(m)
    save(MANIFEST, m)
    return m


def root_for(job):
    return OUT / job['dataset'] / f"seed-{job['seed']:03d}" / job['condition']


def subject_order(dataset):
    if hasattr(dataset, 'subject_ids'):
        return list(dataset.subject_ids)
    return sorted(dataset.subject_map, key=dataset.subject_map.get)


def verify_saved(path, job, m):
    data = read(path)
    if (data['status'] != 'completed' or data['split'] != 'test' or data['test_evaluated'] is not True
            or data['manifest_sha256'] != m['manifest_sha256']
            or data['checkpoint_sha256'] != job['checkpoint_sha256']
            or any(data[k] != job[k] for k in ('dataset', 'seed', 'condition'))
            or sha256(path.with_name('predictions.npz')) != data['prediction_sha256']):
        raise ValueError('Saved test job provenance changed')
    return data


def progress(m, **fields):
    previous = read(RUNTIME / 'status.json') if (RUNTIME / 'status.json').exists() else {}
    if fields.get('state') == 'running':
        previous.pop('error', None)
    previous.update(pid=os.getpid(), updated_unix=time.time(), split='test', test_evaluated=True,
                    manifest_sha256=m['manifest_sha256'], planned_jobs=48, **fields)
    save(RUNTIME / 'status.json', previous)
    complete = []
    for job in m['jobs']:
        path = root_for(job) / 'metrics.json'
        if path.exists():
            data = read(path)
            if data.get('status') == 'completed' and data.get('manifest_sha256') == m['manifest_sha256']:
                complete.append(data)
    stamp = datetime.fromtimestamp(time.time(), timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    lines = ['# 词级上下文模型测试评价', '', f'更新时间：{stamp}（北京时间）', '',
             f"状态：{previous.get('state', 'running')}；完成 **{len(complete)} / 48**。", '',
             '仅对已选定检查点执行测试集推理，不训练或调参。独立 K2 与局部半径 2 共用同一结果。', '',
             f"当前：{previous.get('dataset', '')} / 种子 {previous.get('seed', '')} / {previous.get('condition', '')}。", '']
    if complete and len(complete) < 48:
        remaining = float(np.mean([d['seconds'] for d in complete])) * (48 - len(complete))
        lines += [f'按已完成作业推算，剩余约 {remaining / 60:.0f} 分钟；不含首次准备时间。', '']
    if previous.get('error'):
        lines += [f"错误：{previous['error']}", '']
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'progress.md').write_text('\n'.join(lines), encoding='utf-8')


class ProgressLoader:
    def __init__(self, loader, m, identity):
        self.loader, self.m, self.identity = loader, m, identity

    def __iter__(self):
        for number, batch in enumerate(self.loader, 1):
            yield batch
            if number == 1 or number % 40 == 0:
                progress(self.m, stage='inference', completed_batches=number, **self.identity)


def export(m):
    rows, sources = [], []
    for job in m['jobs']:
        path = root_for(job) / 'metrics.json'
        data = verify_saved(path, job, m)
        sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)))
        for population, values in data['strict_vocab50'].items():
            rows.append(dict(dataset=job['dataset'], seed=job['seed'], condition=job['condition'],
                             population=population, split='test', best_epoch=job['best_epoch'],
                             checkpoint=job['checkpoint'], candidate_count=values['candidate_count'],
                             query_count=values['query_count'], target_count=values['target_count'],
                             **{k: values[k] for k in METRICS}))
    if len(rows) != 96:
        raise ValueError('Incomplete context test export')
    summary = []
    for dataset in m['counts']:
        for condition in CONDITIONS:
            for population in ('all_targets', 'common_targets'):
                group = [r for r in rows if (r['dataset'], r['condition'], r['population']) == (dataset, condition, population)]
                if {r['seed'] for r in group} != {0, 1, 2}:
                    raise ValueError('Missing seed')
                result = dict(dataset=dataset, condition=condition, population=population, split='test', n_seeds=3,
                              candidate_count=50, query_count=group[0]['query_count'])
                for metric in METRICS:
                    values = [r[metric] for r in group]
                    result[metric + '_mean'] = float(np.mean(values))
                    result[metric + '_sd'] = float(np.std(values, ddof=1))
                summary.append(result)
    roll = read(heldout.REPORT / 'word_results.json')
    if roll['manifest_sha256'] != m['test_material_manifest_sha256'] or roll['split'] != 'test':
        raise ValueError('Wrong rolling test source')
    for source in roll['sources']:
        if sha256(ROOT / source['path']) != source['sha256']:
            raise ValueError('Rolling test source changed')
    import csv
    with (EXPORT / 'unified_heldout_test_v2_word_by_seed.csv').open(encoding='utf-8-sig', newline='') as stream:
        rolling = list(csv.DictReader(stream))
    comparisons = []
    aliases = {'independent_k0': 0, 'independent_k1': 1, 'local_r2': 2}
    for row in rows:
        if row['population'] != 'common_targets' or row['condition'] not in aliases:
            continue
        h = aliases[row['condition']]
        reference = next(r for r in rolling if (r['dataset'], int(r['seed']), int(r['horizon']), r['population']) ==
                         (row['dataset'], row['seed'], h, 'common_targets'))
        if int(reference['query_count']) != row['query_count'] or int(reference['candidate_count']) != 50:
            raise ValueError('Independent and rolling test query populations differ')
        item = dict(dataset=row['dataset'], seed=row['seed'], horizon=f'K{h}', split='test', query_count=row['query_count'])
        for metric in METRICS:
            item['independent_' + metric] = row[metric]
            item['rolling_' + metric] = float(reference[metric])
            item['rolling_minus_independent_' + metric] = float(reference[metric]) - row[metric]
        comparisons.append(item)
    files = {}
    for name, values in [('context_by_seed', rows), ('context_summary', summary), ('independent_vs_rolling_by_seed', comparisons)]:
        path = EXPORT / ('unified_context_heldout_test_v2_' + name + '.csv')
        csv_write(path, values)
        files[str(path.relative_to(ROOT))] = sha256(path)
    save(REPORT / 'report.json', dict(status='completed', split='test', test_evaluated=True,
         completed_jobs=48, planned_jobs=48, manifest_sha256=m['manifest_sha256'],
         test_material_manifest_sha256=m['test_material_manifest_sha256'], sources=sources, exports=files,
         completed_unix=time.time(), aliases=m['aliases']))
    progress(m, state='completed', stage='completed', completed_jobs=48)


def run(m):
    import msvcrt
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with (RUNTIME / 'worker.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        progress(m, state='running', stage='verifying_sources', started_unix=time.time(), completed_jobs=0)
        try:
            heldout.neural.check_hashes(m['source_sha256'])
            parent = read(heldout.MANIFEST)
            torch.set_num_threads(4)
            set_seed(0)
            ds = loader = dataset_name = None
            for index, job in enumerate(m['jobs']):
                path = root_for(job) / 'metrics.json'
                if path.exists():
                    verify_saved(path, job, m)
                    continue
                identity = {k: job[k] for k in ('dataset', 'seed', 'condition')}
                progress(m, stage='loading_checkpoint', completed_jobs=index, **identity)
                started = time.perf_counter()
                cfg = job['config']
                if sha256(Path(job['checkpoint'])) != job['checkpoint_sha256']:
                    raise ValueError('Validation-selected context checkpoint changed')
                if dataset_name != job['dataset']:
                    ds = heldout.dataset_for(cfg)
                    loader, _ = make_loader(ds, cfg['training'], shuffle=False, group_column=heldout.neural.group_column(cfg))
                    dataset_name = job['dataset']
                cp = load_checkpoint(job['checkpoint'])
                if (cp['dataset'] != job['dataset'] or cp['seed'] != job['seed'] or cp['condition'] != job['condition']
                        or cp['stage'] != 'context' or cp['selection'] != 'legacy_common_macro_top10'
                        or cp['model_config'] != cfg['model'] or cp['text_embedding_config'] != cfg['text_embedding']
                        or cp['channel_names'] != ds.channel_names or cp['subject_ids'] != subject_order(ds)
                        or cp['epoch'] != job['best_epoch']):
                    raise ValueError('Model identity, inputs or development selection changed')
                model = heldout.neural.build_brain_embedding_model(ds.channel_count, ds.channel_positions, ds.subject_count, cp['model_config']).cuda().eval()
                model.load_state_dict(cp['model_state'], strict=True)
                with torch.inference_mode():
                    encoded = encode_loader(model, ProgressLoader(loader, m, identity), torch.device('cuda'), amp=True)
                info = parent['datasets'][job['dataset']]
                expected = [event for group in info['groups'] for event in group['event_ids']]
                if len(encoded['event_ids']) != len(expected) or set(encoded['event_ids']) != set(expected):
                    raise ValueError('Test event population changed')
                if not torch.isfinite(encoded['predictions']).all():
                    raise FloatingPointError('Non-finite test predictions')
                common = development.common_mask(ds.table, encoded['event_ids'], heldout.neural.group_column(cfg))
                if int(common.sum()) != info['counts']['common_events']:
                    raise ValueError('Common test event population changed')
                evaluation = {**encoded, 'revision_predictions': {0: encoded['predictions']}, 'revision_availability_masks': {0: common}}
                metrics = heldout.neural.strict_bank_metrics(evaluation, ds, info['vocabulary50'])['0']
                for name, expected_count in [('all_targets', info['counts']['strict50_all']), ('common_targets', info['counts']['strict50_common'])]:
                    if metrics[name]['query_count'] != expected_count or metrics[name]['candidate_count'] != 50:
                        raise ValueError('Strict-50 test population changed')
                root_for(job).mkdir(parents=True, exist_ok=True)
                prediction = path.with_name('predictions.npz')
                temp = prediction.with_suffix('.tmp')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream, event_ids=np.asarray(encoded['event_ids']), predictions=encoded['predictions'].numpy(),
                                        words=np.asarray(encoded['words']), common_mask=common.numpy())
                temp.replace(prediction)
                save(path, dict(status='completed', split='test', test_evaluated=True, **identity,
                                manifest_sha256=m['manifest_sha256'], checkpoint_sha256=job['checkpoint_sha256'],
                                best_epoch=job['best_epoch'], prediction_sha256=sha256(prediction), strict_vocab50=metrics,
                                seconds=time.perf_counter() - started))
                del cp, model, encoded, evaluation
                gc.collect()
                torch.cuda.empty_cache()
                progress(m, stage='job_complete', completed_jobs=index + 1, **identity)
                print(f"Completed {index + 1}/48: {identity}", flush=True)
            heldout.neural.check_hashes(m['source_sha256'])
            export(m)
        except BaseException as error:
            progress(m, state='failed', stage='failed', error=repr(error))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'report'))
    args = parser.parse_args()
    manifest = freeze()
    if args.command == 'run':
        run(manifest)
    elif args.command == 'report':
        export(manifest)
    else:
        print(manifest['manifest_sha256'])
