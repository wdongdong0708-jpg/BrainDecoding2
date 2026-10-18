"""Authorized held-out evaluation; frozen development modules remain untouched."""
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
import torch
import yaml

from braindecoding import unified_rolling_v2_training as neural
from braindecoding.config import PROJECT_ROOT, project_path
from braindecoding.training.runtime import load_checkpoint, set_seed
from braindecoding.training.word import make_loader, encode_loader
from braindecoding.unified_sentence_v2 import (
    read, save, digest, sha256, canonical_group_rows, load_lm, cases_for_groups,
)
from braindecoding.unified_sentence_batch import replay_many
from braindecoding.unified_sentence_v2_revision_gain_full import validate_result

CONFIG = PROJECT_ROOT / 'configs/evaluation/heldout_sentence_evaluation.yaml'
PROTOCOL = PROJECT_ROOT / 'data/manifests/rolling_sentence/unified_protocol_v2.md'
MANIFEST = PROTOCOL.with_suffix('.json')
OUT = PROJECT_ROOT / 'outputs/heldout_test/unified_heldout_test_v2'
RUNTIME = PROJECT_ROOT / '.cache/runs/unified_heldout_test_v2'
REPORT = PROJECT_ROOT / 'data/results/studies/unified_heldout_test_v2'


def config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if (cfg['protocol']['split'] != 'test' or not cfg['protocol']['test_evaluation_allowed']
            or cfg['protocol']['training_allowed'] or cfg['protocol']['recalibration_allowed']
            or cfg['sentence']['conditions'] != ['A', 'C'] or cfg['sentence']['language_weight'] != .1):
        raise ValueError('Unauthorized change to test evaluation contract')
    return cfg


def table_for(cfg, split):
    """Explicit test adapter; never relax the development loader's split guard."""
    pallier = cfg['experiment']['dataset'] == 'pallier2025'
    module = neural.pallier.dataset_module if pallier else neural.chinese.dataset_module
    loader = module.load_event_table if pallier else module.载入事件表
    table = loader(cfg['cache']['event_table'], split=split, subjects=cfg['dataset']['subjects'], trainable_only=True)
    if pallier:
        table = module.add_runtime_context_uid(table)
    for target, source in {'event_id':'事件编号', 'subject_id':'受试者', 'recording_id':'记录编号',
                           'normalized_word':'标准词', 'split':'数据划分', 'onset_seconds':'开始时间'}.items():
        table[target] = table[source]
    if table.event_id.duplicated().any() or not table.split.eq(split).all():
        raise ValueError('Duplicate events or incorrect data split')
    table['protocol_group'] = table[neural.group_column(cfg)]
    return table


def target_cache(cfg):
    return PROJECT_ROOT/'derived'/cfg['experiment']['dataset']/'text/unified_heldout_test_v2/embeddings.npz'


def prepare_target_cache(cfg, table):
    from braindecoding.data.text import load_text_embedding_cache, save_text_embedding_cache, encode_words, text_embedding_signature
    signature = text_embedding_signature(cfg['text_embedding'])
    original = load_text_embedding_cache(cfg['cache']['text_embeddings'], expected_signature=signature)
    required = set(table.normalized_word.astype(str))
    destination = target_cache(cfg)
    if destination.exists():
        existing = load_text_embedding_cache(destination, expected_signature=signature)
        if set(existing)!=(set(original)|required) or any(not np.array_equal(v,existing[w]) for w,v in original.items()):
            raise ValueError('Isolated test target cache has changed training vectors or word inventory')
        return destination
    missing = sorted(required-set(original))
    vectors = encode_words(missing,cfg['text_embedding'])
    merged = {**original, **dict(zip(missing,vectors))}
    words = sorted(merged)
    save_text_embedding_cache(destination,words,np.stack([merged[w] for w in words]),cfg['text_embedding'])
    gc.collect()
    torch.cuda.empty_cache()
    return destination


def dataset_for(cfg):
    table = table_for(cfg, 'test')
    builder = neural.pallier.build_dataset if cfg['experiment']['dataset']=='pallier2025' else neural.chinese.build_dataset
    evaluation_cfg = copy.deepcopy(cfg)
    evaluation_cfg['cache']['text_embeddings'] = str(target_cache(cfg))
    return builder(evaluation_cfg, table)


def block_groups(dataset, table, width):
    groups, materials = canonical_group_rows(dataset, table)
    scopes = defaultdict(list)
    for material in materials:
        scopes[material['scope']].append(material)
    blocks = []
    for scope, entries in sorted(scopes.items()):
        entries.sort(key=lambda r:(r['order'], r['material']))
        for start in range(0, len(entries), width):
            key = f'{dataset}|{scope}|test-block-{start//width:03d}'
            members = entries[start:start+width]
            indices = [i for entry in members for i in entry['groups']]
            for i in indices:
                groups[i].update(block=key, partition='test')
            blocks.append(dict(block=key, scope=scope, materials=[e['material'] for e in members], groups=indices))
    return groups, blocks, materials


def prepare():
    if MANIFEST.exists():
        return manifest()
    cfg = config()
    parent_path = project_path(cfg['sentence']['parent'])
    development_path = project_path(cfg['sentence']['development'])
    parent, development = read(parent_path), read(development_path)
    for record in (parent, development):
        if digest({k:v for k,v in record.items() if k!='manifest_sha256'}) != record['manifest_sha256']:
            raise ValueError('Development manifest identity changed')
    datasets, checkpoints, sources = {}, [], {}
    for path in (CONFIG, PROTOCOL, Path(__file__), parent_path, development_path,
                 PROJECT_ROOT/'braindecoding/evaluation/unified_heldout_test_v2_results.py',
                 PROJECT_ROOT/'braindecoding/decoding/unified_sentence_v2.py',
                 PROJECT_ROOT/'braindecoding/decoding/unified_sentence.py',
                 PROJECT_ROOT/'braindecoding/decoding/unified_sentence_batch.py',
                 PROJECT_ROOT/'braindecoding/decoding/unified_sentence_v2_revision_gain.py',
                 PROJECT_ROOT/'braindecoding/decoding/unified_sentence_v2_revision_gain_full.py',
                 PROJECT_ROOT/'braindecoding/evaluation/unified_sentence_v2_revision_gain_full_results.py',
                 PROJECT_ROOT/'braindecoding/evaluation/unified_sentence_v2_results.py'):
        sources[str(path.resolve())] = sha256(path)
    for dataset in cfg['protocol']['datasets']:
        jobs = [j for j in parent['checkpoints'] if j['dataset']==dataset]
        if sorted(j['seed'] for j in jobs) != [0,1,2]:
            raise ValueError('Need the three frozen rolling checkpoints')
        resolved = yaml.safe_load((Path(jobs[0]['path'])/'resolved_config.yaml').read_text(encoding='utf-8'))
        # Only metadata/cache checks here; no model prediction or score is computed.
        table = table_for(resolved, 'test')
        groups, blocks, materials = block_groups(dataset, table, cfg['population']['material_block_size'])
        rule = cfg['population'][dataset]
        if (len(table)!=rule['expected_events'] or table.subject_id.nunique()!=rule['expected_subjects']
                or sorted({g['scope'] for g in groups})!=sorted(rule['scopes'])):
            raise ValueError('Held-out population differs from predefined split')
        if any(len(g['event_ids'])>128 for g in groups):
            raise ValueError('Test segment would exceed the frozen batch size')
        for split in ('train','val'):
            other = table_for(resolved, split)
            if set(table.event_id) & set(other.event_id):
                raise ValueError('Test events overlap development events')
            if set(table.recording_id) & set(other.recording_id):
                raise ValueError('Test recordings overlap development recordings')
        # This validates signal caches and text representations without scoring targets.
        target_path = prepare_target_cache(resolved,table)
        sources[str(target_path.resolve())] = sha256(target_path)
        sources[str(target_path.with_suffix('.json').resolve())] = sha256(target_path.with_suffix('.json'))
        ds = dataset_for(resolved)
        vocab50 = list(neural.pallier.frozen_vocabulary() if dataset=='pallier2025'
                       else neural.chinese.frozen_vocabulary(resolved['cache']['event_table']))
        adapter = neural.config()['adapters'][dataset]
        vocabulary = read(project_path(adapter['vocabulary_manifest']))['vocabulary']
        if vocabulary != parent['datasets'][dataset]['vocabulary'] or len(vocab50)!=50:
            raise ValueError('Training candidate vocabulary changed')
        if development['datasets'][dataset]['selected_weight'] != .1:
            raise ValueError('Frozen validation language weight changed')
        common = table.groupby('protocol_group', sort=False).cumcount(ascending=False).ge(2)
        eligible = table.normalized_word.isin(vocab50)
        datasets[dataset] = dict(language=adapter['language'], embedding_cache=resolved['cache']['text_embeddings'],
            evaluation_target_cache=str(target_path),
            vocabulary=vocabulary, vocabulary50=vocab50, selected_weight=.1, groups=groups, blocks=blocks,
            counts=dict(events=len(table), groups=len(groups), subjects=int(table.subject_id.nunique()),
                        materials=len(materials), blocks=len(blocks), common_events=int(common.sum()),
                        strict50_all=int(eligible.sum()), strict50_common=int((eligible&common).sum())))
        sources.update(neural.source_hashes(resolved, adapter))
        for job in jobs:
            if sha256(Path(job['checkpoint'])) != job['checkpoint_sha256']:
                raise ValueError('Validation-selected checkpoint changed')
            cp = load_checkpoint(job['checkpoint'])
            if cp['stage']!='revision' or cp['seed']!=job['seed'] or cp['selection']!='legacy50':
                raise ValueError('Checkpoint is not the frozen rolling primary')
            job_cfg = yaml.safe_load((Path(job['path'])/'resolved_config.yaml').read_text(encoding='utf-8'))
            if cp['model_config']!=job_cfg['model'] or cp['text_embedding_config']!=parent['neural_settings']['text_embedding']:
                raise ValueError('Model or target embedding changed')
            checkpoints.append({**job, 'config':job_cfg})
            sources[str(Path(job['checkpoint']).resolve())] = job['checkpoint_sha256']
            sources[str((Path(job['path'])/'resolved_config.yaml').resolve())] = sha256(Path(job['path'])/'resolved_config.yaml')
            del cp
        del ds
        gc.collect()
    for relative, expected in parent['model_files'].items():
        path = PROJECT_ROOT/relative
        if sha256(path)!=expected:
            raise ValueError('Language model file changed')
        sources[str(path.resolve())] = expected
    result = dict(protocol=cfg['protocol']['id'], frozen_unix=time.time(), settings=cfg,
        # load_lm reads only these frozen inference settings.
        neural_settings=parent['neural_settings'], decoder=parent['settings']['decoding'],
        datasets=datasets, checkpoints=checkpoints, source_sha256=sources,
        parent_manifest_sha256=parent['manifest_sha256'], development_manifest_sha256=development['manifest_sha256'],
        evidence='heldout_test', test_evaluation_authorized=True, test_predictions_started=False)
    result['settings']['decoding'] = copy.deepcopy(parent['settings']['decoding'])
    result['settings']['decoding']['conditions'] = ['A','C']
    result['manifest_sha256'] = digest(result)
    save(MANIFEST, result)
    print(json.dumps({d:i['counts'] for d,i in datasets.items()}, ensure_ascii=False), flush=True)
    return result


def manifest():
    m = read(MANIFEST)
    if digest({k:v for k,v in m.items() if k!='manifest_sha256'}) != m['manifest_sha256']:
        raise ValueError('Frozen test manifest identity changed')
    neural.check_hashes(m['source_sha256'])
    return m


def status(**fields):
    path = RUNTIME/'status.json'
    old = read(path) if path.exists() else {}
    old.update(pid=os.getpid(), updated_unix=time.time(), split='test', test_evaluated=True, **fields)
    save(path, old)


def seed_root(dataset, seed):
    return OUT/dataset/f'seed-{seed:03d}'


def encode_all(m):
    for job in m['checkpoints']:
        dataset, seed, cfg = job['dataset'],job['seed'],job['config']
        root = seed_root(dataset, seed)
        destination, sidecar = root/'neural_predictions.npz', root/'word_metrics.json'
        if destination.exists() and sidecar.exists():
            saved = read(sidecar)
            if saved['manifest_sha256']!=m['manifest_sha256'] or saved['prediction_sha256']!=sha256(destination):
                raise ValueError('Test neural cache mismatch')
            continue
        status(state='running', stage='neural_encoding', dataset=dataset, seed=seed)
        ds = dataset_for(cfg)
        loader, _ = make_loader(ds, cfg['training'], shuffle=False, group_column=neural.group_column(cfg))
        cp = load_checkpoint(job['checkpoint'])
        model = neural.build_brain_embedding_model(ds.channel_count,ds.channel_positions,ds.subject_count,cp['model_config']).cuda().eval()
        model.load_state_dict(cp['model_state'], strict=True)
        encoded = encode_loader(model,loader,torch.device('cuda'),amp=True)
        info = m['datasets'][dataset]
        expected = [e for g in info['groups'] for e in g['event_ids']]
        if len(encoded['event_ids'])!=len(expected) or set(encoded['event_ids'])!=set(expected):
            raise ValueError('Test encoding omitted or duplicated events')
        lookup = {e:i for i,e in enumerate(encoded['event_ids'])}
        arrays = dict(event_ids=np.asarray(encoded['event_ids']))
        for h in range(3):
            values = encoded['revision_predictions'][h].numpy()
            if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1.,atol=.005):
                raise ValueError('Non-finite or unnormalized test predictions')
            arrays[f'k{h}'] = values
            arrays[f'mask{h}'] = encoded['revision_availability_masks'][h].numpy()
            for group in info['groups']:
                ids = [lookup[e] for e in group['event_ids']]
                if not np.array_equal(arrays[f'mask{h}'][ids], np.arange(len(ids))+h<len(ids)):
                    raise ValueError('Horizon availability differs from frozen boundary')
        strict = neural.strict_bank_metrics(encoded,ds,info['vocabulary50'])
        for values in strict.values():
            for population, count in [('all_targets',info['counts']['strict50_all']),('common_targets',info['counts']['strict50_common'])]:
                if values[population]['query_count']!=count or values[population]['candidate_count']!=50:
                    raise ValueError('Word evaluation changed the frozen query or candidate population')
        full = neural.full_vocabulary_metrics(encoded,ds,info['vocabulary'])
        root.mkdir(parents=True,exist_ok=True)
        with destination.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream,**arrays)
        destination.with_suffix('.tmp').replace(destination)
        save(sidecar,dict(dataset=dataset,seed=seed,split='test',test_evaluated=True,
             manifest_sha256=m['manifest_sha256'],checkpoint_sha256=job['checkpoint_sha256'],
             prediction_sha256=sha256(destination),strict_vocab50=strict,full_vocabulary=full))
        del ds,loader,cp,model,encoded,arrays
        gc.collect()
        torch.cuda.empty_cache()


def validate_chunk(chunk,m,dataset,seed,groups):
    if (chunk['manifest_sha256']!=m['manifest_sha256'] or chunk['dataset']!=dataset or chunk['seed']!=seed
            or chunk['conditions']!=['A','C'] or chunk['split']!='test' or not chunk['test_evaluated']
            or chunk['selected_weight']!=.1 or [r['group'] for r in chunk['records']]!=[g['group'] for g in groups]):
        raise ValueError('Test chunk provenance or population mismatch')
    for group,record in zip(groups,chunk['records']):
        validate_result(group,record['result'])


def decode(m):
    completed = 0
    for dataset, info in m['datasets'].items():
        scorer = embed = bank = None
        cache = OUT/dataset/'lm_cache.pkl'
        for seed in (0,1,2):
            groups = info['groups']
            root = seed_root(dataset,seed)
            with np.load(root/'neural_predictions.npz') as encoded:
                predictions = {h:encoded[f'k{h}'] for h in range(3)}
                lookup = {str(e):i for i,e in enumerate(encoded['event_ids'])}
            width = m['settings']['execution']['segment_batch_size']
            records, sources = [], []
            for start in range(0,len(groups),width):
                batch = groups[start:start+width]
                path = root/'chunks'/f'chunk-{start//width:04d}.json'
                if path.exists():
                    chunk = read(path)
                    validate_chunk(chunk,m,dataset,seed,batch)
                else:
                    if scorer is None:
                        status(stage='loading_sentence_models',dataset=dataset,seed=seed)
                        scorer,embed,bank = load_lm(m,dataset)
                        if cache.exists():
                            scorer.load_cache(cache)
                    started, last = time.perf_counter(), [0.]
                    def progress(arrival,counts):
                        now = time.perf_counter()
                        if now-last[0] >= 30:
                            status(stage='sentence_decoding',dataset=dataset,seed=seed,chunk=start//width,
                                arrival=arrival,longest_segment=max(counts),completed_segment_seed_units=completed)
                            last[0] = now
                    cases = cases_for_groups(batch,predictions,lookup,weight=.1,branch='open')
                    d = m['decoder']
                    results = replay_many(cases,info['vocabulary'],bank,scorer,embed,conditions='AC',progress=progress,
                        candidate_count=d['candidates'],beam_width=d['beam_width'],neural_scale=d['neural_scale'])
                    chunk = dict(manifest_sha256=m['manifest_sha256'],dataset=dataset,seed=seed,conditions=['A','C'],
                        selected_weight=.1,split='test',test_evaluated=True,
                        records=[dict(group=g['group'],result=r) for g,r in zip(batch,results)],seconds=time.perf_counter()-started)
                    validate_chunk(chunk,m,dataset,seed,batch)
                    save(path,chunk)
                    scorer.save_cache(cache)
                records.extend(chunk['records'])
                sources.append(dict(path=str(path.relative_to(PROJECT_ROOT)),sha256=sha256(path)))
                completed += len(batch)
                status(stage='chunk_complete',dataset=dataset,seed=seed,chunk=start//width,
                    completed_segment_seed_units=completed,total_segment_seed_units=sum(len(i['groups']) for i in m['datasets'].values())*3)
            payload = dict(manifest_sha256=m['manifest_sha256'],dataset=dataset,seed=seed,conditions=['A','C'],
                selected_weight=.1,split='test',test_evaluated=True,records=records,status='completed',sources=sources)
            validate_chunk(payload,m,dataset,seed,groups)
            save(root/'main.json',payload)
            del predictions
        del scorer,embed,bank
        gc.collect()
        torch.cuda.empty_cache()


def run():
    m = manifest()
    RUNTIME.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (RUNTIME/'worker.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        torch.set_num_threads(4)
        set_seed(0)
        status(state='running',stage='starting',manifest_sha256=m['manifest_sha256'],started_unix=time.time())
        try:
            encode_all(m)
            from braindecoding.unified_heldout_test_v2_results import export_word, export_sentence
            export_word(m)
            decode(m)
            neural.check_hashes(m['source_sha256'])
            export_sentence(m)
            status(state='completed',stage='completed',completed_unix=time.time())
        except BaseException as error:
            status(state='failed',error=f'{type(error).__name__}: {error}')
            raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    args = parser.parse_args()
    prepare() if args.command=='prepare' else run()

# --- Merged from unified_heldout_test_v2_results.py ---
"""Report a complete, frozen held-out main experiment without selecting runs."""
from collections import defaultdict
import time

import numpy as np

from braindecoding.unified_sentence_v2_artifacts import csv_write
from braindecoding.unified_sentence_v2_results import metric_row, paired_block_statistics, holm
from braindecoding.unified_sentence_v2_revision_gain import transitions
from braindecoding.unified_sentence_v2_revision_gain_full_results import aggregate_seeds

LABELS = {'pallier2025':'Pallier2025','chineseeeg2_littleprince':'ChineseEEG-2'}
EXPORT = PROJECT_ROOT/'data/results/exports'


def word_records(m):
    rows, sources = [], []
    for dataset,info in m['datasets'].items():
        for seed in (0,1,2):
            path = seed_root(dataset,seed)/'word_metrics.json'
            data = read(path)
            assert data['manifest_sha256']==m['manifest_sha256'] and data['test_evaluated'] and data['split']=='test'
            assert data['prediction_sha256']==sha256(path.with_name('neural_predictions.npz'))
            sources.append(dict(path=str(path.relative_to(PROJECT_ROOT)),sha256=sha256(path)))
            for horizon,metrics in data['strict_vocab50'].items():
                for population,values in metrics.items():
                    rows.append(dict(dataset=dataset,seed=seed,horizon=int(horizon),population=population,
                        **{key:values[key] for key in ['candidate_count','query_count','target_count',
                            'macro_recall_at_1','macro_recall_at_10','micro_recall_at_1','micro_recall_at_10']}))
    return rows,sources


def export_word(m):
    rows,sources = word_records(m)
    summary, gains = [], []
    for dataset in m['datasets']:
        for population in ('all_targets','common_targets'):
            for h in range(3):
                group = [r for r in rows if (r['dataset'],r['population'],r['horizon'])==(dataset,population,h)]
                assert len(group)==3 and all(r['candidate_count']==50 for r in group)
                result = dict(dataset=dataset,population=population,horizon=h,n_seeds=3,query_count=group[0]['query_count'],candidate_count=50)
                for metric in ('macro_recall_at_1','macro_recall_at_10','micro_recall_at_1','micro_recall_at_10'):
                    values = [r[metric] for r in group]
                    result[metric+'_mean'],result[metric+'_sd'] = float(np.mean(values)),float(np.std(values,ddof=1))
                summary.append(result)
        for seed in (0,1,2):
            group = {r['horizon']:r for r in rows if r['dataset']==dataset and r['seed']==seed and r['population']=='common_targets'}
            gains.append(dict(dataset=dataset,seed=seed,macro_top10_k2_minus_k0=group[2]['macro_recall_at_10']-group[0]['macro_recall_at_10']))
    csv_write(EXPORT/'unified_heldout_test_v2_word_by_seed.csv',rows)
    csv_write(EXPORT/'unified_heldout_test_v2_word_summary.csv',summary)
    csv_write(EXPORT/'unified_heldout_test_v2_word_gain.csv',gains)
    save(REPORT/'word_results.json',dict(status='completed_word_test',manifest_sha256=m['manifest_sha256'],
        split='test',test_evaluated=True,summary=summary,gains=gains,sources=sources))
    lines = ['# 测试集词级主实验','', '六个固定滚动检查点，保留全部 50 个候选；延迟比较使用共同查询。', '',
             '| 数据集 | K0 | K1 | K2 | K2−K0（百分点） |','|---|---:|---:|---:|---:|']
    for dataset in m['datasets']:
        group = {r['horizon']:r for r in summary if r['dataset']==dataset and r['population']=='common_targets'}
        score = [group[h]['macro_recall_at_10_mean']*100 for h in range(3)]
        lines.append(f"| {LABELS[dataset]} | {score[0]:.2f} | {score[1]:.2f} | {score[2]:.2f} | {score[2]-score[0]:+.2f} |")
    lines += ['', '表中为三个种子的 Macro Top-10 均值（%）；标准差和逐种子结果见正式导出。本表不作句子收益或机会基线推断。']
    (REPORT/'word_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def export_sentence(m):
    datasets, all_rows, all_changes, all_subjects, sources = {},[],[],[],[]
    for dataset,info in m['datasets'].items():
        rows,changes,examples = [],[],[]
        for seed in (0,1,2):
            path = seed_root(dataset,seed)/'main.json'
            payload = read(path)
            assert payload['status']=='completed'
            validate_chunk(payload,m,dataset,seed,info['groups'])
            reconstructed = []
            for source in payload['sources']:
                chunk_path = PROJECT_ROOT/source['path']
                assert sha256(chunk_path)==source['sha256']
                reconstructed.extend(read(chunk_path)['records'])
            assert reconstructed==payload['records']
            sources.append(dict(path=str(path.relative_to(PROJECT_ROOT)),sha256=sha256(path)))
            for group,record in zip(info['groups'],payload['records']):
                result = record['result']
                identity = dict(dataset=dataset,seed=seed,subject=group['subject'],group=group['group'],
                                material=group['material'],block=group['block'],scope=group['scope'])
                for condition in ('A','C'):
                    rows.append(dict(**identity,condition=condition,words=len(group['reference']),
                        **metric_row(group['reference'],result['final'][condition],info['language']=='zh')))
                change = dict(dataset=dataset,seed=seed,material=group['material'],block=group['block'],scope=group['scope'],
                              **transitions(group,result))
                changes.append(change)
                if seed==0:
                    examples.append(dict(group=group['group'],subject=group['subject'],seed=seed,
                                         reference=group['reference'],changes=change,**result))
        aggregate = aggregate_seeds(rows,changes)
        by_condition = {r['condition']:r for r in aggregate['summary']}
        paired,metadata = defaultdict(dict),{}
        for row in rows:
            key = row['seed'],row['subject'],row['group']
            paired[key][row['condition']] = row['sentence_bleu1_mean']
            metadata[key] = {k:row[k] for k in ('seed','subject','group','block','scope')}
        differences = [dict(**metadata[key],difference=v['C']-v['A']) for key,v in paired.items()]
        stats = paired_block_statistics(differences,'difference',m['settings']['statistics'])
        effect = by_condition['C']['sentence_bleu1_mean']-by_condition['A']['sentence_bleu1_mean']
        assert np.isclose(effect,stats['effect'],atol=1e-12)
        chosen = []
        for metric,label in [('rolling_repaired','most_C_repairs'),('rolling_harmed','most_C_harms')]:
            chosen.append(dict(selection=label,**sorted(examples,key=lambda e:(-e['changes'][metric],e['group']))[0]))
        datasets[dataset] = dict(counts=info['counts'],**aggregate,primary_statistics=stats,examples=chosen)
        all_rows.extend(rows); all_changes.extend(changes)
        all_subjects.extend(dict(dataset=dataset,**row) for row in aggregate['by_subject_seed'])
    for info,p in zip(datasets.values(),holm([i['primary_statistics']['p_one_sided'] for i in datasets.values()])):
        info['primary_statistics']['p_holm'] = float(p)
    expected = sum(i['counts']['groups'] for i in m['datasets'].values())*3
    assert len(all_rows)==expected*2 and len(all_changes)==expected
    report = dict(status='completed_heldout_test',manifest_sha256=m['manifest_sha256'],split='test',test_evaluated=True,
                  parameters_selected_on='development_validation',datasets=datasets,sources=sources,
                  completed_unix=time.time(),segment_seed_units=expected,conditions=['A','C'],seeds=[0,1,2])
    for name,values in [('sentence_by_group',all_rows),('sentence_by_subject_seed',all_subjects),('sentence_repairs',all_changes)]:
        csv_write(EXPORT/f'unified_heldout_test_v2_{name}.csv',values)
    save(REPORT/'results.json',report)
    save(EXPORT/'unified_heldout_test_v2_sentence_summary.json',report)
    lines = ['# 冻结方案下的测试集主实验','',
        '测试材料和神经模型训练材料分开；模型、语言权重和搜索设置均由开发阶段固定。本轮不训练或调参。','',
        '| 数据集 | A BLEU-1 (%) | C BLEU-1 (%) | C−A（百分点） | 95% 材料区间 | Holm p |',
        '|---|---:|---:|---:|---|---:|']
    for dataset,info in datasets.items():
        values = {r['condition']:r for r in info['summary']}; stats = info['primary_statistics']
        lines.append(f"| {LABELS[dataset]} | {values['A']['sentence_bleu1_mean']*100:.2f} | {values['C']['sentence_bleu1_mean']*100:.2f} | {stats['effect']*100:+.2f} | [{stats['ci_lower']*100:.2f}, {stats['ci_upper']*100:.2f}] | {stats['p_holm']:.6f} |")
    lines += ['', '区间与检验以材料块配对，保留块内全部受试者和种子；两个数据集作 Holm 校正。该比较检验回修相对固定输出的收益，不检验机会基线。', '',
              '| 数据集 | A WER (%) | C WER (%) | A 对齐词准确率 (%) | C 对齐词准确率 (%) |','|---|---:|---:|---:|---:|']
    for dataset,info in datasets.items():
        v = {r['condition']:r for r in info['summary']}
        lines.append(f"| {LABELS[dataset]} | {v['A']['wer']*100:.2f} | {v['C']['wer']*100:.2f} | {v['A']['aligned_word_accuracy']*100:.2f} | {v['C']['aligned_word_accuracy']*100:.2f} |")
    lines += ['', '逐种子、受试者、片段及修正计数见正式导出。真实回修和误改示例见 results.json，按种子 0 的改对／误改最多片段选取，不能用于估计典型输出质量。']
    (REPORT/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(REPORT/'audit.json',dict(status='verified',manifest_sha256=m['manifest_sha256'],
        report_sha256=sha256(REPORT/'results.json'),sources=sources,segment_seed_units=expected,test_evaluated=True))
    return report
