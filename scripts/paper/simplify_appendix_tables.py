"""Generate compact appendix tables from frozen manifests and verified exports.

This changes presentation only. No training, decoding or statistical estimation
is run. The comprehensive source exports and earlier vector assets are retained.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

from scripts.paper.update_heldout_sentence_results import verify

DATASETS = [('pallier2025', 'Pallier2025'),
            ('chineseeeg2_littleprince', 'ChineseEEG-2')]
AUDIT = ROOT / 'data/results/audits/appendix_rewrite_20261011'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def table(columns, header, rows, break_before=()):
    lines = [
        '% Presentation only: simplify_appendix_tables_test_v2.py; verified frozen sources.',
        r'\begingroup', r'\papertablestyle',
        rf'\begin{{tabular}}{{@{{}}{columns}@{{}}}}', r'\toprule',
        ' & '.join(header) + r' \\', r'\midrule',
    ]
    for index, row in enumerate(rows):
        if index in break_before:
            lines.append(r'\addlinespace')
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
    return '\n'.join(lines)


def main():
    sentence, _, sources = verify()
    sources = dict(sources)
    manifest_rel = 'data/manifests/rolling_sentence/heldout_test_v2.json'
    config_rel = 'configs/training/rolling_revision_protocol.yaml'
    vocabulary_rel = 'data/results/exports/full_vocabulary_context_test_v2_summary.json'
    wait_rel = 'data/results/exports/full_context_sentence_test_v2_input_wait_summary.csv'
    timing_rel = 'data/results/exports/full_context_sentence_test_v2_summary.json'
    timing_manifest_rel = 'data/manifests/rolling_sentence/full_context_sentence_test_v2.json'
    manifest = read(manifest_rel)
    config = yaml.safe_load((ROOT / config_rel).read_text(encoding='utf-8'))
    vocabulary = read(vocabulary_rel)
    timing = read(timing_rel)
    timing_manifest = read(timing_manifest_rel)
    assert manifest['manifest_sha256'] == sentence['manifest_sha256']
    assert timing['status'] == 'completed' and timing['original_AC_unchanged']
    assert timing['manifest_sha256'] == timing_manifest['manifest_sha256']
    assert timing_manifest['parent_manifest_sha256'] == sentence['manifest_sha256']
    assert timing['exports'][wait_rel] == sha(ROOT / wait_rel)
    for item in timing['sources']:
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
        sources[item['path']] = item['sha256']
    for rel, expected in vocabulary['exports'].items():
        assert sha(ROOT / rel) == expected, rel
    for rel in [manifest_rel, config_rel, vocabulary_rel, wait_rel, timing_rel, timing_manifest_rel]:
        sources[rel] = sha(ROOT / rel)

    populations, coverage = {}, {}
    for dataset, _ in DATASETS:
        info = manifest['datasets'][dataset]
        counts = sentence['datasets'][dataset]['counts']
        adapter = config['adapters'][dataset]
        row = next(r for r in vocabulary['summary']
                   if r['dataset'] == dataset and r['population'] == 'all_words'
                   and r['metric'] == 'macro_recall_at_10')
        assert row['candidate_count'] == len(info['vocabulary']) == adapter['expected_train_vocabulary']
        assert row['query_count'] == counts['events']
        coverage[dataset] = {
            'test_words': row['query_count'],
            'covered_words': row['query_count'] - row['oov_queries'],
            'coverage_percent': 100 * (1 - row['oov_queries'] / row['query_count']),
        }
        populations[dataset] = [
            str(len(info['vocabulary'])),
            f"{coverage[dataset]['coverage_percent']:.2f}\\%",
            str(adapter['expected_legacy_word_candidates']),
            str(adapter['expected_legacy_common_candidates']),
            str(len(info['vocabulary50'])),
            *[str(counts[key]) for key in ['strict50_all', 'strict50_common', 'groups', 'events', 'blocks']],
        ]
    names = ['训练词库', '训练词库覆盖率', '验证选模候选：词模型', '验证选模候选：上下文模型',
             '测试词检索候选', '局部范围测试查询', '可见状态共同查询',
             '序列测试片段', '序列参考词位', '序列材料块']
    population_rows = [[name, *[populations[d][i] for d, _ in DATASETS]]
                       for i, name in enumerate(names)]
    population_table = table('lrr', ['项目', *[label for _, label in DATASETS]],
                             population_rows, break_before=(5,))

    with (ROOT / wait_rel).open(encoding='utf-8-sig', newline='') as stream:
        waits = list(csv.DictReader(stream))
    wait_rows, wait_facts = [], []
    for dataset, label in DATASETS:
        for index, (strategy, display) in enumerate([('K0', 'A / K0'), ('C_K2', 'C / K2'), ('F', 'F')]):
            row = next(r for r in waits if r['dataset'] == dataset
                       and r['population'] == 'all_events' and r['strategy'] == strategy)
            assert int(row['queries']) == sentence['datasets'][dataset]['counts']['events']
            wait_rows.append([label if index == 0 else '', display, row['queries'],
                              *[f'{float(row[key]):.3f}' for key in
                                ['mean_input_wait_s', 'median_input_wait_s',
                                 'p90_input_wait_s', 'mean_saved_vs_f_s']]])
            wait_facts.append(row)
    waiting_table = table('llrrrrr',
                          ['数据集', '策略', '词位数', '均值 (s)', '中位 (s)', 'P90 (s)',
                           r'\papertablehead{输入等待节省}{较 F (s)}'],
                          wait_rows, break_before=(3,))
    tables = {'test_population_v2.tex': population_table,
              'input_wait_complete_v2_test.tex': waiting_table}
    for name, contents in tables.items():
        (ROOT / 'paper/tables' / name).write_text(contents, encoding='utf-8')
    AUDIT.mkdir(parents=True, exist_ok=True)
    report = {'status': 'verified_presentation_only', 'sources': sources,
              'coverage': coverage, 'population_rows': population_rows,
              'waiting_population': 'all_events', 'wait_rows': wait_facts,
              'generated_tables': {name: sha(ROOT / 'paper/tables' / name) for name in tables},
              'new_training_decoding_or_resampling': False}
    (AUDIT / 'tables.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                                      + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'tables': len(tables),
                      'population_rows': len(population_rows), 'wait_rows': len(wait_rows),
                      'coverage': coverage}, ensure_ascii=False))


if __name__ == '__main__':
    main()
