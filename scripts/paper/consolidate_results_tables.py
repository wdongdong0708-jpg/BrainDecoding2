"""Generate the consolidated result tables from verified frozen test exports.

Only LaTeX presentation assets and an audit report are written. Author prose,
scientific exports, model parameters, decoding and statistical tests are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.paper import polish_main_assets as presentation

EXPORT = ROOT / 'data/results/exports'
TABLES = ROOT / 'paper/tables'
AUDIT = ROOT / 'data/results/audits/results_consolidation_20261010'
METRICS = [
    ('macro_recall_at_10', 'Macro Top-$10$'),
    ('macro_recall_at_1', 'Macro Top-$1$'),
    ('micro_recall_at_10', 'Micro Top-$10$'),
    ('micro_recall_at_1', 'Micro Top-$1$'),
]
CONDITIONS = [
    ('full_context', '完整上下文'),
    *[(f'local_r{k}', f'局部半径 {k}') for k in [0, 1, 2, 4, 8]],
    ('independent_k0', '独立 K0'), ('independent_k1', '独立 K1'),
    ('rolling_k0', '共享 K0'), ('rolling_k1', '共享 K1'), ('rolling_k2', '共享 K2'),
]


def complete_word_table(context):
    rolling = pd.read_csv(EXPORT / 'unified_heldout_test_v2_word_by_seed.csv')
    rolling['condition'] = rolling.horizon.map(lambda h: f'rolling_k{int(h)}')
    data = pd.concat([context, rolling], ignore_index=True)
    header = '数据集 & 输入/模型 & 查询 & ' + ' & '.join(
        r'\shortstack{' + label.replace(' ', r'\\', 1) + '}' for _, label in METRICS
    ) + r' \\'
    lines = [
        '% Presentation only: formal test exports; consolidate_results_tables_test_v2.py.',
        r'\begingroup', r'\papertablestyle',
        r'\begin{longtable}{@{}lllrrrr@{}}',
        r'\caption{两个数据集的完整 $50$ 词候选检索结果（\%，三个种子均值 $\pm$ 样本标准差）。全部与共同查询分别列出；独立 K2 对应局部半径 $2$ 的共同查询行。}',
        r'\label{tab:word_complete_results}\label{tab:word_complete_pallier}\label{tab:word_complete_chinese}\\',
        r'\toprule', header, r'\midrule', r'\endfirsthead',
        r'\multicolumn{7}{l}{表~\ref{tab:word_complete_results}（续）}\\',
        r'\toprule', header, r'\midrule', r'\endhead',
        r'\midrule\multicolumn{7}{r}{续下页}\\\endfoot',
        r'\bottomrule\endlastfoot',
    ]
    facts = []
    for index, (dataset, label) in enumerate(presentation.DATASETS):
        if index:
            lines.append(r'\midrule')
        for condition, display in CONDITIONS:
            for pop, population_label in [('all_targets', '全部'), ('common_targets', '共同')]:
                part = data[(data.dataset == dataset) & (data.condition == condition) & (data.population == pop)]
                assert len(part) == 3 and set(part.seed) == {0, 1, 2}
                assert (part.candidate_count == 50).all()
                values = {key: {'mean': float(part[key].mean()), 'sample_sd': float(part[key].std(ddof=1))}
                          for key, _ in METRICS}
                cells = [label if pop == 'all_targets' else '',
                         display if pop == 'all_targets' else '', population_label]
                cells += [f'${100 * values[key]["mean"]:.2f}\\pm{100 * values[key]["sample_sd"]:.2f}$'
                          for key, _ in METRICS]
                lines.append(' & '.join(cells) + r' \\')
                facts.append({'dataset': dataset, 'condition': condition, 'population': pop, 'metrics': values})
            lines.append(r'\addlinespace')
    lines += [r'\end{longtable}', r'\endgroup', '']
    return '\n'.join(lines), facts


def main():
    context, _, sentence, core, _, means, _, comparisons, _, full_vocab, sources = presentation.inputs()
    native_core, core_facts = presentation.core_table(core)
    word_table, word_facts = complete_word_table(context)
    tables = {
        'word_core_paired_effects_v2_test.tex': native_core,
        'full_vocab_comparison_v2_test.tex': presentation.vocabulary_table(full_vocab),
        'sentence_paired_effects_v2_test.tex': presentation.paired_table(sentence, comparisons),
        'sentence_wait_budget_metrics_v2_test.tex': presentation.waiting_table(means),
        'word_complete_metrics_two_datasets_v2_test.tex': word_table,
    }
    for name, contents in tables.items():
        (TABLES / name).write_text(contents, encoding='utf-8')
    AUDIT.mkdir(parents=True, exist_ok=True)
    record = {
        'status': 'verified_presentation_only',
        'sources': sources,
        'word_core_values': core_facts,
        'merged_word_table_values': word_facts,
        'sequence_metrics': means.astype(object).where(pd.notna(means), None).to_dict('records'),
        'B_wait_rule': 'same output commitment time as C; historical representations remain K0',
        'paired_table': {'comparisons': 8, 'datasets': 2, 'original_intervals_reused': True},
        'generated_tables': {name: presentation.sha(TABLES / name) for name in tables},
        'new_training_decoding_or_resampling': False,
    }
    (AUDIT / 'tables.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': record['status'], 'tables': len(tables), 'word_rows': len(word_facts),
                      'paired_comparisons': 8, 'conditions_in_absolute_table': sorted(means.condition.unique())},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
