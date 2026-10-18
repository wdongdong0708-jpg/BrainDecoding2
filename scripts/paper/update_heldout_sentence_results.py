"""Build manuscript assets exclusively from the audited held-out v2 exports."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / 'data/results/exports'
STUDY = ROOT / 'data/results/studies/unified_heldout_test_v2'
FIG = ROOT / 'paper/figures'
TABLE = ROOT / 'paper/tables'
AUDIT = ROOT / 'data/results/audits/manuscript_heldout_test_v2'
DATASETS = [('pallier2025', 'Pallier2025'), ('chineseeeg2_littleprince', 'ChineseEEG-2')]
SEEDS = [0, 1, 2]
METRICS = ['sentence_bleu1_mean', 'wer', 'aligned_word_accuracy']


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def close(a, b):
    assert np.isclose(a, b, atol=1e-12, rtol=0), (a, b)


def verify():
    summary = load(EXPORT / 'unified_heldout_test_v2_sentence_summary.json')
    audit = load(STUDY / 'audit.json')
    manifest = load(ROOT / 'data/manifests/rolling_sentence/heldout_test_v2.json')
    identity = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != 'manifest_sha256'},
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert identity == manifest['manifest_sha256'] == summary['manifest_sha256'] == audit['manifest_sha256']
    assert summary['status'] == 'completed_heldout_test'
    assert summary['split'] == 'test' and summary['test_evaluated'] is True
    assert summary['seeds'] == SEEDS and summary['conditions'] == ['A', 'C']
    assert summary['parameters_selected_on'] == 'development_validation'
    assert audit['status'] == 'verified'
    assert digest(STUDY / 'results.json') == audit['report_sha256']
    assert digest(EXPORT / 'unified_heldout_test_v2_sentence_summary.json') == audit['report_sha256']
    assert len(audit['sources']) == 6
    for source in audit['sources']:
        assert digest(ROOT / source['path']) == source['sha256']
    subjects = pd.read_csv(EXPORT / 'unified_heldout_test_v2_sentence_by_subject_seed.csv')
    groups = pd.read_csv(EXPORT / 'unified_heldout_test_v2_sentence_by_group.csv')
    repairs = pd.read_csv(EXPORT / 'unified_heldout_test_v2_sentence_repairs.csv')
    word = pd.read_csv(EXPORT / 'unified_heldout_test_v2_word_by_seed.csv')
    word_summary = pd.read_csv(EXPORT / 'unified_heldout_test_v2_word_summary.csv')
    word_gain = pd.read_csv(EXPORT / 'unified_heldout_test_v2_word_gain.csv')
    word_report = load(STUDY / 'word_results.json')
    assert word_report['manifest_sha256'] == identity and word_report['test_evaluated'] is True
    for source in word_report['sources']:
        assert digest(ROOT / source['path']) == source['sha256']
    assert len(repairs) == summary['segment_seed_units'] == audit['segment_seed_units'] == 8244
    assert len(groups) == 2 * len(repairs)
    assert not subjects.duplicated(['dataset', 'seed', 'subject', 'condition']).any()
    assert not groups.duplicated(['dataset', 'seed', 'group', 'condition']).any()
    assert not repairs.duplicated(['dataset', 'seed', 'group']).any()
    for dataset, values in summary['datasets'].items():
        counts = values['counts']
        s = subjects[subjects.dataset == dataset]
        g = groups[groups.dataset == dataset]
        r = repairs[repairs.dataset == dataset]
        expected_groups = {v['group'] for v in manifest['datasets'][dataset]['groups']}
        for seed in SEEDS:
            for condition in ('A', 'C'):
                part = s[(s.seed == seed) & (s.condition == condition)]
                assert len(part) == counts['subjects']
                assert part.segments.sum() == counts['groups'] and part.words.sum() == counts['events']
                part_groups = g[(g.seed == seed) & (g.condition == condition)]
                assert set(part_groups.group) == expected_groups
        for row in values['summary']:
            part = s[s.condition == row['condition']]
            for metric in METRICS:
                close(part.groupby('seed')[metric].mean().mean(), row[metric])
        for row in values['by_seed']:
            part = s[(s.seed == row['seed']) & (s.condition == row['condition'])]
            for metric in METRICS:
                close(part[metric].mean(), row[metric])
        means = {v['condition']: v for v in values['summary']}
        close(means['C']['sentence_bleu1_mean'] - means['A']['sentence_bleu1_mean'], values['primary_statistics']['effect'])
        stats = values['primary_statistics']
        assert stats['blocks'] == counts['blocks'] and stats['subjects'] == counts['subjects'] and stats['seeds'] == 3
        close(stats['p_holm'], min(1, 2 * stats['p_one_sided']))
        for field in ['words', 'eligible', 'paired_repaired', 'paired_harmed', 'rolling_repaired', 'rolling_harmed']:
            assert int(r[field].sum()) == values['revisions'][field]
        pooled = r.groupby(['seed', 'subject']).sum(numeric_only=True)
        for field in ['paired_repaired', 'paired_harmed', 'rolling_repaired', 'rolling_harmed']:
            denominator = 'eligible' if field.startswith('rolling_') else 'words'
            close((pooled[field] / pooled[denominator]).groupby('seed').mean().mean(), values['revisions'][field + '_rate'])
        assert values['revisions']['paired_repaired'] - values['revisions']['paired_harmed'] == values['revisions']['net_correct_words']
        for horizon in range(3):
            part = word[(word.dataset == dataset) & (word.horizon == horizon) & (word.population == 'common_targets')]
            assert set(part.seed) == set(SEEDS) and (part.candidate_count == 50).all()
            assert (part.query_count == counts['strict50_common']).all()
            target = word_summary[(word_summary.dataset == dataset) & (word_summary.horizon == horizon) & (word_summary.population == 'common_targets')].iloc[0]
            close(part.macro_recall_at_10.mean(), target.macro_recall_at_10_mean)
            close(part.macro_recall_at_10.std(ddof=1), target.macro_recall_at_10_sd)
        pivot = word[(word.dataset == dataset) & (word.population == 'common_targets')].pivot(index='seed', columns='horizon', values='macro_recall_at_10')
        for seed in SEEDS:
            close(pivot.loc[seed, 2] - pivot.loc[seed, 0], word_gain[(word_gain.dataset == dataset) & (word_gain.seed == seed)].iloc[0].macro_top10_k2_minus_k0)
    inputs = [EXPORT / f'unified_heldout_test_v2_{name}' for name in [
        'sentence_summary.json', 'sentence_by_group.csv', 'sentence_by_subject_seed.csv', 'sentence_repairs.csv',
        'word_by_seed.csv', 'word_summary.csv', 'word_gain.csv']]
    return summary, word_summary, {str(p.relative_to(ROOT)): digest(p) for p in inputs}


def latex(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def table(name, columns, header, rows):
    text = ['% Generated from audited reports/exports/unified_heldout_test_v2_*.',
            r'\begin{tabular}{@{}' + columns + r'@{}}', r'\toprule',
            ' & '.join(header) + r' \\', r'\midrule']
    text += [' & '.join(map(str, row)) + r' \\' for row in rows]
    text += [r'\bottomrule', r'\end{tabular}']
    path = TABLE / name
    path.write_text('\n'.join(text) + '\n', encoding='utf-8')
    return path


def build():
    summary, word, sources = verify()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    rows, repair_rows, examples, word_rows, statistics_rows = [], [], [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.7), sharey=True, layout='constrained')
    for ax, (dataset, label) in zip(axes, DATASETS):
        data = summary['datasets'][dataset]
        for item in data['summary']:
            rows.append([label if item['condition'] == 'A' else '', item['condition'],
                         *[f'{item[m] * 100:.2f}' for m in METRICS]])
        r = data['revisions']
        repair_rows.append([label, r['eligible'], r['rolling_repaired'], r['rolling_harmed'],
                            f"{r['rolling_repaired_rate'] * 100:.2f}", f"{r['rolling_harmed_rate'] * 100:.2f}"])
        values = np.array([[next(row['sentence_bleu1_mean'] for row in data['by_seed'] if row['seed'] == seed and row['condition'] == c) * 100
                            for c in ('A', 'C')] for seed in SEEDS])
        for line in values:
            ax.plot([0, 1], line, color='#b9c0c5', marker='o', lw=.8, markersize=3)
        ax.plot([0, 1], values.mean(axis=0), color='#24638A', marker='o', lw=2.1)
        ax.set(xticks=[0, 1], xticklabels=['A: fixed', 'C: revision'], ylim=(0, 20), xlim=(-.25, 1.25), title=label)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        stat = data['primary_statistics']
        ax.text(.04, .95, f"Gain {stat['effect'] * 100:+.2f} pp\n95% CI [{stat['ci_lower'] * 100:.2f}, {stat['ci_upper'] * 100:.2f}]", transform=ax.transAxes, va='top', fontsize=9)
        statistics_rows.append([label, data['counts']['groups'], stat['blocks'], f"{stat['effect'] * 100:+.2f}",
                                f"$[{stat['ci_lower'] * 100:.2f}, {stat['ci_upper'] * 100:.2f}]$", f"{stat['p_holm']:.4f}"])
        for selection, tag in [('most_C_repairs', '改对'), ('most_C_harms', '误改')]:
            example = next(e for e in data['examples'] if e['selection'] == selection)
            trace = {e['arrival']: e['C'] for e in example['trace']}
            initial = [trace[i][i] for i in range(len(example['reference']))]
            final = example['final']['C']
            positions = [i for i, ref in enumerate(example['reference'][:-1]) if
                         (initial[i] != ref and final[i] == ref if tag == '改对' else initial[i] == ref and final[i] != ref)]
            assert positions, (dataset, tag)
            position = positions[0]
            assert example['seed'] == 0
            examples.append([label, latex(example['subject']), tag, position + 1, latex(example['reference'][position]),
                             *[latex(trace[position + offset][position]) if position + offset in trace else '--' for offset in (0, 1, 2)]])
        w = word[(word.dataset == dataset) & (word.population == 'common_targets')].set_index('horizon')
        word_rows.append([label, int(w.loc[0, 'query_count']),
                          *[f"${w.loc[k, 'macro_recall_at_10_mean'] * 100:.2f} \\pm {w.loc[k, 'macro_recall_at_10_sd'] * 100:.2f}$" for k in range(3)],
                          f"${(w.loc[2, 'macro_recall_at_10_mean'] - w.loc[0, 'macro_recall_at_10_mean']) * 100:+.2f}$"])
    axes[0].set_ylabel('Mean segment BLEU-1 (%)')
    assets = []
    for extension in ('pdf', 'svg', 'png'):
        path = FIG / ('sentence_revision_v2_test.' + extension)
        fig.savefig(path, dpi=220, bbox_inches='tight')
        assets.append(path)
    plt.close(fig)
    assets += [table('sentence_v2_test_metrics.tex', 'llrrr', ['数据集', '条件', 'BLEU-1', 'WER', '对齐词准确率'], rows),
               table('sentence_v2_test_statistics.tex', 'lrrrrr', ['数据集', '片段数', '材料块', r'$C-A$', r'$95\%$ 区间', r'Holm $p$'], statistics_rows),
               table('sentence_v2_test_repairs.tex', 'lrrrrr', ['数据集', '可回修词位', '改对数', '误改数', r'改对率（\%）', r'误改率（\%）'], repair_rows),
               table('sentence_v2_test_examples.tex', 'lllrlrrr', ['数据集', '受试者', '变化', '词位', '参考词', r'$t=i$', r'$t=i+1$', r'$t=i+2$'], examples),
               table('rolling_v2_test_metrics.tex', 'lrrrrr', ['数据集', '共同查询数', 'K0', 'K1', 'K2', r'$K2-K0$'], word_rows)]
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / 'assets.json').write_text(json.dumps({'status': 'verified', 'split': 'test', 'manifest_sha256': summary['manifest_sha256'],
                                                 'segment_seed_units': summary['segment_seed_units'], 'source_exports': sources,
                                                 'assets': {str(p.relative_to(ROOT)): digest(p) for p in assets}}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Verified all six test runs, reconciled subject/seed means and repair rates; wrote test figures and five tables.')


if __name__ == '__main__':
    build()
