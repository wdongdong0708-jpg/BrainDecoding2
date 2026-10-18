"""Replace word result assets with complete, audited test-only exports."""
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
FIG = ROOT / 'paper/figures'
TABLE = ROOT / 'paper/tables'
AUDIT = ROOT / 'data/results/audits/manuscript_test_only_v2'
DATASETS = [('pallier2025', 'Pallier2025'), ('chineseeeg2_littleprince', 'ChineseEEG-2')]
METRICS = ('macro_recall_at_1', 'macro_recall_at_10', 'micro_recall_at_1', 'micro_recall_at_10')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    report = load(ROOT / 'data/results/studies/unified_context_heldout_test_v2/report.json')
    manifest = load(ROOT / 'data/manifests/rolling_sentence/context_heldout_test_v2.json')
    identity = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != 'manifest_sha256'},
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert report['manifest_sha256'] == manifest['manifest_sha256'] == identity
    assert report['status'] == 'completed' and report['split'] == 'test' and report['test_evaluated'] is True
    assert report['completed_jobs'] == report['planned_jobs'] == 48
    for source in report['sources']:
        assert sha(ROOT / source['path']) == source['sha256']
    for path, expected in report['exports'].items():
        assert sha(ROOT / path) == expected
    by_seed = pd.read_csv(EXPORT / 'unified_context_heldout_test_v2_context_by_seed.csv')
    context = pd.read_csv(EXPORT / 'unified_context_heldout_test_v2_context_summary.csv')
    paired = pd.read_csv(EXPORT / 'unified_context_heldout_test_v2_independent_vs_rolling_by_seed.csv')
    rolling = pd.read_csv(EXPORT / 'unified_heldout_test_v2_word_by_seed.csv')
    assert len(by_seed) == 96 and len(context) == 32 and len(paired) == 18
    assert not by_seed.duplicated(['dataset', 'seed', 'condition', 'population']).any()
    assert set(by_seed.split) == set(context.split) == set(paired.split) == {'test'}
    assert (by_seed.candidate_count == 50).all() and (context.n_seeds == 3).all()
    for _, row in context.iterrows():
        g = by_seed[(by_seed.dataset == row.dataset) & (by_seed.condition == row.condition) & (by_seed.population == row.population)]
        assert set(g.seed) == {0, 1, 2}
        for metric in METRICS:
            assert np.isclose(g[metric].mean(), row[metric + '_mean'], rtol=0, atol=1e-12)
            assert np.isclose(g[metric].std(ddof=1), row[metric + '_sd'], rtol=0, atol=1e-12)
    for _, row in paired.iterrows():
        condition = {'K0': 'independent_k0', 'K1': 'independent_k1', 'K2': 'local_r2'}[row.horizon]
        independent = by_seed[(by_seed.dataset == row.dataset) & (by_seed.seed == row.seed) & (by_seed.condition == condition) & (by_seed.population == 'common_targets')].iloc[0]
        reference = rolling[(rolling.dataset == row.dataset) & (rolling.seed == row.seed) & (rolling.horizon == int(row.horizon[1:])) & (rolling.population == 'common_targets')].iloc[0]
        assert row.query_count == independent.query_count == reference.query_count
        for metric in METRICS:
            assert np.isclose(row['independent_' + metric], independent[metric], atol=1e-12, rtol=0)
            assert np.isclose(row['rolling_' + metric], reference[metric], atol=1e-12, rtol=0)
            assert np.isclose(row['rolling_minus_independent_' + metric], reference[metric] - independent[metric], atol=1e-12, rtol=0)
    sentence = load(EXPORT / 'unified_heldout_test_v2_sentence_summary.json')
    assert sentence['split'] == 'test' and sentence['status'] == 'completed_heldout_test'
    assert report['test_material_manifest_sha256'] == sentence['manifest_sha256']
    return context, paired, report


def axes_pair():
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.7), sharey=True, layout='constrained')
    for ax, (_, label) in zip(axes, DATASETS):
        ax.set_title(label, fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.set_ylim(15, 85)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Macro Top-10 (%)')
    return fig, axes


def savefig(fig, name):
    paths = []
    for extension in ('pdf', 'svg', 'png'):
        path = FIG / f'{name}.{extension}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def build():
    context, paired, report = verify()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    assets = []
    facts = {}
    fig, axes = axes_pair()
    for ax, (dataset, _) in zip(axes, DATASETS):
        g = context[(context.dataset == dataset) & (context.population == 'all_targets')].set_index('condition')
        conditions = [f'local_r{r}' for r in (0, 1, 2, 4, 8)]
        means = g.loc[conditions, 'macro_recall_at_10_mean'].to_numpy() * 100
        ax.errorbar(range(5), means, yerr=g.loc[conditions, 'macro_recall_at_10_sd'] * 100,
                    color='#24638A', marker='o', capsize=3, label='Local window')
        full = g.loc['full_context']
        ax.axhline(full.macro_recall_at_10_mean * 100, color='#AD6428', ls='--', label='Full segment')
        ax.axhspan((full.macro_recall_at_10_mean - full.macro_recall_at_10_sd) * 100,
                   (full.macro_recall_at_10_mean + full.macro_recall_at_10_sd) * 100, color='#AD6428', alpha=.08)
        ax.set_xticks(range(5), ['0', '1', '2', '4', '8'])
        ax.set_xlabel('Context radius')
        facts[dataset] = {'context_all_targets': {c: float(g.loc[c, 'macro_recall_at_10_mean'] * 100) for c in [*conditions, 'full_context']}}
    axes[1].legend(loc='lower right', fontsize=8, frameon=False)
    assets += savefig(fig, 'context_window_two_datasets_test')
    for kind, name in [('independent', 'future_increment_two_datasets_test'), ('rolling', 'rolling_revision_two_datasets_test')]:
        fig, axes = axes_pair()
        for ax, (dataset, _) in zip(axes, DATASETS):
            g = paired[paired.dataset == dataset]
            values = [g[g.horizon == f'K{k}'][kind + '_macro_recall_at_10'].to_numpy() * 100 for k in range(3)]
            assert all(len(v) == 3 for v in values)
            ax.errorbar(range(3), np.mean(values, axis=1), yerr=np.std(values, axis=1, ddof=1), marker='o',
                        color='#24638A', capsize=3, label=kind.title())
            if kind == 'rolling':
                reference = context[(context.dataset == dataset) & (context.population == 'common_targets') & (context.condition == 'local_r2')].iloc[0]
                ax.axhline(reference.macro_recall_at_10_mean * 100, ls='--', color='#AD6428', label='Independent K2')
            ax.set_xticks(range(3), ['K0', 'K1', 'K2'])
            ax.set_xlabel('Available future word events')
            facts[dataset][kind + '_common_targets'] = {f'K{k}': float(np.mean(values[k])) for k in range(3)}
        axes[1].legend(frameon=False, fontsize=8, loc='lower right')
        assets += savefig(fig, name)
    rows = []
    for dataset, label in DATASETS:
        facts[dataset]['rolling_minus_independent'] = {}
        for k in range(3):
            g = paired[(paired.dataset == dataset) & (paired.horizon == f'K{k}')]
            values = [g[kind + '_macro_recall_at_10'].to_numpy() * 100 for kind in ('independent', 'rolling')]
            difference = float((values[1] - values[0]).mean())
            facts[dataset]['rolling_minus_independent'][f'K{k}'] = difference
            rows.append([label if k == 0 else '', f'K{k}',
                         *[f'${v.mean():.2f} \\pm {v.std(ddof=1):.2f}$' for v in values], f'${difference:+.2f}$'])
    table = ['% Generated from reports/exports/unified_context_heldout_test_v2_*; test split only.',
             r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
             r'数据集 & 状态 & 独立模型 & 滚动模型 & $\Delta$（百分点） \\', r'\midrule',
             *[' & '.join(row) + r' \\' for row in rows], r'\bottomrule', r'\end{tabular}']
    path = TABLE / 'rolling_comparison_v2_test.tex'
    path.write_text('\n'.join(table) + '\n', encoding='utf-8')
    assets.append(path)
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / 'word_assets.json').write_text(json.dumps({'status': 'verified', 'split': 'test',
        'manifest_sha256': report['manifest_sha256'], 'source_exports': report['exports'],
        'assets': {str(p.relative_to(ROOT)): sha(p) for p in assets}, 'facts_percent': facts}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build()
