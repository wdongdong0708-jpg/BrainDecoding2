"""Present verified A/C1/C/F waiting budgets and B/C1 supplementary analyses.

Run with: python scripts/paper/wait_budget_figures_test_v2.py
Uses completed formal exports only; never runs decoding, training, model
selection, resampling, or edits author-owned manuscript prose.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from scripts.paper.publication_suite_style import DATA_COLORS, GRAY, INK, MUTED, clean, figure, style

EXPORT = ROOT / 'data/results/exports'
OUT = ROOT / 'paper/figures/publication_suite_test_v2_wait_budget'
TABLES = ROOT / 'paper/tables'
AUDIT = ROOT / 'data/results/audits/c1_manuscript_20261010'
DATASETS = [('pallier2025', 'Pallier2025'), ('chineseeeg2_littleprince', 'ChineseEEG-2')]
CONDITIONS = ['A', 'C1', 'C', 'F']
METRICS = ['sentence_bleu1_mean', 'wer', 'bleu1', 'aligned_word_accuracy']
PREFIX = 'sentence_wait_budget_test_v2'
FIGURE_NAME = 'quality_wait_ac1cf_v2_test'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-11)


def verified_study(stem):
    path = EXPORT / f'{stem}_summary.json'
    summary = read(path)
    audit = read(ROOT / 'data/results/studies' / stem / 'audit.json')
    assert summary['status'] == 'completed' and summary['split'] == 'test'
    assert audit['status'] == 'verified_complete'
    assert summary['original_AC_unchanged'] is True
    assert summary['manifest_sha256'] == audit['manifest_sha256']
    manifest = read(ROOT / 'data/manifests/rolling_sentence' / f'{stem}.json')
    identity = hashlib.sha256(json.dumps(
        {k: v for k, v in manifest.items() if k != 'manifest_sha256'},
        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert identity == summary['manifest_sha256']
    expected = audit.get('results_sha256', audit.get('result_sha256'))
    assert sha(path) == expected
    hashes = {path.relative_to(ROOT).as_posix(): sha(path)}
    for rel, expected in summary['exports'].items():
        assert sha(ROOT / rel) == expected == audit['exports'][rel], rel
        hashes[rel] = expected
    hashes[(ROOT / 'data/manifests/rolling_sentence' / f'{stem}.json').relative_to(ROOT).as_posix()] = sha(
        ROOT / 'data/manifests/rolling_sentence' / f'{stem}.json')
    return summary, hashes


def export_verified_data():
    full, full_hashes = verified_study('full_context_sentence_test_v2')
    one, one_hashes = verified_study('single_revision_c1_test_v2_stable')
    fixed, fixed_hashes = verified_study('fixed_k0_rolling_search_test_v2_stable')
    assert one['original_AC_reproduced'] and fixed['original_AC_reproduced']
    timing_dir = ROOT / 'data/results/studies/single_revision_c1_input_wait_test_v2'
    timing = read(timing_dir / 'results.json')
    timing_audit = read(timing_dir / 'audit.json')
    assert timing['status'] == 'completed' and timing_audit['status'] == 'verified_complete'
    assert timing['C1_manifest_sha256'] == one['manifest_sha256'] == timing_audit['C1_manifest_sha256']
    assert timing['checks']['duplicate_seed_timing_rows'] is False
    assert timing['checks']['all_test_events_retained'] is True
    assert timing['checks']['schedule_identical_across_seeds'] is True
    timing_hashes = {}
    for rel, expected in timing_audit['output_sha256'].items():
        path = timing_dir / rel
        assert sha(path) == expected, path
        timing_hashes[path.relative_to(ROOT).as_posix()] = expected
    words = pd.read_csv(timing_dir / 'by_word.csv', low_memory=False)
    assert not words.duplicated(['dataset', 'event_id', 'condition']).any()
    waits = pd.read_csv(EXPORT / 'full_context_sentence_test_v2_input_wait_summary.csv')
    old_events = pd.read_csv(EXPORT / 'full_context_sentence_test_v2_input_wait_by_event.csv')
    one_seeds = pd.read_csv(EXPORT / 'single_revision_c1_test_v2_stable_by_seed.csv')
    fixed_seeds = pd.read_csv(EXPORT / 'fixed_k0_rolling_search_test_v2_stable_by_seed.csv')
    seed_rows, mean_rows, comparisons = [], [], []
    for dataset, _ in DATASETS:
        original = full['datasets'][dataset]
        current = one['datasets'][dataset]
        ablation = fixed['datasets'][dataset]
        assert original['counts'] == current['counts'] == ablation['counts']
        time = words[words.dataset == dataset]
        old_time = old_events[old_events.dataset == dataset].set_index('event_id')
        for condition, column in [('A', 'k0_input_wait_s'), ('C1', 'k1_input_wait_s'), ('C', 'c_input_wait_s')]:
            part = time[time.condition == condition].set_index('event_id')
            assert set(part.index) == set(old_time.index)
            assert len(part) == current['counts']['events']
            close(part.nominal_final_input_wait_seconds.loc[old_time.index], old_time[column])
            close(part.nominal_final_input_wait_seconds - part.nominal_extra_input_wait_seconds, 1)
        time_means = time.groupby('condition')[['nominal_extra_input_wait_seconds', 'nominal_final_input_wait_seconds']].mean()
        total_waits = {}
        for condition, strategy in [('A', 'K0'), ('C', 'C_K2'), ('F', 'F')]:
            row = waits[(waits.dataset == dataset) & (waits.population == 'all_events') & (waits.strategy == strategy)].iloc[0]
            total_waits[condition] = float(row.mean_input_wait_s)
            if condition != 'F':
                close(total_waits[condition], time_means.loc[condition, 'nominal_final_input_wait_seconds'])
        total_waits['C1'] = float(time_means.loc['C1', 'nominal_final_input_wait_seconds'])
        for condition in ['A', 'C']:
            old = next(r for r in original['summary'] if r['condition'] == condition)
            for source, seeds in [(current, one_seeds), (ablation, fixed_seeds)]:
                row = next(r for r in source['summary'] if r['condition'] == condition)
                for metric in METRICS:
                    close(old[metric], row[metric])
                for seed in [0, 1, 2]:
                    old_row = next(r for r in original['by_seed'] if r['condition'] == condition and r['seed'] == seed)
                    new_row = seeds[(seeds.dataset == dataset) & (seeds.condition == condition) & (seeds.seed == seed)].iloc[0]
                    for metric in METRICS:
                        close(old_row[metric], new_row[metric])
        for condition in CONDITIONS:
            source = current if condition == 'C1' else original
            mean = next(r for r in source['summary'] if r['condition'] == condition)
            per_seed = [r for r in source['by_seed'] if r['condition'] == condition]
            assert {r['seed'] for r in per_seed} == {0, 1, 2}
            for metric in METRICS:
                close(np.mean([r[metric] for r in per_seed]), mean[metric])
            row = {'dataset': dataset, 'condition': condition,
                   'mean_total_input_wait_s': total_waits[condition],
                   'mean_extra_input_wait_s': total_waits[condition] - 1,
                   'waiting_population': 'all_events',
                   'timing_contract': 'nominal_onset_plus_one_second_consistent_with_method_section',
                   **{metric: mean[metric] for metric in METRICS}}
            gain = current['gains']['sentence_bleu1_mean']
            row['one_window_retained_share'] = gain['one_window_retained_share'] if condition == 'C1' else np.nan
            mean_rows.append(row)
            for item in sorted(per_seed, key=lambda r: r['seed']):
                seed_rows.append({**{k: v for k, v in row.items() if k != 'one_window_retained_share'},
                                  'seed': item['seed'], **{metric: item[metric] for metric in METRICS}})
        one_gain = current['gains']['sentence_bleu1_mean']
        one_mean = {r['condition']: r['sentence_bleu1_mean'] for r in current['summary']}
        close(one_gain['one_window_gain_C1_minus_A'], one_mean['C1'] - one_mean['A'])
        close(one_gain['second_window_gain_C_minus_C1'], one_mean['C'] - one_mean['C1'])
        close(one_gain['one_window_retained_share'], (one_mean['C1'] - one_mean['A']) / (one_mean['C'] - one_mean['A']))
        for analysis, source, names in [
            ('fixed_K0_ablation', ablation, ['B_minus_A', 'C_minus_B']),
            ('one_window_budget', current, ['C1_minus_A', 'C_minus_C1'])]:
            for comparison in names:
                stats = source['supplementary_statistics'][comparison]
                assert stats['bootstrap_samples'] == 10000
                comparisons.append({'dataset': dataset, 'analysis': analysis, 'comparison': comparison,
                                    **{k: stats[k] for k in ['effect', 'ci_lower', 'ci_upper', 'p_holm', 'blocks', 'family']}})
        stats = original['additional_C_minus_F']
        comparisons.append({'dataset': dataset, 'analysis': 'full_segment_comparison', 'comparison': 'C_minus_F',
                            **{k: stats[k] for k in ['effect', 'ci_lower', 'ci_upper']},
                            'p_holm': np.nan, 'blocks': original['counts']['blocks'], 'family': 'interval_only_no_new_p_test'})
    sources = {**full_hashes, **one_hashes, **fixed_hashes, **timing_hashes}
    outputs = {}
    for name, frame in [('by_seed', pd.DataFrame(seed_rows)), ('summary', pd.DataFrame(mean_rows)),
                        ('supplementary_comparisons', pd.DataFrame(comparisons))]:
        path = EXPORT / f'{PREFIX}_{name}.csv'
        frame.to_csv(path, index=False, lineterminator='\n')
        outputs[path.relative_to(ROOT).as_posix()] = sha(path)
    provenance = {'status': 'verified_complete', 'presentation_only': True,
                  'conditions': CONDITIONS, 'sources': sources, 'exports': outputs,
                  'waiting_population': 'all_events_without_seed_duplication',
                  'timing_contract': 'nominal_onset_plus_one_second',
                  'timing_note': 'All four strategies use the same onset+1s definition. The audited 50Hz endpoint sensitivity is retained in its original report.',
                  'new_training_decoding_or_statistical_tests': False,
                  'C1_statistics': 'existing_post_results_supplementary_family',
                  'B_statistics': 'existing_post_results_supplementary_family',
                  'original_AC_statistics_unchanged': True}
    (EXPORT / f'{PREFIX}_sources.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return provenance


def save_figure(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    outside = []
    for artist in fig.findobj(mpl.text.Text):
        if artist.get_visible() and artist.get_text():
            bounds = artist.get_window_extent(renderer)
            if bounds.x0 < box.x0 - 1 or bounds.y0 < box.y0 - 1 or bounds.x1 > box.x1 + 1 or bounds.y1 > box.y1 + 1:
                outside.append(artist.get_text())
    assert not outside, outside
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{FIGURE_NAME}.{ext}', dpi=360, facecolor='white', bbox_inches=None)
    plt.close(fig)


def plot_tradeoff(means, seeds, comparisons, focus_comparison='C_minus_F', auto_detail=False):
    fig = figure(100)
    for col, (dataset, label) in enumerate(DATASETS):
        color = DATA_COLORS[dataset]
        left = .095 + col * .500
        fig.text(.040 + col * .500, .972, chr(97 + col), fontsize=9, weight='bold', color=INK, va='top')
        fig.text(.069 + col * .500, .972, label, fontsize=8, weight='bold', color=color, va='top')
        ax = fig.add_axes([left, .235, .375, .610])
        clean(ax)
        ax.grid(axis='y', color='#E5EAED', lw=.45)
        ax.set_axisbelow(True)
        data = means[means.dataset == dataset].set_index('condition').loc[CONDITIONS]
        x = data.mean_total_input_wait_s.to_numpy()
        y = seeds[seeds.dataset == dataset].pivot(index='seed', columns='condition', values='sentence_bleu1_mean').loc[[0, 1, 2], CONDITIONS].to_numpy() * 100
        for seed, marker in enumerate(['o', 's', '^']):
            ax.plot(x, y[seed], color=color, lw=.8, alpha=.28, marker=marker, ms=3.1, mew=.45, mfc=color, zorder=2)
        average = data.sentence_bleu1_mean.to_numpy() * 100
        close(y.mean(axis=0), average)
        ax.plot(x, average, color=color, lw=1.65, zorder=4)
        for index, condition in enumerate(CONDITIONS):
            ax.plot(x[index], average[index], marker='D' if condition == 'C1' else 'o',
                    color=color, mfc='white', ms=5.2 if condition == 'C1' else 5.6, mew=1.25, zorder=5)
        offsets = [(7, -12), (-6, 7), (7, 7), (8, -3)]
        for index, (condition, offset) in enumerate(zip(CONDITIONS, offsets)):
            alignment = 'right' if offset[0] < 0 else 'left' if offset[0] > 0 else 'center'
            ax.annotate(condition, xy=(x[index], average[index]), xytext=offset, textcoords='offset points',
                        color=color, fontsize=7.1, ha=alignment, va='bottom', zorder=6,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .5, 'alpha': .9})
        saved = x[3] - x[2]
        arrow_y = average.max() + 1.3
        for index in [2, 3]:
            ax.plot([x[index], x[index]], [average[index] + .2, arrow_y],
                    color=GRAY, lw=.65, ls=(0, (2, 2)), zorder=1)
        ax.annotate('', xy=(x[3], arrow_y), xytext=(x[2], arrow_y),
                    arrowprops={'arrowstyle': '<->', 'color': MUTED, 'lw': .85})
        ax.text((x[2] + x[3]) / 2, arrow_y, f'{saved:.2f} s less waiting',
                ha='center', va='center', color=MUTED, fontsize=7,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.8}, zorder=7)
        ax.set(xlim=(.80, 3.85), ylim=(0, 19), xticks=[1, 1.5, 2, 3], yticks=[0, 5, 10, 15],
               xlabel='Mean total input waiting (s)',
               ylabel='Mean segment BLEU-1 (%)' if col == 0 else '')
        # Magnify the actual seconds, never replace waiting with an ordinal axis.
        detail = ax.inset_axes([.48, .065, .49, .32])
        clean(detail)
        detail.set_facecolor('white')
        for spine in detail.spines.values():
            spine.set_visible(True)
            spine.set_color(GRAY)
            spine.set_linewidth(.45)
        for seed, marker in enumerate(['o', 's', '^']):
            detail.plot(x[:3], y[seed, :3], color=color, lw=.65, alpha=.28,
                        marker=marker, ms=2.1, mew=.35, zorder=2)
        detail.plot(x[:3], average[:3], color=color, lw=1.1, zorder=4)
        for index, condition in enumerate(CONDITIONS[:3]):
            detail.plot(x[index], average[index],
                        marker='D' if condition == 'C1' else 'o',
                        mfc='white', mec=color, ms=3.9, mew=.9, zorder=5)
            detail.annotate(condition, xy=(x[index], average[index]), xytext=(0, 4),
                            textcoords='offset points', fontsize=5.9, color=color,
                            ha='center', va='bottom', zorder=6)
        detail.set(xlim=(.96, 1.64), ylim=(1.8, 16.8),
                   xticks=x[:3], xticklabels=[f'{value:.3f}' for value in x[:3]],
                   yticks=[5, 10, 15])
        if auto_detail:
            detail.set_ylim(float(y[:, :3].min()) - .5, float(y[:, :3].max()) + 1)
            detail.set_yticks(np.linspace(np.floor(y[:, :3].min()), np.ceil(y[:, :3].max()), 3))
        detail.tick_params(labelsize=5.6, length=2, pad=1)
        detail.text(.025, .97, 'A–C detail', transform=detail.transAxes,
                    fontsize=6, color=MUTED, ha='left', va='top')
        stat = comparisons[(comparisons.dataset == dataset) & (comparisons.comparison == focus_comparison)].iloc[0]
        contrast_label = focus_comparison.replace('_minus_', ' − ')
        fig.text(left, .095, f'{contrast_label}: {stat.effect * 100:+.2f} pp  ·  95% CI [{stat.ci_lower * 100:+.2f}, {stat.ci_upper * 100:+.2f}]',
                 fontsize=6.6, color=MUTED)
    fig.legend(handles=[Line2D([], [], color=GRAY, lw=.8, marker='s', ms=3, alpha=.55, label='Three paired seed trajectories'),
                        Line2D([], [], color=INK, lw=1.65, marker='o', mfc='white', ms=5, mew=1.2, label='Mean across seeds')],
               loc='center', bbox_to_anchor=(.52, .913), ncol=2, frameon=False, columnspacing=2.8)
    fig.text(.095, .035, 'A  Fixed     C1  One revision     C  Two revisions     F  Full segment', color=INK, fontsize=6.7)
    save_figure(fig)


def table_start(columns, spacing=4):
    return ['% Presentation only: verified sentence_wait_budget_test_v2 formal exports.',
            r'\begingroup', r'\small', rf'\setlength{{\tabcolsep}}{{{spacing}pt}}',
            r'\renewcommand{\arraystretch}{1.12}', rf'\begin{{tabular}}{{@{{}}{columns}@{{}}}}', r'\toprule']


def write_tables(means, comparisons):
    lines = table_start('lrrrrrrrr', 3)
    lines += [r'策略 & \multicolumn{4}{c}{Pallier2025} & \multicolumn{4}{c}{ChineseEEG-2} \\',
              r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',
              r' & BLEU-1 $\uparrow$ & WER $\downarrow$ & Acc. $\uparrow$ & Wait (s) & BLEU-1 $\uparrow$ & WER $\downarrow$ & Acc. $\uparrow$ & Wait (s) \\',
              r'\midrule']
    for condition, label in zip(CONDITIONS, ['A 固定输出', 'C1 一次回修', 'C 滚动修正', 'F 完整片段']):
        fields = [label]
        for dataset, _ in DATASETS:
            row = means[(means.dataset == dataset) & (means.condition == condition)].iloc[0]
            fields.extend(f'{row[metric] * 100:.2f}' for metric in ['sentence_bleu1_mean', 'wer', 'aligned_word_accuracy'])
            fields.append(f'{row.mean_total_input_wait_s:.3f}')
        lines.append(' & '.join(fields) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
    (TABLES / 'sentence_wait_budget_metrics_v2_test.tex').write_text('\n'.join(lines), encoding='utf-8')
    for name, analysis, labels in [
        ('neural_update_ablation_v2_test', 'fixed_K0_ablation', [('B_minus_A', r'$B-A$'), ('C_minus_B', r'$C-B$')]),
        ('revision_budget_effects_v2_test', 'one_window_budget', [('C1_minus_A', r'$C1-A$'), ('C_minus_C1', r'$C-C1$')])]:
        lines = table_start('llrrr')
        lines += [r'数据集 & 比较 & BLEU-1 差值 (pp) & $95\%$ 区间 (pp) & Holm $p$ \\', r'\midrule']
        for index, (dataset, label) in enumerate(DATASETS):
            if index:
                lines.append(r'\midrule')
            for j, (comparison, display) in enumerate(labels):
                row = comparisons[(comparisons.dataset == dataset) & (comparisons.analysis == analysis) & (comparisons.comparison == comparison)].iloc[0]
                fields = [label if j == 0 else '', display, f'${row.effect * 100:+.2f}$',
                          f'$[{row.ci_lower * 100:+.2f},{row.ci_upper * 100:+.2f}]$', f'{row.p_holm:.6f}']
                lines.append(' & '.join(fields) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
        (TABLES / (name + '.tex')).write_text('\n'.join(lines), encoding='utf-8')


def main():
    provenance = export_verified_data()
    means = pd.read_csv(EXPORT / f'{PREFIX}_summary.csv')
    seeds = pd.read_csv(EXPORT / f'{PREFIX}_by_seed.csv')
    comparisons = pd.read_csv(EXPORT / f'{PREFIX}_supplementary_comparisons.csv')
    with style():
        plot_tradeoff(means, seeds, comparisons)
    write_tables(means, comparisons)
    AUDIT.mkdir(parents=True, exist_ok=True)
    artifacts = list(OUT.glob(FIGURE_NAME + '.*')) + [TABLES / n for n in [
        'sentence_wait_budget_metrics_v2_test.tex', 'neural_update_ablation_v2_test.tex', 'revision_budget_effects_v2_test.tex']]
    (AUDIT / 'presentation.json').write_text(json.dumps(
        {**provenance, 'artifacts': {p.relative_to(ROOT).as_posix(): sha(p) for p in artifacts},
         'text_outside_figure_canvas': 0}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'verified_complete', 'conditions': CONDITIONS,
                      'figure': str(OUT / (FIGURE_NAME + '.png')), 'mean_rows': len(means),
                      'seed_rows': len(seeds), 'timing_contract': provenance['timing_contract'],
                      'new_decoding_or_statistical_tests': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
