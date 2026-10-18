"""Apply the accepted publication-suite visual grammar to frozen v2 test assets.

Scientific inputs are the verified reports/exports and frozen settings. This
script changes figure/table presentation only; manuscript prose and captions
are checked byte-for-byte. Use --deploy after reviewing the generated PNGs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import yaml

from scripts.paper.publication_suite_style import (
    BLUE, DATA_COLORS, GRAY, INK, MUTED, ORANGE, PURPLE, RULE, TEAL,
    cjk, clean, figure, style,
)
from scripts.paper import apply_publication_style as previous
from scripts.paper import revise_annotated_manuscript_assets as reviewed
from scripts.paper.update_test_only_word_results import verify as verify_word
from scripts.paper.update_heldout_sentence_results import verify as verify_sentence

OUT = ROOT / 'paper/figures/publication_suite_test_v2_refined'
QA = ROOT / '.cache/publication_suite_test_v2_refined_20261005'
AUDIT = ROOT / 'data/results/audits/publication_suite_test_v2_refined_20261005'
DATASETS = previous.DATASETS
EXPORT = ROOT / 'data/results/exports'
FIGURES = {
    '01_method_overview': 'rolling_revision_method',
    '02_context_window': 'context_window_two_datasets_test',
    '03_future_increment': 'future_increment_two_datasets_test',
    '04_rolling_retrieval': 'rolling_revision_two_datasets_test',
    '05_sentence_revision': 'sentence_revision_v2_test',
    '10_sentence_revision_examples': 'sentence_revision_examples_v2_test',
}
TABLES = {
    '11_context_summary': 'context_summary_v2_test',
    '06_rolling_comparison': 'rolling_comparison_v2_test',
    '07_sentence_metrics': 'sentence_v2_test_metrics',
    '12_sentence_final_changes': 'sentence_v2_test_final_changes',
    '08_sentence_repairs': 'sentence_v2_test_repairs',
    'model_settings': 'model_settings_v2',
    'test_population': 'test_population_v2',
    'sentence_decoder_settings': 'sentence_decoder_settings_v2',
}
TITLES = {
    '01_method_overview': '方法总览与滚动输出',
    '02_context_window': '局部上下文范围',
    '03_future_increment': '未来上下文增量',
    '04_rolling_retrieval': '滚动词检索',
    '05_sentence_revision': '句子性能与增益区间',
    '10_sentence_revision_examples': '真实词位回修过程',
    '11_context_summary': '局部与完整片段对照',
    '06_rolling_comparison': '独立与共享模型对照',
    '07_sentence_metrics': '句子解码结果',
    '12_sentence_final_changes': 'A / C 最终输出变化',
    '08_sentence_repairs': 'C 自身回修统计',
    'model_settings': '模型设置',
    'test_population': '词库与测试范围',
    'sentence_decoder_settings': '句子解码设置',
}
FONT = {
    'font.size': 7.5, 'axes.labelsize': 7.2, 'axes.titlesize': 8,
    'xtick.labelsize': 6.8, 'ytick.labelsize': 6.8, 'legend.fontsize': 6.7,
}
SIZE = 183
BOUNDS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def snapshot():
    QA.mkdir(parents=True, exist_ok=True)
    record = QA / 'before_sha256.json'
    if record.exists():
        before = read(record)
    else:
        manuscript = sorted((ROOT / 'paper/manuscript').glob('*.tex'))
        assets = [ROOT / 'paper/figures' / f'{name}.{ext}'
                  for name in FIGURES.values() for ext in ('pdf', 'svg', 'png')]
        assets += [ROOT / 'paper/tables' / f'{name}.tex' for name in TABLES.values()]
        assets += [ROOT / 'output/pdf/manuscript_unified_protocol_draft.pdf']
        before = {'manuscript': {relative(p): sha(p) for p in manuscript},
                  'assets': {relative(p): sha(p) for p in assets if p.exists()}}
        for source in assets:
            if source.exists():
                destination = QA / 'backup' / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        record.write_text(json.dumps(before, indent=2) + '\n', encoding='utf-8')
    assert all(sha(ROOT / path) == digest for path, digest in before['manuscript'].items())
    return before


def typography(text):
    return {'fontproperties': cjk()} if any('\u3400' <= ch <= '\u9fff' for ch in str(text)) else {}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    box = fig.bbox
    overflow = []
    for artist in fig.findobj(mpl.text.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        b = artist.get_window_extent(renderer)
        if b.x0 < box.x0 - 1 or b.x1 > box.x1 + 1 or b.y0 < box.y0 - 1 or b.y1 > box.y1 + 1:
            overflow.append(artist.get_text())
    assert not overflow, (name, 'text outside canvas', overflow)
    paths = []
    for ext in ('pdf', 'svg', 'png'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, dpi=360, facecolor='white', bbox_inches=None)
        paths.append(path)
    BOUNDS.append({'asset': name, 'text_outside_canvas': 0})
    plt.close(fig)
    return paths


def header(fig, x, y, letter, title, dataset=None):
    fig.text(x, y, letter, fontsize=9, weight='bold', va='top', color=INK)
    fig.text(x + .029, y, title, fontsize=8, weight='bold', va='top',
             color=DATA_COLORS[dataset] if dataset else INK)


def axes_pair(height=86, bottom=.215, top=.770):
    fig = figure(height)
    axes = []
    for col, (dataset, label) in enumerate(DATASETS):
        header(fig, .040 + col * .491, .955, chr(97 + col), label, dataset)
        ax = fig.add_axes([.110 + col * .491, bottom, .348, top - bottom])
        clean(ax)
        ax.grid(axis='y', color='#E5EAED', lw=.45, zorder=0)
        ax.set_axisbelow(True)
        axes.append(ax)
    return fig, axes


def legend(fig, handles, y=.856):
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.516, y),
               ncol=len(handles), frameon=False, handlelength=2.0,
               columnspacing=2.2, handletextpad=.6)


def footer(fig, value, y=.055):
    fig.text(.110, y, value, fontsize=6.4, color=MUTED, va='bottom')


def trajectories(ax, x, values, color, mean=True):
    values = np.asarray(values)
    assert values.shape[0] == 3
    for seed, marker in enumerate(('o', 's', '^')):
        ax.plot(x, values[seed], color=color, alpha=.30, lw=.80,
                marker=marker, ms=3, mew=.45, mfc='white', zorder=2)
    center, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    if mean:
        ax.errorbar(x, center, yerr=sd, color=color, lw=1.75,
                    capsize=2.5, capthick=.8, marker='o', ms=4.5,
                    mfc='white', mew=1.1, zorder=4)
    return center, sd


def seed_key():
    return Line2D([], [], color=GRAY, alpha=.6, marker='s', mfc='white',
                  ms=3, lw=.8, label='Three seed trajectories')


def mean_key(label='Mean ± SD'):
    return Line2D([], [], color=INK, marker='o', mfc='white', ms=4,
                  lw=1.75, label=label)


def context_figure(context, by_seed):
    fig, axes = axes_pair()
    facts = {}
    radii = [0, 1, 2, 4, 8]
    conditions = [f'local_r{k}' for k in radii]
    for ax, (dataset, _) in zip(axes, DATASETS):
        raw = by_seed[(by_seed.dataset == dataset) & (by_seed.population == 'all_targets')]
        values = raw.pivot(index='seed', columns='condition', values='macro_recall_at_10').loc[[0, 1, 2], conditions].to_numpy() * 100
        group = context[(context.dataset == dataset) & (context.population == 'all_targets')].set_index('condition')
        full = group.loc['full_context']
        ref, sd = full.macro_recall_at_10_mean * 100, full.macro_recall_at_10_sd * 100
        ax.axhspan(ref - sd, ref + sd, color=GRAY, alpha=.14, lw=0, zorder=0)
        ax.axhline(ref, color=GRAY, lw=.9, ls=(0, (4, 2)), zorder=1)
        mean, spread = trajectories(ax, radii, values, DATA_COLORS[dataset])
        np.testing.assert_allclose(mean, group.loc[conditions, 'macro_recall_at_10_mean'] * 100, atol=1e-10)
        np.testing.assert_allclose(spread, group.loc[conditions, 'macro_recall_at_10_sd'] * 100, atol=1e-10)
        ax.set(xlim=(-.35, 8.4), ylim=(15, 82), xticks=radii, yticks=[20, 40, 60, 80],
               xlabel='Symmetric context radius (words)', ylabel='Macro Top-10 (%)')
        facts[dataset] = {'radii': radii, 'means_percent': mean.tolist(),
                          'sample_sd_percent': spread.tolist(), 'full_mean_percent': ref,
                          'full_sample_sd_percent': sd, 'population': 'all_targets'}
    legend(fig, [seed_key(), mean_key(), Line2D([], [], color=GRAY, lw=.9,
                 ls=(0, (4, 2)), label='Full segment ± SD')])
    footer(fig, 'Held-out test · all frozen 50-word queries · radius 0 retains the contextual Transformer')
    return save(fig, '02_context_window'), facts


def delayed_values(paired, dataset, model):
    group = paired[paired.dataset == dataset]
    return group.pivot(index='seed', columns='horizon', values=model + '_macro_recall_at_10').loc[[0, 1, 2], ['K0', 'K1', 'K2']].to_numpy() * 100


def future_figure(paired):
    fig, axes = axes_pair(90, bottom=.22)
    facts = {}
    for ax, (dataset, _) in zip(axes, DATASETS):
        values = delayed_values(paired, dataset, 'independent')
        color = DATA_COLORS[dataset]
        mean, sd = trajectories(ax, [0, 1, 2], values, color, mean=False)
        for k, value in enumerate(mean):
            start = 0 if k == 0 else mean[k - 1]
            ax.bar(k, abs(value - start), bottom=min(start, value), width=.48,
                   color=GRAY if k == 0 else color, alpha=.28 if k == 0 else .58,
                   edgecolor='none', zorder=1)
            if k < 2:
                ax.plot([k + .24, k + 1 - .24], [value, value], color=GRAY,
                        ls=(0, (2, 2)), lw=.65, zorder=1)
            if k:
                ax.text(k + .29, (value + start) / 2, f'{value - start:+.2f}',
                        fontsize=7, ha='left', va='center', color=color, weight='bold')
            ax.annotate(f'{value:.2f}', (k, value + sd[k]), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=7, color=INK)
        ax.errorbar([0, 1, 2], mean, yerr=sd, fmt='o', color=INK, ms=4.5,
                    capsize=2.5, capthick=.8, lw=1, mfc='white', mew=.9, zorder=5)
        ax.set(xlim=(-.5, 2.88), ylim=(0, 86), xticks=[0, 1, 2],
               xticklabels=['K0', 'K1', 'K2'], yticks=[0, 20, 40, 60, 80],
               xlabel='Available future word events', ylabel='Macro Top-10 (%)')
        facts[dataset] = {'means_percent': mean.tolist(), 'sample_sd_percent': sd.tolist(),
                          'adjacent_differences_pp': np.diff(mean).tolist(), 'population': 'common_targets'}
    legend(fig, [seed_key(), Line2D([], [], color=INK, marker='o', mfc='white', ms=4,
                 lw=0, label='Mean ± SD'), Line2D([], [], color=GRAY, lw=5, alpha=.5,
                 label='Floating bars: score differences (pp)')])
    footer(fig, 'Held-out test · common queries · independently trained models with a fixed two-word history')
    return save(fig, '03_future_increment'), facts


def rolling_figure(paired):
    fig, axes = axes_pair()
    facts = {}
    for ax, (dataset, _) in zip(axes, DATASETS):
        values = delayed_values(paired, dataset, 'rolling')
        mean, sd = trajectories(ax, [0, 1, 2], values, DATA_COLORS[dataset])
        ref = delayed_values(paired, dataset, 'independent')[:, 2].mean()
        ax.axhline(ref, color=GRAY, lw=.9, ls=(0, (4, 2)), zorder=1)
        ax.set(xlim=(-.16, 2.16), ylim=(25, 82), xticks=[0, 1, 2],
               xticklabels=['Initial\nK0', '+1 word\nK1', '+2 words\nK2'],
               yticks=[30, 40, 50, 60, 70, 80], xlabel='Rolling update stage', ylabel='Macro Top-10 (%)')
        facts[dataset] = {'means_percent': mean.tolist(), 'sample_sd_percent': sd.tolist(),
                          'independent_k2_mean_percent': ref, 'population': 'common_targets'}
    legend(fig, [seed_key(), mean_key(), Line2D([], [], color=GRAY, lw=.9,
                 ls=(0, (4, 2)), label='Independent K2 mean')])
    footer(fig, 'Held-out test · common queries · the same model revisits each target as two later events arrive', y=.035)
    return save(fig, '04_rolling_retrieval'), facts


def sentence_figure(summary):
    fig, axes = axes_pair(118, bottom=.45, top=.81)
    facts = {}
    for ax, (dataset, _) in zip(axes, DATASETS):
        data = summary['datasets'][dataset]
        values = np.array([[next(r['sentence_bleu1_mean'] for r in data['by_seed']
                            if r['seed'] == seed and r['condition'] == condition)
                            for condition in ('A', 'C')] for seed in (0, 1, 2)]) * 100
        for seed, marker in enumerate(('o', 's', '^')):
            ax.plot([0, 1], values[seed], color=GRAY, alpha=.70, lw=.85,
                    marker=marker, ms=3.5, mfc='white', mew=.6, zorder=2)
        mean = values.mean(axis=0)
        color = DATA_COLORS[dataset]
        ax.plot([0, 1], mean, color=color, lw=1.9, marker='o', ms=4.8,
                mfc='white', mew=1.2, zorder=4)
        for x, v in enumerate(mean):
            ax.annotate(f'{v:.2f}', (x, v), xytext=(-7 if x == 0 else 7, 5),
                        textcoords='offset points', ha='right' if x == 0 else 'left',
                        fontsize=7.3, color=color, weight='bold')
        ax.set(xlim=(-.25, 1.30), ylim=(0, 19), xticks=[0, 1],
               xticklabels=['A\nFixed output', 'C\nRolling revision'],
               yticks=[0, 5, 10, 15], ylabel='Mean segment BLEU-1 (%)')
        stats = data['primary_statistics']
        facts[dataset] = {'means_percent': mean.tolist(), 'seed_values_percent': values.tolist(),
                          'gain_pp': stats['effect'] * 100,
                          'ci95_pp': [stats['ci_lower'] * 100, stats['ci_upper'] * 100]}
    legend(fig, [Line2D([], [], color=GRAY, lw=.85, marker='s', ms=3, mfc='white',
                       label='Paired model seeds'), mean_key('Three-seed mean')], y=.885)
    header(fig, .040, .324, 'c', 'Benefit of rolling revision')
    ax = fig.add_axes([.265, .140, .592, .132])
    clean(ax)
    ax.spines['left'].set_visible(False)
    ax.axvline(0, color=GRAY, ls=(0, (2, 2)), lw=.8)
    ax.set(xlim=(-.15, 6.5), ylim=(-.55, 1.55), yticks=[1, 0],
           yticklabels=[label for _, label in DATASETS], xticks=[0, 2, 4, 6],
           xlabel='C minus A · mean segment BLEU-1 (pp)')
    ax.tick_params(axis='y', length=0, pad=8)
    for y, (dataset, _) in zip([1, 0], DATASETS):
        f = facts[dataset]
        lo, hi = f['ci95_pp']
        ax.plot([lo, hi], [y, y], color=DATA_COLORS[dataset], lw=1.5)
        ax.plot([lo, hi], [y, y], '|', color=DATA_COLORS[dataset], ms=5, mew=.9)
        ax.plot(f['gain_pp'], y, 'D', mfc=DATA_COLORS[dataset], mec='white', mew=.5, ms=4.8)
        ax.text(hi + .15, y, f"{f['gain_pp']:+.2f}", va='center', fontsize=7,
                color=DATA_COLORS[dataset], weight='bold')
    footer(fig, 'Held-out test · 95% intervals: material-block bootstrap · equal subject and seed weights', y=.035)
    return save(fig, '05_sentence_revision'), facts


def case_excerpts(summary):
    cases = reviewed.extract_examples(summary)
    for case in cases:
        example = next(e for e in summary['datasets'][case['dataset']]['examples']
                       if e['selection'] == case['selection'])
        trace = {r['arrival']: r['C'] for r in example['trace']}
        target = case['position'] - 1
        indices = [target + offset for offset in (-2, -1, 0, 1, 2)]
        case['reference_excerpt'] = [example['reference'][i] if i >= 0 else None for i in indices]
        case['output_excerpts'] = [[trace[target + step][i] if 0 <= i <= target + step else None
                                    for i in indices] for step in range(3)]
        case['editable_columns'] = [[j for j, i in enumerate(indices)
                                     if max(0, target + step - 2) <= i <= target + step]
                                    for step in range(3)]
        assert [r[2] for r in case['output_excerpts']] == case['tokens']
    return cases


def example_figure(cases):
    height = 126
    fig = figure(height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, SIZE), ylim=(height, 0))
    ax.axis('off')

    def text(x, y, value, size=7, color=INK, bold=False, ha='left'):
        kw = typography(value)
        if kw and bold:
            kw = {'fontproperties': cjk(True)}
        return ax.text(x, y, value, fontsize=size, color=color, va='center', ha=ha,
                       weight='bold' if bold else 'normal', **kw)

    for n, case in enumerate(cases):
        x0, y0 = 6 + (n % 2) * 91, 6 + (n // 2) * 55
        text(x0, y0, chr(97 + n), 9, bold=True)
        text(x0 + 5, y0, case['label'], 8, DATA_COLORS[case['dataset']], bold=True)
        text(x0 + 83, y0, 'Correction' if case['change'] == '改对' else 'Regression',
             7.2, TEAL if case['change'] == '改对' else ORANGE, bold=True, ha='right')
        text(x0, y0 + 6, f"{case['subject']} · seed 0 · target word {case['position']}", 6.5, MUTED)
        base, cell, gap = x0 + 18, 11.8, 1.6
        centers = [base + j * (cell + gap) + cell / 2 for j in range(5)]
        text(x0, y0 + 14, 'Reference', 6.5, MUTED)
        for j, token in enumerate(case['reference_excerpt']):
            text(centers[j], y0 + 14, token if token is not None else '-', 8.2,
                 INK if j == 2 else MUTED, bold=j == 2, ha='center')
        ax.plot([x0, x0 + 83], [y0 + 18, y0 + 18], color=RULE, lw=.55)
        for step, row in enumerate(case['output_excerpts']):
            y = y0 + 24 + step * 9.2
            text(x0, y, ['Initial', '+1 word', '+2 words'][step], 6.7, MUTED)
            editable = case['editable_columns'][step]
            start = base + editable[0] * (cell + gap) - .8
            end = base + editable[-1] * (cell + gap) + cell + .8
            ax.add_patch(FancyBboxPatch((start, y - 3.6), end - start, 7.2,
                         boxstyle='round,pad=0,rounding_size=.9',
                         facecolor='#F1EDF6', edgecolor='none', zorder=0))
            for j, token in enumerate(row):
                if token is None:
                    text(centers[j], y, '-', 8.2, GRAY, ha='center')
                    continue
                is_target = j == 2
                correct = token == case['reference']
                face = ('#E6F1ED' if correct else '#F6EAE3') if is_target else (
                    'white' if j in editable else '#F0F3F4')
                ax.add_patch(FancyBboxPatch((base + j * (cell + gap), y - 3), cell, 6,
                             boxstyle='round,pad=0,rounding_size=.7', facecolor=face,
                             edgecolor=INK if is_target else 'none',
                             linewidth=1.2 if is_target else .4, zorder=2))
                text(centers[j], y, token, 8.2, (TEAL if correct else ORANGE) if is_target else MUTED,
                     bold=is_target, ha='center')
            if step < 2:
                ax.annotate('', xy=(centers[2], y + 6), xytext=(centers[2], y + 3.5),
                            arrowprops={'arrowstyle': '-|>', 'mutation_scale': 4,
                                        'color': GRAY, 'lw': .6})
    ax.plot([6, 177], [115, 115], color=RULE, lw=.55)
    for x, face, edge, label in [
        (6, '#E6F1ED', TEAL, 'Target matches reference'),
        (63, '#F6EAE3', ORANGE, 'Target mismatch'),
        (111, '#F1EDF6', PURPLE, 'Current editable suffix'),
    ]:
        ax.add_patch(FancyBboxPatch((x, 119), 3.5, 3,
                     boxstyle='round,pad=0,rounding_size=.4', facecolor=face,
                     edgecolor=edge, lw=.6))
        text(x + 5, 120.5, label, 6.5, MUTED)
    return save(fig, '10_sentence_revision_examples')


def collect_tables(context, paired, summary, manifest, config):
    """Reuse the current tested row builders while separating data from rendering."""
    records = {}

    def capture(name, headers, rows, widths, **kwargs):
        records[name] = {'headers': headers, 'rows': rows, 'widths': widths, **kwargs}
        return []

    original_previous, original_reviewed = previous.table_figure, reviewed.table_figure
    original_wrapper = reviewed.wrapper
    try:
        previous.table_figure = reviewed.table_figure = capture
        reviewed.wrapper = lambda *args, **kwargs: Path('not-written')
        previous.tables(paired, summary)
        reviewed.test_tables(context, summary)
        reviewed.protocol_tables(manifest, config, summary)
    finally:
        previous.table_figure, reviewed.table_figure = original_previous, original_reviewed
        reviewed.wrapper = original_wrapper
    assert set(TABLES) <= set(records)
    return {name: records[name] for name in TABLES}


def table_figure(name, headers, rows, widths, group_headers=(), bold_rows=(), group_starts=()):
    widths = np.array(widths) * 171
    edges = np.r_[0, np.cumsum(widths)] + 6
    assert np.isclose(edges[-1], 177)
    grouped = bool(group_headers)
    header_h = 16 if grouped else 10
    row_heights = [max(6.8, 5.3 * max(str(v).count('\n') + 1 for v in row)) for row in rows]
    height = 6 + header_h + sum(row_heights)
    fig = figure(height)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, SIZE), ylim=(height, 0))
    ax.axis('off')
    top, bottom = 3, height - 3
    ax.plot([6, 177], [top, top], color=INK, lw=.9)
    ax.plot([6, 177], [top + header_h, top + header_h], color=INK, lw=.5)
    ax.plot([6, 177], [bottom, bottom], color=INK, lw=.9)
    cell_texts = []

    def cell(col, y, value, bold=False, header=False, color=INK):
        kw = typography(value)
        if kw and bold:
            kw = {'fontproperties': cjk(True)}
        artist = ax.text(edges[col] + 1 if col == 0 else (edges[col] + edges[col + 1]) / 2,
                         y, str(value), fontsize=7.8 if header else 8.2,
                         color=color, weight='bold' if bold else 'normal',
                         ha='left' if col == 0 else 'center', va='center', linespacing=1.3, **kw)
        cell_texts.append((artist, edges[col], edges[col + 1]))

    if grouped:
        cell(0, top + header_h / 2, headers[0], True, True)
        for lo, hi, value, color in group_headers:
            ax.text((edges[lo] + edges[hi]) / 2, top + 4.5, value,
                    fontsize=8.2, color=color, weight='bold', va='center', ha='center')
            ax.plot([edges[lo] + 1.5, edges[hi] - 1.5], [top + 8.2, top + 8.2], color=RULE, lw=.5)
        for col in range(1, len(headers)):
            cell(col, top + 12, headers[col], True, True)
    else:
        for col, label in enumerate(headers):
            color = next((DATA_COLORS[d] for d, ds in DATASETS if ds == label), INK)
            cell(col, top + header_h / 2, label, True, True, color)
    y = top + header_h
    for i, (row, h) in enumerate(zip(rows, row_heights)):
        if i in group_starts and i:
            ax.plot([6, 177], [y, y], color=RULE, lw=.4)
        for col, value in enumerate(row):
            cell(col, y + h / 2, value, i in bold_rows)
        y += h
    fig.canvas.draw()
    transform = ax.transData
    renderer = fig.canvas.get_renderer()
    for artist, left, right in cell_texts:
        b = artist.get_window_extent(renderer)
        assert b.x0 >= transform.transform((left, 0))[0] - 1, (name, artist.get_text(), 'left overflow')
        assert b.x1 <= transform.transform((right, 0))[0] + 1, (name, artist.get_text(), 'right overflow')
    return save(fig, name)


def render_tables(records):
    paths = []
    for name, record in records.items():
        headers, rows, widths = record['headers'], record['rows'], record['widths']
        groups, bold, starts = (), (), record.get('group_starts', ())
        if name == '06_rolling_comparison':
            headers = ['可见状态', '独立模型', '共享滚动模型', 'Δ (pp)', '独立模型', '共享滚动模型', 'Δ (pp)']
            rows = [[record['rows'][k][1], *record['rows'][k][2:], *record['rows'][k + 3][2:]] for k in range(3)]
            widths = [.14, .16, .16, .11, .16, .16, .11]
            groups = [(1, 4, DATASETS[0][1], BLUE), (4, 7, DATASETS[1][1], PURPLE)]
            starts = ()
        elif name == '07_sentence_metrics':
            headers = ['解码条件', 'BLEU-1 ↑', 'WER ↓', '对齐词准确率 ↑', 'BLEU-1 ↑', 'WER ↓', '对齐词准确率 ↑']
            rows = [[record['rows'][k][1], *record['rows'][k][2:], *record['rows'][k + 2][2:]] for k in range(2)]
            widths = [.12, .12, .12, .20, .12, .12, .20]
            groups = [(1, 4, DATASETS[0][1], BLUE), (4, 7, DATASETS[1][1], PURPLE)]
            bold, starts = (1,), ()
        elif name == '11_context_summary':
            headers = ['输入范围', DATASETS[0][1], DATASETS[1][1]]
            rows = [[label, record['rows'][0][k + 1], record['rows'][1][k + 1]]
                    for k, label in enumerate(['局部半径0', '局部半径2', '完整片段'])]
            widths = [.34, .33, .33]
        elif name == 'model_settings':
            rows = [list(row) for row in rows]
            rows[-1][2] = '遍历到达时刻\n监督 K0 / K1 / K2'
            widths = [.16, .18, .37, .29]
        paths += table_figure(name, headers, rows, widths, groups, bold, starts)
    return paths


def retained_method():
    """The approved editorial illustration already matches the current protocol."""
    paths = []
    for ext in ('pdf', 'svg', 'png'):
        source = QA / 'backup/paper/figures' / f'rolling_revision_method.{ext}'
        destination = OUT / f'01_method_overview.{ext}'
        shutil.copy2(source, destination)
        assert sha(source) == sha(destination)
        paths.append(destination)
    return paths


def deploy(before):
    paths = []
    for source, target in FIGURES.items():
        for ext in ('pdf', 'svg', 'png'):
            destination = ROOT / 'paper/figures' / f'{target}.{ext}'
            shutil.copy2(OUT / f'{source}.{ext}', destination)
            paths.append(destination)
    for source, target in TABLES.items():
        path = ROOT / 'paper/tables' / f'{target}.tex'
        path.write_text('% Presentation only; formal frozen v2 test sources are verified by the plotter.\n'
                        + r'\includegraphics[width=\linewidth]{' + OUT.name + '/' + source + '.pdf}\n', encoding='utf-8')
        paths.append(path)
    assert all(sha(ROOT / p) == h for p, h in before['manuscript'].items())
    return paths


def overview():
    from PIL import Image, ImageDraw, ImageFont
    names = list(FIGURES) + list(TABLES)
    names = names[:6] + names[6:]
    w, columns, card_w, card_h, pad = 1800, 3, 576, 392, 18
    h = 86 + int(np.ceil(len(names) / columns)) * (card_h + pad) + pad
    canvas = Image.new('RGB', (w, h), '#F1F5F6')
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 27)
    label_font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    draw.text((pad, 17), '统一协议 v2 · 正文图表新样式', font=title_font, fill=INK)
    draw.text((pad, 52), '测试集结果 · 6 图 + 8 表 · 蓝 / 紫数据集配色与矢量三线表', font=label_font, fill=MUTED)
    for i, name in enumerate(names):
        x, y = pad + (i % columns) * (card_w + pad), 86 + (i // columns) * (card_h + pad)
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), 9, fill='white', outline='#DDE5E9')
        draw.text((x + 14, y + 10), TITLES[name], font=label_font, fill=INK)
        picture = Image.open(OUT / f'{name}.png').convert('RGB')
        picture.thumbnail((card_w - 28, card_h - 54), Image.Resampling.LANCZOS)
        canvas.paste(picture, (x + (card_w - picture.width) // 2,
                               y + 44 + (card_h - 54 - picture.height) // 2))
    destination = OUT / 'suite_overview.png'
    canvas.save(destination)
    return destination


def readme():
    (OUT / 'README.md').write_text('''# 统一协议 v2 正文图表：本窗口认可的出版风格

运行：`python scripts/paper/refine_publication_style_test_v2.py --deploy`

所有结果直接读取并核验 `reports/exports/unified_*heldout_test_v2*` 的正式测试导出。
设置表读取 frozen test manifest 与 `configs/unified_rolling/v2.yaml`。
原 publication_suite、原测试资产和正文逐字保留的图注用于对照；本目录不是新结果真源。

- 方法图保留已认可的 editorial 版本；蓝色新信号、黑框固定目标、等宽可回修尾部。
- 局部范围与滚动性能：三条种子轨迹作浅色背景，均值和样本标准差作前景。
- 独立延迟：浮动柱表示相邻独立条件差值，均值点与种子轨迹提供绝对得分。
- 句子 A/C：配对种子、数据集均值及既有材料块 bootstrap 95% 区间。
- 真实词例：沿用原种子0案例和目标词，展开邻近词位；黑框目标，浅紫当前可搜索尾部。
- 三线表：两个数据集并列分组，保持原数值、精度和评价分母。

每项输出 PDF、SVG 与 360 dpi PNG。`suite_overview.png` 为总览。
来源哈希、数值核验与替换记录位于 `reports/audits/publication_suite_test_v2_refined_20261005/`。
原资产备份在 `.cache/publication_suite_test_v2_refined_20261005/backup/`。
绘图程序只改变呈现，不进行训练、解码、校准或重新计算置信区间。
''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    before = snapshot()
    context, paired, word_report = verify_word()
    summary, _, sentence_sources = verify_sentence()
    by_seed = pd.read_csv(EXPORT / 'unified_context_heldout_test_v2_context_by_seed.csv')
    manifest_path = ROOT / 'data/manifests/rolling_sentence/heldout_test_v2.json'
    config_path = ROOT / 'configs/unified_rolling/v2.yaml'
    manifest = read(manifest_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert manifest['manifest_sha256'] == summary['manifest_sha256']
    source_hashes = {**word_report['exports'], **sentence_sources,
                     relative(manifest_path): sha(manifest_path), relative(config_path): sha(config_path)}
    OUT.mkdir(parents=True, exist_ok=True)
    assets, facts = [], {}
    with style(), mpl.rc_context(FONT):
        assets += retained_method()
        for key, function, inputs in [
            ('context', context_figure, (context, by_seed)),
            ('future', future_figure, (paired,)),
            ('rolling', rolling_figure, (paired,)),
            ('sentence', sentence_figure, (summary,)),
        ]:
            paths, values = function(*inputs)
            assets += paths
            facts[key] = values
        cases = case_excerpts(summary)
        assets += example_figure(cases)
        records = collect_tables(context, paired, summary, manifest, config)
        assets += render_tables(records)
    # Presentation must reconcile with the previously audited precise values.
    previous_facts = read(ROOT / 'data/results/audits/publication_style_test_v2/assets.json')['plotted_facts_percent']
    for kind in ('context', 'future', 'rolling', 'sentence'):
        for dataset, values in facts[kind].items():
            for key, value in values.items():
                old_value = previous_facts[kind][dataset][key]
                if isinstance(value, (list, float)):
                    np.testing.assert_allclose(value, old_value, atol=1e-10, rtol=0)
                else:
                    assert value == old_value
    deployed = deploy(before) if args.deploy else []
    assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items())
    assets.append(overview())
    readme()
    AUDIT.mkdir(parents=True, exist_ok=True)
    record = {
        'status': 'verified_deployed' if args.deploy else 'verified_preview',
        'split': 'test', 'sources': source_hashes,
        'source_hashes_unchanged_after_plotting': True,
        'manuscript_text_and_captions_unchanged': True,
        'manuscript_before_sha256': before['manuscript'],
        'retained_approved_method': True,
        'method_matches_current_v2_protocol': True,
        'numerical_facts': facts, 'table_source_rows': records,
        'case_excerpts': cases, 'layout_checks': BOUNDS,
        'assets': {relative(p): sha(p) for p in assets + deployed},
        'figure_count': len(FIGURES), 'table_count': len(TABLES),
        'training_or_inference_performed': False,
    }
    for item in record['table_source_rows'].values():
        item.pop('directory', None)
    (AUDIT / 'assets.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Verified frozen v2 test exports. Generated {len(FIGURES)} figures and {len(TABLES)} tables. Deployed: {bool(deployed)}.')


if __name__ == '__main__':
    main()
