"""Redraw audited test assets in the accepted style, without editing manuscript text.

Run once without --deploy to inspect the figure PNGs; --deploy replaces only the
already referenced graphics and the four external table assets. Scientific
exports, accepted validation previews, manuscript sources and captions are kept.
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

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np
import pandas as pd

from scripts.paper import method_overview_editorial as editorial
from scripts.paper.publication_suite_style import (
    BLUE, GRAY, INK, MUTED, ORANGE, PURPLE, RULE, TEAL,
    DATA_COLORS, cjk, clean, figure, panel, style,
)
from scripts.paper.update_test_only_word_results import verify as verify_word
from scripts.paper.update_heldout_sentence_results import verify as verify_sentence

OUT = ROOT / 'paper/figures/publication_suite_test_v2'
METHOD = ROOT / 'paper/figures/publication_suite/method_redesign/v2'
QA = ROOT / '.cache/publication_style_test_v2_qa'
AUDIT = ROOT / 'data/results/audits/publication_style_test_v2'
EXPORT = ROOT / 'data/results/exports'
DATASETS = [('pallier2025', 'Pallier2025'),
            ('chineseeeg2_littleprince', 'ChineseEEG-2')]
FIGURE_MAP = {
    '02_context_window': 'context_window_two_datasets_test',
    '03_future_increment': 'future_increment_two_datasets_test',
    '04_rolling_retrieval': 'rolling_revision_two_datasets_test',
    '05_sentence_revision': 'sentence_revision_v2_test',
}
TABLE_MAP = {
    '06_rolling_comparison': 'rolling_comparison_v2_test',
    '07_sentence_metrics': 'sentence_v2_test_metrics',
    '08_sentence_repairs': 'sentence_v2_test_repairs',
    '09_sentence_examples': 'sentence_v2_test_examples',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def snapshot():
    """Keep the original assets and establish the no-prose-edit boundary."""
    QA.mkdir(parents=True, exist_ok=True)
    record = QA / 'before_sha256.json'
    if record.exists():
        before = json.loads(record.read_text(encoding='utf-8'))
    else:
        sources = sorted((ROOT / 'paper/manuscript').rglob('*.tex'))
        assets = [ROOT / 'paper/figures' / (name + '.' + ext)
                  for name in ['rolling_revision_method', *FIGURE_MAP.values()]
                  for ext in ('pdf', 'svg', 'png')]
        assets += [ROOT / 'paper/tables' / (name + '.tex') for name in TABLE_MAP.values()]
        before = {'manuscript': {relative(p): sha(p) for p in sources},
                  'assets': {relative(p): sha(p) for p in assets if p.exists()}}
        for path in assets:
            if path.exists():
                destination = QA / 'backup' / path.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
        record.write_text(json.dumps(before, indent=2) + '\n', encoding='utf-8')
    assert all(sha(ROOT / p) == h for p, h in before['manuscript'].items())
    return before


def save(fig, name, directory=OUT):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with style():
        for ext in ('pdf', 'svg', 'png'):
            path = directory / f'{name}.{ext}'
            fig.savefig(path, dpi=360, facecolor='white', bbox_inches=None)
            paths.append(path)
    plt.close(fig)
    return paths


def pair_axes(height=84):
    fig = figure(height)
    axes = []
    for col, (_, label) in enumerate(DATASETS):
        left = .112 + col * .483
        panel(fig, .040 + col * .483, .948, chr(97 + col), label)
        ax = fig.add_axes([left, .245, .354, .555])
        clean(ax)
        ax.set_ylabel('Macro Top-10 (%)')
        axes.append(ax)
    return fig, axes


def trace_mean(ax, x, values, color):
    """Rows are paired seeds; dispersion is sample SD, as in the caption."""
    values = np.asarray(values)
    assert values.shape[0] == 3
    for seed, marker in enumerate(('o', 's', '^')):
        ax.plot(x, values[seed], color=color, alpha=.25, lw=.7,
                marker=marker, ms=2.4, mec='none', zorder=2)
    mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
    ax.errorbar(x, mean, yerr=sd, color=color, lw=1.45, capsize=2.4,
                capthick=.7, marker='o', ms=3.8, mfc='white', mew=1, zorder=4)
    return mean, sd


def context_figure(context, by_seed):
    fig, axes = pair_axes()
    facts = {}
    conditions = [f'local_r{r}' for r in (0, 1, 2, 4, 8)]
    for ax, (dataset, _) in zip(axes, DATASETS):
        data = context[(context.dataset == dataset) & (context.population == 'all_targets')].set_index('condition')
        raw = by_seed[(by_seed.dataset == dataset) & (by_seed.population == 'all_targets')]
        values = raw.pivot(index='seed', columns='condition', values='macro_recall_at_10').loc[[0, 1, 2], conditions].to_numpy() * 100
        mean, sd = trace_mean(ax, [0, 1, 2, 4, 8], values, DATA_COLORS[dataset])
        np.testing.assert_allclose(mean, data.loc[conditions, 'macro_recall_at_10_mean'] * 100, atol=1e-10)
        full = data.loc['full_context']
        ref, spread = full.macro_recall_at_10_mean * 100, full.macro_recall_at_10_sd * 100
        ax.axhspan(ref - spread, ref + spread, color=GRAY, alpha=.14, lw=0, zorder=0)
        ax.axhline(ref, color=GRAY, lw=.9, ls=(0, (4, 2)), zorder=1)
        ax.set(xlim=(-.3, 8.3), ylim=(15, 80), xticks=[0, 1, 2, 4, 8],
               yticks=[20, 40, 60, 80], xlabel='Symmetric context radius (words)')
        facts[dataset] = {'radii': [0, 1, 2, 4, 8], 'means_percent': mean.tolist(),
                          'sample_sd_percent': sd.tolist(), 'full_mean_percent': ref,
                          'full_sample_sd_percent': spread, 'population': 'all_targets'}
    fig.legend(handles=[
        Line2D([], [], color=INK, lw=1.4, marker='o', mfc='white', ms=3, label='Local mean ± SD'),
        Line2D([], [], color=GRAY, lw=.9, ls=(0, (4, 2)), label='Full segment (band: ± SD)'),
        Line2D([], [], color=GRAY, lw=.7, alpha=.6, label='Faint paths: three seeds')],
        loc='center', bbox_to_anchor=(.52, .877), ncol=3, frameon=False,
        handlelength=2.2, columnspacing=1.5)
    fig.text(.5, .083, 'Test set · all 50-word retrieval queries · three model seeds · radius 0 retains the contextual Transformer',
             ha='center', fontsize=6, color=MUTED)
    return save(fig, '02_context_window'), facts


def future_figure(paired):
    fig, axes = pair_axes(89)
    facts = {}
    for ax, (dataset, _) in zip(axes, DATASETS):
        data = paired[paired.dataset == dataset]
        values = data.pivot(index='seed', columns='horizon', values='independent_macro_recall_at_10').loc[[0, 1, 2], ['K0', 'K1', 'K2']].to_numpy() * 100
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        color = DATA_COLORS[dataset]
        for i, value in enumerate(mean):
            previous = 0 if i == 0 else mean[i - 1]
            ax.bar(i, abs(value - previous), bottom=min(value, previous), width=.49,
                   color=GRAY if i == 0 else color, alpha=.33 if i == 0 else .75,
                   edgecolor='white', lw=.5, zorder=2)
            if i < 2:
                ax.plot([i + .245, i + 1 - .245], [value, value], color=GRAY,
                        ls=(0, (2, 2)), lw=.65)
            if i > 0:
                ax.text(i + .30, (previous + value) / 2,
                        f'{value - previous:+.2f}', ha='left', va='center', fontsize=6.3)
            ax.text(i, value + sd[i] + 3.1, f'{value:.2f}', ha='center', fontsize=6.3)
        ax.errorbar([0, 1, 2], mean, yerr=sd, fmt='o', color=INK, ms=3.4,
                    capsize=2.4, capthick=.7, lw=.9, mfc='white', zorder=4)
        ax.set(xlim=(-.5, 2.80), ylim=(0, 84), xticks=[0, 1, 2],
               xticklabels=['K0', 'K1', 'K2'], yticks=[0, 20, 40, 60, 80],
               xlabel='Available future word events')
        facts[dataset] = {'means_percent': mean.tolist(), 'sample_sd_percent': sd.tolist(),
                          'adjacent_differences_pp': np.diff(mean).tolist(), 'population': 'common_targets'}
    fig.text(.112, .868, 'Independent models · fixed two-word history', fontsize=6.5, color=MUTED)
    fig.text(.5, .103, 'Points: three-seed means ± SD. Floating bars and side labels: adjacent score differences (pp).',
             ha='center', fontsize=6, color=MUTED)
    fig.text(.5, .052, 'Test set · common 50-word retrieval queries · each condition trained separately',
             ha='center', fontsize=6, color=MUTED)
    return save(fig, '03_future_increment'), facts


def rolling_figure(paired):
    fig, axes = pair_axes()
    facts = {}
    for ax, (dataset, _) in zip(axes, DATASETS):
        data = paired[paired.dataset == dataset]
        values = data.pivot(index='seed', columns='horizon', values='rolling_macro_recall_at_10').loc[[0, 1, 2], ['K0', 'K1', 'K2']].to_numpy() * 100
        mean, sd = trace_mean(ax, [0, 1, 2], values, DATA_COLORS[dataset])
        ref = data[data.horizon == 'K2'].independent_macro_recall_at_10.mean() * 100
        ax.axhline(ref, color=GRAY, lw=.9, ls=(0, (4, 2)), zorder=1)
        ax.set(xlim=(-.15, 2.15), ylim=(25, 80), xticks=[0, 1, 2],
               xticklabels=['K0: initial', 'K1: +1 word', 'K2: +2 words'],
               yticks=[30, 40, 50, 60, 70, 80], xlabel='Rolling update stage')
        facts[dataset] = {'means_percent': mean.tolist(), 'sample_sd_percent': sd.tolist(),
                          'independent_k2_mean_percent': ref, 'population': 'common_targets'}
    fig.legend(handles=[
        Line2D([], [], color=INK, lw=1.4, marker='o', mfc='white', ms=3, label='Rolling mean ± SD'),
        Line2D([], [], color=GRAY, lw=.9, ls=(0, (4, 2)), label='Independent K2 mean'),
        Line2D([], [], color=GRAY, lw=.7, alpha=.6, label='Faint paths: three seeds')],
        loc='center', bbox_to_anchor=(.52, .877), ncol=3, frameon=False,
        handlelength=2.2, columnspacing=1.5)
    fig.text(.5, .083, 'Test set · common 50-word retrieval queries · one shared model per seed · full-coverage revision training',
             ha='center', fontsize=6, color=MUTED)
    return save(fig, '04_rolling_retrieval'), facts


def sentence_figure(summary):
    fig, axes = pair_axes(90)
    facts = {}
    for ax, (dataset, _) in zip(axes, DATASETS):
        data = summary['datasets'][dataset]
        values = np.array([[next(row['sentence_bleu1_mean'] for row in data['by_seed']
                                if row['seed'] == seed and row['condition'] == condition)
                            for condition in ('A', 'C')] for seed in (0, 1, 2)]) * 100
        for seed, marker in enumerate(('o', 's', '^')):
            ax.plot([0, 1], values[seed], color=GRAY, alpha=.68, lw=.75,
                    marker=marker, ms=3, mfc='white', mew=.6, zorder=2)
        mean = values.mean(axis=0)
        color = DATA_COLORS[dataset]
        ax.plot([0, 1], mean, color=color, lw=1.7, marker='o', ms=4,
                mfc='white', mew=1.1, zorder=4)
        for x, value in enumerate(mean):
            ax.annotate(f'{value:.2f}', (x, value), xytext=(-7 if x == 0 else 7, 7),
                        textcoords='offset points', ha='right' if x == 0 else 'left',
                        fontsize=6.5, color=color, weight='bold')
        ax.set(xlim=(-.30, 1.30), ylim=(0, 19), xticks=[0, 1],
               xticklabels=['A: fixed output', 'C: rolling revision'], yticks=[0, 5, 10, 15],
               ylabel='Mean segment BLEU-1 (%)', xlabel='Final sequence output')
        stats = data['primary_statistics']
        label = f"Gain {stats['effect'] * 100:+.2f} pp\n95% CI [{stats['ci_lower'] * 100:.2f}, {stats['ci_upper'] * 100:.2f}]"
        ax.text(.04, .97, label, transform=ax.transAxes, va='top', fontsize=6.5, linespacing=1.5)
        facts[dataset] = {'means_percent': mean.tolist(), 'seed_values_percent': values.tolist(),
                          'gain_pp': stats['effect'] * 100,
                          'ci95_pp': [stats['ci_lower'] * 100, stats['ci_upper'] * 100]}
    fig.legend(handles=[
        Line2D([], [], color=GRAY, lw=.75, marker='o', mfc='white', ms=3, label='Paired model seeds'),
        Line2D([], [], color=INK, lw=1.7, marker='o', mfc='white', ms=3, label='Dataset-coloured mean')],
        loc='center', bbox_to_anchor=(.52, .877), ncol=2, frameon=False, columnspacing=2)
    fig.text(.5, .103, 'Complete test materials · given word boundaries and counts · same candidate generation and language model',
             ha='center', fontsize=6, color=MUTED)
    fig.text(.5, .052, 'Gain intervals: material-block bootstrap · subject and seed means weighted equally',
             ha='center', fontsize=6, color=MUTED)
    return save(fig, '05_sentence_revision'), facts


def table_figure(name, headers, rows, widths, group_starts=(), bold_rows=(),
                 token_columns=(), token_status=None, directory=OUT):
    """Three-line, vector text tables, matching the accepted results-table style."""
    header_h, row_h, margin = 9, 8.4, 3
    height = 2 * margin + header_h + row_h * len(rows)
    fig = figure(height)
    ax = fig.add_axes([.025, 0, .95, 1])
    ax.set(xlim=(0, 1), ylim=(height, 0))
    ax.axis('off')
    edges = np.r_[0, np.cumsum(widths)]
    assert np.isclose(edges[-1], 1)
    for y, lw in [(margin, .85), (margin + header_h, .5), (height - margin, .85)]:
        ax.plot([0, 1], [y, y], color=INK, lw=lw)
    for col, header in enumerate(headers):
        ax.text(edges[col] + .01 if col == 0 else (edges[col] + edges[col + 1]) / 2,
                margin + header_h / 2, header, va='center',
                ha='left' if col == 0 else 'center', fontproperties=cjk(True), fontsize=7.4)
    for index, row in enumerate(rows):
        top = margin + header_h + row_h * index
        if index in group_starts and index:
            ax.plot([0, 1], [top, top], color=RULE, lw=.45)
        for col, value in enumerate(row):
            x = edges[col] + .01 if col == 0 else (edges[col] + edges[col + 1]) / 2
            color = INK
            if col in token_columns:
                correct = token_status[index][col]
                color = TEAL if correct else ORANGE
                face = '#EDF5F3' if correct else '#F8EFEB'
                ax.add_patch(FancyBboxPatch((edges[col] + .012, top + 1.6),
                    widths[col] - .024, row_h - 3.2,
                    boxstyle='round,pad=0,rounding_size=0.012',
                    facecolor=face, edgecolor='none', zorder=1))
            chinese = any(ord(c) > 127 for c in str(value)) and '±' not in str(value)
            kwargs = {'fontproperties': cjk(index in bold_rows)} if chinese else {'weight': 'bold' if index in bold_rows else 'normal'}
            ax.text(x, top + row_h / 2, str(value), va='center', ha='left' if col == 0 else 'center',
                    fontsize=7.7, color=color, zorder=2, **kwargs)
    return save(fig, name, directory)


def tables(paired, summary):
    paths, facts = [], {}
    rows = []
    for dataset, label in DATASETS:
        for horizon in range(3):
            data = paired[(paired.dataset == dataset) & (paired.horizon == f'K{horizon}')]
            independent = data.independent_macro_recall_at_10.to_numpy() * 100
            rolling = data.rolling_macro_recall_at_10.to_numpy() * 100
            rows.append([label if horizon == 0 else '', f'K{horizon}',
                         f'{independent.mean():.2f} ± {independent.std(ddof=1):.2f}',
                         f'{rolling.mean():.2f} ± {rolling.std(ddof=1):.2f}',
                         f'{(rolling - independent).mean():+.2f}'])
    paths += table_figure('06_rolling_comparison',
                         ['数据集', '状态', '独立模型', '滚动模型', 'Δ（百分点）'],
                         rows, [.23, .10, .245, .245, .18], group_starts=[3])
    facts['06_rolling_comparison'] = rows
    metrics, repairs, examples, statuses = [], [], [], []
    for dataset, label in DATASETS:
        data = summary['datasets'][dataset]
        for condition in ('A', 'C'):
            row = next(r for r in data['summary'] if r['condition'] == condition)
            metrics.append([label if condition == 'A' else '', condition,
                            *[f'{row[m] * 100:.2f}' for m in ('sentence_bleu1_mean', 'wer', 'aligned_word_accuracy')]])
        r = data['revisions']
        repairs.append([label, str(r['eligible']), str(r['rolling_repaired']), str(r['rolling_harmed']),
                        f"{r['rolling_repaired_rate'] * 100:.2f}", f"{r['rolling_harmed_rate'] * 100:.2f}"])
        for selection, tag in [('most_C_repairs', '改对'), ('most_C_harms', '误改')]:
            example = next(e for e in data['examples'] if e['selection'] == selection)
            assert example['seed'] == 0
            trace = {step['arrival']: step['C'] for step in example['trace']}
            initial = [trace[i][i] for i in range(len(example['reference']))]
            final = example['final']['C']
            positions = [i for i, ref in enumerate(example['reference'][:-1]) if
                         (initial[i] != ref and final[i] == ref if tag == '改对' else initial[i] == ref and final[i] != ref)]
            assert positions
            position = positions[0]
            reference = example['reference'][position]
            tokens = [trace[position + offset][position] if position + offset in trace else '--' for offset in (0, 1, 2)]
            examples.append([label, example['subject'], tag, str(position + 1), reference, *tokens])
            statuses.append({5 + j: token == reference for j, token in enumerate(tokens)})
    paths += table_figure('07_sentence_metrics',
                         ['数据集', '条件', 'BLEU-1 ↑', 'WER ↓', '对齐词准确率 ↑'],
                         metrics, [.25, .10, .19, .19, .27], group_starts=[2], bold_rows=[1, 3])
    paths += table_figure('08_sentence_repairs',
                         ['数据集', '可回修词位', '改对数', '误改数', '改对率（%）', '误改率（%）'],
                         repairs, [.24, .17, .13, .13, .165, .165])
    paths += table_figure('09_sentence_examples',
                         ['数据集', '受试者', '变化', '词位', '参考词', '$t=i$', '$t=i+1$', '$t=i+2$'],
                         examples, [.22, .115, .085, .065, .095, .12, .14, .16],
                         group_starts=[2], token_columns=[5, 6, 7], token_status=statuses)
    facts.update({'07_sentence_metrics': metrics, '08_sentence_repairs': repairs, '09_sentence_examples': examples})
    # Preserve all old table values and row selections, including their precision.
    for name, rows in facts.items():
        original = (QA / 'backup/paper/tables' / (TABLE_MAP[name] + '.tex')).read_text(encoding='utf-8')
        original_rows = [line for line in original.splitlines() if ' & ' in line][1:]
        expected_rows = []
        for line in original_rows:
            expected_rows.append([v.strip().replace('$', '').replace('\\pm', '±')
                                  for v in line.removesuffix('\\\\').split(' & ')])
        assert rows == expected_rows, (name, rows, expected_rows)
    return paths, facts


def method_figure():
    fig = figure(169)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 183), ylim=(169, 0))
    ax.axis('off')
    editorial.overview(ax)
    editorial.revision_story(ax)
    replacements = {
        '250-word vocabulary': 'Open-vocabulary search',
        'Top 5 per word + causal LM': 'Top 20 · Qwen LM · beam 5',
        'Target': 'Target $h_i$',
    }
    changed = set()
    for artist in ax.texts:
        if artist.get_text() in replacements:
            changed.add(artist.get_text())
            artist.set_text(replacements[artist.get_text()])
    assert changed == set(replacements)
    # Reserve blue for the newest neural window, as stated in the manuscript caption.
    for patch in ax.patches:
        if isinstance(patch, Circle) and patch.center == (119, 36.5):
            patch.set_facecolor(PURPLE)
    # Preserve the accepted geometry while making the lower mechanism legible.
    for artist in ax.texts:
        if artist.get_position()[1] >= 70 and artist.get_fontsize() < 7.5:
            artist.set_fontsize(min(7.3, artist.get_fontsize() * 1.20))
    editorial.line(ax, [6, 177], [159, 159], color=editorial.RULE, lw=.6)
    editorial.text(ax, 6, 162.3,
        'Shared neural model · three-word beam search (width 5), conditioned on committed output', 6.5, color=MUTED)
    editorial.text(ax, 6, 166,
        'Each window ends 1 s after word onset. Signals, feature glyphs and word tokens are schematic.', 6.5, color=MUTED)
    return save(fig, 'method_overview_editorial', METHOD)


def deploy(before):
    paths = []
    for src, destination in FIGURE_MAP.items():
        for ext in ('pdf', 'svg', 'png'):
            target = ROOT / 'paper/figures' / f'{destination}.{ext}'
            shutil.copy2(OUT / f'{src}.{ext}', target)
            paths.append(target)
    for ext in ('pdf', 'svg', 'png'):
        target = ROOT / 'paper/figures' / f'rolling_revision_method.{ext}'
        shutil.copy2(METHOD / f'method_overview_editorial.{ext}', target)
        paths.append(target)
    for src, name in TABLE_MAP.items():
        target = ROOT / 'paper/tables' / f'{name}.tex'
        target.write_text('% Figure asset generated from audited test exports; caption remains in manuscript.\n'
                          + r'\includegraphics[width=\linewidth]{publication_suite_test_v2/' + src + '.pdf}\n', encoding='utf-8')
        paths.append(target)
    assert all(sha(ROOT / p) == h for p, h in before['manuscript'].items())
    return paths


def readme():
    (OUT / 'README.md').write_text('''# 统一协议 v2：测试集图表

本文件夹只保存正式测试集结果绘制的图表。每项有 PDF、SVG 和 360 dpi PNG。
采用「选择论文结果图表风格」对话中认可的 publication_suite 配色、字体和三线表样式。
方法图单独保存在 `../publication_suite/method_redesign/v2/method_overview_editorial.*`。

## 已替换的论文图表

| 本目录文件 | 正文既有图表 |
|---|---|
| 02_context_window | 局部神经上下文有效范围 |
| 03_future_increment | 独立延迟模型的未来上下文增量 |
| 04_rolling_retrieval | 共享模型的滚动修正曲线 |
| 05_sentence_revision | 句子 A/C 测试集比较 |
| 06_rolling_comparison | 独立模型与滚动模型对照表 |
| 07_sentence_metrics | 句子 BLEU-1、WER 与对齐词准确率表 |
| 08_sentence_repairs | C 自身首次输出到终稿的改对／误改表 |
| 09_sentence_examples | 实际词位回修示例表 |

论文源文件与图注逐字保留；替换的是已引用的图片文件和外部表格资产。
局部半径使用数值坐标 0、1、2、4、8；所有词级误差棒为三个种子的样本标准差。
未来上下文图保留瀑布图样式，同时显示均值点及标准差；浮动柱是相邻独立模型的得分差。
滚动曲线只使用共同查询，参考线为独立 K2 的三种子均值。
句子图的灰线连接同一种子，蓝线表示三种子均值；增益区间来自既有材料块 bootstrap。
句子表加粗只标识 C；示例单元格绿色表示与参考词一致，橙色表示不一致。
示例、计数、分母、数值精度与原正文表格一致。图中示例不代表典型解码质量。

## 未采用的旧套件图

| 旧图 | 本次处理及原因 |
|---|---|
| 01_method_overview | 按用户选择，用 method_overview_editorial 替代 |
| 02_context_future 合并版 | 正文原有两张图，分别重画，保持原有框架 |
| 03_rolling_retrieval 下方差值面板 | 旧版为单种子验证结果；正文已用三种子测试表呈现相同问题 |
| 04_signal_control | 缺少新协议全量测试集错位／机会对照结果，不能直接使用旧验证数字 |
| 05_effect_estimates 完整多指标对照 | 旧版含冻结／错位条件及不同区间口径；本次只将当前 A/C BLEU-1 区间用于既有句子图 |
| 06_participant_gains | 正文没有对应图位；测试集受试者导出可支持另画，需另行决定是否加入 |
| 07_latency_benefit | 旧版等待时间与计算耗时诊断不属于当前测试汇总；需正式测试集时间诊断导出 |
| 08_sentence_revision 完整组合图 | 相邻改写事件统计与当前 C 首次到最终的词位统计分母不同；只采用词例的配色样式 |
| 09_results_table 旧条件与数字 | 旧版为验证集且含随机／错位条件；本次仅采用三线表样式，使用当前 A/C 测试结果 |

旧预览套件保留在原位置；本次没有增加实验、重新训练或运行解码。

## 正式数据来源

- `reports/exports/unified_context_heldout_test_v2_context_by_seed.csv`
- `reports/exports/unified_context_heldout_test_v2_context_summary.csv`
- `reports/exports/unified_context_heldout_test_v2_independent_vs_rolling_by_seed.csv`
- `reports/exports/unified_heldout_test_v2_sentence_summary.json`
- `reports/exports/unified_heldout_test_v2_sentence_by_subject_seed.csv`
- `reports/exports/unified_heldout_test_v2_sentence_repairs.csv`

完整来源核验与资产哈希保存在 `reports/audits/publication_style_test_v2/assets.json`。
绘图程序：`scripts/paper/apply_publication_style_test_v2.py`。
''', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    before = snapshot()
    context, paired, word_report = verify_word()
    summary, _, sentence_sources = verify_sentence()
    by_seed = pd.read_csv(EXPORT / 'unified_context_heldout_test_v2_context_by_seed.csv')
    OUT.mkdir(parents=True, exist_ok=True)
    assets, facts = [], {}
    with style():
        assets += method_figure()
        for key, build, data in [
            ('context', context_figure, (context, by_seed)),
            ('future', future_figure, (paired,)),
            ('rolling', rolling_figure, (paired,)),
            ('sentence', sentence_figure, (summary,)),
            ('tables', tables, (paired, summary)),
        ]:
            paths, values = build(*data)
            assets += paths
            facts[key] = values
    readme()
    deployed = deploy(before) if args.deploy else []
    AUDIT.mkdir(parents=True, exist_ok=True)
    record = {'status': 'verified_deployed' if args.deploy else 'verified_preview',
              'split': 'test', 'style_reference': '01a0c9fc-081e-7ca3-9603-8d3447b55d0c',
              'test_manifests': [word_report['manifest_sha256'], summary['manifest_sha256']],
              'source_exports': {**word_report['exports'], **sentence_sources},
              'source_verification': 'Existing complete-test verifiers reconcile source hashes, seed means, sample SD, subject means, repairs, and frozen populations.',
              'manuscript_before_sha256': before['manuscript'],
              'manuscript_after_sha256': {p: sha(ROOT / p) for p in before['manuscript']},
              'manuscript_sources_changed': False, 'captions_changed': False,
              'method': {'layout': 'method_overview_editorial',
                         'graphic_label_updates': ['open vocabulary', 'Top 20 / Qwen LM / beam 5', 'target h_i'],
                         'schematic': True},
              'assets': {relative(p): sha(p) for p in [*assets, *deployed]},
              'plotted_facts_percent': facts}
    (AUDIT / 'assets.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Verified test sources; wrote {len(assets)} figure files. Deployed: {bool(deployed)}. Manuscript text unchanged.')


if __name__ == '__main__':
    main()
