"""Draw and insert the two requested saved-test diagnostics in results2.tex."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from scripts.paper.publication_suite_style import (
    DATA_COLORS, GRAY, INK, MUTED, ORANGE, RULE, TEAL, clean, figure, style,
)
from scripts.paper.refine_publication_style import FONT, DATASETS, header
from scripts.paper.update_heldout_sentence_results import verify as verify_sentence

EXPORT = ROOT / 'data/results/exports'
PREFIX = 'unified_heldout_test_v2_revision_diagnostics'
OUT = ROOT / 'paper/figures/publication_suite_test_v2_additions'
AUDIT = ROOT / 'data/results/audits/publication_suite_test_v2_additions_20261005'
QA = ROOT / '.cache/publication_suite_test_v2_additions_20261005'
FIGURES = {
    '13_display_edit_composition': 'sentence_display_edit_composition_v2_test',
    '14_participant_gains': 'sentence_participant_gains_v2_test',
}
OUTCOMES = ['wrong_to_correct', 'correct_to_wrong', 'wrong_to_different_wrong']
COLORS = [TEAL, ORANGE, GRAY]
CHECKS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def snapshot():
    QA.mkdir(parents=True, exist_ok=True)
    record = QA / 'before_sha256.json'
    if record.exists():
        return read(record)
    paths = list((ROOT / 'paper/manuscript').glob('*.tex'))
    paths.append(ROOT / 'output/pdf/manuscript_unified_protocol_draft.pdf')
    before = {relative(path): sha(path) for path in paths}
    for path in paths:
        destination = QA / 'backup' / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    record.write_text(json.dumps(before, indent=2) + '\n', encoding='utf-8')
    return before


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bad = []
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.x1 + 1 or bounds.y1 > fig.bbox.y1 + 1:
                bad.append(text.get_text())
    assert not bad, (name, bad)
    paths = []
    for ext in ('pdf', 'svg', 'png'):
        path = OUT / f'{name}.{ext}'
        fig.savefig(path, dpi=360, facecolor='white', bbox_inches=None)
        paths.append(path)
    plt.close(fig)
    CHECKS.append({'asset': name, 'text_outside_canvas': 0})
    return paths


def composition_figure(totals):
    fig = figure(106)
    facts = {}
    detail_max = 2 * np.ceil((totals[[k + '_share' for k in OUTCOMES[:2]]].to_numpy().max() * 100 + 1.5) / 2)
    for col, (dataset, label) in enumerate(DATASETS):
        header(fig, .040 + col * .492, .955, chr(97 + col), label, dataset)
        left = .143 + col * .486
        ax = fig.add_axes([left, .51, .328, .275])
        clean(ax)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=6)
        ax.set(xlim=(0, 100), ylim=(-.4, 1.55), xticks=[0, 50, 100], yticks=[1, 0],
               yticklabels=['First revision\nK0 → K1', 'Second revision\nK1 → K2'],
               xlabel='All changed events (%)')
        detail = fig.add_axes([left, .12, .328, .215])
        clean(detail)
        detail.spines['left'].set_visible(False)
        detail.tick_params(axis='y', length=0, pad=6)
        detail.set(xlim=(0, detail_max), ylim=(-.42, 1.42),
                   xticks=np.arange(0, detail_max + .1, 2), yticks=[1, 0],
                   yticklabels=['K0 → K1', 'K1 → K2'],
                   xlabel='Share of all changed events (%)')
        detail.set_title('Correctness-changing edits · detail', fontsize=7,
                         color=INK, loc='left', pad=9)
        data = totals[totals.dataset == dataset].set_index('stage')
        facts[dataset] = []
        for stage, y in ((1, 1), (2, 0)):
            row = data.loc[stage]
            n = int(row.changed_events)
            assert n > 0
            values = [float(row[key + '_share']) * 100 for key in OUTCOMES]
            np.testing.assert_allclose(sum(values), 100, rtol=0, atol=1e-9)
            left = 0
            for color, width in zip(COLORS, values):
                ax.barh(y, width, left=left, height=.32, color=color, edgecolor='white', lw=.7)
                left += width
            ax.text(sum(values[:2]) + values[2] / 2, y, f'{values[2]:.1f}%',
                    color=INK, fontsize=7.8, ha='center', va='center')
            ax.text(100, y + .29, f'{n:,} changed events', color=MUTED,
                    fontsize=6.7, ha='right', va='center')
            for offset, color, value in zip([.15, -.15], COLORS[:2], values[:2]):
                detail.barh(y + offset, value, height=.23, color=color,
                            edgecolor='white', lw=.5)
                detail.text(value + .13, y + offset, f'{value:.1f}%',
                            fontsize=6.7, color=color, ha='left', va='center')
            facts[dataset].append({'stage': stage, 'eligible_updates': int(row.eligible_updates),
                                   'changed_events': n, 'counts': [int(row[k]) for k in OUTCOMES],
                                   'percentages': values})
    fig.legend(handles=[Patch(facecolor=c, edgecolor='none', label=label) for c, label in zip(
        COLORS, ['Wrong → correct', 'Correct → wrong', 'Wrong → different wrong'])],
         loc='center', bbox_to_anchor=(.51, .87), ncol=3, frameon=False,
        handlelength=1.6, handleheight=.7, columnspacing=2.4)
    return save(fig, '13_display_edit_composition'), facts


def participant_figure(gains):
    fig = figure(118)
    facts = {}
    minimum, maximum = gains.sentence_bleu1_gain_pp.min(), gains.sentence_bleu1_gain_pp.max()
    lower = min(-2, np.floor(minimum / 2) * 2)
    upper = max(8, np.ceil(maximum / 2) * 2)
    ticks = np.arange(lower, upper + .1, 2)
    for col, (dataset, label) in enumerate(DATASETS):
        data = gains[gains.dataset == dataset]
        subjects = sorted(data.subject.unique())
        header(fig, .040 + col * .492, .955, chr(97 + col), f'{label} · {len(subjects)} participants', dataset)
        ax = fig.add_axes([.112 + col * .489, .19, .351, .60])
        clean(ax)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=5)
        ax.axvline(0, color=GRAY, lw=.8, ls=(0, (3, 2)), zorder=0)
        ax.set(xlim=(lower, upper), ylim=(len(subjects) - .5, -.5), xticks=ticks,
               yticks=range(len(subjects)), yticklabels=subjects,
               xlabel='C minus A · mean segment BLEU-1 (pp)')
        values = data.pivot(index='subject', columns='seed', values='sentence_bleu1_gain_pp').loc[subjects, [0, 1, 2]].to_numpy()
        means = values.mean(axis=1)
        color = DATA_COLORS[dataset]
        for y, (seed_values, mean) in enumerate(zip(values, means)):
            for seed, (value, offset, marker) in enumerate(zip(seed_values, [-.20, 0, .20], ['o', 's', '^'])):
                ax.plot([0, value], [y + offset, y + offset], color=RULE, lw=.8, zorder=1)
                ax.plot(value, y + offset, marker=marker, ls='none', color=color,
                        mfc=color, alpha=.56, ms=3.4, mew=.45, zorder=2)
            ax.plot([0, mean], [y, y], color=color, lw=1.0, alpha=.35, zorder=1)
            ax.plot(mean, y, 'o', mfc='none', mec=color, ms=6.2, mew=1.2, zorder=4)
        facts[dataset] = {'subject_order': subjects, 'seed_gains_pp': values.tolist(),
                          'subject_means_pp': means.tolist(), 'dataset_mean_pp': float(means.mean())}
    fig.legend(handles=[
        *[Line2D([], [], marker=m, ms=3.5, lw=0, color=GRAY, label=f'Seed {s}')
          for s, m in enumerate(['o', 's', '^'])],
        Line2D([], [], marker='o', mfc='none', mec=INK, ms=6, mew=1.2, lw=0, color=INK,
               label='Participant mean across seeds')],
        loc='center', bbox_to_anchor=(.515, .868), ncol=4, frameon=False,
        columnspacing=2.8, handlelength=1.3)
    fig.text(.112, .065, 'Paired held-out results · subjects kept in ID order · positive values favour rolling revision C',
             fontsize=6.4, color=MUTED, va='bottom')
    return save(fig, '14_participant_gains'), facts


PARTICIPANT_BLOCK = r'''% BEGIN requested_participant_gains_v2_test
图~\ref{fig:sentence_participant_gains} 给出各受试者的配对增益。

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{sentence_participant_gains_v2_test.pdf}
    \caption{测试集受试者层面的句子回修收益。横轴为同一受试者和种子的 C 减 A 平均片段 BLEU-1（百分点）。小符号表示三个种子，大空心圆为该受试者的种子均值，竖虚线表示零增益；横线从零延伸至相应增益，仅用于显示方向与幅度。受试者按编号排列，训练重复不作为新增受试者。}
    \label{fig:sentence_participant_gains}
\end{figure}
% END requested_participant_gains_v2_test
'''

EDITS_BLOCK = r'''% BEGIN requested_display_edits_v2_test
图~\ref{fig:sentence_display_edit_composition} 汇总两次回修实际改写的类型组成。

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{sentence_display_edit_composition_v2_test.pdf}
    \caption{测试集 C 条件实际改写的类型组成，分别统计 K0 到 K1、K1 到 K2 的回修。每条横柱的分母为该数据集和阶段发生文本变化的目标更新次数，合并三个种子；首次显示和未变化输出不计入。青绿色、橙色、灰色分别表示错改对、对改错、错改为另一错误。柱内数值为灰色类别的比例，柱旁标注改写总次数。各阶段仅纳入当时可回修的词位；同一词若两次改写可贡献两个事件，因此本图与首次到最终输出的词位统计口径不同。}
    \label{fig:sentence_display_edit_composition}
\end{figure}
% END requested_display_edits_v2_test
'''


def remove_blocks(text):
    for name in ('participant_gains', 'display_edits'):
        text = re.sub(r'\n% BEGIN requested_' + name + r'_v2_test\n.*?% END requested_' + name + r'_v2_test\n\n?',
                      '', text, flags=re.S)
    return text


def insert_after_float(text, label, block):
    match = re.search(r'\\begin\{(?:figure|table)\}.*?\\label\{' + re.escape(label) + r'\}.*?\\end\{(?:figure|table)\}\n', text, flags=re.S)
    assert match, label
    return text[:match.end()] + '\n' + block + '\n' + text[match.end():]


def deploy(before):
    paths = []
    for source, destination in FIGURES.items():
        for ext in ('pdf', 'svg', 'png'):
            path = ROOT / 'paper/figures' / f'{destination}.{ext}'
            shutil.copy2(OUT / f'{source}.{ext}', path)
            paths.append(path)
    manuscript = ROOT / 'paper/manuscript/results2.tex'
    original = (QA / 'backup/paper/manuscript/results2.tex').read_text(encoding='utf-8')
    current = manuscript.read_text(encoding='utf-8')
    assert remove_blocks(current) == original, 'Existing author text changed outside the requested figure additions.'
    text = insert_after_float(original, 'tab:sentence_revision_full', PARTICIPANT_BLOCK)
    text = insert_after_float(text, 'fig:sentence_examples_full', EDITS_BLOCK)
    assert remove_blocks(text) == original
    assert text.count(r'\label{fig:sentence_participant_gains}') == 1
    assert text.count(r'\label{fig:sentence_display_edit_composition}') == 1
    manuscript.write_text(text, encoding='utf-8')
    for path, expected in before.items():
        if path.startswith('paper/manuscript/') and path != relative(manuscript):
            assert sha(ROOT / path) == expected
    paths.append(manuscript)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deploy', action='store_true')
    args = parser.parse_args()
    before = snapshot()
    summary, _, _ = verify_sentence()
    metadata_path = EXPORT / f'{PREFIX}_sources.json'
    metadata = read(metadata_path)
    assert metadata['split'] == 'test' and metadata['manifest_sha256'] == summary['manifest_sha256']
    assert metadata['status'] == 'verified_descriptive_saved_test_diagnostics'
    assert all(sha(ROOT / p) == h for p, h in {**metadata['sources'], **metadata['exports']}.items())
    totals = pd.read_csv(EXPORT / f'{PREFIX}_display_edits_summary.csv')
    gains = pd.read_csv(EXPORT / f'{PREFIX}_participant_gains.csv')
    OUT.mkdir(parents=True, exist_ok=True)
    assets, facts = [], {}
    with style(), mpl.rc_context(FONT):
        for key, builder, data in [('edit_composition', composition_figure, totals),
                                   ('participant_gains', participant_figure, gains)]:
            paths, values = builder(data)
            assets += paths
            facts[key] = values
    for dataset, _ in DATASETS:
        np.testing.assert_allclose(facts['participant_gains'][dataset]['dataset_mean_pp'],
                                   summary['datasets'][dataset]['primary_statistics']['effect'] * 100,
                                   atol=1e-10, rtol=0)
    deployed = deploy(before) if args.deploy else []
    assert all(sha(ROOT / p) == h for p, h in {**metadata['sources'], **metadata['exports']}.items())
    (OUT / 'README.md').write_text('''# 新增两张测试集诊断图

13_display_edit_composition：两次回修实际改写的类型组成，分母为发生变化的目标更新事件。
14_participant_gains：同受试者、同种子的 C−A 平均片段 BLEU-1 增益；编号排序，完整保留负增益。

数据来源：reports/exports/unified_heldout_test_v2_revision_diagnostics_*。
统计导出：python -m braindecoding.unified_heldout_test_v2_revision_diagnostics。
绘图与插入：python scripts/paper/add_revision_diagnostic_figures_test_v2.py --deploy。
每图包含矢量 PDF、可编辑 SVG 和 360 dpi PNG。
仅新增图块和定义性图注，原正文、图表和实验协议保留。
''', encoding='utf-8')
    AUDIT.mkdir(parents=True, exist_ok=True)
    record = {'status': 'verified_deployed' if args.deploy else 'verified_preview',
              'split': 'test', 'diagnostic_export_manifest': relative(metadata_path),
              'diagnostic_export_manifest_sha256': sha(metadata_path),
              'sources': {**metadata['sources'], **metadata['exports']},
              'figures': FIGURES, 'facts': facts, 'layout_checks': CHECKS,
              'manuscript_change': 'Two requested figure blocks with descriptive captions and cross-references; original author text preserved.',
              'assets': {relative(p): sha(p) for p in assets + deployed}}
    (AUDIT / 'assets.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Generated two saved-test diagnostic figures. Deployed in results2.tex: {bool(deployed)}.')


if __name__ == '__main__':
    main()
