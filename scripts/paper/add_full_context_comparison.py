"""Present the completed frozen A/C/F comparison without changing its analysis."""
from __future__ import annotations

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
import numpy as np
import pandas as pd

from scripts.paper.publication_suite_style import DATA_COLORS, GRAY, INK, MUTED, RULE, clean, figure, style
from scripts.paper import refine_publication_style as renderer

EXPORT = ROOT / 'data/results/exports'
OUT = ROOT / 'paper/figures/publication_suite_test_v2_full_context'
AUDIT = ROOT / 'data/results/audits/manuscript_full_context_comparison_20261005'
DATASETS = [('pallier2025', 'Pallier2025'), ('chineseeeg2_littleprince', 'ChineseEEG-2')]
CONDITIONS = ['A', 'C', 'F']
WAIT_STRATEGIES = ['K0', 'C_K2', 'F']
PLOT = 'sentence_full_context_tradeoff_v2_test'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-11)


def verify():
    summary_path = EXPORT / 'full_context_sentence_test_v2_summary.json'
    summary = read(summary_path)
    audit = read(ROOT / 'data/results/studies/full_context_sentence_test_v2/audit.json')
    manifest = read(ROOT / 'data/manifests/rolling_sentence/full_context_sentence_test_v2.json')
    identity = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != 'manifest_sha256'},
                                        ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert identity == manifest['manifest_sha256'] == summary['manifest_sha256'] == audit['manifest_sha256']
    assert summary['status'] == 'completed' and summary['split'] == 'test'
    assert summary['conditions'] == CONDITIONS and summary['original_AC_unchanged'] is True
    assert summary['new_significance_tests'] is False and audit['status'] == 'verified_complete'
    assert sha(summary_path) == audit['result_sha256']
    assert audit['total_segment_seed_units'] == manifest['total_segment_seed_units']
    for rel, expected in summary['exports'].items():
        assert sha(ROOT / rel) == expected == audit['exports'][rel], rel
    assert len(summary['sources']) == len(manifest['jobs'])
    for source in summary['sources']:
        assert sha(ROOT / source['path']) == source['sha256'], source['path']
    original = read(EXPORT / 'unified_heldout_test_v2_sentence_summary.json')
    original_audit = read(ROOT / 'data/results/studies/unified_heldout_test_v2/audit.json')
    assert sha(EXPORT / 'unified_heldout_test_v2_sentence_summary.json') == original_audit['report_sha256']
    for source in original_audit['sources']:
        assert sha(ROOT / source['path']) == source['sha256'], source['path']
    subjects = pd.read_csv(EXPORT / 'full_context_sentence_test_v2_by_subject_seed.csv')
    events = pd.read_csv(EXPORT / 'full_context_sentence_test_v2_input_wait_by_event.csv')
    waits = pd.read_csv(EXPORT / 'full_context_sentence_test_v2_input_wait_summary.csv')
    assert not subjects.duplicated(['dataset', 'seed', 'subject', 'condition']).any()
    assert not events.duplicated(['dataset', 'event_id']).any()
    close(events.k0_input_wait_s, 1)
    assert (events.c_input_wait_s <= events.f_input_wait_s + 1e-10).all()
    for _, group in events.groupby(['dataset', 'group'], sort=False):
        group = group.sort_values('position')
        remaining = group.f_input_wait_s.to_numpy()
        positions = np.arange(len(group))
        for offset, col in [(1, 'k1_input_wait_s'), (2, 'c_input_wait_s')]:
            expected = remaining - remaining[np.minimum(positions + offset, len(group) - 1)] + 1
            close(group[col], expected)
    metrics = ['sentence_bleu1_mean', 'wer', 'bleu1', 'aligned_word_accuracy']
    for dataset, values in summary['datasets'].items():
        for row in original['datasets'][dataset]['summary']:
            current = next(v for v in values['summary'] if v['condition'] == row['condition'])
            for metric in metrics:
                close(row[metric], current[metric])
        for row in values['summary']:
            part = subjects[(subjects.dataset == dataset) & (subjects.condition == row['condition'])]
            assert len(part) == values['counts']['subjects'] * 3
            for metric in metrics:
                close(part.groupby('seed')[metric].mean().mean(), row[metric])
        for row in values['by_seed']:
            part = subjects[(subjects.dataset == dataset) & (subjects.condition == row['condition']) & (subjects.seed == row['seed'])]
            for metric in metrics:
                close(part[metric].mean(), row[metric])
        means = {r['condition']: r for r in values['summary']}
        close(means['C']['sentence_bleu1_mean'] - means['F']['sentence_bleu1_mean'], values['additional_C_minus_F']['effect'])
        for population in ['all_events', 'common50']:
            data = events[events.dataset == dataset]
            if population == 'common50':
                data = data[data.common50]
            count_field = 'events' if population == 'all_events' else 'strict50_common'
            assert len(data) == values['counts'][count_field]
            for strategy, column in [('K0', 'k0_input_wait_s'), ('K1', 'k1_input_wait_s'), ('C_K2', 'c_input_wait_s'), ('F', 'f_input_wait_s')]:
                row = waits[(waits.dataset == dataset) & (waits.population == population) & (waits.strategy == strategy)].iloc[0]
                assert row.queries == len(data)
                close(data[column].mean(), row.mean_input_wait_s)
                close(data[column].median(), row.median_input_wait_s)
                close(data[column].quantile(.9), row.p90_input_wait_s)
                close(data[column].mean() - 1, row.mean_extra_wait_after_k0_s)
                close((data.f_input_wait_s - data[column]).mean(), row.mean_saved_vs_f_s)
    return summary, waits, {summary_path.relative_to(ROOT).as_posix(): sha(summary_path), **summary['exports']}


def save(fig, name):
    fig.canvas.draw()
    box = fig.bbox
    paint = fig.canvas.get_renderer()
    for artist in fig.findobj(mpl.text.Text):
        if artist.get_visible() and artist.get_text():
            bounds = artist.get_window_extent(paint)
            assert bounds.x0 >= box.x0 - 1 and bounds.y0 >= box.y0 - 1 and bounds.x1 <= box.x1 + 1 and bounds.y1 <= box.y1 + 1, artist.get_text()
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=360, facecolor='white', bbox_inches=None)
    plt.close(fig)


def tradeoff(summary, waits):
    fig = figure(105)
    facts = {}
    for col, (dataset, label) in enumerate(DATASETS):
        left = .110 + col * .491
        color = DATA_COLORS[dataset]
        renderer.header(fig, .040 + col * .491, .955, chr(97 + col), label, dataset)
        ax = fig.add_axes([left, .300, .348, .48])
        clean(ax)
        ax.grid(axis='y', color='#E5EAED', lw=.45)
        ax.set_axisbelow(True)
        data = summary['datasets'][dataset]
        timing = waits[(waits.dataset == dataset) & (waits.population == 'all_events')].set_index('strategy')
        x = timing.loc[WAIT_STRATEGIES, 'mean_input_wait_s'].to_numpy()
        seed_rows = pd.DataFrame(data['by_seed'])
        y = seed_rows.pivot(index='seed', columns='condition', values='sentence_bleu1_mean').loc[[0, 1, 2], CONDITIONS].to_numpy() * 100
        for seed, marker in enumerate(['o', 's', '^']):
            ax.plot(x, y[seed], color=color, lw=.8, alpha=.30, marker=marker,
                    ms=3.1, mew=.45, mfc=color, zorder=2)
        means = np.array([next(r['sentence_bleu1_mean'] for r in data['summary'] if r['condition'] == c) for c in CONDITIONS]) * 100
        close(y.mean(axis=0), means)
        ax.plot(x, means, color=color, lw=1.65, marker='o', mfc='white', ms=5.6, mew=1.25, zorder=4)
        for xx, yy, text, alignment, dx in zip(x, means, ['Fixed A', 'Rolling C', 'Full F'], ['right', 'left', 'center'], [-.06, .08, 0]):
            ax.text(xx + dx, yy + .9, text, color=color, fontsize=7.1, ha=alignment, va='bottom')
        saved = float(timing.loc['C_K2', 'mean_saved_vs_f_s'])
        ax.annotate('', xy=(x[2], 1.2), xytext=(x[1], 1.2), arrowprops={'arrowstyle': '<->', 'color': MUTED, 'lw': .8})
        ax.text((x[1] + x[2]) / 2, 1.95, f'{saved:.2f} s less waiting', ha='center', va='bottom', color=MUTED, fontsize=6.6)
        ax.set(xlim=(0, 3.85), ylim=(0, 18.5), xticks=[0, 1, 2, 3], yticks=[0, 5, 10, 15],
               xlabel='Mean input waiting time (s)', ylabel='Mean segment BLEU-1 (%)')
        stat = data['additional_C_minus_F']
        effect, low, high = [float(stat[k]) * 100 for k in ['effect', 'ci_lower', 'ci_upper']]
        fig.text(left, .145, f'C − F: {effect:+.2f} pp', fontsize=7.2, color=INK, weight='bold')
        fig.text(left, .096, f'Paired 95% CI [{low:+.2f}, {high:+.2f}]', fontsize=6.7, color=MUTED)
        facts[dataset] = {'conditions': CONDITIONS, 'mean_input_wait_s': x.tolist(),
                          'seed_scores_percent': y.tolist(), 'mean_scores_percent': means.tolist(),
                          'C_minus_F_pp': effect, 'paired_material_block_ci95_pp': [low, high],
                          'mean_wait_saved_C_vs_F_s': saved, 'waiting_population': 'all_events'}
    fig.legend(handles=[Line2D([], [], color=GRAY, lw=.8, marker='s', ms=3, alpha=.55, label='Three paired seed trajectories'),
                        Line2D([], [], color=INK, lw=1.65, marker='o', mfc='white', ms=5, mew=1.2, label='Mean across seeds')],
               loc='center', bbox_to_anchor=(.52, .868), ncol=2, frameon=False, columnspacing=2.8)
    fig.text(.110, .025, 'Input waiting excludes alignment, preprocessing, computation and display.', color=MUTED, fontsize=6.4)
    save(fig, PLOT)
    return facts


def metrics_table(summary, waits):
    headers = ['解码条件', 'BLEU-1 ↑', 'WER ↓', '词准确率 ↑', '输入等待\n(s) ↓', 'BLEU-1 ↑', 'WER ↓', '词准确率 ↑', '输入等待\n(s) ↓']
    rows = []
    for condition, label in zip(CONDITIONS, ['A 固定输出', 'C 滚动修正', 'F 完整片段']):
        row = [label]
        for dataset, _ in DATASETS:
            values = next(r for r in summary['datasets'][dataset]['summary'] if r['condition'] == condition)
            row.extend(f'{values[m] * 100:.2f}' for m in ['sentence_bleu1_mean', 'wer', 'aligned_word_accuracy'])
            strategy = WAIT_STRATEGIES[CONDITIONS.index(condition)]
            timing = waits[(waits.dataset == dataset) & (waits.population == 'all_events') & (waits.strategy == strategy)].iloc[0]
            row.append(f'{timing.mean_input_wait_s:.2f}')
        rows.append(row)
    original_out = renderer.OUT
    try:
        renderer.OUT = OUT
        renderer.table_figure('sentence_metrics_acf', headers, rows,
                              [.14, .11, .09, .13, .10, .11, .09, .13, .10],
                              [(1, 5, DATASETS[0][1], DATA_COLORS[DATASETS[0][0]]),
                               (5, 9, DATASETS[1][1], DATA_COLORS[DATASETS[1][0]])])
    finally:
        renderer.OUT = original_out
    return {'headers': headers, 'rows': rows, 'best_values_not_inferred_as_equivalence': True}


def manuscript_block(summary, waits):
    data = []
    for dataset, _ in DATASETS:
        row = waits[(waits.dataset == dataset) & (waits.population == 'all_events') & (waits.strategy == 'C_K2')].iloc[0]
        data.append(row)
    return r'''% BEGIN completed_full_context_sentence_test_v2
\subsection{完整片段等待与有限回修的权衡}

完整片段策略 F 在全部神经窗可用后进行整段搜索。
两个数据集的平均片段 BLEU-1 均高于 C，
C 减 F 的配对材料块区间也均低于零（图~\ref{fig:sentence_full_context_tradeoff}）。
WER 和对齐词准确率的均值同样有利于 F（表~\ref{tab:sentence_revision_full}），
但训练重复间并非所有指标都具有相同方向。
词级局部检索的优势因此不能直接推广为最终句子输出的优势。

这种序列质量差异对应不同的输入等待需求。
在全部句子词位上，C 相对首次窗口完整可用后的平均额外等待为
$''' + f'{data[0].mean_extra_wait_after_k0_s:.2f}' + r'''~\mathrm{s}$ 和 $''' + f'{data[1].mean_extra_wait_after_k0_s:.2f}' + r'''~\mathrm{s}$，
同时获得前述相对固定输出的改善。
相比 F，C 的平均输入等待分别减少
$''' + f'{data[0].mean_saved_vs_f_s:.2f}' + r'''~\mathrm{s}$ 和 $''' + f'{data[1].mean_saved_vs_f_s:.2f}' + r'''~\mathrm{s}$。
有限回修由此提供了介于首次输出固定与等待完整片段之间的选择：
利用少量后续输入提高序列质量，同时接受相对完整片段策略的得分差距。

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{sentence_full_context_tradeoff_v2_test.pdf}
    \caption{完整测试集上三种策略的序列质量与输入等待。A 为首次输出固定，C 为有限滚动回修，F 为等待完整片段后整段搜索。纵轴的 BLEU-1 先在受试者内平均片段，再对受试者和种子等权平均；横轴的等待按全部词位等权平均，同一材料在三个种子中不重复计数。浅色轨迹连接同种子结果，大空心点表示均值；连接线只用于比较离散策略，不表示连续性能函数。图下 C 减 F 的区间为补充比较的配对材料块 bootstrap $95\%$ 区间，未增加显著性检验。等待包含起点后 $1~\mathrm{s}$ 的首次窗口，不含对齐、预处理、计算或显示耗时；边界处只使用片段内实际存在的后续词。}
    \label{fig:sentence_full_context_tradeoff}
\end{figure}

C/F 同时涉及神经上下文模型和历史冻结约束的变化，
比较反映整体策略的准确率与输入等待权衡。
等待差针对各词位获得最终预测所需的输入，
不表示整段最终文本提前完成。
这些时间统计由事件时间戳推导，不是端到端推断速度的实测结果。
% END completed_full_context_sentence_test_v2
'''


def main():
    summary, waits, sources = verify()
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    with style(), mpl.rc_context(renderer.FONT):
        plot_facts = tradeoff(summary, waits)
        table_facts = metrics_table(summary, waits)
    for ext in ['pdf', 'svg', 'png']:
        shutil.copy2(OUT / f'{PLOT}.{ext}', ROOT / 'paper/figures' / f'{PLOT}.{ext}')
    (ROOT / 'paper/tables/sentence_v2_test_metrics.tex').write_text(
        '% Presentation of verified full_context_sentence_test_v2 formal exports.\n'
        + r'\includegraphics[width=\linewidth]{publication_suite_test_v2_full_context/sentence_metrics_acf.pdf}' + '\n', encoding='utf-8')
    manuscript = ROOT / 'paper/manuscript/results2.tex'
    text = manuscript.read_text(encoding='utf-8-sig')
    text = re.sub(r'\n?% BEGIN completed_full_context_sentence_test_v2\n.*?% END completed_full_context_sentence_test_v2\n?', '', text, flags=re.S)
    manuscript.write_text(text.rstrip() + '\n\n' + manuscript_block(summary, waits), encoding='utf-8')
    assert all(sha(ROOT / rel) == expected for rel, expected in sources.items())
    report = {'status': 'verified_complete', 'manifest_sha256': summary['manifest_sha256'],
              'sources': sources, 'original_AC_unchanged': True, 'new_significance_tests': False,
              'plotted_facts': plot_facts, 'table': table_facts, 'text_outside_canvas': 0,
              'manuscript_layout_review_performed': False}
    (AUDIT / 'assets.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (OUT / 'README.md').write_text('''# 完整片段句子比较与有限回修的权衡

运行 `python scripts/paper/add_full_context_comparison_test_v2.py`。
只读取并核验 `reports/exports/full_context_sentence_test_v2_*` 的正式完成导出。
两个数据集分别使用既有蓝色与紫色；三种子轨迹作浅色背景，均值作前景。
图下区间原样读取已冻结的 C/F 材料块分析，不重跑统计或增加显著性检验。
句子结果表包含 A/C/F 与全部词位的平均标称输入等待。
等待不包含预处理或计算耗时，也不等同于末词之后的整句计算延迟。
旧 A/C 图与原出版风格目录保留；这里只提供新版表和新增对照图。
''', encoding='utf-8')
    print(json.dumps({'status': 'verified_complete', 'figure': PLOT, 'table': 'sentence_metrics_acf',
                      'original_AC_unchanged': True, 'plot_facts': plot_facts}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
