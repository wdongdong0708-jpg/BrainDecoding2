"""Regenerate verified word assets without overwriting author-owned prose."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from braindecoding import word_paired_statistics_test_v2 as stats
from scripts.paper.publication_suite_style import DATA_COLORS, GRAY, INK, MUTED, RULE, clean, figure, style
from scripts.paper import refine_publication_style as renderer
from scripts.paper.add_full_context_comparison import verify as verify_sentence

EXPORT = ROOT / 'data/results/exports'
OUT = ROOT / 'paper/figures/publication_suite_test_v2_word_statistics'
TABLES = ROOT / 'paper/tables'
MANUSCRIPT = ROOT / 'paper/manuscript'
AUDIT = ROOT / 'data/results/audits/word_statistics_appendix_20261005'
BACKUP = ROOT / '.cache/word_statistics_appendix_20261005/before/paper/manuscript'
DATASETS = renderer.DATASETS
METRIC_NAMES = ['Macro Top-10', 'Macro Top-1', 'Micro Top-10', 'Micro Top-1']
FACTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_statistics():
    manifest = stats.read(stats.MANIFEST)
    stats.verify_identity(manifest)
    stats.check_sources(manifest)
    report = stats.read(EXPORT / 'word_paired_statistics_test_v2_summary.json')
    assert report['manifest_sha256'] == manifest['manifest_sha256'] and report['status'] == 'verified_complete'
    for path, expected in report['exports'].items():
        assert sha(ROOT / path) == expected
    result = pd.read_csv(EXPORT / 'word_paired_statistics_test_v2_summary.csv')
    core = result[result.metric == 'macro_recall_at_10']
    assert len(core) == 4 and (core.ci_lower > 0).all()
    assert (core.leave_one_block_effect_min > 0).all() and (core.complete_type_ci_lower > 0).all()
    boots = pd.read_csv(EXPORT / 'word_paired_statistics_test_v2_bootstrap.csv')
    omissions = pd.read_csv(EXPORT / 'word_paired_statistics_test_v2_leave_one_block_out.csv')
    for _, row in result.iterrows():
        part = boots[(boots.dataset == row.dataset) & (boots.contrast == row.contrast)]
        assert len(part) == 10000 and part.replicate.nunique() == 10000
        np.testing.assert_allclose(part[row.metric].quantile([.025, .975]), [row.ci_lower, row.ci_upper], atol=1e-12, rtol=0)
        complete = part[part.represented_types == row.target_count]
        assert len(complete) == row.complete_type_bootstrap_samples
        np.testing.assert_allclose(complete[row.metric].quantile([.025, .975]), [row.complete_type_ci_lower, row.complete_type_ci_upper], atol=1e-12, rtol=0)
        leave = omissions[(omissions.dataset == row.dataset) & (omissions.contrast == row.contrast) & (omissions.metric == row.metric)]
        assert len(leave) == row.blocks
        np.testing.assert_allclose([leave.effect.min(), leave.effect.max()], [row.leave_one_block_effect_min, row.leave_one_block_effect_max], atol=1e-12, rtol=0)
    # Independent oracle: explicitly repeat canonical query rows for the first
    # two block resamples and use the original retrieval summarizer directly.
    parent, context = stats.read(stats.PARENT), stats.read(stats.CONTEXT)
    byseed = dict(rolling=stats.csv_rows(EXPORT / 'unified_heldout_test_v2_word_by_seed.csv'),
                  context=stats.csv_rows(EXPORT / 'unified_context_heldout_test_v2_context_by_seed.csv'))
    index = 0
    oracle_checks = 0
    for dataset, info in parent['datasets'].items():
        meta = stats.metadata(info)
        for contrast in manifest['contrasts']:
            weights = stats.bootstrap_weights(info, manifest, index)
            block_index = {b['block']: i for i, b in enumerate(info['blocks'])}
            selected = boots[(boots.dataset == dataset) & (boots.contrast == contrast['id'])].set_index('replicate')
            scores = np.zeros((2, 3, 2, 4))
            for c, condition in enumerate([contrast['left'], contrast['right']]):
                for seed in (0, 1, 2):
                    ids, words, ranks, _ = stats.ranks_for(dataset, seed, condition, contrast['population'], info, parent, context, meta, byseed)
                    query_blocks = np.asarray([block_index[meta[e]['block']] for e in ids])
                    for rep in (0, 1):
                        repeated = np.repeat(np.arange(len(ids)), weights[rep, query_blocks])
                        score = stats.summarize_retrieval(ranks[repeated], words[repeated], 50, top_ks=(1, 10))
                        scores[c, seed, rep] = [score[k] for k in stats.METRICS]
                        oracle_checks += 1
            np.testing.assert_allclose((scores[1]-scores[0]).mean(axis=0), selected.loc[[0, 1], list(stats.METRICS)], atol=1e-12, rtol=0)
            index += 1
    FACTS['statistics'] = dict(comparisons=4, metrics=4, repetitions=10000, explicit_query_resample_oracle_checks=oracle_checks,
                               manifest_sha256=manifest['manifest_sha256'], p_tests=False,
                               original_sentence_AC_tests_unchanged=True)
    return result


def table(name, headers, rows, widths, group_starts=(), group_headers=()):
    renderer.table_figure(name, headers, rows, widths, group_headers=group_headers, group_starts=group_starts)
    (TABLES / f'{name}.tex').write_text(
        '% Presentation only: verified reports/exports; generated by word_statistics_and_appendix_test_v2.py.\n'
        f'\\includegraphics[width=\\linewidth]{{publication_suite_test_v2_word_statistics/{name}.pdf}}\n', encoding='utf-8')
    FACTS[name] = dict(rows=len(rows), headers=headers)


def word_figure(context, paired):
    fig = figure(142)
    radii = [0, 1, 2, 4, 8]
    names = [f'local_r{x}' for x in radii]
    for row in (0, 1):
        top = .965 - row * .458
        bottom = .595 - row * .458
        for col, (dataset, label) in enumerate(DATASETS):
            color = DATA_COLORS[dataset]
            x = .040 + col * .491
            renderer.header(fig, x, top, chr(97+row*2+col), label, dataset)
            ax = fig.add_axes([x+.070, bottom, .348, .283])
            clean(ax)
            ax.grid(axis='y', color='#E5EAED', lw=.45, zorder=0)
            ax.set_axisbelow(True)
            if row == 0:
                raw = context[(context.dataset == dataset) & (context.population == 'all_targets')]
                values = raw.pivot(index='seed', columns='condition', values='macro_recall_at_10').loc[[0, 1, 2], names].to_numpy()*100
                full = raw[raw.condition == 'full_context'].macro_recall_at_10.to_numpy()*100
                ax.axhspan(full.mean()-full.std(ddof=1), full.mean()+full.std(ddof=1), color=GRAY, alpha=.14, lw=0)
                ax.axhline(full.mean(), color=GRAY, ls=(0, (4, 2)), lw=.9)
                renderer.trajectories(ax, radii, values, color)
                best_index = int(np.argmax(values.mean(axis=0)))
                assert radii[best_index] == 2
                best_mean = values[:, best_index].mean()
                gain = best_mean - full.mean()
                ax.plot(2, best_mean, marker='o', ms=6.5, mfc='white',
                        mec=color, mew=1.8, ls='none', zorder=6)
                ax.annotate('Best k = 2', xy=(2, best_mean),
                            xytext=(3.0, min(78.0, best_mean + 7.5)),
                            color=color, fontsize=7, weight='bold',
                            arrowprops={'arrowstyle': '-', 'color': color, 'lw': .7},
                            ha='left', va='center')
                ax.text(.965, .105, f'{gain:+.2f} pp vs full context',
                        transform=ax.transAxes, fontsize=7, color=color,
                        weight='bold', ha='right', va='bottom')
                ax.set(xlim=(-.35, 8.4), ylim=(15, 82), xticks=radii, yticks=[20, 40, 60, 80],
                       xlabel='Symmetric context radius (words)',
                       ylabel='Macro Top-10 (%)' if col == 0 else '')
                title = 'Local range - all 50-word queries'
                ref = 'Full segment mean ± SD'
            else:
                values = renderer.delayed_values(paired, dataset, 'rolling')
                independent = renderer.delayed_values(paired, dataset, 'independent')[:, 2].mean()
                ax.axhline(independent, color=GRAY, ls=(0, (4, 2)), lw=.9)
                renderer.trajectories(ax, [0, 1, 2], values, color)
                ax.set(xlim=(-.16, 2.16), ylim=(25, 82), xticks=[0, 1, 2], yticks=[30, 40, 50, 60, 70, 80],
                       xticklabels=['Initial\nK0', '+1 word\nK1', '+2 words\nK2'],
                        xlabel='Rolling update stage', ylabel='Macro Top-10 (%)' if col == 0 else '')
                title = 'Shared rolling model - common queries'
                ref = 'Independent K2 mean'
            fig.text(x+.029, top-.038, title, fontsize=6.8, color=MUTED, va='top')
            if col == 0 and row == 1:
                handles = [renderer.seed_key(), renderer.mean_key(),
                           Line2D([], [], color=GRAY, lw=.9, ls=(0, (4, 2)), label='Dashed: reference model')]
                fig.legend(handles=handles, loc='center', bbox_to_anchor=(.518, bottom-.106),
                           ncol=3, frameon=False, handlelength=2, columnspacing=2.2)
    renderer.save(fig, 'word_core_evidence_v2_test')
    FACTS['word_core_evidence_v2_test'] = dict(panels=4, top_population='all_targets', bottom_population='common_targets', error_bars='sample_SD_across_three_seeds_not_CI')


def render_tables(result, context, rolling, sentence, waits):
    main = result[result.metric == 'macro_recall_at_10'].set_index(['dataset', 'contrast'])
    rows = []
    for contrast, name in [('local_r2_minus_full', '局部半径2 − 完整上下文'), ('rolling_k2_minus_k0', '共享 K2 − K0')]:
        row = [name]
        for dataset, _ in DATASETS:
            r = main.loc[(dataset, contrast)]
            row.append(f'{r.effect*100:+.2f} [{r.ci_lower*100:.2f}, {r.ci_upper*100:.2f}]')
        rows.append(row)
    table('word_core_paired_effects_v2_test', ['词级比较', *[label for _, label in DATASETS]], rows, [.30, .35, .35])
    condition_names = {'full_context':'完整上下文', **{f'local_r{k}':f'局部半径{k}' for k in [0, 1, 2, 4, 8]},
                       'independent_k0':'独立 K0', 'independent_k1':'独立 K1', 'rolling_k0':'共享 K0',
                       'rolling_k1':'共享 K1', 'rolling_k2':'共享 K2'}
    order = list(condition_names)
    full = context.copy()
    roll = rolling.copy()
    roll['condition'] = roll.horizon.map(lambda k: f'rolling_k{k}')
    full = pd.concat([full, roll], ignore_index=True)
    for dataset, label in DATASETS:
        rows = []
        for condition in order:
            for population, plabel in [('all_targets','全部'), ('common_targets','共同')]:
                part = full[(full.dataset == dataset) & (full.condition == condition) & (full.population == population)]
                assert set(part.seed) == {0, 1, 2}
                values = [f'{part[k].mean()*100:.2f} ± {part[k].std(ddof=1)*100:.2f}' for k in stats.METRICS]
                rows.append([condition_names[condition], plabel, *values])
        table(f'word_complete_metrics_{dataset}_v2_test', ['输入/模型','查询',*METRIC_NAMES], rows,
              [.18, .10, .18, .18, .18, .18], group_starts=list(range(2, len(rows), 2)))
    # All metrics are shown, including intervals crossing zero and a negative
    # Chinese micro Top-1 effect for the local/full contrast.
    rows = []
    for dataset, label in DATASETS:
        for contrast in ['local_r2_minus_full', 'rolling_k2_minus_k0']:
            for metric, ml in zip(stats.METRICS, METRIC_NAMES):
                r = result[(result.dataset == dataset) & (result.contrast == contrast) & (result.metric == metric)].iloc[0]
                rows.append([label if contrast == 'local_r2_minus_full' and metric == stats.METRICS[0] else '',
                             '局部 − 完整上下文' if contrast == 'local_r2_minus_full' else 'K2 − K0', ml,
                             f'{r.effect*100:+.2f}', f'[{r.ci_lower*100:.2f}, {r.ci_upper*100:.2f}]'])
    table('word_paired_all_metrics_v2_test', ['数据集','配对比较','检索指标','Δ (pp)','95% 区间 (pp)'], rows,
          [.20,.18,.20,.13,.29], group_starts=[4,8,12])
    rows = []
    for dataset, label in DATASETS:
        for contrast in ['local_r2_minus_full', 'rolling_k2_minus_k0']:
            r = main.loc[(dataset, contrast)]
            rows.append([label, '局部 −\n完整上下文' if contrast == 'local_r2_minus_full' else 'K2 − K0',
                         f'{int(r.target_count)} / {r.bootstrap_type_count_mean:.2f}\n最少 {int(r.bootstrap_type_count_min)}',
                         f'{int(r.complete_type_bootstrap_samples)} / 10000',
                         f'[{r.complete_type_ci_lower*100:.2f}, {r.complete_type_ci_upper*100:.2f}]',
                         f'[{r.leave_one_block_effect_min*100:.2f}, {r.leave_one_block_effect_max*100:.2f}]'])
    table('word_paired_sensitivity_v2_test', ['数据集','配对比较','原类型 / 平均\n重采样类型','完整类型\n重采样次数','完整类型95%\n区间 (pp)','逐块删除\n效应范围 (pp)'], rows,
          [.20,.15,.16,.16,.165,.165], group_starts=[2])
    rows = []
    for dataset, label in DATASETS:
        for seed in (0, 1, 2):
            for condition in ('A', 'C', 'F'):
                r = next(r for r in sentence['datasets'][dataset]['by_seed'] if r['seed'] == seed and r['condition'] == condition)
                rows.append([label if seed == 0 and condition == 'A' else '', seed, condition,
                             *[f'{r[k]*100:.2f}' for k in ['sentence_bleu1_mean','wer','aligned_word_accuracy']]])
    table('sentence_by_seed_acf_v2_test', ['数据集','种子','策略','BLEU-1 ↑','WER ↓','对齐词准确率 ↑'], rows,
          [.23,.09,.09,.17,.17,.25], group_starts=[3,6,9,12,15])
    original = stats.read(EXPORT / 'unified_heldout_test_v2_sentence_summary.json')
    rows = []
    for dataset, label in DATASETS:
        for contrast in ('C − A', 'C − F'):
            r = original['datasets'][dataset]['primary_statistics'] if contrast == 'C − A' else sentence['datasets'][dataset]['additional_C_minus_F']
            rows.append([label, contrast, f'{r["effect"]*100:+.2f}',
                         f'[{r["ci_lower"]*100:.2f}, {r["ci_upper"]*100:.2f}]',
                         f'{r["p_holm"]:.4f}' if contrast == 'C − A' else '未检验',
                         '预设主比较' if contrast == 'C − A' else '补充比较'])
    table('sentence_paired_inference_v2_test', ['数据集','比较','Δ BLEU-1 (pp)','95% 区间 (pp)','Holm p','分析性质'], rows,
          [.22,.10,.18,.22,.12,.16], group_starts=[2])
    rows = []
    for dataset, label in DATASETS:
        for population, pl in [('all_events','全部词位'), ('common50','共同50词查询')]:
            for strategy, sl in [('K0','A / K0'),('C_K2','C / K2'),('F','F')]:
                r = waits[(waits.dataset == dataset) & (waits.population == population) & (waits.strategy == strategy)].iloc[0]
                rows.append([label if population == 'all_events' and strategy == 'K0' else '', pl, sl,
                             int(r.queries), f'{r.mean_input_wait_s:.2f}',f'{r.median_input_wait_s:.2f}',
                             f'{r.p90_input_wait_s:.2f}',f'{r.mean_saved_vs_f_s:.2f}'])
    table('input_wait_complete_v2_test', ['数据集','评价范围','策略','词位数','均值 (s)','中位 (s)','P90 (s)','较 F 减少 (s)'], rows,
          [.20,.18,.10,.10,.10,.10,.10,.12], group_starts=[3,6,9])


def float_blocks(source):
    return {re.search(r'\\label\{([^}]+)\}', block).group(1): block
            for block in re.findall(r'\\begin\{(?:figure|table)\}.*?\\end\{(?:figure|table)\}', source, re.S)}


def new_table(name, label, caption):
    return '\n'.join([r'\begin{table}[htbp]',r'    \centering', f'    \\input{{../tables/{name}.tex}}',
                      f'    \\caption{{{caption}}}', f'    \\label{{{label}}}',r'\end{table}'])


def write_manuscript(result, sentence, waits):
    old_results = (BACKUP / 'results2.tex').read_text(encoding='utf-8')
    old_experiments = (BACKUP / 'experiments.tex').read_text(encoding='utf-8')
    rf, ef = float_blocks(old_results), float_blocks(old_experiments)
    main = result[result.metric == 'macro_recall_at_10'].set_index(['dataset','contrast'])
    template = r'''% !TeX root = main.tex

\section{实验结果与分析}

本节从词检索、文本输出和输入等待三个层面评价有限回修。
词级主指标为 Macro Top-$10$；预设句子级主比较为 C 相对 A 的平均片段 BLEU-1 增益。
详细设置、完整指标和统计敏感性分析见附录。

\subsection{有限局部上下文支持有效词检索}

两个数据集的局部范围曲线呈现一致趋势：纳入邻近词窗后检索得分提高，
继续扩大范围未带来持续增益，最高均值均出现在开发阶段固定的半径2
（图~\ref{fig:word_core_evidence}a,b）。
在冻结50候选的全部合格测试查询上，
半径2相对完整片段的 Macro Top-$10$ 分别提高 @@LOCAL_F@@ 和 @@LOCAL_Z@@ 个百分点。
配对材料块区间均高于零（表~\ref{tab:word_core_paired}），
逐一删除材料块后，四项核心比较的效应仍为正
（附录表~\ref{tab:word_paired_sensitivity}）。
该结果支持在当前词检索任务中限定上下文范围，但并非所有检索指标均有相同优势：
中文 Micro Top-$1$ 的局部窗口均值低于完整片段，
部分 Top-$1$ 差值区间跨越零（附录表~\ref{tab:word_paired_all_metrics}）。

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{publication_suite_test_v2_word_statistics/word_core_evidence_v2_test.pdf}
    \caption{词级核心证据。a,b：局部上下文范围，在全部合格50词查询上比较；灰虚线和浅灰带为完整片段模型的均值及一个样本标准差。c,d：同一共享模型的 K0、K1、K2，在三个状态均可用的共同查询上比较；灰虚线为独立 K2 的均值。浅色轨迹为三个种子，粗线和误差棒为均值及样本标准差，误差棒不表示材料块置信区间。半径0仍含上下文 Transformer，片段边界处截断窗口。}
    \label{fig:word_core_evidence}
\end{figure}

\begin{table}[htbp]
    \centering
    \input{../tables/word_core_paired_effects_v2_test.tex}
    \caption{词级核心比较的 Macro Top-$10$ 增益与逐项 $95\%$ 配对材料块区间，单位为百分点。局部半径2减完整片段使用全部合格50词查询；共享滚动 K2 减 K0 使用共同查询，两个数据集的查询数分别为 $8930/7440$ 与 $3444/2872$。按记录或章节分层重采样 $10000$ 次，保留块内全部受试者及三个种子。该统计在测试结果之后固定，属于补充分析；不新增显著性检验。类型覆盖与敏感性结果见附录表~\ref{tab:word_paired_sensitivity}。}
    \label{tab:word_core_paired}
\end{table}

\subsection{共享模型随有限延迟改善同一目标}

在相同测试词位上，共享模型的候选排序随回修改善
（图~\ref{fig:word_core_evidence}c,d）。
Pallier2025 和 ChineseEEG-2 的 K2 相对 K0 增益分别为
@@ROLL_F@@ 和 @@ROLL_Z@@ 个百分点，材料块区间均高于零。
首次回修带来主要改善，第二次回修继续提高平均得分，
三个训练种子的 K2 均高于自身 K0。
一组共享参数由此能够在后续词窗可用时重新评价同一目标，
无需为每个延迟状态分别部署一个模型。

独立延迟模型也呈现相同的改善方向
（附录图~\ref{fig:future_increment_curve}）。
共享模型 K2 的均值接近独立 K2，但未进行等效性检验
（附录表~\ref{tab:rolling_revision_results}）。
这些比较不将新增词窗等同于独立的神经信息：各词输入覆盖起点后的 $1~\mathrm{s}$，
相邻词窗可在物理时间上重叠，独立与共享模型之间也存在监督方式的差异。
词级区间以已观察受试者和训练种子为条件，不能解释为跨受试者或任意训练重复的保证。

\subsection{有限回修改善文本并减少输入等待}

在最终文本上，滚动策略 C 相对首次输出固定的 A 改善平均片段 BLEU-1，
两数据集的增益分别为 @@AC_F@@ 和 @@AC_Z@@ 个百分点。
预设材料块检验经 Holm 校正后均有 $p=0.0002$，
WER 和对齐词准确率也呈现一致的改善方向
（表~\ref{tab:sentence_revision_full}）。
完整配对区间、逐种子结果与受试者差异见
附录表~\ref{tab:sentence_paired_inference}、表~\ref{tab:sentence_by_seed_acf}
和图~\ref{fig:sentence_participant_gains}。

完整片段策略 F 等待全部神经窗并重新搜索整段，其平均片段 BLEU-1 高于 C，
C 减 F 的材料块区间均低于零（图~\ref{fig:sentence_full_context_tradeoff}）。
词级局部检索优势因此不能直接推广为最终句子输出优势。
有限回修的价值体现在输入等待与文本质量之间：
两个数据集在首次窗口之后平均额外等待 @@EXTRA_F@@ 和 @@EXTRA_Z@@ 秒，
获得相对 A 的序列改善；相比 F，各词位平均输入等待减少 @@SAVE_F@@ 和 @@SAVE_Z@@ 秒。

@@TRADEOFF_FIG@@

@@SENTENCE_TABLE@@

C/F 比较同时改变神经上下文模型和历史冻结约束，评价的是整体策略。
等待由词时间戳推导，针对各词位取得最终预测所需的输入，
不表示整段最终文本提前完成，也不包含对齐、预处理、计算和显示开销。
完整等待分布及不同查询范围的结果见附录表~\ref{tab:input_wait_complete}。

\subsection{文本收益伴随改写代价}

图~\ref{fig:sentence_examples_full} 展示了实际词位的改对与误改：
后续输入能够更新较早预测，也可能损害已正确的输出。

A/C 的最终输出比较，以及 C 自身的首次到最终比较，
均显示改对多于误改，完整计数见附录表~\ref{tab:sentence_final_changes}
和表~\ref{tab:sentence_repairs_full}。
然而，实际发生的显示改写大多是将一个错词改成另一错词
（图~\ref{fig:sentence_display_edit_composition}）。
因此，序列得分改善不等于每次改写都有效，
近期回修的准确率收益需要与输出稳定性一并评价。

@@EXAMPLES_FIG@@

@@COMPOSITION_FIG@@
'''
    original = stats.read(EXPORT / 'unified_heldout_test_v2_sentence_summary.json')
    replacements = dict(TRADEOFF_FIG=rf['fig:sentence_full_context_tradeoff'],
                        SENTENCE_TABLE=rf['tab:sentence_revision_full'],
                        EXAMPLES_FIG=rf['fig:sentence_examples_full'],
                        COMPOSITION_FIG=rf['fig:sentence_display_edit_composition'])
    for (dataset, _), suffix in zip(DATASETS, ['F','Z']):
        replacements['LOCAL_'+suffix] = f'{main.loc[(dataset,"local_r2_minus_full"),"effect"]*100:.2f}'
        replacements['ROLL_'+suffix] = f'{main.loc[(dataset,"rolling_k2_minus_k0"),"effect"]*100:.2f}'
        replacements['AC_'+suffix] = f'{original["datasets"][dataset]["primary_statistics"]["effect"]*100:.2f}'
        timing = waits[(waits.dataset == dataset) & (waits.population == 'all_events') & (waits.strategy == 'C_K2')].iloc[0]
        replacements['EXTRA_'+suffix] = f'{timing.mean_extra_wait_after_k0_s:.2f}'
        replacements['SAVE_'+suffix] = f'{timing.mean_saved_vs_f_s:.2f}'
    for key, value in replacements.items():
        template = template.replace('@@'+key+'@@', value)
    assert '@@' not in template
    (MANUSCRIPT / 'results2.tex').write_text(template, encoding='utf-8')
    experiments = old_experiments
    for block in ef.values():
        experiments = experiments.replace(block, '')
    for label in ['tab:test_population','tab:model_settings','tab:sentence_decoder_settings']:
        experiments = experiments.replace('表~\\ref{'+label+'}', '附录表~\\ref{'+label+'}')
    experiments = experiments.replace('辅助导出还包括微平均、Top-$1$ 和完整训练词库检索结果；',
                                      '附录表~\\ref{tab:word_complete_pallier} 和表~\\ref{tab:word_complete_chinese} 完整报告50候选下的宏/微平均 Top-$1$、Top-$10$；\n辅助导出还包括完整训练词库检索结果；')
    old_statistics = '词级上下文与延迟结果报告描述性效应，提供机制层面的证据；\n上述显著性检验仅对应预设的句子 A/C 主比较。'
    new_statistics = r'''上述显著性检验仅对应预设的句子 A/C 主比较。
词级核心比较另报告配对材料块效应和区间：
半径2相对完整片段使用全部合格50词查询，K2 相对 K0 使用共同查询。
两项比较均按词事件标识配对，保留块内全部受试者及三个固定种子，
按记录或章节分层进行 $10000$ 次 bootstrap，并重新计算词类型宏平均后再对种子平均。
区间为逐项百分位 $95\%$ 区间，不增加 $p$ 值或全比较族推断。
这项分析在已有测试结果之后固定，属于补充统计，
以已观察受试者、训练模型和记录/章节为条件。
重采样中的真实词类型覆盖、完整类型敏感性和逐块删除结果见附录~\ref{sec:supp_word_statistics}。'''
    assert old_statistics in experiments
    experiments = experiments.replace(old_statistics, new_statistics)
    # Remove the inactive historical control placeholder from the source;
    # no preview/control results are promoted into the formal appendix.
    experiments = experiments.split('% \\subsection{严格审计}')[0].rstrip()+'\n'
    (MANUSCRIPT / 'experiments.tex').write_text(experiments, encoding='utf-8')
    discussion = (BACKUP / 'discussion.tex').read_text(encoding='utf-8')
    discussion = discussion.replace('共享模型随后在同一目标上获得延迟修正收益，',
        '配对材料块区间进一步支持局部半径2相对完整片段及 K2 相对 K0 的词级效应。\n'
        '这些补充区间以已观察受试者、训练模型和记录/章节为条件，\n'
        '不能替代对新受试者或其他材料的泛化验证。\n'
        '共享模型随后在同一目标上获得延迟修正收益，')
    (MANUSCRIPT / 'discussion.tex').write_text(discussion, encoding='utf-8')
    supplement = r'''% !TeX root = main.tex
% Formal appendix: frozen v2 TEST results only. Old vocabulary placeholders are not used.

\section{数据、模型与解码设置}
\label{sec:supp_settings}

附录与正文使用相同的冻结测试协议和检查点。
所有结果来自正式测试导出，不混入早期验证集预览。
表~\ref{tab:test_population} 区分词检索查询与句子词位，
表~\ref{tab:model_settings} 和表~\ref{tab:sentence_decoder_settings}
给出模型输入、监督范围与搜索设置。

@@POPULATION@@

@@MODEL_SETTINGS@@

@@DECODER_SETTINGS@@

\clearpage
\section{完整词检索结果与补充统计}
\label{sec:supp_word_statistics}

表~\ref{tab:word_complete_pallier} 和表~\ref{tab:word_complete_chinese}
报告全部预设局部范围、完整片段、独立延迟和共享滚动状态，
同时保留全部合格查询与共同查询的区别。
候选集合均包含冻结的50词；Macro 只平均当前查询中出现的真实词类型。
独立 K2 与局部半径2共用检查点，不重复计为独立实验。
完整训练词库检索是另一候选口径，保存在原始运行结果中，不与本文50候选主比较混合。

@@WORD_F@@

@@WORD_Z@@

\subsection{材料块配对与不确定性范围}

补充词级分析固定四个比较：每个数据集的局部半径2减完整片段，
以及共享模型 K2 减 K0。参数在重采样前固定，原测试结果已经可见，
因此该分析不视为预注册验证，也不改变原句子 A/C 检验族。
每次在原记录或章节内有放回抽取与原来数量相同的材料块，
同一个被抽中块保留所有受试者、目标词位、配对条件及三个种子。
每种子先在真实词类型内平均命中，再对词类型等权平均，最后对三个种子等权平均。
种子不作为新增独立样本，受试者也不单独拆散同一刺激的材料块。

每项比较进行 $10000$ 次重采样，随机种子为 $20261005$。
候选词始终保持50个；若某次重采样缺少某真实类型，
该次 Macro 对两个条件中完全相同的已出现类型求平均。
所有重采样均进入主区间，不按结果方向或类型缺失筛除。
表~\ref{tab:word_paired_all_metrics} 报告完整四项指标，
包括跨零区间和中文局部模型较低的 Micro Top-$1$ 均值。
这些区间均为逐项区间，未作覆盖整个比较族的校正。

@@PAIRED_ALL@@

表~\ref{tab:word_paired_sensitivity} 报告真实类型覆盖，
并额外列出保留全部原始真实类型的重采样区间。
该额外区间仅为敏感性分析，不替换主区间；
逐块删除范围也仅描述对单一材料块的依赖，不能作为置信区间。
四项主指标的两种敏感性结果均保持正向，
但区间仅刻画给定受试者、已训练种子与记录/章节下的材料变化，
不覆盖新受试者、其他材料领域、重新训练或未见真实类型的不确定性。

@@SENSITIVITY@@

\subsection{独立延迟模型与共享状态}

独立延迟条件固定两个历史词，只改变可用后续词窗数量。
图~\ref{fig:future_increment_curve} 用于描述可用范围的变化；
表~\ref{tab:rolling_revision_results} 对照同种子的独立与共享模型。
独立条件分别训练，其差值不能单独归因于新增词窗。
共享 K2 接近独立 K2 的平均得分不构成等效性结论。

@@FUTURE@@

@@ROLLING_COMPARISON@@

\clearpage
\section{句子策略的完整结果与个体差异}
\label{sec:supp_sentence}

A/C 为预设句子主比较，C/F 为完成 A/C 后固定的补充策略比较。
两项比较沿用相同材料块，但只有 A/C 使用单侧符号翻转检验及两数据集 Holm 校正。
C/F 不新增显著性检验。
表~\ref{tab:sentence_by_seed_acf} 给出全部种子和策略，
表~\ref{tab:sentence_paired_inference} 给出两类配对效应及区间。

@@SENTENCE_SEEDS@@

@@SENTENCE_INFERENCE@@

图~\ref{fig:sentence_revision_full} 保留 A/C 主比较，
图~\ref{fig:sentence_participant_gains} 展示各受试者的收益幅度，
训练种子不增加受试者样本量。

@@SENTENCE_AC@@

@@PARTICIPANTS@@

\clearpage
\section{输入等待的完整分布}
\label{sec:supp_wait}

表~\ref{tab:input_wait_complete} 分开报告全部句子词位和共同50词查询。
等待包含首次词窗的 $1~\mathrm{s}$，由原始事件时间戳计算，
同一词位不因三个模型种子重复计数。
该时间不包括对齐、预处理、计算及显示。
共同查询排除了片段末尾不能获得两次后续回修的目标，
因此不与全部词位的等待分布混合。
平均等待节省描述较早词位的输入需求，不能解释为整个片段提前完成或实测运行速度。

@@WAIT_TABLE@@

\clearpage
\section{最终输出变化与回修过程诊断}
\label{sec:supp_edits}

表~\ref{tab:sentence_final_changes} 比较 A 和 C 的最终输出，
分母覆盖全部测试词位；表~\ref{tab:sentence_repairs_full}
比较 C 自身首次与最终输出，只含具有后续事件的词位。
正文的显示改写组成图则以每次真正发生文本变化的更新为分母，
同一词位可以在两个回修阶段各贡献一个事件。
三种口径分别反映策略收益、目标最终纠正和显示稳定性，不能互换。
汇总计数合并三次训练的决策，不代表相应数量的独立观测。
正文案例按种子0中改对或误改最多片段的首个对应词位选取，
仅说明更新机制，不能估计典型序列质量。

@@FINAL_CHANGES@@

@@SELF_REPAIRS@@
'''
    blocks = dict(POPULATION=ef['tab:test_population'],MODEL_SETTINGS=ef['tab:model_settings'],
                  DECODER_SETTINGS=ef['tab:sentence_decoder_settings'],
                  FUTURE=rf['fig:future_increment_curve'],ROLLING_COMPARISON=rf['tab:rolling_revision_results'],
                  SENTENCE_AC=rf['fig:sentence_revision_full'],PARTICIPANTS=rf['fig:sentence_participant_gains'],
                  FINAL_CHANGES=rf['tab:sentence_final_changes'],SELF_REPAIRS=rf['tab:sentence_repairs_full'])
    blocks['WORD_F'] = new_table('word_complete_metrics_pallier2025_v2_test','tab:word_complete_pallier',
        r'Pallier2025 的完整50候选词检索结果（\%，三个种子均值 $\pm$ 样本标准差）。全部与共同查询数分别为8930和7440，真实词类型为49和48。独立 K2 即局部半径2的共同查询行。')
    blocks['WORD_Z'] = new_table('word_complete_metrics_chineseeeg2_littleprince_v2_test','tab:word_complete_chinese',
        r'ChineseEEG-2 的完整50候选词检索结果（\%，三个种子均值 $\pm$ 样本标准差）。全部与共同查询数分别为3444和2872，两者均含46个真实词类型。独立 K2 即局部半径2的共同查询行。')
    blocks['PAIRED_ALL'] = new_table('word_paired_all_metrics_v2_test','tab:word_paired_all_metrics',
        r'词级补充配对比较的全部指标。差值及逐项 $95\%$ 材料块区间以百分点报告。局部减完整使用全部合格50词查询，K2 减 K0 使用共同查询。')
    blocks['SENSITIVITY'] = new_table('word_paired_sensitivity_v2_test','tab:word_paired_sensitivity',
        r'Macro Top-$10$ 的类型覆盖与稳健性描述。完整类型区间只使用保留原有全部真实类型的重采样，次数按所示子集计算；逐块删除范围是删除任一单个材料块后的效应极值，不是置信区间。主区间使用全部10000次重采样。')
    blocks['SENTENCE_SEEDS'] = new_table('sentence_by_seed_acf_v2_test','tab:sentence_by_seed_acf',
        r'A/C/F 的逐种子句子结果（\%）。每种子先在受试者内汇总，再对受试者等权平均；A 与 C 沿用原始冻结结果。')
    blocks['SENTENCE_INFERENCE'] = new_table('sentence_paired_inference_v2_test','tab:sentence_paired_inference',
        r'句子策略的配对材料块统计。差值及 $95\%$ 区间以百分点报告；A/C 主检验和 Holm 校正值保持不变，C/F 只报告补充区间。Pallier2025 与 ChineseEEG-2 分别含39和21个材料块，保留所有受试者及三个种子。')
    blocks['WAIT_TABLE'] = new_table('input_wait_complete_v2_test','tab:input_wait_complete',
        r'输入等待的均值、中位数及第90百分位。A/K0 为首次词窗可用，C/K2 为最后一次片段内允许回修，F 为末词窗可用。两个查询范围单独汇总，不按种子重复词位。')
    for key, value in blocks.items():
        supplement = supplement.replace('@@'+key+'@@',value)
    assert '@@' not in supplement
    (MANUSCRIPT / 'supplement.tex').write_text(supplement, encoding='utf-8')
    main_source = (BACKUP / 'main.tex').read_text(encoding='utf-8')
    main_source = main_source.replace('% \\input{supplement}', '')
    appendix = r'''
\clearpage
\appendix
\setcounter{figure}{0}
\setcounter{table}{0}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\theHfigure}{supp.\arabic{figure}}
\renewcommand{\theHtable}{supp.\arabic{table}}
\input{supplement}
'''
    main_source = main_source.replace('\\end{document}', appendix+'\n\\end{document}')
    (MANUSCRIPT / 'main.tex').write_text(main_source, encoding='utf-8')


def audit_sources():
    labels, refs = [], []
    included = ['introduction','related_work','methods','experiments','results2','discussion','supplement']
    texts = {name:(MANUSCRIPT / (name+'.tex')).read_text(encoding='utf-8') for name in included}
    texts = {name:re.sub(r'(?m)^\s*%.*$', '', content) for name,content in texts.items()}
    for content in texts.values():
        labels += re.findall(r'\\label\{([^}]+)\}',content)
        refs += re.findall(r'\\ref\{([^}]+)\}',content)
    assert len(labels) == len(set(labels)), 'Duplicate labels'
    assert set(refs) <= set(labels), set(refs)-set(labels)
    main_text = '\n'.join(texts[name] for name in included if name != 'supplement')
    main_counts = {kind:len(re.findall(r'\\begin\{'+kind+r'\}',main_text)) for kind in ('figure','table')}
    supplement_counts = {kind:len(re.findall(r'\\begin\{'+kind+r'\}',texts['supplement'])) for kind in ('figure','table')}
    assert main_counts == dict(figure=5,table=2)
    assert supplement_counts == dict(figure=3,table=13)
    assert 'supp_vocab_decoding' not in texts['supplement']
    assert '附录附录' not in main_text
    before = json.loads((ROOT / '.cache/word_statistics_appendix_20261005/before_sha256.json').read_text())
    allowed = {'paper/manuscript/main.tex','paper/manuscript/introduction.tex',
               'paper/manuscript/results2.tex','paper/manuscript/experiments.tex',
               'paper/manuscript/supplement.tex','paper/manuscript/discussion.tex',
               'output/pdf/manuscript_unified_protocol_draft.pdf',
               'braindecoding/word_paired_statistics_test_v2.py'}
    changed = [p for p,value in before.items() if sha(ROOT / p) != value]
    assert set(changed) <= allowed, changed
    FACTS['manuscript'] = dict(main_counts=main_counts, supplement_counts=supplement_counts,
                               labels=len(labels), references=len(refs), all_references_resolved_in_source=True,
                               approved_method_figure_unchanged=True, existing_result_assets_unchanged=True,
                               old_validation_placeholder_tables_excluded=True)
    FACTS['preservation'] = dict(existing_files_checked=len(before), changed_existing_files=changed,
                                 all_existing_scientific_results_unchanged=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    AUDIT.mkdir(parents=True,exist_ok=True)
    TABLES.mkdir(parents=True,exist_ok=True)
    result = verify_statistics()
    sentence, waits, sources = verify_sentence()
    context = pd.read_csv(EXPORT / 'unified_context_heldout_test_v2_context_by_seed.csv')
    rolling = pd.read_csv(EXPORT / 'unified_heldout_test_v2_word_by_seed.csv')
    paired = pd.read_csv(EXPORT / 'unified_context_heldout_test_v2_independent_vs_rolling_by_seed.csv')
    renderer.OUT = OUT
    with style(), mpl.rc_context(renderer.FONT):
        word_figure(context, paired)
        render_tables(result, context, rolling, sentence, waits)
    # The appendix was organized once. Subsequent asset regeneration must
    # preserve the author's later Introduction and Results revisions.
    audit_sources()
    FACTS['sources'] = sources
    FACTS['text_bounds'] = renderer.BOUNDS
    (AUDIT / 'content_and_preservation.json').write_text(json.dumps(FACTS,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT / 'README.md').write_text(
        '# 词级配对统计与正式附录\n\n'
        '本目录仅包含呈现资产，科学数据来自 reports/exports。\n'
        '生成入口：scripts/paper/word_statistics_and_appendix_test_v2.py。\n'
        '正文保留5图2表；正式附录使用独立的 S 图表编号。\n'
        '原有已接受图表仍保存在仓库，不替换其科学内容。\n',encoding='utf-8')
    print(json.dumps(FACTS['manuscript'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
