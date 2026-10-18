"""Produce only the figures/tables requested in the handwritten paper review.

Result assets consume the already audited test exports; protocol tables consume
the frozen settings. No inference, training, calibration or statistics are run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import yaml

from scripts.paper.publication_suite_style import (
    DATA_COLORS, INK, MUTED, ORANGE, RULE, TEAL, cjk, figure, style,
)
from scripts.paper.apply_publication_style import (
    DATASETS, OUT, save, sentence_figure, table_figure,
)
from scripts.paper.update_test_only_word_results import verify as verify_word
from scripts.paper.update_heldout_sentence_results import verify as verify_sentence

PROTOCOL = ROOT / 'paper/figures/protocol_tables_v2'
AUDIT = ROOT / 'data/results/audits/handwritten_revision_20261004'
TABLE = ROOT / 'paper/tables'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def wrapper(name, graphic, folder='publication_suite_test_v2'):
    path = TABLE / f'{name}.tex'
    path.write_text('% Generated from verified exports/frozen settings for the handwritten review.\n'
                    + r'\includegraphics[width=\linewidth]{' + folder + '/' + graphic + '.pdf}\n', encoding='utf-8')
    return path


def test_tables(context, summary):
    paths, facts = [], {}
    rows = []
    for dataset, label in DATASETS:
        data = context[(context.dataset == dataset) & (context.population == 'all_targets')].set_index('condition')
        rows.append([label, *[f"{data.loc[c, 'macro_recall_at_10_mean'] * 100:.2f} ± {data.loc[c, 'macro_recall_at_10_sd'] * 100:.2f}"
                             for c in ('local_r0', 'local_r2', 'full_context')]])
    paths += table_figure('11_context_summary', ['数据集', '半径0', '半径2', '完整片段'], rows,
                         [.28, .24, .24, .24])
    paths.append(wrapper('context_summary_v2_test', '11_context_summary'))
    facts['context_summary'] = rows
    rows = []
    for dataset, label in DATASETS:
        r = summary['datasets'][dataset]['revisions']
        assert r['paired_repaired'] - r['paired_harmed'] == r['net_correct_words']
        rows.append([label, str(r['words']), str(r['paired_repaired']),
                     str(r['paired_harmed']), str(r['net_correct_words'])])
    paths += table_figure('12_sentence_final_changes',
                         ['数据集', '词位决策数', '改对数', '误改数', '净增正确词数'], rows,
                         [.26, .20, .16, .16, .22])
    paths.append(wrapper('sentence_v2_test_final_changes', '12_sentence_final_changes'))
    facts['sentence_final_changes'] = rows
    return paths, facts


def protocol_tables(manifest, config, summary):
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    paths, facts = [], {}
    rows = [
        ['词模型', '目标词神经窗', '单词映射', '不含上下文网络'],
        ['局部窗口', '$[-k,k]$', '各词每轮一次', '各半径分别训练'],
        ['完整上下文', '片段内全部词', '各词每轮一次', '单独训练'],
        ['独立可见状态', '$[-2,r]$', '各词每轮一次', '各状态分别训练'],
        ['共享滚动模型', '$[-2,r]$', '遍历到达时刻，监督 K0/K1/K2', '三个状态共享参数'],
    ]
    paths += table_figure('model_settings', ['模型', '输入范围', '训练目标', '参数关系'], rows,
                         [.18, .21, .37, .24], directory=PROTOCOL)
    paths.append(wrapper('model_settings_v2', 'model_settings', PROTOCOL.name))
    facts['model_settings'] = rows

    def counts(key):
        return [str(summary['datasets'][d]['counts'][key]) for d, _ in DATASETS]
    rows = [
        ['训练词库', *[str(len(manifest['datasets'][d]['vocabulary'])) for d, _ in DATASETS]],
        ['验证选模候选：词模型', *[str(config['adapters'][d]['expected_legacy_word_candidates']) for d, _ in DATASETS]],
        ['验证选模候选：上下文模型', *[str(config['adapters'][d]['expected_legacy_common_candidates']) for d, _ in DATASETS]],
        ['测试词检索候选', *[str(len(manifest['datasets'][d]['vocabulary50'])) for d, _ in DATASETS]],
        ['局部范围测试查询', *counts('strict50_all')],
        ['可见状态共同查询', *counts('strict50_common')],
        ['序列测试片段', *counts('groups')],
        ['序列参考词位', *counts('events')],
        ['序列材料块', *counts('blocks')],
    ]
    for dataset, _ in DATASETS:
        assert len(manifest['datasets'][dataset]['vocabulary']) == config['adapters'][dataset]['expected_train_vocabulary']
    paths += table_figure('test_population', ['项目', 'Pallier2025', 'ChineseEEG-2'], rows,
                         [.46, .27, .27], group_starts=[4], directory=PROTOCOL)
    paths.append(wrapper('test_population_v2', 'test_population', PROTOCOL.name))
    facts['test_population'] = rows

    d, lm = manifest['decoder'], config['language_model']
    assert all(manifest['datasets'][dataset]['selected_weight'] == .1 for dataset, _ in DATASETS)
    for key, source in [('candidates', 'candidate_count'), ('beam_width', 'beam_width'),
                        ('neural_scale', 'neural_scale'), ('proposal_count', 'proposal_count'),
                        ('proposal_max_new_tokens', 'proposal_max_new_tokens')]:
        assert d[key] == lm[source]
    rows = [
        ['语言模型', lm['model_name'].split('/')[-1] + ' 基础版'],
        ['神经检索候选／词位', str(d['candidates'])],
        ['神经相似度缩放系数', f"{d['neural_scale']:g}"],
        ['束搜索保留序列数', str(d['beam_width'])],
        ['每个前缀的候选续写数', str(d['proposal_count'])],
        ['每条续写的最大新 token 数', str(d['proposal_max_new_tokens'])],
        ['回修的近期词位数', str(d['revision_words'])],
        ['语言模型上下文上限', str(lm['context_tokens']) + ' token'],
        ['验证语言权重候选', ', '.join(f'{v:g}' for v in lm['language_weight_grid'])],
        ['测试固定语言权重', f"{manifest['datasets'][DATASETS[0][0]]['selected_weight']:g}"],
    ]
    paths += table_figure('sentence_decoder_settings', ['设置', '两个数据集共用的取值'], rows,
                         [.49, .51], directory=PROTOCOL)
    paths.append(wrapper('sentence_decoder_settings_v2', 'sentence_decoder_settings', PROTOCOL.name))
    facts['sentence_decoder_settings'] = rows
    return paths, facts


def extract_examples(summary):
    cases = []
    for dataset, label in DATASETS:
        for selection, change in [('most_C_repairs', '改对'), ('most_C_harms', '误改')]:
            example = next(e for e in summary['datasets'][dataset]['examples'] if e['selection'] == selection)
            assert example['seed'] == 0
            trace = {step['arrival']: step['C'] for step in example['trace']}
            final = example['final']['C']
            matches = [i for i, ref in enumerate(example['reference'][:-1])
                       if (trace[i][i] != ref and final[i] == ref if change == '改对'
                           else trace[i][i] == ref and final[i] != ref)]
            assert matches
            i = matches[0]
            assert all(i + step in trace for step in range(3))
            cases.append({'dataset': dataset, 'label': label, 'subject': example['subject'],
                          'seed': example['seed'], 'selection': selection, 'group': example['group'],
                          'position': i + 1, 'change': change, 'reference': example['reference'][i],
                          'tokens': [trace[i + step][i] for step in range(3)]})
    old = read(ROOT / 'data/results/audits/publication_style_test_v2/assets.json')['plotted_facts_percent']['tables']['09_sentence_examples']
    rows = [[c['label'], c['subject'], c['change'], str(c['position']), c['reference'], *c['tokens']] for c in cases]
    assert rows == old, 'Review must keep the original test examples and selection rule.'
    return cases


def example_figure(cases):
    fig = figure(112)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 183), ylim=(112, 0))
    ax.axis('off')

    def label(x, y, value, size=6.5, color=INK, bold=False, ha='left'):
        kwargs = {'fontproperties': cjk(bold)} if any(ord(ch) > 127 for ch in value) else {'weight': 'bold' if bold else 'normal'}
        ax.text(x, y, value, fontsize=size, color=color, ha=ha, va='center', **kwargs)

    for n, case in enumerate(cases):
        col, row = n % 2, n // 2
        left, top = 6 + col * 90, 7 + row * 47
        color = DATA_COLORS[case['dataset']]
        label(left, top, chr(97 + n), 8, bold=True)
        label(left + 5, top, case['label'], 7.2, color=color, bold=True)
        label(left + 61, top, case['change'], 7, color=TEAL if case['change'] == '改对' else ORANGE, bold=True)
        label(left, top + 7, f"{case['subject']} · target position {case['position']}", 6, color=MUTED)
        label(left, top + 14, 'Reference:', 6.2, color=MUTED)
        label(left + 18, top + 14, case['reference'], 7, bold=True)
        for step, token in enumerate(case['tokens']):
            x, y = left + step * 27.8, top + 22
            correct = token == case['reference']
            ink, face = (TEAL, '#EDF5F3') if correct else (ORANGE, '#F8EFEB')
            ax.add_patch(FancyBboxPatch((x, y), 21.5, 9,
                boxstyle='round,pad=0,rounding_size=1.4', facecolor=face, edgecolor=ink, lw=.6))
            label(x + 10.75, y + 4.5, token, 7.8, ink, ha='center')
            label(x + 10.75, y + 13.5, ['$t=i$', '$t=i+1$', '$t=i+2$'][step], 6.2, MUTED, ha='center')
            if step < 2:
                ax.add_patch(FancyArrowPatch((x + 22.7, y + 4.5), (x + 26.3, y + 4.5),
                    arrowstyle='-|>', mutation_scale=5, color=MUTED, lw=.7))
    ax.plot([6, 177], [100, 100], color=RULE, lw=.6)
    label(6, 105, 'Test C trajectories · fixed target word · initial output, then one and two later word events', 6, MUTED)
    label(6, 109, 'Green: matches reference · orange: mismatch · selected correction and harm cases; not typical sequence quality', 5.8, MUTED)
    return save(fig, '10_sentence_revision_examples')


def main():
    context, _, word_report = verify_word()
    summary, _, sentence_sources = verify_sentence()
    manifest_path = ROOT / 'data/manifests/rolling_sentence/heldout_test_v2.json'
    config_path = ROOT / 'configs/training/rolling_revision_protocol.yaml'
    manifest = read(manifest_path)
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert manifest['manifest_sha256'] == summary['manifest_sha256']
    before = read(ROOT / '.cache/handwritten_revision_20261004/before_sha256.json')
    assert all(sha(ROOT / name) == digest for name, digest in before['science'].items())
    paths, facts = [], {}
    with style():
        new_paths, values = sentence_figure(summary)
        old_values = read(ROOT / 'data/results/audits/publication_style_test_v2/assets.json')['plotted_facts_percent']['sentence']
        assert values == old_values, 'Only the sentence colour contract changes.'
        paths += new_paths
        facts['sentence_figure'] = values
        cases = extract_examples(summary)
        paths += example_figure(cases)
        for kind, build, inputs in [
            ('test_tables', test_tables, (context, summary)),
            ('protocol_tables', protocol_tables, (manifest, config, summary)),
        ]:
            new_paths, values = build(*inputs)
            paths += new_paths
            facts[kind] = values
    for source, target in [('05_sentence_revision', 'sentence_revision_v2_test'),
                           ('10_sentence_revision_examples', 'sentence_revision_examples_v2_test')]:
        for ext in ('pdf', 'svg', 'png'):
            destination = ROOT / 'paper/figures' / f'{target}.{ext}'
            shutil.copy2(OUT / f'{source}.{ext}', destination)
            paths.append(destination)
    AUDIT.mkdir(parents=True, exist_ok=True)
    source_hashes = {**word_report['exports'], **sentence_sources,
                     manifest_path.relative_to(ROOT).as_posix(): sha(manifest_path),
                     config_path.relative_to(ROOT).as_posix(): sha(config_path)}
    report = {'status': 'verified', 'result_split': 'test',
              'scientific_inputs_changed': False, 'source_hashes': source_hashes,
              'assets': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
              'facts': facts, 'examples': cases,
              'sentence_dataset_colors': {d: DATA_COLORS[d] for d, _ in DATASETS},
              'examples_match_previous_table': True,
              'inference_or_training_performed': False}
    (AUDIT / 'assets.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Verified test outputs and unchanged scientific inputs; updated sentence figure, process examples and five tables.')


if __name__ == '__main__':
    main()
